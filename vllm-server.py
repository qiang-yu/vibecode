###
# OpenAI-compatible API proxy for vllm with two-phase inference.
# Renders the chat template locally, sends both phases to vllm via
# /v1/completions (raw-prompt), parses tool_calls from the combined
# output, and returns a proper OpenAI chat.completion response.
#
# Usage (Qwen3):
#   python vllm-server.py \
#     --base-model-path /path/to/Qwen3-8B \
#     --model-type Qwen3 \
#     [--vllm-url http://localhost:19001/v1] \
#     [--base-model-id Qwen3Base] [--lora-model-id lora-model] \
#     [--host localhost] [--port 29001]
#
# Usage (Llama3):
#   python vllm-server.py \
#     --base-model-path /path/to/llama3 \
#     --model-type Llama3 \
#     [--llama3-template tool_chat_template_llama3.1_json.jinja]
###

import argparse
import copy
import json
import logging
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration defaults — edit here or override with CLI arguments
# ---------------------------------------------------------------------------

VLLM_BASE_URL             = "http://localhost:19001/v1"
BASE_MODEL_ID             = "Qwen3Base"
LORA_MODEL_ID             = "lora-model"
BASE_MODEL_PATH           = "/home/qiangyu/Models/Qwen/Qwen3-8B"           # required: local path to load tokenizer
MODEL_TYPE                = "Qwen3"     # Qwen3 | Llama3
LLAMA3_TEMPLATE_PATH      = str(Path(__file__).parent / "tool_chat_template_llama3.1_json.jinja")
MAX_TOKENS_SECURITY       = 512         # hard limit for phase 2 / security block
REQUEST_TIMEOUT           = 300         # seconds

LISTEN_HOST               = "localhost"
LISTEN_PORT               = 29001

STRIP_SECURITY_IN_HISTORY = True
ENABLE_THINKING           = True        # Qwen3: pass enable_thinking to apply_chat_template

TOOL_CALL_END             = "</tool_call>"
TOOL_CALL_SECURITY_START  = "<tool_call_security>"
TOOL_CALL_SECURITY_END    = "</tool_call_security>"

# Headers that must not cross HTTP hops; httpx already decompresses content,
# so forwarding content-encoding would cause the client to double-decompress.
_HOP_BY_HOP = frozenset({
    "content-encoding", "content-length", "transfer-encoding",
    "connection", "keep-alive", "te", "trailers", "upgrade",
})

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state — set in main() / lifespan()
# ---------------------------------------------------------------------------

tokenizer: Any = None
_http: Optional[httpx.AsyncClient] = None
_context_window: int = 0          # max_model_len from vllm, set at startup
_llama3_template: Optional[str] = None   # Llama3 Jinja template, cached at startup

# ---------------------------------------------------------------------------
# FastAPI lifespan: one shared connection pool for the whole process
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http, _context_window
    _http = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    log.info("HTTP client created (timeout=%ds)", REQUEST_TIMEOUT)

    # Validate model IDs and read max_model_len from vllm.
    try:
        r = await _http.get(f"{VLLM_BASE_URL.rstrip('/')}/models")
        r.raise_for_status()
        models = r.json().get("data", [])
        model_ids = {m.get("id") for m in models}

        missing = [mid for mid in (BASE_MODEL_ID, LORA_MODEL_ID) if mid not in model_ids]
        if missing:
            raise ValueError(
                f"Model ID(s) not found in vllm: {missing}. "
                f"Available: {sorted(model_ids)}"
            )

        base_info = next(m for m in models if m.get("id") == BASE_MODEL_ID)
        if not base_info.get("max_model_len"):
            raise ValueError(f"max_model_len missing for {BASE_MODEL_ID}")
        _context_window = int(base_info["max_model_len"])
        log.info("Context window: %d tokens (from %s)", _context_window, BASE_MODEL_ID)
    except Exception as exc:
        await _http.aclose()
        log.error("Startup validation failed: %s", exc)
        raise

    # Probe add_special_tokens=False only for Llama3: Qwen3 has no BOS token so
    # the probe cannot detect anything meaningful there (counts match regardless).
    if MODEL_TYPE.lower() == "llama3":
        try:
            probe = "Hello"
            local_count = len(tokenizer.encode(probe, add_special_tokens=False))
            r = await _http.post(
                f"{VLLM_BASE_URL.rstrip('/')}/completions",
                json={"model": BASE_MODEL_ID, "prompt": probe, "max_tokens": 1,
                      "add_special_tokens": False},
            )
            r.raise_for_status()
            vllm_count = r.json().get("usage", {}).get("prompt_tokens", -1)
            if vllm_count == local_count:
                log.info("add_special_tokens=False verified for Llama3 (%d tokens)", local_count)
            else:
                log.warning(
                    "add_special_tokens probe mismatch: local=%d vllm=%d — "
                    "vllm may be ignoring add_special_tokens=False; Llama3 will get duplicate BOS",
                    local_count, vllm_count,
                )
        except httpx.HTTPStatusError as exc:
            if 400 <= exc.response.status_code < 500:
                # vllm rejected the field — it is not supported in this version.
                await _http.aclose()
                log.error(
                    "add_special_tokens=False was rejected by vllm (%d). "
                    "This vllm version does not support the field; Llama3 will get duplicate BOS "
                    "and prompt token counts will be off by 1.",
                    exc.response.status_code,
                )
                raise
            log.warning("add_special_tokens probe failed (non-fatal, HTTP %d): %s",
                        exc.response.status_code, exc)
        except Exception as exc:
            log.warning("add_special_tokens probe failed (non-fatal): %s", exc)

    yield
    await _http.aclose()
    log.info("HTTP client closed")


app = FastAPI(title="vllm-server", docs_url=None, redoc_url=None, lifespan=lifespan)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_SECURITY_RE  = re.compile(r"<tool_call_security>.*?</tool_call_security>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_THINK_RE     = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_LLAMA3_BOT   = "<|python_tag|>"

# ---------------------------------------------------------------------------
# Message preprocessing before chat-template rendering
# ---------------------------------------------------------------------------

def _strip_security(text: str) -> str:
    return _SECURITY_RE.sub("", text).strip()


def _render_tool_calls_as_text(tool_calls: List[Dict]) -> str:
    """Serialize the OpenAI tool_calls array back to the model's native text."""
    mt = MODEL_TYPE.strip().lower()
    parts = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        if mt == "qwen3":
            obj = {"name": name, "arguments": args}
            parts.append(f"<tool_call>\n{json.dumps(obj, ensure_ascii=False)}\n</tool_call>")
        elif mt == "llama3":
            obj = {"name": name, "parameters": args}
            parts.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(parts)


def _prepare_messages_for_model(messages: List[Dict]) -> List[Dict]:
    """
    Preprocess incoming messages before applying the chat template.

    Security block handling in assistant messages:
      - STRIP_SECURITY_IN_HISTORY=True (default): remove the security block from
        content. Empty content after stripping becomes None.
      - STRIP_SECURITY_IN_HISTORY=False: keep the block but reorder it to come
        *after* the <tool_call> text, matching the original generation order.
        The tool_calls field is dropped and inlined into content so the chat
        template does not re-render it in the wrong position.

    Note on <think> in history: _parse_qwen3 already strips <think> blocks before
    returning content to the client, so assistant messages in subsequent turns
    contain no <think>. No special handling is needed here.
    """
    result = []
    for msg in copy.deepcopy(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")

            # Flatten list content to a plain string for security detection.
            # Without this, list content can never match and the elif branch below
            # is unreachable dead code.
            if isinstance(content, list):
                content_str = "\n".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                content_str = content if isinstance(content, str) else ""

            has_security = "<tool_call_security>" in content_str

            if has_security:
                if STRIP_SECURITY_IN_HISTORY:
                    if isinstance(content, str):
                        # Use empty string, not None: Jinja2 renders None as the
                        # literal "None" and string concatenation with None raises TypeError.
                        msg["content"] = _strip_security(content)
                    elif isinstance(content, list):
                        for p in content:
                            if isinstance(p, dict) and p.get("type") == "text":
                                p["text"] = _strip_security(p.get("text", ""))
                elif tool_calls:
                    # Keep security but fix ordering: reconstruct content as
                    # <tool_call>...</tool_call><security_block> and drop tool_calls
                    # so the template does not re-render tool_call in the wrong slot.
                    sec_match = _SECURITY_RE.search(content_str)
                    security_block = sec_match.group(0) if sec_match else ""
                    msg["content"] = _render_tool_calls_as_text(tool_calls) + security_block
                    msg.pop("tool_calls", None)

        result.append(msg)
    return result

# ---------------------------------------------------------------------------
# Chat template rendering
# ---------------------------------------------------------------------------

_ROLE_ALIASES = {"developer": "system"}


def _normalize_messages(messages: List[Dict]) -> List[Dict]:
    out = []
    for msg in messages:
        msg = dict(msg)
        role = msg.get("role", "")
        msg["role"] = _ROLE_ALIASES.get(role, role)
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = "\n".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        out.append(msg)
    return out


def _render_prompt(messages: List[Dict], tools: Optional[List[Dict]]) -> str:
    msgs = _normalize_messages(messages)
    mt = MODEL_TYPE.strip().lower()
    if mt == "qwen3":
        return tokenizer.apply_chat_template(
            msgs,
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=ENABLE_THINKING,
        )
    if mt == "llama3":
        if _llama3_template is None:
            raise ValueError("Llama3 template not loaded; was --model-type Llama3 set at startup?")
        return tokenizer.apply_chat_template(
            msgs,
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=False,
            chat_template=_llama3_template,
        )
    raise ValueError(f"Unknown model_type: {MODEL_TYPE!r}")

# ---------------------------------------------------------------------------
# Tool-call output parsing
# ---------------------------------------------------------------------------

def _parse_qwen3(text: str) -> Tuple[List[Dict], Optional[str]]:
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return [], _THINK_RE.sub("", text).strip() or None

    segments, cursor = [], 0
    for m in matches:
        if m.start() > cursor:
            segments.append(text[cursor:m.start()])
        cursor = m.end()
    if cursor < len(text):
        segments.append(text[cursor:])
    content = _THINK_RE.sub("", "".join(segments)).strip() or None

    tool_calls = []
    for m in matches:
        try:
            obj = json.loads(m.group(1).strip())
            args = obj.get("arguments", obj.get("parameters", {}))
            tool_calls.append({
                "name": obj.get("name", ""),
                "arguments": json.dumps(args, ensure_ascii=False),
            })
        except json.JSONDecodeError:
            pass
    return tool_calls, content


def _parse_llama3(text: str) -> Tuple[List[Dict], Optional[str]]:
    bot = text.startswith(_LLAMA3_BOT)
    if not bot and not text.startswith("{"):
        return [], text.strip() or None
    start = len(_LLAMA3_BOT) if bot else 0
    try:
        dec = json.JSONDecoder()
        obj, end = dec.raw_decode(text[start:])
        end += start
        args = obj.get("arguments", obj.get("parameters", {}))
        tool_calls = [{"name": obj.get("name", ""), "arguments": json.dumps(args, ensure_ascii=False)}]
        return tool_calls, text[end:].strip() or None
    except (json.JSONDecodeError, ValueError):
        return [], text.strip() or None


def _parse_output(text: str) -> Tuple[List[Dict], Optional[str]]:
    mt = MODEL_TYPE.strip().lower()
    if mt == "qwen3":
        return _parse_qwen3(text)
    if mt == "llama3":
        return _parse_llama3(text)
    return [], text.strip() or None


def _build_response(
    cid: str,
    tool_calls: List[Dict],
    content: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    vllm_finish_reason: str = "stop",
) -> Dict:
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in tool_calls
        ]
        finish_reason = "tool_calls"
    else:
        # Preserve vllm's finish_reason so "length" is not silently swallowed.
        finish_reason = vllm_finish_reason

    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": BASE_MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

# ---------------------------------------------------------------------------
# vllm /v1/completions helper
# ---------------------------------------------------------------------------

# Generation params safe to forward from the client to /v1/completions.
# logprobs/top_logprobs are excluded: Chat API uses bool/int semantics but
# legacy Completions API uses a different int-only scheme; forwarding as-is
# causes vllm to return 422.
_FORWARD_PARAMS = frozenset({
    "temperature", "top_p", "top_k", "seed",
    "frequency_penalty", "presence_penalty", "repetition_penalty",
})


async def _call_completions(
    prompt: str,
    model: str,
    stop: List[str],
    max_tokens: int,
    fwd: Dict,
) -> Dict:
    payload = {
        **fwd,
        "model": model,
        "prompt": prompt,
        "stop": stop,
        "max_tokens": max_tokens,
        "include_stop_str_in_output": False,
        "add_special_tokens": False,   # prompt is fully rendered; avoid duplicate BOS
        "stream": False,
    }
    r = await _http.post(f"{VLLM_BASE_URL.rstrip('/')}/completions", json=payload)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# Tool-call stop detection — only trust vllm's stop_reason field
# ---------------------------------------------------------------------------

def _is_tool_call_stop(choice: Dict) -> bool:
    return choice.get("stop_reason") == TOOL_CALL_END

# ---------------------------------------------------------------------------
# Two-phase request handler
# ---------------------------------------------------------------------------

async def _handle_request(
    cid: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    client_max_tokens: Optional[int],
    client_stop: List[str],
    fwd: Dict,
) -> Dict:
    msgs_for_model = _prepare_messages_for_model(messages)
    prompt = _render_prompt(msgs_for_model, tools)

    # Tokenize locally to compute how much room is left for generation.
    # _context_window is the total context length (vllm --max-model-len), not the
    # generation budget; the actual generation limit is window minus prompt length.
    prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
    p1_available = _context_window - prompt_token_count - 64   # 64-token safety buffer
    if p1_available <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt is too long ({prompt_token_count} tokens) for context window "
                f"({_context_window} tokens). Reduce message history or tool definitions."
            ),
        )
    p1_max = min(
        client_max_tokens if client_max_tokens is not None else _context_window,
        p1_available,
    )

    # TOOL_CALL_END first for deterministic logging; client stops deduped and appended.
    p1_stop = [TOOL_CALL_END] + [s for s in dict.fromkeys(client_stop) if s != TOOL_CALL_END]

    log.info("[phase1] base model  stop=%s  max_tokens=%d  prompt_tokens~=%d",
             p1_stop, p1_max, prompt_token_count)
    p1 = await _call_completions(prompt, BASE_MODEL_ID, p1_stop, p1_max, fwd)
    c1 = p1["choices"][0]
    text1: str = c1.get("text") or ""
    usage1 = p1.get("usage", {})
    p1_prompt_tokens: int = usage1.get("prompt_tokens") or prompt_token_count
    p1_finish_reason: str = c1.get("finish_reason") or "stop"

    if not _is_tool_call_stop(c1):
        log.info("[phase1] normal stop  finish_reason=%s", p1_finish_reason)
        tool_calls, content = _parse_output(text1)
        return _build_response(
            cid, tool_calls, content,
            p1_prompt_tokens,
            usage1.get("completion_tokens", 0),
            p1_finish_reason,
        )

    log.info("[phase1] hit </tool_call> — switching to lora model")

    # Phase 2 prompt: original prompt + phase1 text + connector. No re-rendering.
    p2_prompt = prompt + text1 + TOOL_CALL_END + TOOL_CALL_SECURITY_START

    # Estimate p2 prompt length; use vllm-reported counts where available.
    p2_prompt_est = p1_prompt_tokens + usage1.get("completion_tokens", 0) + 10
    p2_available = _context_window - p2_prompt_est - 64

    if p2_available < 64:
        # Not enough room for a meaningful security block. Sending a 1-token
        # request would produce garbage; skip phase 2 and return without it.
        log.warning(
            "[phase2] context exhausted (available=%d tokens), skipping security phase",
            p2_available,
        )
        tool_calls, content = _parse_output(text1)
        return _build_response(
            cid, tool_calls, content,
            p1_prompt_tokens,
            usage1.get("completion_tokens", 0),
            "length",
        )

    p2_max = min(MAX_TOKENS_SECURITY, p2_available)
    if p2_max < MAX_TOKENS_SECURITY:
        log.warning("[phase2] context nearly full, security max_tokens clamped to %d", p2_max)

    log.info("[phase2] lora model  stop=[%s]  max_tokens=%d", TOOL_CALL_SECURITY_END, p2_max)
    p2 = await _call_completions(p2_prompt, LORA_MODEL_ID, [TOOL_CALL_SECURITY_END], p2_max, fwd)
    c2 = p2["choices"][0]
    text2: str = c2.get("text") or ""
    usage2 = p2.get("usage", {})
    p2_finish_reason: str = c2.get("finish_reason") or "stop"

    # Both prefill runs are real compute; report their combined cost.
    total_prompt_tokens = p1_prompt_tokens + usage2.get("prompt_tokens", 0)
    total_completion_tokens = usage1.get("completion_tokens", 0) + usage2.get("completion_tokens", 0)

    full_text = text1 + TOOL_CALL_END + TOOL_CALL_SECURITY_START + text2 + TOOL_CALL_SECURITY_END
    log.info("[phase2] done  finish_reason=%s", p2_finish_reason)

    tool_calls, content = _parse_output(full_text)
    return _build_response(
        cid, tool_calls, content,
        total_prompt_tokens, total_completion_tokens,
        p2_finish_reason,
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body: Dict = await request.json()

    if body.get("stream"):
        raise HTTPException(status_code=501, detail="streaming is not supported")

    messages: List[Dict] = body.get("messages", [])
    tools: Optional[List[Dict]] = body.get("tools")
    client_max_tokens: Optional[int] = body.get("max_tokens")

    client_stop: Any = body.get("stop") or []
    if isinstance(client_stop, str):
        client_stop = [client_stop]

    fwd = {k: v for k, v in body.items() if k in _FORWARD_PARAMS}
    cid = f"chatcmpl-{uuid.uuid4().hex}"

    try:
        result = await _handle_request(cid, messages, tools, client_max_tokens, client_stop, fwd)
        return JSONResponse(result)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception:
        log.exception("unhandled error in /v1/chat/completions")
        raise HTTPException(status_code=500, detail="internal server error")


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def passthrough(request: Request, path: str):
    """Forward all other /v1/* requests to vllm unchanged."""
    body = await request.body()
    req_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP_BY_HOP | {"host"}}
    r = await _http.request(
        method=request.method,
        url=f"{VLLM_BASE_URL.rstrip('/')}/{path}",
        content=body,
        headers=req_headers,
        params=dict(request.query_params),
    )
    resp_headers = {k: v for k, v in r.headers.items() if k.lower() not in _HOP_BY_HOP}
    return Response(content=r.content, status_code=r.status_code, headers=resp_headers)


@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global VLLM_BASE_URL, BASE_MODEL_ID, LORA_MODEL_ID, BASE_MODEL_PATH, MODEL_TYPE
    global LLAMA3_TEMPLATE_PATH, MAX_TOKENS_SECURITY, REQUEST_TIMEOUT
    global LISTEN_HOST, LISTEN_PORT, STRIP_SECURITY_IN_HISTORY, ENABLE_THINKING
    global tokenizer, _llama3_template

    parser = argparse.ArgumentParser(description="vllm two-phase inference proxy")
    parser.add_argument("--vllm-url",             default=None, metavar="URL",
                        help=f"vllm base URL (default: {VLLM_BASE_URL})")
    parser.add_argument("--base-model-id",         default=None, metavar="ID",
                        help=f"vllm model ID for base model (default: {BASE_MODEL_ID})")
    parser.add_argument("--lora-model-id",         default=None, metavar="ID",
                        help=f"vllm model ID for lora model (default: {LORA_MODEL_ID})")
    parser.add_argument("--base-model-path",       default=None, metavar="PATH",
                        help=f"local path used to load the tokenizer (default: {BASE_MODEL_PATH})")
    parser.add_argument("--model-type",            default=None, choices=["Qwen3", "Llama3"],
                        help=f"model type (default: {MODEL_TYPE})")
    parser.add_argument("--llama3-template",       default=None, metavar="PATH",
                        help=f"path to Llama3 Jinja chat template (default: {LLAMA3_TEMPLATE_PATH})")
    parser.add_argument("--max-tokens-security",   type=int, default=None, metavar="N",
                        help=f"max tokens for phase 2 / lora security block (default: {MAX_TOKENS_SECURITY})")
    parser.add_argument("--timeout",               type=int, default=None, metavar="SEC",
                        help=f"HTTP request timeout in seconds (default: {REQUEST_TIMEOUT})")
    parser.add_argument("--host",                  default=None,
                        help=f"listen host (default: {LISTEN_HOST})")
    parser.add_argument("--port",                  type=int, default=None,
                        help=f"listen port (default: {LISTEN_PORT})")
    parser.add_argument("--no-strip-security-in-history", action="store_true",
                        help="keep <tool_call_security> in history (reordered to match generation order)")
    parser.add_argument("--no-enable-thinking",    action="store_true",
                        help="disable Qwen3 thinking mode (pass enable_thinking=False to template)")
    parser.add_argument("--log-level",             default="info",
                        help="log level: debug/info/warning/error (default: info)")
    args = parser.parse_args()

    if args.vllm_url:            VLLM_BASE_URL        = args.vllm_url
    if args.base_model_id:       BASE_MODEL_ID        = args.base_model_id
    if args.lora_model_id:       LORA_MODEL_ID        = args.lora_model_id
    if args.base_model_path:   BASE_MODEL_PATH      = args.base_model_path
    if args.model_type:          MODEL_TYPE           = args.model_type
    if args.llama3_template:     LLAMA3_TEMPLATE_PATH = args.llama3_template
    if args.max_tokens_security: MAX_TOKENS_SECURITY  = args.max_tokens_security
    if args.timeout:             REQUEST_TIMEOUT      = args.timeout
    if args.host:                LISTEN_HOST          = args.host
    if args.port:                LISTEN_PORT          = args.port
    if args.no_strip_security_in_history:
        STRIP_SECURITY_IN_HISTORY = False
    if args.no_enable_thinking:
        ENABLE_THINKING = False

    logging.getLogger().setLevel(args.log_level.upper())

    log.info("Loading tokenizer from %s", BASE_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    log.info("Tokenizer loaded")

    if MODEL_TYPE.lower() == "llama3":
        log.info("Loading Llama3 template from %s", LLAMA3_TEMPLATE_PATH)
        with open(LLAMA3_TEMPLATE_PATH, encoding="utf-8") as f:
            _llama3_template = f.read()
        log.info("Llama3 template loaded")

    log.info("vllm-server starting up")
    log.info("  listen           : http://%s:%d/v1", LISTEN_HOST, LISTEN_PORT)
    log.info("  vllm             : %s", VLLM_BASE_URL)
    log.info("  model type       : %s", MODEL_TYPE)
    log.info("  base model       : %s", BASE_MODEL_ID)
    log.info("  lora model       : %s  security_max_tokens=%d", LORA_MODEL_ID, MAX_TOKENS_SECURITY)
    log.info("  enable_thinking  : %s", ENABLE_THINKING)
    log.info("  strip security   : %s  timeout=%ds", STRIP_SECURITY_IN_HISTORY, REQUEST_TIMEOUT)
    log.info("  context window   : fetched from vllm at startup")

    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
