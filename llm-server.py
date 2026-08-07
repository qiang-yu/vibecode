###
# LLM server with OpenAI-compatible API implementing two-phase inference:
# phase 1 runs the base model until </tool_call>, then phase 2 runs the LoRA
# adapter starting from <tool_call_security> while sharing the KV cache so the
# common prefix is never re-prefilled.
###

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import importlib as _importlib

_chat_tpl = _importlib.import_module("llm-server-chat-templates")
_tool_parsers = _importlib.import_module("llm-server-tool-parsers")

apply_chat_template = _chat_tpl.apply_chat_template
build_chat_completion = _tool_parsers.build_chat_completion
clean_messages = _tool_parsers.clean_messages
parse_model_output = _tool_parsers.parse_model_output

# --------------------------------------------------------------------------- #
# Console logger (stdout)
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_parser = argparse.ArgumentParser(description="llm-server")
# fmt: off
_parser.add_argument("--config",          default=str(Path(__file__).parent / "llm-server-config.yaml"), help="Path to the YAML config file")
_parser.add_argument("--base-model-path", default=None, help="Override model.base_model_path")
_parser.add_argument("--lora-adapter-path", default=None, help="Override model.lora_adapter_path (empty string = no LoRA)")
_parser.add_argument("--model-type",      default=None, help="Override model.model_type (Qwen3/Llama3/Mistral3)")
_parser.add_argument("--max-tokens",      type=int, default=None, help="Override model.max_tokens")
_parser.add_argument("--device",          default=None, help="Override model.device")
_parser.add_argument("--dtype",           default=None, help="Override model.dtype (bfloat16/float16/float32)")
_parser.add_argument("--host",            default=None, help="Override server.host")
_parser.add_argument("--port",            type=int, default=None, help="Override server.port")
_parser.add_argument("--llama3-chat-template", default=None, help="Override templates.llama3_chat_template")
_parser.add_argument("--raw-message-log", default=None, help="Override logging.raw_message_log")
_parser.add_argument("--strip-tool-call-security-in-history",
                     action=argparse.BooleanOptionalAction, default=None,
                     help="Override strip_tool_call_security_in_history (--no-... to disable)")
# fmt: on
_args, _ = _parser.parse_known_args()

CONFIG_PATH = _args.config
log.info("Loading config from %s", CONFIG_PATH)
cfg = load_config(CONFIG_PATH)

MODEL_CFG = cfg["model"]
SERVER_CFG = cfg["server"]
TMPL_CFG = cfg.get("templates", {})
LOG_CFG = cfg.get("logging", {})

# Apply CLI overrides (None means the flag was not provided; keep config value).
def _cli(arg_val, cfg_val):
    return arg_val if arg_val is not None else cfg_val

BASE_MODEL_PATH: str       = _cli(_args.base_model_path,   MODEL_CFG["base_model_path"])
LORA_ADAPTER_PATH: str     = (_cli(_args.lora_adapter_path, MODEL_CFG.get("lora_adapter_path") or "")).strip()
HAS_LORA: bool             = bool(LORA_ADAPTER_PATH)
MODEL_TYPE: str            = _cli(_args.model_type,         MODEL_CFG["model_type"])
MAX_TOKENS: int            = int(_cli(_args.max_tokens,     MODEL_CFG["max_tokens"]))
DEVICE: str                = _cli(_args.device,             MODEL_CFG.get("device", "cuda"))
DTYPE_STR: str             = _cli(_args.dtype,              MODEL_CFG.get("dtype", "bfloat16"))
_HOST: str                 = _cli(_args.host,               SERVER_CFG.get("host", "0.0.0.0"))
_PORT: int                 = int(_cli(_args.port,           SERVER_CFG.get("port", 19001)))
STRIP_SECURITY_IN_HISTORY: bool = _cli(
    _args.strip_tool_call_security_in_history,
    cfg.get("strip_tool_call_security_in_history", True),
)

_cfg_dir = Path(CONFIG_PATH).parent

def _resolve_path(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    p = Path(val)
    return str(p if p.is_absolute() else _cfg_dir / p)

_llama3_tpl_raw = _cli(_args.llama3_chat_template, TMPL_CFG.get("llama3_chat_template"))
LLAMA3_TEMPLATE_PATH: Optional[str] = _resolve_path(_llama3_tpl_raw)

_raw_log_raw = _cli(_args.raw_message_log, LOG_CFG.get("raw_message_log", "raw_message.log"))

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
DTYPE = DTYPE_MAP[DTYPE_STR]

# --------------------------------------------------------------------------- #
# Raw message file logger (one physical line per entry)
# --------------------------------------------------------------------------- #

class _SingleLineFormatter(logging.Formatter):
    """Each log entry is written as one physical line: newlines become literal \n."""
    def format(self, record: logging.LogRecord) -> str:
        # Let the base class handle %s substitution and all other formatting,
        # then escape newlines in the final output string.
        return super().format(record).replace("\n", "\\n").replace("\r", "\\r")


_raw_log_path = _resolve_path(_raw_log_raw)
_raw_handler = logging.FileHandler(_raw_log_path, encoding="utf-8")
_raw_handler.setFormatter(
    _SingleLineFormatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
raw_log = logging.getLogger("raw")
raw_log.setLevel(logging.DEBUG)
raw_log.addHandler(_raw_handler)
raw_log.propagate = False  # do not echo to stdout

log.info("Raw message log -> %s", _raw_log_path)

# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

log.info("Loading tokenizer from %s", BASE_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

eos_token_ids: set[int] = set()
if tokenizer.eos_token_id is not None:
    eos_token_ids.add(tokenizer.eos_token_id)
for _tok_str in ("<|im_end|>", "<|eot_id|>", "</s>"):
    _tid = tokenizer.convert_tokens_to_ids(_tok_str)
    if isinstance(_tid, int) and _tid != tokenizer.unk_token_id:
        eos_token_ids.add(_tid)

log.info("EOS token IDs: %s", eos_token_ids)

log.info("Loading base model from %s", BASE_MODEL_PATH)
_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    attn_implementation="sdpa",
    trust_remote_code=True,
)

if HAS_LORA:
    log.info("Loading LoRA adapter from %s", LORA_ADAPTER_PATH)
    model: Any = PeftModel.from_pretrained(_base, LORA_ADAPTER_PATH)
else:
    log.info("No LoRA adapter configured — running base model only.")
    model: Any = _base

model.eval()
log.info("Model ready. HAS_LORA=%s", HAS_LORA)

_device = next(model.parameters()).device

# --------------------------------------------------------------------------- #
# Generation helpers
# --------------------------------------------------------------------------- #

def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    if temperature == 0.0:
        return int(logits[0].argmax())

    logits = logits[0].float() / temperature
    # topk first so softmax + multinomial operate on a small tensor
    k = min(top_k if top_k > 0 else 1024, logits.size(-1))
    vals, idx = torch.topk(logits, k)   # already descending
    probs = torch.softmax(vals, dim=-1)

    if 0.0 < top_p < 1.0:
        cum = torch.cumsum(probs, dim=-1)
        probs[(cum - probs) > top_p] = 0.0
        probs.div_(probs.sum())

    return int(idx[torch.multinomial(probs, 1)])


@torch.no_grad()
def _forward(
    token_ids: torch.Tensor,
    past_key_values: Any,
) -> tuple[torch.Tensor, Any]:
    out = model(token_ids, past_key_values=past_key_values, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def generate_two_phase(
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, int, bool]:
    """
    Two-phase generation with shared KV cache.
    Returns (full_generated_text, completion_tokens, tool_call_was_produced).

    When HAS_LORA is False, runs a single-phase base-model generation with no
    tool_call stop detection and no <tool_call_security> injection.
    """
    input_tensor = torch.tensor([input_ids], device=_device)

    raw_log.info("[PHASE-1] Model: base model | path: %s", BASE_MODEL_PATH)

    # ---- Phase 1: base model prefill + decode ----------------------------
    # Wrap the entire phase so adapter toggle happens once, not per step.
    phase1_ids: list[int] = []
    tool_call_stopped = False

    phase1_ctx = model.disable_adapter() if HAS_LORA else contextlib.nullcontext()
    with phase1_ctx:
        logits, past_kv = _forward(input_tensor, None)

        while len(phase1_ids) < max_new_tokens:
            tok = _sample_next_token(logits, temperature, top_p, top_k)
            phase1_ids.append(tok)

            if tok in eos_token_ids:
                break

            logits, past_kv = _forward(
                torch.tensor([[tok]], device=_device), past_kv
            )

            if HAS_LORA:
                # Decode only the tail to avoid O(n) work each step.
                tail = tokenizer.decode(phase1_ids[-16:], skip_special_tokens=False)
                if "</tool_call>" in tail:
                    tool_call_stopped = True
                    break

    phase1_text = tokenizer.decode(phase1_ids, skip_special_tokens=False)

    if not tool_call_stopped:
        raw_log.info(
            "[PHASE-1] Stop: EOS / max_tokens (no tool call)\n"
            "[PHASE-1] Output:\n%s",
            phase1_text,
        )
        return phase1_text, len(phase1_ids), False

    raw_log.info(
        "[PHASE-1] Stop: </tool_call> detected\n"
        "[PHASE-1] Output:\n%s",
        phase1_text,
    )

    # ---- Phase 2: inject <tool_call_security> and LoRA decoding ----------
    # Only reached when HAS_LORA is True. LoRA is active by default (no ctx).
    raw_log.info("[PHASE-2] Model: LoRA adapter | path: %s", LORA_ADAPTER_PATH)

    inject_text = "<tool_call_security>"
    inject_ids: list[int] = tokenizer.encode(inject_text, add_special_tokens=False)

    for inj_tok in inject_ids:
        logits, past_kv = _forward(
            torch.tensor([[inj_tok]], device=_device), past_kv
        )

    phase2_ids: list[int] = list(inject_ids)
    remaining = max_new_tokens - len(phase1_ids) - len(inject_ids)

    for _ in range(max(remaining, 0)):
        tok = _sample_next_token(logits, temperature, top_p, top_k)
        phase2_ids.append(tok)

        if tok in eos_token_ids:
            break

        logits, past_kv = _forward(
            torch.tensor([[tok]], device=_device), past_kv
        )

        tail2 = tokenizer.decode(phase2_ids[-24:], skip_special_tokens=False)
        if "</tool_call_security>" in tail2:
            break

    phase2_text = tokenizer.decode(phase2_ids, skip_special_tokens=False)
    raw_log.info(
        "[PHASE-2] Stop: </tool_call_security> detected\n"
        "[PHASE-2] Output:\n%s",
        phase2_text,
    )

    full_text = phase1_text + phase2_text
    return full_text, len(phase1_ids) + len(phase2_ids), True


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(title="llm-server")


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class Message(BaseModel):
    role: str
    content: Optional[Any] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[Message]
    tools: Optional[list[ToolDefinition]] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    model_config = {"extra": "ignore"}


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request, request: ChatCompletionRequest):
    t0 = time.time()
    log.info("REQUEST  POST /v1/chat/completions  received")

    # Log raw incoming JSON.
    try:
        raw_body = await raw_request.body()
        raw_log.info(
            "[REQUEST] POST /v1/chat/completions\n%s",
            json.dumps(json.loads(raw_body), ensure_ascii=False, indent=2),
        )
    except Exception:
        raw_log.info("[REQUEST] POST /v1/chat/completions  (could not decode body)")

    if request.stream:
        log.warning("RESPONSE POST /v1/chat/completions  501 streaming not supported")
        raise HTTPException(status_code=501, detail="Streaming is not supported.")

    messages = [m.model_dump(exclude_none=True) for m in request.messages]

    # Optionally strip <tool_call_security> from history before feeding to the model.
    messages_for_model = clean_messages(messages) if STRIP_SECURITY_IN_HISTORY else messages

    tools_for_template: Optional[list[dict]] = (
        [t.model_dump() for t in request.tools] if request.tools else None
    )

    try:
        prompt = apply_chat_template(
            messages=messages_for_model,
            tools=tools_for_template,
            tokenizer=tokenizer,
            model_type=MODEL_TYPE,
            llama3_template_path=LLAMA3_TEMPLATE_PATH,
        )
    except Exception as exc:
        log.error("RESPONSE POST /v1/chat/completions  400 chat template error: %s", exc)
        raw_log.info("[TEMPLATE] Error: %s", exc)
        raise HTTPException(status_code=400, detail=f"Chat template error: {exc}")

    raw_log.info("[TEMPLATE] Prompt sent to model:\n%s", prompt)

    input_ids: list[int] = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_tokens = len(input_ids)

    max_new = min(
        request.max_tokens if request.max_tokens else MAX_TOKENS,
        MAX_TOKENS,
    )
    temperature = request.temperature if request.temperature is not None else 1.0
    top_p = request.top_p if request.top_p is not None else 1.0
    top_k = request.top_k if request.top_k is not None else 0

    log.info(
        "Generating: prompt_tokens=%d max_new=%d temp=%.2f top_p=%.2f",
        prompt_tokens, max_new, temperature, top_p,
    )

    try:
        generated_text, completion_tokens, had_tool_call = generate_two_phase(
            input_ids=input_ids,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        log.error(
            "RESPONSE POST /v1/chat/completions  500 generation error (%.2fs): %s",
            elapsed, exc,
        )
        raw_log.info("[ERROR] Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}")

    tool_calls, content = parse_model_output(generated_text, MODEL_TYPE)

    response = build_chat_completion(
        tool_calls=tool_calls,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cleaned_input_messages=messages_for_model,
    )

    elapsed = time.time() - t0
    log.info(
        "RESPONSE POST /v1/chat/completions  200 OK  "
        "completion_tokens=%d had_tool_call=%s elapsed=%.2fs",
        completion_tokens, had_tool_call, elapsed,
    )
    return JSONResponse(content=response)


@app.get("/health")
async def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=_HOST,
        port=_PORT,
        log_level="info",
    )
