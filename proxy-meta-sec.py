###
# OpenAI-compatible API proxy that rewrites a configurable message role
# (default: "tool" -> "input") before forwarding chat/completions requests
# to a backend LLM server. Logs received and forwarded messages on single
# lines, and prints cumulative token-usage statistics every 10 proxy calls
# and on program exit.
###

import argparse
import asyncio
from datetime import datetime
import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

LISTEN_HOST     = "localhost"
LISTEN_PORT     = 39007
BACKEND_URL     = "http://localhost:19007/v1"
SOURCE_ROLE     = "tool"
TARGET_ROLE     = "input"
LOG_FILE_NAME   = "proxy.log"
LOG_LEVEL       = "info"
REQUEST_TIMEOUT = 600

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


def _setup_logging(log_file: str, log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    date_prefix = datetime.now().strftime("%Y%m%d")
    if "." in log_file:
        base, ext = log_file.rsplit(".", 1)
        dated = f"{date_prefix}_{base}.{ext}"
    else:
        dated = f"{date_prefix}_{log_file}"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)
    for handler in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(dated, encoding="utf-8"),
    ]:
        handler.setFormatter(fmt)
        root.addHandler(handler)
    log.info("Logging to file: %s (level=%s)", dated, log_level.upper())


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_http: Optional[httpx.AsyncClient] = None
_stats_lock: Optional[asyncio.Lock] = None

_proxy_call_count: int = 0
_total_prompt_tokens: int = 0
_total_completion_tokens: int = 0
_total_tokens: int = 0
_error_count: int = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_single_line(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _rewrite_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for msg in messages:
        if msg.get("role") == SOURCE_ROLE:
            msg = dict(msg)
            msg["role"] = TARGET_ROLE
        result.append(msg)
    return result


def _log_stats(final: bool = False) -> None:
    label = "[stats_final]" if final else "[stats]"
    log.info(
        "%s calls=%d errors=%d prompt_tokens=%d completion_tokens=%d total_tokens=%d",
        label,
        _proxy_call_count,
        _error_count,
        _total_prompt_tokens,
        _total_completion_tokens,
        _total_tokens,
    )


async def _record_call(prompt_tok: int = 0, comp_tok: int = 0, total_tok: int = 0) -> None:
    global _proxy_call_count, _total_prompt_tokens, _total_completion_tokens, _total_tokens
    if _stats_lock is None:
        return
    async with _stats_lock:
        _proxy_call_count += 1
        _total_prompt_tokens += prompt_tok
        _total_completion_tokens += comp_tok
        _total_tokens += total_tok
        count = _proxy_call_count
    if prompt_tok or comp_tok:
        log.info("[usage] prompt=%d completion=%d total=%d", prompt_tok, comp_tok, total_tok)
    if count % 10 == 0:
        _log_stats()


async def _record_error() -> None:
    global _error_count
    if _stats_lock is None:
        return
    async with _stats_lock:
        _error_count += 1

# ---------------------------------------------------------------------------
# Hop-by-hop headers that must not cross HTTP hops
# ---------------------------------------------------------------------------

_HOP_BY_HOP = frozenset({
    "content-encoding", "content-length", "transfer-encoding",
    "connection", "keep-alive", "te", "trailers", "upgrade",
})

# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http, _stats_lock
    _stats_lock = asyncio.Lock()
    _http = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    log.info(
        "Proxy ready on %s:%d -> %s  (rewriting role %r -> %r)",
        LISTEN_HOST, LISTEN_PORT, BACKEND_URL, SOURCE_ROLE, TARGET_ROLE,
    )
    yield
    _log_stats(final=True)
    await _http.aclose()
    log.info("HTTP client closed")


app = FastAPI(title="proxy-meta-sec", docs_url=None, redoc_url=None, lifespan=lifespan)

# ---------------------------------------------------------------------------
# Streaming response factory
# ---------------------------------------------------------------------------

def _make_streaming_response(
    method: str,
    target_url: str,
    fwd_headers: Dict[str, str],
    send_bytes: bytes,
    query_params: Dict[str, str],
    is_chat: bool,
) -> StreamingResponse:

    async def generate():
        prompt_tok = comp_tok = total_tok = 0
        ok = False
        try:
            async with _http.stream(
                method=method,
                url=target_url,
                headers=fwd_headers,
                content=send_bytes or None,
                params=query_params,
            ) as resp:
                async for raw in resp.aiter_bytes():
                    yield raw
                    if is_chat:
                        try:
                            text = raw.decode("utf-8", errors="ignore")
                            for line in text.splitlines():
                                if line.startswith("data: ") and "[DONE]" not in line:
                                    try:
                                        usage = json.loads(line[6:]).get("usage") or {}
                                        if usage:
                                            prompt_tok = usage.get("prompt_tokens", prompt_tok)
                                            comp_tok = usage.get("completion_tokens", comp_tok)
                                            total_tok = usage.get("total_tokens", prompt_tok + comp_tok)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
            ok = True
        except Exception as exc:
            log.error("[proxy] streaming failed: %s", exc)
            await _record_error()
            return
        if ok:
            await _record_call(prompt_tok, comp_tok, total_tok)

    return StreamingResponse(generate(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Main proxy route — catches all paths
# ---------------------------------------------------------------------------

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(request: Request, full_path: str):
    # Determine if this is a chat completions call that needs rewriting
    is_chat = (
        full_path.strip("/") == "v1/chat/completions"
        and request.method == "POST"
    )

    # Map /v1/<subpath> to BACKEND_URL/<subpath>
    # BACKEND_URL already ends in /v1, so strip the leading v1/ from the path.
    p = full_path.lstrip("/")
    if p.startswith("v1/"):
        subpath = p[len("v1/"):]
    elif p == "v1":
        subpath = ""
    else:
        subpath = p
    target_url = BACKEND_URL.rstrip("/") + ("/" + subpath if subpath else "")

    # Read body
    body_bytes = await request.body()
    body: Dict[str, Any] = {}
    if body_bytes:
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            pass

    # Rewrite messages for chat/completions; log before and after
    send_bytes = body_bytes
    if is_chat and isinstance(body.get("messages"), list):
        original_messages = body["messages"]
        rewritten_messages = _rewrite_messages(original_messages)
        log.info("[recv] messages=%s", _to_single_line(original_messages))
        log.info("[send] messages=%s", _to_single_line(rewritten_messages))
        modified_body = dict(body)
        modified_body["messages"] = rewritten_messages
        send_bytes = json.dumps(modified_body, ensure_ascii=False).encode("utf-8")

    # Build forwarding headers (strip hop-by-hop and host)
    fwd_headers: Dict[str, str] = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
    }
    if is_chat and send_bytes != body_bytes:
        fwd_headers["content-type"] = "application/json; charset=utf-8"

    # Streaming path
    if is_chat and bool(body.get("stream")):
        return _make_streaming_response(
            request.method, target_url, fwd_headers,
            send_bytes, dict(request.query_params), is_chat,
        )

    # Non-streaming path
    try:
        resp = await _http.request(
            method=request.method,
            url=target_url,
            headers=fwd_headers,
            content=send_bytes or None,
            params=dict(request.query_params),
        )
    except Exception as exc:
        log.error("[proxy] backend request failed: %s", exc)
        await _record_error()
        return Response(
            content=json.dumps({"error": {"message": str(exc), "type": "proxy_error"}}).encode(),
            status_code=502,
            media_type="application/json",
        )

    # Extract token usage from chat completions response
    prompt_tok = comp_tok = total_tok = 0
    if is_chat:
        try:
            rj = json.loads(resp.content)
            usage = rj.get("usage") or {}
            prompt_tok = usage.get("prompt_tokens", 0)
            comp_tok = usage.get("completion_tokens", 0)
            total_tok = usage.get("total_tokens", prompt_tok + comp_tok)
        except Exception:
            pass
    await _record_call(prompt_tok, comp_tok, total_tok)

    resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}
    return Response(content=resp.content, status_code=resp.status_code, headers=resp_headers)

# ---------------------------------------------------------------------------
# CLI and entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global LISTEN_HOST, LISTEN_PORT, BACKEND_URL, SOURCE_ROLE, TARGET_ROLE
    global LOG_FILE_NAME, LOG_LEVEL, REQUEST_TIMEOUT

    parser = argparse.ArgumentParser(description="OpenAI-compatible proxy with role rewriting")
    parser.add_argument("--host",          default=LISTEN_HOST,     help="Listen host (default: localhost)")
    parser.add_argument("--port",          default=LISTEN_PORT,     type=int, help="Listen port (default: 39007)")
    parser.add_argument("--backend-url",   default=BACKEND_URL,     help="Backend base URL ending in /v1 (default: http://localhost:19007/v1)")
    parser.add_argument("--source-role",   default=SOURCE_ROLE,     help="Message role to rewrite (default: tool)")
    parser.add_argument("--target-role",   default=TARGET_ROLE,     help="Replacement role value (default: input)")
    parser.add_argument("--log-file-name", default=LOG_FILE_NAME,   help="Log file base name (default: proxy.log)")
    parser.add_argument("--log-level",     default=LOG_LEVEL,       help="Log level: debug|info|warning|error (default: info)")
    parser.add_argument("--timeout",       default=REQUEST_TIMEOUT, type=int, help="Backend request timeout in seconds (default: 600)")
    args = parser.parse_args()

    LISTEN_HOST     = args.host
    LISTEN_PORT     = args.port
    BACKEND_URL     = args.backend_url
    SOURCE_ROLE     = args.source_role
    TARGET_ROLE     = args.target_role
    LOG_FILE_NAME   = args.log_file_name
    LOG_LEVEL       = args.log_level
    REQUEST_TIMEOUT = args.timeout

    _setup_logging(LOG_FILE_NAME, LOG_LEVEL)

    uvicorn.run(
        app,
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        log_level=LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
