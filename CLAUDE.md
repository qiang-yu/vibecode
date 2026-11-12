# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`defence-llm-server` is an OpenAI-compatible HTTP proxy (FastAPI + uvicorn) that sits between an agent client and an LLM, and defends against prompt-injection-driven tool calls. The whole server lives in a single file: `defence-llm-server.py` (~2500 lines). It targets **Qwen3 only** (tokenizer and chat template are loaded from a local Qwen3 model directory).

There is no test suite, linter config, or build step. Dependencies are in `requirements.txt` (tested on Python 3.12, conda env "RQwen").

## Running

```bash
pip install -r requirements.txt
./start_defence-llm-server.sh            # main entry point; all config lives at the top of this script
source set_env_local.sh && ./start_defence-llm-server.sh   # point phase 1 at a local vllm
curl http://localhost:29000/health
```

Configuration precedence (highest first):
1. Shell env vars — only for the phase-1 `LLM_*` settings (`LLM_SERVER_URL`, `LLM_MODEL_ID`, `LLM_SERVER_PROXY`, `LLM_SERVER_TOKEN_LIST`, `LLM_CONTEXT_WINDOW`, `LLM_API_CALL_INTERVAL`). Both the start script and `main()` honor them.
2. CLI args passed by `start_defence-llm-server.sh` (its variable block is the de facto config file).
3. Module-level constants at the top of `defence-llm-server.py`. Some settings exist **only** as constants (e.g. `SECURITY_VALIDATE_TOOL_REASON`, `PREFILL_SECURITY_HEADER`, `SECURITY_RULES_TEXT`, `SECURITY_QUESTIONS`, `FAKE_TOOL_RESPONSE_CONTENT`, `VLLM_INFERENCE_DEBUG`).

When adding a new option, wire it through all three places: the constant, the `argparse` arg + `global` declaration in `main()`, and the variable + `--flag` in the start script. Note the CLI flags mix `--dash-style` and `--underscore_style`; match the existing flag for the setting being touched.

Startup requires `BASE_MODEL_PATH` to hold a Qwen3 tokenizer, and (when `PHASE2_ENABLE`) the secure server's `/models` must list `SECURE_MODEL_ID`, otherwise startup aborts. Logs go to stdout and to `YYYYMMDD_<LOG_FILE_NAME>` next to the script.

## Architecture

### Endpoints
- `POST /v1/chat/completions` — the defended path (`_handle_request`). Streaming is rejected (501). The response is a normal `chat.completion` plus an extra `messages` field holding the (possibly repaired) conversation.
- `/v1/{path}` — raw passthrough to the phase-1 LLM server.
- `GET /health`.

### Two-phase pipeline (`_handle_request`)
1. **Phase 1 — base LLM**: the native conversation (messages + tools) is sent to any OpenAI-compatible chat backend (`_call_chat_completions`, `{LLM_SERVER_URL}/chat/completions`). Calls are serialized by `_llm_call_lock` with a minimum `LLM_API_CALL_INTERVAL`, and bearer tokens from `LLM_SERVER_TOKEN_LIST` are rotated round-robin. The local tokenizer is used only to estimate prompt size against `LLM_CONTEXT_WINDOW`. If no tool calls come back, the answer is returned directly.
2. **Phase 2 — security LoRA**: each tool call is isolated (other calls removed) and its single-call assistant turn is rendered as raw Qwen3 text, then sent to the secure server's raw `/v1/completions` (`_call_completions`) with a prefilled `<tool_call_security>` header (`_build_security_prefill`) so the LoRA only writes the reasoning. The LoRA emits a block containing `<tool_name>`, `<tool_args>`, `<tool_trace>`, `<trigger_words>`, `<tool_reason>` (Q:/Summary: reasoning over `SECURITY_QUESTIONS`) and `<tool_security>` (safe | neutral | suspicious | unsafe). `_validate_and_fix_tool_reason` repairs derailed reasoning; overruns are retried `PHASE2_TOOL_REASON_RETRY_COUNT` times.
3. **Verdict** (`_check_defence_verdict`): verdicts below `SECURITY_DEFENCE_LEVEL` are blocked. Malformed blocks are logged and let through. With `DEFENCE_SAFE_TOOLCALL`, an exact "safe" verdict is double-checked: its trigger words must appear in a **user** message, otherwise it goes down the defence path too (using `DEFENCE_SAFE_METHOD_LIST` and the `DEFENCE_SAFE_*` parameters).
4. **Defence methods**, tried in list order until one succeeds; if none does, the turn passes through undefended (intentionally visible, not masked):
   - `remove_trigger_words` — locate the trigger words inside tool responses (exact, then punctuation-insensitive, optionally fuzzy via `difflib`) and excise them (`_excise_trigger_words`), then re-run phase 1 on the repaired history. Trigger words that match the tool call's own name/args can be treated as a false positive (`*_MATCH_TOOL_CALL`).
   - `fake_tool_response` — append the blocked call plus a fabricated tool response (`FAKE_TOOL_RESPONSE_CONTENT`, placeholders substituted via `str.replace`, not `str.format`) and re-run phase 1.
   - `fake_user_refuse` — independent of `fake_tool_response`: for phase 1, append the blocked call + a rejection tool response (`FAKE_USER_REFUSE_TOOL_RESPONSE`) + a fabricated user refusal (`FAKE_USER_REFUSE_CONTENT`), then re-run phase 1. `_build_heads` renders the phase-2 head from `_strip_fake_user_refuse(msgs)`, so the lora never sees that group and judges the next call as if it directly followed the original history. The fabricated user messages are also excluded from the safe-verdict user-message check.
   The outer loop re-runs phase 1 up to `SECURITY_DEFENCE_MAX_RETRIES` times. To add a method, add its name to the list constants and a matching branch in the defence dispatch loop in `_handle_request`.

### Key invariants
- `<tool_call_security>` is injected only by this server. It is always stripped from incoming history before rendering, and stripped from responses unless `SECURITY_DEFENCE_DEBUG` is on.
- Phase 1 sees native messages; phase 2 sees a locally rendered Qwen3 prompt with all historical `<think>` blocks removed (`_build_heads` builds both, and must be re-run whenever the defence rewrites `work_messages`).
- Native `tool_calls` from phase 1 are re-rendered as Qwen3 `<tool_call>` text (`_render_tool_calls_as_text`) so the text-based phase-2/defence pipeline and `_parse_output` / `_parse_qwen3` can be reused; the trailing `</tool_call>` (`TOOL_CALL_END`) is stripped and re-appended by convention.
- The `developer` role is mapped to `system` (`_normalize_messages`).
