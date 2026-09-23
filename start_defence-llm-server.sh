#!/usr/bin/env bash

# -----------------------------------------------------------------------
# Capture environment variables before the defaults below overwrite them.
# If these are set in the shell environment, they will override the defaults.
# -----------------------------------------------------------------------
_ENV_LLM_SERVER_URL="$LLM_SERVER_URL"
_ENV_LLM_MODEL_ID="$LLM_MODEL_ID"
_ENV_LLM_SERVER_PROXY="$LLM_SERVER_PROXY"
_ENV_LLM_SERVER_TOKEN_LIST="$LLM_SERVER_TOKEN_LIST"
_ENV_LLM_CONTEXT_WINDOW="$LLM_CONTEXT_WINDOW"
_ENV_LLM_API_CALL_INTERVAL="$LLM_API_CALL_INTERVAL"

# -----------------------------------------------------------------------
# Configurable parameters — edit these before starting the server
# -----------------------------------------------------------------------

# Phase 1: LLM server (OpenAI-compatible chat backend, e.g. vllm, Nvidia, OpenRouter).
# The server calls {LLM_SERVER_URL}/chat/completions, so give the base URL ending in /v1.
LLM_SERVER_URL="http://localhost:19000/v1"
LLM_MODEL_ID="Qwen3Base"
# Proxy used to reach the remote LLM; leave empty ("") to connect directly.
LLM_SERVER_PROXY=""
# Comma-separated bearer tokens for the remote LLM; rotated round-robin per call.
# Keep tokens OUT of this file. Export them in your environment instead:
#   export LLM_SERVER_TOKEN_LIST="token1,token2,token3"
LLM_SERVER_TOKEN_LIST=""
# Phase-1 context length in tokens; remote chat APIs cannot report max_model_len via /models.
LLM_CONTEXT_WINDOW=32768
# Minimum seconds between consecutive phase-1 LLM API calls (Nvidia free API rate limit).
LLM_API_CALL_INTERVAL=2.0

# Phase 2: secure server running the lora security model
SECURE_SERVER_URL="http://localhost:19000/v1"
SECURE_MODEL_ID="lora-model"

LISTEN_HOST="localhost"
LISTEN_PORT=29000

BASE_MODEL_PATH="/home/qiangyu/Models/Qwen/Qwen3-8B"

SEC_INFERENCE_MAX_TOKENS=2048  # hard cap on tokens generated per phase-2 SEC model call
LLM_INFERENCE_MAX_TOKENS=4096  # hard cap on tokens generated per phase-1 LLM API call
REQUEST_TIMEOUT=600              # HTTP request timeout in seconds
LOG_LEVEL="info"                 # debug | info | warning | error
LOG_FILE_NAME="defence-llm-server.log"  # log file base name; runtime prepends YYYYMMDD_

OUTPUT_RAW_CLIENT_INPUT=false    # true: log raw client input (Qwen3 format) before stripping

ENABLE_THINKING=true              # true | false
PHASE2_ENABLE=true                # true: run phase-2 security check; false: phase-1 only
PHASE2_TOOL_REASON_RETRY_COUNT=1  # retry phase 2 N times when its security block overruns max_tokens

# Security defence: block tool calls whose lora verdict is below SECURITY_DEFENCE_LEVEL.
# Calls at or above the level pass through. Example: "neutral" allows safe+neutral, blocks suspicious+unsafe.
SECURITY_DEFENCE_ENABLE=true          # true | false
SECURITY_DEFENCE_LEVEL="neutral"       # safe | neutral | suspicious | unsafe
SECURITY_DEFENCE_DEBUG=false           # true: keep <tool_call_security> in response; false: strip it
SECURITY_DEFENCE_MAX_RETRIES=10        # max base-model retries after a defence block
# Defence methods applied in order until one succeeds; if none does, the turn passes through
# undefended. Available: remove_trigger_words | fake_tool_response
DEFENCE_METHOD_LIST="remove_trigger_words,fake_tool_response"
# remove_trigger_words parameters for the DEFENCE_METHOD_LIST (blocked-call) path.
DEFENCE_REMOVE_TRIGGER_WORDS_MATCH_TOOL_CALL=true  # true: treat trigger words that belong to a tool call (name/args) as a false positive
DEFENCE_REMOVE_TRIGGER_WORDS_FUZZY_SEARCH=false     # true: allow fuzzy matching when locating trigger words
# true: for a "safe" verdict, validate its trigger words against the user messages (exact match, no fuzzy);
# if they are not from the user, run the defence methods instead of trusting the "safe" rating.
DEFENCE_SAFE_TOOLCALL=true
# Defence methods for the safe-verdict path (used only when DEFENCE_SAFE_TOOLCALL=true), applied in
# order until one succeeds; if none does, the call passes through. Same names as DEFENCE_METHOD_LIST.
DEFENCE_SAFE_METHOD_LIST="remove_trigger_words"
# remove_trigger_words parameters for the DEFENCE_SAFE_METHOD_LIST (safe-verdict) path.
DEFENCE_SAFE_REMOVE_TRIGGER_WORDS_MATCH_TOOL_CALL=false   # true: match-tool-call false-positive guard
DEFENCE_SAFE_REMOVE_TRIGGER_WORDS_FUZZY_SEARCH=false     # false: exact matching only on the safe path

# -----------------------------------------------------------------------
# Apply environment variable overrides
# If any of the Phase-1 LLM variables were set in the shell environment
# before running this script, they take priority over the defaults above.
# -----------------------------------------------------------------------
[ -n "$_ENV_LLM_SERVER_URL" ]             && LLM_SERVER_URL="$_ENV_LLM_SERVER_URL"
[ -n "$_ENV_LLM_MODEL_ID" ]               && LLM_MODEL_ID="$_ENV_LLM_MODEL_ID"
[ -n "$_ENV_LLM_SERVER_PROXY" ]           && LLM_SERVER_PROXY="$_ENV_LLM_SERVER_PROXY"
[ -n "$_ENV_LLM_SERVER_TOKEN_LIST" ]      && LLM_SERVER_TOKEN_LIST="$_ENV_LLM_SERVER_TOKEN_LIST"
[ -n "$_ENV_LLM_CONTEXT_WINDOW" ]         && LLM_CONTEXT_WINDOW="$_ENV_LLM_CONTEXT_WINDOW"
[ -n "$_ENV_LLM_API_CALL_INTERVAL" ]      && LLM_API_CALL_INTERVAL="$_ENV_LLM_API_CALL_INTERVAL"

# -----------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/defence-llm-server.py" \
    --llm-server-url       "${LLM_SERVER_URL}" \
    --llm-model-id         "${LLM_MODEL_ID}" \
    --llm-server-proxy      "${LLM_SERVER_PROXY}" \
    --llm-server-token-list "${LLM_SERVER_TOKEN_LIST}" \
    --llm-context-window    "${LLM_CONTEXT_WINDOW}" \
    --llm_api_call_interval "${LLM_API_CALL_INTERVAL}" \
    --secure-server-url    "${SECURE_SERVER_URL}" \
    --secure-model-id      "${SECURE_MODEL_ID}" \
    --host                 "${LISTEN_HOST}" \
    --port                 "${LISTEN_PORT}" \
    --base-model-path      "${BASE_MODEL_PATH}" \
    --sec_inference_max_tokens "${SEC_INFERENCE_MAX_TOKENS}" \
    --llm_inference_max_tokens "${LLM_INFERENCE_MAX_TOKENS}" \
    --timeout              "${REQUEST_TIMEOUT}" \
    --log-level            "${LOG_LEVEL}" \
    --log-file-name        "${LOG_FILE_NAME}" \
    --enable_thinking          "${ENABLE_THINKING}" \
    --phase2_enable            "${PHASE2_ENABLE}" \
    --phase2_tool_reason_retry_count "${PHASE2_TOOL_REASON_RETRY_COUNT}" \
    --output_raw_client_input  "${OUTPUT_RAW_CLIENT_INPUT}" \
    --security_defence_enable       "${SECURITY_DEFENCE_ENABLE}" \
    --security_defence_debug        "${SECURITY_DEFENCE_DEBUG}" \
    --security-defence-level        "${SECURITY_DEFENCE_LEVEL}" \
    --security-defence-max-retries  "${SECURITY_DEFENCE_MAX_RETRIES}" \
    --defence_method_list           "${DEFENCE_METHOD_LIST}" \
    --defence_remove_trigger_words_match_tool_call "${DEFENCE_REMOVE_TRIGGER_WORDS_MATCH_TOOL_CALL}" \
    --defence_remove_trigger_words_fuzzy_search "${DEFENCE_REMOVE_TRIGGER_WORDS_FUZZY_SEARCH}" \
    --defence_safe_toolcall "${DEFENCE_SAFE_TOOLCALL}" \
    --defence_safe_method_list "${DEFENCE_SAFE_METHOD_LIST}" \
    --defence_safe_remove_trigger_words_match_tool_call "${DEFENCE_SAFE_REMOVE_TRIGGER_WORDS_MATCH_TOOL_CALL}" \
    --defence_safe_remove_trigger_words_fuzzy_search "${DEFENCE_SAFE_REMOVE_TRIGGER_WORDS_FUZZY_SEARCH}"
