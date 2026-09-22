#!/usr/bin/env bash

# -----------------------------------------------------------------------
# Configurable parameters — edit these before starting the server
# -----------------------------------------------------------------------

VLLM_BASE_URL="http://localhost:19000/v1"
LISTEN_HOST="localhost"
LISTEN_PORT=29000

BASE_MODEL_PATH="/home/qiangyu/Models/Qwen/Qwen3-8B"
BASE_MODEL_ID="Qwen3Base"
LORA_MODEL_ID="lora-model"

MAX_TOKENS_SECURITY=4096          # max tokens for phase-2 lora security block
REQUEST_TIMEOUT=600              # HTTP request timeout in seconds
LOG_LEVEL="info"                 # debug | info | warning | error
LOG_FILE_NAME="defence-llm-server.log"         # log file base name; runtime prepends YYYYMMDD_

OUTPUT_RAW_CLIENT_INPUT=false    # true: log raw client input (Qwen3 format) before stripping

ENABLE_THINKING=true              # true | false
STRIP_SECURITY_IN_HISTORY=true    # true | false
PHASE2_ENABLE=true                # true: run phase-2 security check; false: phase-1 only
PHASE1_THINK_RETRY_COUNT=0        # retry phase 1 N times when its think overruns max_tokens
PHASE2_TOOL_REASON_RETRY_COUNT=1  # retry phase 2 N times when its security block overruns max_tokens

# Security defence: block tool calls whose lora verdict is below SECURITY_DEFENCE_LEVEL.
# Calls at or above the level pass through. Example: "neutral" allows safe+neutral, blocks suspicious+unsafe.
SECURITY_DEFENCE_ENABLE=true          # true | false
SECURITY_DEFENCE_LEVEL="neutral"       # safe | neutral | suspicious | unsafe
SECURITY_DEFENCE_DEBUG=true           # true: keep <tool_call_security> in response; false: strip it
SECURITY_DEFENCE_MAX_RETRIES=10        # max base-model retries after a defence block
# Defence methods applied in order until one succeeds; if none does, the turn passes through
# undefended. Available: remove_trigger_words | fake_tool_response
DEFENCE_METHOD_LIST="remove_trigger_words,fake_tool_response"
# remove_trigger_words parameters for the DEFENCE_METHOD_LIST (blocked-call) path.
DEFENCE_REMOVE_TRIGGER_WORDS_MATCH_TOOL_CALL=true  # true: treat trigger words that belong to a tool call (name/args) as a false positive
DEFENCE_REMOVE_TRIGGER_WORDS_FUZZY_SEARCH=true     # true: allow fuzzy matching when locating trigger words
# true: for a "safe" verdict, validate its trigger words against the user messages (exact match, no fuzzy);
# if they are not from the user, run the defence methods instead of trusting the "safe" rating.
DEFENCE_SAFE_TOOLCALL=true
# Defence methods for the safe-verdict path (used only when DEFENCE_SAFE_TOOLCALL=true), applied in
# order until one succeeds; if none does, the call passes through. Same names as DEFENCE_METHOD_LIST.
DEFENCE_SAFE_METHOD_LIST="remove_trigger_words"
# remove_trigger_words parameters for the DEFENCE_SAFE_METHOD_LIST (safe-verdict) path.
DEFENCE_SAFE_REMOVE_TRIGGER_WORDS_MATCH_TOOL_CALL=true   # true: match-tool-call false-positive guard
DEFENCE_SAFE_REMOVE_TRIGGER_WORDS_FUZZY_SEARCH=false     # false: exact matching only on the safe path

# -----------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/defence-llm-server.py" \
    --vllm-url             "${VLLM_BASE_URL}" \
    --host                 "${LISTEN_HOST}" \
    --port                 "${LISTEN_PORT}" \
    --base-model-path      "${BASE_MODEL_PATH}" \
    --base-model-id        "${BASE_MODEL_ID}" \
    --lora-model-id        "${LORA_MODEL_ID}" \
    --max-tokens-security  "${MAX_TOKENS_SECURITY}" \
    --timeout              "${REQUEST_TIMEOUT}" \
    --log-level            "${LOG_LEVEL}" \
    --log-file-name        "${LOG_FILE_NAME}" \
    --enable_thinking          "${ENABLE_THINKING}" \
    --phase2_enable            "${PHASE2_ENABLE}" \
    --phase1_think_retry_count      "${PHASE1_THINK_RETRY_COUNT}" \
    --phase2_tool_reason_retry_count "${PHASE2_TOOL_REASON_RETRY_COUNT}" \
    --output_raw_client_input  "${OUTPUT_RAW_CLIENT_INPUT}" \
    --strip_security_in_history "${STRIP_SECURITY_IN_HISTORY}" \
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
