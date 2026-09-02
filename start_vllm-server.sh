#!/usr/bin/env bash

# -----------------------------------------------------------------------
# Configurable parameters — edit these before starting the server
# -----------------------------------------------------------------------

VLLM_BASE_URL="http://localhost:19001/v1"
LISTEN_HOST="localhost"
LISTEN_PORT=29001

BASE_MODEL_PATH="/home/qiangyu/Models/Qwen/Qwen3-8B"
BASE_MODEL_ID="Qwen3Base"
LORA_MODEL_ID="lora-model"
MODEL_TYPE="Qwen3"               # Qwen3 | Llama3

MAX_TOKENS_SECURITY=4096          # max tokens for phase-2 lora security block
REQUEST_TIMEOUT=600              # HTTP request timeout in seconds
LOG_LEVEL="info"                 # debug | info | warning | error

OUTPUT_RAW_CLIENT_INPUT=true     # true: log raw client input (Qwen3 format) before stripping

ENABLE_THINKING=true              # true | false
STRIP_SECURITY_IN_HISTORY=true    # true | false
PHASE2_ENABLE=false                # true: run phase-2 security check; false: phase-1 only
PHASE1_THINK_RETRY_COUNT=0        # retry phase 1 N times when its think overruns max_tokens

# Security defence: block tool calls whose lora verdict is below SECURITY_DEFENCE_LEVEL.
# Calls at or above the level pass through. Example: "neutral" allows safe+neutral, blocks suspicious+unsafe.
SECURITY_DEFENCE_ENABLE=true          # true | false
SECURITY_DEFENCE_LEVEL="neutral"       # safe | neutral | suspicious | unsafe
SECURITY_DEFENCE_DEBUG=true           # true: keep <tool_call_security> in response; false: strip it
SECURITY_DEFENCE_MAX_RETRIES=1        # max base-model retries after a defence block

# -----------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/vllm-server.py" \
    --vllm-url             "${VLLM_BASE_URL}" \
    --host                 "${LISTEN_HOST}" \
    --port                 "${LISTEN_PORT}" \
    --base-model-path      "${BASE_MODEL_PATH}" \
    --base-model-id        "${BASE_MODEL_ID}" \
    --lora-model-id        "${LORA_MODEL_ID}" \
    --model-type           "${MODEL_TYPE}" \
    --max-tokens-security  "${MAX_TOKENS_SECURITY}" \
    --timeout              "${REQUEST_TIMEOUT}" \
    --log-level            "${LOG_LEVEL}" \
    --enable_thinking          "${ENABLE_THINKING}" \
    --phase2_enable            "${PHASE2_ENABLE}" \
    --phase1_think_retry_count "${PHASE1_THINK_RETRY_COUNT}" \
    --output_raw_client_input  "${OUTPUT_RAW_CLIENT_INPUT}" \
    --strip_security_in_history "${STRIP_SECURITY_IN_HISTORY}" \
    --security_defence_enable       "${SECURITY_DEFENCE_ENABLE}" \
    --security_defence_debug        "${SECURITY_DEFENCE_DEBUG}" \
    --security-defence-level        "${SECURITY_DEFENCE_LEVEL}" \
    --security-defence-max-retries  "${SECURITY_DEFENCE_MAX_RETRIES}"
