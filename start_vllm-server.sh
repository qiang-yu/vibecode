#!/usr/bin/env bash

# -----------------------------------------------------------------------
# Configurable parameters — edit these before starting the server
# -----------------------------------------------------------------------

VLLM_BASE_URL="http://localhost:19006/v1"
LISTEN_HOST="localhost"
LISTEN_PORT=29006

BASE_MODEL_PATH="/home/qiangyu/Models/Qwen/Qwen3-8B"
BASE_MODEL_ID="Qwen3Base"
LORA_MODEL_ID="lora-model"
MODEL_TYPE="Qwen3"               # Qwen3 | Llama3

MAX_TOKENS_SECURITY=512          # max tokens for phase-2 lora security block
REQUEST_TIMEOUT=300              # HTTP request timeout in seconds
LOG_LEVEL="info"                 # debug | info | warning | error

# Set to "--no-enable-thinking" to disable Qwen3 thinking mode, or "" to keep it enabled.
NO_ENABLE_THINKING=""

# Set to "--no-strip-security-in-history" to keep <tool_call_security> in history, or "" to strip it.
NO_STRIP_SECURITY=""

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
    ${NO_ENABLE_THINKING} \
    ${NO_STRIP_SECURITY}
