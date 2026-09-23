#!/bin/sh

# Local VLLM
export LLM_SERVER_URL="http://localhost:19000/v1"
export LLM_MODEL_ID="Qwen3Base"
unset LLM_SERVER_PROXY
# export LLM_SERVER_PROXY=""
export LLM_CONTEXT_WINDOW="32768"
export LLM_API_CALL_INTERVAL="0.0"
export LLM_SERVER_TOKEN_LIST="NOKEY"

