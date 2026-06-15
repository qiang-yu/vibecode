#!/bin/sh

model_id="/home/qiangyu/Models/Qwen/Qwen3-8B"

log_dir="./runs/Qwen3-8B"
mkdir -p ${log_dir}

export LOCAL_LLM_PORT=19000
# export OPENAI_COMPATIBLE_BASE_URL="http://localhost:19000/v1"
# export OPENAI_COMPATIBLE_API_KEY="NOKEY"


python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id ${model_id} --logdir ${log_dir}
# python -m agentdojo.scripts.benchmark --model openai-compatible --model-id ${model_id} --logdir ${log_dir}

