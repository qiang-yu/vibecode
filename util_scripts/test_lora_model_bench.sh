#!/bin/sh

# model_id="Qwen3Base"
model_id="lora-model"

log_dir="./test/Qwen3-8B"
rm -rf ./test
mkdir -p ${log_dir}

export LOCAL_LLM_PORT=19000
# export OPENAI_COMPATIBLE_BASE_URL="http://localhost:19000/v1"
# export OPENAI_COMPATIBLE_API_KEY="NOKEY"


python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id ${model_id} --logdir ${log_dir} -s workspace -ut user_task_0 
# python -m agentdojo.scripts.benchmark --model openai-compatible --model-id ${model_id} --logdir ${log_dir}

