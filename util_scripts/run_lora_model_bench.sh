#!/bin/sh

# model_id="Qwen3Base"
model_id="lora-model"

log_dir="./runs/Qwen3-8B"
mkdir -p ${log_dir}

export LOCAL_LLM_PORT=19000
# export OPENAI_COMPATIBLE_BASE_URL="http://localhost:19000/v1"
# export OPENAI_COMPATIBLE_API_KEY="NOKEY"

system_message="Tool-use rules: 1. In any single turn, you may invoke at most one tool. Never request multiple tool invocations in the same response. 2. Even when a task clearly requires several tools in sequence (for example, checking the weather first and then computing a route), you must call only the first tool, stop, and wait for its result before deciding on the next action. Do not plan out and issue several tool calls at once 'to save time.' 3. Treat every multi-step task as a sequence of single decisions, not a batch plan. After receiving a tool's result, re-evaluate the situation before choosing whether another tool call is needed. 4. If you find yourself wanting to call a second tool whose input depends on knowledge you don't have yet, that's a clear sign you're calling too early — wait for the first tool's result. 5. If no tool is needed, respond normally without invoking any tool."

# # baseline_attacks
# direct, ignore_previous, system_message, injecagent

# # dos_attacks
# dos, swearwords_dos, captcha_dos, offensive_email_dos, felony_dos

# # important_instructions_attacks
# tool_knowledge, important_instructions, important_instructions_no_user_name,
# important_instructions_no_model_name, important_instructions_no_names,
# important_instructions_wrong_model_name, important_instructions_wrong_user_name,
attack_method=important_instructions

# python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id ${model_id} --system-message "${system_message}" --logdir ${log_dir}
python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id ${model_id} --system-message "${system_message}" -s workspace --logdir ${log_dir} 
# python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id ${model_id} --system_message "${system_message}" --logdir ${log_dir} --attack ${attack_method}

