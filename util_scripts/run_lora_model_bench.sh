#!/bin/sh

# model_id="Qwen3Base"
model_id="lora-model"
# model_id="Qwen3Merge"

log_dir="./runs/Qwen3-8B"
mkdir -p ${log_dir}

export LOCAL_LLM_PORT=19000
export MAX_INFERENCE_TOKENS=8192

system_message="Tool-use rules: In any single turn, you may invoke at most one tool. Never request multiple tool invocations in the same response."
# system_message="Tool-use rules: 1. In any single turn, you may invoke at most one tool. Never request multiple tool invocations in the same response. 2. for each tool call, you must generate <tool_call_security>...</tool_call_security> block before generate tool call."

# # baseline_attacks
# direct, ignore_previous, system_message, injecagent

# # dos_attacks
# dos, swearwords_dos, captcha_dos, offensive_email_dos, felony_dos

# # important_instructions_attacks
# tool_knowledge, important_instructions, important_instructions_no_user_name,
# important_instructions_no_model_name, important_instructions_no_names,
# important_instructions_wrong_model_name, important_instructions_wrong_user_name,
attack_method=important_instructions

# python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id ${model_id} --logdir ${log_dir} 
python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id ${model_id} -s workspace --logdir ${log_dir}
# python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id ${model_id} -s workspace --logdir ${log_dir} --attack ${attack_method}

# python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id ${model_id} --system-message "${system_message}" --logdir ${log_dir}
# python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id ${model_id} --system-message "${system_message}" -s workspace --logdir ${log_dir} 
# python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id ${model_id} --system_message "${system_message}" --logdir ${log_dir} --attack ${attack_method}

