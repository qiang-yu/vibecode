#!/bin/bash
log_dir="logs/update"

mkdir -p $log_dir

model="gpt-4o-2024-08-06"
# model="claude-3-7-sonnet-20250219"
# model="facebook/Meta-SecAlign-70B" # vllm serve meta-llama/Llama-3.3-70B-Instruct   --tokenizer facebook/Meta-SecAlign-70B   --tensor-parallel-size 4   --enable-lora   --max-lora-rank 64   --lora-modules facebook/Meta-SecAlign-70B=facebook/Meta-SecAlign-70B   --gpu_memory_utilization 0.95
# model="claude-sonnet-4-20250514"
# model="gpt-4.1-2025-04-14"
# model="gemini-2.5-flash"

export SECAGENT_POLICY_MODEL="gpt-4o-2024-08-06"
# export SECAGENT_POLICY_MODEL="claude-sonnet-4-20250514"
# export SECAGENT_POLICY_MODEL="gpt-4.1-2025-04-14"
# export SECAGENT_POLICY_MODEL="gemini-2.5-flash"

# auto policies
export SECAGENT_UPDATE="True"
export SECAGENT_IGNORE_UPDATE_ERROR="True"
# export SECAGENT_ONLY_ALLOW_NARROW="True"

export COLUMNS=300

SECAGENT_SUITE=banking nohup python -m agentdojo.scripts.benchmark -s banking --model $model --logdir $log_dir > $log_dir/banking-no-attack.log 2>&1 &
SECAGENT_SUITE=slack nohup python -m agentdojo.scripts.benchmark -s slack --model $model --logdir $log_dir > $log_dir/slack-no-attack.log 2>&1 &
SECAGENT_SUITE=travel nohup python -m agentdojo.scripts.benchmark -s travel --model $model --logdir $log_dir > $log_dir/travel-no-attack.log 2>&1 &
SECAGENT_SUITE=workspace nohup python -m agentdojo.scripts.benchmark -s workspace --model $model --logdir $log_dir > $log_dir/workspace-no-attack.log 2>&1 &

SECAGENT_SUITE=banking nohup python -m agentdojo.scripts.benchmark -s banking --model $model --attack important_instructions --logdir $log_dir > $log_dir/banking-attack.log 2>&1 &
SECAGENT_SUITE=slack nohup python -m agentdojo.scripts.benchmark -s slack --model $model --attack important_instructions --logdir $log_dir > $log_dir/slack-attack.log 2>&1 &
SECAGENT_SUITE=travel nohup python -m agentdojo.scripts.benchmark -s travel --model $model --attack important_instructions --logdir $log_dir > $log_dir/travel-attack.log 2>&1 &
SECAGENT_SUITE=workspace nohup python -m agentdojo.scripts.benchmark -s workspace --model $model --attack important_instructions --logdir $log_dir > $log_dir/workspace-attack.log 2>&1 &

echo "started"
