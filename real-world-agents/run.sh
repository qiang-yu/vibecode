#!/bin/bash
export AGENT_TYPE="openai"
# export AGENT_TYPE="langchain"
# export AGENT_TYPE="openhands"
# export AGENT_TYPE="autogen"
# log_dir="$AGENT_TYPE/plain"
log_dir="$AGENT_TYPE/update"
mkdir -p $log_dir

export ENABLE_SECAGENT="True"

export SECAGENT_POLICY_MODEL="gpt-4o-2024-08-06"
export SECAGENT_UPDATE="True"
export SECAGENT_IGNORE_UPDATE_ERROR="True"

export COLUMNS=300

SECAGENT_SUITE=banking nohup python -u run.py --agent-type $AGENT_TYPE --suite banking >> $log_dir/banking.log 2>&1 &
SECAGENT_SUITE=slack nohup python -u run.py --agent-type $AGENT_TYPE --suite slack >> $log_dir/slack.log 2>&1 &
SECAGENT_SUITE=travel nohup python -u run.py --agent-type $AGENT_TYPE --suite travel >> $log_dir/travel.log 2>&1 &
SECAGENT_SUITE=workspace nohup python -u run.py --agent-type $AGENT_TYPE --suite workspace >> $log_dir/workspace.log 2>&1 &

echo "started"
