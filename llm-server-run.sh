#!/bin/bash
###
# Launch script for llm-server.py.
# Activates the Vllm8 conda environment and starts the server.
###

set -euo pipefail

# PYTHON=/home/qiangyu/anaconda3/envs/Vllm8/bin/python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=3

exec python llm-server.py --config "$SCRIPT_DIR/llm-server-config.yaml" "$@"
