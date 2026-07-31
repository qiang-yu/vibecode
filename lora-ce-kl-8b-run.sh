#!/bin/sh
###
# Launch multi-GPU LoRA SFT with torchrun (one process per GPU).
#
# DeepSpeed / multi-GPU training REQUIRES a distributed launcher. Running
# "python lora-ce-kl-8b.py" directly on several visible GPUs falls back to
# nn.DataParallel, which conflicts with DeepSpeed and fails with device errors.
#
# CUDA_VISIBLE_DEVICES selects the physical GPUs; --nproc_per_node must equal the
# number of GPUs listed. The script detects the launcher and will NOT override
# CUDA_VISIBLE_DEVICES, so each process is correctly pinned to one GPU. This is
# the ONLY place that decides which GPUs are used; the visible_gpus value in
# lora-ce-kl-8b-config.yaml is a fallback that only applies to a plain single-process run.
###

# Physical GPUs to use (must match nproc_per_node below).
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Number of processes = number of GPUs above.
NPROC=4

# ---------------------------------------------------------------------------
# NCCL settings for servers WITHOUT NVLink and WITHOUT PCIe P2P support.
# Peer-to-peer and InfiniBand must be disabled or NCCL hangs during init /
# all-reduce (same symptom seen with vLLM on this machine). Communication then
# falls back to shared memory / sockets, which is correct here.
# ---------------------------------------------------------------------------
export NCCL_P2P_DISABLE=1        # no PCIe peer-to-peer between GPUs
export NCCL_IB_DISABLE=1         # no InfiniBand
export NCCL_NET_GDR_LEVEL=0      # disable GPUDirect RDMA
export NCCL_DEBUG=INFO           # set to INFO for verbose NCCL diagnostics, default WARN

torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=29501 \
    lora-ce-kl-8b.py \
    --config_file lora-ce-kl-8b-config.yaml \
    --per_device_max_memory_gb 78.0
