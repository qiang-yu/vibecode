#!/bin/bash
# Batch version of run_vllm_test.sh:
# for every defense in defense_method_list, run the attack benchmark first,
# then the utility (no attack) benchmark, each with the same parallel-suite logic.

model_id="Llama3Base"
# model_id="Qwen3Base"

# full control of the vllm (OpenAI-compatible) server
export LOCAL_LLM_URL="http://localhost:19000/v1"
export LOCAL_LLM_MODEL_ID=${model_id}   # the command line option --model-id takes precedence over it
export LOCAL_LLM_KEY="EMPTY"            # sent as the api key; only checked if vllm serve was started with --api-key

export SECAGENT_POLICY_MODEL=${model_id}
export MAX_INFERENCE_TOKENS=8192

# device for the HF-based PI detectors (default: cuda:0 if available, else cpu), e.g. cuda:7 or cpu
detector_gpus=(0 1 2 3)
export DETECTOR_DEVICE="cuda:0" # "cpu"

# make python use UTF-8, sometimes json decode error because LLM output utf-8 characters
export PYTHONUTF8=1
export LANG=C.UTF-8 LC_ALL=C.UTF-8

# PI detector models are loaded from $DETECTOR_MODEL_DIR/<org>/<name>, no internet needed
export DETECTOR_MODEL_DIR=/home/qiangyu/HuggingFace
export HF_HUB_OFFLINE=1

# # baseline_attacks
# direct, ignore_previous, system_message, injecagent

# # dos_attacks
# dos, swearwords_dos, captcha_dos, offensive_email_dos, felony_dos

# # important_instructions_attacks
# tool_knowledge, important_instructions, important_instructions_no_user_name,
# important_instructions_no_model_name, important_instructions_no_names,
# important_instructions_wrong_model_name, important_instructions_wrong_user_name,
attack_method=important_instructions

# drift, progent, camel, tool_filter
# transformers_pi_detector, piguard_detector, prompt_guard_2_detector
# spotlighting_with_delimiting, repeat_user_prompt
defense_method_list=tool_filter,spotlighting_with_delimiting,repeat_user_prompt

set -o pipefail
set -m   # Job control: each background pipeline gets its own process group

# Maximum number of concurrent tasks
run_in_parallel=4
suites=(workspace banking travel slack github shopping dailylife)

if [ "${run_in_parallel}" -gt "${#detector_gpus[@]}" ]; then
    echo "[!] run_in_parallel=${run_in_parallel} > ${#detector_gpus[@]} GPUs, capping to ${#detector_gpus[@]}"
    run_in_parallel=${#detector_gpus[@]}
fi

# Cleanup: kill every background process group on Ctrl+C / SIGTERM (aborts the whole batch)
cleanup() {
    trap '' INT TERM            # Ignore further signals during cleanup
    echo -e "\n[!] Interrupted. Killing all background process groups..."
    # Send SIGTERM to each job's process group (PGID == PID of the group leader)
    for pgid in $(jobs -p); do
        kill -TERM -- "-${pgid}" 2>/dev/null
    done
    sleep 2
    # Force-kill anything that is still alive
    for pgid in $(jobs -p); do
        kill -KILL -- "-${pgid}" 2>/dev/null
    done
    wait 2>/dev/null
    exit 130
}
trap cleanup INT TERM

# run_benchmark <log_dir> <cmd...>
# Same logic as run_vllm_test.sh: parallel suites (one GPU per job),
# move error results out, then a final full run.
run_benchmark() {
    local log_dir="$1"
    shift
    local cmd=("$@")
    local slot_pid=()   # slot_pid[k] = PID of the job currently running on detector_gpus[k]
    local s k pid free_slot gpu

    mkdir -p "${log_dir}"
    echo "================================================================"
    echo "[*] log_dir: ${log_dir}"
    echo "[*] cmd    : ${cmd[*]}"
    echo "================================================================"
    sleep 5s

    # 1. Run the suites in parallel, one GPU per job.
    #    Each slot k owns GPU ${detector_gpus[k]}; a new suite only starts on a slot
    #    whose previous job has exited, so two jobs never share one GPU (avoids OOM).
    for s in "${suites[@]}"; do
        # Find a free slot: never used, or its job is no longer alive
        while true; do
            free_slot=-1
            for ((k = 0; k < run_in_parallel; k++)); do
                pid=${slot_pid[$k]:-}
                if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
                    free_slot=$k
                    break
                fi
            done
            [ "${free_slot}" -ge 0 ] && break
            # All GPUs busy: block until some job finishes (requires bash >= 4.3)
            wait -n 2>/dev/null || sleep 1
        done

        gpu=${detector_gpus[$free_slot]}
        echo "[*] start suite ${s} on cuda:${gpu} (slot ${free_slot})"

        # Run in background, printing to screen and saving to a per-suite log.
        # Wrapped in a subshell so $! is the PID of the whole pipeline (python + tee),
        # i.e. the GPU is considered free only after python has really exited.
        ( DETECTOR_DEVICE="cuda:${gpu}" "${cmd[@]}" -s "$s" 2>&1 | tee "${log_dir}/${s}.log" ) &
        slot_pid[$free_slot]=$!
    done

    # 2. Wait for all remaining background jobs to finish
    wait

    # 3. move all error task out and run it again
    python move_error_results.py "${log_dir}" --move

    # 4. Final full run, printing to screen and saving to all.log
    "${cmd[@]}" 2>&1 | tee "${log_dir}/all.log"
}

IFS=',' read -r -a defense_methods <<< "${defense_method_list}"

for defense_method in "${defense_methods[@]}"; do
    defense_method="${defense_method// /}"   # tolerate spaces after commas
    [ -z "${defense_method}" ] && continue
    echo -e "\n######## defense: ${defense_method} ########"

    # attack run
    log_dir="A1_Attack_${model_id}_${defense_method}_AgentDyn_01"
    cmd_attack=(python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id "${model_id}" --logdir "${log_dir}" --attack "${attack_method}"  --defense "${defense_method}")
    run_benchmark "${log_dir}" "${cmd_attack[@]}"

    # utility run (no attack)
    log_dir="A1_Utility_${model_id}_${defense_method}_AgentDyn_01"
    cmd_utility=(python -m agentdojo.scripts.benchmark --model VLLM_PARSED --model-id "${model_id}"  --logdir "${log_dir}" --defense "${defense_method}")
    run_benchmark "${log_dir}" "${cmd_utility[@]}"
done

echo "[*] all defenses done: ${defense_method_list}"
