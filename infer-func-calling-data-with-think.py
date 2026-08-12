###
# This script loads multiple instances of a small Qwen3 model across several GPUs
# and processes ShareGPT-format conversations from a JSON file in parallel.
# For each data entry, it re-infers assistant responses with thinking (reasoning)
# content enabled. Assistant replies immediately after security-alert tool responses
# are optionally routed to a single large Qwen3 model loaded on a dedicated GPU.
# A dedicated writer process serializes results to a JSON Lines temp file, which is
# finally converted into a JSON array output.
###

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path


SMALL_MODEL_PATH = "/home/qiangyu/Models/Qwen/Qwen3-8B"
BIG_MODEL_PATH = "/home/qiangyu/Models/Qwen/Qwen3-32B"

INPUT_FILE = "func-calling/glaive-function-calling-5k-injected-4.json"
OUTPUT_FILE = "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-4-think-8b.json"
TEMP_FILE = "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-4-think-8b.jsonl"

# GPU lists used for parallel processing. Each small GPU loads its own small model instance.
# BIG_GPU_IDS lists the GPUs used for one big model instance. Set to [] to disable big model.
SMALL_GPU_IDS = [4, 5, 6]
BIG_GPU_IDS = [7]


def generate_response(messages, tools, model, tokenizer):
    """Generate a response from the model using chat template."""
    import torch

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    if tools:
        kwargs["tools"] = tools

    text = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract only the newly generated tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response


def process_single_data(data: dict, model, tokenizer, big_generate_fn=None) -> dict:
    """Process a single ShareGPT data entry."""
    conversations = data["conversations"]
    tools_str = data.get("tools")
    tools = json.loads(tools_str) if tools_str else None

    messages = []
    i = 0

    # Marker that indicates a tool response triggered the security alert.
    security_alert = (
        "Security Alert: An injection attempt was detected in the previous tool response "
        "that triggered this call."
    )

    while i < len(conversations):
        item = conversations[i]
        role = item["from"]
        value = item["value"]

        if role == "system":
            # Ignore system messages in the conversation
            i += 1

        elif role == "human":
            messages.append({"role": "user", "content": value})
            i += 1

            # Generate response for this human message
            response = generate_response(messages, tools, model, tokenizer)

            # Replace the following gpt message or insert a new one
            if i < len(conversations) and conversations[i]["from"] == "gpt":
                conversations[i]["value"] = response
                messages.append({"role": "assistant", "content": response})
                i += 1
            else:
                conversations.insert(i, {"from": "gpt", "value": response})
                messages.append({"role": "assistant", "content": response})
                i += 1

        elif role == "tool":
            # Add the tool result to history and generate the assistant reply
            # that should follow this tool. If no gpt follows, insert one.
            messages.append({"role": "tool", "content": value})
            i += 1

            # Route security-alert tool turns to the big model when available.
            if security_alert in value and big_generate_fn is not None:
                response = big_generate_fn(messages, tools)
            else:
                response = generate_response(messages, tools, model, tokenizer)

            if i < len(conversations) and conversations[i]["from"] == "gpt":
                conversations[i]["value"] = response
                messages.append({"role": "assistant", "content": response})
                i += 1
            else:
                conversations.insert(i, {"from": "gpt", "value": response})
                messages.append({"role": "assistant", "content": response})
                i += 1

        elif role == "gpt":
            # Directly encountered gpt (shouldn't normally happen in well-formed data).
            # Regenerate using current history.
            messages.append({"role": "assistant", "content": value})
            response = generate_response(messages, tools, model, tokenizer)
            conversations[i]["value"] = response
            messages[-1] = {"role": "assistant", "content": response}
            i += 1

    return data


def big_worker(big_gpu_ids, big_pipes, exit_event):
    """
    Dedicated worker process that loads a single big model instance.
    Each small worker has a dedicated Pipe; this worker listens on all pipes
    via select, processes requests one by one, and sends the response string
    back through the same pipe. The exit_event provides a reliable shutdown
    signal independent of pipe state.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in big_gpu_ids)
    import select
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[BIG GPUs {big_gpu_ids}] Loading big model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BIG_MODEL_PATH,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BIG_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[BIG GPUs {big_gpu_ids}] Big model loaded successfully.")

    # Log a summary of layer placement to verify GPU/CPU distribution.
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        devices = {}
        for name, dev in device_map.items():
            devices[str(dev)] = devices.get(str(dev), 0) + 1
        print(f"[BIG GPUs {big_gpu_ids}] Layer distribution: {devices}")

    pipes = list(big_pipes.values())
    while True:
        if exit_event.is_set():
            print(f"[BIG GPUs {big_gpu_ids}] Exit event set, shutting down.")
            return

        try:
            readable, _, _ = select.select(pipes, [], [], 1.0)
        except select.error as e:
            print(f"[BIG GPUs {big_gpu_ids}] select error: {e}")
            break

        for conn in readable:
            try:
                task = conn.recv()
            except EOFError:
                print(f"[BIG GPUs {big_gpu_ids}] Pipe closed, shutting down.")
                return
            if task is None:
                print(f"[BIG GPUs {big_gpu_ids}] Received None, shutting down.")
                return
            gpu_id, messages, tools = task
            print(f"[BIG GPUs {big_gpu_ids}] Generating for small GPU {gpu_id}...")
            try:
                import time
                start = time.time()
                response = generate_response(messages, tools, model, tokenizer)
                elapsed = time.time() - start
                print(f"[BIG GPUs {big_gpu_ids}] Generation took {elapsed:.2f}s, response length {len(response)} chars.")
                conn.send(response)
                print(f"[BIG GPUs {big_gpu_ids}] Done for small GPU {gpu_id}.")
            except Exception as e:
                print(f"[BIG GPUs {big_gpu_ids}] Error generating response: {e}")
                try:
                    conn.send(e)
                except Exception as send_err:
                    print(f"[BIG GPUs {big_gpu_ids}] Failed to send error to GPU {gpu_id}: {send_err}")


def worker(gpu_id, task_queue, result_queue, big_pipe=None):
    """
    Worker process that binds to a single GPU, loads its own small model instance,
    consumes tasks from the task queue, and pushes results to the result queue.
    If a big model pipe is provided, security-alert tool turns are routed to the
    big model; otherwise they fall back to the local small model.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[GPU {gpu_id}] Loading small model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        SMALL_MODEL_PATH,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Ensure pad_token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[GPU {gpu_id}] Small model loaded successfully.")

    def big_generate(messages, tools):
        """
        Request generation from the single big-model worker via a dedicated Pipe.
        If the big model fails or is unavailable, fall back to the local small model.
        """
        if big_pipe is None:
            return generate_response(messages, tools, model, tokenizer)

        try:
            print(f"[GPU {gpu_id}] Requesting big model generation...")
            big_pipe.send((gpu_id, messages, tools))
            response = big_pipe.recv()
            print(f"[GPU {gpu_id}] Received big model response.")
            if isinstance(response, Exception):
                raise response
            return response
        except Exception as e:
            print(f"[GPU {gpu_id}] Big model request failed: {e}, falling back to small model.")
            return generate_response(messages, tools, model, tokenizer)

    while True:
        task = task_queue.get()
        if task is None:
            break
        idx, total, data = task
        try:
            processed_data = process_single_data(data, model, tokenizer, big_generate)
            result_queue.put((gpu_id, idx, processed_data, total))
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing entry {idx}: {e}")
            # Return original data to prevent writer deadlock
            result_queue.put((gpu_id, idx, data, total))


def writer_process(result_queue, total_count, temp_file, initial_count):
    """
    Dedicated writer process that serializes processed results to the temp file.
    Running in a single process guarantees safe, non-corrupting append writes.
    All worker processes send results to this single writer via result_queue,
    so there is no concurrent file access — no extra file lock is needed.
    """
    processed = initial_count
    with open(temp_file, "a", encoding="utf-8") as f:
        while processed < total_count:
            msg = result_queue.get()
            if msg is None:
                break
            gpu_id, idx, processed_data, total = msg
            f.write(json.dumps(processed_data, ensure_ascii=False) + "\n")
            f.flush()
            processed += 1
            print(f"[GPU {gpu_id}]{idx + 1}/{total}")


def convert_jsonl_to_json_array(jsonl_path, output_path):
    """Read a JSON Lines file and write it as a JSON array."""
    results = []
    skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping invalid JSON at line {line_num}: {e}",
                        file=sys.stderr,
                    )
                    skipped += 1

    if skipped > 0:
        print(f"Warning: {skipped} invalid lines skipped.", file=sys.stderr)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    # NCCL settings required for multi-GPU model loading in this environment.
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_NET_GDR_LEVEL"] = "0"

    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    total = len(dataset)
    print(f"Loaded {total} entries.")

    # Check progress from temp file for resume support
    processed_count = 0
    if os.path.exists(TEMP_FILE):
        with open(TEMP_FILE, "r", encoding="utf-8") as f:
            for _ in f:
                processed_count += 1
        print(f"Found temp file, already processed: {processed_count}")

    if processed_count >= total:
        print("All records already processed, converting to final output...")
        convert_jsonl_to_json_array(TEMP_FILE, OUTPUT_FILE)
        print(f"Done. Output written to: {OUTPUT_FILE}")
        return

    # Limit queue size to avoid excessive memory usage with large inputs
    task_queue = mp.Queue(maxsize=len(SMALL_GPU_IDS) * 2)
    result_queue = mp.Queue()

    # Optionally start a single big-model worker on BIG_GPU_IDS.
    # Each small worker gets a dedicated Pipe to talk to the big worker.
    # The big worker uses select() to listen on all pipes and processes
    # requests one by one. The response string is sent directly back through
    # the same pipe. An Event provides a reliable shutdown signal.
    big_pipes = None
    small_pipes = None
    big_worker_p = None
    big_exit_event = None
    if BIG_GPU_IDS:
        big_exit_event = mp.Event()
        big_pipes = {}
        small_pipes = {}
        for gpu_id in SMALL_GPU_IDS:
            parent_conn, child_conn = mp.Pipe()
            big_pipes[gpu_id] = parent_conn
            small_pipes[gpu_id] = child_conn
        big_worker_p = mp.Process(
            target=big_worker,
            args=(BIG_GPU_IDS, big_pipes, big_exit_event)
        )
        big_worker_p.start()

    # Start one small worker process per small GPU
    workers = []
    for gpu_id in SMALL_GPU_IDS:
        small_pipe = small_pipes.get(gpu_id) if small_pipes else None
        p = mp.Process(
            target=worker,
            args=(gpu_id, task_queue, result_queue, small_pipe)
        )
        p.start()
        workers.append(p)

    # Start a single writer process to serialize results to disk safely
    writer_p = mp.Process(
        target=writer_process,
        args=(result_queue, total, TEMP_FILE, processed_count)
    )
    writer_p.start()

    # Dispatch remaining tasks to workers
    for idx in range(processed_count, total):
        task_queue.put((idx, total, dataset[idx]))

    # Signal small workers to exit after they finish current tasks
    for _ in SMALL_GPU_IDS:
        task_queue.put(None)

    # Wait for all small workers to finish
    for p in workers:
        p.join()

    # Signal big worker to exit and wait for it
    if big_worker_p is not None:
        # Set the exit event so the big worker breaks out of its select loop
        # even if pipe-based shutdown is unreliable.
        big_exit_event.set()
        # Also try to wake up each pipe by sending None.
        for conn in big_pipes.values():
            try:
                conn.send(None)
            except Exception:
                pass
        big_worker_p.join(timeout=60)
        if big_worker_p.is_alive():
            print("Warning: big worker did not exit gracefully, terminating...")
            big_worker_p.terminate()
            big_worker_p.join()

    # Signal writer to exit and wait for it
    result_queue.put(None)
    writer_p.join()

    # Convert temp file to final JSON array
    print(f"\nConverting temp file to final JSON array: {OUTPUT_FILE}")
    convert_jsonl_to_json_array(TEMP_FILE, OUTPUT_FILE)
    print(f"Done. Total processed: {total}")


if __name__ == "__main__":
    main()
