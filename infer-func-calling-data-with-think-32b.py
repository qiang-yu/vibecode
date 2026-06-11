###
# This script loads multiple instances of the Qwen3-32B model across several GPUs
# and processes ShareGPT-format conversations from a JSON file in parallel.
# For each data entry, it re-infers assistant responses with thinking (reasoning)
# content enabled. A dedicated writer process serializes results to a JSON Lines
# temp file, which is finally converted into a JSON array output.
###

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path


MODEL_PATH = "/home/qiangyu/Models/Qwen/Qwen3-32B"
INPUT_FILE = "func-calling/glaive-function-calling-5k-injected.json"
OUTPUT_FILE = "func-calling/glaive-function-calling-5k-injected-inference-32b.json"
TEMP_FILE = "func-calling/glaive-function-calling-5k-injected-inference-32b.jsonl"

# GPU list used for parallel processing. Each GPU loads its own model instance.
GPU_IDS = [4, 5, 6, 7]


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


def has_tool_call(response: str) -> bool:
    """Check if the response contains a tool call."""
    return "<tool_call>" in response


def process_single_data(data: dict, model, tokenizer) -> dict:
    """Process a single ShareGPT data entry."""
    conversations = data["conversations"]
    tools_str = data.get("tools")
    tools = json.loads(tools_str) if tools_str else None

    messages = []
    i = 0

    while i < len(conversations):
        item = conversations[i]
        role = item["from"]
        value = item["value"]

        if role == "system":
            # Ignore system messages in the conversation
            i += 1

        elif role == "human":
            messages.append({"role": "user", "content": value})

            # Generate response for this human message
            response = generate_response(messages, tools, model, tokenizer)

            # Replace the following gpt message
            if i + 1 < len(conversations) and conversations[i + 1]["from"] == "gpt":
                conversations[i + 1]["value"] = response
                messages.append({"role": "assistant", "content": response})

                # Check if tool call was made
                if has_tool_call(response):
                    i += 2  # Move past human and gpt

                    # Handle tool response and final answer
                    if i < len(conversations) and conversations[i]["from"] == "tool":
                        tool_value = conversations[i]["value"]
                        messages.append({"role": "tool", "content": tool_value})

                        # Generate final answer after tool
                        final_response = generate_response(
                            messages, tools, model, tokenizer
                        )

                        if (
                            i + 1 < len(conversations)
                            and conversations[i + 1]["from"] == "gpt"
                        ):
                            conversations[i + 1]["value"] = final_response
                            messages.append(
                                {"role": "assistant", "content": final_response}
                            )
                            i += 2  # Skip tool and final gpt
                        else:
                            # No gpt after tool, just skip tool
                            i += 1
                    else:
                        # Expected tool but not found
                        i += 1
                else:
                    i += 2  # Skip human and gpt
            else:
                # No gpt after human, insert a new gpt entry
                conversations.insert(i + 1, {"from": "gpt", "value": response})
                messages.append({"role": "assistant", "content": response})
                i += 2  # Move past human and the newly inserted gpt

        elif role == "tool":
            # Unhandled tool (shouldn't normally reach here)
            messages.append({"role": "tool", "content": value})
            i += 1

        elif role == "gpt":
            # Unhandled gpt (shouldn't normally reach here)
            i += 1

    return data


def worker(gpu_id, task_queue, result_queue):
    """
    Worker process that binds to a single GPU, loads its own model instance,
    consumes tasks from the task queue, and pushes results to the result queue.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[GPU {gpu_id}] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Ensure pad_token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[GPU {gpu_id}] Model loaded successfully.")

    while True:
        task = task_queue.get()
        if task is None:
            break
        idx, total, data = task
        try:
            processed_data = process_single_data(data, model, tokenizer)
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
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
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
    task_queue = mp.Queue(maxsize=len(GPU_IDS) * 2)
    result_queue = mp.Queue()

    # Start one worker process per GPU
    workers = []
    for gpu_id in GPU_IDS:
        p = mp.Process(target=worker, args=(gpu_id, task_queue, result_queue))
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

    # Signal workers to exit after they finish current tasks
    for _ in GPU_IDS:
        task_queue.put(None)

    # Wait for all workers to finish
    for p in workers:
        p.join()

    # Signal writer to exit and wait for it
    result_queue.put(None)
    writer_p.join()

    # Convert temp file to final JSON array
    print(f"\nConverting temp file to final JSON array: {OUTPUT_FILE}")
    convert_jsonl_to_json_array(TEMP_FILE, OUTPUT_FILE)
    print(f"Done. Total processed: {total}")


if __name__ == "__main__":
    main()
