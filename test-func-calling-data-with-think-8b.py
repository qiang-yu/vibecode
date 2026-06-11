###
# This script loads the Qwen3-8B model with thinking mode enabled
# and re-infers function calling training data to include thinking
# (reasoning) content. It processes ShareGPT-format conversations,
# replacing all assistant responses with model-generated outputs
# that include think tags.
###

import json
import os
import sys
from pathlib import Path

# Use GPU 7 only. Must be set before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str):
    """Load the Qwen3-8B model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Ensure pad_token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def generate_response(messages, tools, model, tokenizer):
    """Generate a response from the model using chat template."""
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
            messages.append({"role": "tool", "content": value})

            # If the tool response contains <tool_response> and the next
            # message is not from gpt, generate a gpt response and insert it.
            if "<tool_response>" in value:
                if i + 1 < len(conversations) and conversations[i + 1]["from"] == "gpt":
                    # Next is already gpt, just move on
                    i += 1
                else:
                    # No gpt after tool, generate one
                    response = generate_response(messages, tools, model, tokenizer)
                    conversations.insert(i + 1, {"from": "gpt", "value": response})
                    messages.append({"role": "assistant", "content": response})
                    i += 2  # Move past tool and the newly inserted gpt
            else:
                i += 1

        elif role == "gpt":
            # Unhandled gpt (shouldn't normally reach here)
            i += 1

    return data


def main():
    input_path = Path("func-calling/test_data.json")
    output_path = Path("func-calling/test_result.json")
    model_path = "/home/qiangyu/Models/Qwen/Qwen3-32B"

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading model...")
    model, tokenizer = load_model(model_path)
    print("Model loaded.")

    print(f"Loading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} entries.")

    print(f"Writing results to {output_path} (append mode)...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for idx, data in enumerate(dataset):
            print(f"Processing entry {idx + 1}/{len(dataset)}...")
            processed = process_single_data(data, model, tokenizer)
            if idx > 0:
                f.write(",\n")
            json.dump(processed, f, ensure_ascii=False, indent=2)
            f.flush()
        f.write("\n]\n")

    print("Done!")


if __name__ == "__main__":
    main()
