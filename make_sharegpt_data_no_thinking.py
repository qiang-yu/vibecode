###
# This script reads ShareGPT-format JSON array files, strips the reasoning
# content from <think>...</think> blocks inside "from": "gpt" conversation
# turns, and writes the cleaned array to "<original>-no-thinking.json".
###

import json
import os
import re

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

INPUT_FILES = [
    "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-think-8b-clean-clean-tool_call_security-more-tools-clean-cut8192.json",
    "func-calling/Qwen3-8B/glaive-function-calling-5k-think-8b-clean-tool_call_security-more-tools-clean-cut8192.json",
]

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
THINK_REPLACEMENT = "<think>\n\n</think>"


# ------------------------------------------------------------------------------
# Processing
# ------------------------------------------------------------------------------

def process_conversations(conversations):
    """
    Strip <think> content from every 'gpt' turn.
    Returns the number of replacements made.
    """
    count = 0
    for turn in conversations:
        if turn.get("from") == "gpt":
            value = turn.get("value", "")
            new_value, n = THINK_PATTERN.subn(THINK_REPLACEMENT, value)
            if n:
                turn["value"] = new_value
                count += n
    return count


def process_file(input_path):
    """
    Process one input file and write the cleaned output next to it.
    Returns (output_path, number_of_replacements).
    """
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}-no-thinking{ext}"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_replaced = 0
    for item in data:
        conversations = item.get("conversations", [])
        total_replaced += process_conversations(conversations)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return output_path, total_replaced


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    total_replaced = 0

    for input_path in INPUT_FILES:
        if not os.path.isfile(input_path):
            print(f"[SKIP] File not found: {input_path}")
            continue

        print(f"[PROCESSING] {input_path}")
        output_path, count = process_file(input_path)
        print(f"[OUTPUT]     {output_path}")
        print(f"[REPLACED]   {count}")
        total_replaced += count

    print(f"[SUMMARY] Total <think> blocks replaced: {total_replaced}")


if __name__ == "__main__":
    main()
