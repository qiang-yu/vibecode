###
# This script checks the consistency between tool_call names in "gpt" messages
# and tool_response names in the following "tool" messages within ShareGPT format data.
# It reads a JSON file containing an array of ShareGPT conversations, extracts
# tool call and response names, and reports any mismatches by their conversation ID.
###

import json
import re

# Configuration: path to the inference result file
INFERENCE_FILE_PATH = "func-calling/Qwen3-8B/glaive-function-calling-5k-think-8b.json"


def extract_tool_call_name(value: str) -> str | None:
    """Extract the 'name' field from a <tool_call> block."""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", value, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return data.get("name")
    except json.JSONDecodeError:
        return None


def extract_tool_response_name(value: str) -> str | None:
    """Extract the 'name' field from a <tool_response> block."""
    match = re.search(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", value, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return data.get("name")
    except json.JSONDecodeError:
        return None


def main():
    with open(INFERENCE_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    mismatch_count = 0
    total_checked = 0

    for item in data:
        item_id = item.get("id", "unknown")
        conversations = item.get("conversations", [])

        i = 0
        while i < len(conversations):
            conv = conversations[i]
            if conv.get("from") == "gpt":
                tool_call_name = extract_tool_call_name(conv.get("value", ""))
                if tool_call_name is not None:
                    # The next conversation should be from "tool"
                    if i + 1 < len(conversations):
                        next_conv = conversations[i + 1]
                        if next_conv.get("from") == "tool":
                            tool_response_name = extract_tool_response_name(
                                next_conv.get("value", "")
                            )
                            total_checked += 1
                            if (
                                tool_response_name is not None
                                and tool_call_name != tool_response_name
                            ):
                                print(
                                    f"Mismatch found - id: {item_id}, "
                                    f"tool_call name: {tool_call_name}, "
                                    f"tool_response name: {tool_response_name}"
                                )
                                mismatch_count += 1
            i += 1

    print(f"\nTotal tool calls checked: {total_checked}")
    print(f"Mismatches found: {mismatch_count}")


if __name__ == "__main__":
    main()
