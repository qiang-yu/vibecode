###
# This script checks the consistency between tool_call names in "gpt" messages
# and tool_response names in the following "tool" messages within ShareGPT format data.
# It removes any ShareGPT entries where tool_call and tool_response names mismatch,
# and writes the clean data to a new file with "-clean.json" suffix.
###

import json
import re

# Configuration: path to the inference result file
INFERENCE_FILE_PATH = "/home/qiangyu/ClaudeCode/deal-func-calling-file/func-calling/Qwen3-8B/glaive-function-calling-5k-injected-3-think-8b.json"


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


def find_inconsistent_ids(data: list) -> set:
    """Find all ShareGPT IDs that have tool_call/tool_response name mismatches."""
    inconsistent_ids = set()

    for item in data:
        item_id = item.get("id", "unknown")
        conversations = item.get("conversations", [])

        i = 0
        while i < len(conversations):
            conv = conversations[i]
            if conv.get("from") == "gpt":
                tool_call_name = extract_tool_call_name(conv.get("value", ""))
                if tool_call_name is not None:
                    if i + 1 < len(conversations):
                        next_conv = conversations[i + 1]
                        if next_conv.get("from") == "tool":
                            tool_response_name = extract_tool_response_name(
                                next_conv.get("value", "")
                            )
                            if (
                                tool_response_name is not None
                                and tool_call_name != tool_response_name
                            ):
                                inconsistent_ids.add(item_id)
            i += 1

    return inconsistent_ids


def main():
    with open(INFERENCE_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total ShareGPT entries loaded: {len(data)}")

    inconsistent_ids = find_inconsistent_ids(data)
    print(f"Found {len(inconsistent_ids)} inconsistent entries")

    for item_id in sorted(inconsistent_ids):
        print(f"  - id: {item_id}")

    clean_data = [item for item in data if item.get("id") not in inconsistent_ids]
    print(f"Clean entries remaining: {len(clean_data)}")

    output_path = INFERENCE_FILE_PATH.replace(".json", "-clean.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    print(f"\nClean data written to: {output_path}")


if __name__ == "__main__":
    main()
