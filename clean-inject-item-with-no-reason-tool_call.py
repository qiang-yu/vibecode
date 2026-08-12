###
# Clean ShareGPT items with unmatched tool responses.
#
# This script reads a JSON array of ShareGPT-format items and filters out
# any item where a "tool" message containing <tool_response> is not
# immediately preceded by a "gpt" message containing a matching <tool_call>.
# Valid items are written to a new JSON file, and processing progress is
# printed to the console.
###

import json
import re
import sys

INPUT_FILE = "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-3-think-8b-clean.json"
OUTPUT_FILE = "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-3-think-8b-clean-clean.json"

PROGRESS_INTERVAL = 100


def extract_json_inside_tag(text, tag_name):
    """
    Extract the first JSON object found inside XML-style tags.

    Args:
        text (str): The message text that contains the XML tags.
        tag_name (str): The XML tag name, e.g. "tool_call" or "tool_response".

    Returns:
        dict or None: The parsed JSON object, or None if extraction fails.
    """
    pattern = re.compile(
        rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None

    json_str = match.group(1)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def get_tool_name(tool_message, tag_name):
    """
    Parse a tool message and return the function name inside the XML tags.

    Args:
        tool_message (dict): A conversation message dictionary.
        tag_name (str): The XML tag name to look for.

    Returns:
        str or None: The extracted name, or None if not found.
    """
    value = tool_message.get("value", "")
    parsed = extract_json_inside_tag(value, tag_name)
    if not parsed:
        return None
    return parsed.get("name")


def is_valid_item(item):
    """
    Check whether all tool responses in an item have matching tool calls.

    Args:
        item (dict): A ShareGPT-format item.

    Returns:
        bool: True if every tool response is properly preceded by a matching
            tool call, False otherwise.
    """
    conversations = item.get("conversations", [])

    for idx, message in enumerate(conversations):
        if message.get("from") != "tool":
            continue

        response_name = get_tool_name(message, "tool_response")
        if response_name is None:
            # Tool message does not contain a valid <tool_response>.
            return False

        if idx == 0:
            # A tool message cannot be the first message.
            return False

        prev_message = conversations[idx - 1]
        if prev_message.get("from") != "gpt":
            return False

        call_name = get_tool_name(prev_message, "tool_call")
        if call_name is None:
            return False

        if call_name != response_name:
            return False

    return True


def main():
    """
    Read the input file, validate each item, and write valid items to output.
    """
    print(f"Reading input file: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Total items to process: {total}")

    cleaned = []

    for idx, item in enumerate(data, start=1):
        if is_valid_item(item):
            cleaned.append(item)

        if idx % PROGRESS_INTERVAL == 0 or idx == total:
            print(
                f"Processed {idx}/{total} items, "
                f"kept {len(cleaned)} valid items so far"
            )

    print(f"Writing {len(cleaned)} valid items to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4)
        f.write("\n")

    print(f"Done. Kept {len(cleaned)} out of {total} items.")


if __name__ == "__main__":
    main()
