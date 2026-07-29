###
# Convert Qwen3/ShareGPT function-calling training data into the Llama3
# tool-calling format.
#
# For every "from": "gpt" message:
#   - Remove <think>...</think> blocks.
#   - Replace <tool_call>...</tool_call> with <|python_tag|> followed by the
#     inner content, stripping leading whitespace between the tag and content.
#     If the inner content is JSON and has a top-level "arguments" key, it is
#     renamed to "parameters" without any recursive replacement.
#   - Strip leading and trailing whitespace from the whole value.
#   - Leave all other content unchanged.
#
# For every "from": "tool" message:
#   - Remove <tool_response>...</tool_response> tags and keep only the inner
#     content.
#   - Strip leading and trailing whitespace from the whole value.
#
# Two hard-coded input files under func-calling/Qwen3-8B/ are processed and
# written to func-calling/Llama3/ with "-Llama3" appended to the base name.
###

import json
import os
import re


INPUT_FILES = [
    "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-think-8b-clean-clean-tool_call_security-more-tools-clean-cut8192.json",
    "func-calling/Qwen3-8B/glaive-function-calling-5k-think-8b-clean-tool_call_security-more-tools-clean-cut8192.json",
]
OUTPUT_DIR = "func-calling/Llama3"

THINK_PATTERN = re.compile(r"<think>[\s\S]*?</think>")
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*([\s\S]*?)</tool_call>")
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>\s*([\s\S]*?)\s*</tool_response>")
PYTHON_TAG = "<|python_tag|>"


def convert_tool_call_content(content: str) -> str:
    """Convert a single tool_call payload into Llama3 python_tag format."""
    content = content.strip()
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return f"{PYTHON_TAG}{content}"

    if isinstance(data, dict) and "arguments" in data:
        data["parameters"] = data.pop("arguments")
        content = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    return f"{PYTHON_TAG}{content}"


def convert_gpt_value(value: str) -> str:
    """Apply Llama3 conversion rules to a single GPT message value."""
    value = THINK_PATTERN.sub("", value)
    value = TOOL_CALL_PATTERN.sub(lambda m: convert_tool_call_content(m.group(1)), value)
    return value.strip()


def convert_tool_value(value: str) -> str:
    """Apply Llama3 conversion rules to a single tool message value."""
    value = TOOL_RESPONSE_PATTERN.sub(lambda m: m.group(1).strip(), value)
    return value.strip()


def convert_item(item: dict) -> dict:
    """Convert one ShareGPT sample in-place."""
    conversations = item.get("conversations", [])
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        sender = msg.get("from")
        original = msg.get("value", "")
        if sender == "gpt":
            msg["value"] = convert_gpt_value(original)
        elif sender == "tool":
            msg["value"] = convert_tool_value(original)
    return item


def build_output_path(input_path: str) -> str:
    """Map an input path to the corresponding Llama3 output path."""
    base = os.path.basename(input_path)
    name, ext = os.path.splitext(base)
    output_name = f"{name}-Llama3{ext}"
    return os.path.join(OUTPUT_DIR, output_name)


def process_file(input_path: str) -> None:
    """Read a ShareGPT JSON file, convert it, and write the result."""
    output_path = build_output_path(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = [convert_item(item) for item in data]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted: {input_path} -> {output_path}")


def main() -> None:
    for input_path in INPUT_FILES:
        process_file(input_path)


if __name__ == "__main__":
    main()
