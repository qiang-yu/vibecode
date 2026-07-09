###
# This script moves <tool_call_security> blocks inside <think> tags in ShareGPT JSON data.
# For each "gpt" message, if both <think> and <tool_call_security> blocks exist,
# all <tool_call_security> blocks are moved to the end of the first <think> block,
# immediately before </think>. Additionally, all extra whitespace (spaces and
# newlines) between any two of the tags <think>, <tool_call_security>, and
# <tool_call> is removed.
# Input files are configurable; output files are named by appending
# "-security-in-think" before the original ".json" extension.
###

import argparse
import json
import re
from pathlib import Path


# Pattern: any <tool_call_security> block.
_TOOL_CALL_SECURITY_RE = re.compile(
    r"<tool_call_security>.*?</tool_call_security>",
    re.DOTALL,
)

# Pattern: whitespace between any two tags of interest (think, tool_call_security, tool_call).
_TAG_WHITESPACE_RE = re.compile(
    r"(</(?:think|tool_call_security|tool_call)>)\s+(<(?:think|tool_call_security|tool_call)>)",
    re.DOTALL,
)


def move_security_into_think(text: str) -> str:
    """
    Move all <tool_call_security> blocks to the end of the first <think> block
    and remove extra whitespace between relevant tags.
    """
    # If there is no think tag or no security tag, only normalize whitespace.
    if "<think>" not in text or "<tool_call_security>" not in text:
        return _TAG_WHITESPACE_RE.sub(r"\1\2", text)

    # Extract all security blocks in the order they appear and remove them.
    security_blocks = []

    def collect_security(match: re.Match) -> str:
        security_blocks.append(match.group(0))
        return ""

    text = _TOOL_CALL_SECURITY_RE.sub(collect_security, text)

    # Insert all collected security blocks at the end of the first think block.
    def insert_security(match: re.Match) -> str:
        think_content = match.group(1).rstrip()
        return f"<think>{think_content}{''.join(security_blocks)}</think>"

    text = re.sub(
        r"<think>(.*?)</think>",
        insert_security,
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Remove all extra whitespace between any two relevant tags.
    text = _TAG_WHITESPACE_RE.sub(r"\1\2", text)

    return text


def process_conversation_item(item: dict) -> dict:
    """
    Process all messages in a ShareGPT conversation item.
    """
    conversations = item.get("conversations", [])
    for message in conversations:
        if isinstance(message, dict) and message.get("from") == "gpt":
            original_value = message.get("value", "")
            if isinstance(original_value, str):
                new_value = move_security_into_think(original_value)
                message["value"] = new_value
    return item


def process_file(input_path: Path) -> None:
    """
    Read a ShareGPT JSON file, move security blocks into think tags, and write the output.
    """
    input_path = input_path.resolve()

    # Output name: original stem + "-security-in-think.json"
    output_path = input_path.with_name(f"{input_path.stem}-security-in-think.json")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Input file {input_path} must contain a JSON array.")

    processed_data = [process_conversation_item(item) for item in data]

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
        f.write("\n")

    print(f"Processed {input_path} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move <tool_call_security> blocks inside <think> tags in ShareGPT JSON files.",
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        default=[
            "func-calling/glaive-function-calling-5k-inference-32b-clean-tool-call-security-more-tools-clean.json",
            "func-calling/glaive-function-calling-5k-injected-inference-32b-clean-tool-call-security-more-tools-clean.json",
        ],
        help="Paths to input JSON files.",
    )
    args = parser.parse_args()

    if not args.input_files:
        parser.error("At least one input file must be provided.")

    for input_file in args.input_files:
        process_file(Path(input_file))


if __name__ == "__main__":
    main()
