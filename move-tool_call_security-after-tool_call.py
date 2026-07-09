###
# This script rearranges <tool_call_security> blocks in ShareGPT JSON data.
# For each "gpt" message, if both <tool_call_security> and <tool_call> blocks
# exist, the security block is moved immediately after the closing
# </tool_call> tag with no whitespace between them. Additionally, all extra
# whitespace (spaces and newlines) between any two of the tags
# <think>, <tool_call_security>, and <tool_call> is removed.
# Input files are configurable; output files are named by appending
# "-security-after-tool_call" before the original ".json" extension.
###

import argparse
import json
import re
from pathlib import Path


# Pattern: <tool_call_security>...</tool_call_security> (with optional whitespace/newlines) <tool_call>...</tool_call>
_SECURITY_BEFORE_TOOL_CALL_RE = re.compile(
    r"<tool_call_security>(.*?)</tool_call_security>\s*<tool_call>(.*?)</tool_call>",
    re.DOTALL,
)

# Pattern: whitespace between any two tags of interest (think, tool_call_security, tool_call).
_TAG_WHITESPACE_RE = re.compile(
    r"(</(?:think|tool_call_security|tool_call)>)\s+(<(?:think|tool_call_security|tool_call)>)",
    re.DOTALL,
)


def rearrange_security_block(text: str) -> str:
    """
    Move <tool_call_security> blocks immediately after <tool_call> blocks and
    remove all extra whitespace between any two relevant tags.
    """
    # First pass: move security block that appears before the tool_call block.
    text = _SECURITY_BEFORE_TOOL_CALL_RE.sub(
        r"<tool_call>\2</tool_call><tool_call_security>\1</tool_call_security>",
        text,
    )

    # Second pass: remove all whitespace between any two tags of interest.
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
                new_value = rearrange_security_block(original_value)
                message["value"] = new_value
    return item


def process_file(input_path: Path) -> None:
    """
    Read a ShareGPT JSON file, rearrange security blocks, and write the output.
    """
    input_path = input_path.resolve()

    # Output name: original stem + "-security-after-tool_call.json"
    output_path = input_path.with_name(f"{input_path.stem}-security-after-tool_call.json")

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
        description="Move <tool_call_security> blocks after <tool_call> blocks in ShareGPT JSON files.",
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
