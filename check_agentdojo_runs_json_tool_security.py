###
# This script checks tool call security annotations in AgentDojo run JSON files.
# It walks through ./runs, parses every JSON file, and classifies each file as
# valid, invalid, or no_tool_call based on whether assistant messages with
# non-empty tool_calls contain a <tool_call_security>...</tool_call_security>
# block in their content.
###

import json
import re
import sys
from pathlib import Path
from typing import Any


INPUT_DIR = Path("./runs")
SECURITY_BLOCK_PATTERN = re.compile(r"<tool_call_security>.*?</tool_call_security>", re.DOTALL)


def extract_text_content(content: Any) -> str:
    """Concatenate all text-type content blocks into a single string."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("content", "")))
    return "".join(parts)


def check_assistant_message(message: dict[str, Any]) -> tuple[int, int]:
    """Return (valid_tool_calls, invalid_tool_calls) for one assistant message."""
    tool_calls = message.get("tool_calls", [])
    if not isinstance(tool_calls, list) or not tool_calls:
        return 0, 0

    text = extract_text_content(message.get("content", ""))
    has_security_block = SECURITY_BLOCK_PATTERN.search(text) is not None

    if has_security_block:
        return len(tool_calls), 0
    return 0, len(tool_calls)


def check_json_file(path: Path) -> tuple[int, int, str]:
    """
    Check one JSON file.

    Returns:
        (valid_tool_calls, invalid_tool_calls, classification)
        classification is one of "valid", "invalid", "no_tool_call", "parse_error".
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Failed to read {path}: {exc}")
        return 0, 0, "parse_error"

    messages = []
    if isinstance(data, dict):
        raw_messages = data.get("messages", [])
        if isinstance(raw_messages, list):
            messages = raw_messages

    total_valid = 0
    total_invalid = 0
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            v, i = check_assistant_message(message)
            total_valid += v
            total_invalid += i

    if total_invalid > 0:
        return total_valid, total_invalid, "invalid"
    if total_valid > 0:
        return total_valid, total_invalid, "valid"
    return total_valid, total_invalid, "no_tool_call"


def print_path_list(title: str, paths: list[Path]) -> None:
    """Print a section title followed by a list of paths."""
    print("=" * 80)
    print(title)
    print("=" * 80)
    if paths:
        for path in paths:
            print(path)
    else:
        print("None")
    print()


def main() -> int:
    if not INPUT_DIR.is_dir():
        print(f"[ERROR] Input directory not found: {INPUT_DIR.resolve()}")
        return 1

    json_files = sorted(INPUT_DIR.rglob("*.json"))

    total_files = 0
    valid_files = 0
    invalid_files = 0
    no_tool_call_files = 0
    parse_error_files = 0
    total_tool_calls = 0
    valid_tool_calls = 0
    invalid_tool_calls = 0
    valid_paths = []
    invalid_paths = []
    no_tool_call_paths = []

    for json_path in json_files:
        total_files += 1
        v, i, classification = check_json_file(json_path)
        total_tool_calls += v + i
        valid_tool_calls += v
        invalid_tool_calls += i

        if classification == "valid":
            valid_files += 1
            valid_paths.append(json_path)
        elif classification == "invalid":
            invalid_files += 1
            invalid_paths.append(json_path)
        elif classification == "no_tool_call":
            no_tool_call_files += 1
            no_tool_call_paths.append(json_path)
        else:
            parse_error_files += 1

    print_path_list("Valid JSON files", valid_paths)
    print_path_list("Invalid JSON files", invalid_paths)
    print_path_list("No tool_call JSON files", no_tool_call_paths)

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total JSON files:          {total_files}")
    print(f"Valid JSON files:          {valid_files}")
    print(f"Invalid JSON files:        {invalid_files}")
    print(f"No tool_call JSON files:   {no_tool_call_files}")
    print(f"Parse error JSON files:    {parse_error_files}")
    print(f"Total tool calls:          {total_tool_calls}")
    print(f"Valid tool calls:          {valid_tool_calls}")
    print(f"Invalid tool calls:        {invalid_tool_calls}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
