###
# This script checks tool call security annotations in AgentDojo run JSON files.
# It walks through ./runs, parses every JSON file, and verifies that each
# assistant message with non-empty tool_calls contains a matching
# <tool_call_security><tool_name>...</tool_name></tool_call_security> block.
# Statistics and non-compliant file paths are printed at the end.
###

import json
import re
import sys
from pathlib import Path
from typing import Any


INPUT_DIR = Path("./runs")
SECURITY_BLOCK_PATTERN = re.compile(r"<tool_call_security>(.*?)</tool_call_security>", re.DOTALL)
TOOL_NAME_PATTERN = re.compile(r"<tool_name>(.*?)</tool_name>", re.DOTALL)


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


def extract_security_tool_names(text: str) -> set[str]:
    """Return the set of <tool_name> values found inside <tool_call_security> blocks."""
    names = set()
    for sec_match in SECURITY_BLOCK_PATTERN.finditer(text):
        inner = sec_match.group(1)
        for name_match in TOOL_NAME_PATTERN.finditer(inner):
            names.add(name_match.group(1).strip())
    return names


def get_tool_call_name(tool_call: Any) -> str | None:
    """Extract the function/tool name from a tool_call object."""
    if not isinstance(tool_call, dict):
        return None

    function = tool_call.get("function")
    if isinstance(function, str):
        return function
    if isinstance(function, dict):
        return function.get("name")

    return tool_call.get("name")


def check_assistant_message(message: dict[str, Any]) -> tuple[int, int]:
    """Return (valid_tool_calls, invalid_tool_calls) for one assistant message."""
    tool_calls = message.get("tool_calls", [])
    if not isinstance(tool_calls, list) or not tool_calls:
        return 0, 0

    text = extract_text_content(message.get("content", ""))
    security_tool_names = extract_security_tool_names(text)

    valid = 0
    invalid = 0
    for tool_call in tool_calls:
        function_name = get_tool_call_name(tool_call)
        if function_name is None:
            invalid += 1
            continue
        if function_name in security_tool_names:
            valid += 1
        else:
            invalid += 1

    return valid, invalid


def check_json_file(path: Path) -> tuple[int, int, bool, bool]:
    """
    Check one JSON file.

    Returns:
        (valid_tool_calls, invalid_tool_calls, is_file_compliant, parse_error)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Failed to read {path}: {exc}")
        return 0, 0, False, True

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

    return total_valid, total_invalid, total_invalid == 0, False


def main() -> int:
    if not INPUT_DIR.is_dir():
        print(f"[ERROR] Input directory not found: {INPUT_DIR.resolve()}")
        return 1

    json_files = sorted(INPUT_DIR.rglob("*.json"))

    total_files = 0
    compliant_files = 0
    non_compliant_files = 0
    total_tool_calls = 0
    valid_tool_calls = 0
    invalid_tool_calls = 0
    non_compliant_paths = []

    for json_path in json_files:
        total_files += 1
        v, i, is_compliant, _ = check_json_file(json_path)
        total_tool_calls += v + i
        valid_tool_calls += v
        invalid_tool_calls += i

        if is_compliant:
            compliant_files += 1
        else:
            non_compliant_files += 1
            non_compliant_paths.append(json_path)

    print("=" * 80)
    print("Non-compliant JSON files")
    print("=" * 80)
    if non_compliant_paths:
        for path in non_compliant_paths:
            print(path)
    else:
        print("None")
    print()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total JSON files:         {total_files}")
    print(f"Compliant JSON files:     {compliant_files}")
    print(f"Non-compliant JSON files: {non_compliant_files}")
    print(f"Total tool calls:         {total_tool_calls}")
    print(f"Valid tool calls:         {valid_tool_calls}")
    print(f"Invalid tool calls:       {invalid_tool_calls}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
