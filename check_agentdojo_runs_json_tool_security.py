###
# This script checks tool call security annotations in AgentDojo run JSON files.
# For each assistant message whose tool_calls array is non-empty, it counts the
# <tool_call_security>...</tool_call_security> blocks in the text content and
# compares that count to len(tool_calls), then validates the format of each block.
#
# The block scan is STRUCTURAL: opening and closing tags are counted separately and
# only well-formed, non-nested blocks are extracted. A naive non-greedy regex would
# swallow stray/duplicated opening tags into one giant "block" and wrongly report it
# as valid, so any tag imbalance or nesting is reported as "invalid_more".
###

import json
import re
import sys
from pathlib import Path
from typing import Any


INPUT_DIR = Path("./runs")

SECURITY_OPEN_TAG = "<tool_call_security>"
SECURITY_CLOSE_TAG = "</tool_call_security>"

# A well-formed block: its inner content must not contain either boundary tag.
# This makes stray opening tags and nesting impossible to hide inside a match.
WELL_FORMED_BLOCK_PATTERN = re.compile(
    r"<tool_call_security>((?:(?!</?tool_call_security>).)*)</tool_call_security>",
    re.DOTALL,
)

# The five required sub-tags, in the order they must appear.
REQUIRED_SUB_TAGS = ("tool_name", "tool_args", "tool_reason", "tool_trace", "tool_security")

# All five sub-tags must appear in order inside a security block.
VALID_SECURITY_INNER_PATTERN = re.compile(
    r"<tool_name>.*?</tool_name>.*?<tool_args>.*?</tool_args>.*?"
    r"<tool_reason>.*?</tool_reason>.*?<tool_trace>.*?</tool_trace>.*?"
    r"<tool_security>.*?</tool_security>",
    re.DOTALL,
)

# Tags that must never appear inside a security block.
STRAY_TAG_PATTERN = re.compile(r"</?tool_call>|</?tool_call_security>|</?think>")


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


def scan_security_blocks(text: str) -> tuple[list[str], bool]:
    """
    Scan the text for <tool_call_security> blocks.

    Returns (inners, structurally_broken).

    structurally_broken is True when the opening and closing tag counts differ, or
    when the number of well-formed non-nested blocks does not equal the number of
    opening tags. Either case means there are stray, duplicated, nested or unclosed
    tags in the output.
    """
    n_open = text.count(SECURITY_OPEN_TAG)
    n_close = text.count(SECURITY_CLOSE_TAG)
    inners = WELL_FORMED_BLOCK_PATTERN.findall(text)

    broken = (n_open != n_close) or (len(inners) != n_open)
    return inners, broken


def is_valid_inner(inner: str) -> bool:
    """
    Validate the body of one security block.

    Requires that no stray tag leaked in, that each required sub-tag appears
    exactly once as an open/close pair, and that all five appear in order.
    """
    if STRAY_TAG_PATTERN.search(inner):
        return False

    for tag in REQUIRED_SUB_TAGS:
        if inner.count(f"<{tag}>") != 1 or inner.count(f"</{tag}>") != 1:
            return False

    return bool(VALID_SECURITY_INNER_PATTERN.search(inner))


def classify_assistant_message(message: dict[str, Any]) -> tuple[int, str]:
    """
    Classify one assistant message.

    Uses tool_calls (the parsed array) to determine how many tool calls were made,
    then counts <tool_call_security> blocks in the text content and compares.

    Returns (n_tool_calls, classification):
      "no_tool_call"      - tool_calls is empty; nothing to check
      "no_security"       - tool calls present but zero security tags in text
      "missing_security"  - security block count is positive but less than tool call count
      "invalid_more"      - security block count exceeds tool call count, OR the security
                            tags are structurally broken (stray / unclosed / nested tags)
      "invalid_format"    - count matches but at least one block has a malformed body
      "valid"             - count matches and every block is well formed
    """
    tool_calls = message.get("tool_calls", [])
    if not isinstance(tool_calls, list) or not tool_calls:
        return 0, "no_tool_call"

    n = len(tool_calls)
    text = extract_text_content(message.get("content", ""))
    security_inners, broken = scan_security_blocks(text)

    # Nothing at all: no opening tag, no closing tag, no block.
    if not broken and not security_inners:
        return n, "no_security"

    # Stray, duplicated, nested or unclosed tags: too many tags for the tool calls made.
    if broken:
        return n, "invalid_more"

    s = len(security_inners)

    if s < n:
        return n, "missing_security"

    if s > n:
        return n, "invalid_more"

    # s == n: validate the body of every security block
    for inner in security_inners:
        if not is_valid_inner(inner):
            return n, "invalid_format"

    return n, "valid"


def check_json_file(path: Path) -> tuple[dict[str, int], str]:
    """
    Check one JSON file.

    Iterates all assistant messages and aggregates per-type tool call counts.

    Returns (counts, classification) where classification is the worst issue found:
      "valid", "no_security", "invalid_more", "invalid_format", "no_tool_call", "parse_error"
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Failed to read {path}: {exc}")
        return {"valid": 0, "no_security": 0, "missing_security": 0, "invalid_more": 0, "invalid_format": 0}, "parse_error"

    messages = []
    if isinstance(data, dict):
        raw = data.get("messages", [])
        if isinstance(raw, list):
            messages = raw

    total_counts: dict[str, int] = {
        "valid": 0,
        "no_security": 0,
        "missing_security": 0,
        "invalid_more": 0,
        "invalid_format": 0,
    }

    for message in messages:
        if not (isinstance(message, dict) and message.get("role") == "assistant"):
            continue
        n, cls = classify_assistant_message(message)
        if cls in total_counts:
            total_counts[cls] += n

    total_tc = sum(total_counts.values())
    if total_tc == 0:
        return total_counts, "no_tool_call"

    # File-level classification: worst issue wins
    if total_counts["no_security"] > 0:
        return total_counts, "no_security"
    if total_counts["invalid_more"] > 0:
        return total_counts, "invalid_more"
    if total_counts["invalid_format"] > 0:
        return total_counts, "invalid_format"
    return total_counts, "valid"


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
    no_security_files = 0
    missing_security_files = 0
    invalid_more_files = 0
    invalid_format_files = 0
    no_tool_call_files = 0
    parse_error_files = 0

    total_tc = 0
    valid_tc = 0
    no_security_tc = 0
    missing_security_tc = 0
    invalid_more_tc = 0
    invalid_format_tc = 0

    valid_paths: list[Path] = []
    no_security_paths: list[Path] = []
    missing_security_paths: list[Path] = []
    invalid_more_paths: list[Path] = []
    invalid_format_paths: list[Path] = []
    no_tool_call_paths: list[Path] = []

    for json_path in json_files:
        total_files += 1
        counts, classification = check_json_file(json_path)

        total_tc += sum(counts.values())
        valid_tc += counts["valid"]
        no_security_tc += counts["no_security"]
        missing_security_tc += counts["missing_security"]
        invalid_more_tc += counts["invalid_more"]
        invalid_format_tc += counts["invalid_format"]

        if classification == "parse_error":
            parse_error_files += 1
        elif classification == "no_tool_call":
            no_tool_call_files += 1
            no_tool_call_paths.append(json_path)
        else:
            # A file may have multiple issue types; add it to every relevant list.
            has_error = False
            if counts["no_security"] > 0:
                no_security_files += 1
                no_security_paths.append(json_path)
                has_error = True
            if counts["missing_security"] > 0:
                missing_security_files += 1
                missing_security_paths.append(json_path)
                has_error = True
            if counts["invalid_more"] > 0:
                invalid_more_files += 1
                invalid_more_paths.append(json_path)
                has_error = True
            if counts["invalid_format"] > 0:
                invalid_format_files += 1
                invalid_format_paths.append(json_path)
                has_error = True
            if not has_error:
                valid_files += 1
                valid_paths.append(json_path)

    print_path_list("Valid JSON files", valid_paths)
    print_path_list("No tool_call_security JSON files", no_security_paths)
    print_path_list("Missing tool_call_security JSON files", missing_security_paths)
    print_path_list("Invalid more tool_call_security JSON files", invalid_more_paths)
    print_path_list("Invalid tool_call_security JSON files", invalid_format_paths)
    print_path_list("No tool_call JSON files", no_tool_call_paths)

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total JSON files:                             {total_files}")
    print(f"Valid JSON files:                             {valid_files}")
    print(f"No tool_call_security JSON files:             {no_security_files}")
    print(f"Missing tool_call_security JSON files:        {missing_security_files}")
    print(f"Invalid more tool_call_security JSON files:   {invalid_more_files}")
    print(f"Invalid tool_call_security JSON files:        {invalid_format_files}")
    print(f"No tool_call JSON files:                      {no_tool_call_files}")
    print(f"Parse error JSON files:                       {parse_error_files}")
    print(f"Total tool calls:                             {total_tc}")
    print(f"Valid tool calls:                             {valid_tc}")
    print(f"No security tool calls:                       {no_security_tc}")
    print(f"Missing security tool calls:                  {missing_security_tc}")
    print(f"Invalid more tool calls:                      {invalid_more_tc}")
    print(f"Invalid format tool calls:                    {invalid_format_tc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
