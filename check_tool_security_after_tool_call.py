###
# This script checks ShareGPT-format JSON files for a tool-call security rule.
# For every "from": "gpt" message whose value contains <tool_call>, each
# <tool_call> must be followed by exactly one complete
# <tool_call_security>...</tool_call_security> block.
#
# Non-conforming items from all input files are written to OUTPUT_FILE whenever
# at least one non-conforming item exists.
#
# When CLEAN_ERROR_MESSAGE is True, any ShareGPT item containing at least one
# violating <tool_call> is discarded, and the remaining clean items are also
# written to "<input filename>-clean.json".
###

import json
import re
from pathlib import Path


# Input files to check. Modify these paths as needed.
INPUT_FILES = [
    "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-1234-think-8b-clean-clean-tool_call_security-more-tools.json",
]

# Output file for all non-conforming ShareGPT items. It is written only when
# at least one non-conforming item is found, regardless of CLEAN_ERROR_MESSAGE.
OUTPUT_FILE = "non_conforming_tool_call_security.json"

# When True, also write clean items to "<input filename>-clean.json".
CLEAN_ERROR_MESSAGE = True

TOOL_CALL_OPEN = "<tool_call>"
SECURITY_OPEN = "<tool_call_security>"
SECURITY_CLOSE = "</tool_call_security>"

# Match a complete <tool_call_security>...</tool_call_security> block.
SECURITY_BLOCK_RE = re.compile(
    re.escape(SECURITY_OPEN) + r".*?" + re.escape(SECURITY_CLOSE),
    re.DOTALL,
)


def count_security_blocks(value: str) -> int:
    """Return the number of complete security blocks in value."""
    return len(SECURITY_BLOCK_RE.findall(value))


def count_security_blocks_after_each_tool_call(value: str) -> list[int]:
    """
    For each <tool_call> in value, count the number of complete
    <tool_call_security>...</tool_call_security> blocks in the text
    that follows it (until the next <tool_call> or the end).
    """
    if TOOL_CALL_OPEN not in value:
        return []

    parts = value.split(TOOL_CALL_OPEN)
    # parts[0] is the text before the first <tool_call>.
    # parts[i] (i >= 1) is the text between the i-th and (i+1)-th <tool_call>.
    counts = []
    for i in range(1, len(parts)):
        following = parts[i]
        counts.append(count_security_blocks(following))
    return counts


def check_message(value: str) -> tuple[int, int, int]:
    """
    Check one "from": "gpt" message value.

    Returns:
        - total: number of <tool_call> occurrences.
        - missing_count: number of tool calls with 0 following security blocks
          (error type 1).
        - multiple_count: number of tool calls with more than 1 following
          security blocks (error type 2).
    """
    counts = count_security_blocks_after_each_tool_call(value)
    total = len(counts)
    missing_count = sum(1 for c in counts if c == 0)
    multiple_count = sum(1 for c in counts if c > 1)
    return total, missing_count, multiple_count


def check_item(item: dict) -> tuple[bool, dict]:
    """
    Check one ShareGPT item.

    Returns:
        - ok: True if the item has no tool-call security violations.
        - stats: dictionary with detailed statistics for this item.
    """
    conversations = item.get("conversations", [])

    total_messages = 0
    correct_messages = 0
    only_missing_messages = 0
    only_multiple_messages = 0
    both_errors_messages = 0

    total_tool_calls = 0
    correct_tool_calls = 0
    missing_tool_calls = 0
    multiple_tool_calls = 0

    for turn in conversations:
        if turn.get("from") != "gpt":
            continue

        value = turn.get("value", "")
        if TOOL_CALL_OPEN not in value:
            continue

        total_messages += 1
        t, missing_count, multiple_count = check_message(value)

        total_tool_calls += t
        correct_tool_calls += t - missing_count - multiple_count
        missing_tool_calls += missing_count
        multiple_tool_calls += multiple_count

        has_missing = missing_count > 0
        has_multiple = multiple_count > 0

        if not has_missing and not has_multiple:
            correct_messages += 1
        elif has_missing and has_multiple:
            both_errors_messages += 1
        elif has_missing:
            only_missing_messages += 1
        else:
            only_multiple_messages += 1

    ok = missing_tool_calls == 0 and multiple_tool_calls == 0

    stats = {
        "total_messages": total_messages,
        "correct_messages": correct_messages,
        "only_missing_messages": only_missing_messages,
        "only_multiple_messages": only_multiple_messages,
        "both_errors_messages": both_errors_messages,
        "total_tool_calls": total_tool_calls,
        "correct_tool_calls": correct_tool_calls,
        "missing_tool_calls": missing_tool_calls,
        "multiple_tool_calls": multiple_tool_calls,
    }
    return ok, stats


def process_file(input_path: str) -> list[dict]:
    """
    Process one input file.

    Returns the list of non-conforming items. When CLEAN_ERROR_MESSAGE is True,
    clean items are written to "<input filename>-clean.json".
    """
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clean_items = []
    non_conforming_items = []

    total_items = len(data)
    correct_items = 0
    only_missing_items = 0
    only_multiple_items = 0
    both_errors_items = 0

    total_messages = 0
    correct_messages = 0
    only_missing_messages = 0
    only_multiple_messages = 0
    both_errors_messages = 0

    total_tool_calls = 0
    correct_tool_calls = 0
    missing_tool_calls = 0
    multiple_tool_calls = 0

    for item in data:
        ok, stats = check_item(item)

        if ok:
            correct_items += 1
            clean_items.append(item)
        else:
            item_missing = stats["missing_tool_calls"] > 0
            item_multiple = stats["multiple_tool_calls"] > 0
            if item_missing and item_multiple:
                both_errors_items += 1
            elif item_missing:
                only_missing_items += 1
            elif item_multiple:
                only_multiple_items += 1
            non_conforming_items.append(item)

        total_messages += stats["total_messages"]
        correct_messages += stats["correct_messages"]
        only_missing_messages += stats["only_missing_messages"]
        only_multiple_messages += stats["only_multiple_messages"]
        both_errors_messages += stats["both_errors_messages"]

        total_tool_calls += stats["total_tool_calls"]
        correct_tool_calls += stats["correct_tool_calls"]
        missing_tool_calls += stats["missing_tool_calls"]
        multiple_tool_calls += stats["multiple_tool_calls"]

    non_conforming_items_count = (
        only_missing_items + only_multiple_items + both_errors_items
    )
    non_conforming_messages = (
        only_missing_messages
        + only_multiple_messages
        + both_errors_messages
    )

    print(f"Input file: {input_path}")
    print("  ShareGPT item statistics:")
    print(f"    Total items: {total_items}")
    print(f"    Correct items: {correct_items}")
    print(f"    Non-conforming items: {non_conforming_items_count}")
    print(f"      - Only error type 1 (missing security block): {only_missing_items}")
    print(f"      - Only error type 2 (multiple security blocks): {only_multiple_items}")
    print(f"      - Both error types: {both_errors_items}")
    print()
    print(
        f"  Total 'from': 'gpt' messages containing <tool_call>: {total_messages}"
    )
    print(f"  Correct messages: {correct_messages}")
    print(f"  Non-conforming messages: {non_conforming_messages}")
    print(f"    - Only error type 1 (missing security block): {only_missing_messages}")
    print(f"    - Only error type 2 (multiple security blocks): {only_multiple_messages}")
    print(f"    - Both error types: {both_errors_messages}")
    print()
    print("  Tool-call level statistics:")
    print(f"    Total <tool_call> occurrences: {total_tool_calls}")
    print(f"    Correct (exactly 1 following security block): {correct_tool_calls}")
    print(f"    Missing security block (error type 1): {missing_tool_calls}")
    print(f"    Multiple security blocks (error type 2): {multiple_tool_calls}")
    print()

    if CLEAN_ERROR_MESSAGE:
        clean_output_path = input_path.with_name(
            input_path.stem + "-clean" + input_path.suffix
        )
        with clean_output_path.open("w", encoding="utf-8") as f:
            json.dump(clean_items, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"  Wrote {len(clean_items)} clean item(s) to {clean_output_path}")
        print()

    return non_conforming_items


def main() -> None:
    all_non_conforming = []

    for input_file in INPUT_FILES:
        non_conforming = process_file(input_file)
        all_non_conforming.extend(non_conforming)

    if all_non_conforming:
        output_path = Path(OUTPUT_FILE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_non_conforming, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Wrote {len(all_non_conforming)} non-conforming item(s) to {output_path}")
    else:
        print("No non-conforming items found; no output file written.")


if __name__ == "__main__":
    main()
