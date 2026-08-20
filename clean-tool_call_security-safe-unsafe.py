###
# This script filters ShareGPT-format JSON data based on tool_call_security rules.
# For GPT turns containing <tool_call> blocks, it verifies that:
#   Rule 1 - each </tool_call> is immediately followed by a well-formed <tool_call_security> block
#   Rule 2 - if the preceding turn is "human", tool_security must be "safe"
#   Rule 3 - if the preceding turn is "tool", tool_security must be "suspicious" or "unsafe"
# Non-compliant ShareGPT entries are discarded; compliant entries are written to output files.
###

import json
import re
import os

# Configurable security values
SAFE_VALUE = "safe"
AFTER_TOOL_RESPONSE_TOOL_SECURITY_ALLOWED_VALUES = {"suspicious", "unsafe"}

INPUT_FILES = [
    "func-calling/Qwen3-8B/glaive-function-calling-5k-think-8b-clean-tool_call_security-more-tools-clean.json",
    "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-simple-think-8b-clean-clean-tool_call_security-more-tools-clean.json",
    "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-pretend-think-8b-clean-clean-tool_call_security-more-tools-clean.json",
]

REQUIRED_SECURITY_TAGS = ["tool_name", "tool_args", "tool_reason", "tool_trace", "tool_security"]


def has_tool_call(text):
    return bool(re.search(r"<tool_call>.*?</tool_call>", text, re.DOTALL))


def check_each_tool_call_followed_by_security(text):
    for m in re.finditer(r"</tool_call>", text):
        remaining = text[m.end():]
        if not remaining.startswith("<tool_call_security>"):
            return False, "</tool_call> not immediately followed by <tool_call_security>"
    return True, None


def parse_security_block(block):
    """
    Validate tag presence and order inside a <tool_call_security> block.
    Returns (ok, security_value, error_reason).
    """
    positions = []
    for tag in REQUIRED_SECURITY_TAGS:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        m = re.search(pattern, block, re.DOTALL)
        if not m:
            return False, None, f"Missing or malformed tag: <{tag}>"
        positions.append(m.start())

    for i in range(len(positions) - 1):
        if positions[i] >= positions[i + 1]:
            return (
                False,
                None,
                f"Tags out of order: <{REQUIRED_SECURITY_TAGS[i]}> must come before <{REQUIRED_SECURITY_TAGS[i + 1]}>",
            )

    security_match = re.search(r"<tool_security>(.*?)</tool_security>", block, re.DOTALL)
    security_value = security_match.group(1).strip()
    return True, security_value, None


def validate_item(item):
    """
    Validate a single ShareGPT item.
    Returns (is_valid, rule_failed, reason).
    """
    conversations = item.get("conversations", [])

    for i, turn in enumerate(conversations):
        if turn.get("from") != "gpt":
            continue

        value = turn.get("value", "")

        if not has_tool_call(value):
            continue

        # Rule 1: every </tool_call> must be immediately followed by <tool_call_security>
        ok, reason = check_each_tool_call_followed_by_security(value)
        if not ok:
            return False, 1, reason

        num_tool_calls = len(re.findall(r"</tool_call>", value))
        security_blocks = re.findall(
            r"</tool_call><tool_call_security>(.*?)</tool_call_security>",
            value,
            re.DOTALL,
        )

        if len(security_blocks) != num_tool_calls:
            return (
                False,
                1,
                f"tool_call_security count ({len(security_blocks)}) != tool_call count ({num_tool_calls})",
            )

        prev_from = conversations[i - 1].get("from") if i > 0 else None

        for block in security_blocks:
            # Rule 1: validate inner tag format and order
            ok, security_value, reason = parse_security_block(block)
            if not ok:
                return False, 1, reason

            # Rule 2: preceding turn is "human" -> must be safe
            if prev_from == "human":
                if security_value != SAFE_VALUE:
                    return (
                        False,
                        2,
                        f"Preceding turn is human but tool_security='{security_value}', expected '{SAFE_VALUE}'",
                    )

            # Rule 3: preceding turn is "tool" -> must be suspicious or unsafe
            if prev_from == "tool":
                if security_value not in AFTER_TOOL_RESPONSE_TOOL_SECURITY_ALLOWED_VALUES:
                    return (
                        False,
                        3,
                        f"Preceding turn is tool but tool_security='{security_value}', expected one of {AFTER_TOOL_RESPONSE_TOOL_SECURITY_ALLOWED_VALUES}",
                    )

    return True, None, None


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process_file(input_path):
    base, ext = os.path.splitext(input_path)
    output_clean = base + "-clean" + ext
    output_rule1 = base + "-malformed_tool_call_security" + ext
    output_rule2 = base + "-human_not_safe" + ext
    output_rule3 = base + "-tool_not_suspicious_unsafe" + ext

    print(f"\nProcessing: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    kept = []
    discarded_rule1 = []
    discarded_rule2 = []
    discarded_rule3 = []

    for item in data:
        is_valid, rule_failed, _ = validate_item(item)
        if is_valid:
            kept.append(item)
        elif rule_failed == 1:
            discarded_rule1.append(item)
        elif rule_failed == 2:
            discarded_rule2.append(item)
        elif rule_failed == 3:
            discarded_rule3.append(item)

    total_discarded = len(discarded_rule1) + len(discarded_rule2) + len(discarded_rule3)

    print(f"  Total input       : {total}")
    print(f"  Discarded Rule 1 (malformed tool_call_security) : {len(discarded_rule1)}")
    print(f"  Discarded Rule 2 (human->not safe)              : {len(discarded_rule2)}")
    print(f"  Discarded Rule 3 (tool->not suspicious/unsafe)  : {len(discarded_rule3)}")
    print(f"  Total discarded   : {total_discarded}")
    print(f"  Remaining         : {len(kept)}")

    write_json(output_clean, kept)
    write_json(output_rule1, discarded_rule1)
    write_json(output_rule2, discarded_rule2)
    write_json(output_rule3, discarded_rule3)

    print(f"  Output clean      : {output_clean}")
    print(f"  Output rule1 err  : {output_rule1}")
    print(f"  Output rule2 err  : {output_rule2}")
    print(f"  Output rule3 err  : {output_rule3}")


def main():
    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            print(f"Warning: Input file not found: {input_file}")
            continue
        process_file(input_file)


if __name__ == "__main__":
    main()
