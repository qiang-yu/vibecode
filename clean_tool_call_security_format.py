###
# This script filters ShareGPT-format JSON arrays, keeping only records that pass
# all tool_call_security format checks. Invalid records are discarded and a per-file
# summary shows how many records were affected by each error type.
# For each input file, a corresponding "-clean.json" output file is written.
###

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

INPUT_FILES = [
    "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-direct-think-8b-clean-clean-tool_call_security-more-tools-clean-clean-valid.json",
    "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-direct-template-think-8b-clean-clean-tool_call_security-more-tools-clean-clean-valid.json",
    "func-calling/Qwen3-8B/glaive-function-calling-5k-think-8b-clean-tool_call_security-more-tools-clean-clean-valid.json",
]

REQUIRED_TAGS = [
    "tool_name",
    "tool_args",
    "tool_reason",
    "trigger_words",
    "tool_trace",
    "tool_security",
]

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
SECURITY_OPEN = "<tool_call_security>"
SECURITY_CLOSE = "</tool_call_security>"
SECURITY_BLOCK_RE = re.compile(
    r"<tool_call_security>(.*?)</tool_call_security>", re.DOTALL
)
TOOL_CALL_OPEN_RE = re.compile(re.escape(TOOL_CALL_OPEN))
SECURITY_OPEN_RE = re.compile(re.escape(SECURITY_OPEN))
REQUIRED_TAGS_SET = set(REQUIRED_TAGS)
ALL_TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9_]*)>")


def extract_strings(obj):
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        result = []
        for item in obj:
            result.extend(extract_strings(item))
        return result
    if isinstance(obj, dict):
        result = []
        for value in obj.values():
            result.extend(extract_strings(value))
        return result
    return []


def check_security_content(content):
    """
    Validate the content inside a <tool_call_security> block.
    Returns a dict with three keys (all lists; empty means no issue):
      missing   - required tags whose open AND close count are both 0
      extra     - tag names not in REQUIRED_TAGS found in content
      bad_count - required tags whose open or close count is not exactly 1
    """
    result = {"missing": [], "extra": [], "bad_count": []}

    found_names = set(ALL_TAG_RE.findall(content))
    result["extra"] = sorted(found_names - REQUIRED_TAGS_SET)

    for tag in REQUIRED_TAGS:
        open_count = content.count(f"<{tag}>")
        close_count = content.count(f"</{tag}>")
        if open_count == 0 and close_count == 0:
            result["missing"].append(tag)
        elif open_count != 1 or close_count != 1:
            result["bad_count"].append(
                {"tag": tag, "open": open_count, "close": close_count}
            )

    return result


def collect_record_errors(record):
    """
    Check all format requirements for a record.
    Returns a set of error type strings; empty set means the record is valid.
    A record may trigger multiple error types simultaneously.
    """
    error_types = set()

    for text in extract_strings(record):
        if TOOL_CALL_CLOSE not in text:
            continue
        for match in re.finditer(re.escape(TOOL_CALL_CLOSE), text):
            pos = match.end()
            stripped = text[pos:].lstrip()
            if not stripped.startswith(SECURITY_OPEN):
                error_types.add("missing_security_block")
                continue
            sec_match = SECURITY_BLOCK_RE.match(stripped)
            if not sec_match:
                error_types.add("unclosed_security_block")
                continue
            issues = check_security_content(sec_match.group(1))
            if issues["missing"]:
                error_types.add("missing_tags")
            if issues["extra"]:
                error_types.add("extra_tags")
            if issues["bad_count"]:
                error_types.add("bad_tag_count")

    conversations = record.get("conversations", [])
    if isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                value = turn.get("value", "")
                if isinstance(value, str):
                    if len(TOOL_CALL_OPEN_RE.findall(value)) > 1:
                        error_types.add("multiple_tool_calls")
                    if len(SECURITY_OPEN_RE.findall(value)) > 1:
                        error_types.add("multiple_security_blocks")

    return error_types


def load_json_array(filepath):
    try:
        raw = filepath.read_text(encoding="utf-8").strip()
        data = json.loads(raw)
        if not isinstance(data, list):
            print(f"  ERROR: top-level value is not a JSON array in {filepath.name}", file=sys.stderr)
            return None
        return data
    except Exception as e:
        print(f"  ERROR reading {filepath.name}: {e}", file=sys.stderr)
        return None


def process_file(input_path):
    records = load_json_array(input_path)
    if records is None:
        return

    total = len(records)
    kept = []
    discard_reason_counts = defaultdict(int)

    for r in records:
        errors = collect_record_errors(r)
        if errors:
            for e in errors:
                discard_reason_counts[e] += 1
        else:
            kept.append(r)

    discarded = total - len(kept)

    stem = input_path.stem
    output_path = input_path.parent / f"{stem}-clean.json"
    output_path.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"  Input  : {input_path.name}")
    print(f"  Output : {output_path.name}")
    print(f"  Total  : {total}  |  Kept : {len(kept)}  |  Discarded : {discarded}")
    if discard_reason_counts:
        print(f"  Discard reasons (records affected per error type):")
        for reason in sorted(discard_reason_counts):
            print(f"    {reason:<32}: {discard_reason_counts[reason]}")
    print()


def main():
    sep = "=" * 72
    print(sep)
    print("TOOL CALL SECURITY FORMAT CLEANER")
    print(sep)
    print()

    for rel_path in INPUT_FILES:
        input_path = SCRIPT_DIR / rel_path
        if not input_path.exists():
            print(f"  SKIP (not found): {rel_path}", file=sys.stderr)
            print()
            continue
        process_file(input_path)

    print(sep)
    print("DONE")
    print(sep)


if __name__ == "__main__":
    main()
