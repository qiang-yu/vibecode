###
# This script checks the format of tool_call_security tags in fine-tuning data files.
# It scans all *-cut8192.json files in the target directory, verifies that every
# </tool_call> is followed by a valid <tool_call_security> block with all required
# sub-tags, and prints a detailed error report with statistics.
###

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

DATA_DIR = "/home/qiangyu/Models/FineTune/Data"

REQUIRED_TAGS = [
    "tool_name",
    "tool_args",
    "tool_reason",
    "trigger_words",
    "tool_trace",
    "tool_security",
]

TOOL_CALL_CLOSE = "</tool_call>"
SECURITY_OPEN = "<tool_call_security>"
SECURITY_CLOSE = "</tool_call_security>"
SECURITY_BLOCK_RE = re.compile(
    r"<tool_call_security>(.*?)</tool_call_security>", re.DOTALL
)


def extract_strings(obj):
    """Recursively collect all string values from a parsed JSON object."""
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
    """Return a list of tag names that are missing from the security block content."""
    missing = []
    for tag in REQUIRED_TAGS:
        pattern = re.compile(f"<{tag}>.*?</{tag}>", re.DOTALL)
        if not pattern.search(content):
            missing.append(tag)
    return missing


def check_text(text, record_index):
    """
    Scan a single text string for all </tool_call> occurrences and validate
    that each is followed by a well-formed <tool_call_security> block.

    Returns a list of error dicts.
    """
    errors = []
    close_len = len(TOOL_CALL_CLOSE)

    for match in re.finditer(re.escape(TOOL_CALL_CLOSE), text):
        pos = match.end()
        remainder = text[pos:]
        stripped = remainder.lstrip()

        if not stripped.startswith(SECURITY_OPEN):
            context = text[max(0, match.start() - 60): pos + 80].strip()
            errors.append({
                "type": "missing_security_block",
                "record_index": record_index,
                "context": context,
            })
            continue

        sec_match = SECURITY_BLOCK_RE.match(stripped)
        if not sec_match:
            context = stripped[:200].strip()
            errors.append({
                "type": "unclosed_security_block",
                "record_index": record_index,
                "context": context,
            })
            continue

        missing_tags = check_security_content(sec_match.group(1))
        if missing_tags:
            errors.append({
                "type": "missing_tags",
                "record_index": record_index,
                "missing": missing_tags,
                "context": sec_match.group(1)[:200].strip(),
            })

    return errors


def load_records(filepath):
    """
    Load records from a JSON file.
    Supports both a top-level JSON array and JSONL (one object per line).
    Returns (records, parse_errors).
    """
    parse_errors = []
    try:
        raw = filepath.read_text(encoding="utf-8").strip()
    except Exception as e:
        return [], [{"type": "file_read_error", "error": str(e)}]

    try:
        data = json.loads(raw)
        records = data if isinstance(data, list) else [data]
        return records, []
    except json.JSONDecodeError:
        pass

    records = []
    for line_num, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            parse_errors.append({
                "type": "json_parse_error",
                "line": line_num,
                "error": str(e),
            })

    return records, parse_errors


def process_file(filepath):
    """Process one file and return a list of all errors found."""
    records, parse_errors = load_records(filepath)
    errors = list(parse_errors)

    for record_index, record in enumerate(records):
        for text in extract_strings(record):
            if TOOL_CALL_CLOSE not in text:
                continue
            errors.extend(check_text(text, record_index))

    return errors


def print_report(all_reports, total_stats):
    sep = "=" * 72

    print(sep)
    print("TOOL CALL SECURITY FORMAT CHECK REPORT")
    print(sep)
    print(f"Directory : {DATA_DIR}")
    print(f"Files checked      : {total_stats['files_checked']}")
    print(f"Files with errors  : {total_stats['files_with_errors']}")
    print()
    print("Error Summary:")
    print(f"  Missing <tool_call_security> blocks : {total_stats['missing_security_block']}")
    print(f"  Security blocks with missing tags   : {total_stats['missing_tags']}")
    print(f"  Unclosed <tool_call_security> tags  : {total_stats['unclosed_security_block']}")
    print(f"  JSON / file parse errors            : {total_stats['json_errors']}")

    if not all_reports:
        print()
        print("All files passed the format check.")
        print(sep)
        return

    print()
    print(sep)
    print("DETAILED ERROR REPORT")
    print(sep)

    for report in all_reports:
        print(f"\nFile: {report['file']}  ({len(report['errors'])} error(s))")

        by_type = defaultdict(list)
        for err in report["errors"]:
            by_type[err["type"]].append(err)

        if "file_read_error" in by_type:
            for err in by_type["file_read_error"]:
                print(f"  [file_read_error] {err['error']}")

        if "json_parse_error" in by_type:
            errs = by_type["json_parse_error"]
            print(f"  [json_parse_error] {len(errs)} line(s) failed to parse:")
            for err in errs:
                print(f"    line {err['line']}: {err['error']}")

        if "missing_security_block" in by_type:
            errs = by_type["missing_security_block"]
            print(f"  [missing_security_block] {len(errs)} occurrence(s):")
            for i, err in enumerate(errs[:5], start=1):
                ctx = err["context"].replace("\n", " ")[:120]
                print(f"    #{i} (record {err['record_index']}): ...{ctx}...")
            if len(errs) > 5:
                print(f"    ... and {len(errs) - 5} more")

        if "unclosed_security_block" in by_type:
            errs = by_type["unclosed_security_block"]
            print(f"  [unclosed_security_block] {len(errs)} occurrence(s):")
            for i, err in enumerate(errs[:5], start=1):
                ctx = err["context"].replace("\n", " ")[:120]
                print(f"    #{i} (record {err['record_index']}): {ctx}")
            if len(errs) > 5:
                print(f"    ... and {len(errs) - 5} more")

        if "missing_tags" in by_type:
            errs = by_type["missing_tags"]
            tag_counts = defaultdict(int)
            for err in errs:
                for tag in err["missing"]:
                    tag_counts[tag] += 1
            print(f"  [missing_tags] {len(errs)} security block(s) have missing tags:")
            for tag in REQUIRED_TAGS:
                if tag in tag_counts:
                    print(f"    <{tag}>  missing in {tag_counts[tag]} block(s)")

    print()
    print(sep)
    print("CHECK COMPLETE")
    print(sep)


def main():
    data_dir = Path(DATA_DIR)

    if not data_dir.exists():
        print(f"Error: directory not found: {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(data_dir.glob("*-cut8192.json"))

    if not json_files:
        print(f"No *-cut8192.json files found in {DATA_DIR}")
        sys.exit(0)

    total_stats = {
        "files_checked": 0,
        "files_with_errors": 0,
        "missing_security_block": 0,
        "missing_tags": 0,
        "unclosed_security_block": 0,
        "json_errors": 0,
    }

    all_reports = []

    for filepath in json_files:
        errors = process_file(filepath)
        total_stats["files_checked"] += 1

        if errors:
            total_stats["files_with_errors"] += 1
            all_reports.append({"file": filepath.name, "errors": errors})

            for err in errors:
                t = err["type"]
                if t == "missing_security_block":
                    total_stats["missing_security_block"] += 1
                elif t == "missing_tags":
                    total_stats["missing_tags"] += 1
                elif t == "unclosed_security_block":
                    total_stats["unclosed_security_block"] += 1
                elif t in ("json_parse_error", "file_read_error"):
                    total_stats["json_errors"] += 1

    print_report(all_reports, total_stats)


if __name__ == "__main__":
    main()
