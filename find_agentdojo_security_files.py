###
# This script scans a directory recursively for JSON files in the agentdojo
# format and prints the relative paths of files whose root-level "security"
# field matches the configured find_security value.
###

import json
import os

# Configuration
find_security = True
input_dir = "/home/qiangyu/DevPhd/vibecode_agentdojo/util_scripts/attack_slack_defence_remove_tool_fake"


def main():
    matched = []         # security == find_security
    not_matched = []     # security field exists but value != find_security
    missing_field = []   # parsed ok but no "security" key
    parse_errors = []    # JSON decode failed
    read_errors = []     # file open/read failed

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, input_dir)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                parse_errors.append((rel_path, str(e)))
                continue
            except OSError as e:
                read_errors.append((rel_path, str(e)))
                continue

            if "security" not in data:
                missing_field.append(rel_path)
                continue

            if data["security"] == find_security:
                matched.append(rel_path)
            else:
                not_matched.append(rel_path)

    total = len(matched) + len(not_matched) + len(missing_field) + len(parse_errors) + len(read_errors)

    print(f"=== Summary: {total} JSON files total ===")
    print(f"  Matched   (security={find_security}):  {len(matched)}")
    print(f"  Not matched (security!={find_security}): {len(not_matched)}")
    print(f"  Missing 'security' field:              {len(missing_field)}")
    print(f"  JSON parse error:                      {len(parse_errors)}")
    print(f"  File read error:                       {len(read_errors)}")

    print(f"\n=== Matched (security={find_security}): {len(matched)} files ===")
    for rel_path in sorted(matched):
        print(rel_path)

    print(f"\n=== Missing 'security' field: {len(missing_field)} files ===")
    for rel_path in sorted(missing_field):
        print(rel_path)

    print(f"\n=== JSON parse error: {len(parse_errors)} files ===")
    for rel_path, reason in sorted(parse_errors):
        print(f"{rel_path}  [{reason}]")

    print(f"\n=== File read error: {len(read_errors)} files ===")
    for rel_path, reason in sorted(read_errors):
        print(f"{rel_path}  [{reason}]")


if __name__ == "__main__":
    main()
