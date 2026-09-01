###
# This script compares two AgentDojo result directories by finding all JSON files
# in both directories, then reporting missing files and differences in the
# 'utility' and/or 'security' fields based on configuration flags.
###

import json
import os
from pathlib import Path

# --- Configuration ---
DIR_1 = "/home/qiangyu/DevPhd/vibecode_agentdojo/util_scripts/attack_Qwen3Base_slack_old"
DIR_2 = "/home/qiangyu/DevPhd/vibecode_agentdojo/util_scripts/old_attacks/attack_cekl_train_20260828_ce_0.0_nosecurity_10.0_tagw_4epochs_8B_nothink_slack"

COMPARE_UTILITY = True
COMPARE_SECURITY = False

OUTPUT_FILE = "compare_agentdojo_result.txt"


def collect_json_files(base_dir):
    """Return a dict mapping relative_path -> absolute_path for all JSON files under base_dir."""
    base = Path(base_dir)
    result = {}
    for abs_path in base.rglob("*.json"):
        rel_path = abs_path.relative_to(base)
        result[str(rel_path)] = str(abs_path)
    return result


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    files_1 = collect_json_files(DIR_1)
    files_2 = collect_json_files(DIR_2)

    keys_1 = set(files_1.keys())
    keys_2 = set(files_2.keys())

    only_in_1 = sorted(keys_1 - keys_2)
    only_in_2 = sorted(keys_2 - keys_1)
    common = sorted(keys_1 & keys_2)

    diff_utility = []
    diff_security = []

    for rel_path in common:
        data1 = load_json(files_1[rel_path])
        data2 = load_json(files_2[rel_path])

        if COMPARE_UTILITY:
            v1 = data1.get("utility")
            v2 = data2.get("utility")
            if v1 != v2:
                diff_utility.append(
                    f"  {rel_path}  [dir1={v1}, dir2={v2}]"
                )

        if COMPARE_SECURITY:
            v1 = data1.get("security")
            v2 = data2.get("security")
            if v1 != v2:
                diff_security.append(
                    f"  {rel_path}  [dir1={v1}, dir2={v2}]"
                )

    # --- Output ---
    log("=" * 70)
    log("AgentDojo Result Comparison")
    log(f"  DIR_1: {DIR_1}")
    log(f"  DIR_2: {DIR_2}")
    log(f"  compare_utility={COMPARE_UTILITY}  compare_security={COMPARE_SECURITY}")
    log("=" * 70)

    log()
    log(f"[1] Files only in DIR_1 (missing in DIR_2): {len(only_in_1)}")
    if only_in_1:
        for p in only_in_1:
            log(f"  {p}")
    else:
        log("  (none)")

    log()
    log(f"[2] Files only in DIR_2 (missing in DIR_1): {len(only_in_2)}")
    if only_in_2:
        for p in only_in_2:
            log(f"  {p}")
    else:
        log("  (none)")

    log()
    if COMPARE_UTILITY:
        log(f"[3] Utility field differs: {len(diff_utility)} file(s)")
        if diff_utility:
            for p in diff_utility:
                log(p)
        else:
            log("  (none)")
    else:
        log("[3] Utility comparison skipped (compare_utility=False)")

    log()
    if COMPARE_SECURITY:
        log(f"[4] Security field differs: {len(diff_security)} file(s)")
        if diff_security:
            for p in diff_security:
                log(p)
        else:
            log("  (none)")
    else:
        log("[4] Security comparison skipped (compare_security=False)")

    log()
    log("=" * 70)
    log(f"Total common files compared: {len(common)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nResults written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
