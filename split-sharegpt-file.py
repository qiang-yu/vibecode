###
# This script splits a large ShareGPT-format JSON array file into
# multiple smaller files based on a configurable split count.
###

import json
import os

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
SPLIT_NUM = 3
INPUT_FILE = "func-calling/glaive-function-calling-5k-injected-5.json"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_json_array(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_array(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------
def split_array(data, n):
    """Split data into n parts as evenly as possible."""
    total = len(data)
    base, remainder = divmod(total, n)
    parts = []
    start = 0
    for i in range(n):
        size = base + (1 if i < remainder else 0)
        parts.append(data[start:start + size])
        start += size
    return parts


def build_output_path(input_path, index):
    base, ext = os.path.splitext(input_path)
    return f"{base}-{index}{ext}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data = load_json_array(INPUT_FILE)
    total = len(data)
    print(f"Loaded {total} items from {INPUT_FILE}")

    parts = split_array(data, SPLIT_NUM)
    print(f"Splitting into {SPLIT_NUM} parts:")

    for i, part in enumerate(parts):
        out_path = build_output_path(INPUT_FILE, i)
        save_json_array(out_path, part)
        print(f"  [{i}] {out_path} -> {len(part)} items")

    print("Done.")


if __name__ == "__main__":
    main()
