#!/usr/bin/env python3
###
# Extracts one random sample per conversation-turn count from a JSON dataset.
# Reads func-calling/glaive-function-calling-5k.json, groups items by the
# number of human turns, randomly picks one item from each group, and writes
# a pretty-printed JSON array to extract-func-calling-data.json.
###

import json
import random
from collections import defaultdict

# Input and output file paths
INPUT_FILE = "func-calling/glaive-function-calling-5k.json"
OUTPUT_FILE = "extract-func-calling-data.json"


def count_human_turns(item):
    """Return the number of human turns in a single data item."""
    conversations = item.get("conversations", [])
    return sum(1 for msg in conversations if msg.get("from") == "human")


def has_valid_tools(item):
    """Return True if the item has a real tools field (not None or string 'null')."""
    tools = item.get("tools")
    if tools is None:
        return False
    if isinstance(tools, str) and tools.strip().lower() == "null":
        return False
    return True


def main():
    # Load the source dataset
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group items by human-turn count
    groups = defaultdict(list)
    for item in data:
        turns = count_human_turns(item)
        groups[turns].append(item)

    # Randomly pick one sample with non-null tools from each turn-count group
    random.seed(42)
    extracted = []
    for turns in sorted(groups):
        valid_items = [item for item in groups[turns] if has_valid_tools(item)]
        if not valid_items:
            print(f"Turns {turns}: no items with non-null tools, skipped")
            continue
        sample = random.choice(valid_items)
        extracted.append(sample)
        print(f"Turns {turns}: picked 1 sample from {len(valid_items)} valid items")

    # Write pretty-printed JSON array to output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=4)
        f.write("\n")

    print(f"\nExtracted {len(extracted)} samples into {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
