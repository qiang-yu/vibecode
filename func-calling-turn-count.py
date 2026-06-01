#!/usr/bin/env python3
###
# Counts the distribution of conversation turns for each JSON file
# in a configurable directory. One turn = one message with "from": "human".
# Each file is processed and reported independently.
###

"""
==================================================
  File: glaive-function-calling-5k.json
==================================================
 Turns |    Count | Percentage
--------------------------------
     1 |      640 |     12.29%
     2 |     2663 |     51.12%
     3 |     1440 |     27.64%
     4 |      133 |      2.55%
     5 |      282 |      5.41%
     6 |       46 |      0.88%
     7 |        4 |      0.08%
     8 |        1 |      0.02%
--------------------------------
 Total |     5209 |    100.00%
"""

import json
import os
from collections import Counter

# Directory containing JSON files to analyze; change this path as needed
DIRECTORY = "func-calling"


def analyze_file(filepath):
    """
    Analyze a single JSON file and return turn-count statistics.

    Returns a tuple of (filename, turn_counts_dict, total_items).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    turn_counts = Counter()
    total_items = 0

    for item in data:
        total_items += 1
        conversations = item.get("conversations", [])
        human_turns = sum(1 for msg in conversations if msg.get("from") == "human")
        turn_counts[human_turns] += 1

    return os.path.basename(filepath), dict(turn_counts), total_items


def print_report(filename, turn_counts, total_items):
    """Print a formatted statistics report for a single file."""
    print(f"\n{'='*50}")
    print(f"  File: {filename}")
    print(f"{'='*50}")
    print(f"{'Turns':>6} | {'Count':>8} | {'Percentage':>10}")
    print("-" * 32)

    for turns in sorted(turn_counts):
        count = turn_counts[turns]
        percentage = count / total_items * 100
        print(f"{turns:>6} | {count:>8} | {percentage:>9.2f}%")

    print("-" * 32)
    print(f"{'Total':>6} | {total_items:>8} | {'100.00%':>10}")


def main():
    json_files = [
        os.path.join(DIRECTORY, f)
        for f in sorted(os.listdir(DIRECTORY))
        if f.endswith(".json")
    ]

    if not json_files:
        print(f"No JSON files found in directory: {DIRECTORY}")
        return

    for filepath in json_files:
        filename, turn_counts, total_items = analyze_file(filepath)
        print_report(filename, turn_counts, total_items)


if __name__ == "__main__":
    main()
