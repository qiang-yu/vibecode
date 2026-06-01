#!/usr/bin/env python3
###
# Counts the distribution of conversation turns in a JSON dataset.
# One turn = one message with "from": "human".
###

import json
from collections import Counter

# JSON file to analyze; change this path as needed
JSON_FILE = "glaive-function-calling-5k.json"


def main():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    turn_counts = Counter()
    total_items = 0

    for item in data:
        total_items += 1
        conversations = item.get("conversations", [])
        human_turns = sum(1 for msg in conversations if msg.get("from") == "human")
        turn_counts[human_turns] += 1

    print(f"{'Turns':>6} │ {'Count':>8} │ {'Percentage':>10}")
    print("─" * 32)

    for turns in sorted(turn_counts):
        count = turn_counts[turns]
        percentage = count / total_items * 100
        print(f"{turns:>6} │ {count:>8} │ {percentage:>9.2f}%")

    print("─" * 32)
    print(f"{'Total':>6} │ {total_items:>8} │ {'100.00%':>10}")


if __name__ == "__main__":
    main()
