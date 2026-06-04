###
# Remove system messages from ShareGPT-format conversations in a JSON dataset.
# Reads glaive-function-calling-5k.json, filters out every conversation entry
# where "from" equals "system", and writes the result to
# glaive-function-calling-5k-no-system.json.
###

import json
import os


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "glaive-function-calling-5k.json")
    output_path = os.path.join(base_dir, "glaive-function-calling-5k-no-system.json")

    print(f"Loading data from {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_items = len(data)
    removed_count = 0

    for item in data:
        original_len = len(item.get("conversations", []))
        item["conversations"] = [
            conv for conv in item.get("conversations", [])
            if conv.get("from") != "system"
        ]
        removed_count += original_len - len(item["conversations"])

    print(f"Processed {total_items} items, removed {removed_count} system messages.")
    print(f"Writing result to {output_path} ...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()
