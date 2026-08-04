###
# Randomly select a fixed number of ShareGPT-format entries from a large JSON
# array file and write them to a new JSON file with the same format.
# Output filename is derived from the input filename: <stem>-<count>.json,
# placed in the same directory as the input file.
###

import json
import os
import random

# ── configurable parameters ───────────────────────────────────────────────────
INPUT_FILE = "/home/qiangyu/Models/FineTune/Data/FineTome-100k-dedup-think-Qwen3-8B-cut8192.json"
SELECT_COUNT = 3400
# ─────────────────────────────────────────────────────────────────────────────


def build_output_path(input_path: str, count: int) -> str:
    directory = os.path.dirname(input_path)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    filename = f"{stem}-{count}.json"
    return os.path.join(directory, filename)


def main() -> None:
    output_file = build_output_path(INPUT_FILE, SELECT_COUNT)

    print(f"Input  : {INPUT_FILE}")
    print(f"Output : {output_file}")
    print(f"Target : {SELECT_COUNT} samples")
    print("Loading data ...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Total records read: {total}")

    if total <= SELECT_COUNT:
        selected = data
        print(f"Warning: dataset has only {total} records, using all of them.")
    else:
        selected = random.sample(data, SELECT_COUNT)

    written = len(selected)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=4)

    print(f"Records written   : {written}")
    print(f"Done. Output saved to: {output_file}")


if __name__ == "__main__":
    main()
