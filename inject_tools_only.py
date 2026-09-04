#!/usr/bin/env python3
###
# Inject additional tools into multiple ShareGPT-format function calling
# datasets. For each input file, an output file named "<stem>-more-tools.json"
# is produced in the same directory.
#
# Logic:
# 1. Stream through each input JSON Array item by item (two-pass).
# 2. Items with no tools are written unchanged.
# 3. Items with non-empty tools get 1-5 randomly sampled tool sets from other
#    items merged in, deduplicated by function name.
###

import json
import random
import sys
from pathlib import Path

# ==================== Configurable input/output paths ====================
INPUT_FILES = [
    "func-calling/Qwen3-8B/glaive-function-calling-5k-think-8b-clean-tool_call_security.json",
    "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-direct-think-8b-clean-clean-tool_call_security.json",
    "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-direct-template-think-8b-clean-clean-tool_call_security.json",
]
RANDOM_SEED = 42          # Fixed seed for reproducibility; set to None for non-deterministic runs
PROGRESS_INTERVAL = 100  # Print progress every N items
# =========================================================================


try:
    import ijson
except ImportError as exc:  # noqa: W0612
    print("Error: this script requires ijson for streaming JSON parsing.", file=sys.stderr)
    print("Install it with: pip install ijson", file=sys.stderr)
    sys.exit(1)


def parse_tools(item: dict):
    """
    Parse the tools list from an item.

    In the raw data "tools" is usually a JSON-encoded string (as shown in
    example.json), but list values are also accepted for robustness.
    """
    tools = item.get("tools")
    if tools is None:
        return None
    if isinstance(tools, str):
        s = tools.strip()
        if not s:
            return []
        return json.loads(s)
    if isinstance(tools, list):
        return tools
    return []


def has_valid_tools(item: dict) -> bool:
    """Return True if the item contains a non-empty tools list."""
    tools = parse_tools(item)
    return isinstance(tools, list) and len(tools) > 0


def tool_key(tool):
    """
    Return a unique key for a tool, used for deduplication.

    Standard OpenAI function tools look like:
    {"type": "function", "function": {"name": "xxx", ...}}
    so we prefer function.name as the deduplication key.
    """
    if not isinstance(tool, dict):
        return None
    func = tool.get("function")
    if isinstance(func, dict):
        return func.get("name")
    return tool.get("name")


def merge_tools(base_tools: list, extra_tools: list) -> list:
    """
    Merge two tool lists, deduplicate by function name, and preserve the
    relative order: base_tools first, then extra_tools.
    """
    seen = set()
    merged = []
    for t in base_tools + extra_tools:
        key = tool_key(t)
        if key is None:
            # Unrecognized tool; keep it to avoid data loss
            merged.append(t)
            continue
        if key not in seen:
            seen.add(key)
            merged.append(t)
    return merged


def preserve_tools_format(item: dict, merged_tools: list):
    """
    Write merged tools back using the original format:
    - original was str -> write as JSON-encoded string
    - original was list -> write as list
    """
    original_tools = item.get("tools")
    if isinstance(original_tools, list):
        item["tools"] = merged_tools
    else:
        item["tools"] = json.dumps(merged_tools, ensure_ascii=False)


def scan_tools_pool(input_path: Path):
    """First pass: collect indices and parsed tools for items with valid tools."""
    pool = []
    with input_path.open("r", encoding="utf-8") as f:
        for idx, item in enumerate(ijson.items(f, "item")):
            if has_valid_tools(item):
                pool.append((idx, parse_tools(item)))
    return pool


def build_tools_by_index(pool):
    """Convert the pool into a dict mapping index -> parsed tools."""
    return {idx: tools for idx, tools in pool}


def process_and_write(input_path: Path, output_path: Path, tools_by_index: dict):
    """Second pass: process each item and write the output JSON Array."""
    total_tools = len(tools_by_index)
    indices_with_tools = set(tools_by_index.keys())

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        fout.write("[\n")
        first = True

        for idx, item in enumerate(ijson.items(fin, "item")):
            if idx in indices_with_tools and total_tools >= 2:
                # Randomly pick 1 to 5 other items that have tools (cannot include itself)
                candidates = [i for i in indices_with_tools if i != idx]
                max_samples = min(5, len(candidates))
                sample_count = random.randint(1, max_samples)
                selected_indices = random.sample(candidates, sample_count)
                extra_tools = []
                for rand_idx in selected_indices:
                    extra_tools.extend(tools_by_index[rand_idx])
                merged = merge_tools(parse_tools(item), extra_tools)
                preserve_tools_format(item, merged)

            if not first:
                fout.write(",\n")
            first = False
            fout.write(json.dumps(item, ensure_ascii=False, indent=4))

            if (idx + 1) % PROGRESS_INTERVAL == 0:
                print(f"  Processed {idx + 1} items...")

        fout.write("\n]\n")


def derive_output_path(input_path: Path) -> Path:
    """Return '<stem>-more-tools.json' in the same directory as input_path."""
    return input_path.parent / (input_path.stem + "-more-tools.json")


def process_file(input_path: Path, output_path: Path):
    """Run the full two-pass pipeline for a single input/output pair."""
    print(f"\n[{input_path.name}]")

    print("  Phase 1: scanning items with tools...")
    pool = scan_tools_pool(input_path)
    total_with_tools = len(pool)
    print(f"  Found {total_with_tools} items with valid tools")

    if total_with_tools == 0:
        print("  No items with tools found; skipping.")
        return

    tools_by_index = build_tools_by_index(pool)

    print("  Phase 2: processing and writing output file...")
    process_and_write(input_path, output_path, tools_by_index)
    print(f"  Done. Output -> {output_path}")


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    errors = []
    for input_str in INPUT_FILES:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"Error: input file not found -> {input_path}", file=sys.stderr)
            errors.append(input_path)
            continue
        output_path = derive_output_path(input_path)
        process_file(input_path, output_path)

    if errors:
        print(f"\n{len(errors)} file(s) were skipped due to missing input.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
