#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inject additional tools into a ShareGPT-format function calling dataset.

Logic:
1. Stream through the input JSON Array item by item.
2. If an item has no "tools" or its tools list is empty, write it unchanged.
3. If an item has non-empty tools, randomly pick 1 to 5 other items that also
   have non-empty tools (cannot include the same item), merge their tools into
   the current item's tools, and deduplicate by function name.
4. Write the processed item to the output JSON Array.

Two-pass streaming is used so the large input file is never fully loaded.
"""

import json
import random
import sys
from pathlib import Path

# ==================== Configurable input/output paths ====================
INPUT_FILE = "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-pretend-think-8b-clean-clean-tool_call_security.json"
OUTPUT_FILE = "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-pretend-think-8b-clean-clean-tool_call_security-more-tools.json"
# INPUT_FILE = "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-think-8b-clean-clean-tool_call_security.json"
# OUTPUT_FILE = "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-think-8b-clean-clean-tool_call_security-more-tools.json"
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


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        print(f"Error: input file not found -> {input_path}", file=sys.stderr)
        sys.exit(1)

    print("Phase 1: scanning items with tools...")
    pool = scan_tools_pool(input_path)
    total_with_tools = len(pool)
    print(f"  Found {total_with_tools} items with valid tools")

    if total_with_tools == 0:
        print("No items with tools found; nothing to do.")
        return

    tools_by_index = build_tools_by_index(pool)

    print("Phase 2: processing and writing output file...")
    process_and_write(input_path, output_path, tools_by_index)
    print(f"Done. Output file -> {output_path}")


if __name__ == "__main__":
    main()
