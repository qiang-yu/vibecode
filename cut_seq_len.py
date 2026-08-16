###
# Filter ShareGPT-format JSON datasets by token sequence length.
#
# For each ShareGPT item, the script computes the token count in turn_by_turn
# mode using the tokenizer's chat template: for every assistant turn, it takes
# all messages from the start of the conversation up to and including that turn
# and records the prefix length. The entire item is discarded if any prefix
# length reaches or exceeds the configured cut_len. Only items whose every
# prefix length is strictly smaller than cut_len are kept.
#
# Kept items are written to a new file named "<original stem>-cut<cut_len>.json".
#
# Usage:
#     python cut_seq_len.py --dataset your_data.json
#     python cut_seq_len.py --dataset your_data.json --cut_len 8192
#     python cut_seq_len.py --dataset your_data.json --tokenizer /path/to/Qwen3-8B
#
#     python cut_seq_len.py --dataset func-calling/Qwen3-8B/glaive-function-calling-5k-think-8b-clean-tool_call_security-more-tools-clean.json
#
#     python cut_seq_len.py --dataset func-calling/Qwen3-8B/glaive-function-calling-5k-injected-think-8b-clean-clean-tool_call_security-more-tools-clean.json
###
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ROLE_MAP_DEFAULT = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}

# Input files to process when --dataset is not provided. Modify these paths as needed.
INPUT_FILES = [
    "func-calling/Qwen3-8B/glaive-function-calling-5k-think-8b-clean-tool_call_security-more-tools-clean-clean.json",
]

# Default tokenizer or model path. Modify this path as needed.
DEFAULT_TOKENIZER = "/home/qiangyu/Models/Qwen/Qwen3-8B"

# Default length cutoff. Items whose length reaches or exceeds this are discarded.
CUT_LEN = 8192


def format_messages(conversation: List[Dict[str, Any]], role_tag="from", content_tag="value",
                     tool_role_in_template="tool") -> List[Dict[str, str]]:
    """Convert a ShareGPT-style conversation into the model's chat template messages."""
    messages = []
    for turn in conversation:
        raw_role = turn.get(role_tag, "")
        role = ROLE_MAP_DEFAULT.get(raw_role, raw_role)
        if role == "tool" and tool_role_in_template == "user":
            role = "user"
        messages.append({"role": role, "content": turn.get(content_tag, "")})
    return messages


def apply_chat_template_len(tokenizer, messages, enable_thinking: bool) -> int:
    """Return the token count for the given messages, mirroring the training script's fallback logic."""
    base_kwargs = dict(tokenize=True, return_tensors=None, return_dict=True, add_generation_prompt=False)
    try:
        if enable_thinking:
            result = tokenizer.apply_chat_template(messages, **base_kwargs, enable_thinking=True)
        else:
            result = tokenizer.apply_chat_template(messages, **base_kwargs)
    except TypeError:
        result = tokenizer.apply_chat_template(messages, **base_kwargs)
    return len(result["input_ids"])


def compute_item_length(item, tokenizer, role_tag: str, content_tag: str,
                        msg_column: str, enable_thinking: bool,
                        tool_role_in_template: str) -> int:
    """
    Compute the maximum turn_by_turn token length for a ShareGPT item.
    For each assistant turn, all messages from the start up to and including
    that turn are encoded; the largest prefix length is returned.
    """
    conversation = item.get(msg_column, item.get("conversations", item.get("conversation", [])))
    if not conversation:
        return 0

    messages = format_messages(conversation, role_tag, content_tag, tool_role_in_template)

    max_length = 0
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    for idx in assistant_indices:
        prefix = messages[: idx + 1]
        length = apply_chat_template_len(tokenizer, prefix, enable_thinking)
        if length > max_length:
            max_length = length
    return max_length


def process_file(input_path: str, tokenizer, cut_len: int,
                 role_tag: str, content_tag: str, msg_column: str,
                 enable_thinking: bool, tool_role_in_template: str) -> None:
    """Filter one input file by sequence length and write the kept items."""
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    kept_items = []
    total_items = len(data)
    discarded_items = 0

    for item in data:
        length = compute_item_length(
            item, tokenizer, role_tag, content_tag, msg_column, enable_thinking,
            tool_role_in_template,
        )
        if length >= cut_len:
            discarded_items += 1
        else:
            kept_items.append(item)

    output_path = input_path.with_name(f"{input_path.stem}-cut{cut_len}{input_path.suffix}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(kept_items, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Input file: {input_path}")
    print(f"  Cut length: {cut_len}")
    print(f"  Total ShareGPT items: {total_items}")
    print(f"  Kept items: {len(kept_items)}")
    print(f"  Discarded items: {discarded_items}")
    print(f"  Wrote kept items to {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Filter ShareGPT-format datasets by token sequence length")
    parser.add_argument("--dataset", default=None, help="Path to a ShareGPT-format JSON dataset")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Tokenizer or model path")
    parser.add_argument("--cut_len", type=int, default=CUT_LEN, help="Length cutoff; items whose length reaches or exceeds this are discarded")
    parser.add_argument("--role_tag", default="from")
    parser.add_argument("--content_tag", default="value")
    parser.add_argument("--msg_column", default="conversations")
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--tool_role_in_template", default="tool",
                        help="Map tool turns to this role in the chat template")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    datasets = [args.dataset] if args.dataset else INPUT_FILES

    for input_file in datasets:
        process_file(
            input_file,
            tokenizer,
            args.cut_len,
            args.role_tag,
            args.content_tag,
            args.msg_column,
            args.enable_thinking,
            args.tool_role_in_template,
        )


if __name__ == "__main__":
    main()
