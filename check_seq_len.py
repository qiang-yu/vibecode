###
# Analyze the token sequence length distribution of a dataset during actual training.
#
# Unlike simply tokenizing the entire conversation text, this script reproduces the
# turn_by_turn logic used by the training script: for each assistant turn, it takes
# all messages from the beginning of the conversation up to and including that turn,
# applies the chat template, and reports the length of the resulting training sample.
# The "whole" mode encodes the entire conversation once.
#
# Usage:
#     python check_seq_len.py --dataset your_data.json --tokenizer /path/to/Qwen3-8B
#     python check_seq_len.py --dataset your_data.json --tokenizer /path/to/Qwen3-8B --mode whole
#     python check_seq_len.py --dataset your_data.json --tokenizer /path/to/Qwen3-8B --plot out.png
#
#     python check_seq_len.py --dataset func-calling/Qwen3-8B/glaive-function-calling-5k-think-8b-clean-tool_call_security-more-tools-clean.json --tokenizer /home/qiangyu/Models/Qwen/Qwen3-8B
#
#     python check_seq_len.py --dataset func-calling/Qwen3-8B/glaive-function-calling-5k-injected-think-8b-clean-clean-tool_call_security-more-tools-clean.json --tokenizer /home/qiangyu/Models/Qwen/Qwen3-8B
###
import argparse
import json
from typing import Any, Dict, List


ROLE_MAP_DEFAULT = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}


def format_messages(conversation: List[Dict[str, Any]], role_tag="from", content_tag="value",
                     tool_role_in_template="tool") -> List[Dict[str, str]]:
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


def collect_lengths(data, tokenizer, mode: str, role_tag: str, content_tag: str,
                     msg_column: str, enable_thinking: bool):
    """
    Returns two lists of lengths: all_lengths, security_lengths
    (security_lengths only includes turn_by_turn samples whose assistant turn
    contains the tool_call_security tag, for a side-by-side comparison).
    """
    all_lengths = []
    security_lengths = []
    non_security_lengths = []

    for sample in data:
        conversation = sample.get(msg_column, sample.get("conversations", sample.get("conversation", [])))
        if not conversation:
            continue
        messages = format_messages(conversation, role_tag, content_tag)

        if mode == "whole":
            length = apply_chat_template_len(tokenizer, messages, enable_thinking)
            all_lengths.append(length)
            continue

        # turn_by_turn: one sample per assistant turn, using the prefix up to
        # and including that turn.
        assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        for idx in assistant_indices:
            prefix = messages[: idx + 1]
            length = apply_chat_template_len(tokenizer, prefix, enable_thinking)
            all_lengths.append(length)

            raw_content = conversation[idx].get(content_tag, "") if idx < len(conversation) else ""
            if "<tool_call_security>" in raw_content:
                security_lengths.append(length)
            else:
                non_security_lengths.append(length)

    return all_lengths, security_lengths, non_security_lengths


def print_stats(name: str, lengths: List[int]):
    if not lengths:
        print(f"\n[{name}] No samples")
        return
    lengths = sorted(lengths)
    n = len(lengths)

    def pct(p):
        idx = min(n - 1, int(n * p))
        return lengths[idx]

    print(f"\n[{name}] Sample count: {n}")
    print(f"  min:  {lengths[0]}")
    print(f"  p50:  {pct(0.50)}")
    print(f"  p90:  {pct(0.90)}")
    print(f"  p95:  {pct(0.95)}")
    print(f"  p99:  {pct(0.99)}")
    print(f"  max:  {lengths[-1]}")

    # Simple text histogram; bucket edges are fixed.
    buckets = [512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768]
    counts = [0] * (len(buckets) + 1)
    for l in lengths:
        placed = False
        for i, b in enumerate(buckets):
            if l <= b:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    print("  Distribution:")
    prev = 0
    for b, c in zip(buckets, counts[:-1]):
        if c:
            bar = "#" * max(1, int(50 * c / n))
            print(f"    <={b:>6}: {c:>6} ({c/n:.1%}) {bar}")
        prev = b
    if counts[-1]:
        bar = "#" * max(1, int(50 * counts[-1] / n))
        print(f"    > {buckets[-1]:>5}: {counts[-1]:>6} ({counts[-1]/n:.1%}) {bar}")


def main():
    parser = argparse.ArgumentParser(description="Analyze token sequence length distribution of training data")
    parser.add_argument("--dataset", required=True, help="Path to the ShareGPT-format JSON dataset")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer or model path")
    parser.add_argument("--mode", choices=["turn_by_turn", "whole"], default="turn_by_turn",
                         help="Conversation mode used in the training script; default is turn_by_turn")
    parser.add_argument("--role_tag", default="from")
    parser.add_argument("--content_tag", default="value")
    parser.add_argument("--msg_column", default="conversations")
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--plot", default=None, help="Optional path to save the length distribution histogram (requires matplotlib)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_lengths, security_lengths, non_security_lengths = collect_lengths(
        data, tokenizer, args.mode, args.role_tag, args.content_tag, args.msg_column, args.enable_thinking
    )

    print_stats("All training samples", all_lengths)
    if args.mode == "turn_by_turn":
        print_stats("Samples containing tool_call_security tag", security_lengths)
        print_stats("Samples without tool_call_security tag (replay data)", non_security_lengths)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.hist(all_lengths, bins=60)
            plt.xlabel("token length")
            plt.ylabel("count")
            plt.title(f"Sequence length distribution ({args.mode})")
            plt.axvline(16384, color="red", linestyle="--", label="cutoff_len=16384")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plot)
            print(f"\nHistogram saved to {args.plot}")
        except ImportError:
            print("\nmatplotlib not installed; skipping plotting (optional, does not affect statistics)")


if __name__ == "__main__":
    main()
