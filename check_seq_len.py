"""
统计数据集实际训练时的 token 序列长度分布。

跟简单地把整段对话文本 tokenize 一遍不同，这里按你训练脚本实际的
turn_by_turn 逻辑复现：对每个 assistant 轮次，取"从对话开头到这一轮
为止"的所有消息，套用 chat template 编码，得到的长度才是真正喂给模型
的那条训练样本的长度。whole 模式则是整段对话编码一次。

用法：
    python check_seq_len.py --dataset your_data.json --tokenizer /path/to/Qwen3-8B
    python check_seq_len.py --dataset your_data.json --tokenizer /path/to/Qwen3-8B --mode whole
    python check_seq_len.py --dataset your_data.json --tokenizer /path/to/Qwen3-8B --plot out.png

    python check_seq_len.py --dataset func-calling/Qwen3-8B/glaive-function-calling-5k-think-8b-clean-tool_call_security-more-tools-clean.json --tokenizer /home/qiangyu/Models/Qwen/Qwen3-8B

    python check_seq_len.py --dataset func-calling/Qwen3-8B/glaive-function-calling-5k-injected-think-8b-clean-clean-tool_call_security-more-tools-clean.json --tokenizer /home/qiangyu/Models/Qwen/Qwen3-8B

"""
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
        print(f"\n[{name}] 无样本")
        return
    lengths = sorted(lengths)
    n = len(lengths)

    def pct(p):
        idx = min(n - 1, int(n * p))
        return lengths[idx]

    print(f"\n[{name}] 样本数: {n}")
    print(f"  min:  {lengths[0]}")
    print(f"  p50:  {pct(0.50)}")
    print(f"  p90:  {pct(0.90)}")
    print(f"  p95:  {pct(0.95)}")
    print(f"  p99:  {pct(0.99)}")
    print(f"  max:  {lengths[-1]}")

    # 简单文本直方图，桶宽根据最大值自动调整
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

    print("  分布:")
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
    parser = argparse.ArgumentParser(description="统计训练数据的 token 序列长度分布")
    parser.add_argument("--dataset", required=True, help="ShareGPT 格式的 json 数据集路径")
    parser.add_argument("--tokenizer", required=True, help="tokenizer / 模型路径")
    parser.add_argument("--mode", choices=["turn_by_turn", "whole"], default="turn_by_turn",
                         help="对应训练脚本里的 conversation_mode，默认 turn_by_turn")
    parser.add_argument("--role_tag", default="from")
    parser.add_argument("--content_tag", default="value")
    parser.add_argument("--msg_column", default="conversations")
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--plot", default=None, help="可选：保存长度分布直方图到该路径 (需要 matplotlib)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_lengths, security_lengths, non_security_lengths = collect_lengths(
        data, tokenizer, args.mode, args.role_tag, args.content_tag, args.msg_column, args.enable_thinking
    )

    print_stats("全部训练样本", all_lengths)
    if args.mode == "turn_by_turn":
        print_stats("含 tool_call_security 标签的样本", security_lengths)
        print_stats("不含 tool_call_security 标签的样本(replay数据)", non_security_lengths)

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
            print(f"\n直方图已保存到 {args.plot}")
        except ImportError:
            print("\n未安装 matplotlib，跳过画图（可选功能，不影响统计结果）")


if __name__ == "__main__":
    main()
