"""
诊断脚本：统计 ShareGPT 数据集中
1) 有多少 assistant turn 会被 tool_call_security gate 过滤掉（完全不参与训练）
2) 在被保留的 turn 里，think / tool_call_security / tool_call 三段各占多少字符（粗略估计 token 占比）

用法: python diag_tags.py your_dataset.json
"""
import json
import re
import sys

SECURITY_RE = re.compile(r"<tool_call_security>.*?</tool_call_security>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOLCALL_RE = re.compile(r"(?<!_)<tool_call>.*?</tool_call>", re.DOTALL)  # 避免匹配到 tool_call_security

def analyze(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_turns = 0
    gated_in = 0
    think_chars = security_chars = toolcall_chars = other_chars = 0

    for sample in data:
        for turn in sample.get("conversations", sample.get("conversation", [])):
            if turn.get("from") != "gpt":
                continue
            total_turns += 1
            content = turn.get("value", "")
            if "<tool_call_security>" not in content or "</tool_call_security>" not in content:
                continue
            gated_in += 1

            sec_len = sum(len(m.group()) for m in SECURITY_RE.finditer(content))
            think_len = sum(len(m.group()) for m in THINK_RE.finditer(content))
            tool_len = sum(len(m.group()) for m in TOOLCALL_RE.finditer(content))
            security_chars += sec_len
            think_chars += think_len
            toolcall_chars += tool_len
            other_chars += max(0, len(content) - sec_len - think_len - tool_len)

    print(f"assistant turn 总数: {total_turns}")
    print(f"通过 gate（含 tool_call_security）的 turn 数: {gated_in} ({gated_in/total_turns:.1%})")
    print(f"被完全跳过、不参与训练的 turn 数: {total_turns - gated_in} ({(total_turns-gated_in)/total_turns:.1%})")
    print()
    total_weighted = think_chars + security_chars + toolcall_chars
    print(f"think 字符占比:            {think_chars/total_weighted:.1%}" if total_weighted else "无数据")
    print(f"tool_call_security 字符占比: {security_chars/total_weighted:.1%}" if total_weighted else "")
    print(f"tool_call 字符占比:         {toolcall_chars/total_weighted:.1%}" if total_weighted else "")
    print(f"(参考) gated turn 内未被任何tag覆盖的字符: {other_chars}")

if __name__ == "__main__":
    analyze(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "--sample":
        print("\n\n===== 未被tag覆盖的游离文本样例 =====")
        sample_uncovered(sys.argv[1])


def sample_uncovered(path, n=5):
    """打印几条 gated turn 中，未被任何 tag 覆盖的具体文本，方便肉眼检查"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    shown = 0
    for sample in data:
        for turn in sample.get("conversations", sample.get("conversation", [])):
            if turn.get("from") != "gpt":
                continue
            content = turn.get("value", "")
            if "<tool_call_security>" not in content or "</tool_call_security>" not in content:
                continue
            stripped = content
            for pat in (SECURITY_RE, THINK_RE, TOOLCALL_RE):
                stripped = pat.sub("", stripped)
            stripped = stripped.strip()
            if len(stripped) > 20:  # 只看明显游离文本较多的样本
                print("=" * 60)
                print(repr(stripped[:500]))
                shown += 1
                if shown >= n:
                    return
