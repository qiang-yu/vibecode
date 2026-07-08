###
# lora-ce-kl-8b.py
#
# A standalone LoRA fine-tuning script for causal language models such as
# Qwen3, Llama, and Mistral. It reproduces the behavior of the original
# LlamaFactory command used for Qwen3-8B SFT, while keeping every hyper-parameter
# configurable through a YAML/JSON config file or command-line arguments.
#
# Supported features:
#   - Multi-GPU training with configurable GPU visibility (e.g. "4,5,6,7")
#   - DeepSpeed ZeRO-2 integration
#   - Flash Attention 2 / SDPA / eager attention back-ends
#   - ShareGPT-format datasets described by dataset_info.json
#   - Turn-by-turn and whole-conversation training modes
#   - Model-agnostic LoRA targeting (all-linear fallback + custom targets)
#   - Per-tag weighted loss for all assistant content; default weight for untagged text
#   - Gradient checkpointing for long-sequence training
#   - Loss plotting and checkpointing
###

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import torch
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_origin, get_type_hints

# These classes are needed at module load time for subclassing. They import
# transformers but do not allocate GPU memory; model loading stays deferred.
from transformers import DataCollatorForSeq2Seq, Trainer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Holds every tunable option used during LoRA SFT."""

    # Model and tokenizer
    model_name_or_path: str = "/home/qiangyu/Models/Qwen/Qwen3-8B"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"  # LlamaFactory: pure_bf16 True
    chat_template: Optional[str] = None  # override tokenizer.chat_template

    # Attention back-end
    flash_attn: str = "auto"  # auto | fa2 | sdpa | eager

    # LoRA
    finetuning_type: str = "lora"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target: str = "all"  # "all" targets all Linear layers; comma-list for custom modules
    lora_bias: str = "none"
    use_rslora: bool = False

    # Dataset metadata and data loading
    dataset_dir: str = "/home/qiangyu/Models/FineTune/Data"
    dataset_info_path: str = "/home/qiangyu/Models/FineTune/Data/dataset_info.json"
    datasets: List[str] = field(default_factory=lambda: ["no_inject_data", "simple_inject_data"])
    role_tag: str = "from"
    content_tag: str = "value"
    user_tag: str = "human"
    assistant_tag: str = "gpt"
    system_tag: str = "system"
    observation_tag: str = "tool"
    tool_role_in_template: str = "tool"  # tool | user; used when the chat template lacks a tool role
    loss_calc_ce_default_with_security_weight: float = 1.0  # default CE weight for untagged assistant content in security turns
    loss_calc_ce_tool_call_security_tag: Optional[str] = "tool_call_security"  # weighted security-analysis tag
    loss_calc_ce_tool_call_security_with_security_weight: float = 2.0  # loss weight for tool_call_security content
    loss_calc_ce_think_with_security_tag: Optional[str] = "think"  # weighted think block
    loss_calc_ce_think_with_security_weight: float = 1.0  # loss weight for think content in turns with tool_call_security
    loss_calc_ce_tool_call_with_security_tag: Optional[str] = "tool_call"  # weighted tool-call JSON block
    loss_calc_ce_tool_call_with_security_weight: float = 1.0  # loss weight for tool_call content in turns with tool_call_security
    loss_calc_ce_default_without_security_weight: float = 1.0  # uniform loss weight for all assistant tokens in turns without tool_call_security

    # In turn-by-turn mode, the number of non-security assistant turns kept per
    # dataset file is controlled relative to the number of security turns in that
    # file. After generating all turns, keep at most
    # security_turn_count * non_security_to_security_turn_ratio non-security turns.
    # 0.0 means keep only security turns; 0.5 means keep half as many non-security
    # turns as security turns; 1.0 means keep equal numbers. The selection is
    # random within each file's non-security turns and is reproducible given
    # non_security_turn_sample_seed.
    non_security_to_security_turn_ratio: float = 0.0
    non_security_turn_sample_seed: int = 42

    cutoff_len: int = 16384
    max_samples: Optional[int] = 100000
    preprocessing_num_workers: int = 16
    shuffle_seed: int = 42
    eval_data_ratio: float = 0.05

    # Training hyper-parameters
    stage: str = "sft"
    do_train: bool = True
    output_dir: str = "/home/qiangyu/Models/FineTune/Qwen/train_20260630"
    overwrite_output_dir: bool = False
    seed: int = 42
    num_train_epochs: float = 6.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-05
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    max_steps: int = -1  # -1 means use num_train_epochs
    gradient_checkpointing: bool = True  # trade compute for memory; required for long sequences

    # Logging, evaluation and checkpointing
    logging_steps: int = 10  # log loss every N steps
    save_steps: int = 100
    eval_strategy: str = "steps"  # no | steps | epoch
    eval_steps: int = 100
    save_total_limit: Optional[int] = None
    logging_dir: Optional[str] = None
    report_to: str = "none"  # none | tensorboard | wandb | ...
    plot_loss: bool = True
    plot_eval_loss: bool = True
    eval_after_save: bool = True
    final_eval: bool = True
    include_num_input_tokens_seen: bool = True

    # Distributed training
    # GPU selection is controlled by CUDA_VISIBLE_DEVICES in lora-ce-kl-8b-run.sh.
    ddp_timeout: int = 180000000
    deepspeed_config: Optional[str] = "ds_z2_config.json"
    local_rank: int = -1  # automatically populated by torchrun/DeepSpeed

    # Resource limits
    per_device_max_memory_gb: Optional[float] = 70.0  # cap PyTorch memory per GPU; null disables the limit

    # Qwen3-specific thinking mode
    enable_thinking: bool = True

    # KL anchoring against the frozen base weights inside the LoRA model.
    # disable_adapter() is used during training to obtain reference logits, so no
    # separate reference model is loaded.
    loss_calc_kl_enabled: bool = False
    loss_calc_kl_alpha: float = 0.2
    loss_calc_kl_think_with_security_alpha: float = 0.1  # KL coefficient for <think> in security turns when enabled
    loss_calc_kl_enable_without_security: bool = False  # enable KL for assistant turns without tool_call_security
    loss_calc_kl_enable_with_security: bool = False  # enable KL for assistant turns that contain tool_call_security (background + think, excluding tool_call blocks)

    # Optional external config file (YAML or JSON)
    config_file: Optional[str] = None


def _add_argument(parser: argparse.ArgumentParser, name: str, default: Any, arg_type: Any) -> None:
    """
    Register a single CLI argument derived from a dataclass field.

    ``default`` already merges config-file values over dataclass defaults, so it is
    used directly as the argparse default. The ``type`` converters below only run on
    strings supplied on the command line; the default (already a proper Python
    value from YAML/JSON or the dataclass) is passed through untouched.
    """
    if arg_type == bool:
        parser.add_argument(
            f"--{name}",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=default,
        )
    elif arg_type == list or get_origin(arg_type) is list:
        parser.add_argument(
            f"--{name}",
            type=lambda x: [item.strip() for item in x.split(",") if item.strip()],
            default=default,
        )
    elif arg_type == Optional[int]:
        parser.add_argument(
            f"--{name}",
            type=lambda x: None if x.lower() in ("none", "null", "") else int(x),
            default=default,
        )
    elif arg_type == Optional[str]:
        parser.add_argument(
            f"--{name}",
            type=lambda x: None if x.lower() in ("none", "null", "") else x,
            default=default,
        )
    elif arg_type == Optional[float]:
        parser.add_argument(
            f"--{name}",
            type=lambda x: None if x.lower() in ("none", "null", "") else float(x),
            default=default,
        )
    else:
        parser.add_argument(f"--{name}", type=arg_type, default=default)


def _load_config_file(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON config file into a plain dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML config files.") from exc
    return json.loads(text)


def parse_args() -> TrainingConfig:
    """Build a TrainingConfig from defaults, config file and CLI overrides."""
    defaults = TrainingConfig()
    annotations = get_type_hints(TrainingConfig)

    field_types: Dict[str, Any] = {}
    for f in TrainingConfig.__dataclass_fields__.values():
        if f.name in annotations:
            field_types[f.name] = annotations[f.name]
        elif isinstance(f.default, list):
            field_types[f.name] = list
        else:
            field_types[f.name] = type(f.default)

    # First pass: only read --config_file so it can supply defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_file", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    file_values: Dict[str, Any] = {}
    if pre_args.config_file:
        file_values = _load_config_file(pre_args.config_file)

    # Second pass: full parser with config-file values as defaults.
    parser = argparse.ArgumentParser(
        description="LoRA SFT for causal language models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to a YAML or JSON config file whose values are used as defaults.")

    for f_name, f_type in field_types.items():
        if f_name == "config_file":
            continue
        default_value = file_values.get(f_name, getattr(defaults, f_name))
        _add_argument(parser, f_name, default_value, f_type)

    args = parser.parse_args()
    kwargs = vars(args)
    config_file = kwargs.pop("config_file", None)

    config = TrainingConfig(**kwargs)
    config.config_file = config_file
    return config


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def set_per_device_memory_fraction(max_memory_gb: Optional[float], local_rank: int = -1) -> None:
    """Cap the fraction of GPU memory that this PyTorch process may allocate."""
    if max_memory_gb is None or max_memory_gb <= 0:
        return

    import torch
    logger = logging.getLogger(__name__)
    if not torch.cuda.is_available():
        return

    # Under a distributed launcher each process owns a single device
    # (cuda:local_rank); only cap that one to avoid creating CUDA contexts on
    # every visible GPU. Otherwise cap all visible devices.
    if local_rank >= 0:
        device_indices = [local_rank]
    else:
        device_indices = list(range(torch.cuda.device_count()))

    for idx in device_indices:
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        fraction = min(max_memory_gb / total_gb, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, idx)
        logger.info(
            "GPU %d (total %.1f GB): limiting PyTorch allocation to %.1f GB (fraction %.3f)",
            idx, total_gb, min(max_memory_gb, total_gb), fraction
        )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _build_whitespace_flexible_pattern(text: str) -> str:
    """Build a regex pattern that matches ``text`` but treats whitespace as flexible."""
    parts = []
    for char in text:
        if char.isspace():
            if not parts or parts[-1] != r"\s+":
                parts.append(r"\s+")
        else:
            parts.append(re.escape(char))
    return "".join(parts)


def _find_content_span_flexible(
    text: str,
    content: str,
    search_start: int = 0,
) -> Optional[Tuple[int, int]]:
    """Find ``content`` in ``text`` allowing whitespace differences. Return (start, end) or None."""
    if not content:
        return None
    pattern = _build_whitespace_flexible_pattern(content)
    match = re.search(pattern, text[search_start:])
    if match:
        return match.start() + search_start, match.end() + search_start
    return None


def _find_content_span_whitespace_robust(
    text: str,
    content: str,
    search_start: int = 0,
) -> Optional[Tuple[int, int]]:
    """
    Find ``content`` in ``text`` ignoring all whitespace differences.

    Returns the character span in the *original* ``text``.
    """
    if not content:
        return None

    def normalize_with_map(s: str) -> Tuple[str, List[int]]:
        norm_chars = []
        index_map = []
        for i, char in enumerate(s):
            if not char.isspace():
                norm_chars.append(char)
                index_map.append(i)
        return "".join(norm_chars), index_map

    norm_full, full_map = normalize_with_map(text[search_start:])
    norm_content, _ = normalize_with_map(content)

    pos = norm_full.find(norm_content)
    if pos == -1:
        return None

    # Map back to original character positions.
    orig_start = full_map[pos] + search_start
    orig_end = full_map[pos + len(norm_content) - 1] + 1 + search_start
    return orig_start, orig_end


def _find_assistant_content_spans(
    input_ids: List[int],
    messages: List[Dict[str, str]],
    tokenizer: Any,
    tag_patterns: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, int, int]]:
    """
    Return a list of (turn_idx, tok_start, tok_end) for every assistant turn.

    The span is derived by searching for the assistant's raw ``content`` string
    inside the decoded ``full_text`` and converting the character span back to a
    token span. This avoids both chat-template prefix-stability problems and
    BPE-context tokenization mismatches (the same string can tokenize differently
    in isolation vs. inside a longer sequence).

    The returned span covers only the assistant's content (not the assistant
    header added by the chat template). If a content sequence cannot be found,
    ``tok_start`` and ``tok_end`` are both -1.
    """
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    assistant_indices = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]
    spans: List[Tuple[int, int, int]] = []
    search_start = 0
    logger = logging.getLogger(__name__)

    for turn_idx in assistant_indices:
        content = messages[turn_idx]["content"]
        if not content:
            spans.append((turn_idx, -1, -1))
            continue

        char_start = -1
        char_end = -1

        # Attempt 1: exact substring match.
        char_start = full_text.find(content, search_start)
        if char_start != -1:
            char_end = char_start + len(content)

        # Attempt 2: exact match from beginning (handles overlapping content).
        if char_start == -1:
            char_start = full_text.find(content, 0)
            if char_start != -1:
                char_end = char_start + len(content)

        # Attempt 3: whitespace-robust match. Chat templates often insert or remove
        # blank lines between XML tags (e.g. </think>\n\n<tool_call_security> vs
        # </think><tool_call_security>), so exact matching fails even though the
        # content is otherwise preserved.
        if char_start == -1:
            robust_span = _find_content_span_whitespace_robust(full_text, content, search_start)
            if robust_span is None:
                robust_span = _find_content_span_whitespace_robust(full_text, content, 0)
            if robust_span is not None:
                char_start, char_end = robust_span

        # Attempt 4: regex-based whitespace-flexible match as a safety net.
        if char_start == -1:
            flex_span = _find_content_span_flexible(full_text, content, search_start)
            if flex_span is None:
                flex_span = _find_content_span_flexible(full_text, content, 0)
            if flex_span is not None:
                char_start, char_end = flex_span

        # Attempt 5: Qwen3's chat template extracts the text inside
        # <think>...</think> as reasoning_content. For assistant turns before the
        # final user query it outputs only the text after </think> (with leading
        # newlines stripped), so the raw content can never be found verbatim. Match
        # the post-think text instead.
        if char_start == -1 and "</think>" in content:
            no_think = content.split("</think>")[-1].lstrip("\n")
            if no_think:
                for start_pos in (search_start, 0):
                    no_think_start = full_text.find(no_think, start_pos)
                    if no_think_start != -1:
                        char_start, char_end = no_think_start, no_think_start + len(no_think)
                        break

        # Attempt 6: whitespace-robust match on the post-think text. Chat templates
        # may collapse or insert blank lines between XML tags, so exact matching of
        # the stripped text can still fail even though the characters are present.
        if char_start == -1 and "</think>" in content:
            no_think = content.split("</think>")[-1].lstrip("\n")
            if no_think:
                for start_pos in (search_start, 0):
                    robust_span = _find_content_span_whitespace_robust(full_text, no_think, start_pos)
                    if robust_span is not None:
                        char_start, char_end = robust_span
                        break

        # Attempt 7: search for any configured XML tag block in the content after
        # the previous turn. This is a last-resort anchor for security turns where
        # <tool_call_security> is likely preserved even if the surrounding text was
        # rewritten.
        if char_start == -1 and tag_patterns is not None:
            best_start = -1
            best_end = -1
            for pattern in tag_patterns.values():
                for match in pattern.finditer(full_text, search_start):
                    if best_start == -1 or match.start() < best_start:
                        best_start, best_end = match.start(), match.end()
            if best_start != -1:
                char_start, char_end = best_start, best_end

        if char_start == -1:
            logger.warning(
                "Assistant turn %d: content (len=%d chars) not found in decoded "
                "conversation; this turn will be skipped.",
                turn_idx, len(content),
            )
            spans.append((turn_idx, -1, -1))
            continue

        # Convert character offsets to token offsets by re-encoding prefixes.
        prefix_ids = tokenizer.encode(full_text[:char_start], add_special_tokens=False)
        end_ids = tokenizer.encode(full_text[:char_end], add_special_tokens=False)
        tok_start = len(prefix_ids)
        tok_end = min(len(end_ids), len(input_ids))

        spans.append((turn_idx, tok_start, tok_end))
        search_start = char_end

    return spans


def _decode_turn_text(
    input_ids: List[int],
    tok_start: int,
    tok_end: int,
    tokenizer: Any,
) -> str:
    """Decode a token span back to text for tag searching."""
    if tok_start < 0 or tok_end <= tok_start:
        return ""
    return tokenizer.decode(input_ids[tok_start:tok_end], skip_special_tokens=False)


def _turn_text_char_span_to_token_span(
    turn_text: str,
    char_start: int,
    char_end: int,
    tokenizer: Any,
) -> Tuple[int, int]:
    """Convert a character span inside ``turn_text`` to a token span inside ``turn_text``."""
    if char_start < 0 or char_end <= char_start:
        return 0, 0
    prefix_ids = tokenizer.encode(turn_text[:char_start], add_special_tokens=False)
    end_ids = tokenizer.encode(turn_text[:char_end], add_special_tokens=False)
    return len(prefix_ids), len(end_ids)


def format_messages(
    conversation: List[Dict[str, str]],
    dataset_info: Dict[str, Any],
    tool_role_in_template: str,
) -> List[Dict[str, str]]:
    """Convert a ShareGPT conversation into the chat-template message format."""
    tags = dataset_info.get("tags", {})
    role_tag = tags.get("role_tag", "from")
    content_tag = tags.get("content_tag", "value")

    role_map = {
        tags.get("user_tag", "human"): "user",
        tags.get("assistant_tag", "gpt"): "assistant",
        tags.get("system_tag", "system"): "system",
        tags.get("observation_tag", "tool"): "tool",
    }

    messages = []
    for turn in conversation:
        raw_role = turn.get(role_tag, "")
        role = role_map.get(raw_role, raw_role)
        if role == "tool" and tool_role_in_template == "user":
            role = "user"
        messages.append({"role": role, "content": turn.get(content_tag, "")})
    return messages


def _apply_chat_template_with_fallback(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    enable_thinking: bool,
) -> Dict[str, Any]:
    """Call apply_chat_template, gracefully disabling thinking for non-Qwen models."""
    base_kwargs = {
        "tokenize": True,
        "return_tensors": None,
        "return_dict": True,
        "add_generation_prompt": False,
    }

    if enable_thinking:
        try:
            return tokenizer.apply_chat_template(
                messages,
                **base_kwargs,
                enable_thinking=True,
            )
        except TypeError:
            pass  # Tokenizer does not accept enable_thinking; fall through.

    return tokenizer.apply_chat_template(messages, **base_kwargs)


def _mask_labels_for_assistant_turns(
    input_ids: List[int],
    messages: List[Dict[str, str]],
    tokenizer: Any,
    allowed_turn_indices: Optional[List[int]] = None,
    loss_calc_ce_default_with_security_weight: float = 1.0,
    loss_calc_ce_default_without_security_weight: float = 1.0,
    loss_calc_ce_tool_call_security_tag: Optional[str] = None,
    loss_calc_ce_tool_call_security_with_security_weight: float = 1.0,
    loss_calc_ce_think_with_security_tag: Optional[str] = None,
    loss_calc_ce_think_with_security_weight: float = 1.0,
    loss_calc_ce_tool_call_with_security_tag: Optional[str] = None,
    loss_calc_ce_tool_call_with_security_weight: float = 1.0,
    loss_calc_kl_alpha: float = 0.2,
    loss_calc_kl_think_with_security_alpha: float = 0.1,
    loss_calc_kl_enable_without_security: bool = False,
    loss_calc_kl_enable_with_security: bool = False,
    enable_thinking: bool = True,
) -> Tuple[List[int], List[float], List[float], List[float], List[int], bool, bool]:
    """
    Mask assistant turn tokens for supervised fine-tuning.

    Assistant turns are located by searching for each assistant's raw ``content``
    string inside the decoded ``full_text`` and converting the character span back
    to a token span. This avoids both chat-template prefix-stability problems and
    BPE-context tokenization mismatches (the same string can tokenize differently
    in isolation vs. inside a longer sequence). The masked span covers the
    assistant's content (not the assistant header added by the chat template).

    Tokens in assistant turns without ``<tool_call_security>`` receive a uniform
    ``loss_calc_ce_default_without_security_weight``. Tokens outside the configured XML tags in
    security turns receive ``loss_calc_ce_default_with_security_weight``. Tokens inside
    ``<tool_call_security>...</tool_call_security>`` always use
    ``loss_calc_ce_tool_call_security_with_security_weight``. Tokens inside
    ``<tool_call>...</tool_call>`` and ``<think>...</think>`` use different
    weights depending on whether the assistant turn contains a security block.

    KL anchoring is controlled independently of CE loss by two flags:

    * ``loss_calc_kl_enable_without_security``: enables KL for all non-think
      assistant tokens in turns that do not contain a
      ``<tool_call_security>...</tool_call_security>`` block.
    * ``loss_calc_kl_enable_with_security``: enables KL for assistant turns that
      contain a ``<tool_call_security>...</tool_call_security>`` block. Inside
      those turns, ``<think>...</think>`` receives think KL,
      ``<tool_call_security>...</tool_call_security>`` and
      ``<tool_call>...</tool_call>`` are excluded from KL, and all other
      assistant tokens receive background KL.

    The function returns the label list, the per-token CE loss weight list, the two
    per-token KL weight lists, a per-token security-turn mask, a flag indicating
    whether a security block was found, and a flag indicating whether at least one
    assistant token was trained.
    """
    labels = [-100] * len(input_ids)
    loss_weight = [0.0] * len(input_ids)
    masked_any = False
    logger = logging.getLogger(__name__)

    tag_patterns: Dict[str, Any] = {}
    if loss_calc_ce_tool_call_security_tag is not None:
        tag_patterns[loss_calc_ce_tool_call_security_tag] = re.compile(
            rf"<{re.escape(loss_calc_ce_tool_call_security_tag)}>.*?</{re.escape(loss_calc_ce_tool_call_security_tag)}>", re.DOTALL
        )
    if loss_calc_ce_think_with_security_tag is not None:
        tag_patterns[loss_calc_ce_think_with_security_tag] = re.compile(
            rf"<{re.escape(loss_calc_ce_think_with_security_tag)}>.*?</{re.escape(loss_calc_ce_think_with_security_tag)}>", re.DOTALL
        )
    if loss_calc_ce_tool_call_with_security_tag is not None:
        tag_patterns[loss_calc_ce_tool_call_with_security_tag] = re.compile(
            rf"<{re.escape(loss_calc_ce_tool_call_with_security_tag)}>.*?</{re.escape(loss_calc_ce_tool_call_with_security_tag)}>", re.DOTALL
        )

    security_pattern = tag_patterns.get(loss_calc_ce_tool_call_security_tag)

    # Find each assistant turn's content token span by searching for the raw
    # content token sequence inside input_ids. This avoids all chat-template
    # prefix-stability assumptions.
    turn_spans = _find_assistant_content_spans(input_ids, messages, tokenizer, tag_patterns)
    if allowed_turn_indices is not None:
        turn_spans = [
            (turn_idx, tok_start, tok_end)
            for turn_idx, tok_start, tok_end in turn_spans
            if turn_idx in allowed_turn_indices
        ]

    turn_token_spans: List[Tuple[int, int]] = []
    turn_texts: List[str] = []
    turn_has_security_list: List[bool] = []
    has_security = False
    for turn_idx, tok_start, tok_end in turn_spans:
        turn_token_spans.append((tok_start, tok_end))
        turn_texts.append(_decode_turn_text(input_ids, tok_start, tok_end, tokenizer))

        content = messages[turn_idx]["content"]
        turn_has_security = (
            security_pattern is not None and security_pattern.search(content) is not None
        )
        turn_has_security_list.append(turn_has_security)
        if turn_has_security:
            has_security = True

    for turn_pos, (turn_idx, tok_start, tok_end) in enumerate(turn_spans):
        turn_text = turn_texts[turn_pos]
        tok_start, tok_end = turn_token_spans[turn_pos]
        turn_has_security = turn_has_security_list[turn_pos]
        content = messages[turn_idx]["content"]

        if tok_start >= tok_end:
            continue

        if turn_has_security:
            labels[tok_start:tok_end] = input_ids[tok_start:tok_end]
            loss_weight[tok_start:tok_end] = [loss_calc_ce_default_with_security_weight] * (tok_end - tok_start)
            masked_any = True

            found_tags = set()
            for tag_name, pattern in tag_patterns.items():
                if tag_name == loss_calc_ce_tool_call_security_tag:
                    weight = loss_calc_ce_tool_call_security_with_security_weight
                elif tag_name == loss_calc_ce_think_with_security_tag:
                    weight = loss_calc_ce_think_with_security_weight
                elif tag_name == loss_calc_ce_tool_call_with_security_tag:
                    weight = loss_calc_ce_tool_call_with_security_weight
                else:
                    continue

                tag_found = False
                for match in pattern.finditer(turn_text):
                    rel_start, rel_end = _turn_text_char_span_to_token_span(
                        turn_text, match.start(), match.end(), tokenizer
                    )
                    abs_start = tok_start + rel_start
                    abs_end = tok_start + rel_end
                    if abs_start < abs_end:
                        labels[abs_start:abs_end] = input_ids[abs_start:abs_end]
                        loss_weight[abs_start:abs_end] = [weight] * (abs_end - abs_start)
                        tag_found = True
                if tag_found:
                    found_tags.add(tag_name)

            for tag_name, pattern in tag_patterns.items():
                if pattern.search(content) and tag_name not in found_tags:
                    # When enable_thinking=True, Qwen3's chat template rewrites
                    # <think>...</think> into special tokens, so the literal tag is
                    # not present in the decoded text. Suppress the warning for
                    # think tags in that mode.
                    if tag_name == loss_calc_ce_think_with_security_tag and enable_thinking:
                        continue
                    logger.warning(
                        "Assistant turn %d contains <%s> in source but it was not found "
                        "in the tokenized text; the turn is still trained with the default weight.",
                        turn_idx, tag_name,
                    )
        else:
            labels[tok_start:tok_end] = input_ids[tok_start:tok_end]
            loss_weight[tok_start:tok_end] = [loss_calc_ce_default_without_security_weight] * (tok_end - tok_start)
            masked_any = True

    # Some chat templates append a trailing newline (or other formatting tokens)
    # after the EOS token, so the last sequence token is not always the EOS id.
    # Force the EOS token inside every assistant turn to weight 1.0 so that no
    # matter where formatting tokens land, the turn's closing EOS is trained.
    for tok_start, tok_end in turn_token_spans:
        for idx in range(tok_end - 1, tok_start - 1, -1):
            if input_ids[idx] == tokenizer.eos_token_id:
                labels[idx] = input_ids[idx]
                loss_weight[idx] = 1.0
                break

    # Mark which tokens belong to think blocks, security turns, and the XML
    # blocks that should be excluded from KL anchoring in security turns.
    is_think_token = [False] * len(input_ids)
    is_security_turn_token = [False] * len(input_ids)
    is_tool_call_security_token = [False] * len(input_ids)
    is_tool_call_token = [False] * len(input_ids)
    think_pattern = tag_patterns.get(loss_calc_ce_think_with_security_tag)
    tool_call_security_pattern = tag_patterns.get(loss_calc_ce_tool_call_security_tag)
    tool_call_pattern = tag_patterns.get(loss_calc_ce_tool_call_with_security_tag)
    for turn_pos, (tok_start, tok_end), turn_has_sec in zip(
        range(len(turn_token_spans)), turn_token_spans, turn_has_security_list
    ):
        turn_text = turn_texts[turn_pos]

        if think_pattern is not None:
            for match in think_pattern.finditer(turn_text):
                rel_start, rel_end = _turn_text_char_span_to_token_span(
                    turn_text, match.start(), match.end(), tokenizer
                )
                for idx in range(tok_start + rel_start, tok_start + rel_end):
                    is_think_token[idx] = True

        if tool_call_security_pattern is not None and turn_has_sec:
            for match in tool_call_security_pattern.finditer(turn_text):
                rel_start, rel_end = _turn_text_char_span_to_token_span(
                    turn_text, match.start(), match.end(), tokenizer
                )
                for idx in range(tok_start + rel_start, tok_start + rel_end):
                    is_tool_call_security_token[idx] = True

        if tool_call_pattern is not None and turn_has_sec:
            for match in tool_call_pattern.finditer(turn_text):
                rel_start, rel_end = _turn_text_char_span_to_token_span(
                    turn_text, match.start(), match.end(), tokenizer
                )
                for idx in range(tok_start + rel_start, tok_start + rel_end):
                    is_tool_call_token[idx] = True

        if turn_has_sec:
            for idx in range(tok_start, tok_end):
                is_security_turn_token[idx] = True

    security_turn_mask = [1 if flag else 0 for flag in is_security_turn_token]
    kl_weight_background = [0.0] * len(input_ids)
    kl_weight_think_with_security = [0.0] * len(input_ids)
    for idx, (label, think_flag, sec_turn, tc_sec_flag, tc_flag) in enumerate(
        zip(
            labels,
            is_think_token,
            is_security_turn_token,
            is_tool_call_security_token,
            is_tool_call_token,
        )
    ):
        if label == -100:
            continue

        # tool_call_security and tool_call blocks never participate in KL anchoring,
        # regardless of whether KL is enabled for the turn.
        if tc_sec_flag or tc_flag:
            continue

        if think_flag:
            if sec_turn and loss_calc_kl_enable_with_security:
                kl_weight_think_with_security[idx] = loss_calc_kl_think_with_security_alpha
            elif not sec_turn and loss_calc_kl_enable_without_security:
                kl_weight_think_with_security[idx] = loss_calc_kl_alpha
        else:
            if sec_turn and loss_calc_kl_enable_with_security:
                kl_weight_background[idx] = 1.0
            elif not sec_turn and loss_calc_kl_enable_without_security:
                kl_weight_background[idx] = 1.0

    return labels, loss_weight, kl_weight_background, kl_weight_think_with_security, security_turn_mask, has_security, masked_any


def _select_non_security_samples(
    non_security_samples: List[Dict[str, Any]],
    security_count: int,
    ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Randomly select non-security samples for a single dataset file.

    The target number of non-security samples is ``round(security_count * ratio)``.
    If the available non-security samples are fewer than the target, all of them
    are kept.  The selection is random but reproducible because it uses a
    dedicated ``seed``.
    """
    import random

    if ratio <= 0.0 or not non_security_samples:
        return []
    target = min(int(round(security_count * ratio)), len(non_security_samples))
    if target <= 0:
        return []
    if target >= len(non_security_samples):
        return non_security_samples
    return random.Random(seed).sample(non_security_samples, target)


def build_turn_by_turn_samples(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    cutoff_len: int,
    enable_thinking: bool,
    loss_calc_ce_default_with_security_weight: float = 1.0,
    loss_calc_ce_default_without_security_weight: float = 1.0,
    loss_calc_ce_tool_call_security_tag: Optional[str] = None,
    loss_calc_ce_tool_call_security_with_security_weight: float = 1.0,
    loss_calc_ce_think_with_security_tag: Optional[str] = None,
    loss_calc_ce_think_with_security_weight: float = 1.0,
    loss_calc_ce_tool_call_with_security_tag: Optional[str] = None,
    loss_calc_ce_tool_call_with_security_weight: float = 1.0,
    loss_calc_kl_alpha: float = 0.2,
    loss_calc_kl_think_with_security_alpha: float = 0.1,
    loss_calc_kl_enable_without_security: bool = False,
    loss_calc_kl_enable_with_security: bool = False,
) -> List[Dict[str, List[Any]]]:
    """Create one training sample per assistant turn, preserving prior context."""
    samples = []
    assistant_turn_indices = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]

    security_pattern = None
    if loss_calc_ce_tool_call_security_tag is not None:
        security_pattern = re.compile(
            rf"<{re.escape(loss_calc_ce_tool_call_security_tag)}>.*?</{re.escape(loss_calc_ce_tool_call_security_tag)}>",
            re.DOTALL,
        )

    for turn_idx in assistant_turn_indices:
        content = messages[turn_idx]["content"]
        turn_has_security = (
            security_pattern is not None and security_pattern.search(content) is not None
        )

        prefix_messages = messages[:turn_idx + 1]
        result = _apply_chat_template_with_fallback(tokenizer, prefix_messages, enable_thinking)
        input_ids = result["input_ids"]

        labels, loss_weight, kl_weight_background, kl_weight_think_with_security, security_turn_mask, has_security, masked_any = _mask_labels_for_assistant_turns(
            input_ids,
            prefix_messages,
            tokenizer,
            allowed_turn_indices=[turn_idx],
            loss_calc_ce_default_with_security_weight=loss_calc_ce_default_with_security_weight,
            loss_calc_ce_default_without_security_weight=loss_calc_ce_default_without_security_weight,
            loss_calc_ce_tool_call_security_tag=loss_calc_ce_tool_call_security_tag,
            loss_calc_ce_tool_call_security_with_security_weight=loss_calc_ce_tool_call_security_with_security_weight,
            loss_calc_ce_think_with_security_tag=loss_calc_ce_think_with_security_tag,
            loss_calc_ce_think_with_security_weight=loss_calc_ce_think_with_security_weight,
            loss_calc_ce_tool_call_with_security_tag=loss_calc_ce_tool_call_with_security_tag,
            loss_calc_ce_tool_call_with_security_weight=loss_calc_ce_tool_call_with_security_weight,
            loss_calc_kl_alpha=loss_calc_kl_alpha,
            loss_calc_kl_think_with_security_alpha=loss_calc_kl_think_with_security_alpha,
            loss_calc_kl_enable_without_security=loss_calc_kl_enable_without_security,
            loss_calc_kl_enable_with_security=loss_calc_kl_enable_with_security,
            enable_thinking=enable_thinking,
        )

        # Skip assistant turns that produced no trainable tokens (e.g. empty content).
        if not masked_any:
            continue

        if len(input_ids) > cutoff_len:
            input_ids = input_ids[-cutoff_len:]
            labels = labels[-cutoff_len:]
            loss_weight = loss_weight[-cutoff_len:]
            kl_weight_background = kl_weight_background[-cutoff_len:]
            kl_weight_think_with_security = kl_weight_think_with_security[-cutoff_len:]
            security_turn_mask = security_turn_mask[-cutoff_len:]

        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "loss_weight": loss_weight,
            "kl_weight_background": kl_weight_background,
            "kl_weight_think_with_security": kl_weight_think_with_security,
            "security_turn_mask": security_turn_mask,
            "has_security_block": has_security,
            "assistant_content": content,
        })

    return samples


def make_preprocess_function(
    tokenizer: Any,
    dataset_info: Dict[str, Any],
    config: TrainingConfig,
) -> Any:
    """Return a batched preprocessing function compatible with datasets.map."""
    role_tag = dataset_info.get("tags", {}).get("role_tag", config.role_tag)
    content_tag = dataset_info.get("tags", {}).get("content_tag", config.content_tag)
    msg_column = dataset_info.get("columns", {}).get("messages", "conversations")

    def preprocess(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        input_ids, labels, attention_mask, loss_weight = [], [], [], []
        kl_weight_background, kl_weight_think_with_security, security_turn_mask, has_security_block = [], [], [], []

        for conversation in examples.get(msg_column, []):
            # Some datasets may already use the canonical keys.
            messages = format_messages(
                conversation,
                dataset_info,
                config.tool_role_in_template,
            )

            samples = build_turn_by_turn_samples(
                messages,
                tokenizer,
                config.cutoff_len,
                config.enable_thinking,
                config.loss_calc_ce_default_with_security_weight,
                config.loss_calc_ce_default_without_security_weight,
                config.loss_calc_ce_tool_call_security_tag,
                config.loss_calc_ce_tool_call_security_with_security_weight,
                config.loss_calc_ce_think_with_security_tag,
                config.loss_calc_ce_think_with_security_weight,
                config.loss_calc_ce_tool_call_with_security_tag,
                config.loss_calc_ce_tool_call_with_security_weight,
                config.loss_calc_kl_alpha,
                config.loss_calc_kl_think_with_security_alpha,
                config.loss_calc_kl_enable_without_security,
                config.loss_calc_kl_enable_with_security,
            )

            for sample in samples:
                input_ids.append(sample["input_ids"])
                labels.append(sample["labels"])
                attention_mask.append(sample["attention_mask"])
                loss_weight.append(sample["loss_weight"])
                kl_weight_background.append(sample["kl_weight_background"])
                kl_weight_think_with_security.append(sample["kl_weight_think_with_security"])
                security_turn_mask.append(sample["security_turn_mask"])
                has_security_block.append(sample["has_security_block"])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_weight": loss_weight,
            "kl_weight_background": kl_weight_background,
            "kl_weight_think_with_security": kl_weight_think_with_security,
            "security_turn_mask": security_turn_mask,
            "has_security_block": has_security_block,
        }

    return preprocess


def _interleave_datasets(datasets: List[Any], seed: int) -> Any:
    """
    Interleave multiple datasets proportionally.

    For example, if dataset sizes are 5000 and 1000, the mixed dataset will
    contain 5 samples from the first dataset followed by 1 sample from the
    second dataset, repeating until all samples are consumed.
    """
    from datasets import Dataset

    if len(datasets) == 1:
        return datasets[0].shuffle(seed=seed)

    lists = [ds.shuffle(seed=seed + idx).to_list() for idx, ds in enumerate(datasets)]
    sizes = [len(lst) for lst in lists]
    min_size = min(sizes)
    ratios = [max(1, round(size / min_size)) for size in sizes]

    indices = [0] * len(lists)
    interleaved: List[Dict[str, Any]] = []
    while any(indices[i] < len(lists[i]) for i in range(len(lists))):
        for i in range(len(lists)):
            end = min(indices[i] + ratios[i], len(lists[i]))
            if end > indices[i]:
                interleaved.extend(lists[i][indices[i]:end])
                indices[i] = end

    return Dataset.from_list(interleaved)


def _interleave_sample_lists(
    lists: List[List[Dict[str, Any]]], seed: int
) -> List[Dict[str, Any]]:
    """
    Interleave multiple sample lists proportionally by size.

    For example, if list sizes are [200, 100, 100, 50], the mixed list will
    contain 4 samples from the first list, 2 from the second, 2 from the third
    and 1 from the fourth, repeating until all samples are consumed.  Each
    input list is deterministically shuffled using ``seed + idx``.
    """
    import random

    shuffled: List[List[Dict[str, Any]]] = []
    for idx, lst in enumerate(lists):
        cp = lst[:]
        random.Random(seed + idx).shuffle(cp)
        shuffled.append(cp)

    non_empty = [lst for lst in shuffled if lst]
    if not non_empty:
        return []
    if len(non_empty) == 1:
        return non_empty[0]

    sizes = [len(lst) for lst in non_empty]
    min_size = min(sizes)
    ratios = [max(1, round(size / min_size)) for size in sizes]

    indices = [0] * len(non_empty)
    interleaved: List[Dict[str, Any]] = []
    while any(indices[i] < len(non_empty[i]) for i in range(len(non_empty))):
        for i in range(len(non_empty)):
            end = min(indices[i] + ratios[i], len(non_empty[i]))
            if end > indices[i]:
                interleaved.extend(non_empty[i][indices[i]:end])
                indices[i] = end

    return interleaved


def _write_jsonl(path: Path, samples: List[Dict[str, Any]]) -> None:
    """Write a list of sample dicts to a JSONL file, one JSON object per line."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _generate_turn_samples_for_split(
    split: Any,
    info: Dict[str, Any],
    config: TrainingConfig,
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    """Generate all turn-by-turn training samples from a raw dataset split."""
    from functools import partial

    msg_column = info.get("columns", {}).get("messages", "conversations")

    def _map_fn(
        example: Dict[str, Any],
        info: Dict[str, Any],
        config: TrainingConfig,
        tokenizer: Any,
    ) -> Dict[str, Any]:
        messages = format_messages(
            example[msg_column], info, config.tool_role_in_template
        )
        samples = build_turn_by_turn_samples(
            messages,
            tokenizer,
            config.cutoff_len,
            config.enable_thinking,
            config.loss_calc_ce_default_with_security_weight,
            config.loss_calc_ce_default_without_security_weight,
            config.loss_calc_ce_tool_call_security_tag,
            config.loss_calc_ce_tool_call_security_with_security_weight,
            config.loss_calc_ce_think_with_security_tag,
            config.loss_calc_ce_think_with_security_weight,
            config.loss_calc_ce_tool_call_with_security_tag,
            config.loss_calc_ce_tool_call_with_security_weight,
            config.loss_calc_kl_alpha,
            config.loss_calc_kl_think_with_security_alpha,
            config.loss_calc_kl_enable_without_security,
            config.loss_calc_kl_enable_with_security,
        )
        return {"samples": samples}

    fn = partial(_map_fn, info=info, config=config, tokenizer=tokenizer)
    mapped = split.map(
        fn,
        batched=False,
        num_proc=config.preprocessing_num_workers,
        remove_columns=split.column_names,
        desc="Generating turn-by-turn samples",
    )

    samples: List[Dict[str, Any]] = []
    for item in mapped:
        samples.extend(item["samples"])
    return samples


def _tokenization_fingerprint(config: TrainingConfig, tokenizer: Any, split: str) -> str:
    """
    Build a deterministic fingerprint for a tokenized split.

    Every rank must compute the same value so the ``.map()`` cache file is shared
    across processes. The interleaved dataset is produced via ``Dataset.from_list``
    and therefore carries an in-memory (per-process random) fingerprint; without a
    deterministic cache name each rank re-tokenizes instead of reusing rank 0's
    work, which is what produced the repeated "Tokenizing ..." progress bars.
    """
    payload = {
        "split": split,
        "datasets": list(config.datasets),
        "cutoff_len": config.cutoff_len,
        "loss_calc_ce_default_with_security_weight": config.loss_calc_ce_default_with_security_weight,
        "loss_calc_ce_default_without_security_weight": config.loss_calc_ce_default_without_security_weight,
        "loss_calc_ce_tool_call_security_tag": config.loss_calc_ce_tool_call_security_tag,
        "loss_calc_ce_tool_call_security_with_security_weight": config.loss_calc_ce_tool_call_security_with_security_weight,
        "loss_calc_ce_think_with_security_tag": config.loss_calc_ce_think_with_security_tag,
        "loss_calc_ce_think_with_security_weight": config.loss_calc_ce_think_with_security_weight,
        "loss_calc_ce_tool_call_with_security_tag": config.loss_calc_ce_tool_call_with_security_tag,
        "loss_calc_ce_tool_call_with_security_weight": config.loss_calc_ce_tool_call_with_security_weight,
        "enable_thinking": config.enable_thinking,
        "loss_calc_kl_enabled": config.loss_calc_kl_enabled,
        "loss_calc_kl_alpha": config.loss_calc_kl_alpha,
        "loss_calc_kl_think_with_security_alpha": config.loss_calc_kl_think_with_security_alpha,
        "loss_calc_kl_enable_without_security": config.loss_calc_kl_enable_without_security,
        "loss_calc_kl_enable_with_security": config.loss_calc_kl_enable_with_security,
        "non_security_to_security_turn_ratio": config.non_security_to_security_turn_ratio,
        "non_security_turn_sample_seed": config.non_security_turn_sample_seed,
        "max_samples": config.max_samples,
        "eval_data_ratio": config.eval_data_ratio,
        "shuffle_seed": config.shuffle_seed,
        "tool_role_in_template": config.tool_role_in_template,
        "role_tag": config.role_tag,
        "content_tag": config.content_tag,
        "user_tag": config.user_tag,
        "assistant_tag": config.assistant_tag,
        "system_tag": config.system_tag,
        "observation_tag": config.observation_tag,
        "model_name_or_path": config.model_name_or_path,
        "vocab_size": len(tokenizer),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_datasets(
    config: TrainingConfig,
    tokenizer: Any,
    training_args: Any = None,
) -> Any:
    """Load datasets, split per-file security/non-security turns, interleave, and return tokenized datasets."""
    from datasets import load_dataset
    from contextlib import nullcontext

    dataset_dir = Path(config.dataset_dir)
    with open(config.dataset_info_path, "r", encoding="utf-8") as f:
        all_dataset_info = json.load(f)

    requested_names = config.datasets
    if not requested_names or (len(requested_names) == 1 and requested_names[0].lower() == "all"):
        requested_names = list(all_dataset_info.keys())

    logger = logging.getLogger(__name__)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_datasets = []
    dataset_infos = []
    for name in requested_names:
        if name not in all_dataset_info:
            raise ValueError(f"Dataset '{name}' is not defined in {config.dataset_info_path}")

        info = all_dataset_info[name]
        file_name = info.get("file_name", f"{name}.json")
        file_path = dataset_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        raw = load_dataset("json", data_files=str(file_path), split="train")
        logger.info("Loaded dataset '%s': %d samples", name, len(raw))
        raw_datasets.append(raw)
        dataset_infos.append(info)

    # Split each dataset individually so every dataset contributes eval data.
    train_splits = []
    eval_splits = []
    for raw in raw_datasets:
        if config.eval_data_ratio and config.eval_data_ratio > 0:
            split = raw.train_test_split(test_size=config.eval_data_ratio, seed=config.shuffle_seed)
            train_splits.append(split["train"])
            eval_splits.append(split["test"])
        else:
            train_splits.append(raw)
            eval_splits.append(None)

    train_interleaved_path = output_dir / "train_interleaved_turns.jsonl"
    eval_interleaved_path = output_dir / "eval_interleaved_turns.jsonl"

    # Under distributed training only the main process generates and writes the
    # JSONL preview files; the other ranks wait at the context boundary and then
    # load the same files from disk.
    map_context = (
        training_args.main_process_first(desc="dataset generation")
        if training_args is not None
        else nullcontext()
    )

    with map_context:
        if training_args is None or training_args.process_index == 0:
            train_category_lists: List[List[Dict[str, Any]]] = []
            eval_category_lists: List[List[Dict[str, Any]]] = []

            for name, train_split, eval_split, info in zip(
                requested_names, train_splits, eval_splits, dataset_infos
            ):
                train_samples = _generate_turn_samples_for_split(
                    train_split, info, config, tokenizer
                )
                eval_samples = (
                    _generate_turn_samples_for_split(eval_split, info, config, tokenizer)
                    if eval_split is not None
                    else []
                )

                train_security = [s for s in train_samples if s["has_security_block"]]
                train_non_security = [s for s in train_samples if not s["has_security_block"]]
                eval_security = [s for s in eval_samples if s["has_security_block"]]
                eval_non_security = [s for s in eval_samples if not s["has_security_block"]]

                selected_train_non_security = _select_non_security_samples(
                    train_non_security,
                    len(train_security),
                    config.non_security_to_security_turn_ratio,
                    config.non_security_turn_sample_seed,
                )
                selected_eval_non_security = _select_non_security_samples(
                    eval_non_security,
                    len(eval_security),
                    config.non_security_to_security_turn_ratio,
                    config.non_security_turn_sample_seed,
                )

                security_path = output_dir / f"{name}_security_turns.jsonl"
                non_security_path = output_dir / f"{name}_non_security_turns.jsonl"
                _write_jsonl(security_path, train_security)
                _write_jsonl(non_security_path, selected_train_non_security)

                logger.info(
                    "Wrote %s: %d security turns",
                    security_path, len(train_security),
                )
                logger.info(
                    "Wrote %s: %d non-security turns (selected from %d available, ratio=%.3f)",
                    non_security_path,
                    len(selected_train_non_security),
                    len(train_non_security),
                    config.non_security_to_security_turn_ratio,
                )

                if train_security:
                    train_category_lists.append(train_security)
                if selected_train_non_security:
                    train_category_lists.append(selected_train_non_security)
                if eval_security:
                    eval_category_lists.append(eval_security)
                if selected_eval_non_security:
                    eval_category_lists.append(selected_eval_non_security)

            train_interleaved = _interleave_sample_lists(
                train_category_lists, config.shuffle_seed
            )
            if config.max_samples is not None and config.max_samples > 0:
                max_samples = min(config.max_samples, len(train_interleaved))
                train_interleaved = train_interleaved[:max_samples]
                logger.info("Limited train dataset to %d samples", max_samples)

            _write_jsonl(train_interleaved_path, train_interleaved)
            logger.info(
                "Wrote %s: %d interleaved train turns",
                train_interleaved_path, len(train_interleaved),
            )

            eval_interleaved = _interleave_sample_lists(
                eval_category_lists, config.shuffle_seed
            ) if eval_category_lists else []
            if eval_interleaved:
                _write_jsonl(eval_interleaved_path, eval_interleaved)
                logger.info(
                    "Wrote %s: %d interleaved eval turns",
                    eval_interleaved_path, len(eval_interleaved),
                )

    # All ranks load the generated JSONL files.
    train_dataset = load_dataset("json", data_files=str(train_interleaved_path), split="train")
    if "assistant_content" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns(["assistant_content"])

    eval_dataset = None
    if eval_interleaved_path.exists():
        eval_dataset = load_dataset("json", data_files=str(eval_interleaved_path), split="train")
        if "assistant_content" in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns(["assistant_content"])

    return train_dataset, eval_dataset


# ---------------------------------------------------------------------------
# Model and LoRA helpers
# ---------------------------------------------------------------------------

def get_attn_implementation(config: TrainingConfig) -> str:
    """Resolve the requested attention implementation."""
    if config.flash_attn == "fa2":
        return "flash_attention_2"
    if config.flash_attn == "sdpa":
        return "sdpa"
    if config.flash_attn == "eager":
        return "eager"

    # auto: use flash_attention_2 if the package is available, otherwise SDPA.
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def get_lora_target_modules(config: TrainingConfig) -> Any:
    """Resolve LoRA target modules in a model-agnostic way."""
    if config.lora_target == "all":
        return "all-linear"
    if "," in config.lora_target:
        return [module.strip() for module in config.lora_target.split(",") if module.strip()]
    return config.lora_target


def load_model_and_tokenizer(config: TrainingConfig) -> Any:
    """Load the causal LM and tokenizer with the requested dtype and attention."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if config.chat_template is not None:
        tokenizer.chat_template = config.chat_template

    if tokenizer.chat_template is None:
        raise RuntimeError(
            "The tokenizer has no chat_template. Provide one via --chat_template "
            "or use a model whose tokenizer already defines a chat template."
        )

    attn_impl = get_attn_implementation(config)
    torch_dtype = getattr(torch, config.torch_dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False

    return model, tokenizer


def setup_lora(model: Any, config: TrainingConfig) -> Any:
    """Wrap the model with a PEFT LoRA configuration."""
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = get_lora_target_modules(config)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        use_rslora=config.use_rslora,
    )
    model = get_peft_model(model, lora_config)
    return model


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class _BaseCallback:
    """
    Minimal Trainer callback base providing no-op defaults for every event.

    transformers' CallbackHandler invokes ``getattr(callback, event)`` for all
    ``on_*`` events, so a callback must respond to each one. Subclasses override
    only the events they care about; every other event resolves to a no-op here.
    Defining our own base keeps the heavy transformers import lazy (deferred until
    after CUDA_VISIBLE_DEVICES is set in main()).
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


class LogMetricsCallback(_BaseCallback):
    """
    Route training/eval metrics through the logging module.

    The Trainer's default PrinterCallback only ``print``s metrics to stdout, so
    they never reach ``training.log``. This callback logs the same metric dicts via
    the logger (which writes to both stdout and the file handler). Only the main
    process logs, to avoid every rank duplicating the same lines.
    """

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, float]] = None, **kwargs: Any) -> Any:
        if logs is None or not state.is_world_process_zero:
            return control
        logger = logging.getLogger(__name__)
        if "eval_loss" in logs:
            logger.info("Eval metrics at step %d: %s", state.global_step, logs)
        elif "loss" in logs:
            logger.info("Train metrics at step %d: %s", state.global_step, logs)
        else:
            logger.info("Metrics at step %d: %s", state.global_step, logs)
        return control


class PlotLossCallback(_BaseCallback):
    """Collect losses during training and write PNG plots at the end."""

    def __init__(self, output_dir: str, plot_eval_loss: bool = True):
        self.output_dir = Path(output_dir)
        self.plot_eval_loss = plot_eval_loss
        self.train_steps: List[int] = []
        self.train_losses: List[float] = []
        self.eval_steps: List[int] = []
        self.eval_losses: List[float] = []

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, float]] = None, **kwargs: Any) -> None:
        if logs is None:
            return
        if "loss" in logs:
            self.train_steps.append(state.global_step)
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_steps.append(state.global_step)
            self.eval_losses.append(logs["eval_loss"])

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not self.train_losses:
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logging.warning("matplotlib is not installed; skipping loss plot.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training loss plot.
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_steps, self.train_losses, label="train loss", linewidth=1.5)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        train_plot_path = self.output_dir / "training_loss.png"
        plt.savefig(train_plot_path, dpi=150)
        plt.close()
        logging.info("Training loss plot saved to %s", train_plot_path)

        # Eval loss plot (either standalone or combined).
        if self.plot_eval_loss and self.eval_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_steps, self.train_losses, label="train loss", linewidth=1.5)
            plt.plot(self.eval_steps, self.eval_losses, label="eval loss", linewidth=1.5, marker="o")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training and Evaluation Loss")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            combined_plot_path = self.output_dir / "train_eval_loss.png"
            plt.savefig(combined_plot_path, dpi=150)
            plt.close()
            logging.info("Train/eval loss plot saved to %s", combined_plot_path)

            plt.figure(figsize=(10, 6))
            plt.plot(self.eval_steps, self.eval_losses, label="eval loss", linewidth=1.5, marker="o", color="orange")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Evaluation Loss")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            eval_plot_path = self.output_dir / "eval_loss.png"
            plt.savefig(eval_plot_path, dpi=150)
            plt.close()
            logging.info("Eval loss plot saved to %s", eval_plot_path)


class EvaluateAfterSaveCallback(_BaseCallback):
    """
    Run evaluation immediately after a checkpoint is saved.

    ``trainer.evaluate`` is a distributed collective: EVERY rank must call it, or
    the ranks that skip it leave rank 0 waiting forever inside NCCL (this is the
    hang observed right after the first checkpoint). Therefore all ranks call
    ``evaluate`` together; only rank 0 logs the result.

    When the Trainer's own step-based evaluation already ran at this step (e.g.
    ``eval_steps == save_steps``), evaluating again is pure duplication, so we skip
    it. ``on_evaluate`` records the step of the most recent evaluation and fires
    before ``on_save`` within the same step, which makes the check version-agnostic.
    """

    def __init__(self, trainer: Any, eval_dataset: Any):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self._last_eval_step = -1

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._last_eval_step = state.global_step
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.eval_dataset is None:
            return control
        # The Trainer already evaluated at this exact step; do not repeat it.
        if state.global_step == self._last_eval_step:
            return control
        # All ranks must participate in this collective, so no rank-0 early return.
        metrics = self.trainer.evaluate(eval_dataset=self.eval_dataset)
        if state.is_world_process_zero:
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None:
                logging.getLogger(__name__).info(
                    "Eval loss after checkpoint at step %d: %.4f", state.global_step, eval_loss
                )
        return control


class WeightedDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    Pad ``loss_weight``, ``kl_weight_background``, ``kl_weight_think_with_security``,
    ``security_turn_mask`` and ``has_security_block`` alongside the standard seq2seq fields.

    The parent collator does not know how to pad float weight arrays or the
    per-sample security flag, so we remove them before padding, let the parent
    collator handle the remaining fields, then pad the weights ourselves with
    zeros (masked tokens) and stack the security flag as a 1-D tensor.
    """

    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        weights = [f.pop("loss_weight", None) for f in features]
        kl_bg_weights = [f.pop("kl_weight_background", None) for f in features]
        kl_think_weights = [f.pop("kl_weight_think_with_security", None) for f in features]
        security_masks = [f.pop("security_turn_mask", None) for f in features]
        has_security_flags = [f.pop("has_security_block", False) for f in features]
        try:
            batch = super().torch_call(features)
        finally:
            for feature, weight, kl_bg, kl_think, sec_mask, flag in zip(
                features, weights, kl_bg_weights, kl_think_weights, security_masks, has_security_flags
            ):
                if weight is not None:
                    feature["loss_weight"] = weight
                if kl_bg is not None:
                    feature["kl_weight_background"] = kl_bg
                if kl_think is not None:
                    feature["kl_weight_think_with_security"] = kl_think
                if sec_mask is not None:
                    feature["security_turn_mask"] = sec_mask
                feature["has_security_block"] = flag

        max_length = batch["input_ids"].shape[1]

        if weights[0] is not None:
            import torch

            padded_weights = []
            for weight in weights:
                weight_list = list(weight)
                if len(weight_list) < max_length:
                    weight_list.extend([0.0] * (max_length - len(weight_list)))
                else:
                    weight_list = weight_list[:max_length]
                padded_weights.append(weight_list)
            batch["loss_weight"] = torch.tensor(padded_weights, dtype=torch.float32)

        if kl_bg_weights[0] is not None:
            import torch

            padded_kl_weights = []
            for kl_weight in kl_bg_weights:
                kl_list = list(kl_weight)
                if len(kl_list) < max_length:
                    kl_list.extend([0.0] * (max_length - len(kl_list)))
                else:
                    kl_list = kl_list[:max_length]
                padded_kl_weights.append(kl_list)
            batch["kl_weight_background"] = torch.tensor(padded_kl_weights, dtype=torch.float32)

        if kl_think_weights[0] is not None:
            import torch

            padded_kl_weights = []
            for kl_weight in kl_think_weights:
                kl_list = list(kl_weight)
                if len(kl_list) < max_length:
                    kl_list.extend([0.0] * (max_length - len(kl_list)))
                else:
                    kl_list = kl_list[:max_length]
                padded_kl_weights.append(kl_list)
            batch["kl_weight_think_with_security"] = torch.tensor(padded_kl_weights, dtype=torch.float32)

        if security_masks[0] is not None:
            import torch

            padded_security_masks = []
            for sec_mask in security_masks:
                mask_list = list(sec_mask)
                if len(mask_list) < max_length:
                    mask_list.extend([0] * (max_length - len(mask_list)))
                else:
                    mask_list = mask_list[:max_length]
                padded_security_masks.append(mask_list)
            batch["security_turn_mask"] = torch.tensor(padded_security_masks, dtype=torch.int64)

        batch["has_security_block"] = torch.tensor(
            [bool(flag) for flag in has_security_flags], dtype=torch.int64
        )

        return batch


class WeightedLossTrainer(Trainer):
    """
    Trainer that computes a per-token weighted cross-entropy loss with optional
    KL anchoring against the frozen base weights.

    In assistant turns that contain ``<tool_call_security>...</tool_call_security>``,
    tokens outside the configured XML tags use ``loss_calc_ce_default_with_security_weight``, while
    tokens inside ``<tool_call_security>...</tool_call_security>``,
    ``<tool_call>...</tool_call>`` and ``<think>...</think>`` use their configured
    weights, including the opening and closing XML tags themselves.

    In assistant turns without a security block, all assistant tokens use the uniform
    ``loss_calc_ce_default_without_security_weight`` instead of the per-tag weights.

    When a turn contains ``<tool_call_security>...</tool_call_security>`` and
    ``loss_calc_kl_enable_with_security`` is set, the ``<think>...</think>`` block
    is anchored with ``loss_calc_kl_think_with_security_alpha`` (think KL), all
    other assistant tokens except ``<tool_call_security>...</tool_call_security>``
    and ``<tool_call>...</tool_call>`` are anchored with ``loss_calc_kl_alpha``
    (background KL). Assistant turns without a security block participate in KL
    anchoring when ``loss_calc_kl_enable_without_security`` is set.

    KL anchoring is logged in two separate metrics:

    * ``kl_loss_background`` for non-think assistant tokens inside turns where
      background KL is enabled.
    * ``kl_loss_think_with_security`` for every token inside a
      ``<think>...</think>`` block where think KL is enabled.

    Additionally, a separate ``ce_loss_with_security`` metric is logged: the average
    CE loss over non-zero CE positions that sit inside a security turn. This metric
    does not affect the optimized loss; it is provided for monitoring. The number of
    samples with and without a security block is also reported for each logging
    interval.

    The reference logits are obtained by disabling the LoRA adapter on the same
    model, so no separate reference model is loaded.
    """

    def __init__(
        self,
        loss_calc_kl_alpha: float = 0.2,
        loss_calc_kl_think_with_security_alpha: float = 0.1,
        loss_calc_kl_enabled: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_calc_kl_alpha = loss_calc_kl_alpha
        self.loss_calc_kl_think_with_security_alpha = loss_calc_kl_think_with_security_alpha
        self.loss_calc_kl_enabled = loss_calc_kl_enabled
        self._ce_loss_sum = 0.0
        self._kl_bg_sum = 0.0
        self._kl_think_sum = 0.0
        self._accum_count = 0

        # Metrics for the security-aware CE loss (logging only).
        self._ce_with_security_sum = 0.0
        self._ce_with_security_count = 0
        self._samples_with_security = 0
        self._samples_without_security = 0

        # Metrics accumulated during evaluation (logging only).
        self._eval_ce_with_security_sum = 0.0
        self._eval_ce_with_security_count = 0
        self._eval_samples_with_security = 0
        self._eval_samples_without_security = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs: Any):
        import torch
        import torch.nn.functional as F

        if "loss_weight" not in inputs:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch=num_items_in_batch, **kwargs)

        labels = inputs.pop("labels")
        loss_weight = inputs.pop("loss_weight")
        kl_weight_background = inputs.pop("kl_weight_background", None)
        kl_weight_think_with_security = inputs.pop("kl_weight_think_with_security", None)
        security_turn_mask = inputs.pop("security_turn_mask", None)
        has_security_block = inputs.pop("has_security_block", None)

        outputs = model(**inputs)
        logits = outputs.logits

        # Shift so that each position predicts the next token.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weight = loss_weight[..., 1:].contiguous()
        active_mask = shift_labels != -100
        shift_security_turn_mask = None
        if security_turn_mask is not None:
            shift_security_turn_mask = security_turn_mask[..., 1:].contiguous()

        # Only compute cross-entropy on positions that are not masked.
        ce_loss_with_security = None
        batch_with_security = 0
        batch_without_security = 0
        if active_mask.any():
            active_logits = shift_logits[active_mask].float()
            active_labels = shift_labels[active_mask]
            active_weights = shift_weight[active_mask]

            per_token_loss = F.cross_entropy(
                active_logits,
                active_labels,
                reduction="none",
            )

            weighted_sum = (per_token_loss * active_weights).sum()
            weight_sum = active_weights.sum()

            # CE restricted to tokens that sit inside a tool_call_security turn.
            # This metric is logged separately and does not affect the loss.
            if has_security_block is not None:
                batch_with_security = int((has_security_block == 1).sum().item())
                batch_without_security = has_security_block.numel() - batch_with_security
                if shift_security_turn_mask is not None:
                    security_active_mask = active_mask & (shift_security_turn_mask == 1)
                else:
                    shift_has_security = has_security_block.unsqueeze(1).expand(-1, shift_labels.size(1))
                    security_active_mask = active_mask & (shift_has_security == 1)
                if security_active_mask.any():
                    security_weights = shift_weight[security_active_mask]
                    security_loss_sum = (
                        per_token_loss[security_active_mask[active_mask]] * security_weights
                    ).sum()
                    ce_loss_with_security = security_loss_sum / security_weights.sum()
        else:
            weighted_sum = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
            weight_sum = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        ce_loss = weighted_sum / weight_sum if weight_sum > 0 else weighted_sum
        loss = ce_loss

        kl_bg_loss = None
        kl_think_loss = None

        if self.loss_calc_kl_enabled and model.training:
            unwrapped = self.accelerator.unwrap_model(model)
            if not hasattr(unwrapped, "disable_adapter"):
                raise RuntimeError(
                    "KL anchoring requires a PEFT model with disable_adapter(). "
                    "Make sure finetuning_type is lora."
                )

            with torch.no_grad(), unwrapped.disable_adapter():
                ref_outputs = unwrapped(**inputs)
            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()

            # Background KL: zero-CE-weight assistant tokens that are not part of a think block.
            if kl_weight_background is not None:
                shift_kl_bg = kl_weight_background[..., 1:].contiguous()
                bg_mask = shift_kl_bg > 0
                if bg_mask.any():
                    log_p = F.log_softmax(shift_logits[bg_mask].float(), dim=-1)
                    log_p_ref = F.log_softmax(ref_shift_logits[bg_mask].float(), dim=-1)
                    kl_per_token = (log_p.exp() * (log_p - log_p_ref)).sum(dim=-1)

                    bg_weighted_sum = (kl_per_token * shift_kl_bg[bg_mask]).sum()
                    bg_weight_sum = shift_kl_bg[bg_mask].sum()
                    kl_bg_loss = bg_weighted_sum / bg_weight_sum if bg_weight_sum > 0 else bg_weighted_sum

                    loss = loss + self.loss_calc_kl_alpha * kl_bg_loss

            # Think KL: every token inside a <think>...</think> block.
            # kl_weight_think_with_security stores the per-token KL coefficient:
            # 0.0 when disabled, 0.1 for security turns when enabled, 0.2 for
            # non-security turns when enabled.
            if kl_weight_think_with_security is not None:
                shift_kl_think = kl_weight_think_with_security[..., 1:].contiguous()
                think_mask = shift_kl_think > 0
                if think_mask.any():
                    log_p = F.log_softmax(shift_logits[think_mask].float(), dim=-1)
                    log_p_ref = F.log_softmax(ref_shift_logits[think_mask].float(), dim=-1)
                    kl_per_token = (log_p.exp() * (log_p - log_p_ref)).sum(dim=-1)

                    think_count = think_mask.sum()
                    kl_think_loss = kl_per_token.sum() / think_count

                    weighted_think_sum = (kl_per_token * shift_kl_think[think_mask]).sum()
                    loss = loss + weighted_think_sum / think_count

        # Newer transformers versions pass num_items_in_batch and skip the division
        # by gradient_accumulation_steps in training_step.
        if num_items_in_batch is not None and self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        outputs.loss = loss

        # Accumulate CE/KL and security-aware metrics for the next logging event.
        if model.training:
            self._ce_loss_sum += ce_loss.detach().item()
            self._kl_bg_sum += kl_bg_loss.detach().item() if kl_bg_loss is not None else 0.0
            self._kl_think_sum += kl_think_loss.detach().item() if kl_think_loss is not None else 0.0
            self._accum_count += 1

            if ce_loss_with_security is not None:
                self._ce_with_security_sum += ce_loss_with_security.detach().item()
                self._ce_with_security_count += 1
            self._samples_with_security += batch_with_security
            self._samples_without_security += batch_without_security
        else:
            # Evaluation path: accumulate the same security-aware metrics separately.
            if ce_loss_with_security is not None:
                self._eval_ce_with_security_sum += ce_loss_with_security.detach().item()
                self._eval_ce_with_security_count += 1
            self._eval_samples_with_security += batch_with_security
            self._eval_samples_without_security += batch_without_security

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_step: bool = False) -> None:
        """Inject globally averaged CE/KL and security-aware metrics into the log."""
        if "loss" in logs and self._accum_count > 0:
            import torch
            import torch.distributed as dist

            ce = self._ce_loss_sum / self._accum_count
            kl_bg = self._kl_bg_sum / self._accum_count
            kl_think = self._kl_think_sum / self._accum_count

            if dist.is_initialized() and dist.get_world_size() > 1:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                metrics = torch.tensor([ce, kl_bg, kl_think], device=device, dtype=torch.float32)
                dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
                ce, kl_bg, kl_think = metrics.tolist()

            logs["ce_loss"] = ce
            logs["kl_loss_background"] = kl_bg
            logs["kl_loss_think_with_security"] = kl_think

            # Security-aware CE: average only over steps that actually saw a security block.
            if self._ce_with_security_count > 0:
                ce_sec = self._ce_with_security_sum / self._ce_with_security_count
                if dist.is_initialized() and dist.get_world_size() > 1:
                    ce_sec_t = torch.tensor(ce_sec, device=device, dtype=torch.float32)
                    dist.all_reduce(ce_sec_t, op=dist.ReduceOp.AVG)
                    ce_sec = ce_sec_t.item()
                logs["ce_loss_with_security"] = ce_sec

            with_security = self._samples_with_security
            without_security = self._samples_without_security
            if dist.is_initialized() and dist.get_world_size() > 1:
                counts = torch.tensor(
                    [with_security, without_security], device=device, dtype=torch.int64
                )
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                with_security, without_security = counts.tolist()
            logs["num_samples_with_security"] = with_security
            logs["num_samples_without_security"] = without_security

            self._ce_loss_sum = 0.0
            self._kl_bg_sum = 0.0
            self._kl_think_sum = 0.0
            self._accum_count = 0
            self._ce_with_security_sum = 0.0
            self._ce_with_security_count = 0
            self._samples_with_security = 0
            self._samples_without_security = 0

        if "eval_loss" in logs:
            import torch
            import torch.distributed as dist

            if self._eval_ce_with_security_count > 0:
                ce_sec = self._eval_ce_with_security_sum / self._eval_ce_with_security_count
                if dist.is_initialized() and dist.get_world_size() > 1:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ce_sec_t = torch.tensor(ce_sec, device=device, dtype=torch.float32)
                    dist.all_reduce(ce_sec_t, op=dist.ReduceOp.AVG)
                    ce_sec = ce_sec_t.item()
                logs["eval_ce_loss_with_security"] = ce_sec

            with_security = self._eval_samples_with_security
            without_security = self._eval_samples_without_security
            if dist.is_initialized() and dist.get_world_size() > 1:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                counts = torch.tensor(
                    [with_security, without_security], device=device, dtype=torch.int64
                )
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                with_security, without_security = counts.tolist()
            logs["eval_num_samples_with_security"] = with_security
            logs["eval_num_samples_without_security"] = without_security

            self._eval_ce_with_security_sum = 0.0
            self._eval_ce_with_security_count = 0
            self._eval_samples_with_security = 0
            self._eval_samples_without_security = 0

        super().log(logs, start_step)


def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Return the path of the most recent ``checkpoint-N`` directory under ``output_dir``.

    If no checkpoint directory exists, returns ``None``. The checkpoint number is
    extracted from the directory name so that ``checkpoint-12`` wins over
    ``checkpoint-9``.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoint_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-", 1)[1])
                checkpoint_dirs.append((step, str(item)))
            except (ValueError, IndexError):
                continue

    if not checkpoint_dirs:
        return None

    checkpoint_dirs.sort(key=lambda x: x[0])
    return checkpoint_dirs[-1][1]


def build_training_arguments(config: TrainingConfig) -> Any:
    """Create transformers TrainingArguments from the user config."""
    from transformers import TrainingArguments

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -1 tells transformers to derive step count from num_train_epochs.
    max_steps = config.max_steps if config.max_steps and config.max_steps > 0 else -1
    eval_strategy = config.eval_strategy if config.eval_strategy != "no" else "no"

    return TrainingArguments(
        output_dir=config.output_dir,
        seed=config.seed,
        num_train_epochs=config.num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_dir=config.logging_dir,
        report_to=config.report_to or "none",
        include_num_input_tokens_seen=config.include_num_input_tokens_seen,
        ddp_timeout=config.ddp_timeout,
        deepspeed=config.deepspeed_config,
        bf16=config.torch_dtype == "bfloat16",
        fp16=config.torch_dtype == "float16",
        eval_strategy=eval_strategy,
        eval_steps=config.eval_steps if eval_strategy != "no" else None,
        remove_unused_columns=False,
        load_best_model_at_end=False,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full LoRA SFT pipeline."""
    config = parse_args()

    # GPU selection is handled entirely by CUDA_VISIBLE_DEVICES (see lora-ce-kl-8b-run.sh).
    # Under torchrun/deepspeed each process is pinned to a single GPU via
    # LOCAL_RANK, so nothing to set here.
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Determine process rank so only the main process writes to the log file.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in (-1, 0)

    # Prepare the output directory, then clear it if overwrite is requested. This
    # must happen BEFORE the file handler opens training.log, otherwise we would
    # delete the log file we just started writing.
    output_dir = Path(config.output_dir)
    cleared_output = False
    if is_main_process and config.overwrite_output_dir and output_dir.exists():
        import shutil
        for item in output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        cleared_output = True
    # Every rank ensures the directory exists so the file handler never fails.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging as early as possible so the very first messages are captured.
    # In distributed training only the main process prints to stdout/file; other
    # ranks stay silent for INFO logs to avoid the duplicated output seen with
    # multiple GPUs. Errors/crashes on non-main ranks still surface via stderr.
    # mode="w" truncates training.log on every launch so runs never mix together.
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: List[logging.Handler] = []
    if is_main_process:
        handlers.append(logging.StreamHandler(sys.stdout))
        handlers.append(logging.FileHandler(output_dir / "training.log", mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=log_datefmt,
        handlers=handlers,
        force=True,  # override any logging configured by earlier imports
    )
    logger = logging.getLogger(__name__)

    if cleared_output:
        logger.info("Cleared previous output directory: %s", output_dir)

    logger.info("=" * 60)
    logger.info("Starting LoRA SFT pipeline")
    logger.info("Output directory: %s", config.output_dir)
    logger.info("Log file: %s", output_dir / "training.log")
    logger.info("Visible GPUs (CUDA_VISIBLE_DEVICES): %s", os.environ.get("CUDA_VISIBLE_DEVICES", "not set"))
    logger.info("World size: %d | Local rank: %d", world_size, local_rank)
    logger.info("Training config:\n%s", json.dumps(asdict(config), indent=2, ensure_ascii=False))
    logger.info("=" * 60)

    # Delay heavy ML imports until after CUDA_VISIBLE_DEVICES is set.
    import torch
    from transformers import set_seed

    set_per_device_memory_fraction(config.per_device_max_memory_gb, local_rank)

    set_seed(config.seed)
    logger.info("Random seed set to %d", config.seed)

    logger.info("Loading model and tokenizer from %s", config.model_name_or_path)
    model, tokenizer = load_model_and_tokenizer(config)
    logger.info("Model loaded successfully")
    logger.info("Attention implementation: %s", get_attn_implementation(config))
    logger.info("Tokenizer vocab size: %d", len(tokenizer))
    logger.info("Pad token: %s (id=%s)", tokenizer.pad_token, tokenizer.pad_token_id)

    if config.finetuning_type.lower() == "lora":
        logger.info(
            "Applying LoRA (rank=%d, alpha=%d, target=%s, dropout=%.3f)",
            config.lora_rank,
            config.lora_alpha,
            config.lora_target,
            config.lora_dropout,
        )
        model = setup_lora(model, config)
        if config.gradient_checkpointing:
            model.enable_input_require_grads()
            logger.info("Gradient checkpointing enabled; input tensors now require gradients")
        trainable_params, total_params = model.get_nb_trainable_parameters()
        logger.info(
            "Trainable parameters: %d / %d (%.4f%%)",
            trainable_params,
            total_params,
            100 * trainable_params / total_params,
        )
    else:
        raise ValueError(f"Unsupported finetuning_type: {config.finetuning_type}")

    if config.loss_calc_kl_enabled:
        logger.info(
            "KL anchoring enabled (background alpha=%.3f, think-with-security alpha=%.3f, "
            "enable_without_security=%s, enable_think_with_security=%s); "
            "reference logits obtained via disable_adapter()",
            config.loss_calc_kl_alpha,
            config.loss_calc_kl_think_with_security_alpha,
            config.loss_calc_kl_enable_without_security,
            config.loss_calc_kl_enable_with_security,
        )

    # Build training arguments before loading datasets so tokenization can run on
    # the main process first (other ranks reuse the datasets cache).
    training_args = build_training_arguments(config)

    logger.info("Loading datasets defined in %s", config.dataset_info_path)
    train_dataset, eval_dataset = load_datasets(config, tokenizer, training_args)
    logger.info("Train samples: %d", len(train_dataset))
    if eval_dataset is not None:
        logger.info("Eval samples: %d", len(eval_dataset))

    # Count how many training samples contain a tool_call_security block for the
    # final summary. The column is produced by the preprocessing function.
    train_total_with_security = 0
    train_total_without_security = 0
    if "has_security_block" in train_dataset.column_names:
        train_total_with_security = sum(int(flag) for flag in train_dataset["has_security_block"])
        train_total_without_security = len(train_dataset) - train_total_with_security
        logger.info(
            "Training samples with tool_call_security block: %d, without: %d",
            train_total_with_security,
            train_total_without_security,
        )

    data_collator = WeightedDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
    )
    logger.info(
        "CE weights: non-security uniform=%.2f; security turns: default=%.2f, "
        "<%s>=%.2f, <%s>=%.2f, <%s>=%.2f",
        config.loss_calc_ce_default_without_security_weight,
        config.loss_calc_ce_default_with_security_weight,
        config.loss_calc_ce_tool_call_security_tag or "none",
        config.loss_calc_ce_tool_call_security_with_security_weight,
        config.loss_calc_ce_think_with_security_tag or "none",
        config.loss_calc_ce_think_with_security_weight,
        config.loss_calc_ce_tool_call_with_security_tag or "none",
        config.loss_calc_ce_tool_call_with_security_weight,
    )
    logger.info("Data collator initialized")

    logger.info(
        "Checkpoint will be saved every %d steps; loss will be logged every %d steps",
        config.save_steps,
        config.logging_steps,
    )

    # Always log metrics to the file; optionally add loss plotting on top.
    callbacks: List[Any] = [LogMetricsCallback()]
    if config.plot_loss:
        callbacks.append(PlotLossCallback(config.output_dir, config.plot_eval_loss))

    trainer = WeightedLossTrainer(
        model=model,
        loss_calc_kl_alpha=config.loss_calc_kl_alpha,
        loss_calc_kl_think_with_security_alpha=config.loss_calc_kl_think_with_security_alpha,
        loss_calc_kl_enabled=config.loss_calc_kl_enabled,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    logger.info("Trainer initialized")

    if config.eval_after_save and eval_dataset is not None:
        eval_callback = EvaluateAfterSaveCallback(trainer, eval_dataset)
        trainer.add_callback(eval_callback)
        logger.info("Registered post-checkpoint evaluation callback")

    # Automatically resume from the latest checkpoint in the output directory.
    # If overwrite_output_dir was True the directory was cleared, so no checkpoint
    # will be found and training starts from scratch.
    resume_from_checkpoint = _find_latest_checkpoint(config.output_dir)
    if resume_from_checkpoint is not None:
        logger.info("Latest checkpoint found; resuming training from: %s", resume_from_checkpoint)

    if config.do_train:
        logger.info("Starting training for %.2f epochs", config.num_train_epochs)
        start_time = time.time()
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        elapsed = time.time() - start_time
        logger.info("Training finished")

        # Final evaluation if the last step is not already aligned with save_steps.
        final_train_loss = None
        for log in reversed(trainer.state.log_history):
            if "loss" in log:
                final_train_loss = log["loss"]
                break
        if final_train_loss is not None:
            logger.info("Final train loss: %.4f", final_train_loss)

        final_eval_loss = None
        if config.final_eval and eval_dataset is not None:
            # trainer.evaluate() is a distributed collective: EVERY rank must call
            # it together. Guarding the call with is_world_process_zero makes only
            # rank 0 enter the collective while the others move on, leaving rank 0
            # hung forever inside NCCL. So all ranks evaluate; only rank 0 logs.
            if trainer.state.is_world_process_zero:
                logger.info("Running final evaluation")
            eval_metrics = trainer.evaluate()
            final_eval_loss = eval_metrics.get("eval_loss")
            if final_eval_loss is not None and trainer.state.is_world_process_zero:
                logger.info("Final eval loss: %.4f", final_eval_loss)

        # Print training summary (main process only, to avoid duplicated lines).
        if trainer.state.is_world_process_zero:
            logger.info("=" * 60)
            logger.info("Training summary")
            logger.info("Total steps: %d", trainer.state.global_step)
            logger.info("Total time: %s", str(timedelta(seconds=int(elapsed))))
            logger.info("Final train loss: %s", final_train_loss)
            logger.info("Final eval loss: %s", final_eval_loss)
            logger.info(
                "Training samples with tool_call_security block: %d, without: %d",
                train_total_with_security,
                train_total_without_security,
            )
            logger.info("=" * 60)

        # save_model / save_state must run on all ranks (collective under DeepSpeed).
        logger.info("Saving final model to %s", config.output_dir)
        trainer.save_model(config.output_dir)
        trainer.save_state()
        if trainer.state.is_world_process_zero:
            logger.info("Final model and trainer state saved to %s", config.output_dir)


if __name__ == "__main__":
    main()
