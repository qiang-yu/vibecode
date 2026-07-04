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
    conversation_mode: str = "turn_by_turn"  # turn_by_turn | whole
    role_tag: str = "from"
    content_tag: str = "value"
    user_tag: str = "human"
    assistant_tag: str = "gpt"
    system_tag: str = "system"
    observation_tag: str = "tool"
    tool_role_in_template: str = "tool"  # tool | user; used when the chat template lacks a tool role
    loss_calc_default_weight: float = 0.0  # loss weight for assistant content outside special tags (0 = no CE loss)
    loss_calc_tag_tool_call_security: Optional[str] = "tool_call_security"  # weighted security-analysis tag
    loss_calc_tag_tool_call_security_weight: float = 2.0  # loss weight for tool_call_security content
    loss_calc_tag_think: Optional[str] = "think"  # weighted think block
    loss_calc_tag_think_weight: float = 0.0  # loss weight for think content (0 = no CE loss)
    loss_calc_tag_tool_call: Optional[str] = "tool_call"  # weighted tool-call JSON block
    loss_calc_tag_tool_call_weight: float = 1.0  # loss weight for tool_call content
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
    resume_from_checkpoint: Optional[str] = None  # e.g. output_dir/checkpoint-100
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
    kl_anchoring_enabled: bool = False
    kl_anchoring_alpha: float = 0.2

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

def _apply_ct_ids(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    enable_thinking: bool,
    add_generation_prompt: bool,
) -> List[int]:
    """Tokenize messages with the chat template, tolerating enable_thinking absence."""
    kwargs = {"tokenize": True, "add_generation_prompt": add_generation_prompt}
    if enable_thinking:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
        except TypeError:
            pass  # Tokenizer does not accept enable_thinking.
    return tokenizer.apply_chat_template(messages, **kwargs)


def _turn_char_span(
    messages: List[Dict[str, str]],
    target_idx: int,
    tokenizer: Any,
    full_text: str,
    enable_thinking: bool,
) -> Tuple[int, int]:
    """
    Return the [start, end) character span of ``target_idx``'s content in full_text.

    The start is the decoded length of everything before the target turn (with the
    assistant header opened), and the end is the decoded length up to and including
    the target turn. This bounds tag searches to a single assistant turn so that
    only that turn's tokens are trained.
    """
    head_ids = _apply_ct_ids(tokenizer, messages[:target_idx], enable_thinking, add_generation_prompt=True)
    head_text = tokenizer.decode(head_ids, skip_special_tokens=False)
    region_start = len(head_text)

    upto_ids = _apply_ct_ids(tokenizer, messages[:target_idx + 1], enable_thinking, add_generation_prompt=False)
    upto_text = tokenizer.decode(upto_ids, skip_special_tokens=False)
    region_end = len(upto_text)

    full_len = len(full_text)
    region_start = max(0, min(region_start, full_len))
    region_end = max(region_start, min(region_end, full_len))
    return region_start, region_end


def _mask_char_span(
    input_ids: List[int],
    labels: List[int],
    loss_weight: List[float],
    full_text: str,
    char_start: int,
    char_end: int,
    tokenizer: Any,
    weight: float,
) -> bool:
    """
    Mask the tokens that cover ``full_text[char_start:char_end]`` and assign a
    per-token loss weight.

    Prefix-length matching converts character offsets to token offsets. Because
    ``full_text`` is produced by decoding ``input_ids``, re-encoding a prefix of it
    yields exactly the token count up to that character, even when BPE merges
    characters across tag boundaries (e.g. ``><`` becoming one token).
    """
    prefix_ids = tokenizer.encode(full_text[:char_start], add_special_tokens=False)
    end_ids = tokenizer.encode(full_text[:char_end], add_special_tokens=False)
    start = len(prefix_ids)
    end = min(len(end_ids), len(input_ids))
    if start < end:
        labels[start:end] = input_ids[start:end]
        loss_weight[start:end] = [weight] * (end - start)
        return True
    return False


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
    loss_calc_default_weight: float = 1.0,
    loss_calc_tag_tool_call_security: Optional[str] = None,
    loss_calc_tag_tool_call_security_weight: float = 1.0,
    loss_calc_tag_think: Optional[str] = None,
    loss_calc_tag_think_weight: float = 1.0,
    loss_calc_tag_tool_call: Optional[str] = None,
    loss_calc_tag_tool_call_weight: float = 1.0,
    enable_thinking: bool = True,
) -> Tuple[List[int], List[float], List[float], bool]:
    """
    Mask assistant turn tokens for supervised fine-tuning.

    Tokens outside the configured XML tags receive ``loss_calc_default_weight``.
    Tokens inside ``<tool_call_security>...</tool_call_security>``,
    ``<think>...</think>`` and ``<tool_call>...</tool_call>`` are trained with their
    respective weights, including the opening and closing XML tags themselves.

    The returned ``kl_weight`` is derived directly from ``loss_weight``: KL anchoring
    is applied exactly where CE loss is disabled (``loss_weight == 0.0``), so the two
    objectives are mutually exclusive at the token level.

    The function returns the label list, the per-token CE loss weight list, the
    per-token KL anchor weight list, and a flag indicating whether at least one
    assistant token was trained.
    """
    labels = [-100] * len(input_ids)
    loss_weight = [0.0] * len(input_ids)
    masked_any = False
    logger = logging.getLogger(__name__)

    assistant_indices = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]
    if allowed_turn_indices is not None:
        assistant_indices = [i for i in assistant_indices if i in allowed_turn_indices]

    # Decode the token sequence once so character offsets map cleanly back to
    # tokens (full_text always re-tokenizes to input_ids).
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # Build the list of configured XML tags, their weights, and compiled patterns.
    tag_configs = []
    if loss_calc_tag_tool_call_security is not None:
        tag_configs.append((
            loss_calc_tag_tool_call_security,
            loss_calc_tag_tool_call_security_weight,
            re.compile(
                rf"<{re.escape(loss_calc_tag_tool_call_security)}>.*?</{re.escape(loss_calc_tag_tool_call_security)}>", re.DOTALL
            ),
        ))
    if loss_calc_tag_think is not None:
        tag_configs.append((
            loss_calc_tag_think,
            loss_calc_tag_think_weight,
            re.compile(
                rf"<{re.escape(loss_calc_tag_think)}>.*?</{re.escape(loss_calc_tag_think)}>", re.DOTALL
            ),
        ))
    if loss_calc_tag_tool_call is not None:
        tag_configs.append((
            loss_calc_tag_tool_call,
            loss_calc_tag_tool_call_weight,
            re.compile(
                rf"<{re.escape(loss_calc_tag_tool_call)}>.*?</{re.escape(loss_calc_tag_tool_call)}>", re.DOTALL
            ),
        ))

    for turn_idx in assistant_indices:
        content = messages[turn_idx]["content"]
        region_start, region_end = _turn_char_span(
            messages, turn_idx, tokenizer, full_text, enable_thinking
        )

        # Train the entire assistant turn with the default weight first; special
        # tag regions will be overridden with their specific weights afterwards.
        turn_masked = _mask_char_span(
            input_ids, labels, loss_weight, full_text,
            region_start, region_end, tokenizer, loss_calc_default_weight,
        )
        if turn_masked:
            masked_any = True

        found_tags = set()
        for tag_name, weight, pattern in tag_configs:
            tag_found = False
            for match in pattern.finditer(full_text, region_start, region_end):
                _mask_char_span(
                    input_ids, labels, loss_weight, full_text,
                    match.start(), match.end(), tokenizer, weight,
                )
                tag_found = True
            if tag_found:
                found_tags.add(tag_name)

        # Warn once per tag if the source turn carries it but it vanished from
        # the tokenized text (e.g. altered by the chat template).
        for tag_name, _, pattern in tag_configs:
            if pattern.search(content) and tag_name not in found_tags:
                logger.warning(
                    "Assistant turn %d contains <%s> in source but it was not found "
                    "in the tokenized text; the turn is still trained with the default weight.",
                    turn_idx, tag_name,
                )

    # Train on the final EOS token if it is present and not already labeled.
    if labels and labels[-1] == -100 and input_ids[-1] == tokenizer.eos_token_id:
        labels[-1] = input_ids[-1]
        loss_weight[-1] = 1.0

    # KL anchoring is mutually exclusive with CE loss: only assistant-turn tokens
    # that do not contribute to CE loss are anchored to the base model.
    kl_weight = [
        1.0 if (label != -100 and w == 0.0) else 0.0
        for label, w in zip(labels, loss_weight)
    ]

    return labels, loss_weight, kl_weight, masked_any


def build_whole_conversation_sample(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    cutoff_len: int,
    enable_thinking: bool,
    loss_calc_default_weight: float = 1.0,
    loss_calc_tag_tool_call_security: Optional[str] = None,
    loss_calc_tag_tool_call_security_weight: float = 1.0,
    loss_calc_tag_think: Optional[str] = None,
    loss_calc_tag_think_weight: float = 1.0,
    loss_calc_tag_tool_call: Optional[str] = None,
    loss_calc_tag_tool_call_weight: float = 1.0,
) -> Dict[str, List[int]]:
    """Create one training sample from the full conversation."""
    result = _apply_chat_template_with_fallback(tokenizer, messages, enable_thinking)
    input_ids = result["input_ids"]

    labels, loss_weight, kl_weight, _ = _mask_labels_for_assistant_turns(
        input_ids,
        messages,
        tokenizer,
        loss_calc_default_weight=loss_calc_default_weight,
        loss_calc_tag_tool_call_security=loss_calc_tag_tool_call_security,
        loss_calc_tag_tool_call_security_weight=loss_calc_tag_tool_call_security_weight,
        loss_calc_tag_think=loss_calc_tag_think,
        loss_calc_tag_think_weight=loss_calc_tag_think_weight,
        loss_calc_tag_tool_call=loss_calc_tag_tool_call,
        loss_calc_tag_tool_call_weight=loss_calc_tag_tool_call_weight,
        enable_thinking=enable_thinking,
    )

    # Keep the most recent tokens when truncation is required.
    if len(input_ids) > cutoff_len:
        input_ids = input_ids[-cutoff_len:]
        labels = labels[-cutoff_len:]
        loss_weight = loss_weight[-cutoff_len:]
        kl_weight = kl_weight[-cutoff_len:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "loss_weight": loss_weight,
        "kl_weight": kl_weight,
    }


def build_turn_by_turn_samples(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    cutoff_len: int,
    enable_thinking: bool,
    loss_calc_default_weight: float = 1.0,
    loss_calc_tag_tool_call_security: Optional[str] = None,
    loss_calc_tag_tool_call_security_weight: float = 1.0,
    loss_calc_tag_think: Optional[str] = None,
    loss_calc_tag_think_weight: float = 1.0,
    loss_calc_tag_tool_call: Optional[str] = None,
    loss_calc_tag_tool_call_weight: float = 1.0,
) -> List[Dict[str, List[int]]]:
    """Create one training sample per assistant turn, preserving prior context."""
    samples = []
    assistant_turn_indices = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]

    for turn_idx in assistant_turn_indices:
        prefix_messages = messages[:turn_idx + 1]
        result = _apply_chat_template_with_fallback(tokenizer, prefix_messages, enable_thinking)
        input_ids = result["input_ids"]

        labels, loss_weight, kl_weight, masked_any = _mask_labels_for_assistant_turns(
            input_ids,
            prefix_messages,
            tokenizer,
            allowed_turn_indices=[turn_idx],
            loss_calc_default_weight=loss_calc_default_weight,
            loss_calc_tag_tool_call_security=loss_calc_tag_tool_call_security,
            loss_calc_tag_tool_call_security_weight=loss_calc_tag_tool_call_security_weight,
            loss_calc_tag_think=loss_calc_tag_think,
            loss_calc_tag_think_weight=loss_calc_tag_think_weight,
            loss_calc_tag_tool_call=loss_calc_tag_tool_call,
            loss_calc_tag_tool_call_weight=loss_calc_tag_tool_call_weight,
            enable_thinking=enable_thinking,
        )

        # Skip assistant turns that produced no trainable tokens (e.g. empty content).
        if not masked_any:
            continue

        if len(input_ids) > cutoff_len:
            input_ids = input_ids[-cutoff_len:]
            labels = labels[-cutoff_len:]
            loss_weight = loss_weight[-cutoff_len:]
            kl_weight = kl_weight[-cutoff_len:]

        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "loss_weight": loss_weight,
            "kl_weight": kl_weight,
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
        input_ids, labels, attention_mask, loss_weight, kl_weight = [], [], [], [], []

        for conversation in examples.get(msg_column, []):
            # Some datasets may already use the canonical keys.
            messages = format_messages(
                conversation,
                dataset_info,
                config.tool_role_in_template,
            )

            if config.conversation_mode == "turn_by_turn":
                samples = build_turn_by_turn_samples(
                    messages,
                    tokenizer,
                    config.cutoff_len,
                    config.enable_thinking,
                    config.loss_calc_default_weight,
                    config.loss_calc_tag_tool_call_security,
                    config.loss_calc_tag_tool_call_security_weight,
                    config.loss_calc_tag_think,
                    config.loss_calc_tag_think_weight,
                    config.loss_calc_tag_tool_call,
                    config.loss_calc_tag_tool_call_weight,
                )
            else:
                samples = [build_whole_conversation_sample(
                    messages,
                    tokenizer,
                    config.cutoff_len,
                    config.enable_thinking,
                    config.loss_calc_default_weight,
                    config.loss_calc_tag_tool_call_security,
                    config.loss_calc_tag_tool_call_security_weight,
                    config.loss_calc_tag_think,
                    config.loss_calc_tag_think_weight,
                    config.loss_calc_tag_tool_call,
                    config.loss_calc_tag_tool_call_weight,
                )]

            for sample in samples:
                input_ids.append(sample["input_ids"])
                labels.append(sample["labels"])
                attention_mask.append(sample["attention_mask"])
                loss_weight.append(sample["loss_weight"])
                kl_weight.append(sample["kl_weight"])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_weight": loss_weight,
            "kl_weight": kl_weight,
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
        "conversation_mode": config.conversation_mode,
        "cutoff_len": config.cutoff_len,
        "loss_calc_default_weight": config.loss_calc_default_weight,
        "loss_calc_tag_tool_call_security": config.loss_calc_tag_tool_call_security,
        "loss_calc_tag_tool_call_security_weight": config.loss_calc_tag_tool_call_security_weight,
        "loss_calc_tag_think": config.loss_calc_tag_think,
        "loss_calc_tag_think_weight": config.loss_calc_tag_think_weight,
        "loss_calc_tag_tool_call": config.loss_calc_tag_tool_call,
        "loss_calc_tag_tool_call_weight": config.loss_calc_tag_tool_call_weight,
        "enable_thinking": config.enable_thinking,
        "kl_anchoring_enabled": config.kl_anchoring_enabled,
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
    """Load, split, interleave, and tokenize all requested datasets."""
    from datasets import load_dataset
    from contextlib import nullcontext

    dataset_dir = Path(config.dataset_dir)
    with open(config.dataset_info_path, "r", encoding="utf-8") as f:
        all_dataset_info = json.load(f)

    requested_names = config.datasets
    if not requested_names or (len(requested_names) == 1 and requested_names[0].lower() == "all"):
        requested_names = list(all_dataset_info.keys())

    logger = logging.getLogger(__name__)
    raw_datasets = []
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

    # Interleave train and eval splits proportionally.
    logger.info("Interleaving train splits with ratios based on dataset sizes")
    train_dataset = _interleave_datasets(train_splits, config.shuffle_seed)

    eval_dataset = None
    if eval_splits:
        logger.info("Interleaving eval splits with ratios based on dataset sizes")
        eval_dataset = _interleave_datasets(eval_splits, config.shuffle_seed)

    if config.max_samples is not None and config.max_samples > 0:
        max_samples = min(config.max_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(max_samples))
        logger.info("Limited train dataset to %d samples", max_samples)

    # Each dataset may define its own tags; use the first dataset's metadata for
    # preprocessing. Mixed-tag datasets can be extended here if needed.
    first_info = all_dataset_info[requested_names[0]]
    preprocess_fn = make_preprocess_function(tokenizer, first_info, config)

    # Under distributed training only the main process should tokenize; the other
    # ranks then reuse the on-disk datasets cache instead of repeating the work
    # (which is what produced multiple "Tokenizing ..." progress bars).
    #
    # The interleaved dataset lives in memory (Dataset.from_list), so .map() will
    # not cache to disk unless we pass an explicit cache_file_name. We derive that
    # name from a deterministic fingerprint of the tokenization config so every
    # rank computes the SAME path: rank 0 writes it, the other ranks load it.
    cache_dir = Path(config.output_dir) / "tokenized_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_file = str(cache_dir / f"train-{_tokenization_fingerprint(config, tokenizer, 'train')}.arrow")
    eval_cache_file = str(cache_dir / f"eval-{_tokenization_fingerprint(config, tokenizer, 'eval')}.arrow")

    map_context = (
        training_args.main_process_first(desc="dataset tokenization")
        if training_args is not None
        else nullcontext()
    )

    with map_context:
        train_dataset = train_dataset.map(
            preprocess_fn,
            batched=True,
            batch_size=1,
            num_proc=config.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            cache_file_name=train_cache_file,
            desc="Tokenizing train dataset",
        )

        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                preprocess_fn,
                batched=True,
                batch_size=1,
                num_proc=config.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                cache_file_name=eval_cache_file,
                desc="Tokenizing eval dataset",
            )

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
    Pad ``loss_weight`` and ``kl_weight`` alongside the standard seq2seq fields.

    The parent collator does not know how to pad float weight arrays, so we
    remove them before padding, let the parent collator handle the remaining
    fields, then pad the weights ourselves with zeros (masked tokens).
    """

    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        weights = [f.pop("loss_weight", None) for f in features]
        kl_weights = [f.pop("kl_weight", None) for f in features]
        try:
            batch = super().torch_call(features)
        finally:
            for feature, weight, kl_weight in zip(features, weights, kl_weights):
                if weight is not None:
                    feature["loss_weight"] = weight
                if kl_weight is not None:
                    feature["kl_weight"] = kl_weight

        if weights[0] is not None:
            import torch

            max_length = batch["input_ids"].shape[1]
            padded_weights = []
            for weight in weights:
                weight_list = list(weight)
                if len(weight_list) < max_length:
                    weight_list.extend([0.0] * (max_length - len(weight_list)))
                else:
                    weight_list = weight_list[:max_length]
                padded_weights.append(weight_list)
            batch["loss_weight"] = torch.tensor(padded_weights, dtype=torch.float32)

        if kl_weights[0] is not None:
            import torch

            max_length = batch["input_ids"].shape[1]
            padded_kl_weights = []
            for kl_weight in kl_weights:
                kl_list = list(kl_weight)
                if len(kl_list) < max_length:
                    kl_list.extend([0.0] * (max_length - len(kl_list)))
                else:
                    kl_list = kl_list[:max_length]
                padded_kl_weights.append(kl_list)
            batch["kl_weight"] = torch.tensor(padded_kl_weights, dtype=torch.float32)

        return batch


class WeightedLossTrainer(Trainer):
    """
    Trainer that computes a per-token weighted cross-entropy loss with optional
    KL anchoring against the frozen base weights.

    Every assistant turn contributes to the loss. Tokens outside the configured XML
    tags use ``loss_calc_default_weight``, while tokens inside
    ``<tool_call_security>...</tool_call_security>``,
    ``<think>...</think>`` and ``<tool_call>...</tool_call>`` use their
    configured weights, including the opening and closing XML tags themselves. This
    lets the security reasoning tokens receive a higher gradient emphasis than the
    raw tool-call JSON and the rest of the assistant content.

    When KL anchoring is enabled, tokens whose CE loss weight is zero (i.e. assistant
    content outside ``<tool_call_security>`` and ``<tool_call>``) are anchored to the
    base model via ``KL(current || reference)``. The reference logits are obtained by
    disabling the LoRA adapter on the same model, so no separate reference model is
    loaded.
    """

    def __init__(self, kl_alpha: float = 1.0, kl_anchoring_enabled: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_alpha = kl_alpha
        self.kl_anchoring_enabled = kl_anchoring_enabled
        self._ce_loss_sum = 0.0
        self._kl_loss_sum = 0.0
        self._accum_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs: Any):
        import torch
        import torch.nn.functional as F

        if "loss_weight" not in inputs:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch=num_items_in_batch, **kwargs)

        labels = inputs.pop("labels")
        loss_weight = inputs.pop("loss_weight")
        kl_weight = inputs.pop("kl_weight", None)

        outputs = model(**inputs)
        logits = outputs.logits

        # Shift so that each position predicts the next token.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weight = loss_weight[..., 1:].contiguous()

        # Only compute cross-entropy on positions that are not masked. This avoids
        # materializing a full float32 [seq_len, vocab_size] matrix for padding and
        # for user/system/tool tokens, which drastically reduces peak GPU memory.
        active_mask = shift_labels != -100
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
        else:
            weighted_sum = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
            weight_sum = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        # Normalize by the sum of weights so the returned loss is a weighted average
        # per-token cross-entropy, on the same scale as the model's default loss.
        ce_loss = weighted_sum / weight_sum if weight_sum > 0 else weighted_sum
        loss = ce_loss

        # KL anchoring: keep <think>, <tool_call> and ordinary assistant content
        # close to the frozen base model distribution. The reference logits come from
        # the same model with the LoRA adapter disabled.
        kl_loss = None
        if self.kl_anchoring_enabled and kl_weight is not None and model.training:
            shift_kl_weight = kl_weight[..., 1:].contiguous()
            kl_mask = shift_kl_weight > 0
            if kl_mask.any():
                unwrapped = self.accelerator.unwrap_model(model)
                if not hasattr(unwrapped, "disable_adapter"):
                    raise RuntimeError(
                        "KL anchoring requires a PEFT model with disable_adapter(). "
                        "Make sure finetuning_type is lora."
                    )
                with torch.no_grad(), unwrapped.disable_adapter():
                    ref_outputs = unwrapped(**inputs)
                ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()

                # Cast to float32 for stable log-softmax / KL computation.
                log_p = F.log_softmax(shift_logits[kl_mask].float(), dim=-1)
                log_p_ref = F.log_softmax(ref_shift_logits[kl_mask].float(), dim=-1)

                # KL(current || reference) = sum_v p_current(v) * log(p_current(v) / p_ref(v))
                kl_per_token = (log_p.exp() * (log_p - log_p_ref)).sum(dim=-1)

                kl_weighted_sum = (kl_per_token * shift_kl_weight[kl_mask]).sum()
                kl_weight_sum = shift_kl_weight[kl_mask].sum()
                kl_loss = kl_weighted_sum / kl_weight_sum if kl_weight_sum > 0 else kl_weighted_sum

                loss = loss + self.kl_alpha * kl_loss

        # Newer transformers versions pass num_items_in_batch and skip the division
        # by gradient_accumulation_steps in training_step. If we do not divide here,
        # the logged loss and the accumulated gradients are both amplified by the
        # number of accumulation steps.
        if num_items_in_batch is not None and self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        outputs.loss = loss

        # Accumulate CE/KL for the next logging event.
        if model.training:
            self._ce_loss_sum += ce_loss.detach().item()
            self._kl_loss_sum += kl_loss.detach().item() if kl_loss is not None else 0.0
            self._accum_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_step: bool = False) -> None:
        """Inject globally averaged CE/KL into the default training loss log."""
        if "loss" in logs and self._accum_count > 0:
            import torch
            import torch.distributed as dist

            ce = self._ce_loss_sum / self._accum_count
            kl = self._kl_loss_sum / self._accum_count

            if dist.is_initialized() and dist.get_world_size() > 1:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ce_tensor = torch.tensor(ce, device=device)
                kl_tensor = torch.tensor(kl, device=device)
                dist.all_reduce(ce_tensor, op=dist.ReduceOp.AVG)
                dist.all_reduce(kl_tensor, op=dist.ReduceOp.AVG)
                ce = ce_tensor.item()
                kl = kl_tensor.item()

            logs["ce_loss"] = ce
            logs["kl_loss"] = kl
            self._ce_loss_sum = 0.0
            self._kl_loss_sum = 0.0
            self._accum_count = 0
        super().log(logs, start_step)


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
    # mode="w" truncates training.log on every launch so runs never mix together.
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if is_main_process:
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

    if config.kl_anchoring_enabled:
        logger.info(
            "KL anchoring enabled (alpha=%.3f); reference logits obtained via disable_adapter()",
            config.kl_anchoring_alpha,
        )

    # Build training arguments before loading datasets so tokenization can run on
    # the main process first (other ranks reuse the datasets cache).
    training_args = build_training_arguments(config)

    logger.info("Loading datasets defined in %s", config.dataset_info_path)
    train_dataset, eval_dataset = load_datasets(config, tokenizer, training_args)
    logger.info("Train samples: %d", len(train_dataset))
    if eval_dataset is not None:
        logger.info("Eval samples: %d", len(eval_dataset))

    data_collator = WeightedDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
    )
    logger.info(
        "Training on weighted tags: default weight=%.2f, <%s> weight=%.2f, <%s> weight=%.2f, <%s> weight=%.2f",
        config.loss_calc_default_weight,
        config.loss_calc_tag_tool_call_security or "none",
        config.loss_calc_tag_tool_call_security_weight,
        config.loss_calc_tag_think or "none",
        config.loss_calc_tag_think_weight,
        config.loss_calc_tag_tool_call or "none",
        config.loss_calc_tag_tool_call_weight,
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
        kl_alpha=config.kl_anchoring_alpha,
        kl_anchoring_enabled=config.kl_anchoring_enabled,
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

    if config.do_train:
        if config.resume_from_checkpoint:
            logger.info("Resuming training from checkpoint: %s", config.resume_from_checkpoint)
        logger.info("Starting training for %.2f epochs", config.num_train_epochs)
        start_time = time.time()
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
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
            logger.info("=" * 60)

        # save_model / save_state must run on all ranks (collective under DeepSpeed).
        logger.info("Saving final model to %s", config.output_dir)
        trainer.save_model(config.output_dir)
        trainer.save_state()
        if trainer.state.is_world_process_zero:
            logger.info("Final model and trainer state saved to %s", config.output_dir)


if __name__ == "__main__":
    main()
