###
# Parse raw model output into structured tool calls and content for different
# model types (Qwen3/Hermes and Llama3 JSON formats).
###

import copy
import json
import re
import time
import uuid
from typing import Optional


# ------------------------------------------------------------------ #
# <tool_call_security> stripping utilities
# ------------------------------------------------------------------ #

_SECURITY_BLOCK_RE = re.compile(
    r"<tool_call_security>.*?</tool_call_security>", re.DOTALL
)


def strip_tool_call_security(text: str) -> str:
    """Remove all <tool_call_security>...</tool_call_security> blocks from text."""
    return _SECURITY_BLOCK_RE.sub("", text).strip()


def clean_messages(messages: list[dict]) -> list[dict]:
    """
    Return a deep copy of messages with <tool_call_security> stripped from
    every message's content field (string or list-of-parts).
    """
    result = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        content = msg.get("content")
        if isinstance(content, str) and "<tool_call_security>" in content:
            msg["content"] = strip_tool_call_security(content)
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and "<tool_call_security>" in part.get("text", "")
                ):
                    part["text"] = strip_tool_call_security(part["text"])
        result.append(msg)
    return result


# ------------------------------------------------------------------ #
# Qwen3 / Hermes format
#
# Model output example:
#   <think>...</think>
#   <tool_call>
#   {"name": "func", "arguments": {"k": "v"}}
#   </tool_call><tool_call_security>
#   <tool_name>func</tool_name>
#   ...
#   </tool_call_security>
# ------------------------------------------------------------------ #

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def parse_output_qwen3(model_output: str) -> tuple[list[dict], Optional[str]]:
    """
    Returns (tool_calls, content).
    content includes <tool_call_security> block and any plain text outside
    <tool_call> blocks, with the <think> block stripped.
    """
    matches = list(_TOOL_CALL_RE.finditer(model_output))

    if not matches:
        # Regular response; strip think block from content.
        content = _THINK_RE.sub("", model_output).strip() or None
        return [], content

    # Collect all text outside <tool_call>...</tool_call> spans.
    segments: list[str] = []
    cursor = 0
    for m in matches:
        if m.start() > cursor:
            segments.append(model_output[cursor : m.start()])
        cursor = m.end()
    if cursor < len(model_output):
        segments.append(model_output[cursor:])

    # Remove think block from the outside-tool-call text.
    raw_content = "".join(segments)
    raw_content = _THINK_RE.sub("", raw_content).strip()
    content = raw_content or None

    # Parse each tool call JSON.
    tool_calls = []
    for m in matches:
        json_str = m.group(1).strip()
        try:
            obj = json.loads(json_str)
            args = obj.get("arguments", obj.get("parameters", {}))
            tool_calls.append(
                {
                    "name": obj.get("name", ""),
                    "arguments": json.dumps(args, ensure_ascii=False),
                }
            )
        except json.JSONDecodeError:
            pass

    return tool_calls, content


# ------------------------------------------------------------------ #
# Llama3 JSON format
#
# Model output example:
#   {"name": "func", "parameters": {"k": "v"}}<tool_call_security>
#   <tool_name>func</tool_name>
#   ...
#   </tool_call_security>
# ------------------------------------------------------------------ #

_LLAMA3_BOT_TOKEN = "<|python_tag|>"


def parse_output_llama3(model_output: str) -> tuple[list[dict], Optional[str]]:
    """
    Returns (tool_calls, content).
    """
    starts_with_bot = model_output.startswith(_LLAMA3_BOT_TOKEN)
    starts_with_json = model_output.startswith("{")

    if not starts_with_bot and not starts_with_json:
        return [], model_output.strip() or None

    start_idx = len(_LLAMA3_BOT_TOKEN) if starts_with_bot else 0

    try:
        dec = json.JSONDecoder()
        obj, end_pos = dec.raw_decode(model_output[start_idx:])
        end_pos += start_idx

        name = obj.get("name", "")
        args = obj.get("arguments", obj.get("parameters", {}))
        tool_calls = [
            {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}
        ]

        trailing = model_output[end_pos:].strip()
        content = trailing or None
        return tool_calls, content

    except (json.JSONDecodeError, ValueError):
        return [], model_output.strip() or None


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #

def parse_model_output(
    model_output: str, model_type: str
) -> tuple[list[dict], Optional[str]]:
    """Dispatch to the correct parser based on model_type."""
    mt = model_type.strip().lower()
    if mt == "qwen3":
        return parse_output_qwen3(model_output)
    elif mt == "llama3":
        return parse_output_llama3(model_output)
    else:
        # Fallback: no tool call extraction.
        return [], model_output.strip() or None


# ------------------------------------------------------------------ #
# OpenAI response builder
# ------------------------------------------------------------------ #

def build_chat_completion(
    tool_calls: list[dict],
    content: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    cleaned_input_messages: Optional[list[dict]] = None,
) -> dict:
    message: dict = {"role": "assistant", "content": content}

    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
            }
            for tc in tool_calls
        ]
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"

    resp: dict = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "llm-server",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    # Non-standard convenience field: cleaned history + new assistant message.
    # Historical messages have <tool_call_security> stripped; the new assistant
    # message keeps it so the client can inspect it.
    if cleaned_input_messages is not None:
        resp["messages"] = cleaned_input_messages + [message]

    return resp
