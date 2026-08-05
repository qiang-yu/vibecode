###
# Chat template application for different model types (Qwen3, Llama3, Mistral3).
# Converts OpenAI-format messages + tools into a raw prompt string ready for
# tokenization.
###

from typing import Optional

# Roles not natively understood by model chat templates are mapped here before
# the template is applied.
_ROLE_ALIASES = {
    "developer": "system",  # OpenAI o-series "developer" role -> system
}


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """
    Normalise messages so that every model's chat template can handle them:

    1. Map unknown roles (e.g. "developer") to their canonical equivalents.
    2. Convert list-type content (OpenAI multipart format) to a plain string
       by joining all "text" parts, so templates that expect string content
       don't silently produce an empty string.
    """
    normalized = []
    for msg in messages:
        msg = dict(msg)

        # 1. Role normalisation.
        msg["role"] = _ROLE_ALIASES.get(msg["role"], msg["role"])

        # 2. Content normalisation: list -> string.
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = [
                part["text"]
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            msg["content"] = "\n".join(text_parts)

        normalized.append(msg)
    return normalized


def apply_chat_template(
    messages: list[dict],
    tools: Optional[list[dict]],
    tokenizer,
    model_type: str,
    llama3_template_path: Optional[str] = None,
) -> str:
    """
    Apply the correct chat template for the given model type.
    Returns the raw prompt string (not tokenized).
    """
    model_type = model_type.strip().lower()
    messages = _normalize_messages(messages)

    if model_type == "qwen3":
        return _apply_qwen3(messages, tools, tokenizer)
    elif model_type == "llama3":
        return _apply_llama3(messages, tools, tokenizer, llama3_template_path)
    elif model_type == "mistral3":
        raise NotImplementedError("Mistral3 chat template is not yet implemented.")
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


def _apply_qwen3(messages: list[dict], tools: Optional[list[dict]], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        add_generation_prompt=True,
        tokenize=False,
    )


def _apply_llama3(
    messages: list[dict],
    tools: Optional[list[dict]],
    tokenizer,
    template_path: Optional[str],
) -> str:
    if not template_path:
        raise ValueError(
            "llama3_chat_template path must be set in config for model_type=Llama3"
        )
    with open(template_path, "r", encoding="utf-8") as f:
        template_str = f.read()

    return tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        add_generation_prompt=True,
        tokenize=False,
        chat_template=template_str,
    )
