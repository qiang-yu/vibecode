import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, MessageContentBlock, get_text_content_as_str, text_content_block_from_string


def resolve_detector_model_path(model_name: str) -> str:
    """Returns where to load a detector model from.

    If the `DETECTOR_MODEL_DIR` environment variable is set, the model is loaded from
    `$DETECTOR_MODEL_DIR/<model_name>` (e.g. `/home/me/HuggingFace/protectai/deberta-v3-base-prompt-injection-v2`),
    which is useful on machines without internet access. A missing directory raises an error instead of
    silently falling back to the hub. If the variable is not set, `model_name` is returned unchanged and the
    model is resolved through the Hugging Face hub / cache as usual.
    """
    model_dir = os.getenv("DETECTOR_MODEL_DIR")
    if not model_dir:
        return model_name
    path = os.path.join(os.path.expanduser(model_dir), model_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"DETECTOR_MODEL_DIR is set to '{model_dir}' but the model directory '{path}' does not exist. "
            f"Download '{model_name}' there first (e.g. with `huggingface-cli download {model_name} --local-dir {path}`)."
        )
    return path


@dataclass
class DetectorTask:
    index: int
    message: ChatMessage
    text: str


class PromptInjectionDetector(BasePipelineElement):
    """Detects whether a prompt injection is in the tools outputs.

    Args:
        mode: The mode in which the detector should operate. It can be "message" or "full_conversation".
            In "message" mode, the detector is applied per tool output message, while in "full_conversation"
            mode, the detector is applied to the full conversation so far.
        raise_on_injection: Whether to raise a [AbortAgentError][agentdojo.agent_pipeline.AbortAgentError] if
            an injection is detected.
    """

    def __init__(self, mode: Literal["message", "full_conversation"] = "message", raise_on_injection: bool = False):
        super().__init__()
        self.mode = mode
        self.raise_on_injection = raise_on_injection

    def detect(self, tool_output: str) -> bool | tuple[bool, float]:
        """
        Returns True if a prompt injection is detected, False otherwise.

        Raises:
            AbortAgentError: if an injection is detected and if `self.raise_on_injection` is `True`.

        Returns:
            If the detector is a probabilistic one, it can return a tuple with the first element being a boolean
                indicating whether a prompt injection was detected, and the second element being a float representing
                the confidence score. If the detector is not probabilistic, it should return a boolean.
        """
        raise NotImplementedError("to use PromptInjectionDetector, use a subclass with a detect method")

    def transform(self, tool_output: list[MessageContentBlock]) -> list[MessageContentBlock]:
        return [
            text_content_block_from_string("<Data omitted because a prompt injection was detected>")
            if block["type"] == "text"
            else block
            for block in tool_output
        ]

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args
        if messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args
        messages_to_check = [
            DetectorTask(len(messages) - 1, messages[-1], get_text_content_as_str(messages[-1]["content"]) or "")
        ]
        n_tool_results = 1

        for i, message in reversed(list(enumerate(messages[:-1]))):
            if message["role"] != "tool":
                break
            messages_to_check.append(DetectorTask(i, message, get_text_content_as_str(message["content"]) or ""))
            n_tool_results += 1

        processed_messages = list(messages[:-n_tool_results])
        for task in messages_to_check:
            message = task.message
            if self.mode == "message":
                text = task.text  # is ["content"] by default
            elif self.mode == "full_conversation":
                previous_messages = [m for m in messages[: task.index + 1]]
                text = "\n".join(
                    [
                        f"{m['role']}: {get_text_content_as_str(m['content'])}"
                        for m in previous_messages
                        if m["content"] is not None
                    ]
                )
            else:
                raise ValueError(
                    f"Invalid checking mode: {self.mode} (valid options are 'message' and 'full_conversation')"
                )

            detection_result = self.detect(text)
            score = (
                (1.0 if detection_result else 0.0) if not isinstance(detection_result, tuple) else detection_result[1]
            )
            is_injection = detection_result if not isinstance(detection_result, tuple) else detection_result[0]

            if self.raise_on_injection and is_injection:
                raise AbortAgentError(
                    f"aborting execution because a prompt injection was detected (score: {score}) in message: {text}",
                    list(messages),
                    env,
                )

            if is_injection:
                message["content"] = self.transform(message["content"] or [])
            processed_messages.append(message)

        return query, runtime, env, messages, extra_args


class TransformersBasedPIDetector(PromptInjectionDetector):
    """Uses [`protectai/deberta-v3-base-prompt-injection-v2`](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2)
    to detect prompt injections.

    Args:
        model_name: The name of the model to use for prompt injection detection.
        safe_label: The label that indicates a safe prompt.
        threshold: The threshold for the model's prediction to be considered a prompt injection, e.g. safety_score < threshold means
            an input is classified as a prompt injection.
        mode: The mode in which the detector should operate. It can be 'message' or 'full_conversation'.
        raise_on_injection: Whether to raise an exception if a prompt injection is detected.
        max_length: Maximum number of tokens (including special tokens) fed to the classifier at once.
            Longer inputs are split into chunks instead of being truncated, so an injection hidden at the end
            of a long tool output is still seen by the detector.
        chunk_overlap: Number of tokens shared by consecutive chunks.
        batch_size: Batch size used when classifying the chunks of one input.

    The device is taken from the `DETECTOR_DEVICE` environment variable (e.g. `cuda:3`, `cpu`).
    It defaults to `cuda:0` if CUDA is available, otherwise `cpu`.

    If `DETECTOR_MODEL_DIR` is set, the model is loaded from `$DETECTOR_MODEL_DIR/<model_name>` instead of
    the Hugging Face hub (see [`resolve_detector_model_path`][agentdojo.agent_pipeline.pi_detector.resolve_detector_model_path]).
    """

    def __init__(
        self,
        model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
        safe_label: str = "SAFE",
        threshold: float = 0.5,
        mode: Literal["message", "full_conversation"] = "message",
        raise_on_injection: bool = False,
        max_length: int = 512,
        chunk_overlap: int = 32,
        batch_size: int = 8,
    ) -> None:
        super().__init__(mode=mode, raise_on_injection=raise_on_injection)

        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore
        except ImportError:
            raise ImportError(
                "TransformersBasedPIDetector requires the 'transformers' and 'torch' libraries to be installed. "
                "Please install them using 'pip install agentdojo[transformers]'."
            )

        self.model_name = model_name
        self.safe_label = safe_label
        self.threshold = threshold
        self.batch_size = batch_size
        device_name = os.getenv("DETECTOR_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)
        model_path = resolve_detector_model_path(self.model_name)
        self.pipeline = pipeline("text-classification", model=model_path, device=device, trust_remote_code=True)

        # Never exceed what the model supports; some tokenizers report a huge model_max_length.
        model_limit = getattr(self.pipeline.model.config, "max_position_embeddings", None)
        tokenizer_limit = self.pipeline.tokenizer.model_max_length
        limits = [x for x in (max_length, model_limit, tokenizer_limit) if isinstance(x, int) and 0 < x < 1_000_000]
        self.max_length = min(limits)
        n_special = self.pipeline.tokenizer.num_special_tokens_to_add(pair=False)
        self._chunk_size = max(self.max_length - n_special, 1)
        self._chunk_step = max(self._chunk_size - max(chunk_overlap, 0), 1)

    def _split(self, text: str) -> list[str]:
        """Splits `text` into overlapping chunks that fit in the model's context window."""
        tokenizer = self.pipeline.tokenizer
        ids: list[int] = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        if len(ids) <= self._chunk_size:
            return [text]
        chunks = []
        for start in range(0, len(ids), self._chunk_step):
            chunks.append(tokenizer.decode(ids[start : start + self._chunk_size], skip_special_tokens=True))
            if start + self._chunk_size >= len(ids):
                break
        return chunks

    def detect(self, tool_output: str) -> tuple[bool, float]:
        chunks = self._split(tool_output)
        results: list[dict] = self.pipeline(  # type: ignore
            chunks, batch_size=self.batch_size, truncation=True, max_length=self.max_length
        )
        # An input is as unsafe as its most suspicious chunk.
        safety_score: float = min(
            r["score"] if r["label"].lower() == self.safe_label.lower() else 1 - r["score"] for r in results
        )
        is_injection: bool = safety_score < self.threshold

        return is_injection, safety_score
