import json
# [Modified by qiangyu]: Iterable is needed by _format_tool_list_for_prompt (tool_filter fix).
from collections.abc import Iterable, Sequence
from typing import overload

import openai
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionReasoningEffort,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import FunctionDefinition
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.local_llm_config import is_local_llm_configured
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    MessageContentBlock,
    get_text_content_as_str,
    text_content_block_from_string,
)


def _tool_call_to_openai(tool_call: FunctionCall) -> ChatCompletionMessageToolCallParam:
    if tool_call.id is None:
        raise ValueError("`tool_call.id` is required for OpenAI")
    return ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        type="function",
        function={
            "name": tool_call.function,
            "arguments": json.dumps(tool_call.args),
        },
    )


_REASONING_MODELS = {"o1", "o3"}


def _is_reasoning_model(model_name: str) -> bool:
    return any(model in model_name for model in _REASONING_MODELS)


def _supports_temperature(model_name: str) -> bool:
    lowered = model_name.lower()
    unsupported_prefixes = ("o1", "o3", "o4", "gpt-5")
    return not lowered.startswith(unsupported_prefixes)


@overload
def _content_blocks_to_openai_content_blocks(
    message: ChatUserMessage | ChatSystemMessage,
) -> list[ChatCompletionContentPartTextParam]: ...


@overload
def _content_blocks_to_openai_content_blocks(
    message: ChatAssistantMessage | ChatToolResultMessage,
) -> list[ChatCompletionContentPartTextParam] | None: ...


def _content_blocks_to_openai_content_blocks(
    message: ChatUserMessage | ChatAssistantMessage | ChatSystemMessage | ChatToolResultMessage,
) -> list[ChatCompletionContentPartTextParam] | None:
    if message["content"] is None:
        return None
    return [ChatCompletionContentPartTextParam(type="text", text=el["content"] or "") for el in message["content"]]


def _message_to_openai(message: ChatMessage, model_name: str) -> ChatCompletionMessageParam:
    match message["role"]:
        case "system":
            import os
            if model_name == "qwen3-max" or is_local_llm_configured():
                # Qwen3-Max does not support developer messages
                return ChatCompletionSystemMessageParam(
                    role="system", content=_content_blocks_to_openai_content_blocks(message)
                )
            else:
                return ChatCompletionDeveloperMessageParam(
                    role="developer", content=_content_blocks_to_openai_content_blocks(message)
                )
        case "user":
            return ChatCompletionUserMessageParam(
                role="user", content=_content_blocks_to_openai_content_blocks(message)
            )
        case "assistant":
            if message["tool_calls"] is not None and len(message["tool_calls"]) > 0:
                tool_calls = [_tool_call_to_openai(tool_call) for tool_call in message["tool_calls"]]
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=_content_blocks_to_openai_content_blocks(message),
                    tool_calls=tool_calls,
                )
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=_content_blocks_to_openai_content_blocks(message),
            )
        case "tool":
            if message["tool_call_id"] is None:
                raise ValueError("`tool_call_id` should be specified for OpenAI.")
            return ChatCompletionToolMessageParam(
                content=message["error"] or _content_blocks_to_openai_content_blocks(message),
                tool_call_id=message["tool_call_id"],
                role="tool",
                name=message["tool_call"].function,  # type: ignore -- this is actually used, and is important!
            )
        case _:
            raise ValueError(f"Invalid message type: {message}")


def _openai_to_tool_call(tool_call: ChatCompletionMessageToolCall) -> FunctionCall:
    return FunctionCall(
        function=tool_call.function.name,
        args=json.loads(tool_call.function.arguments),
        id=tool_call.id,
    )


def _assistant_message_to_content(message: ChatCompletionMessage) -> list[MessageContentBlock] | None:
    if message.content is None:
        return None
    return [text_content_block_from_string(message.content)]


def _openai_to_assistant_message(message: ChatCompletionMessage) -> ChatAssistantMessage:
    if message.tool_calls is not None:
        # tool_calls = [_openai_to_tool_call(tool_call) for tool_call in message.tool_calls]
        # Add tolerance for malformed tool calls from API responses
        tool_calls = []
        for tool_call in message.tool_calls:
            try:
                temp = _openai_to_tool_call(tool_call)
                tool_calls.append(temp)
            except Exception:
                continue
                # message.tool_calls.remove(tool_call)
                
    else:
        tool_calls = None
    return ChatAssistantMessage(role="assistant", content=_assistant_message_to_content(message), tool_calls=tool_calls)


def _function_to_openai(f: Function) -> ChatCompletionToolParam:
    function_definition = FunctionDefinition(
        name=f.name,
        description=f.description,
        parameters=f.parameters.model_json_schema(),
    )
    return ChatCompletionToolParam(type="function", function=function_definition)


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type((openai.BadRequestError, openai.UnprocessableEntityError)),
)
def chat_completion_request(
    client: openai.OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    tools: Sequence[ChatCompletionToolParam],
    reasoning_effort: ChatCompletionReasoningEffort | None,
    temperature: float | None = 0.0,
):
    import os
    max_tokens_env = os.getenv("MAX_INFERENCE_TOKENS", None)
    if max_tokens_env is None:
        max_tokens = NOT_GIVEN
    else:
        max_tokens = int(max_tokens_env)

    effective_temperature = temperature if _supports_temperature(model) else None
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or NOT_GIVEN,
        tool_choice="auto" if tools else NOT_GIVEN,
        temperature=effective_temperature if effective_temperature is not None else NOT_GIVEN,
        reasoning_effort=reasoning_effort or NOT_GIVEN,
        max_tokens=max_tokens,
    )


class OpenAILLM(BasePipelineElement):
    """LLM pipeline element that uses OpenAI's API.

    Args:
        client: The OpenAI client.
        model: The model name.
        temperature: The temperature to use for generation.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        reasoning_effort: ChatCompletionReasoningEffort | None = None,
        temperature: float | None = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.reasoning_effort: ChatCompletionReasoningEffort | None = reasoning_effort

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        completion = chat_completion_request(
            self.client, self.model, openai_messages, openai_tools, self.reasoning_effort, self.temperature
        )
        output = _openai_to_assistant_message(completion.choices[0].message)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args


# [Modified by qiangyu]: helper for OpenAILLMToolFilter. Servers handle the `tools` parameter under
# tool_choice="none" inconsistently, so the filter call sends no `tools` at all and instead writes
# the available tool names and descriptions into the prompt as the single tool list.
def _format_tool_list_for_prompt(functions: Iterable[Function]) -> str:
    lines = ["Available tools:"]
    for f in functions:
        # Keep only the first line of the description to keep the prompt short.
        description = (f.description or "").strip().split("\n")[0]
        lines.append(f"- {f.name}: {description}" if description else f"- {f.name}")
    return "\n".join(lines)


class OpenAILLMToolFilter(BasePipelineElement):
    def __init__(self, prompt: str, client: openai.OpenAI, model: str, reasoning_effort: ChatCompletionReasoningEffort | None = None, temperature: float | None = 0.0) -> None:
        self.prompt = prompt
        self.client = client
        self.model = model
        self.temperature = temperature
        self.reasoning_effort: ChatCompletionReasoningEffort | None = reasoning_effort

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        # [Modified by qiangyu]: build the filter conversation in a separate list. The original
        # `messages` is returned unchanged at the end so the filter Q&A never enters the main LLM's
        # history (see the comment before the return statement).
        filter_messages = [*messages, ChatUserMessage(role="user", content=[text_content_block_from_string(self.prompt)])]
        openai_messages = [_message_to_openai(message, self.model) for message in filter_messages]
        # [Modified by qiangyu]: the filter LLM must see exactly ONE tool list on every backend.
        # Before, the list was sent via the `tools` API parameter with tool_choice="none" (plus
        # chat_template_kwargs for old vLLM). Whether a server renders `tools` under tool_choice="none"
        # differs between servers: vLLM < 0.10 drops them, vLLM's Harmony path for gpt-oss ignores
        # chat_template_kwargs (the filter saw no tools and invented names -> utility 0), and
        # third-party OpenAI-compatible APIs (gpt-4o, deepseek, glm, ...) each behave differently,
        # so the model could see zero or two lists. Now no `tools`, `tool_choice` or
        # chat_template_kwargs are sent for this call; the only tool list is the plain text below.
        # It is added to the request copy only, so the stored history keeps the original prompt and
        # the main LLM's context is unchanged.
        openai_messages[-1] = ChatCompletionUserMessageParam(
            role="user",
            content=self.prompt + "\n\n" + _format_tool_list_for_prompt(runtime.functions.values()),
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=(
                self.temperature
                if self.temperature is not None and _supports_temperature(self.model)
                else NOT_GIVEN
            ),
            reasoning_effort=self.reasoning_effort or NOT_GIVEN,
        )
        output = _openai_to_assistant_message(completion.choices[0].message)

        # for vllm: Qwen3 may put <think>...</think> in content; keep only text after the last </think>
        raw = get_text_content_as_str(output["content"]) if output["content"] is not None else ""
        if "</think>" in raw:
            raw = raw.split("</think>")[-1]
        output_tool_list_str = raw.strip()
        print(f"defence tool_filter: {output_tool_list_str}")

        new_tools = {}
        for tool_name, tool in runtime.functions.items():
            if tool_name in output_tool_list_str:
                new_tools[tool_name] = tool

        runtime.update_functions(new_tools)

        # [Modified by qiangyu]: upstream appended the filter prompt and the filter answer to the
        # history, so the last user instruction the main LLM saw was "output the comma-separated list
        # of tool names". gpt-oss-120b obeyed it and just repeated the tool list instead of calling
        # tools (156 of 157 utility tasks, utility ~0). The filter now only narrows the runtime's
        # tools and returns the history unchanged, so the main LLM sees the same conversation as
        # without a defense, just with fewer tools. The filter answer is still printed above for the
        # logs. Note: camel's count_tokens.get_input_and_output_text_tool_filter assumes the old
        # layout (filter Q&A at messages[2:4]) and does not match results produced by this version.
        return query, runtime, env, messages, extra_args
