from meshagent.agents.agent import AgentSessionContext
from meshagent.agents.context import SessionUsage, SessionUsageCallback
from meshagent.api import Participant, RoomException
from meshagent.tools import Toolkit, ToolContext, FunctionTool
from meshagent.api.messaging import (
    Content,
    LinkContent,
    FileContent,
    JsonContent,
    TextContent,
    EmptyContent,
    ErrorContent,
    _ControlContent,
)
from meshagent.api.messaging import ensure_content
from meshagent.agents.adapter import (
    DEFAULT_MAX_TOOL_CALL_LENGTH,
    DEFAULT_MAX_TOOL_CALL_LINES,
    LLMAdapter,
    LLMModelInfo,
    SteeringCallback,
    ToolResponseAdapter,
)
from meshagent.agents.agent_event_reader import (
    AccumulatingAgentEventReader,
    AgentEventReader,
    AgentEventReaderCallbacks,
    _BufferedToolCall,
)
from meshagent.agents.messages import ToolChoice
from meshagent.api.http import llm_annotation_headers, normalize_llm_annotations
import json
import base64
import mimetypes
from typing import List
from dataclasses import dataclass

from openai import AsyncOpenAI, APIStatusError
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall

import os
from typing import Optional, Any, Callable
from collections.abc import AsyncIterable, Mapping
import copy

import logging
import re
import asyncio
from urllib.parse import unquote_to_bytes, urlparse

from meshagent.openai.proxy import (
    get_client,
    resolve_api_key,
    resolve_base_url,
    resolve_user_agent,
)
from meshagent.openai.tools.usage import (
    normalize_openai_usage,
    preprocess_openai_usage,
    track_otel_usage_metrics,
)
from html_to_markdown import convert

logger = logging.getLogger("openai_agent")
_OPENAI_COMPLETIONS_MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024
_OPENAI_COMPLETIONS_MAX_INLINE_TEXT_BYTES = 1 * 1024 * 1024
_OPENAI_COMPLETIONS_ACCEPTED_ATTACHMENT_TYPES = (
    "image/*",
    "text/*",
    "application/json",
    "application/xhtml+xml",
)


@dataclass(frozen=True)
class _DataUrlAttachment:
    mime_type: str
    data: bytes


def _decode_data_url_attachment(url: str) -> _DataUrlAttachment | None:
    if not url.startswith("data:"):
        return None
    header, separator, payload = url[5:].partition(",")
    if separator == "":
        return None

    parts = [part.strip() for part in header.split(";") if part.strip()]
    mime_type = "text/plain"
    parameter_parts = parts
    if parts and "/" in parts[0]:
        mime_type = parts[0].lower()
        parameter_parts = parts[1:]

    is_base64 = False
    for part in parameter_parts:
        lower = part.lower()
        if lower == "base64":
            is_base64 = True

    try:
        data = (
            base64.b64decode(payload, validate=False)
            if is_base64
            else unquote_to_bytes(payload)
        )
    except Exception:
        return None
    return _DataUrlAttachment(mime_type=mime_type, data=data)


def _encoded_data_url(*, mime_type: str, data: bytes) -> str:
    normalized_mime_type = (mime_type or "application/octet-stream").lower()
    return f"data:{normalized_mime_type};base64,{base64.b64encode(data).decode()}"


def _replace_non_matching(text: str, allowed_chars: str, replacement: str) -> str:
    """
    Replaces every character in `text` that does not match the given
    `allowed_chars` regex set with `replacement`.

    Parameters:
    -----------
    text : str
        The input string on which the replacement is to be done.
    allowed_chars : str
        A string defining the set of allowed characters (part of a character set).
        For example, "a-zA-Z0-9" will keep only letters and digits.
    replacement : str
        The string to replace non-matching characters with.

    Returns:
    --------
    str
        A new string where all characters not in `allowed_chars` are replaced.
    """
    # Build a regex that matches any character NOT in allowed_chars
    pattern = rf"[^{allowed_chars}]"
    return re.sub(pattern, replacement, text)


class OpenAICompletionsAgentEventReader(AccumulatingAgentEventReader):
    def _append_user_text(self, text: str) -> None:
        self._emit_context_message({"role": "user", "content": text})

    def _append_user_content(self, content: list[dict[str, Any]]) -> None:
        parts: list[dict[str, Any]] = []
        for item in content:
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append({"type": "text", "text": text})
            elif item_type == "file":
                url = item.get("url")
                if isinstance(url, str):
                    filename_value = item.get("name")
                    filename = (
                        filename_value.strip()
                        if isinstance(filename_value, str)
                        and filename_value.strip() != ""
                        else "attachment"
                    )
                    data_url = _decode_data_url_attachment(url)
                    if data_url is not None and data_url.mime_type.startswith("image/"):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": _encoded_data_url(
                                        mime_type=data_url.mime_type,
                                        data=data_url.data,
                                    )
                                },
                            }
                        )
                    elif data_url is not None:
                        parts.append(
                            {
                                "type": "text",
                                "text": (
                                    "the user attached "
                                    f"{filename} with mime type "
                                    f"{data_url.mime_type}"
                                ),
                            }
                        )
                    else:
                        parts.append(
                            {"type": "text", "text": f"the user attached a file: {url}"}
                        )
        if not parts:
            parts.append({"type": "text", "text": json.dumps(content)})
        self._emit_context_message({"role": "user", "content": parts})

    def _append_assistant_text(self, *, text: str, phase: str | None) -> None:
        del phase
        self._emit_context_message({"role": "assistant", "content": text})

    def _append_assistant_reasoning(
        self,
        *,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        del metadata
        self._emit_context_message(
            {"role": "assistant", "content": f"Reasoning: {text}"}
        )

    def _append_assistant_file(self, *, url: str) -> None:
        self._emit_context_message(
            {"role": "assistant", "content": f"Generated file: {url}"}
        )

    def _append_thread_event(self, *, event: dict[str, Any]) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "content": json.dumps({"type": "event", "event": event}),
            }
        )

    def _append_tool_call(
        self,
        *,
        tool_call: _BufferedToolCall,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> None:
        call_id = tool_call.call_id or tool_call.item_id
        self._emit_context_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": self._function_name(tool_call=tool_call),
                            "arguments": tool_call.arguments_json(),
                        },
                    }
                ],
            }
        )
        if result is not None or error is not None or tool_call.logs:
            self._emit_context_message(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": self._result_text(
                        result=result,
                        error=error,
                        logs=tool_call.logs,
                    ),
                }
            )

    def _append_image_generation_event(
        self,
        *,
        event_type: str,
        turn_id: str,
        item_id: str,
        call_id: str | None,
        toolkit: str,
        tool: str,
        arguments: dict[str, Any] | None,
        images: list[dict[str, Any]],
        status: str,
    ) -> None:
        item = {
            "type": "image_generation",
            "event_type": event_type,
            "turn_id": turn_id,
            "item_id": item_id,
            "call_id": call_id,
            "toolkit": toolkit,
            "tool": tool,
            "arguments": arguments,
            "images": images,
            "status": status,
        }
        self._emit_context_message({"role": "assistant", "content": json.dumps(item)})

    def _append_audio_generation_event(self, *, message: Any) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "content": json.dumps(message.model_dump(mode="json")),
            }
        )

    def _append_audio_transcription_event(self, *, message: Any) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "content": json.dumps(message.model_dump(mode="json")),
            }
        )

    def _restore_compacted_messages(self, *, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            self._emit_context_message(message)

    @staticmethod
    def _function_name(*, tool_call: _BufferedToolCall) -> str:
        if tool_call.namespace == "openai.responses":
            provider_tool = (
                f"{tool_call.tool}_call"
                if tool_call.toolkit == "openai"
                else f"mcp_{tool_call.tool}"
            )
            return safe_tool_name(provider_tool)
        if tool_call.toolkit in {"", "function", "tool"}:
            return safe_tool_name(tool_call.tool)
        return safe_tool_name(f"{tool_call.toolkit}_{tool_call.tool}")


def safe_tool_name(name: str):
    return _replace_non_matching(name, "a-zA-Z0-9_-", "_")


def _is_html_mime_type(mime_type: str | None) -> bool:
    normalized = (mime_type or "").strip().lower()
    return normalized in {"text/html", "application/xhtml+xml"}


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


class OpenAICompletionsSessionContext(AgentSessionContext):
    @property
    def supports_images(self) -> bool:
        return True

    @property
    def supports_files(self) -> bool:
        return True

    def append_image_message(self, *, mime_type: str, data: bytes) -> dict:
        normalized_mime_type = (mime_type or "application/octet-stream").lower()
        if not normalized_mime_type.startswith("image/"):
            return self._append_attachment_note(
                f"the user attached an unsupported image with mime type {normalized_mime_type}"
            )
        if len(data) > _OPENAI_COMPLETIONS_MAX_INLINE_IMAGE_BYTES:
            return self._append_attachment_note(
                f"the user attached an image ({normalized_mime_type}) that was too large to include"
            )
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _encoded_data_url(
                            mime_type=normalized_mime_type,
                            data=data,
                        )
                    },
                },
            ],
        }
        self.messages.append(message)
        return message

    def append_image_url(self, *, url: str) -> dict:
        data_url = _decode_data_url_attachment(url)
        if data_url is not None:
            return self.append_image_message(
                mime_type=data_url.mime_type,
                data=data_url.data,
            )
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                },
            ],
        }
        self.messages.append(message)
        return message

    def append_file_message(
        self, *, filename: str, mime_type: str, data: bytes
    ) -> dict:
        normalized_mime_type = (mime_type or "application/octet-stream").lower()
        if normalized_mime_type.startswith("image/"):
            return self.append_image_message(mime_type=normalized_mime_type, data=data)

        if (
            normalized_mime_type.startswith("text/")
            or normalized_mime_type == "application/json"
            or normalized_mime_type == "application/xhtml+xml"
        ):
            if len(data) > _OPENAI_COMPLETIONS_MAX_INLINE_TEXT_BYTES:
                return self._append_attachment_note(
                    f"the user attached {filename} ({normalized_mime_type}) but it was too large to include"
                )
            text = (
                convert(_decode_text(data))
                if _is_html_mime_type(normalized_mime_type)
                else _decode_text(data)
            )
            return self._append_attachment_note(
                f"attached file {filename} ({normalized_mime_type}):\n{text}"
            )

        return self._append_attachment_note(
            f"the user attached {filename} with unsupported mime type {normalized_mime_type}"
        )

    def append_file_url(self, *, url: str, filename: str | None = None) -> dict:
        data_url = _decode_data_url_attachment(url)
        if data_url is not None:
            return self.append_file_message(
                filename=filename or "attachment",
                mime_type=data_url.mime_type,
                data=data_url.data,
            )

        parsed_url = urlparse(url)
        guessed_mime_type, _ = mimetypes.guess_type(parsed_url.path)
        normalized_mime_type = (guessed_mime_type or "application/octet-stream").lower()
        if normalized_mime_type.startswith("image/"):
            return self.append_image_url(url=url)
        return self._append_attachment_note(
            f"the user attached a file available at {url}"
        )

    def _append_attachment_note(self, text: str) -> dict:
        message = {"role": "user", "content": text}
        self.messages.append(message)
        return message


async def _consume_streaming_tool_result(
    *, stream: AsyncIterable[Any], event_handler: Optional[Callable[[dict], None]]
) -> Content:
    has_last = False
    last_item: Any = None
    async for item in stream:
        if (
            has_last
            and isinstance(last_item, JsonContent)
            and event_handler is not None
        ):
            event_handler(last_item.json)
        last_item = item
        has_last = True

    if not has_last:
        return ensure_content(None)

    if isinstance(last_item, _ControlContent):
        return ensure_content(None)

    if isinstance(last_item, dict):
        last_type = last_item.get("type")
        if last_type in ("agent.event", "codex.event"):
            return ensure_content(None)

    return ensure_content(last_item)


# Collects a group of tool proxies and manages execution of openai tool calls
class CompletionsToolBundle:
    def __init__(self, toolkits: List[Toolkit]):
        self._toolkits = toolkits
        self._executors = dict[str, Toolkit]()
        self._safe_names = {}

        open_ai_tools = []

        for toolkit in toolkits:
            for v in toolkit.tools:
                k = v.name

                name = safe_tool_name(k)

                if k in self._executors:
                    raise Exception(
                        f"duplicate in bundle '{k}', tool names must be unique."
                    )

                self._executors[k] = toolkit

                self._safe_names[name] = k

                fn = {
                    "name": name,
                    "parameters": {
                        **v.input_schema,
                    },
                    "strict": True,
                }

                if v.defs is not None:
                    fn["parameters"]["$defs"] = v.defs

                schema = {
                    "type": "function",
                    "function": fn,
                }

                open_ai_tools.append(schema)

        if len(open_ai_tools) == 0:
            open_ai_tools = None

        self._open_ai_tools = open_ai_tools

    async def execute(
        self, *, context: ToolContext, tool_call: ChatCompletionMessageToolCall
    ) -> Content | AsyncIterable[Any]:
        function = tool_call.function
        name = function.name
        arguments = json.loads(function.arguments)

        if name not in self._safe_names:
            raise RoomException(f"Invalid tool name {name}, check the name of the tool")

        name = self._safe_names[name]

        if name not in self._executors:
            raise Exception(f"Unregistered tool name {name}")

        proxy = self._executors[name]
        result = await proxy.execute(
            context=context,
            name=name,
            input=JsonContent(json=arguments),
        )
        return result

    def contains(self, name: str) -> bool:
        return name in self._open_ai_tools

    def to_json(self) -> List[dict] | None:
        if self._open_ai_tools is None:
            return None
        return self._open_ai_tools.copy()


# Converts a tool response into a series of messages that can be inserted into the openai context
class OpenAICompletionsToolResponseAdapter(ToolResponseAdapter):
    def __init__(
        self,
        *,
        max_tool_call_length: int = DEFAULT_MAX_TOOL_CALL_LENGTH,
        max_tool_call_lines: int = DEFAULT_MAX_TOOL_CALL_LINES,
    ):
        super().__init__(
            max_tool_call_length=max_tool_call_length,
            max_tool_call_lines=max_tool_call_lines,
        )

    async def to_plain_text(self, *, response: Content) -> str:
        text_file = await self.file_content_to_text_content(content=response)
        if text_file is not None:
            if isinstance(response, FileContent) and _is_html_mime_type(
                response.mime_type
            ):
                text_file = TextContent(text=convert(text_file.text))
            response = self.truncate(content=text_file)
        else:
            response = self.truncate(content=response)
        if isinstance(response, LinkContent):
            return json.dumps(
                {
                    "name": response.name,
                    "url": response.url,
                }
            )

        elif isinstance(response, JsonContent):
            return json.dumps(response.json)

        elif isinstance(response, TextContent):
            return response.text

        elif isinstance(response, FileContent):
            return f"{response.name}"

        elif isinstance(response, EmptyContent):
            return "ok"

        elif isinstance(response, ErrorContent):
            code = f" (code={response.code})" if response.code is not None else ""
            return f"Error{code}: {response.text}"

        # elif isinstance(response, ImageResponse):
        #     context.messages.append({
        #         "role" : "tool",
        #         "content" : "the user will upload the image",
        #         "tool_call_id" : tool_call.id,
        #     })
        #     context.messages.append({
        #         "role" : "user",
        #         "content" : [
        #             { "type" : "text", "text": "this is the image from tool call id {tool_call.id}" },
        #             { "type" : "image_url", "image_url": {"url": response.url, "detail": "auto"} }
        #         ]
        #     })

        elif isinstance(response, dict):
            return json.dumps(response)

        elif isinstance(response, str):
            return response

        elif response is None:
            return "ok"

        else:
            raise Exception(
                "unexpected return type: {type}".format(type=type(response))
            )

    async def create_messages(
        self,
        *,
        context: AgentSessionContext,
        tool_call: Any,
        response: Content,
    ) -> list:
        del context
        message = {
            "role": "tool",
            "content": await self.to_plain_text(response=response),
            "tool_call_id": tool_call.id,
        }

        return [message]


class OpenAICompletionsAdapter(LLMAdapter):
    def __init__(
        self,
        model: str = os.getenv("OPENAI_MODEL"),
        parallel_tool_calls: Optional[bool] = None,
        client: Optional[AsyncOpenAI] = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        user_agent: str | None = None,
        annotations: Mapping[str, object] | None = None,
        max_tool_call_length: int = DEFAULT_MAX_TOOL_CALL_LENGTH,
        max_tool_call_lines: int = DEFAULT_MAX_TOOL_CALL_LINES,
    ):
        self._model = model
        self._parallel_tool_calls = parallel_tool_calls
        self._client = client
        self._base_url = resolve_base_url(base_url)
        self._has_explicit_api_key = isinstance(api_key, str) and api_key.strip() != ""
        self._api_key = resolve_api_key(api_key)
        self._user_agent = resolve_user_agent(user_agent)
        self._annotations = normalize_llm_annotations(annotations)
        self._max_tool_call_length = max_tool_call_length
        self._max_tool_call_lines = max_tool_call_lines

    def with_runtime_api_key(
        self, *, api_key: str | None
    ) -> "OpenAICompletionsAdapter":
        resolved_api_key = resolve_api_key(api_key)
        if (
            self._client is not None
            or self._has_explicit_api_key
            or resolved_api_key is None
        ):
            return self

        return type(self)(
            model=self._model,
            parallel_tool_calls=self._parallel_tool_calls,
            base_url=self._base_url,
            api_key=resolved_api_key,
            user_agent=self._user_agent,
            annotations=self._annotations,
            max_tool_call_length=self._max_tool_call_length,
            max_tool_call_lines=self._max_tool_call_lines,
        )

    def default_model(self) -> str:
        return self._model

    def list_models(self) -> list[LLMModelInfo]:
        model = self.default_model()
        context_window = self.context_window_size(model)
        return [
            LLMModelInfo(
                name=model,
                context_window=(
                    int(context_window) if context_window != float("inf") else None
                ),
                supports_attachments=True,
                accepts=_OPENAI_COMPLETIONS_ACCEPTED_ATTACHMENT_TYPES,
            )
        ]

    def create_session(
        self, *, usage_callback: SessionUsageCallback | None = None
    ) -> AgentSessionContext:
        system_role = "system"
        if self._model.startswith("o1"):
            system_role = "developer"
        elif self._model.startswith("o3"):
            system_role = "developer"
        elif self._model.startswith("o4"):
            system_role = "developer"

        context = OpenAICompletionsSessionContext(
            system_role=system_role,
            usage_callback=usage_callback,
        )

        return context

    def make_agent_event_reader(
        self,
        *,
        emit_message: Callable[[dict[str, Any]], None],
        callbacks: AgentEventReaderCallbacks | None = None,
    ) -> AgentEventReader:
        return OpenAICompletionsAgentEventReader(
            emit_message=emit_message,
            callbacks=callbacks,
        )

    def _store_usage(
        self, *, context: AgentSessionContext, usage: object, model: str
    ) -> None:
        usage_dict = normalize_openai_usage(usage)
        if usage_dict is None:
            return

        flattened_usage = preprocess_openai_usage(model=model, usage=usage_dict)
        if flattened_usage is None:
            return
        context_used_tokens = self._context_used_tokens_from_usage(flattened_usage)
        context.emit_usage_updated(
            SessionUsage(
                model=model,
                usage=dict(flattened_usage),
                context_window_used=context_used_tokens,
            )
        )
        track_otel_usage_metrics(
            model=model,
            provider="openai",
            tokens=flattened_usage,
            annotations=self._annotations,
        )

    @staticmethod
    def _context_used_tokens_from_usage(usage: dict[str, float]) -> int:
        return max(
            0,
            int(
                usage.get("input_tokens", 0.0)
                + usage.get("cached_tokens", 0.0)
                + usage.get("output_tokens", 0.0)
            ),
        )

    def _make_tool_response_adapter(self) -> OpenAICompletionsToolResponseAdapter:
        return OpenAICompletionsToolResponseAdapter(
            max_tool_call_length=self._max_tool_call_length,
            max_tool_call_lines=self._max_tool_call_lines,
        )

    def get_openai_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        return get_client(
            base_url=self._base_url,
            api_key=self._api_key,
            user_agent=self._user_agent,
        )

    def _resolve_tool_choice(
        self,
        *,
        toolkits: list[Toolkit],
        tool_choice: ToolChoice | None,
    ) -> dict[str, Any] | None:
        if tool_choice is None:
            return None

        selected_toolkit = next(
            (
                toolkit
                for toolkit in toolkits
                if toolkit.name == tool_choice.toolkit_name
            ),
            None,
        )
        if selected_toolkit is None:
            raise RoomException(
                f"unknown toolkit in tool_choice: {tool_choice.toolkit_name}"
            )

        selected_tool = next(
            (
                tool
                for tool in selected_toolkit.tools
                if tool.name == tool_choice.tool_name
            ),
            None,
        )
        if selected_tool is None:
            raise RoomException(
                f"unknown tool in tool_choice: {tool_choice.toolkit_name}.{tool_choice.tool_name}"
            )
        if not isinstance(selected_tool, FunctionTool):
            raise RoomException(
                f"tool_choice is not supported for {type(selected_tool).__name__}"
            )

        return {
            "type": "function",
            "function": {"name": safe_tool_name(selected_tool.name)},
        }

    # Takes the current chat context, executes a completion request and processes the response.
    # If a tool calls are requested, invokes the tools, processes the tool calls results, and appends the tool call results to the context
    async def create_response(
        self,
        *,
        model: Optional[str] = None,
        context: AgentSessionContext,
        caller: Participant,
        toolkits: list[Toolkit],
        output_schema: Optional[dict] = None,
        event_handler: Optional[Callable[[dict], None]] = None,
        steering_callback: SteeringCallback | None = None,
        on_behalf_of: Optional[Participant] = None,
        tool_choice: ToolChoice | None = None,
        options: Optional[dict] = None,
    ):
        del model

        context.turn_count += 1
        tool_adapter = self._make_tool_response_adapter()

        try:
            openai = self.get_openai_client()

            tool_bundle = CompletionsToolBundle(
                toolkits=[
                    *toolkits,
                ]
            )
            open_ai_tools = tool_bundle.to_json()

            if open_ai_tools is not None:
                logger.info("OpenAI Tools: %s", json.dumps(open_ai_tools))
            else:
                logger.info("OpenAI Tools: Empty")

            response_schema = output_schema
            response_name = "response"
            context_messages_snapshot = copy.deepcopy(context.messages)
            iteration_committed = False

            while True:
                context_messages_snapshot = copy.deepcopy(context.messages)
                iteration_committed = False
                logger.info(
                    "model: %s, context: %s, output_schema: %s",
                    self._model,
                    context.messages,
                    output_schema,
                )
                ptc = self._parallel_tool_calls
                extra = {}
                if ptc is not None and not self._model.startswith("o"):
                    extra["parallel_tool_calls"] = ptc

                if output_schema is not None:
                    extra["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_name,
                            "schema": response_schema,
                            "strict": True,
                        },
                    }

                request_options = dict(options or {})
                extra_headers = llm_annotation_headers(self._annotations)
                extra_headers.update(request_options.pop("extra_headers", None) or {})
                response: ChatCompletion = await openai.chat.completions.create(
                    n=1,
                    model=self._model,
                    messages=context.messages,
                    tools=open_ai_tools,
                    tool_choice=self._resolve_tool_choice(
                        toolkits=toolkits,
                        tool_choice=tool_choice,
                    ),
                    extra_headers=extra_headers,
                    **extra,
                    **request_options,
                )
                self._store_usage(
                    context=context,
                    usage=response.usage,
                    model=self._model,
                )
                message = response.choices[0].message

                if message.tool_calls is not None:
                    tasks = []

                    async def do_tool_call(tool_call: ChatCompletionMessageToolCall):
                        try:
                            tool_item_id = (
                                tool_call.id if isinstance(tool_call.id, str) else None
                            )

                            def handle_tool_event(event: dict):
                                if event_handler is None:
                                    return
                                if tool_item_id is not None and "item_id" not in event:
                                    event = {**event, "item_id": tool_item_id}
                                event_handler(event)

                            tool_context = ToolContext(
                                caller=caller,
                                on_behalf_of=on_behalf_of,
                                event_handler=handle_tool_event,
                            )
                            tool_response = await tool_bundle.execute(
                                context=tool_context, tool_call=tool_call
                            )
                            if isinstance(tool_response, AsyncIterable):
                                tool_response = await _consume_streaming_tool_result(
                                    stream=tool_response,
                                    event_handler=event_handler,
                                )
                            else:
                                tool_response = ensure_content(tool_response)
                            return await tool_adapter.create_messages(
                                context=context,
                                tool_call=tool_call,
                                response=tool_response,
                            )

                        except Exception as e:
                            if isinstance(e, RoomException):
                                logger.debug(
                                    "unable to complete tool call %s: %s",
                                    tool_call,
                                    e,
                                )
                            else:
                                logger.error(
                                    f"unable to complete tool call {tool_call}",
                                    exc_info=e,
                                )

                            return [
                                {
                                    "role": "tool",
                                    "content": json.dumps(
                                        {"error": f"unable to complete tool call: {e}"}
                                    ),
                                    "tool_call_id": tool_call.id,
                                }
                            ]

                    for tool_call in message.tool_calls:
                        tasks.append(asyncio.create_task(do_tool_call(tool_call)))

                    results = await asyncio.gather(*tasks)

                    appended_outputs = False
                    next_messages: list[Any] = [message]
                    for result in results:
                        if result is not None:
                            outputs = result if isinstance(result, list) else [result]
                            for output in outputs:
                                next_messages.append(output)
                                appended_outputs = True

                    context.messages.extend(next_messages)
                    iteration_committed = True

                    if steering_callback is not None and appended_outputs:
                        if await steering_callback():
                            self.on_turn_steer(
                                context=context,
                                interrupted=False,
                            )
                            continue

                elif message.content is not None:
                    context.messages.append(message)
                    iteration_committed = True
                    content = message.content

                    if response_schema is None:
                        return content

                    # First try to parse the result
                    try:
                        full_response = json.loads(content)
                    # sometimes open ai packs two JSON chunks seperated by newline, check if that's why we couldn't parse
                    except json.decoder.JSONDecodeError:
                        for part in content.splitlines():
                            if len(part.strip()) > 0:
                                full_response = json.loads(part)

                                try:
                                    self.validate(
                                        response=full_response,
                                        output_schema=response_schema,
                                    )
                                except Exception as e:
                                    logger.error(
                                        "recieved invalid response, retrying",
                                        exc_info=e,
                                    )
                                    error = {
                                        "role": "user",
                                        "content": "encountered a validation error with the output: {error}".format(
                                            error=e
                                        ),
                                    }
                                    context.messages.append(error)
                                    continue

                    return full_response
                else:
                    raise RoomException(
                        "Unexpected response from OpenAI {response}".format(
                            response=message
                        )
                    )
        except asyncio.CancelledError:
            if not iteration_committed:
                context.messages.clear()
                context.messages.extend(copy.deepcopy(context_messages_snapshot))
            raise
        except APIStatusError as e:
            raise RoomException(f"Error from OpenAI: {e}")
