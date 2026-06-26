from meshagent.agents.agent import AgentSessionContext
from meshagent.agents.context import SessionUsage, SessionUsageCallback
from meshagent.api import Participant, RoomClient, RoomException
from meshagent.api.http import (
    llm_annotation_headers,
    new_client_session,
    normalize_extra_headers,
    normalize_llm_annotations,
)
from meshagent.openai.tools.event_publisher import (
    _OpenAIAgentEventPublisher,
    make_openai_agent_event_publisher,
)
from meshagent.agents.images_dataset import ImagesDataset
from meshagent.agents.messages import AgentMessage, ToolChoice
from meshagent.agents.mcp import (
    MCPServerConfig,
    MCPToolkitClientOptions,
    apply_mcp_proxy_config,
)
from meshagent.tools import (
    Toolkit,
    ToolContext,
    FunctionTool,
    BaseTool,
)
from meshagent.tools.container_shell import ContainerShellTool, ProcessShellTool
from meshagent.tools.storage import StorageToolkit
from meshagent.tools._shell_output import (
    DEFAULT_MAX_LOG_LINE_LENGTH,
    StreamOutputAccumulator,
    collect_output_stream,
)
from meshagent.api.messaging import (
    Content,
    LinkContent,
    FileContent,
    JsonContent,
    TextContent,
    EmptyContent,
    ErrorContent,
    RawOutputsContent,
    _ControlContent,
)

from meshagent.api.messaging import ensure_content
from meshagent.agents.adapter import (
    DEFAULT_MAX_TOOL_CALL_LENGTH,
    DEFAULT_MAX_TOOL_CALL_LINES,
    ToolResponseAdapter,
    LLMAdapter,
    LLMModelInfo,
    SteeringCallback,
    ToolCallApprovalHandler,
    ToolCallApprovalRequest,
    llm_model_pricing,
)
from meshagent.agents.agent_event_reader import (
    AccumulatingAgentEventReader,
    AgentEventReader,
    AgentEventReaderCallbacks,
    _BufferedToolCall,
)

from meshagent.tools.script import DEFAULT_CONTAINER_MOUNT_SPEC

from meshagent.api.specs.service import ContainerMountSpec
from meshagent.api.error_codes import ErrorCode
import json
from collections.abc import AsyncIterable, Awaitable, Mapping
import copy
from typing import Any, List, Literal, cast
from meshagent.openai.proxy import (
    get_client,
    get_logging_httpx_client,
    resolve_api_key,
    resolve_base_url,
    resolve_user_agent,
)
from openai import AsyncOpenAI, NOT_GIVEN, APIError, APIStatusError
from openai._models import construct_type
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseStreamEvent,
    ResponseToolSearchCall,
)
import os
from typing import Optional, Callable
import base64
from dataclasses import dataclass

import logging
import re
import asyncio
import aiohttp
import math
import httpx
from pydantic import BaseModel, model_validator
from opentelemetry import trace
from html_to_markdown import convert
from urllib.parse import unquote_to_bytes, urlparse, urlunparse
from meshagent.openai.tools.usage import (
    add_usage_metrics,
    normalize_openai_usage,
    preprocess_openai_image_generation_usage,
    preprocess_openai_usage,
    track_otel_usage_metrics,
)
import contextlib

logger = logging.getLogger("openai_agent")
tracer = trace.get_tracer("openai.llm.responses")
_MAX_LOGGED_WEBSOCKET_PAYLOAD_CHARS = 128000
_MESHAGENT_ERROR_MESSAGE_HEADER = "x-meshagent-error-message"
_OPENAI_OUT_OF_CREDITS_MESSAGE = (
    "Your account is out of credits. Add credits from the account dashboard "
    "or set up auto reload to continue."
)
_OPENAI_RESPONSES_MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024
_OPENAI_RESPONSES_MAX_INLINE_FILE_BYTES = 32 * 1024 * 1024
_OPENAI_RESPONSES_ACCEPTED_ATTACHMENT_TYPES = (
    "image/*",
    "text/*",
    "application/json",
    "application/pdf",
    "application/xhtml+xml",
)
_OPENAI_RESPONSES_IMAGE_GENERATION_CALL_INPUT_FIELDS = frozenset(
    {
        "background",
        "input_image_mask",
        "model",
        "moderation",
        "output_compression",
        "output_format",
        "partial_images",
        "quality",
        "size",
    }
)
OpenAIResponsesToolSearchMode = Literal["server", "client"]
OpenAIResponsesToolSearchConfig = OpenAIResponsesToolSearchMode | bool | None


@dataclass(frozen=True)
class _DataUrlAttachment:
    mime_type: str
    data: bytes


@dataclass(frozen=True)
class OpenAIResponsesToolSearchRequest:
    item_id: str
    call_id: str | None
    query: str | None
    arguments: Mapping[str, Any]

    @classmethod
    def from_call(
        cls, call: ResponseToolSearchCall
    ) -> "OpenAIResponsesToolSearchRequest":
        arguments = call.arguments if isinstance(call.arguments, dict) else {}
        query_value = arguments.get("query")
        query = query_value if isinstance(query_value, str) else None
        return cls(
            item_id=call.id,
            call_id=call.call_id,
            query=query,
            arguments=arguments,
        )


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


def _is_openai_responses_inline_file_mime_type(mime_type: str) -> bool:
    normalized_mime_type = (mime_type or "application/octet-stream").lower()
    return (
        normalized_mime_type.startswith("text/")
        or normalized_mime_type == "application/json"
        or normalized_mime_type == "application/pdf"
        or normalized_mime_type == "application/xhtml+xml"
    )


def _is_openai_out_of_credits_message(message: str) -> bool:
    return "out of credits" in message.lower()


def _is_openai_non_retryable_request_error_message(message: str) -> bool:
    normalized = message.lower()
    return (
        "invalid schema" in normalized
        or "is not valid under any of the given schemas" in normalized
        or "missing required parameter" in normalized
        or "unknown parameter" in normalized
        or "unsupported parameter" in normalized
    )


def _redact_log_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in {"authorization", "x-api-key"}:
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted


def _truncate_log_payload(payload: str) -> str:
    if len(payload) <= _MAX_LOGGED_WEBSOCKET_PAYLOAD_CHARS:
        return payload
    return (
        payload[:_MAX_LOGGED_WEBSOCKET_PAYLOAD_CHARS]
        + f"\n... (truncated, {len(payload)} chars total)"
    )


def _safe_json_for_log(payload: Any) -> str:
    try:
        serialized = json.dumps(payload, ensure_ascii=False)
    except Exception:
        serialized = str(payload)
    return _truncate_log_payload(serialized)


class OpenAIResponsesSessionContext(AgentSessionContext):
    _default_websocket_ping_interval_seconds = 20.0
    _default_websocket_timeout_seconds = 60 * 60

    def __init__(
        self,
        *,
        websocket_timeout: float = _default_websocket_timeout_seconds,
        websocket_ping_interval_seconds: float = _default_websocket_ping_interval_seconds,
        session: aiohttp.ClientSession | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if websocket_timeout <= 0:
            raise ValueError("websocket_timeout must be greater than 0")
        if websocket_ping_interval_seconds <= 0:
            raise ValueError("websocket_ping_interval_seconds must be greater than 0")
        self._websocket_timeout = websocket_timeout
        self._websocket_ping_interval_seconds = websocket_ping_interval_seconds
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = session
        self._owns_session = session is None
        self._websocket_url: str | None = None
        self._websocket_headers_signature: tuple[tuple[str, str], ...] | None = None
        self._websocket_ping_task: asyncio.Task[None] | None = None
        self._websocket_timeout_task: asyncio.Task[None] | None = None
        self._websocket_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._websocket_previous_response_id: str | None = None
        self._websocket_incremental_start_index: int | None = None
        self.pending_supplemental_usage: dict[str, float] | None = None

    @staticmethod
    def _headers_signature(headers: dict[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((k.lower(), v) for k, v in headers.items()))

    @property
    def has_valid_websocket(self) -> bool:
        return self._websocket is not None and not self._websocket.closed

    def clear_websocket_incremental_state(self) -> None:
        self._websocket_previous_response_id = None
        self._websocket_incremental_start_index = None

    def get_websocket_incremental_request(self) -> tuple[str, list[Any]] | None:
        if not self.has_valid_websocket:
            return None

        previous_response_id = self._websocket_previous_response_id
        incremental_start_index = self._websocket_incremental_start_index
        if previous_response_id is None or incremental_start_index is None:
            return None
        if incremental_start_index < 0 or incremental_start_index > len(self.messages):
            self.clear_websocket_incremental_state()
            return None

        incremental_input = self.messages[incremental_start_index:]
        if len(incremental_input) == 0:
            return None
        return previous_response_id, copy.deepcopy(incremental_input)

    def remember_websocket_response(
        self,
        *,
        response_id: str | None,
        incremental_start_index: int,
    ) -> None:
        if (
            not self.has_valid_websocket
            or response_id is None
            or response_id.strip() == ""
        ):
            self.clear_websocket_incremental_state()
            return
        self._websocket_previous_response_id = response_id
        self._websocket_incremental_start_index = max(
            0,
            min(incremental_start_index, len(self.messages)),
        )

    @staticmethod
    def _header_value(
        headers: aiohttp.typedefs.LooseHeaders | None, key: str
    ) -> str | None:
        if headers is None:
            return None
        if isinstance(headers, dict):
            value = headers.get(key)
            if isinstance(value, str):
                trimmed = value.strip()
                return trimmed if trimmed != "" else None
            return None

        try:
            value = headers.get(key)
        except Exception:
            return None
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed if trimmed != "" else None
        return None

    @staticmethod
    def _fallback_handshake_error_message(status: int) -> str:
        if status == 402:
            return _OPENAI_OUT_OF_CREDITS_MESSAGE
        if status in {401, 403}:
            return "You are not authorized to use this OpenAI endpoint."
        if status == 429:
            return "OpenAI request was rate limited. Please retry in a moment."
        if status >= 500:
            return "OpenAI service is currently unavailable. Please try again later."
        return f"OpenAI websocket request failed with status {status}."

    @classmethod
    def _status_error_message(
        cls,
        *,
        status: int,
        headers: Mapping[str, str] | None,
        message: str | None,
    ) -> str:
        header_message = cls._header_value(headers, _MESHAGENT_ERROR_MESSAGE_HEADER)
        if header_message is not None:
            if status == 402 and _is_openai_out_of_credits_message(header_message):
                return _OPENAI_OUT_OF_CREDITS_MESSAGE
            return header_message

        if message is not None and message.strip() != "":
            stripped_message = message.strip()
            if status == 402 and _is_openai_out_of_credits_message(stripped_message):
                return _OPENAI_OUT_OF_CREDITS_MESSAGE
            if stripped_message.lower() != "invalid response status":
                return stripped_message

        return cls._fallback_handshake_error_message(status)

    @classmethod
    def _handshake_error_message(cls, error: aiohttp.WSServerHandshakeError) -> str:
        return cls._status_error_message(
            status=error.status,
            headers=error.headers,
            message=error.message,
        )

    async def _run_websocket_ping(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._websocket_ping_interval_seconds)
                websocket = self._websocket
                if websocket is None or websocket.closed:
                    return
                await websocket.ping()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("responses websocket ping failed, closing socket: %s", error)
            await self.close_websocket()

    async def _run_websocket_timeout(self) -> None:
        try:
            await asyncio.sleep(self._websocket_timeout)
        except asyncio.CancelledError:
            raise
        logger.info(
            "responses websocket session timed out after %.1f seconds",
            self._websocket_timeout,
        )
        await self.close()

    async def _close_websocket_locked(self) -> None:
        current_task = asyncio.current_task()

        tasks_to_cancel: list[asyncio.Task[None]] = []
        if (
            self._websocket_ping_task is not None
            and self._websocket_ping_task is not current_task
        ):
            tasks_to_cancel.append(self._websocket_ping_task)
        if (
            self._websocket_timeout_task is not None
            and self._websocket_timeout_task is not current_task
        ):
            tasks_to_cancel.append(self._websocket_timeout_task)

        self._websocket_ping_task = None
        self._websocket_timeout_task = None

        for task in tasks_to_cancel:
            task.cancel()
        for task in tasks_to_cancel:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        websocket = self._websocket
        self._websocket = None
        self._websocket_url = None
        self._websocket_headers_signature = None
        self.clear_websocket_incremental_state()
        if websocket is not None and not websocket.closed:
            await websocket.close()

    @staticmethod
    async def _close_client_session(
        *,
        session: aiohttp.ClientSession | None,
        close_session: bool,
    ) -> None:
        if session is None or session.closed or not close_session:
            return
        await asyncio.shield(session.close())

    async def close_websocket(self) -> None:
        async with self._websocket_lock:
            await self._close_websocket_locked()

    async def ensure_websocket(
        self, *, url: str, headers: dict[str, str]
    ) -> aiohttp.ClientWebSocketResponse:
        headers_signature = self._headers_signature(headers)
        async with self._websocket_lock:
            if (
                self._websocket is not None
                and not self._websocket.closed
                and self._websocket_url == url
                and self._websocket_headers_signature == headers_signature
            ):
                return self._websocket

            await self._close_websocket_locked()

            session = self._session
            close_session_on_failure = False
            if session is None:
                session = new_client_session(timeout=aiohttp.ClientTimeout(total=None))
                if self._owns_session:
                    self._session = session
                else:
                    close_session_on_failure = True
            try:
                websocket = await session.ws_connect(
                    url,
                    headers=headers,
                    heartbeat=None,
                    autoping=True,
                )
            except aiohttp.WSServerHandshakeError as error:
                if close_session_on_failure:
                    await self._close_client_session(
                        session=session,
                        close_session=True,
                    )
                elif self._owns_session and session is self._session:
                    self._session = None
                    await self._close_client_session(
                        session=session,
                        close_session=True,
                    )
                raise RoomException(
                    self._handshake_error_message(error),
                    status_code=error.status,
                ) from error
            except Exception:
                if close_session_on_failure:
                    await self._close_client_session(
                        session=session,
                        close_session=True,
                    )
                elif self._owns_session and session is self._session:
                    self._session = None
                    await self._close_client_session(
                        session=session,
                        close_session=True,
                    )
                raise

            self._websocket = websocket
            self._websocket_url = url
            self._websocket_headers_signature = headers_signature
            self._websocket_ping_task = asyncio.create_task(self._run_websocket_ping())
            self._websocket_timeout_task = asyncio.create_task(
                self._run_websocket_timeout()
            )
            return websocket

    async def close(self) -> None:
        await self.close_websocket()
        if self._owns_session:
            session = self._session
            self._session = None
            await self._close_client_session(
                session=session,
                close_session=True,
            )

    async def start(self) -> None:
        await super().start()
        if self._session is None and self._owns_session:
            self._session = new_client_session(
                timeout=aiohttp.ClientTimeout(total=None)
            )

    def copy(self) -> "OpenAIResponsesSessionContext":
        shared_session = self._session if not self._owns_session else None
        return self.__class__(
            messages=copy.deepcopy(self.messages),
            system_role=self.system_role,
            websocket_timeout=self._websocket_timeout,
            websocket_ping_interval_seconds=self._websocket_ping_interval_seconds,
            session=shared_session,
        )

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
        if len(data) > _OPENAI_RESPONSES_MAX_INLINE_IMAGE_BYTES:
            return self._append_attachment_note(
                f"the user attached an image ({normalized_mime_type}) that was too large to include"
            )
        message = {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": _encoded_data_url(
                        mime_type=normalized_mime_type,
                        data=data,
                    ),
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
                    "type": "input_image",
                    "image_url": url,
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
        if not _is_openai_responses_inline_file_mime_type(normalized_mime_type):
            return self._append_attachment_note(
                f"the user attached {filename} with unsupported mime type {normalized_mime_type}"
            )
        if len(data) > _OPENAI_RESPONSES_MAX_INLINE_FILE_BYTES:
            return self._append_attachment_note(
                f"the user attached {filename} ({normalized_mime_type}) but it was too large to include"
            )
        message = {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": filename,
                    "file_data": _encoded_data_url(
                        mime_type=normalized_mime_type,
                        data=data,
                    ),
                }
            ],
        }
        self.messages.append(message)
        return message

    def append_file_url(self, *, url: str, filename: str | None = None) -> dict:
        data_url = _decode_data_url_attachment(url)
        if data_url is not None:
            return self.append_file_message(
                filename=filename or "attachment",
                mime_type=data_url.mime_type,
                data=data_url.data,
            )
        message = {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_url": url,
                }
            ],
        }
        self.messages.append(message)
        return message

    def _append_attachment_note(self, text: str) -> dict:
        message = {"role": "user", "content": [{"type": "input_text", "text": text}]}
        self.messages.append(message)
        return message


# Backwards compatibility for code still importing the old class name.
OpenAIResponsesChatContext = OpenAIResponsesSessionContext


class OpenAIResponsesAgentEventReader(AccumulatingAgentEventReader):
    def __init__(
        self,
        *,
        emit_message: Callable[[dict[str, Any]], None],
        callbacks: AgentEventReaderCallbacks | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(emit_message=emit_message, callbacks=callbacks)
        self._provider = provider

    def _append_user_text(self, text: str) -> None:
        self._emit_context_message(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
        )

    def _append_user_content(self, content: list[dict[str, Any]]) -> None:
        parts: list[dict[str, Any]] = []
        for item in content:
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append({"type": "input_text", "text": text})
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
                    if data_url is None:
                        parts.append({"type": "input_file", "file_url": url})
                    elif data_url.mime_type.startswith("image/"):
                        parts.append(
                            {
                                "type": "input_image",
                                "image_url": _encoded_data_url(
                                    mime_type=data_url.mime_type,
                                    data=data_url.data,
                                ),
                            }
                        )
                    elif _is_openai_responses_inline_file_mime_type(data_url.mime_type):
                        parts.append(
                            {
                                "type": "input_file",
                                "filename": filename,
                                "file_data": _encoded_data_url(
                                    mime_type=data_url.mime_type,
                                    data=data_url.data,
                                ),
                            }
                        )
                    else:
                        parts.append(
                            {
                                "type": "input_text",
                                "text": (
                                    "the user attached "
                                    f"{filename} with unsupported mime type "
                                    f"{data_url.mime_type}"
                                ),
                            }
                        )
        if not parts:
            parts.append({"type": "input_text", "text": json.dumps(content)})
        self._emit_context_message({"role": "user", "content": parts})

    def _append_assistant_text(self, *, text: str, phase: str | None) -> None:
        del phase
        self._emit_context_message(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )

    def _append_assistant_reasoning(
        self,
        *,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        message: dict[str, Any] = {"type": "reasoning", "summary": []}
        if text != "":
            message["summary"] = [{"type": "summary_text", "text": text}]
        openai_metadata = metadata.get("openai")
        if isinstance(openai_metadata, dict):
            encrypted_content = openai_metadata.get("encrypted_content")
            if isinstance(encrypted_content, str) and encrypted_content != "":
                message["encrypted_content"] = encrypted_content
        self._emit_context_message(message)

    def _append_assistant_file(self, *, url: str) -> None:
        self._append_assistant_text(text=f"Generated file: {url}", phase=None)

    def _append_thread_event(self, *, event: dict[str, Any]) -> None:
        self._append_assistant_text(
            text=json.dumps({"type": "event", "event": event}),
            phase=None,
        )

    def _append_tool_call(
        self,
        *,
        tool_call: _BufferedToolCall,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> None:
        item_type = self._response_item_type(tool_call=tool_call)
        call_id = tool_call.call_id or tool_call.item_id
        if item_type == "function_call":
            if tool_call.provider != self._provider:
                self._append_cross_provider_tool_call(
                    tool_call=tool_call,
                    result=result,
                    error=error,
                )
                return
            self._emit_context_message(
                {
                    "type": "function_call",
                    "id": tool_call.item_id,
                    "call_id": call_id,
                    "name": self._function_name(tool_call=tool_call),
                    "arguments": tool_call.arguments_json(),
                }
            )
            if result is not None or error is not None or tool_call.logs:
                self._emit_context_message(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": self._result_text(
                            result=result,
                            error=error,
                            logs=tool_call.logs,
                        ),
                    }
                )
            return

        item = self._builtin_call_item(
            item_type=item_type,
            tool_call=tool_call,
            call_id=call_id,
            result=result,
            error=error,
        )
        self._emit_context_message(item)
        output = self._builtin_output_item(
            item_type=item_type,
            call_id=call_id,
            result=result,
            error=error,
            logs=tool_call.logs,
        )
        if output is not None:
            self._emit_context_message(output)

    def _append_cross_provider_tool_call(
        self,
        *,
        tool_call: _BufferedToolCall,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> None:
        tool_name = self._function_name(tool_call=tool_call)
        arguments = tool_call.arguments_json()
        self._emit_context_message(
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"Called tool {tool_name} with arguments: {arguments}",
                    }
                ],
            }
        )
        if result is not None or error is not None or tool_call.logs:
            self._emit_context_message(
                {
                    "role": "user",
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
        del event_type, turn_id, toolkit, tool
        item: dict[str, Any] = {
            "type": "image_generation_call",
            "id": item_id,
            "status": status,
        }
        if call_id is not None:
            item["call_id"] = call_id
        result = self._image_generation_result_from_images(images=images)
        if result is None:
            if status == "completed":
                return
            if arguments is not None:
                for key in _OPENAI_RESPONSES_IMAGE_GENERATION_CALL_INPUT_FIELDS:
                    if key in arguments:
                        item[key] = copy.deepcopy(arguments[key])
        else:
            item["result"] = result
        self._emit_context_message(item)

    def _append_audio_generation_event(self, *, message: AgentMessage) -> None:
        del message
        return

    def _append_audio_transcription_event(self, *, message: AgentMessage) -> None:
        self._append_assistant_text(
            text=json.dumps(message.model_dump(mode="json")),
            phase=None,
        )

    def _restore_compacted_messages(self, *, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            self._emit_context_message(message)

    @staticmethod
    def _image_generation_result_from_images(
        *,
        images: list[dict[str, Any]],
    ) -> str | None:
        for image in images:
            uri = image.get("uri")
            if not isinstance(uri, str):
                continue
            attachment = _decode_data_url_attachment(uri)
            if attachment is None:
                continue
            return base64.b64encode(attachment.data).decode("ascii")
        return None

    @staticmethod
    def _response_item_type(*, tool_call: _BufferedToolCall) -> str:
        if tool_call.namespace == "openai.responses":
            if tool_call.toolkit == "openai":
                return f"{tool_call.tool}_call"
            if tool_call.tool == "list_tools":
                return "mcp_list_tools"
            return "mcp_call"
        return "function_call"

    @staticmethod
    def _function_name(*, tool_call: _BufferedToolCall) -> str:
        if tool_call.toolkit in {"", "function", "tool"}:
            return safe_tool_name(tool_call.tool)
        return safe_tool_name(f"{tool_call.toolkit}_{tool_call.tool}")

    @staticmethod
    def _builtin_call_item(
        *,
        item_type: str,
        tool_call: _BufferedToolCall,
        call_id: str,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> dict[str, Any]:
        arguments = tool_call.arguments_dict()
        status = (
            "failed"
            if error is not None
            else "completed"
            if result is not None
            else "in_progress"
        )
        item: dict[str, Any] = {
            "type": item_type,
            "id": tool_call.item_id,
            "status": status,
        }
        if item_type not in {
            "mcp_list_tools",
            "image_generation_call",
            "web_search_call",
        }:
            item["call_id"] = call_id
        if item_type == "mcp_call":
            item["server_label"] = tool_call.toolkit
            item["name"] = tool_call.tool
            item["arguments"] = tool_call.arguments_json()
        elif item_type == "mcp_list_tools":
            item["server_label"] = tool_call.toolkit
        elif item_type in {"shell_call", "local_shell_call"}:
            action = arguments.get("action")
            if not isinstance(action, dict):
                if "commands" in arguments:
                    action = {"commands": arguments["commands"]}
                else:
                    action = {"commands": [tool_call.arguments_json()]}
            item["action"] = action
        elif item_type == "apply_patch_call":
            operation = arguments.get("operation")
            item["operation"] = operation if isinstance(operation, dict) else arguments
        elif item_type == "code_interpreter_call":
            code = arguments.get("code")
            item["code"] = code if isinstance(code, str) else tool_call.arguments_json()
        else:
            item.update(arguments)
        if result is not None and item_type not in {
            "shell_call",
            "local_shell_call",
            "computer_call",
            "apply_patch_call",
        }:
            item["output"] = copy.deepcopy(result)
        if error is not None:
            item["error"] = copy.deepcopy(error)
        return item

    def _builtin_output_item(
        self,
        *,
        item_type: str,
        call_id: str,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        logs: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        if result is None and error is None and not logs:
            return None
        if item_type in {"shell_call", "local_shell_call"}:
            output = self._shell_output_from_result(
                result=result,
                error=error,
                logs=logs,
            )
            return {
                "type": f"{item_type}_output",
                "call_id": call_id,
                "output": output,
            }
        output_text = self._result_text(result=result, error=error, logs=logs)
        if item_type == "computer_call":
            return {
                "type": f"{item_type}_output",
                "call_id": call_id,
                "output": output_text,
            }
        if item_type == "apply_patch_call":
            payload: dict[str, Any] = {
                "type": "apply_patch_call_output",
                "call_id": call_id,
                "output": output_text,
            }
            payload["status"] = "failed" if error is not None else "completed"
            return payload
        return None

    def _shell_output_from_result(
        self,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        logs: list[dict[str, str]],
    ) -> Any:
        if error is None and result is not None:
            result_payload = result
            json_payload = result.get("json")
            if result.get("type") == "json" and isinstance(json_payload, dict):
                result_payload = json_payload
            output = result_payload.get("output")
            if isinstance(output, list):
                return copy.deepcopy(output)
            results = result_payload.get("results")
            if isinstance(results, list):
                return copy.deepcopy(results)
        return self._result_text(result=result, error=error, logs=logs)

    def _append_assistant_structured_item(self, item: dict[str, Any]) -> None:
        self._emit_context_message(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": json.dumps(item)}],
            }
        )


def _is_html_mime_type(mime_type: str | None) -> bool:
    if not mime_type:
        return False
    normalized = mime_type.split(";")[0].strip().lower()
    return normalized in {"text/html", "application/xhtml+xml"}


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def safe_json_dump(data: dict):
    return json.dumps(copy.deepcopy(data))


def safe_model_dump(model: BaseModel):
    try:
        return safe_json_dump(model.model_dump(mode="json"))
    except Exception:
        return {"error": "unable to dump json for model"}


def _emit_stream_json_item(
    *, item: Any, event_handler: Optional[Callable[[dict], None]]
) -> None:
    if event_handler is None:
        return
    if isinstance(item, JsonContent):
        event_handler(item.json)


async def _consume_streaming_tool_items(
    *,
    tool_name: str,
    tool_call_id: Optional[str],
    item_id: Optional[str],
    stream: AsyncIterable[Any],
    event_handler: Optional[Callable[[dict], None]],
) -> Any:
    del tool_name
    del tool_call_id
    del item_id
    has_last = False
    last_item: Any = None
    async for item in stream:
        if has_last:
            _emit_stream_json_item(item=last_item, event_handler=event_handler)
        last_item = item
        has_last = True

    if not has_last:
        return None

    if isinstance(last_item, _ControlContent):
        return None
    if isinstance(last_item, dict):
        last_type = last_item.get("type")
        if last_type in ("agent.event", "codex.event"):
            return None

    return last_item


async def _consume_streaming_tool_result(
    *,
    tool_name: str,
    tool_call_id: Optional[str],
    item_id: Optional[str],
    stream: AsyncIterable[Any],
    event_handler: Optional[Callable[[dict], None]],
) -> Content:
    item = await _consume_streaming_tool_items(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        item_id=item_id,
        stream=stream,
        event_handler=event_handler,
    )
    return ensure_content(item)


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


def safe_tool_name(name: str):
    return _replace_non_matching(name, "a-zA-Z0-9_-", "_")


# Collects a group of tool proxies and manages execution of openai tool calls
class ResponsesToolBundle:
    def __init__(
        self,
        toolkits: List[Toolkit],
        *,
        tool_call_approval_handler: ToolCallApprovalHandler | None = None,
        tool_search: OpenAIResponsesToolSearchMode | None = None,
    ):
        self._toolkits = toolkits
        self._executors = dict[tuple[str | None, str], Toolkit]()
        self._safe_names = dict[tuple[str | None, str], str]()
        self._unqualified_safe_names = dict[str, tuple[str | None, str]]()
        self._tools_by_name = dict[tuple[str | None, str], BaseTool]()
        self._function_tool_definitions: list[dict[str, Any]] = []

        open_ai_tools = []
        has_deferred_function_tools = False

        for toolkit in toolkits:
            namespace_tools: list[dict[str, Any]] = []
            function_namespace_tools: list[dict[str, Any]] = []
            for tool in toolkit.tools:
                if isinstance(tool, MCPTool):
                    tool = tool.with_tool_call_approval_handler(
                        tool_call_approval_handler
                    )

                v = tool
                k = v.name

                name = safe_tool_name(k)
                namespace = safe_tool_name(toolkit.name) if toolkit.name else None
                tool_key = (namespace, name)

                if tool_key in self._executors:
                    raise Exception(
                        f"duplicate in bundle '{toolkit.name}.{k}', tool names must be unique within a toolkit."
                    )

                self._executors[tool_key] = toolkit
                self._safe_names[tool_key] = k
                self._tools_by_name[tool_key] = v
                if name not in self._unqualified_safe_names:
                    self._unqualified_safe_names[name] = tool_key

                if isinstance(v, OpenAIResponsesTool):
                    fns = v.get_open_ai_tool_definitions()
                    for fn in fns:
                        open_ai_tools.append(fn)

                elif isinstance(v, FunctionTool):
                    fn = {
                        "type": "function",
                        "name": name,
                        "description": v.description,
                        "parameters": {
                            **v.input_schema,
                        },
                        "strict": v.strict,
                    }

                    if v.defs is not None:
                        fn["parameters"]["$defs"] = v.defs

                    if tool_search is not None:
                        fn["defer_loading"] = True
                        has_deferred_function_tools = True

                    function_namespace_tools.append(fn)

                    if tool_search == "client":
                        continue
                    if namespace is None:
                        open_ai_tools.append(fn)
                    else:
                        namespace_tools.append(fn)

                else:
                    raise RoomException(f"unsupported tool type {type(v)}")

            if namespace_tools:
                open_ai_tools.append(self._namespace_tool(toolkit, namespace_tools))

            if function_namespace_tools:
                self._function_tool_definitions.append(
                    self._namespace_tool(toolkit, function_namespace_tools)
                )

        if has_deferred_function_tools or tool_search == "client":
            tool_search_tool: dict[str, Any] = {"type": "tool_search"}
            if tool_search == "client":
                tool_search_tool.update(
                    {
                        "execution": "client",
                        "description": "Search available MeshAgent tools.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    }
                )
            open_ai_tools.insert(0, tool_search_tool)

        if len(open_ai_tools) == 0:
            open_ai_tools = None

        self._open_ai_tools = open_ai_tools

    @staticmethod
    def _namespace_tool(
        toolkit: Toolkit, tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "type": "namespace",
            "name": safe_tool_name(toolkit.name),
            "description": toolkit.description or toolkit.title or toolkit.name,
            "tools": tools,
        }

    def function_tool_definitions(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._function_tool_definitions)

    async def execute(
        self, *, context: ToolContext, tool_call: ResponseFunctionToolCall
    ) -> Content | AsyncIterable[Any]:
        name = tool_call.name
        namespace = safe_tool_name(tool_call.namespace) if tool_call.namespace else None
        arguments = json.loads(tool_call.arguments)
        tool_key = (namespace, name)

        if tool_key not in self._safe_names:
            if namespace is None:
                tool_key = self._unqualified_safe_names.get(name, tool_key)

        if tool_key not in self._safe_names:
            raise RoomException(f"Invalid tool name {name}, check the name of the tool")

        name = self._safe_names[tool_key]

        if tool_key not in self._executors:
            raise Exception(f"Unregistered tool name {name}")

        proxy = self._executors[tool_key]
        result = await proxy.execute(
            context=context,
            name=name,
            input=JsonContent(json=arguments),
        )
        if isinstance(result, AsyncIterable):
            return result
        return ensure_content(result)

    def get_tool(self, name: str) -> BaseTool | None:
        tool_key = self._unqualified_safe_names.get(name)
        if tool_key is None:
            return None
        return self._tools_by_name.get(tool_key, None)

    def resolve_function_tool_name(
        self, safe_name: str, namespace: str | None
    ) -> tuple[str, str] | None:
        tool_key = (safe_tool_name(namespace), safe_name) if namespace else None
        if tool_key is None or tool_key not in self._safe_names:
            tool_key = self._unqualified_safe_names.get(safe_name)
        if tool_key is None:
            return None

        original_name = self._safe_names.get(tool_key)
        if original_name is None:
            return None

        toolkit = self._executors.get(tool_key)
        if toolkit is None:
            return None

        return toolkit.name, original_name

    def contains(self, name: str) -> bool:
        return name in self._open_ai_tools

    def to_json(self) -> List[dict] | None:
        if self._open_ai_tools is None:
            return None
        return self._open_ai_tools.copy()


# Converts a tool response into a series of messages that can be inserted into the openai context
class OpenAIResponsesToolResponseAdapter(ToolResponseAdapter):
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
        #         "role" : "assistant",
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
        tool_call: ResponseFunctionToolCall,
        response: Content,
    ) -> list:
        del context
        with tracer.start_as_current_span("llm.tool_adapter.create_messages") as span:
            if isinstance(response, RawOutputsContent):
                span.set_attribute("kind", "raw")
                return response.outputs

            else:
                span.set_attribute("kind", "text")

                if isinstance(response, FileContent):
                    text_file = await self.file_content_to_text_content(
                        content=response
                    )
                    if text_file is not None:
                        if _is_html_mime_type(response.mime_type):
                            text_file = TextContent(text=convert(text_file.text))
                        output = await self.to_plain_text(response=text_file)
                        span.set_attribute("output", output)
                        message = {
                            "output": output,
                            "call_id": tool_call.call_id,
                            "type": "function_call_output",
                        }
                    elif response.mime_type and response.mime_type.startswith("image/"):
                        span.set_attribute(
                            "output", f"image: {response.name}, {response.mime_type}"
                        )

                        message = {
                            "output": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:{response.mime_type};base64,{base64.b64encode(response.data).decode()}",
                                }
                            ],
                            "call_id": tool_call.call_id,
                            "type": "function_call_output",
                        }
                    else:
                        span.set_attribute(
                            "output", f"file: {response.name}, {response.mime_type}"
                        )

                        if response.mime_type == "application/pdf":
                            message = {
                                "output": [
                                    {
                                        "type": "input_file",
                                        "filename": response.name,
                                        "file_data": f"data:{response.mime_type or 'text/plain'};base64,{base64.b64encode(response.data).decode()}",
                                    }
                                ],
                                "call_id": tool_call.call_id,
                                "type": "function_call_output",
                            }
                        else:
                            message = {
                                "output": f"{response.name} was not in a supported format",
                                "call_id": tool_call.call_id,
                                "type": "function_call_output",
                            }

                    return [message]
                else:
                    output = await self.to_plain_text(response=response)
                    span.set_attribute("output", output)

                    message = {
                        "output": output,
                        "call_id": tool_call.call_id,
                        "type": "function_call_output",
                    }

                    return [message]


class OpenAIResponsesAdapter(LLMAdapter[dict[str, Any]]):
    _known_models = (
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex",
        "gpt-5.2",
        "gpt-5",
        "gpt-4.1",
        "gpt-4o",
        "o4-mini",
        "o3",
        "o1",
    )
    _context_window_sizes = {
        "gpt-5.5": 400000,
        "gpt-4.1": 128000,
        "gpt-4o": 128000,
        "gpt-5.4": 272000,
        "gpt-5.3-codex": 400000,
        "gpt-5.2": 400000,
        "gpt-5": 400000,
        "o1": 200000,
        "o3": 200000,
        "o4": 200000,
    }
    _default_max_retries = 10
    _default_retry_backoff_seconds = 1.0
    _max_retry_backoff_seconds = 30.0
    _default_websocket_timeout_seconds = 60 * 60

    def __init__(
        self,
        model: str = os.getenv("OPENAI_MODEL", "gpt-5.2"),
        parallel_tool_calls: Optional[bool] = None,
        client: Optional[AsyncOpenAI] = None,
        response_options: Optional[dict] = None,
        reasoning_effort: Optional[str] = None,
        provider: str = "openai",
        log_requests: bool = False,
        max_output_tokens: Optional[int] = 32000,
        max_retries: int = _default_max_retries,
        mode: Literal["request", "websocket"] = "websocket",
        websocket_timeout: float = _default_websocket_timeout_seconds,
        context_management: Literal["auto", "standalone", "none"] = "auto",
        compaction_threshold: Optional[int | float] = None,
        tool_search: OpenAIResponsesToolSearchConfig = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        user_agent: str | None = None,
        annotations: Mapping[str, object] | None = None,
        images_dataset: ImagesDataset | None = None,
        max_tool_call_length: int = DEFAULT_MAX_TOOL_CALL_LENGTH,
        max_tool_call_lines: int = DEFAULT_MAX_TOOL_CALL_LINES,
        friendly_name: str | None = None,
        description: str | None = None,
        allowed_models: list[str] | None = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        if mode not in ("request", "websocket"):
            raise ValueError("mode must be either 'request' or 'websocket'")
        if websocket_timeout <= 0:
            raise ValueError("websocket_timeout must be greater than 0")
        if context_management not in ("auto", "standalone", "none"):
            raise ValueError(
                "context_management must be one of 'auto', 'standalone', or 'none'"
            )
        if compaction_threshold is not None and isinstance(compaction_threshold, bool):
            raise ValueError("compaction_threshold must be an integer or infinity")
        resolved_tool_search = self._normalize_tool_search_mode(tool_search)
        resolved_compaction_threshold: Optional[int] = None
        if compaction_threshold is None:
            if self.context_window_size(model) != float("inf"):
                resolved_compaction_threshold = 200000
        elif isinstance(compaction_threshold, float) and math.isinf(
            compaction_threshold
        ):
            if compaction_threshold < 0:
                raise ValueError("compaction_threshold must be positive infinity")
            resolved_compaction_threshold = None
        else:
            if isinstance(compaction_threshold, float):
                if not compaction_threshold.is_integer():
                    raise ValueError(
                        "compaction_threshold must be an integer or infinity"
                    )
                compaction_threshold = int(compaction_threshold)
            if compaction_threshold <= 0:
                raise ValueError("compaction_threshold must be greater than 0")
            resolved_compaction_threshold = int(compaction_threshold)
        self._model = model
        self._parallel_tool_calls = parallel_tool_calls
        self._client = client
        self._base_url = resolve_base_url(base_url)
        self._has_explicit_api_key = isinstance(api_key, str) and api_key.strip() != ""
        self._api_key = resolve_api_key(api_key)
        self._user_agent = resolve_user_agent(user_agent)
        self._annotations = normalize_llm_annotations(annotations)
        self._response_options = response_options
        self._provider = provider
        self._reasoning_effort = reasoning_effort
        self._log_requests = log_requests
        self.max_output_tokens = max_output_tokens
        self._context_management_mode = context_management
        self._compaction_threshold = resolved_compaction_threshold
        self._max_retries = max_retries
        self._mode = mode
        self._websocket_timeout = websocket_timeout
        self._tool_search = resolved_tool_search
        self._max_tool_call_length = max_tool_call_length
        self._max_tool_call_lines = max_tool_call_lines
        self._images_dataset = images_dataset
        self._friendly_name = friendly_name
        self._description = description
        self._allowed_models = (
            list(allowed_models) if allowed_models is not None else None
        )
        self._tool_call_approval_handler: ToolCallApprovalHandler | None = None

    @staticmethod
    def _normalize_tool_search_mode(
        tool_search: OpenAIResponsesToolSearchConfig,
    ) -> OpenAIResponsesToolSearchMode | None:
        if tool_search is None or tool_search is False:
            return None
        if tool_search is True:
            return "server"
        if isinstance(tool_search, str):
            normalized = tool_search.strip().lower()
            if normalized in {"server", "client"}:
                return cast(OpenAIResponsesToolSearchMode, normalized)
        raise ValueError("tool_search must be one of None, 'server', or 'client'")

    def default_model(self) -> str:
        return self._model

    def provider_name(self) -> str | None:
        return self._provider

    def provider_friendly_name(self) -> str:
        return self._friendly_name or "OpenAI"

    def provider_description(self) -> str | None:
        return self._description or "OpenAI Responses API"

    def list_models(self) -> list[LLMModelInfo]:
        names = list(self._allowed_models or self._known_models)
        if self._allowed_models is None and self._model not in names:
            names.insert(0, self._model)
        return [
            LLMModelInfo(
                name=name,
                context_window=int(context_window)
                if (context_window := self.context_window_size(name)) != float("inf")
                else None,
                pricing=llm_model_pricing(provider="openai", model=name),
                supports_attachments=True,
                accepts=_OPENAI_RESPONSES_ACCEPTED_ATTACHMENT_TYPES,
            )
            for name in names
        ]

    def set_images_dataset(self, images_dataset: ImagesDataset | None) -> None:
        self._images_dataset = images_dataset

    def set_tool_call_approval_handler(
        self, handler: ToolCallApprovalHandler | None
    ) -> None:
        self._tool_call_approval_handler = handler

    async def search_toolkits(
        self,
        *,
        context: AgentSessionContext,
        caller: Participant,
        request: OpenAIResponsesToolSearchRequest,
        toolkits: list[Toolkit],
        on_behalf_of: Participant | None = None,
    ) -> list[Toolkit]:
        del context, caller, on_behalf_of
        query = request.query or json.dumps(request.arguments)
        terms = [term.casefold() for term in re.findall(r"[a-zA-Z0-9_/-]+", query)]
        if not terms:
            return []

        matches: list[Toolkit] = []
        for toolkit in toolkits:
            searchable_parts = [
                toolkit.name,
                toolkit.title,
                toolkit.description,
                *toolkit.rules,
            ]
            for tool in toolkit.tools:
                searchable_parts.extend([tool.name, tool.title, tool.description])
                searchable_parts.extend(tool.rules)
            haystack = " ".join(
                part for part in searchable_parts if isinstance(part, str)
            ).casefold()
            if any(term in haystack for term in terms):
                matches.append(toolkit)
        return matches

    async def _client_tool_search_output(
        self,
        *,
        context: AgentSessionContext,
        caller: Participant,
        tool_search_call: ResponseToolSearchCall,
        toolkits: list[Toolkit],
        on_behalf_of: Participant | None,
    ) -> tuple[dict[str, Any], list[Toolkit]]:
        request = OpenAIResponsesToolSearchRequest.from_call(tool_search_call)
        found_toolkits = await self.search_toolkits(
            context=context,
            caller=caller,
            request=request,
            toolkits=toolkits,
            on_behalf_of=on_behalf_of,
        )
        found_bundle = ResponsesToolBundle(
            toolkits=found_toolkits,
            tool_call_approval_handler=self._tool_call_approval_handler,
            tool_search="server",
        )
        output: dict[str, Any] = {
            "type": "tool_search_output",
            "tools": found_bundle.function_tool_definitions(),
            "execution": "client",
            "status": "completed",
        }
        if tool_search_call.call_id is not None:
            output["call_id"] = tool_search_call.call_id
        output["id"] = self._tool_search_output_id(tool_search_call.id)
        return output, found_toolkits

    @staticmethod
    def _tool_search_output_id(tool_search_call_id: str) -> str:
        if tool_search_call_id.startswith("tsc_"):
            return f"tso_{tool_search_call_id.removeprefix('tsc_')}"
        return f"tso_{safe_tool_name(tool_search_call_id)}"

    def with_runtime_api_key(self, *, api_key: str | None) -> "OpenAIResponsesAdapter":
        resolved_api_key = resolve_api_key(api_key)
        if (
            self._client is not None
            or self._has_explicit_api_key
            or resolved_api_key is None
        ):
            return self

        clone = type(self)(
            model=self._model,
            parallel_tool_calls=self._parallel_tool_calls,
            response_options=self._response_options,
            reasoning_effort=self._reasoning_effort,
            provider=self._provider,
            log_requests=self._log_requests,
            max_output_tokens=self.max_output_tokens,
            max_retries=self._max_retries,
            mode=self._mode,
            websocket_timeout=self._websocket_timeout,
            context_management=self._context_management_mode,
            compaction_threshold=(
                self._compaction_threshold
                if self._compaction_threshold is not None
                else float("inf")
            ),
            tool_search=self._tool_search,
            base_url=self._base_url,
            api_key=resolved_api_key,
            user_agent=self._user_agent,
            annotations=self._annotations,
            images_dataset=self._images_dataset,
            max_tool_call_length=self._max_tool_call_length,
            max_tool_call_lines=self._max_tool_call_lines,
            friendly_name=self._friendly_name,
            description=self._description,
            allowed_models=self._allowed_models,
        )
        clone._compaction_threshold = self._compaction_threshold
        clone._tool_call_approval_handler = self._tool_call_approval_handler
        clone._response_options = copy.deepcopy(self._response_options)
        return clone

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback: Callable[[AgentMessage], None],
        custom_event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Callable[[dict[str, Any]], None]:
        return make_openai_agent_event_publisher(
            turn_id=turn_id,
            thread_id=thread_id,
            callback=callback,
            custom_event_callback=custom_event_callback,
            provider=self._provider,
            model=self._model,
        )

    def _set_function_tool_name_resolver(
        self,
        *,
        event_handler: Callable[[dict[str, Any]], None] | None,
        resolver: Callable[[str], tuple[str, str] | None] | None,
    ) -> None:
        if isinstance(event_handler, _OpenAIAgentEventPublisher):
            event_handler.set_function_tool_name_resolver(resolver)

    def create_session(
        self, *, usage_callback: SessionUsageCallback | None = None
    ) -> OpenAIResponsesSessionContext:
        context = OpenAIResponsesSessionContext(
            system_role=None,
            websocket_timeout=self._websocket_timeout,
            usage_callback=usage_callback,
        )
        return context

    def make_agent_event_reader(
        self,
        *,
        emit_message: Callable[[dict[str, Any]], None],
        callbacks: AgentEventReaderCallbacks | None = None,
    ) -> AgentEventReader:
        return OpenAIResponsesAgentEventReader(
            emit_message=emit_message,
            callbacks=callbacks,
            provider=self._provider,
        )

    def restore_context_messages(
        self,
        *,
        context: AgentSessionContext,
        messages: list[dict[str, Any]],
    ) -> None:
        if isinstance(context, OpenAIResponsesSessionContext):
            context.clear_websocket_incremental_state()
        context.messages.clear()
        context.messages.extend(
            [
                self._normalize_restored_context_message(message=message)
                for message in messages
            ]
        )

    @staticmethod
    def _normalize_stateless_context_message(
        *,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        restored = copy.deepcopy(message)
        message_type = restored.get("type")
        if message_type in {"shell_call_output", "local_shell_call_output"}:
            OpenAIResponsesAdapter._normalize_restored_shell_call_output(
                message=restored
            )
            return restored
        if message_type != "reasoning":
            return restored

        encrypted_content = restored.get("encrypted_content")
        if isinstance(encrypted_content, str) and encrypted_content != "":
            restored.pop("id", None)
            restored.setdefault("summary", [])
            return restored

        summary = restored.get("summary")
        summary_text_parts: list[str] = []
        if isinstance(summary, list):
            for part in summary:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip() != "":
                    summary_text_parts.append(text)
        text = "\n".join(summary_text_parts).strip()
        if text == "":
            text = "Restored prior reasoning item without encrypted content."
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }

    @staticmethod
    def _normalize_restored_shell_call_output(*, message: dict[str, Any]) -> None:
        output = message.get("output")
        if isinstance(output, str):
            message["output"] = [
                {
                    "outcome": {"type": "exit", "exit_code": 0},
                    "stdout": output,
                    "stderr": "",
                }
            ]
            return

        if not isinstance(output, list):
            message["output"] = [
                {
                    "outcome": {"type": "error"},
                    "stdout": "",
                    "stderr": json.dumps(output, ensure_ascii=False, default=str),
                }
            ]
            return

        normalized_output: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict):
                normalized_output.append(
                    {
                        "outcome": {"type": "error"},
                        "stdout": "",
                        "stderr": str(item),
                    }
                )
                continue

            normalized_item = copy.deepcopy(item)
            outcome = normalized_item.get("outcome")
            if isinstance(outcome, dict):
                normalized_outcome = dict(outcome)
                if normalized_outcome.get("type") == "exit":
                    exit_code = normalized_outcome.get("exit_code")
                    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
                        normalized_outcome["exit_code"] = 0
                normalized_item["outcome"] = normalized_outcome
            normalized_output.append(normalized_item)

        message["output"] = normalized_output

    @staticmethod
    def _normalize_restored_context_message(
        *,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        return OpenAIResponsesAdapter._normalize_stateless_context_message(
            message=message
        )

    def _make_tool_response_adapter(self) -> OpenAIResponsesToolResponseAdapter:
        return OpenAIResponsesToolResponseAdapter(
            max_tool_call_length=self._max_tool_call_length,
            max_tool_call_lines=self._max_tool_call_lines,
        )

    @staticmethod
    def _reasoning_enabled(response_options: dict[str, Any]) -> bool:
        reasoning = response_options.get("reasoning")
        if not isinstance(reasoning, dict):
            return False
        effort = reasoning.get("effort")
        return not (isinstance(effort, str) and effort == "none")

    @staticmethod
    def _ensure_encrypted_reasoning_include(response_options: dict[str, Any]) -> None:
        if not OpenAIResponsesAdapter._reasoning_enabled(response_options):
            return
        include = response_options.get("include")
        if include is None:
            response_options["include"] = ["reasoning.encrypted_content"]
            return
        if isinstance(include, list):
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            return
        response_options["include"] = [include, "reasoning.encrypted_content"]

    def _compose_instructions(self, *, context: AgentSessionContext) -> str | None:
        instruction_parts = [
            part
            for part in (
                context.get_system_instructions(),
                self.get_additional_instructions(),
            )
            if isinstance(part, str) and part.strip() != ""
        ]
        if len(instruction_parts) == 0:
            return None
        return "\n\n".join(instruction_parts)

    def context_window_size(self, model: str) -> float:
        model_key = model.lower()
        for prefix, size in self._context_window_sizes.items():
            if model_key.startswith(prefix):
                return size
        return float("inf")

    def context_management_mode(self) -> str | None:
        return self._context_management_mode

    def compaction_threshold(self, model: str) -> int | None:
        return self._effective_compaction_threshold(model=model)

    def _usage_context_window_size(self, model: str) -> int | None:
        size = self.context_window_size(model)
        if not math.isfinite(size):
            return None
        return max(0, int(size))

    def needs_compaction(self, *, context: AgentSessionContext) -> bool:
        if self._context_management_mode != "standalone":
            return False
        usage = context.last_usage
        if usage is None or usage.context_window_used is None:
            return False

        context_used_tokens = usage.context_window_used
        model = usage.model
        threshold = self._effective_compaction_threshold(model=model)
        if threshold is None:
            return False

        return int(context_used_tokens) > threshold

    def _effective_compaction_threshold(self, *, model: str) -> int | None:
        threshold = self._compaction_threshold
        if threshold is None:
            return None
        context_window = self.context_window_size(model)
        if context_window != float("inf") and self.max_output_tokens is not None:
            usable = int(context_window - self.max_output_tokens)
            threshold = min(threshold, max(0, usable))
        return threshold

    def _create_kwargs_has_computer_tool(
        self, *, create_kwargs: dict[str, Any]
    ) -> bool:
        tools = create_kwargs.get("tools")
        if tools in (None, NOT_GIVEN):
            return False
        if not isinstance(tools, list):
            return False
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") in {
                "computer_use_preview",
                "computer",
            }:
                return True
        return False

    def _add_auto_compaction_entry(
        self, *, create_kwargs: dict[str, Any], model: str
    ) -> None:
        if self._context_management_mode != "auto":
            return
        threshold = self._effective_compaction_threshold(model=model)
        if threshold is None:
            return
        if self._create_kwargs_has_computer_tool(create_kwargs=create_kwargs):
            return

        context_management = create_kwargs.get("context_management")
        compaction_entry: dict[str, Any] = {
            "type": "compaction",
            "compact_threshold": threshold,
        }

        if context_management in (None, NOT_GIVEN):
            create_kwargs["context_management"] = [compaction_entry]
            return

        if isinstance(context_management, list):
            context_management_entries = [*context_management]
        else:
            context_management_entries = list(context_management)

        has_compaction_entry = False
        normalized_entries = list[Any]()
        for entry in context_management_entries:
            if isinstance(entry, dict) and entry.get("type") == "compaction":
                normalized_entry = {
                    **entry,
                    "compact_threshold": threshold,
                }
                normalized_entries.append(normalized_entry)
                has_compaction_entry = True
            else:
                normalized_entries.append(entry)

        if not has_compaction_entry:
            normalized_entries.append(compaction_entry)

        create_kwargs["context_management"] = normalized_entries

    @staticmethod
    def _response_has_compaction_output(response: Response) -> bool:
        for item in response.output:
            if item.type == "compaction":
                return True
        return False

    @staticmethod
    def _response_output_item_to_context_message(output_item: Any) -> dict[str, Any]:
        return OpenAIResponsesAdapter._normalize_stateless_context_message(
            message=output_item.model_dump(mode="json", exclude_none=True)
        )

    @staticmethod
    def _commit_response_context_messages(
        *,
        context: AgentSessionContext,
        response_output_messages: list[dict[str, Any]],
        next_messages: list[Any],
        compacted: bool,
    ) -> None:
        if compacted:
            context.messages.clear()
        context.messages.extend(response_output_messages)
        context.messages.extend(next_messages)

    @staticmethod
    def _response_has_terminal_image_generation_output(response: Response) -> bool:
        for output_item in response.output:
            item = output_item.model_dump(mode="json", exclude_none=True)
            if item.get("type") != "image_generation_call":
                continue

            status = item.get("status")
            if status in {"completed", "failed", "cancelled"}:
                return True

            if item.get("result") is not None or item.get("images") is not None:
                return True

        return False

    async def compact(
        self,
        *,
        context: AgentSessionContext,
        model: Optional[str] = None,
    ) -> None:
        if model is None:
            model = self.default_model()
        if not context.messages:
            return
        instructions = self._compose_instructions(context=context)
        openai = self.get_openai_client()
        response = await openai.responses.compact(
            model=model,
            input=[*context.messages],
            instructions=instructions or NOT_GIVEN,
        )
        context.messages.clear()
        context.messages.extend(
            [*(x.model_dump(mode="json", exclude_none=True) for x in response.output)]
        )
        if isinstance(context, OpenAIResponsesSessionContext):
            context.clear_websocket_incremental_state()
        context.last_usage = None
        context.pending_supplemental_usage = None

    def _store_image_generation_usage(
        self,
        *,
        context: AgentSessionContext,
        usage: object,
        model: str,
    ) -> None:
        usage_dict = normalize_openai_usage(usage)
        if usage_dict is None:
            return
        flattened_usage = preprocess_openai_image_generation_usage(
            model=model,
            usage=usage_dict,
        )
        if flattened_usage is None:
            return

        supplemental_usage = context.pending_supplemental_usage
        if supplemental_usage is None:
            supplemental_usage = {}
            context.pending_supplemental_usage = supplemental_usage
        add_usage_metrics(totals=supplemental_usage, usage=flattened_usage)

    def _store_image_generation_usage_from_response(
        self,
        *,
        context: AgentSessionContext,
        response: Response,
        model: str,
    ) -> None:
        for output_item in response.output:
            item = output_item.model_dump(mode="json")
            if item.get("type") != "image_generation_call":
                continue
            self._store_image_generation_usage(
                context=context,
                usage=item.get("usage"),
                model=model,
            )

    def _store_usage(
        self,
        *,
        context: AgentSessionContext,
        usage: object,
        model: str,
        compacted: bool = False,
        compaction_threshold: int | None = None,
    ) -> None:
        usage_dict = normalize_openai_usage(usage)
        supplemental_usage = context.pending_supplemental_usage
        context.pending_supplemental_usage = None
        if usage_dict is None and supplemental_usage is None:
            return
        if usage_dict is None:
            usage_dict = {}

        flattened_usage = preprocess_openai_usage(model=model, usage=usage_dict)
        if flattened_usage is None:
            flattened_usage = {}
        if isinstance(supplemental_usage, dict):
            add_usage_metrics(totals=flattened_usage, usage=supplemental_usage)
        if len(flattened_usage) == 0:
            return
        context_used_tokens = self._context_used_tokens_from_usage(flattened_usage)
        if compacted and compaction_threshold is not None:
            context_used_tokens = min(context_used_tokens, compaction_threshold)
        context.emit_usage_updated(
            SessionUsage(
                model=model,
                usage=dict(flattened_usage),
                context_window_used=context_used_tokens,
                context_window_size=self._usage_context_window_size(model),
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
                + usage.get("audio_input_tokens", 0.0)
                + usage.get("audio_cached_tokens", 0.0)
                + usage.get("audio_output_tokens", 0.0)
                + usage.get("image_input_tokens", 0.0)
                + usage.get("image_cached_tokens", 0.0)
                + usage.get("image_output_tokens", 0.0)
            ),
        )

    def _should_publish_stream_event(self, *, event: ResponseStreamEvent) -> bool:
        event_type = event.type

        # Prefer response.output_item.added/done for tool items because those carry
        # richer payload details. Suppress duplicate tool lifecycle stream events.
        if event_type in {
            "response.function_call_arguments.delta",
            "response.mcp_call_arguments.delta",
            "response.mcp_call.arguments.delta",
            "response.code_interpreter_call_code.delta",
            "response.shell_call_command.delta",
            "response.custom_tool_call_input.delta",
        } or (
            (
                event_type.startswith("response.apply_patch_call.")
                or event_type.startswith("response.apply_patch_call_")
            )
            and event_type.endswith(".delta")
        ):
            return True
        if event_type in {
            "response.apply_patch_call.in_progress",
            "response.apply_patch_call.completed",
        }:
            return True

        if (
            event_type.startswith("response.mcp_call.")
            or event_type.startswith("response.mcp_list_tools.")
            or event_type.startswith("response.web_search_call.")
            or event_type.startswith("response.file_search_call.")
            or event_type.startswith("response.apply_patch_call.")
            or event_type.startswith("response.apply_patch_call_")
            or event_type.startswith("response.code_interpreter_call.")
            or event_type.startswith("response.custom_tool_call.")
            or event_type.startswith("response.custom_tool_call_input.")
            or event_type.startswith("response.function_call.")
            or event_type.startswith("response.function_call_arguments.")
        ):
            return False

        return True

    async def get_input_tokens(
        self,
        *,
        context: AgentSessionContext,
        model: str,
        toolkits: Optional[list[Toolkit]] = None,
        output_schema: Optional[dict] = None,
    ) -> int:
        tool_bundle = ResponsesToolBundle(
            toolkits=[
                *(toolkits or []),
            ],
            tool_search=self._tool_search,
        )
        open_ai_tools = tool_bundle.to_json()

        if open_ai_tools is None:
            open_ai_tools = NOT_GIVEN

        openai = self.get_openai_client()

        response_name = "response"
        text = NOT_GIVEN
        if output_schema is not None:
            text = {
                "format": {
                    "type": "json_schema",
                    "name": response_name,
                    "schema": output_schema,
                    "strict": True,
                }
            }

        response = await openai.responses.input_tokens.count(
            model=model,
            tools=open_ai_tools,
            instructions=self._compose_instructions(context=context),
            input=context.messages,
            text=text,
        )

        return response.input_tokens

    async def check_for_termination(self, *, context: AgentSessionContext) -> bool:
        latest_phase = self._get_latest_response_phase_from_messages(context=context)
        if latest_phase is not None:
            return latest_phase == "final_answer"

        for message in reversed(context.messages):
            if message.get("type", "message") != "message":
                return False
            return True

        return False

    @staticmethod
    def _get_latest_response_phase_from_messages(
        *, context: AgentSessionContext
    ) -> str | None:
        for message in reversed(context.messages):
            if message.get("type") != "message":
                break

            phase = message.get("phase")
            if phase is None:
                continue

            if isinstance(phase, str):
                return phase

        return None

    def _resolve_tool_choice(
        self,
        *,
        toolkits: list[Toolkit],
        tool_choice: ToolChoice | None,
    ) -> Any:
        if tool_choice is None:
            return NOT_GIVEN

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

        if isinstance(selected_tool, FunctionTool):
            return {
                "type": "function",
                "name": safe_tool_name(selected_tool.name),
            }
        if isinstance(selected_tool, MCPTool):
            return {
                "type": "mcp",
                "server_label": selected_tool.name,
            }
        if isinstance(selected_tool, ShellTool):
            return {"type": "shell"}
        if isinstance(selected_tool, ApplyPatchTool):
            return {"type": "apply_patch"}

        raise RoomException(
            f"tool_choice is not supported for {type(selected_tool).__name__}"
        )

    def get_openai_client(
        self,
        *,
        session: httpx.AsyncClient | None = None,
    ) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        http_client = session
        if http_client is None and self._log_requests:
            http_client = get_logging_httpx_client()
        return get_client(
            base_url=self._base_url,
            api_key=self._api_key,
            user_agent=self._user_agent,
            http_client=http_client,
            session=session,
        )

    def _is_retryable_openai_error(self, *, error: APIError) -> bool:
        if isinstance(error, APIStatusError):
            return (
                error.status_code == 408
                or error.status_code == 409
                or error.status_code == 429
                or error.status_code >= 500
            )
        return True

    def _is_retryable_websocket_transport_error(
        self,
        *,
        error: RoomException,
    ) -> bool:
        return error.code in {ErrorCode.TIMEOUT, ErrorCode.OPERATION_FAILED} and str(
            error
        ).startswith("OpenAI websocket")

    @staticmethod
    def _is_openai_out_of_credits_message(message: str) -> bool:
        return _is_openai_out_of_credits_message(message)

    @classmethod
    def _is_openai_out_of_credits_payload(cls, payload: dict[str, Any]) -> bool:
        status = payload.get("status")
        if status != 402:
            return False

        message = cls._websocket_error_payload_message(payload=payload)
        if message is not None and cls._is_openai_out_of_credits_message(message):
            return True

        error = payload.get("error")
        if not isinstance(error, dict):
            return False

        code = error.get("code")
        return code == "insufficient_quota"

    @classmethod
    def _is_openai_out_of_credits_error(cls, *, error: RoomException) -> bool:
        return error.status_code == 402 and cls._is_openai_out_of_credits_message(
            str(error)
        )

    def _is_retryable_room_error(self, *, error: RoomException) -> bool:
        if self._is_openai_out_of_credits_error(error=error):
            return False
        if _is_openai_non_retryable_request_error_message(str(error)):
            return False
        return True

    @staticmethod
    def _session_metadata_string(
        *,
        context: AgentSessionContext,
        key: str,
    ) -> str | None:
        value = context.metadata.get(key)
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized if normalized != "" else None

    def _retry_correlation_key(self, *, context: AgentSessionContext) -> str:
        turn_id = self._session_metadata_string(context=context, key="turn_id")
        if turn_id is None:
            return "llm.retry"
        return f"llm.retry:{turn_id}"

    @staticmethod
    def _image_dimensions_from_size(size: object) -> tuple[int | None, int | None]:
        if not isinstance(size, str):
            return None, None
        match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", size)
        if match is None:
            return None, None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _image_generation_mime_type(item: dict[str, Any]) -> str:
        output_format = item.get("output_format")
        mime_type = (
            f"image/{output_format.strip().lower()}"
            if isinstance(output_format, str) and output_format.strip() != ""
            else "image/png"
        )
        if mime_type == "image/jpg":
            return "image/jpeg"
        return mime_type

    @staticmethod
    def _participant_name(participant: Participant) -> str:
        name = participant.get_attribute("name")
        if isinstance(name, str):
            return name
        return ""

    async def _persist_image_generation_output_item(
        self,
        *,
        context: AgentSessionContext,
        caller: Participant,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        images_dataset = self._images_dataset
        if item.get("type") != "image_generation_call":
            return item
        result = item.get("result")
        if not isinstance(result, str) or result.strip() == "":
            return item

        try:
            image_bytes = base64.b64decode(result)
        except Exception:
            return item

        mime_type = self._image_generation_mime_type(item)
        width, height = self._image_dimensions_from_size(item.get("size"))
        if images_dataset is None:
            inline_item = dict(item)
            inline_item.pop("result", None)
            inline_item["images"] = [
                {
                    "uri": f"data:{mime_type};base64,{result.strip()}",
                    "mime_type": mime_type,
                    "created_by": self._participant_name(caller),
                    "width": width,
                    "height": height,
                    "status": "completed",
                }
            ]
            return inline_item

        item_id = item.get("id")
        annotations: dict[str, str] = {"source": "openai.responses"}
        for key in ("background", "output_format", "quality", "size", "status"):
            value = item.get(key)
            if isinstance(value, str) and value.strip() != "":
                annotations[key] = value.strip()
        if isinstance(item_id, str) and item_id.strip() != "":
            annotations["item_id"] = item_id.strip()
        thread_id = self._session_metadata_string(context=context, key="thread_id")
        if thread_id is not None:
            annotations["thread_id"] = thread_id
        turn_id = self._session_metadata_string(context=context, key="turn_id")
        if turn_id is not None:
            annotations["turn_id"] = turn_id

        saved_image = await images_dataset.save(
            data=image_bytes,
            mime_type=mime_type,
            created_by=self._participant_name(caller),
            annotations=annotations,
        )
        persisted_item = dict(item)
        persisted_item.pop("result", None)
        persisted_item["images"] = [
            {
                "uri": f"dataset://{ImagesDataset.TABLE_NAME}?id={saved_image.id}",
                "mime_type": saved_image.mime_type,
                "created_at": saved_image.created_at,
                "created_by": saved_image.created_by,
                "width": width,
                "height": height,
                "status": "completed",
            }
        ]
        return persisted_item

    async def _prepare_stream_event_for_publish(
        self,
        *,
        context: AgentSessionContext,
        caller: Participant,
        event: ResponseStreamEvent,
    ) -> dict[str, Any]:
        payload = event.model_dump(mode="json", exclude_none=True)
        if event.type != "response.output_item.done":
            return payload
        item = payload.get("item")
        if not isinstance(item, dict):
            return payload
        payload["item"] = await self._persist_image_generation_output_item(
            context=context,
            caller=caller,
            item=item,
        )
        return payload

    def _retry_headline(
        self,
        *,
        error: Exception,
        retry_number: int,
        state: Literal["in_progress", "completed", "failed"],
    ) -> str:
        is_reconnect = isinstance(error, RoomException) and (
            self._is_retryable_websocket_transport_error(error=error)
        )
        if state == "in_progress":
            if is_reconnect:
                return (
                    "Reconnecting to the LLM "
                    f"(retry {retry_number}/{self._max_retries})"
                )
            return (
                f"Retrying the LLM request (retry {retry_number}/{self._max_retries})"
            )

        if state == "completed":
            if is_reconnect:
                return "Reconnected to the LLM"
            return "LLM request retry succeeded"

        if is_reconnect:
            return "Unable to reconnect to the LLM"
        return "LLM request retry failed"

    def _retry_detail_lines(
        self,
        *,
        error: Exception,
        retry_number: int,
        state: Literal["in_progress", "completed", "failed"],
        delay_seconds: float | None = None,
    ) -> list[str]:
        if state == "completed":
            if retry_number == 1:
                return ["Recovered after 1 retry."]
            return [f"Recovered after {retry_number} retries."]

        detail_lines: list[str] = []
        if state == "in_progress" and delay_seconds is not None:
            detail_lines.append(
                f"Retry {retry_number} of {self._max_retries} in {delay_seconds:.2f}s."
            )
        if state == "failed":
            detail_lines.append(
                f"Retry budget exhausted after {retry_number} attempts."
            )

        error_message = str(error).strip()
        if error_message != "":
            detail_lines.append(f"Last error: {error_message}")
        return detail_lines

    def _emit_retry_event(
        self,
        *,
        context: AgentSessionContext,
        event_handler: Callable[[dict[str, Any]], None] | None,
        error: Exception,
        retry_number: int,
        state: Literal["in_progress", "completed", "failed"],
        delay_seconds: float | None = None,
    ) -> None:
        if event_handler is None:
            return

        event: dict[str, Any] = {
            "type": "agent.event",
            "source": self._provider,
            "name": "openai.retry",
            "kind": "message",
            "state": state,
            "method": "openai.retry",
            "summary": self._retry_headline(
                error=error,
                retry_number=retry_number,
                state=state,
            ),
            "headline": self._retry_headline(
                error=error,
                retry_number=retry_number,
                state=state,
            ),
            "details": self._retry_detail_lines(
                error=error,
                retry_number=retry_number,
                state=state,
                delay_seconds=delay_seconds,
            ),
            "correlation_key": self._retry_correlation_key(context=context),
            "append_details": True,
        }
        turn_id = self._session_metadata_string(context=context, key="turn_id")
        if turn_id is not None:
            event["turn_id"] = turn_id
        event_handler(event)

    def _retry_delay_seconds(self, *, retry_number: int, error: Exception) -> float:
        if isinstance(error, APIStatusError):
            retry_after = error.response.headers.get("retry-after")
            if retry_after is not None:
                retry_after = retry_after.strip()
                if retry_after != "":
                    try:
                        retry_after_seconds = float(retry_after)
                        if retry_after_seconds > 0:
                            return min(
                                retry_after_seconds,
                                self._max_retry_backoff_seconds,
                            )
                    except ValueError:
                        pass

        return min(
            self._default_retry_backoff_seconds * (2 ** (retry_number - 1)),
            self._max_retry_backoff_seconds,
        )

    def _log_retry(
        self,
        *,
        error: Exception,
        retry_number: int,
        delay_seconds: float,
    ) -> None:
        log_message = "openai request failed, retrying"
        if isinstance(
            error, RoomException
        ) and self._is_retryable_websocket_transport_error(error=error):
            log_message = "openai websocket request failed, retrying"

        request_id = None
        if isinstance(error, APIStatusError):
            request_id = error.request_id

        if request_id:
            logger.warning(
                "%s (%s/%s) in %.2fs (request_id=%s): %s",
                log_message,
                retry_number,
                self._max_retries,
                delay_seconds,
                request_id,
                error,
            )
        else:
            logger.warning(
                "%s (%s/%s) in %.2fs: %s",
                log_message,
                retry_number,
                self._max_retries,
                delay_seconds,
                error,
            )

    async def _create_response_with_retries(
        self,
        *,
        openai: AsyncOpenAI,
        create_kwargs: dict,
    ):
        retry_number = 0
        while True:
            try:
                return await openai.responses.create(**create_kwargs)
            except APIError as error:
                if not self._is_retryable_openai_error(error=error):
                    raise
                if retry_number >= self._max_retries:
                    raise

                retry_number += 1
                delay_seconds = self._retry_delay_seconds(
                    retry_number=retry_number,
                    error=error,
                )
                self._log_retry(
                    error=error,
                    retry_number=retry_number,
                    delay_seconds=delay_seconds,
                )
                await asyncio.sleep(delay_seconds)

    @staticmethod
    def _http_base_url_to_ws_responses_url(base_url: str) -> str:
        parsed = urlparse(base_url)
        if parsed.scheme == "https":
            ws_scheme = "wss"
        elif parsed.scheme == "http":
            ws_scheme = "ws"
        elif parsed.scheme in ("ws", "wss"):
            ws_scheme = parsed.scheme
        else:
            raise RoomException(
                f"unsupported OpenAI base URL scheme for websocket mode: {parsed.scheme}"
            )

        path = parsed.path.rstrip("/") + "/responses"
        return urlunparse(
            (
                ws_scheme,
                parsed.netloc,
                path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

    @staticmethod
    def _coerce_response_stream_event(payload: dict) -> ResponseStreamEvent:
        payload_type = payload.get("type")
        if payload_type == "response.done":
            payload = {**payload, "type": "response.completed"}

        try:
            event = construct_type(value=payload, type_=ResponseStreamEvent)
        except Exception as error:
            raise RoomException(
                f"unable to parse websocket response event '{payload_type}': {error}"
            ) from error
        if not isinstance(event, BaseModel):
            raise RoomException(
                f"unable to parse websocket response event as ResponseStreamEvent: {payload_type}"
            )
        return cast(ResponseStreamEvent, event)

    @staticmethod
    def _response_id_from_payload(payload: dict[str, Any]) -> str | None:
        raw_response_id = payload.get("response_id")
        if isinstance(raw_response_id, str):
            response_id = raw_response_id.strip()
            if response_id != "":
                return response_id

        response = payload.get("response")
        if not isinstance(response, dict):
            return None

        raw_response_object_id = response.get("id")
        if not isinstance(raw_response_object_id, str):
            return None

        response_object_id = raw_response_object_id.strip()
        return response_object_id if response_object_id != "" else None

    @classmethod
    def _payload_matches_response(
        cls,
        *,
        payload: dict[str, Any],
        response_id: str | None,
    ) -> bool:
        if response_id is None:
            return True

        payload_response_id = cls._response_id_from_payload(payload)
        return payload_response_id is None or payload_response_id == response_id

    @staticmethod
    def _is_terminal_response_payload(payload: dict[str, Any]) -> bool:
        payload_type = payload.get("type")
        return payload_type in {
            "response.completed",
            "response.done",
            "response.failed",
            "response.incomplete",
        }

    @staticmethod
    def _websocket_close_error(
        *,
        websocket: aiohttp.ClientWebSocketResponse,
        message: str,
    ) -> RoomException:
        error = websocket.exception()
        close_code = websocket.close_code

        suffix = ""
        if close_code is not None:
            suffix = f" (close_code={close_code})"

        if error is not None:
            detail = str(error).strip()
            if detail != "":
                message = f"{message}{suffix}: {detail}"
            else:
                message = f"{message}{suffix}: {error!r}"
        else:
            message = f"{message}{suffix}"

        if isinstance(error, TimeoutError):
            return RoomException(
                message,
                status_code=408,
                code=ErrorCode.TIMEOUT,
            )

        return RoomException(
            message,
            status_code=503,
            code=ErrorCode.OPERATION_FAILED,
        )

    @staticmethod
    def _websocket_error_payload_message(*, payload: dict[str, Any]) -> str | None:
        message = payload.get("message")
        if isinstance(message, str):
            return message

        error = payload.get("error")
        if not isinstance(error, dict):
            return None

        nested_message = error.get("message")
        if isinstance(nested_message, str):
            return nested_message
        return None

    async def _receive_websocket_payload(
        self,
        *,
        websocket: aiohttp.ClientWebSocketResponse,
        request_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        while True:
            msg = await websocket.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError as error:
                    raise RoomException(
                        f"OpenAI websocket returned invalid JSON: {error}"
                    ) from error
                if not isinstance(payload, dict):
                    continue

                payload_type = payload.get("type")
                if self._log_requests:
                    logger.info("<== WS event=%s", payload_type)
                    logger.info("body=%s", _safe_json_for_log(payload))

                if payload_type == "error":
                    status = payload.get("status")
                    is_out_of_credits = self._is_openai_out_of_credits_payload(payload)
                    should_log_request_payload = not is_out_of_credits and (
                        self._log_requests
                        or isinstance(status, int)
                        and not isinstance(status, bool)
                        and 400 <= status < 500
                    )
                    if should_log_request_payload and request_payload is not None:
                        logger.error(
                            "OpenAI websocket error request body=%s",
                            _safe_json_for_log(request_payload),
                        )

                    message = self._websocket_error_payload_message(payload=payload)
                    if is_out_of_credits:
                        raise RoomException(
                            _OPENAI_OUT_OF_CREDITS_MESSAGE,
                            status_code=402,
                            code=ErrorCode.INVALID_REQUEST,
                        )
                    if message is not None:
                        raise RoomException(f"Error from OpenAI websocket: {message}")
                    raise RoomException(
                        f"Error from OpenAI websocket: {json.dumps(payload)}"
                    )

                return payload

            if msg.type == aiohttp.WSMsgType.CLOSED:
                raise self._websocket_close_error(
                    websocket=websocket,
                    message="OpenAI websocket closed unexpectedly",
                )
            if msg.type == aiohttp.WSMsgType.CLOSE:
                raise self._websocket_close_error(
                    websocket=websocket,
                    message="OpenAI websocket closed unexpectedly",
                )
            if msg.type == aiohttp.WSMsgType.CLOSING:
                raise self._websocket_close_error(
                    websocket=websocket,
                    message="OpenAI websocket is closing unexpectedly",
                )
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise self._websocket_close_error(
                    websocket=websocket,
                    message="OpenAI websocket error",
                )
            if msg.type == aiohttp.WSMsgType.BINARY:
                raise RoomException(
                    "OpenAI websocket returned unexpected binary message"
                )

    @staticmethod
    async def _run_cancelled_cleanup(
        *,
        cleanup: Callable[[], Awaitable[None]],
    ) -> None:
        current_task = asyncio.current_task()
        suppressed_cancellations = 0

        if current_task is not None:
            while current_task.cancelling():
                current_task.uncancel()
                suppressed_cancellations += 1

        try:
            while True:
                try:
                    await cleanup()
                    return
                except asyncio.CancelledError:
                    if current_task is None:
                        raise
                    while current_task.cancelling():
                        current_task.uncancel()
                        suppressed_cancellations += 1
        finally:
            if current_task is not None:
                for _ in range(suppressed_cancellations):
                    current_task.cancel()

    @staticmethod
    def _build_websocket_request_payload(create_kwargs: dict) -> dict:
        payload = {
            "type": "response.create",
        }
        for key, value in create_kwargs.items():
            if key in ("stream", "extra_headers", "background"):
                continue
            if value is NOT_GIVEN:
                continue
            payload[key] = value
        return payload

    def _websocket_headers(
        self, *, openai: AsyncOpenAI, extra_headers: dict[str, str]
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in openai.default_headers.items():
            if isinstance(value, str):
                headers[key] = value
        headers.update(extra_headers)
        headers.pop("Content-Type", None)
        headers.pop("Content-Length", None)
        return headers

    async def _create_response_websocket_stream(
        self,
        *,
        context: AgentSessionContext,
        openai: AsyncOpenAI,
        create_kwargs: dict,
        extra_headers: dict[str, str],
    ) -> AsyncIterable[ResponseStreamEvent]:
        if not isinstance(context, OpenAIResponsesSessionContext):
            raise RoomException(
                "websocket mode requires OpenAIResponsesSessionContext from create_session()"
            )

        websocket_url = self._http_base_url_to_ws_responses_url(str(openai.base_url))
        websocket_headers = self._websocket_headers(
            openai=openai,
            extra_headers=extra_headers,
        )
        try:
            websocket = await context.ensure_websocket(
                url=websocket_url,
                headers=websocket_headers,
            )
        except RoomException:
            raise
        except aiohttp.WSServerHandshakeError as error:
            raise RoomException(
                OpenAIResponsesSessionContext._handshake_error_message(error),
                status_code=error.status,
            ) from error
        except aiohttp.ClientResponseError as error:
            raise RoomException(
                OpenAIResponsesSessionContext._status_error_message(
                    status=error.status,
                    headers=error.headers,
                    message=error.message,
                ),
                status_code=error.status,
            ) from error

        request_payload = self._build_websocket_request_payload(create_kwargs)
        if self._log_requests:
            logger.info("==> WS %s", websocket_url)
            logger.info(
                "headers=%s",
                json.dumps(_redact_log_headers(websocket_headers), indent=2),
            )
            logger.info("body=%s", _safe_json_for_log(request_payload))
        await websocket.send_str(json.dumps(request_payload))

        async def event_stream():
            response_id: str | None = None
            while True:
                payload = await self._receive_websocket_payload(
                    websocket=websocket,
                    request_payload=request_payload,
                )

                payload_response_id = self._response_id_from_payload(payload)
                if response_id is None and payload_response_id is not None:
                    response_id = payload_response_id
                elif not self._payload_matches_response(
                    payload=payload,
                    response_id=response_id,
                ):
                    logger.warning(
                        "ignoring websocket event for response %s while waiting for response %s",
                        payload_response_id,
                        response_id,
                    )
                    continue

                event = self._coerce_response_stream_event(payload)
                yield event
                if self._is_terminal_response_payload(payload):
                    return

        return event_stream()

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
        event_handler: Optional[Callable[[dict[str, Any]], None]] = None,
        steering_callback: SteeringCallback | None = None,
        on_behalf_of: Optional[Participant] = None,
        tool_choice: ToolChoice | None = None,
        options: Optional[dict] = None,
    ):
        if model is None:
            model = self.default_model()

        context.turn_count += 1

        if self.needs_compaction(context=context):
            logger.error("llm request needs compaction, compacting request")
            await self.compact(
                context=context,
                model=model,
            )

        with tracer.start_as_current_span("llm.turn") as span:
            span.set_attributes({"session_id": context.id, "api": "responses"})
            async with contextlib.AsyncExitStack() as exit_stack:
                if isinstance(context, OpenAIResponsesSessionContext):
                    await exit_stack.enter_async_context(context._request_lock)

                tool_adapter = self._make_tool_response_adapter()
                context_messages_snapshot = copy.deepcopy(context.messages)
                loaded_toolkits: list[Toolkit] = []
                iteration_committed = False

                def restore_context_snapshot() -> None:
                    context.messages.clear()
                    context.messages.extend(copy.deepcopy(context_messages_snapshot))

                def commit_local_tool_boundary(
                    *,
                    response_output_messages: list[dict[str, Any]],
                    next_messages: list[Any],
                ) -> None:
                    nonlocal iteration_committed
                    restore_context_snapshot()
                    context.messages.extend(copy.deepcopy(response_output_messages))
                    context.messages.extend(copy.deepcopy(next_messages))
                    iteration_committed = True

                try:
                    while True:
                        with tracer.start_as_current_span("llm.turn.iteration") as span:
                            span.set_attributes(
                                {"model": model, "provider": self._provider}
                            )

                            response_name = "response"
                            effective_toolkits = [*toolkits, *loaded_toolkits]

                            # We need to do this inside the loop because tools can change mid loop
                            # for example computer use adds goto tools after the first interaction
                            tool_bundle = ResponsesToolBundle(
                                toolkits=[
                                    *effective_toolkits,
                                ],
                                tool_call_approval_handler=self._tool_call_approval_handler,
                                tool_search=self._tool_search,
                            )
                            self._set_function_tool_name_resolver(
                                event_handler=event_handler,
                                resolver=tool_bundle.resolve_function_tool_name,
                            )
                            open_ai_tools = tool_bundle.to_json()

                            if open_ai_tools is None:
                                open_ai_tools = NOT_GIVEN

                            ptc = self._parallel_tool_calls
                            extra = {}
                            if ptc is not None and not model.startswith("o"):
                                extra["parallel_tool_calls"] = ptc
                                span.set_attribute("parallel_tool_calls", ptc)
                            else:
                                span.set_attribute("parallel_tool_calls", False)

                            text = NOT_GIVEN
                            if output_schema is not None:
                                span.set_attribute("response_format", "json_schema")
                                text = {
                                    "format": {
                                        "type": "json_schema",
                                        "name": response_name,
                                        "schema": output_schema,
                                        "strict": True,
                                    }
                                }
                            else:
                                span.set_attribute("response_format", "text")

                            instructions = self._compose_instructions(context=context)

                            stream = (
                                self._mode == "websocket" or event_handler is not None
                            )
                            context_messages_snapshot = copy.deepcopy(context.messages)
                            iteration_committed = False

                        with tracer.start_as_current_span("llm.invoke") as span:
                            response_options = copy.deepcopy(self._response_options)
                            if response_options is None:
                                response_options = {}

                            if self._reasoning_effort is not None:
                                response_options["reasoning"] = {
                                    "effort": self._reasoning_effort,
                                    "summary": "detailed",
                                }
                            self._ensure_encrypted_reasoning_include(response_options)
                            request_options = copy.deepcopy(options or {})
                            request_options.pop("output_modalities", None)
                            request_options.pop("voice", None)

                            extra_headers = {}
                            extra_headers.update(
                                llm_annotation_headers(self._annotations)
                            )
                            if on_behalf_of is not None:
                                on_behalf_of_name = on_behalf_of.get_attribute("name")
                                caller_name = caller.get_attribute("name")
                                logger.info(
                                    "%s making openai request on behalf of %s",
                                    caller_name,
                                    on_behalf_of_name,
                                )
                                if isinstance(on_behalf_of_name, str):
                                    extra_headers["Meshagent-On-Behalf-Of"] = (
                                        on_behalf_of_name
                                    )

                            openai = self.get_openai_client()
                            incremental_input = None
                            previous_response_id = None
                            if self._mode == "websocket" and isinstance(
                                context, OpenAIResponsesSessionContext
                            ):
                                incremental_request = (
                                    context.get_websocket_incremental_request()
                                )
                                if incremental_request is not None:
                                    previous_response_id, incremental_input = (
                                        incremental_request
                                    )
                            create_kwargs = {
                                "extra_headers": extra_headers,
                                "stream": stream,
                                "model": model,
                                "input": (
                                    incremental_input
                                    if incremental_input is not None
                                    else context.messages
                                ),
                                "tools": open_ai_tools,
                                "tool_choice": self._resolve_tool_choice(
                                    toolkits=effective_toolkits,
                                    tool_choice=tool_choice,
                                ),
                                "text": text,
                                "instructions": instructions or NOT_GIVEN,
                                "max_output_tokens": self.max_output_tokens,
                                **response_options,
                                **request_options,
                            }
                            if previous_response_id is not None:
                                create_kwargs["previous_response_id"] = (
                                    previous_response_id
                                )
                            create_kwargs["store"] = False
                            normalized_extra_headers = llm_annotation_headers(
                                self._annotations
                            )
                            normalized_extra_headers.update(
                                normalize_extra_headers(
                                    create_kwargs.get("extra_headers")
                                )
                            )
                            create_kwargs["extra_headers"] = normalized_extra_headers
                            self._add_auto_compaction_entry(
                                create_kwargs=create_kwargs,
                                model=model,
                            )
                            response: Content | None = None
                            if not stream:
                                if self._mode == "websocket":
                                    response = (
                                        await self._create_response_websocket_stream(
                                            context=context,
                                            openai=openai,
                                            create_kwargs=create_kwargs,
                                            extra_headers=normalized_extra_headers,
                                        )
                                    )
                                else:
                                    response = await self._create_response_with_retries(
                                        openai=openai,
                                        create_kwargs=create_kwargs,
                                    )

                            if self._mode == "request" and not stream:
                                if response is None:
                                    raise RuntimeError("response must be available")
                                response_compacted = (
                                    self._response_has_compaction_output(response)
                                )
                                self._store_image_generation_usage_from_response(
                                    context=context,
                                    response=response,
                                    model=model,
                                )
                                self._store_usage(
                                    context=context,
                                    usage=response.usage,
                                    model=model,
                                    compacted=response_compacted,
                                    compaction_threshold=self._effective_compaction_threshold(
                                        model=model
                                    ),
                                )

                            async def handle_message(message: BaseModel):
                                with tracer.start_as_current_span(
                                    "llm.handle_response"
                                ) as span:
                                    span.set_attributes(
                                        {
                                            "type": message.type,
                                            "message": safe_model_dump(message),
                                        }
                                    )

                                    if (
                                        message.type == "tool_search_call"
                                        and self._tool_search == "client"
                                        and isinstance(
                                            message,
                                            ResponseToolSearchCall,
                                        )
                                    ):
                                        (
                                            output,
                                            found_toolkits,
                                        ) = await self._client_tool_search_output(
                                            context=context,
                                            caller=caller,
                                            tool_search_call=message,
                                            toolkits=effective_toolkits,
                                            on_behalf_of=on_behalf_of,
                                        )
                                        existing_toolkit_signatures = {
                                            (
                                                toolkit.name,
                                                tuple(
                                                    tool.name for tool in toolkit.tools
                                                ),
                                            )
                                            for toolkit in effective_toolkits
                                        }
                                        for found_toolkit in found_toolkits:
                                            found_signature = (
                                                found_toolkit.name,
                                                tuple(
                                                    tool.name
                                                    for tool in found_toolkit.tools
                                                ),
                                            )
                                            if (
                                                found_signature
                                                not in existing_toolkit_signatures
                                            ):
                                                loaded_toolkits.append(found_toolkit)
                                                existing_toolkit_signatures.add(
                                                    found_signature
                                                )
                                        return [output], False

                                    if message.type == "function_call":
                                        tasks = []

                                        async def do_tool_call(
                                            tool_call: ResponseFunctionToolCall,
                                        ):
                                            try:
                                                with tracer.start_as_current_span(
                                                    "llm.handle_tool_call"
                                                ) as span:
                                                    span.set_attributes(
                                                        {
                                                            "id": tool_call.id,
                                                            "name": tool_call.name,
                                                            "call_id": tool_call.call_id,
                                                            "arguments": json.dumps(
                                                                tool_call.arguments
                                                            ),
                                                        }
                                                    )

                                                    tool_item_id = (
                                                        tool_call.id
                                                        if isinstance(tool_call.id, str)
                                                        else None
                                                    )

                                                    def handle_tool_event(event: dict):
                                                        if event_handler is None:
                                                            return
                                                        if (
                                                            tool_item_id is not None
                                                            and "item_id" not in event
                                                        ):
                                                            event = {
                                                                **event,
                                                                "item_id": tool_item_id,
                                                            }
                                                        event_handler(event)

                                                    tool_context = ToolContext(
                                                        caller=caller,
                                                        on_behalf_of=on_behalf_of,
                                                        event_handler=handle_tool_event,
                                                    )
                                                    if event_handler is not None:
                                                        event_handler(
                                                            {
                                                                "type": "meshagent.handler.added",
                                                                "item": tool_call.model_dump(
                                                                    mode="json"
                                                                ),
                                                            }
                                                        )
                                                    tool_result = (
                                                        await tool_bundle.execute(
                                                            context=tool_context,
                                                            tool_call=tool_call,
                                                        )
                                                    )
                                                    if isinstance(
                                                        tool_result, AsyncIterable
                                                    ):
                                                        tool_response = await _consume_streaming_tool_result(
                                                            tool_name=tool_call.name,
                                                            tool_call_id=tool_call.call_id,
                                                            item_id=tool_call.id,
                                                            stream=tool_result,
                                                            event_handler=event_handler,
                                                        )
                                                    else:
                                                        tool_response = tool_result
                                                    if event_handler is not None:
                                                        handler_result = None
                                                        if isinstance(
                                                            tool_response, JsonContent
                                                        ):
                                                            handler_result = (
                                                                tool_response.json
                                                            )
                                                        elif isinstance(
                                                            tool_response, TextContent
                                                        ):
                                                            handler_result = (
                                                                tool_response.text
                                                            )
                                                        elif isinstance(
                                                            tool_response,
                                                            (dict, list, str),
                                                        ):
                                                            handler_result = (
                                                                tool_response
                                                            )
                                                        event_handler(
                                                            {
                                                                "type": "meshagent.handler.done",
                                                                "item_id": tool_call.id,
                                                                "result": handler_result,
                                                            }
                                                        )
                                                    return await tool_adapter.create_messages(
                                                        context=context,
                                                        tool_call=tool_call,
                                                        response=tool_response,
                                                    )

                                            except asyncio.CancelledError:
                                                if event_handler is not None:
                                                    event_handler(
                                                        {
                                                            "type": "meshagent.handler.done",
                                                            "item_id": tool_call.id,
                                                            "error": "cancelled",
                                                        }
                                                    )
                                                raise
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
                                                if event_handler is not None:
                                                    event_handler(
                                                        {
                                                            "type": "meshagent.handler.done",
                                                            "item_id": tool_call.id,
                                                            "error": f"{e}",
                                                        }
                                                    )

                                                return [
                                                    {
                                                        "output": json.dumps(
                                                            {
                                                                "error": f"unable to complete tool call: {e}"
                                                            }
                                                        ),
                                                        "call_id": tool_call.call_id,
                                                        "type": "function_call_output",
                                                    }
                                                ]

                                        tasks.append(
                                            asyncio.create_task(do_tool_call(message))
                                        )

                                        results = await asyncio.gather(*tasks)

                                        all_results = []
                                        for result in results:
                                            all_results.extend(result)

                                        return all_results, False

                                    elif message.type == "message":
                                        contents = message.content
                                        if output_schema is None:
                                            return [], False
                                        else:
                                            for content in contents:
                                                # First try to parse the result
                                                try:
                                                    full_response = json.loads(
                                                        content.text
                                                    )

                                                # sometimes open ai packs two JSON chunks seperated by newline, check if that's why we couldn't parse
                                                except json.decoder.JSONDecodeError:
                                                    for (
                                                        part
                                                    ) in content.text.splitlines():
                                                        if len(part.strip()) > 0:
                                                            full_response = json.loads(
                                                                part
                                                            )

                                                            try:
                                                                self.validate(
                                                                    response=full_response,
                                                                    output_schema=output_schema,
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
                                                                return [error], False

                                                return [full_response], True

                                    else:
                                        with tracer.start_as_current_span(
                                            "llm.handle_tool_call"
                                        ) as span:
                                            for toolkit in toolkits:
                                                for tool in toolkit.tools:
                                                    if isinstance(
                                                        tool, OpenAIResponsesTool
                                                    ):
                                                        arguments = message.model_dump(
                                                            mode="json"
                                                        )
                                                        span.set_attributes(
                                                            {
                                                                "type": message.type,
                                                                "arguments": safe_json_dump(
                                                                    arguments
                                                                ),
                                                            }
                                                        )

                                                        handlers = tool.get_open_ai_output_handlers()
                                                        if message.type in handlers:
                                                            tool_context = ToolContext(
                                                                caller=caller,
                                                                event_handler=event_handler,
                                                            )

                                                            try:
                                                                publish_handler_events = (
                                                                    message.type
                                                                    != "image_generation_call"
                                                                )
                                                                if (
                                                                    event_handler
                                                                    is not None
                                                                    and publish_handler_events
                                                                ):
                                                                    event_handler(
                                                                        {
                                                                            "type": "meshagent.handler.added",
                                                                            "item": message.model_dump(
                                                                                mode="json"
                                                                            ),
                                                                        }
                                                                    )

                                                                result = await handlers[
                                                                    message.type
                                                                ](
                                                                    tool_context,
                                                                    **arguments,
                                                                )
                                                                if isinstance(
                                                                    result,
                                                                    AsyncIterable,
                                                                ):
                                                                    result = await _consume_streaming_tool_items(
                                                                        tool_name=message.type,
                                                                        tool_call_id=(
                                                                            arguments.get(
                                                                                "call_id"
                                                                            )
                                                                            if isinstance(
                                                                                arguments,
                                                                                dict,
                                                                            )
                                                                            else None
                                                                        ),
                                                                        item_id=(
                                                                            arguments.get(
                                                                                "id"
                                                                            )
                                                                            if isinstance(
                                                                                arguments,
                                                                                dict,
                                                                            )
                                                                            else None
                                                                        ),
                                                                        stream=result,
                                                                        event_handler=event_handler,
                                                                    )

                                                            except Exception as e:
                                                                if (
                                                                    event_handler
                                                                    is not None
                                                                ):
                                                                    event_handler(
                                                                        {
                                                                            "type": "meshagent.handler.done",
                                                                            "error": f"{e}",
                                                                        }
                                                                    )

                                                                raise

                                                            if (
                                                                event_handler
                                                                is not None
                                                                and publish_handler_events
                                                            ):
                                                                done_item = result
                                                                if isinstance(
                                                                    done_item, Content
                                                                ):
                                                                    done_item = done_item.to_json()
                                                                event_handler(
                                                                    {
                                                                        "type": "meshagent.handler.done",
                                                                        "item": done_item,
                                                                    }
                                                                )

                                                            if result is not None:
                                                                span.set_attribute(
                                                                    "result",
                                                                    json.dumps(
                                                                        result,
                                                                        ensure_ascii=False,
                                                                        default=str,
                                                                    ),
                                                                )
                                                                return [result], False

                                                            return [], False

                                            if message.type in {
                                                "compaction",
                                                "reasoning",
                                            }:
                                                logger.debug(
                                                    "OpenAI response handler was not "
                                                    "registered for %s; the item is "
                                                    "handled through response state",
                                                    message.type,
                                                )
                                            else:
                                                logger.warning(
                                                    "OpenAI response handler was "
                                                    "not registered for "
                                                    f"{message.type}"
                                                )

                                    return [], False

                            if not stream:
                                if response is None:
                                    raise RuntimeError("response must be available")
                                final_outputs = []
                                next_messages: list[Any] = []
                                response_output_messages: list[dict[str, Any]] = []
                                restart_after_tool_boundary = False

                                for output_index, message in enumerate(response.output):
                                    response_output_messages.append(
                                        self._response_output_item_to_context_message(
                                            message
                                        )
                                    )
                                    outputs, done = await handle_message(
                                        message=message
                                    )
                                    if done:
                                        final_outputs.extend(outputs)
                                    else:
                                        next_messages.extend(outputs)
                                        if (
                                            steering_callback is not None
                                            and len(outputs) > 0
                                            and output_index < len(response.output) - 1
                                        ):
                                            commit_local_tool_boundary(
                                                response_output_messages=response_output_messages,
                                                next_messages=next_messages,
                                            )
                                            if await steering_callback():
                                                self.on_turn_steer(
                                                    context=context,
                                                    interrupted=False,
                                                )
                                                restart_after_tool_boundary = True
                                                break
                                            restore_context_snapshot()
                                            iteration_committed = False

                                if restart_after_tool_boundary:
                                    continue

                                response_compacted = (
                                    self._response_has_compaction_output(response)
                                )
                                self._commit_response_context_messages(
                                    context=context,
                                    response_output_messages=response_output_messages,
                                    next_messages=next_messages,
                                    compacted=response_compacted,
                                )
                                iteration_committed = True

                                if (
                                    response_compacted
                                    and len(final_outputs) == 0
                                    and len(next_messages) == 0
                                ):
                                    return ""

                                if (
                                    steering_callback is not None
                                    and len(next_messages) > 0
                                ):
                                    if await steering_callback():
                                        self.on_turn_steer(
                                            context=context,
                                            interrupted=False,
                                        )
                                        continue

                                if len(final_outputs) > 0:
                                    return final_outputs[0]

                                if self._response_has_terminal_image_generation_output(
                                    response
                                ):
                                    return ""

                                with tracer.start_as_current_span(
                                    "llm.turn.check_for_termination"
                                ) as span:
                                    term = await self.check_for_termination(
                                        context=context
                                    )
                                    if term:
                                        span.set_attribute("terminate", True)
                                        text = ""
                                        for output in response.output:
                                            if output.type == "message":
                                                for content in output.content:
                                                    text += content.text

                                        return text
                                    else:
                                        span.set_attribute("terminate", False)

                            else:
                                stream_retry_number = 0
                                pending_retry_completion_number: int | None = None
                                pending_retry_error: Exception | None = None
                                while True:
                                    final_outputs = []
                                    all_outputs = []
                                    response_output_messages: list[dict[str, Any]] = []
                                    restart_after_tool_boundary = False
                                    try:
                                        if response is None:
                                            if self._mode == "websocket":
                                                response = await self._create_response_websocket_stream(
                                                    context=context,
                                                    openai=openai,
                                                    create_kwargs=create_kwargs,
                                                    extra_headers=extra_headers,
                                                )
                                            else:
                                                response = (
                                                    await openai.responses.create(
                                                        **create_kwargs
                                                    )
                                                )

                                            if (
                                                pending_retry_completion_number
                                                is not None
                                            ):
                                                self._emit_retry_event(
                                                    context=context,
                                                    event_handler=event_handler,
                                                    error=(
                                                        pending_retry_error
                                                        if pending_retry_error
                                                        is not None
                                                        else RoomException(
                                                            "OpenAI request retry recovered"
                                                        )
                                                    ),
                                                    retry_number=pending_retry_completion_number,
                                                    state="completed",
                                                )
                                                pending_retry_completion_number = None
                                                pending_retry_error = None

                                        async for e in response:
                                            with tracer.start_as_current_span(
                                                "llm.stream.event"
                                            ) as span:
                                                event: ResponseStreamEvent = e
                                                span.set_attributes(
                                                    {
                                                        "type": event.type,
                                                        "event": safe_model_dump(event),
                                                    }
                                                )
                                                if (
                                                    restart_after_tool_boundary
                                                    and event.type
                                                    != "response.completed"
                                                ):
                                                    continue
                                                if (
                                                    self._should_publish_stream_event(
                                                        event=event
                                                    )
                                                    and event_handler is not None
                                                ):
                                                    event_payload = await self._prepare_stream_event_for_publish(
                                                        context=context,
                                                        caller=caller,
                                                        event=event,
                                                    )
                                                    event_handler(event_payload)

                                                if event.type == "response.completed":
                                                    if restart_after_tool_boundary:
                                                        response_compacted = self._response_has_compaction_output(
                                                            event.response
                                                        )
                                                        self._store_image_generation_usage_from_response(
                                                            context=context,
                                                            response=event.response,
                                                            model=model,
                                                        )
                                                        self._store_usage(
                                                            context=context,
                                                            usage=event.response.usage,
                                                            model=model,
                                                            compacted=response_compacted,
                                                            compaction_threshold=self._effective_compaction_threshold(
                                                                model=model
                                                            ),
                                                        )
                                                        break
                                                    response_output_messages = [
                                                        self._response_output_item_to_context_message(
                                                            output
                                                        )
                                                        for output in event.response.output
                                                    ]
                                                    response_compacted = self._response_has_compaction_output(
                                                        event.response
                                                    )
                                                    self._store_image_generation_usage_from_response(
                                                        context=context,
                                                        response=event.response,
                                                        model=model,
                                                    )
                                                    self._store_usage(
                                                        context=context,
                                                        usage=event.response.usage,
                                                        model=model,
                                                        compacted=response_compacted,
                                                        compaction_threshold=self._effective_compaction_threshold(
                                                            model=model
                                                        ),
                                                    )

                                                    self._commit_response_context_messages(
                                                        context=context,
                                                        response_output_messages=response_output_messages,
                                                        next_messages=all_outputs,
                                                        compacted=response_compacted,
                                                    )
                                                    iteration_committed = True
                                                    if isinstance(
                                                        context,
                                                        OpenAIResponsesSessionContext,
                                                    ):
                                                        context.remember_websocket_response(
                                                            response_id=event.response.id,
                                                            incremental_start_index=len(
                                                                context.messages
                                                            )
                                                            - len(all_outputs),
                                                        )

                                                    if (
                                                        response_compacted
                                                        and len(final_outputs) == 0
                                                        and len(all_outputs) == 0
                                                    ):
                                                        return ""

                                                    if (
                                                        steering_callback is not None
                                                        and len(all_outputs) > 0
                                                    ):
                                                        if await steering_callback():
                                                            self.on_turn_steer(
                                                                context=context,
                                                                interrupted=False,
                                                            )
                                                            all_outputs = []
                                                            continue

                                                    if self._response_has_terminal_image_generation_output(
                                                        event.response
                                                    ):
                                                        return ""

                                                    with tracer.start_as_current_span(
                                                        "llm.turn.check_for_termination"
                                                    ) as span:
                                                        term = await self.check_for_termination(
                                                            context=context
                                                        )

                                                        if term:
                                                            span.set_attribute(
                                                                "terminate", True
                                                            )

                                                            text = ""
                                                            for (
                                                                output
                                                            ) in event.response.output:
                                                                if (
                                                                    output.type
                                                                    == "message"
                                                                ):
                                                                    for (
                                                                        content
                                                                    ) in output.content:
                                                                        text += (
                                                                            content.text
                                                                        )

                                                            return text

                                                        span.set_attribute(
                                                            "terminate", False
                                                        )

                                                    all_outputs = []

                                                elif (
                                                    event.type
                                                    == "response.output_item.done"
                                                ):
                                                    response_output_messages.append(
                                                        self._response_output_item_to_context_message(
                                                            event.item
                                                        )
                                                    )
                                                    (
                                                        outputs,
                                                        done,
                                                    ) = await handle_message(
                                                        message=event.item
                                                    )
                                                    if done:
                                                        final_outputs.extend(outputs)
                                                    else:
                                                        for output in outputs:
                                                            all_outputs.append(output)
                                                        if (
                                                            steering_callback
                                                            is not None
                                                            and len(outputs) > 0
                                                        ):
                                                            commit_local_tool_boundary(
                                                                response_output_messages=response_output_messages,
                                                                next_messages=all_outputs,
                                                            )
                                                            if await steering_callback():
                                                                self.on_turn_steer(
                                                                    context=context,
                                                                    interrupted=False,
                                                                )
                                                                final_outputs = []
                                                                all_outputs = []
                                                                restart_after_tool_boundary = True
                                                            else:
                                                                restore_context_snapshot()
                                                                iteration_committed = (
                                                                    False
                                                                )

                                                else:
                                                    for toolkit in toolkits:
                                                        for tool in toolkit.tools:
                                                            if isinstance(
                                                                tool,
                                                                OpenAIResponsesTool,
                                                            ):
                                                                callbacks = tool.get_open_ai_stream_callbacks()

                                                                if (
                                                                    event.type
                                                                    in callbacks
                                                                ):
                                                                    tool_context = ToolContext(
                                                                        caller=caller,
                                                                        event_handler=event_handler,
                                                                    )

                                                                    await callbacks[
                                                                        event.type
                                                                    ](
                                                                        tool_context,
                                                                        **event.to_dict(),
                                                                    )

                                                if len(final_outputs) > 0:
                                                    return final_outputs[0]

                                                if event.type == "response.incomplete":
                                                    response_output_messages = [
                                                        self._response_output_item_to_context_message(
                                                            output
                                                        )
                                                        for output in event.response.output
                                                    ]
                                                    response_compacted = self._response_has_compaction_output(
                                                        event.response
                                                    )
                                                    self._store_image_generation_usage_from_response(
                                                        context=context,
                                                        response=event.response,
                                                        model=model,
                                                    )
                                                    self._store_usage(
                                                        context=context,
                                                        usage=event.response.usage,
                                                        model=model,
                                                        compacted=response_compacted,
                                                        compaction_threshold=self._effective_compaction_threshold(
                                                            model=model
                                                        ),
                                                    )
                                                    self._commit_response_context_messages(
                                                        context=context,
                                                        response_output_messages=response_output_messages,
                                                        next_messages=all_outputs,
                                                        compacted=response_compacted,
                                                    )
                                                    iteration_committed = True
                                                    return ""
                                        if restart_after_tool_boundary:
                                            response = None
                                            continue
                                        break

                                    except APIError as error:
                                        if not self._is_retryable_openai_error(
                                            error=error
                                        ):
                                            if stream_retry_number > 0:
                                                self._emit_retry_event(
                                                    context=context,
                                                    event_handler=event_handler,
                                                    error=error,
                                                    retry_number=stream_retry_number,
                                                    state="failed",
                                                )
                                            raise
                                        if stream_retry_number >= self._max_retries:
                                            if stream_retry_number > 0:
                                                self._emit_retry_event(
                                                    context=context,
                                                    event_handler=event_handler,
                                                    error=error,
                                                    retry_number=stream_retry_number,
                                                    state="failed",
                                                )
                                            raise

                                        stream_retry_number += 1
                                        delay_seconds = self._retry_delay_seconds(
                                            retry_number=stream_retry_number,
                                            error=error,
                                        )
                                        self._log_retry(
                                            error=error,
                                            retry_number=stream_retry_number,
                                            delay_seconds=delay_seconds,
                                        )
                                        self._emit_retry_event(
                                            context=context,
                                            event_handler=event_handler,
                                            error=error,
                                            retry_number=stream_retry_number,
                                            state="in_progress",
                                            delay_seconds=delay_seconds,
                                        )

                                        restore_context_snapshot()
                                        create_kwargs["input"] = context.messages
                                        create_kwargs.pop("previous_response_id", None)
                                        if isinstance(
                                            context,
                                            OpenAIResponsesSessionContext,
                                        ):
                                            context.clear_websocket_incremental_state()
                                        response = None
                                        pending_retry_completion_number = (
                                            stream_retry_number
                                        )
                                        pending_retry_error = error

                                        await asyncio.sleep(delay_seconds)
                                    except RoomException as error:
                                        if not self._is_retryable_room_error(
                                            error=error
                                        ):
                                            if stream_retry_number > 0:
                                                self._emit_retry_event(
                                                    context=context,
                                                    event_handler=event_handler,
                                                    error=error,
                                                    retry_number=stream_retry_number,
                                                    state="failed",
                                                )
                                            raise
                                        if stream_retry_number >= self._max_retries:
                                            if stream_retry_number > 0:
                                                self._emit_retry_event(
                                                    context=context,
                                                    event_handler=event_handler,
                                                    error=error,
                                                    retry_number=stream_retry_number,
                                                    state="failed",
                                                )
                                            raise

                                        stream_retry_number += 1
                                        delay_seconds = self._retry_delay_seconds(
                                            retry_number=stream_retry_number,
                                            error=error,
                                        )
                                        self._log_retry(
                                            error=error,
                                            retry_number=stream_retry_number,
                                            delay_seconds=delay_seconds,
                                        )
                                        self._emit_retry_event(
                                            context=context,
                                            event_handler=event_handler,
                                            error=error,
                                            retry_number=stream_retry_number,
                                            state="in_progress",
                                            delay_seconds=delay_seconds,
                                        )

                                        restore_context_snapshot()
                                        create_kwargs["input"] = context.messages
                                        create_kwargs.pop("previous_response_id", None)
                                        if isinstance(
                                            context,
                                            OpenAIResponsesSessionContext,
                                        ):
                                            context.clear_websocket_incremental_state()
                                        response = None
                                        pending_retry_completion_number = (
                                            stream_retry_number
                                        )
                                        pending_retry_error = error
                                        if self._mode == "websocket" and isinstance(
                                            context, OpenAIResponsesSessionContext
                                        ):
                                            with contextlib.suppress(Exception):
                                                await context.close_websocket()

                                        await asyncio.sleep(delay_seconds)

                except asyncio.CancelledError:
                    if not iteration_committed:
                        restore_context_snapshot()
                    if self._mode == "websocket" and isinstance(
                        context, OpenAIResponsesSessionContext
                    ):
                        with contextlib.suppress(Exception):
                            await self._run_cancelled_cleanup(
                                cleanup=context.close_websocket,
                            )
                    raise
                except APIError as e:
                    raise RoomException(f"Error from OpenAI: {e}")
                finally:
                    self._set_function_tool_name_resolver(
                        event_handler=event_handler,
                        resolver=None,
                    )


class OpenAIResponsesTool(BaseTool):
    def get_open_ai_tool_definitions(self) -> list[dict]:
        return []

    def get_open_ai_stream_callbacks(self) -> dict[str, Callable]:
        return {}

    def get_open_ai_output_handlers(self) -> dict[str, Callable]:
        return {}


DEFAULT_IMAGE_GENERATION_MODEL = "gpt-image-2"


class ImageGenerationTool(OpenAIResponsesTool):
    def __init__(
        self,
        *,
        background: Literal["transparent", "opaque", "auto"] | None = None,
        input_image_mask_url: str | None = None,
        model: str | None = None,
        moderation: str | None = None,
        output_compression: int | None = None,
        output_format: Literal["png", "webp", "jpeg"] | None = None,
        partial_images: int | None = 1,
        quality: Literal["auto", "low", "medium", "high"] | None = None,
        size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] | None = None,
    ):
        super().__init__(name="image_generation")
        self.background = background
        self.input_image_mask_url = input_image_mask_url
        normalized_model = model.strip() if isinstance(model, str) else None
        self.model = normalized_model or DEFAULT_IMAGE_GENERATION_MODEL
        self.moderation = moderation
        self.output_compression = output_compression
        self.output_format = output_format
        self.partial_images = partial_images if partial_images is not None else 1
        self.quality = quality
        self.size = size

    def get_open_ai_tool_definitions(self):
        opts = {"type": "image_generation"}

        if self.background is not None:
            opts["background"] = self.background

        if self.input_image_mask_url is not None:
            opts["input_image_mask"] = {"image_url": self.input_image_mask_url}

        if self.model is not None:
            opts["model"] = self.model

        if self.moderation is not None:
            opts["moderation"] = self.moderation

        if self.output_compression is not None:
            opts["output_compression"] = self.output_compression

        if self.output_format is not None:
            opts["output_format"] = self.output_format

        if self.partial_images is not None:
            opts["partial_images"] = self.partial_images

        if self.quality is not None:
            opts["quality"] = self.quality

        if self.size is not None:
            opts["size"] = self.size

        return [opts]

    def get_open_ai_stream_callbacks(self):
        return {
            "response.image_generation_call.completed": self.on_image_generation_completed,
            "response.image_generation_call.in_progress": self.on_image_generation_in_progress,
            "response.image_generation_call.generating": self.on_image_generation_generating,
            "response.image_generation_call.partial_image": self.on_image_generation_partial,
        }

    def get_open_ai_output_handlers(self):
        return {"image_generation_call": self.handle_image_generated}

    # response.image_generation_call.completed
    async def on_image_generation_completed(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    # response.image_generation_call.in_progress
    async def on_image_generation_in_progress(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    # response.image_generation_call.generating
    async def on_image_generation_generating(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    # response.image_generation_call.partial_image
    async def on_image_generation_partial(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        partial_image_b64: str,
        partial_image_index: int,
        size: str,
        quality: str,
        background: str,
        output_format: str,
        **extra,
    ):
        pass

    async def on_image_generated(
        self,
        context: ToolContext,
        *,
        item_id: str,
        data: bytes,
        status: str,
        size: str,
        quality: str,
        background: str,
        output_format: str,
        **extra,
    ):
        pass

    async def handle_image_generated(
        self,
        context: ToolContext,
        *,
        id: str,
        result: str | None,
        status: str,
        type: str,
        size: str,
        quality: str,
        background: str,
        output_format: str,
        **extra,
    ):
        if result is not None:
            data = base64.b64decode(result)
            await self.on_image_generated(
                context,
                item_id=id,
                data=data,
                status=status,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
            )


MAX_SHELL_OUTPUT_SIZE = 1024 * 100
MAX_SHELL_LOG_LINE_LENGTH = DEFAULT_MAX_LOG_LINE_LENGTH


async def _stream_reader_chunks(
    reader: asyncio.StreamReader | None,
) -> AsyncIterable[bytes]:
    if reader is None:
        return

    while True:
        chunk = await reader.read(4096)
        if chunk == b"":
            break
        yield chunk


async def _collect_shell_output_stream(
    *,
    stream: AsyncIterable[bytes],
    accumulator: StreamOutputAccumulator,
) -> None:
    await collect_output_stream(stream=stream, accumulator=accumulator)


async def _await_output_tasks(*tasks: asyncio.Task[None]) -> None:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception) and not isinstance(
            result, asyncio.CancelledError
        ):
            logger.debug("shell output stream task failed", exc_info=result)


class ShellTool(OpenAIResponsesTool):
    def __init__(
        self,
        *,
        room: RoomClient | None = None,
        name: str = "shell",
        working_dir: Optional[str] = None,
        image: Optional[str] = "python:3.13",
        mounts: Optional[ContainerMountSpec] = DEFAULT_CONTAINER_MOUNT_SPEC,
        env: Optional[dict[str, str]] = None,
    ):
        super().__init__(name=name)
        self._room = room
        self.working_dir = working_dir
        self.image = image
        self.mounts = mounts
        self.env = env
        self._provider: ContainerShellTool | ProcessShellTool | None
        if image is None:
            self._provider = ProcessShellTool(
                name=name,
                working_dir=working_dir,
                env=env,
            )
        elif room is not None:
            self._provider = ContainerShellTool(
                room=room,
                name=name,
                working_dir=working_dir,
                image=image,
                mounts=mounts,
                env=env,
            )
        else:
            self._provider = None

    def get_open_ai_tool_definitions(self):
        return [{"type": "shell"}]

    def get_open_ai_output_handlers(self):
        return {"shell_call": self.handle_shell_call}

    def _provider_context(
        self,
        context: ToolContext,
        *,
        item_id: str,
    ) -> ToolContext:
        def handle_provider_event(event: dict):
            if "item_id" not in event:
                event = {**event, "item_id": item_id}
            context.emit(event)

        return ToolContext(
            caller=context.caller,
            on_behalf_of=context.on_behalf_of,
            event_handler=handle_provider_event,
        )

    def _require_provider(self) -> ContainerShellTool | ProcessShellTool:
        if self._provider is None:
            raise RuntimeError(
                "ShellTool requires room when configured with a container image"
            )
        return self._provider

    async def execute_shell_command(
        self,
        context: ToolContext,
        *,
        commands: list[str],
        item_id: str,
        max_output_length: Optional[int] = None,
        timeout_ms: Optional[int] = None,
    ):
        effective_max_output_length = (
            max_output_length
            if max_output_length is not None
            else MAX_SHELL_OUTPUT_SIZE
        )
        result = await self._require_provider().execute(
            self._provider_context(context, item_id=item_id),
            commands=commands,
            max_output_length=effective_max_output_length,
            timeout_ms=timeout_ms,
        )

        return result["results"]

    async def handle_shell_call(
        self,
        context,
        *,
        id: str,
        action: dict,
        call_id: str,
        status: str,
        type: str,
        **extra,
    ):
        result = await self.execute_shell_command(context, item_id=id, **action)

        output_item = {
            "type": "shell_call_output",
            "call_id": call_id,
            "output": result,
        }

        return output_item


class ContainerFile:
    def __init__(self, *, file_id: str, mime_type: str, container_id: str):
        self.file_id = file_id
        self.mime_type = mime_type
        self.container_id = container_id


class CodeInterpreterTool(OpenAIResponsesTool):
    def __init__(
        self,
        *,
        container_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
    ):
        super().__init__(name="code_interpreter_call")
        self.container_id = container_id
        self.file_ids = file_ids

    def get_open_ai_tool_definitions(self):
        opts = {"type": "code_interpreter"}

        if self.container_id is not None:
            opts["container_id"] = self.container_id

        if self.file_ids is not None:
            if self.container_id is not None:
                raise Exception(
                    "Cannot specify both an existing container and files to upload in a code interpreter tool"
                )

            opts["container"] = {"type": "auto", "file_ids": self.file_ids}

        return [opts]

    def get_open_ai_output_handlers(self):
        return {"code_interpreter_call": self.handle_code_interpreter_call}

    async def on_code_interpreter_result(
        self,
        context: ToolContext,
        *,
        code: str,
        logs: list[str],
        files: list[ContainerFile],
    ):
        pass

    async def handle_code_interpreter_call(
        self,
        context,
        *,
        code: str,
        id: str,
        results: list[dict],
        call_id: str,
        status: str,
        type: str,
        container_id: str,
        **extra,
    ):
        logs = []
        files = []

        for result in results:
            if result.type == "logs":
                logs.append(results["logs"])

            elif result.type == "files":
                files.append(
                    ContainerFile(
                        container_id=container_id,
                        file_id=result["file_id"],
                        mime_type=result["mime_type"],
                    )
                )

        await self.on_code_interpreter_result(
            context, code=code, logs=logs, files=files
        )


class MCPToolDefinition:
    def __init__(
        self,
        *,
        input_schema: dict,
        name: str,
        annotations: dict | None,
        description: str | None,
    ):
        self.input_schema = input_schema
        self.name = name
        self.annotations = annotations
        self.description = description


class MCPServer(BaseModel):
    class Header(BaseModel):
        name: str
        value: str

    server_label: str
    server_url: Optional[str] = None
    allowed_tools: Optional[list[str]] = None
    authorization: Optional[str] = None
    headers: Optional[list[Header]] = None

    # require approval for all tools
    require_approval: Optional[Literal["always", "never"]] = None
    # list of tools that always require approval
    always_require_approval: Optional[list[str]] = None
    # list of tools that never require approval
    never_require_approval: Optional[list[str]] = None

    openai_connector_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_headers(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        headers = value.get("headers")
        if not isinstance(headers, dict):
            return value

        normalized = dict(value)
        normalized["headers"] = [
            {"name": str(key), "value": str(header_value)}
            for key, header_value in headers.items()
        ]
        return normalized


class MCPTool(OpenAIResponsesTool):
    def __init__(
        self,
        *,
        servers: list[MCPServer],
        name: str = "mcp",
        tool_call_approval_handler: ToolCallApprovalHandler | None = None,
    ):
        super().__init__(name=name)
        self.servers = servers
        self._tool_call_approval_handler = tool_call_approval_handler

    def with_tool_call_approval_handler(
        self, handler: ToolCallApprovalHandler | None
    ) -> "MCPTool":
        return MCPTool(
            name=self.name,
            servers=[*self.servers],
            tool_call_approval_handler=handler,
        )

    def get_open_ai_tool_definitions(self):
        defs = []
        for server in self.servers:
            opts = {
                "type": "mcp",
                "server_label": server.server_label,
            }

            if server.server_url is not None:
                opts["server_url"] = server.server_url

            if server.openai_connector_id is not None:
                opts["connector_id"] = server.openai_connector_id

            if server.allowed_tools is not None:
                opts["allowed_tools"] = server.allowed_tools

            opts["authorization"] = server.authorization

            if server.headers is not None:
                opts["headers"] = {
                    header.name: header.value for header in server.headers
                }

            if (
                server.always_require_approval is not None
                or server.never_require_approval is not None
            ):
                opts["require_approval"] = {}

                if server.always_require_approval is not None:
                    opts["require_approval"]["always"] = {
                        "tool_names": server.always_require_approval
                    }

                if server.never_require_approval is not None:
                    opts["require_approval"]["never"] = {
                        "tool_names": server.never_require_approval
                    }

            if server.require_approval:
                opts["require_approval"] = server.require_approval

            defs.append(opts)

        return defs

    def get_open_ai_stream_callbacks(self):
        return {
            "response.mcp_list_tools.in_progress": self.on_mcp_list_tools_in_progress,
            "response.mcp_list_tools.failed": self.on_mcp_list_tools_failed,
            "response.mcp_list_tools.completed": self.on_mcp_list_tools_completed,
            "response.mcp_call.in_progress": self.on_mcp_call_in_progress,
            "response.mcp_call.failed": self.on_mcp_call_failed,
            "response.mcp_call.completed": self.on_mcp_call_completed,
            "response.mcp_call.arguments.done": self.on_mcp_call_arguments_done,
            "response.mcp_call.arguments.delta": self.on_mcp_call_arguments_delta,
        }

    async def on_mcp_list_tools_in_progress(
        self, context: ToolContext, *, sequence_number: int, type: str, **extra
    ):
        pass

    async def on_mcp_list_tools_failed(
        self, context: ToolContext, *, sequence_number: int, type: str, **extra
    ):
        pass

    async def on_mcp_list_tools_completed(
        self, context: ToolContext, *, sequence_number: int, type: str, **extra
    ):
        pass

    async def on_mcp_call_in_progress(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_mcp_call_failed(
        self, context: ToolContext, *, sequence_number: int, type: str, **extra
    ):
        pass

    async def on_mcp_call_completed(
        self, context: ToolContext, *, sequence_number: int, type: str, **extra
    ):
        pass

    async def on_mcp_call_arguments_done(
        self,
        context: ToolContext,
        *,
        arguments: dict,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_mcp_call_arguments_delta(
        self,
        context: ToolContext,
        *,
        delta: dict,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    def get_open_ai_output_handlers(self):
        return {
            "mcp_call": self.handle_mcp_call,
            "mcp_list_tools": self.handle_mcp_list_tools,
            "mcp_approval_request": self.handle_mcp_approval_request,
        }

    async def on_mcp_list_tools(
        self,
        context: ToolContext,
        *,
        server_label: str,
        tools: list[MCPToolDefinition],
        error: str | None,
        **extra,
    ):
        pass

    async def handle_mcp_list_tools(
        self,
        context,
        *,
        id: str,
        server_label: str,
        tools: list,
        type: str,
        error: str | None = None,
        **extra,
    ):
        mcp_tools = []
        for tool in tools:
            mcp_tools.append(
                MCPToolDefinition(
                    input_schema=tool["input_schema"],
                    name=tool["name"],
                    annotations=tool["annotations"],
                    description=tool["description"],
                )
            )

        await self.on_mcp_list_tools(
            context, server_label=server_label, tools=mcp_tools, error=error
        )

    async def on_mcp_call(
        self,
        context: ToolContext,
        *,
        name: str,
        arguments: str,
        server_label: str,
        error: str | None,
        output: str | None,
        **extra,
    ):
        pass

    async def handle_mcp_call(
        self,
        context,
        *,
        arguments: str,
        id: str,
        name: str,
        server_label: str,
        type: str,
        error: str | None,
        output: str | None,
        **extra,
    ):
        await self.on_mcp_call(
            context,
            name=name,
            arguments=arguments,
            server_label=server_label,
            error=error,
            output=output,
        )

    async def on_mcp_approval_request(
        self,
        context: ToolContext,
        *,
        id: str,
        name: str,
        arguments: str,
        server_label: str,
        **extra,
    ) -> bool:
        del extra

        handler = self._tool_call_approval_handler
        if handler is None:
            return True

        parsed_arguments: dict[str, Any] | None = None
        try:
            raw_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            raw_arguments = None

        if isinstance(raw_arguments, dict):
            parsed_arguments = raw_arguments

        return await handler(
            context,
            ToolCallApprovalRequest(
                item_id=id,
                toolkit=server_label,
                tool=name,
                arguments=parsed_arguments,
            ),
        )

    async def handle_mcp_approval_request(
        self,
        context: ToolContext,
        *,
        arguments: str,
        id: str,
        name: str,
        server_label: str,
        type: str,
        **extra,
    ):
        logger.info(f"approval requested for MCP tool {server_label}.{name}")
        should_approve = await self.on_mcp_approval_request(
            context,
            id=id,
            arguments=arguments,
            name=name,
            server_label=server_label,
        )
        if should_approve:
            logger.info(f"approval granted for MCP tool {server_label}.{name}")
            return {
                "type": "mcp_approval_response",
                "approve": True,
                "approval_request_id": id,
            }
        else:
            logger.info(f"approval denied for MCP tool {server_label}.{name}")
            return {
                "type": "mcp_approval_response",
                "approve": False,
                "approval_request_id": id,
            }


def _merge_mcp_server_configs(
    *,
    static_servers: list[MCPServerConfig],
    client_options: dict | None,
) -> list[MCPServerConfig]:
    merged_by_label: dict[str, MCPServerConfig] = {
        server.server_label: server for server in static_servers
    }
    if client_options is None:
        return list(merged_by_label.values())

    options = MCPToolkitClientOptions.model_validate(client_options)
    for server in options.servers:
        merged_by_label[server.server_label] = server
    return [
        apply_mcp_proxy_config(
            server=server,
            proxy_config=options.meshagent_proxy_config,
        )
        for server in merged_by_label.values()
    ]


class OpenAIResponsesMCPToolkit(Toolkit):
    def __init__(
        self,
        *,
        servers: list[MCPServerConfig] | None = None,
        title: str | None = None,
        description: str | None = None,
        hidden: bool = False,
    ) -> None:
        super().__init__(
            name="mcp",
            title=title,
            description=description,
            tools=[],
            client_options=MCPToolkitClientOptions.model_json_schema(),
            hidden=hidden,
        )
        self._servers = [*(servers or [])]

    def get_tools(self, *, client_options: Optional[dict] = None) -> list[BaseTool]:
        servers = _merge_mcp_server_configs(
            static_servers=self._servers,
            client_options=client_options,
        )
        if len(servers) == 0:
            return []

        return [
            MCPTool(
                name=server.server_label,
                servers=[MCPServer.model_validate(server.model_dump(mode="json"))],
            )
            for server in servers
        ]


class ReasoningTool(OpenAIResponsesTool):
    def __init__(self):
        super().__init__(name="reasoning")

    def get_open_ai_output_handlers(self):
        return {
            "reasoning": self.handle_reasoning,
        }

    def get_open_ai_stream_callbacks(self):
        return {
            "response.reasoning_summary_text.done": self.on_reasoning_summary_text_done,
            "response.reasoning_summary_text.delta": self.on_reasoning_summary_text_delta,
            "response.reasoning_summary_part.done": self.on_reasoning_summary_part_done,
            "response.reasoning_summary_part.added": self.on_reasoning_summary_part_added,
        }

    async def on_reasoning_summary_part_added(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        part: dict,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        pass

    async def on_reasoning_summary_part_done(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        part: dict,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        pass

    async def on_reasoning_summary_text_delta(
        self,
        context: ToolContext,
        *,
        delta: str,
        output_index: int,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        pass

    async def on_reasoning_summary_text_done(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        pass

    async def on_reasoning(
        self,
        context: ToolContext,
        *,
        summary: list[str],
        content: Optional[list[str]] = None,
        encrypted_content: str | None,
        status: Literal["in_progress", "completed", "incomplete"],
    ):
        pass

    async def handle_reasoning(
        self,
        context: ToolContext,
        *,
        id: str,
        summary: list[dict],
        type: str,
        content: Optional[list[dict]],
        encrypted_content: str | None,
        status: str,
        **extra,
    ):
        await self.on_reasoning(
            context,
            summary=summary,
            content=content,
            encrypted_content=encrypted_content,
            status=status,
        )


# TODO: computer tool call


class WebSearchTool(OpenAIResponsesTool):
    def __init__(self, *, name: str = "web_search"):
        super().__init__(name=name)

    def get_open_ai_tool_definitions(self) -> list[dict]:
        return [{"type": "web_search"}]

    def get_open_ai_stream_callbacks(self):
        return {
            "response.web_search_call.in_progress": self.on_web_search_call_in_progress,
            "response.web_search_call.searching": self.on_web_search_call_searching,
            "response.web_search_call.completed": self.on_web_search_call_completed,
        }

    def get_open_ai_output_handlers(self):
        return {"web_search_call": self.handle_web_search_call}

    async def on_web_search_call_in_progress(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_web_search_call_searching(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_web_search_call_completed(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_web_search(self, context: ToolContext, *, status: str, **extra):
        pass

    async def handle_web_search_call(
        self, context: ToolContext, *, id: str, status: str, type: str, **extra
    ):
        await self.on_web_search(context, status=status)


class FileSearchResult:
    def __init__(
        self, *, attributes: dict, file_id: str, filename: str, score: float, text: str
    ):
        self.attributes = attributes
        self.file_id = file_id
        self.filename = filename
        self.score = score
        self.text = text


class FileSearchTool(OpenAIResponsesTool):
    def __init__(
        self,
        *,
        vector_store_ids: list[str],
        filters: Optional[dict] = None,
        max_num_results: Optional[int] = None,
        ranking_options: Optional[dict] = None,
    ):
        super().__init__(name="file_search")

        self.vector_store_ids = vector_store_ids
        self.filters = filters
        self.max_num_results = max_num_results
        self.ranking_options = ranking_options

    def get_open_ai_tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "file_search",
                "vector_store_ids": self.vector_store_ids,
                "filters": self.filters,
                "max_num_results": self.max_num_results,
                "ranking_options": self.ranking_options,
            }
        ]

    def get_open_ai_stream_callbacks(self):
        return {
            "response.file_search_call.in_progress": self.on_file_search_call_in_progress,
            "response.file_search_call.searching": self.on_file_search_call_searching,
            "response.file_search_call.completed": self.on_file_search_call_completed,
        }

    def get_open_ai_output_handlers(self):
        return {"handle_file_search_call": self.handle_file_search_call}

    async def on_file_search_call_in_progress(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_file_search_call_searching(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_file_search_call_completed(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_file_search(
        self,
        context: ToolContext,
        *,
        queries: list,
        results: list[FileSearchResult],
        status: Literal["in_progress", "searching", "incomplete", "failed"],
    ):
        pass

    async def handle_file_search_call(
        self,
        context: ToolContext,
        *,
        id: str,
        queries: list,
        status: str,
        results: dict | None,
        type: str,
        **extra,
    ):
        search_results = None
        if results is not None:
            search_results = []
            for result in results:
                search_results.append(FileSearchResult(**result))

        await self.on_file_search(
            context, queries=queries, results=search_results, status=status
        )


class ApplyPatchTool(OpenAIResponsesTool):
    """
    Wrapper for the built-in `apply_patch` tool.

    The model will emit `apply_patch_call` items whenever it wants to create,
    update, or delete a file using a unified diff. The server / host
    environment is expected to actually apply the patch and, if desired,
    log results via `apply_patch_call_output`.

    The two key handler entrypoints you can override are:

      * `on_apply_patch_call`       – called when the model requests a patch
      * `on_apply_patch_call_output` – called when the tool emits a log/output item
    """

    def __init__(self, *, storage: StorageToolkit, name: str = "apply_patch"):
        super().__init__(name=name)
        self._storage = storage

    # FunctionTool definition advertised to OpenAI
    def get_open_ai_tool_definitions(self) -> list[dict]:
        # No extra options for now – the built-in tool just needs the type
        return [{"type": "apply_patch"}]

    # Stream callbacks for `response.apply_patch_call.*` events
    def get_open_ai_stream_callbacks(self):
        return {
            "response.apply_patch_call.in_progress": self.on_apply_patch_call_in_progress,
            "response.apply_patch_call.completed": self.on_apply_patch_call_completed,
        }

    # Output handlers for item types
    def get_open_ai_output_handlers(self):
        return {
            # The tool call itself (what to apply)
            "apply_patch_call": self.handle_apply_patch_call,
        }

    # --- Stream callbacks -------------------------------------------------

    # response.apply_patch_call.in_progress
    async def on_apply_patch_call_in_progress(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        # Default: no-op, but you can log progress / show UI here if you want
        pass

    # response.apply_patch_call.completed
    async def on_apply_patch_call_completed(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        # Default: no-op
        pass

    # --- High-level hooks -------------------------------------------------

    async def on_apply_patch_call(
        self,
        context: ToolContext,
        *,
        call_id: str,
        operation: dict,
        status: str,
        **extra,
    ):
        """
        Called when the model requests an apply_patch operation.

        operation looks like one of:

        create_file:
            {
              "type": "create_file",
              "path": "relative/path/to/file",
              "diff": "...unified diff..."
            }

        update_file:
            {
              "type": "update_file",
              "path": "relative/path/to/file",
              "diff": "...unified diff..."
            }

        delete_file:
            {
              "type": "delete_file",
              "path": "relative/path/to/file"
            }
        """
        # Override this to actually apply the patch in your workspace.
        # Default is no-op.

        from meshagent.openai.tools.apply_patch import apply_diff

        if operation["type"] == "delete_file":
            path = operation["path"]
            logger.info(f"applying patch: deleting file {path}")
            await self._storage.delete(path=path)
            log = f"Deleted file: {path}"
            logger.info(log)
            return {"status": "completed", "output": log}

        elif operation["type"] == "create_file":
            diff = operation["diff"]
            path = operation["path"]
            logger.info(f"applying patch: creating file {path} with {diff}")
            try:
                patched = apply_diff("", diff, "create")
            except Exception as ex:
                return {"status": "failed", "output": f"{ex}"}
            await self._storage.write_text(
                path=path,
                overwrite=False,
                text=patched,
            )

            log = f"Created file: {path} ({len(patched)} bytes)"
            logger.info(log)
            return {"status": "completed", "output": log}

        elif operation["type"] == "update_file":
            path = operation["path"]
            content = await self._storage.read_file(path=path)
            text = content.data.decode()
            diff = operation["diff"]

            logger.info(f"applying patch: updating file {path} with {diff}")

            try:
                patched = apply_diff(text, diff)
            except Exception as ex:
                return {"status": "failed", "output": f"{ex}"}

            await self._storage.write_text(
                path=path,
                overwrite=True,
                text=patched,
            )

            log = f"Updated file: {path} ({len(text)} -> {len(patched)} bytes)"
            logger.info(log)
            return {"status": "completed", "output": log}

            # apply patch
        else:
            raise Exception(f"Unexpected patch operation {operation}")

    async def handle_apply_patch_call(
        self,
        context: ToolContext,
        *,
        call_id: str,
        operation: dict,
        status: str,
        type: str,
        id: str | None = None,
        **extra,
    ):
        result = await self.on_apply_patch_call(
            context,
            call_id=call_id,
            operation=operation,
            status=status,
            **extra,
        )

        return {
            "type": "apply_patch_call_output",
            "call_id": call_id,
            **result,
        }
