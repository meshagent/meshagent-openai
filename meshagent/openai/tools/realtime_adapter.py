from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import logging
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiohttp
from openai import AsyncOpenAI

from meshagent.agents.adapter import (
    DEFAULT_MAX_TOOL_CALL_LENGTH,
    DEFAULT_MAX_TOOL_CALL_LINES,
    LLMAdapter,
    LLMAudioFormat,
    LLMModelInfo,
    LLMRealtimeConnectionInfo,
    SteeringCallback,
    llm_model_pricing,
)
from meshagent.agents.agent import AgentSessionContext
from meshagent.agents.agent_event_reader import (
    AccumulatingAgentEventReader,
    AgentEventReader,
    AgentEventReaderCallbacks,
    _BufferedToolCall,
)
from meshagent.openai.tools.event_publisher import make_realtime_agent_event_publisher
from meshagent.openai.tools.responses_adapter import (
    OpenAIResponsesToolResponseAdapter,
    ResponsesToolBundle,
    safe_tool_name,
)
from meshagent.agents.messages import AgentMessage, ToolChoice
from meshagent.api import Participant, RoomException
from meshagent.api.error_codes import ErrorCode
from meshagent.api.http import (
    llm_annotation_headers,
    new_client_session,
    normalize_llm_annotations,
)
from meshagent.openai.proxy import (
    get_client,
    resolve_api_key,
    resolve_base_url,
    resolve_user_agent,
)
from meshagent.tools import ToolContext, Toolkit

logger = logging.getLogger("openai_realtime_agent")

DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL = "gpt-realtime-whisper"
DEFAULT_OPENAI_REALTIME_VOICE = "echo"
DEFAULT_OPENAI_REALTIME_INPUT_FORMAT = LLMAudioFormat(
    type="audio/pcm",
    sample_rate=24000,
)
DEFAULT_OPENAI_REALTIME_OUTPUT_FORMAT = LLMAudioFormat(
    type="audio/pcm",
    sample_rate=24000,
)
DEFAULT_OPENAI_REALTIME_TURN_DETECTION: Literal["none", "automatic"] = "none"
DEFAULT_OPENAI_REALTIME_PROTOCOLS: tuple[Literal["websocket", "webrtc"], ...] = (
    "websocket",
    "webrtc",
)
OPENAI_REALTIME_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
)


def _normalize_realtime_options(options: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(options)
    modalities = normalized.pop("modalities", None)
    if "output_modalities" not in normalized and modalities is not None:
        normalized["output_modalities"] = modalities
    raw_modalities = normalized.get("output_modalities")
    if isinstance(raw_modalities, (list, tuple)):
        selected: list[Literal["text", "audio"]] = []
        for raw_modality in raw_modalities:
            if raw_modality not in ("text", "audio"):
                continue
            modality: Literal["text", "audio"] = (
                "audio" if raw_modality == "audio" else "text"
            )
            if modality not in selected:
                selected.append(modality)
        if "text" in selected:
            normalized["output_modalities"] = ["text"]
        elif len(selected) > 0:
            normalized["output_modalities"] = [selected[0]]
        else:
            normalized.pop("output_modalities", None)
    return normalized


def _output_modalities_from_options(
    options: Mapping[str, Any],
    *,
    default: tuple[Literal["text", "audio"], ...] = ("text",),
) -> tuple[Literal["text", "audio"], ...]:
    normalized = _normalize_realtime_options(options)
    raw_modalities = normalized.get("output_modalities")
    selected: list[Literal["text", "audio"]] = []
    if isinstance(raw_modalities, (list, tuple)):
        for raw_modality in raw_modalities:
            if raw_modality not in ("text", "audio"):
                continue
            modality: Literal["text", "audio"] = (
                "audio" if raw_modality == "audio" else "text"
            )
            if modality not in selected:
                selected.append(modality)
    return tuple(selected) if len(selected) > 0 else default


def _output_modalities_from_values(
    output_modalities: Sequence[Literal["text", "audio"]],
) -> tuple[Literal["text", "audio"], ...]:
    selected: list[Literal["text", "audio"]] = []
    for output_modality in output_modalities:
        if output_modality not in ("text", "audio"):
            continue
        if output_modality not in selected:
            selected.append(output_modality)
    return tuple(selected) if len(selected) > 0 else ("text",)


def _realtime_tool_definitions(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    realtime_tools: list[dict[str, Any]] = []
    for tool in tools:
        realtime_tool = dict(tool)
        realtime_tool.pop("strict", None)
        realtime_tools.append(realtime_tool)
    return realtime_tools


def _merge_realtime_options(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _merge_realtime_options(existing, value)
        else:
            merged[key] = value
    return merged


def _ensure_realtime_input_audio_options(
    options: dict[str, Any],
    *,
    transcription_model: str | None,
    input_format: LLMAudioFormat,
    output_format: LLMAudioFormat,
    voice: str | None,
    turn_detection: Literal["none", "automatic"],
) -> None:
    audio = options.get("audio")
    if audio is None:
        audio = {}
        options["audio"] = audio
    if not isinstance(audio, dict):
        return

    input_options = audio.get("input")
    if input_options is None:
        input_options = {}
        audio["input"] = input_options
    if not isinstance(input_options, dict):
        return

    input_options.setdefault("format", _openai_realtime_audio_format(input_format))
    input_options.setdefault(
        "turn_detection",
        None if turn_detection == "none" else {"type": "server_vad"},
    )
    if transcription_model is not None:
        input_options.setdefault("transcription", {"model": transcription_model})

    output_options = audio.get("output")
    if output_options is None:
        output_options = {}
        audio["output"] = output_options
    if not isinstance(output_options, dict):
        return
    output_options.setdefault("format", _openai_realtime_audio_format(output_format))
    if voice is not None:
        output_options.setdefault("voice", voice)


def _openai_realtime_audio_format(audio_format: LLMAudioFormat) -> dict[str, Any]:
    out: dict[str, Any] = {"type": audio_format.type}
    if audio_format.sample_rate is not None:
        out["rate"] = audio_format.sample_rate
    if audio_format.bitrate is not None:
        out["bitrate"] = audio_format.bitrate
    return out


def _normalize_audio_format(
    audio_format: LLMAudioFormat | Mapping[str, Any] | None,
    *,
    default: LLMAudioFormat,
) -> LLMAudioFormat:
    if audio_format is None:
        return default
    if isinstance(audio_format, LLMAudioFormat):
        return audio_format
    type_value = audio_format.get("type", default.type)
    if not isinstance(type_value, str) or type_value.strip() == "":
        type_value = default.type
    sample_rate = audio_format.get("sample_rate", audio_format.get("rate"))
    bitrate = audio_format.get("bitrate")
    return LLMAudioFormat(
        type=type_value.strip(),
        sample_rate=sample_rate
        if isinstance(sample_rate, int)
        else default.sample_rate,
        bitrate=bitrate if isinstance(bitrate, int) else default.bitrate,
    )


def _riff_wav_data_chunk(data: bytes) -> bytes | None:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        chunk_start = offset + 8
        chunk_end = len(data) if chunk_size == 0xFFFFFFFF else chunk_start + chunk_size
        if chunk_end > len(data):
            return None
        if chunk_id == b"data":
            return data[chunk_start:chunk_end]
        offset = chunk_end + (chunk_size % 2)
    return None


def _input_audio_pcm_bytes(*, mime_type: str, data: bytes) -> bytes:
    normalized_mime_type = mime_type.partition(";")[0].strip().lower()
    if normalized_mime_type in {"audio/wav", "audio/wave", "audio/x-wav"}:
        wav_data = _riff_wav_data_chunk(data)
        if wav_data is not None:
            return wav_data
    return data


class OpenAIRealtimeSessionContext(AgentSessionContext):
    _default_websocket_ping_interval_seconds = 20.0
    _default_websocket_timeout_seconds = 60 * 60

    def __init__(
        self,
        *,
        websocket_timeout: float = _default_websocket_timeout_seconds,
        websocket_ping_interval_seconds: float = _default_websocket_ping_interval_seconds,
        session: aiohttp.ClientSession | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if websocket_timeout <= 0:
            raise ValueError("websocket_timeout must be greater than 0")
        if websocket_ping_interval_seconds <= 0:
            raise ValueError("websocket_ping_interval_seconds must be greater than 0")
        self._websocket_timeout = websocket_timeout
        self._websocket_ping_interval_seconds = websocket_ping_interval_seconds
        self._session: aiohttp.ClientSession | None = session
        self._owns_session = session is None
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._websocket_url: str | None = None
        self._websocket_headers_signature: tuple[tuple[str, str], ...] | None = None
        self._websocket_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._receive_task: asyncio.Task[None] | None = None
        self._websocket_ping_task: asyncio.Task[None] | None = None
        self._websocket_timeout_task: asyncio.Task[None] | None = None
        self._event_handler: Callable[[dict[str, Any]], None] | None = None
        self._pending_response_futures: deque[asyncio.Future[dict[str, Any]]] = deque()
        self._response_futures_by_id: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._session_update_future: asyncio.Future[dict[str, Any]] | None = None
        self._session_options_signature: str | None = None
        self._synced_message_count = 0
        self._realtime_toolkits: list[Toolkit] = []
        self._realtime_tool_caller: Participant | None = None
        self._realtime_tool_choice: ToolChoice | None = None

    @staticmethod
    def _headers_signature(headers: dict[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((key.lower(), value) for key, value in headers.items()))

    @property
    def is_connected(self) -> bool:
        return self._websocket is not None and not self._websocket.closed

    @property
    def supports_realtime_audio(self) -> bool:
        return True

    async def append_realtime_audio_chunk(
        self,
        *,
        mime_type: str,
        data: bytes,
        sample_rate: int | None = None,
        bitrate: int | None = None,
    ) -> None:
        del sample_rate
        del bitrate
        audio = _input_audio_pcm_bytes(mime_type=mime_type, data=data)
        audio_b64 = base64.b64encode(audio).decode()
        await self.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
        )

    async def commit_realtime_audio(self) -> None:
        await self.send_json({"type": "input_audio_buffer.commit"})

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
            logger.warning("realtime websocket ping failed, closing socket: %s", error)
            await self.close_websocket()

    async def _run_websocket_timeout(self) -> None:
        try:
            await asyncio.sleep(self._websocket_timeout)
        except asyncio.CancelledError:
            raise
        logger.info(
            "realtime websocket session timed out after %.1f seconds",
            self._websocket_timeout,
        )
        await self.close()

    async def _close_websocket_locked(self) -> None:
        current_task = asyncio.current_task()
        tasks: list[asyncio.Task[Any]] = []
        for task in (
            self._receive_task,
            self._websocket_ping_task,
            self._websocket_timeout_task,
        ):
            if task is not None and task is not current_task:
                tasks.append(task)
        self._receive_task = None
        self._websocket_ping_task = None
        self._websocket_timeout_task = None

        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        websocket = self._websocket
        self._websocket = None
        self._websocket_url = None
        self._websocket_headers_signature = None
        self._session_options_signature = None
        self._synced_message_count = 0
        pending_response_futures = list(self._pending_response_futures)
        response_futures = list(self._response_futures_by_id.values())
        self._pending_response_futures.clear()
        self._response_futures_by_id.clear()
        for future in [*pending_response_futures, *response_futures]:
            if not future.done():
                future.set_exception(
                    RoomException(
                        "OpenAI Realtime websocket closed before the response completed",
                        status_code=503,
                        code=ErrorCode.OPERATION_FAILED,
                    )
                )
        session_update_future = self._session_update_future
        self._session_update_future = None
        if session_update_future is not None and not session_update_future.done():
            session_update_future.set_exception(
                RoomException(
                    "OpenAI Realtime websocket closed before the session update completed",
                    status_code=503,
                    code=ErrorCode.OPERATION_FAILED,
                )
            )
        if websocket is not None and not websocket.closed:
            await websocket.close()

    def _response_created(self, event: dict[str, Any]) -> None:
        response_id = OpenAIRealtimeAdapter._response_id(event)
        if response_id is None or len(self._pending_response_futures) == 0:
            return
        future = self._pending_response_futures.popleft()
        self._response_futures_by_id[response_id] = future

    def _complete_response_future(
        self, *, event: dict[str, Any], error: RoomException | None
    ) -> None:
        response_id = OpenAIRealtimeAdapter._response_id(event)
        future: asyncio.Future[dict[str, Any]] | None = None
        if response_id is not None:
            future = self._response_futures_by_id.pop(response_id, None)
        if future is None and len(self._pending_response_futures) > 0:
            future = self._pending_response_futures.popleft()
        if future is None or future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(event)

    def _fail_response_futures(self, error: BaseException) -> None:
        if isinstance(error, RoomException):
            response_error = error
        else:
            response_error = RoomException(
                str(error),
                status_code=503,
                code=ErrorCode.OPERATION_FAILED,
            )
        pending_response_futures = list(self._pending_response_futures)
        response_futures = list(self._response_futures_by_id.values())
        self._pending_response_futures.clear()
        self._response_futures_by_id.clear()
        for future in [*pending_response_futures, *response_futures]:
            if not future.done():
                future.set_exception(response_error)

    async def close_websocket(self) -> None:
        async with self._websocket_lock:
            await self._close_websocket_locked()

    async def close(self) -> None:
        await self.close_websocket()
        if self._owns_session:
            session = self._session
            self._session = None
            if session is not None and not session.closed:
                await asyncio.shield(session.close())

    async def start(self) -> None:
        await super().start()
        if self._session is None and self._owns_session:
            self._session = new_client_session(
                timeout=aiohttp.ClientTimeout(total=None)
            )

    def copy(self) -> "OpenAIRealtimeSessionContext":
        shared_session = self._session if not self._owns_session else None
        return self.__class__(
            messages=copy.deepcopy(self.messages),
            system_role=self.system_role,
            websocket_timeout=self._websocket_timeout,
            websocket_ping_interval_seconds=self._websocket_ping_interval_seconds,
            session=shared_session,
        )

    async def connect(
        self,
        *,
        url: str,
        headers: dict[str, str],
        event_handler: Callable[[dict[str, Any]], None],
        receive_loop: Callable[["OpenAIRealtimeSessionContext"], Any],
    ) -> aiohttp.ClientWebSocketResponse:
        headers_signature = self._headers_signature(headers)
        async with self._websocket_lock:
            if (
                self._websocket is not None
                and not self._websocket.closed
                and self._websocket_url == url
                and self._websocket_headers_signature == headers_signature
            ):
                self._event_handler = event_handler
                return self._websocket

            await self._close_websocket_locked()

            session = self._session
            created_session = False
            if session is None:
                session = new_client_session(timeout=aiohttp.ClientTimeout(total=None))
                created_session = True
                if self._owns_session:
                    self._session = session

            try:
                websocket = await session.ws_connect(
                    url,
                    headers=headers,
                    heartbeat=None,
                    autoping=True,
                )
            except aiohttp.WSServerHandshakeError as error:
                if created_session:
                    if self._session is session:
                        self._session = None
                    await session.close()
                raise RoomException(
                    f"OpenAI Realtime websocket request failed with status {error.status}.",
                    status_code=error.status,
                ) from error
            except Exception:
                if created_session:
                    if self._session is session:
                        self._session = None
                    await session.close()
                raise

            self._websocket = websocket
            self._websocket_url = url
            self._websocket_headers_signature = headers_signature
            self._event_handler = event_handler
            self._receive_task = asyncio.create_task(receive_loop(self))
            self._websocket_ping_task = asyncio.create_task(self._run_websocket_ping())
            self._websocket_timeout_task = asyncio.create_task(
                self._run_websocket_timeout()
            )
            return websocket

    async def send_json(self, payload: dict[str, Any]) -> None:
        websocket = self._websocket
        if websocket is None or websocket.closed:
            raise RoomException(
                "OpenAI Realtime session is not connected. Call connect() before sending events."
            )
        async with self._send_lock:
            await websocket.send_str(json.dumps(payload))


class _OpenAIRealtimeAgentEventReader(AccumulatingAgentEventReader):
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
                    parts.append(
                        {"type": "input_text", "text": f"attached file: {url}"}
                    )
        if not parts:
            parts.append({"type": "input_text", "text": json.dumps(content)})
        self._emit_context_message({"role": "user", "content": parts})

    def _append_assistant_text(self, *, text: str, phase: str | None) -> None:
        del phase
        self._emit_context_message(
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )

    def _append_assistant_reasoning(self, *, text: str) -> None:
        self._append_assistant_text(text=f"Reasoning: {text}", phase=None)

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
        self._emit_context_message(
            {
                "type": "function_call",
                "id": tool_call.item_id,
                "call_id": tool_call.call_id or tool_call.item_id,
                "name": self._function_name(tool_call=tool_call),
                "arguments": tool_call.arguments_json(),
                "status": "completed"
                if result is not None or error is not None or tool_call.logs
                else "in_progress",
            }
        )
        if result is not None or error is not None or tool_call.logs:
            self._emit_context_message(
                {
                    "type": "function_call_output",
                    "id": f"{tool_call.item_id}:output",
                    "call_id": tool_call.call_id or tool_call.item_id,
                    "output": self._result_text(
                        result=result,
                        error=error,
                        logs=tool_call.logs,
                    ),
                    "status": "failed" if error is not None else "completed",
                }
            )

    def _append_image_generation_event(
        self,
        *,
        event_type: str,
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
            "item_id": item_id,
            "call_id": call_id,
            "toolkit": toolkit,
            "tool": tool,
            "arguments": arguments,
            "images": images,
            "status": status,
        }
        self._append_assistant_text(text=json.dumps(item), phase=None)

    def _append_audio_generation_event(self, *, message: Any) -> None:
        return

    def _append_audio_transcription_event(self, *, message: Any) -> None:
        self._append_assistant_text(
            text=json.dumps(message.model_dump(mode="json")),
            phase=None,
        )

    def _restore_compacted_messages(self, *, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            self._emit_context_message(message)

    @staticmethod
    def _function_name(*, tool_call: _BufferedToolCall) -> str:
        if tool_call.namespace == "openai.responses" and tool_call.toolkit == "openai":
            return f"{tool_call.tool}_call"
        if (
            tool_call.namespace == "anthropic.messages"
            and tool_call.toolkit == "anthropic"
        ):
            return f"{tool_call.tool}_tool_use"
        if tool_call.namespace in {"openai.responses", "anthropic.messages"}:
            return f"{tool_call.toolkit}_{tool_call.tool}".strip("_")
        if tool_call.toolkit in {"", "function", "tool"}:
            return tool_call.tool
        return f"{tool_call.toolkit}_{tool_call.tool}"

    def _append_assistant_structured_item(self, item: dict[str, Any]) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": json.dumps(item)}],
            }
        )


class OpenAIRealtimeAdapter(LLMAdapter[dict[str, Any]]):
    _known_models = (
        "gpt-realtime",
        "gpt-realtime-2",
        "gpt-realtime-1.5",
        "gpt-realtime-mini",
    )

    def __init__(
        self,
        model: str = "gpt-realtime",
        *,
        client: AsyncOpenAI | None = None,
        session_options: Mapping[str, Any] | None = None,
        response_options: Mapping[str, Any] | None = None,
        provider: str = "openai-realtime",
        log_requests: bool = False,
        websocket_timeout: float = OpenAIRealtimeSessionContext._default_websocket_timeout_seconds,
        base_url: str | None = None,
        api_key: str | None = None,
        user_agent: str | None = None,
        annotations: Mapping[str, object] | None = None,
        friendly_name: str | None = None,
        description: str | None = None,
        allowed_models: list[str] | None = None,
        transcription_model: str | None = DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL,
        voice: str | None = DEFAULT_OPENAI_REALTIME_VOICE,
        input_format: LLMAudioFormat | Mapping[str, Any] | None = None,
        output_format: LLMAudioFormat | Mapping[str, Any] | None = None,
        turn_detection: Literal[
            "none", "automatic"
        ] = DEFAULT_OPENAI_REALTIME_TURN_DETECTION,
        supported_output_modalities: Sequence[Literal["text", "audio"]] | None = None,
        realtime_protocols: tuple[Literal["websocket", "webrtc"], ...]
        | list[Literal["websocket", "webrtc"]]
        | None = DEFAULT_OPENAI_REALTIME_PROTOCOLS,
    ):
        if websocket_timeout <= 0:
            raise ValueError("websocket_timeout must be greater than 0")
        self._model = model
        self._client = client
        self._session_options = dict(session_options or {})
        self._response_options = dict(response_options or {})
        self._provider = provider
        self._log_requests = log_requests
        self._websocket_timeout = websocket_timeout
        self._base_url = resolve_base_url(base_url)
        self._has_explicit_api_key = isinstance(api_key, str) and api_key.strip() != ""
        self._api_key = resolve_api_key(api_key)
        self._user_agent = resolve_user_agent(user_agent)
        self._annotations = normalize_llm_annotations(annotations)
        self._friendly_name = friendly_name
        self._description = description
        self._allowed_models = (
            list(allowed_models) if allowed_models is not None else None
        )
        if isinstance(transcription_model, str):
            transcription_model = transcription_model.strip() or None
        self._transcription_model = transcription_model
        if isinstance(voice, str):
            voice = voice.strip() or None
        self._voice = voice
        self._input_format = _normalize_audio_format(
            input_format,
            default=DEFAULT_OPENAI_REALTIME_INPUT_FORMAT,
        )
        self._output_format = _normalize_audio_format(
            output_format,
            default=DEFAULT_OPENAI_REALTIME_OUTPUT_FORMAT,
        )
        self._turn_detection = turn_detection
        self._supported_output_modalities = (
            _output_modalities_from_values(supported_output_modalities)
            if supported_output_modalities is not None
            else _output_modalities_from_options(
                self._response_options,
                default=("text", "audio"),
            )
        )
        self._realtime_protocols = tuple(
            dict.fromkeys(realtime_protocols or DEFAULT_OPENAI_REALTIME_PROTOCOLS)
        )

    def default_model(self) -> str:
        return self._model

    def provider_name(self) -> str | None:
        return self._provider

    def provider_friendly_name(self) -> str:
        return self._friendly_name or "OpenAI Realtime"

    def provider_description(self) -> str | None:
        return self._description or "OpenAI Realtime API"

    def list_models(self) -> list[LLMModelInfo]:
        names = list(self._allowed_models or self._known_models)
        if self._allowed_models is None and self._model not in names:
            names.insert(0, self._model)
        output_modalities = self._supported_output_modalities
        return [
            LLMModelInfo(
                name=name,
                context_window=32000,
                pricing=llm_model_pricing(provider="openai", model=name),
                modalities=output_modalities,
                available_voices=OPENAI_REALTIME_VOICES,
                default_output_voice=self._voice,
                input_format=self._input_format,
                output_format=self._output_format,
                turn_detection=self._turn_detection,
                realtime_protocols=self._realtime_protocols,
            )
            for name in names
        ]

    def create_session(self) -> OpenAIRealtimeSessionContext:
        return OpenAIRealtimeSessionContext(
            system_role=None,
            websocket_timeout=self._websocket_timeout,
        )

    def make_agent_event_reader(
        self,
        *,
        emit_message: Callable[[dict[str, Any]], None],
        callbacks: AgentEventReaderCallbacks | None = None,
    ) -> AgentEventReader:
        return _OpenAIRealtimeAgentEventReader(
            emit_message=emit_message,
            callbacks=callbacks,
        )

    def restore_context_messages(
        self,
        *,
        context: AgentSessionContext,
        messages: list[dict[str, Any]],
    ) -> None:
        if not isinstance(context, OpenAIRealtimeSessionContext):
            raise RoomException(
                "OpenAIRealtimeAdapter requires OpenAIRealtimeSessionContext from create_session()"
            )
        context.previous_messages.clear()
        context.previous_messages.extend(copy.deepcopy(messages))
        context.messages.clear()
        context.previous_response_id = None
        context._synced_message_count = 0

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback: Callable[[AgentMessage], None],
        custom_event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Callable[[dict[str, Any]], None]:
        return make_realtime_agent_event_publisher(
            turn_id=turn_id,
            thread_id=thread_id,
            callback=callback,
            custom_event_callback=custom_event_callback,
        )

    def with_runtime_api_key(self, *, api_key: str | None) -> "OpenAIRealtimeAdapter":
        resolved_api_key = resolve_api_key(api_key)
        if (
            self._client is not None
            or self._has_explicit_api_key
            or resolved_api_key is None
        ):
            return self
        return type(self)(
            model=self._model,
            session_options=self._session_options,
            response_options=self._response_options,
            provider=self._provider,
            log_requests=self._log_requests,
            websocket_timeout=self._websocket_timeout,
            base_url=self._base_url,
            api_key=resolved_api_key,
            user_agent=self._user_agent,
            annotations=self._annotations,
            friendly_name=self._friendly_name,
            description=self._description,
            allowed_models=self._allowed_models,
            transcription_model=self._transcription_model,
            voice=self._voice,
            input_format=self._input_format,
            output_format=self._output_format,
            turn_detection=self._turn_detection,
            supported_output_modalities=self._supported_output_modalities,
            realtime_protocols=self._realtime_protocols,
        )

    def _openai_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        return get_client(
            base_url=self._base_url,
            api_key=self._api_key,
            user_agent=self._user_agent,
        )

    @staticmethod
    def _http_base_url_to_ws_realtime_url(*, base_url: str, model: str) -> str:
        parsed = urlparse(base_url)
        if parsed.scheme == "https":
            ws_scheme = "wss"
        elif parsed.scheme == "http":
            ws_scheme = "ws"
        elif parsed.scheme in ("ws", "wss"):
            ws_scheme = parsed.scheme
        else:
            raise RoomException(
                f"unsupported OpenAI base URL scheme for realtime mode: {parsed.scheme}"
            )

        query_items = parse_qsl(parsed.query, keep_blank_values=True)
        query_items.append(("model", model))
        path = parsed.path.rstrip("/") + "/realtime"
        return urlunparse(
            (
                ws_scheme,
                parsed.netloc,
                path,
                parsed.params,
                urlencode(query_items),
                parsed.fragment,
            )
        )

    @staticmethod
    def _http_base_url_to_webrtc_realtime_url(*, base_url: str, model: str) -> str:
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise RoomException(
                f"unsupported OpenAI base URL scheme for realtime WebRTC mode: {parsed.scheme}"
            )
        query_items = parse_qsl(parsed.query, keep_blank_values=True)
        query_items.append(("model", model))
        path = parsed.path.rstrip("/") + "/realtime/calls"
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                path,
                parsed.params,
                urlencode(query_items),
                parsed.fragment,
            )
        )

    async def create_realtime_connection(
        self,
        *,
        protocol: Literal["websocket", "webrtc"],
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMRealtimeConnectionInfo:
        if protocol not in self._realtime_protocols:
            raise RoomException(
                f"OpenAI Realtime adapter does not support realtime protocol {protocol!r}"
            )
        del options
        realtime_model = model or self.default_model()
        openai = self._openai_client()
        extra_headers = llm_annotation_headers(self._annotations)
        headers = self._websocket_headers(openai=openai, extra_headers=extra_headers)
        if protocol == "websocket":
            return LLMRealtimeConnectionInfo(
                protocol=protocol,
                url=self._http_base_url_to_ws_realtime_url(
                    base_url=str(openai.base_url),
                    model=realtime_model,
                ),
                headers=headers,
                web_only_protocol=None,
            )
        headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in {"content-type", "content-length"}
        }
        return LLMRealtimeConnectionInfo(
            protocol=protocol,
            url=self._http_base_url_to_webrtc_realtime_url(
                base_url=str(openai.base_url),
                model=realtime_model,
            ),
            headers=headers,
            web_only_protocol=None,
        )

    def _websocket_headers(
        self, *, openai: AsyncOpenAI, extra_headers: dict[str, str]
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in openai.default_headers.items():
            if isinstance(value, str):
                headers[key] = value
        headers.update(extra_headers)
        excluded_headers = {"openai-beta", "content-type", "content-length"}
        return {
            key: value
            for key, value in headers.items()
            if key.lower() not in excluded_headers
        }

    @staticmethod
    def _terminal_response_event(event_type: str) -> bool:
        return event_type in {
            "response.done",
            "response.completed",
            "response.failed",
            "response.incomplete",
        }

    @staticmethod
    def _response_error(event: dict[str, Any]) -> RoomException | None:
        if event.get("type") != "response.failed":
            return None
        response = event.get("response")
        error: Any = None
        if isinstance(response, dict):
            error = response.get("error")
        if error is None:
            error = event.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code")
            if isinstance(message, str) and message.strip() != "":
                return RoomException(
                    message.strip(),
                    status_code=500,
                    code=code if isinstance(code, str) else ErrorCode.OPERATION_FAILED,
                )
        return RoomException(
            "OpenAI Realtime response failed",
            status_code=500,
            code=ErrorCode.OPERATION_FAILED,
        )

    async def _receive_payload(
        self,
        *,
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> dict[str, Any]:
        while True:
            message = await websocket.receive()
            if message.type == aiohttp.WSMsgType.TEXT:
                try:
                    payload = json.loads(message.data)
                except json.JSONDecodeError as error:
                    raise RoomException(
                        f"OpenAI Realtime websocket returned invalid JSON: {error}"
                    ) from error
                if not isinstance(payload, dict):
                    continue
                if self._log_requests:
                    logger.info("<== realtime event=%s", payload.get("type"))
                if payload.get("type") == "error":
                    error = payload.get("error")
                    if isinstance(error, dict):
                        message_text = error.get("message")
                        if isinstance(message_text, str):
                            raise RoomException(
                                f"Error from OpenAI Realtime websocket: {message_text}"
                            )
                    raise RoomException(
                        f"Error from OpenAI Realtime websocket: {json.dumps(payload)}"
                    )
                return payload

            if message.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            }:
                raise RoomException(
                    "OpenAI Realtime websocket closed unexpectedly",
                    status_code=503,
                    code=ErrorCode.OPERATION_FAILED,
                )

            if message.type == aiohttp.WSMsgType.BINARY:
                raise RoomException(
                    "OpenAI Realtime websocket returned unexpected binary message"
                )

    async def _handle_realtime_tool_calls(
        self,
        *,
        context: OpenAIRealtimeSessionContext,
        event: Mapping[str, Any],
    ) -> bool:
        tool_calls = self._response_function_calls(event)
        if len(tool_calls) == 0:
            return False

        caller = context._realtime_tool_caller
        if caller is None or len(context._realtime_toolkits) == 0:
            return False

        event_handler = context._event_handler
        tool_bundle = ResponsesToolBundle(toolkits=[*context._realtime_toolkits])
        tool_adapter = OpenAIResponsesToolResponseAdapter(
            max_tool_call_length=DEFAULT_MAX_TOOL_CALL_LENGTH,
            max_tool_call_lines=DEFAULT_MAX_TOOL_CALL_LINES,
        )
        for tool_call in tool_calls:
            tool_context = ToolContext(
                caller=caller,
                caller_context=context.to_tool_caller_context(
                    item_id=tool_call.get("id")
                ),
                event_handler=event_handler,
            )
            if event_handler is not None:
                event_handler(
                    {
                        "type": "meshagent.handler.added",
                        "item": dict(tool_call),
                    }
                )
            realtime_tool_call = SimpleNamespace(
                id=tool_call.get("id"),
                call_id=tool_call.get("call_id"),
                name=tool_call["name"],
                arguments=tool_call["arguments"],
            )
            tool_result = await tool_bundle.execute(
                context=tool_context,
                tool_call=realtime_tool_call,
            )
            tool_output_messages = await tool_adapter.create_messages(
                context=context,
                tool_call=realtime_tool_call,
                response=tool_result,
            )
            context.previous_messages.append(dict(tool_call))
            for output_message in tool_output_messages:
                await self._send_conversation_item(
                    context=context, message=output_message
                )
                context.previous_messages.append(output_message)
                if event_handler is not None:
                    event_handler(
                        {
                            "type": "meshagent.handler.done",
                            "item": dict(output_message),
                        }
                    )
            context._synced_message_count = len(
                [*context.previous_messages, *context.messages]
            )

        await self._send_response_create(
            context=context,
            toolkits=context._realtime_toolkits,
            tool_choice=context._realtime_tool_choice,
        )
        return True

    async def _receive_loop(self, context: OpenAIRealtimeSessionContext) -> None:
        try:
            while True:
                websocket = context._websocket
                if websocket is None or websocket.closed:
                    return
                event = await self._receive_payload(websocket=websocket)
                event_type = event.get("type")
                if not isinstance(event_type, str):
                    event_handler = context._event_handler
                    if event_handler is not None:
                        event_handler(event)
                    continue

                if event_type == "session.updated":
                    session_update_future = context._session_update_future
                    if (
                        session_update_future is not None
                        and not session_update_future.done()
                    ):
                        session_update_future.set_result(event)

                if self._terminal_response_event(event_type):
                    error = self._response_error(event)
                    if error is None and await self._handle_realtime_tool_calls(
                        context=context,
                        event=event,
                    ):
                        continue

                event_handler = context._event_handler
                if event_handler is not None:
                    event_handler(event)

                if event_type == "response.created":
                    context._response_created(event)
                    continue
                if self._terminal_response_event(event_type):
                    error = self._response_error(event)
                    context._complete_response_future(event=event, error=error)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            context._fail_response_futures(error)
            session_update_future = context._session_update_future
            if session_update_future is not None and not session_update_future.done():
                if isinstance(error, RoomException):
                    session_update_future.set_exception(error)
                else:
                    session_update_future.set_exception(
                        RoomException(
                            str(error),
                            status_code=503,
                            code=ErrorCode.OPERATION_FAILED,
                        )
                    )
            await context.close_websocket()

    async def connect(
        self,
        *,
        context: OpenAIRealtimeSessionContext,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
        model: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        realtime_model = model or self.default_model()
        openai = self._openai_client()
        extra_headers = llm_annotation_headers(self._annotations)
        headers = self._websocket_headers(openai=openai, extra_headers=extra_headers)
        url = self._http_base_url_to_ws_realtime_url(
            base_url=str(openai.base_url),
            model=realtime_model,
        )

        session_payload: dict[str, Any] = {"type": "session.update"}
        session_options = _normalize_realtime_options(
            copy.deepcopy(self._session_options)
        )
        if options is not None:
            session_options = _merge_realtime_options(session_options, dict(options))
            session_options = _normalize_realtime_options(session_options)
        _ensure_realtime_input_audio_options(
            session_options,
            transcription_model=self._transcription_model,
            input_format=self._input_format,
            output_format=self._output_format,
            voice=self._voice,
            turn_detection=self._turn_detection,
        )
        session_payload["session"] = {"type": "realtime", **session_options}
        session_options_signature = json.dumps(
            session_payload["session"],
            sort_keys=True,
            separators=(",", ":"),
        )
        if (
            context.is_connected
            and context._session_options_signature is not None
            and context._session_options_signature != session_options_signature
        ):
            await context.close_websocket()

        was_connected = context.is_connected
        await context.connect(
            url=url,
            headers=headers,
            event_handler=event_handler or (lambda event: None),
            receive_loop=self._receive_loop,
        )
        if (
            was_connected
            and context._session_options_signature == session_options_signature
        ):
            return

        context._session_update_future = asyncio.get_running_loop().create_future()
        await context.send_json(session_payload)
        await asyncio.wait_for(
            context._session_update_future,
            timeout=min(context._websocket_timeout, 30.0),
        )
        context._session_options_signature = session_options_signature

    async def disconnect(self, *, context: OpenAIRealtimeSessionContext) -> None:
        await context.close_websocket()

    async def start_session(
        self,
        *,
        context: AgentSessionContext,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        await self.start_realtime_session(
            context=context,
            event_handler=event_handler,
        )

    async def start_realtime_session(
        self,
        *,
        context: AgentSessionContext,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
        caller: Participant | None = None,
        toolkits: list[Toolkit] | None = None,
        tool_choice: ToolChoice | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(context, OpenAIRealtimeSessionContext):
            raise RoomException(
                "OpenAIRealtimeAdapter requires OpenAIRealtimeSessionContext from create_session()"
            )

        session_options = _normalize_realtime_options(copy.deepcopy(options or {}))
        instructions = context.get_system_instructions()
        if isinstance(instructions, str) and instructions.strip() != "":
            session_options["instructions"] = instructions
        voice = context.metadata.get("voice")
        if isinstance(voice, str) and voice.strip() != "":
            session_options = _merge_realtime_options(
                session_options,
                {"audio": {"output": {"voice": voice.strip()}}},
            )

        if toolkits is not None:
            tool_bundle = ResponsesToolBundle(toolkits=[*toolkits])
            openai_tools = _realtime_tool_definitions(tool_bundle.to_json())
            if openai_tools is not None:
                session_options["tools"] = openai_tools
        if tool_choice is not None:
            session_options["tool_choice"] = {
                "type": "function",
                "name": safe_tool_name(tool_choice.tool_name),
            }
        context._realtime_toolkits = [*toolkits] if toolkits is not None else []
        context._realtime_tool_caller = caller
        context._realtime_tool_choice = tool_choice

        await self.connect(
            context=context,
            event_handler=event_handler,
            model=model,
            options=session_options or None,
        )

    async def stop_session(self, *, context: AgentSessionContext) -> None:
        if not isinstance(context, OpenAIRealtimeSessionContext):
            raise RoomException(
                "OpenAIRealtimeAdapter requires OpenAIRealtimeSessionContext from create_session()"
            )
        await self.disconnect(context=context)

    async def send_event(
        self, *, context: OpenAIRealtimeSessionContext, event: Mapping[str, Any]
    ) -> None:
        await context.send_json(dict(event))

    async def append_input_audio(
        self,
        *,
        context: OpenAIRealtimeSessionContext,
        audio: bytes | str,
    ) -> None:
        audio_b64 = (
            base64.b64encode(audio).decode() if isinstance(audio, bytes) else audio
        )
        await context.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
        )

    async def commit_input_audio(
        self, *, context: OpenAIRealtimeSessionContext
    ) -> None:
        await context.send_json({"type": "input_audio_buffer.commit"})

    async def clear_input_audio(self, *, context: OpenAIRealtimeSessionContext) -> None:
        await context.send_json({"type": "input_audio_buffer.clear"})

    async def cancel_response(
        self,
        *,
        context: OpenAIRealtimeSessionContext,
        response_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"type": "response.cancel"}
        if response_id is not None:
            payload["response_id"] = response_id
        await context.send_json(payload)

    @staticmethod
    def _content_to_text(content: object) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            nested_text = item.get("input_text")
            if isinstance(nested_text, str):
                parts.append(nested_text)
        return "\n".join(part for part in parts if part != "")

    @classmethod
    def _conversation_item_for_message(
        cls, message: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        message_type = message.get("type")
        if message_type in {"function_call", "function_call_output"}:
            item = dict(message)
            item.setdefault("id", item.get("call_id"))
            if item.get("status") not in {"completed", "incomplete"}:
                item["status"] = "completed"
            return {
                "type": "conversation.item.create",
                "item": item,
            }

        role = message.get("role")
        if role not in {"user", "assistant"}:
            return None

        text = cls._content_to_text(message.get("content", ""))
        if text.strip() == "":
            return None

        content_type = "input_text" if role == "user" else "output_text"
        return {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": role,
                "content": [{"type": content_type, "text": text}],
            },
        }

    @classmethod
    def _response_text(cls, event: Mapping[str, Any]) -> str:
        response = event.get("response")
        if not isinstance(response, dict):
            return ""
        output = response.get("output")
        if not isinstance(output, list):
            return ""

        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            if item.get("role") != "assistant":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text") or content_item.get("transcript")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _response_function_calls(event: Mapping[str, Any]) -> list[dict[str, Any]]:
        response = event.get("response")
        if not isinstance(response, dict):
            return []
        output = response.get("output")
        if not isinstance(output, list):
            return []

        tool_calls: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            name = item.get("name")
            if not isinstance(name, str) or name.strip() == "":
                continue
            call_id = item.get("call_id")
            item_id = item.get("id")
            arguments = item.get("arguments")
            tool_calls.append(
                {
                    "type": "function_call",
                    "id": item_id if isinstance(item_id, str) else call_id,
                    "call_id": call_id if isinstance(call_id, str) else item_id,
                    "name": name,
                    "arguments": arguments if isinstance(arguments, str) else "{}",
                    "status": "completed",
                }
            )
        return tool_calls

    async def _sync_text_messages(
        self, *, context: OpenAIRealtimeSessionContext
    ) -> None:
        messages: list[dict[str, Any]] = [
            *context.previous_messages,
            *context.messages,
        ]
        if context._synced_message_count > len(messages):
            context._synced_message_count = 0

        for message in messages[context._synced_message_count :]:
            payload = self._conversation_item_for_message(message)
            if payload is None:
                context._synced_message_count += 1
                continue
            await context.send_json(payload)
            context._synced_message_count += 1

    @staticmethod
    async def _send_conversation_item(
        *, context: OpenAIRealtimeSessionContext, message: Mapping[str, Any]
    ) -> None:
        payload = OpenAIRealtimeAdapter._conversation_item_for_message(message)
        if payload is not None:
            await context.send_json(payload)

    async def _send_response_create(
        self,
        *,
        context: OpenAIRealtimeSessionContext,
        toolkits: list[Toolkit],
        tool_choice: ToolChoice | None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"type": "response.create"}
        response_options = _normalize_realtime_options(
            copy.deepcopy(self._response_options)
        )
        if options is not None:
            response_options.update(options)
            response_options = _normalize_realtime_options(response_options)
        tool_bundle = ResponsesToolBundle(toolkits=[*toolkits])
        openai_tools = _realtime_tool_definitions(tool_bundle.to_json())
        if openai_tools is not None:
            response_options["tools"] = openai_tools
        if tool_choice is not None:
            response_options["tool_choice"] = {
                "type": "function",
                "name": safe_tool_name(tool_choice.tool_name),
            }
        if len(response_options) > 0:
            payload["response"] = response_options
        await context.send_json(payload)

    @staticmethod
    def _response_id(event: Mapping[str, Any]) -> str | None:
        response = event.get("response")
        if not isinstance(response, dict):
            return None
        response_id = response.get("id")
        if isinstance(response_id, str) and response_id.strip() != "":
            return response_id
        return None

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller: Participant,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
        steering_callback: SteeringCallback | None = None,
        model: str | None = None,
        on_behalf_of: Participant | None = None,
        tool_choice: ToolChoice | None = None,
        options: dict | None = None,
    ) -> dict[str, Any]:
        del output_schema
        del steering_callback
        del model
        del on_behalf_of
        if not isinstance(context, OpenAIRealtimeSessionContext):
            raise RoomException(
                "OpenAIRealtimeAdapter requires OpenAIRealtimeSessionContext from create_session()"
            )
        if not context.is_connected:
            raise RoomException(
                "OpenAI Realtime session is not connected. Call connect() before create_response()."
            )

        if event_handler is not None:
            context._event_handler = event_handler

        tool_adapter = OpenAIResponsesToolResponseAdapter(
            max_tool_call_length=DEFAULT_MAX_TOOL_CALL_LENGTH,
            max_tool_call_lines=DEFAULT_MAX_TOOL_CALL_LINES,
        )
        terminal_event: dict[str, Any] | None = None
        while True:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            async with context._request_lock:
                await self._sync_text_messages(context=context)
                context._pending_response_futures.append(future)
                try:
                    await self._send_response_create(
                        context=context,
                        toolkits=toolkits,
                        tool_choice=tool_choice,
                        options=options,
                    )
                except BaseException:
                    with contextlib.suppress(ValueError):
                        context._pending_response_futures.remove(future)
                    raise

            terminal_event = await future
            response_id = self._response_id(terminal_event)
            if response_id is not None:
                context.track_response(response_id)
            else:
                context.previous_messages.extend(context.messages)
                context.messages.clear()
            assistant_text = self._response_text(terminal_event)
            if assistant_text != "":
                context.previous_messages.append(
                    {"role": "assistant", "content": assistant_text}
                )

            tool_calls = self._response_function_calls(terminal_event)
            if not tool_calls:
                context._synced_message_count = len(
                    [*context.previous_messages, *context.messages]
                )
                break

            tool_bundle = ResponsesToolBundle(toolkits=[*toolkits])
            for tool_call in tool_calls:
                tool_context = ToolContext(
                    caller=caller,
                    caller_context=context.to_tool_caller_context(
                        item_id=tool_call.get("id")
                    ),
                    event_handler=event_handler,
                )
                if event_handler is not None:
                    event_handler(
                        {
                            "type": "meshagent.handler.added",
                            "item": dict(tool_call),
                        }
                    )
                realtime_tool_call = SimpleNamespace(
                    id=tool_call.get("id"),
                    call_id=tool_call.get("call_id"),
                    name=tool_call["name"],
                    arguments=tool_call["arguments"],
                )
                tool_result = await tool_bundle.execute(
                    context=tool_context,
                    tool_call=realtime_tool_call,
                )
                tool_output_messages = await tool_adapter.create_messages(
                    context=context,
                    tool_call=realtime_tool_call,
                    response=tool_result,
                )
                context.previous_messages.append(tool_call)
                for output_message in tool_output_messages:
                    await self._send_conversation_item(
                        context=context, message=output_message
                    )
                    context.previous_messages.append(output_message)
                    if event_handler is not None:
                        event_handler(
                            {
                                "type": "meshagent.handler.done",
                                "item": dict(output_message),
                            }
                        )
                context._synced_message_count = len(
                    [*context.previous_messages, *context.messages]
                )

        if terminal_event is None:
            raise RoomException(
                "OpenAI Realtime response ended without a terminal event."
            )
        context.turn_count += 1
        return terminal_event


__all__ = [
    "OpenAIRealtimeAdapter",
    "OpenAIRealtimeSessionContext",
]
