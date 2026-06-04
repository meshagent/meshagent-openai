import asyncio
import json
import os
from typing import Any

import aiohttp
import pytest
from openai import AsyncOpenAI

from meshagent.agents.messages import (
    AGENT_EVENT_AUDIO_GENERATION_COMPLETED,
    AGENT_EVENT_AUDIO_GENERATION_DELTA,
    AGENT_EVENT_AUDIO_GENERATION_STARTED,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AgentAudioGenerationCompleted,
    AgentAudioGenerationDelta,
    AgentAudioGenerationStarted,
    AgentAudioInputSpeechEnded,
    AgentAudioInputSpeechStarted,
    AgentAudioTranscriptionCompleted,
    AgentAudioTranscriptionDelta,
    AgentAudioTranscriptionStarted,
    AgentMessage,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallArgumentsDelta,
    AgentToolCallEnded,
    AgentToolCallStarted,
)
from meshagent.api import RoomException
from meshagent.api.messaging import TextContent
from meshagent.agents.context import SessionUsage
from meshagent.openai.tools.realtime_adapter import (
    DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL,
    OpenAIRealtimeAdapter,
    OpenAIRealtimeSessionContext,
)
from meshagent.tools import FunctionTool, Toolkit


class _FakeParticipant:
    id = "participant-1"

    def get_attribute(self, key: str) -> str | None:
        if key == "name":
            return "caller"
        return None


class _AnyArgsTool(FunctionTool):
    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="test tool",
        )
        self.calls: list[dict[str, object]] = []

    async def execute(self, context, **kwargs):
        del context
        self.calls.append(dict(kwargs))
        return {"ok": True, "args": kwargs}


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.messages: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.closed = False
        self.close_count = 0

    async def send_str(self, data: str) -> None:
        payload = json.loads(data)
        self.sent.append(payload)
        if payload.get("type") == "session.update":
            await self.messages.put(_text_message({"type": "session.updated"}))

    async def receive(self) -> aiohttp.WSMessage:
        return await self.messages.get()

    async def ping(self) -> None:
        return None

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.close_count += 1
        await self.messages.put(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None))


class _FakeSession:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket
        self.ws_connect_calls: list[dict[str, object]] = []
        self.closed = False

    async def ws_connect(
        self,
        url: str,
        *,
        headers: dict[str, str],
        heartbeat: float | None,
        autoping: bool,
    ) -> _FakeWebSocket:
        self.ws_connect_calls.append(
            {
                "url": url,
                "headers": headers,
                "heartbeat": heartbeat,
                "autoping": autoping,
            }
        )
        return self.websocket

    async def close(self) -> None:
        self.closed = True


def _adapter(**kwargs: object) -> OpenAIRealtimeAdapter:
    return OpenAIRealtimeAdapter(
        model="gpt-realtime-test",
        client=AsyncOpenAI(
            api_key="test-key",
            base_url="https://api.openai.test/v1",
        ),
        **kwargs,
    )


class _FakeOpenAIHeadersClient:
    default_headers = {
        "Authorization": "Bearer test-key",
        "openai-beta": "realtime=v1",
        "content-type": "application/json",
        "Content-Length": "10",
    }


def _context(websocket: _FakeWebSocket) -> OpenAIRealtimeSessionContext:
    return OpenAIRealtimeSessionContext(
        session=_FakeSession(websocket),
        websocket_timeout=3600,
    )


def _text_message(event: dict[str, object]) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event), None)


def test_realtime_conversation_item_normalizes_function_call_status() -> None:
    payload = OpenAIRealtimeAdapter._conversation_item_for_message(
        {
            "type": "function_call",
            "id": "call-item-1",
            "call_id": "call-1",
            "name": "write_file",
            "arguments": "{}",
            "status": "in_progress",
        }
    )

    assert payload == {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call",
            "id": "call-item-1",
            "call_id": "call-1",
            "name": "write_file",
            "arguments": "{}",
            "status": "completed",
        },
    }


async def _wait_for_sent_type(
    websocket: _FakeWebSocket, event_type: str
) -> dict[str, object]:
    for _ in range(100):
        for payload in websocket.sent:
            if payload.get("type") == event_type:
                return payload
        await asyncio.sleep(0)
    raise AssertionError(f"{event_type} was not sent")


async def _wait_for_sent_count(
    websocket: _FakeWebSocket, event_type: str, count: int
) -> list[dict[str, object]]:
    for _ in range(100):
        payloads = [
            payload for payload in websocket.sent if payload.get("type") == event_type
        ]
        if len(payloads) >= count:
            return payloads
        await asyncio.sleep(0)
    raise AssertionError(f"{count} {event_type} events were not sent")


@pytest.mark.asyncio
async def test_connect_opens_realtime_websocket_and_sends_session_update() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter(session_options={"modalities": ["text", "audio"]})
    received: list[dict[str, object]] = []

    await adapter.connect(context=context, event_handler=received.append)

    session = context._session
    assert isinstance(session, _FakeSession)
    assert session.ws_connect_calls[0]["url"] == (
        "wss://api.openai.test/v1/realtime?model=gpt-realtime-test"
    )
    headers = session.ws_connect_calls[0]["headers"]
    assert isinstance(headers, dict)
    assert "OpenAI-Beta" not in headers
    assert websocket.sent == [
        {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["text"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": None,
                        "transcription": {
                            "model": DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": "echo",
                    },
                },
            },
        }
    ]
    assert context.is_connected

    await adapter.disconnect(context=context)


def test_websocket_headers_remove_beta_and_http_headers_case_insensitively() -> None:
    headers = _adapter()._websocket_headers(
        openai=_FakeOpenAIHeadersClient(),
        extra_headers={
            "OpenAI-Beta": "realtime=v1",
            "X-Test": "value",
        },
    )

    assert headers == {
        "Authorization": "Bearer test-key",
        "X-Test": "value",
    }


def test_list_models_advertises_realtime_protocols() -> None:
    adapter = _adapter(realtime_protocols=("webrtc", "websocket"))

    assert adapter.list_models()[0].realtime_protocols == ("webrtc", "websocket")


def test_list_models_advertises_realtime_context_windows() -> None:
    adapter = _adapter()

    context_windows = {
        model.name: model.context_window for model in adapter.list_models()
    }

    assert context_windows["gpt-realtime"] == 32000
    assert context_windows["gpt-realtime-1.5"] == 32000
    assert context_windows["gpt-realtime-2"] == 128000
    assert context_windows["gpt-realtime-mini"] == 32000
    assert context_windows["gpt-4o-realtime-preview"] == 32000
    assert context_windows["gpt-4o-mini-realtime-preview"] == 16000


def test_realtime_context_window_size_prefers_specific_model_prefixes() -> None:
    adapter = _adapter()

    assert adapter.context_window_size("gpt-realtime-2") == 128000
    assert adapter.context_window_size("gpt-realtime-2-2026-05-07") == 128000
    assert adapter.context_window_size("gpt-realtime-2025-08-28") == 32000
    assert (
        adapter.context_window_size("gpt-4o-mini-realtime-preview-2024-12-17") == 16000
    )


def test_list_models_uses_response_output_modalities() -> None:
    adapter = _adapter(response_options={"output_modalities": ["audio"]})

    assert adapter.list_models()[0].modalities == ("audio",)


def test_list_models_uses_supported_output_modalities_over_response_default() -> None:
    adapter = _adapter(
        response_options={"output_modalities": ["text"]},
        supported_output_modalities=("text", "audio"),
    )

    assert adapter.list_models()[0].modalities == ("text", "audio")


@pytest.mark.asyncio
async def test_create_realtime_connection_returns_webrtc_endpoint() -> None:
    connection = await _adapter().create_realtime_connection(
        protocol="webrtc",
        model="gpt-realtime-test",
    )

    assert connection.protocol == "webrtc"
    assert connection.url == (
        "https://api.openai.test/v1/realtime/calls?model=gpt-realtime-test"
    )
    assert connection.headers["Authorization"] == "Bearer test-key"
    assert "content-type" not in {key.lower() for key in connection.headers}
    assert "content-length" not in {key.lower() for key in connection.headers}


@pytest.mark.asyncio
async def test_create_response_requires_an_explicit_realtime_connection() -> None:
    adapter = _adapter()
    context = adapter.create_session()

    with pytest.raises(RoomException, match=r"Call connect\(\) before create_response"):
        await adapter.create_response(
            context=context, caller=_FakeParticipant(), toolkits=[]
        )


@pytest.mark.asyncio
async def test_start_session_connects_realtime_websocket_with_instructions() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    context.instructions = "Reply with concise text."
    adapter = _adapter(session_options={"modalities": ["text"]})
    received: list[dict[str, object]] = []

    await adapter.start_session(context=context, event_handler=received.append)

    assert context.is_connected
    assert websocket.sent == [
        {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["text"],
                "instructions": "Reply with concise text.",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": None,
                        "transcription": {
                            "model": DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": "echo",
                    },
                },
            },
        }
    ]

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_start_realtime_session_advertises_function_tools() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter(
        session_options={"modalities": ["audio"]},
        response_options={"modalities": ["audio"]},
    )
    tool = _AnyArgsTool("write_file")

    await adapter.start_realtime_session(
        context=context,
        toolkits=[Toolkit(name="storage", tools=[tool])],
    )

    assert websocket.sent[0]["session"]["tools"] == [
        {
            "type": "function",
            "name": "write_file",
            "description": "test tool",
            "parameters": {
                "type": "object",
                "additionalProperties": True,
            },
        }
    ]

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_realtime_session_executes_tool_calls_from_automatic_response() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter(
        session_options={"modalities": ["audio"]},
        response_options={"modalities": ["audio"]},
    )
    tool = _AnyArgsTool("write_file")
    received: list[dict[str, object]] = []

    await adapter.start_realtime_session(
        context=context,
        event_handler=received.append,
        caller=_FakeParticipant(),
        toolkits=[Toolkit(name="storage", tools=[tool])],
    )

    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {
                    "id": "resp-tool",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc-1",
                            "call_id": "call-1",
                            "name": "write_file",
                            "arguments": '{"path": "note.txt"}',
                        }
                    ],
                },
            }
        )
    )

    conversation_items = await _wait_for_sent_count(
        websocket, "conversation.item.create", 1
    )
    assert conversation_items[-1]["item"] == {
        "type": "function_call_output",
        "id": "call-1",
        "call_id": "call-1",
        "output": '{"ok": true, "args": {"path": "note.txt"}}',
        "status": "completed",
    }
    response_creates = await _wait_for_sent_count(websocket, "response.create", 1)
    assert response_creates[0]["response"]["output_modalities"] == ["audio"]

    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {"id": "resp-final", "status": "completed"},
            }
        )
    )
    for _ in range(100):
        if any(event.get("type") == "response.done" for event in received):
            break
        await asyncio.sleep(0)

    assert tool.calls == [{"path": "note.txt"}]
    assert [event.get("type") for event in received] == [
        "session.updated",
        "meshagent.handler.added",
        "meshagent.handler.done",
        "response.done",
    ]

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_connect_uses_custom_realtime_transcription_model() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter(
        session_options={"output_modalities": ["text"]},
        transcription_model="custom-transcribe",
    )

    await adapter.connect(context=context)

    assert websocket.sent == [
        {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["text"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": None,
                        "transcription": {"model": "custom-transcribe"},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": "echo",
                    },
                },
            },
        }
    ]

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_connect_uses_automatic_realtime_turn_detection() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter(
        session_options={"output_modalities": ["text"]},
        turn_detection="automatic",
    )

    await adapter.connect(context=context)

    audio_input = websocket.sent[0]["session"]["audio"]["input"]
    assert audio_input["turn_detection"] == {"type": "server_vad"}
    assert adapter.list_models()[0].turn_detection == "automatic"

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_connect_preserves_explicit_realtime_transcription_options() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter(
        session_options={
            "output_modalities": ["text"],
            "audio": {
                "input": {
                    "transcription": {"model": "configured-transcribe"},
                }
            },
        },
        transcription_model="constructor-transcribe",
    )

    await adapter.connect(context=context)

    assert websocket.sent[0]["session"] == {
        "type": "realtime",
        "output_modalities": ["text"],
        "audio": {
            "input": {
                "transcription": {"model": "configured-transcribe"},
                "format": {"type": "audio/pcm", "rate": 24000},
                "turn_detection": None,
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "voice": "echo",
            },
        },
    }

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_stop_session_closes_realtime_websocket() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter()

    await adapter.start_session(context=context)
    await adapter.stop_session(context=context)

    assert websocket.close_count == 1
    assert not context.is_connected


@pytest.mark.asyncio
async def test_create_response_sends_response_create_and_returns_terminal_event() -> (
    None
):
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter(response_options={"modalities": ["text", "audio"]})
    received: list[dict[str, object]] = []
    await adapter.connect(context=context, event_handler=received.append)

    task = asyncio.create_task(
        adapter.create_response(
            context=context,
            caller=_FakeParticipant(),
            toolkits=[],
            event_handler=received.append,
        )
    )
    response_create = await _wait_for_sent_type(websocket, "response.create")
    assert response_create == {
        "type": "response.create",
        "response": {"output_modalities": ["text"]},
    }

    await websocket.messages.put(
        _text_message(
            {"type": "response.audio.delta", "item_id": "audio-1", "delta": "abc"}
        )
    )
    await websocket.messages.put(
        _text_message(
            {
                "type": "response.audio_transcript.delta",
                "item_id": "audio-1",
                "delta": "hi",
            }
        )
    )
    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {"id": "resp-1", "status": "completed"},
            }
        )
    )

    terminal_event = await task

    assert terminal_event == {
        "type": "response.done",
        "response": {"id": "resp-1", "status": "completed"},
    }
    assert [event["type"] for event in received] == [
        "session.updated",
        "response.audio.delta",
        "response.audio_transcript.delta",
        "response.done",
    ]

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_create_response_stores_realtime_usage_from_terminal_event() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    usage_updates: list[SessionUsage] = []
    context.set_usage_callback(usage_updates.append)
    adapter = _adapter()
    await adapter.connect(context=context, event_handler=lambda event: None)

    task = asyncio.create_task(
        adapter.create_response(
            context=context,
            caller=_FakeParticipant(),
            toolkits=[],
        )
    )
    await _wait_for_sent_type(websocket, "response.create")
    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {
                    "id": "resp-usage",
                    "model": "gpt-realtime-2",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            }
        )
    )

    await task

    assert context.last_usage == SessionUsage(
        model="gpt-realtime-2",
        usage={
            "input_tokens": 10.0,
            "output_tokens": 5.0,
            "total_tokens": 15.0,
        },
        context_window_used=15,
        context_window_size=128000,
    )
    assert usage_updates == [context.last_usage]

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_create_response_advertises_and_executes_function_tools() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter(response_options={"modalities": ["text"]})
    tool = _AnyArgsTool("write_file")
    await adapter.connect(context=context)

    task = asyncio.create_task(
        adapter.create_response(
            context=context,
            caller=_FakeParticipant(),
            toolkits=[Toolkit(name="storage", tools=[tool])],
        )
    )
    response_create = await _wait_for_sent_type(websocket, "response.create")
    assert response_create == {
        "type": "response.create",
        "response": {
            "output_modalities": ["text"],
            "tools": [
                {
                    "type": "function",
                    "name": "write_file",
                    "description": "test tool",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                }
            ],
        },
    }

    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {
                    "id": "resp-tool",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc-1",
                            "call_id": "call-1",
                            "name": "write_file",
                            "arguments": '{"path": "note.txt"}',
                        }
                    ],
                },
            }
        )
    )

    conversation_items = await _wait_for_sent_count(
        websocket, "conversation.item.create", 1
    )
    assert conversation_items[-1]["item"] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": '{"ok": true, "args": {"path": "note.txt"}}',
        "status": "completed",
        "id": "call-1",
    }
    response_creates = await _wait_for_sent_count(websocket, "response.create", 2)
    function_tool = response_creates[1]["response"]["tools"][0]
    assert function_tool["type"] == "function"
    assert function_tool["name"] == "write_file"
    assert "strict" not in function_tool

    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {"id": "resp-final", "status": "completed"},
            }
        )
    )

    terminal_event = await task
    assert terminal_event["response"]["id"] == "resp-final"
    assert tool.calls == [{"path": "note.txt"}]

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_create_response_allows_multiple_realtime_responses_in_flight() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter(response_options={"modalities": ["text"]})
    await adapter.connect(context=context)

    first = asyncio.create_task(
        adapter.create_response(
            context=context,
            caller=_FakeParticipant(),
            toolkits=[],
        )
    )
    second = asyncio.create_task(
        adapter.create_response(
            context=context,
            caller=_FakeParticipant(),
            toolkits=[],
        )
    )

    response_creates = await _wait_for_sent_count(websocket, "response.create", 2)
    assert response_creates == [
        {"type": "response.create", "response": {"output_modalities": ["text"]}},
        {"type": "response.create", "response": {"output_modalities": ["text"]}},
    ]

    await websocket.messages.put(
        _text_message(
            {
                "type": "response.created",
                "response": {"id": "resp-1", "status": "in_progress"},
            }
        )
    )
    await websocket.messages.put(
        _text_message(
            {
                "type": "response.created",
                "response": {"id": "resp-2", "status": "in_progress"},
            }
        )
    )
    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {"id": "resp-2", "status": "completed"},
            }
        )
    )
    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {"id": "resp-1", "status": "completed"},
            }
        )
    )

    first_result, second_result = await asyncio.gather(first, second)
    assert first_result == {
        "type": "response.done",
        "response": {"id": "resp-1", "status": "completed"},
    }
    assert second_result == {
        "type": "response.done",
        "response": {"id": "resp-2", "status": "completed"},
    }

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_create_response_syncs_text_messages_and_tracks_assistant_response() -> (
    None
):
    websocket = _FakeWebSocket()
    context = _context(websocket)
    context.append_user_message("hello")
    adapter = _adapter(response_options={"modalities": ["text"]})
    await adapter.connect(context=context)

    task = asyncio.create_task(
        adapter.create_response(
            context=context,
            caller=_FakeParticipant(),
            toolkits=[],
        )
    )
    conversation_item = await _wait_for_sent_type(websocket, "conversation.item.create")
    response_create = await _wait_for_sent_type(websocket, "response.create")

    assert conversation_item == {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
    }
    assert response_create == {
        "type": "response.create",
        "response": {"output_modalities": ["text"]},
    }

    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {
                    "id": "resp-1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "hi"}],
                        }
                    ],
                },
            }
        )
    )

    await task

    assert context.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert context.turn_count == 1

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_create_response_replays_assistant_messages_as_output_text() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    context.messages.append({"role": "assistant", "content": "previous reply"})
    adapter = _adapter(response_options={"modalities": ["text"]})
    await adapter.connect(context=context)

    task = asyncio.create_task(
        adapter.create_response(
            context=context,
            caller=_FakeParticipant(),
            toolkits=[],
        )
    )
    conversation_item = await _wait_for_sent_type(websocket, "conversation.item.create")

    assert conversation_item == {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "previous reply"}],
        },
    }

    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {"id": "resp-1", "status": "completed"},
            }
        )
    )

    await task
    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_make_agent_event_reader_accumulates_text_for_realtime_replay() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter(response_options={"modalities": ["text"]})

    restored_messages: list[dict[str, Any]] = []
    reader = adapter.make_agent_event_reader(emit_message=restored_messages.append)
    for message in [
        AgentTextContentStarted(
            type=AGENT_EVENT_TEXT_CONTENT_STARTED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
        ),
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text="Hi",
        ),
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text=" there",
        ),
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text="Hi there",
        ),
        AgentTextContentEnded(
            type=AGENT_EVENT_TEXT_CONTENT_ENDED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
        ),
    ]:
        reader.consume(message)
    reader.finalize()
    adapter.restore_context_messages(context=context, messages=restored_messages)

    assert context.messages == [
        {"role": "assistant", "content": [{"type": "output_text", "text": "Hi there"}]}
    ]

    await adapter.connect(context=context)
    task = asyncio.create_task(
        adapter.create_response(
            context=context,
            caller=_FakeParticipant(),
            toolkits=[],
        )
    )
    conversation_item = await _wait_for_sent_type(websocket, "conversation.item.create")

    assert conversation_item == {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi there"}],
        },
    }

    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {"id": "resp-1", "status": "completed"},
            }
        )
    )

    await task
    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_make_agent_event_reader_restores_tool_lifecycle_for_realtime_replay() -> (
    None
):
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter(response_options={"modalities": ["text"]})
    restored_messages: list[dict[str, Any]] = []
    reader = adapter.make_agent_event_reader(emit_message=restored_messages.append)
    arguments = {
        "operation": {"type": "update_file", "path": "report.py", "diff": "@@"}
    }
    serialized_arguments = json.dumps(arguments, separators=(",", ":"))

    for delta in (serialized_arguments[:20], serialized_arguments[20:]):
        reader.consume(
            AgentToolCallArgumentsDelta(
                type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
                thread_id="thread-1",
                turn_id="turn-1",
                item_id="tool-1",
                namespace="openai.responses",
                call_id="call-1",
                delta=delta,
            )
        )
    reader.consume(
        AgentToolCallStarted(
            type=AGENT_EVENT_TOOL_CALL_STARTED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="tool-1",
            namespace="openai.responses",
            call_id="call-1",
            toolkit="openai",
            tool="apply_patch",
            arguments=arguments,
        )
    )
    reader.consume(
        AgentToolCallEnded(
            type=AGENT_EVENT_TOOL_CALL_ENDED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="tool-1",
            namespace="openai.responses",
            call_id="call-1",
            toolkit="openai",
            tool="apply_patch",
            result=TextContent(text="patched"),
        )
    )
    reader.finalize()
    adapter.restore_context_messages(context=context, messages=restored_messages)

    assert context.messages == [
        {
            "type": "function_call",
            "id": "tool-1",
            "call_id": "call-1",
            "name": "apply_patch_call",
            "arguments": serialized_arguments,
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "id": "tool-1:output",
            "call_id": "call-1",
            "output": "patched",
            "status": "completed",
        },
    ]

    await adapter.connect(context=context)
    task = asyncio.create_task(
        adapter.create_response(
            context=context,
            caller=_FakeParticipant(),
            toolkits=[],
        )
    )
    for _ in range(100):
        conversation_items = [
            payload
            for payload in websocket.sent
            if payload.get("type") == "conversation.item.create"
        ]
        if len(conversation_items) == 2:
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("restored tool call items were not replayed")

    assert conversation_items[0]["item"] == context.messages[0]
    assert conversation_items[1]["item"] == context.messages[1]

    await websocket.messages.put(
        _text_message(
            {
                "type": "response.done",
                "response": {"id": "resp-1", "status": "completed"},
            }
        )
    )

    await task
    await adapter.disconnect(context=context)


def test_make_agent_event_reader_ignores_realtime_audio_generation_for_replay() -> None:
    adapter = _adapter(response_options={"modalities": ["audio"]})

    restored_messages: list[dict[str, Any]] = []
    reader = adapter.make_agent_event_reader(emit_message=restored_messages.append)
    for message in [
        AgentAudioGenerationStarted(
            type=AGENT_EVENT_AUDIO_GENERATION_STARTED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="audio-1",
        ),
        AgentAudioGenerationDelta(
            type=AGENT_EVENT_AUDIO_GENERATION_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="audio-1",
            data=b"\xf2\x00\x01",
        ),
        AgentAudioGenerationCompleted(
            type=AGENT_EVENT_AUDIO_GENERATION_COMPLETED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="audio-1",
        ),
    ]:
        reader.consume(message)
    reader.finalize()

    assert restored_messages == []


def test_realtime_publisher_emits_audio_generation_and_transcription_lifecycles() -> (
    None
):
    adapter = _adapter()
    messages: list[AgentMessage] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.audio.delta",
            "item_id": "audio-1",
            "content_index": 0,
            "delta": "AAA=",
            "mime_type": "audio/wav",
        }
    )
    publisher(
        {
            "type": "response.audio.done",
            "item_id": "audio-1",
            "content_index": 0,
            "audio": "AAA=",
            "mime_type": "audio/wav",
        }
    )
    publisher(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "user-audio-1",
            "delta": "hel",
        }
    )
    publisher(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "user-audio-1",
            "transcript": "hello",
        }
    )
    publisher(
        {
            "type": "input_audio_buffer.speech_started",
            "item_id": "user-audio-2",
            "audio_start_ms": 120,
        }
    )
    publisher(
        {
            "type": "input_audio_buffer.speech_stopped",
            "item_id": "user-audio-2",
            "audio_end_ms": 450,
        }
    )

    assert [type(message) for message in messages] == [
        AgentAudioGenerationStarted,
        AgentAudioGenerationDelta,
        AgentAudioGenerationCompleted,
        AgentAudioTranscriptionStarted,
        AgentAudioTranscriptionDelta,
        AgentAudioTranscriptionCompleted,
        AgentAudioInputSpeechStarted,
        AgentAudioInputSpeechEnded,
    ]
    generation_completed = messages[2]
    assert isinstance(generation_completed, AgentAudioGenerationCompleted)
    assert generation_completed.audio is not None
    assert generation_completed.audio.uri == "data:audio/wav;base64,AAA="
    transcription_completed = messages[5]
    assert isinstance(transcription_completed, AgentAudioTranscriptionCompleted)
    assert transcription_completed.text == "hello"
    speech_started = messages[6]
    assert isinstance(speech_started, AgentAudioInputSpeechStarted)
    assert speech_started.item_id == "user-audio-2"
    assert speech_started.audio_start_ms == 120
    speech_ended = messages[7]
    assert isinstance(speech_ended, AgentAudioInputSpeechEnded)
    assert speech_ended.item_id == "user-audio-2"
    assert speech_ended.audio_end_ms == 450


@pytest.mark.asyncio
async def test_input_audio_helpers_send_realtime_input_buffer_events() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    adapter = _adapter()
    await adapter.connect(context=context)

    await adapter.append_input_audio(context=context, audio=b"\x00\x01")
    await adapter.commit_input_audio(context=context)
    await adapter.clear_input_audio(context=context)

    assert websocket.sent[1:] == [
        {"type": "input_audio_buffer.append", "audio": "AAE="},
        {"type": "input_audio_buffer.commit"},
        {"type": "input_audio_buffer.clear"},
    ]


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_LIVE_TESTS") != "1",
    reason="set RUN_OPENAI_LIVE_TESTS=1 to run live OpenAI realtime tests",
)
async def test_live_openai_realtime_returns_text_response() -> None:
    events: list[dict[str, object]] = []
    adapter = OpenAIRealtimeAdapter(
        model=os.getenv("OPENAI_REALTIME_LIVE_TEST_MODEL", "gpt-realtime"),
        client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        session_options={"output_modalities": ["text"]},
        response_options={"output_modalities": ["text"]},
        websocket_timeout=120,
    )
    context = adapter.create_session()
    context.append_user_message("Reply with exactly REALTIME_OK and nothing else.")

    try:
        await adapter.start_realtime_session(
            context=context,
            event_handler=events.append,
            caller=_FakeParticipant(),
            toolkits=[],
        )
        terminal_event = await asyncio.wait_for(
            adapter.create_response(
                context=context,
                caller=_FakeParticipant(),
                toolkits=[],
                event_handler=events.append,
            ),
            timeout=120.0,
        )
    finally:
        await context.close()

    assert terminal_event["type"] == "response.done"
    assert any(
        message.get("role") == "assistant"
        and "REALTIME_OK" in str(message.get("content"))
        for message in context.messages
    )
