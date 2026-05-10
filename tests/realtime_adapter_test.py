import asyncio
import json
from typing import Any

import aiohttp
import pytest
from openai import AsyncOpenAI

from meshagent.agents.messages import (
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AgentAudioGenerationCompleted,
    AgentAudioGenerationDelta,
    AgentAudioGenerationStarted,
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
from meshagent.openai.tools.realtime_adapter import (
    OpenAIRealtimeAdapter,
    OpenAIRealtimeSessionContext,
)


class _FakeParticipant:
    id = "participant-1"

    def get_attribute(self, key: str) -> str | None:
        if key == "name":
            return "caller"
        return None


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.messages: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.closed = False
        self.close_count = 0

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

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


async def _wait_for_sent_type(
    websocket: _FakeWebSocket, event_type: str
) -> dict[str, object]:
    for _ in range(100):
        for payload in websocket.sent:
            if payload.get("type") == event_type:
                return payload
        await asyncio.sleep(0)
    raise AssertionError(f"{event_type} was not sent")


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
                "output_modalities": ["text", "audio"],
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
            },
        }
    ]

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
        "response": {"output_modalities": ["text", "audio"]},
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
        "response.audio.delta",
        "response.audio_transcript.delta",
        "response.done",
    ]

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

    assert context.previous_response_id == "resp-1"
    assert context.messages == []
    assert context.previous_messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert context.turn_count == 1

    await adapter.disconnect(context=context)


@pytest.mark.asyncio
async def test_create_response_replays_assistant_messages_as_output_text() -> None:
    websocket = _FakeWebSocket()
    context = _context(websocket)
    context.previous_messages.append({"role": "assistant", "content": "previous reply"})
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

    assert context.messages == []
    assert context.previous_messages == [
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

    assert context.messages == []
    assert context.previous_messages == [
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

    assert conversation_items[0]["item"] == context.previous_messages[0]
    assert conversation_items[1]["item"] == context.previous_messages[1]

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

    assert [type(message) for message in messages] == [
        AgentAudioGenerationStarted,
        AgentAudioGenerationDelta,
        AgentAudioGenerationCompleted,
        AgentAudioTranscriptionStarted,
        AgentAudioTranscriptionDelta,
        AgentAudioTranscriptionCompleted,
    ]
    generation_completed = messages[2]
    assert isinstance(generation_completed, AgentAudioGenerationCompleted)
    assert generation_completed.audio is not None
    assert generation_completed.audio.uri == "data:audio/wav;base64,AAA="
    transcription_completed = messages[5]
    assert isinstance(transcription_completed, AgentAudioTranscriptionCompleted)
    assert transcription_completed.text == "hello"


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

    await adapter.disconnect(context=context)
