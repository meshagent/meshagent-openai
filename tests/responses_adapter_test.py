import asyncio
import httpx
import logging
import aiohttp
import pytest
from aiohttp.client_reqrep import RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from openai._models import BaseModel as OpenAIBaseModel
from types import SimpleNamespace
from openai import APIError
from openai.types.responses.response_computer_tool_call import ResponseComputerToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from yarl import URL

from meshagent.api import RoomException
from meshagent.api.messaging import JsonContent, TextContent
from meshagent.computers.agent import ComputerToolkit
from meshagent.computers.operator import Operator
from meshagent.openai.tools.responses_adapter import (
    OpenAIResponsesAdapter,
    OpenAIResponsesSessionContext,
    _consume_streaming_tool_result,
)


class _FakeDeveloper:
    def log_nowait(self, *, type: str, data: dict) -> None:
        del type
        del data


class _FakeParticipant:
    def __init__(self):
        self.id = "participant_1"

    def get_attribute(self, key: str):
        if key == "name":
            return "assistant"
        return None


class _FakeRoom:
    def __init__(self):
        self.local_participant = _FakeParticipant()
        self.developer = _FakeDeveloper()


class _FakeBrowserComputer:
    environment = "browser"
    dimensions = (1024, 768)

    def __init__(self):
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return self

    async def __aexit__(self, exc_type, exc, exc_tb):
        del exc_type
        del exc
        del exc_tb
        self.exit_count += 1

    async def click(self, x: int, y: int, button: str = "left") -> None:
        self.calls.append(("click", {"x": x, "y": y, "button": button}))

    async def screenshot(self) -> str:
        return "ZmFrZS1zY3JlZW5zaG90"

    async def get_current_url(self) -> str:
        return "https://example.com"


class _FakeResponse:
    def __init__(
        self,
        *,
        response_id: str,
        output: list[OpenAIBaseModel] | None = None,
        usage: dict | None = None,
    ):
        self.id = response_id
        self.output = output or []
        self.usage = usage

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "output": [output.to_dict(mode="json") for output in self.output],
        }


class _FakeCompletedEvent:
    def __init__(self, *, response: _FakeResponse):
        self.type = "response.completed"
        self.response = response

    def model_dump(self, *, mode: str = "json") -> dict:
        del mode
        return {
            "type": self.type,
            "response": self.response.to_dict(),
        }

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class _FailingStream:
    def __init__(self, *, error: Exception):
        self._error = error
        self._raised = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._raised:
            raise StopAsyncIteration
        self._raised = True
        raise self._error


class _CompletedStream:
    def __init__(self, *, event: _FakeCompletedEvent):
        self._event = event
        self._yielded = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        return self._event


class _FakeResponsesClient:
    def __init__(self, *, outcomes: list[object]):
        self._outcomes = outcomes.copy()
        self.calls = 0
        self.create_kwargs: list[dict] = []

    async def create(self, **kwargs):
        self.create_kwargs.append(kwargs)
        self.calls += 1
        if len(self._outcomes) == 0:
            raise AssertionError("no responses.create outcomes configured")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeOpenAIClient:
    def __init__(self, *, outcomes: list[object]):
        self.responses = _FakeResponsesClient(outcomes=outcomes)


class _FakeWebSocket:
    def __init__(self):
        self.closed = False
        self.pings = 0

    async def ping(self):
        self.pings += 1

    async def close(self):
        self.closed = True


class _FakeLoggingWsMessage:
    def __init__(self, *, type: aiohttp.WSMsgType, data: str):
        self.type = type
        self.data = data


class _FakeLoggingWebSocket:
    def __init__(self, *, messages: list[_FakeLoggingWsMessage]):
        self._messages = messages.copy()
        self.sent_payloads: list[str] = []
        self.closed = False

    async def send_str(self, payload: str):
        self.sent_payloads.append(payload)

    async def receive(self):
        if len(self._messages) == 0:
            raise AssertionError("no websocket messages configured")
        return self._messages.pop(0)

    async def close(self):
        self.closed = True

    def exception(self):
        return None


class _FakeClientSession:
    def __init__(self, websocket: _FakeWebSocket):
        self._websocket = websocket
        self.closed = False
        self.connect_calls = 0

    async def ws_connect(self, *args, **kwargs):
        del args
        del kwargs
        self.connect_calls += 1
        return self._websocket

    async def close(self):
        self.closed = True


class _FailingHandshakeClientSession:
    def __init__(self, *, error: aiohttp.WSServerHandshakeError):
        self._error = error
        self.closed = False

    async def ws_connect(self, *args, **kwargs):
        del args, kwargs
        raise self._error

    async def close(self):
        self.closed = True


class _FailingConnectClientSession:
    def __init__(self):
        self.closed = False

    async def ws_connect(self, *args, **kwargs):
        del args
        del kwargs
        raise RuntimeError("ws connect failed")

    async def close(self):
        self.closed = True


def _make_ws_handshake_error(
    *, status: int, headers: dict[str, str] | None = None
) -> aiohttp.WSServerHandshakeError:
    request_headers = CIMultiDictProxy(CIMultiDict())
    url = URL("ws://localhost:8080/openai/v1/responses")
    return aiohttp.WSServerHandshakeError(
        request_info=RequestInfo(
            url=url,
            method="GET",
            headers=request_headers,
            real_url=url,
        ),
        history=(),
        status=status,
        message="Invalid response status",
        headers=CIMultiDictProxy(CIMultiDict(headers or {})),
    )


class _ToolItemStream:
    def __init__(self, *, items: list[object]):
        self._items = items

    def __aiter__(self):
        return self._run()

    async def _run(self):
        for item in self._items:
            yield item


def _make_api_error(message: str) -> APIError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return APIError(message, request=request, body=None)


def _make_output_message(
    *, message_id: str, text: str, phase: str | None = None
) -> ResponseOutputMessage:
    message_kwargs: dict[str, object] = {
        "id": message_id,
        "content": [ResponseOutputText(annotations=[], text=text, type="output_text")],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }
    if phase is not None:
        message_kwargs["phase"] = phase

    return ResponseOutputMessage(**message_kwargs)


@pytest.mark.asyncio
async def test_create_session_returns_openai_responses_session_context():
    adapter = OpenAIResponsesAdapter()
    context = adapter.create_session()
    assert isinstance(context, OpenAIResponsesSessionContext)


@pytest.mark.asyncio
async def test_get_openai_client_passes_optional_session(monkeypatch):
    adapter = OpenAIResponsesAdapter()
    room = _FakeRoom()
    client_session = httpx.AsyncClient()
    fake_client = object()
    call_args: dict[str, object] = {}

    def _fake_get_client(*, room, http_client=None, session=None):
        call_args["room"] = room
        call_args["http_client"] = http_client
        call_args["session"] = session
        return fake_client

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.get_client",
        _fake_get_client,
    )

    try:
        client = adapter.get_openai_client(room=room, session=client_session)
    finally:
        await client_session.aclose()

    assert client is fake_client
    assert call_args["room"] is room
    assert call_args["http_client"] is call_args["session"]
    assert call_args["session"] is client_session


def test_constructor_rejects_invalid_compaction_threshold():
    with pytest.raises(ValueError, match="compaction_threshold must be greater than 0"):
        OpenAIResponsesAdapter(compaction_threshold=0)


def test_constructor_disables_compaction_when_threshold_is_infinity():
    adapter = OpenAIResponsesAdapter(compaction_threshold=float("inf"))
    assert adapter._compaction_threshold is None


def test_constructor_rejects_invalid_context_management_mode():
    with pytest.raises(
        ValueError,
        match="context_management must be one of 'auto', 'standalone', or 'none'",
    ):
        OpenAIResponsesAdapter(context_management="invalid")


@pytest.mark.asyncio
async def test_next_uses_websocket_path_when_mode_is_websocket(monkeypatch):
    adapter = OpenAIResponsesAdapter(
        mode="websocket",
        client=_FakeOpenAIClient(outcomes=[]),
    )
    context = adapter.create_session()
    context.append_user_message("hello")

    call_count = {"count": 0}

    async def _fake_create_response_websocket_stream(**kwargs):
        del kwargs
        call_count["count"] += 1
        return _CompletedStream(
            event=_FakeCompletedEvent(
                response=_FakeResponse(
                    response_id="resp_ws",
                    usage={
                        "input_tokens": 12,
                        "output_tokens": 4,
                        "input_tokens_details": {"cached_tokens": 3},
                    },
                )
            )
        )

    monkeypatch.setattr(
        adapter,
        "_create_response_websocket_stream",
        _fake_create_response_websocket_stream,
    )

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
    )

    assert result == ""
    assert call_count["count"] == 1
    assert context.turn_count == 1
    assert context.metadata["last_response_usage"]["input_tokens"] == 12
    assert context.usage == {
        "input_tokens": 12.0,
        "output_tokens": 4.0,
        "cached_tokens": 3.0,
    }


@pytest.mark.asyncio
async def test_next_tracks_usage_for_non_streaming_request_mode():
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=_FakeOpenAIClient(
            outcomes=[
                _FakeResponse(
                    response_id="resp_request",
                    usage={
                        "input_tokens": 9,
                        "output_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 5},
                    },
                )
            ]
        ),
    )
    context = adapter.create_session()
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
    )

    assert result == ""
    assert context.turn_count == 1
    assert context.metadata["last_response_usage"]["input_tokens"] == 9
    assert context.usage == {
        "input_tokens": 9.0,
        "output_tokens": 2.0,
        "cached_tokens": 5.0,
    }


@pytest.mark.asyncio
async def test_next_continues_until_final_answer_when_phase_is_present():
    client = _FakeOpenAIClient(
        outcomes=[
            _FakeResponse(
                response_id="resp_commentary",
                output=[
                    _make_output_message(
                        message_id="msg_commentary",
                        text="Working on it.",
                        phase="commentary",
                    )
                ],
            ),
            _FakeResponse(
                response_id="resp_final",
                output=[
                    _make_output_message(
                        message_id="msg_final",
                        text="Done.",
                        phase="final_answer",
                    )
                ],
            ),
        ]
    )
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=client,
    )
    context = adapter.create_session()
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
    )

    assert result == "Done."
    assert client.responses.calls == 2
    assert (
        client.responses.create_kwargs[1]["previous_response_id"] == "resp_commentary"
    )
    assert client.responses.create_kwargs[1]["input"] == []


@pytest.mark.asyncio
async def test_next_handles_openai_54_computer_output_items():
    room = _FakeRoom()
    computer = _FakeBrowserComputer()
    toolkit = ComputerToolkit(
        computer=computer,
        operator=Operator(),
        room=room,
        render_screen=None,
    )
    computer_item = ResponseComputerToolCall(
        id="item_computer",
        type="computer_call",
        status="completed",
        call_id="call_computer",
        action={
            "type": "click",
            "x": 140,
            "y": 320,
            "button": "left",
        },
        pending_safety_checks=[],
    )
    client = _FakeOpenAIClient(
        outcomes=[
            _FakeResponse(
                response_id="resp_computer",
                output=[computer_item],
            ),
            _FakeResponse(
                response_id="resp_computer_final",
                output=[
                    _make_output_message(
                        message_id="msg_computer_final",
                        text="Done.",
                        phase="final_answer",
                    )
                ],
            ),
        ]
    )
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=client,
    )
    context = adapter.create_session()
    context.append_user_message("click the page")

    result = await adapter.next(
        context=context,
        room=room,
        toolkits=[toolkit],
    )

    assert result == "Done."
    assert computer.calls == [
        ("click", {"x": 140, "y": 320, "button": "left"}),
    ]
    assert any(
        isinstance(tool, dict) and tool.get("type") == "computer"
        for tool in client.responses.create_kwargs[0]["tools"]
    )
    assert client.responses.create_kwargs[1]["previous_response_id"] == "resp_computer"
    assert any(
        isinstance(item, dict) and item.get("type") == "computer_call_output"
        for item in context.previous_messages
    )


@pytest.mark.asyncio
async def test_next_tracks_usage_for_streaming_request_mode():
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=_FakeOpenAIClient(
            outcomes=[
                _CompletedStream(
                    event=_FakeCompletedEvent(
                        response=_FakeResponse(
                            response_id="resp_stream",
                            usage={
                                "input_tokens": 11,
                                "output_tokens": 7,
                                "input_tokens_details": {"cached_tokens": 4},
                            },
                        )
                    )
                )
            ]
        ),
    )
    context = adapter.create_session()
    context.append_user_message("hello")
    events: list[dict] = []

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
        event_handler=events.append,
    )

    assert result == ""
    assert context.turn_count == 1
    assert events[0]["type"] == "response.completed"
    assert context.metadata["last_response_usage"]["output_tokens"] == 7
    assert context.usage == {
        "input_tokens": 11.0,
        "output_tokens": 7.0,
        "cached_tokens": 4.0,
    }


@pytest.mark.asyncio
async def test_next_stream_continues_until_final_answer_when_phase_is_present():
    client = _FakeOpenAIClient(
        outcomes=[
            _CompletedStream(
                event=_FakeCompletedEvent(
                    response=_FakeResponse(
                        response_id="resp_stream_commentary",
                        output=[
                            _make_output_message(
                                message_id="msg_stream_commentary",
                                text="Still working.",
                                phase="commentary",
                            )
                        ],
                    )
                )
            ),
            _CompletedStream(
                event=_FakeCompletedEvent(
                    response=_FakeResponse(
                        response_id="resp_stream_final",
                        output=[
                            _make_output_message(
                                message_id="msg_stream_final",
                                text="Finished.",
                                phase="final_answer",
                            )
                        ],
                    )
                )
            ),
        ]
    )
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=client,
    )
    context = adapter.create_session()
    context.append_user_message("hello")
    events: list[dict] = []

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
        event_handler=events.append,
    )

    assert result == "Finished."
    assert client.responses.calls == 2
    assert [event["type"] for event in events] == [
        "response.completed",
        "response.completed",
    ]
    assert (
        client.responses.create_kwargs[1]["previous_response_id"]
        == "resp_stream_commentary"
    )


@pytest.mark.asyncio
async def test_next_uses_auto_compaction_context_management_when_compaction_threshold_set(
    monkeypatch,
):
    client = _FakeOpenAIClient(outcomes=[_FakeResponse(response_id="resp_auto")])
    adapter = OpenAIResponsesAdapter(
        client=client,
        mode="request",
        compaction_threshold=10000,
        max_output_tokens=500,
    )
    context = adapter.create_session()
    context.append_user_message("hello")
    context.metadata["last_response_usage"] = {
        "input_tokens": 200000,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 1000,
    }
    context.metadata["last_response_model"] = "gpt-5.2"

    async def _fail_compact(**kwargs):
        del kwargs
        raise AssertionError(
            "manual compact should not run when compaction_threshold is set"
        )

    monkeypatch.setattr(adapter, "compact", _fail_compact)

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
    )

    assert result == ""
    assert len(client.responses.create_kwargs) == 1
    create_kwargs = client.responses.create_kwargs[0]
    assert create_kwargs["context_management"] == [
        {"type": "compaction", "compact_threshold": 10000}
    ]


@pytest.mark.asyncio
async def test_next_disables_auto_compaction_by_default_for_unknown_model(monkeypatch):
    client = _FakeOpenAIClient(
        outcomes=[_FakeResponse(response_id="resp_unknown_model")]
    )
    adapter = OpenAIResponsesAdapter(
        client=client,
        mode="request",
        model="computer-use-preview",
        max_output_tokens=500,
    )
    context = adapter.create_session()
    context.append_user_message("hello")
    context.metadata["last_response_usage"] = {
        "input_tokens": 200000,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 1000,
    }
    context.metadata["last_response_model"] = "computer-use-preview"

    async def _fail_compact(**kwargs):
        del kwargs
        raise AssertionError(
            "manual compact should not run when auto compaction is disabled"
        )

    monkeypatch.setattr(adapter, "compact", _fail_compact)

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
    )

    assert result == ""
    assert len(client.responses.create_kwargs) == 1
    create_kwargs = client.responses.create_kwargs[0]
    assert "context_management" not in create_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_type", ["computer_use_preview", "computer"])
async def test_next_disables_auto_compaction_when_computer_use_tool_present(
    monkeypatch, tool_type: str
):
    client = _FakeOpenAIClient(
        outcomes=[_FakeResponse(response_id="resp_computer_use_tool")]
    )
    adapter = OpenAIResponsesAdapter(
        client=client,
        mode="request",
        model="gpt-5.2",
        compaction_threshold=10000,
        max_output_tokens=500,
    )
    context = adapter.create_session()
    context.append_user_message("hello")

    async def _fail_compact(**kwargs):
        del kwargs
        raise AssertionError(
            "manual compact should not run when auto compaction is disabled"
        )

    monkeypatch.setattr(adapter, "compact", _fail_compact)

    from meshagent.openai.tools import responses_adapter as responses_adapter_module

    monkeypatch.setattr(
        responses_adapter_module.ResponsesToolBundle,
        "to_json",
        lambda self: [  # noqa: ARG005
            {
                "type": tool_type,
                "display_width": 1024,
                "display_height": 768,
                "environment": "browser",
            }
        ],
    )

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
    )

    assert result == ""
    assert len(client.responses.create_kwargs) == 1
    create_kwargs = client.responses.create_kwargs[0]
    assert "context_management" not in create_kwargs


@pytest.mark.asyncio
async def test_next_uses_auto_compaction_by_default(monkeypatch):
    client = _FakeOpenAIClient(
        outcomes=[_FakeResponse(response_id="resp_auto_default")]
    )
    adapter = OpenAIResponsesAdapter(
        client=client,
        mode="request",
        max_output_tokens=500,
    )
    context = adapter.create_session()
    context.append_user_message("hello")
    context.metadata["last_response_usage"] = {
        "input_tokens": 200000,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 1000,
    }
    context.metadata["last_response_model"] = "gpt-5.2"

    async def _fail_compact(**kwargs):
        del kwargs
        raise AssertionError(
            "manual compact should not run with default auto compaction"
        )

    monkeypatch.setattr(adapter, "compact", _fail_compact)

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
    )

    assert result == ""
    assert len(client.responses.create_kwargs) == 1
    create_kwargs = client.responses.create_kwargs[0]
    assert create_kwargs["context_management"] == [
        {"type": "compaction", "compact_threshold": 200000}
    ]


@pytest.mark.asyncio
async def test_next_uses_manual_compaction_in_standalone_mode(monkeypatch):
    client = _FakeOpenAIClient(
        outcomes=[_FakeResponse(response_id="resp_standalone_compaction")]
    )
    adapter = OpenAIResponsesAdapter(
        client=client,
        mode="request",
        context_management="standalone",
    )
    context = adapter.create_session()
    context.append_user_message("hello")
    context.metadata["last_response_usage"] = {
        "input_tokens": 300000,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 1000,
    }
    context.metadata["last_response_model"] = "gpt-5.2"
    compact_call_count = {"count": 0}

    async def _fake_compact(**kwargs):
        del kwargs
        compact_call_count["count"] += 1

    monkeypatch.setattr(adapter, "compact", _fake_compact)

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
    )

    assert result == ""
    assert compact_call_count["count"] == 1
    assert len(client.responses.create_kwargs) == 1
    create_kwargs = client.responses.create_kwargs[0]
    assert "context_management" not in create_kwargs


@pytest.mark.asyncio
async def test_next_disables_compaction_when_context_management_none(monkeypatch):
    client = _FakeOpenAIClient(
        outcomes=[_FakeResponse(response_id="resp_no_compaction")]
    )
    adapter = OpenAIResponsesAdapter(
        client=client,
        mode="request",
        context_management="none",
    )
    context = adapter.create_session()
    context.append_user_message("hello")
    context.metadata["last_response_usage"] = {
        "input_tokens": 300000,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 1000,
    }
    context.metadata["last_response_model"] = "gpt-5.2"

    async def _fail_compact(**kwargs):
        del kwargs
        raise AssertionError(
            "compact should not be called when context_management=none"
        )

    monkeypatch.setattr(adapter, "compact", _fail_compact)

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
    )

    assert result == ""
    assert len(client.responses.create_kwargs) == 1
    create_kwargs = client.responses.create_kwargs[0]
    assert "context_management" not in create_kwargs


@pytest.mark.asyncio
async def test_websocket_mode_logs_request_and_response_payload_when_enabled(
    monkeypatch, caplog
):
    adapter = OpenAIResponsesAdapter(mode="websocket", log_requests=True)
    context = adapter.create_session()

    websocket = _FakeLoggingWebSocket(
        messages=[
            _FakeLoggingWsMessage(
                type=aiohttp.WSMsgType.TEXT,
                data='{"type":"response.completed"}',
            )
        ]
    )

    async def _fake_ensure_websocket(*, url: str, headers: dict[str, str]):
        del url
        del headers
        return websocket

    monkeypatch.setattr(
        context,
        "ensure_websocket",
        _fake_ensure_websocket,
    )
    monkeypatch.setattr(
        adapter,
        "_coerce_response_stream_event",
        lambda payload: _FakeCompletedEvent(
            response=_FakeResponse(response_id="resp_ws")
        ),
    )

    caplog.set_level(logging.INFO, logger="openai_agent")

    stream = await adapter._create_response_websocket_stream(
        context=context,
        room=_FakeRoom(),
        openai=SimpleNamespace(
            base_url="https://example.com/openai/v1",
            default_headers={"Authorization": "Bearer test-token"},
        ),
        create_kwargs={
            "stream": True,
            "model": "gpt-5.2",
            "input": [{"role": "user", "content": "hello"}],
        },
        extra_headers={},
    )

    events = [event async for event in stream]

    assert len(events) == 1
    assert events[0].type == "response.completed"
    assert len(websocket.sent_payloads) == 1

    logged_messages = [record.message for record in caplog.records]
    assert any(message.startswith("==> WS ") for message in logged_messages)
    assert any(
        "headers=" in message and "***REDACTED***" in message
        for message in logged_messages
    )
    assert any(
        message.startswith("<== WS event=response.completed")
        for message in logged_messages
    )


@pytest.mark.asyncio
async def test_create_response_websocket_stream_converts_raw_handshake_errors(
    monkeypatch,
):
    adapter = OpenAIResponsesAdapter(mode="websocket")
    context = adapter.create_session()
    handshake_error = _make_ws_handshake_error(
        status=402,
        headers={
            "X-Meshagent-Error-Message": "Your account is out of credits. Add credits to continue.",
        },
    )

    async def _failing_ensure_websocket(*, url: str, headers: dict[str, str]):
        del url, headers
        raise handshake_error

    monkeypatch.setattr(context, "ensure_websocket", _failing_ensure_websocket)

    with pytest.raises(
        RoomException, match="Your account is out of credits. Add credits to continue."
    ) as exc_info:
        await adapter._create_response_websocket_stream(
            context=context,
            room=_FakeRoom(),
            openai=SimpleNamespace(
                base_url="https://example.com/openai/v1",
                default_headers={"Authorization": "Bearer test-token"},
            ),
            create_kwargs={
                "stream": True,
                "model": "gpt-5.2",
                "input": [{"role": "user", "content": "hello"}],
            },
            extra_headers={},
        )

    assert exc_info.value.status_code == 402


@pytest.mark.asyncio
async def test_session_context_reuses_websocket_and_closes_after_timeout(monkeypatch):
    fake_websocket = _FakeWebSocket()
    fake_session = _FakeClientSession(fake_websocket)

    def _fake_client_session(*args, **kwargs):
        del args
        del kwargs
        return fake_session

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.aiohttp.ClientSession",
        _fake_client_session,
    )

    context = OpenAIResponsesSessionContext(
        system_role=None,
        websocket_timeout=0.05,
        websocket_ping_interval_seconds=0.01,
    )

    ws1 = await context.ensure_websocket(
        url="ws://localhost:8080/openai/v1/responses",
        headers={"Authorization": "Bearer test-token"},
    )
    ws2 = await context.ensure_websocket(
        url="ws://localhost:8080/openai/v1/responses",
        headers={"Authorization": "Bearer test-token"},
    )

    assert ws1 is ws2
    assert fake_session.connect_calls == 1

    await asyncio.sleep(0.08)

    assert fake_websocket.closed is True
    assert fake_session.closed is True
    assert fake_websocket.pings > 0


@pytest.mark.asyncio
async def test_session_context_uses_injected_session_without_closing_by_default():
    fake_websocket = _FakeWebSocket()
    shared_session = _FakeClientSession(fake_websocket)

    context = OpenAIResponsesSessionContext(
        system_role=None,
        session=shared_session,
    )

    await context.ensure_websocket(
        url="ws://localhost:8080/openai/v1/responses",
        headers={"Authorization": "Bearer test-token"},
    )
    await context.close()

    assert shared_session.connect_calls == 1
    assert fake_websocket.closed is True
    assert shared_session.closed is False


@pytest.mark.asyncio
async def test_session_context_closes_created_session_on_close(monkeypatch):
    fake_websocket = _FakeWebSocket()
    fake_session = _FakeClientSession(fake_websocket)

    def _fake_client_session(*args, **kwargs):
        del args, kwargs
        return fake_session

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.aiohttp.ClientSession",
        _fake_client_session,
    )

    context = OpenAIResponsesSessionContext(system_role=None)

    await context.ensure_websocket(
        url="ws://localhost:8080/openai/v1/responses",
        headers={"Authorization": "Bearer test-token"},
    )
    await context.close()

    assert fake_session.connect_calls == 1
    assert fake_websocket.closed is True
    assert fake_session.closed is True


@pytest.mark.asyncio
async def test_session_context_converts_websocket_handshake_errors_to_room_exception(
    monkeypatch,
):
    handshake_error = _make_ws_handshake_error(
        status=402,
        headers={
            "X-Meshagent-Error-Message": "Your account is out of credits. Add credits to continue.",
        },
    )
    fake_session = _FailingHandshakeClientSession(error=handshake_error)

    def _fake_client_session(*args, **kwargs):
        del args, kwargs
        return fake_session

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.aiohttp.ClientSession",
        _fake_client_session,
    )

    context = OpenAIResponsesSessionContext(system_role=None)

    with pytest.raises(
        RoomException, match="Your account is out of credits. Add credits to continue."
    ) as exc_info:
        await context.ensure_websocket(
            url="ws://localhost:8080/openai/v1/responses",
            headers={"Authorization": "Bearer test-token"},
        )

    assert exc_info.value.status_code == 402
    assert fake_session.closed is True


@pytest.mark.asyncio
async def test_session_context_closes_session_when_ws_connect_fails(monkeypatch):
    fake_session = _FailingConnectClientSession()

    def _fake_client_session(*args, **kwargs):
        del args
        del kwargs
        return fake_session

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.aiohttp.ClientSession",
        _fake_client_session,
    )

    context = OpenAIResponsesSessionContext(system_role=None)

    with pytest.raises(RuntimeError, match="ws connect failed"):
        await context.ensure_websocket(
            url="ws://localhost:8080/openai/v1/responses",
            headers={"Authorization": "Bearer test-token"},
        )

    assert fake_session.closed is True


@pytest.mark.asyncio
async def test_consume_streaming_tool_result_emits_intermediate_json_events():
    events: list[dict] = []
    result = await _consume_streaming_tool_result(
        tool_name="computer_call",
        tool_call_id="call_1",
        item_id="item_1",
        stream=_ToolItemStream(
            items=[
                JsonContent(
                    json={
                        "type": "agent.event",
                        "headline": "Starting Playwright container",
                    }
                ),
                TextContent(text="tool-finished"),
            ]
        ),
        event_handler=events.append,
    )

    assert isinstance(result, TextContent)
    assert result.text == "tool-finished"
    assert events == [
        {"type": "agent.event", "headline": "Starting Playwright container"}
    ]


@pytest.mark.asyncio
async def test_consume_streaming_tool_result_ignores_non_json_intermediate_items():
    events: list[dict] = []
    result = await _consume_streaming_tool_result(
        tool_name="computer_call",
        tool_call_id="call_1",
        item_id="item_1",
        stream=_ToolItemStream(
            items=[
                {"type": "agent.event", "headline": "Preparing browser"},
                TextContent(text="done"),
            ]
        ),
        event_handler=events.append,
    )

    assert isinstance(result, TextContent)
    assert result.text == "done"
    assert events == []


@pytest.mark.asyncio
async def test_next_retries_after_openai_api_error(monkeypatch):
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.asyncio.sleep",
        _fake_sleep,
    )

    client = _FakeOpenAIClient(
        outcomes=[
            _make_api_error("temporary failure"),
            _FakeResponse(response_id="resp_ok"),
        ]
    )

    adapter = OpenAIResponsesAdapter(client=client, max_retries=3, mode="request")
    context = adapter.create_session()
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
    )

    assert result == ""
    assert client.responses.calls == 2
    assert sleep_calls == [1.0]


@pytest.mark.asyncio
async def test_next_retries_after_stream_iterator_api_error(monkeypatch):
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.asyncio.sleep",
        _fake_sleep,
    )

    client = _FakeOpenAIClient(
        outcomes=[
            _FailingStream(error=_make_api_error("stream dropped")),
            _CompletedStream(
                event=_FakeCompletedEvent(
                    response=_FakeResponse(response_id="resp_stream")
                )
            ),
        ]
    )

    adapter = OpenAIResponsesAdapter(client=client, max_retries=3, mode="request")
    context = adapter.create_session()
    context.append_user_message("hello")
    stream_events: list[dict] = []

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[],
        event_handler=stream_events.append,
    )

    assert result == ""
    assert client.responses.calls == 2
    assert sleep_calls == [1.0]
    assert [event["type"] for event in stream_events] == ["response.completed"]


@pytest.mark.asyncio
async def test_next_raises_after_retry_budget_is_exhausted(monkeypatch):
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.asyncio.sleep",
        _fake_sleep,
    )

    client = _FakeOpenAIClient(
        outcomes=[
            _make_api_error("first failure"),
            _make_api_error("second failure"),
        ]
    )

    adapter = OpenAIResponsesAdapter(client=client, max_retries=1, mode="request")
    context = adapter.create_session()
    context.append_user_message("hello")

    with pytest.raises(RoomException, match="Error from OpenAI"):
        await adapter.next(
            context=context,
            room=_FakeRoom(),
            toolkits=[],
        )

    assert client.responses.calls == 2
    assert sleep_calls == [1.0]
