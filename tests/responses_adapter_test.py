import asyncio
import httpx
import pytest
from openai import APIError

from meshagent.api import RoomException
from meshagent.api.messaging import JsonContent, TextContent
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


class _FakeResponse:
    def __init__(self, *, response_id: str):
        self.id = response_id
        self.output = []
        self.usage = None

    def to_dict(self) -> dict:
        return {"id": self.id, "output": []}


class _FakeCompletedEvent:
    def __init__(self, *, response: _FakeResponse):
        self.type = "response.completed"
        self.response = response

    def model_dump(self, *, mode: str = "json") -> dict:
        del mode
        return {
            "type": self.type,
            "response": {"id": self.response.id, "output": []},
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

    async def create(self, **kwargs):
        del kwargs
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


@pytest.mark.asyncio
async def test_create_session_returns_openai_responses_session_context():
    adapter = OpenAIResponsesAdapter()
    context = adapter.create_session()
    assert isinstance(context, OpenAIResponsesSessionContext)


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
            event=_FakeCompletedEvent(response=_FakeResponse(response_id="resp_ws"))
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

    adapter = OpenAIResponsesAdapter(client=client, max_retries=3)
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

    adapter = OpenAIResponsesAdapter(client=client, max_retries=3)
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

    adapter = OpenAIResponsesAdapter(client=client, max_retries=1)
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
