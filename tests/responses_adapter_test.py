import httpx
import pytest
from openai import APIError

from meshagent.api import RoomException
from meshagent.openai.tools.responses_adapter import OpenAIResponsesAdapter


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


def _make_api_error(message: str) -> APIError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return APIError(message, request=request, body=None)


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
    context = adapter.create_chat_context()
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
    context = adapter.create_chat_context()
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
    context = adapter.create_chat_context()
    context.append_user_message("hello")

    with pytest.raises(RoomException, match="Error from OpenAI"):
        await adapter.next(
            context=context,
            room=_FakeRoom(),
            toolkits=[],
        )

    assert client.responses.calls == 2
    assert sleep_calls == [1.0]
