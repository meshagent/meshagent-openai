import json
import pytest

from meshagent.api.messaging import JsonContent, TextContent
from meshagent.openai.tools.completions_adapter import (
    OpenAICompletionsAdapter,
    _consume_streaming_tool_result,
)
from meshagent.tools import FunctionTool, Toolkit


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


class _StreamingTool(FunctionTool):
    def __init__(self):
        super().__init__(
            name="stream_tool",
            input_schema={"type": "object", "properties": {}, "required": []},
            description="streaming test tool",
        )

    async def execute(self, context, **kwargs):
        del context
        del kwargs

        async def _run():
            yield JsonContent(json={"type": "agent.event", "headline": "working"})
            yield TextContent(text="tool-output")

        return _run()


class _FakeToolFunction:
    def __init__(self, *, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments)


class _FakeToolCall:
    def __init__(self, *, tool_call_id: str, name: str, arguments: dict):
        self.id = tool_call_id
        self.function = _FakeToolFunction(name=name, arguments=arguments)


class _FakeMessage:
    def __init__(self, *, tool_calls=None, content=None):
        self.tool_calls = tool_calls
        self.content = content

    def to_dict(self) -> dict:
        return {"tool_calls": self.tool_calls, "content": self.content}


class _FakeChoice:
    def __init__(self, *, message: _FakeMessage):
        self.message = message


class _FakeChatCompletion:
    def __init__(self, *, message: _FakeMessage, usage: dict | None = None):
        self.choices = [_FakeChoice(message=message)]
        self.usage = usage


class _FakeChatCompletionsClient:
    def __init__(self, *, responses: list[_FakeChatCompletion]):
        self._responses = responses.copy()

    async def create(self, **kwargs):
        del kwargs
        if len(self._responses) == 0:
            raise AssertionError("unexpected extra chat completion request")
        return self._responses.pop(0)


class _FakeChatClient:
    def __init__(self, *, responses: list[_FakeChatCompletion]):
        self.completions = _FakeChatCompletionsClient(responses=responses)


class _FakeOpenAIClient:
    def __init__(self, *, responses: list[_FakeChatCompletion]):
        self.chat = _FakeChatClient(responses=responses)


class _ToolItemStream:
    def __init__(self, *, items: list[object]):
        self._items = items

    def __aiter__(self):
        return self._run()

    async def _run(self):
        for item in self._items:
            yield item


@pytest.mark.asyncio
async def test_consume_streaming_tool_result_emits_intermediate_json_chunk_events():
    events: list[dict] = []
    result = await _consume_streaming_tool_result(
        stream=_ToolItemStream(
            items=[
                JsonContent(json={"type": "agent.event", "headline": "starting"}),
                TextContent(text="done"),
            ]
        ),
        event_handler=events.append,
    )

    assert events == [{"type": "agent.event", "headline": "starting"}]
    assert isinstance(result, TextContent)
    assert result.text == "done"


@pytest.mark.asyncio
async def test_consume_streaming_tool_result_uses_final_item_as_result():
    events: list[dict] = []
    result = await _consume_streaming_tool_result(
        stream=_ToolItemStream(
            items=[
                JsonContent(json={"progress": 50}),
                JsonContent(json={"answer": "ok"}),
            ]
        ),
        event_handler=events.append,
    )

    assert events == [{"progress": 50}]
    assert isinstance(result, JsonContent)
    assert result.json == {"answer": "ok"}


@pytest.mark.asyncio
async def test_next_consumes_streaming_tool_events_and_uses_final_item_result():
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=_FakeOpenAIClient(
            responses=[
                _FakeChatCompletion(
                    message=_FakeMessage(
                        tool_calls=[
                            _FakeToolCall(
                                tool_call_id="call_1",
                                name="stream_tool",
                                arguments={},
                            )
                        ],
                        content=None,
                    ),
                    usage={"prompt_tokens": 5, "completion_tokens": 1},
                ),
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content="done"),
                    usage={"prompt_tokens": 2, "completion_tokens": 3},
                ),
            ]
        ),
    )
    context = adapter.create_session()
    context.append_user_message("run tool")
    events: list[dict] = []

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[Toolkit(name="tools", tools=[_StreamingTool()])],
        event_handler=events.append,
    )

    assert result == "done"
    assert context.turn_count == 1
    assert context.usage == {"input_tokens": 7.0, "output_tokens": 4.0}
    assert events == [{"type": "agent.event", "headline": "working"}]


@pytest.mark.asyncio
async def test_next_tracks_usage_for_single_completion_response():
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=_FakeOpenAIClient(
            responses=[
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content="done"),
                    usage={
                        "prompt_tokens": 6,
                        "completion_tokens": 2,
                        "reasoning_tokens": 1,
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

    assert result == "done"
    assert context.turn_count == 1
    assert context.metadata["last_response_usage"]["prompt_tokens"] == 6
    assert context.usage == {
        "input_tokens": 6.0,
        "output_tokens": 2.0,
        "reasoning_tokens": 1.0,
    }
