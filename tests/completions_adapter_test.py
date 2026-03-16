import asyncio
import json
import copy
from typing import Any

import pytest

from meshagent.api.messaging import FileContent, JsonContent, TextContent
from meshagent.openai.tools.completions_adapter import (
    OpenAICompletionsAdapter,
    OpenAICompletionsToolResponseAdapter,
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


class _BlockingTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="blocking test tool",
        )
        self.started = asyncio.Event()

    async def execute(self, context, **kwargs):
        del context
        del kwargs
        self.started.set()
        await asyncio.Future()


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
        self.create_kwargs: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.create_kwargs.append(copy.deepcopy(kwargs))
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
async def test_openai_completions_tool_response_adapter_truncates_json_output() -> None:
    adapter = OpenAICompletionsToolResponseAdapter(
        max_tool_call_length=18,
        max_tool_call_lines=4,
    )

    output = await adapter.to_plain_text(
        room=_FakeRoom(),
        response=JsonContent(json={"message": "x" * 40}),
    )

    assert "The tool call returned too much data and was truncated." in output


@pytest.mark.asyncio
async def test_openai_completions_tool_response_adapter_truncates_utf8_file_output() -> (
    None
):
    adapter = OpenAICompletionsToolResponseAdapter(
        max_tool_call_length=16,
        max_tool_call_lines=2,
    )

    output = await adapter.to_plain_text(
        room=_FakeRoom(),
        response=FileContent(
            data=b"line1\nline2\nline3\nline4",
            name="README",
            mime_type="application/octet-stream",
        ),
    )

    assert "line1\nline2" in output
    assert "line3" not in output
    assert "The tool call returned too much data and was truncated." in output


def test_openai_completions_adapter_passes_through_tool_truncation_limits() -> None:
    adapter = OpenAICompletionsAdapter(
        max_tool_call_length=321,
        max_tool_call_lines=9,
    )

    tool_adapter = adapter._make_tool_response_adapter()

    assert tool_adapter.max_tool_call_length == 321
    assert tool_adapter.max_tool_call_lines == 9


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


@pytest.mark.asyncio
async def test_next_inserts_steering_messages_after_tool_results() -> None:
    client = _FakeOpenAIClient(
        responses=[
            _FakeChatCompletion(
                message=_FakeMessage(
                    tool_calls=[
                        _FakeToolCall(
                            tool_call_id="call_1",
                            name="stream_tool",
                            arguments={"path": "/tmp/example.txt"},
                        )
                    ],
                    content=None,
                )
            ),
            _FakeChatCompletion(
                message=_FakeMessage(tool_calls=None, content="done"),
            ),
        ]
    )
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=client,
    )
    context = adapter.create_session()
    context.append_user_message("run tool")
    steering_calls = 0

    async def _steer() -> bool:
        nonlocal steering_calls
        steering_calls += 1
        context.append_user_message("steer now")
        return True

    result = await adapter.next(
        context=context,
        room=_FakeRoom(),
        toolkits=[Toolkit(name="tools", tools=[_StreamingTool()])],
        steering_callback=_steer,
    )

    assert result == "done"
    assert steering_calls == 1
    assert len(client.chat.completions.create_kwargs) == 2
    second_messages = client.chat.completions.create_kwargs[1]["messages"]
    tool_messages = [
        message
        for message in second_messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert second_messages[-1] == {
        "role": "user",
        "content": "steer now",
    }


@pytest.mark.asyncio
async def test_cancellation_restores_context_during_tool_call() -> None:
    blocking_tool = _BlockingTool("write_file")
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=_FakeOpenAIClient(
            responses=[
                _FakeChatCompletion(
                    message=_FakeMessage(
                        tool_calls=[
                            _FakeToolCall(
                                tool_call_id="call_1",
                                name="write_file",
                                arguments={"path": "/tmp/example.txt"},
                            )
                        ],
                        content=None,
                    )
                )
            ]
        ),
    )
    context = adapter.create_session()
    context.append_user_message("run tool")

    task = asyncio.create_task(
        adapter.next(
            context=context,
            room=_FakeRoom(),
            toolkits=[Toolkit(name="storage", tools=[blocking_tool])],
        )
    )

    await asyncio.wait_for(blocking_tool.started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert context.messages == [{"role": "user", "content": "run tool"}]
