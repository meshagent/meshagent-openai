import asyncio
import copy
import json
import logging
from typing import Any

import pytest

from meshagent.agents.messages import (
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AgentToolCallArgumentsDelta,
    AgentToolCallEnded,
    AgentToolCallStarted,
    AgentTextContentDelta,
    AgentTextContentEnded,
    ToolChoice,
)
from meshagent.api import RoomException, ToolContentSpec
from meshagent.api.messaging import FileContent, JsonContent, TextContent
from meshagent.agents.context import SessionUsage
import meshagent.openai.tools.completions_adapter as completions_adapter_module
from meshagent.openai.tools.completions_adapter import (
    OpenAICompletionsAdapter,
    OpenAICompletionsToolResponseAdapter,
    _consume_streaming_tool_result,
)
from meshagent.tools import ContentTool, FunctionTool, Toolkit, ToolContext


def test_list_models_advertises_attachment_capabilities() -> None:
    model = OpenAICompletionsAdapter(model="gpt-4o").list_models()[0]

    assert model.supports_attachments is True
    assert "image/*" in model.accepts
    assert "application/xhtml+xml" in model.accepts


def test_openai_completions_adapter_preserves_none_model_branch() -> None:
    adapter = OpenAICompletionsAdapter(model=None)

    assert adapter.default_model() is None
    assert adapter.list_models()[0].name is None
    with pytest.raises(
        AttributeError,
        match="'NoneType' object has no attribute 'startswith'",
    ):
        adapter.create_session()
    assert adapter.with_runtime_api_key(api_key="runtime-key")._model is None


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


def test_make_agent_event_reader_accumulates_streamed_text_for_restore() -> None:
    adapter = OpenAICompletionsAdapter(model="gpt-4o-mini", client=object())
    context = adapter.create_session()
    restored_messages: list[dict[str, Any]] = []
    reader = adapter.make_agent_event_reader(emit_message=restored_messages.append)

    reader.consume(
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text="Hi",
        )
    )
    reader.consume(
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text=" there",
        )
    )
    reader.consume(
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text="Hi there",
        )
    )
    reader.consume(
        AgentTextContentEnded(
            type=AGENT_EVENT_TEXT_CONTENT_ENDED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
        )
    )
    adapter.restore_context_messages(context=context, messages=restored_messages)

    assert context.messages == [{"role": "assistant", "content": "Hi there"}]


def test_session_context_appends_data_url_text_file_as_text_note() -> None:
    adapter = OpenAICompletionsAdapter(model="gpt-4o-mini", client=object())
    context = adapter.create_session()

    message = context.append_file_url(
        url="data:text/plain;base64,aGVsbG8=", filename="note.txt"
    )

    assert message == {
        "role": "user",
        "content": "attached file note.txt (text/plain):\nhello",
    }


def test_session_context_appends_data_url_image_as_image_url() -> None:
    adapter = OpenAICompletionsAdapter(model="gpt-4o-mini", client=object())
    context = adapter.create_session()

    message = context.append_file_url(
        url="data:image/png;base64,cG5n", filename="image.png"
    )

    assert message == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,cG5n"},
            }
        ],
    }


def test_session_context_replaces_unsupported_data_url_file_with_note() -> None:
    adapter = OpenAICompletionsAdapter(model="gpt-4o-mini", client=object())
    context = adapter.create_session()

    message = context.append_file_url(
        url="data:application/octet-stream;base64,YmxvYg==", filename="blob.bin"
    )

    assert message == {
        "role": "user",
        "content": "the user attached blob.bin with unsupported mime type application/octet-stream",
    }


@pytest.mark.parametrize(
    ("namespace", "toolkit", "tool", "arguments", "expected_name"),
    [
        ("meshagent", "toolkit", "custom_tool", {"value": 1}, "toolkit_custom_tool"),
        (
            "openai.responses",
            "openai",
            "shell",
            {"action": {"commands": ["pwd"]}},
            "shell_call",
        ),
        (
            "openai.responses",
            "openai",
            "apply_patch",
            {"operation": {"type": "update_file", "path": "report.py", "diff": "@@"}},
            "apply_patch_call",
        ),
        (
            "openai.responses",
            "openai",
            "web_search",
            {"query": "meshagent"},
            "web_search_call",
        ),
        (
            "openai.responses",
            "server",
            "search",
            {"query": "meshagent"},
            "mcp_search",
        ),
        (
            "openai.responses",
            "server",
            "list_tools",
            {},
            "mcp_list_tools",
        ),
    ],
)
def test_make_agent_event_reader_restores_tool_lifecycle_as_chat_tool_calls(
    namespace: str,
    toolkit: str,
    tool: str,
    arguments: dict[str, object],
    expected_name: str,
) -> None:
    adapter = OpenAICompletionsAdapter(model="gpt-4o-mini", client=object())
    context = adapter.create_session()
    restored_messages: list[dict[str, Any]] = []
    reader = adapter.make_agent_event_reader(emit_message=restored_messages.append)
    serialized_arguments = json.dumps(arguments, separators=(",", ":"))
    split_at = max(1, len(serialized_arguments) // 2)

    for delta in (
        serialized_arguments[:split_at],
        serialized_arguments[split_at:],
    ):
        reader.consume(
            AgentToolCallArgumentsDelta(
                type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
                thread_id="thread-1",
                turn_id="turn-1",
                item_id="tool-1",
                namespace=namespace,
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
            namespace=namespace,
            call_id="call-1",
            toolkit=toolkit,
            tool=tool,
            arguments=arguments,
        )
    )
    reader.consume(
        AgentToolCallEnded(
            type=AGENT_EVENT_TOOL_CALL_ENDED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="tool-1",
            namespace=namespace,
            call_id="call-1",
            toolkit=toolkit,
            tool=tool,
            result=TextContent(text="tool result"),
        )
    )
    reader.finalize()
    adapter.restore_context_messages(context=context, messages=restored_messages)

    assistant_message = context.messages[0]
    assert assistant_message["role"] == "assistant"
    restored_call = assistant_message["tool_calls"][0]
    assert restored_call["id"] == "call-1"
    assert restored_call["type"] == "function"
    assert restored_call["function"] == {
        "name": expected_name,
        "arguments": serialized_arguments,
    }
    assert context.messages[1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "tool result",
    }


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


def test_store_usage_publishes_otel_usage_metrics(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []

    def _fake_track_otel_usage_metrics(
        *,
        model: str,
        provider: str,
        tokens: dict[str, float],
        annotations: dict[str, str] | None = None,
    ) -> None:
        calls.append(
            {
                "model": model,
                "provider": provider,
                "tokens": tokens,
                "annotations": annotations,
            }
        )

    monkeypatch.setattr(
        completions_adapter_module,
        "track_otel_usage_metrics",
        _fake_track_otel_usage_metrics,
    )
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=object(),
        annotations={"Env": "prod"},
    )
    context = adapter.create_session()

    adapter._store_usage(
        context=context,
        usage={"prompt_tokens": 6, "completion_tokens": 2},
        model="gpt-4o-mini",
    )

    assert calls == [
        {
            "model": "gpt-4o-mini",
            "provider": "openai",
            "tokens": {"input_tokens": 6.0, "output_tokens": 2.0},
            "annotations": {"env": "prod"},
        }
    ]
    assert context.last_usage == SessionUsage(
        model="gpt-4o-mini",
        usage={"input_tokens": 6.0, "output_tokens": 2.0},
        context_window_used=8,
    )


def test_store_usage_splits_cached_prompt_tokens() -> None:
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=object(),
    )
    context = adapter.create_session()

    adapter._store_usage(
        context=context,
        usage={
            "prompt_tokens": 5084,
            "prompt_tokens_details": {"cached_tokens": 4864},
            "completion_tokens": 2,
            "total_tokens": 5086,
        },
        model="gpt-4o-mini",
    )

    assert context.last_usage == SessionUsage(
        model="gpt-4o-mini",
        usage={
            "cached_tokens": 4864.0,
            "input_tokens": 220.0,
            "output_tokens": 2.0,
            "total_tokens": 5086.0,
        },
        context_window_used=5086,
    )


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


class _ContextTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="context test tool",
        )
        self.contexts: list[ToolContext] = []

    async def execute(self, context: ToolContext, **kwargs):
        del kwargs
        self.contexts.append(context)
        return {"ok": True}


class _FailingTool(FunctionTool):
    def __init__(self, name: str, error: Exception):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="failing test tool",
        )
        self._error = error

    async def execute(self, context, **kwargs):
        del context
        del kwargs
        raise self._error


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


class _FakeDisplayMessage(_FakeMessage):
    def __str__(self) -> str:
        return "message-object"


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


class _AttrDict(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as ex:
            raise AttributeError(name) from ex


@pytest.mark.asyncio
async def test_openai_completions_tool_response_adapter_truncates_json_output() -> None:
    adapter = OpenAICompletionsToolResponseAdapter(
        max_tool_call_length=18,
        max_tool_call_lines=4,
    )

    output = await adapter.to_plain_text(
        response=JsonContent(json={"message": "x" * 40}),
    )

    assert "The tool call returned too much data and was truncated." in output


def test_openai_completions_tool_response_adapter_constructor_validation() -> None:
    with pytest.raises(ValueError, match="max_tool_call_length must be greater than 0"):
        OpenAICompletionsToolResponseAdapter(max_tool_call_length=0)

    with pytest.raises(ValueError, match="max_tool_call_lines must be greater than 0"):
        OpenAICompletionsToolResponseAdapter(max_tool_call_lines=0)


@pytest.mark.asyncio
async def test_openai_completions_tool_response_adapter_truncates_utf8_file_output() -> (
    None
):
    adapter = OpenAICompletionsToolResponseAdapter(
        max_tool_call_length=16,
        max_tool_call_lines=2,
    )

    output = await adapter.to_plain_text(
        response=FileContent(
            data=b"line1\nline2\nline3\nline4",
            name="README",
            mime_type="application/octet-stream",
        ),
    )

    assert "line1\nline2" in output
    assert "line3" not in output
    assert "The tool call returned too much data and was truncated." in output


@pytest.mark.asyncio
async def test_openai_completions_tool_response_adapter_raw_outputs_match_python() -> (
    None
):
    adapter = OpenAICompletionsToolResponseAdapter(
        max_tool_call_length=1024,
        max_tool_call_lines=20,
    )

    raw_dict_text = await adapter.to_plain_text(  # type: ignore[arg-type]
        response={"hello": "世界"}
    )
    assert json.loads(raw_dict_text) == {"hello": "世界"}
    assert "\\u4e16\\u754c" in raw_dict_text
    assert await adapter.to_plain_text(response="raw text") == "raw text"  # type: ignore[arg-type]
    assert await adapter.to_plain_text(response=None) == "ok"  # type: ignore[arg-type]

    assert await adapter.create_messages(
        context=None,  # type: ignore[arg-type]
        tool_call=_AttrDict(id="tool-call-dict"),
        response={"hello": "世界"},  # type: ignore[arg-type]
    ) == [
        {
            "role": "tool",
            "content": json.dumps({"hello": "世界"}),
            "tool_call_id": "tool-call-dict",
        }
    ]
    assert await adapter.create_messages(
        context=None,  # type: ignore[arg-type]
        tool_call=_AttrDict(id="tool-call-string"),
        response="raw text",  # type: ignore[arg-type]
    ) == [
        {
            "role": "tool",
            "content": "raw text",
            "tool_call_id": "tool-call-string",
        }
    ]
    assert await adapter.create_messages(
        context=None,  # type: ignore[arg-type]
        tool_call=_AttrDict(id="tool-call-none"),
        response=None,  # type: ignore[arg-type]
    ) == [
        {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "tool-call-none",
        }
    ]


def test_openai_completions_adapter_passes_through_tool_truncation_limits() -> None:
    adapter = OpenAICompletionsAdapter(
        max_tool_call_length=321,
        max_tool_call_lines=9,
    )

    tool_adapter = adapter._make_tool_response_adapter()

    assert tool_adapter.max_tool_call_length == 321
    assert tool_adapter.max_tool_call_lines == 9


@pytest.mark.asyncio
async def test_openai_completions_adapter_passes_base_url_to_get_client(monkeypatch):
    fake_client = _FakeOpenAIClient(
        responses=[
            _FakeChatCompletion(
                message=_FakeMessage(tool_calls=None, content="done"),
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )
        ]
    )
    call_args: dict[str, object] = {}

    def _fake_get_client(
        *, base_url=None, http_client=None, session=None, api_key=None, user_agent=None
    ):
        call_args["base_url"] = base_url
        call_args["http_client"] = http_client
        call_args["session"] = session
        call_args["api_key"] = api_key
        call_args["user_agent"] = user_agent
        return fake_client

    monkeypatch.setattr(
        "meshagent.openai.tools.completions_adapter.get_client",
        _fake_get_client,
    )

    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        base_url="https://example.test/v1",
        api_key="test-token",
        user_agent="custom-app/1.0",
    )
    context = adapter.create_session()
    context.append_user_message("hello")

    result = await adapter.create_response(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[],
    )

    assert result == "done"
    assert call_args["base_url"] == "https://example.test/v1"
    assert call_args["http_client"] is None
    assert call_args["user_agent"] == "custom-app/1.0"
    assert call_args["session"] is None
    assert call_args["api_key"] == "test-token"


@pytest.mark.asyncio
async def test_openai_completions_adapter_publishes_text_events_for_restore() -> None:
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=_FakeOpenAIClient(
            responses=[
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content="done"),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )
            ]
        ),
    )
    context = adapter.create_session()
    context.metadata["thread_id"] = "thread-1"
    context.metadata["turn_id"] = "turn-1"
    context.append_user_message("hello")
    published: list[object] = []

    result = await adapter.create_response(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[],
        event_handler=published.append,
    )

    assert result == "done"
    assert [
        type(message)
        for message in published
        if isinstance(message, (AgentTextContentDelta, AgentTextContentEnded))
    ] == [AgentTextContentDelta, AgentTextContentEnded]
    delta = next(
        message for message in published if isinstance(message, AgentTextContentDelta)
    )
    assert delta.thread_id == "thread-1"
    assert delta.turn_id == "turn-1"
    assert delta.text == "done"


@pytest.mark.asyncio
async def test_openai_completions_adapter_blank_schema_output_matches_python_bug() -> (
    None
):
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=_FakeOpenAIClient(
            responses=[
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content=" \n\t"),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )
            ]
        ),
    )
    context = adapter.create_session()
    context.append_user_message("hello")

    with pytest.raises(
        UnboundLocalError,
        match="cannot access local variable 'full_response'",
    ):
        await adapter.create_response(
            context=context,
            caller=_FakeRoom().local_participant,
            toolkits=[],
            output_schema={"type": "object", "additionalProperties": True},
        )


@pytest.mark.asyncio
async def test_openai_completions_adapter_schema_output_json_errors_match_python() -> (
    None
):
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=_FakeOpenAIClient(
            responses=[
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content="[1,]"),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                ),
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content='{"answer": 1,}'),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                ),
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content="[1 2]"),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                ),
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content="1 2"),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                ),
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content=r'["bad\uZZZZ"]'),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                ),
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content=r'["bad\x01"]'),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                ),
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content='["unterminated]'),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                ),
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content='{"answer": }'),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                ),
            ]
        ),
    )
    caller = _FakeRoom().local_participant
    output_schema = {"type": "object", "additionalProperties": True}

    context = adapter.create_session()
    context.append_user_message("array trailing comma")
    with pytest.raises(
        json.JSONDecodeError,
        match=(
            r"Illegal trailing comma before end of array: "
            r"line 1 column 3 \(char 2\)"
        ),
    ):
        await adapter.create_response(
            context=context,
            caller=caller,
            toolkits=[],
            output_schema=output_schema,
        )

    context = adapter.create_session()
    context.append_user_message("object trailing comma")
    with pytest.raises(
        json.JSONDecodeError,
        match=(
            r"Illegal trailing comma before end of object: "
            r"line 1 column 13 \(char 12\)"
        ),
    ):
        await adapter.create_response(
            context=context,
            caller=caller,
            toolkits=[],
            output_schema=output_schema,
        )

    context = adapter.create_session()
    context.append_user_message("missing array comma")
    with pytest.raises(
        json.JSONDecodeError,
        match=r"Expecting ',' delimiter: line 1 column 4 \(char 3\)",
    ):
        await adapter.create_response(
            context=context,
            caller=caller,
            toolkits=[],
            output_schema=output_schema,
        )

    context = adapter.create_session()
    context.append_user_message("extra data")
    with pytest.raises(
        json.JSONDecodeError,
        match=r"Extra data: line 1 column 3 \(char 2\)",
    ):
        await adapter.create_response(
            context=context,
            caller=caller,
            toolkits=[],
            output_schema=output_schema,
        )

    context = adapter.create_session()
    context.append_user_message("invalid unicode escape")
    with pytest.raises(
        json.JSONDecodeError,
        match=r"Invalid \\uXXXX escape: line 1 column 7 \(char 6\)",
    ):
        await adapter.create_response(
            context=context,
            caller=caller,
            toolkits=[],
            output_schema=output_schema,
        )

    context = adapter.create_session()
    context.append_user_message("invalid escape")
    with pytest.raises(
        json.JSONDecodeError,
        match=r"Invalid \\escape: line 1 column 6 \(char 5\)",
    ):
        await adapter.create_response(
            context=context,
            caller=caller,
            toolkits=[],
            output_schema=output_schema,
        )

    context = adapter.create_session()
    context.append_user_message("unterminated string")
    with pytest.raises(
        json.JSONDecodeError,
        match=r"Unterminated string starting at: line 1 column 2 \(char 1\)",
    ):
        await adapter.create_response(
            context=context,
            caller=caller,
            toolkits=[],
            output_schema=output_schema,
        )

    context = adapter.create_session()
    context.append_user_message("missing object value")
    with pytest.raises(
        json.JSONDecodeError,
        match=r"Expecting value: line 1 column 12 \(char 11\)",
    ):
        await adapter.create_response(
            context=context,
            caller=caller,
            toolkits=[],
            output_schema=output_schema,
        )


@pytest.mark.asyncio
async def test_openai_completions_adapter_unexpected_message_uses_message_str() -> None:
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=_FakeOpenAIClient(
            responses=[
                _FakeChatCompletion(
                    message=_FakeDisplayMessage(tool_calls=None, content=None),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )
            ]
        ),
    )
    context = adapter.create_session()
    context.append_user_message("hello")

    with pytest.raises(
        RoomException,
        match=r"Unexpected response from OpenAI message-object",
    ):
        await adapter.create_response(
            context=context,
            caller=_FakeRoom().local_participant,
            toolkits=[],
        )


@pytest.mark.asyncio
async def test_openai_completions_adapter_publishes_tool_events_for_restore() -> None:
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
    context.metadata["thread_id"] = "thread-1"
    context.metadata["turn_id"] = "turn-1"
    context.append_user_message("run tool")
    published: list[object] = []

    result = await adapter.create_response(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[Toolkit(name="tools", tools=[_StreamingTool()])],
        event_handler=published.append,
    )

    assert result == "done"
    started = next(
        message for message in published if isinstance(message, AgentToolCallStarted)
    )
    ended = next(
        message for message in published if isinstance(message, AgentToolCallEnded)
    )
    assert started.thread_id == "thread-1"
    assert started.turn_id == "turn-1"
    assert started.item_id == "call_1"
    assert started.tool == "stream_tool"
    assert ended.thread_id == "thread-1"
    assert ended.turn_id == "turn-1"
    assert ended.item_id == "call_1"
    assert isinstance(ended.result, TextContent)
    assert ended.result.text == "tool-output"


def test_openai_completions_adapter_reads_base_url_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.test/v1")

    adapter = OpenAICompletionsAdapter(model="gpt-4o-mini")

    assert adapter._base_url == "https://env.example.test/v1"


def test_openai_completions_tool_choice_rejects_content_tools_like_python() -> None:
    adapter = OpenAICompletionsAdapter(model="gpt-4o-mini", client=object())
    toolkit = Toolkit(
        name="content",
        tools=[
            ContentTool(
                name="content_tool",
                input_spec=ToolContentSpec(types=["json"]),
                output_spec=ToolContentSpec(types=["json"]),
            )
        ],
    )

    with pytest.raises(
        completions_adapter_module.RoomException,
        match="tool_choice is not supported for ContentTool",
    ):
        adapter._resolve_tool_choice(
            toolkits=[toolkit],
            tool_choice=ToolChoice(
                toolkit_name="content",
                tool_name="content_tool",
            ),
        )


def test_openai_completions_adapter_with_runtime_api_key_returns_bound_clone(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-token")
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        max_tool_call_length=123,
        max_tool_call_lines=7,
    )

    bound = adapter.with_runtime_api_key(api_key="runtime-token")

    assert bound is not adapter
    assert bound._api_key == "runtime-token"
    assert bound._max_tool_call_length == 123
    assert bound._max_tool_call_lines == 7


def test_openai_completions_adapter_with_runtime_api_key_keeps_explicit_api_key() -> (
    None
):
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        api_key="configured-token",
    )

    assert adapter.with_runtime_api_key(api_key="runtime-token") is adapter


def test_openai_completions_adapter_with_runtime_api_key_keeps_explicit_client() -> (
    None
):
    adapter = OpenAICompletionsAdapter(model="gpt-4o-mini", client=object())

    assert adapter.with_runtime_api_key(api_key="runtime-token") is adapter


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

    result = await adapter.create_response(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[Toolkit(name="tools", tools=[_StreamingTool()])],
        event_handler=events.append,
    )

    assert result == "done"
    assert context.turn_count == 1
    assert context.last_usage == SessionUsage(
        model="gpt-4o-mini",
        usage={"input_tokens": 2.0, "output_tokens": 3.0},
        context_window_used=5,
    )
    assert events == [{"type": "agent.event", "headline": "working"}]


@pytest.mark.asyncio
async def test_next_logs_non_room_tool_exception_and_sends_fallback_result(caplog):
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=_FakeOpenAIClient(
            responses=[
                _FakeChatCompletion(
                    message=_FakeMessage(
                        tool_calls=[
                            _FakeToolCall(
                                tool_call_id="call_1",
                                name="fail_tool",
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

    caplog.set_level(logging.ERROR, logger="openai_agent")
    result = await adapter.create_response(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[
            Toolkit(
                name="tools",
                tools=[_FailingTool("fail_tool", ValueError("non-room boom"))],
            )
        ],
    )

    assert result == "done"
    logged_messages = [
        record.message
        for record in caplog.records
        if record.name == "openai_agent" and record.levelno == logging.ERROR
    ]
    assert len(logged_messages) == 1
    assert "unable to complete tool call" in logged_messages[0]
    assert "non-room boom" not in logged_messages[0]
    second_request_messages = adapter._client.chat.completions.create_kwargs[1][
        "messages"
    ]
    tool_message = next(
        message
        for message in second_request_messages
        if isinstance(message, dict) and message.get("role") == "tool"
    )
    assert tool_message == {
        "role": "tool",
        "content": json.dumps({"error": "unable to complete tool call: non-room boom"}),
        "tool_call_id": "call_1",
    }


@pytest.mark.asyncio
async def test_next_accumulates_cached_usage_across_tool_loop_calls() -> None:
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
                    usage={
                        "prompt_tokens": 5084,
                        "prompt_tokens_details": {"cached_tokens": 4864},
                        "completion_tokens": 2,
                        "total_tokens": 5086,
                    },
                ),
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content="done"),
                    usage={
                        "prompt_tokens": 120,
                        "prompt_tokens_details": {"cached_tokens": 64},
                        "completion_tokens": 3,
                        "total_tokens": 123,
                    },
                ),
            ]
        ),
    )
    context = adapter.create_session()
    context.append_user_message("run tool")

    result = await adapter.create_response(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[Toolkit(name="tools", tools=[_StreamingTool()])],
    )

    assert result == "done"
    assert context.last_usage == SessionUsage(
        model="gpt-4o-mini",
        usage={
            "cached_tokens": 64.0,
            "input_tokens": 56.0,
            "output_tokens": 3.0,
            "total_tokens": 123.0,
        },
        context_window_used=123,
    )


@pytest.mark.asyncio
async def test_next_passes_typed_tool_context_without_agent_lifecycle_ids() -> None:
    tool = _ContextTool("context_tool")
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=_FakeOpenAIClient(
            responses=[
                _FakeChatCompletion(
                    message=_FakeMessage(
                        tool_calls=[
                            _FakeToolCall(
                                tool_call_id="call_1",
                                name="context_tool",
                                arguments={},
                            )
                        ],
                        content=None,
                    )
                ),
                _FakeChatCompletion(
                    message=_FakeMessage(tool_calls=None, content="done"),
                ),
            ]
        ),
    )
    context = adapter.create_session()
    context.append_user_message("run tool")
    context.metadata["thread_id"] = "thread-1"
    context.metadata["turn_id"] = "turn-1"

    result = await adapter.create_response(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[Toolkit(name="tools", tools=[tool])],
    )

    assert result == "done"
    assert len(tool.contexts) == 1
    tool_context = tool.contexts[0]
    assert type(tool_context) is ToolContext
    assert tool_context.caller.id == _FakeRoom().local_participant.id


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

    result = await adapter.create_response(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[],
    )

    assert result == "done"
    assert context.turn_count == 1
    # reasoning_tokens are a subset of completion_tokens, so they are not
    # tracked as a separate billable component when the aggregate is present.
    assert context.last_usage == SessionUsage(
        model="gpt-4o-mini",
        usage={
            "input_tokens": 6.0,
            "output_tokens": 2.0,
        },
        context_window_used=8,
    )


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

    result = await adapter.create_response(
        context=context,
        caller=_FakeRoom().local_participant,
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
async def test_next_newline_json_validation_error_returns_last_line_like_python() -> (
    None
):
    adapter = OpenAICompletionsAdapter(
        model="gpt-4o-mini",
        client=_FakeOpenAIClient(
            responses=[
                _FakeChatCompletion(
                    message=_FakeMessage(
                        tool_calls=None,
                        content='{"answer": 1}\n{"wrong": true}',
                    ),
                )
            ]
        ),
    )
    context = adapter.create_session()
    context.append_user_message("return json")

    result = await adapter.create_response(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[],
        output_schema={
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "integer"}},
        },
    )

    assert result == {"wrong": True}
    assert context.messages[-1]["role"] == "user"
    assert context.messages[-1]["content"].startswith(
        "encountered a validation error with the output: 'answer' is a required property"
    )
    assert "On instance:\n    {'wrong': True}" in context.messages[-1]["content"]


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
        adapter.create_response(
            context=context,
            caller=_FakeRoom().local_participant,
            toolkits=[Toolkit(name="storage", tools=[blocking_tool])],
        )
    )

    await asyncio.wait_for(blocking_tool.started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert context.messages == [{"role": "user", "content": "run tool"}]
