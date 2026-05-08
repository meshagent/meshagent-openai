import asyncio
import base64
import contextlib
import copy
import json
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
from openai.types.responses.response_function_shell_tool_call import (
    ResponseFunctionShellToolCall,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from yarl import URL

from meshagent.agents.adapter import ToolCallApprovalRequest
from meshagent.agents.messages import (
    AGENT_EVENT_CONTEXT_COMPACTED,
    AGENT_EVENT_THREAD_EVENT,
    AgentContextCompacted,
    AgentThreadEvent,
    AgentImageGenerationCompleted,
    AgentImageGenerationStarted,
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentFileContentStarted,
    AgentReasoningContentDelta,
    AgentReasoningContentEnded,
    AgentReasoningContentStarted,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallPending,
    AgentToolCallLogDelta,
    AgentToolCallEnded,
    AgentToolCallStarted,
)
from meshagent.api import RoomException
from meshagent.api.error_codes import ErrorCode
from meshagent.api.messaging import FileContent, JsonContent, TextContent
from meshagent.computers.agent import ComputerToolkit
from meshagent.computers.operator import Operator
import meshagent.openai.tools.responses_adapter as responses_adapter_module
from meshagent.openai.tools.responses_adapter import (
    DEFAULT_IMAGE_GENERATION_MODEL,
    ImageGenerationTool,
    MCPServer,
    MCPTool,
    OpenAIResponsesAdapter,
    OpenAIResponsesToolResponseAdapter,
    OpenAIResponsesSessionContext,
    ResponsesToolBundle,
    ShellTool,
    WebSearchTool,
    _consume_streaming_tool_result,
    safe_tool_name,
)
from meshagent.tools import FunctionTool, Toolkit, ToolContext


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


class _FakeImagesDataset:
    def __init__(self):
        self.save_calls: list[dict[str, object]] = []

    async def save(self, **kwargs):
        self.save_calls.append(kwargs)
        return SimpleNamespace(
            id="image-1",
            mime_type=kwargs["mime_type"],
            created_at="2026-05-05T00:00:00Z",
            created_by=kwargs["created_by"],
        )


class _NamelessParticipant:
    def __init__(self):
        self.id = "participant_2"

    def get_attribute(self, key: str):
        del key
        return None


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
        responses_adapter_module,
        "track_otel_usage_metrics",
        _fake_track_otel_usage_metrics,
    )
    adapter = OpenAIResponsesAdapter(
        model="gpt-5-mini",
        client=object(),
        annotations={"Env": "prod"},
    )
    context = adapter.create_session()

    adapter._store_usage(
        context=context,
        usage={
            "input_tokens": 11,
            "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 4},
        },
        model="gpt-5-mini",
    )

    assert calls == [
        {
            "model": "gpt-5-mini",
            "provider": "openai",
            "tokens": {
                "input_tokens": 7.0,
                "output_tokens": 7.0,
                "cached_tokens": 4.0,
            },
            "annotations": {"env": "prod"},
        }
    ]
    assert context.metadata["last_response_flattened_usage"] == {
        "input_tokens": 7.0,
        "output_tokens": 7.0,
        "cached_tokens": 4.0,
    }
    assert context.metadata["last_response_context_used_tokens"] == 18


def test_store_usage_includes_image_generation_usage() -> None:
    adapter = OpenAIResponsesAdapter(
        model="gpt-5-mini",
        client=object(),
    )
    context = adapter.create_session()
    response = _FakeResponse(
        response_id="resp_1",
        output=[
            _FakeImageGenerationOutputItem(
                type="image_generation_call",
                id="ig_1",
                usage={
                    "input_tokens": 120,
                    "input_tokens_details": {
                        "text_tokens": 30,
                        "image_tokens": 90,
                    },
                    "output_tokens": 400,
                    "total_tokens": 520,
                },
            )
        ],
        usage={
            "input_tokens": 11,
            "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 4},
            "total_tokens": 18,
        },
    )

    adapter._store_image_generation_usage_from_response(
        context=context,
        response=response,  # type: ignore[arg-type]
        model="gpt-5-mini",
    )
    adapter._store_usage(
        context=context,
        usage=response.usage,
        model="gpt-5-mini",
    )

    assert context.metadata["last_response_flattened_usage"] == {
        "input_tokens": 37.0,
        "output_tokens": 7.0,
        "cached_tokens": 4.0,
        "total_tokens": 538.0,
        "image_input_tokens": 90.0,
        "image_output_tokens": 400.0,
    }
    assert context.metadata["last_response_context_used_tokens"] == 538
    assert context.usage == context.metadata["last_response_flattened_usage"]


def test_store_usage_keeps_image_generation_usage_without_response_usage() -> None:
    adapter = OpenAIResponsesAdapter(
        model="gpt-5-mini",
        client=object(),
    )
    context = adapter.create_session()
    response = _FakeResponse(
        response_id="resp_1",
        output=[
            _FakeImageGenerationOutputItem(
                type="image_generation_call",
                id="ig_1",
                usage={
                    "input_tokens": 120,
                    "input_tokens_details": {
                        "text_tokens": 30,
                        "image_tokens": 90,
                    },
                    "output_tokens": 400,
                    "total_tokens": 520,
                },
            )
        ],
        usage=None,
    )

    adapter._store_image_generation_usage_from_response(
        context=context,
        response=response,  # type: ignore[arg-type]
        model="gpt-5-mini",
    )
    adapter._store_usage(
        context=context,
        usage=response.usage,
        model="gpt-5-mini",
    )

    assert context.metadata["last_response_flattened_usage"] == {
        "input_tokens": 30.0,
        "total_tokens": 520.0,
        "image_input_tokens": 90.0,
        "image_output_tokens": 400.0,
    }
    assert context.metadata["last_response_context_used_tokens"] == 520
    assert context.usage == context.metadata["last_response_flattened_usage"]


class _FakeRoom:
    def __init__(self):
        self.local_participant = _FakeParticipant()
        self.developer = _FakeDeveloper()


class _FakeContainerInfo:
    def __init__(self, *, container_id: str):
        self.id = container_id


class _FakeContainerExec:
    def __init__(
        self,
        *,
        stdout_chunks: list[bytes],
        stderr_chunks: list[bytes],
        exit_code: int = 0,
    ) -> None:
        self._stdout_chunks = stdout_chunks
        self._stderr_chunks = stderr_chunks
        self._result = asyncio.get_running_loop().create_future()
        self._result.set_result(exit_code)

    @property
    def result(self):
        return self._result

    async def stdout(self):
        for chunk in self._stdout_chunks:
            await asyncio.sleep(0)
            yield chunk

    async def stderr(self):
        for chunk in self._stderr_chunks:
            await asyncio.sleep(0)
            yield chunk

    async def kill(self) -> None:
        return None


class _FakeContainers:
    def __init__(self, *, exec_factory, run_error: Exception | None = None):
        self._exec_factory = exec_factory
        self._run_error = run_error
        self.run_calls: list[dict[str, object]] = []
        self.exec_calls: list[dict[str, object]] = []

    async def list(self):
        return []

    async def run(self, *, command, image, mounts, writable_root_fs, env):
        self.run_calls.append(
            {
                "command": command,
                "image": image,
                "mounts": mounts,
                "writable_root_fs": writable_root_fs,
                "env": env,
            }
        )
        if self._run_error is not None:
            raise self._run_error
        return "container-1"

    async def exec(self, *, container_id, command, tty):
        self.exec_calls.append(
            {
                "container_id": container_id,
                "command": command,
                "tty": tty,
            }
        )
        return self._exec_factory()


class _FakeContainerRoom(_FakeRoom):
    def __init__(self, *, exec_factory, run_error: Exception | None = None):
        super().__init__()
        self.containers = _FakeContainers(
            exec_factory=exec_factory,
            run_error=run_error,
        )


class _AnyArgsTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="test tool",
        )

    async def execute(self, context, **kwargs):
        del context
        return {"ok": True, "args": kwargs}


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


class _GateTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="gated test tool",
        )
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, context, **kwargs):
        del context
        del kwargs
        self.started.set()
        await self.release.wait()
        return {"ok": True, "tool": self.name}


class _CallerContextTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="caller context test tool",
        )
        self.caller_contexts: list[dict[str, object] | None] = []

    async def execute(self, context, **kwargs):
        del kwargs
        self.caller_contexts.append(context.caller_context)
        return {"ok": True}


class _FakeBrowserComputer:
    environment = "browser"
    dimensions = (1024, 768)

    def __init__(self):
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self, context=None):
        del context
        self.enter_count += 1
        return self

    async def __aexit__(self, exc_type, exc, exc_tb):
        del exc_type
        del exc
        del exc_tb
        self.exit_count += 1

    async def click(
        self,
        context,
        *,
        x: int,
        y: int,
        button: str = "left",
    ) -> None:
        del context
        self.calls.append(("click", {"x": x, "y": y, "button": button}))

    async def double_click(self, context, *, x: int, y: int) -> None:
        del context
        self.calls.append(("double_click", {"x": x, "y": y}))

    async def scroll(
        self,
        context,
        *,
        x: int,
        y: int,
        scroll_x: int,
        scroll_y: int,
    ) -> None:
        del context
        self.calls.append(
            ("scroll", {"x": x, "y": y, "scroll_x": scroll_x, "scroll_y": scroll_y})
        )

    async def type(self, context, *, text: str) -> None:
        del context
        self.calls.append(("type", {"text": text}))

    async def wait(self, context, *, ms: int = 1000) -> None:
        del context
        self.calls.append(("wait", {"ms": ms}))

    async def move(self, context, *, x: int, y: int) -> None:
        del context
        self.calls.append(("move", {"x": x, "y": y}))

    async def keypress(self, context, *, keys: list[str]) -> None:
        del context
        self.calls.append(("keypress", {"keys": keys}))

    async def drag(self, context, *, path: list[dict[str, int]]) -> None:
        del context
        self.calls.append(("drag", {"path": path}))

    async def screenshot(self, context) -> str:
        del context
        return "ZmFrZS1zY3JlZW5zaG90"

    async def get_current_url(self, context) -> str:
        del context
        return "https://example.com"

    async def goto(self, context, *, url: str) -> None:
        del context
        self.calls.append(("goto", {"url": url}))

    async def back(self, context) -> None:
        del context
        self.calls.append(("back", {}))

    async def forward(self, context) -> None:
        del context
        self.calls.append(("forward", {}))


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


class _FakeImageGenerationOutputItem(OpenAIBaseModel):
    type: str
    id: str
    usage: dict


class _InstructionalOpenAIResponsesAdapter(OpenAIResponsesAdapter):
    def get_additional_instructions(self) -> str | None:
        return "extra adapter instructions"


@pytest.mark.asyncio
async def test_openai_responses_tool_response_adapter_truncates_text_output() -> None:
    adapter = OpenAIResponsesToolResponseAdapter(
        max_tool_call_length=16,
        max_tool_call_lines=2,
    )

    output = await adapter.to_plain_text(
        response=TextContent(text="line1\nline2\nline3\nline4"),
    )

    assert "line1\nline2" in output
    assert "line3" not in output
    assert "The tool call returned too much data and was truncated." in output


@pytest.mark.asyncio
async def test_openai_responses_tool_response_adapter_truncates_text_file_output() -> (
    None
):
    adapter = OpenAIResponsesToolResponseAdapter(
        max_tool_call_length=16,
        max_tool_call_lines=2,
    )

    output = await adapter.to_plain_text(
        response=FileContent(
            data=b"line1\nline2\nline3\nline4",
            name="notes.txt",
            mime_type="text/plain",
        ),
    )

    assert "line1\nline2" in output
    assert "line3" not in output
    assert "The tool call returned too much data and was truncated." in output


def test_openai_responses_adapter_passes_through_tool_truncation_limits() -> None:
    adapter = OpenAIResponsesAdapter(
        max_tool_call_length=123,
        max_tool_call_lines=7,
    )

    tool_adapter = adapter._make_tool_response_adapter()

    assert tool_adapter.max_tool_call_length == 123
    assert tool_adapter.max_tool_call_lines == 7


def test_openai_responses_adapter_publishes_function_call_argument_deltas() -> None:
    adapter = OpenAIResponsesAdapter(client=_FakeOpenAIClient(outcomes=[]))

    assert adapter._should_publish_stream_event(  # noqa: SLF001
        event=SimpleNamespace(type="response.function_call_arguments.delta")
    )
    assert not adapter._should_publish_stream_event(  # noqa: SLF001
        event=SimpleNamespace(type="response.function_call_arguments.done")
    )


def test_openai_mcp_tool_coerces_headers_dict_to_strict_header_entries() -> None:
    server = MCPServer.model_validate(
        {
            "server_label": "docs",
            "server_url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer token"},
        }
    )

    assert server.headers is not None
    assert [header.model_dump(mode="json") for header in server.headers] == [
        {"name": "Authorization", "value": "Bearer token"}
    ]

    tool = MCPTool(servers=[server])
    definitions = tool.get_open_ai_tool_definitions()
    assert definitions[0]["headers"] == {"Authorization": "Bearer token"}


def test_image_generation_tool_defaults_to_gpt_image_2() -> None:
    tool = ImageGenerationTool()

    assert tool.model == DEFAULT_IMAGE_GENERATION_MODEL
    assert tool.get_open_ai_tool_definitions() == [
        {
            "type": "image_generation",
            "model": DEFAULT_IMAGE_GENERATION_MODEL,
            "partial_images": 1,
        }
    ]


@pytest.mark.asyncio
async def test_openai_mcp_tool_bundle_threads_tool_call_approval_handler() -> None:
    server = MCPServer.model_validate(
        {
            "server_label": "docs",
            "server_url": "https://example.com/mcp",
        }
    )
    requests = []

    async def handle_approval(
        context: ToolContext, request: ToolCallApprovalRequest
    ) -> bool:
        requests.append((context, request))
        return False

    bundle = ResponsesToolBundle(
        toolkits=[Toolkit(name="mcp", tools=[MCPTool(name="docs", servers=[server])])],
        tool_call_approval_handler=handle_approval,
    )
    tool = bundle.get_tool("docs")
    assert isinstance(tool, MCPTool)

    context = ToolContext(caller=_FakeParticipant())
    response = await tool.handle_mcp_approval_request(
        context,
        arguments='{"path":"/data"}',
        id="approval-1",
        name="read_wiki_structure",
        server_label="docs",
        type="mcp_approval_request",
    )

    assert response == {
        "type": "mcp_approval_response",
        "approve": False,
        "approval_request_id": "approval-1",
    }
    assert len(requests) == 1
    _, request = requests[0]
    assert request.item_id == "approval-1"
    assert request.toolkit == "docs"
    assert request.tool == "read_wiki_structure"
    assert request.arguments == {"path": "/data"}


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


class _FakeIncompleteEvent:
    def __init__(self, *, response: _FakeResponse):
        self.type = "response.incomplete"
        self.response = response

    def model_dump(self, *, mode: str = "json") -> dict:
        del mode
        return {
            "type": self.type,
            "response": self.response.to_dict(),
        }

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class _FakeOutputItem:
    def __init__(self, **payload: object):
        self._payload = payload
        self.type = payload["type"]

    def model_dump(self, *, mode: str = "json", **kwargs) -> dict:
        del mode
        del kwargs
        return dict(self._payload)

    def to_dict(self, *, mode: str = "json") -> dict:
        return self.model_dump(mode=mode)


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


class _FakeOutputItemDoneEvent:
    def __init__(self, *, item: OpenAIBaseModel):
        self.type = "response.output_item.done"
        self.item = item

    def model_dump(self, *, mode: str = "json") -> dict:
        del mode
        return {
            "type": self.type,
            "item": self.item.to_dict(mode="json"),
        }


class _EventStream:
    def __init__(self, *, events: list[object]):
        self._events = events
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


class _FakeResponsesClient:
    def __init__(self, *, outcomes: list[object]):
        self._outcomes = outcomes.copy()
        self.calls = 0
        self.create_kwargs: list[dict] = []

    async def create(self, **kwargs):
        self.create_kwargs.append(copy.deepcopy(kwargs))
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


class _FakeInputTokenCounter:
    def __init__(self, *, input_tokens: int):
        self.input_tokens = input_tokens
        self.calls: list[dict] = []

    async def count(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(input_tokens=self.input_tokens)


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
    def __init__(
        self,
        *,
        messages: list[_FakeLoggingWsMessage],
        exception: Exception | None = None,
        close_code: int | None = None,
    ):
        self._messages = messages.copy()
        self.sent_payloads: list[str] = []
        self.closed = False
        self._exception = exception
        self.close_code = close_code

    async def send_str(self, payload: str):
        self.sent_payloads.append(payload)

    async def receive(self):
        if len(self._messages) == 0:
            raise AssertionError("no websocket messages configured")
        return self._messages.pop(0)

    async def close(self):
        self.closed = True

    def exception(self):
        return self._exception


class _QueuedLoggingWebSocket:
    def __init__(
        self,
        *,
        exception: Exception | None = None,
        close_code: int | None = None,
    ) -> None:
        self._messages: asyncio.Queue[_FakeLoggingWsMessage] = asyncio.Queue()
        self.sent_payloads: list[str] = []
        self.closed = False
        self._exception = exception
        self.close_code = close_code

    def queue_message(self, *, message_type: aiohttp.WSMsgType, data: str = "") -> None:
        self._messages.put_nowait(
            _FakeLoggingWsMessage(
                type=message_type,
                data=data,
            )
        )

    def queue_json(self, payload: dict[str, object]) -> None:
        self.queue_message(
            message_type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(payload),
        )

    async def send_str(self, payload: str):
        self.sent_payloads.append(payload)

    async def receive(self):
        return await self._messages.get()

    async def close(self):
        self.closed = True

    def exception(self):
        return self._exception


class _SequentialClientSession:
    def __init__(self, websockets: list[_QueuedLoggingWebSocket]):
        self._websockets = websockets.copy()
        self.closed = False
        self.connect_calls = 0

    async def ws_connect(self, *args, **kwargs):
        del args
        del kwargs
        self.connect_calls += 1
        if len(self._websockets) == 0:
            raise AssertionError("no websocket configured")
        return self._websockets.pop(0)

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


class _FailingHandshakeClientSession:
    def __init__(self, *, error: aiohttp.WSServerHandshakeError):
        self._error = error
        self.closed = False

    async def ws_connect(self, *args, **kwargs):
        del args, kwargs
        raise self._error

    async def close(self):
        self.closed = True


async def _wait_for(predicate, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise asyncio.TimeoutError()
        await asyncio.sleep(0.01)


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


def _make_function_tool_call(
    *,
    item_id: str,
    tool_name: str,
    call_id: str,
    arguments: dict[str, object],
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=item_id,
        name=tool_name,
        call_id=call_id,
        arguments=json.dumps(arguments),
        type="function_call",
        status="completed",
    )


def _make_shell_tool_call(
    *,
    item_id: str,
    call_id: str,
    commands: list[str],
) -> ResponseFunctionShellToolCall:
    return ResponseFunctionShellToolCall(
        id=item_id,
        action={"commands": commands},
        call_id=call_id,
        type="shell_call",
        status="completed",
    )


@pytest.mark.asyncio
async def test_create_session_returns_openai_responses_session_context():
    adapter = OpenAIResponsesAdapter()
    context = adapter.create_session()
    assert isinstance(context, OpenAIResponsesSessionContext)


@pytest.mark.asyncio
async def test_get_openai_client_passes_optional_session(monkeypatch):
    adapter = OpenAIResponsesAdapter(
        base_url="https://example.test/v1",
        api_key="test-token",
        user_agent="custom-app/1.0",
    )
    client_session = httpx.AsyncClient()
    fake_client = object()
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
        "meshagent.openai.tools.responses_adapter.get_client",
        _fake_get_client,
    )

    try:
        client = adapter.get_openai_client(session=client_session)
    finally:
        await client_session.aclose()

    assert client is fake_client
    assert call_args["base_url"] == "https://example.test/v1"
    assert call_args["http_client"] is call_args["session"]
    assert call_args["session"] is client_session
    assert call_args["api_key"] == "test-token"
    assert call_args["user_agent"] == "custom-app/1.0"


@pytest.mark.asyncio
async def test_get_input_tokens_does_not_require_room() -> None:
    adapter = OpenAIResponsesAdapter()
    context = adapter.create_session()
    context.append_user_message("hello")
    counter = _FakeInputTokenCounter(input_tokens=42)
    adapter._client = SimpleNamespace(
        responses=SimpleNamespace(input_tokens=counter),
    )

    input_tokens = await adapter.get_input_tokens(
        context=context,
        model="gpt-4o-mini",
    )

    assert input_tokens == 42
    assert len(counter.calls) == 1
    assert counter.calls[0]["model"] == "gpt-4o-mini"
    assert counter.calls[0]["input"] == context.messages


def test_openai_responses_adapter_reads_base_url_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.test/v1")

    adapter = OpenAIResponsesAdapter()

    assert adapter._base_url == "https://env.example.test/v1"


def test_openai_responses_adapter_with_runtime_api_key_returns_bound_clone(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-token")
    adapter = OpenAIResponsesAdapter(
        model="gpt-4o",
        response_options={"metadata": {"tag": "original"}},
    )
    approval_handler = lambda request: request  # noqa: E731
    adapter.set_tool_call_approval_handler(approval_handler)

    bound = adapter.with_runtime_api_key(api_key="runtime-token")

    assert bound is not adapter
    assert bound._api_key == "runtime-token"
    assert bound._response_options == {"metadata": {"tag": "original"}}
    assert bound._response_options is not adapter._response_options
    assert bound._tool_call_approval_handler is approval_handler

    assert bound._response_options is not None
    bound._response_options["metadata"]["tag"] = "updated"
    assert adapter._response_options == {"metadata": {"tag": "original"}}


def test_openai_responses_adapter_with_runtime_api_key_keeps_explicit_api_key() -> None:
    adapter = OpenAIResponsesAdapter(api_key="configured-token")

    assert adapter.with_runtime_api_key(api_key="runtime-token") is adapter


def test_openai_responses_adapter_with_runtime_api_key_keeps_explicit_client() -> None:
    adapter = OpenAIResponsesAdapter(client=object())

    assert adapter.with_runtime_api_key(api_key="runtime-token") is adapter


def test_constructor_rejects_invalid_compaction_threshold():
    with pytest.raises(ValueError, match="compaction_threshold must be greater than 0"):
        OpenAIResponsesAdapter(compaction_threshold=0)


def test_constructor_disables_compaction_when_threshold_is_infinity():
    adapter = OpenAIResponsesAdapter(compaction_threshold=float("inf"))
    assert adapter._compaction_threshold is None


def test_context_window_size_uses_specific_gpt_5_family_windows():
    adapter = OpenAIResponsesAdapter()

    assert adapter.context_window_size("gpt-5") == 400000
    assert adapter.context_window_size("gpt-5.2") == 400000
    assert adapter.context_window_size("gpt-5.5") == 400000
    assert adapter.context_window_size("gpt-5.4") == 272000


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
                        "total_tokens": 16,
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
        caller=_FakeRoom().local_participant,
        toolkits=[],
    )

    assert result == ""
    assert call_count["count"] == 1
    assert context.turn_count == 1
    assert context.metadata["last_response_usage"]["input_tokens"] == 12
    assert context.usage == {
        "input_tokens": 9.0,
        "output_tokens": 4.0,
        "cached_tokens": 3.0,
        "total_tokens": 16.0,
    }
    assert context.metadata["last_response_context_used_tokens"] == 16


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
                        "total_tokens": 11,
                    },
                )
            ]
        ),
    )
    context = adapter.create_session()
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[],
    )

    assert result == ""
    assert context.turn_count == 1
    assert context.metadata["last_response_usage"]["input_tokens"] == 9
    assert context.usage == {
        "input_tokens": 4.0,
        "output_tokens": 2.0,
        "cached_tokens": 5.0,
        "total_tokens": 11.0,
    }
    assert context.metadata["last_response_context_used_tokens"] == 11


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
        caller=_FakeRoom().local_participant,
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
        caller=room.local_participant,
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
                                "total_tokens": 18,
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
        caller=_FakeRoom().local_participant,
        toolkits=[],
        event_handler=events.append,
    )

    assert result == ""
    assert context.turn_count == 1
    assert events[0]["type"] == "response.completed"
    assert context.metadata["last_response_usage"]["output_tokens"] == 7
    assert context.usage == {
        "input_tokens": 7.0,
        "output_tokens": 7.0,
        "cached_tokens": 4.0,
        "total_tokens": 18.0,
    }
    assert context.metadata["last_response_context_used_tokens"] == 18


@pytest.mark.asyncio
async def test_next_commits_response_state_when_stream_ends_incomplete_after_compaction():
    compaction_item = _FakeOutputItem(
        type="compaction",
        encrypted_content="opaque",
        status="completed",
    )
    response = _FakeResponse(
        response_id="resp_incomplete_compaction",
        output=[compaction_item],
        usage={
            "input_tokens": 20,
            "output_tokens": 3,
            "input_tokens_details": {"cached_tokens": 2},
        },
    )
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=_FakeOpenAIClient(
            outcomes=[
                _EventStream(
                    events=[
                        _FakeOutputItemDoneEvent(item=compaction_item),
                        _FakeIncompleteEvent(response=response),
                    ]
                )
            ]
        ),
    )
    context = adapter.create_session()
    context.append_user_message("hello")
    events: list[dict] = []

    result = await adapter.next(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[],
        event_handler=events.append,
    )

    assert result == ""
    assert context.previous_response_id == "resp_incomplete_compaction"
    assert context.messages == []
    assert context.previous_messages[-1] == {
        "type": "compaction",
        "encrypted_content": "opaque",
        "status": "completed",
    }
    assert context.metadata["last_response_usage"]["input_tokens"] == 20
    assert context.metadata["last_response_compaction_threshold"] == 200000
    assert context.usage == {
        "input_tokens": 18.0,
        "output_tokens": 3.0,
        "cached_tokens": 2.0,
    }
    assert [event["type"] for event in events] == [
        "response.output_item.done",
        "response.incomplete",
    ]


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
        caller=_FakeRoom().local_participant,
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
        caller=_FakeRoom().local_participant,
        toolkits=[],
    )

    assert result == ""
    assert len(client.responses.create_kwargs) == 1
    create_kwargs = client.responses.create_kwargs[0]
    assert create_kwargs["context_management"] == [
        {"type": "compaction", "compact_threshold": 10000}
    ]


@pytest.mark.asyncio
async def test_next_marks_usage_when_response_contains_auto_compaction() -> None:
    compaction_item = _FakeOutputItem(
        type="compaction",
        encrypted_content="opaque",
        status="completed",
    )
    client = _FakeOpenAIClient(
        outcomes=[
            _FakeResponse(
                response_id="resp_auto_compacted",
                output=[compaction_item],
                usage={"input_tokens": 20000, "output_tokens": 10},
            )
        ]
    )
    adapter = OpenAIResponsesAdapter(
        client=client,
        mode="request",
        compaction_threshold=10000,
        max_output_tokens=500,
    )
    context = adapter.create_session()
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[],
    )

    assert result == ""
    assert context.metadata["last_response_usage"] == {
        "input_tokens": 20000,
        "output_tokens": 10,
    }
    assert context.metadata["last_response_compaction_threshold"] == 10000
    assert context.usage == {
        "input_tokens": 20000.0,
        "output_tokens": 10.0,
    }


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
        caller=_FakeRoom().local_participant,
        toolkits=[],
    )

    assert result == ""
    assert len(client.responses.create_kwargs) == 1
    create_kwargs = client.responses.create_kwargs[0]
    assert "context_management" not in create_kwargs


@pytest.mark.asyncio
async def test_next_combines_context_and_adapter_instructions() -> None:
    client = _FakeOpenAIClient(
        outcomes=[_FakeResponse(response_id="resp_combined_instructions")]
    )
    adapter = _InstructionalOpenAIResponsesAdapter(
        client=client,
        mode="request",
        model="gpt-5.2",
    )
    context = adapter.create_session()
    context.instructions = "base context instructions"
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[],
    )

    assert result == ""
    assert client.responses.create_kwargs[0]["instructions"] == (
        "base context instructions\n\nextra adapter instructions"
    )


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
        caller=_FakeRoom().local_participant,
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
        caller=_FakeRoom().local_participant,
        toolkits=[],
    )

    assert result == ""
    assert len(client.responses.create_kwargs) == 1
    create_kwargs = client.responses.create_kwargs[0]
    assert create_kwargs["context_management"] == [
        {"type": "compaction", "compact_threshold": 200000}
    ]


def test_needs_compaction_uses_adapter_context_used_tokens():
    adapter = OpenAIResponsesAdapter(
        mode="request",
        context_management="standalone",
        compaction_threshold=200,
    )
    context = adapter.create_session()
    context.metadata["last_response_context_used_tokens"] = 330
    context.metadata["last_response_model"] = "gpt-5.2"

    assert adapter.needs_compaction(context=context)


def test_needs_compaction_ignores_raw_usage_without_flattened_usage():
    adapter = OpenAIResponsesAdapter(
        mode="request",
        context_management="standalone",
        compaction_threshold=200,
    )
    context = adapter.create_session()
    context.metadata["last_response_usage"] = {
        "input_tokens": 100,
        "input_tokens_details": {"cached_tokens": 200},
        "output_tokens": 30,
    }
    context.metadata["last_response_model"] = "gpt-5.2"

    assert not adapter.needs_compaction(context=context)


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
    context.metadata["last_response_flattened_usage"] = {
        "input_tokens": 300000,
        "output_tokens": 1000,
    }
    context.metadata["last_response_context_used_tokens"] = 301000
    context.metadata["last_response_model"] = "gpt-5.2"
    compact_call_count = {"count": 0}

    async def _fake_compact(**kwargs):
        del kwargs
        compact_call_count["count"] += 1

    monkeypatch.setattr(adapter, "compact", _fake_compact)

    result = await adapter.next(
        context=context,
        caller=_FakeRoom().local_participant,
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
        caller=_FakeRoom().local_participant,
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

    with pytest.raises(RoomException, match="account dashboard") as exc_info:
        await adapter._create_response_websocket_stream(
            context=context,
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

    def _fake_new_client_session(*args, **kwargs):
        del args
        del kwargs
        return fake_session

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.new_client_session",
        _fake_new_client_session,
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
async def test_next_serializes_websocket_requests_per_session(monkeypatch):
    adapter = OpenAIResponsesAdapter(mode="websocket")
    context = adapter.create_session()
    websocket = _QueuedLoggingWebSocket()

    async def _fake_ensure_websocket(*, url: str, headers: dict[str, str]):
        del url
        del headers
        return websocket

    monkeypatch.setattr(context, "ensure_websocket", _fake_ensure_websocket)
    monkeypatch.setattr(
        adapter,
        "get_openai_client",
        lambda: SimpleNamespace(
            base_url="https://example.com/openai/v1",
            default_headers={"Authorization": "Bearer test-token"},
        ),
    )

    def _fake_coerce_response_stream_event(payload: dict):
        payload_type = payload["type"]
        if payload_type in {"response.completed", "response.done"}:
            response = payload.get("response", {})
            response_id = response.get("id", "resp_ws")
            return _FakeCompletedEvent(response=_FakeResponse(response_id=response_id))
        return SimpleNamespace(type=payload_type)

    monkeypatch.setattr(
        adapter,
        "_coerce_response_stream_event",
        _fake_coerce_response_stream_event,
    )

    first_task = asyncio.create_task(
        adapter.next(
            context=context,
            caller=_FakeRoom().local_participant,
            toolkits=[],
        )
    )
    await _wait_for(lambda: len(websocket.sent_payloads) == 1)

    second_task = asyncio.create_task(
        adapter.next(
            context=context,
            caller=_FakeRoom().local_participant,
            toolkits=[],
        )
    )
    await asyncio.sleep(0.05)
    assert len(websocket.sent_payloads) == 1

    websocket.queue_json(
        {
            "type": "response.completed",
            "response": {"id": "resp_first", "output": [], "usage": None},
        }
    )
    await first_task

    await _wait_for(lambda: len(websocket.sent_payloads) == 2)
    websocket.queue_json(
        {
            "type": "response.completed",
            "response": {"id": "resp_second", "output": [], "usage": None},
        }
    )
    await second_task


@pytest.mark.asyncio
async def test_cancelled_request_closes_websocket_and_next_request_reconnects(
    monkeypatch,
):
    adapter = OpenAIResponsesAdapter(mode="websocket")
    first_websocket = _QueuedLoggingWebSocket()
    second_websocket = _QueuedLoggingWebSocket()
    client_session = _SequentialClientSession([first_websocket, second_websocket])
    context = OpenAIResponsesSessionContext(session=client_session)
    monkeypatch.setattr(
        adapter,
        "get_openai_client",
        lambda: SimpleNamespace(
            base_url="https://example.com/openai/v1",
            default_headers={"Authorization": "Bearer test-token"},
        ),
    )

    def _fake_coerce_response_stream_event(payload: dict):
        payload_type = payload["type"]
        if payload_type in {"response.completed", "response.done"}:
            response = payload.get("response", {})
            response_id = response.get("id", "resp_ws")
            return _FakeCompletedEvent(response=_FakeResponse(response_id=response_id))
        return SimpleNamespace(type=payload_type)

    monkeypatch.setattr(
        adapter,
        "_coerce_response_stream_event",
        _fake_coerce_response_stream_event,
    )

    first_task = asyncio.create_task(
        adapter.next(
            context=context,
            caller=_FakeRoom().local_participant,
            toolkits=[],
        )
    )
    second_task: asyncio.Task | None = None
    try:
        await _wait_for(lambda: len(first_websocket.sent_payloads) == 1)

        first_task.cancel()
        second_task = asyncio.create_task(
            adapter.next(
                context=context,
                caller=_FakeRoom().local_participant,
                toolkits=[],
            )
        )
        await _wait_for(lambda: first_websocket.closed)

        with pytest.raises(asyncio.CancelledError):
            await first_task

        await _wait_for(lambda: len(second_websocket.sent_payloads) == 1)
        assert client_session.connect_calls == 2
        assert len(first_websocket.sent_payloads) == 1

        second_websocket.queue_json(
            {
                "type": "response.completed",
                "response": {"id": "resp_second", "output": [], "usage": None},
            }
        )
        await second_task
    finally:
        if not first_task.done():
            first_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await first_task
        if second_task is not None and not second_task.done():
            second_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await second_task
        await context.close()


@pytest.mark.asyncio
async def test_session_context_closes_created_session_on_close(monkeypatch):
    fake_websocket = _FakeWebSocket()
    fake_session = _FakeClientSession(fake_websocket)

    def _fake_new_client_session(*args, **kwargs):
        del args, kwargs
        return fake_session

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.new_client_session",
        _fake_new_client_session,
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

    def _fake_new_client_session(*args, **kwargs):
        del args, kwargs
        return fake_session

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.new_client_session",
        _fake_new_client_session,
    )

    context = OpenAIResponsesSessionContext(system_role=None)

    with pytest.raises(RoomException, match="account dashboard") as exc_info:
        await context.ensure_websocket(
            url="ws://localhost:8080/openai/v1/responses",
            headers={"Authorization": "Bearer test-token"},
        )

    assert exc_info.value.status_code == 402
    assert fake_session.closed is True


@pytest.mark.asyncio
async def test_session_context_closes_session_when_ws_connect_fails(monkeypatch):
    fake_session = _FailingConnectClientSession()

    def _fake_new_client_session(*args, **kwargs):
        del args
        del kwargs
        return fake_session

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.new_client_session",
        _fake_new_client_session,
    )

    context = OpenAIResponsesSessionContext(system_role=None)

    with pytest.raises(RuntimeError, match="ws connect failed"):
        await context.ensure_websocket(
            url="ws://localhost:8080/openai/v1/responses",
            headers={"Authorization": "Bearer test-token"},
        )

    assert fake_session.closed is True


def test_session_context_appends_remote_image_and_file_urls() -> None:
    context = OpenAIResponsesSessionContext(system_role=None)

    image_message = context.append_image_url(url="https://example.com/image.png")
    file_message = context.append_file_url(url="https://example.com/report.pdf")

    assert image_message["content"][0] == {
        "type": "input_image",
        "image_url": "https://example.com/image.png",
    }
    assert file_message["content"][0] == {
        "type": "input_file",
        "file_url": "https://example.com/report.pdf",
    }


@pytest.mark.asyncio
async def test_next_inserts_steering_messages_after_tool_results() -> None:
    client = _FakeOpenAIClient(
        outcomes=[
            _FakeResponse(
                response_id="resp_tool",
                output=[
                    _make_function_tool_call(
                        item_id="tool-1",
                        tool_name="write_file",
                        call_id="call_1",
                        arguments={"path": "/tmp/example.txt"},
                    )
                ],
            ),
            _FakeResponse(
                response_id="resp_done",
                output=[_make_output_message(message_id="msg_1", text="done")],
            ),
        ]
    )
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=client,
        model="gpt-4.1-mini",
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
        caller=_FakeRoom().local_participant,
        toolkits=[Toolkit(name="storage", tools=[_AnyArgsTool("write_file")])],
        steering_callback=_steer,
    )

    assert result == "done"
    assert steering_calls == 1
    assert len(client.responses.create_kwargs) == 2
    second_input = client.responses.create_kwargs[1]["input"]
    assert second_input == [
        {
            "output": json.dumps(
                {
                    "ok": True,
                    "args": {"path": "/tmp/example.txt"},
                }
            ),
            "call_id": "call_1",
            "type": "function_call_output",
        },
        {
            "role": "user",
            "content": "steer now",
        },
    ]
    assert client.responses.create_kwargs[1]["previous_response_id"] == "resp_tool"
    assert second_input[-1] == {
        "role": "user",
        "content": "steer now",
    }


@pytest.mark.asyncio
async def test_next_drops_post_tool_response_items_before_steering_in_request_mode() -> (
    None
):
    client = _FakeOpenAIClient(
        outcomes=[
            _FakeResponse(
                response_id="resp_tool",
                output=[
                    _make_function_tool_call(
                        item_id="tool-1",
                        tool_name="write_file",
                        call_id="call_1",
                        arguments={"path": "/tmp/example.txt"},
                    ),
                    _make_output_message(
                        message_id="msg_after_tool",
                        text="this should not land before steering",
                    ),
                ],
            ),
            _FakeResponse(
                response_id="resp_done",
                output=[_make_output_message(message_id="msg_1", text="done")],
            ),
        ]
    )
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=client,
        model="gpt-4.1-mini",
    )
    context = adapter.create_session()
    context.append_user_message("run tool")

    async def _steer() -> bool:
        context.append_user_message("steer now")
        return True

    result = await adapter.next(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[Toolkit(name="storage", tools=[_AnyArgsTool("write_file")])],
        steering_callback=_steer,
    )

    assert result == "done"
    second_create_kwargs = client.responses.create_kwargs[1]
    assert second_create_kwargs["input"] == [
        {
            "role": "user",
            "content": "run tool",
        },
        {
            "arguments": json.dumps({"path": "/tmp/example.txt"}),
            "call_id": "call_1",
            "id": "tool-1",
            "name": "write_file",
            "status": "completed",
            "type": "function_call",
        },
        {
            "output": json.dumps(
                {
                    "ok": True,
                    "args": {"path": "/tmp/example.txt"},
                }
            ),
            "call_id": "call_1",
            "type": "function_call_output",
        },
        {
            "role": "user",
            "content": "steer now",
        },
    ]


@pytest.mark.asyncio
async def test_next_passes_thread_and_turn_ids_in_tool_caller_context() -> None:
    tool = _CallerContextTool("context_tool")
    client = _FakeOpenAIClient(
        outcomes=[
            _FakeResponse(
                response_id="resp_tool",
                output=[
                    _make_function_tool_call(
                        item_id="tool-1",
                        tool_name="context_tool",
                        call_id="call_1",
                        arguments={},
                    )
                ],
            ),
            _FakeResponse(
                response_id="resp_done",
                output=[_make_output_message(message_id="msg_1", text="done")],
            ),
        ]
    )
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=client,
        model="gpt-4.1-mini",
    )
    context = adapter.create_session()
    context.append_user_message("run tool")
    context.metadata["thread_id"] = "thread-1"
    context.metadata["turn_id"] = "turn-1"

    result = await adapter.next(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[Toolkit(name="tools", tools=[tool])],
    )

    assert result == "done"
    assert len(tool.caller_contexts) == 1
    caller_context = tool.caller_contexts[0]
    assert isinstance(caller_context, dict)
    assert caller_context["thread_id"] == "thread-1"
    assert caller_context["turn_id"] == "turn-1"
    assert caller_context["item_id"] == "tool-1"
    assert isinstance(caller_context.get("chat"), dict)


@pytest.mark.asyncio
async def test_next_drops_post_tool_stream_items_before_steering() -> None:
    tool_call = _make_function_tool_call(
        item_id="tool-1",
        tool_name="write_file",
        call_id="call_1",
        arguments={"path": "/tmp/example.txt"},
    )
    trailing_message = _make_output_message(
        message_id="msg_after_tool",
        text="this should not land before steering",
    )
    completed_response = _FakeResponse(
        response_id="resp_tool",
        output=[tool_call, trailing_message],
    )
    client = _FakeOpenAIClient(
        outcomes=[
            _EventStream(
                events=[
                    _FakeOutputItemDoneEvent(item=tool_call),
                    _FakeOutputItemDoneEvent(item=trailing_message),
                    _FakeCompletedEvent(response=completed_response),
                ]
            ),
            _CompletedStream(
                event=_FakeCompletedEvent(
                    response=_FakeResponse(
                        response_id="resp_done",
                        output=[
                            _make_output_message(message_id="msg_done", text="done")
                        ],
                    )
                )
            ),
        ]
    )
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=client,
        model="gpt-4.1-mini",
    )
    context = adapter.create_session()
    context.append_user_message("run tool")
    published_events: list[dict] = []

    async def _steer() -> bool:
        context.append_user_message("steer now")
        return True

    result = await adapter.next(
        context=context,
        caller=_FakeRoom().local_participant,
        toolkits=[Toolkit(name="storage", tools=[_AnyArgsTool("write_file")])],
        event_handler=published_events.append,
        steering_callback=_steer,
    )

    assert result == "done"
    second_create_kwargs = client.responses.create_kwargs[1]
    assert second_create_kwargs["input"] == [
        {
            "role": "user",
            "content": "run tool",
        },
        {
            "arguments": json.dumps({"path": "/tmp/example.txt"}),
            "call_id": "call_1",
            "id": "tool-1",
            "name": "write_file",
            "status": "completed",
            "type": "function_call",
        },
        {
            "output": json.dumps(
                {
                    "ok": True,
                    "args": {"path": "/tmp/example.txt"},
                }
            ),
            "call_id": "call_1",
            "type": "function_call_output",
        },
        {
            "role": "user",
            "content": "steer now",
        },
    ]
    assert not any(
        event.get("type") == "response.output_item.done"
        and event.get("item", {}).get("id") == "msg_after_tool"
        for event in published_events
    )


@pytest.mark.asyncio
async def test_next_restarts_after_first_completed_tool_call_before_later_tool_calls() -> (
    None
):
    tool_call_a = _make_function_tool_call(
        item_id="tool-1",
        tool_name="tool_a",
        call_id="call_1",
        arguments={},
    )
    tool_call_b = _make_function_tool_call(
        item_id="tool-2",
        tool_name="tool_b",
        call_id="call_2",
        arguments={},
    )
    completed_response = _FakeResponse(
        response_id="resp_tool",
        output=[tool_call_a, tool_call_b],
    )
    client = _FakeOpenAIClient(
        outcomes=[
            _EventStream(
                events=[
                    _FakeOutputItemDoneEvent(item=tool_call_a),
                    _FakeOutputItemDoneEvent(item=tool_call_b),
                    _FakeCompletedEvent(response=completed_response),
                ]
            ),
            _CompletedStream(
                event=_FakeCompletedEvent(
                    response=_FakeResponse(
                        response_id="resp_done",
                        output=[
                            _make_output_message(message_id="msg_done", text="done")
                        ],
                    )
                )
            ),
        ]
    )
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=client,
        model="gpt-4.1-mini",
    )
    context = adapter.create_session()
    context.append_user_message("run tools")
    tool_a = _GateTool("tool_a")
    tool_b = _AnyArgsTool("tool_b")
    pending_steer = False

    async def _steer() -> bool:
        nonlocal pending_steer
        if not pending_steer:
            return False
        pending_steer = False
        context.append_user_message("steer now")
        return True

    task = asyncio.create_task(
        adapter.next(
            context=context,
            caller=_FakeRoom().local_participant,
            toolkits=[Toolkit(name="test", tools=[tool_a, tool_b])],
            event_handler=lambda event: None,
            steering_callback=_steer,
        )
    )

    await asyncio.wait_for(tool_a.started.wait(), timeout=1)
    pending_steer = True
    tool_a.release.set()
    result = await asyncio.wait_for(task, timeout=1)

    assert result == "done"
    second_create_kwargs = client.responses.create_kwargs[1]
    assert second_create_kwargs["input"] == [
        {"role": "user", "content": "run tools"},
        {
            "arguments": json.dumps({}),
            "call_id": "call_1",
            "id": "tool-1",
            "name": "tool_a",
            "status": "completed",
            "type": "function_call",
        },
        {
            "output": json.dumps({"ok": True, "tool": "tool_a"}),
            "call_id": "call_1",
            "type": "function_call_output",
        },
        {"role": "user", "content": "steer now"},
    ]
    assert not any(
        isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("id") == "tool-2"
        for item in second_create_kwargs["input"]
    )


@pytest.mark.asyncio
async def test_request_mode_cancellation_restores_context_during_tool_call() -> None:
    blocking_tool = _BlockingTool("write_file")
    adapter = OpenAIResponsesAdapter(
        mode="request",
        client=_FakeOpenAIClient(
            outcomes=[
                _FakeResponse(
                    response_id="resp_tool",
                    output=[
                        _make_function_tool_call(
                            item_id="tool-1",
                            tool_name="write_file",
                            call_id="call_1",
                            arguments={"path": "/tmp/example.txt"},
                        )
                    ],
                )
            ]
        ),
        model="gpt-4.1-mini",
    )
    context = adapter.create_session()
    context.append_user_message("run tool")

    task = asyncio.create_task(
        adapter.next(
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
    assert context.previous_messages == []
    assert context.previous_response_id is None


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
        caller=_FakeRoom().local_participant,
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
        caller=_FakeRoom().local_participant,
        toolkits=[],
        event_handler=stream_events.append,
    )

    assert result == ""
    assert client.responses.calls == 2
    assert sleep_calls == [1.0]
    assert [event["type"] for event in stream_events] == [
        "agent.event",
        "agent.event",
        "response.completed",
    ]
    assert stream_events[0]["state"] == "in_progress"
    assert stream_events[0]["headline"] == "Retrying the LLM request (retry 1/3)"
    assert stream_events[1]["state"] == "completed"
    assert stream_events[1]["headline"] == "LLM request retry succeeded"


@pytest.mark.asyncio
async def test_next_omits_on_behalf_of_header_when_name_is_missing() -> None:
    client = _FakeOpenAIClient(
        outcomes=[
            _FakeResponse(
                response_id="resp_done",
                output=[_make_output_message(message_id="msg_done", text="done")],
            )
        ]
    )

    adapter = OpenAIResponsesAdapter(client=client, max_retries=3, mode="request")
    context = adapter.create_session()
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        caller=_FakeRoom().local_participant,
        on_behalf_of=_NamelessParticipant(),
        toolkits=[],
    )

    assert result == "done"
    assert len(client.responses.create_kwargs) == 1
    assert client.responses.create_kwargs[0]["extra_headers"] == {}


@pytest.mark.asyncio
async def test_next_retries_after_shell_tool_room_exception(monkeypatch):
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.asyncio.sleep",
        _fake_sleep,
    )

    shell_call = _make_shell_tool_call(
        item_id="shell_1",
        call_id="call_1",
        commands=["echo hi"],
    )
    client = _FakeOpenAIClient(
        outcomes=[
            _EventStream(
                events=[
                    _FakeOutputItemDoneEvent(item=shell_call),
                ]
            ),
            _CompletedStream(
                event=_FakeCompletedEvent(
                    response=_FakeResponse(
                        response_id="resp_done",
                        output=[_make_output_message(message_id="msg_1", text="done")],
                    )
                )
            ),
        ]
    )

    adapter = OpenAIResponsesAdapter(client=client, max_retries=3, mode="request")
    context = adapter.create_session()
    context.append_user_message("hello")
    stream_events: list[dict] = []
    room = _FakeContainerRoom(
        exec_factory=lambda: _FakeContainerExec(
            stdout_chunks=[],
            stderr_chunks=[],
            exit_code=0,
        ),
        run_error=RoomException(
            "unable to pull image: failed to pull and unpack image "
            '"docker.io/library/none:latest"',
            code=ErrorCode.OPERATION_FAILED,
        ),
    )

    shell_tool = ShellTool(room=room, image="meshagent/python:default")

    result = await adapter.next(
        context=context,
        caller=room.local_participant,
        toolkits=[Toolkit(name="openai", tools=[shell_tool])],
        event_handler=stream_events.append,
    )

    assert result == "done"
    assert client.responses.calls == 2
    assert sleep_calls == [1.0]
    assert len(room.containers.run_calls) == 1
    assert room.containers.run_calls[0]["command"] == "sleep infinity"
    assert room.containers.run_calls[0]["image"] == "meshagent/python:default"
    assert room.containers.run_calls[0]["writable_root_fs"] is True
    assert room.containers.run_calls[0]["env"] is None
    retry_events = [
        event for event in stream_events if event.get("type") == "agent.event"
    ]
    assert [event["state"] for event in retry_events] == ["in_progress", "completed"]
    assert retry_events[0]["headline"] == "Retrying the LLM request (retry 1/3)"
    assert retry_events[1]["headline"] == "LLM request retry succeeded"


@pytest.mark.asyncio
async def test_receive_websocket_payload_marks_timeout_close_as_retryable():
    adapter = OpenAIResponsesAdapter(mode="websocket")
    websocket = _FakeLoggingWebSocket(
        messages=[
            _FakeLoggingWsMessage(
                type=aiohttp.WSMsgType.CLOSED,
                data="",
            )
        ],
        exception=TimeoutError("No PONG received after 15.0 seconds"),
        close_code=1006,
    )

    with pytest.raises(RoomException) as exc_info:
        await adapter._receive_websocket_payload(websocket=websocket)

    assert exc_info.value.code == ErrorCode.TIMEOUT
    assert exc_info.value.status_code == 408
    assert "OpenAI websocket closed unexpectedly" in str(exc_info.value)
    assert "No PONG received after 15.0 seconds" in str(exc_info.value)


@pytest.mark.asyncio
async def test_receive_websocket_payload_logs_request_body_for_4xx_error_event(
    caplog,
):
    adapter = OpenAIResponsesAdapter(mode="websocket")
    websocket = _FakeLoggingWebSocket(
        messages=[
            _FakeLoggingWsMessage(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": (
                                "[ObjectParam] [input[0].output[0].outcome.exit_code] "
                                "[unknown_parameter] Unknown parameter: "
                                "'input[0].output[0].outcome.exit_code'."
                            ),
                        },
                        "status": 400,
                    }
                ),
            )
        ]
    )
    request_payload = {
        "type": "response.create",
        "model": "gpt-5.2",
        "input": [
            {
                "type": "shell_call_output",
                "call_id": "call_1",
                "output": [
                    {
                        "outcome": {"type": "exit", "exit_code": 0},
                        "stdout": "hi\n",
                        "stderr": "",
                    }
                ],
            }
        ],
    }

    caplog.set_level(logging.ERROR, logger="openai_agent")

    with pytest.raises(RoomException) as exc_info:
        await adapter._receive_websocket_payload(
            websocket=websocket,
            request_payload=request_payload,
        )

    assert "unknown_parameter" in str(exc_info.value)
    assert "outcome.exit_code" in str(exc_info.value)
    logged_messages = [record.message for record in caplog.records]
    assert any(
        message.startswith("OpenAI websocket error request body=")
        for message in logged_messages
    )
    assert any('"type": "shell_call_output"' in message for message in logged_messages)
    assert any('"exit_code": 0' in message for message in logged_messages)


@pytest.mark.asyncio
async def test_receive_websocket_payload_does_not_log_request_body_for_out_of_credits(
    caplog,
):
    adapter = OpenAIResponsesAdapter(mode="websocket")
    websocket = _FakeLoggingWebSocket(
        messages=[
            _FakeLoggingWsMessage(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "insufficient_quota",
                            "code": "insufficient_quota",
                            "message": "Your account is out of credits. Add credits to continue.",
                        },
                        "status": 402,
                    }
                ),
            )
        ]
    )
    request_payload = {
        "type": "response.create",
        "model": "gpt-5.5",
        "input": [{"role": "user", "content": "hello"}],
    }

    caplog.set_level(logging.ERROR, logger="openai_agent")

    with pytest.raises(RoomException) as exc_info:
        await adapter._receive_websocket_payload(
            websocket=websocket,
            request_payload=request_payload,
        )

    assert exc_info.value.status_code == 402
    assert "account dashboard" in str(exc_info.value)
    assert "auto reload" in str(exc_info.value)
    logged_messages = [record.message for record in caplog.records]
    assert not any(
        message.startswith("OpenAI websocket error request body=")
        for message in logged_messages
    )


@pytest.mark.asyncio
async def test_next_retries_after_websocket_close(monkeypatch):
    adapter = OpenAIResponsesAdapter(mode="websocket", max_retries=2)
    first_websocket = _QueuedLoggingWebSocket(
        exception=TimeoutError("No PONG received after 15.0 seconds"),
        close_code=1006,
    )
    first_websocket.queue_message(message_type=aiohttp.WSMsgType.CLOSED)
    second_websocket = _QueuedLoggingWebSocket()
    second_websocket.queue_json(
        {
            "type": "response.completed",
            "response": {"id": "resp_second", "output": [], "usage": None},
        }
    )
    client_session = _SequentialClientSession([first_websocket, second_websocket])
    context = OpenAIResponsesSessionContext(
        system_role=None,
        session=client_session,
        websocket_ping_interval_seconds=3600,
        websocket_timeout=3600,
    )
    context.append_user_message("hello")
    stream_events: list[dict] = []

    monkeypatch.setattr(
        adapter,
        "_retry_delay_seconds",
        lambda *, retry_number, error: 0.0,
    )
    monkeypatch.setattr(
        adapter,
        "get_openai_client",
        lambda: SimpleNamespace(
            base_url="https://example.com/openai/v1",
            default_headers={"Authorization": "Bearer test-token"},
        ),
    )

    def _fake_coerce_response_stream_event(payload: dict):
        payload_type = payload["type"]
        if payload_type in {"response.completed", "response.done"}:
            response = payload.get("response", {})
            response_id = response.get("id", "resp_ws")
            return _FakeCompletedEvent(response=_FakeResponse(response_id=response_id))
        return SimpleNamespace(type=payload_type)

    monkeypatch.setattr(
        adapter,
        "_coerce_response_stream_event",
        _fake_coerce_response_stream_event,
    )
    try:
        result = await adapter.next(
            context=context,
            caller=_FakeRoom().local_participant,
            toolkits=[],
            event_handler=stream_events.append,
        )

        assert result == ""
        assert client_session.connect_calls == 2
        assert first_websocket.closed is True
        assert [event["type"] for event in stream_events] == [
            "agent.event",
            "agent.event",
            "response.completed",
        ]
        assert stream_events[0]["state"] == "in_progress"
        assert stream_events[0]["headline"] == "Reconnecting to the LLM (retry 1/2)"
        assert stream_events[1]["state"] == "completed"
        assert stream_events[1]["headline"] == "Reconnected to the LLM"
    finally:
        await context.close()


@pytest.mark.asyncio
async def test_next_does_not_retry_after_websocket_out_of_credits(monkeypatch):
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setattr(
        "meshagent.openai.tools.responses_adapter.asyncio.sleep",
        _fake_sleep,
    )

    adapter = OpenAIResponsesAdapter(mode="websocket", max_retries=2)
    websocket = _QueuedLoggingWebSocket()
    websocket.queue_json(
        {
            "type": "error",
            "error": {
                "type": "insufficient_quota",
                "code": "insufficient_quota",
                "message": "Your account is out of credits. Add credits to continue.",
            },
            "status": 402,
        }
    )
    client_session = _SequentialClientSession([websocket])
    context = OpenAIResponsesSessionContext(
        system_role=None,
        session=client_session,
        websocket_ping_interval_seconds=3600,
        websocket_timeout=3600,
    )
    context.append_user_message("hello")
    stream_events: list[dict] = []

    monkeypatch.setattr(
        adapter,
        "get_openai_client",
        lambda: SimpleNamespace(
            base_url="https://example.com/openai/v1",
            default_headers={"Authorization": "Bearer test-token"},
        ),
    )

    try:
        with pytest.raises(RoomException) as exc_info:
            await adapter.next(
                context=context,
                caller=_FakeRoom().local_participant,
                toolkits=[],
                event_handler=stream_events.append,
            )

        assert exc_info.value.status_code == 402
        assert "account dashboard" in str(exc_info.value)
        assert client_session.connect_calls == 1
        assert sleep_calls == []
        assert stream_events == []
    finally:
        await context.close()


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
            caller=_FakeRoom().local_participant,
            toolkits=[],
        )

    assert client.responses.calls == 2
    assert sleep_calls == [1.0]


def test_make_agent_event_publisher_emits_content_and_tool_messages() -> None:
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "part": {"type": "output_text", "text": ""},
        }
    )
    publisher(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "delta": "Hello",
        }
    )
    publisher(
        {
            "type": "response.output_text.done",
            "item_id": "msg_1",
            "text": "Hello",
        }
    )
    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "status": "in_progress",
                "summary": [],
                "content": [],
            },
        }
    )
    publisher(
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 1,
            "delta": "Thinking",
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "status": "completed",
                "summary": [],
                "content": [],
            },
        }
    )
    publisher(
        {
            "type": "response.content_part.added",
            "item_id": "file_1",
            "part": {
                "type": "output_file",
                "url": "https://example.com/report.pdf",
            },
        }
    )
    publisher(
        {
            "type": "response.content_part.done",
            "item_id": "file_1",
            "part": {
                "type": "output_file",
                "url": "https://example.com/report.pdf",
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {
                "type": "function_call",
                "id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"meshagent"}',
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 2,
            "item": {
                "type": "function_call",
                "id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"meshagent"}',
            },
        }
    )
    publisher({"type": "meshagent.handler.done", "item_id": "call_1"})
    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 3,
            "item": {
                "type": "web_search_call",
                "id": "search_1",
                "status": "in_progress",
                "queries": ["meshagent"],
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 3,
            "item": {
                "type": "web_search_call",
                "id": "search_1",
                "status": "completed",
                "queries": ["meshagent"],
                "results": [{"title": "MeshAgent"}],
            },
        }
    )

    assert [type(event) for event in published] == [
        AgentTextContentStarted,
        AgentTextContentDelta,
        AgentTextContentEnded,
        AgentReasoningContentStarted,
        AgentReasoningContentDelta,
        AgentReasoningContentEnded,
        AgentFileContentStarted,
        AgentFileContentDelta,
        AgentFileContentEnded,
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]
    for event in published:
        assert event.thread_id == "thread-1"

    text_delta = published[1]
    assert isinstance(text_delta, AgentTextContentDelta)
    assert text_delta.turn_id == "turn-1"
    assert text_delta.item_id == "msg_1"
    assert text_delta.text == "Hello"

    file_started = published[6]
    assert isinstance(file_started, AgentFileContentStarted)
    assert file_started.item_id == "file_1"

    file_delta = published[7]
    assert isinstance(file_delta, AgentFileContentDelta)
    assert file_delta.url == "https://example.com/report.pdf"

    function_started = published[10]
    assert isinstance(function_started, AgentToolCallStarted)
    assert function_started.toolkit == "function"
    assert function_started.tool == "lookup"
    assert function_started.arguments == {"q": "meshagent"}

    web_search_ended = published[13]
    assert isinstance(web_search_ended, AgentToolCallEnded)
    assert isinstance(web_search_ended.result, JsonContent)
    assert web_search_ended.result.json == {"results": [{"title": "MeshAgent"}]}


def test_make_agent_event_publisher_forwards_custom_events() -> None:
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    custom_events: list[dict[str, object]] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
        custom_event_callback=custom_events.append,
    )

    publisher(
        {
            "type": "agent.event",
            "headline": "Retrying the LLM request",
        }
    )

    assert published == []
    assert custom_events == [
        {
            "type": "agent.event",
            "headline": "Retrying the LLM request",
        }
    ]


def test_make_agent_event_publisher_emits_image_generation_events() -> None:
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    encoded = base64.b64encode(b"fake-image-bytes").decode("ascii")

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "image_generation_call",
                "id": "ig_1",
                "status": "in_progress",
                "output_format": "png",
                "quality": "high",
                "size": "1024x1024",
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "image_generation_call",
                "id": "ig_1",
                "status": "completed",
                "output_format": "png",
                "quality": "high",
                "size": "1024x1024",
                "result": encoded,
            },
        }
    )

    assert [type(event) for event in published] == [
        AgentImageGenerationStarted,
        AgentImageGenerationCompleted,
    ]

    started = published[0]
    assert isinstance(started, AgentImageGenerationStarted)
    assert started.item_id == "ig_1"
    assert started.toolkit == "openai"
    assert started.tool == "image_generation"
    assert started.arguments == {
        "output_format": "png",
        "quality": "high",
        "size": "1024x1024",
    }

    completed = published[1]
    assert isinstance(completed, AgentImageGenerationCompleted)
    assert completed.item_id == "ig_1"
    assert completed.toolkit == "openai"
    assert completed.tool == "image_generation"
    assert completed.images[0].uri == f"data:image/png;base64,{encoded}"
    assert completed.images[0].mime_type == "image/png"
    assert completed.images[0].width == 1024
    assert completed.images[0].height == 1024


def test_make_agent_event_publisher_emits_persisted_image_generation_result() -> None:
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "image_generation_call",
                "id": "ig_1",
                "status": "completed",
                "output_format": "png",
                "quality": "high",
                "size": "1024x1024",
                "images": [
                    {
                        "uri": "dataset://images?id=image-1",
                        "mime_type": "image/png",
                        "created_at": "2026-05-05T00:00:00Z",
                        "created_by": "assistant",
                        "width": 1024,
                        "height": 1024,
                        "status": "completed",
                        "status_detail": "Image saved",
                    }
                ],
            },
        }
    )

    assert [type(event) for event in published] == [AgentImageGenerationCompleted]
    completed = published[0]
    assert isinstance(completed, AgentImageGenerationCompleted)
    assert completed.item_id == "ig_1"
    assert completed.images[0].uri == "dataset://images?id=image-1"
    assert completed.images[0].width == 1024


@pytest.mark.asyncio
async def test_openai_responses_adapter_persists_image_before_publishing_done_event() -> (
    None
):
    images_dataset = _FakeImagesDataset()
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
        images_dataset=images_dataset,  # type: ignore[arg-type]
    )
    context = OpenAIResponsesSessionContext()
    context.metadata["thread_id"] = "dataset://threads/main"
    context.metadata["turn_id"] = "turn-1"
    encoded = base64.b64encode(b"fake-image-bytes").decode("ascii")

    payload = await adapter._persist_image_generation_output_item(
        context=context,
        caller=_FakeParticipant(),
        item={
            "type": "image_generation_call",
            "id": "ig_1",
            "status": "completed",
            "output_format": "png",
            "quality": "high",
            "size": "1024x1024",
            "result": encoded,
        },
    )

    assert len(images_dataset.save_calls) == 1
    assert images_dataset.save_calls[0]["data"] == b"fake-image-bytes"
    assert "result" not in payload
    assert payload["images"][0]["uri"] == "dataset://images?id=image-1"
    assert payload["images"][0]["width"] == 1024


def test_make_agent_event_publisher_preserves_text_delta_whitespace() -> None:
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "part": {"type": "output_text", "text": ""},
        }
    )
    publisher(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "delta": "Hello",
        }
    )
    publisher(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "delta": " world",
        }
    )
    publisher(
        {
            "type": "response.output_text.done",
            "item_id": "msg_1",
            "text": "Hello world",
        }
    )

    deltas = [
        event.text for event in published if isinstance(event, AgentTextContentDelta)
    ]

    assert deltas == ["Hello", " world"]
    assert "".join(deltas) == "Hello world"


def test_make_agent_event_publisher_resets_output_index_mapping_for_new_response() -> (
    None
):
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher({"type": "response.created", "response": {"id": "resp_1"}})
    publisher(
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "item_id": "msg_1",
            "part": {"type": "output_text", "text": ""},
        }
    )
    publisher(
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "item_id": "msg_1",
            "delta": "First",
        }
    )
    publisher(
        {
            "type": "response.output_text.done",
            "output_index": 0,
            "item_id": "msg_1",
            "text": "First",
        }
    )

    publisher({"type": "response.created", "response": {"id": "resp_2"}})
    publisher(
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "item_id": "msg_2",
            "part": {"type": "output_text", "text": ""},
        }
    )
    publisher(
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "item_id": "msg_2",
            "delta": "Second",
        }
    )
    publisher(
        {
            "type": "response.output_text.done",
            "output_index": 0,
            "item_id": "msg_2",
            "text": "Second",
        }
    )

    assert [type(event) for event in published] == [
        AgentTextContentStarted,
        AgentTextContentDelta,
        AgentTextContentEnded,
        AgentTextContentStarted,
        AgentTextContentDelta,
        AgentTextContentEnded,
    ]

    first_started = published[0]
    assert isinstance(first_started, AgentTextContentStarted)
    assert first_started.item_id == "msg_1"

    first_delta = published[1]
    assert isinstance(first_delta, AgentTextContentDelta)
    assert first_delta.item_id == "msg_1"
    assert first_delta.text == "First"

    second_started = published[3]
    assert isinstance(second_started, AgentTextContentStarted)
    assert second_started.item_id == "msg_2"

    second_delta = published[4]
    assert isinstance(second_delta, AgentTextContentDelta)
    assert second_delta.item_id == "msg_2"
    assert second_delta.text == "Second"


def test_make_agent_event_publisher_keeps_reasoning_item_id_stable_when_delta_arrives_before_output_item() -> (
    None
):
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 1,
            "delta": "official OpenAI deep research overview research agent browse synthesize report",
        }
    )
    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "status": "in_progress",
                "summary": [],
                "content": [],
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "status": "completed",
                "summary": [],
                "content": [],
            },
        }
    )

    assert [type(event) for event in published] == [
        AgentReasoningContentStarted,
        AgentReasoningContentDelta,
        AgentReasoningContentEnded,
    ]

    started = published[0]
    assert isinstance(started, AgentReasoningContentStarted)
    assert started.item_id == "output:1"

    delta = published[1]
    assert isinstance(delta, AgentReasoningContentDelta)
    assert delta.item_id == "output:1"

    ended = published[2]
    assert isinstance(ended, AgentReasoningContentEnded)
    assert ended.item_id == "output:1"


def test_make_agent_event_publisher_unmangles_function_tool_names_from_tool_bundle():
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )
    toolkit = Toolkit(
        name="search",
        tools=[_AnyArgsTool("lookup/web")],
    )
    tool_bundle = ResponsesToolBundle(toolkits=[toolkit])
    adapter._set_function_tool_name_resolver(
        event_handler=publisher,
        resolver=tool_bundle.resolve_function_tool_name,
    )

    safe_name = safe_tool_name("lookup/web")
    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "call_1",
                "name": safe_name,
                "arguments": '{"q":"meshagent"}',
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "call_1",
                "name": safe_name,
                "arguments": '{"q":"meshagent"}',
            },
        }
    )
    publisher({"type": "meshagent.handler.done", "item_id": "call_1"})

    assert [type(event) for event in published] == [
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]
    started = published[1]
    assert isinstance(started, AgentToolCallStarted)
    assert started.toolkit == "search"
    assert started.tool == "lookup/web"
    assert started.arguments == {"q": "meshagent"}


def test_make_agent_event_publisher_updates_function_tool_failure_from_handler_done():
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "call_1",
                "name": "write_file",
                "arguments": "",
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "call_1",
                "name": "write_file",
                "arguments": '{"path":"src/app.py"}',
            },
        }
    )
    publisher(
        {
            "type": "meshagent.handler.done",
            "item_id": "call_1",
            "error": "'text' is a required property",
        }
    )

    assert [type(event) for event in published] == [
        AgentToolCallPending,
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]
    updated_pending = published[1]
    assert isinstance(updated_pending, AgentToolCallPending)
    assert updated_pending.arguments == {"path": "src/app.py"}

    started = published[2]
    assert isinstance(started, AgentToolCallStarted)
    assert started.arguments == {"path": "src/app.py"}

    ended = published[3]
    assert isinstance(ended, AgentToolCallEnded)
    assert ended.error is not None
    assert ended.error.message == "'text' is a required property"


def test_make_agent_event_publisher_emits_web_search_tool_events() -> None:
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "web_search_call",
                "id": "search_1",
                "status": "in_progress",
                "queries": ["meshagent"],
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "web_search_call",
                "id": "search_1",
                "status": "completed",
                "queries": ["meshagent"],
                "results": [{"title": "MeshAgent"}],
            },
        }
    )

    assert [type(event) for event in published] == [
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]
    started = published[0]
    assert isinstance(started, AgentToolCallStarted)
    assert started.namespace == "openai.responses"
    assert started.call_id is None
    assert started.toolkit == "openai"
    assert started.tool == "web_search"
    assert started.arguments == {"queries": ["meshagent"]}

    ended = published[1]
    assert isinstance(ended, AgentToolCallEnded)
    assert ended.namespace == "openai.responses"
    assert ended.call_id is None
    assert isinstance(ended.result, JsonContent)
    assert ended.result.json == {"results": [{"title": "MeshAgent"}]}


def test_web_search_tool_uses_current_responses_tool_definition() -> None:
    tool = WebSearchTool()

    assert tool.get_open_ai_tool_definitions() == [{"type": "web_search"}]


def test_make_agent_event_publisher_emits_shell_tool_events() -> None:
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "shell_call",
                "id": "shell_1",
                "call_id": "call_1",
                "status": "in_progress",
                "action": {"command": ["echo", "hi"]},
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "shell_call",
                "id": "shell_1",
                "call_id": "call_1",
                "status": "completed",
                "action": {"command": ["echo", "hi"]},
            },
        }
    )
    publisher(
        {
            "type": "meshagent.handler.done",
            "item": {
                "type": "shell_call_output",
                "call_id": "call_1",
                "output": [
                    {
                        "outcome": {"type": "exit", "exit_code": 0},
                        "stdout": "hi\n",
                        "stderr": "",
                    }
                ],
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "local_shell_call",
                "id": "local_shell_1",
                "call_id": "call_2",
                "status": "in_progress",
                "action": {"command": "echo hi"},
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "local_shell_call",
                "id": "local_shell_1",
                "call_id": "call_2",
                "status": "completed",
                "action": {"command": "echo hi"},
            },
        }
    )
    publisher(
        {
            "type": "meshagent.handler.done",
            "item": {
                "type": "local_shell_call_output",
                "call_id": "call_2",
                "output": "hi\n",
            },
        }
    )

    assert [type(event) for event in published] == [
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]

    shell_pending = published[0]
    assert isinstance(shell_pending, AgentToolCallPending)
    assert shell_pending.namespace == "openai.responses"
    assert shell_pending.call_id == "call_1"
    assert shell_pending.toolkit == "openai"
    assert shell_pending.tool == "shell"
    assert shell_pending.arguments == {"action": {"command": ["echo", "hi"]}}

    shell_started = published[1]
    assert isinstance(shell_started, AgentToolCallStarted)
    assert shell_started.namespace == "openai.responses"
    assert shell_started.call_id == "call_1"
    assert shell_started.toolkit == "openai"
    assert shell_started.tool == "shell"
    assert shell_started.arguments == {"action": {"command": ["echo", "hi"]}}

    shell_ended = published[2]
    assert isinstance(shell_ended, AgentToolCallEnded)
    assert shell_ended.namespace == "openai.responses"
    assert shell_ended.call_id == "call_1"
    assert isinstance(shell_ended.result, JsonContent)
    assert shell_ended.result.json == {
        "output": [
            {
                "outcome": {"type": "exit", "exit_code": 0},
                "stdout": "hi\n",
                "stderr": "",
            }
        ]
    }

    local_shell_pending = published[3]
    assert isinstance(local_shell_pending, AgentToolCallPending)
    assert local_shell_pending.namespace == "openai.responses"
    assert local_shell_pending.call_id == "call_2"
    assert local_shell_pending.toolkit == "openai"
    assert local_shell_pending.tool == "local_shell"
    assert local_shell_pending.arguments == {"action": {"command": "echo hi"}}

    local_shell_started = published[4]
    assert isinstance(local_shell_started, AgentToolCallStarted)
    assert local_shell_started.namespace == "openai.responses"
    assert local_shell_started.call_id == "call_2"
    assert local_shell_started.toolkit == "openai"
    assert local_shell_started.tool == "local_shell"
    assert local_shell_started.arguments == {"action": {"command": "echo hi"}}

    local_shell_ended = published[5]
    assert isinstance(local_shell_ended, AgentToolCallEnded)
    assert local_shell_ended.namespace == "openai.responses"
    assert local_shell_ended.call_id == "call_2"
    assert isinstance(local_shell_ended.result, TextContent)
    assert local_shell_ended.result.text == "hi\n"


def test_make_agent_event_publisher_emits_compaction_without_openai_item_id() -> None:
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "response.output_item.added",
            "response_id": "resp-1",
            "sequence_number": 3,
            "item": {
                "type": "compaction",
                "encrypted_content": "opaque",
                "status": "in_progress",
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "response_id": "resp-1",
            "sequence_number": 4,
            "item": {
                "type": "compaction",
                "encrypted_content": "opaque",
                "status": "completed",
            },
        }
    )

    assert isinstance(published[0], AgentThreadEvent)
    assert published[0].type == AGENT_EVENT_THREAD_EVENT
    assert published[0].event["headline"] == "Compacting context"
    compacted = next(
        message for message in published if isinstance(message, AgentContextCompacted)
    )
    assert compacted.type == AGENT_EVENT_CONTEXT_COMPACTED
    assert compacted.checkpoint_id == "compaction:resp-1:4"
    assert compacted.messages == [
        {
            "id": "compaction:resp-1:4",
            "type": "compaction",
            "encrypted_content": "opaque",
            "status": "completed",
        }
    ]
    assert isinstance(published[-1], AgentThreadEvent)
    assert published[-1].event["state"] == "completed"


def test_make_agent_event_publisher_emits_compaction_from_completed_response_snapshot() -> (
    None
):
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "output": [
                    {
                        "type": "compaction",
                        "encrypted_content": "opaque",
                        "status": "completed",
                    }
                ],
            },
        }
    )

    compacted = next(
        message for message in published if isinstance(message, AgentContextCompacted)
    )
    assert compacted.type == AGENT_EVENT_CONTEXT_COMPACTED
    assert compacted.messages == [
        {
            "id": "output:0",
            "type": "compaction",
            "encrypted_content": "opaque",
            "status": "completed",
        }
    ]


def test_make_agent_event_publisher_updates_shell_tool_arguments_before_handler_completion() -> (
    None
):
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "shell_call",
                "id": "shell_1",
                "call_id": "call_1",
                "status": "in_progress",
                "action": {},
            },
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "shell_call",
                "id": "shell_1",
                "call_id": "call_1",
                "status": "completed",
                "action": {
                    "command": [
                        "mkdir",
                        "-p",
                        "/data/docs2",
                        "&&",
                        "cat",
                        ">",
                        "/data/docs2/index.html",
                    ]
                },
            },
        }
    )
    publisher(
        {
            "type": "meshagent.handler.done",
            "item": {
                "type": "shell_call_output",
                "call_id": "call_1",
                "output": [
                    {
                        "outcome": {"type": "exit", "exit_code": 0},
                        "stdout": "",
                        "stderr": "",
                    }
                ],
            },
        }
    )

    assert [type(event) for event in published] == [
        AgentToolCallPending,
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]

    initial_pending = published[0]
    assert isinstance(initial_pending, AgentToolCallPending)
    assert initial_pending.arguments == {"action": {}}

    updated_pending = published[1]
    assert isinstance(updated_pending, AgentToolCallPending)
    assert updated_pending.arguments == {
        "action": {
            "command": [
                "mkdir",
                "-p",
                "/data/docs2",
                "&&",
                "cat",
                ">",
                "/data/docs2/index.html",
            ]
        }
    }

    started = published[2]
    assert isinstance(started, AgentToolCallStarted)
    assert started.arguments == updated_pending.arguments

    ended = published[3]
    assert isinstance(ended, AgentToolCallEnded)
    assert ended.item_id == "shell_1"


def test_make_agent_event_publisher_ends_shell_tool_when_handler_done_arrives_without_output_item_done() -> (
    None
):
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "shell_call",
                "id": "shell_1",
                "call_id": "call_1",
                "status": "in_progress",
                "action": {"command": ["sleep", "5"]},
            },
        }
    )
    publisher(
        {
            "type": "meshagent.handler.done",
            "item": {
                "type": "shell_call_output",
                "call_id": "call_1",
                "output": [
                    {
                        "outcome": {"type": "timeout"},
                        "stdout": "",
                        "stderr": "",
                    }
                ],
            },
        }
    )

    assert [type(event) for event in published] == [
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]
    pending = published[0]
    assert isinstance(pending, AgentToolCallPending)
    assert pending.item_id == "shell_1"

    started = published[1]
    assert isinstance(started, AgentToolCallStarted)
    assert started.item_id == "shell_1"

    ended = published[2]
    assert isinstance(ended, AgentToolCallEnded)
    assert ended.item_id == "shell_1"
    assert isinstance(ended.result, JsonContent)
    assert ended.result.json == {
        "output": [
            {
                "outcome": {"type": "timeout"},
                "stdout": "",
                "stderr": "",
            }
        ]
    }


def test_make_agent_event_publisher_emits_tool_log_delta() -> None:
    adapter = OpenAIResponsesAdapter(
        client=_FakeOpenAIClient(outcomes=[]),
        mode="request",
    )
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "meshagent.handler.output",
            "item_id": "shell_1",
            "lines": [
                {"source": "stdout", "text": "one"},
                {"source": "stderr", "text": "two"},
            ],
        }
    )

    assert len(published) == 1
    log_delta = published[0]
    assert isinstance(log_delta, AgentToolCallLogDelta)
    assert log_delta.item_id == "shell_1"
    assert [(line.source, line.text) for line in log_delta.lines] == [
        ("stdout", "one"),
        ("stderr", "two"),
    ]


@pytest.mark.asyncio
async def test_shell_tool_container_exec_emits_live_output_events() -> None:
    room = _FakeContainerRoom(
        exec_factory=lambda: _FakeContainerExec(
            stdout_chunks=[b"one\n", b"three\n"],
            stderr_chunks=[b"two\n"],
            exit_code=0,
        )
    )
    tool = ShellTool(room=room, image="meshagent/python:default")
    emitted_events: list[dict[str, object]] = []
    context = ToolContext(
        caller=_FakeParticipant(),
        event_handler=emitted_events.append,
    )

    result = await tool.execute_shell_command(
        context,
        commands=["echo hi"],
        item_id="shell-1",
        timeout_ms=5000,
    )

    assert room.containers.run_calls[0]["image"] == "meshagent/python:default"
    assert room.containers.exec_calls[0]["command"] == ["bash", "-lc", "echo hi"]
    assert result == [
        {
            "outcome": {"type": "exit", "exit_code": 0},
            "stdout": "one\nthree\n",
            "stderr": "two\n",
        }
    ]
    assert emitted_events == [
        {
            "type": "meshagent.handler.output",
            "item_id": "shell-1",
            "lines": [{"source": "stdout", "text": "one"}],
        },
        {
            "type": "meshagent.handler.output",
            "item_id": "shell-1",
            "lines": [{"source": "stderr", "text": "two"}],
        },
        {
            "type": "meshagent.handler.output",
            "item_id": "shell-1",
            "lines": [{"source": "stdout", "text": "three"}],
        },
    ]


@pytest.mark.asyncio
async def test_shell_tool_container_exec_truncates_success_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(responses_adapter_module, "MAX_SHELL_OUTPUT_SIZE", 8)

    room = _FakeContainerRoom(
        exec_factory=lambda: _FakeContainerExec(
            stdout_chunks=[b"abcdefghijk"],
            stderr_chunks=[],
            exit_code=0,
        )
    )
    tool = ShellTool(room=room, image="meshagent/python:default")
    emitted_events: list[dict[str, object]] = []
    context = ToolContext(
        caller=_FakeParticipant(),
        event_handler=emitted_events.append,
    )

    result = await tool.execute_shell_command(
        context,
        commands=["echo hi"],
        item_id="shell-1",
        timeout_ms=5000,
    )

    assert result == [
        {
            "outcome": {"type": "exit", "exit_code": 0},
            "stdout": "abcdefgh\n\n[output truncated after 8 characters]",
            "stderr": "",
        }
    ]
    assert emitted_events == [
        {
            "type": "meshagent.handler.output",
            "item_id": "shell-1",
            "lines": [
                {"source": "stdout", "text": "abcdefgh"},
                {
                    "source": "stdout",
                    "text": "[output truncated after 8 characters]",
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_shell_tool_container_exec_uses_configured_working_dir() -> None:
    room = _FakeContainerRoom(
        exec_factory=lambda: _FakeContainerExec(
            stdout_chunks=[b"/workspace\n"],
            stderr_chunks=[],
            exit_code=0,
        )
    )
    tool = ShellTool(
        room=room,
        image="meshagent/python:default",
        working_dir="/workspace",
    )
    context = ToolContext(
        caller=_FakeParticipant(),
        event_handler=lambda event: None,
    )

    result = await tool.execute_shell_command(
        context,
        commands=["pwd"],
        item_id="shell-1",
        timeout_ms=5000,
    )

    assert room.containers.exec_calls[0]["command"] == [
        "bash",
        "-lc",
        "cd /workspace && pwd",
    ]
    assert result == [
        {
            "outcome": {"type": "exit", "exit_code": 0},
            "stdout": "/workspace\n",
            "stderr": "",
        }
    ]


@pytest.mark.asyncio
async def test_shell_tool_local_exec_truncates_success_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(responses_adapter_module, "MAX_SHELL_OUTPUT_SIZE", 8)

    tool = ShellTool(image=None)
    emitted_events: list[dict[str, object]] = []
    context = ToolContext(
        caller=_FakeParticipant(),
        event_handler=emitted_events.append,
    )

    result = await tool.execute_shell_command(
        context,
        commands=["printf 'abcdefghijk'"],
        item_id="shell-1",
        timeout_ms=5000,
    )

    assert result == [
        {
            "outcome": {"type": "exit", "exit_code": 0},
            "stdout": "abcdefgh\n\n[output truncated after 8 characters]",
            "stderr": "",
        }
    ]
    assert emitted_events == [
        {
            "type": "meshagent.handler.output",
            "item_id": "shell-1",
            "lines": [
                {"source": "stdout", "text": "abcdefgh"},
                {
                    "source": "stdout",
                    "text": "[output truncated after 8 characters]",
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_shell_tool_local_exec_uses_configured_env() -> None:
    tool = ShellTool(image=None, env={"EXAMPLE_VAR": "hello"})
    emitted_events: list[dict[str, object]] = []
    context = ToolContext(
        caller=_FakeParticipant(),
        event_handler=emitted_events.append,
    )

    result = await tool.execute_shell_command(
        context,
        commands=["printf '%s' \"$EXAMPLE_VAR\""],
        item_id="shell-1",
        timeout_ms=5000,
    )

    assert result == [
        {
            "outcome": {"type": "exit", "exit_code": 0},
            "stdout": "hello",
            "stderr": "",
        }
    ]
    assert emitted_events == [
        {
            "type": "meshagent.handler.output",
            "item_id": "shell-1",
            "lines": [{"source": "stdout", "text": "hello"}],
        }
    ]
