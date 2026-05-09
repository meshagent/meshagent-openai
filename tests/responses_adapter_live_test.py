import asyncio
import base64
import copy
import json
import os
import struct
import zlib

import aiohttp
import pytest
from openai import AsyncOpenAI
from openai.types.responses.response_computer_tool_call import ResponseComputerToolCall

from meshagent.agents.messages import (
    AGENT_MESSAGE_TURN_START,
    AgentMessage,
    AgentToolCallArgumentsDelta,
    ToolChoice,
    TurnStart,
)
from meshagent.agents.process import AgentSupervisor, LLMAgentProcess, Message
from meshagent.computers.agent import ComputerToolkit
from meshagent.computers.operator import Operator
from meshagent.openai.tools.responses_adapter import (
    ApplyPatchTool,
    OpenAIResponsesAdapter,
    ShellTool,
)
from meshagent.tools import FunctionTool, Toolkit
from meshagent.tools.storage import StorageToolkit, StorageToolLocalMount


def _should_run_live_openai_tests() -> bool:
    return (
        os.getenv("RUN_OPENAI_LIVE_TESTS") == "1"
        and isinstance(os.getenv("OPENAI_API_KEY"), str)
        and os.getenv("OPENAI_API_KEY", "").strip() != ""
    )


pytestmark = pytest.mark.skipif(
    not _should_run_live_openai_tests(),
    reason="set RUN_OPENAI_LIVE_TESTS=1 and OPENAI_API_KEY to run live OpenAI tests",
)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        len(data).to_bytes(4, "big")
        + chunk_type
        + data
        + zlib.crc32(chunk_type + data).to_bytes(4, "big")
    )


def _make_one_by_one_png_bytes() -> bytes:
    ihdr = _png_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0),
    )
    idat = _png_chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    iend = _png_chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


_ONE_BY_ONE_PNG_BYTES = _make_one_by_one_png_bytes()


class _FakeDeveloper:
    def log_nowait(self, **kwargs) -> None:
        del kwargs


class _FakeParticipant:
    id = "local"

    def get_attribute(self, key: str) -> str | None:
        if key == "name":
            return "agent"
        return None


class _FakeRoom:
    local_participant = _FakeParticipant()
    developer = _FakeDeveloper()


class _RecordingSupervisor(AgentSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)


class _RecordingThreadStatusPublisher:
    def __init__(self) -> None:
        self.turn_ids: list[str | None] = []
        self.pending_messages: list[list[dict[str, object]]] = []
        self.statuses: list[dict[str, object]] = []
        self.clear_count = 0

    async def set_thread_turn_id(self, *, turn_id: str | None) -> None:
        self.turn_ids.append(turn_id)

    async def set_pending_messages(
        self,
        *,
        pending_messages: list[dict[str, object]],
    ) -> None:
        self.pending_messages.append(pending_messages)

    async def set_thread_status(
        self,
        *,
        status: str | None,
        mode=None,
        pending_item_id: str | None = None,
        total_bytes: int | None = None,
        lines_added: int | None = None,
        lines_removed: int | None = None,
    ) -> None:
        self.statuses.append(
            {
                "status": status,
                "mode": mode,
                "pending_item_id": pending_item_id,
                "total_bytes": total_bytes,
                "lines_added": lines_added,
                "lines_removed": lines_removed,
            }
        )

    async def clear_thread_status(self) -> None:
        self.clear_count += 1


class _SteeringProbeTool(FunctionTool):
    def __init__(self, *, name: str = "steering_probe", wait_for_release: bool = True):
        super().__init__(
            name=name,
            input_schema={
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                },
                "required": ["note"],
                "additionalProperties": False,
            },
            description="A probe tool used to verify steering order across tool calls.",
        )
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[str] = []
        self._wait_for_release = wait_for_release

    async def execute(self, context, note: str) -> dict[str, object]:
        del context
        self.calls.append(note)
        self.started.set()
        if self._wait_for_release:
            await self.release.wait()
        return {"ok": True, "note": note}


def _event_type_summary(events: list[dict[str, object]]) -> list[str | None]:
    return [
        event.get("type") if isinstance(event.get("type"), str) else None
        for event in events
    ]


def _message_text(item: dict[str, object]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


async def _wait_until(predicate, *, interval: float = 0.05) -> None:
    while not await predicate():
        await asyncio.sleep(interval)


def _is_apply_patch_delta_event(event: dict[str, object]) -> bool:
    event_type = event.get("type")
    return (
        isinstance(event_type, str)
        and (
            event_type.startswith("response.apply_patch_call.")
            or event_type.startswith("response.apply_patch_call_")
        )
        and event_type.endswith(".delta")
    )


def _is_apply_patch_done_event(event: dict[str, object]) -> bool:
    if event.get("type") != "response.output_item.done":
        return False
    item = event.get("item")
    return isinstance(item, dict) and item.get("type") == "apply_patch_call"


class _RecordingOpenAIResponsesAdapter(OpenAIResponsesAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recorded_create_kwargs: list[dict[str, object]] = []
        self.recorded_response_outputs: list[list[dict[str, object]]] = []
        self.recorded_response_usages: list[dict[str, object] | None] = []

    async def _create_response_with_retries(self, *, openai, create_kwargs: dict):
        self.recorded_create_kwargs.append(copy.deepcopy(create_kwargs))
        response = await super()._create_response_with_retries(
            openai=openai,
            create_kwargs=create_kwargs,
        )
        self.recorded_response_outputs.append(
            [item.to_dict() for item in response.output]
        )
        self.recorded_response_usages.append(
            response.usage.to_dict() if response.usage is not None else None
        )
        return response

    async def _create_response_websocket_stream(
        self,
        *,
        context,
        openai,
        create_kwargs: dict,
        extra_headers: dict[str, str],
    ):
        self.recorded_create_kwargs.append(copy.deepcopy(create_kwargs))
        return await super()._create_response_websocket_stream(
            context=context,
            openai=openai,
            create_kwargs=create_kwargs,
            extra_headers=extra_headers,
        )


class _LiveBrowserComputer:
    environment = "browser"
    dimensions = (1440, 900)

    def __init__(self):
        self.calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, exc_tb):
        del exc_type
        del exc
        del exc_tb

    async def screenshot(self) -> str:
        self.calls.append("screenshot")
        return base64.b64encode(_ONE_BY_ONE_PNG_BYTES).decode("ascii")

    async def click(self, x: int, y: int, button: str = "left") -> None:
        del x
        del y
        del button
        self.calls.append("click")

    async def double_click(self, x: int, y: int) -> None:
        del x
        del y
        self.calls.append("double_click")

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        del x
        del y
        del scroll_x
        del scroll_y
        self.calls.append("scroll")

    async def type(self, text: str) -> None:
        del text
        self.calls.append("type")

    async def wait(self, ms: int = 1000) -> None:
        del ms
        self.calls.append("wait")

    async def move(self, x: int, y: int) -> None:
        del x
        del y
        self.calls.append("move")

    async def keypress(self, keys: list[str]) -> None:
        del keys
        self.calls.append("keypress")

    async def drag(self, path: list[dict[str, int]]) -> None:
        del path
        self.calls.append("drag")

    async def get_current_url(self) -> str:
        return "https://example.com"


@pytest.mark.asyncio
async def test_live_openai_auto_compaction_threshold_reports_compacted_next_call():
    threshold = int(os.getenv("OPENAI_COMPACTION_TEST_THRESHOLD", "1000"))
    model = os.getenv("OPENAI_COMPACTION_TEST_MODEL", "gpt-5.2")
    adapter = _RecordingOpenAIResponsesAdapter(
        model=model,
        mode="request",
        client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        context_management="auto",
        compaction_threshold=threshold,
        max_output_tokens=2048,
        max_retries=2,
    )
    context = adapter.create_session()
    memory_block = (
        "This is durable context used only for a compaction threshold probe. "
        "Keep it in mind, but do not repeat it. "
        + ("alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 80)
    )
    context.append_user_message("Store the following synthetic notes.")
    for index in range(20):
        context.append_assistant_message(f"Note block {index}: {memory_block}")
    context.append_user_message("Reply with OK only.")

    first_result = await asyncio.wait_for(
        adapter.next(
            context=context,
            caller=_FakeRoom().local_participant,
            toolkits=[],
        ),
        timeout=180.0,
    )

    assert isinstance(first_result, str)
    assert adapter.recorded_create_kwargs[0]["context_management"] == [
        {"type": "compaction", "compact_threshold": threshold}
    ]
    compaction_response_index = next(
        index
        for index, outputs in enumerate(adapter.recorded_response_outputs)
        if any(output.get("type") == "compaction" for output in outputs)
    )
    compaction_usage = adapter.recorded_response_usages[compaction_response_index]
    assert compaction_usage is not None
    assert context.metadata["last_response_compaction_threshold"] == threshold

    context.append_user_message("Reply with OK only.")
    second_result = await asyncio.wait_for(
        adapter.next(
            context=context,
            caller=_FakeRoom().local_participant,
            toolkits=[],
        ),
        timeout=180.0,
    )

    assert isinstance(second_result, str)
    second_usage = context.metadata["last_response_usage"]
    second_input_tokens = second_usage["input_tokens"]
    assert second_input_tokens <= threshold


@pytest.mark.asyncio
async def test_live_openai_adapter_receives_tool_preamble_message():
    room = _FakeRoom()
    tool = _SteeringProbeTool(name="commentary_probe", wait_for_release=False)
    events: list[dict[str, object]] = []
    adapter = OpenAIResponsesAdapter(
        model=os.getenv("OPENAI_PREAMBLE_TEST_MODEL", "gpt-5.2"),
        mode="request",
        client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        reasoning_effort=os.getenv("OPENAI_PREAMBLE_TEST_REASONING_EFFORT", "none"),
        max_output_tokens=2048,
        max_retries=2,
    )
    context = adapter.create_session()
    context.instructions = (
        "Before you call a tool, explain why you are calling it. Keep the explanation "
        "to one short sentence. Only the last user-facing answer should be final."
    )
    context.append_user_message(
        "Call commentary_probe exactly once with note='live-commentary'. After the tool "
        "returns, reply with exactly DONE and nothing else."
    )

    result = await asyncio.wait_for(
        adapter.next(
            context=context,
            caller=room.local_participant,
            toolkits=[Toolkit(name="test", tools=[tool])],
            event_handler=events.append,
        ),
        timeout=120.0,
    )

    function_call_index = next(
        (
            index
            for index, item in enumerate(context.previous_messages)
            if item.get("type") == "function_call"
        ),
        None,
    )
    preamble = (
        None
        if function_call_index is None
        else next(
            (
                item
                for item in reversed(context.previous_messages[:function_call_index])
                if item.get("type") == "message" and item.get("role") == "assistant"
            ),
            None,
        )
    )
    assert len(tool.calls) == 1
    assert isinstance(result, str)
    assert result.strip() == "DONE"
    assert preamble is not None, json.dumps(
        {
            "event_types": _event_type_summary(events),
            "previous_messages": context.previous_messages,
        },
        default=str,
    )
    assert _message_text(preamble).strip() != ""


@pytest.mark.asyncio
async def test_live_openai_process_increments_preparing_command_byte_status():
    room = _FakeRoom()
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    adapter = OpenAIResponsesAdapter(
        model=os.getenv("OPENAI_SHELL_COUNTER_TEST_MODEL", "gpt-5.5"),
        mode=os.getenv("OPENAI_SHELL_COUNTER_TEST_MODE", "websocket"),
        client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        reasoning_effort=os.getenv(
            "OPENAI_SHELL_COUNTER_TEST_REASONING_EFFORT", "none"
        ),
        max_output_tokens=2048,
        max_retries=1,
    )
    process = LLMAgentProcess(
        thread_id="thread-1",
        participant=room.local_participant,
        llm_adapter=adapter,
        toolkits=[Toolkit(name="openai", tools=[ShellTool(image=None)])],
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        literal_payload = "x" * 512
        command = (
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            f"Path('/tmp/meshagent_live_counter_probe.txt').write_text('{literal_payload}')\n"
            "print('DONE')\n"
            "PY"
        )
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Use the shell tool exactly once with exactly this command:\n"
                                f"{command}\n"
                                "After the shell command finishes, reply with exactly DONE."
                            ),
                        }
                    ],
                    tool_choice=ToolChoice(toolkit_name="openai", tool_name="shell"),
                )
            )
        )

        async def saw_shell_argument_byte_status() -> bool:
            totals = [
                status["total_bytes"]
                for status in publisher.statuses
                if status.get("status") == "Preparing"
                and status.get("pending_item_id") is not None
                and isinstance(status.get("total_bytes"), int)
            ]
            return len(totals) >= 2 and max(totals) > min(totals) and max(totals) > 100

        try:
            await asyncio.wait_for(
                _wait_until(saw_shell_argument_byte_status),
                timeout=120.0,
            )
        except TimeoutError as exc:
            raise AssertionError(
                json.dumps(
                    {
                        "statuses": publisher.statuses,
                        "sent_types": [
                            message.data.type for message in supervisor.sent
                        ],
                    },
                    default=str,
                    indent=2,
                )
            ) from exc
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_live_openai_apply_patch_streams_patch_deltas_before_done(tmp_path):
    report_path = tmp_path / "report.py"
    report_path.write_text(
        "def main():\n    print('hello')\n\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    storage = StorageToolkit(
        mounts=[
            StorageToolLocalMount(
                path="/",
                local_path=str(tmp_path),
            )
        ]
    )
    events: list[dict[str, object]] = []
    agent_messages: list[AgentMessage] = []
    adapter = OpenAIResponsesAdapter(
        model=os.getenv("OPENAI_APPLY_PATCH_DELTA_TEST_MODEL", "gpt-5.5"),
        mode=os.getenv("OPENAI_APPLY_PATCH_DELTA_TEST_MODE", "websocket"),
        client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        reasoning_effort=os.getenv(
            "OPENAI_APPLY_PATCH_DELTA_TEST_REASONING_EFFORT", "none"
        ),
        max_output_tokens=2048,
        max_retries=1,
    )
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-live-apply-patch",
        thread_id="thread-live-apply-patch",
        callback=agent_messages.append,
    )
    context = adapter.create_session()
    context.instructions = (
        "Use the apply_patch tool exactly once when asked to edit files. "
        "Do not call any other tools."
    )
    patch = (
        "@@\n"
        " def main():\n"
        "+    # Keep the example output small for the live streaming probe.\n"
        "     print('hello')\n"
    )
    context.append_user_message(
        "Use the apply_patch tool exactly once to update report.py with this "
        "exact operation, then reply DONE:\n"
        "{"
        '"type":"update_file",'
        '"path":"report.py",'
        f'"diff":{json.dumps(patch)}'
        "}"
    )

    def record_event(event: dict[str, object]) -> None:
        events.append(event)
        publisher(event)

    next_task = asyncio.create_task(
        adapter.next(
            context=context,
            caller=_FakeRoom().local_participant,
            toolkits=[Toolkit(name="openai", tools=[ApplyPatchTool(storage=storage)])],
            tool_choice=ToolChoice(toolkit_name="openai", tool_name="apply_patch"),
            event_handler=record_event,
        )
    )

    async def saw_apply_patch_done() -> bool:
        return any(_is_apply_patch_done_event(event) for event in events)

    try:
        try:
            await asyncio.wait_for(
                _wait_until(saw_apply_patch_done),
                timeout=120.0,
            )
        except TimeoutError as exc:
            raise AssertionError(
                json.dumps(
                    {
                        "event_types": _event_type_summary(events),
                        "patch_event_types": [
                            event.get("type")
                            for event in events
                            if isinstance(event.get("type"), str)
                            and str(event.get("type")).startswith(
                                "response.apply_patch_call"
                            )
                        ],
                    },
                    default=str,
                    indent=2,
                )
            ) from exc
    finally:
        next_task.cancel()
        await asyncio.gather(next_task, return_exceptions=True)
        await context.close()

    delta_indices = [
        index
        for index, event in enumerate(events)
        if _is_apply_patch_delta_event(event)
    ]
    done_index = next(
        (
            index
            for index, event in enumerate(events)
            if _is_apply_patch_done_event(event)
        ),
        None,
    )
    accumulated_delta = "".join(
        str(events[index].get("delta", "")) for index in delta_indices
    )
    agent_argument_deltas = [
        message
        for message in agent_messages
        if isinstance(message, AgentToolCallArgumentsDelta)
    ]
    assert len(delta_indices) >= 2, json.dumps(
        {
            "event_types": _event_type_summary(events),
            "patch_event_types": [
                event.get("type")
                for event in events
                if isinstance(event.get("type"), str)
                and str(event.get("type")).startswith("response.apply_patch_call")
            ],
            "apply_patch_done_index": done_index,
        },
        default=str,
        indent=2,
    )
    assert len(agent_argument_deltas) >= 2, json.dumps(
        {
            "event_types": _event_type_summary(events),
            "agent_message_types": [
                message.__class__.__name__ for message in agent_messages
            ],
        },
        default=str,
        indent=2,
    )
    assert done_index is not None, json.dumps(_event_type_summary(events), indent=2)
    assert max(delta_indices) < done_index, json.dumps(
        {
            "delta_indices": delta_indices,
            "apply_patch_done_index": done_index,
            "event_types": _event_type_summary(events),
        },
        indent=2,
    )
    assert "report.py" in accumulated_delta or "print('hello')" in accumulated_delta
    assert (
        "report.py" in "".join(message.delta for message in agent_argument_deltas)
        or "print('hello')"
        in "".join(message.delta for message in agent_argument_deltas)
    )
    assert "Keep the example output small" in report_path.read_text(encoding="utf-8")


async def _read_websocket_payload(
    websocket: aiohttp.ClientWebSocketResponse,
) -> dict[str, object]:
    message = await websocket.receive()
    if message.type != aiohttp.WSMsgType.TEXT:
        raise AssertionError(f"unexpected websocket message type: {message.type}")

    payload = json.loads(message.data)
    if not isinstance(payload, dict):
        raise AssertionError(f"unexpected websocket payload: {payload!r}")

    return payload


async def _collect_concurrent_response_summary(
    *,
    adapter: OpenAIResponsesAdapter,
    openai: AsyncOpenAI,
    model: str,
    conversation: str | None,
) -> dict[str, object]:
    context = adapter.create_session()
    websocket = await context.ensure_websocket(
        url=adapter._http_base_url_to_ws_responses_url(str(openai.base_url)),
        headers=adapter._websocket_headers(openai=openai, extra_headers={}),
    )

    prompts = [
        "Output ALPHA-001 through ALPHA-120, separated by spaces, with no intro or outro.",
        "Output BETA-001 through BETA-120, separated by spaces, with no intro or outro.",
    ]

    try:
        for index, prompt in enumerate(prompts, start=1):
            create_kwargs: dict[str, object] = {
                "stream": True,
                "model": model,
                "input": [{"role": "user", "content": prompt}],
                "max_output_tokens": 256,
                "metadata": {"probe": f"request-{index}"},
            }
            if conversation is not None:
                create_kwargs["conversation"] = conversation

            payload = adapter._build_websocket_request_payload(create_kwargs)
            await websocket.send_str(json.dumps(payload))

        event_sequence: list[dict[str, object | None]] = []
        response_ids: list[str] = []
        terminal_response_ids: list[str] = []
        errors: list[dict[str, object]] = []

        while True:
            try:
                payload = await asyncio.wait_for(
                    _read_websocket_payload(websocket),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                break

            payload_type = payload.get("type")
            response_id = adapter._response_id_from_payload(payload)
            event_sequence.append(
                {
                    "type": payload_type if isinstance(payload_type, str) else None,
                    "response_id": response_id,
                }
            )

            if payload_type == "error":
                errors.append(payload)
                break

            if payload_type == "response.created" and response_id is not None:
                if response_id not in response_ids:
                    response_ids.append(response_id)

            if (
                adapter._is_terminal_response_payload(payload)
                and response_id is not None
            ):
                if response_id not in terminal_response_ids:
                    terminal_response_ids.append(response_id)
                if len(terminal_response_ids) >= 2:
                    break

        first_terminal_index = next(
            (
                index
                for index, event in enumerate(event_sequence)
                if event["type"]
                in {
                    "response.completed",
                    "response.done",
                    "response.failed",
                    "response.incomplete",
                }
            ),
            None,
        )
        if first_terminal_index is None:
            interleaved_before_first_terminal = False
        else:
            seen_before_first_terminal = {
                event["response_id"]
                for event in event_sequence[: first_terminal_index + 1]
                if isinstance(event["response_id"], str)
            }
            interleaved_before_first_terminal = len(seen_before_first_terminal) > 1

        return {
            "conversation": conversation or "default",
            "response_ids": response_ids,
            "terminal_response_ids": terminal_response_ids,
            "error_count": len(errors),
            "errors": errors,
            "interleaved_before_first_terminal": interleaved_before_first_terminal,
            "event_sequence_head": event_sequence[:20],
            "event_count": len(event_sequence),
        }
    finally:
        await context.close()


@pytest.mark.asyncio
async def test_openai_gpt_54_returns_live_computer_call():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await client.responses.create(
        model="gpt-5.4",
        input=(
            "Use the computer tool exactly once to take a screenshot and nothing else."
        ),
        tools=[{"type": "computer"}],
        max_output_tokens=256,
    )

    computer_calls = [
        item for item in response.output if isinstance(item, ResponseComputerToolCall)
    ]

    assert len(computer_calls) >= 1
    first_call = computer_calls[0]
    assert isinstance(first_call.call_id, str)
    assert first_call.call_id != ""
    assert isinstance(first_call.actions, list)
    assert len(first_call.actions) >= 1
    assert isinstance(first_call.actions[0], dict)
    assert isinstance(first_call.actions[0].get("type"), str)


@pytest.mark.asyncio
async def test_openai_gpt_54_adapter_uses_native_computer_tool():
    room = _FakeRoom()
    computer = _LiveBrowserComputer()
    toolkit = ComputerToolkit(
        computer=computer,
        operator=Operator(),
        caller=room.local_participant,
        render_screen=None,
    )
    adapter = OpenAIResponsesAdapter(
        model="gpt-5.4",
        mode="websocket",
        client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        max_output_tokens=256,
    )
    context = adapter.create_session()
    context.append_user_message(
        "Use the computer tool exactly once to take a screenshot, then reply with DONE."
    )

    try:
        result = await asyncio.wait_for(
            adapter.next(
                context=context,
                caller=room.local_participant,
                toolkits=[toolkit],
            ),
            timeout=30.0,
        )
    finally:
        await context.close()

    assert isinstance(result, str)
    assert result.strip() != ""
    assert "screenshot" in computer.calls
    assert "goto" not in computer.calls
    assert any(
        isinstance(item, dict) and item.get("type") == "computer_call"
        for item in context.previous_messages
    )
    assert not any(
        isinstance(item, dict) and item.get("type") == "function_call"
        for item in context.previous_messages
    )


@pytest.mark.asyncio
async def test_explore_openai_responses_websocket_concurrent_requests():
    model = os.getenv("OPENAI_CONCURRENCY_MODEL", "gpt-5.2")
    openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    adapter = OpenAIResponsesAdapter(
        model=model,
        mode="websocket",
        client=openai,
        max_output_tokens=256,
    )

    default_summary = await _collect_concurrent_response_summary(
        adapter=adapter,
        openai=openai,
        model=model,
        conversation=None,
    )
    out_of_band_summary = await _collect_concurrent_response_summary(
        adapter=adapter,
        openai=openai,
        model=model,
        conversation="none",
    )

    print(
        json.dumps(
            {
                "default": default_summary,
                "conversation_none": out_of_band_summary,
            },
            indent=2,
        )
    )

    assert default_summary["event_count"] > 0
    assert out_of_band_summary["event_count"] > 0


@pytest.mark.asyncio
async def test_live_openai_responses_inserts_steer_immediately_after_tool_boundary():
    room = _FakeRoom()
    tool = _SteeringProbeTool()
    adapter = _RecordingOpenAIResponsesAdapter(
        model=os.getenv("OPENAI_STEERING_TEST_MODEL", "gpt-5.4"),
        mode="request",
        client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        max_output_tokens=512,
    )
    context = adapter.create_session()
    context.append_user_message(
        "You must call the steering_probe tool exactly once with any note string. "
        "After the tool result is available, reply with exactly ORIGINAL and nothing else."
    )

    pending_steer = False

    async def _steer() -> bool:
        nonlocal pending_steer
        if not pending_steer:
            return False
        pending_steer = False
        context.append_user_message(
            "New instruction: after the tool result, reply with exactly STEERED and nothing else."
        )
        return True

    task = asyncio.create_task(
        adapter.next(
            context=context,
            caller=room.local_participant,
            toolkits=[Toolkit(name="test", tools=[tool])],
            event_handler=lambda event: None,
            steering_callback=_steer,
        )
    )

    await asyncio.wait_for(tool.started.wait(), timeout=30.0)
    pending_steer = True
    tool.release.set()
    result = await asyncio.wait_for(task, timeout=90.0)

    assert tool.calls
    assert "STEERED" in result
    assert "ORIGINAL" not in result
    assert len(adapter.recorded_create_kwargs) >= 2

    second_request = adapter.recorded_create_kwargs[1]
    second_input = second_request["input"]
    assert isinstance(second_input, list)
    assert len(second_input) >= 2
    assert second_input[-2]["type"] == "function_call_output"
    assert second_input[-1] == {
        "role": "user",
        "content": (
            "New instruction: after the tool result, reply with exactly STEERED "
            "and nothing else."
        ),
    }
    assert not any(
        isinstance(item, dict)
        and item.get("role") == "assistant"
        and isinstance(item.get("content"), str)
        and "ORIGINAL" in item["content"]
        for item in second_input
    )


@pytest.mark.asyncio
async def test_live_openai_websocket_inserts_steer_immediately_after_tool_boundary():
    room = _FakeRoom()
    tool = _SteeringProbeTool()
    adapter = _RecordingOpenAIResponsesAdapter(
        model=os.getenv("OPENAI_STEERING_TEST_MODEL", "gpt-5.4"),
        mode="websocket",
        client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        max_output_tokens=512,
    )
    context = adapter.create_session()
    context.append_user_message(
        "You must call the steering_probe tool exactly once with any note string. "
        "After the tool result is available, reply with exactly ORIGINAL and nothing else."
    )

    pending_steer = False

    async def _steer() -> bool:
        nonlocal pending_steer
        if not pending_steer:
            return False
        pending_steer = False
        context.append_user_message(
            "New instruction: after the tool result, reply with exactly STEERED and nothing else."
        )
        return True

    task = asyncio.create_task(
        adapter.next(
            context=context,
            caller=room.local_participant,
            toolkits=[Toolkit(name="test", tools=[tool])],
            event_handler=lambda event: None,
            steering_callback=_steer,
        )
    )

    try:
        await asyncio.wait_for(tool.started.wait(), timeout=30.0)
        pending_steer = True
        tool.release.set()
        result = await asyncio.wait_for(task, timeout=90.0)
    finally:
        await context.close()

    assert tool.calls
    assert "STEERED" in result
    assert "ORIGINAL" not in result
    assert len(adapter.recorded_create_kwargs) >= 2

    second_request = adapter.recorded_create_kwargs[1]
    second_input = second_request["input"]
    assert isinstance(second_input, list)
    assert len(second_input) >= 2
    assert second_input[-2]["type"] == "function_call_output"
    assert second_input[-1] == {
        "role": "user",
        "content": (
            "New instruction: after the tool result, reply with exactly STEERED "
            "and nothing else."
        ),
    }
    assert not any(
        isinstance(item, dict)
        and item.get("role") == "assistant"
        and isinstance(item.get("content"), str)
        and "ORIGINAL" in item["content"]
        for item in second_input
    )


@pytest.mark.asyncio
async def test_live_openai_request_inserts_steer_after_first_completed_tool_when_two_tools_are_scheduled():
    room = _FakeRoom()
    alpha_tool = _SteeringProbeTool(name="alpha_probe", wait_for_release=True)
    beta_tool = _SteeringProbeTool(name="beta_probe", wait_for_release=False)
    adapter = _RecordingOpenAIResponsesAdapter(
        model=os.getenv("OPENAI_STEERING_TEST_MODEL", "gpt-5.4"),
        mode="request",
        client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        max_output_tokens=512,
    )
    context = adapter.create_session()
    context.append_user_message(
        "You must call alpha_probe and beta_probe exactly once each before any natural language. "
        "After the tool results are available, reply with exactly ORIGINAL and nothing else."
    )

    pending_steer = False

    async def _steer() -> bool:
        nonlocal pending_steer
        if not pending_steer:
            return False
        pending_steer = False
        context.append_user_message(
            "New instruction: after the tool result, reply with exactly STEERED and nothing else."
        )
        return True

    task = asyncio.create_task(
        adapter.next(
            context=context,
            caller=room.local_participant,
            toolkits=[Toolkit(name="test", tools=[alpha_tool, beta_tool])],
            event_handler=lambda event: None,
            steering_callback=_steer,
        )
    )

    await asyncio.wait_for(alpha_tool.started.wait(), timeout=30.0)
    pending_steer = True
    alpha_tool.release.set()
    result = await asyncio.wait_for(task, timeout=90.0)

    assert "STEERED" in result
    assert "ORIGINAL" not in result
    assert len(adapter.recorded_response_outputs) >= 1
    first_response_output = adapter.recorded_response_outputs[0]
    assert any(
        item.get("type") == "function_call" and item.get("name") == "alpha_probe"
        for item in first_response_output
    )
    assert any(
        item.get("type") == "function_call" and item.get("name") == "beta_probe"
        for item in first_response_output
    )
    assert beta_tool.calls == []
