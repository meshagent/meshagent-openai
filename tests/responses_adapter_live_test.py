import asyncio
import base64
import json
import os
import struct
import zlib

import aiohttp
import pytest
from openai import AsyncOpenAI
from openai.types.responses.response_computer_tool_call import ResponseComputerToolCall

from meshagent.computers.agent import ComputerToolkit
from meshagent.computers.operator import Operator
from meshagent.openai.tools.responses_adapter import OpenAIResponsesAdapter


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
        room=room,
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
                room=room,
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
