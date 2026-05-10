import asyncio
import os

import pytest

from meshagent.openai.tools.realtime_adapter import OpenAIRealtimeAdapter


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


class _FakeParticipant:
    id = "local"

    def get_attribute(self, key: str) -> str | None:
        if key == "name":
            return "agent"
        return None


@pytest.mark.asyncio
async def test_live_realtime_create_response_streams_text_deltas() -> None:
    adapter = OpenAIRealtimeAdapter(
        model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        base_url=os.getenv("OPENAI_REALTIME_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        session_options={
            "instructions": "You are a test assistant. Keep responses short.",
            "output_modalities": ["text"],
        },
        response_options={
            "instructions": "Reply with exactly the lowercase word: pong",
            "output_modalities": ["text"],
        },
    )
    context = adapter.create_session()
    events: list[dict[str, object]] = []

    await adapter.connect(context=context, event_handler=events.append)
    try:
        terminal_event = await asyncio.wait_for(
            adapter.create_response(
                context=context,
                caller=_FakeParticipant(),
                toolkits=[],
                event_handler=events.append,
            ),
            timeout=60,
        )
    finally:
        await adapter.disconnect(context=context)

    terminal_type = terminal_event.get("type")
    assert terminal_type in {"response.done", "response.completed"}

    event_types = [
        event_type
        for event in events
        if isinstance((event_type := event.get("type")), str)
    ]
    assert "response.create" not in event_types
    assert any(event_type.endswith(".delta") for event_type in event_types)
    assert any(
        event_type
        in {
            "response.output_text.delta",
            "response.text.delta",
            "response.audio_transcript.delta",
            "response.output_audio_transcript.delta",
        }
        for event_type in event_types
    )
    first_delta_index = event_types.index(
        next(event_type for event_type in event_types if event_type.endswith(".delta"))
    )
    terminal_index = (
        event_types.index(terminal_type)
        if isinstance(terminal_type, str) and terminal_type in event_types
        else len(event_types)
    )
    assert first_delta_index < terminal_index
