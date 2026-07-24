import asyncio
import io
import os
from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI
from PIL import Image

from meshagent.agents.messages import (
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_MESSAGE_TURN_START,
    AgentToolCallEnded,
    TurnStart,
)
from meshagent.agents.process import AgentSupervisor, LLMAgentProcess, Message
from meshagent.api import Participant
from meshagent.api.messaging import JsonContent
from meshagent.openai.tools.image_generation import ImageGenerationToolkit
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


class _LiveImagesDataset:
    def __init__(self) -> None:
        self.records: dict[str, SimpleNamespace] = {}

    async def read_record(self, *, image_id: str):
        return self.records.get(image_id)

    async def save(
        self,
        *,
        data: bytes,
        mime_type: str,
        created_by: str,
        annotations: dict,
    ):
        assert created_by == "agent"
        assert isinstance(annotations.get("prompt"), str)
        image_id = f"generated-{len(self.records) + 1}"
        record = SimpleNamespace(id=image_id, data=data, mime_type=mime_type)
        self.records[image_id] = record
        return record

    async def delete(self, *, image_id: str):
        return self.records.pop(image_id, None) is not None


class _UnusedStorage:
    async def read_file(self, *, path: str):
        raise AssertionError(f"unexpected storage read: {path}")

    async def write_bytes(self, *, path: str, data: bytes, overwrite: bool):
        raise AssertionError(f"unexpected storage write: {path}")


class _RecordingSupervisor(AgentSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)


async def _wait_until(predicate, *, interval: float = 0.05) -> None:
    while not predicate():
        await asyncio.sleep(interval)


@pytest.mark.asyncio
async def test_live_agent_generates_and_saves_real_png() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    images = _LiveImagesDataset()
    participant = Participant(id="agent", attributes={"name": "agent"})
    client = AsyncOpenAI(api_key=api_key)
    adapter = OpenAIResponsesAdapter(
        model=os.getenv("OPENAI_IMAGE_AGENT_TEST_MODEL", "gpt-5.4"),
        mode="request",
        client=client,
        reasoning_effort="none",
        max_output_tokens=512,
        max_retries=1,
    )
    toolkit = ImageGenerationToolkit(
        images_dataset=images,
        storage_toolkit=_UnusedStorage(),
        model=os.getenv("OPENAI_IMAGE_TEST_MODEL", "gpt-image-2"),
        client=client,
    )
    supervisor = _RecordingSupervisor()
    process = LLMAgentProcess(
        thread_id="image-generation-live",
        participant=participant,
        llm_adapter=adapter,
        toolkits=[toolkit],
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="image-generation-live",
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Generate one simple square icon of a cheerful yellow duck "
                                "on a plain white background. Call imagegen exactly once with "
                                "referenced_image_ids=[]; after it succeeds, reply exactly DONE."
                            ),
                        }
                    ],
                )
            )
        )

        await asyncio.wait_for(
            _wait_until(
                lambda: any(
                    message.data.type == AGENT_EVENT_TURN_ENDED
                    for message in supervisor.sent
                )
            ),
            timeout=300.0,
        )

        turn_ended = next(
            message.data
            for message in reversed(supervisor.sent)
            if message.data.type == AGENT_EVENT_TURN_ENDED
        )
        assert turn_ended.error is None
        tool_ended = [
            message.data
            for message in supervisor.sent
            if message.data.type == AGENT_EVENT_TOOL_CALL_ENDED
        ]
        assert len(tool_ended) == 1
        assert isinstance(tool_ended[0], AgentToolCallEnded)
        assert tool_ended[0].error is None
        assert isinstance(tool_ended[0].result, JsonContent)
        saved_image_id = tool_ended[0].result.json["saved_image_id"]
        record = images.records[saved_image_id]
        assert record.mime_type == "image/png"
        assert record.data.startswith(b"\x89PNG\r\n\x1a\n")
        with Image.open(io.BytesIO(record.data)) as generated:
            assert generated.format == "PNG"
            assert generated.width > 0
            assert generated.height > 0
    finally:
        await process.stop(supervisor)
        await client.close()
