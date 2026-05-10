import asyncio
import os
import time
import uuid
from typing import Any

import pyarrow as pa
import pytest

from meshagent.agents.dataset_thread_storage import DatasetThreadStorage
from meshagent.agents.messages import (
    AGENT_EVENT_THREAD_STATUS,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_MESSAGE_TURN_START,
    AgentAudioGenerationDelta,
    AgentMessage,
    AgentTextContentDelta,
    AgentThreadStatus,
    TurnStart,
    TurnStartAccepted,
    TurnStarted,
)
from meshagent.agents.process import AgentSupervisor, LLMAgentProcess, Message
from meshagent.agents.thread_status_publisher import AgentMessageThreadStatusPublisher
from meshagent.api import DatasetJson, Participant
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


class _FakeDatasets:
    def __init__(self) -> None:
        self.schemas: dict[tuple[tuple[str, ...], str], pa.Schema] = {}
        self.rows: dict[tuple[tuple[str, ...], str], list[dict[str, Any]]] = {}
        self.inserts: list[dict[str, Any]] = []

    @staticmethod
    def _key(
        *,
        table: str | None = None,
        name: str | None = None,
        namespace: list[str] | None,
    ) -> tuple[tuple[str, ...], str]:
        table_name = table if table is not None else name
        assert table_name is not None
        return (tuple(namespace or []), table_name)

    async def create_table_with_schema(
        self,
        *,
        name: str,
        schema: pa.Schema,
        mode: str,
        namespace: list[str] | None = None,
    ) -> None:
        del mode
        key = self._key(name=name, namespace=namespace)
        self.schemas.setdefault(key, schema)
        self.rows.setdefault(key, [])

    async def inspect(
        self,
        *,
        table: str,
        namespace: list[str] | None = None,
    ) -> pa.Schema:
        return self.schemas[self._key(table=table, namespace=namespace)]

    async def add_columns(
        self,
        *,
        table: str,
        new_columns: dict[str, pa.Field],
        namespace: list[str] | None = None,
    ) -> None:
        key = self._key(table=table, namespace=namespace)
        schema = self.schemas[key]
        self.schemas[key] = pa.schema([*schema, *new_columns.values()])

    async def search(
        self,
        *,
        table: str,
        namespace: list[str] | None = None,
        where: str | dict[str, Any] | None = None,
        limit: int | None = None,
        select: list[str] | None = None,
    ) -> pa.Table:
        key = self._key(table=table, namespace=namespace)
        rows = list(self.rows.get(key, []))
        if isinstance(where, dict):
            rows = [
                row
                for row in rows
                if all(row.get(field) == value for field, value in where.items())
            ]
        if limit is not None:
            rows = rows[:limit]
        if select is not None:
            rows = [{field: row.get(field) for field in select} for row in rows]
        return pa.Table.from_pylist(rows)

    async def insert(
        self,
        *,
        table: str,
        records: list[dict[str, Any]],
        namespace: list[str] | None = None,
    ) -> None:
        key = self._key(table=table, namespace=namespace)
        stored_records = [self._stored_record(record) for record in records]
        self.rows.setdefault(key, []).extend(stored_records)
        inserted_at = time.monotonic()
        for record in stored_records:
            self.inserts.append({"inserted_at": inserted_at, "record": record})

    async def update(
        self,
        *,
        table: str,
        where: str,
        values: dict[str, Any],
        namespace: list[str] | None = None,
    ) -> None:
        prefix = "sequence = "
        assert where.startswith(prefix)
        sequence = int(where[len(prefix) :])
        key = self._key(table=table, namespace=namespace)
        stored_values = self._stored_record(values)
        for row in self.rows.setdefault(key, []):
            if row.get("sequence") == sequence:
                row.update(stored_values)
                return
        raise AssertionError(f"row not found for {where}")

    async def optimize(
        self,
        *,
        table: str,
        namespace: list[str] | None = None,
        config: Any = None,
    ) -> None:
        del table
        del namespace
        del config

    @staticmethod
    def _stored_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to_json() if isinstance(value, DatasetJson) else value
            for key, value in record.items()
        }


class _FakeRoom:
    def __init__(self) -> None:
        self.datasets = _FakeDatasets()
        self.local_participant = Participant(
            id="local",
            attributes={"name": "agent"},
        )


class _TimedSupervisor(AgentSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple[float, Message]] = []

    def send(self, message: Message) -> None:
        self.sent.append((time.monotonic(), message))

    def messages(self, *, message_type: str) -> list[tuple[float, Message]]:
        return [
            (sent_at, message)
            for sent_at, message in self.sent
            if message.data.type == message_type
        ]


class _RealtimeDatasetSupervisor(_TimedSupervisor):
    def __init__(self, *, room: _FakeRoom) -> None:
        super().__init__()
        self.room = room
        self.created_processes: list[LLMAgentProcess] = []

    def create_thread_process(self, thread_id: str) -> LLMAgentProcess:
        storage = DatasetThreadStorage(
            room=self.room,
            path=thread_id,
            max_append_message_count=1,
            optimize_after_append_count=1000,
            persist_deltas=True,
        )
        adapter = OpenAIRealtimeAdapter(
            model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime"),
            base_url=os.getenv("OPENAI_REALTIME_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY"),
            session_options={
                "instructions": "You are a test assistant. Keep responses short.",
                "output_modalities": ["text"],
            },
            response_options={
                "output_modalities": ["text"],
            },
        )
        process: LLMAgentProcess

        def publish_thread_status(message: AgentMessage) -> None:
            process.emit(sender=None, payload=message)

        process = LLMAgentProcess(
            thread_id=thread_id,
            participant=self.room.local_participant,
            llm_adapter=adapter,
            thread_storage=storage,
            thread_status_publisher=AgentMessageThreadStatusPublisher(
                thread_id=thread_id,
                publish=publish_thread_status,
            ),
        )
        self.created_processes.append(process)
        return process


async def _wait_until(predicate, *, interval: float = 0.05) -> None:
    while not predicate():
        await asyncio.sleep(interval)


def _stored_message_type(insert: dict[str, Any]) -> str | None:
    record = insert.get("record")
    if not isinstance(record, dict):
        return None
    message_type = record.get("type")
    return message_type if isinstance(message_type, str) else None


def _stored_message_data(insert: dict[str, Any]) -> dict[str, Any] | None:
    record = insert.get("record")
    if not isinstance(record, dict):
        return None
    data = record.get("data")
    return data if isinstance(data, dict) else None


def _turn_started_messages(
    supervisor: _TimedSupervisor,
) -> list[tuple[float, TurnStarted]]:
    messages: list[tuple[float, TurnStarted]] = []
    for sent_at, message in supervisor.messages(message_type=AGENT_EVENT_TURN_STARTED):
        if isinstance(message.data, TurnStarted):
            messages.append((sent_at, message.data))
    return messages


def _turn_start_accepted_messages(
    supervisor: _TimedSupervisor,
) -> list[tuple[float, TurnStartAccepted]]:
    messages: list[tuple[float, TurnStartAccepted]] = []
    for sent_at, message in supervisor.messages(
        message_type=AGENT_EVENT_TURN_START_ACCEPTED
    ):
        if isinstance(message.data, TurnStartAccepted):
            messages.append((sent_at, message.data))
    return messages


def _thread_status_messages(
    supervisor: _TimedSupervisor,
) -> list[tuple[float, AgentThreadStatus]]:
    messages: list[tuple[float, AgentThreadStatus]] = []
    for sent_at, message in supervisor.messages(message_type=AGENT_EVENT_THREAD_STATUS):
        if isinstance(message.data, AgentThreadStatus):
            messages.append((sent_at, message.data))
    return messages


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


@pytest.mark.asyncio
async def test_live_realtime_create_response_streams_audio_deltas() -> None:
    adapter = OpenAIRealtimeAdapter(
        model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        base_url=os.getenv("OPENAI_REALTIME_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        session_options={
            "instructions": "You are a test assistant. Keep responses short.",
            "output_modalities": ["audio"],
        },
        response_options={
            "instructions": "Say the word pong once.",
            "output_modalities": ["audio"],
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
    assert (
        "response.audio.delta" in event_types
        or "response.output_audio.delta" in event_types
    )
    assert not any(
        event_type in {"response.output_text.delta", "response.text.delta"}
        for event_type in event_types
    )


@pytest.mark.asyncio
async def test_live_realtime_audio_deltas_publish_agent_audio_events() -> None:
    adapter = OpenAIRealtimeAdapter(
        model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        base_url=os.getenv("OPENAI_REALTIME_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        session_options={
            "instructions": "You are a test assistant. Keep responses short.",
            "output_modalities": ["audio"],
        },
        response_options={
            "instructions": "Say the word pong once.",
            "output_modalities": ["audio"],
        },
    )
    context = adapter.create_session()
    messages: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        thread_id="thread-1",
        turn_id="turn-1",
        callback=messages.append,
    )

    await adapter.connect(context=context, event_handler=publisher)
    try:
        terminal_event = await asyncio.wait_for(
            adapter.create_response(
                context=context,
                caller=_FakeParticipant(),
                toolkits=[],
                event_handler=publisher,
            ),
            timeout=60,
        )
    finally:
        await adapter.disconnect(context=context)

    terminal_type = terminal_event.get("type")
    assert terminal_type in {"response.done", "response.completed"}
    assert any(isinstance(message, AgentAudioGenerationDelta) for message in messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attempt",
    range(int(os.getenv("OPENAI_REALTIME_SECOND_TURN_ATTEMPTS", "3"))),
)
async def test_live_realtime_process_dataset_second_turn_surfaces_thinking_promptly(
    attempt: int,
):
    room = _FakeRoom()
    supervisor = _TimedSupervisor()
    storage = DatasetThreadStorage(
        room=room,
        path=(f"dataset://live/realtime-second-turn-{attempt}-{uuid.uuid4().hex}"),
        max_append_message_count=1,
        optimize_after_append_count=1000,
        persist_deltas=True,
    )
    adapter = OpenAIRealtimeAdapter(
        model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        base_url=os.getenv("OPENAI_REALTIME_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        session_options={
            "instructions": "You are a test assistant. Keep responses short.",
            "output_modalities": ["text"],
        },
        response_options={
            "output_modalities": ["text"],
        },
    )

    process: LLMAgentProcess

    def publish_thread_status(message: AgentMessage) -> None:
        process.emit(sender=None, payload=message)

    process = LLMAgentProcess(
        thread_id="thread-1",
        participant=room.local_participant,
        llm_adapter=adapter,
        thread_storage=storage,
        thread_status_publisher=AgentMessageThreadStatusPublisher(
            thread_id="thread-1",
            publish=publish_thread_status,
        ),
    )

    async def wait_for_turn_end_count(count: int) -> None:
        await _wait_until(
            lambda: (
                len(supervisor.messages(message_type=AGENT_EVENT_TURN_ENDED)) >= count
            )
        )

    async def send_turn_and_wait_for_end(*, text: str, expected_end_count: int) -> None:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": text}],
                )
            )
        )
        await asyncio.wait_for(wait_for_turn_end_count(expected_end_count), timeout=90)

    await process.start(supervisor)
    try:
        await send_turn_and_wait_for_end(
            text="Reply with exactly: first",
            expected_end_count=1,
        )

        first_status_count = len(
            supervisor.messages(message_type=AGENT_EVENT_THREAD_STATUS)
        )
        first_dataset_insert_count = len(room.datasets.inserts)
        second_send_at = time.monotonic()
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "Reply with exactly: second"}],
                )
            )
        )

        await asyncio.wait_for(
            _wait_until(lambda: len(_turn_started_messages(supervisor)) >= 2),
            timeout=10,
        )
        second_started_at, second_started = _turn_started_messages(supervisor)[1]

        await asyncio.wait_for(
            _wait_until(
                lambda: any(
                    status.turn_id == second_started.turn_id
                    and status.status == "Thinking"
                    for _, status in _thread_status_messages(supervisor)[
                        first_status_count:
                    ]
                )
            ),
            timeout=5,
        )
        second_thinking_at = next(
            sent_at
            for sent_at, status in _thread_status_messages(supervisor)[
                first_status_count:
            ]
            if status.turn_id == second_started.turn_id and status.status == "Thinking"
        )

        await asyncio.wait_for(
            _wait_until(
                lambda: any(
                    (data := _stored_message_data(insert)) is not None
                    and data.get("turn_id") == second_started.turn_id
                    and data.get("status") == "Thinking"
                    for insert in room.datasets.inserts[first_dataset_insert_count:]
                    if _stored_message_type(insert) == AGENT_EVENT_THREAD_STATUS
                )
            ),
            timeout=5,
        )
        second_dataset_thinking_at = next(
            insert["inserted_at"]
            for insert in room.datasets.inserts[first_dataset_insert_count:]
            if _stored_message_type(insert) == AGENT_EVENT_THREAD_STATUS
            and (data := _stored_message_data(insert)) is not None
            and data.get("turn_id") == second_started.turn_id
            and data.get("status") == "Thinking"
        )

        await asyncio.wait_for(
            _wait_until(
                lambda: any(
                    isinstance(message.data, AgentTextContentDelta)
                    and message.data.turn_id == second_started.turn_id
                    for _, message in supervisor.sent
                )
            ),
            timeout=30,
        )
        second_delta_at = next(
            sent_at
            for sent_at, message in supervisor.sent
            if isinstance(message.data, AgentTextContentDelta)
            and message.data.turn_id == second_started.turn_id
        )

        await asyncio.wait_for(wait_for_turn_end_count(2), timeout=90)
        second_end_at = supervisor.messages(message_type=AGENT_EVENT_TURN_ENDED)[1][0]

        accepted = _turn_start_accepted_messages(supervisor)
        assert len(accepted) >= 2
        assert accepted[1][1].turn_id == second_started.turn_id
        assert second_started_at - second_send_at < 2.0
        assert second_thinking_at - second_send_at < 2.0
        assert second_dataset_thinking_at - second_send_at < 2.0
        assert second_delta_at - second_send_at < 15.0
        assert second_end_at - second_send_at < 30.0
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attempt",
    range(int(os.getenv("OPENAI_REALTIME_ROUTED_SECOND_TURN_ATTEMPTS", "3"))),
)
async def test_live_realtime_routed_dataset_second_turn_surfaces_thinking_promptly(
    attempt: int,
):
    room = _FakeRoom()
    supervisor = _RealtimeDatasetSupervisor(room=room)
    thread_id = (
        f"dataset://live/realtime-routed-second-turn-{attempt}-{uuid.uuid4().hex}"
    )

    async def wait_for_turn_end_count(count: int) -> None:
        await _wait_until(
            lambda: (
                len(supervisor.messages(message_type=AGENT_EVENT_TURN_ENDED)) >= count
            )
        )

    async def route_turn(*, text: str) -> None:
        await supervisor.route(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id=thread_id,
                    content=[{"type": "text", "text": text}],
                )
            )
        )

    await supervisor.start()
    try:
        await route_turn(text="Reply with exactly: first")
        await asyncio.wait_for(wait_for_turn_end_count(1), timeout=90)

        first_status_count = len(
            supervisor.messages(message_type=AGENT_EVENT_THREAD_STATUS)
        )
        first_dataset_insert_count = len(room.datasets.inserts)
        second_send_at = time.monotonic()
        await route_turn(text="Reply with exactly: second")

        await asyncio.wait_for(
            _wait_until(lambda: len(_turn_started_messages(supervisor)) >= 2),
            timeout=10,
        )
        second_started_at, second_started = _turn_started_messages(supervisor)[1]

        await asyncio.wait_for(
            _wait_until(
                lambda: any(
                    status.turn_id == second_started.turn_id
                    and status.status == "Thinking"
                    for _, status in _thread_status_messages(supervisor)[
                        first_status_count:
                    ]
                )
            ),
            timeout=5,
        )
        second_thinking_at = next(
            sent_at
            for sent_at, status in _thread_status_messages(supervisor)[
                first_status_count:
            ]
            if status.turn_id == second_started.turn_id and status.status == "Thinking"
        )

        await asyncio.wait_for(
            _wait_until(
                lambda: any(
                    (data := _stored_message_data(insert)) is not None
                    and data.get("turn_id") == second_started.turn_id
                    and data.get("status") == "Thinking"
                    for insert in room.datasets.inserts[first_dataset_insert_count:]
                    if _stored_message_type(insert) == AGENT_EVENT_THREAD_STATUS
                )
            ),
            timeout=5,
        )
        second_dataset_thinking_at = next(
            insert["inserted_at"]
            for insert in room.datasets.inserts[first_dataset_insert_count:]
            if _stored_message_type(insert) == AGENT_EVENT_THREAD_STATUS
            and (data := _stored_message_data(insert)) is not None
            and data.get("turn_id") == second_started.turn_id
            and data.get("status") == "Thinking"
        )

        await asyncio.wait_for(
            _wait_until(
                lambda: any(
                    isinstance(message.data, AgentTextContentDelta)
                    and message.data.turn_id == second_started.turn_id
                    for _, message in supervisor.sent
                )
            ),
            timeout=30,
        )
        second_delta_at = next(
            sent_at
            for sent_at, message in supervisor.sent
            if isinstance(message.data, AgentTextContentDelta)
            and message.data.turn_id == second_started.turn_id
        )

        await asyncio.wait_for(wait_for_turn_end_count(2), timeout=90)
        second_end_at = supervisor.messages(message_type=AGENT_EVENT_TURN_ENDED)[1][0]

        accepted = _turn_start_accepted_messages(supervisor)
        assert len(accepted) >= 2
        assert accepted[1][1].turn_id == second_started.turn_id
        assert second_started_at - second_send_at < 2.0
        assert second_thinking_at - second_send_at < 2.0
        assert second_dataset_thinking_at - second_send_at < 2.0
        assert second_delta_at - second_send_at < 15.0
        assert second_end_at - second_send_at < 30.0
    finally:
        await supervisor.stop()
