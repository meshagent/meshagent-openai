from __future__ import annotations

from collections.abc import Callable
from typing import Any

from meshagent.agents.event_publisher import (
    AgentEventCallback,
    FunctionToolNameResolver,
    _OpenAIAgentEventPublisher,
    make_openai_agent_event_publisher,
)

__all__ = [
    "AgentEventCallback",
    "FunctionToolNameResolver",
    "_OpenAIAgentEventPublisher",
    "make_openai_agent_event_publisher",
]


def make_realtime_agent_event_publisher(
    *,
    turn_id: str,
    thread_id: str,
    callback: AgentEventCallback,
    function_tool_name_resolver: FunctionToolNameResolver | None = None,
    custom_event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[dict[str, Any]], None]:
    return make_openai_agent_event_publisher(
        turn_id=turn_id,
        thread_id=thread_id,
        callback=callback,
        function_tool_name_resolver=function_tool_name_resolver,
        custom_event_callback=custom_event_callback,
        provider_tool_namespace="openai.realtime",
    )
