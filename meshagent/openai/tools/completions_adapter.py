from meshagent.agents.agent import AgentChatContext
from meshagent.api import RoomClient, RoomException
from meshagent.tools.blob import Blob, BlobStorage
from meshagent.tools import Toolkit, ToolContext
from meshagent.api.messaging import (
    Response,
    LinkResponse,
    FileResponse,
    JsonResponse,
    TextResponse,
    EmptyResponse,
)
from meshagent.agents.adapter import ToolResponseAdapter, LLMAdapter
import json
from typing import List

from openai import AsyncOpenAI, APIStatusError
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall

import os
from typing import Optional, Any

import logging
import re
import asyncio

from meshagent.openai.proxy import get_client

logger = logging.getLogger("openai_agent")


def _replace_non_matching(text: str, allowed_chars: str, replacement: str) -> str:
    """
    Replaces every character in `text` that does not match the given
    `allowed_chars` regex set with `replacement`.

    Parameters:
    -----------
    text : str
        The input string on which the replacement is to be done.
    allowed_chars : str
        A string defining the set of allowed characters (part of a character set).
        For example, "a-zA-Z0-9" will keep only letters and digits.
    replacement : str
        The string to replace non-matching characters with.

    Returns:
    --------
    str
        A new string where all characters not in `allowed_chars` are replaced.
    """
    # Build a regex that matches any character NOT in allowed_chars
    pattern = rf"[^{allowed_chars}]"
    return re.sub(pattern, replacement, text)


def safe_tool_name(name: str):
    return _replace_non_matching(name, "a-zA-Z0-9_-", "_")


# Collects a group of tool proxies and manages execution of openai tool calls
class CompletionsToolBundle:
    def __init__(self, toolkits: List[Toolkit]):
        self._toolkits = toolkits
        self._executors = dict[str, Toolkit]()
        self._safe_names = {}

        open_ai_tools = []

        for toolkit in toolkits:
            for v in toolkit.tools:
                k = v.name

                name = safe_tool_name(k)

                if k in self._executors:
                    raise Exception(
                        f"duplicate in bundle '{k}', tool names must be unique."
                    )

                self._executors[k] = toolkit

                self._safe_names[name] = k

                fn = {
                    "name": name,
                    "parameters": {
                        **v.input_schema,
                    },
                    "strict": True,
                }

                if v.defs is not None:
                    fn["parameters"]["$defs"] = v.defs

                schema = {
                    "type": "function",
                    "function": fn,
                }

                open_ai_tools.append(schema)

        if len(open_ai_tools) == 0:
            open_ai_tools = None

        self._open_ai_tools = open_ai_tools

    async def execute(
        self, *, context: ToolContext, tool_call: ChatCompletionMessageToolCall
    ) -> Response:
        function = tool_call.function
        name = function.name
        arguments = json.loads(function.arguments)

        if name not in self._safe_names:
            raise RoomException(f"Invalid tool name {name}, check the name of the tool")

        name = self._safe_names[name]

        if name not in self._executors:
            raise Exception(f"Unregistered tool name {name}")

        proxy = self._executors[name]
        result = await proxy.execute(context=context, name=name, arguments=arguments)
        return result

    def contains(self, name: str) -> bool:
        return name in self._open_ai_tools

    def to_json(self) -> List[dict] | None:
        if self._open_ai_tools is None:
            return None
        return self._open_ai_tools.copy()


# Converts a tool response into a series of messages that can be inserted into the openai context
class OpenAICompletionsToolResponseAdapter(ToolResponseAdapter):
    def __init__(self, blob_storage: Optional[BlobStorage] = None):
        self._blob_storage = blob_storage
        pass

    async def to_plain_text(self, *, room: RoomClient, response: Response) -> str:
        if isinstance(response, LinkResponse):
            return json.dumps(
                {
                    "name": response.name,
                    "url": response.url,
                }
            )

        elif isinstance(response, JsonResponse):
            return json.dumps(response.json)

        elif isinstance(response, TextResponse):
            return response.text

        elif isinstance(response, FileResponse):
            blob = Blob(mime_type=response.mime_type, data=response.data)
            uri = self._blob_storage.store(blob=blob)

            return f"The results have been written to a blob with the uri {uri} with the mime type {blob.mime_type}."

        elif isinstance(response, EmptyResponse):
            return "ok"

        # elif isinstance(response, ImageResponse):
        #     context.messages.append({
        #         "role" : "tool",
        #         "content" : "the user will upload the image",
        #         "tool_call_id" : tool_call.id,
        #     })
        #     context.messages.append({
        #         "role" : "user",
        #         "content" : [
        #             { "type" : "text", "text": "this is the image from tool call id {tool_call.id}" },
        #             { "type" : "image_url", "image_url": {"url": response.url, "detail": "auto"} }
        #         ]
        #     })

        elif isinstance(response, dict):
            return json.dumps(response)

        elif isinstance(response, str):
            return response

        elif response is None:
            return "ok"

        else:
            raise Exception(
                "unexpected return type: {type}".format(type=type(response))
            )

    async def create_messages(
        self,
        *,
        context: AgentChatContext,
        tool_call: Any,
        room: RoomClient,
        response: Response,
    ) -> list:
        message = {
            "role": "tool",
            "content": await self.to_plain_text(room=room, response=response),
            "tool_call_id": tool_call.id,
        }

        room.developer.log_nowait(
            type="llm.message",
            data={
                "context": context.id,
                "participant_id": room.local_participant.id,
                "participant_name": room.local_participant.get_attribute("name"),
                "message": message,
            },
        )

        return [message]


class OpenAICompletionsAdapter(LLMAdapter):
    def __init__(
        self,
        model: str = os.getenv("OPENAI_MODEL"),
        parallel_tool_calls: Optional[bool] = None,
        client: Optional[AsyncOpenAI] = None,
    ):
        self._model = model
        self._parallel_tool_calls = parallel_tool_calls
        self._client = client

    def create_chat_context(self):
        system_role = "system"
        if self._model.startswith("o1"):
            system_role = "developer"
        elif self._model.startswith("o3"):
            system_role = "developer"
        elif self._model.startswith("o4"):
            system_role = "developer"

        context = AgentChatContext(system_role=system_role)

        return context

    # Takes the current chat context, executes a completion request and processes the response.
    # If a tool calls are requested, invokes the tools, processes the tool calls results, and appends the tool call results to the context
    async def next(
        self,
        *,
        context: AgentChatContext,
        room: RoomClient,
        toolkits: Toolkit,
        tool_adapter: Optional[ToolResponseAdapter] = None,
        output_schema: Optional[dict] = None,
    ):
        if tool_adapter is None:
            tool_adapter = OpenAICompletionsToolResponseAdapter()

        try:
            openai = get_client(room=room)

            tool_bundle = CompletionsToolBundle(
                toolkits=[
                    *toolkits,
                ]
            )
            open_ai_tools = tool_bundle.to_json()

            if open_ai_tools is not None:
                logger.info("OpenAI Tools: %s", json.dumps(open_ai_tools))
            else:
                logger.info("OpenAI Tools: Empty")

            response_schema = output_schema
            response_name = "response"

            while True:
                logger.info(
                    "model: %s, context: %s, output_schema: %s",
                    self._model,
                    context.messages,
                    output_schema,
                )
                ptc = self._parallel_tool_calls
                extra = {}
                if ptc is not None and not self._model.startswith("o"):
                    extra["parallel_tool_calls"] = ptc

                if output_schema is not None:
                    extra["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_name,
                            "schema": response_schema,
                            "strict": True,
                        },
                    }

                response: ChatCompletion = await openai.chat.completions.create(
                    n=1,
                    model=self._model,
                    messages=context.messages,
                    tools=open_ai_tools,
                    **extra,
                )
                message = response.choices[0].message
                room.developer.log_nowait(
                    type="llm.message",
                    data={
                        "context": context.id,
                        "participant_id": room.local_participant.id,
                        "participant_name": room.local_participant.get_attribute(
                            "name"
                        ),
                        "message": message.to_dict(),
                    },
                )
                context.messages.append(message)

                if message.tool_calls is not None:
                    tasks = []

                    async def do_tool_call(tool_call: ChatCompletionMessageToolCall):
                        try:
                            tool_context = ToolContext(
                                room=room,
                                caller=room.local_participant,
                                caller_context={"chat": context.to_json},
                            )
                            tool_response = await tool_bundle.execute(
                                context=tool_context, tool_call=tool_call
                            )
                            logger.info(f"tool response {tool_response}")
                            return await tool_adapter.create_messages(
                                context=context,
                                tool_call=tool_call,
                                room=room,
                                response=tool_response,
                            )

                        except Exception as e:
                            logger.error(
                                f"unable to complete tool call {tool_call}", exc_info=e
                            )
                            room.developer.log_nowait(
                                type="llm.error",
                                data={
                                    "participant_id": room.local_participant.id,
                                    "participant_name": room.local_participant.get_attribute(
                                        "name"
                                    ),
                                    "error": f"{e}",
                                },
                            )

                            return [
                                {
                                    "role": "tool",
                                    "content": json.dumps(
                                        {"error": f"unable to complete tool call: {e}"}
                                    ),
                                    "tool_call_id": tool_call.id,
                                }
                            ]

                    for tool_call in message.tool_calls:
                        tasks.append(asyncio.create_task(do_tool_call(tool_call)))

                    results = await asyncio.gather(*tasks)

                    for result in results:
                        if result is not None:
                            room.developer.log_nowait(
                                type="llm.message",
                                data={
                                    "context": context.id,
                                    "participant_id": room.local_participant.id,
                                    "participant_name": room.local_participant.get_attribute(
                                        "name"
                                    ),
                                    "message": result,
                                },
                            )
                            context.messages.append(result)

                elif message.content is not None:
                    content = message.content

                    logger.info("RESPONSE FROM OPENAI %s", content)
                    if response_schema is None:
                        return content

                    # First try to parse the result
                    try:
                        full_response = json.loads(content)
                    # sometimes open ai packs two JSON chunks seperated by newline, check if that's why we couldn't parse
                    except json.decoder.JSONDecodeError:
                        for part in content.splitlines():
                            if len(part.strip()) > 0:
                                full_response = json.loads(part)

                                try:
                                    self.validate(
                                        response=full_response,
                                        output_schema=response_schema,
                                    )
                                except Exception as e:
                                    logger.error(
                                        "recieved invalid response, retrying",
                                        exc_info=e,
                                    )
                                    error = {
                                        "role": "user",
                                        "content": "encountered a validation error with the output: {error}".format(
                                            error=e
                                        ),
                                    }
                                    room.developer.log_nowait(
                                        type="llm.message",
                                        data={
                                            "context": context.id,
                                            "participant_id": room.local_participant.id,
                                            "participant_name": room.local_participant.get_attribute(
                                                "name"
                                            ),
                                            "message": error,
                                        },
                                    )
                                    context.messages.append(error)
                                    continue

                    return full_response
                else:
                    raise RoomException(
                        "Unexpected response from OpenAI {response}".format(
                            response=message
                        )
                    )
        except APIStatusError as e:
            raise RoomException(f"Error from OpenAI: {e}")
