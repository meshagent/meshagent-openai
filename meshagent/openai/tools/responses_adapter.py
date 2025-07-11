from meshagent.agents.agent import AgentChatContext
from meshagent.api import RoomClient, RoomException
from meshagent.tools.blob import Blob, BlobStorage
from meshagent.tools import Toolkit, ToolContext, Tool, BaseTool
from meshagent.api.messaging import (
    Response,
    LinkResponse,
    FileResponse,
    JsonResponse,
    TextResponse,
    EmptyResponse,
    RawOutputs,
    ensure_response,
)
from meshagent.agents.adapter import ToolResponseAdapter, LLMAdapter
import json
from typing import List, Literal
from meshagent.openai.proxy import get_client
from openai import AsyncOpenAI, NOT_GIVEN, APIStatusError
from openai.types.responses import ResponseFunctionToolCall, ResponseStreamEvent
import os
from typing import Optional, Callable
import base64

import logging
import re
import asyncio

from pydantic import BaseModel
import copy
from opentelemetry import trace

logger = logging.getLogger("openai_agent")
tracer = trace.get_tracer("openai.llm.responses")


def safe_json_dump(data: dict):
    return json.dumps(copy.deepcopy(data))


def safe_model_dump(model: BaseModel):
    try:
        return safe_json_dump(model.model_dump(mode="json"))
    except Exception:
        return {"error": "unable to dump json for model"}


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
class ResponsesToolBundle:
    def __init__(self, toolkits: List[Toolkit]):
        self._toolkits = toolkits
        self._executors = dict[str, Toolkit]()
        self._safe_names = {}
        self._tools_by_name = {}

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
                self._tools_by_name[name] = v

                if isinstance(v, OpenAIResponsesTool):
                    fns = v.get_open_ai_tool_definitions()
                    for fn in fns:
                        open_ai_tools.append(fn)

                elif isinstance(v, Tool):
                    strict = True
                    if hasattr(v, "strict"):
                        strict = getattr(v, "strict")

                    fn = {
                        "type": "function",
                        "name": name,
                        "description": v.description,
                        "parameters": {
                            **v.input_schema,
                        },
                        "strict": strict,
                    }

                    if v.defs is not None:
                        fn["parameters"]["$defs"] = v.defs

                    open_ai_tools.append(fn)

                else:
                    raise RoomException(f"unsupported tool type {type(v)}")

        if len(open_ai_tools) == 0:
            open_ai_tools = None

        self._open_ai_tools = open_ai_tools

    async def execute(
        self, *, context: ToolContext, tool_call: ResponseFunctionToolCall
    ) -> Response:
        name = tool_call.name
        arguments = json.loads(tool_call.arguments)

        if name not in self._safe_names:
            raise RoomException(f"Invalid tool name {name}, check the name of the tool")

        name = self._safe_names[name]

        if name not in self._executors:
            raise Exception(f"Unregistered tool name {name}")

        proxy = self._executors[name]
        result = await proxy.execute(context=context, name=name, arguments=arguments)
        return ensure_response(result)

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools_by_name.get(name, None)

    def contains(self, name: str) -> bool:
        return name in self._open_ai_tools

    def to_json(self) -> List[dict] | None:
        if self._open_ai_tools is None:
            return None
        return self._open_ai_tools.copy()


# Converts a tool response into a series of messages that can be inserted into the openai context
class OpenAIResponsesToolResponseAdapter(ToolResponseAdapter):
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
        #         "role" : "assistant",
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
        tool_call: ResponseFunctionToolCall,
        room: RoomClient,
        response: Response,
    ) -> list:
        with tracer.start_as_current_span("llm.tool_adapter.create_messages") as span:
            if isinstance(response, RawOutputs):
                span.set_attribute("kind", "raw")
                for output in response.outputs:
                    room.developer.log_nowait(
                        type="llm.message",
                        data={
                            "context": context.id,
                            "participant_id": room.local_participant.id,
                            "participant_name": room.local_participant.get_attribute(
                                "name"
                            ),
                            "message": output,
                        },
                    )

                return response.outputs
            else:
                span.set_attribute("kind", "text")
                output = await self.to_plain_text(room=room, response=response)
                span.set_attribute("output", output)

                message = {
                    "output": output,
                    "call_id": tool_call.call_id,
                    "type": "function_call_output",
                }

                room.developer.log_nowait(
                    type="llm.message",
                    data={
                        "context": context.id,
                        "participant_id": room.local_participant.id,
                        "participant_name": room.local_participant.get_attribute(
                            "name"
                        ),
                        "message": message,
                    },
                )

                return [message]


class OpenAIResponsesAdapter(LLMAdapter[ResponsesToolBundle]):
    def __init__(
        self,
        model: str = os.getenv("OPENAI_MODEL", "gpt-4.1"),
        parallel_tool_calls: Optional[bool] = None,
        client: Optional[AsyncOpenAI] = None,
        response_options: Optional[dict] = None,
        provider: str = "openai",
    ):
        self._model = model
        self._parallel_tool_calls = parallel_tool_calls
        self._client = client
        self._response_options = response_options
        self._provider = provider

    def create_chat_context(self):
        system_role = "system"
        if self._model.startswith("o1"):
            system_role = "developer"
        elif self._model.startswith("o3"):
            system_role = "developer"
        elif self._model.startswith("o4"):
            system_role = "developer"
        elif self._model.startswith("computer-use"):
            system_role = "developer"

        context = AgentChatContext(system_role=system_role)

        return context

    async def check_for_termination(
        self, *, context: AgentChatContext, room: RoomClient
    ) -> bool:
        for message in context.messages:
            if message.get("type", "message") != "message":
                return False

        return True

    # Takes the current chat context, executes a completion request and processes the response.
    # If a tool calls are requested, invokes the tools, processes the tool calls results, and appends the tool call results to the context
    async def next(
        self,
        *,
        context: AgentChatContext,
        room: RoomClient,
        toolkits: list[Toolkit],
        tool_adapter: Optional[ToolResponseAdapter] = None,
        output_schema: Optional[dict] = None,
        event_handler: Optional[Callable[[ResponseStreamEvent], None]] = None,
    ):
        with tracer.start_as_current_span("llm.turn") as span:
            span.set_attributes({"chat_context": context.id, "api": "responses"})

            if tool_adapter is None:
                tool_adapter = OpenAIResponsesToolResponseAdapter()

            try:
                while True:
                    with tracer.start_as_current_span("llm.turn.iteration") as span:
                        span.set_attributes(
                            {"model": self._model, "provider": self._provider}
                        )

                        openai = get_client(room=room)

                        response_schema = output_schema
                        response_name = "response"

                        # We need to do this inside the loop because tools can change mid loop
                        # for example computer use adds goto tools after the first interaction
                        tool_bundle = ResponsesToolBundle(
                            toolkits=[
                                *toolkits,
                            ]
                        )
                        open_ai_tools = tool_bundle.to_json()

                        if open_ai_tools is None:
                            open_ai_tools = NOT_GIVEN

                        ptc = self._parallel_tool_calls
                        extra = {}
                        if ptc is not None and not self._model.startswith("o"):
                            extra["parallel_tool_calls"] = ptc
                            span.set_attribute("parallel_tool_calls", ptc)
                        else:
                            span.set_attribute("parallel_tool_calls", False)

                        text = NOT_GIVEN
                        if output_schema is not None:
                            span.set_attribute("response_format", "json_schema")
                            text = {
                                "format": {
                                    "type": "json_schema",
                                    "name": response_name,
                                    "schema": response_schema,
                                    "strict": True,
                                }
                            }
                        else:
                            span.set_attribute("response_format", "text")

                        previous_response_id = NOT_GIVEN
                        if context.previous_response_id is not None:
                            previous_response_id = context.previous_response_id

                        stream = event_handler is not None

                        with tracer.start_as_current_span("llm.invoke") as span:
                            response_options = self._response_options
                            if response_options is None:
                                response_options = {}
                            response: Response = await openai.responses.create(
                                stream=stream,
                                model=self._model,
                                input=context.messages,
                                tools=open_ai_tools,
                                text=text,
                                previous_response_id=previous_response_id,
                                **response_options,
                            )

                            async def handle_message(message: BaseModel):
                                with tracer.start_as_current_span(
                                    "llm.handle_response"
                                ) as span:
                                    span.set_attributes(
                                        {
                                            "type": message.type,
                                            "message": safe_model_dump(message),
                                        }
                                    )

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

                                    if message.type == "function_call":
                                        tasks = []

                                        async def do_tool_call(
                                            tool_call: ResponseFunctionToolCall,
                                        ):
                                            try:
                                                with tracer.start_as_current_span(
                                                    "llm.handle_tool_call"
                                                ) as span:
                                                    span.set_attributes(
                                                        {
                                                            "id": tool_call.id,
                                                            "name": tool_call.name,
                                                            "call_id": tool_call.call_id,
                                                            "arguments": json.dumps(
                                                                tool_call.arguments
                                                            ),
                                                        }
                                                    )

                                                    tool_context = ToolContext(
                                                        room=room,
                                                        caller=room.local_participant,
                                                        caller_context={
                                                            "chat": context.to_json()
                                                        },
                                                    )
                                                    tool_response = (
                                                        await tool_bundle.execute(
                                                            context=tool_context,
                                                            tool_call=tool_call,
                                                        )
                                                    )
                                                    if (
                                                        tool_response.caller_context
                                                        is not None
                                                    ):
                                                        if (
                                                            tool_response.caller_context.get(
                                                                "chat", None
                                                            )
                                                            is not None
                                                        ):
                                                            tool_chat_context = AgentChatContext.from_json(
                                                                tool_response.caller_context[
                                                                    "chat"
                                                                ]
                                                            )
                                                            if (
                                                                tool_chat_context.previous_response_id
                                                                is not None
                                                            ):
                                                                context.track_response(
                                                                    tool_chat_context.previous_response_id
                                                                )

                                                    logger.info(
                                                        f"tool response {tool_response}"
                                                    )
                                                    return await tool_adapter.create_messages(
                                                        context=context,
                                                        tool_call=tool_call,
                                                        room=room,
                                                        response=tool_response,
                                                    )

                                            except Exception as e:
                                                logger.error(
                                                    f"unable to complete tool call {tool_call}",
                                                    exc_info=e,
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
                                                        "output": json.dumps(
                                                            {
                                                                "error": f"unable to complete tool call: {e}"
                                                            }
                                                        ),
                                                        "call_id": tool_call.call_id,
                                                        "type": "function_call_output",
                                                    }
                                                ]

                                        tasks.append(
                                            asyncio.create_task(do_tool_call(message))
                                        )

                                        results = await asyncio.gather(*tasks)

                                        all_results = []
                                        for result in results:
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
                                            all_results.extend(result)

                                        return all_results, False

                                    elif message.type == "message":
                                        contents = message.content
                                        if response_schema is None:
                                            return [], False
                                        else:
                                            for content in contents:
                                                # First try to parse the result
                                                try:
                                                    full_response = json.loads(
                                                        content.text
                                                    )

                                                # sometimes open ai packs two JSON chunks seperated by newline, check if that's why we couldn't parse
                                                except json.decoder.JSONDecodeError:
                                                    for (
                                                        part
                                                    ) in content.text.splitlines():
                                                        if len(part.strip()) > 0:
                                                            full_response = json.loads(
                                                                part
                                                            )

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
                                                                        "context": message.id,
                                                                        "participant_id": room.local_participant.id,
                                                                        "participant_name": room.local_participant.get_attribute(
                                                                            "name"
                                                                        ),
                                                                        "message": error,
                                                                    },
                                                                )
                                                                context.messages.append(
                                                                    error
                                                                )
                                                                continue

                                                return [full_response], True
                                    # elif message.type == "computer_call" and tool_bundle.get_tool("computer_call"):
                                    #    with tracer.start_as_current_span("llm.handle_computer_call") as span:
                                    #
                                    #        computer_call :ResponseComputerToolCall = message
                                    #        span.set_attributes({
                                    #            "id": computer_call.id,
                                    #            "action": computer_call.action,
                                    #            "call_id": computer_call.call_id,
                                    #            "type": json.dumps(computer_call.type)
                                    #        })

                                    #        tool_context = ToolContext(
                                    #            room=room,
                                    #            caller=room.local_participant,
                                    #            caller_context={ "chat" : context.to_json }
                                    #        )
                                    #        outputs = (await tool_bundle.get_tool("computer_call").execute(context=tool_context, arguments=message.model_dump(mode="json"))).outputs

                                    #    return outputs, False

                                    else:
                                        for toolkit in toolkits:
                                            for tool in toolkit.tools:
                                                if isinstance(
                                                    tool, OpenAIResponsesTool
                                                ):
                                                    with tracer.start_as_current_span(
                                                        "llm.handle_tool_call"
                                                    ) as span:
                                                        arguments = message.model_dump(
                                                            mode="json"
                                                        )
                                                        span.set_attributes(
                                                            {
                                                                "type": message.type,
                                                                "arguments": safe_json_dump(
                                                                    arguments
                                                                ),
                                                            }
                                                        )

                                                        handlers = tool.get_open_ai_output_handlers()
                                                        if message.type in handlers:
                                                            tool_context = ToolContext(
                                                                room=room,
                                                                caller=room.local_participant,
                                                                caller_context={
                                                                    "chat": context.to_json()
                                                                },
                                                            )
                                                            result = await handlers[
                                                                message.type
                                                            ](tool_context, **arguments)

                                                            if result is not None:
                                                                span.set_attribute(
                                                                    "result",
                                                                    safe_json_dump(
                                                                        result
                                                                    ),
                                                                )
                                                                return [result], False
                                                        else:
                                                            logger.warning(
                                                                f"OpenAI response handler was not registered for {message.type}"
                                                            )

                                    return [], False

                            if not stream:
                                room.developer.log_nowait(
                                    type="llm.message",
                                    data={
                                        "context": context.id,
                                        "participant_id": room.local_participant.id,
                                        "participant_name": room.local_participant.get_attribute(
                                            "name"
                                        ),
                                        "response": response.to_dict(),
                                    },
                                )

                                context.track_response(response.id)

                                final_outputs = []

                                for message in response.output:
                                    context.previous_messages.append(message.to_dict())
                                    outputs, done = await handle_message(
                                        message=message
                                    )
                                    if done:
                                        final_outputs.extend(outputs)
                                    else:
                                        for output in outputs:
                                            context.messages.append(output)

                                if len(final_outputs) > 0:
                                    return final_outputs[0]

                                with tracer.start_as_current_span(
                                    "llm.turn.check_for_termination"
                                ) as span:
                                    term = await self.check_for_termination(
                                        context=context, room=room
                                    )
                                    if term:
                                        span.set_attribute("terminate", True)
                                        text = ""
                                        for output in response.output:
                                            if output.type == "message":
                                                for content in output.content:
                                                    text += content.text

                                        return text
                                    else:
                                        span.set_attribute("terminate", False)

                            else:
                                final_outputs = []
                                all_outputs = []
                                async for e in response:
                                    with tracer.start_as_current_span(
                                        "llm.stream.event"
                                    ) as span:
                                        event: ResponseStreamEvent = e
                                        span.set_attributes(
                                            {
                                                "type": event.type,
                                                "event": safe_model_dump(event),
                                            }
                                        )

                                        event_handler(event)

                                        if event.type == "response.completed":
                                            context.track_response(event.response.id)

                                            context.messages.extend(all_outputs)

                                            with tracer.start_as_current_span(
                                                "llm.turn.check_for_termination"
                                            ) as span:
                                                term = await self.check_for_termination(
                                                    context=context, room=room
                                                )

                                                if term:
                                                    span.set_attribute(
                                                        "terminate", True
                                                    )

                                                    text = ""
                                                    for output in event.response.output:
                                                        if output.type == "message":
                                                            for (
                                                                content
                                                            ) in output.content:
                                                                text += content.text

                                                    return text

                                                span.set_attribute("terminate", False)

                                            all_outputs = []

                                        elif event.type == "response.output_item.done":
                                            context.previous_messages.append(
                                                event.item.to_dict()
                                            )

                                            outputs, done = await handle_message(
                                                message=event.item
                                            )
                                            if done:
                                                final_outputs.extend(outputs)
                                            else:
                                                for output in outputs:
                                                    all_outputs.append(output)

                                        else:
                                            for toolkit in toolkits:
                                                for tool in toolkit.tools:
                                                    if isinstance(
                                                        tool, OpenAIResponsesTool
                                                    ):
                                                        callbacks = tool.get_open_ai_stream_callbacks()

                                                        if event.type in callbacks:
                                                            tool_context = ToolContext(
                                                                room=room,
                                                                caller=room.local_participant,
                                                                caller_context={
                                                                    "chat": context.to_json()
                                                                },
                                                            )

                                                            await callbacks[event.type](
                                                                tool_context,
                                                                **event.to_dict(),
                                                            )

                                        if len(final_outputs) > 0:
                                            return final_outputs[0]

            except APIStatusError as e:
                raise RoomException(f"Error from OpenAI: {e}")


class OpenAIResponsesTool(BaseTool):
    def get_open_ai_tool_definitions(self) -> list[dict]:
        return []

    def get_open_ai_stream_callbacks(self) -> dict[str, Callable]:
        return {}

    def get_open_ai_output_handlers(self) -> dict[str, Callable]:
        return {}


class ImageGenerationTool(OpenAIResponsesTool):
    def __init__(
        self,
        *,
        background: Literal["transparent", "opaque", "auto"] = None,
        input_image_mask_url: Optional[str] = None,
        model: Optional[str] = None,
        moderation: Optional[str] = None,
        output_compression: Optional[int] = None,
        output_format: Optional[Literal["png", "webp", "jpeg"]] = None,
        partial_images: Optional[int] = None,
        quality: Optional[Literal["auto", "low", "medium", "high"]] = None,
        size: Optional[Literal["1024x1024", "1024x1536", "1536x1024", "auto"]] = None,
    ):
        super().__init__(name="image_generation")
        self.background = background
        self.input_image_mask_url = input_image_mask_url
        self.model = model
        self.moderation = moderation
        self.output_compression = output_compression
        self.output_format = output_format
        if partial_images is None:
            partial_images = 1  # streaming wants non zero, and we stream by default
        self.partial_images = partial_images
        self.quality = quality
        self.size = size

    def get_open_ai_tool_definitions(self):
        opts = {"type": "image_generation"}

        if self.background is not None:
            opts["background"] = self.background

        if self.input_image_mask_url is not None:
            opts["input_image_mask"] = {"image_url": self.input_image_mask_url}

        if self.model is not None:
            opts["model"] = self.model

        if self.moderation is not None:
            opts["moderation"] = self.moderation

        if self.output_compression is not None:
            opts["output_compression"] = self.output_compression

        if self.output_format is not None:
            opts["output_format"] = self.output_format

        if self.partial_images is not None:
            opts["partial_images"] = self.partial_images

        if self.quality is not None:
            opts["quality"] = self.quality

        if self.size is not None:
            opts["size"] = self.size

        return [opts]

    def get_open_ai_stream_callbacks(self):
        return {
            "response.image_generation_call.completed": self.on_image_generation_completed,
            "response.image_generation_call.in_progress": self.on_image_generation_in_progress,
            "response.image_generation_call.generating": self.on_image_generation_generating,
            "response.image_generation_call.partial_image": self.on_image_generation_partial,
        }

    def get_open_ai_output_handlers(self):
        return {"image_generation_call": self.handle_image_generated}

    # response.image_generation_call.completed
    async def on_image_generation_completed(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    # response.image_generation_call.in_progress
    async def on_image_generation_in_progress(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    # response.image_generation_call.generating
    async def on_image_generation_generating(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    # response.image_generation_call.partial_image
    async def on_image_generation_partial(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        partial_image_b64: str,
        partial_image_index: int,
        size: str,
        quality: str,
        background: str,
        output_format: str,
        **extra,
    ):
        pass

    async def on_image_generated(
        self,
        context: ToolContext,
        *,
        item_id: str,
        data: bytes,
        status: str,
        size: str,
        quality: str,
        background: str,
        output_format: str,
        **extra,
    ):
        pass

    async def handle_image_generated(
        self,
        context: ToolContext,
        *,
        id: str,
        result: str | None,
        status: str,
        type: str,
        size: str,
        quality: str,
        background: str,
        output_format: str,
        **extra,
    ):
        if result is not None:
            data = base64.b64decode(result)
            await self.on_image_generated(
                context,
                item_id=id,
                data=data,
                status=status,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
            )


class LocalShellTool(OpenAIResponsesTool):
    def __init__(self):
        super().__init__(name="local_shell")

    def get_open_ai_tool_definitions(self):
        return [{"type": "local_shell"}]

    def get_open_ai_output_handlers(self):
        return {"local_shell_call": self.handle_local_shell_call}

    async def execute_shell_command(
        self,
        context: ToolContext,
        *,
        command: list[str],
        env: dict,
        type: str,
        timeout_ms: int | None = None,
        user: str | None = None,
        working_directory: str | None = None,
    ):
        merged_env = {**os.environ, **(env or {})}

        # Spawn the process
        proc = await asyncio.create_subprocess_exec(
            *(command if isinstance(command, (list, tuple)) else [command]),
            cwd=working_directory or os.getcwd(),
            env=merged_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_ms / 1000 if timeout_ms else None,
            )
        except asyncio.TimeoutError:
            proc.kill()  # send SIGKILL / TerminateProcess
            stdout, stderr = await proc.communicate()
            raise  # re-raise so caller sees the timeout

        encoding = os.device_encoding(1) or "utf-8"
        stdout = stdout.decode(encoding, errors="replace")
        stderr = stderr.decode(encoding, errors="replace")

        return stdout + stderr

    async def handle_local_shell_call(
        self,
        context,
        *,
        id: str,
        action: dict,
        call_id: str,
        status: str,
        type: str,
        **extra,
    ):
        result = await self.execute_shell_command(context, **action)

        output_item = {
            "type": "local_shell_call_output",
            "call_id": call_id,
            "output": result,
        }

        return output_item


class ContainerFile:
    def __init__(self, *, file_id: str, mime_type: str, container_id: str):
        self.file_id = file_id
        self.mime_type = mime_type
        self.container_id = container_id


class CodeInterpreterTool(OpenAIResponsesTool):
    def __init__(
        self,
        *,
        container_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
    ):
        super().__init__(name="code_interpreter_call")
        self.container_id = container_id
        self.file_ids = file_ids

    def get_open_ai_tool_definitions(self):
        opts = {"type": "code_interpreter"}

        if self.container_id is not None:
            opts["container_id"] = self.container_id

        if self.file_ids is not None:
            if self.container_id is not None:
                raise Exception(
                    "Cannot specify both an existing container and files to upload in a code interpreter tool"
                )

            opts["container"] = {"type": "auto", "file_ids": self.file_ids}

        return [opts]

    def get_open_ai_output_handlers(self):
        return {"code_interpreter_call": self.handle_code_interpreter_call}

    async def on_code_interpreter_result(
        self,
        context: ToolContext,
        *,
        code: str,
        logs: list[str],
        files: list[ContainerFile],
    ):
        pass

    async def handle_code_interpreter_call(
        self,
        context,
        *,
        code: str,
        id: str,
        results: list[dict],
        call_id: str,
        status: str,
        type: str,
        container_id: str,
        **extra,
    ):
        logs = []
        files = []

        for result in results:
            if result.type == "logs":
                logs.append(results["logs"])

            elif result.type == "files":
                files.append(
                    ContainerFile(
                        container_id=container_id,
                        file_id=result["file_id"],
                        mime_type=result["mime_type"],
                    )
                )

        await self.on_code_interpreter_result(
            context, code=code, logs=logs, files=files
        )


class MCPToolDefinition:
    def __init__(
        self,
        *,
        input_schema: dict,
        name: str,
        annotations: dict | None,
        description: str | None,
    ):
        self.input_schema = input_schema
        self.name = name
        self.annotations = annotations
        self.description = description


class MCPServer:
    def __init__(
        self,
        *,
        server_label: str,
        server_url: str,
        allowed_tools: Optional[list[str]] = None,
        headers: Optional[dict] = None,
        # require approval for all tools
        require_approval: Optional[Literal["always", "never"]] = None,
        # list of tools that always require approval
        always_require_approval: Optional[list[str]] = None,
        # list of tools that never require approval
        never_require_approval: Optional[list[str]] = None,
    ):
        self.server_label = server_label
        self.server_url = server_url
        self.allowed_tools = allowed_tools
        self.headers = headers
        self.require_approval = require_approval
        self.always_require_approval = always_require_approval
        self.never_require_approval = never_require_approval


class MCPTool(OpenAIResponsesTool):
    def __init__(self, *, servers: list[MCPServer]):
        super().__init__(name="mcp")
        self.servers = servers

    def get_open_ai_tool_definitions(self):
        defs = []
        for server in self.servers:
            opts = {
                "type": "mcp",
                "server_label": server.server_label,
                "server_url": server.server_url,
            }

            if server.allowed_tools is not None:
                opts["allowed_tools"] = server.allowed_tools

            if server.headers is not None:
                opts["headers"] = server.headers

            if (
                server.always_require_approval is not None
                or server.never_require_approval is not None
            ):
                opts["require_approval"] = {}

                if server.always_require_approval is not None:
                    opts["require_approval"]["always"] = {
                        "tool_names": server.always_require_approval
                    }

                if server.never_require_approval is not None:
                    opts["require_approval"]["never"] = {
                        "tool_names": server.never_require_approval
                    }

            if server.require_approval:
                opts["require_approval"] = server.require_approval

            defs.append(opts)

        return defs

    def get_open_ai_stream_callbacks(self):
        return {
            "response.mcp_list_tools.in_progress": self.on_mcp_list_tools_in_progress,
            "response.mcp_list_tools.failed": self.on_mcp_list_tools_failed,
            "response.mcp_list_tools.completed": self.on_mcp_list_tools_completed,
            "response.mcp_call.in_progress": self.on_mcp_call_in_progress,
            "response.mcp_call.failed": self.on_mcp_call_failed,
            "response.mcp_call.completed": self.on_mcp_call_completed,
            "response.mcp_call.arguments.done": self.on_mcp_call_arguments_done,
            "response.mcp_call.arguments.delta": self.on_mcp_call_arguments_delta,
        }

    async def on_mcp_list_tools_in_progress(
        self, context: ToolContext, *, sequence_number: int, type: str, **extra
    ):
        pass

    async def on_mcp_list_tools_failed(
        self, context: ToolContext, *, sequence_number: int, type: str, **extra
    ):
        pass

    async def on_mcp_list_tools_completed(
        self, context: ToolContext, *, sequence_number: int, type: str, **extra
    ):
        pass

    async def on_mcp_call_in_progress(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_mcp_call_failed(
        self, context: ToolContext, *, sequence_number: int, type: str, **extra
    ):
        pass

    async def on_mcp_call_completed(
        self, context: ToolContext, *, sequence_number: int, type: str, **extra
    ):
        pass

    async def on_mcp_call_arguments_done(
        self,
        context: ToolContext,
        *,
        arguments: dict,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_mcp_call_arguments_delta(
        self,
        context: ToolContext,
        *,
        delta: dict,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    def get_open_ai_output_handlers(self):
        return {
            "mcp_call": self.handle_mcp_call,
            "mcp_list_tools": self.handle_mcp_list_tools,
            "mcp_approval_request": self.handle_mcp_approval_request,
        }

    async def on_mcp_list_tools(
        self,
        context: ToolContext,
        *,
        server_label: str,
        tools: list[MCPToolDefinition],
        error: str | None,
        **extra,
    ):
        pass

    async def handle_mcp_list_tools(
        self,
        context,
        *,
        id: str,
        server_label: str,
        tools: list,
        type: str,
        error: str | None = None,
        **extra,
    ):
        mcp_tools = []
        for tool in tools:
            mcp_tools.append(
                MCPToolDefinition(
                    input_schema=tool["input_schema"],
                    name=tool["name"],
                    annotations=tool["annotations"],
                    description=tool["description"],
                )
            )

        await self.on_mcp_list_tools(
            context, server_label=server_label, tools=mcp_tools, error=error
        )

    async def on_mcp_call(
        self,
        context: ToolContext,
        *,
        name: str,
        arguments: str,
        server_label: str,
        error: str | None,
        output: str | None,
        **extra,
    ):
        pass

    async def handle_mcp_call(
        self,
        context,
        *,
        arguments: str,
        id: str,
        name: str,
        server_label: str,
        type: str,
        error: str | None,
        output: str | None,
        **extra,
    ):
        await self.on_mcp_call(
            context,
            name=name,
            arguments=arguments,
            server_label=server_label,
            error=error,
            output=output,
        )

    async def on_mcp_approval_request(
        self,
        context: ToolContext,
        *,
        name: str,
        arguments: str,
        server_label: str,
        **extra,
    ) -> bool:
        return True

    async def handle_mcp_approval_request(
        self,
        context: ToolContext,
        *,
        arguments: str,
        id: str,
        name: str,
        server_label: str,
        type: str,
        **extra,
    ):
        logger.info("approval requested for MCP tool {server_label}.{name}")
        should_approve = await self.on_mcp_approval_request(
            context, arguments=arguments, name=name, server_label=server_label
        )
        if should_approve:
            logger.info("approval granted for MCP tool {server_label}.{name}")
            return {
                "type": "mcp_approval_response",
                "approve": True,
                "approval_request_id": id,
            }
        else:
            logger.info("approval denied for MCP tool {server_label}.{name}")
            return {
                "type": "mcp_approval_response",
                "approve": False,
                "approval_request_id": id,
            }


class ReasoningTool(OpenAIResponsesTool):
    def __init__(self):
        super().__init__(name="reasoning")

    def get_open_ai_output_handlers(self):
        return {
            "reasoning": self.handle_reasoning,
        }

    def get_open_ai_stream_callbacks(self):
        return {
            "response.reasoning_summary_text.done": self.on_reasoning_summary_text_done,
            "response.reasoning_summary_text.delta": self.on_reasoning_summary_text_delta,
            "response.reasoning_summary_part.done": self.on_reasoning_summary_part_done,
            "response.reasoning_summary_part.added": self.on_reasoning_summary_part_added,
        }

    async def on_reasoning_summary_part_added(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        part: dict,
        sequence_number: int,
        summary_index: int,
        text: str,
        type: str,
        **extra,
    ):
        pass

    async def on_reasoning_summary_part_done(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        part: dict,
        sequence_number: int,
        summary_index: int,
        text: str,
        type: str,
        **extra,
    ):
        pass

    async def on_reasoning_summary_text_delta(
        self,
        context: ToolContext,
        *,
        delta: str,
        output_index: int,
        sequence_number: int,
        summary_index: int,
        text: str,
        type: str,
        **extra,
    ):
        pass

    async def on_reasoning_summary_text_done(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        summary_index: int,
        text: str,
        type: str,
        **extra,
    ):
        pass

    async def on_reasoning(
        self,
        context: ToolContext,
        *,
        summary: str,
        encrypted_content: str | None,
        status: Literal["in_progress", "completed", "incomplete"],
    ):
        pass

    async def handle_reasoning(
        self,
        context: ToolContext,
        *,
        id: str,
        summary: str,
        type: str,
        encrypted_content: str | None,
        status: str,
        **extra,
    ):
        await self.on_reasoning(
            context, summary=summary, encrypted_content=encrypted_content, status=status
        )


# TODO: computer tool call


class WebSearchTool(OpenAIResponsesTool):
    def __init__(self):
        super().__init__(name="web_search")

    def get_open_ai_tool_definitions(self) -> list[dict]:
        return [{"type": "web_search_preview"}]

    def get_open_ai_stream_callbacks(self):
        return {
            "response.web_search_call.in_progress": self.on_web_search_call_in_progress,
            "response.web_search_call.searching": self.on_web_search_call_searching,
            "response.web_search_call.completed": self.on_web_search_call_completed,
        }

    def get_open_ai_output_handlers(self):
        return {"web_search_call": self.handle_web_search_call}

    async def on_web_search_call_in_progress(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_web_search_call_searching(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_web_search_call_completed(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_web_search(self, context: ToolContext, *, status: str, **extra):
        pass

    async def handle_web_search_call(
        self, context: ToolContext, *, id: str, status: str, type: str, **extra
    ):
        await self.on_web_search(context, status=status)


class FileSearchResult:
    def __init__(
        self, *, attributes: dict, file_id: str, filename: str, score: float, text: str
    ):
        self.attributes = attributes
        self.file_id = file_id
        self.filename = filename
        self.score = score
        self.text = text


class FileSearchTool(OpenAIResponsesTool):
    def __init__(
        self,
        *,
        vector_store_ids: list[str],
        filters: Optional[dict] = None,
        max_num_results: Optional[int] = None,
        ranking_options: Optional[dict] = None,
    ):
        super().__init__(name="file_search")

        self.vector_store_ids = vector_store_ids
        self.filters = filters
        self.max_num_results = max_num_results
        self.ranking_options = ranking_options

    def get_open_ai_tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "file_search",
                "vector_store_ids": self.vector_store_ids,
                "filters": self.filters,
                "max_num_results": self.max_num_results,
                "ranking_options": self.ranking_options,
            }
        ]

    def get_open_ai_stream_callbacks(self):
        return {
            "response.file_search_call.in_progress": self.on_file_search_call_in_progress,
            "response.file_search_call.searching": self.on_file_search_call_searching,
            "response.file_search_call.completed": self.on_file_search_call_completed,
        }

    def get_open_ai_output_handlers(self):
        return {"handle_file_search_call": self.handle_file_search_call}

    async def on_file_search_call_in_progress(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_file_search_call_searching(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_file_search_call_completed(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        type: str,
        **extra,
    ):
        pass

    async def on_file_search(
        self,
        context: ToolContext,
        *,
        queries: list,
        results: list[FileSearchResult],
        status: Literal["in_progress", "searching", "incomplete", "failed"],
    ):
        pass

    async def handle_file_search_call(
        self,
        context: ToolContext,
        *,
        id: str,
        queries: list,
        status: str,
        results: dict | None,
        type: str,
        **extra,
    ):
        search_results = None
        if results is not None:
            search_results = []
            for result in results:
                search_results.append(FileSearchResult(**result))

        await self.on_file_search(
            context, queries=queries, results=search_results, status=status
        )
