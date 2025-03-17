
from meshagent.agents.agent import Agent, AgentChatContext, AgentCallContext
from meshagent.api import WebSocketClientProtocol, RoomClient, RoomException
from meshagent.tools.blob import Blob, BlobStorage
from meshagent.tools import Toolkit, ToolContext, Tool
from meshagent.api.messaging import Response, LinkResponse, FileResponse, JsonResponse, TextResponse, EmptyResponse, RawOutputs
from meshagent.api.schema_util import prompt_schema
from meshagent.agents.adapter import ToolResponseAdapter, LLMAdapter
from uuid import uuid4
import json
from jsonschema import validate
from typing import List, Dict

from openai import AsyncOpenAI, APIStatusError, NOT_GIVEN
from openai.types.responses import ResponseFunctionToolCall, ResponseStreamEvent

from copy import deepcopy
from abc import abstractmethod, ABC
import os
import jsonschema
from typing import Optional, Any, Callable

import logging
import re
import asyncio

logging.basicConfig()
logger = logging.getLogger("openai_agent")
logger.setLevel(logging.INFO)




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
                    raise Exception(f"duplicate in bundle '{k}', tool names must be unique.")

                self._executors[k] = toolkit

                self._safe_names[name] = k
                self._tools_by_name[name] = v

                if v.name != "computer_call":

                    fn = {
                        "type" : "function",
                        "name" : name,
                        "description" : v.description,
                        "parameters" : {
                            **v.input_schema,
                        },
                        "strict": True,
                    }


                    if v.defs != None:
                        fn["parameters"]["$defs"] = v.defs
            
                    open_ai_tools.append(fn)

                else:

                    open_ai_tools.append(v.options)

        if len(open_ai_tools) == 0:
            open_ai_tools = None

        self._open_ai_tools = open_ai_tools

    async def execute(self, *, context: ToolContext, tool_call: ResponseFunctionToolCall) -> Response:
        try:
            
            name = tool_call.name
            arguments = json.loads(tool_call.arguments)

            if name not in self._safe_names:
                raise RoomException(f"Invalid tool name {name}, check the name of the tool")
            
            name = self._safe_names[name]

            if name not in self._executors:
                raise Exception(f"Unregistered tool name {name}")

            logger.info("executing %s %s %s", tool_call.id, name, arguments)

            proxy = self._executors[name]
            result = await proxy.execute(context=context, name=name, arguments=arguments)
            logger.info("success calling %s %s %s", tool_call.id, name, result)        
            return result

        except Exception as e:
            logger.error("failed calling %s %s", tool_call.id, name, exc_info=e)
            raise
    
    def get_tool(self, name: str) -> Tool | None:
        return self._tools_by_name.get(name, None)
    
    def contains(self, name: str) -> bool:
        return name in self._open_ai_tools

    def to_json(self) -> List[dict] | None:
        if self._open_ai_tools == None:
            return None
        return self._open_ai_tools.copy()
    

# Converts a tool response into a series of messages that can be inserted into the openai context
class OpenAIResponsesToolResponseAdapter(ToolResponseAdapter):
    def __init__(self, blob_storage: Optional[BlobStorage] = None):
        self._blob_storage = blob_storage
        pass

    async def to_plain_text(self, *, room: RoomClient, response: Response) -> str:
        if isinstance(response, LinkResponse):                                            
           return json.dumps({
                "name" : response.name,
                "url" : response.url,
            })
           
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
        
       #elif isinstance(response, ImageResponse):
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

        elif response == None:
            return "ok"
        
        else:
            raise Exception("unexpected return type: {type}".format(type=type(response)))

    async def create_messages(self, *, context: AgentChatContext, tool_call: ResponseFunctionToolCall, room: RoomClient, response: Response) -> list:

        if isinstance(response, RawOutputs):
            
            for output in response.outputs:

                room.developer.log_nowait(type="llm.message", data={ "context" : context.id,  "participant_id" : room.local_participant.id, "participant_name" : room.local_participant.get_attribute("name"), "message" : output })
                                    
            return response.outputs
        else:
            output = await self.to_plain_text(room=room, response=response)
                
            message = {
                "output" : output,
                "call_id" : tool_call.call_id,
                "type" : "function_call_output"
            }                                 
        

            room.developer.log_nowait(type="llm.message", data={ "context" : context.id,  "participant_id" : room.local_participant.id, "participant_name" : room.local_participant.get_attribute("name"), "message" : message })
                                
            return [ message ]
            



class OpenAIResponsesAdapter(LLMAdapter[ResponsesToolBundle]):
    def __init__(self,      
        model: str = os.getenv("OPENAI_MODEL"),
        parallel_tool_calls : Optional[bool] = None,
        client: Optional[AsyncOpenAI] = None,
        reasoning_effort: Optional[str] = None,
        retries : int = 10,
    ):
        self._model = model
        self._parallel_tool_calls = parallel_tool_calls
        self._client = client
        self._reasoning_effort = reasoning_effort
        self._retries = retries

    def create_chat_context(self):
        system_role = "system"
        if self._model.startswith("o1"):
            system_role = "developer"
        elif self._model.startswith("o3"):
            system_role = "developer"
        elif self._model.startswith("computer-use"):
            system_role = "developer"

        context = AgentChatContext(
            system_role=system_role
        )

        return context
    
    async def check_for_termination(self, *, context: AgentChatContext, room: RoomClient) -> bool:

        if len(context.previous_messages) > 0:
            last_message = context.previous_messages[-1]
            logger.info(f"last_message {last_message}")

        for message in context.messages:

            if message.get("type", "message") != "message":
                logger.info(f"found {message.get("type", "message")}")

                return False

        return True

    def _get_client(self, *, room: RoomClient) -> AsyncOpenAI:
        if self._client != None:
            openai = self._client
        else:
            token : str = room.protocol.token
            url : str = room.room_url
            
            room_proxy_url = f"{url}/v1"
        
            openai=AsyncOpenAI(
                api_key=token,
                base_url=room_proxy_url,
                default_headers={
                    "Meshagent-Session" : room.session_id
                }
            )
       
        return openai
    
    # Takes the current chat context, executes a completion request and processes the response.
    # If a tool calls are requested, invokes the tools, processes the tool calls results, and appends the tool call results to the context
    async def next(self,
        *,
        context: AgentChatContext,
        room: RoomClient,
        toolkits: Toolkit,
        tool_adapter: Optional[ToolResponseAdapter] = None,
        output_schema: Optional[dict] = None,
        event_handler: Optional[Callable[[ResponseStreamEvent],None]] = None
    ):  
        if tool_adapter == None:
            tool_adapter = OpenAIResponsesToolResponseAdapter()
            
        try:
            
            openai = self._get_client(room=room)

            tool_bundle = ResponsesToolBundle(toolkits=[
                *toolkits,
            ])
            open_ai_tools = tool_bundle.to_json()

            if open_ai_tools != None:
                logger.info("OpenAI Tools: %s", json.dumps(open_ai_tools))
            else:
                logger.info("OpenAI Tools: Empty")
                open_ai_tools = NOT_GIVEN
            
            response_schema = output_schema
            response_name = "response"
            

            while True:
  
                logger.info("model: %s, context: %s, output_schema: %s", self._model, context.messages, output_schema)
                ptc = self._parallel_tool_calls
                extra = {}
                if ptc != None and self._model.startswith("o") == False:
                    extra["parallel_tool_calls"] = ptc 
                
                trunc = NOT_GIVEN
                if self._model == "computer-use-preview":
                    trunc = "auto"

                text = NOT_GIVEN
                if output_schema != None:
                    text = {
                        "format" : {
                            "type" : "json_schema",
                            "name" : response_name,
                            "schema" :  response_schema,
                            "strict" : True,
                        }
                    }

                reasoning = NOT_GIVEN
                if self._reasoning_effort != None:
                    reasoning = {
                        "effort" : self._reasoning_effort
                    }

                previous_response_id = NOT_GIVEN
                if context.previous_response_id != None:
                    previous_response_id = context.previous_response_id
                
                stream = event_handler != None
                
                for i in range(self._retries + 1):
                    if range == self._retries:
                        raise RoomException("exceeded maximum attempts calling openai")
                    try:
                        response = await openai.responses.create(
                            stream=stream,
                            model = self._model,
                            input = context.messages,
                            tools = open_ai_tools,
                            text = text,
                            previous_response_id=previous_response_id,
                            reasoning=reasoning,
                            truncation=trunc
                        )
                        break
                    except Exception as e:
                        logger.error(f"error calling openai attempt: {i+1}", exc_info=e)
                        if i == self._retries:
                            raise
                

                async def handle_message(message):
                    

                    if message.type == "function_call":
                    
                        tasks = []

                        async def do_tool_call(tool_call: ResponseFunctionToolCall):
                            try:
                                tool_context = ToolContext(
                                    room=room,
                                    caller=room.local_participant,
                                )
                                tool_response = await tool_bundle.execute(context=tool_context, tool_call=tool_call)
                                logger.info(f"tool response {tool_response}")
                                return await tool_adapter.create_messages(context=context, tool_call=tool_call, room=room, response=tool_response)
                            except Exception as e:
                                logger.error(f"unable to complete tool call {tool_call}", exc_info=e)
                                room.developer.log_nowait(type="llm.error", data={ "participant_id" : room.local_participant.id, "participant_name" : room.local_participant.get_attribute("name"), "error" : f"{e}" })
                    
                                return [{
                                    "output" : json.dumps({"error":f"unable to complete tool call: {e}"}),
                                    "call_id" : tool_call.call_id,
                                    "type" : "function_call_output"
                                }]


                        tasks.append(asyncio.create_task(do_tool_call(message)))

                        results = await asyncio.gather(*tasks)

                        all_results = []
                        for result in results:
                            room.developer.log_nowait(type="llm.message", data={ "context" : context.id, "participant_id" : room.local_participant.id, "participant_name" : room.local_participant.get_attribute("name"), "message" : result })
                            all_results.extend(result)

                        return all_results, False

                    elif message.type == "computer_call" and tool_bundle.get_tool("computer_call"):
                        tool_context = ToolContext(
                            room=room,
                            caller=room.local_participant,
                        )
                        outputs = (await tool_bundle.get_tool("computer_call").execute(context=tool_context, arguments=message.to_dict(mode="json"))).outputs

                        return outputs, False
                    
                    elif message.type == "reasoning":
                        reasoning = tool_bundle.get_tool("reasoning_tool")
                        if reasoning != None:
                            await tool_bundle.get_tool("reasoning_tool").execute(context=tool_context, arguments=message.to_dict(mode="json"))
                        
                    elif message.type == "message":
                        
                        contents = message.content
                        if response_schema == None:
                            return [], False
                        else:
                            for content in contents:
                                # First try to parse the result
                                try:
                                    full_response = json.loads(content.text)
                                # sometimes open ai packs two JSON chunks seperated by newline, check if that's why we couldn't parse
                                except json.decoder.JSONDecodeError as e:
                                    for part in content.text.splitlines():
                                        if len(part.strip()) > 0:
                                            full_response = json.loads(part)
                                            
                                            try:
                                                self.validate(response=full_response, output_schema=response_schema)
                                            except Exception as e:
                                                logger.error("recieved invalid response, retrying", exc_info=e)
                                                error = { "role" : "user", "content" : "encountered a validation error with the output: {error}".format(error=e)}
                                                room.developer.log_nowait(type="llm.message", data={ "context" : message.id, "participant_id" : room.local_participant.id, "participant_name" : room.local_participant.get_attribute("name"), "message" : error })
                                                context.messages.append(error)
                                                continue
                                    
                                return [ full_response ], True
                    else:
                        raise RoomException("Unexpected response from OpenAI {response}".format(response=message))
                    
                    return [], False
                    
                if stream == False:
                    room.developer.log_nowait(type="llm.message", data={ "context" : context.id, "participant_id" : room.local_participant.id, "participant_name" : room.local_participant.get_attribute("name"), "response" : response.to_dict() })
                
                    context.create_response(response.id)

                    final_outputs = []
                    
                    for message in response.output:
                        outputs, done = await handle_message(message=message)
                        if done:
                            final_outputs.extend(outputs)
                        else:
                            for output in outputs:
                                context.messages.append(output)

                    if len(final_outputs) > 0:

                        return final_outputs[0]

                else:
                    
                    final_outputs = []
                    all_outputs = []
                    async for e in response:
                        
                        event : ResponseStreamEvent = e

                        event_handler(event)

                        if event.type == "response.completed":
                            context.create_response(event.response.id)
                         
                            context.messages.extend(all_outputs)

                            term = await self.check_for_termination(context=context, room=room)
                            if term:
                                text = ""
                                for output in event.response.output:
                                    if output.type == "message":
                                        for content in output.content:
                                            text += content.text

                                return text


                            all_outputs = []

                        elif event.type == "response.output_item.done":
                          
                            context.previous_messages.append(event.item.to_dict())
                        
                            outputs, done = await handle_message(message=event.item)
                            if done:
                                final_outputs.extend(outputs)
                            else:
                                for output in outputs:
                                    all_outputs.append(output)

                        if len(final_outputs) > 0:

                            return final_outputs[0]

                                        
                    
                        
        except APIStatusError as e:
            raise RoomException(f"Error from OpenAI: {e}")
            


  