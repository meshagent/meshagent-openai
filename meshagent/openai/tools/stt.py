from meshagent.tools import (
    ToolContext,
    LocalRoomTool,
    Toolkit,
    JsonContent,
    TextContent,
)
from meshagent.api import RoomClient
from openai import AsyncOpenAI
from pydantic import BaseModel
from meshagent.openai.proxy import get_client
from typing import Optional
import io
import pathlib
import os


async def _transcribe(
    *,
    client: AsyncOpenAI,
    data: bytes,
    model: str,
    filename: str,
    response_format: str,
    timestamp_granularities: list[str] = None,
    prompt: Optional[str] = None,
    language: Optional[str] = None,
):
    buf = io.BytesIO(data)
    buf.name = filename
    transcript: BaseModel = await client.audio.transcriptions.create(
        model=model,
        response_format=response_format,
        file=buf,
        prompt=prompt,
        language=language,
        timestamp_granularities=timestamp_granularities,
        stream=False,
    )

    if isinstance(transcript, str):
        return TextContent(text=transcript)

    return JsonContent(json=transcript.model_dump(mode="json"))


class OpenAIAudioFileSTT(LocalRoomTool):
    def __init__(
        self,
        *,
        room: RoomClient,
        client: Optional[AsyncOpenAI] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(
            room=room,
            name="openai-file-stt",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "model",
                    "path",
                    "response_format",
                    "timestamp_granularities",
                    "prompt",
                ],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "the path to a file in the room storage",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "a prompt. can improve the accuracy of the transcript",
                    },
                    "model": {
                        "type": "string",
                        "enum": [
                            "whisper-1",
                            "gpt-4o-mini-transcribe",
                            "gpt-4o-transcribe",
                        ],
                    },
                    "response_format": {
                        "type": "string",
                        "description": "text and json are supported for all models, srt, verbose_json, and vtt are only supported for whisper-1",
                        "enum": ["text", "json", "srt", "verbose_json", "vtt"],
                    },
                    "timestamp_granularities": {
                        "description": "timestamp_granularities are only valid with whisper-1",
                        "type": "array",
                        "items": {"type": "string", "enum": ["word", "segment"]},
                    },
                },
            },
            title="OpenAI audio file STT",
            description="transcribes an audio file to text",
        )
        self.client = client
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")
        if base_url is not None:
            base_url = base_url.strip() or None
        self.base_url = base_url

    async def execute(
        self,
        context: ToolContext,
        *,
        model: str,
        prompt: str,
        path: str,
        response_format: str,
        timestamp_granularities: list,
    ):
        del context
        file_data = await self.room.storage.download(path=path)
        client = self.client
        if client is None:
            client = get_client(base_url=self.base_url)

        return await _transcribe(
            client=client,
            data=file_data.data,
            model=model,
            prompt=prompt,
            filename=pathlib.Path(path).name,
            response_format=response_format,
        )


class OpenAISTTToolkit(Toolkit):
    def __init__(
        self,
        *,
        room: RoomClient,
        base_url: Optional[str] = None,
    ):
        super().__init__(
            name="openai-stt",
            description="tools for speech to text using openai",
            room=room,
            tools=[OpenAIAudioFileSTT(room=room, base_url=base_url)],
        )
