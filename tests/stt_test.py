import math
import os
import pytest
import wave
from io import BytesIO

from meshagent.openai.proxy import get_client
from meshagent.openai.tools.stt import _transcribe
from meshagent.tools import TextContent


class _FakeTranscriptions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return "transcript"


class _FakeAudio:
    def __init__(self):
        self.transcriptions = _FakeTranscriptions()


class _FakeOpenAIClient:
    def __init__(self):
        self.audio = _FakeAudio()


def _should_run_live_openai_tests() -> bool:
    return (
        os.getenv("RUN_OPENAI_LIVE_TESTS") == "1"
        and isinstance(os.getenv("OPENAI_API_KEY"), str)
        and os.getenv("OPENAI_API_KEY", "").strip() != ""
    )


def _tiny_wav_bytes() -> bytes:
    sample_rate = 16_000
    duration_seconds = 0.35
    frames = int(sample_rate * duration_seconds)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(frames):
            sample = int(
                0.1 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate)
            )
            wav.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True))
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_transcribe_forwards_timestamp_granularities_to_openai_client():
    client = _FakeOpenAIClient()

    result = await _transcribe(
        client=client,
        data=b"audio-bytes",
        filename="sample.wav",
        model="whisper-1",
        response_format="text",
        timestamp_granularities=["word", "segment"],
        prompt="names",
        language="en",
    )

    assert isinstance(result, TextContent)
    assert result.text == "transcript"
    assert client.audio.transcriptions.calls == [
        {
            "model": "whisper-1",
            "response_format": "text",
            "file": client.audio.transcriptions.calls[0]["file"],
            "prompt": "names",
            "language": "en",
            "timestamp_granularities": ["word", "segment"],
            "stream": False,
        }
    ]
    assert client.audio.transcriptions.calls[0]["file"].name == "sample.wav"
    assert client.audio.transcriptions.calls[0]["file"].getvalue() == b"audio-bytes"


@pytest.mark.skipif(
    not _should_run_live_openai_tests(),
    reason="set RUN_OPENAI_LIVE_TESTS=1 and OPENAI_API_KEY to run live OpenAI STT tests",
)
@pytest.mark.asyncio
async def test_live_transcribe_returns_text_content_from_openai_provider():
    result = await _transcribe(
        client=get_client(),
        data=_tiny_wav_bytes(),
        filename="tone.wav",
        model=os.getenv("OPENAI_STT_LIVE_TEST_MODEL", "gpt-4o-mini-transcribe"),
        response_format="text",
        prompt=None,
        language="en",
    )

    assert isinstance(result, TextContent)
    assert isinstance(result.text, str)
