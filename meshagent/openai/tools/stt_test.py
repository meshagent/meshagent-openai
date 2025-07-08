import os
import asyncio
import pytest

from openai import AsyncOpenAI
from meshagent.tools import JsonResponse, TextResponse

from .tts import _transcribe


################################################################################
# Fixtures
################################################################################
@pytest.fixture(scope="session")
def client() -> AsyncOpenAI:
    """Real async OpenAI client – no mocks, hits the network."""
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@pytest.fixture(scope="session")
def audio_bytes() -> bytes:
    """Loads the test clip only once per session."""
    with open("harvard.wav", "rb") as fp:
        return fp.read()


################################################################################
# Tests – one for “text”, one for “json”.  Add more if you need other formats.
################################################################################
@pytest.mark.asyncio
async def test_transcribe_text(client, audio_bytes):
    """_transcribe should return non-empty TextResponse for plain-text format."""
    result = await asyncio.wait_for(
        _transcribe(
            client=client,
            data=audio_bytes,
            filename="harvard.wav",
            model="gpt-4o-mini-transcribe",
            prompt="",
            response_format="text",
        ),
        timeout=90,
    )

    # Basic sanity checks
    assert isinstance(result, TextResponse)
    assert result.text.strip() != ""


@pytest.mark.asyncio
async def test_transcribe_json(client, audio_bytes):
    """_transcribe should return a well-formed JsonResponse for JSON format."""
    result = await asyncio.wait_for(
        _transcribe(
            client=client,
            data=audio_bytes,
            filename="harvard.wav",
            model="gpt-4o-mini-transcribe",
            prompt="",
            response_format="json",
        ),
        timeout=90,
    )

    # Basic sanity checks
    assert isinstance(result, JsonResponse)
    assert isinstance(result.json["text"], str)


@pytest.mark.asyncio
async def test_transcribe_verbose_json(client, audio_bytes):
    """_transcribe should return a well-formed JsonResponse for JSON format."""
    result = await asyncio.wait_for(
        _transcribe(
            client=client,
            data=audio_bytes,
            filename="harvard.wav",
            model="whisper-1",
            prompt="",
            response_format="verbose_json",
        ),
        timeout=90,
    )

    # Basic sanity checks
    assert isinstance(result, JsonResponse)
    assert isinstance(result.json["segments"], list)
