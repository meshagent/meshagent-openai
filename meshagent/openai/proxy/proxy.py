from meshagent.api import RoomClient
from openai import AsyncOpenAI
import logging
import json
import httpx
from typing import Optional

logger = logging.getLogger("openai.client")


def _redact_headers(headers: httpx.Headers) -> dict:
    h = dict(headers)
    if "authorization" in {k.lower() for k in h.keys()}:
        # Remove any case variant of Authorization
        for k in list(h.keys()):
            if k.lower() == "authorization":
                h[k] = "***REDACTED***"
    return h


def _truncate_bytes(b: bytes, limit: int = 4000) -> str:
    # Avoid dumping giant base64 screenshots into logs
    s = b.decode("utf-8", errors="replace")
    return (
        s
        if len(s) <= limit
        else (s[:limit] + f"\n... (truncated, {len(s)} chars total)")
    )


async def log_request(request: httpx.Request):
    logging.info("==> %s %s", request.method, request.url)
    logging.info("headers=%s", json.dumps(_redact_headers(request.headers), indent=2))
    if request.content:
        logging.info("body=%s", _truncate_bytes(request.content))


async def log_response(response: httpx.Response):
    body = await response.aread()
    logging.info("<== %s %s", response.status_code, response.request.url)
    logging.info("headers=%s", json.dumps(_redact_headers(response.headers), indent=2))
    if body:
        logging.info("body=%s", _truncate_bytes(body))


def get_logging_httpx_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        event_hooks={"request": [log_request], "response": [log_response]},
        timeout=60.0,
    )


def get_client(
    *, room: RoomClient, http_client: Optional[httpx.AsyncClient] = None
) -> AsyncOpenAI:
    token: str = room.protocol.token

    # when running inside the room pod, the room.room_url currently points to the external url
    # so we need to use url off the protocol (if available).
    # TODO: room_url should be set properly, but may need a claim in the token to be set during call to say it is local
    url = getattr(room.protocol, "url")
    if url is None:
        logger.debug(
            f"protocol does not have url, openai client falling back to room url {room.room_url}"
        )
        url = room.room_url
    else:
        logger.debug(f"protocol had url, openai client will use {url}")

    room_proxy_url = f"{url}/v1"

    if room_proxy_url.startswith("ws:") or room_proxy_url.startswith("wss:"):
        room_proxy_url = room_proxy_url.replace("ws", "http", 1)

    openai = AsyncOpenAI(
        http_client=http_client,
        api_key=token,
        base_url=room_proxy_url,
        default_headers={"Meshagent-Session": room.session_id},
    )
    return openai
