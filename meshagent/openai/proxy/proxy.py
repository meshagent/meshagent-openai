from openai import AsyncOpenAI
import logging
import json
import httpx
import os
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


def _truncate_bytes(b: bytes, limit: int = 128000) -> str:
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
    *,
    base_url: str | None = None,
    http_client: Optional[httpx.AsyncClient] = None,
    session: Optional[httpx.AsyncClient] = None,
    api_key: str | None = None,
) -> AsyncOpenAI:
    resolved_http_client = http_client if http_client is not None else session
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL")
    if base_url is not None:
        base_url = base_url.strip() or None
    kwargs: dict[str, object] = {}
    if resolved_http_client is not None:
        kwargs["http_client"] = resolved_http_client
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key is not None:
        kwargs["api_key"] = api_key
    return AsyncOpenAI(**kwargs)
