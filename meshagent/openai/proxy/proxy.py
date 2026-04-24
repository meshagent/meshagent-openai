from openai import AsyncOpenAI
import logging
import json
import httpx
import os
from typing import Optional
from meshagent.api.urls import meshagent_base_url
from meshagent.openai.version import __version__

logger = logging.getLogger("openai.client")
DEFAULT_USER_AGENT = f"meshagent/{__version__}"


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


def resolve_base_url(base_url: str | None = None) -> str:
    resolved = base_url
    if resolved is None:
        resolved = os.getenv("OPENAI_BASE_URL")
    if resolved is not None:
        resolved = resolved.strip() or None
    if resolved is not None:
        return resolved
    return f"{meshagent_base_url().rstrip('/')}/openai/v1"


def resolve_api_key(api_key: str | None = None) -> str | None:
    resolved = api_key
    if resolved is None:
        resolved = os.getenv("OPENAI_API_KEY")
    if resolved is None or resolved.strip() == "":
        resolved = os.getenv("MESHAGENT_TOKEN")
    if resolved is None:
        return None
    resolved = resolved.strip()
    return resolved or None


def resolve_user_agent(user_agent: str | None = None) -> str:
    resolved = user_agent.strip() if isinstance(user_agent, str) else ""
    return resolved or DEFAULT_USER_AGENT


def get_client(
    *,
    base_url: str | None = None,
    http_client: Optional[httpx.AsyncClient] = None,
    session: Optional[httpx.AsyncClient] = None,
    api_key: str | None = None,
    user_agent: str | None = None,
) -> AsyncOpenAI:
    resolved_http_client = http_client if http_client is not None else session
    resolved_base_url = resolve_base_url(base_url)
    resolved_api_key = resolve_api_key(api_key)
    kwargs: dict[str, object] = {}
    if resolved_http_client is not None:
        kwargs["http_client"] = resolved_http_client
    kwargs["base_url"] = resolved_base_url
    kwargs["default_headers"] = {"User-Agent": resolve_user_agent(user_agent)}
    if resolved_api_key is not None:
        kwargs["api_key"] = resolved_api_key
    return AsyncOpenAI(**kwargs)
