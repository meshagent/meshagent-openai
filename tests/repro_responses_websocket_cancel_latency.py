import asyncio
import json
import os
import time
from urllib.parse import urlparse, urlunparse

import aiohttp
from openai import AsyncOpenAI


def _responses_ws_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme == "https":
        scheme = "wss"
    elif parsed.scheme == "http":
        scheme = "ws"
    elif parsed.scheme in {"ws", "wss"}:
        scheme = parsed.scheme
    else:
        raise ValueError(f"unsupported base URL scheme: {parsed.scheme}")

    return urlunparse(
        (
            scheme,
            parsed.netloc,
            parsed.path.rstrip("/") + "/responses",
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _websocket_headers(openai: AsyncOpenAI) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in openai.default_headers.items():
        if isinstance(value, str):
            headers[key] = value

    headers.pop("Content-Type", None)
    headers.pop("Content-Length", None)
    return headers


def _response_id(payload: dict[str, object]) -> str | None:
    raw_response_id = payload.get("response_id")
    if isinstance(raw_response_id, str) and raw_response_id.strip() != "":
        return raw_response_id

    response = payload.get("response")
    if not isinstance(response, dict):
        return None

    nested_response_id = response.get("id")
    if isinstance(nested_response_id, str) and nested_response_id.strip() != "":
        return nested_response_id

    return None


def _is_terminal(payload: dict[str, object]) -> bool:
    payload_type = payload.get("type")
    return payload_type in {
        "response.done",
        "response.completed",
        "response.failed",
        "response.incomplete",
    }


async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key == "":
        raise SystemExit("OPENAI_API_KEY must be set")

    model = os.getenv("OPENAI_CANCEL_MODEL", "gpt-5.2")
    max_output_tokens = int(os.getenv("OPENAI_CANCEL_MAX_OUTPUT_TOKENS", "4096"))
    timeout_seconds = float(os.getenv("OPENAI_CANCEL_TIMEOUT_SECONDS", "120"))
    prompt = os.getenv(
        "OPENAI_CANCEL_PROMPT",
        "Output the numbers 1 through 2000, one per line, with no intro or outro.",
    )

    openai = AsyncOpenAI(api_key=api_key)
    websocket_url = _responses_ws_url(str(openai.base_url))
    websocket_headers = _websocket_headers(openai)

    create_payload = {
        "type": "response.create",
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": max_output_tokens,
    }

    print(
        json.dumps(
            {
                "event": "probe_start",
                "model": model,
                "websocket_url": websocket_url,
                "max_output_tokens": max_output_tokens,
            }
        ),
        flush=True,
    )

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(
            websocket_url,
            headers=websocket_headers,
            heartbeat=None,
            autoping=True,
        ) as websocket:
            start = time.perf_counter()
            await websocket.send_str(json.dumps(create_payload))

            response_id: str | None = None
            cancel_sent_at: float | None = None
            first_delta_at: float | None = None
            delta_count_before_cancel = 0
            delta_count_after_cancel = 0
            last_delta_after_cancel_at: float | None = None
            tail_events: list[dict[str, object | None]] = []

            while True:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=timeout_seconds,
                )
                now = time.perf_counter()

                if message.type != aiohttp.WSMsgType.TEXT:
                    print(
                        json.dumps(
                            {
                                "event": "unexpected_websocket_message",
                                "message_type": str(message.type),
                                "elapsed_s": round(now - start, 4),
                            }
                        ),
                        flush=True,
                    )
                    return

                payload = json.loads(message.data)
                if not isinstance(payload, dict):
                    continue

                payload_type = payload.get("type")
                current_response_id = _response_id(payload)
                if response_id is None and current_response_id is not None:
                    response_id = current_response_id

                tail_events.append(
                    {
                        "type": payload_type if isinstance(payload_type, str) else None,
                        "elapsed_s": round(now - start, 4),
                        "response_id": current_response_id,
                    }
                )
                tail_events = tail_events[-15:]

                if payload_type == "error":
                    print(
                        json.dumps(
                            {
                                "event": "server_error",
                                "payload": payload,
                            },
                            indent=2,
                        ),
                        flush=True,
                    )
                    return

                if payload_type == "response.created":
                    print(
                        json.dumps(
                            {
                                "event": "response_created",
                                "response_id": current_response_id,
                                "elapsed_s": round(now - start, 4),
                            }
                        ),
                        flush=True,
                    )
                    continue

                if payload_type == "response.output_text.delta":
                    if cancel_sent_at is None:
                        delta_count_before_cancel += 1
                        first_delta_at = now
                        cancel_payload: dict[str, str] = {"type": "response.cancel"}
                        if response_id is not None:
                            cancel_payload["response_id"] = response_id
                        await websocket.send_str(json.dumps(cancel_payload))
                        cancel_sent_at = time.perf_counter()
                        print(
                            json.dumps(
                                {
                                    "event": "cancel_sent",
                                    "response_id": response_id,
                                    "time_from_start_s": round(
                                        cancel_sent_at - start, 4
                                    ),
                                    "time_from_first_delta_s": round(
                                        cancel_sent_at - first_delta_at, 4
                                    ),
                                    "delta_count_before_cancel": (
                                        delta_count_before_cancel
                                    ),
                                }
                            ),
                            flush=True,
                        )
                    else:
                        delta_count_after_cancel += 1
                        last_delta_after_cancel_at = now
                    continue

                if cancel_sent_at is not None and _is_terminal(payload):
                    response = payload.get("response")
                    response_status = (
                        response.get("status") if isinstance(response, dict) else None
                    )
                    print(
                        json.dumps(
                            {
                                "event": "terminal_after_cancel",
                                "terminal_type": payload_type,
                                "response_id": current_response_id,
                                "time_from_start_s": round(now - start, 4),
                                "time_from_cancel_s": round(
                                    now - cancel_sent_at,
                                    4,
                                ),
                                "response_status": response_status,
                                "delta_count_after_cancel": delta_count_after_cancel,
                                "time_to_last_delta_after_cancel_s": (
                                    None
                                    if last_delta_after_cancel_at is None
                                    else round(
                                        last_delta_after_cancel_at - cancel_sent_at,
                                        4,
                                    )
                                ),
                                "tail_events": tail_events,
                            },
                            indent=2,
                        ),
                        flush=True,
                    )
                    return


if __name__ == "__main__":
    asyncio.run(main())
