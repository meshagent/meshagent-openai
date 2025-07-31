from meshagent.api import RoomClient
from openai import AsyncOpenAI
import logging

logger = logging.getLogger("openai.client")


def get_client(*, room: RoomClient) -> AsyncOpenAI:
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
        api_key=token,
        base_url=room_proxy_url,
        default_headers={"Meshagent-Session": room.session_id},
    )
    return openai
