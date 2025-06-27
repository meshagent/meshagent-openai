import os
from meshagent.api import RoomClient
from openai import AsyncOpenAI

def get_client(*, room: RoomClient) -> AsyncOpenAI:

    token : str = room.protocol.token
    url : str = room.room_url
    
    room_proxy_url = f"{url}/v1"

    if room_proxy_url.startswith("ws:") or room_proxy_url.startswith("wss:"):
        room_proxy_url = room_proxy_url.replace("ws","http",1)

    openai=AsyncOpenAI(
        api_key=token,
        base_url=room_proxy_url,
        default_headers={
            "Meshagent-Session" : room.session_id
        }
    )
    return openai
