import json
import uuid
from datetime import timedelta

from livekit import api

from app.core.config import settings
from app.core.exceptions import RestaurantNotFoundError
from app.integrations.livekit_client import get_livekit_api
from app.schemas.api.session_schema import LiveKitToken


async def start_session(restaurant_id: str) -> LiveKitToken:
    if restaurant_id not in (settings.RESTAURANT_A_ID, settings.RESTAURANT_B_ID):
        raise RestaurantNotFoundError(restaurant_id)

    room_name = f"{restaurant_id}-{uuid.uuid4().hex[:8]}"

    livekit_api = get_livekit_api()
    await livekit_api.room.create_room(
        api.CreateRoomRequest(
            name=room_name,
            metadata=json.dumps({"restaurant_id": restaurant_id}),
        )
    )

    grants = api.VideoGrants(room_join=True, room=room_name)
    token = (
        api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
        .with_identity(f"caller-{uuid.uuid4().hex[:8]}")
        .with_grants(grants)
        .with_ttl(timedelta(seconds=settings.LIVEKIT_TOKEN_TTL))
        .to_jwt()
    )

    return LiveKitToken(token=token, room_name=room_name, url=settings.LIVEKIT_URL)
