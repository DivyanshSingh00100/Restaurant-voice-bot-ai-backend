from livekit import api
from app.core.config import settings

_livekit_api: api.LiveKitAPI | None = None

def get_livekit_api() -> api.LiveKitAPI:
    global _livekit_api
    if _livekit_api is None:
        _livekit_api = api.LiveKitAPI(
            url=settings.LIVEKIT_URL,
            api_key=settings.LIVEKIT_API_KEY,
            api_secret=settings.LIVEKIT_API_SECRET
)
    return _livekit_api