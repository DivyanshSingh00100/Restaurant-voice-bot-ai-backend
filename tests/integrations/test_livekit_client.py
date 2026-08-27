from livekit import api
from app.integrations.livekit_client import get_livekit_api
import pytest

@pytest.mark.asyncio
async def test_livekit_api_is_configured():
        livekit_api = get_livekit_api()
        assert isinstance(livekit_api, api.LiveKitAPI)