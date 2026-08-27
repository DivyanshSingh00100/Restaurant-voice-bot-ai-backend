import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.core.exceptions import RestaurantNotFoundError
from app.services import session_service


def _fake_livekit_api():
    fake_api = MagicMock()
    fake_api.room.create_room = AsyncMock()
    return fake_api


@pytest.mark.asyncio
async def test_start_session_creates_room_with_restaurant_id_metadata(monkeypatch):
    fake_api = _fake_livekit_api()
    monkeypatch.setattr(session_service, "get_livekit_api", lambda: fake_api)

    await session_service.start_session(settings.RESTAURANT_A_ID)

    fake_api.room.create_room.assert_awaited_once()
    create_room_request = fake_api.room.create_room.call_args.args[0]
    assert create_room_request.name.startswith(settings.RESTAURANT_A_ID)
    assert json.loads(create_room_request.metadata) == {
        "restaurant_id": settings.RESTAURANT_A_ID
    }


@pytest.mark.asyncio
async def test_start_session_returns_token_matching_the_created_room(monkeypatch):
    fake_api = _fake_livekit_api()
    monkeypatch.setattr(session_service, "get_livekit_api", lambda: fake_api)

    result = await session_service.start_session(settings.RESTAURANT_A_ID)

    create_room_request = fake_api.room.create_room.call_args.args[0]
    assert result.room_name == create_room_request.name
    assert result.url == settings.LIVEKIT_URL
    assert isinstance(result.token, str)
    assert len(result.token) > 0


@pytest.mark.asyncio
async def test_start_session_raises_for_unknown_restaurant(monkeypatch):
    fake_api = _fake_livekit_api()
    monkeypatch.setattr(session_service, "get_livekit_api", lambda: fake_api)

    with pytest.raises(RestaurantNotFoundError):
        await session_service.start_session("not-a-real-restaurant")

    fake_api.room.create_room.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_session_generates_unique_room_names(monkeypatch):
    fake_api = _fake_livekit_api()
    monkeypatch.setattr(session_service, "get_livekit_api", lambda: fake_api)

    first = await session_service.start_session(settings.RESTAURANT_A_ID)
    second = await session_service.start_session(settings.RESTAURANT_A_ID)

    assert first.room_name != second.room_name
