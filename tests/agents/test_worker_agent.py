import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents import worker_agent as worker_agent_module
from app.agents.worker_agent import entrypoint, get_restaurant_id_from_room
from app.core.config import settings
from app.core.exceptions import RestaurantNotFoundError


def test_get_restaurant_id_from_room_parses_metadata():
    fake_ctx = MagicMock()
    fake_ctx.room.metadata = json.dumps({"restaurant_id": settings.RESTAURANT_B_ID})

    assert get_restaurant_id_from_room(fake_ctx) == settings.RESTAURANT_B_ID


def test_get_restaurant_id_from_room_returns_empty_string_when_no_metadata():
    fake_ctx = MagicMock()
    fake_ctx.room.metadata = ""

    assert get_restaurant_id_from_room(fake_ctx) == ""


@pytest.mark.asyncio
async def test_entrypoint_dispatches_to_restaurant_a_agent(monkeypatch):
    fake_ctx = MagicMock()
    fake_ctx.connect = AsyncMock()
    fake_ctx.room.metadata = json.dumps({"restaurant_id": settings.RESTAURANT_A_ID})
    fake_ctx.room.name = "room-1"

    fake_agent = MagicMock()
    fake_build_a = MagicMock(return_value=fake_agent)
    fake_build_b = MagicMock()
    monkeypatch.setattr(worker_agent_module, "build_restaurant_a_agent", fake_build_a)
    monkeypatch.setattr(worker_agent_module, "build_restaurant_b_agent", fake_build_b)

    fake_session = MagicMock()
    fake_session.start = AsyncMock()
    monkeypatch.setattr(worker_agent_module, "build_agent_session", MagicMock(return_value=fake_session))

    await entrypoint(fake_ctx)

    fake_build_a.assert_called_once_with("room-1")
    fake_build_b.assert_not_called()
    fake_session.start.assert_awaited_once_with(room=fake_ctx.room, agent=fake_agent)


@pytest.mark.asyncio
async def test_entrypoint_dispatches_to_restaurant_b_agent(monkeypatch):
    fake_ctx = MagicMock()
    fake_ctx.connect = AsyncMock()
    fake_ctx.room.metadata = json.dumps({"restaurant_id": settings.RESTAURANT_B_ID})
    fake_ctx.room.name = "room-2"

    fake_agent = MagicMock()
    fake_build_a = MagicMock()
    fake_build_b = MagicMock(return_value=fake_agent)
    monkeypatch.setattr(worker_agent_module, "build_restaurant_a_agent", fake_build_a)
    monkeypatch.setattr(worker_agent_module, "build_restaurant_b_agent", fake_build_b)

    fake_session = MagicMock()
    fake_session.start = AsyncMock()
    monkeypatch.setattr(worker_agent_module, "build_agent_session", MagicMock(return_value=fake_session))

    await entrypoint(fake_ctx)

    fake_build_b.assert_called_once_with("room-2")
    fake_build_a.assert_not_called()
    fake_session.start.assert_awaited_once_with(room=fake_ctx.room, agent=fake_agent)


@pytest.mark.asyncio
async def test_entrypoint_raises_for_unknown_restaurant_id(monkeypatch):
    fake_ctx = MagicMock()
    fake_ctx.connect = AsyncMock()
    fake_ctx.room.metadata = json.dumps({"restaurant_id": "not-a-real-restaurant"})
    fake_ctx.room.name = "room-3"

    with pytest.raises(RestaurantNotFoundError):
        await entrypoint(fake_ctx)


@pytest.mark.asyncio
async def test_wait_for_restaurant_id_returns_immediately_when_already_present():
    fake_ctx = MagicMock()
    fake_ctx.room.metadata = json.dumps({"restaurant_id": settings.RESTAURANT_A_ID})

    result = await worker_agent_module.wait_for_restaurant_id(fake_ctx)

    assert result == settings.RESTAURANT_A_ID


@pytest.mark.asyncio
async def test_wait_for_restaurant_id_retries_until_metadata_replicates(monkeypatch):
    fake_ctx = MagicMock()
    fake_ctx.room.metadata = ""

    async def fake_sleep(seconds):
        # Simulate the room metadata finally replicating while we wait.
        fake_ctx.room.metadata = json.dumps({"restaurant_id": settings.RESTAURANT_B_ID})

    fake_sleep_mock = AsyncMock(side_effect=fake_sleep)
    monkeypatch.setattr(worker_agent_module.asyncio, "sleep", fake_sleep_mock)

    result = await worker_agent_module.wait_for_restaurant_id(fake_ctx)

    assert result == settings.RESTAURANT_B_ID
    fake_sleep_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_for_restaurant_id_gives_up_after_all_retries(monkeypatch):
    fake_ctx = MagicMock()
    fake_ctx.room.metadata = ""

    fake_sleep_mock = AsyncMock()
    monkeypatch.setattr(worker_agent_module.asyncio, "sleep", fake_sleep_mock)

    result = await worker_agent_module.wait_for_restaurant_id(fake_ctx)

    assert result == ""
    assert fake_sleep_mock.await_count == worker_agent_module.METADATA_RETRY_ATTEMPTS - 1
