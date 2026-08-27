from unittest.mock import AsyncMock, MagicMock
import pytest
from app.services import context_service
import json
from app.core.config import settings

@pytest.mark.asyncio
async def test_save_context_service_calls_redis_set(monkeypatch):
    fake_redis = AsyncMock()
    monkeypatch.setattr(context_service, "get_redis_client", MagicMock(return_value=fake_redis))

    await context_service.save_context("session-1", {"turn_count": 1})

    fake_redis.set.assert_awaited_once()

@pytest.mark.asyncio
async def test_save_context_encodes_state_and_sets_ttl(monkeypatch):
    fake_redis = AsyncMock()
    monkeypatch.setattr(context_service, "get_redis_client", MagicMock(return_value=fake_redis))

    await context_service.save_context("session-1", {"turn_count": 1})

    fake_redis.set.assert_awaited_once_with(
        "session-1",
        json.dumps({"turn_count": 1}),
        ex=settings.REDIS_TTL,
    )

@pytest.mark.asyncio
async def test_get_context_returns_none_when_nothing_stored(monkeypatch):
    fake_redis = AsyncMock()
    fake_redis.get.return_value=None
    monkeypatch.setattr(context_service, "get_redis_client", MagicMock(return_value=fake_redis))

    result = await context_service.get_context("session-1")

    assert result is None

@pytest.mark.asyncio
async def test_get_context_returns_decoded_state_when_data_exists(monkeypatch):
    fake_redis = AsyncMock()
    fake_redis.get.return_value = json.dumps({"turn_count": 2})

    monkeypatch.setattr(context_service, "get_redis_client", MagicMock(return_value=fake_redis))

    result = await context_service.get_context("session-1")

    assert result == {"turn_count": 2}

@pytest.mark.asyncio
async def test_get_redis_client_is_lazily_created_once(monkeypatch):
    monkeypatch.setattr(context_service, "_redis_client", None)
    fake_instance = AsyncMock()
    fake_from_url = MagicMock(return_value=fake_instance)
    monkeypatch.setattr(context_service.redis, "from_url", fake_from_url)

    first = context_service.get_redis_client()
    second = context_service.get_redis_client()

    fake_from_url.assert_called_once()
    assert first is fake_instance
    assert second is fake_instance
