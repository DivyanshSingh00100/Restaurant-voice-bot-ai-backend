import json

import redis.asyncio as redis

from app.core.config import settings

_redis_client: redis.Redis | None = None

def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.REDIS_URL, db=settings.REDIS_DB)
    return _redis_client

async def get_context(session_id: str) -> dict | None:
    redis_client = get_redis_client()
    raw = await redis_client.get(session_id)
    if raw is None:
        return None
    return json.loads(raw)

async def save_context(session_id: str, state: dict) -> None:
    redis_client = get_redis_client()
    await redis_client.set(session_id, json.dumps(state), ex=settings.REDIS_TTL)
