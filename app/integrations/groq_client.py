import time

import structlog
from groq import AsyncGroq
from structlog.contextvars import get_contextvars

from app.core.config import settings
from app.services import conversation_logger

logger = structlog.get_logger(__name__)


def _wrap_with_timing(create_fn):
    # session_id comes from structlog's contextvars (bound once per turn in
    # session_agent.py's llm_node), not a parameter -- groq_client is a
    # shared singleton with no notion of "which call session" on its own,
    # and threading session_id through every call site (guardrails,
    # tool_loop, greeter_node) would touch far more files for no benefit.
    # Contextvars propagate through the same asyncio task automatically.
    async def _timed_create(*args, **kwargs):
        start = time.monotonic()
        session_id = get_contextvars().get("session_id", "unknown")
        model = kwargs.get("model")
        has_tools = bool(kwargs.get("tools"))
        try:
            response = await create_fn(*args, **kwargs)
            duration_ms = round((time.monotonic() - start) * 1000, 1)
            logger.info("groq_llm_call", model=model, duration_ms=duration_ms, has_tools=has_tools)
            conversation_logger.log_event(
                session_id, "llm_call", model=model, duration_ms=duration_ms, has_tools=has_tools
            )
            return response
        except Exception:
            duration_ms = round((time.monotonic() - start) * 1000, 1)
            logger.warning("groq_llm_call_failed", model=model, duration_ms=duration_ms)
            conversation_logger.log_event(
                session_id, "llm_call_failed", model=model, duration_ms=duration_ms
            )
            raise

    return _timed_create


groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
groq_client.chat.completions.create = _wrap_with_timing(groq_client.chat.completions.create)
