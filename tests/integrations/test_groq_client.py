from unittest.mock import AsyncMock, MagicMock

import pytest
from groq import AsyncGroq
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.integrations import groq_client as groq_client_module
from app.integrations.groq_client import groq_client

def test_groq_client_is_configured():
    assert isinstance(groq_client, AsyncGroq)


@pytest.mark.asyncio
async def test_wrap_with_timing_calls_the_original_and_returns_its_result():
    fake_response = MagicMock()
    fake_create = AsyncMock(return_value=fake_response)
    timed_create = groq_client_module._wrap_with_timing(fake_create)

    result = await timed_create(model="openai/gpt-oss-120b", messages=[])

    assert result is fake_response
    fake_create.assert_awaited_once_with(model="openai/gpt-oss-120b", messages=[])


@pytest.mark.asyncio
async def test_wrap_with_timing_logs_a_conversation_event_with_session_id(monkeypatch):
    clear_contextvars()
    bind_contextvars(session_id="room-1")

    logged_events = []
    monkeypatch.setattr(
        groq_client_module.conversation_logger,
        "log_event",
        lambda session_id, event_type, **fields: logged_events.append((session_id, event_type, fields)),
    )

    fake_create = AsyncMock(return_value=MagicMock())
    timed_create = groq_client_module._wrap_with_timing(fake_create)

    await timed_create(model="openai/gpt-oss-120b", messages=[], tools=[MagicMock()])

    clear_contextvars()

    assert len(logged_events) == 1
    session_id, event_type, fields = logged_events[0]
    assert session_id == "room-1"
    assert event_type == "llm_call"
    assert fields["model"] == "openai/gpt-oss-120b"
    assert fields["has_tools"] is True
    assert isinstance(fields["duration_ms"], float)


@pytest.mark.asyncio
async def test_wrap_with_timing_logs_unknown_session_when_no_context_bound(monkeypatch):
    clear_contextvars()

    logged_events = []
    monkeypatch.setattr(
        groq_client_module.conversation_logger,
        "log_event",
        lambda session_id, event_type, **fields: logged_events.append((session_id, event_type, fields)),
    )

    fake_create = AsyncMock(return_value=MagicMock())
    timed_create = groq_client_module._wrap_with_timing(fake_create)

    await timed_create(model="openai/gpt-oss-120b", messages=[])

    assert logged_events[0][0] == "unknown"


@pytest.mark.asyncio
async def test_wrap_with_timing_reraises_and_does_not_swallow_errors(monkeypatch):
    clear_contextvars()
    monkeypatch.setattr(groq_client_module.conversation_logger, "log_event", lambda *a, **k: None)

    fake_create = AsyncMock(side_effect=RuntimeError("groq is down"))
    timed_create = groq_client_module._wrap_with_timing(fake_create)

    with pytest.raises(RuntimeError, match="groq is down"):
        await timed_create(model="openai/gpt-oss-120b", messages=[])