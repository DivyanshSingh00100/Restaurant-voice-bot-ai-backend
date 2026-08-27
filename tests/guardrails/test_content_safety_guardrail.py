import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.guardrails import content_safety_guardrail as content_safety_guardrail_module
from app.guardrails.content_safety_guardrail import ContentSafetyGuardrail


def _fake_groq_response(off_topic: int, profanity: int):
    fake_message = MagicMock()
    fake_message.content = json.dumps({"off_topic": off_topic, "profanity": profanity})
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    return fake_response


@pytest.mark.asyncio
async def test_content_safety_guardrail_allows_clean_on_topic_text(monkeypatch):
    fake_create = AsyncMock(return_value=_fake_groq_response(off_topic=0, profanity=0))
    monkeypatch.setattr(
        content_safety_guardrail_module.groq_client.chat.completions, "create", fake_create
    )

    guardrail = ContentSafetyGuardrail()
    assert await guardrail.check("I'd like to order a pizza") == (True, True)


@pytest.mark.asyncio
async def test_content_safety_guardrail_blocks_off_topic_text(monkeypatch):
    fake_create = AsyncMock(return_value=_fake_groq_response(off_topic=1, profanity=0))
    monkeypatch.setattr(
        content_safety_guardrail_module.groq_client.chat.completions, "create", fake_create
    )

    guardrail = ContentSafetyGuardrail()
    assert await guardrail.check("What's the weather like today?") == (False, True)


@pytest.mark.asyncio
async def test_content_safety_guardrail_blocks_profane_text(monkeypatch):
    fake_create = AsyncMock(return_value=_fake_groq_response(off_topic=0, profanity=1))
    monkeypatch.setattr(
        content_safety_guardrail_module.groq_client.chat.completions, "create", fake_create
    )

    guardrail = ContentSafetyGuardrail()
    assert await guardrail.check("This damn order is late") == (True, False)


@pytest.mark.asyncio
async def test_content_safety_guardrail_makes_exactly_one_call_using_the_guardrail_model(monkeypatch):
    # The whole point of merging topic+profanity: one Groq round-trip per
    # turn instead of two.
    fake_create = AsyncMock(return_value=_fake_groq_response(off_topic=0, profanity=0))
    monkeypatch.setattr(
        content_safety_guardrail_module.groq_client.chat.completions, "create", fake_create
    )

    guardrail = ContentSafetyGuardrail()
    await guardrail.check("Hello, can I order a pizza?")

    fake_create.assert_awaited_once()
    assert (
        fake_create.call_args.kwargs["model"]
        == content_safety_guardrail_module.settings.GROQ_GUARDRAIL_MODEL
    )


@pytest.mark.asyncio
async def test_content_safety_guardrail_fails_open_on_classifier_error(monkeypatch):
    fake_create = AsyncMock(side_effect=RuntimeError("groq is down"))
    monkeypatch.setattr(
        content_safety_guardrail_module.groq_client.chat.completions, "create", fake_create
    )

    guardrail = ContentSafetyGuardrail()
    assert await guardrail.check("Hello, can I order a pizza?") == (True, True)


@pytest.mark.asyncio
async def test_content_safety_guardrail_fails_open_on_malformed_json(monkeypatch):
    fake_message = MagicMock()
    fake_message.content = "not valid json"
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    fake_create = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(
        content_safety_guardrail_module.groq_client.chat.completions, "create", fake_create
    )

    guardrail = ContentSafetyGuardrail()
    assert await guardrail.check("Hello, can I order a pizza?") == (True, True)
