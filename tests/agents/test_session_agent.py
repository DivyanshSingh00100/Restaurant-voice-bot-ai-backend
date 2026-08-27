from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents import session_agent as session_agent_module
from app.agents.session_agent import OFF_TOPIC_REPLY, PROFANITY_REPLY, RestaurantVoiceAgent
from app.core.config import settings
from app.prompts.escalation_prompt import ESCALATION_CLOSING_MESSAGE


def _build_agent(graph_result):
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value=graph_result)

    agent = RestaurantVoiceAgent(
        restaurant_id=settings.RESTAURANT_A_ID,
        session_id="room-1",
        graph=fake_graph,
        instructions="You are a waiter.",
        llm_model=MagicMock(),
    )

    # ContentSafetyGuardrail is LLM-backed (calls Groq's guardrail model in
    # one combined call). Swap in a mock that defaults to "passes both" so
    # normal-flow tests don't make real network calls; tests that
    # specifically exercise blocking behavior override this return value.
    agent._content_safety_guardrail = MagicMock()
    agent._content_safety_guardrail.check = AsyncMock(return_value=(True, True))

    return agent, fake_graph


@pytest.mark.asyncio
async def test_on_user_turn_completed_stores_pending_text():
    agent, _ = _build_agent(graph_result={})
    fake_message = MagicMock()
    fake_message.text_content = "I'd like a pizza"

    await agent.on_user_turn_completed(MagicMock(), fake_message)

    assert agent._pending_user_text == "I'd like a pizza"


@pytest.mark.asyncio
async def test_on_enter_triggers_first_reply(monkeypatch):
    agent, _ = _build_agent(graph_result={})

    fake_session = MagicMock()
    fake_session.generate_reply = AsyncMock()
    monkeypatch.setattr(RestaurantVoiceAgent, "session", property(lambda self: fake_session))

    await agent.on_enter()

    fake_session.generate_reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_node_starts_fresh_state_when_nothing_in_redis(monkeypatch):
    graph_result = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "assistant", "content": "Welcome! Here's the menu."}],
        "turn_count": 1,
        "cart": [],
    }
    agent, fake_graph = _build_agent(graph_result)

    monkeypatch.setattr(
        session_agent_module.context_service, "get_context", AsyncMock(return_value=None)
    )
    fake_save = AsyncMock()
    monkeypatch.setattr(session_agent_module.context_service, "save_context", fake_save)

    chunks = [chunk async for chunk in agent.llm_node(MagicMock(), [], MagicMock())]

    assert chunks == ["Welcome! Here's the menu."]

    called_state = fake_graph.ainvoke.call_args.args[0]
    assert called_state == {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [],
        "turn_count": 0,
        "cart": [],
    }
    fake_save.assert_awaited_once_with("room-1", graph_result)


@pytest.mark.asyncio
async def test_llm_node_appends_pending_user_message_before_calling_graph(monkeypatch):
    existing_state = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "assistant", "content": "Welcome!"}],
        "turn_count": 1,
        "cart": [],
    }
    graph_result = {**existing_state, "turn_count": 2}
    agent, fake_graph = _build_agent(graph_result)
    agent._pending_user_text = "I'd like a pizza"

    monkeypatch.setattr(
        session_agent_module.context_service,
        "get_context",
        AsyncMock(return_value=existing_state),
    )
    monkeypatch.setattr(session_agent_module.context_service, "save_context", AsyncMock())

    async for _ in agent.llm_node(MagicMock(), [], MagicMock()):
        pass

    called_state = fake_graph.ainvoke.call_args.args[0]
    assert called_state["messages"] == [
        {"role": "assistant", "content": "Welcome!"},
        {"role": "user", "content": "I'd like a pizza"},
    ]
    assert agent._pending_user_text is None


@pytest.mark.asyncio
async def test_llm_node_hands_off_to_human_on_escalation_closing_message(monkeypatch):
    graph_result = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "assistant", "content": ESCALATION_CLOSING_MESSAGE}],
        "turn_count": 3,
        "cart": [],
    }
    agent, _ = _build_agent(graph_result)

    monkeypatch.setattr(
        session_agent_module.context_service, "get_context", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(session_agent_module.context_service, "save_context", AsyncMock())

    fake_session = MagicMock()
    monkeypatch.setattr(RestaurantVoiceAgent, "session", property(lambda self: fake_session))

    fake_handoff = MagicMock()
    monkeypatch.setattr(session_agent_module.escalation_agent, "handoff_to_human", fake_handoff)

    async for _ in agent.llm_node(MagicMock(), [], MagicMock()):
        pass

    fake_handoff.assert_called_once_with(fake_session)


@pytest.mark.asyncio
async def test_llm_node_does_not_hand_off_on_normal_reply(monkeypatch):
    graph_result = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "assistant", "content": "Here's the menu..."}],
        "turn_count": 1,
        "cart": [],
    }
    agent, _ = _build_agent(graph_result)

    monkeypatch.setattr(
        session_agent_module.context_service, "get_context", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(session_agent_module.context_service, "save_context", AsyncMock())

    fake_handoff = MagicMock()
    monkeypatch.setattr(session_agent_module.escalation_agent, "handoff_to_human", fake_handoff)

    async for _ in agent.llm_node(MagicMock(), [], MagicMock()):
        pass

    fake_handoff.assert_not_called()


@pytest.mark.asyncio
async def test_llm_node_blocks_off_topic_user_message_without_calling_graph(monkeypatch):
    agent, fake_graph = _build_agent(graph_result={})
    agent._pending_user_text = "What's the weather like today?"
    agent._content_safety_guardrail.check = AsyncMock(return_value=(False, True))

    monkeypatch.setattr(
        session_agent_module.context_service, "get_context", AsyncMock(return_value=None)
    )
    fake_save = AsyncMock()
    monkeypatch.setattr(session_agent_module.context_service, "save_context", fake_save)

    chunks = [chunk async for chunk in agent.llm_node(MagicMock(), [], MagicMock())]

    assert chunks == [OFF_TOPIC_REPLY]
    fake_graph.ainvoke.assert_not_called()

    saved_state = fake_save.call_args.args[1]
    assert saved_state["messages"] == [{"role": "assistant", "content": OFF_TOPIC_REPLY}]


@pytest.mark.asyncio
async def test_llm_node_blocks_profane_user_message_without_calling_graph(monkeypatch):
    agent, fake_graph = _build_agent(graph_result={})
    agent._pending_user_text = "This menu is crap"
    agent._content_safety_guardrail.check = AsyncMock(return_value=(True, False))

    monkeypatch.setattr(
        session_agent_module.context_service, "get_context", AsyncMock(return_value=None)
    )
    fake_save = AsyncMock()
    monkeypatch.setattr(session_agent_module.context_service, "save_context", fake_save)

    chunks = [chunk async for chunk in agent.llm_node(MagicMock(), [], MagicMock())]

    assert chunks == [PROFANITY_REPLY]
    fake_graph.ainvoke.assert_not_called()

    saved_state = fake_save.call_args.args[1]
    assert saved_state["messages"] == [{"role": "assistant", "content": PROFANITY_REPLY}]


@pytest.mark.asyncio
async def test_llm_node_checks_content_safety_with_a_single_call(monkeypatch):
    """Topic and profanity are checked via one combined call (not two
    separate guardrail round-trips) -- this is the whole point of merging
    them: one Groq request per turn instead of two."""
    agent, _ = _build_agent(graph_result={
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "assistant", "content": "ok"}],
        "turn_count": 1,
        "cart": [],
    })
    agent._pending_user_text = "I'd like a pizza"

    check_mock = AsyncMock(return_value=(True, True))
    agent._content_safety_guardrail.check = check_mock

    monkeypatch.setattr(
        session_agent_module.context_service, "get_context", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(session_agent_module.context_service, "save_context", AsyncMock())

    async for _ in agent.llm_node(MagicMock(), [], MagicMock()):
        pass

    check_mock.assert_awaited_once_with("I'd like a pizza")


@pytest.mark.asyncio
async def test_llm_node_redacts_pii_from_user_message_before_calling_graph(monkeypatch):
    graph_result = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "assistant", "content": "Got it, thanks!"}],
        "turn_count": 1,
        "cart": [],
    }
    agent, fake_graph = _build_agent(graph_result)
    agent._pending_user_text = "Email me the receipt at advik@example.com"

    monkeypatch.setattr(
        session_agent_module.context_service, "get_context", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(session_agent_module.context_service, "save_context", AsyncMock())

    async for _ in agent.llm_node(MagicMock(), [], MagicMock()):
        pass

    called_state = fake_graph.ainvoke.call_args.args[0]
    assert called_state["messages"] == [
        {"role": "user", "content": "Email me the receipt at [REDACTED]"}
    ]


@pytest.mark.asyncio
async def test_llm_node_logs_user_and_assistant_turns_to_the_conversation_log(monkeypatch):
    graph_result = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "assistant", "content": "Sure thing!"}],
        "turn_count": 1,
        "cart": [],
    }
    agent, _ = _build_agent(graph_result)
    agent._pending_user_text = "I'd like a pizza"

    monkeypatch.setattr(
        session_agent_module.context_service, "get_context", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(session_agent_module.context_service, "save_context", AsyncMock())

    logged_events = []
    monkeypatch.setattr(
        session_agent_module.conversation_logger,
        "log_event",
        lambda session_id, event_type, **fields: logged_events.append((session_id, event_type, fields)),
    )

    async for _ in agent.llm_node(MagicMock(), [], MagicMock()):
        pass

    event_types = [event_type for _, event_type, _ in logged_events]
    assert "user_turn" in event_types
    assert "assistant_turn" in event_types

    user_event = next(e for e in logged_events if e[1] == "user_turn")
    assert user_event[0] == "room-1"
    assert user_event[2]["text"] == "I'd like a pizza"

    assistant_event = next(e for e in logged_events if e[1] == "assistant_turn")
    assert assistant_event[0] == "room-1"
    assert assistant_event[2]["text"] == "Sure thing!"


@pytest.mark.asyncio
async def test_llm_node_redacts_pii_from_graph_reply_before_saving_and_speaking(monkeypatch):
    graph_result = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [
            {"role": "assistant", "content": "Sure, reach us at staff@example.com anytime."}
        ],
        "turn_count": 1,
        "cart": [],
    }
    agent, _ = _build_agent(graph_result)

    monkeypatch.setattr(
        session_agent_module.context_service, "get_context", AsyncMock(return_value=None)
    )
    fake_save = AsyncMock()
    monkeypatch.setattr(session_agent_module.context_service, "save_context", fake_save)

    chunks = [chunk async for chunk in agent.llm_node(MagicMock(), [], MagicMock())]

    assert chunks == ["Sure, reach us at [REDACTED] anytime."]
    saved_state = fake_save.call_args.args[1]
    assert saved_state["messages"][-1]["content"] == "Sure, reach us at [REDACTED] anytime."
