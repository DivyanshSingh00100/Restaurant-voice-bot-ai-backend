from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.graphs.conversation_state import ConversationState
from app.graphs.nodes import menu_handler_node as menu_handler_node_module
from app.graphs.nodes.menu_handler_node import menu_handler_node
from app.guardrails.hallucination_guardrail import UNVERIFIED_CLAIM_REPLY


@pytest.mark.asyncio
async def test_menu_handler_node_returns_grounded_reply_and_increments_turn(monkeypatch):
    fake_run_with_tools = AsyncMock(
        return_value=("The Margherita Pizza is $12.99!", [{"name": "Margherita Pizza", "price": 12.99}])
    )
    monkeypatch.setattr(menu_handler_node_module, "run_with_tools", fake_run_with_tools)

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "What's good here?"}],
        "turn_count": 1,
        "cart": [],
    }

    result = await menu_handler_node(state)

    assert result["turn_count"] == 2
    assert result["messages"][-1] == {"role": "assistant", "content": "The Margherita Pizza is $12.99!"}
    fake_run_with_tools.assert_awaited_once()
    call_kwargs = fake_run_with_tools.call_args.kwargs
    assert call_kwargs["messages"] == state["messages"]
    tool_names = {t.name for t in call_kwargs["tools"]}
    assert tool_names == {"search_menu", "get_item_details"}


@pytest.mark.asyncio
async def test_menu_handler_node_replaces_ungrounded_reply_with_fallback(monkeypatch):
    fake_run_with_tools = AsyncMock(
        return_value=("The Margherita Pizza is $99.99!", [])
    )
    monkeypatch.setattr(menu_handler_node_module, "run_with_tools", fake_run_with_tools)

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "How much is the paneer?"}],
        "turn_count": 1,
        "cart": [],
    }

    result = await menu_handler_node(state)

    assert result["messages"][-1] == {"role": "assistant", "content": UNVERIFIED_CLAIM_REPLY}
