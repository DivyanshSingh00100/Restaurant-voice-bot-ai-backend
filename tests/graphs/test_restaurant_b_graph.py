from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.graphs.conversation_state import ConversationState
from app.graphs.nodes import greeter_node as greeter_node_module
from app.graphs.restaurant_b_graph import restaurant_b_graph


@pytest.mark.asyncio
async def test_restaurant_b_graph_greets_on_first_turn(monkeypatch):
    fake_response = MagicMock()
    fake_response.choices[0].message.content = "Welcome to Spice Garden!"

    fake_groq_client = MagicMock()
    fake_groq_client.chat.completions.create = AsyncMock(return_value=fake_response)

    monkeypatch.setattr(greeter_node_module, "groq_client", fake_groq_client)

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_B_ID,
        "messages": [],
        "turn_count": 0,
        "cart": [],
    }

    result = await restaurant_b_graph.ainvoke(state)

    assert result["turn_count"] == 1
    assert result["messages"][-1]["content"] == "Welcome to Spice Garden!"