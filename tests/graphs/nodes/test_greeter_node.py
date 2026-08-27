from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.graphs.nodes import greeter_node as greeter_node_module
from app.graphs.nodes.greeter_node import greeter_node
from app.graphs.conversation_state import ConversationState


@pytest.mark.asyncio
async def test_greeter_node_returns_greeting_and_increments_turn(monkeypatch):
    fake_response = MagicMock()
    fake_response.choices[0].message.content = "Welcome! Here's today's menu..."

    fake_groq_client = MagicMock()
    fake_groq_client.chat.completions.create = AsyncMock(return_value=fake_response)

    monkeypatch.setattr(greeter_node_module, "groq_client", fake_groq_client)

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [],
        "turn_count": 0,
        "cart": [],
    }

    result = await greeter_node(state)

    assert result["turn_count"] == 1
    assert result["messages"] == [{"role": "assistant", "content": "Welcome! Here's today's menu..."}]
    fake_groq_client.chat.completions.create.assert_awaited_once()