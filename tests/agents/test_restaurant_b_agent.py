from unittest.mock import MagicMock

from app.agents import restaurant_b_agent as restaurant_b_agent_module
from app.agents.restaurant_b_agent import build_restaurant_b_agent
from app.agents.session_agent import RestaurantVoiceAgent
from app.core.config import settings
from app.graphs.restaurant_b_graph import restaurant_b_graph


def test_build_restaurant_b_agent_wires_correct_graph_and_restaurant_id(monkeypatch):
    fake_llm = MagicMock()
    monkeypatch.setattr(restaurant_b_agent_module, "build_llm", MagicMock(return_value=fake_llm))

    agent = build_restaurant_b_agent("room-2")

    assert isinstance(agent, RestaurantVoiceAgent)
    assert agent.restaurant_id == settings.RESTAURANT_B_ID
    assert agent.session_id == "room-2"
    assert agent.graph is restaurant_b_graph
