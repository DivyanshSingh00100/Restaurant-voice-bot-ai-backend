from unittest.mock import MagicMock

from app.agents import restaurant_a_agent as restaurant_a_agent_module
from app.agents.restaurant_a_agent import build_restaurant_a_agent
from app.agents.session_agent import RestaurantVoiceAgent
from app.core.config import settings
from app.graphs.restaurant_a_graph import restaurant_a_graph


def test_build_restaurant_a_agent_wires_correct_graph_and_restaurant_id(monkeypatch):
    fake_llm = MagicMock()
    monkeypatch.setattr(restaurant_a_agent_module, "build_llm", MagicMock(return_value=fake_llm))

    agent = build_restaurant_a_agent("room-1")

    assert isinstance(agent, RestaurantVoiceAgent)
    assert agent.restaurant_id == settings.RESTAURANT_A_ID
    assert agent.session_id == "room-1"
    assert agent.graph is restaurant_a_graph
