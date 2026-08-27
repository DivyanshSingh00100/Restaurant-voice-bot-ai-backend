from typing import get_type_hints
from app.graphs.conversation_state import ConversationState

def test_conversation_state_has_expected_files():
    hints = get_type_hints(ConversationState)
    assert set(hints.keys()) == {"restaurant_id", "messages", "turn_count", "cart"}

def test_conversation_state_can_be_built_and_accessed():
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [],
        "turn_count": 0,
        "cart": [],
    }
    assert state["restaurant_id"] == "restaurant-a"
    assert state["turn_count"] == 0