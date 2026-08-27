from app.core.config import settings
from app.graphs.conversation_state import ConversationState
from app.graphs.nodes.escalation_handler_node import escalation_handler_node
from app.prompts.escalation_prompt import ESCALATION_CLOSING_MESSAGE, ESCALATION_TRANSITION_MESSAGE


def test_escalation_handler_node_adds_handoff_messages():
    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "I want to talk to a person"}],
        "turn_count": 2,
        "cart": [],
    }

    result = escalation_handler_node(state)

    assert result["turn_count"] == 3
    assert result["messages"][-2] == {"role": "assistant", "content": ESCALATION_TRANSITION_MESSAGE}
    assert result["messages"][-1] == {"role": "assistant", "content": ESCALATION_CLOSING_MESSAGE}