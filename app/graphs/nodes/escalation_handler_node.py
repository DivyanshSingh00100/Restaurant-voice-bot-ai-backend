from app.graphs.conversation_state import ConversationState
from app.prompts.escalation_prompt import ESCALATION_CLOSING_MESSAGE, ESCALATION_TRANSITION_MESSAGE
from app.tools.escalation_tool import notify_human_agent, request_call_transfer


def escalation_handler_node(state: ConversationState) -> dict:
    notify_human_agent.invoke({"reason": "Escalation triggered during call"})
    request_call_transfer.invoke({"session_id": state["restaurant_id"]}) 

    transition_message = {"role": "assistant", "content": ESCALATION_TRANSITION_MESSAGE}
    closing_message = {"role": "assistant", "content": ESCALATION_CLOSING_MESSAGE}
    updated_messages = state["messages"] + [transition_message, closing_message]
    updated_turn_count = state["turn_count"] + 1

    return {
        "messages": updated_messages,
        "turn_count": updated_turn_count,
    }