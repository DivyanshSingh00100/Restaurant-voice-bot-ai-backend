from app.prompts.escalation_prompt import ESCALATION_CLOSING_MESSAGE, ESCALATION_TRANSITION_MESSAGE
from app.tools.escalation_tool import notify_human_agent, request_call_transfer


def test_notify_human_agent_flags_escalation():
    result = notify_human_agent.invoke({"reason": "customer requested a human"})
    assert result["escalate"] is True
    assert result["message"] == ESCALATION_TRANSITION_MESSAGE


def test_request_call_transfer_flags_transfer():
    result = request_call_transfer.invoke({"session_id": "session-1"})
    assert result["transfer_requested"] is True
    assert result["message"] == ESCALATION_CLOSING_MESSAGE