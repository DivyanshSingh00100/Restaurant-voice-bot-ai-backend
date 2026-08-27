from langchain_core.tools import tool
from app.prompts.escalation_prompt import ESCALATION_CLOSING_MESSAGE, ESCALATION_TRANSITION_MESSAGE

@tool
def notify_human_agent(reason: str) -> dict:
    """Flag the current call for human escalation, with a reason."""
    return {"escalate": True, "reason": reason, "message": ESCALATION_TRANSITION_MESSAGE}

@tool
def request_call_transfer(session_id: str) -> dict:
    """Request that current call be transferred to a human staff member."""
    return {"transfer_requested": True, "session_id": session_id, "message": ESCALATION_CLOSING_MESSAGE}
