from app.prompts.escalation_prompt import ESCALATION_CLOSING_MESSAGE, ESCALATION_TRANSITION_MESSAGE

def test_escalation_transition_prompt_is_non_empty_string():
    transition = ESCALATION_TRANSITION_MESSAGE
    assert isinstance(transition, str) 
    assert len(transition) > 0

def test_escalation_closing_prompt_is_non_empty_string():
    closing = ESCALATION_CLOSING_MESSAGE
    assert isinstance(closing, str) 
    assert len(closing) > 0