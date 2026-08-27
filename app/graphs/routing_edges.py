from app.graphs.conversation_state import ConversationState

# Keywords that send a turn to the order handler (the only node with cart
# tools bound). Live-call lesson: "Yes, I would like to add the ..." matched
# none of the original three keywords, routed to menu_handler, and the model
# then *claimed* the items were added -- a phantom-cart action claim the
# hallucination guardrail can't catch (it grounds prices, not actions). Keep
# this list generous: an order-ish turn that lands in menu_handler can't
# touch the cart at all, while a menu-ish turn landing in order_handler can
# still answer via get_item_details.
ORDER_KEYWORDS = [
    "order",
    "want",
    "i'll have",
    "add",
    "cart",
    "checkout",
    "check out",
    "would like",
    "confirm",
]

# Live-call lesson: "Can I speak to a staff member?" matched none of the
# original three keywords, fell through to menu_handler (no escalation
# tools bound there), and the model fabricated a fake "I'll flag this for a
# manager" reply instead of actually escalating.
ESCALATION_KEYWORDS = [
    "human",
    "person",
    "agent",
    "staff",
    "manager",
    "supervisor",
    "representative",
    "someone",
]


def _get_last_user_message(state: ConversationState) -> str:
    user_messages = [m["content"] for m in state["messages"] if m["role"] == "user"]
    if not user_messages:
        return ""
    return user_messages[-1].lower()


def route_initial_turn(state: ConversationState) -> str:
    if state["turn_count"] == 0:
        return "greeter"

    last_message = _get_last_user_message(state)

    if any(keyword in last_message for keyword in ESCALATION_KEYWORDS):
        return "escalation_handler"
    if any(keyword in last_message for keyword in ORDER_KEYWORDS):
        return "order_handler"

    return "menu_handler"


def route_after_handling(state: ConversationState) -> str:
    last_message = _get_last_user_message(state)

    if any(keyword in last_message for keyword in ESCALATION_KEYWORDS):
        return "escalation_handler"

    return "end"