from app.graphs.conversation_state import ConversationState
from app.graphs.routing_edges import route_after_handling, route_initial_turn


def test_route_initial_turn_greets_on_first_turn():
    state: ConversationState = {"restaurant_id": "restaurant-a", "messages": [], "turn_count": 0, "cart": []}
    assert route_initial_turn(state) == "greeter"


def test_route_initial_turn_routes_to_escalation_on_human_request():
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [{"role": "user", "content": "Can I speak to a human?"}],
        "turn_count": 1,
        "cart": [],
    }
    assert route_initial_turn(state) == "escalation_handler"


def test_route_initial_turn_routes_to_escalation_on_staff_request():
    # Live-call regression: "Can I speak to a staff member?" matched none of
    # human/person/agent, fell through to menu_handler (no escalation tools),
    # and the model fabricated a fake "I'll flag this for a manager" reply.
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [{"role": "user", "content": "Can I speak to a staff member?"}],
        "turn_count": 1,
        "cart": [],
    }
    assert route_initial_turn(state) == "escalation_handler"


def test_route_initial_turn_routes_to_escalation_on_manager_request():
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [{"role": "user", "content": "Is there a manager I can talk to?"}],
        "turn_count": 1,
        "cart": [],
    }
    assert route_initial_turn(state) == "escalation_handler"


def test_route_initial_turn_routes_to_order_handler():
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [{"role": "user", "content": "I want to order a pizza"}],
        "turn_count": 1,
        "cart": [],
    }
    assert route_initial_turn(state) == "order_handler"


def test_route_initial_turn_routes_add_requests_to_order_handler():
    # Live-call regression: "I would like to add X" contains none of the old
    # keywords (order/want/i'll have), routed to menu_handler (which has no
    # cart tools), and the model then *claimed* the add happened -- phantom
    # cart entries the guardrail can't catch (it grounds prices, not actions).
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [{"role": "user", "content": "Yes, I would like to add the Margherita Pizza and Grilled Salmon."}],
        "turn_count": 1,
        "cart": [],
    }
    assert route_initial_turn(state) == "order_handler"


def test_route_initial_turn_routes_cart_questions_to_order_handler():
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [{"role": "user", "content": "What do I have in the cart right now?"}],
        "turn_count": 1,
        "cart": [],
    }
    assert route_initial_turn(state) == "order_handler"


def test_route_initial_turn_routes_checkout_to_order_handler():
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [{"role": "user", "content": "Let's check out please"}],
        "turn_count": 1,
        "cart": [],
    }
    assert route_initial_turn(state) == "order_handler"


def test_route_initial_turn_defaults_to_menu_handler():
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [{"role": "user", "content": "What do you recommend?"}],
        "turn_count": 1,
        "cart": [],
    }
    assert route_initial_turn(state) == "menu_handler"


def test_route_after_handling_escalates_on_human_request():
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [
            {"role": "user", "content": "Can I talk to a person?"},
            {"role": "assistant", "content": "Sure, let me connect you."},
        ],
        "turn_count": 2,
        "cart": [],
    }
    assert route_after_handling(state) == "escalation_handler"


def test_route_after_handling_escalates_on_staff_request():
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [
            {"role": "user", "content": "Can I speak to a staff member?"},
            {"role": "assistant", "content": "Sure, let me connect you."},
        ],
        "turn_count": 2,
        "cart": [],
    }
    assert route_after_handling(state) == "escalation_handler"


def test_route_after_handling_ends_normally():
    state: ConversationState = {
        "restaurant_id": "restaurant-a",
        "messages": [
            {"role": "user", "content": "Thanks, that's all"},
            {"role": "assistant", "content": "You're welcome!"},
        ],
        "turn_count": 2,
        "cart": [],
    }
    assert route_after_handling(state) == "end"