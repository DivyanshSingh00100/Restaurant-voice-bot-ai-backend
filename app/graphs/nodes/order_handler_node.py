from app.core.config import settings
from app.core.exceptions import RestaurantNotFoundError
from app.graphs.conversation_state import ConversationState
from app.graphs.nodes.tool_loop import run_with_tools
from app.guardrails.hallucination_guardrail import HallucinationGuardrail, UNVERIFIED_CLAIM_REPLY
from app.prompts.restaurant_a_prompt import RESTAURANT_A_PERSONA
from app.prompts.restaurant_b_prompt import RESTAURANT_B_PERSONA
from app.services.order_service import calculate_total
from app.tools.menu_tool import get_item_details
from app.tools.order_tool import add_item_to_cart, confirm_order, get_order_status, remove_item_from_cart

ORDER_TOOLS = [add_item_to_cart, remove_item_from_cart, confirm_order, get_order_status, get_item_details]
_hallucination_guardrail = HallucinationGuardrail()


async def order_handler_node(state: ConversationState) -> dict:
    if state["restaurant_id"] == settings.RESTAURANT_A_ID:
        persona = RESTAURANT_A_PERSONA
    elif state["restaurant_id"] == settings.RESTAURANT_B_ID:
        persona = RESTAURANT_B_PERSONA
    else:
        raise RestaurantNotFoundError(state["restaurant_id"])

    cart_total = calculate_total(state["cart"])
    system_prompt = (
        f"{persona}\n\nThe customer's current cart total is ${cart_total:.2f}. Use "
        f"add_item_to_cart to add anything new, remove_item_from_cart to remove one "
        f"instance of an item, and get_order_status or confirm_order to report "
        f"totals -- never state a new price or total from memory alone. The "
        f"restaurant_id to use for tool calls is '{state['restaurant_id']}', and the "
        f"current cart is {state['cart']}."
    )

    # Held as a dict (rather than passing the list straight through) so we
    # can read the final, real cart back out below: run_with_tools injects
    # this cart into every tool call in place of whatever the model
    # supplied, and keeps this entry updated with each call's real result
    # so it always reflects the true, accumulated cart -- never a
    # model-fabricated one.
    live_cart_override = {"cart": list(state["cart"])}

    reply_text, tool_results = await run_with_tools(
        system_prompt=system_prompt,
        messages=state["messages"],
        tools=ORDER_TOOLS,
        arg_overrides=live_cart_override,
    )

    if not await _hallucination_guardrail.check(reply_text, tool_results):
        reply_text = UNVERIFIED_CLAIM_REPLY

    new_message = {"role": "assistant", "content": reply_text}
    updated_messages = state["messages"] + [new_message]
    updated_turn_count = state["turn_count"] + 1

    return {
        "messages": updated_messages,
        "turn_count": updated_turn_count,
        "cart": live_cart_override["cart"],
    }
