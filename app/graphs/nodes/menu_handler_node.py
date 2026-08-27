from app.core.config import settings
from app.core.exceptions import RestaurantNotFoundError
from app.graphs.conversation_state import ConversationState
from app.graphs.nodes.tool_loop import run_with_tools
from app.guardrails.hallucination_guardrail import HallucinationGuardrail, UNVERIFIED_CLAIM_REPLY
from app.prompts.restaurant_a_prompt import RESTAURANT_A_PERSONA
from app.prompts.restaurant_b_prompt import RESTAURANT_B_PERSONA
from app.tools.menu_tool import get_item_details, search_menu

MENU_TOOLS = [search_menu, get_item_details]
_hallucination_guardrail = HallucinationGuardrail()


async def menu_handler_node(state: ConversationState) -> dict:
    if state["restaurant_id"] == settings.RESTAURANT_A_ID:
        persona = RESTAURANT_A_PERSONA
    elif state["restaurant_id"] == settings.RESTAURANT_B_ID:
        persona = RESTAURANT_B_PERSONA
    else:
        raise RestaurantNotFoundError(state["restaurant_id"])

    system_prompt = (
        f"{persona}\n\nYou do not have the menu memorized. Always use the "
        f"search_menu or get_item_details tools to answer any question about "
        f"dishes, prices, ingredients, or availability -- never state a price "
        f"or claim an item exists from memory alone. The restaurant_id to use "
        f"for tool calls is '{state['restaurant_id']}'."
    )

    reply_text, tool_results = await run_with_tools(
        system_prompt=system_prompt,
        messages=state["messages"],
        tools=MENU_TOOLS,
    )

    if not await _hallucination_guardrail.check(reply_text, tool_results):
        reply_text = UNVERIFIED_CLAIM_REPLY

    new_message = {"role": "assistant", "content": reply_text}
    updated_messages = state["messages"] + [new_message]
    updated_turn_count = state["turn_count"] + 1

    return {
        "messages": updated_messages,
        "turn_count": updated_turn_count,
    }
