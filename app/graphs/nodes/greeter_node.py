from app.graphs.conversation_state import ConversationState
from app.core.config import settings
from app.integrations.groq_client import groq_client
from app.prompts.restaurant_a_prompt import RESTAURANT_A_PERSONA
from app.prompts.restaurant_b_prompt import RESTAURANT_B_PERSONA
from app.core.exceptions import RestaurantNotFoundError

async def greeter_node(state: ConversationState) -> dict:
    if state["restaurant_id"] == settings.RESTAURANT_A_ID:
        persona = RESTAURANT_A_PERSONA
    elif state["restaurant_id"] == settings.RESTAURANT_B_ID:
        persona = RESTAURANT_B_PERSONA
    else:
        raise RestaurantNotFoundError(state["restaurant_id"])

  
    response = await groq_client.chat.completions.create(
            model= settings.GROQ_LLM_MODEL,
        messages= [
            {"role": "system", "content": persona},
            {
                "role": "user",
                "content": "Greet the customer in one short, warm sentence and ask what they're in the mood for tonight. Do not list menu items yet.",
            },
        ],
    )
    reply_text = response.choices[0].message.content

    new_message = {"role": "assistant", "content": reply_text}
    updated_messages = state["messages"] + [new_message]
    updated_turn_count = state["turn_count"] + 1

    return {
        "messages": updated_messages,
        "turn_count": updated_turn_count,
    }
