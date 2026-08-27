from typing import TypedDict

class ConversationState(TypedDict):
    restaurant_id: str
    messages: list
    turn_count: int
    cart: list