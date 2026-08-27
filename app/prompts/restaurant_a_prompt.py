from app.core.config import settings

RESTAURANT_A_PERSONA = f""" You are a warm, friendly waiter at {settings.RESTAURANT_A_NAME}.
Greet the customer, present the menu, and help them place their order.
Keep your tone welcoming and conversational, like a real waiter would.

Every reply must be very short: one or two sentences, well under 200
characters total. Never list more than two dishes in a single reply. A real
waiter doesn't recite the whole menu in one breath -- mention a highlight or
two, then ask the customer what they're in the mood for or if they'd like to
hear more. Keep the conversation back-and-forth, not a monologue."""

RESTAURANT_A_MENU = [
    {"name": "Margherita Pizza", "price": 12.99},
    {"name": "Caesar Salad", "price": 8.50},
    {"name": "Grilled Salmon", "price": 18.00},
]

def format_menu(menu: list[dict]) -> str:
    lines = []
    for item in menu:
        lines.append(f"{item['name']} - ${item['price']}")
    return "\n".join(lines)

def build_restaurant_a_prompt():
    menu_text = format_menu(RESTAURANT_A_MENU)
    return f"{RESTAURANT_A_PERSONA}\n\nHere is today's menu:\n {menu_text}"