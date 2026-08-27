from langchain_core.tools import tool
from app.services.menu_service import get_menu
from app.services.order_service import add_to_cart
from app.services.order_service import calculate_total
from app.services.order_service import remove_from_cart


@tool
def add_item_to_cart(restaurant_id: str, item_name: str, cart: list) -> list:
    """Add a menu item to the customer's cart by name."""
    menu = get_menu(restaurant_id)
    return add_to_cart(cart, item_name, menu)

@tool
def remove_item_from_cart(item_name: str, cart: list) -> list:
    """Remove one instance of a menu item from the customer's cart by name."""
    return remove_from_cart(cart, item_name)

@tool
def confirm_order(cart: list) -> dict:
    """Confirm the current order and return a summary with the total price."""
    total = calculate_total(cart)
    return {"items": cart, "total": total}

@tool
def get_order_status(cart: list) -> dict:
    """Get the current status of the order, including items and total so far."""
    total = calculate_total(cart)
    return {"items": cart, "total": total}