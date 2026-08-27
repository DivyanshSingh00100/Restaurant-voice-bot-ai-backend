from langchain_core.tools import tool
from app.services.menu_service import get_menu
from app.core.exceptions import MenuItemNotFoundError

@tool 
def search_menu(restaurant_id: str) -> list[dict]:
    """Look up the full menu for a given restaurant."""
    return get_menu(restaurant_id)

@tool
def get_item_details(restaurant_id: str, item_name: str) -> dict:
    """Get details for a specific item on the menu at a restaurant, including it's price"""
    menu = get_menu(restaurant_id)
    for item in menu:
        if item["name"] == item_name:
            return item
    raise MenuItemNotFoundError(item_name)