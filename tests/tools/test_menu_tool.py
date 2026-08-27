from app.core.config import settings
from app.prompts.restaurant_a_prompt import RESTAURANT_A_MENU
from app.tools.menu_tool import search_menu, get_item_details
import pytest
from app.core.exceptions import MenuItemNotFoundError

def test_search_menu_returns_restaurant_a_menu():
    result = search_menu.invoke(settings.RESTAURANT_A_ID)
    assert result == RESTAURANT_A_MENU

def test_get_item_details_returns_matching_item():
    result = get_item_details.invoke({"restaurant_id": settings.RESTAURANT_A_ID, "item_name": "Margherita Pizza"})
    assert result["name"] == "Margherita Pizza"

def test_get_item_details_raises_for_unknown_item():
    with pytest.raises(MenuItemNotFoundError):
        get_item_details.invoke({"restaurant_id": settings.RESTAURANT_A_ID, "item_name": "Non-existent Dish"})