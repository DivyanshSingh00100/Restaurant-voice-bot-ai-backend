import pytest
from app.core.config import settings
from app.core.exceptions import RestaurantNotFoundError
from app.prompts.restaurant_a_prompt import RESTAURANT_A_MENU
from app.prompts.restaurant_b_prompt import RESTAURANT_B_MENU
from app.services.menu_service import get_menu

def test_get_menu_returns_returns_restaurant_a_menu():
    assert get_menu(settings.RESTAURANT_A_ID) == RESTAURANT_A_MENU

def test_get_menu_returns_returns_restaurant_b_menu():
    assert get_menu(settings.RESTAURANT_B_ID) == RESTAURANT_B_MENU

def test_get_menu_raises_for_unknown_request():
    with pytest.raises(RestaurantNotFoundError):
        get_menu("unknown-restaurant")
