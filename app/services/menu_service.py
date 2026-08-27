from app.core.config import settings
from app.core.exceptions import RestaurantNotFoundError
from app.prompts.restaurant_a_prompt import RESTAURANT_A_MENU
from app.prompts.restaurant_b_prompt import RESTAURANT_B_MENU

def get_menu(restaurant_id: str) -> list[dict]:
    if restaurant_id == settings.RESTAURANT_A_ID:
        return RESTAURANT_A_MENU
    elif restaurant_id == settings.RESTAURANT_B_ID:
        return RESTAURANT_B_MENU
    else:
        raise RestaurantNotFoundError(restaurant_id)