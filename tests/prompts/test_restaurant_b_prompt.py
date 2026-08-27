from app.prompts.restaurant_b_prompt import RESTAURANT_B_PERSONA, RESTAURANT_B_MENU, build_restaurant_b_prompt
from app.core.config import settings
from app.prompts.restaurant_a_prompt import format_menu

def test_restaurant_b_persona_includes_restaurant_name():
    assert settings.RESTAURANT_B_NAME in RESTAURANT_B_PERSONA

def test_restaurant_b_menu_includes_every_item_name():
    formatted = format_menu(RESTAURANT_B_MENU)

    for item in RESTAURANT_B_MENU:
        assert item["name"] in formatted

def test_restaurant_b_prompt_includes_persona_and_menu():
    prompt = build_restaurant_b_prompt()

    assert RESTAURANT_B_PERSONA in prompt
    for item in RESTAURANT_B_MENU:
        assert item["name"] in prompt