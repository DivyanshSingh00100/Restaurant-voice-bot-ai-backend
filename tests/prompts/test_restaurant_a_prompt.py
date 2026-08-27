from app.core.config import settings
from app.prompts.restaurant_a_prompt import RESTAURANT_A_PERSONA, RESTAURANT_A_MENU, format_menu, build_restaurant_a_prompt

def test_restaurant_a_persona_includes_restaurant_name():
    assert settings.RESTAURANT_A_NAME in RESTAURANT_A_PERSONA

def test_restaurant_a_menu_includes_every_item_name():
    formatted = format_menu(RESTAURANT_A_MENU)

    for item in RESTAURANT_A_MENU:
        assert item["name"] in formatted

def test_restaurant_a_prompt_includes_persona_and_menu():
    prompt = build_restaurant_a_prompt()

    assert RESTAURANT_A_PERSONA in prompt
    for item in RESTAURANT_A_MENU:
        assert item["name"] in prompt