import pytest
from app.core.exceptions import MenuItemNotFoundError
from app.services.order_service import add_to_cart, calculate_total, remove_from_cart
from app.prompts.restaurant_b_prompt import RESTAURANT_B_MENU


def test_add_to_cart_adds_matching_item():
    menu = [{"name": "Margherita Pizza", "price": 12.99}]
    cart = []

    result = add_to_cart(cart, "Margherita Pizza", menu)

    assert result == [{"name": "Margherita Pizza", "price": 12.99}]


def test_add_to_cart_appends_a_copy_not_the_menu_item_reference():
    menu_item = {"name": "Margherita Pizza", "price": 12.99}
    menu = [menu_item]
    cart = []

    result = add_to_cart(cart, "Margherita Pizza", menu)

    assert result[0] == menu_item
    assert result[0] is not menu_item


def test_add_to_cart_raises_for_unknown_item():
    menu = [{"name": "Margherita Pizza", "price": 12.99}]
    cart = []

    with pytest.raises(MenuItemNotFoundError):
        add_to_cart(cart, "Nonexistent Dish", menu) 

def test_calculate_total_sums_cart_price():
    cart = [
        {"name": "Margherita Pizza", "price": 12.99},
        {"name": "Caesar Salad", "price": 8.50},
    ]
    assert calculate_total(cart) == pytest.approx(21.49)

def test_calculate_total_when_cart_empty():
    assert calculate_total([]) == 0.0


def test_calculate_total_works_with_restaurant_b_menu_items():
    cart = [RESTAURANT_B_MENU[0], RESTAURANT_B_MENU[1]]
    assert calculate_total(cart) == pytest.approx(13.50 + 6.99)


def test_remove_from_cart_removes_matching_item():
    cart = [
        {"name": "Margherita Pizza", "price": 12.99},
        {"name": "Grilled Salmon", "price": 18.00},
    ]

    result = remove_from_cart(cart, "Margherita Pizza")

    assert result == [{"name": "Grilled Salmon", "price": 18.00}]


def test_remove_from_cart_removes_only_one_instance_when_duplicates_exist():
    # Live-call regression: cart had three Margherita Pizzas after repeated
    # adds; removing "one" must leave the other two, not wipe them all.
    cart = [
        {"name": "Margherita Pizza", "price": 12.99},
        {"name": "Margherita Pizza", "price": 12.99},
        {"name": "Margherita Pizza", "price": 12.99},
    ]

    result = remove_from_cart(cart, "Margherita Pizza")

    assert result == [
        {"name": "Margherita Pizza", "price": 12.99},
        {"name": "Margherita Pizza", "price": 12.99},
    ]


def test_remove_from_cart_raises_for_item_not_in_cart():
    cart = [{"name": "Margherita Pizza", "price": 12.99}]

    with pytest.raises(MenuItemNotFoundError):
        remove_from_cart(cart, "Grilled Salmon")