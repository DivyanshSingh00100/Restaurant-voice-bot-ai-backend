import pytest
from app.core.config import settings
from app.core.exceptions import MenuItemNotFoundError
from app.tools.order_tool import add_item_to_cart, confirm_order, get_order_status, remove_item_from_cart


def test_add_item_to_cart_adds_item():
    result = add_item_to_cart.invoke({
        "restaurant_id": settings.RESTAURANT_A_ID,
        "item_name": "Margherita Pizza",
        "cart": [],
    })
    assert result[0]["name"] == "Margherita Pizza"


def test_remove_item_from_cart_removes_item():
    cart = [{"name": "Margherita Pizza", "price": 12.99}, {"name": "Grilled Salmon", "price": 18.00}]

    result = remove_item_from_cart.invoke({"item_name": "Margherita Pizza", "cart": cart})

    assert result == [{"name": "Grilled Salmon", "price": 18.00}]


def test_remove_item_from_cart_raises_for_item_not_in_cart():
    cart = [{"name": "Margherita Pizza", "price": 12.99}]

    with pytest.raises(MenuItemNotFoundError):
        remove_item_from_cart.invoke({"item_name": "Grilled Salmon", "cart": cart})

def test_confirm_order_returns_items_and_total():
    cart = [{"name": "Margherita Pizza", "price": 12.99}]
    result = confirm_order.invoke({"cart": cart})
    assert result["total"] == pytest.approx(12.99)
    assert result["items"] == cart


def test_get_order_status_returns_items_and_total():
    cart = [{"name": "Margherita Pizza", "price": 12.99}]
    result = get_order_status.invoke({"cart": cart})
    assert result["total"] == pytest.approx(12.99)
    assert result["items"] == cart