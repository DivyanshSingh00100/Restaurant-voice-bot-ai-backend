from app.core.exceptions import MenuItemNotFoundError

def add_to_cart(cart: list, item_name: str, menu: list[dict]) -> list:
    for item in menu:
        if item["name"] == item_name:
            cart.append(dict(item))
            return cart
    raise MenuItemNotFoundError(item_name)

def remove_from_cart(cart: list, item_name: str) -> list:
    for item in cart:
        if item["name"] == item_name:
            cart.remove(item)
            return cart
    raise MenuItemNotFoundError(item_name)

def calculate_total(cart: list) ->float:
    total = 0.0
    for item in cart:
        total += item["price"]
    return total

