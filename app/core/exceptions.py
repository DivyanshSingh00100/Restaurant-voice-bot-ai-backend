class AppError(Exception):
    """Base class for all custom errors in this app."""
    pass

class RestaurantNotFoundError (AppError):
    """Raised when a restaurant ID does not match any configured restaurant."""

    def __init__ (self, restaurant_id: str):
        self.restaurant_id = restaurant_id
        super().__init__(f"Restaurant not found: {restaurant_id}")

class SessionExpiredError(AppError):
    """Raised when a call's Redis-backed conversation state has expired mid-call. """

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session Expired: {session_id}")

class MenuItemNotFoundError(AppError):
    """Raised when menu item is not found in the list"""

    def __init__(self, item_name: str):
        self.item_name = item_name
        super().__init__(f"Menu item not found: {item_name}")