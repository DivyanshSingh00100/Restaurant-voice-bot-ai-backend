import pytest
from app.core.exceptions import AppError, RestaurantNotFoundError, SessionExpiredError

def test_app_error_is_an_exception():
    error = AppError()
    assert isinstance(error, Exception)

def test_restaurant_not_found_stores_id_and_message():
    error = RestaurantNotFoundError("xyz")
    assert error.restaurant_id == "xyz"
    assert str(error) == "Restaurant not found: xyz"
    assert isinstance(error, AppError)

def test_session_expired_error_stores_session_id():
    error = SessionExpiredError("pqr")
    assert error.session_id == "pqr"
    assert str(error) == "Session Expired: pqr"
    assert isinstance(error, AppError)

def test_restaurant_not_found_can_be_raised():
    with pytest.raises(RestaurantNotFoundError):
        raise RestaurantNotFoundError("abc")
    
def test_session_expired_can_be_raised():
    with pytest.raises(SessionExpiredError):
        raise SessionExpiredError("defg")