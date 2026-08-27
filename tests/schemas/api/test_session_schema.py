import pytest
from pydantic import ValidationError
from app.schemas.api.session_schema import SessionStartRequest, LiveKitToken, SessionStartResponse

def test_session_start_request_accepts_valid_data():
    request = SessionStartRequest(restaurant_id="restaurant-a")
    assert request.restaurant_id == "restaurant-a"

def test_session_start_request_rejects_missing_restaurant_id():
    with pytest.raises(ValidationError):
        SessionStartRequest()  # type: ignore[call-arg]

def test_livekit_token_accepts_valid_data():
    token = LiveKitToken(token="abc123", url="https//:example.com", room_name="room-1")
    assert token.room_name == "room-1"

def test_session_start_response_wraps_livekit_token():
    token = LiveKitToken(token="abc123", url="https//:example.com", room_name="room=1")
    response = SessionStartResponse(livekit_token=token)

    assert response.livekit_token.token == "abc123"