from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.api.v1 import session_router as session_router_module
from app.core.exceptions import RestaurantNotFoundError
from app.main import app
from app.schemas.api.session_schema import LiveKitToken

client = TestClient(app)


def test_session_start_returns_livekit_token(monkeypatch):
    fake_token = LiveKitToken(token="fake-jwt", room_name="restaurant-a-abc123", url="wss://example.livekit.cloud")
    monkeypatch.setattr(
        session_router_module.session_service, "start_session", AsyncMock(return_value=fake_token)
    )

    response = client.post("/api/v1/session/start", json={"restaurant_id": "restaurant-a"})

    assert response.status_code == 200
    assert response.json() == {
        "livekit_token": {
            "token": "fake-jwt",
            "room_name": "restaurant-a-abc123",
            "url": "wss://example.livekit.cloud",
        }
    }


def test_session_start_rejects_missing_restaurant_id():
    response = client.post("/api/v1/session/start", json={})

    assert response.status_code == 422


def test_session_start_returns_404_for_unknown_restaurant(monkeypatch):
    monkeypatch.setattr(
        session_router_module.session_service,
        "start_session",
        AsyncMock(side_effect=RestaurantNotFoundError("not-a-real-restaurant")),
    )

    response = client.post("/api/v1/session/start", json={"restaurant_id": "not-a-real-restaurant"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Restaurant not found: not-a-real-restaurant"}
