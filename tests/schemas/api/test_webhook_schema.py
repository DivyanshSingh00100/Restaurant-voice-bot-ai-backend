from app.schemas.api.webhook_schema import ParticipantEvent, WebhookEvent

def test_webhook_event_defaults_to_none():
    event = WebhookEvent(event="room_finished", room_name="room-1")
    assert event.participant is None

def test_webhook_event_accepts_participant():
    participant = ParticipantEvent(participant_id="p1", participant_name="Divyansh")
    event = WebhookEvent(event="participant_joined", room_name="room-1", participant=participant)

    assert event.participant.participant_name == "Divyansh" # type: ignore[union-attr]