from pydantic import BaseModel

class ParticipantEvent(BaseModel):
    participant_id: str
    participant_name: str

class WebhookEvent(BaseModel):
    event: str
    room_name: str
    participant: ParticipantEvent | None = None