from pydantic import BaseModel

class SessionStartRequest(BaseModel):
    restaurant_id: str

class LiveKitToken(BaseModel):
    token: str
    room_name: str
    url: str

class SessionStartResponse(BaseModel):
    livekit_token: LiveKitToken