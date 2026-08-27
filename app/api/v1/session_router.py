from fastapi import APIRouter

from app.schemas.api.session_schema import SessionStartRequest, SessionStartResponse
from app.services import session_service

router = APIRouter(tags=["Session"])

@router.post("/session/start", response_model=SessionStartResponse)
async def session_start(request: SessionStartRequest) -> SessionStartResponse:
    livekit_token = await session_service.start_session(request.restaurant_id)
    return SessionStartResponse(livekit_token=livekit_token)
