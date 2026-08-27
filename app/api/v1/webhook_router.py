from fastapi import APIRouter, Request

router = APIRouter(tags=["Webhooks"])

@router.post("/webhooks")
async def webhooks(request: Request):
    return {"received": True}