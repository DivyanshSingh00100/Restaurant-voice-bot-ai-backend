from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
import structlog
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.health_router import router as health_router
from app.api.v1.session_router import router as session_router
from app.api.v1.webhook_router import router as webhook_router
from app.core.config import settings
from app.core.exceptions import RestaurantNotFoundError
from app.core.logging import configure_logging

logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Application Starting", environment=settings.ENVIRONMENT)

    yield

    logger.info("Application Shutting Down")

app = FastAPI(
    title="Restaurant Voice Bot",
    description="Real-Time Voice Ordering Assistant Powered By LiveKit And Groq",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router,  prefix="/api/v1")
app.include_router(session_router, prefix="/api/v1")
app.include_router(webhook_router, prefix="/api/v1")

@app.exception_handler(RestaurantNotFoundError)
async def restaurant_not_found_handler(request: Request, exc: RestaurantNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})