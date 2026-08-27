import asyncio
import json

from livekit.agents import AgentSession, JobContext, WorkerOptions, cli

from app.agents.restaurant_a_agent import build_restaurant_a_agent
from app.agents.restaurant_b_agent import build_restaurant_b_agent
from app.agents.voice_pipeline_agent import build_agent_session
from app.core.config import settings
from app.core.exceptions import RestaurantNotFoundError

METADATA_RETRY_ATTEMPTS = 5
METADATA_RETRY_DELAY_SECONDS = 0.5


def get_restaurant_id_from_room(ctx: JobContext) -> str:
    metadata = json.loads(ctx.room.metadata) if ctx.room.metadata else {}
    return metadata.get("restaurant_id", "")


async def wait_for_restaurant_id(ctx: JobContext) -> str:
    # Room metadata is written by our own FastAPI backend via a separate API
    # call just before the room is joined. Occasionally that write hasn't
    # finished replicating to whichever region the worker connects through
    # yet, so the very first read can come back empty. Give it a few short
    # retries before concluding the room genuinely has no restaurant_id.
    for attempt in range(METADATA_RETRY_ATTEMPTS):
        restaurant_id = get_restaurant_id_from_room(ctx)
        if restaurant_id:
            return restaurant_id
        if attempt < METADATA_RETRY_ATTEMPTS - 1:
            await asyncio.sleep(METADATA_RETRY_DELAY_SECONDS)
    return ""


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    restaurant_id = await wait_for_restaurant_id(ctx)
    session_id = ctx.room.name

    if restaurant_id == settings.RESTAURANT_A_ID:
        agent = build_restaurant_a_agent(session_id)
    elif restaurant_id == settings.RESTAURANT_B_ID:
        agent = build_restaurant_b_agent(session_id)
    else:
        raise RestaurantNotFoundError(restaurant_id)

    session: AgentSession = build_agent_session(restaurant_id, session_id)
    await session.start(room=ctx.room, agent=agent)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=settings.LIVEKIT_URL,
            api_key=settings.LIVEKIT_API_KEY,
            api_secret=settings.LIVEKIT_API_SECRET,
        )
    )
