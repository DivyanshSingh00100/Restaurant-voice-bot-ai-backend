from app.agents.session_agent import RestaurantVoiceAgent
from app.agents.voice_pipeline_agent import build_llm
from app.core.config import settings
from app.graphs.restaurant_b_graph import restaurant_b_graph
from app.prompts.restaurant_b_prompt import RESTAURANT_B_PERSONA


def build_restaurant_b_agent(session_id: str) -> RestaurantVoiceAgent:
    return RestaurantVoiceAgent(
        restaurant_id=settings.RESTAURANT_B_ID,
        session_id=session_id,
        graph=restaurant_b_graph,
        instructions=RESTAURANT_B_PERSONA,
        llm_model=build_llm(),
    )
