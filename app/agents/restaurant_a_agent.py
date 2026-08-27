from app.agents.session_agent import RestaurantVoiceAgent
from app.agents.voice_pipeline_agent import build_llm
from app.core.config import settings
from app.graphs.restaurant_a_graph import restaurant_a_graph
from app.prompts.restaurant_a_prompt import RESTAURANT_A_PERSONA


def build_restaurant_a_agent(session_id: str) -> RestaurantVoiceAgent:
    return RestaurantVoiceAgent(
        restaurant_id=settings.RESTAURANT_A_ID,
        session_id=session_id,
        graph=restaurant_a_graph,
        instructions=RESTAURANT_A_PERSONA,
        llm_model=build_llm(),
    )
