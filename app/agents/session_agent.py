from collections.abc import AsyncIterable

from livekit.agents import Agent, ChatContext, ChatMessage, FunctionTool, ModelSettings, llm
from structlog.contextvars import bind_contextvars

from app.agents import escalation_agent
from app.graphs.conversation_state import ConversationState
from app.guardrails.content_safety_guardrail import ContentSafetyGuardrail
from app.guardrails.pii_guardrail import PiiGuardrail
from app.prompts.escalation_prompt import ESCALATION_CLOSING_MESSAGE
from app.services import context_service, conversation_logger

OFF_TOPIC_REPLY = "I can only help with your order here -- let's get back to the menu. What can I get started for you?"
PROFANITY_REPLY = "Let's keep things friendly here -- what can I get started for you?"


class RestaurantVoiceAgent(Agent):
    """
    Base voice agent shared by every restaurant.

    A restaurant-specific subclass (RestaurantAAgent, RestaurantBAgent) only
    supplies its own compiled LangGraph graph and prompt persona. Everything
    about *how* a call is actually driven -- loading state from Redis,
    feeding the graph, saving state back, detecting escalation, screening
    text through guardrails -- lives here once, so it isn't duplicated per
    restaurant.
    """

    def __init__(
        self,
        restaurant_id: str,
        session_id: str,
        graph,
        instructions: str,
        llm_model: llm.LLM,
    ):
        super().__init__(instructions=instructions, llm=llm_model)
        self.restaurant_id = restaurant_id
        self.session_id = session_id
        self.graph = graph
        self._pending_user_text: str | None = None
        self._content_safety_guardrail = ContentSafetyGuardrail()
        self._pii_guardrail = PiiGuardrail()

    async def on_enter(self) -> None:
        await self.session.generate_reply()

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        self._pending_user_text = new_message.text_content

    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        # Binds session_id for the rest of this task's call stack (structlog
        # contextvars propagate through awaited calls within the same
        # asyncio task) -- lets groq_client.py's timing wrapper attribute
        # every LLM call this turn to this session without threading
        # session_id through every guardrail/node/tool_loop call site.
        bind_contextvars(session_id=self.session_id)

        state = await context_service.get_context(self.session_id)
        if state is None:
            state: ConversationState = {
                "restaurant_id": self.restaurant_id,
                "messages": [],
                "turn_count": 0,
                "cart": [],
            }

        if self._pending_user_text:
            user_text = self._pending_user_text
            self._pending_user_text = None

            # One combined LLM-backed classification call for both topic and
            # profanity (see content_safety_guardrail.py) -- halves the
            # guardrail round-trips per turn versus two separate calls.
            topic_ok, profanity_ok = await self._content_safety_guardrail.check(user_text)

            if not topic_ok:
                reply_text = OFF_TOPIC_REPLY
                state["messages"] = state["messages"] + [
                    {"role": "assistant", "content": reply_text}
                ]
                await context_service.save_context(self.session_id, state)
                conversation_logger.log_event(self.session_id, "user_turn", text=user_text)
                conversation_logger.log_event(self.session_id, "assistant_turn", text=reply_text)
                yield reply_text
                return

            if not profanity_ok:
                reply_text = PROFANITY_REPLY
                state["messages"] = state["messages"] + [
                    {"role": "assistant", "content": reply_text}
                ]
                await context_service.save_context(self.session_id, state)
                conversation_logger.log_event(self.session_id, "user_turn", text=user_text)
                conversation_logger.log_event(self.session_id, "assistant_turn", text=reply_text)
                yield reply_text
                return

            if not await self._pii_guardrail.check(user_text):
                user_text = self._pii_guardrail.redact(user_text)

            conversation_logger.log_event(self.session_id, "user_turn", text=user_text)
            state["messages"] = state["messages"] + [{"role": "user", "content": user_text}]

        result: ConversationState = await self.graph.ainvoke(state)

        reply_text = result["messages"][-1]["content"]
        if not await self._pii_guardrail.check(reply_text):
            reply_text = self._pii_guardrail.redact(reply_text)
            result["messages"][-1]["content"] = reply_text

        await context_service.save_context(self.session_id, result)
        conversation_logger.log_event(self.session_id, "assistant_turn", text=reply_text)

        yield reply_text

        if reply_text == ESCALATION_CLOSING_MESSAGE:
            escalation_agent.handoff_to_human(self.session)
