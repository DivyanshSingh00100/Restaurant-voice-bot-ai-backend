import structlog
from livekit.agents import AgentSession, inference, llm
from livekit.agents.metrics import STTMetrics, TTSMetrics
from livekit.plugins import groq, openai, silero
from livekit.plugins.google.beta import gemini_tts

from app.core.config import settings
from app.services import conversation_logger
from app.services.menu_service import get_menu

logger = structlog.get_logger(__name__)


def _build_stt_prompt(restaurant_id: str) -> str:
    # Whisper biases transcription toward vocabulary seen in its prompt.
    # Without this, non-English menu names get phonetically mangled
    # (observed live: "margherita pizza" -> "Margarita Virgil"), and the
    # garbled transcript then cascades into wrong guardrail/LLM behavior.
    item_names = ", ".join(item["name"] for item in get_menu(restaurant_id))
    return f"Menu items: {item_names}."


def _build_tts():
    # Manual switch only (settings.TTS_PROVIDER) -- no mid-call auto-fallback.
    # Groq's Orpheus TTS preview tier hits 429 rate limits quickly during
    # testing; Gemini/Kokoro are alternatives that require editing .env and
    # restarting the worker, not automatic failover.
    # Case-insensitive: a casing typo here (e.g. "Kokoro") would otherwise
    # silently fall through to Groq with no error.
    provider = settings.TTS_PROVIDER.lower()
    if provider == "gemini":
        return gemini_tts.TTS(
            model=settings.GEMINI_TTS_MODEL,
            voice_name=settings.GEMINI_TTS_VOICE,
            api_key=settings.GOOGLE_API_KEY,
        )
    if provider == "kokoro":
        # Kokoro-FastAPI exposes an OpenAI-compatible endpoint; use the generic
        # openai.TTS class with a custom base_url pointing to the local server.
        # response_format must be explicit -- the plugin's default (unset,
        # which the OpenAI API interprets as mp3) doesn't match what
        # Kokoro-FastAPI actually returns, so frames silently fail to parse.
        # model MUST be "tts-1" here (not settings.KOKORO_TTS_MODEL) -- the
        # plugin picks its response-parsing code path by checking this exact
        # string against a hardcoded set {"tts-1", "tts-1-hd"}; anything else
        # (including "kokoro") routes to an SSE parser built for OpenAI's
        # newer models, which can't parse Kokoro-FastAPI's plain audio blob
        # response and silently yields zero audio frames. Kokoro-FastAPI
        # itself ignores the model field entirely, so this is purely a
        # protocol-compatibility shim, not a real model selection.
        return openai.TTS(
            model="tts-1",
            voice=settings.KOKORO_TTS_VOICE,
            api_key="not-needed",
            base_url=settings.KOKORO_API_BASE_URL,
            response_format="wav",
        )
    return groq.TTS(
        model=settings.GROQ_TTS_MODEL,
        voice=settings.GROQ_TTS_VOICE,
        api_key=settings.GROQ_API_KEY,
    )


def _log_metrics(event, session_id: str) -> None:
    m = event.metrics
    if isinstance(m, STTMetrics):
        logger.info(
            "stt_metrics",
            session_id=session_id,
            duration_s=m.duration,
            audio_duration_s=m.audio_duration,
            streamed=m.streamed,
        )
        conversation_logger.log_event(
            session_id,
            "stt_metrics",
            duration_s=m.duration,
            audio_duration_s=m.audio_duration,
            streamed=m.streamed,
        )
    elif isinstance(m, TTSMetrics):
        logger.info(
            "tts_metrics",
            session_id=session_id,
            duration_s=m.duration,
            ttfb_s=m.ttfb,
            audio_duration_s=m.audio_duration,
            characters=m.characters_count,
        )
        conversation_logger.log_event(
            session_id,
            "tts_metrics",
            duration_s=m.duration,
            ttfb_s=m.ttfb,
            audio_duration_s=m.audio_duration,
            characters=m.characters_count,
        )


def build_agent_session(restaurant_id: str, session_id: str) -> AgentSession:
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=groq.STT(
            model=settings.GROQ_STT_MODEL,
            language="en",
            api_key=settings.GROQ_API_KEY,
            prompt=_build_stt_prompt(restaurant_id),
        ),
        tts=_build_tts(),
        # Our llm_node writes straight to Redis as a side effect (state fetch,
        # graph.ainvoke, save_context) before yielding text. Preemptive
        # generation can invoke llm_node speculatively -- on a changed/interim
        # transcript -- and cancel it after the Redis write already landed,
        # leaving a phantom assistant turn in state and doubling real API
        # calls per user turn. llm_node isn't written to be safely
        # re-entrant/cancellable, so disable preemptive generation rather
        # than restructure it right now.
        preemptive_generation=False,
    )
    # metrics_collected is soft-deprecated in favor of session_usage_updated
    # (token/cost tracking) and ChatMessage.metrics (per-turn UX latency like
    # time-to-first-token) -- neither exposes per-component STT/TTS execution
    # duration as directly as this event still does, which is exactly what's
    # needed for latency work. Still functional, just logs a one-time
    # deprecation warning on registration.
    session.on("metrics_collected", lambda ev: _log_metrics(ev, session_id))
    return session


def build_llm() -> llm.LLM:
    # This is a placeholder passed to Agent's constructor because the base
    # class requires one. It is never actually called: RestaurantVoiceAgent
    # overrides llm_node and drives replies through our own LangGraph graph
    # instead of this LLM.
    return inference.LLM(
        f"groq/{settings.GROQ_LLM_MODEL}",
        api_key=settings.LIVEKIT_API_KEY,
        api_secret=settings.LIVEKIT_API_SECRET,
    )
