from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    GROQ_API_KEY: str = ""
    GROQ_STT_MODEL: str = "whisper-large-v3"
    GROQ_LLM_MODEL: str = "openai/gpt-oss-120b"
    GROQ_TTS_MODEL: str = "canopylabs/orpheus-v1-english"
    GROQ_TTS_VOICE: str = "autumn"

    # Manual TTS provider switch for testing -- "groq" | "gemini". No
    # mid-call auto-fallback; change this and restart the worker to take
    # effect. Added because Groq's Orpheus TTS preview tier hits 429 rate
    # limits quickly, so Gemini is a way to keep testing without waiting
    # out the quota.
    TTS_PROVIDER: str = "groq"
    GOOGLE_API_KEY: str = ""
    GEMINI_TTS_MODEL: str = "gemini-3.1-flash-tts-preview"
    GEMINI_TTS_VOICE: str = "Kore"

    # Kokoro TTS server — self-hosted via Docker, exposes OpenAI-compatible
    # endpoint. Run locally: docker run -d -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:v0.1.4
    # No model setting -- Kokoro-FastAPI only ever runs the one model it's
    # loaded with and ignores whatever "model" value is sent to it.
    KOKORO_API_BASE_URL: str = "http://localhost:8880/v1"
    KOKORO_TTS_VOICE: str = "af_sarah"

    # Dedicated policy-following safety/classification model used by the
    # topic and profanity guardrails, kept separate from GROQ_LLM_MODEL since
    # it's a different model purpose-built for Trust & Safety classification
    # rather than conversational replies.
    GROQ_GUARDRAIL_MODEL: str = "openai/gpt-oss-safeguard-20b"

    LIVEKIT_URL: str = "wss://restaurant-ai-voice-bot-0et6v99h.livekit.cloud"
    LIVEKIT_API_KEY: str = ""
    LIVEKIT_API_SECRET: str = ""
    LIVEKIT_TOKEN_TTL: int = 3600
    LIVEKIT_WEBHOOK_SECRET: str = ""

    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 0
    REDIS_TTL: int = 1800

    RESTAURANT_A_ID: str = "restaurant-a"
    RESTAURANT_A_NAME: str = "The Grand Bistro"
    RESTAURANT_B_ID: str = "restaurant-b"
    RESTAURANT_B_NAME: str = "Spice Garden"

    GUARDRAIL_CONFIDENCE_THRESHOLD: float = 0.6
    GUARDRAIL_PII_ENABLED: bool = True
    GUARDRAIL_PROFANITY_ENABLED: bool = True

settings = Settings()