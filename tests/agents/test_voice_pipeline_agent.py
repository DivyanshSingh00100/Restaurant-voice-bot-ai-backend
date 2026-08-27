from unittest.mock import MagicMock

from livekit.agents.metrics import STTMetrics, TTSMetrics

from app.agents import voice_pipeline_agent as voice_pipeline_agent_module
from app.agents.voice_pipeline_agent import build_agent_session, build_llm
from app.core.config import settings
from app.prompts.restaurant_a_prompt import RESTAURANT_A_MENU
from app.prompts.restaurant_b_prompt import RESTAURANT_B_MENU


def test_build_agent_session_configures_stt_and_tts_with_groq_settings(monkeypatch):
    fake_vad = MagicMock()
    fake_stt_cls = MagicMock()
    fake_tts_cls = MagicMock()

    monkeypatch.setattr(voice_pipeline_agent_module.silero.VAD, "load", MagicMock(return_value=fake_vad))
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "STT", fake_stt_cls)
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "TTS", fake_tts_cls)
    monkeypatch.setattr(voice_pipeline_agent_module.settings, "TTS_PROVIDER", "groq")

    fake_session_cls = MagicMock()
    monkeypatch.setattr(voice_pipeline_agent_module, "AgentSession", fake_session_cls)

    build_agent_session(settings.RESTAURANT_A_ID, "room-1")

    expected_prompt = "Menu items: " + ", ".join(item["name"] for item in RESTAURANT_A_MENU) + "."
    fake_stt_cls.assert_called_once_with(
        model=settings.GROQ_STT_MODEL,
        language="en",
        api_key=settings.GROQ_API_KEY,
        prompt=expected_prompt,
    )
    fake_tts_cls.assert_called_once_with(
        model=settings.GROQ_TTS_MODEL,
        voice=settings.GROQ_TTS_VOICE,
        api_key=settings.GROQ_API_KEY,
    )
    fake_session_cls.assert_called_once_with(
        vad=fake_vad,
        stt=fake_stt_cls.return_value,
        tts=fake_tts_cls.return_value,
        preemptive_generation=False,
    )


def test_build_agent_session_uses_restaurant_b_menu_for_its_own_stt_prompt(monkeypatch):
    fake_stt_cls = MagicMock()

    monkeypatch.setattr(voice_pipeline_agent_module.silero.VAD, "load", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "STT", fake_stt_cls)
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "TTS", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module, "AgentSession", MagicMock())

    build_agent_session(settings.RESTAURANT_B_ID, "room-1")

    expected_prompt = "Menu items: " + ", ".join(item["name"] for item in RESTAURANT_B_MENU) + "."
    assert fake_stt_cls.call_args.kwargs["prompt"] == expected_prompt


def test_build_agent_session_uses_gemini_tts_when_configured(monkeypatch):
    fake_gemini_tts_cls = MagicMock()

    monkeypatch.setattr(voice_pipeline_agent_module.silero.VAD, "load", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "STT", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "TTS", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.gemini_tts, "TTS", fake_gemini_tts_cls)
    monkeypatch.setattr(voice_pipeline_agent_module.settings, "TTS_PROVIDER", "gemini")

    fake_session_cls = MagicMock()
    monkeypatch.setattr(voice_pipeline_agent_module, "AgentSession", fake_session_cls)

    build_agent_session(settings.RESTAURANT_A_ID, "room-1")

    fake_gemini_tts_cls.assert_called_once_with(
        model=settings.GEMINI_TTS_MODEL,
        voice_name=settings.GEMINI_TTS_VOICE,
        api_key=settings.GOOGLE_API_KEY,
    )
    assert fake_session_cls.call_args.kwargs["tts"] == fake_gemini_tts_cls.return_value


def test_build_agent_session_uses_kokoro_tts_when_configured(monkeypatch):
    fake_kokoro_tts_cls = MagicMock()

    monkeypatch.setattr(voice_pipeline_agent_module.silero.VAD, "load", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "STT", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "TTS", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.gemini_tts, "TTS", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.openai, "TTS", fake_kokoro_tts_cls)
    monkeypatch.setattr(voice_pipeline_agent_module.settings, "TTS_PROVIDER", "kokoro")

    fake_session_cls = MagicMock()
    monkeypatch.setattr(voice_pipeline_agent_module, "AgentSession", fake_session_cls)

    build_agent_session(settings.RESTAURANT_A_ID, "room-1")

    fake_kokoro_tts_cls.assert_called_once_with(
        model="tts-1",
        voice=settings.KOKORO_TTS_VOICE,
        api_key="not-needed",
        base_url=settings.KOKORO_API_BASE_URL,
        response_format="wav",
    )
    assert fake_session_cls.call_args.kwargs["tts"] == fake_kokoro_tts_cls.return_value


def test_build_agent_session_tts_provider_switch_is_case_insensitive(monkeypatch):
    fake_kokoro_tts_cls = MagicMock()

    monkeypatch.setattr(voice_pipeline_agent_module.silero.VAD, "load", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "STT", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "TTS", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.gemini_tts, "TTS", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.openai, "TTS", fake_kokoro_tts_cls)
    monkeypatch.setattr(voice_pipeline_agent_module.settings, "TTS_PROVIDER", "Kokoro")

    fake_session_cls = MagicMock()
    monkeypatch.setattr(voice_pipeline_agent_module, "AgentSession", fake_session_cls)

    build_agent_session(settings.RESTAURANT_A_ID, "room-1")

    fake_kokoro_tts_cls.assert_called_once()
    assert fake_session_cls.call_args.kwargs["tts"] == fake_kokoro_tts_cls.return_value


def test_build_agent_session_uses_groq_tts_when_provider_is_groq(monkeypatch):
    fake_groq_tts_cls = MagicMock()
    fake_gemini_tts_cls = MagicMock()
    fake_kokoro_tts_cls = MagicMock()

    monkeypatch.setattr(voice_pipeline_agent_module.silero.VAD, "load", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "STT", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "TTS", fake_groq_tts_cls)
    monkeypatch.setattr(voice_pipeline_agent_module.gemini_tts, "TTS", fake_gemini_tts_cls)
    monkeypatch.setattr(voice_pipeline_agent_module.openai, "TTS", fake_kokoro_tts_cls)
    monkeypatch.setattr(voice_pipeline_agent_module.settings, "TTS_PROVIDER", "groq")

    fake_session_cls = MagicMock()
    monkeypatch.setattr(voice_pipeline_agent_module, "AgentSession", fake_session_cls)

    build_agent_session(settings.RESTAURANT_A_ID, "room-1")

    fake_groq_tts_cls.assert_called_once()
    fake_gemini_tts_cls.assert_not_called()
    fake_kokoro_tts_cls.assert_not_called()
    assert fake_session_cls.call_args.kwargs["tts"] == fake_groq_tts_cls.return_value


def test_build_agent_session_logs_stt_metrics_via_the_metrics_collected_event(monkeypatch):
    monkeypatch.setattr(voice_pipeline_agent_module.silero.VAD, "load", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "STT", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "TTS", MagicMock())

    fake_session = MagicMock()
    monkeypatch.setattr(voice_pipeline_agent_module, "AgentSession", MagicMock(return_value=fake_session))

    logged_events = []
    monkeypatch.setattr(
        voice_pipeline_agent_module.conversation_logger,
        "log_event",
        lambda session_id, event_type, **fields: logged_events.append((session_id, event_type, fields)),
    )

    build_agent_session(settings.RESTAURANT_A_ID, "room-1")

    # Capture the handler registered via session.on("metrics_collected", handler)
    # and invoke it directly with a real STTMetrics instance.
    registered_handler = fake_session.on.call_args.args[1]
    stt_metrics = STTMetrics(
        type="stt_metrics",
        label="groq",
        request_id="req-1",
        timestamp=0.0,
        duration=1.2,
        audio_duration=3.0,
        streamed=True,
    )
    fake_event = MagicMock(metrics=stt_metrics)
    registered_handler(fake_event)

    assert len(logged_events) == 1
    session_id, event_type, fields = logged_events[0]
    assert session_id == "room-1"
    assert event_type == "stt_metrics"
    assert fields["duration_s"] == 1.2
    assert fields["audio_duration_s"] == 3.0


def test_build_agent_session_logs_tts_metrics_via_the_metrics_collected_event(monkeypatch):
    monkeypatch.setattr(voice_pipeline_agent_module.silero.VAD, "load", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "STT", MagicMock())
    monkeypatch.setattr(voice_pipeline_agent_module.groq, "TTS", MagicMock())

    fake_session = MagicMock()
    monkeypatch.setattr(voice_pipeline_agent_module, "AgentSession", MagicMock(return_value=fake_session))

    logged_events = []
    monkeypatch.setattr(
        voice_pipeline_agent_module.conversation_logger,
        "log_event",
        lambda session_id, event_type, **fields: logged_events.append((session_id, event_type, fields)),
    )

    build_agent_session(settings.RESTAURANT_A_ID, "room-1")

    registered_handler = fake_session.on.call_args.args[1]
    tts_metrics = TTSMetrics(
        type="tts_metrics",
        label="groq",
        request_id="req-2",
        timestamp=0.0,
        ttfb=0.3,
        duration=1.5,
        audio_duration=2.0,
        cancelled=False,
        characters_count=42,
        streamed=True,
    )
    fake_event = MagicMock(metrics=tts_metrics)
    registered_handler(fake_event)

    assert len(logged_events) == 1
    session_id, event_type, fields = logged_events[0]
    assert session_id == "room-1"
    assert event_type == "tts_metrics"
    assert fields["duration_s"] == 1.5
    assert fields["ttfb_s"] == 0.3
    assert fields["characters"] == 42


def test_build_llm_uses_groq_model_via_livekit_inference(monkeypatch):
    fake_llm_cls = MagicMock()
    monkeypatch.setattr(voice_pipeline_agent_module, "inference", MagicMock(LLM=fake_llm_cls))

    build_llm()

    fake_llm_cls.assert_called_once_with(
        f"groq/{settings.GROQ_LLM_MODEL}",
        api_key=settings.LIVEKIT_API_KEY,
        api_secret=settings.LIVEKIT_API_SECRET,
    )
