import pytest

from app.services import conversation_logger


@pytest.fixture(autouse=True)
def _isolate_conversation_logs(tmp_path, monkeypatch):
    # conversation_logger.log_event writes real JSONL files to LOG_DIR.
    # Without this, any test that exercises llm_node/groq_client/voice
    # pipeline code paths without explicitly mocking log_event silently
    # appends fake entries into the real logs/conversations/ directory,
    # polluting it alongside genuine call data. Redirect every test to a
    # throwaway per-test directory by default; tests that want to assert
    # on log_event calls directly still monkeypatch it themselves.
    monkeypatch.setattr(conversation_logger, "LOG_DIR", tmp_path)
