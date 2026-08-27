import json

from app.services import conversation_logger


def test_log_event_writes_a_json_line_to_the_session_file(tmp_path, monkeypatch):
    monkeypatch.setattr(conversation_logger, "LOG_DIR", tmp_path)

    conversation_logger.log_event("room-1", "user_turn", text="I'd like a pizza")

    log_file = tmp_path / "room-1.jsonl"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["session_id"] == "room-1"
    assert entry["event"] == "user_turn"
    assert entry["text"] == "I'd like a pizza"
    assert "timestamp" in entry


def test_log_event_appends_multiple_events_to_the_same_session_file(tmp_path, monkeypatch):
    monkeypatch.setattr(conversation_logger, "LOG_DIR", tmp_path)

    conversation_logger.log_event("room-1", "user_turn", text="Hi")
    conversation_logger.log_event("room-1", "llm_call", model="openai/gpt-oss-120b", duration_ms=450.2)

    log_file = tmp_path / "room-1.jsonl"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["duration_ms"] == 450.2


def test_log_event_creates_a_separate_file_per_session(tmp_path, monkeypatch):
    monkeypatch.setattr(conversation_logger, "LOG_DIR", tmp_path)

    conversation_logger.log_event("room-1", "user_turn", text="Hi")
    conversation_logger.log_event("room-2", "user_turn", text="Hello")

    assert (tmp_path / "room-1.jsonl").exists()
    assert (tmp_path / "room-2.jsonl").exists()


def test_log_event_creates_log_dir_if_missing(tmp_path, monkeypatch):
    target_dir = tmp_path / "nested" / "conversations"
    monkeypatch.setattr(conversation_logger, "LOG_DIR", target_dir)

    conversation_logger.log_event("room-1", "user_turn", text="Hi")

    assert (target_dir / "room-1.jsonl").exists()
