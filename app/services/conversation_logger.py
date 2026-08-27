import json
import time
from pathlib import Path

# One JSONL file per call session, separate from the console's noisy
# groq._base_client DEBUG dump -- a clean, greppable per-turn timeline
# (user text, LLM call durations, STT/TTS metrics, assistant text) for
# latency work, without wading through raw HTTP request/response logs.
LOG_DIR = Path("logs/conversations")


def log_event(session_id: str, event_type: str, **fields) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": time.time(), "session_id": session_id, "event": event_type, **fields}
    with open(LOG_DIR / f"{session_id}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
