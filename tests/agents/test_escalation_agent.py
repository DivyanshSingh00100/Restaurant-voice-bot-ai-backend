from unittest.mock import MagicMock

from app.agents.escalation_agent import handoff_to_human


def test_handoff_to_human_shuts_down_session_with_drain():
    fake_session = MagicMock()

    handoff_to_human(fake_session)

    fake_session.shutdown.assert_called_once_with(drain=True)
