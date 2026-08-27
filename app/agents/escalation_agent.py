"""
Real handoff logic for when a caller needs a human.

Scope note: this project is web-based voice ordering over a LiveKit room,
not a SIP phone line, so there is no SIP trunk to "transfer" a call across.
The honest, real action available through LiveKit's own API is to end the
AI agent's participation in the room -- gracefully, after it finishes
speaking the closing message -- so a human staff member can join the same
room and pick up the conversation. If this project ever grows a telephony
layer, this is the file where a real SIP participant transfer would live.
"""

from livekit.agents import AgentSession


def handoff_to_human(session: AgentSession) -> None:
    session.shutdown(drain=True)
