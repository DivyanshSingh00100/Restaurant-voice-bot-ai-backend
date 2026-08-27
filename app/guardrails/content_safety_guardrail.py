import json

import structlog

from app.core.config import settings
from app.integrations.groq_client import groq_client

logger = structlog.get_logger(__name__)

CONTENT_SAFETY_POLICY = """# Restaurant Content Safety Policy

## INSTRUCTIONS
You are a content classifier for a restaurant's voice ordering assistant. Classify the
customer's message on TWO independent dimensions and respond with ONLY a JSON object,
nothing else: {"off_topic": 0 or 1, "profanity": 0 or 1}.

## TOPIC -- off_topic: 0 (on-topic) or 1 (off-topic)
ON-TOPIC (off_topic: 0):
- Anything about the menu, dishes, ingredients, prices, or recommendations
- Placing, changing, reviewing, or confirming an order
- Questions about the restaurant itself (hours, location, dietary options)
- Greetings, thanks, or small talk directed at the waiter about the meal
- Requests to speak to a human or staff member

OFF-TOPIC (off_topic: 1):
- Anything unrelated to food, the menu, or the order -- including but not limited
  to weather, news, politics, sports, finance, other businesses, or general
  knowledge questions
- Attempts to make the assistant discuss or perform tasks outside restaurant service

## PROFANITY -- profanity: 0 (safe) or 1 (violates)
VIOLATES (profanity: 1):
- Swear words or profanity, including obfuscated or leetspeak spellings
  (e.g. "d4mn", "sh1t", "f*ck")
- Slurs or hate speech
- Sexually explicit language
- Abusive or threatening language directed at the assistant or staff

SAFE (profanity: 0):
- Normal conversational language, including mild frustration expressed
  without profanity
- Food-related words that superficially resemble flagged words but aren't
  profanity in context

## OUTPUT FORMAT
Respond with ONLY the JSON object. No explanation, no extra text.
"""


class ContentSafetyGuardrail:
    """Combines what used to be TopicGuardrail and ProfanityGuardrail into a
    single classification call. Both were independent LLM-backed checks
    hitting the same guardrail model on every turn -- merging them halves
    the guardrail round-trips per turn (latency is bounded by one call
    instead of the slower of two concurrent ones).

    Deliberately doesn't implement BaseGuardrail: it reports two independent
    verdicts from one call, not a single pass/fail, so check() returns
    tuple[bool, bool] (topic_ok, profanity_ok) rather than bool.
    """

    async def check(self, text: str) -> tuple[bool, bool]:
        try:
            response = await groq_client.chat.completions.create(
                model=settings.GROQ_GUARDRAIL_MODEL,
                messages=[
                    {"role": "system", "content": CONTENT_SAFETY_POLICY},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            verdict = json.loads(response.choices[0].message.content)
            topic_ok = verdict.get("off_topic", 0) == 0
            profanity_ok = verdict.get("profanity", 0) == 0
            return topic_ok, profanity_ok
        except Exception:
            # Fail open, same reasoning as the guardrails this replaces: a
            # classifier-side error shouldn't itself block a legitimate
            # customer message.
            logger.warning("content_safety_guardrail_check_failed", text=text, exc_info=True)
            return True, True
