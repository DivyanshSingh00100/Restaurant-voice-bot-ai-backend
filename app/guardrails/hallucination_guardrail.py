import re
from typing import Any

from app.guardrails.base_guardrail import BaseGuardrail

UNVERIFIED_CLAIM_REPLY = (
    "I'm not able to confirm that for you right now -- I can note it down "
    "and have someone follow up."
)

# Provisional -- remove an entry once a real tool exists for that domain
# (e.g. a get_restaurant_hours() tool would mean "hours" claims can be
# grounded like menu/order claims instead of always rejected).
POLICY_KEYWORDS = [
    "refund",
    "cancel my order",
    "cancel the order",
    "cancellation policy",
    "order cancellation",
    "delivery fee",
    "delivery charge",
    "return policy",
    "open until",
    "close at",
    "opening hours",
    "closing time",
    "business hours",
]

# Live-call lesson: "What would you recommend?" got hard-blocked because
# "i'd recommend"/"how about"/"try the" were treated as factual claims
# needing fresh tool-grounding every turn. A recommendation is an opinion,
# not a fact -- same reasoning already applied to denial phrases below.
# "our specialty is" stays: it asserts a specific dish literally IS the
# restaurant's signature item, closer to a factual claim than a suggestion.
CLAIM_PHRASES = [
    "we have",
    "we've got",
    "we offer",
    "we serve",
    "it's available",
    "is available",
    "in stock",
    "our specialty is",
]

DOLLAR_SIGN_PATTERN = re.compile(r"\$(\d+(?:\.\d{1,2})?)")
BARE_DECIMAL_PATTERN = re.compile(r"(?<!\d)(\d+\.\d{2})(?!\d)")
DOLLARS_WORD_PATTERN = re.compile(r"(\d+(?:\.\d{1,2})?)\s+dollars?\b", re.IGNORECASE)


def _extract_stated_prices(text: str) -> set[str]:
    prices: set[str] = set()
    for pattern in (DOLLAR_SIGN_PATTERN, BARE_DECIMAL_PATTERN, DOLLARS_WORD_PATTERN):
        for match in pattern.findall(text):
            prices.add(f"{float(match):.2f}")
    return prices


def _collect_prices(value: Any, prices: set[str]) -> None:
    if isinstance(value, dict):
        for key, val in value.items():
            if key in ("price", "total") and isinstance(val, (int, float, str)):
                try:
                    prices.add(f"{float(val):.2f}")
                except (TypeError, ValueError):
                    pass
            else:
                _collect_prices(val, prices)
    elif isinstance(value, list):
        for item in value:
            _collect_prices(item, prices)


def _extract_verified_prices(tool_results: list[dict]) -> set[str]:
    prices: set[str] = set()
    for result in tool_results:
        _collect_prices(result, prices)
    return prices


class HallucinationGuardrail(BaseGuardrail):
    async def check(self, text: str, tool_results: list[dict]) -> bool:
        # Purely rule-based -- no LLM call needed, but kept async to match
        # the shared BaseGuardrail interface used by the other guardrails.
        lowered = text.lower()

        if any(keyword in lowered for keyword in POLICY_KEYWORDS):
            return False

        stated_prices = _extract_stated_prices(text)
        makes_claim = bool(stated_prices) or any(phrase in lowered for phrase in CLAIM_PHRASES)

        if not makes_claim:
            return True

        if not tool_results:
            return False

        if stated_prices:
            verified_prices = _extract_verified_prices(tool_results)
            if not stated_prices.issubset(verified_prices):
                return False

        return True
