import re
from app.guardrails.base_guardrail import BaseGuardrail

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.-]+"
PHONE_PATTERN = r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
CREDIT_CARD_PATTERN = r"(?<!\d)\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)"
SSN_PATTERN = r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"
IP_ADDRESS_PATTERN = (
    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d)"
)
# Common prefixed secret formats (OpenAI/Stripe-style, AWS, GitHub, Slack).
# Deliberately prefix-anchored rather than a generic high-entropy match, to
# avoid flagging order numbers / confirmation codes as "API keys".
API_KEY_PATTERN = (
    r"\bsk-[A-Za-z0-9]{20,}\b"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}\b"
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}\b"
)
AADHAAR_PATTERN = r"(?<!\d)\d{4}[-\s]\d{4}[-\s]\d{4}(?!\d)"
PAN_PATTERN = r"\b[A-Z]{5}\d{4}[A-Z]\b"
# Best-effort only: passport number formats vary widely by issuing country,
# so this covers the common "1-2 letters + 6-8 digits" shape and will miss
# many real-world formats.
PASSPORT_PATTERN = r"\b[A-Z]{1,2}\d{6,8}\b"
# Labeled (not standalone) so bare digit sequences -- table numbers, order
# numbers, quantities -- aren't misflagged as PII.
PASSWORD_PATTERN = r"(?i)((?:password|passwd|pwd|passcode)\s*(?:is|:)\s*)(\S+)"
PIN_PATTERN = r"(?i)(\bpin\s*(?:is|:)\s*)(\d+)"
OTP_PATTERN = r"(?i)((?:otp|one[-\s]?time[-\s]?password)\s*(?:is|:)\s*)(\d+)"

# (pattern, replacement) pairs. Labeled patterns capture the label as group 1
# so redaction removes only the sensitive value, not the surrounding phrase.
PII_PATTERNS = [
    (EMAIL_PATTERN, "[REDACTED]"),
    (PHONE_PATTERN, "[REDACTED]"),
    (CREDIT_CARD_PATTERN, "[REDACTED]"),
    (SSN_PATTERN, "[REDACTED]"),
    (API_KEY_PATTERN, "[REDACTED]"),
    (AADHAAR_PATTERN, "[REDACTED]"),
    (PAN_PATTERN, "[REDACTED]"),
    (PASSPORT_PATTERN, "[REDACTED]"),
    (IP_ADDRESS_PATTERN, "[REDACTED]"),
    (PASSWORD_PATTERN, r"\1[REDACTED]"),
    (PIN_PATTERN, r"\1[REDACTED]"),
    (OTP_PATTERN, r"\1[REDACTED]"),
]


class PiiGuardrail(BaseGuardrail):
    async def check(self, text: str) -> bool:
        # Purely regex-based -- no LLM call needed here, but kept async to
        # match the shared BaseGuardrail interface used by the other
        # guardrails, so callers can `await` them uniformly.
        return not any(re.search(pattern, text) for pattern, _ in PII_PATTERNS)

    def redact(self, text: str) -> str:
        for pattern, replacement in PII_PATTERNS:
            text = re.sub(pattern, replacement, text)
        return text