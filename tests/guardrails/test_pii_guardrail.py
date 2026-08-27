import pytest

from app.guardrails.pii_guardrail import PiiGuardrail


@pytest.mark.asyncio
async def test_pii_guardrail_allows_text_without_email():
    guardrail = PiiGuardrail()
    assert await guardrail.check("I'd like to order a pizza") is True


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_email():
    guardrail = PiiGuardrail()
    assert await guardrail.check("Contact me at test@example.com") is False


def test_pii_guardrail_redacts_email():
    guardrail = PiiGuardrail()
    result = guardrail.redact("Contact me at test@example.com please")
    assert result == "Contact me at [REDACTED] please"
    assert "test@example.com" not in result


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_phone_number():
    guardrail = PiiGuardrail()
    assert await guardrail.check("Call me back at 555-123-4567") is False


def test_pii_guardrail_redacts_phone_number():
    guardrail = PiiGuardrail()
    result = guardrail.redact("Call me back at 555-123-4567 please")
    assert result == "Call me back at [REDACTED] please"
    assert "555-123-4567" not in result


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_credit_card_number():
    guardrail = PiiGuardrail()
    assert await guardrail.check("My card number is 4111 1111 1111 1111") is False


def test_pii_guardrail_redacts_credit_card_number():
    guardrail = PiiGuardrail()
    result = guardrail.redact("My card number is 4111 1111 1111 1111 thanks")
    assert result == "My card number is [REDACTED] thanks"
    assert "4111 1111 1111 1111" not in result


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_ssn():
    guardrail = PiiGuardrail()
    assert await guardrail.check("My SSN is 123-45-6789") is False


def test_pii_guardrail_redacts_ssn():
    guardrail = PiiGuardrail()
    result = guardrail.redact("My SSN is 123-45-6789 ok")
    assert result == "My SSN is [REDACTED] ok"
    assert "123-45-6789" not in result


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_ip_address():
    guardrail = PiiGuardrail()
    assert await guardrail.check("My IP is 192.168.1.1") is False


def test_pii_guardrail_redacts_ip_address():
    guardrail = PiiGuardrail()
    result = guardrail.redact("My IP is 192.168.1.1 ok")
    assert result == "My IP is [REDACTED] ok"
    assert "192.168.1.1" not in result


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_api_key():
    guardrail = PiiGuardrail()
    assert await guardrail.check("Here is my key sk-abcdefghijklmnopqrstuvwx") is False


def test_pii_guardrail_redacts_api_key():
    guardrail = PiiGuardrail()
    result = guardrail.redact("Here is my key sk-abcdefghijklmnopqrstuvwx ok")
    assert result == "Here is my key [REDACTED] ok"
    assert "sk-abcdefghijklmnopqrstuvwx" not in result


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_aws_access_key():
    guardrail = PiiGuardrail()
    assert await guardrail.check("Key: AKIAIOSFODNN7EXAMPLE") is False


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_aadhaar():
    guardrail = PiiGuardrail()
    assert await guardrail.check("My Aadhaar is 1234 5678 9123") is False


def test_pii_guardrail_redacts_aadhaar():
    guardrail = PiiGuardrail()
    result = guardrail.redact("My Aadhaar is 1234 5678 9123 ok")
    assert result == "My Aadhaar is [REDACTED] ok"
    assert "1234 5678 9123" not in result


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_pan():
    guardrail = PiiGuardrail()
    assert await guardrail.check("My PAN is ABCDE1234F") is False


def test_pii_guardrail_redacts_pan():
    guardrail = PiiGuardrail()
    result = guardrail.redact("My PAN is ABCDE1234F ok")
    assert result == "My PAN is [REDACTED] ok"
    assert "ABCDE1234F" not in result


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_passport_number():
    guardrail = PiiGuardrail()
    assert await guardrail.check("My passport number is K1234567") is False


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_labeled_password():
    guardrail = PiiGuardrail()
    assert await guardrail.check("My password is Sw0rdfish!") is False


def test_pii_guardrail_redacts_labeled_password():
    guardrail = PiiGuardrail()
    result = guardrail.redact("My password is Sw0rdfish! ok")
    assert result == "My password is [REDACTED] ok"
    assert "Sw0rdfish!" not in result


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_labeled_pin():
    guardrail = PiiGuardrail()
    assert await guardrail.check("My PIN is 4821") is False


@pytest.mark.asyncio
async def test_pii_guardrail_blocks_text_with_labeled_otp():
    guardrail = PiiGuardrail()
    assert await guardrail.check("My OTP is 482991") is False


def test_pii_guardrail_redacts_labeled_otp():
    guardrail = PiiGuardrail()
    result = guardrail.redact("My OTP is 482991 ok")
    assert result == "My OTP is [REDACTED] ok"
    assert "482991" not in result


@pytest.mark.asyncio
async def test_pii_guardrail_allows_restaurant_order_numbers():
    guardrail = PiiGuardrail()
    assert await guardrail.check("I'd like table 4 for 6 people, order number 482991") is True


@pytest.mark.asyncio
async def test_pii_guardrail_allows_delivery_address():
    guardrail = PiiGuardrail()
    assert await guardrail.check("Please deliver to 221B Baker Street, London") is True
