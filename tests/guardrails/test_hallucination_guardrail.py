import pytest

from app.guardrails.hallucination_guardrail import HallucinationGuardrail


@pytest.mark.asyncio
async def test_hallucination_guardrail_allows_plain_conversation_with_no_tool_results():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("Sure, what would you like today?", tool_results=[]) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_allows_price_matching_a_tool_result():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Margherita Pizza", "price": 12.99}]
    assert await guardrail.check("The Margherita Pizza is $12.99.", tool_results=tool_results) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_price_with_no_tool_results():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("The Margherita Pizza is $12.99.", tool_results=[]) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_price_not_matching_any_tool_result():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Margherita Pizza", "price": 12.99}]
    assert await guardrail.check("The Margherita Pizza is $99.99.", tool_results=tool_results) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_availability_claim_with_no_tool_call():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("We have vegan options available.", tool_results=[]) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_allows_availability_claim_backed_by_a_tool_result():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Caesar Salad", "price": 8.50}]
    assert await guardrail.check("Yes, we have Caesar Salad!", tool_results=tool_results) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_refund_policy_claim_regardless_of_tool_results():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Margherita Pizza", "price": 12.99}]
    assert await guardrail.check("Our refund policy allows returns within 30 days.", tool_results=tool_results) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_hours_claim_with_no_tool_results():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("We're open until 10 PM tonight.", tool_results=[]) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_verifies_prices_nested_in_list_tool_results():
    guardrail = HallucinationGuardrail()
    tool_results = [[{"name": "Caesar Salad", "price": 8.50}, {"name": "Grilled Salmon", "price": 18.00}]]
    assert await guardrail.check("Your total comes to $18.00.", tool_results=tool_results) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_verifies_totals_under_the_total_key():
    guardrail = HallucinationGuardrail()
    tool_results = [{"items": [{"name": "Caesar Salad", "price": 8.50}], "total": 8.50}]
    assert await guardrail.check("Your total comes to $8.50.", tool_results=tool_results) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_verifies_dollar_sign_price_unchanged():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Margherita Pizza", "price": 12.99}]
    assert await guardrail.check("That'll be $12.99.", tool_results=tool_results) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_verifies_bare_decimal_price():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Margherita Pizza", "price": 12.99}]
    assert await guardrail.check("The total is 12.99.", tool_results=tool_results) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_bare_decimal_price_with_no_tool_results():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("The total is 12.99.", tool_results=[]) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_verifies_dollars_word_form_price():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Margherita Pizza", "price": 99.00}]
    assert await guardrail.check("That's 99 dollars.", tool_results=tool_results) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_dollars_word_form_price_with_no_tool_results():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("That's 99 dollars.", tool_results=[]) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_allows_truthful_denial_with_no_tool_results():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("We don't have pizza, sorry.", tool_results=[]) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_allows_order_edit_language_mentioning_cancel():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Caesar Salad", "price": 8.50}]
    assert (
        await guardrail.check("No problem, I have cancelled the Caesar Salad.", tool_results=tool_results)
        is True
    )


@pytest.mark.asyncio
async def test_hallucination_guardrail_still_blocks_cancellation_policy_question():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("What's your cancellation policy?", tool_results=[]) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_allows_recommendation_with_no_tool_results():
    # Live-call regression: "What would you recommend?" got hard-blocked
    # even though the menu was already established earlier in the same
    # conversation -- a recommendation is an opinion, not a factual claim
    # that needs fresh tool-grounding every single turn.
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("I'd recommend the Margherita Pizza!", tool_results=[]) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_allows_how_about_suggestion_with_no_tool_results():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("How about the Grilled Salmon?", tool_results=[]) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_allows_try_the_suggestion_with_no_tool_results():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("Try the Caesar Salad, it's a favorite!", tool_results=[]) is True
