# Hallucination Guardrail: Tool Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dead, broken `HallucinationGuardrail` (exact-string menu match, sync `check()` violating the async `BaseGuardrail` contract, never wired in) with a real tool-grounded guardrail: the LLM must call real menu/order tools to back any price or availability claim, and a post-generation check verifies those claims against what the tools actually returned.

**Architecture:** `menu_handler_node` and `order_handler_node` bind their respective tools to the Groq call via a shared tool-execution-loop helper (`app/graphs/nodes/tool_loop.py`), which executes any tool calls the model makes (handling failures gracefully) and returns the final reply plus every tool result produced that turn. Each node then runs the redesigned `HallucinationGuardrail.check(text, tool_results)` on the reply before returning it; a failed check substitutes a fixed fallback message. As a side effect, `add_item_to_cart`'s result now actually flows into `ConversationState["cart"]`, fixing the previously-dead cart tracking.

**Tech Stack:** Python 3.14, pytest + pytest-asyncio, `groq` (`AsyncGroq`), `langchain-core` (`@tool`, `convert_to_openai_tool`), LangGraph `ConversationState` (`TypedDict`).

## Global Constraints

- Follow TDD: write the failing test, watch it fail, write minimal code, watch it pass, then commit. This project has been built this way throughout (see recent PII guardrail work).
- `HallucinationGuardrail.check(text, tool_results)` takes `tool_results` as a **required** parameter (no default) — see design doc rationale (`docs/superpowers/specs/2026-07-28-hallucination-guardrail-tool-grounding-design.md`).
- Commit after each task with a descriptive message, matching this repo's existing commit style (see `git log`).
- Every new/modified test file must pass in isolation (`pytest <file> -q`) and the full suite must be re-verified with no new failures at the end (Task 7).

---

### Task 1: Fix `RESTAURANT_B_MENU` price type bug (prerequisite)

**Discovered while planning, not in the original design doc:** `RESTAURANT_B_MENU` in `app/prompts/restaurant_b_prompt.py` stores prices as strings (`"13.50"`), while `RESTAURANT_A_MENU` stores them as floats (`12.99`). `calculate_total()` in `app/services/order_service.py` does `total += item["price"]` starting from `total = 0.0` — this raises `TypeError: unsupported operand type(s) for +=: 'float' and 'str'` the moment a Restaurant B cart is totaled. Today this is latent because `state["cart"]` is never actually populated (the bug this whole plan fixes). Once `add_item_to_cart` is wired into `order_handler_node` (Task 6), Restaurant B's order flow would crash on first use unless this is fixed first.

**Files:**
- Modify: `app/prompts/restaurant_b_prompt.py:15-19`
- Test: `tests/services/test_order_service.py`

**Interfaces:**
- Produces: `RESTAURANT_B_MENU` — same shape as before (`list[dict]` with `name`/`price` keys), `price` now `float` instead of `str`. No other task depends on this beyond it not crashing.

- [ ] **Step 1: Write the failing test**

Add to `tests/services/test_order_service.py`:
```python
from app.prompts.restaurant_b_prompt import RESTAURANT_B_MENU


def test_calculate_total_works_with_restaurant_b_menu_items():
    cart = [RESTAURANT_B_MENU[0], RESTAURANT_B_MENU[1]]
    assert calculate_total(cart) == pytest.approx(13.50 + 6.99)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_order_service.py::test_calculate_total_works_with_restaurant_b_menu_items -v`
Expected: FAIL with `TypeError: unsupported operand type(s) for +=: 'float' and 'str'`

- [ ] **Step 3: Fix the menu data**

In `app/prompts/restaurant_b_prompt.py`, replace:
```python
RESTAURANT_B_MENU = [
    {"name": "Kung Pao Chicken", "price": "13.50"},
    {"name": "Vegetable Spring Rolls", "price": "6.99"},
    {"name": "Szechuan Noodles", "price": "11.25"},
    {"name": "Sweet and Sour Pork", "price": "14.00"},
    {"name": "Mapo Tofu", "price": "10.75"},
]
```
with:
```python
RESTAURANT_B_MENU = [
    {"name": "Kung Pao Chicken", "price": 13.50},
    {"name": "Vegetable Spring Rolls", "price": 6.99},
    {"name": "Szechuan Noodles", "price": 11.25},
    {"name": "Sweet and Sour Pork", "price": 14.00},
    {"name": "Mapo Tofu", "price": 10.75},
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/services/test_order_service.py -v`
Expected: all PASS, including the new test

- [ ] **Step 5: Run the full prompts test suite to confirm no regression**

Run: `pytest tests/prompts -v`
Expected: all PASS (`test_restaurant_b_prompt.py` only checks item names appear in formatted text, unaffected by the type change)

- [ ] **Step 6: Commit**

```bash
git add app/prompts/restaurant_b_prompt.py tests/services/test_order_service.py
git commit -m "fix: store Restaurant B menu prices as floats, matching Restaurant A

calculate_total() crashes on string prices (0.0 += str). This was
latent because cart tracking was never wired up; fixing before Task 6
makes it live."
```

---

### Task 2: Fix stale menu-item fixtures in existing tool tests

**Context:** `tests/tools/test_menu_tool.py::test_get_item_details_returns_matching_item` and `tests/tools/test_order_tool.py::test_add_item_to_cart_adds_item` both hardcode `"Margherita Pizza"`, which is not on the current `RESTAURANT_A_MENU` (`Kadhai Paneer`, `Garlic Naan`, `Butter Chicken`). This is an existing, unrelated bug found during codebase review — fixing it now since it blocks a clean baseline before this plan's own test suite grows.

**Files:**
- Modify: `tests/tools/test_menu_tool.py:11-13`
- Modify: `tests/tools/test_order_tool.py:6-12`

**Interfaces:** None — isolated fixture fix, no other task depends on this.

- [ ] **Step 1: Confirm current failure**

Run: `pytest tests/tools/test_menu_tool.py::test_get_item_details_returns_matching_item tests/tools/test_order_tool.py::test_add_item_to_cart_adds_item -v`
Expected: both FAIL with `MenuItemNotFoundError: Menu item not found: Margherita Pizza`

- [ ] **Step 2: Fix the fixtures**

In `tests/tools/test_menu_tool.py`, replace:
```python
def test_get_item_details_returns_matching_item():
    result = get_item_details.invoke({"restaurant_id": settings.RESTAURANT_A_ID, "item_name": "Margherita Pizza"})
    assert result["name"] == "Margherita Pizza"
```
with:
```python
def test_get_item_details_returns_matching_item():
    result = get_item_details.invoke({"restaurant_id": settings.RESTAURANT_A_ID, "item_name": "Kadhai Paneer"})
    assert result["name"] == "Kadhai Paneer"
```

In `tests/tools/test_order_tool.py`, replace:
```python
def test_add_item_to_cart_adds_item():
    result = add_item_to_cart.invoke({
        "restaurant_id": settings.RESTAURANT_A_ID,
        "item_name": "Margherita Pizza",
        "cart": [],
    })
    assert result[0]["name"] == "Margherita Pizza"
```
with:
```python
def test_add_item_to_cart_adds_item():
    result = add_item_to_cart.invoke({
        "restaurant_id": settings.RESTAURANT_A_ID,
        "item_name": "Kadhai Paneer",
        "cart": [],
    })
    assert result[0]["name"] == "Kadhai Paneer"
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/tools -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/tools/test_menu_tool.py tests/tools/test_order_tool.py
git commit -m "fix: update stale Margherita Pizza test fixtures to current menu

RESTAURANT_A_MENU was changed to Indian cuisine; these two tests were
never updated and have been failing since."
```

---

### Task 3: Redesign `HallucinationGuardrail`

**Files:**
- Modify: `app/guardrails/hallucination_guardrail.py` (full rewrite)
- Test: `tests/guardrails/test_hallucination_guardrail.py` (full rewrite)

**Interfaces:**
- Produces: `HallucinationGuardrail` — zero-arg constructor (no longer takes `restaurant_id`; the old design called `get_menu()` internally, the new one only ever looks at `tool_results` passed in, so the restaurant lookup is dead). `async def check(self, text: str, tool_results: list[dict]) -> bool`. `tool_results` is a list of whatever the executed tools returned this turn (dicts and/or lists — `add_item_to_cart` returns a `list[dict]`, `get_item_details`/`get_order_status`/`confirm_order`/`search_menu` return `dict` or `list[dict]`).
- Produces: `UNVERIFIED_CLAIM_REPLY` — module-level string constant, the fallback message. Tasks 5 and 6 import this.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/guardrails/test_hallucination_guardrail.py`:
```python
import pytest

from app.guardrails.hallucination_guardrail import HallucinationGuardrail


@pytest.mark.asyncio
async def test_hallucination_guardrail_allows_plain_conversation_with_no_tool_results():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("Sure, what would you like today?", tool_results=[]) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_allows_price_matching_a_tool_result():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Kadhai Paneer", "price": 12.99}]
    assert await guardrail.check("The Kadhai Paneer is $12.99.", tool_results=tool_results) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_price_with_no_tool_results():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("The Kadhai Paneer is $12.99.", tool_results=[]) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_price_not_matching_any_tool_result():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Kadhai Paneer", "price": 12.99}]
    assert await guardrail.check("The Kadhai Paneer is $99.99.", tool_results=tool_results) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_availability_claim_with_no_tool_call():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("We have vegan options available.", tool_results=[]) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_allows_availability_claim_backed_by_a_tool_result():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Garlic Naan", "price": 8.50}]
    assert await guardrail.check("Yes, we have Garlic Naan!", tool_results=tool_results) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_refund_policy_claim_regardless_of_tool_results():
    guardrail = HallucinationGuardrail()
    tool_results = [{"name": "Kadhai Paneer", "price": 12.99}]
    assert await guardrail.check("Our refund policy allows returns within 30 days.", tool_results=tool_results) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_blocks_hours_claim_with_no_tool_results():
    guardrail = HallucinationGuardrail()
    assert await guardrail.check("We're open until 10 PM tonight.", tool_results=[]) is False


@pytest.mark.asyncio
async def test_hallucination_guardrail_verifies_prices_nested_in_list_tool_results():
    guardrail = HallucinationGuardrail()
    tool_results = [[{"name": "Garlic Naan", "price": 8.50}, {"name": "Butter Chicken", "price": 18.00}]]
    assert await guardrail.check("Your total comes to $18.00.", tool_results=tool_results) is True


@pytest.mark.asyncio
async def test_hallucination_guardrail_verifies_totals_under_the_total_key():
    guardrail = HallucinationGuardrail()
    tool_results = [{"items": [{"name": "Garlic Naan", "price": 8.50}], "total": 8.50}]
    assert await guardrail.check("Your total comes to $8.50.", tool_results=tool_results) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/guardrails/test_hallucination_guardrail.py -v`
Expected: FAIL — `TypeError: HallucinationGuardrail() takes no arguments` / `check() missing 1 required positional argument: 'tool_results'`, since the current class still takes `restaurant_id` and a text-only sync `check`.

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `app/guardrails/hallucination_guardrail.py`:
```python
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
    "cancel",
    "cancellation",
    "delivery fee",
    "delivery charge",
    "return policy",
    "open until",
    "close at",
    "opening hours",
    "closing time",
    "business hours",
]

CLAIM_PHRASES = [
    "we have",
    "we've got",
    "we offer",
    "we serve",
    "it's available",
    "is available",
    "we don't have",
    "not available",
    "no longer available",
    "in stock",
    "out of stock",
    "try the",
    "i'd recommend",
    "i recommend",
    "our specialty is",
    "how about the",
]

PRICE_PATTERN = re.compile(r"\$(\d+(?:\.\d{1,2})?)")


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

        stated_prices = {f"{float(p):.2f}" for p in PRICE_PATTERN.findall(text)}
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/guardrails/test_hallucination_guardrail.py -v`
Expected: all 10 PASS

- [ ] **Step 5: Commit**

```bash
git add app/guardrails/hallucination_guardrail.py tests/guardrails/test_hallucination_guardrail.py
git commit -m "feat: redesign HallucinationGuardrail around tool-result grounding

Replaces exact-string menu-item matching (which could never evaluate a
real LLM reply) with verification that price/availability claims trace
back to real tool results from this turn. Fixes the async/BaseGuardrail
interface violation. Policy claims (refunds, hours, etc.) are always
rejected -- no tool exists to ground them yet.

See docs/superpowers/specs/2026-07-28-hallucination-guardrail-tool-grounding-design.md"
```

---

### Task 4: Build the shared tool-execution loop

**Files:**
- Create: `app/graphs/nodes/tool_loop.py`
- Test: `tests/graphs/nodes/test_tool_loop.py`

**Interfaces:**
- Consumes: `settings.GROQ_LLM_MODEL` (`app/core/config.py`), `groq_client` (`app/integrations/groq_client.py`).
- Produces: `async def run_with_tools(system_prompt: str, messages: list[dict], tools: list) -> tuple[str, list]`. `tools` is a list of `langchain_core.tools`-decorated (`@tool`) callables. Returns `(reply_text, tool_results)` where `reply_text: str` is the final assistant reply and `tool_results: list` is every value successfully returned by an executed tool this turn (failed tool calls contribute nothing). Tasks 5 and 6 import and call this.
- Produces: `MAX_TOOL_ROUNDS = 2` module constant.

- [ ] **Step 1: Write the failing tests**

Create `tests/graphs/nodes/test_tool_loop.py`:
```python
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import tool

from app.graphs.nodes import tool_loop as tool_loop_module
from app.graphs.nodes.tool_loop import run_with_tools


@tool
def fake_search_menu(restaurant_id: str) -> list[dict]:
    """Fake search_menu tool for testing."""
    return [{"name": "Kadhai Paneer", "price": 12.99}]


@tool
def fake_failing_tool(item_name: str) -> dict:
    """Fake tool that always raises, for testing failure handling."""
    raise ValueError(f"not found: {item_name}")


def _fake_response(content, tool_calls=None):
    response = MagicMock()
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    return response


def _fake_tool_call(call_id, name, arguments: dict):
    call = MagicMock()
    call.id = call_id
    call.function.name = name
    call.function.arguments = json.dumps(arguments)
    return call


@pytest.mark.asyncio
async def test_run_with_tools_returns_content_directly_when_no_tool_calls(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        return_value=_fake_response("Hi there!", tool_calls=None)
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "hi"}],
        tools=[fake_search_menu],
    )

    assert reply_text == "Hi there!"
    assert tool_results == []
    fake_client.chat.completions.create.assert_awaited_once()
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert "tools" in call_kwargs
    assert call_kwargs["messages"][0] == {"role": "system", "content": "You are a waiter."}


@pytest.mark.asyncio
async def test_run_with_tools_executes_tool_call_and_returns_final_reply(monkeypatch):
    first_call = _fake_tool_call("call_1", "fake_search_menu", {"restaurant_id": "restaurant-a"})
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[first_call]),
            _fake_response("The Kadhai Paneer is $12.99!", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "what do you have?"}],
        tools=[fake_search_menu],
    )

    assert reply_text == "The Kadhai Paneer is $12.99!"
    assert tool_results == [[{"name": "Kadhai Paneer", "price": 12.99}]]
    assert fake_client.chat.completions.create.await_count == 2

    second_call_messages = fake_client.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert json.loads(tool_messages[0]["content"]) == [{"name": "Kadhai Paneer", "price": 12.99}]


@pytest.mark.asyncio
async def test_run_with_tools_handles_tool_execution_failure_without_crashing(monkeypatch):
    failing_call = _fake_tool_call("call_1", "fake_failing_tool", {"item_name": "Unicorn Steak"})
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[failing_call]),
            _fake_response("I couldn't find that on the menu.", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "what's in the unicorn steak?"}],
        tools=[fake_failing_tool],
    )

    assert reply_text == "I couldn't find that on the menu."
    assert tool_results == []

    second_call_messages = fake_client.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "error" in json.loads(tool_messages[0]["content"])


@pytest.mark.asyncio
async def test_run_with_tools_stops_after_max_rounds_and_forces_a_final_answer(monkeypatch):
    repeated_call = _fake_tool_call("call_1", "fake_search_menu", {"restaurant_id": "restaurant-a"})
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[repeated_call]),
            _fake_response(None, tool_calls=[repeated_call]),
            _fake_response("Here's what we have.", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "what do you have?"}],
        tools=[fake_search_menu],
    )

    assert reply_text == "Here's what we have."
    assert fake_client.chat.completions.create.await_count == 3
    final_call_kwargs = fake_client.chat.completions.create.await_args_list[2].kwargs
    assert "tools" not in final_call_kwargs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/graphs/nodes/test_tool_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.graphs.nodes.tool_loop'`

- [ ] **Step 3: Write the implementation**

Create `app/graphs/nodes/tool_loop.py`:
```python
import json

from langchain_core.utils.function_calling import convert_to_openai_tool

from app.core.config import settings
from app.integrations.groq_client import groq_client

MAX_TOOL_ROUNDS = 2


async def run_with_tools(
    system_prompt: str,
    messages: list[dict],
    tools: list,
) -> tuple[str, list]:
    """Call the LLM with the given tools bound, executing any tool calls it
    makes (up to MAX_TOOL_ROUNDS rounds) before returning the final reply.

    Returns (reply_text, tool_results). tool_results holds the return value
    of every successfully-executed tool call this turn -- a failed call
    contributes nothing, since there's no real data there to ground a claim
    in.
    """
    tools_by_name = {t.name: t for t in tools}
    openai_tools = [convert_to_openai_tool(t) for t in tools]
    conversation = [{"role": "system", "content": system_prompt}, *messages]
    tool_results: list = []

    for _round in range(MAX_TOOL_ROUNDS):
        response = await groq_client.chat.completions.create(
            model=settings.GROQ_LLM_MODEL,
            messages=conversation,
            tools=openai_tools,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content, tool_results

        conversation.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ],
        })

        for call in message.tool_calls:
            tool = tools_by_name[call.function.name]
            try:
                args = json.loads(call.function.arguments)
                result = tool.invoke(args)
                tool_results.append(result)
                content = json.dumps(result)
            except Exception as e:
                content = json.dumps({"error": str(e)})
            conversation.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": content,
            })

    # Ran out of rounds while the model kept requesting tools -- force a
    # final answer without offering more tools, rather than looping forever.
    response = await groq_client.chat.completions.create(
        model=settings.GROQ_LLM_MODEL,
        messages=conversation,
    )
    return response.choices[0].message.content, tool_results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/graphs/nodes/test_tool_loop.py -v`
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add app/graphs/nodes/tool_loop.py tests/graphs/nodes/test_tool_loop.py
git commit -m "feat: add shared tool-execution loop for menu/order graph nodes

Binds tools to the Groq call, executes any tool_calls the model makes
(capped at 2 rounds), and returns the final reply plus every tool
result produced this turn. Tool execution failures degrade to an error
message fed back to the model instead of crashing the node."
```

---

### Task 5: Wire tool grounding into `menu_handler_node`

**Files:**
- Modify: `app/graphs/nodes/menu_handler_node.py` (full rewrite)
- Modify: `tests/graphs/nodes/test_menu_handler_node.py` (full rewrite)

**Interfaces:**
- Consumes: `run_with_tools` (Task 4), `HallucinationGuardrail` + `UNVERIFIED_CLAIM_REPLY` (Task 3), `search_menu`/`get_item_details` (`app/tools/menu_tool.py`, unchanged).
- Produces: `menu_handler_node(state: ConversationState) -> dict` — same public signature as before, returning `{"messages": ..., "turn_count": ...}`.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/graphs/nodes/test_menu_handler_node.py`:
```python
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.graphs.conversation_state import ConversationState
from app.graphs.nodes import menu_handler_node as menu_handler_node_module
from app.graphs.nodes.menu_handler_node import menu_handler_node
from app.guardrails.hallucination_guardrail import UNVERIFIED_CLAIM_REPLY


@pytest.mark.asyncio
async def test_menu_handler_node_returns_grounded_reply_and_increments_turn(monkeypatch):
    fake_run_with_tools = AsyncMock(
        return_value=("The Kadhai Paneer is $12.99!", [{"name": "Kadhai Paneer", "price": 12.99}])
    )
    monkeypatch.setattr(menu_handler_node_module, "run_with_tools", fake_run_with_tools)

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "What's good here?"}],
        "turn_count": 1,
        "cart": [],
    }

    result = await menu_handler_node(state)

    assert result["turn_count"] == 2
    assert result["messages"][-1] == {"role": "assistant", "content": "The Kadhai Paneer is $12.99!"}
    fake_run_with_tools.assert_awaited_once()
    call_kwargs = fake_run_with_tools.call_args.kwargs
    assert call_kwargs["messages"] == state["messages"]
    tool_names = {t.name for t in call_kwargs["tools"]}
    assert tool_names == {"search_menu", "get_item_details"}


@pytest.mark.asyncio
async def test_menu_handler_node_replaces_ungrounded_reply_with_fallback(monkeypatch):
    fake_run_with_tools = AsyncMock(
        return_value=("The Kadhai Paneer is $99.99!", [])
    )
    monkeypatch.setattr(menu_handler_node_module, "run_with_tools", fake_run_with_tools)

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "How much is the paneer?"}],
        "turn_count": 1,
        "cart": [],
    }

    result = await menu_handler_node(state)

    assert result["messages"][-1] == {"role": "assistant", "content": UNVERIFIED_CLAIM_REPLY}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/graphs/nodes/test_menu_handler_node.py -v`
Expected: FAIL — `AttributeError: module 'app.graphs.nodes.menu_handler_node' has no attribute 'run_with_tools'` (current node doesn't import it yet).

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `app/graphs/nodes/menu_handler_node.py`:
```python
from app.core.config import settings
from app.core.exceptions import RestaurantNotFoundError
from app.graphs.conversation_state import ConversationState
from app.graphs.nodes.tool_loop import run_with_tools
from app.guardrails.hallucination_guardrail import HallucinationGuardrail, UNVERIFIED_CLAIM_REPLY
from app.prompts.restaurant_a_prompt import RESTAURANT_A_PERSONA
from app.prompts.restaurant_b_prompt import RESTAURANT_B_PERSONA
from app.tools.menu_tool import get_item_details, search_menu

MENU_TOOLS = [search_menu, get_item_details]
_hallucination_guardrail = HallucinationGuardrail()


async def menu_handler_node(state: ConversationState) -> dict:
    if state["restaurant_id"] == settings.RESTAURANT_A_ID:
        persona = RESTAURANT_A_PERSONA
    elif state["restaurant_id"] == settings.RESTAURANT_B_ID:
        persona = RESTAURANT_B_PERSONA
    else:
        raise RestaurantNotFoundError(state["restaurant_id"])

    system_prompt = (
        f"{persona}\n\nYou do not have the menu memorized. Always use the "
        f"search_menu or get_item_details tools to answer any question about "
        f"dishes, prices, ingredients, or availability -- never state a price "
        f"or claim an item exists from memory alone. The restaurant_id to use "
        f"for tool calls is '{state['restaurant_id']}'."
    )

    reply_text, tool_results = await run_with_tools(
        system_prompt=system_prompt,
        messages=state["messages"],
        tools=MENU_TOOLS,
    )

    if not await _hallucination_guardrail.check(reply_text, tool_results):
        reply_text = UNVERIFIED_CLAIM_REPLY

    new_message = {"role": "assistant", "content": reply_text}
    updated_messages = state["messages"] + [new_message]
    updated_turn_count = state["turn_count"] + 1

    return {
        "messages": updated_messages,
        "turn_count": updated_turn_count,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/graphs/nodes/test_menu_handler_node.py -v`
Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add app/graphs/nodes/menu_handler_node.py tests/graphs/nodes/test_menu_handler_node.py
git commit -m "feat: ground menu_handler_node replies in real tool calls

Removes the always-injected full-menu text from the system prompt (it
let the model answer without ever needing a tool) and binds
search_menu/get_item_details instead. Replies are checked against
actual tool results via HallucinationGuardrail before being returned."
```

---

### Task 6: Wire tool grounding into `order_handler_node`, propagate cart

**Files:**
- Modify: `app/graphs/nodes/order_handler_node.py` (full rewrite)
- Modify: `tests/graphs/nodes/test_order_handler_node.py` (full rewrite)

**Interfaces:**
- Consumes: `run_with_tools` (Task 4), `HallucinationGuardrail` + `UNVERIFIED_CLAIM_REPLY` (Task 3), `add_item_to_cart`/`confirm_order`/`get_order_status` (`app/tools/order_tool.py`), `get_item_details` (`app/tools/menu_tool.py`), `calculate_total` (`app/services/order_service.py`), all unchanged.
- Produces: `order_handler_node(state: ConversationState) -> dict` — now returns `{"messages": ..., "turn_count": ..., "cart": ...}` (adds `cart` to the returned dict, which it never did before).

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/graphs/nodes/test_order_handler_node.py`:
```python
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.graphs.conversation_state import ConversationState
from app.graphs.nodes import order_handler_node as order_handler_node_module
from app.graphs.nodes.order_handler_node import order_handler_node
from app.guardrails.hallucination_guardrail import UNVERIFIED_CLAIM_REPLY


@pytest.mark.asyncio
async def test_order_handler_node_returns_grounded_reply_and_updates_cart(monkeypatch):
    new_cart = [{"name": "Kadhai Paneer", "price": 12.99}]
    fake_run_with_tools = AsyncMock(
        return_value=("I've added the Kadhai Paneer, your total is $12.99!", [new_cart])
    )
    monkeypatch.setattr(order_handler_node_module, "run_with_tools", fake_run_with_tools)

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "I'll have the Kadhai Paneer"}],
        "turn_count": 1,
        "cart": [],
    }

    result = await order_handler_node(state)

    assert result["turn_count"] == 2
    assert result["messages"][-1] == {
        "role": "assistant",
        "content": "I've added the Kadhai Paneer, your total is $12.99!",
    }
    assert result["cart"] == new_cart
    call_kwargs = fake_run_with_tools.call_args.kwargs
    tool_names = {t.name for t in call_kwargs["tools"]}
    assert tool_names == {"add_item_to_cart", "confirm_order", "get_order_status", "get_item_details"}


@pytest.mark.asyncio
async def test_order_handler_node_keeps_existing_cart_when_no_cart_update(monkeypatch):
    fake_run_with_tools = AsyncMock(return_value=("Sure, anything else?", []))
    monkeypatch.setattr(order_handler_node_module, "run_with_tools", fake_run_with_tools)

    existing_cart = [{"name": "Garlic Naan", "price": 8.50}]
    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "thanks"}],
        "turn_count": 1,
        "cart": existing_cart,
    }

    result = await order_handler_node(state)

    assert result["cart"] == existing_cart


@pytest.mark.asyncio
async def test_order_handler_node_replaces_ungrounded_reply_with_fallback(monkeypatch):
    fake_run_with_tools = AsyncMock(return_value=("Your total is $999.99!", []))
    monkeypatch.setattr(order_handler_node_module, "run_with_tools", fake_run_with_tools)

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "what's my total?"}],
        "turn_count": 1,
        "cart": [],
    }

    result = await order_handler_node(state)

    assert result["messages"][-1] == {"role": "assistant", "content": UNVERIFIED_CLAIM_REPLY}
    assert result["cart"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/graphs/nodes/test_order_handler_node.py -v`
Expected: FAIL — `AttributeError: module 'app.graphs.nodes.order_handler_node' has no attribute 'run_with_tools'`, and `KeyError: 'cart'` once that's fixed (current node doesn't return `cart` at all).

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `app/graphs/nodes/order_handler_node.py`:
```python
from app.core.config import settings
from app.core.exceptions import RestaurantNotFoundError
from app.graphs.conversation_state import ConversationState
from app.graphs.nodes.tool_loop import run_with_tools
from app.guardrails.hallucination_guardrail import HallucinationGuardrail, UNVERIFIED_CLAIM_REPLY
from app.prompts.restaurant_a_prompt import RESTAURANT_A_PERSONA
from app.prompts.restaurant_b_prompt import RESTAURANT_B_PERSONA
from app.services.order_service import calculate_total
from app.tools.menu_tool import get_item_details
from app.tools.order_tool import add_item_to_cart, confirm_order, get_order_status

ORDER_TOOLS = [add_item_to_cart, confirm_order, get_order_status, get_item_details]
_hallucination_guardrail = HallucinationGuardrail()


async def order_handler_node(state: ConversationState) -> dict:
    if state["restaurant_id"] == settings.RESTAURANT_A_ID:
        persona = RESTAURANT_A_PERSONA
    elif state["restaurant_id"] == settings.RESTAURANT_B_ID:
        persona = RESTAURANT_B_PERSONA
    else:
        raise RestaurantNotFoundError(state["restaurant_id"])

    cart_total = calculate_total(state["cart"])
    system_prompt = (
        f"{persona}\n\nThe customer's current cart total is ${cart_total:.2f}. Use "
        f"the add_item_to_cart tool to add anything new, and get_order_status or "
        f"confirm_order to report totals -- never state a new price or total from "
        f"memory alone. The restaurant_id to use for tool calls is "
        f"'{state['restaurant_id']}', and the current cart is {state['cart']}."
    )

    reply_text, tool_results = await run_with_tools(
        system_prompt=system_prompt,
        messages=state["messages"],
        tools=ORDER_TOOLS,
    )

    if not await _hallucination_guardrail.check(reply_text, tool_results):
        reply_text = UNVERIFIED_CLAIM_REPLY

    updated_cart = state["cart"]
    for result in tool_results:
        if isinstance(result, list):
            updated_cart = result

    new_message = {"role": "assistant", "content": reply_text}
    updated_messages = state["messages"] + [new_message]
    updated_turn_count = state["turn_count"] + 1

    return {
        "messages": updated_messages,
        "turn_count": updated_turn_count,
        "cart": updated_cart,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/graphs/nodes/test_order_handler_node.py -v`
Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add app/graphs/nodes/order_handler_node.py tests/graphs/nodes/test_order_handler_node.py
git commit -m "feat: ground order_handler_node replies in real tool calls, fix cart tracking

Binds add_item_to_cart/confirm_order/get_order_status/get_item_details
and checks replies via HallucinationGuardrail. add_item_to_cart's
result now propagates into ConversationState['cart'] -- previously
state['cart'] was initialized to [] and never updated anywhere (see
PR_DESCRIPTION.md Known Limitation #1)."
```

---

### Task 7: Full regression pass

**Files:** None modified — verification only.

**Interfaces:** None.

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: all tests pass, no failures. If `tests/agents/test_session_agent.py` fails, investigate — it mocks `graph.ainvoke` directly (confirmed during design review) so it should be unaffected by node internals changing, but confirm this holds.

- [ ] **Step 2: If anything fails, fix and re-run**

Do not proceed until `pytest -q` reports zero failures.

- [ ] **Step 3: Update `PR_DESCRIPTION.md`'s Known Limitations section**

In `PR_DESCRIPTION.md`, the two items below are now resolved by this work. Update item 1 and item 2 under "Known limitations / flags for review":

Replace:
```
1. **Cart tracking and order totals are not actually deterministic — this is the
   biggest one.** `state["cart"]` is initialized to `[]` in `session_agent.py` and is
   never appended to anywhere in the codebase. The `add_item_to_cart`, `confirm_order`,
   `get_order_status`, `search_menu`, and `get_item_details` tools in `app/tools/` are
   fully implemented but **never bound to the LLM** — neither `menu_handler_node.py` nor
   `order_handler_node.py` passes a `tools=` argument to `groq_client.chat.completions.
   create`, and there's no `bind_tools` call anywhere in the repo. `order_handler_node`
   calls `calculate_total(state["cart"])`, but since the cart is always empty, this
   always evaluates to `0.0`. In practice, when the bot states a running total during a
   call (verified live — e.g. correctly stating "$20.49" for two items), that number is
   the LLM reasoning it out from free-text conversation history, not a real calculation.
   It happened to be right in testing, but it's not guaranteed to be — this is worth a
   real conversation about whether tool-calling needs to be wired in before this is
   trusted for anything resembling a real order.
2. **`HallucinationGuardrail` is implemented but not wired into the live pipeline.**
   It also only does exact string equality against menu item names
   (`text in real_names`), not scanning a sentence for a hallucinated dish mentioned
   mid-reply — so even wiring it in as-is wouldn't catch most real hallucinations.
   Deliberately scoped out rather than shipped half-working.
```
with:
```
1. **Resolved.** `search_menu`, `get_item_details`, `add_item_to_cart`, `confirm_order`,
   and `get_order_status` are now bound to the LLM via a shared tool-execution loop
   (`app/graphs/nodes/tool_loop.py`), used by both `menu_handler_node.py` and
   `order_handler_node.py`. `add_item_to_cart`'s result now propagates into
   `ConversationState["cart"]`, so cart contents and totals are real, not LLM-guessed.
   See `docs/superpowers/specs/2026-07-28-hallucination-guardrail-tool-grounding-design.md`.
2. **Resolved.** `HallucinationGuardrail` is redesigned around tool-result grounding
   (verifies price/availability claims against real tool output this turn, rather than
   exact-string menu-item matching) and is wired into both `menu_handler_node.py` and
   `order_handler_node.py`. A documented residual gap remains: a fabricated dish name
   mentioned without a price or availability phrase attached isn't caught — full
   coverage needs NER, evaluated and deliberately not built (see design doc).
```

- [ ] **Step 4: Commit**

```bash
git add PR_DESCRIPTION.md
git commit -m "docs: update PR_DESCRIPTION known limitations for tool-grounded guardrail work

Cart tracking and HallucinationGuardrail wiring were the two biggest
flagged limitations; both are resolved by this plan."
```

---

## Post-implementation, not part of this plan

Flagged during design review, deliberately deferred:
- Tool-argument correctness checking (verifying `get_item_details`/`add_item_to_cart`'s `item_name` argument actually matches what the user asked about) — real gap, real false-positive risk, needs the user's message threaded into the guardrail call. See design doc "Considered and rejected."
- A real policy data source (hours, refund windows, delivery fees) so those questions can be answered instead of always refused.
- Splitting `UNVERIFIED_CLAIM_REPLY` into rule-specific messages (policy vs. grounding failure) if the single unified message turns out to read oddly in practice.
- Live smoke test confirming `GROQ_LLM_MODEL` (`openai/gpt-oss-120b`) actually supports tool-calling as expected — flagged as an unverified assumption throughout this plan.
