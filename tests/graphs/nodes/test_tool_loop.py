import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import tool

from app.graphs.nodes import tool_loop as tool_loop_module
from app.graphs.nodes.tool_loop import run_with_tools


@tool
def fake_search_menu(restaurant_id: str) -> list[dict]:
    """Fake search_menu tool for testing."""
    return [{"name": "Margherita Pizza", "price": 12.99}]


@tool
def fake_failing_tool(item_name: str) -> dict:
    """Fake tool that always raises, for testing failure handling."""
    raise ValueError(f"not found: {item_name}")


@tool
def fake_cart_tool(cart: list) -> list:
    """Fake tool that just echoes back whatever cart it received."""
    return cart


@tool
def fake_unserializable_tool(item_name: str) -> set:
    """Fake tool that returns a value json.dumps can't serialize."""
    return {item_name}


@tool
def fake_add_to_cart_tool(item_name: str, cart: list) -> list:
    """Fake add-to-cart tool mirroring the real add_to_cart service: mutates
    the given cart list in place and returns the same object."""
    cart.append({"name": item_name, "price": 1.0})
    return cart


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
            _fake_response("The Margherita Pizza is $12.99!", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "what do you have?"}],
        tools=[fake_search_menu],
    )

    assert reply_text == "The Margherita Pizza is $12.99!"
    assert tool_results == [[{"name": "Margherita Pizza", "price": 12.99}]]
    assert fake_client.chat.completions.create.await_count == 2

    second_call_messages = fake_client.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert json.loads(tool_messages[0]["content"]) == [{"name": "Margherita Pizza", "price": 12.99}]


@pytest.mark.asyncio
async def test_run_with_tools_drops_non_final_content_emitted_alongside_tool_calls(monkeypatch):
    # Live-call regression: gpt-oss-120b via Groq sometimes returns a
    # populated `content` string ALONGSIDE tool_calls in the same response
    # (e.g. "Got it! Anything else?" plus a tool_calls list). That content is
    # never spoken -- run_with_tools only returns message.content from a
    # round with no tool_calls -- but it was still being written verbatim
    # into the assistant history entry recording the tool call. Left there,
    # it resends unspoken "thinking" text back to the model on every
    # subsequent round, wasting tokens and risking the model treating it as
    # something it already said (which produced doubled/concatenated-looking
    # replies across two separate live calls). It must not appear in the
    # history sent to the next round.
    first_call = _fake_tool_call("call_1", "fake_search_menu", {"restaurant_id": "restaurant-a"})
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response("Got it! Anything else you'd like to add?", tool_calls=[first_call]),
            _fake_response("Would you like anything else?", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, _ = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "what do you have?"}],
        tools=[fake_search_menu],
    )

    assert reply_text == "Would you like anything else?"

    second_call_messages = fake_client.chat.completions.create.await_args_list[1].kwargs["messages"]
    assistant_messages = [m for m in second_call_messages if m["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["content"] is None


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
async def test_run_with_tools_returns_empty_string_when_content_is_none_and_no_tool_calls(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        return_value=_fake_response(None, tool_calls=None)
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "hi"}],
        tools=[fake_search_menu],
    )

    assert reply_text == ""
    assert tool_results == []


@pytest.mark.asyncio
async def test_run_with_tools_handles_unknown_tool_name_without_crashing(monkeypatch):
    unknown_call = _fake_tool_call("call_1", "not_a_real_tool", {"item_name": "Unicorn Steak"})
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[unknown_call]),
            _fake_response("I couldn't do that.", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "do something weird"}],
        tools=[fake_search_menu],
    )

    assert reply_text == "I couldn't do that."
    assert tool_results == []

    second_call_messages = fake_client.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "error" in json.loads(tool_messages[0]["content"])


@pytest.mark.asyncio
async def test_run_with_tools_merges_arg_overrides_into_tool_call_arguments(monkeypatch):
    # The model supplies a fabricated cart; arg_overrides should replace it
    # with the real server-side cart before the tool is invoked.
    fabricated_cart = [{"name": "Free Lobster", "price": 0.01}]
    real_cart = [{"name": "Margherita Pizza", "price": 12.99}]
    cart_call = _fake_tool_call("call_1", "fake_cart_tool", {"cart": fabricated_cart})
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[cart_call]),
            _fake_response("Here's your cart.", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "show my cart"}],
        tools=[fake_cart_tool],
        arg_overrides={"cart": real_cart},
    )

    assert reply_text == "Here's your cart."
    assert tool_results == [real_cart]


@pytest.mark.asyncio
async def test_run_with_tools_arg_overrides_accumulate_across_batched_same_round_calls(monkeypatch):
    # Two add-to-cart calls issued in the SAME round (one assistant message
    # with two tool_calls). fake_add_to_cart_tool (like the real
    # add_to_cart service) mutates its *own* cart argument in place, but
    # LangChain's tool.ainvoke() validates arguments through the tool's
    # pydantic schema first, which copies list arguments -- so that
    # mutation never reaches the original object passed in via
    # arg_overrides. run_with_tools compensates by writing each call's
    # list-shaped result back into the arg_overrides dict itself, so the
    # second call in this round starts from the first call's real result,
    # not the stale original -- and the model's fabricated per-call cart
    # argument is discarded either way.
    overrides = {"cart": []}
    call_a = _fake_tool_call(
        "call_1", "fake_add_to_cart_tool", {"item_name": "Margherita Pizza", "cart": ["fabricated"]}
    )
    call_b = _fake_tool_call(
        "call_2", "fake_add_to_cart_tool", {"item_name": "Caesar Salad", "cart": ["also fabricated"]}
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[call_a, call_b]),
            _fake_response("Added both!", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "I'll have the Margherita Pizza and a Caesar Salad"}],
        tools=[fake_add_to_cart_tool],
        arg_overrides=overrides,
    )

    assert reply_text == "Added both!"
    assert overrides["cart"] == [
        {"name": "Margherita Pizza", "price": 1.0},
        {"name": "Caesar Salad", "price": 1.0},
    ]
    # The second call's recorded result reflects the fully-accumulated
    # cart -- not a fabricated or partial one.
    assert tool_results[-1] == overrides["cart"]
    assert len(tool_results) == 2


@pytest.mark.asyncio
async def test_run_with_tools_arg_overrides_accumulate_across_sequential_rounds(monkeypatch):
    # Two add-to-cart calls issued across two SEPARATE sequential rounds
    # (MAX_TOOL_ROUNDS = 2). The arg_overrides dict must be threaded
    # through both rounds so the second round's call starts from the
    # first round's real result, with no duplication and no fabricated
    # cart surviving.
    overrides = {"cart": []}
    round_1_call = _fake_tool_call(
        "call_1", "fake_add_to_cart_tool", {"item_name": "Margherita Pizza", "cart": ["fabricated"]}
    )
    round_2_call = _fake_tool_call(
        "call_2", "fake_add_to_cart_tool", {"item_name": "Caesar Salad", "cart": ["also fabricated"]}
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[round_1_call]),
            _fake_response(None, tool_calls=[round_2_call]),
            _fake_response("Added both across two rounds!", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "I'll have the Margherita Pizza, then a Caesar Salad"}],
        tools=[fake_add_to_cart_tool],
        arg_overrides=overrides,
    )

    assert reply_text == "Added both across two rounds!"
    assert fake_client.chat.completions.create.await_count == 3
    assert overrides["cart"] == [
        {"name": "Margherita Pizza", "price": 1.0},
        {"name": "Caesar Salad", "price": 1.0},
    ]
    assert tool_results[-1] == overrides["cart"]
    assert len(tool_results) == 2


@pytest.mark.asyncio
async def test_run_with_tools_arg_overrides_ignores_keys_tool_does_not_accept(monkeypatch):
    # fake_search_menu only takes restaurant_id -- an override for an
    # unrelated key ("cart") must not be injected since the tool never
    # declared that parameter.
    search_call = _fake_tool_call("call_1", "fake_search_menu", {"restaurant_id": "restaurant-a"})
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[search_call]),
            _fake_response("Here's the menu.", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "what do you have?"}],
        tools=[fake_search_menu],
        arg_overrides={"cart": [{"name": "should not appear", "price": 1.0}]},
    )

    assert reply_text == "Here's the menu."
    assert tool_results == [[{"name": "Margherita Pizza", "price": 12.99}]]


@pytest.mark.asyncio
async def test_run_with_tools_does_not_record_result_when_serialization_fails(monkeypatch):
    bad_call = _fake_tool_call("call_1", "fake_unserializable_tool", {"item_name": "Naan"})
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[bad_call]),
            _fake_response("Sorry, something went wrong.", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "what's in the naan?"}],
        tools=[fake_unserializable_tool],
    )

    assert reply_text == "Sorry, something went wrong."
    assert tool_results == []

    second_call_messages = fake_client.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "error" in json.loads(tool_messages[0]["content"])


@pytest.mark.asyncio
async def test_run_with_tools_stops_after_max_rounds_and_forces_a_final_answer(monkeypatch):
    # MAX_TOOL_ROUNDS is 3: a status-check plus two separate adds was observed
    # live and needs to fit within the cap without triggering the forced call.
    repeated_call = _fake_tool_call("call_1", "fake_search_menu", {"restaurant_id": "restaurant-a"})
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[repeated_call]),
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
    assert fake_client.chat.completions.create.await_count == 4
    final_call_kwargs = fake_client.chat.completions.create.await_args_list[3].kwargs
    assert "tools" not in final_call_kwargs


@pytest.mark.asyncio
async def test_run_with_tools_caps_history_sent_to_the_llm_to_the_most_recent_messages(monkeypatch):
    # Live-call regression: a long call's full history was resent to Groq
    # every turn with no cap, so the payload kept growing until it blew
    # through Groq's per-minute TOKEN limit (57 LLM calls in one session,
    # observed 429s despite the request-count budget being nowhere near
    # hit). Only the most recent MAX_HISTORY_MESSAGES messages should be
    # sent -- older ones are dropped from the LLM payload (but still kept
    # in Redis/state elsewhere; this function doesn't touch that).
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        return_value=_fake_response("Sure thing!", tool_calls=None)
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    long_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
        for i in range(tool_loop_module.MAX_HISTORY_MESSAGES + 10)
    ]

    await run_with_tools(
        system_prompt="You are a waiter.",
        messages=long_history,
        tools=[fake_search_menu],
    )

    sent_messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages[0] == {"role": "system", "content": "You are a waiter."}
    sent_history = sent_messages[1:]
    assert len(sent_history) == tool_loop_module.MAX_HISTORY_MESSAGES
    assert sent_history == long_history[-tool_loop_module.MAX_HISTORY_MESSAGES:]


@pytest.mark.asyncio
async def test_run_with_tools_returns_fallback_when_forced_final_call_fails(monkeypatch):
    # Live-call regression: after exhausting MAX_TOOL_ROUNDS, the forced
    # final call (sent without tools) can 400 if the model still tries to
    # emit a tool call anyway (Groq: "Tool choice is none, but model called
    # a tool"). That exception was unhandled and crashed the whole voice
    # turn -- it must degrade to a safe fallback reply instead.
    repeated_call = _fake_tool_call("call_1", "fake_search_menu", {"restaurant_id": "restaurant-a"})
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[repeated_call]),
            _fake_response(None, tool_calls=[repeated_call]),
            _fake_response(None, tool_calls=[repeated_call]),
            RuntimeError("400 tool_use_failed: Tool choice is none, but model called a tool"),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    reply_text, tool_results = await run_with_tools(
        system_prompt="You are a waiter.",
        messages=[{"role": "user", "content": "add pizza and salmon"}],
        tools=[fake_search_menu],
    )

    assert reply_text == tool_loop_module.TOOL_LOOP_FALLBACK_REPLY
    assert tool_results == [
        [{"name": "Margherita Pizza", "price": 12.99}],
        [{"name": "Margherita Pizza", "price": 12.99}],
        [{"name": "Margherita Pizza", "price": 12.99}],
    ]
