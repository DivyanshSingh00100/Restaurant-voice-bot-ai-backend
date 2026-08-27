import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.graphs.conversation_state import ConversationState
from app.graphs.nodes import order_handler_node as order_handler_node_module
from app.graphs.nodes import tool_loop as tool_loop_module
from app.graphs.nodes.order_handler_node import order_handler_node
from app.guardrails.hallucination_guardrail import UNVERIFIED_CLAIM_REPLY


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
async def test_order_handler_node_returns_grounded_reply_and_updates_cart(monkeypatch):
    async def fake_run_with_tools(**kwargs):
        live_cart = kwargs["arg_overrides"]["cart"]
        live_cart.append({"name": "Margherita Pizza", "price": 12.99})
        return ("I've added the Margherita Pizza, your total is $12.99!", [live_cart])

    fake_run_with_tools_mock = AsyncMock(side_effect=fake_run_with_tools)
    monkeypatch.setattr(order_handler_node_module, "run_with_tools", fake_run_with_tools_mock)

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "I'll have the Margherita Pizza"}],
        "turn_count": 1,
        "cart": [],
    }

    result = await order_handler_node(state)

    assert result["turn_count"] == 2
    assert result["messages"][-1] == {
        "role": "assistant",
        "content": "I've added the Margherita Pizza, your total is $12.99!",
    }
    assert result["cart"] == [{"name": "Margherita Pizza", "price": 12.99}]
    call_kwargs = fake_run_with_tools_mock.call_args.kwargs
    tool_names = {t.name for t in call_kwargs["tools"]}
    assert tool_names == {
        "add_item_to_cart",
        "remove_item_from_cart",
        "confirm_order",
        "get_order_status",
        "get_item_details",
    }


@pytest.mark.asyncio
async def test_order_handler_node_passes_live_server_side_cart_as_arg_override(monkeypatch):
    fake_run_with_tools = AsyncMock(return_value=("Sure, anything else?", []))
    monkeypatch.setattr(order_handler_node_module, "run_with_tools", fake_run_with_tools)

    existing_cart = [{"name": "Caesar Salad", "price": 8.50}]
    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "thanks"}],
        "turn_count": 1,
        "cart": existing_cart,
    }

    await order_handler_node(state)

    call_kwargs = fake_run_with_tools.call_args.kwargs
    assert call_kwargs["arg_overrides"] == {"cart": existing_cart}
    # The override cart must be a distinct list object from state["cart"],
    # not the same reference -- state input must never be mutated directly.
    assert call_kwargs["arg_overrides"]["cart"] is not existing_cart


@pytest.mark.asyncio
async def test_order_handler_node_keeps_existing_cart_when_no_cart_update(monkeypatch):
    fake_run_with_tools = AsyncMock(return_value=("Sure, anything else?", []))
    monkeypatch.setattr(order_handler_node_module, "run_with_tools", fake_run_with_tools)

    existing_cart = [{"name": "Caesar Salad", "price": 8.50}]
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


@pytest.mark.asyncio
async def test_order_handler_node_accumulates_cart_when_two_items_added_in_one_turn(monkeypatch):
    # run_with_tools is mocked here (its real accumulation mechanism --
    # writing each call's result back into the arg_overrides dict -- is
    # tested directly against real tools in test_tool_loop.py). This mock
    # simulates the end state: by the time run_with_tools returns, the
    # "cart" key of arg_overrides holds the fully-accumulated cart.
    async def fake_run_with_tools(**kwargs):
        live_cart = kwargs["arg_overrides"]["cart"]
        live_cart.append({"name": "Margherita Pizza", "price": 12.99})
        live_cart.append({"name": "Caesar Salad", "price": 8.50})
        return ("I've added both, anything else?", [live_cart, live_cart])

    monkeypatch.setattr(
        order_handler_node_module, "run_with_tools", AsyncMock(side_effect=fake_run_with_tools)
    )

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "I'll have the Margherita Pizza and a Caesar Salad"}],
        "turn_count": 1,
        "cart": [],
    }

    result = await order_handler_node(state)

    assert result["cart"] == [
        {"name": "Margherita Pizza", "price": 12.99},
        {"name": "Caesar Salad", "price": 8.50},
    ]


@pytest.mark.asyncio
async def test_order_handler_node_end_to_end_with_real_tools_across_sequential_rounds(monkeypatch):
    # Full integration test: real order_handler_node, real ORDER_TOOLS
    # (real add_item_to_cart -> real add_to_cart service), only the Groq
    # client is faked. The model calls add_item_to_cart twice across two
    # separate sequential rounds, each time supplying a fabricated cart
    # argument (wrong items, wrong prices) -- proving the server-side
    # override wins every time, and that the final cart is the true
    # accumulated union with no duplication and no fabricated data.
    round_1_call = _fake_tool_call(
        "call_1",
        "add_item_to_cart",
        {
            "restaurant_id": settings.RESTAURANT_A_ID,
            "item_name": "Margherita Pizza",
            "cart": [{"name": "Free Lobster", "price": 0.01}],
        },
    )
    round_2_call = _fake_tool_call(
        "call_2",
        "add_item_to_cart",
        {
            "restaurant_id": settings.RESTAURANT_A_ID,
            "item_name": "Caesar Salad",
            "cart": [{"name": "Free Yacht", "price": 0.01}],
        },
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[round_1_call]),
            _fake_response(None, tool_calls=[round_2_call]),
            _fake_response(
                "I've added the Margherita Pizza and the Caesar Salad!", tool_calls=None
            ),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "I'll have the Margherita Pizza, then a Caesar Salad"}],
        "turn_count": 1,
        "cart": [],
    }

    result = await order_handler_node(state)

    assert result["cart"] == [
        {"name": "Margherita Pizza", "price": 12.99},
        {"name": "Caesar Salad", "price": 8.50},
    ]
    assert result["messages"][-1] == {
        "role": "assistant",
        "content": "I've added the Margherita Pizza and the Caesar Salad!",
    }


@pytest.mark.asyncio
async def test_order_handler_node_end_to_end_with_real_tools_batched_same_round(monkeypatch):
    # Same as above but both add_item_to_cart calls arrive in a SINGLE
    # round (one assistant message, two tool_calls) -- the other shape the
    # model can produce. Same guarantee: real tools, fabricated cart
    # arguments ignored, correct accumulated union, no duplication.
    call_a = _fake_tool_call(
        "call_1",
        "add_item_to_cart",
        {
            "restaurant_id": settings.RESTAURANT_A_ID,
            "item_name": "Margherita Pizza",
            "cart": [{"name": "Free Lobster", "price": 0.01}],
        },
    )
    call_b = _fake_tool_call(
        "call_2",
        "add_item_to_cart",
        {
            "restaurant_id": settings.RESTAURANT_A_ID,
            "item_name": "Caesar Salad",
            "cart": [{"name": "Free Yacht", "price": 0.01}],
        },
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_response(None, tool_calls=[call_a, call_b]),
            _fake_response("I've added both items!", tool_calls=None),
        ]
    )
    monkeypatch.setattr(tool_loop_module, "groq_client", fake_client)

    state: ConversationState = {
        "restaurant_id": settings.RESTAURANT_A_ID,
        "messages": [{"role": "user", "content": "I'll have the Margherita Pizza and a Caesar Salad"}],
        "turn_count": 1,
        "cart": [],
    }

    result = await order_handler_node(state)

    assert result["cart"] == [
        {"name": "Margherita Pizza", "price": 12.99},
        {"name": "Caesar Salad", "price": 8.50},
    ]
