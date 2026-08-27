import json

import structlog
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.core.config import settings
from app.integrations.groq_client import groq_client

logger = structlog.get_logger(__name__)

# 3, not 2: a status-check plus two separate add_item_to_cart calls was
# observed on a real call and needs to fit without hitting the forced-answer
# fallback below.
MAX_TOOL_ROUNDS = 3

# Caps how much conversation history gets resent to Groq each call. Without
# this, a long call's full history is resent every single turn, so the
# payload keeps growing until it blows through Groq's per-minute TOKEN
# limit -- observed live: a 57-call session hit repeated 429s even though
# the request-count budget was nowhere near exhausted. Only the most recent
# messages are sent to the LLM; the full history is untouched in Redis/state,
# so this only trims what gets billed against the token budget per call.
MAX_HISTORY_MESSAGES = 20

# Deliberately claim-free (no prices, no availability, no policy words) so it
# always passes the hallucination guardrail's checks.
TOOL_LOOP_FALLBACK_REPLY = (
    "Sorry, I lost my train of thought there -- could you say that again?"
)


async def run_with_tools(
    system_prompt: str,
    messages: list[dict],
    tools: list,
    arg_overrides: dict | None = None,
) -> tuple[str, list]:
    """Call the LLM with the given tools bound, executing any tool calls it
    makes (up to MAX_TOOL_ROUNDS rounds) before returning the final reply.

    arg_overrides, if given, is merged into every tool call's parsed
    arguments before invocation -- but only for keys already present in
    those arguments, so tools that don't take a given argument are left
    untouched. This lets callers inject server-side truth (e.g. the real
    cart) in place of whatever the model supplied, since model-supplied
    arguments can't be trusted as ground truth.

    Note on accumulation across multiple calls in the same turn: LangChain's
    tool.invoke()/.ainvoke() validates arguments through the tool's pydantic
    schema, which copies list/dict arguments before calling the underlying
    function -- so a tool that mutates its argument in place (e.g.
    add_to_cart appending to `cart`) only mutates that per-call copy, never
    the original object the caller passed in via arg_overrides. To still
    accumulate correctly across multiple tool calls in one turn (whether
    batched in one round or spread across sequential rounds), this function
    writes each call's list-shaped result back into arg_overrides itself
    (for whichever override key that call's args used), so the *next* call
    starts from the real, up-to-date state instead of the stale original.
    Callers that need the final value back should keep their own reference
    to the arg_overrides dict and read the key back out after this returns.

    Returns (reply_text, tool_results). tool_results holds the return value
    of every successfully-executed tool call this turn -- a failed call
    contributes nothing, since there's no real data there to ground a claim
    in.
    """
    tools_by_name = {t.name: t for t in tools}
    openai_tools = [convert_to_openai_tool(t) for t in tools]
    conversation = [
        {"role": "system", "content": system_prompt},
        *messages[-MAX_HISTORY_MESSAGES:],
    ]
    tool_results: list = []

    for _round in range(MAX_TOOL_ROUNDS):
        response = await groq_client.chat.completions.create(
            model=settings.GROQ_LLM_MODEL,
            messages=conversation,
            tools=openai_tools,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or "", tool_results

        # message.content is sometimes populated alongside tool_calls (seen
        # with openai/gpt-oss-120b via Groq) even though it's never spoken --
        # this function only returns content from a round with no tool_calls.
        # Recording that text as if it were a real assistant turn resent it
        # to the model on every later round, wasting tokens and risking the
        # model treating unspoken "thinking" text as something it already
        # said (produced doubled/concatenated-looking replies live). Drop it;
        # only the tool-call record itself is a real turn.
        conversation.append({
            "role": "assistant",
            "content": None,
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
            try:
                tool = tools_by_name[call.function.name]
                args = json.loads(call.function.arguments)
                override_keys: list = []
                if arg_overrides:
                    override_keys = [k for k in arg_overrides if k in args]
                    args.update({k: arg_overrides[k] for k in override_keys})
                result = await tool.ainvoke(args)
                content = json.dumps(result)
                tool_results.append(result)
                if isinstance(result, list):
                    for key in override_keys:
                        arg_overrides[key] = result
            except Exception as e:
                content = json.dumps({"error": str(e)})
            conversation.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": content,
            })

    # Ran out of rounds while the model kept requesting tools -- force a
    # final answer without offering more tools, rather than looping forever.
    # This call can itself fail: Groq 400s with "tool_use_failed" when the
    # model tries to emit yet another tool call despite tools being absent
    # (observed live, mid-order). A crashed voice turn is the worst outcome,
    # so degrade to a safe fallback reply instead of raising.
    try:
        response = await groq_client.chat.completions.create(
            model=settings.GROQ_LLM_MODEL,
            messages=conversation,
        )
        return response.choices[0].message.content or "", tool_results
    except Exception:
        logger.warning("tool_loop_forced_final_call_failed", exc_info=True)
        return TOOL_LOOP_FALLBACK_REPLY, tool_results
