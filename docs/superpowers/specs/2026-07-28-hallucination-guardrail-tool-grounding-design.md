# Hallucination guardrail: tool grounding

**Date:** 2026-07-28
**Status:** Approved design, not yet implemented

## Problem

`HallucinationGuardrail` (`app/guardrails/hallucination_guardrail.py`) exists but is not
wired into the live pipeline and has real bugs even in isolation:

1. `check()` is a synchronous `def`, but `BaseGuardrail.check` is declared `async def`.
   Every other guardrail (Topic, Profanity, Pii) is `async` and `await`ed uniformly in
   `session_agent.py`; calling this one the same way would raise `TypeError: object bool
   can't be used in 'await' expression'`.
2. The check is `text in real_names` — exact, whole-string equality against a menu item
   name. It cannot evaluate a real LLM reply ("We recommend the Kadhai Paneer!") because
   the full sentence is never equal to an item name. This is already flagged as a known
   limitation in `PR_DESCRIPTION.md`.
3. It is never imported or instantiated outside its own test file.
4. Separately (unrelated bug, not part of this design): its one meaningful test,
   `test_hallucination_guardrail_allows_real_menu_item`, currently fails because it
   asserts against `"Margherita Pizza"`, which is not on the current
   (`restaurant_a_prompt.py`) menu — the menu was changed to Indian cuisine (`Kadhai
   Paneer`, `Garlic Naan`, `Butter Chicken`) without updating the test. The same stale
   fixture breaks `test_menu_tool.py::test_get_item_details_returns_matching_item` and
   `test_order_tool.py::test_add_item_to_cart_adds_item`.

More fundamentally: even a fixed version of the exact-match check isn't a real defense.
A useful hallucination guardrail for this bot needs to verify that specific claims
(prices, item availability) actually came from real data, not from the LLM's own
free-text generation or memory of the conversation.

## Scope decisions (confirmed with user before design)

- **Tool grounding, not generic confidence-scoring.** The bot should verify claims
  against real data (menu, order state) rather than have the LLM self-report a
  confidence score, which is itself unverifiable.
- **Tool binding is in scope.** Per `PR_DESCRIPTION.md` Known Limitation #1, no tools are
  currently bound to the LLM at all (`menu_handler_node` / `order_handler_node` call the
  LLM with free-text history only). Real grounding requires actually wiring
  `search_menu`, `get_item_details`, `add_item_to_cart`, `confirm_order`,
  `get_order_status` into the LLM calls — this also fixes cart/order-total
  non-determinism as a side effect.
- **Hard block on failure**, not regeneration. Same pattern as the existing
  Topic/Profanity guardrails: discard the ungrounded reply, substitute a fixed short
  fallback message. No retry loop, no doubled latency.
- **Policy questions (refunds, cancellations, delivery fees, hours) are always
  ungrounded for now.** No policy data source exists anywhere in this codebase. Rather
  than build one (out of scope), any reply touching these topics is unconditionally
  replaced with an uncertainty/escalation-flavored message.
- **Explicitly out of scope:** physical addresses, standalone OTP/PIN-style digit
  detection, name/NER-based detection, bank account/medical record numbers — these were
  scoped out of the *PII* guardrail earlier in this project for the same reasons
  (false-positive risk to core ordering/delivery flow, or infeasible without NER) and
  the same reasoning applies here; not revisited in this design.

## Architecture

Two independent defenses, both scoped to `menu_handler_node.py` and
`order_handler_node.py` — the only two places the LLM currently generates free text.

1. **Grounding by construction.** Bind the existing `@tool`-decorated functions to the
   Groq call via `tools=`. Remove the current shortcuts that hand the LLM facts for
   free: the always-injected full-menu text in `menu_handler_node`'s system prompt, and
   reliance on remembered conversation history for prices. If the model needs a fact, it
   has to fetch it.
2. **Post-generation verification.** A redesigned `HallucinationGuardrail` that, after
   the reply is generated, checks whether it makes a price/availability claim and, if
   so, whether that claim traces back to a real tool result from this turn.

Policy questions get a third, independent, simpler rule inside the same guardrail:
unconditional rejection by keyword match, since there is nothing to verify them against.

## Tool-calling loop (per node)

Both `menu_handler_node.py` and `order_handler_node.py` get the same shape of change:

- **Remove free facts from the system prompt.** Drop the injected `format_menu(menu)`
  block from `menu_handler_node`; replace with an instruction that the model does not
  have the menu memorized and must call `search_menu` / `get_item_details` for any
  dish/price/availability question. `order_handler_node` keeps its pre-computed
  `cart_total` (that number is derived from already-verified state, not a guess), but
  gains an instruction to use tools rather than recompute from memory for anything new.
- **Bind tools** via `langchain_core.utils.function_calling.convert_to_openai_tool`,
  reusing the existing `@tool` functions in `app/tools/` unchanged, passed as `tools=` to
  `groq_client.chat.completions.create`.
- **Execution loop**, capped at 2 rounds to prevent runaway loops:
  1. Call the LLM with `tools=`.
  2. If `response.choices[0].message.tool_calls` is present: execute each. On success,
     append a `tool`-role message with the JSON result and record the result in a
     `tool_results` list. On failure (e.g. `MenuItemNotFoundError`, or any other
     exception), append a `tool`-role message containing an error payload instead of
     raising, and do **not** add anything to `tool_results` — a failed lookup grounds
     nothing. Call the LLM again for the final reply.
  3. If no `tool_calls`: that response is the final reply.
- **Cart propagation.** `add_item_to_cart`'s return value is captured and included in
  the node's returned state dict (`{"cart": new_cart, "messages": ..., "turn_count":
  ...}`), so it actually flows through `ConversationState` for the first time. This is
  what fixes Known Limitation #1.

**Risk, not blocking:** this assumes `GROQ_LLM_MODEL` (`openai/gpt-oss-120b`) reliably
supports tool-calling per Groq's OpenAI-compatible API. Not verified live yet — same
category of risk as the already-documented Orpheus TTS issues. Worth a smoke test once
built.

## `HallucinationGuardrail` redesign

```python
async def check(self, text: str, tool_results: list[dict]) -> bool
```

- Fixes the async bug.
- `tool_results` has **no default value** — every call site must supply real data
  explicitly. This is a deliberate, narrow deviation from `BaseGuardrail.check(text)`.
  Considered and rejected: a generic `GuardrailContext` object uniformly adopted by all
  four guardrails. Rejected because (a) `HallucinationGuardrail` is never called
  alongside Topic/Profanity/Pii in a shared loop — it runs from a structurally different
  location (inside the graph nodes, after tool execution, not in `session_agent.py`'s
  pre-check gather) — so the "generic loop forgets extra context" failure mode this
  would guard against does not exist in this codebase's actual call pattern; and (b) it
  would require rewriting three already-stable, already-tested guardrails and every
  call site in `session_agent.py` for a benefit with no current or planned use.
  Requiring `tool_results` with no default closes the actual silent-misuse risk at a
  fraction of the cost.

**Rule 1 — policy keywords (always fail, independent of tool_results):** a fixed keyword
list (`refund`, `cancel`/`cancellation`, `delivery fee`/`delivery charge`, `hours`,
`open until`/`close at`, `return policy`, etc.). Any match fails the check unconditionally.
Marked with an inline comment noting it's provisional — remove entries once a real
tool exists for that domain, rather than building a general domain-classification
registry for tools that don't exist yet.

**Rule 2 — menu/price grounding:**
1. Does the reply contain a `$price` pattern, or an availability/recommendation phrase
   ("we have", "it's available", "no longer available", "in stock", "we offer", "try
   the", "I'd recommend", "our specialty is", "how about the")? If neither, pass
   immediately — this is what keeps the guardrail from being overly cautious on plain
   conversation ("Sure, what would you like?").
2. If it does make such a claim: fail if `tool_results` is empty. If not empty, extract
   every `$price` mentioned in the reply and confirm each one appears among the prices
   present in `tool_results` (or the already-verified cart snapshot); any price that
   doesn't trace to real data fails the check.

**Known, documented gap:** a fabricated dish name mentioned without a price or
availability/recommendation phrase attached would not be caught — this needs real named
entity recognition to close fully, which was evaluated and rejected as disproportionate
tooling for a single-digit-item menu (see "Considered and rejected" below). The broadened
phrase list above narrows this gap using the same cheap mechanism already in place,
without closing it entirely.

**Considered and rejected:**
- *Noun-phrase extraction as a cheaper alternative to NER* — evaluated and rejected.
  Reliable noun-phrase chunking (e.g. spaCy's `noun_chunks`) runs on the same
  dependency-parse pipeline as NER — it is not meaningfully cheaper, and is a new model
  dependency for a 3-item menu. A regex fallback (capitalized-word-sequence matching)
  would false-positive on the restaurant's own name and any proper noun in a reply.
- *Tool-argument correctness* (verifying `get_item_details`/`add_item_to_cart` was
  called with an `item_name` that actually matches what the user asked about) — a real,
  narrower gap than "wrong tool selection" (the tools here don't offer a selection
  ambiguity; `search_menu` takes no query argument at all, so there's no scenario
  matching "model calls search_menu('pizza')" against this codebase's actual tool
  signatures). Deferred as a v-next item: implementable cheaply against the user's last
  message, but enforcing it as a hard block risks false positives on indirect item
  references ("the spicy paneer one"), and it requires threading the user's message
  text into the guardrail call on top of `tool_results` — more scope than this pass
  warrants.

## Fallback message

One unified constant, not split by failure rule:

```python
UNVERIFIED_CLAIM_REPLY = "I'm not able to confirm that for you right now — I can note it down and have someone follow up."
```

`check()` returns only `bool`, not a failure reason, so both the policy-keyword and
grounding-failure paths land on the same message. Splitting this into two
rule-specific messages is a small, isolated future change if wanted — not built now
without a concrete need.

## Testing

**`HallucinationGuardrail` unit tests** (TDD — write failing, watch it fail, implement,
watch it pass, matching the process already used for the PII guardrail expansion this
session):
- Plain conversational text, `tool_results=[]` → `True`
- Price mention + `tool_results` containing that exact price → `True`
- Price mention + `tool_results=[]` → `False`
- Price mention + `tool_results` containing a different price → `False`
- Availability/recommendation phrase + no tool call → `False`
- Availability/recommendation phrase + matching tool result → `True`
- Policy keyword → `False` regardless of `tool_results`

**Node-level tests** (`menu_handler_node`, `order_handler_node`), mocking
`groq_client.chat.completions.create`:
- First call returns `tool_calls`, second call returns final content → assert `tools=`
  was passed, the tool was invoked with correct arguments, final state includes the
  reply
- `add_item_to_cart`'s result propagates into the node's returned `cart`
- A tool raising `MenuItemNotFoundError` produces an error tool-message and the loop
  still completes without crashing
- Guardrail returning `False` results in `UNVERIFIED_CLAIM_REPLY` being substituted for
  the generated reply

**Regression:** fix the pre-existing stale-fixture failures
(`test_hallucination_guardrail.py`, `test_menu_tool.py`, `test_order_tool.py` all
hardcode `"Margherita Pizza"`, not on the current menu) as part of this work, since the
guardrail test file is being substantially rewritten anyway. Re-run the full suite to
confirm `test_session_agent.py` is unaffected (it drives the graph via
`self.graph.ainvoke(state)`, so it likely doesn't touch node internals directly, but
this should be confirmed during implementation, not assumed).

## Out of scope / explicitly not built here

- A real policy data source (hours, refund windows, delivery fees) — policy questions
  are always refused for now.
- Physical address / OTP-PIN / name / bank-account / medical-record PII detection (see
  `PiiGuardrail` scoping decisions earlier in this project — same reasoning applies).
- Tool-argument correctness checking (see "Considered and rejected" above).
- A generalized domain-classifier/tool-registry system for extensibility (see "Rule 1"
  above) — the keyword list is deliberately simple and provisional instead.
