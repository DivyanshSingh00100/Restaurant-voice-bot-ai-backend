# Restaurant Voice Bot Backend — full implementation

## Summary

This PR is the entire backend implementation on top of the initial template commit: a
FastAPI service that issues LiveKit call tokens, a LiveKit Agents voice pipeline
(STT → LangGraph → TTS) over Groq, a LangGraph conversation graph per restaurant with
routing/menu/order/escalation nodes, guardrails on bot output, and 140 passing tests.

Two restaurants are served from one deployment (`The Grand Bistro`, `Spice Garden`),
each with its own persona, menu, and isolated conversation graph, selected by
`restaurant_id` at session-start time and carried through as LiveKit room metadata.

This has been tested both with the automated suite and with real live voice calls
against both restaurants — not just unit tests in isolation.

---

## Architecture, layer by layer

**API layer** (`app/api/v1/`)
- `health_router.py` — `GET /api/v1/health`, trivial liveness check.
- `session_router.py` — `POST /api/v1/session/start`: takes `restaurant_id`, creates a
  LiveKit room with that ID stamped into room metadata, mints a scoped access token,
  returns both. This is the entry point a caller's client hits before joining a call.
- `webhook_router.py` — `POST /api/v1/webhooks`, currently a stub (`return {"received":
  True}`). Doesn't verify `LIVEKIT_WEBHOOK_SECRET` or do anything with the payload yet.

**Core** (`app/core/`)
- `config.py` — `pydantic-settings`-based config, all Groq/LiveKit/Redis/restaurant
  settings loaded from `.env`.
- `exceptions.py` — `AppError` base plus `RestaurantNotFoundError`,
  `SessionExpiredError`, `MenuItemNotFoundError`. `RestaurantNotFoundError` is caught
  globally in `main.py` and turned into a 404.
- `logging.py` — `structlog` configuration.

**Integrations** (`app/integrations/`) — thin, lazily-initialized clients: `groq_client`
(`AsyncGroq`) and `get_livekit_api()` (lazy singleton, see Known Limitations for why).

**Schemas** (`app/schemas/api/`) — Pydantic request/response models for the session and
webhook endpoints.

**Prompts** (`app/prompts/`) — `RESTAURANT_A_PERSONA`/`RESTAURANT_B_PERSONA` (system
prompts, each with its menu baked in via `format_menu`) and the escalation transition/
closing messages spoken during handoff.

**Services** (`app/services/`)
- `menu_service.py` — `get_menu(restaurant_id)`, looks up the hardcoded menu list.
- `order_service.py` — `add_to_cart`, `calculate_total`.
- `context_service.py` — Redis-backed conversation state, get/save by session ID.
- `session_service.py` — LiveKit room + token creation for `/session/start`.

**Tools** (`app/tools/`) — LangChain `@tool`-decorated functions: `search_menu`,
`get_item_details`, `add_item_to_cart`, `confirm_order`, `get_order_status`,
`notify_human_agent`, `request_call_transfer`. `search_menu`, `get_item_details`,
`add_item_to_cart`, `confirm_order`, and `get_order_status` are bound to the LLM via the
shared tool-execution loop (`app/graphs/nodes/tool_loop.py`); `notify_human_agent` and
`request_call_transfer` are invoked deterministically by `escalation_handler_node.py`,
not by the LLM.

**Guardrails** (`app/guardrails/`) — a shared `BaseGuardrail.check(text) -> bool`
interface:
- `TopicGuardrail` — keyword block list (`weather`, `stock market`, `politics`,
  `sports scores`).
- `ProfanityGuardrail` — keyword block list (`damn`, `hell`, `crap`).
- `PiiGuardrail` — regex email match; `check()` for detection, `redact()` for replacing
  with `[REDACTED]`.
- `HallucinationGuardrail` — verifies price/availability claims in the bot's reply against
  real tool output from this turn, and is **wired into the live pipeline** in both
  `menu_handler_node.py` and `order_handler_node.py` (see below).

**LangGraph conversation graph** (`app/graphs/`) — one compiled `StateGraph` per
restaurant (`restaurant_a_graph`, `restaurant_b_graph`), sharing the same node/edge
logic:
- `routing_edges.py` — `route_initial_turn` sends turn 0 to `greeter`; afterwards routes
  by keyword match on the last user message (`human`/`person`/`agent` → escalation;
  `order`/`want`/`i'll have` → order handler; else → menu handler).
  `route_after_handling` checks the same escalation keywords after menu/order handling,
  otherwise ends the turn.
- `nodes/greeter_node.py` — builds a system prompt (persona + relevant context) and calls
  `groq_client.chat.completions.create` directly with the full message history.
  `menu_handler_node.py` and `order_handler_node.py` instead build their system prompt
  and go through the shared tool-execution loop (`run_with_tools` in
  `nodes/tool_loop.py`), which binds their respective tools, executes any tool calls the
  model makes, and runs the reply through `HallucinationGuardrail` before returning it.
- `nodes/escalation_handler_node.py` — directly `.invoke()`s the `notify_human_agent`
  and `request_call_transfer` tools (deterministic, not LLM-driven), appends the
  transition and closing messages.
- `conversation_state.py` — the shared `TypedDict`: `restaurant_id`, `messages`,
  `turn_count`, `cart`.

**Voice layer / LiveKit agents** (`app/agents/`) — the new real-time layer added this
round:
- `voice_pipeline_agent.py` — builds the `AgentSession` (Silero VAD, Groq STT, Groq TTS).
- `restaurant_a_agent.py` / `restaurant_b_agent.py` — per-restaurant `Agent` subclasses.
- `session_agent.py` (`RestaurantVoiceAgent`) — overrides `on_enter`,
  `on_user_turn_completed`, `llm_node`. `llm_node` is where guardrails run
  (topic/profanity hard-block before the graph call; PII redacted on both directions),
  the LangGraph graph gets invoked, and the reply is saved to Redis before being
  streamed to TTS.
- `escalation_agent.py` — detects the graph's closing message and calls
  `session.shutdown()`.
- `worker_agent.py` — LiveKit worker entrypoint; reads `restaurant_id` from room
  metadata (with a retry loop, see below) and dispatches to the right agent.

**Tests** (`tests/`) — 140 tests across all of the above: guardrails, tools, services,
schemas, graph nodes/routing, prompts, exceptions, and the new agents layer.

---

## Bugs found and fixed via live testing

Live voice calls surfaced three real bugs that unit tests alone didn't catch:

1. **Redis client crashing across event loops.** `context_service.py` used to create its
   Redis client eagerly at import time, binding it to whatever event loop was active
   then. LiveKit runs each call in its own process, so real calls hit `RuntimeError:
   ...attached to a different loop`. Fixed with a lazy singleton (`get_redis_client()`),
   mirroring the existing pattern in `integrations/livekit_client.py`.
2. **Room-metadata replication race.** Occasionally a worker got dispatched before a
   newly-created room's metadata had replicated to its region, producing a false
   `RestaurantNotFoundError` on a genuinely valid room. Mitigated with a
   retry-with-backoff read in `worker_agent.py` (`wait_for_restaurant_id`, 5 attempts,
   0.5s apart) instead of trusting the first read.
3. **Duplicate replies per turn.** `AgentSession`'s "preemptive generation" speculatively
   runs `llm_node` on an interim transcript before the turn is confirmed. Our `llm_node`
   isn't safe to run speculatively — it writes to Redis before yielding — so a discarded
   speculative run could still leave a phantom assistant message saved, and the real run
   would build on top of it, visibly producing two near-identical replies per turn and
   roughly doubling real Groq API calls. Fixed by disabling `preemptive_generation` on
   the `AgentSession` (traded a bit of latency for correctness, confirmed fixed against
   live logs — exactly one reply per turn afterward).

Also updated `GROQ_TTS_MODEL` (`playai-tts`, permanently shut down, → `canopylabs/
orpheus-v1-english`) and `GROQ_LLM_MODEL` (`llama-3.3-70b-versatile`, deprecating
Aug 2026, → `openai/gpt-oss-120b`), and fixed broken version pins in `requirements.txt`.

---

## Known limitations / flags for review

1. **Resolved.** `search_menu`, `get_item_details`, `add_item_to_cart`, `confirm_order`,
   and `get_order_status` are now bound to the LLM via a shared tool-execution loop
   (`app/graphs/nodes/tool_loop.py`), used by both `menu_handler_node.py` and
   `order_handler_node.py`. The model's `cart` tool argument is never trusted: every
   `order_handler_node.py` tool call has its `cart` argument overridden server-side with
   the real `ConversationState["cart"]` via `run_with_tools(..., arg_overrides=...)`, so
   the model cannot fabricate items or prices via its tool-call JSON. `add_item_to_cart`'s
   real result then propagates back into `ConversationState["cart"]`, so cart contents and
   totals are always real, never LLM-guessed. See
   `docs/superpowers/specs/2026-07-28-hallucination-guardrail-tool-grounding-design.md`.
2. **Resolved.** `HallucinationGuardrail` is redesigned around tool-result grounding
   (verifies price/availability claims against real tool output this turn, rather than
   exact-string menu-item matching) and is wired into both `menu_handler_node.py` and
   `order_handler_node.py`. A documented residual gap remains: a fabricated dish name
   mentioned without a price or availability phrase attached isn't caught — full
   coverage needs NER, evaluated and deliberately not built (see design doc).
3. **Groq's Orpheus TTS model is still in preview and has real operational
   constraints:**
   - Requires explicit terms acceptance per Groq account, done manually via the Groq
     console. Without it, every TTS call 400s — not something code can fix.
   - Hard 200-character input limit per request. Both personas were given short-reply
     instructions specifically to stay under this — this is load-bearing prompt
     engineering, not just style preference.
   - During extended live testing, TTS started returning 429 on effectively every
     request even several seconds apart, which doesn't fit a simple per-minute rate
     limit. Current guess is a daily/preview-tier quota, unconfirmed. Worth checking
     current usage on console.groq.com before any live demo — this can silently degrade
     to "bot never speaks" with zero code changes involved.
4. **One unexplained STT anomaly during testing**: three consecutive phantom user
   transcripts with identical text appeared with no one speaking. Not reproduced on
   demand; leading theory is audio bleed/feedback in the test environment rather than a
   code bug (genuine STT hallucination-on-silence tends to produce generic junk text,
   not a coherent repeated sentence). Flagging so it's not a surprise if it recurs.
5. **`webhook_router.py` is a stub** — accepts any POST and returns `{"received":
   True}`. Doesn't verify `LIVEKIT_WEBHOOK_SECRET` or process any LiveKit event
   (participant joined/left, room finished, etc.).
6. **Escalation ends the call entirely.** Saying anything containing "human", "person",
   or "agent" triggers `session.shutdown()` — there's no live queue to actually transfer
   to yet, so this is a hard stop, not a warm handoff. Worth confirming this is the
   intended behavior for now.
7. **No conversation history trimming.** Full message history is resent to the LLM every
   turn with no cap or summarization. Fine for test-length calls; will grow the prompt
   (and cost, and eventually context-window pressure) on long ones.
8. **CORS is wide open** (`allow_origins=["*"]`) in `main.py` — fine for local dev,
   should be scoped down before anything resembling a real deployment.
9. **Secrets**: confirmed `.env` is gitignored and was never tracked — no real API keys
   or credentials are part of this diff.

---

## How this was tested

- Full automated suite: `pytest tests`, 140 passing.
- Live voice calls against both restaurant personas via LiveKit's hosted test client:
  greeting, menu Q&A, ordering, running totals, mixed menu/order back-and-forth,
  all three active guardrails, and escalation.
- Manual verification of each of the three bug fixes above, confirmed against real
  before/after logs from live calls.
