# Intelligent Model Routing & Oversight — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add intelligent per-turn model routing, quality-based failure escalation, and periodic oversight to Hermes Agent.

**Architecture:** Three-layer system — heuristic turn router (pre-inference), quality signal detector (post-inference), periodic oversight reviewer (every N turns, async). Builds on existing fallback/background_review patterns.

**Tech Stack:** Python 3.11, existing AIAgent loop, follows `background_review.py` architectural pattern for oversight.

**ADR:** `docs/adrs/ADR-0040-intelligent-model-routing.md`

**Branch:** `feature/model-routing-and-oversight`

---

## Phase 1: Turn Router (Foundation)

### Task 1.1: Routing config schema and defaults

**Objective:** Define the config shape for model routing with sensible defaults.

**Files:**
- Create: `agent/routing_config.py`
- Modify: `hermes_cli/config.py` (add routing section to DEFAULT_CONFIG)

**Design decisions:**
- Config lives under `model.routing:` in config.yaml
- Disabled by default (`enabled: false`) — zero behavior change for existing users
- When enabled, requires at least `primary` and `escalation` model definitions
- All thresholds have defaults that work without tuning

**Schema:**
```python
ROUTING_DEFAULTS = {
    "enabled": False,
    "strategy": "hybrid",           # hybrid | manual | always-primary
    "primary": {},                  # model/provider/base_url/api_key
    "escalation": {},               # model/provider/base_url/api_key
    "triggers": {
        "user_prefix": ["@opus", "@hard"],
        "keywords": [
            {"pattern": r"architect|design\s+system|refactor\s+entire|debug\s+complex", "weight": 0.6},
            {"pattern": r"why\s+does|explain\s+why|what\s+went\s+wrong", "weight": 0.3},
        ],
        "tool_call_failures": 2,
        "stagnation_turns": 3,
        "context_size": 100000,
        "message_length": 2000,
        "complexity_threshold": 0.7,
    },
    "hysteresis": {
        "escalation_sticky_turns": 3,   # min turns to stay on escalation model
        "deescalation_threshold": 0.3,  # must score below this to return to primary
    },
    "oversight": {
        "enabled": False,
        "model": None,                  # defaults to escalation model if not set
        "provider": None,
        "every_n_turns": 10,
        "review_window": 10,
        "max_reviews_per_session": 5,
        "min_turns_before_first": 5,
        "skip_if_escalated": True,
        "actions": {
            "approve": "silent",
            "correct": "inject",
            "escalate": "takeover",
            "flag": "notify_user",
        },
    },
}
```

---

### Task 1.2: TurnRouter classifier

**Objective:** Implement the heuristic classifier that scores message complexity and returns a routing decision.

**Files:**
- Create: `agent/model_router.py`
- Create: `tests/test_model_router.py`

**Key design:**
- Pure function, no LLM calls, no I/O
- Takes message + context → returns `ModelTier.ROUTINE | ModelTier.COMPLEX`
- Context includes: recent_tool_failures, turns_since_progress, context_tokens, session_complexity_score
- Hysteresis: tracks turns since last escalation, resists ping-ponging

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import re

class ModelTier(Enum):
    ROUTINE = "routine"
    COMPLEX = "complex"

@dataclass
class RoutingContext:
    recent_tool_failures: int = 0
    turns_since_progress: int = 0
    current_context_tokens: int = 0
    session_complexity_score: float = 0.0
    turns_since_escalation: int = 999
    total_turns: int = 0
    recent_messages: List[str] = None

class TurnRouter:
    def __init__(self, config: dict):
        self.config = config
        self._triggers = config.get("triggers", {})
    
    def classify(self, message: str, context: RoutingContext) -> ModelTier:
        """Classify a user message into a routing tier."""
        # 1. Explicit user triggers (highest priority)
        if self._has_explicit_trigger(message):
            return ModelTier.COMPLEX
        
        # 2. Hysteresis — resist de-escalation
        if context.turns_since_escalation < self.config.get("hysteresis", {}).get("escalation_sticky_turns", 3):
            return ModelTier.COMPLEX
        
        # 3. Context-based signals
        if context.recent_tool_failures >= self._triggers.get("tool_call_failures", 2):
            return ModelTier.COMPLEX
        if context.turns_since_progress > self._triggers.get("stagnation_turns", 3):
            return ModelTier.COMPLEX
        if context.current_context_tokens > self._triggers.get("context_size", 100000):
            return ModelTier.COMPLEX
        
        # 4. Message complexity scoring
        score = self._score_complexity(message)
        context.session_complexity_score = (
            context.session_complexity_score * 0.9 + score * 0.1
        )  # EMA
        
        threshold = self._triggers.get("complexity_threshold", 0.7)
        if score >= threshold:
            return ModelTier.COMPLEX
        
        return ModelTier.ROUTINE
```

---

### Task 1.3: Router integration into AIAgent

**Objective:** Wire the router into the conversation loop so it selects the model before each inference call.

**Files:**
- Modify: `run_agent.py` — add router invocation in the main loop
- Modify: `agent/agent_init.py` — initialize router from config
- Create: `agent/routing_runtime.py` — runtime state management (which model is active, swap logic)

**Key integration points:**
- Router is initialized in `_init_routing()` during `AIAgent.__init__`
- Before each `client.chat.completions.create()`, call `self._route_turn(user_message)`
- `_route_turn()` may swap `self.client`, `self.model`, `self.api_mode` for one turn
- After the turn completes, routing state is updated (turn counter, failure tracking)
- Swapping uses the same mechanism as `_try_activate_fallback()` / `_restore_primary_runtime()`

**Critical constraint:** Must not break prompt caching. If models share a prefix cache (same provider), swapping is cheap. If they don't (local vs cloud), the swap inherently breaks cache — this is acceptable (the escalation is worth the cache miss).

---

### Task 1.4: TUI status bar integration

**Objective:** Show which model is handling the current turn in the TUI status bar.

**Files:**
- Modify: `ui-tui/src/components/StatusBar.tsx` — add routing indicator
- Modify: `tui_gateway/` — emit routing state in status updates

**Design:**
- When routing is enabled: show model name or short label (e.g., `🏠 local` vs `☁️ opus`)
- On escalation: brief flash/highlight to indicate the swap
- Show routing decision in verbose mode (`/verbose`)

---

## Phase 2: Quality Signal Escalation

### Task 2.1: Quality signal detector

**Objective:** Detect when the primary model's response indicates struggle.

**Files:**
- Create: `agent/quality_signals.py`
- Create: `tests/test_quality_signals.py`

**Signals to detect:**
- Malformed tool calls (JSON parse failure in tool_call arguments)
- Empty responses or very short non-answers
- Repetitive content (cosine similarity > 0.9 with previous response)
- Tool call loops (same tool + same args called 3+ times)
- Hedging language density ("I'm not sure", "I think maybe", "let me try")

**Important:** Signal baselines must be model-specific. Some local models have structurally higher tool-call failure rates even on routine tasks. Without per-model baselines, you get permanent escalation. Track per-model signal frequencies during a calibration period (first 20 turns) and compare against the model's own baseline, not a universal threshold.

---

### Task 2.2: Post-inference escalation hook

**Objective:** After getting a response from the primary model, check quality signals. If escalation is warranted, discard the response and replay on the escalation model.

**Files:**
- Modify: `run_agent.py` — add post-response quality check
- Modify: `agent/routing_runtime.py` — add escalation replay logic

**Key behavior:**
- Quality check runs ONLY when primary model is active (not when already on escalation)
- Discarded response is never shown to user or appended to history
- Escalation model gets the same messages + optional note about why escalation fired
- Escalation is logged for later tuning

---

## Phase 2.5: `ask_upper` Tool

### Task 2.5.1: `ask_upper` tool implementation

**Objective:** Give the lower model a tool to proactively request help from the upper model.

**Files:**
- Create: `agent/tools/ask_upper.py`
- Create: `tests/test_ask_upper.py`
- Modify: `agent/tool_registry.py` — conditionally register when routing is enabled

**Key design:**
- Tool only appears in the tool list when `model.routing.enabled: true` AND an upper model is configured
- Uses `auxiliary_client` pattern — separate client instance for upper model calls
- Input `context` parameter is truncated to 4K tokens before forwarding
- Cost tracked as cumulative token budget (not call count): warn at configurable dollar threshold
- Upper model gets a mentor system prompt that encourages concise, actionable responses

### Task 2.5.2: Mentor prompt engineering

**Objective:** Design and test the system prompt for the upper model when acting as a mentor.

**Files:**
- Add to: `agent/tools/ask_upper.py` — prompt templates per request_type

**The mentor prompt should:**
- Frame itself as a peer mentor, not an authority
- Encourage the lower model's independence
- Produce structured output (numbered steps for plans, yes/no + reasoning for verify)
- Stay concise (upper model has `max_tokens=2000` cap)
- Adapt tone to `request_type` (explain is pedagogical, verify is binary, plan is structured)

---

## Phase 3: Periodic Oversight

### Task 3.1: Oversight reviewer

**Objective:** Implement periodic oversight that reviews recent turns using a stronger model.

**Files:**
- Create: `agent/oversight.py`
- Create: `tests/test_oversight.py`

**Architecture:**
- Runs **synchronously between turns** (not async — compression requires the context rebuild to complete before the lower model continues)
- Reuses `background_review.py` infrastructure (same agent-forking machinery, different prompt/tools)
- Sends last N turns + oversight prompt to upper model
- Parses structured JSON response → `OversightAction`
- Performs context compression and rebuild in the main conversation loop
- Brief pause every N turns (~3-5s) — acceptable tradeoff for guaranteed correctness
- The review window is dynamically capped: `min(config.review_window, int(upper_model_context * 0.6 / avg_tokens_per_turn))`

---

### Task 3.2: Oversight action handlers

**Objective:** Implement the four oversight actions: approve, correct, escalate, flag.

**Files:**
- Modify: `agent/oversight.py` — action dispatch
- Modify: `run_agent.py` — injection point for corrections
- Modify: `agent/display.py` or TUI — user notification for flags

**Action semantics:**
- `approve` → no-op, log only; compression still happens
- `correct` → inject system message before next user turn
- `escalate` → set routing to `ModelTier.COMPLEX` for next N turns (sticky)
- `flag` → emit user-visible notification via TUI event

**Critical:** Oversight runs synchronously. The context rebuild (compression) happens in the main loop AFTER oversight returns and BEFORE the next user turn is processed. No race conditions with concurrent inference.

---

### Task 3.3: Oversight turn counter and budget

**Objective:** Track turns and trigger oversight at the configured interval with budget caps.

**Files:**
- Modify: `agent/oversight.py` — scheduling logic
- Modify: `run_agent.py` — increment counter after each turn

**Logic:**
```python
def should_run_oversight(self) -> bool:
    if not self.oversight_config.get("enabled"):
        return False
    if self.oversight_calls >= self.oversight_config.get("max_reviews_per_session", 5):
        return False
    if self.total_turns < self.oversight_config.get("min_turns_before_first", 5):
        return False
    if self.oversight_config.get("skip_if_escalated") and self._last_turn_was_escalated:
        return False
    return self.turns_since_last_oversight >= self.oversight_config.get("every_n_turns", 10)
```

---

## Phase 4: Observability

### Task 4.1: Routing decision log

**Objective:** Log every routing decision for debugging and threshold tuning.

**Files:**
- Create: `agent/routing_log.py`
- Modify: agent/model_router.py — emit log entries

**Log format:**
```json
{"turn": 5, "tier": "routine", "score": 0.23, "signals": [], "model": "qwen3-coder-next", "ts": "..."}
{"turn": 12, "tier": "complex", "score": 0.85, "signals": ["keyword:architect", "context_size"], "model": "opus-4-6", "ts": "..."}
{"turn": 15, "tier": "complex", "source": "oversight_escalation", "reason": "circular debugging", "ts": "..."}
```

---

### Task 4.2: Slash commands

**Objective:** Add `/routing` and `/oversight` commands for in-session visibility.

**Files:**
- Modify: `hermes_cli/commands.py` — register commands
- Modify: `cli.py` — command handlers

**`/routing`:** Shows current routing state, active model, recent decisions, session complexity score.  
**`/oversight`:** Shows oversight history (when it ran, what actions it took), next scheduled review.

---

## Dependencies & Prerequisites

1. **Qwen3-Coder-Next running locally** — needed to test routing between local and cloud
2. **llama-server with KV cache quantization** — for the 256K context local setup
3. **Existing test suite passes** — `python -m pytest tests/ -o 'addopts=' -q` before starting

## Risk Notes

- Phase 1 is independently useful (just routing, no oversight) — can ship alone
- Phase 2.5 (`ask_upper`) is independently useful — exercises the upper-model-as-tool pattern needed by Phase 3
- Phase 3 follows a proven pattern (background_review.py) — lower risk than it looks
- All phases are behind `routing.enabled: true` flag — zero impact on users who don't opt in

---

## Phase 5: Resilience, Calibration & Auto-Configuration

### Task 5.1: Fallback chain design

**Objective:** Each node in the capability graph has a designated fallback when its upper model is unreachable.

**Design:**

The fallback chain is separate from the capability graph. It answers: "If I need my upper model and it's not responding, what do I do?"

```
Scenario: Lower model hits oversight checkpoint, upper model (cloud) is unreachable.

Fallback chain (tried in order):
1. Retry with backoff (3 attempts, 2s/4s/8s)
2. Try alternate upper model if configured (e.g., Sonnet instead of Opus)
3. Skip this checkpoint — continue with stale context, schedule retry at next natural break
4. If context is at emergency levels (>85% budget): naive truncation (drop oldest non-system messages)
5. If all else fails: halt and notify user ("⚠️ Upper model unreachable, context growing. /routing manual to continue without oversight.")

Scenario: Lower model calls ask_upper, upper model unreachable.

Fallback:
1. Return a structured error to the lower model: "Upper model unavailable. Proceed with your best judgment."
2. The lower model continues (it's just a tool call that returned an error — it handles this already)
```

**Files:**
- Create: `agent/routing_fallback.py`
- Modify: `agent/oversight.py` — use fallback chain on upper model failure
- Modify: `agent/tools/ask_upper.py` — graceful degradation

**Config:**
```yaml
model:
  routing:
    fallback_chain:
      retry_attempts: 3
      retry_backoff_base: 2          # seconds
      alternate_upper:               # optional secondary upper model
        model: us.anthropic.claude-sonnet-4-20250514
        provider: bedrock
      on_persistent_failure: skip    # skip | halt | truncate
      halt_message: "⚠️ Upper model unreachable. Use `/routing manual` to continue without oversight."
```

---

### Task 5.2: Full calibration pass

**Objective:** Prevent compression drift over very long sessions by periodically re-reading raw history from disk.

**Design:**

Every M checkpoints (default: 5), the upper model performs a "full calibration" — instead of just reading the previous summary + recent turns, it re-ingests the full raw session history from the SQLite message store.

```python
def should_full_calibrate(self) -> bool:
    """Every M checkpoints, do a full re-read instead of differential."""
    return self.checkpoint_count % self.full_calibration_interval == 0

def perform_full_calibration(self, session_id: str):
    """Re-read full raw history and produce fresh summary."""
    # Load all messages from SQLite (not from context window)
    raw_history = self.session_db.get_all_messages(session_id)
    
    # Chunk if needed (raw history may exceed upper model context)
    if self.token_count(raw_history) > self.upper_model_context * 0.7:
        # Progressive summarization: summarize in chunks, then summarize summaries
        chunks = self._chunk_history(raw_history, target_chunk_size=50000)
        chunk_summaries = [self._summarize_chunk(c) for c in chunks]
        raw_for_calibration = "\n".join(chunk_summaries)
    else:
        raw_for_calibration = raw_history
    
    # Produce fresh summary (not differential — from scratch)
    return self._full_summarize(raw_for_calibration)
```

**Key insight:** The upper model's context window (200K) is large enough to hold most sessions' raw history for calibration. Only sessions that have produced >200K tokens of raw messages need the chunked approach. Most sessions (even long ones) will fit in a single calibration call.

**Files:**
- Modify: `agent/oversight.py` — add calibration scheduling
- Create: `agent/calibration.py` — full re-read + summarization logic

**Config:**
```yaml
model:
  routing:
    oversight:
      full_calibration_interval: 5    # every Nth checkpoint, re-read from disk
```

---

### Task 5.3: Auto-configuration via `/routing setup`

**Objective:** Users should not need to write routing YAML by hand. A slash command discovers configured models, infers a capability graph, and presents it for approval.

**Design:**

```
User types: /routing setup

Hermes responds:
┌──────────────────────────────────────────────────────────┐
│  🔍 Discovered Models                                     │
├──────────────────────────────────────────────────────────┤
│  ☁️  Claude Opus 4.6        (bedrock, 200K ctx, $$$)     │
│  ☁️  Claude Sonnet 4        (bedrock, 200K ctx, $$)      │
│  🏠 Qwen3-Coder-Next       (local:58080, 262K ctx, $0)  │
│  🏠 Qwen3.6-27B            (local:58080, 262K ctx, $0)  │
├──────────────────────────────────────────────────────────┤
│  📊 Proposed Capability Graph                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Opus 4.6 ─── oversight, mentor, escalation              │
│      │                                                    │
│      ├── Sonnet 4 ─── mid-tier fallback                  │
│      │       │                                            │
│      │       ├── Coder-Next ─── code tasks (primary)     │
│      │       └── Qwen3.6-27B ─── prose tasks             │
│      │                                                    │
│      └── [direct escalation for hardest problems]        │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  ⚙️  Settings                                             │
│  Primary model: Qwen3-Coder-Next (code specialist)       │
│  Oversight every: 10 turns                               │
│  Escalation to: Sonnet 4 (then Opus for complex)         │
│                                                          │
│  [Accept] [Customize] [Cancel]                           │
└──────────────────────────────────────────────────────────┘
```

**How it works:**

1. **Discovery:** Scan `config.yaml` for all configured models/providers (including custom providers with base_url). Check which local models are currently serving (ping health endpoint).

2. **Classification:** For each model, infer capabilities from:
   - Known model name → lookup in a capability database (built-in for popular models)
   - Context window size (from config or probed via API)
   - Cost tier (from provider + model → known pricing)
   - Locality (custom provider on localhost = local/free)
   - If unknown: ask the user "What's this model good at?" or probe with a complexity test

3. **Graph construction:** Given the capability classifications, construct a directed graph:
   - Sort models by capability tier (cost is a strong proxy)
   - Local models become lower nodes
   - Cheapest cloud becomes mid-tier (if multiple cloud models configured)
   - Most expensive cloud becomes top-tier oversight
   - Multiple local models at same tier → specialize by strength (code vs prose)

4. **Presentation:** Show the proposed graph to the user with explanation. Allow customization (drag nodes, change edges, adjust thresholds).

5. **Persisting:** On accept, write the routing config to `config.yaml`. Show the user what was written.

**Files:**
- Create: `agent/routing_autoconfig.py` — discovery, classification, graph construction
- Create: `agent/model_capabilities.py` — built-in capability database for known models
- Modify: `hermes_cli/commands.py` — add `/routing setup` command
- Create: `tests/test_routing_autoconfig.py`

**Capability database (built-in, extensible):**
```python
KNOWN_MODELS = {
    "claude-opus-4-6": {
        "tier": "frontier",
        "strengths": ["reasoning", "architecture", "planning", "summarization"],
        "context": 200000,
        "cost_tier": "high",
    },
    "claude-sonnet-4": {
        "tier": "strong",
        "strengths": ["general", "code", "fast"],
        "context": 200000,
        "cost_tier": "medium",
    },
    "qwen3-coder-next": {
        "tier": "capable",
        "strengths": ["code", "tool-calling", "instruction-following"],
        "context": 262144,
        "cost_tier": "free",
        "locality": "local",
    },
    "qwen3.6-27b": {
        "tier": "capable",
        "strengths": ["prose", "research", "general"],
        "context": 262144,
        "cost_tier": "free",
        "locality": "local",
    },
    # ... extensible
}
```

---

### Task 5.4: `--dry-run` mode for router

**Objective:** Let users test routing decisions without actually swapping models.

**Files:**
- Modify: `agent/model_router.py` — add `dry_run` flag
- Modify: `agent/routing_log.py` — emit verbose dry-run output

**Behavior:**
- `hermes config set model.routing.dry_run true` or `/routing dry-run on`
- Router runs normally, logs what it WOULD do, but always uses the primary model
- After 20+ turns of dry-run data: `/routing stats` shows distribution ("72% routine, 28% complex — estimated savings: $X/session")

---

### Task 5.5: `/routing` slash command suite

**Objective:** In-session visibility and control over routing behavior.

**Commands:**
| Command | Effect |
|---------|--------|
| `/routing` | Show current state: active model, turn count per model, cost estimate |
| `/routing setup` | Auto-configuration wizard (Task 5.3) |
| `/routing stats` | Session statistics: routing decisions, model usage %, estimated savings |
| `/routing manual` | Disable automatic routing, user picks model manually |
| `/routing auto` | Re-enable automatic routing |
| `/routing dry-run [on\|off]` | Toggle dry-run mode |
| `/routing escalate` | Force next turn to upper model |
| `/routing handback` | Force handback to lower model on next turn |

**Files:**
- Modify: `hermes_cli/commands.py` — register suite
- Create: `agent/routing_commands.py` — command handlers
