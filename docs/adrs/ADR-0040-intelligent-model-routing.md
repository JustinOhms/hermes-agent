# ADR-0040: Intelligent Model Routing & Periodic Oversight

**Status:** Draft  
**Date:** 2026-06-06  
**Author:** Justin Ohms + Hermes  
**Deciders:** Justin Ohms  
**Tags:** model-routing, oversight, cost-optimization, local-models, multi-model

## Context

### The Problem

Hermes currently has a binary model selection:
- **Primary model** — handles every turn
- **Fallback model** — activates only on hard failures (API down, auth broken, rate limit)

This forces users to choose between:
1. **Cloud-primary (expensive, best quality)** — every turn costs money, even trivial ones
2. **Local-primary (free, fast, good-not-great)** — complex tasks suffer from weaker reasoning
3. **Manual switching** — `/model opus` mid-session when you "know" you need it

None of these serve the emerging use case: a capable local model (Qwen3-Coder-Next, 256K context, Sonnet-competitive on benchmarks) as the daily driver with automatic escalation to a frontier model (Opus) for genuinely difficult work — plus periodic quality oversight to catch drift.

### Current Architecture

```
User message → AIAgent → Primary model → response
                              ↓ (on failure)
                         Fallback model → response
```

Key existing patterns we can build on:
- `_try_activate_fallback()` / `_restore_primary_runtime()` — already swap models mid-session
- `background_review.py` — forks the agent with a different prompt for post-turn review (memory/skill extraction)
- `agent/transports/base.py` — clean transport abstraction (provider-agnostic)
- `credential_pool.py` — already rotates credentials; routing builds alongside this
- `auxiliary_client.py` — separate client for non-primary tasks (vision, compression)

### Motivation

With 128 GB Apple Silicon machines and models like Qwen3-Coder-Next:
- **48.73 GB** model weights at Q4_K_M
- **12 GB** KV cache at full 256K context with Q8_0 quantization
- **~15-25 tok/s** — fast enough for interactive use
- **$0/turn** — no API cost

Meanwhile Opus on Bedrock costs ~$0.50-2.00 per complex turn. If 80% of turns are routine (read file, edit, run test, navigate code), routing those locally saves significant cost while maintaining quality where it matters.

## Decision

Implement a three-layer intelligent routing system:

1. **Turn Router** — decides which model handles each turn before inference
2. **Failure Escalation** — auto-retries on a stronger model when the primary stumbles (extends existing fallback)
3. **Periodic Oversight** — a stronger model reviews recent turns and can course-correct, escalate, or flag

### Non-goals (this ADR)
- Fine-tuning a custom router model
- Automatic model discovery/benchmarking
- Multi-model ensemble (running multiple models and picking best response)
- Token-level routing (different models for different parts of one response)

### Terminology

- **Upper model** — the more capable (and typically more expensive) model in a pair. Provides oversight, mentoring, compression, and handles escalated work.
- **Lower model** — the primary workhorse (typically local, free, fast). Handles routine turns and can proactively ask the upper model for help.
- **Capability graph** — a directed graph where nodes are models and edges represent upper/lower relationships. For v1, this is a simple two-node graph.
- **Handback** — when the upper model finishes its work and returns control to the lower model, along with a compressed context package.
- **Oversight checkpoint** — a synchronous pause every N turns where the upper model reviews and optionally compresses context.

---

## Design

### 1. Turn Router

#### Strategy: Hybrid (heuristic + explicit + post-hoc)

The router runs **before** sending to any model. Zero LLM overhead — pure heuristic.

```
User message → TurnRouter.classify() → ModelTier → select model → inference
```

#### Classification Tiers

| Tier | Description | Route to |
|------|-------------|----------|
| `routine` | File reads, simple edits, test runs, navigation, short answers | Primary (local) |
| `complex` | Multi-file refactors, architecture, debugging complex issues, long planning | Escalation model |
| `explicit` | User prefixed with trigger (e.g., `@opus`, `!hard`) | Escalation model |

#### Heuristic Signals (no LLM call)

```python
class TurnRouter:
    def classify(self, message: str, context: RoutingContext) -> ModelTier:
        # 1. Explicit user triggers (highest priority)
        if self._has_explicit_trigger(message):
            return ModelTier.COMPLEX
        
        # 2. Context-based signals
        if context.recent_tool_failures >= self.config.failure_threshold:
            return ModelTier.COMPLEX
        if context.turn_count_since_progress > self.config.stagnation_threshold:
            return ModelTier.COMPLEX
        if context.current_context_tokens > self.config.context_size_threshold:
            return ModelTier.COMPLEX
        
        # 3. Message-based heuristics
        complexity_score = self._score_message_complexity(message)
        if complexity_score >= self.config.complexity_threshold:
            return ModelTier.COMPLEX
        
        return ModelTier.ROUTINE
```

#### Complexity Scoring (lightweight, no ML)

Factors:
- **Message length** — longer prompts tend to be more complex asks
- **Keyword presence** — "architect", "design", "debug", "why does", "refactor entire", "think through"
- **Question density** — multiple questions in one message
- **Code block size** — large code dumps for review suggest complex work
- **Reference to multiple files** — cross-cutting changes are harder
- **Negation/correction patterns** — "that's wrong", "no, I meant", "why didn't you" suggest the local model may have already struggled

These are **tunable weights** with sensible defaults, not magic numbers.

#### Config Shape

```yaml
model:
  routing:
    enabled: true
    strategy: hybrid                    # hybrid | manual | always-primary
    
    primary:
      model: qwen3-coder-next
      provider: custom
      base_url: http://127.0.0.1:58080/v1
      api_key: sk-local
    
    escalation:
      model: us.anthropic.claude-opus-4-6-v1
      provider: bedrock
    
    triggers:
      user_prefix: ["@opus", "@hard", "!think"]
      keywords:
        - pattern: "architect|design.*system|refactor.*entire|debug.*complex"
          weight: 0.6
        - pattern: "why does|explain.*why|what went wrong"
          weight: 0.3
      tool_call_failures: 2             # escalate after N consecutive failures
      stagnation_turns: 3               # escalate if no progress in N turns
      context_size: 100000              # escalate when context > N tokens
      message_length: 2000             # long messages get complexity boost
      complexity_threshold: 0.7         # score above this → escalate
```

#### Open Questions — Turn Router

1. **Should routing be sticky?** If Opus takes over for one turn, does it stay on Opus until the subtask completes? Or does each turn get re-evaluated?
   - Leaning: **re-evaluate each turn** but with hysteresis (once escalated, require a lower score to de-escalate, preventing ping-pong)
   
2. **Should the user see routing decisions?** 
   - Leaning: **subtle indicator** in TUI status bar (icon showing which model is active), verbose log for debugging, but no interruption

3. **What happens when the local model is down?**
   - Behaves like current fallback: escalation model becomes the only option until local recovers

---

### 1b. Interaction Mode — Latency vs. Quality Tradeoff

#### The Insight

The Turn Router classifies **what** to route. Interaction Mode classifies **when** — specifically, whether anyone is waiting for the response. This creates a second axis for model selection:

- **Interactive mode:** User is present, conversational, expecting sub-second token streaming. Optimize for **latency** (tok/s). Use the fastest capable model.
- **Autonomous mode:** No one is watching — overnight sessions, cron jobs, delegate_task subagents, deep non-interactive work. Optimize for **reasoning quality**. Latency is irrelevant; a 30-second dense-model response that's smarter is strictly better.

This resolves the "Qwen3.6-27B is too slow" problem: it's too slow *interactively* (6 tok/s), but for autonomous work it's fine — and its denser architecture may produce better reasoning than a faster MoE with 3B active parameters.

#### Model Selection Matrix

| Interaction Mode | Complexity Tier | Selected Model | Rationale |
|------------------|-----------------|----------------|-----------|
| `interactive` | `routine` | Coder-Next (MoE, 33 tok/s) | Smart + fast + free, user waiting |
| `interactive` | `complex` | Opus (cloud) | User waiting, quality needed |
| `autonomous` | `routine` | Qwen3.6-27B (dense, 6 tok/s) | Quality > speed, nobody watching |
| `autonomous` | `complex` | Opus (cloud) | Hardest problems still escalate |

The interaction mode doesn't replace complexity classification — it **modulates the lower model selection**. Complex work still goes to the upper model regardless of mode. But for routine/moderate work, autonomous mode prefers a smarter-but-slower model because the latency penalty is invisible.

#### Detection Heuristics (automatic, no explicit mode switch)

```python
class InteractionModeDetector:
    """Classifies the current session's interaction mode."""
    
    def classify(self, context: RoutingContext) -> InteractionMode:
        # 1. Platform signals (highest confidence)
        if context.platform == "cron":
            return InteractionMode.AUTONOMOUS
        if context.is_subagent:  # delegate_task child
            return InteractionMode.AUTONOMOUS
        
        # 2. Skill signals
        if "autonomous-overnight-work" in context.loaded_skills:
            return InteractionMode.AUTONOMOUS
        
        # 3. Temporal signals (progressive detection)
        time_since_last_user_msg = now() - context.last_user_message_time
        if time_since_last_user_msg > self.config.autonomous_threshold:
            return InteractionMode.AUTONOMOUS
        
        # 4. Session continuity patterns
        if context.consecutive_agent_turns > self.config.unattended_turn_count:
            # Agent has been working alone for many turns — user likely away
            return InteractionMode.AUTONOMOUS
        
        # 5. Explicit user signals (override)
        if context.user_declared_mode:
            return context.user_declared_mode
        
        return InteractionMode.INTERACTIVE
```

#### Temporal Detection: The "Dropoff" Pattern

The most common case: user is working interactively, then stops responding (goes to bed, steps away). The system should detect this transition and switch models:

```
User messages at: 10:01, 10:03, 10:05, 10:12, 10:15
                                                      └─── 10:25 (no message for 10min)
                                                            ↓
                                                      Mode: INTERACTIVE → AUTONOMOUS
                                                      Model: little-qwen → Qwen3.6-27B
```

**Transition rules:**
- `interactive → autonomous`: After `autonomous_threshold` (default: 10 min) of no user messages, *if the agent has pending work*. If the session is idle (no pending work), no transition needed.
- `autonomous → interactive`: **Immediately** on next user message. No delay — the user's back, swap to the fast model.

The asymmetry is intentional: going autonomous is cautious (wait, confirm pattern), going interactive is instant (user is literally waiting right now).

#### Config Shape

```yaml
model:
  routing:
    interaction_mode:
      enabled: true
      autonomous_threshold: 600         # seconds without user message → autonomous
      unattended_turn_count: 5          # N agent turns without user → autonomous
      transition_delay: 0               # seconds to wait before switching (0 = immediate)
      
      # Model overrides per mode (override the primary model selection)
      interactive:
        model: qwen3-coder-30b-moe      # fast MoE, 139 tok/s
        provider: custom
      autonomous:
        model: qwen3.6-27b              # dense, 6 tok/s but smarter
        provider: custom
      
      # Platform-based overrides (skip heuristics entirely)
      platform_overrides:
        cron: autonomous
        delegate_task: autonomous
```

#### Interaction with Existing Router

The interaction mode is evaluated **before** the complexity classifier:

```
User message → InteractionModeDetector.classify()
                        │
                        ├── INTERACTIVE → use interactive lower model pool
                        │                    └── TurnRouter.classify() → ROUTINE or COMPLEX
                        │
                        └── AUTONOMOUS → use autonomous lower model pool  
                                         └── TurnRouter.classify() → ROUTINE or COMPLEX
```

This means the complexity router's behavior is unchanged — it still escalates to the upper model for hard tasks. The interaction mode only affects which *lower* model handles routine work.

#### Oversight Frequency Adaptation

Autonomous mode also adjusts oversight parameters:

| Parameter | Interactive | Autonomous | Rationale |
|-----------|-------------|------------|-----------|
| `oversight.every_n_turns` | 10 | 5 | Tighter review when unattended |
| `oversight.max_reviews_per_session` | 5 | 20 | Longer sessions need more |
| `oversight.actions.flag` | `notify_user` | `inject_and_continue` | User not watching; can't wait for them |
| `compression.safety_margin` | 1.3 | 1.5 | More conservative when nobody monitors |

When autonomous, the oversight model takes on greater responsibility — it can't flag and wait for a user who's asleep. Instead, it must decide: correct, escalate, or continue.

#### User-Initiated Mode Override

Sometimes the user knows they're about to step away:

```
User: "I'm heading out — keep working on this overnight"
       └── Explicit signal: set mode = AUTONOMOUS immediately
       └── Also loads autonomous-overnight-work skill behaviors
```

Or the inverse:

```
User: "I'm back, let me see what you've done"  
       └── Implicit signal: any user message → INTERACTIVE (default)
```

TUI command for explicit control:

```
/mode autonomous    # Switch now (for deep non-interactive work while present)
/mode interactive   # Switch back
/mode auto          # Resume automatic detection (default)
```

#### Why This Matters for the Capability Graph

In the v2+ expanded graph, interaction mode affects *which specialist* at each tier is selected:

```
                    ┌─────────────┐
                    │   Opus 4.6  │  (always upper — mode doesn't affect upper selection)
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
     INTERACTIVE      AUTONOMOUS     AUTONOMOUS
     (routine)        (routine)      (complex)
            │              │              │
     ┌──────┴──────┐ ┌────┴─────┐       │
     │  little-qwen│ │Qwen3.6   │       │
     │  MoE 139t/s│ │27B 6t/s  │       Opus
     └─────────────┘ └──────────┘
```

The interaction mode is not a property of a model — it's a property of the *session*. The model selection responds to it, but the models themselves are mode-agnostic.

---

### 2. Failure Escalation (Enhanced)

Extends the existing `_try_activate_fallback()` mechanism with smarter triggers.

#### Current: Binary failure detection
- API timeout / 5xx / auth failure → activate fallback

#### Enhanced: Quality-signal detection
- **Malformed tool calls** — JSON parse failures, wrong parameter names
- **Empty/nonsense responses** — model returns empty content or repetitive text
- **Tool call loops** — same tool called 3+ times with identical args
- **Context amnesia** — model asks about something clearly stated in recent context
- **Stuck pattern** — no meaningful progress across N turns (measured by: no new file writes, no test results changing, no git diff growth)

```python
class QualitySignalDetector:
    """Post-inference check: did the response indicate struggle?"""
    
    def evaluate(self, response: NormalizedResponse, context: SessionContext) -> EscalationSignal:
        signals = []
        
        if response.has_malformed_tool_calls:
            signals.append(Signal.MALFORMED_TOOLS)
        if self._is_repetitive(response, context.recent_responses):
            signals.append(Signal.REPETITION)
        if self._is_empty_or_hedging(response):
            signals.append(Signal.LOW_CONFIDENCE)
        if context.same_tool_repeated >= 3:
            signals.append(Signal.TOOL_LOOP)
        
        if len(signals) >= self.config.signal_threshold:
            return EscalationSignal.ESCALATE
        return EscalationSignal.OK
```

#### Escalation behavior

When escalation fires:
1. **Discard** the local model's response (don't show to user)
2. **Replay** the turn to the escalation model with the same context
3. **Log** the escalation reason (for tuning thresholds)
4. **Optional:** inject a note to the escalation model: "The previous model struggled with this. Signals: [X, Y]. Please handle carefully."

---

### 3. Periodic Oversight

The most novel component. A stronger model periodically reviews the session and can intervene.

#### Architecture

Follows the `background_review.py` pattern: fork the agent on a daemon thread with a specialized prompt.

```
Every N turns:
  oversight_model.review(last_N_messages) → OversightAction
    → approve (silent, no cost beyond the review call)
    → correct (inject guidance into next turn)
    → escalate (oversight model takes over next turn)
    → flag (surface warning to user)
```

#### Oversight Prompt

```python
OVERSIGHT_PROMPT = """You are reviewing the last {window} turns of an AI agent session.

Working model: {primary_model}
You are: {oversight_model}

The agent is performing tasks autonomously. Review for:

1. **Logical errors** — wrong assumptions, hallucinated file contents or APIs
2. **Missed context** — information clearly available that the agent ignored
3. **Circular work** — repeating failed approaches without changing strategy
4. **Architecture mistakes** — technically works but wrong design direction
5. **Scope drift** — doing work the user didn't ask for, or missing the actual request
6. **Silent failures** — tool calls that returned errors the agent didn't notice

Respond with exactly one JSON action:

{"action": "approve"} — work is correct, no intervention needed
{"action": "correct", "note": "..."} — inject this guidance for the next turn
{"action": "escalate", "reason": "..."} — you should handle the next turn directly
{"action": "flag", "warning": "..."} — alert the user about a concern
"""
```

#### Oversight Injection

When the oversight model returns `"correct"`:

```python
# Injected as a system message before the next user message
correction_msg = {
    "role": "system",
    "content": f"[OVERSIGHT NOTE from {oversight_model}]: {note}\n"
               f"Adjust your approach accordingly."
}
messages.insert(-1, correction_msg)  # Before latest user message
```

When it returns `"escalate"`:
- The oversight model itself handles the next turn
- After that turn, routing returns to normal (re-evaluated per turn)
- The escalation response includes the oversight model's reasoning as context

#### Cost Control

```yaml
    oversight:
      enabled: true
      model: us.anthropic.claude-opus-4-6-v1
      provider: bedrock
      every_n_turns: 10                 # review frequency
      review_window: 10                 # turns included in review
      max_reviews_per_session: 5        # budget cap
      min_turns_before_first: 5         # don't review trivially short sessions
      skip_if_escalated: true           # don't review turns the oversight model already handled
      actions:
        approve: silent
        correct: inject                 # inject into next turn context
        escalate: takeover              # oversight model handles next turn
        flag: notify_user               # surface in UI
```

#### Estimated cost per oversight call:
- Input: ~10 turns × ~1000 tokens/turn = ~10K tokens (~$0.15 on Opus)
- Output: ~100 tokens ($0.01)
- Per session (5 reviews max): **$0.80 worst case**
- vs. running Opus for every turn: **$15-50+ per session**

---

## Implementation Plan

### Phase 1: Turn Router (foundation)

1. Create `agent/model_router.py` — `TurnRouter` class with heuristic scoring
2. Create `agent/routing_config.py` — config schema, defaults, validation
3. Hook into `AIAgent.run_conversation()` — call router before model selection
4. Add `routing:` section to config schema with validation
5. Add TUI status bar indicator (which model is active this turn)
6. Tests: unit tests for classifier, integration tests for routing flow

### Phase 2: Enhanced Failure Escalation

1. Create `agent/quality_signals.py` — `QualitySignalDetector` class
2. Hook after response normalization, before returning to user
3. On escalation: discard response, replay with escalation model
4. Add escalation logging (for threshold tuning)
5. Tests: mock responses with various failure patterns

### Phase 2.5: `ask_upper` Tool

1. Create `agent/routing/ask_upper.py` — tool implementation following `auxiliary_client` pattern
2. Synchronous call to upper model with structured request types (guidance, review, distill)
3. Soft budget with warning after N calls per session (default 5)
4. Token cap on context parameter (~4K tokens max)
5. Graceful degradation when upper model unreachable ("proceed with best judgment")
6. Register tool dynamically only when routing is enabled and model is in lower position
7. Tests: mock upper model responses, budget enforcement, unreachable fallback

### Phase 3: Periodic Oversight

1. Create `agent/oversight.py` — synchronous reviewer (NOT the async `background_review.py` pattern; runs between turns, blocks until complete per RD-5)
2. Oversight prompt + action parsing
3. Hook into turn counter in conversation loop (synchronous pause every N turns)
4. Injection mechanism for corrections (insert before next lower-model turn)
5. Dynamic review window cap: `min(config.review_window, upper_ctx * 0.6 / avg_tokens_per_turn)` to prevent upper model context overflow (see RD-18)
6. Escalation handoff (oversight model takes over for one turn)
7. User notification for flags
8. TUI indicator when oversight is active
9. Full calibration pass every M checkpoints (re-read raw session from SQLite, fresh summary from ground truth per RD-10)
10. Tests: mock oversight responses, verify injection behavior, window cap math

### Phase 4: Observability & Tuning

1. Routing decision log (which model, why, per turn)
2. Oversight action log (approve/correct/escalate/flag history)
3. `/routing` slash command — show current routing state and recent decisions
4. `/oversight` slash command — show oversight history this session
5. Cost comparison dashboard (estimated cost with routing vs. without)

---

### 4. The `ask_upper` Tool (Lower Model Agency)

The lower model has a tool that lets it proactively request help from its upper model. This is NOT escalation (full handoff) — it's a synchronous query that returns guidance, and the lower model continues working.

#### Tool Schema

```python
{
    "name": "ask_upper",
    "description": "Ask the upper (more capable) model for help with the current task. "
                   "Use when you need: a complex problem simplified, a plan or checklist "
                   "for a multi-step task, verification of your approach, or context "
                   "distilled from a large conversation. The upper model will provide "
                   "guidance and you continue working.",
    "parameters": {
        "type": "object",
        "properties": {
            "request_type": {
                "type": "string",
                "enum": ["simplify", "plan", "verify", "distill", "explain"],
                "description": "What kind of help you need"
            },
            "question": {
                "type": "string",
                "description": "Your specific question or request for the upper model"
            },
            "context": {
                "type": "string",
                "description": "Relevant context the upper model needs to help you (current state, what you've tried)"
            }
        },
        "required": ["request_type", "question"]
    }
}
```

#### Request Types

| Type | Lower model says... | Upper model provides... |
|------|-------------------|------------------------|
| `simplify` | "This problem is too complex for me to reason about" | Broken-down steps, simplified framing |
| `plan` | "I need a strategy for this multi-step task" | Numbered checklist, decision tree |
| `verify` | "Here's my plan — is this right before I execute?" | Approval, corrections, warnings |
| `distill` | "There's too much context — what matters for this task?" | Key facts, relevant decisions, filtered state |
| `explain` | "I don't understand why X is happening" | Explanation, mental model, analogies |

#### Implementation

```python
class AskUpperTool:
    """Tool that lets the lower model query the upper model for guidance."""
    
    def execute(self, request_type: str, question: str, context: str = "") -> str:
        # Build a prompt for the upper model
        prompt = self._build_mentor_prompt(request_type, question, context)
        
        # Call upper model synchronously (like auxiliary_client pattern)
        response = self.upper_client.chat.completions.create(
            model=self.upper_model,
            messages=[
                {"role": "system", "content": MENTOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000  # Guidance should be concise
        )
        
        # Track budget
        self.ask_upper_calls += 1
        if self.ask_upper_calls >= self.soft_budget:
            return response.content + "\n\n[NOTE: You've used ask_upper frequently this session. Consider whether you can proceed independently.]"
        
        return response.content
```

#### System Prompt for Upper Model (when acting as mentor)

```
You are a senior mentor assisting a capable but less powerful AI model.
The model asking you for help is working on a task autonomously. It needs
your guidance, not your takeover. Provide:

- Clear, actionable guidance
- Concrete steps when asked for plans
- Direct answers when asked to verify
- Distilled key facts when asked to simplify context

Keep responses focused and concise. The model will continue working after
receiving your response — give it what it needs to succeed independently.
```

---

### 5. Context Compression on Handback

At every oversight checkpoint, the upper model produces a compressed context package regardless of oversight action. This is the mechanism that gives the lower model effectively unlimited session length.

#### Core Insight: Adaptive Compression Targeting

The upper model doesn't just compress — it **targets a specific output size** based on observed session dynamics. Because the upper model has full visibility into the session history, it knows:

1. **How much context the lower model uses per turn** (observed average over the session)
2. **How many turns until the next checkpoint** (configured interval)
3. **The lower model's total context budget** (from capability graph)
4. **The current system prompt + tool definitions overhead** (relatively stable)

This means the upper model can calculate:

```
headroom_needed = avg_tokens_per_turn × turns_until_next_checkpoint × safety_margin
max_summary_size = lower_model_context_budget - system_prompt_size - headroom_needed - recent_raw_turns_size
```

If the session has been using 3K tokens per turn and the next checkpoint is 10 turns away:
- `headroom_needed = 3000 × 10 × 1.3 = 39,000 tokens`
- For a 256K context model: `max_summary_size = 256K - 8K(sys) - 39K(headroom) - 6K(recent) = ~203K`

That's generous. But if the session involves lots of code dumps and the per-turn average is 15K:
- `headroom_needed = 15000 × 10 × 1.3 = 195,000 tokens`
- `max_summary_size = 256K - 8K - 195K - 6K = ~47K`

The compression adapts dynamically. Early in a session with small turns, the summary can be expansive. In a code-heavy session with huge tool outputs, it compresses more aggressively.

#### Differential Compression

Rather than replacing the entire summary at each checkpoint, the upper model performs differential updates:

```
Previous summary (from checkpoint N-1)
    + New decisions made (turns N-1 to N)
    + Completed work (mark done, collapse detail)
    + Active threads (keep full detail)
    − Dead ends (remove entirely)
    − Superseded context (old state replaced by new)
    = New summary (for checkpoint N)
```

This means the summary **grows slowly and organically** rather than being rewritten from scratch. Key benefits:
- Stable facts don't get lost to summarization drift
- The lower model sees consistent framing across checkpoints (less disorienting)
- Cheaper for the upper model (it only needs to reason about the delta, not re-summarize everything)

#### Compression Architecture

```
                    Oversight Checkpoint
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
┌─────────┐       ┌──────────────┐      ┌────────────────┐
│ Review  │       │  Compress    │      │   Handback     │
│ (action)│       │  (context)   │      │   (package)    │
└─────────┘       └──────────────┘      └────────────────┘
    │                      │                      │
    │ approve/correct/     │ differential         │ structured
    │ escalate/flag        │ summary + target     │ prompt for
    │                      │ size from budget     │ lower model
    └──────────────────────┴──────────────────────┘
                           │
                           ▼
                  Lower model continues
                  with fresh context
```

#### The Compression Prompt (to upper model)

```
You are compressing a conversation for handoff to a less capable model.
It needs to continue working seamlessly.

## Compression Parameters
- Target summary size: {max_summary_tokens} tokens (HARD LIMIT — do not exceed)
- Previous summary (update differentially): {previous_summary}
- New turns to incorporate: {recent_turns}
- Average context per turn this session: {avg_tokens_per_turn}
- Turns until next checkpoint: {turns_until_checkpoint}

## Instructions
Update the previous summary differentially:
1. ADD new decisions, discoveries, and state changes from the recent turns
2. MARK COMPLETE any work that finished — collapse to one-line acknowledgment
3. KEEP FULL DETAIL on active threads and in-progress work
4. REMOVE dead ends, abandoned approaches, and superseded state
5. TRIM older completed items if approaching the token limit

## Output Structure

### Session State
- What the user originally asked for
- What has been accomplished so far (brief)
- What is currently in progress (detailed)
- What remains to be done

### Key Decisions Made
- [Decision]: [rationale] (keep all — these are never trimmed)

### Active Context
- Files being worked on and their current state
- Environment state (running processes, git branch, etc.)
- Constraints or requirements discovered during work

### Guidance for Next Steps
- What the model should do next
- Pitfalls to avoid (learned from this session)
- Approach recommendations

Prioritize actionability over completeness.
The model receiving this has tools — it can look things up. Focus on
what it can't rediscover: decisions, rationale, and current state.
```

#### Context Rebuild After Compression

```python
def rebuild_context_after_oversight(self, compressed_summary: str, recent_turns: int = 3):
    """Replace conversation history with compressed summary + recent turns."""
    
    # Keep system prompt unchanged
    system_msg = self.messages[0]
    
    # Keep only the last N raw turns (user + assistant pairs)
    recent = self._get_last_n_turns(recent_turns)
    
    # Build the compressed context message
    context_msg = {
        "role": "system",  # or "user" with [CONTEXT] framing
        "content": (
            "[SESSION CONTEXT — compressed by oversight model]\n\n"
            f"{compressed_summary}\n\n"
            "[END SESSION CONTEXT — the most recent messages follow]\n"
        )
    }
    
    # Rebuild: system + compressed context + recent raw turns
    self.messages = [system_msg, context_msg] + recent
    
    # Log the compression event
    logger.info(f"Context compressed: {pre_compression_tokens} → {post_compression_tokens} tokens "
                f"({compression_ratio:.1f}x reduction)")
```

#### Adaptive Budget Calculation

```python
class CompressionBudget:
    """Calculates target summary size based on session dynamics."""
    
    def __init__(self, lower_model_context_budget: int, system_prompt_tokens: int,
                 checkpoint_interval: int, safety_margin: float = 1.3):
        self.context_budget = lower_model_context_budget
        self.system_tokens = system_prompt_tokens
        self.checkpoint_interval = checkpoint_interval
        self.safety_margin = safety_margin
        self.turn_token_history: List[int] = []
    
    def record_turn(self, tokens_used: int):
        """Track how many tokens each turn adds to context."""
        self.turn_token_history.append(tokens_used)
    
    @property
    def avg_tokens_per_turn(self) -> int:
        if not self.turn_token_history:
            return 3000  # conservative default
        # Use recent window (last 20 turns) weighted toward recent
        recent = self.turn_token_history[-20:]
        return int(sum(recent) / len(recent))
    
    @property
    def p90_tokens_per_turn(self) -> int:
        """90th percentile — accounts for spiky turns (large code dumps)."""
        if len(self.turn_token_history) < 5:
            return self.avg_tokens_per_turn * 2
        recent = sorted(self.turn_token_history[-20:])
        idx = int(len(recent) * 0.9)
        return recent[idx]
    
    def max_summary_tokens(self, recent_raw_tokens: int = 6000) -> int:
        """Calculate how large the summary can be while leaving headroom."""
        headroom = self.p90_tokens_per_turn * self.checkpoint_interval * self.safety_margin
        available = self.context_budget - self.system_tokens - headroom - recent_raw_tokens
        # Floor: summary should be at least 2K tokens to be useful
        # Ceiling: summaries over 30K start degrading lower model attention
        return max(2000, min(30000, int(available)))
```

#### Emergency Compression Triggers

Beyond scheduled checkpoints, compression can fire early:

1. **Context budget alert:** lower model's context usage exceeds 70% of budget → trigger early checkpoint
2. **`ask_upper` with type=distill:** lower model explicitly requests compression
3. **Router context_size trigger:** router sees context growing too large → triggers compression before routing decision
4. **Turn token spike:** single turn adds > 3× the average (huge code dump) → consider early compression

```python
def should_emergency_compress(self) -> bool:
    """Check if we need an unscheduled compression."""
    current_usage = self.current_context_tokens / self.context_budget
    if current_usage > 0.70:
        return True
    if self.last_turn_tokens > self.p90_tokens_per_turn * 3:
        return True
    return False
```

#### The Closed Loop: Self-Tuning Compression

The compression system forms a closed feedback loop that self-tunes over the session:

```
Upper model observes session dynamics
         │
         ▼
Calculates: "this session uses ~8K/turn on average"
         │
         ▼
At checkpoint: compresses to leave exactly enough headroom
         │
         ▼
Lower model works for N turns (uses the headroom)
         │
         ▼
Upper model observes how much was actually used
         │
         ▼
Next checkpoint: adjusts compression target accordingly
         │
         └──── (repeat — the system self-tunes)
```

The upper model isn't just compressing — it's **managing a resource budget for a subordinate agent**, adapting in real-time to how that agent actually works. If the lower model starts doing more code-heavy work (reading large files, reviewing diffs), the upper model automatically tightens the summary at the next checkpoint. If the session shifts to lightweight Q&A, it relaxes and preserves more detail.

**Critical escalation signal:** If the adaptive budget calculation yields `max_summary_tokens < 2000` (can't compress enough to leave room), the lower model's context window is fundamentally too small for what's happening. This triggers automatic escalation — the upper model stays active for the remainder of the session, or until the workload lightens enough that handback becomes viable again.

This creates a natural "complexity ceiling" detection: the system doesn't just route based on message content, it routes based on **observed resource consumption patterns**. A session that starts simple but gradually becomes resource-intensive will organically transition to the upper model without any explicit trigger.

#### Compression Budget

Each compression costs:
- Input: ~10 turns × ~1000 tok/turn = ~10K tokens (+ previous summary ~3K)
- Output: ~2-3K tokens (the differential update)
- Cost: ~$0.20 per checkpoint on Opus

Combined with the oversight review (which reads the same context), total per-checkpoint cost: **~$0.20** (the context is read once for both review + compression).

---

## Alternatives Considered

### A. Router model (dedicated small LLM for classification)
- **Rejected:** Adds latency (200-500ms per turn), requires another model running, overkill for single-user. Heuristics are sufficient when the user has explicit triggers as a safety net.

### B. Always-escalate pattern (local tries, cloud always re-does)
- **Rejected:** Doubles cost and latency for every turn. Defeats the purpose of local-primary.

### C. User-only manual switching
- **Rejected:** Requires the user to know when they need better reasoning. Oversight catches cases they wouldn't notice (subtle logical errors, scope drift).

### D. Mixture of Agents (MoA) — run both, pick best
- **Rejected:** Doubles inference cost. Requires a judge model (more cost). Interesting for evaluation but not for daily driving.

### E. Token-level routing (different models for different parts of response)
- **Rejected:** Not supported by any current inference infrastructure. Speculative.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Router misclassifies complex turns as routine | Local model struggles, wastes turns | Failure escalation catches it post-hoc; oversight catches it periodically |
| Router over-escalates (everything goes to cloud) | Cost savings disappear | Hysteresis + tunable thresholds + explicit `strategy: always-primary` override |
| Oversight corrections confuse the primary model | Degraded responses after correction injection | Correction framing tested empirically; fallback: flag user instead of injecting |
| Oversight adds latency | User waits for oversight before next turn | Oversight runs synchronously but is fast (~3-5s for 10 turns review on Opus) |
| Local model goes down mid-session | Session breaks | Existing fallback mechanism handles this — escalation model becomes sole provider |
| Config complexity | Users confused by routing config | Sensible defaults + `hermes setup` wizard step + `hermes doctor` validates routing config |
| Compression loses critical context | Lower model confused after handback | Differential compression preserves key decisions; lower model has `ask_upper(type=distill)` as escape hatch; oversight catches confusion at next checkpoint |
| `ask_upper` cost explosion | Lower model over-relies on upper | Soft budget with warning after N calls; upper model's mentor prompt encourages independence |
| Adaptive budget miscalculates | Summary too large (no headroom) or too small (lost context) | p90 with 1.3× safety margin is conservative; emergency compression at 70% as backstop |
| Mode transition mid-work | Model switches from fast MoE to dense (or vice versa) mid-task | Transitions only fire at turn boundaries; in-progress inference completes on current model. Handoff context ensures continuity. |
| False autonomous detection | Agent misreads a brief user pause as "gone for the night" | 10-min threshold is conservative; requires pending work; instant snap-back on user message. Tunable + `/mode` override. |
| Dense model quality assumption | Qwen3.6-27B may not reason better than MoE for all task types | Validate empirically in Phase 1 benchmarking. Architecture is model-agnostic — swap autonomous model if delta is negligible. |

---

## Target Model Lineup (v1 Validation)

The initial implementation targets a four-node capability graph with measured benchmarks from M5 Max 128 GB hardware. **All model assignments are placeholders.** The routing architecture is model-agnostic — any model can occupy any position in the graph. A separate **model evaluation and placement system** (see RD-14) determines which models fill which roles based on measured performance, quality, cost, and hardware constraints. The lineup below is the v1 validation target, not a permanent assignment.

### Interactive Lower Model: Qwen3-Coder-Next (Q4_K_M)

| Property | Value |
|----------|-------|
| Architecture | 80B MoE, 3B active parameters per token |
| Quantization | Q4_K_M (45 GB on disk) |
| Context window | 262,144 tokens (native, no RoPE hacks) |
| KV cache | 2 heads × 128 dim × 80 layers = tiny; Q8_0 at 262K ≈ 6 GB |
| Total RAM | ~61 GB (weights + KV + overhead) |
| **Measured speed** | **Prompt: 376–720 tok/s, Generation: 32–38 tok/s** |
| Strengths | Code generation, tool calling, instruction following, deep expert routing |
| Weaknesses | Complex multi-step reasoning, architecture decisions, long-horizon planning |
| llama-server flags | `--flash-attn --cache-type-k q8_0 --cache-type-v q8_0 -c 262144 -ngl 99` |
| Provider config | `custom` provider, `http://127.0.0.1:58080/v1`, `sk-local` |
| **Role** | Primary interactive workhorse. Handles routine work when user is present. |

### Autonomous Lower Model: Qwen3.6-27B-UD (Q4_K_XL)

| Property | Value |
|----------|-------|
| Architecture | 27B dense, hybrid attention (16/64 layers full KV) |
| Quantization | Q4_K_XL (16 GB on disk) |
| Context window | 262,144 tokens |
| Total RAM | ~25 GB (weights + KV + overhead) |
| **Measured speed** | **Prompt: 148–199 tok/s, Generation: 6.3–6.9 tok/s** |
| Strengths | Dense all-parameter reasoning, prose quality, nuanced understanding |
| Weaknesses | Too slow for interactive use (6 tok/s feels glacial) |
| llama-server flags | `--flash-attn --cache-type-k q8_0 --cache-type-v q8_0 -c 262144 -ngl 99` |
| Provider config | `custom` provider, `http://127.0.0.1:58080/v1`, `sk-local` |
| **Role** | Overnight/cron/subagent workhorse. Higher quality reasoning when latency is invisible. |

### Fast Fallback Model: Qwen3-Coder-30B-A3B (Q4_K_M)

| Property | Value |
|----------|-------|
| Architecture | 30B MoE, 3B active parameters per token |
| Quantization | Q4_K_M (17 GB on disk) |
| Context window | 131,072 tokens |
| Total RAM | ~22 GB (weights + KV + overhead) |
| **Measured speed** | **Generation: ~139 tok/s** |
| Strengths | Blazing fast, proven reliability, small footprint |
| Weaknesses | Smaller expert pool than Coder-Next, shorter context window |
| Provider config | `custom` provider, `http://127.0.0.1:58080/v1`, `sk-local` |
| **Role** | De-escalation target. When even Coder-Next is more model than needed. |

### Upper Model: Claude Opus 4.6

| Property | Value |
|----------|-------|
| Architecture | Frontier cloud model |
| Context window | 200,000 tokens (native) |
| Cost | ~$15/M input, ~$75/M output (Bedrock) |
| Strengths | Complex reasoning, architecture, planning, summarization, oversight |
| Weaknesses | Cost, latency (~5-15s for complex turns) |
| Provider config | `bedrock`, `us.anthropic.claude-opus-4-6-v1` |
| **Role** | Escalation target, oversight, compression, mentor (ask_upper). |

### Capability Graph (v1) — Bidirectional

```
┌──────────────────────────────────────────────────────────────┐
│  Claude Opus 4.6 (upper)                                     │
│  200K ctx │ $15/$75 per M │ 5-15s                            │
│  Roles: oversight, mentor, escalation, compression, handback │
└────────────────────────────┬─────────────────────────────────┘
                             │ upper_of (escalation ↑)
                             │
┌────────────────────────────┴─────────────────────────────────┐
│  Qwen3-Coder-Next 80B MoE (interactive lower)               │
│  262K ctx │ $0 │ 33 tok/s                                    │
│  Roles: primary interactive workhorse, tool: ask_upper       │
├──────────────────────────────────────────────────────────────┤
│      ↕ mode switch (interaction mode detector)               │
├──────────────────────────────────────────────────────────────┤
│  Qwen3.6-27B dense (autonomous lower)                       │
│  262K ctx │ $0 │ 6.5 tok/s                                   │
│  Roles: overnight/cron workhorse, tool: ask_upper            │
└────────────────────────────┬─────────────────────────────────┘
                             │ lower_of (de-escalation ↓)
                             │
┌────────────────────────────┴─────────────────────────────────┐
│  little-qwen 30B MoE (fast fallback)                         │
│  131K ctx │ $0 │ 139 tok/s                                   │
│  Roles: rapid-fire, trivial tasks, speed-critical bursts     │
└──────────────────────────────────────────────────────────────┘
```

### De-escalation: The Downward Edge

Escalation moves **up** the graph when work is too complex for the current model. De-escalation is the inverse — moving **down** when the current model is more capability than needed and speed becomes the bottleneck.

**When de-escalation fires:**
- Rapid-fire trivial exchanges (user sending many short messages in quick succession)
- Simple acknowledgments, confirmations, status checks
- Tool-only turns (the model just needs to call `read_file` or `terminal`, no reasoning)
- The primary model is loaded/slow (e.g., long KV cache, high context) and a lighter model could respond instantly

**The symmetry:**

| Direction | Trigger | Trades | Example |
|-----------|---------|--------|---------|
| **Escalate ↑** | Complexity exceeds capability | Speed/cost → quality | "Debug this race condition" → Opus |
| **De-escalate ↓** | Task is trivially below capability | Quality → speed | "What branch am I on?" → little-qwen |

De-escalation is conservative by default — most routine turns stay on the primary lower model (Coder-Next). It only drops to the fast fallback when speed is critical *and* the task is genuinely trivial. The fast fallback never has `ask_upper` access (it's too light to be a mentor relationship) and doesn't receive oversight (too transient to be worth reviewing).

**Config:**

```yaml
model:
  routing:
    de_escalation:
      enabled: true
      model: qwen3-coder-30b-moe         # fast fallback
      provider: custom
      triggers:
        message_length_below: 50          # very short user messages
        rapid_fire_threshold: 3           # N messages within rapid_fire_window
        rapid_fire_window: 30             # seconds
        tool_only_pattern: true           # turns that are pure tool dispatch
      # De-escalation is never sticky — re-evaluates every turn
      sticky: false
```

### Model Swap Mechanics

Only one local model runs at a time on `:58080`. Mode/model transitions require a server swap:

| Transition | Trigger | Server Action | Latency |
|------------|---------|---------------|---------|
| interactive → autonomous | User idle 10 min | `llm start prose` (Qwen3.6-27B) | ~10-15s |
| autonomous → interactive | User message arrives | `llm start coder-next` | ~17s |
| primary → fast fallback | De-escalation fires | `llm start little-qwen` | ~8s |
| fast fallback → primary | Non-trivial message | `llm start coder-next` | ~17s |

The swap latency (8-17s) is acceptable for mode transitions because:
- **interactive → autonomous:** user is already gone, no one is waiting
- **autonomous → interactive:** symmetric delay applies (see RD-16) — don't swap eagerly on a single message
- **primary ↔ fast fallback:** these are rare edge cases; the 8s swap for little-qwen is fast enough

### RD-16: Symmetric lazy swap-back + cloud-orchestrated transitions
**Resolved 2026-06-06:** Model swap timing is symmetric in both directions. Just as we don't eagerly switch to the autonomous model when the user goes idle (10-min threshold), we don't eagerly switch back to the interactive model when the user returns. The slower model handles the first message(s) — a little slower, but no wasted swap.

**Lazy swap-back triggers:**
- User sends 2-3 messages within a short window (sustained engagement detected)
- User explicitly requests faster responses
- Turn router scores a turn as complex enough to warrant the interactive model's speed

If the user types one message and leaves again, no swap occurs — the autonomous model handles it at 6.5 tok/s. Slower, but avoids a 17s swap that would have been immediately reversed.

**Cloud-orchestrated swap:** When the swap *does* trigger, the cloud model (upper) serves a dual role:

1. **Operational:** It orchestrates the mechanical transition — unloads model A, loads model B, verifies readiness, confirms handoff. This is *useful work*, not wasted tokens.
2. **Conversational:** It may respond to the user's current turn as a side effect, but the primary purpose is supervision of the swap.

This means the cloud turn during a transition is a **micro-oversight action**, not gap-filling. The cloud model earns its tokens by:
- Deciding whether the swap is warranted (confirms the router's signal)
- Managing the swap mechanics (issues `llm start <model>`, waits for health check)
- Verifying the new model is ready before handing off
- Optionally continuing the conversation during the swap window

**State machine:**
```
USER_RETURNS → [slow model handles turn 1]
              → [user sends turn 2 within threshold]
              → SWAP_TRIGGERED
              → cloud model takes turn 2, begins swap in background
              → [swap completes, health check passes]
              → SWAP_COMPLETE → local interactive model handles turn 3+
```

**Failure handling:**
- If local model fails to load within 30s: cloud model stays active, warns user, logs diagnostic
- If user stops interacting during swap: abort swap, stay on current model (the trigger was wrong)

**Dependency on model profiling (RD-14):** The swap orchestrator relies on profiled startup latency to make informed decisions:
- **Swap budget calculation:** If profiled startup for Coder-Next is 17s, the orchestrator knows to budget ~20s (with margin) before declaring failure
- **Out-of-bounds detection:** If a swap that normally takes 17s is taking 40s, something is wrong (disk pressure, RAM contention, corrupt model file). The orchestrator can abort early and stay on cloud rather than waiting blindly.
- **Graph placement constraint:** A model with 45s startup latency is unsuitable for the "interactive lower" role even if its tok/s is excellent — the swap cost makes mode transitions too painful. The placement system accounts for this.
- **Cloud TTFT monitoring:** If the cloud model's observed TTFT exceeds its profiled p90 (e.g., Bedrock is having a bad day), the orchestrator may decide the "slow local model" is actually faster right now and skip the cloud-orchestrated swap entirely.

For v2, preloading two models simultaneously (if RAM permits) eliminates swap latency entirely. With 128 GB: Coder-Next (61 GB) + little-qwen (22 GB) = 83 GB leaves 45 GB for system — tight but feasible on separate ports.

### RD-17: Cost-aware routing filter (anti-thrashing)
**Resolved 2026-06-06:** The routing system uses a two-dimensional cost model as a secondary gate after complexity/mode classification. The router's recommendation is an *ideal* — the cost filter decides whether acting on it is *worth it* given the economics of the transition.

**Two cost axes per graph position:**

1. **Per-turn cost** — the marginal cost of a single inference call on this model.
   - Local model (running): 0 (compute is already paid for via hardware)
   - Local model (not running): 0 per-turn, but see transition cost
   - Cloud model: f(input_tokens, output_tokens, price_per_1k) — configured per position

2. **Transition cost** — the cost of *getting to* this model from the current state.
   - Cloud model: 0 (always available, no startup)
   - Local model already loaded on :58080: 0 (just point the agent at it)
   - Local model NOT loaded: HIGH (8-17s wall clock, RAM churn, user waiting)
   - Conceptually a scalar that combines time-cost and disruption-cost

**Decision logic:** The cost filter runs after the Turn Router selects a target position but before `execute_routing_swap()` commits to the transition:

```
routing_benefit = quality_delta(target) - quality_delta(current)
routing_cost = transition_cost(target) + per_turn_cost(target) × expected_turns
stay_cost = per_turn_cost(current) × expected_turns

if routing_benefit > (routing_cost - stay_cost):
    approve_swap()
else:
    stay_on_current()
```

The `expected_turns` estimate uses **consecutive-target-agreement** as a confidence multiplier: if the router has recommended the same target for N consecutive turns, confidence that we'll stay there goes up. First turn it says "escalate" → cost filter may say "wait one more turn." Third turn in a row → commit.

**Concrete scenarios:**

- **Scenario A: Escalate to Opus.** Per-turn cost = ~$0.03. Transition cost = 0 (cloud, always available). Current model is local (per-turn cost = 0). Question: is the complexity delta worth $0.03/turn? For genuinely hard tasks (complexity > 0.8), yes — approve immediately. For borderline messages (complexity 0.72), the cost filter may wait for a second consecutive high-complexity turn before committing.

- **Scenario B: De-escalate to little-qwen (fast fallback).** Per-turn cost = 0. Transition cost = HIGH (must `llm start little-qwen`, unloading coder-next — 12s disruption). Current model handles the message fine, just slightly overkill. Cost filter rejects: transition cost vastly exceeds the zero per-turn savings. Only approved if the router has been recommending de-escalation for 5+ consecutive turns (sustained trivial workload).

- **Scenario C: Return to interactive_lower (coder-next) from Opus.** Per-turn cost goes from ~$0.03 → 0. Transition cost = 0 (coder-next already loaded on :58080). Clear win — every turn we stay on Opus costs money, switching to local costs nothing. Cost filter approves immediately.

- **Scenario D: Complexity oscillation.** Messages alternate between 0.65 and 0.75 complexity (straddling the 0.7 threshold). Router flip-flops between interactive_lower and upper. Cost filter sees no consecutive-target-agreement — never reaches confidence threshold. Result: stays on current model, oscillation absorbed.

**Key properties:**

- **Local↔cloud transitions are cheap** — cloud is always available, local stays warm on :58080. These transitions are approved with minimal friction.
- **Local↔local transitions are expensive** — only one model fits on :58080. These require strong, sustained routing signals before approval.
- **Escalation bias** — the cost filter can have asymmetric thresholds: escalation (safety/quality) requires less confidence than de-escalation (optimization). Spending $0.03 on a turn that *might* need Opus is cheaper than a compounding error.
- **Natural debounce** — the consecutive-target-agreement requirement means single-turn spikes ("thanks" between complex requests) never trigger swaps. The cost filter absorbs transient noise without explicit cooldown timers.

**Config schema extension:**

```yaml
model:
  routing:
    graph:
      interactive_lower:
        provider: "custom:llm-local"
        model: "qwen3-coder-next"
        llm_config_name: "coder-next"
        cost:
          per_1k_input_tokens: 0
          per_1k_output_tokens: 0
          transition_cold: 100    # scalar: high cost to cold-start
          transition_warm: 0      # already loaded
      upper:
        provider: "bedrock"
        model: "us.anthropic.claude-opus-4-6-v1"
        cost:
          per_1k_input_tokens: 15
          per_1k_output_tokens: 75
          transition_cold: 0      # always available
          transition_warm: 0
      fast_fallback:
        provider: "custom:llm-local"
        model: "qwen3-coder-30b"
        llm_config_name: "little-qwen"
        cost:
          per_1k_input_tokens: 0
          per_1k_output_tokens: 0
          transition_cold: 100    # requires llm start (unloads current)
          transition_warm: 0
    cost_filter:
      enabled: true
      min_consecutive_agreement: 2    # turns router must agree before approving
      escalation_override: true       # bypass cost filter for upper (safety)
      cold_swap_min_agreement: 5      # local↔local needs strong signal
```

**Bypass conditions** — the cost filter is skipped when:
- Explicit user override (`/routing swap <position>`, `/mode autonomous`)
- `escalation_override: true` and target is upper (quality/safety trumps cost)
- Current model is unreachable (failover, not routing)

**Implementation phase:** Phase 5 (after the routing system is validated end-to-end with manual tuning). The cost filter adds optimization sophistication on top of a working system — it should not gate initial deployment.

### Future Graph Expansion (v2+)

```
         ┌─────────────┐
         │   Opus 4.6  │  (reasoning, architecture, oversight)
         └──────┬──────┘
                │ upper_of
         ┌──────┴──────┐
         │  Sonnet 4   │  (general purpose, fast cloud, mid-tier oversight)
         └──────┬──────┘
                │ upper_of
     ┌──────────┼──────────┐
     │                     │
┌────┴─────┐         ┌────┴─────┐
│  Coder   │         │ Qwen3.6  │  (local specialists)
│  Next    │◄───────►│   27B    │  (mode-switched, same port)
│  (code)  │         │  (prose) │
└────┬─────┘         └──────────┘
     │ lower_of
┌────┴─────┐
│  little- │  (fast fallback, de-escalation)
│  qwen    │
└──────────┘
```

In the expanded graph:
- Coder-Next handles code tasks, Qwen3.6-27B handles prose/research (mode-switched)
- Either can escalate to Sonnet (fast cloud, cheaper than Opus)
- Sonnet escalates to Opus only for the hardest problems
- little-qwen sits below as the speed-optimized de-escalation target
- Each edge has its own handoff protocol and compression parameters
- The router selects tier AND specialist AND speed class

---

## Success Criteria

1. **Cost reduction:** 60-80% fewer cloud API calls for typical coding sessions
2. **Quality maintained:** Oversight catches errors before they compound (< 3 turns of drift)
3. **Transparent:** User always knows which model is active (TUI indicator)
4. **Zero-config viable:** Works with just `routing.enabled: true` + primary/escalation models defined
5. **No routing latency penalty:** Router adds < 5ms per turn; oversight runs synchronously every N turns (~3-5s pause, amortized over the interval)

---

## Resolved Decisions

### RD-1: Oversight visibility
**Resolved 2026-06-06:** Oversight activity is always visible to the user. When active, the user sees which model is handling the current turn. Escalation/correction events are surfaced with a brief explanation.

### RD-2: Lower model awareness
**Resolved 2026-06-06:** The lower model knows it is being watched AND has the ability to proactively ask the upper model for help. This is exposed as a tool (`ask_upper`) available to the lower model. The lower model can request:
- Simplification of a complex problem
- Breakdowns, checklists, plans
- Verification of an approach before executing
- Context distillation ("too much history, what matters?")

This reframes the relationship as **mentor/mentee** rather than primary/fallback. The lower model has agency.

### RD-3: Routing stickiness
**Resolved 2026-06-06:** When the upper model takes over, it stays sticky for a minimum number of turns (default: 3). After the sticky period, the upper model is explicitly prompted to evaluate whether work can be handed back to the lower model. If yes, it constructs a **handback prompt** — a compressed context + guidance package tailored for the lower model to continue from.

### RD-4: Visibility of routing decisions
**Resolved 2026-06-06:** The user always knows which model is active (TUI status bar indicator). On escalation, a brief notification explains why (e.g., "↑ Escalated to Opus: complexity score 0.85"). Running totals of % use per model available via `/routing` command.

### RD-5: Oversight timing
**Resolved 2026-06-06:** Oversight runs synchronously between turns. A brief pause every N turns while the upper model reviews, gives the go-ahead, and optionally compresses context before the lower model continues. This guarantees corrections land before the next turn and enables the compression-on-handback pattern.

### RD-6: Build target
**Resolved 2026-06-06:** Build as a proper upstream feature from the beginning (PR-ready quality, tests, docs). If it never gets merged upstream, it's still available on the fork. Local experimentation first to validate the design, but written to upstream standards.

### RD-7: Context compression on handback
**Resolved 2026-06-06:** At every oversight checkpoint, even if the upper model approves ("all good"), it produces a **compressed context summary** before handing back to the lower model. This means the lower model's effective context window resets at each checkpoint — it never hits its 256K wall in any reasonable session. The upper model (with superior summarization ability and its own large context) acts as a periodic context compressor.

The compression flow:
1. Upper model reads recent N turns during oversight
2. Produces: `oversight_action` + `compressed_context`
3. Lower model's context is rebuilt: `[system prompt] + [compressed summary] + [recent 2-3 turns raw]`
4. Lower model continues with fresh context and guidance from upper

### RD-8: Capability graph (not just two models)
**Resolved 2026-06-06:** The architecture models a **directed graph of capabilities**, not a hardcoded pair. Each edge represents an upper/lower relationship. The simplest case is two nodes (local ↔ cloud), but the system supports N nodes with directional edges.

Each node in the graph has:
- Model identity (provider, model name, endpoint)
- Capability profile (what it's good at)
- Cost tier
- Context window size

Each edge carries:
- Direction (who is upper, who is lower)
- Handoff protocol (how context is transferred/compressed)
- Escalation triggers specific to this pair

For v1, we implement the two-node case. The data structures support N nodes from day one.

---

## Remaining Open Questions

1. **`ask_upper` tool budget:** Should the lower model have unlimited access to `ask_upper`, or should there be a per-session cap to prevent cost explosion?
   - Leaning: **token budget** (cumulative input+output tokens tracked against a dollar threshold, not call count)

2. **Handback prompt authoring:** Should the upper model's handback prompt be structured (template + fields) or freeform?
   - Leaning: structured template with sections (summary, active files, current task, guidance)

3. **Graph edge weights:** When there are multiple possible upper models, how does the lower model choose which to ask?
   - Leaning: for v1, there's only one upper. For vN, select based on capability match to the question type.

4. **Compression fidelity:** How do we verify the compression didn't lose critical context?
   - Leaning: Full calibration pass every 5th checkpoint re-reads raw history from disk. Plus: append-only "key decisions" section never trimmed.

5. **How does this interact with `delegate_task`?** Routing is brain-swap (same agent, different model); delegation is subagent dispatch (different agent, isolated context). These are orthogonal. Subagents use `delegation.model` as today. Routing may apply *within* a subagent if routing is enabled for subagent sessions — but routing is never *implemented as* delegation. See RD-15.

6. **Should the routing config be part of a profile?** So you can have `hermes -p local` (routing on) and `hermes -p opus` (always Opus)?
   - Leaning: yes, naturally — each profile has its own config.yaml

7. **30K summary ceiling:** Is this empirically validated for Qwen3-Coder-Next? Does it degrade with long system messages?
   - Needs testing: validate attention quality at various summary sizes during model benchmarking

8. **Prompt caching interaction:** Compression rebuilds `self.messages` every N turns, breaking any cached prefix. Is this cost-significant?
   - Likely acceptable: compression happens infrequently (every 10 turns), and the cost savings from routing overwhelm the cache miss penalty. But should be measured.

---

## Resolved Decisions (continued)

### RD-9: Fallback chain for upper model unreachable
**Resolved 2026-06-06:** Each level in the capability graph has a designated fallback chain when its upper model is unreachable:

1. Retry with exponential backoff (3 attempts)
2. Try alternate upper model if configured (e.g., Sonnet as fallback for Opus)
3. Skip checkpoint — continue with stale context, retry at next natural break
4. If context at emergency levels (>85%): naive truncation (drop oldest non-system messages)
5. If all else fails: halt and notify user

For `ask_upper` specifically: return a structured error ("Upper model unavailable, proceed with best judgment"). The lower model already handles failed tool calls gracefully.

The philosophy: graceful degradation first, halt only as last resort. The system should never silently lose data — either compress properly or stop and tell the user.

### RD-10: Full calibration to prevent compression drift
**Resolved 2026-06-06:** Every 5th oversight checkpoint (configurable), the upper model re-reads the full raw session history from the SQLite message store (not from the in-memory compressed summary) and produces a fresh summary from ground truth.

This prevents the "telephone game" failure where summaries-of-summaries drift from reality over many checkpoints. Most sessions' raw history fits within the upper model's 200K context window. For sessions exceeding 200K raw tokens: progressive summarization (chunk → summarize → combine).

### RD-11: Interaction mode as routing signal
**Resolved 2026-06-06:** The routing system has a second classification axis: interaction mode (interactive vs. autonomous). This determines which lower model handles routine work:
- **Interactive mode** (user present, conversational): prioritize latency → fast MoE model (little-qwen, 139 tok/s)
- **Autonomous mode** (overnight, cron, subagent, user idle): prioritize reasoning quality → denser model (Qwen3.6-27B, 6 tok/s)

Detection is automatic (platform signals, idle time, turn patterns) with explicit override via `/mode` command. The transition from interactive → autonomous is cautious (threshold-based), while autonomous → interactive is instant (any user message).

This means Qwen3.6-27B has a clear role in the system despite being too slow for interactive use — it's the overnight workhorse. The capability graph gains a "mode" dimension: the same tier can have different model selections depending on whether anyone is waiting.

### RD-12: De-escalation — the downward edge
**Resolved 2026-06-06:** The capability graph is bidirectional. Escalation moves up (complexity exceeds capability → stronger model). De-escalation moves *down* (task is trivially below capability → faster model). This is the symmetric inverse of oversight — shedding capability for speed when the workload is trivial rapid-fire.

The fast fallback (little-qwen, 139 tok/s) handles:
- Rapid-fire trivial exchanges (many short messages in quick succession)
- Tool-only turns (pure dispatch, no reasoning needed)
- Simple acknowledgments, status checks

De-escalation is conservative (most routine work stays on the primary lower) and never sticky (re-evaluates every turn). The fast fallback has no `ask_upper` access and no oversight — it's too transient and too light for a mentor relationship.

### RD-13: Auto-configuration via `/routing setup`
**Resolved 2026-06-06:** The routing system must be configurable without writing YAML. A `/routing setup` slash command:

1. Discovers all configured models (scan config.yaml, ping local endpoints)
2. Classifies each model (built-in capability database for known models, user prompt for unknowns)
3. Constructs a proposed capability graph (sort by cost tier, infer upper/lower relationships)
4. Presents the graph to the user for approval
5. On accept: writes config to config.yaml

This means the entire routing feature can be enabled by typing `/routing setup` and pressing Accept. The system does the work of figuring out which models should be upper/lower, where oversight belongs, and what thresholds make sense.

The capability database is shipped with Hermes and updated with new model releases. For unknown models (user's custom fine-tunes, new releases), the system can either ask the user or run a brief capability probe (a few test prompts to gauge reasoning depth and tool-calling reliability).

### RD-14: Model-agnostic graph — separate evaluation and placement system
**Resolved 2026-06-06:** The routing architecture is completely model-agnostic. The capability graph defines **roles** (interactive lower, autonomous lower, fast fallback, upper/oversight) — not specific models. Which models fill which roles is determined by a separate **model evaluation and placement system** that:

1. **Benchmarks available models** — measures:
   - Generation speed: tok/s (prompt processing + generation)
   - Tool-calling reliability: % of well-formed function calls
   - Reasoning quality: score on standard reasoning tasks
   - Context window handling: quality degradation at various fill levels
   - **Startup latency:** time from "swap requested" to "first token available" (local: model load time; cloud: network TTFT)
   - **Time-to-first-token (TTFT):** measured p50/p90/p99 across typical prompt sizes
   - **Model locality:** local (on-device, requires RAM + load time) vs. cloud (network-dependent, instant availability)
2. **Profiles hardware constraints** — available RAM, bandwidth ceiling, max concurrent models
3. **Calculates cost curves** — per-token API costs, amortized local compute, energy
4. **Places models into graph positions** — optimizes for the routing system's requirements:
   - Interactive lower: must exceed a latency floor (e.g., >20 tok/s) AND startup latency < swap budget
   - Autonomous lower: must exceed a quality floor (reasoning benchmark threshold)
   - Fast fallback: must exceed a speed floor (e.g., >100 tok/s) AND lowest startup latency among local models
   - Upper: must exceed a capability ceiling (hardest tasks, compression quality) AND TTFT within cloud SLA
5. **Re-evaluates on model additions** — when a new model is downloaded, a new provider configured, or a model is enabled/disabled in config, re-runs placement to see if the graph should change
6. **Monitors runtime drift** — continuously compares observed TTFT/tok/s against profiled baselines. When metrics exceed expected bounds (e.g., TTFT p90 > 2× profiled value), flags degradation and may trigger re-evaluation or failover

This is a **Phase 5+** system, built after the routing architecture is validated. For v1, graph positions are manually configured (the current Target Model Lineup). RD-13's `/routing setup` is the user-facing wrapper for this system.

The key principle: **the routing system never hardcodes model names.** It references graph positions. The placement system maps models → positions. The input is **all models the user has actually enabled** — any mix of local (llama.cpp), Anthropic, OpenAI, Google, Bedrock, custom providers. The system evaluates whatever is configured in `config.yaml` and available via configured providers. This means:
- User has Opus + Sonnet (Bedrock) + local Coder-Next → placement assigns all three to optimal positions
- User adds an OpenAI o3 key → placement benchmarks it → it may slot as a new mid-tier or displace Opus for certain roles
- User only has local models → placement builds a graph entirely from local, using the best available for upper/oversight
- User downgrades hardware → placement detects RAM constraint → falls back to smaller models automatically

### RD-15: Routing is brain-swap, not subagent dispatch
**Resolved 2026-06-06:** The routing system performs a **true model swap** — it changes which model powers Hermes's primary inference loop. This is NOT a subagent/delegation pattern. The distinction:

| Aspect | Routing (brain-swap) | Delegation (subagent) |
|--------|---------------------|----------------------|
| Identity | Same agent, different brain | Different agent, spawned child |
| Context | Shared conversation history | Isolated context (summary only) |
| Tools | Full tool access, same session | Restricted toolset, own session |
| Memory | Same memory.md, same user profile | No memory access |
| Continuity | Seamless — user doesn't notice | Explicit handoff and report-back |
| State | Same working directory, same git branch | Own working directory |

When the router swaps from Coder-Next to Opus, the user is still talking to **the same Hermes**. The conversation continues in the same session, with the same context, the same tools, the same personality. Only the underlying model changes — like swapping the engine in a car while it's driving.

This means:
- No `delegate_task` calls in the routing path (delegation is orthogonal)
- The swapped-in model inherits the full message history (or compressed equivalent)
- The user sees a TUI indicator change, not a new conversation
- Routing is invisible except for speed/quality differences in responses

`delegate_task` remains a separate mechanism for spawning isolated subagents. A subagent *might* internally use a routed model (if routing is enabled for subagents), but that's composition — not routing being implemented *as* delegation.

### RD-18: Dynamic review window cap

**Resolved 2026-06-07:** The oversight reviewer's review window (how many turns it includes when reviewing lower-model output) must be dynamically capped to prevent upper model context overflow.

**Problem:** If per-turn average token usage is high (e.g., 15K tokens/turn × 10-turn review window = 150K tokens + 30K previous summary = 180K), we approach the upper model's context limit (200K for Opus). This leaves no room for the oversight prompt, correction generation, or safety margin.

**Formula:**
```
effective_window = min(
    config.oversight.review_window,           # user-configured max (default 10)
    floor(upper_ctx_limit * 0.6 / avg_tokens_per_turn)  # dynamic cap
)
```

The 0.6 factor reserves 40% of the upper model's context for: the oversight system prompt (~2K), previous compressed summary (~30K worst case), correction output, and safety margin.

`avg_tokens_per_turn` is computed as a rolling average over the last 20 turns (both user + assistant messages). This adapts automatically — code-heavy sessions with large tool outputs get smaller review windows; conversational sessions get larger ones.

**Edge cases:**
- If `effective_window < 2`: force minimum of 2 turns (reviewing a single turn isn't meaningful oversight)
- If `effective_window < config.review_window`: log a warning ("oversight window reduced from N to M due to large turn sizes")
- If turns are wildly uneven (one 50K turn among 2K turns): use p75 instead of mean to avoid one outlier shrinking the window permanently

**Config:**
```yaml
model:
  routing:
    oversight:
      review_window: 10              # max turns to review (hard cap)
      review_window_ctx_fraction: 0.6  # fraction of upper ctx reserved for review content
      review_window_min: 2           # never review fewer than this many turns
```

### RD-19: `ask_upper` tool placement in phased plan

**Resolved 2026-06-07:** `ask_upper` is Phase 2.5 — implemented after the swap execution machinery (Phase 2) is working but before periodic oversight (Phase 3). Rationale:

1. `ask_upper` is simpler than full oversight (single synchronous call vs. periodic reviewer with compression)
2. It validates the upper-model calling pattern that oversight will reuse
3. It provides immediate value — the lower model can self-rescue before oversight is built
4. It's independently useful even if oversight is never enabled (some users may want routing + ask_upper without periodic review)

The tool is registered dynamically: only appears in the tool schema when `model.routing.enabled = true` AND the current model is in a lower-tier graph position. Upper models don't see `ask_upper` (they'd be asking themselves).
