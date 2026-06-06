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

### Phase 3: Periodic Oversight

1. Create `agent/oversight.py` — follows `background_review.py` pattern
2. Oversight prompt + action parsing
3. Hook into turn counter in conversation loop
4. Injection mechanism for corrections
5. Escalation handoff (oversight model takes one turn)
6. User notification for flags
7. TUI indicator when oversight is active
8. Tests: mock oversight responses, verify injection behavior

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
    │ approve/correct/     │ compressed           │ structured
    │ escalate/flag        │ summary              │ prompt for
    │                      │                      │ lower model
    └──────────────────────┴──────────────────────┘
                           │
                           ▼
                  Lower model continues
                  with fresh context
```

#### The Compression Prompt (to upper model)

```
You are compressing a conversation for handoff to a less capable model.
It needs to continue working seamlessly. Produce a structured summary:

## Session State
- What the user originally asked for
- What has been accomplished so far
- What is currently in progress
- What remains to be done

## Key Decisions Made
- [Decision]: [rationale] (turn N)

## Active Context
- Files being worked on and their current state
- Environment state that matters (running processes, git branch, etc.)
- Constraints or requirements discovered during work

## Guidance for Next Steps
- What the model should do next
- Any pitfalls to avoid (learned from this session)
- Approach recommendations

Keep this under 3000 tokens. Prioritize actionability over completeness.
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

#### Compression Budget

Each compression costs:
- Input: ~10 turns × ~1000 tok/turn = ~10K tokens
- Output: ~2-3K tokens (the summary)
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
| Oversight adds latency | User waits for oversight before next turn | Oversight runs async (background thread, like background_review.py) — user never waits |
| Local model goes down mid-session | Session breaks | Existing fallback mechanism handles this — escalation model becomes sole provider |
| Config complexity | Users confused by routing config | Sensible defaults + `hermes setup` wizard step + `hermes doctor` validates routing config |

---

## Success Criteria

1. **Cost reduction:** 60-80% fewer cloud API calls for typical coding sessions
2. **Quality maintained:** Oversight catches errors before they compound (< 3 turns of drift)
3. **Transparent:** User always knows which model is active (TUI indicator)
4. **Zero-config viable:** Works with just `routing.enabled: true` + primary/escalation models defined
5. **No latency penalty:** Router adds < 5ms per turn; oversight is async

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
   - Leaning: soft budget (warning after N calls) rather than hard cap

2. **Handback prompt authoring:** Should the upper model's handback prompt be structured (template + fields) or freeform?
   - Leaning: structured template with sections (summary, active files, current task, guidance)

3. **Graph edge weights:** When there are multiple possible upper models, how does the lower model choose which to ask?
   - Leaning: for v1, there's only one upper. For vN, select based on capability match to the question type.

4. **Compression fidelity:** How do we verify the compression didn't lose critical context?
   - Leaning: upper model includes a `key_facts` list that the lower model can reference; if the lower model seems confused, oversight catches it next cycle

5. **How does this interact with `delegate_task`?** Subagents have their own model config — should they inherit routing?
   - Leaning: subagents use `delegation.model` as today; routing is session-level only

6. **Should the routing config be part of a profile?** So you can have `hermes -p local` (routing on) and `hermes -p opus` (always Opus)?
   - Leaning: yes, naturally — each profile has its own config.yaml
