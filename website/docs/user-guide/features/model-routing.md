---
title: Intelligent Model Routing
description: Automatically route each turn to the right model based on complexity, interaction mode, and cost — without manual /model switching.
sidebar_label: Model Routing
sidebar_position: 6
---

# Intelligent Model Routing

Intelligent model routing lets Hermes automatically pick the best model for each turn. Instead of manually switching between a powerful (expensive) model and a fast (cheap) one, the router scores each message's complexity, detects whether you're actively chatting or the agent is working autonomously, and routes to the appropriate model in your configured graph.

:::tip When to use this
Model routing is most valuable when you have multiple models available — e.g., a cloud model (Opus, Sonnet) for complex reasoning and a local model for fast, routine work. If you only use one model, you don't need routing.
:::

## Quick Start

Add a `model.routing` section to `~/.hermes/config.yaml`:

```yaml
model:
  routing:
    enabled: true
    graph:
      upper:
        provider: bedrock
        model: us.anthropic.claude-opus-4-6-v1
        display_name: "Claude Opus 4"
        profile:
          generation_tok_s: 80
          ttft_p50_ms: 2000
      lower:
        provider: custom:llm-local
        model: qwen3-coder-30b
        base_url: http://127.0.0.1:58080/v1
        llm_config_name: little-qwen
        display_name: "Qwen3 Coder 30B"
        profile:
          generation_tok_s: 139
          startup_latency_s: 8
          ttft_p50_ms: 200
```

Then start Hermes normally. The router activates on every turn and logs its decisions.

## How It Works

Each turn, the router:

1. **Scores complexity** of the user message (0.0–1.0) using keyword and structural analysis
2. **Detects interaction mode** — are you actively chatting (interactive) or is the agent working autonomously?
3. **Routes to a graph position** — picks `upper` for complex/critical work, `lower` for routine tasks
4. **Executes a swap** if the target differs from the currently loaded model

The system is designed to be invisible — you interact normally and the agent uses the best model for the job. Use `/routing` anytime to inspect what's happening.

## Configuration Reference

### Top-Level Structure

```yaml
model:
  routing:
    enabled: true                    # Master switch (default: false)
    graph: { ... }                   # Named model positions
    complexity: { ... }              # Scoring thresholds
    interaction_mode: { ... }        # Mode detection tuning
    de_escalation: { ... }           # Whether to route down from upper
    ask_upper: { ... }               # Upper-model consultation tool
    oversight: { ... }               # Periodic review system
```

### Graph Positions

The `graph` maps position names to model configurations. Position names are arbitrary — the router uses complexity scores and thresholds to pick between them.

```yaml
graph:
  upper:
    provider: bedrock               # Any Hermes provider
    model: us.anthropic.claude-opus-4-6-v1
    display_name: "Claude Opus 4"   # Shown in TUI status bar
    profile:
      startup_latency_s: 0          # Cold-start time (0 for cloud)
      ttft_p50_ms: 2000             # Time-to-first-token median
      ttft_p90_ms: 4000             # Time-to-first-token P90
      generation_tok_s: 80          # Tokens/second generation speed
  lower:
    provider: custom:llm-local
    model: qwen3-coder-30b
    base_url: http://127.0.0.1:58080/v1
    llm_config_name: little-qwen    # For `llm start <name>` auto-launch
    display_name: "Qwen3 30B MoE"
    profile:
      startup_latency_s: 8
      ttft_p50_ms: 200
      ttft_p90_ms: 400
      generation_tok_s: 139
```

#### Position Fields

| Field | Required | Description |
|-------|----------|-------------|
| `provider` | ✔ | Provider name (same values as `model.provider`) |
| `model` | ✔ | Model identifier for the provider |
| `display_name` | — | Human-friendly name for TUI/status display |
| `base_url` | — | Custom API endpoint (for `custom:*` providers) |
| `api_key` | — | API key override (usually resolved from env vars) |
| `api_mode` | — | `chat_completions` or `messages` |
| `llm_config_name` | — | Local model preset name (for `llm start <name>`) |
| `profile.startup_latency_s` | — | Cold-start time in seconds (0 for always-on cloud) |
| `profile.ttft_p50_ms` | — | Time-to-first-token P50 in milliseconds |
| `profile.ttft_p90_ms` | — | Time-to-first-token P90 in milliseconds |
| `profile.generation_tok_s` | — | Generation speed in tokens/second |

:::info Profile is informational
The `profile` block provides latency/speed metadata for the swap decision engine. It doesn't enforce anything — it helps the router decide whether a swap is worth the latency cost.
:::

### Complexity Thresholds

```yaml
complexity:
  escalation_threshold: 0.7         # Route to upper above this score
  de_escalation_threshold: 0.2      # Route to lower below this score
```

Messages scoring between the two thresholds stay on whichever model is currently active (hysteresis — avoids ping-ponging).

### Interaction Mode

```yaml
interaction_mode:
  idle_threshold_s: 600             # Seconds of silence → autonomous mode
  swap_back_messages: 3             # Messages in quick succession → interactive
  swap_back_window_s: 60            # Window for swap_back_messages
```

The router treats **interactive mode** (user actively chatting) differently from **autonomous mode** (agent working on a long task):

- **Interactive**: Favors the upper model for complex queries, de-escalates aggressively to lower for simple ones (fast responses matter)
- **Autonomous**: More conservative — stays on the current model unless complexity strongly demands otherwise

### De-escalation

```yaml
de_escalation:
  enabled: false                    # Default: disabled
```

When `enabled: true`, the router can route *down* from upper to lower when complexity drops below `de_escalation_threshold`. When `false` (default), once routed to upper, the session stays there until a manual switch or mode change.

### Ask Upper (Mentor Tool)

When the lower model is active, it can optionally consult the upper model for guidance via the `ask_upper` tool:

```yaml
ask_upper:
  enabled: true
  soft_budget_calls: 3              # Warn after this many consultations
  hard_budget_calls: 8              # Refuse after this many
```

When enabled, the lower model gets an `ask_upper` tool it can invoke when it encounters something beyond its capability — architecture decisions, complex debugging, etc. The upper model responds with guidance without taking over the full conversation.

Budget limits prevent runaway costs. After `soft_budget_calls`, the tool warns about budget pressure. After `hard_budget_calls`, it refuses and suggests escalation.

### Oversight

The oversight system periodically has the upper model review what the lower model has been doing:

```yaml
oversight:
  enabled: true
  every_n_turns: 10                 # Review every N turns
  review_window: 10                 # How many turns to include in review
  review_window_ctx_fraction: 0.6   # Fraction of context budget for window
  review_window_min: 2              # Minimum turns in review (even if budget tight)
  max_reviews_per_session: 5        # Stop reviewing after this many
  min_turns_before_first: 5         # Don't review the first N turns
  skip_if_escalated: true           # Skip review if already on upper model
  model: ""                         # Override oversight model (default: upper)
  provider: ""                      # Override oversight provider
  base_url: ""                      # Override oversight endpoint
  mode_factor_interactive: 1.5      # Multiply interval in interactive mode
  mode_factor_autonomous: 0.8       # Multiply interval in autonomous mode
```

The oversight reviewer can take four actions:
- **approve** — lower model is doing fine, continue
- **correct** — inject a correction into the conversation
- **flag** — mark the turn for human review (logged)
- **escalate** — swap to upper model immediately

## The `/routing` Command

Use `/routing` in a session to inspect the routing system:

| Subcommand | What it shows |
|------------|---------------|
| `/routing` or `/routing status` | Current position, mode, complexity, decision history |
| `/routing graph` | All configured positions and their profiles |
| `/routing mode` | Current interaction mode and detection state |
| `/routing history` | Recent routing decisions with timestamps |
| `/routing oversight` | Oversight review history and budget |
| `/routing swap <position>` | Force-swap to a named position |
| `/routing upgrade` | Force-swap to upper |
| `/routing downgrade` | Force-swap to lower |

## Full Example

A complete configuration for a 3-position graph (cloud upper, local coder, local fast fallback):

```yaml
model:
  provider: bedrock
  default: us.anthropic.claude-opus-4-6-v1
  routing:
    enabled: true
    graph:
      upper:
        provider: bedrock
        model: us.anthropic.claude-opus-4-6-v1
        display_name: "Claude Opus 4"
        profile:
          generation_tok_s: 80
          startup_latency_s: 0
          ttft_p50_ms: 2000
          ttft_p90_ms: 4000
      interactive_lower:
        provider: custom:llm-local
        model: qwen3-coder-next
        base_url: http://127.0.0.1:58080/v1
        llm_config_name: coder-next
        display_name: "Qwen3 Coder Next"
        profile:
          generation_tok_s: 33
          startup_latency_s: 17
          ttft_p50_ms: 800
          ttft_p90_ms: 1500
      fast_fallback:
        provider: custom:llm-local
        model: qwen3-coder-30b
        base_url: http://127.0.0.1:58080/v1
        llm_config_name: little-qwen
        display_name: "Qwen3 30B MoE"
        profile:
          generation_tok_s: 139
          startup_latency_s: 8
          ttft_p50_ms: 200
          ttft_p90_ms: 400
    complexity:
      escalation_threshold: 0.7
      de_escalation_threshold: 0.2
    interaction_mode:
      idle_threshold_s: 600
      swap_back_messages: 3
      swap_back_window_s: 60
    de_escalation:
      enabled: false
    ask_upper:
      enabled: true
      soft_budget_calls: 3
      hard_budget_calls: 8
    oversight:
      enabled: true
      every_n_turns: 5
      mode_factor_autonomous: 0.8
      mode_factor_interactive: 1.5
```

## How It Relates to Other Features

| Feature | Relationship |
|---------|-------------|
| [Fallback Providers](./fallback-providers.md) | Fallback activates on *errors* (429, 500). Routing activates on every turn by *design*. Both can be active — routing picks the target, fallback catches failures. |
| [Provider Routing](./provider-routing.md) | OpenRouter sub-provider preferences. Orthogonal — controls *which datacenter* serves a model, not *which model* to use. |
| `/model` command | Manual model switch. Routing automates this. A manual `/model` switch overrides the router for that turn. |
| [Delegation](./delegation.md) | Subagents inherit the parent's routing config. They make their own per-turn routing decisions independently. |

:::note Experimental Feature
Intelligent model routing is new and actively evolving. The configuration surface and behavior may change between releases. The core design (ADR-0040) is stable but the specific thresholds, oversight actions, and swap heuristics are being refined based on real-world usage.
:::
