# ADR-0044: Runtime robustness (stream backpressure, tool watchdog, live iteration cap)

**Status:** Accepted (carried patch) — documented 2026-08-03
**Source of record:** `archive/fixes/stream-delta-backpressure`
**Upstream:** the stream-writer piece is PR **#37633** (still OPEN as of 2026-08-03).

## Context

Three independent, provider-agnostic runtime hardening fixes accumulated on the
`stream-delta-backpressure` branch. They were never documented — this ADR captures
their intent so they can be rebuilt cleanly on current upstream and dropped
individually if/when upstream absorbs them. They are unrelated to the (now retired)
Bedrock work and remain relevant under the Codex + qwen-coder setup.

## Decision — carry these three fixes as separate, independently-droppable patches

### 1. Non-blocking stream delta writer  (`tui_gateway`) — PR #37633
**Problem:** the gateway wrote streaming deltas to stdout synchronously; when the
consumer (TUI) applied backpressure, the write blocked, stalling the whole turn.
**Fix:** make the stream-delta writer non-blocking so a slow/paused consumer can't
stall generation. Keep carrying until #37633 merges upstream; then drop.

### 2. Tool watchdog — kill hung tools after 300s  (`tool_executor`)
**Problem:** a tool call that hangs (network, subprocess, external service) could
wedge a turn indefinitely with no recovery.
**Fix:** a watchdog timer terminates a tool that exceeds a timeout (default 300s)
and surfaces a typed error, so the agent loop recovers instead of hanging.
(Timeout should be configurable; default 300s.)

### 3. Hot-reload `max_iterations` from config each turn
**Problem:** `max_iterations` was read once at startup, so tuning it required a
full restart — painful for a long-lived gateway.
**Fix:** re-read `max_iterations` from config on each turn so it can be tuned live
without restarting the gateway.

## Consequences

- Each is a small, self-contained change touching a single subsystem
  (`tui_gateway/`, `tools/tool_executor`, agent loop config read) — they rebuild
  independently on current upstream and each maps to its own `fixes/*` branch.
- The stream-writer fix has an open upstream PR (#37633); track it and drop the
  local carry once merged.

## Rebuild notes (Phase 3)

Re-derive from `archive/fixes/stream-delta-backpressure` commits
`be9033f4b` (writer), `a34f8a51c` (watchdog), `dd2f7d330` (hot-reload). The 4th
commit on that branch (`8119bdad1`, blue/green slot indicator) is a *separate*
concern — see the slot-indicator patch under ADR-0042.
