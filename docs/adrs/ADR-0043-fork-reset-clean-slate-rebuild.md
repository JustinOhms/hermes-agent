# ADR-0043: Fork reset and spec-first rebuild (clean-slate migration)

**Status:** Accepted (in progress) — 2026-08-03
**Relates:** ADR-0042 (reliable blue/green deploy), ADR-0040 (routing), ADR-0041
(sandwich), `versioned-patched-fork-strategy.md`.

## Context

The fork (`JustinOhms/hermes-agent`) drifted **~7,100 commits behind upstream**
while carrying patches pinned to old SHAs, with the *intent* behind most branches
undocumented — the root cause of the reliability problems in ADR-0042. A Phase 0
triage (2026-08-03) found the ~20 branches are heavily **cross-contaminated**
(cherry-picking mixed concerns), e.g. the "blue/green slot indicator" commit
exists in 5 branches, and `fixes/tui-slot-indicator` contains bedrock commits.
Rebasing that mess across 7k commits would be error-prone and would drag along
patches we no longer want.

**Provider change:** primary cloud LLM is now **OpenAI Codex** with **qwen-coder**
local; **Amazon Bedrock is retired.** This alone drops most of the patch debt.

## Decision

Declare bankruptcy on the accumulated patch debt. **Reset the fork to current
upstream and rebuild the (few) patches we still want from written specs**, rather
than merging old diffs. The missing documentation is written as part of the
rebuild — that is the actual fix, not overhead.

### Branch disposition (from Phase 0 triage)

**KEEP — 4 concerns (rebuild from spec):**
| Concern | Canonical source | Spec |
|---|---|---|
| A. Model routing (ADR-0040 heuristic + ADR-0041 sandwich + `/routing`) | `feat/routing-integrated` (superset) | ADR-0040/0041 — refresh |
| B. Runtime robustness: non-blocking stream writer (PR #37633, still OPEN), tool watchdog @300s, hot-reload `max_iterations` | `fixes/stream-delta-backpressure` | NEW ADR needed |
| C. Blue/green TUI slot indicator | `stream-delta-backpressure` | folds under ADR-0042 |
| D. `pre_failover_decision` hook | `feat/pre-failover-hook-v2` | **DEFERRED** — decide during A's rebuild whether routing subsumes it |

**DROP:**
- All bedrock: `fixes/bedrock-{bearer-token,read-timeout}`,
  `fixes/delegation-bedrock-and-empty-sentinel`, `fix/bedrock-*` (6 branches).
- `feat/routing` (superseded by `routing-integrated`),
  `feat/adr-0041-sandwich` (folded into routing; local-only),
  `enhance/tui-fixes` (redundant subset of B/C),
  `fixes/tui-slot-indicator` (contaminated),
  `feat/pre-failover-hook` v1 (superseded by v2).

**RESET / ARCHIVE (infra, not patches):**
- `main` / `origin/main` / `origin/fork/main` → reset to `upstream/main`. The lone
  divergent commit on main is *"Add sandwich pipeline: agent/routing"* — routing
  code accidentally on main (the invariant violation we're removing); captured in A.
- `main-blue` / `main-green` / `patched/v*` → archive. `main-green`=`a030ef3c` is
  the live baseline, tagged `archive/deployed-2026-08-03-green` before any change.

## Plan (phased)

- **Phase 0 — Triage** ✅ (this ADR).
- **Phase 1 — Reset:** tag baseline + `archive/*` tags for everything dropped (so
  nothing is lost), reset `origin/main = upstream/main`, archive old branches.
- **Phase 2 — Specs:** refresh ADR-0040/0041 (routing); write a new ADR for B
  (runtime robustness); evaluate D.
- **Phase 3 — Rebuild** clean, single-concern `fixes/*` / `feat/*` branches on
  current main from the ADRs (drop what upstream already has — e.g. re-check PRs).
- **Phase 4 — Deploy:** consolidate → `hermes-upgrade` to the **idle** slot →
  verify → swap → rebuild the other slot. Live slot untouched until the swap.
- **Phase 5 — Cleanup:** remove bedrock scripts/hooks/cron; pin cron jobs to
  codex/qwen (the `bedrock→openai-codex` cron drift is *intentional*, not a bug).

## Safety

The live green slot is never touched until Phase 4's deploy-to-idle-then-swap; the
ADR-0042 failover/health infrastructure protects throughout; every dropped branch
is preserved as an `archive/*` tag; outward fork pushes are done deliberately with
Justin's confirmation.

## Who executes

Per the "Claude does the upgrades" decision: this migration (and future upgrades)
are executed by **Claude Code in a supervised shell**, not by Hermes updating
itself. The `hermes-upgrade` skill will be **distilled from this execution**.
Human gates remain on fork pushes and the live slot swap.
