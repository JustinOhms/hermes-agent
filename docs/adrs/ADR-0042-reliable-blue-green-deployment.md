# ADR-0042: Reliable blue/green deployment, update, and recovery

**Status:** Accepted — initial implementation landed 2026-08-03
**Supersedes/relates:** the `hermes-blue-green-deployment` skill, the archived
`hermes-blue-green-update` skill, and `versioned-patched-fork-strategy.md`.
**Context RCA:** `docs/rca-ssl-cacert-post-git-pull.md`.

## Context

Hermes runs from a blue/green slot layout under `~/.hermes/`, deployed from a
patched fork (`JustinOhms/hermes-agent`) that carries local patches on
`patched/vYYYYMMDD` branches over `upstream/main` (NousResearch). The design is
sound, but the safety guarantees were not *enforced*, so a single slip had no
floor. On 2026-08-03 the gateway went fully down and could not self-recover.

Post-incident analysis found the cascade:

1. An **in-place update repaired the venv of the ACTIVE (blue) slot** while the
   gateway was live on it; the repair was interrupted, leaving blue's venv
   broken (missing `certifi/cacert.pem` — the exact RCA above), then the slot
   directory was lost.
2. The **launchd gateway hard-coded `hermes-agent-blue`** instead of following
   the `hermes-agent` symlink, so a missing/broken blue = infinite crash-loop.
3. The **recovery tool (`hermes-slot`) had a bug** (`check_slots` required the
   *blue* dir even to switch *to green*) — it refused in exactly the situation
   it existed for.
4. The **idle slot was treated as "ephemeral"** so there was no maintained twin
   to fall back to; the **rescue parachute was 2+ weeks stale**; and nothing was
   watching, so the degradation was silent for ~2 weeks.

## Decision — the invariants (enforced, not conventional)

1. **The `hermes-agent` symlink is the single source of truth for "active slot."**
   Everything that runs Hermes resolves the slot through the symlink, never a
   hard-coded path.
2. **The active slot is sacred.** No update / venv / git-mutating operation may
   ever write the active slot. Updates build the **idle** slot; going live is an
   atomic symlink flip + gateway restart.
3. **Always two bootable slots.** The idle slot is a maintained, tested twin —
   never dangling, never disposable.
4. **Recovery is one command and always works**, and a **fresh parachute** is
   guaranteed before every deploy.
5. **Fail-closed + observable.** Verify gates cannot false-pass; a broken slot,
   dangling symlink, or slot/gateway mismatch is detected, auto-healed where
   safe, and alerted otherwise.

## Architecture

```
dev repo  ~/Dropbox/Dev/hermes-agent   (feat/* work; edits ONLY here)
   │  git push origin
   ▼
fork  github.com/JustinOhms/hermes-agent
   fork/main == upstream/main   (invariant; patches never on main)
   fixes/*    living patch branches (cherry-pick source)
   patched/vYYYYMMDD  immutable deploy snapshot = fork/main + fixes/*
   │  deploy the IDLE slot to patched/vLATEST, verify, swap
   ▼
slots ~/.hermes/hermes-agent-{blue,green}   (deploy targets ONLY; read-only via slot-guard)
   hermes-agent -> active slot   (symlink = source of truth)
   │
   ▼
launchd ai.hermes.gateway -> hermes-gateway-launch (resolves symlink, fails over)
```

Remotes (note: opposite of the old strategy doc): **`origin` = the fork
(JustinOhms), `upstream` = NousResearch.**

## Tooling (all under `~/Dropbox/Dev/hermes-config-backup/rescue/`, symlinked into `~/.local/bin` and `~/.hermes/bin`)

| Tool | Role |
|------|------|
| `hermes-slot-lib.sh` | shared helpers: `hermes_active_slot`, `hermes_slot_healthy` (python+import+CA), `assert_not_active_slot` |
| `hermes-gateway-launch` | launchd entrypoint: resolves the active slot via symlink; **fails over** to the other bootable slot instead of crash-looping |
| `hermes-slot` | switch active slot (bug fixed: only requires the *target* slot) |
| `hermes-slot-guard` | keeps slot source read-only (blocks in-slot edits) |
| `hermes-update` | gated blue/green deploy to the idle slot (see gates below) |
| `hermes-rescue` | dated self-restoring snapshot; prunes to last 5 |
| `restore.sh` | rebuild from a snapshot (the scripted escape hatch) |
| `hermes-health` / `hermes-health-cron` | health surface + 30-min self-heal + alert |

### Enforcement points implemented 2026-08-03
- **Gateway follows the symlink + fails over.** `ai.hermes.gateway.plist` now
  runs `~/.hermes/bin/hermes-gateway-launch` (original backed up to
  `rescue/ai.hermes.gateway.plist.orig`). It picks the active slot, health-checks
  it, and repoints to the healthy twin if the active slot is broken.
- **Active-slot-sacred guard.** `hermes-update` gate 1 calls
  `assert_not_active_slot`. The dangerous **`hermes update` is rerouted** in
  `~/.oh-my-zsh/custom/04-functions.zsh` to the safe orchestrator (escape hatch:
  `command hermes update`).
- **Fresh parachute.** `hermes-update` gate 0 auto-generates a snapshot before
  applying; `backup-hermes-config.sh` (daily launchd) regenerates one too.
- **Fail-closed verify.** Gate 4 installs pytest plugins; gate 5 aborts on pytest
  exit ≥ 2 / unrecognized args; gate 6 runs `hermes_slot_healthy` (CA bundle)
  before swap.
- **Swap without clobbering the launcher.** Gate 8 flips the symlink and
  `launchctl kickstart`s the gateway — it must **not** run
  `hermes gateway install --force` (that regenerates a hard-coded-slot plist).
- **Health + self-heal + alert.** `com.justinohms.hermes-health` runs
  `hermes-health --fix --quiet` every 30 min: repoints a dangling symlink,
  restarts a mismatched gateway, and raises a macOS notification on a critical
  condition. Logs to `~/.hermes/logs/hermes-health.log`.

## The update cycle (target: one command)

Manual/interactive today via `hermes-update` (dry-run by default, `--apply` to
execute). The documented patched-fork cycle it builds on:

1. `git fetch upstream && git fetch origin`
2. fast-forward `fork/main` → `upstream/main` (patches never live on main)
3. `git checkout -B patched/vSTAMP origin/main`
4. cherry-pick each `fixes/*` tip listed in `active-patches.yaml` (push resolved
   conflicts back to the `fixes/*` branch so next cycle is free)
5. `git push origin patched/vSTAMP`
6. `hermes-update --apply` → deploys `patched/vSTAMP` to the **idle** slot,
   verifies, swaps.

`hermes update` (bare) is intentionally disabled; use `hermes-update`.

## Recovery runbook (one command, always works)

- **Gateway won't start / dangling symlink:** `hermes-health --fix` (repoints to
  the healthy twin) — or, if the launchd gateway restarts on its own, the
  `hermes-gateway-launch` wrapper fails over automatically.
- **Switch slots manually:** `hermes-slot green` / `hermes-slot blue` /
  `hermes-slot swap`, then `launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`.
- **Both slots broken:** restore from the newest snapshot —
  `unzip -d /tmp/r ~/Dropbox/Dev/hermes-config-backup/rescue/hermes-rescue-YYYYMMDD.zip && cd /tmp/r && bash restore.sh`.

## Consequences

- The 2026-08-03 failure mode is closed: the gateway follows the symlink and
  fails over; the active slot cannot be rebuilt in place; a parachute is always
  fresh; and health is watched and self-heals.
- Slight complexity increase (a wrapper + a health job), all versioned in the
  config-backup repo and reversible (backups kept alongside each change).

## Open items (tracked)

- **One-command upgrade** wrapping the full patched-fork cycle end-to-end.
- **Branch/remotes reconcile:** re-assert `fork/main == upstream/main` (the
  fork's `origin/main` had drifted behind upstream), prune stale `patched/v*`,
  and formalize `feat/* → fixes/* → patched/vX → slot`. (Outward-facing fork
  pushes — do deliberately.)
- **Fold loose patches:** move `fix-bedrock-*.sh` into the `fixes/*` branches so
  nothing is re-applied by side scripts.
- **Config drift:** reconcile the `bedrock → openai-codex` provider drift that
  silently skips unpinned cron jobs.
- **`restore.sh`:** install the `~/.hermes/bin/hermes-gateway-launch` symlink so a
  bare-metal restore reproduces the wrapper-based gateway.
