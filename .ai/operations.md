# Fork operating rules — deployment, slots, branches

> **This file is fork-specific** (JustinOhms/hermes-agent). It does not come from
> upstream NousResearch. It is carried by the `feat/agent-meta` living patch and
> re-applied on every upstream resync. **Read this before any deploy, slot, or
> branch action.**

The rules below exist because this fork runs a **blue/green deployment** of a
*live* agent. The full source of truth is
`~/Dropbox/Dev/hermes-config-backup/RUNBOOK.md` and the `hermes-upgrade` skill;
this file is the in-repo, self-contained essentials so an agent working here
discovers them without needing that other repo.

## The one rule that matters most

**Edit ONLY the working copy (`~/Dropbox/Dev/hermes-agent/`). The deployed slots
are DEPLOY-ONLY — never edit them in place.**

If you ever type a path under `~/.hermes/hermes-agent-blue/` or
`~/.hermes/hermes-agent-green/` in an editor or write command, **stop — wrong
place.** Edits made directly in a slot are off-the-books: a `git reset --hard`
during the next deploy silently discards them, and editing the *active* slot can
break the running gateway mid-turn ("never perform surgery on the running
brain"). Slots' source files are kept read-only by `hermes-slot-guard` precisely
to make this mistake loud.

## Layout

- **Working copy** — `~/Dropbox/Dev/hermes-agent/`. All edits happen here.
- **Remotes** — `origin` = `JustinOhms/hermes-agent` (the fork), `upstream` =
  `NousResearch/hermes-agent`.
- **`main`** — a pure mirror of `upstream/main`. Never put fork content on `main`;
  a clean-slate resync resets it to upstream. (That is why these rules live in
  `.ai/` + a small AGENTS.md pointer, not inside upstream's AGENTS.md.)
- **Deploy branches** — `main-blue`, `main-green` (on `origin`). Each slot is a
  git checkout of its color branch.
- **Slots** — `~/.hermes/hermes-agent-{blue,green}`. `~/.hermes/hermes-agent` is a
  symlink to the **active** slot. The other is **idle** (the fallback + the next
  deploy target).

## Living-patch / branch model

Fork changes are **independent living patches**, not commits piled onto a deploy
branch:

- `feat/*` and `fixes/*` — one concern each, branched off current `main`
  (upstream mirror). Tracked in `~/.hermes/active-patches.yaml`.
- `patched/vX` — a consolidation of all active patches.
- On upgrade, the consolidation is applied to the **idle** color branch, deployed
  to the idle slot, then swapped live. Old branch tips are preserved as
  `archive/*` tags.
- **Do not cherry-pick old patches across a large upstream gap** — re-derive each
  from its ADR + `archive/*` source against *current* upstream (see the
  `hermes-upgrade` skill).

## Deploying (the only supported path)

Never run bare `hermes update` (it does an in-place `git pull` and destroys the
slot model). Use the gated orchestrator `hermes-update` (in `~/.local/bin`):

```bash
# 1. Preview — dry-run, mutates nothing:
hermes-update --ref origin/main-blue
# 2. Validate on the idle slot WITHOUT going live (deps + tests + boot verify,
#    re-lock, but NO swap):
hermes-update --ref origin/main-blue --apply --no-swap
# 3. Full deploy (all gates, then swap + gateway restart):
hermes-update --ref origin/main-blue --apply
```

Gates (all fail-closed — any failure aborts and leaves the symlink untouched):
0 rescue-snapshot · 1 identify active/idle · 2 fetch · 3 sync idle→ref
(guard-unlock, preserves venv/node_modules) · 4 deps · 5 test gate (blocks on our
features/guards; upstream-flaky dirs are advisory vs a recorded baseline) ·
6 live-verify (boots the idle slot, imports `agent.routing` etc.) · 7 re-lock ·
8 swap · 9 log to `deploy-log.md`.

## The swap gotcha (do not skip the plist regen)

A bare `hermes gateway restart` only kickstarts the *existing* launchd plist,
which hardcodes the resolved **old-slot** python path — it does NOT re-point the
gateway at the new slot. A swap must regenerate the plist. Manual swap / rollback:

```bash
hermes-slot green                              # or blue — flip the symlink
hermes gateway stop
hermes gateway install --force --no-start-now  # REGENERATE the plist for the new slot
hermes gateway start
hermes gateway status                          # confirm the live PID runs from the new slot
```

Always confirm the running gateway PID's interpreter is under the newly-active
slot (`ps -p <pid> -o command=`), not the old one.

## Recovery

- `hermes-health` — status; `hermes-health --fix` repoints a dangling symlink to
  the healthy twin.
- Full rebuild: unzip the newest `~/Dropbox/Dev/hermes-config-backup/rescue/hermes-rescue-*.zip`
  and run `restore.sh` (idempotent; moves old state to `.bak-*`, never deletes).

## Non-negotiables (summary)

1. Edit the working copy; never a slot. Never touch the **active** slot at all.
2. Two human gates — pause before (a) pushing to the fork and (b) the live swap.
3. `hermes-update` only (never bare `hermes update`); dry-run, then `--no-swap`,
   then `--apply`.
4. Regenerate the plist on every swap; verify the live PID is on the new slot.
5. Keep `main` a pure upstream mirror; carry fork changes as `feat/*`/`fixes/*`
   living patches registered in `active-patches.yaml`.
