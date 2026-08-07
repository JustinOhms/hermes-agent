# `.ai/` — fork operating rules

Fork-specific rules and directives for **JustinOhms/hermes-agent** (a patched
fork of NousResearch/hermes-agent). This content is **not** from upstream; it is
carried by the `feat/agent-meta` living patch so it survives upstream resyncs.
`AGENTS.md` (an upstream file) only holds a short pointer to here.

> **Why this folder exists:** the rules that govern how this fork is built and
> deployed used to live only in a *separate* repo
> (`~/Dropbox/Dev/hermes-config-backup/RUNBOOK.md`), so an agent working inside
> `hermes-agent` had no way to discover them — and nearly edited the live slot as
> a result. These files put the essentials in-repo, where they are found.

## Contents

- **[operations.md](operations.md)** — deployment, blue/green slots, the branch /
  living-patch model, the swap gotcha, recovery. **Read before any deploy, slot,
  or branch action.**

## Conventions

- One concern per file; keep each self-contained (an agent may read only one).
- These are fork rules. Upstream code conventions still live in the repo's
  top-level `AGENTS.md` / `CONTRIBUTING.md`.
- Full operational depth (tooling internals, gate sources) is in
  `~/Dropbox/Dev/hermes-config-backup/RUNBOOK.md` and the `hermes-upgrade` skill.
