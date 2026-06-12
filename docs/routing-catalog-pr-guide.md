# Routing PR + Catalog PR Reference Guide

## Overview
The routing feature was split into **two PRs** to maintain clean separation of concerns:

1. **Routing PR** (`feat/routing`) — the *contract* and *minimal resolver*
2. **Catalog PR** — the *live implementation* (adds OpenRouter + models.dev + AI Model Directory)

## How They Relate

```
┌─────────────────────────────────────────────────────────────┐
│  Routing PR (feat/routing)                                  │
│  • agent/routing/types.py (dataclasses)                     │
│  • agent/routing/fingerprint.py (builtin catalog only)      │
│  • TUI model switch notifications                           │
│  • Swap execution logic                                     │
│  • 0 network deps, self-contained                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    catalog.install()
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Catalog PR (separate PR, depends on Routing PR)            │
│  • agent/routing/catalog.py (live sources)                  │
│  • patches fingerprint._match_catalog                       │
│  • adds: OpenRouter, models.dev, AI Model Directory         │
│  • adds: capability extraction (context window, pricing, etc.) │
└─────────────────────────────────────────────────────────────┘
```

## Routing PR Description Template

Include this in the **"Additional Notes"** section of the routing PR:

> ### Future Work: Live Model Catalog
> 
> This PR implements the *minimal* fingerprint resolver using a static builtin catalog (20 well-known models).  
> A follow-up PR will add a live-source catalog module (`agent/routing/catalog.py`) that:
> 
> - Fetches model metadata from **OpenRouter API**, **models.dev**, and **AI Model Directory**
> - Provides exact model ID resolution and URL-based matching
> - Extracts capabilities (context window, vision, tool calling, pricing)
> - Gratefully degrades to builtin catalog if network fails
> 
> The two PRs share the same datacontract (`agent/routing/types.py`) — no breaking changes.

---

## Catalog PR Description Template

Include this in the **"Summary"** section of the catalog PR:

> ### Summary
> 
> This PR adds live-source model catalog resolution to the fingerprint system introduced in PR #XXX (Routing PR).
> 
> It **does not** change the public API — instead, it *patches* the minimal resolver at runtime via `catalog.install()`.
> 
> ### What Changed
> 
> - **`agent/routing/catalog.py`** (new): Live sources (OpenRouter, models.dev, AI Model Directory)
> - **`agent/routing/fingerprint.py`** (minimal changes): Now delegates to catalog when installed
> - **`tests/agent/routing/test_catalog.py`** (new): 20 tests for extraction, matching, install
> 
> ### Design Rationale
> 
> - **Separation of concerns**: Routing PR is self-contained (no network deps, no rate limits, no TTL management). Catalog PR can evolve independently.
> - **Graceful degradation**: If catalog fetch fails, builtin catalog (20 models) handles the request.
> - **No breaking changes**: `fingerprint.py` exports unchanged. `catalog.py` monkey-patches internals via `install()`.
> 
> ### Related
> 
> - Depends on: PR #XXX (Routing PR) — for `agent/routing/types.py` contract
> - Builds on: ADR-0040 (Phase 3 routing architecture)

---

## PR Checklist for Both

### Routing PR
- [ ] `agent/routing/types.py` — shared dataclasses (no network deps)
- [ ] `agent/routing/fingerprint.py` — minimal resolver (builtin catalog, no network)
- [ ] `tests/agent/routing/test_fingerprint.py` — all passing
- [ ] PR description includes "Future Work: Live Model Catalog" note (see above)
- [ ] No imports of `catalog` module in routing PR files

### Catalog PR
- [ ] `agent/routing/catalog.py` — live sources, 5min TTL cache
- [ ] `catalog.install()` — monkey-patches `fingerprint._match_catalog` cleanly
- [ ] `tests/agent/routing/test_catalog.py` — 20 tests (all passing)
- [ ] PR description references Routing PR (link to #XXX)
- [ ] PR description explains graceful degradation to builtin catalog
- [ ] No changes to routing PR's `fingerprint.py` *exports* — only internal patching

---

## Merge Order

**Recommended order** (if upstream wants to keep PRs atomic):

1. **Merge Routing PR first** → establishes the contract, minimal resolver, tests
2. **Merge Catalog PR second** → adds live sources, patches routing's minimal resolver

This way:
- Routing is usable immediately (20 models, no network)
- Catalog is additive (no breaking changes, only enhancements)

**Alternative** (if upstream prefers single PR):

- Squash both PRs into one `feat/routing-catalog` branch
- Keep `types.py` + `fingerprint.py` + `catalog.py` together
- Update tests to cover both minimal + live-source behavior
