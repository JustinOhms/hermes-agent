# Routing + Catalog PR Reference

## PR Numbers (fill in when PRs are opened)

| PR | Branch | Status | Description |
|----|--------|--------|-------------|
| #XXX | `feat/routing` | Open/Ready | Routing PR (minimal resolver + contract) |
| #YYY | `feat/catalog` | Open/Ready | Catalog PR (live sources + monkey-patch) |

## How They Relate

1. **Routing PR (#XXX)** is the *contract* — dataclasses, minimal resolver, tests
2. **Catalog PR (#YYY)** is the *implementation* — live sources, patches routing's resolver
3. Both PRs are *independent* — catalog can be reviewed/merged after routing
4. **No breaking changes** — catalog only monkey-patches internals, same public API

## Merge Order (Recommended)

1. **Merge Routing PR first** → establishes contract, minimal resolver, tests
2. **Merge Catalog PR second** → adds live sources, patches minimal resolver

This gives upstream:
- Immediate value from routing PR (20 models, no network)
- Incremental addition of live catalog without blocking

## When Opening PRs

### Routing PR Checklist

- [ ] Add `Fixes #ISSUE` line
- [ ] Add `Depends on: #YYY (Catalog PR)` to "Related PRs" section
- [ ] Include note: "Catalog PR adds live model catalog resolution"

### Catalog PR Checklist

- [ ] Add `Fixes #ISSUE` line
- [ ] Add `Depends on: #XXX (Routing PR)` to "Related PRs" section
- [ ] Include note: "Builds on routing PR, adds live catalog"

## Documentation Files

- `/docs/ROUTING_CATALOG_PR_LINKS.md` — this file (fill in PR numbers)
- `/docs/routing-catalog-pr-guide.md` — detailed PR relationship guide
- `/.github/PULL_REQUEST_TEMPLATE.md` — updated template (includes catalog note)
