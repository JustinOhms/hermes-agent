# ADR-0045: Two-tier deploy test gate & upstream bulk-suite pollution

**Status:** Accepted — landed 2026-08-04
**Relates:** ADR-0042 (reliable blue/green deployment — this refines its gate 5),
ADR-0044 (runtime-robustness — the sequential-tool watchdog specced there).
**Implementation:** `hermes-config-backup/rescue/hermes-update` (gate 5),
`hermes-config-backup/rescue/pytest_noctty.py`.

## Context

ADR-0042's deploy gate 5 ran a set of test directories, **each whole directory
in one isolated pytest process**, captured `FAILED` node-ids, and blocked the
deploy on any node-id *new* versus a recorded baseline
(`test-baseline-failures.txt`).

Rebuilding the fork onto current upstream (ADR-0043) broke that gate in two
ways, discovered during the 2026-08-04 blue deploy:

1. **The baseline was meaningless after a base change.** It had been recorded on
   the old fork; against current upstream the first run reported 86 "new"
   failures, all in `tests/hermes_cli`. Re-recording did not stabilise it — a
   later run reported 100, now also spanning `tests/run_agent`.

2. **Bulk single-process runs of upstream's integration-heavy suites do not
   reproduce in isolation.** Every one of those "failing" tests **passes when run
   alone** (spot-checked repeatedly: web-server topology 11/11, webhook 14/14, a
   `run_agent` subset 49/49 once the real bug below was fixed). The failures come
   from **cross-test pollution / shared-state coupling** — leaked env vars,
   singletons, ports, cwd, un-torn-down monkeypatches, a shared TestClient/server
   — concentrated in specific areas: the web dashboard/profile server tests,
   web-UI build (spawns real `npm`/`tsc`), webhook, telegram onboarding, and
   venv-health. **The bulk of the codebase (e.g. our routing suite, 218 tests)
   is deterministic.**

   Calibration note: we did **not** prove strict run-to-run non-determinism on
   *identical* inputs — the 86→100 observation is confounded (the baseline was
   re-recorded between runs, and conditions differed). What is firmly
   established is: *deterministic in isolation, unstable under our bulk
   single-process gate*. A large part of this is a **harness/environment
   mismatch** — we run these in a stripped, freshly-built deploy venv, whole-dir
   in one process, without the fixtures/plugins/ordering upstream's own CI uses.
   They may well be reliable in upstream CI. **This is a known and accepted
   limitation, not an upstream indictment.**

3. **Terminal-reading tests hang the gate.** Some `hermes_cli` tests invoke a CLI
   that calls `getpass("Bot token:")`, reading `/dev/tty`. In the foreground the
   gate blocked ~60s per such test (until pytest's SIGALRM timeout); when the run
   was backgrounded, the `/dev/tty` read raised SIGTTIN and **stopped the whole
   process group indefinitely**.

Separately, the deploy surfaced a **real regression** (not pollution) in the
sequential-tool watchdog (ADR-0044): `_invoke_with_watchdog` caught the worker
thread's exception with `except Exception`, which excludes `KeyboardInterrupt`
(a `BaseException`). A tool that raised `KeyboardInterrupt` had it die unhandled
in the worker — `join()` does not surface a worker exception — so the caller got
`None` and the interrupt was swallowed (no cancelled post-tool hook, no
results-for-all-calls). Two `tests/run_agent` interrupt tests caught it; they
pass with the watchdog disabled and fail with it enabled.

## Decision

**Split gate 5 into two tiers.**

- **BLOCKING** — our own feature + guard tests (deterministic; derived from
  `git diff main…patched`), plus the two upstream node-ids the watchdog fix
  guards. Each target runs in its own process. **Any failure aborts the deploy.**
  Fast (seconds). Default set (`BLOCKING_TESTS_DEFAULT`):
  `tests/agent/routing`, `tests/agent/test_sandwich_pipeline`,
  `tests/agent/test_sequential_tool_watchdog.py`, `tests/test_stream_delta_writer.py`,
  and the `test_keyboard_interrupt_emits_cancelled_post_tool_hook` /
  `test_sequential_keyboard_interrupt_emits_results_for_all_calls` node-ids.

- **ADVISORY** — the upstream bulk-flaky dirs (`tests/hermes_cli`,
  `tests/gateway`, `tests/run_agent`, `tests/agent/test_turn_context.py`).
  Baseline-diffed and **reported, but NEVER blocking**. Run via `pytest_noctty.py`
  (a `fork` + `setsid` wrapper that drops the controlling terminal and points
  stdin at `/dev/null`), so a `getpass`-style test **fails fast** instead of
  blocking or wedging. `--no-advisory` skips this tier for a fast deploy;
  `--record-baseline` snapshots it.

New `hermes-update` flags: `--blocking-scope`, `--no-advisory`; `--test-scope`
now names the advisory dirs (back-compat).

**Fix the watchdog** to catch `BaseException` so `KeyboardInterrupt`/`SystemExit`
are stashed and re-raised on the caller thread (commit `961ebd805`,
consolidated in `8bb1e7760`).

## Consequences

- **Deploys block on signal we trust** (our deterministic tests) and are no
  longer wedged or falsely-blocked by upstream bulk-suite noise. Blue deployed
  clean in ~21s with `--no-advisory`.
- **The advisory tier is reliable but slow** (~5 min: `hermes_cli` runs real
  subprocess/build tests), so routine deploys use `--no-advisory`. This
  slowness is accepted and documented — not a regression to chase.
- **We give up automated bulk coverage of upstream's integration suites at
  deploy time.** Acceptable: they are additive upstream features (much unused
  here, e.g. the web dashboard), every test is checkable in isolation, and the
  app's boot/import is still verified (gate 6). Run the advisory tier on demand
  when upstream areas matter.
- **If we ever want real per-test isolation** for those dirs (to reclaim
  blocking coverage), the follow-up is per-test forking (e.g. `pytest-forked`),
  which is heavier and out of scope here.
