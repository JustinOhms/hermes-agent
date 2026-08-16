"""Wiring guard for the runtime-robustness max_iterations hot-reload patch.

The fork keeps ONE concern from ADR-0044 after the upstream #85125 sequential-
tool timeout superseded the watchdog: hot-reloading ``agent.max_iterations`` from
``config.yaml`` once per turn so a mid-session ``agent.max_turns`` bump takes
effect without a ``/new``.  These tests fail loudly if a future upstream resync
drops either half of the wiring (the ``_refresh_max_iterations`` helper in
conversation_loop, or its call site in build_turn_context).
"""
from types import SimpleNamespace

import agent.conversation_loop as conversation_loop
from agent.conversation_loop import _refresh_max_iterations


def _fake_agent(**kw):
    base = dict(max_iterations=120, session_id="test", _cfg_max_turns_mtime=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_reads_updated_max_turns_from_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("agent:\n  max_turns: 1000\n")
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path, raising=False
    )
    monkeypatch.delenv("HERMES_TUI_MAX_TURNS", raising=False)
    agent = _fake_agent()
    assert _refresh_max_iterations(agent) == 1000


def test_env_var_takes_priority(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("agent:\n  max_turns: 1000\n")
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path, raising=False
    )
    monkeypatch.setenv("HERMES_TUI_MAX_TURNS", "555")
    agent = _fake_agent()
    assert _refresh_max_iterations(agent) == 555


def test_missing_config_falls_back_to_current(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path, raising=False
    )
    monkeypatch.delenv("HERMES_TUI_MAX_TURNS", raising=False)
    agent = _fake_agent(max_iterations=77)
    assert _refresh_max_iterations(agent) == 77


def test_never_raises_on_bad_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("this: [is: not: valid: yaml")
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path, raising=False
    )
    monkeypatch.delenv("HERMES_TUI_MAX_TURNS", raising=False)
    agent = _fake_agent(max_iterations=42)
    # A broken config must never crash the conversation loop.
    assert _refresh_max_iterations(agent) == 42


def test_call_site_present_in_build_turn_context():
    """build_turn_context must call _refresh_max_iterations before building the
    iteration budget — the wiring that makes the hot-reload actually fire."""
    import inspect
    import agent.turn_context as turn_context

    src = inspect.getsource(turn_context.build_turn_context)
    assert "_refresh_max_iterations" in src, (
        "build_turn_context no longer calls _refresh_max_iterations — the "
        "max_iterations hot-reload wiring was dropped (likely by a resync)."
    )
    # The refresh must precede the budget construction, or it has no effect.
    assert src.index("_refresh_max_iterations") < src.index(
        "IterationBudget(agent.max_iterations)"
    )
