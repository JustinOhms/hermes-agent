"""Behavioral coverage for the sequential-path tool watchdog backstop.

The sequential execution path has no batch deadline like the concurrent
path's ``_resolve_concurrent_tool_timeout``. ``_invoke_with_watchdog`` is the
last-resort hard-timeout backstop that keeps a wedged barrier/interactive
tool (terminal, execute_code, write_file/patch, a deadlocked approval flow,
an unresponsive MCP server) from hanging the whole turn.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from agent import tool_executor
from agent.tool_executor import (
    _invoke_with_watchdog,
    _resolve_sequential_tool_timeout,
)


def _make_agent():
    return SimpleNamespace(
        _tool_worker_threads=set(),
        _tool_worker_threads_lock=threading.Lock(),
        _interrupt_requested=False,
        log_prefix="",
        _vprint=lambda *a, **k: None,
    )


@pytest.fixture
def interrupt_recorder(monkeypatch):
    """Stub _ra()._set_interrupt so the timeout path is observable in-test."""
    calls = []
    stub = SimpleNamespace(
        _set_interrupt=lambda flag, tid: calls.append((flag, tid))
    )
    monkeypatch.setattr(tool_executor, "_ra", lambda: stub)
    return calls


def test_resolve_reads_env(monkeypatch):
    monkeypatch.setenv("HERMES_SEQUENTIAL_TOOL_TIMEOUT_S", "12.5")
    assert _resolve_sequential_tool_timeout() == 12.5
    # <= 0 disables the backstop
    monkeypatch.setenv("HERMES_SEQUENTIAL_TOOL_TIMEOUT_S", "0")
    assert _resolve_sequential_tool_timeout() is None
    # unset falls back to the 300s default
    monkeypatch.delenv("HERMES_SEQUENTIAL_TOOL_TIMEOUT_S", raising=False)
    assert _resolve_sequential_tool_timeout() == 300.0


def test_fast_tool_returns_result_unchanged(monkeypatch, interrupt_recorder):
    monkeypatch.setenv("HERMES_SEQUENTIAL_TOOL_TIMEOUT_S", "30")
    agent = _make_agent()

    result = _invoke_with_watchdog(agent, "read_file", lambda: "the-result")

    assert result == "the-result"
    assert agent._interrupt_requested is False
    assert interrupt_recorder == []  # watchdog never fired


def test_exception_is_reraised(monkeypatch, interrupt_recorder):
    monkeypatch.setenv("HERMES_SEQUENTIAL_TOOL_TIMEOUT_S", "30")
    agent = _make_agent()

    def _boom():
        raise ValueError("tool blew up")

    with pytest.raises(ValueError, match="tool blew up"):
        _invoke_with_watchdog(agent, "terminal", _boom)
    assert interrupt_recorder == []  # a raised tool is not a timeout


def test_disabled_runs_inline(monkeypatch, interrupt_recorder):
    monkeypatch.setenv("HERMES_SEQUENTIAL_TOOL_TIMEOUT_S", "0")
    agent = _make_agent()

    # No worker thread is spawned when disabled, so a tid-recording invoke
    # runs on the calling thread.
    ran_on = {}

    def _invoke():
        ran_on["tid"] = threading.current_thread().ident
        return "inline"

    result = _invoke_with_watchdog(agent, "terminal", _invoke)
    assert result == "inline"
    assert ran_on["tid"] == threading.current_thread().ident


def test_timeout_returns_error_and_requests_interrupt(monkeypatch, interrupt_recorder):
    monkeypatch.setenv("HERMES_SEQUENTIAL_TOOL_TIMEOUT_S", "0.3")
    agent = _make_agent()

    started = threading.Event()

    def _hang():
        started.set()
        # Longer than the watchdog timeout; the watchdog fires while we sleep.
        # A real wedged native call would ignore the interrupt too — this
        # mirrors the honest "abandon the worker" behavior.
        time.sleep(1.0)
        return "too-late"

    t0 = time.time()
    result = _invoke_with_watchdog(agent, "terminal", _hang)
    elapsed = time.time() - t0

    assert started.is_set()
    assert isinstance(result, str)
    assert result.startswith("Error executing tool 'terminal'")
    assert "watchdog timeout" in result
    # The backstop fired well before the tool's own 1.0s "completion".
    assert elapsed < 5.0
    # It requested interrupt on the abandoned worker.
    assert agent._interrupt_requested is True
    assert any(flag is True for flag, _tid in interrupt_recorder)
