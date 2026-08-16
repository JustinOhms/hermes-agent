"""Tests for agent/routing/swap_manager.py."""

from __future__ import annotations

import time
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.routing.config import (
    ComplexityConfig,
    DeEscalationConfig,
    GraphPosition,
    GraphPositionProfile,
    InteractionModeConfig,
    RoutingConfig,
)
from agent.routing.interaction_mode import InteractionMode
from agent.routing.swap_manager import (
    SwapManager,
    SwapResult,
    SwapState,
    _start_local_model,
)
from agent.routing.turn_router import RoutingDecision


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_config(
    local_positions: dict | None = None,
    cloud_positions: dict | None = None,
) -> RoutingConfig:
    """Build a RoutingConfig with named graph positions."""
    graph: dict[str, GraphPosition] = {}
    for name, cfg_name in (local_positions or {}).items():
        graph[name] = GraphPosition(
            provider="custom:llm-local",
            model=f"model-{name}",
            base_url="http://127.0.0.1:58080/v1",
            api_mode="chat_completions",
            llm_config_name=cfg_name,
            profile=GraphPositionProfile(ttft_p90_ms=800.0, generation_tok_s=60.0),
        )
    for name, (provider, model) in (cloud_positions or {}).items():
        graph[name] = GraphPosition(
            provider=provider,
            model=model,
            base_url="",
            api_mode="anthropic_messages",
        )
    return RoutingConfig(
        enabled=True,
        graph=graph,
        interaction_mode=InteractionModeConfig(
            swap_back_messages=3,
            swap_back_window_s=60,
        ),
    )


def _make_decision(
    target: str,
    mode: InteractionMode = InteractionMode.INTERACTIVE,
    swap_required: bool = True,
) -> RoutingDecision:
    return RoutingDecision(
        target_position=target,
        complexity_score=0.3,
        interaction_mode=mode,
        reason="test",
        swap_required=swap_required,
    )


def _make_detector(sustained: bool = False) -> MagicMock:
    detector = MagicMock()
    detector.sustained_engagement_detected.return_value = sustained
    return detector


# ── SwapState enum ────────────────────────────────────────────────────────────

class TestSwapStateEnum:
    def test_all_states_present(self):
        states = {s.value for s in SwapState}
        assert states == {"idle", "awaiting_engagement", "swap_requested", "swapping", "ready", "failed"}


# ── SwapManager initialization ────────────────────────────────────────────────

class TestSwapManagerInit:
    def test_initial_state_is_idle(self):
        cfg = _make_config()
        mgr = SwapManager(cfg)
        assert mgr.state == SwapState.IDLE

    def test_initial_position_is_none(self):
        cfg = _make_config()
        mgr = SwapManager(cfg)
        assert mgr.current_position is None

    def test_set_current_position(self):
        cfg = _make_config(local_positions={"interactive_lower": "coder"})
        mgr = SwapManager(cfg)
        mgr.set_current_position("interactive_lower")
        assert mgr.current_position == "interactive_lower"
        assert mgr.state == SwapState.IDLE


# ── should_swap logic ─────────────────────────────────────────────────────────

class TestShouldSwap:
    def test_no_swap_when_current_position_unknown(self):
        cfg = _make_config(local_positions={"interactive_lower": "coder"})
        mgr = SwapManager(cfg)
        # current_position is None
        decision = _make_decision("interactive_lower")
        assert mgr.should_swap(decision, _make_detector()) is False

    def test_no_swap_when_target_equals_current(self):
        cfg = _make_config(local_positions={"interactive_lower": "coder"})
        mgr = SwapManager(cfg)
        mgr.set_current_position("interactive_lower")
        decision = _make_decision("interactive_lower")
        assert mgr.should_swap(decision, _make_detector()) is False

    def test_escalation_to_upper_swaps_immediately(self):
        cfg = _make_config(
            local_positions={"interactive_lower": "coder"},
            cloud_positions={"upper": ("bedrock", "claude-opus")},
        )
        mgr = SwapManager(cfg)
        mgr.set_current_position("interactive_lower")
        decision = _make_decision("upper")
        assert mgr.should_swap(decision, _make_detector()) is True

    def test_different_local_positions_swap_immediately(self):
        cfg = _make_config(
            local_positions={
                "interactive_lower": "coder",
                "autonomous_lower": "prose",
            }
        )
        mgr = SwapManager(cfg)
        mgr.set_current_position("autonomous_lower")
        # Interactive mode moving to interactive_lower (not an auto→interactive)
        decision = _make_decision("interactive_lower", mode=InteractionMode.INTERACTIVE)
        # "autonomous" is in "autonomous_lower" so this is auto→interactive
        # First call → AWAITING_ENGAGEMENT, returns False
        result = mgr.should_swap(decision, _make_detector(sustained=False))
        assert result is False
        assert mgr.state == SwapState.AWAITING_ENGAGEMENT

    def test_lazy_swapback_first_interactive_message_no_swap(self):
        cfg = _make_config(
            local_positions={
                "autonomous_lower": "prose",
                "interactive_lower": "coder",
            }
        )
        mgr = SwapManager(cfg)
        mgr.set_current_position("autonomous_lower")
        decision = _make_decision("interactive_lower", mode=InteractionMode.INTERACTIVE)
        result = mgr.should_swap(decision, _make_detector(sustained=False))
        assert result is False
        assert mgr.state == SwapState.AWAITING_ENGAGEMENT

    def test_lazy_swapback_sustained_engagement_triggers_swap(self):
        cfg = _make_config(
            local_positions={
                "autonomous_lower": "prose",
                "interactive_lower": "coder",
            }
        )
        mgr = SwapManager(cfg)
        mgr.set_current_position("autonomous_lower")
        decision = _make_decision("interactive_lower", mode=InteractionMode.INTERACTIVE)

        # Turn 1: enter AWAITING_ENGAGEMENT
        mgr.should_swap(decision, _make_detector(sustained=False))
        assert mgr.state == SwapState.AWAITING_ENGAGEMENT

        # Turn 2: sustained engagement detected → swap
        result = mgr.should_swap(decision, _make_detector(sustained=True))
        assert result is True
        assert mgr.state == SwapState.SWAP_REQUESTED

    def test_failed_state_resets_to_idle_on_next_call(self):
        cfg = _make_config(
            local_positions={
                "interactive_lower": "coder",
                "autonomous_lower": "prose",
            }
        )
        mgr = SwapManager(cfg)
        mgr.set_current_position("interactive_lower")
        # Manually set failed state
        with mgr._lock:
            mgr._state = SwapState.FAILED
        decision = _make_decision("autonomous_lower")
        # should_swap resets FAILED → IDLE then evaluates normally
        result = mgr.should_swap(decision, _make_detector())
        assert mgr.state != SwapState.FAILED


# ── resolve_effective_model ───────────────────────────────────────────────────

class TestResolveEffectiveModel:
    def test_cloud_target_returns_cloud_directly(self):
        cfg = _make_config(
            local_positions={"interactive_lower": "coder"},
            cloud_positions={"upper": ("bedrock", "claude-opus")},
        )
        mgr = SwapManager(cfg)
        mgr.set_current_position("interactive_lower")
        decision = _make_decision("upper")
        resolved = mgr.resolve_effective_model(decision)
        assert resolved is not None
        assert resolved.is_local is False
        assert resolved.provider == "bedrock"

    def test_already_loaded_local_returns_target(self):
        cfg = _make_config(local_positions={"interactive_lower": "coder"})
        mgr = SwapManager(cfg)
        mgr.set_current_position("interactive_lower")
        decision = _make_decision("interactive_lower")
        resolved = mgr.resolve_effective_model(decision)
        assert resolved is not None
        assert resolved.is_local is True
        assert resolved.llm_config_name == "coder"

    def test_swapping_state_returns_upper_cloud_cover(self):
        cfg = _make_config(
            local_positions={
                "autonomous_lower": "prose",
                "interactive_lower": "coder",
            },
            cloud_positions={"upper": ("bedrock", "claude-opus")},
        )
        mgr = SwapManager(cfg)
        mgr.set_current_position("autonomous_lower")
        with mgr._lock:
            mgr._state = SwapState.SWAPPING
        decision = _make_decision("interactive_lower")
        resolved = mgr.resolve_effective_model(decision)
        assert resolved is not None
        assert resolved.is_local is False
        assert resolved.provider == "bedrock"

    def test_awaiting_engagement_returns_current_model(self):
        cfg = _make_config(
            local_positions={
                "autonomous_lower": "prose",
                "interactive_lower": "coder",
            },
            cloud_positions={"upper": ("bedrock", "claude-opus")},
        )
        mgr = SwapManager(cfg)
        mgr.set_current_position("autonomous_lower")
        with mgr._lock:
            mgr._state = SwapState.AWAITING_ENGAGEMENT
        decision = _make_decision("interactive_lower")
        resolved = mgr.resolve_effective_model(decision)
        assert resolved is not None
        assert resolved.llm_config_name == "prose"  # current model

    def test_unknown_target_returns_none(self):
        cfg = _make_config(local_positions={"interactive_lower": "coder"})
        mgr = SwapManager(cfg)
        mgr.set_current_position("interactive_lower")
        decision = _make_decision("nonexistent_position")
        resolved = mgr.resolve_effective_model(decision)
        assert resolved is None


# ── execute_swap_background ───────────────────────────────────────────────────

class TestExecuteSwapBackground:
    def test_success_updates_position_and_state(self):
        cfg = _make_config(local_positions={"interactive_lower": "coder"})
        mgr = SwapManager(cfg)
        mgr.set_current_position("autonomous_lower_stub")  # different position

        with patch("agent.routing.swap_manager._start_local_model") as mock_start:
            mock_start.return_value = SwapResult(success=True, startup_time_s=5.0)
            # Start background swap
            done = threading.Event()
            original_start = mgr.execute_swap_background.__func__ if hasattr(mgr.execute_swap_background, '__func__') else None

            mgr.execute_swap_background("interactive_lower")
            # Give thread time to complete
            time.sleep(0.1)

        assert mgr.current_position == "interactive_lower"
        assert mgr.state == SwapState.READY

    def test_failure_sets_failed_state(self):
        cfg = _make_config(local_positions={"interactive_lower": "coder"})
        mgr = SwapManager(cfg)
        mgr.set_current_position("some_other")

        with patch("agent.routing.swap_manager._start_local_model") as mock_start:
            mock_start.return_value = SwapResult(success=False, error="timeout")
            mgr.execute_swap_background("interactive_lower")
            time.sleep(0.1)

        assert mgr.state == SwapState.FAILED
        assert mgr.current_position == "some_other"  # unchanged

    def test_unknown_position_is_no_op(self):
        cfg = _make_config(local_positions={"interactive_lower": "coder"})
        mgr = SwapManager(cfg)
        mgr.set_current_position("interactive_lower")
        # Should not crash or change state
        mgr.execute_swap_background("does_not_exist")
        assert mgr.state == SwapState.IDLE

    def test_no_llm_config_name_is_no_op(self):
        cfg = _make_config(
            cloud_positions={"upper": ("bedrock", "claude-opus")},
        )
        mgr = SwapManager(cfg)
        mgr.set_current_position("upper")
        mgr.execute_swap_background("upper")
        assert mgr.state == SwapState.IDLE


# ── execute_swap_sync ─────────────────────────────────────────────────────────

class TestExecuteSwapSync:
    def test_success(self):
        cfg = _make_config(local_positions={"interactive_lower": "coder"})
        mgr = SwapManager(cfg)

        with patch("agent.routing.swap_manager._start_local_model") as mock_start:
            mock_start.return_value = SwapResult(success=True, startup_time_s=3.0)
            result = mgr.execute_swap_sync("interactive_lower")

        assert result.success is True
        assert mgr.current_position == "interactive_lower"
        assert mgr.state == SwapState.READY

    def test_failure_sets_failed_state(self):
        cfg = _make_config(local_positions={"interactive_lower": "coder"})
        mgr = SwapManager(cfg)

        with patch("agent.routing.swap_manager._start_local_model") as mock_start:
            mock_start.return_value = SwapResult(success=False, error="no such model")
            result = mgr.execute_swap_sync("interactive_lower")

        assert result.success is False
        assert mgr.state == SwapState.FAILED

    def test_unknown_position_returns_error(self):
        cfg = _make_config()
        mgr = SwapManager(cfg)
        result = mgr.execute_swap_sync("nonexistent")
        assert result.success is False
        assert "Unknown or non-local" in result.error


# ── _start_local_model ────────────────────────────────────────────────────────

class TestStartLocalModel:
    def test_success(self):
        with patch("agent.routing.swap_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            result = _start_local_model("coder-next")
        assert result.success is True
        assert result.startup_time_s >= 0

    def test_non_zero_exit_is_failure(self):
        with patch("agent.routing.swap_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="model not found", stdout="")
            result = _start_local_model("bad-model")
        assert result.success is False
        assert "model not found" in result.error

    def test_timeout_is_failure(self):
        import subprocess
        with patch("agent.routing.swap_manager.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["llm"], timeout=30)
            result = _start_local_model("slow-model")
        assert result.success is False
        assert "timed out" in result.error

    def test_file_not_found_is_failure(self):
        with patch("agent.routing.swap_manager.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("llm not found")
            result = _start_local_model("any-model")
        assert result.success is False
        assert "not found" in result.error
