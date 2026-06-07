"""Phase 2 integration tests — routing decision → swap → effective model."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.routing import execute_routing_swap, get_routing_decision
from agent.routing.config import (
    GraphPosition,
    GraphPositionProfile,
    RoutingConfig,
)
from agent.routing.interaction_mode import InteractionMode, InteractionModeDetector
from agent.routing.model_resolver import ResolvedModel
from agent.routing.swap_manager import SwapManager, SwapResult, SwapState
from agent.routing.turn_router import RoutingDecision


# ── Helpers ──────────────────────────────────────────────────────────────────

def _local_pos(llm_config_name: str, model: str = "") -> GraphPosition:
    return GraphPosition(
        provider="custom:llm-local",
        model=model or f"model-{llm_config_name}",
        base_url="http://127.0.0.1:58080/v1",
        api_mode="chat_completions",
        llm_config_name=llm_config_name,
        profile=GraphPositionProfile(ttft_p90_ms=800.0, generation_tok_s=60.0),
    )


def _cloud_pos(provider: str = "bedrock", model: str = "claude-opus") -> GraphPosition:
    return GraphPosition(
        provider=provider,
        model=model,
        base_url="",
        api_mode="anthropic_messages",
        llm_config_name="",
    )


def _make_full_config() -> RoutingConfig:
    return RoutingConfig(
        enabled=True,
        graph={
            "interactive_lower": _local_pos("coder", model="qwen3-coder"),
            "autonomous_lower": _local_pos("prose", model="qwen3-prose"),
            "upper": _cloud_pos("bedrock", "claude-opus-4"),
            "fast_fallback": _local_pos("fast", model="qwen3-fast"),
        },
    )


def _make_agent(
    provider: str = "custom:llm-local",
    model: str = "qwen3-coder",
    **kwargs,
) -> SimpleNamespace:
    defaults = {
        "provider": provider,
        "model": model,
        "_turn_count": 1,
        "is_cron": False,
        "is_subagent": False,
        "platform": "cli",
        "_recent_tool_calls": [],
        "_last_response_had_errors": False,
        "_explicit_mode_override": None,
        "_routing_mode_detector": None,
        "_routing_swap_manager": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


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


def _preloaded_swap_mgr(config: RoutingConfig, position: str) -> SwapManager:
    mgr = SwapManager(config)
    mgr.set_current_position(position)
    return mgr


# ── get_routing_decision: swap_required flag (Phase 2) ───────────────────────

class TestSwapRequiredFlag:
    def test_swap_required_false_when_no_graph_configured(self):
        """No graph → current_position stays None → swap_required stays False."""
        agent = _make_agent()
        cfg = RoutingConfig(enabled=True, graph={})
        with patch("agent.routing.load_routing_config", return_value=cfg):
            decision = get_routing_decision(agent, "hello")
        assert decision is not None
        assert decision.swap_required is False

    def test_swap_required_false_when_target_matches_current_position(self):
        cfg = _make_full_config()
        agent = _make_agent(provider="custom:llm-local", model="qwen3-coder")
        # SwapManager will init current_position = "interactive_lower" from agent.model
        with patch("agent.routing.load_routing_config", return_value=cfg):
            decision = get_routing_decision(agent, "hello")
        assert decision is not None
        # Simple "hello" routes to interactive_lower; current is interactive_lower → no swap
        assert decision.target_position == "interactive_lower"
        assert decision.swap_required is False

    def test_swap_required_true_when_target_differs_from_current(self):
        cfg = _make_full_config()
        # Agent is currently on autonomous_lower model
        agent = _make_agent(provider="custom:llm-local", model="qwen3-prose")
        # A complex message escalates to "upper"
        complex_msg = "explain the architecture tradeoffs of this refactoring and why the race condition vulnerability requires a different approach across multiple files"
        with patch("agent.routing.load_routing_config", return_value=cfg):
            decision = get_routing_decision(agent, complex_msg)
        assert decision is not None
        if decision.target_position != "autonomous_lower":
            assert decision.swap_required is True

    def test_swap_required_true_on_model_mismatch_after_cloud_cover(self):
        """After a cloud cover turn, agent is on upper but routing position is interactive_lower.
        Next turn: same target, different agent model → swap_required=True."""
        cfg = _make_full_config()
        mgr = _preloaded_swap_mgr(cfg, "interactive_lower")
        mgr._state = SwapState.READY  # local model finished loading

        # Agent is on cloud (cover turn happened)
        agent = _make_agent(provider="bedrock", model="claude-opus-4")
        agent._routing_swap_manager = mgr

        with patch("agent.routing.load_routing_config", return_value=cfg):
            decision = get_routing_decision(agent, "ok")

        assert decision is not None
        # target is interactive_lower, current_position is interactive_lower,
        # but agent.provider/model is cloud → mismatch → swap_required=True
        assert decision.swap_required is True


# ── execute_routing_swap ──────────────────────────────────────────────────────

class TestExecuteRoutingSwap:
    def test_routing_disabled_returns_none(self):
        agent = _make_agent()
        decision = _make_decision("interactive_lower")
        cfg = RoutingConfig(enabled=False)
        with patch("agent.routing.load_routing_config", return_value=cfg):
            result = execute_routing_swap(agent, decision)
        assert result is None

    def test_should_swap_false_returns_none(self):
        """No position change and no mismatch → no swap."""
        cfg = _make_full_config()
        mgr = _preloaded_swap_mgr(cfg, "interactive_lower")
        agent = _make_agent(provider="custom:llm-local", model="qwen3-coder")
        agent._routing_swap_manager = mgr

        decision = _make_decision("interactive_lower")
        with patch("agent.routing.load_routing_config", return_value=cfg):
            result = execute_routing_swap(agent, decision)
        assert result is None

    def test_cloud_target_returns_cloud_model_directly(self):
        cfg = _make_full_config()
        mgr = _preloaded_swap_mgr(cfg, "interactive_lower")
        agent = _make_agent(provider="custom:llm-local", model="qwen3-coder")
        agent._routing_swap_manager = mgr

        decision = _make_decision("upper")
        with patch("agent.routing.load_routing_config", return_value=cfg):
            result = execute_routing_swap(agent, decision)

        assert result is not None
        assert result.is_local is False
        assert result.provider == "bedrock"
        assert result.model == "claude-opus-4"

    def test_local_swap_triggers_background_and_returns_cloud_cover(self):
        """Switching to a different local model → cloud covers this turn, swap in background.

        We patch should_swap=True to isolate the cloud-cover+background-swap path
        from the lazy-swap-back logic (which would normally intercept this
        autonomous→interactive transition on its first call).
        """
        cfg = _make_full_config()
        mgr = _preloaded_swap_mgr(cfg, "autonomous_lower")
        agent = _make_agent(provider="custom:llm-local", model="qwen3-prose")
        agent._routing_swap_manager = mgr

        cloud_cover = ResolvedModel(
            provider="bedrock", model="claude-opus-4", base_url="",
            api_key="", api_mode="anthropic_messages",
            is_local=False, llm_config_name="",
        )

        decision = _make_decision("interactive_lower")
        with patch("agent.routing.load_routing_config", return_value=cfg):
            with patch.object(mgr, "should_swap", return_value=True):
                with patch.object(mgr, "execute_swap_background") as mock_bg:
                    with patch.object(mgr, "resolve_effective_model", return_value=cloud_cover):
                        result = execute_routing_swap(agent, decision)

        assert result is not None
        assert result.is_local is False  # cloud cover
        mock_bg.assert_called_once_with("interactive_lower")

    def test_swap_already_in_progress_no_duplicate_background_swap(self):
        """When SWAPPING, don't start another background swap."""
        cfg = _make_full_config()
        mgr = _preloaded_swap_mgr(cfg, "autonomous_lower")
        with mgr._lock:
            mgr._state = SwapState.SWAPPING
        agent = _make_agent(provider="custom:llm-local", model="qwen3-prose")
        agent._routing_swap_manager = mgr

        decision = _make_decision("interactive_lower")
        with patch("agent.routing.load_routing_config", return_value=cfg):
            with patch.object(mgr, "execute_swap_background") as mock_bg:
                execute_routing_swap(agent, decision)

        mock_bg.assert_not_called()

    def test_lazy_swapback_first_interactive_message_returns_none(self):
        """First message after autonomous period → lazy wait, no swap, no model change."""
        cfg = _make_full_config()
        mgr = _preloaded_swap_mgr(cfg, "autonomous_lower")
        agent = _make_agent(provider="custom:llm-local", model="qwen3-prose")
        agent._routing_swap_manager = mgr

        # Simulate InteractionModeDetector with no sustained engagement
        detector = MagicMock(spec=InteractionModeDetector)
        detector.sustained_engagement_detected.return_value = False
        agent._routing_mode_detector = detector

        decision = _make_decision("interactive_lower", mode=InteractionMode.INTERACTIVE)
        with patch("agent.routing.load_routing_config", return_value=cfg):
            result = execute_routing_swap(agent, decision)

        assert result is None
        assert mgr.state == SwapState.AWAITING_ENGAGEMENT

    def test_lazy_swapback_sustained_engagement_triggers_swap(self):
        """3 messages in 60s → sustained engagement → swap triggered."""
        cfg = _make_full_config()
        mgr = _preloaded_swap_mgr(cfg, "autonomous_lower")
        # Already in AWAITING_ENGAGEMENT state
        with mgr._lock:
            mgr._state = SwapState.AWAITING_ENGAGEMENT
            mgr._pending_position = "interactive_lower"

        agent = _make_agent(provider="custom:llm-local", model="qwen3-prose")
        agent._routing_swap_manager = mgr

        detector = MagicMock(spec=InteractionModeDetector)
        detector.sustained_engagement_detected.return_value = True
        agent._routing_mode_detector = detector

        decision = _make_decision("interactive_lower", mode=InteractionMode.INTERACTIVE)
        with patch("agent.routing.load_routing_config", return_value=cfg):
            with patch.object(mgr, "execute_swap_background"):
                result = execute_routing_swap(agent, decision)

        # Should return cloud cover for the turn when swap is triggered
        # (either cloud cover or the target; depends on graph config)
        # The swap was requested so state transitions
        assert mgr.state in (SwapState.SWAP_REQUESTED, SwapState.SWAPPING)

    def test_return_from_cloud_cover_switches_to_local(self):
        """After cloud-cover turn: swap_mgr.current = interactive_lower (loaded),
        agent is on cloud → execute_routing_swap returns the local model."""
        cfg = _make_full_config()
        mgr = _preloaded_swap_mgr(cfg, "interactive_lower")
        with mgr._lock:
            mgr._state = SwapState.READY

        # Agent is on cloud (cover model from last turn)
        agent = _make_agent(provider="bedrock", model="claude-opus-4")
        agent._routing_swap_manager = mgr

        decision = _make_decision("interactive_lower", swap_required=True)
        with patch("agent.routing.load_routing_config", return_value=cfg):
            result = execute_routing_swap(agent, decision)

        assert result is not None
        assert result.is_local is True
        assert result.model == "qwen3-coder"

    def test_error_returns_none_gracefully(self):
        """Any error in execute_routing_swap should return None (non-fatal)."""
        agent = _make_agent()
        decision = _make_decision("interactive_lower")
        with patch("agent.routing.load_routing_config", side_effect=RuntimeError("disk error")):
            result = execute_routing_swap(agent, decision)
        assert result is None
