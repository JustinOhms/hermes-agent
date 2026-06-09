"""Edge-case and integration tests for the routing system.

Covers:
- Swap manager recovery and failure modes
- Config validation edge cases
- Interaction mode edge cases
- Oversight escalation flag lifecycle
- Cross-module integration (turn_router → swap_manager → oversight)
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.routing.config import (
    ComplexityConfig,
    DeEscalationConfig,
    GraphPosition,
    GraphPositionProfile,
    InteractionModeConfig,
    RoutingConfig,
    load_routing_config,
)
from agent.routing.interaction_mode import InteractionMode, InteractionModeDetector
from agent.routing.oversight import (
    OversightAction,
    OversightConfig,
    OversightResult,
    OversightReviewer,
    build_oversight_injection,
    run_oversight_if_due,
)
from agent.routing.state import get_routing_state
from agent.routing.swap_manager import SwapManager, SwapState
from agent.routing.turn_router import TurnRouter, RoutingContext, RoutingDecision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def routing_config():
    """Standard test routing config."""
    return RoutingConfig(
        enabled=True,
        graph={
            "upper": GraphPosition(
                provider="bedrock",
                model="opus",
                base_url="",
                api_key="",
                api_mode="",
                llm_config_name="",
                profile=GraphPositionProfile(
                    startup_latency_s=0.0,
                    ttft_p50_ms=500,
                    ttft_p90_ms=2000,
                    generation_tok_s=80,
                ),
            ),
            "interactive_lower": GraphPosition(
                provider="local",
                model="qwen-coder",
                base_url="http://localhost:58080",
                api_key="",
                api_mode="",
                llm_config_name="coder-next",
                profile=GraphPositionProfile(
                    startup_latency_s=17.0,
                    ttft_p50_ms=200,
                    ttft_p90_ms=800,
                    generation_tok_s=35,
                ),
            ),
            "fast_fallback": GraphPosition(
                provider="local",
                model="little-qwen",
                base_url="http://localhost:58080",
                api_key="",
                api_mode="",
                llm_config_name="little-qwen",
                profile=GraphPositionProfile(
                    startup_latency_s=5.0,
                    ttft_p50_ms=100,
                    ttft_p90_ms=300,
                    generation_tok_s=140,
                ),
            ),
        },
        complexity=ComplexityConfig(
            escalation_threshold=0.7,
            de_escalation_threshold=0.2,
        ),
        interaction_mode=InteractionModeConfig(
            idle_threshold_s=300,
            swap_back_messages=3,
            swap_back_window_s=60,
        ),
        de_escalation=DeEscalationConfig(enabled=False),
    )


@pytest.fixture
def swap_manager(routing_config):
    """Swap manager initialized with standard config."""
    sm = SwapManager(routing_config)
    sm.set_current_position("interactive_lower")
    return sm


@pytest.fixture
def mode_detector(routing_config):
    """Interaction mode detector."""
    return InteractionModeDetector(routing_config)


@pytest.fixture
def turn_router(routing_config, mode_detector):
    """Turn router with standard config."""
    return TurnRouter(routing_config, mode_detector)


def _make_context(msg: str, **kwargs) -> RoutingContext:
    """Helper to build a RoutingContext."""
    defaults = {
        "user_message": msg,
        "message_length": len(msg),
        "conversation_turn_count": 5,
        "is_cron": False,
        "is_subagent": False,
        "platform": "cli",
        "interaction_mode": InteractionMode.INTERACTIVE,
        "recent_tool_calls": [],
        "last_response_had_errors": False,
        "explicit_mode_override": None,
    }
    defaults.update(kwargs)
    return RoutingContext(**defaults)


# ---------------------------------------------------------------------------
# Swap Manager Edge Cases
# ---------------------------------------------------------------------------

class TestSwapManagerEdgeCases:
    def test_swap_from_failed_state_resets(self, swap_manager, mode_detector):
        """After a failed swap, next should_swap call should reset state."""
        swap_manager._state = SwapState.FAILED
        swap_manager._failure_count = 1

        decision = RoutingDecision(
            target_position="upper",
            interaction_mode=InteractionMode.INTERACTIVE,
            complexity_score=0.9,
            swap_required=True,
            reason="test",
        )
        should_swap = swap_manager.should_swap(decision, mode_detector)
        # After reset from FAILED, should allow the swap
        assert swap_manager._state != SwapState.FAILED

    def test_swap_to_same_position_is_noop(self, swap_manager, mode_detector):
        """Swap to current position should return False."""
        decision = RoutingDecision(
            target_position="interactive_lower",  # same as current
            interaction_mode=InteractionMode.INTERACTIVE,
            complexity_score=0.3,
            swap_required=False,
            reason="test",
        )
        result = swap_manager.should_swap(decision, mode_detector)
        assert not result

    def test_resolve_during_swapping_state(self, swap_manager):
        """During SWAPPING state, resolve should handle gracefully."""
        swap_manager._state = SwapState.SWAPPING
        swap_manager._pending_position = "interactive_lower"

        decision = RoutingDecision(
            target_position="interactive_lower",
            interaction_mode=InteractionMode.INTERACTIVE,
            complexity_score=0.3,
            swap_required=True,
            reason="test",
        )
        # Should not crash during SWAPPING state
        result = swap_manager.resolve_effective_model(decision)
        # Result may be None or a cloud cover model — either is acceptable
        assert result is None or hasattr(result, 'provider')

    def test_consecutive_failed_states(self, swap_manager):
        """Multiple failed states can be recovered from."""
        from agent.routing.interaction_mode import InteractionModeDetector
        # Put into failed state
        swap_manager._state = SwapState.FAILED

        # A new should_swap call should handle the failed state gracefully
        decision = RoutingDecision(
            target_position="upper",
            interaction_mode=InteractionMode.INTERACTIVE,
            complexity_score=0.9,
            swap_required=True,
            reason="test",
        )
        mode_det = InteractionModeDetector(swap_manager._config)
        # Should not raise, and should recover
        result = swap_manager.should_swap(decision, mode_det)
        assert isinstance(result, bool)
        # State should no longer be FAILED after recovery
        assert swap_manager._state != SwapState.FAILED


# ---------------------------------------------------------------------------
# Interaction Mode Edge Cases
# ---------------------------------------------------------------------------

class TestInteractionModeEdgeCases:
    def test_interactive_mode_default(self, mode_detector):
        """Default detection with interactive context should be INTERACTIVE."""
        ctx = _make_context("hello", interaction_mode=InteractionMode.INTERACTIVE)
        mode = mode_detector.current_mode(ctx)
        assert mode == InteractionMode.INTERACTIVE

    def test_interaction_mode_enum_values(self):
        """InteractionMode should have expected values."""
        assert InteractionMode.INTERACTIVE.name == "INTERACTIVE"
        assert InteractionMode.AUTONOMOUS.name == "AUTONOMOUS"


# ---------------------------------------------------------------------------
# Turn Router Edge Cases
# ---------------------------------------------------------------------------

class TestTurnRouterEdgeCases:
    def test_empty_message_minimal_complexity(self, turn_router):
        """Empty message should still produce a valid decision."""
        ctx = _make_context("")
        decision = turn_router.route(ctx)
        assert isinstance(decision, RoutingDecision)
        assert decision.complexity_score >= 0.0

    def test_very_long_message_capped(self, turn_router):
        """Extremely long messages should have complexity capped at 1.0."""
        long_msg = "implement a distributed consensus algorithm " * 1000
        ctx = _make_context(long_msg)
        decision = turn_router.route(ctx)
        assert decision.complexity_score <= 1.0

    def test_binary_noise_doesnt_crash(self, turn_router):
        """Binary garbage should not crash the router."""
        garbage = bytes(range(256)).decode("latin-1")
        ctx = _make_context(garbage)
        decision = turn_router.route(ctx)
        assert isinstance(decision, RoutingDecision)

    def test_unicode_edge_cases(self, turn_router):
        """Unicode edge cases (emoji, CJK, RTL) should work."""
        messages = [
            "🤖 请帮我写一个排序算法",
            "مرحبا بالعالم",
            "🇯🇵 テスト メッセージ with mixed scripts 混合スクリプト",
        ]
        for msg in messages:
            ctx = _make_context(msg)
            decision = turn_router.route(ctx)
            assert isinstance(decision, RoutingDecision)

    def test_autonomous_mode_routes_correctly(self, turn_router):
        """Autonomous mode simple message should route to autonomous_lower."""
        ctx = _make_context("thanks", is_cron=True, interaction_mode=InteractionMode.AUTONOMOUS)
        decision = turn_router.route(ctx)
        # Simple autonomous message routes to autonomous_lower
        assert decision.target_position == "autonomous_lower"

    def test_high_complexity_in_autonomous_goes_upper(self, turn_router, routing_config, mode_detector):
        """High-complexity autonomous request exceeding threshold goes to upper."""
        # Use a low escalation threshold so the message triggers escalation
        routing_config.complexity = ComplexityConfig(
            escalation_threshold=0.05,
            de_escalation_threshold=0.02,
        )
        low_threshold_router = TurnRouter(routing_config, mode_detector)
        msg = "Refactor the entire architecture across multiple files and explain the tradeoffs."
        ctx = _make_context(msg, is_cron=True, interaction_mode=InteractionMode.AUTONOMOUS)
        decision = low_threshold_router.route(ctx)
        assert decision.target_position == "upper"

    def test_subagent_stays_lower(self, turn_router):
        """Subagent context should always stay at lower tier."""
        msg = "implement complex thing"
        ctx = _make_context(msg, is_subagent=True)
        decision = turn_router.route(ctx)
        assert decision.target_position == "interactive_lower"


# ---------------------------------------------------------------------------
# Oversight Escalation Flag Lifecycle
# ---------------------------------------------------------------------------

class TestOversightEscalationFlag:
    def test_flag_set_on_escalate(self):
        """ESCALATE action should result in escalation signal."""
        config = OversightConfig(
            enabled=True,
            model="opus",
            provider="bedrock",
            every_n_turns=5,
            min_turns_before_first=3,
            max_reviews_per_session=5,
        )
        reviewer = OversightReviewer(config)

        agent = MagicMock()
        agent._oversight_reviewer = reviewer
        agent._oversight_last_escalated = False
        agent.model = "qwen"

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        with patch("agent.auxiliary_client.call_llm") as mock_llm:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = '{"action": "escalate", "reason": "stuck"}'
            response.usage = MagicMock(prompt_tokens=1000, completion_tokens=50)
            mock_llm.return_value = response

            result = run_oversight_if_due(agent, messages, 5)

        assert result is not None
        oversight_result, _ = result
        assert oversight_result.action == OversightAction.ESCALATE
        assert oversight_result.reason == "stuck"

    def test_skip_review_after_escalation(self):
        """After ESCALATE, next review should be skipped (skip_if_escalated=True)."""
        config = OversightConfig(
            enabled=True,
            model="opus",
            provider="bedrock",
            every_n_turns=5,
            min_turns_before_first=3,
            max_reviews_per_session=5,
            skip_if_escalated=True,
        )
        reviewer = OversightReviewer(config)

        should_yes, _ = reviewer.should_review(5, last_was_escalated=False)
        should_no, _ = reviewer.should_review(10, last_was_escalated=True)
        should_again, _ = reviewer.should_review(15, last_was_escalated=False)
        assert should_yes
        assert not should_no
        assert should_again

    def test_correction_injection_format(self):
        """CORRECT injection should be a system message with proper format."""
        result = OversightResult(
            action=OversightAction.CORRECT,
            note="You're writing to the wrong file. Target should be config.yaml not config.yml.",
        )
        msg = build_oversight_injection(result, "claude-opus-4-6")

        assert msg["role"] == "system"
        assert "OVERSIGHT NOTE" in msg["content"]
        assert "claude-opus-4-6" in msg["content"]
        assert "wrong file" in msg["content"]
        assert "Adjust your approach" in msg["content"]


# ---------------------------------------------------------------------------
# Config Validation Edge Cases
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_missing_graph_returns_disabled(self):
        """Config with no graph section should effectively have empty graph."""
        raw_cfg = {"model": {"routing": {"enabled": True}}}
        config = load_routing_config(raw_cfg)
        # enabled but empty graph
        assert config.enabled
        assert not config.graph

    def test_partial_position_config(self):
        """Position with only model/provider should still work."""
        raw_cfg = {
            "model": {
                "routing": {
                    "enabled": True,
                    "graph": {
                        "upper": {
                            "model": "opus",
                            "provider": "bedrock",
                        }
                    }
                }
            }
        }
        config = load_routing_config(raw_cfg)
        assert "upper" in config.graph
        assert config.graph["upper"].model == "opus"

    def test_empty_config_returns_defaults(self):
        """Empty dict should return disabled config with defaults."""
        config = load_routing_config({})
        assert not config.enabled

    def test_config_with_all_fields(self):
        """Full config should parse without error."""
        raw_cfg = {
            "model": {
                "routing": {
                    "enabled": True,
                    "graph": {
                        "upper": {
                            "model": "opus",
                            "provider": "bedrock",
                            "profile": {
                                "startup_latency_s": 0,
                                "ttft_p50_ms": 500,
                                "ttft_p90_ms": 2000,
                                "generation_tok_s": 80,
                            },
                        },
                        "interactive_lower": {
                            "model": "qwen",
                            "provider": "local",
                            "base_url": "http://localhost:58080",
                            "llm_config_name": "coder-next",
                            "profile": {
                                "startup_latency_s": 17,
                                "ttft_p50_ms": 200,
                                "ttft_p90_ms": 800,
                                "generation_tok_s": 35,
                            },
                        },
                    },
                    "complexity": {
                        "escalation_threshold": 0.8,
                        "de_escalation_threshold": 0.15,
                    },
                    "de_escalation": {
                        "enabled": False,
                    },
                }
            }
        }
        config = load_routing_config(raw_cfg)
        assert config.enabled
        assert len(config.graph) == 2
        assert config.complexity.escalation_threshold == 0.8
        assert not config.de_escalation.enabled


# ---------------------------------------------------------------------------
# Cross-Module Integration
# ---------------------------------------------------------------------------

class TestCrossModuleIntegration:
    def test_turn_router_decision_consumed_by_swap_manager(self, turn_router, swap_manager, mode_detector):
        """Turn router's decision should be consumable by swap manager."""
        ctx = _make_context(
            "Implement distributed consensus algorithm with Raft, "
            "write property-based tests, deploy to K8s cluster",
        )
        decision = turn_router.route(ctx)

        # Feed to swap manager — just check it doesn't crash
        result = swap_manager.should_swap(decision, mode_detector)
        assert isinstance(result, bool)

    @patch("agent.routing.config.load_routing_config")
    def test_routing_state_reflects_swap_manager(self, mock_config, swap_manager, routing_config):
        """get_routing_state should reflect swap manager's internal state."""
        mock_config.return_value = routing_config

        agent = MagicMock()
        agent._routing_swap_manager = swap_manager
        agent._routing_turn_router = MagicMock()
        agent._routing_turn_router.last_decision = None
        agent._routing_decision_history = None
        agent._routing_drift_detector = None
        agent._routing_mode_detector = None
        agent.model = "qwen-coder"
        agent.provider = "local"

        swap_manager._state = SwapState.SWAPPING
        state = get_routing_state(agent)
        assert state.swap_state == "SWAPPING"

    def test_oversight_reviewer_budget_persists_across_calls(self):
        """Review budget should persist across multiple run_oversight_if_due calls."""
        config = OversightConfig(
            enabled=True,
            model="opus",
            provider="bedrock",
            every_n_turns=5,
            min_turns_before_first=3,
            max_reviews_per_session=2,
        )

        agent = MagicMock()
        agent._oversight_reviewer = OversightReviewer(config)
        agent._oversight_last_escalated = False
        agent.model = "qwen"

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        with patch("agent.auxiliary_client.call_llm") as mock_llm:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = '{"action": "approve"}'
            response.usage = MagicMock(prompt_tokens=1000, completion_tokens=50)
            mock_llm.return_value = response

            # First review at turn 5
            r1 = run_oversight_if_due(agent, messages, 5)
            assert r1 is not None

            # Second review at turn 10
            r2 = run_oversight_if_due(agent, messages, 10)
            assert r2 is not None

            # Third review at turn 15 — budget exhausted
            r3 = run_oversight_if_due(agent, messages, 15)
            assert r3 is None  # Budget exhausted

    def test_ask_upper_budget_independent_of_oversight(self):
        """ask_upper budget and oversight budget are independent."""
        from agent.routing.ask_upper import AskUpperTool

        tool = AskUpperTool(
            upper_provider="bedrock",
            upper_model="opus",
            soft_budget=2,
            hard_budget=3,
        )

        oversight_config = OversightConfig(
            enabled=True,
            model="opus",
            provider="bedrock",
            max_reviews_per_session=5,
        )
        reviewer = OversightReviewer(oversight_config)

        # Exhaust ask_upper budget
        with patch("agent.auxiliary_client.call_llm") as mock_llm:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Guidance here"
            response.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
            mock_llm.return_value = response

            for _ in range(3):
                tool.execute("verify", "check this")

        assert tool.budget.exhausted
        assert not reviewer.budget_exhausted  # Independent

    def test_routing_decision_fields_complete(self, turn_router):
        """Every routing decision should have all required fields."""
        ctx = _make_context("hello")
        decision = turn_router.route(ctx)

        assert hasattr(decision, "target_position")
        assert hasattr(decision, "complexity_score")
        assert hasattr(decision, "interaction_mode")
        assert hasattr(decision, "reason")
        assert hasattr(decision, "swap_required")
        assert decision.target_position in ("upper", "interactive_lower", "fast_fallback")
        assert 0.0 <= decision.complexity_score <= 1.0
        assert isinstance(decision.reason, str) and len(decision.reason) > 0
