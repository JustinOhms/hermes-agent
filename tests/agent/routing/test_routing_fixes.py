"""Tests for the specific fixes applied to agent/routing/ modules."""

from __future__ import annotations

import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from agent.routing.config import (
    ComplexityConfig,
    GraphPosition,
    GraphPositionProfile,
    InteractionModeConfig,
    auto_assign_tiers,
    build_tier_ladder,
    load_routing_config,
)
from agent.routing.oversight import (
    OversightAction,
    OversightConfig,
    OversightResult,
    OversightReviewer,
    load_oversight_config,
)
from agent.routing.interaction_mode import InteractionMode, InteractionModeDetector
from agent.routing.budget import BudgetTracker


class TestDequeRollingWindow:
    """Test that oversight.py uses deque for rolling window (performance fix)."""

    def test_oversight_reviewer_uses_deque(self):
        """OversightReviewer should use deque with maxlen=20 for token counts."""
        config = OversightConfig(
            enabled=True,
            model="us.anthropic.claude-opus-4-6-v1",
            provider="bedrock",
            every_n_turns=5,
            review_window=5,
        )
        reviewer = OversightReviewer(config)

        # Verify _turn_token_counts is a deque
        assert isinstance(reviewer._turn_token_counts, deque)

        # Verify it has maxlen=20
        assert reviewer._turn_token_counts.maxlen == 20

        # Test that it auto-trims
        for i in range(25):
            reviewer._turn_token_counts.append(i)

        assert len(reviewer._turn_token_counts) == 20
        assert list(reviewer._turn_token_counts) == list(range(5, 25))


class TestBudgetTracker:
    """Test the shared BudgetTracker used by ask_upper and oversight."""

    def test_initial_state(self):
        budget = BudgetTracker()
        assert budget.calls == 0
        assert not budget.exhausted
        assert not budget.over_soft

    def test_record_call_increments(self):
        budget = BudgetTracker(soft_limit=3, hard_limit=5)
        budget.record_call(input_tokens=100, output_tokens=50)
        assert budget.calls == 1
        assert budget.total_input_tokens == 100
        assert budget.total_output_tokens == 50
        assert len(budget.timestamps) == 1

    def test_soft_limit(self):
        budget = BudgetTracker(soft_limit=2, hard_limit=5)
        budget.record_call()
        assert not budget.over_soft
        budget.record_call()
        assert budget.over_soft

    def test_hard_limit(self):
        budget = BudgetTracker(soft_limit=2, hard_limit=3)
        for _ in range(2):
            budget.record_call()
        assert not budget.exhausted
        budget.record_call()
        assert budget.exhausted

    def test_reset(self):
        budget = BudgetTracker(soft_limit=2, hard_limit=5)
        budget.record_call(input_tokens=100, output_tokens=50)
        budget.record_call(input_tokens=200, output_tokens=100)
        budget.reset()
        assert budget.calls == 0
        assert budget.total_input_tokens == 0
        assert budget.total_output_tokens == 0
        assert len(budget.timestamps) == 0

    def test_get_status(self):
        budget = BudgetTracker(soft_limit=3, hard_limit=10)
        budget.record_call(input_tokens=500, output_tokens=100)
        status = budget.get_status()
        assert status["calls"] == 1
        assert status["soft_limit"] == 3
        assert status["hard_limit"] == 10
        assert status["total_input_tokens"] == 500
        assert status["total_output_tokens"] == 100
        assert status["exhausted"] is False


class TestConfigBoundsChecking:
    """Test that config.py properly handles bounds checking."""

    def test_interaction_mode_bounds_checking(self):
        """Interaction mode config should enforce bounds."""
        # Test idle_threshold_s < 1
        cfg = load_routing_config({
            "model": {
                "routing": {
                    "enabled": True,
                    "interaction_mode": {"idle_threshold_s": 0},
                }
            }
        })
        assert cfg.interaction_mode.idle_threshold_s == 1

        # Test swap_back_messages < 1
        cfg = load_routing_config({
            "model": {
                "routing": {
                    "enabled": True,
                    "interaction_mode": {"swap_back_messages": -3},
                }
            }
        })
        assert cfg.interaction_mode.swap_back_messages == 1

        # Test swap_back_window_s < 1
        cfg = load_routing_config({
            "model": {
                "routing": {
                    "enabled": True,
                    "interaction_mode": {"swap_back_window_s": 0},
                }
            }
        })
        assert cfg.interaction_mode.swap_back_window_s == 1


class TestOversightRegexFix:
    """Test that oversight.py correctly handles nested braces."""

    def test_parse_response_with_nested_json_in_note(self):
        """Nested braces in note/reason/warning should parse correctly."""
        config = OversightConfig(
            enabled=True,
            model="us.anthropic.claude-opus-4-6-v1",
            provider="bedrock",
            every_n_turns=5,
        )
        reviewer = OversightReviewer(config)

        # Test with nested braces in note
        content = """{
  "action": "correct",
  "note": "Use format: {\\"key\\": \\"value\\"} for JSON"
}"""
        result = reviewer._parse_response(content)
        assert result.action == OversightAction.CORRECT
        assert "key" in result.note

    def test_parse_response_with_nested_json_in_reason(self):
        """Nested braces in reason should parse correctly."""
        config = OversightConfig(
            enabled=True,
            model="us.anthropic.claude-opus-4-6-v1",
            provider="bedrock",
            every_n_turns=5,
        )
        reviewer = OversightReviewer(config)

        content = '{"action": "escalate", "reason": "The model is stuck in a loop"}'
        result = reviewer._parse_response(content)
        assert result.action == OversightAction.ESCALATE
        assert "stuck" in result.reason

    def test_parse_response_with_nested_json_in_warning(self):
        """Nested braces in warning should parse correctly."""
        config = OversightConfig(
            enabled=True,
            model="us.anthropic.claude-opus-4-6-v1",
            provider="bedrock",
            every_n_turns=5,
        )
        reviewer = OversightReviewer(config)

        content = '{"action": "flag", "warning": "Potential drift: check status"}'
        result = reviewer._parse_response(content)
        assert result.action == OversightAction.FLAG
        assert "status" in result.warning


class TestRoutingDecisionTimestamp:
    """Test that RoutingDecision has _timestamp field."""

    def test_routing_decision_has_timestamp_field(self):
        """RoutingDecision should have _timestamp field."""
        from agent.routing.turn_router import RoutingDecision

        decision = RoutingDecision(
            target_position="interactive_lower",
            complexity_score=0.5,
            interaction_mode=InteractionMode.INTERACTIVE,
            reason="test",
            swap_required=False,
        )
        assert hasattr(decision, "_timestamp")
        assert decision._timestamp == 0.0

    def test_routing_decision_timestamp_settable(self):
        """_timestamp should be settable for history tracking."""
        from agent.routing.turn_router import RoutingDecision

        decision = RoutingDecision(
            target_position="interactive_lower",
            complexity_score=0.5,
            interaction_mode=InteractionMode.INTERACTIVE,
            reason="test",
            swap_required=False,
            _timestamp=12345.678,
        )
        assert decision._timestamp == 12345.678

    def test_decision_history_with_timestamps(self):
        """Decision history should properly store timestamps."""
        from agent.routing.state import get_routing_state

        agent = MagicMock()
        agent._routing_swap_manager = None
        agent._routing_mode_detector = None
        agent._routing_drift_detector = None

        # Create a deque with decisions that have timestamps
        history = deque(maxlen=20)
        d1 = MagicMock()
        d1._timestamp = 100.0
        d2 = MagicMock()
        d2._timestamp = 200.0
        history.extend([d1, d2])

        agent._routing_decision_history = history

        mock_config = MagicMock()
        mock_config.enabled = True

        with patch("agent.routing.config.load_routing_config", return_value=mock_config):
            state = get_routing_state(agent)

        assert len(state.decision_history) == 2
        assert state.last_decision._timestamp == 200.0


class TestAskUpperErrorSanitization:
    """Test that ask_upper.py sanitizes error messages."""

    def test_error_message_uses_type_not_details(self):
        """Error message should use type name, not raw exception."""
        from agent.routing.ask_upper import AskUpperTool

        tool = AskUpperTool(
            upper_provider="bedrock",
            upper_model="claude-opus-4",
        )

        # Mock auxiliary_client.call_llm to raise with sensitive details
        with patch.dict("sys.modules", {"agent.auxiliary_client": MagicMock()}):
            import sys
            mock_module = sys.modules["agent.auxiliary_client"]
            mock_module.call_llm = MagicMock(
                side_effect=ConnectionError("api_key=sk-abc123 host=secret.internal.corp")
            )
            result = tool.execute("simplify", "test question")

        # Should contain error type but NOT the sensitive message
        assert "ConnectionError" in result
        assert "sk-abc123" not in result
        assert "secret.internal.corp" not in result


class TestBuildTierLadder:
    """Test build_tier_ladder() ordering logic."""

    def _pos(self, tier: int = 0) -> GraphPosition:
        return GraphPosition(provider="test", model="test-model", tier=tier)

    def test_empty_graph(self):
        assert build_tier_ladder({}) is None

    def test_no_tiers_assigned_returns_none(self):
        """Without explicit tiers, returns None (upgrade/downgrade disabled)."""
        graph = {
            "upper": self._pos(),
            "fast_fallback": self._pos(),
            "interactive_lower": self._pos(),
        }
        assert build_tier_ladder(graph) is None

    def test_partial_tiers_returns_none(self):
        """If any position has tier=0, returns None."""
        graph = {
            "upper": self._pos(tier=3),
            "fast_fallback": self._pos(tier=1),
            "interactive_lower": self._pos(),  # tier=0
        }
        assert build_tier_ladder(graph) is None

    def test_explicit_tiers(self):
        """Explicit tier values are used for ordering."""
        graph = {
            "upper": self._pos(tier=3),
            "fast_fallback": self._pos(tier=1),
            "coder": self._pos(tier=2),
        }
        ladder = build_tier_ladder(graph)
        assert ladder == ["fast_fallback", "coder", "upper"]

    def test_all_custom_positions_with_tiers(self):
        """Entirely custom position names work with explicit tiers."""
        graph = {
            "brain": self._pos(tier=3),
            "muscle": self._pos(tier=2),
            "speed": self._pos(tier=1),
        }
        ladder = build_tier_ladder(graph)
        assert ladder == ["speed", "muscle", "brain"]


class TestAutoAssignTiers:
    """Test auto_assign_tiers() scoring logic."""

    def test_empty_graph(self):
        assert auto_assign_tiers({}) == {}

    def test_cloud_ranks_above_local(self):
        """Cloud positions get higher tier than local positions."""
        graph = {
            "cloud": GraphPosition(
                provider="bedrock",
                model="claude-opus",
                profile=GraphPositionProfile(generation_tok_s=80, ttft_p50_ms=2000),
            ),
            "local": GraphPosition(
                provider="custom:llm-local",
                model="qwen3",
                base_url="http://127.0.0.1:58080/v1",
                llm_config_name="qwen3",
                profile=GraphPositionProfile(generation_tok_s=139, ttft_p50_ms=200),
            ),
        }
        result = auto_assign_tiers(graph)
        assert result["local"] < result["cloud"]

    def test_slower_local_ranks_above_faster_local(self):
        """Among local models, slower (larger) models get higher tier."""
        graph = {
            "fast": GraphPosition(
                provider="custom:llm-local",
                model="small-model",
                base_url="http://127.0.0.1:58080/v1",
                llm_config_name="fast",
                profile=GraphPositionProfile(generation_tok_s=139, ttft_p50_ms=200),
            ),
            "slow": GraphPosition(
                provider="custom:llm-local",
                model="large-model",
                base_url="http://127.0.0.1:58080/v1",
                llm_config_name="slow",
                profile=GraphPositionProfile(generation_tok_s=33, ttft_p50_ms=800),
            ),
        }
        result = auto_assign_tiers(graph)
        assert result["fast"] < result["slow"]

    def test_three_position_real_world(self):
        """Real-world 3-position graph produces correct ordering."""
        graph = {
            "fast_fallback": GraphPosition(
                provider="custom:llm-local",
                model="qwen3-coder-30b",
                base_url="http://127.0.0.1:58080/v1",
                llm_config_name="little-qwen",
                profile=GraphPositionProfile(generation_tok_s=139, ttft_p50_ms=200),
            ),
            "interactive_lower": GraphPosition(
                provider="custom:llm-local",
                model="qwen3-coder-next",
                base_url="http://127.0.0.1:58080/v1",
                llm_config_name="coder-next",
                profile=GraphPositionProfile(generation_tok_s=33, ttft_p50_ms=800),
            ),
            "upper": GraphPosition(
                provider="bedrock",
                model="us.anthropic.claude-opus-4-6-v1",
                profile=GraphPositionProfile(generation_tok_s=80, ttft_p50_ms=2000),
            ),
        }
        result = auto_assign_tiers(graph)
        # Expected: fast_fallback=1, interactive_lower=2, upper=3
        assert result["fast_fallback"] == 1
        assert result["interactive_lower"] == 2
        assert result["upper"] == 3

    def test_assigns_sequential_tiers(self):
        """Tiers are always 1-indexed and sequential."""
        graph = {
            "a": GraphPosition(
                provider="p", model="m",
                profile=GraphPositionProfile(generation_tok_s=100, ttft_p50_ms=100),
            ),
            "b": GraphPosition(
                provider="p", model="m",
                profile=GraphPositionProfile(generation_tok_s=50, ttft_p50_ms=500),
            ),
            "c": GraphPosition(
                provider="p", model="m",
                profile=GraphPositionProfile(generation_tok_s=10, ttft_p50_ms=1000),
            ),
        }
        result = auto_assign_tiers(graph)
        assert sorted(result.values()) == [1, 2, 3]

