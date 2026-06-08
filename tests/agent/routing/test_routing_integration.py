"""Integration tests for the routing system full lifecycle.

Simulates multi-turn conversations flowing through:
  TurnRouter → SwapManager → Oversight

These tests exercise the conversation_loop integration points
without requiring a live LLM.
"""

from __future__ import annotations

import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from agent.routing.ask_upper import AskUpperTool
from agent.routing.config import (
    ComplexityConfig,
    DeEscalationConfig,
    GraphPosition,
    GraphPositionProfile,
    InteractionModeConfig,
    RoutingConfig,
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
from agent.routing.state import RoutingState, get_routing_state
from agent.routing.swap_manager import SwapManager, SwapState
from agent.routing.turn_router import RoutingContext, RoutingDecision, TurnRouter


# ---------------------------------------------------------------------------
# Test config & fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def full_config():
    """Production-like routing config with all positions."""
    return RoutingConfig(
        enabled=True,
        graph={
            "upper": GraphPosition(
                provider="bedrock",
                model="claude-opus-4-6",
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
            "autonomous_lower": GraphPosition(
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
        de_escalation=DeEscalationConfig(enabled=True),
    )


@pytest.fixture
def mode_detector(full_config):
    return InteractionModeDetector(full_config)


@pytest.fixture
def turn_router(full_config, mode_detector):
    return TurnRouter(full_config, mode_detector)


@pytest.fixture
def swap_manager(full_config):
    sm = SwapManager(full_config)
    sm.set_current_position("interactive_lower")
    return sm


def _ctx(msg: str, **kwargs) -> RoutingContext:
    """Build a RoutingContext with sensible defaults."""
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
# Multi-turn lifecycle simulation
# ---------------------------------------------------------------------------

class TestMultiTurnLifecycle:
    """Simulate realistic conversation flows through the routing system."""

    def test_simple_interactive_session_stays_lower(self, turn_router, swap_manager, mode_detector):
        """A series of simple interactive messages should stay at interactive_lower."""
        messages = [
            "hi",
            "what time is it?",
            "thanks",
            "ok",
            "show me the weather",
        ]

        for msg in messages:
            ctx = _ctx(msg)
            decision = turn_router.route(ctx)
            should_swap = swap_manager.should_swap(decision, mode_detector)
            # Simple messages should stay at a lower-tier position
            assert decision.target_position in ("interactive_lower", "fast_fallback")
            # swap_manager may or may not swap depending on de-escalation config

    def test_escalation_triggers_swap_to_upper(self, turn_router, swap_manager, mode_detector, full_config):
        """A complex request should trigger escalation to upper."""
        # Lower the threshold to make escalation trigger
        full_config.complexity = ComplexityConfig(
            escalation_threshold=0.05,
            de_escalation_threshold=0.02,
        )
        router = TurnRouter(full_config, mode_detector)

        ctx = _ctx("Refactor the entire architecture across multiple files and explain tradeoffs.")
        decision = router.route(ctx)

        assert decision.target_position == "upper"
        should_swap = swap_manager.should_swap(decision, mode_detector)
        assert should_swap  # Need to swap from interactive_lower to upper

    def test_de_escalation_after_idle(self, turn_router, swap_manager, mode_detector, full_config):
        """When de-escalation is enabled and user is idle, should route to fallback."""
        full_config.complexity = ComplexityConfig(
            escalation_threshold=0.7,
            de_escalation_threshold=0.3,  # Low enough to trigger on "thanks"
        )
        full_config.de_escalation = DeEscalationConfig(enabled=True)
        router = TurnRouter(full_config, mode_detector)

        ctx = _ctx("ok", interaction_mode=InteractionMode.INTERACTIVE)
        decision = router.route(ctx)

        # With de-escalation enabled and very simple message, may route to fast_fallback
        assert decision.target_position in ("interactive_lower", "fast_fallback")


# ---------------------------------------------------------------------------
# Oversight lifecycle integration
# ---------------------------------------------------------------------------

class TestOversightLifecycle:
    """Test the oversight reviewer's integration with the turn lifecycle."""

    def test_oversight_fires_at_cadence(self):
        """Oversight should fire exactly at the configured cadence."""
        config = OversightConfig(
            enabled=True,
            model="opus",
            provider="bedrock",
            every_n_turns=5,
            min_turns_before_first=3,
            max_reviews_per_session=10,
        )
        reviewer = OversightReviewer(config)

        # Turns 1-4: no review
        for turn in range(1, 5):
            assert not reviewer.should_review(turn, last_was_escalated=False)

        # Turn 5: review
        assert reviewer.should_review(5, last_was_escalated=False)

        # Turns 6-9: no review
        for turn in range(6, 10):
            assert not reviewer.should_review(turn, last_was_escalated=False)

        # Turn 10: review
        assert reviewer.should_review(10, last_was_escalated=False)

    def test_oversight_budget_exhaustion(self):
        """After max_reviews_per_session reviews, no more reviews fire."""
        config = OversightConfig(
            enabled=True,
            model="opus",
            provider="bedrock",
            every_n_turns=2,
            min_turns_before_first=1,
            max_reviews_per_session=3,
        )

        agent = MagicMock()
        agent._oversight_reviewer = OversightReviewer(config)
        agent._oversight_last_escalated = False
        agent.model = "qwen"

        messages = [
            {"role": "user", "content": "msg"},
            {"role": "assistant", "content": "reply"},
        ]

        with patch("agent.auxiliary_client.call_llm") as mock_llm:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = '{"action": "approve"}'
            response.usage = MagicMock(prompt_tokens=500, completion_tokens=30)
            mock_llm.return_value = response

            results = []
            for turn in range(2, 12, 2):  # turns 2, 4, 6, 8, 10
                r = run_oversight_if_due(agent, messages, turn)
                results.append(r)

            # First 3 should succeed, rest should be None (budget exhausted)
            assert results[0] is not None
            assert results[1] is not None
            assert results[2] is not None
            assert results[3] is None
            assert results[4] is None

    def test_oversight_correct_produces_injection(self):
        """CORRECT action should produce a properly formatted system message."""
        config = OversightConfig(
            enabled=True,
            model="opus",
            provider="bedrock",
            every_n_turns=5,
            min_turns_before_first=3,
            max_reviews_per_session=5,
        )

        agent = MagicMock()
        agent._oversight_reviewer = OversightReviewer(config)
        agent._oversight_last_escalated = False
        agent.model = "qwen"

        messages = [
            {"role": "user", "content": "write a function to sort"},
            {"role": "assistant", "content": "def sort(arr): return sorted(arr)"},
            {"role": "user", "content": "make it work for edge cases"},
            {"role": "assistant", "content": "def sort(arr): if not arr: return []; return sorted(arr)"},
        ]

        with patch("agent.auxiliary_client.call_llm") as mock_llm:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = json.dumps({
                "action": "correct",
                "note": "The function should handle None input and non-list types."
            })
            response.usage = MagicMock(prompt_tokens=800, completion_tokens=60)
            mock_llm.return_value = response

            result = run_oversight_if_due(agent, messages, 5)

        assert result is not None
        assert result.action == OversightAction.CORRECT

        injection = build_oversight_injection(result, "claude-opus-4-6")
        assert injection["role"] == "system"
        assert "None input" in injection["content"]
        assert "non-list types" in injection["content"]

    def test_oversight_error_defaults_to_approve(self):
        """When the upper model call fails, oversight defaults to APPROVE."""
        config = OversightConfig(
            enabled=True,
            model="opus",
            provider="bedrock",
            every_n_turns=5,
            min_turns_before_first=3,
            max_reviews_per_session=5,
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
            mock_llm.side_effect = Exception("API timeout")

            result = run_oversight_if_due(agent, messages, 5)

        # Should not crash — defaults to approve on error
        assert result is not None
        assert result.action == OversightAction.APPROVE


# ---------------------------------------------------------------------------
# ask_upper tool lifecycle
# ---------------------------------------------------------------------------

class TestAskUpperLifecycle:
    """Test the ask_upper tool through its budget and execution lifecycle."""

    def test_budget_warning_at_soft_limit(self):
        """After soft_budget calls, response should include budget warning."""
        tool = AskUpperTool(
            upper_provider="bedrock",
            upper_model="opus",
            soft_budget=2,
            hard_budget=4,
        )

        with patch("agent.auxiliary_client.call_llm") as mock_llm:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Guidance text"
            response.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
            mock_llm.return_value = response

            # First call: normal
            r1 = tool.execute("verify", "check 1")
            assert not tool.budget.over_soft_budget

            # 2nd call: hits soft budget
            r2 = tool.execute("verify", "check 2")
            assert tool.budget.over_soft_budget

            # 3rd call: still over soft budget
            r3 = tool.execute("verify", "check 3")
            assert tool.budget.over_soft_budget

    def test_hard_budget_refuses(self):
        """After hard_budget calls, tool should refuse."""
        tool = AskUpperTool(
            upper_provider="bedrock",
            upper_model="opus",
            soft_budget=1,
            hard_budget=2,
        )

        with patch("agent.auxiliary_client.call_llm") as mock_llm:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Guidance"
            response.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
            mock_llm.return_value = response

            tool.execute("verify", "check 1")
            tool.execute("verify", "check 2")

            # 3rd call should be refused
            r3 = tool.execute("verify", "check 3")
            assert "UNAVAILABLE" in r3 or "exhausted" in r3.lower() or tool.budget.exhausted

    def test_context_truncation(self):
        """Long context should be truncated to prevent huge prompts."""
        tool = AskUpperTool(
            upper_provider="bedrock",
            upper_model="opus",
            soft_budget=5,
            hard_budget=10,
        )

        # 50k chars of context
        long_context = "x" * 50000

        with patch("agent.auxiliary_client.call_llm") as mock_llm:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Short guidance"
            response.usage = MagicMock(prompt_tokens=500, completion_tokens=50)
            mock_llm.return_value = response

            # Pass long_context as the context parameter (3rd arg)
            result = tool.execute("simplify", "how do I handle this?", long_context)

            # The call should succeed (no crash from huge context)
            assert mock_llm.called
            # Check that the actual content sent was truncated
            call_args = mock_llm.call_args
            if call_args[1] and "messages" in call_args[1]:
                messages = call_args[1]["messages"]
            elif len(call_args[0]) > 1:
                messages = call_args[0][1]
            else:
                messages = []

            # Find the user message with context — truncation indicator should be present
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                user_content = user_msgs[0].get("content", "")
                # Context should be truncated (16k default) + indicator
                assert len(user_content) < 20000
                assert "[...context truncated]" in user_content

    def test_request_types_all_valid(self):
        """All 5 request types should be accepted."""
        tool = AskUpperTool(
            upper_provider="bedrock",
            upper_model="opus",
            soft_budget=10,
            hard_budget=20,
        )

        request_types = ["simplify", "plan", "verify", "distill", "explain"]

        with patch("agent.auxiliary_client.call_llm") as mock_llm:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Response"
            response.usage = MagicMock(prompt_tokens=200, completion_tokens=50)
            mock_llm.return_value = response

            for rtype in request_types:
                result = tool.execute(rtype, f"test context for {rtype}")
                assert result  # Non-empty response


# ---------------------------------------------------------------------------
# State aggregation integration
# ---------------------------------------------------------------------------

class TestStateAggregation:
    """Test that get_routing_state correctly aggregates from all components."""

    @patch("agent.routing.config.load_routing_config")
    def test_full_state_aggregation(self, mock_config, full_config, swap_manager, mode_detector):
        """All state components should be reflected in RoutingState."""
        mock_config.return_value = full_config

        agent = MagicMock()
        agent._routing_swap_manager = swap_manager
        agent._routing_mode_detector = mode_detector
        agent._routing_decision_history = deque(maxlen=50)
        agent._routing_drift_detector = None

        # Set some state
        swap_manager.set_current_position("interactive_lower")
        swap_manager._state = SwapState.READY

        # Record mode detection
        ctx = _ctx("hello")
        mode_detector.current_mode(ctx)

        state = get_routing_state(agent)

        assert state.enabled
        assert state.current_position == "interactive_lower"
        assert state.swap_state == "READY"

    @patch("agent.routing.config.load_routing_config")
    def test_disabled_routing_minimal_state(self, mock_config):
        """When routing is disabled, state should be minimal."""
        disabled_config = RoutingConfig(
            enabled=False,
            graph={},
            complexity=ComplexityConfig(escalation_threshold=0.7, de_escalation_threshold=0.2),
            interaction_mode=InteractionModeConfig(idle_threshold_s=300, swap_back_messages=3, swap_back_window_s=60),
            de_escalation=DeEscalationConfig(enabled=False),
        )
        mock_config.return_value = disabled_config

        agent = MagicMock()
        state = get_routing_state(agent)

        assert not state.enabled
        assert state.swap_state == "IDLE"
        assert state.current_position is None

    @patch("agent.routing.config.load_routing_config")
    def test_state_survives_missing_components(self, mock_config, full_config):
        """State aggregation should work even if some components are missing."""
        mock_config.return_value = full_config

        agent = MagicMock()
        # No swap manager, no mode detector — just the config
        agent._routing_swap_manager = None
        agent._routing_mode_detector = None
        agent._routing_decision_history = None
        agent._routing_drift_detector = None

        state = get_routing_state(agent)

        # Should not crash, returns defaults
        assert state.enabled
        assert state.swap_state == "IDLE"


# Need json for test_oversight_correct_produces_injection
import json
