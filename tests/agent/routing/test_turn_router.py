"""Tests for agent/routing/turn_router.py — TurnRouter complexity scoring and routing."""

from __future__ import annotations

import pytest

from agent.routing.config import ComplexityConfig, DeEscalationConfig, RoutingConfig
from agent.routing.interaction_mode import InteractionMode, InteractionModeDetector
from agent.routing.turn_router import RoutingContext, RoutingDecision, TurnRouter


def _make_config(
    escalation_threshold: float = 0.7,
    de_escalation_threshold: float = 0.2,
    de_escalation_enabled: bool = False,
) -> RoutingConfig:
    cfg = RoutingConfig(enabled=True)
    cfg.complexity = ComplexityConfig(
        escalation_threshold=escalation_threshold,
        de_escalation_threshold=de_escalation_threshold,
    )
    cfg.de_escalation = DeEscalationConfig(enabled=de_escalation_enabled)
    return cfg


def _make_context(
    user_message: str = "hello",
    interaction_mode: InteractionMode = InteractionMode.INTERACTIVE,
    is_cron: bool = False,
    is_subagent: bool = False,
    last_response_had_errors: bool = False,
    explicit_mode_override: str = None,
) -> RoutingContext:
    return RoutingContext(
        user_message=user_message,
        message_length=len(user_message),
        conversation_turn_count=1,
        is_cron=is_cron,
        is_subagent=is_subagent,
        platform="cli",
        interaction_mode=interaction_mode,
        last_response_had_errors=last_response_had_errors,
        explicit_mode_override=explicit_mode_override,
    )


def _make_router(
    escalation_threshold: float = 0.7,
    de_escalation_threshold: float = 0.2,
    de_escalation_enabled: bool = False,
) -> TurnRouter:
    cfg = _make_config(escalation_threshold, de_escalation_threshold, de_escalation_enabled)
    detector = InteractionModeDetector(cfg)
    return TurnRouter(cfg, detector)


class TestComplexityScoring:
    def test_short_simple_message_low_complexity(self):
        router = _make_router()
        ctx = _make_context("ok")
        score = router._score_complexity(ctx)
        assert score < 0.3

    def test_thanks_is_low_complexity(self):
        router = _make_router()
        ctx = _make_context("thanks")
        assert router._score_complexity(ctx) < 0.3

    def test_yes_is_low_complexity(self):
        router = _make_router()
        ctx = _make_context("yes")
        assert router._score_complexity(ctx) < 0.3

    def test_long_message_increases_complexity(self):
        router = _make_router()
        long_msg = "Please help me with this. " * 100  # ~2500 chars
        ctx = _make_context(long_msg)
        score = router._score_complexity(ctx)
        assert score >= 0.25

    def test_structured_data_discounts_length(self):
        router = _make_router()
        # JSON lines inflate length but are discounted
        structured_msg = '{"key": "value"}\n' * 200  # ~3400 chars but mostly structured
        plain_msg = "x" * 3400
        ctx_structured = _make_context(structured_msg)
        ctx_plain = _make_context(plain_msg)
        score_structured = router._score_complexity(ctx_structured)
        score_plain = router._score_complexity(ctx_plain)
        # Structured should score lower (or equal) due to discount
        assert score_structured <= score_plain

    def test_technical_keywords_increase_complexity(self):
        router = _make_router()
        ctx = _make_context("Can you refactor this and fix the architecture issue?")
        score = router._score_complexity(ctx)
        assert score >= 0.1

    def test_multi_file_reference_increases_complexity(self):
        router = _make_router()
        ctx = _make_context("Update this across multiple files in the codebase.")
        score = router._score_complexity(ctx)
        assert score >= 0.15

    def test_reasoning_patterns_increase_complexity(self):
        router = _make_router()
        ctx = _make_context("Explain why this approach is better and compare it with alternatives.")
        score = router._score_complexity(ctx)
        assert score >= 0.1

    def test_error_context_increases_complexity(self):
        router = _make_router()
        ctx = _make_context("Fix this", last_response_had_errors=True)
        score_no_error = router._score_complexity(_make_context("Fix this"))
        score_with_error = router._score_complexity(ctx)
        assert score_with_error > score_no_error

    def test_complexity_capped_at_1(self):
        router = _make_router()
        # Pile on every signal
        msg = (
            "Please refactor the architecture and analyze performance bottlenecks "
            "across multiple files. Explain why we have deadlocks and compare options. " * 10
        )
        ctx = _make_context(msg, last_response_had_errors=True)
        score = router._score_complexity(ctx)
        assert score <= 1.0

    def test_complexity_non_negative(self):
        router = _make_router()
        ctx = _make_context("")
        score = router._score_complexity(ctx)
        assert score >= 0.0


class TestRoutingDecisions:
    def test_simple_message_routes_to_interactive_lower(self):
        router = _make_router()
        ctx = _make_context("ok", interaction_mode=InteractionMode.INTERACTIVE)
        decision = router.route(ctx)
        assert decision.target_position == "interactive_lower"

    def test_high_complexity_routes_to_upper(self):
        router = _make_router(escalation_threshold=0.1)
        msg = "Refactor the entire architecture across multiple files and explain the tradeoffs."
        ctx = _make_context(msg, interaction_mode=InteractionMode.INTERACTIVE)
        decision = router.route(ctx)
        assert decision.target_position == "upper"

    def test_cron_context_routes_autonomous_lower(self):
        router = _make_router()
        ctx = _make_context("run the daily report", is_cron=True, interaction_mode=InteractionMode.AUTONOMOUS)
        decision = router.route(ctx)
        assert decision.target_position == "autonomous_lower"
        assert decision.interaction_mode == InteractionMode.AUTONOMOUS

    def test_cron_context_high_complexity_routes_upper(self):
        router = _make_router(escalation_threshold=0.1)
        msg = "Refactor everything across multiple files with full analysis."
        ctx = _make_context(msg, is_cron=True, interaction_mode=InteractionMode.AUTONOMOUS)
        decision = router.route(ctx)
        assert decision.target_position == "upper"

    def test_subagent_always_interactive_lower(self):
        router = _make_router(escalation_threshold=0.1)
        # Even very complex message from subagent stays on interactive_lower
        msg = "Refactor the architecture across multiple files and analyze performance."
        ctx = _make_context(msg, is_subagent=True)
        decision = router.route(ctx)
        assert decision.target_position == "interactive_lower"
        assert "subagent" in decision.reason

    def test_de_escalation_disabled_never_fast_fallback(self):
        router = _make_router(de_escalation_enabled=False)
        # Very simple message
        ctx = _make_context("ok", interaction_mode=InteractionMode.INTERACTIVE)
        decision = router.route(ctx)
        assert decision.target_position != "fast_fallback"

    def test_de_escalation_enabled_routes_fast_fallback(self):
        router = _make_router(
            de_escalation_threshold=0.9,  # Very high threshold — easy to be below
            de_escalation_enabled=True,
        )
        ctx = _make_context("ok", interaction_mode=InteractionMode.INTERACTIVE)
        decision = router.route(ctx)
        assert decision.target_position == "fast_fallback"

    def test_routing_decision_has_reason(self):
        router = _make_router()
        ctx = _make_context("hello")
        decision = router.route(ctx)
        assert decision.reason
        assert len(decision.reason) > 0

    def test_complexity_score_in_decision(self):
        router = _make_router()
        ctx = _make_context("hello")
        decision = router.route(ctx)
        assert 0.0 <= decision.complexity_score <= 1.0

    def test_swap_required_always_false_phase1(self):
        router = _make_router(escalation_threshold=0.1)
        msg = "Refactor everything across multiple files."
        ctx = _make_context(msg)
        decision = router.route(ctx)
        # Phase 1: swap detection not implemented
        assert decision.swap_required is False

    def test_interaction_mode_propagated_to_decision(self):
        router = _make_router()
        ctx = _make_context("hello", interaction_mode=InteractionMode.AUTONOMOUS)
        decision = router.route(ctx)
        assert decision.interaction_mode == InteractionMode.AUTONOMOUS
