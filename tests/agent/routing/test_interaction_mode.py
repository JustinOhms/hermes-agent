"""Tests for agent/routing/interaction_mode.py — InteractionModeDetector."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from agent.routing.config import InteractionModeConfig, RoutingConfig
from agent.routing.interaction_mode import InteractionMode, InteractionModeDetector
from agent.routing.turn_router import RoutingContext


def _make_config(
    idle_threshold_s: int = 600,
    swap_back_messages: int = 3,
    swap_back_window_s: int = 60,
) -> RoutingConfig:
    cfg = RoutingConfig()
    cfg.interaction_mode = InteractionModeConfig(
        idle_threshold_s=idle_threshold_s,
        swap_back_messages=swap_back_messages,
        swap_back_window_s=swap_back_window_s,
    )
    return cfg


def _make_context(
    is_cron: bool = False,
    explicit_mode_override: str = None,
    is_subagent: bool = False,
) -> RoutingContext:
    return RoutingContext(
        user_message="hello",
        message_length=5,
        conversation_turn_count=1,
        is_cron=is_cron,
        is_subagent=is_subagent,
        platform="cli",
        interaction_mode=InteractionMode.INTERACTIVE,
        explicit_mode_override=explicit_mode_override,
    )


class TestExplicitOverride:
    def test_explicit_autonomous_override(self):
        detector = InteractionModeDetector(_make_config())
        ctx = _make_context(explicit_mode_override="autonomous")
        assert detector.current_mode(ctx) == InteractionMode.AUTONOMOUS

    def test_explicit_interactive_override(self):
        detector = InteractionModeDetector(_make_config())
        detector.record_agent_turn()
        detector.record_agent_turn()
        detector.record_agent_turn()
        detector.record_agent_turn()
        detector.record_agent_turn()
        detector.record_agent_turn()
        ctx = _make_context(explicit_mode_override="interactive")
        # Even with many agent turns, explicit override wins
        assert detector.current_mode(ctx) == InteractionMode.INTERACTIVE

    def test_no_override_returns_interactive_by_default(self):
        detector = InteractionModeDetector(_make_config())
        detector.record_user_message()
        ctx = _make_context()
        assert detector.current_mode(ctx) == InteractionMode.INTERACTIVE


class TestCronContext:
    def test_cron_always_autonomous(self):
        detector = InteractionModeDetector(_make_config())
        detector.record_user_message()
        ctx = _make_context(is_cron=True)
        assert detector.current_mode(ctx) == InteractionMode.AUTONOMOUS

    def test_non_cron_not_autonomous_by_default(self):
        detector = InteractionModeDetector(_make_config())
        detector.record_user_message()
        ctx = _make_context(is_cron=False)
        assert detector.current_mode(ctx) == InteractionMode.INTERACTIVE


class TestIdleThreshold:
    def test_idle_beyond_threshold_is_autonomous(self):
        detector = InteractionModeDetector(_make_config(idle_threshold_s=5))
        # Simulate a message 10 seconds ago
        with patch("agent.routing.interaction_mode.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            detector.record_user_message()
            # Now 10 seconds later
            mock_time.monotonic.return_value = 10.0
            ctx = _make_context()
            assert detector.current_mode(ctx) == InteractionMode.AUTONOMOUS

    def test_within_threshold_is_interactive(self):
        detector = InteractionModeDetector(_make_config(idle_threshold_s=600))
        with patch("agent.routing.interaction_mode.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            detector.record_user_message()
            # Only 10 seconds later (well within 600s threshold)
            mock_time.monotonic.return_value = 10.0
            ctx = _make_context()
            assert detector.current_mode(ctx) == InteractionMode.INTERACTIVE

    def test_no_user_message_yet_does_not_trigger_idle(self):
        detector = InteractionModeDetector(_make_config(idle_threshold_s=1))
        ctx = _make_context()
        # No user message recorded yet → threshold doesn't apply
        assert detector.current_mode(ctx) == InteractionMode.INTERACTIVE


class TestConsecutiveAgentTurns:
    def test_many_agent_turns_without_user_is_autonomous(self):
        detector = InteractionModeDetector(_make_config())
        detector.record_user_message()
        for _ in range(6):
            detector.record_agent_turn()
        ctx = _make_context()
        assert detector.current_mode(ctx) == InteractionMode.AUTONOMOUS

    def test_exactly_five_agent_turns_is_still_interactive(self):
        detector = InteractionModeDetector(_make_config())
        detector.record_user_message()
        for _ in range(5):
            detector.record_agent_turn()
        ctx = _make_context()
        # threshold is > 5, so exactly 5 should still be interactive
        assert detector.current_mode(ctx) == InteractionMode.INTERACTIVE

    def test_user_message_resets_agent_turn_counter(self):
        detector = InteractionModeDetector(_make_config())
        detector.record_user_message()
        for _ in range(6):
            detector.record_agent_turn()
        # New user message resets counter
        detector.record_user_message()
        ctx = _make_context()
        assert detector.current_mode(ctx) == InteractionMode.INTERACTIVE


class TestSustainedEngagement:
    def test_rapid_messages_triggers_sustained_engagement(self):
        detector = InteractionModeDetector(
            _make_config(swap_back_messages=3, swap_back_window_s=60)
        )
        with patch("agent.routing.interaction_mode.time") as mock_time:
            now = 1000.0
            mock_time.monotonic.return_value = now
            detector.record_user_message()
            mock_time.monotonic.return_value = now + 10
            detector.record_user_message()
            mock_time.monotonic.return_value = now + 20
            detector.record_user_message()
            mock_time.monotonic.return_value = now + 30
            assert detector.sustained_engagement_detected() is True

    def test_sparse_messages_do_not_trigger(self):
        detector = InteractionModeDetector(
            _make_config(swap_back_messages=3, swap_back_window_s=60)
        )
        with patch("agent.routing.interaction_mode.time") as mock_time:
            now = 1000.0
            mock_time.monotonic.return_value = now
            detector.record_user_message()
            # Second message is outside the 60s window
            mock_time.monotonic.return_value = now + 120
            detector.record_user_message()
            mock_time.monotonic.return_value = now + 150
            # Only 1 message within the 60s window
            assert detector.sustained_engagement_detected() is False

    def test_exactly_at_threshold_triggers(self):
        detector = InteractionModeDetector(
            _make_config(swap_back_messages=2, swap_back_window_s=60)
        )
        with patch("agent.routing.interaction_mode.time") as mock_time:
            now = 1000.0
            mock_time.monotonic.return_value = now
            detector.record_user_message()
            mock_time.monotonic.return_value = now + 5
            detector.record_user_message()
            mock_time.monotonic.return_value = now + 10
            assert detector.sustained_engagement_detected() is True
