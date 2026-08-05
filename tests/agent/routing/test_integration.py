"""Tests for agent/routing/__init__.py — get_routing_decision() entry point."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.routing import get_routing_decision
from agent.routing.config import GraphPosition, RoutingConfig
from agent.routing.interaction_mode import InteractionMode
from agent.routing.turn_router import RoutingDecision


def _make_agent(**kwargs) -> SimpleNamespace:
    defaults = {
        "_turn_count": 1,
        "is_cron": False,
        "is_subagent": False,
        "platform": "cli",
        "_recent_tool_calls": [],
        "_last_response_had_errors": False,
        "_explicit_mode_override": None,
        "_routing_mode_detector": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _enabled_config() -> RoutingConfig:
    cfg = RoutingConfig(enabled=True)
    cfg.graph = {
        "alpha": GraphPosition(provider="custom:llm-local", model="qwen-local", tier=1, alias="Alpha"),
        "beta": GraphPosition(provider="openai-codex", model="gpt-5.6-terra", tier=2, alias="Beta"),
        "gamma": GraphPosition(provider="openai-codex", model="gpt-5.6-sol", tier=3, alias="Gamma"),
    }
    return cfg


def _disabled_config() -> RoutingConfig:
    return RoutingConfig(enabled=False)


class TestGetRoutingDecisionEnabled:
    def test_returns_decision_when_routing_enabled(self):
        agent = _make_agent()
        with patch("agent.routing.load_routing_config", return_value=_enabled_config()):
            result = get_routing_decision(agent, "hello")
        assert result is not None
        assert isinstance(result, RoutingDecision)

    def test_returns_target_position(self):
        agent = _make_agent()
        with patch("agent.routing.load_routing_config", return_value=_enabled_config()):
            result = get_routing_decision(agent, "hello")
        assert result.target_position in {"alpha", "beta", "gamma"}

    def test_simple_message_routes_base(self):
        agent = _make_agent()
        with patch("agent.routing.load_routing_config", return_value=_enabled_config()):
            result = get_routing_decision(agent, "ok")
        assert result is not None
        assert result.target_position == "alpha"  # base tier

    def test_cron_agent_gets_autonomous_mode(self):
        agent = _make_agent(is_cron=True)
        with patch("agent.routing.load_routing_config", return_value=_enabled_config()):
            result = get_routing_decision(agent, "run report")
        assert result is not None
        assert result.interaction_mode == InteractionMode.AUTONOMOUS

    def test_subagent_always_base(self):
        agent = _make_agent(is_subagent=True)
        with patch("agent.routing.load_routing_config", return_value=_enabled_config()):
            result = get_routing_decision(agent, "complex task with refactoring")
        assert result is not None
        assert result.target_position == "alpha"  # base tier

    def test_detector_cached_on_agent(self):
        agent = _make_agent()
        with patch("agent.routing.load_routing_config", return_value=_enabled_config()):
            get_routing_decision(agent, "first message")
            detector_after_first = getattr(agent, "_routing_mode_detector", None)
            get_routing_decision(agent, "second message")
            detector_after_second = getattr(agent, "_routing_mode_detector", None)
        assert detector_after_first is not None
        # Same detector instance reused
        assert detector_after_first is detector_after_second

    def test_missing_agent_attributes_handled_gracefully(self):
        # Agent with no routing-related attributes at all
        agent = SimpleNamespace()
        with patch("agent.routing.load_routing_config", return_value=_enabled_config()):
            result = get_routing_decision(agent, "hello")
        assert result is not None


class TestGetRoutingDecisionDisabled:
    def test_returns_none_when_routing_disabled(self):
        agent = _make_agent()
        with patch("agent.routing.load_routing_config", return_value=_disabled_config()):
            result = get_routing_decision(agent, "hello")
        assert result is None

    def test_returns_none_when_config_load_fails(self):
        agent = _make_agent()
        with patch("agent.routing.load_routing_config", side_effect=RuntimeError("disk error")):
            result = get_routing_decision(agent, "hello")
        assert result is None

    def test_returns_none_on_unexpected_error(self):
        agent = _make_agent()
        # Patch TurnRouter.route to raise so we exercise the outer except
        with patch("agent.routing.TurnRouter.route", side_effect=RuntimeError("boom")):
            with patch("agent.routing.load_routing_config", return_value=_enabled_config()):
                result = get_routing_decision(agent, "hello")
        assert result is None


class TestRoutingDecisionContent:
    def test_decision_has_complexity_score(self):
        agent = _make_agent()
        with patch("agent.routing.load_routing_config", return_value=_enabled_config()):
            result = get_routing_decision(agent, "hello")
        assert result is not None
        assert 0.0 <= result.complexity_score <= 1.0

    def test_decision_has_reason_string(self):
        agent = _make_agent()
        with patch("agent.routing.load_routing_config", return_value=_enabled_config()):
            result = get_routing_decision(agent, "hello")
        assert result is not None
        assert isinstance(result.reason, str)
        assert result.reason

    def test_swap_required_false_phase1(self):
        agent = _make_agent()
        with patch("agent.routing.load_routing_config", return_value=_enabled_config()):
            result = get_routing_decision(agent, "hello")
        assert result is not None
        assert result.swap_required is False
