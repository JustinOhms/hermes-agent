"""Tests for agent.routing.__init__ — entry points and wiring logic."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from agent.routing.config import GraphPosition, RoutingConfig
from agent.routing.swap_manager import SwapManager


class TestInitSwapManagerPosition:
    """Test _init_swap_manager_position() HTTP path and fallback chain."""

    def _make_config(self) -> RoutingConfig:
        """Create a minimal config with a graph for testing."""
        config = RoutingConfig(enabled=True)
        config.graph = {
            "interactive_lower": GraphPosition(
                provider="custom:llm-local",
                model="qwen3-coder-next",
            ),
            "upper": GraphPosition(
                provider="bedrock",
                model="us.anthropic.claude-opus-4-6-v1",
            ),
        }
        return config

    def _make_mgr(self) -> SwapManager:
        return SwapManager(config=self._make_config())

    def test_http_success_matches_local_model(self):
        """When local server returns a matching model, position is set."""
        from agent.routing import _init_swap_manager_position

        config = self._make_config()
        mgr = self._make_mgr()
        agent = MagicMock()

        response_data = json.dumps({
            "models": [{"model": "qwen3-coder-next", "name": "qwen3-coder-next"}]
        }).encode()

        mock_response = MagicMock()
        mock_response.read.return_value = response_data
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            _init_swap_manager_position(mgr, agent, config)

        assert mgr._current_position == "interactive_lower"

    def test_http_success_no_match_falls_back_to_agent(self):
        """When local server returns an unknown model, falls back to agent attrs."""
        from agent.routing import _init_swap_manager_position

        config = self._make_config()
        mgr = self._make_mgr()
        agent = MagicMock()
        agent.provider = "bedrock"
        agent.model = "us.anthropic.claude-opus-4-6-v1"

        response_data = json.dumps({
            "models": [{"model": "some-other-model"}]
        }).encode()

        mock_response = MagicMock()
        mock_response.read.return_value = response_data
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            _init_swap_manager_position(mgr, agent, config)

        assert mgr._current_position == "upper"

    def test_http_failure_falls_back_to_agent_attrs(self):
        """When HTTP fails (server down), uses agent provider/model."""
        from agent.routing import _init_swap_manager_position

        config = self._make_config()
        mgr = self._make_mgr()
        agent = MagicMock()
        agent.provider = "bedrock"
        agent.model = "us.anthropic.claude-opus-4-6-v1"

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            _init_swap_manager_position(mgr, agent, config)

        assert mgr._current_position == "upper"

    def test_http_timeout_falls_back(self):
        """When HTTP times out, falls back gracefully."""
        from agent.routing import _init_swap_manager_position
        import urllib.error

        config = self._make_config()
        mgr = self._make_mgr()
        agent = MagicMock()
        agent.provider = "custom:llm-local"
        agent.model = "qwen3-coder-next"

        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            _init_swap_manager_position(mgr, agent, config)

        assert mgr._current_position == "interactive_lower"

    def test_no_match_defaults_to_first_position(self):
        """When nothing matches, defaults to first graph position."""
        from agent.routing import _init_swap_manager_position

        config = self._make_config()
        mgr = self._make_mgr()
        agent = MagicMock()
        agent.provider = "openai"
        agent.model = "gpt-4o"  # Not in graph

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            _init_swap_manager_position(mgr, agent, config)

        # Falls back to first position in graph
        assert mgr._current_position == "interactive_lower"

    def test_empty_models_response(self):
        """When local server returns empty models list, falls back."""
        from agent.routing import _init_swap_manager_position

        config = self._make_config()
        mgr = self._make_mgr()
        agent = MagicMock()
        agent.provider = "bedrock"
        agent.model = "us.anthropic.claude-opus-4-6-v1"

        response_data = json.dumps({"models": []}).encode()

        mock_response = MagicMock()
        mock_response.read.return_value = response_data
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            _init_swap_manager_position(mgr, agent, config)

        assert mgr._current_position == "upper"

    def test_no_local_positions_skips_http_call(self):
        """When no local positions configured, HTTP call should not be made."""
        from agent.routing import _init_swap_manager_position

        # Config with only cloud positions (no llm-local)
        config = RoutingConfig(enabled=True)
        config.graph = {
            "interactive_lower": GraphPosition(
                provider="openai",
                model="gpt-4",
                base_url="https://api.openai.com/v1",
                api_key="sk-test",
            ),
            "upper": GraphPosition(
                provider="bedrock",
                model="us.anthropic.claude-opus-4-6-v1",
            ),
        }
        mgr = self._make_mgr()
        mgr._config = config
        agent = MagicMock()
        agent.provider = "openai"
        agent.model = "gpt-4"

        # Mock HTTP to verify it's NOT called
        with patch("urllib.request.urlopen") as mock_urlopen:
            _init_swap_manager_position(mgr, agent, config)
            mock_urlopen.assert_not_called()

        # Should fall back to first position in graph
        assert mgr._current_position == "interactive_lower"


class TestOversightEscalationOverride:
    """Test RD-21: oversight escalation forces upper model."""

    def _make_config(self) -> RoutingConfig:
        config = RoutingConfig(enabled=True)
        config.graph = {
            "interactive_lower": GraphPosition(
                provider="custom:llm-local",
                model="qwen3-coder-next",
                base_url="http://127.0.0.1:58080/v1",
                llm_config_name="coder-next",
            ),
            "upper": GraphPosition(
                provider="bedrock",
                model="us.anthropic.claude-opus-4-6-v1",
            ),
        }
        return config

    def test_escalation_pending_forces_upper(self):
        """When _oversight_escalation_pending is True, route to upper."""
        from agent.routing import get_routing_decision

        agent = MagicMock()
        agent._oversight_escalation_pending = True
        agent._routing_decision_history = None

        with patch("agent.routing._load_cached_config", return_value=self._make_config()):
            decision = get_routing_decision(agent, "hello")

        assert decision is not None
        assert decision.target_position == "upper"
        assert decision.reason == "oversight_escalation"
        assert decision.swap_required is True
        assert decision.complexity_score == 1.0
        # Flag should be cleared
        assert agent._oversight_escalation_pending is False

    def test_escalation_clears_flag(self):
        """Escalation flag is cleared after one use (single-turn override)."""
        from agent.routing import get_routing_decision

        agent = MagicMock()
        agent._oversight_escalation_pending = True
        agent._routing_decision_history = None

        config = self._make_config()

        with patch("agent.routing._load_cached_config", return_value=config):
            # First call: escalation fires
            d1 = get_routing_decision(agent, "hello")
            assert d1.target_position == "upper"

            # Second call: normal routing (no escalation)
            agent._routing_mode_detector = None
            agent._routing_swap_manager = None
            d2 = get_routing_decision(agent, "hello again")
            # Should not be forced to upper (depends on heuristics, but reason != oversight_escalation)
            if d2:
                assert d2.reason != "oversight_escalation"

    def test_no_escalation_when_flag_false(self):
        """Normal routing when flag is not set."""
        from agent.routing import get_routing_decision

        agent = MagicMock()
        agent._oversight_escalation_pending = False
        agent._routing_mode_detector = None
        agent._routing_swap_manager = None
        agent._routing_decision_history = None

        with patch("agent.routing._load_cached_config", return_value=self._make_config()):
            decision = get_routing_decision(agent, "hello")

        if decision:
            assert decision.reason != "oversight_escalation"

    def test_escalation_appends_to_history(self):
        """Escalation decision is tracked in decision history."""
        from collections import deque
        from agent.routing import get_routing_decision

        agent = MagicMock()
        agent._oversight_escalation_pending = True
        history = deque(maxlen=20)
        agent._routing_decision_history = history

        with patch("agent.routing._load_cached_config", return_value=self._make_config()):
            get_routing_decision(agent, "hello")

        assert len(history) == 1
        assert history[0].reason == "oversight_escalation"
        assert history[0]._timestamp > 0
