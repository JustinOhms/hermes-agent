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
