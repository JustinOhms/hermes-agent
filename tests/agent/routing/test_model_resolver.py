"""Tests for agent/routing/model_resolver.py."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from agent.routing.config import GraphPosition, GraphPositionProfile, RoutingConfig
from agent.routing.model_resolver import ModelResolver, ResolvedModel, _LOCAL_ENDPOINT


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_config_with_positions(**positions: dict) -> RoutingConfig:
    graph = {}
    for name, kwargs in positions.items():
        graph[name] = GraphPosition(**kwargs)
    return RoutingConfig(enabled=True, graph=graph)


def _local_pos(llm_config_name: str = "coder", **kwargs) -> dict:
    return {
        "provider": "custom:llm-local",
        "model": "qwen3-coder",
        "base_url": "http://127.0.0.1:58080/v1",
        "api_mode": "chat_completions",
        "llm_config_name": llm_config_name,
        **kwargs,
    }


def _cloud_pos(provider: str = "bedrock", model: str = "claude-opus") -> dict:
    return {
        "provider": provider,
        "model": model,
        "base_url": "",
        "api_mode": "anthropic_messages",
        "llm_config_name": "",
    }


# ── resolve() ────────────────────────────────────────────────────────────────

class TestResolve:
    def test_unknown_position_returns_none(self):
        cfg = _make_config_with_positions()
        resolver = ModelResolver(cfg)
        assert resolver.resolve("nonexistent") is None

    def test_local_position_resolved(self):
        cfg = _make_config_with_positions(interactive_lower=_local_pos("coder-next"))
        resolver = ModelResolver(cfg)
        result = resolver.resolve("interactive_lower")
        assert result is not None
        assert result.is_local is True
        assert result.llm_config_name == "coder-next"
        assert result.provider == "custom:llm-local"
        assert result.model == "qwen3-coder"
        assert result.base_url == "http://127.0.0.1:58080/v1"
        assert result.api_mode == "chat_completions"

    def test_cloud_position_resolved(self):
        cfg = _make_config_with_positions(upper=_cloud_pos("bedrock", "claude-opus-4"))
        resolver = ModelResolver(cfg)
        result = resolver.resolve("upper")
        assert result is not None
        assert result.is_local is False
        assert result.provider == "bedrock"
        assert result.model == "claude-opus-4"
        assert result.llm_config_name == ""

    def test_local_detected_by_127_base_url(self):
        """A position with 127.0.0.1 base_url but no llm_config_name is still local."""
        cfg = _make_config_with_positions(
            interactive_lower={
                "provider": "openai",
                "model": "local-llm",
                "base_url": "http://127.0.0.1:58080/v1",
                "api_mode": "chat_completions",
                "llm_config_name": "",
            }
        )
        resolver = ModelResolver(cfg)
        result = resolver.resolve("interactive_lower")
        assert result is not None
        assert result.is_local is True

    def test_local_detected_by_llm_config_name_alone(self):
        """A position with llm_config_name is local even without explicit base_url."""
        cfg = _make_config_with_positions(
            interactive_lower={
                "provider": "custom:llm-local",
                "model": "qwen3",
                "base_url": "",
                "api_mode": "",
                "llm_config_name": "coder",
            }
        )
        resolver = ModelResolver(cfg)
        result = resolver.resolve("interactive_lower")
        assert result is not None
        assert result.is_local is True
        # Default base_url for local models
        assert result.base_url == _LOCAL_ENDPOINT

    def test_default_api_mode_chat_completions_for_local(self):
        cfg = _make_config_with_positions(
            interactive_lower={
                "provider": "custom:llm-local",
                "model": "qwen3",
                "base_url": "http://127.0.0.1:58080/v1",
                "api_mode": "",  # not set
                "llm_config_name": "coder",
            }
        )
        resolver = ModelResolver(cfg)
        result = resolver.resolve("interactive_lower")
        assert result is not None
        assert result.api_mode == "chat_completions"

    def test_explicit_api_mode_preserved(self):
        cfg = _make_config_with_positions(upper=_cloud_pos())
        resolver = ModelResolver(cfg)
        result = resolver.resolve("upper")
        assert result is not None
        assert result.api_mode == "anthropic_messages"


# ── is_position_local() ───────────────────────────────────────────────────────

class TestIsPositionLocal:
    def test_local_by_llm_config_name(self):
        cfg = _make_config_with_positions(interactive_lower=_local_pos("coder"))
        resolver = ModelResolver(cfg)
        assert resolver.is_position_local("interactive_lower") is True

    def test_cloud_is_not_local(self):
        cfg = _make_config_with_positions(upper=_cloud_pos())
        resolver = ModelResolver(cfg)
        assert resolver.is_position_local("upper") is False

    def test_unknown_position_is_not_local(self):
        cfg = _make_config_with_positions()
        resolver = ModelResolver(cfg)
        assert resolver.is_position_local("ghost") is False


# ── current_local_model() ─────────────────────────────────────────────────────

class TestCurrentLocalModel:
    def test_parses_name_from_llm_status(self):
        # `llm status` prints plain text (colors auto-disabled off a TTY), not JSON.
        cfg = _make_config_with_positions()
        resolver = ModelResolver(cfg)
        with patch("agent.routing.model_resolver.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=(
                    "active:     coder-next\n"
                    "status:     running (pid=1234)\n"
                    "endpoint:   http://127.0.0.1:58080/v1\n"
                ),
            )
            result = resolver.current_local_model()
        assert result == "coder-next"

    def test_returns_none_when_active_but_not_running(self):
        # `active` names the last-selected config even after the server has
        # stopped — must not be mistaken for a live model.
        cfg = _make_config_with_positions()
        resolver = ModelResolver(cfg)
        with patch("agent.routing.model_resolver.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="active:     coder-next\nstatus:     not running\n",
            )
            result = resolver.current_local_model()
        assert result is None

    def test_returns_none_on_subprocess_failure(self):
        cfg = _make_config_with_positions()
        resolver = ModelResolver(cfg)
        with patch("agent.routing.model_resolver.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = resolver.current_local_model()
        assert result is None

    def test_returns_none_on_exception(self):
        cfg = _make_config_with_positions()
        resolver = ModelResolver(cfg)
        with patch("agent.routing.model_resolver.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("llm not found")
            result = resolver.current_local_model()
        assert result is None
