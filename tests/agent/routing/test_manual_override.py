"""Manual routing override honored by the engine.

Regression guard for the attribute-name disconnect between the TUI /routing
command and the routing engine. The TUI (`_handle_routing_command`) writes
`agent._routing_explicit_override` (a manual position pin from `/routing swap
<pos>`) and `agent._routing_mode_override` (from `/routing mode <m>`). The
engine's `get_routing_decision` MUST honor both — otherwise a manual swap is
reverted by the next turn's auto-routing and a mode override is ignored.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.routing import get_routing_decision
from agent.routing.config import GraphPosition, GraphPositionProfile, RoutingConfig
from agent.routing.interaction_mode import InteractionMode


def _local_pos(name: str, model: str = "") -> GraphPosition:
    return GraphPosition(
        provider="custom:llm-local",
        model=model or f"model-{name}",
        base_url="http://127.0.0.1:58080/v1",
        api_mode="chat_completions",
        llm_config_name=name,
        profile=GraphPositionProfile(ttft_p90_ms=800.0, generation_tok_s=60.0),
    )


def _upper_pos() -> GraphPosition:
    return GraphPosition(
        provider="openai-codex",
        model="gpt-5.6-terra",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="",
        llm_config_name="",
    )


def _cfg() -> RoutingConfig:
    return RoutingConfig(
        enabled=True,
        graph={
            "interactive_lower": _local_pos("coder", "qwen3-coder"),
            "autonomous_lower": _local_pos("prose", "qwen3-prose"),
            "upper": _upper_pos(),
            "fast_fallback": _local_pos("fast", "qwen3-fast"),
        },
    )


def _agent(**kw) -> SimpleNamespace:
    d = {
        "provider": "custom:llm-local",
        "model": "qwen3-coder",  # SwapManager inits current_position=interactive_lower
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
    d.update(kw)
    return SimpleNamespace(**d)


class TestManualPositionPin:
    def test_pin_forces_position_over_auto_routing(self):
        """A pin to 'upper' overrides what complexity-based routing would pick."""
        agent = _agent(_routing_explicit_override="upper")
        with patch("agent.routing.load_routing_config", return_value=_cfg()):
            decision = get_routing_decision(agent, "hello")  # trivial → normally a lower
        assert decision is not None
        assert decision.target_position == "upper"
        assert decision.swap_required is True  # current is interactive_lower

    def test_invalid_pin_is_ignored(self):
        agent = _agent(_routing_explicit_override="does-not-exist")
        with patch("agent.routing.load_routing_config", return_value=_cfg()):
            decision = get_routing_decision(agent, "hello")
        assert decision is not None
        assert decision.target_position in _cfg().graph
        assert decision.target_position != "does-not-exist"

    def test_no_pin_leaves_auto_routing(self):
        agent = _agent()
        with patch("agent.routing.load_routing_config", return_value=_cfg()):
            decision = get_routing_decision(agent, "hello")
        assert decision is not None
        assert decision.target_position in _cfg().graph


class TestModeOverrideReconciliation:
    def test_routing_mode_override_name_is_honored(self):
        """The TUI writes `_routing_mode_override`; the engine must read it."""
        agent = _agent(_routing_mode_override="autonomous")
        with patch("agent.routing.load_routing_config", return_value=_cfg()):
            decision = get_routing_decision(agent, "hello")
        assert decision is not None
        assert decision.interaction_mode == InteractionMode.AUTONOMOUS
