"""Integration tests for the ask_upper wiring (ADR-0040 §4).

Covers the three hooks that connect ``agent.routing.ask_upper`` to the agent:

* REGISTER — ``sync_ask_upper_registration`` offers the schema ONLY while a
  lower-tier routed model is active and strips it for upper-tier / disabled.
* DISPATCH — an ``ask_upper`` tool call round-trips to ``AskUpperTool.execute``
  using the model's argument names and returns the guidance string.
* ENCOURAGE — ``ask_upper_prompt_nudge`` yields the nudge only when gated on.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.routing.ask_upper import (
    ASK_UPPER_PROMPT_NUDGE,
    ASK_UPPER_TOOL_SCHEMA,
    ask_upper_prompt_nudge,
    get_or_create_ask_upper_tool,
    sync_ask_upper_registration,
)
from agent.routing.config import GraphPosition, RoutingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _routing_config(*, enabled=True):
    """A real routing config (tier-navigation must resolve) with a base local
    tier and an upper tier above it."""
    return RoutingConfig(
        enabled=enabled,
        graph={
            "interactive_lower": GraphPosition(
                provider="local", model="qwen", base_url="", api_key="", tier=1, alias="Alpha"
            ),
            "upper": GraphPosition(
                provider="bedrock", model="opus", base_url="", api_key="", tier=2, alias="Beta"
            ),
        },
    )


def _agent(provider, model):
    """A minimal agent stand-in with real ``tools`` / ``valid_tool_names``."""
    return SimpleNamespace(
        provider=provider,
        model=model,
        tools=[],
        valid_tool_names=set(),
        _ask_upper_tool=None,
    )


def _ask_upper_entries(tools):
    return [
        t
        for t in tools
        if isinstance(t, dict) and t.get("function", {}).get("name") == "ask_upper"
    ]


# ---------------------------------------------------------------------------
# REGISTER
# ---------------------------------------------------------------------------

class TestSyncAskUpperRegistration:
    @patch("agent.routing.config.load_routing_config")
    def test_lower_tier_registers_wrapped_schema(self, mock_config):
        mock_config.return_value = _routing_config()
        agent = _agent("local", "qwen")

        assert sync_ask_upper_registration(agent) is True

        entries = _ask_upper_entries(agent.tools)
        assert len(entries) == 1
        # Wrapped in the OpenAI envelope agent.tools uses, with the module schema.
        assert entries[0]["type"] == "function"
        assert entries[0]["function"] is ASK_UPPER_TOOL_SCHEMA
        assert entries[0]["function"]["name"] == "ask_upper"
        assert "ask_upper" in agent.valid_tool_names

    @patch("agent.routing.config.load_routing_config")
    def test_upper_tier_absent(self, mock_config):
        mock_config.return_value = _routing_config()
        agent = _agent("bedrock", "opus")

        assert sync_ask_upper_registration(agent) is False
        assert _ask_upper_entries(agent.tools) == []
        assert "ask_upper" not in agent.valid_tool_names

    @patch("agent.routing.config.load_routing_config")
    def test_disabled_routing_absent(self, mock_config):
        mock_config.return_value = _routing_config(enabled=False)
        agent = _agent("local", "qwen")

        assert sync_ask_upper_registration(agent) is False
        assert _ask_upper_entries(agent.tools) == []

    @patch("agent.routing.config.load_routing_config")
    def test_idempotent_no_duplicate(self, mock_config):
        mock_config.return_value = _routing_config()
        agent = _agent("local", "qwen")

        sync_ask_upper_registration(agent)
        sync_ask_upper_registration(agent)
        sync_ask_upper_registration(agent)

        assert len(_ask_upper_entries(agent.tools)) == 1

    @patch("agent.routing.config.load_routing_config")
    def test_strips_on_tier_swap_to_upper(self, mock_config):
        """Registered while lower, then removed once the model swaps to upper."""
        cfg = _routing_config()
        mock_config.return_value = cfg

        agent = _agent("local", "qwen")
        assert sync_ask_upper_registration(agent) is True
        assert len(_ask_upper_entries(agent.tools)) == 1

        # Model swaps to the upper tier between turns.
        agent.provider, agent.model = "bedrock", "opus"
        assert sync_ask_upper_registration(agent) is False
        assert _ask_upper_entries(agent.tools) == []
        assert "ask_upper" not in agent.valid_tool_names

    @patch("agent.routing.config.load_routing_config")
    def test_preserves_other_tools(self, mock_config):
        mock_config.return_value = _routing_config()
        agent = _agent("local", "qwen")
        other = {"type": "function", "function": {"name": "read_file"}}
        agent.tools.append(other)

        sync_ask_upper_registration(agent)
        assert other in agent.tools
        assert len(agent.tools) == 2


# ---------------------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------------------

class TestAskUpperDispatch:
    @patch("agent.auxiliary_client.call_llm")
    @patch("agent.routing.config.load_routing_config")
    def test_dispatch_round_trips_to_execute(self, mock_config, mock_call_llm):
        mock_config.return_value = _routing_config()

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Do X then Y."
        response.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        mock_call_llm.return_value = response

        agent = _agent("local", "qwen")

        # Mirror exactly what the tool_executor ask_upper branch does: resolve
        # the cached tool and call execute() with the model's argument names.
        tool = get_or_create_ask_upper_tool(agent)
        assert tool is not None
        assert tool.upper_provider == "bedrock"
        assert tool.upper_model == "opus"

        model_args = {
            "request_type": "plan",
            "question": "How should I structure this?",
            "context": "Building a parser.",
        }
        result = tool.execute(
            request_type=model_args.get("request_type", ""),
            question=model_args.get("question", ""),
            context=model_args.get("context", "") or "",
        )
        assert result == "Do X then Y."
        assert tool.budget.calls == 1

    @patch("agent.routing.config.load_routing_config")
    def test_dispatch_unavailable_when_no_upper(self, mock_config):
        cfg = MagicMock(enabled=True)
        cfg.graph = {}  # no upper position
        mock_config.return_value = cfg

        agent = _agent("local", "qwen")
        assert get_or_create_ask_upper_tool(agent) is None

    def test_execute_signature_matches_schema_params(self):
        """The dispatched kwargs are exactly the schema's declared params."""
        import inspect

        sig = inspect.signature(
            get_or_create_ask_upper_tool.__globals__["AskUpperTool"].execute
        )
        params = set(sig.parameters) - {"self"}
        assert params == {"request_type", "question", "context"}
        schema_props = set(ASK_UPPER_TOOL_SCHEMA["parameters"]["properties"])
        assert schema_props == {"request_type", "question", "context"}


# ---------------------------------------------------------------------------
# ENCOURAGE (prompt nudge)
# ---------------------------------------------------------------------------

class TestAskUpperPromptNudge:
    @patch("agent.routing.config.load_routing_config")
    def test_nudge_present_for_lower_tier(self, mock_config):
        mock_config.return_value = _routing_config()
        agent = _agent("local", "qwen")
        assert ask_upper_prompt_nudge(agent) == ASK_UPPER_PROMPT_NUDGE

    @patch("agent.routing.config.load_routing_config")
    def test_nudge_absent_for_upper_tier(self, mock_config):
        mock_config.return_value = _routing_config()
        agent = _agent("bedrock", "opus")
        assert ask_upper_prompt_nudge(agent) == ""

    @patch("agent.routing.config.load_routing_config")
    def test_nudge_absent_when_disabled(self, mock_config):
        mock_config.return_value = _routing_config(enabled=False)
        agent = _agent("local", "qwen")
        assert ask_upper_prompt_nudge(agent) == ""

    @patch("agent.routing.config.load_routing_config")
    def test_nudge_gated_identically_to_registration(self, mock_config):
        """The nudge and the tool are offered under the exact same condition."""
        mock_config.return_value = _routing_config()
        lower = _agent("local", "qwen")
        upper = _agent("bedrock", "opus")

        assert bool(ask_upper_prompt_nudge(lower)) == sync_ask_upper_registration(lower)
        assert bool(ask_upper_prompt_nudge(upper)) == sync_ask_upper_registration(upper)
