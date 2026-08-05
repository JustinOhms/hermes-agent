"""Tests for agent.routing.ask_upper — the ask_upper tool."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agent.routing.ask_upper import (
    ASK_UPPER_TOOL_SCHEMA,
    MENTOR_SYSTEM_PROMPT,
    AskUpperBudget,
    AskUpperTool,
    get_or_create_ask_upper_tool,
    should_register_ask_upper,
)
from agent.routing.config import GraphPosition, RoutingConfig


def _two_tier_config() -> RoutingConfig:
    """Real config (not a MagicMock) so tier-navigation helpers work: a base
    local tier and an upper tier above it."""
    return RoutingConfig(
        enabled=True,
        graph={
            "interactive_lower": GraphPosition(provider="local", model="qwen", tier=1, alias="Alpha"),
            "upper": GraphPosition(provider="bedrock", model="opus", base_url="", api_key="", tier=2, alias="Beta"),
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool():
    """Create an AskUpperTool with default config."""
    return AskUpperTool(
        upper_provider="bedrock",
        upper_model="us.anthropic.claude-opus-4-6-v1",
        soft_budget=3,
        hard_budget=5,
    )


@pytest.fixture
def mock_response():
    """Create a mock LLM response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Here is your guidance."
    response.usage = MagicMock()
    response.usage.prompt_tokens = 1500
    response.usage.completion_tokens = 200
    return response


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestAskUpperSchema:
    def test_schema_has_required_fields(self):
        assert ASK_UPPER_TOOL_SCHEMA["name"] == "ask_upper"
        params = ASK_UPPER_TOOL_SCHEMA["parameters"]
        assert "request_type" in params["properties"]
        assert "question" in params["properties"]
        assert "context" in params["properties"]
        assert params["required"] == ["request_type", "question"]

    def test_request_types_enum(self):
        enum = ASK_UPPER_TOOL_SCHEMA["parameters"]["properties"]["request_type"]["enum"]
        assert set(enum) == {"simplify", "plan", "verify", "distill", "explain"}


# ---------------------------------------------------------------------------
# Budget tests
# ---------------------------------------------------------------------------

class TestAskUpperBudget:
    def test_initial_state(self):
        budget = AskUpperBudget()
        assert budget.calls == 0
        assert not budget.exhausted
        assert not budget.over_soft_budget

    def test_soft_budget(self):
        budget = AskUpperBudget(soft_budget_calls=3)
        budget.calls = 2
        assert not budget.over_soft_budget
        budget.calls = 3
        assert budget.over_soft_budget

    def test_hard_budget(self):
        budget = AskUpperBudget(hard_budget_calls=5)
        budget.calls = 4
        assert not budget.exhausted
        budget.calls = 5
        assert budget.exhausted


# ---------------------------------------------------------------------------
# Tool execution tests
# ---------------------------------------------------------------------------

class TestAskUpperTool:
    @patch("agent.auxiliary_client.call_llm")
    def test_successful_call(self, mock_call_llm, tool, mock_response):
        mock_call_llm.return_value = mock_response

        result = tool.execute("verify", "Is my plan correct?", "Plan: do X then Y")
        assert result == "Here is your guidance."
        assert tool.budget.calls == 1
        assert tool.budget.total_input_tokens == 1500
        assert tool.budget.total_output_tokens == 200

    @patch("agent.auxiliary_client.call_llm")
    def test_call_with_all_request_types(self, mock_call_llm, tool, mock_response):
        mock_call_llm.return_value = mock_response

        for rtype in ["simplify", "plan", "verify", "distill", "explain"]:
            result = tool.execute(rtype, "Help me")
            assert "guidance" in result.lower() or "here" in result.lower()

        assert tool.budget.calls == 5

    def test_invalid_request_type(self, tool):
        result = tool.execute("invalid_type", "Help me")
        assert "ERROR" in result
        assert "Invalid request_type" in result
        assert tool.budget.calls == 0  # Should not count against budget

    @patch("agent.auxiliary_client.call_llm")
    def test_soft_budget_warning(self, mock_call_llm, tool, mock_response):
        mock_call_llm.return_value = mock_response

        # Use up to soft budget
        for _ in range(3):
            tool.execute("verify", "Check this")

        # Next call should include warning
        result = tool.execute("verify", "Check this too")
        assert "NOTE:" in result
        assert "ask_upper" in result
        assert "independently" in result

    @patch("agent.auxiliary_client.call_llm")
    def test_hard_budget_refuses(self, mock_call_llm, tool, mock_response):
        mock_call_llm.return_value = mock_response

        # Exhaust hard budget
        for _ in range(5):
            tool.execute("verify", "Check")

        # Should refuse
        result = tool.execute("verify", "One more")
        assert "BUDGET EXHAUSTED" in result
        assert tool.budget.calls == 5  # Should not increment past limit
        mock_call_llm.assert_called()  # Was called 5 times, not 6

    @patch("agent.auxiliary_client.call_llm")
    def test_context_truncation(self, mock_call_llm, tool, mock_response):
        mock_call_llm.return_value = mock_response

        long_context = "x" * 20000
        tool.execute("distill", "Summarize", long_context)

        # Verify the context was truncated in the call
        call_args = mock_call_llm.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert "[...context truncated]" in user_content

    @patch("agent.auxiliary_client.call_llm")
    def test_empty_response(self, mock_call_llm, tool):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = ""
        response.usage = None
        mock_call_llm.return_value = response

        result = tool.execute("verify", "Check")
        assert "ERROR" in result
        assert "empty response" in result

    @patch("agent.auxiliary_client.call_llm")
    def test_unreachable_upper_model(self, mock_call_llm, tool):
        mock_call_llm.side_effect = ConnectionError("Connection refused")

        result = tool.execute("verify", "Check this")
        assert "UNAVAILABLE" in result
        assert "Connection refused" in result
        assert "best judgment" in result

    @patch("agent.auxiliary_client.call_llm")
    def test_mentor_system_prompt_used(self, mock_call_llm, tool, mock_response):
        mock_call_llm.return_value = mock_response

        tool.execute("explain", "Why does X happen?")

        call_args = mock_call_llm.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == MENTOR_SYSTEM_PROMPT

    @patch("agent.auxiliary_client.call_llm")
    def test_provider_model_passed_correctly(self, mock_call_llm, tool, mock_response):
        mock_call_llm.return_value = mock_response

        tool.execute("plan", "Plan my day")

        call_args = mock_call_llm.call_args
        assert call_args.kwargs["provider"] == "bedrock"
        assert call_args.kwargs["model"] == "us.anthropic.claude-opus-4-6-v1"
        assert call_args.kwargs["max_tokens"] == 2000
        assert call_args.kwargs["temperature"] == 0.3

    def test_get_status(self, tool):
        status = tool.get_status()
        assert status["calls"] == 0
        assert status["soft_budget"] == 3
        assert status["hard_budget"] == 5
        assert not status["exhausted"]


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

class TestShouldRegisterAskUpper:
    @patch("agent.routing.config.load_routing_config")
    def test_disabled_routing(self, mock_config):
        mock_config.return_value = MagicMock(enabled=False)
        agent = MagicMock(provider="bedrock", model="opus")
        assert not should_register_ask_upper(agent)

    @patch("agent.routing.config.load_routing_config")
    def test_top_tier_not_registered(self, mock_config):
        # Model at the top tier has nothing above to ask → not registered.
        mock_config.return_value = _two_tier_config()
        agent = MagicMock(provider="bedrock", model="opus")
        assert not should_register_ask_upper(agent)

    @patch("agent.routing.config.load_routing_config")
    def test_lower_model_registered(self, mock_config):
        # Base tier has a tier above → registered.
        mock_config.return_value = _two_tier_config()
        agent = MagicMock(provider="local", model="qwen")
        assert should_register_ask_upper(agent)

    @patch("agent.routing.config.load_routing_config")
    def test_unknown_model_not_registered(self, mock_config):
        mock_config.return_value = _two_tier_config()
        agent = MagicMock(provider="openai", model="gpt-4")
        assert not should_register_ask_upper(agent)


class TestGetOrCreateAskUpperTool:
    @patch("agent.routing.config.load_routing_config")
    def test_creates_and_caches(self, mock_config):
        # Agent not found in the graph → falls back to the top tier to ask.
        mock_config.return_value = _two_tier_config()

        agent = MagicMock(spec=["_ask_upper_tool"])  # no provider/model attrs
        agent._ask_upper_tool = None

        tool = get_or_create_ask_upper_tool(agent)
        assert tool is not None
        assert tool.upper_provider == "bedrock"  # top tier ("upper")
        assert tool.upper_model == "opus"

    @patch("agent.routing.config.load_routing_config")
    def test_returns_cached(self, mock_config):
        existing_tool = MagicMock()
        agent = MagicMock()
        agent._ask_upper_tool = existing_tool

        result = get_or_create_ask_upper_tool(agent)
        assert result is existing_tool
        mock_config.assert_not_called()

    @patch("agent.routing.config.load_routing_config")
    def test_disabled_returns_none(self, mock_config):
        mock_config.return_value = MagicMock(enabled=False)
        agent = MagicMock()
        agent._ask_upper_tool = None

        assert get_or_create_ask_upper_tool(agent) is None
