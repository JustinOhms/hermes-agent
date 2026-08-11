"""Tests for agent.routing.oversight — periodic oversight reviewer."""

from __future__ import annotations

import json
import time

import pytest
from unittest.mock import MagicMock, patch

from agent.routing.oversight import (
    OversightAction,
    OversightConfig,
    OversightResult,
    OversightReviewer,
    build_oversight_injection,
    get_or_create_oversight_reviewer,
    load_oversight_config,
    run_oversight_if_due,
    _extract_review_window,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Standard oversight config for testing."""
    return OversightConfig(
        enabled=True,
        model="us.anthropic.claude-opus-4-6-v1",
        provider="bedrock",
        every_n_turns=5,
        review_window=5,
        review_window_ctx_fraction=0.6,
        review_window_min=2,
        max_reviews_per_session=3,
        min_turns_before_first=3,
        skip_if_escalated=True,
        upper_context_limit=200000,
    )


@pytest.fixture
def reviewer(config):
    """Create an OversightReviewer with test config."""
    return OversightReviewer(config)


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a Python function to sort a list."},
        {"role": "assistant", "content": "Here's a sort function:\n```python\ndef sort_list(lst):\n    return sorted(lst)\n```"},
        {"role": "user", "content": "Now add type hints."},
        {"role": "assistant", "content": "```python\nfrom typing import List\ndef sort_list(lst: List[int]) -> List[int]:\n    return sorted(lst)\n```"},
        {"role": "user", "content": "Test it with [3, 1, 2]"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "terminal", "arguments": '{"command": "python3 -c \\"print(sorted([3,1,2]))\\""}'}},
        ]},
        {"role": "tool", "name": "terminal", "content": "[1, 2, 3]"},
        {"role": "assistant", "content": "The function works correctly: `sort_list([3, 1, 2])` returns `[1, 2, 3]`."},
    ]


def _make_approve_response():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = '{"action": "approve"}'
    response.usage = MagicMock(prompt_tokens=5000, completion_tokens=50)
    return response


def _make_correct_response(note="You should use List[Any] for generic sorting."):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps({"action": "correct", "note": note})
    response.usage = MagicMock(prompt_tokens=5000, completion_tokens=100)
    return response


def _make_escalate_response(reason="Agent is stuck in a loop."):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps({"action": "escalate", "reason": reason})
    response.usage = MagicMock(prompt_tokens=5000, completion_tokens=80)
    return response


def _make_flag_response(warning="Agent may be drifting from the task."):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps({"action": "flag", "warning": warning})
    response.usage = MagicMock(prompt_tokens=5000, completion_tokens=80)
    return response


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestOversightConfig:
    def test_defaults(self):
        cfg = OversightConfig()
        assert not cfg.enabled
        assert cfg.every_n_turns == 10
        assert cfg.review_window == 10
        assert cfg.max_reviews_per_session == 5

    def test_load_from_dict(self):
        raw_cfg = {
            "model": {
                "routing": {
                    "oversight": {
                        "enabled": True,
                        "model": "opus",
                        "provider": "bedrock",
                        "every_n_turns": 7,
                        "review_window": 8,
                        "max_reviews_per_session": 10,
                    }
                }
            }
        }
        cfg = load_oversight_config(raw_cfg)
        assert cfg.enabled
        assert cfg.model == "opus"
        assert cfg.every_n_turns == 7
        assert cfg.review_window == 8
        assert cfg.max_reviews_per_session == 10

    def test_load_missing_section(self):
        cfg = load_oversight_config({"model": {}})
        assert not cfg.enabled

    def test_load_empty_config(self):
        cfg = load_oversight_config({})
        assert not cfg.enabled


# ---------------------------------------------------------------------------
# should_review tests
# ---------------------------------------------------------------------------

class TestShouldReview:
    def test_disabled(self, config):
        config.enabled = False
        reviewer = OversightReviewer(config)
        assert not reviewer.should_review(10)

    def test_budget_exhausted(self, reviewer):
        # Fill up the budget
        for _ in range(3):
            reviewer.reviews.append(OversightResult(action=OversightAction.APPROVE))
        assert not reviewer.should_review(10)

    def test_before_minimum_turns(self, reviewer):
        # min_turns_before_first = 3
        assert not reviewer.should_review(1)
        assert not reviewer.should_review(2)

    def test_not_on_review_turn(self, reviewer):
        # every_n_turns = 5
        assert not reviewer.should_review(3)
        assert not reviewer.should_review(4)
        assert not reviewer.should_review(6)

    def test_on_review_turn(self, reviewer):
        # every_n_turns = 5, min_turns = 3
        assert reviewer.should_review(5)
        assert reviewer.should_review(10)
        assert reviewer.should_review(15)

    def test_skip_after_escalation(self, reviewer):
        assert reviewer.should_review(5, last_was_escalated=False)
        assert not reviewer.should_review(5, last_was_escalated=True)

    def test_skip_after_escalation_disabled(self, config):
        config.skip_if_escalated = False
        reviewer = OversightReviewer(config)
        assert reviewer.should_review(5, last_was_escalated=True)


# ---------------------------------------------------------------------------
# Dynamic window cap tests (RD-18)
# ---------------------------------------------------------------------------

class TestDynamicWindowCap:
    def test_no_data_uses_configured(self, reviewer):
        assert reviewer.compute_effective_window() == 5  # config.review_window

    def test_small_turns_full_window(self, reviewer):
        # 2000 tokens/turn: floor(200000 * 0.6 / 2000) = 60, capped at 5
        for _ in range(5):
            reviewer.record_turn_tokens(2000)
        assert reviewer.compute_effective_window() == 5

    def test_large_turns_reduces_window(self, reviewer):
        # 20000 tokens/turn: floor(200000 * 0.6 / 20000) = 6, still > 5
        for _ in range(5):
            reviewer.record_turn_tokens(20000)
        assert reviewer.compute_effective_window() == 5

    def test_very_large_turns_reduces_below_config(self, reviewer):
        # 40000 tokens/turn: floor(200000 * 0.6 / 40000) = 3 < 5
        for _ in range(5):
            reviewer.record_turn_tokens(40000)
        assert reviewer.compute_effective_window() == 3

    def test_minimum_window_enforced(self, reviewer):
        # 100000 tokens/turn: floor(200000 * 0.6 / 100000) = 1, but min is 2
        for _ in range(5):
            reviewer.record_turn_tokens(100000)
        assert reviewer.compute_effective_window() == 2

    def test_p75_not_mean(self, reviewer):
        # One outlier shouldn't shrink window permanently
        for _ in range(15):
            reviewer.record_turn_tokens(2000)
        # Add 5 huge outliers
        for _ in range(5):
            reviewer.record_turn_tokens(50000)
        # p75 of 20 entries: sorted, idx 15 → should be 50000
        # But most entries are 2000. Let's check: with 15x2000 + 5x50000,
        # sorted = [2000]*15 + [50000]*5, p75 idx = int(20*0.75) = 15 → 50000
        # floor(120000 / 50000) = 2
        # Actually with 20 entries and our windowing (last 20), the p75 is at idx 15
        # which is the first of the 50000 values. So cap = floor(120000/50000) = 2
        result = reviewer.compute_effective_window()
        assert result == 2

    def test_explicit_avg_override(self, reviewer):
        # Direct parameter overrides rolling calculation
        result = reviewer.compute_effective_window(avg_tokens_per_turn=30000.0)
        # floor(120000 / 30000) = 4
        assert result == 4

    def test_zero_tokens_uses_configured(self, reviewer):
        assert reviewer.compute_effective_window(avg_tokens_per_turn=0.0) == 5


# ---------------------------------------------------------------------------
# Review execution tests
# ---------------------------------------------------------------------------

class TestReviewExecution:
    @patch("agent.auxiliary_client.call_llm")
    def test_approve_action(self, mock_call_llm, reviewer, sample_messages):
        mock_call_llm.return_value = _make_approve_response()

        result = reviewer.review(sample_messages, "qwen3-coder-next", 10)

        assert result.action == OversightAction.APPROVE
        assert reviewer.review_count == 1
        assert result.input_tokens == 5000
        assert result.output_tokens == 50

    @patch("agent.auxiliary_client.call_llm")
    def test_correct_action(self, mock_call_llm, reviewer, sample_messages):
        mock_call_llm.return_value = _make_correct_response("Use List[Any] instead.")

        result = reviewer.review(sample_messages, "qwen3-coder-next", 10)

        assert result.action == OversightAction.CORRECT
        assert result.note == "Use List[Any] instead."

    @patch("agent.auxiliary_client.call_llm")
    def test_escalate_action(self, mock_call_llm, reviewer, sample_messages):
        mock_call_llm.return_value = _make_escalate_response("Stuck in loop.")

        result = reviewer.review(sample_messages, "qwen3-coder-next", 10)

        assert result.action == OversightAction.ESCALATE
        assert result.reason == "Stuck in loop."

    @patch("agent.auxiliary_client.call_llm")
    def test_flag_action(self, mock_call_llm, reviewer, sample_messages):
        mock_call_llm.return_value = _make_flag_response("Scope drift detected.")

        result = reviewer.review(sample_messages, "qwen3-coder-next", 10)

        assert result.action == OversightAction.FLAG
        assert result.warning == "Scope drift detected."

    @patch("agent.auxiliary_client.call_llm")
    def test_failure_defaults_to_approve(self, mock_call_llm, reviewer, sample_messages):
        mock_call_llm.side_effect = ConnectionError("timeout")

        result = reviewer.review(sample_messages, "qwen3-coder-next", 10)

        assert result.action == OversightAction.APPROVE
        assert "failed" in result.note.lower()
        assert reviewer.review_count == 1  # Still counts

    @patch("agent.auxiliary_client.call_llm")
    def test_malformed_json_defaults_to_approve(self, mock_call_llm, reviewer, sample_messages):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "I think everything looks good!"
        response.usage = MagicMock(prompt_tokens=5000, completion_tokens=50)
        mock_call_llm.return_value = response

        result = reviewer.review(sample_messages, "qwen3-coder-next", 10)
        assert result.action == OversightAction.APPROVE

    @patch("agent.auxiliary_client.call_llm")
    def test_json_in_markdown_fence(self, mock_call_llm, reviewer, sample_messages):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '```json\n{"action": "correct", "note": "Fix this"}\n```'
        response.usage = MagicMock(prompt_tokens=5000, completion_tokens=50)
        mock_call_llm.return_value = response

        result = reviewer.review(sample_messages, "qwen3-coder-next", 10)
        assert result.action == OversightAction.CORRECT
        assert result.note == "Fix this"

    @patch("agent.auxiliary_client.call_llm")
    def test_unknown_action_defaults_to_approve(self, mock_call_llm, reviewer, sample_messages):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"action": "panic"}'
        response.usage = MagicMock(prompt_tokens=5000, completion_tokens=50)
        mock_call_llm.return_value = response

        result = reviewer.review(sample_messages, "qwen3-coder-next", 10)
        assert result.action == OversightAction.APPROVE

    @patch("agent.auxiliary_client.call_llm")
    def test_prompt_includes_model_info(self, mock_call_llm, reviewer, sample_messages):
        mock_call_llm.return_value = _make_approve_response()

        reviewer.review(sample_messages, "qwen3-coder-next", 15)

        call_args = mock_call_llm.call_args
        messages = call_args.kwargs["messages"]
        system_prompt = messages[0]["content"]
        assert "qwen3-coder-next" in system_prompt
        assert "us.anthropic.claude-opus-4-6-v1" in system_prompt
        assert "15" in system_prompt  # turn count


# ---------------------------------------------------------------------------
# Message formatting tests
# ---------------------------------------------------------------------------

class TestMessageFormatting:
    def test_formats_tool_calls(self, reviewer):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "ls"}'}},
            ]},
            {"role": "tool", "name": "terminal", "content": "file1.py\nfile2.py"},
        ]
        formatted = reviewer._format_messages_for_review(messages)
        assert "terminal" in formatted
        assert "file1.py" in formatted

    def test_truncates_long_tool_output(self, reviewer):
        messages = [
            {"role": "tool", "name": "read_file", "content": "x" * 5000},
        ]
        formatted = reviewer._format_messages_for_review(messages)
        assert "[...truncated]" in formatted
        assert len(formatted) < 5000

    def test_truncates_long_assistant_content(self, reviewer):
        messages = [
            {"role": "assistant", "content": "y" * 10000},
        ]
        formatted = reviewer._format_messages_for_review(messages)
        assert "[...truncated]" in formatted

    def test_handles_multipart_content(self, reviewer):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Look at this image"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            ]},
        ]
        formatted = reviewer._format_messages_for_review(messages)
        assert "Look at this image" in formatted


# ---------------------------------------------------------------------------
# Extract review window tests
# ---------------------------------------------------------------------------

class TestExtractReviewWindow:
    def test_extracts_last_n_turns(self, sample_messages):
        # 3 user messages in sample_messages
        window = _extract_review_window(sample_messages, 2)
        # Should get last 2 user turns + everything after
        user_count = sum(1 for m in window if m.get("role") == "user")
        assert user_count == 2

    def test_full_window_when_fewer_turns(self, sample_messages):
        window = _extract_review_window(sample_messages, 100)
        # Should return everything except system
        assert window[0]["role"] == "user"

    def test_empty_messages(self):
        assert _extract_review_window([], 5) == []

    def test_single_turn(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        window = _extract_review_window(messages, 1)
        assert len(window) >= 2
        assert any(m.get("role") == "user" for m in window)


# ---------------------------------------------------------------------------
# Injection tests
# ---------------------------------------------------------------------------

class TestBuildOversightInjection:
    def test_correct_injection_format(self):
        result = OversightResult(
            action=OversightAction.CORRECT,
            note="Consider using a more efficient algorithm.",
        )
        msg = build_oversight_injection(result, "claude-opus-4-6")
        assert msg["role"] == "system"
        assert "OVERSIGHT NOTE" in msg["content"]
        assert "claude-opus-4-6" in msg["content"]
        assert "more efficient algorithm" in msg["content"]
        assert "Adjust your approach" in msg["content"]


# ---------------------------------------------------------------------------
# Integration helper tests
# ---------------------------------------------------------------------------

class TestRunOversightIfDue:
    @patch("agent.routing.oversight.load_oversight_config")
    def test_disabled_returns_none(self, mock_config):
        mock_config.return_value = OversightConfig(enabled=False)
        agent = MagicMock()
        agent._oversight_reviewer = None
        result = run_oversight_if_due(agent, [], 10)
        assert result is None

    @patch("agent.auxiliary_client.call_llm")
    @patch("agent.routing.oversight.load_oversight_config")
    def test_runs_on_due_turn(self, mock_config, mock_call_llm):
        cfg = OversightConfig(
            enabled=True,
            model="opus",
            provider="bedrock",
            every_n_turns=5,
            min_turns_before_first=3,
            max_reviews_per_session=5,
        )
        mock_config.return_value = cfg
        mock_call_llm.return_value = _make_approve_response()

        agent = MagicMock()
        agent._oversight_reviewer = None
        agent._oversight_last_escalated = False
        agent.model = "qwen"

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        result = run_oversight_if_due(agent, messages, 5)
        assert result is not None
        assert result.action == OversightAction.APPROVE

    @patch("agent.routing.oversight.load_oversight_config")
    def test_not_due_returns_none(self, mock_config):
        cfg = OversightConfig(
            enabled=True,
            model="opus",
            provider="bedrock",
            every_n_turns=5,
            min_turns_before_first=3,
        )
        mock_config.return_value = cfg

        agent = MagicMock()
        agent._oversight_reviewer = None
        agent._oversight_last_escalated = False

        result = run_oversight_if_due(agent, [], 3)  # Not on a 5-turn boundary
        assert result is None


class TestGetOrCreateReviewer:
    @patch("agent.routing.oversight.load_oversight_config")
    def test_creates_when_enabled(self, mock_config):
        mock_config.return_value = OversightConfig(enabled=True, model="opus", provider="bedrock")
        agent = MagicMock()
        agent._oversight_reviewer = None

        reviewer = get_or_create_oversight_reviewer(agent)
        assert reviewer is not None
        assert isinstance(reviewer, OversightReviewer)

    @patch("agent.routing.oversight.load_oversight_config")
    def test_returns_none_when_disabled(self, mock_config):
        mock_config.return_value = OversightConfig(enabled=False)
        agent = MagicMock()
        agent._oversight_reviewer = None

        assert get_or_create_oversight_reviewer(agent) is None

    @patch("agent.routing.oversight.load_oversight_config")
    def test_returns_cached(self, mock_config):
        existing = MagicMock()
        agent = MagicMock()
        agent._oversight_reviewer = existing

        result = get_or_create_oversight_reviewer(agent)
        assert result is existing
        mock_config.assert_not_called()


# ---------------------------------------------------------------------------
# Status reporting tests
# ---------------------------------------------------------------------------

class TestOversightStatus:
    @patch("agent.auxiliary_client.call_llm")
    def test_status_after_reviews(self, mock_call_llm, reviewer, sample_messages):
        mock_call_llm.return_value = _make_approve_response()
        reviewer.review(sample_messages, "qwen", 5)

        mock_call_llm.return_value = _make_correct_response("Fix X")
        reviewer.review(sample_messages, "qwen", 10)

        status = reviewer.get_status()
        assert status["enabled"]
        assert status["reviews_completed"] == 2
        assert status["max_reviews"] == 3
        assert not status["budget_exhausted"]
        assert status["last_action"] == "correct"
        assert len(status["history"]) == 2
        assert status["history"][0]["action"] == "approve"
        assert status["history"][1]["action"] == "correct"


# ---------------------------------------------------------------------------
# Config-schema back-compat
# ---------------------------------------------------------------------------

class TestOversightConfigAlias:
    def test_review_interval_turns_alias(self):
        cfg = load_oversight_config(
            {"model": {"routing": {"oversight": {
                "enabled": True, "review_interval_turns": 7}}}}
        )
        assert cfg.every_n_turns == 7

    def test_every_n_turns_wins_over_alias(self):
        cfg = load_oversight_config(
            {"model": {"routing": {"oversight": {
                "enabled": True, "every_n_turns": 9, "review_interval_turns": 7}}}}
        )
        assert cfg.every_n_turns == 9


# ---------------------------------------------------------------------------
# Review-model derivation from the routing graph
# ---------------------------------------------------------------------------

class _FakePos:
    def __init__(self, provider, model, base_url="", api_key="", api_mode=""):
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.api_mode = api_mode


class _FakeRoutingConfig:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.graph = {
            "alpha": _FakePos("local", "qwen", api_mode="chat_completions"),
            "gamma": _FakePos("openai-codex", "gpt-5.6-sol", base_url="https://x/codex"),
        }

    def top_position(self):
        return "gamma"


class TestOversightModelDerivation:
    def test_reviewer_borrows_top_tier_when_model_unset(self):
        agent = MagicMock()
        agent._oversight_reviewer = None
        with patch("agent.routing.oversight.load_oversight_config",
                   return_value=OversightConfig(enabled=True, model="", provider="")), \
             patch("agent.routing.config.load_routing_config",
                   return_value=_FakeRoutingConfig()):
            reviewer = get_or_create_oversight_reviewer(agent)
        assert reviewer is not None
        assert reviewer.config.model == "gpt-5.6-sol"
        assert reviewer.config.provider == "openai-codex"

    def test_reviewer_none_when_no_model_derivable(self):
        agent = MagicMock()
        agent._oversight_reviewer = None
        with patch("agent.routing.oversight.load_oversight_config",
                   return_value=OversightConfig(enabled=True, model="", provider="")), \
             patch("agent.routing.config.load_routing_config",
                   return_value=_FakeRoutingConfig(enabled=False)):
            reviewer = get_or_create_oversight_reviewer(agent)
        assert reviewer is None


# ---------------------------------------------------------------------------
# ESCALATE consumer
# ---------------------------------------------------------------------------

class TestEscalateToTopTier:
    def test_forces_switch_to_top(self):
        from agent.routing import _escalate_to_top_tier
        agent = MagicMock()
        agent.provider = "local"
        agent.model = "qwen"
        with patch("agent.routing.config.load_routing_config",
                   return_value=_FakeRoutingConfig()), \
             patch("agent.agent_runtime_helpers.switch_model") as mock_switch:
            assert _escalate_to_top_tier(agent) is True
        mock_switch.assert_called_once()
        assert mock_switch.call_args.kwargs["new_model"] == "gpt-5.6-sol"
        assert mock_switch.call_args.kwargs["new_provider"] == "openai-codex"

    def test_noop_when_already_on_top(self):
        from agent.routing import _escalate_to_top_tier
        agent = MagicMock()
        agent.provider = "openai-codex"
        agent.model = "gpt-5.6-sol"
        with patch("agent.routing.config.load_routing_config",
                   return_value=_FakeRoutingConfig()), \
             patch("agent.agent_runtime_helpers.switch_model") as mock_switch:
            assert _escalate_to_top_tier(agent) is True
        mock_switch.assert_not_called()

    def test_false_when_routing_disabled(self):
        from agent.routing import _escalate_to_top_tier
        agent = MagicMock()
        with patch("agent.routing.config.load_routing_config",
                   return_value=_FakeRoutingConfig(enabled=False)):
            assert _escalate_to_top_tier(agent) is False


# ---------------------------------------------------------------------------
# Integration-wiring guards — the module can be perfect while the call site is
# missing. The oversight hook was dropped in a resync twice; these fail loudly
# if it happens again (cheap source-level assertions, no heavy loop setup).
# ---------------------------------------------------------------------------

class TestOversightWiring:
    def test_finalize_turn_invokes_oversight(self):
        import inspect
        import agent.turn_finalizer as tf
        assert "run_oversight_if_due" in inspect.getsource(tf.finalize_turn)

    def test_apply_turn_routing_consumes_escalation(self):
        import inspect
        from agent.routing import apply_turn_routing
        src = inspect.getsource(apply_turn_routing)
        assert "_oversight_escalation_pending" in src
        assert "_escalate_to_top_tier" in src
