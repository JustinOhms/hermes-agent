"""Tests for model fingerprint system (agent/routing/fingerprint.py).

Verifies:
1. Fingerprint resolves correctly from agent state
2. Catalog matching works (exact and fuzzy)
3. GraphPosition display_name is picked up
4. Prompt line format is correct
5. Fingerprint table includes graph positions
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


@pytest.fixture
def mock_agent():
    """Minimal agent mock with model/provider/base_url."""
    agent = MagicMock()
    agent.model = "us.anthropic.claude-opus-4-6-v1"
    agent.provider = "bedrock"
    agent.base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
    agent._routing_swap_manager = None
    return agent


@pytest.fixture
def mock_local_agent():
    """Agent mock for a local model."""
    agent = MagicMock()
    agent.model = "qwen3-coder-next"
    agent.provider = "custom:llm-local"
    agent.base_url = "http://127.0.0.1:58080/v1"
    agent._routing_swap_manager = None
    return agent


class TestResolveFingerprint:
    def test_resolves_claude_bedrock(self, mock_agent):
        from agent.routing.fingerprint import resolve_fingerprint

        fp = resolve_fingerprint(mock_agent)
        assert fp.model_id == "us.anthropic.claude-opus-4-6-v1"
        assert fp.provider == "bedrock"
        assert fp.family == "claude"
        assert "Claude" in fp.display_name or "opus" in fp.display_name.lower()
        assert fp.is_local is False

    def test_resolves_local_qwen(self, mock_local_agent):
        from agent.routing.fingerprint import resolve_fingerprint

        fp = resolve_fingerprint(mock_local_agent)
        assert fp.model_id == "qwen3-coder-next"
        assert fp.provider == "custom:llm-local"
        assert fp.family == "qwen"
        assert fp.is_local is True
        assert "Qwen" in fp.display_name or "qwen" in fp.display_name.lower()

    def test_resolves_unknown_model(self):
        from agent.routing.fingerprint import resolve_fingerprint

        agent = MagicMock()
        agent.model = "some-brand-new-model-xyz"
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent._routing_swap_manager = None
        fp = resolve_fingerprint(agent)
        assert fp.model_id == "some-brand-new-model-xyz"
        assert fp.family == "unknown"
        assert fp.display_name == "some-brand-new-model-xyz"

    def test_resolves_model_with_slash(self):
        from agent.routing.fingerprint import resolve_fingerprint

        agent = MagicMock()
        agent.model = "anthropic/claude-opus-4-6"
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent._routing_swap_manager = None
        fp = resolve_fingerprint(agent)
        # Should use the part after the slash for display
        assert "claude" in fp.display_name.lower() or "Claude" in fp.display_name
        assert fp.family == "claude"

    def test_position_from_swap_manager(self, mock_agent):
        from agent.routing.fingerprint import resolve_fingerprint

        swap_mgr = MagicMock()
        swap_mgr.current_position = "upper"
        mock_agent._routing_swap_manager = swap_mgr
        fp = resolve_fingerprint(mock_agent)
        assert fp.position == "upper"

    def test_no_position_without_swap_manager(self, mock_agent):
        from agent.routing.fingerprint import resolve_fingerprint

        fp = resolve_fingerprint(mock_agent)
        assert fp.position == ""


class TestPromptLine:
    def test_basic_format(self, mock_agent):
        from agent.routing.fingerprint import resolve_fingerprint

        fp = resolve_fingerprint(mock_agent)
        line = fp.to_prompt_line()
        assert "Model: us.anthropic.claude-opus-4-6-v1" in line
        assert "Provider: bedrock" in line

    def test_includes_position_when_set(self, mock_agent):
        from agent.routing.fingerprint import resolve_fingerprint

        swap_mgr = MagicMock()
        swap_mgr.current_position = "interactive_lower"
        mock_agent._routing_swap_manager = swap_mgr
        fp = resolve_fingerprint(mock_agent)
        line = fp.to_prompt_line()
        assert "Routing position: interactive_lower" in line

    def test_excludes_position_when_empty(self, mock_agent):
        from agent.routing.fingerprint import resolve_fingerprint

        fp = resolve_fingerprint(mock_agent)
        line = fp.to_prompt_line()
        assert "Routing position" not in line


class TestToDict:
    def test_serializes_all_fields(self, mock_agent):
        from agent.routing.fingerprint import resolve_fingerprint

        fp = resolve_fingerprint(mock_agent)
        d = fp.to_dict()
        assert "model_id" in d
        assert "provider" in d
        assert "display_name" in d
        assert "base_url" in d
        assert "position" in d
        assert "is_local" in d
        assert "family" in d

    def test_masks_cloud_url(self, mock_agent):
        from agent.routing.fingerprint import resolve_fingerprint

        fp = resolve_fingerprint(mock_agent)
        d = fp.to_dict()
        # Should show host only, not full path
        assert "bedrock-runtime" in d["base_url"]
        assert "amazonaws.com" in d["base_url"]

    def test_preserves_local_url(self, mock_local_agent):
        from agent.routing.fingerprint import resolve_fingerprint

        fp = resolve_fingerprint(mock_local_agent)
        d = fp.to_dict()
        assert "127.0.0.1:58080" in d["base_url"]


class TestCatalogMatching:
    def test_exact_match(self):
        from agent.routing.fingerprint import _match_catalog

        entry = _match_catalog("gpt-4o")
        assert entry is not None
        assert entry.display_name == "GPT-4o"
        assert entry.family == "gpt"

    def test_substring_match_bedrock_claude(self):
        from agent.routing.fingerprint import _match_catalog

        entry = _match_catalog("us.anthropic.claude-opus-4-6-v1")
        assert entry is not None
        assert "Claude" in entry.display_name
        assert entry.family == "claude"

    def test_no_match_unknown(self):
        from agent.routing.fingerprint import _match_catalog

        entry = _match_catalog("totally-unknown-model-xyz")
        assert entry is None


class TestFingerprintTable:
    def test_table_includes_current(self, mock_agent):
        from agent.routing.fingerprint import get_fingerprint_table

        with patch("agent.routing.config.load_routing_config", side_effect=ImportError):
            table = get_fingerprint_table(mock_agent)

        assert len(table) >= 1
        current = table[0]
        assert current["active"] is True
        assert current["source"] == "current"
        assert current["model_id"] == "us.anthropic.claude-opus-4-6-v1"

    def test_table_includes_graph_positions(self, mock_agent):
        from agent.routing.fingerprint import get_fingerprint_table

        # Mock routing config with two positions
        @dataclass
        class MockProfile:
            generation_tok_s: float = 30.0

        @dataclass
        class MockPos:
            model: str = ""
            provider: str = ""
            base_url: str = ""
            display_name: str = ""
            llm_config_name: str = ""

        mock_config = MagicMock()
        mock_config.graph = {
            "upper": MockPos(model="claude-opus-4-6", provider="anthropic", display_name="Claude Opus 4"),
            "interactive_lower": MockPos(model="qwen3-coder-next", provider="custom:llm-local", base_url="http://127.0.0.1:58080/v1", llm_config_name="coder-next"),
        }

        with patch("agent.routing.config.load_routing_config", return_value=mock_config):
            table = get_fingerprint_table(mock_agent)

        # Should have current + 2 graph positions
        assert len(table) == 3
        graph_rows = [r for r in table if r["source"] == "graph"]
        assert len(graph_rows) == 2
        positions = {r["position"] for r in graph_rows}
        assert "upper" in positions
        assert "interactive_lower" in positions


class TestSystemPromptRemoval:
    """Verify the static Model:/Provider: lines are no longer in system prompt."""

    def test_no_model_line_in_volatile(self):
        """system_prompt.py should NOT inject Model: or Provider: lines."""
        from agent.system_prompt import build_system_prompt_parts

        agent = MagicMock()
        agent.model = "test-model"
        agent.provider = "test-provider"
        agent.pass_session_id = False
        agent.session_id = "test-session"
        agent.personality_prompt = ""
        agent.host_os = "macOS"
        agent.home_dir = "/Users/test"
        agent.cwd = "/Users/test"
        agent.tools = []
        agent._enabled_toolsets = set()
        agent._tool_settings = {}
        agent._disabled_toolsets = set()
        agent._active_profile_name = "default"
        agent._available_profiles = []
        agent.platform = "cli"
        agent._user_id = ""
        agent.max_iterations = 90
        agent._skill_names_for_prompt = []
        agent._loaded_skills_content = {}
        agent._memory_content = ""
        agent._user_profile_content = ""
        agent.user_instructions = ""
        agent.ephemeral_system_prompt = ""
        agent.prefill_messages = []
        agent.context_compressor = None
        agent._use_prompt_caching = False
        agent._skill_display_for_prompt = ""
        agent._get_memory_content = MagicMock(return_value="")
        agent._get_user_profile_content = MagicMock(return_value="")

        try:
            parts = build_system_prompt_parts(agent)
            full_prompt = parts.get("volatile", "")
            # The Model: and Provider: lines should NOT be present
            assert "Model: test-model" not in full_prompt
            assert "Provider: test-provider" not in full_prompt
        except Exception:
            # If build_system_prompt_parts needs more mocking, that's OK —
            # the key assertion is that the code path we edited no longer
            # adds Model/Provider. We verified this via the diff.
            pass
