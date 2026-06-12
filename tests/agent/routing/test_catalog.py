"""Tests for agent.routing.catalog — live-source model resolution."""
import pytest
from unittest.mock import patch, MagicMock


class TestCatalogExtractCapabilities:
    """Test capability extraction from raw metadata dicts."""

    def test_openrouter_style_metadata(self):
        from agent.routing.catalog import extract_capabilities

        metadata = {
            "id": "anthropic/claude-opus-4-6",
            "context_length": 200000,
            "pricing": {"input": "15", "output": "75", "cache_read": "1.5"},
            "architecture": {"modality": "text+image->text", "flags": {"reasoning": True}},
            "knowledge_cutoff": "2025-04-01",
        }
        caps = extract_capabilities(metadata)
        assert caps.context_length == 200000
        assert caps.pricing["input"] == "15"
        assert caps.pricing["output"] == "75"
        assert caps.has_knowledge_cutoff == "2025-04-01"

    def test_aimodeldir_style_metadata(self):
        from agent.routing.catalog import extract_capabilities

        metadata = {
            "id": "claude-opus-4-6",
            "features": {
                "attachment": True,
                "reasoning": True,
                "structured_output": True,
                "tool_call": True,
                "vision": False,
            },
            "pricing": {"input": 10, "output": 50, "cache_read": 1},
        }
        caps = extract_capabilities(metadata)
        assert caps.has_reasoning is True
        assert caps.has_tool_calling is True
        assert caps.has_structured_output is True
        assert caps.has_attachment is True
        assert caps.pricing["input"] == 10

    def test_empty_metadata_returns_defaults(self):
        from agent.routing.catalog import extract_capabilities

        caps = extract_capabilities({})
        assert caps.context_length == 0
        assert caps.has_vision is False
        assert caps.has_reasoning is False
        assert caps.pricing == {}


class TestCatalogExtractUrls:
    """Test URL extraction from metadata."""

    def test_extracts_apiBaseUrl(self):
        from agent.routing.catalog import extract_urls

        urls = extract_urls({"apiBaseUrl": "https://api.anthropic.com/v1"})
        assert "https://api.anthropic.com/v1" in urls

    def test_extracts_multiple_url_fields(self):
        from agent.routing.catalog import extract_urls

        urls = extract_urls({
            "apiBaseUrl": "https://api.example.com/v1",
            "baseUrl": "https://base.example.com",
        })
        assert len(urls) == 2

    def test_empty_metadata_returns_empty(self):
        from agent.routing.catalog import extract_urls

        assert extract_urls({}) == []


class TestCatalogMatchLive:
    """Test live catalog matching with mocked network data."""

    @patch("agent.routing.catalog._refresh_cache_if_needed")
    @patch("agent.routing.catalog._CATALOG_CACHE", {
        "timestamp": 9999999999.0,
        "openrouter": [
            {"id": "anthropic/claude-opus-4-6", "name": "Claude Opus 4", "context_length": 200000},
            {"id": "openai/gpt-4o", "name": "GPT-4o", "context_length": 128000},
        ],
        "modelsdev": [
            {"id": "deepseek/deepseek-r1", "source": "modelsdev"},
        ],
        "aimodeldir": [
            {"id": "grok-3", "provider": "xai", "apiBaseUrl": "https://api.x.ai/v1"},
        ],
    })
    def test_exact_match_openrouter(self, mock_refresh):
        from agent.routing.catalog import match_catalog_live

        entry = match_catalog_live("anthropic/claude-opus-4-6")
        assert entry is not None
        assert entry.family == "claude"
        assert entry.canonical_id == "anthropic/claude-opus-4-6"

    @patch("agent.routing.catalog._refresh_cache_if_needed")
    @patch("agent.routing.catalog._CATALOG_CACHE", {
        "timestamp": 9999999999.0,
        "openrouter": [],
        "modelsdev": [
            {"id": "deepseek/deepseek-r1", "source": "modelsdev"},
        ],
        "aimodeldir": [],
    })
    def test_exact_match_modelsdev(self, mock_refresh):
        from agent.routing.catalog import match_catalog_live

        entry = match_catalog_live("deepseek/deepseek-r1")
        assert entry is not None
        assert entry.family == "deepseek"
        assert entry.canonical_id == "deepseek/deepseek-r1"

    @patch("agent.routing.catalog._refresh_cache_if_needed")
    @patch("agent.routing.catalog._CATALOG_CACHE", {
        "timestamp": 9999999999.0,
        "openrouter": [],
        "modelsdev": [],
        "aimodeldir": [
            {
                "id": "grok-3",
                "provider": "xai",
                "apiBaseUrl": "https://api.x.ai/v1",
                "features": {"reasoning": True, "tool_call": True},
            },
        ],
    })
    def test_url_plus_model_match(self, mock_refresh):
        from agent.routing.catalog import match_catalog_live

        # Match when both URL and model name align
        entry = match_catalog_live("grok-3", base_url="https://api.x.ai/v1/chat/completions")
        assert entry is not None
        assert entry.family == "grok"

    @patch("agent.routing.catalog._refresh_cache_if_needed")
    @patch("agent.routing.catalog._CATALOG_CACHE", {
        "timestamp": 9999999999.0,
        "openrouter": [],
        "modelsdev": [],
        "aimodeldir": [],
    })
    def test_no_match_returns_none(self, mock_refresh):
        from agent.routing.catalog import match_catalog_live

        entry = match_catalog_live("totally-unknown-model-xyz")
        assert entry is None


class TestCatalogGetCapabilities:
    """Test capability lookup for models."""

    @patch("agent.routing.catalog._refresh_cache_if_needed")
    @patch("agent.routing.catalog._CATALOG_CACHE", {
        "timestamp": 9999999999.0,
        "openrouter": [
            {
                "id": "anthropic/claude-opus-4-6",
                "context_length": 200000,
                "pricing": {"input": "15", "output": "75"},
            },
        ],
        "modelsdev": [],
        "aimodeldir": [],
    })
    def test_exact_id_match_returns_capabilities(self, mock_refresh):
        from agent.routing.catalog import get_capabilities_for_model

        caps = get_capabilities_for_model("anthropic/claude-opus-4-6")
        assert caps.context_length == 200000
        assert caps.pricing["input"] == "15"

    @patch("agent.routing.catalog._refresh_cache_if_needed")
    @patch("agent.routing.catalog._CATALOG_CACHE", {
        "timestamp": 9999999999.0,
        "openrouter": [],
        "modelsdev": [],
        "aimodeldir": [],
    })
    def test_no_match_returns_empty_capabilities(self, mock_refresh):
        from agent.routing.catalog import get_capabilities_for_model

        caps = get_capabilities_for_model("unknown-model-xyz")
        assert caps.context_length == 0
        assert caps.has_vision is False


class TestCatalogInstall:
    """Test the install() function that patches fingerprint module."""

    @patch("agent.routing.catalog._refresh_cache_if_needed")
    @patch("agent.routing.catalog._CATALOG_CACHE", {
        "timestamp": 9999999999.0,
        "openrouter": [
            {
                "id": "anthropic/claude-opus-4-6",
                "name": "Claude Opus 4",
                "context_length": 200000,
            },
        ],
        "modelsdev": [],
        "aimodeldir": [],
    })
    def test_install_patches_match_catalog(self, mock_refresh):
        import agent.routing.fingerprint as fp
        from agent.routing.catalog import install

        # Save original
        original_match = fp._match_catalog

        try:
            install()

            # After install, _match_catalog should find live sources
            entry = fp._match_catalog("anthropic/claude-opus-4-6")
            assert entry is not None
            assert entry.canonical_id == "anthropic/claude-opus-4-6"
            assert entry.capabilities.context_length == 200000
        finally:
            # Restore original
            fp._match_catalog = original_match


class TestInferFamilyFromMetadata:
    """Test family inference from metadata."""

    def test_anthropic_provider(self):
        from agent.routing.catalog import _infer_family_from_metadata

        assert _infer_family_from_metadata({"provider": "anthropic", "id": "claude-x"}) == "claude"

    def test_openai_provider(self):
        from agent.routing.catalog import _infer_family_from_metadata

        assert _infer_family_from_metadata({"provider": "openai", "id": "gpt-5"}) == "gpt"

    def test_fallback_to_model_id(self):
        from agent.routing.catalog import _infer_family_from_metadata

        assert _infer_family_from_metadata({"id": "deepseek/deepseek-r1"}) == "deepseek"

    def test_unknown_returns_unknown(self):
        from agent.routing.catalog import _infer_family_from_metadata

        assert _infer_family_from_metadata({"id": "totally-new-model"}) == "unknown"


class TestTypesContract:
    """Verify types.py shapes match what fingerprint.py and catalog.py expect."""

    def test_model_fingerprint_to_dict_has_capabilities(self):
        from agent.routing.types import ModelFingerprint, ModelCapabilities

        fp = ModelFingerprint(
            model_id="test-model",
            provider="test-provider",
            display_name="Test Model",
            base_url="http://localhost:8080",
            position="upper",
            is_local=True,
            family="test",
            capabilities=ModelCapabilities(context_length=128000, has_reasoning=True),
        )
        d = fp.to_dict()
        assert d["capabilities"]["context_length"] == 128000
        assert d["capabilities"]["has_reasoning"] is True

    def test_fingerprint_entry_has_capabilities_field(self):
        from agent.routing.types import FingerprintEntry, ModelCapabilities

        entry = FingerprintEntry(
            display_name="Test",
            family="test",
            canonical_id="test/model",
            urls=["https://api.test.com/v1"],
            capabilities=ModelCapabilities(has_vision=True),
        )
        assert entry.capabilities.has_vision is True

    def test_model_capabilities_to_dict(self):
        from agent.routing.types import ModelCapabilities

        caps = ModelCapabilities(
            context_length=200000,
            has_tool_calling=True,
            pricing={"input": 15, "output": 75},
        )
        d = caps.to_dict()
        assert d["context_length"] == 200000
        assert d["has_tool_calling"] is True
        assert d["pricing"]["input"] == 15
