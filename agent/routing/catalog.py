"""Model catalog — live-source resolution and capability enrichment.

This module extends the minimal fingerprint resolver (fingerprint.py) with
rich model metadata from external sources:
  - OpenRouter API (https://openrouter.ai/api/v1/models)
  - models.dev (https://models.dev/models.json)
  - AI Model Directory (github.com/The-Best-Codes/ai-model-directory)

It provides:
  - Exact model ID resolution against canonical registries
  - URL-based matching using provider API endpoints as primary identifiers
  - Capability extraction (context window, vision, tool calling, reasoning, pricing)
  - Family inference with much broader coverage than the builtin catalog

Architecture:
  On import, this module patches `agent.routing.fingerprint._match_catalog`
  and `agent.routing.fingerprint._get_capabilities` to use live sources.
  The fingerprint module's public API (resolve_fingerprint, get_fingerprint_table)
  remains unchanged — consumers don't need to know whether the catalog is loaded.

Cache:
  All external fetches are cached with a 5-minute TTL.  Network failures
  degrade gracefully — the builtin catalog in fingerprint.py handles the request
  and capabilities default to ModelCapabilities() (all zeros/False).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from agent.routing.types import (
    FingerprintEntry,
    ModelCapabilities,
)

logger = logging.getLogger(__name__)

# ─── Module-level cache (5-minute TTL) ────────────────────────────────────────

_CATALOG_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "openrouter": [],       # List of model dicts from OpenRouter
    "modelsdev": [],        # List of model dicts from models.dev
    "aimodeldir": [],       # List of model dicts from AI Model Directory
}
_CATALOG_CACHE_TTL = 300


# ─── Fetch functions ──────────────────────────────────────────────────────────


def _fetch_openrouter_models() -> List[Dict[str, Any]]:
    """Fetch the list of models from OpenRouter API.

    Returns raw model dicts with fields:
      id, name, description, context_length, architecture, pricing,
      top_provider, supported_parameters, knowledge_cutoff, links
    """
    try:
        import urllib.request
        import json as json_module

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "Hermes-Agent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json_module.loads(resp.read().decode("utf-8"))
            return data.get("data", [])
    except Exception as exc:
        logger.warning("catalog: failed to fetch OpenRouter models: %s", exc)
        return []


def _fetch_modelsdev_models() -> List[Dict[str, Any]]:
    """Fetch the list of models from models.dev API.

    models.dev returns a flat dict keyed by canonical model ID
    (e.g. "anthropic/claude-opus-4-6"), values are empty dicts or metadata.
    We normalize to a list of dicts with "id" field.
    """
    try:
        import urllib.request
        import json as json_module

        req = urllib.request.Request(
            "https://models.dev/models.json",
            headers={"User-Agent": "Hermes-Agent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json_module.loads(resp.read().decode("utf-8"))
            result = []
            for model_id, metadata in data.items():
                entry: Dict[str, Any] = {"id": model_id, "source": "modelsdev"}
                if isinstance(metadata, dict):
                    entry.update(metadata)
                result.append(entry)
            return result
    except Exception as exc:
        logger.warning("catalog: failed to fetch models.dev models: %s", exc)
        return []


def _fetch_aimodeldir_models() -> List[Dict[str, Any]]:
    """Fetch the list of models from AI Model Directory.

    This source provides per-provider model data with features, pricing,
    and API base URLs.  We flatten the nested structure into a list with
    each entry carrying its provider name and apiBaseUrl.
    """
    try:
        import urllib.request
        import json as json_module

        url = (
            "https://raw.githubusercontent.com/The-Best-Codes/"
            "ai-model-directory/refs/heads/main/data/all.min.json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Hermes-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json_module.loads(resp.read().decode("utf-8"))

            result = []
            for provider_key, provider_data in data.items():
                if not isinstance(provider_data, dict) or "models" not in provider_data:
                    continue
                provider_url = provider_data.get("apiBaseUrl", "")
                for model_key, model_data in provider_data["models"].items():
                    entry: Dict[str, Any] = {
                        "id": model_data.get("id", model_key),
                        "source": "aimodeldir",
                        "provider": provider_key,
                        "apiBaseUrl": provider_url,
                    }
                    if isinstance(model_data, dict):
                        entry.update(model_data)
                    result.append(entry)
            return result
    except Exception as exc:
        logger.warning("catalog: failed to fetch AI Model Directory: %s", exc)
        return []


# ─── Cache access ─────────────────────────────────────────────────────────────


def _refresh_cache_if_needed() -> None:
    """Refresh all caches if TTL expired."""
    now = time.time()
    if now - _CATALOG_CACHE["timestamp"] > _CATALOG_CACHE_TTL:
        _CATALOG_CACHE["openrouter"] = _fetch_openrouter_models()
        _CATALOG_CACHE["modelsdev"] = _fetch_modelsdev_models()
        _CATALOG_CACHE["aimodeldir"] = _fetch_aimodeldir_models()
        _CATALOG_CACHE["timestamp"] = now


def get_openrouter_models() -> List[Dict[str, Any]]:
    """Get cached OpenRouter models."""
    _refresh_cache_if_needed()
    return _CATALOG_CACHE["openrouter"]


def get_modelsdev_models() -> List[Dict[str, Any]]:
    """Get cached models.dev models."""
    _refresh_cache_if_needed()
    return _CATALOG_CACHE["modelsdev"]


def get_aimodeldir_models() -> List[Dict[str, Any]]:
    """Get cached AI Model Directory models."""
    _refresh_cache_if_needed()
    return _CATALOG_CACHE["aimodeldir"]


# ─── Capability extraction ────────────────────────────────────────────────────


def extract_capabilities(metadata: Dict[str, Any]) -> ModelCapabilities:
    """Extract model capabilities from a raw metadata dict.

    Works across all three sources by checking field names used by each.
    """
    caps = ModelCapabilities()

    # Context length (various field names across sources)
    caps.context_length = (
        metadata.get("context_length")
        or metadata.get("context_window")
        or metadata.get("max_context_length")
        or 0
    )

    # Max completion tokens
    caps.max_completion_tokens = (
        metadata.get("max_completion_tokens")
        or metadata.get("max_output_tokens")
        or 0
    )

    # Features (from AI Model Directory — "features" dict)
    features = metadata.get("features", {})
    if isinstance(features, dict):
        caps.has_vision = features.get("vision", False) or features.get("attachment", False)
        caps.has_tool_calling = features.get("tool_call", False)
        caps.has_structured_output = features.get("structured_output", False)
        caps.has_reasoning = features.get("reasoning", False)
        caps.has_attachment = features.get("attachment", False)

    # Architecture details (from OpenRouter — "architecture" dict)
    architecture = metadata.get("architecture", {})
    if isinstance(architecture, dict):
        caps.architecture = architecture

    # Pricing (from OpenRouter or AI Model Directory)
    pricing = metadata.get("pricing", {})
    if isinstance(pricing, dict):
        caps.pricing = {
            k: v for k, v in {
                "input": pricing.get("input"),
                "output": pricing.get("output"),
                "cache_read": pricing.get("cache_read"),
                "cache_write": pricing.get("cache_write"),
            }.items() if v is not None
        }

    # Knowledge cutoff
    caps.has_knowledge_cutoff = (
        metadata.get("knowledge_cutoff")
        or metadata.get("knowledge_date")
        or None
    )

    return caps


def extract_urls(metadata: Dict[str, Any]) -> List[str]:
    """Extract API endpoint URLs from model metadata."""
    urls = []
    for key in ("apiBaseUrl", "apiBaseURL", "baseUrl", "base_url"):
        val = metadata.get(key)
        if val and isinstance(val, str):
            urls.append(val)
    return urls


# ─── Enhanced matching ────────────────────────────────────────────────────────


def match_catalog_live(model_id: str, base_url: str = "") -> Optional[FingerprintEntry]:
    """Match a model_id against live sources (OpenRouter, models.dev, AI Model Dir).

    This is the enhanced version that replaces _match_catalog in fingerprint.py.
    Falls back to the builtin catalog if no live match is found.

    Matching strategy (in priority order):
    1. Exact ID match across all live sources
    2. URL + model substring match (both base_url and model_id must partially match)
    3. Fallback to builtin catalog via fingerprint._match_catalog
    """
    model_lower = model_id.lower()
    base_url_lower = base_url.lower() if base_url else ""

    # 1. Exact ID match across live sources
    for source in [get_openrouter_models(), get_modelsdev_models(), get_aimodeldir_models()]:
        for item in source:
            item_id = item.get("id", "")
            if item_id.lower() == model_lower:
                urls = extract_urls(item)
                caps = extract_capabilities(item)
                family = _infer_family_from_metadata(item)
                display = item.get("name") or item_id.split("/")[-1]
                return FingerprintEntry(
                    display_name=display,
                    family=family,
                    canonical_id=item_id,
                    urls=urls,
                    capabilities=caps,
                )

    # 2. URL + model substring match (high confidence when both match)
    if base_url_lower:
        for source in [get_aimodeldir_models(), get_openrouter_models()]:
            for item in source:
                item_urls = extract_urls(item)
                url_match = any(
                    u.lower() in base_url_lower or base_url_lower in u.lower()
                    for u in item_urls
                )
                if not url_match:
                    continue
                item_id = item.get("id", "")
                # Check if model_id contains significant parts of the item's ID
                id_parts = item_id.lower().split("/")
                model_part = id_parts[-1] if id_parts else ""
                if model_part and model_part in model_lower:
                    urls = item_urls
                    caps = extract_capabilities(item)
                    family = _infer_family_from_metadata(item)
                    display = item.get("name") or item_id.split("/")[-1]
                    return FingerprintEntry(
                        display_name=display,
                        family=family,
                        canonical_id=item_id,
                        urls=urls,
                        capabilities=caps,
                    )

    # 3. Fallback: no live match found
    return None


def get_capabilities_for_model(model_id: str) -> ModelCapabilities:
    """Get capabilities for a model by looking it up across all live sources.

    Returns empty ModelCapabilities if no match found.
    """
    model_lower = model_id.lower()

    # 1. Exact ID match
    for source in [get_openrouter_models(), get_modelsdev_models(), get_aimodeldir_models()]:
        for item in source:
            item_id = item.get("id", "")
            if item_id.lower() == model_lower:
                return extract_capabilities(item)

    # 2. Substring match (model name appears in item ID)
    for source in [get_openrouter_models(), get_modelsdev_models(), get_aimodeldir_models()]:
        for item in source:
            item_id = item.get("id", "")
            model_part = item_id.lower().split("/")[-1] if "/" in item_id else item_id.lower()
            if model_part and model_part in model_lower:
                return extract_capabilities(item)

    return ModelCapabilities()


def _infer_family_from_metadata(metadata: Dict[str, Any]) -> str:
    """Infer model family from metadata (provider field or model ID)."""
    provider = metadata.get("provider", "")
    model_id = metadata.get("id", "")

    # Use provider field if available (from AI Model Directory)
    if provider:
        provider_lower = provider.lower()
        if provider_lower in ("anthropic",):
            return "claude"
        if provider_lower in ("openai",):
            return "gpt"
        if provider_lower in ("google",):
            return "gemini"
        if provider_lower in ("xai",):
            return "grok"
        if provider_lower in ("alibaba", "alibaba-cn"):
            return "qwen"
        if provider_lower in ("deepseek",):
            return "deepseek"
        if provider_lower in ("mistral",):
            return "mistral"
        if provider_lower in ("meta",):
            return "llama"

    # Fallback: infer from model_id
    model_lower = model_id.lower()
    if "claude" in model_lower:
        return "claude"
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower or "o4" in model_lower:
        return "gpt"
    if "qwen" in model_lower:
        return "qwen"
    if "gemini" in model_lower or "gemma" in model_lower:
        return "gemini"
    if "grok" in model_lower:
        return "grok"
    if "llama" in model_lower:
        return "llama"
    if "deepseek" in model_lower:
        return "deepseek"
    if "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"
    return "unknown"


# ─── Integration with fingerprint module ──────────────────────────────────────


def install() -> None:
    """Patch the fingerprint module to use live catalog resolution.

    Call this at startup (e.g., from agent.routing.__init__) to upgrade
    fingerprint resolution from builtin-only to live-source-backed.
    """
    import agent.routing.fingerprint as fp

    # Save the original builtin matcher
    _original_match = fp._match_catalog

    def _enhanced_match(model_id: str, base_url: str = "") -> Optional[FingerprintEntry]:
        """Enhanced catalog match: try live sources first, then builtin."""
        live_result = match_catalog_live(model_id, base_url)
        if live_result is not None:
            return live_result
        return _original_match(model_id, base_url)

    fp._match_catalog = _enhanced_match  # type: ignore[assignment]
    logger.info("catalog: installed live-source model resolution")
