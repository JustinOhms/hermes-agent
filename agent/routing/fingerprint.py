"""Model fingerprint resolver — minimal builtin-only implementation.

This module resolves the active model identity at API-call time using a
static builtin catalog.  It is self-contained (no network calls) and ships
with the routing PR.

The catalog module (separate PR) replaces the internal matching logic with
live sources (OpenRouter, models.dev, AI Model Directory) while preserving
the same public API:
  - resolve_fingerprint(agent) → ModelFingerprint
  - get_fingerprint_table(agent) → List[Dict]

Consumers:
  - Ephemeral system prompt injection (tells the model who it is)
  - TUI status bar (shows the user which model is active)
  - /routing identity command (displays the full fingerprint dictionary)
  - Logging / observability
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.routing.config import is_local_position
from agent.routing.types import (
    FingerprintEntry,
    ModelCapabilities,
    ModelFingerprint,
    RoutingGraphContext,
)

logger = logging.getLogger(__name__)


# ─── Builtin catalog of well-known models ─────────────────────────────────────
# Keyed by canonical_id (provider/model).  Extended at runtime by the catalog
# module when available.

_BUILTIN_CATALOG: List[FingerprintEntry] = [
    # Anthropic / Bedrock
    FingerprintEntry(
        display_name="Claude Opus 4",
        family="claude",
        canonical_id="anthropic/claude-opus-4-6",
        urls=["https://api.anthropic.com/v1", "https://bedrock-runtime.us-east-1.amazonaws.com"],
    ),
    FingerprintEntry(
        display_name="Claude Sonnet 4",
        family="claude",
        canonical_id="anthropic/claude-sonnet-4-6",
        urls=["https://api.anthropic.com/v1", "https://bedrock-runtime.us-east-1.amazonaws.com"],
    ),
    FingerprintEntry(
        display_name="Claude Sonnet 4.5",
        family="claude",
        canonical_id="anthropic/claude-sonnet-4-5",
        urls=["https://api.anthropic.com/v1", "https://bedrock-runtime.us-east-1.amazonaws.com"],
    ),
    FingerprintEntry(
        display_name="Claude Haiku 3.5",
        family="claude",
        canonical_id="anthropic/claude-3-5-haiku",
        urls=["https://api.anthropic.com/v1", "https://bedrock-runtime.us-east-1.amazonaws.com"],
    ),
    # OpenAI
    FingerprintEntry(
        display_name="GPT-4o",
        family="gpt",
        canonical_id="openai/gpt-4o",
        urls=["https://api.openai.com/v1"],
    ),
    FingerprintEntry(
        display_name="GPT-4.1",
        family="gpt",
        canonical_id="openai/gpt-4.1",
        urls=["https://api.openai.com/v1"],
    ),
    FingerprintEntry(
        display_name="o3",
        family="gpt",
        canonical_id="openai/o3",
        urls=["https://api.openai.com/v1"],
    ),
    FingerprintEntry(
        display_name="o4-mini",
        family="gpt",
        canonical_id="openai/o4-mini",
        urls=["https://api.openai.com/v1"],
    ),
    FingerprintEntry(
        display_name="GPT-5",
        family="gpt",
        canonical_id="openai/gpt-5",
        urls=["https://api.openai.com/v1"],
    ),
    # Qwen (Alibaba)
    FingerprintEntry(
        display_name="Qwen3 Coder Next",
        family="qwen",
        canonical_id="alibaba/qwen3-coder-next",
        urls=["https://api.qwen.cn/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1"],
    ),
    FingerprintEntry(
        display_name="Qwen3 Coder 30B",
        family="qwen",
        canonical_id="alibaba/qwen3-coder-30b-a3b-instruct",
        urls=["https://api.qwen.cn/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1"],
    ),
    FingerprintEntry(
        display_name="Qwen3.6 27B",
        family="qwen",
        canonical_id="alibaba/qwen3.6-27b",
        urls=["https://api.qwen.cn/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1"],
    ),
    FingerprintEntry(
        display_name="Qwen3.6 35B-A3B MoE",
        family="qwen",
        canonical_id="alibaba/qwen3.6-35b-a3b",
        urls=["https://api.qwen.cn/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1"],
    ),
    # Google
    FingerprintEntry(
        display_name="Gemini 2.5 Pro",
        family="gemini",
        canonical_id="google/gemini-2.5-pro",
        urls=["https://generativelanguage.googleapis.com/v1"],
    ),
    FingerprintEntry(
        display_name="Gemini 2.5 Flash",
        family="gemini",
        canonical_id="google/gemini-2.5-flash",
        urls=["https://generativelanguage.googleapis.com/v1"],
    ),
    # xAI
    FingerprintEntry(
        display_name="Grok 3",
        family="grok",
        canonical_id="xai/grok-3",
        urls=["https://api.x.ai/v1"],
    ),
    FingerprintEntry(
        display_name="Grok 3 Mini",
        family="grok",
        canonical_id="xai/grok-3-mini",
        urls=["https://api.x.ai/v1"],
    ),
    # DeepSeek
    FingerprintEntry(
        display_name="DeepSeek R1",
        family="deepseek",
        canonical_id="deepseek/deepseek-r1",
        urls=["https://api.deepseek.com/v1"],
    ),
    FingerprintEntry(
        display_name="DeepSeek Chat",
        family="deepseek",
        canonical_id="deepseek/deepseek-chat",
        urls=["https://api.deepseek.com/v1"],
    ),
]


# ─── Matching logic ───────────────────────────────────────────────────────────


def _match_catalog(model_id: str, base_url: str = "") -> Optional[FingerprintEntry]:
    """Match a model_id against the builtin catalog.

    Matching strategy (in priority order):
    1. Exact canonical_id match (e.g. "anthropic/claude-opus-4-6")
    2. URL + model_id substring match (both must match for confidence)
    3. Longest substring match on canonical_id parts
    4. Provider/model suffix match for slash-separated IDs

    The catalog module overrides this with live-source lookups.
    """
    model_lower = model_id.lower()
    base_url_lower = base_url.lower()

    # 1. Exact canonical_id match
    for entry in _BUILTIN_CATALOG:
        if entry.canonical_id and entry.canonical_id.lower() == model_lower:
            return entry

    # 2. URL + model substring match (high confidence)
    if base_url_lower:
        for entry in _BUILTIN_CATALOG:
            if not entry.urls:
                continue
            url_match = any(u.lower() in base_url_lower or base_url_lower in u.lower() for u in entry.urls)
            if url_match and entry.canonical_id:
                # Check if model_id contains the model part of canonical_id
                _, model_part = entry.canonical_id.lower().rsplit("/", 1)
                if model_part in model_lower:
                    return entry

    # 3. Longest substring match on canonical_id parts
    best: Optional[FingerprintEntry] = None
    best_len = 0

    for entry in _BUILTIN_CATALOG:
        if entry.canonical_id:
            for part in entry.canonical_id.lower().split("/"):
                if part in model_lower and len(part) > best_len:
                    best = entry
                    best_len = len(part)

    # 4. Provider/model suffix match
    if "/" in model_lower:
        provider, model = model_lower.split("/", 1)
        for entry in _BUILTIN_CATALOG:
            if entry.canonical_id:
                ep, em = entry.canonical_id.lower().split("/", 1)
                if em == model and provider in (ep, ep.replace("-", "")):
                    return entry

    return best


def _infer_family(model_id: str, provider: str) -> str:
    """Infer model family from model_id or provider when catalog miss."""
    model_lower = model_id.lower()
    if "claude" in model_lower or provider in ("anthropic", "bedrock"):
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


def _mask_url(url: str) -> str:
    """Mask sensitive parts of a URL for display (keep host, hide keys in path)."""
    if not url:
        return ""
    # For local endpoints, show as-is
    if "127.0.0.1" in url or "localhost" in url:
        return url
    # For cloud endpoints, show just the host
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.hostname}" if parsed.hostname else url
    except Exception:
        return url[:50] + "..." if len(url) > 50 else url


def _resolve_position(agent: object) -> str:
    """Get the current routing position from the swap manager if available."""
    swap_mgr = getattr(agent, "_routing_swap_manager", None)
    if swap_mgr is not None:
        pos = swap_mgr.current_position
        if pos:
            return pos
    return ""


def _resolve_display_name_from_graph(model_id: str, provider: str, position: str) -> str:
    """Check if the routing graph has a display_name for this position."""
    if not position:
        return ""
    try:
        from agent.routing.config import load_routing_config
        config = load_routing_config()
        pos_cfg = config.graph.get(position)
        if pos_cfg and hasattr(pos_cfg, "display_name"):
            return getattr(pos_cfg, "display_name", "") or ""
    except Exception:
        pass
    return ""


# ─── Public API ───────────────────────────────────────────────────────────────


def _build_routing_graph_context(current_position: str) -> Optional[RoutingGraphContext]:
    """Build a RoutingGraphContext from the active routing config.

    Returns None if routing is disabled or the graph is empty.
    Provides just enough info for the model to understand the routing topology
    without bloating the system prompt.
    """
    try:
        from agent.routing.config import load_routing_config
        config = load_routing_config()
        if not config or not config.enabled or not config.graph:
            return None
    except Exception:
        return None

    positions: Dict[str, str] = {}
    configured_upper = ""

    for pos_name, pos_cfg in config.graph.items():
        # Build a compact description for each position
        pos_display = getattr(pos_cfg, "display_name", "") or ""
        if not pos_display:
            # Try catalog lookup for a friendly name
            entry = _match_catalog(pos_cfg.model, pos_cfg.base_url)
            pos_display = entry.display_name if entry else pos_cfg.model

        pos_desc = f"{pos_display} ({pos_cfg.provider})"
        if is_local_position(pos_cfg.base_url, pos_cfg.llm_config_name):
            pos_desc += " [local]"

        positions[pos_name] = pos_desc

        if pos_name == "upper":
            configured_upper = pos_desc

    if not positions:
        return None

    return RoutingGraphContext(
        active_position=current_position,
        positions=positions,
        configured_upper=configured_upper,
    )


def resolve_fingerprint(agent: object) -> ModelFingerprint:
    """Resolve the model fingerprint from the agent's CURRENT runtime state.

    Called just before each API call.  Reads agent.model, agent.provider,
    agent.base_url — the actual values that will be used for the request.

    The catalog module monkey-patches this function to add live-source
    resolution and capability enrichment.
    """
    model_id = getattr(agent, "model", "") or ""
    provider = getattr(agent, "provider", "") or ""
    base_url = getattr(agent, "base_url", "") or ""
    is_local = is_local_position(base_url)

    position = _resolve_position(agent)

    # Try graph display_name first, then catalog, then fallback
    display_name = _resolve_display_name_from_graph(model_id, provider, position)
    family = ""

    if not display_name:
        entry = _match_catalog(model_id, base_url)
        if entry:
            display_name = entry.display_name
            family = entry.family
        else:
            # Use the model_id itself as display name (strip provider prefix if present)
            display_name = model_id.split("/")[-1] if "/" in model_id else model_id
            family = _infer_family(model_id, provider)
    else:
        family = _infer_family(model_id, provider)

    # ── Build routing graph context for model self-awareness ──
    routing_graph = _build_routing_graph_context(position)

    return ModelFingerprint(
        model_id=model_id,
        provider=provider,
        display_name=display_name,
        base_url=_mask_url(base_url),
        position=position,
        is_local=is_local,
        family=family,
        capabilities=ModelCapabilities(),  # Enriched by catalog module when available
        routing_graph=routing_graph,
    )


def get_fingerprint_table(agent: object) -> List[Dict[str, Any]]:
    """Return the full fingerprint dictionary for display (/routing identity).

    Shows all positions in the routing graph with their fingerprint data,
    plus the currently active model highlighted.
    """
    rows: List[Dict[str, Any]] = []

    # Current model (always shown, even if routing is disabled)
    current = resolve_fingerprint(agent)
    rows.append({
        **current.to_dict(),
        "active": True,
        "source": "current",
    })

    # Graph positions (if routing is configured)
    try:
        from agent.routing.config import load_routing_config
        config = load_routing_config()
        if config and config.graph:
            for pos_name, pos_cfg in config.graph.items():
                pos_model = pos_cfg.model
                pos_provider = pos_cfg.provider
                pos_base_url = pos_cfg.base_url or ""
                pos_is_local = is_local_position(pos_base_url, pos_cfg.llm_config_name)

                # Resolve display name
                graph_display = getattr(pos_cfg, "display_name", "") if hasattr(pos_cfg, "display_name") else ""
                if not graph_display:
                    entry = _match_catalog(pos_model, pos_base_url)
                    graph_display = entry.display_name if entry else pos_model
                pos_family = _infer_family(pos_model, pos_provider)

                is_active = (pos_name == current.position)
                rows.append({
                    "model_id": pos_model,
                    "provider": pos_provider,
                    "display_name": graph_display,
                    "base_url": _mask_url(pos_base_url),
                    "position": pos_name,
                    "is_local": pos_is_local,
                    "family": pos_family,
                    "capabilities": ModelCapabilities().to_dict(),
                    "active": is_active,
                    "source": "graph",
                })
    except Exception as exc:
        logger.debug("get_fingerprint_table: graph lookup failed: %s", exc)

    return rows
