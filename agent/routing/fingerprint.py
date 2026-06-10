"""Model fingerprint registry — resolves the active model identity at API-call time.

The fingerprint is the authoritative source of "what model is actually processing
this request" — derived from the concrete API endpoint + model parameter that will
be sent over the wire, NOT from cached session state.

Consumers:
  - Ephemeral system prompt injection (tells the model who it is)
  - TUI status bar (shows the user which model is active)
  - /routing identity command (displays the full fingerprint dictionary)
  - Logging / observability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.routing.config import is_local_position

logger = logging.getLogger(__name__)


@dataclass
class ModelFingerprint:
    """Resolved identity of the model processing the current request."""
    model_id: str           # Exact model string sent to the API (e.g. "us.anthropic.claude-opus-4-6-v1")
    provider: str           # Provider name (e.g. "bedrock", "custom:llm-local")
    display_name: str       # Human-friendly name (e.g. "Claude Opus 4", "Qwen3 Coder Next")
    base_url: str           # API endpoint being called (masked for display)
    position: str           # Routing graph position if routing is active (e.g. "upper", "interactive_lower")
    is_local: bool          # Whether this is a local model on :58080
    family: str             # Model family for capability hints (e.g. "claude", "qwen", "gpt")

    def to_prompt_line(self) -> str:
        """Format for injection into the ephemeral system prompt."""
        parts = [f"Model: {self.model_id}"]
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        if self.display_name and self.display_name != self.model_id:
            parts.append(f"Display name: {self.display_name}")
        if self.position:
            parts.append(f"Routing position: {self.position}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON-RPC / TUI events."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "display_name": self.display_name,
            "base_url": _mask_url(self.base_url),
            "position": self.position,
            "is_local": self.is_local,
            "family": self.family,
        }


# ── Registry: maps (provider, model) → display metadata ─────────────────────

@dataclass
class FingerprintEntry:
    """Static metadata for a known model, used to populate display_name and family."""
    display_name: str
    family: str


# Built-in catalog of well-known models.  Extended at runtime by the
# routing graph's display_name fields and custom_providers config.
_BUILTIN_CATALOG: Dict[str, FingerprintEntry] = {
    # Anthropic / Bedrock
    "claude-opus-4-6": FingerprintEntry("Claude Opus 4", "claude"),
    "claude-sonnet-4-6": FingerprintEntry("Claude Sonnet 4", "claude"),
    "claude-sonnet-4-5": FingerprintEntry("Claude Sonnet 4.5", "claude"),
    "claude-haiku-3-5": FingerprintEntry("Claude Haiku 3.5", "claude"),
    # OpenAI
    "gpt-4o": FingerprintEntry("GPT-4o", "gpt"),
    "gpt-4.1": FingerprintEntry("GPT-4.1", "gpt"),
    "o3": FingerprintEntry("o3", "gpt"),
    "o4-mini": FingerprintEntry("o4-mini", "gpt"),
    "codex-mini": FingerprintEntry("Codex Mini", "gpt"),
    # Qwen
    "qwen3-coder-next": FingerprintEntry("Qwen3 Coder Next", "qwen"),
    "qwen3-coder-30b": FingerprintEntry("Qwen3 Coder 30B", "qwen"),
    "qwen3.6-27b": FingerprintEntry("Qwen3.6 27B", "qwen"),
    "qwen3.6-35b-a3b": FingerprintEntry("Qwen3.6 35B-A3B MoE", "qwen"),
    # Google
    "gemini-2.5-pro": FingerprintEntry("Gemini 2.5 Pro", "gemini"),
    "gemini-2.5-flash": FingerprintEntry("Gemini 2.5 Flash", "gemini"),
    # xAI
    "grok-3": FingerprintEntry("Grok 3", "grok"),
    "grok-3-mini": FingerprintEntry("Grok 3 Mini", "grok"),
}


def _match_catalog(model_id: str) -> Optional[FingerprintEntry]:
    """Fuzzy-match a model_id against the builtin catalog.

    Tries exact match first, then substring containment (longest match wins).
    """
    model_lower = model_id.lower()

    # Exact match
    if model_lower in _BUILTIN_CATALOG:
        return _BUILTIN_CATALOG[model_lower]

    # Substring match — longest catalog key that appears in the model_id wins
    best: Optional[FingerprintEntry] = None
    best_len = 0
    for key, entry in _BUILTIN_CATALOG.items():
        if key in model_lower and len(key) > best_len:
            best = entry
            best_len = len(key)
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


# ── Main entry point ─────────────────────────────────────────────────────────


def resolve_fingerprint(agent: object) -> ModelFingerprint:
    """Resolve the model fingerprint from the agent's CURRENT runtime state.

    Called just before each API call.  Reads agent.model, agent.provider,
    agent.base_url — the actual values that will be used for the request.
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
        entry = _match_catalog(model_id)
        if entry:
            display_name = entry.display_name
            family = entry.family
        else:
            # Use the model_id itself as display name (strip provider prefix if present)
            display_name = model_id.split("/")[-1] if "/" in model_id else model_id
            family = _infer_family(model_id, provider)
    else:
        family = _infer_family(model_id, provider)

    return ModelFingerprint(
        model_id=model_id,
        provider=provider,
        display_name=display_name,
        base_url=base_url,
        position=position,
        is_local=is_local,
        family=family,
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
                    entry = _match_catalog(pos_model)
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
                    "active": is_active,
                    "source": "graph",
                })
    except Exception as exc:
        logger.debug("get_fingerprint_table: graph lookup failed: %s", exc)

    return rows
