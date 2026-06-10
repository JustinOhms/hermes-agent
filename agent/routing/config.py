"""RoutingConfig dataclass — parses model.routing section of config.yaml."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_section(cfg: Optional[Dict[str, Any]], *path: str) -> Dict[str, Any]:
    """Safely extract a nested config section.

    Handles both hermes_cli.config.cfg_get() (when available) and manual
    dict traversal as fallback.  Used by load_routing_config() and
    load_oversight_config() to avoid duplicating the load-from-disk dance.
    """
    if cfg is None:
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        except Exception:
            return {}

    try:
        from hermes_cli.config import cfg_get
        return cfg_get(cfg, *path) or {}
    except Exception:
        # Manual traversal
        node: Any = cfg
        for key in path:
            if not isinstance(node, dict):
                return {}
            node = node.get(key)
            if node is None:
                return {}
        return node if isinstance(node, dict) else {}


def _safe_int(value: Any, default: int, min_val: int = 1) -> int:
    """Safely convert value to int with bounds checking."""
    if value is None:
        return default
    try:
        result = int(value)
        return max(min_val, result) if result < min_val else result
    except (ValueError, TypeError):
        logger.warning("config: invalid int value %r, using default %d", value, default)
        return default


def _safe_float(value: Any, default: float, min_val: float = 0.0) -> float:
    """Safely convert value to float with bounds checking."""
    if value is None:
        return default
    try:
        result = float(value)
        return max(min_val, result) if result < min_val else result
    except (ValueError, TypeError):
        logger.warning("config: invalid float value %r, using default %.2f", value, default)
        return default


def is_local_position(base_url: str = "", llm_config_name: str = "") -> bool:
    """Determine if a routing position targets the local LLM server.

    Shared by model_resolver, fingerprint, and __init__.py to avoid
    inconsistent is_local heuristics scattered across modules.
    """
    if llm_config_name:
        return True
    if base_url and ("127.0.0.1" in base_url or "localhost" in base_url):
        return True
    return False



@dataclass
class GraphPositionProfile:
    """Performance profile for a graph position (latency, throughput)."""

    startup_latency_s: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    generation_tok_s: float = 0.0


@dataclass
class GraphPosition:
    """A named model slot in the routing graph (provider + model + profile)."""

    provider: str = ""
    model: str = ""
    profile: GraphPositionProfile = field(default_factory=GraphPositionProfile)
    base_url: str = ""
    api_mode: str = ""
    api_key: str = ""
    llm_config_name: str = ""  # name for `llm start <name>`; set for local models
    display_name: str = ""     # human-friendly name (e.g. "Claude Opus 4"); used by fingerprint
    tier: int = 0              # explicit ordering for upgrade/downgrade (higher = more capable)


@dataclass
class InteractionModeConfig:
    """Thresholds for interactive vs autonomous mode detection."""

    idle_threshold_s: int = 600
    swap_back_messages: int = 3
    swap_back_window_s: int = 60


@dataclass
class ComplexityConfig:
    """Complexity scoring thresholds for escalation/de-escalation decisions."""

    escalation_threshold: float = 0.7
    de_escalation_threshold: float = 0.2


@dataclass
class DeEscalationConfig:
    """Controls whether the router can de-escalate from upper to lower tiers."""

    enabled: bool = False


@dataclass
class RoutingConfig:
    """Top-level routing configuration parsed from model.routing in config.yaml."""

    enabled: bool = False
    graph: Dict[str, GraphPosition] = field(default_factory=dict)
    interaction_mode: InteractionModeConfig = field(default_factory=InteractionModeConfig)
    complexity: ComplexityConfig = field(default_factory=ComplexityConfig)
    de_escalation: DeEscalationConfig = field(default_factory=DeEscalationConfig)


def _parse_profile(raw: Dict[str, Any]) -> GraphPositionProfile:
    return GraphPositionProfile(
        startup_latency_s=_safe_float(raw.get("startup_latency_s"), 0.0, min_val=0.0),
        ttft_p50_ms=_safe_float(raw.get("ttft_p50_ms"), 0.0, min_val=0.0),
        ttft_p90_ms=_safe_float(raw.get("ttft_p90_ms"), 0.0, min_val=0.0),
        generation_tok_s=_safe_float(raw.get("generation_tok_s"), 0.0, min_val=0.0),
    )


def _parse_graph(raw: Dict[str, Any]) -> Dict[str, GraphPosition]:
    positions: Dict[str, GraphPosition] = {}
    for name, pos_raw in (raw or {}).items():
        if not isinstance(pos_raw, dict):
            continue
        profile = _parse_profile(pos_raw.get("profile") or {})
        positions[name] = GraphPosition(
            provider=str(pos_raw.get("provider", "")),
            model=str(pos_raw.get("model", "")),
            profile=profile,
            base_url=str(pos_raw.get("base_url", "")),
            api_mode=str(pos_raw.get("api_mode", "")),
            api_key=str(pos_raw.get("api_key", "")),
            llm_config_name=str(pos_raw.get("llm_config_name", "")),
            display_name=str(pos_raw.get("display_name", "")),
            tier=_safe_int(pos_raw.get("tier"), default=0, min_val=0),
        )
    return positions


# ── Tier ladder for upgrade/downgrade ──────────────────────────────────────

# Fallback ordering when positions have no explicit tier (backwards compat)
_WELL_KNOWN_TIER_ORDER: List[str] = [
    "fast_fallback", "interactive_lower", "autonomous_lower", "upper",
]


def build_tier_ladder(graph: Dict[str, "GraphPosition"]) -> List[str]:
    """Return position names ordered lowest-tier → highest-tier.

    Resolution:
    1. If all positions have tier=0 (unset), fall back to _WELL_KNOWN_TIER_ORDER
       filtered to graph keys, with unknown positions appended alphabetically.
    2. If any position has tier > 0, sort by tier ascending.  Positions with
       tier=0 (unset) are assigned a synthetic tier based on _WELL_KNOWN_TIER_ORDER
       to maintain backwards compatibility.
    """
    positions = list(graph.keys())
    if not positions:
        return []

    # Check if any explicit tier is set
    has_explicit = any(pos.tier > 0 for pos in graph.values())

    if not has_explicit:
        # Pure fallback: use well-known ordering + alphabetical for unknowns
        ladder = [p for p in _WELL_KNOWN_TIER_ORDER if p in graph]
        unknowns = sorted(p for p in positions if p not in ladder)
        # Insert unknowns before "upper" if it exists, else append
        if "upper" in ladder and unknowns:
            idx = ladder.index("upper")
            for u in unknowns:
                ladder.insert(idx, u)
                idx += 1
        else:
            ladder.extend(unknowns)
        return ladder

    # Explicit tiers present — build synthetic tiers for unset positions
    def _tier_key(name: str) -> int:
        explicit = graph[name].tier
        if explicit > 0:
            return explicit
        # Assign synthetic tier from well-known order (10, 20, 30, 40)
        if name in _WELL_KNOWN_TIER_ORDER:
            return (_WELL_KNOWN_TIER_ORDER.index(name) + 1) * 10
        # Unknown position with no tier — place between lower and upper
        return 25
    
    return sorted(positions, key=_tier_key)


def load_routing_config(cfg: Optional[Dict[str, Any]] = None) -> RoutingConfig:
    """Build a RoutingConfig from a loaded config dict (or load from disk).

    Args:
        cfg: Already-loaded config dict. If None, loads from config.yaml via
             hermes_cli.config.load_config().
    """
    routing_raw = _load_section(cfg, "model", "routing")
    if not isinstance(routing_raw, dict):
        return RoutingConfig()

    enabled = bool(routing_raw.get("enabled", False))
    graph = _parse_graph(routing_raw.get("graph") or {})

    im_raw = routing_raw.get("interaction_mode") or {}
    interaction_mode = InteractionModeConfig(
        idle_threshold_s=_safe_int(im_raw.get("idle_threshold_s"), 600, min_val=1),
        swap_back_messages=_safe_int(im_raw.get("swap_back_messages"), 3, min_val=1),
        swap_back_window_s=_safe_int(im_raw.get("swap_back_window_s"), 60, min_val=1),
    )

    cplx_raw = routing_raw.get("complexity") or {}
    complexity = ComplexityConfig(
        escalation_threshold=_safe_float(cplx_raw.get("escalation_threshold"), 0.7, min_val=0.0),
        de_escalation_threshold=_safe_float(cplx_raw.get("de_escalation_threshold"), 0.2, min_val=0.0),
    )

    de_raw = routing_raw.get("de_escalation") or {}
    de_escalation = DeEscalationConfig(
        enabled=bool(de_raw.get("enabled", False)),
    )

    return RoutingConfig(
        enabled=enabled,
        graph=graph,
        interaction_mode=interaction_mode,
        complexity=complexity,
        de_escalation=de_escalation,
    )
