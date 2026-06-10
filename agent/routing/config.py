"""RoutingConfig dataclass — parses model.routing section of config.yaml."""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Module-level cache for load_routing_config() with 30s TTL ────────────────
_CONFIG_CACHE: dict = {"timestamp": 0.0, "config": None}
_CONFIG_TTL: int = 30


def _load_cached_config() -> RoutingConfig:
    """Load routing config with 30-second cache TTL.
    
    Returns the cached config if valid, otherwise reloads from disk.
    """
    now = time.time()
    if now - _CONFIG_CACHE["timestamp"] > _CONFIG_TTL:
        _CONFIG_CACHE["config"] = load_routing_config()
        _CONFIG_CACHE["timestamp"] = now
    return _CONFIG_CACHE["config"]


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


def build_tier_ladder(graph: Dict[str, "GraphPosition"]) -> Optional[List[str]]:
    """Return position names ordered lowest-tier → highest-tier.

    Returns None if any position has tier=0 (unset), meaning upgrade/downgrade
    is disabled until tiers are assigned (manually or via auto_assign_tiers).
    """
    positions = list(graph.keys())
    if not positions:
        return None

    # All positions must have explicit tier > 0
    if any(pos.tier == 0 for pos in graph.values()):
        return None

    return sorted(positions, key=lambda name: graph[name].tier)


def auto_assign_tiers(graph: Dict[str, "GraphPosition"]) -> Dict[str, int]:
    """Score each position and assign tier values (1 = lowest, N = highest).

    Scoring heuristic (higher score → higher tier):
      - Cloud positions score higher than local (cloud models are generally
        more capable but slower/costlier).
      - Within the same locality class, lower generation speed (tok/s) implies
        larger/more capable model → higher tier.
      - TTFT is used as a tiebreaker: higher TTFT → more compute → higher tier.

    The intent is: fast-cheap-local models get low tiers, slow-expensive-cloud
    models get high tiers — matching the typical capability ordering.

    Returns a dict mapping position name → assigned tier (1-indexed).
    """
    if not graph:
        return {}

    def _score(pos: "GraphPosition") -> float:
        """Composite score: higher = more capable (higher tier)."""
        is_local = is_local_position(pos.base_url, pos.llm_config_name)
        # Locality: cloud positions get a large base boost
        locality_score = 0.0 if is_local else 1000.0

        # Generation speed: slower generally means larger/more capable.
        # Invert and scale: a 30 tok/s model scores higher than 139 tok/s.
        gen_speed = pos.profile.generation_tok_s
        if gen_speed > 0:
            speed_score = 1000.0 / gen_speed  # 139→7.2, 33→30.3, 80→12.5
        else:
            speed_score = 50.0  # unknown → mid-range

        # TTFT: higher TTFT → more compute → tiebreaker for capability.
        ttft_score = pos.profile.ttft_p50_ms / 100.0  # 200ms→2, 800ms→8, 2000ms→20

        return locality_score + speed_score + ttft_score

    scored = [(name, _score(pos)) for name, pos in graph.items()]
    scored.sort(key=lambda x: x[1])

    return {name: tier for tier, (name, _) in enumerate(scored, start=1)}


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
