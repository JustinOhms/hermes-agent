"""RoutingConfig dataclass — parses model.routing section of config.yaml."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class GraphPositionProfile:
    startup_latency_s: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    generation_tok_s: float = 0.0


@dataclass
class GraphPosition:
    provider: str = ""
    model: str = ""
    profile: GraphPositionProfile = field(default_factory=GraphPositionProfile)
    base_url: str = ""
    api_mode: str = ""
    api_key: str = ""
    llm_config_name: str = ""  # name for `llm start <name>`; set for local models
    tier: int = 0             # capability rank in the pipeline (higher = more capable)
    alias: str = ""           # user-facing name, e.g. "Alpha" (addressable via /route)


@dataclass
class InteractionModeConfig:
    idle_threshold_s: int = 600
    swap_back_messages: int = 3
    swap_back_window_s: int = 60


@dataclass
class ComplexityConfig:
    escalation_threshold: float = 0.7
    de_escalation_threshold: float = 0.2


@dataclass
class DeEscalationConfig:
    enabled: bool = False


@dataclass
class RoutingConfig:
    enabled: bool = False
    graph: Dict[str, GraphPosition] = field(default_factory=dict)
    interaction_mode: InteractionModeConfig = field(default_factory=InteractionModeConfig)
    complexity: ComplexityConfig = field(default_factory=ComplexityConfig)
    de_escalation: DeEscalationConfig = field(default_factory=DeEscalationConfig)

    # ── Relative-tier navigation ────────────────────────────────────────────
    # The graph is an ordered pipeline by `tier` (higher = more capable). The
    # engine reasons relatively — "is there a tier above me to escalate to, one
    # below to drop to" — instead of hardcoded role names. See ADR-0041.

    def ordered_positions(self) -> list[str]:
        """Graph position keys sorted by tier ascending (least→most capable)."""
        return sorted(self.graph, key=lambda k: (self.graph[k].tier, k))

    def base_position(self) -> Optional[str]:
        """Lowest-tier position — the default seat."""
        ordered = self.ordered_positions()
        return ordered[0] if ordered else None

    def top_position(self) -> Optional[str]:
        """Highest-tier position."""
        ordered = self.ordered_positions()
        return ordered[-1] if ordered else None

    def position_above(self, name: Optional[str]) -> Optional[str]:
        """Next position up the pipeline from ``name`` (None if at/above top)."""
        ordered = self.ordered_positions()
        if name not in ordered:
            return None
        i = ordered.index(name)
        return ordered[i + 1] if i + 1 < len(ordered) else None

    def position_below(self, name: Optional[str]) -> Optional[str]:
        """Next position down the pipeline from ``name`` (None if at/below base)."""
        ordered = self.ordered_positions()
        if name not in ordered:
            return None
        i = ordered.index(name)
        return ordered[i - 1] if i > 0 else None

    def resolve_label(self, selector: str) -> Optional[str]:
        """Resolve a user selector to a graph position key.

        Accepts the position key, its ``alias`` (case-insensitive), or a tier
        number (as ``str`` or ``int``). Returns None if nothing matches.
        """
        if selector is None:
            return None
        sel = str(selector).strip()
        if not sel:
            return None
        low = sel.lower()
        if low in self.graph:
            return low
        for key, pos in self.graph.items():
            if key.lower() == low or (pos.alias and pos.alias.lower() == low):
                return key
        if sel.isdigit():
            want = int(sel)
            for key, pos in self.graph.items():
                if pos.tier == want:
                    return key
        return None


def _parse_profile(raw: Dict[str, Any]) -> GraphPositionProfile:
    return GraphPositionProfile(
        startup_latency_s=float(raw.get("startup_latency_s", 0.0)),
        ttft_p50_ms=float(raw.get("ttft_p50_ms", 0.0)),
        ttft_p90_ms=float(raw.get("ttft_p90_ms", 0.0)),
        generation_tok_s=float(raw.get("generation_tok_s", 0.0)),
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
            tier=int(pos_raw.get("tier", 0) or 0),
            alias=str(pos_raw.get("alias", "")),
        )
    return positions


def load_routing_config(cfg: Optional[Dict[str, Any]] = None) -> RoutingConfig:
    """Build a RoutingConfig from a loaded config dict (or load from disk).

    Args:
        cfg: Already-loaded config dict. If None, loads from config.yaml via
             hermes_cli.config.load_config().
    """
    if cfg is None:
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        except Exception as exc:
            logger.debug("load_routing_config: could not load config: %s", exc)
            cfg = {}

    try:
        from hermes_cli.config import cfg_get
        routing_raw = cfg_get(cfg, "model", "routing") or {}
    except Exception:
        routing_raw = (cfg.get("model") or {}).get("routing") or {}

    if not isinstance(routing_raw, dict):
        return RoutingConfig()

    enabled = bool(routing_raw.get("enabled", False))
    graph = _parse_graph(routing_raw.get("graph") or {})

    im_raw = routing_raw.get("interaction_mode") or {}
    interaction_mode = InteractionModeConfig(
        idle_threshold_s=int(im_raw.get("idle_threshold_s", 600)),
        swap_back_messages=int(im_raw.get("swap_back_messages", 3)),
        swap_back_window_s=int(im_raw.get("swap_back_window_s", 60)),
    )

    cplx_raw = routing_raw.get("complexity") or {}
    complexity = ComplexityConfig(
        escalation_threshold=float(cplx_raw.get("escalation_threshold", 0.7)),
        de_escalation_threshold=float(cplx_raw.get("de_escalation_threshold", 0.2)),
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
