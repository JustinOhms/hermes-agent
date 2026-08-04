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
