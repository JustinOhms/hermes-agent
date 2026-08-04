"""Routing state aggregation for /routing command and TUI indicator.

Phase 3b: read-only aggregation of all routing state from agent-cached objects.
Does not modify routing logic (Phase 1-2 frozen).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from agent.routing.turn_router import RoutingDecision
    from agent.routing.drift_detector import DriftAlert


@dataclass
class RoutingState:
    """Aggregated snapshot of routing state for a single agent session."""

    enabled: bool
    current_position: Optional[str]
    interaction_mode: str          # "interactive" | "autonomous"
    swap_state: str                # SwapState enum name: "IDLE" | "AWAITING_ENGAGEMENT" | ...
    last_decision: Optional["RoutingDecision"]
    drift_alerts: List["DriftAlert"] = field(default_factory=list)
    decision_history: List["RoutingDecision"] = field(default_factory=list)


def get_routing_state(agent: object) -> RoutingState:
    """Aggregate all routing state from agent-cached objects.

    Always returns a RoutingState — never raises.  When routing is disabled
    or the agent hasn't processed any turns yet, returns a minimal state with
    enabled=False.
    """
    try:
        from agent.routing.config import load_routing_config
        config = load_routing_config()
    except Exception:
        config = None

    enabled = bool(config and config.enabled)

    if not enabled:
        return RoutingState(
            enabled=False,
            current_position=None,
            interaction_mode="interactive",
            swap_state="IDLE",
            last_decision=None,
        )

    current_position: Optional[str] = None
    swap_state = "IDLE"
    interaction_mode = "interactive"
    last_decision = None
    drift_alerts: list = []
    decision_history: list = []

    # ── Swap manager ──────────────────────────────────────────────────────────
    try:
        swap_mgr = getattr(agent, "_routing_swap_manager", None)
        if swap_mgr is not None:
            current_position = swap_mgr.current_position
            raw_state = swap_mgr.state
            swap_state = raw_state.name if hasattr(raw_state, "name") else str(raw_state)
    except Exception:
        pass

    # ── Interaction mode detector ─────────────────────────────────────────────
    try:
        detector = getattr(agent, "_routing_mode_detector", None)
        if detector is not None:
            # InteractionModeDetector exposes _last_mode (set in current_mode())
            mode = getattr(detector, "_last_mode", None)
            if mode is not None:
                interaction_mode = mode.value if hasattr(mode, "value") else str(mode)
    except Exception:
        pass

    # ── Decision history ring buffer ──────────────────────────────────────────
    try:
        history_deque = getattr(agent, "_routing_decision_history", None)
        if history_deque is not None:
            decision_history = list(history_deque)
            if decision_history:
                last_decision = decision_history[-1]
    except Exception:
        pass

    # ── Drift detector ────────────────────────────────────────────────────────
    try:
        drift_detector = getattr(agent, "_routing_drift_detector", None)
        if drift_detector is not None:
            alerts = getattr(drift_detector, "_recent_alerts", [])
            drift_alerts = list(alerts)
    except Exception:
        pass

    return RoutingState(
        enabled=enabled,
        current_position=current_position,
        interaction_mode=interaction_mode,
        swap_state=swap_state,
        last_decision=last_decision,
        drift_alerts=drift_alerts,
        decision_history=decision_history,
    )
