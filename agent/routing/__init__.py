"""agent.routing — Turn Router entry point.

Phase 1: decision-only (no model swaps).
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.routing.config import RoutingConfig, load_routing_config
from agent.routing.interaction_mode import InteractionMode, InteractionModeDetector
from agent.routing.turn_router import RoutingContext, RoutingDecision, TurnRouter

logger = logging.getLogger(__name__)


def get_routing_decision(agent: object, user_message: str) -> Optional[RoutingDecision]:
    """Called at the start of each turn to get a routing recommendation.

    Returns None if routing is disabled or on any error.
    Phase 1: logs the decision but does NOT perform model swaps.
    The agent continues on whatever model is currently active.
    """
    try:
        config = load_routing_config()
        if not config.enabled:
            return None

        # Build or retrieve the per-agent mode detector
        detector = _get_or_create_detector(agent, config)
        detector.record_user_message()

        # Build a minimal context for mode detection (avoids needing mode upfront)
        minimal_ctx = _build_context(user_message, agent, InteractionMode.INTERACTIVE)
        mode = detector.current_mode(minimal_ctx)
        context = _build_context(user_message, agent, mode)

        router = TurnRouter(config, detector)
        return router.route(context)

    except Exception as exc:
        logger.debug("get_routing_decision failed (non-fatal): %s", exc)
        return None


def _get_or_create_detector(agent: object, config: RoutingConfig) -> InteractionModeDetector:
    """Retrieve or create the InteractionModeDetector cached on the agent."""
    attr = "_routing_mode_detector"
    detector = getattr(agent, attr, None)
    if detector is None:
        detector = InteractionModeDetector(config)
        try:
            setattr(agent, attr, detector)
        except Exception:
            pass
    return detector


def _build_context(user_message: str, agent: object, mode: InteractionMode) -> RoutingContext:
    """Build a RoutingContext from agent state."""
    return RoutingContext(
        user_message=user_message,
        message_length=len(user_message),
        conversation_turn_count=getattr(agent, "_turn_count", 0),
        is_cron=getattr(agent, "is_cron", False),
        is_subagent=getattr(agent, "is_subagent", False),
        platform=getattr(agent, "platform", "cli") or "cli",
        interaction_mode=mode,
        recent_tool_calls=list(getattr(agent, "_recent_tool_calls", []) or [])[-5:],
        last_response_had_errors=getattr(agent, "_last_response_had_errors", False),
        explicit_mode_override=getattr(agent, "_explicit_mode_override", None),
    )
