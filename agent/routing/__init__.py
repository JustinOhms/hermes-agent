"""agent.routing — Turn Router entry point.

Phase 1: decision-only (log but no swap).
Phase 2: swap execution via SwapManager + ModelResolver.
Phase 3b: decision history ring buffer + state aggregation (get_routing_state).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

from agent.routing.state import get_routing_state  # noqa: F401 — re-exported

from agent.routing.config import RoutingConfig, load_routing_config
from agent.routing.interaction_mode import InteractionMode, InteractionModeDetector
from agent.routing.turn_router import RoutingContext, RoutingDecision, TurnRouter

logger = logging.getLogger(__name__)


def get_routing_decision(agent: object, user_message: str) -> Optional[RoutingDecision]:
    """Called at the start of each turn to get a routing recommendation.

    Returns None if routing is disabled or on any error.
    Phase 2: also sets swap_required=True when the target position differs from
    the current routing position (or when the agent model mismatches the current
    position after a cloud-cover turn).
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
        decision = router.route(context)

        # ── Phase 2: update swap_required based on current routing position ──
        try:
            swap_mgr = _get_or_create_swap_manager(agent, config)
            current_pos = swap_mgr.current_position
            if current_pos is not None:
                if decision.target_position != current_pos:
                    decision.swap_required = True
                else:
                    # Same routing position — but check if the agent model drifted
                    # (e.g., we put agent on cloud cover last turn; local is now ready)
                    from agent.routing.model_resolver import ModelResolver
                    resolver = ModelResolver(config)
                    current_resolved = resolver.resolve(current_pos)
                    if current_resolved:
                        agent_model = getattr(agent, "model", "")
                        agent_provider = getattr(agent, "provider", "")
                        if (
                            current_resolved.model != agent_model
                            or current_resolved.provider != agent_provider
                        ):
                            decision.swap_required = True
        except Exception as exc:
            logger.debug("swap_required check failed (non-fatal): %s", exc)

        # ── Phase 3b: append to per-session decision history ring buffer ──
        try:
            history = getattr(agent, "_routing_decision_history", None)
            if history is None:
                history = deque(maxlen=20)
                setattr(agent, "_routing_decision_history", history)
            decision._timestamp = time.time()
            history.append(decision)
        except Exception:
            pass

        return decision

    except Exception as exc:
        logger.debug("get_routing_decision failed (non-fatal): %s", exc)
        return None


def execute_routing_swap(
    agent: object, decision: RoutingDecision
) -> Optional["ResolvedModel"]:  # noqa: F821
    """Execute a model swap based on the routing decision.

    Returns the ResolvedModel that should handle THIS turn, or None if the agent
    should stay on its current model unchanged.

    Called from conversation_loop.py when routing_decision.swap_required is True.
    """
    try:
        config = load_routing_config()
        if not config.enabled:
            return None

        from agent.routing.model_resolver import ModelResolver, ResolvedModel
        from agent.routing.swap_manager import SwapManager, SwapState

        swap_mgr = _get_or_create_swap_manager(agent, config)
        detector = _get_or_create_detector(agent, config)
        resolver = ModelResolver(config)

        target = decision.target_position
        current_pos = swap_mgr.current_position

        # ── Case 1: Return from cloud-cover turn ──
        # Current routing position matches target, but the agent is on a different
        # model (the cloud cover model used last turn). Switch back to the local model.
        if current_pos is not None and current_pos == target:
            current_resolved = resolver.resolve(current_pos)
            if current_resolved:
                agent_model = getattr(agent, "model", "")
                agent_provider = getattr(agent, "provider", "")
                if (
                    current_resolved.model != agent_model
                    or current_resolved.provider != agent_provider
                ):
                    logger.info(
                        "routing: returning from cloud cover → position=%r model=%s",
                        current_pos,
                        current_resolved.model,
                    )
                    return current_resolved
            return None

        # ── Case 2: Routing position change ──
        if not swap_mgr.should_swap(decision, detector):
            return None

        effective = swap_mgr.resolve_effective_model(decision)

        # If the target is a local model, kick off the background swap so the
        # new model is loading while the cloud model (cover) handles this turn.
        target_resolved = resolver.resolve(target)
        if (
            target_resolved is not None
            and target_resolved.is_local
            and swap_mgr.state not in (SwapState.SWAPPING, SwapState.READY)
        ):
            swap_mgr.execute_swap_background(target)

        return effective

    except Exception as exc:
        logger.debug("execute_routing_swap failed (non-fatal): %s", exc)
        return None


# ── Internal helpers ────────────────────────────────────────────────────────


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


def _get_or_create_swap_manager(agent: object, config: RoutingConfig) -> "SwapManager":
    """Retrieve or create the SwapManager cached on the agent (singleton per session)."""
    from agent.routing.swap_manager import SwapManager

    attr = "_routing_swap_manager"
    mgr = getattr(agent, attr, None)
    if mgr is None:
        mgr = SwapManager(config)
        _init_swap_manager_position(mgr, agent, config)
        try:
            setattr(agent, attr, mgr)
        except Exception:
            pass
    return mgr


def _init_swap_manager_position(
    mgr: "SwapManager", agent: object, config: RoutingConfig
) -> None:
    """Set swap manager's current position by checking the actual model loaded.
    
    Tries to match the currently loaded model (via local server or agent attributes)
    against the routing graph to determine the correct current position.
    """
    import json
    import urllib.request
    
    # First, check if a local model is loaded on :58080
    try:
        req = urllib.request.Request("http://127.0.0.1:58080/v1/models")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            local_models = data.get("models", [])
            if local_models:
                local_model_name = local_models[0].get("model") or local_models[0].get("name", "")
                logger.debug("routing: detected local model = %r", local_model_name)
                # Match against config graph
                for name, pos in config.graph.items():
                    if pos.provider == "custom:llm-local" and pos.model == local_model_name:
                        mgr.set_current_position(name)
                        logger.info(
                            "routing: initialized position=%r from local model %r",
                            name, local_model_name
                        )
                        return
    except Exception as exc:
        logger.debug("routing: failed to detect local model: %s", exc)
    
    # Fallback: use agent's provider/model attributes
    current_provider = getattr(agent, "provider", "")
    current_model = getattr(agent, "model", "")
    logger.debug("routing: falling back to agent attributes: %s/%s", current_provider, current_model)
    for name, pos in config.graph.items():
        if pos.provider == current_provider and pos.model == current_model:
            mgr.set_current_position(name)
            logger.info(
                "routing: initialized position=%r from agent attributes",
                name
            )
            return
    
    # Last resort: set to first position in graph (usually interactive_lower)
    if config.graph:
        first_position = next(iter(config.graph))
        mgr.set_current_position(first_position)
        logger.warning(
            "routing: could not detect model, defaulting to first position=%r",
            first_position
        )


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
