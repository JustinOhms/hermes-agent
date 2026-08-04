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


def apply_turn_routing(agent: object, user_message: str) -> Optional[RoutingDecision]:
    """Per-turn routing entry point for ``build_turn_context``.

    Runs the Phase-1 decision and, when a swap is required, executes the
    Phase-2 model swap — switching the agent's runtime for THIS turn. Wired to
    fire after ``_restore_primary_runtime`` and before the ``pre_llm_call``
    hook so the routed model is the one the turn actually uses. Fully
    non-fatal: any error leaves the agent on its current model unchanged.

    Returns the RoutingDecision (for callers/tests), or None when routing is
    disabled or errored.
    """
    decision = get_routing_decision(agent, user_message)
    if decision is None:
        return None
    logger.info(
        "routing_decision: target=%s complexity=%.2f mode=%s reason=%s swap_required=%s",
        decision.target_position,
        decision.complexity_score,
        decision.interaction_mode.value,
        decision.reason,
        decision.swap_required,
    )
    if not decision.swap_required:
        return decision
    try:
        effective = execute_routing_swap(agent, decision)
        if effective and (
            effective.provider != getattr(agent, "provider", None)
            or effective.model != getattr(agent, "model", None)
        ):
            from agent.agent_runtime_helpers import switch_model as _switch_model
            _switch_model(
                agent,
                new_model=effective.model,
                new_provider=effective.provider,
                api_key=effective.api_key,
                base_url=effective.base_url,
                api_mode=effective.api_mode,
            )
            logger.info(
                "routing_swap: %s → %s (%s)",
                decision.target_position,
                effective.model,
                effective.provider,
            )
    except Exception as exc:
        logger.debug("apply_turn_routing swap failed (non-fatal): %s", exc)
    return decision


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
    """Set swap manager's current position for a freshly created SwapManager.

    SwapManager is cached per-agent-instance, so this runs fresh on every new
    session (e.g. each cron turn) — prefer the model actually running on the
    local server (via `llm status`) over guessing from the agent's own
    provider/model, otherwise a fresh session with no memory of a prior swap
    would conclude a swap is needed and redundantly restart an already-loaded
    local model. Falls back to matching the agent's active model/provider
    when nothing local is running (e.g. current position is a cloud model).
    """
    from agent.routing.model_resolver import ModelResolver

    running_local = ModelResolver(config).current_local_model()
    if running_local:
        for name, pos in config.graph.items():
            if pos.llm_config_name == running_local:
                mgr.set_current_position(name)
                return

    current_provider = getattr(agent, "provider", "")
    current_model = getattr(agent, "model", "")
    for name, pos in config.graph.items():
        if pos.provider == current_provider and pos.model == current_model:
            mgr.set_current_position(name)
            return


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
