"""Swap state machine — manages the lifecycle of model swaps."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent.routing.config import RoutingConfig
    from agent.routing.interaction_mode import InteractionModeDetector
    from agent.routing.model_resolver import ResolvedModel
    from agent.routing.turn_router import RoutingDecision

logger = logging.getLogger(__name__)

_LLM_CLI = "llm"
_SWAP_TIMEOUT_S = 30.0


class SwapState(Enum):
    IDLE = "idle"
    AWAITING_ENGAGEMENT = "awaiting_engagement"  # autonomous→interactive, not sustained yet
    SWAP_REQUESTED = "swap_requested"             # sustained engagement confirmed, about to swap
    SWAPPING = "swapping"                         # local model loading in background
    READY = "ready"                               # new model loaded, next turn will use it
    FAILED = "failed"                             # swap failed, staying on current model


@dataclass
class SwapResult:
    success: bool
    startup_time_s: float = 0.0
    error: str = ""


class SwapManager:
    """Manages model swap lifecycle.

    Responsibilities:
    - Track current routing position (which graph node is authoritative)
    - Decide WHEN to execute a swap (lazy swap-back for autonomous→interactive)
    - Execute local model swaps via `llm` CLI in background threads
    - Cloud-orchestrated transitions: swap in background, cloud covers current turn
    """

    def __init__(self, config: "RoutingConfig") -> None:
        self._config = config
        self._lock = threading.Lock()
        self._state: SwapState = SwapState.IDLE
        self._current_position: Optional[str] = None
        self._pending_position: Optional[str] = None

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def current_position(self) -> Optional[str]:
        """Currently active graph position name (routing's source of truth)."""
        with self._lock:
            return self._current_position

    @property
    def state(self) -> SwapState:
        """Current swap state."""
        with self._lock:
            return self._state

    # ── Position initialization ──────────────────────────────────────────────

    def set_current_position(self, position: str) -> None:
        """Set current position (called during initialization)."""
        with self._lock:
            self._current_position = position
            self._state = SwapState.IDLE

    # ── Decision logic ───────────────────────────────────────────────────────

    def should_swap(
        self,
        decision: "RoutingDecision",
        mode_detector: "InteractionModeDetector",
    ) -> bool:
        """Determine if a swap should actually execute now.

        Implements lazy swap-back (RD-16):
        - Escalation to upper: always swap immediately
        - Autonomous→interactive transition: wait for sustained engagement
        - Other position changes: swap immediately
        """
        from agent.routing.interaction_mode import InteractionMode

        target = decision.target_position

        with self._lock:
            current = self._current_position
            state = self._state

        if current is None or target == current:
            return False

        # Reset from a previous failure so the next cycle can retry
        if state == SwapState.FAILED:
            with self._lock:
                self._state = SwapState.IDLE
                state = SwapState.IDLE

        # Escalation to upper model: always immediate
        if target == "upper":
            return True

        # Autonomous→interactive transition (lazy swap-back, RD-16)
        is_auto_to_interactive = (
            decision.interaction_mode == InteractionMode.INTERACTIVE
            and current is not None
            and "autonomous" in current
        )

        if is_auto_to_interactive:
            if state == SwapState.AWAITING_ENGAGEMENT:
                if mode_detector.sustained_engagement_detected():
                    with self._lock:
                        self._state = SwapState.SWAP_REQUESTED
                        self._pending_position = target
                    return True
                return False
            elif state not in (SwapState.SWAPPING,):
                with self._lock:
                    self._state = SwapState.AWAITING_ENGAGEMENT
                    self._pending_position = target
                return False
            # If SWAPPING, fall through to return True (keep routing to cloud cover)

        # Still waiting for sustained engagement from a previous cycle
        if state == SwapState.AWAITING_ENGAGEMENT:
            if mode_detector.sustained_engagement_detected():
                with self._lock:
                    self._state = SwapState.SWAP_REQUESTED
                    self._pending_position = target
                return True
            return False

        # Default: swap immediately
        return True

    # ── Effective model resolution ───────────────────────────────────────────

    def resolve_effective_model(
        self, decision: "RoutingDecision"
    ) -> Optional["ResolvedModel"]:
        """Return the model that should handle THIS turn given current swap state.

        Cases:
        - Target is cloud → use it directly (no local swap needed)
        - Target local, already loaded → use it directly
        - Target local, swap in progress → cloud cover (upper position)
        - Target local, swap requested/about to start → cloud cover
        - Awaiting engagement → keep using current model
        """
        from agent.routing.model_resolver import ModelResolver

        resolver = ModelResolver(self._config)
        target = decision.target_position

        with self._lock:
            current = self._current_position
            state = self._state

        target_resolved = resolver.resolve(target)
        if target_resolved is None:
            return None

        # Cloud target: use directly, no local swap involved
        if not target_resolved.is_local:
            return target_resolved

        # Already on the right position and not in a degraded state
        if target == current and state not in (SwapState.FAILED,):
            return target_resolved

        # Swap in progress or just requested → cover with cloud/upper
        if state in (SwapState.SWAPPING, SwapState.SWAP_REQUESTED):
            upper = resolver.resolve("upper")
            if upper is not None and not upper.is_local:
                return upper
            # No cloud cover configured — return target anyway (will block on load)
            return target_resolved

        # Lazy-waiting for sustained engagement → stay on current model
        if state == SwapState.AWAITING_ENGAGEMENT:
            current_resolved = resolver.resolve(current) if current else None
            return current_resolved if current_resolved else target_resolved

        # Default: position change needed, cloud cover this turn while swap starts
        upper = resolver.resolve("upper")
        if upper is not None and not upper.is_local:
            return upper
        return target_resolved

    # ── Swap execution ───────────────────────────────────────────────────────

    def execute_swap_background(self, target_position: str) -> None:
        """Start a local model swap in the background (non-blocking).

        On success, updates current_position and state to READY.
        On failure, sets state to FAILED and logs a warning.
        """
        from agent.routing.model_resolver import ModelResolver

        resolver = ModelResolver(self._config)
        resolved = resolver.resolve(target_position)
        if resolved is None or not resolved.is_local:
            logger.warning(
                "execute_swap_background: position %r is unknown or not local",
                target_position,
            )
            return
        if not resolved.llm_config_name:
            logger.warning(
                "execute_swap_background: position %r has no llm_config_name",
                target_position,
            )
            return

        with self._lock:
            self._state = SwapState.SWAPPING
            self._pending_position = target_position

        config_name = resolved.llm_config_name

        def _run() -> None:
            logger.info("routing_swap: starting llm start %r", config_name)
            result = _start_local_model(config_name)
            with self._lock:
                if result.success:
                    self._current_position = target_position
                    self._state = SwapState.READY
                    logger.info(
                        "routing_swap: loaded %r in %.1fs",
                        config_name,
                        result.startup_time_s,
                    )
                else:
                    self._state = SwapState.FAILED
                    logger.warning(
                        "routing_swap: failed to load %r: %s",
                        config_name,
                        result.error,
                    )

        t = threading.Thread(
            target=_run,
            daemon=True,
            name=f"model-swap-{target_position}",
        )
        t.start()

    def execute_swap_sync(self, target_position: str) -> SwapResult:
        """Execute a local model swap synchronously (blocks until done or timeout).

        Primarily for use in tests and recovery paths.
        """
        from agent.routing.model_resolver import ModelResolver

        resolver = ModelResolver(self._config)
        resolved = resolver.resolve(target_position)
        if resolved is None or not resolved.is_local:
            return SwapResult(success=False, error=f"Unknown or non-local position: {target_position!r}")
        if not resolved.llm_config_name:
            return SwapResult(success=False, error=f"No llm_config_name for position: {target_position!r}")

        with self._lock:
            self._state = SwapState.SWAPPING
            self._pending_position = target_position

        result = _start_local_model(resolved.llm_config_name)

        with self._lock:
            if result.success:
                self._current_position = target_position
                self._state = SwapState.READY
            else:
                self._state = SwapState.FAILED
                logger.warning(
                    "routing_swap_sync: failed to load %r: %s",
                    resolved.llm_config_name,
                    result.error,
                )
        return result


# ── Module-level helper ─────────────────────────────────────────────────────

def _start_local_model(config_name: str, timeout: float = _SWAP_TIMEOUT_S) -> SwapResult:
    """Start a local model via llm CLI. Blocking — run in a thread.

    `llm start <name>` stops the current model, loads the new one, and
    performs an internal health check before returning.
    """
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [_LLM_CLI, "start", config_name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        if proc.returncode != 0:
            return SwapResult(
                success=False,
                error=(proc.stderr or proc.stdout or "non-zero exit").strip(),
                startup_time_s=elapsed,
            )
        return SwapResult(success=True, startup_time_s=elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return SwapResult(
            success=False,
            error=f"llm start {config_name!r} timed out after {timeout}s",
            startup_time_s=elapsed,
        )
    except FileNotFoundError:
        return SwapResult(success=False, error="llm CLI not found on PATH")
    except Exception as exc:
        return SwapResult(success=False, error=str(exc), startup_time_s=time.monotonic() - start)
