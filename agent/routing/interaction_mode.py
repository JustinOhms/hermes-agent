"""Interaction mode detection — classifies turns as INTERACTIVE vs AUTONOMOUS."""

from __future__ import annotations

import logging
import time
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent.routing.config import RoutingConfig
    from agent.routing.turn_router import RoutingContext

logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    INTERACTIVE = "interactive"
    AUTONOMOUS = "autonomous"


class InteractionModeDetector:
    """Tracks user interaction patterns to classify mode.

    Detection heuristics (RD-11), ordered by confidence:
    1. Explicit override (user typed /mode autonomous)
    2. Cron/scheduled context (always autonomous)
    3. Platform signals (e.g., user offline on Telegram)
    4. Temporal: time since last user message > idle_threshold
    5. Consecutive agent turns > 5 without user input
    """

    _MAX_CONSECUTIVE_AGENT_TURNS = 5

    def __init__(self, config: "RoutingConfig") -> None:
        self._config = config
        self._last_user_message_ts: Optional[float] = None
        self._consecutive_agent_turns: int = 0
        # Timestamps of recent user messages within the swap-back window
        self._recent_user_message_ts: deque[float] = deque()

    def record_user_message(self) -> None:
        """Call when a user message arrives."""
        now = time.monotonic()
        self._last_user_message_ts = now
        self._consecutive_agent_turns = 0
        self._recent_user_message_ts.append(now)
        # Prune entries outside the window
        window = self._config.interaction_mode.swap_back_window_s
        cutoff = now - window
        while self._recent_user_message_ts and self._recent_user_message_ts[0] < cutoff:
            self._recent_user_message_ts.popleft()

    def record_agent_turn(self) -> None:
        """Call when the agent completes a turn."""
        self._consecutive_agent_turns += 1

    def current_mode(self, context: "RoutingContext") -> InteractionMode:
        """Classify current interaction mode."""
        # 1. Explicit override
        if context.explicit_mode_override == "autonomous":
            logger.debug("interaction_mode: explicit override → autonomous")
            return InteractionMode.AUTONOMOUS
        if context.explicit_mode_override == "interactive":
            logger.debug("interaction_mode: explicit override → interactive")
            return InteractionMode.INTERACTIVE

        # 2. Cron/scheduled context
        if context.is_cron:
            logger.debug("interaction_mode: cron context → autonomous")
            return InteractionMode.AUTONOMOUS

        # 3. Platform signals (stub — no platform-specific offline detection yet)
        # Future: check platform-specific presence signals

        # 4. Temporal idle threshold
        if self._last_user_message_ts is not None:
            idle_s = time.monotonic() - self._last_user_message_ts
            if idle_s > self._config.interaction_mode.idle_threshold_s:
                logger.debug(
                    "interaction_mode: idle %.0fs > threshold %ds → autonomous",
                    idle_s,
                    self._config.interaction_mode.idle_threshold_s,
                )
                return InteractionMode.AUTONOMOUS

        # 5. Consecutive agent turns
        if self._consecutive_agent_turns > self._MAX_CONSECUTIVE_AGENT_TURNS:
            logger.debug(
                "interaction_mode: %d consecutive agent turns → autonomous",
                self._consecutive_agent_turns,
            )
            return InteractionMode.AUTONOMOUS

        return InteractionMode.INTERACTIVE

    def sustained_engagement_detected(self) -> bool:
        """True if user sent N messages within the swap-back window (RD-16)."""
        now = time.monotonic()
        window = self._config.interaction_mode.swap_back_window_s
        cutoff = now - window
        recent_count = sum(1 for ts in self._recent_user_message_ts if ts >= cutoff)
        threshold = self._config.interaction_mode.swap_back_messages
        result = recent_count >= threshold
        logger.debug(
            "sustained_engagement: %d messages in window (threshold=%d) → %s",
            recent_count,
            threshold,
            result,
        )
        return result
