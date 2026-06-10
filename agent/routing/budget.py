"""Shared budget tracking for routing subsystems (ask_upper, oversight).

Both ask_upper and oversight need call-count tracking with soft/hard limits,
timestamp history, and status reporting. This module provides a single
BudgetTracker that both consume.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class BudgetTracker:
    """Generic call-budget tracker with soft/hard limits.

    Attributes:
        calls: Number of calls made this session.
        soft_limit: Warn (but don't refuse) after this many calls.
        hard_limit: Refuse calls after this many.
        total_input_tokens: Cumulative input tokens consumed.
        total_output_tokens: Cumulative output tokens consumed.
        timestamps: Timestamps of each call (for rate analysis).
    """

    soft_limit: int = 5
    hard_limit: int = 20
    calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    timestamps: List[float] = field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        """True when hard limit reached — further calls should be refused."""
        return self.calls >= self.hard_limit

    @property
    def over_soft(self) -> bool:
        """True when soft limit reached — warn but allow."""
        return self.calls >= self.soft_limit

    def record_call(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record a successful call with optional token counts."""
        self.calls += 1
        self.timestamps.append(time.time())
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def reset(self) -> None:
        """Reset all counters (e.g. /reset-oversight)."""
        self.calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.timestamps.clear()

    def get_status(self) -> Dict[str, Any]:
        """Return status dict suitable for /routing display."""
        return {
            "calls": self.calls,
            "soft_limit": self.soft_limit,
            "hard_limit": self.hard_limit,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "exhausted": self.exhausted,
        }
