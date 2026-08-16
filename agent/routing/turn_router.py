"""Turn Router — classifies each turn and returns a routing decision."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from agent.routing.config import RoutingConfig
    from agent.routing.interaction_mode import InteractionMode, InteractionModeDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context & Decision dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RoutingContext:
    """Everything the router needs to make a decision."""
    user_message: str
    message_length: int
    conversation_turn_count: int
    is_cron: bool
    is_subagent: bool
    platform: str  # "cli", "telegram", etc.
    interaction_mode: "InteractionMode"
    recent_tool_calls: List[str] = field(default_factory=list)
    last_response_had_errors: bool = False
    explicit_mode_override: Optional[str] = None


@dataclass
class RoutingDecision:
    """The router's output."""
    target_position: str  # "interactive_lower", "autonomous_lower", "upper", "fast_fallback"
    complexity_score: float  # 0.0 - 1.0
    interaction_mode: "InteractionMode"
    reason: str
    swap_required: bool


# ---------------------------------------------------------------------------
# Complexity scoring — compiled patterns
# ---------------------------------------------------------------------------

_TECHNICAL_KEYWORDS = re.compile(
    r"\b(?:refactor|architecture|design|debug|tradeoff|trade.off|"
    r"explain why|compare|optimis|optimiz|migrate|implement|"
    r"analyse|analyze|diagnos|performance|bottleneck|concurren|"
    r"deadlock|race condition|security|vulnerab|injection)\b",
    re.IGNORECASE,
)

_REASONING_PATTERNS = re.compile(
    r"\b(?:explain why|why does|why is|why are|how does|how do|"
    r"compare .{0,40}(?:with|vs|versus|against)|"
    r"trade.?offs?|pros and cons|best approach|best way|"
    r"difference between|what is the difference)\b",
    re.IGNORECASE,
)

_MULTI_FILE_PATTERNS = re.compile(
    r"(?:multiple files?|across files?|several files?|"
    r"all files?|every file|throughout the codebase|"
    r"in file\b.{0,60}and\b.{0,60}in file\b)",
    re.IGNORECASE,
)

# Structured data markers that inflate message length without adding complexity
_STRUCTURED_DATA_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:\{|\[|---|\|)",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# TurnRouter
# ---------------------------------------------------------------------------

class TurnRouter:
    def __init__(
        self,
        config: "RoutingConfig",
        mode_detector: "InteractionModeDetector",
    ) -> None:
        self._config = config
        self._mode_detector = mode_detector

    def route(self, context: RoutingContext) -> RoutingDecision:
        """Classify a turn and return the routing decision.

        Relative-tier model (ADR-0041): the graph is an ordered pipeline by
        ``tier``. We sit on the base (lowest) tier by default and escalate one
        step up when a turn is complex; de-escalation (if enabled) steps down.
        The top tier is reserved for a manual ``/route`` — auto-routing never
        burns it. There are no more hardcoded role names.
        """
        cfg = self._config
        complexity = self._score_complexity(context)
        mode = context.interaction_mode

        base = cfg.base_position()
        if base is None:
            return RoutingDecision(
                target_position="",
                complexity_score=complexity,
                interaction_mode=mode,
                reason="no graph configured",
                swap_required=False,
            )

        # Subagent: stay on the base (local) model — respect delegation.
        if context.is_subagent:
            return RoutingDecision(
                target_position=base,
                complexity_score=complexity,
                interaction_mode=mode,
                reason="subagent context — base tier",
                swap_required=False,
            )

        if complexity >= cfg.complexity.escalation_threshold:
            target = cfg.position_above(base) or base   # one step up from base
            reason = (
                f"high complexity ({complexity:.2f} ≥ "
                f"{cfg.complexity.escalation_threshold}) → escalate to {target}"
            )
        elif (
            cfg.de_escalation.enabled
            and complexity <= cfg.complexity.de_escalation_threshold
        ):
            target = cfg.position_below(base) or base
            reason = (
                f"low complexity ({complexity:.2f} ≤ "
                f"{cfg.complexity.de_escalation_threshold}) → de-escalate to {target}"
            )
        else:
            target = base
            reason = f"nominal complexity ({complexity:.2f}) → base tier {base}"

        logger.debug("routing: target=%s complexity=%.2f reason=%s", target, complexity, reason)
        return RoutingDecision(
            target_position=target,
            complexity_score=complexity,
            interaction_mode=mode,
            reason=reason,
            swap_required=self._swap_required(target),
        )

    def _score_complexity(self, context: RoutingContext) -> float:
        """Heuristic complexity scorer (0.0 – 1.0).

        Signals:
        - Message length (>2000 chars = complex, discounted for structured data)
        - Technical keywords
        - Multi-file references
        - Reasoning request patterns
        - Recent error context
        """
        score = 0.0
        msg = context.user_message

        # --- Length signal (up to 0.25) ---
        structured_matches = len(_STRUCTURED_DATA_PATTERNS.findall(msg))
        effective_length = max(0, context.message_length - structured_matches * 200)
        if effective_length > 2000:
            length_contribution = 0.25
        elif effective_length > 800:
            length_contribution = 0.15
        elif effective_length > 300:
            length_contribution = 0.05
        else:
            length_contribution = 0.0
        score += length_contribution
        logger.debug(
            "complexity: length=%d effective=%d → +%.2f",
            context.message_length, effective_length, length_contribution,
        )

        # --- Technical keywords (up to 0.25) ---
        kw_matches = len(_TECHNICAL_KEYWORDS.findall(msg))
        kw_contribution = min(0.25, kw_matches * 0.08)
        score += kw_contribution
        logger.debug("complexity: tech_keywords=%d → +%.2f", kw_matches, kw_contribution)

        # --- Multi-file references (0.15) ---
        if _MULTI_FILE_PATTERNS.search(msg):
            score += 0.15
            logger.debug("complexity: multi-file ref → +0.15")

        # --- Reasoning patterns (up to 0.20) ---
        reasoning_matches = len(_REASONING_PATTERNS.findall(msg))
        reasoning_contribution = min(0.20, reasoning_matches * 0.10)
        score += reasoning_contribution
        logger.debug(
            "complexity: reasoning_patterns=%d → +%.2f",
            reasoning_matches, reasoning_contribution,
        )

        # --- Error context from last turn (0.15) ---
        if context.last_response_had_errors:
            score += 0.15
            logger.debug("complexity: last turn had errors → +0.15")

        result = min(1.0, score)
        logger.debug("complexity: final=%.2f", result)
        return result

    def _swap_required(self, target: str) -> bool:
        """Phase 1 stub — always False.

        Phase 2 swap_required logic lives in agent/routing/__init__.py:
        get_routing_decision() updates the flag after comparing target vs.
        the current SwapManager position.
        """
        return False
