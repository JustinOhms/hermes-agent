"""Periodic Oversight — Upper model reviews lower model output every N turns.

Runs synchronously between turns (per RD-5). Blocks until complete, then
injects corrections or escalates before the lower model's next turn.

Architecture (ADR-0040 §3):
  Every N turns:
    oversight_model.review(last_N_messages) → OversightAction
      → approve (silent, no cost beyond the review call)
      → correct (inject guidance into next turn)
      → escalate (oversight model takes over next turn)
      → flag (surface warning to user)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Oversight Actions
# ---------------------------------------------------------------------------

class OversightAction(Enum):
    APPROVE = "approve"
    CORRECT = "correct"
    ESCALATE = "escalate"
    FLAG = "flag"


@dataclass
class OversightResult:
    """Result of a single oversight review."""
    action: OversightAction
    note: str = ""       # For CORRECT: the guidance to inject
    reason: str = ""     # For ESCALATE: why the oversight model needs to take over
    warning: str = ""    # For FLAG: what to surface to the user
    timestamp: float = field(default_factory=time.time)
    window_size: int = 0  # How many turns were reviewed
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0          # wall-clock of the review call (metrics)
    est_cost_usd: Optional[float] = None  # estimated $ of the review call
    turn_count: int = 0           # session turn the review ran at


# ---------------------------------------------------------------------------
# Oversight Config (parsed from model.routing.oversight in config.yaml)
# ---------------------------------------------------------------------------

@dataclass
class OversightConfig:
    enabled: bool = False
    model: str = ""
    provider: str = ""
    base_url: str = ""
    api_key: str = ""
    every_n_turns: int = 10
    review_window: int = 10
    review_window_ctx_fraction: float = 0.6
    review_window_min: int = 2
    max_reviews_per_session: int = 5
    min_turns_before_first: int = 5
    skip_if_escalated: bool = True
    upper_context_limit: int = 200000  # tokens — Opus 4 default


def load_oversight_config(cfg: Optional[Dict[str, Any]] = None) -> OversightConfig:
    """Load oversight config from model.routing.oversight section."""
    if cfg is None:
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        except Exception:
            cfg = {}

    try:
        from hermes_cli.config import cfg_get
        oversight_raw = cfg_get(cfg, "model", "routing", "oversight") or {}
    except Exception:
        routing = (cfg.get("model") or {}).get("routing") or {}
        oversight_raw = routing.get("oversight") or {}

    if not isinstance(oversight_raw, dict):
        return OversightConfig()

    return OversightConfig(
        enabled=bool(oversight_raw.get("enabled", False)),
        model=str(oversight_raw.get("model", "")),
        provider=str(oversight_raw.get("provider", "")),
        base_url=str(oversight_raw.get("base_url", "")),
        api_key=str(oversight_raw.get("api_key", "")),
        every_n_turns=int(
            oversight_raw.get(
                "every_n_turns",
                # Back-compat alias: hand-written configs use review_interval_turns.
                oversight_raw.get("review_interval_turns", 10),
            )
        ),
        review_window=int(oversight_raw.get("review_window", 10)),
        review_window_ctx_fraction=float(
            oversight_raw.get("review_window_ctx_fraction", 0.6)
        ),
        review_window_min=int(oversight_raw.get("review_window_min", 2)),
        max_reviews_per_session=int(oversight_raw.get("max_reviews_per_session", 5)),
        min_turns_before_first=int(oversight_raw.get("min_turns_before_first", 5)),
        skip_if_escalated=bool(oversight_raw.get("skip_if_escalated", True)),
        upper_context_limit=int(oversight_raw.get("upper_context_limit", 200000)),
    )


# ---------------------------------------------------------------------------
# Oversight Prompt
# ---------------------------------------------------------------------------

OVERSIGHT_PROMPT = """\
You are reviewing the last {window} turns of an AI agent session.

Working model: {primary_model}
You are: {oversight_model}
Session turn count: {turn_count}
Review #{review_number} this session.

The agent is performing tasks autonomously. Review for:

1. **Logical errors** — wrong assumptions, hallucinated file contents or APIs
2. **Missed context** — information clearly available that the agent ignored
3. **Circular work** — repeating failed approaches without changing strategy
4. **Architecture mistakes** — technically works but wrong design direction
5. **Scope drift** — doing work the user didn't ask for, or missing the actual request
6. **Silent failures** — tool calls that returned errors the agent didn't notice

Respond with EXACTLY ONE valid JSON object (no markdown fencing, no explanation outside the JSON):

{{"action": "approve"}}
  — Work is correct, no intervention needed.

{{"action": "correct", "note": "..."}}
  — Inject this guidance for the next turn. Be specific and actionable.

{{"action": "escalate", "reason": "..."}}
  — The working model is stuck or making serious errors. You should handle the next turn directly.

{{"action": "flag", "warning": "..."}}
  — Alert the user about a concern (they may not be watching)."""


# ---------------------------------------------------------------------------
# OversightReviewer
# ---------------------------------------------------------------------------

class OversightReviewer:
    """Synchronous periodic reviewer that runs between turns.

    Cached on the agent instance. Tracks review count, history, and budget.
    """

    def __init__(self, config: OversightConfig) -> None:
        self.config = config
        self.reviews: List[OversightResult] = []
        self._turn_token_counts: List[int] = []  # rolling window for RD-18

    @property
    def review_count(self) -> int:
        return len(self.reviews)

    @property
    def budget_exhausted(self) -> bool:
        return self.review_count >= self.config.max_reviews_per_session

    def should_review(
        self,
        turn_count: int,
        last_was_escalated: bool = False,
    ) -> bool:
        """Determine if a review should run at this turn count."""
        if not self.config.enabled:
            return False

        if self.budget_exhausted:
            logger.debug("oversight: budget exhausted (%d/%d reviews)",
                         self.review_count, self.config.max_reviews_per_session)
            return False

        if turn_count < self.config.min_turns_before_first:
            return False

        if last_was_escalated and self.config.skip_if_escalated:
            logger.debug("oversight: skipping — last turn was escalated")
            return False

        # Check if it's a review turn (every N turns)
        if turn_count % self.config.every_n_turns != 0:
            return False

        return True

    def compute_effective_window(self, avg_tokens_per_turn: Optional[float] = None) -> int:
        """Compute the dynamic review window per RD-18.

        Formula: min(config.review_window, floor(upper_ctx * 0.6 / avg_tokens_per_turn))
        """
        configured_window = self.config.review_window

        if avg_tokens_per_turn is None:
            # Compute from our rolling tracker
            if len(self._turn_token_counts) >= 3:
                # Use p75 to avoid outlier shrinkage (per RD-18)
                sorted_counts = sorted(self._turn_token_counts)
                p75_idx = int(len(sorted_counts) * 0.75)
                avg_tokens_per_turn = float(sorted_counts[p75_idx])
            else:
                # Not enough data — use configured window
                return configured_window

        if avg_tokens_per_turn <= 0:
            return configured_window

        dynamic_cap = int(
            (self.config.upper_context_limit * self.config.review_window_ctx_fraction)
            / avg_tokens_per_turn
        )

        effective = min(configured_window, dynamic_cap)
        effective = max(effective, self.config.review_window_min)

        if effective < configured_window:
            logger.info(
                "oversight: review window reduced from %d to %d "
                "(avg_tokens_per_turn=%.0f, upper_ctx=%d)",
                configured_window, effective,
                avg_tokens_per_turn, self.config.upper_context_limit,
            )

        return effective

    def record_turn_tokens(self, token_count: int) -> None:
        """Record token count for a turn (used by dynamic window cap)."""
        self._turn_token_counts.append(token_count)
        # Keep rolling window of 20
        if len(self._turn_token_counts) > 20:
            self._turn_token_counts = self._turn_token_counts[-20:]

    def review(
        self,
        messages: List[Dict[str, Any]],
        primary_model: str,
        turn_count: int,
    ) -> OversightResult:
        """Execute a synchronous oversight review.

        Args:
            messages: Recent conversation messages to review (already windowed).
            primary_model: The lower model currently handling the session.
            turn_count: Current session turn count.

        Returns:
            OversightResult with the action and any associated data.
        """
        window_size = len([m for m in messages if m.get("role") in ("user", "assistant")])

        # Build the oversight prompt
        prompt = OVERSIGHT_PROMPT.format(
            window=window_size,
            primary_model=primary_model,
            oversight_model=self.config.model,
            turn_count=turn_count,
            review_number=self.review_count + 1,
        )

        # Format the messages for review
        review_content = self._format_messages_for_review(messages)

        try:
            from agent.auxiliary_client import call_llm

            _t0 = time.time()
            response = call_llm(
                task="oversight",
                provider=self.config.provider,
                model=self.config.model,
                base_url=self.config.base_url or "",
                api_key=self.config.api_key or "",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": review_content},
                ],
                max_tokens=500,  # Oversight responses should be concise
                temperature=0.1,  # Very focused
            )

            # Extract response
            content = ""
            if hasattr(response, "choices") and response.choices:
                msg = response.choices[0].message
                content = getattr(msg, "content", "") or ""
            elif hasattr(response, "content"):
                content = response.content or ""

            duration_ms = int((time.time() - _t0) * 1000)

            # Track tokens
            input_tokens = 0
            output_tokens = 0
            usage = getattr(response, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0

            # Estimate the review's cost for the metrics ledger (mirrors the aux
            # accounting recorder; the task="oversight" label above also books it
            # into session_model_usage for the usage dashboards).
            est_cost = _estimate_review_cost(
                usage, self.config.model, self.config.provider,
                self.config.base_url or "",
            )

            # Parse the action
            result = self._parse_response(content)
            result.window_size = window_size
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.duration_ms = duration_ms
            result.est_cost_usd = est_cost

            self.reviews.append(result)

            logger.info(
                "oversight: review #%d → %s (window=%d, tokens=%d+%d)",
                self.review_count,
                result.action.value,
                window_size,
                input_tokens,
                output_tokens,
            )

            return result

        except Exception as exc:
            logger.warning("oversight review failed: %s", exc)
            # On failure, approve by default (don't block the session)
            result = OversightResult(
                action=OversightAction.APPROVE,
                note=f"[Review failed: {exc}]",
                window_size=window_size,
                duration_ms=int((time.time() - _t0) * 1000),
            )
            self.reviews.append(result)
            return result

    def _format_messages_for_review(self, messages: List[Dict[str, Any]]) -> str:
        """Format conversation messages into a readable review payload."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle tool calls in assistant messages
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_summary = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "")
                    # Truncate long args
                    if len(args) > 500:
                        args = args[:500] + "..."
                    tc_summary.append(f"  → {name}({args})")
                tool_section = "\n".join(tc_summary)
                if content:
                    parts.append(f"[{role}]: {content}\n{tool_section}")
                else:
                    parts.append(f"[{role}]:\n{tool_section}")
            elif role == "tool":
                # Tool results — truncate aggressively
                name = msg.get("name", "tool")
                if isinstance(content, str) and len(content) > 1000:
                    content = content[:1000] + "\n[...truncated]"
                parts.append(f"[tool:{name}]: {content}")
            else:
                # User or assistant text
                if isinstance(content, str) and len(content) > 3000:
                    content = content[:3000] + "\n[...truncated]"
                elif isinstance(content, list):
                    # Multi-part content (vision, etc.)
                    text_parts = [
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = "\n".join(text_parts) or "[non-text content]"
                parts.append(f"[{role}]: {content}")

        return "\n\n".join(parts)

    def _parse_response(self, content: str) -> OversightResult:
        """Parse the oversight model's JSON response into an OversightResult."""
        content = content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning("oversight: could not parse response: %s", content[:200])
                    return OversightResult(action=OversightAction.APPROVE)
            else:
                logger.warning("oversight: no JSON in response: %s", content[:200])
                return OversightResult(action=OversightAction.APPROVE)

        action_str = data.get("action", "approve")
        try:
            action = OversightAction(action_str)
        except ValueError:
            logger.warning("oversight: unknown action '%s', defaulting to approve", action_str)
            action = OversightAction.APPROVE

        return OversightResult(
            action=action,
            note=str(data.get("note", "")),
            reason=str(data.get("reason", "")),
            warning=str(data.get("warning", "")),
        )

    def summarize(self, turns: Optional[int] = None) -> Dict[str, Any]:
        """Aggregate this session's reviews into a metrics summary.

        Shared by the live ``/routing oversight`` display and the durable
        per-session ledger, so both answer "was oversight used + working?":
        invocation count, the verdict mix (the working signal), duration, tokens,
        estimated cost, and — when ``turns`` is given — % of turns reviewed.
        """
        reviews = self.reviews
        actions = {a.value: 0 for a in OversightAction}
        for r in reviews:
            actions[r.action.value] = actions.get(r.action.value, 0) + 1
        total_dur = sum(r.duration_ms for r in reviews)
        return {
            "reviews": len(reviews),
            "actions": actions,
            "total_duration_ms": total_dur,
            "avg_duration_ms": int(total_dur / len(reviews)) if reviews else 0,
            "input_tokens": sum(r.input_tokens for r in reviews),
            "output_tokens": sum(r.output_tokens for r in reviews),
            "est_cost_usd": round(sum((r.est_cost_usd or 0.0) for r in reviews), 6),
            "turns": turns,
            "pct_of_turns": (
                round(100.0 * len(reviews) / turns, 1) if turns else None
            ),
            "first_ts": reviews[0].timestamp if reviews else None,
            "last_ts": reviews[-1].timestamp if reviews else None,
            "recent": [
                {
                    "turn": r.turn_count,
                    "ts": r.timestamp,
                    "action": r.action.value,
                    "duration_ms": r.duration_ms,
                }
                for r in reviews[-20:]
            ],
        }

    def get_status(self, turns: Optional[int] = None) -> Dict[str, Any]:
        """Return current oversight status for /routing display."""
        last_review = self.reviews[-1] if self.reviews else None
        summary = self.summarize(turns=turns)
        return {
            "enabled": self.config.enabled,
            "reviews_completed": self.review_count,
            "max_reviews": self.config.max_reviews_per_session,
            "budget_exhausted": self.budget_exhausted,
            "every_n_turns": self.config.every_n_turns,
            "last_action": last_review.action.value if last_review else None,
            "last_timestamp": last_review.timestamp if last_review else None,
            # Metrics (see summarize): verdict mix, duration, cost, %-of-turns.
            "actions": summary["actions"],
            "total_duration_ms": summary["total_duration_ms"],
            "avg_duration_ms": summary["avg_duration_ms"],
            "input_tokens": summary["input_tokens"],
            "output_tokens": summary["output_tokens"],
            "est_cost_usd": summary["est_cost_usd"],
            "pct_of_turns": summary["pct_of_turns"],
            "history": [
                {
                    "action": r.action.value,
                    "timestamp": r.timestamp,
                    "window_size": r.window_size,
                    "duration_ms": r.duration_ms,
                    "note": r.note[:100] if r.note else "",
                }
                for r in self.reviews[-5:]  # Last 5 reviews
            ],
        }


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def get_or_create_oversight_reviewer(agent: object) -> Optional[OversightReviewer]:
    """Get or create the OversightReviewer cached on the agent."""
    attr = "_oversight_reviewer"
    reviewer = getattr(agent, attr, None)
    if reviewer is not None:
        return reviewer

    config = load_oversight_config()
    if not config.enabled:
        return None

    # The reviewer needs an upper model to call. When the oversight config names
    # none explicitly, borrow the routing graph's top tier (the natural "upper"
    # model, e.g. gamma) so oversight works out of the box on a routed setup
    # instead of silently no-opping against an empty model (ADR-0040 §3).
    if not config.model or not config.provider:
        try:
            from agent.routing.config import load_routing_config

            rcfg = load_routing_config()
            top = rcfg.top_position() if (rcfg and rcfg.enabled) else None
            pos = rcfg.graph.get(top) if top else None
            if pos is not None:
                config.model = config.model or pos.model
                config.provider = config.provider or pos.provider
                config.base_url = config.base_url or pos.base_url
                config.api_key = config.api_key or pos.api_key
        except Exception:
            logger.debug(
                "oversight: could not derive review model from routing graph",
                exc_info=True,
            )

    if not config.model or not config.provider:
        logger.warning(
            "oversight: enabled but no review model is configured and none could "
            "be derived from the routing graph — reviews will be skipped"
        )
        return None

    reviewer = OversightReviewer(config)
    try:
        setattr(agent, attr, reviewer)
    except Exception:
        pass
    return reviewer


def run_oversight_if_due(
    agent: object,
    messages: List[Dict[str, Any]],
    turn_count: int,
) -> Optional[OversightResult]:
    """Check if oversight is due and run it if so.

    Called from conversation_loop.py after a turn completes.
    Returns the OversightResult if a review ran, None otherwise.
    """
    reviewer = get_or_create_oversight_reviewer(agent)
    if reviewer is None:
        return None

    # Check if last turn was an escalation
    last_was_escalated = getattr(agent, "_oversight_last_escalated", False)

    if not reviewer.should_review(turn_count, last_was_escalated):
        return None

    # Compute effective window
    effective_window = reviewer.compute_effective_window()

    # Extract the last N turns from messages
    # Count user+assistant pairs as "turns"
    review_messages = _extract_review_window(messages, effective_window)

    if not review_messages:
        return None

    # Get current primary model name
    primary_model = getattr(agent, "model", "unknown")

    # Execute the review
    result = reviewer.review(review_messages, primary_model, turn_count)
    result.turn_count = turn_count

    # Persist the durable per-session metrics ledger so a past session can be
    # inspected ("was oversight used + working?"). Best-effort, non-fatal.
    persist_oversight_ledger(reviewer, turn_count)

    # Clear escalation flag
    try:
        setattr(agent, "_oversight_last_escalated", result.action == OversightAction.ESCALATE)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Metrics: per-review cost + durable per-session ledger (state_meta KV)
# ---------------------------------------------------------------------------

_LEDGER_KEY_PREFIX = "oversight:"


def _ledger_key(session_id: str) -> str:
    return f"{_LEDGER_KEY_PREFIX}{session_id}"


def _estimate_review_cost(
    usage: Any, model: str, provider: str, base_url: str
) -> Optional[float]:
    """Estimate a review call's USD cost from its usage (best-effort, mirrors
    the aux-accounting recorder). Returns None when it cannot be priced."""
    if usage is None:
        return None
    try:
        from agent.usage_pricing import estimate_usage_cost, normalize_usage

        norm = normalize_usage(usage, provider=provider)
        cost = estimate_usage_cost(model, norm, provider=provider, base_url=base_url)
        return float(cost.amount_usd) if cost.amount_usd is not None else None
    except Exception:
        logger.debug("oversight: review cost estimate failed", exc_info=True)
        return None


def persist_oversight_ledger(reviewer: "OversightReviewer", turn_count: int) -> None:
    """Write the session's oversight summary to the session DB (best-effort).

    Reuses the ambient aux-accounting context (session_db, session_id) — the same
    handles the aux cost recorder uses, live during ``finalize_turn`` — and stores
    a JSON summary in the ``state_meta`` KV under ``oversight:<session_id>``. No
    schema change (fork-resync-safe). Never raises.
    """
    try:
        from agent.aux_accounting import get_accounting_context

        ctx = get_accounting_context()
        if not ctx:
            return
        session_db, session_id = ctx
        if session_db is None or not session_id:
            return
        summary = reviewer.summarize(turns=turn_count)
        session_db.set_meta(_ledger_key(session_id), json.dumps(summary))
    except Exception:
        logger.debug("oversight ledger persist failed (non-fatal)", exc_info=True)


def load_oversight_ledger(
    session_db: Any, session_id: str
) -> Optional[Dict[str, Any]]:
    """Read the persisted oversight summary for a session, or None if absent."""
    try:
        raw = session_db.get_meta(_ledger_key(session_id))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def build_oversight_injection(result: OversightResult, oversight_model: str) -> Dict[str, str]:
    """Build the message to inject into conversation when action is CORRECT.

    Returns a system message dict ready to insert into the messages list.
    """
    return {
        "role": "system",
        "content": (
            f"[OVERSIGHT NOTE from {oversight_model}]: {result.note}\n"
            f"Adjust your approach accordingly."
        ),
    }


def _extract_review_window(
    messages: List[Dict[str, Any]],
    window_turns: int,
) -> List[Dict[str, Any]]:
    """Extract the last N user-turn windows from messages.

    A "turn" is a user message + assistant response + any tool calls/results
    between them.
    """
    if not messages:
        return []

    # Walk backwards counting user messages
    user_count = 0
    start_idx = len(messages)

    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            user_count += 1
            if user_count >= window_turns:
                start_idx = i
                break

    # If we didn't find enough turns, take everything (minus system prompt)
    if start_idx == len(messages):
        # Find first non-system message
        start_idx = 0
        for i, m in enumerate(messages):
            if m.get("role") != "system":
                start_idx = i
                break

    return messages[start_idx:]
