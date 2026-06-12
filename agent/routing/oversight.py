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
from collections import deque

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Oversight Actions
# ---------------------------------------------------------------------------

class OversightAction(Enum):
    """Possible actions an oversight review can recommend."""

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


# ---------------------------------------------------------------------------
# Oversight Config (parsed from model.routing.oversight in config.yaml)
# ---------------------------------------------------------------------------

@dataclass
class OversightConfig:
    """Configuration for the periodic oversight reviewer."""

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
    prompt_file: str = ""  # Path to custom prompt file (YAML with 'prompt' key)


def load_oversight_config(cfg: Optional[Dict[str, Any]] = None) -> OversightConfig:
    """Load oversight config from model.routing.oversight section."""
    from agent.routing.config import _load_section, _safe_int, _safe_float

    oversight_raw = _load_section(cfg, "model", "routing", "oversight")
    if not isinstance(oversight_raw, dict):
        return OversightConfig()

    return OversightConfig(
        enabled=bool(oversight_raw.get("enabled", False)),
        model=str(oversight_raw.get("model", "")),
        provider=str(oversight_raw.get("provider", "")),
        base_url=str(oversight_raw.get("base_url", "")),
        api_key=str(oversight_raw.get("api_key", "")),
        every_n_turns=_safe_int(oversight_raw.get("every_n_turns"), 10, min_val=1),
        review_window=_safe_int(oversight_raw.get("review_window"), 10, min_val=1),
        review_window_ctx_fraction=_safe_float(
            oversight_raw.get("review_window_ctx_fraction"), 0.6, min_val=0.0
        ),
        review_window_min=_safe_int(oversight_raw.get("review_window_min"), 2, min_val=1),
        max_reviews_per_session=_safe_int(oversight_raw.get("max_reviews_per_session"), 5, min_val=0),
        min_turns_before_first=_safe_int(oversight_raw.get("min_turns_before_first"), 5, min_val=1),
        skip_if_escalated=bool(oversight_raw.get("skip_if_escalated", True)),
        upper_context_limit=_safe_int(oversight_raw.get("upper_context_limit"), 200000, min_val=10000),
        prompt_file=str(oversight_raw.get("prompt_file", "")),
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
        self._turn_token_counts = deque(maxlen=20)  # rolling window for RD-18
        self._budget_warning_shown: bool = False  # Track if user was warned about exhausted budget

    @property
    def review_count(self) -> int:
        """Number of reviews performed this session."""
        return len(self.reviews)

    @property
    def budget_exhausted(self) -> bool:
        """True when max_reviews_per_session has been reached."""
        return self.review_count >= self.config.max_reviews_per_session

    def should_review(
        self,
        turn_count: int,
        last_was_escalated: bool = False,
        ask_user_if_budget_exhausted: bool = True,
    ) -> tuple[bool, bool]:
        """Determine if a review should run at this turn count.
        
        Returns:
            (should_review, budget_exhausted)
            - should_review: True if review should be attempted
            - budget_exhausted: True if budget is exhausted (user should be prompted)
        """
        if not self.config.enabled:
            return (False, False)

        if self.budget_exhausted:
            if not self._budget_warning_shown and ask_user_if_budget_exhausted:
                # First time budget exhausted - prompt user
                self._budget_warning_shown = True
                logger.info(
                    "oversight: budget exhausted (%d/%d reviews). "
                    "Use /reset-oversight to reset budget or update max_reviews_per_session in config.",
                    self.review_count, self.config.max_reviews_per_session
                )
                # Return (False, True) to signal budget exhausted for UI to handle
                return (False, True)
            else:
                # Budget exhausted and user either declined or not prompted
                logger.debug(
                    "oversight: budget exhausted (%d/%d reviews). "
                    "Use /reset-oversight to reset budget.",
                    self.review_count, self.config.max_reviews_per_session
                )
                return (False, True)

        if turn_count < self.config.min_turns_before_first:
            return (False, False)

        if last_was_escalated and self.config.skip_if_escalated:
            logger.debug("oversight: skipping — last turn was escalated")
            return (False, False)

        # Check if it's a review turn (every N turns)
        if turn_count % self.config.every_n_turns != 0:
            return (False, False)

        return (True, False)

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
        self._turn_token_counts.append(token_count)  # deque handles maxlen=20 automatically

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

        # Build the oversight prompt (from file if specified, else builtin with examples)
        if self.config.prompt_file:
            prompt = self._load_custom_prompt(window_size, primary_model, turn_count)
        else:
            # Use builtin prompt with examples for each action type
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

            response = call_llm(
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

            # Track tokens
            input_tokens = 0
            output_tokens = 0
            usage = getattr(response, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0

            # Parse the action
            result = self._parse_response(content)
            result.window_size = window_size
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens

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
            )
            self.reviews.append(result)
            return result

    def _load_custom_prompt(self, window_size: int, primary_model: str, turn_count: int) -> str:
        """Load custom prompt from YAML file if specified."""
        import yaml
        from pathlib import Path
        
        prompt_path = Path(self.config.prompt_file)
        if not prompt_path.exists():
            logger.warning(
                "oversight: prompt_file '%s' not found, using builtin prompt",
                self.config.prompt_file,
            )
            return OVERSIGHT_PROMPT.format(
                window=window_size,
                primary_model=primary_model,
                oversight_model=self.config.model,
                turn_count=turn_count,
                review_number=self.review_count + 1,
            )
        
        try:
            data = yaml.safe_load(prompt_path.read_text())
            if not isinstance(data, dict) or "prompt" not in data:
                logger.warning(
                    "oversight: prompt file must have 'prompt' key, using builtin"
                )
                return OVERSIGHT_PROMPT.format(
                    window=window_size,
                    primary_model=primary_model,
                    oversight_model=self.config.model,
                    turn_count=turn_count,
                    review_number=self.review_count + 1,
                )
            
            base_prompt = data["prompt"]
            # Fill in the dynamic placeholders
            return base_prompt.format(
                window=window_size,
                primary_model=primary_model,
                oversight_model=self.config.model,
                turn_count=turn_count,
                review_number=self.review_count + 1,
            )
        except Exception as exc:
            logger.warning(
                "oversight: failed to load prompt from '%s': %s, using builtin",
                self.config.prompt_file, exc,
            )
            return OVERSIGHT_PROMPT.format(
                window=window_size,
                primary_model=primary_model,
                oversight_model=self.config.model,
                turn_count=turn_count,
                review_number=self.review_count + 1,
            )

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

        # Try direct JSON parsing first
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = self._extract_json(content)
            if data is None:
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

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract first valid JSON object from content using brace-counting."""
        import re
        
        # FIX #5: Use brace-counting to find the first valid JSON object
        # This handles nested braces in note/reason fields correctly
        brace_count = 0
        start_idx = None
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    try:
                        return json.loads(content[start_idx:i+1])
                    except json.JSONDecodeError:
                        pass
        
        # Fallback: try non-greedy regex
        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        logger.warning("oversight: could not extract JSON from: %s", content[:200])
        return None

    def reset_budget(self) -> None:
        """Reset the budget and clear all reviews."""
        self.reviews = []
        self._budget_warning_shown = False
        logger.info("oversight: budget reset - reviews cleared")

    def get_status(self) -> Dict[str, Any]:
        """Return current oversight status for /routing display."""
        last_review = self.reviews[-1] if self.reviews else None
        return {
            "enabled": self.config.enabled,
            "reviews_completed": self.review_count,
            "max_reviews": self.config.max_reviews_per_session,
            "budget_exhausted": self.budget_exhausted,
            "every_n_turns": self.config.every_n_turns,
            "last_action": last_review.action.value if last_review else None,
            "last_timestamp": last_review.timestamp if last_review else None,
            "history": [
                {
                    "action": r.action.value,
                    "timestamp": r.timestamp,
                    "window_size": r.window_size,
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
    ask_user_if_budget_exhausted: bool = True,
) -> Optional[tuple[OversightResult, bool]]:
    """Check if oversight is due and run it if so.

    Called from conversation_loop.py after a turn completes.
    
    Returns:
        (OversightResult, budget_exhausted) if a review ran
        None if no review was needed or budget exhausted (user needs to be prompted)
        
    Side effect: When budget exhausted, sets agent._oversight_budget_exhausted = True
    """
    reviewer = get_or_create_oversight_reviewer(agent)
    if reviewer is None:
        return None

    # Check if last turn was an escalation
    last_was_escalated = getattr(agent, "_oversight_last_escalated", False)

    should_review, budget_exhausted = reviewer.should_review(
        turn_count, last_was_escalated, ask_user_if_budget_exhausted
    )
    
    if not should_review:
        if budget_exhausted:
            # Budget exhausted - signal to prompt user
            try:
                setattr(agent, "_oversight_budget_exhausted", True)
            except Exception:
                pass
            return None
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

    # Clear escalation flag
    try:
        setattr(agent, "_oversight_last_escalated", result.action == OversightAction.ESCALATE)
    except Exception:
        pass

    return (result, budget_exhausted)


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


def build_routing_transition_injection(
    transition_type: str,
    from_model: str,
    to_model: str,
    reason: str,
    context: str = "",
) -> Dict[str, str]:
    """Build the message to inject when a routing transition occurs.

    Used for escalation, de-escalation, manual upgrade/downgrade.

    Args:
        transition_type: One of "escalate", "de-escalate", "upgrade", "downgrade"
        from_model: The model that was active before the transition
        to_model: The model that is now active
        reason: Why the transition occurred (escalation reason, user request, etc.)
        context: Optional additional context (handback summary, user prompt, etc.)

    Returns:
        A system message dict ready to insert into the messages list.
    """
    context_part = ""
    if context:
        context_part = f"Context: {context}\n"
    return {
        "role": "system",
        "content": (
            f"[ROUTING TRANSITION: {transition_type.upper()}]\n"
            f"Trigger: {reason}\n"
            f"From: {from_model}\n"
            f"To: {to_model}\n"
            f"\n{context_part}"
            f"You are now active. Proceed with the current task."
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


def reset_budget_for_agent(agent: object) -> bool:
    """Reset the budget for an agent's oversight reviewer.
    
    Returns True if successful, False if oversight is disabled or no reviewer exists.
    """
    try:
        reviewer = get_or_create_oversight_reviewer(agent)
        if reviewer is None:
            return False
        reviewer.reset_budget()
        # Clear agent flag
        try:
            delattr(agent, "_oversight_budget_exhausted")
        except AttributeError:
            pass
        return True
    except Exception as exc:
        logger.warning("Failed to reset oversight budget: %s", exc)
        return False
