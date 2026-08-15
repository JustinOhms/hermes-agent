"""ask_upper — Tool that lets the lower model query the upper model for guidance.

This is NOT escalation (full handoff). It's a synchronous query that returns
guidance, and the lower model continues working with the response.

Registered dynamically when:
  - model.routing.enabled = True
  - Current model occupies a lower-tier graph position

Per ADR-0040 §4, RD-19.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format)
# ---------------------------------------------------------------------------

ASK_UPPER_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "ask_upper",
    "description": (
        "Ask the upper (more capable) model for help with the current task. "
        "Use when you need: a complex problem simplified, a plan or checklist "
        "for a multi-step task, verification of your approach, or context "
        "distilled from a large conversation. The upper model will provide "
        "guidance and you continue working."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_type": {
                "type": "string",
                "enum": ["simplify", "plan", "verify", "distill", "explain"],
                "description": "What kind of help you need",
            },
            "question": {
                "type": "string",
                "description": "Your specific question or request for the upper model",
            },
            "context": {
                "type": "string",
                "description": (
                    "Relevant context the upper model needs to help you "
                    "(current state, what you've tried). Max ~4K tokens."
                ),
            },
        },
        "required": ["request_type", "question"],
    },
}


# ---------------------------------------------------------------------------
# Mentor system prompt
# ---------------------------------------------------------------------------

MENTOR_SYSTEM_PROMPT = """\
You are a senior mentor model providing guidance to a working AI agent.

The agent is handling a user's task and has asked you for help. Provide:
- Clear, actionable guidance (not vague)
- Concrete next steps when applicable
- Warnings about potential pitfalls
- Concise responses (the agent needs to get back to work)

You do NOT have access to tools or the user's environment. You can only provide
intellectual guidance based on what the agent tells you.

Respond directly — no preamble, no "Sure!" — just the guidance."""


# ---------------------------------------------------------------------------
# Request type prompts
# ---------------------------------------------------------------------------

_REQUEST_TYPE_PROMPTS = {
    "simplify": (
        "The agent says this problem is too complex to reason about directly. "
        "Break it down into manageable steps. Simplify the framing."
    ),
    "plan": (
        "The agent needs a strategy for a multi-step task. "
        "Provide a numbered checklist or decision tree."
    ),
    "verify": (
        "The agent has a plan and wants verification before executing. "
        "Review it: approve, correct, or warn about issues."
    ),
    "distill": (
        "The agent has too much context and needs the key facts extracted. "
        "Identify what matters for the current task, discard noise."
    ),
    "explain": (
        "The agent doesn't understand why something is happening. "
        "Provide an explanation, mental model, or analogy."
    ),
}


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------

@dataclass
class AskUpperBudget:
    """Tracks ask_upper usage per session."""
    calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    soft_budget_calls: int = 5  # warn after this many
    hard_budget_calls: int = 20  # refuse after this many
    timestamps: List[float] = field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.hard_budget_calls

    @property
    def over_soft_budget(self) -> bool:
        return self.calls >= self.soft_budget_calls


# ---------------------------------------------------------------------------
# AskUpperTool
# ---------------------------------------------------------------------------

class AskUpperTool:
    """Tool implementation for ask_upper.

    Follows the auxiliary_client.call_llm pattern for making the actual
    upper model call.
    """

    def __init__(
        self,
        upper_provider: str,
        upper_model: str,
        upper_base_url: str = "",
        upper_api_key: str = "",
        soft_budget: int = 5,
        hard_budget: int = 20,
        max_context_chars: int = 16000,  # ~4K tokens
        max_output_tokens: int = 2000,
    ) -> None:
        self.upper_provider = upper_provider
        self.upper_model = upper_model
        self.upper_base_url = upper_base_url
        self.upper_api_key = upper_api_key
        self.max_context_chars = max_context_chars
        self.max_output_tokens = max_output_tokens
        self.budget = AskUpperBudget(
            soft_budget_calls=soft_budget,
            hard_budget_calls=hard_budget,
        )

    def execute(
        self,
        request_type: str,
        question: str,
        context: str = "",
    ) -> str:
        """Execute an ask_upper call. Returns guidance text or error message."""

        # ── Budget check ──
        if self.budget.exhausted:
            return (
                "[ask_upper BUDGET EXHAUSTED] You have used ask_upper "
                f"{self.budget.calls} times this session (hard limit: "
                f"{self.budget.hard_budget_calls}). Proceed with your best "
                "judgment for the remainder of this session."
            )

        # ── Validate request_type ──
        if request_type not in _REQUEST_TYPE_PROMPTS:
            return (
                f"[ask_upper ERROR] Invalid request_type '{request_type}'. "
                f"Valid types: {', '.join(_REQUEST_TYPE_PROMPTS.keys())}"
            )

        # ── Truncate context if too long ──
        if context and len(context) > self.max_context_chars:
            context = context[: self.max_context_chars] + "\n\n[...context truncated]"

        # ── Build the mentor prompt ──
        type_instruction = _REQUEST_TYPE_PROMPTS[request_type]
        user_prompt = f"""{type_instruction}

**Agent's question:** {question}"""
        if context:
            user_prompt += f"\n\n**Context provided:**\n{context}"

        # ── Call upper model via auxiliary_client ──
        try:
            from agent.auxiliary_client import call_llm

            response = call_llm(
                task="ask_upper",
                provider=self.upper_provider,
                model=self.upper_model,
                base_url=self.upper_base_url or "",
                api_key=self.upper_api_key or "",
                messages=[
                    {"role": "system", "content": MENTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_output_tokens,
                temperature=0.3,  # Focused, deterministic guidance
            )

            # Extract response text
            content = ""
            if hasattr(response, "choices") and response.choices:
                msg = response.choices[0].message
                content = getattr(msg, "content", "") or ""
            elif hasattr(response, "content"):
                content = response.content or ""

            if not content:
                return "[ask_upper ERROR] Upper model returned empty response."

            # ── Track budget ──
            self.budget.calls += 1
            self.budget.timestamps.append(time.time())

            # Track tokens if available
            usage = getattr(response, "usage", None)
            if usage:
                self.budget.total_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.budget.total_output_tokens += getattr(usage, "completion_tokens", 0) or 0

            # ── Append budget warning if over soft limit ──
            if self.budget.over_soft_budget:
                content += (
                    f"\n\n[NOTE: ask_upper call {self.budget.calls}/"
                    f"{self.budget.hard_budget_calls}. Consider whether you "
                    "can proceed independently.]"
                )

            logger.info(
                "ask_upper: type=%s calls=%d/%d",
                request_type,
                self.budget.calls,
                self.budget.hard_budget_calls,
            )
            return content

        except Exception as exc:
            # Graceful degradation — don't crash the session
            logger.warning("ask_upper failed: %s", exc)
            return (
                "[ask_upper UNAVAILABLE] The upper model could not be reached. "
                f"Error: {exc}\n\n"
                "Proceed with your best judgment."
            )

    def get_status(self) -> Dict[str, Any]:
        """Return current budget status for /routing display."""
        return {
            "calls": self.budget.calls,
            "soft_budget": self.budget.soft_budget_calls,
            "hard_budget": self.budget.hard_budget_calls,
            "total_input_tokens": self.budget.total_input_tokens,
            "total_output_tokens": self.budget.total_output_tokens,
            "exhausted": self.budget.exhausted,
        }


# ---------------------------------------------------------------------------
# Lower-tier prompt nudge
# ---------------------------------------------------------------------------

ASK_UPPER_PROMPT_NUDGE = (
    "You are currently running as a fast, lower-tier model. If a task is "
    "complex, requires a multi-step plan, or you are unsure how to proceed, "
    "call the `ask_upper` tool to consult the stronger upper model for "
    "guidance, then continue the work yourself."
)


def ask_upper_prompt_nudge(agent: object) -> str:
    """Return the lower-tier prompt nudge, or ``""`` when it should not apply.

    Gated on exactly the same condition as tool registration
    (:func:`should_register_ask_upper`) so upper-tier models never see it.
    Non-fatal: any failure yields no nudge.
    """
    try:
        return ASK_UPPER_PROMPT_NUDGE if should_register_ask_upper(agent) else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def should_register_ask_upper(agent: object) -> bool:
    """Determine if ask_upper should be registered for this agent.

    Returns True when:
    - routing is enabled
    - the current model is in a lower-tier graph position (not upper)
    """
    try:
        from agent.routing.config import load_routing_config

        config = load_routing_config()
        if not config.enabled:
            return False

        # Check if current model is in a lower position
        current_provider = getattr(agent, "provider", "")
        current_model = getattr(agent, "model", "")

        for name, pos in config.graph.items():
            if pos.provider == current_provider and pos.model == current_model:
                # Register only if there's a higher tier to ask (nothing above
                # the top tier to consult).
                return config.position_above(name) is not None

        # Model not in graph — conservative: don't register
        return False
    except Exception:
        return False


def sync_ask_upper_registration(agent: object) -> bool:
    """Add or remove the ask_upper tool on ``agent`` for the current turn.

    Idempotent and position-aware: appends ``ASK_UPPER_TOOL_SCHEMA`` (wrapped in
    the OpenAI ``{"type": "function", "function": ...}`` envelope that
    ``agent.tools`` uses) and registers the name in ``agent.valid_tool_names``
    while :func:`should_register_ask_upper` holds, and strips both when it does
    not. Called each turn from the per-turn prologue, AFTER routing may have
    swapped the model tier and BEFORE the request's ``tools=`` is assembled.

    Returns True when ask_upper is present on ``agent.tools`` after the call.
    Non-fatal — never raises.
    """
    try:
        want = should_register_ask_upper(agent)
        tools = getattr(agent, "tools", None) or []

        def _is_ask_upper(entry: object) -> bool:
            return (
                isinstance(entry, dict)
                and entry.get("function", {}).get("name") == "ask_upper"
            )

        has = any(_is_ask_upper(t) for t in tools)

        if want and not has:
            if getattr(agent, "tools", None) is None:
                agent.tools = []
            agent.tools.append(
                {"type": "function", "function": ASK_UPPER_TOOL_SCHEMA}
            )
            names = getattr(agent, "valid_tool_names", None)
            if names is not None:
                names.add("ask_upper")
            return True
        if has and not want:
            agent.tools = [t for t in tools if not _is_ask_upper(t)]
            names = getattr(agent, "valid_tool_names", None)
            if names is not None:
                names.discard("ask_upper")
            return False
        return has
    except Exception:
        logger.debug("sync_ask_upper_registration failed", exc_info=True)
        return False


def get_or_create_ask_upper_tool(agent: object) -> Optional[AskUpperTool]:
    """Get or create the AskUpperTool instance cached on the agent."""
    attr = "_ask_upper_tool"
    tool = getattr(agent, attr, None)
    if tool is not None:
        return tool

    try:
        from agent.routing.config import load_routing_config

        config = load_routing_config()
        if not config.enabled:
            return None

        # Ask the tier immediately ABOVE the current model (relative), so a
        # routine consult hits the next rung up rather than always burning the
        # top tier. Fall back to the top tier if the current model isn't in the
        # graph.
        current_provider = getattr(agent, "provider", "")
        current_model = getattr(agent, "model", "")
        target_name: Optional[str] = None
        for name, pos in config.graph.items():
            if pos.provider == current_provider and pos.model == current_model:
                target_name = config.position_above(name)
                break
        if target_name is None:
            target_name = config.top_position()
        upper_pos = config.graph.get(target_name) if target_name else None
        if not upper_pos:
            return None

        tool = AskUpperTool(
            upper_provider=upper_pos.provider,
            upper_model=upper_pos.model,
            upper_base_url=upper_pos.base_url,
            upper_api_key=upper_pos.api_key,
        )
        try:
            setattr(agent, attr, tool)
        except Exception:
            pass
        return tool
    except Exception as exc:
        logger.debug("get_or_create_ask_upper_tool failed: %s", exc)
        return None
