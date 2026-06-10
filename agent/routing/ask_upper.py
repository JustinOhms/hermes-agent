"""ask_upper — Tool that lets the lower model query the upper model for guidance.

This is NOT escalation (full handoff). It's a synchronous query that returns
guidance, and the lower model continues working with the response.

Registered dynamically when:
  - model.routing.enabled = True
  - Current model occupies a lower-tier graph position

Per ADR-0040 §4, RD-19.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

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
# Budget tracking (uses shared BudgetTracker)
# ---------------------------------------------------------------------------

from agent.routing.budget import BudgetTracker


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
        self.budget = BudgetTracker(
            soft_limit=soft_budget,
            hard_limit=hard_budget,
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
                f"{self.budget.hard_limit}). Proceed with your best "
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
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
            self.budget.record_call(input_tokens=input_tokens, output_tokens=output_tokens)

            # ── Append budget warning if over soft limit ──
            if self.budget.over_soft:
                content += (
                    f"\n\n[NOTE: ask_upper call {self.budget.calls}/"
                    f"{self.budget.hard_limit}. Consider whether you "
                    "can proceed independently.]"
                )

            logger.info(
                "ask_upper: type=%s calls=%d/%d",
                request_type,
                self.budget.calls,
                self.budget.hard_limit,
            )
            return content

        except Exception as exc:
            # Graceful degradation — don't crash the session
            logger.warning("ask_upper failed: %s", type(exc).__name__)
            return (
                "[ask_upper UNAVAILABLE] The upper model could not be reached. "
                f"Error: {type(exc).__name__}\n\n"
                "Proceed with your best judgment."
            )

    def get_status(self) -> Dict[str, Any]:
        """Return current budget status for /routing display."""
        return self.budget.get_status()


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
                # Don't register for upper models (they'd be asking themselves)
                return name != "upper"

        # Model not in graph — conservative: don't register
        return False
    except Exception:
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

        # Get upper model config from graph
        upper_pos = config.graph.get("upper")
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
