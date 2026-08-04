"""Model resolver — translates graph position names to concrete model configs."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent.routing.config import RoutingConfig

logger = logging.getLogger(__name__)

_LOCAL_ENDPOINT = "http://127.0.0.1:58080/v1"
_LLM_CLI = "llm"


@dataclass
class ResolvedModel:
    """A fully resolved model ready for agent.switch_model()."""
    provider: str
    model: str
    base_url: str
    api_key: str
    api_mode: str
    is_local: bool
    llm_config_name: str  # e.g. "coder-next" — for `llm start <name>`


class ModelResolver:
    """Resolves graph position names to concrete model configurations."""

    def __init__(self, config: "RoutingConfig") -> None:
        self._config = config

    def resolve(self, position_name: str) -> Optional[ResolvedModel]:
        """Resolve a graph position to a concrete model config.

        Returns None if the position is not in the graph.
        """
        pos = self._config.graph.get(position_name)
        if pos is None:
            return None

        is_local = self._is_local(pos.base_url, pos.llm_config_name)
        # Default api_mode for local models is chat_completions
        api_mode = pos.api_mode or ("chat_completions" if is_local else "")
        base_url = pos.base_url or (_LOCAL_ENDPOINT if is_local else "")

        return ResolvedModel(
            provider=pos.provider,
            model=pos.model,
            base_url=base_url,
            api_key=pos.api_key,
            api_mode=api_mode,
            is_local=is_local,
            llm_config_name=pos.llm_config_name,
        )

    def is_position_local(self, position_name: str) -> bool:
        """True if this position uses the local llm server."""
        pos = self._config.graph.get(position_name)
        if pos is None:
            return False
        return self._is_local(pos.base_url, pos.llm_config_name)

    def current_local_model(self) -> Optional[str]:
        """Query `llm status` for the config name of the model actually running.

        Returns None if nothing is loaded (or the process isn't reported as
        running) so callers don't mistake a stale `active` name for a live
        server. `llm status` prints plain text (colors auto-disabled off a
        TTY), not JSON — e.g.:
            active:     coder-next
            status:     running (pid=15961)
        """
        try:
            proc = subprocess.run(
                [_LLM_CLI, "status"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if proc.returncode != 0:
                return None
            active = None
            running = False
            for line in proc.stdout.splitlines():
                if line.startswith("active:"):
                    active = line.split(":", 1)[1].strip()
                elif line.startswith("status:"):
                    running = line.split(":", 1)[1].strip().startswith("running")
            return active if (active and running) else None
        except Exception as exc:
            logger.debug("current_local_model: llm status failed: %s", exc)
        return None

    @staticmethod
    def _is_local(base_url: str, llm_config_name: str) -> bool:
        """Determine if a position uses the local llm server."""
        if llm_config_name:
            return True
        if base_url and "127.0.0.1" in base_url:
            return True
        return False
