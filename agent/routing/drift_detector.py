"""Drift detector — monitors model performance against profiled baselines."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from agent.routing.config import RoutingConfig

logger = logging.getLogger(__name__)

# Thresholds: alert when observed p90 TTFT > 1.3× baseline,
# or observed p50 generation speed < 0.7× baseline.
_TTFT_DRIFT_RATIO = 1.3
_GEN_SPEED_DRIFT_RATIO = 0.7
_MIN_OBSERVATIONS = 3    # require at least this many before checking drift
_MAX_OBSERVATIONS = 20   # keep only the most recent N observations per position


@dataclass
class DriftAlert:
    position: str
    metric: str        # "ttft" or "generation_speed"
    observed: float    # observed value (p90 TTFT ms or p50 tok/s)
    baseline: float    # profiled baseline
    ratio: float       # observed / baseline


class DriftDetector:
    """Monitors model performance and flags drift from profiled baselines.

    Records TTFT (ms) and generation speed (tok/s) per graph position.
    Checks:
    - p90 TTFT > 1.3× profiled p90_ttft_ms
    - p50 generation speed < 0.7× profiled generation_tok_s
    """

    def __init__(self, config: "RoutingConfig") -> None:
        self._config = config
        # position → list of (ttft_ms, generation_tok_s)
        self._observations: Dict[str, List[Tuple[float, float]]] = {}
        self._lock = threading.Lock()

    def record_response(
        self, position: str, ttft_ms: float, generation_tok_s: float
    ) -> None:
        """Record observed metrics for a response."""
        with self._lock:
            obs = self._observations.setdefault(position, [])
            obs.append((ttft_ms, generation_tok_s))
            # Keep only the most recent N observations
            if len(obs) > _MAX_OBSERVATIONS:
                self._observations[position] = obs[-_MAX_OBSERVATIONS:]

    def check_drift(self, position: str) -> Optional[DriftAlert]:
        """Check if recent observations drift significantly from the profile.

        Returns a DriftAlert if:
        - observed p90 TTFT > 1.3× profiled p90_ttft_ms, or
        - observed p50 generation speed < 0.7× profiled generation_tok_s.

        Returns None if there are fewer than _MIN_OBSERVATIONS or no baseline.
        """
        pos_config = self._config.graph.get(position)
        if pos_config is None:
            return None

        with self._lock:
            obs = list(self._observations.get(position, []))

        if len(obs) < _MIN_OBSERVATIONS:
            return None

        ttft_values = sorted(o[0] for o in obs)
        gen_values = sorted(o[1] for o in obs)

        # ── TTFT drift: check p90 ──
        baseline_ttft = pos_config.profile.ttft_p90_ms
        if baseline_ttft > 0:
            p90_idx = min(int(len(ttft_values) * 0.9), len(ttft_values) - 1)
            p90_ttft = ttft_values[p90_idx]
            if p90_ttft > baseline_ttft * _TTFT_DRIFT_RATIO:
                ratio = p90_ttft / baseline_ttft
                logger.warning(
                    "drift: position=%r TTFT p90=%.0fms baseline=%.0fms ratio=%.2f",
                    position, p90_ttft, baseline_ttft, ratio,
                )
                return DriftAlert(
                    position=position,
                    metric="ttft",
                    observed=p90_ttft,
                    baseline=baseline_ttft,
                    ratio=ratio,
                )

        # ── Generation speed drift: check p50 (median) ──
        baseline_gen = pos_config.profile.generation_tok_s
        if baseline_gen > 0:
            p50_idx = len(gen_values) // 2
            p50_gen = gen_values[p50_idx]
            if p50_gen < baseline_gen * _GEN_SPEED_DRIFT_RATIO:
                ratio = p50_gen / baseline_gen
                logger.warning(
                    "drift: position=%r gen_speed p50=%.1f tok/s baseline=%.1f tok/s ratio=%.2f",
                    position, p50_gen, baseline_gen, ratio,
                )
                return DriftAlert(
                    position=position,
                    metric="generation_speed",
                    observed=p50_gen,
                    baseline=baseline_gen,
                    ratio=ratio,
                )

        return None
