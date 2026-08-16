"""Tests for agent/routing/drift_detector.py."""

from __future__ import annotations

import pytest

from agent.routing.config import (
    GraphPosition,
    GraphPositionProfile,
    RoutingConfig,
)
from agent.routing.drift_detector import DriftDetector, DriftAlert


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_config(
    ttft_p90_ms: float = 800.0,
    generation_tok_s: float = 60.0,
) -> RoutingConfig:
    profile = GraphPositionProfile(
        ttft_p90_ms=ttft_p90_ms,
        ttft_p50_ms=ttft_p90_ms * 0.6,
        generation_tok_s=generation_tok_s,
        startup_latency_s=10.0,
    )
    return RoutingConfig(
        enabled=True,
        graph={
            "interactive_lower": GraphPosition(
                provider="custom:llm-local",
                model="qwen3",
                profile=profile,
            )
        },
    )


# ── record_response ───────────────────────────────────────────────────────────

class TestRecordResponse:
    def test_records_without_error(self):
        cfg = _make_config()
        detector = DriftDetector(cfg)
        # Should not raise
        detector.record_response("interactive_lower", ttft_ms=500.0, generation_tok_s=55.0)

    def test_multiple_records_accumulated(self):
        cfg = _make_config()
        detector = DriftDetector(cfg)
        for i in range(5):
            detector.record_response("interactive_lower", ttft_ms=float(300 + i * 10), generation_tok_s=60.0)
        with detector._lock:
            assert len(detector._observations["interactive_lower"]) == 5

    def test_keeps_last_20_observations(self):
        cfg = _make_config()
        detector = DriftDetector(cfg)
        for i in range(25):
            detector.record_response("interactive_lower", ttft_ms=float(i * 10), generation_tok_s=60.0)
        with detector._lock:
            assert len(detector._observations["interactive_lower"]) == 20

    def test_records_for_multiple_positions(self):
        profile = GraphPositionProfile(ttft_p90_ms=800.0, generation_tok_s=60.0)
        cfg = RoutingConfig(
            enabled=True,
            graph={
                "pos_a": GraphPosition(provider="x", model="m1", profile=profile),
                "pos_b": GraphPosition(provider="y", model="m2", profile=profile),
            },
        )
        detector = DriftDetector(cfg)
        detector.record_response("pos_a", 500.0, 60.0)
        detector.record_response("pos_b", 600.0, 55.0)
        with detector._lock:
            assert "pos_a" in detector._observations
            assert "pos_b" in detector._observations


# ── check_drift ───────────────────────────────────────────────────────────────

class TestCheckDrift:
    def test_not_enough_data_returns_none(self):
        cfg = _make_config()
        detector = DriftDetector(cfg)
        # Only 2 observations — below MIN_OBSERVATIONS (3)
        detector.record_response("interactive_lower", 900.0, 60.0)
        detector.record_response("interactive_lower", 1100.0, 60.0)
        assert detector.check_drift("interactive_lower") is None

    def test_unknown_position_returns_none(self):
        cfg = _make_config()
        detector = DriftDetector(cfg)
        assert detector.check_drift("ghost_position") is None

    def test_no_drift_within_thresholds(self):
        cfg = _make_config(ttft_p90_ms=800.0, generation_tok_s=60.0)
        detector = DriftDetector(cfg)
        # TTFT around 700ms (within 1.3× of 800ms = 1040ms)
        # gen speed around 55 tok/s (above 0.7× of 60 = 42 tok/s)
        for _ in range(5):
            detector.record_response("interactive_lower", ttft_ms=700.0, generation_tok_s=55.0)
        assert detector.check_drift("interactive_lower") is None

    def test_ttft_drift_detected(self):
        cfg = _make_config(ttft_p90_ms=800.0, generation_tok_s=60.0)
        detector = DriftDetector(cfg)
        # p90 TTFT = 1100ms, which is 1.375× of 800ms > 1.3× threshold
        for _ in range(10):
            detector.record_response("interactive_lower", ttft_ms=1100.0, generation_tok_s=60.0)
        alert = detector.check_drift("interactive_lower")
        assert alert is not None
        assert alert.metric == "ttft"
        assert alert.position == "interactive_lower"
        assert alert.observed == pytest.approx(1100.0)
        assert alert.baseline == pytest.approx(800.0)
        assert alert.ratio > 1.3

    def test_generation_speed_drift_detected(self):
        cfg = _make_config(ttft_p90_ms=800.0, generation_tok_s=60.0)
        detector = DriftDetector(cfg)
        # p50 gen speed = 35 tok/s, which is 0.583× of 60 < 0.7× threshold
        for _ in range(5):
            detector.record_response("interactive_lower", ttft_ms=500.0, generation_tok_s=35.0)
        alert = detector.check_drift("interactive_lower")
        assert alert is not None
        assert alert.metric == "generation_speed"
        assert alert.observed == pytest.approx(35.0)
        assert alert.baseline == pytest.approx(60.0)
        assert alert.ratio < 0.7

    def test_no_baseline_ttft_skips_ttft_check(self):
        cfg = _make_config(ttft_p90_ms=0.0, generation_tok_s=60.0)
        detector = DriftDetector(cfg)
        # Even with high TTFT, no alert if baseline is 0
        for _ in range(5):
            detector.record_response("interactive_lower", ttft_ms=9999.0, generation_tok_s=55.0)
        alert = detector.check_drift("interactive_lower")
        assert alert is None  # ttft has no baseline; gen speed is fine

    def test_no_baseline_gen_speed_skips_gen_check(self):
        cfg = _make_config(ttft_p90_ms=800.0, generation_tok_s=0.0)
        detector = DriftDetector(cfg)
        for _ in range(5):
            detector.record_response("interactive_lower", ttft_ms=700.0, generation_tok_s=0.1)
        # No gen speed baseline, and TTFT is fine
        alert = detector.check_drift("interactive_lower")
        assert alert is None

    def test_ttft_alert_takes_priority_over_gen_speed(self):
        """TTFT drift is checked first; if detected, gen speed check is skipped."""
        cfg = _make_config(ttft_p90_ms=800.0, generation_tok_s=60.0)
        detector = DriftDetector(cfg)
        # Both TTFT and gen speed are drifting
        for _ in range(5):
            detector.record_response("interactive_lower", ttft_ms=1200.0, generation_tok_s=30.0)
        alert = detector.check_drift("interactive_lower")
        assert alert is not None
        assert alert.metric == "ttft"
