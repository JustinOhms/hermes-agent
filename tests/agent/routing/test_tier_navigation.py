"""RoutingConfig tier-navigation helpers — the addressing layer behind /route.

These back the /route command (tier number | alias | up | down) and the
engine's relative escalate/de-escalate, so they get their own focused coverage.
"""
from __future__ import annotations

from agent.routing.config import GraphPosition, RoutingConfig


def _cfg() -> RoutingConfig:
    return RoutingConfig(
        enabled=True,
        graph={
            # Intentionally inserted out of tier order to prove sorting.
            "gamma": GraphPosition(provider="openai-codex", model="gpt-5.6-sol", tier=3, alias="Gamma"),
            "alpha": GraphPosition(provider="custom:llm-local", model="qwen", tier=1, alias="Alpha"),
            "beta": GraphPosition(provider="openai-codex", model="gpt-5.6-terra", tier=2, alias="Beta"),
        },
    )


class TestOrdering:
    def test_ordered_by_tier(self):
        assert _cfg().ordered_positions() == ["alpha", "beta", "gamma"]

    def test_base_and_top(self):
        cfg = _cfg()
        assert cfg.base_position() == "alpha"
        assert cfg.top_position() == "gamma"

    def test_empty_graph(self):
        cfg = RoutingConfig(enabled=True, graph={})
        assert cfg.ordered_positions() == []
        assert cfg.base_position() is None
        assert cfg.top_position() is None


class TestRelativeNavigation:
    def test_above(self):
        cfg = _cfg()
        assert cfg.position_above("alpha") == "beta"
        assert cfg.position_above("beta") == "gamma"
        assert cfg.position_above("gamma") is None  # nothing above the top

    def test_below(self):
        cfg = _cfg()
        assert cfg.position_below("gamma") == "beta"
        assert cfg.position_below("beta") == "alpha"
        assert cfg.position_below("alpha") is None  # nothing below the base

    def test_unknown_position(self):
        cfg = _cfg()
        assert cfg.position_above("nope") is None
        assert cfg.position_below("nope") is None


class TestResolveLabel:
    def test_by_key(self):
        assert _cfg().resolve_label("beta") == "beta"

    def test_by_alias_case_insensitive(self):
        cfg = _cfg()
        assert cfg.resolve_label("Gamma") == "gamma"
        assert cfg.resolve_label("gamma") == "gamma"
        assert cfg.resolve_label("ALPHA") == "alpha"

    def test_by_tier_number(self):
        cfg = _cfg()
        assert cfg.resolve_label("1") == "alpha"
        assert cfg.resolve_label("2") == "beta"
        assert cfg.resolve_label(3) == "gamma"

    def test_unknown_returns_none(self):
        cfg = _cfg()
        assert cfg.resolve_label("delta") is None
        assert cfg.resolve_label("99") is None
        assert cfg.resolve_label("") is None
        assert cfg.resolve_label(None) is None
