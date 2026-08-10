# hermes-agent - Sandwich Types Tests
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 JustinOhms
"""Tests for ADR-0041 data types."""

import pytest

from agent.routing.sandwich.types import (
    ModelInfo,
    Ranking,
    RouteResult,
    AggregateResult,
    FilterResult,
    ForceDecision,
)


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_modelinfo_basic(self):
        """Test basic ModelInfo creation."""
        info = ModelInfo(
            model="claude-sonnet-4",
            provider="anthropic",
            position="upper",
        )
        assert info.model == "claude-sonnet-4"
        assert info.provider == "anthropic"
        assert info.position == "upper"
        assert info.is_local is False

    def test_modelinfo_hashable(self):
        """Test ModelInfo is hashable (frozen)."""
        info1 = ModelInfo(
            model="claude-sonnet-4",
            provider="anthropic",
            position="upper",
        )
        info2 = ModelInfo(
            model="claude-sonnet-4",
            provider="anthropic",
            position="upper",
        )
        assert info1 == info2
        assert hash(info1) == hash(info2)
        s = {info1, info2}
        assert len(s) == 1

    def test_modelinfo_equality(self):
        """Test ModelInfo equality comparison."""
        info1 = ModelInfo(
            model="claude-sonnet-4",
            provider="anthropic",
            position="upper",
        )
        info2 = ModelInfo(
            model="claude-sonnet-4",
            provider="anthropic",
            position="upper",
        )
        info3 = ModelInfo(
            model="gpt-4o",
            provider="openai",
            position="lower",
        )
        assert info1 == info2
        assert info1 != info3

    def test_modelinfo_with_metadata(self):
        """Test ModelInfo with custom metadata."""
        info = ModelInfo(
            model="claude-sonnet-4",
            provider="anthropic",
            position="upper",
            metadata={"quota": 1000, "rate_limit": 100},
        )
        assert info.metadata["quota"] == 1000
        assert info.metadata["rate_limit"] == 100


class TestRanking:
    """Tests for Ranking dataclass."""

    def test_ranking_basic(self):
        """Test basic Ranking creation."""
        ranking = Ranking(
            model="claude-sonnet-4",
            provider="anthropic",
            score=0.95,
            reason="best for complex tasks",
            plugin_name="smart-router",
        )
        assert ranking.model == "claude-sonnet-4"
        assert ranking.provider == "anthropic"
        assert ranking.score == 0.95
        assert ranking.reason == "best for complex tasks"
        assert ranking.plugin_name == "smart-router"
        assert ranking.priority == 0.5

    def test_ranking_serialization(self):
        """Test Ranking to_dict conversion."""
        ranking = Ranking(
            model="claude-sonnet-4",
            provider="anthropic",
            score=0.95,
            reason="best for complex tasks",
            plugin_name="smart-router",
            priority=0.8,
            sticky_turns=3,
            sticky_timeout_s=300,
        )
        d = ranking.to_dict()
        assert d["model"] == "claude-sonnet-4"
        assert d["score"] == 0.95
        assert d["sticky_turns"] == 3
        assert d["sticky_timeout_s"] == 300


class TestFilterResult:
    """Tests for FilterResult dataclass."""

    def test_filter_result_basic(self):
        """Test basic FilterResult creation."""
        from agent.routing.sandwich.types import ModelInfo
        models = [
            ModelInfo(model="m1", provider="p1", position="a"),
            ModelInfo(model="m2", provider="p2", position="b"),
        ]
        result = FilterResult(
            models=models,
            providers=["p1", "p2"],
            reasons=["passed filter"],
        )
        assert len(result.models) == 2
        assert result.providers == ["p1", "p2"]
        assert result.reasons == ["passed filter"]


class TestForceDecision:
    """Tests for ForceDecision dataclass."""

    def test_force_decision_basic(self):
        """Test basic ForceDecision creation."""
        decision = ForceDecision(
            model="claude-sonnet-4",
            provider="anthropic",
            reason="user explicitly requested",
            sticky_turns=2,
        )
        assert decision.model == "claude-sonnet-4"
        assert decision.provider == "anthropic"
        assert decision.reason == "user explicitly requested"
        assert decision.sticky_turns == 2

    def test_from_ranking(self):
        """Test ForceDecision.from_ranking classmethod."""
        from agent.routing.sandwich.types import Ranking
        ranking = Ranking(
            model="claude-sonnet-4",
            provider="anthropic",
            score=0.95,
            reason="best match",
            plugin_name="smart-router",
            sticky_turns=3,
            sticky_timeout_s=300,
        )
        decision = ForceDecision.from_ranking(ranking)
        assert decision.model == "claude-sonnet-4"
        assert decision.provider == "anthropic"
        assert decision.reason == "best match"
        assert decision.sticky_turns == 3
        assert decision.sticky_timeout_s == 300


class TestRouteResult:
    """Tests for RouteResult dataclass."""

    def test_route_result_basic(self):
        """Test basic RouteResult creation."""
        result = RouteResult(
            rankings=[
                Ranking(
                    model="claude-sonnet-4",
                    provider="anthropic",
                    score=0.95,
                    reason="best for complex tasks",
                    plugin_name="smart-router",
                ),
            ],
        )
        assert result.rankings is not None
        assert len(result.rankings) == 1

    def test_route_result_merge(self):
        """Test RouteResult merge operation."""
        r1 = RouteResult(
            rankings=[
                Ranking(
                    model="m1",
                    provider="p1",
                    score=0.8,
                    reason="from r1",
                    plugin_name="r1",
                ),
            ],
        )
        r2 = RouteResult(
            rankings=[
                Ranking(
                    model="m2",
                    provider="p2",
                    score=0.9,
                    reason="from r2",
                    plugin_name="r2",
                ),
            ],
        )
        merged = r1.merge(r2)
        assert merged.rankings is not None
        assert len(merged.rankings) == 2


class TestAggregateResult:
    """Tests for AggregateResult dataclass."""

    def test_aggregate_result_basic(self):
        """Test basic AggregateResult creation."""
        result = AggregateResult(
            model="claude-sonnet-4",
            provider="anthropic",
            context="routing: selected for complex tasks",
            reasons=["best match for user message"],
        )
        assert result.model == "claude-sonnet-4"
        assert result.provider == "anthropic"
        assert result.context == "routing: selected for complex tasks"
        assert result.reasons == ["best match for user message"]

    def test_aggregate_result_serialization(self):
        """Test AggregateResult to_dict conversion."""
        result = AggregateResult(
            model="claude-sonnet-4",
            provider="anthropic",
            context="routing: selected",
            reasons=["best match"],
            rankings=[
                Ranking(
                    model="claude-sonnet-4",
                    provider="anthropic",
                    score=0.95,
                    reason="best match",
                    plugin_name="smart-router",
                ),
            ],
        )
        d = result.to_dict()
        assert d["model"] == "claude-sonnet-4"
        assert result.rankings is not None
        assert len(result.rankings) == 1
        assert result.rankings[0].score == 0.95
