# hermes-agent - Null Router Plugin
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 JustinOhms

"""
Null router plugin for ADR-0041 - provides default behavior.

This module implements the "null" version of each sandwich component
that maintains backward compatibility with the existing legacy routing
system while the pipeline is being tested.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from .interfaces import (
    AvailabilityFilter,
    RoutingPlugin,
    Aggregator,
)
from .types import (
    ModelInfo,
    Ranking,
    RouteResult,
    AggregateResult,
    FilterResult,
    ForceDecision,
)


class NullAvailabilityFilter(AvailabilityFilter):
    """Availability filter that passes all models through unchanged."""
    
    def filter_available(
        self,
        models: List[ModelInfo],
        providers: List[str],
        *,
        user_message: str,
        session_id: str,
        platform: str,
    ) -> FilterResult:
        return FilterResult(
            models=models,
            providers=providers,
            reasons=["null: all models passed through"],
        )


class NullRoutingPlugin(RoutingPlugin):
    """Routing plugin that returns no opinion - falls back to legacy behavior."""
    
    def route(
        self,
        *,
        user_message: str,
        conversation_history: List[Dict],
        available_models: List[ModelInfo],
        available_providers: List[str],
        current_rankings: List[Ranking],
        session_id: str,
        turn_id: str,
        platform: str,
        current_model: str,
        current_provider: str,
    ) -> Optional[RouteResult]:
        # Return None for no opinion - this allows the aggregator to use
        # the first available model as a fallback
        return None


class DefaultAggregator(Aggregator):
    """Aggregator that picks the first available model."""
    
    def aggregate(
        self,
        *,
        available_models: List[ModelInfo],
        available_providers: List[str],
        rankings: List[Ranking],
        force_decision: Optional[ForceDecision],
        user_message: str,
        session_id: str,
        turn_id: str,
        platform: str,
    ) -> Optional[AggregateResult]:
        # If there's a forced decision, use it
        if force_decision:
            return AggregateResult(
                model=force_decision.model,
                provider=force_decision.provider,
                context=f"routing: forced by plugin ({force_decision.reason})",
                reasons=[force_decision.reason],
            )
        
        # If rankings exist, pick the highest-scoring one
        if rankings:
            best = max(rankings, key=lambda r: r.score)
            return AggregateResult(
                model=best.model,
                provider=best.provider,
                context=f"routing: selected by {best.plugin_name} ({best.reason})",
                reasons=[best.reason],
            )
        
        # Fall back to first available model
        if available_models:
            first = available_models[0]
            return AggregateResult(
                model=first.model,
                provider=first.provider,
                context="routing: default (first available model)",
                reasons=["no plugins provided opinion, using first available"],
            )
        
        # No models available - this should be rare
        if available_providers:
            return AggregateResult(
                model="unknown",
                provider=available_providers[0],
                context="routing: no models for provider",
                reasons=["no models available for provider"],
            )
        
        return None
