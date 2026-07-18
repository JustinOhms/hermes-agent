# hermes-agent - Sandwich Routing Interfaces
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 JustinOhms

"""
Abstract base classes for ADR-0041 Cooperative Model Routing (Sandwich Model).

This module defines the plugin interfaces that third-party routing plugins
must implement: AvailabilityFilter, RoutingPlugin, and Aggregator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from typing_extensions import Protocol

from .types import ModelInfo, Ranking, RouteResult, AggregateResult, FilterResult, ForceDecision


class AvailabilityFilter(ABC):
    """
    Top Bun: Constrains which models/providers are available for routing.
    
    Filters are run first in the pipeline and can reduce the candidate set
    based on availability, rate limits, quotas, or other constraints.
    """
    
    @abstractmethod
    def filter_available(
        self,
        models: List[ModelInfo],
        providers: List[str],
        *,
        user_message: str,
        session_id: str,
        platform: str,
    ) -> FilterResult:
        """
        Filter the available models and providers.
        
        Args:
            models: All available model definitions from config
            providers: All available provider names from config
            user_message: The current user message
            session_id: The active session ID
            platform: The client platform (e.g., "cli", "telegram")
        
        Returns:
            FilterResult with the filtered models/providers and reasons
        """
        pass


class RoutingPlugin(ABC):
    """
    Filling: Makes routing decisions based on message content and context.
    
    Routing plugins analyze the user message and conversation history to
    determine which model/provider combination is optimal for this turn.
    
    Plugins run in order defined in config. If a plugin returns a ForceDecision,
    subsequent plugins are skipped.
    """
    
    @property
    def name(self) -> str:
        """Return a unique identifier for this plugin."""
        return self.__class__.__name__
    
    @abstractmethod
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
        """
        Analyze the message and return routing recommendations.
        
        Args:
            user_message: The current user message
            conversation_history: Full conversation history (last 20 turns)
            available_models: Models that passed availability filter
            available_providers: Providers that passed availability filter
            current_rankings: Rankings from previous plugins in this turn
            session_id: The active session ID
            turn_id: The current turn identifier
            platform: The client platform (e.g., "cli", "telegram")
            current_model: The agent's current model name
            current_provider: The agent's current provider name
        
        Returns:
            RouteResult with rankings and/or force decision, or None for no opinion
        """
        pass


class Aggregator(ABC):
    """
    Bottom Bun: Chooses final model/provider and builds context.
    
    The aggregator receives all rankings from plugins and must select
    the winning model/provider. It also builds context metadata that
    will be injected into the user message via pre_llm_call.
    """
    
    @abstractmethod
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
        """
        Aggregate all routing opinions and produce final decision.
        
        Args:
            available_models: Models that passed availability filter
            available_providers: Providers that passed availability filter
            rankings: All rankings from routing plugins
            force_decision: First ForceDecision from plugins (if any)
            user_message: The current user message
            session_id: The active session ID
            turn_id: The current turn identifier
            platform: The client platform (e.g., "cli", "telegram")
        
        Returns:
            AggregateResult with final model/provider choice and context
        """
        pass


class PipelineConfig(Protocol):
    """Protocol for sandwich pipeline configuration."""
    
    @property
    def enabled(self) -> bool:
        """Whether the sandwich pipeline is enabled."""
        ...
    
    @property
    def availability_filter(self) -> AvailabilityFilter:
        """The availability filter plugin instance."""
        ...
    
    @property
    def routing_plugins(self) -> List[RoutingPlugin]:
        """List of routing plugin instances in execution order."""
        ...
    
    @property
    def aggregator_plugin(self) -> Aggregator:
        """The aggregator plugin instance."""
        ...


# Convenience imports
__all__ = [
    "AvailabilityFilter",
    "RoutingPlugin",
    "Aggregator",
    "PipelineConfig",
]
