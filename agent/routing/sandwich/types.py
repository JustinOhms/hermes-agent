# hermes-agent - Sandwich Routing Types
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 JustinOhms

"""
Data types for ADR-0041 Cooperative Model Routing (Sandwich Model).

This module defines the core data structures used throughout the sandwich
routing pipeline: ModelInfo, Ranking, RouteResult, AggregateResult, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class ModelInfo:
    """Represents an available model with its provider and metadata."""
    model: str
    provider: str
    position: str  # e.g., "upper", "lower", "interactive"
    base_url: str = ""
    api_mode: str = ""
    is_local: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash((self.model, self.provider, self.position))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelInfo):
            return False
        return (self.model == other.model and 
                self.provider == other.provider and 
                self.position == other.position)


@dataclass
class Ranking:
    """A single model/provider ranking from a routing plugin."""
    model: str
    provider: str
    score: float  # 0.0 - 1.0
    reason: str  # Human-readable explanation
    plugin_name: str  # Which plugin produced this
    priority: float = 0.5  # 0.0 - 1.0, for tie-breaking
    sticky_turns: Optional[int] = None  # Number of turns to stick
    sticky_timeout_s: Optional[int] = None  # Timeout in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "score": self.score,
            "reason": self.reason,
            "plugin_name": self.plugin_name,
            "priority": self.priority,
            "sticky_turns": self.sticky_turns,
            "sticky_timeout_s": self.sticky_timeout_s,
        }


@dataclass
class FilterResult:
    """Result of an availability filter run."""
    models: List[ModelInfo]
    providers: List[str]
    reasons: List[str] = field(default_factory=list)
    filtered_out: List[ModelInfo] = field(default_factory=list)


@dataclass
class ForceDecision:
    """A forced routing decision that stops plugin evaluation."""
    model: str
    provider: str
    reason: str
    sticky_turns: Optional[int] = None
    sticky_timeout_s: Optional[int] = None
    
    @classmethod
    def from_ranking(cls, ranking: Ranking) -> ForceDecision:
        return cls(
            model=ranking.model,
            provider=ranking.provider,
            reason=ranking.reason,
            sticky_turns=ranking.sticky_turns,
            sticky_timeout_s=ranking.sticky_timeout_s,
        )


@dataclass
class RouteResult:
    """The output of a RoutingPlugin.route() call."""
    rankings: Optional[List[Ranking]] = None
    models: Optional[List[ModelInfo]] = None
    providers: Optional[List[str]] = None
    force: Optional[ForceDecision] = None  # First force wins
    
    def merge(self, other: RouteResult) -> RouteResult:
        """Merge another RouteResult into this one."""
        new_rankings = list(self.rankings or []) + list(other.rankings or [])
        new_models = list(self.models or []) + list(other.models or [])
        new_providers = list(self.providers or []) + list(other.providers or [])
        new_force = self.force or other.force
        
        return RouteResult(
            rankings=new_rankings if new_rankings else None,
            models=new_models if new_models else None,
            providers=new_providers if new_providers else None,
            force=new_force,
        )


@dataclass
class AggregateResult:
    """The output of an Aggregator.aggregate() call."""
    model: str
    provider: str
    context: str  # Context to inject via pre_llm_call
    reasons: List[str] = field(default_factory=list)
    rankings: Optional[List[Ranking]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "context": self.context,
            "reasons": self.reasons,
            "rankings": [r.to_dict() for r in self.rankings or []],
        }


# Plugin type aliases for configuration
# These are forward references - actual classes defined in interfaces.py
AvailabilityFilterFactory = Callable[[], Any]
RoutingPluginFactory = Callable[[], Any]
AggregatorFactory = Callable[[], Any]
