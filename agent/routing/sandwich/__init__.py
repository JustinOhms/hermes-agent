# hermes-agent - Sandwich Routing Public API
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 JustinOhms

"""
Public API for ADR-0041 Cooperative Model Routing (Sandwich Model).

This package provides a plugin-extensible model routing system with
three layers:
    1. Availability Filter (top bun) - constrain available models
    2. Routing Plugins (fillings) - rank models by preference
    3. Aggregator (bottom bun) - choose winner, build context

Example usage:
    from agent.routing.sandwich import run_sandwich_pipeline, build_default_config
    
    config = build_default_config()
    result = run_sandwich_pipeline(
        config=config,
        user_message="Hello",
        turn_id="1",
        session_id="abc",
        platform="cli",
    )
    
    if result:
        print(f"Selected: {result.model}@{result.provider}")
"""

from .pipeline import (
    PipelineConfig,
    run_sandwich_pipeline,
    build_default_config,
)

from .null_router import (
    NullAvailabilityFilter,
    NullRoutingPlugin,
    DefaultAggregator,
)

from .types import (
    ModelInfo,
    Ranking,
    RouteResult,
    AggregateResult,
    FilterResult,
    ForceDecision,
)

from .interfaces import (
    AvailabilityFilter,
    RoutingPlugin,
    Aggregator,
)

__all__ = [
    # Pipeline
    "PipelineConfig",
    "run_sandwich_pipeline",
    "build_default_config",
    # Null router (default behavior)
    "NullAvailabilityFilter",
    "NullRoutingPlugin",
    "DefaultAggregator",
    # Types
    "ModelInfo",
    "Ranking",
    "RouteResult",
    "AggregateResult",
    "FilterResult",
    "ForceDecision",
    # Interfaces
    "AvailabilityFilter",
    "RoutingPlugin",
    "Aggregator",
]
