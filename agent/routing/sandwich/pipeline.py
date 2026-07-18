# hermes-agent - Sandwich Pipeline Orchestrator
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 JustinOhms

"""
Sandwich pipeline orchestrator for ADR-0041 Cooperative Model Routing.

This module contains the core logic that runs the three-layer routing
pipeline: AvailabilityFilter → RoutingPlugins → Aggregator.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from collections import deque

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

from .null_router import (
    NullAvailabilityFilter,
    NullRoutingPlugin,
    DefaultAggregator,
)

logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for the sandwich pipeline."""
    
    def __init__(
        self,
        enabled: bool = False,
        availability_filter: Optional[AvailabilityFilter] = None,
        routing_plugins: Optional[List[RoutingPlugin]] = None,
        aggregator_plugin: Optional[Aggregator] = None,
    ):
        self.enabled = enabled
        self.availability_filter = availability_filter or NullAvailabilityFilter()
        self.routing_plugins = routing_plugins or [NullRoutingPlugin()]
        self.aggregator_plugin = aggregator_plugin or DefaultAggregator()


def _build_initial_candidates(
    config: PipelineConfig,
) -> tuple[List[ModelInfo], List[str]]:
    """
    Build initial model/provider candidates from routing config.
    
    This is a placeholder that will be replaced by loading from
    the routing graph in a later phase.
    """
    # For now, return empty - actual implementation will
    # parse config.graph and convert positions to ModelInfo
    return [], []


def run_sandwich_pipeline(
    config: PipelineConfig,
    user_message: str,
    turn_id: str,
    session_id: str,
    platform: str,
    conversation_history: Optional[List[Dict]] = None,
    current_model: str = "",
    current_provider: str = "",
) -> Optional[AggregateResult]:
    """
    Execute the sandwich pipeline.
    
    Args:
        config: The pipeline configuration
        user_message: The current user message
        turn_id: The current turn identifier
        session_id: The active session ID
        platform: The client platform (e.g., "cli", "telegram")
        conversation_history: Full conversation history (optional)
        current_model: The agent's current model
        current_provider: The agent's current provider
    
    Returns:
        AggregateResult with final decision, or None if pipeline disabled
    """
    if not config.enabled:
        logger.debug("sandwich_pipeline: disabled, skipping")
        return None
    
    try:
        # 1. Build initial candidates from config
        models, providers = _build_initial_candidates(config)
        logger.debug(
            "sandwich_pipeline: initial candidates: %d models, %d providers",
            len(models), len(providers)
        )
        
        # 2. Top Bun: Availability filter
        filter_result: FilterResult = config.availability_filter.filter_available(
            models=models,
            providers=providers,
            user_message=user_message,
            session_id=session_id,
            platform=platform,
        )
        
        logger.debug(
            "sandwich_pipeline: after availability filter: %d models, %d providers",
            len(filter_result.models), len(filter_result.providers)
        )
        
        # 3. Fillings: Routing plugins (in order, first force wins)
        rankings: List[Ranking] = []
        force_decision: Optional[ForceDecision] = None
        
        for plugin in config.routing_plugins:
            result: Optional[RouteResult] = plugin.route(
                user_message=user_message,
                conversation_history=conversation_history or [],
                available_models=filter_result.models,
                available_providers=filter_result.providers,
                current_rankings=list(rankings),
                session_id=session_id,
                turn_id=turn_id,
                platform=platform,
                current_model=current_model,
                current_provider=current_provider,
            )
            
            if result is None:
                continue  # No opinion from this plugin
            
            # Collect rankings
            if result.rankings:
                rankings.extend(result.rankings)
            
            # Check for forced decision (first wins)
            if result.force and force_decision is None:
                force_decision = result.force
                logger.info(
                    "sandwich_pipeline: plugin %s forced decision: %s/%s",
                    plugin.name, result.force.model, result.force.provider
                )
                break  # Stop consulting other plugins
        
        # 4. Bottom Bun: Aggregator (always runs)
        agg_result: Optional[AggregateResult] = config.aggregator_plugin.aggregate(
            available_models=filter_result.models,
            available_providers=filter_result.providers,
            rankings=rankings,
            force_decision=force_decision,
            user_message=user_message,
            session_id=session_id,
            turn_id=turn_id,
            platform=platform,
        )
        
        if agg_result:
            logger.info(
                "sandwich_pipeline: selected %s/%s",
                agg_result.model, agg_result.provider
            )
        
        return agg_result
        
    except Exception as exc:
        logger.warning("sandwich_pipeline failed: %s", exc, exc_info=True)
        return None


def build_default_config() -> PipelineConfig:
    """Build a default config with null components (backward compatible)."""
    return PipelineConfig(
        enabled=True,
        availability_filter=NullAvailabilityFilter(),
        routing_plugins=[NullRoutingPlugin()],
        aggregator_plugin=DefaultAggregator(),
    )
