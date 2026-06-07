"""Tests for agent/routing/config.py — RoutingConfig loading and defaults."""

from __future__ import annotations

import pytest

from agent.routing.config import (
    ComplexityConfig,
    DeEscalationConfig,
    GraphPosition,
    GraphPositionProfile,
    InteractionModeConfig,
    RoutingConfig,
    load_routing_config,
)


class TestLoadRoutingConfigDefaults:
    def test_empty_dict_returns_disabled_config(self):
        cfg = load_routing_config({})
        assert cfg.enabled is False

    def test_none_falls_back_to_defaults(self):
        # Pass empty dict directly to avoid touching disk
        cfg = load_routing_config({"model": {}})
        assert isinstance(cfg, RoutingConfig)
        assert cfg.enabled is False

    def test_enabled_flag_parsed(self):
        cfg = load_routing_config({"model": {"routing": {"enabled": True}}})
        assert cfg.enabled is True

    def test_enabled_false_explicit(self):
        cfg = load_routing_config({"model": {"routing": {"enabled": False}}})
        assert cfg.enabled is False

    def test_non_dict_routing_section_returns_defaults(self):
        cfg = load_routing_config({"model": {"routing": "bad_value"}})
        assert cfg.enabled is False
        assert cfg.graph == {}


class TestInteractionModeConfigDefaults:
    def test_defaults_when_section_missing(self):
        cfg = load_routing_config({"model": {"routing": {"enabled": True}}})
        im = cfg.interaction_mode
        assert im.idle_threshold_s == 600
        assert im.swap_back_messages == 3
        assert im.swap_back_window_s == 60

    def test_custom_values_parsed(self):
        cfg = load_routing_config({
            "model": {
                "routing": {
                    "enabled": True,
                    "interaction_mode": {
                        "idle_threshold_s": 300,
                        "swap_back_messages": 5,
                        "swap_back_window_s": 120,
                    },
                }
            }
        })
        im = cfg.interaction_mode
        assert im.idle_threshold_s == 300
        assert im.swap_back_messages == 5
        assert im.swap_back_window_s == 120


class TestComplexityConfigDefaults:
    def test_defaults(self):
        cfg = load_routing_config({"model": {"routing": {}}})
        assert cfg.complexity.escalation_threshold == 0.7
        assert cfg.complexity.de_escalation_threshold == 0.2

    def test_custom_thresholds(self):
        cfg = load_routing_config({
            "model": {
                "routing": {
                    "complexity": {
                        "escalation_threshold": 0.8,
                        "de_escalation_threshold": 0.1,
                    }
                }
            }
        })
        assert cfg.complexity.escalation_threshold == 0.8
        assert cfg.complexity.de_escalation_threshold == 0.1


class TestDeEscalationConfig:
    def test_disabled_by_default(self):
        cfg = load_routing_config({"model": {"routing": {}}})
        assert cfg.de_escalation.enabled is False

    def test_can_be_enabled(self):
        cfg = load_routing_config({
            "model": {"routing": {"de_escalation": {"enabled": True}}}
        })
        assert cfg.de_escalation.enabled is True


class TestGraphParsing:
    def test_full_graph_parsed(self):
        cfg = load_routing_config({
            "model": {
                "routing": {
                    "enabled": True,
                    "graph": {
                        "interactive_lower": {
                            "provider": "custom:llm-local",
                            "model": "qwen3-coder",
                            "profile": {
                                "startup_latency_s": 17,
                                "ttft_p50_ms": 200,
                                "generation_tok_s": 33,
                            },
                        },
                        "upper": {
                            "provider": "bedrock",
                            "model": "claude-opus-4",
                            "profile": {
                                "startup_latency_s": 0,
                                "ttft_p50_ms": 2000,
                                "generation_tok_s": 50,
                            },
                        },
                    },
                }
            }
        })
        assert "interactive_lower" in cfg.graph
        assert "upper" in cfg.graph
        pos = cfg.graph["interactive_lower"]
        assert pos.provider == "custom:llm-local"
        assert pos.model == "qwen3-coder"
        assert pos.profile.startup_latency_s == 17.0
        assert pos.profile.ttft_p50_ms == 200.0
        assert pos.profile.generation_tok_s == 33.0

    def test_empty_graph(self):
        cfg = load_routing_config({"model": {"routing": {"graph": {}}}})
        assert cfg.graph == {}

    def test_non_dict_position_skipped(self):
        cfg = load_routing_config({
            "model": {"routing": {"graph": {"bad_pos": "not_a_dict"}}}
        })
        assert "bad_pos" not in cfg.graph

    def test_missing_profile_uses_defaults(self):
        cfg = load_routing_config({
            "model": {
                "routing": {
                    "graph": {
                        "interactive_lower": {
                            "provider": "local",
                            "model": "test-model",
                        }
                    }
                }
            }
        })
        pos = cfg.graph["interactive_lower"]
        assert pos.profile.startup_latency_s == 0.0
        assert pos.profile.ttft_p50_ms == 0.0
