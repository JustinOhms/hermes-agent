"""Tests for agent.routing.state — Phase 3b routing state aggregation."""
import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from agent.routing.state import RoutingState, get_routing_state


class TestRoutingState:
    """Test RoutingState dataclass."""

    def test_basic_construction(self):
        state = RoutingState(
            enabled=True,
            current_position="interactive_lower",
            interaction_mode="interactive",
            swap_state="IDLE",
            last_decision=None,
        )
        assert state.enabled is True
        assert state.current_position == "interactive_lower"
        assert state.interaction_mode == "interactive"
        assert state.swap_state == "IDLE"

    def test_defaults(self):
        state = RoutingState(
            enabled=False,
            current_position=None,
            interaction_mode="interactive",
            swap_state="IDLE",
            last_decision=None,
        )
        assert state.drift_alerts == []
        assert state.decision_history == []


class TestGetRoutingState:
    """Test get_routing_state aggregation."""

    def test_disabled_when_config_not_loaded(self):
        agent = MagicMock()
        with patch("agent.routing.config.load_routing_config", side_effect=ImportError):
            state = get_routing_state(agent)
        assert state.enabled is False

    def test_disabled_when_config_says_disabled(self):
        agent = MagicMock()
        mock_config = MagicMock()
        mock_config.enabled = False
        with patch("agent.routing.config.load_routing_config", return_value=mock_config):
            state = get_routing_state(agent)
        assert state.enabled is False

    def test_enabled_reads_swap_manager(self):
        agent = MagicMock()
        agent._routing_swap_manager = MagicMock()
        agent._routing_swap_manager.current_position = "upper"
        agent._routing_swap_manager.state = MagicMock()
        agent._routing_swap_manager.state.name = "IDLE"
        agent._routing_mode_detector = None
        agent._routing_decision_history = None
        agent._routing_drift_detector = None

        mock_config = MagicMock()
        mock_config.enabled = True
        with patch("agent.routing.config.load_routing_config", return_value=mock_config):
            state = get_routing_state(agent)
        assert state.enabled is True
        assert state.current_position == "upper"
        assert state.swap_state == "IDLE"

    def test_decision_history_from_deque(self):
        agent = MagicMock()
        agent._routing_swap_manager = None
        agent._routing_mode_detector = None
        agent._routing_drift_detector = None

        decision1 = MagicMock()
        decision1._timestamp = time.time()
        decision2 = MagicMock()
        decision2._timestamp = time.time()
        agent._routing_decision_history = deque([decision1, decision2], maxlen=20)

        mock_config = MagicMock()
        mock_config.enabled = True
        with patch("agent.routing.config.load_routing_config", return_value=mock_config):
            state = get_routing_state(agent)
        assert len(state.decision_history) == 2
        assert state.last_decision is decision2

    def test_none_agent_attributes(self):
        """Agent with no routing attributes — should not crash."""
        agent = object()  # no attributes at all
        mock_config = MagicMock()
        mock_config.enabled = True
        with patch("agent.routing.config.load_routing_config", return_value=mock_config):
            state = get_routing_state(agent)
        assert state.enabled is True
        assert state.current_position is None
        assert state.swap_state == "IDLE"

    def test_never_raises(self):
        """get_routing_state should never raise, regardless of input."""
        # Even with None
        state = get_routing_state(None)
        assert isinstance(state, RoutingState)
