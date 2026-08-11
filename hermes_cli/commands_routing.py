"""hermes routing — model routing status and control.

Subcommands:
  hermes routing status    Show routing position, mode, swap state, last decision, drift
  hermes routing graph     Show configured positions with active marker
  hermes routing swap <pos>  Force swap to a position (bypasses cost filter)
  hermes routing mode <mode>  Override mode detection (interactive|autonomous|auto)
  hermes routing history   Last 10 routing decisions with timestamps
  hermes routing oversight Show oversight decisions

Storage: Routing state in agent/routing/state.py
"""

from __future__ import annotations

import argparse
from typing import Any, Dict


def cmd_routing(args: argparse.Namespace) -> None:
    """Handle /routing command subdispatch."""
    from hermes_cli.config import load_config

    config = load_config()

    # Get the actual subcommand name (it's stored in routing_command)
    subcommand = getattr(args, "routing_command", None)

    if subcommand:
        if subcommand == "status":
            cmd_routing_status(config)
        elif subcommand == "graph":
            cmd_routing_graph(config)
        elif subcommand == "swap":
            cmd_routing_swap(config, args.position)
        elif subcommand == "mode":
            cmd_routing_mode(config, args.mode)
        elif subcommand == "history":
            cmd_routing_history(config)
        elif subcommand == "oversight":
            cmd_routing_oversight(config)
        else:
            print(f"Unknown routing subcommand: {subcommand}")
            print("Usage: hermes routing [status|graph|swap|mode|history|oversight]")
    else:
        # No subcommand - show status by default
        cmd_routing_status(config)


def cmd_routing_status(config: Dict[str, Any]) -> None:
    """Show routing position, mode, swap state, last decision, drift."""
    print("Routing status:")
    print("  Position: <current position>")
    print("  Mode: <current mode>")
    print("  Swap state: <active/inactive>")
    print("  Last decision: <timestamp> - <decision>")
    print("  Drift: <drift value>")


def cmd_routing_graph(config: Dict[str, Any]) -> None:
    """Show configured positions with active marker."""
    print("Routing positions:")
    print("  0. <position> [ACTIVE]")
    print("  1. <position>")
    print("  2. <position>")


def cmd_routing_swap(config: Dict[str, Any], position: int) -> None:
    """Force swap to a position."""
    print(f"Swapping to position {position}...")


def cmd_routing_mode(config: Dict[str, Any], mode: str) -> None:
    """Override mode detection."""
    print(f"Setting mode to {mode}...")


def cmd_routing_history(config: Dict[str, Any]) -> None:
    """Show last 10 routing decisions."""
    print("Routing history (last 10 decisions):")


def cmd_routing_oversight(config: Dict[str, Any]) -> None:
    """Show oversight configuration + the resolved upper review model.

    The one-shot CLI has no live session, so per-session review history isn't
    available here — use ``/routing oversight`` inside a TUI/desktop session for
    live decisions. This shows whether oversight is enabled, its interval, and
    which model would perform reviews.
    """
    try:
        from agent.routing.oversight import load_oversight_config
        ovc = load_oversight_config(config)
    except Exception as e:
        print(f"Oversight unavailable: {e}")
        return

    print("Oversight:")
    print(f"  Enabled: {ovc.enabled}")
    if not ovc.enabled:
        print("  (set model.routing.oversight.enabled: true to activate)")
        return
    print(f"  Every: {ovc.every_n_turns} turns")
    print(f"  Max reviews/session: {ovc.max_reviews_per_session}")

    model, provider = ovc.model, ovc.provider
    if not model or not provider:
        try:
            from agent.routing.config import load_routing_config
            rcfg = load_routing_config(config)
            top = rcfg.top_position() if rcfg else None
            pos = rcfg.graph.get(top) if (rcfg and top) else None
            if pos is not None:
                model = model or pos.model
                provider = provider or pos.provider
        except Exception:
            pass
    if model and provider:
        print(f"  Review model: {model} ({provider})")
    else:
        print(
            "  Review model: UNSET — configure model.routing.oversight.model "
            "or a routing graph top tier (reviews will be skipped)"
        )
    print("  (live per-session decisions: use /routing oversight in a session)")
