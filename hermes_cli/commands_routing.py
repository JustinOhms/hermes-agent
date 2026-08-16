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
from typing import Any, Dict, Optional


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
            cmd_routing_oversight(config, session_id=getattr(args, "session", None))
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


def _print_oversight_ledger(session_id: str, summary: Dict[str, Any]) -> None:
    """Render a durable oversight ledger summary for a past session."""
    import time as _time

    print(f"Oversight — session {session_id}:")
    n = summary.get("reviews", 0)
    turns = summary.get("turns")
    pct = summary.get("pct_of_turns")
    pct_str = f"  ({pct}% of {turns} turns)" if pct is not None else ""
    print(f"  Reviews: {n}{pct_str}")
    if not n:
        print("  (oversight did not fire this session)")
        return
    acts = summary.get("actions") or {}
    mix = "  ".join(f"{k}={v}" for k, v in acts.items() if v) or "none"
    print(f"  Verdicts: {mix}")   # the 'is it working?' signal
    print(
        f"  Duration: {summary.get('total_duration_ms', 0)}ms total, "
        f"{summary.get('avg_duration_ms', 0)}ms avg"
    )
    print(
        f"  Tokens: {summary.get('input_tokens', 0)}+{summary.get('output_tokens', 0)}"
        f"   Est cost: ${(summary.get('est_cost_usd') or 0.0):.4f}"
    )
    recent = summary.get("recent") or []
    if recent:
        print("  Recent:")
        for h in recent[-10:]:
            ts = h.get("ts")
            ts_str = _time.strftime("%H:%M:%S", _time.localtime(ts)) if ts else "?"
            print(
                f"    [{ts_str}] turn {h.get('turn')}: {h.get('action')} "
                f"({h.get('duration_ms', 0)}ms)"
            )


def cmd_routing_oversight(config: Dict[str, Any], session_id: Optional[str] = None) -> None:
    """Show oversight config, or a past session's durable metrics ledger.

    With ``--session <id>`` this reads the persisted per-session oversight ledger
    (count, verdict mix, duration, est cost, %% of turns) so you can look back and
    answer "was oversight actually used + working?". Without it, shows whether
    oversight is enabled, its interval, and the resolved review model.
    """
    if session_id:
        try:
            from hermes_state import SessionDB
            from agent.routing.oversight import load_oversight_ledger

            summary = load_oversight_ledger(SessionDB(), session_id)
        except Exception as e:
            print(f"Could not read oversight ledger: {e}")
            return
        if summary is None:
            print(f"No oversight ledger recorded for session {session_id} "
                  "(oversight disabled, never fired, or unknown session).")
            return
        _print_oversight_ledger(session_id, summary)
        return

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
