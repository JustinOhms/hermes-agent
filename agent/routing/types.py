"""Shared types for the model fingerprint / catalog system.

These types define the contract between the routing system and the model
identity layer.  They ship with the routing PR and are stable — the catalog
PR adds richer resolution logic without changing these shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelCapabilities:
    """Model capabilities extracted from metadata sources.

    Populated by the catalog module when available; otherwise defaults to
    unknown (zeros / False).  Consumers should treat zero/False as "unknown",
    not "absent".
    """
    context_length: int = 0
    max_completion_tokens: int = 0
    has_vision: bool = False
    has_tool_calling: bool = False
    has_structured_output: bool = False
    has_reasoning: bool = False
    has_attachment: bool = False
    has_knowledge_cutoff: Optional[str] = None
    pricing: Dict[str, Any] = field(default_factory=dict)  # input/output $/1M tokens
    architecture: Dict[str, Any] = field(default_factory=dict)  # model arch details

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON-RPC / TUI events."""
        return {
            "context_length": self.context_length,
            "max_completion_tokens": self.max_completion_tokens,
            "has_vision": self.has_vision,
            "has_tool_calling": self.has_tool_calling,
            "has_structured_output": self.has_structured_output,
            "has_reasoning": self.has_reasoning,
            "has_attachment": self.has_attachment,
            "has_knowledge_cutoff": self.has_knowledge_cutoff,
            "pricing": self.pricing,
            "architecture": self.architecture,
        }


@dataclass
class RoutingGraphContext:
    """Summary of the routing graph for model self-awareness.

    Tells the model where it sits in the routing graph and what other
    positions exist, so it can accurately answer "what model are you"
    vs. "what's the configured primary/upper".
    """
    active_position: str = ""                          # e.g. "interactive_lower"
    positions: Dict[str, str] = field(default_factory=dict)  # name → "display_name (provider/model)"
    configured_upper: str = ""                         # display string for the upper model


@dataclass
class ModelFingerprint:
    """Resolved identity of the model processing the current request."""
    model_id: str           # Exact model string sent to the API (e.g. "us.anthropic.claude-opus-4-6-v1")
    provider: str           # Provider name (e.g. "bedrock", "custom:llm-local")
    display_name: str       # Human-friendly name (e.g. "Claude Opus 4", "Qwen3 Coder Next")
    base_url: str           # API endpoint being called (masked for display)
    position: str           # Routing graph position if routing is active (e.g. "upper", "interactive_lower")
    is_local: bool          # Whether this is a local model on :58080
    family: str             # Model family for capability hints (e.g. "claude", "qwen", "gpt")
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    routing_graph: Optional[RoutingGraphContext] = None  # populated when routing is active

    def to_prompt_line(self) -> str:
        """Format for injection into the ephemeral system prompt."""
        parts = [f"Model: {self.model_id}"]
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        if self.display_name and self.display_name != self.model_id:
            parts.append(f"Display name: {self.display_name}")
        if self.position:
            parts.append(f"Routing position: {self.position}")
        # Add key capabilities (only when catalog has populated them)
        caps = self.capabilities
        if caps.has_reasoning:
            parts.append("Reasoning: enabled")
        if caps.has_vision:
            parts.append("Vision: enabled")
        if caps.has_tool_calling:
            parts.append("Tool calling: enabled")
        if caps.has_structured_output:
            parts.append("Structured output: enabled")
        if caps.context_length > 0:
            parts.append(f"Context: {caps.context_length:,} tokens")
        if caps.has_knowledge_cutoff:
            parts.append(f"Knowledge cutoff: {caps.has_knowledge_cutoff}")
        # Routing graph context — tells the model about the full routing topology
        if self.routing_graph and self.routing_graph.positions:
            rg = self.routing_graph
            parts.append("")  # blank line separator
            parts.append("Routing graph (model routing is active):")
            for pos_name, pos_desc in rg.positions.items():
                marker = " ← you are here" if pos_name == rg.active_position else ""
                parts.append(f"  {pos_name}: {pos_desc}{marker}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON-RPC / TUI events."""
        d: Dict[str, Any] = {
            "model_id": self.model_id,
            "provider": self.provider,
            "display_name": self.display_name,
            "base_url": self.base_url,
            "position": self.position,
            "is_local": self.is_local,
            "family": self.family,
            "capabilities": self.capabilities.to_dict(),
        }
        if self.routing_graph:
            d["routing_graph"] = {
                "active_position": self.routing_graph.active_position,
                "positions": self.routing_graph.positions,
                "configured_upper": self.routing_graph.configured_upper,
            }
        return d


@dataclass
class FingerprintEntry:
    """Static metadata for a known model, used to populate display_name and family.

    The builtin catalog uses these with just display_name + family.
    The catalog module enriches them with canonical_id, urls, and capabilities.
    """
    display_name: str
    family: str
    canonical_id: Optional[str] = None  # Canonical model ID (provider/model format)
    urls: List[str] = field(default_factory=list)  # Known API endpoint URLs
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
