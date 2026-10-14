"""Shared interpretation of structured paper progress events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .terminal_ui import terminal_text

PAPER_GATE_LABELS = {
    "preparation": "Preparation",
    "strategy": "Strategy",
    "draft": "Draft",
    "references": "References",
    "figures": "Figures",
    "validation": "Validation",
    "evidence_audit": "Evidence Audit",
    "review": "Review",
    "repair": "Repair",
    "finalization": "Finalization",
}


def paper_event_fields(fields: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return bridge fields from either a direct or wrapper event."""
    nested = fields.get("paper_fields")
    return nested if isinstance(nested, Mapping) else fields


def is_paper_heartbeat(state: Any, fields: Mapping[str, Any]) -> bool:
    """Identify liveness records that must not look like new gate starts."""
    return state == "started" and paper_event_fields(fields).get("heartbeat") is True


def paper_gate_name(gate: Any) -> str:
    """Build a terminal-safe display name for a paper gate."""
    gate_key = str(gate)
    return PAPER_GATE_LABELS.get(
        gate_key,
        terminal_text(gate_key).replace("_", " ").title(),
    )


def paper_gate_qualifier(fields: Mapping[str, Any]) -> str:
    """Describe the phase or round that distinguishes a gate lifecycle."""
    event_fields = paper_event_fields(fields)
    qualifiers: list[str] = []
    phase = event_fields.get("phase")
    if isinstance(phase, str) and phase:
        qualifiers.append(terminal_text(phase, limit=64).replace("_", " "))
    round_number = event_fields.get("round")
    if isinstance(round_number, int) and not isinstance(round_number, bool):
        qualifiers.append(f"round={terminal_text(round_number, limit=32)}")
    return ", ".join(qualifiers)


def paper_gate_label(gate: Any, fields: Mapping[str, Any]) -> str:
    """Build a gate label with lifecycle qualifiers for progress displays."""
    name = paper_gate_name(gate)
    qualifier = paper_gate_qualifier(fields)
    return f"{name} ({qualifier})" if qualifier else name
