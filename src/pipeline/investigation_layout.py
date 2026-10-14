"""Resolve optimizer state inside a pipeline run envelope."""

from __future__ import annotations

from pathlib import Path


def investigation_workspace(run_envelope: Path) -> Path:
    """Return the dedicated optimizer workspace for a pipeline run."""
    return run_envelope / "investigation"
