"""Attach validated observations from known evaluation directories to paper data."""

from pathlib import Path
import re
from typing import Any

from src.investigation.evaluation_evidence import evidence_metadata, load_evaluation_evidence


def paper_evaluation_evidence(root: Path, program: Path) -> dict[str, Any]:
    """Load optional scientific data without following directory symlinks."""
    try:
        relative = program.relative_to(root)
    except ValueError as exc:
        raise ValueError("Evaluation evidence directory escapes its source root") from exc
    current = root
    for part in ("", *relative.parts):
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Evaluation evidence directory contains symlink: {current}")
    metadata = evidence_metadata(program)
    payload = load_evaluation_evidence(program)
    if payload is not None:
        metadata["episodes"] = payload["episodes"]
    return metadata


def ablation_evaluation_evidence(
    root: Path, variant_id: str, program_value: str | None,
) -> dict[str, Any]:
    """Resolve CSV program labels only inside the corresponding variant workspace."""
    unavailable = {"status": "unavailable"}
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", variant_id) or not program_value:
        return unavailable
    label = Path(program_value)
    if not re.fullmatch(r"program_\d+", label.name):
        return unavailable
    program = root / "variants" / variant_id / "task_bundle" / label.name
    if label != Path(label.name) and label != program:
        return unavailable
    return paper_evaluation_evidence(root, program)
