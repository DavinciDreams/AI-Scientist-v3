"""Structured progress events for the post-hoc analysis pipeline."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.investigation.runtime import format_elapsed, log_event

from .models import AnalysisCandidate, AnalysisRunResult, AnalysisSelection, AnalysisSpec

STAGE_LABELS = {
    0: "Extract bundle",
    1: "Bootstrap analysis-ready bundle",
    2: "Generate analysis candidates",
    3: "Select analysis candidates",
    4: "Materialize analysis workspaces",
    5: "Execute analysis agents",
    6: "Aggregate and publish valid analyses",
}


@contextmanager
def stage_event(stage: int, *, mode: str) -> Iterator[dict[str, Any]]:
    """Emit a paired stage event, including failures raised by the stage."""
    started_at = time.monotonic()
    label = f"Stage {stage}: {STAGE_LABELS[stage]}"
    log_event(
        "stage_start",
        stage=stage,
        label=label,
        mode=mode,
        status="started",
        current=stage + 1,
        total=len(STAGE_LABELS),
    )
    fields: dict[str, Any] = {}
    try:
        yield fields
    except BaseException as exc:
        fields.setdefault("error_type", type(exc).__name__)
        fields["status"] = "failed"
        raise
    finally:
        status = str(fields.pop("status", "completed"))
        log_event(
            "stage_end",
            stage=stage,
            label=label,
            mode=mode,
            status=status,
            elapsed=format_elapsed(time.monotonic() - started_at),
            **fields,
        )


def emit_pipeline_summary(
    *,
    status: str,
    stages: set[int],
    candidates: list[AnalysisCandidate],
    selection: AnalysisSelection,
    analysis_specs: list[AnalysisSpec],
    results: list[AnalysisRunResult],
    output_dir: Path,
    started_at: float,
) -> None:
    """Emit aggregate counts at post-hoc pipeline completion."""
    fields = {
        "status": status,
        "stages": sorted(stages),
        "candidate_count": len(candidates),
        "selected_count": len(selection.accepted),
        "analysis_count": len(analysis_specs),
        "completed_count": sum(result.status == "completed" for result in results),
        "rubric_passed_count": sum(result.rubric_passed for result in results),
        "output_dir": output_dir,
    }
    log_event("pipeline_summary", **fields)
    log_event(
        "pipeline_end",
        **fields,
        elapsed=format_elapsed(time.monotonic() - started_at),
    )
