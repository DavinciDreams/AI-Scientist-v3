"""Stage 6: Aggregate and publish valid post-hoc analyses."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import write_json
from .models import AnalysisRunResult, AnalysisSpec


def aggregate_and_publish_valid_analyses(
    *,
    output_dir: Path,
    analysis_specs: list[AnalysisSpec],
    results: list[AnalysisRunResult],
    checkpoint_mode: str = "required",
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    results_by_id = {result.analysis_id: result for result in results}

    valid_entries: list[dict[str, Any]] = []
    rejected_entries: list[dict[str, Any]] = []

    for spec in analysis_specs:
        result = results_by_id.get(spec.analysis_id)
        if result is None:
            rejected_entries.append(
                {
                    "analysis_id": spec.analysis_id,
                    "title": spec.title,
                    "family": spec.family,
                    "reason": "not_executed",
                }
            )
            continue
        if (
            result.status != "completed"
            or not result.selected_attempt
            or not result.rubric_passed
            or not result.required_outputs_present
            or result.guardrail_passed is False
            or (
                checkpoint_mode == "required"
                and not _selected_attempt_has_checkpoint(result)
            )
        ):
            rejected_entries.append(
                {
                    "analysis_id": spec.analysis_id,
                    "title": spec.title,
                    "family": spec.family,
                    "reason": _rejection_reason(result, checkpoint_mode),
                }
            )
            continue

        attempt_dir = spec.analysis_dir / result.selected_attempt
        valid_entries.append(
            {
                "analysis_id": spec.analysis_id,
                "title": spec.title,
                "family": spec.family,
                "analysis_dir": str(spec.analysis_dir),
                "selected_attempt": result.selected_attempt,
                "selected_attempt_dir": str(attempt_dir),
                "status": result.status,
                "attempt_count": result.attempt_count,
                "analysis_manifest_path": str(attempt_dir / "analysis_manifest.json"),
                "analysis_summary_path": str(attempt_dir / "analysis_summary.md"),
                "interestingness_rubric_path": str(attempt_dir / "interestingness_rubric.json"),
            }
        )

    publication = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "delegated_to_paper_agent": True,
        "valid": valid_entries,
        "rejected": rejected_entries,
        "selected_analysis_id": None,
    }
    write_json(output_dir / "ranking.json", publication)

    latest_manifest_path: str | None = None
    if valid_entries:
        latest_manifest = {
            "manifest_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "delegated_to_paper_agent": True,
            "selected_analysis_id": None,
            "valid_analyses": valid_entries,
            "rejected_analyses": rejected_entries,
        }
        manifest_path = output_dir / "latest_manifest.json"
        write_json(manifest_path, latest_manifest)
        _publish_stable_pointer(output_dir, manifest_path)
        latest_manifest_path = str(manifest_path)
    else:
        _remove_stale_manifests(output_dir)

    summary = {
        "status": "completed" if valid_entries else "no_valid_analyses",
        "selected_analysis_id": None,
        "delegated_to_paper_agent": True,
        "total_analyses": len(analysis_specs),
        "total_valid": len(valid_entries),
        "total_rejected": len(rejected_entries),
        "ranking_path": str(output_dir / "ranking.json"),
        "latest_manifest_path": latest_manifest_path,
    }
    write_json(output_dir / "pipeline_summary.json", summary)
    return publication


def _rejection_reason(result: AnalysisRunResult, checkpoint_mode: str) -> str:
    if result.status == "timeout":
        return "timeout"
    if result.selected_attempt is None:
        return "no_successful_attempt"
    if not result.rubric_passed:
        return "rubric_failed"
    if not result.required_outputs_present:
        return "missing_required_outputs"
    if result.guardrail_passed is False:
        return "guardrail_failed"
    if checkpoint_mode == "required" and not _selected_attempt_has_checkpoint(result):
        return "checkpoint_required"
    return "invalid"


def _selected_attempt_has_checkpoint(result: AnalysisRunResult) -> bool:
    return any(
        attempt.attempt_id == result.selected_attempt and attempt.checkpoint_validated
        for attempt in result.attempts
    )


def _publish_stable_pointer(output_dir: Path, manifest_path: Path) -> None:
    stable_dir = output_dir.parent
    stable_dir.mkdir(parents=True, exist_ok=True)
    stable_path = stable_dir / "latest_manifest.json"
    shutil.copy2(manifest_path, stable_path)


def _remove_stale_manifests(output_dir: Path) -> None:
    run_manifest = output_dir / "latest_manifest.json"
    run_manifest.unlink(missing_ok=True)
    stable_manifest = output_dir.parent / "latest_manifest.json"
    stable_payload = None
    if stable_manifest.is_file():
        try:
            import json

            loaded = json.loads(stable_manifest.read_text(encoding="utf-8"))
            stable_payload = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError):
            stable_payload = None
    if stable_payload and Path(str(stable_payload.get("output_dir", ""))).resolve() == output_dir:
        stable_manifest.unlink(missing_ok=True)
