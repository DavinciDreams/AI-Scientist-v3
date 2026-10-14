"""Validation helpers for generated idea artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.orchestrator.idea_bundle import (
    IdeaBundleValidationError,
    normalize_idea_bundle,
    normalize_target_rubrics_payload,
    without_target_rubric_verification,
)

from .rubric_checks import MAX_HARNESS_RUBRICS, rubric_metric_check


class IdeaArtifactsError(ValueError):
    """Generated idea artifacts are missing, invalid, or inconsistent."""

    def __init__(self, reason: str, *, path: Path | None = None, detail: str = "") -> None:
        self.reason = reason
        self.path = path
        self.detail = detail
        super().__init__(detail or reason)


@dataclass(frozen=True)
class IdeaArtifacts:
    bundle: dict[str, Any]
    target_rubrics: list[dict[str, Any]]


def load_idea_artifacts(
    workspace: Path,
    cycle_index: int,
    read_json: Callable[[Path], dict[str, Any] | None],
    *,
    require_harness_verification: bool = False,
    harness_incumbent_score: float | None = None,
) -> IdeaArtifacts:
    """Load and cross-check the required idea-generation outputs."""
    bundle_path = workspace / "idea_bundle.json"
    bundle_raw = read_json(bundle_path)
    if bundle_raw is None:
        raise IdeaArtifactsError(
            "missing_or_invalid_idea_bundle",
            path=bundle_path,
        )
    if not require_harness_verification:
        bundle_raw = without_target_rubric_verification(bundle_raw)
    try:
        bundle = normalize_idea_bundle(bundle_raw, cycle_index)
    except IdeaBundleValidationError as exc:
        raise IdeaArtifactsError(
            "invalid_idea_bundle",
            path=bundle_path,
            detail=str(exc),
        ) from exc

    rubrics_path = workspace / "target_rubrics.json"
    if not rubrics_path.exists():
        raise IdeaArtifactsError("missing_target_rubrics", path=rubrics_path)
    try:
        rubrics_raw = json.loads(rubrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdeaArtifactsError(
            "invalid_target_rubrics_json",
            path=rubrics_path,
            detail=str(exc),
        ) from exc
    if not require_harness_verification:
        rubrics_raw = without_target_rubric_verification(rubrics_raw)
    try:
        target_rubrics = normalize_target_rubrics_payload(rubrics_raw)
    except IdeaBundleValidationError as exc:
        raise IdeaArtifactsError(
            "invalid_target_rubrics_payload",
            path=rubrics_path,
            detail=str(exc),
        ) from exc
    if target_rubrics != bundle["target_rubrics"]:
        raise IdeaArtifactsError("target_rubrics_mismatch", path=rubrics_path)
    if require_harness_verification:
        target_rubrics = _replace_metric_rubrics(
            target_rubrics,
            harness_incumbent_score,
        )
        if not any(rubric.get("type") == "implementation" for rubric in target_rubrics):
            raise IdeaArtifactsError(
                "missing_harness_implementation_rubric",
                path=rubrics_path,
                detail="harness mode requires at least one implementation rubric",
            )
        bundle["target_rubrics"] = target_rubrics
        if len(target_rubrics) > MAX_HARNESS_RUBRICS:
            raise IdeaArtifactsError(
                "too_many_harness_rubrics",
                path=rubrics_path,
                detail=f"harness mode supports at most {MAX_HARNESS_RUBRICS} rubrics",
            )
        for rubric in target_rubrics:
            rubric_type = rubric.get("type")
            if rubric_type == "metric":
                if rubric_metric_check(rubric) is None:
                    raise IdeaArtifactsError(
                        "missing_harness_metric_threshold",
                        path=rubrics_path,
                        detail=(
                            f"metric rubric {rubric.get('id')!r} requires a numeric metric "
                            "threshold using one direct evaluator-emitted metric key"
                        ),
                    )
                continue
            if rubric_type != "implementation":
                continue
            verification = rubric.get("verification")
            if not isinstance(verification, dict) or verification.get("type") != "source_contains":
                raise IdeaArtifactsError(
                    "missing_harness_implementation_verification",
                    path=rubrics_path,
                    detail=f"implementation rubric {rubric.get('id')!r} requires source_contains verification",
                )
    return IdeaArtifacts(bundle=bundle, target_rubrics=target_rubrics)


def _replace_metric_rubrics(
    target_rubrics: list[dict[str, Any]],
    incumbent_score: float | None,
) -> list[dict[str, Any]]:
    implementation_rubrics = [
        rubric for rubric in target_rubrics if rubric.get("type") == "implementation"
    ]
    if incumbent_score is None:
        return implementation_rubrics
    if isinstance(incumbent_score, bool) or not math.isfinite(float(incumbent_score)):
        raise IdeaArtifactsError(
            "invalid_harness_incumbent_score",
            detail="harness incumbent combined_score must be finite",
        )
    threshold = float(incumbent_score)
    threshold_text = repr(threshold)
    score_gate = {
        "id": "incumbent_score_improvement",
        "description": f"combined_score > {threshold_text}",
        "type": "metric",
        "metric": "combined_score",
        "operator": ">",
        "threshold": threshold,
    }
    return [score_gate, *implementation_rubrics]
