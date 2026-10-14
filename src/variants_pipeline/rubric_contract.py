"""Rubric contract builder and validator for the variants pipeline.

Every variant directory must satisfy the same 9 rubric families.  This module
builds the ``rubric_contract.json`` and ``target_rubrics.json`` artifacts and
provides validation helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.investigation.safe_staging import read_text as safe_read_text

from .constants import REQUIRED_RUBRIC_IDS

# Descriptions for each rubric family.
_RUBRIC_DESCRIPTIONS: dict[str, str] = {
    "analysis_files_present": (
        "All required analysis files (run_summary.csv, metric_values.csv, "
        "timeseries.csv) exist under analysis/."
    ),
    "run_summary_schema_valid": (
        "run_summary.csv exists and contains all required columns."
    ),
    "metric_values_schema_valid": (
        "metric_values.csv exists and contains all required columns."
    ),
    "timeseries_schema_valid": (
        "timeseries.csv exists and contains all required columns, even if "
        "the file contains only headers."
    ),
    "metrics_consistent_with_metrics_json": (
        "combined_score and primary metric in run_summary.csv agree with "
        "the values in metrics.json produced by the evaluator."
    ),
    "correctness_recorded": (
        "The 'correct' field is present in run_summary.csv and matches "
        "correct.json."
    ),
    "variant_intent_preserved": (
        "The implementation matches the declared baseline/ablation intent "
        "described in variant_spec.json."
    ),
    "single_change_scope": (
        "For ablations and extra baselines, only the declared comparison "
        "axis was changed relative to the control."
    ),
    "paper_analysis_ready": (
        "Analysis files contain the declared evaluator metrics, correctness, and "
        "variant identity. State which evidence is available and missing in the "
        "rubric evidence, including per-seed observations, geometry, runtime, and "
        "timeseries. A header-only timeseries is valid; do not require undeclared "
        "diagnostics or infer support for claims whose evidence is unavailable."
    ),
}

CONTRACT_VERSION: int = 1
_MAX_RUBRIC_STATUS_BYTES = 1024 * 1024
_MAX_RUBRICS = 256
_MAX_EVIDENCE_LENGTH = 4096
_MAX_RECOMMENDATION_LENGTH = 4096


def build_rubric_contract(
    variant_id: str,
    variant_kind: str,
    base_program: str = "best",
) -> dict[str, Any]:
    """Build the ``rubric_contract.json`` for a single variant.

    The contract is identical across variants except for identifying metadata.
    External references require adaptation audits. The reserved original-seed
    baseline is evaluated unchanged and does not require external documentation.
    """
    rubrics = []
    for rubric_id in REQUIRED_RUBRIC_IDS:
        rubrics.append(
            {
                "id": rubric_id,
                "description": _RUBRIC_DESCRIPTIONS[rubric_id],
                "type": "structural" if "schema" in rubric_id or "files" in rubric_id else "semantic",
            }
        )

    if base_program == "reference" and variant_id != "baseline_reference":
        rubrics.append(
            {
                "id": "hyperparameter_audit_complete",
                "description": (
                    "A hyperparameter_comparison.md file was created listing all "
                    "hyperparameters from both the reference and control programs, "
                    "with each difference categorized as 'align' or 'preserve' "
                    "with reasoning."
                ),
                "type": "semantic",
            }
        )
        rubrics.append(
            {
                "id": "reference_conditions_aligned",
                "description": (
                    "Experimental conditions (e.g. training budget) are aligned "
                    "with the control program so the reference implementation "
                    "has no unfair advantage or disadvantage on factors outside "
                    "its core contribution."
                ),
                "type": "semantic",
            }
        )
        rubrics.append(
            {
                "id": "reference_performance_check",
                "description": (
                    "Compare the adapted program's performance against the "
                    "expected performance documented in program_0/ (e.g. README). "
                    "If there is a large degradation, investigate which adaptation "
                    "changes caused it and revert any that are not strictly "
                    "required for evaluate.py to run."
                ),
                "type": "semantic",
            }
        )

    return {
        "contract_version": CONTRACT_VERSION,
        "variant_id": variant_id,
        "variant_kind": variant_kind,
        "rubrics": rubrics,
    }


def build_target_rubrics(
    rubric_contract: dict[str, Any],
) -> list[dict[str, str]]:
    """Extract the ``target_rubrics.json`` list from a rubric contract."""
    return [
        {"id": r["id"], "description": r["description"]}
        for r in rubric_contract["rubrics"]
    ]


def validate_variant_rubric_status(
    rubric_status: dict[str, Any] | None,
    variant_id: str,
    required_rubric_ids: set[str] | None = None,
) -> tuple[bool, str | None]:
    """Validate that ``rubric_status.json`` covers all required rubric families.

    Returns ``(passed, error_message)``.  *error_message* is ``None`` on
    success.
    """
    if rubric_status is None:
        return False, "missing or invalid rubric_status.json"

    # Required top-level keys.
    for key in ("variant_id", "overall_pass", "rubrics"):
        if key not in rubric_status:
            return False, f"rubric_status missing key: {key}"

    if rubric_status.get("variant_id") != variant_id:
        return False, (
            f"variant_id mismatch: expected {variant_id!r}, "
            f"got {rubric_status.get('variant_id')!r}"
        )

    required = (
        set(required_rubric_ids)
        if required_rubric_ids is not None
        else set(REQUIRED_RUBRIC_IDS)
    )
    rubrics_by_id, error = _validated_rubrics(rubric_status.get("rubrics"), required)
    if error is not None:
        return False, error

    # Check overall_pass consistency.
    assert rubrics_by_id is not None
    all_pass = all(rubrics_by_id[rubric_id]["pass"] for rubric_id in required)
    declared_pass = rubric_status.get("overall_pass")
    if not isinstance(declared_pass, bool):
        return False, "rubric_status.overall_pass must be boolean"
    if declared_pass != all_pass:
        return False, "rubric_status.overall_pass does not match required rubrics"

    return True, None


def canonicalize_variant_rubric_status(
    rubric_status: dict[str, Any] | None,
    *,
    variant_id: str,
    attempt_id: str,
    idea_id: str,
    required_rubric_ids: list[str],
    combined_score: Any,
    correct: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Build a host-owned status from bounded agent rubric judgments."""
    if rubric_status is None:
        return None, "missing or invalid rubric_status.json"
    required = set(required_rubric_ids)
    rubrics_by_id, error = _validated_rubrics(rubric_status.get("rubrics"), required)
    if error is not None:
        return None, error
    recommendation = rubric_status.get("next_recommendation", "")
    if not isinstance(recommendation, str):
        return None, "rubric_status.next_recommendation must be a string"
    if len(recommendation) > _MAX_RECOMMENDATION_LENGTH:
        return None, "rubric_status.next_recommendation exceeds the length limit"
    assert rubrics_by_id is not None
    canonical_rubrics = [rubrics_by_id[rubric_id] for rubric_id in required_rubric_ids]
    return (
        {
            "variant_id": variant_id,
            "attempt_id": attempt_id,
            "idea_id": idea_id,
            "overall_pass": correct
            and all(entry["pass"] for entry in canonical_rubrics),
            "rubrics": canonical_rubrics,
            "metrics": {
                "combined_score": combined_score,
                "correct": correct,
            },
            "next_recommendation": recommendation,
        },
        None,
    )


def _validated_rubrics(
    rubrics: Any,
    required: set[str],
) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    if not isinstance(rubrics, list):
        return None, "rubric_status.rubrics must be a list"
    if len(rubrics) > _MAX_RUBRICS:
        return None, "rubric_status.rubrics exceeds the count limit"
    by_id: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(rubrics):
        if not isinstance(entry, dict):
            return None, f"rubric entry at index {index} must be an object"
        rubric_id = entry.get("id")
        if not isinstance(rubric_id, str):
            return None, f"rubric entry at index {index} is missing a string id"
        if rubric_id in by_id:
            return None, f"rubric_status contains a duplicate rubric id at index {index}"
        passed = entry.get("pass")
        if not isinstance(passed, bool):
            return None, f"rubric entry pass at index {index} must be boolean"
        evidence = entry.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            return None, f"rubric entry evidence at index {index} must be non-empty"
        if len(evidence) > _MAX_EVIDENCE_LENGTH:
            return None, f"rubric entry evidence at index {index} exceeds the length limit"
        by_id[rubric_id] = {
            "id": rubric_id,
            "pass": passed,
            "evidence": evidence.strip(),
        }
    found = set(by_id)
    missing = required - found
    if missing:
        return None, f"rubric_status missing rubric ids: {sorted(missing)}"
    unexpected = found - required
    if unexpected:
        return None, f"rubric_status contains {len(unexpected)} unexpected rubric ids"
    return by_id, None


def load_rubric_status(path: Path) -> dict[str, Any] | None:
    """Load ``rubric_status.json`` from *path*, returning ``None`` on failure."""
    try:
        raw = safe_read_text(
            path.parent,
            path.name,
            maximum_bytes=_MAX_RUBRIC_STATUS_BYTES,
        )
        data = json.loads(raw)
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
