"""Rubric parsing and validation helpers for attempt outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RubricStatusValidation:
    valid: bool
    overall_pass: bool
    error: str | None
    status: dict[str, Any] | None


def load_json_dict(path: Path) -> dict[str, Any] | None:
    """Load json dict."""
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_target_rubric_ids(rubrics_data: Any) -> set[str]:
    """Extract target rubric ids."""
    items: list[Any]
    if isinstance(rubrics_data, list):
        items = rubrics_data
    elif isinstance(rubrics_data, dict) and isinstance(rubrics_data.get("target_rubrics"), list):
        items = rubrics_data["target_rubrics"]
    else:
        items = []
    ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        rubric_id = item.get("id")
        if isinstance(rubric_id, str) and rubric_id.strip():
            ids.add(rubric_id)
    return ids


def validate_rubric_status(
    status: dict[str, Any] | None,
    target_rubric_ids: set[str],
    expected_attempt_id: str,
    expected_idea_id: str,
) -> RubricStatusValidation:
    """Validate rubric status."""
    if status is None:
        return RubricStatusValidation(
            valid=False,
            overall_pass=False,
            error="missing or invalid rubric_status.json",
            status=None,
        )

    required = ("attempt_id", "idea_id", "overall_pass", "rubrics")
    for key in required:
        if key not in status:
            return RubricStatusValidation(
                valid=False,
                overall_pass=False,
                error=f"rubric_status missing key: {key}",
                status=status,
            )

    if status.get("attempt_id") != expected_attempt_id:
        return RubricStatusValidation(
            valid=False,
            overall_pass=False,
            error=f"attempt_id mismatch (expected {expected_attempt_id})",
            status=status,
        )
    if status.get("idea_id") != expected_idea_id:
        return RubricStatusValidation(
            valid=False,
            overall_pass=False,
            error=f"idea_id mismatch (expected {expected_idea_id})",
            status=status,
        )
    if not isinstance(status.get("overall_pass"), bool):
        return RubricStatusValidation(
            valid=False,
            overall_pass=False,
            error="overall_pass must be boolean",
            status=status,
        )

    rubrics = status.get("rubrics")
    if not isinstance(rubrics, list):
        return RubricStatusValidation(
            valid=False,
            overall_pass=False,
            error="rubrics must be a list",
            status=status,
        )

    seen_ids: set[str] = set()
    all_pass = True
    for idx, item in enumerate(rubrics):
        if not isinstance(item, dict):
            return RubricStatusValidation(
                valid=False,
                overall_pass=False,
                error=f"rubrics[{idx}] must be object",
                status=status,
            )
        rubric_id = item.get("id")
        if not isinstance(rubric_id, str) or not rubric_id.strip():
            return RubricStatusValidation(
                valid=False,
                overall_pass=False,
                error=f"rubrics[{idx}].id must be non-empty string",
                status=status,
            )
        passed = item.get("pass")
        if not isinstance(passed, bool):
            return RubricStatusValidation(
                valid=False,
                overall_pass=False,
                error=f"rubrics[{idx}].pass must be boolean",
                status=status,
            )
        if "assessment" in item and (
            item["assessment"] not in ("passed", "failed", "not_assessed")
            or (item["assessment"] == "passed") != passed
        ):
            return RubricStatusValidation(
                valid=False,
                overall_pass=False,
                error=f"rubrics[{idx}].assessment must agree with pass",
                status=status,
            )
        seen_ids.add(rubric_id)
        if not passed:
            all_pass = False

    if target_rubric_ids and seen_ids != target_rubric_ids:
        return RubricStatusValidation(
            valid=False,
            overall_pass=False,
            error=(
                f"rubric id set mismatch (expected {sorted(target_rubric_ids)} got {sorted(seen_ids)})"
            ),
            status=status,
        )

    if status["overall_pass"] != all_pass:
        return RubricStatusValidation(
            valid=False,
            overall_pass=False,
            error="overall_pass must equal all rubric pass values",
            status=status,
        )

    return RubricStatusValidation(
        valid=True,
        overall_pass=bool(status["overall_pass"]),
        error=None,
        status=status,
    )
