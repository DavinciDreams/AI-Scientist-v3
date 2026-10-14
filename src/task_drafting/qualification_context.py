"""Bind judge attestations to current qualification evidence, never authored output."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.task_drafting.publication import task_publication_plan_sha256
from src.task_drafting.qualification import (
    TaskQualificationError,
    _read_owned_json,
    load_task_qualification_receipt,
)
from src.task_drafting.qualification_calibration import file_sha256
from src.investigation.task_alignment import task_tree_fingerprint


def qualification_success_context(
    *,
    task_dir: Path,
    publication_task_dir: Path,
    report_path: Path,
    receipt_path: Path,
    max_runtime_bytes: int,
) -> dict[str, object]:
    """Reject stale evidence before reusing a production-search success."""
    payload = _read_owned_json(report_path, maximum_bytes=1024 * 1024)
    fingerprint = task_tree_fingerprint(task_dir)
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "pass"
        or payload.get("category") != "qualification_passed"
        or payload.get("stage") != "production-search"
        or payload.get("task_fingerprint") != fingerprint
        or payload.get("receipt_path") != str(receipt_path)
    ):
        raise TaskQualificationError("qualification success does not match current scaffold")
    elapsed = payload.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise TaskQualificationError("qualification success has invalid elapsed time")
    plan = task_publication_plan_sha256(
        task_dir=publication_task_dir,
        staged_task_dir=task_dir,
        max_runtime_bytes=max_runtime_bytes,
    )
    load_task_qualification_receipt(receipt_path, expected_publication_plan_sha256=plan)
    return {
        "status": "pass",
        "stage": "production-search",
        "task_fingerprint": fingerprint,
        "elapsed_seconds": elapsed,
        "report_sha256": file_sha256(report_path),
        "receipt_sha256": file_sha256(receipt_path),
    }


def qualification_failure_projection(
    path: Path,
    *,
    current_fingerprint: str,
) -> dict[str, object]:
    """Historical failures are inert context, not the repaired scaffold's verdict."""
    payload = _read_owned_json(path, maximum_bytes=1024 * 1024)
    if not isinstance(payload, dict):
        raise TaskQualificationError("qualification failure must contain an object")
    projection: dict[str, object] = {
        key: payload[key]
        for key in ("status", "category", "stage", "elapsed_seconds", "task_fingerprint")
        if isinstance(payload.get(key), (str, int, float))
        and not isinstance(payload.get(key), bool)
    }
    allowed_evidence_fields = {
        "deadline_owner",
        "contract_wall_clock_seconds",
        "evaluator_timeout_seconds",
        "candidate_rpc_timeout_seconds",
        "broker_teardown_reserve_seconds",
        "container_cleanup_reserve_seconds",
        "candidate_rpc_calls",
        "candidate_calls_before_first_reset",
        "candidate_reset_count",
        "candidate_peer_disconnects",
        "max_output_bytes",
        "observed_output_bytes",
    }
    for field in ("failure_evidence", "output_evidence"):
        evidence = payload.get(field)
        if isinstance(evidence, dict):
            projection[field] = {
                key: value
                for key, value in evidence.items()
                if key in allowed_evidence_fields
                and isinstance(value, (str, int, float))
                and not isinstance(value, bool)
            }
    projection["diagnostic_sha256"] = file_sha256(path)
    projection["untrusted_diagnostic_omitted"] = True
    projection["scope"] = "historical"
    projection["matches_current_scaffold"] = payload.get("task_fingerprint") == current_fingerprint
    return projection


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--publication-task-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-runtime-bytes", type=int, required=True)
    args = parser.parse_args()
    try:
        qualification_success_context(
            task_dir=args.task_dir,
            publication_task_dir=args.publication_task_dir,
            report_path=args.report,
            receipt_path=args.receipt,
            max_runtime_bytes=args.max_runtime_bytes,
        )
    except (OSError, ValueError):
        print("Current production qualification evidence is missing, invalid, or stale")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
