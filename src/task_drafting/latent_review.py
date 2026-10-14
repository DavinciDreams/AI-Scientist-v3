"""Rebind unchanged full-review failures to a retained parent design."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from .design import (
    DESIGN_DIMENSIONS,
    MAX_DESIGN_ARTIFACT_BYTES,
    TaskDesign,
    TaskDesignDeltaReview,
    TaskDesignError,
    TaskDesignReview,
    compare_task_design_reviews,
    confirm_task_design_delta,
    load_task_design,
    load_task_design_delta_review,
    load_task_design_review,
    task_design_changed_paths,
)


def _paths_overlap(first: str, second: str) -> bool:
    return first == second or first.startswith(f"{second}/") or second.startswith(f"{first}/")


def _unchanged_paths(paths: list[str], changed_paths: tuple[str, ...]) -> list[str]:
    return [
        path
        for path in paths
        if not any(_paths_overlap(path, changed_path) for changed_path in changed_paths)
    ]


def rebind_latent_failures(
    *,
    parent: TaskDesign,
    candidate: TaskDesign,
    parent_review: TaskDesignReview,
    candidate_review: TaskDesignReview,
    delta_review: TaskDesignDeltaReview,
    consensus_regressions: tuple[str, ...],
) -> TaskDesignReview | None:
    """Authorize only candidate findings proven to concern unchanged parent paths."""
    changed_paths = task_design_changed_paths(parent, candidate)
    direct_regressions = set(delta_review.regressed_dimensions) | set(consensus_regressions)
    parent_rows = {row["id"]: row for row in parent_review.payload["dimensions"]}
    latent_rows: dict[str, dict[str, Any]] = {}
    for source in candidate_review.payload["dimensions"]:
        identifier = source["id"]
        if (
            source["verdict"] != "fail"
            or parent_rows[identifier]["verdict"] == "fail"
            or identifier in direct_regressions
        ):
            continue
        repair_paths = _unchanged_paths(source["repair_paths"], changed_paths)
        design_paths = _unchanged_paths(source["design_paths"], changed_paths)
        if (
            repair_paths != source["repair_paths"]
            or design_paths != source["design_paths"]
            or not repair_paths
            or not design_paths
        ):
            continue
        latent = deepcopy(source)
        latent["design_paths"] = design_paths
        latent_rows[identifier] = latent
    if not latent_rows:
        return None

    dimensions = [
        deepcopy(latent_rows[identifier])
        if identifier in latent_rows
        else deepcopy(parent_rows[identifier])
        for identifier in DESIGN_DIMENSIONS
    ]
    failed = tuple(row["id"] for row in dimensions if row["verdict"] == "fail")
    payload = {
        "schema_version": 3,
        "verdict": "fail",
        "summary": "Retained-parent repair authority with unchanged full-audit failures.",
        "design_fingerprint": parent.fingerprint,
        "dimensions": dimensions,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if len(encoded) + 1 > MAX_DESIGN_ARTIFACT_BYTES:
        return None
    return TaskDesignReview("fail", failed, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-design", type=Path, required=True)
    parser.add_argument("--candidate-design", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--parent-review", type=Path, required=True)
    parser.add_argument("--candidate-review", type=Path, required=True)
    parser.add_argument("--delta-review", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        parent = load_task_design(args.parent_design, canonical_task=args.task)
        candidate = load_task_design(args.candidate_design, canonical_task=args.task)
        parent_review = load_task_design_review(
            args.parent_review, design=parent, require_schema_version=3
        )
        candidate_review = load_task_design_review(
            args.candidate_review, design=candidate, require_schema_version=3
        )
        delta_review = load_task_design_delta_review(
            args.delta_review, parent=parent, candidate=candidate
        )
        progress = compare_task_design_reviews(
            parent_review,
            candidate_review,
            parent_design=parent,
            candidate_design=candidate,
        )
        confirm_task_design_delta(delta_review, progress, candidate_review)
        rebound = rebind_latent_failures(
            parent=parent,
            candidate=candidate,
            parent_review=parent_review,
            candidate_review=candidate_review,
            delta_review=delta_review,
            consensus_regressions=progress.regressed,
        )
    except TaskDesignError as exc:
        print(f"invalid task design artifact: {exc}", file=sys.stderr)
        return 2
    if rebound is None:
        return 3
    print(json.dumps(rebound.payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
