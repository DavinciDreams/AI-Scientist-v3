"""Command-line adapter for task-design artifact contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .design import (
    TaskDesign,
    TaskDesignAdjudicationRequired,
    TaskDesignError,
    TaskDesignReview,
    aggregate_task_design_reviews,
    apply_task_design_patch,
    compare_task_design_reviews,
    confirm_task_design_delta,
    load_task_design,
    load_task_design_adjudication,
    load_task_design_delta_review,
    load_task_design_review,
    task_design_changed_paths,
    task_design_adjudication_request,
    task_design_scope_alternatives,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate task-drafting design artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    design_parser = subparsers.add_parser("validate-design")
    design_parser.add_argument("--design", type=Path, required=True)
    design_parser.add_argument("--task", type=Path, required=True)
    review_parser = subparsers.add_parser("validate-review")
    review_parser.add_argument("--design", type=Path, required=True)
    review_parser.add_argument("--task", type=Path, required=True)
    review_parser.add_argument("--review", type=Path, required=True)
    review_parser.add_argument("--require-schema-version", type=int)
    patch_parser = subparsers.add_parser("apply-patch")
    patch_parser.add_argument("--design", type=Path, required=True)
    patch_parser.add_argument("--task", type=Path, required=True)
    patch_parser.add_argument("--patch", type=Path, required=True)
    patch_parser.add_argument("--review", type=Path, required=True)
    aggregate_parser = subparsers.add_parser("aggregate-reviews")
    aggregate_parser.add_argument("--design", type=Path, required=True)
    aggregate_parser.add_argument("--task", type=Path, required=True)
    aggregate_parser.add_argument("--review", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--adjudication", type=Path)
    aggregate_parser.add_argument("--require-schema-version", type=int)
    aggregate_parser.add_argument("--allow-unanimous-scope-fallback", action="store_true")
    request_parser = subparsers.add_parser("adjudication-request")
    request_parser.add_argument("--design", type=Path, required=True)
    request_parser.add_argument("--task", type=Path, required=True)
    request_parser.add_argument("--review", type=Path, action="append", required=True)
    adjudication_parser = subparsers.add_parser("validate-adjudication")
    adjudication_parser.add_argument("--design", type=Path, required=True)
    adjudication_parser.add_argument("--task", type=Path, required=True)
    adjudication_parser.add_argument("--review", type=Path, action="append", required=True)
    adjudication_parser.add_argument("--adjudication", type=Path, required=True)
    alternatives_parser = subparsers.add_parser("scope-alternatives")
    alternatives_parser.add_argument("--design", type=Path, required=True)
    alternatives_parser.add_argument("--task", type=Path, required=True)
    alternatives_parser.add_argument("--review", type=Path, action="append", required=True)
    alternatives_parser.add_argument("--adjudication", type=Path)
    alternatives_parser.add_argument("--require-schema-version", type=int)
    compare_parser = subparsers.add_parser("compare-reviews")
    compare_parser.add_argument("--parent-design", type=Path, required=True)
    compare_parser.add_argument("--candidate-design", type=Path, required=True)
    compare_parser.add_argument("--task", type=Path, required=True)
    compare_parser.add_argument("--parent-review", type=Path, required=True)
    compare_parser.add_argument("--candidate-review", type=Path, required=True)
    changes_parser = subparsers.add_parser("changed-paths")
    changes_parser.add_argument("--parent-design", type=Path, required=True)
    changes_parser.add_argument("--candidate-design", type=Path, required=True)
    changes_parser.add_argument("--task", type=Path, required=True)
    delta_parser = subparsers.add_parser("validate-delta")
    delta_parser.add_argument("--parent-design", type=Path, required=True)
    delta_parser.add_argument("--candidate-design", type=Path, required=True)
    delta_parser.add_argument("--task", type=Path, required=True)
    delta_parser.add_argument("--delta", type=Path, required=True)
    delta_parser.add_argument("--parent-review", type=Path, required=True)
    delta_parser.add_argument("--candidate-review", type=Path, required=True)
    return parser


def _compare_reviews(args: argparse.Namespace) -> int:
    parent = load_task_design(args.parent_design, canonical_task=args.task)
    candidate = load_task_design(args.candidate_design, canonical_task=args.task)
    progress = compare_task_design_reviews(
        load_task_design_review(args.parent_review, design=parent),
        load_task_design_review(args.candidate_review, design=candidate),
        parent_design=parent,
        candidate_design=candidate,
    )
    print(
        json.dumps(
            {
                "fixed": progress.fixed,
                "still_failed": progress.still_failed,
                "regressed": progress.regressed,
                "improved": progress.improved,
            },
            sort_keys=True,
        )
    )
    return 0 if progress.improved else 3


def _validate_delta(args: argparse.Namespace) -> int:
    parent = load_task_design(args.parent_design, canonical_task=args.task)
    candidate = load_task_design(args.candidate_design, canonical_task=args.task)
    delta = load_task_design_delta_review(args.delta, parent=parent, candidate=candidate)
    candidate_review = load_task_design_review(
        args.candidate_review, design=candidate, require_schema_version=3
    )
    progress = compare_task_design_reviews(
        load_task_design_review(args.parent_review, design=parent),
        candidate_review,
        parent_design=parent,
        candidate_design=candidate,
    )
    confirm_task_design_delta(delta, progress, candidate_review)
    print(
        json.dumps(
            {
                "verdict": delta.verdict,
                "fixed_dimensions": delta.fixed_dimensions,
                "regressed_dimensions": delta.regressed_dimensions,
            },
            sort_keys=True,
        )
    )
    return 0 if delta.verdict == "pass" else 3


def _aggregate_reviews(args: argparse.Namespace, design: TaskDesign) -> int:
    if len(args.review) != 3:
        raise TaskDesignError("exactly three --review arguments are required")
    reviews = [
        load_task_design_review(
            path,
            design=design,
            require_schema_version=args.require_schema_version,
        )
        for path in args.review
    ]
    adjudication = (
        load_task_design_adjudication(
            args.adjudication,
            design=design,
            reviews=reviews,
        )
        if args.adjudication is not None
        else None
    )
    aggregate = aggregate_task_design_reviews(
        reviews,
        adjudication=adjudication,
        allow_unanimous_scope_fallback=args.allow_unanimous_scope_fallback,
    )
    print(json.dumps(aggregate.payload, indent=2, sort_keys=True))
    return 0


def _load_v3_reviews(
    args: argparse.Namespace,
    design: TaskDesign,
) -> list[TaskDesignReview]:
    if len(args.review) != 3:
        raise TaskDesignError("exactly three --review arguments are required")
    return [
        load_task_design_review(path, design=design, require_schema_version=3)
        for path in args.review
    ]


def _scope_alternatives(args: argparse.Namespace, design: TaskDesign) -> int:
    if len(args.review) != 3:
        raise TaskDesignError("exactly three --review arguments are required")
    reviews = [
        load_task_design_review(
            path,
            design=design,
            require_schema_version=args.require_schema_version,
        )
        for path in args.review
    ]
    adjudication = (
        load_task_design_adjudication(
            args.adjudication,
            design=design,
            reviews=reviews,
        )
        if args.adjudication is not None
        else None
    )
    print(
        json.dumps(
            task_design_scope_alternatives(reviews, adjudication=adjudication),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "compare-reviews":
            return _compare_reviews(args)
        if args.command == "changed-paths":
            parent = load_task_design(args.parent_design, canonical_task=args.task)
            candidate = load_task_design(args.candidate_design, canonical_task=args.task)
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "paths": task_design_changed_paths(parent, candidate),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "validate-delta":
            return _validate_delta(args)
        design = load_task_design(args.design, canonical_task=args.task)
        if args.command == "validate-design":
            print(json.dumps({"fingerprint": design.fingerprint}, sort_keys=True))
            return 0
        if args.command == "apply-patch":
            repair_review = load_task_design_review(
                args.review, design=design, require_schema_version=3
            )
            candidate = apply_task_design_patch(
                args.patch,
                parent=design,
                canonical_task=args.task,
                repair_review=repair_review,
            )
            print(json.dumps(candidate.payload, indent=2, sort_keys=True))
            return 0
        if args.command == "aggregate-reviews":
            return _aggregate_reviews(args, design)
        if args.command == "adjudication-request":
            print(
                json.dumps(
                    task_design_adjudication_request(_load_v3_reviews(args, design)),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "validate-adjudication":
            reviews = _load_v3_reviews(args, design)
            adjudication = load_task_design_adjudication(
                args.adjudication,
                design=design,
                reviews=reviews,
            )
            print(
                json.dumps(
                    {
                        "request_fingerprint": adjudication.payload["request_fingerprint"],
                        "decisions": len(adjudication.payload["decisions"]),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "scope-alternatives":
            return _scope_alternatives(args, design)
        review = load_task_design_review(
            args.review,
            design=design,
            require_schema_version=args.require_schema_version,
        )
    except TaskDesignAdjudicationRequired as exc:
        print(f"task design review requires adjudication: {exc}")
        return 4
    except TaskDesignError as exc:
        print(f"invalid task design artifact: {exc}")
        return 2
    print(
        json.dumps(
            {
                "verdict": review.verdict,
                "failed_dimensions": review.failed_dimensions,
                "fingerprint": design.fingerprint,
            },
            sort_keys=True,
        )
    )
    return 0 if review.verdict == "pass" else 3
