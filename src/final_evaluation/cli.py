"""CLI for once-only final evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import (
    FinalEvaluationError,
    FinalEvaluationInfrastructureError,
    run_final_evaluation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--investigation-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--container-image-recovery-ref")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout_seconds < 1:
        print("--timeout-seconds must be positive", file=sys.stderr)
        return 2
    try:
        result = run_final_evaluation(
            task_dir=args.task_dir,
            investigation_results_dir=args.investigation_results_dir,
            output_dir=args.output_dir,
            python_bin=args.python_bin,
            container_image=args.container_image,
            container_image_recovery_ref=args.container_image_recovery_ref,
            timeout_seconds=args.timeout_seconds,
        )
    except FinalEvaluationInfrastructureError as exc:
        print(f"final evaluation infrastructure failure: {exc}", file=sys.stderr)
        return 4
    except FinalEvaluationError as exc:
        print(f"final evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
