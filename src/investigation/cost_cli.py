"""Command-line ingestion for cost accounting."""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from .cost_storage import (
    CostTrackingError,
    format_cost_checkpoint,
    summary_output_tokens,
    record_events,
)
from .cost_tracking import (
    _decimal,
    _decimal_text,
    _utc_timestamp,
    import_autoillustrate_metrics,
    parse_headless_log,
    preserve_autoillustrate_metrics,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("headless", "autoillustrate"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--phase", required=True)
        command.add_argument("--stage", required=True)
        command.add_argument("--invocation", required=True)
        command.add_argument("--status", default="completed")
        if name == "headless":
            command.add_argument("--agent", default="unknown")
        command.add_argument("--print-summary", action="store_true")
    return parser.parse_args()


def _events(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.command == "headless":
        return parse_headless_log(
            args.input,
            phase=args.phase,
            stage=args.stage,
            invocation_id=args.invocation,
            agent=args.agent,
            status=args.status,
        )
    preserved_input = preserve_autoillustrate_metrics(
        args.input,
        args.root,
        args.invocation,
    )
    return import_autoillustrate_metrics(
        preserved_input,
        phase=args.phase,
        stage=args.stage,
        invocation_id=args.invocation,
        status=args.status,
    )


def _print_checkpoint(
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> None:
    label = f"{args.phase}/{args.stage}"
    previous = summary.get("_previous_summary")
    baseline = previous if isinstance(previous, dict) else {}
    baseline_reported = _decimal(baseline.get("reported_cost_usd", 0))
    baseline_estimated = _decimal(baseline.get("estimated_cost_usd", 0))
    reported = max(
        Decimal(0), _decimal(summary.get("reported_cost_usd", 0)) - baseline_reported
    )
    estimated = max(
        Decimal(0), _decimal(summary.get("estimated_cost_usd", 0)) - baseline_estimated
    )
    estimated_events = max(
        0, int(summary.get("estimated_events", 0)) - int(baseline.get("estimated_events", 0))
    )
    usage_only = max(
        0, int(summary.get("usage_only_events", 0)) - int(baseline.get("usage_only_events", 0))
    )
    unknown = max(
        0, int(summary.get("unknown_events", 0)) - int(baseline.get("unknown_events", 0))
    )
    run_tokens = summary_output_tokens(summary)
    baseline_tokens = summary_output_tokens(baseline)
    baseline_unreported = int(baseline.get("unreported_usage_events", 0))
    unreported = max(0, int(summary.get("unreported_usage_events", 0)) - baseline_unreported)
    message = format_cost_checkpoint(
        summary,
        label=label,
        baseline_output_tokens=baseline_tokens,
        baseline_unreported=baseline_unreported,
    )
    if os.environ.get("AGENTIC_COST_OUTPUT_FORMAT") != "json":
        print(message)
        return
    print(
        json.dumps(
            {
                "timestamp": _utc_timestamp(),
                "source": "cost_tracking",
                "event": "cost_summary",
                "phase": args.phase,
                "fields": {
                    "label": label,
                    "message": message,
                    "reported_cost_usd": _decimal_text(reported),
                    "estimated_cost_usd": _decimal_text(estimated),
                    "total_cost_usd": _decimal_text(reported + estimated),
                    "run_reported_cost_usd": summary["reported_cost_usd"],
                    "run_estimated_cost_usd": summary["estimated_cost_usd"],
                    "run_total_cost_usd": summary["total_cost_usd"],
                    "estimated_events": estimated_events,
                    "usage_only_events": usage_only,
                    "unknown_events": unknown,
                    "output_tokens": max(0, run_tokens - baseline_tokens),
                    "run_output_tokens": run_tokens,
                    "unreported_usage_events": unreported,
                },
            },
            sort_keys=True,
        )
    )


def main() -> int:
    args = _arguments()
    try:
        events = _events(args)
        summary = record_events(
            args.root,
            events,
            replace_invocations=args.command == "headless",
        )
        if args.print_summary and int(summary.get("_recorded_event_count", 0)) > 0:
            _print_checkpoint(args, summary)
    except CostTrackingError as exc:
        print(f"[cost] warning: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
