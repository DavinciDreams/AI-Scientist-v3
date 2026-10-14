"""Validated progress transport for the task-drafting subprocess."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from src.investigation.cost_storage import (
    CostTrackingError,
    format_cost_checkpoint,
    load_summary,
    summary_output_tokens,
)
from src.investigation.terminal_ui import terminal_text

TRANSPORT_PREFIX = "AGENTIC_SCIENTIST_TASK_EVENT "
COST_BASELINE_PREFIX = "AGENTIC_SCIENTIST_TASK_COST_BASELINE "
VALID_GATES = {
    "workflow",
    "resource_preflight",
    "agent_call",
    "design_schema",
    "design_review",
    "design_consensus",
    "design_repair",
    "scaffold",
    "candidate_boundary",
    "contract",
    "smoke",
    "qualification",
    "alignment",
    "publication",
}
VALID_STATES = {"started", "completed", "failed", "retrying", "skipped"}
_FIELD_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_TRANSPORT_PAYLOAD_CHARS = 64 * 1024
_RECORD_KEYS = {
    "auth",
    "timestamp",
    "source",
    "event",
    "session_id",
    "gate",
    "state",
    "fields",
}
_COST_KEYS = (
    "reported_cost_usd",
    "estimated_cost_usd",
    "estimated_events",
    "usage_only_events",
    "unknown_events",
    "output_tokens",
    "unreported_usage_events",
)
TASK_PROGRESS_COST_FIELDS = {
    "cost_message",
    "run_cost_usd",
    "run_output_tokens",
    "task_cost_usd",
    "task_estimated_cost_usd",
    "task_estimated_events",
    "task_output_tokens",
    "task_reported_cost_usd",
    "task_unknown_events",
    "task_unreported_usage_events",
    "task_usage_only_events",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _decimal(value: object) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CostTrackingError("invalid task drafting cost") from exc
    if not parsed.is_finite() or parsed < 0:
        raise CostTrackingError("invalid task drafting cost")
    return parsed


def cost_baseline(summary: Mapping[str, Any]) -> dict[str, str | int]:
    """Return the stable counters needed to measure task-drafting cost deltas."""
    return {
        "reported_cost_usd": format(_decimal(summary.get("reported_cost_usd", 0)), "f"),
        "estimated_cost_usd": format(_decimal(summary.get("estimated_cost_usd", 0)), "f"),
        "estimated_events": max(0, int(summary.get("estimated_events", 0))),
        "usage_only_events": max(0, int(summary.get("usage_only_events", 0))),
        "unknown_events": max(0, int(summary.get("unknown_events", 0))),
        "output_tokens": summary_output_tokens(summary),
        "unreported_usage_events": max(0, int(summary.get("unreported_usage_events", 0))),
    }


def build_cost_fields(
    summary: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, str | int]:
    """Build bounded checkpoint cost fields relative to the supplied baseline."""
    reported_total = _decimal(summary.get("reported_cost_usd", 0))
    estimated_total = _decimal(summary.get("estimated_cost_usd", 0))
    run_total = _decimal(summary.get("total_cost_usd", reported_total + estimated_total))
    baseline_reported = _decimal(baseline.get("reported_cost_usd", 0))
    baseline_estimated = _decimal(baseline.get("estimated_cost_usd", 0))
    reported = max(Decimal(0), reported_total - baseline_reported)
    estimated = max(Decimal(0), estimated_total - baseline_estimated)
    estimated_events = max(
        0,
        int(summary.get("estimated_events", 0)) - int(baseline.get("estimated_events", 0)),
    )
    usage_only = max(
        0,
        int(summary.get("usage_only_events", 0)) - int(baseline.get("usage_only_events", 0)),
    )
    unknown = max(
        0,
        int(summary.get("unknown_events", 0)) - int(baseline.get("unknown_events", 0)),
    )
    run_tokens = summary_output_tokens(summary)
    baseline_tokens = max(0, int(baseline.get("output_tokens", 0)))
    baseline_unreported = max(0, int(baseline.get("unreported_usage_events", 0)))
    unreported = max(
        0, int(summary.get("unreported_usage_events", 0)) - baseline_unreported
    )
    return {
        "task_reported_cost_usd": f"{reported:.2f}",
        "task_estimated_cost_usd": f"{estimated:.2f}",
        "task_cost_usd": f"{reported + estimated:.2f}",
        "run_cost_usd": f"{run_total:.2f}",
        "task_estimated_events": estimated_events,
        "task_usage_only_events": usage_only,
        "task_unknown_events": unknown,
        "task_output_tokens": max(0, run_tokens - baseline_tokens),
        "run_output_tokens": run_tokens,
        "task_unreported_usage_events": unreported,
        "cost_message": format_cost_checkpoint(
            summary,
            label="task drafting",
            baseline_output_tokens=baseline_tokens,
            baseline_unreported=baseline_unreported,
        ),
    }


def build_record(
    *,
    gate: str,
    state: str,
    fields: Mapping[str, Any],
    session_id: str,
    auth_token: str | None = None,
    timestamp_fn: Callable[[], str] = utc_timestamp,
) -> dict[str, Any]:
    if gate not in VALID_GATES:
        raise ValueError(f"unknown task drafting gate: {gate}")
    if state not in VALID_STATES:
        raise ValueError(f"unknown task drafting gate state: {state}")
    if not session_id.strip():
        raise ValueError("session_id must be non-empty")
    invalid_keys = [key for key in fields if not _FIELD_KEY_RE.fullmatch(key)]
    if invalid_keys:
        raise ValueError(f"invalid task drafting event field: {invalid_keys[0]}")
    record = {
        "timestamp": timestamp_fn(),
        "source": "task_drafting",
        "event": "task_drafting_checkpoint",
        "session_id": session_id,
        "gate": gate,
        "state": state,
        "fields": _normalize_value(fields),
    }
    canonical = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    record["auth"] = (
        hmac.new(auth_token.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
        if auth_token
        else ""
    )
    return record


def emit_event(
    *,
    gate: str,
    state: str,
    fields: Mapping[str, Any],
    transport: str,
    stream: TextIO,
    session_id: str,
    auth_token: str | None = None,
    timestamp_fn: Callable[[], str] = utc_timestamp,
) -> dict[str, Any]:
    if transport not in {"direct", "wrapper"}:
        raise ValueError(f"unknown task drafting transport: {transport}")
    record = build_record(
        gate=gate,
        state=state,
        fields=fields,
        session_id=session_id,
        auth_token=auth_token,
        timestamp_fn=timestamp_fn,
    )
    if transport == "wrapper":
        stream.write(f"{TRANSPORT_PREFIX}{json.dumps(record, sort_keys=True)}\n")
    else:
        label = terminal_text(fields.get("label") or gate.replace("_", " ").title())
        verdict = str(fields.get("verdict", "")).upper()
        suffix = f" · {verdict}" if verdict else f" · {state}"
        elapsed = fields.get("elapsed")
        if elapsed:
            suffix += f" · {terminal_text(elapsed)}"
        stream.write(f"[TASK] {label}{suffix}\n")
        if state != "started" and fields.get("cost_message"):
            stream.write(f"{terminal_text(fields['cost_message'])}\n")
        for key, prefix in (
            ("failed_dimensions", "Failed"),
            ("fixed", "Fixed"),
            ("still_failed", "Still failing"),
            ("regressed", "Regressed"),
        ):
            values = fields.get(key)
            if isinstance(values, list) and values:
                rendered = ", ".join(terminal_text(value, limit=80) for value in values[:12])
                stream.write(f"  {prefix}: {rendered}\n")
    stream.flush()
    return record


def _validated_record(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or set(payload) != _RECORD_KEYS:
        return None
    if (
        payload.get("source") != "task_drafting"
        or payload.get("event") != "task_drafting_checkpoint"
        or payload.get("gate") not in VALID_GATES
        or payload.get("state") not in VALID_STATES
        or not isinstance(payload.get("timestamp"), str)
        or not isinstance(payload.get("session_id"), str)
        or not payload["session_id"].strip()
        or not isinstance(payload.get("fields"), dict)
        or not isinstance(payload.get("auth"), str)
    ):
        return None
    fields = payload["fields"]
    if any(not isinstance(key, str) or not _FIELD_KEY_RE.fullmatch(key) for key in fields):
        return None
    return payload


def extract_transport_events(
    output: str,
    *,
    expected_session_id: str | None = None,
    auth_token: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Remove validated task progress sentinels from captured subprocess output."""
    retained: list[str] = []
    records: list[dict[str, Any]] = []
    for line in output.splitlines(keepends=True):
        if not line.startswith(TRANSPORT_PREFIX):
            retained.append(line)
            continue
        encoded = line[len(TRANSPORT_PREFIX) :].rstrip("\r\n")
        if len(encoded) > _MAX_TRANSPORT_PAYLOAD_CHARS:
            retained.append(line)
            continue
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError:
            retained.append(line)
            continue
        record = _validated_record(payload)
        if record is None:
            retained.append(line)
            continue
        supplied_auth = str(record["auth"])
        unsigned = {key: value for key, value in record.items() if key != "auth"}
        canonical = json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        expected_auth = (
            hmac.new(auth_token.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
            if auth_token
            else ""
        )
        if (
            expected_session_id is not None and record["session_id"] != expected_session_id
        ) or not hmac.compare_digest(supplied_auth, expected_auth):
            retained.append(line)
            continue
        records.append(record)
    return "".join(retained), records


def _parse_json_list(value: str | None, label: str) -> list[str] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a JSON list") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"{label} must be a JSON list of strings")
    return parsed[:32]


def _emit_cli(args: argparse.Namespace) -> int:
    fields: dict[str, Any] = {"label": args.label}
    transport = os.environ.get("AGENTIC_TASK_PROGRESS_TRANSPORT", "direct")
    next_baseline: dict[str, str | int] | None = None
    for name in (
        "round",
        "current",
        "total",
        "attempt",
        "max_attempts",
        "infrastructure_attempt",
        "max_infrastructure_attempts",
        "repairs_used",
        "max_repairs",
        "validation_kind",
        "failure_kind",
        "repair_session_id",
        "trigger_category",
    ):
        value = getattr(args, name)
        if value is not None:
            fields[name] = value
    if args.verdict:
        fields["verdict"] = args.verdict
    if args.elapsed_seconds is not None:
        fields["elapsed_seconds"] = args.elapsed_seconds
        fields["elapsed"] = _format_elapsed(args.elapsed_seconds)
    for name in ("failed_dimensions", "fixed", "still_failed", "regressed"):
        value = _parse_json_list(getattr(args, name), name)
        if value is not None:
            fields[name] = value
    if args.include_cost:
        if transport == "wrapper":
            fields["cost_checkpoint"] = True
        else:
            try:
                root = Path(os.environ["AGENTIC_COST_ROOT"])
                baseline = json.loads(os.environ.get("TASK_PROGRESS_COST_BASELINE", "{}"))
                if not isinstance(baseline, dict):
                    baseline = {}
                summary = load_summary(root)
                fields.update(build_cost_fields(summary, baseline))
                next_baseline = cost_baseline(summary)
            except (KeyError, CostTrackingError, json.JSONDecodeError, OSError, ValueError):
                pass
    emit_event(
        gate=args.gate,
        state=args.state,
        fields=fields,
        transport=transport,
        stream=sys.stdout,
        session_id=os.environ.get("TASK_PROGRESS_SESSION_ID", "task-drafting"),
        auth_token=os.environ.get("TASK_PROGRESS_AUTH_TOKEN"),
    )
    if args.return_cost_baseline and next_baseline is not None:
        sys.stdout.write(f"{COST_BASELINE_PREFIX}{json.dumps(next_baseline, sort_keys=True)}\n")
        sys.stdout.flush()
    return 0


def _format_elapsed(seconds: float) -> str:
    whole = max(0, int(seconds))
    minutes, remainder = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {remainder}s"
    return f"{remainder}s"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit validated task-drafting progress")
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--root", type=Path, required=True)
    baseline.add_argument("--strict", action="store_true")
    emit = subparsers.add_parser("emit")
    emit.add_argument("--gate", choices=sorted(VALID_GATES), required=True)
    emit.add_argument("--state", choices=sorted(VALID_STATES), required=True)
    emit.add_argument("--label", required=True)
    emit.add_argument("--round", type=int)
    emit.add_argument("--current", type=int)
    emit.add_argument("--total", type=int)
    emit.add_argument("--attempt", type=int)
    emit.add_argument("--repair-session-id")
    emit.add_argument("--trigger-category", choices=("runtime_dependencies",))
    emit.add_argument("--max-attempts", type=int)
    emit.add_argument("--infrastructure-attempt", type=int)
    emit.add_argument("--max-infrastructure-attempts", type=int)
    emit.add_argument("--repairs-used", type=int)
    emit.add_argument("--max-repairs", type=int)
    emit.add_argument(
        "--validation-kind",
        choices=(
            "initial",
            "post_repair",
            "rollback_confirmation",
            "targeted_post_repair",
        ),
    )
    emit.add_argument("--verdict", choices=("pass", "fail"))
    emit.add_argument(
        "--failure-kind",
        choices=("authored_failure", "infrastructure_failure"),
    )
    emit.add_argument("--elapsed-seconds", type=float)
    emit.add_argument("--failed-dimensions")
    emit.add_argument("--fixed")
    emit.add_argument("--still-failed")
    emit.add_argument("--regressed")
    emit.add_argument("--include-cost", action="store_true")
    emit.add_argument("--return-cost-baseline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "baseline":
        try:
            summary = load_summary(args.root)
            print(json.dumps(cost_baseline(summary), sort_keys=True))
        except (CostTrackingError, OSError, ValueError):
            if args.strict:
                return 1
            print(json.dumps(cost_baseline({}), sort_keys=True))
        return 0
    return _emit_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
