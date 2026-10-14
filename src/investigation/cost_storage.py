"""Concurrent private storage for cost-accounting events."""

from __future__ import annotations

import fcntl
import json
import math
import os
import stat
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

from .billing_metadata import validated_billing_metadata

SCHEMA_VERSION = 3
_SUPPORTED_SCHEMA_VERSIONS = {1, 2, SCHEMA_VERSION}
_MAX_USAGE_VALUE = 10**15
_MAX_REPORTED_COST = Decimal("1000000000")
_MAX_LEDGER_BYTES = 64 * 1024 * 1024
_MAX_EVENT_LINE_BYTES = 1024 * 1024
_MAX_SUMMARY_BYTES = 2 * 1024 * 1024
_BILLING_BASES = {"reported", "estimated", "usage_only", "unknown"}
_COST_BREAKDOWN_FIELDS = {"input", "cache_read", "cache_write", "output", "total"}
_AGGREGATED_COST_ROUNDING_TOLERANCE = Decimal("0.000000001")
_V2_EVENT_FIELDS = {
    "estimated_cost_usd",
    "total_cost_usd",
    "provider",
    "cost_basis",
    "pricing_source",
    "pricing_status",
    "cost_breakdown_usd",
}
_USAGE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_output_tokens",
    "images",
}


class CostTrackingError(ValueError):
    """Raised when an unsafe or invalid cost record is supplied."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_label(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 240 or any(ord(character) < 32 for character in text):
        raise CostTrackingError(f"invalid {field}")
    return text


def _decimal(value: object) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise CostTrackingError("reported cost must be a decimal") from exc
    if not amount.is_finite():
        raise CostTrackingError("reported cost must be finite and non-negative")
    if amount < 0 or amount > _MAX_REPORTED_COST:
        raise CostTrackingError("reported cost must be finite and non-negative")
    if isinstance(value, float):
        amount = amount.quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN)
    decimal_tuple = amount.as_tuple()
    if (
        not isinstance(decimal_tuple.exponent, int)
        or decimal_tuple.exponent < -12
        or len(decimal_tuple.digits) > 24
    ):
        raise CostTrackingError("reported cost must be finite and non-negative")
    return amount


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _validated_usage(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    usage: dict[str, int] = {}
    for key, value in raw.items():
        if key not in _USAGE_FIELDS:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CostTrackingError(f"usage field {key} must be a non-negative integer")
        finite = math.isfinite(value) if isinstance(value, float) else True
        if not finite or value < 0 or value > _MAX_USAGE_VALUE or int(value) != value:
            raise CostTrackingError(f"usage field {key} must be a non-negative integer")
        usage[str(key)] = int(value)
    return usage


def _validated_cost_breakdown(raw: object) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise CostTrackingError("cost breakdown must be an object")
    breakdown: dict[str, str] = {}
    for key, value in raw.items():
        if key not in _COST_BREAKDOWN_FIELDS:
            continue
        breakdown[str(key)] = _decimal_text(_decimal(value))
    return breakdown


def _validate_cost_provenance(
    event: Mapping[str, Any],
    *,
    billed_amount: Decimal,
    breakdown: Mapping[str, str],
) -> None:
    basis = event["billing_basis"]
    provenance = (
        event.get("cost_basis"),
        event.get("pricing_source"),
        event.get("pricing_status"),
    )
    valid_provenance: dict[str, set[tuple[object, object, object]]] = {
        "reported": {
            (None, None, None),
            ("native-reported", "native", "native"),
        },
        "estimated": {
            ("api-list-price-estimate", "models.dev", "priced"),
            ("api-list-price-estimate", "openai", "pinned"),
        },
        "usage_only": {
            (None, None, None),
            (None, None, "missing"),
        },
        "unknown": {
            (None, None, None),
            (None, None, "missing"),
        },
    }
    if provenance not in valid_provenance[basis]:
        raise CostTrackingError(f"cost provenance contradicts {basis} billing basis")
    if not breakdown:
        return
    if basis not in {"reported", "estimated"} or "total" not in breakdown:
        raise CostTrackingError("cost breakdown requires a billed total")
    breakdown_total = _decimal(breakdown["total"])
    if breakdown_total != billed_amount:
        raise CostTrackingError("cost breakdown total contradicts billed cost")
    components = {"input", "cache_read", "cache_write", "output"}
    present_components = components & set(breakdown)
    component_total = sum(
        (_decimal(breakdown[field]) for field in present_components), Decimal(0)
    )
    difference = component_total - breakdown_total
    if difference > _AGGREGATED_COST_ROUNDING_TOLERANCE:
        raise CostTrackingError("cost breakdown components exceed total")
    if (
        present_components == components
        and abs(difference) > _AGGREGATED_COST_ROUNDING_TOLERANCE
    ):
        raise CostTrackingError("cost breakdown components contradict total")


def _read_json_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r", encoding="utf-8", errors="replace") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_LEDGER_BYTES:
            raise CostTrackingError("cost ledger is invalid or too large")
        for line in stream:
            if len(line) > _MAX_EVENT_LINE_BYTES:
                raise CostTrackingError("cost ledger event is too large")
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _metrics_dir(root: Path) -> Path:
    root = root.expanduser()
    if root.is_symlink():
        raise CostTrackingError("cost root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    metrics = root / "metrics"
    if metrics.is_symlink():
        raise CostTrackingError("metrics directory must not be a symlink")
    metrics.mkdir(mode=0o700, exist_ok=True)
    return metrics


def _validated_event(
    raw: Mapping[str, Any], *, allow_legacy_schema: bool = False
) -> dict[str, Any]:
    event = dict(raw)
    version = event.get("schema_version", SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise CostTrackingError("unsupported cost event schema")
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise CostTrackingError("unsupported cost event schema")
    if not allow_legacy_schema and version != SCHEMA_VERSION:
        raise CostTrackingError("new cost events require the current schema")
    if version == 1:
        for field in _V2_EVENT_FIELDS:
            value = event.get(field)
            if value is not None and value != {}:
                raise CostTrackingError(f"legacy cost event cannot contain {field}")
    for field in (
        "event_id",
        "invocation_id",
        "phase",
        "stage",
        "source",
        "agent",
        "model",
        "status",
    ):
        event[field] = _safe_label(event.get(field), field=field)
    basis = event.get("billing_basis")
    allowed_bases = _BILLING_BASES if version >= 2 else _BILLING_BASES - {"estimated"}
    if basis not in allowed_bases:
        raise CostTrackingError("invalid billing_basis")
    reported = event.get("reported_cost_usd")
    estimated = event.get("estimated_cost_usd")
    expected_cost_presence = {
        "reported": (True, False),
        "estimated": (False, True),
        "usage_only": (False, False),
        "unknown": (False, False),
    }
    if (reported is not None, estimated is not None) != expected_cost_presence[basis]:
        raise CostTrackingError(f"cost fields contradict {basis} billing basis")
    reported_amount = _decimal(reported) if reported is not None else Decimal(0)
    estimated_amount = _decimal(estimated) if estimated is not None else Decimal(0)
    event["reported_cost_usd"] = _decimal_text(reported_amount) if reported is not None else None
    if version >= 2:
        event["estimated_cost_usd"] = (
            _decimal_text(estimated_amount) if estimated is not None else None
        )
        event["total_cost_usd"] = _decimal_text(reported_amount + estimated_amount)
        for field in ("provider", "cost_basis", "pricing_source", "pricing_status"):
            value = event.get(field)
            event[field] = _safe_label(value, field=field) if value is not None else None
        event["cost_breakdown_usd"] = _validated_cost_breakdown(event.get("cost_breakdown_usd"))
        if version == SCHEMA_VERSION:
            _validate_cost_provenance(
                event,
                billed_amount=reported_amount + estimated_amount,
                breakdown=event["cost_breakdown_usd"],
            )
    else:
        for field in _V2_EVENT_FIELDS:
            event.pop(field, None)
    event["usage"] = _validated_usage(event.get("usage"))
    if "billing" in event:
        try:
            event["billing"] = validated_billing_metadata(event["billing"])
        except ValueError as exc:
            raise CostTrackingError("invalid billing metadata") from exc
    event["schema_version"] = version
    event.setdefault("timestamp", _utc_timestamp())
    _safe_label(event["timestamp"], field="timestamp")
    artifact = event.get("artifact")
    if artifact is not None and not isinstance(artifact, str):
        raise CostTrackingError("artifact must be text or null")
    if isinstance(artifact, str) and (
        len(artifact) > 4096 or any(ord(character) < 32 for character in artifact)
    ):
        raise CostTrackingError("invalid artifact")
    return event


def _relative_artifact(value: object, root: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _event_scope(event: Mapping[str, Any]) -> tuple[object, ...]:
    return tuple(
        event.get(field)
        for field in ("invocation_id", "phase", "stage", "source", "agent", "artifact")
    )


def _material_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "timestamp"}


def _summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    reported_total = Decimal(0)
    estimated_total = Decimal(0)
    estimated_events = 0
    usage_only = 0
    unknown = 0
    unreported_usage = 0
    usage: dict[str, int] = {}
    grouped: dict[str, dict[str, dict[str, Any]]] = {
        "by_phase": {},
        "by_stage": {},
        "by_source": {},
        "by_model": {},
    }
    for event in events:
        reported = (
            _decimal(event["reported_cost_usd"]) if event["reported_cost_usd"] else Decimal(0)
        )
        estimated = (
            _decimal(event["estimated_cost_usd"]) if event.get("estimated_cost_usd") else Decimal(0)
        )
        reported_total += reported
        estimated_total += estimated
        estimated_events += event["billing_basis"] == "estimated"
        usage_only += event["billing_basis"] == "usage_only"
        unknown += event["billing_basis"] == "unknown"
        unreported_usage += "output_tokens" not in event["usage"]
        for key, value in event["usage"].items():
            usage[key] = usage.get(key, 0) + value
        for grouping, field in (
            ("by_phase", "phase"),
            ("by_stage", "stage"),
            ("by_source", "source"),
            ("by_model", "model"),
        ):
            group = grouped[grouping].setdefault(
                event[field],
                {
                    "reported_cost_usd": Decimal(0),
                    "estimated_cost_usd": Decimal(0),
                    "event_count": 0,
                    "estimated_events": 0,
                    "usage_only_events": 0,
                    "unknown_events": 0,
                },
            )
            group["reported_cost_usd"] += reported
            group["estimated_cost_usd"] += estimated
            group["event_count"] += 1
            group["estimated_events"] += event["billing_basis"] == "estimated"
            group["usage_only_events"] += event["billing_basis"] == "usage_only"
            group["unknown_events"] += event["billing_basis"] == "unknown"
    result = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _utc_timestamp(),
        "reported_cost_usd": _decimal_text(reported_total),
        "estimated_cost_usd": _decimal_text(estimated_total),
        "total_cost_usd": _decimal_text(reported_total + estimated_total),
        "event_count": len(events),
        "estimated_events": estimated_events,
        "usage_only_events": usage_only,
        "unknown_events": unknown,
        "unreported_usage_events": unreported_usage,
        "usage": dict(sorted(usage.items())),
    }
    for grouping, groups in grouped.items():
        result[grouping] = {
            key: {
                **value,
                "reported_cost_usd": _decimal_text(value["reported_cost_usd"]),
                "estimated_cost_usd": _decimal_text(value["estimated_cost_usd"]),
                "total_cost_usd": _decimal_text(
                    value["reported_cost_usd"] + value["estimated_cost_usd"]
                ),
            }
            for key, value in sorted(groups.items())
        }
    return result


def record_events(
    root: Path,
    events: Iterable[Mapping[str, Any]],
    *,
    replace_invocations: bool = False,
) -> dict[str, Any]:
    """Append unseen events and atomically rebuild the summary under one lock."""
    metrics = _metrics_dir(root)
    lock_path = metrics / ".cost.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(lock_fd, "r+") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            ledger = metrics / "cost_events.jsonl"
            existing: list[dict[str, Any]] = []
            if ledger.exists():
                if ledger.is_symlink():
                    raise CostTrackingError("cost ledger must not be a symlink")
                _repair_truncated_tail(ledger)
                for raw in _read_json_objects(ledger):
                    try:
                        existing.append(_validated_event(raw, allow_legacy_schema=True))
                    except CostTrackingError:
                        continue
            previous_summary = _summary(existing)
            seen = {str(event.get("event_id")) for event in existing}
            incoming = []
            incoming_by_id: dict[str, dict[str, Any]] = {}
            for raw_event in events:
                event = _validated_event(raw_event)
                event["artifact"] = _relative_artifact(event.get("artifact"), root)
                duplicate = incoming_by_id.get(event["event_id"])
                if duplicate is not None:
                    if _material_event(duplicate) != _material_event(event):
                        raise CostTrackingError("duplicate event_id has conflicting content")
                    continue
                incoming_by_id[event["event_id"]] = event
                if event["event_id"] in seen and not replace_invocations:
                    continue
                incoming.append(event)
                seen.add(event["event_id"])
            if replace_invocations and any(event["source"] != "headless" for event in incoming):
                raise CostTrackingError("only Headless event sets can replace prior events")
            replaced = False
            if replace_invocations and incoming:
                scopes = {_event_scope(event) for event in incoming}
                existing_scoped = [event for event in existing if _event_scope(event) in scopes]
                existing_by_id = {event["event_id"]: _material_event(event) for event in existing_scoped}
                incoming_by_id = {event["event_id"]: _material_event(event) for event in incoming}
                if existing_scoped and existing_by_id != incoming_by_id:
                    existing = [
                        event
                        for event in existing
                        if _event_scope(event) not in scopes
                    ]
                    existing.extend(incoming)
                    _write_ledger(ledger, existing)
                    replaced = True
            current_ids = {event["event_id"] for event in existing}
            additions = (
                []
                if replaced
                else [event for event in incoming if event["event_id"] not in current_ids]
            )
            if additions:
                flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW
                fd = os.open(ledger, flags, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as stream:
                    for event in additions:
                        stream.write(
                            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                        )
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(ledger, 0o600)
                existing.extend(additions)
            summary = _summary(existing)
            _write_summary(metrics, summary)
            recorded_count = len(incoming) if replaced else len(additions)
            return {
                **summary,
                "_recorded_event_count": recorded_count,
                "_previous_summary": previous_summary,
            }
    except OSError as exc:
        raise CostTrackingError("cost storage is unavailable") from exc


def _repair_truncated_tail(path: Path) -> None:
    flags = os.O_RDWR | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "r+b") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        if size == 0:
            return
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return
        start = max(0, size - _MAX_EVENT_LINE_BYTES)
        stream.seek(start)
        content = stream.read()
        boundary_in_content = content.rfind(b"\n") + 1
        if start > 0 and boundary_in_content == 0:
            raise CostTrackingError("cost ledger event is too large")
        tail = content[boundary_in_content:]
        try:
            complete = isinstance(json.loads(tail), dict)
        except (json.JSONDecodeError, UnicodeDecodeError):
            complete = False
        if complete:
            stream.seek(0, os.SEEK_END)
            stream.write(b"\n")
        else:
            stream.truncate(start + boundary_in_content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_ledger(path: Path, events: Iterable[Mapping[str, Any]]) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".cost-events-", delete=False
        ) as stream:
            temporary = stream.name
            os.chmod(temporary, 0o600)
            for event in events:
                stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _write_summary(metrics: Path, summary: Mapping[str, Any]) -> None:
    summary_path = metrics / "cost_summary.json"
    if summary_path.is_symlink():
        raise CostTrackingError("cost summary must not be a symlink")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=metrics, prefix=".cost-summary-", delete=False
        ) as stream:
            temporary = stream.name
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, summary_path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def store_cost_artifact(root: Path, *, category: str, name: str, content: bytes) -> Path:
    """Atomically retain an owner-only source artifact beside the ledger."""
    safe_category = _safe_label(category, field="artifact category")
    safe_name = _safe_label(name, field="artifact name")
    if "/" in safe_category or "/" in safe_name:
        raise CostTrackingError("invalid cost artifact path")
    metrics = _metrics_dir(root)
    directory = metrics / safe_category
    if directory.is_symlink():
        raise CostTrackingError("cost artifact directory must not be a symlink")
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / safe_name
    if path.is_symlink():
        raise CostTrackingError("cost artifact must not be a symlink")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=".cost-artifact-", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_summary(root: Path) -> dict[str, Any]:
    path = root.expanduser() / "metrics" / "cost_summary.json"
    if path.is_symlink():
        raise CostTrackingError("cost summary must not be a symlink")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_SUMMARY_BYTES:
                raise CostTrackingError("invalid cost summary")
            value = json.loads(stream.read(_MAX_SUMMARY_BYTES + 1))
    except FileNotFoundError:
        return _summary([])
    except (OSError, json.JSONDecodeError) as exc:
        raise CostTrackingError("cannot read cost summary") from exc
    if not isinstance(value, dict):
        raise CostTrackingError("invalid cost summary")
    return value


def format_token_count(tokens: int) -> str:
    """Render a token count the way Claude Code's footer does: ``12.3k``, ``4.8M``."""
    if tokens < 1_000:
        return str(tokens)
    if tokens < 1_000_000:
        return f"{tokens / 1_000:.1f}k"
    return f"{tokens / 1_000_000:.1f}M"


def summary_output_tokens(summary: Mapping[str, Any]) -> int:
    usage = summary.get("usage")
    value = usage.get("output_tokens", 0) if isinstance(usage, Mapping) else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CostTrackingError("invalid cost summary token count")
    return value


def format_cost_checkpoint(
    summary: Mapping[str, Any],
    *,
    label: str,
    baseline_output_tokens: int = 0,
    baseline_unreported: int = 0,
) -> str:
    """Render a stage checkpoint as output tokens, never dollars.

    Dollar figures stay in the ledger only: Claude Code and other subscription
    agents self-report API list prices that nobody is billed for. Like
    Autoillustrate and Claude Code's own footer, the line counts response
    length (output tokens) and flags calls that reported no token usage.
    """
    run_tokens = summary_output_tokens(summary)
    tokens = max(0, run_tokens - baseline_output_tokens)
    unreported = max(0, int(summary.get("unreported_usage_events", 0)) - baseline_unreported)
    delta_label = (
        "no usage" if not tokens and unreported else f"{format_token_count(tokens)} output"
    )
    activity = [delta_label, f"run {format_token_count(run_tokens)}"]
    if unreported:
        activity.append(f"{unreported} unreported")
    return f"Tokens · {label} · " + " · ".join(activity)
