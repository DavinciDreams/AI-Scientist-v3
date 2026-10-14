"""Provenance-aware API and Headless cost accounting."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

from .billing_metadata import validated_billing_metadata
from .cost_storage import (
    CostTrackingError,
    format_cost_checkpoint as format_cost_checkpoint,
    format_token_count as format_token_count,
    summary_output_tokens as summary_output_tokens,
    load_summary as load_summary,
    record_events,
    store_cost_artifact,
)

SCHEMA_VERSION = 3
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_METRICS_BYTES = 8 * 1024 * 1024
_MAX_USAGE_VALUE = 10**15
_MAX_REPORTED_COST = Decimal("1000000000")
_USAGE_ALIASES = {
    "input_tokens": ("input_tokens", "inputTokens", "prompt_tokens"),
    "output_tokens": ("output_tokens", "outputTokens", "completion_tokens"),
    "cached_input_tokens": (
        "cached_input_tokens",
        "cached",
        "cache_read_tokens",
        "cacheReadTokens",
        "cacheReadInputTokens",
        "cache_read_input_tokens",
    ),
    "cache_creation_input_tokens": (
        "cache_creation_input_tokens",
        "cache_creation_tokens",
        "cacheWriteTokens",
        "cacheCreationInputTokens",
    ),
    "reasoning_output_tokens": (
        "reasoning_output_tokens",
        "reasoning_tokens",
        "reasoningOutputTokens",
    ),
    "images": ("images", "image_count"),
}


@dataclass(frozen=True)
class ApiTokenPrice:
    """Pinned per-million-token rates for one API model."""

    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal
    pricing_source: str
    pricing_status: str


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


def _usage_from_mapping(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    usage: dict[str, int] = {}
    for canonical, aliases in _USAGE_ALIASES.items():
        for alias in aliases:
            value = raw.get(alias)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                break
            finite = math.isfinite(value) if isinstance(value, float) else True
            if not finite or value < 0 or value > _MAX_USAGE_VALUE or int(value) != value:
                raise CostTrackingError(f"usage field {alias} must be a non-negative integer")
            usage[canonical] = int(value)
            break
    return usage


def _normalized_usage_from_mapping(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    usage: dict[str, int] = {}
    for canonical, aliases in _USAGE_ALIASES.items():
        values: list[int] = []
        for alias in aliases:
            if alias not in raw:
                continue
            value = raw[alias]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CostTrackingError(f"usage field {alias} must be a non-negative integer")
            finite = math.isfinite(value) if isinstance(value, float) else True
            if not finite or value < 0 or value > _MAX_USAGE_VALUE or int(value) != value:
                raise CostTrackingError(f"usage field {alias} must be a non-negative integer")
            values.append(int(value))
        if len(set(values)) > 1:
            raise CostTrackingError(f"conflicting aliases for usage field {canonical}")
        if values:
            usage[canonical] = values[0]
    return usage


def _sum_usage(items: Iterable[Mapping[str, int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for usage in items:
        for key, value in usage.items():
            result[key] = result.get(key, 0) + value
    return result


def _read_bounded_bytes(path: Path, *, max_bytes: int, tail: bool) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise CostTrackingError(f"cost source is not a regular file: {path.name}")
            if metadata.st_size > max_bytes:
                if not tail:
                    raise CostTrackingError(f"cost source is too large: {path.name}")
                stream.seek(metadata.st_size - max_bytes)
                stream.readline()
            return stream.read(max_bytes + 1)
    except OSError as exc:
        raise CostTrackingError(f"cannot read cost source: {path.name}") from exc


def _read_json_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    content = _read_bounded_bytes(path, max_bytes=_MAX_SOURCE_BYTES, tail=True)
    for line in content.decode("utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _event(
    *,
    phase: str,
    stage: str,
    invocation_id: str,
    source: str,
    agent: str,
    model: str,
    status: str,
    reported_cost: object | None,
    usage: Mapping[str, int],
    artifact: str | None,
    item: str,
    estimated_cost: object | None = None,
    provider: object | None = None,
    cost_basis: object | None = None,
    pricing_source: object | None = None,
    pricing_status: object | None = None,
    cost_breakdown: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    amount = _decimal(reported_cost) if reported_cost is not None else None
    estimate = _decimal(estimated_cost) if estimated_cost is not None else None
    if amount is not None and estimate is not None:
        raise CostTrackingError("cost cannot be both reported and estimated")
    basis = (
        "reported"
        if amount is not None
        else "estimated"
        if estimate is not None
        else "usage_only"
        if usage
        else "unknown"
    )
    identity = {
        "invocation_id": _safe_label(invocation_id, field="invocation_id"),
        "phase": _safe_label(phase, field="phase"),
        "stage": _safe_label(stage, field="stage"),
        "source": _safe_label(source, field="source"),
        "artifact": artifact,
        "item": _safe_label(item, field="item"),
    }
    event_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "timestamp": _utc_timestamp(),
        "invocation_id": identity["invocation_id"],
        "phase": identity["phase"],
        "stage": identity["stage"],
        "source": identity["source"],
        "agent": _safe_label(agent, field="agent"),
        "model": _safe_label(model or "unknown", field="model"),
        "status": _safe_label(status, field="status"),
        "billing_basis": basis,
        "reported_cost_usd": _decimal_text(amount) if amount is not None else None,
        "estimated_cost_usd": _decimal_text(estimate) if estimate is not None else None,
        "total_cost_usd": _decimal_text((amount or Decimal(0)) + (estimate or Decimal(0))),
        "provider": _safe_label(provider, field="provider") if provider else None,
        "cost_basis": _safe_label(cost_basis, field="cost_basis") if cost_basis else None,
        "pricing_source": (
            _safe_label(pricing_source, field="pricing_source") if pricing_source else None
        ),
        "pricing_status": (
            _safe_label(pricing_status, field="pricing_status") if pricing_status else None
        ),
        "cost_breakdown_usd": dict(sorted((cost_breakdown or {}).items())),
        "usage": dict(sorted(usage.items())),
        "artifact": artifact,
    }


_COST_BREAKDOWN_FIELDS = {
    "input": "input",
    "cacheRead": "cache_read",
    "cacheWrite": "cache_write",
    "output": "output",
    "total": "total",
}


def _normalized_cost_breakdown(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    breakdown: dict[str, str] = {}
    for source, destination in _COST_BREAKDOWN_FIELDS.items():
        value = raw.get(source)
        if value is not None:
            breakdown[destination] = _decimal_text(_decimal(value))
    return breakdown


def _normalized_schema_fields(
    summary: Mapping[str, object],
) -> tuple[object, object, bool] | None:
    has_usage_status = "usageStatus" in summary
    has_cost_basis = "costBasis" in summary
    if has_usage_status != has_cost_basis:
        return None
    if has_usage_status:
        return summary.get("usageStatus"), summary.get("costBasis"), False
    provenance = (summary.get("pricingSource"), summary.get("pricingStatus"))
    if provenance == ("native", "native"):
        cost_basis = "native-reported"
    elif provenance == ("models.dev", "priced"):
        cost_basis = "api-list-price-estimate"
    else:
        cost_basis = None
    return "reported", cost_basis, True


def _normalized_headless_event(
    records: list[dict[str, Any]],
    *,
    phase: str,
    stage: str,
    invocation_id: str,
    agent: str,
    status: str,
    artifact: str,
    item: str = "normalized_usage",
) -> dict[str, Any] | None:
    """Parse Headless's appended normalized usage record when present and valid."""
    if not records:
        return None
    wrapper = records[-1]
    if set(wrapper) != {"usage"}:
        return None
    summary = wrapper.get("usage")
    if not isinstance(summary, Mapping) or summary.get("agent") != agent:
        return None

    try:
        schema_fields = _normalized_schema_fields(summary)
        if schema_fields is None:
            return None
        usage_status, cost_basis, legacy_schema = schema_fields
        if usage_status not in {"reported", "missing"}:
            return None
        usage = _normalized_usage_from_mapping(summary) if usage_status == "reported" else {}
        if usage_status == "reported" and (
            not usage or (legacy_schema and not any(usage.values()))
        ):
            return None
        raw_cost = summary.get("cost")
        pricing_source = summary.get("pricingSource")
        pricing_status = summary.get("pricingStatus")
        for field, value in (
            ("model", summary.get("model")),
            ("provider", summary.get("provider")),
            ("costBasis", cost_basis),
            ("pricingSource", pricing_source),
            ("pricingStatus", pricing_status),
        ):
            if value is not None:
                if not isinstance(value, str):
                    return None
                _safe_label(value, field=field)

        native = (
            cost_basis == "native-reported"
            and pricing_source == "native"
            and pricing_status == "native"
        )
        estimated = (
            cost_basis == "api-list-price-estimate"
            and pricing_source == "models.dev"
            and pricing_status == "priced"
        )
        unpriced = (
            raw_cost is None
            and cost_basis is None
            and pricing_source is None
            and pricing_status == "missing"
        )
        if usage_status == "missing" and not unpriced:
            return None
        if not (native or estimated or unpriced):
            return None

        breakdown: dict[str, str] = {}
        total: object | None = None
        if native or estimated:
            if not isinstance(raw_cost, Mapping) or raw_cost.get("total") is None:
                return None
            breakdown = _normalized_cost_breakdown(raw_cost)
            total = raw_cost["total"]
        reported_cost: object | None = None
        estimated_cost: object | None = None
        if native:
            reported_cost = total
        elif estimated:
            estimated_cost = total
        event = _event(
            phase=phase,
            stage=stage,
            invocation_id=invocation_id,
            source="headless",
            agent=agent,
            model=str(summary.get("model") or agent),
            status=status,
            reported_cost=reported_cost,
            estimated_cost=estimated_cost,
            usage=usage,
            artifact=artifact,
            item=item,
            provider=summary.get("provider"),
            cost_basis=cost_basis,
            pricing_source=pricing_source,
            pricing_status=pricing_status,
            cost_breakdown=breakdown,
        )
        if "billing" in summary:
            try:
                event["billing"] = validated_billing_metadata(summary["billing"])
            except ValueError:
                pass
        return event
    except CostTrackingError:
        return None


def _billing_headless_events(
    normalized: dict[str, Any], records: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    attempts = normalized["billing"]["attempts"]
    results = [record for record in records if record.get("type") == "result"]
    events = []
    for index, attempt in enumerate(attempts):
        event = _normalized_headless_event(
            [{"usage": attempt["usage"]}],
            phase=normalized["phase"],
            stage=normalized["stage"],
            invocation_id=normalized["invocation_id"],
            agent=normalized["agent"],
            status=normalized["status"] if index == len(attempts) - 1 else "failed",
            artifact=normalized["artifact"],
            item=f"billing_attempt:{index}",
        )
        if event is None:
            return None
        model_events = _billing_attempt_model_events(event, results, index, len(attempts))
        for model_event in model_events:
            model_event["billing"] = {"attempts": [attempt]}
            events.append(model_event)
    return events


def _billing_attempt_model_events(
    event: dict[str, Any], results: list[dict[str, Any]], index: int, attempt_count: int,
) -> list[dict[str, Any]]:
    if event["agent"] != "claude":
        return [event]
    if len(results) != attempt_count:
        mixed = any(isinstance(row.get("modelUsage"), Mapping)
                    and len(row["modelUsage"]) > 1 for row in results)
        return [{**event, "model": "mixed"}] if mixed else [event]
    result = results[index]
    model_usage = result.get("modelUsage")
    if not isinstance(model_usage, Mapping) or len(model_usage) <= 1:
        return [event]
    if len(model_usage) <= 32:
        try:
            split = _multi_model_headless_events(
                [result], event, phase=event["phase"], stage=event["stage"],
                invocation_id=event["invocation_id"], agent=event["agent"],
                status=event["status"], artifact=event["artifact"],
                item_prefix=f"billing_attempt:{index}:",
            )
            if split and _usage_totals_match(split, event):
                return split
        except (CostTrackingError, TypeError, ValueError):
            pass
    return [{**event, "model": "mixed"}]


def _usage_only_event(event: Mapping[str, Any]) -> dict[str, Any]:
    usage = event.get("usage")
    return {
        **event,
        "billing_basis": "usage_only" if usage else "unknown",
        "reported_cost_usd": None,
        "estimated_cost_usd": None,
        "total_cost_usd": "0",
        "cost_basis": None,
        "pricing_source": None,
        "pricing_status": None,
        "cost_breakdown_usd": {},
    }


def _normalized_cost_only_event(event: Mapping[str, Any]) -> dict[str, Any] | None:
    if event.get("billing_basis") not in {"reported", "estimated"}:
        return None
    return {**event, "model": "mixed", "usage": {}}


def _usage_totals_match(
    events: Iterable[Mapping[str, Any]], normalized: Mapping[str, Any]
) -> bool:
    event_usage = _sum_usage(
        event.get("usage", {})
        for event in events
        if isinstance(event.get("usage"), Mapping)
    )
    normalized_usage = normalized.get("usage")
    if not isinstance(normalized_usage, Mapping):
        return False
    fields = set(event_usage) | set(normalized_usage)
    return all(event_usage.get(field, 0) == normalized_usage.get(field, 0) for field in fields)


def _unique_model_usage_matches(
    events: Iterable[Mapping[str, Any]], normalized: Mapping[str, Any]
) -> bool:
    matching = [event for event in events if _usage_totals_match([event], normalized)]
    return len(matching) == 1


def _gemini_model_usage(details: object) -> dict[str, int]:
    usage = _usage_from_mapping(details)
    if "input_tokens" in usage and "cached_input_tokens" in usage:
        usage["input_tokens"] = max(
            0, usage["input_tokens"] - usage["cached_input_tokens"]
        )
    return usage


def _multi_model_headless_events(
    records: list[dict[str, Any]],
    normalized: Mapping[str, Any],
    *,
    phase: str,
    stage: str,
    invocation_id: str,
    agent: str,
    status: str,
    artifact: str,
    item_prefix: str = "",
) -> list[dict[str, Any]] | None:
    final = next((record for record in reversed(records) if record.get("type") == "result"), None)
    if final is None:
        return None

    model_usage = final.get("modelUsage")
    if agent == "claude" and isinstance(model_usage, Mapping) and len(model_usage) > 1:
        events = []
        for model, details in sorted(model_usage.items()):
            details_map = details if isinstance(details, Mapping) else {}
            events.append(
                _event(
                    phase=phase,
                    stage=stage,
                    invocation_id=invocation_id,
                    source="headless",
                    agent=agent,
                    model=str(model),
                    status=status,
                    reported_cost=details_map.get("costUSD"),
                    usage=_usage_from_mapping(details_map),
                    artifact=artifact,
                    item=f"{item_prefix}model:{model}",
                    provider=normalized.get("provider"),
                )
            )
        if len({event["event_id"] for event in events}) != len(events):
            return None
        reported_total = normalized.get("reported_cost_usd")
        raw_costs = [event.get("reported_cost_usd") for event in events]
        usage_reconciled = _usage_totals_match(
            events, normalized
        ) or _unique_model_usage_matches(events, normalized)
        if reported_total is not None and all(cost is not None for cost in raw_costs):
            if sum((_decimal(cost) for cost in raw_costs), Decimal(0)) == _decimal(reported_total):
                if usage_reconciled:
                    return [
                        {
                            **event,
                            "cost_basis": normalized.get("cost_basis"),
                            "pricing_source": normalized.get("pricing_source"),
                            "pricing_status": normalized.get("pricing_status"),
                        }
                        for event in events
                    ]
        usage_events = [_usage_only_event(event) for event in events]
        if not _usage_totals_match(usage_events, normalized):
            return None
        cost_event = _normalized_cost_only_event(normalized)
        return [*usage_events, *([cost_event] if cost_event is not None else [])]

    stats = final.get("stats")
    stats_models = stats.get("models") if isinstance(stats, Mapping) else None
    if agent != "gemini" or not isinstance(stats_models, Mapping) or len(stats_models) <= 1:
        return None
    events = [
        _event(
            phase=phase,
            stage=stage,
            invocation_id=invocation_id,
            source="headless",
            agent=agent,
            model=str(model),
            status=status,
            reported_cost=None,
            usage=_gemini_model_usage(details),
            artifact=artifact,
            item=f"{item_prefix}model:{model}",
            provider=normalized.get("provider"),
        )
        for model, details in sorted(stats_models.items())
    ]
    if len({event["event_id"] for event in events}) != len(events):
        return None
    if not _usage_totals_match(events, normalized):
        return None
    cost_event = _normalized_cost_only_event(normalized)
    return [*events, *([cost_event] if cost_event is not None else [])]


def parse_headless_log(
    path: Path,
    *,
    phase: str,
    stage: str,
    invocation_id: str,
    agent: str,
    status: str,
) -> list[dict[str, Any]]:
    """Parse one Headless JSONL invocation and retain Headless cost provenance."""
    records = _read_json_objects(path)
    artifact = str(path)
    normalized = _normalized_headless_event(
        records,
        phase=phase,
        stage=stage,
        invocation_id=invocation_id,
        agent=agent,
        status=status,
        artifact=artifact,
    )
    if normalized is not None:
        if "billing" in normalized:
            # Count each route once; a missing first attempt must not erase paid usage.
            return _billing_headless_events(normalized, records) or [normalized]
        try:
            multi_model = _multi_model_headless_events(
                records,
                normalized,
                phase=phase,
                stage=stage,
                invocation_id=invocation_id,
                agent=agent,
                status=status,
                artifact=artifact,
            )
        except CostTrackingError:
            multi_model = None
        if multi_model is not None:
            return multi_model
        return [normalized]
    final = next((record for record in reversed(records) if record.get("type") == "result"), None)
    if final is not None:
        model_usage = final.get("modelUsage")
        if isinstance(model_usage, Mapping) and model_usage:
            has_complete_model_cost = all(
                isinstance(details, Mapping) and details.get("costUSD") is not None
                for details in model_usage.values()
            )
            if not has_complete_model_cost and final.get("total_cost_usd") is not None:
                raw_usage = final.get("usage") or final.get("stats")
                return [
                    _event(
                        phase=phase,
                        stage=stage,
                        invocation_id=invocation_id,
                        source="headless",
                        agent=agent,
                        model=str(final.get("model") or agent),
                        status=status,
                        reported_cost=final.get("total_cost_usd"),
                        usage=_usage_from_mapping(raw_usage),
                        artifact=artifact,
                        item="result",
                    )
                ]
            events = []
            for model, details in sorted(model_usage.items()):
                details_map = details if isinstance(details, Mapping) else {}
                cost = details_map.get("costUSD")
                events.append(
                    _event(
                        phase=phase,
                        stage=stage,
                        invocation_id=invocation_id,
                        source="headless",
                        agent=agent,
                        model=str(model),
                        status=status,
                        reported_cost=cost,
                        usage=_usage_from_mapping(details_map),
                        artifact=artifact,
                        item=f"model:{model}",
                    )
                )
            return events
        stats = final.get("stats")
        stats_models = stats.get("models") if isinstance(stats, Mapping) else None
        if isinstance(stats_models, Mapping) and stats_models:
            events = []
            single_model = len(stats_models) == 1
            top_usage = _usage_from_mapping(stats)
            for model, details in sorted(stats_models.items()):
                usage = _usage_from_mapping(details)
                if single_model:
                    usage = {**top_usage, **usage}
                events.append(
                    _event(
                        phase=phase,
                        stage=stage,
                        invocation_id=invocation_id,
                        source="headless",
                        agent=agent,
                        model=str(model),
                        status=status,
                        reported_cost=None,
                        usage=usage,
                        artifact=artifact,
                        item=f"model:{model}",
                    )
                )
            return events
        raw_usage = final.get("usage") or final.get("stats")
        return [
            _event(
                phase=phase,
                stage=stage,
                invocation_id=invocation_id,
                source="headless",
                agent=agent,
                model=str(final.get("model") or agent),
                status=status,
                reported_cost=final.get("total_cost_usd"),
                usage=_usage_from_mapping(raw_usage),
                artifact=artifact,
                item="result",
            )
        ]

    completed = next(
        (record for record in reversed(records) if record.get("type") == "turn.completed"),
        None,
    )
    if completed is not None:
        return [
            _event(
                phase=phase,
                stage=stage,
                invocation_id=invocation_id,
                source="headless",
                agent=agent,
                model=str(completed.get("model") or agent),
                status=status,
                reported_cost=None,
                usage=_usage_from_mapping(completed.get("usage")),
                artifact=artifact,
                item="turn.completed",
            )
        ]

    seen: set[str] = set()
    usages: list[dict[str, int]] = []
    model = agent
    for index, record in enumerate(records):
        message = record.get("message")
        if not isinstance(message, Mapping):
            continue
        message_id = str(message.get("id") or record.get("id") or index)
        if message_id in seen:
            continue
        seen.add(message_id)
        model = str(message.get("model") or model)
        usage = _usage_from_mapping(message.get("usage"))
        if usage:
            usages.append(usage)
    return [
        _event(
            phase=phase,
            stage=stage,
            invocation_id=invocation_id,
            source="headless",
            agent=agent,
            model=model,
            status=status,
            reported_cost=None,
            usage=_sum_usage(usages),
            artifact=artifact,
            item="partial",
        )
    ]


def import_autoillustrate_metrics(
    path: Path,
    *,
    phase: str,
    stage: str,
    invocation_id: str,
    status: str,
) -> list[dict[str, Any]]:
    """Import Autoillustrate costs only when their provenance is trustworthy."""
    try:
        content = _read_bounded_bytes(path, max_bytes=_MAX_METRICS_BYTES, tail=False)
        payload = json.loads(content)
    except (CostTrackingError, json.JSONDecodeError) as exc:
        raise CostTrackingError(f"cannot read Autoillustrate metrics: {path.name}") from exc
    entries: object = []
    if isinstance(payload, Mapping):
        entries = (
            payload.get("costs") or payload.get("cost_entries") or payload.get("entries") or []
        )
        phases = payload.get("phases")
        if not entries and isinstance(phases, list):
            entries = [
                cost
                for phase_metrics in phases
                if isinstance(phase_metrics, Mapping)
                for cost in phase_metrics.get("costs", [])
                if isinstance(cost, Mapping)
            ]
    if not isinstance(entries, list):
        raise CostTrackingError("Autoillustrate cost entries must be a list")
    events: list[dict[str, Any]] = []
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            continue
        usage = _usage_from_mapping(raw.get("usage"))
        if raw.get("images") is not None:
            usage.update(_usage_from_mapping({"images": raw.get("images")}))
        source_name = str(raw.get("source") or "unknown")
        recognized_source = source_name in {
            "headless:claude",
            "headless:codex",
            "headless:cursor",
            "headless:gemini",
            "headless:opencode",
            "headless:pi",
        }
        pricing_source = raw.get("pricing_source")
        pricing_status = raw.get("pricing_status")
        self_reported = (
            recognized_source
            and raw.get("estimated") is False
            and raw.get("basis") == "self-reported"
            and (
                (pricing_source is None and pricing_status is None)
                or (pricing_source == "native" and pricing_status == "native")
            )
        )
        estimated = (
            recognized_source
            and raw.get("estimated") is True
            and raw.get("basis") == "api-list-price-estimate"
            and pricing_source == "models.dev"
            and pricing_status == "priced"
        )
        raw_cost = raw.get("cost_usd")
        valid_cost: object | None = None
        if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
            try:
                _decimal(raw_cost)
            except CostTrackingError:
                pass
            else:
                valid_cost = raw_cost
        reported_cost = valid_cost if self_reported else None
        estimated_cost = valid_cost if estimated else None
        trusted_cost = reported_cost is not None or estimated_cost is not None
        cost_basis = None
        if estimated_cost is not None:
            cost_basis = "api-list-price-estimate"
        elif reported_cost is not None and pricing_source == "native":
            cost_basis = "native-reported"
        events.append(
            _event(
                phase=phase,
                stage=stage,
                invocation_id=invocation_id,
                source=f"autoillustrate:{source_name}",
                agent="autoillustrate",
                model=str(raw.get("model") or "unknown"),
                status=status,
                reported_cost=reported_cost,
                estimated_cost=estimated_cost,
                usage=usage,
                artifact=str(path),
                item=f"entry:{index}",
                cost_basis=cost_basis,
                pricing_source=pricing_source if trusted_cost else None,
                pricing_status=pricing_status if trusted_cost else None,
            )
        )
    return events


def preserve_autoillustrate_metrics(path: Path, root: Path, invocation_id: str) -> Path:
    """Retain raw Autoillustrate metrics before its temporary workspace is removed."""
    content = _read_bounded_bytes(path, max_bytes=_MAX_METRICS_BYTES, tail=False)
    name = hashlib.sha256(invocation_id.encode()).hexdigest() + ".json"
    return store_cost_artifact(
        root,
        category="autoillustrate",
        name=name,
        content=content,
    )


def _object_value(obj: object, name: str) -> object:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def record_api_response(
    root: Path,
    response: object,
    *,
    phase: str,
    stage: str,
    invocation_id: str,
    source: str,
    status: str = "completed",
    token_prices: Mapping[str, ApiTokenPrice] | None = None,
) -> dict[str, Any]:
    """Record response metadata only; never serialize response content or identifiers."""
    raw_usage = _object_value(response, "usage")
    usage = _usage_from_mapping(
        {
            "prompt_tokens": _object_value(raw_usage, "prompt_tokens"),
            "completion_tokens": _object_value(raw_usage, "completion_tokens"),
        }
    )
    prompt_details = _object_value(raw_usage, "prompt_tokens_details")
    completion_details = _object_value(raw_usage, "completion_tokens_details")
    details_usage = _usage_from_mapping(
        {
            "cached_input_tokens": _object_value(prompt_details, "cached_tokens"),
            "reasoning_tokens": _object_value(completion_details, "reasoning_tokens"),
        }
    )
    usage.update(details_usage)
    model = str(_object_value(response, "model") or "unknown")
    estimated_cost: Decimal | None = None
    cost_breakdown: dict[str, str] = {}
    price = token_prices.get(model) if token_prices is not None else None
    if price is not None:
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        cached_input_tokens = usage.get("cached_input_tokens", 0)
        if (
            input_tokens is not None
            and output_tokens is not None
            and cached_input_tokens <= input_tokens
        ):
            million = Decimal(1_000_000)
            input_cost = (
                Decimal(input_tokens - cached_input_tokens)
                * price.input_per_million
                / million
            )
            cache_read_cost = (
                Decimal(cached_input_tokens) * price.cached_input_per_million / million
            )
            output_cost = Decimal(output_tokens) * price.output_per_million / million
            estimated_cost = input_cost + cache_read_cost + output_cost
            cost_breakdown = {
                "input": _decimal_text(input_cost),
                "cache_read": _decimal_text(cache_read_cost),
                "output": _decimal_text(output_cost),
                "total": _decimal_text(estimated_cost),
            }
    event = _event(
        phase=phase,
        stage=stage,
        invocation_id=invocation_id,
        source=source,
        agent=source,
        model=model,
        status=status,
        reported_cost=None,
        estimated_cost=estimated_cost,
        usage=usage,
        artifact=None,
        item="response",
        cost_basis="api-list-price-estimate" if estimated_cost is not None else None,
        pricing_source=(
            price.pricing_source if price is not None and estimated_cost is not None else None
        ),
        pricing_status=(
            price.pricing_status if price is not None and estimated_cost is not None else None
        ),
        cost_breakdown=cost_breakdown,
    )
    return record_events(root, [event])


def record_api_failure(
    root: Path,
    *,
    phase: str,
    stage: str,
    invocation_id: str,
    source: str,
    model: str,
) -> dict[str, Any]:
    event = _event(
        phase=phase,
        stage=stage,
        invocation_id=invocation_id,
        source=source,
        agent=source,
        model=model,
        status="failed",
        reported_cost=None,
        usage={},
        artifact=None,
        item="failure",
    )
    return record_events(root, [event])


if __name__ == "__main__":
    from .cost_cli import main

    raise SystemExit(main())
