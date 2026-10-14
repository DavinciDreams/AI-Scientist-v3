"""Bounded, diagnostic-only Headless billing provenance; never retry authority."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

_ROUTES = {"subscription", "openai-api", "bedrock", "native"}
_REASONS = {"subscription-quota", "subscription-model-unsupported"}
_COUNTS = {
    "inputTokens", "cacheReadTokens", "cacheWriteTokens", "outputTokens",
    "reasoningOutputTokens", "totalTokens",
}
_ENUMS: dict[str, set[str | None]] = {
    "usageStatus": {"reported", "missing"},
    "costBasis": {"native-reported", "api-list-price-estimate", None},
    "pricingSource": {"native", "models.dev", None},
    "pricingStatus": {"native", "priced", "missing"},
}
_LABEL = re.compile(r"[A-Za-z0-9._:/-]{1,240}\Z")


def _number(value: object, maximum: int) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= maximum
        or not math.isfinite(value)
    ):
        raise ValueError("invalid billing usage number")
    return value


def _usage(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("invalid billing usage")
    clean: dict[str, Any] = {}
    for field in ("agent", "model", "provider"):
        value = raw.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not _LABEL.fullmatch(value):
            raise ValueError("invalid billing usage label")
        clean[field] = value
    for field, choices in _ENUMS.items():
        value = raw.get(field)
        if not isinstance(value, (str, type(None))) or value not in choices:
            raise ValueError("invalid billing usage provenance")
        clean[field] = value
    for field in _COUNTS:
        if field in raw and raw[field] is not None:
            value = _number(raw[field], 10**15)
            if int(value) != value:
                raise ValueError("invalid billing token count")
            clean[field] = int(value)
    cost = raw.get("cost")
    if cost is None:
        clean["cost"] = None
    elif isinstance(cost, Mapping):
        clean["cost"] = {
            key: None if cost[key] is None else _number(cost[key], 10**9)
            for key in ("input", "cacheRead", "cacheWrite", "output", "total") if key in cost
        }
    else:
        raise ValueError("invalid billing cost")
    return clean


def validated_billing_metadata(raw: object) -> dict[str, Any]:
    """Retain known bounded fields; reject malformed metadata without copying arbitrary text."""
    if not isinstance(raw, Mapping):
        raise ValueError("invalid billing metadata")
    attempts = raw.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 5:
        raise ValueError("invalid billing attempt count")
    clean = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("invalid billing attempt")
        route, reason = attempt.get("route"), attempt.get("reason")
        if not isinstance(route, str) or route not in _ROUTES:
            raise ValueError("invalid billing route")
        record = {"route": route, "usage": _usage(attempt.get("usage"))}
        if reason is not None:
            if not isinstance(reason, str) or reason not in _REASONS:
                raise ValueError("invalid billing reason")
            record["reason"] = reason
        retry_reason = attempt.get("retryReason")
        if retry_reason is not None:
            if retry_reason != "model-capacity":
                raise ValueError("invalid billing retry reason")
            record["retryReason"] = retry_reason
        clean.append(record)
    return {"attempts": clean}
