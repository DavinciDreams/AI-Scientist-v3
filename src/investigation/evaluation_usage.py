"""Host-measured compute use, separate from candidate-reported diagnostics."""

from __future__ import annotations

import math
import json
import logging
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from src.investigation.safe_staging import atomic_write_bytes


def compute_usage(
    *,
    wall_clock_limit_seconds: float,
    elapsed_seconds: float,
    max_rpc_calls: int,
    rpc_calls: int | None,
    outcome: str,
) -> dict[str, Any]:
    result = {
        "wall_clock_limit_seconds": wall_clock_limit_seconds,
        "elapsed_seconds": elapsed_seconds,
        "max_rpc_calls": max_rpc_calls,
        "rpc_calls": rpc_calls,
        "outcome": outcome,
        "objective_counts": "advisory",
    }
    validate_compute_usage(result)
    return result


def validate_compute_usage(value: Any) -> None:
    fields = {
        "wall_clock_limit_seconds",
        "elapsed_seconds",
        "max_rpc_calls",
        "rpc_calls",
        "outcome",
        "objective_counts",
    }
    if not isinstance(value, dict) or value.keys() != fields:
        raise ValueError("compute usage fields are invalid")
    for field in ("wall_clock_limit_seconds", "elapsed_seconds"):
        number = value[field]
        try:
            valid = (
                not isinstance(number, bool)
                and isinstance(number, (int, float))
                and math.isfinite(number)
                and number >= 0
            )
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"compute usage {field} must be finite and nonnegative")
    if value["wall_clock_limit_seconds"] == 0:
        raise ValueError("compute usage time limit must be positive")
    if type(value["max_rpc_calls"]) is not int or value["max_rpc_calls"] < 1:
        raise ValueError("compute usage RPC limit must be positive")
    calls = value["rpc_calls"]
    if calls is not None and (type(calls) is not int or calls < 0):
        raise ValueError("compute usage RPC calls must be nonnegative or unknown")
    if not isinstance(value["outcome"], str) or value["outcome"] not in {
        "completed",
        "timeout",
        "rpc_limit",
        "failed",
    }:
        raise ValueError("compute usage outcome is invalid")
    if value["outcome"] == "completed" and calls is not None and calls > value["max_rpc_calls"]:
        raise ValueError("completed compute usage exceeds RPC limit")
    if value["objective_counts"] != "advisory":
        raise ValueError("internal objective counts cannot certify hard budget compliance")


@contextmanager
def record_compute_failure(
    root: Path,
    *,
    wall_clock_limit_seconds: float,
    started_at: float,
    max_rpc_calls: int,
    broker: Any = None,
) -> Iterator[None]:
    """Retain host-only failure diagnostics outside successful evaluator outputs."""
    try:
        yield
    except BaseException as exc:
        calls = getattr(broker, "calls", None)
        outcome = "failed"
        if isinstance(calls, int) and calls > max_rpc_calls:
            outcome = "rpc_limit"
        elif isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
            outcome = "timeout"
        elif time.monotonic() - started_at >= wall_clock_limit_seconds:
            outcome = "timeout"
        try:
            payload = compute_usage(
                wall_clock_limit_seconds=wall_clock_limit_seconds,
                elapsed_seconds=max(0.0, time.monotonic() - started_at),
                max_rpc_calls=max_rpc_calls,
                rpc_calls=calls,
                outcome=outcome,
            )
            atomic_write_bytes(root, "evaluation_failure.json", json.dumps(payload).encode())
        except (OSError, ValueError):
            logging.getLogger(__name__).warning("Could not retain compute failure diagnostics")
        raise
