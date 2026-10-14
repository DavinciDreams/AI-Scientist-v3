"""Collect production-search compute observations without evaluator self-reports."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
import time
from typing import Any, Callable, ParamSpec, Protocol, TypeVar, cast

from src.investigation.evaluation_usage import compute_usage
from src.investigation.task_contract import TaskContract

P = ParamSpec("P")
R = TypeVar("R")


class BrokerObservation(Protocol):
    @property
    def calls(self) -> int: ...


_report_observations: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "qualification_report_observations",
    default=None,
)
_baseline_broker: ContextVar[BrokerObservation | None] = ContextVar(
    "qualification_baseline_broker",
    default=None,
)


def observe_qualification_broker(broker: BrokerObservation) -> None:
    """Keep the host broker until teardown finishes so terminal calls are counted."""
    if _report_observations.get() is not None:
        _baseline_broker.set(broker)


def retain_qualification_usage(function: Callable[P, R]) -> Callable[P, R]:
    """Attach observations to a runtime report; legacy/cache-only reports stay unknown."""

    @wraps(function)
    def report(*args: P.args, **kwargs: P.kwargs) -> R:
        observations: list[dict[str, Any]] = []
        token = _report_observations.set(observations)
        try:
            result = function(*args, **kwargs)
            return replace(
                cast(Any, result), compute_usage=observations[-1] if observations else None
            )
        finally:
            _report_observations.reset(token)

    return report


def measure_qualification_usage(function: Callable[P, R]) -> Callable[P, R]:
    """Measure baseline setup, evaluation and cleanup inside its effective deadline."""

    @wraps(function)
    def measure(*args: P.args, **kwargs: P.kwargs) -> R:
        observations = _report_observations.get()
        if observations is None:
            return function(*args, **kwargs)
        contract = cast(TaskContract, kwargs["contract"])
        started = time.monotonic()
        limit = min(
            contract.compute.search.wall_clock_seconds,
            max(0.001, float(cast(float, kwargs["deadline"])) - started),
        )
        broker_token = _baseline_broker.set(None)
        outcome = "failed"
        try:
            result = function(*args, **kwargs)
            if getattr(result, "status", None) == "succeeded":
                outcome = "completed"
            return result
        except TimeoutError:
            outcome = "timeout"
            raise
        finally:
            elapsed = max(0.0, time.monotonic() - started)
            broker = _baseline_broker.get()
            calls = broker.calls if broker is not None else None
            _baseline_broker.reset(broker_token)
            if calls is not None and calls > contract.compute.search.max_candidate_rpc_calls:
                outcome = "rpc_limit"
            elif elapsed >= limit:
                outcome = "timeout"
            observations.append(
                compute_usage(
                    wall_clock_limit_seconds=limit,
                    elapsed_seconds=elapsed,
                    max_rpc_calls=contract.compute.search.max_candidate_rpc_calls,
                    rpc_calls=calls,
                    outcome=outcome,
                )
            )

    return measure
