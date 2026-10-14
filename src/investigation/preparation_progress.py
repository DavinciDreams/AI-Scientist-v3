"""Host-owned, diagnostic progress for task runtime preparation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import time
from typing import Any, Iterator

from .runtime import format_elapsed, log_event

_ENABLED: ContextVar[bool] = ContextVar("preparation_progress", default=False)
_LABELS = {
    "resources": "Validating resource inventory",
    "contract": "Checking task contract",
    "search_smoke": "Running reduced search smoke test",
    "final_smoke": "Running synthetic final smoke test",
    "production_entry": "Checking production entry points",
    "runtime_cache": "Checking prepared runtime cache",
    "runtime_capacity": "Waiting for runtime build capacity",
    "runtime_create": "Creating isolated Python runtime",
    "tooling_build": "Installing runtime build tools",
    "tooling_install": "Installing evaluator tooling",
    "dependency_install": "Installing pinned task dependencies",
    "runtime_publish": "Publishing prepared runtime cache",
    "runtime_copy": "Copying prepared runtime into task",
    "baseline_evaluation": "Evaluating starting candidate",
    "baseline_reuse": "Reusing validated baseline",
}


@dataclass
class PreparationStep:
    state: str = "completed"

    def fail(self) -> None:
        self.state = "failed"


def _emit(step: str, state: str, **fields: Any) -> None:
    try:
        log_event("preparation_step", step=step, label=_LABELS[step], state=state, **fields)
    except Exception:
        # Progress is diagnostic; a broken terminal must not decide task readiness.
        pass


@contextmanager
def preparation_progress() -> Iterator[None]:
    token = _ENABLED.set(True)
    try:
        yield
    finally:
        _ENABLED.reset(token)


@contextmanager
def preparation_step(step: str) -> Iterator[PreparationStep]:
    if step not in _LABELS:
        raise ValueError("Unknown preparation step")
    progress = PreparationStep()
    if not _ENABLED.get():
        yield progress
        return
    started = time.monotonic()
    _emit(step, "started")
    try:
        yield progress
    except BaseException:
        progress.fail()
        raise
    finally:
        elapsed = max(0.0, time.monotonic() - started)
        _emit(step, progress.state, elapsed_seconds=round(elapsed, 1), elapsed=format_elapsed(elapsed))
