"""Typed dataclasses for investigation step inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunAttemptResult:
    output: str
    exit_code: int
    experiment_dir: Path | None
    task_failure_category: str | None = None
    infrastructure_failure_category: str | None = None


@dataclass
class IdeaGenerationResult:
    output: str
    exit_code: int
    idea_dir: Path | None
    infrastructure_failure_category: str | None = None


@dataclass
class TaskDraftResult:
    output: str
    exit_code: int
    draft_dir: Path
    infrastructure_failure_category: str | None = None


@dataclass
class PaperGenerationResult:
    output: str
    exit_code: int
    paper_dir: Path | None
    ai_review_status: str | None = None
    ai_review_path: Path | None = None


@dataclass
class ParallelWorkerPlan:
    worker_index: int
    assigned_cycles: int
    cycle_start: int
    cycle_end: int
    results_dir: Path
    command: list[str]
    cycle_map: dict[int, int] | None = None
    lease_index: int | None = None
    log_path: Path | None = None
    environment: dict[str, str] | None = None


@dataclass
class ParallelWorkerResult:
    plan: ParallelWorkerPlan
    exit_code: int
    log_path: Path
    elapsed_seconds: float
    output_tail: str
    suppressed_output_lines: int = 0
