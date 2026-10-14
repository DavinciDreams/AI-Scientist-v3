"""Parallel pruning: round-based campaigns, per-worker seeds, archives, resume and task failure."""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest

from src.investigation.config import (
    ConfigValidationError,
    ContainerConfig,
    ExecutionConfig,
    InvestigationConfig,
    validate,
)
from src.investigation.defaults import split_cycle_budget
from src.investigation.models import ParallelWorkerPlan, ParallelWorkerResult
from src.investigation.parallel_helpers import (
    _restore_parallel_root_state_for_resume,
    aggregate_parallel_worker_outputs,
    build_parallel_campaign_summary,
    build_parallel_worker_command,
    run_parallel_campaign,
)
from src.orchestrator import state as state_store
from tests.program_testutils import make_program_dir

CONFIRM_PRUNE_WARNING = "src.investigation.parallel_helpers.confirm_prune_warning"


def _NO_LOG(*args: Any, **kwargs: Any) -> None:
    return None


def _worker_should_not_run(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
    raise AssertionError("worker should not run")


_REAL_AGGREGATE = partial(aggregate_parallel_worker_outputs, log_event_fn=_NO_LOG)


def _cfg(
    max_cycles: int,
    workers: int,
    prune_n: int,
    *,
    resume: bool = False,
    parallel_fail_fast: bool = False,
    isolate_worker_memory: bool = False,
    container: bool = False,
) -> InvestigationConfig:
    return InvestigationConfig(
        execution=ExecutionConfig(
            max_idea_cycles=max_cycles,
            workers=workers,
            prune_every_n_cycles=prune_n,
            resume=resume,
            parallel_fail_fast=parallel_fail_fast,
            isolate_worker_memory=isolate_worker_memory,
        ),
        container=ContainerConfig(enabled=container),
    )


def _even_split(total: int, workers: int) -> list[int]:
    return [total // workers + (1 if i < total % workers else 0) for i in range(workers)]


@dataclass
class Campaign:
    root: Path
    task: Path
    seed: Path
    results: Path
    memory: Path


def _campaign(root: Path, *, seed_file: bool = False) -> Campaign:
    """Create task/seed/results/memory directories for a campaign run (root is created if missing)."""
    root.mkdir(parents=True, exist_ok=True)
    task = root / "task"
    task.mkdir()
    if seed_file:
        seed = root / "program.py"
        seed.write_text("# seed", encoding="utf-8")
    else:
        seed = task / "program_0"
        make_program_dir(seed, content="# seed\n")
    results = root / "results"
    results.mkdir()
    memory = root / "mem"
    memory.mkdir()
    state_store.ensure_global_memory_files(memory)
    return Campaign(root, task, seed, results, memory)


def _run_campaign(cfg: InvestigationConfig, campaign: Campaign, **overrides: Any) -> int:
    kwargs: dict[str, Any] = dict(
        cfg=cfg,
        task_dir=campaign.task,
        results_dir=campaign.results,
        global_memory_dir=campaign.memory,
        instruct_path=campaign.root / "instruct.md",
        seed_program_dir=campaign.seed,
        split_cycle_budget_fn=_even_split,
        build_parallel_worker_command_fn=lambda **kw: ["echo", "test"],
        run_parallel_worker_fn=_worker_success,
        aggregate_parallel_worker_outputs_fn=_aggregate,
        log_event_fn=_NO_LOG,
        format_elapsed_fn=lambda seconds: f"{seconds:.1f}s",
        terminate_all_active_processes_fn=lambda **kw: [],
        stop_container_session_fn=lambda *args, **kw: False,
        build_container_session_name_fn=lambda **kw: "test",
    )
    kwargs.update(overrides)
    return run_parallel_campaign(**kwargs)


def _build_recorder(calls: list[dict]) -> Callable[..., list[str]]:
    def record(**kw: Any) -> list[str]:
        calls.append(kw)
        return ["echo", "test"]

    return record


def _result(plan: ParallelWorkerPlan, exit_code: int = 0, log_path: Path | None = None) -> ParallelWorkerResult:
    return ParallelWorkerResult(
        plan=plan,
        exit_code=exit_code,
        log_path=log_path or plan.results_dir / "log",
        elapsed_seconds=1.0,
        output_tail="",
    )


def _write_best(
    results_dir: Path, program: Path, *, score: float | None, correct: bool, attempt_id: str, key: str = "program_dir"
) -> None:
    state_store.write_json(
        results_dir / state_store.BEST_MANIFEST_FILE,
        {key: str(program), "combined_score": score, "correct": correct, "source_attempt_id": attempt_id},
    )


def _append_history(results_dir: Path, idea_id: str, attempt_id: str, cycle_index: int | None = 1) -> None:
    state_store.append_jsonl(results_dir / state_store.IDEAS_HISTORY_FILE, {"idea_id": idea_id, "cycle_index": cycle_index})
    state_store.append_jsonl(
        results_dir / state_store.ATTEMPTS_HISTORY_FILE,
        {"attempt_id": attempt_id, "idea_id": idea_id, "cycle_index": cycle_index},
    )


def _worker_success(
    plan: ParallelWorkerPlan,
    *,
    program: str = "program_1",
    content: str = "# best\n",
    score: float = 1.0,
    key: str = "program_dir",
) -> ParallelWorkerResult:
    """Worker that writes a valid best program (dir or file manifest) plus matching history."""
    plan.results_dir.mkdir(parents=True, exist_ok=True)
    state_store.ensure_histories(plan.results_dir, include_global_memory=False)
    program_dir = plan.results_dir / program
    program_path = make_program_dir(program_dir, content=content)
    _write_best(
        plan.results_dir, program_dir if key == "program_dir" else program_path,
        score=score, correct=True, attempt_id=f"att_{plan.worker_index}", key=key,
    )
    _append_history(plan.results_dir, f"idea_{plan.worker_index}", f"att_{plan.worker_index}")
    return _result(plan)


def _worker_plain(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
    plan.results_dir.mkdir(parents=True, exist_ok=True)
    state_store.ensure_histories(plan.results_dir, include_global_memory=False)
    return _result(plan)


def _worker_file_program(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
    plan.results_dir.mkdir(parents=True, exist_ok=True)
    state_store.ensure_histories(plan.results_dir, include_global_memory=False)
    program_file = plan.results_dir / "best_program.py"
    program_file.write_text("# best", encoding="utf-8")
    _write_best(
        plan.results_dir, program_file, score=1.0, correct=True,
        attempt_id=f"att_{plan.worker_index}", key="program_path",
    )
    _append_history(plan.results_dir, f"idea_{plan.worker_index}", f"att_{plan.worker_index}")
    return _result(plan)


def _barrier_worker(barrier: threading.Barrier, **success_kwargs: Any) -> Callable[[ParallelWorkerPlan], ParallelWorkerResult]:
    """`_worker_success` that makes both workers' first leases rendezvous before returning."""

    def run(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
        if plan.lease_index == 1:
            barrier.wait(timeout=1)
        return _worker_success(plan, **success_kwargs)

    return run


def _flaky_worker(completed_cycles: list[int], *, histories: bool = False) -> Callable[[ParallelWorkerPlan], ParallelWorkerResult]:
    """Worker whose first cycle fails while the others wait for the failure before succeeding."""
    failure_seen = threading.Event()

    def run(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
        plan.results_dir.mkdir(parents=True, exist_ok=True)
        if histories:
            state_store.ensure_histories(plan.results_dir, include_global_memory=False)
        log_path = plan.log_path or plan.results_dir / "log"
        log_path.write_text("ok\n", encoding="utf-8")
        completed_cycles.append(plan.cycle_start)
        exit_code = 1 if plan.cycle_start == 1 else 0
        if exit_code:
            failure_seen.set()
        else:
            failure_seen.wait(timeout=1.0)
        result = _result(plan, exit_code, log_path)
        result.elapsed_seconds = 0.01
        return result

    return run


def _aggregate(
    *, results_dir: Path, seed_program_dir: Path, worker_results: list[ParallelWorkerResult],
    rounds: list[list[ParallelWorkerResult]] | None = None,
) -> dict[str, Any]:
    """Mock aggregate: records each round's results, ensures root histories, counts failures."""
    if rounds is not None:
        rounds.append(list(worker_results))
    state_store.ensure_histories(results_dir, include_global_memory=False)
    return {
        "failed_worker_count": sum(1 for result in worker_results if result.exit_code != 0),
        "idea_count": 0,
        "attempt_count": 0,
    }


def _make_worker_result(
    results_dir: Path, worker_index: int, *, best_score: float | None = None,
    idea_id: str = "idea_1", attempt_id: str = "attempt_1",
) -> ParallelWorkerResult:
    """ParallelWorkerResult with idea/attempt history, a best manifest and a dummy program file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    state_store.append_jsonl(
        results_dir / state_store.IDEAS_HISTORY_FILE,
        {"idea_id": idea_id, "cycle_index": 1, "title": f"Test idea from worker {worker_index}",
         "timestamp": "2026-01-01T00:00:00+00:00"},
    )
    state_store.append_jsonl(
        results_dir / state_store.ATTEMPTS_HISTORY_FILE,
        {"attempt_id": attempt_id, "idea_id": idea_id, "cycle_index": 1, "attempt_number": 1,
         "timestamp": "2026-01-01T00:00:01+00:00"},
    )
    _write_best(
        results_dir, results_dir / "program.py", score=best_score, correct=best_score is not None,
        attempt_id=attempt_id, key="program_path",
    )
    (results_dir / "program.py").write_text("# test", encoding="utf-8")
    plan = ParallelWorkerPlan(
        worker_index=worker_index, assigned_cycles=2, cycle_start=1, cycle_end=2,
        results_dir=results_dir, command=["echo", "test"],
    )
    result = _result(plan, log_path=results_dir / "parallel_worker.log")
    result.elapsed_seconds = 10.0
    return result


def _results_and_memory(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    results, memory = root / "results", root / "memory"
    results.mkdir()
    memory.mkdir()
    state_store.ensure_global_memory_files(memory)
    return results, memory


def _seed(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    seed_program_dir = root / "seed"
    make_program_dir(seed_program_dir, content="# seed\n")
    return seed_program_dir


def _calls_in_round(build_calls: list[dict], round_name: str) -> list[dict]:
    return [call for call in build_calls if round_name in str(call["results_dir"])]


# ---------------------------------------------------------------------------
# Worker command / config validation
# ---------------------------------------------------------------------------


PRUNE_CONFIG_CASES = [
    ("prune-not-multiple-of-workers", 6, 4, False, "must be a multiple"),
    ("prune-multiple-of-workers", 8, 4, False, None),
    ("prune-zero", 0, 4, False, None),
    ("single-worker", 5, 1, False, None),
    ("parallel-resume-needs-pruning", 0, 2, True, "parallel resume requires"),
    ("parallel-resume-with-pruning", 10, 2, True, None),
    ("parallel-resume-needs-two-rounds", 20, 2, True, "parallel resume requires"),
    ("single-worker-resume-needs-two-rounds", 20, 1, True, "pruned resume requires"),
]


def test_worker_command_isolation_flag_and_prune_config_validation() -> None:
    # the worker command forwards the memory isolation value
    for enabled in (False, True):
        command = build_parallel_worker_command(
            cfg=InvestigationConfig(execution=ExecutionConfig(isolate_worker_memory=enabled)),
            task_dir=Path("/tmp/task"),
            instruct_path=Path("/tmp/instruct.md"),
            results_dir=Path("/tmp/results/worker_01"),
            global_memory_dir=Path("/tmp/results/worker_01"),
            assigned_cycles=1,
        )

        assert f"execution.isolate_worker_memory={'true' if enabled else 'false'}" in command, enabled

    # prune / resume configuration validation
    for case, prune_n, workers, resume, error in PRUNE_CONFIG_CASES:
        cfg = _cfg(20, workers, prune_n, resume=resume)
        if error is None:
            validate(cfg)
            continue
        with pytest.raises(ConfigValidationError) as exc_info:
            validate(cfg)
        assert error in exc_info.value.message, case


# ---------------------------------------------------------------------------
# archive_for_parallel_prune
# ---------------------------------------------------------------------------


def test_archive_for_parallel_prune_resets_histories_conclusions_and_round_summaries(tmp_path: Path) -> None:
    # archives histories and memory, then resets them with a prune marker
    results, memory = _results_and_memory(tmp_path / "histories")
    state_store.ensure_histories(results)
    state_store.append_jsonl(results / state_store.IDEAS_HISTORY_FILE, {"idea_id": "idea_1", "cycle_index": 1})
    state_store.append_jsonl(results / state_store.ATTEMPTS_HISTORY_FILE, {"attempt_id": "att_1", "cycle_index": 1})
    (memory / state_store.GLOBAL_MEMORY_MD_FILE).write_text("# Some memory content\n", encoding="utf-8")

    state_store.archive_for_parallel_prune(results, memory, 0)

    archive_dir = results / "_pruned" / "round_000"
    assert archive_dir.is_dir()
    for name in (
        state_store.IDEAS_HISTORY_FILE,
        state_store.ATTEMPTS_HISTORY_FILE,
        state_store.GLOBAL_MEMORY_MD_FILE,
        state_store.GLOBAL_MEMORY_JSONL_FILE,
    ):
        assert (archive_dir / name).is_file()
    assert (results / state_store.IDEAS_HISTORY_FILE).read_text() == ""
    assert (results / state_store.ATTEMPTS_HISTORY_FILE).read_text() == ""
    memory_md = (memory / state_store.GLOBAL_MEMORY_MD_FILE).read_text()
    assert "Round 1" in memory_md
    assert "round_000" in memory_md
    # memory.jsonl holds only the prune marker (complete context reset).
    lines = [
        line for line in (memory / state_store.GLOBAL_MEMORY_JSONL_FILE).read_text().strip().split("\n")
        if line.strip()
    ]
    assert len(lines) == 1
    marker = json.loads(lines[0])
    assert marker["step"] == "parallel_prune"
    assert marker["round"] == 1

    # archives and resets conclusions
    results, memory = _results_and_memory(tmp_path / "conclusions")
    state_store.ensure_histories(results)
    (memory / state_store.CONCLUSIONS_FILE).write_text("Promising lead: X.\n", encoding="utf-8")
    (memory / state_store.CONCLUSIONS_HISTORY_FILE).write_text(
        json.dumps({"cycle_index": 1, "new": "Promising lead: X."}) + "\n", encoding="utf-8"
    )
    (memory / state_store.OTHER_WORKERS_CONCLUSIONS_FILE).write_text("Worker 2 conclusions.\n", encoding="utf-8")

    state_store.archive_for_parallel_prune(results, memory, 0)

    archive_subdir = results / "_pruned" / "round_000" / state_store.SHARED_CONCLUSIONS_SUBDIR
    assert archive_subdir.is_dir()
    for name in (
        state_store.CONCLUSIONS_FILE,
        state_store.CONCLUSIONS_HISTORY_FILE,
        state_store.OTHER_WORKERS_CONCLUSIONS_FILE,
    ):
        assert (archive_subdir / name).is_file()
    assert "No conclusions yet" in (memory / state_store.CONCLUSIONS_FILE).read_text(encoding="utf-8")
    assert "No conclusions from other workers yet" in (
        memory / state_store.OTHER_WORKERS_CONCLUSIONS_FILE
    ).read_text(encoding="utf-8")
    assert not (memory / state_store.CONCLUSIONS_HISTORY_FILE).exists()

    # archives the round summary files
    results, memory = _results_and_memory(tmp_path / "summaries")
    state_store.ensure_histories(results)
    state_store.write_json(results / state_store.OUTER_STATE_FILE, {"current_cycle": 4, "best_combined_score": 1.25})
    state_store.write_json(
        results / state_store.BEST_MANIFEST_FILE,
        {"program_dir": "/tmp/program_1", "combined_score": 1.25, "correct": True},
    )
    state_store.write_json(
        results / "parallel_summary.json", {"idea_count": 2, "attempt_count": 3, "best_combined_score": 1.25}
    )

    state_store.archive_for_parallel_prune(results, memory, 0)

    archive_dir = results / "_pruned" / "round_000"
    assert (archive_dir / state_store.OUTER_STATE_FILE).is_file()
    assert (archive_dir / state_store.BEST_MANIFEST_FILE).is_file()
    assert (archive_dir / "parallel_summary.json").is_file()


def test_archive_for_parallel_prune_respects_umask_and_replaces_symlinks_safely(tmp_path: Path) -> None:
    # new reset memory files respect a restrictive umask
    memory = tmp_path / "memory"
    previous_umask = os.umask(0o077)
    try:
        state_store.reset_parallel_global_memory_for_round(memory, round_index=0)
    finally:
        os.umask(previous_umask)

    for name in (state_store.GLOBAL_MEMORY_MD_FILE, state_store.GLOBAL_MEMORY_JSONL_FILE):
        assert stat.S_IMODE((memory / name).stat().st_mode) == 0o600

    # local memory / conclusion / history symlinks are replaced without touching their targets
    root = tmp_path / "symlinks"
    results, memory = _results_and_memory(root)
    targets = {name: root / f"{name}-target" for name in ("memory", "conclusions", "history")}
    for name, target in targets.items():
        target.write_text(f"keep {name} target\n", encoding="utf-8")
    memory_path = memory / state_store.GLOBAL_MEMORY_MD_FILE
    conclusions_path = memory / state_store.CONCLUSIONS_FILE
    history_path = results / state_store.IDEAS_HISTORY_FILE
    memory_path.unlink()
    memory_path.symlink_to(targets["memory"])
    conclusions_path.symlink_to(targets["conclusions"])
    history_path.symlink_to(targets["history"])

    state_store.archive_for_parallel_prune(results, memory, 0)

    for name, target in targets.items():
        assert target.read_text(encoding="utf-8") == f"keep {name} target\n"
    assert not memory_path.is_symlink()
    assert not conclusions_path.is_symlink()
    assert not history_path.is_symlink()
    assert history_path.read_text(encoding="utf-8") == ""
    assert "Round 1" in memory_path.read_text(encoding="utf-8")
    assert "No conclusions yet" in conclusions_path.read_text(encoding="utf-8")
    archive_dir = results / "_pruned" / "round_000"
    assert not (archive_dir / state_store.GLOBAL_MEMORY_MD_FILE).exists()
    assert not (archive_dir / state_store.SHARED_CONCLUSIONS_SUBDIR / state_store.CONCLUSIONS_FILE).exists()


def test_archive_shared_conclusions_moves_files_and_rejects_symlinked_dirs(tmp_path: Path) -> None:
    # moves worker conclusion files into the archive subdir
    shared_dir, archive_dir = tmp_path / "shared", tmp_path / "archive"
    shared_dir.mkdir()
    target = state_store.shared_conclusions_dir(shared_dir)
    target.mkdir(parents=True)
    (target / "worker_01.md").write_text("W1.\n", encoding="utf-8")
    (target / "worker_02.md").write_text("W2.\n", encoding="utf-8")

    state_store.archive_shared_conclusions_for_prune(shared_dir, archive_dir)

    shared_archive = archive_dir / f"shared_{state_store.SHARED_CONCLUSIONS_SUBDIR}"
    assert (shared_archive / "worker_01.md").is_file()
    assert (shared_archive / "worker_02.md").is_file()
    assert list(target.iterdir()) == []

    # a symlinked shared conclusions directory is rejected without touching the outside
    source_root = tmp_path / "symlinked-source"
    source_root.mkdir()
    shared_dir, archive_dir, outside = source_root / "shared", source_root / "archive", source_root / "outside"
    shared_dir.mkdir()
    outside.mkdir()
    outside_file = outside / "secret.md"
    outside_file.write_text("keep outside\n", encoding="utf-8")
    state_store.shared_conclusions_dir(shared_dir).symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        state_store.archive_shared_conclusions_for_prune(shared_dir, archive_dir)

    assert outside_file.read_text(encoding="utf-8") == "keep outside\n"
    assert not (archive_dir / "shared_conclusions" / "secret.md").exists()

    # a symlinked archive destination is rejected
    dest_root = tmp_path / "symlinked-destination"
    dest_root.mkdir()
    shared_dir, archive_dir, outside = dest_root / "shared", dest_root / "archive", dest_root / "outside"
    source_dir = state_store.shared_conclusions_dir(shared_dir)
    source_dir.mkdir(parents=True)
    source = source_dir / "worker_01.md"
    source.write_text("worker conclusion\n", encoding="utf-8")
    archive_dir.mkdir()
    outside.mkdir()
    (archive_dir / "shared_conclusions").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        state_store.archive_shared_conclusions_for_prune(shared_dir, archive_dir)

    assert list(outside.iterdir()) == []
    assert source.read_text(encoding="utf-8") == "worker conclusion\n"


# ---------------------------------------------------------------------------
# Worker aggregation / campaign summary
# ---------------------------------------------------------------------------


def test_idea_generation_failures_counted_once_across_resume_restore_and_prune_archives(tmp_path: Path) -> None:
    # resume restore does not duplicate archived idea failures
    restore_root = tmp_path / "restore"
    results_dir = restore_root / "results"
    archive_dir = results_dir / "_pruned" / "round_000"
    archive_dir.mkdir(parents=True)
    seed_program_dir = _seed(restore_root)
    state_store.append_jsonl(
        archive_dir / state_store.IDEA_GENERATION_FAILURES_FILE,
        {"cycle_index": 1, "generation_attempt": 2, "max_attempts": 2},
    )

    _restore_parallel_root_state_for_resume(
        results_dir=results_dir, global_memory_dir=results_dir, seed_program_dir=seed_program_dir,
        next_round_index=1, best_manifest=None, worker_entries=[],
    )
    restored_summary = state_store.load_json(results_dir / "parallel_summary.json")
    assert isinstance(restored_summary, dict)
    assert restored_summary["idea_generation_failure_count"] == 1

    summary = build_parallel_campaign_summary(
        results_dir=results_dir, seed_program_dir=seed_program_dir, worker_results=[]
    )

    assert state_store.tail_jsonl(state_store.idea_generation_failures_path(results_dir), max_lines=10) == []
    assert summary["idea_generation_failure_count"] == 1

    # the campaign summary counts archived and current idea failures
    count_root = tmp_path / "count"
    results_dir = count_root / "results"
    seed_program_dir = _seed(count_root)
    state_store.ensure_histories(results_dir)
    state_store.append_jsonl(
        state_store.idea_generation_failures_path(results_dir),
        {"cycle_index": 1, "generation_attempt": 2, "max_attempts": 2},
    )
    state_store.archive_for_parallel_prune(results_dir, results_dir, 0)
    state_store.append_jsonl(
        state_store.idea_generation_failures_path(results_dir),
        {"cycle_index": 2, "generation_attempt": 1, "max_attempts": 2},
    )

    summary = build_parallel_campaign_summary(
        results_dir=results_dir, seed_program_dir=seed_program_dir, worker_results=[]
    )

    assert summary["idea_generation_failure_count"] == 2


def test_aggregate_worker_outputs_root_state_failure_mapping_and_memory_cleanup(tmp_path: Path) -> None:
    # a partial retry from a failed worker does not advance root state
    partial_root = tmp_path / "partial"
    results_dir = partial_root / "results"
    worker_dir = results_dir / "worker_01"
    seed_program_dir = _seed(partial_root)
    worker = _make_worker_result(worker_dir, 1, best_score=None)
    worker.exit_code = 1
    (worker_dir / state_store.IDEAS_HISTORY_FILE).write_text("", encoding="utf-8")
    (worker_dir / state_store.ATTEMPTS_HISTORY_FILE).write_text("", encoding="utf-8")
    state_store.append_jsonl(
        state_store.idea_generation_failures_path(worker_dir),
        {"cycle_index": 1, "generation_attempt": 1, "max_attempts": 2, "reason": "idea_generation_exit_nonzero"},
    )

    aggregate_parallel_worker_outputs(
        results_dir=results_dir, seed_program_dir=seed_program_dir, worker_results=[worker], log_event_fn=_NO_LOG
    )

    outer_state = state_store.load_json(results_dir / state_store.OUTER_STATE_FILE)
    assert isinstance(outer_state, dict)
    assert outer_state["current_cycle"] == 0

    # skipped-cycle failures are mapped to global cycles and advance root state
    skipped_root = tmp_path / "skipped"
    results_dir = skipped_root / "results"
    worker_dir = results_dir / "worker_02"
    seed_program_dir = _seed(skipped_root)
    worker = _make_worker_result(worker_dir, 2, best_score=None)
    worker.plan.cycle_start = 5
    worker.plan.cycle_end = 6
    (worker_dir / state_store.IDEAS_HISTORY_FILE).write_text("", encoding="utf-8")
    (worker_dir / state_store.ATTEMPTS_HISTORY_FILE).write_text("", encoding="utf-8")
    for generation_attempt in (1, 2):
        state_store.append_jsonl(
            state_store.idea_generation_failures_path(worker_dir),
            {"cycle_index": 2, "generation_attempt": generation_attempt, "max_attempts": 2,
             "reason": "idea_generation_exit_nonzero"},
        )

    summary = aggregate_parallel_worker_outputs(
        results_dir=results_dir, seed_program_dir=seed_program_dir, worker_results=[worker], log_event_fn=_NO_LOG
    )

    failures = state_store.tail_jsonl(state_store.idea_generation_failures_path(results_dir), max_lines=10)
    outer_state = state_store.load_json(results_dir / state_store.OUTER_STATE_FILE)
    assert [failure["cycle_index"] for failure in failures] == [6, 6]
    assert all(failure["parallel_worker_index"] == 2 for failure in failures)
    assert isinstance(outer_state, dict)
    assert outer_state["current_cycle"] == 6
    assert summary["idea_generation_failure_count"] == 2

    # shared worker memory is cleaned after aggregation
    cleanup_root = tmp_path / "cleanup"
    results_dir = cleanup_root / "results"
    worker_dir = results_dir / "worker_01"
    seed_program_dir = _seed(cleanup_root)
    worker = _make_worker_result(worker_dir, 1, best_score=1.0)
    worker.plan.command = ["execution.isolate_worker_memory=false"]
    state_store.ensure_global_memory_files(worker_dir)

    aggregate_parallel_worker_outputs(
        results_dir=results_dir, seed_program_dir=seed_program_dir, worker_results=[worker], log_event_fn=_NO_LOG
    )

    assert not (worker_dir / state_store.GLOBAL_MEMORY_MD_FILE).exists()
    assert not (worker_dir / state_store.GLOBAL_MEMORY_JSONL_FILE).exists()


# ---------------------------------------------------------------------------
# run_parallel_campaign round loop
# ---------------------------------------------------------------------------


def test_isolated_memory_survives_aggregation(tmp_path: Path) -> None:
    def mock_build(**kwargs):
        enabled = kwargs["cfg"].execution.isolate_worker_memory
        return [f"execution.isolate_worker_memory={'true' if enabled else 'false'}"]

    def mock_run_worker(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
        worker = _make_worker_result(
            plan.results_dir, plan.worker_index, best_score=1.0,
            idea_id=f"idea_{plan.cycle_start}", attempt_id=f"attempt_{plan.cycle_start}",
        )
        worker.plan = plan
        (plan.results_dir / state_store.GLOBAL_MEMORY_MD_FILE).write_text(
            f"worker memory for cycle {plan.cycle_start}\n", encoding="utf-8"
        )
        (plan.results_dir / state_store.GLOBAL_MEMORY_JSONL_FILE).write_text(
            json.dumps({"cycle": plan.cycle_start}) + "\n", encoding="utf-8"
        )
        return worker

    # single-round (prune_n=0) and pruned (prune_n=1) campaigns
    for case, prune_n in (("single-round", 0), ("pruned", 1)):
        campaign = _campaign(tmp_path / case)

        exit_code = _run_campaign(
            _cfg(2, 1, prune_n, isolate_worker_memory=True, container=True),
            campaign,
            build_parallel_worker_command_fn=mock_build,
            run_parallel_worker_fn=mock_run_worker,
            aggregate_parallel_worker_outputs_fn=_REAL_AGGREGATE,
        )

        assert exit_code == 0, case
        worker_dirs = (
            [campaign.results / "worker_01"] if prune_n == 0
            else [campaign.results / "worker_01" / f"round_{i:03d}" for i in (0, 1)]
        )
        for worker_dir in worker_dirs:
            assert (worker_dir / state_store.GLOBAL_MEMORY_MD_FILE).is_file(), case
            assert (worker_dir / state_store.GLOBAL_MEMORY_JSONL_FILE).is_file(), case
        if prune_n:
            resume_state = state_store.load_parallel_resume_state(campaign.results)
            assert resume_state is not None
            assert resume_state["isolate_worker_memory"] is True


def test_round_schedule_single_round_and_multi_round_pruning(tmp_path: Path) -> None:
    # no pruning delegates to a single round with shared memory per worker
    campaign = _campaign(tmp_path / "no-prune")
    rounds: list[list[ParallelWorkerResult]] = []
    build_calls: list[dict] = []

    exit_code = _run_campaign(
        _cfg(4, 2, 0),
        campaign,
        build_parallel_worker_command_fn=_build_recorder(build_calls),
        run_parallel_worker_fn=_worker_plain,
        aggregate_parallel_worker_outputs_fn=partial(_aggregate, rounds=rounds),
    )

    assert exit_code == 0
    assert [len(results) for results in rounds] == [2]
    assert len(build_calls) == 4
    for call in build_calls:
        assert call["shared_memory_dir"] == campaign.memory
        assert call["global_memory_dir"] == call["results_dir"]
        assert call["conclusions_dir"] == call["results_dir"]

    # prune_every_n_cycles == max_idea_cycles -> a single aggregated round
    campaign = _campaign(tmp_path / "prune-equals-max")
    rounds = []

    exit_code = _run_campaign(
        _cfg(8, 2, 8),
        campaign,
        run_parallel_worker_fn=_worker_plain,
        aggregate_parallel_worker_outputs_fn=partial(_aggregate, rounds=rounds),
    )

    assert exit_code == 0
    assert [len(results) for results in rounds] == [2]

    # max_cycles=20, workers=4, prune_n=8 -> 3 rounds (8, 8, 4) and 2 prune events
    campaign = _campaign(tmp_path / "multi-round")
    rounds = []
    prune_events: list[dict] = []

    def mock_log(*args, **kwargs):
        if args and args[0] == "parallel_prune":
            prune_events.append(kwargs)

    exit_code = _run_campaign(
        _cfg(20, 4, 8),
        campaign,
        aggregate_parallel_worker_outputs_fn=partial(_aggregate, rounds=rounds),
        log_event_fn=mock_log,
    )

    assert exit_code == 0
    assert len(rounds) == 3
    assert [event["round_index"] for event in prune_events] == [0, 1]
    assert (campaign.results / "_pruned" / "round_000").is_dir()
    assert (campaign.results / "_pruned" / "round_001").is_dir()
    assert not (campaign.results / "_pruned" / "round_002").exists()


def test_failed_worker_lease_tolerance_and_fail_fast(tmp_path: Path) -> None:
    # without pruning, a failed dynamic lease does not stop healthy workers
    campaign = _campaign(tmp_path / "tolerant")
    completed_cycles: list[int] = []
    rounds: list[list[ParallelWorkerResult]] = []

    exit_code = _run_campaign(
        _cfg(5, 2, 0),
        campaign,
        run_parallel_worker_fn=_flaky_worker(completed_cycles),
        aggregate_parallel_worker_outputs_fn=partial(_aggregate, rounds=rounds),
    )

    assert exit_code == 2
    assert sorted(completed_cycles) == [1, 2, 3, 4, 5]
    failed_workers = [
        result.plan.worker_index for results in rounds for result in results if result.exit_code != 0
    ]
    assert len(failed_workers) == 1

    # parallel_fail_fast preserves the old global stop behavior
    campaign = _campaign(tmp_path / "fail-fast")
    completed_cycles = []

    exit_code = _run_campaign(
        _cfg(5, 2, 0, parallel_fail_fast=True),
        campaign,
        run_parallel_worker_fn=_flaky_worker(completed_cycles),
        aggregate_parallel_worker_outputs_fn=lambda **kw: {
            "failed_worker_count": 1, "idea_count": 0, "attempt_count": 0
        },
    )

    assert exit_code == 2
    assert max(completed_cycles) < 5

    # a failed pruning-round lease does not stop healthy workers
    campaign = _campaign(tmp_path / "pruned")
    completed_cycles = []
    rounds = []

    exit_code = _run_campaign(
        _cfg(4, 2, 4),
        campaign,
        run_parallel_worker_fn=_flaky_worker(completed_cycles, histories=True),
        aggregate_parallel_worker_outputs_fn=partial(_aggregate, rounds=rounds),
    )

    assert exit_code == 2
    assert sorted(completed_cycles) == [1, 2, 3, 4]
    assert any(result.exit_code != 0 for results in rounds for result in results)


def test_final_parallel_summary_preserves_best_and_counts_across_rounds(tmp_path: Path) -> None:
    """Later failed rounds must not wipe the run-level best score or counts."""
    campaign = _campaign(tmp_path)
    round_counter = 0

    def mock_run_worker(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
        nonlocal round_counter
        plan.results_dir.mkdir(parents=True, exist_ok=True)
        state_store.ensure_histories(plan.results_dir, include_global_memory=False)
        if round_counter == 0:
            program_dir = plan.results_dir / "program_1"
            make_program_dir(program_dir, content="# best round 0\n")
            _write_best(plan.results_dir, program_dir, score=1.0, correct=True, attempt_id="att_1")
            _append_history(plan.results_dir, "idea_round_0", "att_1")
        else:
            _write_best(
                plan.results_dir, plan.results_dir / "missing_program", score=None, correct=False, attempt_id="att_2"
            )
        round_counter += 1
        return _result(plan, exit_code=1 if round_counter == 2 else 0)

    exit_code = _run_campaign(
        _cfg(2, 1, 1),
        campaign,
        run_parallel_worker_fn=mock_run_worker,
        aggregate_parallel_worker_outputs_fn=_REAL_AGGREGATE,
    )

    assert exit_code == 2
    summary = json.loads((campaign.results / "parallel_summary.json").read_text(encoding="utf-8"))
    assert summary["best_combined_score"] == 1.0
    assert summary["idea_count"] == 1
    assert summary["attempt_count"] == 1
    assert summary["failed_worker_count"] == 1
    root_manifest = json.loads((campaign.results / state_store.BEST_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert root_manifest["combined_score"] == 1.0
    assert root_manifest["correct"] is True


def test_seed_program_forwarding_and_persistence_across_rounds(tmp_path: Path) -> None:
    # after round 0, worker seeds are updated from best manifests
    campaign = _campaign(tmp_path / "forwarded")
    build_calls: list[dict] = []

    _run_campaign(
        _cfg(8, 2, 4),
        campaign,
        build_parallel_worker_command_fn=_build_recorder(build_calls),
        run_parallel_worker_fn=_barrier_worker(threading.Barrier(2)),
    )

    round_0_calls = _calls_in_round(build_calls, "round_000")
    assert len(round_0_calls) == 4
    for call in round_0_calls:
        assert call["seed_program"] == campaign.seed
    round_1_calls = _calls_in_round(build_calls, "round_001")
    assert len(round_1_calls) == 4
    for call in round_1_calls:
        assert call["seed_program"] is not None
        assert call["seed_program"].name == "program_1"

    # round isolation: best programs from round N seed round N+1; memory lives inside the round dir
    campaign = _campaign(tmp_path / "isolation")
    build_calls = []

    _run_campaign(
        _cfg(4, 2, 2, container=True),
        campaign,
        build_parallel_worker_command_fn=_build_recorder(build_calls),
        run_parallel_worker_fn=_barrier_worker(threading.Barrier(2), key="program_path"),
    )

    for call in build_calls:
        assert call["global_memory_dir"] == call["results_dir"], (
            f"global_memory_dir={call['global_memory_dir']} should equal "
            f"results_dir={call['results_dir']} for round isolation"
        )
    round_0_calls = _calls_in_round(build_calls, "round_000")
    assert len(round_0_calls) == 2
    for call in round_0_calls:
        assert call["seed_program"] == campaign.seed
    round_1_calls = _calls_in_round(build_calls, "round_001")
    assert len(round_1_calls) == 2
    for call in round_1_calls:
        assert call["seed_program"] is not None
        assert call["seed_program"].name == "program_1"
        assert "round_000" in str(call["seed_program"])

    # a worker keeps its best seed when later rounds yield no valid best or a lower-scoring one
    for case, first_score, later in (("no-new-valid-best", 1.0, "missing"), ("worse-valid-best", 10.0, "worse")):
        campaign = _campaign(tmp_path / case)
        build_calls = []
        round_counter = 0
        first_round_seed: Path | None = None

        def mock_run_worker(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
            nonlocal round_counter
            plan.results_dir.mkdir(parents=True, exist_ok=True)
            state_store.ensure_histories(plan.results_dir, include_global_memory=False)
            attempt_id = f"att_{plan.worker_index}"
            if round_counter == 0:
                program_dir = plan.results_dir / "program_1"
                make_program_dir(program_dir, content="# best round 0\n")
                _write_best(plan.results_dir, program_dir, score=first_score, correct=True, attempt_id=attempt_id)
            elif later == "missing":
                _write_best(
                    plan.results_dir, plan.results_dir / "missing_program", score=None, correct=False,
                    attempt_id=attempt_id,
                )
            else:
                program_dir = plan.results_dir / "program_worse"
                make_program_dir(program_dir, content="# worse\n")
                _write_best(plan.results_dir, program_dir, score=5.0, correct=True, attempt_id=attempt_id)
            round_counter += 1
            _append_history(plan.results_dir, f"idea_{plan.worker_index}", attempt_id)
            return _result(plan)

        def mock_aggregate(*, results_dir, seed_program_dir, worker_results):
            nonlocal first_round_seed
            if first_round_seed is None:
                first_round_seed = worker_results[0].plan.results_dir / "program_1"
            return _aggregate(results_dir=results_dir, seed_program_dir=seed_program_dir, worker_results=worker_results)

        _run_campaign(
            _cfg(3, 1, 1),
            campaign,
            build_parallel_worker_command_fn=_build_recorder(build_calls),
            run_parallel_worker_fn=mock_run_worker,
            aggregate_parallel_worker_outputs_fn=mock_aggregate,
        )

        assert len(build_calls) == 3, case
        assert build_calls[0]["seed_program"] == campaign.seed, case
        assert first_round_seed is not None, case
        assert build_calls[1]["seed_program"].resolve() == first_round_seed.resolve(), case
        assert build_calls[2]["seed_program"].resolve() == first_round_seed.resolve(), case


def test_pruning_round_uses_dynamic_worker_leases(tmp_path: Path) -> None:
    """Pruning rounds should lease cycles dynamically while preserving round dirs."""
    campaign = _campaign(tmp_path)
    build_calls: list[dict] = []
    first_round_maps: dict[int, dict[int, int] | None] | None = None
    first_lease_barrier = threading.Barrier(2)

    def mock_run_worker(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
        round_name = plan.results_dir.name
        if round_name == "round_000" and plan.lease_index == 1:
            first_lease_barrier.wait(timeout=1)
        if round_name == "round_000" and plan.worker_index == 2 and plan.lease_index == 1:
            time.sleep(0.05)
        plan.results_dir.mkdir(parents=True, exist_ok=True)
        state_store.ensure_histories(plan.results_dir, include_global_memory=False)
        program_dir = plan.results_dir / "program_1"
        make_program_dir(program_dir, content="# best\n")
        _write_best(plan.results_dir, program_dir, score=1.0, correct=True, attempt_id=f"att_{plan.worker_index}")
        _append_history(
            plan.results_dir,
            f"idea_{plan.worker_index}_{plan.lease_index}",
            f"att_{plan.worker_index}_{plan.lease_index}",
            cycle_index=plan.lease_index,
        )
        log_path = plan.log_path or plan.results_dir / "parallel_worker.log"
        log_path.write_text("ok\n", encoding="utf-8")
        result = _result(plan, log_path=log_path)
        result.elapsed_seconds = 0.01
        return result

    def mock_aggregate(*, results_dir, seed_program_dir, worker_results):
        nonlocal first_round_maps
        if first_round_maps is None:
            first_round_maps = {result.plan.worker_index: result.plan.cycle_map for result in worker_results}
        return _aggregate(results_dir=results_dir, seed_program_dir=seed_program_dir, worker_results=worker_results)

    exit_code = _run_campaign(
        _cfg(6, 2, 4),
        campaign,
        build_parallel_worker_command_fn=_build_recorder(build_calls),
        run_parallel_worker_fn=mock_run_worker,
        aggregate_parallel_worker_outputs_fn=mock_aggregate,
    )

    assert exit_code == 0
    assert first_round_maps is not None
    first_round_cycles = sorted(
        global_cycle for cycle_map in first_round_maps.values() for global_cycle in (cycle_map or {}).values()
    )
    assert first_round_cycles == [1, 2, 3, 4]
    assert any(len(cycle_map or {}) > 2 for cycle_map in first_round_maps.values())
    round_0_calls = _calls_in_round(build_calls, "round_000")
    assert len(round_0_calls) == 4
    assert {str(call["results_dir"].relative_to(campaign.results)) for call in round_0_calls} == {
        "worker_01/round_000", "worker_02/round_000",
    }
    assert any(call["resume"] is True for call in round_0_calls)


def test_parallel_resume_restarts_from_next_round_after_successful_checkpoint(tmp_path: Path) -> None:
    """Resume should continue from the next round using persisted worker seeds."""
    campaign = _campaign(tmp_path)
    first_interrupt_triggered = False
    first_lease_barrier = threading.Barrier(2)

    def first_run_worker(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
        nonlocal first_interrupt_triggered
        round_index = int(plan.results_dir.name.split("_")[1])
        if round_index == 0:
            first_lease_barrier.wait(timeout=10)
        if round_index == 1 and not first_interrupt_triggered:
            first_interrupt_triggered = True
            raise KeyboardInterrupt
        return _worker_success(plan, content=f"# worker {plan.worker_index}\n", score=float(plan.worker_index))

    with pytest.raises(KeyboardInterrupt):
        _run_campaign(
            _cfg(4, 2, 2),
            campaign,
            run_parallel_worker_fn=first_run_worker,
            aggregate_parallel_worker_outputs_fn=_REAL_AGGREGATE,
        )

    resume_state = state_store.load_parallel_resume_state(campaign.results)
    assert resume_state is not None
    assert resume_state["next_round_index"] == 1
    assert resume_state["global_cycle_offset"] == 3
    assert resume_state["worker_seeds"]["1"].endswith("worker_01/round_000/program_1")
    assert resume_state["worker_seeds"]["2"].endswith("worker_02/round_000/program_1")

    resume_build_calls: list[dict] = []
    resume_lease_barrier = threading.Barrier(2)

    def resume_run_worker(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
        resume_lease_barrier.wait(timeout=10)
        plan.results_dir.mkdir(parents=True, exist_ok=True)
        state_store.ensure_histories(plan.results_dir, include_global_memory=False)
        program_dir = plan.results_dir / "program_2"
        make_program_dir(program_dir, content="# resumed\n")
        _write_best(
            plan.results_dir, program_dir, score=float(plan.worker_index) + 10.0, correct=True,
            attempt_id=f"resume_att_{plan.worker_index}",
        )
        _append_history(plan.results_dir, f"resume_idea_{plan.worker_index}", f"resume_att_{plan.worker_index}")
        return _result(plan)

    exit_code = _run_campaign(
        _cfg(4, 2, 2, resume=True),
        campaign,
        build_parallel_worker_command_fn=_build_recorder(resume_build_calls),
        run_parallel_worker_fn=resume_run_worker,
        aggregate_parallel_worker_outputs_fn=_REAL_AGGREGATE,
    )

    assert exit_code == 0
    assert len(resume_build_calls) == 2
    for call in resume_build_calls:
        assert "round_001" in str(call["results_dir"])
        assert call["seed_program"] is not None
        assert call["seed_program"].name == "program_1"


RESUME_ISOLATION_CASES: list[tuple[str, object, bool, str]] = [
    ("legacy_shared", None, True, "isolate_worker_memory"),
    ("shared", False, True, "isolate_worker_memory"),
    ("isolated", True, False, "isolate_worker_memory"),
    ("malformed", "{invalid", False, "invalid"),
    ("non_object", [], False, "invalid"),
]


def test_parallel_resume_rejects_incompatible_or_invalid_isolation_state(tmp_path: Path) -> None:
    for case, checkpoint_value, current_isolation, expected_error in RESUME_ISOLATION_CASES:
        campaign = _campaign(tmp_path / case)
        global_mem = campaign.root / "missing_mem"
        resume_path = state_store.parallel_resume_state_path(campaign.results)
        resume_state = {
            "version": 1,
            "mode": "parallel_pruned",
            "total_rounds": 2,
            "cycles_per_round": 2,
            "workers": 2,
            "next_round_index": 1,
            "global_cycle_offset": 3,
            "completed_rounds": [0],
            "worker_seeds": {"1": str(campaign.seed), "2": str(campaign.seed)},
            "worker_seed_scores": {},
            "shared_memory_source": None,
            "resume_basis": "persisted",
        }
        if isinstance(checkpoint_value, str):
            resume_path.write_text(checkpoint_value, encoding="utf-8")
        elif isinstance(checkpoint_value, list):
            state_store.write_json(resume_path, checkpoint_value)
        else:
            if checkpoint_value is not None:
                resume_state["isolate_worker_memory"] = checkpoint_value
            state_store.write_json(resume_path, resume_state)
        events: list[tuple[str, dict]] = []

        exit_code = _run_campaign(
            _cfg(4, 2, 2, resume=True, isolate_worker_memory=current_isolation),
            campaign,
            global_memory_dir=global_mem,
            run_parallel_worker_fn=_worker_should_not_run,
            aggregate_parallel_worker_outputs_fn=lambda **kwargs: {},
            log_event_fn=lambda event, **kwargs: events.append((event, kwargs)),
        )

        assert exit_code == 2, case
        assert events[-1][0] == "parallel_resume_failed", case
        assert expected_error in events[-1][1]["error"], case
        assert not global_mem.exists(), case


def test_parallel_resume_legacy_round_boundary_and_complete_campaign_noop(tmp_path: Path) -> None:
    # legacy runs resume from the first round after the last healthy archive
    campaign = _campaign(tmp_path / "legacy")
    results_dir = campaign.results
    archive_dir = results_dir / "_pruned" / "round_000"
    archive_dir.mkdir(parents=True, exist_ok=True)
    state_store.append_jsonl(archive_dir / state_store.IDEAS_HISTORY_FILE, {"idea_id": "idea_round_0", "cycle_index": 1})
    state_store.append_jsonl(
        archive_dir / state_store.ATTEMPTS_HISTORY_FILE,
        {"attempt_id": "att_round_0", "idea_id": "idea_round_0", "cycle_index": 1},
    )
    shared_archive = archive_dir / "_shared_memory"
    shared_archive.mkdir(parents=True, exist_ok=True)
    (shared_archive / state_store.GLOBAL_MEMORY_MD_FILE).write_text("# Global Memory Log\n\n", encoding="utf-8")
    (shared_archive / state_store.GLOBAL_MEMORY_JSONL_FILE).write_text("", encoding="utf-8")
    stale_archive = results_dir / "_pruned" / "round_001"
    stale_archive.mkdir(parents=True, exist_ok=True)
    state_store.append_jsonl(stale_archive / state_store.IDEAS_HISTORY_FILE, {"idea_id": "stale", "cycle_index": 3})

    def worker_entry(worker_index: int, cycle: int, round_name: str, exit_code: int) -> dict:
        round_dir = results_dir / f"worker_{worker_index:02d}" / round_name
        return {
            "worker_index": worker_index,
            "assigned_cycles": 1,
            "cycle_start": cycle,
            "cycle_end": cycle,
            "results_dir": str(round_dir),
            "log_path": str(round_dir / "parallel_worker.log"),
            "exit_code": exit_code,
            "elapsed_seconds": 1.0,
        }

    state_store.write_json(
        results_dir / "parallel_summary.json",
        {
            "workers": [
                worker_entry(1, 1, "round_000", 0),
                worker_entry(2, 2, "round_000", 0),
                worker_entry(1, 3, "round_001", 2),
                worker_entry(2, 4, "round_001", 2),
            ]
        },
    )
    stale_round_dir = results_dir / "worker_01" / "round_001"
    stale_round_dir.mkdir(parents=True, exist_ok=True)
    (stale_round_dir / "obsolete.txt").write_text("stale", encoding="utf-8")
    for worker_index in (1, 2):
        round_dir = results_dir / f"worker_{worker_index:02d}" / "round_001"
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "parallel_worker.log").write_text(
            f"paths.seed_program={results_dir / f'legacy_seed_w{worker_index}'} execution.workers=1\n",
            encoding="utf-8",
        )
        make_program_dir(results_dir / f"legacy_seed_w{worker_index}", content=f"# w{worker_index}\n")

    build_calls: list[dict] = []

    exit_code = _run_campaign(
        _cfg(4, 2, 2, resume=True),
        campaign,
        build_parallel_worker_command_fn=_build_recorder(build_calls),
        run_parallel_worker_fn=_barrier_worker(threading.Barrier(2), content="# resumed legacy\n"),
        aggregate_parallel_worker_outputs_fn=_REAL_AGGREGATE,
    )

    assert exit_code == 0
    assert len(build_calls) == 2
    assert {call["worker_index"]: call["seed_program"].name for call in build_calls} == {
        1: "legacy_seed_w1", 2: "legacy_seed_w2",
    }
    assert not stale_archive.exists()
    assert not (stale_round_dir / "obsolete.txt").exists()

    # resuming a complete campaign is a no-op
    campaign = _campaign(tmp_path / "complete")
    state_store.write_json(
        state_store.parallel_resume_state_path(campaign.results),
        {
            "version": 1,
            "mode": "parallel_pruned",
            "total_rounds": 2,
            "cycles_per_round": 2,
            "workers": 2,
            "next_round_index": 2,
            "global_cycle_offset": 5,
            "completed_rounds": [0, 1],
            "worker_seeds": {"1": str(campaign.seed), "2": str(campaign.seed)},
            "shared_memory_source": None,
            "resume_basis": "persisted",
        },
    )
    state_store.write_json(
        campaign.results / "parallel_summary.json",
        {
            "worker_count": 4,
            "failed_worker_count": 0,
            "idea_count": 2,
            "attempt_count": 2,
            "best_program_dir": str(campaign.seed),
            "best_entrypoint_path": "src/program.py",
            "best_combined_score": 1.0,
            "workers": [],
        },
    )
    state_store.write_json(
        campaign.results / state_store.BEST_MANIFEST_FILE,
        {
            "program_dir": str(campaign.seed),
            "entrypoint_path": "src/program.py",
            "combined_score": 1.0,
            "correct": True,
            "source_attempt_id": "done",
        },
    )
    build_calls = []

    exit_code = _run_campaign(
        _cfg(4, 2, 2, resume=True),
        campaign,
        build_parallel_worker_command_fn=_build_recorder(build_calls),
        run_parallel_worker_fn=_worker_should_not_run,
        aggregate_parallel_worker_outputs_fn=_REAL_AGGREGATE,
    )

    assert exit_code == 0
    assert build_calls == []


# ---------------------------------------------------------------------------
# Round isolation (worker_XX/round_NNN) with and without containers
# ---------------------------------------------------------------------------


def _prune_confirmation(container: bool):
    """Non-container pruning prompts for confirmation; patch it out there."""
    return nullcontext() if container else patch(CONFIRM_PRUNE_WARNING)


def test_round_archive_skips_symlinked_shared_memory_file(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    outside = tmp_path / "outside-secret"
    outside.write_text("do not archive\n", encoding="utf-8")

    def mock_run_worker(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
        plan.results_dir.mkdir(parents=True, exist_ok=True)
        log_path = plan.log_path or plan.results_dir / "log"
        log_path.write_text("ok\n", encoding="utf-8")
        if plan.cycle_start == 1:
            shared_md = campaign.results / "_shared_memory" / "round_000" / state_store.GLOBAL_MEMORY_MD_FILE
            shared_md.unlink()
            shared_md.symlink_to(outside)
        result = _result(plan, log_path=log_path)
        result.elapsed_seconds = 0.01
        return result

    exit_code = _run_campaign(
        _cfg(2, 1, 1, container=True),
        campaign,
        run_parallel_worker_fn=mock_run_worker,
        aggregate_parallel_worker_outputs_fn=lambda **kwargs: {},
    )

    assert exit_code == 0
    assert outside.read_text(encoding="utf-8") == "do not archive\n"
    archived_md = campaign.results / "_pruned" / "round_000" / "_shared_memory" / state_store.GLOBAL_MEMORY_MD_FILE
    assert not archived_md.exists()


def test_pruning_uses_round_subdirs_with_and_without_container(tmp_path: Path) -> None:
    for case, container in (("container", True), ("no-container", False)):
        # two workers: each pruning round uses worker_XX/round_NNN/
        campaign = _campaign(tmp_path / f"two-workers-{case}")
        build_calls: list[dict] = []

        with _prune_confirmation(container):
            exit_code = _run_campaign(
                _cfg(8, 2, 4, container=container),
                campaign,
                build_parallel_worker_command_fn=_build_recorder(build_calls),
                run_parallel_worker_fn=partial(_worker_success, key="program_path"),
            )

        assert exit_code == 0, case
        expected = ("worker_01/round_000", "worker_02/round_000", "worker_01/round_001", "worker_02/round_001")
        for call in build_calls:
            assert "round_" in str(call["results_dir"]), case
            assert str(call["results_dir"]).endswith(expected), case
        assert len(_calls_in_round(build_calls, "round_000")) == 4, case
        assert len(_calls_in_round(build_calls, "round_001")) == 4, case
        for relative in expected:
            assert (campaign.results / relative).is_dir(), case

        # single worker with a file seed also uses round subdirs
        campaign = _campaign(tmp_path / f"single-worker-{case}", seed_file=True)
        build_calls = []

        with _prune_confirmation(container):
            exit_code = _run_campaign(
                _cfg(4, 1, 2, container=container),
                campaign,
                build_parallel_worker_command_fn=_build_recorder(build_calls),
                run_parallel_worker_fn=_worker_file_program,
            )

        assert exit_code == 0, case
        assert [str(call["results_dir"].relative_to(campaign.results)) for call in build_calls] == [
            "worker_01/round_000",
            "worker_01/round_000",
            "worker_01/round_001",
            "worker_01/round_001",
        ], case
        assert (campaign.results / "worker_01" / "round_000").is_dir(), case
        assert (campaign.results / "worker_01" / "round_001").is_dir(), case


def test_prune_warning_only_without_container(tmp_path: Path) -> None:
    for case, container, max_cycles, prune_n, expected_warnings in (
        ("no-container-warns", False, 8, 4, 1),
        ("container-silent", True, 4, 2, 0),
    ):
        campaign = _campaign(tmp_path / case)
        events: list[tuple] = []

        with _prune_confirmation(container):
            _run_campaign(
                _cfg(max_cycles, 2, prune_n, container=container),
                campaign,
                run_parallel_worker_fn=partial(_worker_success, key="program_path"),
                log_event_fn=lambda *args, **kwargs: events.append((args, kwargs)),
            )

        warning_events = [(args, kw) for args, kw in events if args and args[0] == "prune_warning"]
        assert len(warning_events) == expected_warnings, case
        if expected_warnings:
            assert "NOT containerized" in warning_events[0][1]["message"], case


# ---------------------------------------------------------------------------
# Task failure stops leases and promotion
# ---------------------------------------------------------------------------


def test_parallel_task_failure_stops_leases_and_promotion(tmp_path: Path) -> None:
    for prune in (0, 2):
        for broken_logging in (False, True, "sibling"):
            for terminal_code in (86, 87):
                case = f"prune={prune} broken_logging={broken_logging} terminal_code={terminal_code}"
                root = tmp_path / f"{prune}-{broken_logging}-{terminal_code}"
                root.mkdir()
                first_leases = threading.Barrier(2)
                cancelled = threading.Event()
                plans: list[ParallelWorkerPlan] = []
                aggregates: list[dict] = []

                def run_worker(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
                    plans.append(plan)
                    plan.results_dir.mkdir(parents=True, exist_ok=True)
                    if plan.lease_index == 1:
                        first_leases.wait(timeout=3)
                    if plan.worker_index == 1:
                        code = terminal_code
                    else:
                        cancelled.wait(timeout=0.1)
                        if broken_logging == "sibling" and cancelled.is_set():
                            raise OSError("sibling consolidation log unavailable")
                        code = 0
                    assert plan.log_path is not None
                    return ParallelWorkerResult(plan, code, plan.log_path, 0.0, "")

                def cancel(**kwargs):
                    cancelled.set()
                    return []

                def log(event, **kwargs):
                    if broken_logging and event == "parallel_task_failure":
                        raise OSError("shared journal unavailable")

                def aggregate(**kwargs: Any) -> dict[str, Any]:
                    aggregates.append(kwargs)
                    return {}

                code = run_parallel_campaign(
                    cfg=_cfg(6, 2, prune, parallel_fail_fast=False),
                    task_dir=root / "task",
                    results_dir=root / "run",
                    global_memory_dir=root / "memory",
                    instruct_path=root / "instruct.md",
                    seed_program_dir=root / "seed",
                    split_cycle_budget_fn=split_cycle_budget,
                    build_parallel_worker_command_fn=lambda **kwargs: ["true"],
                    run_parallel_worker_fn=run_worker,
                    aggregate_parallel_worker_outputs_fn=aggregate,
                    log_event_fn=log,
                    format_elapsed_fn=str,
                    terminate_all_active_processes_fn=cancel,
                    stop_container_session_fn=lambda *args, **kwargs: True,
                    build_container_session_name_fn=lambda **kwargs: "test-session",
                )

                assert code == terminal_code, case
                assert cancelled.is_set(), case
                assert len(plans) == 2, case
                assert aggregates == [], case
