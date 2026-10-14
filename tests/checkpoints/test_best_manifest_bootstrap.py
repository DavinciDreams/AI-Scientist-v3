"""Seed freezing, best-manifest bootstrap, and harness incumbent reconciliation on resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.investigation.harness_incumbent import (
    reconcile_harness_incumbent,
    runtime_baseline_correct,
    runtime_baseline_score,
    seed_matches_validated_baseline,
)
from src.investigation.main_entry import _freeze_seed_program
from src.orchestrator import state as state_store
from src.pipeline.config import validate_completed_investigation
from tests.program_testutils import write_program_entrypoint, write_program_spec_toml


def _program(program_dir: Path, body: str = "print('ok')\n", entrypoint: str = "program.py") -> Path:
    write_program_spec_toml(program_dir, entrypoint=entrypoint)
    write_program_entrypoint(program_dir, body, entrypoint=entrypoint)
    return program_dir


def _seed(tmp_path: Path, **kw: str) -> Path:
    return _program(tmp_path / "task" / "program_0", **kw)


def _write_sealed_score(program_dir: Path, score: float) -> None:
    (program_dir / "metrics.json").write_text(
        json.dumps({"combined_score": score}), encoding="utf-8"
    )
    (program_dir / "correct.json").write_text('{"correct": true}', encoding="utf-8")
    (program_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "producer": "sandbox_host_eval",
                "execution_boundary": "sandbox",
                "program_id": program_dir.name,
                "returncode": 0,
                "exit_code": 0,
                "command": ["docker", "start", "--attach", "sandbox"],
                "started_at": "2026-08-21T00:00:00Z",
                "finished_at": "2026-08-21T00:00:01Z",
            }
        ),
        encoding="utf-8",
    )


def _reconcile(tmp_path: Path, seed: Path, best: dict, outer: dict | None = None, **kw) -> bool:
    defaults = dict(
        task_dir=tmp_path / "task",
        results_dir=tmp_path / "results",
        seed_program_dir=seed,
        baseline_score=0.3,
        baseline_correct=True,
        best_manifest=best,
        outer_state={} if outer is None else outer,
    )
    return reconcile_harness_incumbent(**{**defaults, **kw})


def test_frozen_seed_is_copied_matches_by_content_and_bootstraps_the_harness(
    tmp_path: Path,
) -> None:
    # the frozen seed is copied under results without its evaluation artifacts
    seed = _seed(tmp_path, entrypoint="main.py")
    (seed / "metrics.json").write_text("{}", encoding="utf-8")
    results_dir = tmp_path / "results"

    frozen = _freeze_seed_program(seed, results_dir)

    assert frozen.is_relative_to(results_dir.resolve())
    assert frozen.name == "program_0"
    assert (frozen / "src" / "main.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert not (frozen / "metrics.json").exists()
    assert (seed / "metrics.json").is_file()

    # the frozen baseline remains a valid harness bootstrap
    best = state_store.load_or_init_best_manifest(
        results_dir=results_dir, seed_program_dir=frozen, resume=False
    )
    outer = state_store.load_or_init_outer_state(
        results_dir=results_dir, seed_program_dir=frozen, resume=False
    )

    assert _reconcile(tmp_path, frozen, best, outer, validated_baseline_program_dir=frozen)
    assert best["program_dir"] == str(frozen)
    assert best["combined_score"] == 0.3

    # the frozen seed matches the validated baseline by content
    assert seed_matches_validated_baseline(seed, seed)
    assert seed_matches_validated_baseline(frozen, seed)
    assert not seed_matches_validated_baseline(tmp_path / "missing", seed)

    (frozen / "src" / "main.py").write_text("print('changed')\n", encoding="utf-8")
    assert not seed_matches_validated_baseline(frozen, seed)


def test_resume_recovers_the_baseline_or_reconstructs_a_superior_candidate(
    tmp_path: Path,
) -> None:
    # a resume without a score recovers the validated baseline
    base = tmp_path / "no-score"
    seed = _seed(base, entrypoint="main.py")
    best: dict[str, Any] = {"program_dir": str(seed), "combined_score": None}
    outer: dict[str, object] = {}

    assert _reconcile(base, seed, best, outer)
    assert best["combined_score"] == 0.3
    assert best["program_dir"] == str(seed.resolve())
    assert outer["best_combined_score"] == 0.3

    # a resume replaces an inferior candidate with the baseline
    base = tmp_path / "inferior"
    seed = _seed(base)
    candidate = _program(base / "results" / "candidate")
    best = {"program_dir": str(candidate), "combined_score": 0.2, "correct": True}
    _write_sealed_score(candidate, 0.2)

    assert _reconcile(base, seed, best)
    assert best["combined_score"] == 0.3
    assert best["program_dir"] == str(seed.resolve())

    # a resume reconstructs a superior candidate score from its artifacts
    base = tmp_path / "superior"
    seed = _seed(base)
    candidate = _program(base / "results" / "candidate")
    best = {
        "program_path": str(candidate / "src" / "program.py"),
        "combined_score": -100.0,
        "correct": True,
    }
    _write_sealed_score(candidate, 0.5)
    outer = {}

    assert _reconcile(base, seed, best, outer)
    assert best["combined_score"] == 0.5
    assert best["program_dir"] == str(candidate)
    assert outer["best_program_dir"] == str(candidate.resolve())


def test_resume_rejects_unvalidated_baselines_unmanaged_paths_and_symlinked_metrics(
    tmp_path: Path,
) -> None:
    # a baseline without runtime or sealed artifacts is rejected
    base = tmp_path / "unvalidated"
    seed = _seed(base)
    (seed / "metrics.json").write_text('{"combined_score": -100.0}', encoding="utf-8")
    (seed / "correct.json").write_text('{"correct": true}', encoding="utf-8")
    best: dict[str, Any] = {"program_dir": str(seed), "combined_score": None}

    assert not _reconcile(base, seed, best, baseline_score=None, baseline_correct=None)

    # an unmanaged candidate path is rejected
    base = tmp_path / "unmanaged"
    seed = _seed(base)
    unmanaged = _program(base / "outside-results")
    _write_sealed_score(unmanaged, 99.0)
    best = {"program_dir": str(unmanaged), "combined_score": 99.0, "correct": True}

    assert _reconcile(base, seed, best)
    assert best["combined_score"] == 0.3
    assert best["program_dir"] == str(seed.resolve())

    # symlinked candidate metrics are rejected
    base = tmp_path / "symlinked"
    seed = _seed(base)
    candidate = _program(base / "results" / "candidate")
    _write_sealed_score(candidate, 99.0)
    external_metrics = base / "external-metrics.json"
    external_metrics.write_text('{"combined_score": 99.0}', encoding="utf-8")
    (candidate / "metrics.json").unlink()
    (candidate / "metrics.json").symlink_to(external_metrics)
    best = {"program_dir": str(candidate), "combined_score": 99.0, "correct": True}

    assert _reconcile(base, seed, best)
    assert best["combined_score"] == 0.3
    assert best["program_dir"] == str(seed.resolve())


def test_runtime_baseline_is_read_from_the_manifest_and_never_laundered_into_incumbent(
    tmp_path: Path,
) -> None:
    # the runtime baseline score comes from the validated manifest
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / ".agentic_runtime.json").write_text(
        '{"baseline_combined_score": 0.375}', encoding="utf-8"
    )

    assert runtime_baseline_score(manifest_dir) == 0.375
    assert runtime_baseline_correct(manifest_dir) is True

    # an incorrect runtime baseline is not laundered into the incumbent
    task_dir = tmp_path / "task"
    seed = _seed(tmp_path, body="print('seed')\n")
    results_dir = tmp_path / "results"
    frozen_seed = _freeze_seed_program(seed, results_dir)
    (task_dir / ".agentic_runtime.json").write_text(
        json.dumps({"baseline_combined_score": 0.9, "baseline_correct": False}), encoding="utf-8"
    )
    best = state_store.load_or_init_best_manifest(
        results_dir=results_dir, seed_program_dir=frozen_seed, resume=False
    )
    outer: dict[str, object] = {}
    runtime = dict(
        baseline_score=runtime_baseline_score(task_dir),
        baseline_correct=runtime_baseline_correct(task_dir),
        validated_baseline_program_dir=frozen_seed,
    )

    assert _reconcile(tmp_path, frozen_seed, best, outer, **runtime)
    state_store.save_best_manifest(results_dir, best)

    assert best["correct"] is False
    assert outer["best_correct"] is False
    assert validate_completed_investigation(results_dir) == [
        "investigation retained its baseline and produced no optimized winner: "
        f"{results_dir / 'best_program_manifest.json'}"
    ]

    correct_candidate = _program(results_dir / "program_1", body="print('candidate')\n")
    _write_sealed_score(correct_candidate, 0.1)
    best.update(program_dir=str(correct_candidate), correct=True)

    assert _reconcile(tmp_path, frozen_seed, best, outer, **runtime)
    assert best["program_dir"] == str(correct_candidate.resolve())
    assert best["correct"] is True


def test_bootstrap_derives_entrypoint_from_program_spec(tmp_path: Path) -> None:
    seed = _seed(tmp_path, entrypoint="main.py")

    manifest = state_store.load_or_init_best_manifest(
        results_dir=tmp_path / "results", seed_program_dir=seed, resume=False
    )

    assert manifest["program_dir"] == str(seed)
    assert manifest["entrypoint_path"] == "src/main.py"
