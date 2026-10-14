"""Standalone best-program extraction pipeline."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from src.investigation.best_program_extraction import ExtractionError, extract_best_program_bundle
from tests.program_testutils import make_program_dir

ARTIFACTS = ["metrics.json", "correct.json", "rubric_status.json", "stdout.log", "stderr.log"]


def _task_dir(root: Path) -> Path:
    """Minimal task directory with scaffold files and a seed program_0 under root."""
    task = root / "task"
    task.mkdir(parents=True)
    (task / "task.md").write_text("# Goal\nTest task\n# Rubrics\n- r1\n", encoding="utf-8")
    (task / "evaluate.py").write_text("print('eval')\n", encoding="utf-8")
    (task / "capture.py").write_text("print('capture')\n", encoding="utf-8")
    make_program_dir(task / "program_0", content="print('seed')\n")
    return task


def _winner(root: Path) -> Path:
    make_program_dir(root / "winner", content="print('winner')\n")
    return root / "winner"


def _results_dir(root: Path, program_dir: Path, *, allow_external_manifest: bool = False, **overrides: object) -> Path:
    """Results directory with a best_program_manifest.json pointing at program_dir."""
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if not allow_external_manifest and program_dir.exists() and not program_dir.resolve().is_relative_to(results_dir.resolve()):
        managed_program_dir = results_dir / "managed_program"
        shutil.copytree(program_dir, managed_program_dir, symlinks=True)
        program_dir = managed_program_dir
    manifest: dict[str, object] = {
        "program_dir": str(program_dir),
        "entrypoint_path": "src/program.py",
        "combined_score": 0.85,
        "correct": True,
        "source_attempt_id": "idea_1_attempt_1",
        "updated_at": "2025-01-01T00:00:00+00:00",
        **overrides,
    }
    (results_dir / "best_program_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results_dir


def _extract(task_dir: Path, results_dir: Path) -> Path:
    output_dir = task_dir.parent / "output"
    extract_best_program_bundle(task_dir=task_dir, results_dir=results_dir, output_dir=output_dir)
    return output_dir


def _manifest(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "extraction_manifest.json").read_text(encoding="utf-8"))


def _program_source(output_dir: Path, name: str = "program_0") -> str:
    return (output_dir / name / "src" / "program.py").read_text(encoding="utf-8")


def test_extraction_bundle_contents_and_manifest(tmp_path: Path) -> None:
    # the bundle carries scaffold files and the winning program
    root = tmp_path / "bundle"
    task_dir, winner = _task_dir(root), _winner(root)
    output_dir = root / "output"
    result = extract_best_program_bundle(task_dir=task_dir, results_dir=_results_dir(root, winner), output_dir=output_dir)
    assert result == output_dir.resolve()
    for name in ("task.md", "evaluate.py", "capture.py"):
        assert (output_dir / name).is_file()
    assert (output_dir / "task.md").read_text(encoding="utf-8") == (task_dir / "task.md").read_text(encoding="utf-8")
    assert "winner" in _program_source(output_dir)

    # only the root runtime manifest is excluded; the extraction manifest is complete
    root = tmp_path / "runtime-manifest"
    task_dir, winner = _task_dir(root), _winner(root)
    (task_dir / ".agentic_runtime.json").write_text("root runtime")
    nested_manifest = task_dir / "resources" / ".agentic_runtime.json"
    nested_manifest.parent.mkdir()
    nested_manifest.write_text("scientific input")
    output_dir = _extract(task_dir, _results_dir(root, winner))
    assert not (output_dir / ".agentic_runtime.json").exists()
    assert (output_dir / "resources" / ".agentic_runtime.json").read_text() == "scientific input"
    assert "seed" in _program_source(output_dir, "reference_program")
    manifest = _manifest(output_dir)
    assert manifest["extraction_version"] == 1
    assert "extracted_at" in manifest
    assert manifest["combined_score"] == 0.85
    assert manifest["correct"] is True
    assert manifest["source_attempt_id"] == "idea_1_attempt_1"
    assert manifest["entrypoint_path"] == "src/program.py"
    assert manifest["parallel_worker_index"] is None

    # a bootstrap extraction (seed program, no score) succeeds
    root = tmp_path / "bootstrap"
    task_dir = _task_dir(root)
    results_dir = _results_dir(root, task_dir / "program_0", source_attempt_id="bootstrap", combined_score=None, correct=None)
    output_dir = _extract(task_dir, results_dir)
    assert (output_dir / "program_0" / "program_spec.toml").is_file()
    assert (output_dir / "reference_program" / "program_spec.toml").is_file()
    manifest = _manifest(output_dir)
    assert manifest["source_attempt_id"] == "bootstrap"
    assert manifest["combined_score"] is None
    assert manifest["correct"] is None

    # nested program files are copied
    root = tmp_path / "nested"
    task_dir = _task_dir(root)
    winner_dir = root / "winner"
    make_program_dir(winner_dir, content="import utils.helper\n")
    (winner_dir / "src" / "utils").mkdir(parents=True)
    (winner_dir / "src" / "utils" / "helper.py").write_text("def h(): pass\n", encoding="utf-8")
    output_dir = _extract(task_dir, _results_dir(root, winner_dir))
    assert (output_dir / "program_0" / "src" / "utils" / "helper.py").is_file()
    assert (output_dir / "program_0" / "program_spec.toml").is_file()
    assert (output_dir / "reference_program" / "src" / "program.py").is_file()

    # parallel_worker_index is propagated, or null when absent
    worker_cases: list[tuple[str, dict[str, Any], int | None]] = [
        ("propagated", {"parallel_worker_index": 2}, 2), ("null_when_absent", {}, None),
    ]
    for name, overrides, expected in worker_cases:
        root = tmp_path / f"worker-{name}"
        task_dir, winner = _task_dir(root), _winner(root)
        output_dir = _extract(task_dir, _results_dir(root, winner, **overrides))
        assert _manifest(output_dir)["parallel_worker_index"] == expected, name


def test_extraction_honors_copy_ignore_and_skips_evaluator_artifacts(tmp_path: Path) -> None:
    # .agentic_task_ignore entries are not copied (and not deleted from the task)
    root = tmp_path / "ignore"
    task_dir, winner = _task_dir(root), _winner(root)
    (task_dir / "resources").mkdir()
    (task_dir / "resources" / "large.bin").write_bytes(b"benchmark")
    (task_dir / ".agentic_task_ignore").write_text("resources/\n", encoding="utf-8")
    output_dir = _extract(task_dir, _results_dir(root, winner))
    assert not (output_dir / "resources").exists()
    assert (task_dir / "resources" / "large.bin").is_file()

    # evaluator artifacts are not copied into either bundle
    root = tmp_path / "artifacts"
    task_dir, winner = _task_dir(root), _winner(root)
    for program in (task_dir / "program_0", winner):
        for name in ARTIFACTS:
            (program / name).write_text("{}", encoding="utf-8")
    output_dir = _extract(task_dir, _results_dir(root, winner))
    for bundle in ("program_0", "reference_program"):
        for name in ARTIFACTS:
            assert not (output_dir / bundle / name).exists(), f"Artifact {name} should not be in {bundle}"
        assert (output_dir / bundle / "src" / "program.py").is_file()


def _missing_entrypoint(program: Path) -> None:
    make_program_dir(program, content="print('ok')\n")
    (program / "src" / "program.py").unlink()


def _invalid_spec(program: Path) -> None:
    program.mkdir(parents=True)
    (program / "program_spec.toml").write_text('schema_version = 99\ncode_root = "src"\n', encoding="utf-8")
    (program / "src").mkdir()
    (program / "src" / "program.py").write_text("print('ok')\n", encoding="utf-8")


def test_extraction_rejects_invalid_manifests_and_program_dirs(tmp_path: Path) -> None:
    # invalid or missing manifests are rejected
    for name, manifest_text in [
        ("missing_manifest_and_no_programs", None),
        ("malformed_json", "not json {{"),
        ("missing_program_dir_key", json.dumps({"entrypoint_path": "src/program.py"})),
        ("missing_entrypoint_path_key", json.dumps({"program_dir": "/some/path"})),
    ]:
        root = tmp_path / f"manifest-{name}"
        task_dir = _task_dir(root)
        results_dir = root / "results"
        results_dir.mkdir()
        if manifest_text is not None:
            (results_dir / "best_program_manifest.json").write_text(manifest_text, encoding="utf-8")
        with pytest.raises(ExtractionError):
            _extract(task_dir, results_dir)

    # invalid program directories are rejected
    for name, build in [
        ("program_dir_does_not_exist", None),
        ("missing_entrypoint_file", _missing_entrypoint),
        ("invalid_program_spec", _invalid_spec),
    ]:
        root = tmp_path / f"program-{name}"
        task_dir = _task_dir(root)
        program = root / "bad_program"
        if build is not None:
            build(program)
        with pytest.raises(ExtractionError):
            _extract(task_dir, _results_dir(root, program))

    # a program directory outside the results directory is rejected
    root = tmp_path / "outside"
    task_dir = _task_dir(root)
    outside = root / "outside_program"
    make_program_dir(outside, content="print('outside')\n")
    with pytest.raises(ExtractionError, match="outside results directory"):
        _extract(task_dir, _results_dir(root, outside, allow_external_manifest=True))

    # a symlinked program directory escaping the results directory is rejected
    root = tmp_path / "symlink"
    task_dir = _task_dir(root)
    outside = root / "outside_program"
    make_program_dir(outside, content="print('outside')\n")
    results_dir = root / "results"
    results_dir.mkdir()
    escaped = results_dir / "program_1"
    escaped.symlink_to(outside, target_is_directory=True)
    _results_dir(root, escaped, allow_external_manifest=True)
    with pytest.raises(ExtractionError, match="outside results directory"):
        _extract(task_dir, results_dir)


def test_output_dir_must_be_empty(tmp_path: Path) -> None:
    # a non-empty output directory is rejected and left untouched
    root = tmp_path / "occupied"
    task_dir, winner = _task_dir(root), _winner(root)
    output_dir = root / "output"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(ExtractionError):
        _extract(task_dir, _results_dir(root, winner))
    assert (output_dir / "existing.txt").read_text(encoding="utf-8") == "occupied"

    # an empty output directory succeeds
    root = tmp_path / "empty"
    task_dir, winner = _task_dir(root), _winner(root)
    output_dir = root / "output"
    output_dir.mkdir()
    result = extract_best_program_bundle(task_dir=task_dir, results_dir=_results_dir(root, winner), output_dir=output_dir)
    assert (result / "extraction_manifest.json").is_file()


# --- fallback scan when best_program_manifest.json is missing ------------------------------------


def _evaluated_program(parent: Path, name: str, *, content: str = "print('ok')\n", combined_score: float = 0.5, correct: bool = True) -> Path:
    program_dir = parent / name
    make_program_dir(program_dir, content=content)
    (program_dir / "metrics.json").write_text(json.dumps({"combined_score": combined_score}), encoding="utf-8")
    (program_dir / "correct.json").write_text(json.dumps({"correct": correct, "message": "ok"}), encoding="utf-8")
    return program_dir


def _idea_dir(results_dir: Path, *parts: str) -> Path:
    idea = results_dir.joinpath(*parts)
    idea.mkdir(parents=True)
    return idea


def test_scan_fallback_picks_best_correct_program_under_worker_dirs(tmp_path: Path) -> None:
    # the best correct program under a worker is found, with a warning about the missing manifest
    root = tmp_path / "worker"
    task_dir = _task_dir(root)
    results_dir = root / "results"
    idea = _idea_dir(results_dir, "worker_01", "round_000", "ideas", "01_test_idea")
    _evaluated_program(idea, "program_0", content="print('p0')\n", combined_score=0.3)
    _evaluated_program(idea, "program_1", content="print('p1_winner')\n", combined_score=0.9)
    _evaluated_program(idea, "program_2", content="print('p2')\n", combined_score=0.99, correct=False)
    with pytest.warns(UserWarning, match="best_program_manifest.json"):
        output_dir = _extract(task_dir, results_dir)
    assert "p1_winner" in _program_source(output_dir)
    manifest = _manifest(output_dir)
    assert manifest["combined_score"] == 0.9
    assert manifest["correct"] is True
    assert manifest["source_best_manifest_path"] is None

    # the scan spans multiple workers
    root = tmp_path / "multi"
    task_dir = _task_dir(root)
    results_dir = root / "results"
    _evaluated_program(_idea_dir(results_dir, "worker_01", "round_000", "ideas", "01_idea"), "program_0", content="print('w1')\n", combined_score=0.5)
    _evaluated_program(_idea_dir(results_dir, "worker_02", "round_000", "ideas", "01_idea"), "program_0", content="print('w2_best')\n", combined_score=0.8)
    assert "w2_best" in _program_source(_extract(task_dir, results_dir))

    # programs outside worker directories are ignored
    root = tmp_path / "outside"
    task_dir = _task_dir(root)
    results_dir = root / "results"
    _evaluated_program(_idea_dir(results_dir, "combined_paper_input", "_worker_01", "ideas", "01_idea"), "program_0", content="print('bad')\n", combined_score=0.99)
    _evaluated_program(_idea_dir(results_dir, "ideas", "01_idea"), "program_0", content="print('flat')\n", combined_score=0.95)
    _evaluated_program(_idea_dir(results_dir, "worker_01", "round_000", "ideas", "01_idea"), "program_0", content="print('good')\n", combined_score=0.5)
    output_dir = _extract(task_dir, results_dir)
    assert "good" in _program_source(output_dir)
    assert _manifest(output_dir)["combined_score"] == 0.5

    # no correct programs fails
    root = tmp_path / "incorrect"
    task_dir = _task_dir(root)
    results_dir = root / "results"
    _evaluated_program(_idea_dir(results_dir, "worker_01", "round_000", "ideas", "01_idea"), "program_0", combined_score=0.9, correct=False)
    with pytest.raises(ExtractionError):
        _extract(task_dir, results_dir)
