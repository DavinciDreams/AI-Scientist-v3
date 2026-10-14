"""Scientific episode observations reach paper grounding without arbitrary path reads."""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.orchestrator.paper_data import build_paper_data
from src.investigation.evaluation_evidence import EVIDENCE_FILENAME


EVIDENCE = {"schema_version": 1, "episodes": [{
    "seed": 7, "score": 2.4, "correct": True, "duration_seconds": 0.1,
    "solution": {"circles": [[0.5, 0.5, 0.1]]},
    "candidate_diagnostics": {"iterations": 3},
}]}


def _write_evaluation(program: Path) -> None:
    program.mkdir(parents=True)
    (program / "metrics.json").write_text('{"combined_score": 2.4}')
    (program / "correct.json").write_text('{"correct": true}')


def test_search_program_exposes_episode_evidence(tmp_path: Path) -> None:
    for available in (True, False):
        root = tmp_path / ("available" if available else "unavailable")
        program = root / "ideas/demo/program_1"
        _write_evaluation(program)
        if available:
            (program / EVIDENCE_FILENAME).write_text(json.dumps(EVIDENCE))
        evidence = build_paper_data(root)["all_programs"][0]["evaluation_evidence"]
        assert evidence["status"] == ("available" if available else "unavailable")
        if available:
            assert evidence["episodes"] == EVIDENCE["episodes"]
            assert evidence["candidate_diagnostics_trust"] == "unverified"


def _ablation(root: Path, program_value: str = "program_1") -> Path:
    program = root / "variants/control_best/task_bundle/program_1"
    _write_evaluation(program)
    aggregate = root / "aggregated"
    aggregate.mkdir()
    row = {"evaluation_id": "control_best_eval_1", "variant_id": "control_best",
           "program_dir": program_value, "combined_score": "2.4", "correct": "True"}
    with (aggregate / "run_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return program


def test_ablation_attaches_same_evidence_and_rejects_outside_or_symlinked_evidence(
    tmp_path: Path,
) -> None:
    # ablation attaches the same evidence to run and program
    root = tmp_path / "attached"
    program = _ablation(root)
    (program / EVIDENCE_FILENAME).write_text(json.dumps(EVIDENCE))
    data = build_paper_data(root)
    evidence = data["all_programs"][0]["evaluation_evidence"]
    run = data["ablation_studies"][0]["variants"][0]["runs"][0]
    assert evidence["episodes"] == EVIDENCE["episodes"]
    assert run["evaluation_evidence"] == evidence

    # the ablation csv cannot select outside evidence
    outside = tmp_path / "outside/program_1"
    _write_evaluation(outside)
    (outside / EVIDENCE_FILENAME).write_text("not JSON: must never read")
    root = tmp_path / "ablation"
    _ablation(root, str(outside))
    data = build_paper_data(root)
    assert data["all_programs"][0]["evaluation_evidence"]["status"] == "unavailable"

    # symlinked evidence (file or program directory) is rejected
    for symlink_part in ("file", "program"):
        root = tmp_path / f"symlink_{symlink_part}"
        program = _ablation(root)
        outside = root / "outside"
        outside.mkdir()
        (outside / EVIDENCE_FILENAME).write_text(json.dumps(EVIDENCE))
        if symlink_part == "file":
            (program / EVIDENCE_FILENAME).symlink_to(outside / EVIDENCE_FILENAME)
        else:
            moved = program.with_name("old_program")
            program.rename(moved)
            program.symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink|regular file"):
            build_paper_data(root)


def test_verified_final_exposes_sealed_episode_evidence(tmp_path: Path) -> None:
    final = tmp_path / "final"
    results = final / "results"
    results.mkdir(parents=True)
    (results / EVIDENCE_FILENAME).write_text(json.dumps(EVIDENCE))
    (final / "final_manifest.json").write_text("{}")
    manifest = {"correct": True, "metrics": {"final_score": 2.4},
                **{name: "sha256:" + "1" * 64 for name in (
                    "task_tree_sha256", "evaluator_sha256", "program_tree_sha256",
                    "results_tree_sha256")}}
    with patch("src.orchestrator.paper_data.verify_final_manifest", return_value=manifest):
        data = build_paper_data(tmp_path, final_evaluation_dir=final)
    assert data["final_evaluation"]["evaluation_evidence"]["episodes"] == EVIDENCE["episodes"]
