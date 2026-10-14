"""Frozen downstream selection for exhausted searches, and the consumers that must study it."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.final_evaluation import runner
from src.investigation import best_program_extraction
from src.investigation.program_spec import render_default_program_spec_toml
from src.pipeline.config import (
    investigation_completion_payload,
    investigation_completion_receipt_path,
)
from src.pipeline.downstream_selection import (
    load_downstream_selection,
    prepare_downstream_selection,
    preview_downstream_selection,
    resolve_downstream_program,
)
from src.variants_pipeline import extract
from tests.program_testutils import make_program_dir


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def program(path: Path, text: str = "baseline = True\n") -> Path:
    (path / "src").mkdir(parents=True)
    (path / "src/program.py").write_text(text)
    (path / "program_spec.toml").write_text(render_default_program_spec_toml())
    return path


def search(tmp_path: Path, correct: bool = True) -> Path:
    root = tmp_path / "investigation"
    baseline = program(root / ".baseline/program_0")
    write_json(root / "best_program_manifest.json", {
        "program_dir": str(baseline), "entrypoint_path": "src/program.py",
        "combined_score": 0.811173, "correct": correct, "source_attempt_id": "bootstrap",
    })
    write_json(investigation_completion_receipt_path(root), investigation_completion_payload(root))
    (root / "attempts_history.jsonl").write_text("")
    return root


def candidate(root: Path, attempt: str = "20", index: int = 1,
              score: object = 0.809734, correct: bool = True) -> Path:
    result = program(root / "ideas" / attempt / f"program_{index}", f"candidate = {index}\n")
    write_json(result / "metrics.json", {"combined_score": score})
    write_json(result / "correct.json", {"correct": correct})
    write_json(result / "evaluation_manifest.json", {
        "producer": "harness_eval", "attempt_id": attempt, "idea_id": attempt,
        "program_id": result.name, "command": ["evaluate"], "started_at": "start",
        "finished_at": "end", "exit_code": 0,
    })
    write_json(result / "rubric_status.json", {"overall_pass": False})
    history = root / "attempts_history.jsonl"
    with history.open("a") as stream:
        stream.write(json.dumps({"attempt_id": attempt, "experiment_dir": str(result.parent),
                                 "metrics": {"combined_score": score, "correct": correct}}) + "\n")
    return result


def selected_program(root: Path) -> dict[str, Any]:
    selected = prepare_downstream_selection(root)
    assert selected is not None
    return selected


# --- selection -------------------------------------------------------------------------------


def test_selection_publishes_frozen_snapshot_after_preview_and_resolves(tmp_path: Path) -> None:
    base = tmp_path / "select"
    root = search(base)
    # a missing selection resolves to None
    assert resolve_downstream_program(root) is None
    history = root / "attempts_history.jsonl"
    history.write_text("".join(json.dumps({"attempt_id": str(i), "metrics": {
        "correct": False, "combined_score": None}}) + "\n" for i in range(19)))
    source = candidate(root)
    # preview does not publish files
    preview = preview_downstream_selection(root)
    assert preview is not None
    assert preview["selection_kind"] == "non_winner_candidate"
    assert not (base / ".downstream_selection.json").exists()
    assert not (base / "downstream").exists()
    # selects the scored underperformer despite the rubric failure
    before = (root / "best_program_manifest.json").read_bytes()
    selected = prepare_downstream_selection(root)
    assert selected is not None
    assert selected["selection_kind"] == "non_winner_candidate"
    assert selected["source_program_dir"] == str(source)
    assert selected["combined_score"] == 0.809734
    assert selected["correct"] is True
    assert selected["baseline_score"] == 0.811173
    assert selected["attempt_counts"] == {"total": 20, "scored": 1, "unscored": 19}
    frozen = Path(selected["program_dir"])
    assert frozen == base / "downstream/program"
    assert (frozen / "src/program.py").read_bytes() == (source / "src/program.py").read_bytes()
    assert not (frozen / "metrics.json").exists()
    assert (root / "best_program_manifest.json").read_bytes() == before
    assert resolve_downstream_program(root) == (frozen, frozen / "src/program.py")
    assert prepare_downstream_selection(root) == selected

    # the snapshot preserves nested sources named like evaluator outputs
    root = search(tmp_path / "nested")
    source = candidate(root)
    helper = source / "src/analysis/helper.py"
    helper.parent.mkdir()
    helper.write_text("VALUE = 42\n")
    data = source / "src/metrics.json"
    data.write_text('{"parameter": 42}')
    frozen = Path(selected_program(root)["program_dir"])
    assert (frozen / "src/analysis/helper.py").read_bytes() == helper.read_bytes()
    assert (frozen / "src/metrics.json").read_bytes() == data.read_bytes()
    assert not (frozen / "metrics.json").exists()


_INVALID_CANDIDATES = [(True, True), (float("nan"), True), (float("inf"), True), (None, True), (0.9, False)]
_UNBOUND_EVALUATIONS = ["missing_manifest", "wrong_program", "wrong_attempt", "bool_exit"]


def test_invalid_candidates_fall_back_to_valid_baseline_or_none(tmp_path: Path) -> None:
    # an invalid candidate score or correctness uses the valid baseline
    for index, (score, correct) in enumerate(_INVALID_CANDIDATES):
        root = search(tmp_path / f"invalid_{index}")
        candidate(root, score=score, correct=correct)
        assert selected_program(root)["selection_kind"] == "baseline", (score, correct)

    # no valid candidate or baseline returns None
    assert prepare_downstream_selection(search(tmp_path / "none", correct=False)) is None

    # requires an authenticated terminal outcome
    root = search(tmp_path / "receipt")
    candidate(root)
    write_json(investigation_completion_receipt_path(root), {})
    assert prepare_downstream_selection(root) is None

    # requires a completed evaluation bound to the source
    for change in _UNBOUND_EVALUATIONS:
        root = search(tmp_path / change)
        source = candidate(root)
        manifest_path = source / "evaluation_manifest.json"
        if change == "missing_manifest":
            manifest_path.rename(source / "old_evaluation_manifest.json")
        else:
            manifest = json.loads(manifest_path.read_text())
            manifest.update({"wrong_program": {"program_id": "program_9"},
                             "wrong_attempt": {"attempt_id": "other"},
                             "bool_exit": {"exit_code": False}}[change])
            write_json(manifest_path, manifest)
        assert selected_program(root)["selection_kind"] == "baseline", change

    # invalid summary metadata uses the valid baseline
    root = search(tmp_path / "summary")
    source = candidate(root)
    manifest_path = source / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(producer="isolated_evaluator", schema_version=1, returncode=0,
                    output_dir=str(source), published_summaries=[["metrics.json"], "correct.json"])
    write_json(manifest_path, manifest)
    assert selected_program(root)["selection_kind"] == "baseline"

    # skips an experiment escaping the investigation
    root = search(tmp_path / "escape")
    external = candidate(search(tmp_path / "external"))
    (root / "attempts_history.jsonl").write_text(json.dumps({
        "attempt_id": "escape", "experiment_dir": str(external.parent)}) + "\n")
    assert selected_program(root)["selection_kind"] == "baseline"

    # skips candidate code containing a symlink
    root = search(tmp_path / "symlink")
    source = candidate(root)
    (source / "link.py").symlink_to(tmp_path / "outside")
    assert selected_program(root)["selection_kind"] == "baseline"


def test_candidate_ordering_skips_unfinished_and_prefers_score_then_attempt_order(tmp_path: Path) -> None:
    # skips seeded and unfinished programs
    root = search(tmp_path / "unfinished")
    candidate(root, index=0, score=4)
    pending = candidate(root, attempt="pending", score=3)
    (pending / ".evaluation_pending").touch()
    failed = candidate(root, attempt="failed", score=2)
    write_json(failed / "evaluation_manifest.json", {})
    chosen = candidate(root, attempt="valid", score=0.7)
    assert selected_program(root)["source_program_dir"] == str(chosen)

    # ties follow recorded attempt order then numeric program order
    root = search(tmp_path / "ties")
    candidate(root, "z", index=10)
    chosen = candidate(root, "z", index=2)
    candidate(root, "a", index=1)
    assert selected_program(root)["source_program_dir"] == str(chosen)

    # a higher candidate score beats attempt order
    root = search(tmp_path / "score")
    candidate(root, attempt="first", score=0.3)
    chosen = candidate(root, attempt="second", score=0.6)
    assert selected_program(root)["source_program_dir"] == str(chosen)

    # merged parallel history uses the contained worker experiment
    root = search(tmp_path / "worker")
    source = candidate(root, "worker/ideas/1")
    assert selected_program(root)["source_program_dir"] == str(source)


# --- staleness and tamper detection ---------------------------------------------------------


def test_rejects_stale_or_tampered_selection_evidence(tmp_path: Path) -> None:
    # rejects a changed source artifact without reselection
    for artifact in ("src/program.py", "metrics.json", "rubric_status.json"):
        root = search(tmp_path / artifact.replace("/", "_"))
        source = candidate(root)
        prepare_downstream_selection(root)
        (source / artifact).write_text("changed")
        with pytest.raises(ValueError, match="stale|changed|mismatch"):
            prepare_downstream_selection(root)

    # rejects a changed snapshot
    root = search(tmp_path / "snapshot")
    candidate(root)
    selected = selected_program(root)
    (Path(selected["program_dir"]) / "src/program.py").write_text("changed")
    with pytest.raises(ValueError, match="stale|changed|mismatch"):
        load_downstream_selection(root)

    # rejects changed search provenance
    for artifact in ("attempts_history.jsonl", "best_program_manifest.json"):
        root = search(tmp_path / artifact)
        candidate(root)
        prepare_downstream_selection(root)
        with (root / artifact).open("a") as stream:
            stream.write(" ")
        with pytest.raises(ValueError, match="stale|changed|mismatch"):
            load_downstream_selection(root)

    # rejects modified selection fields
    base = tmp_path / "fields"
    root = search(base)
    candidate(root)
    selected = selected_program(root)
    selected["combined_score"] = 99
    write_json(base / ".downstream_selection.json", selected)
    with pytest.raises(ValueError, match="mismatch"):
        load_downstream_selection(root)

    # rejects a changed baseline even when studying the candidate
    root = search(tmp_path / "baseline")
    candidate(root)
    selected = selected_program(root)
    (Path(selected["baseline_program_dir"]) / "src/program.py").write_text("changed")
    with pytest.raises(ValueError, match="baseline changed"):
        load_downstream_selection(root)


def test_rejects_unusable_manifest_symlinked_parent_and_final_results(tmp_path: Path) -> None:
    # rejects an unusable selection manifest
    for kind in ("malformed", "symlink_manifest"):
        base = tmp_path / kind
        root = search(base)
        manifest = base / ".downstream_selection.json"
        if kind == "malformed":
            write_json(manifest, {})
        else:
            manifest.symlink_to(base / "absent")
        with pytest.raises(ValueError):
            load_downstream_selection(root)

    # rejects a symlinked snapshot parent
    base = tmp_path / "parent"
    root = search(base)
    candidate(root)
    (base / "downstream").symlink_to(base / "outside", target_is_directory=True)
    with pytest.raises(ValueError):
        prepare_downstream_selection(root)
    assert not (base / "outside").exists()

    # refuses a new selection after final results exist
    base = tmp_path / "final"
    root = search(base)
    candidate(root)
    final = base / "final_evaluation"
    final.mkdir()
    write_json(final / "final_manifest.json", {})
    with pytest.raises(ValueError, match="final"):
        prepare_downstream_selection(root)


# --- downstream consumers (extraction, variants, final) study the frozen fallback ------------


@pytest.fixture
def fallback_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    task = tmp_path / "task"
    task.mkdir()
    (task / "evaluate.py").write_text("# evaluator\n")
    (task / "task.md").write_text("# Study\n")
    make_program_dir(task / "program_0", content="print('baseline')\n")
    investigation = tmp_path / "run" / "investigation"
    baseline = investigation / "best_program"
    make_program_dir(baseline, content="print('baseline')\n")
    write_json(investigation / "best_program_manifest.json", {
        "program_dir": str(baseline), "entrypoint_path": "src/program.py",
        "combined_score": 0.811173, "correct": True,
    })
    frozen = investigation.parent / "downstream" / "program"
    make_program_dir(frozen, content="print('candidate')\n")
    selection = {
        "program_dir": str(frozen),
        "entrypoint_path": "src/program.py",
        "selection_kind": "non_winner_candidate",
        "investigation_outcome": "completed_no_winner",
        "source_attempt_id": "attempt_20",
        "combined_score": 0.809734,
        "correct": True,
        "program_tree_sha256": "frozen-tree-digest",
    }
    monkeypatch.setattr(
        best_program_extraction, "load_downstream_selection", lambda _: selection, raising=False
    )
    monkeypatch.setattr(extract, "load_downstream_selection", lambda _: selection, raising=False)
    monkeypatch.setattr(
        runner, "resolve_downstream_program",
        lambda _: (frozen, frozen / "src/program.py"), raising=False,
    )
    return task, investigation, frozen, selection


def _invalid_selection(_: Path):
    raise ValueError("downstream selection snapshot changed")


def test_consumers_use_frozen_fallback_and_reject_changed_or_invalid_selection(
    tmp_path: Path, fallback_sources, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, investigation, frozen, selection = fallback_sources
    # consumers use the frozen fallback and keep the original baseline
    kwargs = dict(task_dir=task, results_dir=investigation, output_dir=tmp_path / "variants")
    bundle = extract.extract_bundle(**kwargs)
    assert (bundle / "program_0/src/program.py").read_text() == "print('candidate')\n"
    assert (bundle / "reference_program/src/program.py").read_text() == "print('baseline')\n"
    assert runner.resolve_frozen_incumbent(investigation) == (frozen, frozen / "src/program.py")
    manifest = json.loads((bundle / "extraction_manifest.json").read_text())
    assert manifest["downstream_selection"] == selection
    assert manifest["source_downstream_selection_fingerprint"]

    # bundle reuse rejects changed fallback evidence
    assert extract.extract_bundle(**kwargs) == bundle
    selection["baseline_score"] = 0.9
    with pytest.raises(extract.BundleValidationError, match="downstream_selection"):
        extract.extract_bundle(**kwargs)

    # consumers reject an invalid selection without using the baseline
    monkeypatch.setattr(best_program_extraction, "load_downstream_selection", _invalid_selection)
    with pytest.raises(best_program_extraction.ExtractionError, match="snapshot changed"):
        best_program_extraction.extract_best_program_bundle(
            task_dir=task, results_dir=investigation, output_dir=tmp_path / "extraction"
        )
    monkeypatch.setattr(runner, "resolve_downstream_program", _invalid_selection)
    with pytest.raises(runner.FinalEvaluationError, match="snapshot changed"):
        runner.resolve_frozen_incumbent(investigation)


@pytest.fixture
def persisted_fallback(tmp_path: Path):
    task = tmp_path / "task"
    make_program_dir(task / "program_0", content="print('current task seed')\n")
    (task / "evaluate.py").write_text("# evaluator\n")
    investigation = tmp_path / "run/investigation"
    baseline = investigation / ".baseline/program_0"
    make_program_dir(baseline, content="print('baseline')\n")
    write_json(investigation / "best_program_manifest.json", {
        "program_dir": str(baseline), "entrypoint_path": "src/program.py",
        "combined_score": 0.811173, "correct": True, "source_attempt_id": "bootstrap",
    })
    write_json(investigation_completion_receipt_path(investigation),
               investigation_completion_payload(investigation))
    candidate_dir = investigation / "ideas/attempt_20/program_1"
    make_program_dir(candidate_dir, content="print('candidate')\n")
    for name, payload in {
        "metrics.json": {"combined_score": 0.809734},
        "correct.json": {"correct": True},
        "evaluation_manifest.json": {
            "producer": "harness_eval", "attempt_id": "attempt_20", "idea_id": "idea",
            "program_id": "program_1", "command": ["evaluate"],
            "started_at": "start", "finished_at": "end", "exit_code": 0,
        },
    }.items():
        write_json(candidate_dir / name, payload)
    (investigation / "attempts_history.jsonl").write_text(json.dumps({
        "attempt_id": "attempt_20", "experiment_dir": str(candidate_dir.parent),
        "metrics": {"correct": True, "combined_score": 0.809734},
    }) + "\n")
    return task, investigation, selected_program(investigation)


def test_persisted_selection_agrees_across_consumers_and_keeps_explicit_override(
    tmp_path: Path, persisted_fallback,
) -> None:
    task, investigation, selected = persisted_fallback
    # the persisted selection agrees across extraction and final evaluation
    bundle = extract.extract_bundle(
        task_dir=task, results_dir=investigation, output_dir=tmp_path / "variants"
    )
    final_program, final_entrypoint = runner.resolve_frozen_incumbent(investigation)
    assert final_program == Path(selected["program_dir"])
    assert final_entrypoint.read_bytes() == (bundle / "program_0/src/program.py").read_bytes()
    assert (bundle / "reference_program/src/program.py").read_text() == "print('baseline')\n"
    manifest = json.loads((bundle / "extraction_manifest.json").read_text())
    assert manifest["downstream_selection"] == selected

    # the fallback bundle preserves an explicit baseline override
    override = tmp_path / "comparison"
    make_program_dir(override, content="print('explicit comparator')\n")
    kwargs = dict(
        task_dir=task, results_dir=investigation, output_dir=tmp_path / "variants_override",
        baseline_program_override=override,
    )
    bundle = extract.extract_bundle(**kwargs)
    assert (bundle / "reference_program/src/program.py").read_text() == (
        "print('explicit comparator')\n"
    )
    assert extract.extract_bundle(**kwargs) == bundle


def test_persisted_fallback_rejects_old_baseline_bundle_and_tampered_snapshot(
    tmp_path: Path, persisted_fallback, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, investigation, selected = persisted_fallback
    # an old baseline bundle cannot be reused for the fallback
    kwargs = dict(task_dir=task, results_dir=investigation, output_dir=tmp_path / "variants")
    with monkeypatch.context() as context:
        context.setattr(best_program_extraction, "load_downstream_selection", lambda _: None)
        context.setattr(extract, "load_downstream_selection", lambda _: None)
        bundle = extract.extract_bundle(**kwargs)
        assert (bundle / "program_0/src/program.py").read_text() == "print('baseline')\n"
    with pytest.raises(extract.BundleValidationError, match="source_downstream_selection"):
        extract.extract_bundle(**kwargs)

    # tampering with the persisted snapshot blocks each consumer
    (Path(selected["program_dir"]) / "src/program.py").write_text("tampered\n")
    with pytest.raises(runner.FinalEvaluationError, match="downstream selection"):
        runner.resolve_frozen_incumbent(investigation)
    with pytest.raises(extract.BundleValidationError, match="selected program"):
        extract.extract_bundle(
            task_dir=task, results_dir=investigation, output_dir=tmp_path / "variants_tampered"
        )
    with pytest.raises(best_program_extraction.ExtractionError, match="downstream selection"):
        best_program_extraction.extract_best_program_bundle(
            task_dir=task, results_dir=investigation, output_dir=tmp_path / "extraction"
        )
