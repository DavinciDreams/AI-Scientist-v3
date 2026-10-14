"""Tests for variants_pipeline materialize, rubric_contract, schemas and constants."""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.investigation.evaluation_evidence import EVIDENCE_FILENAME
from src.investigation.program_clone import clone_program_dir
from src.variants_pipeline.constants import (
    ANALYSIS_CSV_SCHEMAS,
    MAX_ABLATIONS,
    MAX_EVALS_PER_VARIANT,
    MAX_EXTRA_BASELINES,
    METRIC_VALUES_COLUMNS,
    REQUIRED_RUBRIC_IDS,
    RUN_SUMMARY_COLUMNS,
    TIMESERIES_COLUMNS,
    VALID_VARIANT_KINDS,
)
from src.variants_pipeline.host_eval import _validate_candidate
from src.variants_pipeline.materialize import (
    _copy_reference_as_program_0,
    _copy_task_files,
    _reference_differs_from_best,
    _remove_stale_files,
    materialize_variant_workspaces,
)
from src.variants_pipeline.models import VariantCandidate, VariantSelection
from src.variants_pipeline.rubric_contract import (
    build_rubric_contract,
    build_target_rubrics,
    canonicalize_variant_rubric_status,
    load_rubric_status,
    validate_variant_rubric_status,
)
from src.variants_pipeline.schemas import validate_all_analysis_files, validate_csv_schema

_SPEC_TOML = (
    'schema_version = 1\ncode_root = "src"\n\n'
    '[entrypoint]\npath = "program.py"\nlanguage = "python"\ntype = "script"\n\n'
    "[copy]\nexclude = []\ninclude = []\n\n"
    '[integration]\nkind = "generic"\n'
)
_REFERENCE_EXTRA_RUBRICS = (
    "hyperparameter_audit_complete",
    "reference_conditions_aligned",
    "reference_performance_check",
)


def _write_program(path: Path, body: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "program_spec.toml").write_text(_SPEC_TOML)
    (path / "src").mkdir(exist_ok=True)
    (path / "src" / "program.py").write_text(body)


def _create_minimal_bundle(bundle_dir: Path, reference_differs: bool = True) -> None:
    """Create a minimal extracted bundle (program_0, reference_program, task files)."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_program(bundle_dir / "program_0", "print('best program')\n")
    _write_program(
        bundle_dir / "reference_program",
        "print('reference program')\n" if reference_differs else "print('best program')\n",
    )
    (bundle_dir / "evaluate.py").write_text("# evaluator\n")
    (bundle_dir / "capture.py").write_text("# capture\n")
    (bundle_dir / "task.md").write_text("# Task\n")
    (bundle_dir / "extraction_manifest.json").write_text("{}")


def _candidate(variant_id: str, kind: str = "ablation", **overrides: Any) -> VariantCandidate:
    fields: dict[str, Any] = dict(
        variant_id=variant_id,
        title=variant_id,
        kind=kind,
        base_program="best",
        hypothesis="test",
        relevance_reason="test",
        feasibility_reason="test",
        target_components=[],
        implementation_plan=[],
        expected_effect="test",
    )
    fields.update(overrides)
    return VariantCandidate(**fields)


def _materialize(root: Path, accepted=(), *, reference_differs=False, controls=("control_best",), **kwargs):
    """Create bundle + retrofitted bundle under ``root``, materialize, return (specs, output_dir)."""
    bundle_dir, retrofitted, output = root / "bundle", root / "retrofitted", root / "output"
    _create_minimal_bundle(bundle_dir, reference_differs=reference_differs)
    _create_minimal_bundle(retrofitted, reference_differs=reference_differs)
    specs = materialize_variant_workspaces(
        bundle_dir=bundle_dir,
        retrofitted_bundle_dir=retrofitted,
        variants_run_dir=output,
        selection=VariantSelection(accepted=list(accepted), rejected=[], controls=list(controls)),
        agent="claude",
        agent_model=None,
        **kwargs,
    )
    return specs, output


def _task_md(output: Path, variant_id: str) -> str:
    return (output / "variants" / variant_id / "task_bundle" / "task.md").read_text()


def _create_reference_folder(folder: Path) -> Path:
    """Minimal reference implementation folder (no program_spec.toml)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "train.py").write_text("# DFA training code\nprint('train')\n")
    (folder / "model.py").write_text("# DFA model\nclass DFAModel: pass\n")
    (folder / "README.md").write_text("# DFA Reference\nA reference implementation.\n")
    return folder


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------


def test_reference_differs_from_best(tmp_path: Path):
    cases = [
        ("differs", True, False, True),
        ("identical", False, False, False),
        ("no_reference", True, True, False),
    ]
    for name, reference_differs, remove_reference, expected in cases:
        bundle = tmp_path / name
        _create_minimal_bundle(bundle, reference_differs=reference_differs)
        if remove_reference:
            shutil.rmtree(bundle / "reference_program")
        assert _reference_differs_from_best(bundle) is expected, name


def test_materialize_controls_baseline_reference_custom_only_and_reserved_ids(tmp_path: Path):
    # control_best plus a conditional baseline_reference when the reference differs from best
    specs, output = _materialize(tmp_path / "differs", reference_differs=True)
    control = specs[0]
    assert control.variant_id == "control_best" and control.kind == "control"
    assert (output / "variants" / "control_best").is_dir()
    assert "baseline_reference" in [s.variant_id for s in specs]
    seed = output / "variants" / "baseline_reference"
    contract = json.loads((seed / "rubric_contract.json").read_text())
    assert {r["id"] for r in contract["rubrics"]} == set(REQUIRED_RUBRIC_IDS)
    instructions = (seed / "task_bundle" / "task.md").read_text()
    assert "original seed" in instructions
    assert "control_program/" not in instructions
    assert "external source" not in instructions

    # control workspace files, without baseline_reference when the reference is identical
    specs, output = _materialize(tmp_path / "identical", reasoning_effort="xhigh")
    assert "baseline_reference" not in [s.variant_id for s in specs]
    control = output / "variants" / "control_best"
    spec_data = json.loads((control / "variant_spec.json").read_text())
    assert spec_data["variant_id"] == "control_best" and spec_data["kind"] == "control"
    contract = json.loads((control / "rubric_contract.json").read_text())
    assert {r["id"] for r in contract["rubrics"]} == set(REQUIRED_RUBRIC_IDS)
    staged = json.loads((control / "task_bundle" / "rubric_contract.json").read_text())
    assert "rubrics" in staged
    manifest = json.loads((control / "run_manifest.json").read_text())
    assert manifest["status"] == "created"
    assert manifest["eval_count"] == 0
    assert manifest["started_at"] is None
    assert manifest["reasoning_effort"] == "xhigh"
    assert "rubric_status.json" in manifest["expected_analysis_files"]
    assert "analysis/rubric_status.json" not in manifest["expected_analysis_files"]
    task_md = (control / "task_bundle" / "task.md").read_text()
    assert "# Variant:" in task_md
    assert "Do NOT modify the program" in task_md
    assert (control / "task_bundle" / "original_task.md").exists()

    # custom_only can still materialize a required control
    specs, _ = _materialize(
        tmp_path / "custom_only", reference_differs=True, controls=(), custom_only=True,
        required_controls={"control_best"},
    )
    assert [spec.variant_id for spec in specs] == ["control_best"]

    # candidates may not reuse a reserved control ID
    candidate = _candidate("control_best", "baseline", title="Counterfeit control")
    with pytest.raises(ValueError, match="reserved variant IDs: control_best"):
        _materialize(tmp_path / "reserved", [candidate], reference_differs=True)


def test_materialize_ablation_standard_baseline_and_reference_folder_baseline(tmp_path: Path):
    # ablation variant workspace
    candidate = _candidate(
        "ablation_no_dropout",
        title="Remove Dropout",
        hypothesis="Dropout is unnecessary",
        target_components=["dropout"],
        implementation_plan=["remove dropout layer", "adjust forward pass"],
        expected_effect="accuracy drops",
    )
    specs, output = _materialize(tmp_path / "ablation", [candidate], max_evals_per_variant=5)
    assert "ablation_no_dropout" in [s.variant_id for s in specs]
    variant_dir = output / "variants" / "ablation_no_dropout"
    for name in ("variant_spec.json", "rubric_contract.json", "target_rubrics.json", "run_manifest.json"):
        assert (variant_dir / name).exists()
    variant_bundle = variant_dir / "task_bundle"
    assert (variant_bundle / "program_0").is_dir()
    task_md = (variant_bundle / "task.md").read_text()
    assert "# Variant: Remove Dropout" in task_md
    assert "ablation_no_dropout" in task_md
    assert "Dropout is unnecessary" in task_md
    assert "`dropout`" in task_md
    assert "remove dropout layer" in task_md
    assert "Do NOT evaluate `program_0`" in task_md
    assert "original_task.md" in task_md
    assert "# Task" in (variant_bundle / "original_task.md").read_text()
    assert "## Detailed Description" not in task_md
    assert "**5**" in task_md
    assert "`program_1` through `program_5`" in task_md
    assert "stop immediately" in task_md
    assert "Max program attempts:** 5" in task_md

    # standard baseline workspace with a description body
    body = "## DFA Details\n\nDFA sends error directly to each layer."
    candidate = _candidate(
        "baseline_dfa",
        "baseline",
        title="Direct Feedback Alignment",
        hypothesis="DFA uses random feedback",
        target_components=["backward_pass"],
        implementation_plan=["Replace backward pass with DFA"],
        expected_effect="Lower accuracy",
        description_body=body,
    )
    _, output = _materialize(tmp_path / "baseline", [candidate])
    task_md = _task_md(output, "baseline_dfa")
    assert "## Detailed Description" in task_md
    assert "DFA sends error directly to each layer" in task_md
    assert "DFA uses random feedback" in task_md
    assert "Replace backward pass with DFA" in task_md
    assert "implementing the simpler approach" in task_md
    assert "control_program" not in task_md
    spec_data = json.loads((output / "variants" / "baseline_dfa" / "variant_spec.json").read_text())
    assert spec_data["description_body"] == body

    # reference-folder baseline workspace
    ref_folder = _create_reference_folder(tmp_path / "dfa_ref")
    candidate = _candidate(
        "baseline_dfa", "baseline", title="DFA", base_program="reference",
        hypothesis="DFA uses random feedback", reference_dir=ref_folder,
    )
    specs, output = _materialize(tmp_path / "reference", [candidate])
    tb = next(s for s in specs if s.variant_id == "baseline_dfa").variant_dir / "task_bundle"
    assert (tb / "program_0" / "train.py").is_file()
    assert (tb / "program_0" / "model.py").is_file()
    assert "DFA training" in (tb / "program_0" / "train.py").read_text()
    assert (tb / "control_program").is_dir()
    assert "best program" in (tb / "control_program" / "src" / "program.py").read_text()
    task_md = _task_md(output, "baseline_dfa")
    for needle in (
        "reference implementation", "control_program", "core algorithm", "Hyperparameter Audit",
        "hyperparameter_comparison.md", "align", "preserve", "DFA uses random feedback",
    ):
        assert needle in task_md
    assert "adapt" in task_md.lower()


def test_copy_task_files_excludes_results_logs_root_runtime_manifest_and_ignored_paths(tmp_path: Path):
    # excludes results and logs
    src = tmp_path / "excl" / "src_bundle"
    src.mkdir(parents=True)
    excluded = (
        "correct.json", "metrics.json", "_results.json", "extraction_manifest.json",
        ".agentic_runtime.json", "job-12345.out", "job-12345.err", "agent.log", "run.log",
    )
    for name in excluded + ("evaluate.py", "task.md"):
        (src / name).write_text("x")
    dst = tmp_path / "excl" / "dst"
    _copy_task_files(src, dst)
    assert not any((dst / name).exists() for name in excluded)
    assert (dst / "evaluate.py").exists() and (dst / "task.md").exists()

    # preserves a nested runtime manifest (scientific input) while dropping the root one
    src = tmp_path / "nested" / "src_bundle"
    nested = src / "resources" / ".agentic_runtime.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("scientific input")
    (src / ".agentic_runtime.json").write_text("root runtime")
    dst = tmp_path / "nested" / "dst"
    _copy_task_files(src, dst)
    assert not (dst / ".agentic_runtime.json").exists()
    assert (dst / "resources" / ".agentic_runtime.json").read_text() == "scientific input"

    # reuse of an existing destination removes only the root runtime manifest
    src = tmp_path / "reuse" / "src_bundle"
    src.mkdir(parents=True)
    (src / "task.md").write_text("task")
    dst = tmp_path / "reuse" / "dst"
    nested = dst / "resources" / ".agentic_runtime.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("scientific input")
    (dst / ".agentic_runtime.json").write_text("stale runtime")
    _copy_task_files(src, dst)
    assert not (dst / ".agentic_runtime.json").exists()
    assert nested.read_text() == "scientific input"

    # honors .agentic_task_ignore
    src = tmp_path / "ignore" / "src_bundle"
    (src / "resources").mkdir(parents=True)
    (src / "resources" / "large.bin").write_bytes(b"benchmark")
    (src / ".agentic_task_ignore").write_text("resources/\n")
    (src / "evaluate.py").write_text("# eval")
    dst = tmp_path / "ignore" / "dst"
    _copy_task_files(src, dst)
    assert not (dst / "resources").exists()
    assert (dst / "evaluate.py").is_file()


def test_remove_stale_files(tmp_path: Path):
    prog = tmp_path / "program_0"
    (prog / "src").mkdir(parents=True)
    stale = (
        "evaluator_stdout.log", "evaluator_stderr.log", "metrics.json", "correct.json",
        "evaluation_manifest.json", "rubric_status.json", "_results.json", EVIDENCE_FILENAME,
    )
    for name in stale:
        (prog / name).write_text("{}")
    (prog / "src" / "_results.json").write_text("{}")
    (prog / "src" / "program.py").write_text("pass\n")
    (prog / "program_spec.toml").write_text("ok")
    nested_manifest = prog / "resources" / ".agentic_runtime.json"
    nested_manifest.parent.mkdir()
    nested_manifest.write_text("scientific input")

    _remove_stale_files(prog)

    assert not any((prog / name).exists() for name in stale)
    assert not (prog / "src" / "_results.json").exists()
    assert (prog / "src" / "program.py").read_text() == "pass\n"
    assert (prog / "program_spec.toml").exists()
    assert nested_manifest.read_text() == "scientific input"


def test_copy_reference_as_program_0_generates_spec_filters_and_preserves_existing_spec(tmp_path: Path):
    # generates a spec and filters baseline.md / __pycache__
    ref = _create_reference_folder(tmp_path / "gen" / "dfa_ref")
    (ref / "baseline.md").write_text("---\nvariant_id: baseline_dfa\n---\n")
    (ref / "__pycache__").mkdir()
    (ref / "__pycache__" / "module.cpython-311.pyc").write_bytes(b"\x00")
    dst = tmp_path / "gen" / "program_0"
    _copy_reference_as_program_0(ref, dst)
    for name in ("train.py", "model.py", "README.md", "program_spec.toml", "src/program.py"):
        assert (dst / name).is_file()
    assert not (dst / "baseline.md").exists()
    assert not (dst / "__pycache__").exists()

    # preserves an existing program_spec.toml and entrypoint
    ref = _create_reference_folder(tmp_path / "keep" / "dfa_ref")
    (ref / "program_spec.toml").write_text(_SPEC_TOML)
    (ref / "src").mkdir()
    (ref / "src" / "program.py").write_text("# real entrypoint\n")
    dst = tmp_path / "keep" / "program_0"
    _copy_reference_as_program_0(ref, dst)
    assert "real entrypoint" in (dst / "src" / "program.py").read_text()


# ---------------------------------------------------------------------------
# Evaluator evidence must not be cloned into fresh candidates
# ---------------------------------------------------------------------------


def test_evaluator_evidence_never_cloned_into_fresh_candidates(tmp_path: Path):
    # clone excludes evidence even when the spec explicitly includes it
    for index, include in enumerate(("[]", '["evaluation_evidence.json"]', '["**"]')):
        root = tmp_path / f"clone_{index}"
        bundle = root / "bundle"
        _create_minimal_bundle(bundle)
        source = bundle / "program_0"
        evidence = source / EVIDENCE_FILENAME
        evidence.write_bytes(b"archived evaluator observations")
        spec = source / "program_spec.toml"
        spec.write_text(spec.read_text().replace("include = []", f"include = {include}"))
        destination = root / "fresh" / "program_0"

        clone_program_dir(src_dir=source, dst_dir=destination)

        assert not (destination / EVIDENCE_FILENAME).exists(), include
        assert evidence.read_bytes() == b"archived evaluator observations", include
        assert (destination / "src/program.py").read_bytes() == (source / "src/program.py").read_bytes(), include
        assert _validate_candidate(destination.parent, 0) == (destination, "src/program.py"), include

    # materialized control and reference workspaces exclude archived evidence
    bundle = tmp_path / "materialized" / "bundle"
    _create_minimal_bundle(bundle)
    for name in ("program_0", "reference_program"):
        (bundle / name / EVIDENCE_FILENAME).write_bytes(b"archived observations")
    output = tmp_path / "materialized" / "variants_run"
    specs = materialize_variant_workspaces(
        bundle_dir=bundle, retrofitted_bundle_dir=bundle, variants_run_dir=output,
        selection=VariantSelection(accepted=[], rejected=[], controls=["control_best"]),
        agent="codex", agent_model=None,
    )
    assert {spec.variant_id for spec in specs} == {"control_best", "baseline_reference"}
    for variant_spec in specs:
        task = output / "variants" / variant_spec.variant_id / "task_bundle"
        assert _validate_candidate(task, 0) == (task / "program_0", "src/program.py")
    for name in ("program_0", "reference_program"):
        assert (bundle / name / EVIDENCE_FILENAME).read_bytes() == b"archived observations"

    # the evaluator still rejects a candidate that contains evidence
    rejected = tmp_path / "rejected"
    _create_minimal_bundle(rejected)
    (rejected / "program_0" / EVIDENCE_FILENAME).write_text("{}")
    with pytest.raises(ValueError, match="candidate contains evaluator artifact"):
        _validate_candidate(rejected, 0)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_constants_contract():
    assert len(RUN_SUMMARY_COLUMNS) == 12
    assert len(METRIC_VALUES_COLUMNS) == 5
    assert len(TIMESERIES_COLUMNS) == 5
    assert len(REQUIRED_RUBRIC_IDS) == 9
    assert len(REQUIRED_RUBRIC_IDS) == len(set(REQUIRED_RUBRIC_IDS))
    assert set(ANALYSIS_CSV_SCHEMAS) == {"run_summary.csv", "metric_values.csv", "timeseries.csv"}
    assert (MAX_ABLATIONS, MAX_EXTRA_BASELINES, MAX_EVALS_PER_VARIANT) == (3, 3, 2)
    assert VALID_VARIANT_KINDS == {"control", "baseline", "ablation"}


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


def _write_csv(path: Path, columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(columns)


def _write_analysis_dir(analysis_dir: Path, *, run_summary=RUN_SUMMARY_COLUMNS, timeseries=True) -> None:
    analysis_dir.mkdir(exist_ok=True)
    _write_csv(analysis_dir / "run_summary.csv", run_summary)
    _write_csv(analysis_dir / "metric_values.csv", METRIC_VALUES_COLUMNS)
    if timeseries:
        _write_csv(analysis_dir / "timeseries.csv", TIMESERIES_COLUMNS)


_CSV_SCHEMA_CASES = [
    # (name, written columns [None = no file, [] = empty file], expected columns, passes, error needles)
    ("valid", ["a", "b", "c"], ["a", "b", "c"], True, ()),
    ("missing_file", None, ["a"], False, ("does not exist",)),
    ("empty_file", [], ["a"], False, ("empty",)),
    ("missing_columns", ["a", "b"], ["a", "b", "c"], False, ("Missing columns", "'c'")),
    ("extra_columns_ok", ["a", "b", "c", "d"], ["a", "b"], True, ()),
]


def test_validate_csv_schema(tmp_path: Path):
    for name, written, expected, passed, needles in _CSV_SCHEMA_CASES:
        p = tmp_path / f"{name}.csv"
        if written == []:
            p.write_text("")
        elif written is not None:
            _write_csv(p, written)
        ok, error = validate_csv_schema(p, expected)
        assert ok is passed, name
        if passed:
            assert error is None, name
        else:
            assert error is not None, name
            for needle in needles:
                assert needle in error, name


def test_validate_all_analysis_files_valid_missing_file_missing_dir_and_wrong_schema(tmp_path: Path):
    # valid
    _write_analysis_dir(tmp_path / "valid")
    assert validate_all_analysis_files(tmp_path / "valid") == (True, [])
    # missing one file
    _write_analysis_dir(tmp_path / "missing_file", timeseries=False)
    passed, errors = validate_all_analysis_files(tmp_path / "missing_file")
    assert not passed and len(errors) == 1
    # missing directory
    passed, errors = validate_all_analysis_files(tmp_path / "nonexistent")
    assert not passed and len(errors) == 1
    # wrong schema
    _write_analysis_dir(tmp_path / "wrong_schema", run_summary=["wrong", "columns"])
    passed, errors = validate_all_analysis_files(tmp_path / "wrong_schema")
    assert not passed
    assert any("Missing columns" in e for e in errors)


# ---------------------------------------------------------------------------
# rubric_contract
# ---------------------------------------------------------------------------


def _status(variant_id: str = "v1", all_pass: bool = True) -> dict:
    return {
        "variant_id": variant_id,
        "overall_pass": all_pass,
        "rubrics": [{"id": rid, "pass": all_pass, "evidence": "test evidence"} for rid in REQUIRED_RUBRIC_IDS],
    }


def _canonicalize(status, **overrides):
    kwargs = dict(
        variant_id="v1", attempt_id="attempt-1", idea_id="idea-1",
        required_rubric_ids=REQUIRED_RUBRIC_IDS, combined_score=1.0, correct=True,
    )
    kwargs.update(overrides)
    return canonicalize_variant_rubric_status(status, **kwargs)


def test_build_rubric_contract_target_rubrics_and_reference_baseline_extras():
    contract = build_rubric_contract("my_variant", "baseline")
    assert contract["variant_id"] == "my_variant"
    assert contract["variant_kind"] == "baseline"
    assert contract["contract_version"] == 1
    assert {r["id"] for r in contract["rubrics"]} == set(REQUIRED_RUBRIC_IDS)
    for rubric in contract["rubrics"]:
        assert {"id", "description", "type"} <= set(rubric)
    rubrics = build_target_rubrics(build_rubric_contract("v", "control"))
    assert isinstance(rubrics, list) and len(rubrics) == len(REQUIRED_RUBRIC_IDS)
    for r in rubrics:
        assert "id" in r and "description" in r

    # reference-based baselines get three extra rubrics; best-based ones do not
    ids = [r["id"] for r in build_rubric_contract("baseline_x", "baseline", base_program="reference")["rubrics"]]
    assert all(extra in ids for extra in _REFERENCE_EXTRA_RUBRICS)
    assert len(ids) == len(REQUIRED_RUBRIC_IDS) + 3
    ids = [r["id"] for r in build_rubric_contract("baseline_x", "baseline", base_program="best")["rubrics"]]
    assert set(ids) == set(REQUIRED_RUBRIC_IDS)
    assert not any(extra in ids for extra in _REFERENCE_EXTRA_RUBRICS)


def _with(mutate):
    status = _status("v1")
    mutate(status)
    return status


_INVALID_STATUS_CASES = [
    # (name, status, extra required rubric ids, error needle)
    ("none_status", None, None, "missing"),
    ("missing_key", {"variant_id": "v1", "overall_pass": True}, None, "rubrics"),
    ("variant_id_mismatch", _status("wrong_id"), None, "mismatch"),
    (
        "missing_rubric_ids",
        {"variant_id": "v1", "overall_pass": True,
         "rubrics": [{"id": "analysis_files_present", "pass": True, "evidence": "test evidence"}]},
        None, "missing rubric ids",
    ),
    (
        "overall_pass_inconsistency",
        {"variant_id": "v1", "overall_pass": True,
         "rubrics": [{"id": rid, "pass": rid != "analysis_files_present", "evidence": "test evidence"}
                     for rid in REQUIRED_RUBRIC_IDS]},
        None, "does not match required rubrics",
    ),
    ("caller_supplied_reference_rubrics", _status("v1"), {*REQUIRED_RUBRIC_IDS, "reference_performance_check"},
     "reference_performance_check"),
    ("missing_evidence", _with(lambda s: s["rubrics"][0].pop("evidence")), None, "evidence"),
    ("non_boolean_pass", _with(lambda s: s["rubrics"][0].__setitem__("pass", 1)), None, "boolean"),
]


def test_validate_variant_rubric_status_accepts_valid_and_rejects_invalid():
    assert validate_variant_rubric_status(_status("v1"), "v1") == (True, None)
    for name, status, required, needle in _INVALID_STATUS_CASES:
        extra = (required,) if required else ()
        passed, error = validate_variant_rubric_status(status, "v1", *extra)
        assert not passed, name
        assert needle in error, name


def test_canonical_status_trusted_identity_duplicate_rubrics_and_host_correctness():
    # uses trusted identity, metrics and overall_pass
    status = _status("agent-chosen")
    status.pop("overall_pass")
    canonical, error = _canonicalize(
        status, variant_id="trusted-variant", attempt_id="trusted-attempt", idea_id="trusted-idea",
        combined_score=9.5,
    )
    assert error is None and canonical is not None
    assert canonical["variant_id"] == "trusted-variant"
    assert canonical["attempt_id"] == "trusted-attempt"
    assert canonical["idea_id"] == "trusted-idea"
    assert canonical["overall_pass"] is True
    assert canonical["metrics"] == {"combined_score": 9.5, "correct": True}

    # rejects duplicate or extra rubrics
    status = _status("v1")
    status["rubrics"].append(dict(status["rubrics"][0]))
    canonical, error = _canonicalize(status)
    assert canonical is None
    assert "duplicate" in error

    # cannot pass when host correctness fails
    canonical, error = _canonicalize(_status("v1"), correct=False)
    assert error is None and canonical is not None
    assert canonical["overall_pass"] is False
    assert canonical["metrics"]["correct"] is False


def _symlinked_file(root: Path) -> Path:
    target = root / "target.json"
    target.write_text(json.dumps({"rubrics": []}))
    path = root / "rubric_status.json"
    path.symlink_to(target)
    return path


def _symlinked_parent(root: Path) -> Path:
    outside = root / "outside"
    outside.mkdir()
    (outside / "rubric_status.json").write_text(json.dumps({"rubrics": []}))
    linked_parent = root / "program_0"
    linked_parent.symlink_to(outside, target_is_directory=True)
    return linked_parent / "rubric_status.json"


def _oversized(root: Path) -> Path:
    path = root / "rubric_status.json"
    path.write_text("{" + (" " * 1_048_576) + "}")
    return path


def _invalid_json(root: Path) -> Path:
    path = root / "bad.json"
    path.write_text("not json")
    return path


_REJECTED_STATUS_FILES = [
    ("missing", lambda root: root / "nonexistent.json"),
    ("invalid_json", _invalid_json),
    ("symlink", _symlinked_file),
    ("symlinked_parent", _symlinked_parent),
    ("oversized", _oversized),
]


def test_load_rubric_status_accepts_valid_and_rejects_unsafe_files(tmp_path: Path):
    p = tmp_path / "rubric_status.json"
    p.write_text(json.dumps({"variant_id": "v1", "overall_pass": True, "rubrics": []}))
    result = load_rubric_status(p)
    assert result is not None and result["variant_id"] == "v1"
    for name, make_path in _REJECTED_STATUS_FILES:
        root = tmp_path / name
        root.mkdir()
        assert load_rubric_status(make_path(root)) is None, name


def test_placeholder_script_emits_validator_identity(tmp_path: Path) -> None:
    (tmp_path / "active_rubrics.json").write_text(
        json.dumps([{"id": "analysis_files_present", "description": "present"}])
    )
    script = Path(__file__).parent.parent.parent / "src/agents/skills/verify-rubrics/scripts/create_rubric_status_placeholder.py"
    subprocess.run(
        [sys.executable, str(script), str(tmp_path), "--variant-id", "control_best"],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads((tmp_path / "rubric_status.json").read_text())
    assert payload["variant_id"] == "control_best"
