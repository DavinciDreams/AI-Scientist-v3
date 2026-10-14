"""Post-hoc pipeline stages: extract, bootstrap, generate, select, materialize, publish, tree digest."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import stat
from typing import Any, cast
from unittest.mock import patch

import pytest

from src.investigation.evaluation_evidence import EVIDENCE_FILENAME
from src.posthoc_pipeline import tree_digest as tree_digest_module
from src.posthoc_pipeline.bootstrap import _reset_bootstrap_bundle, _run_bootstrap_agent, bootstrap_analysis_bundle
from src.posthoc_pipeline.extract import (
    BundleValidationError,
    _stage_evidence,
    extract_bundle,
    restore_verified_evidence,
    verify_staged_evidence,
)
from src.posthoc_pipeline.generate import _build_generation_prompt, _parse_posthoc_bundle, generate_analysis_candidates
from src.posthoc_pipeline.materialize import materialize_analysis_workspaces
from src.posthoc_pipeline.models import AnalysisCandidate, AnalysisRunResult, AnalysisSelection, AnalysisSpec, AttemptInspection
from src.posthoc_pipeline.publish import aggregate_and_publish_valid_analyses
from src.posthoc_pipeline.select import _apply_selection_constraints, _parse_selection_decisions, validate_and_select_analyses
from src.posthoc_pipeline.tree_digest import TreeDigestError, tree_digest
from tests.program_testutils import make_program_dir
from tests.paper.test_paper_evaluation_evidence import EVIDENCE

RUN_KWARGS: dict[str, Any] = dict(timeout_seconds=10, timeout_grace_seconds=1)


# --------------------------------------------------------------------------- shared helpers


def _candidate(analysis_id: str, family: str = "mechanistic", **overrides: Any) -> AnalysisCandidate:
    fields: dict[str, Any] = dict(
        analysis_id=analysis_id,
        title=analysis_id,
        family=family,
        hypothesis="hyp",
        relevance_reason="rel",
        feasibility_reason="feas",
        required_artifacts=[],
        requires_retrain=False,
        implementation_plan=[],
        expected_claim="claim",
        expected_outputs=[],
    )
    fields.update(overrides)
    return AnalysisCandidate(**fields)


def _write_completed_evidence_anchor(evidence_dir: Path) -> None:
    (evidence_dir / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "total_variants": 1,
                "total_completed": 1,
                "variants": [{"variant_id": "control_best", "status": "completed"}],
            }
        ),
        encoding="utf-8",
    )


def _make_evidence_dir(root: Path, *, csv: str | None = "variant,value\nbest,1\n", anchor: bool = True) -> Path:
    evidence_dir = root / "ablations"
    aggregate = evidence_dir / "aggregated" / "task"
    aggregate.mkdir(parents=True)
    if csv is not None:
        (aggregate / "metric_values.csv").write_text(csv, encoding="utf-8")
    if anchor:
        _write_completed_evidence_anchor(evidence_dir)
    return evidence_dir


# --------------------------------------------------------------------------- extract


def _make_task_dir(root: Path) -> Path:
    task_dir = root / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.md").write_text("# Goal\nAnalyze model behavior\n", encoding="utf-8")
    (task_dir / "evaluate.py").write_text("print('eval')\n", encoding="utf-8")
    (task_dir / "capture.py").write_text("print('capture')\n", encoding="utf-8")
    (task_dir / "settings.yaml").write_text("seed: 7\n", encoding="utf-8")
    (task_dir / "helpers").mkdir()
    (task_dir / "helpers" / "util.py").write_text("VALUE = 7\n", encoding="utf-8")
    (task_dir / "helper_link.py").symlink_to("helpers/util.py")
    make_program_dir(task_dir / "program_0", content="print('seed')\n")
    return task_dir


def _make_results_dir(root: Path) -> Path:
    winner_dir = root / "winner"
    make_program_dir(winner_dir, content="print('winner')\n")
    results_dir = root / "results"
    results_dir.mkdir(parents=True)
    managed_winner_dir = results_dir / "winner"
    shutil.copytree(winner_dir, managed_winner_dir, symlinks=True)
    manifest = {
        "program_dir": str(managed_winner_dir),
        "entrypoint_path": "src/program.py",
        "combined_score": 0.91,
        "correct": True,
        "source_attempt_id": "idea_1_attempt_1",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }
    (results_dir / "best_program_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results_dir


@pytest.fixture
def extract_inputs(tmp_path: Path) -> dict[str, Path]:
    return {"task_dir": _make_task_dir(tmp_path), "results_dir": _make_results_dir(tmp_path)}


def test_extract_bundle_preserves_artifacts_stages_read_only_evidence_and_restores_it(
    tmp_path: Path, extract_inputs: dict[str, Path]
) -> None:
    # preserves task-level artifacts, including relative symlinks
    output_dir = tmp_path / "posthoc_run"

    bundle_dir = extract_bundle(**extract_inputs, output_dir=output_dir)

    assert bundle_dir == output_dir / "_extracted_bundle"
    assert (bundle_dir / "program_0" / "src" / "program.py").read_text(encoding="utf-8") == "print('winner')\n"
    assert (bundle_dir / "reference_program" / "src" / "program.py").read_text(encoding="utf-8") == "print('seed')\n"
    assert (bundle_dir / "capture.py").read_text(encoding="utf-8") == "print('capture')\n"
    assert (bundle_dir / "settings.yaml").read_text(encoding="utf-8") == "seed: 7\n"
    assert (bundle_dir / "helpers" / "util.py").read_text(encoding="utf-8") == "VALUE = 7\n"
    assert (bundle_dir / "helper_link.py").is_symlink()
    assert (bundle_dir / "helper_link.py").readlink() == Path("helpers/util.py")

    # stages the ablations bundle as read-only evidence, excluding untrusted files
    evidence_dir = _make_evidence_dir(tmp_path)
    aggregate = evidence_dir / "aggregated" / "task"
    (evidence_dir / "README.md").write_text("# Variants\n", encoding="utf-8")
    (aggregate / "untrusted_notes.md").write_text("private\n", encoding="utf-8")
    variant_dir = evidence_dir / "variants" / "control_best"
    variant_dir.mkdir(parents=True)
    (variant_dir / "run_manifest.json").write_text('{"variant_dir":"/private/host/path"}\n', encoding="utf-8")

    source_bundle = extract_bundle(**extract_inputs, evidence_dir=evidence_dir, output_dir=tmp_path / "posthoc_run_evidence")

    staged_csv = source_bundle / "evidence" / "aggregated" / "task" / "metric_values.csv"
    assert staged_csv.is_file()
    assert stat.S_IMODE(staged_csv.stat().st_mode) == 0o400
    manifest = json.loads((source_bundle / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert "evidence/aggregated/task/metric_values.csv" in manifest["files"]
    assert not (source_bundle / "evidence" / "aggregated" / "task" / "untrusted_notes.md").exists()
    assert not (source_bundle / "evidence" / "variants").exists()

    # restore_verified_evidence replaces an agent mutation with the verified copy
    target_bundle = tmp_path / "analysis_ready"
    forged = target_bundle / "evidence" / "forged.md"
    forged.parent.mkdir(parents=True)
    forged.write_text("fake evidence\n", encoding="utf-8")
    (target_bundle / "evidence_manifest.json").write_text("{}\n", encoding="utf-8")

    restore_verified_evidence(source_bundle, target_bundle)

    assert not forged.exists()
    restored = target_bundle / "evidence" / "aggregated" / "task" / "metric_values.csv"
    assert restored.read_text(encoding="utf-8") == "variant,value\nbest,1\n"


def _symlinked_evidence(root: Path) -> Path:
    evidence_dir = _make_evidence_dir(root, csv=None)
    protected = root / "protected.csv"
    protected.write_text("secret\n", encoding="utf-8")
    (evidence_dir / "aggregated" / "task" / "metric_values.csv").symlink_to(protected)
    return evidence_dir


_INVALID_EVIDENCE_CASES: list[tuple[str, Callable[[Path], Path], str]] = [
    ("symlink_in_evidence_tree", _symlinked_evidence, "contains symlink"),
    ("incomplete_bundle", lambda root: _make_evidence_dir(root, csv=None, anchor=False), "pipeline_summary"),
]


def test_extract_bundle_rejects_invalid_evidence_and_missing_entries_without_partial_trees(
    tmp_path: Path, extract_inputs: dict[str, Path]
) -> None:
    # rejects symlinked and incomplete evidence bundles
    for name, build_evidence, match in _INVALID_EVIDENCE_CASES:
        root = tmp_path / name
        root.mkdir()
        evidence_dir = build_evidence(root)

        with pytest.raises(BundleValidationError, match=match):
            extract_bundle(**extract_inputs, evidence_dir=evidence_dir, output_dir=root / "posthoc_run")

    # failed evidence staging leaves no partial tree
    evidence_dir = _make_evidence_dir(tmp_path, csv="too large\n")
    output_dir = tmp_path / "posthoc_run"

    with patch("src.posthoc_pipeline.extract._MAX_EVIDENCE_FILE_BYTES", 1):
        with pytest.raises(BundleValidationError, match="safety limit"):
            extract_bundle(**extract_inputs, evidence_dir=evidence_dir, output_dir=output_dir)

    assert not (output_dir / "_extracted_bundle" / "evidence").exists()
    assert not list((output_dir / "_extracted_bundle").glob(".evidence-staging-*"))

    # required bundle entries are validated after extraction
    def fake_extract(*, task_dir: Path, results_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        make_program_dir(output_dir / "program_0", content="print('winner')\n")
        make_program_dir(output_dir / "reference_program", content="print('seed')\n")
        (output_dir / "evaluate.py").write_text("print('eval')\n", encoding="utf-8")
        (output_dir / "extraction_manifest.json").write_text("{}", encoding="utf-8")

    with patch("src.posthoc_pipeline.extract.extract_best_program_bundle") as mock_extract:
        mock_extract.side_effect = fake_extract

        with pytest.raises(BundleValidationError, match="task.md"):
            extract_bundle(
                task_dir=tmp_path / "missing" / "task",
                results_dir=tmp_path / "missing" / "results",
                output_dir=tmp_path / "missing" / "posthoc_run",
            )


# --------------------------------------------------------------------------- extract: evaluator evidence records


def _evidence_source(root: Path) -> Path:
    program = root / "variants/control_best/task_bundle/program_0"
    program.mkdir(parents=True)
    (root / "aggregated/task").mkdir(parents=True)
    _write_completed_evidence_anchor(root)
    return program


def test_posthoc_stages_episode_records_and_rejects_invalid_episode_artifacts(tmp_path: Path) -> None:
    # stages the episode record and verifies its hash
    root = tmp_path / "valid"
    source = root / "ablations"
    program = _evidence_source(source)
    (program / EVIDENCE_FILENAME).write_text(json.dumps(EVIDENCE))
    bundle = root / "bundle"
    bundle.mkdir()

    _stage_evidence(source, bundle)

    relative = "evidence/variants/control_best/task_bundle/program_0/" + EVIDENCE_FILENAME
    assert json.loads((bundle / relative).read_text()) == EVIDENCE
    assert relative in verify_staged_evidence(bundle)["files"]

    # rejects malformed records and symlinked records or program dirs
    for invalid in ("malformed", "symlink", "program_symlink"):
        root = tmp_path / invalid
        source = root / "ablations"
        program = _evidence_source(source)
        if invalid == "malformed":
            (program / EVIDENCE_FILENAME).write_text('{"episodes": []}')
        else:
            outside = root / "outside"
            outside.mkdir()
            (outside / EVIDENCE_FILENAME).write_text(json.dumps(EVIDENCE))
            if invalid == "symlink":
                (program / EVIDENCE_FILENAME).symlink_to(outside / EVIDENCE_FILENAME)
            else:
                program.rmdir()
                program.symlink_to(outside, target_is_directory=True)
        bundle = root / "bundle"
        bundle.mkdir()

        with pytest.raises(BundleValidationError, match="evidence|symlink"):
            _stage_evidence(source, bundle)


# --------------------------------------------------------------------------- bootstrap


def _evaluator_source(score: float) -> str:
    return (
        "import argparse, json\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--program_path', required=True)\n"
        "parser.add_argument('--results_dir', required=True)\n"
        "args = parser.parse_args()\n"
        "results_dir = Path(args.results_dir)\n"
        "program_dir = Path(args.program_path).resolve().parent.parent\n"
        "results_dir.mkdir(parents=True, exist_ok=True)\n"
        f"(results_dir / 'metrics.json').write_text(json.dumps({{'combined_score': {score}, 'val_accuracy': {score}}}), encoding='utf-8')\n"
        "(results_dir / 'correct.json').write_text(json.dumps({'correct': True}), encoding='utf-8')\n"
        "import pickle\n"
        "(program_dir / 'best_checkpoint.pt').write_bytes(pickle.dumps({'weights': [1.0]}))\n"
        f"(program_dir / '_results.json').write_text(json.dumps({{'val_accuracy': {score}, 'checkpoint_path': 'best_checkpoint.pt'}}), encoding='utf-8')\n"
    )


def _make_source_program(
    root: Path,
    *,
    metrics: dict[str, float] | None = None,
    results: dict[str, Any] | None = None,
    checkpoint: bool = False,
) -> Path:
    source_program_dir = root / "source_program"
    make_program_dir(source_program_dir, content="print('winner')\n")
    if metrics is not None:
        (source_program_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (source_program_dir / "correct.json").write_text(json.dumps({"correct": True}), encoding="utf-8")
    if results is not None:
        (source_program_dir / "_results.json").write_text(json.dumps(results), encoding="utf-8")
    if checkpoint:
        (source_program_dir / "best_model.pt").write_bytes(pickle.dumps({"weights": [1.0]}))
    return source_program_dir


_WINNER_METRICS = {"combined_score": 0.91, "val_accuracy": 0.91}


def _checkpointless_source(root: Path) -> Path:
    return _make_source_program(root, metrics=_WINNER_METRICS, results={"val_accuracy": 0.91})


def _make_bundle(root: Path, source_program_dir: Path, *, evaluator_score: float = 0.89) -> Path:
    bundle_dir = root / "bundle"
    bundle_dir.mkdir(parents=True)
    make_program_dir(bundle_dir / "program_0", content="print('winner')\n")
    make_program_dir(bundle_dir / "reference_program", content="print('seed')\n")
    (bundle_dir / "task.md").write_text("# Goal\nAnalyze\n", encoding="utf-8")
    (bundle_dir / "evaluate.py").write_text(_evaluator_source(evaluator_score), encoding="utf-8")
    (bundle_dir / "extraction_manifest.json").write_text(
        json.dumps({"source_best_program_dir": str(source_program_dir), "combined_score": 0.91, "correct": True}),
        encoding="utf-8",
    )
    return bundle_dir


def _write_agent_script(path: Path, body: str = "exit 0\n") -> Path:
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _bootstrap(bundle_dir: Path, output_bundle_dir: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        agent="claude",
        agent_model=None,
        max_attempts=2,
        primary_metric_key="val_accuracy",
        primary_metric_tolerance=0.0,
        retrain_if_checkpoint_missing=True,
        **RUN_KWARGS,
    )
    kwargs.update(overrides)
    return bootstrap_analysis_bundle(bundle_dir=bundle_dir, output_bundle_dir=output_bundle_dir, **kwargs)


def test_bootstrap_excludes_evidence_reuses_checkpoint_honors_optional_mode_and_default_runner(tmp_path: Path) -> None:
    # the bootstrap workspace excludes scientific evidence
    root = tmp_path / "evidence"
    source_program_dir = _make_source_program(root)
    bundle_dir = _make_bundle(root, source_program_dir)
    (bundle_dir / "evidence").mkdir()
    (bundle_dir / "evidence" / "metric_values.csv").write_text("value\n1\n", encoding="utf-8")
    (bundle_dir / "evidence_manifest.json").write_text("{}\n", encoding="utf-8")

    _reset_bootstrap_bundle(bundle_dir, root / "analysis_ready", source_program_dir)

    assert not (root / "analysis_ready" / "evidence").exists()
    assert not (root / "analysis_ready" / "evidence_manifest.json").exists()

    # an existing checkpoint is reused without a rerun
    root = tmp_path / "reuse"
    source_program_dir = _make_source_program(
        root,
        metrics=_WINNER_METRICS,
        results={"val_accuracy": 0.91, "checkpoint_path": "best_model.pt"},
        checkpoint=True,
    )
    output_bundle_dir = root / "analysis_ready"

    report = _bootstrap(_make_bundle(root, source_program_dir), output_bundle_dir)

    assert report["checkpoint"]["exists"] is True
    assert report["rerun"] is None
    assert (output_bundle_dir / "program_0" / "best_model.pt").exists()
    assert (output_bundle_dir / "program_0" / "metrics.json").exists()

    # the optional checkpoint mode uses artifacts without a rerun
    root = tmp_path / "optional"
    source_program_dir = _make_source_program(root, metrics={"combined_score": 0.91}, results={"combined_score": 0.91})

    report = _bootstrap(
        _make_bundle(root, source_program_dir),
        root / "analysis_ready",
        agent="pi",
        agent_model="deepseek",
        primary_metric_key="combined_score",
        checkpoint_mode="optional",
    )

    assert report["status"] == "completed"
    assert report["bootstrap_ready"] is True
    assert report["checkpoint_mode"] == "optional"
    assert report["checkpoint"]["exists"] is False
    assert report["bootstrap_agent"] is None
    assert report["rerun"] is None

    # the bootstrap agent defaults to the posthoc agent runner script
    root = tmp_path / "runner"
    bundle_dir = _make_bundle(root, _checkpointless_source(root))
    captured: dict[str, list[str]] = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout: int | None = None) -> tuple[str, str]:
            return ("", "")

    def fake_popen(cmd: list[str], stdout: object, stderr: object, text: bool, env: dict[str, str], start_new_session: bool) -> FakeProcess:
        captured["cmd"] = cmd
        return FakeProcess()

    report = _run_bootstrap_agent(
        bundle_dir=bundle_dir,
        agent="claude",
        agent_model=None,
        reasoning_effort="high",
        task_source_dir=None,
        containerize=False,
        container_image="",
        container_network="bridge",
        container_auth_root=None,
        container_env_file=None,
        container_session_name=None,
        container_mount_root=None,
        script_path=None,
        popen_factory=cast(Any, fake_popen),
        register_subprocess=lambda process: None,
        unregister_subprocess=lambda process: None,
        terminate_process_group=lambda process, grace_seconds=1: None,
        container_execution_env=lambda **kwargs: {},
        ensure_text=lambda text: text if isinstance(text, str) else "",
        **RUN_KWARGS,
    )

    assert report["exit_code"] == 0
    assert str(report["script_path"]).endswith("src/scripts/posthoc_agent.sh")
    assert str(captured["cmd"][1]).endswith("src/scripts/posthoc_agent.sh")
    assert captured["cmd"][-2:] == ["--reasoning-effort", "high"]
    assert stat.S_IMODE((bundle_dir / "bootstrap_prompt.md").stat().st_mode) == 0o600


def test_bootstrap_reruns_evaluator_and_degrades_on_guardrail_or_agent_failure(tmp_path: Path) -> None:
    # reruns the evaluator when the checkpoint is missing
    root = tmp_path / "rerun"
    output_bundle_dir = root / "analysis_ready"
    script = _write_agent_script(tmp_path / "fake_bootstrap_agent.sh")

    report = _bootstrap(
        _make_bundle(root, _checkpointless_source(root)),
        output_bundle_dir,
        primary_metric_tolerance=0.03,
        script_path=script,
    )

    assert report["rerun"] is not None
    assert report["guardrail"]["passed"] is True
    assert (output_bundle_dir / "program_0" / "best_checkpoint.pt").exists()

    # a guardrail failure degrades instead of raising
    root = tmp_path / "guardrail"
    bundle_dir = _make_bundle(root, _checkpointless_source(root), evaluator_score=0.70)

    report = _bootstrap(bundle_dir, root / "analysis_ready", script_path=script)

    assert report["status"] == "degraded"
    assert report["bootstrap_ready"] is False
    assert "guardrail" in report["failure_reason"]

    # an agent failure persists the agent stdio
    root = tmp_path / "failure"
    failing_script = _write_agent_script(
        tmp_path / "failing_bootstrap_agent.sh",
        "echo 'bootstrap stdout'\necho 'bootstrap stderr' >&2\nexit 125\n",
    )

    report = _bootstrap(
        _make_bundle(root, _checkpointless_source(root)),
        root / "analysis_ready",
        max_attempts=1,
        primary_metric_tolerance=0.01,
        script_path=failing_script,
    )

    agent_report = report["bootstrap_agent"]
    assert report["status"] == "degraded"
    assert agent_report is not None
    assert agent_report["exit_code"] == 125
    assert Path(agent_report["stdout_path"]).read_text(encoding="utf-8").strip() == "bootstrap stdout"
    assert Path(agent_report["stderr_path"]).read_text(encoding="utf-8").strip() == "bootstrap stderr"


# --------------------------------------------------------------------------- generate


_FAKE_GENERATE_SCRIPT = """#!/bin/bash
set -euo pipefail
WORK_DIR=""
PROMPT_FILE=""
POSTHOC_ENV_POLICY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --posthoc-env-policy) POSTHOC_ENV_POLICY=1; shift ;;
    *) shift ;;
  esac
done
test -n "$PROMPT_FILE"
test -f "$WORK_DIR/$PROMPT_FILE"
test "$POSTHOC_ENV_POLICY" -eq 1
test -f "$WORK_DIR/evidence/aggregated/task/metric_values.csv"
cat > "$WORK_DIR/posthoc_bundle.json" <<'JSON'
{
  "decision": "propose_analyses",
  "summary": "Generated analyses",
  "analyses": [
    {
      "id": "activation_probe",
      "title": "Activation Probe",
      "family": "mechanistic",
      "hypothesis": "probe features",
      "relevance_reason": "paper value",
      "feasibility_reason": "checkpoint exists",
      "required_artifacts": ["task_bundle/program_0/best_model.pt"],
      "requires_retrain": false,
      "implementation_plan": ["write probe", "run analysis"],
      "expected_claim": "hidden units align with classes",
      "expected_outputs": ["figures/probe.png", "data/probe.csv"]
    },
    {
      "id": "error_clusters",
      "title": "Error Clusters",
      "family": "failure",
      "hypothesis": "errors cluster",
      "relevance_reason": "diagnostic",
      "feasibility_reason": "analysis only",
      "required_artifacts": ["task_bundle/program_0/_results.json"],
      "requires_retrain": false,
      "implementation_plan": ["collect errors"],
      "expected_claim": "specific classes dominate errors",
      "expected_outputs": ["figures/errors.png"]
    }
  ]
}
JSON
"""


def test_generate_analysis_candidates_parses_bundle_and_prompt_lists_staged_evidence(tmp_path: Path) -> None:
    # the generation agent output is parsed into multiple candidates
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "task.md").write_text("# Goal\nAnalyze\n", encoding="utf-8")
    (bundle_dir / "evaluate.py").write_text("# evaluator\n", encoding="utf-8")
    (bundle_dir / "extraction_manifest.json").write_text("{}", encoding="utf-8")
    evidence_file = bundle_dir / "evidence" / "aggregated" / "task" / "metric_values.csv"
    evidence_file.parent.mkdir(parents=True)
    evidence_file.write_text("variant,value\nbest,1\n", encoding="utf-8")
    evidence_bytes = evidence_file.read_bytes()
    (bundle_dir / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "source_role": "completed_ablations_bundle",
                "files": {
                    "evidence/aggregated/task/metric_values.csv": {
                        "size": len(evidence_bytes),
                        "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
                    }
                },
                "total_bytes": len(evidence_bytes),
            }
        ),
        encoding="utf-8",
    )
    make_program_dir(bundle_dir / "program_0", content="print('winner')\n")
    (bundle_dir / "program_0" / "_results.json").write_text("{}", encoding="utf-8")
    script_path = tmp_path / "fake_generate.sh"
    script_path.write_text(_FAKE_GENERATE_SCRIPT, encoding="utf-8")
    script_path.chmod(0o755)
    generation_dir = tmp_path / "generation"

    candidates = generate_analysis_candidates(
        bundle_dir=bundle_dir,
        generation_dir=generation_dir,
        agent="claude",
        agent_model=None,
        script_path=script_path,
        **RUN_KWARGS,
    )

    assert [candidate.analysis_id for candidate in candidates] == ["activation_probe", "error_clusters"]
    assert (generation_dir / "change_summary.json").exists()
    assert stat.S_IMODE((generation_dir / "posthoc_generate_prompt.md").stat().st_mode) == 0o600
    assert (generation_dir / "evidence" / "aggregated" / "task" / "metric_values.csv").is_file()

    # the generation prompt lists the staged evidence
    bundle = tmp_path / "prompt_bundle"
    bundle.mkdir()
    (bundle / "evidence_manifest.json").write_text(
        '{"files":{"evidence/aggregated/task/metric_values.csv":{"size":12}}}\n', encoding="utf-8"
    )

    assert "evidence/aggregated/task/metric_values.csv" in _build_generation_prompt(bundle)


def _parse_single(analysis: dict[str, Any]) -> tuple[str, list[AnalysisCandidate]]:
    return _parse_posthoc_bundle({"decision": "propose_analyses", "analyses": [analysis]})


_ANALYSIS_ID_CASES: list[tuple[str, dict[str, Any], list[str]]] = [
    (
        "legacy_analysis_id_alias",
        {
            "analysis_id": "legacy_probe",
            "title": "Legacy Probe",
            "family": "mechanistic",
            "hypothesis": "probe features",
            "relevance_reason": "paper value",
            "feasibility_reason": "checkpoint exists",
            "required_artifacts": [],
            "requires_retrain": False,
            "implementation_plan": ["inspect checkpoint"],
            "expected_claim": "features are separable",
            "expected_outputs": ["probe.png"],
        },
        ["legacy_probe"],
    ),
    ("path_like_id_rejected", {"id": "../../outside", "title": "Unsafe", "family": "test"}, []),
]


def test_parse_posthoc_bundle_normalizes_plans_outputs_and_analysis_ids() -> None:
    # a numbered string plan is split into steps
    decision, candidates = _parse_single(
        {
            "id": "ablation_mean_centering",
            "title": "Ablation: Mean-Centered Error Signal",
            "family": "ablation",
            "hypothesis": "mean-centering matters",
            "relevance_reason": "paper value",
            "feasibility_reason": "checkpoint exists",
            "required_artifacts": ["src/program.py", "src/checkpoint.pt"],
            "requires_retrain": True,
            "implementation_plan": (
                "1. Copy program.py, remove mean-centering. 2. Retrain for 250 epochs. 3. Compare accuracy and plots."
            ),
            "expected_claim": "mean-centering is essential",
            "expected_outputs": ["accuracy_curve_comparison.png", "confusion_matrix_no_meancentering.png"],
        }
    )

    assert decision == "propose_analyses"
    assert len(candidates) == 1
    assert candidates[0].implementation_plan == [
        "Copy program.py, remove mean-centering.",
        "Retrain for 250 epochs.",
        "Compare accuracy and plots.",
    ]

    # prose expected outputs are normalized to paths
    _, candidates = _parse_single(
        {
            "id": "trace_forensics",
            "title": "Trace forensics",
            "family": "forensics",
            "required_artifacts": ["evidence/metrics.csv"],
            "expected_outputs": [
                "Within-run correlation table (median/IQR rho)",
                "Scatter plot of epsilon vs disagreement",
                "figures/already_safe.png",
            ],
        }
    )

    assert candidates[0].expected_outputs == [
        "outputs/01_within_run_correlation_table_median_iqr_rho.csv",
        "outputs/02_scatter_plot_of_epsilon_vs_disagreement.png",
        "figures/already_safe.png",
    ]

    # the legacy analysis_id alias is accepted and path-like ids are rejected
    for name, analysis, expected_ids in _ANALYSIS_ID_CASES:
        decision, candidates = _parse_single(analysis)

        assert decision == "propose_analyses", name
        assert [candidate.analysis_id for candidate in candidates] == expected_ids, name


# --------------------------------------------------------------------------- select


def _select_bundle(root: Path, *, bootstrap_ready: bool = True) -> Path:
    bundle_dir = root / "bundle"
    (bundle_dir / "program_0").mkdir(parents=True)
    (bundle_dir.parent / "bootstrap_report.json").write_text(json.dumps({"bootstrap_ready": bootstrap_ready}), encoding="utf-8")
    return bundle_dir


def _apply(candidates: list[AnalysisCandidate], bundle_dir: Path, max_selected: int = 2) -> AnalysisSelection:
    return _apply_selection_constraints(
        candidates=candidates,
        llm_decisions={candidate.analysis_id: (True, "ok") for candidate in candidates},
        max_selected_analyses=max_selected,
        bundle_dir=bundle_dir,
    )


def _escaped_symlink_artifact(root: Path, bundle_dir: Path) -> str:
    external = root / "external.csv"
    external.write_text("secret\n", encoding="utf-8")
    (bundle_dir / "escaped.csv").symlink_to(external)
    return "escaped.csv"


def _external_artifact(root: Path, bundle_dir: Path) -> str:
    external = root / "external.csv"
    external.write_text("secret\n", encoding="utf-8")
    return str(external)


_UNSAFE_CANDIDATE_CASES: list[tuple[str, dict[str, Any], bool, Callable[[Path, Path], str] | None, int, str, bool]] = [
    (
        "retrain_candidate_when_bootstrap_not_ready",
        dict(analysis_id="needs_retrain", title="Needs Retrain", requires_retrain=True),
        False,
        None,
        2,
        "post-hoc analysis pipeline is analysis-only and bootstrap is not relevant for retraining candidates",
        True,
    ),
    (
        "ablation_family",
        dict(analysis_id="ablation_mean_centering", family="ablation", title="Ablation: Mean-Centered Error"),
        True,
        None,
        2,
        "post-hoc analysis pipeline excludes ablations; use the dedicated ablation pipeline",
        True,
    ),
    (
        "retrain_like_plan_text",
        dict(analysis_id="offline_probe", title="Offline Probe", implementation_plan=["Train from scratch and compare checkpoints"]),
        True,
        None,
        2,
        "post-hoc analysis pipeline is analysis-only; use existing artifacts rather than retraining or evaluator reruns",
        True,
    ),
    (
        "required_artifact_outside_bundle",
        dict(analysis_id="external_artifact", family="forensics", title="External Artifact"),
        True,
        _external_artifact,
        1,
        "missing required artifacts",
        False,
    ),
    (
        "required_artifact_symlink_escape",
        dict(analysis_id="symlink_artifact", family="forensics", title="Symlink Artifact"),
        True,
        _escaped_symlink_artifact,
        1,
        "missing required artifacts",
        False,
    ),
]

_SELECTION_DECISION_CASES: list[tuple[str, object, dict[str, tuple[bool, str]]]] = [
    (
        "top_level_list",
        [{"id": "a1", "accepted": True, "reason": "good"}, {"id": "a2", "accepted": False, "reason": "bad"}],
        {"a1": (True, "good"), "a2": (False, "bad")},
    ),
    ("documented_object", {"decisions": [{"id": "a1", "accepted": True, "reason": "good"}]}, {"a1": (True, "good")}),
    (
        "legacy_analysis_id_alias",
        {"decisions": [{"analysis_id": "a1", "accepted": True, "reason": "good"}]},
        {"a1": (True, "good")},
    ),
]


def test_selection_constraints_dedupe_cap_resolve_artifacts_reject_unsafe_and_parse_decisions(tmp_path: Path) -> None:
    # duplicates and family repeats are rejected and the cap is respected
    candidates = [
        _candidate("a1", "mechanistic", title="Probe"),
        _candidate("a1_dup", "mechanistic", title="Probe"),
        _candidate("a2", "failure"),
        _candidate("a3", "sensitivity"),
    ]

    selection = _apply(candidates, _select_bundle(tmp_path / "dedupe"))

    assert [candidate.analysis_id for candidate in selection.accepted] == ["a1", "a2"]
    assert len(selection.rejected) == 2
    assert any("duplicate" in reason or "family already represented" in reason for _, reason in selection.rejected)
    assert any("cap" in reason for _, reason in selection.rejected)

    # program-root artifacts resolve under program_0
    bundle_dir = _select_bundle(tmp_path / "program_root")
    (bundle_dir / "program_0" / "src").mkdir()
    (bundle_dir / "program_0" / "src" / "program.py").write_text("print('winner')\n", encoding="utf-8")
    (bundle_dir / "program_0" / "src" / "checkpoint.pt").write_text("checkpoint\n", encoding="utf-8")
    candidate = _candidate(
        "representation_filters",
        "representation",
        title="Representation Filters",
        required_artifacts=["src/program.py", "src/checkpoint.pt"],
    )

    selection = _apply([candidate], bundle_dir)

    assert [item.analysis_id for item in selection.accepted] == ["representation_filters"]
    assert selection.rejected == []

    # unsafe candidates are rejected with the documented reason
    for name, candidate_kwargs, bootstrap_ready, artifact, max_selected, reason, exact in _UNSAFE_CANDIDATE_CASES:
        root = tmp_path / name
        bundle_dir = _select_bundle(root, bootstrap_ready=bootstrap_ready)
        if artifact is not None:
            candidate_kwargs = {**candidate_kwargs, "required_artifacts": [artifact(root, bundle_dir)]}

        selection = _apply([_candidate(**candidate_kwargs)], bundle_dir, max_selected)

        assert selection.accepted == [], name
        rejected_reason = selection.rejected[0][1]
        assert (rejected_reason == reason if exact else reason in rejected_reason), name

    # every supported selection decision schema is parsed
    for name, raw, expected in _SELECTION_DECISION_CASES:
        assert _parse_selection_decisions(raw) == expected, name


def test_selection_stages_private_prompt(tmp_path: Path) -> None:
    bundle_dir = _select_bundle(tmp_path)
    generation_dir = tmp_path / "generation"
    script = tmp_path / "select.sh"
    script.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "posthoc_env_policy=0\n"
        "while [[ $# -gt 0 ]]; do\n"
        '  case "$1" in\n'
        '    --work-dir) work_dir="$2"; shift 2 ;;\n'
        '    --prompt-file) prompt_file="$2"; shift 2 ;;\n'
        "    --posthoc-env-policy) posthoc_env_policy=1; shift ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        'test "$posthoc_env_policy" -eq 1\n'
        'touch "$work_dir/posthoc_env_policy_seen"\n'
        'test -f "$work_dir/$prompt_file"\n'
        "printf '{\"decisions\":[]}' > \"$work_dir/posthoc_selection.json\"\n",
        encoding="utf-8",
    )
    script.chmod(0o755)

    validate_and_select_analyses(
        candidates=[_candidate("probe")],
        bundle_dir=bundle_dir,
        generation_dir=generation_dir,
        agent="codex",
        agent_model=None,
        max_selected_analyses=1,
        script_path=script,
        **RUN_KWARGS,
    )

    assert stat.S_IMODE((generation_dir / "posthoc_select_prompt.md").stat().st_mode) == 0o600
    assert (generation_dir / "posthoc_env_policy_seen").is_file()


# --------------------------------------------------------------------------- materialize


@pytest.fixture
def analysis_ready_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "analysis_ready"
    bundle_dir.mkdir(parents=True)
    make_program_dir(bundle_dir / "program_0", content="print('winner')\n")
    make_program_dir(bundle_dir / "reference_program", content="print('seed')\n")
    (bundle_dir / "task.md").write_text("# Task\n", encoding="utf-8")
    (bundle_dir / "evaluate.py").write_text("# evaluator\n", encoding="utf-8")
    (bundle_dir / "capture.py").write_text("# capture\n", encoding="utf-8")
    (bundle_dir / "helpers").mkdir()
    (bundle_dir / "helpers" / "util.py").write_text("VALUE = 7\n", encoding="utf-8")
    (bundle_dir / "extraction_manifest.json").write_text("{}", encoding="utf-8")
    return bundle_dir


def _materialize(bundle_dir: Path, output_dir: Path, candidate: AnalysisCandidate, **overrides: Any) -> list[AnalysisSpec]:
    kwargs: dict[str, Any] = dict(agent="claude", agent_model=None, max_attempts_per_analysis=2)
    kwargs.update(overrides)
    return materialize_analysis_workspaces(
        analysis_ready_bundle_dir=bundle_dir,
        posthoc_run_dir=output_dir,
        selection=AnalysisSelection(accepted=[candidate], rejected=[]),
        **kwargs,
    )


def test_materialize_copies_bundle_rewrites_task_removes_stale_dirs_and_strips_symlinks(
    tmp_path: Path, analysis_ready_bundle: Path
) -> None:
    # copies the bundle and rewrites the task for the analysis
    output_dir = tmp_path / "output"
    candidate = _candidate(
        "activation_probe",
        title="Activation Probe",
        hypothesis="probe",
        required_artifacts=["task_bundle/program_0/_results.json"],
        implementation_plan=["inspect activations"],
        expected_claim="units align",
        expected_outputs=["figures/probe.png", "data/probe.csv"],
    )

    specs = _materialize(analysis_ready_bundle, output_dir, candidate, reasoning_effort="high", max_attempts_per_analysis=3)

    assert [spec.analysis_id for spec in specs] == ["activation_probe"]
    analysis_dir = output_dir / "analyses" / "activation_probe"
    for name in ("analysis_spec.json", "rubric_contract.json", "target_rubrics.json", "run_manifest.json"):
        assert (analysis_dir / name).exists()
    task_bundle = analysis_dir / "task_bundle"
    assert (task_bundle / "program_0").is_dir()
    assert (task_bundle / "reference_program").is_dir()
    assert (task_bundle / "original_task.md").exists()
    assert not (task_bundle / "evaluate.py").exists()
    assert not (task_bundle / "extraction_manifest.json").exists()
    assert "Post-Hoc Analysis" in (task_bundle / "task.md").read_text(encoding="utf-8")
    manifest = json.loads((analysis_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["max_attempts_per_analysis"] == 3
    assert manifest["reasoning_effort"] == "high"

    # stale analysis dirs are removed
    output_dir = tmp_path / "output_stale"
    stale_dir = output_dir / "analyses" / "stale_analysis"
    stale_dir.mkdir(parents=True)
    (stale_dir / "analysis_spec.json").write_text("{}", encoding="utf-8")

    specs = _materialize(analysis_ready_bundle, output_dir, _candidate("selected_analysis", title="Selected Analysis"))

    assert [spec.analysis_id for spec in specs] == ["selected_analysis"]
    assert not stale_dir.exists()

    # symlinks are removed from the task bundle without touching their targets
    external = tmp_path / "external.txt"
    external.write_text("original", encoding="utf-8")
    (analysis_ready_bundle / "external_link").symlink_to(external)
    candidate = _candidate("safe_analysis", "test", title="Safe")

    specs = _materialize(analysis_ready_bundle, tmp_path / "output_symlink", candidate, agent="pi", max_attempts_per_analysis=1)

    assert not (specs[0].analysis_dir / "task_bundle" / "external_link").exists()
    assert external.read_text(encoding="utf-8") == "original"


def test_materialize_rejects_escaping_analysis_ids_and_skips_special_files(
    tmp_path: Path, analysis_ready_bundle: Path
) -> None:
    # an analysis id outside the analyses root is rejected before any write
    marker = tmp_path / "outside" / "keep.txt"
    marker.parent.mkdir()
    marker.write_text("keep", encoding="utf-8")
    candidate = _candidate("../../outside", "test", title="Unsafe")

    with pytest.raises(ValueError, match="analysis_id"):
        _materialize(analysis_ready_bundle, tmp_path / "output", candidate, agent="pi", max_attempts_per_analysis=1)

    assert marker.read_text(encoding="utf-8") == "keep"

    # special files in the analysis-ready bundle are skipped
    fifo_path = analysis_ready_bundle / ".posthoc_stream_test.fifo"
    os.mkfifo(fifo_path)
    output_dir = tmp_path / "output_fifo"

    specs = _materialize(analysis_ready_bundle, output_dir, _candidate("representation_probe", "representation"))

    assert [spec.analysis_id for spec in specs] == ["representation_probe"]
    assert not (output_dir / "analyses" / "representation_probe" / "task_bundle" / fifo_path.name).exists()


# --------------------------------------------------------------------------- publish


def _make_attempt(analysis_dir: Path, *, figures: list[str], data_files: list[str] | None = None) -> Path:
    data_files = data_files or []
    attempt_dir = analysis_dir / "attempt_1"
    attempt_dir.mkdir(parents=True)
    for relpath in [*figures, *data_files]:
        path = attempt_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")
    (attempt_dir / "analysis_summary.md").write_text("summary", encoding="utf-8")
    (attempt_dir / "interestingness_rubric.json").write_text(
        json.dumps({"grounding_signal": "high", "paper_signal": "high"}), encoding="utf-8"
    )
    (attempt_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "analysis_id": analysis_dir.name,
                "attempt_id": "attempt_1",
                "summary": "summary",
                "claim": "claim",
                "grounded": True,
                "reran_evaluator": False,
                "primary_metric_tolerance": 0.0,
                "outputs": {"figures": figures, "data_files": data_files},
                "evidence_paths": [*figures, *data_files],
                "checkpoint_paths": ["task_bundle/program_0/best_model.pt"],
            }
        ),
        encoding="utf-8",
    )
    return attempt_dir


def _inspection(attempt_dir: Path, *, checkpoint_validated: bool = True) -> AttemptInspection:
    return AttemptInspection(
        attempt_dir=attempt_dir,
        attempt_id="attempt_1",
        overall_pass=True,
        rubric_error=None,
        required_outputs_present=True,
        missing_outputs=[],
        manifest_valid=True,
        manifest_error=None,
        guardrail_passed=None,
        checkpoint_validated=checkpoint_validated,
    )


def _completed_result(analysis_id: str, attempts: list[AttemptInspection]) -> AnalysisRunResult:
    return AnalysisRunResult(
        analysis_id=analysis_id,
        status="completed",
        attempt_count=1,
        exit_code=0,
        selected_attempt="attempt_1",
        rubric_passed=True,
        required_outputs_present=True,
        guardrail_passed=None,
        output="",
        produced_files={},
        attempts=attempts,
    )


def test_publish_delegates_valid_analyses_withholds_manifest_on_failure_and_revalidates_checkpoints(
    tmp_path: Path,
) -> None:
    # all valid analyses are published for the paper agent
    output_dir = tmp_path / "delegated" / "posthoc_run"
    analysis_a_dir = output_dir / "analyses" / "analysis_a"
    analysis_b_dir = output_dir / "analyses" / "analysis_b"
    attempt_a = _make_attempt(analysis_a_dir, figures=["figures/a.png"])
    attempt_b = _make_attempt(analysis_b_dir, figures=["figures/b1.png", "figures/b2.png"], data_files=["data/b.csv"])
    spec_a = AnalysisSpec("analysis_a", "A", "mechanistic", analysis_a_dir, expected_claim="claim")
    spec_b = AnalysisSpec("analysis_b", "B", "failure", analysis_b_dir, expected_claim="claim")

    ranking = aggregate_and_publish_valid_analyses(
        output_dir=output_dir,
        analysis_specs=[spec_a, spec_b],
        results=[
            _completed_result("analysis_a", [_inspection(attempt_a)]),
            _completed_result("analysis_b", [_inspection(attempt_b)]),
        ],
    )

    assert ranking["selected_analysis_id"] is None
    assert ranking["delegated_to_paper_agent"] is True
    assert [entry["analysis_id"] for entry in ranking["valid"]] == ["analysis_a", "analysis_b"]
    assert (output_dir / "latest_manifest.json").exists()
    assert (output_dir.parent / "latest_manifest.json").exists()
    latest_manifest = json.loads((output_dir / "latest_manifest.json").read_text(encoding="utf-8"))
    assert latest_manifest["selected_analysis_id"] is None
    assert latest_manifest["delegated_to_paper_agent"] is True
    assert [entry["analysis_id"] for entry in latest_manifest["valid_analyses"]] == ["analysis_a", "analysis_b"]

    # the latest manifest is only published when a success exists
    output_dir = tmp_path / "failed" / "posthoc_run"
    analysis_dir = output_dir / "analyses" / "analysis_a"
    analysis_dir.mkdir(parents=True)
    result = AnalysisRunResult(
        analysis_id="analysis_a",
        status="failed",
        attempt_count=1,
        exit_code=1,
        selected_attempt=None,
        rubric_passed=False,
        required_outputs_present=False,
        guardrail_passed=False,
        output="",
        produced_files={},
        attempts=[],
    )

    aggregate_and_publish_valid_analyses(
        output_dir=output_dir,
        analysis_specs=[AnalysisSpec("analysis_a", "A", "mechanistic", analysis_dir)],
        results=[result],
    )

    assert not (output_dir / "latest_manifest.json").exists()
    summary = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "no_valid_analyses"
    assert summary["selected_analysis_id"] is None

    # the checkpoint policy is revalidated for resumed results
    output_dir = tmp_path / "resumed" / "posthoc_run"
    analysis_dir = output_dir / "analyses" / "artifact_only"
    attempt_dir = _make_attempt(analysis_dir, figures=["figures/a.png"])
    spec = AnalysisSpec("artifact_only", "A", "metrics", analysis_dir)
    result = _completed_result("artifact_only", [_inspection(attempt_dir, checkpoint_validated=False)])

    optional = aggregate_and_publish_valid_analyses(
        output_dir=output_dir, analysis_specs=[spec], results=[result], checkpoint_mode="optional"
    )
    assert (output_dir / "latest_manifest.json").exists()
    assert (output_dir.parent / "latest_manifest.json").exists()

    required = aggregate_and_publish_valid_analyses(
        output_dir=output_dir, analysis_specs=[spec], results=[result], checkpoint_mode="required"
    )

    assert required["valid"] == []
    assert required["rejected"][0]["reason"] == "checkpoint_required"
    assert [entry["analysis_id"] for entry in optional["valid"]] == ["artifact_only"]
    assert not (output_dir / "latest_manifest.json").exists()
    assert not (output_dir.parent / "latest_manifest.json").exists()


# --------------------------------------------------------------------------- tree digest


def _two_files(root: Path) -> None:
    (root / "first").write_bytes(b"123")
    (root / "second").write_bytes(b"456")


_TREE_LIMIT_CASES: list[tuple[str, dict[str, int], Callable[[Path], object], str]] = [
    ("entry_limit", {"MAX_TREE_ENTRIES": 1}, _two_files, "entry limit"),
    ("depth_limit", {"MAX_TREE_DEPTH": 1}, lambda root: (root / "first" / "second").mkdir(parents=True), "depth limit"),
    ("aggregate_byte_limit", {"MAX_TREE_FILE_BYTES": 1024, "MAX_TREE_TOTAL_BYTES": 5}, _two_files, "aggregate byte limit"),
]


def test_tree_digest_enforces_limits_ignores_symlink_targets_and_frames_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # entry, depth and aggregate byte limits are enforced
    for name, limits, populate, match in _TREE_LIMIT_CASES:
        root = tmp_path / name
        root.mkdir()
        populate(root)

        with monkeypatch.context() as patch_limits:
            for limit_name, value in limits.items():
                patch_limits.setattr(tree_digest_module, limit_name, value)
            with pytest.raises(TreeDigestError, match=match):
                tree_digest(root)

    # file symlinks are not followed
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("first", encoding="utf-8")
    (root / "linked").symlink_to(outside)
    initial_digest = tree_digest(root)

    outside.write_text("second", encoding="utf-8")

    assert tree_digest(root) == initial_digest

    # distinct tree records are framed so concatenation cannot collide
    split_tree = tmp_path / "split"
    split_tree.mkdir()
    (split_tree / "a").write_bytes(b"b")
    (split_tree / "c").write_bytes(b"")
    combined_tree = tmp_path / "combined"
    combined_tree.mkdir()
    (combined_tree / "a").write_bytes(b"bcF")

    assert tree_digest(split_tree) != tree_digest(combined_tree)

    # the digest is independent of short read boundaries
    payload_root = tmp_path / "payload_root"
    payload_root.mkdir()
    (payload_root / "payload").write_bytes(b"abcdef")
    expected_digest = tree_digest(payload_root)
    original_read = tree_digest_module.os.read

    def short_read(file_descriptor: int, size: int) -> bytes:
        return original_read(file_descriptor, min(size, 2))

    with monkeypatch.context() as patch_read:
        patch_read.setattr(tree_digest_module.os, "read", short_read)

        assert tree_digest(payload_root) == expected_digest
