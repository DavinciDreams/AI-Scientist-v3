"""Post-hoc pipeline config and basic orchestration helpers."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

from src.investigation.config import AgentConfig
from src.posthoc_pipeline.cli import load_posthoc_config
from src.posthoc_pipeline.config import (
    PosthocExecutionConfig,
    PosthocPathsConfig,
    PosthocPipelineConfig,
    validate_posthoc_config,
)
from src.posthoc_pipeline.models import AnalysisCandidate
from src.posthoc_pipeline.pipeline import (
    _load_analysis_specs_from_disk,
    _load_results_from_disk,
    _load_selection_from_disk,
    _resolve_output_dir,
    run_posthoc_pipeline,
)
from tests.program_testutils import make_program_dir

_TIMEOUTS: dict[str, Any] = dict(
    bootstrap_timeout_seconds=10,
    generation_timeout_seconds=10,
    selection_timeout_seconds=10,
    analysis_timeout_seconds=10,
    analysis_timeout_grace_seconds=1,
)


def _get(cfg: Any, key: str) -> object:
    node = cfg
    for part in key.split("."):
        node = getattr(node, part)
    return node


def _make_task_dir(root: Path) -> Path:
    task_dir = root / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.md").write_text("# Goal\nAnalyze\n", encoding="utf-8")
    (task_dir / "evaluate.py").write_text("print('eval')\n", encoding="utf-8")
    make_program_dir(task_dir / "program_0", content="print('seed')\n")
    return task_dir


def _make_results_dir(root: Path, *, metrics: dict, results: dict, checkpoint: bool) -> Path:
    """A results dir whose manifest points at a managed copy of a winner program."""
    winner_dir = root / "winner"
    make_program_dir(winner_dir, content="print('winner')\n")
    (winner_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (winner_dir / "correct.json").write_text(json.dumps({"correct": True}), encoding="utf-8")
    (winner_dir / "_results.json").write_text(json.dumps(results), encoding="utf-8")
    if checkpoint:
        (winner_dir / "best_model.pt").write_bytes(pickle.dumps({"weights": [1.0]}))
    results_dir = root / "results"
    results_dir.mkdir(parents=True)
    managed_winner_dir = results_dir / "winner"
    shutil.copytree(winner_dir, managed_winner_dir, symlinks=True)
    (results_dir / "best_program_manifest.json").write_text(
        json.dumps(
            {
                "program_dir": str(managed_winner_dir),
                "entrypoint_path": "src/program.py",
                "combined_score": 0.88,
                "correct": True,
                "source_attempt_id": "idea_1_attempt_1",
            }
        ),
        encoding="utf-8",
    )
    return results_dir


def _posthoc_cfg(
    agent: str, task_dir: Path, results_dir: Path, *, run_id: str, output_dir: Path | None = None, **execution: Any
) -> PosthocPipelineConfig:
    return PosthocPipelineConfig(
        agent=AgentConfig(name=agent),
        paths=PosthocPathsConfig(
            task_dir=str(task_dir),
            results_dir=str(results_dir),
            output_dir=None if output_dir is None else str(output_dir),
        ),
        execution=PosthocExecutionConfig(**_TIMEOUTS, **execution),
        run_id=run_id,
    )


def _make_candidate(analysis_id: str) -> AnalysisCandidate:
    return AnalysisCandidate(
        analysis_id=analysis_id,
        title=analysis_id,
        family="mechanistic",
        hypothesis="hyp",
        relevance_reason="rel",
        feasibility_reason="feas",
        required_artifacts=[],
        requires_retrain=False,
        implementation_plan=[],
        expected_claim="claim",
        expected_outputs=[],
    )


# --- config -------------------------------------------------------------------------


_OVERRIDE_CASES = [
    (
        "run-id",
        ["paths.task_dir={tmp}/task", "paths.results_dir={tmp}/results", "run_id=smoketest"],
        "run_id",
        "smoketest",
    ),
    ("evidence-dir", ["paths.evidence_dir={tmp}/ablations"], "paths.evidence_dir", "{tmp}/ablations"),
    ("reasoning-effort", ["agent.reasoning_effort=medium"], "agent.reasoning_effort", "medium"),
    ("checkpoint-mode", ["execution.checkpoint_mode=optional"], "execution.checkpoint_mode", "optional"),
]

_INVALID_CONFIG_CASES = [
    (
        "unknown-checkpoint-mode",
        {"execution.checkpoint_mode": "sometimes"},
        "execution.checkpoint_mode must be one of ('required', 'optional')",
        True,
    ),
    (
        "gemini-reasoning-effort",
        {"agent.name": "gemini", "agent.reasoning_effort": "medium"},
        "not supported for gemini",
        False,
    ),
    (
        "profile-without-model",
        {"agent.name": "codex", "agent.profile": "custom"},
        "agent.model is required when agent.profile is provided",
        True,
    ),
    (
        "profile-with-blank-model",
        {"agent.name": "codex", "agent.model": "   ", "agent.profile": "custom"},
        "agent.model is required when agent.profile is provided",
        True,
    ),
]


def test_posthoc_config_resolves_output_dir_accepts_overrides_and_rejects_invalid_values(tmp_path: Path) -> None:
    # the output dir is the run-scoped posthoc path under the results dir
    cfg = PosthocPipelineConfig(
        agent=AgentConfig(name="codex"),
        paths=PosthocPathsConfig(task_dir=str(tmp_path / "task"), results_dir=str(tmp_path / "results")),
        run_id="20260102_030405",
    )

    assert _resolve_output_dir(cfg) == (tmp_path / "results" / "posthoc" / "task_codex_20260102_030405").resolve()

    # load_posthoc_config accepts hydra-style overrides
    (tmp_path / "ablations").mkdir()
    for name, overrides, key, expected in _OVERRIDE_CASES:
        cfg = load_posthoc_config([override.format(tmp=tmp_path) for override in overrides])

        assert _get(cfg, key) == expected.format(tmp=tmp_path), name

    # validate_posthoc_config rejects invalid mutations
    for name, mutations, message, exact in _INVALID_CONFIG_CASES:
        cfg = load_posthoc_config([])
        for key, value in mutations.items():
            head, _, tail = key.partition(".")
            setattr(getattr(cfg, head), tail, value)

        errors = validate_posthoc_config(cfg)

        assert (message in errors if exact else any(message in error for error in errors)), name


# --- pipeline stages ------------------------------------------------------------------


def test_pipeline_runs_extract_and_bootstrap_stages_and_reports_unusable_bootstrap(tmp_path: Path) -> None:
    # extract and bootstrap stages can run on their own
    root = tmp_path / "required"
    task_dir = _make_task_dir(root)
    results_dir = _make_results_dir(
        root,
        metrics={"combined_score": 0.91, "val_accuracy": 0.91},
        results={"val_accuracy": 0.91, "checkpoint_path": "best_model.pt"},
        checkpoint=True,
    )
    cfg = _posthoc_cfg("claude", task_dir, results_dir, run_id="20260102_030405")

    rc = run_posthoc_pipeline(cfg, stages={0, 1})

    output_dir = _resolve_output_dir(cfg)
    assert rc == 0
    assert (output_dir / "_extracted_bundle" / "program_0" / "src" / "program.py").exists()
    assert (output_dir / "_analysis_ready_bundle" / "program_0" / "best_model.pt").exists()
    assert (output_dir / "bootstrap_report.json").exists()

    # the optional checkpoint mode propagates into the bootstrap report
    root = tmp_path / "optional"
    task_dir = _make_task_dir(root)
    results_dir = _make_results_dir(
        root, metrics={"combined_score": 0.91}, results={"combined_score": 0.91}, checkpoint=False
    )
    cfg = _posthoc_cfg("pi", task_dir, results_dir, run_id="20260102_030406", checkpoint_mode="optional")

    assert run_posthoc_pipeline(cfg, stages={0, 1}) == 0
    report = json.loads((_resolve_output_dir(cfg) / "bootstrap_report.json").read_text(encoding="utf-8"))
    assert report["checkpoint_mode"] == "optional"
    assert report["bootstrap_ready"] is True
    assert report["bootstrap_agent"] is None
    assert report["rerun"] is None

    # a nonzero return code when the bootstrap is unusable and no valid analyses exist
    root = tmp_path / "unusable"
    task_dir = _make_task_dir(root)
    results_dir = root / "results"
    results_dir.mkdir(parents=True)
    output_dir = root / "posthoc_run"
    (output_dir / "_extracted_bundle").mkdir(parents=True)
    (output_dir / "_analysis_ready_bundle").mkdir(parents=True)
    (output_dir / "bootstrap_report.json").write_text(
        json.dumps(
            {"stage": "bootstrap", "bootstrap_ready": False, "failure_reason": "Bootstrap agent exited with code 125"}
        ),
        encoding="utf-8",
    )
    cfg = _posthoc_cfg("claude", task_dir, results_dir, run_id="20260102_030405", output_dir=output_dir)

    rc = run_posthoc_pipeline(cfg, stages={6})

    assert rc == 1
    summary = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "no_valid_analyses"


# --- disk loaders ---------------------------------------------------------------------


def _spec_payload(analysis_id: str) -> dict[str, object]:
    return {
        "analysis_id": analysis_id,
        "title": analysis_id,
        "family": "mechanistic",
        "implementation_plan": ["inspect"],
        "required_artifacts": [],
        "expected_outputs": [],
    }


def _result_payload(analysis_id: str) -> dict[str, object]:
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "attempt_count": 1,
        "exit_code": 0,
        "selected_attempt": "attempt_1",
        "rubric_passed": True,
        "required_outputs_present": True,
        "guardrail_passed": None,
        "output": "",
        "produced_files": {},
        "attempts": [],
    }


def test_disk_loaders_repair_plans_filter_selected_ids_and_accept_id_schema(tmp_path: Path) -> None:
    # analysis specs with a character-level plan are repaired on load
    output_dir = tmp_path / "repair"
    analysis_dir = output_dir / "analyses" / "ablation_mean_centering"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "analysis_spec.json").write_text(
        json.dumps(
            {
                "analysis_id": "ablation_mean_centering",
                "title": "Ablation: Mean-Centered Error Signal",
                "family": "ablation",
                "implementation_plan": list("1. Copy program.py, remove mean-centering. 2. Retrain for 250 epochs."),
                "required_artifacts": ["src/program.py", "src/checkpoint.pt"],
                "expected_outputs": ["accuracy_curve_comparison.png"],
            }
        ),
        encoding="utf-8",
    )

    specs = _load_analysis_specs_from_disk(output_dir)

    assert len(specs) == 1
    assert specs[0].implementation_plan == ["Copy program.py, remove mean-centering.", "Retrain for 250 epochs."]

    # spec and result loaders filter to the selected analysis ids
    for filename, payload, loader in (
        ("analysis_spec.json", _spec_payload, _load_analysis_specs_from_disk),
        ("analysis_result.json", _result_payload, _load_results_from_disk),
    ):
        output_dir = tmp_path / filename
        for analysis_id in ("keep_me", "stale_analysis"):
            analysis_dir = output_dir / "analyses" / analysis_id
            analysis_dir.mkdir(parents=True)
            (analysis_dir / filename).write_text(json.dumps(payload(analysis_id)), encoding="utf-8")

        loaded = loader(output_dir, accepted_analysis_ids=["keep_me"])

        assert [item.analysis_id for item in loaded] == ["keep_me"], filename

    # the selection loader accepts the id schema
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir(parents=True)
    (generation_dir / "posthoc_selection.json").write_text(
        json.dumps(
            {"accepted": [{"id": "keep_me", "reason": "good"}], "rejected": [{"id": "drop_me", "reason": "bad"}]}
        ),
        encoding="utf-8",
    )
    candidates = [_make_candidate("keep_me"), _make_candidate("drop_me")]

    selection = _load_selection_from_disk(generation_dir, candidates)

    assert [candidate.analysis_id for candidate in selection.accepted] == ["keep_me"]
    assert [(c.analysis_id, reason) for c, reason in selection.rejected] == [("drop_me", "bad")]
