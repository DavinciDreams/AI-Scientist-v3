"""Post-hoc analysis execution: agent runs, attempt validation, rubric contract, runner env policy, logging."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import pickle
import stat
import subprocess
import sys
from typing import Any

import pytest

from src.posthoc_pipeline import cli, execute, pipeline, progress, tree_digest
from src.posthoc_pipeline.common import validate_expected_outputs
from src.posthoc_pipeline.config import (
    PosthocPathsConfig,
    PosthocPipelineConfig,
    validate_posthoc_config,
)
from src.posthoc_pipeline.execute import execute_analysis_agents
from src.posthoc_pipeline.models import AnalysisSpec
from src.posthoc_pipeline.rubric_contract import (
    REQUIRED_RUBRIC_IDS,
    load_rubric_status,
    normalize_posthoc_rubric_status,
    validate_posthoc_rubric_status,
)
from tests.program_testutils import make_program_dir

POSTHOC_AGENT_SCRIPT = "src/scripts/posthoc_agent.sh"


# --------------------------------------------------------------------------- helpers


def _make_analysis_workspace(root: Path, analysis_id: str, *, requires_retrain: bool = False) -> AnalysisSpec:
    analysis_dir = root / "analyses" / analysis_id
    program = analysis_dir / "task_bundle" / "program_0"
    make_program_dir(program, content="print('winner')\n")
    (program / "best_model.pt").write_bytes(pickle.dumps({"weights": [1.0]}))
    (program / "metrics.json").write_text(json.dumps({"combined_score": 0.91, "val_accuracy": 0.91}), encoding="utf-8")
    (program / "correct.json").write_text(json.dumps({"correct": True}), encoding="utf-8")
    (program / "_results.json").write_text(
        json.dumps({"val_accuracy": 0.91, "checkpoint_path": "best_model.pt"}), encoding="utf-8"
    )
    expected_outputs = ["figures/probe.png", "data/probe.csv"]
    (analysis_dir / "analysis_spec.json").write_text(
        json.dumps(
            {
                "analysis_id": analysis_id,
                "title": "Activation Probe",
                "family": "mechanistic",
                "analysis_dir": str(analysis_dir),
                "expected_outputs": expected_outputs,
            }
        ),
        encoding="utf-8",
    )
    (analysis_dir / "rubric_contract.json").write_text('{"rubrics":[]}', encoding="utf-8")
    (analysis_dir / "target_rubrics.json").write_text("[]", encoding="utf-8")
    (analysis_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "started_at": "old-start",
                "completed_at": "old-complete",
                "agent": "old-agent",
                "agent_model": "old-model",
                "agent_profile": "old-profile",
                "reasoning_effort": "low",
            }
        ),
        encoding="utf-8",
    )
    return AnalysisSpec(
        analysis_id=analysis_id,
        title="Activation Probe",
        family="mechanistic",
        analysis_dir=analysis_dir,
        expected_outputs=expected_outputs,
        requires_retrain=requires_retrain,
    )


_SCRIPT_HEAD = """#!/bin/bash
set -euo pipefail
WORK_DIR=""
ATTEMPT_ID=""
PROMPT_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --attempt-id) ATTEMPT_ID="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    *) shift ;;
  esac
done
test -n "$PROMPT_FILE"
test -f "$WORK_DIR/$PROMPT_FILE"
mkdir -p "$WORK_DIR/figures" "$WORK_DIR/data"
printf 'summary' > "$WORK_DIR/analysis_summary.md"
cat > "$WORK_DIR/interestingness_rubric.json" <<'JSON'
{"grounding_signal":"high","paper_signal":"high"}
JSON
cat > "$WORK_DIR/analysis_manifest.json" <<JSON
{
  "analysis_id": "${ANALYSIS_ID}",
  "attempt_id": "${ATTEMPT_ID}",
  "summary": "summary",
  "claim": "claim",
  "grounded": true,
  "reran_evaluator": false,
  "primary_metric_tolerance": 0.0,
  "outputs": {"figures": ["figures/probe.png"], "data_files": ["data/probe.csv"]},
  "evidence_paths": ["figures/probe.png", "data/probe.csv"],
  "checkpoint_paths": ["task_bundle/program_0/best_model.pt"]
}
JSON
"""

_SCRIPT_TAIL = """printf 'plot' > "$WORK_DIR/figures/probe.png"
printf 'csv' > "$WORK_DIR/data/probe.csv"
"""


def _list_rubric_json(overall_pass: bool = True, *, failing: str | None = None) -> str:
    rubrics = ",\n".join(
        f'    {{"id":"{rubric_id}","pass":{str(rubric_id != failing).lower()}}}' for rubric_id in REQUIRED_RUBRIC_IDS
    )
    return (
        f'{{\n  "analysis_id": "${{ANALYSIS_ID}}",\n  "overall_pass": {str(overall_pass).lower()},\n'
        f'  "rubrics": [\n{rubrics}\n  ]\n}}'
    )


_DICT_STYLE_RUBRIC_JSON = """{
  "analysis_id": "${ANALYSIS_ID}",
  "overall_pass": true,
  "analysis_summary_present": {"status": "pass", "message": "summary present"},
  "analysis_manifest_present": {"status": "pass", "message": "manifest present"},
  "interestingness_rubric_present": {"status": "pass", "message": "rubric present"},
  "required_outputs_present": {"status": "pass", "message": "outputs present"},
  "claim_grounded": {"status": "pass", "message": "claim grounded"},
  "checkpoint_guardrail_respected": {"status": "pass", "message": "read-only checkpoint analysis"}
}"""

_STATUS_STYLE_RUBRIC_JSON = """{
  "analysis_id": "${ANALYSIS_ID}",
  "all_passed": true,
  "rubrics": [
    {"id":"analysis_summary_present","status":"passed","message":"summary present"},
    {"id":"analysis_manifest_present","status":"passed","message":"manifest present"},
    {"id":"interestingness_rubric_present","status":"passed","message":"rubric present"},
    {"id":"required_outputs_present","status":"passed","message":"outputs present"},
    {"id":"claim_grounded","status":"passed","message":"claim grounded"},
    {"id":"checkpoint_guardrail_respected","status":"passed","message":"checkpoint respected"}
  ]
}"""


def _rubric_block(rubric_json: str) -> str:
    return f'cat > "$WORK_DIR/rubric_status.json" <<JSON\n{rubric_json}\nJSON\n'


def _write_fake_execute_script(
    path: Path,
    *,
    rubric_block: str | None = None,
    replacements: dict[str, str] | None = None,
    extra: str = "",
) -> Path:
    """Write a fake agent script that produces a passing attempt, with optional manifest edits/extra bash."""
    script = _SCRIPT_HEAD + (rubric_block or _rubric_block(_list_rubric_json())) + _SCRIPT_TAIL + extra
    for old, new in (replacements or {}).items():
        assert old in script, old
        script = script.replace(old, new)
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


_FULL_AGENT = dict(agent="codex", agent_model="opus", agent_profile="research", reasoning_effort="high")


def _run(specs: list[AnalysisSpec], run_dir: Path, script: Path, **overrides: Any) -> list[Any]:
    kwargs: dict[str, Any] = dict(
        agent="pi",
        agent_model="deepseek",
        timeout_seconds=10,
        timeout_grace_seconds=1,
        max_attempts_per_analysis=1,
        script_path=script,
    )
    kwargs.update(overrides)
    return execute_analysis_agents(analysis_specs=specs, posthoc_run_dir=run_dir, **kwargs)


def _run_manifest(spec: AnalysisSpec) -> dict[str, Any]:
    return json.loads((spec.analysis_dir / "run_manifest.json").read_text(encoding="utf-8"))


def _only_attempt_1(body: str) -> str:
    return f'\nif [[ "$ATTEMPT_ID" == "attempt_1" ]]; then\n{body}\nfi\n'


# --------------------------------------------------------------------------- execute: happy paths


def test_execute_runs_all_selected_analyses_and_accepts_artifact_only_analysis(tmp_path: Path) -> None:
    # runs every selected analysis and renders a private attempt prompt
    root = tmp_path / "all"
    spec_a = _make_analysis_workspace(root, "activation_probe")
    spec_b = _make_analysis_workspace(root, "error_clusters")
    script = _write_fake_execute_script(root / "fake_execute.sh")

    results = _run([spec_a, spec_b], root, script, max_attempts_per_analysis=2, **_FULL_AGENT)

    assert len(results) == 2
    assert all(result.status == "completed" for result in results)
    assert all(result.selected_attempt == "attempt_1" for result in results)
    assert (spec_a.analysis_dir / "analysis_result.json").exists()
    assert (spec_b.analysis_dir / "analysis_result.json").exists()
    prompt = spec_a.analysis_dir / "attempt_1" / "posthoc_attempt_prompt.md"
    assert stat.S_IMODE(prompt.stat().st_mode) == 0o600
    rendered_prompt = prompt.read_text(encoding="utf-8")
    assert 'attempt_id: "attempt_1"' in rendered_prompt
    assert "task_bundle/evidence/" in rendered_prompt
    assert '`rubrics` list with one `{"id": ..., "pass": boolean}`' in rendered_prompt

    # accepts an artifact-only analysis without a checkpoint
    root = tmp_path / "artifact"
    spec = _make_analysis_workspace(root, "score_landscape")
    (spec.analysis_dir / "task_bundle" / "program_0" / "best_model.pt").unlink()
    script = _write_fake_execute_script(
        root / "fake_artifact_execute.sh",
        replacements={'"checkpoint_paths": ["task_bundle/program_0/best_model.pt"]': '"checkpoint_paths": []'},
    )

    result = _run([spec], root, script, checkpoint_mode="optional")[0]

    assert result.status == "completed"
    assert result.selected_attempt == "attempt_1"
    assert result.guardrail_passed is None
    assert result.attempts[0].checkpoint_validated is False


_RUBRIC_SHAPE_CASES: list[tuple[str, str, dict[str, Any], bool]] = [
    ("dict_style", _DICT_STYLE_RUBRIC_JSON, _FULL_AGENT, True),
    ("all_passed_status", _STATUS_STYLE_RUBRIC_JSON, {"agent": "codex", "agent_model": None}, False),
]


def test_execute_accepts_alternative_rubric_shapes_and_retries_after_fifo(tmp_path: Path) -> None:
    # accepts alternative rubric status shapes and refreshes the run manifest
    for name, rubric_json, agent_kwargs, expect_manifest_refresh in _RUBRIC_SHAPE_CASES:
        root = tmp_path / name
        spec = _make_analysis_workspace(root, "weight_distribution_analysis")
        script = _write_fake_execute_script(root / "fake_rubric_execute.sh", rubric_block=_rubric_block(rubric_json))

        results = _run([spec], root, script, **agent_kwargs)

        assert len(results) == 1, name
        assert results[0].status == "completed", name
        assert results[0].selected_attempt == "attempt_1", name
        assert results[0].rubric_passed is True, name
        manifest = _run_manifest(spec)
        assert manifest["status"] == "completed", name
        if expect_manifest_refresh:
            assert manifest["agent"] == "codex"
            assert manifest["agent_model"] == "opus"
            assert manifest["agent_profile"] == "research"
            assert manifest["reasoning_effort"] == "high"
            assert manifest["started_at"] != "old-start"
            assert manifest["completed_at"] != "old-complete"

    # retries when the previous attempt contains a fifo
    root = tmp_path / "fifo"
    spec = _make_analysis_workspace(root, "activation_probe")
    failing = _list_rubric_json(False, failing="checkpoint_guardrail_respected")
    rubric_block = (
        'if [[ "$ATTEMPT_ID" == "attempt_1" ]]; then\n'
        + _rubric_block(failing)
        + '  mkfifo "$WORK_DIR/.posthoc_stream_retry.fifo"\nelse\n'
        + _rubric_block(_list_rubric_json())
        + "fi\n"
    )
    script = _write_fake_execute_script(root / "fake_retry_execute.sh", rubric_block=rubric_block)

    results = _run([spec], root, script, max_attempts_per_analysis=2, **{**_FULL_AGENT, "agent": "claude"})

    assert len(results) == 1
    assert results[0].status == "completed"
    assert results[0].selected_attempt == "attempt_2"
    assert (spec.analysis_dir / "attempt_1" / ".posthoc_stream_retry.fifo").exists()
    assert not (spec.analysis_dir / "attempt_2" / ".posthoc_stream_retry.fifo").exists()


# --------------------------------------------------------------------------- execute: attempt rejection


def _unlink_checkpoint(spec: AnalysisSpec, monkeypatch: pytest.MonkeyPatch) -> None:
    (spec.analysis_dir / "task_bundle" / "program_0" / "best_model.pt").unlink()


def _corrupt_checkpoint(spec: AnalysisSpec, monkeypatch: pytest.MonkeyPatch) -> None:
    (spec.analysis_dir / "task_bundle" / "program_0" / "best_model.pt").write_bytes(b"not-a-checkpoint")


def _shrink_tree_file_limit(spec: AnalysisSpec, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tree_digest, "MAX_TREE_FILE_BYTES", 1024)


_NO_CHECKPOINT = {'"checkpoint_paths": ["task_bundle/program_0/best_model.pt"]': '"checkpoint_paths": []'}
_NON_BOOLEAN_VALUES: tuple[object, ...] = (None, 1, "false", {})


@pytest.mark.parametrize(
    ("prepare", "replacements", "extra", "match", "attempt_guardrail"),
    [
        pytest.param(
            _unlink_checkpoint,
            {**_NO_CHECKPOINT, '"reran_evaluator": false': '"reran_evaluator": true'},
            "",
            "reran_evaluator must be false",
            False,
            id="optional_checkpoint_still_rejects_evaluator_rerun",
        ),
        *[
            pytest.param(
                None,
                {'"reran_evaluator": false': f'"reran_evaluator": {json.dumps(value)}'},
                "",
                "reran_evaluator must be false",
                ...,
                id=f"non_boolean_reran_evaluator[{value!r}]",
            )
            for value in _NON_BOOLEAN_VALUES
        ],
        pytest.param(
            None,
            {"task_bundle/program_0/best_model.pt": "../../outside.pt"},
            "",
            "checkpoint_paths",
            ...,
            id="missing_declared_checkpoint",
        ),
        pytest.param(_corrupt_checkpoint, None, "", "checkpoint not loadable", ..., id="corrupt_declared_checkpoint"),
        pytest.param(
            None,
            None,
            '\nprintf "mutated" >> "$WORK_DIR/task_bundle/program_0/src/program.py"\n',
            "task_bundle was modified",
            ...,
            id="task_bundle_mutation",
        ),
        pytest.param(
            _shrink_tree_file_limit,
            None,
            '\ntruncate -s 4096 "$WORK_DIR/task_bundle/oversized.bin"\n',
            "tree digest per-file byte limit",
            ...,
            id="oversized_sparse_task_bundle",
        ),
    ],
)
def test_execute_rejects_invalid_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prepare: Callable[[AnalysisSpec, pytest.MonkeyPatch], None] | None,
    replacements: dict[str, str] | None,
    extra: str,
    match: str,
    attempt_guardrail: object,
) -> None:
    spec = _make_analysis_workspace(tmp_path, "score_landscape")
    if prepare is not None:
        prepare(spec, monkeypatch)
    script = _write_fake_execute_script(tmp_path / "fake_execute.sh", replacements=replacements, extra=extra)

    result = _run([spec], tmp_path, script, checkpoint_mode="optional")[0]

    assert result.status == "failed"
    assert result.selected_attempt is None
    assert result.guardrail_passed is None
    assert match in str(result.attempts[0].manifest_error)
    if attempt_guardrail is not ...:
        assert result.attempts[0].guardrail_passed is attempt_guardrail


@pytest.mark.parametrize(
    ("attempt_1_body", "source_error"),
    [
        pytest.param('  printf "mutated" >> "$WORK_DIR/task_bundle/program_0/src/program.py"', None, id="mutated_bundle"),
        pytest.param(
            '  printf "mutated" >> "$WORK_DIR/../task_bundle/program_0/src/program.py"',
            "source task_bundle was modified",
            id="mutated_source_bundle",
        ),
        pytest.param(
            '  mv "$WORK_DIR/task_bundle" "$WORK_DIR/task_bundle_saved"\n  ln -s task_bundle_saved "$WORK_DIR/task_bundle"',
            None,
            id="symlink_to_saved",
        ),
        pytest.param(
            '  mv "$WORK_DIR/task_bundle" "$WORK_DIR/task_bundle_saved"\n  ln -s missing_bundle "$WORK_DIR/task_bundle"',
            None,
            id="dangling_symlink",
        ),
        pytest.param(
            '  mv "$WORK_DIR/task_bundle" "$WORK_DIR/task_bundle_saved"\n  printf "replacement" > "$WORK_DIR/task_bundle"',
            None,
            id="replaced_by_file",
        ),
    ],
)
def test_execute_retry_restores_pristine_task_bundle(
    tmp_path: Path, attempt_1_body: str, source_error: str | None
) -> None:
    spec = _make_analysis_workspace(tmp_path, "score_landscape")
    script = _write_fake_execute_script(tmp_path / "fake_retry.sh", extra=_only_attempt_1(attempt_1_body))

    result = _run([spec], tmp_path, script, max_attempts_per_analysis=2, checkpoint_mode="optional")[0]

    assert result.status == "completed"
    assert result.selected_attempt == "attempt_2"
    assert result.attempts[0].overall_pass is False
    if source_error is not None:
        assert source_error in str(result.attempts[0].manifest_error)
    retry_bundle = spec.analysis_dir / "attempt_2" / "task_bundle"
    assert retry_bundle.is_dir()
    assert not retry_bundle.is_symlink()
    for program in (retry_bundle, spec.analysis_dir / "task_bundle"):
        assert "mutated" not in (program / "program_0" / "src" / "program.py").read_text(encoding="utf-8")


def test_execute_rejects_retraining_specs_before_launch_and_emits_identity_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # retraining specs are rejected before any agent launch and the manifest records the agent
    root = tmp_path / "reject"
    spec = _make_analysis_workspace(root, "ablation_mean_centering", requires_retrain=True)

    results = _run([spec], root, root / "missing.sh", max_attempts_per_analysis=2, **_FULL_AGENT)

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].attempt_count == 0
    assert results[0].guardrail_passed is False
    assert "analysis-only" in results[0].output
    manifest = _run_manifest(spec)
    assert manifest["status"] == "failed"
    assert manifest["agent"] == "codex"
    assert manifest["agent_model"] == "opus"
    assert manifest["agent_profile"] == "research"
    assert manifest["reasoning_effort"] == "high"
    assert manifest["started_at"] is None

    # analysis identity and result checks are emitted as events
    root = tmp_path / "events"
    analysis_dir = root / "analyses" / "retraining_rejected"
    analysis_dir.mkdir(parents=True)
    spec = AnalysisSpec(
        analysis_id="retraining_rejected",
        title="Retraining Rejected",
        family="ablation",
        analysis_dir=analysis_dir,
        requires_retrain=True,
    )
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(execute, "log_event", lambda event, **fields: events.append((event, fields)))

    results = execute.execute_analysis_agents(
        analysis_specs=[spec],
        posthoc_run_dir=root,
        agent="codex",
        agent_model="gpt-5",
        timeout_seconds=10,
        timeout_grace_seconds=1,
        max_attempts_per_analysis=2,
    )

    assert results[0].status == "failed"
    identity = {"analysis_id": "retraining_rejected", "title": "Retraining Rejected", "family": "ablation"}
    assert events == [
        ("analysis_start", {**identity, "current": 1, "total": 1, "max_attempts": 2}),
        (
            "analysis_result",
            {
                **identity,
                "status": "failed",
                "attempt_count": 0,
                "exit_code": 1,
                "rubric_passed": False,
                "required_outputs_present": False,
                "guardrail_passed": False,
                "produced_file_count": 0,
                "output_dir": analysis_dir.resolve(),
            },
        ),
        ("analyses_summary", {"total": 1, "passed": 0, "failed": 1, "timeout": 0}),
    ]


# --------------------------------------------------------------------------- common + rubric contract


def _list_style_status(analysis_id: str, overall_pass: bool = True) -> dict[str, object]:
    return {
        "analysis_id": analysis_id,
        "overall_pass": overall_pass,
        "rubrics": [{"id": rubric_id, "pass": overall_pass} for rubric_id in REQUIRED_RUBRIC_IDS],
    }


def _dict_style_status(analysis_id: str, overall_pass: bool = True) -> dict[str, Any]:
    status: dict[str, Any] = {"analysis_id": analysis_id, "overall_pass": overall_pass}
    for rubric_id in REQUIRED_RUBRIC_IDS:
        status[rubric_id] = {"status": "pass" if overall_pass else "fail", "message": f"{rubric_id} checked"}
    return status


def _nested_dict_style_status(analysis_id: str) -> dict[str, Any]:
    nested = _dict_style_status(analysis_id)
    return {
        "analysis_id": nested.pop("analysis_id"),
        "overall_pass": nested.pop("overall_pass"),
        "rubric_status": nested,
    }


def _status_entry_list_status(analysis_id: str) -> dict[str, Any]:
    return {
        "analysis_id": analysis_id,
        "all_passed": True,
        "rubrics": [{"id": rubric_id, "status": "passed", "message": f"{rubric_id} ok"} for rubric_id in REQUIRED_RUBRIC_IDS],
    }


_SUPPORTED_STATUS_SHAPES: list[tuple[str, Callable[[str], dict[str, Any]]]] = [
    ("list_style", _list_style_status),
    ("dict_style", _dict_style_status),
    ("nested_dict_style", _nested_dict_style_status),
    ("list_style_status_entries", _status_entry_list_status),
]


def test_rubric_contract_validates_outputs_status_shapes_loading_and_normalization(tmp_path: Path) -> None:
    # expected outputs must be regular files
    (tmp_path / "outputs" / "figure.png").mkdir(parents=True)

    result = validate_expected_outputs(tmp_path, ["outputs/figure.png"])

    assert result["all_present"] is False
    assert result["missing"] == ["outputs/figure.png"]

    # every supported rubric status shape validates
    for name, build in _SUPPORTED_STATUS_SHAPES:
        passed, error = validate_posthoc_rubric_status(build("probe"), "probe")

        assert passed, name
        assert error is None, name

    # a dict-style status missing a required rubric is rejected
    status = _dict_style_status("probe")
    status.pop("claim_grounded")

    passed, error = validate_posthoc_rubric_status(status, "probe")

    assert not passed
    assert error == "rubric_status missing rubric ids: ['claim_grounded']"

    # load_rubric_status returns None for invalid JSON and loads a dict
    path = tmp_path / "rubric_status.json"
    path.write_text("{invalid", encoding="utf-8")
    assert load_rubric_status(path) is None

    path.write_text(json.dumps(_dict_style_status("probe")), encoding="utf-8")
    loaded = load_rubric_status(path)
    assert loaded is not None
    assert loaded["analysis_id"] == "probe"

    # normalization canonicalizes overall and per-rubric passes
    normalized, error = normalize_posthoc_rubric_status(_status_entry_list_status("probe"))

    assert error is None
    assert normalized is not None
    assert normalized["overall_pass"] is True
    assert all(entry["pass"] is True for entry in normalized["rubrics"])


# --------------------------------------------------------------------------- posthoc_agent.sh env policy


_ENV_POLICY_CASES = [
    (
        "native_env_allowlisted",
        'test "${OPENROUTER_API_KEY:-}" = openrouter-key\ntest -z "${S2_API_KEY:-}"\n',
        {"OPENROUTER_API_KEY": "openrouter-key", "S2_API_KEY": "paper-only-key"},
        ["--agent-model", "openrouter/deepseek/deepseek-v4-flash"],
        None,
    ),
    (
        "pi_provider_selects_matching_credentials",
        'test "${PI_CODING_AGENT_PROVIDER:-}" = anthropic\n'
        'test "${ANTHROPIC_API_KEY:-}" = anthropic-key\n'
        'test -z "${OPENAI_API_KEY:-}"\n',
        {
            "PI_CODING_AGENT_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "anthropic-key",
            "OPENAI_API_KEY": "wrong-provider-key",
        },
        [],
        "PI_CODING_AGENT_PROVIDER=openrouter\nOPENROUTER_API_KEY=container-only-key\n",
    ),
]


def test_posthoc_agent_script_environment_is_provider_allowlisted(tmp_path: Path) -> None:
    # the container env path is allowlisted in the script source
    script = Path(POSTHOC_AGENT_SCRIPT).read_text(encoding="utf-8")

    assert "POSTHOC_AGENT_ALLOWED_ENV_KEYS=" in script
    assert '"$POSTHOC_AGENT_ALLOWED_ENV_KEYS"' in script
    assert "sanitize_env_file_for_docker" in script
    assert "posthoc-agent-env-XXXXXX" in script

    # the native agent environment is provider-allowlisted at runtime
    for name, headless_checks, env, extra_args, ignored_env_file in _ENV_POLICY_CASES:
        root = tmp_path / name
        work_dir = root / "attempt"
        work_dir.mkdir(parents=True)
        fake_headless = root / "bin" / "headless"
        fake_headless.parent.mkdir()
        fake_headless.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + headless_checks, encoding="utf-8")
        fake_headless.chmod(0o755)
        environment = {**os.environ, "HEADLESS_BIN": str(fake_headless), "WRAPPER_PYTHON": sys.executable, **env}
        args = ["bash", POSTHOC_AGENT_SCRIPT, "--agent", "pi", *extra_args]
        args += ["--work-dir", str(work_dir), "--prompt", "stop", "--analysis-id", "test_analysis"]
        if ignored_env_file is not None:
            env_file = root / "container.env"
            env_file.write_text(ignored_env_file, encoding="utf-8")
            args += ["--container-env-file", str(env_file)]

        result = subprocess.run(args, capture_output=True, text=True, timeout=20, env=environment)

        assert result.returncode == 0, (name, result.stdout + result.stderr)


# --------------------------------------------------------------------------- logging / CLI


def _config(tmp_path: Path) -> PosthocPipelineConfig:
    return PosthocPipelineConfig(
        paths=PosthocPathsConfig(
            task_dir=str(tmp_path / "task"),
            results_dir=str(tmp_path / "results"),
            output_dir=str(tmp_path / "posthoc"),
        ),
        run_id="test-run",
    )


def test_posthoc_logging_defaults_overrides_validation_and_heartbeat_flags(tmp_path: Path) -> None:
    # logging defaults and overrides
    defaults = cli.load_posthoc_config([])
    overridden = cli.load_posthoc_config(
        ["logging.format=json", "logging.verbose=true", "logging.heartbeat_seconds=15", "logging.show_run_header=false"]
    )

    assert (defaults.logging.format, defaults.logging.verbose) == ("human", False)
    assert (defaults.logging.heartbeat_seconds, defaults.logging.show_run_header) == (60, True)
    assert (overridden.logging.format, overridden.logging.verbose) == ("json", True)
    assert (overridden.logging.heartbeat_seconds, overridden.logging.show_run_header) == (15, False)

    # logging validation rejects invalid values
    cfg = _config(tmp_path)
    cfg.logging.format = "legacy"
    cfg.logging.heartbeat_seconds = -1

    errors = validate_posthoc_config(cfg)

    assert any("logging.format" in error for error in errors)
    assert any("logging.heartbeat_seconds" in error for error in errors)

    # heartbeat convenience flags become hydra overrides
    for arguments in (["--heartbeat-seconds", "15"], ["--heartbeat-seconds=15"]):
        stages, overrides = cli._parse_cli_arguments(arguments)

        assert stages is None, arguments
        assert overrides == ["logging.heartbeat_seconds=15"], arguments


_PIPELINE_END_CASES: list[tuple[str, BaseException, dict[str, object]]] = [
    ("raises", RuntimeError("boom"), {"status": "failed", "error_type": "RuntimeError"}),
    ("interrupt", KeyboardInterrupt(), {"status": "interrupted", "return_code": 130}),
]


def test_cli_configures_event_logging_and_reports_pipeline_end_before_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # convenience flags configure event logging and shut it down afterwards
    cfg = _config(tmp_path)
    loaded_overrides: list[str] = []
    configured: list[dict[str, object]] = []
    shutdown_calls: list[None] = []

    def fake_load(overrides: list[str] | None = None) -> PosthocPipelineConfig:
        loaded_overrides.extend(overrides or [])
        cfg.logging.format = "json"
        cfg.logging.verbose = True
        return cfg

    with monkeypatch.context() as patch:
        patch.setattr(cli, "load_posthoc_config", fake_load)
        patch.setattr(cli, "configure_event_logging", lambda **kwargs: configured.append(kwargs))
        patch.setattr(cli, "shutdown_event_logging", lambda: shutdown_calls.append(None))
        patch.setattr(cli, "run_posthoc_pipeline", lambda config, stages: 0)

        result = cli.main(["--stages", "5", "--log-format", "json", "--verbose"])

    assert result == 0
    assert loaded_overrides == ["logging.format=json", "logging.verbose=true"]
    assert configured == [
        {
            "log_format": "json",
            "verbose": True,
            "heartbeat_seconds": 60,
            "results_dir": (tmp_path / "posthoc").resolve(),
            "worker_index": 0,
            "source": "posthoc",
            "renderer": "posthoc",
            "journal_name": "events.jsonl",
        }
    ]
    assert shutdown_calls == [None]

    # pipeline_end is reported before event logging shuts down, for failures and interrupts
    for name, exc, expected_event in _PIPELINE_END_CASES:
        cfg = _config(tmp_path)
        events: list[tuple[str, dict[str, object]]] = []
        shutdown_calls = []

        def fail_pipeline(config: PosthocPipelineConfig, stages: set[int] | None, exc: BaseException = exc) -> int:
            raise exc

        with monkeypatch.context() as patch:
            patch.setattr(cli, "load_posthoc_config", lambda overrides=None, cfg=cfg: cfg)
            patch.setattr(cli, "configure_event_logging", lambda **kwargs: None)
            patch.setattr(cli, "log_event", lambda event, **fields: events.append((event, fields)))
            patch.setattr(cli, "shutdown_event_logging", lambda: shutdown_calls.append(None))
            patch.setattr(cli, "run_posthoc_pipeline", fail_pipeline)

            if isinstance(exc, KeyboardInterrupt):
                assert cli.main([]) == 130, name
            else:
                with pytest.raises(RuntimeError, match="boom"):
                    cli.main([])

        assert events == [("pipeline_end", expected_event)], name
        assert shutdown_calls == [None], name


def test_pipeline_emits_paired_stage_and_summary_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(tmp_path)
    output_dir = Path(cfg.paths.output_dir or "")
    (output_dir / "_extracted_bundle").mkdir(parents=True)
    (output_dir / "_analysis_ready_bundle").mkdir(parents=True)
    events: list[tuple[str, dict[str, object]]] = []
    for module in (pipeline, progress):
        monkeypatch.setattr(module, "log_event", lambda event, **fields: events.append((event, fields)))

    result = pipeline.run_posthoc_pipeline(cfg, stages=set())

    assert result == 0
    assert events[0][0] == "pipeline_start"
    assert events[0][1]["show_run_header"] is True
    starts = [fields for event, fields in events if event == "stage_start"]
    ends = [fields for event, fields in events if event == "stage_end"]
    assert [fields["stage"] for fields in starts] == list(range(7))
    assert [fields["stage"] for fields in ends] == list(range(7))
    assert all(fields["status"] == "completed" for fields in ends)
    summary = next(fields for event, fields in events if event == "pipeline_summary")
    assert summary == {
        "status": "completed",
        "stages": [],
        "candidate_count": 0,
        "selected_count": 0,
        "analysis_count": 0,
        "completed_count": 0,
        "rubric_passed_count": 0,
        "output_dir": output_dir.resolve(),
    }
    assert events[-1][0] == "pipeline_end"
    assert events[-1][1]["status"] == "completed"
