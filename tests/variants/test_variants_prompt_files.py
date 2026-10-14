"""Prompt staging, retrofit sealing and pipeline stage-resume guards."""

from __future__ import annotations

import json
import os
import stat
from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO, Any
from unittest.mock import patch

import pytest

from src.variants_pipeline import pipeline as variants_pipeline
from src.variants_pipeline import retrofit
from src.variants_pipeline.agent_prompt import stage_agent_prompt
from src.variants_pipeline.config import VariantsPipelineConfig
from src.variants_pipeline.generate import generate_variant_candidates
from src.variants_pipeline.models import VariantCandidate, VariantSpec
from src.variants_pipeline.pipeline import run_variants_pipeline
from src.variants_pipeline.readme import generate_variants_readme
from src.variants_pipeline.retrofit import evaluator_tree_digest, retrofit_bundle_complete, retrofit_program
from src.variants_pipeline.select import validate_and_select_variants

_NO_SPACE = OSError(28, "No space left on device")
_RUNNER = "src.investigation.script_runners._execute_script_command"
_RETROFIT_COMPLETE = "src.variants_pipeline.retrofit.retrofit_bundle_complete"
_MATERIALIZE = "src.variants_pipeline.pipeline.materialize_variant_workspaces"
_RESOLVE_SANDBOX = "src.variants_pipeline.execute.resolve_sandbox_image"
_VALIDATE_SUMMARY = "src.variants_pipeline.pipeline._validate_pipeline_summary"
_CROSS_TASK = "src.variants_pipeline.cross_task.prepare_and_execute_cross_task"
_AGENT_KW: dict[str, Any] = dict(agent="codex", agent_model=None, timeout_seconds=20, timeout_grace_seconds=1)
_FAKE_AGENT_SCRIPT = """#!/bin/bash
set -euo pipefail
work_dir=""
prompt_file=""
printf '%s\\n' "$@" > "$CAPTURED_ARGS"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --work-dir) work_dir="$2"; shift 2 ;;
    --prompt-file) prompt_file="$2"; shift 2 ;;
    --prompt) echo "prompt leaked into argv" >&2; exit 91 ;;
    *) shift ;;
  esac
done
test -n "$work_dir"
test -n "$prompt_file"
cp "$work_dir/$prompt_file" "$CAPTURED_PROMPT"
printf '{"decision":"none","summary":"done","candidates":[]}\\n' > "$work_dir/variant_bundle.json"
"""


def _bundle(root: Path, **files: str) -> Path:
    bundle_dir = root / "bundle"
    bundle_dir.mkdir(parents=True)
    for name, content in files.items():
        (bundle_dir / name).write_text(content, encoding="utf-8")
    return bundle_dir


def _retrofit(bundle_dir: Path, output_dir: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        bundle_dir=bundle_dir, output_bundle_dir=output_dir, agent="pi", agent_model="model",
        timeout_seconds=20, timeout_grace_seconds=1, container_execution_env=lambda **_: {},
    )
    kwargs.update(overrides)
    return retrofit_program(**kwargs)


def _spec(output_dir: Path, *, kind: str = "control", title: str = "Control", **kw: Any) -> VariantSpec:
    return VariantSpec(
        variant_id="control_best", title=title, kind=kind, base_program="best",
        variant_dir=output_dir / "variants" / "control_best", **kw,
    )


def _cfg(root: Path, *, container: bool = True, extracted: bool = True, extracted_eval: bool = False,
         retrofitted: bool = False, retrofitted_eval: bool = False, extra_task: bool = False):
    """Build a pipeline config with task/results/output dirs under ``root``; return (cfg, output_dir)."""
    root.mkdir(parents=True, exist_ok=True)
    task_dir, results_dir, output_dir = root / "task", root / "results", root / "output"
    task_dir.mkdir()
    results_dir.mkdir()
    output_dir.mkdir()
    if extracted:
        (output_dir / "_extracted_bundle").mkdir()
        if extracted_eval:
            (output_dir / "_extracted_bundle" / "evaluate.py").write_text("# evaluator\n")
    if retrofitted:
        (output_dir / "_retrofitted_bundle").mkdir()
        if retrofitted_eval:
            (output_dir / "_retrofitted_bundle" / "evaluate.py").write_text("# evaluator\n")
    cfg = VariantsPipelineConfig()
    cfg.paths.task_dir = str(task_dir)
    cfg.paths.results_dir = str(results_dir)
    cfg.paths.output_dir = str(output_dir)
    cfg.container.enabled = container
    if extra_task:
        (root / "extra").mkdir()
        cfg.paths.extra_tasks = [str(root / "extra")]
    return cfg, output_dir


# ---------------------------------------------------------------------------
# Prompt staging
# ---------------------------------------------------------------------------


def test_generation_stages_large_prompt_before_subprocess_launch(tmp_path: Path) -> None:
    bundle_dir = _bundle(tmp_path, **{"task.md": "LARGE_VARIANT_PROMPT_SENTINEL\n" + ("X" * 200_000)})
    generation_dir = tmp_path / "generation"
    captured_prompt, captured_args = tmp_path / "captured_prompt.md", tmp_path / "captured_args.txt"
    fake_agent_script = tmp_path / "variant-agent-stub.sh"
    fake_agent_script.write_text(_FAKE_AGENT_SCRIPT, encoding="utf-8")
    fake_agent_script.chmod(0o755)
    env = {**os.environ, "CAPTURED_PROMPT": str(captured_prompt), "CAPTURED_ARGS": str(captured_args)}

    candidates = generate_variant_candidates(
        bundle_dir=bundle_dir, generation_dir=generation_dir, script_path=fake_agent_script,
        container_execution_env=lambda **_: env, **_AGENT_KW,
    )

    staged_prompt = generation_dir / "variant_generate_prompt.md"
    assert candidates == []
    captured_prompt_bytes = captured_prompt.read_bytes()
    assert captured_prompt_bytes == staged_prompt.read_bytes()
    assert b"LARGE_VARIANT_PROMPT_SENTINEL" in captured_prompt_bytes
    assert stat.S_IMODE(staged_prompt.stat().st_mode) == 0o600
    args = captured_args.read_text(encoding="utf-8").splitlines()
    assert "--prompt" not in args
    assert args[args.index("--skill-step") + 1] == "variant_edit"
    assert args[args.index("--prompt-file") + 1] == staged_prompt.name


def test_prompt_staging_rejects_non_basename_replaces_symlink_and_cleans_partial_file(tmp_path: Path) -> None:
    # rejects anything that is not a plain basename
    for filename in ("", "../prompt.md", "/tmp/prompt.md"):
        with pytest.raises(ValueError, match="must be a basename"):
            stage_agent_prompt(tmp_path, filename, "prompt")

    # replaces a symlink without following it
    outside = tmp_path / "outside.md"
    outside.write_text("preserve", encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.symlink_to(outside)
    stage_agent_prompt(tmp_path, prompt_path.name, "private prompt")
    assert not prompt_path.is_symlink()
    assert prompt_path.read_text(encoding="utf-8") == "private prompt"
    assert outside.read_text(encoding="utf-8") == "preserve"
    assert stat.S_IMODE(prompt_path.stat().st_mode) == 0o600

    # cleans the partial file after a write failure
    partial_root = tmp_path / "partial"
    partial_root.mkdir()
    temporary_path = partial_root / ".prompt.md.partial"

    class FailingTemporaryFile(AbstractContextManager[IO[str]]):
        name = str(temporary_path)

        def __enter__(self) -> IO[str]:
            temporary_path.write_text("partial", encoding="utf-8")
            return self  # type: ignore[return-value]

        def write(self, _: str) -> int:
            raise OSError(28, "No space left on device")

        def __exit__(self, *args: object) -> None:
            return None

    with (
        patch("src.investigation.agent_prompt.tempfile.NamedTemporaryFile", return_value=FailingTemporaryFile()),
        pytest.raises(OSError, match="No space left on device"),
    ):
        stage_agent_prompt(partial_root, "prompt.md", "prompt")
    assert not temporary_path.exists()


def test_stages_degrade_gracefully_when_prompt_staging_fails(tmp_path: Path) -> None:
    # generation returns an empty result
    bundle_dir = _bundle(tmp_path / "generation")
    with patch("src.variants_pipeline.generate.stage_agent_prompt", side_effect=_NO_SPACE):
        candidates = generate_variant_candidates(
            bundle_dir=bundle_dir, generation_dir=tmp_path / "generation" / "generation", **_AGENT_KW
        )
    assert candidates == []

    # selection rejects every candidate with an explicit reason
    candidate = VariantCandidate(
        variant_id="ablation_test", title="Test ablation", kind="ablation", base_program="best",
        hypothesis="test", relevance_reason="test", feasibility_reason="test",
        target_components=["component"], implementation_plan=["change component"], expected_effect="test",
    )
    select_root = tmp_path / "select"
    select_root.mkdir()
    with patch("src.variants_pipeline.select.stage_agent_prompt", side_effect=_NO_SPACE):
        selection = validate_and_select_variants(
            candidates=[candidate], bundle_dir=select_root, generation_dir=select_root / "generation", **_AGENT_KW
        )
    assert selection.accepted == []
    assert selection.rejected == [(candidate, "rejected (prompt staging failed)")]

    # retrofit returns an error report
    bundle_dir = _bundle(tmp_path / "retrofit", **{"task.md": "# Task\n", "evaluate.py": "# evaluator\n"})
    with patch("src.variants_pipeline.retrofit.stage_agent_prompt", side_effect=_NO_SPACE):
        report = retrofit_program(
            bundle_dir=bundle_dir, output_bundle_dir=tmp_path / "retrofit" / "retrofitted", **_AGENT_KW
        )
    assert report["exit_code"] == 1
    assert "Failed to stage prompt" in report["error"]

    # readme falls back to the spec hypothesis
    output_dir = tmp_path / "readme" / "output"
    output_dir.mkdir(parents=True)
    task_dir = tmp_path / "readme" / "task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text("# Task\nOptimize accuracy.\n", encoding="utf-8")
    spec = _spec(output_dir, hypothesis="Fallback summary.")
    with patch("src.variants_pipeline.readme.stage_agent_prompt", side_effect=_NO_SPACE):
        readme_path = generate_variants_readme(
            variant_specs=[spec], output_dir=output_dir, task_dir=task_dir, **_AGENT_KW
        )
    assert "Fallback summary." in readme_path.read_text(encoding="utf-8")


def test_retrofit_fallback_prompt_requires_finite_timeseries(tmp_path: Path) -> None:
    bundle_dir = _bundle(tmp_path)
    with patch.object(retrofit, "_PROMPT_TEMPLATE_PATH", tmp_path / "missing.md"):
        prompt = retrofit._build_retrofit_prompt(bundle_dir)
    assert "finite numeric values" in prompt
    assert "omit observations whose values are unavailable" in prompt.lower()
    assert "Do not modify `evaluate.py`" in prompt


# ---------------------------------------------------------------------------
# Retrofit sealing
# ---------------------------------------------------------------------------


def test_retrofit_restores_tampered_evaluator_and_marker_binds_source_tree(tmp_path: Path) -> None:
    # restores the host evaluator after agent tampering; later edits invalidate the marker
    root = tmp_path / "tamper"
    bundle_dir = _bundle(root, **{"evaluate.py": "# sealed evaluator\n"})
    output_dir = root / "retrofitted"
    outside = root / "outside.py"
    outside.write_text("# preserve\n", encoding="utf-8")

    def tamper(**_: object) -> tuple[str, int]:
        evaluator = output_dir / "evaluate.py"
        evaluator.unlink()
        evaluator.symlink_to(outside)
        return "tampered", 0

    with patch(_RUNNER, side_effect=tamper):
        report = _retrofit(bundle_dir, output_dir)

    assert report["exit_code"] == 0
    assert not (output_dir / "evaluate.py").is_symlink()
    assert (output_dir / "evaluate.py").read_text() == "# sealed evaluator\n"
    assert outside.read_text() == "# preserve\n"
    assert retrofit_bundle_complete(output_dir, bundle_dir)
    (output_dir / "evaluate.py").write_text("# changed after completion\n")
    assert not retrofit_bundle_complete(output_dir, bundle_dir)

    # the completion marker binds the complete source tree
    root = tmp_path / "marker"
    bundle_dir = _bundle(root, **{"evaluate.py": "import capture\n", "capture.py": "VALUE = 1\n"})
    output_dir = root / "retrofitted"
    with patch(_RUNNER, return_value=("ok", 0)):
        report = _retrofit(bundle_dir, output_dir)
    assert report["exit_code"] == 0
    assert retrofit_bundle_complete(output_dir, bundle_dir)
    (bundle_dir / "capture.py").write_text("VALUE = 2\n")
    assert not retrofit_bundle_complete(output_dir, bundle_dir)


def test_retrofit_fails_on_source_change_and_restores_evaluator_after_agent_failure(tmp_path: Path) -> None:
    # fails when the agent changes the sealed source tree
    root = tmp_path / "source_changed"
    bundle_dir = _bundle(root, **{"evaluate.py": "import capture\n", "capture.py": "VALUE = 1\n"})
    output_dir = root / "retrofitted"

    def tamper_source(**_: object) -> tuple[str, int]:
        (bundle_dir / "capture.py").write_text("VALUE = 2\n")
        return "ok", 0

    with patch(_RUNNER, side_effect=tamper_source):
        report = _retrofit(bundle_dir, output_dir)
    assert report["exit_code"] == 1
    assert "source changed" in report["error"]
    assert not retrofit_bundle_complete(output_dir, bundle_dir)

    # restores the evaluator after a failed agent run
    root = tmp_path / "failed_agent"
    bundle_dir = _bundle(root, **{"evaluate.py": "# sealed evaluator\n"})
    output_dir = root / "retrofitted"

    def tamper_then_fail(**_: object) -> tuple[str, int]:
        (output_dir / "evaluate.py").write_text("# tampered\n")
        return "failed", 124

    with patch(_RUNNER, side_effect=tamper_then_fail):
        report = _retrofit(bundle_dir, output_dir)
    assert report["exit_code"] == 124
    assert (output_dir / "evaluate.py").read_text() == "# sealed evaluator\n"
    assert not retrofit_bundle_complete(output_dir, bundle_dir)

    # restores the evaluator when the runner raises
    root = tmp_path / "runner_raises"
    bundle_dir = _bundle(root, **{"evaluate.py": "# sealed evaluator\n"})
    output_dir = root / "retrofitted"

    def tamper_then_raise(**_: object) -> tuple[str, int]:
        (output_dir / "evaluate.py").write_text("# tampered\n")
        raise RuntimeError("runner crashed")

    with patch(_RUNNER, side_effect=tamper_then_raise), pytest.raises(RuntimeError, match="runner crashed"):
        _retrofit(bundle_dir, output_dir)
    assert (output_dir / "evaluate.py").read_text() == "# sealed evaluator\n"
    assert not retrofit_bundle_complete(output_dir, bundle_dir)


def test_evaluator_tree_digest_binds_data_and_masked_roots_rejects_symlinks_and_is_unambiguous(
    tmp_path: Path,
) -> None:
    # binds data files, including nested results directories
    for subdir in ("resources", "resources/results"):
        source = tmp_path / subdir.replace("/", "_")
        (source / subdir).mkdir(parents=True)
        (source / "evaluate.py").write_text("# evaluator\n")
        data = source / subdir / "benchmark.json"
        data.write_text('{"score": 1}\n')
        original_digest = evaluator_tree_digest(source)
        data.write_text('{"score": 2}\n')
        assert evaluator_tree_digest(source) != original_digest, subdir

    # binds the presence of masked root directories
    source = tmp_path / "masked"
    source.mkdir()
    (source / "evaluate.py").write_text("# evaluator\n")
    original_digest = evaluator_tree_digest(source)
    (source / "results").mkdir()
    assert evaluator_tree_digest(source) != original_digest

    # rejects source symlinks
    source = tmp_path / "symlinked"
    source.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("# outside\n")
    (source / "capture.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        evaluator_tree_digest(source)

    # file records cannot be confused by NUL-separated names
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a").write_bytes(b"x")
    (first / "b").write_bytes(b"y")
    (second / "a").write_bytes(b"x\0b\0y")
    assert evaluator_tree_digest(first) != evaluator_tree_digest(second)


# ---------------------------------------------------------------------------
# Pipeline stage guards
# ---------------------------------------------------------------------------


def test_pipeline_stage_resume_guards(tmp_path: Path) -> None:
    # aborts after a failed retrofit report
    cfg, output_dir = _cfg(tmp_path / "failed_retrofit", extracted_eval=True)
    with (
        patch("src.variants_pipeline.pipeline.extract_bundle", return_value=output_dir / "_extracted_bundle"),
        patch("src.variants_pipeline.pipeline.retrofit_program", return_value={"exit_code": 1, "error": "seal failed"}),
    ):
        assert run_variants_pipeline(cfg, stages={0, 1}) == 1

    # rejects an unmarked retrofit on stage resume
    cfg, _ = _cfg(tmp_path / "unmarked_retrofit", retrofitted=True, retrofitted_eval=True)
    assert run_variants_pipeline(cfg, stages={4}) == 1

    # rejects a missing retrofit on stage resume
    cfg, _ = _cfg(tmp_path / "missing_retrofit", extracted_eval=True)
    assert run_variants_pipeline(cfg, stages={4}) == 1

    # stage four does not require a container
    cfg, _ = _cfg(tmp_path / "stage_four", container=False, retrofitted=True)
    with patch(_RETROFIT_COMPLETE, return_value=True), patch(_MATERIALIZE, return_value=[]):
        assert run_variants_pipeline(cfg, stages={4}) == 0

    # a non-evaluation stage does not hash the task tree
    cfg, output_dir = _cfg(tmp_path / "no_hash", container=False)
    with (
        patch("src.variants_pipeline.pipeline.extract_bundle", return_value=output_dir / "_extracted_bundle"),
        patch(
            "src.variants_pipeline.retrofit.evaluator_tree_digest",
            side_effect=AssertionError("non-evaluation stage hashed task tree"),
        ),
    ):
        assert run_variants_pipeline(cfg, stages={0}) == 0


def test_pipeline_downstream_resume_requires_prior_summary_and_passing_control(tmp_path: Path) -> None:
    # requires a prior pipeline summary
    cfg, _ = _cfg(tmp_path / "no_summary", container=False)
    assert run_variants_pipeline(cfg, stages={7}) == 1

    # rejects a failed control recorded in the summary
    cfg, output_dir = _cfg(tmp_path / "failed_control", container=False)
    spec = _spec(output_dir)
    (spec.variant_dir / "task_bundle").mkdir(parents=True)
    (spec.variant_dir / "variant_spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant_specs_sha256": variants_pipeline._variant_specs_digest([spec]),
                "rubric_passed": 0,
                "variants": [{"variant_id": "control_best", "status": "failed", "rubric_passed": False}],
            }
        ),
        encoding="utf-8",
    )
    assert run_variants_pipeline(cfg, stages={7}) == 1


def test_cross_task_custom_only_materializes_control_and_stops_without_control(tmp_path: Path) -> None:
    # custom_only cross-task still materializes the required control
    cfg, output_dir = _cfg(tmp_path / "custom_only", retrofitted=True, extra_task=True)
    with (
        patch(_RETROFIT_COMPLETE, return_value=True),
        patch(_RESOLVE_SANDBOX, return_value="sandbox@sha256:verified"),
        patch(_MATERIALIZE, return_value=[_spec(output_dir)]) as materialize,
        patch(_VALIDATE_SUMMARY, return_value=None),
        patch(_CROSS_TASK, return_value={}),
    ):
        exit_code = run_variants_pipeline(cfg, stages={4, 6}, custom_only=True)
    assert exit_code == 0
    assert materialize.call_args.kwargs["custom_only"] is True
    assert materialize.call_args.kwargs["required_controls"] == {"control_best"}

    # cross-task stops before execution when no real control is present
    cfg, output_dir = _cfg(tmp_path / "no_control", retrofitted=True, extra_task=True)
    specs = [_spec(output_dir), _spec(output_dir, kind="baseline", title="Counterfeit control")]
    with (
        patch(_RETROFIT_COMPLETE, return_value=True),
        patch(_RESOLVE_SANDBOX, return_value="sandbox@sha256:verified") as resolve_sandbox,
        patch(_MATERIALIZE, return_value=specs),
        patch(_VALIDATE_SUMMARY, return_value=None) as validate_summary,
        patch(_CROSS_TASK, return_value={}) as execute_cross_task,
    ):
        exit_code = run_variants_pipeline(cfg, stages={4, 6})
    assert exit_code == 1
    resolve_sandbox.assert_not_called()
    validate_summary.assert_not_called()
    execute_cross_task.assert_not_called()


def test_variant_spec_loader_rejects_symlinks(tmp_path: Path) -> None:
    spec_payload = json.dumps(
        {"variant_id": "control_best", "title": "Control", "kind": "control", "base_program": "best"}
    )
    for symlink_kind in ("directory", "spec", "variants_root"):
        root = tmp_path / symlink_kind
        root.mkdir()
        output_dir = root / "output"
        variants_dir = output_dir / "variants"
        outside_dir = root / "outside"
        outside_dir.mkdir()
        if symlink_kind == "variants_root":
            output_dir.mkdir()
            (outside_dir / "control_best").mkdir()
            (outside_dir / "control_best" / "variant_spec.json").write_text(spec_payload)
            variants_dir.symlink_to(outside_dir, target_is_directory=True)
        elif symlink_kind == "directory":
            variants_dir.mkdir(parents=True)
            (outside_dir / "variant_spec.json").write_text(spec_payload)
            (variants_dir / "control_best").symlink_to(outside_dir, target_is_directory=True)
        else:
            (variants_dir / "control_best").mkdir(parents=True)
            (outside_dir / "variant_spec.json").write_text(spec_payload)
            (variants_dir / "control_best" / "variant_spec.json").symlink_to(outside_dir / "variant_spec.json")

        assert variants_pipeline._load_variant_specs_from_disk(output_dir) == [], symlink_kind
