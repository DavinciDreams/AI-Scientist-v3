"""Cross-task adaptation: workspace preparation, runtime preflight, and brokered retrofit."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.investigation.evaluator_tree import managed_evaluator_tree_digest
from src.variants_pipeline.config import VariantsPipelineConfig
from src.variants_pipeline.cross_task import (
    _find_best_program_dir,
    _prepare_control_workspace,
    _prepare_variant_workspaces,
    _write_control_cross_task_md,
    _write_variant_cross_task_md,
    prepare_and_execute_cross_task,
)
from src.variants_pipeline.models import VariantRunResult, VariantSpec
from src.variants_pipeline.pipeline import run_variants_pipeline
from src.variants_pipeline.retrofit import (
    evaluator_tree_digest,
    retrofit_bundle_complete,
    retrofit_program,
)
from tests.program_testutils import make_program_dir
from tests.task_drafting.test_task_contract import _write_scaffold

IMAGE = "sha256:" + "a" * 64
_PROGRAM_SPEC = """\
schema_version = 1
code_root = "src"

[entrypoint]
path = "program.py"
language = "python"
type = "script"

[copy]
exclude = []
include = []

[integration]
kind = "generic"
"""
_WORKSPACE_KW: dict[str, Any] = dict(
    original_task_md_text="# Original\n", agent="claude", agent_model=None, max_evals_per_variant=2,
)
_RUNTIME_KW: dict[str, Any] = dict(
    agent="claude", agent_model="sonnet", container_image="evaluator:latest",
    container_network="bridge", sandbox_image=IMAGE,
)
_PREPARE_RUNTIME = "src.investigation.task_runtime.prepare_task_runtime"


@pytest.fixture(autouse=True)
def _managed_runtime_ready():
    with patch(_PREPARE_RUNTIME, return_value=True):
        yield


def _run_dir(tmp_path: Path, name: str = "variants-run") -> Path:
    variants_run_dir = tmp_path / name
    variants_run_dir.mkdir(parents=True)
    return variants_run_dir


def _make_program(task_bundle: Path, idx: int, *, with_metrics: bool = False) -> Path:
    prog = task_bundle / f"program_{idx}"
    (prog / "src").mkdir(parents=True)
    (prog / "src" / "program.py").write_text("# program\n", encoding="utf-8")
    (prog / "program_spec.toml").write_text(_PROGRAM_SPEC, encoding="utf-8")
    if with_metrics:
        (prog / "metrics.json").write_text('{"acc": 0.9}', encoding="utf-8")
    return prog


def _make_spec(variant_id: str, variant_dir: Path, kind: str = "ablation") -> VariantSpec:
    return VariantSpec(
        variant_id=variant_id, title=f"Test variant {variant_id}", kind=kind,
        base_program="best", variant_dir=variant_dir, hypothesis="Test hypothesis.",
        expected_effect="Test effect.",
    )


def _make_extra_task(tmp_path: Path, name: str) -> Path:
    task_dir = tmp_path / name
    task_dir.mkdir(parents=True)
    (task_dir / "evaluate.py").write_text("# evaluate\n", encoding="utf-8")
    (task_dir / "task.md").write_text(f"# {name} task\nDo the task.\n", encoding="utf-8")
    return task_dir


def _bare_tasks(tmp_path: Path, *names: str) -> list[Path]:
    tasks = [tmp_path / name for name in names]
    for task in tasks:
        task.mkdir(parents=True)
    return tasks


def _run_cross_task(
    extra_task_paths: Iterable[Path], variants_run_dir: Path, variant_specs: Iterable[VariantSpec] = (),
    **overrides: Any,
) -> dict[str, list[VariantRunResult]]:
    kwargs: dict[str, Any] = dict(
        extra_task_paths=list(extra_task_paths), variants_run_dir=variants_run_dir,
        variant_specs=list(variant_specs), timeout_seconds=20, timeout_grace_seconds=1,
        retrofit_timeout_seconds=20, max_evals_per_variant=1, num_replications=1, agent="pi",
        agent_model="model", containerize=True,
    )
    kwargs.update(overrides)
    return prepare_and_execute_cross_task(**kwargs)


# ── prepare_and_execute_cross_task ────────────────────────────────


def test_cross_task_aborts_on_retrofit_failure_failed_control_or_missing_control(tmp_path):
    # A failed retrofit aborts.
    root = tmp_path / "retrofit"
    with (
        patch(
            "src.variants_pipeline.retrofit.retrofit_program",
            return_value={"exit_code": 1, "error": "seal failed"},
        ),
        pytest.raises(RuntimeError, match="retrofit failed"),
    ):
        _run_cross_task([_make_extra_task(root, "extra")], _run_dir(root))

    # A failed adapted control aborts.
    root = tmp_path / "control_failed"
    run_dir = _run_dir(root)
    original_control = _make_spec("control_best", run_dir / "variants/control_best", kind="control")
    prepared_control = _make_spec(
        "control_best", run_dir / "extra_tasks/extra/control_best", kind="baseline",
    )
    failed = VariantRunResult(
        variant_id="control_best", status="failed", eval_count=1, rubric_passed=False,
        exit_code=1, output="failed", analysis_files={},
    )
    with (
        patch("src.variants_pipeline.retrofit.retrofit_bundle_complete", return_value=True),
        patch("src.variants_pipeline.retrofit.evaluator_tree_digest", return_value="sealed"),
        patch(
            "src.variants_pipeline.cross_task._prepare_control_workspace",
            return_value=[prepared_control],
        ),
        patch("src.variants_pipeline.execute.execute_variant_agents", return_value=[failed]),
        pytest.raises(RuntimeError, match="adapted control failed verification"),
    ):
        _run_cross_task([_make_extra_task(root, "extra")], run_dir, [original_control])

    # control_best is required.
    root = tmp_path / "no_control"
    run_dir = _run_dir(root)
    ablation = _make_spec("ablation", run_dir / "variants" / "ablation")
    with pytest.raises(RuntimeError, match="requires control_best"):
        _run_cross_task([_make_extra_task(root, "extra")], run_dir, [ablation])

    # The control must be preparable.
    root = tmp_path / "unpreparable"
    run_dir = _run_dir(root)
    control = _make_spec("control_best", run_dir / "variants" / "control_best", kind="control")
    with (
        patch("src.variants_pipeline.retrofit.retrofit_bundle_complete", return_value=True),
        patch("src.variants_pipeline.retrofit.evaluator_tree_digest", return_value="sealed"),
        patch("src.variants_pipeline.cross_task._prepare_control_workspace", return_value=[]),
        pytest.raises(RuntimeError, match="no evaluated program"),
    ):
        _run_cross_task([_make_extra_task(root, "extra")], run_dir, [control])


# ── managed runtime preflight before cross-task outputs ──────────


def test_cross_task_runtime_preflight_runs_for_every_task_before_any_output(tmp_path) -> None:
    # A runtime failure or exception precedes any task outputs.
    for name, effect in (
        ("runtime_failure", {"return_value": False}),
        ("runtime_exception", {"side_effect": PermissionError("denied")}),
    ):
        root = tmp_path / name
        run_dir = _run_dir(root)
        (extra_task,) = _bare_tasks(root, "extra")
        with (
            patch(_PREPARE_RUNTIME, **effect) as prepare_runtime,
            patch("src.variants_pipeline.retrofit.retrofit_program") as retrofit,
            pytest.raises(RuntimeError, match="managed runtime preflight failed"),
        ):
            _run_cross_task([extra_task], run_dir, **_RUNTIME_KW)

        if "return_value" in effect:
            prepare_runtime.assert_called_once_with(
                extra_task, containerize=True, container_image=IMAGE, container_network="bridge",
                evaluator_tree_sha256=managed_evaluator_tree_digest(extra_task),
            )
        retrofit.assert_not_called()
        assert not (run_dir / "extra_tasks").exists(), name

    # All cross-task runtimes are checked before agents run.
    root = tmp_path / "all_checked"
    run_dir = _run_dir(root)
    extra_tasks = _bare_tasks(root, "extra-one", "extra-two")
    with (
        patch(_PREPARE_RUNTIME, side_effect=[True, False]) as prepare_runtime,
        patch("src.variants_pipeline.retrofit.retrofit_program") as retrofit,
        pytest.raises(RuntimeError, match="extra-two.*managed runtime preflight failed"),
    ):
        _run_cross_task(extra_tasks, run_dir, **_RUNTIME_KW)

    assert prepare_runtime.call_count == 2
    retrofit.assert_not_called()
    assert not (run_dir / "extra_tasks").exists()

    # Evaluators remain sealed across all preflights.
    root = tmp_path / "sealed"
    run_dir = _run_dir(root)
    first_task, second_task = _bare_tasks(root, "extra-one", "extra-two")
    prepare_count = 0

    def prepare_runtime_mutating(*_args: object, **_kwargs: object) -> bool:
        nonlocal prepare_count
        prepare_count += 1
        if prepare_count == 2:
            (first_task / "capture.py").write_text("CHANGED = True\n", encoding="utf-8")
        return True

    with (
        patch(_PREPARE_RUNTIME, side_effect=prepare_runtime_mutating),
        patch("src.variants_pipeline.retrofit.retrofit_program") as retrofit,
        pytest.raises(RuntimeError, match="extra-one.*evaluator changed during preflight"),
    ):
        _run_cross_task([first_task, second_task], run_dir, **_RUNTIME_KW)

    retrofit.assert_not_called()
    assert not (run_dir / "extra_tasks").exists()

    # The evaluator is rechecked before each adaptation.
    root = tmp_path / "rechecked"
    run_dir = _run_dir(root)
    first_task, second_task = _bare_tasks(root, "extra-one", "extra-two")
    retrofit_checks = 0

    def retrofit_complete(*_args: object, **_kwargs: object) -> bool:
        nonlocal retrofit_checks
        retrofit_checks += 1
        if retrofit_checks == 1:
            (second_task / "capture.py").write_text("CHANGED = True\n", encoding="utf-8")
        return True

    with (
        patch(
            "src.variants_pipeline.retrofit.retrofit_bundle_complete",
            side_effect=retrofit_complete,
        ),
        pytest.raises(RuntimeError, match="extra-two.*evaluator changed before adaptation"),
    ):
        _run_cross_task([first_task, second_task], run_dir, **_RUNTIME_KW)

    assert not (run_dir / "extra_tasks" / "extra-two").exists()


# ── _find_best_program_dir ────────────────────────────────────────


def test_find_best_program_dir_picks_highest_evaluated_and_honours_manifest(tmp_path) -> None:
    # The highest evaluated program wins; unevaluated bundles yield None.
    for name, programs, expected in (
        ("highest_evaluated", {0: True, 1: False, 2: True}, "program_2"),
        ("no_evaluations", {0: False, 1: False}, None),
        ("program_0_only", {0: True}, "program_0"),
    ):
        task_bundle = tmp_path / name / "task_bundle"
        for idx, with_metrics in programs.items():
            _make_program(task_bundle, idx, with_metrics=with_metrics)
        result = _find_best_program_dir(task_bundle)
        assert (result.name if result is not None else None) == expected, name

    # A missing directory yields None.
    assert _find_best_program_dir(tmp_path / "nonexistent") is None

    # The evaluation manifest must be complete.
    for name, manifest, accepted in (
        ("incomplete_manifest", "{}", False),
        (
            "completed_host_manifest",
            json.dumps({
                "producer": "sandbox_host_eval", "execution_boundary": "sandbox",
                "program_id": "program_1", "command": ["docker", "run"],
                "started_at": "2026-08-15T00:00:00Z", "finished_at": "2026-08-15T00:01:00Z",
                "returncode": 0,
            }),
            True,
        ),
    ):
        task_bundle = tmp_path / name / "task_bundle"
        program = _make_program(task_bundle, 1, with_metrics=True)
        (program / "evaluation_manifest.json").write_text(manifest)
        assert _find_best_program_dir(task_bundle) == (program if accepted else None), name


# ── task.md writers ───────────────────────────────────────────────


def test_cross_task_md_writers_for_control_and_variants(tmp_path: Path) -> None:
    # Control task.md.
    task_bundle = tmp_path / "control" / "task_bundle"
    task_bundle.mkdir(parents=True)
    _write_control_cross_task_md(task_bundle, "mnist_task", max_evals=2)
    content = (task_bundle / "task.md").read_text()
    for expected in ("control_best", "mnist_task", "source_task.md", "original_task.md", "program_1"):
        assert expected in content

    # Variant task.md with and without an adapted control.
    for has_adapted_control in (True, False):
        root = tmp_path / f"variant_{has_adapted_control}"
        task_bundle = root / "task_bundle"
        task_bundle.mkdir(parents=True)
        spec = _make_spec("ablation_foo", root)
        _write_variant_cross_task_md(
            task_bundle, spec, "mnist_task", has_adapted_control=has_adapted_control, max_evals=2,
        )
        content = (task_bundle / "task.md").read_text()
        assert "ablation_foo" in content, has_adapted_control
        assert ("original_variant_program" in content) is has_adapted_control
        if has_adapted_control:
            assert "mnist_task" in content
            assert "port" in content.lower()
            assert "Test hypothesis" in content


# ── workspace preparation ─────────────────────────────────────────


def test_prepare_control_workspace_structure_and_skips_unevaluated(tmp_path: Path) -> None:
    # Structure of a prepared control workspace.
    root = tmp_path / "structure"
    variants_run_dir = root / "run"
    ctrl_dir = variants_run_dir / "variants" / "control_best"
    _make_program(ctrl_dir / "task_bundle", 0, with_metrics=True)
    _make_program(ctrl_dir / "task_bundle", 1, with_metrics=True)
    extra_task = _make_extra_task(root, "mnist_task")

    specs = _prepare_control_workspace(
        control_spec=_make_spec("control_best", ctrl_dir, kind="control"),
        extra_task_path=extra_task, retrofitted_extra=extra_task,
        cross_task_dir=variants_run_dir / "extra_tasks" / "mnist_task", **_WORKSPACE_KW,
    )

    assert len(specs) == 1
    ct = specs[0]
    tb = ct.variant_dir / "task_bundle"
    assert (tb / "evaluate.py").read_text() == "# evaluate\n"
    assert (tb / "program_0" / "src" / "program.py").is_file()
    assert (tb / "source_task.md").read_text() == "# Original\n"
    assert "mnist_task" in (tb / "original_task.md").read_text()
    assert "control_best" in (tb / "task.md").read_text()
    assert (ct.variant_dir / "variant_spec.json").is_file()

    # No evaluated program means no control workspace.
    root = tmp_path / "unevaluated"
    ctrl_dir = root / "variants" / "control_best"
    _make_program(ctrl_dir / "task_bundle", 0, with_metrics=False)
    extra_task = _make_extra_task(root, "mnist_task")

    specs = _prepare_control_workspace(
        control_spec=_make_spec("control_best", ctrl_dir, kind="control"),
        extra_task_path=extra_task, retrofitted_extra=extra_task,
        cross_task_dir=root / "extra_tasks" / "mnist_task", **_WORKSPACE_KW,
    )
    assert specs == []


def _prepare_variants(tmp_path: Path, spec: VariantSpec, adapted_control: Path):
    extra_task = _make_extra_task(tmp_path, "mnist_task")
    return _prepare_variant_workspaces(
        variant_specs=[spec], extra_task_path=extra_task, retrofitted_extra=extra_task,
        cross_task_dir=tmp_path / "extra_tasks" / "mnist_task",
        adapted_control_dir=adapted_control, **_WORKSPACE_KW,
    )


def test_prepare_variant_workspaces_uses_adapted_control_and_rejects_invalid_inputs(tmp_path):
    # The variant workspace uses the adapted control as program_0 and keeps the variant code.
    root = tmp_path / "adapted"
    v_dir = root / "variants" / "ablation_foo"
    _make_program(v_dir / "task_bundle", 0, with_metrics=True)
    prog1 = _make_program(v_dir / "task_bundle", 1, with_metrics=True)
    (prog1 / "src" / "program.py").write_text("# variant code\n", encoding="utf-8")
    adapted_ctrl = root / "adapted_control"
    (adapted_ctrl / "src").mkdir(parents=True)
    (adapted_ctrl / "src" / "program.py").write_text("# adapted control\n", encoding="utf-8")
    (adapted_ctrl / "program_spec.toml").write_text(_PROGRAM_SPEC, encoding="utf-8")

    result = _prepare_variants(root, _make_spec("ablation_foo", v_dir), adapted_ctrl)

    assert len(result) == 1
    tb = result[0].variant_dir / "task_bundle"
    assert (tb / "program_0" / "src" / "program.py").read_text() == "# adapted control\n"
    assert (tb / "original_variant_program/src/program.py").read_text() == "# variant code\n"
    task_md = (tb / "task.md").read_text()
    assert "original_variant_program" in task_md
    assert "port" in task_md.lower()

    # Unevaluated variants are skipped.
    root = tmp_path / "unevaluated"
    v_dir = root / "variants" / "ablation_bad"
    _make_program(v_dir / "task_bundle", 0, with_metrics=False)
    adapted_control = root / "adapted-control"
    adapted_control.mkdir()
    assert _prepare_variants(root, _make_spec("ablation_bad", v_dir), adapted_control) == []

    # A missing adapted control is rejected.
    root = tmp_path / "missing"
    v_dir = root / "variants" / "ablation_foo"
    prog = _make_program(v_dir / "task_bundle", 1, with_metrics=True)
    (prog / "src" / "program.py").write_text("# variant only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="adapted control directory is missing"):
        _prepare_variants(root, _make_spec("ablation_foo", v_dir), root / "missing-control")


# ── brokered retrofit ─────────────────────────────────────────────


def _brokered_bundle(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    bundle = root / "bundle"
    _write_scaffold(bundle)
    (bundle / "capture.py").write_text("VALUE = 1\n")
    make_program_dir(bundle / "program_0", content="def handle(request): return request\n")
    return bundle


def _retrofit(bundle: Path, output: Path, **kwargs):
    return retrofit_program(
        bundle_dir=bundle, output_bundle_dir=output, agent="codex", agent_model=None,
        timeout_seconds=20, timeout_grace_seconds=1, container_execution_env=lambda **_: {},
        **kwargs,
    )


def _no_agent():
    return patch("src.investigation.script_runners._execute_script_command")


def _legacy_marker(bundle: Path) -> str:
    return json.dumps({
        "schema_version": 1,
        "evaluator_sha256": hashlib.sha256((bundle / "evaluate.py").read_bytes()).hexdigest(),
        "source_tree_sha256": evaluator_tree_digest(bundle),
    })


def test_brokered_retrofit_preserves_program_and_rejects_runtime_or_legacy_markers(tmp_path):
    # The program is preserved without invoking an agent.
    root = tmp_path / "preserve"
    bundle = _brokered_bundle(root)
    output = root / "output"
    original_digest = evaluator_tree_digest(bundle / "program_0")

    with _no_agent() as execute:
        report = _retrofit(bundle, output)

    execute.assert_not_called()
    assert report["exit_code"] == 0
    assert report["mode"] == "brokered_preserve"
    assert evaluator_tree_digest(output / "program_0") == original_digest
    assert retrofit_bundle_complete(output, bundle)

    # A runtime added to the sanitized copy invalidates completion.
    root = tmp_path / "runtime"
    task = _brokered_bundle(root)
    output = root / "output"
    assert _retrofit(task, output)["exit_code"] == 0
    (output / ".venv").mkdir()
    assert not retrofit_bundle_complete(output, task)

    # A legacy completion marker is rejected.
    root = tmp_path / "legacy"
    bundle = _brokered_bundle(root)
    output = root / "output"
    shutil.copytree(bundle, output)
    (output / ".retrofit_complete.json").write_text(_legacy_marker(bundle))
    assert not retrofit_bundle_complete(output, bundle)


def test_brokered_retrofit_with_external_contract_validates_source_and_omits_managed_venv(
    tmp_path: Path,
) -> None:
    # The validated external contract is used and tracked.
    root = tmp_path / "external"
    task = _brokered_bundle(root)
    bundle = root / "extracted"
    bundle.mkdir()
    shutil.copytree(task / "program_0", bundle / "program_0")
    shutil.copy2(task / "evaluate.py", bundle / "evaluate.py")
    output = root / "output"

    with _no_agent() as execute:
        report = _retrofit(bundle, output, task_source_dir=task)

    execute.assert_not_called()
    assert report["exit_code"] == 0
    assert retrofit_bundle_complete(output, bundle, task_source_dir=task)
    (task / "evaluate_final.py").write_text("CHANGED = True\n")
    assert not retrofit_bundle_complete(output, bundle, task_source_dir=task)

    # The managed root venv is omitted while the candidate venv is preserved.
    root = tmp_path / "venv"
    task = _brokered_bundle(root)
    managed = task / ".venv"
    (managed / "lib").mkdir(parents=True)
    (managed / "lib64").symlink_to("lib")
    with (managed / "large-runtime").open("wb") as handle:
        handle.truncate(513 * 1024 * 1024)
    weights = task / "program_0" / ".venv" / "weights.json"
    weights.parent.mkdir()
    weights.write_text('{"weights":[1]}\n')
    output = root / "output"

    with _no_agent() as execute:
        report = _retrofit(task, output, task_source_dir=task)

    execute.assert_not_called()
    assert report["exit_code"] == 0, report
    assert not (output / ".venv").exists()
    assert (managed / "lib64").is_symlink()
    assert (output / "program_0" / ".venv" / "weights.json").read_bytes() == weights.read_bytes()
    assert retrofit_bundle_complete(output, task, task_source_dir=task)

    # Resume rejects a legacy marker with an external contract.
    root = tmp_path / "resume"
    task = _brokered_bundle(root)
    output = root / "output"
    extracted = output / "_extracted_bundle"
    extracted.mkdir(parents=True)
    shutil.copytree(task / "program_0", extracted / "program_0")
    shutil.copy2(task / "evaluate.py", extracted / "evaluate.py")
    retrofitted = output / "_retrofitted_bundle"
    shutil.copytree(extracted, retrofitted)
    (retrofitted / ".retrofit_complete.json").write_text(_legacy_marker(extracted))
    results = root / "results"
    results.mkdir()
    cfg = VariantsPipelineConfig()
    cfg.paths.task_dir = str(task)
    cfg.paths.results_dir = str(results)
    cfg.paths.output_dir = str(output)
    cfg.container.enabled = True

    with patch("src.variants_pipeline.pipeline.materialize_variant_workspaces") as materialize:
        assert run_variants_pipeline(cfg, stages={4}) == 1

    materialize.assert_not_called()


def test_brokered_retrofit_marker_rejects_artifact_changes_hidden_dirs_and_unsafe_entries(
    tmp_path: Path,
) -> None:
    # Any change to a hashed artifact invalidates the marker.
    for changed in (
        "program", "source_program", "evaluator", "source_helper", "copied_task", "copied_helper",
        "copied_final",
    ):
        root = tmp_path / "changed" / changed
        bundle = _brokered_bundle(root)
        output = root / "output"
        with patch(
            "src.investigation.script_runners._execute_script_command", return_value=("", 0),
        ):
            report = _retrofit(bundle, output)
        assert report["exit_code"] == 0, changed
        target = {
            "program": output / "program_0" / "src" / "program.py",
            "source_program": bundle / "program_0" / "src" / "program.py",
            "evaluator": output / "evaluate.py",
            "source_helper": bundle / "capture.py",
            "copied_task": output / "task.md",
            "copied_helper": output / "capture.py",
            "copied_final": output / "evaluate_final.py",
        }[changed]
        target.write_text("CHANGED = True\n")

        assert not retrofit_bundle_complete(output, bundle), changed

    # All program directories are hashed, including hidden and scratch ones.
    for directory in ("results", ".venv", ".git", "__pycache__"):
        for changed_tree in ("source", "copy"):
            root = tmp_path / "hashed" / directory / changed_tree
            bundle = _brokered_bundle(root)
            weights = bundle / "program_0" / directory / "weights.json"
            weights.parent.mkdir()
            weights.write_text('{"weights":[1]}\n')
            output = root / "output"
            assert _retrofit(bundle, output)["exit_code"] == 0, (directory, changed_tree)
            assert retrofit_bundle_complete(output, bundle), (directory, changed_tree)

            changed_root = bundle if changed_tree == "source" else output
            (changed_root / "program_0" / directory / "weights.json").write_text(
                '{"weights":[2]}\n'
            )

            assert not retrofit_bundle_complete(output, bundle), (directory, changed_tree)

    # Unsafe hidden copy entries are rejected.
    for entry_kind in ("symlink", "fifo"):
        root = tmp_path / "unsafe" / entry_kind
        bundle = _brokered_bundle(root)
        (bundle / "program_0" / "results").mkdir()
        output = root / "output"
        assert _retrofit(bundle, output)["exit_code"] == 0, entry_kind
        unsafe = output / "program_0" / "results" / "unsafe"
        if entry_kind == "symlink":
            unsafe.symlink_to(root / "outside")
        else:
            os.mkfifo(unsafe)

        assert not retrofit_bundle_complete(output, bundle), entry_kind


def test_retrofit_rejects_invalid_contract_keeps_legacy_agent_path_and_prompt_semantics(
    tmp_path: Path,
) -> None:
    # An invalid contract is rejected before the agent runs.
    root = tmp_path / "invalid"
    bundle = _brokered_bundle(root)
    (bundle / "task.md").write_text("Changed canonical task\n")
    with _no_agent() as execute:
        report = _retrofit(bundle, root / "output")

    execute.assert_not_called()
    assert report["exit_code"] == 1
    assert "canonical task digest mismatch" in report["error"]

    # A legacy bundle still invokes the agent.
    bundle = tmp_path / "legacy"
    bundle.mkdir()
    (bundle / "evaluate.py").write_text("# legacy evaluator\n")
    with patch(
        "src.investigation.script_runners._execute_script_command", return_value=("", 0)
    ) as execute:
        report = _retrofit(bundle, tmp_path / "legacy_output")

    execute.assert_called_once()
    assert report["exit_code"] == 0

    # The retrofit prompt preserves experiment and error semantics.
    prompt = Path("src/prompts/variant_retrofit.md").read_text()
    assert "Do not change stopping conditions, seeds, or compute budgets" in prompt
    assert "Do not swallow program, evaluator, or CSV-writing errors" in prompt
    assert "Remove search-phase shortcuts" not in prompt
