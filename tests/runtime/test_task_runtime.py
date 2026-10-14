"""Task runtime preparation: contracted baselines, docker commands, archives, build leases, sealed mounts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import src.investigation.task_runtime as runtime
from src.investigation import task_runtime_archive as archive
from src.investigation.bounded_process import BoundedProcessResult, ProcessOutputLimitError
from src.investigation.container_gpus import GpuLeaseTimeout
from src.investigation.evaluator_tree import managed_evaluator_tree_digest
from src.investigation.harness_incumbent import runtime_baseline_score
from src.investigation.main_entry import _configure_contracted_agent_isolation
from src.investigation.reset_sequence import (
    declared_reset_sequence_matches,
    reduced_draft_reset_sequence_matches,
)
from src.investigation.runtime_build_coordination import (
    RUNTIME_BUILD_LEASE_ROOT,
    RuntimeBuildLeaseError,
    RuntimeBuildLeaseTimeout,
    acquire_runtime_build_lease,
)
from src.investigation.task_contract import load_task_contract
from src.investigation.task_runtime import (
    TaskRuntimeEvaluationRequirement,
    TaskRuntimePreparationReport,
    TaskRuntimePreparationResult,
    _docker_command,
    _draft_validation_reset_sequence_matches,
    _remove_stale_runtime_builds,
    _requirements_match_contract,
    discard_managed_task_runtime,
    prepare_task_runtime,
    prepare_task_runtime_report,
    prepare_task_runtime_result,
    runtime_preparation_failure_is_repairable,
    runtime_preparation_report_from_result,
)
from src.investigation.task_runtime_cache import (
    TaskRuntimeInstallCache,
    TaskRuntimeInstallCacheDeadlineExceeded,
)
from src.variants_pipeline.host_eval import _append_search_seals, evaluate_variant_program
from tests.program_testutils import make_program_dir
from tests.final_evaluation.test_final_evaluation import _RecordingGpuLease, _write_contract_task
from tests.task_drafting.test_task_contract import _write_scaffold

Status = runtime._CheckedCommandStatus
PrepResult = TaskRuntimePreparationResult
IMAGE_ID = "sha256:" + "a" * 64
_DOCKER_INSPECT = ["docker", "image", "inspect", "--format={{.Id}}"]
_GPU_ENV = {"PATH": "/bin", "AGENT_CONTAINER_GPUS": "device=GPU-aaaa"}
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_AGENT = REPO_ROOT / "src/scripts/lib/container_agent.sh"
CONCEPT_FIGURE = REPO_ROOT / "src/scripts/lib/concept_figure.sh"


# --- shared helpers -------------------------------------------------------------------------


def _completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _contract(task_dir: Path):
    return load_task_contract(task_dir / "task_contract.json")


def _rewrite_contract(task_dir: Path, mutate):
    path = task_dir / "task_contract.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_task_contract(path)


def _gpu_contract(task_dir: Path, *, gpu_count=1, evaluator_gpu_count=1, candidate_gpu_count=0):
    return _rewrite_contract(
        task_dir,
        lambda payload: payload["compute"]["search"].update(
            accelerator="gpu",
            accelerator_model="H100",
            gpu_count=gpu_count,
            evaluator_gpu_count=evaluator_gpu_count,
            candidate_gpu_count=candidate_gpu_count,
        ),
    )


def _baseline_task(root: Path, *, requirements: str | None = None) -> Path:
    root.mkdir(exist_ok=True)
    make_program_dir(root / "program_0", content="print('baseline')\n")
    (root / "evaluate.py").write_text("", encoding="utf-8")
    if requirements is not None:
        (root / "requirements.txt").write_text(requirements, encoding="utf-8")
    return root


@pytest.fixture
def baseline_task(tmp_path: Path) -> Path:
    return _baseline_task(tmp_path)


def _install_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "install-cache"
    cache.mkdir(mode=0o700)
    return cache


def _mounted_path(command: list[str], destination: str) -> Path:
    """Host path bound at ``destination`` (``host:/dst:rw`` or ``type=bind,src=host,dst=/dst``)."""
    for value in command:
        if value.endswith(f":{destination}:rw"):
            return Path(value.removesuffix(f":{destination}:rw"))
        if value.startswith("type=bind,src=") and value.endswith(f",dst={destination}"):
            return Path(value.removeprefix("type=bind,src=").removesuffix(f",dst={destination}"))
    raise AssertionError(f"no {destination} mount in {command}")


def _create_venv(command: list[str], *, site_packages: bool = False) -> None:
    venv = _mounted_path(command, "/runtime") / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text("", encoding="utf-8")
    if site_packages:
        (venv / "lib/python3.12/site-packages/src/investigation").mkdir(parents=True)


def _write_results(results: Path, *, score=1.0, correct=True, feasible=None) -> None:
    (results / "metrics.json").write_text(json.dumps({"combined_score": score}), encoding="utf-8")
    (results / "correct.json").write_text(json.dumps({"correct": correct}), encoding="utf-8")
    if feasible is not None:
        (results / "qualification.json").write_text(
            json.dumps({"schema_version": 1, "feasible": feasible}), encoding="utf-8"
        )


def _counts() -> dict[str, int]:
    return {"evaluations": 0, "venv_creations": 0}


def _docker_fake_run(
    *,
    image_id=IMAGE_ID,
    score=1.0,
    correct=True,
    feasible=None,
    site_packages=False,
    commands=None,
    counts=None,
    hook=None,
):
    """Simulate docker: resolve the image, create the venv on demand, write baseline results.

    ``hook(command, kwargs)`` runs first for every command; a non-None reply short-circuits.
    """

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if commands is not None:
            commands.append(command)
        if hook is not None and (reply := hook(command, kwargs)) is not None:
            return reply
        if command[:4] == _DOCKER_INSPECT:
            return _completed(command, stdout=image_id + "\n")
        if "venv" in command:
            if counts is not None:
                counts["venv_creations"] += 1
            _create_venv(command, site_packages=site_packages)
        if "/task/evaluate.py" in command:
            if counts is not None:
                counts["evaluations"] += 1
            _write_results(
                _mounted_path(command, "/results"), score=score, correct=correct, feasible=feasible
            )
        return _completed(command)

    return fake_run


def _fake_broker(
    *, calls=2, calls_before_first_reset=0, reset_seeds=(1,), peer_disconnects=0, on_init=None, on_exit=None
):
    class FakeBroker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            if on_init is not None:
                on_init(*args, **kwargs)

        def __enter__(self) -> "FakeBroker":
            return self

        def __exit__(self, *_args: object) -> None:
            if on_exit is not None:
                on_exit()

    FakeBroker.calls = calls
    FakeBroker.calls_before_first_reset = calls_before_first_reset
    FakeBroker.reset_seeds = reset_seeds
    FakeBroker.peer_disconnects = peer_disconnects
    return FakeBroker


def _stub_commands(monkeypatch: pytest.MonkeyPatch, broker=None) -> None:
    monkeypatch.setattr(runtime, "_evaluation_command", lambda **_kwargs: ["eval"])
    monkeypatch.setattr(runtime, "_candidate_command", lambda **_kwargs: ["candidate"])
    if broker is not None:
        monkeypatch.setattr(runtime, "CandidateBroker", broker)


def _stub_validation(monkeypatch: pytest.MonkeyPatch, *, score=1.0, correct=True) -> None:
    monkeypatch.setattr(runtime, "_validated_contracted_baseline_score", lambda *_a, **_k: score)
    monkeypatch.setattr(runtime, "_validated_evaluation_correctness", lambda *_a: correct)


def _run_baseline(task_dir: Path, contract, run, *, budget=60, environment=None, **kwargs):
    return runtime._run_contracted_baseline(
        task_dir=task_dir,
        runtime_venv=task_dir / ".venv",
        image_id=IMAGE_ID,
        contract=contract,
        entrypoint="main.py",
        deadline=time.monotonic() + budget,
        environment={"PATH": "/bin"} if environment is None else environment,
        run=run,
        **kwargs,
    )


def _run_checked(task_dir: Path, run, *, command=("docker", "run", "image"), deadline=None, **kwargs):
    return runtime._run_checked(
        list(command),
        task_dir=task_dir,
        deadline=time.monotonic() + 30 if deadline is None else deadline,
        run=run,
        environment={"PATH": "/bin"},
        **kwargs,
    )


def _succeed(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
    return _completed(command)


def _timeout_run(command: list[str], *, timeout: float, **_kwargs: object):
    assert isinstance(timeout, (int, float))
    raise subprocess.TimeoutExpired(command, timeout)


def _contract_results(tmp_path: Path, contract, *, correct=True, feasible=None, extra=None) -> Path:
    results = tmp_path / "results"
    results.mkdir(parents=True)
    metrics = {metric.name: 0.5 for metric in contract.metrics.search}
    (results / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (results / "correct.json").write_text(json.dumps({"correct": correct, **(extra or {})}), encoding="utf-8")
    if feasible is not None:
        (results / "qualification.json").write_text(
            json.dumps({"schema_version": 1, "feasible": feasible}), encoding="utf-8"
        )
    return results


def _broker_recovery_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    recovery_root = tmp_path / "broker-recovery"
    recovery_root.mkdir(mode=0o700)
    monkeypatch.setattr(runtime, "_broker_recovery_root", lambda: recovery_root)
    monkeypatch.setattr(runtime, "_ensure_broker_recovery_root", lambda: recovery_root)
    return recovery_root


def _docker_command_for(task_dir: Path, **overrides: Any) -> list[str]:
    options: dict[str, Any] = {
        "task_dir": task_dir,
        "image_id": IMAGE_ID,
        "network": "none",
        "inner_command": ["python", "evaluate.py"],
        "cid_path": task_dir / "container.cid",
    }
    return _docker_command(**(options | overrides))


class _QuarantineLease(_RecordingGpuLease):
    active = False
    quarantined = False

    def __enter__(self) -> "_QuarantineLease":
        self.active = True
        return self

    def __exit__(self, *_args: object) -> None:
        self.active = False

    def quarantine(self) -> None:
        assert self.active
        self.quarantined = True


def _container_agent_shell(command: str, **environment: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("HEADLESS_CAPACITY_RETRIES", None)
    env.pop("AGENT_CHATGPT_SUBSCRIPTION", None)
    return subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(CONTAINER_AGENT))}\n{command}"],
        env=env | environment, capture_output=True, text=True, timeout=10,
    )


# --- contracted baseline classification -----------------------------------------------------


def test_docker_commands_expose_opt_in_markers_mask_snapshot_and_fit_socket_limit(tmp_path: Path) -> None:
    # Former: task broker socket path fits the Linux unix-socket limit for both phases.
    for phase in ("runtime-search", "final-smoke"):
        broker_dir = runtime._task_generation_broker_directory(tmp_path, phase)
        assert len(str(broker_dir / "candidate.sock").encode()) <= 107, phase

    # Former: draft smoke marker is explicit and opt-in; the snapshot metadata is always masked.
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / ".agentic_task_snapshot.json").write_text("{}\n", encoding="utf-8")
    production = _docker_command_for(task_dir)
    drafting = _docker_command_for(task_dir, draft_smoke=True)

    assert "AGENTIC_TASK_DRAFT_SMOKE=1" not in production
    assert "AGENTIC_TASK_DRAFT_SMOKE=1" in drafting
    snapshot_mask = "type=bind,src=/dev/null,dst=/task/.agentic_task_snapshot.json,readonly"
    assert snapshot_mask in production
    assert snapshot_mask in drafting

    # Former: qualification marker is explicit and opt-in.
    (task_dir / "evaluate.py").write_text("", encoding="utf-8")
    make_program_dir(task_dir / "program_0")
    for name in ("results", "broker"):
        (task_dir / name).mkdir()
    evaluation_command = partial(
        runtime._evaluation_command,
        task_dir=task_dir,
        program_dir=task_dir / "program_0",
        entrypoint=task_dir / "program_0" / "main.py",
        evaluator=task_dir / "evaluate.py",
        results_dir=task_dir / "results",
        python_bin=Path("/runtime/.venv/bin/python"),
        container_image=IMAGE_ID,
        gpu_count=0,
        environment={},
        broker_dir=task_dir / "broker",
    )

    assert "AGENTIC_TASK_QUALIFICATION=1" not in evaluation_command()
    assert "AGENTIC_TASK_QUALIFICATION=1" in evaluation_command(qualification=True)

    # Former: task container rejects non-regular snapshot metadata (symlink, directory).
    for entry_type in ("symlink", "directory"):
        unsafe_dir = tmp_path / entry_type
        unsafe_dir.mkdir()
        snapshot_metadata = unsafe_dir / ".agentic_task_snapshot.json"
        if entry_type == "symlink":
            snapshot_metadata.symlink_to(unsafe_dir / "outside")
        else:
            snapshot_metadata.mkdir()
        with pytest.raises(ValueError, match="snapshot metadata"):
            _docker_command_for(unsafe_dir)


_RESET_SEQUENCE_CASES = [
    (f"{name}-{'_'.join(map(str, observed)) or 'empty'}", matcher, observed, declared, matches)
    for name, matcher, declared, cases in (
        (
            "draft_validation",
            _draft_validation_reset_sequence_matches,
            (4, 4, 4),
            [((4, 4, 4), True), ((4, 4, 4, 4), True), ((4,), False), ((4, 4), False),
             ((4, 2), False), ((4, 2, 6), False), ((2,), False)],
        ),
        (
            "declared",
            declared_reset_sequence_matches,
            (4, 2, 6),
            [((4, 2, 6), True), ((4, 2, 6, 4), True), ((4, 2, 6, 2), False), ((4, 2, 6, 4, 4), False),
             ((4, 2), False), ((2, 4, 6), False), ((), False)],
        ),
        (
            "declared_repeated",
            declared_reset_sequence_matches,
            (4, 4, 2, 2, 6, 6),
            [((4, 4, 2, 2, 6, 6), True), ((4, 4, 2, 2, 6, 6, 4), True),
             ((4, 4, 2, 2, 6, 6, 4, 4), False), ((4, 2, 4, 2, 6, 6), False)],
        ),
        (
            "reduced_draft",
            reduced_draft_reset_sequence_matches,
            (4, 2, 6),
            [((4,), True), ((4, 4), True), ((4, 2), False), ((4, 4, 4), False), ((2,), False), ((), False)],
        ),
    )
    for observed, matches in cases
]


_CANDIDATE_CALL_DIAGNOSTIC_CASES = [
    # (calls, calls_before_reset, observed, peer_disconnects, expected)
    (2, 0, (4,), 0, None),
    (0, 0, (), 0, "made no candidate calls"),
    (2, 1, (4,), 0, "made 1 call before the first reset"),
    (2, 0, (4,), 1, "disconnected 1 time before receiving a response"),
    (2, 0, (2,), 0, "expected primary [4]"),
]


def test_reset_sequence_matchers_and_candidate_call_diagnostic_identify_the_actual_violation() -> None:
    # Former: reset sequence matchers allow exactly one deterministic replay.
    for name, matcher, observed, declared, matches in _RESET_SEQUENCE_CASES:
        assert matcher(observed, declared) is matches, name

    # Former: candidate call sequence diagnostic identifies the actual violation.
    for calls, calls_before_reset, observed, peer_disconnects, expected in _CANDIDATE_CALL_DIAGNOSTIC_CASES:
        diagnostic = runtime._candidate_call_sequence_diagnostic(
            calls=calls,
            calls_before_first_reset=calls_before_reset,
            observed_reset_seeds=observed,
            expected_primary_reset_seeds=(4,),
            peer_disconnects=peer_disconnects,
            context="Draft search smoke",
        )
        if expected is None:
            assert diagnostic is None, (calls, calls_before_reset, observed, peer_disconnects)
        else:
            assert diagnostic is not None and expected in diagnostic, expected


# --- preparation reports --------------------------------------------------------------------


def _stub_preparation(monkeypatch, result, *, log=None, extra=None):
    def fail_runtime(*_args: object, **_kwargs: object) -> TaskRuntimePreparationResult:
        if log is not None:
            runtime.logger.error(log, extra=extra)
        return result

    monkeypatch.setattr(runtime, "prepare_task_runtime_result", fail_runtime)


def test_runtime_report_sanitizes_bounds_and_classifies_preparation_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        # Former: runtime report retains the typed, sanitized failure.
        (tmp_path / "evaluate.py").write_text("", encoding="utf-8")
        _stub_preparation(m, PrepResult.DEPENDENCY_FAILURE, log="pip rejected api_key=sk-proj-abcdefghijklmnop")

        report = prepare_task_runtime_report(tmp_path)

        assert isinstance(report, TaskRuntimePreparationReport)
        assert report.result is PrepResult.DEPENDENCY_FAILURE
        assert report.failure_stage == "dependency_install"
        assert report.failure_category == "runtime_dependencies"
        assert "sk-proj" not in report.diagnostic
        assert report.task_fingerprint.startswith("sha256:")
        assert report.elapsed_seconds >= 0

        # Former: runtime report bounds the diagnostic tail.
        _stub_preparation(m, PrepResult.INFRASTRUCTURE_FAILURE, log="x" * 20_000)

        report = prepare_task_runtime_report(tmp_path)

        assert len(report.diagnostic.encode("utf-8")) <= 8 * 1024
        assert report.failure_stage == "runtime_preparation"
        assert report.failure_category == "runtime_bootstrap"

        # Former: runtime report ignores untyped log evidence.
        _stub_preparation(
            m,
            PrepResult.INFRASTRUCTURE_FAILURE,
            log="runtime failed",
            extra={"task_runtime_failure_evidence": {"evaluator_output": "private"}},
        )

        assert prepare_task_runtime_report(tmp_path).failure_evidence is None

    # Former: runtime report classifies the task-authored search deadline.
    def exceed_contract(*_args: object, **_kwargs: object) -> TaskRuntimePreparationResult:
        raise runtime._TaskRuntimeContractDeadlineExceeded(
            "Production search exceeded the task-authored 7200-second wall-clock ceiling; "
            "calibrate or reduce the production workload.",
            evidence=runtime.TaskRuntimeFailureEvidence(
                deadline_owner="task_contract",
                contract_wall_clock_seconds=7200,
                evaluator_timeout_seconds=7179,
                candidate_rpc_timeout_seconds=7179,
                broker_teardown_reserve_seconds=21.0,
                container_cleanup_reserve_seconds=30.0,
                candidate_rpc_calls=12,
                candidate_calls_before_first_reset=0,
                candidate_reset_count=1,
                candidate_peer_disconnects=0,
            ),
        )

    monkeypatch.setattr(runtime, "_prepare_task_runtime", exceed_contract)

    report = prepare_task_runtime_report(tmp_path, timeout_seconds=8000)

    assert report.result is PrepResult.AUTHORED_FAILURE
    assert report.failure_stage == "baseline_validation"
    assert report.failure_category == "runtime_baseline_validation"
    assert "7200-second wall-clock ceiling" in report.diagnostic
    assert "calibrate or reduce" in report.diagnostic
    assert report.failure_evidence is not None
    assert report.failure_evidence.deadline_owner == "task_contract"
    assert report.failure_evidence.candidate_rpc_calls == 12


def test_runtime_report_maps_legacy_result_and_classifies_repairability(tmp_path: Path) -> None:
    # Former: runtime report maps a legacy result without losing its kind.
    for result, failure_stage, failure_category, diagnostic in (
        (PrepResult.READY, None, None, ""),
        (PrepResult.AUTHORED_FAILURE, "baseline_validation", "runtime_baseline_validation", "authored failure"),
    ):
        report = runtime_preparation_report_from_result(result, task_dir=tmp_path, diagnostic="authored failure")

        assert report.result is result, result
        assert report.failure_stage == failure_stage, result
        assert report.failure_category == failure_category, result
        assert report.diagnostic == diagnostic, result
        assert report.output_evidence is None, result

    # Former: runtime failure repairability.
    for result, expected in (
        (PrepResult.DEPENDENCY_FAILURE, True),
        (PrepResult.AUTHORED_FAILURE, True),
        (PrepResult.INFRASTRUCTURE_FAILURE, False),
    ):
        assert runtime_preparation_failure_is_repairable(result) is expected, result


# --- baseline result validation -------------------------------------------------------------


def test_contracted_baseline_validation_enforces_requirements_correctness_and_qualification(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Former: requirements must match the declared dependency provenance.
    task = _write_contract_task(tmp_path)
    contract = _contract(task)

    assert _requirements_match_contract(task / "requirements.txt", contract)

    (task / "requirements.txt").write_text("other-package==1.0.0\n", encoding="utf-8")

    assert not _requirements_match_contract(task / "requirements.txt", contract)

    # Former: contracted baseline reports extra correctness fields.
    results = _contract_results(tmp_path / "extra", contract, extra={"debug": "leak"})

    assert runtime._validated_contracted_baseline_score(results, contract) is None
    assert "correct.json must contain exactly one boolean key" in caplog.text

    # Former: drafting validation accepts a scientifically incorrect baseline.
    results = _contract_results(tmp_path / "incorrect", contract, correct=False)

    drafting_score = runtime._validated_contracted_baseline_score(
        results, contract, requirement=TaskRuntimeEvaluationRequirement.VALID_RESULT
    )

    assert drafting_score == pytest.approx(0.5)
    assert runtime._validated_contracted_baseline_score(results, contract) is None
    assert runtime._validated_evaluation_correctness(results) is False

    # Former: qualification accepts only a feasible baseline.
    for feasible, expected_score in ((True, 0.5), (False, None)):
        results = _contract_results(tmp_path / f"feasible-{feasible}", contract, correct=False, feasible=feasible)

        score = runtime._validated_contracted_baseline_score(
            results, contract, requirement=TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT
        )

        assert score == expected_score, feasible

    # Former: non-qualification rejects a qualification result.
    results = _contract_results(tmp_path / "nonqualification", contract, correct=True, feasible=True)

    assert runtime._validated_contracted_baseline_score(results, contract) is None


_QUALIFICATION_SCHEMA_CASES = [
    ('{"schema_version": 1, "feasible": true}', True),
    ('{"schema_version": 1, "feasible": false}', False),
    ('{"schema_version": 1, "feasible": 1}', None),
    ('{"schema_version": 2, "feasible": true}', None),
    ('{"schema_version": true, "feasible": true}', None),
    ('{"schema_version": 1.0, "feasible": true}', None),
    ('{"feasible": true}', None),
    ('{"schema_version": 1, "feasible": true, "detail": "leak"}', None),
    ('{"schema_version":1,"feasible":true,"feasible":true}', None),  # duplicate keys
]


def test_qualification_result_requires_exact_schema(tmp_path: Path) -> None:
    for index, (payload, expected) in enumerate(_QUALIFICATION_SCHEMA_CASES):
        results = tmp_path / f"results-{index}"
        results.mkdir()
        (results / "qualification.json").write_text(payload, encoding="utf-8")

        assert runtime._validated_evaluation_feasibility(results) is expected, payload


def test_repository_declares_managed_runtime_dependencies_and_tooling_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Former: circpack declares managed runtime dependencies.
    requirements = (REPO_ROOT / "examples/circpack/requirements.txt").read_text(encoding="utf-8")

    assert requirements.splitlines() == ["numpy==2.4.2", "scipy==1.17.1"]

    # Former: tooling package dirs follow pyproject.
    declared = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = tuple(
        Path(*name.split(".")) for name in declared["tool"]["setuptools"]["packages"] if name != "src"
    )
    assert runtime._tooling_package_dirs() == expected
    assert Path("src/investigation") in expected

    (tmp_path / "pyproject.toml").write_text("[tool.setuptools]\nname = 'x'\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "_REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="packages"):
        runtime._tooling_package_dirs()


def test_discard_managed_runtime_unlinks_symlinked_lib64_but_rejects_symlinked_root(tmp_path: Path) -> None:
    # Former: discard managed runtime unlinks the standard-library symlink.
    task_dir = tmp_path / "managed"
    venv = task_dir / ".venv"
    (venv / "lib").mkdir(parents=True)
    (venv / "lib" / "installed.py").write_text("installed\n", encoding="utf-8")
    (venv / "lib64").symlink_to("lib", target_is_directory=True)
    (task_dir / ".agentic_runtime.json").write_text("{}\n", encoding="utf-8")

    discard_managed_task_runtime(task_dir)

    assert not venv.exists()
    assert not (task_dir / ".agentic_runtime.json").exists()

    # Former: discard managed runtime rejects a symlinked root.
    task_dir = tmp_path / "symlinked"
    task_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (task_dir / ".venv").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="managed runtime directory"):
        discard_managed_task_runtime(task_dir)

    assert outside.is_dir()


# --- prepare_task_runtime end to end (fake docker) ------------------------------------------


def test_prepare_runtime_creates_image_venv_and_installs_pinned_requirements(baseline_task: Path) -> None:
    (baseline_task / "requirements.txt").write_text("yahpo-gym==1.0.2\nonnxruntime==1.23.2\n", encoding="utf-8")
    commands: list[list[str]] = []

    def resolve_and_check_tmp(command, _kwargs):
        if command[:4] == _DOCKER_INSPECT:
            requested = command[-1]
            return _completed(command, stdout=(requested if requested.startswith("sha256:") else IMAGE_ID) + "\n")
        writable = next((value for value in command if value.endswith(":/runtime:rw")), None)
        if writable is not None:
            tmp = Path(writable.removesuffix(":/runtime:rw")) / ".tmp"
            assert tmp.is_dir()
            assert tmp.stat().st_mode & 0o777 == 0o700
        return None

    fake_run = _docker_fake_run(score=0.1, commands=commands, hook=resolve_and_check_tmp)

    assert prepare_task_runtime(
        baseline_task,
        timeout_seconds=30,
        containerize=True,
        container_image="evaluator:latest",
        run=fake_run,
        base_environment={"PATH": "/bin", "DEEPSEEK_API_KEY": "secret"},
    )
    assert all(command[0] == "docker" for command in commands)
    assert any("--only-binary=:all:" in command for command in commands)
    evaluation = next(command for command in commands if "/task/evaluate.py" in command)
    assert evaluation[evaluation.index("--network") : evaluation.index("--network") + 2] == ["--network", "none"]
    writable_runtime_commands = [
        command for command in commands if any(value.endswith(":/runtime:rw") for value in command)
    ]
    assert writable_runtime_commands
    assert all("TMPDIR=/runtime/.tmp" in command for command in writable_runtime_commands)
    assert all("PIP_NO_CACHE_DIR=1" in command for command in writable_runtime_commands)
    assert "TMPDIR=/runtime/.tmp" not in evaluation
    assert "PIP_NO_CACHE_DIR=1" not in evaluation
    assert "CUDA_VISIBLE_DEVICES=" in evaluation
    assert "--gpus" not in evaluation
    assert not any("DEEPSEEK_API_KEY" in argument for command in commands for argument in command)
    assert (baseline_task / ".venv" / "bin" / "python").is_file()
    manifest = json.loads((baseline_task / ".agentic_runtime.json").read_text())
    assert manifest["baseline_combined_score"] == pytest.approx(0.1)
    assert runtime_baseline_score(baseline_task) == pytest.approx(0.1)
    assert manifest["gpu_evaluation"] is False
    assert manifest["cuda_visible_devices"] == ""
    assert manifest["evaluator_sha256"] == hashlib.sha256((baseline_task / "evaluate.py").read_bytes()).hexdigest()
    assert manifest["evaluator_tree_sha256"] == managed_evaluator_tree_digest(baseline_task)

    replacement_image = "sha256:" + "b" * 64
    assert prepare_task_runtime(
        baseline_task,
        timeout_seconds=30,
        containerize=True,
        container_image=replacement_image,
        run=fake_run,
        base_environment={"PATH": "/bin"},
    )
    assert replacement_image in (baseline_task / ".agentic_runtime.json").read_text()


def _prepare(task_dir: Path, run, **kwargs) -> TaskRuntimePreparationResult:
    return prepare_task_runtime_result(task_dir, containerize=True, container_image=IMAGE_ID, run=run, **kwargs)


def test_compatible_runtime_reuses_validated_baseline_unless_fresh_or_stricter_validation_is_requested(tmp_path: Path) -> None:
    # Former: compatible runtime reuses the validated baseline.
    task = _baseline_task(tmp_path / "reuse")
    counts = _counts()
    fake_run = _docker_fake_run(score=0.4, counts=counts)

    first = _prepare(task, fake_run)
    manifest_before = (task / ".agentic_runtime.json").read_bytes()
    second = _prepare(task, fake_run)

    assert (first, second) == (PrepResult.READY, PrepResult.READY)
    assert counts == {"evaluations": 1, "venv_creations": 1}
    assert (task / ".agentic_runtime.json").read_bytes() == manifest_before
    assert runtime_baseline_score(task) == pytest.approx(0.4)

    # Former: fresh baseline requests revalidate a compatible runtime.
    task = _baseline_task(tmp_path / "fresh")
    counts = _counts()
    fake_run = _docker_fake_run(score=0.4, counts=counts)

    assert _prepare(task, fake_run) is PrepResult.READY
    assert _prepare(task, fake_run, retained_results_dir=task.parent / "retained") is PrepResult.READY
    assert counts == {"evaluations": 2, "venv_creations": 1}

    # Former: an incorrect retained baseline is revalidated under the strict requirement.
    task = _baseline_task(tmp_path / "incorrect")
    counts = _counts()
    fake_run = _docker_fake_run(score=0.4, correct=False, counts=counts)
    relaxed_options = {"evaluation_requirement": TaskRuntimeEvaluationRequirement.VALID_RESULT}

    relaxed = _prepare(task, fake_run, **relaxed_options)
    relaxed_again = _prepare(task, fake_run, **relaxed_options)
    strict = _prepare(task, fake_run)

    assert (relaxed, relaxed_again) == (PrepResult.READY, PrepResult.READY)
    assert strict is PrepResult.AUTHORED_FAILURE
    assert counts == {"evaluations": 2, "venv_creations": 1}


def test_runtime_preparation_shares_one_decreasing_stage_budget_and_stops_when_it_is_exhausted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # Former: runtime preparation uses one decreasing stage budget.
    task = _baseline_task(tmp_path / "budget")
    real_monotonic = time.monotonic
    now = [100.0]
    stage_timeouts: list[float] = []
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])

    def record_timeout(_command, kwargs):
        assert isinstance(kwargs["timeout"], (int, float))
        stage_timeouts.append(float(kwargs["timeout"]))
        now[0] += 1.0

    assert prepare_task_runtime(
        task, timeout_seconds=30, containerize=True, container_image=IMAGE_ID,
        run=_docker_fake_run(hook=record_timeout),
    )

    assert stage_timeouts == sorted(stage_timeouts, reverse=True)
    assert stage_timeouts[0] == pytest.approx(30.0)
    assert stage_timeouts[1] == pytest.approx(26.1)
    assert all(
        left - right == pytest.approx(0.9)
        for left, right in zip(stage_timeouts[1:], stage_timeouts[2:], strict=False)
    )

    # Former: runtime preparation stops when the shared deadline is exhausted.
    task = _baseline_task(tmp_path / "exhausted")
    now[0] = 0.0
    commands: list[list[str]] = []
    stage_timeouts = []

    def advance(command, kwargs):
        assert isinstance(kwargs["timeout"], (int, float))
        stage_timeouts.append(float(kwargs["timeout"]))
        now[0] += 0.6 if command[:4] == _DOCKER_INSPECT else 0.5

    result = _prepare(task, _docker_fake_run(commands=commands, hook=advance), timeout_seconds=1)

    assert result is PrepResult.INFRASTRUCTURE_FAILURE
    assert stage_timeouts == pytest.approx([1.0, 0.36])
    assert len(commands) == 2
    assert not (task / ".venv").exists()

    # Former: runtime preparation does not publish after the final stage exhausts the deadline.
    task = _baseline_task(tmp_path / "final-stage")
    now[0] = 0.0

    def exhaust_on_evaluation(command, _kwargs):
        if "/task/evaluate.py" in command:
            now[0] = 1.0

    result = _prepare(task, _docker_fake_run(hook=exhaust_on_evaluation), timeout_seconds=1)

    assert result is PrepResult.INFRASTRUCTURE_FAILURE
    assert not (task / ".venv").exists()
    assert not (task / ".agentic_runtime.json").exists()

    # Former: a runtime stage timeout has a deterministic error.
    monkeypatch.setattr(runtime.time, "monotonic", real_monotonic)
    task = _baseline_task(tmp_path / "stage-timeout")

    def expire(command, _kwargs):
        if command[:4] != _DOCKER_INSPECT:
            raise subprocess.TimeoutExpired(command, 1)

    result = _prepare(task, _docker_fake_run(hook=expire), timeout_seconds=30)

    assert result is PrepResult.INFRASTRUCTURE_FAILURE
    assert "Task runtime preparation timed out: task runtime preparation deadline exceeded" in caplog.text


def test_prepare_runtime_gpu_baseline_preserves_cuda_visibility_and_requires_explicit_selector(tmp_path: Path) -> None:
    # Former: GPU baseline preserves CUDA visibility.
    task = _baseline_task(tmp_path / "visible")
    commands: list[list[str]] = []
    gpu_spec = '"device=GPU-one,GPU-two"'

    assert prepare_task_runtime(
        task,
        containerize=True,
        container_image=IMAGE_ID,
        gpu_evaluation=True,
        run=_docker_fake_run(commands=commands),
        base_environment={"CUDA_VISIBLE_DEVICES": "0,1", "AGENT_CONTAINER_GPUS": gpu_spec},
    )

    evaluation = next(command for command in commands if "/task/evaluate.py" in command)
    assert evaluation[evaluation.index("--gpus") + 1] == gpu_spec
    assert "CUDA_VISIBLE_DEVICES=0,1" in evaluation
    build_commands = [command for command in commands if any(value.endswith(":/runtime:rw") for value in command)]
    assert all("--gpus" not in command for command in build_commands)
    manifest = json.loads((task / ".agentic_runtime.json").read_text())
    assert manifest["gpu_evaluation"] is True
    assert manifest["cuda_visible_devices"] == "0,1"

    # Former: GPU baseline requires an explicit selector.
    assert not prepare_task_runtime(
        _baseline_task(tmp_path / "unselected"),
        containerize=True,
        container_image=IMAGE_ID,
        gpu_evaluation=True,
        base_environment={"CUDA_VISIBLE_DEVICES": "0"},
    )


def test_contracted_qualification_retains_feasibility_in_disjoint_brokered_topology(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir = _write_contract_task(tmp_path)
    shutil.rmtree(task_dir / ".venv")
    make_program_dir(task_dir / "program_0")
    _gpu_contract(task_dir, gpu_count=2, candidate_gpu_count=1)
    _rewrite_contract(
        task_dir,
        lambda payload: payload["metrics"].__setitem__(
            "search", [{"name": "combined_score", "data_role": "development"}]
        ),
    )
    candidate_commands: list[list[str]] = []
    removed_containers: list[Path] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(
        runtime,
        "CandidateBroker",
        _fake_broker(on_init=lambda command, *_a, **_k: candidate_commands.append(command)),
    )

    def record_removal(cid_path: Path, **_kwargs: object) -> bool:
        removed_containers.append(cid_path)
        return True

    monkeypatch.setattr(runtime, "_force_remove_container", record_removal)

    assert prepare_task_runtime(
        task_dir,
        containerize=True,
        container_image=IMAGE_ID,
        gpu_evaluation=True,
        run=_docker_fake_run(score=0.5, correct=False, feasible=True, site_packages=True, commands=commands),
        base_environment={"PATH": "/bin", "AGENT_CONTAINER_GPUS": '"device=GPU-a1,GPU-b2"'},
        evaluation_requirement=TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT,
    )

    evaluator = next(command for command in commands if "/task/evaluate.py" in command)
    candidate = candidate_commands[0]
    evaluator_rendered = " ".join(evaluator)
    candidate_rendered = " ".join(candidate)
    assert str(task_dir / "program_0") not in evaluator_rendered
    assert "dst=/results" in evaluator_rendered
    assert f"src={task_dir},dst=/task,readonly" in evaluator_rendered
    assert f"src={task_dir / 'program_0'},dst=/program,readonly" in candidate_rendered
    assert f"src={task_dir},dst=/task,readonly" not in candidate_rendered
    assert "dst=/results" not in candidate_rendered
    assert "candidate.sock" not in candidate_rendered
    assert "src/investigation,readonly" in candidate_rendered
    assert evaluator[evaluator.index("--gpus") + 1] == "device=GPU-a1"
    assert candidate[candidate.index("--gpus") + 1] == "device=GPU-b2"
    assert "AGENTIC_TASK_QUALIFICATION=1" in evaluator
    assert "AGENTIC_TASK_QUALIFICATION=1" not in candidate
    manifest = json.loads((task_dir / ".agentic_runtime.json").read_text(encoding="utf-8"))
    assert manifest["baseline_combined_score"] == pytest.approx(0.5)
    assert manifest["baseline_correct"] is False
    assert manifest["baseline_feasible"] is True
    assert runtime_baseline_score(task_dir) == pytest.approx(0.5)
    assert removed_containers == []


_RESET_MODE_CASES = [
    # (draft_smoke, observed_reset_seeds, expected_score)
    (True, (1, 1, 1), 0.5),
    (True, (1, 1, 1, 1), 0.5),
    (True, (1,), None),
    (True, (1, 1), None),
    (True, (1, 3, 5), None),
    (True, (999,), None),
    (False, (1, 3, 5), None),
    (False, (1, 3, 5, 1, 3, 5), 0.5),
    (False, (1, 3, 5, 1, 3, 5, 1), 0.5),
    (False, (1, 3, 5, 1, 3, 5, 3), None),
]


def test_contracted_baseline_uses_explicit_reset_sequence_for_each_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir = _write_contract_task(tmp_path)

    def declare_sequences(payload):
        payload["protocol"]["search_seeds"] = [1, 3, 5]
        payload["protocol"]["search_reset_sequence"] = [1, 3, 5, 1, 3, 5]
        payload["protocol"]["search_smoke_reset_sequence"] = [1, 1, 1]

    contract = _rewrite_contract(task_dir, declare_sequences)
    _stub_validation(monkeypatch, score=0.5)

    for draft_smoke, observed_reset_seeds, expected_score in _RESET_MODE_CASES:
        case = (draft_smoke, observed_reset_seeds)
        broker_options: list[dict[str, object]] = []
        _stub_commands(
            monkeypatch,
            _fake_broker(reset_seeds=observed_reset_seeds, on_init=lambda *_a, **kwargs: broker_options.append(kwargs)),
        )

        result = _run_baseline(task_dir, contract, _succeed, draft_smoke=draft_smoke)

        assert result.combined_score == expected_score, case
        assert len(broker_options) == 1, case
        assert broker_options[0]["max_rpc_calls"] == contract.compute.search.max_candidate_rpc_calls, case
        assert broker_options[0]["expected_reset_seeds"] == ((1, 1, 1) if draft_smoke else (1, 3, 5, 1, 3, 5)), case


def test_contracted_baseline_timeouts_and_cleanup_failures_keep_their_deadline_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Former: contracted task ceiling timeout is authored.
    task_dir = _write_contract_task(tmp_path / "ceiling")
    contract = _rewrite_contract(
        task_dir, lambda payload: payload["compute"]["search"].__setitem__("wall_clock_seconds", 14_400)
    )
    _stub_commands(monkeypatch, _fake_broker(calls=7, peer_disconnects=1))
    observed_timeouts: list[float] = []

    def timeout_run(command, *, timeout, **_kwargs):
        observed_timeouts.append(timeout)
        raise subprocess.TimeoutExpired(command, timeout)

    with pytest.raises(runtime._TaskRuntimeContractDeadlineExceeded, match="14400-second wall-clock ceiling") as error:
        _run_baseline(task_dir, contract, timeout_run, budget=20_000)

    evidence = error.value.failure_evidence
    assert evidence is not None
    assert evidence.deadline_owner == "task_contract"
    assert evidence.contract_wall_clock_seconds == 14_400
    assert evidence.candidate_rpc_timeout_seconds <= 14_379
    assert evidence.evaluator_timeout_seconds <= 14_349
    assert 29 <= evidence.candidate_rpc_timeout_seconds - evidence.evaluator_timeout_seconds <= 31
    assert observed_timeouts[0] <= evidence.evaluator_timeout_seconds < observed_timeouts[0] + 1
    assert evidence.broker_teardown_reserve_seconds == 21.0
    assert evidence.container_cleanup_reserve_seconds == 30.0
    assert evidence.candidate_rpc_calls == 7
    assert evidence.candidate_reset_count == 1
    assert evidence.candidate_peer_disconnects == 1

    # Former: contracted outer deadline timeout remains infrastructure.
    task_dir = _write_contract_task(tmp_path / "outer")
    _stub_commands(monkeypatch, _fake_broker(calls=0, reset_seeds=()))

    with pytest.raises(runtime.TaskRuntimeDeadlineExceeded) as outer_error:
        _run_baseline(task_dir, _contract(task_dir), _timeout_run)

    assert type(outer_error.value) is runtime.TaskRuntimeDeadlineExceeded
    assert outer_error.value.failure_evidence is not None
    assert outer_error.value.failure_evidence.deadline_owner == "task_drafting"

    # Former: contracted cleanup failure remains infrastructure.
    task_dir = _write_contract_task(tmp_path / "cleanup")

    def fail_cleanup() -> None:
        raise runtime.CandidateBrokerInfrastructureError("cleanup failed")

    _stub_commands(monkeypatch, _fake_broker(on_exit=fail_cleanup))

    result = _run_baseline(task_dir, _contract(task_dir), _timeout_run, budget=1000)

    assert result.status is Status.INFRASTRUCTURE_FAILED

    # Former: contracted GPU baseline classifies a busy allocation as infrastructure.
    task_dir = _write_contract_task(tmp_path / "busy")
    contract = _gpu_contract(task_dir)
    _stub_commands(monkeypatch)

    def allocation_busy(*_args: object, **_kwargs: object) -> object:
        raise GpuLeaseTimeout("GPU allocation remained busy until the deadline: GPU-aaaa")

    monkeypatch.setattr(runtime, "acquire_gpu_allocation_lease", allocation_busy)

    result = _run_baseline(task_dir, contract, subprocess.run, environment=_GPU_ENV)

    assert result.status is Status.INFRASTRUCTURE_FAILED


def test_contracted_evaluator_cleanup_failure_is_not_authored_and_quarantine_follows_cleanup_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Former: contracted evaluator cleanup failure is not authored.
    task_dir = _write_contract_task(tmp_path / "cleanup")
    contract = _gpu_contract(task_dir)
    observed_deadlines: list[float] = []
    lease = _QuarantineLease()

    def acquire(*_args: object, deadline: float, **_kwargs: object) -> _QuarantineLease:
        observed_deadlines.append(deadline)
        return lease

    _stub_commands(monkeypatch, _fake_broker())
    monkeypatch.setattr(runtime, "acquire_gpu_allocation_lease", acquire)
    monkeypatch.setattr(runtime, "_remove_container_before_deadline", lambda *_a, **_k: False)

    with pytest.raises(runtime._TaskRuntimeCleanupDeadlineExceeded) as error:
        _run_baseline(task_dir, contract, _timeout_run, budget=1000, environment=_GPU_ENV)

    assert error.value.failure_evidence is not None
    assert error.value.failure_evidence.deadline_owner == "task_contract"
    assert lease.quarantined
    assert observed_deadlines
    assert observed_deadlines[0] < time.monotonic() + 970

    # Former: contracted exception quarantine depends on cleanup verification.
    task_dir = _write_contract_task(tmp_path / "quarantine")
    contract = _gpu_contract(task_dir)
    lease = _QuarantineLease()
    monkeypatch.setattr(runtime, "acquire_gpu_allocation_lease", lambda *_a, **_k: lease)

    def interrupt(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run_baseline(task_dir, contract, interrupt, budget=1000, environment=_GPU_ENV)

    assert lease.quarantined

    lease.quarantined = False
    monkeypatch.setattr(runtime, "_remove_container_before_deadline", lambda *_a, **_k: True)

    with pytest.raises(runtime.TaskRuntimeDeadlineExceeded):
        _run_baseline(task_dir, contract, _timeout_run, budget=1000, environment=_GPU_ENV)

    assert not lease.quarantined


def test_contracted_baseline_recovers_retained_broker_on_next_invocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir = _write_contract_task(tmp_path / "task")
    contract = _contract(task_dir)
    _broker_recovery_root(tmp_path, monkeypatch)
    _stub_commands(monkeypatch)
    invocations = 0

    class FailingThenCleanBroker:
        calls = 1
        calls_before_first_reset = 0
        reset_seeds = (1,)
        peer_disconnects = 0

        def __init__(self, _command: list[str], broker_dir: Path, **_kwargs: object) -> None:
            self.broker_dir = broker_dir

        def __enter__(self) -> "FailingThenCleanBroker":
            nonlocal invocations
            invocations += 1
            assert not self.broker_dir.exists()
            self.broker_dir.mkdir(mode=0o700)
            (self.broker_dir / "scoring-mask").mkdir(mode=0o700)
            return self

        def __exit__(self, *_args: object) -> None:
            if invocations == 1:
                (self.broker_dir / "candidate.cid").write_text("a" * 64, encoding="ascii")
                (self.broker_dir / "evaluator.cid").write_text("b" * 64, encoding="ascii")
                raise runtime.CandidateBrokerInfrastructureError("cleanup failed")
            shutil.rmtree(self.broker_dir)

    monkeypatch.setattr(runtime, "CandidateBroker", FailingThenCleanBroker)
    removed: list[Path] = []

    def remove_retained(cid_path: Path, **_kwargs: object) -> bool:
        removed.append(cid_path)
        return True

    monkeypatch.setattr(runtime, "_force_remove_container", remove_retained)
    _stub_validation(monkeypatch, score=1.0)

    first = _run_baseline(task_dir, contract, _succeed)
    second = _run_baseline(task_dir, contract, _succeed)

    expected_dir = runtime._task_generation_broker_directory(task_dir, "runtime-search")
    assert first.status is Status.INFRASTRUCTURE_FAILED
    assert second.status is Status.SUCCEEDED
    assert second.combined_score == pytest.approx(1.0)
    assert removed == [expected_dir / "candidate.cid", expected_dir / "evaluator.cid"]


def test_task_broker_directory_recovery_reclaims_legacy_and_failed_containers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _broker_recovery_root(tmp_path, monkeypatch)
    removed: list[Path] = []

    def remove(cid_path: Path, **_kwargs: object) -> bool:
        removed.append(cid_path)
        return True

    monkeypatch.setattr(runtime, "_force_remove_container", remove)

    # Former: task broker recovers the legacy directory before the shortened identity.
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    legacy_dir = runtime._legacy_task_generation_broker_directory(task_dir, "runtime-search")
    legacy_dir.mkdir(mode=0o700)
    (legacy_dir / "scoring-mask").mkdir(mode=0o700)
    legacy_cid = legacy_dir / "candidate.cid"
    legacy_cid.write_text("c" * 64, encoding="ascii")

    broker_dir = runtime._prepare_task_broker_directory(task_dir, "runtime-search", deadline=time.monotonic() + 30)

    assert removed == [legacy_cid]
    assert not legacy_dir.exists()
    assert broker_dir == runtime._task_generation_broker_directory(task_dir, "runtime-search")

    # Former: final smoke recovery retains the failed cid, then reclaims it.
    task_dir = tmp_path / "final-task"
    task_dir.mkdir()
    broker_dir = runtime._task_generation_broker_directory(task_dir, "final-smoke")
    broker_dir.mkdir(mode=0o700)
    (broker_dir / "scoring-mask").mkdir(mode=0o700)
    candidate_cid = broker_dir / "candidate.cid"
    candidate_cid.write_text("b" * 64, encoding="ascii")
    failed_cid = broker_dir / "evaluator.cid"
    failed_cid.write_text("a" * 64, encoding="ascii")
    attempts: list[Path] = []

    def fail_then_remove(cid_path: Path, **_kwargs: object) -> bool:
        attempts.append(cid_path)
        return len(attempts) > 1

    monkeypatch.setattr(runtime, "_force_remove_container", fail_then_remove)
    prepare = partial(runtime._prepare_task_broker_directory, task_dir, "final-smoke")

    with pytest.raises(runtime.CandidateBrokerInfrastructureError):
        prepare(deadline=time.monotonic() + 30)
    assert failed_cid.is_file()

    recovered = prepare(deadline=time.monotonic() + 30)

    assert recovered == broker_dir
    assert not broker_dir.exists()
    assert attempts == [candidate_cid, candidate_cid, failed_cid]


def test_prepare_runtime_rejects_invalid_requirements_unmanaged_venv_missing_isolation_and_tree_change(tmp_path: Path) -> None:
    # Former: prepare runtime rejects an unpinned requirement or pip directive.
    task = tmp_path / "unpinned"
    task.mkdir()
    (task / "requirements.txt").write_text(
        "--extra-index-url=https://example.invalid\nyahpo-gym>=1\n", encoding="utf-8"
    )

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("invalid requirements must fail before execution")

    assert not prepare_task_runtime(task, containerize=True, container_image=IMAGE_ID, run=fake_run)
    assert _prepare(task, fake_run) is PrepResult.AUTHORED_FAILURE

    # Former: prepare runtime rejects an existing unmanaged venv.
    task = tmp_path / "unmanaged"
    python_bin = task / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")

    assert not prepare_task_runtime(task, containerize=True, container_image=IMAGE_ID)

    # Former: prepare runtime requires container isolation.
    task = tmp_path / "uncontained"
    task.mkdir()
    assert not prepare_task_runtime(task, containerize=False)
    assert prepare_task_runtime_result(task, containerize=False) is PrepResult.INFRASTRUCTURE_FAILURE

    # Former: prepare runtime rejects an evaluator tree change during the baseline.
    task = _baseline_task(tmp_path / "tree-change")
    with patch("src.investigation.task_runtime.managed_evaluator_tree_digest", side_effect=["sealed", "changed"]):
        prepared = prepare_task_runtime(
            task,
            containerize=True,
            container_image=IMAGE_ID,
            evaluator_tree_sha256="sealed",
            run=_docker_fake_run(),
        )

    assert prepared is False
    assert not (task / ".venv").exists()


def test_failed_install_leaves_no_partial_runtime_and_retry_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _baseline_task(tmp_path, requirements="example==1.0\n")
    interrupted_build = tmp_path.parent / f".{tmp_path.name}.runtime-build-interrupted"
    interrupted_build.mkdir(mode=0o700)
    (interrupted_build / ".tmp").mkdir(mode=0o700)
    (interrupted_build / ".tmp" / "partial-wheel").write_text("incomplete\n", encoding="utf-8")
    interrupted_cid = tmp_path.parent / f".{tmp_path.name}.runtime-install-{'1' * 32}.cid"
    interrupted_cid.write_text("a" * 64, encoding="utf-8")
    interrupted_cid.chmod(0o600)
    removed_containers: list[Path] = []

    def remove_interrupted_container(cid_path: Path, *, timeout_seconds: float) -> bool:
        assert timeout_seconds > 0
        assert interrupted_build.exists()
        removed_containers.append(cid_path)
        return True

    monkeypatch.setattr(runtime, "_force_remove_container", remove_interrupted_container)
    fail_install = [True]

    def fail_first_install(command, _kwargs):
        if "pip" in command and fail_install[0]:
            fail_install[0] = False
            return _completed(command, 1, "", "install failed")
        return None

    fake_run = _docker_fake_run(hook=fail_first_install)

    assert not prepare_task_runtime(tmp_path, containerize=True, container_image=IMAGE_ID, run=fake_run)
    assert not (tmp_path / ".venv").exists()
    assert removed_containers == [interrupted_cid]
    assert not interrupted_cid.exists()
    assert not interrupted_build.exists()
    assert not list(tmp_path.parent.glob(f".{tmp_path.name}.runtime-build-*"))

    assert prepare_task_runtime(tmp_path, containerize=True, container_image=IMAGE_ID, run=fake_run)
    assert (tmp_path / ".venv/bin/python").is_file()


def test_run_local_install_cache_reuses_dependencies_after_baseline_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir = _baseline_task(tmp_path / "task", requirements="example==1.0\n")
    install_cache = _install_cache(tmp_path)
    install_commands = 0
    evaluation_commands = 0
    build_lease_active = False
    build_lease_acquisitions = 0
    owned_runtime_sources: list[Path] = []
    original_store_owned = TaskRuntimeInstallCache.store_owned

    def reject_copying_store(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("cache misses must publish the cache-owned runtime without copying")

    def record_owned_store(self, key, source_venv, *, deadline=None):
        owned_runtime_sources.append(source_venv)
        return original_store_owned(self, key, source_venv, deadline=deadline)

    monkeypatch.setattr(TaskRuntimeInstallCache, "store", reject_copying_store)
    monkeypatch.setattr(TaskRuntimeInstallCache, "store_owned", record_owned_store)

    class BuildLease:
        def release(self) -> None:
            nonlocal build_lease_active
            assert build_lease_active
            build_lease_active = False

    def acquire_build_lease(*, deadline: float) -> BuildLease:
        nonlocal build_lease_active, build_lease_acquisitions
        assert deadline > 0
        assert not build_lease_active
        build_lease_active = True
        build_lease_acquisitions += 1
        return BuildLease()

    monkeypatch.setattr("src.investigation.task_runtime.acquire_runtime_build_lease", acquire_build_lease)

    def check_lease(command, _kwargs):
        nonlocal install_commands, evaluation_commands
        if "venv" in command:
            assert build_lease_active
        if "pip" in command:
            assert build_lease_active
            install_commands += 1
        if "/task/evaluate.py" in command:
            assert not build_lease_active
            evaluation_commands += 1
            if evaluation_commands == 1:
                return _completed(command, 1, "", "baseline exploded")
        return None

    fake_run = _docker_fake_run(hook=check_lease)
    prepare = partial(_prepare, task_dir, fake_run, install_cache_dir=install_cache)

    first = prepare()
    first_install_commands = install_commands
    (task_dir / "evaluate.py").write_text("# repaired evaluator\n", encoding="utf-8")
    second = prepare()

    assert first is PrepResult.AUTHORED_FAILURE
    assert second is PrepResult.READY
    assert first_install_commands > 0
    assert install_commands == first_install_commands
    assert evaluation_commands == 2
    assert (task_dir / ".venv/bin/python").is_file()
    assert len(owned_runtime_sources) == 1
    assert owned_runtime_sources[0].parent.parent == install_cache
    assert owned_runtime_sources[0].parent.name.startswith(".build-")
    assert not list(tmp_path.glob(".task.runtime-build-*"))

    installs_before_requirement_change = install_commands
    (task_dir / "requirements.txt").write_text("example==2.0\n", encoding="utf-8")
    third = prepare()

    assert third is PrepResult.READY
    assert install_commands > installs_before_requirement_change
    assert evaluation_commands == 3
    assert len(owned_runtime_sources) == 2
    assert build_lease_acquisitions == 2
    assert not build_lease_active
    assert not list(tmp_path.glob(".task.runtime-build-*"))


def test_install_cache_and_requirement_install_failures_are_classified_distinctly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Former: install cache build-root deadline is an infrastructure failure.
    task_dir = _baseline_task(tmp_path / "task")
    install_cache = _install_cache(tmp_path)

    def fail_build_root(_self, *, deadline=None) -> Path:
        assert deadline is not None
        raise TaskRuntimeInstallCacheDeadlineExceeded("deadline")

    def reject(command, _kwargs):
        if command[:4] != _DOCKER_INSPECT:
            raise AssertionError(f"unexpected command after cache deadline: {command}")

    with monkeypatch.context() as m:
        m.setattr(TaskRuntimeInstallCache, "create_build_root", fail_build_root)
        result = _prepare(task_dir, _docker_fake_run(hook=reject), install_cache_dir=install_cache)

    assert result is PrepResult.INFRASTRUCTURE_FAILURE

    # Former: install cache store failure preserves the recovered runtime.
    retained_source: Path | None = None

    def fail_after_restore(_self, _key, source_venv, *, deadline=None) -> Path:
        nonlocal retained_source
        assert deadline is not None
        retained_source = source_venv
        raise OSError("injected post-rollback sync failure")

    with monkeypatch.context() as m:
        m.setattr(TaskRuntimeInstallCache, "store_owned", fail_after_restore)
        result = _prepare(task_dir, _docker_fake_run(), install_cache_dir=install_cache)

    assert result is PrepResult.INFRASTRUCTURE_FAILURE
    assert retained_source is not None
    assert (retained_source / "bin/python").is_file()

    # Former: requirement install failures have distinct results (docker launch vs. resolution).
    for name, returncode, stderr, expected in (
        ("docker-launch-failure", 125, "daemon unavailable", PrepResult.INFRASTRUCTURE_FAILURE),
        ("resolution-failure", 1, "no matching distribution", PrepResult.DEPENDENCY_FAILURE),
    ):
        task = _baseline_task(tmp_path / name, requirements="missing-wheel==1.0\n")

        def fail_requirements(command, _kwargs, returncode=returncode, stderr=stderr):
            if "--requirement" in command:
                return _completed(command, returncode, "", stderr)
            return None

        assert _prepare(task, _docker_fake_run(hook=fail_requirements)) is expected, name


# --- checked commands, timeouts and container cleanup ---------------------------------------


def test_runtime_timeouts_reserve_cleanup_budget_and_retain_failed_cids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])

    # Former: runtime timeout reserves the cleanup budget and retains the failed cid.
    cid_path = tmp_path / "runtime.cid"
    cid_path.write_text("a" * 64, encoding="ascii")
    command_timeouts: list[float] = []
    cleanup_timeouts: list[float] = []

    def timed_out_run(command, *, timeout, **_kwargs):
        assert isinstance(timeout, (int, float))
        command_timeouts.append(float(timeout))
        now[0] += float(timeout)
        raise subprocess.TimeoutExpired(command, timeout)

    def failed_cleanup(_cid_path: Path, *, timeout_seconds: float) -> bool:
        cleanup_timeouts.append(timeout_seconds)
        return False

    monkeypatch.setattr(runtime, "_force_remove_container", failed_cleanup)

    with pytest.raises(runtime.TaskRuntimeDeadlineExceeded):
        _run_checked(tmp_path, timed_out_run, command=("docker", "run"), deadline=10, cid_path=cid_path)

    assert command_timeouts == pytest.approx([9.0])
    assert cleanup_timeouts == pytest.approx([1.0])
    assert cid_path.is_file()

    # Former: runtime timeout without a cleanup budget retains the cid.
    now[0] = 0.0
    cleanup_calls: list[Path] = []

    def overrun(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        now[0] = 11.0
        raise subprocess.TimeoutExpired(command, 10)

    def record_cleanup(cid: Path, **_kwargs: object) -> bool:
        cleanup_calls.append(cid)
        return True

    monkeypatch.setattr(runtime, "_force_remove_container", record_cleanup)

    with pytest.raises(runtime.TaskRuntimeDeadlineExceeded):
        _run_checked(tmp_path, overrun, command=("docker", "run"), deadline=10, cid_path=cid_path)

    assert cleanup_calls == []
    assert cid_path.is_file()

    # Former: stale container timeout preserves the cid and the deadline error.
    now[0] = 0.0
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    stale_cid_path = tmp_path / f".task.runtime-create-{'1' * 32}.cid"
    stale_cid_path.write_text("a" * 64, encoding="ascii")
    stale_cid_path.chmod(0o600)

    def timed_out_cleanup(_cid_path: Path, *, timeout_seconds: float) -> bool:
        now[0] += timeout_seconds
        return False

    monkeypatch.setattr(runtime, "_force_remove_container", timed_out_cleanup)

    with pytest.raises(runtime.TaskRuntimeDeadlineExceeded):
        runtime._remove_stale_runtime_containers(task_dir, deadline=1.0)

    assert stale_cid_path.is_file()


def test_checked_command_classifies_silent_failure_output_flood_launch_failure_and_umask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # Former: checked command reports a silent authored failure.
    command = ["python", "evaluate.py"]

    result = _run_checked(tmp_path, lambda *_a, **_k: _completed(command, 1), command=command)

    assert result.status is Status.COMMAND_FAILED
    assert "exited with status 1 without writing a diagnostic to stderr" in caplog.text

    # Former: container command output is bounded.
    def flood(*_args: object, **_kwargs: object) -> object:
        raise ProcessOutputLimitError("bounded at 65536 bytes")

    with monkeypatch.context() as m:
        m.setattr(runtime, "run_bounded_process", flood)
        outcome = _run_checked(tmp_path, subprocess.run, command=("docker", "run", "image", "/task/evaluate.py"))

    assert outcome.status.value == "command_failed"
    assert outcome.process is None

    # Former: checked runtime reports unverified container cleanup.
    cid_path = tmp_path / "evaluator.cid"
    cid_path.write_text("a" * 64, encoding="ascii")

    def fail_launch(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("container runtime unavailable")

    with monkeypatch.context() as m:
        m.setattr(runtime, "_remove_container_before_deadline", lambda *_a, **_k: False)
        outcome = _run_checked(tmp_path, fail_launch, cid_path=cid_path)

    assert outcome.status is Status.INFRASTRUCTURE_FAILED
    assert not outcome.cleanup_succeeded
    assert cid_path.exists()

    # Former: runtime cid commands use a private child umask.
    observed_umasks: list[int | None] = []

    def succeed(*_args: object, **kwargs: object) -> BoundedProcessResult:
        observed_umasks.append(kwargs.get("child_umask"))  # type: ignore[arg-type]
        return BoundedProcessResult(0, b"", b"")

    monkeypatch.setattr(runtime, "run_bounded_process", succeed)
    deadline = time.monotonic() + 30
    for umask_cid_path in (tmp_path / "runtime.cid", None):
        outcome = _run_checked(tmp_path, subprocess.run, deadline=deadline, cid_path=umask_cid_path)
        assert outcome.status.value == "succeeded"

    assert observed_umasks == [0o077, None]


def test_stale_runtime_cleanup_refuses_symlinks_recovers_interrupted_quarantine_and_tolerates_removed_containers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Former: stale runtime cleanup refuses a symlinked build root.
    root = tmp_path / "symlinked"
    task_dir = root / "task"
    task_dir.mkdir(parents=True)
    outside = root / "outside"
    outside.mkdir()
    marker = outside / "keep"
    marker.write_text("safe\n", encoding="utf-8")
    (root / ".task.runtime-build-untrusted").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="stale task runtime build is unsafe"):
        _remove_stale_runtime_builds(task_dir)

    assert marker.read_text(encoding="utf-8") == "safe\n"

    # Former: stale runtime cleanup recovers an interrupted quarantine.
    root = tmp_path / "interrupted"
    task_dir = root / "task"
    task_dir.mkdir(parents=True)
    stale_build = root / ".task.runtime-build-interrupted"
    stale_build.mkdir(mode=0o700)
    (stale_build / "partial-wheel").write_text("incomplete\n", encoding="utf-8")

    def interrupted_rmtree(*_args: object, **_kwargs: object) -> None:
        raise OSError("interrupted")

    with monkeypatch.context() as m:
        m.setattr(runtime.shutil, "rmtree", interrupted_rmtree)
        with pytest.raises(OSError, match="interrupted"):
            _remove_stale_runtime_builds(task_dir)

    interrupted = list(root.glob(".task.runtime-build-*"))
    assert len(interrupted) == 1
    _remove_stale_runtime_builds(task_dir)
    assert not interrupted[0].exists()

    # Former: force remove container accepts an already removed container.
    cid_path = tmp_path / "stale.cid"
    cid_path.write_text("a" * 64, encoding="utf-8")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *_a, **_k: _completed(
            ["docker", "rm"], 1, "", "Error response from daemon: No such container: " + "a" * 64
        ),
    )

    assert runtime._force_remove_container(cid_path)


# --- production baseline archives -----------------------------------------------------------


def test_archive_preserves_binary_results_and_never_publishes_partial_or_racing_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Former: archive preserves binary results without overwrite.
    root = tmp_path / "binary"
    source = root / "results"
    source.mkdir(parents=True)
    (source / "rollout.png").write_bytes(bytes(range(256)))
    (source / "plots").mkdir()
    (source / "plots/loss.json").write_text("[1, 0.5]")
    target = root / "archive"
    archive.archive_baseline_results(source, target, task_dir=root / "task")
    assert (target / "rollout.png").read_bytes() == bytes(range(256))
    assert (target / "plots/loss.json").read_text() == "[1, 0.5]"
    assert target.stat().st_mode & 0o777 == 0o700
    with pytest.raises((OSError, ValueError)):
        archive.archive_baseline_results(source, target, task_dir=root / "task")

    # Former: archive failure never publishes a partial tree (symlink, oversize, fifo).
    for kind in ("symlink", "oversize", "fifo"):
        root = tmp_path / kind
        source = root / "results"
        source.mkdir(parents=True)
        (source / "first").write_bytes(b"123")
        with monkeypatch.context() as m:
            if kind == "symlink":
                (source / "unsafe").symlink_to(source / "first")
            elif kind == "fifo":
                os.mkfifo(source / "unsafe")
            else:
                m.setattr(archive, "MAX_ARCHIVE_BYTES", 2)
            with pytest.raises((OSError, ValueError)):
                archive.archive_baseline_results(source, root / "archive", task_dir=root / "task")
        assert not (root / "archive").exists(), kind
        assert sorted(path.name for path in root.iterdir()) == ["results"], kind

    # Former: a racing archive destination is preserved.
    root = tmp_path / "racing"
    source = root / "results"
    source.mkdir(parents=True)
    (source / "evidence").write_text("validated")
    target = root / "archive"
    publish = archive.rename_directory_no_replace

    def race(*args, **kwargs):
        target.mkdir()
        (target / "previous").write_text("keep")
        publish(*args, **kwargs)

    monkeypatch.setattr(archive, "rename_directory_no_replace", race)
    with pytest.raises(OSError, match="already exists"):
        archive.archive_baseline_results(source, target, task_dir=root / "task")
    assert (target / "previous").read_text() == "keep"
    assert not (target / "evidence").exists()
    assert not list(root.glob(".baseline-archive-*"))


@pytest.mark.parametrize("kind", ["inside_task", "symlink_parent", "existing", "draft_smoke"])
def test_unsafe_archive_rejected_before_runtime_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind) -> None:
    task = _write_contract_task(tmp_path)
    target = tmp_path / "archive"
    if kind == "inside_task":
        target = task / "evidence"
    elif kind == "symlink_parent":
        (tmp_path / "alias").symlink_to(task, target_is_directory=True)
        target = tmp_path / "alias/evidence"
    elif kind == "existing":
        target.mkdir()
    monkeypatch.setattr(runtime, "_runtime_build_lock", lambda *_a: pytest.fail("runtime started"))
    report = prepare_task_runtime_report(task, draft_smoke=kind == "draft_smoke", baseline_results_dir=target)
    assert report.result is PrepResult.INFRASTRUCTURE_FAILURE
    if kind == "draft_smoke":
        assert not target.exists()


@pytest.mark.parametrize("outcome", ["success", "invalid_output", "archive_failure"])
def test_contracted_archive_runs_after_cleanup_and_before_temp_removal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome) -> None:
    task = _write_contract_task(tmp_path)
    state: dict[str, Any] = {"cleaned": False}

    def command(**kwargs):
        state["results"] = kwargs["results_dir"]
        (kwargs["results_dir"] / "evidence.json").write_text("evidence")
        assert "baseline_results_dir" not in kwargs
        return ["evaluate"]

    def copy(source, destination, *, task_dir):
        assert state["cleaned"]
        if outcome == "archive_failure":
            raise OSError("archive unavailable")
        archive.archive_baseline_results(source, destination, task_dir=task_dir)

    monkeypatch.setattr(
        runtime, "CandidateBroker", _fake_broker(reset_seeds=(0,), on_exit=lambda: state.__setitem__("cleaned", True))
    )
    monkeypatch.setattr(runtime, "_prepare_task_broker_directory", lambda *_a, **_k: tmp_path)
    monkeypatch.setattr(runtime, "_evaluation_command", command)
    monkeypatch.setattr(runtime, "_candidate_command", lambda **_k: ["candidate"])
    monkeypatch.setattr(runtime, "_candidate_call_sequence_diagnostic", lambda **_k: None)
    _stub_validation(monkeypatch, score=None if outcome == "invalid_output" else 0.8)
    monkeypatch.setattr(runtime, "archive_baseline_results", copy)
    monkeypatch.setattr(
        runtime,
        "_run_checked",
        lambda *_a, **_k: runtime._CheckedCommandResult(Status.SUCCEEDED, subprocess.CompletedProcess([], 0)),
    )
    target = tmp_path / "archive"

    result = _run_baseline(
        task, _contract(task), subprocess.run, budget=300, environment={}, baseline_results_dir=target
    )

    assert not state["results"].exists()
    assert target.exists() == (outcome == "success")
    expected = {
        "success": Status.SUCCEEDED,
        "invalid_output": Status.COMMAND_FAILED,
        "archive_failure": Status.INFRASTRUCTURE_FAILED,
    }
    assert result.status is expected[outcome]


# --- runtime build lease --------------------------------------------------------------------


def test_runtime_build_lease_serializes_builders_and_rejects_unsafe_roots_and_deadlines(tmp_path: Path) -> None:
    # Former: runtime build lease uses a host-stable namespace.
    assert RUNTIME_BUILD_LEASE_ROOT.parent == Path("/tmp")

    # Former: runtime build lease serializes builders.
    lease_root = tmp_path / "leases"
    first = acquire_runtime_build_lease(lease_root=lease_root, deadline=time.monotonic() + 1)
    try:
        with pytest.raises(RuntimeBuildLeaseTimeout):
            acquire_runtime_build_lease(lease_root=lease_root, deadline=time.monotonic() + 0.02)
    finally:
        first.release()

    acquire_runtime_build_lease(lease_root=lease_root, deadline=time.monotonic() + 1).release()

    # Former: runtime build lease rejects a symlinked root.
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    symlinked_root = tmp_path / "symlinked-leases"
    symlinked_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeBuildLeaseError, match="unsafe"):
        acquire_runtime_build_lease(lease_root=symlinked_root, deadline=time.monotonic() + 1)

    # Former: runtime build lease rejects a non-finite deadline.
    for deadline in (math.inf, -math.inf, math.nan):
        with pytest.raises(RuntimeBuildLeaseError, match="finite"):
            acquire_runtime_build_lease(lease_root=tmp_path / "nonfinite-leases", deadline=deadline)


# --- sealed task mounts ---------------------------------------------------------------------

_PRINT_TASK_SOURCE_MOUNTS = (
    "docker_cmd=(); append_task_source_options /task-source "
    "'/host/task:/task-source:ro'; printf '%s\\n' \"${docker_cmd[@]}\""
)


def test_task_source_mount_masks_declared_final_artifacts_and_rejects_unsafe_sealed_path(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Former: task source mount masks the declared final artifacts.
    result = _container_agent_shell(
        _PRINT_TASK_SOURCE_MOUNTS,
        AGENT_TASK_SEALED_FILES="evaluate_final.py\ntask_contract.json",
        AGENT_TASK_SEALED_DIRS="resources/final",
        AGENT_TASK_SEAL_EMPTY_DIR=str(empty_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "/dev/null:/task-source/evaluate_final.py:ro" in result.stdout
    assert "/dev/null:/task-source/task_contract.json:ro" in result.stdout
    assert f"{empty_dir}:/task-source/resources/final:ro" in result.stdout

    # Former: task source mount rejects an unsafe sealed path.
    result = _container_agent_shell(
        "docker_cmd=(); append_task_source_options /task-source ''",
        AGENT_TASK_SEALED_FILES="../evaluate_final.py",
        AGENT_TASK_SEAL_EMPTY_DIR=str(empty_dir),
    )

    assert result.returncode != 0
    assert "unsafe sealed task path" in result.stderr


@pytest.mark.parametrize("has_manual_report", [False, True])
def test_contracted_agent_mounts_seal_retained_manual_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_manual_report) -> None:
    task_dir = tmp_path / "task"
    _write_scaffold(task_dir)
    report = task_dir / "manual_validation.json"
    report_contents = '{"final": {"metrics": {"combined_score": 0.75}}}\n'
    if has_manual_report:
        report.write_text(report_contents)
    monkeypatch.setattr(os, "environ", dict(os.environ))

    _configure_contracted_agent_isolation(task_dir, tmp_path / "results")
    result = _container_agent_shell(_PRINT_TASK_SOURCE_MOUNTS)

    assert result.returncode == 0, result.stderr
    mounts = result.stdout.splitlines()
    assert ("/dev/null:/task-source/manual_validation.json:ro" in mounts) is has_manual_report
    for private_file in ("evaluate_final.py", "task_contract.json", ".agentic_task_validation.json"):
        assert f"/dev/null:/task-source/{private_file}:ro" in mounts
    assert f"{tmp_path / 'results/.task-seal-empty'}:/task-source/resources/final:ro" in mounts
    assert os.environ["AGENT_CONTAINER_MOUNT_DOCKER_SOCK"] == "0"
    if has_manual_report:
        assert report.read_text() == report_contents
    else:
        assert not report.exists()


def test_search_evaluator_and_runtime_smoke_mask_contract_final_artifacts(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    _write_scaffold(task_dir)

    # Former: search evaluator masks the contract final artifacts.
    command: list[str] = []

    _append_search_seals(command, task_source_dir=task_dir, evaluator_source_dir=task_dir)

    assert "/task-source/resources/final:ro,noexec,nosuid,nodev,size=64k,nr_inodes=16" in command
    assert "type=bind,src=/dev/null,dst=/task-source/evaluate_final.py,readonly" in command

    # Former: runtime search smoke masks the contract final artifacts.
    command = _docker_command_for(
        task_dir, image_id="sha256:test", inner_command=["true"], cid_path=tmp_path / "runtime.cid",
        search_evaluation=True,
    )

    assert "/task/resources/final:ro,noexec,nosuid,nodev,size=64k,nr_inodes=16" in command
    assert "type=bind,src=/dev/null,dst=/task/evaluate_final.py,readonly" in command


@pytest.mark.parametrize("has_manual_report", [False, True])
def test_idea_shell_seals_optional_manual_report(tmp_path: Path, has_manual_report: bool) -> None:
    from tests.e2e.test_agent_execution_pipeline import (
        _script_path,
        _write_contracted_task,
        _write_env_capture_docker,
    )

    task = _write_contracted_task(tmp_path)
    report = task / "manual_validation.json"
    if has_manual_report:
        report.write_text('{"final": "private validation sentinel"}\n')
    private_temp = tmp_path / "private-temp"
    private_temp.mkdir(mode=0o700)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_env_capture_docker(bin_dir)
    captured_args = tmp_path / "docker.args"
    source_copy = tmp_path / "captured-task-source"
    source_env = tmp_path / "source.env"
    source_env.write_text("")
    result = subprocess.run(
        [
            "bash", str(_script_path("idea.sh")), "--agent", "codex", "--task-dir", str(task),
            "--results-dir", str(tmp_path / "results"), "--cycle-index", "1",
            "--output-dir", str(tmp_path / "results/idea"), "--containerize",
            "--container-env-file", str(source_env),
        ],
        capture_output=True, text=True, timeout=30,
        env=os.environ | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}", "WRAPPER_PYTHON": sys.executable,
            "FAKE_DOCKER_ARGS_PATH": str(captured_args),
            "FAKE_DOCKER_CONTROLS_PATH": str(tmp_path / "docker.controls"),
            "FAKE_DOCKER_ENV_PATH": str(tmp_path / "docker.env"),
            "FAKE_DOCKER_TASK_SOURCE_COPY": str(source_copy),
            "AGENTIC_PRIVATE_TMPDIR": str(private_temp),
        },
    )

    assert result.returncode != 0
    assert captured_args.is_file(), result.stdout + result.stderr
    arguments = captured_args.read_text().splitlines()
    for destination in ("/task-source", "/workspace/task"):
        assert "/dev/null:" + destination + "/task_contract.json:ro" in arguments
        assert "/dev/null:" + destination + "/evaluate_final.py:ro" in arguments
        assert ("/dev/null:" + destination + "/manual_validation.json:ro" in arguments) is has_manual_report
    assert not any("/var/run/docker.sock" in argument for argument in arguments)
    if has_manual_report:
        assert (source_copy / "manual_validation.json").read_bytes() == b""
        assert "private validation sentinel" in report.read_text()
    else:
        public_report = source_copy / "manual_validation.json"
        assert not public_report.exists() or public_report.read_bytes() == b""


# --- headless capacity environment (container_agent.sh) -------------------------------------


def _dispatch_codex(tmp_path: Path, provider: Path | None, *, session: str = "", gpus: str = "0") -> str:
    env_file = shlex.quote(str(provider)) if provider is not None else repr("")
    return (
        f"_dispatch_inner_command codex {shlex.quote(str(tmp_path))} {gpus} image host home "
        f"{env_file} '' {shlex.quote(session)}"
    )


@pytest.mark.parametrize("value", ["0", "1", "2", "3"])
@pytest.mark.parametrize("source", ["host", "provider"])
@pytest.mark.parametrize("subscription", ["0", "1"])
def test_capacity_override_reaches_restricted_native_command(tmp_path: Path, value, source, subscription) -> None:
    provider = tmp_path / "provider.env"
    provider.write_text(f"HEADLESS_CAPACITY_RETRIES={value}\n")
    result = _container_agent_shell(
        'AGENT_ENV_ALLOWLIST_KEYS="PATH HOME"\n'
        "AGENT_INNER_CMD=(bash -c 'printf \"%s\" \"${HEADLESS_CAPACITY_RETRIES:-missing}\"')\n"
        + _dispatch_codex(tmp_path, provider if source == "provider" else None),
        AGENT_CHATGPT_SUBSCRIPTION=subscription,
        **({"HEADLESS_CAPACITY_RETRIES": value} if source == "host" else {}),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == value


@pytest.mark.parametrize("value", ["", "4", "-1", "01", " 1", "1;echo unsafe"])
@pytest.mark.parametrize("source", ["host", "provider"])
def test_invalid_override_rejected_before_container_launch(tmp_path: Path, value, source) -> None:
    provider = tmp_path / "provider.env"
    provider.write_text(f"HEADLESS_CAPACITY_RETRIES='{value}'\n")
    result = _container_agent_shell(
        "docker() { printf launched; return 99; }\nAGENT_INNER_CMD=(true)\n"
        + _dispatch_codex(tmp_path, provider if source == "provider" else None, gpus="1"),
        **({"HEADLESS_CAPACITY_RETRIES": value} if source == "host" else {}),
    )
    assert result.returncode == 2
    assert "HEADLESS_CAPACITY_RETRIES" in result.stderr
    assert "launched" not in result.stdout


def test_host_capacity_setting_overrides_provider_and_concept_capacity_control_is_codex_only(tmp_path: Path) -> None:
    # Former: host capacity setting overrides the provider setting.
    provider = tmp_path / "provider.env"
    provider.write_text("HEADLESS_CAPACITY_RETRIES=3\n")
    result = _container_agent_shell(
        f"resolve_agent_capacity_retries codex {shlex.quote(str(provider))}", HEADLESS_CAPACITY_RETRIES="0"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"

    # Former: concept runtime capacity control is codex-only.
    for agent in ("claude", "pi", "codex"):
        result = _container_agent_shell(
            f"source {shlex.quote(str(CONCEPT_FIGURE))}\nAGENT={agent}\n"
            "concept_figure_log() { :; }\nselect_concept_figure_backend\n"
            'eval "$CONCEPT_FIGURE_ENV_COMMANDS"\n'
            'printf "%s" "${HEADLESS_CAPACITY_RETRIES:-missing}"',
            HEADLESS_CAPACITY_RETRIES="0",
        )
        assert result.returncode == 0, (agent, result.stderr)
        assert result.stdout == ("0" if agent == "codex" else "missing"), agent


@pytest.mark.parametrize("agent", ["codex", "pi"])
def test_concept_subscription_scoped_file_preserves_only_codex_capacity(tmp_path: Path, agent) -> None:
    provider = tmp_path / "provider.env"
    provider.write_text("HEADLESS_CAPACITY_RETRIES=0\nOPENAI_API_KEY=denied\n")
    result = _container_agent_shell(
        f"source {shlex.quote(str(CONCEPT_FIGURE))}\nAGENT={agent}\n"
        f"CONTAINER_ENV_FILE={shlex.quote(str(provider))}\n"
        "concept_figure_log() { :; }\nselect_concept_figure_backend\n"
        "prepare_concept_figure_container_env\n"
        'cat "$CONCEPT_FIGURE_SCOPED_ENV_FILE"\ncleanup_concept_figure_scoped_env',
        AGENT_CHATGPT_SUBSCRIPTION="1",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ("HEADLESS_CAPACITY_RETRIES=0\n" if agent == "codex" else "")


@pytest.mark.parametrize("session", ["", "reuse"])
@pytest.mark.parametrize("source", ["host", "provider"])
def test_capacity_override_reaches_one_shot_and_reused_containers(tmp_path: Path, session, source) -> None:
    provider = tmp_path / "provider.env"
    provider.write_text("HEADLESS_CAPACITY_RETRIES=0\nOPENAI_API_KEY=denied\n")
    result = _container_agent_shell(
        "wait_for_docker_daemon() { return 0; }\n"
        "ensure_agent_session_container() { return 0; }\n"
        'run_with_bounded_capture() { printf "%s\\n" "${@:3}"; }\n'
        'docker() { printf "%s\\n" "$@"; }\n'
        "AGENT_INNER_CMD=(/usr/local/bin/headless codex --prompt test)\n"
        + _dispatch_codex(tmp_path, provider if source == "provider" else None, session=session, gpus="1"),
        AGENT_CHATGPT_SUBSCRIPTION="1",
        **({"HEADLESS_CAPACITY_RETRIES": "0"} if source == "host" else {}),
    )
    assert result.returncode == 0, result.stderr
    assert "HEADLESS_CAPACITY_RETRIES=0" in result.stdout.splitlines()
    assert "OPENAI_API_KEY" not in result.stdout


# --- docker integration (opt-in) ------------------------------------------------------------


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is unavailable")
def test_image_built_runtime_executes_in_sealed_variant_sandbox(tmp_path: Path) -> None:
    image = os.environ.get("AGENTIC_TEST_SANDBOX_IMAGE")
    if not image:
        pytest.skip("set AGENTIC_TEST_SANDBOX_IMAGE to run Docker runtime integration")

    task_dir = _baseline_task(tmp_path / "task")
    (task_dir / "evaluate.py").write_text(
        """import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--program_path')
p.add_argument('--results_dir')
a = p.parse_args()
r = Path(a.results_dir)
(r / 'metrics.json').write_text(json.dumps({'combined_score': 1.0}))
(r / 'correct.json').write_text(json.dumps({'correct': True}))
""",
        encoding="utf-8",
    )

    assert prepare_task_runtime(task_dir, containerize=True, container_image=image, timeout_seconds=120)
    manifest = json.loads((task_dir / ".agentic_runtime.json").read_text(encoding="utf-8"))

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    make_program_dir(bundle_dir / "program_1", content="print('candidate')\n")
    result = evaluate_variant_program(
        task_bundle_dir=bundle_dir,
        evaluator_source_dir=task_dir,
        task_source_dir=task_dir,
        program_index=1,
        python_bin=task_dir / ".venv/bin/python",
        sandbox_image=str(manifest["image_id"]),
        timeout_seconds=120,
    )

    assert result.correct is True
    assert result.metrics["combined_score"] == 1.0

    container_home = tmp_path / "agent-home"
    container_home.mkdir()
    shell_script = r'''set -euo pipefail
source "$1"
export TASK_SOURCE_DIR="$2"
export EVAL_RESULTS_SUBDIR="$7"
export AGENT_CONTAINER_PASSTHROUGH_KEYS=EVAL_RESULTS_SUBDIR
AGENT_INNER_CMD=(bash -lc '"$TASK_PYTHON" -m src.investigation.program_clone --help >/dev/null && "$TASK_PYTHON" "$TASK_SOURCE_DIR/evaluate.py" --program_path "$TASK_SOURCE_DIR/program_0/src/program.py" --results_dir "$TASK_SOURCE_DIR/$EVAL_RESULTS_SUBDIR"')
_dispatch_inner_command pi "$2" 1 "$4" bridge "$5" "" "$3" "$6"
'''
    session_name = f"agentic-runtime-test-{uuid.uuid4().hex[:12]}"
    try:
        for result_name, session in (("one-shot-results", ""), ("session-results", session_name)):
            (task_dir / result_name).mkdir()
            subprocess.run(
                [
                    "bash", "-c", shell_script, "runtime-test", str(CONTAINER_AGENT), str(task_dir),
                    str(tmp_path), str(manifest["image_id"]), str(container_home), session, result_name,
                ],
                check=True, capture_output=True, text=True, timeout=120,
            )
            assert (task_dir / result_name / "metrics.json").is_file()
    finally:
        subprocess.run(["docker", "rm", "--force", session_name], capture_output=True, text=True, check=False)
