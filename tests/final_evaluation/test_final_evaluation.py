"""Sealed final evaluation runner: lifecycle, container launch, cleanup, manifests, usage."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal
from unittest.mock import patch

import pytest

from src.final_evaluation.runner import (
    MAX_FINAL_RESULT_BYTES,
    FinalEvaluationError,
    FinalEvaluationInfrastructureError,
    _candidate_command,
    _classify_launch_failure,
    _ensure_broker_recovery_root,
    _evaluation_command,
    _exact_gpu_selector,
    _force_remove_container,
    _final_broker_directory,
    _recover_broker_directory,
    _retryable_launch_failure,
    _sanitize_launch_stderr,
    run_final_evaluation as _run_final_evaluation,
    verify_final_manifest,
)
from src.investigation.container_gpus import GpuLeaseTimeout
from src.investigation.evaluation_usage import (
    compute_usage,
    record_compute_failure,
    validate_compute_usage,
)
from src.investigation.task_contract import load_task_contract

RUNNER = "src.final_evaluation.runner"
IMAGE = "example/image:latest"
PINNED_IMAGE = "sha256:" + "a" * 64
GPU_ENV = {"AGENT_CONTAINER_GPUS": "device=GPU-aaaa"}
GPU_FINAL_COMPUTE = {
    "accelerator": "gpu",
    "accelerator_model": "H100",
    "gpu_count": 1,
    "evaluator_gpu_count": 1,
    "candidate_gpu_count": 0,
}


# --- shared helpers (imported by other test modules; keep names and signatures) -------------


def run_final_evaluation(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("_allow_native_for_tests", True)
    kwargs.setdefault("_resolve_container_image", lambda image: image)
    kwargs.setdefault("_remove_named_container", lambda _name: None)
    kwargs.setdefault("_sleep", lambda _seconds: None)
    return _run_final_evaluation(**kwargs)


class _RecordingGpuLease:
    def __init__(self) -> None:
        self.active = False
        self.released = False
        self.quarantined = False
        self.prepared = False

    def __enter__(self) -> "_RecordingGpuLease":
        self.active = True
        return self

    def __exit__(self, *_args: object) -> None:
        self.active = False
        self.released = True

    def quarantine(self) -> None:
        assert self.active
        self.quarantined = True

    def prepare_runtime(self) -> None:
        assert self.active
        self.prepared = True

    def wrap_command(self, command: list[str]) -> list[str]:
        assert self.active and self.prepared
        return command


def _write_contract_task(root: Path) -> Path:
    task = root / "task"
    (task / "resources" / "search").mkdir(parents=True)
    (task / "resources" / "final").mkdir()
    (task / ".venv/lib/python3.12/site-packages/src/investigation").mkdir(parents=True)
    (task / "task.md").write_text("# Frozen task\n", encoding="utf-8")
    (task / "evaluate.py").write_text("# search\n", encoding="utf-8")
    (task / "requirements.txt").write_text("example==1.0.0\n", encoding="utf-8")
    (task / "evaluate_final.py").write_text(
        """import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--program_path', required=True)
p.add_argument('--results_dir', required=True)
a = p.parse_args()
out = Path(a.results_dir)
out.mkdir(parents=True, exist_ok=True)
(out / 'metrics.json').write_text(json.dumps({'final_accuracy': 0.75}))
(out / 'correct.json').write_text(json.dumps({'correct': True}))
""",
        encoding="utf-8",
    )
    task_digest = hashlib.sha256((task / "task.md").read_bytes()).hexdigest()
    compute = {
        "accelerator": "cpu",
        "accelerator_model": "none",
        "gpu_count": 0,
        "evaluator_gpu_count": 0,
        "candidate_gpu_count": 0,
        "cpu_cores": 1,
        "memory_gb": 2,
        "precision": "float64",
        "wall_clock_seconds": 120,
    }
    contract = {
        "schema_version": 1,
        "canonical_task": {"path": "task.md", "sha256": task_digest},
        "method": {"class": "test_method", "description": "Test final evaluator."},
        "candidate": {"interface": "python_rpc_v1:handle", "editable_paths": ["program_0/src"]},
        "evaluators": {"search": "evaluate.py", "final": "evaluate_final.py"},
        "resources": {
            "search_root": "resources/search",
            "final_root": "resources/final",
            "public_artifact_bundle": None,
        },
        "data": {"search_roles": ["development"], "final_roles": ["sealed_test"]},
        "metrics": {
            "search": [{"name": "combined_score", "data_role": "development"}],
            "final": [{"name": "final_accuracy", "data_role": "sealed_test"}],
        },
        "upstream": [
            {
                "repository": "https://example.test/repo",
                "revision": "abc123",
                "package_version": "1.0.0",
                "license": "MIT",
                "entrypoints": ["example:main"],
                "patches": [],
            }
        ],
        "provenance": {"dependencies": [{"name": "example", "version": "1.0.0"}], "artifacts": []},
        "compute": {"search": dict(compute), "final": dict(compute)},
        "protocol": {
            "search_seeds": [1],
            "final_seeds": [2],
            "final_reset_sequence": [2],
            "baselines": ["baseline"],
            "promotion_rule": "Freeze the best development candidate.",
            "search_proxies": [],
        },
        "feedback": {"search_visible": ["development metrics"], "final_sealed": ["sealed test examples"]},
        "publication": {
            "search_is_proxy": True,
            "requires_final_evaluation": True,
            "required_evidence": ["sealed final metrics"],
        },
    }
    (task / "task_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    return task


def _write_investigation(root: Path) -> tuple[Path, Path]:
    investigation = root / "investigation"
    program = investigation / "best_program"
    (program / "src").mkdir(parents=True)
    entrypoint = program / "src" / "program.py"
    entrypoint.write_text("print('frozen')\n", encoding="utf-8")
    (investigation / "best_program_manifest.json").write_text(
        json.dumps(
            {
                "program_dir": str(program),
                "entrypoint_path": "src/program.py",
                "combined_score": 0.5,
                "correct": True,
            }
        ),
        encoding="utf-8",
    )
    return investigation, entrypoint


# --- local helpers ---------------------------------------------------------------------------


def _edit_contract(task: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    contract_path = task / "task_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    edit(contract)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")


def _set_final_reset_sequence(task: Path, seeds: list[int], sequence: list[int]) -> None:
    def edit(contract: dict[str, Any]) -> None:
        contract["protocol"]["final_seeds"] = seeds
        contract["protocol"]["final_reset_sequence"] = sequence

    _edit_contract(task, edit)


def _prepare(tmp_path: Path, *, runtime: bool = False, gpu: bool = False) -> tuple[Path, Path, Path]:
    """Return (task, investigation, output) with optional runtime python and GPU final compute."""
    task = _write_contract_task(tmp_path)
    if runtime:
        runtime_python = task / ".venv/bin/python"
        runtime_python.parent.mkdir(parents=True)
        runtime_python.write_text("", encoding="utf-8")
    if gpu:
        _edit_contract(task, lambda contract: contract["compute"]["final"].update(GPU_FINAL_COMPUTE))
    investigation, _ = _write_investigation(tmp_path)
    return task, investigation, tmp_path / "final"


def _patch_evaluator(task: Path, old: str, new: str) -> None:
    evaluator = task / "evaluate_final.py"
    evaluator.write_text(evaluator.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


def _evaluate(task: Path, investigation: Path, output: Path, **kwargs: Any) -> dict[str, Any]:
    return run_final_evaluation(
        task_dir=task,
        investigation_results_dir=investigation,
        output_dir=output,
        python_bin=Path(sys.executable),
        **kwargs,
    )


def _write_results(output: Path) -> None:
    results = output / "results"
    (results / "metrics.json").write_text('{"final_accuracy": 0.75}', encoding="utf-8")
    (results / "correct.json").write_text('{"correct": true}', encoding="utf-8")


def _fake_broker(
    *,
    calls: int = 0,
    reset_seeds: tuple[int, ...] = (2,),
    peer_disconnects: int = 0,
    on_init: Callable[..., None] | None = None,
    on_enter: Callable[[], None] | None = None,
) -> type[Any]:
    """CandidateBroker stand-in; entered instances are appended to ``FakeBroker.created``."""
    created: list[Any] = []

    class FakeBroker:
        calls_before_first_reset = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.calls = calls
            self.reset_seeds = reset_seeds
            self.peer_disconnects = peer_disconnects
            if on_init is not None:
                on_init(*args, **kwargs)

        def __enter__(self) -> "FakeBroker":
            created.append(self)
            if on_enter is not None:
                on_enter()
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    FakeBroker.created = created  # type: ignore[attr-defined]
    return FakeBroker


def _succeed(output: Path, broker: type[Any] | None = None) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A ``run`` stand-in that writes valid results (and marks the live fake broker as used)."""

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if broker is not None:
            broker.created[-1].calls = 1
        _write_results(output)
        return subprocess.CompletedProcess(argv, 0, "", "")

    return run


def _launch_failure(stderr: str, returncode: int = 125) -> Callable[..., subprocess.CompletedProcess[str]]:
    return lambda argv, **_kwargs: subprocess.CompletedProcess(argv, returncode, "", stderr)


def _raise(error: BaseException) -> Callable[..., Any]:
    def raiser(*_args: object, **_kwargs: object) -> Any:
        raise error

    return raiser


def _docker_rm(returncode: int, stderr: str) -> Any:
    return patch(
        f"{RUNNER}.subprocess.run",
        return_value=subprocess.CompletedProcess(["docker", "rm"], returncode, "", stderr),
    )


def _eval_command(task: Path, tmp_path: Path, **overrides: Any) -> list[str]:
    for name in ("results", "broker"):
        (tmp_path / name).mkdir(exist_ok=True)
    kwargs: dict[str, Any] = dict(
        task_dir=task,
        program_dir=task / "program_0",
        entrypoint=task / "program_0/src/program.py",
        evaluator=task / "evaluate_final.py",
        results_dir=tmp_path / "results",
        python_bin=Path(sys.executable),
        container_image=PINNED_IMAGE,
        gpu_count=0,
        environment={},
        broker_dir=tmp_path / "broker",
    )
    kwargs.update(overrides)
    return _evaluation_command(**kwargs)


def _public_artifacts(tmp_path: Path, *, bundle_mode: int = 0o500, mutable_file: bool = False) -> tuple[Path, Path]:
    artifacts = tmp_path / "public-artifacts"
    artifacts.mkdir(mode=0o700)
    bundle = artifacts / "test-bundle"
    bundle.mkdir(mode=0o700)
    if mutable_file:
        checkpoint = bundle / "checkpoint.bin"
        checkpoint.write_bytes(b"checkpoint")
        checkpoint.chmod(0o600)
    bundle.chmod(bundle_mode)
    return artifacts, bundle


def _load_manifest(output: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = output / "final_manifest.json"
    manifest_path.chmod(0o644)
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def _started_output(output: Path) -> None:
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    (output / "invocation.json").write_text('{"status":"started"}\n', encoding="utf-8")


# --- lifecycle: isolation, sealing, reuse ---------------------------------------------------


def test_final_evaluation_refuses_unisolated_interrupted_and_stale_invocations(tmp_path: Path) -> None:
    # requires container isolation
    task, investigation, output = _prepare(tmp_path / "a")
    with pytest.raises(FinalEvaluationError, match="requires container isolation"):
        _run_final_evaluation(
            task_dir=task,
            investigation_results_dir=investigation,
            output_dir=output,
            python_bin=Path(sys.executable),
        )

    # refuses interrupted invocation
    task, investigation, output = _prepare(tmp_path / "b")
    _started_output(output)
    with pytest.raises(FinalEvaluationError, match="already started"):
        _evaluate(task, investigation, output)

    # retry recovers stale broker before refusing rerun
    task, investigation, output = _prepare(tmp_path / "c")
    _started_output(output)
    _ensure_broker_recovery_root()
    broker_dir = _final_broker_directory(output)
    broker_dir.mkdir(mode=0o700)
    (broker_dir / "scoring-mask").mkdir(mode=0o700)
    (broker_dir / "candidate.cid").write_text("a" * 64, encoding="ascii")
    with _docker_rm(0, ""), pytest.raises(FinalEvaluationError, match="already started"):
        _evaluate(task, investigation, output)
    assert not broker_dir.exists()


def test_container_removal_and_broker_recovery_outcomes(tmp_path: Path) -> None:
    # evaluator removal outcomes
    for stderr, retained in [
        ("daemon unavailable", True),
        ("Error response from daemon: No such container", False),
    ]:
        cid_path = tmp_path / "evaluator.cid"
        cid_path.write_text("a" * 64, encoding="ascii")
        with _docker_rm(1, stderr):
            if retained:
                with pytest.raises(FinalEvaluationError, match="could not remove"):
                    _force_remove_container(cid_path)
            else:
                _force_remove_container(cid_path)
        assert cid_path.exists() is retained, stderr

    # broker recovery retries retained candidate and evaluator cids
    broker_dir = tmp_path / "broker"
    broker_dir.mkdir(mode=0o700)
    (broker_dir / "scoring-mask").mkdir(mode=0o700)
    (broker_dir / "candidate.cid").write_text("a" * 64, encoding="ascii")
    (broker_dir / "evaluator.cid").write_text("b" * 64, encoding="ascii")
    with _docker_rm(1, "daemon unavailable"), pytest.raises(FinalEvaluationError, match="could not remove"):
        _recover_broker_directory(broker_dir)
    assert (broker_dir / "candidate.cid").is_file()
    assert (broker_dir / "evaluator.cid").is_file()
    with _docker_rm(0, "") as remove:
        _recover_broker_directory(broker_dir)
    assert not broker_dir.exists()
    assert [call.args[0][-1] for call in remove.call_args_list] == ["a" * 64, "b" * 64]


EDITED_MANIFEST_CASES = [
    ("unknown-field", "untrusted_note", "ignore verification", "unexpected fields"),
    ("inconsistent-outcome", "outcome", "not_qualified", "outcome does not match correct"),
]


def test_final_evaluation_runs_once_seals_results_and_verifies_manifest(tmp_path: Path) -> None:
    # runs once and seals reused result
    for correct in (True, False):
        task, investigation, output = _prepare(tmp_path / ("qualified" if correct else "not_qualified"))
        if not correct:
            _patch_evaluator(task, "{'correct': True}", "{'correct': False}")
        calls = 0

        def run_once(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            nonlocal calls
            calls += 1
            return subprocess.run(argv, **kwargs)

        first = _evaluate(task, investigation, output, run=run_once)
        second = _evaluate(task, investigation, output, run=run_once)
        assert calls == 1, correct
        assert first == second, correct
        assert first["schema_version"] == 3
        assert first["status"] == "completed"
        assert first["outcome"] == ("qualified" if correct else "not_qualified")
        assert first["correct"] is correct
        assert first["metrics"] == {"final_accuracy": 0.75}
        assert first["program_tree_sha256"].startswith("sha256:")
        assert first["task_tree_sha256"].startswith("sha256:")
        assert (output / "final_manifest.json").is_file()
        assert (output / "results").stat().st_mode & 0o222 == 0
        assert (output / "results" / "correct.json").stat().st_mode & 0o222 == 0
        if correct:
            qualified = (task, investigation, output, first)

    task, investigation, output, manifest = qualified
    assert verify_final_manifest(output, task_dir=task, program_dir=investigation / "best_program") == manifest
    manifest_path, original = _load_manifest(output)
    original_text = manifest_path.read_text(encoding="utf-8")

    # verify_final_manifest accepts legacy qualified schema
    for legacy_version in (1, 2):
        edited = json.loads(original_text)
        edited["schema_version"] = legacy_version
        if legacy_version == 1:
            edited.pop("outcome")
        edited.pop("evaluation_evidence")
        edited.pop("compute_usage")
        manifest_path.write_text(json.dumps(edited), encoding="utf-8")
        verified = verify_final_manifest(output)
        assert verified["schema_version"] == legacy_version
        assert verified["correct"] is True

    # verify_final_manifest rejects edited manifest
    for name, field, value, match in EDITED_MANIFEST_CASES:
        edited = json.loads(original_text)
        edited[field] = value
        manifest_path.write_text(json.dumps(edited), encoding="utf-8")
        with pytest.raises(FinalEvaluationError, match=match):
            verify_final_manifest(output)
    manifest_path.write_text(original_text, encoding="utf-8")
    assert verify_final_manifest(output) == manifest

    # verify_final_manifest rejects tampered results
    (output / "results" / "metrics.json").chmod(0o644)
    (output / "results" / "metrics.json").write_text('{"final_accuracy": 0.99}', encoding="utf-8")
    with pytest.raises(FinalEvaluationError, match="results digest mismatch"):
        verify_final_manifest(output)


NONCONFORMING_RESULT_CASES = [
    ("missing-declared-metric", "'final_accuracy': 0.75", "'different_metric': 0.75",
     "exactly match declared metrics"),
    ("undeclared-metric", "'final_accuracy': 0.75", "'final_accuracy': 0.75, 'search_only_score': 1.0",
     "exactly match declared metrics"),
    ("extra-correctness-field", "{'correct': True}", "{'correct': True, 'debug': 'leak'}",
     "exactly one boolean field"),
]


def test_final_evaluation_rejects_nonconforming_results(tmp_path: Path) -> None:
    for name, old, new, match in NONCONFORMING_RESULT_CASES:
        task, investigation, output = _prepare(tmp_path / name)
        _patch_evaluator(task, old, new)
        with pytest.raises(FinalEvaluationError, match=match):
            _evaluate(task, investigation, output)


def test_final_evaluation_captures_process_output_and_rejects_oversized_results(tmp_path: Path) -> None:
    # captures process output without inheriting terminal
    task, investigation, output = _prepare(tmp_path / "a")
    observed: dict[str, object] = {}

    def quiet_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs)
        _write_results(output)
        return subprocess.CompletedProcess(argv, 0)

    _evaluate(task, investigation, output, run=quiet_run)
    assert observed["stdout"] is subprocess.PIPE
    assert observed["stderr"] is subprocess.PIPE
    assert "capture_output" not in observed

    # rejects oversized results
    task, investigation, output = _prepare(tmp_path / "b")

    def oversized_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        _write_results(output)
        with (output / "results" / "oversized.bin").open("wb") as handle:
            handle.truncate(MAX_FINAL_RESULT_BYTES + 1)
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(FinalEvaluationError, match="undeclared result artifacts"):
        _evaluate(task, investigation, output, run=oversized_run)


# --- GPU admission --------------------------------------------------------------------------


def test_final_evaluation_gpu_admission_budget_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GPU wait consumes evaluation budget
    task, investigation, output = _prepare(tmp_path / "budget", runtime=True, gpu=True)
    clock = [100.0]
    broker_timeouts: list[float] = []
    observed_timeouts: list[float] = []

    def record_timeout(*_args: object, rpc_timeout_seconds: float, **_kwargs: object) -> None:
        broker_timeouts.append(rpc_timeout_seconds)

    def enter_broker() -> None:
        clock[0] = 105.0

    broker = _fake_broker(calls=1, on_init=record_timeout, on_enter=enter_broker)

    def acquire(*_args: object, deadline: float, **_kwargs: object) -> _RecordingGpuLease:
        assert deadline == pytest.approx(110.0)
        clock[0] = 104.0
        return _RecordingGpuLease()

    def successful_run(argv: list[str], *, timeout: float, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        observed_timeouts.append(timeout)
        return _succeed(output)(argv)

    with monkeypatch.context() as m:
        m.setattr(f"{RUNNER}.time.monotonic", lambda: clock[0])
        m.setattr(f"{RUNNER}.acquire_gpu_allocation_lease", acquire)
        m.setattr(f"{RUNNER}.CandidateBroker", broker)
        result = _evaluate(
            task, investigation, output,
            timeout_seconds=10,
            container_image="example/image@sha256:abc",
            environment=GPU_ENV,
            run=successful_run,
        )
    assert broker.created
    assert result["status"] == "completed"
    assert broker_timeouts == pytest.approx([6.0])
    assert observed_timeouts == pytest.approx([5.0])

    # preflights GPU before recording invocation
    task, investigation, output = _prepare(tmp_path / "invalid", gpu=True)
    with pytest.raises(FinalEvaluationInfrastructureError, match="gpu_allocation_invalid"):
        _evaluate(
            task, investigation, output, container_image="example/image@sha256:abc", environment={}
        )
    assert not (output / "invocation.json").exists()
    assert not (output / "results").exists()

    # reports busy GPU allocation as infrastructure
    task, investigation, output = _prepare(tmp_path / "busy", runtime=True, gpu=True)
    with monkeypatch.context() as m:
        m.setattr(
            f"{RUNNER}.acquire_gpu_allocation_lease",
            _raise(GpuLeaseTimeout("GPU allocation remained busy until the deadline: GPU-aaaa")),
        )
        with pytest.raises(FinalEvaluationInfrastructureError, match="gpu_allocation_busy"):
            _evaluate(
                task, investigation, output, container_image="example/image@sha256:abc", environment=GPU_ENV
            )
    assert not (output / "invocation.json").exists()
    assert not (output / "results").exists()

    # holds GPU lease while recording invocation
    task, investigation, output = _prepare(tmp_path / "hold", runtime=True, gpu=True)
    lease = _RecordingGpuLease()

    def fail_invocation_write(*_args: object, **_kwargs: object) -> None:
        assert lease.active
        assert lease.prepared
        raise OSError("injected invocation write failure")

    with monkeypatch.context() as m:
        m.setattr(f"{RUNNER}.acquire_gpu_allocation_lease", lambda *_args, **_kwargs: lease)
        m.setattr(f"{RUNNER}._write_exclusive_json", fail_invocation_write)
        with pytest.raises(OSError, match="injected invocation write failure"):
            _evaluate(
                task, investigation, output, container_image="example/image@sha256:abc", environment=GPU_ENV
            )
    assert lease.released

    # GPU admission exhaustion is infrastructure
    task, investigation, output = _prepare(tmp_path / "exhausted", runtime=True, gpu=True)
    clock = [100.0]
    lease = _RecordingGpuLease()

    def acquire_exhausted(*_args: object, deadline: float, **_kwargs: object) -> _RecordingGpuLease:
        assert deadline == pytest.approx(110.0)
        clock[0] = deadline
        return lease

    with monkeypatch.context() as m:
        m.setattr(f"{RUNNER}.time.monotonic", lambda: clock[0])
        m.setattr(f"{RUNNER}.acquire_gpu_allocation_lease", acquire_exhausted)
        with pytest.raises(FinalEvaluationInfrastructureError, match="gpu_allocation_busy"):
            _evaluate(
                task, investigation, output,
                timeout_seconds=10,
                container_image="example/image@sha256:abc",
                environment=GPU_ENV,
            )
    assert lease.released
    assert not (output / "invocation.json").exists()


def test_gpu_selector_quoting_and_cli_infrastructure_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # exact GPU selector preserves docker CSV quoting for multiple devices
    assert _exact_gpu_selector({"AGENT_CONTAINER_GPUS": '"device=0,1"'}, 2, offset=0) == '"device=0,1"'

    # evaluation command quotes two evaluator GPUs
    task = _write_contract_task(tmp_path)
    command = _eval_command(
        task, tmp_path,
        evaluator=task / "evaluate.py",
        gpu_count=2,
        environment={"AGENT_CONTAINER_GPUS": '"device=0,1"'},
    )
    assert command[command.index("--gpus") + 1] == '"device=0,1"'

    # exact GPU selector rejects unbalanced quotes
    with pytest.raises(FinalEvaluationInfrastructureError, match="gpu_allocation_invalid"):
        _exact_gpu_selector({"AGENT_CONTAINER_GPUS": '"device=0,1'}, 2, offset=0)

    # CLI returns infrastructure exit code
    from src.final_evaluation import cli

    monkeypatch.setattr(
        cli, "run_final_evaluation", _raise(FinalEvaluationInfrastructureError("gpu_allocation_busy"))
    )
    exit_code = cli.main([
        "--task-dir", "task",
        "--investigation-results-dir", "investigation",
        "--output-dir", "final",
        "--python-bin", "python",
        "--container-image", "image",
    ])
    assert exit_code == 4


# --- container launch: preflight, retries, diagnostics --------------------------------------


def test_final_evaluation_image_preflight_and_prestart_retries(tmp_path: Path) -> None:
    # retries prestart container failures
    task, investigation, output = _prepare(tmp_path / "retry", runtime=True)
    attempts = 0
    broker = _fake_broker()

    def flaky_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return subprocess.CompletedProcess(argv, 125, "", "daemon race")
        return _succeed(output, broker)(argv)

    with patch(f"{RUNNER}.CandidateBroker", side_effect=broker):
        result = _evaluate(task, investigation, output, container_image=IMAGE, run=flaky_run)
    assert attempts == 3
    assert result["status"] == "completed"
    assert not (output / "final_launch_failure.json").exists()

    # preflights image before recording invocation
    task, investigation, output = _prepare(tmp_path / "preflight")
    with pytest.raises(FinalEvaluationError, match="image unavailable"):
        _evaluate(
            task, investigation, output,
            container_image=IMAGE,
            _resolve_container_image=_raise(ValueError("image unavailable")),
        )
    assert not (output / "invocation.json").exists()
    assert not (output / "results").exists()

    # exhausted prestart failures remain retryable
    task, investigation, output = _prepare(tmp_path / "exhausted", runtime=True)
    attempts = 0

    def unavailable_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        stderr = (
            "\x1b[31mCannot connect to the Docker daemon\x1b[0m "
            "API_TOKEN=should-not-be-recorded\n" + ("x" * 5000)
        )
        return subprocess.CompletedProcess(argv, 125, "", stderr)

    with (
        patch(f"{RUNNER}.CandidateBroker", side_effect=_fake_broker(reset_seeds=())),
        pytest.raises(
            FinalEvaluationError,
            match="daemon_unavailable.*Cannot connect to the container runtime daemon",
        ),
    ):
        _evaluate(task, investigation, output, container_image=IMAGE, run=unavailable_run)

    diagnostic_path = output / "final_launch_failure.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert attempts == 3
    assert not (output / "invocation.json").exists()
    assert not (output / "results").exists()
    assert diagnostic["reason"] == "container_runtime_launch_failed"
    assert diagnostic["return_code"] == 125
    assert diagnostic["attempts"] == 3
    assert diagnostic["stderr_sha256"].startswith("sha256:")
    assert diagnostic["schema_version"] == 2
    launch_attempts = diagnostic["launch_attempts"]
    assert len(launch_attempts) == 3
    assert all(item["category"] == "daemon_unavailable" for item in launch_attempts)
    assert launch_attempts[-1]["stderr_truncated"] is True
    assert len(launch_attempts[-1]["stderr_excerpt"].encode()) <= 4096
    assert "should-not-be-recorded" not in launch_attempts[-1]["stderr_excerpt"]
    assert launch_attempts[-1]["stderr_excerpt"] == "Cannot connect to the container runtime daemon."
    assert "\x1b" not in launch_attempts[-1]["stderr_excerpt"]
    assert diagnostic_path.stat().st_mode & 0o777 == 0o400


SINGLE_ATTEMPT_LAUNCH_CASES = [
    ("retry-deadline-terminalizes-evidence", "Cannot connect to the Docker daemon", "daemon_unavailable", True),
    ("deterministic-failure-not-retried", "error mounting /task: not a dir", "mount_failure", False),
]


def test_final_evaluation_single_attempt_launch_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, stderr, category, deadline in SINGLE_ATTEMPT_LAUNCH_CASES:
        task, investigation, output = _prepare(tmp_path / name, runtime=True)
        clock = [100.0]
        attempts = 0
        sleeps: list[float] = []
        kwargs: dict[str, Any] = {}

        def failing(argv: list[str], _stderr=stderr, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal attempts
            attempts += 1
            clock[0] = 101.5
            return subprocess.CompletedProcess(argv, 125, "", _stderr)

        with monkeypatch.context() as m:
            if deadline:
                m.setattr(f"{RUNNER}.time.monotonic", lambda: clock[0])
                kwargs["timeout_seconds"] = 2
            m.setattr(f"{RUNNER}.CandidateBroker", _fake_broker(reset_seeds=()))
            with pytest.raises(FinalEvaluationError, match="failed to start after 1 attempt"):
                _evaluate(
                    task, investigation, output,
                    container_image=IMAGE, run=failing, _sleep=sleeps.append, **kwargs,
                )

        diagnostic = json.loads((output / "final_launch_failure.json").read_text())
        assert attempts == 1, name
        assert sleeps == [], name
        assert diagnostic["attempts"] == 1, name
        assert diagnostic["launch_attempts"][0]["category"] == category, name


LAUNCH_FAILURE_CLASSIFICATION_CASES = [
    (125, "Cannot connect to the Docker daemon", "daemon_unavailable"),
    (125, "manifest unknown", "image_unavailable"),
    (125, "container name is already in use", "container_name_conflict"),
    (125, "cidfile already exists", "cidfile_conflict"),
    (125, "error mounting /host/path", "mount_failure"),
    (125, "operation not permitted", "permission_failure"),
    (125, "cannot allocate memory", "resource_exhaustion"),
    (127, "runtime failed", "executable_failure"),
    (125, "runtime failed", "unknown"),
]

LAUNCH_RETRY_POLICY_CASES = [
    (125, "daemon_unavailable", True),
    (125, "container_name_conflict", True),
    (125, "cidfile_conflict", True),
    (125, "resource_exhaustion", True),
    (125, "image_unavailable", True),
    (125, "unknown", True),
    (126, "unknown", False),
    (125, "mount_failure", False),
    (125, "permission_failure", False),
    (127, "executable_failure", False),
]


def test_launch_failure_classification_sanitization_and_retry_policy() -> None:
    # classifies container launch failures
    for return_code, stderr, expected in LAUNCH_FAILURE_CLASSIFICATION_CASES:
        assert _classify_launch_failure(return_code, ("warning " * 800) + stderr) == expected, stderr

    # launch diagnostic escapes terminal controls
    excerpt, truncated = _sanitize_launch_stderr(
        "first\nAuthorization: Basic dXNlcjpwYXNz31m‮forged\x7f".encode(), "unknown"
    )
    assert excerpt == "Container runtime stderr withheld; use its byte count and digest."
    assert truncated is False

    # launch retry policy
    for return_code, category, retryable in LAUNCH_RETRY_POLICY_CASES:
        assert _retryable_launch_failure(return_code, category) is retryable, (return_code, category)


def test_final_evaluation_reresolves_image_before_retry_and_rejects_drift(tmp_path: Path) -> None:
    # re-resolves image before retry
    task, investigation, output = _prepare(tmp_path / "a", runtime=True)
    resolution_requests: list[str] = []
    evaluator_images: list[str] = []
    candidate_images: list[str] = []

    def record_candidate_image(command: list[str], *_args: object, **_kwargs: object) -> None:
        candidate_images.append(next(arg for arg in command if arg.startswith("sha256:")))

    broker = _fake_broker(on_init=record_candidate_image)

    def resolve(image: str) -> str:
        resolution_requests.append(image)
        if len(resolution_requests) == 2:
            raise ValueError("temporary daemon outage")
        return PINNED_IMAGE

    def image_race(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        evaluator_images.append(next(arg for arg in argv if arg.startswith("sha256:")))
        if len(evaluator_images) < 3:
            return subprocess.CompletedProcess(argv, 125, "", "No such image")
        return _succeed(output, broker)(argv)

    with patch(f"{RUNNER}.CandidateBroker", side_effect=broker):
        result = _evaluate(
            task, investigation, output,
            container_image=PINNED_IMAGE,
            container_image_recovery_ref="agentic-scientist/pipeline-lease:test",
            run=image_race,
            _resolve_container_image=resolve,
        )
    assert result["status"] == "completed"
    assert resolution_requests == [
        PINNED_IMAGE,
        "agentic-scientist/pipeline-lease:test",
        "agentic-scientist/pipeline-lease:test",
    ]
    assert evaluator_images == [PINNED_IMAGE] * 3
    assert candidate_images == [PINNED_IMAGE] * 3

    # rejects recovery image drift
    task, investigation, output = _prepare(tmp_path / "b", runtime=True)
    resolved_images = iter((PINNED_IMAGE, "sha256:" + "b" * 64))
    attempts = 0

    def image_missing(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        return subprocess.CompletedProcess(argv, 125, "", "No such image")

    with (
        patch(f"{RUNNER}.CandidateBroker", side_effect=_fake_broker(reset_seeds=())),
        pytest.raises(FinalEvaluationError, match="failed to start after 1 attempt"),
    ):
        _evaluate(
            task, investigation, output,
            container_image=PINNED_IMAGE,
            container_image_recovery_ref="agentic-scientist/pipeline-lease:test",
            run=image_missing,
            _resolve_container_image=lambda _image: next(resolved_images),
        )
    assert attempts == 1


# --- container cleanup ----------------------------------------------------------------------


def test_final_evaluation_recovers_cleanup_before_retry_and_after_success(tmp_path: Path) -> None:
    # recovers cleanup before retry
    task, investigation, output = _prepare(tmp_path / "a", runtime=True)
    attempts = 0
    removal_attempts = 0
    broker = _fake_broker()

    def remove_named_container(_name: str) -> None:
        nonlocal removal_attempts
        removal_attempts += 1
        if removal_attempts == 1:
            raise FinalEvaluationError("daemon unavailable")

    def daemon_race(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return subprocess.CompletedProcess(argv, 125, "", "Cannot connect to the Docker daemon")
        return _succeed(output, broker)(argv)

    with patch(f"{RUNNER}.CandidateBroker", side_effect=broker):
        result = _evaluate(
            task, investigation, output,
            container_image=IMAGE, run=daemon_race, _remove_named_container=remove_named_container,
        )
    assert result["status"] == "completed"
    assert attempts == 2
    assert removal_attempts == 3

    # recovers cleanup after success
    task, investigation, output = _prepare(tmp_path / "b", runtime=True, gpu=True)
    removal_attempts = 0
    removal_lease_states: list[bool] = []
    lease = _RecordingGpuLease()
    broker = _fake_broker()

    def remove_named_container_after_success(_name: str) -> None:
        nonlocal removal_attempts
        removal_lease_states.append(lease.active)
        removal_attempts += 1
        if removal_attempts == 1:
            raise FinalEvaluationError("temporary daemon outage")

    with (
        patch(f"{RUNNER}.CandidateBroker", side_effect=broker),
        patch(f"{RUNNER}.acquire_gpu_allocation_lease", return_value=lease),
    ):
        result = _evaluate(
            task, investigation, output,
            container_image=IMAGE,
            environment=GPU_ENV,
            run=_succeed(output, broker),
            _remove_named_container=remove_named_container_after_success,
        )
    assert result["status"] == "completed"
    assert removal_attempts == 2
    assert removal_lease_states == [True, True]


def test_final_evaluation_records_unverified_cleanup_and_retains_invocation(tmp_path: Path) -> None:
    # retains invocation when cleanup cannot be verified
    task, investigation, output = _prepare(tmp_path / "a", runtime=True)
    evaluator_attempts = 0
    removal_attempts = 0

    def remove_named_container(_name: str) -> None:
        nonlocal removal_attempts
        removal_attempts += 1
        raise FinalEvaluationError("daemon unavailable")

    def daemon_failure(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal evaluator_attempts
        evaluator_attempts += 1
        return subprocess.CompletedProcess(argv, 125, "", "Cannot connect to the Docker daemon")

    with (
        patch(f"{RUNNER}.CandidateBroker", side_effect=_fake_broker(reset_seeds=())),
        pytest.raises(FinalEvaluationError, match="cleanup could not be verified"),
    ):
        _evaluate(
            task, investigation, output,
            container_image=IMAGE, run=daemon_failure, _remove_named_container=remove_named_container,
        )
    assert evaluator_attempts == 1
    assert removal_attempts == 3
    assert (output / "invocation.json").exists()
    assert (output / "results").is_dir()
    cleanup_diagnostic = json.loads((output / "final_cleanup_failure.json").read_text(encoding="utf-8"))
    assert cleanup_diagnostic["reason"] == "container_cleanup_failed"
    assert cleanup_diagnostic["attempts"] == 3

    # records persistent cleanup failure after success
    task, investigation, output = _prepare(tmp_path / "b", runtime=True, gpu=True)
    lease = _RecordingGpuLease()
    broker = _fake_broker()
    with (
        patch(f"{RUNNER}.CandidateBroker", side_effect=broker),
        patch(f"{RUNNER}.acquire_gpu_allocation_lease", return_value=lease),
        pytest.raises(FinalEvaluationError, match="final_cleanup_failure.json"),
    ):
        _evaluate(
            task, investigation, output,
            container_image=IMAGE,
            environment=GPU_ENV,
            run=_succeed(output, broker),
            _remove_named_container=_raise(FinalEvaluationError("daemon unavailable")),
        )
    diagnostic_path = output / "final_cleanup_failure.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostic == {
        "attempts": 3,
        "failed_at": diagnostic["failed_at"],
        "invocation_retained": True,
        "reason": "container_cleanup_failed",
        "schema_version": 1,
        "status": "cleanup_failed",
    }
    assert diagnostic_path.stat().st_mode & 0o777 == 0o400
    assert (output / "invocation.json").exists()
    assert (output / "results").is_dir()
    assert lease.quarantined
    assert lease.released


def test_final_evaluation_interrupts_and_started_evaluators_are_not_retried(tmp_path: Path) -> None:
    # interrupt quarantines unverified evaluator
    task, investigation, output = _prepare(tmp_path / "a", runtime=True, gpu=True)
    lease = _RecordingGpuLease()
    with (
        patch(f"{RUNNER}.CandidateBroker", _fake_broker(reset_seeds=())),
        patch(f"{RUNNER}.acquire_gpu_allocation_lease", return_value=lease),
        pytest.raises(KeyboardInterrupt),
    ):
        _evaluate(
            task, investigation, output,
            container_image=IMAGE,
            environment=GPU_ENV,
            run=_raise(KeyboardInterrupt()),
            _remove_named_container=_raise(FinalEvaluationError("daemon unavailable")),
        )
    assert lease.quarantined
    assert lease.released

    # does not retry after evaluator starts
    task, investigation, output = _prepare(tmp_path / "b", runtime=True)
    attempts = 0

    def failed_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        witness_path = _final_broker_directory(output) / "evaluator-start.sock"
        witness = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            witness.connect(str(witness_path))
            witness.sendall(b"evaluator-started-v1")
        finally:
            witness.close()
        witness_path.unlink()
        return subprocess.CompletedProcess(argv, 125, "", "evaluator failed")

    with (
        patch(f"{RUNNER}.CandidateBroker", side_effect=_fake_broker(reset_seeds=())),
        pytest.raises(FinalEvaluationError, match="return code 125"),
    ):
        _evaluate(task, investigation, output, container_image=IMAGE, run=failed_run)
    assert attempts == 1
    assert (output / "invocation.json").exists()

    # retains invocation when retry cleanup fails
    task, investigation, output = _prepare(tmp_path / "c", runtime=True)
    actual_rmdir = os.rmdir

    def fail_results_rmdir(path: str | bytes | Path, **kwargs: Any) -> None:
        if path == "results":
            raise OSError("transient filesystem failure")
        actual_rmdir(path, **kwargs)

    with (
        patch(f"{RUNNER}.CandidateBroker", side_effect=_fake_broker(reset_seeds=())),
        patch(f"{RUNNER}.os.rmdir", side_effect=fail_results_rmdir),
        pytest.raises(FinalEvaluationError, match="return code 125"),
    ):
        _evaluate(task, investigation, output, container_image=IMAGE, run=_launch_failure("daemon race"))
    assert (output / "invocation.json").exists()
    assert (output / "results").is_dir()


# --- evaluator and candidate container commands ---------------------------------------------


def test_evaluator_and_candidate_commands_mount_only_permitted_paths(tmp_path: Path) -> None:
    # public artifacts mount only in evaluator container
    task = _write_contract_task(tmp_path)
    artifacts, bundle = _public_artifacts(tmp_path)
    command = _eval_command(
        task, tmp_path,
        environment={"AGENTIC_PUBLIC_ASSETS_DIR": str(artifacts)},
        public_artifact_bundle="test-bundle",
    )
    assert f"type=bind,src={bundle.resolve()},dst=/public-artifacts/test-bundle,readonly" in command
    assert "AGENTIC_PUBLIC_ASSETS_DIR=/public-artifacts" in command
    assert "HOME=/tmp" in command
    assert "XDG_CACHE_HOME=/tmp/.cache" in command
    assert "USER=agent" in command
    assert "LOGNAME=agent" in command
    assert command[:3] == ["docker", "run", "--rm"]
    assert "/broker-code/evaluator_launcher.py" not in command
    assert "AGENTIC_EVALUATOR_STARTED_SOCKET=/broker/evaluator-start.sock" not in command

    # unused global public artifact root is ignored
    command = _eval_command(task, tmp_path, environment={"AGENTIC_PUBLIC_ASSETS_DIR": "/"})
    assert not any("/public-artifacts" in argument for argument in command)

    # final container masks search plane artifacts
    (task / ".agentic_task_snapshot.json").write_text("{}\n", encoding="utf-8")
    command = _eval_command(task, tmp_path, sealed_task_paths=("evaluate.py", "resources/search"))
    assert "type=bind,src=/dev/null,dst=/task/evaluate.py,readonly" in command
    assert "type=bind,src=/dev/null,dst=/task/.agentic_task_snapshot.json,readonly" in command
    assert "/task/resources/search:ro,noexec,nosuid,nodev,size=64k,nr_inodes=16" in command

    # evaluator container masks distinct evaluator source snapshot
    evaluator_source = tmp_path / "evaluator-source"
    evaluator_source.mkdir()
    evaluator = evaluator_source / "evaluate_final.py"
    evaluator.write_text("print('ok')\n", encoding="utf-8")
    (evaluator_source / ".agentic_task_snapshot.json").write_text("{}\n", encoding="utf-8")
    command = _eval_command(task, tmp_path, evaluator=evaluator, evaluator_source_dir=evaluator_source)
    assert "type=bind,src=/dev/null,dst=/evaluator-source/.agentic_task_snapshot.json,readonly" in command

    # candidate container cannot mount final task or results
    _, entrypoint = _write_investigation(tmp_path)
    program_dir = entrypoint.parents[1]
    command = _candidate_command(
        task_dir=task,
        program_dir=program_dir,
        entrypoint=entrypoint,
        container_image="example/image@sha256:abc",
        contract=load_task_contract(task / "task_contract.json"),
        environment={},
        broker_dir=tmp_path / "broker",
    )
    rendered = "\n".join(command)
    assert "--rm" not in command
    assert "--network\nnone" in rendered
    assert f"src={program_dir},dst=/program,readonly" in rendered
    assert f"src={task / '.venv'},dst=/task-venv,readonly" in rendered
    assert f"src={task},dst=/task" not in rendered
    assert "dst=/results" not in rendered
    assert "resources/final" not in rendered
    assert "scoring-mask" in rendered
    assert "dst=/task-venv/lib/python3.12/site-packages/src/investigation,readonly" in rendered
    assert "exchange" not in rendered
    assert "AGENTIC_CANDIDATE_EXCHANGE" not in rendered


UNSAFE_PUBLIC_ROOT_CASES = [
    ("writable-directory", "writable_directory", "must be immutable"),
    ("writable-bundle-file", "writable_bundle_file", "must be immutable"),
    ("filesystem-root", "filesystem_root", "narrow and owner-controlled"),
]


def test_evaluator_command_rejects_unsafe_snapshot_metadata_and_public_roots(tmp_path: Path) -> None:
    # evaluator container rejects non-regular snapshot metadata
    for entry_type in ["symlink", "directory"]:
        task = _write_contract_task(tmp_path / entry_type)
        snapshot_metadata = task / ".agentic_task_snapshot.json"
        if entry_type == "symlink":
            snapshot_metadata.symlink_to(tmp_path / "outside")
        else:
            snapshot_metadata.mkdir()
        with pytest.raises(FinalEvaluationError, match="snapshot metadata"):
            _eval_command(task, tmp_path / entry_type)

    # public artifact mount rejects unsafe roots
    for name, kind, match in UNSAFE_PUBLIC_ROOT_CASES:
        task = _write_contract_task(tmp_path / name)
        if kind == "filesystem_root":
            root = "/"
        else:
            artifacts, _ = _public_artifacts(
                tmp_path / name,
                bundle_mode=0o700 if kind == "writable_directory" else 0o500,
                mutable_file=kind == "writable_bundle_file",
            )
            root = str(artifacts)
        with pytest.raises(FinalEvaluationError, match=match):
            _eval_command(
                task, tmp_path / name,
                environment={"AGENTIC_PUBLIC_ASSETS_DIR": root},
                public_artifact_bundle="test-bundle",
            )


BROKEN_PROTOCOL_CASES: list[tuple[str, list[int] | None, dict[str, Any], int, str]] = [
    ("wrong-reset-sequence", [4, 2, 6], dict(calls=1, reset_seeds=(4,)), 0, "expected primary.*observed"),
    ("incomplete-round-trip", None, dict(calls=1, peer_disconnects=1), 0, "disconnected"),
    ("evaluator-failure-beats-peer-disconnect", None, dict(calls=1, peer_disconnects=1), 9, "return code 9"),
]


def test_containerized_final_evaluation_command_and_candidate_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # containerized final evaluation is networkless and read-only
    task, investigation, output = _prepare(tmp_path / "sealed", runtime=True)
    _set_final_reset_sequence(task, [2, 4, 6], [4, 2, 6, 4, 2, 6])
    observed: list[str] = []

    def sealed_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.extend(argv)
        return _succeed(output)(argv)

    broker = _fake_broker(calls=7, reset_seeds=(4, 2, 6, 4, 2, 6, 4))
    with patch(f"{RUNNER}.CandidateBroker", return_value=broker()) as broker_factory:
        _evaluate(
            task, investigation, output, container_image="example/image@sha256:abc", run=sealed_run
        )
    broker_options = broker_factory.call_args.kwargs
    assert broker_options["max_rpc_calls"] == 100_000
    assert broker_options["expected_reset_seeds"] == (4, 2, 6, 4, 2, 6)
    assert observed[:3] == ["docker", "run", "--name"]
    assert ["--network", "none"] == observed[observed.index("--network") : observed.index("--network") + 2]
    assert "--read-only" in observed
    assert any("dst=/task,readonly" in argument for argument in observed)
    assert not any("dst=/program" in argument for argument in observed)
    assert any("dst=/broker" in argument for argument in observed)
    assert "/broker-code/evaluator_launcher.py" in observed
    assert "AGENTIC_EVALUATOR_STARTED_SOCKET=/broker/evaluator-start.sock" in observed
    launcher_index = observed.index("/broker-code/evaluator_launcher.py")
    assert observed[launcher_index - 2 : launcher_index] == ["-I", "-S"]
    assert "/broker-code/candidate_proxy.py" in observed

    # uses absolute container mount sources
    task, investigation, _ = _prepare(tmp_path / "mounts", runtime=True)
    output = tmp_path / "mounts" / "relative" / "final"
    broker = _fake_broker()

    def capture_mounts(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
        bind_sources = [
            mount.split(",src=", 1)[1].split(",dst=", 1)[0]
            for mount in mounts
            if mount.startswith("type=bind,")
        ]
        assert bind_sources
        assert all(Path(source).is_absolute() for source in bind_sources)
        return _succeed(output, broker)(argv)

    with monkeypatch.context() as m:
        m.chdir(tmp_path / "mounts")
        with patch(f"{RUNNER}.CandidateBroker", side_effect=broker):
            result = _evaluate(
                task, investigation, Path("relative/final"), container_image=IMAGE, run=capture_mounts
            )
    assert result["status"] == "completed"
    assert (output / "final_manifest.json").is_file()

    # rejects broken candidate protocol
    for name, reset_sequence, broker_kwargs, returncode, match in BROKEN_PROTOCOL_CASES:
        task, investigation, output = _prepare(tmp_path / name, runtime=True)
        if reset_sequence is not None:
            _set_final_reset_sequence(task, [2, 4, 6], reset_sequence)
        run = _succeed(output) if returncode == 0 else _launch_failure("", returncode)
        with (
            patch(f"{RUNNER}.CandidateBroker", return_value=_fake_broker(**broker_kwargs)()),
            pytest.raises(FinalEvaluationError, match=match),
        ):
            _evaluate(task, investigation, output, container_image="example/image@sha256:abc", run=run)


# --- compute usage accounting ---------------------------------------------------------------


CORRUPT_USAGE_CASES = [
    ("outcome", []),
    ("elapsed_seconds", 10**400),
    ("elapsed_seconds", float("nan")),
    ("rpc_calls", True),
    ("wall_clock_limit_seconds", 0),
    ("max_rpc_calls", False),
    ("objective_counts", "enforced"),
]

FAILURE_USAGE_CASES = [
    (TimeoutError("private payload"), 1, "timeout"),
    (RuntimeError("private payload"), 3, "rpc_limit"),
    (RuntimeError("private payload"), 1, "failed"),
]


def test_compute_usage_validation_and_failure_records(tmp_path: Path) -> None:
    # completed usage cannot exceed RPC limit
    with pytest.raises(ValueError, match="exceeds RPC"):
        compute_usage(
            wall_clock_limit_seconds=10, elapsed_seconds=1, max_rpc_calls=2, rpc_calls=3, outcome="completed"
        )

    # corrupt compute usage rejected
    for field, value in CORRUPT_USAGE_CASES:
        usage = compute_usage(
            wall_clock_limit_seconds=10, elapsed_seconds=1, max_rpc_calls=2, rpc_calls=1, outcome="completed"
        )
        usage[field] = value
        with pytest.raises(ValueError):
            validate_compute_usage(usage)

    # failure usage survives without success artifacts
    for index, (error, calls, outcome) in enumerate(FAILURE_USAGE_CASES):
        results = tmp_path / f"failure-{index}"
        results.mkdir()
        with pytest.raises(type(error)):
            with record_compute_failure(
                results,
                wall_clock_limit_seconds=10,
                started_at=time.monotonic(),
                max_rpc_calls=2,
                broker=SimpleNamespace(calls=calls),
            ):
                raise error
        text = (results / "evaluation_failure.json").read_text()
        assert "private payload" not in text, outcome
        usage = json.loads(text)
        assert usage["outcome"] == outcome
        assert usage["rpc_calls"] == calls, outcome
        assert not (results / "metrics.json").exists(), outcome

    # failure usage does not follow symlinks
    results = tmp_path / "symlink"
    results.mkdir()
    target = results / "private"
    target.write_text("unchanged")
    (results / "evaluation_failure.json").symlink_to(target)
    with pytest.raises(TimeoutError):
        with record_compute_failure(
            results, wall_clock_limit_seconds=10, started_at=time.monotonic(), max_rpc_calls=2
        ):
            raise TimeoutError
    assert target.read_text() == "unchanged"


FINAL_FAILURE_DIAGNOSTIC_CASES = [
    ("timeout", _raise(subprocess.TimeoutExpired("private command", 1)), "timeout"),
    ("invalid-results", _launch_failure("", 0), "failed"),
    ("exit-failure", _launch_failure("", 1), "failed"),
]


def test_final_failures_retain_compute_diagnostics(tmp_path: Path) -> None:
    for name, run, outcome in FINAL_FAILURE_DIAGNOSTIC_CASES:
        task, investigation, output = _prepare(tmp_path / name)
        with pytest.raises(FinalEvaluationError):
            _evaluate(task, investigation, output, run=run)
        assert json.loads((output / "evaluation_failure.json").read_text())["outcome"] == outcome, name
        assert not (output / "final_manifest.json").exists(), name


# --- public assets reach trusted evaluators without entering candidate containers -----------


@pytest.fixture
def public_task(tmp_path: Path):
    task = _write_contract_task(tmp_path)
    _edit_contract(task, lambda contract: contract["resources"].update(public_artifact_bundle="test-bundle"))
    root = tmp_path / "public-assets"
    root.mkdir(mode=0o700)
    (root / "test-bundle").mkdir(mode=0o500)
    broker = tmp_path / "broker"
    broker.mkdir()
    return task, load_task_contract(task / "task_contract.json"), root, broker


def _assert_private_assets(evaluator: list[str], candidate: list[str], root: Path) -> None:
    assert f"type=bind,src={root / 'test-bundle'},dst=/public-artifacts/test-bundle,readonly" in evaluator
    assert "AGENTIC_PUBLIC_ASSETS_DIR=/public-artifacts" in evaluator
    assert not any("public-assets" in item or "public-artifacts" in item for item in candidate)
    assert not any("AGENTIC_PUBLIC_ASSETS_DIR" in item for item in candidate)


class CommandsCaptured(BaseException):
    """Stop after real command construction, before any container launch."""


def test_search_runtime_and_canary_mount_public_assets_only_for_evaluator(
    public_task, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.investigation import task_runtime
    from src.task_drafting import production_canary
    from src.variants_pipeline import host_eval

    task, contract, root, broker = public_task
    environment = {"AGENTIC_PUBLIC_ASSETS_DIR": str(root)}

    # search commands mount public bundle only for evaluator
    evaluator, candidate = host_eval._brokered_search_commands(
        task_source_dir=task, evaluator_source_dir=task, program_dir=task / "program_0",
        entrypoint=task / "program_0/src/program.py", results_dir=tmp_path / "results",
        broker_dir=broker, sandbox_image=PINNED_IMAGE, contract=contract,
        environment=environment,
    )
    _assert_private_assets(evaluator, candidate, root)

    # runtime and canary mount public assets only for evaluator
    for entrypoint in ["runtime", "canary_search", "canary_final"]:
        module = task_runtime if entrypoint == "runtime" else production_canary
        commands: dict[str, list[str]] = {}

        def evaluator_command(_commands=commands, **kwargs: Any) -> list[str]:
            _commands["evaluator"] = _evaluation_command(**kwargs)
            return _commands["evaluator"]

        def candidate_command(_commands=commands, **kwargs: Any) -> list[str]:
            _commands["candidate"] = _candidate_command(**kwargs)
            raise CommandsCaptured()

        with monkeypatch.context() as m:
            m.setattr(module, "_prepare_task_broker_directory", lambda *a, **k: broker)
            m.setattr(module, "_evaluation_command", evaluator_command)
            m.setattr(module, "_candidate_command", candidate_command)
            with pytest.raises(CommandsCaptured):
                if entrypoint == "runtime":
                    task_runtime._run_contracted_baseline(
                        task_dir=task, runtime_venv=task / ".venv", image_id=PINNED_IMAGE,
                        contract=contract, entrypoint="src/program.py", deadline=time.monotonic() + 60,
                        environment=environment, run=subprocess.run,
                    )
                else:
                    m.setattr(production_canary, "resolve_immutable_image", lambda image: image)
                    m.setattr(
                        production_canary, "entrypoint_relpath_from_spec", lambda path: "src/program.py",
                    )
                    plane: Literal["search", "final"] = "search" if entrypoint == "canary_search" else "final"
                    production_canary.run_production_entry_canary(
                        task_dir=task, source_task_dir=task, container_image=PINNED_IMAGE,
                        plane=plane,
                        caller_deadline=time.monotonic() + 60, environment=environment,
                    )
        _assert_private_assets(commands["evaluator"], commands["candidate"], root)


def test_variant_state_machine_preserves_host_asset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.variants.test_variants_execute import _run_variant_with_mock, _write_complete_rubric

    root = str(tmp_path / "public-assets")
    monkeypatch.setenv("AGENTIC_PUBLIC_ASSETS_DIR", root)
    environments: list[dict[str, str]] = []

    def verify(bundle: Path):
        _write_complete_rubric(bundle / "program_0", passed=True)
        return "verified", 0

    result, _, evaluations = _run_variant_with_mock(
        tmp_path, [verify], max_evals=1, kind="control", host_environments=environments,
    )
    assert evaluations == 1 and result.status == "completed"
    assert environments[0]["AGENTIC_PUBLIC_ASSETS_DIR"] == root
