"""GPU callers: cleanup classification, interruption handling, launch registration, and diagnostics."""

from __future__ import annotations

import errno
import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import Mock

import pytest

from src.final_evaluation import runner
from src.final_evaluation.candidate_broker import (
    CandidateBroker, CandidateBrokerInfrastructureError, CandidateContainerCleanupError,
    ContainerMetadataCleanupError,
)
from src.task_drafting import checks, production_canary as canary
from src.variants_pipeline import host_eval
from src.investigation import task_runtime
from src.investigation.bounded_process import BoundedProcessResult, ProcessCleanupWitness
from src.investigation.container_gpus import (
    GPU_ALLOCATION_BUSY_CATEGORY, GPU_ALLOCATION_INVALID_CATEGORY, GPU_ALLOCATION_QUARANTINED_CATEGORY,
    GpuLeaseError, GpuLeaseQuarantined, GpuLeaseTimeout,
)
from src.investigation.task_contract import TaskContract, load_task_contract
from src.investigation.task_runtime import TaskRuntimePreparationResult as RuntimeResult
from tests.program_testutils import make_program_dir
from tests.final_evaluation.test_final_evaluation import (
    _RecordingGpuLease, _write_contract_task, _write_investigation, run_final_evaluation,
)
from tests.task_drafting.test_task_drafting_checks import _ObservedGpuLease, _settings, _stub_final_topology, _write_scaffold
from tests.task_drafting.test_task_drafting_production_canary import _task

TASK_SH = Path(__file__).parents[2] / "src/scripts/task.sh"
GPU_ENV = {"AGENT_CONTAINER_GPUS": "device=GPU-aaaa"}
GPU_CATEGORIES = (
    GPU_ALLOCATION_BUSY_CATEGORY, GPU_ALLOCATION_INVALID_CATEGORY, GPU_ALLOCATION_QUARANTINED_CATEGORY,
)
HANDLERS = (task_runtime._RuntimeReportDiagnosticHandler, checks._RuntimeDiagnosticHandler)
FAILURES = [
    ("metadata", ContainerMetadataCleanupError, False),
    ("other-infrastructure", CandidateBrokerInfrastructureError, False),
    ("unverified-container", CandidateContainerCleanupError, True),
]
INFRASTRUCTURE_FAILED = task_runtime._CheckedCommandStatus.INFRASTRUCTURE_FAILED


def _sub(tmp_path: Path, name: str) -> Path:
    """Private directory for one merged test section."""
    path = tmp_path / name
    path.mkdir()
    return path


def _raise(error: BaseException) -> NoReturn:
    raise error


def _completed(command, **_kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, "", "")


def _bounded_ok(*_args: object, **_kwargs: object) -> BoundedProcessResult:
    return BoundedProcessResult(0, b"", b"")


def _no_launch(*_args: object, **_kwargs: object) -> NoReturn:
    pytest.fail("no evaluator launch expected")


def _gpu_contract(
    task: Path, *planes: str, gpu_count: int = 1, evaluator_gpus: int = 1, candidate_gpus: int = 0,
) -> TaskContract:
    path = task / "task_contract.json"
    payload = json.loads(path.read_text())
    for plane in planes or ("search",):
        payload["compute"][plane].update(
            accelerator="gpu", accelerator_model="H100", gpu_count=gpu_count,
            evaluator_gpu_count=evaluator_gpus, candidate_gpu_count=candidate_gpus,
        )
    path.write_text(json.dumps(payload))
    return load_task_contract(path)


def _stub_baseline(monkeypatch: pytest.MonkeyPatch, broker_dir: Path, lease=None) -> None:
    monkeypatch.setattr(task_runtime, "_prepare_task_broker_directory", lambda *a, **kw: broker_dir)
    monkeypatch.setattr(task_runtime, "_evaluation_command", lambda **kw: ["eval"])
    monkeypatch.setattr(task_runtime, "_candidate_command", lambda **kw: ["candidate"])
    if lease is not None:
        monkeypatch.setattr(task_runtime, "acquire_gpu_allocation_lease", lambda *a, **kw: lease)


def _baseline(task: Path, contract: TaskContract, run, **overrides):
    kwargs = dict(
        task_dir=task, runtime_venv=task / ".venv", image_id="image", contract=contract,
        entrypoint="main.py", deadline=time.monotonic() + 1000, environment=dict(GPU_ENV), run=run,
    )
    return task_runtime._run_contracted_baseline(**(kwargs | overrides))


def _record_removals(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    removals: list[list[str]] = []

    def remove(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        removals.append(command)
        return _completed(command)

    monkeypatch.setattr(subprocess, "run", remove)
    return removals


def _broker_with_cleanup_error(error: CandidateBrokerInfrastructureError | None) -> type:
    class Broker:
        calls = 1
        calls_before_first_reset = 0
        reset_seeds = (2,)
        peer_disconnects = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> "Broker":
            return self

        def __exit__(self, *_args: object) -> None:
            if error is not None:
                raise error

    return Broker


def _cleanup_broker(error_type: type | None) -> type:
    return _broker_with_cleanup_error(
        None if error_type is None else error_type("candidate container cleanup failed: injected"),
    )


def _stub_host_eval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lease, broker: type) -> None:
    monkeypatch.setattr(host_eval, "CandidateBroker", broker)
    monkeypatch.setattr(host_eval, "acquire_gpu_allocation_lease", lambda *a, **kw: lease)
    monkeypatch.setattr(host_eval, "_ensure_broker_recovery_root", lambda: tmp_path)
    monkeypatch.setattr(host_eval, "_uses_image_managed_runtime", lambda **kw: True)
    monkeypatch.setattr(host_eval, "run_bounded_process", _bounded_ok)


def _brokered_search(task: Path, program: Path, contract: TaskContract, **overrides: Any):
    kwargs: dict[str, Any] = dict(
        task_source_dir=task, evaluator_source_dir=task, program_dir=program,
        entrypoint=program / "src/program.py", python_bin=task / ".venv/bin/python",
        sandbox_image="image", timeout_seconds=100, environment={},
        sandbox_gpus="device=GPU-aaaa", contract=contract,
    )
    return host_eval._evaluate_brokered_search_locked(**(kwargs | overrides))


def _final_evaluation(tmp_path: Path, task: Path):
    python = task / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    investigation, _ = _write_investigation(tmp_path)
    return run_final_evaluation(
        task_dir=task, investigation_results_dir=investigation, output_dir=tmp_path / "final",
        python_bin=Path(sys.executable), container_image="image", environment=dict(GPU_ENV), run=_completed,
    )


def _inject_evaluator_cid_unlink_failure(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    attempted: list[Path] = []
    original_unlink = Path.unlink

    def unlink(path: Path, missing_ok: bool = False) -> None:
        if path.name == "evaluator.cid" and path.exists():
            attempted.append(path)
            raise OSError(errno.EACCES, "cannot remove evaluator metadata")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    return attempted


def _final_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broker: type, *, removed: bool):
    _write_scaffold(tmp_path)
    lease = _ObservedGpuLease([])
    _stub_final_topology(tmp_path, monkeypatch, lease)
    monkeypatch.setattr(checks, "CandidateBroker", broker)
    monkeypatch.setattr(checks, "run_bounded_process", _bounded_ok)
    monkeypatch.setattr(checks, "_remove_container_before_deadline", lambda *a, **kw: removed)
    with pytest.raises(checks.DraftGateFailure) as failure:
        checks._final_topology_smoke(_settings(tmp_path), "digest", "src/program.py")
    assert not lease.active
    return lease, failure.value


# --------------------------------------------------------------------------- broker cleanup classification


def test_final_smoke_broker_cleanup_classifies_gpu_uncertainty_and_metadata_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # broker cleanup failures quarantine only unverified container shutdown
    for name, error_type, quarantined in FAILURES:
        with monkeypatch.context() as m:
            lease, failure = _final_smoke(_sub(tmp_path, name), m, _cleanup_broker(error_type), removed=True)
        assert failure.status == "infrastructure_failure", name
        assert lease.quarantined is quarantined, name
    # an uncertain evaluator survives a broker metadata failure
    with monkeypatch.context() as m:
        broker = _broker_with_cleanup_error(ContainerMetadataCleanupError("broker metadata remains"))
        lease, _ = _final_smoke(_sub(tmp_path, "uncertain-evaluator"), m, broker, removed=False)
    assert lease.quarantined
    # an evaluator.cid unlink failure does not quarantine
    base = _sub(tmp_path, "cid-unlink")
    _write_scaffold(base)
    lease = _ObservedGpuLease([])
    _stub_final_topology(base, monkeypatch, lease)
    cid = base / "broker/evaluator.cid"
    cid.write_text("a" * 64)
    attempted = _inject_evaluator_cid_unlink_failure(monkeypatch)
    monkeypatch.setattr(checks, "run_bounded_process", _bounded_ok)
    monkeypatch.setattr(checks, "_remove_container_before_deadline", lambda *a, **kw: True)
    with pytest.raises(checks.DraftGateFailure) as failure:
        checks._final_topology_smoke(_settings(base), "digest", "src/program.py")
    assert failure.value.status == "infrastructure_failure"
    assert attempted == [cid]
    assert not lease.quarantined
    assert not lease.active


def test_baseline_broker_cleanup_classifies_gpu_uncertainty_and_cid_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # broker cleanup failures quarantine only unverified container shutdown
    for name, error_type, quarantined in FAILURES:
        base = _sub(tmp_path, name)
        task = _write_contract_task(base)
        contract = _gpu_contract(task, "search")
        lease = _RecordingGpuLease()
        broker_dir = base / "broker"
        broker_dir.mkdir()
        with monkeypatch.context() as m:
            _stub_baseline(m, broker_dir, lease)
            m.setattr(task_runtime, "CandidateBroker", _cleanup_broker(error_type))
            m.setattr(task_runtime, "_run_checked", lambda *a, **kw: task_runtime._CheckedCommandResult(
                task_runtime._CheckedCommandStatus.SUCCEEDED, process=None, cleanup_succeeded=True,
            ))
            result = _baseline(task, contract, run=lambda *a, **kw: pytest.fail("no subprocess expected"))
        assert result.status is INFRASTRUCTURE_FAILED, name
        assert lease.quarantined is quarantined, name
        assert lease.released
    # a checked success with an evaluator.cid unlink failure does not quarantine
    base = _sub(tmp_path, "cid-unlink")
    task = _write_contract_task(base)
    contract = _gpu_contract(task, "search")
    lease = _RecordingGpuLease()
    broker_dir = base / "broker"
    broker_dir.mkdir()
    cid = broker_dir / "evaluator.cid"
    cid.write_text("a" * 64)
    attempted = _inject_evaluator_cid_unlink_failure(monkeypatch)
    _stub_baseline(monkeypatch, broker_dir, lease)
    monkeypatch.setattr(task_runtime, "CandidateBroker", _cleanup_broker(None))
    result = _baseline(task, contract, run=_completed)
    assert result.status is INFRASTRUCTURE_FAILED
    assert attempted == [cid]
    assert not lease.quarantined
    assert lease.released


def test_final_evaluation_and_variants_broker_cleanup_classify_gpu_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # final evaluation
    for name, error_type, quarantined in FAILURES:
        base = _sub(tmp_path, f"final-{name}")
        task = _write_contract_task(base)
        _gpu_contract(task, "final")
        lease = _RecordingGpuLease()
        with monkeypatch.context() as m:
            m.setattr(runner, "CandidateBroker", _cleanup_broker(error_type))
            m.setattr(runner, "acquire_gpu_allocation_lease", lambda *a, **kw: lease)
            m.setattr(runner, "_ensure_broker_recovery_root", lambda: base)
            with pytest.raises(runner.FinalEvaluationInfrastructureError):
                _final_evaluation(base, task)
        assert lease.quarantined is quarantined, name
        assert lease.released
    # variants host search
    for name, error_type, quarantined in FAILURES:
        base = _sub(tmp_path, f"variants-{name}")
        task = _write_contract_task(base)
        contract = _gpu_contract(task, "search")
        program = base / "program"
        make_program_dir(program)
        lease = _RecordingGpuLease()
        with monkeypatch.context() as m:
            _stub_host_eval(m, base, lease, _cleanup_broker(error_type))
            m.setattr(host_eval, "_force_remove_container", lambda *a, **kw: None)
            with pytest.raises(CandidateBrokerInfrastructureError):
                _brokered_search(task, program, contract)
        assert lease.quarantined is quarantined, name
        assert lease.released


def test_final_container_removal_reports_metadata_failure_and_still_attempts_named_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a removed container whose cid cannot be unlinked is a metadata failure
    cid = tmp_path / "evaluator.cid"
    cid.write_text("a" * 64)
    with monkeypatch.context() as m:
        calls = _record_removals(m)
        original_unlink = Path.unlink

        def unlink(path: Path, missing_ok: bool = False) -> None:
            if path == cid:
                raise OSError(errno.EACCES, "cannot unlink metadata")
            original_unlink(path, missing_ok=missing_ok)

        m.setattr(Path, "unlink", unlink)
        with pytest.raises(ContainerMetadataCleanupError):
            runner._force_remove_container(cid)
    assert calls == [["docker", "rm", "-f", "a" * 64]]
    assert cid.exists()
    # named removal is always attempted after a metadata failure
    monkeypatch.setattr(runner, "_force_remove_container", lambda *a: _raise(
        ContainerMetadataCleanupError("verified removal; metadata remains"),
    ))
    for named_fails in [False, True]:
        names: list[str] = []

        def remove_named(name: str, names=names, named_fails=named_fails) -> None:
            names.append(name)
            if named_fails:
                raise runner.FinalEvaluationError("named container still running")

        if named_fails:
            assert runner._cleanup_evaluator_container(tmp_path / "cid", "evaluator", remove_named) is False
        else:
            with pytest.raises(ContainerMetadataCleanupError):
                runner._cleanup_evaluator_container(tmp_path / "cid", "evaluator", remove_named)
        assert names == ["evaluator"], named_fails


def test_evaluator_cid_metadata_failure_stays_infrastructure_without_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Broker:
        calls = 1
        calls_before_first_reset = 0
        reset_seeds = (2,)
        peer_disconnects = 0

        def __init__(self, _command: object, broker_dir: Path, **_kwargs: object) -> None:
            broker_dir.parent.mkdir(mode=0o700, exist_ok=True)
            broker_dir.mkdir(mode=0o700, exist_ok=True)
            (broker_dir / "evaluator.cid").write_text("a" * 64)

        def __enter__(self) -> "Broker":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    for consumer in ["final", "variants"]:
        base = _sub(tmp_path, consumer)
        task = _write_contract_task(base)
        contract = _gpu_contract(task, "final" if consumer == "final" else "search")
        lease = _RecordingGpuLease()
        module = runner if consumer == "final" else host_eval
        with monkeypatch.context() as m:
            attempted = _inject_evaluator_cid_unlink_failure(m)
            removals = _record_removals(m)
            m.setattr(module, "CandidateBroker", Broker)
            m.setattr(module, "acquire_gpu_allocation_lease", lambda *a, **kw: lease)
            m.setattr(module, "_ensure_broker_recovery_root", lambda: base)
            if consumer == "final":
                with pytest.raises(runner.FinalEvaluationInfrastructureError):
                    _final_evaluation(base, task)
            else:
                program = base / "program"
                make_program_dir(program)
                m.setattr(host_eval, "_uses_image_managed_runtime", lambda **kw: True)
                m.setattr(host_eval, "run_bounded_process", _bounded_ok)
                with pytest.raises(ContainerMetadataCleanupError):
                    _brokered_search(task, program, contract)
        assert len(attempted) == 1, consumer
        assert removals == [["docker", "rm", "-f", "a" * 64]]
        assert not lease.quarantined
        assert lease.released


# --------------------------------------------------------------------------- interrupted baseline cleanup


def _baseline_broker(base: Path, monkeypatch: pytest.MonkeyPatch):
    """Contracted GPU baseline whose broker and lease are observable; returns (task, contract, broker, lease)."""
    task = _write_contract_task(base)
    contract = _gpu_contract(task, "search")
    broker_dir = base / "broker"
    broker = CandidateBroker([], broker_dir)
    lease = _RecordingGpuLease()
    _stub_baseline(monkeypatch, broker_dir, lease)
    monkeypatch.setattr(task_runtime, "CandidateBroker", lambda *a, **kw: broker)
    return task, contract, broker, lease


@pytest.fixture
def baseline_broker(tmp_path, monkeypatch):
    return _baseline_broker(tmp_path, monkeypatch)


@pytest.mark.parametrize("interruption_type", [KeyboardInterrupt, SystemExit, RuntimeError])
@pytest.mark.parametrize("cleanup_succeeded", [True, False])
@pytest.mark.parametrize("container_id", ["a" * 64, "", None])
@pytest.mark.parametrize("launcher_reaped", [True, False])
@pytest.mark.parametrize("pending_broker_failure", [False, True])
def test_baseline_interruption_quarantines_only_unverified_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, baseline_broker,
    interruption_type: type[BaseException], cleanup_succeeded: bool, container_id: str | None,
    launcher_reaped: bool, pending_broker_failure: bool,
) -> None:
    task, contract, broker, lease = baseline_broker
    interruption = interruption_type()
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(
        command, 0 if cleanup_succeeded else 1, "", "cleanup failed",
    ))

    def interrupt(*_args: object, cleanup_witness: ProcessCleanupWitness, **_kwargs: object) -> NoReturn:
        if container_id is not None:
            (tmp_path / "broker" / "evaluator.cid").write_text(container_id)
        cleanup_witness.reaped = launcher_reaped
        if pending_broker_failure:
            broker._error = CandidateBrokerInfrastructureError("candidate stderr output limit exceeded")
        raise interruption

    monkeypatch.setattr(task_runtime, "run_bounded_process", interrupt)
    with pytest.raises(interruption_type) as raised:
        _baseline(task, contract, run=subprocess.run)
    assert raised.value is interruption
    assert lease.quarantined is not (cleanup_succeeded and bool(container_id) and launcher_reaped)
    assert lease.released


def test_broker_bind_failure_and_directory_collision_are_bounded_without_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a bind failure is bounded infrastructure and closes the socket
    broker = CandidateBroker([], tmp_path / "broker")
    opened: list[socket.socket] = []

    class FailingSocket(socket.socket):
        def bind(self, address: object) -> None:
            opened.append(self)
            raise OSError(errno.ENOSPC, "private content", "/private/socket")

    with monkeypatch.context() as m:
        m.setattr(socket, "socket", FailingSocket)
        with pytest.raises(CandidateBrokerInfrastructureError) as raised:
            broker.__enter__()
    assert opened[0].fileno() == -1
    assert "ENOSPC" in str(raised.value)
    assert "private" not in str(raised.value)
    assert broker.candidate_process_starts == 0
    # a broker directory collision does not quarantine
    base = _sub(tmp_path, "collision")
    task = _write_contract_task(base)
    contract = _gpu_contract(task, "search")
    broker_dir = base / "broker"
    broker_dir.mkdir()
    retained = broker_dir / "operator-owned"
    retained.write_text("keep")
    lease = _RecordingGpuLease()
    _stub_baseline(monkeypatch, broker_dir, lease)
    result = _baseline(task, contract, run=_no_launch)
    assert result.status is INFRASTRUCTURE_FAILED
    assert not lease.quarantined
    assert lease.released
    assert retained.read_text() == "keep"


def test_interrupted_baseline_quarantines_unverified_shutdown_and_survives_pending_broker_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # an interrupted baseline quarantines an unverified candidate shutdown
    base = _sub(tmp_path, "unverified")
    with monkeypatch.context() as m:
        task, contract, broker, lease = _baseline_broker(base, m)
        removals = _record_removals(m)
        m.setattr(broker, "_stop_process", Mock(side_effect=OSError(errno.EIO, "injected")))

        def interrupt(*_args: object, cleanup_witness: ProcessCleanupWitness, **_kwargs: object) -> NoReturn:
            (base / "broker" / "evaluator.cid").write_text("a" * 64)
            cleanup_witness.reaped = True
            broker._process = Mock()
            raise KeyboardInterrupt

        m.setattr(task_runtime, "run_bounded_process", interrupt)
        with pytest.raises(KeyboardInterrupt):
            _baseline(task, contract, run=subprocess.run)
    assert lease.quarantined
    assert lease.released
    assert not broker.shutdown_verified
    assert removals == [["docker", "rm", "--force", "a" * 64]]
    # a pre-evaluator interruption survives a pending broker failure
    task, contract, broker, lease = _baseline_broker(_sub(tmp_path, "pending"), monkeypatch)
    interruption = KeyboardInterrupt()

    def interrupt_observation(_broker: CandidateBroker) -> NoReturn:
        broker._error = CandidateBrokerInfrastructureError("pending infrastructure failure")
        raise interruption

    monkeypatch.setattr(task_runtime, "observe_qualification_broker", interrupt_observation)
    with pytest.raises(KeyboardInterrupt) as raised:
        _baseline(task, contract, run=_no_launch)
    assert raised.value is interruption
    assert lease.quarantined
    assert lease.released


# --------------------------------------------------------------------------- production broker callers


class EvaluationFinished(Exception):
    """Stop the host-search test before unrelated result publication."""


class RecordingLease:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.active = False
        self.prepared = False
        self.release_failure = False

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *_args: object) -> None:
        self.events.append("release")
        self.active = False
        if self.release_failure:
            raise GpuLeaseQuarantined("owned cleanup remains unverified")

    def prepare_runtime(self) -> None:
        assert self.active and not self.prepared
        self.prepared = True
        self.events.append("prepare")

    def wrap_command(self, command: list[str]) -> list[str]:
        assert self.active and self.prepared
        assert command[:2] == ["docker", "run"]
        self.events.append("wrap-" + command[-1])
        return ["owned-launch", *command]

    def quarantine(self) -> None:
        assert self.active


def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer: str, gpu: bool):
    task = _task(tmp_path)
    contract = _gpu_contract(task, "search", "final") if gpu else load_task_contract(task / "task_contract.json")
    module = {"smoke": checks, "canary": canary, "search": host_eval}[consumer]
    events: list[str] = []
    lease = RecordingLease(events)
    expected_prefix = ["owned-launch", "docker", "run"] if gpu else ["docker", "run"]

    class Broker:
        calls = 1
        calls_before_first_reset = 0
        peer_disconnects = 0

        def __init__(self, command, directory, **kwargs):
            assert command == [*expected_prefix, "candidate"]
            assert not gpu or lease.prepared
            self.reset_seeds = kwargs["expected_reset_seeds"]
            self.completed_reset_seeds = self.reset_seeds
            events.append("broker")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            events.append("broker-stop")

    def evaluate(command, **kwargs):
        assert command == [*expected_prefix, "evaluator"]
        assert not gpu or lease.prepared
        events.append("evaluate")
        return BoundedProcessResult(0, b"", b"")

    def acquire(*args, **kwargs):
        assert gpu, "CPU work acquired a GPU lease"
        return lease

    monkeypatch.setattr(module, "acquire_gpu_allocation_lease", acquire)
    monkeypatch.setattr(module, "CandidateBroker", Broker)
    monkeypatch.setattr(module, "run_bounded_process", evaluate)
    monkeypatch.setenv("AGENT_CONTAINER_GPUS", "device=GPU-aaaa")
    if consumer == "search":
        monkeypatch.setattr(module, "_ensure_broker_recovery_root", lambda: tmp_path)
        monkeypatch.setattr(module, "_uses_image_managed_runtime", lambda **kw: True)
        monkeypatch.setattr(module, "_recover_broker_directory", lambda *a: None)
        monkeypatch.setattr(
            module, "_brokered_search_commands",
            lambda **kw: (["docker", "run", "evaluator"], ["docker", "run", "candidate"]),
        )
        monkeypatch.setattr(module, "_force_remove_container", lambda *a: None)
        monkeypatch.setattr(module, "_validate_search_result_schema", lambda *a: _raise(EvaluationFinished()))
    else:
        monkeypatch.setattr(module, "resolve_immutable_image", lambda image: image)
        monkeypatch.setattr(module, "_prepare_task_broker_directory", lambda *a, **kw: tmp_path / "broker")
        monkeypatch.setattr(module, "_evaluation_command", lambda **kw: ["docker", "run", "evaluator"])
        monkeypatch.setattr(module, "_candidate_command", lambda **kw: ["docker", "run", "candidate"])
        monkeypatch.setattr(module, "_remove_container_before_deadline", lambda *a: True)
        if consumer == "smoke":
            monkeypatch.setattr(module, "_require_score", lambda *a: None)

    def run():
        if consumer == "smoke":
            return checks._final_topology_smoke(_settings(task), "digest", "src/program.py")
        if consumer == "canary":
            return canary.run_production_entry_canary(
                task_dir=task, source_task_dir=task, container_image="image", plane="search",
                caller_deadline=time.monotonic() + 10,
            )
        return _brokered_search(
            task, task / "program_0", contract, timeout_seconds=10,
            sandbox_gpus="device=GPU-aaaa" if gpu else None,
        )

    return run, lease, events


def test_broker_callers_register_gpu_launches_preserve_cpu_commands_and_classify_release_quarantine(
    tmp_path, monkeypatch,
):
    consumers = ["smoke", "canary", "search"]
    # GPU launches are registered through the lease; CPU commands are preserved untouched
    for consumer in consumers:
        for gpu in [False, True]:
            name = f"{consumer}-{'gpu' if gpu else 'cpu'}"
            with monkeypatch.context() as m:
                run, lease, events = configure(_sub(tmp_path, name), m, consumer, gpu)
                if consumer == "search":
                    with pytest.raises(EvaluationFinished):
                        run()
                else:
                    result = run()
                    if consumer == "canary":
                        assert result.status == "pass", result
            if gpu:
                assert events[:3] == ["prepare", "wrap-evaluator", "wrap-candidate"], name
                assert events.count("prepare") == 1
                assert events[-1] == "release"
            else:
                assert not lease.prepared, name
                assert "release" not in events
            assert events.index("broker") < events.index("evaluate") < events.index("broker-stop"), name
    # a release quarantine remains a GPU infrastructure failure
    for consumer in consumers:
        with monkeypatch.context() as m:
            run, lease, events = configure(_sub(tmp_path, f"{consumer}-release"), m, consumer, True)
            lease.release_failure = True
            if consumer == "smoke":
                with pytest.raises(checks.DraftGateFailure) as caught:
                    run()
                assert caught.value.category == GPU_ALLOCATION_QUARANTINED_CATEGORY
                assert caught.value.status == "infrastructure_failure"
            elif consumer == "search":
                with pytest.raises(CandidateBrokerInfrastructureError, match=GPU_ALLOCATION_QUARANTINED_CATEGORY):
                    run()
            else:
                result = run()
                assert GPU_ALLOCATION_QUARANTINED_CATEGORY in result.category
                assert result.status == "infrastructure_failure"
                assert "entry_status=pass" in result.diagnostic
        assert events[-1] == "release", consumer


# --------------------------------------------------------------------------- GPU failure diagnostics


def _record(message: str, extra: dict[str, object] | None = None) -> logging.LogRecord:
    return task_runtime.logger.makeRecord(
        task_runtime.logger.name, logging.ERROR, __file__, 0, message, (), None, extra=extra,
    )


def _fake_prepare(monkeypatch: pytest.MonkeyPatch, result: RuntimeResult, extra: dict[str, object]) -> None:
    def prepare(*_args: object, **_kwargs: object) -> RuntimeResult:
        task_runtime.logger.error("GPU allocation unavailable", extra=extra)
        return result

    monkeypatch.setattr(task_runtime, "prepare_task_runtime_result", prepare)
    monkeypatch.setattr(checks, "prepare_task_runtime_result", prepare)


def _gate_failure(task: Path, monkeypatch: pytest.MonkeyPatch) -> checks.DraftGateFailure:
    monkeypatch.setattr(checks, "resolve_immutable_image", lambda _image: "image")
    settings = checks.GateSettings(task, task, True, "image", "none", 10)
    with pytest.raises(checks.DraftGateFailure) as failure:
        checks._prepare_runtime(settings, "digest")
    return failure.value


def _shell_functions(*names: str) -> str:
    source = TASK_SH.read_text()
    return "\n".join(
        f"{name}() {{" + source.split(f"{name}() {{", 1)[1].split("\n}\n", 1)[0] + "\n}" for name in names
    )


def test_handlers_accept_only_trusted_structured_gpu_categories() -> None:
    for handler_type in HANDLERS:
        # trusted categories are preserved
        for category in GPU_CATEGORIES:
            handler = handler_type()
            handler.emit(_record("GPU allocation unavailable", {"task_runtime_gpu_failure": category}))
            assert handler.gpu_failure_category == category, (handler_type, category)
        # untyped categories are rejected
        untyped: list[object] = [None, "other_failure", 1, True, {}, [], ["gpu_allocation_busy"]]
        for metadata in untyped:
            handler = handler_type()
            handler.emit(_record("candidate printed gpu_allocation_quarantined", {"task_runtime_gpu_failure": metadata}))
            assert handler.gpu_failure_category is None, (handler_type, metadata)
        # plaintext categories cannot spoof a structured failure
        handler = handler_type()
        handler.emit(_record('gpu_allocation_quarantined {"task_runtime_gpu_failure":"gpu_allocation_busy"}'))
        assert handler.gpu_failure_category is None, handler_type
        # a later untyped log cannot erase the GPU failure
        handler = handler_type()
        handler.emit(_record("allocation failed", {"task_runtime_gpu_failure": GPU_ALLOCATION_QUARANTINED_CATEGORY}))
        handler.emit(_record("later noise", {"task_runtime_gpu_failure": "not_a_category"}))
        assert handler.gpu_failure_category == GPU_ALLOCATION_QUARANTINED_CATEGORY, handler_type


def test_runtime_report_preserves_gpu_category_unless_result_or_storage_failure_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # an infrastructure failure preserves the GPU category
    for category in GPU_CATEGORIES:
        with monkeypatch.context() as m:
            _fake_prepare(m, RuntimeResult.INFRASTRUCTURE_FAILURE, {"task_runtime_gpu_failure": category})
            report = task_runtime.prepare_task_runtime_report(_sub(tmp_path, category))
        assert report.result is RuntimeResult.INFRASTRUCTURE_FAILURE
        assert (report.failure_stage, report.failure_category) == ("resource_selection", category)
    # GPU metadata does not override a non-infrastructure result
    for result, expected in [
        (RuntimeResult.READY, None),
        (RuntimeResult.AUTHORED_FAILURE, "runtime_baseline_validation"),
        (RuntimeResult.DEPENDENCY_FAILURE, "runtime_dependencies"),
    ]:
        with monkeypatch.context() as m:
            _fake_prepare(m, result, {"task_runtime_gpu_failure": GPU_ALLOCATION_QUARANTINED_CATEGORY})
            report = task_runtime.prepare_task_runtime_report(_sub(tmp_path, result.name))
        assert report.result is result
        assert report.failure_category == expected, result
    # a storage failure has precedence over the GPU category
    _fake_prepare(monkeypatch, RuntimeResult.INFRASTRUCTURE_FAILURE, {
        "task_runtime_gpu_failure": GPU_ALLOCATION_QUARANTINED_CATEGORY,
        "task_runtime_storage_failure": ("runtime_publication", "storage_space_exhausted"),
    })
    report = task_runtime.prepare_task_runtime_report(_sub(tmp_path, "storage"))
    assert (report.failure_stage, report.failure_category) == ("runtime_publication", "storage_space_exhausted")


def test_runtime_gate_and_shell_keep_gpu_failure_retry_policy_but_ignore_plaintext_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # structured GPU categories reach the gate and only busy allocations are retryable
    for category in GPU_CATEGORIES:
        task = _task(_sub(tmp_path, category))
        with monkeypatch.context() as m:
            _fake_prepare(m, RuntimeResult.INFRASTRUCTURE_FAILURE, {"task_runtime_gpu_failure": category})
            failure = _gate_failure(task, m)
        assert failure.status == "infrastructure_failure"
        assert failure.category == category
        assert failure.stage == "resource_selection"

        retry = subprocess.run(
            ["bash", "-c", _shell_functions("smoke_infrastructure_retryable") + '\nsmoke_infrastructure_retryable "$1"',
             "gpu-retry-test", failure.category],
            capture_output=True, text=True, timeout=5, check=False,
        )
        assert retry.returncode == (0 if category == GPU_ALLOCATION_BUSY_CATEGORY else 1), category
    # a plaintext GPU category does not override the runtime report or the gate
    def prepare(*_args: object, **_kwargs: object) -> RuntimeResult:
        task_runtime.logger.error("gpu_allocation_quarantined")
        return RuntimeResult.INFRASTRUCTURE_FAILURE

    monkeypatch.setattr(task_runtime, "prepare_task_runtime_result", prepare)
    monkeypatch.setattr(checks, "prepare_task_runtime_result", prepare)
    task = _task(_sub(tmp_path, "plaintext"))
    report = task_runtime.prepare_task_runtime_report(task)
    assert report.failure_category == "runtime_bootstrap"
    assert _gate_failure(task, monkeypatch).category == "runtime_bootstrap"


@pytest.mark.parametrize("selector", ["device=/private/malformed", "device=GPU-aaaa"])
@pytest.mark.parametrize("evaluator_gpus,candidate_gpus", [(2, 0), (0, 2), (1, 1)])
def test_invalid_selector_is_terminal_through_runtime_report_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: str, evaluator_gpus: int, candidate_gpus: int,
) -> None:
    task = _write_contract_task(tmp_path)
    contract = _gpu_contract(task, gpu_count=2, evaluator_gpus=evaluator_gpus, candidate_gpus=candidate_gpus)
    monkeypatch.setenv("AGENT_CONTAINER_GPUS", selector)
    monkeypatch.setattr(task_runtime, "_prepare_task_broker_directory", lambda *a, **kw: tmp_path / "broker")

    def forbidden_launch(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("invalid selector must fail before GPU admission or container launch")

    monkeypatch.setattr(task_runtime, "acquire_gpu_allocation_lease", forbidden_launch)

    def prepare(*_args: object, **_kwargs: object) -> RuntimeResult:
        baseline = _baseline(
            task, contract, run=forbidden_launch, entrypoint="src/program.py",
            deadline=time.monotonic() + 60, environment={"AGENT_CONTAINER_GPUS": selector},
        )
        assert baseline.status is INFRASTRUCTURE_FAILED
        return RuntimeResult.INFRASTRUCTURE_FAILURE

    monkeypatch.setattr(task_runtime, "prepare_task_runtime_result", prepare)
    monkeypatch.setattr(checks, "prepare_task_runtime_result", prepare)
    report = task_runtime.prepare_task_runtime_report(task)
    assert (report.failure_stage, report.failure_category) == ("resource_selection", GPU_ALLOCATION_INVALID_CATEGORY)
    assert "/private" not in report.diagnostic

    failure = _gate_failure(task, monkeypatch)
    assert failure.status == "infrastructure_failure"
    assert (failure.stage, failure.category) == ("resource_selection", GPU_ALLOCATION_INVALID_CATEGORY)
    assert "/private" not in failure.diagnostic


def test_baseline_lease_failure_emits_trusted_metadata_without_launching_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    for error_type, category in [
        (GpuLeaseTimeout, GPU_ALLOCATION_BUSY_CATEGORY),
        (GpuLeaseError, GPU_ALLOCATION_INVALID_CATEGORY),
        (GpuLeaseQuarantined, GPU_ALLOCATION_QUARANTINED_CATEGORY),
    ]:
        caplog.clear()
        base = _sub(tmp_path, category)
        task = _write_contract_task(base)
        contract = _gpu_contract(task, "search")

        def acquire(*_args: object, error_type=error_type, **_kwargs: object) -> object:
            raise error_type("private path /secret/runtime or fake gpu_allocation_busy")

        with monkeypatch.context() as m:
            _stub_baseline(m, base / "broker")
            m.setattr(task_runtime, "acquire_gpu_allocation_lease", acquire)
            result = _baseline(
                task, contract, run=lambda *a, **kw: pytest.fail("GPU allocation failed before evaluator launch"),
                deadline=time.monotonic() + 60,
            )
        assert result.status is INFRASTRUCTURE_FAILED, category
        records = [record for record in caplog.records if record.name == task_runtime.logger.name]
        gpu_records = [record for record in records if getattr(record, "task_runtime_gpu_failure", None) == category]
        assert len(gpu_records) == 1, category
        diagnostic = gpu_records[0].getMessage().lower()
        assert "/secret" not in diagnostic
        if category == GPU_ALLOCATION_QUARANTINED_CATEGORY:
            assert "automatic" in diagnostic and "recover" in diagnostic and "inspect" in diagnostic
            assert "safe reuse" in diagnostic
            assert "could not establish" in diagnostic


@pytest.mark.parametrize("category,expected_attempts", [
    (GPU_ALLOCATION_QUARANTINED_CATEGORY, 1),
    (GPU_ALLOCATION_INVALID_CATEGORY, 1),
    (GPU_ALLOCATION_BUSY_CATEGORY, 3),
    ("runtime_bootstrap", 3),
    ("qualification_runtime_failure", 3),
    (None, 3),
])
def test_full_qualification_loop_stops_terminal_gpu_failure_and_preserves_other_retries(
    tmp_path: Path, category: str | None, expected_attempts: int,
) -> None:
    original_report = tmp_path / "original-report.json"
    original_report.write_text(
        json.dumps({"status": "infrastructure_failure", "category": category})
        if category is not None else '{"status":invalid,"category":"gpu_allocation_quarantined"}'
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    script = r'''
set -eu
PYTHON_BIN="$1"
VALIDATION_ROOT="$2"
ORIGINAL_REPORT="$3"
QUALIFICATION_MAX_INFRASTRUCTURE_ATTEMPTS=3
QUALIFICATION_MAX_REPAIRS=0
qualification_attempt=0
calls=0
task_progress_emit() { :; }
record_qualification_failure() { return 0; }
qualification_staged_task() {
  calls=$((calls + 1))
  cat "$ORIGINAL_REPORT" > "$1"
  return 4
}
''' + _shell_functions("qualification_report_field", "qualify_with_infrastructure_retry") + r'''
result=0
qualify_with_infrastructure_retry || result=$?
printf '%s %s\n' "$calls" "$result"
printf '%s\n' "$FAILED_QUALIFICATION_REPORT"
'''
    result = subprocess.run(
        ["bash", "-c", script, "qualification-retry-test", sys.executable, str(reports), str(original_report)],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == f"{expected_attempts} 4"
    last_report = reports / f"qualification_{expected_attempts - 1}.json"
    assert lines[1] == str(last_report)
    assert last_report.read_text() == original_report.read_text()
    assert len(list(reports.glob("qualification_*.json"))) == expected_attempts
