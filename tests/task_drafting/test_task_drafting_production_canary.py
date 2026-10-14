"""Production-entry canaries: broker reset accounting, plane classification, retries, teardown."""

from __future__ import annotations

import errno
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Callable, cast

import pytest

import src.task_drafting.checks as checks_module
import src.task_drafting.production_canary as canary_module
from src.task_drafting.checks import GateSettings, run_gate
from src.task_drafting.production_canary import (
    CanaryStatus,
    ProductionCanaryResult,
    run_production_entry_canaries,
    run_production_entry_canary,
)
from src.final_evaluation.candidate_broker import (
    CandidateBroker,
    CandidateBrokerInfrastructureError,
    CandidateContainerCleanupError,
)
from src.investigation.bounded_process import BoundedProcessResult
from src.investigation.container_gpus import GpuLeaseTimeout
from tests.task_drafting.test_task_alignment import _write_scaffold

IMAGE = "sha256:" + "a" * 64
ECHO_CANDIDATE = """
import json, sys
for raw in sys.stdin:
    request = json.loads(raw)
    print(json.dumps({'ok': True, 'value': request}), flush=True)
"""
GPU_ENVIRONMENT = {"PATH": "/bin", "AGENT_CONTAINER_GPUS": "device=GPU-aaaa"}


def _task(root: Path) -> Path:
    task = root / "task"
    task.mkdir()
    _write_scaffold(task)
    (task / ".venv/lib/python3.12/site-packages/src/investigation").mkdir(parents=True)
    contract_path = task / "task_contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    for plane in ("search", "final"):
        payload["compute"][plane].update(
            accelerator="cpu",
            accelerator_model="none",
            gpu_count=0,
            evaluator_gpu_count=0,
            candidate_gpu_count=0,
            max_candidate_rpc_calls=4,
        )
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    return task


def _update_contract(task: Path, section: str, plane: str, **changes: object) -> None:
    contract_path = task / "task_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract[section][plane].update(changes)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")


def _gpu_task(root: Path) -> Path:
    root.mkdir(exist_ok=True)
    task = _task(root)
    _update_contract(
        task,
        "compute",
        "search",
        accelerator="gpu",
        accelerator_model="H100",
        gpu_count=1,
        evaluator_gpu_count=1,
        candidate_gpu_count=0,
    )
    return task


class _FakeBroker:
    completed_reset_seeds: tuple[int, ...] = ()

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _Lease:
    def __init__(self) -> None:
        self.active = False
        self.released = False
        self.quarantined = False
        self.quarantined_while_active: bool | None = None

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *_args: object) -> None:
        self.active = False
        self.released = True

    def prepare_runtime(self) -> None:
        assert self.active

    def wrap_command(self, command: list[str]) -> list[str]:
        assert self.active
        return command

    def quarantine(self) -> None:
        self.quarantined = True
        self.quarantined_while_active = self.active


def _stub_canary(
    monkeypatch: pytest.MonkeyPatch,
    broker_dir: Path,
    *,
    broker: type | None = _FakeBroker,
    evaluation=lambda **_kwargs: ["eval"],
    candidate=lambda **_kwargs: ["candidate"],
) -> None:
    monkeypatch.setattr(canary_module, "resolve_immutable_image", lambda image: image)
    monkeypatch.setattr(
        canary_module, "_prepare_task_broker_directory", lambda *_a, **_k: broker_dir
    )
    monkeypatch.setattr(canary_module, "_evaluation_command", evaluation)
    monkeypatch.setattr(canary_module, "_candidate_command", candidate)
    if broker is not None:
        monkeypatch.setattr(canary_module, "CandidateBroker", broker)


def _canary(task: Path, **overrides: object) -> ProductionCanaryResult:
    options: dict[str, object] = dict(
        task_dir=task,
        source_task_dir=task,
        container_image=IMAGE,
        plane="search",
        caller_deadline=time.monotonic() + 2,
        environment={"PATH": "/bin"},
    )
    options.update(overrides)
    return run_production_entry_canary(**options)  # type: ignore[arg-type]


def _canaries(task: Path, **overrides: object) -> ProductionCanaryResult:
    options: dict[str, object] = dict(
        task_dir=task,
        source_task_dir=task,
        container_image=IMAGE,
        caller_deadline=time.monotonic() + 2,
        environment={"PATH": "/bin"},
    )
    options.update(overrides)
    return run_production_entry_canaries(**options)  # type: ignore[arg-type]


def _failure(
    status: CanaryStatus, plane: str, category: str, component: str, diagnostic: str
) -> ProductionCanaryResult:
    return ProductionCanaryResult(status, plane, category, component, "production-entry", diagnostic)


def _reset_via_socket(broker: CandidateBroker) -> bytes:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(broker.socket_path))
        client.sendall(b'{"op":"reset","seed":7}\n')
        return client.makefile("rb").readline()


def _wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 1
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.001)


# ---------------------------------------------------------------- broker reset accounting


def test_broker_records_reset_only_after_delivered_successful_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a successful delivered response records the reset
    with CandidateBroker(
        [sys.executable, "-u", "-c", ECHO_CANDIDATE], tmp_path / "ok", expected_reset_seeds=(7,)
    ) as broker:
        response = _reset_via_socket(broker)
        assert json.loads(response) == {"ok": True, "value": {"op": "reset", "seed": 7}}
        _wait_until(lambda: bool(broker.completed_reset_seeds))
        assert broker.completed_reset_seeds == (7,)

    # a candidate error response does not complete the reset
    candidate = """
import sys
for _raw in sys.stdin:
    print('{"ok":false,"value":"reset failed"}', flush=True)
"""
    with CandidateBroker(
        [sys.executable, "-u", "-c", candidate], tmp_path / "error", expected_reset_seeds=(7,)
    ) as broker:
        response = _reset_via_socket(broker)
        assert json.loads(response) == {"ok": False, "value": "reset failed"}
        assert broker.completed_reset_seeds == ()

    # a failed response delivery does not complete the reset
    monkeypatch.setattr(CandidateBroker, "_send_response", lambda *_args: False)
    with CandidateBroker(
        [sys.executable, "-u", "-c", ECHO_CANDIDATE],
        tmp_path / "undelivered",
        expected_reset_seeds=(7,),
    ) as broker:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(broker.socket_path))
            client.sendall(b'{"op":"reset","seed":7}\n')
        _wait_until(lambda: broker.reset_count != 0)
        assert broker.reset_count == 1
        assert broker.completed_reset_seeds == ()


# ---------------------------------------------------------------- single-plane canary


def test_production_canary_classifies_gpu_allocation_exhaustion_as_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # busy GPU allocation
    task = _gpu_task(tmp_path / "busy")
    with monkeypatch.context() as patch:
        _stub_canary(patch, tmp_path / "busy/broker", broker=None)

        def allocation_busy(*_args: object, **_kwargs: object) -> object:
            raise GpuLeaseTimeout("GPU allocation remained busy until the deadline: GPU-aaaa")

        patch.setattr(canary_module, "acquire_gpu_allocation_lease", allocation_busy)
        result = _canary(task, environment=GPU_ENVIRONMENT)
    assert result.status == "infrastructure_failure"
    assert result.category == "search_production_gpu_allocation_busy"
    assert result.component == "harness"

    # GPU admission exhausting the caller deadline
    task = _gpu_task(tmp_path / "admission")
    clock = [100.0]

    def acquire(*_args: object, **_kwargs: object) -> _Lease:
        clock[0] = 101.0
        return _Lease()

    with monkeypatch.context() as patch:
        patch.setattr(canary_module.time, "monotonic", lambda: clock[0])
        _stub_canary(patch, tmp_path / "admission/broker", broker=None)
        patch.setattr(canary_module, "acquire_gpu_allocation_lease", acquire)
        result = _canary(
            task, caller_deadline=101.0, max_duration_seconds=10, environment=GPU_ENVIRONMENT
        )
    assert result.status == "infrastructure_failure"
    assert result.category == "search_production_gpu_allocation_busy"


def test_prepared_cpu_runtime_selector_and_gpu_scope_cleanup_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # prepared CPU runtime uses an empty CUDA selector
    task = _task(tmp_path)
    compatibility: dict[str, object] = {}
    monkeypatch.setattr(checks_module, "resolve_immutable_image", lambda image: image)

    def runtime_is_compatible(_task_dir: Path, **kwargs: object) -> bool:
        compatibility.update(kwargs)
        return True

    monkeypatch.setattr(checks_module, "_runtime_is_compatible", runtime_is_compatible)
    assert checks_module._prepared_runtime_available(
        GateSettings(
            task_dir=task,
            source_task_dir=task,
            containerize=True,
            container_image=IMAGE,
            container_network="none",
            timeout_seconds=300,
        ),
        "evaluator-tree-digest",
    )
    assert compatibility["gpu_evaluation"] is False
    assert compatibility["cuda_visible_devices"] == ""

    # GPU scope quarantines the lease on candidate cleanup failure
    lease = _Lease()
    with pytest.raises(CandidateBrokerInfrastructureError, match="cleanup failed"):
        with canary_module._gpu_allocation_scope(lease):  # type: ignore[arg-type]
            raise CandidateContainerCleanupError(
                "candidate container cleanup failed: daemon unavailable; recovery metadata retained"
            )
    assert lease.quarantined


def test_production_canary_uses_unmarked_exact_plane_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(tmp_path)
    contract_path = task / "task_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["protocol"]["search_reset_sequence"] = [12, 11, 12, 11]
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    broker_instances: list[object] = []
    broker_kwargs: list[dict[str, object]] = []
    evaluator_kwargs: list[dict[str, object]] = []
    candidate_kwargs: list[dict[str, object]] = []
    removed = threading.Event()

    class RecordingBroker(_FakeBroker):
        def __init__(self, _command: list[str], *_args: object, **_kwargs: object) -> None:
            broker_instances.append(self)
            broker_kwargs.append(_kwargs)

    def evaluator_command(**kwargs: object) -> list[str]:
        evaluator_kwargs.append(kwargs)
        return ["docker", "run", "production-evaluator"]

    def candidate_command(**kwargs: object) -> list[str]:
        candidate_kwargs.append(kwargs)
        return ["docker", "run", "production-candidate"]

    def run_process(command: list[str], **_kwargs: object) -> BoundedProcessResult:
        assert command == ["docker", "run", "production-evaluator"]
        setattr(broker_instances[-1], "completed_reset_seeds", (12,))
        assert removed.wait(timeout=1)
        return BoundedProcessResult(137, b"", b"")

    def remove_container(_path: Path, _deadline: float) -> bool:
        removed.set()
        return True

    _stub_canary(
        monkeypatch,
        tmp_path / "broker",
        broker=RecordingBroker,
        evaluation=evaluator_command,
        candidate=candidate_command,
    )
    monkeypatch.setattr(canary_module, "run_bounded_process", run_process)
    monkeypatch.setattr(canary_module, "_remove_container_before_deadline", remove_container)

    result = _canary(task)

    assert result.status == "pass", result
    assert evaluator_kwargs[0]["draft_smoke"] is False
    assert evaluator_kwargs[0]["plane"] == "search"
    assert candidate_kwargs[0]["plane"] == "search"
    assert "AGENTIC_TASK_DRAFT_SMOKE=1" not in evaluator_kwargs[0]
    assert broker_kwargs[0]["expected_reset_seeds"] == (12,)
    assert broker_kwargs[0]["completed_reset_limit"] == 1


def test_production_canary_passes_when_evaluator_immediately_requests_next_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(tmp_path)
    broker_dir = tmp_path / "broker"
    evaluator = f"""
import json, socket

def rpc(request):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect({str(broker_dir / 'candidate.sock')!r})
        client.sendall(json.dumps(request).encode() + b'\\n')
        while client.recv(4096):
            pass

rpc({{'op': 'reset', 'seed': 11}})
try:
    rpc({{'op': 'reset', 'seed': 12}})
except OSError:
    pass
"""
    monkeypatch.setattr(canary_module, "_POLL_SECONDS", 1)
    _stub_canary(
        monkeypatch,
        broker_dir,
        broker=None,
        evaluation=lambda **_kwargs: [sys.executable, "-u", "-c", evaluator],
        candidate=lambda **_kwargs: [sys.executable, "-u", "-c", ECHO_CANDIDATE],
    )
    monkeypatch.setattr(canary_module, "_remove_container_before_deadline", lambda *_a, **_k: True)

    result = _canary(task)

    assert result.status == "pass", result


PRE_RESET_EXIT_CASES = [
    (1, "authored_failure", "search_production_entry_nonzero"),
    (125, "infrastructure_failure", "search_production_container_runtime"),
]


def test_production_canary_classifies_evaluator_exit_candidate_runtime_and_selector_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(tmp_path)

    # pre-reset evaluator exit is classified by return code and stderr is sanitized
    for returncode, expected_status, expected_category in PRE_RESET_EXIT_CASES:
        with monkeypatch.context() as patch:
            _stub_canary(patch, tmp_path / "broker")
            patch.setattr(
                canary_module,
                "run_bounded_process",
                lambda *_a, _code=returncode, **_k: BoundedProcessResult(
                    _code, b"", b"\x1b[31m two-GPU allocation is unavailable"
                ),
            )
            result = _canary(task)
        assert result.status == expected_status, returncode
        assert result.category == expected_category, returncode
        assert "two-GPU allocation is unavailable" in result.diagnostic, returncode
        assert "\x1b" not in result.diagnostic, returncode
        assert len(result.diagnostic.encode("utf-8")) <= 4096, returncode

    # candidate infrastructure diagnostic is preserved verbatim
    class FailingBroker(_FakeBroker):
        def __enter__(self):
            raise CandidateBrokerInfrastructureError("candidate container runtime failed to launch")

    with monkeypatch.context() as patch:
        _stub_canary(patch, tmp_path / "broker", broker=FailingBroker)
        result = _canary(task, plane="final")
    assert result.status == "infrastructure_failure"
    assert result.category == "final_production_candidate_runtime"
    assert result.diagnostic == "candidate container runtime failed to launch"

    # invalid GPU selector is an infrastructure failure
    def invalid_selector(**_kwargs: object) -> list[str]:
        raise canary_module.FinalEvaluationInfrastructureError("gpu_allocation_invalid")

    with monkeypatch.context() as patch:
        _stub_canary(patch, tmp_path / "broker", broker=None, evaluation=invalid_selector)
        result = _canary(task)
    assert result.status == "infrastructure_failure"
    assert result.category == "search_production_gpu_allocation_invalid"


def test_production_canary_timeout_and_interrupt_complete_cleanup_and_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # timeout completes intentional cleanup; failed cleanup quarantines the lease
    for cleanup_succeeds in (True, False):
        root = tmp_path / f"timeout-{cleanup_succeeds}"
        task = _gpu_task(root)
        removed = threading.Event()
        finished = threading.Event()
        lease = _Lease()

        def run_process(*_args: object, **_kwargs: object) -> BoundedProcessResult:
            assert removed.wait(timeout=1)
            finished.set()
            return BoundedProcessResult(137, b"", b"")

        def remove_container(_path: Path, _deadline: float, _ok=cleanup_succeeds) -> bool:
            removed.set()
            return _ok

        with monkeypatch.context() as patch:
            _stub_canary(patch, root / "broker")
            patch.setattr(canary_module, "run_bounded_process", run_process)
            patch.setattr(canary_module, "_remove_container_before_deadline", remove_container)
            patch.setattr(canary_module, "acquire_gpu_allocation_lease", lambda *_a, **_k: lease)
            result = _canary(
                task, caller_deadline=time.monotonic() + 0.1, environment=GPU_ENVIRONMENT
            )
        if cleanup_succeeds:
            assert result.status == "authored_failure"
            assert result.category == "search_production_entry_timeout"
        else:
            assert result.status == "infrastructure_failure"
            assert result.category == "search_production_container_cleanup"
            assert lease.quarantined_while_active is True
        assert lease.quarantined is not cleanup_succeeds, cleanup_succeeds
        assert removed.is_set(), cleanup_succeeds
        assert finished.is_set(), cleanup_succeeds

    # an interrupt quarantines the unverified evaluator lease
    root = tmp_path / "interrupt"
    task = _gpu_task(root)
    lease = _Lease()

    class InterruptingBroker(_FakeBroker):
        @property
        def completed_reset_seeds(self) -> tuple[int, ...]:  # type: ignore[override]
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        _stub_canary(patch, root / "broker", broker=InterruptingBroker)
        patch.setattr(canary_module, "acquire_gpu_allocation_lease", lambda *_a, **_k: lease)
        patch.setattr(canary_module, "_remove_container_before_deadline", lambda *_a, **_k: False)
        patch.setattr(
            canary_module, "run_bounded_process", lambda *_a, **_k: BoundedProcessResult(0, b"", b"")
        )
        with pytest.raises(KeyboardInterrupt):
            _canary(task, environment=GPU_ENVIRONMENT)
    assert lease.quarantined
    assert lease.quarantined_while_active is True


# ---------------------------------------------------------------- production-entry gate


@pytest.mark.parametrize(
    ("canary_result", "expected"),
    [
        pytest.param(
            ProductionCanaryResult(
                status="authored_failure",
                plane="final",
                category="final_production_entry_nonzero",
                component="final_evaluator",
                stage="production-entry",
                diagnostic="production-only failure",
            ),
            {
                "status": "authored_failure",
                "category": "final_production_entry_nonzero",
                "component": "final_evaluator",
                "stage": "production-entry",
            },
            id="typed_failure",
        ),
        pytest.param(
            ProductionCanaryResult.passed(
                "both",
                diagnostic=(
                    "attempt 1 [final_production_candidate_runtime]: runtime unavailable\n"
                    "attempt 2 [production_entry_passed]: passed"
                ),
            ),
            {"status": "pass"},
            id="recovered_retry_diagnostic",
        ),
    ],
)
def test_production_entry_gate_preserves_canary_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canary_result: ProductionCanaryResult,
    expected: dict[str, str],
) -> None:
    task = _task(tmp_path)
    monkeypatch.setattr("src.task_drafting.checks._prepared_runtime_available", lambda *_a: True)
    monkeypatch.setattr(
        "src.task_drafting.checks.run_production_entry_canaries", lambda **_k: canary_result
    )

    report = run_gate(
        "production-entry",
        GateSettings(
            task_dir=task,
            source_task_dir=task,
            containerize=True,
            container_image=IMAGE,
            container_network="none",
            timeout_seconds=300,
        ),
    )

    for attribute, value in expected.items():
        assert getattr(report, attribute) == value, attribute
    if report.status == "pass":
        assert "final_production_candidate_runtime" in report.diagnostic


# ---------------------------------------------------------------- both-plane retries


def test_production_canaries_run_both_planes_and_retry_only_infrastructure_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(tmp_path)

    # both search and final planes are required
    planes: list[str] = []

    def run_both(**kwargs: object) -> ProductionCanaryResult:
        planes.append(str(kwargs["plane"]))
        return ProductionCanaryResult.passed(str(kwargs["plane"]))

    with monkeypatch.context() as patch:
        patch.setattr(canary_module, "run_production_entry_canary", run_both)
        result = _canaries(task)
    assert result.status == "pass"
    assert planes == ["search", "final"]

    # only the plane whose infrastructure failed is retried
    for retried_plane, expected_planes in (
        ("search", ["search", "search", "final"]),
        ("final", ["search", "final", "final"]),
    ):
        planes = []

        def run_retry(_retried=retried_plane, _planes=planes, **kwargs: object):
            plane = str(kwargs["plane"])
            _planes.append(plane)
            if plane == _retried and _planes.count(plane) == 1:
                return _failure(
                    "infrastructure_failure",
                    plane,
                    f"{plane}_production_candidate_runtime",
                    "harness",
                    "candidate container runtime failed to launch",
                )
            return ProductionCanaryResult.passed(plane)

        with monkeypatch.context() as patch:
            patch.setattr(canary_module, "run_production_entry_canary", run_retry)
            result = _canaries(task)
        assert result.status == "pass", retried_plane
        assert planes == expected_planes, retried_plane
        assert f"attempt 1 [{retried_plane}_production_candidate_runtime]" in result.diagnostic
        assert "attempt 2 [production_entry_passed]: passed" in result.diagnostic, retried_plane

    # a recovered search attempt is retained when final fails
    planes = []

    def run_final_fails(**kwargs: object) -> ProductionCanaryResult:
        plane = str(kwargs["plane"])
        planes.append(plane)
        if planes == ["search"]:
            return _failure(
                "infrastructure_failure",
                plane,
                "search_production_candidate_runtime",
                "harness",
                "search runtime unavailable",
            )
        if plane == "final":
            return _failure(
                "authored_failure",
                plane,
                "final_production_candidate_protocol",
                "final_evaluator",
                "final candidate protocol failed",
            )
        return ProductionCanaryResult.passed(plane)

    with monkeypatch.context() as patch:
        patch.setattr(canary_module, "run_production_entry_canary", run_final_fails)
        result = _canaries(task)
    assert result.status == "authored_failure"
    assert planes == ["search", "search", "final"]
    assert "attempt 1 [search_production_candidate_runtime]" in result.diagnostic
    assert "attempt 2 [production_entry_passed]: passed" in result.diagnostic
    assert "attempt 1 [final_production_candidate_protocol]" in result.diagnostic

    # broker preparation infrastructure failure is retried
    preparation_calls = 0

    class PassingBroker(_FakeBroker):
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self.completed_reset_seeds = kwargs["expected_reset_seeds"]  # type: ignore[assignment]

    def prepare_broker(*_args: object, **_kwargs: object) -> Path:
        nonlocal preparation_calls
        preparation_calls += 1
        if preparation_calls == 1:
            raise CandidateBrokerInfrastructureError("broker recovery unavailable")
        return tmp_path / f"broker-{preparation_calls}"

    with monkeypatch.context() as patch:
        _stub_canary(patch, tmp_path / "broker", broker=PassingBroker)
        patch.setattr(canary_module, "_prepare_task_broker_directory", prepare_broker)
        patch.setattr(canary_module, "_stop_evaluator", lambda *_a, **_k: True)
        patch.setattr(
            canary_module,
            "run_bounded_process",
            lambda *_a, **_k: BoundedProcessResult(137, b"", b""),
        )
        result = _canaries(task)
    assert result.status == "pass"
    assert preparation_calls == 3
    assert "search_production_harness_runtime" in result.diagnostic
    assert "broker recovery unavailable" in result.diagnostic


NO_RETRY_CASES = [
    (
        "authored_failure",
        "authored_failure",
        "production_candidate_protocol",
        "{plane}_evaluator",
        "candidate protocol failed",
        None,
    ),
    (
        "after_deadline",
        "infrastructure_failure",
        "production_candidate_runtime",
        "harness",
        "candidate runtime failed",
        10.0,
    ),
]


def test_production_canaries_stop_after_repeated_authored_or_post_deadline_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(tmp_path)

    # two infrastructure failures stop the retries
    calls = 0

    def run_plane(**kwargs: object) -> ProductionCanaryResult:
        nonlocal calls
        calls += 1
        plane = str(kwargs["plane"])
        return _failure(
            "infrastructure_failure",
            plane,
            f"{plane}_production_candidate_runtime",
            "harness",
            "x" * 4096 if calls == 1 else "runtime failure 2",
        )

    with monkeypatch.context() as patch:
        patch.setattr(canary_module, "run_production_entry_canary", run_plane)
        result = _canaries(task)
    assert result.status == "infrastructure_failure"
    assert calls == 2
    assert "attempt 1 [search_production_candidate_runtime]" in result.diagnostic
    assert "attempt 2 [search_production_candidate_runtime]: runtime failure 2" in result.diagnostic

    # authored failures and failures after the deadline are not retried
    for name, status, category_suffix, component, diagnostic, deadline in NO_RETRY_CASES:
        calls = 0

        def run_once(
            _status=status, _suffix=category_suffix, _component=component, _diag=diagnostic,
            **kwargs: object,
        ) -> ProductionCanaryResult:
            nonlocal calls
            calls += 1
            plane = str(kwargs["plane"])
            return _failure(
                _status, plane, f"{plane}_{_suffix}", _component.format(plane=plane), _diag
            )

        overrides: dict[str, object] = {}
        with monkeypatch.context() as patch:
            patch.setattr(canary_module, "run_production_entry_canary", run_once)
            if deadline is not None:
                patch.setattr(canary_module.time, "monotonic", lambda _d=deadline: _d)
                overrides["caller_deadline"] = deadline
            result = _canaries(task, **overrides)
        assert result.status == status, name
        assert calls == 1, name
        assert result.diagnostic == diagnostic, name


# ---------------------------------------------------------------- evaluator teardown


class FinishingWorker:
    def __init__(self, finish: Callable[[], object]) -> None:
        self.finish = finish

    def join(self, timeout: float) -> None:
        assert timeout > 0
        self.finish()

    def is_alive(self) -> bool:
        return False


def _worker(finish: Callable[..., object] = lambda: None) -> threading.Thread:
    return cast(threading.Thread, FinishingWorker(finish))


def test_stop_evaluator_removes_late_cid_before_broker_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a CID written during the worker join is removed before the broker exits
    for index, initial_cid in enumerate((None, "")):
        broker_dir = tmp_path / f"broker-{index}"
        observed: list[str | None] = []

        def remove(cid: Path, deadline: float, _observed=observed) -> bool:
            assert deadline > time.monotonic()
            _observed.append(cid.read_text() if cid.exists() else None)
            return True

        with monkeypatch.context() as patch:
            patch.setattr(canary_module, "_remove_container_before_deadline", remove)
            with CandidateBroker(["unused"], broker_dir) as broker:
                cid = broker.socket_dir / "evaluator.cid"
                if initial_cid is not None:
                    cid.write_text(initial_cid)
                worker = _worker(lambda cid=cid: cid.write_text("a" * 64))
                assert canary_module._stop_evaluator(cid, worker, time.monotonic() + 2)
                assert not cid.exists()
        assert observed == [initial_cid, "a" * 64], initial_cid
        assert not broker_dir.exists(), initial_cid

    # an empty CID stays linked until its producer closes it
    cid = tmp_path / "evaluator.cid"
    observed_empty: list[str] = []
    with cid.open("w") as writer:

        def finish() -> None:
            assert cid.exists(), "cleanup unlinked a CID still held by its producer"
            writer.write("b" * 64)
            writer.flush()

        def remove_empty(path: Path, _deadline: float) -> bool:
            observed_empty.append(path.read_text())
            return True

        monkeypatch.setattr(canary_module, "_remove_container_before_deadline", remove_empty)
        assert canary_module._stop_evaluator(cid, _worker(finish), time.monotonic() + 2)
    assert observed_empty == ["", "b" * 64]
    assert not cid.exists()


FINAL_REMOVAL_CASES = [
    ("both_removed", [True, True], True),
    ("final_removal_decides", [False, True], True),
    ("failed_final_removal_retains_cid", [True, False], False),
]


def test_stop_evaluator_retains_recovery_cid_unless_final_removal_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cid = tmp_path / "evaluator.cid"

    # the final removal after the worker finishes decides cleanup
    for name, removals, stopped in FINAL_REMOVAL_CASES:
        cid.write_text("c" * 64)
        results = iter(removals)
        with monkeypatch.context() as patch:
            patch.setattr(
                canary_module, "_remove_container_before_deadline", lambda *_: next(results)
            )
            assert canary_module._stop_evaluator(cid, _worker(), time.monotonic() + 2) is stopped, name
        if stopped:
            assert not cid.exists(), name
        else:
            assert cid.read_text() == "c" * 64, name
            cid.unlink()

    # a live worker retains the recovery CID
    cid.write_text("e" * 64)

    class LiveWorker(FinishingWorker):
        def is_alive(self) -> bool:
            return True

    with monkeypatch.context() as patch:
        patch.setattr(canary_module, "_remove_container_before_deadline", lambda *_: True)
        worker = cast(threading.Thread, LiveWorker(lambda: None))
        assert not canary_module._stop_evaluator(cid, worker, time.monotonic() + 2)
    assert cid.read_text() == "e" * 64

    # a deadline expiring during the join retains the recovery CID
    cid.write_text("f" * 64)
    now = [2.0]
    with monkeypatch.context() as patch:
        patch.setattr(canary_module.time, "monotonic", lambda: now[0])
        patch.setattr(
            canary_module, "_remove_container_before_deadline", lambda path, end: now[0] < end
        )
        assert not canary_module._stop_evaluator(cid, _worker(lambda: now.__setitem__(0, 3.0)), 3.0)
    assert cid.read_text() == "f" * 64


def test_canary_timeout_cleans_cid_created_during_worker_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(tmp_path)
    broker_dir = tmp_path / "broker"
    cid = broker_dir / "evaluator.cid"
    first_removal = threading.Event()
    seen: list[str | None] = []

    def evaluator(*_args: object, **_kwargs: object) -> BoundedProcessResult:
        assert first_removal.wait(timeout=2)
        cid.write_text("a" * 64)
        return BoundedProcessResult(137, b"", b"")

    def remove(path: Path, deadline: float) -> bool:
        assert time.monotonic() < deadline
        seen.append(path.read_text() if path.exists() else None)
        first_removal.set()
        return True

    _stub_canary(
        monkeypatch, broker_dir, broker=None, evaluation=lambda **_k: ["evaluator"],
        candidate=lambda **_k: ["unused"],
    )
    monkeypatch.setattr(canary_module, "run_bounded_process", evaluator)
    monkeypatch.setattr(canary_module, "_remove_container_before_deadline", remove)

    result = _canary(task, caller_deadline=time.monotonic() + 1)

    assert result.status == "authored_failure"
    assert result.category == "search_production_entry_timeout"
    assert seen == [None, "a" * 64]
    assert not broker_dir.exists()


@pytest.mark.parametrize(
    "reset_seeds,returncode,entry_category",
    [((11,), 0, "production_entry_passed"), ((), 1, "search_production_entry_nonzero")],
)
@pytest.mark.parametrize("failure_stage", [
    "broker", "evaluator", "evaluator_unlink", "broker_container",
    "evaluator_unlink_broker_container", "evaluator_broker", "evaluator_remove_raises",
])
def test_cleanup_failure_preserves_observed_entry_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_seeds: tuple[int, ...],
    returncode: int,
    entry_category: str,
    failure_stage: str,
) -> None:
    task = _gpu_task(tmp_path)
    broker_dir = tmp_path / "broker"
    lease = _Lease()
    monkeypatch.setattr(canary_module, "acquire_gpu_allocation_lease", lambda *a, **kw: lease)
    if failure_stage.startswith("evaluator_unlink"):
        broker_dir.mkdir()
        cid = broker_dir / "evaluator.cid"
        cid.write_text("a" * 64)
        original_unlink = Path.unlink

        def unlink(path: Path, missing_ok: bool = False) -> None:
            if path == cid:
                raise PermissionError(errno.EACCES, "private runtime text", str(cid))
            original_unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", unlink)

    class Broker(_FakeBroker):
        completed_reset_seeds = reset_seeds

        def __exit__(self, *args: object) -> None:
            if "broker_container" in failure_stage:
                raise CandidateContainerCleanupError("operation=candidate_stop errno=unknown")
            if failure_stage in {"broker", "evaluator_broker"}:
                raise CandidateBrokerInfrastructureError(
                    "candidate container cleanup failed: operation=broker_directory_remove errno=39"
                )

    def remove(*_args: object) -> bool:
        if failure_stage == "evaluator_remove_raises":
            raise OSError(errno.EIO, "private Docker error")
        return failure_stage not in {"evaluator", "evaluator_broker"}

    _stub_canary(monkeypatch, broker_dir, broker=Broker, evaluation=lambda **kw: ["evaluator"])
    monkeypatch.setattr(canary_module, "_remove_container_before_deadline", remove)
    monkeypatch.setattr(
        canary_module,
        "run_bounded_process",
        lambda *a, **kw: BoundedProcessResult(returncode, b"", b"original evaluator failure"),
    )

    result = _canary(task)

    assert result.status == "infrastructure_failure"
    assert f"entry_category={entry_category}" in result.diagnostic
    expected_status = "pass" if reset_seeds else "authored_failure"
    assert f"entry_status={expected_status}" in result.diagnostic
    if not reset_seeds:
        assert "entry_diagnostic=original evaluator failure" in result.diagnostic
    assert "private" not in result.diagnostic
    assert str(tmp_path) not in result.diagnostic
    assert lease.released
    assert lease.quarantined is (failure_stage not in {"broker", "evaluator_unlink"})
    if failure_stage == "evaluator_unlink":
        assert result.category == "search_production_harness_runtime"
        assert "errno=EACCES" in result.diagnostic
        assert (broker_dir / "evaluator.cid").read_text() == "a" * 64
