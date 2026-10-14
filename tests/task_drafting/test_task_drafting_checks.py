from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from tests.task_drafting.test_task_alignment import _write_scaffold

from src.task_drafting.checks import (
    GateSettings,
    _final_topology_smoke,
    _final_topology_category,
    _reset_sequence_diagnostic,
    run_gate,
)
from src.task_drafting.gate import TaskGateReport
from src.task_drafting.publication import RuntimeResourceValidationDeadlineExceeded
from src.final_evaluation.candidate_broker import CandidateContainerCleanupError
from src.investigation.task_runtime import TaskRuntimePreparationResult
from src.investigation.container_images import ContainerImageResolutionError
from src.investigation.bounded_process import BoundedProcessResult, ProcessTimeoutError
from src.investigation.container_gpus import GpuLeaseTimeout

__all__ = ["_ObservedGpuLease", "_settings", "_stub_final_topology", "_write_scaffold"]

CHECKS = "src.task_drafting.checks."
EVALUATOR = """from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--program_path", required=True)
parser.add_argument("--results_dir", required=True)
args = parser.parse_args()
output = Path(args.results_dir)
output.mkdir(parents=True, exist_ok=True)
metric = "test_score" if os.environ.get("TASK_EVALUATION_PLANE") == "final" else "combined_score"
(output / "metrics.json").write_text(json.dumps({metric: 1.0}))
(output / "correct.json").write_text(json.dumps({"correct": True}))
"""
EXCHANGE_DIR_PROGRAM = "import os\n\ndef handle(request):\n    return os.environ.get('EXCHANGE_DIR')\n"


def _settings(root: Path, **overrides: object) -> GateSettings:
    options: dict[str, object] = dict(
        task_dir=root,
        source_task_dir=root,
        containerize=False,
        container_image="unused",
        container_network="none",
        timeout_seconds=10,
    )
    options.update(overrides)
    return GateSettings(**options)  # type: ignore[arg-type]


def _write(root: Path, name: str, text: str) -> None:
    (root / name).write_text(text, encoding="utf-8")


def _scaffold(root: Path, *, search: str | None = None, final: str | None = None) -> Path:
    root.mkdir(exist_ok=True)
    _write_scaffold(root)
    if search is not None:
        _write(root, "evaluate.py", search)
    if final is not None:
        _write(root, "evaluate_final.py", final)
    return root


def _throw(error: BaseException):
    def raise_error(*_args: object, **_kwargs: object) -> None:
        raise error

    return raise_error


# ---------------------------------------------------------------- native smoke gates


def test_native_smoke_gates_pass_valid_and_scientifically_incorrect_baselines(tmp_path: Path) -> None:
    # contract and native smoke gates pass on the same scaffold
    valid = _scaffold(tmp_path / "valid", search=EVALUATOR, final=EVALUATOR)
    _write(valid, "program_0/src/program.py", "def handle(request):\n    return request\n")
    for gate in ("contract", "search-smoke", "final-smoke"):
        assert run_gate(gate, _settings(valid)).status == "pass", gate

    # native smoke accepts a scientifically incorrect baseline
    evaluator = EVALUATOR.replace('json.dumps({"correct": True})', 'json.dumps({"correct": False})')
    incorrect = _scaffold(tmp_path / "incorrect", search=evaluator, final=evaluator)
    assert run_gate("search-smoke", _settings(incorrect)).status == "pass"
    assert run_gate("final-smoke", _settings(incorrect)).status == "pass"


def test_native_search_gate_returns_bounded_failures_for_crash_and_extra_fields(
    tmp_path: Path,
) -> None:
    # nonzero exit returns a bounded actionable failure
    broken = _scaffold(tmp_path / "broken", search="raise RuntimeError('search evaluator broke')\n")
    report = run_gate("search-smoke", _settings(broken))
    assert report.status == "authored_failure"
    assert report.component == "search_evaluator"
    assert report.category == "search_nonzero_exit"
    assert "search evaluator broke" in report.diagnostic

    # extra correctness fields are rejected
    evaluator = EVALUATOR.replace(
        'json.dumps({"correct": True})', 'json.dumps({"correct": True, "debug": "leak"})'
    )
    leaky = _scaffold(tmp_path / "leaky", search=evaluator)
    report = run_gate("search-smoke", _settings(leaky))
    assert report.status == "authored_failure"
    assert report.category == "search_invalid_results"


# ---------------------------------------------------------------- acceptance smoke


def test_acceptance_smoke_stops_at_exact_failing_gate(tmp_path: Path) -> None:
    _scaffold(tmp_path, search=EVALUATOR, final="raise RuntimeError('final evaluator broke')\n")
    contract_path = tmp_path / "task_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["compute"]["search"]["max_candidate_rpc_calls"] = 10
    contract["compute"]["final"]["max_candidate_rpc_calls"] = 10
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    _write(tmp_path, "requirements.txt", "numpy==2.1.0\n")

    report = run_gate("acceptance-smoke", _settings(tmp_path))

    assert report.status == "authored_failure"
    assert report.gate == "final-smoke"
    assert report.category == "final_nonzero_exit"


def test_acceptance_smoke_shares_explicit_deadline_and_classifies_resource_scan_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_scaffold(tmp_path)

    # explicit shared deadline is forwarded to the first gate
    observed_deadlines: list[float | None] = []

    def stop_at_contract(*args, deadline=None, **kwargs):
        del args, kwargs
        observed_deadlines.append(deadline)
        return TaskGateReport.authored_failure(
            gate="contract",
            component="contract",
            stage="contract",
            category="contract_invalid",
            scaffold_fingerprint="sha256:" + "a" * 64,
            exit_code=3,
            elapsed_seconds=0.1,
            diagnostic="stop",
        )

    with monkeypatch.context() as patch:
        patch.setattr(CHECKS + "_run_gate", stop_at_contract)
        patch.setattr(CHECKS + "time.monotonic", lambda: 100.0)
        report = run_gate(
            "acceptance-smoke",
            _settings(tmp_path, timeout_seconds=1800, acceptance_timeout_seconds=5),
        )
    assert report.status == "authored_failure"
    assert observed_deadlines == [105.0]

    # resource scan deadline is classified as infrastructure
    with monkeypatch.context() as patch:
        patch.setattr(
            CHECKS + "validate_staged_runtime_resources",
            _throw(RuntimeResourceValidationDeadlineExceeded("expired")),
        )
        report = run_gate("acceptance-smoke", _settings(tmp_path))
    assert report.status == "infrastructure_failure"
    assert report.category == "acceptance_deadline_exceeded"


def test_acceptance_smoke_rejects_modified_task_and_runtime_resources_before_evaluators(
    tmp_path: Path,
) -> None:
    # canonical task.md changed from source
    task_dir = _scaffold(tmp_path / "staged")
    source_dir = _scaffold(tmp_path / "source")
    _write(task_dir, "task.md", "changed by repair\n")
    report = run_gate("acceptance-smoke", _settings(task_dir, source_task_dir=source_dir))
    assert report.status == "authored_failure"
    assert report.gate == "contract"
    assert report.category == "canonical_task_modified"

    # runtime resources exceeding the limit stop before evaluators
    limited = _scaffold(tmp_path / "limited")
    report = run_gate("acceptance-smoke", _settings(limited, max_runtime_bytes=1))
    assert report.status == "authored_failure"
    assert report.gate == "runtime"
    assert report.category == "runtime_resources_exceed_limit"

    # unsafe runtime resources are distinguished from oversized ones
    unsafe = _scaffold(tmp_path / "unsafe")
    outside = tmp_path / "outside"
    outside.mkdir()
    (unsafe / "resources" / "unsafe").symlink_to(outside, target_is_directory=True)
    report = run_gate("acceptance-smoke", _settings(unsafe))
    assert report.status == "authored_failure"
    assert report.gate == "runtime"
    assert report.category == "runtime_resources_invalid"
    assert report.scaffold_fingerprint == "sha256:" + "0" * 64


# ---------------------------------------------------------------- contract gate


def _invalid_contract(root: Path) -> None:
    contract_path = root / "task_contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["upstream"] = {}
    contract_path.write_text(json.dumps(payload), encoding="utf-8")


def _extensionless_entrypoint(root: Path) -> None:
    program_dir = root / "program_0"
    (program_dir / "src/program.py").rename(program_dir / "src/program")
    spec_path = program_dir / "program_spec.toml"
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace('path = "program.py"', 'path = "program"'),
        encoding="utf-8",
    )
    _write(root, "program_0/src/program", EXCHANGE_DIR_PROGRAM)


def _writer(name: str, text: str):
    return lambda root: _write(root, name, text)


CONTRACT_CASES = [
    pytest.param(
        _invalid_contract,
        {"status": "authored_failure", "category": "contract_invalid", "exit_code": 3},
        ["upstream must be an array"],
        id="invalid_contract",
    ),
    pytest.param(
        lambda root: (root / "program_0/src/program.py").unlink(),
        {"status": "authored_failure", "category": "candidate_entrypoint_invalid"},
        ["program_0/program_spec.toml", "program_0"],
        id="missing_canonical_entrypoint",
    ),
    pytest.param(
        _writer("program_0/src/program.py", EXCHANGE_DIR_PROGRAM),
        {
            "status": "authored_failure",
            "category": "rpc_transport_invalid",
            "component": "candidate",
            "stage": "contract",
        },
        ["program_0/src/program.py:4", "finite inline JSON"],
        id="shared_filesystem_rpc_transport",
    ),
    pytest.param(
        _writer(
            "evaluate.py",
            "import os\nVALUE = os.environ.get('AGENTIC_HARNESS_ROLE_TELEMETRY_JSON')\n",
        ),
        {
            "status": "authored_failure",
            "category": "harness_capability_invalid",
            "component": "search_evaluator",
        },
        ["AGENTIC_HARNESS_ROLE_TELEMETRY_JSON", "not supplied"],
        id="unavailable_harness_environment",
    ),
    pytest.param(
        _writer("evaluate.py", "import os\nSMOKE = os.environ.get('AGENTIC_TASK_DRAFT_SMOKE')\n"),
        {"status": "pass"},
        [],
        id="documented_harness_environment",
    ),
    pytest.param(
        _writer(
            "evaluate.py",
            '"""AGENTIC_HARNESS_ROLE_TELEMETRY_JSON is intentionally unavailable."""\n',
        ),
        {"status": "pass"},
        [],
        id="harness_name_in_docstring",
    ),
    pytest.param(
        _extensionless_entrypoint,
        {"status": "authored_failure", "category": "rpc_transport_invalid"},
        ["program_0/src/program:4"],
        id="extensionless_declared_entrypoint",
    ),
]


@pytest.mark.parametrize(("mutate", "expected", "fragments"), CONTRACT_CASES)
def test_contract_gate_classifies_scaffold(
    tmp_path: Path, mutate, expected: dict[str, object], fragments: list[str]
) -> None:
    _write_scaffold(tmp_path)
    mutate(tmp_path)

    report = run_gate("contract", _settings(tmp_path))

    for attribute, value in expected.items():
        assert getattr(report, attribute) == value, attribute
    for fragment in fragments:
        assert fragment in report.diagnostic


# ---------------------------------------------------------------- runtime gate


RUNTIME_FAILURE_CASES = [
    (
        "authored",
        TaskRuntimePreparationResult.AUTHORED_FAILURE,
        "candidate reset request rejected",
        "authored_failure",
        "runtime_baseline_validation",
    ),
    (
        "infrastructure",
        TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE,
        "runtime launch failed: connection refused",
        "infrastructure_failure",
        "runtime_bootstrap",
    ),
]


def test_runtime_gate_classifies_sanitized_preparation_selector_and_image_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_scaffold(tmp_path)

    # preparation failures keep their class and are sanitized
    for name, result, message, status, category in RUNTIME_FAILURE_CASES:

        def fail_runtime(*args, _message=message, _result=result, **kwargs):
            del args, kwargs
            runtime_logger = logging.getLogger("src.investigation.task_runtime")
            runtime_logger.error(_message)
            runtime_logger.error("api_key=sk-proj-abcdefghijklmnop")
            return _result

        with monkeypatch.context() as patch:
            patch.setenv("AGENT_CONTAINER_GPUS", "0")
            patch.setattr(CHECKS + "prepare_task_runtime_result", fail_runtime)
            patch.setattr(CHECKS + "resolve_immutable_image", lambda image: image)
            report = run_gate("runtime", _settings(tmp_path, containerize=True))
        assert report.status == status, name
        assert report.category == category, name
        assert message in report.diagnostic, name
        assert "sk-proj" not in report.diagnostic, name

    # missing GPU selector is classified without starting preparation
    with monkeypatch.context() as patch:
        patch.delenv("AGENT_CONTAINER_GPUS", raising=False)
        patch.setattr(CHECKS + "_prepared_runtime_available", lambda *_args: False)
        patch.setattr(
            CHECKS + "prepare_task_runtime_result",
            lambda *args, **kwargs: pytest.fail("runtime preparation must not start"),
        )
        report = run_gate("runtime", _settings(tmp_path, containerize=True))
    assert report.status == "infrastructure_failure"
    assert report.category == "gpu_selector_missing"

    # container image resolution failure is infrastructure
    with monkeypatch.context() as patch:
        patch.setenv("AGENT_CONTAINER_GPUS", "0")
        patch.setattr(CHECKS + "_prepared_runtime_available", lambda *_args: False)
        patch.setattr(
            CHECKS + "resolve_immutable_image",
            _throw(ContainerImageResolutionError("missing image")),
        )
        report = run_gate(
            "runtime", _settings(tmp_path, containerize=True, container_image="missing")
        )
    assert report.status == "infrastructure_failure"
    assert report.category == "container_image_resolution"


# ---------------------------------------------------------------- pure helpers


def test_reset_sequence_diagnostic_and_final_topology_category_helpers() -> None:
    # reset sequence diagnostic explains the explicit smoke contract
    diagnostic = _reset_sequence_diagnostic((11, 11), (11, 11, 11))
    assert "expected primary [11, 11, 11]" in diagnostic
    assert "observed [11, 11]" in diagnostic
    assert "count=2" in diagnostic
    assert "one trailing deterministic replay of seed 11 is allowed" in diagnostic

    # reset sequence diagnostic bounds untrusted observations
    oversized_seed = 1 << 100_000
    diagnostic = _reset_sequence_diagnostic((oversized_seed,) * 100, (oversized_seed,) * 3)
    assert "<int bits=100001>" in diagnostic
    assert "count=100" in diagnostic
    assert len(diagnostic.encode("utf-8")) < 512

    # final topology category distinguishes evaluator exit from missing RPC
    assert _final_topology_category(returncode=9, broker_calls=0) == "final_evaluator_nonzero"
    assert _final_topology_category(returncode=0, broker_calls=0) == "final_candidate_not_called"
    assert _final_topology_category(returncode=0, broker_calls=1) is None


# ---------------------------------------------------------------- final smoke GPU lease


class _ObservedGpuLease:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.active = False
        self.quarantined = False
        self.acquisition: tuple[str | None, int, Path, float] | None = None

    def __enter__(self):
        self.active = True
        self.events.append("lease-enter")
        return self

    def __exit__(self, *_args: object) -> None:
        self.events.append("lease-exit")
        self.active = False

    def prepare_runtime(self) -> None:
        assert self.active

    def wrap_command(self, command: list[str]) -> list[str]:
        assert self.active
        return command

    def quarantine(self) -> None:
        self.quarantined = True
        self.events.append("lease-quarantine")


def _stub_final_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease: _ObservedGpuLease
) -> None:
    broker_dir = tmp_path / "broker"
    broker_dir.mkdir()

    class Broker:
        calls = 1
        calls_before_first_reset = 0
        reset_seeds = (21,)
        peer_disconnects = 0

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            assert lease.active
            lease.events.append("broker-enter")
            return self

        def __exit__(self, *_args: object) -> None:
            assert lease.active
            lease.events.append("broker-exit")

    def prepare_broker(*_args, **_kwargs) -> Path:
        assert lease.active
        lease.events.append("broker-prepare")
        return broker_dir

    def acquire(raw_selector, gpu_count, *, lease_root, deadline):
        lease.acquisition = (raw_selector, gpu_count, lease_root, deadline)
        return lease

    monkeypatch.setenv("AGENT_CONTAINER_GPUS", "device=GPU-a,GPU-b,GPU-c,GPU-d")
    monkeypatch.setattr(CHECKS + "resolve_immutable_image", lambda *_args: "image")
    monkeypatch.setattr(CHECKS + "_prepare_task_broker_directory", prepare_broker)
    monkeypatch.setattr(CHECKS + "_evaluation_command", lambda **_kwargs: ["eval"])
    monkeypatch.setattr(CHECKS + "_candidate_command", lambda **_kwargs: ["candidate"])
    monkeypatch.setattr(CHECKS + "CandidateBroker", Broker)
    monkeypatch.setattr(CHECKS + "_require_score", lambda *_args: None)
    monkeypatch.setattr(CHECKS + "acquire_gpu_allocation_lease", acquire)


def test_final_smoke_reuses_prepared_runtime_and_classifies_busy_gpu_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_scaffold(tmp_path)

    # runtime prepared by the search gate is reused, never rebuilt
    with monkeypatch.context() as patch:
        patch.setattr(CHECKS + "_prepared_runtime_available", lambda *_args: True)
        patch.setattr(
            CHECKS + "_prepare_runtime",
            lambda *_args: pytest.fail("prepared runtime must not be rebuilt"),
        )
        patch.setattr(CHECKS + "_final_topology_smoke", lambda *_args, **_kwargs: None)
        report = run_gate(
            "final-smoke",
            _settings(tmp_path, containerize=True, container_image="sha256:" + "a" * 64),
        )
    assert report.status == "pass"

    # busy GPU allocation is classified as infrastructure
    with monkeypatch.context() as patch:
        patch.setenv("AGENT_CONTAINER_GPUS", "device=GPU-a,GPU-b,GPU-c,GPU-d")
        patch.setattr(CHECKS + "_prepared_runtime_available", lambda *_args: True)
        patch.setattr(CHECKS + "resolve_immutable_image", lambda *_args: "image")
        broker_dir = tmp_path / "broker"
        broker_dir.mkdir()
        patch.setattr(CHECKS + "_prepare_task_broker_directory", lambda *_a, **_k: broker_dir)
        patch.setattr(CHECKS + "acquire_gpu_allocation_lease", _throw(GpuLeaseTimeout("busy")))
        report = run_gate(
            "final-smoke", _settings(tmp_path, containerize=True, container_image="image")
        )
    assert report.status == "infrastructure_failure"
    assert report.category == "gpu_allocation_busy"


def test_final_smoke_holds_gpu_lease_through_cleanup_and_quarantines_uncertain_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # lease is held through evaluator container cleanup
    root = _scaffold(tmp_path / "held")
    events: list[str] = []
    lease = _ObservedGpuLease(events)
    with monkeypatch.context() as patch:
        _stub_final_topology(root, patch, lease)

        def run(*_args, **_kwargs) -> BoundedProcessResult:
            assert lease.active
            events.append("evaluator")
            return BoundedProcessResult(0, b"", b"")

        def remove(*_args, **_kwargs) -> bool:
            assert lease.active
            events.append("evaluator-cleanup")
            return True

        patch.setattr(CHECKS + "run_bounded_process", run)
        patch.setattr(CHECKS + "_remove_container_before_deadline", remove)
        shared_deadline = time.monotonic() + 5
        _final_topology_smoke(_settings(root), "digest", "src/program.py", deadline=shared_deadline)
    assert events == [
        "lease-enter",
        "broker-prepare",
        "broker-enter",
        "evaluator",
        "evaluator-cleanup",
        "broker-exit",
        "lease-exit",
    ]
    assert lease.acquisition is not None
    selector, gpu_count, lease_root, lease_deadline = lease.acquisition
    assert selector == "device=GPU-a,GPU-b,GPU-c,GPU-d"
    assert gpu_count == 4
    assert lease_root.name.startswith("agentic-scientist-gpu-leases-")
    assert lease_deadline <= shared_deadline

    # an uncertain interrupted launch quarantines the lease
    root = _scaffold(tmp_path / "interrupted")
    lease = _ObservedGpuLease([])
    with monkeypatch.context() as patch:
        _stub_final_topology(root, patch, lease)
        patch.setattr(CHECKS + "run_bounded_process", _throw(ProcessTimeoutError("timeout")))
        patch.setattr(CHECKS + "_remove_container_before_deadline", lambda *_a, **_k: True)
        with pytest.raises(Exception, match="final_timeout"):
            _final_topology_smoke(_settings(root), "digest", "src/program.py")
    assert lease.quarantined is True
    assert lease.active is False

    # cleanup raising quarantines the lease and reports infrastructure failure
    root = _scaffold(tmp_path / "cleanup")
    lease = _ObservedGpuLease([])
    with monkeypatch.context() as patch:
        _stub_final_topology(root, patch, lease)
        patch.setattr(
            CHECKS + "run_bounded_process", lambda *_a, **_k: BoundedProcessResult(0, b"", b"")
        )
        patch.setattr(
            CHECKS + "_remove_container_before_deadline", _throw(OSError("cleanup failed"))
        )
        patch.setattr(CHECKS + "_prepared_runtime_available", lambda *_args: True)
        report = run_gate("final-smoke", _settings(root, containerize=True, container_image="image"))
    assert report.status == "infrastructure_failure"
    assert report.category == "evaluator_container_cleanup"
    assert lease.quarantined is True
    assert lease.active is False

    # failed broker recovery quarantines the lease
    root = _scaffold(tmp_path / "broker-recovery")
    lease = _ObservedGpuLease([])
    with monkeypatch.context() as patch:
        _stub_final_topology(root, patch, lease)
        patch.setattr(
            CHECKS + "_prepare_task_broker_directory",
            _throw(CandidateContainerCleanupError("container cleanup failed")),
        )
        with pytest.raises(Exception, match="container cleanup failed"):
            _final_topology_smoke(_settings(root), "digest", "src/program.py")
    assert lease.quarantined is True
    assert lease.active is False
