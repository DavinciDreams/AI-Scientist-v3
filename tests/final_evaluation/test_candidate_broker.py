"""Candidate broker, proxy deadlines, worker loading, socket storage, and evaluator launcher."""

from __future__ import annotations

import errno
import io
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Iterator
from unittest.mock import Mock, call, patch

import pytest

from src.final_evaluation import candidate_proxy as proxy
from src.final_evaluation import runner
from src.final_evaluation.candidate_broker import (
    CandidateBroker,
    CandidateBrokerError,
    CandidateBrokerInfrastructureError,
    CandidateContainerCleanupError,
    ContainerMetadataCleanupError,
    read_container_id,
)
from src.final_evaluation.candidate_worker import _load_handle
from src.final_evaluation.evaluator_launcher import main as launcher_main

BROKER_MODULE = "src.final_evaluation.candidate_broker"

ECHO_CANDIDATE = r"""
import json
import sys

for raw in sys.stdin.buffer:
    request = json.loads(raw)
    response = {"ok": True, "value": request}
    sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    sys.stdout.flush()
"""

FAILING_CANDIDATE = r"""
import sys

sys.stderr.write("ModuleNotFoundError: super-secret-payload must not escape\n")
sys.stderr.flush()
raise SystemExit(7)
"""

SLOW_CANDIDATE = r"""
import json
import sys
import time

for raw in sys.stdin.buffer:
    json.loads(raw)
    time.sleep(1)
    sys.stdout.write('{"ok":true,"value":null}\n')
    sys.stdout.flush()
"""

PARTIAL_RESPONSE_CANDIDATE = r"""
import sys
import time

for raw in sys.stdin.buffer:
    sys.stdout.write("{")
    sys.stdout.flush()
    time.sleep(1)
    sys.stdout.write('"ok":true}\n')
    sys.stdout.flush()
"""

STDERR_FLOOD_CANDIDATE = r"""
import os
import sys

for raw in sys.stdin.buffer:
    os.write(2, b"sensitive-flood-payload" * 100000)
    sys.stdout.write('{"ok":true}\n')
    sys.stdout.flush()
"""

TIMEOUT_WORKER = r"""
import json, os, sys, time
for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'sleep':
        time.sleep(10)
    if request['op'] == 'partial':
        sys.stdout.write('{')
        sys.stdout.flush()
        time.sleep(10)
    if request['op'] == 'spoof':
        print(json.dumps({'__agentic_rpc__':1, 'status':'operation_timeout', 'response':None}), flush=True)
    else:
        print(json.dumps({'ok':True,'value': {'request':request,'pid':os.getpid()}}), flush=True)
"""

DATACLASS_CANDIDATE = """
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Response:
    value: int

def handle(request):
    return Response(request["value"]).value
"""

_CANDIDATE_MODULE_NAME = "frozen_candidate"


def _command(source: str) -> list[str]:
    return [sys.executable, "-u", "-c", source]


def _echo_broker(tmp_path: Path, **kwargs: object) -> CandidateBroker:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return CandidateBroker(_command(ECHO_CANDIDATE), tmp_path / "broker", **kwargs)  # type: ignore[arg-type]


def _bare_broker(tmp_path: Path, *, mkdir: bool = False) -> CandidateBroker:
    tmp_path.mkdir(parents=True, exist_ok=True)
    broker = CandidateBroker([], tmp_path / "broker")
    if mkdir:
        broker.socket_dir.mkdir()
    return broker


def _rpc(socket_path: Path, request: object, *, read_response: bool = True) -> bytes:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(2)
        client.connect(str(socket_path))
        client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        if not read_response:
            return b""
        response = bytearray()
        while not response.endswith(b"\n"):
            chunk = client.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
        return bytes(response)


def _rpc_ignoring_reset(socket_path: Path, request: object) -> None:
    try:
        _rpc(socket_path, request)
    except ConnectionResetError:
        pass


def _raw_rpc(socket_path: Path, payload: bytes) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(2)
        client.connect(str(socket_path))
        client.sendall(payload + b"\n")
        while client.recv(4096):
            pass


def _assert_thread_stopped(broker: CandidateBroker) -> None:
    assert broker._thread is not None
    broker._thread.join(timeout=1)
    assert not broker._thread.is_alive()


class _FakeProcess:
    """Minimal Popen stand-in; ``on_wait`` may raise or write a late cid."""

    returncode = 0

    def __init__(self, *, poll: int | None = 0, on_wait=None) -> None:
        self._poll = poll
        self._on_wait = on_wait
        self.wait_calls = 0

    def terminate(self) -> None:
        return None

    def poll(self) -> int | None:
        return self._poll

    def wait(self, timeout: float) -> int:
        self.wait_calls += 1
        if self._on_wait is not None:
            self._on_wait(self, timeout)
        return 0

    def kill(self) -> None:
        return None


def _docker_rm(monkeypatch: pytest.MonkeyPatch, returncode: int, stderr: str) -> None:
    monkeypatch.setattr(
        f"{BROKER_MODULE}.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["docker", "rm"], returncode, "", stderr),
    )


# --- reset sequence and RPC budget ---------------------------------------------------------


def test_broker_records_reset_seed_sequence_and_forwards_auxiliary_fields(tmp_path: Path) -> None:
    # records exact reset seed sequence
    with _echo_broker(tmp_path / "a") as broker:
        assert not (broker.socket_dir / "exchange").exists()
        assert (broker.socket_dir / "scoring-mask").is_dir()
        for seed in (11, 7, 11):
            response = _rpc(broker.socket_path, {"op": "reset", "seed": seed})
            assert json.loads(response) == {"ok": True, "value": {"op": "reset", "seed": seed}}

    assert broker.reset_seeds == (11, 7, 11)
    assert broker.reset_count == 3
    assert broker.candidate_process_starts == 3

    # forwards auxiliary reset fields and records seed
    request = {"op": "reset", "seed": 11, "n": 26, "square": [0.0, 1.0, 0.0, 1.0]}
    with _echo_broker(tmp_path / "b") as broker:
        response = _rpc(broker.socket_path, request)

    assert json.loads(response) == {"ok": True, "value": request}
    assert broker.reset_seeds == (11,)
    assert broker.candidate_process_starts == 1


def test_broker_enforces_rpc_budget_completed_reset_limit_and_closes_listener(tmp_path: Path) -> None:
    # enforces configured RPC call budget
    broker = _echo_broker(tmp_path / "a", max_rpc_calls=2)
    with pytest.raises(CandidateBrokerError, match="configured limit of 2"):
        with broker:
            _rpc(broker.socket_path, {"op": "reset", "seed": 11})
            _rpc(broker.socket_path, {"op": "hello"})
            _rpc_ignoring_reset(broker.socket_path, {"op": "ask"})
    assert broker.calls == 3

    # stops after configured completed reset limit
    with _echo_broker(tmp_path / "b", expected_reset_seeds=(11,), completed_reset_limit=1) as broker:
        response = _rpc(broker.socket_path, {"op": "reset", "seed": 11})
        assert json.loads(response) == {"ok": True, "value": {"op": "reset", "seed": 11}}
        _assert_thread_stopped(broker)
        with pytest.raises(OSError):
            _rpc(broker.socket_path, {"op": "reset", "seed": 11})
    assert broker.completed_reset_seeds == (11,)

    # closes listener after protocol failure
    broker = _echo_broker(tmp_path / "c", expected_reset_seeds=(11,))
    with pytest.raises(CandidateBrokerError, match="declared sequence"):
        with broker:
            _rpc_ignoring_reset(broker.socket_path, {"op": "reset", "seed": 12})
            _assert_thread_stopped(broker)
            started = time.monotonic()
            with pytest.raises(OSError):
                _rpc(broker.socket_path, {"op": "reset", "seed": 11})
            assert time.monotonic() - started < 0.5


RESET_VIOLATION_CASES = [
    ("mismatch", (11, 11, 12), (11,), 1),
    ("extra-reset", (11, 11, 12), (11, 11, 12, 11), 4),
]


def test_broker_reset_violations_refuse_restart_and_report_safe_index(tmp_path: Path) -> None:
    # rejects reset sequence before restarting candidate
    broker = _echo_broker(tmp_path / "a", expected_reset_seeds=(11, 12))
    with pytest.raises(CandidateBrokerError, match="declared sequence"):
        with broker:
            _rpc_ignoring_reset(broker.socket_path, {"op": "reset", "seed": 12})
    assert broker.reset_seeds == ()
    assert broker.candidate_process_starts == 0

    # reset violation reports safe index and allowed count
    for case, (name, declared, accepted, index) in enumerate(RESET_VIOLATION_CASES):
        broker = _echo_broker(tmp_path / f"v{case}", expected_reset_seeds=declared)
        with pytest.raises(CandidateBrokerError) as error:
            with broker:
                for seed in accepted:
                    _rpc(broker.socket_path, {"op": "reset", "seed": seed})
                _rpc_ignoring_reset(broker.socket_path, {"op": "reset", "seed": 987_654_321})
        assert str(error.value) == (
            f"candidate reset does not match declared sequence at index {index} (allowed count 4)"
        ), name
        assert "987654321" not in str(error.value), name


DECLARED_SEQUENCE_CASES = [
    ("trailing-replay-of-first-seed", (11, 12), (11, 12, 11)),
    ("repeated-primary-seeds", (11, 11, 12, 12), (11, 11, 12, 12)),
    ("replay-after-repeated-primary", (11, 11, 12, 12), (11, 11, 12, 12, 11)),
]


def test_broker_accepts_declared_sequences_with_one_replay(tmp_path: Path) -> None:
    for index, (name, declared, sent) in enumerate(DECLARED_SEQUENCE_CASES):
        with _echo_broker(tmp_path / f"s{index}", expected_reset_seeds=declared) as broker:
            for seed in sent:
                _rpc(broker.socket_path, {"op": "reset", "seed": seed})
        assert broker.reset_seeds == sent, name
        assert broker.candidate_process_starts == len(sent), name


INVALID_LIMIT_CASES = [
    *[("completed_reset_limit", v, "completed reset limit") for v in (True, 0, -1)],
    *[("max_rpc_calls", v, "RPC call budget") for v in (True, 0, -1, 10_000_001)],
]


def test_broker_rejects_invalid_limits_and_non_integer_reset_seeds(tmp_path: Path) -> None:
    # rejects invalid limits
    for option, value, match in INVALID_LIMIT_CASES:
        with pytest.raises(ValueError, match=match):
            _echo_broker(tmp_path, **{option: value})

    # rejects non-integer reset seeds
    for index, seed in enumerate([True, 2.0, None]):
        broker = _echo_broker(tmp_path / f"n{index}")
        with pytest.raises(CandidateBrokerError, match="non-negative integer"):
            with broker:
                _rpc(broker.socket_path, {"op": "reset", "seed": seed})
        assert broker.reset_seeds == (), seed


def test_malformed_evaluator_requests_are_authored_protocol_failures(tmp_path: Path) -> None:
    # malformed JSON
    for index, payload in enumerate([b"{not-json}", b'"\xff"']):
        broker = _echo_broker(tmp_path / f"j{index}")
        with pytest.raises(CandidateBrokerError, match="invalid JSON") as raised:
            with broker:
                _raw_rpc(broker.socket_path, payload)
        assert not isinstance(raised.value, CandidateBrokerInfrastructureError), payload
        assert broker.calls == 1, payload
        assert broker.candidate_process_starts == 0, payload

    # malformed request shape
    for index, request_value in enumerate([[], {}, {"op": 1}, {"op": ""}, {"op": "reset"}]):
        broker = _echo_broker(tmp_path / f"r{index}")
        with pytest.raises(CandidateBrokerError, match="request") as raised:
            with broker:
                _rpc(broker.socket_path, request_value)
        assert not isinstance(raised.value, CandidateBrokerInfrastructureError), request_value


# --- candidate failures, stderr, and infrastructure classification -------------------------


def test_candidate_failures_classify_typed_stderr_exit_125_and_launch_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # candidate failure reports bounded typed stderr without payload
    broker = CandidateBroker(_command(FAILING_CANDIDATE), tmp_path / "a")
    with pytest.raises(CandidateBrokerError) as raised:
        with broker:
            _rpc(broker.socket_path, {"op": "reset", "seed": 3})
    message = str(raised.value)
    assert not isinstance(raised.value, CandidateBrokerInfrastructureError)
    assert "candidate_stderr=module_not_found" in message
    assert "sha256=" in message
    assert "super-secret-payload" not in message
    assert broker.candidate_diagnostic is not None
    assert broker.candidate_diagnostic.category == "module_not_found"
    assert broker.candidate_diagnostic.captured_bytes <= 64 * 1024

    # candidate runtime exit 125 is infrastructure
    broker = CandidateBroker([sys.executable, "-c", "raise SystemExit(125)"], tmp_path / "b")
    with pytest.raises(CandidateBrokerInfrastructureError, match="candidate_exit_code=125") as raised:
        with broker:
            _rpc(broker.socket_path, {"op": "reset", "seed": 5})
    assert str(raised.value).count("candidate_exit_code=125") == 1

    # candidate runtime launch failure is infrastructure
    def fail_launch(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError("runtime unavailable")

    monkeypatch.setattr(f"{BROKER_MODULE}.subprocess.Popen", fail_launch)
    broker = CandidateBroker(["docker", "run"], tmp_path / "c")
    with pytest.raises(CandidateBrokerInfrastructureError, match="failed to launch"):
        with broker:
            _rpc(broker.socket_path, {"op": "reset", "seed": 5})


def test_reset_timeouts_and_stderr_flood_are_bounded(tmp_path: Path) -> None:
    # reset response timeout is bounded by RPC timeout
    for name, source in (("slow", SLOW_CANDIDATE), ("partial", PARTIAL_RESPONSE_CANDIDATE)):
        broker = CandidateBroker(_command(source), tmp_path / name, rpc_timeout_seconds=0.1)
        started = time.monotonic()
        with pytest.raises(CandidateBrokerError, match="candidate reset timed out"):
            with broker:
                _rpc(broker.socket_path, {"op": "reset", "seed": 5})
        assert time.monotonic() - started < 1, name

    # candidate stderr flood terminates as infrastructure failure
    broker = CandidateBroker(
        _command(STDERR_FLOOD_CANDIDATE), tmp_path / "flood", rpc_timeout_seconds=2
    )
    started = time.monotonic()
    with pytest.raises(CandidateBrokerInfrastructureError, match="stderr output limit") as raised:
        with broker:
            _rpc(broker.socket_path, {"op": "reset", "seed": 5})
    assert time.monotonic() - started < 3
    assert "sensitive-flood-payload" not in str(raised.value)
    assert broker._process is not None
    assert broker._process.poll() is not None


def test_broker_cleanup_does_not_mask_primary_lifecycle_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # recorded secondary error does not mask the primary error
    broker = _echo_broker(tmp_path / "a")
    with pytest.raises(RuntimeError, match="primary evaluator failure"):
        with broker:
            broker._error = BrokenPipeError("secondary peer disconnect")
            raise RuntimeError("primary evaluator failure")

    # throwing process cleanup does not mask the primary error
    broker = _echo_broker(tmp_path / "b")
    with pytest.raises(RuntimeError, match="primary evaluator failure"):
        with broker:
            _rpc(broker.socket_path, {"op": "reset", "seed": 3})
            assert broker._process is not None

            def fail_cleanup() -> None:
                raise OSError("secondary cleanup failure")

            monkeypatch.setattr(broker._process, "terminate", fail_cleanup)
            raise RuntimeError("primary evaluator failure")

    assert broker._process is not None
    assert broker._process.poll() is not None
    assert not broker.socket_path.exists()


def test_broker_internal_diagnostics_are_bounded_and_redacted(tmp_path: Path) -> None:
    # records peer disconnect without raising
    class DisconnectedPeer:
        def sendall(self, _: bytes) -> None:
            raise BrokenPipeError("peer closed")

    broker = _echo_broker(tmp_path)
    broker._send_response(DisconnectedPeer(), b'{"ok":true}')
    assert broker.peer_disconnects == 1
    assert broker._error is None

    # candidate stderr retention is bounded
    broker = _echo_broker(tmp_path)
    payload = b"x" * (80 * 1024) + b"SyntaxError\n"
    broker._capture_stderr(io.BytesIO(payload))
    diagnostic = broker.candidate_diagnostic
    assert diagnostic is not None
    assert diagnostic.category == "syntax_error"
    assert diagnostic.total_bytes == len(payload)
    assert diagnostic.captured_bytes == 64 * 1024
    assert diagnostic.truncated is True

    # first infrastructure error cannot be overwritten by serving thread
    broker = _echo_broker(tmp_path)
    start = threading.Barrier(2)
    infrastructure_recorded = threading.Event()

    def stderr_thread() -> None:
        start.wait()
        broker._record_error_once(
            CandidateBrokerInfrastructureError("candidate stderr output limit exceeded")
        )
        infrastructure_recorded.set()

    def serving_thread() -> None:
        start.wait()
        assert infrastructure_recorded.wait(timeout=1)
        broker._record_error_once(CandidateBrokerError("candidate response missing"))

    threads = (threading.Thread(target=stderr_thread), threading.Thread(target=serving_thread))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
    assert all(not thread.is_alive() for thread in threads)
    assert isinstance(broker._error, CandidateBrokerInfrastructureError)
    assert str(broker._error) == "candidate stderr output limit exceeded"

    # candidate cleanup diagnostic does not expose exception path
    class FailingServer:
        def close(self) -> None:
            raise PermissionError("/sensitive/broker/socket")

    broker = _echo_broker(tmp_path)
    broker._server = FailingServer()  # type: ignore[assignment]
    with pytest.raises(CandidateBrokerInfrastructureError) as raised:
        broker.__exit__(None, None, None)
    diagnostic_text = str(raised.value)
    assert "PermissionError" in diagnostic_text
    assert "/sensitive/broker/socket" not in diagnostic_text

    # unexpected broker diagnostic does not expose exception path
    broker = _echo_broker(tmp_path)
    error = broker._error_with_diagnostic(PermissionError("/sensitive/broker/candidate.cid"))
    assert isinstance(error, CandidateBrokerInfrastructureError)
    assert "PermissionError" in str(error)
    assert "/sensitive/broker/candidate.cid" not in str(error)

    # container id rejects noncanonical prefixes
    cid_path = tmp_path / "candidate.cid"
    for container_id in ["a", "a" * 11, "a" * 65, "A" * 64]:
        cid_path.write_text(container_id, encoding="ascii")
        with pytest.raises(CandidateBrokerInfrastructureError, match="invalid"):
            read_container_id(cid_path)


# --- container removal on exit --------------------------------------------------------------


def _wait_timeout_once(process: _FakeProcess, timeout: float) -> None:
    if process.wait_calls == 1:
        raise subprocess.TimeoutExpired("docker run", timeout)


RECOVERABLE_REMOVAL_CASES = [
    ("already-removed", 1, "Error: No such container: candidate", lambda: _FakeProcess()),
    ("wait-timeout-recovered", 0, "", lambda: _FakeProcess(poll=None, on_wait=_wait_timeout_once)),
]


def test_container_removal_on_exit_outcomes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # candidate removal tolerates recoverable outcomes
    for index, (name, returncode, stderr, process) in enumerate(RECOVERABLE_REMOVAL_CASES):
        with monkeypatch.context() as m:
            _docker_rm(m, returncode, stderr)
            broker = _echo_broker(tmp_path / f"r{index}")
            with broker:
                broker._process = process()  # type: ignore[assignment]
                broker.cid_path.write_text("a" * 64, encoding="ascii")
            assert not broker.socket_dir.exists(), name

    # late candidate id is removed after process wait
    broker = _echo_broker(tmp_path / "l")
    removed: list[str] = []

    def remove(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        removed.append(command[-1])
        return subprocess.CompletedProcess(command, 0, "", "")

    with monkeypatch.context() as m:
        m.setattr(f"{BROKER_MODULE}.subprocess.run", remove)
        with broker:
            broker._process = _FakeProcess(  # type: ignore[assignment]
                poll=None, on_wait=lambda *_: broker.cid_path.write_text("b" * 64, encoding="ascii")
            )
    assert removed == ["b" * 64]
    assert not broker.socket_dir.exists()

    # failed container removal retains candidate recovery metadata
    _docker_rm(monkeypatch, 1, "daemon unavailable")
    broker = _echo_broker(tmp_path / "f")
    with pytest.raises(CandidateBrokerInfrastructureError, match="remove"):
        with broker:
            broker._process = _FakeProcess()  # type: ignore[assignment]
            broker.cid_path.write_text("a" * 64, encoding="ascii")
    assert broker.cid_path.is_file()
    assert broker.socket_dir.is_dir()
    assert (broker.socket_dir / "scoring-mask").is_dir()
    assert not broker.socket_path.exists()


# --- bounded teardown diagnostics (recovery artifacts retained, paths redacted) --------------


def test_metadata_cleanup_failures_report_operation_errno_and_redact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # nonempty runtime directory reports errno and retains unknown file
    broker = _bare_broker(tmp_path / "nonempty", mkdir=True)
    unknown = broker.socket_dir / "private-evaluator-container-id"
    unknown.write_text("private-recovery-data", encoding="utf-8")
    with pytest.raises(CandidateBrokerInfrastructureError) as raised:
        broker.__exit__(None)
    diagnostic = str(raised.value)
    assert isinstance(raised.value, ContainerMetadataCleanupError)
    assert "operation=runtime_directory_remove" in diagnostic
    assert f"errno={errno.ENOTEMPTY} (ENOTEMPTY)" in diagnostic
    assert unknown.read_text(encoding="utf-8") == "private-recovery-data"
    assert str(tmp_path) not in diagnostic
    assert unknown.name not in diagnostic
    assert "private-recovery-data" not in diagnostic

    # unlink failure reports operation and redacts path
    original_unlink = Path.unlink
    for target, operation in [("candidate.sock", "socket_unlink"), ("candidate.cid", "candidate_cid_unlink")]:
        broker = _bare_broker(tmp_path / operation, mkdir=True)

        def fail_unlink(path: Path, missing_ok: bool = False, *, _broker=broker, _target=target) -> None:
            if path == _broker.socket_dir / _target:
                raise PermissionError(errno.EACCES, "private-candidate-text", "/private/socket")
            original_unlink(path, missing_ok=missing_ok)

        with monkeypatch.context() as m:
            m.setattr(Path, "unlink", fail_unlink)
            with pytest.raises(CandidateBrokerInfrastructureError) as raised:
                broker.__exit__(None)
        diagnostic = str(raised.value)
        assert isinstance(raised.value, ContainerMetadataCleanupError), operation
        assert f"operation={operation}" in diagnostic
        assert f"errno={errno.EACCES} (EACCES)" in diagnostic, operation
        assert "PermissionError" in diagnostic, operation
        assert "private" not in diagnostic, operation
        assert str(tmp_path) not in diagnostic, operation

    # scoring mask removal reports original errno
    broker = _bare_broker(tmp_path / "mask", mkdir=True)
    (broker.socket_dir / "scoring-mask").mkdir()

    def fail_rmtree(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(errno.EACCES, "private-candidate-text", "/private/mask")

    monkeypatch.setattr(f"{BROKER_MODULE}.shutil.rmtree", fail_rmtree)
    with pytest.raises(CandidateBrokerInfrastructureError) as raised:
        broker.__exit__(None)
    diagnostic = str(raised.value)
    assert isinstance(raised.value, ContainerMetadataCleanupError)
    assert "operation=scoring_mask_remove" in diagnostic
    assert f"errno={errno.EACCES} (EACCES)" in diagnostic
    assert "private" not in diagnostic
    assert (broker.socket_dir / "scoring-mask").is_dir()


def test_lifecycle_exit_failures_classify_certainty_and_retain_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # lifecycle failure reports operation and redacts message
    for operation in ["server_close", "process_stop", "thread_join"]:
        broker = _bare_broker(tmp_path / operation)
        failing_action = Mock(side_effect=OSError(errno.EIO, "private-candidate-text", "/private/runtime"))
        if operation == "server_close":
            broker._server = Mock(close=failing_action)
        elif operation == "process_stop":
            broker._process = Mock()
            monkeypatch.setattr(broker, "_stop_process", failing_action)
        else:
            broker._thread = Mock(join=failing_action)
        with pytest.raises(CandidateBrokerInfrastructureError) as raised:
            broker.__exit__(None)
        diagnostic = str(raised.value)
        expected_type = (
            CandidateContainerCleanupError
            if operation in {"process_stop", "thread_join"}
            else ContainerMetadataCleanupError
        )
        assert isinstance(raised.value, expected_type), operation
        assert f"operation={operation}" in diagnostic
        assert f"errno={errno.EIO} (EIO)" in diagnostic, operation
        assert "private" not in diagnostic, operation

    # container removal failure takes priority and retains metadata
    broker = _bare_broker(tmp_path / "priority", mkdir=True)
    broker.cid_path.write_text("a" * 64, encoding="ascii")
    broker.socket_path.touch()
    (broker.socket_dir / "scoring-mask").mkdir()
    broker._server = Mock(close=Mock(side_effect=PermissionError(errno.EACCES, "private")))
    broker._process = Mock()
    monkeypatch.setattr(
        broker,
        "_stop_process",
        Mock(side_effect=CandidateBrokerInfrastructureError(
            "candidate container runtime could not remove container"
        )),
    )
    with pytest.raises(CandidateBrokerInfrastructureError) as raised:
        broker.__exit__(RuntimeError)
    assert isinstance(raised.value, CandidateContainerCleanupError)
    assert str(raised.value) == (
        "candidate container cleanup failed: "
        "candidate container runtime could not remove container; recovery metadata retained"
    )
    assert broker.cid_path.read_text(encoding="ascii") == "a" * 64
    assert (broker.socket_dir / "scoring-mask").is_dir()
    assert not broker.socket_path.exists()

    # raw process stop error retains recovery metadata
    broker = _bare_broker(tmp_path / "raw", mkdir=True)
    broker.cid_path.write_text("a" * 64)
    (broker.socket_dir / "scoring-mask").mkdir()
    broker._process = Mock()
    monkeypatch.setattr(broker, "_stop_process", Mock(side_effect=OSError(errno.EIO, "/private")))
    with pytest.raises(CandidateContainerCleanupError) as raised:
        broker.__exit__(ValueError)
    assert broker.cid_path.exists()
    assert (broker.socket_dir / "scoring-mask").exists()
    assert "/private" not in str(raised.value)

    # verified stop metadata error is not promoted to uncertainty
    broker = _bare_broker(tmp_path / "verified")
    broker._process = Mock()
    monkeypatch.setattr(broker, "_stop_process", Mock(side_effect=ContainerMetadataCleanupError("safe")))
    with pytest.raises(ContainerMetadataCleanupError):
        broker.__exit__(None)


def test_restart_and_stop_process_preserve_shutdown_certainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # diagnostic enrichment preserves cleanup certainty
    for error_type in [CandidateContainerCleanupError, ContainerMetadataCleanupError]:
        broker = _bare_broker(tmp_path / "enrich")
        broker._process = Mock(returncode=125)
        broker._stderr_total_bytes = 10
        broker._stderr_tail.extend(b"private stderr")
        error = broker._error_with_diagnostic(error_type("safe diagnostic"))
        assert type(error) is error_type
        assert str(error) == "safe diagnostic", error_type

    # restart preserves uncertain shutdown type
    broker = _bare_broker(tmp_path / "restart")
    broker._process = Mock()
    monkeypatch.setattr(broker, "_stop_process", Mock(side_effect=OSError(errno.EIO, "/private")))
    with pytest.raises(CandidateContainerCleanupError):
        broker._restart_process(time.monotonic() + 1)

    # restart metadata failure does not imply live container
    broker = _bare_broker(tmp_path / "restart-metadata")
    broker._process = Mock()
    monkeypatch.setattr(broker, "_stop_process", Mock())
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", Mock(side_effect=PermissionError(errno.EACCES, "/private")))
        with pytest.raises(ContainerMetadataCleanupError) as raised:
            broker._restart_process(time.monotonic() + 1)
    assert "/private" not in str(raised.value)

    # stop_process unverified wait has uncertain shutdown type
    for failure in [
        subprocess.TimeoutExpired("private command", 1), PermissionError(errno.EACCES, "/private"),
    ]:
        broker = _bare_broker(tmp_path / "unverified")
        process = Mock(wait=Mock(side_effect=failure))
        with pytest.raises(CandidateContainerCleanupError) as cleanup_error:
            broker._stop_process(process)
        assert "private" not in str(cleanup_error.value), failure

    # stop_process rechecks late cid and preserves uncertainty
    broker = _bare_broker(tmp_path / "late-cid", mkdir=True)
    process = Mock(wait=Mock(side_effect=lambda **_: broker.cid_path.write_text("a" * 64)))
    remove = Mock(return_value=Mock(returncode=1, stderr="private daemon error"))
    monkeypatch.setattr(f"{BROKER_MODULE}.subprocess.run", remove)
    with pytest.raises(CandidateContainerCleanupError):
        broker._stop_process(process)
    assert remove.call_count == 1
    assert broker.cid_path.exists()


def test_launch_thread_and_late_candidate_teardown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # late candidate after thread join is stopped before metadata removal
    for name, initial_process in [("none", None), ("mock", Mock())]:
        broker = _bare_broker(tmp_path / f"late-{name}", mkdir=True)
        broker._process = initial_process
        replacement = Mock()
        stopped: list[object] = []
        deadlines: list[float] = []

        def finish_launch(_broker=broker, _replacement=replacement, **kwargs):
            assert 0 < kwargs["timeout"] <= 2
            _broker._process = _replacement
            _broker.cid_path.write_text("a" * 64)

        def stop(process, *, deadline, _broker=broker, _replacement=replacement, _stopped=stopped, _deadlines=deadlines):
            _deadlines.append(deadline)
            _stopped.append(process)
            if process is _replacement:
                assert _broker.cid_path.exists()

        broker._thread = Mock(join=Mock(side_effect=finish_launch), is_alive=Mock(return_value=False))
        monkeypatch.setattr(broker, "_stop_process", stop)
        broker.__exit__(None)
        assert stopped == ([initial_process, replacement] if initial_process else [replacement]), name
        assert len(set(deadlines)) == 1, name
        assert not broker.socket_dir.exists(), name

    # live launch thread retains all recovery metadata
    broker = _bare_broker(tmp_path / "live", mkdir=True)
    broker.cid_path.write_text("a" * 64)
    unknown = broker.socket_dir / "unknown"
    unknown.touch()
    broker._thread = Mock(is_alive=Mock(return_value=True))
    remove = Mock(side_effect=PermissionError(errno.EACCES, "/private"))
    with monkeypatch.context() as m:
        m.setattr(Path, "rmdir", remove)
        with pytest.raises(CandidateContainerCleanupError):
            broker.__exit__(None)
    assert broker.cid_path.exists()
    assert unknown.exists()
    remove.assert_not_called()

    # late candidate shutdown failure retains metadata
    broker = _bare_broker(tmp_path / "late-failure", mkdir=True)

    def finish_late_launch(**_):
        broker._process = Mock()
        broker.cid_path.write_text("a" * 64)

    broker._thread = Mock(join=Mock(side_effect=finish_late_launch), is_alive=Mock(return_value=False))
    monkeypatch.setattr(broker, "_stop_process", Mock(side_effect=OSError(errno.EIO, "/private")))
    with pytest.raises(CandidateContainerCleanupError):
        broker.__exit__(None)
    assert broker.cid_path.exists()


# --- opt-in operation deadlines via the candidate proxy -------------------------------------


def _timeout_broker(tmp_path: Path, source: str = TIMEOUT_WORKER, **kwargs: object) -> CandidateBroker:
    kwargs.setdefault("rpc_timeout_seconds", 2)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return CandidateBroker(_command(source), tmp_path / "broker", **kwargs)  # type: ignore[arg-type]


def _connect_proxy(monkeypatch: pytest.MonkeyPatch, broker: CandidateBroker) -> None:
    monkeypatch.setenv("AGENTIC_CANDIDATE_SOCKET", str(broker.socket_path))
    monkeypatch.setenv("AGENTIC_CANDIDATE_RPC_TIMEOUT", "2")


def test_operation_timeout_recovery_and_forged_authority(tmp_path, monkeypatch):
    # operation timeout kills old process then accepts declared reset
    with _timeout_broker(tmp_path / "a", expected_reset_seeds=(7, 7)) as broker:
        _connect_proxy(monkeypatch, broker)
        first = proxy.handle({"op": "reset", "seed": 7})
        old_process = broker._process
        with pytest.raises(proxy.CandidateOperationTimeout):
            proxy.handle({"op": "sleep"}, timeout_seconds=0.05)
        assert old_process.poll() is not None
        second = proxy.handle({"op": "reset", "seed": 7})
        assert first["pid"] != second["pid"]
        result = proxy.handle({"op": "echo", "payload": 9}, timeout_seconds=1)
        assert result["request"] == {"op": "echo", "payload": 9}
    assert broker.reset_seeds == (7, 7)
    assert broker.shutdown_verified

    # partial response is discarded before recovery
    with _timeout_broker(tmp_path / "b", expected_reset_seeds=(7, 7)) as broker:
        _connect_proxy(monkeypatch, broker)
        proxy.handle({"op": "reset", "seed": 7})
        with pytest.raises(proxy.CandidateOperationTimeout):
            proxy.handle({"op": "partial"}, timeout_seconds=0.05)
        assert proxy.handle({"op": "reset", "seed": 7})["request"]["seed"] == 7
        assert proxy.handle({"op": "echo"}, timeout_seconds=1)["request"] == {"op": "echo"}

    # candidate cannot forge broker timeout authority
    with _timeout_broker(tmp_path / "c", expected_reset_seeds=(7, 7)) as broker:
        _connect_proxy(monkeypatch, broker)
        proxy.handle({"op": "reset", "seed": 7})
        with pytest.raises(RuntimeError, match="invalid candidate RPC response"):
            proxy.handle({"op": "spoof"}, timeout_seconds=1)
        assert broker._process.poll() is None


def test_proxy_rejects_invalid_deadlines_before_socket_use():
    # invalid operation deadline values
    for value in [True, 0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="timeout"):
            proxy.handle({"op": "echo"}, timeout_seconds=value)

    # opt-in reset timeout is rejected
    with pytest.raises(ValueError, match="reset"):
        proxy.handle({"op": "reset", "seed": 7}, timeout_seconds=1)


POST_TIMEOUT_FOLLOW_UP_CASES = [
    ("nonreset-rejected", {"op": "echo", "timeout_seconds": 1}, "reset required"),
    ("undeclared-reset-rejected", {"op": "reset", "seed": 8}, "declared sequence"),
]


def test_requests_after_timeout_and_oversized_deadlines_are_rejected(tmp_path, monkeypatch):
    # requests after operation timeout must follow declared reset
    for index, (name, follow_up, match) in enumerate(POST_TIMEOUT_FOLLOW_UP_CASES):
        follow_up = dict(follow_up)
        timeout = follow_up.pop("timeout_seconds", None)
        with pytest.raises(CandidateBrokerError, match=match):
            with _timeout_broker(tmp_path / f"f{index}", expected_reset_seeds=(7, 7)) as broker:
                _connect_proxy(monkeypatch, broker)
                proxy.handle({"op": "reset", "seed": 7})
                with pytest.raises(proxy.CandidateOperationTimeout):
                    proxy.handle({"op": "sleep"}, timeout_seconds=0.05)
                with pytest.raises((RuntimeError, OSError)):
                    proxy.handle(follow_up, timeout_seconds=timeout)
                assert broker.candidate_process_starts == 1, name

    # broker rejects deadline above global limit
    with pytest.raises(CandidateBrokerError, match="configured RPC timeout"):
        with _timeout_broker(tmp_path / "g", expected_reset_seeds=(7, 7)) as broker:
            _connect_proxy(monkeypatch, broker)
            with pytest.raises((RuntimeError, OSError)):
                proxy.handle({"op": "echo"}, timeout_seconds=3)
            assert broker.candidate_process_starts == 0


def test_timeout_cleanup_failure_and_blocked_request_write(tmp_path, monkeypatch):
    # cleanup failure never becomes recoverable timeout
    broker = _timeout_broker(tmp_path / "a", expected_reset_seeds=(7, 7))
    original = broker._stop_process
    stops = 0

    def fail_first_stop(process, *, deadline=None):
        nonlocal stops
        stops += 1
        if stops == 1:
            raise CandidateContainerCleanupError("injected unresolved container")
        return original(process, deadline=deadline)

    monkeypatch.setattr(broker, "_stop_process", fail_first_stop)
    with pytest.raises(CandidateContainerCleanupError, match="unresolved container"):
        with broker:
            _connect_proxy(monkeypatch, broker)
            proxy.handle({"op": "reset", "seed": 7})
            with pytest.raises(RuntimeError, match="ended early"):
                proxy.handle({"op": "sleep"}, timeout_seconds=0.05)

    # operation deadline bounds blocked request write
    source = """
import json, sys, time
request = json.loads(sys.stdin.readline())
print(json.dumps({'ok':True, 'value':request}), flush=True)
time.sleep(10)
"""
    with _timeout_broker(tmp_path / "b", source) as broker:
        _connect_proxy(monkeypatch, broker)
        proxy.handle({"op": "reset", "seed": 7})
        process = broker._process
        with pytest.raises(proxy.CandidateOperationTimeout):
            proxy.handle({"op": "echo", "payload": "a" * (1024 * 1024)}, timeout_seconds=0.05)
        assert process.poll() is not None


# --- candidate worker module loading --------------------------------------------------------


@pytest.fixture
def restore_candidate_module() -> Iterator[None]:
    previous = sys.modules.get(_CANDIDATE_MODULE_NAME)
    yield
    if previous is None:
        sys.modules.pop(_CANDIDATE_MODULE_NAME, None)
    else:
        sys.modules[_CANDIDATE_MODULE_NAME] = previous


def _write_candidate(tmp_path: Path, source: str) -> Path:
    tmp_path.mkdir(exist_ok=True)
    candidate = tmp_path / "program.py"
    candidate.write_text(source, encoding="utf-8")
    return candidate


def test_load_handle_registers_serves_and_restores_candidate_module(
    tmp_path: Path, restore_candidate_module
) -> None:
    # load_handle registers dataclass candidate
    handle = _load_handle(_write_candidate(tmp_path / "dataclass", DATACLASS_CANDIDATE))
    assert handle({"value": 7}) == 7
    assert sys.modules[_CANDIDATE_MODULE_NAME].handle is handle

    # candidate worker serves dataclass candidate
    completed = subprocess.run(
        [sys.executable, "-m", "src.final_evaluation.candidate_worker", str(tmp_path / "dataclass" / "program.py")],
        input='{"value":7}\n',
        capture_output=True,
        check=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"ok": True, "value": 7}

    # load_handle restores previous module after failure
    for index, (source, error) in enumerate([
        ("raise RuntimeError('broken import')", "broken import"),
        ("VALUE = 1", "candidate must define handle"),
    ]):
        previous = ModuleType(_CANDIDATE_MODULE_NAME)
        sys.modules[_CANDIDATE_MODULE_NAME] = previous
        with pytest.raises(Exception, match=error):
            _load_handle(_write_candidate(tmp_path / f"restore-{index}", source))
        assert sys.modules[_CANDIDATE_MODULE_NAME] is previous, source

    # load_handle removes new module after failure
    sys.modules.pop(_CANDIDATE_MODULE_NAME, None)
    with pytest.raises(RuntimeError, match="broken import"):
        _load_handle(_write_candidate(tmp_path / "removed", "raise RuntimeError('broken import')"))
    assert _CANDIDATE_MODULE_NAME not in sys.modules


# --- broker socket storage (survives login IPC cleanup, legacy roots fail closed) -----------


def _make_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    from src.final_evaluation import broker_storage

    current, legacy = tmp_path / "current", tmp_path / "legacy"
    monkeypatch.setattr(broker_storage, "legacy_broker_root", lambda: legacy)
    monkeypatch.setattr(runner, "_broker_recovery_root", lambda: current)
    return current, legacy


UNSAFE_SYSTEM_TEMP_CASES = [
    (stat.S_IFLNK | 0o777, 0),
    (stat.S_IFDIR | 0o777, 0),
    (stat.S_IFDIR | 0o1777, 1000),
]


def test_socket_root_ignores_temp_storage_and_rejects_unsafe_paths(tmp_path, monkeypatch):
    # candidate socket root ignores temp and IPC storage
    with monkeypatch.context() as m:
        m.setenv("TMPDIR", "/shared/" + "long-" * 50)
        assert runner._broker_recovery_root() == Path(f"/tmp/agentic-scientist-broker-{os.getuid()}")
        directory = runner._final_broker_directory(Path("/results/final"))
        assert len(os.fsencode(directory / "candidate.sock")) <= 107

    # private socket root rejects unsafe existing path
    for kind in ["symlink", "permissions", "file", "owner"]:
        with monkeypatch.context() as m:
            current, _ = _make_roots(tmp_path / kind, m)
            current.parent.mkdir(exist_ok=True)
            if kind == "symlink":
                current.symlink_to(current.parent, target_is_directory=True)
            elif kind == "file":
                current.write_text("occupied")
            else:
                current.mkdir(mode=0o700 if kind == "owner" else 0o755)
                if kind == "owner":
                    m.setattr(os, "getuid", lambda: current.stat().st_uid + 1)
            with pytest.raises(runner.FinalEvaluationError, match="unsafe broker"):
                runner._ensure_broker_recovery_root()

    # socket root rejects unsafe system temp directory
    original = Path.lstat
    for index, (mode, owner) in enumerate(UNSAFE_SYSTEM_TEMP_CASES):
        with monkeypatch.context() as m:
            current, _ = _make_roots(tmp_path / f"temp-{index}", m)

            def metadata(path, *args, _mode=mode, _owner=owner, **kwargs):
                if path == Path("/tmp"):
                    return SimpleNamespace(st_mode=_mode, st_uid=_owner)
                return original(path, *args, **kwargs)

            m.setattr(Path, "lstat", metadata)
            with pytest.raises(runner.FinalEvaluationError, match="unsafe broker temporary"):
                runner._ensure_broker_recovery_root()
            assert not current.exists(), (mode, owner)


def test_legacy_broker_recovery_fails_closed_and_ignores_unrelated_work(tmp_path, monkeypatch):
    # unrelated legacy work does not block new broker
    current, legacy = _make_roots(tmp_path / "unrelated", monkeypatch)
    current.parent.mkdir()
    legacy.mkdir(mode=0o700)
    unrelated = legacy / "final-unrelated"
    unrelated.mkdir(mode=0o700)
    runner._recover_broker_directory(current / "final-current")
    assert unrelated.is_dir()

    # socket remains connectable after legacy IPC root disappears
    current, legacy = _make_roots(tmp_path / "connectable", monkeypatch)
    current.parent.mkdir()
    runner._ensure_broker_recovery_root()
    legacy.mkdir(mode=0o700)
    address = current / "candidate.sock"
    with socket.socket(socket.AF_UNIX) as server, socket.socket(socket.AF_UNIX) as client:
        server.bind(str(address))
        server.listen(1)
        legacy.rmdir()
        client.connect(str(address))
        accepted, _ = server.accept()
        accepted.close()

    # matching legacy metadata blocks recovery without removal
    current, legacy = _make_roots(tmp_path / "matching", monkeypatch)
    current.parent.mkdir()
    legacy.mkdir(mode=0o700)
    old = legacy / "final-example"
    old.mkdir(mode=0o700)
    cid = old / "candidate.cid"
    cid.write_text("a" * 64)
    with pytest.raises(runner.FinalEvaluationError, match="legacy broker"):
        runner._recover_broker_directory(current / old.name)
    assert cid.read_text() == "a" * 64
    assert not current.exists()

    # legacy symlinks fail closed
    for target in ["root", "matching_directory"]:
        current, legacy = _make_roots(tmp_path / f"symlink-{target}", monkeypatch)
        current.parent.mkdir()
        if target == "root":
            legacy.symlink_to(legacy.parent, target_is_directory=True)
        else:
            legacy.mkdir(mode=0o700)
            (legacy / "final-example").symlink_to("missing", target_is_directory=True)
        with pytest.raises(runner.FinalEvaluationError, match="legacy|unsafe broker"):
            runner._recover_broker_directory(current / "final-example")


def test_launch_paths_reject_matching_legacy_metadata_before_new_root_exists(tmp_path, monkeypatch):
    from src.investigation import task_runtime
    from src.variants_pipeline import host_eval

    # final launch rejects legacy before new root exists
    current, legacy = _make_roots(tmp_path / "final", monkeypatch)
    current.parent.mkdir()
    output = current.parent / "final-output"
    legacy.mkdir(mode=0o700)
    old = legacy / runner._final_broker_directory(output).name
    old.mkdir(mode=0o700)
    with pytest.raises(runner.FinalEvaluationError, match="legacy broker"):
        runner.run_final_evaluation(
            task_dir=current.parent / "task", investigation_results_dir=current.parent / "study",
            output_dir=output, python_bin=Path("/unused/python"),
        )
    assert old.is_dir()
    assert not current.exists()

    # runtime recovery preserves matching legacy metadata
    for naming in ["current", "older"]:
        current, legacy = _make_roots(tmp_path / f"runtime-{naming}", monkeypatch)
        current.parent.mkdir()
        monkeypatch.setattr(task_runtime, "_broker_recovery_root", lambda _current=current: _current)
        task = current.parent / "task"
        directory = (task_runtime._task_generation_broker_directory if naming == "current"
                     else task_runtime._legacy_task_generation_broker_directory)(task, "runtime-search")
        legacy.mkdir(mode=0o700)
        old = legacy / directory.name
        old.mkdir(mode=0o700)
        with pytest.raises(task_runtime.CandidateBrokerInfrastructureError, match="legacy broker"):
            task_runtime._prepare_task_broker_directory(
                task, "runtime-search", deadline=time.monotonic() + 10,
            )
        assert old.is_dir(), naming
        assert not directory.exists(), naming

    # search stale recovery checks legacy before new metadata
    current, legacy = _make_roots(tmp_path / "search", monkeypatch)
    current.parent.mkdir()
    monkeypatch.setattr(host_eval, "_broker_recovery_root", lambda: current)
    lease = SimpleNamespace(identity="a" * 24, recovery_dir=current.parent / "sandbox")
    legacy.mkdir(mode=0o700)
    old = legacy / host_eval._search_broker_directory(lease).name
    old.mkdir(mode=0o700)
    with pytest.raises(host_eval.SandboxTeardownError, match="legacy broker"):
        host_eval._recover_stale_sandbox(lease)
    assert old.is_dir()
    assert not current.exists()


# --- final evaluator launcher ----------------------------------------------------------------


def test_launcher_records_start_before_exec_and_rejects_external_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # records start before exec
    monkeypatch.setenv("AGENTIC_EVALUATOR_STARTED_SOCKET", "/broker/evaluator-start.sock")
    events: list[object] = []

    class FakeSocket:
        def connect(self, path: str) -> None:
            events.append(call.connect(path))

        def sendall(self, payload: bytes) -> None:
            events.append(call.sendall(payload))

        def close(self) -> None:
            events.append(call.close())

    def execv(executable: str, arguments: list[str]) -> None:
        events.append(call.execv(executable, arguments))
        raise RuntimeError("exec intercepted")

    with (
        patch("src.final_evaluation.evaluator_launcher.socket.socket", return_value=FakeSocket()),
        patch("src.final_evaluation.evaluator_launcher.os.execv", side_effect=execv),
    ):
        with pytest.raises(RuntimeError, match="exec intercepted"):
            launcher_main(["/runtime/python", "/task/evaluate_final.py"])

    assert events == [
        call.connect("/broker/evaluator-start.sock"),
        call.sendall(b"evaluator-started-v1"),
        call.close(),
        call.execv("/runtime/python", ["/runtime/python", "/task/evaluate_final.py"]),
    ]

    # rejects external witness path
    monkeypatch.setenv("AGENTIC_EVALUATOR_STARTED_SOCKET", "/tmp/started")
    with pytest.raises(SystemExit, match="invalid final evaluator launch witness"):
        launcher_main(["/runtime/python"])
