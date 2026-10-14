"""Task gate protocol, socket runtime, startup/cleanup, launcher, and RPC transport policy."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from src.task_drafting import gate, rpc_transport_policy, socket_runtime
from src.task_drafting.gate import (
    ALLOWED_GATES,
    GateRequestError,
    GateServer,
    TaskGateReport,
    parse_gate_report,
    parse_smoke_markers,
    request_gate,
    sanitize_diagnostic,
)
from src.task_drafting.rpc_transport_policy import isolated_rpc_transport_findings
from src.task_drafting.service import RuntimeRecordingExecutor

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "secret-token"
STANDALONE_CLIENT = ROOT / "src/task_drafting/standalone_client.py"


def _passed(operation: str, letter: str = "f") -> TaskGateReport:
    return TaskGateReport.passed(operation, "sha256:" + letter * 64, 0.1)


def _authored_failure(**overrides: object) -> TaskGateReport:
    fields: dict = dict(
        gate="final-smoke", component="final_evaluator", stage="execution",
        category="final_nonzero_exit", scaffold_fingerprint="sha256:" + "f" * 64,
        exit_code=3, elapsed_seconds=0.1, diagnostic="failed",
    )
    fields.update(overrides)
    return TaskGateReport.authored_failure(**fields)


def _forbidden_execute(operation: str) -> NoReturn:
    pytest.fail("this gate server must not execute a gate operation")


@contextlib.contextmanager
def _serving(
    socket_path: Path,
    executor: Callable[[str], TaskGateReport] = _passed,
    *,
    max_requests: int = 1,
    **kwargs: Any,
) -> Iterator[GateServer]:
    server = GateServer(
        socket_path, token=TOKEN, executor=executor, max_requests=max_requests, **kwargs
    )
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    server.wait_until_ready()
    try:
        yield server
    finally:
        server.close()
        thread.join(timeout=2)


def _connect(socket_path: Path) -> socket.socket:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(socket_path))
    return client


def _standalone_client(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(STANDALONE_CLIENT), "acceptance-smoke"],
        env=env, capture_output=True, text=True, check=False,
    )


# ---------------------------------------------------------------------------
# Gate report parsing and diagnostics
# ---------------------------------------------------------------------------


def test_gate_report_payload_round_trips_and_rejects_malformed_typed_fields() -> None:
    # acceptance-smoke is an additive gate operation
    assert "acceptance-smoke" in ALLOWED_GATES

    # parse_gate_report preserves the exact failing sub-gate
    report = _authored_failure()
    parsed = parse_gate_report([b"noise\n", json.dumps(report.to_payload()).encode() + b"\n"])
    assert parsed == report

    # parse_gate_report rejects malformed typed fields
    for field, value in [
        ("exit_code", 0), ("scaffold_fingerprint", "short"), ("elapsed_seconds", float("inf")),
        ("category", 7),
    ]:
        payload = _authored_failure().to_payload()
        payload[field] = value
        assert parse_gate_report([json.dumps(payload).encode()]) is None, field

    # the report payload omits sealed values
    payload = _authored_failure(
        component="evaluator", stage="candidate_rpc", scaffold_fingerprint="sha256:" + "a" * 64,
        category="final_evaluator_nonzero", diagnostic="candidate proxy exited before reset",
        elapsed_seconds=1.25,
    ).to_payload()
    assert payload["status"] == "authored_failure"
    assert set(payload) == {
        "schema_version", "status", "gate", "component", "stage", "category",
        "scaffold_fingerprint", "exit_code", "elapsed_seconds", "diagnostic",
    }


def test_sanitize_diagnostic_redacts_secrets_and_credentials_escapes_controls_and_bounds() -> None:
    # secrets, control characters, and output bounds
    diagnostic = sanitize_diagnostic(
        "failure\x00 api_key=sk-proj-abcdefghijklmnop\n" + "x" * 1000, max_bytes=96
    )
    assert "sk-proj" not in diagnostic
    assert len(diagnostic.encode("utf-8")) <= 99

    # control characters are escaped even without a secret
    assert sanitize_diagnostic("failure\x00") == "failure\\x00"

    # transport credentials are redacted
    for secret in (
        "Authorization: Bearer abc123",
        "Authorization: Basic YTo=",
        "https://runtime-user:runtime-password@example.invalid/image",
        "https://opaque-token@example.invalid/image",
        "https://:opaque-token@example.invalid/image",
        "https://opaque-token:@example.invalid/image",
    ):
        diagnostic = sanitize_diagnostic(f"runtime launch failed: {secret}")
        assert secret not in diagnostic, secret
        assert diagnostic.startswith("<redacted-sensitive-text:"), secret


def test_parse_smoke_markers_preserves_bounded_typed_diagnostic() -> None:
    for name, raw, expected in [
        ("bounded", b'"expected [11]; observed [11, 11]"', "expected [11]; observed [11, 11]"),
        ("oversized", b'"' + b"x" * 4097 + b'"', None),
    ]:
        category, diagnostic = parse_smoke_markers(
            [b"AGENTIC_SMOKE_CATEGORY=candidate_reset_sequence\n",
             b"AGENTIC_SMOKE_DIAGNOSTIC=" + raw + b"\n"]
        )
        assert category == "candidate_reset_sequence", name
        assert diagnostic == expected, name


# ---------------------------------------------------------------------------
# Gate server protocol
# ---------------------------------------------------------------------------


def test_gate_server_authenticates_allowlists_and_budgets_requests(tmp_path: Path) -> None:
    # authenticated allowlisted request is executed exactly once
    seen: list[str] = []

    def execute(operation: str) -> TaskGateReport:
        seen.append(operation)
        return _passed(operation, "b")

    with _serving(tmp_path / "gate.sock", execute, max_requests=2):
        report = request_gate(tmp_path / "gate.sock", token=TOKEN, operation="contract")
    assert report.status == "pass"
    assert seen == ["contract"]

    # non-allowlisted operations are rejected
    with _serving(tmp_path / "reject.sock"):
        for operation in ["../final", "shell", "", "FINAL-SMOKE"]:
            with pytest.raises(GateRequestError, match="operation"):
                request_gate(tmp_path / "reject.sock", token=TOKEN, operation=operation)

    # a wrong token does not consume the request budget
    called = False

    def execute_once(operation: str) -> TaskGateReport:
        nonlocal called
        called = True
        return _passed(operation, "d")

    with _serving(tmp_path / "token.sock", execute_once):
        with pytest.raises(GateRequestError, match="authentication"):
            request_gate(tmp_path / "token.sock", token="wrong-token", operation="contract")
        report = request_gate(tmp_path / "token.sock", token=TOKEN, operation="contract")
    assert called is True
    assert report.status == "pass"

    # the server stays available after the request budget is exhausted
    with _serving(tmp_path / "budget.sock"):
        assert request_gate(tmp_path / "budget.sock", token=TOKEN, operation="contract").status == "pass"
        with pytest.raises(GateRequestError, match="request_budget_exhausted"):
            request_gate(tmp_path / "budget.sock", token=TOKEN, operation="contract")


def test_gate_server_survives_disconnects_rejects_oversized_requests_and_writes_receipts(
    tmp_path: Path,
) -> None:
    # a client that disconnects mid-execution does not break the server
    executor_started = threading.Event()
    release_executor = threading.Event()

    def execute(operation: str) -> TaskGateReport:
        executor_started.set()
        assert release_executor.wait(timeout=2)
        return _passed(operation, "d")

    with _serving(tmp_path / "gate.sock", execute, max_requests=2):
        client = _connect(tmp_path / "gate.sock")
        client.sendall(
            json.dumps({"schema_version": 1, "token": TOKEN, "operation": "contract"}).encode()
            + b"\n"
        )
        assert executor_started.wait(timeout=2)
        client.shutdown(socket.SHUT_RDWR)
        client.close()
        release_executor.set()
        report = request_gate(tmp_path / "gate.sock", token=TOKEN, operation="contract")
    assert report.status == "pass"

    # oversized requests are rejected by the protocol
    with _serving(tmp_path / "large.sock"):
        client = _connect(tmp_path / "large.sock")
        client.sendall(json.dumps({"padding": "x" * 70_000}).encode() + b"\n")
        response = client.makefile("rb").readline()
        client.close()
    assert json.loads(response)["error"] == "request_too_large"

    # bounded receipts are written without the token
    receipt_path = tmp_path / "summary.json"
    with _serving(tmp_path / "receipt.sock", max_requests=2, receipt_path=receipt_path):
        request_gate(tmp_path / "receipt.sock", token=TOKEN, operation="contract")

    text = receipt_path.read_text(encoding="utf-8")
    assert json.loads(text) == {
        "max_requests": 2,
        "receipts": [{
            "category": "gate_passed", "operation": "contract", "requests_remaining": 1,
            "scaffold_fingerprint": "sha256:" + "f" * 64, "status": "pass",
        }],
        "requests_used": 1,
        "schema_version": 1,
        "service_status": "closed",
    }
    assert TOKEN not in text
    assert receipt_path.stat().st_mode & 0o777 == 0o600


def test_standalone_client_needs_only_socket_and_token_and_rejects_invalid_timeout(
    tmp_path: Path,
) -> None:
    # socket and token are sufficient
    with _serving(tmp_path / "gate.sock"):
        result = _standalone_client(
            {"TASK_GATE_SOCKET": str(tmp_path / "gate.sock"), "TASK_GATE_TOKEN": TOKEN}
        )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "pass"

    # an invalid host timeout is rejected
    result = _standalone_client({
        "TASK_GATE_SOCKET": "/unused", "TASK_GATE_TOKEN": TOKEN,
        "TASK_GATE_TIMEOUT_SECONDS": "unbounded",
    })
    assert result.returncode == 2
    assert "must be an integer" in result.stderr


def test_runtime_recording_executor_marks_only_gate_created_runtime(tmp_path: Path) -> None:
    for preexisting in (False, True):
        root = tmp_path / f"preexisting-{preexisting}"
        task_dir = root / "task"
        task_dir.mkdir(parents=True)
        marker = root / "runtime-created"

        def create_runtime(task_dir: Path = task_dir) -> None:
            (task_dir / ".venv").mkdir()
            (task_dir / ".agentic_runtime.json").write_text("{}", encoding="utf-8")

        if preexisting:
            create_runtime()

        def execute(operation: str, preexisting: bool = preexisting,
                    create_runtime: Callable[[], None] = create_runtime) -> TaskGateReport:
            if not preexisting:
                create_runtime()
            return _passed(operation, "a")

        assert RuntimeRecordingExecutor(task_dir, marker, execute)("runtime").status == "pass"
        if preexisting:
            assert not marker.exists()
        else:
            assert marker.read_text(encoding="ascii") == "gate-created\n"


# ---------------------------------------------------------------------------
# Gate startup publishes readiness only after a safe listening endpoint exists
# ---------------------------------------------------------------------------


@pytest.fixture
def start_root() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="gate-start-") as directory:
        yield Path(directory)


class RecordingSocket:
    def __init__(self, server: GateServer, ready: Path, failure: str = "") -> None:
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server = server
        self.ready = ready
        self.failure = failure

    def _check(self, operation: str) -> None:
        assert not self.ready.exists()
        assert not self.server._ready.is_set()
        if operation == self.failure:
            raise OSError("private-token /private/socket")

    def bind(self, address: str) -> None:
        self._check("bind")
        self.socket.bind(address)

    def listen(self, backlog: int) -> None:
        self._check("listen")
        self.socket.listen(backlog)

    def settimeout(self, seconds: float) -> None:
        self._check("settimeout")
        self.socket.settimeout(seconds)

    def close(self) -> None:
        self.socket.close()


def _closed_server(root: Path) -> GateServer:
    server = GateServer(
        root / "gate.sock", token="private-token", executor=_forbidden_execute,
        max_requests=1, ready_path=root / "ready",
    )
    server.close()
    return server


def _raise_private(*args: object) -> NoReturn:
    raise OSError("private-token")


def test_startup_failure_closes_descriptor_and_removes_only_owned_paths(
    start_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for operation in (
        "bind", "chmod", "listen", "settimeout", "ready_chmod", "ready_write", "ready_short_write",
    ):
        root = start_root / operation
        root.mkdir()
        with monkeypatch.context() as patch:
            server = _closed_server(root)
            transport = RecordingSocket(server, root / "ready", operation)
            patch.setattr(gate.socket, "socket", lambda *args, transport=transport: transport)
            if operation == "chmod":
                patch.setattr(Path, "chmod", _raise_private)
            if operation == "ready_chmod":
                patch.setattr(gate.os, "fchmod", _raise_private)
            descriptors: list[int] = []
            if operation in {"ready_write", "ready_short_write"}:
                original_write = os.write

                def partial_write(descriptor: int, payload: bytes, operation: str = operation,
                                  descriptors: list[int] = descriptors) -> int:
                    descriptors.append(descriptor)
                    original_write(descriptor, payload[:1])
                    if operation == "ready_short_write":
                        return 1
                    raise OSError("private-token")

                patch.setattr(gate.os, "write", partial_write)
            with pytest.raises(GateRequestError) as failure:
                server.serve()
        assert "private-token" not in str(failure.value), operation
        assert "/private" not in str(failure.value), operation
        assert transport.socket.fileno() == -1, operation
        assert not server._ready.is_set(), operation
        assert not (root / "gate.sock").exists(), operation
        assert not (root / "ready").exists(), operation
        for descriptor in descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)


def test_ready_marker_is_private_written_after_listen_and_shutdown_preserves_replacements(
    start_root: Path, monkeypatch,
) -> None:
    # the ready marker is private and written only after listen
    server = _closed_server(start_root)
    transport = RecordingSocket(server, start_root / "ready")
    monkeypatch.setattr(gate.socket, "socket", lambda *args: transport)
    original_set = server._ready.set
    observed = []

    def mark_ready() -> None:
        observed.append((start_root / "ready").read_bytes())
        assert stat.S_IMODE((start_root / "ready").stat().st_mode) == 0o600
        assert stat.S_IMODE((start_root / "gate.sock").stat().st_mode) == 0o600
        original_set()

    monkeypatch.setattr(server._ready, "set", mark_ready)
    server.serve()
    assert observed == [b"ready\n"]
    assert transport.socket.fileno() == -1
    assert not (start_root / "ready").exists()
    assert not (start_root / "gate.sock").exists()
    monkeypatch.undo()

    # shutdown preserves a path replaced after readiness
    for name in ("gate.sock", "ready"):
        root = start_root / f"replaced-{name}"
        root.mkdir()
        server = _closed_server(root)
        path = root / name

        def replace_after_ready(path: Path = path, root: Path = root) -> None:
            path.rename(root / "original")
            path.write_text("replacement")

        with monkeypatch.context() as patch:
            patch.setattr(server._ready, "set", replace_after_ready)
            server.serve()
        assert path.read_text() == "replacement", name


def test_startup_preserves_preexisting_paths_and_counts_encoded_socket_bytes(
    start_root: Path,
) -> None:
    # pre-existing files and symlinks at the socket or ready path are preserved
    for name in ("gate.sock", "ready"):
        for kind in ("file", "symlink"):
            root = start_root / f"{name}-{kind}"
            root.mkdir()
            path = root / name
            if kind == "file":
                path.write_text("preexisting")
            else:
                path.symlink_to(root / "absent")
            identity = path.lstat()
            with pytest.raises(GateRequestError):
                _closed_server(root).serve()
            assert path.lstat().st_ino == identity.st_ino, (name, kind)
            if kind == "file":
                assert path.read_text() == "preexisting", name
            else:
                assert path.is_symlink(), name

    # the socket path limit counts encoded bytes
    for encoded_length in (107, 108):
        remaining = encoded_length - len(os.fsencode(start_root)) - 1
        path = start_root / ("é" * (remaining // 2) + "x" * (remaining % 2))
        assert len(os.fsencode(path)) == encoded_length
        server = GateServer(path, token="token", executor=_forbidden_execute, max_requests=1)
        server.close()
        if encoded_length == 107:
            server.serve()
            assert not path.exists()
        else:
            with pytest.raises(GateRequestError, match="107"):
                server.serve()
            assert server._server is None


def test_service_forwards_optional_ready_file(start_root: Path, monkeypatch) -> None:
    from src.task_drafting import service

    (start_root / "token").write_text("fixture-token")
    for include_ready_file in (False, True):
        arguments = [
            "service", "--socket", str(start_root / "gate.sock"),
            "--token-file", str(start_root / "token"),
            "--task-dir", str(start_root), "--source-task-dir", str(start_root),
            "--container-image", "fixture", "--container-network", "none",
            "--timeout-seconds", "10", "--production-entry-timeout-seconds", "10",
            "--max-runtime-bytes", "1000", "--max-requests", "1",
            "--receipt-path", str(start_root / "receipt"),
            "--runtime-created-marker", str(start_root / "runtime"),
        ]
        if include_ready_file:
            arguments += ["--ready-file", str(start_root / "ready")]
        observed = []

        class Server:
            def __init__(self, *args, **kwargs):
                observed.append(kwargs["ready_path"])

            def serve(self):
                pass

        with monkeypatch.context() as patch:
            patch.setattr("sys.argv", arguments)
            patch.setattr(service, "GateServer", Server)
            patch.setattr(service.signal, "signal", lambda *args: None)
            assert service._main() == 0, include_ready_file
        assert observed == [start_root / "ready" if include_ready_file else None], include_ready_file


# ---------------------------------------------------------------------------
# Private short IPC socket directories (socket_runtime)
# ---------------------------------------------------------------------------


def _make_root(tmp_path_factory, monkeypatch) -> Path:
    """Point SOCKET_RUNTIME_ROOT at a fresh, not-yet-created private root."""
    path = tmp_path_factory.mktemp("ipc") / "gates"
    monkeypatch.setattr(socket_runtime, "SOCKET_RUNTIME_ROOT", path)
    return path


@pytest.fixture
def ipc_root(tmp_path_factory, monkeypatch):
    return _make_root(tmp_path_factory, monkeypatch)


def _bind_socket(path: Path) -> os.stat_result:
    """Bind and close an AF_UNIX socket, leaving its filesystem entry behind."""
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(path))
    return path.stat()


def _leaf_with_socket() -> Path:
    path = socket_runtime.create_socket_directory()
    _bind_socket(path / "s")
    return path


def _patch_foreign_owner(monkeypatch, attribute: str, matches) -> None:
    original = getattr(os, attribute)

    def foreign(target, *args, **kwargs):
        metadata = original(target, *args, **kwargs)
        if matches(target, metadata):
            return SimpleNamespace(st_mode=metadata.st_mode, st_uid=os.getuid() + 1)
        return metadata

    monkeypatch.setattr(os, attribute, foreign)


def test_concurrent_creation_is_unique_private_short_and_ignores_long_temp_roots(
    ipc_root, monkeypatch,
):
    # concurrent creation yields unique private short leaves
    with ThreadPoolExecutor(max_workers=4) as workers:
        paths = list(workers.map(lambda _: socket_runtime.create_socket_directory(), range(12)))
    assert len(set(paths)) == 12
    assert stat.S_IMODE(ipc_root.stat().st_mode) == 0o700
    for path in paths:
        assert path.parent == ipc_root
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
        assert len(os.fsencode(path / "s")) <= 107
        socket_runtime.remove_socket_directory(path)
    assert ipc_root.is_dir()

    # long environment temporary roots are ignored
    for name in ("TMPDIR", "AGENTIC_TEMP_ROOT", "AGENTIC_TEMP_DIR"):
        monkeypatch.setenv(name, "/" + "long" * 100)
    path = socket_runtime.create_socket_directory()
    assert path.parent == ipc_root
    socket_runtime.remove_socket_directory(path)


def test_creation_rejects_unsafe_roots_and_oversized_paths(tmp_path_factory, monkeypatch):
    # an existing root must be private
    for mode in (0o755, 0o777, 0o1777, 0o750):
        root = _make_root(tmp_path_factory, monkeypatch)
        root.mkdir(mode=mode)
        root.chmod(mode)
        with pytest.raises((ValueError, OSError)):
            socket_runtime.create_socket_directory()

    # a symlinked root is rejected
    root = _make_root(tmp_path_factory, monkeypatch)
    target = root.parent / "target"
    target.mkdir(mode=0o700)
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        socket_runtime.create_socket_directory()
    assert list(target.iterdir()) == []

    # a symlinked ancestor is rejected
    root = _make_root(tmp_path_factory, monkeypatch)
    link = root.parent / "alias"
    link.symlink_to(root.parent, target_is_directory=True)
    monkeypatch.setattr(socket_runtime, "SOCKET_RUNTIME_ROOT", link / "gates")
    with pytest.raises((ValueError, OSError)):
        socket_runtime.create_socket_directory()

    # a foreign-owned root is rejected
    root = _make_root(tmp_path_factory, monkeypatch)
    root.mkdir(mode=0o700)
    root_inode = root.stat().st_ino
    with monkeypatch.context() as patch:
        _patch_foreign_owner(patch, "fstat", lambda _, meta: meta.st_ino == root_inode)
        with pytest.raises(ValueError):
            socket_runtime.create_socket_directory()

    # an unsafe writable ancestor is rejected
    root = _make_root(tmp_path_factory, monkeypatch)
    root.parent.chmod(0o777)
    with pytest.raises(ValueError):
        socket_runtime.create_socket_directory()
    assert not root.exists()

    # the encoded socket path limit is checked before creation
    root = _make_root(tmp_path_factory, monkeypatch)
    oversized = root.parent / ("é" * 45)
    monkeypatch.setattr(socket_runtime, "SOCKET_RUNTIME_ROOT", oversized)
    with pytest.raises(ValueError, match="107"):
        socket_runtime.create_socket_directory()
    assert not oversized.exists()


def test_cleanup_removes_captured_socket_idempotently_and_preserves_residuals(
    ipc_root, tmp_path_factory, monkeypatch,
):
    # cleanup with a captured identity removes only the socket and leaf, idempotently
    path = _leaf_with_socket()
    metadata = (path / "s").stat()
    socket_runtime.remove_socket_directory(path, socket_identity=(metadata.st_dev, metadata.st_ino))
    socket_runtime.remove_socket_directory(path)
    assert not path.exists()
    assert ipc_root.is_dir()

    # cleanup without an identity preserves a residual socket
    path = _leaf_with_socket()
    with pytest.raises(ValueError):
        socket_runtime.remove_socket_directory(path)
    assert (path / "s").is_socket()

    # cleanup preserves a socket replaced after capture
    path = socket_runtime.create_socket_directory()
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(path / "s"))
        original = (path / "s").stat()
        (path / "s").unlink()
        with socket.socket(socket.AF_UNIX) as replacement:
            replacement.bind(str(path / "s"))
            with pytest.raises(ValueError):
                socket_runtime.remove_socket_directory(
                    path, socket_identity=(original.st_dev, original.st_ino),
                )
            assert (path / "s").is_socket()

    # cleanup of a missing root is idempotent
    fresh_root = _make_root(tmp_path_factory, monkeypatch)
    socket_runtime.remove_socket_directory(fresh_root / ("gate-" + "a" * 16))
    assert not fresh_root.exists()


def test_cleanup_refuses_uncaptured_paths_unknown_files_non_sockets_and_symlinks(ipc_root):
    # uncaptured paths are rejected
    for relative in ("", "other", "gate-invalid", "gate-" + "a" * 16 + "/child"):
        with pytest.raises(ValueError):
            socket_runtime.remove_socket_directory(ipc_root / relative)

    # unknown files are refused without unlinking the socket
    path = _leaf_with_socket()
    (path / "unexpected").write_text("keep")
    with pytest.raises(ValueError):
        socket_runtime.remove_socket_directory(path)
    assert (path / "s").is_socket()
    assert (path / "unexpected").read_text() == "keep"

    # a non-socket `s` entry is refused
    for kind in ("file", "symlink", "directory"):
        path = socket_runtime.create_socket_directory()
        target = path / "s"
        if kind == "file":
            target.write_text("keep")
        elif kind == "directory":
            target.mkdir()
        else:
            target.symlink_to(ipc_root)
        with pytest.raises(ValueError):
            socket_runtime.remove_socket_directory(path)
        assert target.exists(), kind

    # a symlinked leaf is refused
    path = socket_runtime.create_socket_directory()
    link = ipc_root / ("gate-" + "f" * 16)
    link.symlink_to(path, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        socket_runtime.remove_socket_directory(link)
    assert path.is_dir()


def test_cleanup_revalidates_permissions_and_ownership(ipc_root, monkeypatch):
    # a foreign-owned leaf is refused
    path = socket_runtime.create_socket_directory()
    leaf_inode = path.stat().st_ino
    with monkeypatch.context() as patch:
        _patch_foreign_owner(patch, "fstat", lambda _, meta: meta.st_ino == leaf_inode)
        with pytest.raises(ValueError):
            socket_runtime.remove_socket_directory(path)
    assert path.exists()
    socket_runtime.remove_socket_directory(path)

    # a foreign-owned socket is refused
    path = _leaf_with_socket()
    with monkeypatch.context() as patch:
        _patch_foreign_owner(patch, "stat", lambda name, _: name == "s")
        with pytest.raises(ValueError):
            socket_runtime.remove_socket_directory(path)
    assert (path / "s").is_socket()

    # non-private leaf and root permissions are revalidated (root last: it poisons the fixture)
    for nonprivate in ("leaf", "root"):
        path = socket_runtime.create_socket_directory()
        (path if nonprivate == "leaf" else ipc_root).chmod(0o755)
        with pytest.raises(ValueError):
            socket_runtime.remove_socket_directory(path)
        assert path.exists(), nonprivate


def test_identify_refuses_unsafe_or_missing_endpoint(ipc_root):
    for kind in ("file", "symlink", "nonprivate", "missing"):
        path = socket_runtime.create_socket_directory()
        if kind == "file":
            (path / "s").write_text("not a socket")
        elif kind == "symlink":
            (path / "s").symlink_to(ipc_root)
        elif kind == "nonprivate":
            _bind_socket(path / "s")
            path.chmod(0o755)
        with pytest.raises((ValueError, OSError)):
            socket_runtime.identify_socket(path)


def test_cli_create_identify_and_cleanup_flows(ipc_root, capsys):
    # identify then cleanup a residual socket
    path = _leaf_with_socket()
    metadata = (path / "s").stat()
    assert socket_runtime.main(["identify", "--directory", str(path)]) == 0
    identity = capsys.readouterr().out.strip()
    assert identity == f"{metadata.st_dev}:{metadata.st_ino}"
    assert socket_runtime.main([
        "cleanup", "--directory", str(path), "--socket-identity", identity,
    ]) == 0
    assert not path.exists()

    # malformed socket identities are rejected by the CLI
    path = socket_runtime.create_socket_directory()
    for identity in (
        "", "1", "1:2:3", "-1:2", "+1:2", "1: 2", "1:2\n", "x:2", "١:2", f"{2**64}:2",
    ):
        with pytest.raises(SystemExit) as failure:
            socket_runtime.main(["cleanup", "--directory", str(path), "--socket-identity", identity])
        assert failure.value.code == 2, identity
        assert path.is_dir(), identity
    capsys.readouterr()

    # identify of an empty leaf fails without output
    path = socket_runtime.create_socket_directory()
    assert socket_runtime.main(["identify", "--directory", str(path)]) == 1
    assert not capsys.readouterr().out

    # cleanup failure is non-zero without success output
    assert socket_runtime.main(["cleanup", "--directory", str(ipc_root)]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "invalid task gate socket directory" in output.err

    # create and cleanup
    assert socket_runtime.main(["create"]) == 0
    path = Path(capsys.readouterr().out.strip())
    assert path.parent == ipc_root
    assert socket_runtime.main(["cleanup", "--directory", str(path)]) == 0
    assert capsys.readouterr().out == ""
    assert not path.exists()


# ---------------------------------------------------------------------------
# task_gate.sh cleanup preserves replacement endpoints (stop and startup abort)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _socket_directory() -> Iterator[Path]:
    path = socket_runtime.create_socket_directory()
    try:
        yield path
    finally:
        # These paths are deliberately retained by fail-closed cleanup.
        for name in ("s", "original"):
            (path / name).unlink(missing_ok=True)
        if path.exists():
            path.rmdir()


def _cleanup(directory: Path, operation: str, identity: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", '''set -eu
source "$REPO_ROOT/src/scripts/lib/task_gate.sh"
if "$CLEANUP_OPERATION"; then
  result=0
else
  result=$?
fi
printf 'result=%s\ndirectory=%s\nidentity=%s\n' \
  "$result" "${GATE_SOCKET_DIRECTORY:-}" "${GATE_SOCKET_IDENTITY:-}"
test -z "${TASK_GATE_TOKEN:-}"
test -z "${TASK_GATE_SOCKET:-}"
test -z "${TASK_GATE_CLIENT:-}"
test -z "${TASK_GATE_TIMEOUT_SECONDS:-}"
test -z "${TASK_GATE_EXTRA_MOUNTS:-}"
exit "$result"
'''],
        cwd=ROOT,
        env=os.environ | {
            "REPO_ROOT": str(ROOT), "PYTHON_BIN": sys.executable,
            "CLEANUP_OPERATION": operation, "GATE_SOCKET_DIRECTORY": str(directory),
            "GATE_SOCKET_IDENTITY": identity, "GATE_SERVICE_PID": "",
            "TASK_GATE_TOKEN": "fixture-token", "TASK_GATE_SOCKET": str(directory / "s"),
            "TASK_GATE_CLIENT": "fixture-client", "TASK_GATE_TIMEOUT_SECONDS": "10",
            "TASK_GATE_EXTRA_MOUNTS": "fixture-mounts",
        },
        capture_output=True, text=True, timeout=10,
    )


def test_gate_cleanup_removes_only_captured_socket_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a normal stop preserves a socket replacement retained by the server
    with _socket_directory() as directory:
        path = directory / "s"
        server = GateServer(path, token="fixture", executor=_forbidden_execute, max_requests=1)
        server.close()
        captured = []
        with socket.socket(socket.AF_UNIX) as replacement:
            def replace_after_listen() -> None:
                metadata = path.stat()
                captured.append(f"{metadata.st_dev}:{metadata.st_ino}")
                path.rename(directory / "original")
                replacement.bind(str(path))
                (directory / "original").unlink()

            monkeypatch.setattr(server._ready, "set", replace_after_listen)
            server.serve()
            replacement_identity = path.stat().st_ino
            result = _cleanup(directory, "stop_task_repair_gate", captured[0])
            assert result.returncode == 4, result.stdout + result.stderr
            assert path.stat().st_ino == replacement_identity
            assert f"directory={directory}\n" in result.stdout

    # a startup abort removes only the captured socket identity
    for replace_socket in (False, True):
        with _socket_directory() as directory:
            path = directory / "s"
            with socket.socket(socket.AF_UNIX) as original, \
                    socket.socket(socket.AF_UNIX) as replacement:
                original.bind(str(path))
                metadata = path.stat()
                identity = f"{metadata.st_dev}:{metadata.st_ino}"
                if replace_socket:
                    path.rename(directory / "original")
                    replacement.bind(str(path))
                    (directory / "original").unlink()
                surviving_inode = path.stat().st_ino
                original.close()
                result = _cleanup(directory, "abort_task_repair_gate_startup", identity)
                if replace_socket:
                    assert result.returncode == 4, result.stdout + result.stderr
                    assert path.stat().st_ino == surviving_inode
                    assert f"directory={directory}\n" in result.stdout
                    assert f"identity={identity}\n" in result.stdout
                else:
                    assert result.returncode == 0, result.stdout + result.stderr
                    assert not directory.exists()
                    assert "directory=\nidentity=\n" in result.stdout


# ---------------------------------------------------------------------------
# Production repair launcher (task_gate.sh) without Docker or live agents
# ---------------------------------------------------------------------------

LABELS = ("preflight_repair_0", "smoke_repair_0", "repair_0", "qualification_repair_0")
GATE_ENV = ("TASK_GATE_TOKEN", "TASK_GATE_SOCKET", "TASK_GATE_CLIENT",
            "TASK_GATE_TIMEOUT_SECONDS")

SERVICE = '''import json, os, signal, socket, sys, threading
from pathlib import Path
if sys.argv[1:3] != ["-m", "src.task_drafting.service"]:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
sys.path.insert(0, os.environ["REPO_ROOT"])
from src.task_drafting.gate import GateServer, TaskGateReport
args = sys.argv[1:]
def argument(flag):
    return Path(args[args.index(flag) + 1])
Path(os.environ["ARGUMENT_TRACE"]).write_text(json.dumps({
    "arguments": args, "pid": os.getpid(), "socket": str(argument("--socket"))}))
mode = os.environ["SERVICE_MODE"]
print("fixture service " + mode, flush=True)
if mode == "crash":
    raise SystemExit(17)
if mode == "unready":
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(argument("--socket")))
    signal.pause()
    raise SystemExit(0)
server = GateServer(argument("--socket"),
    token=argument("--token-file").read_text(),
    executor=lambda op: TaskGateReport.passed(op, "sha256:" + "b" * 64, 0.1),
    max_requests=3, receipt_path=argument("--receipt-path"))
thread = threading.Thread(target=server.serve, daemon=True)
signal.signal(signal.SIGTERM, lambda *_: server.close())
thread.start()
server.wait_until_ready()
if "--ready-file" in args:
    argument("--ready-file").write_text("ready\\n")
thread.join()
'''

SNAPSHOT = '''
"$REAL_PYTHON" - "$SNAPSHOT" <<'PY'
import json, os, sys
keys = ["TASK_GATE_TOKEN", "TASK_GATE_SOCKET", "TASK_GATE_CLIENT",
        "TASK_GATE_TIMEOUT_SECONDS"]
with open(sys.argv[1], "w") as stream:
    json.dump({key: os.environ.get(key, "") for key in keys}, stream)
PY
printf '%s' "${GATE_SERVICE_PID:-}" > "$PID_STATE"
printf '%s' "${TASK_GATE_EXTRA_MOUNTS:-}" > "$MOUNT_STATE"
printf '%s' "${GATE_SOCKET_DIRECTORY:-}" > "$SOCKET_STATE"
'''


def _launcher_environment(tmp_path: Path, *, mode: str, containerize: int) -> dict[str, str]:
    root = tmp_path / ("validation-" + "v" * 110)
    root.mkdir()
    shim = tmp_path / "python"
    shim.write_text(f"#!{sys.executable}\n" + SERVICE)
    shim.chmod(0o700)
    return os.environ | {
        "PYTHON_BIN": str(shim), "REAL_PYTHON": sys.executable, "REPO_ROOT": str(ROOT),
        "VALIDATION_ROOT": str(root), "STAGED_TASK_DIR": str(tmp_path / "task"),
        "TASK_SOURCE_DIR": str(tmp_path / "source"), "CONTAINER_IMAGE": "fixture",
        "CONTAINER_NETWORK": "none", "EVALUATOR_SMOKE_TIMEOUT_SECONDS": "10",
        "PRODUCTION_ENTRY_TIMEOUT_SECONDS": "10", "TASK_DRAFT_MAX_RUNTIME_BYTES": "1000",
        "MAX_GATE_CHECKS_PER_REPAIR": "3", "RUNTIME_INSTALL_CACHE_ROOT": str(tmp_path / "cache"),
        "CONTAINERIZE": str(containerize), "ARGUMENT_TRACE": str(tmp_path / "arguments.json"),
        "SERVICE_MODE": mode, "SNAPSHOT": str(tmp_path / "after.json"),
        "PID_STATE": str(tmp_path / "pid.txt"), "MOUNT_STATE": str(tmp_path / "mounts.txt"),
        "SOCKET_STATE": str(tmp_path / "socket-directory.txt"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _run_launcher(env: dict[str, str], body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", 'set -eu\nsource "$REPO_ROOT/src/scripts/lib/task_gate.sh"\n'
         'trap stop_task_repair_gate EXIT\n' + body],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=15,
    )


def _assert_released(env: dict[str, str]) -> dict[str, object]:
    trace = json.loads(Path(env["ARGUMENT_TRACE"]).read_text())
    assert not Path(trace["socket"]).exists()
    assert not Path(trace["socket"]).parent.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(trace["pid"], 0)
    assert json.loads(Path(env["SNAPSHOT"]).read_text()) == dict.fromkeys(GATE_ENV, "")
    assert Path(env["PID_STATE"]).read_text() == ""
    assert Path(env["MOUNT_STATE"]).read_text() == ""
    assert Path(env["SOCKET_STATE"]).read_text() == ""
    return trace


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("containerize", [0, 1])
def test_long_validation_root_supports_authenticated_gate_and_cleanup(
    tmp_path: Path, label: str, containerize: int,
) -> None:
    env = _launcher_environment(tmp_path, mode="ready", containerize=containerize)
    env["GATE_LABEL"] = label
    result = _run_launcher(env, '''
start_task_repair_gate "$GATE_LABEL"
"$REAL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from src.task_drafting.gate import GateRequestError, request_gate
trace = json.loads(Path(os.environ["ARGUMENT_TRACE"]).read_text())
path = Path(trace["socket"])
assert len(os.fsencode(path)) < 108
assert path.parent.parent == Path("/dev/shm") / f"agentic-task-gates-{os.getuid()}"
assert path.parent.stat().st_mode & 0o777 == 0o700
assert path.stat().st_mode & 0o777 == 0o600
assert request_gate(path, token=os.environ["TASK_GATE_TOKEN"], operation="contract").status == "pass"
try:
    request_gate(path, token="wrong-token", operation="contract")
except GateRequestError:
    pass
else:
    raise AssertionError("unauthenticated gate request was accepted")
expected = "/run/agentic-task-gate.sock" if os.environ["CONTAINERIZE"] == "1" else str(path)
assert os.environ["TASK_GATE_SOCKET"] == expected
expected_client = "/tmp/agentic-task-draft-check.py" if os.environ["CONTAINERIZE"] == "1" else str(
    Path(os.environ["REPO_ROOT"]) / "src/task_drafting/standalone_client.py")
assert os.environ["TASK_GATE_CLIENT"] == expected_client
PY
printf '%s' "$TASK_GATE_EXTRA_MOUNTS" > "$MOUNT_STATE.before"
stop_task_repair_gate
''' + SNAPSHOT)
    assert result.returncode == 0, result.stderr
    trace = _assert_released(env)
    mounts = Path(env["MOUNT_STATE"] + ".before").read_text().splitlines()
    if containerize:
        assert mounts == [
            f"{ROOT}/src/task_drafting/standalone_client.py:/tmp/agentic-task-draft-check.py:ro",
            f"{trace['socket']}:/run/agentic-task-gate.sock",
        ]
    else:
        assert mounts == []
    audit = Path(env["VALIDATION_ROOT"]) / f"gate_{label}"
    assert "fixture service ready" in (audit / "service.log").read_text()
    assert (audit / "summary.json").is_file()


@pytest.mark.parametrize("mode", ["crash", "unready"])
def test_failed_gate_startup_releases_process_socket_and_credentials(
    tmp_path: Path, mode: str,
) -> None:
    env = _launcher_environment(tmp_path, mode=mode, containerize=1)
    result = _run_launcher(env, '''
if start_task_repair_gate preflight_repair_0; then
  echo "failed service was reported ready" >&2
  exit 99
else
  [ "$?" -eq 4 ]
fi
''' + SNAPSHOT)
    assert result.returncode == 0, result.stderr
    _assert_released(env)
    audit = Path(env["VALIDATION_ROOT"]) / "gate_preflight_repair_0"
    assert f"fixture service {mode}" in (audit / "service.log").read_text()
    assert "Task repair gate" in result.stderr


# ---------------------------------------------------------------------------
# Isolated RPC transport policy
# ---------------------------------------------------------------------------

_PROGRAM = "program_0/src/program.py"
_HANDLER = "def handle(request):\n    return request\n"


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_isolated_rpc_transport_findings_cover_exchange_names_benign_sources_and_symlinks(
    tmp_path: Path,
) -> None:
    # reserved exchange names are rejected
    reserved = tmp_path / "reserved"
    _write_tree(reserved, {
        "evaluate.py": "import os\nexchange = os.environ.get('EXCHANGE_DIR')\n",
        "evaluate_final.py": "VALUE = 1\n",
        _PROGRAM: "import os\n\nroot = os.getenv('AGENTIC_CANDIDATE_EXCHANGE')\n",
    })
    findings = isolated_rpc_transport_findings(reserved)
    assert len(findings) == 2
    assert "evaluate.py:2" in findings[0]
    assert "`EXCHANGE_DIR`" in findings[0]
    assert "program_0/src/program.py:3" in findings[1]
    assert "`AGENTIC_CANDIDATE_EXCHANGE`" in findings[1]
    assert all("finite inline JSON" in finding for finding in findings)

    # benign sources are allowed
    for name, files in [
        ("inline_and_nonfilesystem_handles", {
            "evaluate.py": "request = {'array_base64': 'AA==', 'model_handle': 'candidate-state-1'}\n",
            "evaluate_final.py": "MY_EXCHANGE_DIRECTORY = 'candidate-owned'\n",
            _PROGRAM: _HANDLER,
        }),
        ("inert_comments_and_docstrings", {
            "evaluate.py": '"""EXCHANGE_DIR is unsupported."""\n# AGENTIC_CANDIDATE_EXCHANGE\nVALUE = 1\n',
        }),
    ]:
        benign = tmp_path / name
        _write_tree(benign, files)
        assert isolated_rpc_transport_findings(benign) == (), name

    # findings are bounded
    bounded = tmp_path / "bounded"
    _write_tree(bounded, {
        "evaluate.py": "\n".join(f"value_{index} = 'EXCHANGE_DIR'" for index in range(100)) + "\n",
        "evaluate_final.py": "VALUE = 1\n",
        _PROGRAM: _HANDLER,
    })
    findings = isolated_rpc_transport_findings(bounded)
    assert len(findings) == 64
    assert "evaluate.py:1" in findings[0]
    assert "evaluate.py:64" in findings[-1]

    # symlinked sources are rejected
    linked = tmp_path / "linked"
    linked.mkdir()
    target = linked / "outside.py"
    target.write_text("EXCHANGE_DIR = '/outside'\n", encoding="utf-8")
    (linked / "evaluate.py").symlink_to(target)
    assert isolated_rpc_transport_findings(linked) == (
        "evaluate.py: isolated RPC source must be a regular non-symlink file",
    )


def test_isolated_rpc_transport_bounds_source_tree_and_fails_closed_on_parser_recursion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # source tree limits
    for name, attribute, limit, files, expected in [
        (
            "source_files", "_MAX_SOURCE_FILES", 2,
            {"evaluate.py": "VALUE = 1\n", "evaluate_final.py": "VALUE = 2\n", _PROGRAM: "VALUE = 3\n"},
            "isolated RPC source contains more than 2 Python files",
        ),
        (
            "aggregate_bytes", "_MAX_SOURCE_BYTES", 4, {"evaluate.py": "12345"},
            "isolated RPC source exceeds the 4 byte validation limit",
        ),
        (
            "per_file_bytes", "_MAX_SOURCE_FILE_BYTES", 4, {"evaluate.py": "12345"},
            "evaluate.py: isolated RPC source exceeds the 4 byte per-file validation limit",
        ),
        (
            "tree_entries", "_MAX_TREE_ENTRIES", 2,
            {f"program_0/artifact-{index}.txt": "x" for index in range(3)},
            "isolated RPC source tree contains more than 2 entries",
        ),
    ]:
        root = tmp_path / name
        _write_tree(root, files)
        with monkeypatch.context() as patch:
            patch.setattr(rpc_transport_policy, attribute, limit)
            assert isolated_rpc_transport_findings(root) == (expected,), name

    # parser recursion fails closed
    root = tmp_path / "recursion"
    root.mkdir()
    (root / "evaluate.py").write_text("VALUE = 1\n", encoding="utf-8")

    def recurse(*args: object, **kwargs: object) -> object:
        raise RecursionError("pathological syntax")

    monkeypatch.setattr(rpc_transport_policy, "_parse_source", recurse)
    assert isolated_rpc_transport_findings(root) == (
        "evaluate.py: isolated RPC source could not be parsed safely",
    )
