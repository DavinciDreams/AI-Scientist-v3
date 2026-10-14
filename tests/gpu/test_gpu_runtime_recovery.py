"""Durable ownership journals, process identities, and ownership-bound GPU reclaim."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import errno
import fcntl
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from typing import Any
from unittest.mock import Mock

import pytest

from src.investigation import gpu_owner_process as owner, gpu_runtime_cleanup as cleanup
from src.investigation.bounded_process import BoundedProcessResult, ProcessTimeoutError
from src.investigation.gpu_owner_process import ProcessIdentityError, current_identity
from src.investigation.gpu_runtime_journal import (
    Journal, LaunchRecord, RuntimeJournalError, create_runtime, open_runtime,
)

LAUNCH = "b" * 32
CONTAINER = "c" * 64


def _root(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir(mode=0o700)
    return root


def _open(root: Path, token: str, seconds: float = 1.0):
    return open_runtime(root, token, deadline=time.monotonic() + seconds)


def _path(root: Path, token: str) -> Path:
    return root / f"runtime-{token}.json"


def _read(root: Path, token: str) -> dict:
    return json.loads(_path(root, token).read_text())


def _rewrite(root: Path, token: str, **fields: object) -> None:
    _path(root, token).write_text(json.dumps(_read(root, token) | fields))


@contextmanager
def _edit(root: Path, token: str):
    with _open(root, token) as journal:
        yield journal.record
        journal.save()


def _set_launch(root: Path, token: str, launch_id: str = LAUNCH, **fields: object) -> None:
    with _edit(root, token) as record:
        for name, value in fields.items():
            setattr(record.launches[launch_id], name, value)


def _refused(root: Path, token: str, name: str, match: str | None = None) -> None:
    with pytest.raises(RuntimeJournalError, match=match):
        with _open(root, token):
            pytest.fail(f"{name} accepted")


# --------------------------------------------------------------------------- journal


def test_runtime_round_trip_durable_revocation_and_short_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # round trip and durable revocation
    token = create_runtime(tmp_path, ("GPU-aaaa", "GPU-bbbb"))
    with _open(tmp_path, token) as journal:
        assert journal.record.owner == current_identity()
        assert journal.record.devices == ("GPU-aaaa", "GPU-bbbb")
        journal.record.launches["a" * 32] = LaunchRecord("owned-container", current_identity())
        journal.record.state = "revoked"
        journal.save()
    with _open(tmp_path, token) as journal:
        assert journal.record.state == "revoked"
        assert journal.record.launches["a" * 32].name == "owned-container"
    assert _path(tmp_path, token).stat().st_mode & 0o777 == 0o600
    # short writes complete the durable record
    root = _root(tmp_path, "short-writes")
    with monkeypatch.context() as m:
        real_write = os.pwrite
        m.setattr(os, "pwrite", lambda fd, data, offset: real_write(fd, data[:13], offset))
        token = create_runtime(root, ("GPU-aaaa",))
        with _open(root, token) as journal:
            assert journal.record.devices == ("GPU-aaaa",)


def test_invalid_identity_unsafe_copied_or_busy_runtimes_are_refused(tmp_path: Path) -> None:
    # invalid runtime identity
    for token in ["../escape", "A" * 32, "a" * 33, ""]:
        _refused(tmp_path, token, f"invalid token {token!r}")
    # unsafe runtime files
    for unsafe in ["symlink", "hardlink", "permissions", "oversized", "malformed"]:
        root = _root(tmp_path, unsafe)
        token = create_runtime(root, ("GPU-aaaa",))
        path = _path(root, token)
        if unsafe == "symlink":
            original = root / "original"
            path.rename(original)
            path.symlink_to(original)
        elif unsafe == "hardlink":
            os.link(path, root / "alias")
        elif unsafe == "permissions":
            path.chmod(0o644)
        elif unsafe == "oversized":
            with path.open("r+b") as stream:
                stream.truncate(4 * 1024 * 1024 + 1)
        else:
            path.write_text("{")
        _refused(root, token, f"unsafe journal ({unsafe})")
    # token binding refuses a copied record
    root = _root(tmp_path, "copied")
    first = create_runtime(root, ("GPU-aaaa",))
    second = create_runtime(root, ("GPU-bbbb",))
    _path(root, second).write_bytes(_path(root, first).read_bytes())
    _refused(root, second, "copied record")
    # busy journal respects the deadline
    root = _root(tmp_path, "busy")
    token = create_runtime(root, ("GPU-aaaa",))
    with _path(root, token).open("rb") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeJournalError, match="busy"):
            with _open(root, token, 0.03):
                pytest.fail("competing fence acquired")


def test_invalid_schema_fields_and_inconsistent_launch_receipts_fail_closed(tmp_path: Path) -> None:
    # invalid schema fields
    fields = [
        ("state", []), ("schema_version", True), ("devices", ["GPU-aaaa", "gpu-AAAA"]),
        ("launches", []), ("daemon_id", "invalid\nprivate"),
        ("owner", current_identity().fields() | {"uid": os.getuid() + 1}),
    ]
    for index, (field, value) in enumerate(fields):
        root = _root(tmp_path, f"field-{index}")
        token = create_runtime(root, ("GPU-aaaa",))
        _rewrite(root, token, **{field: value})
        _refused(root, token, f"invalid schema ({field})")
    # inconsistent launch receipts
    updates: list[dict[str, Any]] = [
        {"create_finished": True}, {"container_id": "c" * 64},
        {"starter": current_identity().fields()}, {"unexpected": True},
    ]
    for index, update in enumerate(updates):
        root = _root(tmp_path, f"launch-{index}")
        token = create_runtime(root, ("GPU-aaaa",))
        launch = LaunchRecord("owned", current_identity()).fields() | update
        _rewrite(root, token, launches={LAUNCH: launch})
        _refused(root, token, f"inconsistent launch ({update})")


def test_failed_or_replaced_persistence_cannot_report_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # save refuses a replaced file
    root = _root(tmp_path, "replaced-file")
    token = create_runtime(root, ("GPU-aaaa",))
    path = _path(root, token)
    with _open(root, token) as journal:
        path.rename(root / "old")
        path.write_text("replacement")
        path.chmod(0o600)
        journal.record.state = "revoked"
        with pytest.raises(RuntimeJournalError):
            journal.save()
    assert path.read_text() == "replacement"
    # save refuses a replaced root
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    token = create_runtime(root, ("GPU-aaaa",))
    with _open(root, token) as journal:
        root.rename(tmp_path / "previous-root")
        root.mkdir(mode=0o700)
        with pytest.raises((RuntimeJournalError, OSError)):
            journal.save()
    # failed fence persistence (pwrite / fsync / ftruncate)
    for operation in ["pwrite", "fsync", "ftruncate"]:
        root = _root(tmp_path, operation)
        token = create_runtime(root, ("GPU-aaaa",))
        with monkeypatch.context() as m, pytest.raises(RuntimeJournalError):
            with _open(root, token) as journal:
                def fail(*args: object) -> None:
                    raise OSError(errno.ENOSPC, "injected")
                m.setattr(os, operation, fail)
                journal.record.state = "revoked"
                journal.save()
    # zero write refuses persistence
    root = _root(tmp_path, "zero-write")
    token = create_runtime(root, ("GPU-aaaa",))
    with monkeypatch.context() as m, _open(root, token) as journal:
        m.setattr(os, "pwrite", lambda *args: 0)
        journal.record.state = "revoked"
        with pytest.raises(RuntimeJournalError, match="progress"):
            journal.save()


# --------------------------------------------------------------------------- owner process


def identity() -> owner.ProcessIdentity:
    return owner.ProcessIdentity(12345, 23456, "12345678-1234-1234-1234-123456789abc", os.getuid())


def _mock_process(monkeypatch: pytest.MonkeyPatch) -> owner.ProcessIdentity:
    """Mock kernel and /proc access so `identity()` looks like a live, signalable process."""
    expected = identity()
    monkeypatch.setattr(owner, "_boot_id", lambda: expected.boot_id)
    monkeypatch.setattr(owner, "_read_process", lambda pid: (expected.start_ticks, expected.uid, "S"))
    monkeypatch.setattr(owner, "_pidfd_open", Mock(return_value=97))
    monkeypatch.setattr(owner.os, "close", Mock())
    monkeypatch.setattr(owner, "_exited", Mock(return_value=False))
    monkeypatch.setattr(owner, "_pidfd_send_signal", Mock())
    return expected


def _terminate(process, seconds: float = 1.0) -> None:
    owner.terminate_process(process, deadline=time.monotonic() + seconds)


def _mock(target: object) -> Mock:
    """The ``Mock`` a test installed in place of ``target``."""
    assert isinstance(target, Mock)
    return target


def _sent_signals() -> list[int]:
    return [call.args[1] for call in _mock(owner._pidfd_send_signal).call_args_list]


def test_identity_schema_round_trips_and_rejects_malformed_or_incomplete_fields():
    # current identity round trips and is alive
    current = owner.current_identity()
    assert current.pid == os.getpid()
    assert current.uid == os.getuid()
    assert owner.ProcessIdentity.from_fields(current.fields()) == current
    assert owner.process_is_alive(current)
    # malformed identity fields
    for updates in [
        {"pid": True}, {"pid": 1}, {"pid": -2}, {"pid": 2**31},
        {"start_ticks": 0}, {"start_ticks": "10"}, {"start_ticks": 2**64},
        {"uid": False}, {"uid": -1}, {"boot_id": "garbage"}, {"unexpected": 1},
    ]:
        with pytest.raises(owner.ProcessIdentityError):
            owner.ProcessIdentity.from_fields(identity().fields() | updates)
    # incomplete identity
    for fields in [None, [], {}, {"pid": 100}]:
        with pytest.raises(owner.ProcessIdentityError):
            owner.ProcessIdentity.from_fields(fields)
    # malformed identity never reaches the kernel
    with pytest.raises(owner.ProcessIdentityError, match="identity"):
        _terminate(None)
    with pytest.raises(owner.ProcessIdentityError, match="identity"):
        owner.process_is_alive(None)


def test_liveness_opens_pidfd_first_and_never_signals_absent_reused_zombie_or_rebooted(monkeypatch):
    # pidfd is opened before /proc validation
    with monkeypatch.context() as m:
        mocked = _mock_process(m)
        events = []
        m.setattr(owner, "_pidfd_open", lambda pid: events.append("open") or 97)
        m.setattr(owner, "_read_process", lambda pid: events.append("read") or (
            mocked.start_ticks, mocked.uid, "S"
        ))
        assert owner.process_is_alive(mocked)
        assert events == ["open", "read"]
    # absent, reused, and zombie processes are never signaled
    for replacement in [None, (99999, os.getuid(), "S"), (23456, os.getuid(), "Z")]:
        with monkeypatch.context() as m:
            mocked = _mock_process(m)
            m.setattr(owner, "_read_process", lambda pid, r=replacement: r)
            assert not owner.process_is_alive(mocked), replacement
            _terminate(mocked)
            owner._pidfd_send_signal.assert_not_called()
    # an old boot is never signaled
    with monkeypatch.context() as m:
        mocked = _mock_process(m)
        m.setattr(owner, "_boot_id", lambda: "87654321-4321-4321-4321-abcdefabcdef")
        assert not owner.process_is_alive(mocked)
        _terminate(mocked)
        owner._pidfd_open.assert_not_called()


def test_pidfd_open_failures_distinguish_permission_from_absence(monkeypatch):
    # permission failure is not absence
    with monkeypatch.context() as m:
        mocked = _mock_process(m)
        m.setattr(owner, "_pidfd_open", Mock(side_effect=PermissionError()))
        with pytest.raises(owner.ProcessIdentityError, match="pidfd"):
            owner.process_is_alive(mocked)
    # missing process is absence
    with monkeypatch.context() as m:
        mocked = _mock_process(m)
        m.setattr(owner, "_pidfd_open", Mock(side_effect=ProcessLookupError()))
        assert not owner.process_is_alive(mocked)


def test_termination_rejects_foreign_uid_changed_owner_self_and_invalid_deadline(monkeypatch):
    # foreign uid is rejected before pidfd
    for operation in [owner.process_is_alive, owner.terminate_process]:
        with monkeypatch.context() as m:
            mocked = _mock_process(m)
            kwargs = {"deadline": time.monotonic() + 1} if operation is owner.terminate_process else {}
            with pytest.raises(owner.ProcessIdentityError, match="user"):
                operation(replace(mocked, uid=os.getuid() + 1), **kwargs)
            owner._pidfd_open.assert_not_called()
    # changed process owner fails closed
    with monkeypatch.context() as m:
        mocked = _mock_process(m)
        m.setattr(owner, "_read_process", lambda pid: (23456, os.getuid() + 1, "S"))
        with pytest.raises(owner.ProcessIdentityError, match="user"):
            _terminate(mocked)
        owner._pidfd_send_signal.assert_not_called()
    # self termination is rejected
    with monkeypatch.context() as m:
        mocked = _mock_process(m)
        with pytest.raises(owner.ProcessIdentityError, match="self"):
            _terminate(replace(mocked, pid=os.getpid()))
        owner._pidfd_open.assert_not_called()
    # invalid or elapsed deadline never signals
    for offset in [-1, float("nan"), float("inf")]:
        with monkeypatch.context() as m:
            mocked = _mock_process(m)
            with pytest.raises(owner.ProcessIdentityError, match="deadline"):
                _terminate(mocked, offset)
            owner._pidfd_send_signal.assert_not_called()


def test_termination_requires_confirmed_exit_and_signal_failure_preserves_uncertainty(monkeypatch):
    # termination requires confirmed exit
    with monkeypatch.context() as m:
        mocked = _mock_process(m)
        m.setattr(owner, "_wait_for_exit", Mock(return_value=False))
        with pytest.raises(owner.ProcessIdentityError, match="exit"):
            _terminate(mocked)
        assert _sent_signals() == [signal.SIGTERM, signal.SIGKILL]
        owner.os.close.assert_called_once_with(97)
    # signal failure preserves uncertainty
    with monkeypatch.context() as m:
        mocked = _mock_process(m)
        m.setattr(owner, "_pidfd_send_signal", Mock(side_effect=PermissionError()))
        with pytest.raises(owner.ProcessIdentityError, match="signal"):
            _terminate(mocked)
        owner.os.close.assert_called_once_with(97)


def test_exit_races_and_deadline_exhaustion_never_escalate(monkeypatch):
    # exit concurrent with validation never signals
    for states in [[True], [False, True]]:
        with monkeypatch.context() as m:
            mocked = _mock_process(m)
            m.setattr(owner, "_exited", Mock(side_effect=states))
            _terminate(mocked)
            owner._pidfd_send_signal.assert_not_called()
    # a signal race still requires pidfd exit
    with monkeypatch.context() as m:
        mocked = _mock_process(m)
        m.setattr(owner, "_pidfd_send_signal", Mock(side_effect=ProcessLookupError()))
        wait = Mock(return_value=True)
        m.setattr(owner, "_wait_for_exit", wait)
        _terminate(mocked)
        wait.assert_called_once()
    # deadline exhaustion never escalates (after SIGTERM / during validation)
    for name, clock, wait_returns, expected_signals in [
        ("after-term", [0, 0, 2], False, [signal.SIGTERM]),
        ("during-validation", [0, 2], None, []),
    ]:
        with monkeypatch.context() as m:
            mocked = _mock_process(m)
            m.setattr(owner.time, "monotonic", Mock(side_effect=clock))
            if wait_returns is not None:
                m.setattr(owner, "_wait_for_exit", Mock(return_value=wait_returns))
            with pytest.raises(owner.ProcessIdentityError, match="deadline"):
                owner.terminate_process(mocked, deadline=1)
            assert _sent_signals() == expected_signals, name


def test_terminates_only_disposable_child():
    for ignore_term in [False, True]:
        code = (
            "import json,signal,time; from src.investigation.gpu_owner_process import current_identity; "
            + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
            + "print(json.dumps(current_identity().fields()), flush=True); time.sleep(30)"
        )
        with subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True) as child:
            try:
                assert child.stdout is not None
                child_identity = owner.ProcessIdentity.from_fields(json.loads(child.stdout.readline()))
                assert owner.process_is_alive(child_identity)
                _terminate(child_identity, 3)
                assert not owner.process_is_alive(child_identity)
                child.wait(timeout=1)
                assert child.returncode == -(signal.SIGKILL if ignore_term else signal.SIGTERM)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=2)


def test_stat_parser_handles_parentheses_and_rejects_malformed_input():
    tail = ["S"] + ["0"] * 18 + ["123456"] + ["0"] * 5
    assert owner._parse_stat(b"4321 (worker ) name (part)) " + " ".join(tail).encode(), 4321) == (
        123456, "S"
    )
    for data in [b"", b"4321 bad", b"9999 (x) S", b"4321 (x) ???"]:
        with pytest.raises(owner.ProcessIdentityError, match="stat"):
            owner._parse_stat(data, 4321)


def test_proc_inspection_distinguishes_permission_absence_bounds_and_foreign_uid(monkeypatch):
    # permission failure is not absence
    with monkeypatch.context() as m:
        m.setattr(owner.os, "open", Mock(side_effect=PermissionError()))
        with pytest.raises(owner.ProcessIdentityError, match="inspect"):
            owner._read_process(12345)
    # disappearance is absence
    with monkeypatch.context() as m:
        m.setattr(owner.os, "open", Mock(side_effect=FileNotFoundError()))
        assert owner._read_process(12345) is None
    # the stat read is bounded
    with monkeypatch.context() as m:
        m.setattr(owner.os, "read", Mock(return_value=b"x" * 4097))
        with pytest.raises(owner.ProcessIdentityError, match="Oversized"):
            owner._read_process(os.getpid())
    # current identity requires the current uid
    with monkeypatch.context() as m:
        m.setattr(owner, "_read_process", lambda pid: (1, os.getuid() + 1, "S"))
        with pytest.raises(owner.ProcessIdentityError, match="current user"):
            owner.current_identity()


def test_boot_id_and_pidfd_poll_failures_are_uncertain(monkeypatch):
    # kernel boot id is validated or uncertain
    for opener in [
        lambda *a, **k: io.StringIO("invalid"), lambda *a, **k: io.StringIO("x" * 80),
        lambda *a, **k: io.StringIO(""), Mock(side_effect=PermissionError()),
    ]:
        with monkeypatch.context() as m:
            m.setattr(Path, "open", opener)
            with pytest.raises(owner.ProcessIdentityError, match="boot"):
                owner._boot_id()
    # poll error or syscall failure is not exit
    for outcome, match in [
        ({"return_value": [(97, select.POLLERR)]}, "observation"),
        ({"return_value": [(97, select.POLLNVAL)]}, "observation"),
        ({"side_effect": OSError()}, "exit"),
    ]:
        with monkeypatch.context() as m:
            poller = Mock()
            poller.poll.configure_mock(**outcome)
            m.setattr(owner.select, "poll", lambda: poller)
            with pytest.raises(owner.ProcessIdentityError, match=match):
                owner._exited(97)


def test_pidfd_bindings_prefer_native_and_fall_back_to_bounded_syscalls(monkeypatch):
    # native bindings are used when available
    with monkeypatch.context() as m:
        opener = Mock(return_value=97)
        sender = Mock()
        m.setattr(owner.os, "pidfd_open", opener, raising=False)
        m.setattr(owner.signal, "pidfd_send_signal", sender, raising=False)
        assert owner._pidfd_open(12345) == 97
        owner._pidfd_send_signal(97, signal.SIGTERM)
        opener.assert_called_once_with(12345, 0)
        sender.assert_called_once_with(97, signal.SIGTERM)
    # missing bindings use only pidfd syscalls
    with monkeypatch.context() as m:
        m.delattr(owner.os, "pidfd_open", raising=False)
        m.delattr(owner.signal, "pidfd_send_signal", raising=False)
        syscall = Mock(return_value=97)
        m.setattr(owner, "_pidfd_syscall", syscall)
        assert owner._pidfd_open(12345) == 97
        owner._pidfd_send_signal(97, signal.SIGTERM)
        assert [call.args for call in syscall.call_args_list] == [
            (434, 12345, 0), (424, 97, int(signal.SIGTERM), 0, 0),
        ]
    # the syscall fallback rejects unknown architectures
    with monkeypatch.context() as m:
        m.setattr(owner.os, "uname", lambda: Mock(machine="unknown"))
        with pytest.raises(owner.ProcessIdentityError, match="platform"):
            owner._pidfd_syscall(434, 12345, 0)
    # the syscall fallback preserves the kernel errno
    with monkeypatch.context() as m:
        m.setattr(owner.ctypes, "CDLL", lambda *args, **kwargs: Mock(syscall=Mock(return_value=-1)))
        m.setattr(owner.ctypes, "get_errno", lambda: errno.ESRCH)
        with pytest.raises(ProcessLookupError):
            owner._pidfd_syscall(434, 12345, 0)


# --------------------------------------------------------------------------- reclaim_runtime


def _container(token: str) -> dict:
    return {"id": CONTAINER, "labels": {
        "agentic-scientist.lease": token, "agentic-scientist.launch": LAUNCH,
    }}


def _make_runtime(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """Owned runtime with one fully started launch; process and Docker access are mocked."""
    token = create_runtime(root, ("GPU-aaaa", "GPU-bbbb"))
    me = current_identity()
    with _edit(root, token) as record:
        record.daemon_id = "daemon-1"
        record.launches[LAUNCH] = LaunchRecord(
            "owned-name", replace(me, pid=101), creator=replace(me, pid=102),
            starter=replace(me, pid=103), container_id=CONTAINER, create_finished=True,
            create_command_sha256="d" * 64, start_command_sha256="e" * 64,
        )
    monkeypatch.setattr(cleanup, "process_is_alive", Mock(return_value=False))
    monkeypatch.setattr(cleanup, "terminate_process", Mock())
    monkeypatch.setattr(cleanup, "_daemon_id", Mock(return_value="daemon-1"))
    monkeypatch.setattr(cleanup, "_inspect_container", Mock(return_value=None))
    monkeypatch.setattr(cleanup, "_remove_container", Mock())
    return root, token


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    return _make_runtime(tmp_path, monkeypatch)


@contextmanager
def _fresh_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str):
    """A private owned runtime whose mocks are discarded at the end of the section."""
    with monkeypatch.context() as m:
        yield _make_runtime(_root(tmp_path, name), m)


def _reclaim(runtime, **kwargs):
    root, token = runtime
    return cleanup.reclaim_runtime(root, token, deadline=time.monotonic() + 2, **kwargs)


def _unacknowledge(runtime) -> None:
    _set_launch(*runtime, container_id=None, starter=None, start_command_sha256=None, create_finished=False)


def test_live_owner_or_wrong_daemon_is_never_reclaimed(tmp_path, monkeypatch) -> None:
    # a live owner is never terminated
    with _fresh_runtime(tmp_path, monkeypatch, "live-owner") as runtime:
        _mock(cleanup.process_is_alive).return_value = True
        _mock(cleanup.terminate_process).side_effect = AssertionError("live owner killed")
        with pytest.raises(cleanup.RuntimeRecoveryError, match="owner.*alive"):
            _reclaim(runtime)
        assert _read(*runtime)["state"] == "active"
    # the wrong daemon refuses absence
    with _fresh_runtime(tmp_path, monkeypatch, "wrong-daemon") as runtime:
        _mock(cleanup._daemon_id).return_value = "different-daemon"
        _mock(cleanup._inspect_container).side_effect = AssertionError("wrong daemon queried")
        with pytest.raises(cleanup.RuntimeRecoveryError, match="daemon"):
            _reclaim(runtime)


def test_missing_container_requires_settled_creation(tmp_path, monkeypatch) -> None:
    for acknowledged in [False, True]:
        with _fresh_runtime(tmp_path, monkeypatch, f"acknowledged-{acknowledged}") as runtime:
            if not acknowledged:
                _unacknowledge(runtime)
            if acknowledged:
                assert _reclaim(runtime) == ("GPU-aaaa", "GPU-bbbb")
            else:
                with pytest.raises(cleanup.RuntimeRecoveryError, match="creation"):
                    _reclaim(runtime)
            assert _read(*runtime)["state"] == ("recovered" if acknowledged else "revoked")


def test_revocation_precedes_termination_and_labels_bind_removal(runtime) -> None:
    root, token = runtime
    calls: list[str] = []

    def terminate(*args: object, **kwargs: object) -> None:
        assert _read(root, token)["state"] == "revoked"
        calls.append("terminate")

    _mock(cleanup.terminate_process).side_effect = terminate
    _mock(cleanup._inspect_container).side_effect = [_container(token), None]
    _mock(cleanup._remove_container).side_effect = lambda *a: calls.append("remove")
    _reclaim(runtime)
    assert calls[-1] == "remove"
    assert "terminate" in calls[:-1]


def test_failed_or_unverified_shutdown_retains_durable_revocation(tmp_path, monkeypatch):
    # failed process termination retains durable revocation
    for index, failure in enumerate([ProcessIdentityError("unverified"), PermissionError()]):
        with _fresh_runtime(tmp_path, monkeypatch, f"failure-{index}") as runtime:
            cleanup.terminate_process.side_effect = failure
            with pytest.raises(cleanup.RuntimeRecoveryError, match="shutdown"):
                _reclaim(runtime)
            assert _read(*runtime)["state"] == "revoked"
            cleanup._daemon_id.assert_not_called()
    # a process still alive prevents container cleanup
    with _fresh_runtime(tmp_path, monkeypatch, "still-alive") as runtime:
        cleanup.process_is_alive.side_effect = [False, True]
        with pytest.raises(cleanup.RuntimeRecoveryError, match="shutdown"):
            _reclaim(runtime)
        assert _read(*runtime)["state"] == "revoked"
        cleanup._daemon_id.assert_not_called()
    # a Docker timeout retains revocation
    with _fresh_runtime(tmp_path, monkeypatch, "docker-timeout") as runtime:
        cleanup._daemon_id.side_effect = ProcessTimeoutError()
        with pytest.raises(cleanup.RuntimeRecoveryError, match="shutdown"):
            _reclaim(runtime)
        assert _read(*runtime)["state"] == "revoked"


def test_all_supervisors_stop_before_helpers_and_duplicate_identities_terminate_once(tmp_path, monkeypatch):
    # all supervisors stop before create and start processes
    with _fresh_runtime(tmp_path, monkeypatch, "ordering") as runtime:
        with _edit(*runtime) as record:
            record.launches["f" * 32] = LaunchRecord(
                "second", replace(current_identity(), pid=104),
                creator=replace(current_identity(), pid=105), create_command_sha256="d" * 64,
                container_id="a" * 64, create_finished=True,
            )
        _reclaim(runtime)
        assert [call.args[0].pid for call in cleanup.terminate_process.call_args_list] == [
            101, 104, 102, 103, 105,
        ]
    # a duplicate process identity is terminated once
    with _fresh_runtime(tmp_path, monkeypatch, "duplicate") as runtime:
        supervisor = replace(current_identity(), pid=101)
        _set_launch(*runtime, creator=supervisor, starter=supervisor)
        _reclaim(runtime)
        cleanup.terminate_process.assert_called_once()


def test_unacknowledged_creation_is_saved_before_removal_and_survives_retry(runtime):
    root, token = runtime
    _unacknowledge(runtime)
    cleanup._inspect_container.side_effect = [_container(token), None]

    def failed_removal(*args):
        persisted = _read(root, token)["launches"][LAUNCH]
        assert persisted["container_id"] == CONTAINER
        assert persisted["create_finished"] is True
        raise cleanup.RuntimeRecoveryError("injected removal interruption")

    cleanup._remove_container.side_effect = failed_removal
    with pytest.raises(cleanup.RuntimeRecoveryError, match="interruption"):
        _reclaim(runtime)
    assert _read(root, token)["state"] == "revoked"
    cleanup._remove_container.side_effect = None
    assert _reclaim(runtime) == ("GPU-aaaa", "GPU-bbbb")
    assert [call.args[0] for call in cleanup._inspect_container.call_args_list] == [
        "owned-name", CONTAINER,
    ]
    assert _read(root, token)["state"] == "recovered"


def test_wrong_container_lingering_container_or_daemon_change_stays_revoked(tmp_path, monkeypatch):
    # wrong container identity or labels never removes
    changes = [
        {"id": "a" * 64}, {"id": "short"}, {"id": None}, {"labels": None}, {"labels": {}},
        {"labels": {"agentic-scientist.lease": "wrong", "agentic-scientist.launch": LAUNCH}},
    ]
    for index, change in enumerate(changes):
        with _fresh_runtime(tmp_path, monkeypatch, f"change-{index}") as runtime:
            cleanup._inspect_container.return_value = _container(runtime[1]) | change
            with pytest.raises(cleanup.RuntimeRecoveryError, match="ownership"):
                _reclaim(runtime)
            cleanup._remove_container.assert_not_called()
            assert _read(*runtime)["state"] == "revoked", change
    # a container still present after a successful rm stays revoked
    with _fresh_runtime(tmp_path, monkeypatch, "lingering") as runtime:
        cleanup._inspect_container.return_value = _container(runtime[1])
        with pytest.raises(cleanup.RuntimeRecoveryError, match="removal.*unverified"):
            _reclaim(runtime)
        assert _read(*runtime)["state"] == "revoked"
    # a daemon change after removal stays revoked
    with _fresh_runtime(tmp_path, monkeypatch, "daemon-change") as runtime:
        cleanup._daemon_id.side_effect = ["daemon-1", "replacement-daemon"]
        with pytest.raises(cleanup.RuntimeRecoveryError, match="changed"):
            _reclaim(runtime)
        assert _read(*runtime)["state"] == "revoked"


def test_voluntary_release_requires_current_recorded_owner_and_never_terminates_it(tmp_path, monkeypatch):
    # a mismatched caller or record refuses the release
    for change in ["caller", "record"]:
        with _fresh_runtime(tmp_path, monkeypatch, f"mismatch-{change}") as runtime:
            releasing_owner = current_identity()
            if change == "caller":
                releasing_owner = replace(releasing_owner, pid=108)
            else:
                with _edit(*runtime) as record:
                    record.owner = replace(releasing_owner, pid=108)
            with pytest.raises(cleanup.RuntimeRecoveryError, match="owner.*match"):
                _reclaim(runtime, releasing_owner=releasing_owner)
            assert _read(*runtime)["state"] == "active"
            cleanup.terminate_process.assert_not_called()
    # the current owner may release but is never terminated
    with _fresh_runtime(tmp_path, monkeypatch, "current-owner") as runtime:
        cleanup.process_is_alive.return_value = False
        _reclaim(runtime, releasing_owner=current_identity())
        assert all(call.args[0] != current_identity() for call in cleanup.terminate_process.call_args_list)


def test_runtimes_without_container_work_recover_without_querying_docker(tmp_path, monkeypatch):
    # an abandoned runtime without launches recovers without Docker
    with _fresh_runtime(tmp_path, monkeypatch, "no-launch") as runtime:
        with _edit(*runtime) as record:
            record.launches.clear()
            record.daemon_id = None
        assert _reclaim(runtime) == ("GPU-aaaa", "GPU-bbbb")
        cleanup.terminate_process.assert_not_called()
        cleanup._daemon_id.assert_not_called()
    # an unstarted supervisor is stopped without querying Docker
    with _fresh_runtime(tmp_path, monkeypatch, "unstarted") as runtime:
        _set_launch(
            *runtime, creator=None, starter=None, container_id=None,
            create_command_sha256=None, start_command_sha256=None, create_finished=False,
        )
        _reclaim(runtime)
        cleanup.terminate_process.assert_called_once()
        cleanup._daemon_id.assert_not_called()
    # replaying a completed recovery has no process or Docker actions
    with _fresh_runtime(tmp_path, monkeypatch, "replay") as runtime:
        _reclaim(runtime)
        cleanup.terminate_process.reset_mock()
        cleanup._daemon_id.reset_mock()
        cleanup.process_is_alive.side_effect = AssertionError("recovered owner queried")
        assert _reclaim(runtime) == ("GPU-aaaa", "GPU-bbbb")
        cleanup.terminate_process.assert_not_called()
        cleanup._daemon_id.assert_not_called()


def test_failed_journal_saves_prevent_termination_and_removal(tmp_path, monkeypatch):
    # a failed revocation save prevents process termination
    with _fresh_runtime(tmp_path, monkeypatch, "revocation-save") as runtime, monkeypatch.context() as m:
        m.setattr(Journal, "save", Mock(side_effect=RuntimeJournalError("injected write failure")))
        with pytest.raises(cleanup.RuntimeRecoveryError, match="shutdown"):
            _reclaim(runtime)
        cleanup.terminate_process.assert_not_called()
        cleanup._daemon_id.assert_not_called()
        assert _read(*runtime)["state"] == "active"
    # a failed container acknowledgement save prevents removal
    with _fresh_runtime(tmp_path, monkeypatch, "acknowledgement-save") as runtime, monkeypatch.context() as m:
        root, token = runtime
        _unacknowledge(runtime)
        cleanup._inspect_container.return_value = _container(token)
        save = Journal.save

        def fail_new_acknowledgement(journal):
            if journal.record.launches[LAUNCH].container_id is not None:
                raise RuntimeJournalError("injected acknowledgement failure")
            save(journal)

        m.setattr(Journal, "save", fail_new_acknowledgement)
        with pytest.raises(cleanup.RuntimeRecoveryError, match="shutdown"):
            _reclaim(runtime)
        cleanup._remove_container.assert_not_called()
        record = _read(root, token)
        assert record["state"] == "revoked"
        assert record["launches"][LAUNCH]["container_id"] is None


def test_recorded_self_supervisor_fails_closed_without_signaling(runtime, monkeypatch):
    _set_launch(*runtime, supervisor=current_identity())
    opener = Mock(side_effect=AssertionError("self guard must run before kernel signaling"))
    monkeypatch.setattr(owner, "_pidfd_open", opener)
    monkeypatch.setattr(cleanup, "terminate_process", owner.terminate_process)
    with pytest.raises(cleanup.RuntimeRecoveryError, match="shutdown"):
        _reclaim(runtime, releasing_owner=current_identity())
    opener.assert_not_called()
    assert _read(*runtime)["state"] == "revoked"


# --------------------------------------------------------------------------- Docker helpers


def _docker_result(monkeypatch, code: int, stdout: bytes, stderr: bytes = b"") -> None:
    monkeypatch.setattr(cleanup, "_docker", lambda *a: BoundedProcessResult(code, stdout, stderr))


def test_docker_inspection_and_removal_report_absence_uncertainty_and_ownership_fields(monkeypatch):
    # malformed or failed inspection is uncertain
    for stdout, code in [(b"garbage", 0), (b"[]", 0), (b"{}", 1), (b"\xff", 0)]:
        with monkeypatch.context() as m:
            _docker_result(m, code, stdout)
            with pytest.raises(cleanup.RuntimeRecoveryError, match="inspection"):
                cleanup._inspect_container(CONTAINER, time.monotonic() + 1)
    # not-found is absence for inspection and removal
    for message in [b"No such container: id", b"Error: No such object: id"]:
        with monkeypatch.context() as m:
            _docker_result(m, 1, b"", message)
            assert cleanup._inspect_container(CONTAINER, time.monotonic() + 1) is None, message
            cleanup._remove_container(CONTAINER, time.monotonic() + 1)
    # a removal failure is not absence
    with monkeypatch.context() as m:
        _docker_result(m, 1, b"", b"permission denied")
        with pytest.raises(cleanup.RuntimeRecoveryError, match="removal"):
            cleanup._remove_container(CONTAINER, time.monotonic() + 1)
    # inspection preserves container ownership fields
    with monkeypatch.context() as m:
        container = _container("a" * 32)
        docker = Mock(return_value=BoundedProcessResult(0, json.dumps(container).encode(), b""))
        m.setattr(cleanup, "_docker", docker)
        assert cleanup._inspect_container(CONTAINER, time.monotonic() + 1) == container
        arguments = docker.call_args.args[0]
        assert arguments[:3] == ["inspect", "--type", "container"]
        assert arguments[-1] == CONTAINER


def test_docker_daemon_identity_is_bounded_identifier_or_uncertain(monkeypatch):
    for stdout, code in [(b"", 0), (b"bad daemon", 0), (b"daemon-1", 1), (b"\xff", 0)]:
        with monkeypatch.context() as m:
            _docker_result(m, code, stdout)
            with pytest.raises(cleanup.RuntimeRecoveryError, match="daemon identity"):
                cleanup._daemon_id(time.monotonic() + 1)
    with monkeypatch.context() as m:
        _docker_result(m, 0, b"daemon-1\n")
        assert cleanup._daemon_id(time.monotonic() + 1) == "daemon-1"


def test_docker_command_pins_local_daemon_bounds_execution_and_fails_closed(monkeypatch):
    # the command pins the local daemon and bounds execution
    with monkeypatch.context() as m:
        m.setattr(cleanup.time, "monotonic", lambda: 1.0)
        m.setattr(cleanup.shutil, "which", lambda command: "/usr/bin/docker")
        keys = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")
        for key in keys:
            m.setenv(key, "foreign")
        runner = Mock(return_value=BoundedProcessResult(0, b"", b""))
        m.setattr(cleanup, "run_bounded_process", runner)
        cleanup._docker(["info"], 3.0)
        call = runner.call_args
        assert call.args[0] == ["/usr/bin/docker", "--host", "unix:///var/run/docker.sock", "info"]
        assert call.kwargs["timeout_seconds"] == 2.0
        assert call.kwargs["max_output_bytes"] == 128 * 1024
        assert call.kwargs["cwd"] == Path("/")
        assert not any(key in call.kwargs["env"] for key in keys)
        cleanup._docker(["info"], 100.0)
        assert runner.call_args.kwargs["timeout_seconds"] == 10.0
    # Docker cannot start after the deadline
    with monkeypatch.context() as m:
        runner = Mock()
        m.setattr(cleanup, "run_bounded_process", runner)
        with pytest.raises(cleanup.RuntimeRecoveryError, match="deadline"):
            cleanup._docker(["info"], time.monotonic() - 1)
        runner.assert_not_called()
    # a missing Docker binary is uncertain
    with monkeypatch.context() as m:
        m.setattr(cleanup.shutil, "which", lambda command: None)
        with pytest.raises(cleanup.RuntimeRecoveryError, match="unavailable"):
            cleanup._docker(["info"], time.monotonic() + 1)
