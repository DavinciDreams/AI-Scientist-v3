"""GPU allocation leases: locking, quarantine provenance, owned-runtime recovery, admin, preflight."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.task_drafting import resource_preflight
from src.investigation import container_gpus as gpus
from src.investigation import gpu_lease_admin as admin
from src.investigation import gpu_lease_recovery as recovery
from src.investigation import gpu_runtime_cleanup as cleanup
from src.investigation.container_gpus import (
    GpuLeaseError, GpuLeaseQuarantined, GpuLeaseTimeout, acquire_gpu_allocation_lease,
)
from src.investigation.gpu_lease_admin import _filename
from src.investigation.gpu_quarantine import LEGACY_MARKER, MAX_MARKER_BYTES
from src.investigation.gpu_runtime_journal import open_runtime

DEVICE = "GPU-11111111-1111-1111-1111-111111111111"


def _sub(tmp_path: Path, name: str) -> Path:
    """Private, owner-only parent directory for one merged test section."""
    path = tmp_path / name
    path.mkdir(mode=0o700)
    return path


def _acquire(root: Path, selector: str = "device=GPU-aaaa", seconds: float = 1.0, **kwargs):
    count = selector.count(",") + 1
    return acquire_gpu_allocation_lease(
        selector, count, lease_root=root, deadline=time.monotonic() + seconds, **kwargs,
    )


def _lock(root: Path, device: str = "GPU-aaaa") -> Path:
    return root / _filename(device)


def _abandon(lease: gpus.GpuAllocationLease) -> None:
    for descriptor in lease._descriptors:
        os.close(descriptor)
    lease._descriptors = []


def _markers_clear(root: Path) -> bool:
    return all(path.stat().st_size == 0 for path in root.glob("*.lock"))


def _fail_recovery(*args: object, **kwargs: object) -> None:
    raise cleanup.RuntimeRecoveryError("injected failure")


def _prepared(root: Path) -> gpus.GpuAllocationLease:
    lease = _acquire(root)
    lease.prepare_runtime()
    return lease


# --------------------------------------------------------------------------- lease acquisition


def test_gpu_allocation_lease_sorts_devices_supports_mig_and_releases_cleanly(tmp_path: Path) -> None:
    # sorted stable device ids
    lease = _acquire(tmp_path / "sorted", '"device=GPU-bbbb,GPU-aaaa"')
    try:
        assert lease.devices == ("GPU-aaaa", "GPU-bbbb")
    finally:
        lease.release()
    # MIG identifiers are hashed into the lock filename
    root = tmp_path / "mig"
    lease = _acquire(root, "device=MIG-GPU-aaaa/1/2")
    try:
        names = [path.name for path in root.iterdir()]
        assert len(names) == 1
        assert names[0].endswith(".lock")
        assert "/" not in names[0]
        assert "MIG" not in names[0]
    finally:
        lease.release()
    # modern MIG UUIDs are accepted
    lease = _acquire(tmp_path / "mig-uuid", "device=MIG-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    assert lease.devices == ("MIG-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",)
    lease.release()
    # release is idempotent
    lease = _acquire(tmp_path / "idempotent")
    lease.release()
    lease.release()
    # the context manager releases after an exception
    root = tmp_path / "context"
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with _acquire(root):
            raise RuntimeError("evaluation failed")
    _acquire(root, seconds=0.05).release()
    # disjoint devices are leased concurrently
    root = tmp_path / "disjoint"
    first = _acquire(root)
    try:
        _acquire(root, "device=GPU-bbbb", seconds=0.05).release()
    finally:
        first.release()


def test_gpu_allocation_lease_rejects_unstable_names_invalid_timing_and_unsafe_storage(tmp_path: Path) -> None:
    # unstable device names
    for selector in ["device=0", "device=MIG-device/name"]:
        with pytest.raises(GpuLeaseError, match="stable GPU UUID"):
            _acquire(tmp_path / "unstable", selector)
    # nothing is acquired after the deadline
    with pytest.raises(GpuLeaseTimeout, match="deadline"):
        _acquire(tmp_path / "deadline", seconds=-1)
    # invalid timing
    for deadline, poll_interval in [(float("inf"), 0.05), (float("nan"), 0.05), (0.0, 0.0), (0.0, float("nan"))]:
        with pytest.raises(GpuLeaseError):
            acquire_gpu_allocation_lease(
                "device=GPU-aaaa", 1, lease_root=tmp_path / "timing",
                deadline=deadline, poll_interval_seconds=poll_interval,
            )
    # unsafe shared storage
    for unsafe, match in [
        ("public_root", "owner-controlled"),
        ("symlinked_root", "unavailable or unsafe"),
        ("symlinked_lock", "lock file is unavailable or unsafe"),
    ]:
        base = _sub(tmp_path, unsafe)
        root = base / "leases"
        if unsafe == "public_root":
            root.mkdir(mode=0o777)
            root.chmod(0o777)
        elif unsafe == "symlinked_root":
            (base / "target").mkdir()
            root.symlink_to(base / "target", target_is_directory=True)
        else:
            root.mkdir(mode=0o700)
            _lock(root).symlink_to(base / "target")
        with pytest.raises(GpuLeaseError, match=match):
            _acquire(root)


def test_gpu_allocation_lease_times_out_on_contention_and_releases_partial_acquisition(tmp_path: Path) -> None:
    # an overlapping device times out
    root = tmp_path / "overlap"
    first = _acquire(root)
    try:
        with pytest.raises(GpuLeaseTimeout, match="GPU-aaaa"):
            _acquire(root, seconds=0.05, poll_interval_seconds=0.005)
    finally:
        first.release()
    # a partial acquisition is released on timeout
    root = tmp_path / "partial"
    blocker = _acquire(root, "device=GPU-bbbb")
    try:
        with pytest.raises(GpuLeaseTimeout):
            _acquire(root, '"device=GPU-aaaa,GPU-bbbb"', seconds=0.05, poll_interval_seconds=0.005)
        _acquire(root, seconds=0.05).release()
    finally:
        blocker.release()


def test_gpu_allocation_quarantine_survives_lease_release(tmp_path: Path) -> None:
    root = tmp_path / "leases"
    lease = _acquire(root)
    lease.quarantine()
    lease.release()
    with pytest.raises(GpuLeaseQuarantined, match="GPU-aaaa"):
        _acquire(root)
    _acquire(root, "device=GPU-bbbb").release()


# --------------------------------------------------------------------------- quarantine provenance


def _rewrite_provenance(root: Path, change: dict) -> None:
    prefix, raw = _lock(root).read_bytes().split(b"\n", 1)
    _lock(root).write_bytes(prefix + b"\n" + json.dumps(json.loads(raw) | change).encode())


def test_quarantine_retains_origin_omits_invalid_metadata_and_discards_unknown_fields(tmp_path, monkeypatch):
    # the origin is retained across runs and the marker inode is stable
    root = tmp_path / "origin"
    with monkeypatch.context() as m:
        m.setattr(socket, "gethostname", lambda: "node-a")
        with _acquire(root) as lease:
            inode = _lock(root).stat().st_ino
            lease.quarantine()
            first_marker = _lock(root).read_bytes()
            lease.quarantine()
            assert _lock(root).read_bytes() == first_marker
        m.setattr(socket, "gethostname", lambda: "node-b")
        with pytest.raises(GpuLeaseQuarantined) as error:
            _acquire(root)
        record = error.value.quarantines[0]
        assert record.device == "GPU-aaaa"
        assert record.origin["hostname"] == "node-a"
        assert record.origin["pid"] == os.getpid()
        assert record.marker_sha256 == hashlib.sha256(first_marker).hexdigest()
        assert _lock(root).stat().st_ino == inode
    # invalid environment metadata is omitted
    root = tmp_path / "metadata"
    with monkeypatch.context() as m:
        m.setattr(socket, "gethostname", lambda: "secret\nforged" + "x" * 10000)
        with _acquire(root) as lease:
            lease.quarantine()
        with pytest.raises(GpuLeaseQuarantined) as error:
            _acquire(root)
        assert error.value.quarantines[0].origin["hostname"] is None
        assert "secret" not in _lock(root).read_text()
    # unknown provenance fields are discarded before display
    root = tmp_path / "unknown"
    with _acquire(root) as lease:
        lease.quarantine()
    _rewrite_provenance(root, {"unexpected": "secret material"})
    with pytest.raises(GpuLeaseQuarantined) as error:
        _acquire(root)
    assert error.value.quarantines[0].origin is not None
    assert "secret" not in json.dumps(error.value.quarantines[0].fields())


def test_legacy_damaged_or_malformed_markers_still_block_without_exposing_contents(tmp_path):
    # legacy and damaged markers
    for name, marker, expected_format in [
        ("legacy", b"agentic-scientist-gpu-quarantine-v1\n", "legacy"),
        ("malformed", b"unexpected secret payload", "unrecognized"),
        ("oversized", b"x" * 5000, "oversized"),
    ]:
        root = tmp_path / name
        with _acquire(root):
            _lock(root).write_bytes(marker)
        with pytest.raises(GpuLeaseQuarantined) as error:
            _acquire(root)
        record = error.value.quarantines[0]
        assert record.format == expected_format, name
        assert record.origin is None
        assert "secret" not in json.dumps(record.fields())
        assert len(json.dumps(record.fields())) < 2048
    # malformed v2 provenance stays quarantined
    for index, change in enumerate([
        {"schema_version": []}, {"reason": "secret"}, {"created_at_ns": True}, {"pid": -1}, {"pid": 2**64},
    ]):
        root = tmp_path / f"v2-{index}"
        with _acquire(root) as lease:
            lease.quarantine()
        _rewrite_provenance(root, change)
        with pytest.raises(GpuLeaseQuarantined) as error:
            _acquire(root)
        record = error.value.quarantines[0]
        assert record.format == "unrecognized", change
        assert record.origin is None


# --------------------------------------------------------------------------- owned runtimes under lease locks


def test_managed_work_is_marked_before_launch_and_cleared_after_release(tmp_path: Path) -> None:
    lease = _acquire(tmp_path)
    lease.prepare_runtime()
    command = lease.wrap_command(["docker", "run", "image", "true"])
    assert "--lease-id" in command
    assert "-I" in command
    assert any(path.stat().st_size for path in tmp_path.glob("*.lock"))
    lease.release()
    with _acquire(tmp_path):
        assert _markers_clear(tmp_path)


def test_abrupt_owner_death_recovers_whole_runtime_but_live_legacy_or_partial_groups_are_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # abrupt owner death recovers the whole durable runtime
    for name, devices in [("single", "GPU-aaaa"), ("partial-of-group", "GPU-aaaa,GPU-bbbb")]:
        root = tmp_path / name
        lease = _acquire(root, "device=" + devices)
        lease.prepare_runtime()
        _abandon(lease)
        with monkeypatch.context() as m:
            m.setattr(cleanup, "process_is_alive", lambda identity: False)
            with _acquire(root):
                assert _markers_clear(root), name
    # a live owner with a released lock is not reclaimed
    root = tmp_path / "live"
    _abandon(_prepared(root))
    with pytest.raises(gpus.GpuLeaseQuarantined):
        _acquire(root)
    # a legacy quarantine never triggers automatic termination
    root = tmp_path / "legacy"
    with _acquire(root) as lease:
        lease.quarantine()
    with monkeypatch.context() as m:
        m.setattr(recovery, "reclaim_runtime", lambda *a, **kw: pytest.fail("legacy recovery attempted"))
        with pytest.raises(gpus.GpuLeaseQuarantined):
            _acquire(root)
    # another held GPU lock prevents partial group reclamation
    root = tmp_path / "group"
    lease = _acquire(root, "device=GPU-aaaa,GPU-bbbb")
    lease.prepare_runtime()
    _abandon(lease)
    with _lock(root, "GPU-bbbb").open("rb") as held, monkeypatch.context() as m:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m.setattr(recovery, "reclaim_runtime", lambda *a, **kw: pytest.fail("partial group killed"))
        with pytest.raises(gpus.GpuLeaseTimeout):
            _acquire(root)


def test_failed_cleanup_retains_marker_fence_and_primary_or_authored_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery, "reclaim_runtime", _fail_recovery)
    # failed cleanup retains the marker and the revocation fence
    root = tmp_path / "fence"
    lease = _prepared(root)
    token = next(root.glob("runtime-*.json")).stem.removeprefix("runtime-")
    with pytest.raises(gpus.GpuLeaseQuarantined):
        lease.release()
    with open_runtime(root, token, deadline=time.monotonic() + 1) as journal:
        assert journal.record.state != "recovered"
    assert next(root.glob("*.lock")).stat().st_size > 0
    # a primary interruption survives failed release cleanup
    for interruption in [KeyboardInterrupt, SystemExit]:
        root = tmp_path / interruption.__name__
        with pytest.raises(interruption):
            with _acquire(root) as lease:
                lease.prepare_runtime()
                raise interruption
        assert next(root.glob("*.lock")).stat().st_size > 0
    # failed cleanup promotes an authored failure to quarantine
    root = tmp_path / "authored"
    authored_failure = ValueError("invalid evaluator output")
    with pytest.raises(gpus.GpuLeaseQuarantined) as raised:
        with _acquire(root) as lease:
            lease.prepare_runtime()
            raise authored_failure
    assert raised.value.__cause__ is authored_failure
    assert next(root.glob("*.lock")).stat().st_size > 0


def test_interrupted_release_replaced_lock_and_failed_clear_preserve_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # an interrupted release drops locks but preserves the quarantine
    root = tmp_path / "interrupted"
    lease = _prepared(root)

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(recovery, "reclaim_runtime", interrupt)
        with pytest.raises(KeyboardInterrupt):
            lease.release()
    with _lock(root).open("rb") as descriptor:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert descriptor.read()
    # a replaced lock prevents process termination
    root = tmp_path / "replaced"
    lease = _prepared(root)
    path = _lock(root)
    path.rename(root / "previous-lock")
    path.write_text("replacement")
    path.chmod(0o600)
    with monkeypatch.context() as m:
        m.setattr(recovery, "reclaim_runtime", lambda *a, **kw: pytest.fail("unlocked allocation killed"))
        with pytest.raises(gpus.GpuLeaseQuarantined):
            lease.release()
    assert path.read_text() == "replacement"
    # a failed clear retries verified recovery without launching
    root = tmp_path / "clear"
    lease = _prepared(root)

    def fail(*args: object) -> None:
        raise OSError("injected")

    with monkeypatch.context() as m:
        m.setattr(recovery, "_clear_marker", fail)
        with pytest.raises(gpus.GpuLeaseQuarantined):
            lease.release()
    with _acquire(root):
        assert next(root.glob("*.lock")).stat().st_size == 0


def _abandoned_quarantine(root: Path, devices: str) -> dict:
    lease = _acquire(root, "device=" + devices)
    lease.prepare_runtime()
    _abandon(lease)
    quarantine = admin.inspect_device("GPU-aaaa", lease_root=root)["quarantine"]
    assert isinstance(quarantine, dict)
    return quarantine


def test_manual_recovery_requires_a_dead_owner_and_cleans_the_whole_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # manual recovery cannot bypass a live owned runtime
    root = tmp_path / "live"
    quarantine = _abandoned_quarantine(root, "GPU-aaaa")
    with pytest.raises(gpus.GpuLeaseError, match="unverified"):
        admin.recover_device("GPU-aaaa", quarantine["marker_sha256"], lease_root=root)
    assert next(root.glob("*.lock")).stat().st_size > 0
    # manual recovery uses the full owned-group cleanup
    root = tmp_path / "group"
    quarantine = _abandoned_quarantine(root, "GPU-aaaa,GPU-bbbb")
    monkeypatch.setattr(cleanup, "process_is_alive", lambda identity: False)
    admin.recover_device("GPU-aaaa", quarantine["marker_sha256"], lease_root=root)
    assert _markers_clear(root)


def test_detached_lease_cannot_publish_launch_authority(tmp_path: Path) -> None:
    for transition in ["prepare", "wrap"]:
        for replacement in ["file", "root"]:
            base = _sub(tmp_path, f"{transition}-{replacement}")
            root = base / "leases"
            lease = _acquire(root)
            try:
                if transition == "wrap":
                    lease.prepare_runtime()
                if replacement == "file":
                    path = _lock(root)
                    path.rename(root / "old-lock")
                    path.touch(mode=0o600)
                else:
                    root.rename(base / "old-root")
                    root.mkdir(mode=0o700)
                    _lock(root).touch(mode=0o600)
                with pytest.raises(gpus.GpuLeaseError):
                    if transition == "prepare":
                        lease.prepare_runtime()
                    else:
                        lease.wrap_command(["docker", "run", "image"])
            finally:
                _abandon(lease)


# --------------------------------------------------------------------------- gpu_lease_admin


def _marker(base: Path, data: bytes = LEGACY_MARKER) -> tuple[Path, Path]:
    root = base / "leases"
    root.mkdir(mode=0o700)
    marker = _lock(root, DEVICE)
    marker.write_bytes(data)
    marker.chmod(0o600)
    return root, marker


def _recover(root: Path, data: bytes = LEGACY_MARKER) -> dict[str, object]:
    return admin.recover_device(DEVICE, hashlib.sha256(data).hexdigest(), lease_root=root)


def _status(root: Path) -> str:
    status = admin.inspect_device(DEVICE, lease_root=root)["status"]
    assert isinstance(status, str)
    return status


def test_status_reports_markers_without_creating_or_modifying_storage(tmp_path: Path) -> None:
    # status does not create a missing root or marker
    root = tmp_path / "absent"
    assert _status(root) == "missing"
    assert not root.exists()
    root.mkdir(mode=0o700)
    assert _status(root) == "missing"
    assert list(root.iterdir()) == []
    # status reports a bounded marker without modification
    for data, kind in [(LEGACY_MARKER, "legacy"), (b"unknown", "unrecognized")]:
        root, marker = _marker(_sub(tmp_path, kind), data)
        before = marker.stat()
        report = admin.inspect_device(DEVICE, lease_root=root)
        assert report["status"] == "quarantined"
        quarantine = report["quarantine"]
        assert isinstance(quarantine, dict)
        assert quarantine["format"] == kind
        assert quarantine["marker_sha256"] == hashlib.sha256(data).hexdigest()
        assert marker.read_bytes() == data
        assert marker.stat().st_mtime_ns == before.st_mtime_ns
    # an oversized marker omits the digest
    root, marker = _marker(_sub(tmp_path, "oversized"), b"x" * (MAX_MARKER_BYTES + 1))
    quarantine = admin.inspect_device(DEVICE, lease_root=root)["quarantine"]
    assert isinstance(quarantine, dict)
    assert quarantine["marker_sha256"] is None
    assert marker.stat().st_size == MAX_MARKER_BYTES + 1
    # an invalid device never creates storage
    for device in ["0", "GPU-a,GPU-b", "../secret", "GPU-" + "a" * 300]:
        root = tmp_path / "absent-device"
        with pytest.raises(GpuLeaseError):
            admin.inspect_device(device, lease_root=root)
        assert not root.exists(), device
    # a relative root is rejected
    with pytest.raises(GpuLeaseError, match="absolute"):
        admin.inspect_device(DEVICE, lease_root=Path("relative"))


def test_recovery_truncates_matching_markers_and_refuses_missing_or_mismatched_ones(tmp_path: Path) -> None:
    # recovery truncates the existing inode and leaves it lockable
    for name, data in [("legacy", LEGACY_MARKER), ("unrecognized", b"older unrecognized marker")]:
        root, marker = _marker(_sub(tmp_path, name), data)
        before = marker.stat()
        assert _recover(root, data)["status"] == "recovered"
        assert marker.read_bytes() == b""
        assert marker.stat().st_ino == before.st_ino
        assert _status(root) == "clear"
    # recovery does not create missing storage
    root = tmp_path / "absent"
    with pytest.raises(GpuLeaseError, match="missing"):
        _recover(root)
    assert not root.exists()
    # recovery refuses an invalid digest
    root, marker = _marker(_sub(tmp_path, "digest"))
    for digest in ["", "a" * 63, "f" * 65, "A" * 64, "secret\n"]:
        with pytest.raises(GpuLeaseError, match="digest"):
            admin.recover_device(DEVICE, digest, lease_root=root)
        assert marker.read_bytes() == LEGACY_MARKER
    # recovery refuses stale, oversized, or cleared markers
    for name, data in [("stale", b"changed"), ("oversized", b"x" * (MAX_MARKER_BYTES + 1)), ("cleared", b"")]:
        root, marker = _marker(_sub(tmp_path, name), data)
        with pytest.raises(GpuLeaseError):
            _recover(root)
        assert marker.read_bytes() == data, name


def test_recovery_refuses_busy_foreign_symlinked_or_unsafe_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a busy lease is never recovered
    root, marker = _marker(_sub(tmp_path, "busy"))
    with marker.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(GpuLeaseError, match="busy"):
            _recover(root)
    assert marker.read_bytes() == LEGACY_MARKER
    # the wrong owner is rejected
    root, marker = _marker(_sub(tmp_path, "owner"))
    owner = os.getuid()
    with monkeypatch.context() as m:
        m.setattr(admin.os, "getuid", lambda: owner + 1)
        with pytest.raises(GpuLeaseError, match="owner-controlled"):
            _recover(root)
    assert marker.read_bytes() == LEGACY_MARKER
    # a symlinked root is rejected
    base = _sub(tmp_path, "symlink-root")
    root, marker = _marker(base)
    alias = base / "alias"
    alias.symlink_to(root)
    with pytest.raises(GpuLeaseError):
        _recover(alias)
    assert marker.read_bytes() == LEGACY_MARKER
    # unsafe lease storage is rejected
    for unsafe in ["symlink", "hardlink", "public_file", "public_root", "fifo"]:
        base = _sub(tmp_path, unsafe)
        root, marker = _marker(base)
        original = base / "original"
        if unsafe in {"symlink", "fifo"}:
            marker.rename(original)
            if unsafe == "symlink":
                marker.symlink_to(original)
            else:
                os.mkfifo(marker, 0o600)
        elif unsafe == "hardlink":
            original.hardlink_to(marker)
        elif unsafe == "public_file":
            marker.chmod(0o644)
        else:
            root.chmod(0o755)
        with pytest.raises(GpuLeaseError):
            _recover(root)
        if original.exists():
            assert original.read_bytes() == LEGACY_MARKER, unsafe


def test_recovery_revalidates_path_after_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for replace in ["marker", "root"]:
        base = _sub(tmp_path, replace)
        root, marker = _marker(base)
        original_flock = fcntl.flock
        moved = base / "original"

        def swap_path(descriptor: int, operation: int, replace=replace, root=root, marker=marker, moved=moved) -> None:
            original_flock(descriptor, operation)
            if replace == "marker":
                marker.rename(moved)
                marker.write_bytes(LEGACY_MARKER)
                marker.chmod(0o600)
            else:
                root.rename(moved)
                root.mkdir(mode=0o700)

        with monkeypatch.context() as m:
            m.setattr(admin.fcntl, "flock", swap_path)
            with pytest.raises(GpuLeaseError):
                _recover(root)
        preserved = moved if replace == "marker" else moved / marker.name
        assert preserved.read_bytes() == LEGACY_MARKER, replace


def _cli_recover(root: Path, *extra: str) -> list[str]:
    return [
        "recover", "--device", DEVICE, "--lease-root", str(root),
        "--marker-sha256", hashlib.sha256(LEGACY_MARKER).hexdigest(), *extra,
    ]


def test_cli_recovery_requires_attestation_and_reports_without_exposing_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    # status reads an existing marker
    root, marker = _marker(_sub(tmp_path, "status"))
    assert admin.main(["status", "--device", DEVICE, "--lease-root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "quarantined"
    # recovery requires the cleanup attestation
    with pytest.raises(SystemExit) as error:
        admin.main(_cli_recover(root))
    assert error.value.code == 2
    assert marker.read_bytes() == LEGACY_MARKER
    # recovery with a matching digest and attestation succeeds
    assert admin.main(_cli_recover(root, "--confirm-cleanup")) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "recovered"
    assert marker.read_bytes() == b""
    # errors do not expose paths or raw I/O errors
    def fail_open(*args: object, **kwargs: object) -> int:
        raise OSError("private-secret-path")

    monkeypatch.setattr(admin.os, "open", fail_open)
    assert admin.main(["status", "--device", DEVICE, "--lease-root", str(tmp_path)]) == 4
    output = capsys.readouterr().out
    assert "private-secret-path" not in output
    assert str(tmp_path) not in output
    assert json.loads(output)["status"] == "error"


def _failing_first_fsync(monkeypatch: pytest.MonkeyPatch, failure: BaseException) -> list[int]:
    real_fsync = os.fsync
    syncs: list[int] = []

    def fsync(descriptor: int) -> None:
        syncs.append(descriptor)
        if len(syncs) == 1:
            raise failure
        real_fsync(descriptor)

    monkeypatch.setattr(admin.os, "fsync", fsync)
    return syncs


def test_failed_or_interrupted_recovery_sync_restores_exact_marker_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a failed sync restores the exact marker (short and full writes)
    data = b"unrecognized complete marker\n"
    real_pwrite = os.pwrite
    for write_size in [3, MAX_MARKER_BYTES]:
        root, marker = _marker(_sub(tmp_path, f"write-{write_size}"), data)
        inode = marker.stat().st_ino
        with monkeypatch.context() as m:
            syncs = _failing_first_fsync(m, OSError("private sync failure"))
            m.setattr(
                admin.os, "pwrite",
                lambda descriptor, data, offset, n=write_size: real_pwrite(descriptor, data[:n], offset),
            )
            with pytest.raises(GpuLeaseError, match="original quarantine marker restored"):
                _recover(root, data)
        assert marker.read_bytes() == data
        assert marker.stat().st_ino == inode
        assert len(syncs) == 2, write_size
        assert _status(root) == "quarantined"
    # an interrupted sync restores the marker before propagating the interrupt
    root, marker = _marker(_sub(tmp_path, "interrupted"))
    with monkeypatch.context() as m:
        _failing_first_fsync(m, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            _recover(root)
    assert marker.read_bytes() == LEGACY_MARKER
    assert _status(root) == "quarantined"


def test_failed_marker_restoration_or_descriptor_close_reports_bounded_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # failed restoration reports an uncertain state without the raw error
    real_truncate = os.ftruncate

    def fail_sync(descriptor: int) -> None:
        raise OSError("private-sync-secret")

    def fail_restore_truncate(descriptor: int, length: int) -> None:
        if length:
            raise OSError("private-truncate-secret")
        real_truncate(descriptor, length)

    for restore_failure in ["sync", "write", "truncate"]:
        root, _ = _marker(_sub(tmp_path, restore_failure))
        with monkeypatch.context() as m:
            m.setattr(admin.os, "fsync", fail_sync)
            if restore_failure == "write":
                m.setattr(admin.os, "pwrite", lambda descriptor, data, offset: 0)
            elif restore_failure == "truncate":
                m.setattr(admin.os, "ftruncate", fail_restore_truncate)
            with pytest.raises(GpuLeaseError, match="restoration could not be verified") as error:
                _recover(root)
        assert "private" not in str(error.value), restore_failure
        assert "keep allocation out of service" in str(error.value)
    # a descriptor close failure still closes the root and preserves the primary error
    real_close = os.close
    for name, original_marker in [("legacy", LEGACY_MARKER), ("changed", b"changed")]:
        root, _ = _marker(_sub(tmp_path, f"close-{name}"), original_marker)
        closed: list[int] = []

        def fail_first_close(descriptor: int, closed=closed) -> None:
            real_close(descriptor)
            closed.append(descriptor)
            if len(closed) == 1:
                raise OSError("private-close-secret")

        expected = "descriptor release" if original_marker == LEGACY_MARKER else "marker changed"
        with monkeypatch.context() as m:
            m.setattr(admin.os, "close", fail_first_close)
            with pytest.raises(GpuLeaseError, match=expected):
                _recover(root)
        assert len(closed) == 2, name


# --------------------------------------------------------------------------- drafting resource preflight


def _preflight(root: Path, selector: str | None = "device=GPU-aaaa", **kwargs):
    return resource_preflight.check_gpu_allocation(selector, lease_root=root, **kwargs)


def test_preflight_skips_cpu_passes_healthy_and_blocks_quarantined_devices(tmp_path: Path) -> None:
    # CPU preflight skips without touching the lease root
    root = tmp_path / "cpu"
    for selector in [None, "", "  "]:
        report = _preflight(root, selector)
        assert report.status == "skipped", selector
        assert report.exit_code == 0
        assert report.category is None
        assert not root.exists()
    # a healthy allocation releases every device
    root = tmp_path / "healthy"
    report = _preflight(root, '"device=GPU-aaaa,GPU-bbbb"')
    assert report.status == "passed"
    assert report.exit_code == 0
    assert report.category is None
    with _acquire(root), _acquire(root, "device=GPU-bbbb"):
        pass
    # a quarantine on the second device blocks preflight without touching markers
    root = tmp_path / "quarantined"
    with _acquire(root, "device=GPU-bbbb") as lease:
        lease.quarantine()
    before = {p.name: (p.stat().st_ino, p.read_bytes()) for p in root.iterdir()}
    report = _preflight(root, "device=GPU-aaaa,GPU-bbbb")
    assert report.status == "infrastructure_failure"
    assert report.category == "gpu_allocation_quarantined"
    assert report.exit_code == 4
    for name, identity in before.items():
        path = root / name
        assert (path.stat().st_ino, path.read_bytes()) == identity
    with _acquire(root):
        pass


def test_busy_allocation_is_retryable_and_releases_partial_locks(tmp_path: Path) -> None:
    root = tmp_path / "leases"
    with _acquire(root, "device=GPU-bbbb"):
        start = time.monotonic()
        report = _preflight(root, "device=GPU-aaaa,GPU-bbbb", timeout_seconds=0.02)
        assert time.monotonic() - start < 1
        assert report.category == "gpu_allocation_busy"
        assert report.exit_code == 4
        with _acquire(root):
            pass


def test_invalid_allocations_timeouts_storage_and_release_failures_are_classified_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # invalid allocations fail without leaking the selector
    for selector in [
        "all", "device=0,1", "device=GPU-aaaa,GPU-AAAA", '"device=GPU-aaaa',
        "device=GPU-aaaa,,GPU-bbbb", "device=GPU-aaaa\nsecret", "x" * 8193,
        "device=" + ",".join(f"GPU-{index:04x}" for index in range(17)),
        "device=GPU-" + "a" * 160,
    ]:
        report = _preflight(tmp_path / "invalid", selector)
        assert report.status == "infrastructure_failure", selector[:40]
        assert report.category == "gpu_allocation_invalid"
        assert report.exit_code == 4
        assert report.fields()["diagnostic"] == "GPU allocation or lease storage is invalid"
        assert "selector" not in report.fields()
        assert len(json.dumps(report.fields())) < 1024
    # invalid timeouts cannot make preflight unbounded
    for timeout in [0, -1, float("nan"), float("inf"), 31]:
        report = _preflight(tmp_path / "timeout", timeout_seconds=timeout)
        assert report.category == "gpu_allocation_invalid", timeout
        assert report.exit_code == 4
    # an unsafe lease root is classified invalid
    destination = tmp_path / "destination"
    destination.mkdir(mode=0o700)
    root = tmp_path / "unsafe"
    root.symlink_to(destination, target_is_directory=True)
    assert _preflight(root).category == "gpu_allocation_invalid"
    assert not list(destination.iterdir())
    # a release failure is classified without exposing the error
    @contextmanager
    def failing_release(*_args, **_kwargs):
        yield
        raise GpuLeaseError("secret diagnostic\nprivate path")

    monkeypatch.setattr(resource_preflight, "acquire_gpu_allocation_lease", failing_release)
    report = _preflight(tmp_path / "release")
    assert report.category == "gpu_allocation_invalid"
    assert report.exit_code == 4
    assert "secret" not in json.dumps(report.fields())


def test_cli_emits_one_host_json_report_and_identifies_quarantined_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # one host JSON report per run
    for selector, expected_exit, expected_status in [
        ("", 0, "skipped"), ("device=GPU-aaaa", 0, "passed"), ("device=0", 4, "infrastructure_failure"),
    ]:
        with monkeypatch.context() as m:
            m.setenv("AGENT_CONTAINER_GPUS", selector)
            m.setattr(resource_preflight, "GPU_ALLOCATION_LEASE_ROOT", tmp_path / "leases")
            assert resource_preflight.main([]) == expected_exit, selector
        captured = capsys.readouterr()
        assert not captured.err
        lines = captured.out.splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["schema_version"] == 1
        assert payload["gate"] == "resource_preflight"
        assert payload["stage"] == "resource_selection"
        assert payload["status"] == expected_status
        assert payload["exit_code"] == expected_exit
    # the quarantined device and prior host are identified
    root = tmp_path / "quarantined"
    monkeypatch.setattr(socket, "gethostname", lambda: "node-81001")
    with _acquire(root, "device=GPU-bbbb") as lease:
        lease.quarantine()
    monkeypatch.setattr(socket, "gethostname", lambda: "node-81435")
    monkeypatch.setenv("AGENT_CONTAINER_GPUS", "device=GPU-aaaa,GPU-bbbb")
    monkeypatch.setattr(resource_preflight, "GPU_ALLOCATION_LEASE_ROOT", root)
    assert resource_preflight.main([]) == 4
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert [record["device"] for record in payload["quarantines"]] == ["GPU-bbbb"]
    assert payload["quarantines"][0]["origin"]["hostname"] == "node-81001"
    assert "GPU-bbbb" in captured.err
    assert "node-81001" in captured.err
    assert "gpu_lease_admin status --device GPU-bbbb" in captured.err
    assert "Automatic recovery requires a complete owned-runtime record" in captured.err
    assert "Legacy markers require explicit recovery" in captured.err
