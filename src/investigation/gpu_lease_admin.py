"""Inspect GPU quarantine, recover owned runtimes, or clear verified legacy markers.

Recovery enforces lock ownership and marker identity. Owned runtimes require complete
shutdown proof. Legacy markers require independent operator verification; scheduler
state, GPU idleness, and marker age are not proof of cleanup.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.investigation.container_gpus import (
    GPU_ALLOCATION_LEASE_ROOT,
    GpuLeaseError,
    _stable_gpu_devices,
)
from src.investigation.gpu_quarantine import MAX_MARKER_BYTES, read_quarantine


def _filename(device: str) -> str:
    if len(device) > 256:
        raise GpuLeaseError("GPU device identifier is invalid")
    devices = _stable_gpu_devices("device=" + device, 1)
    if len(devices) != 1 or devices[0] != device:
        raise GpuLeaseError("Exactly one stable GPU device identifier is required")
    return hashlib.sha256(device.casefold().encode("ascii")).hexdigest() + ".lock"


def _validate_root(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise GpuLeaseError("GPU lease root must be a private owner-controlled directory")


def _validate_file(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
    ):
        raise GpuLeaseError("GPU lease must be a private owner-controlled regular file")


def _revalidate_paths(root: Path, root_fd: int, filename: str, descriptor: int) -> None:
    root_metadata = os.fstat(root_fd)
    linked_root = os.stat(root, follow_symlinks=False)
    _validate_root(root_metadata)
    _validate_root(linked_root)
    if (root_metadata.st_dev, root_metadata.st_ino) != (linked_root.st_dev, linked_root.st_ino):
        raise GpuLeaseError("GPU lease root changed during inspection")
    metadata = os.fstat(descriptor)
    linked_file = os.stat(filename, dir_fd=root_fd, follow_symlinks=False)
    _validate_file(metadata)
    _validate_file(linked_file)
    if (metadata.st_dev, metadata.st_ino) != (linked_file.st_dev, linked_file.st_ino):
        raise GpuLeaseError("GPU lease file changed during inspection")


@contextmanager
def _existing_lease(
    device: str, root: Path, access: int,
) -> Iterator[tuple[int, int, str] | None]:
    filename = _filename(device)
    if not root.is_absolute():
        raise GpuLeaseError("GPU lease root must be absolute")
    root_fd: int | None = None
    descriptor: int | None = None
    primary_error: BaseException | None = None
    try:
        try:
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except FileNotFoundError:
            yield None
            return
        _validate_root(os.fstat(root_fd))
        try:
            descriptor = os.open(
                filename, access | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=root_fd,
            )
        except FileNotFoundError:
            yield None
            return
        _validate_file(os.fstat(descriptor))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise GpuLeaseError("GPU allocation is busy; marker remains unchanged") from exc
            raise
        _revalidate_paths(root, root_fd, filename, descriptor)
        yield descriptor, root_fd, filename
    except OSError as exc:
        primary_error = GpuLeaseError("GPU lease storage operation failed or storage is unsafe")
        raise primary_error from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        close_error: OSError | None = None
        for open_descriptor in (descriptor, root_fd):
            if open_descriptor is not None:
                try:
                    os.close(open_descriptor)
                except OSError as exc:
                    close_error = close_error or exc
        if close_error is not None and primary_error is None:
            raise GpuLeaseError("GPU lease descriptor release could not be verified") from close_error


def inspect_device(
    device: str, *, lease_root: Path = GPU_ALLOCATION_LEASE_ROOT,
) -> dict[str, object]:
    """Read a consistent marker without creating or altering lease storage."""
    with _existing_lease(device, lease_root, os.O_RDONLY) as opened:
        if opened is None:
            return {"device": device, "status": "missing", "quarantine": None}
        descriptor, root_fd, filename = opened
        quarantine = read_quarantine(descriptor, device)
        _revalidate_paths(lease_root, root_fd, filename, descriptor)
        return {
            "device": device,
            "status": "clear" if quarantine is None else "quarantined",
            "quarantine": None if quarantine is None else quarantine.fields(),
        }


def _clear_marker(descriptor: int, original_marker: bytes) -> None:
    try:
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
    except BaseException as clear_error:
        try:
            offset = 0
            while offset < len(original_marker):
                written = os.pwrite(descriptor, original_marker[offset:], offset)
                if not 0 < written <= len(original_marker) - offset:
                    raise OSError("GPU quarantine restoration write did not progress")
                offset += written
            os.ftruncate(descriptor, len(original_marker))
            os.fsync(descriptor)
        except BaseException as restore_error:
            raise GpuLeaseError(
                "GPU recovery failed; quarantine restoration could not be verified; "
                "keep allocation out of service"
            ) from restore_error
        if isinstance(clear_error, OSError):
            raise GpuLeaseError(
                "GPU recovery failed; original quarantine marker restored"
            ) from clear_error
        raise


def recover_device(
    device: str, marker_sha256: str, *, lease_root: Path = GPU_ALLOCATION_LEASE_ROOT,
) -> dict[str, object]:
    """Clear the exact inspected marker after the caller verifies external shutdown."""
    if re.fullmatch(r"[a-f0-9]{64}", marker_sha256) is None:
        raise GpuLeaseError("An exact lowercase SHA-256 marker digest is required")
    with _existing_lease(device, lease_root, os.O_RDWR) as opened:
        if opened is None:
            raise GpuLeaseError("GPU quarantine marker is missing; inspect before recovery")
        descriptor, root_fd, filename = opened
        quarantine = read_quarantine(descriptor, device)
        if quarantine is None:
            raise GpuLeaseError("GPU quarantine marker is already clear")
        if quarantine.marker_sha256 is None:
            raise GpuLeaseError("GPU quarantine marker is oversized; recovery refused")
        if quarantine.marker_sha256 != marker_sha256:
            raise GpuLeaseError("GPU quarantine marker changed; inspect before recovery")
        original_marker = os.pread(descriptor, MAX_MARKER_BYTES + 1, 0)
        if (
            len(original_marker) > MAX_MARKER_BYTES
            or len(original_marker) != os.fstat(descriptor).st_size
            or hashlib.sha256(original_marker).hexdigest() != marker_sha256
        ):
            raise GpuLeaseError("GPU quarantine marker changed; inspect before recovery")
        _revalidate_paths(lease_root, root_fd, filename, descriptor)
        if quarantine.recovery_id is not None:
            from .gpu_lease_recovery import recover_group
            from .gpu_runtime_cleanup import RuntimeRecoveryError
            from .gpu_runtime_journal import RuntimeJournalError
            try:
                recover_group(
                    lease_root, quarantine.recovery_id, (device,), [descriptor],
                    deadline=time.monotonic() + 30,
                )
            except (RuntimeRecoveryError, RuntimeJournalError) as exc:
                raise GpuLeaseError("Owned GPU runtime cleanup is unverified; recovery refused") from exc
        else:
            _clear_marker(descriptor, original_marker)
        return {"device": device, "status": "recovered", "previous_quarantine": quarantine.fields()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "recover"):
        command = commands.add_parser(name)
        command.add_argument("--device", required=True)
        command.add_argument("--lease-root", type=Path, default=GPU_ALLOCATION_LEASE_ROOT)
        if name == "recover":
            command.add_argument("--marker-sha256", required=True)
            command.add_argument(
                "--confirm-cleanup", action="store_true", required=True,
                help="Attest all evaluator/candidate containers and launchers are verified stopped",
            )
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            report = inspect_device(args.device, lease_root=args.lease_root)
        else:
            report = recover_device(args.device, args.marker_sha256, lease_root=args.lease_root)
    except GpuLeaseError as exc:
        print(json.dumps({"status": "error", "diagnostic": str(exc)}, sort_keys=True))
        return 4
    except OSError:
        print(json.dumps({"status": "error", "diagnostic": "GPU lease storage operation failed"}))
        return 4
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
