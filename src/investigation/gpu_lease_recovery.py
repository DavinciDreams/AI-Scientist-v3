"""Coordinate owned runtime cleanup with every GPU lock in the original allocation."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from .container_gpus import GpuLeaseError, GpuLeaseTimeout, _open_gpu_lock_files, _try_lock_all
from .gpu_lease_admin import _clear_marker, _filename, _revalidate_paths
from .gpu_owner_process import ProcessIdentity
from .gpu_quarantine import MAX_MARKER_BYTES, read_quarantine, runtime_marker
from .gpu_runtime_cleanup import RuntimeRecoveryError, reclaim_runtime
from .gpu_runtime_journal import open_runtime


def validate_held_allocation(
    root: Path, devices: tuple[str, ...], descriptors: list[int],
) -> None:
    root_fd = None
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        for device, descriptor in zip(devices, descriptors, strict=True):
            _revalidate_paths(root, root_fd, _filename(device), descriptor)
    except OSError as exc:
        raise GpuLeaseError("GPU allocation lock identity changed or storage is unsafe") from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)


def mark_runtime(descriptors: list[int], token: str) -> None:
    marker = runtime_marker(token)
    for descriptor in descriptors:
        if os.fstat(descriptor).st_size:
            raise GpuLeaseError("GPU runtime ownership requires an unmarked allocation")
        offset = 0
        while offset < len(marker):
            written = os.pwrite(descriptor, marker[offset:], offset)
            if not 0 < written <= len(marker) - offset:
                raise GpuLeaseError("GPU runtime ownership marker write did not progress")
            offset += written
        os.ftruncate(descriptor, len(marker))
        os.fsync(descriptor)


def owned_command(root: Path, token: str, command: list[str]) -> list[str]:
    if command[:2] != ["docker", "run"]:
        raise GpuLeaseError("GPU runtime ownership requires container execution")
    return [
        sys.executable, "-I", str(Path(__file__).with_name("gpu_runtime_launch.py").resolve()),
        "--lease-root", str(root), "--lease-id", token, "--", *command,
    ]


def recover_group(
    root: Path, token: str, held_devices: tuple[str, ...], held_descriptors: list[int],
    *, deadline: float, releasing_owner: ProcessIdentity | None = None,
) -> None:
    with open_runtime(root, token, deadline=deadline) as journal:
        old_devices = journal.record.devices
    held = {device.casefold(): fd for device, fd in zip(held_devices, held_descriptors, strict=True)}
    if not any(device.casefold() in held for device in old_devices):
        raise RuntimeRecoveryError("GPU runtime ownership does not cover requested allocation")
    extra_devices = tuple(device for device in old_devices if device.casefold() not in held)
    extra_descriptors: list[int] = []
    try:
        extra_descriptors = _open_gpu_lock_files(root, extra_devices)
        if not _try_lock_all(extra_descriptors):
            raise GpuLeaseTimeout("Original GPU allocation is still busy; recovery can retry")
        held.update(zip((device.casefold() for device in extra_devices), extra_descriptors, strict=True))
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            for device in old_devices:
                descriptor = held[device.casefold()]
                _revalidate_paths(root, root_fd, _filename(device), descriptor)
                marker = read_quarantine(descriptor, device)
                if marker is not None and marker.recovery_id != token:
                    raise RuntimeRecoveryError("Original GPU allocation ownership changed")
            recovered_devices = reclaim_runtime(
                root, token, deadline=deadline, releasing_owner=releasing_owner,
            )
            if recovered_devices != old_devices:
                raise RuntimeRecoveryError("GPU ownership changed during recovery")
            for device in old_devices:
                descriptor = held[device.casefold()]
                _revalidate_paths(root, root_fd, _filename(device), descriptor)
                marker = read_quarantine(descriptor, device)
                if marker is None:
                    continue
                if marker.recovery_id != token:
                    raise RuntimeRecoveryError("GPU quarantine changed during recovery")
                original = os.pread(descriptor, MAX_MARKER_BYTES + 1, 0)
                _clear_marker(descriptor, original)
        finally:
            os.close(root_fd)
    finally:
        for descriptor in reversed(extra_descriptors):
            os.close(descriptor)
