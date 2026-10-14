"""Host-wide capacity lease for expensive task runtime construction."""

from __future__ import annotations

import fcntl
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path

RUNTIME_BUILD_LEASE_ROOT = Path("/tmp") / (
    f"agentic-scientist-runtime-build-leases-{os.getuid()}"
)
_LOCK_POLL_SECONDS = 0.05


class RuntimeBuildLeaseError(RuntimeError):
    """Raised when the build-capacity lease cannot be acquired safely."""


class RuntimeBuildLeaseTimeout(RuntimeBuildLeaseError):
    """Raised when build capacity remains occupied until the deadline."""


@dataclass(slots=True)
class RuntimeBuildLease:
    """Exclusive host runtime-build slot backed by an advisory file lock."""

    _descriptor: int
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        os.close(self._descriptor)
        self._released = True

    def __enter__(self) -> RuntimeBuildLease:
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def acquire_runtime_build_lease(
    *,
    deadline: float,
    lease_root: Path = RUNTIME_BUILD_LEASE_ROOT,
) -> RuntimeBuildLease:
    """Acquire the single owner-private runtime-build capacity slot."""
    if not math.isfinite(deadline):
        raise RuntimeBuildLeaseError("runtime build lease deadline must be finite")
    _ensure_private_root(lease_root)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    try:
        descriptor = os.open(lease_root / "build.lock", flags, 0o600)
    except OSError as exc:
        raise RuntimeBuildLeaseError(f"could not open runtime build lease: {exc}") from exc
    try:
        _validate_lock(descriptor)
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return RuntimeBuildLease(descriptor)
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeBuildLeaseTimeout(
                        "runtime build capacity remained busy until the deadline"
                    )
                time.sleep(min(_LOCK_POLL_SECONDS, remaining))
    except BaseException:
        os.close(descriptor)
        raise


def _ensure_private_root(root: Path) -> None:
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise RuntimeBuildLeaseError(f"could not inspect runtime build lease root: {exc}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RuntimeBuildLeaseError(f"unsafe runtime build lease root: {root}")


def _validate_lock(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise RuntimeBuildLeaseError("unsafe runtime build lease file")
