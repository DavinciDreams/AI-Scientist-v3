"""Bounded recovery of Linux processes whose identities were recorded before launch."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import select
import signal
import sys
import time
from typing import Iterator


class ProcessIdentityError(RuntimeError):
    """A process identity or its verified shutdown could not be established."""


_BOOT_ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_TERM_GRACE_SECONDS = 0.25


def _pidfd_syscall(number: int, *arguments: int) -> int:
    # The bundled CPython can omit pidfd bindings despite kernel support. These
    # syscall numbers are shared by Linux x86_64 and asm-generic (aarch64).
    if sys.platform != "linux" or os.uname().machine not in {"x86_64", "aarch64"}:
        raise ProcessIdentityError("Unsupported platform for process pidfd recovery")
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise ProcessIdentityError("Unsupported process pidfd ABI")
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.restype = ctypes.c_long
    result = syscall(ctypes.c_long(number), *(ctypes.c_long(arg) for arg in arguments))
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, "Process pidfd syscall failed")
    return int(result)


def _pidfd_open(pid: int) -> int:
    native = getattr(os, "pidfd_open", None)
    return int(native(pid, 0)) if native is not None else _pidfd_syscall(434, pid, 0)


def _pidfd_send_signal(descriptor: int, signum: signal.Signals) -> None:
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(descriptor, signum)
    else:
        _pidfd_syscall(424, descriptor, int(signum), 0, 0)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    boot_id: str
    uid: int

    def __post_init__(self) -> None:
        bounds = ((self.pid, 2, 2**31), (self.start_ticks, 1, 2**64), (self.uid, 0, 2**32))
        if any(type(value) is not int or not lower <= value < upper for value, lower, upper in bounds):
            raise ProcessIdentityError("Invalid process identity numbers")
        if not isinstance(self.boot_id, str) or _BOOT_ID.fullmatch(self.boot_id) is None:
            raise ProcessIdentityError("Invalid process identity boot ID")

    def fields(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "start_ticks": self.start_ticks,
            "boot_id": self.boot_id,
            "uid": self.uid,
        }

    @classmethod
    def from_fields(cls, fields: object) -> ProcessIdentity:
        if not isinstance(fields, dict) or fields.keys() != {"pid", "start_ticks", "boot_id", "uid"}:
            raise ProcessIdentityError("Invalid process identity fields")
        return cls(**fields)


def _boot_id() -> str:
    try:
        with Path("/proc/sys/kernel/random/boot_id").open("r", encoding="ascii") as stream:
            value = stream.read(64).strip()
    except (OSError, UnicodeError) as exc:
        raise ProcessIdentityError("Cannot read kernel boot identity") from exc
    if _BOOT_ID.fullmatch(value) is None:
        raise ProcessIdentityError("Invalid kernel boot identity")
    return value


def _parse_stat(data: bytes, pid: int) -> tuple[int, str]:
    try:
        prefix, tail = data.rsplit(b") ", 1)
        observed_pid, _ = prefix.split(b" (", 1)
        fields = tail.split()
        start_ticks = int(fields[19])
        state = fields[0].decode("ascii")
        if (
            int(observed_pid) != pid or not 0 < start_ticks < 2**64
            or len(state) != 1 or state not in "RSDZTWtXxKPI"
        ):
            raise ValueError
    except (ValueError, IndexError, UnicodeError) as exc:
        raise ProcessIdentityError("Invalid process stat identity") from exc
    return start_ticks, state


def _read_process(pid: int) -> tuple[int, int, str] | None:
    descriptor: int | None = None
    try:
        descriptor = os.open(f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        uid = os.fstat(descriptor).st_uid
        stat_fd = os.open("stat", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            data = os.read(stat_fd, 4097)
        finally:
            os.close(stat_fd)
        if len(data) > 4096:
            raise ProcessIdentityError("Oversized process stat identity")
        start_ticks, state = _parse_stat(data, pid)
        return start_ticks, uid, state
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise ProcessIdentityError("Cannot inspect process identity") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def current_identity() -> ProcessIdentity:
    pid = os.getpid()
    observed = _read_process(pid)
    if observed is None or observed[1] != os.getuid():
        raise ProcessIdentityError("Cannot establish current user process identity")
    return ProcessIdentity(pid, observed[0], _boot_id(), observed[1])


def _exited(descriptor: int, timeout_ms: int = 0) -> bool:
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    try:
        events = poller.poll(timeout_ms)
    except OSError as exc:
        raise ProcessIdentityError("Cannot verify pidfd exit") from exc
    if any(flags & (select.POLLNVAL | select.POLLERR) for _, flags in events):
        raise ProcessIdentityError("Invalid pidfd exit observation")
    return any(flags & (select.POLLIN | select.POLLHUP) for _, flags in events)


@contextmanager
def _matching_process(identity: ProcessIdentity) -> Iterator[int | None]:
    if not isinstance(identity, ProcessIdentity):
        raise ProcessIdentityError("Invalid process identity")
    if identity.uid != os.getuid():
        raise ProcessIdentityError("Process identity belongs to a different user")
    if identity.boot_id != _boot_id():
        yield None
        return
    try:
        descriptor = _pidfd_open(identity.pid)
    except ProcessLookupError:
        yield None
        return
    except (OSError, AttributeError) as exc:
        raise ProcessIdentityError("Cannot open process pidfd") from exc
    try:
        if _exited(descriptor):
            yield None
            return
        observed = _read_process(identity.pid)
        if observed is None or observed[0] != identity.start_ticks or observed[2] in {"Z", "X", "x"}:
            yield None
            return
        if observed[1] != identity.uid:
            raise ProcessIdentityError("Process user identity changed")
        yield None if _exited(descriptor) else descriptor
    finally:
        os.close(descriptor)


def process_is_alive(identity: ProcessIdentity) -> bool:
    """Return false for absent/replaced processes; inspection failures remain errors."""
    with _matching_process(identity) as descriptor:
        return descriptor is not None


def _wait_for_exit(descriptor: int, deadline: float) -> bool:
    while True:
        remaining = deadline - time.monotonic()
        if _exited(descriptor, max(0, int(remaining * 1000))):
            return True
        if remaining <= 0:
            return False


def _signal_process(descriptor: int, signum: signal.Signals) -> None:
    try:
        _pidfd_send_signal(descriptor, signum)
    except ProcessLookupError:
        # Confirm exit through the pidfd, including an exit concurrent with signaling.
        return
    except (OSError, AttributeError) as exc:
        raise ProcessIdentityError("Cannot signal identified process") from exc


def terminate_process(identity: ProcessIdentity, *, deadline: float) -> None:
    """Terminate only the captured process; return after confirmed exit or identity absence."""
    if not isinstance(identity, ProcessIdentity):
        raise ProcessIdentityError("Invalid process identity")
    if not math.isfinite(deadline) or deadline <= time.monotonic():
        raise ProcessIdentityError("Process termination deadline elapsed or invalid")
    if identity.pid == os.getpid():
        raise ProcessIdentityError("Refusing self termination")
    with _matching_process(identity) as descriptor:
        if descriptor is None:
            return
        now = time.monotonic()
        if now >= deadline:
            raise ProcessIdentityError("Process termination deadline elapsed")
        _signal_process(descriptor, signal.SIGTERM)
        if _wait_for_exit(descriptor, min(deadline, now + _TERM_GRACE_SECONDS)):
            return
        if time.monotonic() >= deadline:
            raise ProcessIdentityError("Process exit unverified before deadline")
        _signal_process(descriptor, signal.SIGKILL)
        if not _wait_for_exit(descriptor, deadline):
            raise ProcessIdentityError("Process exit unverified before deadline")
