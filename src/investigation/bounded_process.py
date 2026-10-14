"""Bounded outer-process output control for trusted evaluator smoke checks.

This helper is not a candidate sandbox. Process-group signals cannot contain an
adversarial descendant that creates a new session or otherwise escapes the group.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence

DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
MAX_OUTPUT_EVIDENCE_BYTES = 4 * 1024
_READ_SIZE = 8192
_POLL_INTERVAL_SECONDS = 0.01
_TERMINATE_GRACE_SECONDS = 0.1
_CLEANUP_TIMEOUT_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    """Immutable completed-process data with bounded byte output."""

    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(slots=True)
class ProcessCleanupWitness:
    """Host-only observation that the direct child has been reaped."""

    reaped: bool = False
    started: bool = False


class BoundedProcessError(RuntimeError):
    """Base error for a started process that could not complete safely."""


class ProcessCancelledError(BoundedProcessError):
    """The host cancelled a controller after another phase failed."""


class _CapturedProcessError(BoundedProcessError):
    """Bounded output retained separately from public exception messages."""

    _message = "process failed"

    def __init__(
        self,
        message: str = "process failed",
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        observed_output_bytes: int = 0,
        stdout_truncated: bool = False,
        stderr_truncated: bool = False,
    ) -> None:
        # Keep output out of args, str, and repr, including compatibility callers.
        super().__init__(self._message)
        self.stdout = stdout[:MAX_OUTPUT_EVIDENCE_BYTES]
        self.stderr = stderr[:MAX_OUTPUT_EVIDENCE_BYTES]
        self.stdout_truncated = stdout_truncated or len(stdout) > MAX_OUTPUT_EVIDENCE_BYTES
        self.stderr_truncated = stderr_truncated or len(stderr) > MAX_OUTPUT_EVIDENCE_BYTES
        self.max_output_bytes = max_output_bytes
        self.observed_output_bytes = observed_output_bytes


class ProcessTimeoutError(_CapturedProcessError, TimeoutError):
    """Raised after the process exceeds its execution-and-drain deadline."""

    _message = "process exceeded the execution deadline"


class ProcessOutputLimitError(_CapturedProcessError):
    """Raised after combined stdout and stderr exceed the configured cap."""

    _message = "process exceeded the combined output limit"


@dataclass(slots=True)
class _CaptureState:
    max_output_bytes: int
    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)
    stdout_evidence: bytearray = field(default_factory=bytearray)
    stderr_evidence: bytearray = field(default_factory=bytearray)
    total_bytes: int = 0
    observed_bytes: int = 0
    stdout_observed_bytes: int = 0
    stderr_observed_bytes: int = 0
    stdout_complete: bool = False
    stderr_complete: bool = False
    reader_error: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    output_limit: threading.Event = field(default_factory=threading.Event)
    stop: threading.Event = field(default_factory=threading.Event)

    def append(self, target: bytearray, chunk: bytes) -> bool:
        with self.lock:
            self.observed_bytes += len(chunk)
            if target is self.stdout:
                self.stdout_observed_bytes += len(chunk)
            else:
                self.stderr_observed_bytes += len(chunk)
            evidence = self.stdout_evidence if target is self.stdout else self.stderr_evidence
            evidence.extend(chunk[: max(0, MAX_OUTPUT_EVIDENCE_BYTES - len(evidence))])
            remaining = self.max_output_bytes - self.total_bytes
            if len(chunk) > remaining:
                target.extend(chunk[:remaining])
                self.total_bytes = self.max_output_bytes
                self.output_limit.set()
                self.stop.set()
                return False
            target.extend(chunk)
            self.total_bytes += len(chunk)
            return True

    def capture_error(
        self, error_type: type[_CapturedProcessError],
    ) -> _CapturedProcessError:
        with self.lock:
            return error_type(
                stdout=bytes(self.stdout_evidence),
                stderr=bytes(self.stderr_evidence),
                max_output_bytes=self.max_output_bytes,
                observed_output_bytes=self.observed_bytes,
                stdout_truncated=(
                    not self.stdout_complete
                    or self.stdout_observed_bytes > MAX_OUTPUT_EVIDENCE_BYTES
                ),
                stderr_truncated=(
                    not self.stderr_complete
                    or self.stderr_observed_bytes > MAX_OUTPUT_EVIDENCE_BYTES
                ),
            )


def _validate_limits(timeout_seconds: float, max_output_bytes: int) -> None:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive and finite")
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes <= 0
    ):
        raise ValueError("max_output_bytes must be a positive integer")


def _validate_argv(argv: Sequence[str | os.PathLike[str]]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
        raise ValueError("argv must be a non-empty sequence")
    command: list[str] = []
    for argument in argv:
        if isinstance(argument, bytes):
            raise ValueError("argv entries must be strings or path-like values")
        try:
            rendered = os.fspath(argument)
        except TypeError as exc:
            raise ValueError("argv entries must be strings or path-like values") from exc
        if not isinstance(rendered, str):
            raise ValueError("argv entries must resolve to strings")
        command.append(rendered)
    return tuple(command)


def _read_stream(
    stream: BinaryIO,
    target: bytearray,
    state: _CaptureState,
    done: threading.Event,
) -> None:
    try:
        while not state.stop.is_set():
            try:
                chunk = os.read(stream.fileno(), _READ_SIZE)
            except BlockingIOError:
                state.stop.wait(_POLL_INTERVAL_SECONDS)
                continue
            if not chunk:
                with state.lock:
                    if target is state.stdout:
                        state.stdout_complete = True
                    else:
                        state.stderr_complete = True
                return
            if not state.append(target, chunk):
                return
    except (OSError, ValueError):
        if not state.stop.is_set():
            state.reader_error = True
    finally:
        done.set()


def _signal_process_group(process: subprocess.Popen[bytes], signal_number: int) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal_number)
        except ProcessLookupError:
            pass
        return
    try:
        if signal_number == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _stop_and_reap(
    process: subprocess.Popen[bytes],
    state: _CaptureState,
    readers: Sequence[threading.Thread],
) -> None:
    cleanup_deadline = time.monotonic() + _CLEANUP_TIMEOUT_SECONDS
    state.stop.set()
    try:
        _signal_process_group(process, signal.SIGTERM)
    except Exception:
        pass
    try:
        process.wait(
            timeout=min(
                _TERMINATE_GRACE_SECONDS,
                max(0.0, cleanup_deadline - time.monotonic()),
            )
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        pass
    try:
        _signal_process_group(process, getattr(signal, "SIGKILL", signal.SIGTERM))
    except Exception:
        pass
    remaining = max(0.0, cleanup_deadline - time.monotonic())
    if remaining > 0:
        try:
            process.wait(timeout=remaining)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            pass
    for reader in readers:
        remaining = max(0.0, cleanup_deadline - time.monotonic())
        if remaining <= 0:
            break
        try:
            reader.join(timeout=remaining)
        except RuntimeError:
            pass


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except Exception:
            pass


def run_bounded_process(
    argv: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str],
    env: Mapping[str, str],
    timeout_seconds: float,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    child_umask: int | None = None,
    cleanup_witness: ProcessCleanupWitness | None = None,
    cancel_event: threading.Event | None = None,
) -> BoundedProcessResult:
    """Control a trusted evaluator process's output and execution deadline.

    The new process group supports best-effort cleanup of ordinary descendants,
    not security containment. Candidate code requires a separate sandbox.
    """
    if cleanup_witness is not None:
        cleanup_witness.reaped = False
        cleanup_witness.started = False
    _validate_limits(timeout_seconds, max_output_bytes)
    if child_umask is not None and (
        isinstance(child_umask, bool)
        or not isinstance(child_umask, int)
        or child_umask < 0
        or child_umask > 0o777
    ):
        raise ValueError("child_umask must be an integer from 0 to 0o777")
    command = _validate_argv(argv)
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=Path(cwd),
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        start_new_session=True,
        umask=-1 if child_umask is None else child_umask,
    )
    if cleanup_witness is not None:
        cleanup_witness.started = True
    assert process.stdout is not None
    assert process.stderr is not None

    state = _CaptureState(max_output_bytes)
    stdout_done = threading.Event()
    stderr_done = threading.Event()
    readers = (
        threading.Thread(
            target=_read_stream,
            args=(process.stdout, state.stdout, state, stdout_done),
            daemon=True,
        ),
        threading.Thread(
            target=_read_stream,
            args=(process.stderr, state.stderr, state, stderr_done),
            daemon=True,
        ),
    )
    deadline = started + float(timeout_seconds)
    completed = False
    try:
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        for reader in readers:
            reader.start()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise ProcessCancelledError("controller cancelled")
            if state.output_limit.is_set():
                raise state.capture_error(ProcessOutputLimitError)
            if time.monotonic() >= deadline:
                raise state.capture_error(ProcessTimeoutError)
            returncode = process.poll()
            if returncode is not None and stdout_done.is_set() and stderr_done.is_set():
                for reader in readers:
                    reader.join()
                if state.output_limit.is_set():
                    raise state.capture_error(ProcessOutputLimitError)
                if state.reader_error:
                    raise BoundedProcessError("process output capture failed")
                completed = True
                return BoundedProcessResult(
                    returncode,
                    bytes(state.stdout),
                    bytes(state.stderr),
                )
            state.output_limit.wait(
                timeout=min(_POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic()))
            )
    finally:
        if not completed:
            state.stop.set()
            try:
                _stop_and_reap(process, state, readers)
            except Exception:
                pass
        if cleanup_witness is not None:
            cleanup_witness.reaped = process.returncode is not None
        _close_process_pipes(process)
