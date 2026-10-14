"""Host-owned bridge between a final evaluator and an isolated candidate container."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import select
import shutil
import socket
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Protocol

from src.investigation.task_contract_types import (
    DEFAULT_MAX_CANDIDATE_RPC_CALLS,
    HARD_MAX_CANDIDATE_RPC_CALLS,
)

MAX_RPC_BYTES = 64 * 1024 * 1024
MAX_CANDIDATE_STDERR_BYTES = 64 * 1024
MAX_CANDIDATE_STDERR_TOTAL_BYTES = 1024 * 1024
MAX_CANDIDATE_CLEANUP_SECONDS = 30.0
_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}\Z")


class CandidateBrokerError(RuntimeError):
    """Candidate broker failed or violated its bounded protocol."""


class CandidateOperationDeadlineExceeded(CandidateBrokerError):
    """Internal deadline signal; only the host can produce a recoverable timeout."""


class CandidateBrokerInfrastructureError(CandidateBrokerError):
    """The broker could not launch or clean up its container runtime."""


class CandidateContainerCleanupError(CandidateBrokerInfrastructureError):
    """Candidate container or launch-process shutdown could not be verified."""


class ContainerMetadataCleanupError(CandidateBrokerInfrastructureError):
    """Host cleanup failed after container and launch-process shutdown was verified."""


def _cleanup_diagnostic(error: BaseException, operation: str) -> str:
    if isinstance(error, CandidateBrokerError):
        return str(error)[:1024]
    diagnostic = f"unexpected {type(error).__name__[:80]} operation={operation}"
    if isinstance(error, OSError):
        errno_name = errno.errorcode.get(error.errno) if type(error.errno) is int else None
        diagnostic += f" errno={error.errno} ({errno_name})" if errno_name else " errno=unknown"
    return diagnostic


def read_container_id(cid_path: Path) -> str | None:
    try:
        descriptor = os.open(
            cid_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CandidateBrokerInfrastructureError(
            "candidate container recovery metadata is unsafe"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise CandidateBrokerInfrastructureError(
                "candidate container recovery metadata is unsafe"
            )
        raw_container_id = os.read(descriptor, 129)
    finally:
        os.close(descriptor)
    if len(raw_container_id) > 128:
        raise CandidateBrokerInfrastructureError("invalid candidate container recovery metadata")
    try:
        container_id = raw_container_id.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise CandidateBrokerInfrastructureError("invalid candidate recovery metadata") from exc
    if _CONTAINER_ID.fullmatch(container_id) is None:
        raise CandidateBrokerInfrastructureError("invalid candidate container recovery metadata")
    return container_id


class _ResponseConnection(Protocol):
    def sendall(self, data: bytes) -> None: ...


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateBrokerError("candidate RPC request contains duplicate fields")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise CandidateBrokerError(f"candidate RPC request contains invalid constant: {value}")


def _decode_evaluator_request(payload: bytes) -> object:
    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except CandidateBrokerError:
        raise
    except (UnicodeError, RecursionError, ValueError) as exc:
        raise CandidateBrokerError("candidate RPC request contains invalid JSON") from exc
    return parsed


def _parse_evaluator_request(payload: bytes) -> dict[str, object]:
    parsed = _decode_evaluator_request(payload)
    if not isinstance(parsed, dict):
        raise CandidateBrokerError("candidate RPC request must be an object")
    operation = parsed.get("op")
    if not isinstance(operation, str) or not operation:
        raise CandidateBrokerError("candidate RPC request requires a non-empty string op")
    if operation == "reset" and "seed" not in parsed:
        raise CandidateBrokerError("candidate reset request requires seed")
    return parsed


def _parse_broker_request(payload: bytes, maximum: float) -> tuple[dict[str, object], float | None]:
    envelope = _decode_evaluator_request(payload)
    if not isinstance(envelope, dict) or "__agentic_rpc__" not in envelope:
        return _parse_evaluator_request(payload), None
    if set(envelope) != {"__agentic_rpc__", "timeout_seconds", "request"} or type(
        envelope["__agentic_rpc__"]
    ) is not int or envelope["__agentic_rpc__"] != 1:
        raise CandidateBrokerError("invalid broker operation envelope")
    timeout = envelope["timeout_seconds"]
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 0 < timeout <= maximum):
        raise CandidateBrokerError("operation timeout must fit configured RPC timeout")
    request = _parse_evaluator_request(json.dumps(envelope["request"], allow_nan=False).encode())
    if request["op"] == "reset":
        raise CandidateBrokerError("operation timeout does not apply to reset")
    return request, float(timeout)


def _operation_response(status: str, response: bytes | None = None) -> bytes:
    payload = json.dumps({"__agentic_rpc__": 1, "status": status,
                          "response": None if response is None else response.decode("utf-8")},
                         separators=(",", ":")).encode()
    if len(payload) > MAX_RPC_BYTES:
        raise CandidateBrokerError("broker operation response exceeds 64 MiB")
    return payload


@dataclass(frozen=True)
class CandidateDiagnostic:
    category: str
    sha256: str
    total_bytes: int
    captured_bytes: int
    truncated: bool

    def summary(self) -> str:
        return (
            f"candidate_stderr={self.category} sha256={self.sha256} "
            f"bytes={self.total_bytes} truncated={str(self.truncated).lower()}"
        )


class CandidateBroker:
    def __init__(
        self,
        command: list[str],
        socket_dir: Path,
        *,
        rpc_timeout_seconds: float = 21600,
        reset_timeout_seconds: float | None = None,
        max_rpc_calls: int = DEFAULT_MAX_CANDIDATE_RPC_CALLS,
        expected_reset_seeds: tuple[int, ...] | None = None,
        completed_reset_limit: int | None = None,
    ) -> None:
        if not math.isfinite(rpc_timeout_seconds) or rpc_timeout_seconds <= 0:
            raise ValueError("candidate RPC timeout must be positive and finite")
        if reset_timeout_seconds is None:
            reset_timeout_seconds = rpc_timeout_seconds
        if (
            not math.isfinite(reset_timeout_seconds)
            or reset_timeout_seconds <= 0
            or reset_timeout_seconds > rpc_timeout_seconds
        ):
            raise ValueError(
                "candidate reset timeout must be positive, finite, and no greater than RPC timeout"
            )
        if (
            isinstance(max_rpc_calls, bool)
            or not isinstance(max_rpc_calls, int)
            or not 1 <= max_rpc_calls <= HARD_MAX_CANDIDATE_RPC_CALLS
        ):
            raise ValueError(
                "candidate RPC call budget must be an integer between 1 and "
                f"{HARD_MAX_CANDIDATE_RPC_CALLS}"
            )
        if expected_reset_seeds is not None and (
            not expected_reset_seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in expected_reset_seeds
            )
        ):
            raise ValueError("candidate expected reset seeds must be non-negative integers")
        if completed_reset_limit is not None and (
            isinstance(completed_reset_limit, bool)
            or not isinstance(completed_reset_limit, int)
            or completed_reset_limit <= 0
        ):
            raise ValueError("candidate completed reset limit must be a positive integer")
        self.command = command
        self.socket_dir = socket_dir
        self.socket_path = socket_dir / "candidate.sock"
        self.cid_path = socket_dir / "candidate.cid"
        self.rpc_timeout_seconds = rpc_timeout_seconds
        self.reset_timeout_seconds = reset_timeout_seconds
        self.max_rpc_calls = max_rpc_calls
        self.completed_reset_limit = completed_reset_limit
        self._allowed_reset_seeds = (
            None
            if expected_reset_seeds is None
            else (*expected_reset_seeds, expected_reset_seeds[0])
        )
        self._server: socket.socket | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._stderr_tail = bytearray()
        self._stderr_sha256 = hashlib.sha256()
        self._stderr_total_bytes = 0
        self._stdout_buffer = bytearray()
        self._error: BaseException | None = None
        self._stopping = False
        self._shutdown_verified = False
        self.calls = 0
        self.candidate_process_starts = 0
        self._reset_seeds: list[object] = []
        self._completed_reset_seeds: list[int] = []
        self.peer_disconnects = 0
        self.calls_before_first_reset = 0
        self._reset_required = False

    @property
    def reset_seeds(self) -> tuple[object, ...]:
        return tuple(self._reset_seeds)

    @property
    def reset_count(self) -> int:
        return len(self._reset_seeds)

    @property
    def completed_reset_seeds(self) -> tuple[int, ...]:
        with self._state_lock:
            return tuple(self._completed_reset_seeds)

    @property
    def candidate_diagnostic(self) -> CandidateDiagnostic | None:
        with self._state_lock:
            if self._stderr_total_bytes == 0:
                return None
            captured = bytes(self._stderr_tail)
            total = self._stderr_total_bytes
            digest = self._stderr_sha256.hexdigest()
        return CandidateDiagnostic(
            category=_stderr_category(captured),
            sha256=digest,
            total_bytes=total,
            captured_bytes=len(captured),
            truncated=total > len(captured),
        )

    @property
    def shutdown_verified(self) -> bool:
        return self._shutdown_verified

    def __enter__(self) -> "CandidateBroker":
        self._shutdown_verified = False
        created_paths: list[Path] = []
        socket_bound = False
        try:
            self.socket_dir.mkdir(mode=0o700)
            created_paths.append(self.socket_dir)
            mask_dir = self.socket_dir / "scoring-mask"
            mask_dir.mkdir(mode=0o700)
            created_paths.append(mask_dir)
            self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server.bind(str(self.socket_path))
            socket_bound = True
            os.chmod(self.socket_path, 0o600)
            self._server.listen(1)
            self._server.settimeout(1.0)
        except OSError as exc:
            if self._server is not None:
                try:
                    self._server.close()
                except OSError:
                    pass
            if socket_bound:
                try:
                    self.socket_path.unlink()
                except OSError:
                    pass
            for path in reversed(created_paths):
                try:
                    path.rmdir()
                except OSError:
                    pass
            raise CandidateBrokerInfrastructureError(
                "candidate broker setup failed: " + _cleanup_diagnostic(exc, "broker_setup")
            ) from exc
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def _start_process(self) -> subprocess.Popen[bytes]:
        with self._state_lock:
            self._stderr_tail.clear()
            self._stderr_sha256 = hashlib.sha256()
            self._stderr_total_bytes = 0
        self._stdout_buffer.clear()
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise CandidateBrokerInfrastructureError(
                "candidate container runtime failed to launch"
            ) from exc
        self.candidate_process_starts += 1
        assert process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._capture_stderr,
            args=(process.stderr, process),
            daemon=True,
        )
        self._stderr_thread.start()
        return process

    def _capture_stderr(
        self,
        stream: IO[bytes],
        process: subprocess.Popen[bytes] | None = None,
    ) -> None:
        while chunk := stream.read(4096):
            exceeded = False
            with self._state_lock:
                self._stderr_sha256.update(chunk)
                self._stderr_total_bytes += len(chunk)
                self._stderr_tail.extend(chunk)
                excess = len(self._stderr_tail) - MAX_CANDIDATE_STDERR_BYTES
                if excess > 0:
                    del self._stderr_tail[:excess]
                exceeded = self._stderr_total_bytes > MAX_CANDIDATE_STDERR_TOTAL_BYTES
            if exceeded:
                self._record_error_once(
                    CandidateBrokerInfrastructureError("candidate stderr output limit exceeded")
                )
                if process is not None:
                    try:
                        process.terminate()
                    except OSError:
                        pass
                return

    def _record_error_once(self, error: BaseException) -> None:
        with self._state_lock:
            self._error = self._error or error

    def _restart_process(self, deadline: float) -> None:
        if self._process is not None:
            try:
                self._stop_process(self._process, deadline=deadline)
            except (CandidateContainerCleanupError, ContainerMetadataCleanupError):
                raise
            except BaseException as exc:
                raise CandidateContainerCleanupError(
                    _cleanup_diagnostic(exc, "process_stop")
                ) from exc
            try:
                self.cid_path.unlink(missing_ok=True)
            except OSError as exc:
                raise ContainerMetadataCleanupError(
                    _cleanup_diagnostic(exc, "candidate_cid_unlink")
                ) from exc
        if time.monotonic() >= deadline:
            raise CandidateBrokerError("candidate reset timed out during container restart")
        self._process = self._start_process()

    def _stop_process(
        self, process: subprocess.Popen[bytes], *, deadline: float | None = None
    ) -> None:
        cleanup_errors: list[BaseException] = []

        def remove_container() -> CandidateContainerCleanupError | None:
            if not (self.cid_path.exists() or self.cid_path.is_symlink()):
                return None
            try:
                container_id = read_container_id(self.cid_path)
                assert container_id is not None
                result = subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=_remaining_timeout(deadline, 10),
                    check=False,
                )
                if result.returncode != 0 and "no such container" not in result.stderr.lower():
                    raise CandidateContainerCleanupError(
                        "candidate container runtime could not remove container"
                    )
            except CandidateBrokerInfrastructureError as exc:
                return CandidateContainerCleanupError(_cleanup_diagnostic(exc, "container_remove"))
            except BaseException:
                return CandidateContainerCleanupError(
                    "candidate container runtime could not remove container"
                )
            return None

        try:
            process.terminate()
        except BaseException as exc:
            if process.poll() is None:
                cleanup_errors.append(exc)
        remove_container()
        try:
            process.wait(timeout=_remaining_timeout(deadline, 5))
            cleanup_errors.clear()
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=_remaining_timeout(deadline, 2))
                cleanup_errors.clear()
            except BaseException as exc:
                cleanup_errors.append(exc)
        except BaseException as exc:
            cleanup_errors.append(exc)
        removal_error = remove_container()
        if removal_error is not None:
            cleanup_errors.append(removal_error)
        metadata_error = None
        if self._stderr_thread is not None:
            try:
                self._stderr_thread.join(timeout=_remaining_timeout(deadline, 1))
            except BaseException as exc:
                metadata_error = exc
        if cleanup_errors:
            infrastructure_error = next(
                (
                    error
                    for error in cleanup_errors
                    if isinstance(error, CandidateBrokerInfrastructureError)
                ),
                None,
            )
            cause = infrastructure_error or cleanup_errors[0]
            raise CandidateContainerCleanupError(
                _cleanup_diagnostic(cause, "process_stop")
            ) from cause
        if metadata_error is not None:
            raise ContainerMetadataCleanupError(
                _cleanup_diagnostic(metadata_error, "stderr_thread_join")
            ) from metadata_error

    def __exit__(self, exception_type: object, *_: object) -> None:
        self._stopping = True
        cleanup_deadline = time.monotonic() + MAX_CANDIDATE_CLEANUP_SECONDS
        cleanup_errors: list[tuple[str, BaseException]] = []

        def clean(operation: str, action: Callable[[], object]) -> bool:
            try:
                action()
            except BaseException as exc:
                if operation in {"process_stop", "thread_join"} and not isinstance(
                    exc, (CandidateContainerCleanupError, ContainerMetadataCleanupError)
                ):
                    exc = CandidateContainerCleanupError(_cleanup_diagnostic(exc, operation))
                cleanup_errors.append((operation, exc))
                return False
            return True

        if self._server is not None:
            clean("server_close", self._server.close)
        process = self._process
        if process is not None:
            clean("process_stop", lambda: self._stop_process(process, deadline=cleanup_deadline))
        thread = self._thread
        if thread is not None:
            def stop_launch_thread() -> None:
                thread.join(timeout=_remaining_timeout(cleanup_deadline, 2))
                if thread.is_alive():
                    raise CandidateContainerCleanupError("candidate launch thread did not stop")

            clean("thread_join", stop_launch_thread)
        late_process = self._process
        if late_process is not None and late_process is not process:
            clean("process_stop", lambda: self._stop_process(late_process, deadline=cleanup_deadline))

        def remove_scoring_mask() -> None:
            try:
                shutil.rmtree(self.socket_dir / "scoring-mask")
            except FileNotFoundError:
                pass

        def remove_runtime_directory() -> None:
            try:
                self.socket_dir.rmdir()
            except FileNotFoundError:
                pass

        def remove_runtime_paths() -> None:
            if not clean("socket_unlink", lambda: self.socket_path.unlink(missing_ok=True)):
                return
            if not clean("candidate_cid_unlink", lambda: self.cid_path.unlink(missing_ok=True)):
                return
            if not clean("scoring_mask_remove", remove_scoring_mask):
                return
            clean("runtime_directory_remove", remove_runtime_directory)

        shutdown_error = next(
            (
                error
                for _, error in cleanup_errors
                if isinstance(error, CandidateContainerCleanupError)
            ),
            None,
        )
        self._shutdown_verified = shutdown_error is None
        if shutdown_error is None:
            remove_runtime_paths()
        else:
            clean("socket_unlink", lambda: self.socket_path.unlink(missing_ok=True))
        if cleanup_errors and (exception_type is None or shutdown_error is not None):
            operation, first_error = cleanup_errors[0]
            cause = shutdown_error or first_error
            cause_diagnostic = _cleanup_diagnostic(cause, operation)
            error_type = CandidateContainerCleanupError if shutdown_error else ContainerMetadataCleanupError
            raise error_type(
                "candidate container cleanup failed: "
                f"{cause_diagnostic}; recovery metadata retained"
            ) from cause
        if self._error is not None and (
            exception_type is None or isinstance(self._error, CandidateBrokerInfrastructureError)
        ):
            error = self._error_with_diagnostic(self._error)
            raise error from self._error

    def _cancel_operation(self) -> None:
        assert self._process is not None
        try:
            self._stop_process(self._process,
                               deadline=time.monotonic() + MAX_CANDIDATE_CLEANUP_SECONDS)
        except (CandidateContainerCleanupError, ContainerMetadataCleanupError):
            raise
        except BaseException as exc:
            raise CandidateContainerCleanupError(_cleanup_diagnostic(exc, "operation_cancel")) from exc
        try:
            self.cid_path.unlink(missing_ok=True)
        except OSError as exc:
            raise ContainerMetadataCleanupError(_cleanup_diagnostic(exc, "candidate_cid_unlink")) from exc
        self._process = None
        self._stdout_buffer.clear()
        self._reset_required = True

    def _serve(self) -> None:
        assert self._server is not None
        try:
            while not self._stopping:
                try:
                    connection, _ = self._server.accept()
                except TimeoutError:
                    if self._process is not None and self._process.poll() is not None:
                        raise self._error_with_diagnostic(
                            CandidateBrokerError("candidate container exited before RPC completed")
                        )
                    continue
                try:
                    self.calls += 1
                    if self.calls > self.max_rpc_calls:
                        raise CandidateBrokerError(
                            "candidate RPC call budget exceeded configured limit of "
                            f"{self.max_rpc_calls}"
                        )
                    request = _read_line(connection.makefile("rb"))
                    parsed, operation_timeout = _parse_broker_request(request, self.rpc_timeout_seconds)
                    if operation_timeout is not None:
                        request = json.dumps(parsed, allow_nan=False, separators=(",", ":")).encode()
                    is_reset = parsed["op"] == "reset"
                    timeout = self.reset_timeout_seconds if is_reset else self.rpc_timeout_seconds
                    if operation_timeout is not None:
                        timeout = operation_timeout
                    if self._reset_required and not is_reset:
                        raise CandidateBrokerError("candidate reset required after operation timeout")
                    deadline = time.monotonic() + timeout
                    if is_reset:
                        seed = parsed.get("seed")
                        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                            raise CandidateBrokerError(
                                "candidate reset seed must be a non-negative integer"
                            )
                        reset_index = len(self._reset_seeds)
                        if self._allowed_reset_seeds is not None and (
                            reset_index >= len(self._allowed_reset_seeds)
                            or seed != self._allowed_reset_seeds[reset_index]
                        ):
                            raise CandidateBrokerError(
                                "candidate reset does not match declared sequence "
                                f"at index {reset_index} "
                                f"(allowed count {len(self._allowed_reset_seeds)})"
                            )
                        self._reset_seeds.append(seed)
                        self._restart_process(deadline)
                        self._reset_required = False
                    elif self._process is None:
                        self.calls_before_first_reset += 1
                        self._process = self._start_process()
                    assert self._process is not None
                    assert self._process.stdin is not None
                    assert self._process.stdout is not None
                    try:
                        if operation_timeout is None:
                            self._process.stdin.write(request + b"\n")
                            self._process.stdin.flush()
                        else:
                            _write_operation_request(self._process.stdin, request + b"\n", deadline)
                        response = _read_line_with_timeout(
                            self._process.stdout,
                            deadline,
                            "candidate reset" if is_reset else "candidate RPC response",
                            self._stdout_buffer,
                        )
                    except CandidateOperationDeadlineExceeded:
                        if operation_timeout is None:
                            raise
                        self._cancel_operation()
                        self._send_response(connection, _operation_response("operation_timeout"))
                        continue
                    wire_response = (response if operation_timeout is None else
                                     _operation_response("candidate_response", response))
                    response_sent = self._send_response(connection, wire_response)
                    if is_reset and response_sent and _candidate_response_succeeded(response):
                        assert isinstance(seed, int)
                        with self._state_lock:
                            self._completed_reset_seeds.append(seed)
                            completed_reset_count = len(self._completed_reset_seeds)
                        if (
                            self.completed_reset_limit is not None
                            and completed_reset_count >= self.completed_reset_limit
                        ):
                            return
                except BaseException as exc:
                    # Record before the client connection closes: a client that sees
                    # the closed connection may stop the broker before this thread
                    # reaches the outer handler, and a stopping broker drops errors.
                    if not self._stopping:
                        self._record_error_once(self._error_with_diagnostic(exc))
                    raise
                finally:
                    connection.close()
        except BaseException as exc:
            if not self._stopping:
                self._record_error_once(self._error_with_diagnostic(exc))
        finally:
            self._server.close()

    def _error_with_diagnostic(self, error: BaseException) -> CandidateBrokerError:
        if isinstance(error, (CandidateContainerCleanupError, ContainerMetadataCleanupError)):
            return type(error)(str(error)[:1024])
        if self._process is not None and self._process.poll() is not None:
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=0.2)
        diagnostic = self.candidate_diagnostic
        message = (
            str(error)
            if isinstance(error, CandidateBrokerError)
            else f"unexpected {type(error).__name__}"
        )
        returncode = self._process.returncode if self._process is not None else None
        exit_diagnostic = f"candidate_exit_code={returncode}"
        if returncode in {125, 126, 127} and exit_diagnostic not in message:
            message = f"{message}; {exit_diagnostic}"
        if diagnostic is not None and diagnostic.summary() not in message:
            message = f"{message}; {diagnostic.summary()}"
        if isinstance(error, CandidateBrokerInfrastructureError) or returncode in {125, 126, 127}:
            return CandidateBrokerInfrastructureError(message)
        if not isinstance(error, CandidateBrokerError):
            return CandidateBrokerInfrastructureError(message)
        return CandidateBrokerError(message)

    def _send_response(self, connection: _ResponseConnection, response: bytes) -> bool:
        try:
            connection.sendall(response + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            self.peer_disconnects += 1
            return False
        return True


def _candidate_response_succeeded(response: bytes) -> bool:
    try:
        payload = json.loads(response)
    except (UnicodeError, RecursionError, ValueError):
        return False
    return isinstance(payload, dict) and set(payload) == {"ok", "value"} and payload["ok"] is True


def _read_line(stream: IO[bytes]) -> bytes:
    payload = stream.readline(MAX_RPC_BYTES + 1)
    if not payload or len(payload) > MAX_RPC_BYTES or not payload.endswith(b"\n"):
        raise CandidateBrokerError("candidate RPC message is missing or exceeds 64 MiB")
    return payload[:-1]


def _read_line_with_timeout(
    stream: IO[bytes], deadline: float, label: str, buffered: bytearray
) -> bytes:
    while True:
        newline_index = buffered.find(b"\n")
        if newline_index >= 0:
            payload = bytes(buffered[:newline_index])
            del buffered[: newline_index + 1]
            if len(payload) > MAX_RPC_BYTES:
                raise CandidateBrokerError("candidate RPC message exceeds 64 MiB")
            return payload
        if len(buffered) > MAX_RPC_BYTES:
            raise CandidateBrokerError("candidate RPC message exceeds 64 MiB")
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
            raise CandidateOperationDeadlineExceeded(f"{label} timed out")
        chunk = os.read(stream.fileno(), min(64 * 1024, MAX_RPC_BYTES + 1 - len(buffered)))
        if not chunk:
            raise CandidateBrokerError("candidate RPC message is missing or incomplete")
        buffered.extend(chunk)


def _write_operation_request(stream: IO[bytes], payload: bytes, deadline: float) -> None:
    descriptor = stream.fileno()
    was_blocking = os.get_blocking(descriptor)
    os.set_blocking(descriptor, False)
    remaining_payload = memoryview(payload)
    try:
        while remaining_payload:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [stream], [], remaining)[1]:
                raise CandidateOperationDeadlineExceeded("candidate RPC request write timed out")
            try:
                count = os.write(descriptor, remaining_payload[:65536])
            except BlockingIOError:
                continue
            remaining_payload = remaining_payload[count:]
    finally:
        os.set_blocking(descriptor, was_blocking)


def _remaining_timeout(deadline: float | None, maximum: float) -> float:
    if deadline is None:
        return maximum
    return max(0.001, min(maximum, deadline - time.monotonic()))


def _stderr_category(payload: bytes) -> str:
    normalized = payload.decode("utf-8", errors="ignore").lower()
    categories = (
        ("modulenotfounderror", "module_not_found"),
        ("syntaxerror", "syntax_error"),
        ("memoryerror", "memory_error"),
        ("permissionerror", "permission_error"),
        ("importerror", "import_error"),
        ("timeouterror", "timeout"),
        ("traceback", "runtime_error"),
    )
    for marker, category in categories:
        if marker in normalized:
            return category
    return "stderr_output"
