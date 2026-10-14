"""Authenticated local RPC for task-drafting validation gates."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import hmac
import json
import math
import os
import re
import socket
import stat
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

from src.investigation.checkpoint_security import redact_sensitive_text


ALLOWED_GATES = frozenset(
    {
        "contract",
        "runtime",
        "search-smoke",
        "final-smoke",
        "production-entry",
        "acceptance-smoke",
    }
)
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_SOCKET_PATH_BYTES = 107
_SMOKE_CATEGORY = re.compile(rb"^AGENTIC_SMOKE_CATEGORY=([a-z0-9_]+)$")
_SMOKE_DIAGNOSTIC = re.compile(rb"^AGENTIC_SMOKE_DIAGNOSTIC=(.+)$")
_MAX_SMOKE_DIAGNOSTIC_BYTES = 4096
_TRANSPORT_CREDENTIAL = re.compile(
    r"(?i)(?:\bauthorization\s*:\s*(?:bearer|basic)\s+\S+|"
    r"\b[a-z][a-z0-9+.-]*://[^\s/@]+@)"
)


class GateRequestError(RuntimeError):
    """Raised for rejected or malformed gate requests."""


@dataclass(frozen=True)
class TaskGateReport:
    schema_version: int
    status: Literal["pass", "authored_failure", "infrastructure_failure"]
    gate: str
    component: str
    stage: str
    category: str
    scaffold_fingerprint: str
    exit_code: int
    elapsed_seconds: float
    diagnostic: str

    @classmethod
    def passed(
        cls,
        gate: str,
        scaffold_fingerprint: str,
        elapsed_seconds: float,
        *,
        diagnostic: str = "",
    ) -> TaskGateReport:
        return cls(
            1,
            "pass",
            gate,
            "harness",
            gate,
            "gate_passed",
            scaffold_fingerprint,
            0,
            elapsed_seconds,
            sanitize_diagnostic(diagnostic),
        )

    @classmethod
    def authored_failure(cls, **values: object) -> TaskGateReport:
        return cls(schema_version=1, status="authored_failure", **values)  # type: ignore[arg-type]

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: object) -> TaskGateReport:
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise GateRequestError("invalid gate report")
        try:
            report = cls(**payload)
        except TypeError as exc:
            raise GateRequestError("invalid gate report") from exc
        if report.schema_version != 1 or report.gate not in ALLOWED_GATES:
            raise GateRequestError("invalid gate report")
        if report.status not in {"pass", "authored_failure", "infrastructure_failure"}:
            raise GateRequestError("invalid gate report")
        expected_exit = {"pass": 0, "authored_failure": 3, "infrastructure_failure": 4}
        text_fields = (report.component, report.stage, report.category, report.diagnostic)
        if (
            type(report.exit_code) is not int
            or report.exit_code != expected_exit[report.status]
            or not isinstance(report.elapsed_seconds, (int, float))
            or isinstance(report.elapsed_seconds, bool)
            or not math.isfinite(report.elapsed_seconds)
            or report.elapsed_seconds < 0
            or not isinstance(report.scaffold_fingerprint, str)
            or not report.scaffold_fingerprint.startswith("sha256:")
            or len(report.scaffold_fingerprint) != 71
            or any(
                character not in "0123456789abcdef"
                for character in report.scaffold_fingerprint.removeprefix("sha256:")
            )
            or any(not isinstance(value, str) for value in text_fields)
            or any(len(value.encode("utf-8")) > MAX_RESPONSE_BYTES for value in text_fields)
        ):
            raise GateRequestError("invalid gate report")
        return report


def sanitize_diagnostic(value: str, *, max_bytes: int = 16 * 1024) -> str:
    normalized = "".join(
        character
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
        else f"\\x{ord(character):02x}"
        for character in value
    )
    if _TRANSPORT_CREDENTIAL.search(normalized):
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        safe = f"<redacted-sensitive-text:{digest}>"
    else:
        safe = redact_sensitive_text(normalized)
    encoded = safe.encode("utf-8")
    if len(encoded) <= max_bytes:
        return safe
    marker = "…".encode()
    return encoded[: max_bytes - len(marker)].decode("utf-8", errors="ignore") + "…"


def parse_smoke_markers(lines: Iterable[bytes]) -> tuple[str | None, str | None]:
    """Extract bounded harness markers from legacy smoke output."""
    category = None
    diagnostic = None
    for raw_line in lines:
        line = raw_line.rstrip(b"\r\n")
        category_match = _SMOKE_CATEGORY.fullmatch(line)
        if category_match:
            category = category_match.group(1).decode("ascii")
        diagnostic_match = _SMOKE_DIAGNOSTIC.fullmatch(line)
        if (
            diagnostic_match is None
            or len(diagnostic_match.group(1)) > _MAX_SMOKE_DIAGNOSTIC_BYTES
        ):
            continue
        try:
            candidate = json.loads(diagnostic_match.group(1))
        except (UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(candidate, str):
            diagnostic = candidate
    return category, diagnostic


def parse_gate_report(lines: Iterable[bytes]) -> TaskGateReport | None:
    """Return the last valid bounded task-gate report in command output."""
    report = None
    for raw_line in lines:
        if len(raw_line) > MAX_RESPONSE_BYTES:
            continue
        try:
            candidate = TaskGateReport.from_payload(json.loads(raw_line))
        except (UnicodeError, json.JSONDecodeError, GateRequestError):
            continue
        report = candidate
    return report


class GateServer:
    """Single-session Unix-socket server with a fixed operation allowlist."""

    def __init__(
        self,
        socket_path: Path,
        *,
        token: str,
        executor: Callable[[str], TaskGateReport],
        max_requests: int,
        receipt_path: Path | None = None,
        ready_path: Path | None = None,
    ) -> None:
        if not token or max_requests < 1:
            raise ValueError("gate server requires a token and positive request limit")
        self._path = socket_path
        self._token = token
        self._executor = executor
        self._max_requests = max_requests
        self._receipt_path = receipt_path
        self._ready_path = ready_path
        self._owned_socket: tuple[int, int, int] | None = None
        self._owned_ready: tuple[int, int, int] | None = None
        self._receipts: list[dict[str, object]] = []
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._server: socket.socket | None = None

    def wait_until_ready(self, timeout: float = 2.0) -> None:
        if not self._ready.wait(timeout):
            raise RuntimeError("gate server did not become ready")

    def close(self) -> None:
        self._closed.set()

    def serve(self) -> None:
        if len(os.fsencode(self._path)) > MAX_SOCKET_PATH_BYTES:
            raise GateRequestError(
                "gate socket path exceeds 107 encoded bytes; use a shorter private socket directory"
            )
        if self._path.exists() or self._path.is_symlink():
            raise GateRequestError("gate socket path already exists")
        handled = 0
        started = False
        try:
            try:
                server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._server = server
                server.bind(str(self._path))
                self._owned_socket = self._identity(self._path.lstat())
                self._path.chmod(0o600)
                server.listen(1)
                server.settimeout(0.1)
                self._publish_ready()
                self._ready.set()
                started = True
            except (OSError, ValueError):
                raise GateRequestError(
                    "gate startup failed; check private socket directory permissions and readiness path"
                ) from None
            while not self._closed.is_set():
                try:
                    client, _ = server.accept()
                except TimeoutError:
                    continue
                with client:
                    try:
                        consumed = self._handle(client, handled)
                    except OSError:
                        consumed = False
                    if consumed:
                        handled += 1
        finally:
            self._ready.clear()
            if self._server is not None:
                self._server.close()
            self._cleanup_endpoint()
            if started:
                self._write_receipts(handled, service_status="closed")

    @staticmethod
    def _identity(metadata: os.stat_result) -> tuple[int, int, int]:
        return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)

    def _publish_ready(self) -> None:
        if self._ready_path is None:
            return
        descriptor = os.open(
            self._ready_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600,
        )
        try:
            self._owned_ready = self._identity(os.fstat(descriptor))
            os.fchmod(descriptor, 0o600)
            if os.write(descriptor, b"ready\n") != len(b"ready\n"):
                raise OSError("incomplete readiness marker")
        finally:
            os.close(descriptor)

    def _cleanup_endpoint(self) -> None:
        cleanup_failed = False
        for path, identity in (
            (self._ready_path, self._owned_ready), (self._path, self._owned_socket),
        ):
            if path is None or identity is None:
                continue
            try:
                if self._identity(path.lstat()) == identity:
                    path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                cleanup_failed = True
        if cleanup_failed:
            raise GateRequestError("gate endpoint cleanup failed; inspect the private socket directory")

    def _handle(self, client: socket.socket, requests_used: int) -> bool:
        stream = client.makefile("rwb")
        request = stream.readline(MAX_REQUEST_BYTES + 1)
        if len(request) > MAX_REQUEST_BYTES or not request.endswith(b"\n"):
            self._write(stream, {"error": "request_too_large"})
            return False
        try:
            payload = json.loads(request)
        except (UnicodeError, json.JSONDecodeError):
            self._write(stream, {"error": "invalid_json"})
            return False
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "token",
            "operation",
        }:
            self._write(stream, {"error": "invalid_request"})
            return False
        if payload["schema_version"] != 1:
            self._write(stream, {"error": "invalid_request"})
            return False
        if not isinstance(payload["token"], str) or not hmac.compare_digest(
            payload["token"], self._token
        ):
            self._write(stream, {"error": "authentication_failed"})
            return False
        operation = payload["operation"]
        if operation not in ALLOWED_GATES:
            self._write(stream, {"error": "operation_not_allowed"})
            return False
        if requests_used >= self._max_requests:
            self._write(
                stream,
                {"error": "request_budget_exhausted", "requests_remaining": 0},
            )
            return False
        try:
            report = self._executor(operation)
        except Exception:
            report = TaskGateReport(
                1,
                "infrastructure_failure",
                operation,
                "harness",
                operation,
                "gate_executor_failure",
                "sha256:" + "0" * 64,
                4,
                0.0,
                "Harness gate execution failed.",
            )
        requests_remaining = self._max_requests - requests_used - 1
        self._receipts.append(
            {
                "category": report.category,
                "operation": operation,
                "requests_remaining": requests_remaining,
                "scaffold_fingerprint": report.scaffold_fingerprint,
                "status": report.status,
            }
        )
        self._write_receipts(requests_used + 1, service_status="running")
        try:
            self._write(
                stream,
                {
                    "report": report.to_payload(),
                    "requests_remaining": requests_remaining,
                },
            )
        except OSError:
            pass
        return True

    def _write_receipts(self, requests_used: int, *, service_status: str) -> None:
        if self._receipt_path is None:
            return
        payload = {
            "schema_version": 1,
            "service_status": service_status,
            "max_requests": self._max_requests,
            "requests_used": requests_used,
            "receipts": self._receipts,
        }
        self._receipt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self._receipt_path.parent / f".{self._receipt_path.name}.{os.getpid()}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, self._receipt_path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _write(stream: object, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > MAX_RESPONSE_BYTES:
            encoded = b'{"error":"response_too_large"}\n'
        stream.write(encoded)  # type: ignore[attr-defined]
        stream.flush()  # type: ignore[attr-defined]


def request_gate(
    socket_path: Path,
    *,
    token: str,
    operation: str,
    timeout_seconds: float = 30.0,
) -> TaskGateReport:
    if operation not in ALLOWED_GATES:
        raise GateRequestError("operation is not allowed")
    request = (
        json.dumps(
            {"schema_version": 1, "token": token, "operation": operation},
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    started = time.monotonic()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout_seconds)
    try:
        client.connect(str(socket_path))
        client.sendall(request)
        response = client.makefile("rb").readline(MAX_RESPONSE_BYTES + 1)
    except OSError as exc:
        raise GateRequestError("gate service unavailable") from exc
    finally:
        client.close()
    if time.monotonic() - started > timeout_seconds or len(response) > MAX_RESPONSE_BYTES:
        raise GateRequestError("gate response exceeded limits")
    try:
        payload = json.loads(response)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GateRequestError("invalid gate response") from exc
    if not isinstance(payload, dict):
        raise GateRequestError("invalid gate response")
    if "error" in payload:
        error = str(payload["error"])
        if error == "authentication_failed":
            raise GateRequestError("gate authentication failed")
        raise GateRequestError(f"gate request rejected: {error}")
    return TaskGateReport.from_payload(payload.get("report"))
