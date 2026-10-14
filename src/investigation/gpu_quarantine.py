"""Bounded, diagnostic-only provenance for persistent GPU quarantine markers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
from dataclasses import dataclass

MAX_MARKER_BYTES = 4096
LEGACY_MARKER = b"agentic-scientist-gpu-quarantine-v1\n"
_PREFIX = b"agentic-scientist-gpu-quarantine-v2\n"
_RUNTIME_PREFIX = b"agentic-scientist-gpu-quarantine-v3\n"
_RUNTIME_ID = re.compile(r"[a-f0-9]{32}\Z")
_HOSTNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}\Z")


@dataclass(frozen=True)
class GpuQuarantine:
    device: str
    marker_sha256: str | None
    modified_at_ns: int
    format: str
    origin: dict[str, object] | None
    recovery_id: str | None = None

    def fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "device": self.device,
            "marker_sha256": self.marker_sha256,
            "modified_at_ns": self.modified_at_ns,
            "format": self.format,
            "origin": self.origin,
        }
        if self.recovery_id is not None:
            fields["recovery_id"] = self.recovery_id
        return fields


def _identifier(value: object, pattern: re.Pattern[str]) -> str | None:
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def quarantine_marker() -> bytes:
    try:
        host = _identifier(socket.gethostname(), _HOSTNAME)
    except OSError:
        host = None
    payload = {
        "schema_version": 2,
        "reason": "cleanup_not_verified",
        "created_at_ns": time.time_ns(),
        "hostname": host,
        "pid": os.getpid(),
    }
    return _PREFIX + json.dumps(payload, sort_keys=True).encode("ascii") + b"\n"


def runtime_marker(token: str) -> bytes:
    if _RUNTIME_ID.fullmatch(token) is None:
        raise ValueError("Invalid GPU runtime identity")
    payload = json.loads(quarantine_marker()[len(_PREFIX):])
    payload.update(schema_version=3, recovery_id=token)
    return _RUNTIME_PREFIX + json.dumps(payload, sort_keys=True).encode("ascii") + b"\n"


def _origin(data: bytes) -> dict[str, object] | None:
    prefix = _RUNTIME_PREFIX if data.startswith(_RUNTIME_PREFIX) else _PREFIX
    if not data.startswith(prefix):
        return None
    try:
        payload = json.loads(data[len(prefix) :])
    except (ValueError, UnicodeError, RecursionError):
        return None
    schema = 3 if prefix == _RUNTIME_PREFIX else 2
    if not isinstance(payload, dict) or payload.get("schema_version") != schema:
        return None
    if schema == 3 and _identifier(payload.get("recovery_id"), _RUNTIME_ID) is None:
        return None
    if payload.get("reason") != "cleanup_not_verified":
        return None
    for key in ("pid", "created_at_ns"):
        value = payload.get(key)
        if type(value) is not int or not 0 < value < 2**63:
            return None
    return {
        "schema_version": schema,
        "reason": "cleanup_not_verified",
        "created_at_ns": payload["created_at_ns"],
        "pid": payload["pid"],
        "hostname": _identifier(payload.get("hostname"), _HOSTNAME),
    }


def read_quarantine(descriptor: int, device: str) -> GpuQuarantine | None:
    metadata = os.fstat(descriptor)
    data = os.pread(descriptor, MAX_MARKER_BYTES + 1, 0)
    if not data and metadata.st_size == 0:
        return None
    oversized = len(data) > MAX_MARKER_BYTES or metadata.st_size != len(data)
    origin = None if oversized else _origin(data)
    recovery_id = None
    if origin is not None and origin["schema_version"] == 3:
        recovery_id = json.loads(data[len(_RUNTIME_PREFIX):])["recovery_id"]
    if oversized:
        kind = "oversized"
    elif recovery_id:
        kind = "v3"
    elif origin is not None:
        kind = "v2"
    else:
        kind = "legacy" if data == LEGACY_MARKER else "unrecognized"
    return GpuQuarantine(
        device=device,
        marker_sha256=None if oversized else hashlib.sha256(data).hexdigest(),
        modified_at_ns=metadata.st_mtime_ns,
        format=kind,
        origin=origin,
        recovery_id=recovery_id,
    )
