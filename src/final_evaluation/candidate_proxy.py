"""Trusted JSON-RPC proxy presented to sealed final evaluators as candidate code."""

from __future__ import annotations

import json
import math
import os
import socket
from typing import Any

_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class CandidateOperationTimeout(TimeoutError):
    """The host broker canceled an operation and verified worker shutdown."""


def handle(request: Any, *, timeout_seconds: float | None = None) -> Any:
    """Forward one JSON-compatible request to the isolated candidate process."""
    if timeout_seconds is not None:
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("operation timeout must be positive and finite")
        if isinstance(request, dict) and request.get("op") == "reset":
            raise ValueError("operation timeout does not apply to reset")
        request = {"__agentic_rpc__": 1, "timeout_seconds": timeout_seconds, "request": request}
    socket_path = os.environ.get("AGENTIC_CANDIDATE_SOCKET", "")
    if not socket_path:
        raise RuntimeError("candidate RPC socket is unavailable")
    payload = json.dumps(request, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError("candidate RPC request exceeds 64 MiB")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        socket_timeout = float(os.environ.get("AGENTIC_CANDIDATE_RPC_TIMEOUT", "21600"))
        if timeout_seconds is not None:
            # Execution stops at the operation deadline; allow bounded verified cleanup to finish.
            socket_timeout = timeout_seconds + 35.0
        client.settimeout(socket_timeout)
        client.connect(socket_path)
        client.sendall(payload)
        response = _read_line(client)
    message = json.loads(response)
    if timeout_seconds is not None:
        if (not isinstance(message, dict)
                or set(message) != {"__agentic_rpc__", "status", "response"}
                or message["__agentic_rpc__"] != 1):
            raise RuntimeError("invalid broker operation response")
        if message["status"] == "operation_timeout" and message["response"] is None:
            raise CandidateOperationTimeout("candidate operation canceled; reset required")
        if message["status"] != "candidate_response" or not isinstance(message["response"], str):
            raise RuntimeError("invalid broker operation response")
        message = json.loads(message["response"])
    if not isinstance(message, dict) or set(message) != {"ok", "value"}:
        raise RuntimeError("invalid candidate RPC response")
    if message["ok"] is not True:
        raise RuntimeError(str(message["value"]))
    return message["value"]


def _read_line(client: socket.socket) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = client.recv(min(65536, _MAX_MESSAGE_BYTES + 1 - size))
        if not chunk:
            raise RuntimeError("candidate RPC response ended early")
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunks.append(chunk[:newline])
            return b"".join(chunks)
        chunks.append(chunk)
        size += len(chunk)
        if size > _MAX_MESSAGE_BYTES:
            raise RuntimeError("candidate RPC response exceeds 64 MiB")
