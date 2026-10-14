"""Dependency-free client mounted read-only into task-repair containers."""

from __future__ import annotations

import json
import os
import socket
import sys


ALLOWED_OPERATIONS = {
    "contract",
    "runtime",
    "search-smoke",
    "final-smoke",
    "production-entry",
    "acceptance-smoke",
}
MAX_RESPONSE_BYTES = 64 * 1024


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ALLOWED_OPERATIONS:
        print(
            "usage: task-draft-check contract|runtime|search-smoke|final-smoke|production-entry|acceptance-smoke",
            file=sys.stderr,
        )
        return 2
    socket_path = os.environ.get("TASK_GATE_SOCKET", "")
    token = os.environ.get("TASK_GATE_TOKEN", "")
    if not socket_path or not token:
        print("repair gate is not available", file=sys.stderr)
        return 2
    request = (
        json.dumps(
            {"schema_version": 1, "token": token, "operation": sys.argv[1]},
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        timeout_seconds = int(os.environ.get("TASK_GATE_TIMEOUT_SECONDS", "7200"))
    except ValueError:
        print("TASK_GATE_TIMEOUT_SECONDS must be an integer", file=sys.stderr)
        return 2
    if not 1 <= timeout_seconds <= 7200:
        print("TASK_GATE_TIMEOUT_SECONDS must be from 1 to 7200", file=sys.stderr)
        return 2
    client.settimeout(timeout_seconds)
    try:
        client.connect(socket_path)
        client.sendall(request)
        response = client.makefile("rb").readline(MAX_RESPONSE_BYTES + 1)
    except OSError as exc:
        print(f"repair gate unavailable: {exc}", file=sys.stderr)
        return 4
    finally:
        client.close()
    if len(response) > MAX_RESPONSE_BYTES:
        print("repair gate response exceeded limit", file=sys.stderr)
        return 4
    try:
        payload = json.loads(response)
    except (UnicodeError, json.JSONDecodeError):
        print("repair gate returned invalid JSON", file=sys.stderr)
        return 4
    if not isinstance(payload, dict) or "report" not in payload:
        print(
            f"repair gate rejected request: {payload.get('error', 'invalid response')}",
            file=sys.stderr,
        )
        return 4
    report = payload["report"]
    if not isinstance(report, dict):
        print("repair gate returned invalid report", file=sys.stderr)
        return 4
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    status = report.get("status")
    if status == "pass":
        return 0
    return 3 if status == "authored_failure" else 4


if __name__ == "__main__":
    raise SystemExit(main())
