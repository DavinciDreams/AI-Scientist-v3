"""Standalone paper-workflow gate events for shell and wrapper consumers."""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import math
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, TextIO

try:
    from src.investigation.paper_events import (
        is_paper_heartbeat,
        paper_gate_name,
        paper_gate_qualifier,
    )
    from src.investigation.terminal_ui import SPINNER_FRAMES, colors_enabled, style, terminal_text
except ModuleNotFoundError:  # Direct script execution from outside the repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.investigation.paper_events import (
        is_paper_heartbeat,
        paper_gate_name,
        paper_gate_qualifier,
    )
    from src.investigation.terminal_ui import SPINNER_FRAMES, colors_enabled, style, terminal_text

TRANSPORT_PREFIX = "AGENTIC_SCIENTIST_PAPER_EVENT "
VALID_GATES = {
    "preparation",
    "strategy",
    "draft",
    "references",
    "figures",
    "validation",
    "evidence_audit",
    "review",
    "repair",
    "finalization",
}
VALID_STATES = {"started", "completed", "failed", "skipped"}
_FIELD_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_STATE_GLYPHS = {
    "started": (SPINNER_FRAMES[0], "running"),
    "completed": ("✓", "success"),
    "failed": ("✗", "failure"),
    "skipped": ("–", "detail"),
}
_TRANSPORT_RECORD_KEYS = {
    "timestamp",
    "source",
    "event",
    "session_id",
    "gate",
    "state",
    "fields",
}
_MAX_TRANSPORT_PAYLOAD_CHARS = 64 * 1024
_DIAGNOSTIC_LOG_MAX_BYTES = 16 * 1024 * 1024
_DIAGNOSTIC_LOG_TAIL_BYTES = 64 * 1024
_DIAGNOSTIC_TRUNCATION_MARKER = b"\n...[paper diagnostics truncated]...\n"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_record(
    *,
    gate: str,
    state: str,
    fields: dict[str, Any],
    session_id: str,
    timestamp_fn: Callable[[], str] = utc_timestamp,
) -> dict[str, Any]:
    if gate not in VALID_GATES:
        raise ValueError(f"unknown paper gate: {gate}")
    if state not in VALID_STATES:
        raise ValueError(f"unknown paper gate state: {state}")
    if not session_id.strip():
        raise ValueError("session_id must be non-empty")
    invalid_keys = [key for key in fields if not _FIELD_KEY_RE.fullmatch(key)]
    if invalid_keys:
        raise ValueError(f"invalid paper event field: {invalid_keys[0]}")
    return {
        "timestamp": timestamp_fn(),
        "source": "paper",
        "event": "paper_gate",
        "session_id": session_id,
        "gate": gate,
        "state": state,
        "fields": normalize_value(fields),
    }


def emit_event(
    *,
    gate: str,
    state: str,
    fields: dict[str, Any],
    output_format: str,
    transport: str,
    journal_path: Path,
    session_id: str,
    stream: TextIO,
    verbose: bool = False,
    timestamp_fn: Callable[[], str] = utc_timestamp,
) -> dict[str, Any]:
    if output_format not in {"human", "json"}:
        raise ValueError(f"unknown paper event format: {output_format}")
    if transport not in {"direct", "wrapper"}:
        raise ValueError(f"unknown paper event transport: {transport}")
    record = build_record(
        gate=gate,
        state=state,
        fields=fields,
        session_id=session_id,
        timestamp_fn=timestamp_fn,
    )
    _append_private_jsonl(journal_path, record)
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    if transport == "wrapper":
        stream.write(f"{TRANSPORT_PREFIX}{encoded}\n")
    elif output_format == "json":
        stream.write(f"{encoded}\n")
    else:
        if is_paper_heartbeat(state, record["fields"]):
            return record
        label = paper_gate_name(gate)
        qualifier = paper_gate_qualifier(record["fields"])
        glyph, semantic = _STATE_GLYPHS[state]
        glyph = style(glyph, semantic, enabled=colors_enabled(stream))
        line = f"[PAPER] {glyph} {label} · {state}"
        if qualifier:
            line = f"{line} ({qualifier})"
        elapsed = record["fields"].get("elapsed_seconds")
        if (
            state in {"completed", "failed"}
            and not isinstance(elapsed, bool)
            and isinstance(elapsed, (int, float))
            and math.isfinite(elapsed)
            and elapsed >= 0
        ):
            line = f"{line} · {elapsed:.1f}s"
        if verbose and record["fields"]:
            details = " ".join(
                f"{terminal_text(key, limit=64)}={terminal_text(value, limit=160)}"
                for key, value in record["fields"].items()
            )
            line = f"{line} · {details}"
        diagnostic_log = record["fields"].get("diagnostic_log")
        if state == "failed" and isinstance(diagnostic_log, str) and diagnostic_log:
            line = f"{line} · diagnostics: {terminal_text(diagnostic_log, limit=512)}"
        stream.write(f"{line}\n")
    stream.flush()
    return record


def extract_transport_events(output: str) -> tuple[str, list[dict[str, Any]]]:
    """Remove wrapper sentinels from captured output and return validated records."""
    retained_lines: list[str] = []
    records: list[dict[str, Any]] = []
    for line in output.splitlines(keepends=True):
        if not line.startswith(TRANSPORT_PREFIX):
            retained_lines.append(line)
            continue
        payload = line[len(TRANSPORT_PREFIX) :].rstrip("\r\n")
        if len(payload) > _MAX_TRANSPORT_PAYLOAD_CHARS:
            continue
        try:
            record = json.loads(
                payload,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except (json.JSONDecodeError, RecursionError, ValueError):
            continue
        if _is_transport_record(record):
            records.append(record)
    return "".join(retained_lines), records


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _is_transport_record(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != _TRANSPORT_RECORD_KEYS:
        return False
    timestamp = value.get("timestamp")
    session_id = value.get("session_id")
    fields = value.get("fields")
    return (
        value.get("source") == "paper"
        and value.get("event") == "paper_gate"
        and isinstance(timestamp, str)
        and 0 < len(timestamp) <= 64
        and isinstance(session_id, str)
        and 0 < len(session_id) <= 160
        and value.get("gate") in VALID_GATES
        and value.get("state") in VALID_STATES
        and isinstance(fields, dict)
        and len(fields) <= 64
        and all(isinstance(key, str) and _FIELD_KEY_RE.fullmatch(key) for key in fields)
    )


def _open_directory_no_follow(directory: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    expanded = directory.expanduser()
    if expanded.is_absolute():
        descriptor = os.open(expanded.anchor, flags)
        components = expanded.parts[1:]
    else:
        descriptor = os.open(".", flags)
        components = expanded.parts
    try:
        for component in components:
            if component in {"", "."}:
                continue
            if component == "..":
                raise OSError(errno.EPERM, "paper event journal path must not contain '..'")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _append_private_jsonl(path: Path, record: dict[str, Any]) -> None:
    directory_fd = _open_directory_no_follow(path.parent)
    flags = (
        os.O_APPEND
        | os.O_CREAT
        | os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError(errno.EINVAL, "paper event journal must be a regular file")
        os.fchmod(descriptor, 0o600)
        line = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        size = os.fstat(descriptor).st_size
        if size and os.pread(descriptor, 1, size - 1) != b"\n":
            os.write(descriptor, b"\n")
        view = memoryview(line)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(errno.EIO, "failed to append paper event")
            view = view[written:]
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def stream_private_diagnostics(path: Path, source: BinaryIO) -> None:
    """Drain diagnostics into a bounded private file without following symlinks."""
    directory_fd = _open_directory_no_follow(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError(errno.EINVAL, "paper diagnostic log must be a regular file")
        os.fchmod(descriptor, 0o600)
        max_bytes = _DIAGNOSTIC_LOG_MAX_BYTES
        tail_bytes = _DIAGNOSTIC_LOG_TAIL_BYTES
        retained_limit = len(_DIAGNOSTIC_TRUNCATION_MARKER) + tail_bytes
        prefix_limit = max_bytes - retained_limit
        if prefix_limit < 0:
            raise ValueError("paper diagnostic limit is smaller than its truncation metadata")
        prefix_written = 0
        overflow_size = 0
        retained = bytearray()
        while chunk := source.read(64 * 1024):
            prefix = chunk[: max(0, prefix_limit - prefix_written)]
            if prefix:
                _write_all(descriptor, prefix)
                prefix_written += len(prefix)
            overflow = chunk[len(prefix) :]
            if overflow:
                overflow_size += len(overflow)
                retained.extend(overflow)
                if len(retained) > retained_limit:
                    del retained[:-retained_limit]
        if overflow_size <= retained_limit:
            _write_all(descriptor, retained)
        else:
            _write_all(descriptor, _DIAGNOSTIC_TRUNCATION_MARKER)
            _write_all(descriptor, retained[-tail_bytes:])
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes | bytearray) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError(errno.EIO, "failed to write paper diagnostics")
        view = view[written:]


def _field_value(raw: str) -> Any:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(value, (dict, list)):
        return raw
    return value


def _fields(raw_fields: list[str]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for raw in raw_fields:
        key, separator, value = raw.partition("=")
        if not separator:
            raise ValueError(f"paper event field must use key=value: {raw}")
        if key in fields:
            raise ValueError(f"duplicate paper event field: {key}")
        fields[key] = _field_value(value)
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-sink", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--format", choices=("human", "json"), default="human")
    parser.add_argument("--transport", choices=("direct", "wrapper"), default="direct")
    parser.add_argument("--session-id")
    parser.add_argument("--gate", choices=sorted(VALID_GATES))
    parser.add_argument("--state", choices=sorted(VALID_STATES))
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.diagnostic_sink is not None:
            if any((args.journal, args.session_id, args.gate, args.state, args.field)):
                raise ValueError("diagnostic sink does not accept event arguments")
            stream_private_diagnostics(args.diagnostic_sink, sys.stdin.buffer)
            return 0
        if not all((args.journal, args.session_id, args.gate, args.state)):
            raise ValueError("event emission requires journal, session, gate, and state")
        fields = _fields(args.field)
        emit_event(
            gate=args.gate,
            state=args.state,
            fields=fields,
            output_format=args.format,
            transport=args.transport,
            journal_path=args.journal,
            session_id=args.session_id,
            stream=sys.stdout,
            verbose=args.verbose,
        )
    except (OSError, ValueError) as exc:
        print(f"paper event bridge failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
