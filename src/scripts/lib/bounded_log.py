#!/usr/bin/env python3
"""Retain a bounded prefix and tail from a byte stream."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.investigation.storage_failures import storage_failure_category
from src.investigation.temp_storage import temporary_directory


READ_CHUNK_BYTES = 64 * 1024
TAIL_BYTES = 64 * 1024
LOG_STORAGE_EXIT_CODE = 4


def _warning(
    operation: str, error: OSError, *, spool: Path | None = None, recovery: str | None = None,
) -> None:
    fields = {
        "status": "infrastructure_failure",
        "failure_category": storage_failure_category(error) or "log_persistence_failed",
        "operation": operation,
    }
    if spool is not None:
        fields["local_spool"] = str(spool)[:512]
    if recovery is not None:
        fields["local_spool_status"] = recovery
    try:
        print(json.dumps(fields, sort_keys=True), file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def _open_log(path: Path) -> BinaryIO:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    return os.fdopen(descriptor, "wb")


class _LogSink:
    def __init__(self, path: Path, name: str) -> None:
        self.name = name
        self.failed = False
        self.opened = False
        self.stream: BinaryIO | None = None
        try:
            self.stream = _open_log(path)
            self.opened = True
        except OSError as error:
            self._fail("open", error)

    def _fail(self, operation: str, error: OSError) -> None:
        self.failed = True
        _warning(f"log_{self.name}_{operation}", error)
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass

    def write(self, prefix: bytes, prefix_bytes: int, tail: bytes, marker: bytes) -> None:
        if self.stream is None:
            return
        try:
            if prefix:
                self.stream.write(prefix)
            if tail:
                self.stream.seek(prefix_bytes)
                self.stream.write(marker)
                self.stream.write(tail)
                self.stream.truncate()
            self.stream.flush()
        except OSError as error:
            self._fail("write", error)

    def close(self) -> None:
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.close()
            except OSError as error:
                self._fail("close", error)


def _report_recovery(local: _LogSink | None, spool: Path | None, phase: str) -> None:
    if local is None or not local.opened:
        state, spool = "unavailable", None
    else:
        state = "partial" if local.failed else phase
    _warning("log_recovery", OSError(), spool=spool, recovery=state)


def retain_bounded_stream(
    output_path: Path,
    max_bytes: int,
    *,
    passthrough: bool = False,
) -> int:
    marker = (
        "\n[agent-log] output truncated after exceeding "
        f"{max_bytes} bytes; retaining prefix and tail.\n"
    ).encode("utf-8")
    if max_bytes <= len(marker):
        raise ValueError("max-bytes is too small for the truncation marker")

    tail_limit = min(TAIL_BYTES, (max_bytes - len(marker)) // 4)
    prefix_limit = max_bytes - len(marker) - tail_limit
    untruncated_remainder_limit = max_bytes - prefix_limit
    prefix_bytes = 0
    remainder = bytearray()
    truncated = False

    spool: Path | None = None
    spool_dir: Path | None = None
    sinks: list[_LogSink] = []
    local: _LogSink | None = None
    failed = False
    try:
        spool_dir = Path(tempfile.mkdtemp(prefix="agentic-log-", dir=temporary_directory()))
        spool = spool_dir / "agent.log"
        local = _LogSink(spool, "local")
        sinks.append(local)
    except (OSError, ValueError) as error:
        failed = True
        failure = error if isinstance(error, OSError) else OSError()
        _warning("log_local_create", failure)
    sinks.append(_LogSink(output_path, "shared"))
    recovery_reported = False

    def report_first_failure() -> None:
        nonlocal recovery_reported
        if not recovery_reported and (failed or any(sink.failed for sink in sinks)):
            _report_recovery(local, spool, "in_progress")
            recovery_reported = True

    report_first_failure()
    try:
        while chunk := sys.stdin.buffer.read(READ_CHUNK_BYTES):
            if passthrough:
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except OSError as error:
                    failed = True
                    passthrough = False
                    _warning("log_passthrough", error)
            prefix_remaining = prefix_limit - prefix_bytes
            prefix = chunk[:prefix_remaining]
            prefix_bytes += len(prefix)
            chunk = chunk[len(prefix):]
            remainder.extend(chunk)
            if len(remainder) > untruncated_remainder_limit:
                truncated = True
            if truncated and len(remainder) > tail_limit:
                del remainder[: len(remainder) - tail_limit]
            for sink in sinks:
                sink.write(prefix, prefix_bytes, bytes(remainder), marker if truncated else b"")
            report_first_failure()
    finally:
        for sink in sinks:
            sink.close()
    failed = failed or any(sink.failed for sink in sinks)
    if failed:
        _report_recovery(local, spool, "retained")
        return LOG_STORAGE_EXIT_CODE
    if spool is not None and spool_dir is not None:
        try:
            spool.unlink()
            spool_dir.rmdir()
        except OSError as error:
            _warning("log_local_cleanup", error, spool=spool)
            return LOG_STORAGE_EXIT_CODE
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-bytes", required=True, type=int)
    parser.add_argument("--passthrough", action="store_true")
    args = parser.parse_args()
    try:
        return retain_bounded_stream(
            args.output,
            args.max_bytes,
            passthrough=args.passthrough,
        )
    except ValueError:
        parser.error("invalid bounded log configuration")
    except OSError as error:
        _warning("log_input", error)
        return LOG_STORAGE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
