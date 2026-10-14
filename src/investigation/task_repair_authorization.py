"""Host-private authorization for dependency failures discovered inside a repair."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any

from src.pipeline.safe_fs import UnsafeGluePathError, open_directory
from src.task_drafting.gate import TaskGateReport, sanitize_diagnostic
from .task_repair_scope import (
    TaskRepairScopeError, _load_snapshot,
    _validate_synchronized_contract_digests, validate_dependency_removal_repair,
    validate_repair_snapshot,
)
from .task_repair_transaction import _load_manifest, _verify_complete_transaction

_MAX_BYTES = 2 * 1024 * 1024
_MAX_AUTHORED_BYTES = 64 * 1024 * 1024 * 1024
_MAX_AUTHORED_ENTRIES = 100_000
_HOST_ENTRIES = {".venv", ".agentic_runtime.json", ".agentic_repair_prompt.md"}
_AUTHORIZED = "Harness authorized dependency removal: unused direct pins may now be removed."
_FIELDS = {
    "schema_version", "task_dir", "session_id", "snapshot_sha256", "transaction_dir",
    "transaction_sha256", "scaffold_fingerprint", "category",
}


def _read_private(path: Path, *, private: bool = False) -> bytes:
    try:
        with open_directory(path.parent, role="repair authorization parent") as parent:
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            descriptor = os.open(path.name, flags, dir_fd=parent)
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_BYTES
                    or metadata.st_uid != os.getuid()
                    or (private and stat.S_IMODE(metadata.st_mode) != 0o600)
                ):
                    raise TaskRepairScopeError("unsafe repair authorization input")
                content = stream.read(_MAX_BYTES + 1)
                if len(content) > _MAX_BYTES:
                    raise TaskRepairScopeError("repair authorization input exceeds size limit")
                return content
    except FileNotFoundError:
        raise
    except (OSError, UnsafeGluePathError) as exc:
        raise TaskRepairScopeError("unsafe repair authorization input") from exc


def _validate_private_parent(path: Path, task_dir: Path) -> None:
    try:
        with open_directory(path.parent, role="repair authorization parent") as parent:
            metadata = os.fstat(parent)
            if (
                metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700
                or path.parent.is_relative_to(task_dir)
            ):
                raise TaskRepairScopeError("repair authorization requires a host-private parent")
    except (OSError, UnsafeGluePathError) as exc:
        raise TaskRepairScopeError("unsafe repair authorization parent") from exc


def _bindings(
    task_dir: Path, snapshot_path: Path, transaction_dir: Path, session_id: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        raise TaskRepairScopeError("invalid repair authorization session")
    snapshot_bytes = _read_private(snapshot_path)
    snapshot = _load_snapshot(snapshot_path)
    if snapshot["task_dir"] != str(task_dir):
        raise TaskRepairScopeError("repair authorization snapshot belongs to another task")
    transaction_bytes = _read_private(transaction_dir / "manifest.json")
    transaction = _load_manifest(transaction_dir)
    if transaction["task_dir"] != str(task_dir):
        raise TaskRepairScopeError("repair authorization transaction belongs to another task")
    _verify_complete_transaction(transaction_dir, transaction)
    return {
        "schema_version": 1, "task_dir": str(task_dir), "session_id": session_id,
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "transaction_dir": str(transaction_dir),
        "transaction_sha256": hashlib.sha256(transaction_bytes).hexdigest(),
        "category": "runtime_dependencies",
    }


def _load_authorization(path: Path, bindings: dict[str, Any]) -> dict[str, Any] | None:
    _validate_private_parent(path, Path(bindings["task_dir"]))
    try:
        content = _read_private(path, private=True)
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TaskRepairScopeError("invalid repair authorization") from exc
    if (
        not isinstance(payload, dict) or set(payload) != _FIELDS
        or any(payload.get(key) != value for key, value in bindings.items())
        or not isinstance(payload.get("scaffold_fingerprint"), str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", payload["scaffold_fingerprint"])
    ):
        raise TaskRepairScopeError("invalid or stale repair authorization")
    return payload


def _publish_authorization(path: Path, payload: dict[str, Any]) -> None:
    """Publish only a complete receipt, refusing to replace any existing entry."""
    content = (json.dumps(payload, sort_keys=True) + "\n").encode()
    temporary = f".{path.name}.{secrets.token_hex(12)}"
    try:
        with open_directory(path.parent, role="repair authorization parent") as parent:
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=parent,
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent,
                        follow_symlinks=False)
                os.fsync(parent)
            finally:
                os.unlink(temporary, dir_fd=parent)
    except (OSError, UnsafeGluePathError) as exc:
        raise TaskRepairScopeError("could not publish repair authorization") from exc


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size,
        metadata.st_mtime_ns, metadata.st_ctime_ns,
    )


def _authored_state(task_dir: Path) -> tuple[dict[str, str], str]:
    """Read a bounded descriptor-pinned tree, using the gate's fingerprint encoding."""
    files: dict[str, str] = {}
    tree_digest = hashlib.sha256()
    entries = 0
    total_bytes = 0

    def visit(directory: int, relative_dir: Path) -> None:
        nonlocal entries, total_bytes
        directory_before = os.fstat(directory)
        with os.scandir(directory) as stream:
            names = []
            for entry in stream:
                entries += 1
                if entries > _MAX_AUTHORED_ENTRIES:
                    raise TaskRepairScopeError("authored repair tree exceeds entry limit")
                names.append(entry.name)
        for name in sorted(names):
            if relative_dir == Path() and name in _HOST_ENTRIES:
                continue
            before = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if not (stat.S_ISDIR(before.st_mode) or stat.S_ISREG(before.st_mode)):
                raise TaskRepairScopeError("unsafe authored repair path")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if stat.S_ISDIR(before.st_mode):
                flags |= os.O_DIRECTORY
            descriptor = os.open(name, flags, dir_fd=directory)
            try:
                if _file_identity(before) != _file_identity(os.fstat(descriptor)):
                    raise TaskRepairScopeError("authored repair path changed during scan")
                relative_path = relative_dir / name
                relative = relative_path.as_posix()
                if stat.S_ISDIR(before.st_mode):
                    tree_digest.update(b"dir\0" + relative.encode() + b"\0")
                    visit(descriptor, relative_path)
                else:
                    if total_bytes + before.st_size > _MAX_AUTHORED_BYTES:
                        raise TaskRepairScopeError("authored repair tree exceeds byte limit")
                    tree_digest.update(
                        b"file\0" + relative.encode() + b"\0" + str(before.st_size).encode() + b"\0"
                    )
                    file_digest = hashlib.sha256()
                    observed = 0
                    while chunk := os.read(descriptor, min(1024 * 1024, before.st_size - observed + 1)):
                        observed += len(chunk)
                        total_bytes += len(chunk)
                        if observed > before.st_size or total_bytes > _MAX_AUTHORED_BYTES:
                            raise TaskRepairScopeError("authored repair file exceeds read limit")
                        tree_digest.update(chunk)
                        file_digest.update(chunk)
                    if observed != before.st_size:
                        raise TaskRepairScopeError("authored repair file changed during scan")
                    files[relative] = file_digest.hexdigest()
                if (
                    _file_identity(before) != _file_identity(os.fstat(descriptor))
                    or _file_identity(before) != _file_identity(
                        os.stat(name, dir_fd=directory, follow_symlinks=False)
                    )
                ):
                    raise TaskRepairScopeError("authored repair path changed during scan")
            finally:
                os.close(descriptor)
        if _file_identity(directory_before) != _file_identity(os.fstat(directory)):
            raise TaskRepairScopeError("authored repair directory changed during scan")
    try:
        with open_directory(task_dir, role="authored repair tree") as root:
            visit(root, Path())
    except (OSError, UnsafeGluePathError) as exc:
        raise TaskRepairScopeError("unsafe authored repair tree") from exc
    return files, "sha256:" + tree_digest.hexdigest()


def _validate_before_authorization(task_dir: Path, snapshot_path: Path) -> str:
    snapshot = _load_snapshot(snapshot_path)
    if snapshot["contract_policy"] != "synchronized-digests":
        raise TaskRepairScopeError("dependency transition requires an original digest repair scope")
    before = snapshot["files"]
    after, fingerprint = _authored_state(task_dir)
    changed = {path for path in before.keys() | after.keys() if before.get(path) != after.get(path)}
    if changed - set(snapshot["mutable_paths"]):
        raise TaskRepairScopeError("out-of-scope changes before dependency authorization")
    if before.get("requirements.txt") != after.get("requirements.txt"):
        raise TaskRepairScopeError("dependencies changed before authorization")
    contract_bytes = _read_private(task_dir / "task_contract.json")
    if hashlib.sha256(contract_bytes).hexdigest() != after.get("task_contract.json"):
        raise TaskRepairScopeError("authored contract changed during authorization")
    contract = json.loads(contract_bytes)
    if not isinstance(contract, dict):
        raise TaskRepairScopeError("invalid authored repair contract")
    _validate_synchronized_contract_digests(
        snapshot["contract_invariants"], contract,
    )
    return fingerprint


def validate_authorized_repair(
    task_dir: Path, snapshot_path: Path, transaction_dir: Path,
    authorization_path: Path, session_id: str,
) -> tuple[str, ...]:
    """Validate original scope, or the recorded host-authorized dependency transition."""
    task_dir = task_dir.resolve(strict=True)
    bindings = _bindings(task_dir, snapshot_path, transaction_dir, session_id)
    authorization = _load_authorization(authorization_path, bindings)
    if authorization is None:
        return validate_repair_snapshot(task_dir, snapshot_path)
    return validate_dependency_removal_repair(task_dir, snapshot_path)


class DependencyRepairExecutor:
    """Authorize only a trusted gate's current, unchanged dependency-failure scaffold."""

    def __init__(
        self, task_dir: Path, snapshot_path: Path, transaction_dir: Path,
        authorization_path: Path, session_id: str,
        executor: Callable[[str], TaskGateReport],
    ) -> None:
        self._task = task_dir.resolve(strict=True)
        self._snapshot = snapshot_path
        self._transaction = transaction_dir
        self._authorization = authorization_path
        self._session = session_id
        self._executor = executor
        self._bindings = _bindings(self._task, snapshot_path, transaction_dir, session_id)
        _load_authorization(authorization_path, self._bindings)

    def __call__(self, operation: str) -> TaskGateReport:
        report = self._executor(operation)
        if report.status != "authored_failure" or report.category != "runtime_dependencies":
            return report
        try:
            current = _bindings(self._task, self._snapshot, self._transaction, self._session)
            if current != self._bindings:
                raise TaskRepairScopeError("repair authorization inputs changed during gate")
            authorization = _load_authorization(self._authorization, current)
            if authorization is None:
                if _load_snapshot(self._snapshot)["contract_policy"] == (
                    "synchronized-dependency-removals"
                ):
                    return report
                fingerprint = _validate_before_authorization(self._task, self._snapshot)
                if fingerprint != report.scaffold_fingerprint:
                    raise TaskRepairScopeError("stale dependency failure fingerprint")
                _publish_authorization(
                    self._authorization,
                    {**current, "scaffold_fingerprint": report.scaffold_fingerprint},
                )
            return replace(report, diagnostic=_AUTHORIZED + "\n" + sanitize_diagnostic(report.diagnostic))
        except (TaskRepairScopeError, OSError, ValueError) as exc:
            return replace(
                report, status="infrastructure_failure", component="harness",
                stage="repair_authorization", category="repair_authorization_failed", exit_code=4,
                diagnostic=sanitize_diagnostic(str(exc)),
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate",))
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--transaction-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    try:
        validate_authorized_repair(
            args.task_dir, args.snapshot, args.transaction_dir, args.authorization, args.session_id,
        )
    except (TaskRepairScopeError, OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
