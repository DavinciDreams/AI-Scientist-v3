"""Rollback support for dependency-removal scaffold repairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .task_repair_scope import TaskRepairScopeError, _load_snapshot, _sha256
from .task_runtime import discard_managed_task_runtime

_SCHEMA_VERSION = 4
_MAX_MANIFEST_BYTES = 128 * 1024


def _relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TaskRepairScopeError("invalid repair transaction path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise TaskRepairScopeError("invalid repair transaction path")
    return Path(*relative.parts)


def _safe_target(task_dir: Path, relative: Path) -> Path:
    current = task_dir
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink() or not current.is_dir():
            raise TaskRepairScopeError("unsafe repair transaction target")
    target = task_dir / relative
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise TaskRepairScopeError("unsafe repair transaction target")
    return target


def _safe_regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TaskRepairScopeError(f"missing {label}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise TaskRepairScopeError(f"unsafe {label}")


def _atomic_copy(source: Path, destination: Path, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(mode)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(content) > _MAX_MANIFEST_BYTES:
        raise TaskRepairScopeError("repair transaction manifest is too large")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".manifest.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _transaction_state_paths(transaction_dir: Path) -> tuple[Path, Path, Path]:
    parent = transaction_dir.parent
    if (
        transaction_dir.name in {"", ".", ".."}
        or parent.is_symlink()
        or not parent.is_dir()
        or parent.resolve(strict=True) != parent
    ):
        raise TaskRepairScopeError("unsafe repair transaction path")
    return (
        parent / f".{transaction_dir.name}.creating",
        parent / f".{transaction_dir.name}.committing",
        parent / f".{transaction_dir.name}.committed.json",
    )


def _discard_unpublished_transaction(directory: Path) -> None:
    if not directory.exists() and not directory.is_symlink():
        return
    if directory.is_symlink() or not directory.is_dir():
        raise TaskRepairScopeError("unsafe partial repair transaction")
    entries = list(directory.rglob("*"))
    if any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in entries):
        raise TaskRepairScopeError("unsafe partial repair transaction")
    for path in (path for path in entries if path.is_file()):
        path.unlink()
    for path in sorted(
        (path for path in entries if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.rmdir()
    directory.rmdir()


def _capture_transaction(
    task_dir: Path,
    snapshot: dict[str, Any],
    snapshot_sha256: str,
    transaction_dir: Path,
    *,
    tree_snapshot: bool = False,
) -> None:
    transaction_dir.mkdir(mode=0o700, parents=False)
    directory_paths: set[Path] = set()
    for value in snapshot["mutable_paths"]:
        parent = _relative_path(value).parent
        while parent != Path("."):
            directory_paths.add(parent)
            parent = parent.parent
    if tree_snapshot:
        directory_paths.update(Path(value) for value in snapshot["tree_directories"])
    directories: list[dict[str, object]] = []
    for relative in sorted(directory_paths, key=lambda item: (len(item.parts), item.as_posix())):
        directory = task_dir / relative
        if directory.is_symlink() or not directory.is_dir():
            raise TaskRepairScopeError("unsafe repair transaction source directory")
        directories.append(
            {
                "path": relative.as_posix(),
                "mode": stat.S_IMODE(directory.stat().st_mode),
            }
        )
    records: list[dict[str, object]] = []
    for value in snapshot["mutable_paths"]:
        relative = _relative_path(value)
        target = _safe_target(task_dir, relative)
        expected_digest = snapshot["files"].get(value)
        if expected_digest is None:
            if target.exists():
                raise TaskRepairScopeError("repair transaction source changed before capture")
            records.append({"path": value, "existed": False})
            continue
        _safe_regular_file(target, "repair transaction source")
        if _sha256(target) != expected_digest:
            raise TaskRepairScopeError("repair transaction source changed before capture")
        mode = stat.S_IMODE(target.stat().st_mode)
        backup = transaction_dir / "files" / relative
        _atomic_copy(target, backup, mode)
        if _sha256(backup) != expected_digest:
            raise TaskRepairScopeError("repair transaction backup verification failed")
        records.append(
            {
                "path": value,
                "existed": True,
                "mode": mode,
                "sha256": expected_digest,
            }
        )
    _write_manifest(
        transaction_dir / "manifest.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "task_dir": str(task_dir),
            "snapshot_sha256": snapshot_sha256,
            "tree_snapshot": tree_snapshot,
            "directories": directories,
            "records": records,
        },
    )


def create_repair_transaction(task_dir: Path, snapshot_path: Path, transaction_dir: Path) -> None:
    """Capture every scoped mutable file before a repair starts."""
    task_dir = task_dir.resolve(strict=True)
    snapshot = _load_snapshot(snapshot_path)
    if snapshot["task_dir"] != str(task_dir):
        raise TaskRepairScopeError("repair transaction snapshot belongs to another task")
    snapshot_sha256 = _sha256(snapshot_path)
    creating_dir, committing_dir, committed_marker = _transaction_state_paths(transaction_dir)
    if committed_marker.exists() or committing_dir.exists():
        raise TaskRepairScopeError("repair transaction was already committed")
    if transaction_dir.exists() or transaction_dir.is_symlink():
        payload = _load_manifest(transaction_dir)
        if payload["task_dir"] != str(task_dir) or payload["snapshot_sha256"] != snapshot_sha256:
            raise TaskRepairScopeError("repair transaction belongs to another snapshot")
        _verify_complete_transaction(transaction_dir, payload)
        return
    _discard_unpublished_transaction(creating_dir)
    try:
        _capture_transaction(task_dir, snapshot, snapshot_sha256, creating_dir)
    except Exception:
        _discard_unpublished_transaction(creating_dir)
        raise
    os.replace(creating_dir, transaction_dir)


def _tree_directories(task_dir: Path) -> tuple[str, ...]:
    directories: list[str] = []
    for path in sorted(task_dir.rglob("*")):
        relative = path.relative_to(task_dir).as_posix()
        if path.is_symlink():
            raise TaskRepairScopeError(f"repair transaction tree contains symlink: {relative}")
        if path.is_dir():
            directories.append(relative)
        elif not path.is_file():
            raise TaskRepairScopeError(f"repair transaction tree contains special path: {relative}")
    return tuple(directories)


def _tree_files(task_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(task_dir.rglob("*")):
        relative = path.relative_to(task_dir).as_posix()
        if path.is_symlink():
            raise TaskRepairScopeError(f"repair transaction tree contains symlink: {relative}")
        if path.is_file():
            files[relative] = _sha256(path)
        elif not path.is_dir():
            raise TaskRepairScopeError(f"repair transaction tree contains special path: {relative}")
    return files


def create_tree_repair_transaction(task_dir: Path, transaction_dir: Path) -> None:
    """Capture the complete safe scaffold tree before a provisional repair."""
    task_dir = task_dir.resolve(strict=True)
    files = _tree_files(task_dir)
    snapshot = {
        "files": files,
        "mutable_paths": sorted(files),
        "tree_directories": list(_tree_directories(task_dir)),
    }
    snapshot_sha256 = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    creating_dir, committing_dir, committed_marker = _transaction_state_paths(transaction_dir)
    if committed_marker.exists() or committing_dir.exists():
        raise TaskRepairScopeError("repair transaction was already committed")
    if transaction_dir.exists() or transaction_dir.is_symlink():
        payload = _load_manifest(transaction_dir)
        if (
            payload["task_dir"] != str(task_dir)
            or payload["snapshot_sha256"] != snapshot_sha256
            or not payload["tree_snapshot"]
        ):
            raise TaskRepairScopeError("repair transaction belongs to another snapshot")
        _verify_complete_transaction(transaction_dir, payload)
        return
    _discard_unpublished_transaction(creating_dir)
    try:
        _capture_transaction(
            task_dir,
            snapshot,
            snapshot_sha256,
            creating_dir,
            tree_snapshot=True,
        )
    except Exception:
        _discard_unpublished_transaction(creating_dir)
        raise
    os.replace(creating_dir, transaction_dir)


def create_authored_tree_repair_transaction(task_dir: Path, transaction_dir: Path) -> None:
    """Discard harness runtime state, then capture the complete authored scaffold tree."""
    task_dir = task_dir.resolve(strict=True)
    try:
        discard_managed_task_runtime(task_dir)
    except ValueError as exc:
        raise TaskRepairScopeError("unsafe managed runtime before repair transaction") from exc
    create_tree_repair_transaction(task_dir, transaction_dir)


def _load_manifest(transaction_dir: Path) -> dict[str, Any]:
    if transaction_dir.is_symlink() or not transaction_dir.is_dir():
        raise TaskRepairScopeError("unsafe repair transaction directory")
    path = transaction_dir / "manifest.json"
    _safe_regular_file(path, "repair transaction manifest")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise TaskRepairScopeError("repair transaction manifest is too large")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskRepairScopeError("invalid repair transaction manifest") from exc
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema_version",
            "task_dir",
            "snapshot_sha256",
            "tree_snapshot",
            "directories",
            "records",
        }
        or payload.get("schema_version") != _SCHEMA_VERSION
        or not isinstance(payload.get("task_dir"), str)
        or not isinstance(payload.get("snapshot_sha256"), str)
        or len(payload["snapshot_sha256"]) != 64
        or not isinstance(payload.get("directories"), list)
        or not isinstance(payload.get("records"), list)
        or not isinstance(payload.get("tree_snapshot"), bool)
    ):
        raise TaskRepairScopeError("invalid repair transaction manifest")
    return payload


def _validated_records(payload: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    validated: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    for record in payload["records"]:
        if not isinstance(record, dict) or not isinstance(record.get("existed"), bool):
            raise TaskRepairScopeError("invalid repair transaction record")
        expected_fields = (
            {"path", "existed", "mode", "sha256"} if record["existed"] else {"path", "existed"}
        )
        if set(record) != expected_fields:
            raise TaskRepairScopeError("invalid repair transaction record")
        relative = _relative_path(record.get("path"))
        if relative in seen:
            raise TaskRepairScopeError("duplicate repair transaction path")
        seen.add(relative)
        if record["existed"] and (
            not isinstance(record.get("mode"), int)
            or isinstance(record.get("mode"), bool)
            or not 0 <= record["mode"] <= 0o777
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise TaskRepairScopeError("invalid repair transaction record")
        validated.append((relative, record))
    return validated


def _validated_directories(payload: dict[str, Any]) -> list[tuple[Path, int]]:
    validated: list[tuple[Path, int]] = []
    seen: set[Path] = set()
    for record in payload["directories"]:
        if not isinstance(record, dict) or set(record) != {"path", "mode"}:
            raise TaskRepairScopeError("invalid repair transaction directory record")
        relative = _relative_path(record.get("path"))
        mode = record.get("mode")
        if (
            relative in seen
            or not isinstance(mode, int)
            or isinstance(mode, bool)
            or not 0 <= mode <= 0o777
        ):
            raise TaskRepairScopeError("invalid repair transaction directory record")
        seen.add(relative)
        validated.append((relative, mode))
    if validated != sorted(validated, key=lambda item: (len(item[0].parts), item[0].as_posix())):
        raise TaskRepairScopeError("repair transaction directories must be ordered")
    return validated


def _expected_transaction_files(
    records: list[tuple[Path, dict[str, Any]]],
) -> set[Path]:
    return {Path("manifest.json")} | {
        Path("files") / relative for relative, record in records if record["existed"]
    }


def _verify_complete_transaction(transaction_dir: Path, payload: dict[str, Any]) -> None:
    _validated_directories(payload)
    records = _validated_records(payload)
    expected_files = _expected_transaction_files(records)
    entries = list(transaction_dir.rglob("*"))
    if any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in entries):
        raise TaskRepairScopeError("unexpected repair transaction content")
    actual_files = {path.relative_to(transaction_dir) for path in entries if path.is_file()}
    if actual_files != expected_files:
        raise TaskRepairScopeError("unexpected repair transaction content")
    for relative, record in records:
        if not record["existed"]:
            continue
        backup = transaction_dir / "files" / relative
        _safe_regular_file(backup, "repair transaction backup")
        if _sha256(backup) != record["sha256"]:
            raise TaskRepairScopeError("repair transaction backup digest mismatch")


def restore_repair_transaction(task_dir: Path, transaction_dir: Path) -> None:
    """Restore the pre-repair mutable file set exactly; safe to replay."""
    task_dir = task_dir.resolve(strict=True)
    payload = _load_manifest(transaction_dir)
    if payload["task_dir"] != str(task_dir):
        raise TaskRepairScopeError("repair transaction belongs to another task")
    if payload["tree_snapshot"]:
        _remove_untracked_tree_entries(task_dir, payload)
    for relative, mode in _validated_directories(payload):
        directory = task_dir / relative
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise TaskRepairScopeError("unsafe repair transaction target directory")
        if not directory.exists():
            directory.mkdir(mode=mode)
        directory.chmod(mode)
    records = _validated_records(payload)
    for relative, record in records:
        target = _safe_target(task_dir, relative)
        if record["existed"]:
            backup = transaction_dir / "files" / relative
            _safe_regular_file(backup, "repair transaction backup")
            if _sha256(backup) != record["sha256"]:
                raise TaskRepairScopeError("repair transaction backup digest mismatch")
            _atomic_copy(backup, target, record["mode"])
        elif target.exists():
            target.unlink()
    for relative, record in records:
        target = _safe_target(task_dir, relative)
        if record["existed"]:
            if _sha256(target) != record["sha256"]:
                raise TaskRepairScopeError("repair transaction restore verification failed")
        elif target.exists():
            raise TaskRepairScopeError("repair transaction restore verification failed")


def _remove_untracked_tree_entries(task_dir: Path, payload: dict[str, Any]) -> None:
    expected_files = {relative for relative, _ in _validated_records(payload)}
    expected_directories = {relative for relative, _ in _validated_directories(payload)}

    def clean(directory: Path, relative_parent: Path) -> None:
        with os.scandir(directory) as entries:
            current_entries = list(entries)
        for entry in current_entries:
            relative = relative_parent / entry.name
            path = Path(entry.path)
            if entry.is_symlink():
                path.unlink()
            elif entry.is_dir(follow_symlinks=False):
                clean(path, relative)
                if relative not in expected_directories:
                    path.rmdir()
            elif not entry.is_file(follow_symlinks=False) or relative not in expected_files:
                path.unlink()

    clean(task_dir, Path())


def _commit_marker_path(transaction_dir: Path) -> tuple[Path, Path]:
    _, committing_dir, committed_marker = _transaction_state_paths(transaction_dir)
    return committing_dir, committed_marker


def _load_commit_marker(path: Path, task_dir: Path) -> None:
    _safe_regular_file(path, "repair transaction commit marker")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise TaskRepairScopeError("repair transaction commit marker is too large")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskRepairScopeError("invalid repair transaction commit marker") from exc
    if payload != {"schema_version": 1, "task_dir": str(task_dir)}:
        raise TaskRepairScopeError("invalid repair transaction commit marker")


def _discard_committed_transaction(transaction_dir: Path) -> None:
    """Resume-safe cleanup for an atomically committed rollback directory."""
    if not transaction_dir.exists() and not transaction_dir.is_symlink():
        return
    if transaction_dir.is_symlink() or not transaction_dir.is_dir():
        raise TaskRepairScopeError("unsafe committed repair transaction")
    manifest_path = transaction_dir / "manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        payload = _load_manifest(transaction_dir)
        _validated_directories(payload)
        records = _validated_records(payload)
        expected_files = _expected_transaction_files(records)
        entries = list(transaction_dir.rglob("*"))
        if any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in entries):
            raise TaskRepairScopeError("unexpected committed repair transaction content")
        actual_files = {path.relative_to(transaction_dir) for path in entries if path.is_file()}
        if not actual_files.issubset(expected_files) or Path("manifest.json") not in actual_files:
            raise TaskRepairScopeError("unexpected committed repair transaction content")
        for relative in sorted(
            expected_files - {Path("manifest.json")},
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            path = transaction_dir / relative
            if path.exists():
                _safe_regular_file(path, "committed repair transaction backup")
                path.unlink()
        manifest_path.unlink()
    remaining = list(transaction_dir.rglob("*"))
    if any(not path.is_dir() or path.is_symlink() for path in remaining):
        raise TaskRepairScopeError("unexpected committed repair transaction content")
    for directory in sorted(
        remaining,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.rmdir()
    transaction_dir.rmdir()


def commit_repair_transaction(task_dir: Path, transaction_dir: Path) -> None:
    """Atomically commit a repair and resume-safe cleanup of its rollback copy."""
    task_dir = task_dir.resolve(strict=True)
    committing_dir, committed_marker = _commit_marker_path(transaction_dir)
    if committed_marker.exists() or committed_marker.is_symlink():
        _load_commit_marker(committed_marker, task_dir)
        if transaction_dir.exists() or transaction_dir.is_symlink():
            raise TaskRepairScopeError("repair transaction exists after commit marker")
        _discard_committed_transaction(committing_dir)
        return
    if transaction_dir.exists() or transaction_dir.is_symlink():
        payload = _load_manifest(transaction_dir)
        if payload["task_dir"] != str(task_dir):
            raise TaskRepairScopeError("repair transaction belongs to another task")
        _verify_complete_transaction(transaction_dir, payload)
        if committing_dir.exists() or committing_dir.is_symlink():
            raise TaskRepairScopeError("ambiguous repair transaction commit state")
        os.replace(transaction_dir, committing_dir)
    elif committing_dir.exists() or committing_dir.is_symlink():
        payload = _load_manifest(committing_dir)
        if payload["task_dir"] != str(task_dir):
            raise TaskRepairScopeError("repair transaction belongs to another task")
        _verify_complete_transaction(committing_dir, payload)
    else:
        raise TaskRepairScopeError("missing repair transaction commit state")
    _write_manifest(
        committed_marker,
        {"schema_version": 1, "task_dir": str(task_dir)},
    )
    _discard_committed_transaction(committing_dir)


def settle_repair_transaction(task_dir: Path, transaction_dir: Path) -> str:
    """Finish a linearized commit or roll back an uncommitted repair during cleanup."""
    task_dir = task_dir.resolve(strict=True)
    committing_dir, committed_marker = _commit_marker_path(transaction_dir)
    if (
        committed_marker.exists()
        or committed_marker.is_symlink()
        or committing_dir.exists()
        or committing_dir.is_symlink()
    ):
        commit_repair_transaction(task_dir, transaction_dir)
        return "committed"
    if not transaction_dir.exists() and not transaction_dir.is_symlink():
        raise TaskRepairScopeError("missing repair transaction cleanup state")
    try:
        discard_managed_task_runtime(task_dir)
    except ValueError as exc:
        raise TaskRepairScopeError("unsafe managed runtime during repair cleanup") from exc
    restore_repair_transaction(task_dir, transaction_dir)
    commit_repair_transaction(task_dir, transaction_dir)
    return "rolled_back"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "create",
        "create-tree",
        "create-authored-tree",
        "restore",
        "commit",
        "settle",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--task-dir", type=Path, required=True)
        subparser.add_argument("--transaction-dir", type=Path, required=True)
        if command == "create":
            subparser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            create_repair_transaction(args.task_dir, args.snapshot, args.transaction_dir)
        elif args.command == "create-tree":
            create_tree_repair_transaction(args.task_dir, args.transaction_dir)
        elif args.command == "create-authored-tree":
            create_authored_tree_repair_transaction(args.task_dir, args.transaction_dir)
        elif args.command == "restore":
            restore_repair_transaction(args.task_dir, args.transaction_dir)
        elif args.command == "settle":
            print(settle_repair_transaction(args.task_dir, args.transaction_dir))
        else:
            commit_repair_transaction(args.task_dir, args.transaction_dir)
    except TaskRepairScopeError as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
