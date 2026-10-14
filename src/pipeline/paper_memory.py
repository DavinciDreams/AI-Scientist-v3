"""Safe migration of paper memory materialized by older container layouts."""

from __future__ import annotations

import json
import os
import shutil
import stat
import time
from pathlib import Path

from src.investigation.files import ensure_directory_alias, resolve_path


MEMORY_FILES = frozenset({"memory.md", "memory.jsonl", "conclusions_history.jsonl"})
MAX_MEMORY_FILE_BYTES = 64 * 1024 * 1024


def restore_materialized_paper_memory(
    alias_path: str | Path, target_dir: str | Path
) -> Path:
    """Merge allowlisted materialized memory, then restore its canonical alias."""
    alias = Path(alias_path).expanduser()
    target = resolve_path(target_dir)
    if not alias.is_dir() or alias.is_symlink() or alias.resolve() == target:
        return ensure_directory_alias(alias, target)

    backup = alias.parent / f".{alias.name}.materialized.{os.getpid()}.{time.time_ns()}"
    alias.rename(backup)
    try:
        _merge_append_only_memory(backup, target)
        result = ensure_directory_alias(alias, target)
    except BaseException:
        if not alias.exists() and not alias.is_symlink():
            backup.rename(alias)
        raise
    shutil.rmtree(backup)
    return result


def _merge_append_only_memory(source_dir: Path, target_dir: Path) -> None:
    if not target_dir.is_dir() or target_dir.is_symlink():
        raise ValueError(f"Paper memory target must be a real directory: {target_dir}")
    source_fd = os.open(source_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    target_fd = os.open(target_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        source_names = {entry.name for entry in os.scandir(source_fd)}
        unknown_names = source_names - MEMORY_FILES
        if unknown_names:
            rendered = ", ".join(sorted(unknown_names))
            raise ValueError(f"Unexpected materialized paper memory paths: {rendered}")

        writes: list[tuple[str, bytes]] = []
        for name in sorted(source_names):
            source_bytes = _read_owned_regular_file_at(source_fd, name)
            destination_bytes = _read_optional_owned_regular_file_at(target_fd, name)
            merged = _merge_memory_file(name, destination_bytes, source_bytes)
            if merged is not None:
                writes.append((name, merged))

        for name, payload in writes:
            _replace_owned_regular_file_at(target_fd, name, payload)
    finally:
        os.close(target_fd)
        os.close(source_fd)


def _merge_memory_file(
    name: str, canonical: bytes | None, materialized: bytes
) -> bytes | None:
    if canonical is None:
        return materialized
    if canonical == materialized or canonical.startswith(materialized):
        return None
    if materialized.startswith(canonical):
        return materialized
    if name.endswith(".jsonl"):
        return _merge_jsonl(canonical, materialized)
    if name == "memory.md":
        separator = b"\n\n<!-- merged resumed paper memory -->\n\n"
        return canonical.rstrip() + separator + materialized.lstrip()
    raise ValueError(f"Materialized and canonical paper memory diverged: {name}")


def _read_owned_regular_file_at(directory_fd: int, name: str) -> bytes:
    file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    try:
        file_stat = os.fstat(file_fd)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.getuid()
            or file_stat.st_nlink != 1
        ):
            raise ValueError(f"Unsafe materialized paper memory file: {name}")
        if file_stat.st_size > MAX_MEMORY_FILE_BYTES:
            raise ValueError(f"Materialized paper memory exceeds size limit: {name}")
        chunks: list[bytes] = []
        remaining = MAX_MEMORY_FILE_BYTES + 1
        while remaining > 0:
            chunk = os.read(file_fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_MEMORY_FILE_BYTES:
            raise ValueError(f"Materialized paper memory exceeds size limit: {name}")
        return payload
    finally:
        os.close(file_fd)


def _read_optional_owned_regular_file_at(
    directory_fd: int, name: str
) -> bytes | None:
    try:
        return _read_owned_regular_file_at(directory_fd, name)
    except FileNotFoundError:
        return None


def _replace_owned_regular_file_at(directory_fd: int, name: str, payload: bytes) -> None:
    if len(payload) > MAX_MEMORY_FILE_BYTES:
        raise ValueError(f"Merged paper memory exceeds size limit: {name}")
    temporary_name = f".{name}.merge.{os.getpid()}.{time.time_ns()}"
    file_fd = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(file_fd, view)
            view = view[written:]
        os.fsync(file_fd)
    finally:
        os.close(file_fd)
    try:
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _merge_jsonl(canonical: bytes, materialized: bytes) -> bytes:
    merged_lines: list[bytes] = []
    seen: set[str] = set()
    for payload in (canonical, materialized):
        for raw_line in payload.splitlines():
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSONL in materialized paper memory: {exc}") from exc
            identity = json.dumps(record, sort_keys=True, separators=(",", ":"))
            if identity in seen:
                continue
            seen.add(identity)
            merged_lines.append(raw_line)
    return b"\n".join(merged_lines) + (b"\n" if merged_lines else b"")
