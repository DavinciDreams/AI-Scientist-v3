"""Bounded metadata I/O and owned-directory helpers for portable storage."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Mapping

from src.pipeline.portable_storage_paths import ArchiveError

METADATA_MAX_BYTES = 16 * 1024


def ensure_real_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    validate_owned_directory(path)


def validate_owned_directory(path: Path, *, require_private: bool = False) -> None:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ArchiveError(f"storage path is not an owned real directory: {path}")
    if require_private and metadata.st_mode & 0o077:
        raise ArchiveError(f"storage path is not owner-private: {path}")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_small_json(path: Path) -> object:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ArchiveError(f"archive metadata is unreadable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_size > METADATA_MAX_BYTES
        ):
            raise ArchiveError(f"archive metadata is unsafe: {path}")
        payload = os.read(descriptor, METADATA_MAX_BYTES + 1)
        after = os.fstat(descriptor)
        if (
            len(payload) > METADATA_MAX_BYTES
            or len(payload) != after.st_size
            or (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
        ):
            raise ArchiveError(f"archive metadata changed while reading: {path}")
        return json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"archive metadata is unreadable: {path}") from exc
    finally:
        os.close(descriptor)


def write_json_file(path: Path, payload: Mapping[str, object]) -> None:
    write_text_file(path, json.dumps(payload, sort_keys=True) + "\n")


def write_text_file(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def tree_size(root: Path) -> int:
    if not root.is_dir():
        return 0
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        for entry in os.scandir(current):
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
    return total
