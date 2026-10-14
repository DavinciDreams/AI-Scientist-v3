"""Validate private temporary artifacts before caller-owned cleanup."""
from __future__ import annotations

import os
import stat
from pathlib import Path

from .temp_storage import _validate_base


def _temporary_child(path: Path, parent: Path, prefix: str) -> os.stat_result | None:
    if not parent.is_absolute() or parent != parent.resolve(strict=True):
        raise ValueError("temporary cleanup parent must be absolute and normalized")
    _validate_base(parent)
    if not prefix or path.parent != parent or not path.name.startswith(prefix):
        raise ValueError(f"temporary cleanup path is outside the captured scope: {path}")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if metadata.st_uid != os.getuid():
        raise ValueError(f"temporary cleanup artifact must be owned: {path}")
    return metadata


def validate_temp_file(path: Path, parent: Path, prefix: str) -> bool:
    metadata = _temporary_child(path, parent, prefix)
    if metadata is None:
        return False
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError(f"temporary cleanup file must be private: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"temporary cleanup artifact must be a regular file: {path}")
    return True


def validate_temp_directory(path: Path, parent: Path, prefix: str) -> bool:
    metadata = _temporary_child(path, parent, prefix)
    if metadata is None:
        return False
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ValueError(f"temporary cleanup directory must not allow other writers: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"temporary cleanup artifact must be a directory: {path}")
    return True
