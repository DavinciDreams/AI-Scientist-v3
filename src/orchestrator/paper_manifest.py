"""Serialized, atomic publication for the stable paper manifest."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def latest_manifest_lock(paper_root: Path) -> Iterator[None]:
    """Serialize readers that update and writers that publish the stable manifest."""
    paper_root.mkdir(parents=True, exist_ok=True)
    lock_path = paper_root / ".latest_manifest.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"paper manifest lock is not a regular file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def paper_publication_lock(paper_root: Path) -> Iterator[None]:
    """Serialize operations across atomic replacement of the paper directory."""
    parent = paper_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{paper_root.name}.publication.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"paper publication lock is not a regular file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def write_manifest_atomically(path: Path, payload: dict[str, object]) -> None:
    """Replace a manifest without exposing partially written JSON."""
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlinked paper manifest: {path}")
    mode = stat.S_IMODE(path.stat().st_mode) if path.is_file() else 0o644
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), mode)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
