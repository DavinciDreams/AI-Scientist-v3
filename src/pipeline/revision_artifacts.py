"""Bounded regular-file and JSON helpers for revision publication."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from src.orchestrator.paper_manifest import write_manifest_atomically


def read_regular(path: Path, *, max_bytes: int) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError as exc:
        raise ValueError(f"missing regular file: {path}") from exc
    except OSError as exc:
        raise ValueError(f"expected regular file: {path}") from exc
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"expected regular file: {path}")
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"file is too large: {path}")
    return content


def reject_symlink_components(path: Path) -> None:
    for component in reversed((path, *path.parents)):
        if component == Path(component.anchor):
            continue
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"path contains a symlink: {component}")


def atomic_write_file(target: Path, content: bytes) -> None:
    reject_symlink_components(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError(f"refusing to replace symlinked artifact: {target}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, payload: dict[str, object]) -> None:
    write_manifest_atomically(path, payload)


def read_json(path: Path, *, max_bytes: int) -> dict[str, Any]:
    try:
        payload = json.loads(read_regular(path, max_bytes=max_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
