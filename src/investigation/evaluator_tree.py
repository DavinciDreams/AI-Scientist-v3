"""Bounded digests for host-sealed evaluator source trees."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

MANAGED_RUNTIME_MANIFEST_FILENAME = ".agentic_runtime.json"
TASK_SNAPSHOT_METADATA_FILENAME = ".agentic_task_snapshot.json"

_MAX_EVALUATOR_TREE_BYTES = 512 * 1024 * 1024
_MAX_EVALUATOR_TREE_ENTRIES = 100_000
_EVALUATOR_TREE_MASKED_DIRS = {
    ".git",
    ".venv",
    "results",
}
_EVALUATOR_TREE_OMITTED_DIRS = {"__pycache__"}
_MANAGED_RUNTIME_OMITTED_ROOT_ENTRIES = {
    ".venv",
    MANAGED_RUNTIME_MANIFEST_FILENAME,
    ".agentic_resource_manifest.json",
    ".agentic_task_ignore",
    ".agentic_task_publication.json",
    ".agentic_task_qualification.json",
    TASK_SNAPSHOT_METADATA_FILENAME,
    ".agentic_task_validation.json",
    "resources",
}


def task_snapshot_metadata_present(root: Path) -> bool:
    """Return whether pipeline snapshot metadata exists, rejecting unsafe entry types."""
    try:
        metadata = (root / TASK_SNAPSHOT_METADATA_FILENAME).lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("task snapshot metadata must be a regular file")
    return True


def evaluator_tree_digest(root: Path) -> str:
    """Hash bounded evaluator files while rejecting source-tree symlinks."""
    return _evaluator_tree_digest(root, omitted_root_entries=frozenset())


def managed_evaluator_tree_digest(root: Path) -> str:
    """Hash evaluator inputs without task-local managed-runtime artifacts."""
    return _evaluator_tree_digest(
        root,
        omitted_root_entries=frozenset(_MANAGED_RUNTIME_OMITTED_ROOT_ENTRIES),
    )


def _evaluator_tree_digest(
    root: Path,
    *,
    omitted_root_entries: frozenset[str],
) -> str:
    digest = hashlib.sha256()
    entry_count = 0
    total_bytes = 0

    def update_record(record_type: bytes, relative: str, payload: bytes) -> None:
        path_bytes = relative.encode()
        digest.update(record_type)
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def visit(directory_fd: int, prefix: str) -> None:
        nonlocal entry_count, total_bytes
        update_record(b"D", prefix, b"")
        for name in sorted(os.listdir(directory_fd)):
            if not prefix and name in omitted_root_entries:
                continue
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if name in _EVALUATOR_TREE_OMITTED_DIRS and stat.S_ISDIR(metadata.st_mode):
                continue
            if not prefix and name in _EVALUATOR_TREE_MASKED_DIRS and stat.S_ISDIR(
                metadata.st_mode
            ):
                update_record(b"I", name, b"masked")
                continue
            entry_count += 1
            if entry_count > _MAX_EVALUATOR_TREE_ENTRIES:
                raise ValueError("evaluator source tree exceeds the entry limit")
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"evaluator source tree contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(child_fd)
                    if (opened.st_dev, opened.st_ino) != (
                        metadata.st_dev,
                        metadata.st_ino,
                    ):
                        raise ValueError("evaluator source tree changed while hashing")
                    visit(child_fd, relative)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"evaluator source tree contains a special file: {relative}")
            total_bytes += metadata.st_size
            if total_bytes > _MAX_EVALUATOR_TREE_BYTES:
                raise ValueError("evaluator source tree exceeds the byte limit")
            file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            try:
                opened = os.fstat(file_fd)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (metadata.st_dev, metadata.st_ino)
                ):
                    raise ValueError("evaluator source tree changed while hashing")
                file_digest = hashlib.sha256()
                while chunk := os.read(file_fd, 1024 * 1024):
                    file_digest.update(chunk)
                if os.fstat(file_fd).st_size != opened.st_size:
                    raise ValueError("evaluator source file changed while hashing")
            finally:
                os.close(file_fd)
            update_record(b"F", relative, file_digest.digest())

    root_fd = os.open(root.absolute(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        visit(root_fd, "")
    finally:
        os.close(root_fd)
    return digest.hexdigest()
