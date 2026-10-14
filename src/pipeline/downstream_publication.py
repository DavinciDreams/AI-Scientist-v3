"""Publish frozen programs on filesystems without exclusive directory rename."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from .safe_fs import AtomicNoReplaceUnsupported, UnsafeGluePathError, rename_directory_no_replace
from .task_tree import copy_task_tree_at, verify_copied_task_tree


def publish_frozen_program_at(parent_fd: int, staged_name: str) -> None:
    """Reserve the program path once; its selection manifest commits the verified tree."""
    try:
        rename_directory_no_replace(parent_fd, staged_name, "program",
                                    role="frozen downstream program")
    except AtomicNoReplaceUnsupported:
        _copy_frozen_program_at(parent_fd, staged_name)
    os.fsync(parent_fd)


def _copy_frozen_program_at(parent_fd: int, staged_name: str) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    staged_fd = os.open(staged_name, flags, dir_fd=parent_fd)
    try:
        # Never adopt an existing tree, including a previous interrupted publication.
        os.mkdir("program", 0o700, dir_fd=parent_fd)
        created = os.stat("program", dir_fd=parent_fd, follow_symlinks=False)
        destination_fd = os.open("program", flags, dir_fd=parent_fd)
        try:
            if not os.path.samestat(created, os.fstat(destination_fd)):
                raise UnsafeGluePathError("frozen downstream program changed before copying")
            copy_task_tree_at(staged_fd, destination_fd, relative_dir=Path(), ignore_patterns=())
            verify_copied_task_tree(staged_fd, destination_fd, ignore_patterns=())
            current = os.stat("program", dir_fd=parent_fd, follow_symlinks=False)
            if not os.path.samestat(created, current):
                raise UnsafeGluePathError("frozen downstream program changed while copying")
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
    finally:
        os.close(staged_fd)


def publish_selection_manifest_at(parent_fd: int, name: str, content: bytes) -> None:
    """Commit a complete manifest atomically without replacing another publication."""
    temporary_name = f".{name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent_fd)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                follow_symlinks=False)
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.unlink(temporary_name, dir_fd=parent_fd)
