"""Descriptor-safe merging for interrupted task-tree publication."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from pathlib import Path

from .descriptor_copy import copy_regular_file_no_replace_at
from .safe_fs import UnsafeGluePathError
from .task_tree import TaskTreeManifest, task_tree_manifest_at


def merge_staged_task_tree_at(
    source_fd: int,
    destination_fd: int,
    *,
    expected_manifest: TaskTreeManifest,
    excluded_paths: frozenset[str],
    role: str,
    recover_incomplete_copies: bool = False,
) -> None:
    """Move a staged tree into an absent or matching incomplete destination."""
    _merge_directory(
        source_fd,
        destination_fd,
        relative_dir=Path(),
        role=role,
        recover_incomplete_copies=recover_incomplete_copies,
    )
    actual_manifest = task_tree_manifest_at(
        destination_fd,
        relative_dir=Path(),
        ignore_patterns=(),
    )
    for path in excluded_paths:
        actual_manifest.pop(path, None)
    if actual_manifest != expected_manifest:
        raise UnsafeGluePathError(f"{role} does not match the staged task tree")


def _merge_directory(
    source_fd: int,
    destination_fd: int,
    *,
    relative_dir: Path,
    role: str,
    recover_incomplete_copies: bool,
) -> None:
    for name in sorted(os.listdir(source_fd)):
        relative_path = relative_dir / name
        source_stat = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        _require_owned_entry(source_stat, relative_path, role=role)
        try:
            destination_stat = os.stat(
                name,
                dir_fd=destination_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            destination_stat = None
        if destination_stat is not None:
            _require_owned_entry(destination_stat, relative_path, role=role)

        if stat.S_ISDIR(source_stat.st_mode):
            _merge_directory_entry(
                source_fd,
                destination_fd,
                name=name,
                relative_path=relative_path,
                source_stat=source_stat,
                destination_stat=destination_stat,
                role=role,
                recover_incomplete_copies=recover_incomplete_copies,
            )
            continue
        if stat.S_ISREG(source_stat.st_mode) or stat.S_ISLNK(source_stat.st_mode):
            _merge_leaf_entry(
                source_fd,
                destination_fd,
                name=name,
                relative_path=relative_path,
                source_stat=source_stat,
                destination_stat=destination_stat,
                role=role,
                recover_incomplete_copies=recover_incomplete_copies,
            )
            continue
        raise UnsafeGluePathError(f"{role} contains unsupported entry: {relative_path}")
    os.fsync(destination_fd)


def _merge_directory_entry(
    source_parent_fd: int,
    destination_parent_fd: int,
    *,
    name: str,
    relative_path: Path,
    source_stat: os.stat_result,
    destination_stat: os.stat_result | None,
    role: str,
    recover_incomplete_copies: bool,
) -> None:
    created = destination_stat is None
    if destination_stat is None:
        try:
            os.mkdir(
                name,
                stat.S_IMODE(source_stat.st_mode) | stat.S_IRWXU,
                dir_fd=destination_parent_fd,
            )
        except FileExistsError:
            created = False
        destination_stat = os.stat(
            name,
            dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
        _require_owned_entry(destination_stat, relative_path, role=role)
    if not stat.S_ISDIR(destination_stat.st_mode):
        raise UnsafeGluePathError(f"{role} conflicts at {relative_path}")

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    source_fd = os.open(name, flags, dir_fd=source_parent_fd)
    destination_fd = os.open(name, flags, dir_fd=destination_parent_fd)
    try:
        _require_same_entry(source_stat, os.fstat(source_fd), relative_path, role=role)
        if created:
            os.fchmod(
                destination_fd,
                stat.S_IMODE(source_stat.st_mode) | stat.S_IRWXU,
            )
            destination_stat = os.fstat(destination_fd)
        _require_same_entry(
            destination_stat,
            os.fstat(destination_fd),
            relative_path,
            role=role,
        )
        _merge_directory(
            source_fd,
            destination_fd,
            relative_dir=relative_path,
            role=role,
            recover_incomplete_copies=recover_incomplete_copies,
        )
        os.fchmod(destination_fd, stat.S_IMODE(source_stat.st_mode))
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)
        os.close(source_fd)


def _merge_leaf_entry(
    source_parent_fd: int,
    destination_parent_fd: int,
    *,
    name: str,
    relative_path: Path,
    source_stat: os.stat_result,
    destination_stat: os.stat_result | None,
    role: str,
    recover_incomplete_copies: bool,
) -> None:
    if destination_stat is None:
        try:
            if stat.S_ISREG(source_stat.st_mode):
                try:
                    os.link(
                        name,
                        name,
                        src_dir_fd=source_parent_fd,
                        dst_dir_fd=destination_parent_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    if exc.errno not in {
                        errno.EACCES,
                        errno.EPERM,
                        errno.EXDEV,
                        errno.ENOTSUP,
                        errno.EOPNOTSUPP,
                    }:
                        raise
                    copy_regular_file_no_replace_at(
                        source_parent_fd,
                        name,
                        destination_parent_fd,
                        name,
                        expected_source=source_stat,
                        role=f"{role} file {relative_path}",
                    )
            else:
                os.symlink(
                    os.readlink(name, dir_fd=source_parent_fd),
                    name,
                    dir_fd=destination_parent_fd,
                )
            return
        except FileExistsError:
            destination_stat = os.stat(
                name,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
            _require_owned_entry(destination_stat, relative_path, role=role)
    if stat.S_IFMT(source_stat.st_mode) != stat.S_IFMT(destination_stat.st_mode):
        raise UnsafeGluePathError(f"{role} conflicts at {relative_path}")
    if stat.S_ISLNK(source_stat.st_mode):
        if os.readlink(name, dir_fd=source_parent_fd) != os.readlink(
            name,
            dir_fd=destination_parent_fd,
        ):
            raise UnsafeGluePathError(f"{role} conflicts at {relative_path}")
        current_destination = os.stat(
            name,
            dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
        _require_same_entry(destination_stat, current_destination, relative_path, role=role)
        return
    source_identity = _file_identity(
        source_parent_fd,
        name,
        source_stat,
        relative_path,
        role=role,
    )
    destination_identity = _file_identity(
        destination_parent_fd,
        name,
        destination_stat,
        relative_path,
        role=role,
    )
    if source_identity == destination_identity:
        return
    if not recover_incomplete_copies or destination_stat.st_nlink != 1:
        raise UnsafeGluePathError(f"{role} conflicts at {relative_path}")
    current = os.stat(name, dir_fd=destination_parent_fd, follow_symlinks=False)
    _require_same_entry(destination_stat, current, relative_path, role=role)
    os.unlink(name, dir_fd=destination_parent_fd)
    _merge_leaf_entry(
        source_parent_fd,
        destination_parent_fd,
        name=name,
        relative_path=relative_path,
        source_stat=source_stat,
        destination_stat=None,
        role=role,
        recover_incomplete_copies=False,
    )


def _file_identity(
    parent_fd: int,
    name: str,
    expected_stat: os.stat_result,
    relative_path: Path,
    *,
    role: str,
) -> tuple[str, int]:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise UnsafeGluePathError(f"{role} conflicts at {relative_path}")
        _require_same_entry(expected_stat, metadata, relative_path, role=role)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        _require_same_entry(metadata, os.fstat(descriptor), relative_path, role=role)
        return digest, stat.S_IMODE(metadata.st_mode)
    finally:
        os.close(descriptor)


def _require_same_entry(
    expected: os.stat_result,
    actual: os.stat_result,
    relative_path: Path,
    *,
    role: str,
) -> None:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(expected, field) != getattr(actual, field) for field in fields):
        raise UnsafeGluePathError(f"{role} changed while publishing: {relative_path}")


def _require_owned_entry(
    metadata: os.stat_result,
    relative_path: Path,
    *,
    role: str,
) -> None:
    if metadata.st_uid != os.getuid():
        raise UnsafeGluePathError(f"{role} contains an unowned entry: {relative_path}")
