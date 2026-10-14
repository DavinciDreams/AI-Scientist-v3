"""Descriptor-safe task-tree copying, verification, and cleanup."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
from pathlib import Path

from src.investigation.files import task_copy_entry_is_ignored

from .safe_fs import UnsafeGluePathError, entry_exists_at

TaskTreeManifest = dict[str, tuple[str, ...]]
logger = logging.getLogger(__name__)


def copy_task_tree_at(
    source_fd: int,
    destination_fd: int,
    *,
    relative_dir: Path,
    ignore_patterns: tuple[str, ...],
) -> None:
    directory_stat = os.fstat(source_fd)
    names = _list_directory(source_fd, relative_dir)
    for name in names:
        _copy_task_entry_at(
            source_fd,
            destination_fd,
            name=name,
            relative_dir=relative_dir,
            ignore_patterns=ignore_patterns,
        )
    _require_unchanged_directory(source_fd, directory_stat, names, relative_dir)


def verify_copied_task_tree(
    source_fd: int,
    destination_fd: int,
    *,
    ignore_patterns: tuple[str, ...],
    destination_only_paths: frozenset[str] = frozenset(),
) -> None:
    source_manifest_before = task_tree_manifest_at(
        source_fd,
        relative_dir=Path(),
        ignore_patterns=ignore_patterns,
    )
    destination_manifest = task_tree_manifest_at(
        destination_fd,
        relative_dir=Path(),
        ignore_patterns=(),
    )
    if not destination_only_paths.issubset(destination_manifest):
        raise UnsafeGluePathError("staged task is missing required generated entries")
    for path in destination_only_paths:
        del destination_manifest[path]
    source_manifest_after = task_tree_manifest_at(
        source_fd,
        relative_dir=Path(),
        ignore_patterns=ignore_patterns,
    )
    if not (
        source_manifest_before == destination_manifest == source_manifest_after
    ):
        raise UnsafeGluePathError("task source changed while copying")


def require_unchanged_task_root(
    directory_fd: int,
    expected_stat: os.stat_result,
) -> None:
    actual_stat = os.fstat(directory_fd)
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(expected_stat, field) != getattr(actual_stat, field) for field in fields):
        raise UnsafeGluePathError("task source changed while copying: .")


def require_unchanged_pinned_task_root(
    directory_fd: int,
    expected_stat: os.stat_result,
) -> None:
    actual_stat = os.fstat(directory_fd)
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(expected_stat, field) != getattr(actual_stat, field) for field in fields):
        raise UnsafeGluePathError("task source changed while copying: .")


def task_tree_manifest_at(
    directory_fd: int,
    *,
    relative_dir: Path,
    ignore_patterns: tuple[str, ...],
) -> TaskTreeManifest:
    directory_stat = os.fstat(directory_fd)
    names = _list_directory(directory_fd, relative_dir)
    manifest: TaskTreeManifest = {}
    for name in names:
        relative_path = relative_dir / name
        try:
            manifest.update(
                _task_entry_manifest_at(
                    directory_fd,
                    name=name,
                    relative_path=relative_path,
                    ignore_patterns=ignore_patterns,
                )
            )
        except UnsafeGluePathError:
            raise
        except OSError as exc:
            raise UnsafeGluePathError(
                f"task source changed while copying: {relative_path}"
            ) from exc
    _require_unchanged_directory(directory_fd, directory_stat, names, relative_dir)
    return manifest


def _task_entry_manifest_at(
    directory_fd: int,
    *,
    name: str,
    relative_path: Path,
    ignore_patterns: tuple[str, ...],
) -> TaskTreeManifest:
    entry_stat = _lstat_entry(directory_fd, name, relative_path)
    is_directory = stat.S_ISDIR(entry_stat.st_mode)
    if task_copy_entry_is_ignored(
        name=name,
        relative_path=relative_path.as_posix(),
        is_directory=is_directory,
        patterns=ignore_patterns,
    ):
        return {}
    if is_directory:
        manifest: TaskTreeManifest = {
            relative_path.as_posix(): (
                "directory",
                oct(stat.S_IMODE(entry_stat.st_mode)),
            )
        }
        child_fd = _open_directory_entry(directory_fd, name, entry_stat, relative_path)
        try:
            manifest.update(
                task_tree_manifest_at(
                    child_fd,
                    relative_dir=relative_path,
                    ignore_patterns=ignore_patterns,
                )
            )
        finally:
            os.close(child_fd)
        return manifest
    if stat.S_ISREG(entry_stat.st_mode):
        digest = _regular_file_digest_at(directory_fd, name, entry_stat, relative_path)
        return {
            relative_path.as_posix(): (
                "file",
                digest,
                oct(stat.S_IMODE(entry_stat.st_mode)),
            )
        }
    if stat.S_ISLNK(entry_stat.st_mode):
        target = os.readlink(name, dir_fd=directory_fd)
        current_stat = _lstat_entry(directory_fd, name, relative_path)
        _require_unchanged_entry(entry_stat, current_stat, relative_path)
        return {relative_path.as_posix(): ("symlink", target)}
    raise UnsafeGluePathError(f"task source contains unsupported entry: {relative_path}")


def cleanup_task_tree_preserving_error(parent_fd: int, name: str, *, role: str) -> None:
    try:
        if entry_exists_at(parent_fd, name):
            _remove_task_tree_at(parent_fd, name)
    except Exception:
        logger.exception("could not clean %s: %s", role, name)


def _copy_task_entry_at(
    source_fd: int,
    destination_fd: int,
    *,
    name: str,
    relative_dir: Path,
    ignore_patterns: tuple[str, ...],
) -> None:
    relative_path = relative_dir / name
    try:
        source_stat = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        is_directory = stat.S_ISDIR(source_stat.st_mode)
        if task_copy_entry_is_ignored(
            name=name,
            relative_path=relative_path.as_posix(),
            is_directory=is_directory,
            patterns=ignore_patterns,
        ):
            return
        if is_directory:
            _copy_task_directory_at(
                source_fd,
                destination_fd,
                name=name,
                relative_path=relative_path,
                source_stat=source_stat,
                ignore_patterns=ignore_patterns,
            )
        elif stat.S_ISREG(source_stat.st_mode):
            _copy_task_file_at(
                source_fd,
                destination_fd,
                name=name,
                relative_path=relative_path,
                source_stat=source_stat,
            )
        elif stat.S_ISLNK(source_stat.st_mode):
            _copy_task_symlink_at(
                source_fd,
                destination_fd,
                name=name,
                relative_path=relative_path,
                source_stat=source_stat,
            )
        else:
            raise UnsafeGluePathError(f"task source contains unsupported entry: {relative_path}")
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError(f"task source changed while copying: {relative_path}") from exc


def _copy_task_directory_at(
    source_parent_fd: int,
    destination_parent_fd: int,
    *,
    name: str,
    relative_path: Path,
    source_stat: os.stat_result,
    ignore_patterns: tuple[str, ...],
) -> None:
    source_fd = _open_directory_entry(source_parent_fd, name, source_stat, relative_path)
    try:
        temporary_mode = stat.S_IMODE(source_stat.st_mode) | stat.S_IRWXU
        os.mkdir(name, temporary_mode, dir_fd=destination_parent_fd)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        destination_fd = os.open(name, flags, dir_fd=destination_parent_fd)
        try:
            copy_task_tree_at(
                source_fd,
                destination_fd,
                relative_dir=relative_path,
                ignore_patterns=ignore_patterns,
            )
            os.fchmod(destination_fd, stat.S_IMODE(source_stat.st_mode))
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)


def _copy_task_file_at(
    source_parent_fd: int,
    destination_parent_fd: int,
    *,
    name: str,
    relative_path: Path,
    source_stat: os.stat_result,
) -> None:
    source_fd = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=source_parent_fd,
    )
    try:
        opened_stat = os.fstat(source_fd)
        _require_unchanged_entry(source_stat, opened_stat, relative_path)
        destination_fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            stat.S_IMODE(source_stat.st_mode) | stat.S_IWUSR,
            dir_fd=destination_parent_fd,
        )
        with os.fdopen(destination_fd, "wb") as destination_file:
            with os.fdopen(os.dup(source_fd), "rb") as source_file:
                shutil.copyfileobj(source_file, destination_file)
            _require_unchanged_entry(opened_stat, os.fstat(source_fd), relative_path)
            os.fchmod(destination_file.fileno(), stat.S_IMODE(source_stat.st_mode))
    finally:
        os.close(source_fd)


def _copy_task_symlink_at(
    source_parent_fd: int,
    destination_parent_fd: int,
    *,
    name: str,
    relative_path: Path,
    source_stat: os.stat_result,
) -> None:
    target = os.readlink(name, dir_fd=source_parent_fd)
    current_stat = _lstat_entry(source_parent_fd, name, relative_path)
    _require_unchanged_entry(source_stat, current_stat, relative_path)
    os.symlink(target, name, dir_fd=destination_parent_fd)


def _regular_file_digest_at(
    parent_fd: int,
    name: str,
    expected_stat: os.stat_result,
    relative_path: Path,
) -> str:
    file_fd = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=parent_fd,
    )
    try:
        opened_stat = os.fstat(file_fd)
        _require_unchanged_entry(expected_stat, opened_stat, relative_path)
        with os.fdopen(os.dup(file_fd), "rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        _require_unchanged_entry(opened_stat, os.fstat(file_fd), relative_path)
        return digest
    finally:
        os.close(file_fd)


def _open_directory_entry(
    parent_fd: int,
    name: str,
    expected_stat: os.stat_result,
    relative_path: Path,
) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        _require_unchanged_entry(expected_stat, os.fstat(directory_fd), relative_path)
    except Exception:
        os.close(directory_fd)
        raise
    return directory_fd


def _lstat_entry(parent_fd: int, name: str, relative_path: Path) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise UnsafeGluePathError(f"task source changed while copying: {relative_path}") from exc


def _list_directory(directory_fd: int, relative_dir: Path) -> list[str]:
    try:
        return sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise UnsafeGluePathError(
            f"task source changed while copying: {_display_path(relative_dir)}"
        ) from exc


def _require_unchanged_directory(
    directory_fd: int,
    expected_stat: os.stat_result,
    expected_names: list[str],
    relative_dir: Path,
) -> None:
    if expected_names != _list_directory(directory_fd, relative_dir):
        raise UnsafeGluePathError(
            f"task source changed while copying: {_display_path(relative_dir)}"
        )
    _require_unchanged_entry(expected_stat, os.fstat(directory_fd), _display_path(relative_dir))


def _require_unchanged_entry(
    expected: os.stat_result,
    actual: os.stat_result,
    relative_path: Path,
) -> None:
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(expected, field) != getattr(actual, field) for field in fields):
        raise UnsafeGluePathError(f"task source changed while copying: {relative_path}")


def _display_path(relative_path: Path) -> Path:
    return relative_path if relative_path.parts else Path(".")


def _remove_task_tree_at(parent_fd: int, name: str) -> None:
    mode = os.stat(name, dir_fd=parent_fd, follow_symlinks=False).st_mode
    if not stat.S_ISDIR(mode):
        os.unlink(name, dir_fd=parent_fd)
        return

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(directory_fd)
        os.fchmod(directory_fd, stat.S_IRWXU)
        for child_name in os.listdir(directory_fd):
            _remove_task_tree_at(directory_fd, child_name)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise UnsafeGluePathError(f"temporary task snapshot changed during cleanup: {name}")
    finally:
        os.close(directory_fd)
    os.rmdir(name, dir_fd=parent_fd)
