"""Descriptor-safe exclusive regular-file publication."""

from __future__ import annotations

import os
import stat

from .safe_fs import UnsafeGluePathError


def copy_regular_file_no_replace_at(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
    *,
    expected_source: os.stat_result,
    role: str,
) -> None:
    """Copy one stable regular file to an exclusively-created destination."""
    source_fd = os.open(
        source_name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=source_parent_fd,
    )
    destination_fd: int | None = None
    destination_identity: tuple[int, int] | None = None
    try:
        source_stat = os.fstat(source_fd)
        _require_same_source(expected_source, source_stat, role=role)
        destination_fd = os.open(
            destination_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_parent_fd,
        )
        created = os.fstat(destination_fd)
        destination_identity = (created.st_dev, created.st_ino)
        while chunk := os.read(source_fd, 1024 * 1024):
            view = memoryview(chunk)
            while view:
                view = view[os.write(destination_fd, view) :]
        os.fchmod(destination_fd, stat.S_IMODE(source_stat.st_mode))
        os.fsync(destination_fd)
        _require_same_source(source_stat, os.fstat(source_fd), role=role)
        published = os.fstat(destination_fd)
        if (
            not stat.S_ISREG(published.st_mode)
            or published.st_uid != os.getuid()
            or published.st_nlink != 1
            or published.st_size != source_stat.st_size
        ):
            raise UnsafeGluePathError(f"{role} copy is not a safe regular file")
    except BaseException:
        if destination_identity is not None:
            try:
                current = os.stat(
                    destination_name,
                    dir_fd=destination_parent_fd,
                    follow_symlinks=False,
                )
                if (current.st_dev, current.st_ino) == destination_identity:
                    os.unlink(destination_name, dir_fd=destination_parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)


def _require_same_source(
    expected: os.stat_result,
    actual: os.stat_result,
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
    if not stat.S_ISREG(actual.st_mode) or any(
        getattr(expected, field) != getattr(actual, field) for field in fields
    ):
        raise UnsafeGluePathError(f"{role} source changed while publishing")
