"""Descriptor-relative file operations for untrusted artifact directories."""

from __future__ import annotations

import fcntl
import os
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator


def absolute_path(path: Path) -> Path:
    """Normalize a path lexically without following symlinks."""
    return Path(os.path.abspath(path.expanduser()))


def open_directory_tree_no_follow(path: Path) -> int:
    """Open and pin a directory path without following any component."""
    absolute = absolute_path(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_relative_directory_no_follow(root_fd: int, relative_path: Path) -> int:
    """Open and pin a descendant directory relative to an existing descriptor."""
    descriptor = os.dup(root_fd)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for component in relative_path.parts:
            if component in {"", "."}:
                continue
            if component == "..":
                raise ValueError("directory is outside the pinned root")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_at(directory_fd: int, name: str) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_fd,
    )
    if stat.S_ISREG(os.fstat(descriptor).st_mode):
        return descriptor
    os.close(descriptor)
    raise OSError(f"unsafe non-regular artifact: {name}")


def read_regular_at(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int,
    description: str | None = None,
) -> bytes:
    """Read a bounded regular file from a pinned directory."""
    with os.fdopen(_open_regular_at(directory_fd, name), "rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"{description or 'artifact'} is too large: {name}")
    return content


def copy_regular_at(
    directory_fd: int,
    name: str,
    destination: BinaryIO,
    *,
    max_bytes: int,
    description: str = "artifact",
) -> None:
    """Copy a bounded regular file from a pinned directory into an open stream."""
    with os.fdopen(_open_regular_at(directory_fd, name), "rb") as source:
        if os.fstat(source.fileno()).st_size > max_bytes:
            raise ValueError(f"{description} exceeds {max_bytes} bytes")
        copied = 0
        while chunk := source.read(min(1024 * 1024, max_bytes + 1 - copied)):
            copied += len(chunk)
            if copied > max_bytes:
                raise ValueError(f"{description} exceeds {max_bytes} bytes")
            destination.write(chunk)


def atomic_write_at(directory_fd: int, name: str, content: bytes, *, mode: int) -> None:
    """Atomically replace a file relative to a pinned directory."""
    temporary_name = f".{name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


@contextmanager
def manifest_lock_at(paper_root_fd: int) -> Iterator[None]:
    """Serialize manifest updates relative to a pinned paper root."""
    descriptor = os.open(
        ".latest_manifest.lock",
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=paper_root_fd,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("paper manifest lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
