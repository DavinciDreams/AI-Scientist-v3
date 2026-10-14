"""Bounded, descriptor-relative durability and integrity checks for paper trees."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from typing import Protocol

_MAX_ENTRIES = 100_000
_MAX_FILE_BYTES = 100 * 1024 * 1024
_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_DEPTH = 64
_READ_CHUNK_BYTES = 1024 * 1024


class PublicationTreeError(RuntimeError):
    """A publication tree is unsafe, too large, or changed while hashing."""


class _Digest(Protocol):
    def update(self, content: bytes) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass
class _Budget:
    entries: int = 0
    total_bytes: int = 0


def sync_and_digest_directory_at(parent_fd: int, name: str) -> str:
    """Fsync a directory tree bottom-up and return its bounded content digest."""
    return _digest_directory_at(parent_fd, name, sync=True)


def digest_directory_at(parent_fd: int, name: str) -> str:
    """Return a bounded digest without following symlinks."""
    return _digest_directory_at(parent_fd, name, sync=False)


def _digest_directory_at(parent_fd: int, name: str, *, sync: bool) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise PublicationTreeError("publication tree root is not a directory")
        digest: _Digest = hashlib.sha256()
        _visit_directory(descriptor, digest, _Budget(), depth=0, sync=sync)
        after = os.fstat(descriptor)
        _require_unchanged(before, after, "publication tree root")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _visit_directory(
    directory_fd: int,
    digest: _Digest,
    budget: _Budget,
    *,
    depth: int,
    sync: bool,
) -> None:
    if depth > _MAX_DEPTH:
        raise PublicationTreeError("publication tree is too deep")
    names = sorted(os.listdir(directory_fd), key=os.fsencode)
    for name in names:
        budget.entries += 1
        if budget.entries > _MAX_ENTRIES:
            raise PublicationTreeError("publication tree has too many entries")
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _update_digest(digest, b"name", os.fsencode(name))
        if stat.S_ISREG(metadata.st_mode):
            _update_digest(digest, b"type", b"file")
            _digest_regular_file(directory_fd, name, metadata, digest, budget, sync=sync)
        elif stat.S_ISDIR(metadata.st_mode):
            _update_digest(digest, b"type", b"directory")
            _digest_child_directory(
                directory_fd,
                name,
                metadata,
                digest,
                budget,
                depth=depth + 1,
                sync=sync,
            )
        elif stat.S_ISLNK(metadata.st_mode):
            _update_digest(digest, b"type", b"symlink")
            target = os.readlink(name, dir_fd=directory_fd)
            after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            _require_unchanged(metadata, after, f"publication symlink {name}")
            _update_digest(digest, b"target", os.fsencode(target))
        else:
            raise PublicationTreeError(f"publication tree contains a special file: {name}")
    if sync:
        os.fsync(directory_fd)


def _digest_regular_file(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
    digest: _Digest,
    budget: _Budget,
    *,
    sync: bool,
) -> None:
    if expected.st_size > _MAX_FILE_BYTES:
        raise PublicationTreeError(f"publication file is too large: {name}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        _require_unchanged(expected, opened, f"publication file {name}")
        read_bytes = 0
        _update_digest(digest, b"content-size", expected.st_size.to_bytes(8, "big"))
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            read_bytes += len(chunk)
            budget.total_bytes += len(chunk)
            if read_bytes > _MAX_FILE_BYTES or budget.total_bytes > _MAX_TOTAL_BYTES:
                raise PublicationTreeError("publication tree content is too large")
            digest.update(chunk)
        if read_bytes != expected.st_size:
            raise PublicationTreeError(f"publication file changed while hashing: {name}")
        if sync:
            os.fsync(descriptor)
        after = os.fstat(descriptor)
        _require_unchanged(opened, after, f"publication file {name}")
    finally:
        os.close(descriptor)


def _digest_child_directory(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    digest: _Digest,
    budget: _Budget,
    *,
    depth: int,
    sync: bool,
) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        _require_unchanged(expected, opened, f"publication directory {name}")
        _visit_directory(descriptor, digest, budget, depth=depth, sync=sync)
        _update_digest(digest, b"end-directory", b"")
        after = os.fstat(descriptor)
        _require_unchanged(opened, after, f"publication directory {name}")
    finally:
        os.close(descriptor)


def _require_unchanged(
    before: os.stat_result, after: os.stat_result, description: str
) -> None:
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise PublicationTreeError(f"{description} changed while hashing")


def _update_digest(digest: _Digest, label: bytes, value: bytes) -> None:
    framed = len(label).to_bytes(2, "big") + label + len(value).to_bytes(8, "big") + value
    digest.update(framed)
