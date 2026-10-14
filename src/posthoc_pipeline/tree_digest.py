"""Bounded, no-follow hashing for agent-writable directory trees."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

MAX_TREE_ENTRIES = 100_000
MAX_TREE_DEPTH = 64
MAX_TREE_FILE_BYTES = 1024 * 1024 * 1024
MAX_TREE_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_TREE_DIGEST_SECONDS = 60.0
_READ_CHUNK_BYTES = 1024 * 1024


class TreeDigestError(ValueError):
    """Tree cannot be hashed safely within the inspection budget."""


class _Digest(Protocol):
    def update(self, value: bytes) -> None: ...


@dataclass
class _DigestBudget:
    deadline: float
    entries: int = 0
    total_bytes: int = 0

    def consume_entry(self, *, depth: int) -> None:
        self.check_deadline()
        if depth > MAX_TREE_DEPTH:
            raise TreeDigestError("tree digest depth limit exceeded")
        self.entries += 1
        if self.entries > MAX_TREE_ENTRIES:
            raise TreeDigestError("tree digest entry limit exceeded")

    def consume_file(self, size: int) -> None:
        if size > MAX_TREE_FILE_BYTES:
            raise TreeDigestError("tree digest per-file byte limit exceeded")
        self.total_bytes += size
        if self.total_bytes > MAX_TREE_TOTAL_BYTES:
            raise TreeDigestError("tree digest aggregate byte limit exceeded")

    def consume_growth(self, size: int) -> None:
        self.total_bytes += size
        if self.total_bytes > MAX_TREE_TOTAL_BYTES:
            raise TreeDigestError("tree digest aggregate byte limit exceeded")

    def check_deadline(self) -> None:
        if time.monotonic() > self.deadline:
            raise TreeDigestError("tree digest time limit exceeded")


def tree_digest(root: Path) -> str:
    """Hash one tree without following links or exceeding fixed resource budgets."""
    digest = hashlib.sha256()
    try:
        root_metadata = root.lstat()
    except FileNotFoundError:
        _update_framed(digest, b"R", b"missing")
        return digest.hexdigest()
    if stat.S_ISLNK(root_metadata.st_mode):
        _update_framed(digest, b"R", b"symlink")
        _update_framed(digest, b"L", os.fsencode(os.readlink(root)))
        return digest.hexdigest()
    if not stat.S_ISDIR(root_metadata.st_mode):
        _update_framed(digest, b"R", b"other")
        return digest.hexdigest()

    _update_framed(digest, b"R", b"directory")
    budget = _DigestBudget(deadline=time.monotonic() + MAX_TREE_DIGEST_SECONDS)
    root_descriptor = _open_directory(root)
    try:
        _digest_directory(root_descriptor, "", digest, budget, depth=0)
    finally:
        os.close(root_descriptor)
    return digest.hexdigest()


def _digest_directory(
    directory_descriptor: int,
    relative_parent: str,
    digest: _Digest,
    budget: _DigestBudget,
    *,
    depth: int,
) -> None:
    with os.scandir(directory_descriptor) as entries:
        entry_names: list[str] = []
        for entry in entries:
            budget.consume_entry(depth=depth + 1)
            entry_names.append(entry.name)
    entry_names.sort()
    for name in entry_names:
        relative = f"{relative_parent}/{name}" if relative_parent else name
        entry_depth = depth + 1
        _update_framed(digest, b"P", os.fsencode(relative))
        try:
            metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise TreeDigestError(f"unable to inspect tree entry: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            _update_framed(digest, b"T", b"symlink")
            try:
                target = os.readlink(name, dir_fd=directory_descriptor)
            except OSError as exc:
                raise TreeDigestError(f"unable to inspect tree symlink: {relative}") from exc
            _update_framed(digest, b"L", os.fsencode(target))
        elif stat.S_ISREG(metadata.st_mode):
            _update_framed(digest, b"T", b"file")
            _digest_file(
                directory_descriptor,
                name,
                relative,
                digest,
                budget,
            )
        elif stat.S_ISDIR(metadata.st_mode):
            _update_framed(digest, b"T", b"directory")
            child_descriptor = _open_directory(name, dir_fd=directory_descriptor)
            try:
                _digest_directory(
                    child_descriptor,
                    relative,
                    digest,
                    budget,
                    depth=entry_depth,
                )
            finally:
                os.close(child_descriptor)
        else:
            _update_framed(digest, b"T", b"special")


def _digest_file(
    directory_descriptor: int,
    name: str,
    relative: str,
    digest: _Digest,
    budget: _DigestBudget,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        file_descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise TreeDigestError(f"unable to open regular tree file: {relative}") from exc
    try:
        initial_metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(initial_metadata.st_mode):
            raise TreeDigestError(f"tree entry changed type during inspection: {relative}")
        budget.consume_file(initial_metadata.st_size)
        _update_framed(
            digest,
            b"Z",
            initial_metadata.st_size.to_bytes(8, byteorder="big"),
        )
        content_digest = hashlib.sha256()
        bytes_read = 0
        while True:
            budget.check_deadline()
            chunk = os.read(file_descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > MAX_TREE_FILE_BYTES:
                raise TreeDigestError("tree digest per-file byte limit exceeded")
            if bytes_read > initial_metadata.st_size:
                budget.consume_growth(
                    bytes_read - max(initial_metadata.st_size, bytes_read - len(chunk))
                )
            content_digest.update(chunk)
        final_metadata = os.fstat(file_descriptor)
        if (
            final_metadata.st_size != initial_metadata.st_size
            or final_metadata.st_mtime_ns != initial_metadata.st_mtime_ns
        ):
            raise TreeDigestError(f"tree file changed during inspection: {relative}")
        _update_framed(digest, b"C", content_digest.digest())
    finally:
        os.close(file_descriptor)


def _open_directory(path: str | Path, *, dir_fd: int | None = None) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise TreeDigestError(f"unable to open tree directory: {path}") from exc


def _update_framed(digest: _Digest, domain: bytes, value: bytes) -> None:
    digest.update(domain)
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)
