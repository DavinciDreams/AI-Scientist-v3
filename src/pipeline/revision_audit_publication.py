"""Bounded publication of revision inputs covered by the evidence audit."""

from __future__ import annotations

import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from src.investigation.paper_revision_workspace import open_pinned_directory

from .publication_cleanup import PinnedDeletionError, remove_pinned_directory_at

_MAX_BOUND_PATHS = 4096
_MAX_ENTRIES = 100_000
_MAX_FILE_BYTES = 100 * 1024 * 1024
_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_DEPTH = 64
_READ_BYTES = 1024 * 1024
_TREE_PREFIX = "evidence-tree/"
_FILE_PREFIXES = (
    "evidence-source/",
    "tex-source/",
    "compile-input/",
    "validator-input/",
    "figure-asset/",
)


@dataclass
class _Budget:
    entries: int = 0
    total_bytes: int = 0

    def add_entry(self) -> None:
        self.entries += 1
        if self.entries > _MAX_ENTRIES:
            raise ValueError("audit-bound publication has too many entries")

    def add_bytes(self, count: int) -> None:
        self.total_bytes += count
        if self.total_bytes > _MAX_TOTAL_BYTES:
            raise ValueError("audit-bound publication is too large")


def _validated_relative_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if (
        not raw_path
        or "\\" in raw_path
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe audit-bound path: {raw_path!r}")
    return path


def _audit_bound_paths(input_sha256: object) -> tuple[list[Path], list[Path]]:
    if not isinstance(input_sha256, dict) or len(input_sha256) > _MAX_BOUND_PATHS:
        raise ValueError("paper evidence audit has invalid input_sha256 paths")
    files: set[Path] = set()
    trees: set[Path] = set()
    for key, digest in input_sha256.items():
        if (
            not isinstance(key, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("paper evidence audit has invalid input_sha256 entry")
        if key.startswith(_TREE_PREFIX):
            trees.add(_validated_relative_path(key.removeprefix(_TREE_PREFIX)))
            continue
        relative_key = key
        for prefix in _FILE_PREFIXES:
            if key.startswith(prefix):
                relative_key = key.removeprefix(prefix)
                break
        files.add(_validated_relative_path(relative_key))
    if files & trees:
        raise ValueError("paper evidence audit binds a path as both file and tree")
    ordered_trees: list[Path] = []
    for tree in sorted(trees, key=lambda path: (len(path.parts), path.as_posix())):
        if not any(tree.is_relative_to(parent) for parent in ordered_trees):
            ordered_trees.append(tree)
    ordered_files = sorted(
        (path for path in files if not any(path.is_relative_to(tree) for tree in ordered_trees)),
        key=lambda path: path.as_posix(),
    )
    return ordered_files, ordered_trees


def _open_relative_parent(root_fd: int, path: Path, *, create: bool) -> int:
    descriptor = os.dup(root_fd)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in path.parts[:-1]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise ValueError(f"audit-bound path has an unsafe directory: {path}") from exc


def _entry_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written == 0:
            raise OSError("could not write audit-bound publication input")
        offset += written


def _temporary_file(target_fd: int, name: str) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for _ in range(100):
        temporary_name = f".{name}.{secrets.token_hex(8)}.tmp"
        try:
            return os.open(temporary_name, flags, 0o600, dir_fd=target_fd), temporary_name
        except FileExistsError:
            continue
    raise ValueError("could not allocate audit-bound publication temporary file")


def _copy_regular_file(
    source_fd: int,
    source_name: str,
    target_fd: int,
    target_name: str,
    budget: _Budget,
) -> None:
    expected = os.stat(source_name, dir_fd=source_fd, follow_symlinks=False)
    if not stat.S_ISREG(expected.st_mode) or expected.st_size > _MAX_FILE_BYTES:
        detail = "too large" if stat.S_ISREG(expected.st_mode) else "not a regular file"
        raise ValueError(f"audit-bound input is {detail}: {source_name}")
    source_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    source_flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        source_descriptor = os.open(source_name, source_flags, dir_fd=source_fd)
    except OSError as exc:
        raise ValueError(f"audit-bound input is not a regular file: {source_name}") from exc
    target_descriptor: int | None = None
    temporary_name: str | None = None
    try:
        opened = os.fstat(source_descriptor)
        if not stat.S_ISREG(opened.st_mode) or _entry_identity(opened) != _entry_identity(expected):
            raise ValueError(f"audit-bound input changed while opening: {source_name}")
        try:
            existing = os.stat(target_name, dir_fd=target_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError(f"audit-bound publication target is not regular: {target_name}")
        target_descriptor, temporary_name = _temporary_file(target_fd, target_name)
        read_bytes = 0
        while chunk := os.read(source_descriptor, _READ_BYTES):
            read_bytes += len(chunk)
            budget.add_bytes(len(chunk))
            if read_bytes > _MAX_FILE_BYTES:
                raise ValueError(f"audit-bound input is too large: {source_name}")
            _write_all(target_descriptor, chunk)
        if read_bytes != expected.st_size:
            raise ValueError(f"audit-bound input changed while copying: {source_name}")
        os.fsync(target_descriptor)
        if _entry_identity(os.fstat(source_descriptor)) != _entry_identity(opened):
            raise ValueError(f"audit-bound input changed while copying: {source_name}")
        os.close(target_descriptor)
        target_descriptor = None
        os.replace(temporary_name, target_name, src_dir_fd=target_fd, dst_dir_fd=target_fd)
        temporary_name = None
    finally:
        os.close(source_descriptor)
        if target_descriptor is not None:
            os.close(target_descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=target_fd)
            except FileNotFoundError:
                pass


def _remove_target_entry(target_fd: int, name: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=target_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode):
        try:
            remove_pinned_directory_at(
                target_fd,
                name,
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
            )
        except PinnedDeletionError as exc:
            raise ValueError(f"audit-bound publication target changed: {name}") from exc
        return
    if stat.S_ISREG(metadata.st_mode):
        os.unlink(name, dir_fd=target_fd)
        return
    raise ValueError(f"audit-bound publication target is not regular: {name}")


def _copy_tree(
    source_fd: int,
    source_name: str,
    target_fd: int,
    target_name: str,
    budget: _Budget,
    *,
    depth: int,
) -> None:
    if depth > _MAX_DEPTH:
        raise ValueError("audit-bound publication tree is too deep")
    expected = os.stat(source_name, dir_fd=source_fd, follow_symlinks=False)
    if not stat.S_ISDIR(expected.st_mode):
        raise ValueError(f"audit-bound tree is not a regular directory: {source_name}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        source_child_fd = os.open(source_name, flags, dir_fd=source_fd)
    except OSError as exc:
        raise ValueError(f"audit-bound tree is not a regular directory: {source_name}") from exc
    target_child_fd: int | None = None
    try:
        opened = os.fstat(source_child_fd)
        if _entry_identity(opened) != _entry_identity(expected):
            raise ValueError(f"audit-bound tree changed while opening: {source_name}")
        _remove_target_entry(target_fd, target_name)
        os.mkdir(target_name, 0o700, dir_fd=target_fd)
        target_child_fd = os.open(target_name, flags, dir_fd=target_fd)
        for name in sorted(os.listdir(source_child_fd), key=os.fsencode):
            budget.add_entry()
            metadata = os.stat(name, dir_fd=source_child_fd, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                _copy_tree(
                    source_child_fd,
                    name,
                    target_child_fd,
                    name,
                    budget,
                    depth=depth + 1,
                )
            elif stat.S_ISREG(metadata.st_mode):
                _copy_regular_file(source_child_fd, name, target_child_fd, name, budget)
            else:
                raise ValueError(f"audit-bound tree entry is not regular: {name}")
        if _entry_identity(os.fstat(source_child_fd)) != _entry_identity(opened):
            raise ValueError(f"audit-bound tree changed while copying: {source_name}")
    finally:
        os.close(source_child_fd)
        if target_child_fd is not None:
            os.close(target_child_fd)


def copy_audit_bound_inputs(
    source_root: Path, target_root: Path, input_sha256: object
) -> None:
    """Copy every safe relative input named by a validated evidence audit."""
    files, trees = _audit_bound_paths(input_sha256)
    source_root_fd = open_pinned_directory(source_root)
    target_root_fd = open_pinned_directory(target_root)
    budget = _Budget()
    try:
        for tree in trees:
            budget.add_entry()
            source_parent = _open_relative_parent(source_root_fd, tree, create=False)
            target_parent = _open_relative_parent(target_root_fd, tree, create=True)
            try:
                _copy_tree(
                    source_parent,
                    tree.name,
                    target_parent,
                    tree.name,
                    budget,
                    depth=0,
                )
            finally:
                os.close(source_parent)
                os.close(target_parent)
        for path in files:
            budget.add_entry()
            source_parent = _open_relative_parent(source_root_fd, path, create=False)
            target_parent = _open_relative_parent(target_root_fd, path, create=True)
            try:
                _copy_regular_file(
                    source_parent,
                    path.name,
                    target_parent,
                    path.name,
                    budget,
                )
            finally:
                os.close(source_parent)
                os.close(target_parent)
    finally:
        os.close(source_root_fd)
        os.close(target_root_fd)
