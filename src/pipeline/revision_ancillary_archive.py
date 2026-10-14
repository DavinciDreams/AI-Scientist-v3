"""Preserve displaced original evidence outside a revision's active audit inventory."""

from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from pathlib import Path

from src.orchestrator.paper_evidence_audit import audit_input_sha256
from src.investigation.paper_revision_workspace import open_pinned_directory

from .revision_audit_publication import (
    _Budget,
    _MAX_DEPTH,
    _audit_bound_paths,
    _copy_regular_file,
    _entry_identity,
    _open_relative_parent,
)


def _revision_tree_contains_file(root_fd: int, path: Path) -> bool:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    with ExitStack() as stack:
        descriptor = root_fd
        try:
            for component in path.parts[:-1]:
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISREG(metadata.st_mode):
                    return False
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(f"revision tree ancestor is not regular: {path}")
                descriptor = os.open(component, flags, dir_fd=descriptor)
                stack.callback(os.close, descriptor)
                if _entry_identity(os.fstat(descriptor)) != _entry_identity(metadata):
                    raise ValueError(f"revision tree ancestor changed: {path}")
            metadata = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISDIR(metadata.st_mode):
            return False
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"revision tree input is not regular: {path}")
        return True


def _tree_files(root_fd: int, tree: Path, budget: _Budget) -> set[Path]:
    files: set[Path] = set()

    def collect(parent_fd: int, path: Path, depth: int) -> None:
        if depth > _MAX_DEPTH:
            raise ValueError("original evidence tree is too deep")
        budget.add_entry()
        metadata = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISREG(metadata.st_mode):
            budget.add_bytes(metadata.st_size)
            files.add(path)
            return
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"original evidence tree input is not regular: {path}")
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        try:
            if _entry_identity(os.fstat(descriptor)) != _entry_identity(metadata):
                raise ValueError(f"original evidence tree changed: {path}")
            for name in sorted(os.listdir(descriptor)):
                collect(descriptor, path / name, depth + 1)
            if _entry_identity(os.fstat(descriptor)) != _entry_identity(metadata):
                raise ValueError(f"original evidence tree changed: {path}")
        finally:
            os.close(descriptor)

    parent_fd = _open_relative_parent(root_fd, tree, create=False)
    try:
        collect(parent_fd, tree, 0)
    finally:
        os.close(parent_fd)
    return files


def _archive_file(source_fd: int, archive_fd: int, path: Path, budget: _Budget) -> None:
    with ExitStack() as stack:
        source_parent = _open_relative_parent(source_fd, path, create=False)
        stack.callback(os.close, source_parent)
        target_parent = _open_relative_parent(archive_fd, path, create=True)
        stack.callback(os.close, target_parent)
        expected = os.stat(path.name, dir_fd=source_parent, follow_symlinks=False)
        try:
            os.stat(path.name, dir_fd=target_parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"ancillary archive destination already exists: {path}")
        budget.add_entry()
        _copy_regular_file(source_parent, path.name, target_parent, path.name, budget)
        current = os.stat(path.name, dir_fd=source_parent, follow_symlinks=False)
        if _entry_identity(current) != _entry_identity(expected):
            raise ValueError(f"original input changed during archival: {path}")
        os.unlink(path.name, dir_fd=source_parent)


def archive_original_only_inputs(
    staged_dir: Path,
    revision_snapshot: Path,
    original_snapshot: Path,
    revision_input_sha256: object,
) -> None:
    """Archive only original audited files absent from the validated revision inputs.

    The caller owns the private staging tree; the live paper is never modified here.
    Publication still verifies the entire resulting audit inventory after overlay.
    """
    from .revision_evidence import _published_audit_hashes

    original_snapshot.relative_to(staged_dir)
    revision_files, revision_trees = _audit_bound_paths(revision_input_sha256)
    protected_files = set(revision_files)
    original_hashes = _published_audit_hashes(audit_input_sha256(staged_dir))
    _, original_trees = _audit_bound_paths(original_hashes)
    original_files, _ = _audit_bound_paths(
        {
            key: digest
            for key, digest in original_hashes.items()
            if not key.startswith("evidence-tree/")
        }
    )
    with ExitStack() as stack:
        source_fd = open_pinned_directory(staged_dir)
        stack.callback(os.close, source_fd)
        original_paths = set(original_files)
        inventory_budget = _Budget()
        for tree in original_trees:
            original_paths.update(_tree_files(source_fd, tree, inventory_budget))
        revision_fd = open_pinned_directory(revision_snapshot)
        stack.callback(os.close, revision_fd)
        snapshot_fd = open_pinned_directory(original_snapshot)
        stack.callback(os.close, snapshot_fd)
        os.mkdir("ancillary", mode=0o700, dir_fd=snapshot_fd)
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        archive_fd = os.open("ancillary", flags, dir_fd=snapshot_fd)
        stack.callback(os.close, archive_fd)
        budget = _Budget()
        for path in sorted(original_paths):
            if path in protected_files:
                continue
            if any(path.is_relative_to(tree) for tree in revision_trees):
                if _revision_tree_contains_file(revision_fd, path):
                    continue
            _archive_file(source_fd, archive_fd, path, budget)
