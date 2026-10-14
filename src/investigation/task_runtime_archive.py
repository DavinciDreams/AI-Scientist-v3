"""Publish bounded production baseline evidence after isolated workers stop."""
from __future__ import annotations

import os
from pathlib import Path
import secrets
import shutil
import stat

from src.pipeline.safe_fs import UnsafeGluePathError, rename_directory_no_replace
from .pinned_files import absolute_path, open_directory_tree_no_follow, read_regular_at

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 4096


def validate_baseline_results_dir(destination: Path, *, task_dir: Path) -> Path:
    """Require a new destination outside the task, with an existing safe parent."""
    destination = absolute_path(destination)
    if destination.is_relative_to(task_dir.resolve()):
        raise ValueError("baseline archive must be outside the task directory")
    parent_fd = open_directory_tree_no_follow(destination.parent)
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return destination
        raise ValueError("baseline archive destination already exists")
    finally:
        os.close(parent_fd)


def _copy_tree(source_fd: int, target_fd: int, budget: list[int], depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("baseline archive nesting exceeds limit")
    for name in os.listdir(source_fd):
        budget[1] -= 1
        if budget[1] < 0:
            raise ValueError("baseline archive entry count exceeds limit")
        info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            child_fd = os.open(name, flags, dir_fd=source_fd)
            try:
                os.mkdir(name, mode=0o700, dir_fd=target_fd)
                output_fd = os.open(name, flags, dir_fd=target_fd)
                try:
                    _copy_tree(child_fd, output_fd, budget, depth + 1)
                finally:
                    os.close(output_fd)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(info.st_mode):
            content = read_regular_at(source_fd, name, max_bytes=budget[0])
            budget[0] -= len(content)
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=target_fd)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            raise ValueError("baseline archive contains a non-regular entry")
    os.fsync(target_fd)


def archive_baseline_results(source: Path, destination: Path, *, task_dir: Path) -> None:
    """Atomically retain a complete private result tree; never replace prior evidence."""
    destination = validate_baseline_results_dir(destination, task_dir=task_dir)
    parent_fd = open_directory_tree_no_follow(destination.parent)
    staging = f".baseline-archive-{secrets.token_hex(16)}"
    created = False
    try:
        os.mkdir(staging, mode=0o700, dir_fd=parent_fd)
        created = True
        target_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=parent_fd)
        try:
            source_fd = open_directory_tree_no_follow(source)
            try:
                _copy_tree(source_fd, target_fd, [MAX_ARCHIVE_BYTES, MAX_ARCHIVE_ENTRIES])
            finally:
                os.close(source_fd)
        finally:
            os.close(target_fd)
        try:
            rename_directory_no_replace(parent_fd, staging, destination.name,
                                        role="baseline result archive")
        except UnsafeGluePathError as exc:
            raise OSError(str(exc)) from exc
        created = False
        os.fsync(parent_fd)
    finally:
        if created:
            shutil.rmtree(staging, dir_fd=parent_fd)
        os.close(parent_fd)
