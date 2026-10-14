"""Descriptor-relative cleanup for publication trees selected by inode."""

from __future__ import annotations

import os
import secrets
import stat


class PinnedDeletionError(RuntimeError):
    """The selected directory changed before or during deletion."""


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _open_directory_at(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(name, flags, dir_fd=parent_fd)


def _require_identity(parent_fd: int, name: str, expected: tuple[int, int]) -> None:
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _identity(current) != expected:
        raise PinnedDeletionError("pinned directory changed")


def _quarantine_entry(parent_fd: int, name: str) -> str:
    for _ in range(100):
        quarantine = f".{name}.delete-{secrets.token_hex(8)}"
        try:
            os.stat(quarantine, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            os.rename(name, quarantine, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            return quarantine
    raise PinnedDeletionError("could not quarantine pinned entry")


def _remove_contents(directory_fd: int) -> None:
    for name in os.listdir(directory_fd):
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        expected = _identity(metadata)
        quarantine = _quarantine_entry(directory_fd, name)
        _require_identity(directory_fd, quarantine, expected)
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = _open_directory_at(directory_fd, quarantine)
            try:
                if _identity(os.fstat(child_fd)) != expected:
                    raise PinnedDeletionError("pinned directory changed")
                _remove_contents(child_fd)
                _require_identity(directory_fd, quarantine, expected)
                os.rmdir(quarantine, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
        else:
            os.unlink(quarantine, dir_fd=directory_fd)


def remove_pinned_directory_at(
    parent_fd: int,
    name: str,
    *,
    expected_device: int,
    expected_inode: int,
    quarantine_name: str | None = None,
) -> bool:
    """Delete one directory only while its opened inode matches the expected inode."""
    expected = (expected_device, expected_inode)
    selected_name = name
    try:
        directory_fd = _open_directory_at(parent_fd, selected_name)
    except FileNotFoundError:
        if quarantine_name is None:
            return False
        selected_name = quarantine_name
        try:
            directory_fd = _open_directory_at(parent_fd, selected_name)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise PinnedDeletionError("pinned directory changed") from exc
    except OSError as exc:
        raise PinnedDeletionError("pinned directory changed") from exc
    try:
        if _identity(os.fstat(directory_fd)) != expected:
            raise PinnedDeletionError("pinned directory changed")
    finally:
        os.close(directory_fd)
    if selected_name == name:
        if quarantine_name is None:
            quarantine = _quarantine_entry(parent_fd, name)
        else:
            try:
                os.stat(quarantine_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                os.rename(
                    name,
                    quarantine_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                quarantine = quarantine_name
            else:
                raise PinnedDeletionError("pinned quarantine already exists")
    else:
        quarantine = selected_name
    _require_identity(parent_fd, quarantine, expected)
    directory_fd = _open_directory_at(parent_fd, quarantine)
    try:
        if _identity(os.fstat(directory_fd)) != expected:
            raise PinnedDeletionError("pinned directory changed")
        _remove_contents(directory_fd)
        _require_identity(parent_fd, quarantine, expected)
        os.rmdir(quarantine, dir_fd=parent_fd)
    finally:
        os.close(directory_fd)
    return True
