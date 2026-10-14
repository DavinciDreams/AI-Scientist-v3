"""Allocate short, private Linux IPC paths independently of bulk scratch storage."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import secrets
import stat
import sys

from src.investigation.temp_storage import _open_base, _private_child

SOCKET_RUNTIME_ROOT = Path("/dev/shm") / f"agentic-task-gates-{os.getuid()}"
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_LEAF_PATTERN = re.compile(r"gate-[0-9a-f]{16}\Z")
_MAX_SOCKET_PATH_BYTES = 107


def _validate_path(path: Path) -> None:
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path.parent != SOCKET_RUNTIME_ROOT
        or not _LEAF_PATTERN.fullmatch(path.name)
    ):
        raise ValueError("invalid task gate socket directory")
    if len(os.fsencode(path / "s")) > _MAX_SOCKET_PATH_BYTES:
        raise ValueError("task gate socket path exceeds 107 encoded bytes")


def _require_private_directory(descriptor: int) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("task gate directory must be owned by current user with mode 0700")
    return metadata


def create_socket_directory() -> Path:
    """Create one private leaf under the fixed per-user IPC root."""
    path = SOCKET_RUNTIME_ROOT / f"gate-{secrets.token_hex(8)}"
    _validate_path(path)
    parent_fd = _open_base(SOCKET_RUNTIME_ROOT.parent)
    try:
        root_fd = _private_child(parent_fd, SOCKET_RUNTIME_ROOT.name)
    finally:
        os.close(parent_fd)
    try:
        # mkdir is exclusive: a collision never adopts an existing leaf.
        os.mkdir(path.name, mode=0o700, dir_fd=root_fd)
        leaf_fd = os.open(path.name, _DIRECTORY_FLAGS, dir_fd=root_fd)
        try:
            _require_private_directory(leaf_fd)
        finally:
            os.close(leaf_fd)
    finally:
        os.close(root_fd)
    return path


def _socket_identity(leaf_fd: int) -> tuple[int, int]:
    metadata = os.stat("s", dir_fd=leaf_fd, follow_symlinks=False)
    if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ValueError("task gate socket must be an owned socket")
    return metadata.st_dev, metadata.st_ino


def identify_socket(path: Path) -> tuple[int, int]:
    """Capture the owned socket identity inside a validated private leaf."""
    _validate_path(path)
    root_fd = _open_base(SOCKET_RUNTIME_ROOT)
    try:
        _require_private_directory(root_fd)
        leaf_fd = os.open(path.name, _DIRECTORY_FLAGS, dir_fd=root_fd)
        try:
            _require_private_directory(leaf_fd)
            return _socket_identity(leaf_fd)
        finally:
            os.close(leaf_fd)
    finally:
        os.close(root_fd)


def _remove_leaf(root_fd: int, path: Path, socket_identity: tuple[int, int] | None) -> None:
    try:
        leaf_fd = os.open(path.name, _DIRECTORY_FLAGS, dir_fd=root_fd)
    except FileNotFoundError:
        return
    try:
        metadata = _require_private_directory(leaf_fd)
        entries = os.listdir(leaf_fd)
        if any(entry != "s" for entry in entries):
            raise ValueError("task gate socket directory contains unexpected files")
        if "s" in entries:
            if socket_identity is None or _socket_identity(leaf_fd) != socket_identity:
                raise ValueError("residual task gate socket requires its captured identity")
            os.unlink("s", dir_fd=leaf_fd)
        captured = os.stat(path.name, dir_fd=root_fd, follow_symlinks=False)
        if (captured.st_dev, captured.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError("task gate socket directory changed during cleanup")
        os.rmdir(path.name, dir_fd=root_fd)
    finally:
        os.close(leaf_fd)


def remove_socket_directory(path: Path, *, socket_identity: tuple[int, int] | None = None) -> None:
    """Remove an empty leaf; a residual socket needs its previously captured identity."""
    _validate_path(path)
    try:
        root_fd = _open_base(SOCKET_RUNTIME_ROOT)
    except FileNotFoundError:
        return
    try:
        _require_private_directory(root_fd)
        _remove_leaf(root_fd, path, socket_identity)
    finally:
        os.close(root_fd)


def _parse_socket_identity(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"[0-9]{1,20}:[0-9]{1,20}", value):
        raise argparse.ArgumentTypeError("socket identity must be decimal dev:ino")
    device, inode = (int(part) for part in value.split(":"))
    if device >= 2**64 or inode >= 2**64:
        raise argparse.ArgumentTypeError("socket identity exceeds unsigned 64-bit bounds")
    return device, inode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("create")
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--directory", type=Path, required=True)
    cleanup.add_argument("--socket-identity", type=_parse_socket_identity)
    identify = commands.add_parser("identify")
    identify.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "create":
            print(create_socket_directory())
        elif arguments.command == "identify":
            device, inode = identify_socket(arguments.directory)
            print(f"{device}:{inode}")
        else:
            remove_socket_directory(arguments.directory, socket_identity=arguments.socket_identity)
    except (OSError, ValueError) as error:
        print(f"task gate socket directory operation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
