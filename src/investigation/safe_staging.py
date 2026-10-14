from __future__ import annotations

import argparse
import errno
import os
import secrets
import stat
import sys
from pathlib import Path, PurePath
from typing import NoReturn

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
_MAX_STAGED_FILE_BYTES = 64 * 1024 * 1024
_ALLOWED_FILE_MODES = {0o600, 0o644}


def _relative_parts(relative_path: str | os.PathLike[str]) -> tuple[str, ...]:
    path = PurePath(relative_path)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe staging path: {relative_path}")
    return path.parts


def _open_absolute_directory(path: str | os.PathLike[str]) -> int:
    absolute = Path(path).absolute()
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_or_create_directory(parent_fd: int, name: str) -> int:
    for _ in range(20):
        try:
            return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError:
            try:
                os.mkdir(name, mode=0o755, dir_fd=parent_fd)
            except FileExistsError:
                continue
        except OSError as error:
            if error.errno not in {errno.ELOOP, errno.ENOTDIR}:
                raise
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISLNK(metadata.st_mode):
                raise NotADirectoryError(f"staging ancestor is not a directory: {name}") from error
            try:
                os.unlink(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
    raise RuntimeError(f"staging ancestor changed too frequently: {name}")


def _open_parent(workspace_fd: int, parts: tuple[str, ...]) -> tuple[int, str]:
    parent_fd = os.dup(workspace_fd)
    try:
        for part in parts[:-1]:
            next_fd = _open_or_create_directory(parent_fd, part)
            os.close(parent_fd)
            parent_fd = next_fd
    except BaseException:
        os.close(parent_fd)
        raise
    return parent_fd, parts[-1]


def _remove_entry(parent_fd: int, name: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(metadata.st_mode):
        os.unlink(name, dir_fd=parent_fd)
        return

    try:
        child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except PermissionError:
        os.chmod(name, 0o700, dir_fd=parent_fd, follow_symlinks=False)
        child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    try:
        os.fchmod(child_fd, 0o700)
        for child in os.listdir(child_fd):
            _remove_entry(child_fd, child)
    finally:
        os.close(child_fd)
    try:
        os.rmdir(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def _evict_destination(parent_fd: int, name: str) -> str | None:
    quarantine = f".host-staging-old-{secrets.token_hex(12)}"
    try:
        os.rename(name, quarantine, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    return quarantine


def _cleanup_quarantines(parent_fd: int, quarantines: list[str]) -> None:
    for quarantine in quarantines:
        try:
            _remove_entry(parent_fd, quarantine)
        except (OSError, RecursionError):
            pass


def _publish(parent_fd: int, temporary: str, destination: str) -> None:
    quarantines: list[str] = []
    for _ in range(20):
        try:
            os.replace(
                temporary,
                destination,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            _cleanup_quarantines(parent_fd, quarantines)
            return
        except IsADirectoryError:
            quarantine = _evict_destination(parent_fd, destination)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR}:
                raise
            quarantine = _evict_destination(parent_fd, destination)
        if quarantine is not None:
            quarantines.append(quarantine)
    raise RuntimeError(f"staging destination changed too frequently: {destination}")


def _write_all(file_fd: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(file_fd, remaining)
        remaining = remaining[written:]


def _read_up_to(file_fd: int, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(file_fd, min(1024 * 1024, maximum_bytes + 1 - total)):
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum_bytes:
            raise ValueError(f"staged file exceeds {maximum_bytes} bytes")
    return b"".join(chunks)


def atomic_write_bytes(
    workspace: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
    content: bytes,
    *,
    mode: int = 0o644,
) -> None:
    workspace_fd = _open_absolute_directory(workspace)
    parent_fd = -1
    temporary = f".host-staging-new-{secrets.token_hex(12)}"
    try:
        parent_fd, destination = _open_parent(workspace_fd, _relative_parts(relative_path))
        file_fd = os.open(temporary, _FILE_FLAGS, mode, dir_fd=parent_fd)
        try:
            _write_all(file_fd, content)
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        _publish(parent_fd, temporary, destination)
    finally:
        if parent_fd >= 0:
            try:
                _remove_entry(parent_fd, temporary)
            finally:
                os.close(parent_fd)
        os.close(workspace_fd)


def atomic_copy_file(
    workspace: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
    source: str | os.PathLike[str],
    *,
    allow_source_symlinks: bool = False,
    maximum_bytes: int = _MAX_STAGED_FILE_BYTES,
    mode: int = 0o644,
) -> None:
    if not 0 < maximum_bytes <= _MAX_STAGED_FILE_BYTES:
        raise ValueError(f"copy size limit must be within 1..{_MAX_STAGED_FILE_BYTES}")
    if mode not in _ALLOWED_FILE_MODES:
        raise ValueError(f"unsupported staged file mode: {mode:o}")
    source_flags = os.O_RDONLY
    if not allow_source_symlinks:
        source_flags |= os.O_NOFOLLOW
    source_fd = os.open(source, source_flags)
    try:
        metadata = os.fstat(source_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"staging source is not a regular file: {source}")
        content = _read_up_to(source_fd, maximum_bytes)
    finally:
        os.close(source_fd)
    atomic_write_bytes(workspace, relative_path, content, mode=mode)


def atomic_refresh_file(
    workspace: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
) -> None:
    content = read_text(workspace, relative_path).encode()
    atomic_write_bytes(workspace, relative_path, content)


def ensure_directory(
    workspace: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
) -> None:
    workspace_fd = _open_absolute_directory(workspace)
    parent_fd = -1
    child_fd = -1
    try:
        parent_fd, name = _open_parent(workspace_fd, _relative_parts(relative_path))
        child_fd = _open_or_create_directory(parent_fd, name)
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        if parent_fd >= 0:
            os.close(parent_fd)
        os.close(workspace_fd)


def remove_path(
    workspace: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
) -> None:
    workspace_fd = _open_absolute_directory(workspace)
    parent_fd = -1
    try:
        parent_fd, name = _open_parent(workspace_fd, _relative_parts(relative_path))
        quarantine = _evict_destination(parent_fd, name)
        if quarantine is not None:
            _remove_entry(parent_fd, quarantine)
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
        os.close(workspace_fd)


def read_text(
    workspace: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
    *,
    maximum_bytes: int = 1024 * 1024,
) -> str:
    workspace_fd = _open_absolute_directory(workspace)
    parent_fd = -1
    try:
        parent_fd, name = _open_parent(workspace_fd, _relative_parts(relative_path))
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum_bytes:
                raise ValueError(f"unsafe staged input: {relative_path}")
            content = _read_up_to(file_fd, maximum_bytes)
        finally:
            os.close(file_fd)
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
        os.close(workspace_fd)
    return content.decode("utf-8")


def _copy_tree(source: Path, destination_fd: int, *, allow_source_symlinks: bool) -> None:
    for child in source.iterdir():
        metadata = child.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            if not allow_source_symlinks:
                raise ValueError(f"staging tree contains a symlink: {child}")
            os.symlink(os.readlink(child), child.name, dir_fd=destination_fd)
            continue
        if stat.S_ISDIR(metadata.st_mode):
            os.mkdir(child.name, stat.S_IMODE(metadata.st_mode), dir_fd=destination_fd)
            child_fd = os.open(child.name, _DIRECTORY_FLAGS, dir_fd=destination_fd)
            try:
                _copy_tree(
                    child,
                    child_fd,
                    allow_source_symlinks=allow_source_symlinks,
                )
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(metadata.st_mode):
            target_fd = os.open(
                child.name,
                _FILE_FLAGS,
                stat.S_IMODE(metadata.st_mode),
                dir_fd=destination_fd,
            )
            try:
                with child.open("rb") as source_file:
                    while chunk := source_file.read(1024 * 1024):
                        _write_all(target_fd, chunk)
            finally:
                os.close(target_fd)
        else:
            raise ValueError(f"staging tree contains a special file: {child}")


def atomic_copy_tree(
    workspace: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
    source: str | os.PathLike[str],
    *,
    allow_source_symlinks: bool = False,
) -> None:
    source_path = Path(source)
    if source_path.is_symlink():
        if not allow_source_symlinks:
            raise ValueError(f"staging source is a symlink: {source}")
        source_path = source_path.resolve(strict=True)
    if not source_path.is_dir():
        raise ValueError(f"staging source is not a directory: {source}")
    workspace_fd = _open_absolute_directory(workspace)
    parent_fd = -1
    temporary = f".host-staging-new-{secrets.token_hex(12)}"
    try:
        parent_fd, destination = _open_parent(workspace_fd, _relative_parts(relative_path))
        os.mkdir(temporary, mode=0o755, dir_fd=parent_fd)
        temporary_fd = os.open(temporary, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        try:
            _copy_tree(
                source_path,
                temporary_fd,
                allow_source_symlinks=allow_source_symlinks,
            )
        finally:
            os.close(temporary_fd)
        _publish(parent_fd, temporary, destination)
    finally:
        if parent_fd >= 0:
            try:
                _remove_entry(parent_fd, temporary)
            finally:
                os.close(parent_fd)
        os.close(workspace_fd)


def exec_with_log(
    workspace: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
    command: list[str],
) -> NoReturn:
    if not command:
        raise ValueError("log command is required")
    workspace_fd = _open_absolute_directory(workspace)
    parent_fd, destination = _open_parent(workspace_fd, _relative_parts(relative_path))
    temporary = f".host-staging-new-{secrets.token_hex(12)}"
    log_fd = os.open(temporary, _FILE_FLAGS, 0o644, dir_fd=parent_fd)
    _publish(parent_fd, temporary, destination)
    os.close(parent_fd)
    os.close(workspace_fd)
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    os.close(log_fd)
    os.execvp(command[0], command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in (
        "copy-file",
        "copy-tree",
        "write",
        "read",
        "refresh-file",
        "ensure-directory",
        "remove-path",
    ):
        subparser = subparsers.add_parser(operation)
        subparser.add_argument("workspace")
        subparser.add_argument("destination")
        if operation in {"copy-file", "copy-tree"}:
            subparser.add_argument("source")
            subparser.add_argument("--allow-source-symlinks", action="store_true")
        if operation == "copy-file":
            subparser.add_argument("--maximum-bytes", type=int, default=_MAX_STAGED_FILE_BYTES)
            subparser.add_argument("--mode", type=lambda value: int(value, 8), default=0o644)
        elif operation == "write":
            subparser.add_argument("content")
    execute = subparsers.add_parser("exec-log")
    execute.add_argument("workspace")
    execute.add_argument("destination")
    execute.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.operation == "copy-file":
        atomic_copy_file(
            arguments.workspace,
            arguments.destination,
            arguments.source,
            allow_source_symlinks=arguments.allow_source_symlinks,
            maximum_bytes=arguments.maximum_bytes,
            mode=arguments.mode,
        )
    elif arguments.operation == "copy-tree":
        atomic_copy_tree(
            arguments.workspace,
            arguments.destination,
            arguments.source,
            allow_source_symlinks=arguments.allow_source_symlinks,
        )
    elif arguments.operation == "write":
        atomic_write_bytes(arguments.workspace, arguments.destination, arguments.content.encode())
    elif arguments.operation == "read":
        sys.stdout.write(read_text(arguments.workspace, arguments.destination))
    elif arguments.operation == "refresh-file":
        atomic_refresh_file(arguments.workspace, arguments.destination)
    elif arguments.operation == "ensure-directory":
        ensure_directory(arguments.workspace, arguments.destination)
    elif arguments.operation == "remove-path":
        remove_path(arguments.workspace, arguments.destination)
    else:
        exec_with_log(arguments.workspace, arguments.destination, arguments.command)


if __name__ == "__main__":
    main()
