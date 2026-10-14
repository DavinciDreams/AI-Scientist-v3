"""Descriptor-safe filesystem helpers for pipeline glue operations."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
import ctypes
import errno
import hashlib
import os
import secrets
import shutil
import stat
from pathlib import Path

from src.investigation.files import task_copy_entry_is_ignored


class UnsafeGluePathError(RuntimeError):
    """Raised when a glue copy would follow an unsafe filesystem entry."""


class AtomicNoReplaceUnsupported(UnsafeGluePathError):
    """Raised when a filesystem lacks atomic no-replace directory rename."""


def rename_directory_no_replace(
    directory_fd: int,
    source: str,
    destination: str,
    *,
    role: str,
) -> None:
    """Atomically publish a directory without replacing an existing entry."""
    rename_directory_between_no_replace(
        directory_fd,
        source,
        directory_fd,
        destination,
        role=role,
    )


def rename_directory_between_no_replace(
    source_directory_fd: int,
    source: str,
    destination_directory_fd: int,
    destination: str,
    *,
    role: str,
) -> None:
    """Atomically move a directory between pinned parents without replacement."""
    libc = ctypes.CDLL(None, use_errno=True)
    if _renameat2_no_replace(
        libc,
        source_directory_fd,
        source,
        destination_directory_fd,
        destination,
        role=role,
    ):
        return
    if _renameatx_np_no_replace(
        libc,
        source_directory_fd,
        source,
        destination_directory_fd,
        destination,
        role=role,
    ):
        return
    raise AtomicNoReplaceUnsupported(
        f"atomic no-replace publication is not supported for {role} on this platform"
    )


def _renameat2_no_replace(
    libc,
    source_directory_fd: int,
    source: str,
    destination_directory_fd: int,
    destination: str,
    *,
    role: str,
) -> bool:
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd,
        os.fsencode(source),
        destination_directory_fd,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise UnsafeGluePathError(f"{role} already exists: {destination}")
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
        return False
    raise UnsafeGluePathError(
        f"could not atomically publish {role}: {os.strerror(error_number)}"
    )


def _renameatx_np_no_replace(
    libc,
    source_directory_fd: int,
    source: str,
    destination_directory_fd: int,
    destination: str,
    *,
    role: str,
) -> bool:
    renameatx_np = getattr(libc, "renameatx_np", None)
    if renameatx_np is None:
        return False
    renameatx_np.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameatx_np.restype = ctypes.c_int
    result = renameatx_np(
        source_directory_fd,
        os.fsencode(source),
        destination_directory_fd,
        os.fsencode(destination),
        0x00000004,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise UnsafeGluePathError(f"{role} already exists: {destination}")
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
        return False
    raise UnsafeGluePathError(
        f"could not atomically publish {role}: {os.strerror(error_number)}"
    )


def tree_manifest(root: Path) -> dict[str, tuple[str, ...]]:
    manifest: dict[str, tuple[str, ...]] = {}
    for entry in sorted(root.rglob("*")):
        relative_path = entry.relative_to(root).as_posix()
        mode = entry.lstat().st_mode
        if stat.S_ISDIR(mode):
            manifest[relative_path] = ("directory",)
        elif stat.S_ISLNK(mode):
            manifest[relative_path] = ("symlink", os.readlink(entry))
        elif stat.S_ISREG(mode):
            with entry.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            manifest[relative_path] = ("file", digest)
        else:
            raise UnsafeGluePathError(f"task tree contains unsupported entry: {entry}")
    return manifest


def real_directory_exists(path: Path, *, role: str) -> bool:
    try:
        with open_directory(path, role=role):
            return True
    except FileNotFoundError:
        return False


def validate_tree_symlinks(
    root: Path,
    *,
    role: str,
    ignore_patterns: tuple[str, ...] = (),
) -> None:
    try:
        with open_directory(root, role=role) as directory_fd:
            validate_tree_symlinks_fd(
                directory_fd,
                relative_dir=Path(),
                role=role,
                ignore_patterns=ignore_patterns,
            )
    except FileNotFoundError:
        raise UnsafeGluePathError(f"{role} does not exist: {root}")


@contextmanager
def open_directory(path: Path, *, role: str) -> Iterator[int]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    absolute_path = Path(os.path.abspath(path))
    directory_fd = os.open("/", flags)
    try:
        for component in absolute_path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except FileNotFoundError:
        os.close(directory_fd)
        raise
    except OSError as exc:
        os.close(directory_fd)
        raise UnsafeGluePathError(f"could not safely open {role}: {path}") from exc
    try:
        yield directory_fd
    finally:
        os.close(directory_fd)


@contextmanager
def ensure_directory(path: Path, *, role: str) -> Iterator[int]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    absolute_path = Path(os.path.abspath(path))
    directory_fd = os.open("/", flags)
    try:
        for component in absolute_path.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                os.mkdir(component, 0o755, dir_fd=directory_fd)
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except OSError as exc:
        os.close(directory_fd)
        raise UnsafeGluePathError(f"could not safely create {role}: {path}") from exc
    try:
        yield directory_fd
    finally:
        os.close(directory_fd)


def directory_fd_path(directory_fd: int, *, role: str) -> Path:
    """Return an OS-provided path alias for an already-open directory."""
    expected = os.fstat(directory_fd)
    for base in (Path("/proc/self/fd"), Path("/dev/fd")):
        candidate = base / str(directory_fd)
        try:
            actual = candidate.stat()
        except OSError:
            continue
        if (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
            return candidate
    raise UnsafeGluePathError(f"could not address pinned {role}")


def validate_tree_symlinks_at(directory_fd: int, name: str, *, role: str) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        tree_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise UnsafeGluePathError(f"could not safely open {role}: {name}") from exc
    try:
        validate_tree_symlinks_fd(tree_fd, relative_dir=Path(), role=role)
    finally:
        os.close(tree_fd)


def validate_tree_symlinks_fd(
    directory_fd: int,
    *,
    relative_dir: Path,
    role: str,
    ignore_patterns: tuple[str, ...] = (),
) -> None:
    symlink_targets: dict[str, str] = {}
    _collect_validated_tree_entries(
        directory_fd,
        relative_dir=relative_dir,
        role=role,
        ignore_patterns=ignore_patterns,
        symlink_targets=symlink_targets,
    )
    _validate_symlink_targets(symlink_targets, role=role)


def _collect_validated_tree_entries(
    directory_fd: int,
    *,
    relative_dir: Path,
    role: str,
    ignore_patterns: tuple[str, ...],
    symlink_targets: dict[str, str],
) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_stat = os.fstat(directory_fd)
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as exc:
        display_path = relative_dir if relative_dir.parts else Path(".")
        raise UnsafeGluePathError(f"{role} changed while validating: {display_path}") from exc
    for name in names:
        relative_path = relative_dir / name
        try:
            entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            mode = entry_stat.st_mode
            is_directory = stat.S_ISDIR(mode)
            if task_copy_entry_is_ignored(
                name=name,
                relative_path=relative_path.as_posix(),
                is_directory=is_directory,
                patterns=ignore_patterns,
            ):
                continue
            if is_directory:
                child_fd = os.open(name, flags, dir_fd=directory_fd)
                try:
                    _collect_validated_tree_entries(
                        child_fd,
                        relative_dir=relative_path,
                        role=role,
                        ignore_patterns=ignore_patterns,
                        symlink_targets=symlink_targets,
                    )
                finally:
                    os.close(child_fd)
            elif stat.S_ISLNK(mode):
                target = os.readlink(name, dir_fd=directory_fd)
                current_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                _require_unchanged_validation_entry(
                    expected_stat=entry_stat,
                    current_stat=current_stat,
                    role=role,
                    relative_path=relative_path,
                )
                symlink_targets[relative_path.as_posix()] = target
            elif not stat.S_ISREG(mode):
                raise UnsafeGluePathError(f"{role} contains unsupported entry: {relative_path}")
        except UnsafeGluePathError:
            raise
        except OSError as exc:
            raise UnsafeGluePathError(f"{role} changed while validating: {relative_path}") from exc
    try:
        current_names = sorted(os.listdir(directory_fd))
        current_directory_stat = os.fstat(directory_fd)
    except OSError as exc:
        display_path = relative_dir if relative_dir.parts else Path(".")
        raise UnsafeGluePathError(f"{role} changed while validating: {display_path}") from exc
    display_path = relative_dir if relative_dir.parts else Path(".")
    if names != current_names:
        raise UnsafeGluePathError(f"{role} changed while validating: {display_path}")
    _require_unchanged_validation_entry(
        expected_stat=directory_stat,
        current_stat=current_directory_stat,
        role=role,
        relative_path=display_path,
    )


def _require_unchanged_validation_entry(
    *,
    expected_stat: os.stat_result,
    current_stat: os.stat_result,
    role: str,
    relative_path: Path,
) -> None:
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(expected_stat, field) != getattr(current_stat, field) for field in fields):
        raise UnsafeGluePathError(f"{role} changed while validating: {relative_path}")


def _validate_symlink_targets(symlink_targets: dict[str, str], *, role: str) -> None:
    for relative_path, target in symlink_targets.items():
        path = Path(relative_path)
        if os.path.isabs(target):
            raise UnsafeGluePathError(f"{role} contains absolute symlink: {path}")
        _resolve_symlink_target(path, target, symlink_targets, role=role)


def _resolve_symlink_target(
    relative_path: Path,
    target: str,
    symlink_targets: dict[str, str],
    *,
    role: str,
) -> None:
    resolved_parts = list(relative_path.parent.parts)
    pending_parts = deque(target.split("/"))
    expansions = 0
    while pending_parts:
        component = pending_parts.popleft()
        if component in {"", "."}:
            continue
        if component == "..":
            if not resolved_parts:
                raise UnsafeGluePathError(f"{role} contains escaping symlink: {relative_path}")
            resolved_parts.pop()
            continue
        candidate = "/".join((*resolved_parts, component))
        nested_target = symlink_targets.get(candidate)
        if nested_target is None:
            resolved_parts.append(component)
            continue
        expansions += 1
        if expansions > 40:
            raise UnsafeGluePathError(f"{role} contains invalid symlink: {relative_path}")
        if os.path.isabs(nested_target):
            raise UnsafeGluePathError(f"{role} contains absolute symlink: {Path(candidate)}")
        pending_parts.extendleft(reversed(nested_target.split("/")))


def regular_file_exists_at(directory_fd: int, name: str, *, role: str) -> bool:
    try:
        mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(mode):
        raise UnsafeGluePathError(f"{role} must be a regular file: {name}")
    return True


def entry_exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def regular_file_digest_at(directory_fd: int, name: str, *, role: str) -> str:
    try:
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as exc:
        raise UnsafeGluePathError(f"could not safely open {role}: {name}") from exc
    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise UnsafeGluePathError(f"{role} must be a regular file: {name}")
    except Exception:
        os.close(file_fd)
        raise
    with os.fdopen(file_fd, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def copy_regular_file_at(
    source_directory_fd: int,
    destination_directory_fd: int,
    name: str,
    *,
    replace_existing: bool = False,
) -> None:
    try:
        source_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=source_directory_fd)
    except OSError as exc:
        raise UnsafeGluePathError(f"could not safely open memory source: {name}") from exc

    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise UnsafeGluePathError(f"memory source must be a regular file: {name}")
        temporary_name = f".{name}.{secrets.token_hex(8)}.tmp"
        destination_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_directory_fd,
        )
    except OSError as exc:
        os.close(source_fd)
        raise UnsafeGluePathError(f"could not safely create memory destination: {name}") from exc
    except Exception:
        os.close(source_fd)
        raise

    try:
        with (
            os.fdopen(source_fd, "rb") as source_file,
            os.fdopen(destination_fd, "wb") as destination_file,
        ):
            shutil.copyfileobj(source_file, destination_file)
            destination_file.flush()
            os.fsync(destination_file.fileno())
        if replace_existing:
            os.replace(
                temporary_name,
                name,
                src_dir_fd=destination_directory_fd,
                dst_dir_fd=destination_directory_fd,
            )
        else:
            os.link(
                temporary_name,
                name,
                src_dir_fd=destination_directory_fd,
                dst_dir_fd=destination_directory_fd,
                follow_symlinks=False,
            )
    except OSError as exc:
        raise UnsafeGluePathError(f"could not safely publish memory destination: {name}") from exc
    finally:
        try:
            os.unlink(temporary_name, dir_fd=destination_directory_fd)
        except FileNotFoundError:
            pass
