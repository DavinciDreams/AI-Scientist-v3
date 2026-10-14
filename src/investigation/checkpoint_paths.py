"""Work-tree traversal and literal pathspec policy for checkpoints."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .checkpoint_security import SENSITIVE_DIRECTORY_NAMES
from .constants import TASK_COPY_EXCLUDE_DIRS

_PATHSPEC_FILENAME = "checkpoint-pathspec"
_SAFE_GIT_DIR_NAME = re.compile(r"[A-Za-z0-9._-]+\Z")
_GIT_DIR_NAME_ERROR = (
    "versioning.git_dir_name must be one non-empty path component using only "
    "letters, digits, dot, underscore, or hyphen"
)

_ALWAYS_EXCLUDED = (
    "**/.agent_home/**",
    "**/.agent_home",
    "**/.git/**",
    "**/.git",
    "**/*.lock",
)

_DERIVED_EXCLUDED = (
    "**/.*.runtime-build-*/**",
    "**/.*.tooling-install-*.cid",
    "**/.task.tmp.*/**",
    "**/.evaluation_pending",
    "**/.evaluation_stage_*/**",
)

_RUNTIME_BUILD_DIR = re.compile(r"^\..+\.runtime-build-.+$")
_TOOLING_INSTALL_CID = re.compile(r"^\..+\.tooling-install-.+\.cid$")


class PathSettings(Protocol):
    @property
    def git_dir_name(self) -> str: ...

    @property
    def max_file_bytes(self) -> int: ...

    @property
    def max_total_bytes(self) -> int: ...

    @property
    def final_include_environments(self) -> bool: ...

    @property
    def _deadline(self) -> float | None: ...


@dataclass(frozen=True)
class PinnedCheckpointPaths:
    root: Path
    git_dir: Path


class MissingCheckpointRoot(FileNotFoundError):
    pass


def validate_git_dir_name(git_dir_name: str) -> str:
    """Return a portable sidecar suffix or reject unsafe path syntax."""
    name = str(git_dir_name)
    if name in {".", ".."} or _SAFE_GIT_DIR_NAME.fullmatch(name) is None:
        raise ValueError(_GIT_DIR_NAME_ERROR)
    return name


@contextmanager
def pinned_checkpoint_paths(
    run_root: Path, *, git_dir_name: str
) -> Iterator[PinnedCheckpointPaths]:
    """Pin the work tree and sidecar parent without following path symlinks."""
    git_dir_name = validate_git_dir_name(git_dir_name)
    absolute = Path(os.path.abspath(run_root.expanduser()))
    if not absolute.name:
        raise ValueError("checkpoint run root cannot be the filesystem root")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open("/", flags)
    root_fd: int | None = None
    try:
        components = absolute.parts[1:]
        for index, component in enumerate(components):
            try:
                next_fd = os.open(component, flags, dir_fd=parent_fd)
            except FileNotFoundError as exc:
                raise MissingCheckpointRoot from exc
            if index == len(components) - 1:
                root_fd = next_fd
                break
            os.close(parent_fd)
            parent_fd = next_fd
        if root_fd is None:
            raise ValueError("checkpoint run root cannot be the filesystem root")
        parent_alias = _directory_fd_path(parent_fd)
        yield PinnedCheckpointPaths(
            root=_directory_fd_path(root_fd),
            git_dir=parent_alias / f".{absolute.name}{git_dir_name}",
        )
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _directory_fd_path(directory_fd: int) -> Path:
    expected = os.fstat(directory_fd)
    for base in (Path("/proc/self/fd"), Path("/dev/fd")):
        candidate = base / str(directory_fd)
        try:
            actual = candidate.stat()
        except OSError:
            continue
        if (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
            return candidate
    raise OSError("could not address pinned checkpoint directory")


def descriptor_aliases(path: Path) -> set[int]:
    """Return descriptors referenced by /proc/self/fd or /dev/fd paths."""
    parts = path.parts
    if parts[:4] == ("/", "proc", "self", "fd"):
        descriptor_index = 4
    elif parts[:3] == ("/", "dev", "fd"):
        descriptor_index = 3
    else:
        return set()
    if descriptor_index >= len(parts) or not parts[descriptor_index].isdigit():
        return set()
    descriptor = int(parts[descriptor_index])
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        return set()
    return {descriptor}


def scan_excluded_directory_names(*, settings: PathSettings, final: bool) -> frozenset[str]:
    names = {settings.git_dir_name, ".git"}
    if not (final and settings.final_include_environments):
        names.update(TASK_COPY_EXCLUDE_DIRS)
    return frozenset(names)


def is_checkpoint_sidecar_path(path: str, git_dir_name: str) -> bool:
    """Recognize sibling sidecar names at any work-tree depth."""
    return any(
        part.startswith(".") and part != git_dir_name and part.endswith(git_dir_name)
        for part in Path(path).parts
    )


def scan_oversize(
    root: Path,
    *,
    settings: PathSettings,
    final: bool,
) -> tuple[str, ...]:
    """Conservatively reject obviously oversized work-tree entries before staging."""
    if settings.max_file_bytes <= 0:
        return ()
    excluded = scan_excluded_directory_names(settings=settings, final=final)
    excluded = excluded | SENSITIVE_DIRECTORY_NAMES
    oversize: list[str] = []
    stack = [root]
    while stack:
        _require_scan_deadline(settings._deadline)
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            _require_scan_deadline(settings._deadline)
            if entry.is_dir(follow_symlinks=False):
                nested_sidecar = (
                    entry.name.startswith(".")
                    and entry.name != settings.git_dir_name
                    and entry.name.endswith(settings.git_dir_name)
                )
                if entry.name not in excluded and not nested_sidecar:
                    stack.append(Path(entry.path))
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
                continue
            if metadata.st_size > settings.max_file_bytes:
                oversize.append(Path(entry.path).relative_to(root).as_posix())
    return tuple(sorted(oversize))


def checkpoint_total_bytes(
    root: Path,
    *,
    settings: PathSettings,
    final: bool,
    sensitive: tuple[str, ...],
) -> int:
    """Return included work-tree bytes, stopping once the configured cap is exceeded."""
    sensitive_paths = set(sensitive)
    total_bytes = 0
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    stack: list[tuple[int, tuple[str, ...]]] = [(root_fd, ())]
    try:
        while stack:
            _require_scan_deadline(settings._deadline)
            directory_fd, prefix = stack.pop()
            try:
                entries = list(os.scandir(directory_fd))
                for entry in entries:
                    _require_scan_deadline(settings._deadline)
                    relative_parts = (*prefix, entry.name)
                    relative = "/".join(relative_parts)
                    if relative in sensitive_paths or is_checkpoint_excluded_path(
                        relative, settings=settings, final=final
                    ):
                        continue
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(metadata.st_mode):
                        child_fd = os.open(
                            entry.name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory_fd,
                        )
                        opened = os.fstat(child_fd)
                        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                            os.close(child_fd)
                            raise OSError("checkpoint tree changed during size validation")
                        stack.append((child_fd, relative_parts))
                        continue
                    if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
                        continue
                    if settings.max_file_bytes > 0 and metadata.st_size > settings.max_file_bytes:
                        continue
                    total_bytes += metadata.st_size
                    if settings.max_total_bytes > 0 and total_bytes > settings.max_total_bytes:
                        return total_bytes
            finally:
                os.close(directory_fd)
        return total_bytes
    finally:
        for directory_fd, _prefix in stack:
            os.close(directory_fd)


def is_checkpoint_excluded_path(
    relative: str, *, settings: PathSettings, final: bool
) -> bool:
    """Match the path classes omitted by the generated Git pathspec."""
    parts = Path(relative).parts
    if any(part in {settings.git_dir_name, ".git", ".agent_home"} for part in parts):
        return True
    if any(
        part.startswith(".")
        and part != settings.git_dir_name
        and part.endswith(settings.git_dir_name)
        for part in parts
    ):
        return True
    if parts and parts[-1].endswith(".lock"):
        return True
    if final and settings.final_include_environments:
        return False
    return any(
        part in TASK_COPY_EXCLUDE_DIRS
        or _RUNTIME_BUILD_DIR.fullmatch(part) is not None
        or part.startswith(".task.tmp.")
        or part.startswith(".evaluation_stage_")
        for part in parts
    ) or bool(
        parts
        and (
            parts[-1] == ".evaluation_pending"
            or _TOOLING_INSTALL_CID.fullmatch(parts[-1]) is not None
        )
    )


def _require_scan_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise subprocess.TimeoutExpired("checkpoint tree scan", 0)


def write_pathspec(
    storage_dir: Path,
    *,
    settings: PathSettings,
    final: bool,
    oversize: tuple[str, ...],
    sensitive: tuple[str, ...],
) -> Path:
    """Write a NUL-separated pathspec outside the agent work tree."""
    entries = _pathspec_entries(
        settings=settings, final=final, oversize=oversize, sensitive=sensitive
    )
    pathspec = storage_dir / _PATHSPEC_FILENAME
    pathspec.write_bytes(b"\0".join(entry.encode("utf-8") for entry in entries) + b"\0")
    return pathspec


def _pathspec_entries(
    *,
    settings: PathSettings,
    final: bool,
    oversize: tuple[str, ...],
    sensitive: tuple[str, ...],
) -> list[str]:
    entries = [
        ".",
        f":(exclude,literal){settings.git_dir_name}",
        f":(exclude,glob){settings.git_dir_name}/**",
    ]
    nested_sidecar = f".*{_escape_git_glob(settings.git_dir_name)}"
    entries.extend(
        (
            f":(exclude,glob)**/{nested_sidecar}",
            f":(exclude,glob)**/{nested_sidecar}/**",
        )
    )
    entries.extend(f":(exclude,glob){pattern}" for pattern in _ALWAYS_EXCLUDED)
    if not (final and settings.final_include_environments):
        entries.extend(f":(exclude,glob)**/{name}/**" for name in sorted(TASK_COPY_EXCLUDE_DIRS))
        entries.extend(f":(exclude,glob){pattern}" for pattern in _DERIVED_EXCLUDED)
    entries.extend(f":(exclude,literal){path}" for path in oversize)
    entries.extend(f":(exclude,literal){path}" for path in sensitive)
    return entries


def _escape_git_glob(value: str) -> str:
    return "".join(f"\\{character}" if character in "\\*?[" else character for character in value)
