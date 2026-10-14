"""Select portable scratch storage without exposing host control files to agents."""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

LOCAL_SSD = Path("/mnt/localssd")
_INHERITED_NAMES = ("AGENTIC_TEMP_DIR", "AGENTIC_CACHE_ROOT", "AGENTIC_CONTAINER_TEMP_ROOT")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


@dataclass(frozen=True)
class TempStorage:
    base: Path
    temporary: Path
    cache: Path
    containers: Path


def _absolute_path(value: str) -> Path:
    if not value or any(character in value for character in "\n\r\0"):
        raise ValueError("temporary storage path must be nonempty and contain no control lines")
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("temporary storage path must be absolute without parent traversal")
    return path


def _open_base(base: Path) -> int:
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for part in base.parts[1:]:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as error:
                if error.errno in (20, 40):
                    raise ValueError(
                        f"temporary storage path contains a symlink or nondirectory: {base}"
                    ) from error
                raise
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            mode = stat.S_IMODE(metadata.st_mode)
            if metadata.st_uid not in (0, os.getuid()) or (
                mode & 0o022 and not mode & stat.S_ISVTX
            ):
                raise ValueError(
                    f"unsafe writable or foreign-owned temporary storage ancestor: {base}"
                )
        if not os.access(base, os.W_OK | os.X_OK):
            raise ValueError(f"temporary storage base must be writable: {base}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_base(base: Path) -> None:
    os.close(_open_base(base))


def _is_scratch_mount(base: Path) -> bool:
    return os.path.ismount(base) and base.stat().st_dev != Path("/").stat().st_dev


def _system_temporary_base() -> Path:
    # Resolve platform aliases (e.g. macOS /var); explicit settings reject aliases.
    for name in ("TMP", "TEMP"):
        if os.environ.get(name):
            return _absolute_path(os.environ[name]).resolve(strict=True)
    return Path(tempfile.gettempdir()).resolve(strict=True)


def _select_base() -> Path:
    for name in ("AGENTIC_TEMP_ROOT", "TMPDIR"):
        if name in os.environ:
            base = _absolute_path(os.environ[name])
            _validate_base(base)
            return base
    try:
        _validate_base(LOCAL_SSD)
        if _is_scratch_mount(LOCAL_SSD):
            return LOCAL_SSD
    except (OSError, ValueError):
        pass
    base = _system_temporary_base()
    _validate_base(base)
    return base


def _private_child(parent: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent)
    except FileExistsError:
        pass
    try:
        child = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
    except OSError as error:
        raise ValueError(f"unsafe private temporary storage directory: {name}") from error
    metadata = os.fstat(child)
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        os.close(child)
        raise ValueError(
            f"temporary storage directory must be private and owned by current user: {name}"
        )
    return child


def _namespace(base: Path) -> Path:
    return base / f"agentic-scientist-{os.getuid()}"


def _create_storage(base: Path) -> TempStorage:
    namespace = _namespace(base)
    invocation = f"run-{uuid.uuid4().hex}"
    paths = (
        namespace / "temp" / invocation,
        namespace / "cache",
        namespace / "containers" / invocation,
    )
    base_fd = _open_base(base)
    try:
        namespace_fd = _private_child(base_fd, namespace.name)
        try:
            for path in paths:
                descriptor = os.dup(namespace_fd)
                try:
                    for part in path.relative_to(namespace).parts:
                        child = _private_child(descriptor, part)
                        os.close(descriptor)
                        descriptor = child
                finally:
                    os.close(descriptor)
        finally:
            os.close(namespace_fd)
    finally:
        os.close(base_fd)
    return TempStorage(base, *paths)


def _inherited_storage() -> TempStorage | None:
    if not any(name in os.environ for name in _INHERITED_NAMES):
        return None
    if not all(name in os.environ for name in ("AGENTIC_TEMP_ROOT", *_INHERITED_NAMES)):
        raise ValueError("incomplete inherited temporary storage configuration")
    base = _absolute_path(os.environ["AGENTIC_TEMP_ROOT"])
    temporary, cache, containers = (
        _absolute_path(os.environ[name]) for name in _INHERITED_NAMES
    )
    namespace = _namespace(base)
    invocation = temporary.name
    if (
        temporary.parent != namespace / "temp"
        or cache != namespace / "cache"
        or containers != namespace / "containers" / invocation
        or not invocation.startswith("run-")
        or len(invocation) != 36
        or any(char not in "0123456789abcdef" for char in invocation[4:])
    ):
        raise ValueError("invalid inherited temporary storage namespace")
    for path in (temporary, cache, containers):
        _validate_base(path)
        directories: tuple[Path, ...] = (namespace, path)
        if path != cache:
            directories = (namespace, path.parent, path)
        for directory in directories:
            metadata = directory.lstat()
            if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
                raise ValueError(
                    "inherited temporary storage must be private and owned by current user"
                )
    return TempStorage(base, temporary, cache, containers)


def _export(storage: TempStorage) -> TempStorage:
    os.environ.update({
        "AGENTIC_TEMP_ROOT": str(storage.base),
        "AGENTIC_TEMP_DIR": str(storage.temporary),
        "AGENTIC_CACHE_ROOT": str(storage.cache),
        "AGENTIC_CONTAINER_TEMP_ROOT": str(storage.containers),
        "TMPDIR": str(storage.temporary), "TMP": str(storage.temporary),
        "TEMP": str(storage.temporary),
    })
    # tempfile caches its initial directory; entrypoint initialization may follow imports.
    tempfile.tempdir = str(storage.temporary)
    return storage


def initialize_temp_storage() -> TempStorage:
    """Initialize once per invocation; children validate and reuse inherited storage."""
    return _export(_inherited_storage() or _create_storage(_select_base()))


def preview_temp_storage() -> str:
    """Describe selection without creating directories or modifying environment."""
    inherited = _inherited_storage()
    base = inherited.base if inherited else _select_base()
    return f"temporary storage base={base} free_bytes={shutil.disk_usage(base).free}"


def temporary_directory() -> Path:
    """Preserve active legacy callers; configured callers use managed storage."""
    if "AGENTIC_TEMP_ROOT" in os.environ or any(name in os.environ for name in _INHERITED_NAMES):
        return initialize_temp_storage().temporary
    return Path("/tmp")


def runtime_cache_directory() -> Path:
    """Stable across phases; pre-initialization callers retain their legacy caches."""
    if "AGENTIC_TEMP_ROOT" in os.environ or any(name in os.environ for name in _INHERITED_NAMES):
        return initialize_temp_storage().cache
    return Path("/tmp")


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        storage = initialize_temp_storage()
        print(
            f"[temp-storage] base={storage.base} device={storage.base.stat().st_dev} "
            f"free_bytes={shutil.disk_usage(storage.base).free}", file=sys.stderr,
        )
        print(storage.base, storage.temporary, storage.cache, storage.containers, sep="\n")
    except (OSError, ValueError) as error:
        print(f"temporary storage initialization failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
