"""Private, run-local cache for installed drafted-task runtimes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .temp_storage import runtime_cache_directory

_CACHE_SCHEMA_VERSION = 1
_CACHE_MANIFEST = "install-cache.json"
_CACHE_ROOT_PREFIX = "agentic-task-runtime-cache-"
_BUILD_ROOT_PREFIX = ".build-"
_COPY_CHUNK_BYTES = 1024 * 1024
_EVICTION_TOMBSTONE_PREFIX = ".evict-"
_RECOVERY_ROOT_PREFIX = ".recovery-"
_OWNED_RECOVERY_MANIFEST = ".owned-runtime-recovery.json"
_OWNED_RECOVERY_SCHEMA_VERSION = 1


class TaskRuntimeInstallCacheDeadlineExceeded(RuntimeError):
    """Cache publication exceeded its caller-owned monotonic deadline."""


def task_runtime_install_cache_root(task_dir: Path) -> Path:
    """Return the private host cache shared by phases for one run-local task."""
    identity = str(task_dir.expanduser().resolve(strict=False)).encode("utf-8")
    return runtime_cache_directory() / f"{_CACHE_ROOT_PREFIX}{hashlib.sha256(identity).hexdigest()}"


def prepare_task_runtime_install_cache(task_dir: Path, *, reset: bool) -> Path:
    """Create or validate a private cache root without accepting unsafe stale state."""
    root = task_runtime_install_cache_root(task_dir)
    if root.exists() or root.is_symlink():
        TaskRuntimeInstallCache._validate_private_directory(root, "install cache root")
        if reset:
            shutil.rmtree(root)
    if not root.exists():
        root.mkdir(mode=0o700)
        root.chmod(0o700)
    TaskRuntimeInstallCache._validate_private_directory(root, "install cache root")
    return root


def discard_task_runtime_install_cache(task_dir: Path) -> None:
    """Remove only the deterministic owned cache for one task."""
    root = task_runtime_install_cache_root(task_dir)
    if not root.exists() and not root.is_symlink():
        return
    TaskRuntimeInstallCache._validate_private_directory(root, "install cache root")
    shutil.rmtree(root)


@dataclass(frozen=True, slots=True)
class TaskRuntimeInstallCacheKey:
    image_id: str
    requirements_sha256: str
    tooling_sha256: str

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            {
                "image_id": self.image_id,
                "requirements_sha256": self.requirements_sha256,
                "schema_version": _CACHE_SCHEMA_VERSION,
                "tooling_sha256": self.tooling_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "key": self.digest,
            "requirements_sha256": self.requirements_sha256,
            "schema_version": _CACHE_SCHEMA_VERSION,
            "tooling_sha256": self.tooling_sha256,
        }


class TaskRuntimeInstallCache:
    """Own at most two complete venv installs outside the agent workspace."""

    def __init__(self, root: Path, *, max_entries: int = 2) -> None:
        if max_entries < 1:
            raise ValueError("task runtime install cache must retain at least one entry")
        self.root = root.resolve(strict=True)
        self.max_entries = max_entries
        self._validate_private_directory(self.root, "install cache root")
        self._lock_path = self.root / ".lock"

    def lookup(
        self,
        key: TaskRuntimeInstallCacheKey,
        *,
        deadline: float | None = None,
    ) -> Path | None:
        with self._locked(deadline=deadline):
            self._raise_if_deadline_exceeded(deadline)
            entry = self.root / key.digest
            if not entry.exists() and not entry.is_symlink():
                return None
            self._validate_entry(entry, key, deadline=deadline)
            self._raise_if_deadline_exceeded(deadline)
            os.utime(entry, follow_symlinks=False)
            return entry

    def create_build_root(self, *, deadline: float | None = None) -> Path:
        """Create private staging on the cache filesystem for an owned runtime build."""
        with self._locked(deadline=deadline):
            self._raise_if_deadline_exceeded(deadline)
            self._validate_private_directory(self.root, "install cache root")
            build_root = self.root / f"{_BUILD_ROOT_PREFIX}{uuid.uuid4().hex}"
            build_root.mkdir(mode=0o700)
            self._validate_private_directory(build_root, "task runtime cache-owned build root")
            self._sync_directory(self.root)
            return build_root

    def store(
        self,
        key: TaskRuntimeInstallCacheKey,
        source_venv: Path,
        *,
        deadline: float | None = None,
    ) -> Path:
        """Copy one validated runtime into cache-local staging, then publish atomically."""
        self._raise_if_deadline_exceeded(deadline)
        self._validate_runtime_tree(source_venv, deadline=deadline)
        self._raise_if_deadline_exceeded(deadline)
        with self._locked(deadline=deadline):
            self._raise_if_deadline_exceeded(deadline)
            entry = self.root / key.digest
            if entry.exists() or entry.is_symlink():
                self._validate_entry(entry, key, deadline=deadline)
                self._raise_if_deadline_exceeded(deadline)
                os.utime(entry, follow_symlinks=False)
                return entry
            staging = self.root / f".{key.digest}.staging-{uuid.uuid4().hex}"
            staging.mkdir(mode=0o700)
            published_entry = False
            try:
                self._copy_tree_before_deadline(
                    source_venv,
                    staging / ".venv",
                    deadline=deadline,
                )
                self._raise_if_deadline_exceeded(deadline)
                self._write_manifest(staging / _CACHE_MANIFEST, key.manifest())
                self._validate_entry(staging, key, deadline=deadline)
                self._raise_if_deadline_exceeded(deadline)
                staging.replace(entry)
                published_entry = True
                self._evict(exclude=entry, deadline=deadline)
                self._raise_if_deadline_exceeded(deadline)
                return entry
            except BaseException:
                if published_entry and entry.exists():
                    tombstone = self._hide_cache_entry(entry)
                    try:
                        shutil.rmtree(tombstone)
                    except OSError:
                        pass
                raise
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

    def store_owned(
        self,
        key: TaskRuntimeInstallCacheKey,
        source_venv: Path,
        *,
        deadline: float | None = None,
    ) -> Path:
        """Move a cache-local staged runtime into the cache, restoring it on failure."""
        source_venv = self._validate_owned_staged_runtime(source_venv, deadline=deadline)
        with self._locked(deadline=deadline):
            self._raise_if_deadline_exceeded(deadline)
            entry = self.root / key.digest
            if entry.exists() or entry.is_symlink():
                self._validate_entry(entry, key, deadline=deadline)
                self._raise_if_deadline_exceeded(deadline)
                os.utime(entry, follow_symlinks=False)
                return entry
            staging = self.root / f".{key.digest}.staging-{uuid.uuid4().hex}"
            staging.mkdir(mode=0o700)
            self._sync_directory(self.root)
            published_entry = False
            owned_runtime_moved = False
            recovery_root = staging
            cleanup_staging = True
            try:
                self._write_owned_recovery_record(
                    staging,
                    key=key,
                    source_build_root=source_venv.parent.name,
                )
                source_venv.replace(staging / ".venv")
                owned_runtime_moved = True
                self._sync_directory(source_venv.parent)
                self._sync_directory(staging)
                self._raise_if_deadline_exceeded(deadline)
                self._write_manifest(staging / _CACHE_MANIFEST, key.manifest())
                self._validate_entry(staging, key, deadline=deadline)
                self._raise_if_deadline_exceeded(deadline)
                staging.replace(entry)
                published_entry = True
                recovery_root = entry
                self._sync_directory(self.root)
                self._evict(exclude=entry, deadline=deadline)
                self._raise_if_deadline_exceeded(deadline)
                self._sync_directory(entry)
                (entry / _OWNED_RECOVERY_MANIFEST).unlink()
                return entry
            except BaseException as publication_error:
                if not owned_runtime_moved:
                    raise
                try:
                    if published_entry:
                        recovery_candidate = self.root / (
                            f"{_RECOVERY_ROOT_PREFIX}{entry.name}-{uuid.uuid4().hex}"
                        )
                        entry.replace(recovery_candidate)
                        recovery_root = recovery_candidate
                        self._sync_directory(self.root)
                    if source_venv.exists() or source_venv.is_symlink():
                        raise OSError(f"runtime rollback destination exists: {source_venv}")
                    (recovery_root / ".venv").replace(source_venv)
                    self._sync_directory(source_venv.parent)
                    self._sync_directory(recovery_root)
                except OSError as rollback_error:
                    cleanup_staging = False
                    raise OSError(
                        "task runtime cache rollback failed; recovery state retained at "
                        f"{recovery_root}: {rollback_error}"
                    ) from publication_error
                if recovery_root.exists():
                    shutil.rmtree(recovery_root, ignore_errors=True)
                raise
            finally:
                if cleanup_staging and staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _copy_tree_before_deadline(
        source: Path,
        destination: Path,
        *,
        deadline: float | None,
    ) -> None:
        TaskRuntimeInstallCache._raise_if_deadline_exceeded(deadline)
        destination.mkdir()
        with os.scandir(source) as entries:
            for entry in entries:
                TaskRuntimeInstallCache._raise_if_deadline_exceeded(deadline)
                source_path = Path(entry.path)
                destination_path = destination / entry.name
                if entry.is_symlink():
                    os.symlink(os.readlink(source_path), destination_path)
                    shutil.copystat(
                        source_path,
                        destination_path,
                        follow_symlinks=False,
                    )
                elif entry.is_dir(follow_symlinks=False):
                    TaskRuntimeInstallCache._copy_tree_before_deadline(
                        source_path,
                        destination_path,
                        deadline=deadline,
                    )
                elif entry.is_file(follow_symlinks=False):
                    TaskRuntimeInstallCache._copy_file_before_deadline(
                        str(source_path),
                        str(destination_path),
                        deadline=deadline,
                    )
                else:
                    raise ValueError(f"unsafe cached runtime entry: {source_path}")
        TaskRuntimeInstallCache._raise_if_deadline_exceeded(deadline)
        shutil.copystat(source, destination, follow_symlinks=False)

    @staticmethod
    def _copy_file_before_deadline(
        source: str,
        destination: str,
        *,
        deadline: float | None,
    ) -> str:
        with open(source, "rb") as source_stream, open(destination, "xb") as destination_stream:
            while True:
                TaskRuntimeInstallCache._raise_if_deadline_exceeded(deadline)
                chunk = source_stream.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                destination_stream.write(chunk)
        shutil.copystat(source, destination, follow_symlinks=False)
        return destination

    @staticmethod
    def _raise_if_deadline_exceeded(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TaskRuntimeInstallCacheDeadlineExceeded(
                "task runtime install cache publication deadline exceeded"
            )

    def materialize(
        self,
        key: TaskRuntimeInstallCacheKey,
        destination_venv: Path,
        *,
        deadline: float | None = None,
    ) -> None:
        """Atomically copy a private install into an untrusted task workspace."""
        self._raise_if_deadline_exceeded(deadline)
        if destination_venv.exists() or destination_venv.is_symlink():
            raise ValueError(f"runtime publication destination already exists: {destination_venv}")
        staging = destination_venv.parent / (
            f".{destination_venv.name}.materialize-{uuid.uuid4().hex}"
        )
        with self._locked(deadline=deadline):
            self._raise_if_deadline_exceeded(deadline)
            entry = self.root / key.digest
            self._validate_entry(entry, key, deadline=deadline)
            try:
                os.utime(entry, follow_symlinks=False)
                self._copy_tree_before_deadline(
                    entry / ".venv",
                    staging,
                    deadline=deadline,
                )
                self._validate_runtime_tree(staging, deadline=deadline)
                self._raise_if_deadline_exceeded(deadline)
                staging.replace(destination_venv)
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

    @contextmanager
    def _locked(self, *, deadline: float | None = None) -> Iterator[None]:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self._lock_path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "r+", encoding="ascii") as stream:
                if deadline is None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                else:
                    while True:
                        self._raise_if_deadline_exceeded(deadline)
                        try:
                            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                self._raise_if_deadline_exceeded(deadline)
                            time.sleep(min(0.05, remaining))
                self._reconcile_owned_staging(deadline=deadline)
                yield
        finally:
            pass

    def _evict(self, *, exclude: Path, deadline: float | None = None) -> None:
        self._reap_eviction_tombstones(deadline=deadline)
        entries: list[Path] = []
        for path in self.root.iterdir():
            self._raise_if_deadline_exceeded(deadline)
            if path == self._lock_path or path.name.startswith("."):
                continue
            if not path.is_dir() or path.is_symlink():
                raise ValueError(f"unsafe install cache entry: {path}")
            self._validate_entry(path, deadline=deadline)
            entries.append(path)
        excess = len(entries) - self.max_entries
        if excess <= 0:
            return
        candidates = sorted(
            (path for path in entries if path != exclude),
            key=lambda path: path.stat().st_mtime_ns,
        )
        for path in candidates[:excess]:
            self._raise_if_deadline_exceeded(deadline)
            tombstone = self._hide_cache_entry(path)
            self._remove_tree_before_deadline(tombstone, deadline=deadline)

    def _hide_cache_entry(self, entry: Path) -> Path:
        tombstone = self.root / (
            f"{_EVICTION_TOMBSTONE_PREFIX}{entry.name}-{uuid.uuid4().hex}"
        )
        entry.replace(tombstone)
        return tombstone

    def _reap_eviction_tombstones(self, *, deadline: float | None) -> None:
        for path in self.root.iterdir():
            self._raise_if_deadline_exceeded(deadline)
            if not path.name.startswith(_EVICTION_TOMBSTONE_PREFIX):
                continue
            self._validate_private_directory(path, "install cache eviction tombstone")
            self._remove_tree_before_deadline(path, deadline=deadline)

    @classmethod
    def _remove_tree_before_deadline(
        cls,
        path: Path,
        *,
        deadline: float | None,
    ) -> None:
        cls._raise_if_deadline_exceeded(deadline)
        with os.scandir(path) as entries:
            for entry in entries:
                cls._raise_if_deadline_exceeded(deadline)
                entry_path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    cls._remove_tree_before_deadline(entry_path, deadline=deadline)
                else:
                    entry_path.unlink()
        cls._raise_if_deadline_exceeded(deadline)
        path.rmdir()

    def _validate_entry(
        self,
        entry: Path,
        expected_key: TaskRuntimeInstallCacheKey | None = None,
        *,
        deadline: float | None = None,
    ) -> None:
        self._raise_if_deadline_exceeded(deadline)
        self._validate_private_directory(entry, "install cache entry")
        if len(entry.name) != 64 or any(char not in "0123456789abcdef" for char in entry.name):
            if not entry.name.startswith("."):
                raise ValueError(f"invalid install cache key path: {entry}")
        manifest_path = entry / _CACHE_MANIFEST
        self._validate_regular_file(manifest_path, "install cache manifest")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid install cache manifest: {manifest_path}") from exc
        if not isinstance(payload, dict) or set(payload) != set(
            TaskRuntimeInstallCacheKey("", "", "").manifest()
        ):
            raise ValueError(f"invalid install cache manifest: {manifest_path}")
        image_id = payload.get("image_id")
        requirements_sha256 = payload.get("requirements_sha256")
        tooling_sha256 = payload.get("tooling_sha256")
        if not all(
            isinstance(value, str) for value in (image_id, requirements_sha256, tooling_sha256)
        ):
            raise ValueError(f"invalid install cache manifest: {manifest_path}")
        assert isinstance(image_id, str)
        assert isinstance(requirements_sha256, str)
        assert isinstance(tooling_sha256, str)
        key = TaskRuntimeInstallCacheKey(image_id, requirements_sha256, tooling_sha256)
        if (
            payload != key.manifest()
            or entry.name.lstrip(".").split(".staging-", 1)[0] != key.digest
        ):
            raise ValueError(f"install cache key mismatch: {entry}")
        if expected_key is not None and key != expected_key:
            raise ValueError(f"unexpected install cache entry: {entry}")
        self._validate_runtime_tree(entry / ".venv", deadline=deadline)

    def _validate_owned_staged_runtime(
        self,
        source_venv: Path,
        *,
        deadline: float | None,
    ) -> Path:
        self._raise_if_deadline_exceeded(deadline)
        build_root = source_venv.parent
        if source_venv.name != ".venv" or not self._is_build_root_name(build_root.name):
            raise ValueError(f"runtime is not in a cache-owned build root: {source_venv}")
        self._validate_private_directory(build_root, "task runtime cache-owned build root")
        if build_root.resolve(strict=True).parent != self.root:
            raise ValueError(f"runtime is not in a cache-owned build root: {source_venv}")
        self._validate_runtime_tree(source_venv, deadline=deadline)
        return source_venv.resolve(strict=True)

    def _write_owned_recovery_record(
        self,
        staging: Path,
        *,
        key: TaskRuntimeInstallCacheKey,
        source_build_root: str,
    ) -> None:
        self._write_manifest(
            staging / _OWNED_RECOVERY_MANIFEST,
            {
                "cache_key": key.digest,
                "schema_version": _OWNED_RECOVERY_SCHEMA_VERSION,
                "source_build_root": source_build_root,
            },
        )
        self._sync_directory(staging)

    def _reconcile_owned_staging(self, *, deadline: float | None = None) -> None:
        for path in self.root.iterdir():
            self._raise_if_deadline_exceeded(deadline)
            recovery_manifest = path / _OWNED_RECOVERY_MANIFEST
            if not recovery_manifest.exists() and not recovery_manifest.is_symlink():
                continue
            self._validate_private_directory(path, "task runtime owned recovery root")
            cache_key, source_build_root = self._read_owned_recovery_record(
                recovery_manifest, deadline=deadline
            )
            if path.name == cache_key:
                self._validate_entry(path, deadline=deadline)
                self._raise_if_deadline_exceeded(deadline)
                recovery_manifest.unlink()
                self._sync_directory(path)
                continue
            if not self._owned_recovery_name_matches(path.name, cache_key):
                raise ValueError(f"invalid task runtime owned recovery root: {path}")
            self._restore_owned_runtime(
                path,
                source_build_root=source_build_root,
                deadline=deadline,
            )

    def _restore_owned_runtime(
        self,
        recovery_root: Path,
        *,
        source_build_root: str,
        deadline: float | None,
    ) -> None:
        self._raise_if_deadline_exceeded(deadline)
        build_root = self.root / source_build_root
        if build_root.exists() or build_root.is_symlink():
            self._validate_private_directory(build_root, "task runtime cache-owned build root")
        else:
            build_root.mkdir(mode=0o700)
            self._sync_directory(self.root)
        source_venv = build_root / ".venv"
        retained_venv = recovery_root / ".venv"
        if retained_venv.exists() or retained_venv.is_symlink():
            if source_venv.exists() or source_venv.is_symlink():
                raise OSError(
                    "task runtime cache recovery destination already exists; "
                    f"recovery state retained at {recovery_root}"
                )
            self._validate_runtime_tree(retained_venv, deadline=deadline)
            self._raise_if_deadline_exceeded(deadline)
            retained_venv.replace(source_venv)
            self._sync_directory(build_root)
            self._sync_directory(recovery_root)
        else:
            try:
                self._validate_runtime_tree(source_venv, deadline=deadline)
            except (OSError, ValueError) as exc:
                raise OSError(
                    f"task runtime cache recovery is incomplete: {recovery_root}"
                ) from exc
        self._remove_owned_recovery_shell(recovery_root, deadline=deadline)
        self._sync_directory(self.root)

    def _read_owned_recovery_record(
        self, path: Path, *, deadline: float | None
    ) -> tuple[str, str]:
        self._raise_if_deadline_exceeded(deadline)
        self._validate_regular_file(path, "task runtime owned recovery manifest")
        metadata = path.stat(follow_symlinks=False)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError(f"unsafe task runtime owned recovery manifest: {path}")
        if metadata.st_size > 4096:
            raise ValueError(f"invalid task runtime owned recovery manifest: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid task runtime owned recovery manifest: {path}") from exc
        self._raise_if_deadline_exceeded(deadline)
        if not isinstance(payload, dict) or set(payload) != {
            "cache_key",
            "schema_version",
            "source_build_root",
        }:
            raise ValueError(f"invalid task runtime owned recovery manifest: {path}")
        cache_key = payload["cache_key"]
        source_build_root = payload["source_build_root"]
        if (
            payload["schema_version"] != _OWNED_RECOVERY_SCHEMA_VERSION
            or not isinstance(cache_key, str)
            or len(cache_key) != 64
            or any(character not in "0123456789abcdef" for character in cache_key)
            or not isinstance(source_build_root, str)
            or not self._is_build_root_name(source_build_root)
        ):
            raise ValueError(f"invalid task runtime owned recovery manifest: {path}")
        return cache_key, source_build_root

    @staticmethod
    def _owned_recovery_name_matches(name: str, cache_key: str) -> bool:
        staging_prefix = f".{cache_key}.staging-"
        recovery_prefix = f"{_RECOVERY_ROOT_PREFIX}{cache_key}-"
        for prefix in (staging_prefix, recovery_prefix):
            suffix = name.removeprefix(prefix)
            if name.startswith(prefix) and len(suffix) == 32 and all(
                character in "0123456789abcdef" for character in suffix
            ):
                return True
        return False

    @staticmethod
    def _is_build_root_name(name: str) -> bool:
        suffix = name.removeprefix(_BUILD_ROOT_PREFIX)
        return name.startswith(_BUILD_ROOT_PREFIX) and len(suffix) == 32 and all(
            character in "0123456789abcdef" for character in suffix
        )

    @staticmethod
    def _remove_owned_recovery_shell(
        path: Path, *, deadline: float | None
    ) -> None:
        allowed = {_CACHE_MANIFEST, _OWNED_RECOVERY_MANIFEST}
        children = list(path.iterdir())
        unexpected = [child for child in children if child.name not in allowed]
        if unexpected:
            raise OSError(f"unexpected task runtime cache recovery entry: {unexpected[0]}")
        marker = path / _OWNED_RECOVERY_MANIFEST
        for child in children:
            if child == marker:
                continue
            TaskRuntimeInstallCache._raise_if_deadline_exceeded(deadline)
            TaskRuntimeInstallCache._validate_regular_file(
                child, "task runtime owned recovery metadata"
            )
            child.unlink()
        TaskRuntimeInstallCache._validate_regular_file(
            marker, "task runtime owned recovery metadata"
        )
        marker.unlink()
        path.rmdir()

    @staticmethod
    def _sync_directory(path: Path) -> None:
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_private_directory(path: Path, label: str) -> None:
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"unsafe {label}: {path}")
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError(f"{label} must be private and owned by the current user: {path}")

    @staticmethod
    def _validate_owned_directory(path: Path, label: str) -> None:
        metadata = path.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
        ):
            raise ValueError(f"unsafe {label}: {path}")

    @staticmethod
    def _validate_regular_file(path: Path, label: str) -> None:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"unsafe {label}: {path}")

    @classmethod
    def _validate_runtime_tree(
        cls,
        runtime: Path,
        *,
        deadline: float | None = None,
    ) -> None:
        cls._raise_if_deadline_exceeded(deadline)
        cls._validate_owned_directory(runtime, "cached runtime")
        runtime_root = runtime.resolve(strict=True)
        for directory, directory_names, file_names in os.walk(runtime, followlinks=False):
            cls._raise_if_deadline_exceeded(deadline)
            current = Path(directory)
            for name in (*directory_names, *file_names):
                cls._raise_if_deadline_exceeded(deadline)
                path = current / name
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    if Path(os.readlink(path)).is_absolute():
                        raise ValueError(f"cached runtime contains an absolute symlink: {path}")
                    target = path.resolve(strict=True)
                    if not target.is_relative_to(runtime_root):
                        raise ValueError(f"cached runtime symlink escapes cache entry: {path}")
                elif not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError(f"unsafe cached runtime entry: {path}")
        cls._validate_regular_file(runtime / "bin" / "python", "cached runtime Python")

    @staticmethod
    def _write_manifest(path: Path, payload: dict[str, object]) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
