"""Create and reuse immutable task snapshots for pipeline runs."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import secrets
import stat
from pathlib import Path
from typing import cast

from src.investigation.files import (
    TASK_COPY_IGNORE_FILENAME,
)
from src.investigation.evaluator_tree import TASK_SNAPSHOT_METADATA_FILENAME

from .paths import PipelinePaths
from .safe_fs import (
    UnsafeGluePathError,
    ensure_directory,
    open_directory,
    regular_file_digest_at,
    real_directory_exists,
    validate_tree_symlinks_at,
    validate_tree_symlinks_fd,
)
from .task_tree import (
    cleanup_task_tree_preserving_error,
    copy_task_tree_at,
    require_unchanged_task_root,
    verify_copied_task_tree,
    task_tree_manifest_at,
)
from .task_publication import (
    TASK_SNAPSHOT_READY_NAME as PUBLICATION_READY_NAME,
    manifest_digest,
    publish_staged_task_tree_at,
    snapshot_is_ready_at,
    snapshot_ready_name,
)

logger = logging.getLogger(__name__)

TASK_SNAPSHOT_METADATA_MAX_BYTES = 4096
TASK_SNAPSHOT_METADATA_VERSION = 2
TASK_COPY_IGNORE_MAX_BYTES = 64 * 1024
STAGED_TASK_DIRECTORY_NAME = "task"
TASK_SNAPSHOT_READY_NAME = PUBLICATION_READY_NAME


def prepare_run_task(paths: PipelinePaths) -> None:
    """Create the immutable-source task snapshot used by one investigation run."""
    try:
        _prepare_run_task(paths)
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError("could not safely prepare run task snapshot") from exc


def _prepare_run_task(paths: PipelinePaths) -> None:
    source, destination = _validate_fresh_run_task(paths)
    logger.info("task snapshot: copy %s -> %s", source, destination)

    with open_directory(source, role="task source") as source_fd:
        ignore_patterns = _load_task_copy_ignore_patterns_at(source_fd)
        validate_tree_symlinks_fd(
            source_fd,
            relative_dir=Path(),
            role="task source",
            ignore_patterns=ignore_patterns,
        )
        with ensure_directory(destination.parent, role="run task parent") as parent_fd:
            _publish_run_task_copy(
                source_fd,
                destination,
                parent_fd,
                ignore_patterns,
                task_name=paths.task_name,
            )


def preview_run_task(paths: PipelinePaths) -> None:
    """Validate and log a fresh task snapshot without creating it."""
    source, destination = _validate_fresh_run_task(paths)
    try:
        with open_directory(source, role="task source") as source_fd:
            ignore_patterns = _load_task_copy_ignore_patterns_at(source_fd)
            validate_tree_symlinks_fd(
                source_fd,
                relative_dir=Path(),
                role="task source",
                ignore_patterns=ignore_patterns,
            )
    except FileNotFoundError as exc:
        raise UnsafeGluePathError(f"task source does not exist: {source}") from exc
    logger.info("task snapshot: copy %s -> %s", source, destination)


def reuse_run_task(paths: PipelinePaths) -> None:
    """Validate and reuse an existing run-local task snapshot."""
    try:
        with open_directory(paths.task_dir.parent, role="run task snapshot parent") as parent_fd:
            snapshot_fd = os.open(
                paths.task_dir.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                _reuse_run_task_at(parent_fd, snapshot_fd, paths.task_dir.name)
            finally:
                os.close(snapshot_fd)
    except FileNotFoundError as exc:
        raise UnsafeGluePathError(f"run task snapshot does not exist: {paths.task_dir}") from exc
    except OSError as exc:
        raise UnsafeGluePathError("could not safely reuse run task snapshot") from exc
    logger.info("task snapshot: reuse %s", paths.task_dir)


def reseal_run_task(paths: PipelinePaths) -> None:
    """Atomically reseal a successfully drafted run-local task tree."""
    try:
        with open_directory(paths.task_dir.parent, role="run task snapshot parent") as parent_fd:
            snapshot_fd = os.open(
                paths.task_dir.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                _reseal_run_task_at(parent_fd, snapshot_fd, paths.task_dir.name)
            finally:
                os.close(snapshot_fd)
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError("could not safely reseal drafted task snapshot") from exc


def _reseal_run_task_at(parent_fd: int, snapshot_fd: int, snapshot_name: str) -> None:
    if not task_snapshot_publication_is_complete_at(parent_fd, snapshot_fd, snapshot_name):
        raise UnsafeGluePathError("cannot reseal an incomplete task snapshot")
    validate_tree_symlinks_fd(snapshot_fd, relative_dir=Path(), role="drafted task snapshot")
    metadata = _read_task_snapshot_metadata_at(snapshot_fd)
    if metadata is None:
        raise UnsafeGluePathError("cannot reseal a task snapshot without metadata")
    old_marker = snapshot_ready_name(snapshot_name, snapshot_fd)
    root_stat = os.fstat(snapshot_fd)
    if root_stat.st_uid != os.getuid():
        raise UnsafeGluePathError("drafted task snapshot must be owned by this user")
    manifest = task_tree_manifest_at(
        snapshot_fd,
        relative_dir=Path(),
        ignore_patterns=(f"/{TASK_SNAPSHOT_METADATA_FILENAME}",),
    )
    payload = _task_snapshot_metadata_payload(
        str(metadata["task_name"]),
        source_digest=manifest_digest(manifest),
        source_root_mode=stat.S_IMODE(root_stat.st_mode),
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    new_marker = f".{snapshot_name}{PUBLICATION_READY_NAME}.{digest}"
    if new_marker == old_marker:
        return
    temporary = f".{snapshot_name}.task-reseal.{secrets.token_hex(8)}.tmp"
    marker_linked = False
    metadata_replaced = False
    try:
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(temporary_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary,
                new_marker,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if regular_file_digest_at(
                parent_fd,
                new_marker,
                role="drafted task snapshot ready marker",
            ) != digest:
                raise UnsafeGluePathError("drafted task snapshot has a conflicting ready marker")
        marker_linked = True
        os.replace(
            temporary,
            TASK_SNAPSHOT_METADATA_FILENAME,
            src_dir_fd=parent_fd,
            dst_dir_fd=snapshot_fd,
        )
        metadata_replaced = True
        if not snapshot_is_ready_at(parent_fd, snapshot_name, snapshot_fd):
            raise UnsafeGluePathError("drafted task snapshot reseal did not become ready")
        if old_marker != new_marker:
            os.unlink(old_marker, dir_fd=parent_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        if marker_linked and (
            not metadata_replaced
            or not snapshot_is_ready_at(parent_fd, snapshot_name, snapshot_fd)
        ):
            try:
                os.unlink(new_marker, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def _reuse_run_task_at(parent_fd: int, snapshot_fd: int, snapshot_name: str) -> None:
    validate_tree_symlinks_fd(
        snapshot_fd,
        relative_dir=Path(),
        role="run task snapshot",
    )
    metadata = _read_task_snapshot_metadata_at(snapshot_fd)
    if metadata is not None and metadata["version"] == 2:
        if not task_snapshot_publication_is_complete_at(
            parent_fd,
            snapshot_fd,
            snapshot_name,
        ):
            raise UnsafeGluePathError("run task snapshot publication is incomplete")


def ensure_run_task_for_reuse(
    paths: PipelinePaths,
    *,
    dry_run: bool,
    source_is_explicit: bool,
) -> bool:
    """Create the missing snapshot for a run produced before snapshots existed."""
    if real_directory_exists(paths.task_dir, role="run task snapshot"):
        if task_snapshot_publication_is_complete(
            paths.task_dir
        ) or task_snapshot_is_legacy(
            paths.task_dir
        ):
            return True
        if not source_is_explicit:
            raise UnsafeGluePathError(
                "incomplete run task snapshot recovery requires explicit "
                "task.dir=<original task source>"
            )
        logger.warning("resume incomplete run task snapshot: %s", paths.task_dir)
        if dry_run:
            preview_run_task(paths)
            return False
        prepare_run_task(paths)
        return True
    if not source_is_explicit:
        raise UnsafeGluePathError(
            "legacy run task bootstrap requires explicit task.dir=<original task source>"
        )
    logger.warning(
        "legacy run has no task snapshot; bootstrap %s from %s",
        paths.task_dir,
        paths.source_task_dir,
    )
    if dry_run:
        preview_run_task(paths)
        return False
    prepare_run_task(paths)
    return True


def read_task_snapshot_name(snapshot_dir: Path) -> str | None:
    """Read the stable task name stored in a run or paper task snapshot."""
    try:
        return _read_existing_task_snapshot_name(snapshot_dir)
    except FileNotFoundError:
        return None
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError("could not safely read task snapshot metadata") from exc


def task_snapshot_is_complete(snapshot_dir: Path) -> bool:
    """Return whether a versioned snapshot has completed publication."""
    try:
        return _task_snapshot_is_complete(snapshot_dir)
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError("could not safely inspect task snapshot publication") from exc


def _task_snapshot_is_complete(snapshot_dir: Path) -> bool:
    with open_directory(snapshot_dir.parent, role="task snapshot parent") as parent_fd:
        snapshot_fd = os.open(
            snapshot_dir.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            return task_snapshot_is_complete_at(parent_fd, snapshot_fd, snapshot_dir.name)
        finally:
            os.close(snapshot_fd)


def task_snapshot_is_complete_at(
    parent_fd: int,
    snapshot_fd: int,
    snapshot_name: str,
) -> bool:
    """Return strict completeness for an immutable freshly published snapshot."""
    if not task_snapshot_publication_is_complete_at(parent_fd, snapshot_fd, snapshot_name):
        return False
    metadata = _read_task_snapshot_metadata_at(snapshot_fd)
    if metadata is None or metadata["version"] == 1:
        return True
    if stat.S_IMODE(os.fstat(snapshot_fd).st_mode) != cast(
        int,
        metadata["source_root_mode"],
    ):
        return False
    validate_task_snapshot_manifest_at(
        snapshot_fd,
        ignore_patterns=(f"/{TASK_SNAPSHOT_METADATA_FILENAME}",),
    )
    return True


def task_snapshot_publication_is_complete(snapshot_dir: Path) -> bool:
    """Return publication state without requiring mutable run content to stay frozen."""
    try:
        with open_directory(snapshot_dir.parent, role="task snapshot parent") as parent_fd:
            snapshot_fd = os.open(
                snapshot_dir.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                return task_snapshot_publication_is_complete_at(
                    parent_fd,
                    snapshot_fd,
                    snapshot_dir.name,
                )
            finally:
                os.close(snapshot_fd)
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError("could not safely inspect task snapshot publication") from exc


def task_snapshot_publication_is_complete_at(
    parent_fd: int,
    snapshot_fd: int,
    snapshot_name: str,
) -> bool:
    """Return publication state for a snapshot pinned with its parent."""
    visible_stat = os.stat(snapshot_name, dir_fd=parent_fd, follow_symlinks=False)
    pinned_stat = os.fstat(snapshot_fd)
    if (
        not stat.S_ISDIR(visible_stat.st_mode)
        or pinned_stat.st_uid != os.getuid()
        or (visible_stat.st_dev, visible_stat.st_ino)
        != (pinned_stat.st_dev, pinned_stat.st_ino)
    ):
        raise UnsafeGluePathError("task snapshot changed while validating publication")
    metadata = _read_task_snapshot_metadata_at(snapshot_fd)
    if metadata is None:
        return False
    if metadata["version"] == 1:
        return True
    return snapshot_is_ready_at(parent_fd, snapshot_name, snapshot_fd)


def task_snapshot_is_legacy(snapshot_dir: Path) -> bool:
    """Return whether an anchored historical snapshot has unversioned content."""
    try:
        with open_directory(snapshot_dir, role="legacy task snapshot") as snapshot_fd:
            return task_snapshot_is_legacy_at(snapshot_fd)
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError("could not safely inspect legacy task snapshot") from exc


def task_snapshot_is_legacy_at(snapshot_fd: int) -> bool:
    """Return whether pinned historical task content is unversioned and non-empty."""
    return _read_task_snapshot_metadata_at(snapshot_fd) is None and bool(
        os.listdir(snapshot_fd)
    )


def task_snapshot_ready_path(snapshot_dir: Path) -> Path:
    """Return the sibling ready-marker path for tests and diagnostics."""
    with open_directory(snapshot_dir, role="task snapshot") as snapshot_fd:
        ready_name = snapshot_ready_name(snapshot_dir.name, snapshot_fd)
    return snapshot_dir.parent / ready_name


def _read_existing_task_snapshot_name(snapshot_dir: Path) -> str | None:
    with open_directory(snapshot_dir, role="task snapshot") as snapshot_fd:
        return read_task_snapshot_name_at(snapshot_fd)


def read_task_snapshot_name_at(snapshot_fd: int) -> str | None:
    """Read a stable task name from an already pinned snapshot directory."""
    metadata = _read_task_snapshot_metadata_at(snapshot_fd)
    return None if metadata is None else str(metadata["task_name"])


def validate_task_snapshot_manifest_at(
    snapshot_fd: int,
    *,
    ignore_patterns: tuple[str, ...],
) -> None:
    """Validate immutable v2 snapshot content against its recorded digest."""
    metadata = _read_task_snapshot_metadata_at(snapshot_fd)
    if metadata is None or metadata["version"] == 1:
        return
    manifest = task_tree_manifest_at(
        snapshot_fd,
        relative_dir=Path(),
        ignore_patterns=ignore_patterns,
    )
    if manifest_digest(manifest) != metadata["source_digest"]:
        raise UnsafeGluePathError("task snapshot content does not match its source digest")


def _read_task_snapshot_metadata_at(snapshot_fd: int) -> dict[str, object] | None:
    try:
        metadata_fd = os.open(
            TASK_SNAPSHOT_METADATA_FILENAME,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=snapshot_fd,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise UnsafeGluePathError("could not safely open task snapshot metadata") from exc
    try:
        metadata_stat = os.fstat(metadata_fd)
        if not stat.S_ISREG(metadata_stat.st_mode):
            raise UnsafeGluePathError("task snapshot metadata must be a regular file")
        if metadata_stat.st_size > TASK_SNAPSHOT_METADATA_MAX_BYTES:
            raise UnsafeGluePathError("task snapshot metadata is too large")
        with os.fdopen(metadata_fd, encoding="utf-8") as metadata_file:
            metadata_fd = -1
            payload = metadata_file.read(TASK_SNAPSHOT_METADATA_MAX_BYTES + 1)
        if len(payload) > TASK_SNAPSHOT_METADATA_MAX_BYTES:
            raise UnsafeGluePathError("task snapshot metadata is too large")
        metadata = json.loads(payload)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise UnsafeGluePathError("task snapshot metadata is invalid") from exc
    finally:
        if metadata_fd >= 0:
            os.close(metadata_fd)

    if not isinstance(metadata, dict) or type(metadata.get("version")) is not int:
        raise UnsafeGluePathError("task snapshot metadata has an unsupported version")
    version = metadata["version"]
    if version not in {1, TASK_SNAPSHOT_METADATA_VERSION}:
        raise UnsafeGluePathError("task snapshot metadata has an unsupported version")
    task_name = metadata.get("task_name")
    if not isinstance(task_name, str) or not _valid_task_name(task_name):
        raise UnsafeGluePathError("task snapshot metadata has an invalid task name")
    if version == 2:
        source_digest = metadata.get("source_digest")
        if not isinstance(source_digest, str) or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
            raise UnsafeGluePathError("task snapshot metadata has an invalid source digest")
        source_root_mode = metadata.get("source_root_mode")
        if type(source_root_mode) is not int or not 0 <= source_root_mode <= 0o7777:
            raise UnsafeGluePathError("task snapshot metadata has an invalid source root mode")
    return metadata


def _validate_fresh_run_task(paths: PipelinePaths) -> tuple[Path, Path]:
    source = paths.source_task_dir
    destination = paths.task_dir
    if not real_directory_exists(source, role="task source"):
        raise UnsafeGluePathError(f"task source does not exist: {source}")
    source_resolved = source.resolve(strict=True)
    destination_resolved = destination.resolve(strict=False)
    if destination_resolved.is_relative_to(source_resolved) or source_resolved.is_relative_to(
        destination_resolved
    ):
        raise UnsafeGluePathError(
            f"task source and run task snapshot must not overlap: {source} -> {destination}"
        )
    if destination.is_symlink():
        raise UnsafeGluePathError(f"fresh run task snapshot already exists: {destination}")
    if destination.exists() and not destination.is_dir():
        raise UnsafeGluePathError(f"fresh run task snapshot already exists: {destination}")
    return source, destination


def _load_task_copy_ignore_patterns_at(directory_fd: int) -> tuple[str, ...]:
    try:
        ignore_fd = os.open(
            TASK_COPY_IGNORE_FILENAME,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise UnsafeGluePathError(f"could not safely open {TASK_COPY_IGNORE_FILENAME}") from exc
    try:
        opened_stat = os.fstat(ignore_fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise UnsafeGluePathError(f"{TASK_COPY_IGNORE_FILENAME} must be a regular file")
        if opened_stat.st_size > TASK_COPY_IGNORE_MAX_BYTES:
            raise UnsafeGluePathError(f"{TASK_COPY_IGNORE_FILENAME} is too large")
        chunks: list[bytes] = []
        remaining = TASK_COPY_IGNORE_MAX_BYTES + 1
        while remaining:
            chunk = os.read(ignore_fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > TASK_COPY_IGNORE_MAX_BYTES:
            raise UnsafeGluePathError(f"{TASK_COPY_IGNORE_FILENAME} is too large")
        current_stat = os.fstat(ignore_fd)
        fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(opened_stat, field) != getattr(current_stat, field) for field in fields):
            raise UnsafeGluePathError(f"{TASK_COPY_IGNORE_FILENAME} changed while reading")
        if len(payload) != opened_stat.st_size:
            raise UnsafeGluePathError(f"{TASK_COPY_IGNORE_FILENAME} changed while reading")
        text = payload.decode("utf-8")
        return tuple(
            line
            for raw_line in text.splitlines()
            if (line := raw_line.strip()) and not line.startswith("#")
        )
    except UnicodeError as exc:
        raise UnsafeGluePathError(f"{TASK_COPY_IGNORE_FILENAME} is invalid") from exc
    finally:
        os.close(ignore_fd)


def _publish_run_task_copy(
    source_fd: int,
    destination: Path,
    parent_fd: int,
    ignore_patterns: tuple[str, ...],
    *,
    task_name: str,
) -> None:
    temporary_name = f".{destination.name}.{secrets.token_hex(8)}.tmp"
    source_root_stat = os.fstat(source_fd)
    os.mkdir(temporary_name, 0o700, dir_fd=parent_fd)
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        staging_parent_fd = os.open(temporary_name, flags, dir_fd=parent_fd)
        try:
            os.mkdir(STAGED_TASK_DIRECTORY_NAME, 0o700, dir_fd=staging_parent_fd)
            staged_task_fd = os.open(
                STAGED_TASK_DIRECTORY_NAME,
                flags,
                dir_fd=staging_parent_fd,
            )
            try:
                copy_task_tree_at(
                    source_fd,
                    staged_task_fd,
                    relative_dir=Path(),
                    ignore_patterns=ignore_patterns,
                )
                if ignore_patterns != _load_task_copy_ignore_patterns_at(source_fd):
                    raise UnsafeGluePathError("task copy ignore patterns changed while copying")
                verify_copied_task_tree(
                    source_fd,
                    staged_task_fd,
                    ignore_patterns=ignore_patterns,
                )
                require_unchanged_task_root(source_fd, source_root_stat)
                source_manifest = task_tree_manifest_at(
                    source_fd,
                    relative_dir=Path(),
                    ignore_patterns=ignore_patterns,
                )
                write_task_snapshot_metadata_at(
                    staged_task_fd,
                    task_name,
                    source_digest=manifest_digest(source_manifest),
                    source_root_mode=stat.S_IMODE(source_root_stat.st_mode),
                )
                validate_tree_symlinks_at(
                    staging_parent_fd,
                    STAGED_TASK_DIRECTORY_NAME,
                    role="staged run task",
                )
                if ignore_patterns != _load_task_copy_ignore_patterns_at(source_fd):
                    raise UnsafeGluePathError("task copy ignore patterns changed while copying")
                verify_copied_task_tree(
                    source_fd,
                    staged_task_fd,
                    ignore_patterns=ignore_patterns,
                    destination_only_paths=frozenset({TASK_SNAPSHOT_METADATA_FILENAME}),
                )
                require_unchanged_task_root(source_fd, source_root_stat)
                publish_staged_task_tree_at(
                    staging_parent_fd,
                    STAGED_TASK_DIRECTORY_NAME,
                    parent_fd,
                    destination.name,
                    root_mode=stat.S_IMODE(source_root_stat.st_mode),
                    role="fresh run task snapshot",
                )
            finally:
                os.close(staged_task_fd)
        finally:
            os.close(staging_parent_fd)
        cleanup_task_tree_preserving_error(
            parent_fd,
            temporary_name,
            role="empty task staging parent",
        )
    except BaseException:
        cleanup_task_tree_preserving_error(
            parent_fd,
            temporary_name,
            role="temporary task snapshot",
        )
        raise


def write_task_snapshot_metadata_at(
    directory_fd: int,
    task_name: str,
    *,
    source_digest: str,
    source_root_mode: int,
) -> None:
    payload = _task_snapshot_metadata_payload(
        task_name,
        source_digest=source_digest,
        source_root_mode=source_root_mode,
    )
    try:
        metadata_fd = os.open(
            TASK_SNAPSHOT_METADATA_FILENAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise UnsafeGluePathError(
            f"task source reserves {TASK_SNAPSHOT_METADATA_FILENAME}"
        ) from exc
    with os.fdopen(metadata_fd, "w", encoding="utf-8") as metadata_file:
        metadata_file.write(payload)


def _task_snapshot_metadata_payload(
    task_name: str,
    *,
    source_digest: str,
    source_root_mode: int,
) -> str:
    if not _valid_task_name(task_name):
        raise UnsafeGluePathError(f"invalid task name for snapshot metadata: {task_name!r}")
    return json.dumps(
        {
            "source_digest": source_digest,
            "source_root_mode": source_root_mode,
            "task_name": task_name,
            "version": TASK_SNAPSHOT_METADATA_VERSION,
        },
        separators=(",", ":"),
    ) + "\n"


def _valid_task_name(task_name: str) -> bool:
    return bool(task_name) and task_name not in {".", ".."} and not any(
        separator in task_name for separator in ("/", "\\", "\0")
    )
