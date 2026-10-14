"""Portable, resumable publication for validated task directories."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path

from .safe_fs import (
    AtomicNoReplaceUnsupported,
    UnsafeGluePathError,
    entry_exists_at,
    regular_file_digest_at,
    rename_directory_between_no_replace,
)
from .task_tree import TaskTreeManifest, task_tree_manifest_at
from .task_tree_merge import merge_staged_task_tree_at

TASK_SNAPSHOT_READY_NAME = ".agentic_task_ready"
TASK_SNAPSHOT_METADATA_NAME = ".agentic_task_snapshot.json"
_CLAIM_PREFIX = ".agentic-task-source-"


def publish_staged_task_tree_at(
    staging_parent_fd: int,
    staged_name: str,
    destination_parent_fd: int,
    destination_name: str,
    *,
    root_mode: int,
    role: str,
) -> None:
    """Publish a staged task tree and normalize filesystem failures."""
    try:
        _publish_staged_task_tree_at(
            staging_parent_fd,
            staged_name,
            destination_parent_fd,
            destination_name,
            root_mode=root_mode,
            role=role,
        )
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError(f"could not publish {role}") from exc


def _publish_staged_task_tree_at(
    staging_parent_fd: int,
    staged_name: str,
    destination_parent_fd: int,
    destination_name: str,
    *,
    root_mode: int,
    role: str,
) -> None:
    """Publish a complete staged tree, falling back to a resumable live merge."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    staged_fd = os.open(staged_name, flags, dir_fd=staging_parent_fd)
    try:
        expected_manifest = task_tree_manifest_at(
            staged_fd,
            relative_dir=Path(),
            ignore_patterns=(),
        )
        source_digest = manifest_digest(expected_manifest)
        if not entry_exists_at(destination_parent_fd, destination_name):
            _remove_matching_orphan_ready_marker(
                destination_parent_fd,
                destination_name,
                staged_fd,
                role=role,
            )
            _set_directory_mode(staged_fd, root_mode, role=role)
            try:
                rename_directory_between_no_replace(
                    staging_parent_fd,
                    staged_name,
                    destination_parent_fd,
                    destination_name,
                    role=role,
                )
                _publish_ready_marker(
                    destination_parent_fd,
                    destination_name,
                    staged_fd,
                    role=role,
                )
                return
            except AtomicNoReplaceUnsupported:
                _set_directory_mode(staged_fd, 0o700, role=role)

        _publish_by_live_merge(
            staging_parent_fd,
            staged_fd,
            destination_parent_fd,
            destination_name,
            expected_manifest=expected_manifest,
            source_digest=source_digest,
            root_mode=root_mode,
            role=role,
        )
    finally:
        os.close(staged_fd)


def snapshot_ready_name(destination_name: str, snapshot_fd: int) -> str:
    """Return the sibling completion-marker name for one snapshot directory."""
    metadata_digest = regular_file_digest_at(
        snapshot_fd,
        TASK_SNAPSHOT_METADATA_NAME,
        role="task snapshot metadata",
    )
    return f".{destination_name}{TASK_SNAPSHOT_READY_NAME}.{metadata_digest}"


def snapshot_is_ready_at(
    parent_fd: int,
    destination_name: str,
    snapshot_fd: int,
) -> bool:
    """Return whether a valid atomic completion marker is present."""
    marker_name = snapshot_ready_name(destination_name, snapshot_fd)
    try:
        marker_fd = os.open(
            marker_name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise UnsafeGluePathError("could not safely open task snapshot ready marker") from exc
    try:
        marker_stat = os.fstat(marker_fd)
        metadata_fd = os.open(
            TASK_SNAPSHOT_METADATA_NAME,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=snapshot_fd,
        )
        try:
            metadata_stat = os.fstat(metadata_fd)
        finally:
            os.close(metadata_fd)
        if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_uid != os.getuid():
            raise UnsafeGluePathError("task snapshot ready marker must be an owned regular file")
        if not stat.S_ISREG(metadata_stat.st_mode) or metadata_stat.st_uid != os.getuid():
            raise UnsafeGluePathError("task snapshot metadata must be an owned regular file")
        metadata_digest = regular_file_digest_at(
            snapshot_fd,
            TASK_SNAPSHOT_METADATA_NAME,
            role="task snapshot metadata",
        )
        if marker_stat.st_size == 0 and marker_stat.st_nlink == 1:
            marker_matches = True
        else:
            marker_digest = regular_file_digest_at(
                parent_fd,
                marker_name,
                role="task snapshot ready marker",
            )
            marker_matches = marker_digest == metadata_digest
        if not marker_matches:
            raise UnsafeGluePathError("task snapshot ready marker does not match its metadata")
        current_marker_stat = os.fstat(marker_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(marker_stat, field) != getattr(current_marker_stat, field)
            for field in stable_fields
        ):
            raise UnsafeGluePathError("task snapshot ready marker changed while validating")
    finally:
        os.close(marker_fd)
    return True


def _publish_by_live_merge(
    staging_parent_fd: int,
    staged_fd: int,
    destination_parent_fd: int,
    destination_name: str,
    *,
    expected_manifest: TaskTreeManifest,
    source_digest: str,
    root_mode: int,
    role: str,
) -> None:
    claim_name = f".{destination_name}{_CLAIM_PREFIX}{source_digest}"
    claim_identity, claim_created, claim_fd = _claim_source(
        destination_parent_fd,
        destination_name,
        claim_name,
        role=role,
    )
    remove_claim = claim_created
    try:
        try:
            os.mkdir(destination_name, 0o700, dir_fd=destination_parent_fd)
        except FileExistsError:
            pass
        destination_fd = os.open(
            destination_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=destination_parent_fd,
        )
        try:
            destination_stat = os.fstat(destination_fd)
            if destination_stat.st_uid != os.getuid():
                raise UnsafeGluePathError(f"{role} is not owned by the current user")
            existing_names = set(os.listdir(destination_fd))
            if existing_names and TASK_SNAPSHOT_METADATA_NAME not in existing_names:
                raise UnsafeGluePathError(
                    f"{role} conflicts with an unclaimed incomplete task tree"
                )
            if existing_names:
                if snapshot_is_ready_at(
                    destination_parent_fd,
                    destination_name,
                    destination_fd,
                ):
                    remove_claim = True
                    raise UnsafeGluePathError(f"{role} already exists: {destination_name}")
                try:
                    _require_matching_metadata(staged_fd, destination_fd, role=role)
                except UnsafeGluePathError:
                    if claim_created:
                        raise
                    _remove_interrupted_metadata(destination_fd, role=role)
            _set_directory_mode(destination_fd, 0o700, role=role)
            remove_claim = False
            merge_staged_task_tree_at(
                staged_fd,
                destination_fd,
                expected_manifest=expected_manifest,
                excluded_paths=frozenset(),
                role=role,
                recover_incomplete_copies=not claim_created,
            )
            _set_directory_mode(destination_fd, root_mode, role=role)
            _publish_ready_marker(
                destination_parent_fd,
                destination_name,
                destination_fd,
                role=role,
            )
            remove_claim = True
        finally:
            os.close(destination_fd)
    finally:
        try:
            if remove_claim:
                try:
                    _remove_source_claim(
                        destination_parent_fd,
                        claim_name,
                        claim_identity,
                        role=role,
                    )
                except FileNotFoundError:
                    pass
        finally:
            os.close(claim_fd)


def _claim_source(
    parent_fd: int,
    destination_name: str,
    claim_name: str,
    *,
    role: str,
) -> tuple[tuple[int, int], bool, int]:
    prefix = f".{destination_name}{_CLAIM_PREFIX}"
    claims = sorted(name for name in os.listdir(parent_fd) if name.startswith(prefix))
    if claims and claims != [claim_name]:
        raise UnsafeGluePathError(f"{role} conflicts with a different staged task tree")
    while True:
        created = False
        claim_identity: tuple[int, int] | None = None
        try:
            os.mkdir(claim_name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        claim_fd: int | None = None
        try:
            claim_fd = os.open(
                claim_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            fcntl.flock(claim_fd, fcntl.LOCK_EX)
            if created:
                os.fchmod(claim_fd, 0o700)
            claim_stat = os.fstat(claim_fd)
            if (
                claim_stat.st_uid != os.getuid()
                or stat.S_IMODE(claim_stat.st_mode) != 0o700
            ):
                raise UnsafeGluePathError(f"{role} has an unsafe source claim")
            if os.listdir(claim_fd):
                raise UnsafeGluePathError(f"{role} has a non-empty source claim")
            claim_identity = (claim_stat.st_dev, claim_stat.st_ino)
            try:
                current = os.stat(claim_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                os.close(claim_fd)
                continue
            if (current.st_dev, current.st_ino) != claim_identity:
                raise UnsafeGluePathError(f"{role} source claim changed while locking")
            claims = sorted(name for name in os.listdir(parent_fd) if name.startswith(prefix))
            if claims != [claim_name]:
                raise UnsafeGluePathError(f"{role} conflicts with a different staged task tree")
            if created:
                os.fsync(parent_fd)
            return claim_identity, created, claim_fd
        except BaseException:
            if created and claim_identity is not None:
                try:
                    _remove_source_claim(
                        parent_fd,
                        claim_name,
                        claim_identity,
                        role=role,
                    )
                except (OSError, UnsafeGluePathError):
                    pass
            if claim_fd is not None:
                os.close(claim_fd)
            raise


def _publish_ready_marker(
    destination_parent_fd: int,
    destination_name: str,
    snapshot_fd: int,
    *,
    role: str,
) -> None:
    marker_name = snapshot_ready_name(destination_name, snapshot_fd)
    marker_fd: int | None = None
    try:
        os.fsync(snapshot_fd)
        os.fsync(destination_parent_fd)
        marker_fd = os.open(
            marker_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
            dir_fd=destination_parent_fd,
        )
        os.fsync(marker_fd)
        os.close(marker_fd)
        marker_fd = None
        os.fsync(destination_parent_fd)
        if not snapshot_is_ready_at(
            destination_parent_fd,
            destination_name,
            snapshot_fd,
        ):
            raise UnsafeGluePathError(f"{role} completion marker was not published")
    except FileExistsError as exc:
        raise UnsafeGluePathError(f"{role} completion marker already exists") from exc
    except OSError as exc:
        raise UnsafeGluePathError(f"could not publish {role} completion marker") from exc
    finally:
        if marker_fd is not None:
            os.close(marker_fd)


def _remove_matching_orphan_ready_marker(
    parent_fd: int,
    destination_name: str,
    staged_fd: int,
    *,
    role: str,
) -> None:
    marker_name = snapshot_ready_name(destination_name, staged_fd)
    try:
        marker_fd = os.open(
            marker_name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return
    try:
        marker_stat = os.fstat(marker_fd)
        if (
            not stat.S_ISREG(marker_stat.st_mode)
            or marker_stat.st_uid != os.getuid()
            or marker_stat.st_nlink != 1
            or (
                marker_stat.st_size != 0
                and regular_file_digest_at(
                    parent_fd,
                    marker_name,
                    role="orphan task snapshot ready marker",
                )
                != regular_file_digest_at(
                    staged_fd,
                    TASK_SNAPSHOT_METADATA_NAME,
                    role="staged task metadata",
                )
            )
        ):
            raise UnsafeGluePathError(f"{role} has an unsafe orphan completion marker")
        current = os.stat(marker_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            (current.st_dev, current.st_ino) != (marker_stat.st_dev, marker_stat.st_ino)
            or current.st_nlink != 1
            or entry_exists_at(parent_fd, destination_name)
        ):
            raise UnsafeGluePathError(f"{role} orphan completion marker changed")
        os.unlink(marker_name, dir_fd=parent_fd)
    finally:
        os.close(marker_fd)


def _require_matching_metadata(source_fd: int, destination_fd: int, *, role: str) -> None:
    source_stat = os.stat(
        TASK_SNAPSHOT_METADATA_NAME,
        dir_fd=source_fd,
        follow_symlinks=False,
    )
    destination_stat = os.stat(
        TASK_SNAPSHOT_METADATA_NAME,
        dir_fd=destination_fd,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(source_stat.st_mode)
        or not stat.S_ISREG(destination_stat.st_mode)
        or destination_stat.st_uid != os.getuid()
        or stat.S_IMODE(source_stat.st_mode) != stat.S_IMODE(destination_stat.st_mode)
        or regular_file_digest_at(
            source_fd,
            TASK_SNAPSHOT_METADATA_NAME,
            role="staged task metadata",
        )
        != regular_file_digest_at(
            destination_fd,
            TASK_SNAPSHOT_METADATA_NAME,
            role="incomplete task metadata",
        )
    ):
        raise UnsafeGluePathError(f"{role} conflicts with a different staged task tree")
    current = os.stat(
        TASK_SNAPSHOT_METADATA_NAME,
        dir_fd=destination_fd,
        follow_symlinks=False,
    )
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(destination_stat, field) != getattr(current, field) for field in fields):
        raise UnsafeGluePathError(f"{role} metadata changed while validating")


def _remove_interrupted_metadata(destination_fd: int, *, role: str) -> None:
    """Remove only an owned single-link metadata copy from an interrupted merge."""
    try:
        metadata = os.stat(
            TASK_SNAPSHOT_METADATA_NAME,
            dir_fd=destination_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
    ):
        raise UnsafeGluePathError(f"{role} has unsafe interrupted metadata")
    current = os.stat(
        TASK_SNAPSHOT_METADATA_NAME,
        dir_fd=destination_fd,
        follow_symlinks=False,
    )
    if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise UnsafeGluePathError(f"{role} interrupted metadata changed")
    os.unlink(TASK_SNAPSHOT_METADATA_NAME, dir_fd=destination_fd)


def _remove_source_claim(
    parent_fd: int,
    claim_name: str,
    expected_identity: tuple[int, int],
    *,
    role: str,
) -> None:
    current = os.stat(claim_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(current.st_mode)
        or current.st_uid != os.getuid()
        or (current.st_dev, current.st_ino) != expected_identity
    ):
        raise UnsafeGluePathError(f"{role} source claim changed before cleanup")
    os.rmdir(claim_name, dir_fd=parent_fd)


def manifest_digest(manifest: TaskTreeManifest) -> str:
    """Return a deterministic digest for a descriptor-validated tree manifest."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _set_directory_mode(directory_fd: int, mode: int, *, role: str) -> None:
    try:
        os.fchmod(directory_fd, mode)
    except OSError as exc:
        raise UnsafeGluePathError(f"could not set {role} permissions") from exc
