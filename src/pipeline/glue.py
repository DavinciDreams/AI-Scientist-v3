"""Glue steps that bridge variants output to the paper stage."""

from __future__ import annotations

import logging
import os
import secrets
import stat
from pathlib import Path

from src.investigation.evaluator_tree import MANAGED_RUNTIME_MANIFEST_FILENAME

from .config import PipelineConfig
from .paths import PipelinePaths
from .safe_fs import (
    UnsafeGluePathError,
    copy_regular_file_at as _copy_regular_file_at,
    ensure_directory as _ensure_directory,
    entry_exists_at as _entry_exists_at,
    open_directory as _open_directory,
    real_directory_exists as _real_directory_exists,
    regular_file_digest_at as _regular_file_digest_at,
    regular_file_exists_at as _regular_file_exists_at,
    validate_tree_symlinks_at as _validate_tree_symlinks_at,
    validate_tree_symlinks_fd as _validate_tree_symlinks_fd,
)
from .task_snapshot import (
    TASK_SNAPSHOT_METADATA_FILENAME,
    ensure_run_task_for_reuse as ensure_run_task_for_reuse,
    prepare_run_task as prepare_run_task,
    preview_run_task as preview_run_task,
    read_task_snapshot_name as read_task_snapshot_name,
    read_task_snapshot_name_at,
    reuse_run_task as reuse_run_task,
    task_snapshot_is_legacy_at,
    task_snapshot_publication_is_complete_at,
    validate_task_snapshot_manifest_at,
    write_task_snapshot_metadata_at,
)
from .task_publication import manifest_digest, publish_staged_task_tree_at
from .task_tree import (
    cleanup_task_tree_preserving_error as _cleanup_task_tree_preserving_error,
    copy_task_tree_at as _copy_task_tree_at,
    require_unchanged_pinned_task_root as _require_unchanged_pinned_task_root,
    task_tree_manifest_at as _task_tree_manifest_at,
    verify_copied_task_tree as _verify_copied_task_tree,
)

logger = logging.getLogger(__name__)

AGGREGATED_CSVS = ("run_summary.csv", "metric_values.csv", "timeseries.csv")
MEMORY_FILES = ("memory.jsonl", "memory.md")
STAGED_TASK_DIRECTORY_NAME = "task"
_PAPER_TASK_RUNTIME_IGNORE_PATTERNS = (
    "/.venv/",
    f"/{MANAGED_RUNTIME_MANIFEST_FILENAME}",
)
_PAPER_TASK_PUBLICATION_IGNORE_PATTERNS = (f"/{TASK_SNAPSHOT_METADATA_FILENAME}",)
_PAPER_TASK_SOURCE_IGNORE_PATTERNS = (
    *_PAPER_TASK_RUNTIME_IGNORE_PATTERNS,
    *_PAPER_TASK_PUBLICATION_IGNORE_PATTERNS,
)
_PAPER_TASK_RUNTIME_ROOT_ENTRIES = (
    ".venv",
    MANAGED_RUNTIME_MANIFEST_FILENAME,
)


def populate_task_dir(paths: PipelinePaths, *, dry_run: bool) -> None:
    """Copy the task dir into the ablations bundle for paper.sh."""
    try:
        _populate_task_dir(paths, dry_run=dry_run)
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError("could not safely publish paper task snapshot") from exc


def _populate_task_dir(paths: PipelinePaths, *, dry_run: bool) -> None:
    src = paths.task_dir
    dst = paths.paper_task_dir
    if dst.is_symlink():
        raise UnsafeGluePathError(f"task destination must be a real directory: {dst}")
    if _reuse_existing_paper_task(paths):
        logger.info("glue: trusted task dir already present: %s (skip)", dst)
        return
    if dry_run and not _real_directory_exists(src, role="task source"):
        logger.info("glue: copy task dir %s -> %s", src, dst)
        return
    logger.info("glue: copy task dir %s -> %s", src, dst)
    with _open_directory(src, role="task source") as source_fd:
        task_name = read_task_snapshot_name_at(source_fd) or paths.task_name
        _validate_tree_symlinks_fd(
            source_fd,
            relative_dir=Path(),
            role="task source",
            ignore_patterns=_PAPER_TASK_SOURCE_IGNORE_PATTERNS,
        )
        if dry_run:
            return
        source_root_stat = os.fstat(source_fd)
        with _ensure_directory(dst.parent, role="task destination parent") as parent_fd:
            temporary_name = f".{dst.name}.{secrets.token_hex(8)}.tmp"
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
                        _copy_task_tree_at(
                            source_fd,
                            staged_task_fd,
                            relative_dir=Path(),
                            ignore_patterns=_PAPER_TASK_SOURCE_IGNORE_PATTERNS,
                        )
                        _verify_copied_task_tree(
                            source_fd,
                            staged_task_fd,
                            ignore_patterns=_PAPER_TASK_SOURCE_IGNORE_PATTERNS,
                        )
                        _require_unchanged_pinned_task_root(source_fd, source_root_stat)
                        source_manifest = _task_tree_manifest_at(
                            source_fd,
                            relative_dir=Path(),
                            ignore_patterns=_PAPER_TASK_SOURCE_IGNORE_PATTERNS,
                        )
                        write_task_snapshot_metadata_at(
                            staged_task_fd,
                            task_name,
                            source_digest=manifest_digest(source_manifest),
                            source_root_mode=stat.S_IMODE(source_root_stat.st_mode),
                        )
                        _validate_tree_symlinks_at(
                            staging_parent_fd,
                            STAGED_TASK_DIRECTORY_NAME,
                            role="staged task",
                        )
                        _verify_copied_task_tree(
                            source_fd,
                            staged_task_fd,
                            ignore_patterns=_PAPER_TASK_SOURCE_IGNORE_PATTERNS,
                            destination_only_paths=frozenset({TASK_SNAPSHOT_METADATA_FILENAME}),
                        )
                        _require_unchanged_pinned_task_root(source_fd, source_root_stat)
                        publish_staged_task_tree_at(
                            staging_parent_fd,
                            STAGED_TASK_DIRECTORY_NAME,
                            parent_fd,
                            dst.name,
                            root_mode=stat.S_IMODE(source_root_stat.st_mode),
                            role="task destination",
                        )
                    finally:
                        os.close(staged_task_fd)
                finally:
                    os.close(staging_parent_fd)
                _cleanup_task_tree_preserving_error(
                    parent_fd,
                    temporary_name,
                    role="empty paper task staging parent",
                )
            except BaseException:
                _cleanup_task_tree_preserving_error(
                    parent_fd,
                    temporary_name,
                    role="temporary paper task",
                )
                raise


def _require_no_managed_runtime(directory_fd: int, destination: Path) -> None:
    for entry in _PAPER_TASK_RUNTIME_ROOT_ENTRIES:
        if _entry_exists_at(directory_fd, entry):
            raise UnsafeGluePathError(
                f"task destination contains managed runtime entry {entry!r}: {destination}"
            )


def _reuse_existing_paper_task(paths: PipelinePaths) -> bool:
    src = paths.task_dir
    dst = paths.paper_task_dir
    if not dst.exists():
        return False
    with _open_directory(dst.parent, role="task destination parent") as parent_fd:
        destination_fd = os.open(
            dst.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            complete = task_snapshot_publication_is_complete_at(
                parent_fd,
                destination_fd,
                dst.name,
            )
            legacy = task_snapshot_is_legacy_at(destination_fd)
            if not complete and not legacy:
                return False
            _validate_tree_symlinks_fd(
                destination_fd,
                relative_dir=Path(),
                role="task destination",
            )
            _require_no_managed_runtime(destination_fd, dst)
            if complete:
                validate_task_snapshot_manifest_at(
                    destination_fd,
                    ignore_patterns=_PAPER_TASK_PUBLICATION_IGNORE_PATTERNS,
                )
            if not _real_directory_exists(src, role="task source"):
                return True
            with _open_directory(src, role="task source") as source_fd:
                _validate_tree_symlinks_fd(
                    source_fd,
                    relative_dir=Path(),
                    role="task source",
                    ignore_patterns=_PAPER_TASK_SOURCE_IGNORE_PATTERNS,
                )
                source_manifest = _task_tree_manifest_at(
                    source_fd,
                    relative_dir=Path(),
                    ignore_patterns=_PAPER_TASK_SOURCE_IGNORE_PATTERNS,
                )
                destination_manifest = _task_tree_manifest_at(
                    destination_fd,
                    relative_dir=Path(),
                    ignore_patterns=_PAPER_TASK_PUBLICATION_IGNORE_PATTERNS,
                )
                if source_manifest != _task_tree_manifest_at(
                    source_fd,
                    relative_dir=Path(),
                    ignore_patterns=_PAPER_TASK_SOURCE_IGNORE_PATTERNS,
                ):
                    raise UnsafeGluePathError("task source changed while validating")
                if source_manifest != destination_manifest:
                    raise UnsafeGluePathError(
                        f"task destination does not match trusted task source: {dst}"
                    )
            return True
        finally:
            os.close(destination_fd)


def reuse_paper_task_dir(paths: PipelinePaths) -> None:
    """Validate and reuse the task snapshot already published for paper."""
    try:
        _reuse_paper_task_dir(paths)
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError("could not safely reuse paper task snapshot") from exc


def _reuse_paper_task_dir(paths: PipelinePaths) -> None:
    snapshot = paths.paper_task_dir
    with _open_directory(snapshot.parent, role="paper task snapshot parent") as parent_fd:
        snapshot_fd = os.open(
            snapshot.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            complete = task_snapshot_publication_is_complete_at(
                parent_fd, snapshot_fd, snapshot.name
            )
            if not complete and not task_snapshot_is_legacy_at(snapshot_fd):
                raise UnsafeGluePathError("paper task snapshot publication is incomplete")
            _validate_tree_symlinks_fd(
                snapshot_fd,
                relative_dir=Path(),
                role="paper task snapshot",
            )
            _require_no_managed_runtime(snapshot_fd, snapshot)
            validate_task_snapshot_manifest_at(
                snapshot_fd,
                ignore_patterns=_PAPER_TASK_PUBLICATION_IGNORE_PATTERNS,
            )
            _task_tree_manifest_at(
                snapshot_fd,
                relative_dir=Path(),
                ignore_patterns=(),
            )
        finally:
            os.close(snapshot_fd)
    logger.info("glue: reuse paper task snapshot: %s", snapshot)


def link_aggregated_csvs(paths: PipelinePaths, *, dry_run: bool) -> None:
    """Create flat top-level ``aggregated/<name>`` symlinks -> ``<task>/<name>``."""
    agg_task = paths.ablations_aggregated_task_dir
    agg_root = paths.ablations_output_dir / "aggregated"
    if Path(paths.task_name).name != paths.task_name or paths.task_name in {".", ".."}:
        raise UnsafeGluePathError(
            f"aggregated task name must be one path component: {paths.task_name}"
        )

    try:
        root_context = _open_directory(agg_root, role="aggregated root")
        with root_context as root_fd:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            try:
                task_fd = os.open(paths.task_name, flags, dir_fd=root_fd)
            except FileNotFoundError:
                logger.warning(
                    "glue: aggregated task dir missing: %s "
                    "(the ablations aggregate step may not have run); skipping csv links",
                    agg_task,
                )
                return
            except OSError as exc:
                raise UnsafeGluePathError(
                    f"could not safely open aggregated task directory: {agg_task}"
                ) from exc

            try:
                for name in AGGREGATED_CSVS:
                    if not _regular_file_exists_at(task_fd, name, role="aggregated csv"):
                        logger.warning(
                            "glue: aggregated csv missing: %s (skip link)",
                            agg_task / name,
                        )
                        continue
                    rel_target = Path(paths.task_name) / name
                    link = agg_root / name
                    logger.info("glue: symlink %s -> %s", link, rel_target)
                    if dry_run:
                        continue
                    temporary_name = f".{name}.{secrets.token_hex(8)}.tmp"
                    try:
                        os.symlink(rel_target, temporary_name, dir_fd=root_fd)
                        os.replace(
                            temporary_name,
                            name,
                            src_dir_fd=root_fd,
                            dst_dir_fd=root_fd,
                        )
                    finally:
                        try:
                            os.unlink(temporary_name, dir_fd=root_fd)
                        except FileNotFoundError:
                            pass
            finally:
                os.close(task_fd)
    except FileNotFoundError:
        logger.warning(
            "glue: aggregated task dir missing: %s "
            "(the ablations aggregate step may not have run); skipping csv links",
            agg_task,
        )


def resolve_aggregated_task_name(
    aggregated_root: Path,
    requested_task_name: str | None,
) -> str:
    """Resolve one task with a regular run summary under a trusted aggregate root."""
    if requested_task_name is not None:
        _validate_aggregated_task_name(requested_task_name)

    with _open_directory(aggregated_root, role="aggregated root") as root_fd:
        if requested_task_name is not None:
            candidates = [requested_task_name]
        else:
            candidates = sorted(os.listdir(root_fd))

        matched: list[str] = []
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        for candidate in candidates:
            try:
                task_fd = os.open(candidate, flags, dir_fd=root_fd)
            except FileNotFoundError:
                continue
            except OSError as exc:
                if requested_task_name is not None:
                    raise UnsafeGluePathError(
                        f"could not safely open aggregated task directory: "
                        f"{aggregated_root / candidate}"
                    ) from exc
                continue
            try:
                if _regular_file_exists_at(
                    task_fd,
                    "run_summary.csv",
                    role="aggregated run summary",
                ):
                    matched.append(candidate)
            finally:
                os.close(task_fd)

    if requested_task_name is not None and not matched:
        raise UnsafeGluePathError(
            "TASK_NAME has no aggregated run summary: "
            f"{aggregated_root / requested_task_name / 'run_summary.csv'}"
        )
    if not matched:
        raise UnsafeGluePathError(
            "could not auto-detect TASK_NAME under "
            f"{aggregated_root}/*/run_summary.csv; set TASK_NAME explicitly"
        )
    if len(matched) > 1:
        raise UnsafeGluePathError(
            f"multiple tasks found under {aggregated_root}; set TASK_NAME explicitly: "
            + " ".join(matched)
        )
    return matched[0]


def link_aggregated_csvs_in_root(
    aggregated_root: Path,
    task_name: str,
    *,
    dry_run: bool,
) -> None:
    """Create flat CSV links while pinning a caller-provided aggregate root."""
    _validate_aggregated_task_name(task_name)
    with _open_directory(aggregated_root, role="aggregated root") as root_fd:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            task_fd = os.open(task_name, flags, dir_fd=root_fd)
        except OSError as exc:
            raise UnsafeGluePathError(
                f"could not safely open aggregated task directory: {aggregated_root / task_name}"
            ) from exc
        try:
            for name in AGGREGATED_CSVS:
                if not _regular_file_exists_at(task_fd, name, role="aggregated csv"):
                    continue
                logger.info(
                    "glue: symlink %s -> %s",
                    aggregated_root / name,
                    Path(task_name) / name,
                )
                if dry_run:
                    continue
                temporary_name = f".{name}.{secrets.token_hex(8)}.tmp"
                try:
                    os.symlink(Path(task_name) / name, temporary_name, dir_fd=root_fd)
                    os.replace(
                        temporary_name,
                        name,
                        src_dir_fd=root_fd,
                        dst_dir_fd=root_fd,
                    )
                finally:
                    try:
                        os.unlink(temporary_name, dir_fd=root_fd)
                    except FileNotFoundError:
                        pass
        finally:
            os.close(task_fd)


def _validate_aggregated_task_name(task_name: str) -> None:
    if Path(task_name).name != task_name or task_name in {".", ".."}:
        raise UnsafeGluePathError(f"aggregated TASK_NAME must be one path component: {task_name}")


def copy_investigation_memory(paths: PipelinePaths, *, dry_run: bool) -> None:
    """Copy investigation memory into the ablations output directory."""
    source_root = paths.investigation_workspace_dir
    destination_root = paths.ablations_output_dir
    if not _real_directory_exists(source_root, role="memory source directory"):
        for name in MEMORY_FILES:
            logger.info("glue: investigation memory %s absent (skip)", source_root / name)
        return

    with _open_directory(source_root, role="memory source directory") as source_fd:
        if dry_run:
            for name in MEMORY_FILES:
                src = source_root / name
                if _regular_file_exists_at(source_fd, name, role="memory source"):
                    logger.info("glue: copy memory %s -> %s", src, destination_root / name)
                else:
                    logger.info("glue: investigation memory %s absent (skip)", src)
            return

        with _open_directory(
            destination_root, role="memory destination directory"
        ) as destination_fd:
            for name in MEMORY_FILES:
                src = source_root / name
                dst = destination_root / name
                if not _regular_file_exists_at(source_fd, name, role="memory source"):
                    logger.info("glue: investigation memory %s absent (skip)", src)
                    continue
                destination_exists = _regular_file_exists_at(
                    destination_fd,
                    name,
                    role="memory destination",
                )
                if destination_exists:
                    source_digest = _regular_file_digest_at(
                        source_fd,
                        name,
                        role="memory source",
                    )
                    destination_digest = _regular_file_digest_at(
                        destination_fd,
                        name,
                        role="memory destination",
                    )
                    if source_digest == destination_digest:
                        logger.info("glue: memory already current: %s (skip)", dst)
                        continue
                    logger.info("glue: refresh memory %s -> %s", src, dst)
                    _copy_regular_file_at(
                        source_fd,
                        destination_fd,
                        name,
                        replace_existing=True,
                    )
                    continue
                logger.info("glue: copy memory %s -> %s", src, dst)
                _copy_regular_file_at(source_fd, destination_fd, name)


def run_glue(paths: PipelinePaths, _cfg: PipelineConfig, *, dry_run: bool) -> None:
    """Expose optional aggregate convenience links without copying phase inputs."""
    link_aggregated_csvs(paths, dry_run=dry_run)
