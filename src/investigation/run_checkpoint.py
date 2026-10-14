"""Deterministic sidecar git checkpoints of a run directory.

The repository lives in a sibling sidecar git dir and is driven exclusively through
explicit ``--git-dir``/``--work-tree`` arguments. Container agents receive only the
run directory mount, so they cannot inspect or mutate checkpoint metadata.

Staging uses ``git add -f`` with pathspec exclusions rather than any ignore file.
Run directories contain agent-authored ``.gitignore`` files inside program trees
(``ideas/*/program_*/src/.gitignore`` listing ``results/``, ``eval-artifacts/``),
which outrank ``info/exclude`` and would silently drop the very outputs a
checkpoint exists to capture. The sidecar sits outside the work tree, so forced
staging cannot recursively add its metadata.

Checkpointing is non-gating: no function here raises, and a degraded checkpoint
never changes a run's exit code. Restoring or branching from a checkpoint is out
of scope; use ordinary git commands against the sidecar.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .checkpoint_git import (
    Quarantine as _Quarantine,
    commit_count as _commit_count,
    create_quarantined_commit as _create_quarantined_commit,
    ensure_repository as _ensure_repository,
    git_error as _git_error,
    head_commit as _head_commit,
    initialize_index as _initialize_index,
    promote_commit as _promote_commit,
    quarantine_store as _quarantine,
    run_git as _run_git,
    remaining_timeout_seconds as _remaining_timeout_seconds,
    scan_staged_index as _scan_staged_index,
    staged_paths as _staged_paths,
)
from .checkpoint_lock import (
    checkpoint_lock as _checkpoint_lock,
    ensure_private_git_dir as _ensure_private_git_dir,
)
from .checkpoint_paths import (
    MissingCheckpointRoot as _MissingCheckpointRoot,
    checkpoint_total_bytes as _checkpoint_total_bytes,
    pinned_checkpoint_paths as _pinned_checkpoint_paths,
    scan_excluded_directory_names as _scan_excluded_directory_names,
    scan_oversize as _scan_oversize,
    validate_git_dir_name as _validate_git_dir_name,
    write_pathspec as _write_pathspec,
)
from .checkpoint_security import (
    report_sensitive_path,
    redact_sensitive_text,
    scan_sensitive_paths,
)
from .constants import DEFAULT_GIT_DIR_NAME

logger = logging.getLogger(__name__)

_EXCLUDED_PATHSPEC_FILENAME = "checkpoint-excluded-pathspec"
_SUBJECT_MAX_CHARS = 72
_SLUG_MAX_CHARS = 40
_CHECKPOINT_STORED_EXTRA = {"checkpoint_stored": True}
_UNREADABLE_RE = re.compile(
    r"^error: (?:open|unable to index file) [\"']?(.+?)[\"']?$", re.MULTILINE
)
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")

CheckpointStatus = Literal["committed", "unchanged", "disabled", "degraded"]


@dataclass(frozen=True)
class CheckpointSettings:
    """Resolved checkpoint knobs shared by the pipeline parent and wrapper child."""

    enabled: bool = True
    git_dir_name: str = DEFAULT_GIT_DIR_NAME
    max_file_bytes: int = 0
    max_total_bytes: int = 0
    timeout_seconds: int = 600
    tag_checkpoints: bool = True
    final_include_environments: bool = True
    _deadline: float | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CheckpointResult:
    """Observable outcome of one non-gating checkpoint commit."""

    status: CheckpointStatus
    label: str
    commit: str | None = None
    tag: str | None = None
    files: int = 0
    skipped_oversize: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None
    skipped_sensitive: tuple[str, ...] = field(default_factory=tuple)


def describe_checkpoint(
    *, run_root: Path, label: str, settings: CheckpointSettings, final: bool = False
) -> str:
    """Describe the checkpoint a real run would take, for dry-run previews."""
    scope = (
        "full run directory"
        if final and settings.final_include_environments
        else "run artifacts without derived environments"
        if final
        else "run artifacts"
    )
    file_cap = (
        "no per-file cap"
        if settings.max_file_bytes <= 0
        else f"per-file cap {settings.max_file_bytes}B"
    )
    total_cap = (
        "no aggregate cap"
        if settings.max_total_bytes <= 0
        else f"aggregate cap {settings.max_total_bytes}B"
    )
    git_dir = report_sensitive_path(str(checkpoint_git_dir(run_root, settings.git_dir_name)))
    return (
        f"checkpoint '{label}' -> {git_dir} "
        f"({scope}, credentials always excluded, {file_cap}, {total_cap})"
    )


def checkpoint_git_dir(run_root: Path, git_dir_name: str) -> Path:
    """Return the sibling metadata directory kept outside the agent work tree."""
    git_dir_name = _validate_git_dir_name(git_dir_name)
    root = Path(run_root).expanduser().resolve()
    if not root.name:
        raise ValueError("checkpoint run root cannot be the filesystem root")
    git_dir = root.parent / f".{root.name}{git_dir_name}"
    if git_dir == root or git_dir.is_relative_to(root):
        raise ValueError(f"checkpoint git dir must be outside the run root: {git_dir}")
    return git_dir


def write_checkpoint(
    *,
    run_root: Path,
    label: str,
    phase: str,
    settings: CheckpointSettings,
    run_id: str | None = None,
    cycle: int | None = None,
    idea_id: str | None = None,
    agent: str | None = None,
    event: str | None = None,
    final: bool = False,
    pin_run_root: bool = False,
) -> CheckpointResult:
    """Commit the run directory. Never raises; failures degrade to a result."""
    if not settings.enabled:
        return CheckpointResult(status="disabled", label=label)
    settings = replace(settings, _deadline=time.monotonic() + settings.timeout_seconds)
    try:
        if pin_run_root:
            try:
                with _pinned_checkpoint_paths(
                    Path(run_root), git_dir_name=settings.git_dir_name
                ) as pinned:
                    return _checkpoint(
                        root=pinned.root,
                        git_dir=pinned.git_dir,
                        label=label,
                        phase=phase,
                        settings=settings,
                        run_id=run_id,
                        cycle=cycle,
                        idea_id=idea_id,
                        agent=agent,
                        event=event,
                        final=final,
                    )
            except _MissingCheckpointRoot:
                return CheckpointResult(
                    status="disabled",
                    label=label,
                    error=redact_sensitive_text(f"run root is not a directory: {run_root}"),
                )
        root = Path(run_root).expanduser().resolve()
        if not root.is_dir():
            return CheckpointResult(
                status="disabled",
                label=label,
                error=redact_sensitive_text(f"run root is not a directory: {root}"),
            )
        return _checkpoint(
            root=root,
            git_dir=checkpoint_git_dir(root, settings.git_dir_name),
            label=label,
            phase=phase,
            settings=settings,
            run_id=run_id,
            cycle=cycle,
            idea_id=idea_id,
            agent=agent,
            event=event,
            final=final,
        )
    except subprocess.TimeoutExpired as exc:
        error = redact_sensitive_text(f"git timed out: {exc}")
        return CheckpointResult(status="degraded", label=label, error=error)
    except OSError as exc:
        return CheckpointResult(
            status="degraded", label=label, error=redact_sensitive_text(str(exc))
        )
    except Exception as exc:  # checkpointing must never fail a science run
        error = redact_sensitive_text(f"{type(exc).__name__}: {exc}")
        return CheckpointResult(status="degraded", label=label, error=error)


def _checkpoint(
    *,
    root: Path,
    git_dir: Path,
    label: str,
    phase: str,
    settings: CheckpointSettings,
    run_id: str | None,
    cycle: int | None,
    idea_id: str | None,
    agent: str | None,
    event: str | None,
    final: bool,
) -> CheckpointResult:
    _ensure_private_git_dir(git_dir)
    with _checkpoint_lock(
        git_dir, timeout_seconds=_remaining_timeout_seconds(settings)
    ) as acquired:
        if not acquired:
            return CheckpointResult(status="degraded", label=label, error="checkpoint lock busy")
        worktree_oversize = _scan_oversize(root, settings=settings, final=final)
        known_sensitive = scan_sensitive_paths(
            root,
            skipped_directory_names=_scan_excluded_directory_names(settings=settings, final=final),
            checkpoint_git_dir_name=settings.git_dir_name,
            deadline=settings._deadline,
        )
        _remaining_timeout_seconds(settings)
        included_bytes = _checkpoint_total_bytes(
            root,
            settings=settings,
            final=final,
            sensitive=known_sensitive,
        )
        if settings.max_total_bytes > 0 and included_bytes > settings.max_total_bytes:
            return CheckpointResult(
                status="degraded",
                label=label,
                skipped_oversize=tuple(
                    report_sensitive_path(path) for path in worktree_oversize
                ),
                skipped_sensitive=tuple(
                    report_sensitive_path(path) for path in known_sensitive
                ),
                error=f"checkpoint byte budget exceeded ({settings.max_total_bytes} bytes)",
            )
        _remaining_timeout_seconds(settings)
        _ensure_repository(git_dir, root=root, settings=settings)
        with _quarantine(git_dir) as quarantine:
            parent = _head_commit(git_dir, root, settings)
            initialized = _initialize_index(
                git_dir, root, settings, quarantine=quarantine, parent=parent
            )
            if initialized is not None:
                return CheckpointResult(status="degraded", label=label, error=initialized)
            pathspec = _write_pathspec(
                quarantine.index.parent,
                settings=settings,
                final=final,
                oversize=worktree_oversize,
                sensitive=known_sensitive,
            )
            added = _run_git(
                git_dir,
                root,
                settings,
                "add",
                "-A",
                "-f",
                "--ignore-errors",
                f"--pathspec-from-file={pathspec}",
                "--pathspec-file-nul",
                quarantine=quarantine,
            )
            if added.returncode not in (0, 1):
                return CheckpointResult(
                    status="degraded",
                    label=label,
                    skipped_oversize=tuple(
                        report_sensitive_path(path) for path in worktree_oversize
                    ),
                    error=_git_error("add", added),
                )
            staged_scan = _scan_staged_index(
                git_dir,
                root,
                settings,
                quarantine=quarantine,
                final=final,
            )
            if (
                settings.max_total_bytes > 0
                and staged_scan.total_bytes > settings.max_total_bytes
            ):
                return CheckpointResult(
                    status="degraded",
                    label=label,
                    error=(
                        "checkpoint staged byte budget exceeded "
                        f"({settings.max_total_bytes} bytes)"
                    ),
                )
            oversize_removals = tuple(sorted(set(worktree_oversize) | set(staged_scan.oversized)))
            oversize = tuple(report_sensitive_path(path) for path in oversize_removals)
            sensitive_removals = tuple(sorted(set(known_sensitive) | set(staged_scan.excluded)))
            sensitive = tuple(
                sorted(
                    {report_sensitive_path(path) for path in known_sensitive}
                    | set(staged_scan.reported)
                )
            )
            exclusion_removal = _remove_excluded_from_index(
                git_dir,
                root,
                settings,
                excluded=tuple(sorted(set(sensitive_removals) | set(oversize_removals))),
                quarantine=quarantine,
            )
            if exclusion_removal is not None:
                return CheckpointResult(
                    status="degraded",
                    label=label,
                    skipped_oversize=oversize,
                    skipped_sensitive=sensitive,
                    error=exclusion_removal,
                )
            unreadable = tuple(
                report_sensitive_path(path) for path in _UNREADABLE_RE.findall(added.stderr or "")
            )
            partial = f"unreadable paths skipped: {', '.join(unreadable)}" if unreadable else None
            staged = _staged_paths(git_dir, root, settings, quarantine=quarantine)
            if not staged and not final:
                return CheckpointResult(
                    status="unchanged",
                    label=label,
                    skipped_oversize=oversize,
                    skipped_sensitive=sensitive,
                    error=partial,
                )

            stamp = _timestamp()
            sequence = _commit_count(git_dir, root, settings) + 1
            subject, trailers = _commit_message(
                label=label,
                phase=phase,
                sequence=sequence,
                run_id=run_id,
                cycle=cycle,
                idea_id=idea_id,
                agent=agent,
                event=event,
                files=len(staged),
                oversize=oversize,
                sensitive=sensitive,
            )
            commit_sha, commit_error = _create_quarantined_commit(
                git_dir,
                root,
                settings,
                quarantine=quarantine,
                parent=parent,
                message=f"{subject}\n\n{trailers}\n",
                stamp=stamp,
            )
            if commit_error is not None or commit_sha is None:
                return CheckpointResult(
                    status="degraded",
                    label=label,
                    skipped_oversize=oversize,
                    skipped_sensitive=sensitive,
                    error=commit_error,
                )
            promoted = _promote_commit(
                git_dir,
                root,
                settings,
                quarantine=quarantine,
                commit=commit_sha,
                parent=parent,
            )
            if promoted is not None:
                return CheckpointResult(
                    status="degraded",
                    label=label,
                    skipped_oversize=oversize,
                    skipped_sensitive=sensitive,
                    error=promoted,
                )
        tag = _create_tag(git_dir, root, settings, sequence=sequence, label=label, final=final)
        if final:
            try:
                _run_git(git_dir, root, settings, "repack", "-ad", "-q")
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning("Checkpoint repack skipped: %s", redact_sensitive_text(str(exc)))
        _warn_exclusions(
            label=label,
            sensitive=sensitive,
            oversize=oversize,
            max_file_bytes=settings.max_file_bytes,
        )

    return CheckpointResult(
        status="committed",
        label=label,
        commit=commit_sha,
        tag=tag,
        files=len(staged),
        skipped_oversize=oversize,
        skipped_sensitive=sensitive,
        error=partial,
    )


def _warn_exclusions(
    *,
    label: str,
    sensitive: tuple[str, ...],
    oversize: tuple[str, ...],
    max_file_bytes: int,
) -> None:
    if sensitive:
        logger.warning(
            "Checkpoint '%s' excluded %d sensitive path(s): %s",
            label,
            len(sensitive),
            ", ".join(sensitive[:5]),
            extra=_CHECKPOINT_STORED_EXTRA,
        )
    if oversize:
        logger.warning(
            "Checkpoint '%s' skipped %d file(s) over the %d byte cap: %s",
            label,
            len(oversize),
            max_file_bytes,
            ", ".join(oversize[:5]),
            extra=_CHECKPOINT_STORED_EXTRA,
        )


def _remove_excluded_from_index(
    git_dir: Path,
    root: Path,
    settings: CheckpointSettings,
    *,
    excluded: tuple[str, ...],
    quarantine: _Quarantine,
) -> str | None:
    """Remove excluded paths that may have existed in an earlier tree."""
    if not excluded:
        return None
    pathspec = quarantine.index.parent / _EXCLUDED_PATHSPEC_FILENAME
    entries = (f":(literal){path}".encode("utf-8") for path in excluded)
    pathspec.write_bytes(b"\0".join(entries) + b"\0")
    removed = _run_git(
        git_dir,
        root,
        settings,
        "rm",
        "--cached",
        "-r",
        "-q",
        "-f",
        "--ignore-unmatch",
        f"--pathspec-from-file={pathspec}",
        "--pathspec-file-nul",
        quarantine=quarantine,
    )
    return _git_error("rm-excluded", removed) if removed.returncode != 0 else None


def _timestamp() -> str:
    """Stamp author and committer identically, with a fixed UTC offset."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+0000")


def _sanitize(value: object) -> str:
    """Flatten a trailer value so agent-authored text cannot forge trailer lines."""
    return " ".join(str(value).split())


def _slug(label: str) -> str:
    slug = _SLUG_STRIP_RE.sub("-", label.lower()).strip("-")
    return slug[:_SLUG_MAX_CHARS].strip("-") or "checkpoint"


def _commit_message(
    *,
    label: str,
    phase: str,
    sequence: int,
    run_id: str | None,
    cycle: int | None,
    idea_id: str | None,
    agent: str | None,
    event: str | None,
    files: int,
    oversize: tuple[str, ...],
    sensitive: tuple[str, ...],
) -> tuple[str, str]:
    subject = f"checkpoint: {_sanitize(label)}"[:_SUBJECT_MAX_CHARS]
    candidates: list[tuple[str, object | None]] = [
        ("Checkpoint-Run-Id", run_id),
        ("Checkpoint-Phase", phase),
        ("Checkpoint-Sequence", sequence),
        ("Checkpoint-Event", event),
        ("Checkpoint-Cycle", cycle),
        ("Checkpoint-Idea-Id", idea_id),
        ("Checkpoint-Agent", agent),
        ("Checkpoint-Files", files),
        ("Checkpoint-Skipped-Oversize", len(oversize) if oversize else None),
        ("Checkpoint-Skipped-Sensitive", len(sensitive) if sensitive else None),
    ]
    trailers = "\n".join(
        f"{key}: {_sanitize(value)}" for key, value in candidates if value is not None
    )
    return subject, trailers


def _create_tag(
    git_dir: Path,
    root: Path,
    settings: CheckpointSettings,
    *,
    sequence: int,
    label: str,
    final: bool,
) -> str | None:
    """Create a lightweight navigation tag. Failure is a warning, never fatal."""
    if not settings.tag_checkpoints:
        return None
    name = f"checkpoint/{sequence:04d}-{'final' if final else _slug(label)}"
    try:
        tagged = _run_git(git_dir, root, settings, "tag", name, "HEAD")
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Checkpoint tag %s not created: %s", name, redact_sensitive_text(str(exc)))
        return None
    if tagged.returncode != 0:
        logger.warning("Checkpoint tag %s not created: %s", name, _git_error("tag", tagged))
        return None
    return name
