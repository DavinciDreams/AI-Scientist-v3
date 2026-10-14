"""Transactional publication helpers for review-driven paper revisions."""

from __future__ import annotations

import logging
import os
import shutil
import stat
import tempfile
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.orchestrator.paper_manifest import paper_publication_lock, write_manifest_atomically
from src.investigation.paper_revision_workspace import (
    open_pinned_directory,
    same_filesystem_entry,
    validate_regular_tree,
)

from .portable_publication import (
    PortablePublicationError,
    atomic_exchange_prepared as _atomic_exchange,
    cancel_prepared_replacement,
    directory_identity,
    exchange_is_unsupported,
    finish_directory_replacement,
    portable_replacement_pending,
    prepare_directory_replacement,
    recover_portable_replacement,
    sync_directory_tree as _sync_publication_tree,
)
from .revision_artifacts import (
    atomic_write_file as _atomic_write_file,
    read_json as _read_json_bounded,
    read_regular as _read_regular,
    reject_symlink_components as _reject_symlink_components,
    write_json as _write_json,
)
from .revision_audit_publication import copy_audit_bound_inputs
from .revision_ancillary_archive import archive_original_only_inputs
from .revision_manifest import rebase_manifest

logger = logging.getLogger(__name__)

_MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
_MAX_REVIEW_BYTES = 1024 * 1024
_MAX_COPY_ENTRIES = 100_000
_MAX_COPY_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_COPY_DEPTH = 64
_REQUIRED_REVISION_FILES = (
    "paper.tex",
    "paper.pdf",
    "claims_grounding.json",
    "references.bib",
    "references_provenance.json",
    "validation_report.json",
    "latest_manifest.json",
)
_PROMOTED_FILES = (
    "paper.tex",
    "paper.pdf",
    "paper_data.json",
    "paper_prompt.md",
    "claims_grounding.json",
    "references.bib",
    "references_provenance.json",
    "figure_manifest.json",
    "validation_report.json",
    "validation_feedback.json",
    "validation_feedback.md",
    "paper_evidence_audit.json",
    "paper_evidence_audit.md",
    "paper_evidence_audit_feedback.json",
    "paper_evidence_audit_feedback.md",
    "method_scope_audit.json",
    "method_scope_audit.md",
)
_SNAPSHOT_FILES = _PROMOTED_FILES
_REVISION_SOURCE_FILES = (*_PROMOTED_FILES, "latest_manifest.json")


@dataclass
class _CopyBudget:
    entries: int = 0
    total_bytes: int = 0

    def add(self, *, size: int = 0) -> None:
        self.entries += 1
        self.total_bytes += size
        if self.entries > _MAX_COPY_ENTRIES:
            raise ValueError("publication copy has too many entries")
        if self.total_bytes > _MAX_COPY_TOTAL_BYTES:
            raise ValueError("publication copy is too large")


def _atomic_copy_file(source: Path, target: Path) -> None:
    _atomic_write_file(target, _read_regular(source, max_bytes=_MAX_ARTIFACT_BYTES))


def _atomic_copy_file_at(
    source_fd: int,
    name: str,
    source: Path,
    target: Path,
    expected: os.stat_result,
) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(name, flags, dir_fd=source_fd)
    with os.fdopen(descriptor, "rb") as stream:
        actual = os.fstat(stream.fileno())
        if not stat.S_ISREG(actual.st_mode) or not same_filesystem_entry(expected, actual):
            raise ValueError(f"publication source changed while opening it: {source}")
        content = stream.read(_MAX_ARTIFACT_BYTES + 1)
    if len(content) > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"file is too large: {source}")
    _atomic_write_file(target, content)


def _copy_named_file(source_root: Path, name: str, target: Path) -> None:
    source_fd = open_pinned_directory(source_root)
    try:
        metadata = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"expected regular file: {source_root / name}")
        _atomic_copy_file_at(source_fd, name, source_root / name, target, metadata)
    finally:
        os.close(source_fd)


def _copy_present_named_files(
    source_root: Path, target_root: Path, names: Collection[str]
) -> list[str]:
    source_fd = open_pinned_directory(source_root)
    copied: list[str] = []
    try:
        for name in names:
            try:
                metadata = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"expected regular file: {source_root / name}")
            _atomic_copy_file_at(
                source_fd,
                name,
                source_root / name,
                target_root / name,
                metadata,
            )
            copied.append(name)
    finally:
        os.close(source_fd)
    return copied


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_bounded(path, max_bytes=_MAX_ARTIFACT_BYTES)


def _validate_revision_workspace(
    revision_dir: Path, *, manifest_artifact_root: Path | None = None
) -> dict[str, Any]:
    _reject_symlink_components(revision_dir)
    if not revision_dir.is_dir():
        raise ValueError(f"revision directory must be a regular directory: {revision_dir}")
    for name in _REQUIRED_REVISION_FILES:
        _read_regular(revision_dir / name, max_bytes=_MAX_ARTIFACT_BYTES)
    for name in _PROMOTED_FILES:
        candidate = revision_dir / name
        if candidate.exists() or candidate.is_symlink():
            _read_regular(candidate, max_bytes=_MAX_ARTIFACT_BYTES)
    figures = revision_dir / "figures"
    if figures.exists() or figures.is_symlink():
        validate_regular_tree(figures)
    manifest = _read_json(revision_dir / "latest_manifest.json")
    artifact_root = manifest_artifact_root or revision_dir
    if manifest.get("paper_pdf") != str((artifact_root / "paper.pdf").resolve()):
        raise ValueError("revision manifest does not identify the revised PDF")
    report = _read_json(revision_dir / "validation_report.json")
    if report.get("ok") is not True:
        raise ValueError("revision validation did not pass")
    return manifest


def _copy_directory_contents(
    source_root: Path,
    target_root: Path,
    *,
    workspace_alias_targets: Mapping[str, Collection[Path]] | None = None,
) -> list[str]:
    source_fd = open_pinned_directory(source_root)
    copied_files: list[str] = []
    budget = _CopyBudget()

    def copy_directory(
        directory_fd: int,
        source_directory: Path,
        target_directory: Path,
        relative_directory: Path,
        depth: int,
    ) -> None:
        if depth > _MAX_COPY_DEPTH:
            raise ValueError("publication copy is too deep")
        for name in sorted(os.listdir(directory_fd)):
            source = source_directory / name
            relative = relative_directory / name
            target = target_directory / name
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            mode = metadata.st_mode
            budget.add(size=metadata.st_size if stat.S_ISREG(mode) else 0)
            if stat.S_ISLNK(mode):
                expected_targets = (
                    workspace_alias_targets.get(name) if workspace_alias_targets else None
                )
                if len(relative.parts) != 1 or not expected_targets:
                    raise ValueError(f"publication tree contains an unexpected symlink: {source}")
                try:
                    link_value = os.readlink(name, dir_fd=directory_fd)
                    link_target = Path(link_value)
                    if not link_target.is_absolute():
                        link_target = source.parent / link_target
                    resolved_source = link_target.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise ValueError(f"workspace alias target is unavailable: {source}") from exc
                resolved_expected = set()
                for expected_target in expected_targets:
                    try:
                        resolved_expected.add(expected_target.resolve(strict=True))
                    except (OSError, RuntimeError):
                        continue
                if resolved_source not in resolved_expected or not resolved_source.is_dir():
                    raise ValueError(
                        f"publication tree contains a noncanonical workspace alias: {source}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(link_value, target, target_is_directory=True)
            elif stat.S_ISDIR(mode):
                flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                child_fd = os.open(name, flags, dir_fd=directory_fd)
                opened = os.fstat(child_fd)
                if not stat.S_ISDIR(opened.st_mode) or not same_filesystem_entry(metadata, opened):
                    os.close(child_fd)
                    raise ValueError(f"publication directory changed while opening it: {source}")
                target.mkdir(parents=True, exist_ok=True)
                try:
                    copy_directory(child_fd, source, target, relative, depth + 1)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(mode):
                _atomic_copy_file_at(directory_fd, name, source, target, metadata)
                copied_files.append(str(relative))
            else:
                raise ValueError(f"publication tree contains a special file: {source}")

    try:
        copy_directory(source_fd, source_root, target_root, Path(), 0)
    finally:
        os.close(source_fd)
    return copied_files


def _copy_regular_tree(
    source_root: Path,
    target_root: Path,
    *,
    workspace_alias_targets: Mapping[str, Collection[Path]],
) -> None:
    _copy_directory_contents(
        source_root,
        target_root,
        workspace_alias_targets=workspace_alias_targets,
    )


def _snapshot_original(*, source_dir: Path, staged_dir: Path, before_review_path: Path) -> Path:
    snapshot_dir = staged_dir / "revisions" / "revision_01" / "before"
    if snapshot_dir.exists() or snapshot_dir.is_symlink():
        raise ValueError(f"revision snapshot already exists: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True)
    present = _copy_present_named_files(source_dir, snapshot_dir, _SNAPSHOT_FILES)
    figure_files: list[str] = []
    _copy_named_file(source_dir, "latest_manifest.json", snapshot_dir / "original_manifest.json")
    _copy_named_file(source_dir, before_review_path.name, snapshot_dir / before_review_path.name)
    figures = source_dir / "figures"
    if figures.exists() or figures.is_symlink():
        figure_files.extend(_copy_directory_contents(figures, snapshot_dir / "figures"))
    _write_json(
        snapshot_dir / "snapshot_complete.json",
        {"files": present, "figure_files": figure_files, "status": "complete"},
    )
    return snapshot_dir


def _copy_publication(revision_dir: Path, paper_dir: Path) -> None:
    _copy_present_named_files(revision_dir, paper_dir, _PROMOTED_FILES)
    figures = revision_dir / "figures"
    if figures.exists():
        _copy_directory_contents(figures, paper_dir / "figures")


def _snapshot_revision_source(revision_dir: Path, snapshot_dir: Path) -> None:
    _copy_present_named_files(revision_dir, snapshot_dir, _REVISION_SOURCE_FILES)
    figures = revision_dir / "figures"
    if figures.exists() or figures.is_symlink():
        _copy_directory_contents(figures, snapshot_dir / "figures")


def _remove_temporary_tree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError:
        logger.warning("Could not remove temporary revision publication tree: %s", path.name)


def _record_dir(paper_dir: Path) -> Path:
    return paper_dir / "revisions" / "revision_01"


def initial_review_must_be_preserved(paper_dir: Path) -> bool:
    """Return whether a replay must retain the immutable before-revision review."""
    try:
        manifest = _read_json(paper_dir / "latest_manifest.json")
    except (FileNotFoundError, OSError, ValueError):
        manifest = {}
    before = manifest.get("before_revision")
    named = paper_dir / "ai_review.before_revision.json"
    if isinstance(before, dict) and before.get("status") == "completed":
        review_path = before.get("review_json")
        if isinstance(review_path, str) and Path(review_path).resolve() == named.resolve():
            try:
                return isinstance(_read_json(named), dict)
            except (OSError, ValueError):
                return True
    revision = manifest.get("revision")
    if isinstance(revision, dict) and revision.get("status") == "promoted":
        return True
    journal = _record_dir(paper_dir) / "promotion_state.json"
    if not journal.exists() and not journal.is_symlink():
        return False
    state = _read_json(journal)
    return state.get("status") == "promoting"


def recover_incomplete_promotion(paper_dir: Path) -> None:
    """Finish portable publication or reject an interrupted legacy publication."""
    with paper_publication_lock(paper_dir):
        recover_incomplete_promotion_locked(paper_dir)


def _validate_publication_transition(source_dir: Path, candidate_dir: Path) -> None:
    from .revision_evidence import validate_published_revision_evidence

    validate_published_revision_evidence(source_dir, candidate_dir)


def recover_incomplete_promotion_locked(paper_dir: Path) -> None:
    """Recover one paper while its parent-scoped publication lock is already held."""
    recover_portable_replacement(
        paper_dir,
        validate_transition=_validate_publication_transition,
    )
    if portable_replacement_pending(paper_dir):
        raise ValueError("portable revision publication cleanup remains incomplete")
    record_dir = _record_dir(paper_dir)
    journal_path = record_dir / "promotion_state.json"
    if not journal_path.exists() and not journal_path.is_symlink():
        return
    state = _read_json(journal_path)
    if state.get("status") == "promoting":
        raise ValueError("legacy interrupted revision publication requires manual recovery")


def promote_revision_workspace(
    *,
    paper_dir: Path,
    revision_dir: Path,
    before_review_path: Path,
    workspace_alias_targets: Mapping[str, Collection[Path]] | None = None,
) -> Path:
    """Publish a validated revision with create-once snapshot and rollback."""
    paper_dir = Path(os.path.abspath(paper_dir.expanduser()))
    revision_dir = Path(os.path.abspath(revision_dir.expanduser()))
    before_review_path = Path(os.path.abspath(before_review_path.expanduser()))
    try:
        _reject_symlink_components(paper_dir)
    except ValueError as exc:
        raise ValueError(f"paper directory must be a regular directory: {paper_dir}") from exc
    if not paper_dir.is_dir():
        raise ValueError(f"paper directory must be a regular directory: {paper_dir}")
    _read_regular(before_review_path, max_bytes=_MAX_REVIEW_BYTES)
    with paper_publication_lock(paper_dir):
        if portable_replacement_pending(paper_dir):
            raise PortablePublicationError("portable publication journal already exists")
        staged_dir = Path(
            tempfile.mkdtemp(prefix=f".{paper_dir.name}.revision-", dir=paper_dir.parent)
        )
        revision_snapshot = Path(tempfile.mkdtemp(prefix=".revision-source-", dir=paper_dir.parent))
        exchange_completed = False
        try:
            original_identity = directory_identity(paper_dir)
            _snapshot_revision_source(revision_dir, revision_snapshot)
            evidence_audit = _read_json(revision_snapshot / "paper_evidence_audit.json")
            copy_audit_bound_inputs(
                revision_dir,
                revision_snapshot,
                evidence_audit.get("input_sha256"),
            )
            manifest = _validate_revision_workspace(
                revision_snapshot,
                manifest_artifact_root=revision_dir,
            )
            _copy_regular_tree(
                paper_dir,
                staged_dir,
                workspace_alias_targets=workspace_alias_targets or {},
            )
            current_manifest = _read_json(staged_dir / "latest_manifest.json")
            snapshot_dir = _snapshot_original(
                source_dir=staged_dir,
                staged_dir=staged_dir,
                before_review_path=staged_dir / before_review_path.name,
            )
            archive_original_only_inputs(
                staged_dir, revision_snapshot, snapshot_dir, evidence_audit.get("input_sha256")
            )
            _copy_publication(revision_snapshot, staged_dir)
            copy_audit_bound_inputs(
                revision_snapshot,
                staged_dir,
                evidence_audit.get("input_sha256"),
            )
            promoted = rebase_manifest(manifest, old_root=revision_dir, new_root=paper_dir)
            assert isinstance(promoted, dict)
            final_snapshot_dir = paper_dir / snapshot_dir.relative_to(staged_dir)
            promoted["revision"] = {
                "status": "promoted",
                "source_dir": str(revision_dir),
                "original_snapshot": str(final_snapshot_dir),
                "before_review": str(before_review_path),
            }
            before_status = current_manifest.get("before_revision")
            promoted["before_revision"] = (
                before_status
                if isinstance(before_status, dict)
                else {
                    "status": "completed",
                    "review_json": str(before_review_path),
                    "log": str(paper_dir / "ai_review.before_revision.log"),
                }
            )
            write_manifest_atomically(staged_dir / "latest_manifest.json", promoted)
            _write_json(
                staged_dir / "revisions" / "revision_01" / "promotion_state.json",
                {"status": "completed", "revision_dir": str(revision_dir)},
            )
            from .revision_evidence import validate_published_revision_evidence

            _sync_publication_tree(staged_dir)
            validate_published_revision_evidence(snapshot_dir, staged_dir)
            replacement_identity = directory_identity(staged_dir)
            record = prepare_directory_replacement(
                paper_dir,
                staged_dir,
                expected_original=original_identity,
                expected_replacement=replacement_identity,
            )
            try:
                _atomic_exchange(record, parent=paper_dir.parent)
            except OSError as exc:
                if not exchange_is_unsupported(exc):
                    cancel_prepared_replacement(paper_dir, record)
                    raise
            finish_directory_replacement(
                paper_dir,
                record,
                validate_transition=_validate_publication_transition,
            )
            exchange_completed = True
        except Exception:
            if not exchange_completed and not portable_replacement_pending(paper_dir):
                _remove_temporary_tree(staged_dir)
            _remove_temporary_tree(revision_snapshot)
            raise
        _remove_temporary_tree(revision_snapshot)
        _remove_temporary_tree(staged_dir)
    return final_snapshot_dir
