"""Safe preparation helpers for review-driven paper revisions."""

from __future__ import annotations

import json
import os
import shutil
import stat
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_REFERENCE_ARTIFACT_BYTES = 100 * 1024 * 1024
REFERENCE_ARTIFACTS = ("references.bib", "references_provenance.json")
PREVIOUS_PAPER_FILES = frozenset(
    {
        "VENUE.md",
        "claims_grounding.json",
        "figure_manifest.json",
        "latest_manifest.json",
        "method_scope_audit.json",
        "method_scope_audit.md",
        "paper.pdf",
        "paper.tex",
        "paper_authors.tex",
        "paper_data.json",
        "paper_evidence_audit.json",
        "paper_evidence_audit.md",
        "paper_prompt.md",
        "paper_strategy.json",
        "paper_strategy.md",
        "references.bib",
        "references_provenance.json",
        "validation_feedback.json",
        "validation_feedback.md",
        "validation_report.json",
    }
)
SOURCE_EVIDENCE_DIRECTORY = "source_evidence"
PREVIOUS_PAPER_DIRECTORIES = frozenset(
    {"appendix", "assets", "figures", "sections", SOURCE_EVIDENCE_DIRECTORY}
)
MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
MAX_SNAPSHOT_FILES = 5000
MAX_PAPER_DATA_BYTES = 32 * 1024 * 1024


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path.expanduser()))
    for component in reversed((absolute, *absolute.parents)):
        if component == Path(component.anchor):
            continue
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"path contains a symlink: {component}")


def same_filesystem_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def open_pinned_directory(path: Path) -> int:
    absolute = Path(os.path.abspath(path.expanduser()))
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        if "descriptor" in locals():
            os.close(descriptor)
        raise ValueError(f"could not open regular directory: {absolute}: {exc}") from None
    return descriptor


def validate_regular_tree(source_root: Path) -> None:
    """Validate a directory tree without following mutable path components."""
    source_fd = open_pinned_directory(source_root)

    def validate_directory(directory_fd: int, source_directory: Path) -> None:
        for name in sorted(os.listdir(directory_fd)):
            source = source_directory / name
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            mode = metadata.st_mode
            if stat.S_ISDIR(mode):
                flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                child_fd = os.open(name, flags, dir_fd=directory_fd)
                opened = os.fstat(child_fd)
                if not stat.S_ISDIR(opened.st_mode) or not same_filesystem_entry(metadata, opened):
                    os.close(child_fd)
                    raise ValueError(f"tree directory changed while opening it: {source}")
                try:
                    validate_directory(child_fd, source)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(mode):
                flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
                flags |= getattr(os, "O_NONBLOCK", 0)
                descriptor = os.open(name, flags, dir_fd=directory_fd)
                actual = os.fstat(descriptor)
                os.close(descriptor)
                if not stat.S_ISREG(actual.st_mode) or not same_filesystem_entry(metadata, actual):
                    raise ValueError(f"tree file changed while opening it: {source}")
            else:
                raise ValueError(f"tree contains a non-regular entry: {source}")

    try:
        validate_directory(source_fd, source_root)
    finally:
        os.close(source_fd)


def _read_regular_at(directory_fd: int, name: str, path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"Could not preserve revision reference artifact {path}: {exc}") from None
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"Revision reference artifact must be regular: {path}")
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"Revision reference artifact is too large: {path}")
    return content


def _write_exclusive_at(directory_fd: int, name: str, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o400, dir_fd=directory_fd)
    with os.fdopen(descriptor, "wb") as output:
        output.write(content)


def preserve_revision_references(source_dir: Path, baseline_dir: Path) -> None:
    """Copy immutable reference artifacts without following links."""
    _reject_symlink_components(source_dir)
    _reject_symlink_components(baseline_dir)
    source_fd = open_pinned_directory(source_dir)
    try:
        baseline_fd = open_pinned_directory(baseline_dir)
        try:
            for name in REFERENCE_ARTIFACTS:
                content = _read_regular_at(
                    source_fd,
                    name,
                    source_dir / name,
                    max_bytes=MAX_REFERENCE_ARTIFACT_BYTES,
                )
                _write_exclusive_at(baseline_fd, name, content)
        finally:
            os.close(baseline_fd)
    finally:
        os.close(source_fd)


def _skip_reason(mode: int) -> str | None:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "named_pipe"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISBLK(mode) or stat.S_ISCHR(mode):
        return "device_file"
    if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
        return "non_regular"
    return None


@dataclass
class _SnapshotBudget:
    files: int = 0
    bytes: int = 0

    def start_file(self, source: Path) -> None:
        self.files += 1
        if self.files > MAX_SNAPSHOT_FILES:
            raise ValueError(f"previous-paper snapshot has too many files: {source}")

    def add_bytes(self, source: Path, count: int) -> None:
        self.bytes += count
        if self.bytes > MAX_SNAPSHOT_BYTES:
            raise ValueError(f"previous-paper snapshot is too large: {source}")


def _copy_regular_file_at(
    source_fd: int,
    name: str,
    source: Path,
    destination_fd: int,
    expected: os.stat_result,
    budget: _SnapshotBudget,
) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(name, flags, dir_fd=source_fd)
    with os.fdopen(descriptor, "rb") as input_stream:
        actual = os.fstat(input_stream.fileno())
        if not stat.S_ISREG(actual.st_mode) or not same_filesystem_entry(expected, actual):
            raise ValueError(f"snapshot source is not a regular file: {source}")
        budget.start_file(source)
        output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        output_flags |= getattr(os, "O_NOFOLLOW", 0)
        output_descriptor = os.open(name, output_flags, 0o600, dir_fd=destination_fd)
        with os.fdopen(output_descriptor, "wb") as output_stream:
            while chunk := input_stream.read(1024 * 1024):
                budget.add_bytes(source, len(chunk))
                output_stream.write(chunk)


def _open_child_directory_at(parent_fd: int, name: str, metadata: os.stat_result) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(name, flags, dir_fd=parent_fd)
    opened = os.fstat(directory_fd)
    if not stat.S_ISDIR(opened.st_mode) or not same_filesystem_entry(metadata, opened):
        os.close(directory_fd)
        raise ValueError(f"snapshot directory changed while opening it: {name}")
    return directory_fd


def _copy_directory_contents(
    source_fd: int,
    source: Path,
    destination_fd: int,
    budget: _SnapshotBudget,
    skipped: list[dict[str, str]],
) -> None:
    for name in sorted(os.listdir(source_fd)):
        child = source / name
        metadata = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        reason = _skip_reason(metadata.st_mode)
        if reason is not None:
            skipped.append({"path": str(child), "reason": reason})
        elif stat.S_ISDIR(metadata.st_mode):
            _copy_directory_at(source_fd, child, destination_fd, metadata, budget, skipped)
        else:
            _copy_regular_file_at(source_fd, name, child, destination_fd, metadata, budget)


def _copy_directory_at(
    parent_fd: int,
    source: Path,
    destination_parent_fd: int,
    metadata: os.stat_result,
    budget: _SnapshotBudget,
    skipped: list[dict[str, str]],
) -> None:
    directory_fd = _open_child_directory_at(parent_fd, source.name, metadata)
    try:
        os.mkdir(source.name, mode=0o700, dir_fd=destination_parent_fd)
        target_metadata = os.stat(source.name, dir_fd=destination_parent_fd, follow_symlinks=False)
        target_fd = _open_child_directory_at(destination_parent_fd, source.name, target_metadata)
        try:
            _copy_directory_contents(directory_fd, source, target_fd, budget, skipped)
        finally:
            os.close(target_fd)
    finally:
        os.close(directory_fd)


def snapshot_previous_paper(source_dir: Path, snapshot_dir: Path) -> list[dict[str, str]]:
    """Copy a paper tree while omitting links and special filesystem entries."""
    _reject_symlink_components(source_dir)
    _reject_symlink_components(snapshot_dir.parent)
    source_fd = open_pinned_directory(source_dir)
    skipped: list[dict[str, str]] = []
    budget = _SnapshotBudget()

    try:
        snapshot_dir.mkdir(parents=True, exist_ok=False)
    except Exception:
        os.close(source_fd)
        raise
    try:
        snapshot_fd = open_pinned_directory(snapshot_dir)
    except Exception:
        os.close(source_fd)
        raise
    try:
        for name in sorted(os.listdir(source_fd)):
            source = source_dir / name
            metadata = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            if name == "paper_data.json" and not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    "Could not preserve revision posthoc inventory: paper data must be regular"
                )
            reason = _skip_reason(metadata.st_mode)
            if reason is not None:
                skipped.append({"path": str(source), "reason": reason})
            elif stat.S_ISREG(metadata.st_mode) and name in PREVIOUS_PAPER_FILES:
                _copy_regular_file_at(
                    source_fd,
                    name,
                    source,
                    snapshot_fd,
                    metadata,
                    budget,
                )
            elif stat.S_ISDIR(metadata.st_mode) and name in PREVIOUS_PAPER_DIRECTORIES:
                _copy_directory_at(source_fd, source, snapshot_fd, metadata, budget, skipped)
    except Exception:
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        raise
    finally:
        os.close(snapshot_fd)
        os.close(source_fd)
    return skipped


def seed_revision_source_evidence(snapshot_dir: Path, revision_dir: Path) -> list[dict[str, str]]:
    """Seed inspectable sources at unchanged manuscript-relative paths."""
    _reject_symlink_components(revision_dir)
    source_fd = open_pinned_directory(snapshot_dir)
    source = snapshot_dir / SOURCE_EVIDENCE_DIRECTORY
    skipped: list[dict[str, str]] = []
    try:
        try:
            metadata = os.stat(source.name, dir_fd=source_fd, follow_symlinks=False)
        except FileNotFoundError:
            return []
        if not stat.S_ISDIR(metadata.st_mode):
            return [
                {"path": str(source), "reason": _skip_reason(metadata.st_mode) or "not_directory"}
            ]
        directory_fd = _open_child_directory_at(source_fd, source.name, metadata)
        try:
            revision_fd = open_pinned_directory(revision_dir)
            try:
                # Exclusive creation is outside cleanup: conflicts never belong to this copy.
                os.mkdir(source.name, mode=0o700, dir_fd=revision_fd)
                try:
                    target_metadata = os.stat(
                        source.name, dir_fd=revision_fd, follow_symlinks=False
                    )
                    target_fd = _open_child_directory_at(revision_fd, source.name, target_metadata)
                    try:
                        _copy_directory_contents(
                            directory_fd, source, target_fd, _SnapshotBudget(), skipped
                        )
                    finally:
                        os.close(target_fd)
                except Exception:
                    shutil.rmtree(source.name, dir_fd=revision_fd)
                    raise
            finally:
                os.close(revision_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.close(source_fd)
    return skipped


def _previous_paper_data(snapshot_dir: Path) -> dict[str, Any]:
    directory_fd = open_pinned_directory(snapshot_dir)
    try:
        try:
            os.stat("paper_data.json", dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return {}
        content = _read_regular_at(
            directory_fd,
            "paper_data.json",
            snapshot_dir / "paper_data.json",
            max_bytes=MAX_PAPER_DATA_BYTES,
        )
    finally:
        os.close(directory_fd)
    payload = json.loads(content.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("paper data must be an object")
    return payload


def _revision_posthoc_inventory(payload: dict[str, Any]) -> list[dict[str, Any]]:
    inventory = payload.get("posthoc_analyses", [])
    if not isinstance(inventory, list):
        raise ValueError("posthoc_analyses must be a list")
    seen: set[str] = set()
    for entry in inventory:
        if not isinstance(entry, dict):
            raise ValueError("posthoc entries must be objects")
        analysis_id = entry.get("analysis_id")
        if not isinstance(analysis_id, str) or not analysis_id.strip() or analysis_id in seen:
            raise ValueError("posthoc analysis IDs must be non-empty and unique")
        seen.add(analysis_id)
        if "rank" in entry and (type(entry["rank"]) is not int or entry["rank"] < 1):
            raise ValueError("posthoc rank must be a positive integer")
    return inventory


def preserve_revision_posthoc(snapshot_dir: Path, paper_data: dict[str, Any]) -> dict[str, Any]:
    """Seed original posthoc evidence before augmenting rebuilt revision data."""
    try:
        previous = _previous_paper_data(snapshot_dir)
        if "posthoc_analyses" not in previous:
            return paper_data
        original = _revision_posthoc_inventory(previous)
        existing = _revision_posthoc_inventory(paper_data)
        original_ids = {entry["analysis_id"] for entry in original}
        preserved = {
            **paper_data,
            "posthoc_analyses": original
            + [entry for entry in existing if entry["analysis_id"] not in original_ids],
        }
        for field, prefix in (("totals", "posthoc_"), ("grounding_sources", "posthoc.")):
            previous_mapping = previous.get(field, {})
            current_mapping = paper_data.get(field, {})
            if not isinstance(previous_mapping, dict) or not isinstance(current_mapping, dict):
                raise ValueError(f"{field} must be an object")
            retained = {
                key: value for key, value in previous_mapping.items() if key.startswith(prefix)
            }
            if retained:
                preserved[field] = {**current_mapping, **retained}
        return deepcopy(preserved)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not preserve revision posthoc inventory: {exc}") from exc
