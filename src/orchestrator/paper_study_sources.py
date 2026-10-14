"""Stage bounded, verified implementation sources inside fallback manuscripts."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any

from src.pipeline.downstream_selection import _CODE_EXCLUDES, load_downstream_selection
from src.pipeline.safe_fs import (
    AtomicNoReplaceUnsupported, UnsafeGluePathError, ensure_directory, open_directory,
    rename_directory_no_replace,
)
from src.pipeline.task_tree import cleanup_task_tree_preserving_error
from src.investigation.files import task_copy_entry_is_ignored
from src.investigation.pinned_files import open_relative_directory_no_follow, read_regular_at

MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_ENTRIES = 5000
_MAX_DEPTH = 64
SourceTree = dict[str, tuple[bytes | None, int]]


def _read_tree(directory_fd: int) -> SourceTree:
    tree: SourceTree = {}
    remaining = MAX_SOURCE_BYTES

    def visit(parent_fd: int, relative: Path) -> None:
        nonlocal remaining
        if len(relative.parts) > _MAX_DEPTH:
            raise ValueError("paper source depth limit exceeded")
        names = sorted(os.listdir(parent_fd))
        if len(tree) + len(names) > MAX_SOURCE_ENTRIES:
            raise ValueError("paper source entry limit exceeded")
        for name in names:
            if len(tree) >= MAX_SOURCE_ENTRIES:
                raise ValueError("paper source entry limit exceeded")
            path = relative / name
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = open_relative_directory_no_follow(parent_fd, Path(name))
                try:
                    tree[path.as_posix()] = (None, mode)
                    visit(child_fd, path)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_size > remaining:
                    raise ValueError("paper source byte limit exceeded")
                content = read_regular_at(parent_fd, name, max_bytes=remaining)
                remaining -= len(content)
                tree[path.as_posix()] = (content, mode)
            else:
                raise ValueError(f"paper source contains a non-regular entry: {path}")

    visit(directory_fd, Path())
    return tree


def _digest(tree: SourceTree) -> str:
    manifest = {
        path: ("directory", oct(mode)) if content is None else
        ("file", hashlib.sha256(content).hexdigest(), oct(mode))
        for path, (content, mode) in tree.items()
    }
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _code_only(tree: SourceTree) -> SourceTree:
    result: SourceTree = {}
    ignored: set[Path] = set()
    for relative, value in sorted(tree.items()):
        path = Path(relative)
        if any(parent in ignored for parent in path.parents):
            continue
        if task_copy_entry_is_ignored(
            name=path.name, relative_path=relative, is_directory=value[0] is None,
            patterns=_CODE_EXCLUDES,
        ):
            ignored.add(path)
        else:
            result[relative] = value
    return result


def _write_tree(directory_fd: int, tree: SourceTree) -> None:
    for relative, (content, _) in sorted(tree.items()):
        path = Path(relative)
        parent_fd = open_relative_directory_no_follow(directory_fd, path.parent)
        try:
            if content is None:
                os.mkdir(path.name, 0o700, dir_fd=parent_fd)
            else:
                descriptor = os.open(
                    path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600, dir_fd=parent_fd,
                )
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
        finally:
            os.close(parent_fd)


def _publish_sources_without_rename(parent_fd: int, tree: SourceTree) -> None:
    """Reserve once; interrupted copies remain visible and fail full-tree verification."""
    os.mkdir("downstream", 0o700, dir_fd=parent_fd)
    created = os.stat("downstream", dir_fd=parent_fd, follow_symlinks=False)
    destination_fd = open_relative_directory_no_follow(parent_fd, Path("downstream"))
    try:
        if not os.path.samestat(created, os.fstat(destination_fd)):
            raise UnsafeGluePathError("paper sources changed before copying")
        _write_tree(destination_fd, tree)
        _require_same_sources(_read_tree(destination_fd), tree)
        current = os.stat("downstream", dir_fd=parent_fd, follow_symlinks=False)
        if not os.path.samestat(created, current):
            raise UnsafeGluePathError("paper sources changed while copying")
        os.fsync(destination_fd)
        os.fsync(parent_fd)
    finally:
        os.close(destination_fd)


def _publish_sources(output_dir: Path, tree: SourceTree) -> None:
    with ensure_directory(output_dir / "source_evidence", role="paper source parent") as parent_fd:
        try:
            existing_fd = open_relative_directory_no_follow(parent_fd, Path("downstream"))
        except FileNotFoundError:
            existing_fd = None
        if existing_fd is not None:
            try:
                existing = _read_tree(existing_fd)
            finally:
                os.close(existing_fd)
            _require_same_sources(existing, tree)
            return
        staging = f".downstream-{secrets.token_hex(8)}"
        os.mkdir(staging, 0o700, dir_fd=parent_fd)
        try:
            staging_fd = open_relative_directory_no_follow(parent_fd, Path(staging))
            try:
                _write_tree(staging_fd, tree)
            finally:
                os.close(staging_fd)
            try:
                rename_directory_no_replace(parent_fd, staging, "downstream", role="paper sources")
            except AtomicNoReplaceUnsupported:
                _publish_sources_without_rename(parent_fd, tree)
        finally:
            cleanup_task_tree_preserving_error(parent_fd, staging, role="paper source staging")


def _require_same_sources(existing: SourceTree, expected: SourceTree) -> None:
    if {key: value[0] for key, value in existing.items()} != {
        key: value[0] for key, value in expected.items()
    }:
        raise ValueError("existing downstream paper sources differ from frozen selection")


def _selected_sources(selection: dict[str, Any]) -> tuple[SourceTree, dict[str, str]]:
    combined: SourceTree = {}
    paths: dict[str, str] = {}
    for role, source_key, digest_key in (
        ("program", "program_dir", "program_tree_sha256"),
        ("baseline", "baseline_program_dir", "baseline_tree_sha256"),
    ):
        if selection.get(source_key) is None:
            continue
        with open_directory(Path(selection[source_key]), role="paper source") as source_fd:
            tree = _read_tree(source_fd)
        if _digest(tree) != selection[digest_key]:
            raise ValueError(f"paper {role} source digest differs from frozen selection")
        combined[role] = (None, 0o700)
        combined.update({f"{role}/{path}": value for path, value in _code_only(tree).items()})
        paths[role] = f"source_evidence/downstream/{role}"
    if len(combined) > MAX_SOURCE_ENTRIES or sum(
        len(content) for content, _ in combined.values() if content is not None
    ) > MAX_SOURCE_BYTES:
        raise ValueError("combined paper source limit exceeded")
    return combined, paths


def _require_selection_context(context: object, selection: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(context, dict):
        raise ValueError("paper is missing downstream study context")
    for key in (
        "selection_kind", "investigation_outcome", "combined_score", "baseline_score",
        "source_attempt_id", "source_program_dir", "program_dir", "entrypoint_path",
        "program_tree_sha256", "source_tree_sha256", "baseline_program_dir",
        "baseline_tree_sha256", "attempt_counts",
    ):
        if key not in selection:
            continue
        expected = selection[key]
        if key == "baseline_score" and selection.get("baseline_program_dir") is None:
            expected = None
        if key not in context or context[key] != expected:
            raise ValueError(f"paper study context differs from frozen selection: {key}")
    if "combined_score" in selection and "baseline_score" in selection:
        baseline = selection["baseline_score"] if selection.get("baseline_program_dir") else None
        delta = selection["combined_score"] - baseline if baseline is not None else None
        if context.get("search_score_delta") != delta:
            raise ValueError("paper study score difference differs from frozen selection")
    return context


def stage_study_sources(
    paper_data: dict[str, Any], output_dir: Path, investigation_dir: Path,
) -> None:
    """Preserve verified code at stable relative paths; never overwrite prior evidence."""
    context = paper_data.get("downstream_study")
    if not isinstance(context, dict):
        return
    selection = load_downstream_selection(investigation_dir)
    if selection is None or any(
        context.get(key) != selection.get(key)
        for key in ("program_dir", "program_tree_sha256", "baseline_program_dir",
                    "baseline_tree_sha256")
    ):
        raise ValueError("paper source selection no longer matches study context")
    try:
        combined, paths = _selected_sources(selection)
        _publish_sources(output_dir, combined)
    except (OSError, UnsafeGluePathError) as exc:
        raise ValueError(f"could not safely stage paper sources: {exc}") from exc
    context["source_evidence"] = paths


def verify_study_sources(output_dir: Path, investigation_dir: Path) -> None:
    """Reject author mutations before auditing; never repair or recreate missing evidence."""
    selection = load_downstream_selection(investigation_dir)
    try:
        with open_directory(output_dir, role="paper workspace") as output_fd:
            paper_data = json.loads(read_regular_at(output_fd, "paper_data.json",
                                                   max_bytes=32 * 1024 * 1024))
        if not isinstance(paper_data, dict):
            raise ValueError("paper data must contain an object")
        context = paper_data.get("downstream_study")
        if selection is None:
            if context is not None:
                raise ValueError("paper declares a downstream study without a frozen selection")
            return
        context = _require_selection_context(context, selection)
        expected, paths = _selected_sources(selection)
        if context["source_evidence"] != paths:
            raise ValueError("paper source paths differ from frozen selection")
        with open_directory(output_dir / "source_evidence/downstream",
                            role="staged paper sources") as source_fd:
            actual = _read_tree(source_fd)
        _require_same_sources(actual, expected)
    except (OSError, UnsafeGluePathError, KeyError) as exc:
        raise ValueError(f"could not verify frozen paper sources: {exc}") from exc
