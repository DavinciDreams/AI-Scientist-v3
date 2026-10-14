"""Freeze a valid study program after authenticated search exhaustion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from src.investigation.files import evaluation_results_complete
from src.investigation.pinned_files import absolute_path, read_regular_at
from src.investigation.program_clone import _NONOVERRIDABLE_EXCLUDES
from src.investigation.program_spec import entrypoint_relpath_from_spec

from .downstream_publication import publish_frozen_program_at, publish_selection_manifest_at
from .safe_fs import (
    UnsafeGluePathError,
    ensure_directory,
    open_directory,
)
from .task_tree import copy_task_tree_at, task_tree_manifest_at, verify_copied_task_tree

_MANIFEST = ".downstream_selection.json"
_HISTORY = "attempts_history.jsonl"
_MAX_BYTES = 64 * 1024 * 1024
_PROGRAM_PATTERN = re.compile(r"program_([1-9][0-9]*)")
_CODE_EXCLUDES = tuple(f"/{name}" for name in sorted(_NONOVERRIDABLE_EXCLUDES)) + (
    "/.evaluation_stage_*", "__pycache__", "*.pyc", ".venv", "node_modules", ".git",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox", "dist", "build",
)


def _read(path: Path) -> bytes:
    with open_directory(path.parent, role="downstream artifact parent") as parent_fd:
        return read_regular_at(parent_fd, path.name, max_bytes=_MAX_BYTES)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(_read(path))
    if not isinstance(value, dict):
        raise ValueError(f"downstream artifact must contain an object: {path}")
    return value


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _tree_digest(path: Path, *, excludes: tuple[str, ...] = ()) -> str:
    with open_directory(path, role="downstream program") as directory_fd:
        entries = task_tree_manifest_at(directory_fd, relative_dir=Path(), ignore_patterns=excludes)
    if any(value[0] == "symlink" for value in entries.values()):
        raise ValueError(f"downstream program contains a symlink: {path}")
    return _digest(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode())


def _score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _contained(root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("missing downstream source path")
    path = Path(raw)
    if ".." in path.parts:
        raise ValueError("downstream source path escapes workspace")
    path = absolute_path(path if path.is_absolute() else root / path)
    if not path.is_relative_to(root) or path == root:
        raise ValueError("downstream source path escapes workspace")
    return path


def _history(root: Path) -> list[dict[str, Any]]:
    try:
        content = _read(root / _HISTORY)
    except FileNotFoundError:
        return []
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("malformed downstream attempts history")
    return rows


def _binding(root: Path) -> dict[str, str | None]:
    from .config import investigation_completion_receipt_path

    result: dict[str, str | None] = {}
    for label, path in {
        "investigation_manifest_sha256": root / "best_program_manifest.json",
        "completion_receipt_sha256": investigation_completion_receipt_path(root),
        "attempts_history_sha256": root / _HISTORY,
    }.items():
        try:
            result[label] = _digest(_read(path))
        except FileNotFoundError:
            if label != "attempts_history_sha256":
                raise
            result[label] = None
    return result


def _candidate(root: Path, source: Path, attempt_id: object) -> dict[str, Any] | None:
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        return None
    try:
        source = _contained(root, str(source))
        source_digest = _tree_digest(source)
        evaluation = _json(source / "evaluation_manifest.json")
        if "published_summaries" in evaluation:
            summaries = evaluation["published_summaries"]
            if not isinstance(summaries, list) or any(
                not isinstance(name, str) for name in summaries
            ):
                return None
        if not evaluation_results_complete(source):
            return None
        if any(isinstance(evaluation.get(key), bool) for key in ("exit_code", "returncode")):
            return None
        if evaluation.get("program_id") != source.name:
            return None
        if "attempt_id" in evaluation and evaluation["attempt_id"] != attempt_id:
            return None
        metrics = _json(source / "metrics.json")
        correct = _json(source / "correct.json")
        score = _score(metrics.get("combined_score"))
        entrypoint = entrypoint_relpath_from_spec(source)
        if correct.get("correct") is not True or score is None or entrypoint is None:
            return None
        return {
            "selection_kind": "non_winner_candidate",
            "source_program_dir": str(source), "source_attempt_id": attempt_id,
            "combined_score": score, "entrypoint_path": entrypoint,
            "source_tree_sha256": source_digest,
        }
    except (OSError, ValueError, UnsafeGluePathError):
        return None


def _baseline(root: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    score = _score(manifest.get("combined_score"))
    if manifest.get("correct") is not True or score is None:
        return None
    try:
        source = _contained(root, manifest.get("program_dir"))
        digest = _tree_digest(source)
        entrypoint = entrypoint_relpath_from_spec(source)
        if entrypoint is None:
            return None
        return {
            "selection_kind": "baseline", "source_program_dir": str(source),
            "source_attempt_id": manifest.get("source_attempt_id"), "combined_score": score,
            "entrypoint_path": entrypoint, "source_tree_sha256": digest,
        }
    except (OSError, ValueError, UnsafeGluePathError):
        return None


def _new_selection(root: Path) -> dict[str, Any] | None:
    from .config import completed_investigation_outcome

    if completed_investigation_outcome(root) != "completed_no_winner":
        return None
    binding = _binding(root)
    manifest = _json(root / "best_program_manifest.json")
    baseline = _baseline(root, manifest)
    rows = _history(root)
    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for order, row in enumerate(rows):
        try:
            experiment = _contained(root, row.get("experiment_dir"))
            with open_directory(experiment, role="downstream experiment") as experiment_fd:
                names = os.listdir(experiment_fd)
        except (OSError, ValueError, UnsafeGluePathError):
            continue
        for name in names:
            match = _PROGRAM_PATTERN.fullmatch(name)
            if match is None:
                continue
            item = _candidate(root, experiment / name, row.get("attempt_id"))
            if item is not None:
                candidates.append((-item["combined_score"], order, int(match[1]), item))
    chosen = min(candidates, key=lambda item: item[:3])[3] if candidates else baseline
    if chosen is None:
        return None
    scored = sum(
        isinstance(row.get("metrics"), dict)
        and _score(row["metrics"].get("combined_score")) is not None
        for row in rows
    )
    source = Path(chosen["source_program_dir"])
    return {
        "schema_version": 1, "investigation_outcome": "completed_no_winner", **chosen,
        "correct": True,
        "program_dir": str(root.parent / "downstream/program"),
        "program_tree_sha256": _tree_digest(source, excludes=_CODE_EXCLUDES),
        "baseline_score": _score(manifest.get("combined_score")),
        "baseline_program_dir": baseline["source_program_dir"] if baseline else None,
        "baseline_tree_sha256": baseline["source_tree_sha256"] if baseline else None,
        "attempt_counts": {"total": len(rows), "scored": scored, "unscored": len(rows) - scored},
        **binding,
    }


def _validate(root: Path, selection: dict[str, Any]) -> None:
    from .config import completed_investigation_outcome

    if selection.get("schema_version") != 1 or selection.get("investigation_outcome") != (
        "completed_no_winner"
    ) or selection.get("selection_kind") not in {"baseline", "non_winner_candidate"}:
        raise ValueError("malformed downstream selection manifest")
    if completed_investigation_outcome(root) != "completed_no_winner":
        raise ValueError("stale downstream selection: investigation receipt changed")
    for key, digest in _binding(root).items():
        if selection.get(key) != digest:
            raise ValueError(f"stale downstream selection: {key} mismatch")
    source = _contained(root, selection.get("source_program_dir"))
    frozen = root.parent / "downstream/program"
    if selection.get("program_dir") != str(frozen):
        raise ValueError("malformed downstream snapshot path")
    if _tree_digest(source) != selection.get("source_tree_sha256"):
        raise ValueError("stale downstream selection: source evidence changed")
    if _tree_digest(frozen) != selection.get("program_tree_sha256"):
        raise ValueError("stale downstream selection: snapshot changed")
    if _tree_digest(source, excludes=_CODE_EXCLUDES) != selection.get("program_tree_sha256"):
        raise ValueError("stale downstream selection: source program changed")
    if entrypoint_relpath_from_spec(frozen) != selection.get("entrypoint_path"):
        raise ValueError("stale downstream selection: entrypoint mismatch")
    if selection.get("baseline_program_dir") is not None:
        baseline = _contained(root, selection["baseline_program_dir"])
        if _tree_digest(baseline) != selection.get("baseline_tree_sha256"):
            raise ValueError("stale downstream selection: baseline changed")
    expected = _new_selection(root)
    if expected != selection:
        raise ValueError("stale downstream selection: recorded selection mismatch")


def load_downstream_selection(investigation_dir: Path) -> dict[str, Any] | None:
    """Read and verify a frozen selection; missing is distinct from invalid."""
    root = absolute_path(investigation_dir)
    try:
        selection = _json(root.parent / _MANIFEST)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnsafeGluePathError) as exc:
        raise ValueError(f"invalid downstream selection manifest: {exc}") from exc
    try:
        _validate(root, selection)
    except (OSError, ValueError, UnsafeGluePathError) as exc:
        raise ValueError(f"invalid or stale downstream selection: {exc}") from exc
    return selection


def _reject_existing_final(root: Path) -> None:
    path = root.parent / "final_evaluation"
    try:
        with open_directory(path, role="final evaluation") as directory_fd:
            if os.listdir(directory_fd):
                raise ValueError("cannot select a downstream program after final results exist")
    except FileNotFoundError:
        pass


def preview_downstream_selection(investigation_dir: Path) -> dict[str, Any] | None:
    """Resolve the proposed source without writing a snapshot or manifest."""
    root = absolute_path(investigation_dir)
    existing = load_downstream_selection(root)
    if existing is not None:
        return existing
    try:
        selection = _new_selection(root)
        if selection is not None:
            _reject_existing_final(root)
        return selection
    except (OSError, UnsafeGluePathError) as exc:
        raise ValueError(f"invalid downstream selection source: {exc}") from exc


def prepare_downstream_selection(investigation_dir: Path) -> dict[str, Any] | None:
    """Publish a frozen program once; never replace a previously studied selection."""
    root = absolute_path(investigation_dir)
    existing = load_downstream_selection(root)
    if existing is not None:
        return existing
    selection = preview_downstream_selection(root)
    if selection is None:
        return None
    source = Path(selection["source_program_dir"])
    destination = root.parent / "downstream"
    try:
        with ensure_directory(destination, role="downstream snapshot parent") as destination_fd:
            with tempfile.TemporaryDirectory(prefix=".selection-", dir=destination) as temporary:
                staging = Path(temporary)
                with open_directory(source, role="downstream program") as source_fd:
                    with open_directory(staging, role="downstream staging") as staging_fd:
                        copy_task_tree_at(source_fd, staging_fd, relative_dir=Path(),
                                          ignore_patterns=_CODE_EXCLUDES)
                        verify_copied_task_tree(source_fd, staging_fd,
                                                ignore_patterns=_CODE_EXCLUDES)
                if _tree_digest(staging) != selection["program_tree_sha256"]:
                    raise ValueError("downstream program changed while freezing")
                binding = _binding(root)
                if binding != {key: selection[key] for key in binding}:
                    raise ValueError("search evidence changed while freezing downstream program")
                if _tree_digest(source) != selection["source_tree_sha256"]:
                    raise ValueError("source evidence changed while freezing downstream program")
                publish_frozen_program_at(destination_fd, staging.name)
        _validate(root, selection)
        with open_directory(root.parent, role="pipeline run envelope") as run_fd:
            publish_selection_manifest_at(run_fd, _MANIFEST, json.dumps(selection, sort_keys=True,
                                          indent=2).encode() + b"\n")
        return load_downstream_selection(root)
    except (OSError, UnsafeGluePathError) as exc:
        raise ValueError(f"could not freeze downstream program: {exc}") from exc


def resolve_downstream_program(investigation_dir: Path) -> tuple[Path, Path] | None:
    """Resolve only an already frozen, verified downstream selection."""
    selection = load_downstream_selection(investigation_dir)
    if selection is None:
        return None
    root = Path(selection["program_dir"])
    return root, root / selection["entrypoint_path"]
