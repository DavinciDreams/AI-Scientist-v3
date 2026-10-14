"""Standalone best-program extraction pipeline.

Extracts the winning program from a completed scientist run and materializes
a new task bundle for consumption by downstream pipelines.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.orchestrator.state import BEST_MANIFEST_FILE
from src.pipeline.downstream_selection import load_downstream_selection
from src.investigation.constants import PROGRAM_DIR_RE, TASK_COPY_EXCLUDE_DIRS
from src.investigation.files import (
    find_best_program_in_experiment,
    load_task_copy_ignore_patterns,
    task_copy_path_is_ignored,
)
from src.investigation.program_clone import clone_program_dir
from src.investigation.program_spec import (
    ProgramSpecError,
    load_program_spec,
    resolve_entrypoint,
)

logger = logging.getLogger(__name__)

_TASK_COPY_EXCLUDE_FILES: set[str] = {
    ".agentic_runtime.json",
}


_WORKER_DIR_RE = re.compile(r"^worker_\d+$")


class ExtractionError(RuntimeError):
    """Raised when extraction validation or materialization fails."""


def _is_program_dir_name(name: str) -> bool:
    return PROGRAM_DIR_RE.match(name) is not None


def _has_worker_ancestor(path: Path, root: Path) -> bool:
    """Return True if *path* has a ``worker_XX`` ancestor between it and *root*."""
    current = path.parent
    root_resolved = root.resolve()
    while current != root_resolved and current != current.parent:
        if _WORKER_DIR_RE.match(current.name):
            return True
        current = current.parent
    return False


def _copy_task_files(*, src_task_dir: Path, dst_dir: Path) -> None:
    """Copy task-level files from *src_task_dir* into *dst_dir*.

    Excludes program variant directories (``program_<N>``) and common
    transient/cache directories.  Symlinks are preserved as symlinks.
    """
    ignore_patterns = load_task_copy_ignore_patterns(src_task_dir)
    for root, dirnames, filenames in os.walk(src_task_dir, topdown=True, followlinks=False):
        root_path = Path(root)
        rel_root = root_path.relative_to(src_task_dir).as_posix()
        if rel_root == ".":
            rel_root = ""

        # Handle symlinked directories: copy as symlinks, do not traverse.
        for dirname in sorted(dirnames):
            src_path = root_path / dirname
            if not src_path.is_symlink():
                continue
            if _is_program_dir_name(dirname) or dirname in TASK_COPY_EXCLUDE_DIRS:
                dirnames.remove(dirname)
                continue
            relpath = f"{rel_root}/{dirname}".lstrip("/")
            if task_copy_path_is_ignored(
                relpath, is_dir=True, patterns=ignore_patterns
            ):
                dirnames.remove(dirname)
                continue
            dst_path = dst_dir / relpath
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            link_target = os.readlink(src_path)
            dst_path.symlink_to(link_target, target_is_directory=True)
            dirnames.remove(dirname)

        # Prune excluded and program directories from traversal.
        for dirname in list(dirnames):
            relpath = f"{rel_root}/{dirname}".lstrip("/")
            if (
                _is_program_dir_name(dirname)
                or dirname in TASK_COPY_EXCLUDE_DIRS
                or task_copy_path_is_ignored(
                    relpath, is_dir=True, patterns=ignore_patterns
                )
            ):
                dirnames.remove(dirname)

        dirnames.sort()
        filenames.sort()

        for filename in filenames:
            if not rel_root and filename in _TASK_COPY_EXCLUDE_FILES:
                continue
            src_path = root_path / filename
            relpath = f"{rel_root}/{filename}".lstrip("/")
            if task_copy_path_is_ignored(
                relpath, is_dir=False, patterns=ignore_patterns
            ):
                continue
            dst_path = dst_dir / relpath
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_symlink():
                link_target = os.readlink(src_path)
                dst_path.symlink_to(link_target, target_is_directory=src_path.is_dir())
            elif src_path.is_file():
                shutil.copy2(src_path, dst_path)


def _load_and_validate_manifest(results_dir: Path) -> dict[str, Any] | None:
    """Read and validate ``best_program_manifest.json`` from *results_dir*.

    Returns ``None`` if the manifest file does not exist (caller should
    fall back to scanning).  Raises :class:`ExtractionError` if the file
    exists but is malformed.
    """
    manifest_path = results_dir / BEST_MANIFEST_FILE
    if not manifest_path.is_file():
        return None

    try:
        raw = manifest_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        raise ExtractionError(
            f"Failed to read best-program manifest {manifest_path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ExtractionError(
            f"Best-program manifest is not a JSON object: {manifest_path}"
        )

    program_dir = data.get("program_dir")
    if not isinstance(program_dir, str) or not program_dir:
        raise ExtractionError(
            f"Best-program manifest missing or empty 'program_dir': {manifest_path}"
        )

    entrypoint_path = data.get("entrypoint_path")
    if not isinstance(entrypoint_path, str) or not entrypoint_path:
        raise ExtractionError(
            f"Best-program manifest missing or empty 'entrypoint_path': {manifest_path}"
        )

    return data


def _scan_for_best_program(results_dir: Path) -> dict[str, Any]:
    """Walk *results_dir* for the best correct program by combined_score.

    Only considers ``program_<N>`` directories that sit under a
    ``worker_XX`` ancestor directory.  Each candidate must have
    ``correct.json`` (with ``"correct": true``) and ``metrics.json``
    (with a numeric ``combined_score``).

    Raises :class:`ExtractionError` if no valid program is found.
    """
    best_score: float | None = None
    best_program_dir: Path | None = None
    best_entrypoint: str | None = None

    resolved_root = results_dir.resolve()
    seen_experiment_dirs: set[Path] = set()

    for dirpath, dirnames, _ in os.walk(results_dir, topdown=True, followlinks=False):
        current = Path(dirpath)

        # Only consider directories that contain program_<N> children.
        has_program_child = False
        for d in dirnames:
            if PROGRAM_DIR_RE.match(d):
                has_program_child = True
                break
        if not has_program_child:
            continue

        # Require a worker_XX ancestor between this dir and results_dir.
        if not _has_worker_ancestor(current, resolved_root):
            continue

        resolved = current.resolve()
        if resolved in seen_experiment_dirs:
            continue
        seen_experiment_dirs.add(resolved)

        result = find_best_program_in_experiment(current)
        if result is None:
            continue
        program_dir, entrypoint_rel, score, _ = result
        if best_score is None or score > best_score:
            best_score = score
            best_program_dir = program_dir
            best_entrypoint = entrypoint_rel

    if best_program_dir is None or best_entrypoint is None or best_score is None:
        raise ExtractionError(
            f"No valid program found by scanning results directory: {results_dir}. "
            "Looked for program_<N> directories under worker_XX with "
            "correct.json (correct: true) and metrics.json (combined_score)."
        )

    logger.info(
        "Scan found best program: %s (score=%.6f, entrypoint=%s)",
        best_program_dir,
        best_score,
        best_entrypoint,
    )

    return {
        "program_dir": str(best_program_dir),
        "entrypoint_path": best_entrypoint,
        "combined_score": best_score,
        "correct": True,
        "source_attempt_id": None,
    }


def resolve_best_program(results_dir: Path) -> dict[str, Any]:
    """Resolve the downstream selection, winner, or legacy scan fallback."""
    results_dir = results_dir.expanduser().resolve()
    downstream_selection = _verified_downstream_selection(results_dir)
    if downstream_selection is not None:
        return downstream_selection
    manifest = _load_and_validate_manifest(results_dir)
    if manifest is not None:
        return manifest

    logger.warning(
        "best_program_manifest.json not found in %s — "
        "falling back to scanning results for the best correct program.",
        results_dir,
    )
    warnings.warn(
        f"best_program_manifest.json not found in {results_dir}. "
        "Falling back to scanning results for the best correct program "
        "(by combined_score).",
        stacklevel=2,
    )
    return _scan_for_best_program(results_dir)


def _verified_downstream_selection(results_dir: Path) -> dict[str, Any] | None:
    try:
        return load_downstream_selection(results_dir)
    except ValueError as exc:
        raise ExtractionError(f"Invalid downstream selection: {exc}") from exc


def materialize_task_inputs(
    *, task_dir: Path, output_dir: Path, reference_program_dir: Path | None = None,
) -> None:
    """Copy the task-level files and seed program used by extracted bundles."""
    _copy_task_files(src_task_dir=task_dir, dst_dir=output_dir)
    clone_program_dir(
        src_dir=reference_program_dir if reference_program_dir is not None else task_dir / "program_0",
        dst_dir=output_dir / "reference_program",
    )


def extract_best_program_bundle(
    *,
    task_dir: Path,
    results_dir: Path,
    output_dir: Path,
) -> Path:
    """Create an extracted task bundle at *output_dir* and return *output_dir*.

    The bundle contains all task-level files, the winning program as
    ``program_0``, the original seed as ``reference_program``, and an
    ``extraction_manifest.json`` recording provenance.
    """
    # --- resolve paths ---
    task_dir = task_dir.expanduser().resolve()
    results_dir = results_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    # --- validate inputs ---
    if not task_dir.is_dir():
        raise ExtractionError(f"Task directory does not exist: {task_dir}")
    if not (task_dir / "program_0").is_dir():
        raise ExtractionError(f"Task directory missing program_0: {task_dir}")
    if not results_dir.is_dir():
        raise ExtractionError(f"Results directory does not exist: {results_dir}")

    downstream_selection = _verified_downstream_selection(results_dir)
    manifest = downstream_selection or resolve_best_program(results_dir)

    best_program_dir = Path(manifest["program_dir"]).expanduser().resolve()
    if downstream_selection is None and not best_program_dir.is_relative_to(results_dir):
        raise ExtractionError(
            "Best-program manifest points outside results directory: "
            f"{best_program_dir}"
        )
    if not best_program_dir.is_dir():
        raise ExtractionError(
            f"Best program directory does not exist: {best_program_dir}"
        )

    try:
        spec = load_program_spec(best_program_dir, required=True)
    except ProgramSpecError as exc:
        raise ExtractionError(
            f"Invalid program_spec.toml in best program {best_program_dir}: {exc}"
        ) from exc
    if spec is None:
        raise ExtractionError(
            f"Missing program_spec.toml in best program: {best_program_dir}"
        )

    try:
        entrypoint_abs = resolve_entrypoint(best_program_dir, spec)
    except ProgramSpecError as exc:
        raise ExtractionError(
            f"Cannot resolve entrypoint in best program {best_program_dir}: {exc}"
        ) from exc
    if not entrypoint_abs.is_file():
        raise ExtractionError(
            f"Entrypoint file does not exist: {entrypoint_abs}"
        )

    # --- validate output directory ---
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ExtractionError(f"Output path exists but is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ExtractionError(f"Output directory is not empty: {output_dir}")

    # --- materialize bundle ---
    logger.info("Extracting best-program bundle to %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    reference_source = (downstream_selection or {}).get("baseline_program_dir")
    reference_program_dir = Path(reference_source) if reference_source is not None else None
    materialize_task_inputs(
        task_dir=task_dir, output_dir=output_dir, reference_program_dir=reference_program_dir,
    )

    clone_program_dir(
        src_dir=best_program_dir,
        dst_dir=output_dir / "program_0",
    )

    extraction_manifest = {
        "extraction_version": 1,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "source_task_dir": str(task_dir),
        "source_results_dir": str(results_dir),
        "source_best_manifest_path": str(results_dir / BEST_MANIFEST_FILE)
        if (results_dir / BEST_MANIFEST_FILE).is_file()
        else None,
        "source_best_program_dir": str(best_program_dir),
        "reference_program_dir": str(output_dir / "reference_program"),
        "extracted_program_dir": str(output_dir / "program_0"),
        "entrypoint_path": manifest["entrypoint_path"],
        "combined_score": manifest.get("combined_score"),
        "correct": manifest.get("correct"),
        "source_attempt_id": manifest.get("source_attempt_id"),
        "parallel_worker_index": manifest.get("parallel_worker_index"),
        "downstream_selection": downstream_selection,
    }

    manifest_path = output_dir / "extraction_manifest.json"
    manifest_path.write_text(
        json.dumps(extraction_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Extraction manifest written to %s", manifest_path)

    return output_dir
