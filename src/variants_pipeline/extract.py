"""Stage 0: Extract the best-program bundle as the pipeline working directory."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
from pathlib import Path

from src.orchestrator.state import BEST_MANIFEST_FILE
from src.pipeline.downstream_selection import load_downstream_selection
from src.investigation.best_program_extraction import (
    ExtractionError,
    extract_best_program_bundle,
    materialize_task_inputs,
    resolve_best_program,
)
from src.investigation.program_clone import clone_program_dir

logger = logging.getLogger(__name__)

_REQUIRED_BUNDLE_ENTRIES = [
    "program_0",
    "evaluate.py",
    "extraction_manifest.json",
]


class BundleValidationError(RuntimeError):
    """Raised when the extracted bundle is incomplete."""


def extract_bundle(
    *,
    task_dir: Path,
    results_dir: Path,
    output_dir: Path,
    baseline_program_override: Path | None = None,
) -> Path:
    """Extract the best-program bundle and validate it for the variants pipeline.

    Calls :func:`extract_best_program_bundle` and then checks that the output
    contains the minimum set of artifacts required by subsequent stages.

    Parameters
    ----------
    baseline_program_override:
        If provided, replaces the ``reference_program/`` in the extracted bundle
        with a clone of this directory.  Only the program files are taken from
        the override; task-level files (``task.md``, ``evaluate.py``) still come
        from *task_dir*.

    Returns the path to the extracted bundle directory.
    """
    bundle_dir = output_dir / "_extracted_bundle"
    if bundle_dir.exists():
        _validate_reusable_bundle(
            bundle_dir,
            task_dir=task_dir,
            results_dir=results_dir,
            baseline_program_override=baseline_program_override,
        )
        logger.info("Reusing existing extracted bundle: %s", bundle_dir)
        return bundle_dir

    logger.info(
        "Extracting best-program bundle: task_dir=%s results_dir=%s -> %s",
        task_dir,
        results_dir,
        bundle_dir,
    )

    source_provenance = _capture_source_provenance(task_dir, results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=output_dir) as temp_dir:
        temporary_bundle = Path(temp_dir) / "bundle"
        extract_best_program_bundle(
            task_dir=task_dir,
            results_dir=results_dir,
            output_dir=temporary_bundle,
        )

        if baseline_program_override is not None:
            _apply_baseline_override(temporary_bundle, baseline_program_override)

        current_provenance = _capture_source_provenance(task_dir, results_dir)
        if current_provenance != source_provenance:
            raise BundleValidationError(
                "Best-program or task inputs changed during bundle extraction"
            )
        if (
            _fingerprint_program_tree(temporary_bundle / "program_0")
            != source_provenance["source_best_program_fingerprint"]
        ):
            raise BundleValidationError(
                "Extracted program does not match the selected best program"
            )
        _record_published_provenance(
            temporary_bundle,
            published_bundle_dir=bundle_dir,
            source_provenance=source_provenance,
            baseline_program_override=baseline_program_override,
        )
        _validate_bundle(temporary_bundle)
        temporary_bundle.replace(bundle_dir)

    logger.info("Bundle extraction complete: %s", bundle_dir)
    return bundle_dir


def _validate_reusable_bundle(
    bundle_dir: Path,
    *,
    task_dir: Path,
    results_dir: Path,
    baseline_program_override: Path | None,
) -> None:
    """Validate that an existing bundle belongs to this exact pipeline run."""
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise BundleValidationError(
            f"Existing extracted bundle is not a real directory: {bundle_dir}"
        )
    _validate_bundle(bundle_dir)

    manifest_path = bundle_dir / "extraction_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(
            f"Existing extraction manifest is invalid: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise BundleValidationError(
            f"Existing extraction manifest is not an object: {manifest_path}"
        )

    expected_identity = {
        "source_task_dir": str(task_dir.expanduser().resolve()),
        "source_results_dir": str(results_dir.expanduser().resolve()),
    }
    for field, expected_value in expected_identity.items():
        if manifest.get(field) != expected_value:
            raise BundleValidationError(
                f"Existing extraction manifest {field} does not match: "
                f"expected {expected_value!r}, got {manifest.get(field)!r}"
            )

    expected = {
        **_capture_source_provenance(task_dir, results_dir),
        "baseline_program_override": (
            str(baseline_program_override.expanduser().resolve())
            if baseline_program_override is not None
            else None
        ),
        "baseline_program_override_fingerprint": (
            _fingerprint_cloned_program(baseline_program_override)
            if baseline_program_override is not None
            else None
        ),
    }
    for provenance_field, provenance_value in expected.items():
        if manifest.get(provenance_field) != provenance_value:
            raise BundleValidationError(
                f"Existing extraction manifest {provenance_field} does not match: "
                f"expected {provenance_value!r}, "
                f"got {manifest.get(provenance_field)!r}"
            )

    bundle_fingerprint = manifest.get("bundle_fingerprint")
    if not isinstance(bundle_fingerprint, str):
        raise BundleValidationError(
            "Existing extraction manifest has no bundle fingerprint"
        )
    if _fingerprint_bundle_tree(bundle_dir) != bundle_fingerprint:
        raise BundleValidationError(
            "Existing extracted bundle fingerprint does not match bundled contents"
        )
    manifest_payload_fingerprint = manifest.get("manifest_payload_fingerprint")
    if not isinstance(manifest_payload_fingerprint, str):
        raise BundleValidationError(
            "Existing extraction manifest has no payload fingerprint"
        )
    if (
        _fingerprint_json_payload(
            manifest,
            excluded_keys={"manifest_payload_fingerprint"},
        )
        != manifest_payload_fingerprint
    ):
        raise BundleValidationError(
            "Existing extraction manifest payload fingerprint does not match"
        )

    if baseline_program_override is not None:
        bundled_reference = bundle_dir / "reference_program"
        if not bundled_reference.is_dir():
            raise BundleValidationError(
                f"Existing baseline override bundle is missing: {bundled_reference}"
            )
        bundled_fingerprint = _fingerprint_program_tree(bundled_reference)
        if bundled_fingerprint != expected["baseline_program_override_fingerprint"]:
            raise BundleValidationError(
                "Existing baseline override bundle fingerprint does not match "
                f"the configured source: {bundled_reference}"
            )


def _record_published_provenance(
    bundle_dir: Path,
    *,
    published_bundle_dir: Path,
    source_provenance: dict[str, str | None],
    baseline_program_override: Path | None,
) -> None:
    """Persist final bundle paths and baseline provenance before publication."""
    manifest_path = bundle_dir / "extraction_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(
            f"Extraction manifest is invalid: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise BundleValidationError(
            f"Extraction manifest is not an object: {manifest_path}"
        )
    if (
        _fingerprint_selection_payload(_selection_payload_from_manifest(manifest))
        != source_provenance["source_selection_fingerprint"]
    ):
        raise BundleValidationError(
            "Extraction manifest does not match the selected best program"
        )
    downstream_selection = manifest.get("downstream_selection")
    if (
        _fingerprint_json_payload(downstream_selection)
        if isinstance(downstream_selection, dict)
        else None
    ) != source_provenance["source_downstream_selection_fingerprint"]:
        raise BundleValidationError(
            "Extraction manifest does not match the frozen downstream selection"
        )
    published_bundle_dir = published_bundle_dir.expanduser().resolve()
    manifest["reference_program_dir"] = str(
        published_bundle_dir / "reference_program"
    )
    manifest["extracted_program_dir"] = str(published_bundle_dir / "program_0")
    manifest.update(source_provenance)
    manifest["baseline_program_override"] = (
        str(baseline_program_override.expanduser().resolve())
        if baseline_program_override is not None
        else None
    )
    manifest["baseline_program_override_fingerprint"] = (
        _fingerprint_program_tree(bundle_dir / "reference_program")
        if baseline_program_override is not None
        else None
    )
    manifest["bundle_fingerprint"] = _fingerprint_bundle_tree(bundle_dir)
    manifest["manifest_payload_fingerprint"] = _fingerprint_json_payload(
        manifest,
        excluded_keys={"manifest_payload_fingerprint"},
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _capture_source_provenance(
    task_dir: Path,
    results_dir: Path,
) -> dict[str, str | None]:
    """Fingerprint the exact task and winning program inputs to extraction."""
    try:
        selection = resolve_best_program(results_dir)
        downstream_selection = load_downstream_selection(results_dir)
    except (ExtractionError, ValueError) as exc:
        raise BundleValidationError(f"Invalid selected program: {exc}") from exc
    source_program_raw = selection.get("program_dir")
    if not isinstance(source_program_raw, str) or not source_program_raw:
        raise BundleValidationError("Selected best program has no program directory")
    source_program_dir = Path(source_program_raw).expanduser().resolve()
    reference_source = (downstream_selection or {}).get("baseline_program_dir")
    reference_program_dir = Path(reference_source) if reference_source is not None else None
    with tempfile.TemporaryDirectory(prefix="agentic-scientist-task-fingerprint-") as temp_dir:
        staged_task = Path(temp_dir) / "task"
        materialize_task_inputs(
            task_dir=task_dir, output_dir=staged_task, reference_program_dir=reference_program_dir,
        )
        source_task_fingerprint = _fingerprint_program_tree(staged_task)
    return {
        "source_downstream_selection_fingerprint": (
            _fingerprint_json_payload(downstream_selection)
            if downstream_selection is not None
            else None
        ),
        "source_best_manifest_fingerprint": _fingerprint_optional_regular_file(
            results_dir / BEST_MANIFEST_FILE,
            role="best-program manifest",
        ),
        "source_best_program_dir": str(source_program_dir),
        "source_best_program_fingerprint": _fingerprint_cloned_program(
            source_program_dir
        ),
        "source_selection_fingerprint": _fingerprint_selection_payload(
            _selection_payload(selection)
        ),
        "source_task_fingerprint": source_task_fingerprint,
    }


def _selection_payload(selection: dict[str, object]) -> dict[str, object]:
    """Normalize the selector fields persisted by extraction manifests."""
    program_dir = selection.get("program_dir")
    if not isinstance(program_dir, str) or not program_dir:
        raise BundleValidationError("Selected best program has no program directory")
    return {
        "program_dir": str(Path(program_dir).expanduser().resolve()),
        "entrypoint_path": selection.get("entrypoint_path"),
        "combined_score": selection.get("combined_score"),
        "correct": selection.get("correct"),
        "source_attempt_id": selection.get("source_attempt_id"),
        "parallel_worker_index": selection.get("parallel_worker_index"),
    }


def _selection_payload_from_manifest(manifest: dict[str, object]) -> dict[str, object]:
    """Read selector fields back from an extraction manifest."""
    return {
        "program_dir": manifest.get("source_best_program_dir"),
        "entrypoint_path": manifest.get("entrypoint_path"),
        "combined_score": manifest.get("combined_score"),
        "correct": manifest.get("correct"),
        "source_attempt_id": manifest.get("source_attempt_id"),
        "parallel_worker_index": manifest.get("parallel_worker_index"),
    }


def _fingerprint_selection_payload(payload: dict[str, object]) -> str:
    """Fingerprint the scientific metadata identifying a selected program."""
    return _fingerprint_json_payload(payload)


def _fingerprint_json_payload(
    payload: dict[str, object],
    *,
    excluded_keys: set[str] | None = None,
) -> str:
    """Return a deterministic SHA-256 for a JSON object."""
    excluded = excluded_keys or set()
    normalized = {key: value for key, value in payload.items() if key not in excluded}
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint_optional_regular_file(path: Path, *, role: str) -> str | None:
    """Hash a regular file without following a symlink at the final component."""
    try:
        file_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise BundleValidationError(f"Cannot safely read {role}: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise BundleValidationError(f"{role} must be a regular file: {path}")
    except Exception:
        os.close(file_fd)
        raise
    with os.fdopen(file_fd, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _fingerprint_cloned_program(source_dir: Path) -> str:
    """Fingerprint exactly the files that the program clone policy retains."""
    with tempfile.TemporaryDirectory(prefix="agentic-scientist-fingerprint-") as temp_dir:
        clone_dir = Path(temp_dir) / "program"
        try:
            clone_program_dir(src_dir=source_dir, dst_dir=clone_dir)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise BundleValidationError(
                f"Cannot fingerprint baseline program override: {source_dir}"
            ) from exc
        return _fingerprint_program_tree(clone_dir)


def _fingerprint_program_tree(program_dir: Path) -> str:
    """Return a deterministic SHA-256 for a cloned program tree."""
    return _fingerprint_tree(program_dir)


def _fingerprint_bundle_tree(bundle_dir: Path) -> str:
    """Fingerprint every published artifact except the provenance manifest."""
    return _fingerprint_tree(
        bundle_dir,
        excluded_paths={"extraction_manifest.json"},
    )


def _fingerprint_tree(
    root_dir: Path,
    *,
    excluded_paths: set[str] | None = None,
) -> str:
    """Return a deterministic SHA-256 for a directory tree."""
    digest = hashlib.sha256()
    excluded = excluded_paths or set()

    def update(kind: bytes, relative_path: str, payload: bytes = b"") -> None:
        for value in (kind, relative_path.encode("utf-8"), payload):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)

    for root, dirnames, filenames in os.walk(
        root_dir, topdown=True, followlinks=False
    ):
        root_path = Path(root)
        relative_root = root_path.relative_to(root_dir).as_posix()
        update(b"D", relative_root)

        for dirname in sorted(dirnames):
            path = root_path / dirname
            relative_path = path.relative_to(root_dir).as_posix()
            if relative_path in excluded:
                dirnames.remove(dirname)
                continue
            if path.is_symlink():
                update(b"L", relative_path, os.readlink(path).encode("utf-8"))
                dirnames.remove(dirname)
        dirnames.sort()

        for filename in sorted(filenames):
            path = root_path / filename
            relative_path = path.relative_to(root_dir).as_posix()
            if relative_path in excluded:
                continue
            if path.is_symlink():
                update(b"L", relative_path, os.readlink(path).encode("utf-8"))
                continue
            update(b"F", relative_path, path.read_bytes())

    return digest.hexdigest()


def _apply_baseline_override(bundle_dir: Path, override_dir: Path) -> None:
    """Replace ``reference_program/`` in the bundle with a clone of *override_dir*.

    Only experiment files (``*.py``, ``*.toml``, etc.) are replaced — task-level
    files in the bundle root are untouched.
    """
    override_dir = override_dir.resolve()
    if not override_dir.is_dir():
        raise BundleValidationError(
            f"Baseline program override is not a directory: {override_dir}"
        )

    ref_dir = bundle_dir / "reference_program"
    if ref_dir.exists():
        shutil.rmtree(ref_dir)

    clone_program_dir(src_dir=override_dir, dst_dir=ref_dir)
    logger.info(
        "Replaced reference_program with baseline override from %s",
        override_dir,
    )


def _validate_bundle(bundle_dir: Path) -> None:
    """Raise :class:`BundleValidationError` if the bundle is incomplete."""
    missing: list[str] = []
    for name in _REQUIRED_BUNDLE_ENTRIES:
        if not (bundle_dir / name).exists():
            missing.append(name)

    if missing:
        raise BundleValidationError(
            f"Extracted bundle at {bundle_dir} is missing: {missing}"
        )

    # reference_program is optional but log a warning if absent.
    if not (bundle_dir / "reference_program").exists():
        logger.warning(
            "Extracted bundle has no reference_program/ — "
            "baseline_reference variant will be skipped."
        )
