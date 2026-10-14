"""Stage 0: Extract the best-program bundle for post-hoc analysis."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shutil
import stat
import tempfile
from itertools import islice
from pathlib import Path
from typing import Any

from src.investigation.best_program_extraction import extract_best_program_bundle
from src.investigation.evaluation_evidence import EVIDENCE_FILENAME, load_evaluation_evidence

from .evaluation_evidence import variant_evaluation_evidence

logger = logging.getLogger(__name__)

_REQUIRED_BUNDLE_ENTRIES = [
    "program_0",
    "evaluate.py",
    "task.md",
    "extraction_manifest.json",
]
_MAX_EVIDENCE_FILES = 512
_MAX_EVIDENCE_FILE_BYTES = 50 * 1024 * 1024
_MAX_EVIDENCE_TOTAL_BYTES = 150 * 1024 * 1024
_MAX_EVIDENCE_TASK_DIRS = 64
_MAX_EVIDENCE_SUMMARY_BYTES = 5 * 1024 * 1024
_MAX_EVIDENCE_MANIFEST_BYTES = 1024 * 1024
_MAX_EVIDENCE_TREE_ENTRIES = 1024
_MAX_EVIDENCE_TREE_DEPTH = 4
_AGGREGATE_FILENAMES = ("run_summary.csv", "metric_values.csv", "timeseries.csv")


class BundleValidationError(RuntimeError):
    """Raised when the extracted bundle is incomplete."""


def extract_bundle(
    *,
    task_dir: Path,
    results_dir: Path,
    output_dir: Path,
    evidence_dir: Path | None = None,
) -> Path:
    """Extract the best-program bundle and validate it for post-hoc analysis."""
    bundle_dir = output_dir / "_extracted_bundle"
    logger.info(
        "Extracting post-hoc bundle: task_dir=%s results_dir=%s -> %s",
        task_dir,
        results_dir,
        bundle_dir,
    )

    extract_best_program_bundle(
        task_dir=task_dir,
        results_dir=results_dir,
        output_dir=bundle_dir,
    )
    if evidence_dir is not None:
        _stage_evidence(evidence_dir, bundle_dir)

    _validate_bundle(bundle_dir)
    logger.info("Post-hoc bundle extraction complete: %s", bundle_dir)
    return bundle_dir


def _stage_evidence(source_root: Path, bundle_dir: Path) -> None:
    """Copy a bounded, analysis-only allowlist from a completed ablations bundle."""
    if source_root.is_symlink() or not source_root.is_dir():
        raise BundleValidationError(f"Evidence source is not a real directory: {source_root}")
    _validate_evidence_anchor(source_root)
    destination_root = bundle_dir / "evidence"
    manifest_path = bundle_dir / "evidence_manifest.json"
    if destination_root.exists() or destination_root.is_symlink():
        raise BundleValidationError(f"Extracted bundle reserves evidence/: {destination_root}")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise BundleValidationError(
            f"Extracted bundle reserves evidence_manifest.json: {manifest_path}"
        )
    staging_root = Path(
        tempfile.mkdtemp(prefix=".evidence-staging-", dir=bundle_dir)
    )
    temporary_manifest = bundle_dir / f".evidence-manifest-{secrets.token_hex(8)}.tmp"
    try:
        manifest_files, total_bytes = _copy_evidence_candidates(
            source_root,
            staging_root,
        )
        _make_evidence_files_read_only(staging_root)
        manifest_bytes = (
            json.dumps(
                {
                    "manifest_version": 1,
                    "source_role": "completed_ablations_bundle",
                    "files": manifest_files,
                    "total_bytes": total_bytes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        _write_exclusive_file(temporary_manifest, manifest_bytes, mode=0o600)
        staging_root.rename(destination_root)
        try:
            temporary_manifest.rename(manifest_path)
        except BaseException:
            shutil.rmtree(destination_root, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        if temporary_manifest.exists() or temporary_manifest.is_symlink():
            temporary_manifest.unlink()


def _copy_evidence_candidates(
    source_root: Path,
    destination_root: Path,
) -> tuple[dict[str, dict[str, object]], int]:
    candidates = _evidence_candidates(source_root)
    if len(candidates) > _MAX_EVIDENCE_FILES:
        raise BundleValidationError("Evidence file count exceeds safety limit")
    total_bytes = 0
    manifest_files: dict[str, dict[str, object]] = {}
    for source in candidates:
        source_stat = source.lstat()
        if not stat.S_ISREG(source_stat.st_mode):
            raise BundleValidationError(f"Evidence entry is not a regular file: {source}")
        if source_stat.st_size > _MAX_EVIDENCE_FILE_BYTES:
            raise BundleValidationError(f"Evidence file exceeds safety limit: {source}")
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        digest, copied_bytes = _copy_evidence_file(
            source,
            destination,
            expected_stat=source_stat,
            max_bytes=min(
                _MAX_EVIDENCE_FILE_BYTES,
                _MAX_EVIDENCE_TOTAL_BYTES - total_bytes,
            ),
        )
        total_bytes += copied_bytes
        if source.name == EVIDENCE_FILENAME:
            try:
                load_evaluation_evidence(destination.parent)
            except ValueError as exc:
                raise BundleValidationError(f"Invalid evaluation evidence: {source}") from exc
        manifest_files[(Path("evidence") / relative).as_posix()] = {
            "size": copied_bytes,
            "sha256": digest,
        }
    return manifest_files, total_bytes


def _make_evidence_files_read_only(root: Path) -> None:
    for entry in root.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            entry.chmod(0o400)


def restore_verified_evidence(source_bundle: Path, target_bundle: Path) -> None:
    """Restore pristine evidence after bootstrap and verify every staged byte."""
    source_manifest = source_bundle / "evidence_manifest.json"
    source_evidence = source_bundle / "evidence"
    if not source_manifest.exists() and not source_evidence.exists():
        return
    manifest = verify_staged_evidence(source_bundle)
    target_bundle.mkdir(parents=True, exist_ok=True)
    destination_root = target_bundle / "evidence"
    destination_manifest = target_bundle / "evidence_manifest.json"
    _remove_generated_path(destination_root)
    _remove_generated_path(destination_manifest)

    staging_root = Path(
        tempfile.mkdtemp(prefix=".evidence-restore-", dir=target_bundle)
    )
    temporary_manifest = (
        target_bundle / f".evidence-manifest-{secrets.token_hex(8)}.tmp"
    )
    try:
        files = manifest["files"]
        for manifest_path, metadata in sorted(files.items()):
            relative = Path(manifest_path)
            source = source_bundle / relative
            destination = staging_root.joinpath(*relative.parts[1:])
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            source_stat = source.lstat()
            digest, copied_bytes = _copy_evidence_file(
                source,
                destination,
                expected_stat=source_stat,
                max_bytes=metadata["size"],
            )
            if copied_bytes != metadata["size"] or digest != metadata["sha256"]:
                raise BundleValidationError(
                    f"Evidence file does not match manifest: {manifest_path}"
                )
        _make_evidence_files_read_only(staging_root)
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        _write_exclusive_file(temporary_manifest, manifest_bytes, mode=0o600)
        staging_root.rename(destination_root)
        try:
            temporary_manifest.rename(destination_manifest)
        except BaseException:
            shutil.rmtree(destination_root, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        if temporary_manifest.exists() or temporary_manifest.is_symlink():
            temporary_manifest.unlink()


def verify_staged_evidence(bundle_dir: Path) -> dict[str, Any]:
    """Validate the evidence manifest, tree shape, sizes, and SHA-256 digests."""
    manifest_path = bundle_dir / "evidence_manifest.json"
    evidence_root = bundle_dir / "evidence"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise BundleValidationError(f"Evidence manifest is missing or unsafe: {manifest_path}")
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise BundleValidationError(f"Evidence directory is missing or unsafe: {evidence_root}")
    manifest_stat = manifest_path.stat()
    if manifest_stat.st_size > _MAX_EVIDENCE_MANIFEST_BYTES:
        raise BundleValidationError("Evidence manifest exceeds safety limit")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError("Evidence manifest is invalid") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    total_bytes = manifest.get("total_bytes") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_version") != 1
        or manifest.get("source_role") != "completed_ablations_bundle"
        or not isinstance(files, dict)
        or len(files) > _MAX_EVIDENCE_FILES
        or isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or total_bytes < 0
        or total_bytes > _MAX_EVIDENCE_TOTAL_BYTES
    ):
        raise BundleValidationError("Evidence manifest has an invalid schema")

    expected_paths: set[str] = set()
    expected_total = 0
    for manifest_file, metadata in files.items():
        relative = Path(manifest_file) if isinstance(manifest_file, str) else Path()
        if (
            not isinstance(manifest_file, str)
            or relative.is_absolute()
            or relative.as_posix() != manifest_file
            or not relative.parts
            or relative.parts[0] != "evidence"
            or ".." in relative.parts
            or not isinstance(metadata, dict)
        ):
            raise BundleValidationError("Evidence manifest contains an unsafe path")
        size = metadata.get("size")
        digest = metadata.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > _MAX_EVIDENCE_FILE_BYTES
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise BundleValidationError("Evidence manifest contains invalid metadata")
        expected_paths.add(manifest_file)
        expected_total += size
    if expected_total != total_bytes:
        raise BundleValidationError("Evidence manifest total_bytes is inconsistent")

    actual_paths = _bounded_evidence_files(evidence_root)
    if actual_paths != expected_paths:
        raise BundleValidationError("Evidence tree does not match manifest")
    for manifest_file, metadata in files.items():
        path = bundle_dir / manifest_file
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_size != metadata["size"]:
            raise BundleValidationError(f"Evidence file size mismatch: {manifest_file}")
        digest = _hash_regular_file(path, path_stat)
        if digest != metadata["sha256"]:
            raise BundleValidationError(f"Evidence file digest mismatch: {manifest_file}")
    return manifest


def _bounded_evidence_files(root: Path) -> set[str]:
    paths: set[str] = set()
    stack: list[tuple[Path, int]] = [(root, 0)]
    entries_seen = 0
    while stack:
        directory, depth = stack.pop()
        if depth > _MAX_EVIDENCE_TREE_DEPTH:
            raise BundleValidationError("Evidence tree exceeds depth safety limit")
        try:
            iterator = os.scandir(directory)
        except OSError as exc:
            raise BundleValidationError(f"Could not inspect evidence tree: {directory}") from exc
        with iterator:
            for entry in iterator:
                entries_seen += 1
                if entries_seen > _MAX_EVIDENCE_TREE_ENTRIES:
                    raise BundleValidationError("Evidence tree exceeds entry safety limit")
                if entry.is_symlink():
                    raise BundleValidationError(f"Evidence tree contains symlink: {entry.path}")
                entry_path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    stack.append((entry_path, depth + 1))
                elif entry.is_file(follow_symlinks=False):
                    paths.add((Path("evidence") / entry_path.relative_to(root)).as_posix())
                else:
                    raise BundleValidationError(
                        f"Evidence tree contains a special file: {entry.path}"
                    )
    return paths


def _hash_regular_file(path: Path, expected_stat: os.stat_result) -> str:
    file_descriptor = -1
    try:
        file_descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        opened_stat = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or (opened_stat.st_dev, opened_stat.st_ino)
            != (expected_stat.st_dev, expected_stat.st_ino)
        ):
            raise BundleValidationError(f"Evidence file changed while verifying: {path}")
        digest = hashlib.sha256()
        with os.fdopen(file_descriptor, "rb") as source_file:
            file_descriptor = -1
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise BundleValidationError(f"Could not verify evidence file: {path}") from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)


def _remove_generated_path(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _evidence_candidates(source_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    try:
        candidates.update(variant_evaluation_evidence(source_root))
    except (OSError, ValueError) as exc:
        raise BundleValidationError(f"Invalid evaluation evidence: {exc}") from exc
    for relative in (
        Path("README.md"),
        Path("pipeline_summary.json"),
        Path("generation/variant_bundle.json"),
        Path("generation/variant_selection.json"),
    ):
        candidate = source_root / relative
        if candidate.exists() or candidate.is_symlink():
            if candidate.is_symlink():
                raise BundleValidationError(f"Evidence source contains symlink: {candidate}")
            candidates.add(candidate)
    aggregate_root = source_root / "aggregated"
    if aggregate_root.is_symlink() or not aggregate_root.is_dir():
        raise BundleValidationError(
            f"Evidence aggregate is not a real directory: {aggregate_root}"
        )
    task_directories = list(
        islice(aggregate_root.iterdir(), _MAX_EVIDENCE_TASK_DIRS + 1)
    )
    if len(task_directories) > _MAX_EVIDENCE_TASK_DIRS:
        raise BundleValidationError("Evidence task directory count exceeds safety limit")
    for task_directory in sorted(task_directories):
        if task_directory.is_symlink():
            continue
        if not task_directory.is_dir():
            continue
        for filename in _AGGREGATE_FILENAMES:
            candidate = task_directory / filename
            if not candidate.exists() and not candidate.is_symlink():
                continue
            if candidate.is_symlink():
                raise BundleValidationError(
                    f"Evidence source contains symlink: {candidate}"
                )
            candidates.add(candidate)
    return sorted(candidates)


def _validate_evidence_anchor(source_root: Path) -> None:
    summary_path = source_root / "pipeline_summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise BundleValidationError(
            f"Evidence pipeline_summary.json is missing or unsafe: {summary_path}"
        )
    summary_stat = summary_path.stat()
    if summary_stat.st_size > _MAX_EVIDENCE_SUMMARY_BYTES:
        raise BundleValidationError("Evidence pipeline_summary.json exceeds safety limit")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError("Evidence pipeline_summary.json is invalid") from exc
    if not isinstance(summary, dict) or summary.get("schema_version") != 1:
        raise BundleValidationError("Evidence pipeline_summary.json has an unsupported schema")
    total_variants = summary.get("total_variants")
    total_completed = summary.get("total_completed")
    entries = summary.get("variants")
    if (
        isinstance(total_variants, bool)
        or not isinstance(total_variants, int)
        or total_variants < 1
        or isinstance(total_completed, bool)
        or not isinstance(total_completed, int)
        or total_completed < 1
        or total_completed > total_variants
        or not isinstance(entries, list)
        or len(entries) != total_variants
    ):
        raise BundleValidationError("Evidence pipeline_summary.json is incomplete")
    completed = 0
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("variant_id"), str)
            or entry.get("status") not in {"completed", "failed", "timeout"}
        ):
            raise BundleValidationError("Evidence pipeline_summary.json is incomplete")
        completed += int(entry["status"] == "completed")
    if completed != total_completed:
        raise BundleValidationError("Evidence pipeline_summary.json is inconsistent")


def _copy_evidence_file(
    source: Path,
    destination: Path,
    *,
    expected_stat: os.stat_result,
    max_bytes: int,
) -> tuple[str, int]:
    source_fd = -1
    destination_fd = -1
    try:
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise BundleValidationError(f"Could not safely stage evidence file: {source}") from exc
    try:
        opened_stat = os.fstat(source_fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise BundleValidationError(f"Evidence entry is not a regular file: {source}")
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            expected_stat.st_dev,
            expected_stat.st_ino,
        ):
            raise BundleValidationError(f"Evidence file changed while staging: {source}")
    except BaseException:
        os.close(source_fd)
        raise
    try:
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
        )
    except OSError as exc:
        os.close(source_fd)
        raise BundleValidationError(f"Could not safely stage evidence file: {source}") from exc
    digest = hashlib.sha256()
    copied_bytes = 0
    try:
        with os.fdopen(source_fd, "rb") as source_file, os.fdopen(
            destination_fd, "wb"
        ) as destination_file:
            while chunk := source_file.read(1024 * 1024):
                copied_bytes += len(chunk)
                if copied_bytes > max_bytes:
                    raise BundleValidationError(
                        f"Evidence file or bundle exceeds safety limit: {source}"
                    )
                digest.update(chunk)
                destination_file.write(chunk)
            destination_file.flush()
            os.fsync(destination_file.fileno())
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    return digest.hexdigest(), copied_bytes


def _write_exclusive_file(path: Path, content: bytes, *, mode: int) -> None:
    file_descriptor = -1
    try:
        file_descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode,
        )
        with os.fdopen(file_descriptor, "wb") as output_file:
            file_descriptor = -1
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())
    except OSError as exc:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        raise BundleValidationError(f"Could not safely write evidence manifest: {path}") from exc


def _validate_bundle(bundle_dir: Path) -> None:
    missing: list[str] = []
    for name in _REQUIRED_BUNDLE_ENTRIES:
        if not (bundle_dir / name).exists():
            missing.append(name)

    if missing:
        raise BundleValidationError(
            f"Extracted bundle at {bundle_dir} is missing: {missing}"
        )

    if not (bundle_dir / "reference_program").exists():
        logger.warning(
            "Extracted bundle has no reference_program/; downstream comparisons may be limited."
        )
