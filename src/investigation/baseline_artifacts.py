"""Private archival of bounded, validated production baseline outputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile

from .evaluation_evidence import EVIDENCE_FILENAME, MAX_EVIDENCE_BYTES
from .evaluator_outputs import (
    MAX_RESULT_JSON_BYTES, _load_mapping, _unique_object, validate_evaluator_outputs,
)


def _read_regular(path: Path, maximum: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError("Baseline artifact is unsafe or exceeds its byte limit")
        raw = handle.read(maximum + 1)
        after = os.fstat(handle.fileno())
        if len(raw) > maximum or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Baseline artifact changed while archiving")
        return raw


def _check_directory(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink() or not component.is_dir():
            raise ValueError(f"Unsafe baseline archive directory: {component}")
    if path.stat().st_uid != os.getuid():
        raise ValueError("Baseline archive directory must be owned by the current user")


def _entry_identity(path: Path) -> tuple[int, ...]:
    metadata = path.lstat()
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def _validated_archive(path: Path) -> tuple[tuple[int, int], str]:
    """Validate bounded contents and hash the same manifest bytes, without rereading."""
    _check_directory(path)
    identity = path.stat()
    entries: dict[str, tuple[int, ...]] = {}

    def read(name: str, maximum: int) -> bytes:
        entries[name] = _entry_identity(path / name)
        return _read_regular(path / name, maximum)

    try:
        manifest_bytes = read("archive_manifest.json", MAX_RESULT_JSON_BYTES)
    except OSError as exc:
        raise ValueError("Existing destination is not a recognized baseline archive") from exc
    try:
        manifest = json.loads(manifest_bytes, object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Baseline archive manifest must be bounded valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Baseline archive manifest must be a JSON object")
    files = manifest.get("files")
    limits = {
        "metrics.json": MAX_RESULT_JSON_BYTES,
        "correct.json": MAX_RESULT_JSON_BYTES,
        "qualification.json": MAX_RESULT_JSON_BYTES,
        EVIDENCE_FILENAME: MAX_EVIDENCE_BYTES,
    }
    if (
        set(manifest) != {"schema_version", "files"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not isinstance(files, dict)
        or not {"metrics.json", "correct.json"} <= files.keys() <= limits.keys()
        or {entry.name for entry in path.iterdir()} != {*files, "archive_manifest.json"}
    ):
        raise ValueError("Existing destination is not a recognized baseline archive")
    for name, expected_digest in files.items():
        payload = read(name, limits[name])
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            raise ValueError("Existing baseline archive changed after publication")
    _check_directory(path)
    current = path.stat()
    if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
        raise ValueError("Existing baseline archive directory changed during validation")
    if {entry.name for entry in path.iterdir()} != {*files, "archive_manifest.json"}:
        raise ValueError("Baseline archive inventory changed during validation")
    if any(_entry_identity(path / name) != before for name, before in entries.items()):
        raise ValueError("Baseline archive file changed during validation")
    return (identity.st_dev, identity.st_ino), hashlib.sha256(manifest_bytes).hexdigest()


def baseline_archive_digest(path: Path) -> str:
    """Bind the exact manifest after verifying every declared file against its digest."""
    return _validated_archive(path)[1]


def _existing_archive_identity(path: Path) -> tuple[int, int]:
    """Recognize only intact archives produced here before moving a prior result."""
    return _validated_archive(path)[0]


def archive_baseline_artifacts(
    results_dir: Path,
    destination: Path,
    *,
    metric_names: tuple[str, ...],
    combined_score: float,
    correct: bool,
    feasible: bool | None,
) -> None:
    """Publish only a revalidated snapshot; retain a previous archive on successful reruns."""
    _check_directory(results_dir)
    destination = destination.absolute()
    _check_directory(destination.parent)
    if destination.is_symlink():
        raise ValueError("Unsafe baseline archive destination")
    if destination.is_relative_to(results_dir.absolute()) or results_dir.absolute().is_relative_to(
        destination
    ):
        raise ValueError("Baseline archive must not overlap evaluator results")
    previous_identity = _existing_archive_identity(destination) if destination.exists() else None
    filenames = {"metrics.json": MAX_RESULT_JSON_BYTES, "correct.json": MAX_RESULT_JSON_BYTES}
    for name, maximum in (
        (EVIDENCE_FILENAME, MAX_EVIDENCE_BYTES),
        ("qualification.json", MAX_RESULT_JSON_BYTES),
    ):
        if (results_dir / name).exists() or (results_dir / name).is_symlink():
            filenames[name] = maximum
    with tempfile.TemporaryDirectory(
        prefix=".baseline-stage-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        hashes = {}
        for name, maximum in filenames.items():
            payload = _read_regular(results_dir / name, maximum)
            descriptor = os.open(staging / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            hashes[name] = hashlib.sha256(payload).hexdigest()
        metrics, archived_correct = validate_evaluator_outputs(staging, metric_names)
        if metrics["combined_score"] != combined_score or archived_correct is not correct:
            raise ValueError("Baseline result changed after validation")
        if feasible is not None:
            qualification = _load_mapping(staging / "qualification.json")
            if (
                set(qualification) != {"schema_version", "feasible"}
                or type(qualification["schema_version"]) is not int
                or qualification["schema_version"] != 1
                or qualification["feasible"] is not feasible
            ):
                raise ValueError("Baseline qualification changed after validation")
        elif "qualification.json" in filenames:
            raise ValueError("Unexpected baseline qualification artifact")
        descriptor = os.open(
            staging / "archive_manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"schema_version": 1, "files": hashes}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        previous = None
        if previous_identity is not None:
            if _existing_archive_identity(destination) != previous_identity:
                raise ValueError("Existing baseline archive directory changed before publication")
            previous = destination.with_name(f"{destination.name}.previous-{secrets.token_hex(8)}")
            destination.rename(previous)
        elif destination.exists() or destination.is_symlink():
            raise ValueError("Baseline archive destination appeared during preparation")
        try:
            staging.rename(destination)
        except BaseException:
            if previous is not None:
                previous.rename(destination)
            raise
