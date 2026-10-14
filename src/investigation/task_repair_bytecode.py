"""Verified quarantine for bytecode emitted during scaffold repair."""

from __future__ import annotations

import hashlib
import _imp
import importlib.util
import json
import os
import re
import struct
import sys
from pathlib import Path

_BYTECODE_NAME = re.compile(r"^(?P<source>.+)\.(?P<tag>cpython-\d{2,3})(?:\.opt-\d+)?\.pyc$")
_PYC_HEADER_SIZE = 16
_MAX_GENERATED_BYTECODE_BYTES = 64 * 1024 * 1024


class TaskRepairBytecodeError(ValueError):
    """Generated repair bytecode could not be safely quarantined."""


def quarantine_generated_bytecode(
    task_dir: Path,
    quarantine_dir: Path,
    manifest_path: Path | None = None,
    accepted_caches: frozenset[tuple[str, bytes]] | None = None,
) -> tuple[str, ...]:
    """Move verified CPython caches out of the repair tree without deleting evidence."""
    _task_files(task_dir)
    cache_directories = tuple(
        sorted(
            (path for path in task_dir.rglob("__pycache__") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
    )
    accepted = accepted_caches or _current_cache_metadata()
    artifacts: list[Path] = []
    for cache_directory in cache_directories:
        for artifact in cache_directory.iterdir():
            relative = artifact.relative_to(task_dir).as_posix()
            if not _is_generated_bytecode(artifact, cache_directory, accepted):
                raise TaskRepairBytecodeError(f"unexpected repair cache artifact: {relative}")
            artifacts.append(artifact)
    if not artifacts:
        return ()
    records = [_bytecode_record(task_dir, artifact) for artifact in artifacts]
    _prepare_quarantine_directory(quarantine_dir)
    quarantine_device = quarantine_dir.parent.stat().st_dev
    if any(cache.stat().st_dev != quarantine_device for cache in cache_directories):
        quarantine_dir.rmdir()
        raise TaskRepairBytecodeError("repair cache quarantine must share the task filesystem")
    for cache_directory in cache_directories:
        cache_relative = cache_directory.relative_to(task_dir)
        destination = quarantine_dir / cache_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(cache_directory, destination)
        except OSError as exc:
            raise TaskRepairBytecodeError(
                f"could not quarantine repair cache: {cache_relative.as_posix()}"
            ) from exc
    quarantined_files = _task_files(quarantine_dir)
    expected_files = {str(record["path"]): str(record["sha256"]) for record in records}
    if quarantined_files != expected_files:
        raise TaskRepairBytecodeError("repair cache changed while it was being quarantined")
    for record in records:
        record_relative = Path(str(record["path"]))
        artifact = quarantine_dir / record_relative
        source_cache = task_dir / record_relative.parent
        if not _is_generated_bytecode(artifact, source_cache, accepted):
            raise TaskRepairBytecodeError(
                f"repair cache changed while it was being quarantined: {record_relative.as_posix()}"
            )
    final_records = [
        _bytecode_record(quarantine_dir, quarantine_dir / path) for path in expected_files
    ]
    if {str(record["path"]): str(record["sha256"]) for record in final_records} != expected_files:
        raise TaskRepairBytecodeError("repair cache changed before audit publication")
    if manifest_path is not None:
        _atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "artifacts": sorted(final_records, key=lambda item: item["path"]),
            },
        )
    return tuple(sorted(artifact.relative_to(task_dir).as_posix() for artifact in artifacts))


def _prepare_quarantine_directory(quarantine_dir: Path) -> None:
    if quarantine_dir.is_symlink() or quarantine_dir.exists():
        raise TaskRepairBytecodeError(f"unsafe repair cache quarantine: {quarantine_dir}")
    try:
        quarantine_dir.mkdir(mode=0o700, parents=False)
    except OSError as exc:
        raise TaskRepairBytecodeError(
            f"could not create repair cache quarantine: {quarantine_dir}"
        ) from exc


def _current_cache_metadata() -> frozenset[tuple[str, bytes]]:
    cache_tag = sys.implementation.cache_tag
    if cache_tag is None:  # pragma: no cover - CPython always supplies a tag
        raise TaskRepairBytecodeError("Python runtime has no cache tag")
    return frozenset({(cache_tag, importlib.util.MAGIC_NUMBER)})


def _is_generated_bytecode(
    artifact: Path,
    cache_directory: Path,
    accepted_caches: frozenset[tuple[str, bytes]],
) -> bool:
    if artifact.is_symlink() or not artifact.is_file():
        return False
    match = _BYTECODE_NAME.fullmatch(artifact.name)
    if match is None:
        return False
    source = cache_directory.parent / f"{match.group('source')}.py"
    if source.is_symlink() or not source.is_file():
        return False
    try:
        artifact_stat = artifact.stat()
        with artifact.open("rb") as handle:
            header = handle.read(_PYC_HEADER_SIZE)
            has_payload = bool(handle.read(1))
        source_stat = source.stat()
    except OSError:
        return False
    if (
        artifact_stat.st_size > _MAX_GENERATED_BYTECODE_BYTES
        or source_stat.st_size > _MAX_GENERATED_BYTECODE_BYTES
        or len(header) != _PYC_HEADER_SIZE
        or not has_payload
        or (match.group("tag"), header[:4]) not in accepted_caches
    ):
        return False
    flags = struct.unpack_from("<I", header, 4)[0]
    if flags == 0:
        timestamp, source_size = struct.unpack_from("<II", header, 8)
        return (
            timestamp == int(source_stat.st_mtime) & 0xFFFFFFFF
            and source_size == source_stat.st_size
        )
    if flags in {1, 3}:
        magic = int.from_bytes(header[:4], "little")
        return header[8:16] == _imp.source_hash(magic, source.read_bytes())
    return False


def _bytecode_record(task_dir: Path, artifact: Path) -> dict[str, str | int]:
    try:
        with artifact.open("rb") as handle:
            magic = handle.read(4).hex()
        size = artifact.stat().st_size
    except OSError as exc:  # pragma: no cover - concurrent filesystem mutation
        raise TaskRepairBytecodeError(f"could not record repair cache: {artifact}") from exc
    return {
        "path": artifact.relative_to(task_dir).as_posix(),
        "size": size,
        "sha256": _sha256(artifact),
        "magic": magic,
    }


def parse_cache_metadata(values: list[str]) -> frozenset[tuple[str, bytes]]:
    if not values:
        return _current_cache_metadata()
    parsed: set[tuple[str, bytes]] = set()
    for value in values:
        tag, separator, magic_hex = value.partition(":")
        try:
            magic = bytes.fromhex(magic_hex)
        except ValueError as exc:
            raise TaskRepairBytecodeError(f"invalid accepted cache metadata: {value}") from exc
        if not separator or re.fullmatch(r"cpython-\d{2,3}", tag) is None or len(magic) != 4:
            raise TaskRepairBytecodeError(f"invalid accepted cache metadata: {value}")
        parsed.add((tag, magic))
    return frozenset(parsed)


def _task_files(task_dir: Path) -> dict[str, str]:
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise TaskRepairBytecodeError(f"unsafe task directory: {task_dir}")
    files: dict[str, str] = {}
    for path in sorted(task_dir.rglob("*")):
        relative = path.relative_to(task_dir)
        if path.is_symlink():
            raise TaskRepairBytecodeError(f"repair scope contains symlink: {relative.as_posix()}")
        if path.is_file():
            files[relative.as_posix()] = _sha256(path)
        elif not path.is_dir():
            raise TaskRepairBytecodeError(
                f"repair scope contains special path: {relative.as_posix()}"
            )
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
