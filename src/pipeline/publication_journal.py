"""Bounded, inode-checked storage for revision publication journals."""

from __future__ import annotations

import json
import os
import secrets
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

PublicationStage = Literal["prepared", "original_moved", "replacement_moved", "exchanged"]
_SCHEMA_VERSION = 1
_MAX_BYTES = 64 * 1024


class JournalError(RuntimeError):
    """A publication journal is malformed or changed unexpectedly."""


@dataclass(frozen=True)
class DirectoryIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class PublicationRecord:
    paper_name: str
    replacement_name: str
    backup_name: str
    original: DirectoryIdentity
    replacement: DirectoryIdentity
    replacement_digest: str
    stage: PublicationStage
    evidence_validated: bool = False
    journal_identity: DirectoryIdentity | None = None


def journal_path(paper_dir: Path) -> Path:
    return paper_dir.parent / f".{paper_dir.name}.revision-publication.json"


def read_record(path: Path, *, expected_paper_name: str) -> PublicationRecord:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise JournalError("portable publication journal is not regular")
        content = stream.read(_MAX_BYTES + 1)
    if len(content) > _MAX_BYTES:
        raise JournalError("portable publication journal is too large")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise JournalError("portable publication journal is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION:
        raise JournalError("unsupported portable publication journal")
    paper_name = _record_name(payload, "paper_name")
    replacement_name = _record_name(payload, "replacement_name")
    backup_name = _record_name(payload, "backup_name")
    if paper_name != expected_paper_name:
        raise JournalError("portable publication journal names another paper")
    if not replacement_name.startswith(f".{paper_name}.revision-"):
        raise JournalError("portable publication replacement name is invalid")
    if not backup_name.startswith(f".{paper_name}.original-"):
        raise JournalError("portable publication backup name is invalid")
    stage = payload.get("stage")
    if not isinstance(stage, str) or stage not in {
        "prepared",
        "original_moved",
        "replacement_moved",
        "exchanged",
    }:
        raise JournalError("portable publication stage is invalid")
    digest = payload.get("replacement_digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise JournalError("portable publication replacement digest is invalid")
    return PublicationRecord(
        paper_name=paper_name,
        replacement_name=replacement_name,
        backup_name=backup_name,
        original=_record_identity(payload, "original"),
        replacement=_record_identity(payload, "replacement"),
        replacement_digest=digest,
        stage=cast(PublicationStage, stage),
        evidence_validated=payload.get("evidence_validated") is True,
        journal_identity=DirectoryIdentity(metadata.st_dev, metadata.st_ino),
    )


def create_record(parent_fd: int, journal_name: str, record: PublicationRecord) -> PublicationRecord:
    temporary_name = _unused_name(parent_fd, f".{journal_name}")
    descriptor = _create_exclusive(parent_fd, temporary_name)
    try:
        _write_payload(descriptor, record)
        try:
            os.link(
                temporary_name,
                journal_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise JournalError("portable publication journal already exists") from exc
        os.fsync(parent_fd)
        identity = _identity_at(parent_fd, journal_name)
        return replace(record, journal_identity=identity)
    finally:
        os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.fsync(parent_fd)


def write_record(parent_fd: int, journal_name: str, record: PublicationRecord) -> PublicationRecord:
    _require_journal_identity(parent_fd, journal_name, record)
    temporary_name = _unused_name(parent_fd, f".{journal_name}")
    descriptor = _create_exclusive(parent_fd, temporary_name)
    try:
        _write_payload(descriptor, record)
        _require_journal_identity(parent_fd, journal_name, record)
        os.replace(
            temporary_name,
            journal_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
        return replace(record, journal_identity=_identity_at(parent_fd, journal_name))
    finally:
        os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass


def remove_record(parent_fd: int, journal_name: str, record: PublicationRecord) -> None:
    _require_journal_identity(parent_fd, journal_name, record)
    os.unlink(journal_name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _record_payload(record: PublicationRecord) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "stage": record.stage,
        "paper_name": record.paper_name,
        "replacement_name": record.replacement_name,
        "backup_name": record.backup_name,
        "original": {"device": record.original.device, "inode": record.original.inode},
        "replacement": {
            "device": record.replacement.device,
            "inode": record.replacement.inode,
        },
        "replacement_digest": record.replacement_digest,
        "evidence_validated": record.evidence_validated,
    }


def _write_payload(descriptor: int, record: PublicationRecord) -> None:
    content = (json.dumps(_record_payload(record), indent=2, sort_keys=True) + "\n").encode()
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written == 0:
            raise OSError("could not write portable publication journal")
        offset += written
    os.fsync(descriptor)


def _create_exclusive(parent_fd: int, name: str) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(name, flags, 0o600, dir_fd=parent_fd)


def _unused_name(parent_fd: int, prefix: str) -> str:
    for _ in range(100):
        candidate = f"{prefix}.{secrets.token_hex(8)}.tmp"
        try:
            os.stat(candidate, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return candidate
    raise JournalError("could not allocate portable publication journal temporary")


def _require_journal_identity(
    parent_fd: int, journal_name: str, record: PublicationRecord
) -> None:
    if record.journal_identity is None or _identity_at(parent_fd, journal_name) != (
        record.journal_identity
    ):
        raise JournalError("portable publication journal changed")


def _identity_at(parent_fd: int, name: str) -> DirectoryIdentity:
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise JournalError("portable publication journal is not regular")
    return DirectoryIdentity(metadata.st_dev, metadata.st_ino)


def _record_name(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise JournalError(f"portable publication {key} is invalid")
    return value


def _record_identity(payload: dict[str, object], key: str) -> DirectoryIdentity:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise JournalError(f"portable publication {key} identity is invalid")
    device = value.get("device")
    inode = value.get("inode")
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise JournalError(f"portable publication {key} device is invalid")
    if not isinstance(inode, int) or isinstance(inode, bool) or inode <= 0:
        raise JournalError(f"portable publication {key} inode is invalid")
    return DirectoryIdentity(device, inode)
