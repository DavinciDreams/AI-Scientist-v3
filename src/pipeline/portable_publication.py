"""Crash-recoverable directory publication for filesystems without rename exchange."""

from __future__ import annotations

import ctypes
import errno
import os
import secrets
import stat
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .descriptor_paths import directory_descriptor_path
from .publication_cleanup import PinnedDeletionError, remove_pinned_directory_at
from .publication_journal import (
    DirectoryIdentity,
    JournalError as PortablePublicationError,
    PublicationRecord,
    PublicationStage,
    create_record as _create_journal_record,
    journal_path,
    read_record as _read_journal_record,
    remove_record as _remove_journal_record,
    write_record as _write_journal_record,
)
from .publication_tree import (
    PublicationTreeError,
    digest_directory_at,
    sync_and_digest_directory_at,
)

_STAGE_RANK: dict[PublicationStage, int] = {
    "prepared": 0,
    "original_moved": 1,
    "replacement_moved": 2,
    "exchanged": 2,
}
_UNSUPPORTED_EXCHANGE_ERRORS = {
    errno.EINVAL,
    errno.ENOSYS,
    errno.ENOTSUP,
    errno.EOPNOTSUPP,
}


TransitionValidator = Callable[[Path, Path], None]


def atomic_exchange(left: Path, right: Path) -> None:
    """Atomically exchange two directories when libc and the filesystem support it."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("atomic directory exchange is unavailable on this platform")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(left), -100, os.fsencode(right), 2)
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(left), str(right))


def atomic_exchange_prepared(record: PublicationRecord, *, parent: Path) -> None:
    """Exchange the two identities named by a durable prepared record."""
    parent_fd = _open_parent(parent)
    try:
        _require_identity(parent_fd, record.paper_name, record.original)
        _require_identity(parent_fd, record.replacement_name, record.replacement)
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError("atomic directory exchange is unavailable on this platform")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_fd,
            os.fsencode(record.paper_name),
            parent_fd,
            os.fsencode(record.replacement_name),
            2,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
    finally:
        os.close(parent_fd)


def sync_directory_path(path: Path) -> None:
    """Fsync one directory after a namespace transition."""
    directory_fd = _open_parent(path)
    try:
        _sync_directory(directory_fd)
    finally:
        os.close(directory_fd)


def sync_directory_tree(path: Path) -> None:
    """Fsync every regular file and directory in a tree, bottom-up."""
    path = Path(os.path.abspath(path))
    parent_fd = _open_parent(path.parent)
    try:
        try:
            sync_and_digest_directory_at(parent_fd, path.name)
        except PublicationTreeError as exc:
            raise PortablePublicationError(str(exc)) from exc
    finally:
        os.close(parent_fd)


def exchange_is_unsupported(error: OSError) -> bool:
    """Return whether an exchange failed because the primitive is unavailable."""
    return error.errno is None or error.errno in _UNSUPPORTED_EXCHANGE_ERRORS


def publication_journal_path(paper_dir: Path) -> Path:
    """Return the parent-scoped journal protected by the paper publication lock."""
    return journal_path(paper_dir)


def portable_replacement_pending(paper_dir: Path) -> bool:
    """Return whether publication recovery owns sibling staging paths."""
    try:
        publication_journal_path(paper_dir).lstat()
    except FileNotFoundError:
        return False
    return True


def portable_replacement_is_recoverable(paper_dir: Path) -> bool:
    """Return whether a journal and its directory identities form a recoverable state."""
    paper_dir = Path(os.path.abspath(paper_dir))
    try:
        record = _read_record(
            publication_journal_path(paper_dir),
            expected_paper_name=paper_dir.name,
        )
        parent_fd = _open_parent(paper_dir.parent)
        try:
            observed_stage = _observed_stage(parent_fd, record)
            if _STAGE_RANK[observed_stage] < _STAGE_RANK[record.stage]:
                return False
        finally:
            os.close(parent_fd)
    except (OSError, PortablePublicationError):
        return False
    return True


def directory_identity(path: Path) -> DirectoryIdentity:
    """Return a no-follow directory identity relative to its pinned parent."""
    absolute = Path(os.path.abspath(path))
    parent_fd = _open_parent(absolute.parent)
    try:
        return _directory_identity(parent_fd, absolute.name)
    finally:
        os.close(parent_fd)


def prepare_directory_replacement(
    paper_dir: Path,
    replacement_dir: Path,
    *,
    expected_original: DirectoryIdentity | None = None,
    expected_replacement: DirectoryIdentity | None = None,
) -> PublicationRecord:
    """Sync, identify, and durably journal a sibling replacement."""
    paper_dir = Path(os.path.abspath(paper_dir))
    replacement_dir = Path(os.path.abspath(replacement_dir))
    if paper_dir.parent != replacement_dir.parent:
        raise PortablePublicationError("portable publication requires sibling directories")
    if portable_replacement_pending(paper_dir):
        raise PortablePublicationError("portable publication journal already exists")
    parent = paper_dir.parent
    parent_fd = _open_parent(parent)
    try:
        original = _directory_identity(parent_fd, paper_dir.name)
        replacement = _directory_identity(parent_fd, replacement_dir.name)
        if expected_original is not None and original != expected_original:
            raise PortablePublicationError("portable publication original entry changed")
        if expected_replacement is not None and replacement != expected_replacement:
            raise PortablePublicationError("portable publication replacement entry changed")
        try:
            replacement_digest = sync_and_digest_directory_at(
                parent_fd, replacement_dir.name
            )
        except PublicationTreeError as exc:
            raise PortablePublicationError(str(exc)) from exc
        backup_name = _unused_backup_name(parent_fd, paper_dir.name)
        record = PublicationRecord(
            paper_name=paper_dir.name,
            replacement_name=replacement_dir.name,
            backup_name=backup_name,
            original=original,
            replacement=replacement,
            replacement_digest=replacement_digest,
            stage="prepared",
        )
        return _create_record(parent, parent_fd, record)
    finally:
        os.close(parent_fd)


def finish_directory_replacement(
    paper_dir: Path,
    record: PublicationRecord,
    *,
    validate_transition: TransitionValidator | None = None,
) -> None:
    """Complete either an exchanged or ordinary-rename prepared replacement."""
    _advance_publication(
        Path(os.path.abspath(paper_dir)).parent,
        record,
        validate_transition=validate_transition,
    )


def cancel_prepared_replacement(paper_dir: Path, record: PublicationRecord) -> None:
    """Remove an intent record only while both prepared identities remain unchanged."""
    parent_fd = _open_parent(Path(os.path.abspath(paper_dir)).parent)
    try:
        if _observed_stage(parent_fd, record) != "prepared":
            raise PortablePublicationError("cannot cancel a publication that already transitioned")
        _remove_journal(Path(os.path.abspath(paper_dir)).parent, parent_fd, record)
    finally:
        os.close(parent_fd)


def replace_directory_portably(
    paper_dir: Path,
    replacement_dir: Path,
    *,
    validate_transition: TransitionValidator | None = None,
) -> None:
    """Publish a prepared sibling directory using journaled ordinary renames."""
    record = prepare_directory_replacement(paper_dir, replacement_dir)
    finish_directory_replacement(
        paper_dir,
        record,
        validate_transition=validate_transition,
    )


def recover_portable_replacement(
    paper_dir: Path, *, validate_transition: TransitionValidator | None = None
) -> bool:
    """Finish an interrupted portable replacement while the caller holds its lock."""
    paper_dir = Path(os.path.abspath(paper_dir))
    journal = publication_journal_path(paper_dir)
    try:
        record = _read_record(journal, expected_paper_name=paper_dir.name)
    except FileNotFoundError:
        return False
    _advance_publication(
        paper_dir.parent,
        record,
        validate_transition=validate_transition,
    )
    return True


def _advance_publication(
    parent: Path,
    record: PublicationRecord,
    *,
    validate_transition: TransitionValidator | None = None,
) -> None:
    parent_fd = _open_parent(parent)
    try:
        record = _normalize_stage(parent, parent_fd, record)
        _require_replacement_digest(parent_fd, record)
        if validate_transition is not None:
            source_name, candidate_name = _validation_names(parent_fd, record)
            if source_name is None:
                if not record.evidence_validated:
                    raise PortablePublicationError(
                        "portable publication lost its unvalidated original"
                    )
            else:
                pinned_parent = directory_descriptor_path(parent_fd)
                validate_transition(pinned_parent / source_name, pinned_parent / candidate_name)
                if not record.evidence_validated:
                    record = replace(record, evidence_validated=True)
                    record = _write_record(parent, parent_fd, record)
        _require_transition_identities(parent_fd, record)
        if record.stage == "prepared":
            _rename_entry(parent_fd, record.paper_name, record.backup_name)
            _sync_directory(parent_fd)
            record = replace(record, stage="original_moved")
            record = _write_record(parent, parent_fd, record)
        if record.stage == "original_moved":
            _rename_entry(parent_fd, record.replacement_name, record.paper_name)
            _sync_directory(parent_fd)
            record = replace(record, stage="replacement_moved")
            record = _write_record(parent, parent_fd, record)
        _require_identity(parent_fd, record.paper_name, record.replacement)
        retired_name = (
            record.replacement_name if record.stage == "exchanged" else record.backup_name
        )
        _remove_backup(parent_fd, retired_name, record.original)
        _sync_directory(parent_fd)
        _remove_journal(parent, parent_fd, record)
    finally:
        os.close(parent_fd)


def _normalize_stage(
    parent: Path, parent_fd: int, record: PublicationRecord
) -> PublicationRecord:
    observed_stage = _observed_stage(parent_fd, record)
    if _STAGE_RANK[observed_stage] < _STAGE_RANK[record.stage]:
        raise PortablePublicationError(
            f"portable publication state is inconsistent at stage {record.stage}"
        )
    if observed_stage != record.stage:
        _sync_directory(parent_fd)
        record = replace(record, stage=observed_stage)
        record = _write_record(parent, parent_fd, record)
    return record


def _observed_stage(parent_fd: int, record: PublicationRecord) -> PublicationStage:
    paper = _optional_directory_identity(parent_fd, record.paper_name)
    replacement = _optional_directory_identity(parent_fd, record.replacement_name)
    backup = _optional_directory_identity(parent_fd, record.backup_name)
    replacement_quarantine = _optional_directory_identity(
        parent_fd, f"{record.replacement_name}.delete"
    )
    backup_quarantine = _optional_directory_identity(parent_fd, f"{record.backup_name}.delete")
    if paper == record.original and replacement == record.replacement and backup is None:
        return "prepared"
    if paper is None and replacement == record.replacement and backup == record.original:
        return "original_moved"
    if (
        paper == record.replacement
        and backup is None
        and replacement in {record.original, None}
        and replacement_quarantine in {record.original, None}
        and (replacement == record.original or replacement_quarantine == record.original)
    ):
        return "exchanged"
    if (
        paper == record.replacement
        and replacement is None
        and replacement_quarantine is None
        and backup in {record.original, None}
        and backup_quarantine in {record.original, None}
    ):
        if backup == record.original or backup_quarantine == record.original:
            return "replacement_moved"
        return "exchanged" if record.stage == "exchanged" else "replacement_moved"
    raise PortablePublicationError(
        f"portable publication state is inconsistent at stage {record.stage}"
    )


def _create_record(
    parent: Path, parent_fd: int, record: PublicationRecord
) -> PublicationRecord:
    journal_name = publication_journal_path(parent / record.paper_name).name
    return _create_journal_record(parent_fd, journal_name, record)


def _write_record(
    parent: Path, parent_fd: int, record: PublicationRecord
) -> PublicationRecord:
    journal_name = publication_journal_path(parent / record.paper_name).name
    return _write_journal_record(parent_fd, journal_name, record)


def _read_record(path: Path, *, expected_paper_name: str) -> PublicationRecord:
    return _read_journal_record(path, expected_paper_name=expected_paper_name)


def _open_parent(parent: Path) -> int:
    return os.open(
        parent,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )


def _directory_identity(parent_fd: int, name: str) -> DirectoryIdentity:
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise PortablePublicationError(f"portable publication entry is not a directory: {name}")
    return DirectoryIdentity(device=metadata.st_dev, inode=metadata.st_ino)


def _optional_directory_identity(parent_fd: int, name: str) -> DirectoryIdentity | None:
    try:
        return _directory_identity(parent_fd, name)
    except FileNotFoundError:
        return None


def _require_identity(parent_fd: int, name: str, expected: DirectoryIdentity) -> None:
    if _directory_identity(parent_fd, name) != expected:
        raise PortablePublicationError(f"portable publication entry changed: {name}")


def _require_replacement_digest(parent_fd: int, record: PublicationRecord) -> None:
    name = (
        record.paper_name
        if _optional_directory_identity(parent_fd, record.paper_name) == record.replacement
        else record.replacement_name
    )
    try:
        actual = digest_directory_at(parent_fd, name)
    except PublicationTreeError as exc:
        raise PortablePublicationError(str(exc)) from exc
    if actual != record.replacement_digest:
        raise PortablePublicationError("portable publication replacement digest changed")


def _unused_backup_name(parent_fd: int, paper_name: str) -> str:
    for _ in range(100):
        candidate = f".{paper_name}.original-{secrets.token_hex(8)}"
        try:
            os.stat(candidate, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return candidate
    raise PortablePublicationError("could not allocate portable publication backup name")


def _rename_entry(parent_fd: int, source: str, destination: str) -> None:
    os.rename(source, destination, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)


def _remove_backup(
    parent_fd: int, backup_name: str, expected: DirectoryIdentity
) -> None:
    try:
        try:
            remove_pinned_directory_at(
                parent_fd,
                backup_name,
                expected_device=expected.device,
                expected_inode=expected.inode,
                quarantine_name=f"{backup_name}.delete",
            )
        except PinnedDeletionError as exc:
            raise PortablePublicationError("portable publication backup changed") from exc
    finally:
        pass


def _remove_journal(parent: Path, parent_fd: int, record: PublicationRecord) -> None:
    journal_name = publication_journal_path(parent / record.paper_name).name
    _remove_journal_record(parent_fd, journal_name, record)


def _validation_names(
    parent_fd: int, record: PublicationRecord
) -> tuple[str | None, str]:
    if record.stage == "prepared":
        source_name, candidate_name = record.paper_name, record.replacement_name
    elif record.stage == "original_moved":
        source_name, candidate_name = record.backup_name, record.replacement_name
    elif record.stage == "replacement_moved":
        source_name, candidate_name = record.backup_name, record.paper_name
    else:
        source_name, candidate_name = record.replacement_name, record.paper_name
    if _optional_directory_identity(parent_fd, source_name) == record.original:
        return source_name, candidate_name
    quarantine_name = f"{source_name}.delete"
    if _optional_directory_identity(parent_fd, quarantine_name) == record.original:
        return quarantine_name, candidate_name
    return None, candidate_name


def _require_transition_identities(parent_fd: int, record: PublicationRecord) -> None:
    source_name, candidate_name = _validation_names(parent_fd, record)
    if source_name is None and record.stage in {"prepared", "original_moved"}:
        raise PortablePublicationError("portable publication original entry changed")
    if source_name is not None:
        _require_identity(parent_fd, source_name, record.original)
    _require_identity(parent_fd, candidate_name, record.replacement)


def _sync_directory(directory_fd: int) -> None:
    os.fsync(directory_fd)
