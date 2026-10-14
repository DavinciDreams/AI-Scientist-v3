"""Validated contract and program identities for brokered retrofit copies."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat

from src.investigation.task_contract import (
    TASK_CONTRACT_FILENAME,
    load_task_contract,
    task_scaffold_fingerprint,
)

_MAX_TREE_BYTES = 512 * 1024 * 1024
_MAX_TREE_ENTRIES = 100_000
_COMPLETION_MARKER = ".retrofit_complete.json"
_TASK_RUNTIME_ROOT_ENTRIES = frozenset({".venv"})
_BUNDLE_DIGEST_OMISSIONS = _TASK_RUNTIME_ROOT_ENTRIES | {_COMPLETION_MARKER}


def copy_brokered_bundle(source: Path, destination: Path) -> None:
    """Stage scientific inputs without the host-managed task environment."""
    def ignore(directory: str, names: list[str]) -> set[str]:
        if Path(directory) == source:
            return set(names) & _TASK_RUNTIME_ROOT_ENTRIES
        return set()

    shutil.copytree(source, destination, symlinks=True, ignore=ignore)


def brokered_preservation_metadata(
    bundle_dir: Path,
    task_source_dir: Path | None,
) -> dict[str, str] | None:
    """Require a valid contract before treating a program as host-brokered."""
    sources = [bundle_dir]
    if task_source_dir is not None:
        sources.insert(0, task_source_dir)
    for source in sources:
        contract_path = source / TASK_CONTRACT_FILENAME
        if not contract_path.exists() and not contract_path.is_symlink():
            continue
        contract = load_task_contract(contract_path)
        if contract.candidate.interface != "python_rpc_v1:handle":
            raise ValueError("unsupported candidate interface for brokered retrofit")
        return {
            "mode": "brokered_preserve",
            "source_program_tree_sha256": _strict_tree_digest(bundle_dir / "program_0"),
            "source_bundle_tree_sha256": _strict_tree_digest(
                bundle_dir, omitted_root_entries=_BUNDLE_DIGEST_OMISSIONS,
            ),
            "task_scaffold_sha256": task_scaffold_fingerprint(source),
        }
    return None


def preserved_program_digest(bundle_dir: Path, metadata: dict[str, str]) -> str:
    for name in _TASK_RUNTIME_ROOT_ENTRIES:
        runtime = bundle_dir / name
        if runtime.exists() or runtime.is_symlink():
            raise ValueError("brokered copy contains an unstaged task runtime")
    digest = _strict_tree_digest(bundle_dir / "program_0")
    if digest != metadata["source_program_tree_sha256"]:
        raise ValueError("brokered retrofit changed the selected program")
    bundle_digest = _strict_tree_digest(
        bundle_dir, omitted_root_entries=_BUNDLE_DIGEST_OMISSIONS,
    )
    if bundle_digest != metadata["source_bundle_tree_sha256"]:
        raise ValueError("brokered retrofit changed the sealed bundle")
    return digest


def _strict_tree_digest(
    root: Path, *, omitted_root_entries: frozenset[str] = frozenset(),
) -> str:
    """Hash all files and directories, rejecting links and special files."""
    digest = hashlib.sha256()
    entry_count = 0
    total_bytes = 0

    def record(kind: bytes, relative: str, payload: bytes = b"") -> None:
        for value in (kind, relative.encode(), payload):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)

    def visit(directory_fd: int, prefix: str, depth: int) -> None:
        nonlocal entry_count, total_bytes
        if depth > 256:
            raise ValueError("preserved bundle exceeds directory depth limit")
        directory_before = os.fstat(directory_fd)
        record(b"D", prefix)
        for name in sorted(os.listdir(directory_fd)):
            if not prefix and name in omitted_root_entries:
                continue
            entry_count += 1
            if entry_count > _MAX_TREE_ENTRIES:
                raise ValueError("preserved bundle exceeds entry limit")
            relative = f"{prefix}/{name}" if prefix else name
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                raise ValueError(f"preserved bundle contains a link or special file: {relative}")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if stat.S_ISDIR(metadata.st_mode):
                flags |= os.O_DIRECTORY
            child_fd = os.open(name, flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(child_fd)
                if _file_identity(opened) != _file_identity(metadata):
                    raise ValueError("preserved bundle changed while hashing")
                if stat.S_ISDIR(opened.st_mode):
                    visit(child_fd, relative, depth + 1)
                    continue
                total_bytes += opened.st_size
                if total_bytes > _MAX_TREE_BYTES:
                    raise ValueError("preserved bundle exceeds byte limit")
                remaining = opened.st_size
                file_digest = hashlib.sha256()
                while remaining:
                    chunk = os.read(child_fd, min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("preserved file changed while hashing")
                    remaining -= len(chunk)
                    file_digest.update(chunk)
                if os.read(child_fd, 1) or (
                    _file_identity(os.fstat(child_fd)) != _file_identity(opened)
                ):
                    raise ValueError("preserved file changed while hashing")
                record(b"F", relative, file_digest.digest())
            finally:
                os.close(child_fd)
        if _file_identity(os.fstat(directory_fd)) != _file_identity(directory_before):
            raise ValueError("preserved directory changed while hashing")

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        visit(root_fd, "", 0)
    finally:
        os.close(root_fd)
    return digest.hexdigest()


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
