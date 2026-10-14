from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_SEALED_METADATA_FILES = (
    PurePosixPath("task_contract.json"),
    PurePosixPath(".agentic_task_validation.json"),
    PurePosixPath("manual_validation.json"),
)
_OPTIONAL_SEALED_FILES = frozenset(_SEALED_METADATA_FILES[1:])
_IGNORED_ROOTS = frozenset(
    {
        PurePosixPath(".git"),
        PurePosixPath(".venv"),
        PurePosixPath(".search-runtime"),
        PurePosixPath(".final-runtime"),
        PurePosixPath("results"),
    }
)
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True)
class _Entry:
    relative: PurePosixPath
    mode: int
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    symlink_target: str | None = None
    sha256: str | None = None

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode

    @property
    def metadata(self) -> tuple[object, ...]:
        return (
            self.relative,
            self.mode,
            self.device,
            self.inode,
            self.size,
            self.modified_ns,
            self.changed_ns,
            self.symlink_target,
        )


@dataclass(frozen=True)
class _Snapshot:
    root: _Entry
    entries: tuple[_Entry, ...]

    @property
    def metadata(self) -> tuple[tuple[object, ...], ...]:
        return (self.root.metadata, *(entry.metadata for entry in self.entries))


def _safe_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe task-relative path: {value!r}")
    return relative


def _ignored(relative: PurePosixPath) -> bool:
    return (
        relative in _IGNORED_ROOTS
        or "__pycache__" in relative.parts
        or relative.name.endswith((".pyc", ".pyo"))
    )


def _is_below(relative: PurePosixPath, parent: PurePosixPath) -> bool:
    return relative == parent or parent in relative.parents


def _from_stat(
    relative: PurePosixPath,
    metadata: os.stat_result,
    *,
    symlink_target: str | None = None,
    sha256: str | None = None,
) -> _Entry:
    return _Entry(
        relative=relative,
        mode=metadata.st_mode,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
        symlink_target=symlink_target,
        sha256=sha256,
    )


def _same_file(actual: os.stat_result, expected: _Entry) -> bool:
    return (
        actual.st_mode == expected.mode
        and actual.st_dev == expected.device
        and actual.st_ino == expected.inode
        and actual.st_size == expected.size
        and actual.st_mtime_ns == expected.modified_ns
        and actual.st_ctime_ns == expected.changed_ns
    )


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    while block := os.read(descriptor, 1024 * 1024):
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _trusted_digests(
    source: Path, expected_contract_sha256: str
) -> dict[PurePosixPath, str]:
    contract_bytes = (source / "task_contract.json").read_bytes()
    if hashlib.sha256(contract_bytes).hexdigest() != expected_contract_sha256:
        raise RuntimeError("task contract changed before private snapshot creation")
    contract = json.loads(contract_bytes)
    canonical = contract["canonical_task"]
    trusted = {_safe_relative(canonical["path"]): canonical["sha256"]}
    for artifact in contract["provenance"]["artifacts"]:
        trusted[_safe_relative(artifact["path"])] = artifact["sha256"]
    for path, digest in trusted.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid trusted task digest: {path}")
    return trusted


def _scan(
    source: Path,
    *,
    sealed_files: frozenset[PurePosixPath],
    final_resources: PurePosixPath,
    hash_public_files: bool,
    trusted_digests: dict[PurePosixPath, str],
) -> _Snapshot:
    root_descriptor = os.open(source, _DIRECTORY_FLAGS)
    try:
        root_entry = _from_stat(PurePosixPath("."), os.fstat(root_descriptor))
        entries: list[_Entry] = []

        def visit(directory_descriptor: int, relative_parent: PurePosixPath) -> None:
            with os.scandir(directory_descriptor) as children:
                for child in sorted(children, key=lambda item: item.name):
                    relative = relative_parent / child.name
                    if child.name == ".data-cache":
                        raise ValueError("legacy .data-cache is forbidden in task shadows")
                    if _ignored(relative):
                        continue
                    observed = child.stat(follow_symlinks=False)
                    if stat.S_ISDIR(observed.st_mode):
                        child_descriptor = os.open(
                            child.name,
                            _DIRECTORY_FLAGS,
                            dir_fd=directory_descriptor,
                        )
                        try:
                            pinned = os.fstat(child_descriptor)
                            item = _from_stat(relative, pinned)
                            if item.identity != (observed.st_dev, observed.st_ino):
                                raise RuntimeError(f"task source changed while scanning: {relative}")
                            entries.append(item)
                            visit(child_descriptor, relative)
                        finally:
                            os.close(child_descriptor)
                    elif stat.S_ISREG(observed.st_mode):
                        digest = None
                        if (
                            hash_public_files
                            and relative not in sealed_files
                            and not _is_below(relative, final_resources)
                        ):
                            digest = trusted_digests.get(relative)
                            if digest is None:
                                file_descriptor = os.open(
                                    child.name,
                                    _FILE_FLAGS,
                                    dir_fd=directory_descriptor,
                                )
                                try:
                                    before = os.fstat(file_descriptor)
                                    expected = _from_stat(relative, observed)
                                    if not _same_file(before, expected):
                                        raise RuntimeError(
                                            f"task source changed while scanning: {relative}"
                                        )
                                    digest = _hash_descriptor(file_descriptor)
                                    if not _same_file(os.fstat(file_descriptor), expected):
                                        raise RuntimeError(
                                            f"task source changed while hashing: {relative}"
                                        )
                                finally:
                                    os.close(file_descriptor)
                        entries.append(_from_stat(relative, observed, sha256=digest))
                    elif stat.S_ISLNK(observed.st_mode):
                        entries.append(
                            _from_stat(
                                relative,
                                observed,
                                symlink_target=os.readlink(
                                    child.name, dir_fd=directory_descriptor
                                ),
                            )
                        )
                    else:
                        raise ValueError(f"unsupported task source path type: {relative}")

        visit(root_descriptor, PurePosixPath())
        return _Snapshot(root=root_entry, entries=tuple(entries))
    finally:
        os.close(root_descriptor)


def _verify_required_paths(
    entries: tuple[_Entry, ...],
    sealed_files: frozenset[PurePosixPath],
    final_resources: PurePosixPath,
) -> None:
    by_path = {entry.relative: entry for entry in entries}
    for relative in sealed_files:
        entry = by_path.get(relative)
        if entry is None and relative in _OPTIONAL_SEALED_FILES:
            continue
        if entry is None or not stat.S_ISREG(entry.mode):
            raise ValueError(f"missing regular sealed task file: {relative}")
    final_entry = by_path.get(final_resources)
    if final_entry is None or not stat.S_ISDIR(final_entry.mode):
        raise ValueError(f"missing sealed task resource directory: {final_resources}")


def _open_parent(
    root_descriptor: int,
    relative: PurePosixPath,
    expected: dict[PurePosixPath, _Entry],
) -> int:
    current = os.dup(root_descriptor)
    parent = PurePosixPath()
    try:
        for part in relative.parent.parts:
            parent /= part
            following = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            if not _same_file(os.fstat(following), expected[parent]):
                os.close(following)
                raise RuntimeError(f"task source directory changed: {parent}")
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def _copy_regular(
    root_descriptor: int,
    destination: Path,
    expected: _Entry,
    entries_by_path: dict[PurePosixPath, _Entry],
) -> None:
    parent_descriptor = _open_parent(root_descriptor, expected.relative, entries_by_path)
    source_descriptor = -1
    destination_descriptor = -1
    try:
        source_descriptor = os.open(
            expected.relative.name,
            _FILE_FLAGS,
            dir_fd=parent_descriptor,
        )
        if not _same_file(os.fstat(source_descriptor), expected):
            raise RuntimeError(f"task source changed while snapshotting: {expected.relative}")
        destination_descriptor = os.open(
            destination,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            stat.S_IMODE(expected.mode),
        )
        digest = hashlib.sha256()
        while block := os.read(source_descriptor, 1024 * 1024):
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        if digest.hexdigest() != expected.sha256:
            raise RuntimeError(f"task source content changed: {expected.relative}")
        if not _same_file(os.fstat(source_descriptor), expected):
            raise RuntimeError(f"task source changed while snapshotting: {expected.relative}")
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
        os.close(parent_descriptor)


def _create_placeholder(root: Path, relative: PurePosixPath) -> None:
    destination = root.joinpath(*relative.parts)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    os.close(descriptor)


def create_sealed_task_shadow(
    source: Path,
    shadow: Path,
    *,
    final_evaluator: str,
    final_resources: str,
    contract_sha256: str,
) -> None:
    source = source.resolve(strict=True)
    final_evaluator_path = _safe_relative(final_evaluator)
    final_resources_path = _safe_relative(final_resources)
    sealed_files = frozenset({final_evaluator_path, *_SEALED_METADATA_FILES})
    trusted_digests = _trusted_digests(source, contract_sha256)
    initial = _scan(
        source,
        sealed_files=sealed_files,
        final_resources=final_resources_path,
        hash_public_files=True,
        trusted_digests=trusted_digests,
    )
    _verify_required_paths(initial.entries, sealed_files, final_resources_path)

    sealed_inodes = {
        entry.identity
        for entry in initial.entries
        if stat.S_ISREG(entry.mode)
        and (
            entry.relative in sealed_files
            or _is_below(entry.relative, final_resources_path)
        )
    }
    for entry in initial.entries:
        if (
            stat.S_ISREG(entry.mode)
            and entry.identity in sealed_inodes
            and entry.relative not in sealed_files
            and not _is_below(entry.relative, final_resources_path)
        ):
            raise ValueError(f"public task path aliases sealed content: {entry.relative}")

    if any(shadow.iterdir()):
        raise ValueError(f"task shadow is not empty: {shadow}")
    os.chmod(shadow, 0o700)
    entries_by_path = {entry.relative: entry for entry in initial.entries}
    root_descriptor = os.open(source, _DIRECTORY_FLAGS)
    try:
        if not _same_file(os.fstat(root_descriptor), initial.root):
            raise RuntimeError("task source root changed while snapshotting")
        for item in initial.entries:
            if item.relative in sealed_files or _is_below(
                item.relative, final_resources_path
            ):
                continue
            destination = shadow.joinpath(*item.relative.parts)
            if stat.S_ISDIR(item.mode):
                destination.mkdir(mode=0o700, parents=True, exist_ok=False)
            elif stat.S_ISLNK(item.mode):
                destination.symlink_to(item.symlink_target or "")
            else:
                _copy_regular(root_descriptor, destination, item, entries_by_path)
    finally:
        os.close(root_descriptor)

    for relative in sealed_files:
        _create_placeholder(shadow, relative)
    shadow.joinpath(*final_resources_path.parts).mkdir(mode=0o700, parents=True)
    final = _scan(
        source,
        sealed_files=sealed_files,
        final_resources=final_resources_path,
        hash_public_files=False,
        trusted_digests=trusted_digests,
    )
    if final.metadata != initial.metadata:
        raise RuntimeError("task source changed while building its private snapshot")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("shadow", type=Path)
    parser.add_argument("final_evaluator")
    parser.add_argument("final_resources")
    parser.add_argument("contract_sha256")
    args = parser.parse_args()
    create_sealed_task_shadow(
        args.source,
        args.shadow,
        final_evaluator=args.final_evaluator,
        final_resources=args.final_resources,
        contract_sha256=args.contract_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
