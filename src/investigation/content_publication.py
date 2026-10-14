"""Task-neutral checks that keep unpublished file content out of public trees."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _regular_file_digest(path: Path) -> bytes:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("content publication path must be a regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError("content publication path changed while opening")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        finished = os.fstat(descriptor)
        if (finished.st_size, finished.st_mtime_ns) != (
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise ValueError("content publication path changed while hashing")
        return digest.digest()
    finally:
        os.close(descriptor)


def _digests(paths: Iterable[Path]) -> frozenset[bytes]:
    return frozenset(_regular_file_digest(path) for path in paths)


@dataclass(frozen=True, slots=True)
class ContentPublicationGuard:
    """Detect copies of sealed bytes that were not already publicly visible."""

    unpublished_digests: frozenset[bytes]

    @classmethod
    def capture(
        cls,
        *,
        sealed_files: Iterable[Path],
        public_files: Iterable[Path],
    ) -> ContentPublicationGuard:
        return cls(_digests(sealed_files) - _digests(public_files))

    def first_violation(self, published_files: Iterable[Path]) -> Path | None:
        for path in published_files:
            if _regular_file_digest(path) in self.unpublished_digests:
                return path
        return None
