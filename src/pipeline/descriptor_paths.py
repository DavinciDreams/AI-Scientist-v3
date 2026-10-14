"""Portable filesystem paths for already-open directory descriptors."""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from pathlib import Path

_DESCRIPTOR_BASES = (Path("/proc/self/fd"), Path("/dev/fd"))


class DescriptorPathError(RuntimeError):
    """No filesystem path safely identifies an open directory descriptor."""


def directory_descriptor_path(
    descriptor: int,
    *,
    descriptor_bases: Sequence[Path] = _DESCRIPTOR_BASES,
) -> Path:
    """Return the first descriptor path whose target matches the open directory."""
    expected = os.fstat(descriptor)
    if not stat.S_ISDIR(expected.st_mode):
        raise DescriptorPathError("descriptor path target is not a directory")
    for base in descriptor_bases:
        candidate = base / str(descriptor)
        try:
            actual = candidate.stat()
        except OSError:
            continue
        if (
            stat.S_ISDIR(actual.st_mode)
            and actual.st_dev == expected.st_dev
            and actual.st_ino == expected.st_ino
        ):
            return candidate
    raise DescriptorPathError("no safe directory descriptor path is available")
