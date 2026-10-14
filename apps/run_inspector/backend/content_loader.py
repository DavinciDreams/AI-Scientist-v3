"""Secure file loading helpers for artifact previews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_FILE_PREVIEW_BYTES = 1_000_000


@dataclass(frozen=True)
class LoadedFile:
    path: Path
    size_bytes: int
    truncated: bool
    text: str


def resolve_path_under_run(run_dir: Path, requested_path: str) -> Path:
    """Resolve a requested path and enforce that it stays under `run_dir`."""
    run_root = run_dir.expanduser().resolve()
    candidate = Path(requested_path).expanduser()
    if not candidate.is_absolute():
        candidate = (run_root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate != run_root and run_root not in candidate.parents:
        raise ValueError("requested path is outside run directory")
    if not candidate.exists():
        raise FileNotFoundError(str(candidate))
    if not candidate.is_file():
        raise IsADirectoryError(str(candidate))
    return candidate


def load_text_file(path: Path, *, max_bytes: int = MAX_FILE_PREVIEW_BYTES) -> LoadedFile:
    """Load UTF-8-ish text content from disk with size cap and binary checks."""
    size_bytes = path.stat().st_size

    with path.open("rb") as handle:
        sample = handle.read(8192)
        if b"\x00" in sample:
            raise ValueError("binary file previews are not supported")

    read_bytes = min(size_bytes, max_bytes)
    with path.open("rb") as handle:
        payload = handle.read(read_bytes)

    text = payload.decode("utf-8", errors="replace")
    return LoadedFile(
        path=path,
        size_bytes=size_bytes,
        truncated=size_bytes > max_bytes,
        text=text,
    )
