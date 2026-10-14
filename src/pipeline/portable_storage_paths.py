"""Filesystem selection and validated path identities for portable runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.pipeline.safe_fs import UnsafeGluePathError, rename_directory_no_replace

_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ArchiveError(RuntimeError):
    """A portable archive cannot be created or restored safely."""


@dataclass(frozen=True)
class RunIdentity:
    task: str
    agent: str
    run_id: str

    def __post_init__(self) -> None:
        for label, value in (
            ("task", self.task),
            ("agent", self.agent),
            ("run ID", self.run_id),
        ):
            if _COMPONENT_RE.fullmatch(value) is None:
                raise ArchiveError(f"invalid {label}: {value!r}")

    def run_root(self, results_root: Path) -> Path:
        return Path(results_root) / self.task / self.agent / self.run_id


def atomic_no_replace_supported(directory: Path) -> bool:
    """Probe the exact publication primitive used by run-local task snapshots."""
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    probe = Path(tempfile.mkdtemp(prefix=".agentic-storage-probe-", dir=root))
    source = probe / "source"
    source.mkdir()
    descriptor = os.open(probe, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            rename_directory_no_replace(
                descriptor,
                source.name,
                "destination",
                role="portable storage probe",
            )
        except UnsafeGluePathError as exc:
            if "not supported" in str(exc):
                return False
            raise ArchiveError(str(exc)) from exc
        return True
    finally:
        os.close(descriptor)
        shutil.rmtree(probe)


def select_scratch_base(
    *,
    environ: Mapping[str, str],
    localssd: Path = Path("/mnt/localssd"),
    system_temp: Path | None = None,
) -> Path:
    """Select the first configured scratch filesystem with safe publication."""
    candidates: list[tuple[str, str]] = []
    if environ.get("AGENTIC_SCRATCH_ROOT"):
        candidates.append(("AGENTIC_SCRATCH_ROOT", environ["AGENTIC_SCRATCH_ROOT"]))
    candidates.append(("local SSD", str(localssd)))
    if environ.get("TMPDIR"):
        candidates.append(("TMPDIR", environ["TMPDIR"]))
    candidates.append(("system temporary directory", str(system_temp or tempfile.gettempdir())))

    failures: list[str] = []
    for label, raw_path in candidates:
        if any(ord(character) < 32 or ord(character) == 127 for character in raw_path):
            if label == "AGENTIC_SCRATCH_ROOT":
                raise ArchiveError("AGENTIC_SCRATCH_ROOT must not contain control characters")
            failures.append(f"{label} contains control characters")
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            if label == "AGENTIC_SCRATCH_ROOT":
                raise ArchiveError("AGENTIC_SCRATCH_ROOT must be absolute")
            failures.append(f"{label} is not absolute")
            continue
        try:
            path.mkdir(parents=True, exist_ok=True)
            if atomic_no_replace_supported(path):
                return path.resolve()
            failures.append(f"{label} lacks atomic no-replace rename")
        except OSError as exc:
            failures.append(f"{label} is unavailable ({exc.strerror or type(exc).__name__})")
    raise ArchiveError("no compatible local scratch storage: " + "; ".join(failures))


def scratch_results_root(*, base: Path, repo_root: Path) -> Path:
    """Return a stable, owner-private results path shared by cluster nodes."""
    digest = _repo_digest(repo_root)
    private_root = Path(base).resolve() / f"agentic-scientist-{os.getuid()}"
    _ensure_private_directory(private_root)
    project_root = private_root / digest
    _ensure_private_directory(project_root)
    results_root = project_root / "results"
    _ensure_private_directory(results_root)
    return results_root


def validate_recorded_results_root(path: Path, *, repo_root: Path) -> Path:
    """Validate a marker-selected root against this repo's private namespace."""
    recorded = Path(path)
    expected_suffix = (
        f"agentic-scientist-{os.getuid()}",
        _repo_digest(repo_root),
        "results",
    )
    if not recorded.is_absolute() or recorded.parts[-3:] != expected_suffix:
        raise ArchiveError("archived path is outside this repository's scratch namespace")
    validated = scratch_results_root(base=recorded.parents[2], repo_root=repo_root)
    if validated != recorded:
        raise ArchiveError("archived scratch namespace is not canonical")
    return validated


def build_pipeline_command(
    command: Sequence[str],
    *,
    identity: RunIdentity,
    results_root: Path,
    resume: bool,
) -> list[str]:
    """Attach the one storage anchor required for a fresh run or resume."""
    built = [*command]
    if resume:
        built.extend(
            (
                "investigation.execution.resume=true",
                _path_override("paths.run_dir", identity.run_root(results_root)),
            )
        )
    else:
        built.append(_path_override("paths.results_root", results_root))
    return built


def _path_override(key: str, path: Path) -> str:
    return f"{key}={json.dumps(str(path))}"


def _repo_digest(repo_root: Path) -> str:
    return hashlib.sha256(str(Path(repo_root).resolve()).encode()).hexdigest()[:12]


def _ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise ArchiveError(f"storage path is not owner-private: {path}")
