"""Private Git storage and quarantined staging for run checkpoints."""

from __future__ import annotations

import logging
import os
import select
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .checkpoint_security import (
    SecretDetector,
    contains_secret,
    is_sensitive_path,
    redact_sensitive_text,
    report_sensitive_path,
)
from .checkpoint_paths import (
    descriptor_aliases,
    is_checkpoint_excluded_path,
    is_checkpoint_sidecar_path,
)

logger = logging.getLogger(__name__)


def _resolve_git_executable(path: str | None) -> str:
    executable = shutil.which("git", path=path)
    return str(Path(executable).resolve()) if executable is not None else "/usr/bin/git"


CHECKPOINT_IDENTITY_NAME = "Agentic Scientist Checkpoint"
CHECKPOINT_IDENTITY_EMAIL = "checkpoint@agentic-scientist.invalid"
_QUARANTINE_PREFIX = "agentic-checkpoint-quarantine-"
_SYSTEM_PATH = os.defpath
_GIT_EXECUTABLE = _resolve_git_executable(os.environ.get("PATH"))


class GitSettings(Protocol):
    @property
    def timeout_seconds(self) -> int: ...

    @property
    def _deadline(self) -> float | None: ...

    @property
    def max_file_bytes(self) -> int: ...

    @property
    def max_total_bytes(self) -> int: ...

    @property
    def final_include_environments(self) -> bool: ...

    @property
    def git_dir_name(self) -> str: ...


@dataclass(frozen=True)
class Quarantine:
    """Temporary index and object store whose rejected blobs are discarded."""

    index: Path
    objects: Path


@dataclass(frozen=True)
class StagedScan:
    """Credential and size exclusions discovered in the temporary index."""

    excluded: tuple[str, ...]
    reported: tuple[str, ...]
    oversized: tuple[str, ...]
    total_bytes: int


@dataclass(frozen=True)
class _IndexEntry:
    mode: str
    object_id: str


class _DeadlinePipeReader:
    def __init__(self, descriptor: int, *, deadline: float, command: list[str]) -> None:
        self._descriptor = descriptor
        self._deadline = deadline
        self._command = command
        self._buffer = bytearray()

    def line(self) -> bytes:
        while b"\n" not in self._buffer:
            self._fill()
        index = self._buffer.index(b"\n") + 1
        return self._take(index)

    def exact(self, size: int) -> bytes:
        while len(self._buffer) < size:
            self._fill()
        return self._take(size)

    def _fill(self) -> None:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(self._command, 0)
        readable, _, _ = select.select([self._descriptor], [], [], remaining)
        if not readable:
            raise subprocess.TimeoutExpired(self._command, 0)
        chunk = os.read(self._descriptor, 1024 * 1024)
        if not chunk:
            raise RuntimeError("git cat-file closed its output unexpectedly")
        self._buffer.extend(chunk)

    def _take(self, size: int) -> bytes:
        value = bytes(self._buffer[:size])
        del self._buffer[:size]
        return value


def git_environment() -> dict[str, str]:
    """Return the minimal deterministic environment allowed for Git children."""
    return {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": _SYSTEM_PATH,
    }


def base_argv(git_dir: Path, root: Path) -> list[str]:
    """Build a Git prefix that neutralizes ambient configuration sources."""
    return [
        _GIT_EXECUTABLE,
        f"--git-dir={git_dir}",
        f"--work-tree={root}",
        "-c",
        "core.excludesFile=",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "gc.auto=0",
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.safecrlf=false",
        "-c",
        "core.quotepath=false",
        "-c",
        f"user.name={CHECKPOINT_IDENTITY_NAME}",
        "-c",
        f"user.email={CHECKPOINT_IDENTITY_EMAIL}",
    ]


def run_git(
    git_dir: Path,
    root: Path,
    settings: GitSettings,
    *args: str,
    stamp: str | None = None,
    quarantine: Quarantine | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one Git command against the private sidecar."""
    argv = [*base_argv(git_dir, root), *args]
    logger.debug("git checkpoint command: %s", shlex.join(args))
    environment = git_environment()
    if quarantine is not None:
        environment.update(quarantine_environment(git_dir, quarantine))
    if stamp is not None:
        environment["GIT_AUTHOR_DATE"] = stamp
        environment["GIT_COMMITTER_DATE"] = stamp
    pass_fds = descriptor_aliases(git_dir) | descriptor_aliases(root)
    return subprocess.run(  # noqa: S603 - trusted argv, never shell
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=remaining_timeout_seconds(settings),
        cwd=str(root),
        env=environment,
        input=input_text,
        pass_fds=tuple(sorted(pass_fds)),
    )


def remaining_timeout_seconds(settings: GitSettings) -> float:
    """Return this command's share of one caller-owned checkpoint deadline."""
    deadline = settings._deadline
    if deadline is None:
        return float(settings.timeout_seconds)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired("git checkpoint", settings.timeout_seconds)
    return min(float(settings.timeout_seconds), remaining)


def quarantine_environment(git_dir: Path, quarantine: Quarantine) -> dict[str, str]:
    return {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": _quote_alternate_path(git_dir / "objects"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_INDEX_FILE": str(quarantine.index),
        "GIT_OBJECT_DIRECTORY": str(quarantine.objects),
        "GIT_TERMINAL_PROMPT": "0",
    }


@contextmanager
def quarantine_store(git_dir: Path) -> Iterator[Quarantine]:
    identity = _quarantine_identity(git_dir)
    prefix = f"{_QUARANTINE_PREFIX}{identity}-"
    _reap_stale_quarantines(Path(tempfile.gettempdir()), prefix=prefix)
    temporary = tempfile.TemporaryDirectory(prefix=prefix, ignore_cleanup_errors=True)
    try:
        raw = temporary.name
        directory = Path(raw)
        objects = directory / "objects"
        objects.mkdir(mode=0o700)
        yield Quarantine(index=directory / "index", objects=objects)
    finally:
        try:
            temporary.cleanup()
        except Exception as exc:  # cleanup must not rewrite an already-committed result
            logger.warning(
                "Checkpoint quarantine cleanup skipped: %s",
                redact_sensitive_text(str(exc)),
            )


def _quarantine_identity(git_dir: Path) -> str:
    parent = git_dir.parent.stat()
    identity_source = f"{parent.st_dev}:{parent.st_ino}:{git_dir.name}".encode()
    return hashlib.sha256(identity_source).hexdigest()[:12]


def _quote_alternate_path(path: Path) -> str:
    escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _reap_stale_quarantines(parent: Path, *, prefix: str) -> None:
    try:
        entries = list(os.scandir(parent))
    except OSError:
        return
    for entry in entries:
        if not entry.name.startswith(prefix):
            continue
        path = Path(entry.path)
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
            logger.warning("Refusing unsafe stale checkpoint quarantine: %s", path)
            continue
        shutil.rmtree(path)


def head_commit(git_dir: Path, root: Path, settings: GitSettings) -> str | None:
    head = run_git(git_dir, root, settings, "rev-parse", "--verify", "--quiet", "HEAD")
    if head.returncode == 0:
        return head.stdout.strip()
    if head.returncode == 1:
        return None
    raise RuntimeError(git_error("resolve HEAD", head))


def initialize_index(
    git_dir: Path,
    root: Path,
    settings: GitSettings,
    *,
    quarantine: Quarantine,
    parent: str | None,
) -> str | None:
    args = ("read-tree", parent) if parent is not None else ("read-tree", "--empty")
    initialized = run_git(git_dir, root, settings, *args, quarantine=quarantine)
    return git_error("read-tree", initialized) if initialized.returncode != 0 else None


def staged_paths(
    git_dir: Path,
    root: Path,
    settings: GitSettings,
    *,
    quarantine: Quarantine,
) -> tuple[str, ...]:
    proc = run_git(
        git_dir,
        root,
        settings,
        "diff",
        "--cached",
        "--name-only",
        "-z",
        quarantine=quarantine,
    )
    if proc.returncode != 0:
        raise RuntimeError(git_error("diff", proc))
    return tuple(path for path in proc.stdout.split("\0") if path)


def scan_staged_index(
    git_dir: Path,
    root: Path,
    settings: GitSettings,
    *,
    quarantine: Quarantine,
    final: bool,
) -> StagedScan:
    """Classify the candidate tree and inspect exact changed blobs."""
    entries = _index_entries(git_dir, root, settings, quarantine=quarantine)
    sensitive = {
        path
        for path, entry in entries.items()
        if is_sensitive_path(path)
        or is_checkpoint_sidecar_path(path, settings.git_dir_name)
        or is_checkpoint_excluded_path(path, settings=settings, final=final)
        or contains_secret((path.encode("utf-8", errors="surrogateescape"),))
        or entry.mode == "160000"
    }
    changed = set(staged_paths(git_dir, root, settings, quarantine=quarantine))
    candidate_entries = {path: entry for path, entry in entries.items() if path not in sensitive}
    for path, entry in candidate_entries.items():
        if entry.mode not in {"100644", "100755", "120000"}:
            raise RuntimeError(f"unsupported staged mode {entry.mode} for {path}")
    sizes = (
        _blob_sizes(
            git_dir,
            root,
            settings,
            quarantine=quarantine,
            object_ids={entry.object_id for entry in candidate_entries.values()},
        )
        if settings.max_file_bytes > 0 or settings.max_total_bytes > 0
        else {}
    )
    oversized = {
        path
        for path, entry in candidate_entries.items()
        if settings.max_file_bytes > 0 and sizes[entry.object_id] > settings.max_file_bytes
    }
    blobs: dict[str, list[str]] = {}
    for path in (candidate_entries.keys() & changed) - oversized:
        blobs.setdefault(candidate_entries[path].object_id, []).append(path)
    if not blobs:
        excluded = tuple(sorted(sensitive))
        return StagedScan(
            excluded=excluded,
            reported=tuple(report_sensitive_path(path) for path in excluded),
            oversized=tuple(sorted(oversized)),
            total_bytes=sum(
                sizes[entry.object_id]
                for path, entry in candidate_entries.items()
                if path not in oversized
            ) if settings.max_total_bytes > 0 else 0,
        )
    sensitive.update(
        _scan_blobs(
            git_dir,
            root,
            settings,
            quarantine=quarantine,
            blobs=blobs,
        )
    )
    excluded = tuple(sorted(sensitive))
    return StagedScan(
        excluded=excluded,
        reported=tuple(report_sensitive_path(path) for path in excluded),
        oversized=tuple(sorted(oversized)),
        total_bytes=sum(
            sizes[entry.object_id]
            for path, entry in candidate_entries.items()
            if path not in sensitive and path not in oversized
        ) if settings.max_total_bytes > 0 else 0,
    )


def _blob_sizes(
    git_dir: Path,
    root: Path,
    settings: GitSettings,
    *,
    quarantine: Quarantine,
    object_ids: set[str],
) -> dict[str, int]:
    if not object_ids:
        return {}
    ordered = sorted(object_ids)
    checked = run_git(
        git_dir,
        root,
        settings,
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        quarantine=quarantine,
        input_text="".join(f"{object_id}\n" for object_id in ordered),
    )
    if checked.returncode != 0:
        raise RuntimeError(git_error("cat-file sizes", checked))
    sizes: dict[str, int] = {}
    for line in checked.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3 or fields[1] != "blob":
            raise RuntimeError(f"git cat-file returned invalid size record: {line[:100]}")
        sizes[fields[0]] = int(fields[2])
    if sizes.keys() != set(ordered):
        raise RuntimeError("git cat-file omitted staged blob sizes")
    return sizes


def _index_entries(
    git_dir: Path,
    root: Path,
    settings: GitSettings,
    *,
    quarantine: Quarantine,
) -> dict[str, _IndexEntry]:
    listed = run_git(git_dir, root, settings, "ls-files", "--stage", "-z", quarantine=quarantine)
    if listed.returncode != 0:
        raise RuntimeError(git_error("ls-files", listed))
    entries: dict[str, _IndexEntry] = {}
    for record in listed.stdout.split("\0"):
        if not record:
            continue
        metadata, path = record.split("\t", 1)
        _mode, object_id, stage = metadata.split()
        if stage == "0":
            entries[path] = _IndexEntry(mode=_mode, object_id=object_id)
    return entries


def _scan_blobs(
    git_dir: Path,
    root: Path,
    settings: GitSettings,
    *,
    quarantine: Quarantine,
    blobs: dict[str, list[str]],
) -> set[str]:
    remaining_timeout_seconds(settings)
    argv = [*base_argv(git_dir, root), "cat-file", "--batch"]
    environment = git_environment()
    environment.update(quarantine_environment(git_dir, quarantine))
    process = subprocess.Popen(  # noqa: S603 - trusted argv, never shell
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(root),
        env=environment,
        pass_fds=tuple(sorted(descriptor_aliases(git_dir) | descriptor_aliases(root))),
    )
    deadline = settings._deadline or time.monotonic() + settings.timeout_seconds
    sensitive: set[str] = set()
    try:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("git cat-file pipes unavailable")
        reader = _DeadlinePipeReader(process.stdout.fileno(), deadline=deadline, command=argv)
        for object_id, paths in blobs.items():
            process.stdin.write(f"{object_id}\n".encode())
            process.stdin.flush()
            header = reader.line().decode("ascii", errors="replace").split()
            if len(header) != 3 or header[1] != "blob":
                raise RuntimeError(f"git cat-file returned invalid header for {object_id}")
            remaining = int(header[2])
            detector = SecretDetector()
            found = False
            while remaining:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(argv, settings.timeout_seconds)
                chunk = reader.exact(min(1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError(f"git cat-file truncated blob {object_id}")
                remaining -= len(chunk)
                found = detector.feed(chunk) or found
            if reader.exact(1) != b"\n":
                raise RuntimeError(f"git cat-file malformed blob {object_id}")
            if found:
                sensitive.update(paths)
        process.stdin.close()
        process.wait(timeout=max(0.001, deadline - time.monotonic()))
        if process.returncode != 0:
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            raise RuntimeError(f"git cat-file failed: {stderr.strip()[:200]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return sensitive


def create_quarantined_commit(
    git_dir: Path,
    root: Path,
    settings: GitSettings,
    *,
    quarantine: Quarantine,
    parent: str | None,
    message: str,
    stamp: str,
) -> tuple[str | None, str | None]:
    tree = run_git(git_dir, root, settings, "write-tree", quarantine=quarantine)
    if tree.returncode != 0:
        return None, git_error("write-tree", tree)
    args = ["commit-tree", tree.stdout.strip()]
    if parent is not None:
        args.extend(("-p", parent))
    committed = run_git(
        git_dir,
        root,
        settings,
        *args,
        stamp=stamp,
        quarantine=quarantine,
        input_text=message,
    )
    if committed.returncode != 0:
        return None, git_error("commit-tree", committed)
    return committed.stdout.strip(), None


def promote_commit(
    git_dir: Path,
    root: Path,
    settings: GitSettings,
    *,
    quarantine: Quarantine,
    commit: str,
    parent: str | None,
) -> str | None:
    revision_args = ["rev-list", "--objects", commit]
    if parent is not None:
        revision_args.append(f"^{parent}")
    reachable = run_git(git_dir, root, settings, *revision_args, quarantine=quarantine)
    if reachable.returncode != 0:
        return git_error("rev-list", reachable)
    created: list[Path] = []
    try:
        for line in reachable.stdout.splitlines():
            remaining_timeout_seconds(settings)
            object_id = line.split(" ", 1)[0]
            source = quarantine.objects / object_id[:2] / object_id[2:]
            if not source.is_file():
                continue
            destination = git_dir / "objects" / object_id[:2] / object_id[2:]
            destination.parent.mkdir(exist_ok=True)
            if _promote_object(source, destination, settings=settings):
                created.append(destination)
    except BaseException:
        for destination in reversed(created):
            destination.unlink(missing_ok=True)
        raise
    update_args = ["update-ref", "HEAD", commit]
    if parent is not None:
        update_args.append(parent)
    updated = run_git(git_dir, root, settings, *update_args)
    if updated.returncode != 0:
        return git_error("update-ref", updated)
    try:
        refreshed = run_git(git_dir, root, settings, "read-tree", commit)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Checkpoint index refresh skipped: %s", redact_sensitive_text(str(exc)))
    else:
        if refreshed.returncode != 0:
            logger.warning(
                "Checkpoint index refresh skipped: %s", git_error("read-tree", refreshed)
            )
    return None


def _promote_object(source: Path, destination: Path, *, settings: GitSettings) -> bool:
    try:
        os.link(source, destination)
        return True
    except FileExistsError:
        return False
    except OSError:
        pass
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            prefix=f".{destination.name}.promote-",
            dir=destination.parent,
            delete=False,
        ) as target, source.open("rb") as source_stream:
            temporary_path = Path(target.name)
            while chunk := source_stream.read(16 * 1024 * 1024):
                remaining_timeout_seconds(settings)
                target.write(chunk)
        remaining_timeout_seconds(settings)
        try:
            os.link(temporary_path, destination)
            return True
        except FileExistsError:
            return False
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def commit_count(git_dir: Path, root: Path, settings: GitSettings) -> int:
    proc = run_git(git_dir, root, settings, "rev-list", "--count", "HEAD")
    if proc.returncode != 0:
        return 0
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return 0


def git_error(stage: str, proc: subprocess.CompletedProcess[str]) -> str:
    detail = (proc.stderr or "").strip() or (proc.stdout or "").strip() or "no output"
    detail = redact_sensitive_text(detail)
    return f"git {stage} failed (rc={proc.returncode}): {detail.splitlines()[0][:200]}"


def ensure_repository(git_dir: Path, *, root: Path, settings: GitSettings) -> None:
    """Initialize the sidecar without persisting an absolute work-tree path."""
    if (git_dir / "HEAD").is_file():
        return
    initialized = subprocess.run(  # noqa: S603 - trusted argv, never shell
        [_GIT_EXECUTABLE, f"--git-dir={git_dir}", "init", "-q"],
        capture_output=True,
        text=True,
        check=False,
        timeout=remaining_timeout_seconds(settings),
        cwd=str(root),
        env=git_environment(),
        pass_fds=tuple(sorted(descriptor_aliases(git_dir) | descriptor_aliases(root))),
    )
    if initialized.returncode != 0:
        raise RuntimeError(git_error("init", initialized))
    run_git(git_dir, root, settings, "config", "core.bare", "false")
