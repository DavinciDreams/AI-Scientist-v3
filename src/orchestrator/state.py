"""Persistent state and history helpers for campaign execution."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import stat
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, TypedDict

from src.investigation.program_spec import entrypoint_relpath_from_spec

logger = logging.getLogger(__name__)

OUTER_STATE_FILE = "outer_state.json"
IDEAS_HISTORY_FILE = "ideas_history.jsonl"
ATTEMPTS_HISTORY_FILE = "attempts_history.jsonl"
IDEA_GENERATION_FAILURES_FILE = "idea_generation_failures.jsonl"
BEST_MANIFEST_FILE = "best_program_manifest.json"
PARALLEL_RESUME_STATE_FILE = "parallel_resume_state.json"
GLOBAL_MEMORY_MD_FILE = "memory.md"
GLOBAL_MEMORY_JSONL_FILE = "memory.jsonl"
MEMORY_LOCK_FILE = ".memory.lock"
LEGACY_GLOBAL_MEMORY_MD_FILE = "global_memory.md"
LEGACY_GLOBAL_MEMORY_JSONL_FILES = ("memory.json", "global_memory.jsonl")
CONCLUSIONS_FILE = "conclusions.md"
OTHER_WORKERS_CONCLUSIONS_FILE = "conclusions_other_workers.md"
CONCLUSIONS_HISTORY_FILE = "conclusions_history.jsonl"
SHARED_CONCLUSIONS_SUBDIR = "conclusions"
EMPTY_CONCLUSIONS_PLACEHOLDER = "_No conclusions yet._\n"
EMPTY_OTHER_WORKERS_CONCLUSIONS_PLACEHOLDER = "_No conclusions from other workers yet._\n"
IDEA_ID_INDEX_RE = re.compile(r"(\d+)$")
ATTEMPT_ID_INDEX_RE = re.compile(r"(?:^|_)attempt_(\d+)$")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
MAX_JSONL_DEDUP_SCAN_BYTES = 64 * 1024 * 1024
JSONL_SCAN_CHUNK_BYTES = 64 * 1024


class ConclusionsSyncState(TypedDict, total=False):
    """Snapshot used to decide whether a worker's conclusions changed."""

    own_conclusions_size: int
    own_conclusions_sha256: str


def utc_now() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any | None:
    """Load json."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, data: Any) -> None:
    """Write JSON data to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_private_text(path: Path, text: str) -> None:
    """Atomically write owner-only text without following the destination."""
    directory_fd = _create_memory_directory_no_follow(path.parent)
    try:
        _atomic_replace_regular_file_at(
            directory_fd,
            path.name,
            text.encode("utf-8"),
            0o600,
        )
    finally:
        os.close(directory_fd)


def append_jsonl(path: Path, data: dict[str, Any], *, mode: int = 0o666) -> None:
    """Append one JSON object as a JSONL record."""
    line = (json.dumps(data, sort_keys=True) + "\n").encode("utf-8")
    fd = _open_regular_for_append(path, mode=mode)
    try:
        if mode != 0o666:
            os.fchmod(fd, mode)
        fcntl.flock(fd, fcntl.LOCK_EX)
        _append_jsonl_line_locked(fd, line)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def append_jsonl_if_absent(
    path: Path,
    data: dict[str, Any],
    *,
    matching: dict[str, object],
) -> bool:
    """Append a JSONL record unless one matching all requested fields exists."""
    line = (json.dumps(data, sort_keys=True) + "\n").encode("utf-8")
    fd = _open_jsonl_for_append(path)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        size = os.fstat(fd).st_size
        if size > MAX_JSONL_DEDUP_SCAN_BYTES:
            raise OSError(
                errno.EFBIG,
                f"JSONL file exceeds safe deduplication scan limit: {path}",
            )
        if _jsonl_fd_contains_match(fd, size=size, matching=matching):
            return False
        _append_jsonl_line_locked(fd, line)
        return True
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _jsonl_fd_contains_match(
    fd: int,
    *,
    size: int,
    matching: dict[str, object],
) -> bool:
    """Incrementally scan a bounded JSONL file for one matching record."""
    offset = 0
    pending = b""
    while offset < size:
        chunk = os.pread(fd, min(JSONL_SCAN_CHUNK_BYTES, size - offset), offset)
        if not chunk:
            break
        offset += len(chunk)
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        if _jsonl_lines_contain_match(lines, matching=matching):
            return True
    return _jsonl_lines_contain_match([pending], matching=matching)


def _jsonl_lines_contain_match(
    lines: list[bytes],
    *,
    matching: dict[str, object],
) -> bool:
    """Return whether any parseable JSON object matches all requested fields."""
    for raw in lines:
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(parsed, dict) and all(
            parsed.get(key) == value for key, value in matching.items()
        ):
            return True
    return False


def append_memory_markdown_entry_if_absent(
    path: Path,
    event: dict[str, Any],
    *,
    matching_line: str,
) -> bool:
    """Safely append a campaign-memory projection unless its marker exists."""
    entry = format_memory_markdown_entry(event).encode("utf-8")
    fd = _open_regular_for_append(path, mode=0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        size = os.fstat(fd).st_size
        if size > MAX_JSONL_DEDUP_SCAN_BYTES:
            raise OSError(
                errno.EFBIG,
                f"Markdown file exceeds safe deduplication scan limit: {path}",
            )
        if _fd_contains_markdown_entry(
            fd,
            size=size,
            step=str(event.get("step", "")),
            scope=str(event.get("scope", "")),
            matching_line=matching_line,
        ):
            return False
        if not size:
            _write_all(fd, b"# Global Memory Log\n\n")
        elif os.pread(fd, 1, size - 1) != b"\n":
            _write_all(fd, b"\n")
        _write_all(fd, entry)
        return True
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _fd_contains_markdown_entry(
    fd: int,
    *,
    size: int,
    step: str,
    scope: str,
    matching_line: str,
) -> bool:
    """Find a marker line inside a markdown entry with matching step and scope."""
    if not step or not scope or not matching_line:
        raise ValueError("matching line must not be empty")
    header_marker = f"| step={step} | scope={scope}".encode("utf-8")
    marker = matching_line.encode("utf-8")
    offset = 0
    pending = b""
    header_matches = False
    marker_matches = False

    def consume(candidate: bytes) -> bool:
        nonlocal header_matches, marker_matches
        normalized = candidate.rstrip(b"\r")
        if normalized.startswith(b"### "):
            if header_matches and marker_matches:
                return True
            header_matches = normalized.endswith(header_marker)
            marker_matches = False
        elif normalized == marker:
            marker_matches = True
        return False

    while offset < size:
        chunk = os.pread(fd, min(JSONL_SCAN_CHUNK_BYTES, size - offset), offset)
        if not chunk:
            break
        offset += len(chunk)
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        for candidate in lines:
            if consume(candidate):
                return True
    consume(pending)
    return header_matches and marker_matches


def _open_jsonl_for_append(path: Path) -> int:
    """Open a JSONL file through a pinned directory without following links."""
    return _open_regular_for_append(path, mode=0o666)


def _open_regular_for_append(path: Path, *, mode: int) -> int:
    """Open a regular append target through a pinned directory."""
    directory_fd = _create_memory_directory_no_follow(path.parent)
    try:
        return _open_regular_file_at(
            directory_fd,
            path.name,
            os.O_APPEND | os.O_CREAT | os.O_RDWR,
            mode=mode,
        )
    finally:
        os.close(directory_fd)


def _append_jsonl_line_locked(fd: int, line: bytes) -> None:
    """Append one line while the caller holds the file lock."""
    size = os.fstat(fd).st_size
    if size and os.pread(fd, 1, size - 1) != b"\n":
        _write_all(fd, b"\n")
    _write_all(fd, line)


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte or propagate the write failure."""
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError(errno.EIO, "write returned no progress")
        remaining = remaining[written:]


def tail_jsonl(path: Path, max_lines: int = 20) -> list[dict[str, Any]]:
    """Handle tail jsonl logic for this module."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-max_lines:]:
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load all parseable JSONL object records from disk."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def format_memory_markdown_entry(event: dict[str, Any]) -> str:
    """Render one global-memory markdown entry from a structured event."""
    timestamp = str(event.get("timestamp", "")).strip() or "unknown_ts"
    step = str(event.get("step", "")).strip() or "unknown_step"
    scope = str(event.get("scope", "")).strip() or "unknown_scope"
    summary = str(event.get("summary", "")).strip() or "(missing summary)"
    evidence = str(event.get("evidence", "")).strip() or "(missing evidence)"
    next_action = str(event.get("next_action", "")).strip() or "(missing next_action)"
    lines = [
        f"### {timestamp} | step={step} | scope={scope}",
        f"- summary: {summary}",
        f"- evidence: {evidence}",
        f"- next_action: {next_action}",
    ]
    for key in ("idea_id", "attempt_id", "program_id"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"- {key}: {value.strip()}")
    metrics = event.get("metrics")
    if isinstance(metrics, dict) and metrics:
        lines.append(f"- metrics: {json.dumps(metrics, sort_keys=True)}")
    rubrics = event.get("rubrics")
    if isinstance(rubrics, list) and rubrics:
        lines.append(f"- rubrics: {json.dumps(rubrics, sort_keys=True)}")
    tags = event.get("tags")
    if isinstance(tags, list) and tags:
        lines.append(f"- tags: {', '.join(str(item) for item in tags)}")
    return "\n".join(lines) + "\n\n"


def _global_memory_md_has_entries(path: Path) -> bool:
    """Return whether memory.md contains content beyond the default header."""
    if not path.exists():
        return False
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return False
    return not (len(lines) == 1 and lines[0] == "# Global Memory Log")


def _backfill_global_memory_md_from_jsonl(global_md: Path, global_json: Path) -> None:
    """Populate memory.md from JSONL when markdown is still header-only."""
    if _global_memory_md_has_entries(global_md):
        return
    events = _load_jsonl_records(global_json)
    if not events:
        return
    rendered = "# Global Memory Log\n\n" + "".join(
        format_memory_markdown_entry(event) for event in events
    )
    global_md.write_text(rendered, encoding="utf-8")


def ensure_global_memory_files(memory_dir: Path) -> None:
    """Ensure global memory files."""
    memory_dir.mkdir(parents=True, exist_ok=True)
    global_md = memory_dir / GLOBAL_MEMORY_MD_FILE
    global_json = memory_dir / GLOBAL_MEMORY_JSONL_FILE
    legacy_md = memory_dir / LEGACY_GLOBAL_MEMORY_MD_FILE

    if not global_md.exists() and legacy_md.exists():
        global_md.write_text(legacy_md.read_text(encoding="utf-8"), encoding="utf-8")
    if not global_json.exists():
        for legacy_name in LEGACY_GLOBAL_MEMORY_JSONL_FILES:
            legacy_json = memory_dir / legacy_name
            if legacy_json.exists():
                global_json.write_text(legacy_json.read_text(encoding="utf-8"), encoding="utf-8")
                break

    if not global_md.exists():
        global_md.write_text("# Global Memory Log\n\n", encoding="utf-8")
    if not global_json.exists():
        global_json.write_text("", encoding="utf-8")
    _backfill_global_memory_md_from_jsonl(global_md, global_json)


def ensure_histories(results_dir: Path, *, include_global_memory: bool = True) -> None:
    """Ensure histories."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for rel in (
        IDEAS_HISTORY_FILE,
        ATTEMPTS_HISTORY_FILE,
        IDEA_GENERATION_FAILURES_FILE,
    ):
        path = results_dir / rel
        if not path.exists():
            path.write_text("", encoding="utf-8")
    if include_global_memory:
        ensure_global_memory_files(results_dir)


def load_or_init_outer_state(
    results_dir: Path,
    seed_program_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """Load or init outer state."""
    state_path = results_dir / OUTER_STATE_FILE
    if resume:
        loaded = load_json(state_path)
        if isinstance(loaded, dict):
            return loaded

    now = utc_now()
    state: dict[str, Any] = {
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "current_cycle": 0,
        "failure_streak": 0,
        "last_idea_id": None,
        "best_program_dir": str(seed_program_dir),
        "best_combined_score": None,
        "best_correct": None,
        "prune_chapter": 0,
    }
    write_json(state_path, state)
    return state


def save_outer_state(results_dir: Path, state: dict[str, Any]) -> None:
    """Save outer state."""
    state["updated_at"] = utc_now()
    write_json(results_dir / OUTER_STATE_FILE, state)


def load_or_init_best_manifest(
    results_dir: Path,
    seed_program_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """Load or init best manifest."""
    manifest_path = results_dir / BEST_MANIFEST_FILE
    if resume:
        loaded = load_json(manifest_path)
        if isinstance(loaded, dict):
            return loaded

    entrypoint_path = entrypoint_relpath_from_spec(seed_program_dir, required=False) or "src/program.py"
    manifest = {
        "program_dir": str(seed_program_dir),
        "entrypoint_path": entrypoint_path,
        "combined_score": None,
        "correct": None,
        "source_attempt_id": "bootstrap",
        "updated_at": utc_now(),
    }
    write_json(manifest_path, manifest)
    return manifest


def save_best_manifest(results_dir: Path, manifest: dict[str, Any]) -> None:
    """Save best manifest."""
    manifest["updated_at"] = utc_now()
    write_json(results_dir / BEST_MANIFEST_FILE, manifest)


def parallel_resume_state_path(results_dir: Path) -> Path:
    """Return the persisted parallel resume-state path."""
    return results_dir / PARALLEL_RESUME_STATE_FILE


def load_parallel_resume_state(results_dir: Path) -> dict[str, Any] | None:
    """Load persisted parallel resume-state if present."""
    loaded = load_json(parallel_resume_state_path(results_dir))
    return loaded if isinstance(loaded, dict) else None


def save_parallel_resume_state(results_dir: Path, state: dict[str, Any]) -> None:
    """Persist parallel resume-state."""
    payload = dict(state)
    payload["updated_at"] = utc_now()
    write_json(parallel_resume_state_path(results_dir), payload)


def ideas_history_path(results_dir: Path) -> Path:
    """Return the ideas history path."""
    return results_dir / IDEAS_HISTORY_FILE


def attempts_history_path(results_dir: Path) -> Path:
    """Return the attempts history path."""
    return results_dir / ATTEMPTS_HISTORY_FILE


def idea_generation_failures_path(results_dir: Path) -> Path:
    """Return the idea-generation failure history path."""
    return results_dir / IDEA_GENERATION_FAILURES_FILE


def global_memory_md_path(results_dir: Path) -> Path:
    """Return the global memory md path."""
    return results_dir / GLOBAL_MEMORY_MD_FILE


def global_memory_jsonl_path(results_dir: Path) -> Path:
    """Return the global memory jsonl path."""
    return results_dir / GLOBAL_MEMORY_JSONL_FILE


def conclusions_md_path(conclusions_dir: Path) -> Path:
    """Return the path of the worker's own distilled conclusions file."""
    return conclusions_dir / CONCLUSIONS_FILE


def other_workers_conclusions_md_path(conclusions_dir: Path) -> Path:
    """Return the path of the file holding other workers' concatenated conclusions."""
    return conclusions_dir / OTHER_WORKERS_CONCLUSIONS_FILE


def conclusions_history_path(conclusions_dir: Path) -> Path:
    """Return the path of the conclusions history JSONL."""
    return conclusions_dir / CONCLUSIONS_HISTORY_FILE


def shared_conclusions_dir(shared_memory_dir: Path) -> Path:
    """Return the cross-worker conclusions subdirectory."""
    return shared_memory_dir / SHARED_CONCLUSIONS_SUBDIR


def shared_worker_conclusions_path(shared_memory_dir: Path, worker_index: int) -> Path:
    """Return the per-worker file under the cross-worker conclusions subdirectory."""
    return shared_conclusions_dir(shared_memory_dir) / f"worker_{worker_index:02d}.md"


def ensure_conclusions_files(conclusions_dir: Path) -> None:
    """Ensure the worker-local conclusion files exist with placeholder content.

    Creates ``conclusions.md`` and ``conclusions_other_workers.md`` with
    placeholder bodies if they are missing. Conclusions history JSONL is left
    absent until the first write so its presence indicates real activity.
    """
    conclusions_dir.mkdir(parents=True, exist_ok=True)
    own = conclusions_md_path(conclusions_dir)
    other = other_workers_conclusions_md_path(conclusions_dir)
    if not own.exists():
        own.write_text(EMPTY_CONCLUSIONS_PLACEHOLDER, encoding="utf-8")
    if not other.exists():
        other.write_text(EMPTY_OTHER_WORKERS_CONCLUSIONS_PLACEHOLDER, encoding="utf-8")


def _slugify_component(text: str | None, fallback: str = "idea") -> str:
    """Handle slugify component logic for this module."""
    candidate = text or ""
    slug = NON_ALNUM_RE.sub("_", candidate.strip().lower()).strip("_")
    if slug:
        return slug
    fallback_slug = NON_ALNUM_RE.sub("_", fallback.strip().lower()).strip("_")
    return fallback_slug or "idea"


def idea_index_from_id(idea_id: str) -> int | None:
    """Handle idea index from id logic for this module."""
    match = IDEA_ID_INDEX_RE.search(idea_id)
    if not match:
        return None
    try:
        idx = int(match.group(1))
    except ValueError:
        return None
    return idx if idx > 0 else None


def attempt_index_from_id(attempt_id: str) -> int | None:
    """Handle attempt index from id logic for this module."""
    match = ATTEMPT_ID_INDEX_RE.search(attempt_id)
    if not match:
        return None
    try:
        idx = int(match.group(1))
    except ValueError:
        return None
    return idx if idx > 0 else None


def idea_dir_name(
    *,
    cycle_index: int | None,
    idea_title: str | None,
    idea_id: str,
) -> str:
    """Handle idea dir name logic for this module."""
    idx = cycle_index if isinstance(cycle_index, int) and cycle_index > 0 else idea_index_from_id(idea_id)
    if idx is None:
        idx = 1
    label = _slugify_component(idea_title, fallback=idea_id)
    return f"{idx:02d}_{label}"


def attempt_dir_name(*, attempt_number: int | None, attempt_id: str) -> str:
    """Handle attempt dir name logic for this module."""
    idx = (
        attempt_number
        if isinstance(attempt_number, int) and attempt_number > 0
        else attempt_index_from_id(attempt_id)
    )
    if idx is None:
        idx = 1
    return f"{idx:02d}_attempt"


def idea_dir(results_dir: Path, idea_dir_name: str) -> Path:
    """Handle idea dir logic for this module."""
    return results_dir / "ideas" / idea_dir_name


def attempt_dir(results_dir: Path, idea_dir_name: str, attempt_dir_name: str) -> Path:
    """Handle attempt dir logic for this module."""
    return idea_dir(results_dir, idea_dir_name) / attempt_dir_name


def archive_for_prune(
    results_dir: Path,
    global_memory_dir: Path,
    chapter: int,
) -> None:
    """Archive current memory and history files, then reset them for a new chapter.

    Called at the start of a prune boundary cycle.  Copies the live history
    and memory files into ``_pruned/chapter_NNN/`` and resets them so the
    ideation agent starts with a clean slate while keeping the best program.
    """
    archive_dir = results_dir / "_pruned" / f"chapter_{chapter:03d}"
    new_chapter = chapter + 1
    memory_content = (
        f"# Global Memory Log (Chapter {new_chapter})\n\n"
        f"_Pruned at cycle boundary. Previous context archived in _pruned/chapter_{chapter:03d}/_\n\n"
    ).encode("utf-8")
    prune_entry = {
        "timestamp": utc_now(),
        "step": "prune",
        "scope": "global",
        "summary": f"Idea pruning: reset context for chapter {new_chapter}. Previous context archived.",
        "chapter": new_chapter,
    }
    with ExitStack() as directory_stack:
        archive_directory_fd = _create_memory_directory_no_follow(archive_dir)
        directory_stack.callback(os.close, archive_directory_fd)
        results_directory_fd = _open_memory_directory_no_follow(results_dir)
        directory_stack.callback(os.close, results_directory_fd)
        memory_directory_fd = _open_memory_directory_no_follow(global_memory_dir)
        directory_stack.callback(os.close, memory_directory_fd)
        for name in (
            IDEAS_HISTORY_FILE,
            ATTEMPTS_HISTORY_FILE,
            IDEA_GENERATION_FAILURES_FILE,
        ):
            source_exists = _entry_exists_at(results_directory_fd, name)
            source_stat = _archive_regular_file_at(
                results_directory_fd,
                name,
                archive_directory_fd,
            )
            if source_exists:
                _replace_file_content_at(results_directory_fd, name, b"", source_stat)

        lock_fd = _open_regular_file_at(
            memory_directory_fd,
            MEMORY_LOCK_FILE,
            os.O_RDWR | os.O_CREAT,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                memory_stats = {
                    name: _archive_regular_file_at(
                        memory_directory_fd,
                        name,
                        archive_directory_fd,
                    )
                    for name in (GLOBAL_MEMORY_MD_FILE, GLOBAL_MEMORY_JSONL_FILE)
                }
                _replace_file_content_at(
                    memory_directory_fd,
                    GLOBAL_MEMORY_MD_FILE,
                    memory_content,
                    memory_stats[GLOBAL_MEMORY_MD_FILE],
                )
                _replace_file_content_at(
                    memory_directory_fd,
                    GLOBAL_MEMORY_JSONL_FILE,
                    (json.dumps(prune_entry, sort_keys=True) + "\n").encode("utf-8"),
                    memory_stats[GLOBAL_MEMORY_JSONL_FILE],
                )
                _archive_and_reset_conclusions_at(
                    memory_directory_fd,
                    archive_directory_fd,
                )
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def archive_for_parallel_prune(
    results_dir: Path,
    global_memory_dir: Path,
    round_index: int,
) -> None:
    """Archive memory/history and reset for the next parallel round.

    Similar to ``archive_for_prune`` but designed for multi-worker rounds.
    Archives merged histories + memory to ``_pruned/round_NNN/``, then resets
    ``memory.md`` and ``memory.jsonl`` with a prune marker.
    """
    archive_dir = results_dir / "_pruned" / f"round_{round_index:03d}"
    with ExitStack() as directory_stack:
        archive_directory_fd = _create_memory_directory_no_follow(archive_dir)
        directory_stack.callback(os.close, archive_directory_fd)
        results_directory_fd = _open_memory_directory_no_follow(results_dir)
        directory_stack.callback(os.close, results_directory_fd)
        memory_directory_fd = _open_memory_directory_no_follow(global_memory_dir)
        directory_stack.callback(os.close, memory_directory_fd)
        for name in (
            IDEAS_HISTORY_FILE,
            ATTEMPTS_HISTORY_FILE,
            IDEA_GENERATION_FAILURES_FILE,
        ):
            source_exists = _entry_exists_at(results_directory_fd, name)
            source_stat = _archive_regular_file_at(
                results_directory_fd,
                name,
                archive_directory_fd,
            )
            if source_exists:
                _replace_file_content_at(results_directory_fd, name, b"", source_stat)

        for name in (OUTER_STATE_FILE, BEST_MANIFEST_FILE, "parallel_summary.json"):
            _archive_regular_file_at(results_directory_fd, name, archive_directory_fd)

        lock_fd = _open_regular_file_at(
            memory_directory_fd,
            MEMORY_LOCK_FILE,
            os.O_RDWR | os.O_CREAT,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                for name in (GLOBAL_MEMORY_MD_FILE, GLOBAL_MEMORY_JSONL_FILE):
                    _archive_regular_file_at(
                        memory_directory_fd,
                        name,
                        archive_directory_fd,
                    )
                _archive_and_reset_conclusions_at(
                    memory_directory_fd,
                    archive_directory_fd,
                )
                _reset_parallel_global_memory_for_round_at(
                    memory_directory_fd,
                    round_index=round_index,
                )
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _archive_and_reset_conclusions(conclusions_dir: Path, archive_dir: Path) -> None:
    """Archive a worker's conclusion files into ``archive_dir`` and reset them.

    Copies ``conclusions.md``, ``conclusions_history.jsonl`` and
    ``conclusions_other_workers.md`` (when present) under
    ``archive_dir/conclusions/`` and then resets the live files to their
    empty/placeholder state. ``conclusions_history.jsonl`` is removed (rather
    than emptied) so its absence again signals "no real activity yet".
    """
    with ExitStack() as directory_stack:
        conclusions_directory_fd = _open_memory_directory_no_follow(conclusions_dir)
        directory_stack.callback(os.close, conclusions_directory_fd)
        archive_directory_fd = _create_memory_directory_no_follow(archive_dir)
        directory_stack.callback(os.close, archive_directory_fd)
        _archive_and_reset_conclusions_at(
            conclusions_directory_fd,
            archive_directory_fd,
        )


def _archive_and_reset_conclusions_at(
    conclusions_directory_fd: int,
    archive_directory_fd: int,
) -> None:
    """Archive and reset conclusion entries through pinned directories."""
    archive_subdir_fd = _open_child_directory_at(
        archive_directory_fd,
        SHARED_CONCLUSIONS_SUBDIR,
        create=True,
    )
    assert archive_subdir_fd is not None
    try:
        own_stat = _archive_regular_file_at(
            conclusions_directory_fd,
            CONCLUSIONS_FILE,
            archive_subdir_fd,
        )
        history_stat = _archive_regular_file_at(
            conclusions_directory_fd,
            CONCLUSIONS_HISTORY_FILE,
            archive_subdir_fd,
        )
        other_stat = _archive_regular_file_at(
            conclusions_directory_fd,
            OTHER_WORKERS_CONCLUSIONS_FILE,
            archive_subdir_fd,
        )
        _replace_file_content_at(
            conclusions_directory_fd,
            CONCLUSIONS_FILE,
            EMPTY_CONCLUSIONS_PLACEHOLDER.encode("utf-8"),
            own_stat,
        )
        _replace_file_content_at(
            conclusions_directory_fd,
            OTHER_WORKERS_CONCLUSIONS_FILE,
            EMPTY_OTHER_WORKERS_CONCLUSIONS_PLACEHOLDER.encode("utf-8"),
            other_stat,
        )
        if history_stat is not None:
            _unlink_regular_file_if_unchanged_at(
                conclusions_directory_fd,
                CONCLUSIONS_HISTORY_FILE,
                history_stat,
            )
        else:
            _unlink_non_directory_entry_at(
                conclusions_directory_fd,
                CONCLUSIONS_HISTORY_FILE,
            )
    finally:
        os.close(archive_subdir_fd)


def archive_shared_conclusions_for_prune(
    shared_memory_dir: Path,
    archive_dir: Path,
) -> None:
    """Archive the cross-worker shared conclusions subdir and clear it.

    Should be called once per prune boundary by whichever process owns the
    archive step (campaign loop in solo mode, parallel runner in parallel
    mode). Idempotent if the shared dir is missing.
    """
    try:
        shared_directory_fd = _open_memory_directory_no_follow(shared_memory_dir)
    except FileNotFoundError:
        return
    with ExitStack() as directory_stack:
        directory_stack.callback(os.close, shared_directory_fd)
        archive_directory_fd = _create_memory_directory_no_follow(archive_dir)
        directory_stack.callback(os.close, archive_directory_fd)
        archive_subdir_fd = _open_child_directory_at(
            archive_directory_fd,
            f"shared_{SHARED_CONCLUSIONS_SUBDIR}",
            create=True,
        )
        assert archive_subdir_fd is not None
        directory_stack.callback(os.close, archive_subdir_fd)
        lock_fd = _open_regular_file_at(
            shared_directory_fd,
            MEMORY_LOCK_FILE,
            os.O_RDWR | os.O_CREAT,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                source_subdir_fd = _open_conclusions_directory_at(
                    shared_directory_fd,
                    create=False,
                )
                if source_subdir_fd is None:
                    return
                try:
                    for name in sorted(os.listdir(source_subdir_fd)):
                        source_stat = _archive_regular_file_at(
                            source_subdir_fd,
                            name,
                            archive_subdir_fd,
                        )
                        if source_stat is not None:
                            _unlink_regular_file_if_unchanged_at(
                                source_subdir_fd,
                                name,
                                source_stat,
                            )
                        else:
                            _unlink_non_directory_entry_at(source_subdir_fd, name)
                finally:
                    os.close(source_subdir_fd)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def archive_memory_files_for_prune(
    source_memory_dir: Path,
    archive_dir: Path,
) -> bool:
    """Archive memory files from a pinned source without following entries."""
    try:
        source_directory_fd = _open_memory_directory_no_follow(source_memory_dir)
    except FileNotFoundError:
        return False
    with ExitStack() as directory_stack:
        directory_stack.callback(os.close, source_directory_fd)
        archive_directory_fd = _create_memory_directory_no_follow(archive_dir)
        directory_stack.callback(os.close, archive_directory_fd)
        lock_fd = _open_regular_file_at(
            source_directory_fd,
            MEMORY_LOCK_FILE,
            os.O_RDWR | os.O_CREAT,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            try:
                for name in (GLOBAL_MEMORY_MD_FILE, GLOBAL_MEMORY_JSONL_FILE):
                    _archive_regular_file_at(
                        source_directory_fd,
                        name,
                        archive_directory_fd,
                    )
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    return True


def reset_parallel_global_memory_for_round(
    global_memory_dir: Path,
    *,
    round_index: int,
) -> None:
    """Reset root global memory to the post-prune state for the next round."""
    memory_directory_fd = _create_memory_directory_no_follow(global_memory_dir)
    try:
        lock_fd = _open_regular_file_at(
            memory_directory_fd,
            MEMORY_LOCK_FILE,
            os.O_RDWR | os.O_CREAT,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                _reset_parallel_global_memory_for_round_at(
                    memory_directory_fd,
                    round_index=round_index,
                )
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    finally:
        os.close(memory_directory_fd)


def _reset_parallel_global_memory_for_round_at(
    memory_directory_fd: int,
    *,
    round_index: int,
) -> None:
    """Reset global memory entries through a pinned directory."""
    next_round = round_index + 1
    md_stat = _open_regular_file_for_read_at(memory_directory_fd, GLOBAL_MEMORY_MD_FILE)
    if md_stat is not None:
        os.close(md_stat[0])
    jsonl_stat = _open_regular_file_for_read_at(
        memory_directory_fd,
        GLOBAL_MEMORY_JSONL_FILE,
    )
    if jsonl_stat is not None:
        os.close(jsonl_stat[0])
    _replace_file_content_at(
        memory_directory_fd,
        GLOBAL_MEMORY_MD_FILE,
        (
        f"# Global Memory Log (Round {next_round})\n\n"
            f"_Pruned at round boundary. Previous context archived in _pruned/round_{round_index:03d}/_\n\n"
        ).encode("utf-8"),
        md_stat[1] if md_stat is not None else None,
    )
    prune_entry = {
        "timestamp": utc_now(),
        "step": "parallel_prune",
        "scope": "global",
        "summary": f"Parallel pruning: reset context for round {next_round}.",
        "round": next_round,
    }
    _replace_file_content_at(
        memory_directory_fd,
        GLOBAL_MEMORY_JSONL_FILE,
        (json.dumps(prune_entry, sort_keys=True) + "\n").encode("utf-8"),
        jsonl_stat[1] if jsonl_stat is not None else None,
    )


def _open_memory_directory_no_follow(directory: Path) -> int:
    """Open a memory directory without following any path component."""
    return _walk_memory_directory_no_follow(directory, create=False)


def _create_memory_directory_no_follow(directory: Path) -> int:
    """Create and pin a memory directory without following any path component."""
    return _walk_memory_directory_no_follow(directory, create=True)


def _walk_memory_directory_no_follow(directory: Path, *, create: bool) -> int:
    """Walk a directory path with dirfds, rejecting symlinks at every level."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if directory.is_absolute():
        descriptor = os.open(directory.anchor, flags)
        components = directory.parts[1:]
    else:
        descriptor = os.open(".", flags)
        components = directory.parts
    try:
        for component in components:
            if component in ("", "."):
                continue
            if create:
                try:
                    os.mkdir(component, mode=0o777, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_file_at(
    directory_fd: int,
    name: str,
    flags: int,
    *,
    mode: int = 0o666,
) -> int:
    """Open and verify a regular directory entry without following links."""
    descriptor = os.open(
        name,
        flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        mode,
        dir_fd=directory_fd,
    )
    if stat.S_ISREG(os.fstat(descriptor).st_mode):
        return descriptor
    os.close(descriptor)
    raise OSError(errno.EINVAL, f"Unsafe non-regular memory file: {name}")


def _read_shared_regular_file_at(
    directory_fd: int,
    name: str,
) -> tuple[bytes, os.stat_result] | None:
    """Read a shared regular file from a pinned directory, rejecting unsafe entries."""
    try:
        descriptor = _open_regular_file_at(directory_fd, name, os.O_RDONLY)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as handle:
        opened_stat = os.fstat(handle.fileno())
        return handle.read(), opened_stat


def _open_regular_file_for_read_at(
    directory_fd: int,
    name: str,
) -> tuple[int, os.stat_result] | None:
    """Open a regular directory entry without following links or special files."""
    try:
        entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(entry_stat.st_mode):
        return None

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENXIO}:
            return None
        raise
    opened_stat = os.fstat(descriptor)
    if stat.S_ISREG(opened_stat.st_mode):
        return descriptor, opened_stat
    os.close(descriptor)
    return None


def _read_regular_file_with_stat_at(
    directory_fd: int,
    name: str,
) -> tuple[bytes, os.stat_result] | None:
    """Read a safe regular directory entry and return its opened metadata."""
    opened_file = _open_regular_file_for_read_at(directory_fd, name)
    if opened_file is None:
        return None
    descriptor, opened_stat = opened_file
    with os.fdopen(descriptor, "rb") as handle:
        return handle.read(), opened_stat


def _read_regular_file_at(directory_fd: int, name: str) -> bytes | None:
    """Read bytes from a safe regular directory entry, if present."""
    opened_file = _read_regular_file_with_stat_at(directory_fd, name)
    return opened_file[0] if opened_file is not None else None


def _snapshot_regular_file_at(
    directory_fd: int,
    name: str,
) -> tuple[int, str] | None:
    """Return size and SHA-256 for a safe regular file without buffering it."""
    opened_file = _open_regular_file_for_read_at(directory_fd, name)
    if opened_file is None:
        return None
    descriptor, opened_stat = opened_file
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return opened_stat.st_size, digest.hexdigest()


def _open_conclusions_directory_at(
    shared_directory_fd: int,
    *,
    create: bool,
) -> int | None:
    """Open and pin the shared conclusions subdirectory without following links."""
    if create:
        try:
            os.mkdir(SHARED_CONCLUSIONS_SUBDIR, mode=0o777, dir_fd=shared_directory_fd)
        except FileExistsError:
            pass
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            SHARED_CONCLUSIONS_SUBDIR,
            flags,
            dir_fd=shared_directory_fd,
        )
    except FileNotFoundError:
        if not create:
            return None
        raise
    if stat.S_ISDIR(os.fstat(descriptor).st_mode):
        return descriptor
    os.close(descriptor)
    raise NotADirectoryError(SHARED_CONCLUSIONS_SUBDIR)


def _open_child_directory_at(
    parent_directory_fd: int,
    name: str,
    *,
    create: bool,
) -> int | None:
    """Open and pin a direct child directory without following links."""
    if create:
        try:
            os.mkdir(name, mode=0o777, dir_fd=parent_directory_fd)
        except FileExistsError:
            pass
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_directory_fd)
    except FileNotFoundError:
        if not create:
            return None
        raise
    if stat.S_ISDIR(os.fstat(descriptor).st_mode):
        return descriptor
    os.close(descriptor)
    raise NotADirectoryError(name)


def _atomic_replace_regular_file_at(
    directory_fd: int,
    name: str,
    content: bytes,
    mode: int,
) -> None:
    """Publish bytes through a fresh file, replacing any destination link."""
    temporary_name = f".{name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _atomic_copy_regular_file_at(
    directory_fd: int,
    name: str,
    source: BinaryIO,
    mode: int,
) -> None:
    """Stream a pinned source into a fresh file and atomically publish it."""
    temporary_name = f".{name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            source.seek(0)
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fchmod(destination.fileno(), mode)
            os.fsync(destination.fileno())
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _effective_creation_mode_at(directory_fd: int, requested_mode: int) -> int:
    """Resolve a requested creation mode through the caller's current umask."""
    temporary_file_flag = getattr(os, "O_TMPFILE", 0)
    if temporary_file_flag == 0:
        return 0o600
    try:
        descriptor = os.open(
            ".",
            os.O_RDWR | temporary_file_flag,
            requested_mode,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        unsupported_errors = {
            errno.EACCES,
            errno.EINVAL,
            errno.EISDIR,
            errno.ENOENT,
            errno.ENOSYS,
            errno.EOPNOTSUPP,
            errno.EPERM,
        }
        if exc.errno in unsupported_errors:
            return 0o600
        raise
    with os.fdopen(descriptor, "rb") as probe:
        return stat.S_IMODE(os.fstat(probe.fileno()).st_mode)


def _atomic_create_regular_file_at(
    directory_fd: int,
    name: str,
    content: bytes,
    requested_mode: int = 0o666,
) -> None:
    """Atomically create generated content with caller-umask permissions."""
    _atomic_replace_regular_file_at(
        directory_fd,
        name,
        content,
        _effective_creation_mode_at(directory_fd, requested_mode),
    )


def _atomic_write_generated_file_at(
    directory_fd: int,
    name: str,
    content: bytes,
) -> None:
    """Publish generated content, preserving a safe existing regular mode."""
    previous_stat: os.stat_result | None = None
    try:
        entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISREG(entry_stat.st_mode):
            previous_stat = entry_stat
    _replace_file_content_at(directory_fd, name, content, previous_stat)


def _archive_regular_file_at(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str | None = None,
) -> os.stat_result | None:
    """Safely archive one regular source entry into a pinned destination."""
    opened_file = _open_regular_file_for_read_at(source_directory_fd, source_name)
    if opened_file is None:
        return None
    descriptor, source_stat = opened_file
    with os.fdopen(descriptor, "rb") as source:
        _atomic_copy_regular_file_at(
            destination_directory_fd,
            destination_name or source_name,
            source,
            stat.S_IMODE(source_stat.st_mode),
        )
    return source_stat


def _unlink_regular_file_if_unchanged_at(
    directory_fd: int,
    name: str,
    opened_stat: os.stat_result,
) -> None:
    """Unlink the archived entry only if its identity has not changed."""
    try:
        current_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current_stat.st_mode)
        and current_stat.st_dev == opened_stat.st_dev
        and current_stat.st_ino == opened_stat.st_ino
    ):
        os.unlink(name, dir_fd=directory_fd)


def _entry_exists_at(directory_fd: int, name: str) -> bool:
    """Return whether a directory entry exists without following it."""
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _unlink_non_directory_entry_at(directory_fd: int, name: str) -> None:
    """Remove one file-like entry without following it."""
    try:
        entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(entry_stat.st_mode):
        os.unlink(name, dir_fd=directory_fd)


def _replace_file_content_at(
    directory_fd: int,
    name: str,
    content: bytes,
    previous_stat: os.stat_result | None,
) -> None:
    """Reset a pinned entry atomically without following its prior value."""
    if previous_stat is None:
        _atomic_create_regular_file_at(directory_fd, name, content)
        return
    _atomic_replace_regular_file_at(
        directory_fd,
        name,
        content,
        stat.S_IMODE(previous_stat.st_mode),
    )


def _atomic_replace_memory_file(
    directory_fd: int,
    name: str,
    content: bytes,
    source_stat: os.stat_result,
) -> None:
    """Publish copied memory while preserving the source file mode."""
    _atomic_replace_regular_file_at(
        directory_fd,
        name,
        content,
        stat.S_IMODE(source_stat.st_mode),
    )


def sync_memory_before_cycle(
    shared_memory_dir: Path,
    local_memory_dir: Path,
) -> dict[str, int]:
    """Copy shared memory files into a worker's local memory directory.

    Returns a pre-cycle snapshot dict (line count for JSONL, byte size for MD)
    so that ``sync_memory_after_cycle`` can compute the diff of new entries.
    """
    local_memory_dir.mkdir(parents=True, exist_ok=True)

    shared_directory_fd = _create_memory_directory_no_follow(shared_memory_dir)
    try:
        lock_fd = _open_regular_file_at(
            shared_directory_fd,
            MEMORY_LOCK_FILE,
            os.O_RDWR | os.O_CREAT,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            try:
                shared_jsonl_data = _read_shared_regular_file_at(
                    shared_directory_fd,
                    GLOBAL_MEMORY_JSONL_FILE,
                )
                shared_md_data = _read_shared_regular_file_at(
                    shared_directory_fd,
                    GLOBAL_MEMORY_MD_FILE,
                )
                local_directory_fd = _open_memory_directory_no_follow(local_memory_dir)
                try:
                    if shared_jsonl_data is not None:
                        _atomic_replace_memory_file(
                            local_directory_fd,
                            GLOBAL_MEMORY_JSONL_FILE,
                            *shared_jsonl_data,
                        )
                    if shared_md_data is not None:
                        _atomic_replace_memory_file(
                            local_directory_fd,
                            GLOBAL_MEMORY_MD_FILE,
                            *shared_md_data,
                        )
                    local_jsonl_data = (
                        shared_jsonl_data[0]
                        if shared_jsonl_data is not None
                        else _read_regular_file_at(
                            local_directory_fd,
                            GLOBAL_MEMORY_JSONL_FILE,
                        )
                    )
                    local_md_data = (
                        shared_md_data[0]
                        if shared_md_data is not None
                        else _read_regular_file_at(local_directory_fd, GLOBAL_MEMORY_MD_FILE)
                    )
                finally:
                    os.close(local_directory_fd)

                jsonl_lines = (
                    len(local_jsonl_data.decode("utf-8").splitlines())
                    if local_jsonl_data is not None
                    else 0
                )
                md_size = len(local_md_data) if local_md_data is not None else 0
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    finally:
        os.close(shared_directory_fd)

    logger.debug(
        "sync_memory_before_cycle: shared=%s -> local=%s (jsonl_lines=%d, md_size=%d)",
        shared_memory_dir,
        local_memory_dir,
        jsonl_lines,
        md_size,
    )
    return {"jsonl_lines": jsonl_lines, "md_size": md_size}


def sync_memory_after_cycle(
    shared_memory_dir: Path,
    local_memory_dir: Path,
    pre_state: dict[str, int],
) -> None:
    """Append new memory entries from a worker's local dir to the shared dir.

    Uses ``fcntl.flock(LOCK_EX)`` to serialize concurrent writers.
    Only entries added *after* the snapshot in ``pre_state`` are appended.
    """
    try:
        local_directory_fd = _open_memory_directory_no_follow(local_memory_dir)
    except OSError:
        logger.warning("sync_memory_after_cycle: refusing unsafe local memory=%s", local_memory_dir)
        return
    try:
        local_jsonl_data = _read_regular_file_at(
            local_directory_fd,
            GLOBAL_MEMORY_JSONL_FILE,
        )
        local_md_data = _read_regular_file_at(local_directory_fd, GLOBAL_MEMORY_MD_FILE)
    finally:
        os.close(local_directory_fd)

    new_jsonl_lines: list[str] = []
    if local_jsonl_data is not None:
        all_lines = local_jsonl_data.decode("utf-8").splitlines()
        prev_count = pre_state.get("jsonl_lines", 0)
        if len(all_lines) > prev_count:
            new_jsonl_lines = all_lines[prev_count:]

    new_md_bytes = b""
    if local_md_data is not None:
        prev_size = pre_state.get("md_size", 0)
        if len(local_md_data) > prev_size:
            new_md_bytes = local_md_data[prev_size:]

    if not new_jsonl_lines and not new_md_bytes:
        logger.debug(
            "sync_memory_after_cycle: no new entries from local=%s",
            local_memory_dir,
        )
        return

    shared_directory_fd = _create_memory_directory_no_follow(shared_memory_dir)
    try:
        lock_fd = _open_regular_file_at(
            shared_directory_fd,
            MEMORY_LOCK_FILE,
            os.O_RDWR | os.O_CREAT,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                if new_jsonl_lines:
                    shared_jsonl_fd = _open_regular_file_at(
                        shared_directory_fd,
                        GLOBAL_MEMORY_JSONL_FILE,
                        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                        mode=0o666,
                    )
                    with os.fdopen(shared_jsonl_fd, "a", encoding="utf-8") as f:
                        for line in new_jsonl_lines:
                            f.write(line)
                            if not line.endswith("\n"):
                                f.write("\n")

                if new_md_bytes:
                    shared_md_fd = _open_regular_file_at(
                        shared_directory_fd,
                        GLOBAL_MEMORY_MD_FILE,
                        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                        mode=0o666,
                    )
                    with os.fdopen(shared_md_fd, "ab") as f:
                        f.write(new_md_bytes)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    finally:
        os.close(shared_directory_fd)

    logger.debug(
        "sync_memory_after_cycle: appended %d jsonl lines, %d md bytes from local=%s to shared=%s",
        len(new_jsonl_lines),
        len(new_md_bytes),
        local_memory_dir,
        shared_memory_dir,
    )


def sync_conclusions_before_cycle(
    shared_memory_dir: Path,
    conclusions_dir: Path,
    worker_index: int,
) -> ConclusionsSyncState:
    """Refresh ``conclusions_other_workers.md`` from the shared dir.

    Concatenates each ``<shared>/conclusions/worker_*.md`` file (excluding
    this worker's own copy) into ``<conclusions_dir>/conclusions_other_workers.md``,
    one section per worker. Returns a snapshot for ``sync_conclusions_after_cycle``
    (currently the byte size of the worker's own ``conclusions.md`` for cheap
    change detection).
    """
    sections = _read_shared_conclusion_sections(shared_memory_dir, worker_index)
    other_content = (
        "\n".join(sections) if sections else EMPTY_OTHER_WORKERS_CONCLUSIONS_PLACEHOLDER
    ).encode("utf-8")
    conclusions_directory_fd = _create_memory_directory_no_follow(conclusions_dir)
    try:
        _atomic_write_generated_file_at(
            conclusions_directory_fd,
            OTHER_WORKERS_CONCLUSIONS_FILE,
            other_content,
        )
        own_snapshot = _snapshot_regular_file_at(
            conclusions_directory_fd,
            CONCLUSIONS_FILE,
        )
    finally:
        os.close(conclusions_directory_fd)

    own_size, own_hash = own_snapshot if own_snapshot is not None else (0, "")
    logger.debug(
        "sync_conclusions_before_cycle: worker=%d shared=%s -> conclusions_dir=%s "
        "(other_sections=%d, own_size=%d)",
        worker_index,
        shared_memory_dir,
        conclusions_dir,
        len(sections),
        own_size,
    )
    return {"own_conclusions_size": own_size, "own_conclusions_sha256": own_hash}


def sync_conclusions_after_cycle(
    shared_memory_dir: Path,
    conclusions_dir: Path,
    worker_index: int,
    pre_state: ConclusionsSyncState,
) -> None:
    """Publish this worker's ``conclusions.md`` to the shared dir if it changed.

    Compares the current content hash of ``<conclusions_dir>/conclusions.md``
    against the snapshot in ``pre_state``. If unchanged, this is a no-op.
    Otherwise the file is copied to
    ``<shared>/conclusions/worker_{i:02d}.md`` while holding the existing
    cross-worker memory lock so the read-side concat in
    ``sync_conclusions_before_cycle`` cannot observe a half-written file.
    """
    try:
        conclusions_directory_fd = _open_memory_directory_no_follow(conclusions_dir)
    except OSError:
        logger.warning(
            "sync_conclusions_after_cycle: refusing unsafe conclusions_dir=%s",
            conclusions_dir,
        )
        return
    try:
        own_file = _open_regular_file_for_read_at(
            conclusions_directory_fd,
            CONCLUSIONS_FILE,
        )
    finally:
        os.close(conclusions_directory_fd)
    if own_file is None:
        return
    own_descriptor, own_stat = own_file
    with os.fdopen(own_descriptor, "rb") as own_handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: own_handle.read(1024 * 1024), b""):
            digest.update(chunk)
        current_hash = digest.hexdigest()
        prev_hash = pre_state.get("own_conclusions_sha256", "")
        if current_hash == prev_hash:
            logger.debug(
                "sync_conclusions_after_cycle: worker=%d no change (sha256=%s)",
                worker_index,
                current_hash,
            )
            return

        target_name = f"worker_{worker_index:02d}.md"
        shared_directory_fd = _create_memory_directory_no_follow(shared_memory_dir)
        try:
            lock_fd = _open_regular_file_at(
                shared_directory_fd,
                MEMORY_LOCK_FILE,
                os.O_RDWR | os.O_CREAT,
            )
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                try:
                    shared_conclusions_fd = _open_conclusions_directory_at(
                        shared_directory_fd,
                        create=True,
                    )
                    assert shared_conclusions_fd is not None
                    try:
                        _atomic_copy_regular_file_at(
                            shared_conclusions_fd,
                            target_name,
                            own_handle,
                            stat.S_IMODE(own_stat.st_mode),
                        )
                    finally:
                        os.close(shared_conclusions_fd)
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        finally:
            os.close(shared_directory_fd)

    logger.debug(
        "sync_conclusions_after_cycle: worker=%d copied conclusions=%s -> shared=%s (size=%d)",
        worker_index,
        conclusions_md_path(conclusions_dir),
        shared_worker_conclusions_path(shared_memory_dir, worker_index),
        own_stat.st_size,
    )


def _read_shared_conclusion_sections(shared_memory_dir: Path, worker_index: int) -> list[str]:
    """Return formatted conclusion sections from other workers under the shared lock."""
    shared_directory_fd = _create_memory_directory_no_follow(shared_memory_dir)
    try:
        lock_fd = _open_regular_file_at(
            shared_directory_fd,
            MEMORY_LOCK_FILE,
            os.O_RDWR | os.O_CREAT,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            try:
                conclusions_directory_fd = _open_conclusions_directory_at(
                    shared_directory_fd,
                    create=False,
                )
                if conclusions_directory_fd is None:
                    return []
                try:
                    return _collect_shared_conclusion_sections(
                        conclusions_directory_fd,
                        worker_index,
                    )
                finally:
                    os.close(conclusions_directory_fd)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    finally:
        os.close(shared_directory_fd)


def _collect_shared_conclusion_sections(
    conclusions_directory_fd: int,
    worker_index: int,
) -> list[str]:
    """Collect formatted conclusion sections from other workers."""
    self_filename = f"worker_{worker_index:02d}.md"
    sections: list[str] = []
    for name in sorted(os.listdir(conclusions_directory_fd)):
        if name == self_filename:
            continue
        content = _read_regular_file_at(conclusions_directory_fd, name)
        if content is None:
            continue
        try:
            body = content.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if body:
            sections.append(f"### From {Path(name).stem}\n\n{body}\n")
    return sections
