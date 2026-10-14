#!/usr/bin/env python3
"""Orchestrator state: global memory log, shared memory/conclusions sync, append_memory.py, rubrics, idea bundles."""
from __future__ import annotations

import errno
import fcntl
import io
import json
import os
import stat
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from src.orchestrator import idea_bundle, rubrics
from src.orchestrator import state as state_store
from src.orchestrator.state import ConclusionsSyncState

REPO_ROOT = Path(__file__).resolve().parents[2]
APPEND_MEMORY_SCRIPT = REPO_ROOT / "src/agents/skills/memory-log/scripts/append_memory.py"
JSONL = state_store.GLOBAL_MEMORY_JSONL_FILE
MD = state_store.GLOBAL_MEMORY_MD_FILE
LOCK = state_store.MEMORY_LOCK_FILE
NO_PRIOR: ConclusionsSyncState = {"own_conclusions_sha256": ""}
EMPTY_PRE: ConclusionsSyncState = {"own_conclusions_size": 0, "own_conclusions_sha256": ""}
EVAL_EVENT = {
    "step": "evaluate",
    "scope": "attempt",
    "summary": "evaluated program_1",
    "evidence": "combined_score=2.71",
    "next_action": "stop",
}


def _md_event(**overrides: object) -> dict[str, object]:
    base = {
        "timestamp": "2026-08-15T00:00:00+00:00",
        "step": "idea",
        "scope": "idea",
        "summary": "generated idea",
        "evidence": "idea_bundle.json",
        "next_action": "run attempt",
        "idea_id": "idea_0001",
    }
    return {**base, **overrides}


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dirs(tmp_path: Path, *, shared: bool = True, local: bool = True) -> tuple[Path, Path]:
    shared_dir, local_dir = tmp_path / "shared", tmp_path / "local"
    if shared:
        shared_dir.mkdir(parents=True)
    if local:
        local_dir.mkdir(parents=True)
    return shared_dir, local_dir


@contextmanager
def _umask(value: int) -> Iterator[None]:
    previous = os.umask(value)
    try:
        yield
    finally:
        os.umask(previous)


def _run_append_memory(
    local_md: Path, global_md: Path, global_jsonl: Path, event: dict, *, popen: bool = False
):
    cmd = [
        sys.executable, str(APPEND_MEMORY_SCRIPT),
        "--local-memory-md", str(local_md),
        "--global-memory-md", str(global_md),
        "--global-memory-jsonl", str(global_jsonl),
        "--event-json", json.dumps(event),
    ]
    if popen:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=20)


def _write_conclusions(
    conclusions_md: Path,
    history: Path,
    text: str | None,
    *,
    cycle_index: int = 1,
    gate_trigger: str | None = "b",
    gate_reason: str = "first cycle",
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable, str(APPEND_MEMORY_SCRIPT), "--write-conclusions",
        "--conclusions-md", str(conclusions_md),
        "--conclusions-history-jsonl", str(history),
        "--cycle-index", str(cycle_index),
        "--worker-index", "0",
        "--gate-reason", gate_reason,
    ]
    if gate_trigger is not None:
        cmd += ["--gate-trigger", gate_trigger]
    if text is not None:
        cmd += ["--conclusions-text", text]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=20)


def _seed_shared_worker(shared_dir: Path, worker_index: int, body: str) -> None:
    target_dir = state_store.shared_conclusions_dir(shared_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"worker_{worker_index:02d}.md").write_text(body, encoding="utf-8")


def _other_text(conclusions_dir: Path) -> str:
    return state_store.other_workers_conclusions_md_path(conclusions_dir).read_text(encoding="utf-8")


def _symlink_outside_dir(tmp_path: Path, link: Path) -> Path:
    outside = tmp_path / "outside"
    outside.mkdir()
    link.symlink_to(outside, target_is_directory=True)
    return outside


# --------------------------------------------------------------------------- append_jsonl


def test_append_jsonl_terminates_records_retries_short_writes_and_accepts_bare_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # separates an unterminated existing record
    memory_path = tmp_path / "memory.jsonl"
    first = {"step": "idea", "idea_id": "idea_0001"}
    second = {"step": "evaluate", "attempt_id": "attempt_01"}
    memory_path.write_text(json.dumps(first), encoding="utf-8")
    original_bytes = memory_path.read_bytes()

    state_store.append_jsonl(memory_path, second)

    assert _records(memory_path) == [first, second]
    assert memory_path.read_bytes()[: len(original_bytes)] == original_bytes
    assert memory_path.read_bytes()[len(original_bytes) : len(original_bytes) + 1] == b"\n"

    # retries short writes
    memory_path = tmp_path / "short.jsonl"
    memory_path.write_text('{"step":"idea"}', encoding="utf-8")
    real_write = os.write

    def short_write(fd: int, data: bytes | memoryview) -> int:
        return real_write(fd, data[: max(1, len(data) // 2)])

    with mock.patch("src.orchestrator.state.os.write", side_effect=short_write):
        state_store.append_jsonl(memory_path, {"step": "evaluate"})

    assert _records(memory_path) == [{"step": "idea"}, {"step": "evaluate"}]

    # supports a bare relative filename
    relative_root = tmp_path / "relative"
    relative_root.mkdir()
    monkeypatch.chdir(relative_root)
    state_store.append_jsonl(Path("memory.jsonl"), {"step": "idea"})
    assert json.loads(Path("memory.jsonl").read_text(encoding="utf-8")) == {"step": "idea"}


def test_append_jsonl_rejects_symlink_target_and_symlinked_ancestor(tmp_path: Path) -> None:
    # rejects a symlink target
    target = tmp_path / "target.jsonl"
    target.write_text("unchanged", encoding="utf-8")
    memory_path = tmp_path / "memory.jsonl"
    memory_path.symlink_to(target)

    with pytest.raises(OSError):
        state_store.append_jsonl(memory_path, {"step": "evaluate"})

    assert target.read_text(encoding="utf-8") == "unchanged"

    # rejects a symlinked ancestor
    escape = tmp_path / "escape"
    escape.mkdir()
    linked_parent = tmp_path / "results"
    linked_parent.symlink_to(escape, target_is_directory=True)

    with pytest.raises(OSError):
        state_store.append_jsonl(linked_parent / "nested" / "memory.jsonl", {"step": "evaluate"})

    assert not (escape / "nested").exists()


def test_append_jsonl_if_absent_dedups_under_one_lock_rejects_oversized_scan_and_skips_invalid_utf8(
    tmp_path: Path,
) -> None:
    event: dict[str, object] = {"step": "idea", "idea_id": "idea_0001"}

    # checks and appends under one lock
    memory_path = tmp_path / "memory.jsonl"
    first = state_store.append_jsonl_if_absent(memory_path, event, matching=event)
    second = state_store.append_jsonl_if_absent(memory_path, event, matching=event)
    assert (first, second) == (True, False)
    assert len(memory_path.read_text(encoding="utf-8").splitlines()) == 1

    # rejects an oversized scan
    memory_path = tmp_path / "oversized.jsonl"
    with memory_path.open("wb") as handle:
        handle.truncate(state_store.MAX_JSONL_DEDUP_SCAN_BYTES + 1)
    with pytest.raises(OSError) as raised:
        state_store.append_jsonl_if_absent(memory_path, event, matching=event)
    assert raised.value.errno == errno.EFBIG

    # skips invalid UTF-8 records
    memory_path = tmp_path / "invalid-utf8.jsonl"
    memory_path.write_bytes(b"\xff\xfe\n")
    assert state_store.append_jsonl_if_absent(memory_path, event, matching=event)
    assert b'"idea_id": "idea_0001"' in memory_path.read_bytes()


def test_append_jsonl_if_absent_is_atomic_under_process_contention(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    worker_code = (
        "from pathlib import Path; from src.orchestrator import state; "
        "print(int(state.append_jsonl_if_absent(Path(__import__('sys').argv[1]), "
        "{'step':'idea','idea_id':'idea_0001'}, matching={'step':'idea','idea_id':'idea_0001'})))"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker_code, str(memory_path)],
            cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(16)
    ]

    results = [process.communicate(timeout=30) for process in processes]

    for process, (stdout, stderr) in zip(processes, results, strict=True):
        assert process.returncode == 0, (stdout or "") + (stderr or "")
    assert sum(int(stdout.strip()) for stdout, _ in results) == 1
    assert len(memory_path.read_text(encoding="utf-8").splitlines()) == 1


# --------------------------------------------------------------------------- markdown projection

MD_DEDUP_CASES = [
    (
        "exact-lines",
        _md_event(summary="generated idea 10", idea_id="idea_10"), "- idea_id: idea_10",
        _md_event(summary="generated idea 1", idea_id="idea_1"), "- idea_id: idea_1",
        (("- idea_id: idea_10", 1), ("- idea_id: idea_1\n", 1)),
    ),
    (
        "matching-step",
        _md_event(step="evaluate", summary="evaluated idea 1", idea_id="idea_1"), "- idea_id: idea_1",
        _md_event(summary="generated idea 1", idea_id="idea_1"), "- idea_id: idea_1",
        (("step=evaluate", 1), ("step=idea", 1)),
    ),
    (
        "exact-scope",
        _md_event(scope="idea-summary", summary="summarized idea 1", idea_id="idea_1"), "- idea_id: idea_1",
        _md_event(summary="generated idea 1", idea_id="idea_1"), "- idea_id: idea_1",
        (("scope=idea-summary", 1), ("scope=idea\n", 1)),
    ),
]


def test_markdown_projection_repairs_idempotently_and_dedups_only_on_exact_match(tmp_path: Path) -> None:
    # is repaired idempotently after a missing write
    memory_path = tmp_path / "memory.md"
    event = _md_event()
    repaired = state_store.append_memory_markdown_entry_if_absent(
        memory_path, event, matching_line="- idea_id: idea_0001"
    )
    duplicate = state_store.append_memory_markdown_entry_if_absent(
        memory_path, event, matching_line="- idea_id: idea_0001"
    )
    assert (repaired, duplicate) == (True, False)
    assert memory_path.read_text().count("idea_id: idea_0001") == 1

    # deduplication requires an exact match
    for name, first, first_line, second, second_line, checks in MD_DEDUP_CASES:
        memory_path = tmp_path / f"{name}.md"
        state_store.append_memory_markdown_entry_if_absent(memory_path, first, matching_line=first_line)

        appended = state_store.append_memory_markdown_entry_if_absent(
            memory_path, second, matching_line=second_line
        )

        assert appended, name
        text = memory_path.read_text(encoding="utf-8")
        for needle, count in checks:
            assert text.count(needle) == count, (name, needle)


# --------------------------------------------------------------------------- ensure_* files


def test_ensure_global_memory_files_create_migrate_legacy_names_and_backfill_markdown(tmp_path: Path) -> None:
    # ensure_histories creates the global memory files
    results_dir = tmp_path / "results"
    state_store.ensure_histories(results_dir)
    for name in (state_store.IDEAS_HISTORY_FILE, state_store.ATTEMPTS_HISTORY_FILE, JSONL, MD):
        assert (results_dir / name).is_file()
    assert "# Global Memory Log" in (results_dir / MD).read_text(encoding="utf-8")

    # migrates legacy filenames
    results_dir = tmp_path / "legacy"
    results_dir.mkdir()
    (results_dir / state_store.LEGACY_GLOBAL_MEMORY_MD_FILE).write_text(
        "# Global Memory Log\n\nlegacy md\n", encoding="utf-8"
    )
    (results_dir / state_store.LEGACY_GLOBAL_MEMORY_JSONL_FILES[-1]).write_text(
        '{"step":"idea"}\n', encoding="utf-8"
    )
    state_store.ensure_global_memory_files(results_dir)
    assert "legacy md" in (results_dir / MD).read_text(encoding="utf-8")
    assert '"step":"idea"' in (results_dir / JSONL).read_text(encoding="utf-8")

    # backfills markdown from jsonl
    results_dir = tmp_path / "backfill"
    results_dir.mkdir()
    (results_dir / MD).write_text("# Global Memory Log\n\n", encoding="utf-8")
    (results_dir / JSONL).write_text(
        json.dumps(_md_event(timestamp="2026-03-16T00:00:00+00:00", summary="generated idea bundle")) + "\n",
        encoding="utf-8",
    )
    state_store.ensure_global_memory_files(results_dir)
    global_md = (results_dir / MD).read_text(encoding="utf-8")
    assert "step=idea" in global_md
    assert "idea_id: idea_0001" in global_md


def test_ensure_conclusions_files_creates_then_preserves_existing(tmp_path: Path) -> None:
    conclusions_dir = tmp_path / "conclusions_dir"
    own_path = conclusions_dir / state_store.CONCLUSIONS_FILE
    other_path = conclusions_dir / state_store.OTHER_WORKERS_CONCLUSIONS_FILE

    state_store.ensure_conclusions_files(conclusions_dir)

    assert own_path.is_file() and other_path.is_file()
    assert "No conclusions yet" in own_path.read_text(encoding="utf-8")
    assert "No conclusions from other workers yet" in other_path.read_text(encoding="utf-8")

    own_path.write_text("custom content", encoding="utf-8")
    state_store.ensure_conclusions_files(conclusions_dir)
    assert own_path.read_text(encoding="utf-8") == "custom content"


# --------------------------------------------------------------------------- append_memory.py


def test_append_memory_writes_local_and_global_files_and_separates_unterminated_record(tmp_path: Path) -> None:
    assert APPEND_MEMORY_SCRIPT.is_file()
    # writes local and global files
    root = tmp_path / "write"
    local_memory = root / "attempt" / "memory.md"
    global_md = root / "results" / "memory.md"
    global_jsonl = root / "results" / "memory.jsonl"
    event = {
        **EVAL_EVENT,
        "next_action": "verify rubrics",
        "idea_id": "idea_0001",
        "attempt_id": "idea_0001_attempt_01",
        "program_id": "program_1",
        "metrics": {"combined_score": 2.71, "correct": True},
    }

    run = _run_append_memory(local_memory, global_md, global_jsonl, event)

    assert run.returncode == 0, run.stdout + run.stderr
    assert local_memory.is_file() and global_md.is_file() and global_jsonl.is_file()
    local_text = local_memory.read_text(encoding="utf-8")
    assert "# Memory Log" in local_text
    assert "step=evaluate" in local_text
    assert "program_id: program_1" in local_text
    records = _records(global_jsonl)
    assert len(records) == 1
    payload = records[0]
    assert payload["step"] == "evaluate"
    assert payload["scope"] == "attempt"
    assert payload["attempt_id"] == "idea_0001_attempt_01"
    assert payload["program_id"] == "program_1"
    assert "timestamp" in payload

    # separates an unterminated existing record
    root = tmp_path / "unterminated"
    global_jsonl = root / "results" / "memory.jsonl"
    first = {"step": "idea", "idea_id": "idea_0001"}
    global_jsonl.parent.mkdir(parents=True)
    global_jsonl.write_text(json.dumps(first), encoding="utf-8")
    event = {**EVAL_EVENT, "attempt_id": "attempt_01", "program_id": "program_1"}

    run = _run_append_memory(root / "attempt" / "memory.md", root / "results" / "memory.md", global_jsonl, event)

    assert run.returncode == 0, run.stdout + run.stderr
    records = _records(global_jsonl)
    assert records == [first, {**event, "timestamp": records[1]["timestamp"]}]


def test_append_memory_jsonl_remains_parseable_under_concurrent_writers(tmp_path: Path) -> None:
    global_md = tmp_path / "results" / "memory.md"
    global_jsonl = tmp_path / "results" / "memory.jsonl"
    seed = {"step": "seed", "summary": "unterminated"}
    global_jsonl.parent.mkdir(parents=True)
    global_jsonl.write_text(json.dumps(seed), encoding="utf-8")
    process_count = 24
    procs = []
    for idx in range(process_count):
        event = {
            **EVAL_EVENT,
            "summary": f"evaluated program_{idx}",
            "evidence": f"combined_score={2.5 + idx / 1000:.3f}",
            "next_action": "compare against best",
            "idea_id": "idea_concurrency",
            "attempt_id": f"attempt_{idx:03d}",
            "program_id": f"program_{idx}",
        }
        local_memory = tmp_path / "attempts" / f"memory_{idx:03d}.md"
        procs.append((idx, _run_append_memory(local_memory, global_md, global_jsonl, event, popen=True)))

    for idx, proc in procs:
        stdout, stderr = proc.communicate(timeout=30)
        assert proc.returncode == 0, f"writer={idx}\n{(stdout or '') + (stderr or '')}"

    parsed = _records(global_jsonl)
    assert len(parsed) == process_count + 1
    assert parsed[0] == seed
    observed = {str(entry.get("attempt_id", "")).strip() for entry in parsed if entry.get("attempt_id")}
    assert observed == {f"attempt_{idx:03d}" for idx in range(process_count)}


def test_append_memory_rejects_symlinked_paths_and_missing_required_fields(tmp_path: Path) -> None:
    # rejects a symlinked ancestor
    root = tmp_path / "ancestor"
    root.mkdir()
    escape = root / "escape"
    escape.mkdir()
    linked_parent = root / "results"
    linked_parent.symlink_to(escape, target_is_directory=True)

    run = _run_append_memory(
        root / "attempt" / "memory.md", linked_parent / "memory.md", linked_parent / "memory.jsonl", EVAL_EVENT
    )

    assert run.returncode != 0
    assert list(escape.iterdir()) == []

    # rejects a symlinked jsonl target
    root = tmp_path / "jsonl-target"
    target = root / "target.jsonl"
    target.parent.mkdir()
    target.write_text("unchanged", encoding="utf-8")
    global_jsonl = root / "results" / "memory.jsonl"
    global_jsonl.parent.mkdir(parents=True)
    global_jsonl.symlink_to(target)

    run = _run_append_memory(root / "attempt" / "memory.md", root / "results" / "memory.md", global_jsonl, EVAL_EVENT)

    assert run.returncode != 0
    assert target.read_text(encoding="utf-8") == "unchanged"

    # fails on a missing required field
    root = tmp_path / "missing-field"
    invalid_event = {"step": "evaluate", "scope": "attempt", "summary": "missing evidence", "next_action": "verify"}

    run = _run_append_memory(
        root / "attempt" / "memory.md", root / "results" / "memory.md", root / "results" / "memory.jsonl", invalid_event
    )

    assert run.returncode != 0


# --------------------------------------------------------------------------- sync_memory_before_cycle


def test_before_cycle_copies_shared_snapshot_under_shared_lock_and_reports_pre_state(tmp_path: Path) -> None:
    # copies shared files to local
    shared, local = _dirs(tmp_path / "copy")
    (shared / JSONL).write_text('{"step":"idea","summary":"test"}\n', encoding="utf-8")
    (shared / MD).write_text("# Memory\n\nSome content\n", encoding="utf-8")

    pre = state_store.sync_memory_before_cycle(shared, local)

    assert "test" in (local / JSONL).read_text(encoding="utf-8")
    assert "Some content" in (local / MD).read_text(encoding="utf-8")
    assert pre["jsonl_lines"] == 1
    assert pre["md_size"] > 0

    # holds the shared lock while copying the snapshot
    shared, local = _dirs(tmp_path / "lock", local=False)
    state_store.ensure_global_memory_files(shared)

    with mock.patch("src.orchestrator.state.fcntl.flock") as flock_mock:
        state_store.sync_memory_before_cycle(shared, local)

    assert [call.args[1] for call in flock_mock.call_args_list] == [fcntl.LOCK_SH, fcntl.LOCK_UN]

    # creates a nested local dir and reports an empty shared log
    shared, _ = _dirs(tmp_path / "nested", local=False)
    local = tmp_path / "nested" / "deeper" / "local"
    state_store.ensure_global_memory_files(shared)

    pre = state_store.sync_memory_before_cycle(shared, local)

    assert local.is_dir()
    assert pre["jsonl_lines"] == 0
    assert pre["md_size"] > 0  # header text only

    # without shared files returns a zero pre-state
    shared, local = _dirs(tmp_path / "empty")

    pre = state_store.sync_memory_before_cycle(shared, local)

    assert pre == {"jsonl_lines": 0, "md_size": 0}


def test_before_cycle_replaces_local_symlinks_preserves_source_mode_and_rejects_lock_symlink(
    tmp_path: Path,
) -> None:
    # replaces local symlinks without overwriting their targets
    root = tmp_path / "symlinks"
    shared, local = _dirs(root)
    shared_jsonl, shared_md = '{"step":"shared"}\n', "# Shared memory\n"
    (shared / JSONL).write_text(shared_jsonl, encoding="utf-8")
    (shared / MD).write_text(shared_md, encoding="utf-8")
    jsonl_target, md_target = root / "jsonl-target", root / "md-target"
    jsonl_target.write_text("keep jsonl target\n", encoding="utf-8")
    md_target.write_text("keep md target\n", encoding="utf-8")
    (local / JSONL).symlink_to(jsonl_target)
    (local / MD).symlink_to(md_target)

    pre = state_store.sync_memory_before_cycle(shared, local)

    assert jsonl_target.read_text(encoding="utf-8") == "keep jsonl target\n"
    assert md_target.read_text(encoding="utf-8") == "keep md target\n"
    assert not (local / JSONL).is_symlink() and not (local / MD).is_symlink()
    assert (local / JSONL).read_text(encoding="utf-8") == shared_jsonl
    assert (local / MD).read_text(encoding="utf-8") == shared_md
    assert pre == {"jsonl_lines": 1, "md_size": len(shared_md.encode())}

    # the copy preserves the source mode despite a restrictive umask
    shared, local = _dirs(tmp_path / "mode")
    source = shared / MD
    source.write_text("shared memory\n", encoding="utf-8")
    source.chmod(0o666)

    with _umask(0o077):
        state_store.sync_memory_before_cycle(shared, local)

    assert stat.S_IMODE((local / MD).stat().st_mode) == 0o666

    # rejects a shared lock symlink without truncating its target
    root = tmp_path / "lock-symlink"
    shared, local = _dirs(root, local=False)
    lock_target = root / "lock-target"
    lock_target.write_text("keep lock target\n", encoding="utf-8")
    (shared / LOCK).symlink_to(lock_target)

    with pytest.raises(OSError):
        state_store.sync_memory_before_cycle(shared, local)

    assert lock_target.read_text(encoding="utf-8") == "keep lock target\n"


# --------------------------------------------------------------------------- sync_memory_after_cycle


def test_after_cycle_appends_new_entries_and_creates_shared_dir_with_default_modes(tmp_path: Path) -> None:
    # appends new jsonl lines
    shared, local = _dirs(tmp_path / "jsonl")
    (shared / JSONL).write_text('{"step":"init"}\n', encoding="utf-8")
    state_store.ensure_global_memory_files(shared)
    pre = state_store.sync_memory_before_cycle(shared, local)
    assert pre["jsonl_lines"] == 1
    with open(local / JSONL, "a", encoding="utf-8") as f:
        f.write('{"step":"idea","summary":"new idea"}\n{"step":"evaluate","summary":"evaluated"}\n')

    state_store.sync_memory_after_cycle(shared, local, pre)

    assert len(_records(shared / JSONL)) == 3

    # appends new md bytes
    shared, local = _dirs(tmp_path / "md")
    (shared / MD).write_text("# Memory\n", encoding="utf-8")
    state_store.ensure_global_memory_files(shared)
    pre = state_store.sync_memory_before_cycle(shared, local)
    with open(local / MD, "a", encoding="utf-8") as f:
        f.write("\n## New Section\nworker added this\n")

    state_store.sync_memory_after_cycle(shared, local, pre)

    shared_md = (shared / MD).read_text(encoding="utf-8")
    assert "# Memory" in shared_md
    assert "worker added this" in shared_md

    # is a no-op when there are no new entries
    shared, local = _dirs(tmp_path / "noop", local=False)
    state_store.ensure_global_memory_files(shared)
    pre = state_store.sync_memory_before_cycle(shared, local)
    before = (shared / JSONL).read_text(encoding="utf-8")

    state_store.sync_memory_after_cycle(shared, local, pre)

    after = (shared / JSONL).read_text(encoding="utf-8")
    assert before == after
    assert after.strip() == ""

    # creates the shared dir if missing
    shared = tmp_path / "create" / "nested" / "shared"
    _, local = _dirs(tmp_path / "create", shared=False)
    state_store.ensure_global_memory_files(local)
    (local / JSONL).write_text('{"step":"test"}\n', encoding="utf-8")

    state_store.sync_memory_after_cycle(shared, local, {"jsonl_lines": 0, "md_size": 0})

    assert shared.is_dir()
    assert "test" in (shared / JSONL).read_text(encoding="utf-8")

    # created shared paths preserve the default requested modes
    shared, local = _dirs(tmp_path / "modes", shared=False)
    (local / JSONL).write_text('{"step":"test"}\n', encoding="utf-8")

    with _umask(0):
        state_store.sync_memory_after_cycle(shared, local, {"jsonl_lines": 0, "md_size": 0})

    assert stat.S_IMODE(shared.stat().st_mode) == 0o777
    assert stat.S_IMODE((shared / LOCK).stat().st_mode) == 0o666
    assert stat.S_IMODE((shared / JSONL).stat().st_mode) == 0o666


def test_after_cycle_concurrent_and_multi_cycle_workers_lose_no_entries(tmp_path: Path) -> None:
    # concurrent writers lose no entries
    root = tmp_path / "concurrent"
    shared, _ = _dirs(root, local=False)
    state_store.ensure_global_memory_files(shared)
    worker_count, entries_per_worker = 8, 5
    errors: list[str] = []

    def worker_fn(worker_idx: int) -> None:
        try:
            local = root / f"worker_{worker_idx}"
            local.mkdir()
            pre = state_store.sync_memory_before_cycle(shared, local)
            with open(local / JSONL, "a", encoding="utf-8") as f:
                for entry_idx in range(entries_per_worker):
                    f.write(json.dumps({"worker": worker_idx, "entry": entry_idx, "step": "evaluate"}) + "\n")
            state_store.sync_memory_after_cycle(shared, local, pre)
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion below
            errors.append(f"worker_{worker_idx}: {exc}")

    threads = [threading.Thread(target=worker_fn, args=(i,)) for i in range(worker_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    parsed = _records(shared / JSONL)
    assert len(parsed) == worker_count * entries_per_worker
    assert {entry["worker"] for entry in parsed} == set(range(worker_count))

    # two workers share memory across cycles
    root = tmp_path / "cycles"
    shared, _ = _dirs(root, local=False)
    state_store.ensure_global_memory_files(shared)
    w1, w2 = root / "worker1", root / "worker2"

    def append(local: Path, line: str) -> None:
        with open(local / JSONL, "a", encoding="utf-8") as f:
            f.write(line)

    pre1 = state_store.sync_memory_before_cycle(shared, w1)
    pre2 = state_store.sync_memory_before_cycle(shared, w2)
    append(w1, '{"worker":1,"cycle":1}\n')
    state_store.sync_memory_after_cycle(shared, w1, pre1)
    append(w2, '{"worker":2,"cycle":1}\n')
    state_store.sync_memory_after_cycle(shared, w2, pre2)
    assert len(_records(shared / JSONL)) == 2

    pre1 = state_store.sync_memory_before_cycle(shared, w1)
    pre2 = state_store.sync_memory_before_cycle(shared, w2)
    assert pre1["jsonl_lines"] == 2 and pre2["jsonl_lines"] == 2
    append(w1, '{"worker":1,"cycle":2}\n')
    state_store.sync_memory_after_cycle(shared, w1, pre1)
    state_store.sync_memory_after_cycle(shared, w2, pre2)
    assert len(_records(shared / JSONL)) == 3


def test_after_cycle_never_follows_symlinked_local_or_shared_memory_paths(tmp_path: Path) -> None:
    # ignores symlinked local memory files
    root = tmp_path / "local-symlinks"
    shared, local = _dirs(root)
    state_store.ensure_global_memory_files(shared)
    jsonl_target, md_target = root / "jsonl-target", root / "md-target"
    jsonl_target.write_text('{"step":"untrusted"}\n', encoding="utf-8")
    md_target.write_text("untrusted markdown\n", encoding="utf-8")
    (local / JSONL).symlink_to(jsonl_target)
    (local / MD).symlink_to(md_target)

    state_store.sync_memory_after_cycle(shared, local, {"jsonl_lines": 0, "md_size": 0})

    assert (shared / JSONL).read_text(encoding="utf-8") == ""
    assert (shared / MD).read_text(encoding="utf-8") == "# Global Memory Log\n\n"

    # rejects shared memory symlinks without appending to their targets
    for filename, local_content in ((JSONL, '{"step":"untrusted"}\n'), (MD, "untrusted markdown\n")):
        root = tmp_path / f"shared-symlink-{filename}"
        shared, local = _dirs(root)
        outside_target = root / "outside-target"
        outside_target.write_text("keep outside target\n", encoding="utf-8")
        (shared / filename).symlink_to(outside_target)
        (local / filename).write_text(local_content, encoding="utf-8")

        with pytest.raises(OSError):
            state_store.sync_memory_after_cycle(shared, local, {"jsonl_lines": 0, "md_size": 0})

        assert outside_target.read_text(encoding="utf-8") == "keep outside target\n", filename

    # rejects a symlinked shared memory directory
    root = tmp_path / "shared-dir-symlink"
    shared, local = _dirs(root, shared=False)
    outside = _symlink_outside_dir(root, shared)
    (local / JSONL).write_text('{"step":"untrusted"}\n', encoding="utf-8")

    with pytest.raises(OSError):
        state_store.sync_memory_after_cycle(shared, local, {"jsonl_lines": 0, "md_size": 0})

    assert list(outside.iterdir()) == []


# --------------------------------------------------------------------------- sync_conclusions_before_cycle


def test_conclusions_before_cycle_concats_other_workers_under_memory_lock(tmp_path: Path) -> None:
    # concatenates other workers' conclusions, excluding self
    shared, conclusions_dir = _dirs(tmp_path / "concat")
    _seed_shared_worker(shared, 1, "Self lessons.\n")
    _seed_shared_worker(shared, 2, "Worker two lessons.\n")
    _seed_shared_worker(shared, 3, "Worker three lessons.\n")

    pre = state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)

    assert state_store.other_workers_conclusions_md_path(conclusions_dir).is_file()
    text = _other_text(conclusions_dir)
    assert "Worker two lessons" in text and "Worker three lessons" in text
    assert "Self lessons" not in text
    assert pre["own_conclusions_size"] == 0

    # handles an empty shared dir
    shared, conclusions_dir = _dirs(tmp_path / "empty", local=False)

    state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)

    assert state_store.other_workers_conclusions_md_path(conclusions_dir).is_file()
    assert "No conclusions from other workers yet" in _other_text(conclusions_dir)

    # skips blank worker files
    shared, conclusions_dir = _dirs(tmp_path / "blank", local=False)
    _seed_shared_worker(shared, 2, "\n   \n")
    _seed_shared_worker(shared, 3, "Real content from worker three.")

    state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)

    text = _other_text(conclusions_dir)
    assert "worker three" in text
    assert "worker_02" not in text

    # reads shared worker files under the memory lock
    shared, conclusions_dir = _dirs(tmp_path / "lock", local=False)
    _seed_shared_worker(shared, 2, "Worker two lessons.")
    finished = threading.Event()

    def sync_before_cycle() -> None:
        state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)
        finished.set()

    with (shared / LOCK).open("w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        thread = threading.Thread(target=sync_before_cycle)
        thread.start()
        assert not finished.wait(timeout=0.1)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)

    thread.join(timeout=1.0)
    assert finished.is_set()
    assert "Worker two lessons" in _other_text(conclusions_dir)


def test_conclusions_before_cycle_never_follows_symlinks(tmp_path: Path) -> None:
    # ignores symlinked own conclusions
    root = tmp_path / "own"
    shared, conclusions_dir = _dirs(root)
    secret = root / "secret"
    secret.write_text("do not disclose\n", encoding="utf-8")
    state_store.conclusions_md_path(conclusions_dir).symlink_to(secret)

    pre = state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)

    assert pre == EMPTY_PRE
    assert "do not disclose" not in _other_text(conclusions_dir)
    assert secret.read_text(encoding="utf-8") == "do not disclose\n"

    # skips a symlinked shared worker file
    root = tmp_path / "shared-worker"
    shared, conclusions_dir = _dirs(root, shared=False, local=False)
    shared_conclusions = state_store.shared_conclusions_dir(shared)
    shared_conclusions.mkdir(parents=True)
    secret = root / "secret"
    secret.write_text("do not disclose\n", encoding="utf-8")
    (shared_conclusions / "worker_02.md").symlink_to(secret)

    state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)

    assert "do not disclose" not in _other_text(conclusions_dir)
    assert secret.read_text(encoding="utf-8") == "do not disclose\n"

    # replaces a local other-workers symlink
    root = tmp_path / "other-workers"
    shared, conclusions_dir = _dirs(root)
    outside = root / "outside"
    outside.write_text("keep outside\n", encoding="utf-8")
    other = state_store.other_workers_conclusions_md_path(conclusions_dir)
    other.symlink_to(outside)

    state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)

    assert outside.read_text(encoding="utf-8") == "keep outside\n"
    assert not other.is_symlink()
    assert "No conclusions" in other.read_text(encoding="utf-8")


def test_conclusions_before_cycle_other_workers_output_is_private_without_named_probe(tmp_path: Path) -> None:
    # the other-workers output is not world-writable
    shared, conclusions_dir = _dirs(tmp_path / "private", local=False)

    with _umask(0o077):
        state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)

    other = state_store.other_workers_conclusions_md_path(conclusions_dir)
    assert stat.S_IMODE(other.stat().st_mode) == 0o600

    # mode resolution creates no named probe
    shared, conclusions_dir = _dirs(tmp_path / "probe", local=False)
    opened_paths: list[str] = []
    real_open = os.open

    def recording_open(path, flags, *args, **kwargs):
        opened_paths.append(str(path))
        return real_open(path, flags, *args, **kwargs)

    with mock.patch.object(state_store.os, "open", side_effect=recording_open):
        state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)

    assert not any(Path(path).name.startswith(".mode-probe-") for path in opened_paths)

    # mode falls back to private when O_TMPFILE is unsupported
    for error_number in (errno.EOPNOTSUPP, errno.EACCES, errno.EPERM):
        shared, conclusions_dir = _dirs(tmp_path / f"tmpfile-{error_number}", local=False)

        def unsupported_tmpfile(path, flags, *args, error_number=error_number, **kwargs):
            tmpfile_flag = getattr(os, "O_TMPFILE", 0)
            if tmpfile_flag and flags & tmpfile_flag == tmpfile_flag:
                raise OSError(error_number, "O_TMPFILE unavailable")
            return real_open(path, flags, *args, **kwargs)

        with _umask(0), mock.patch.object(state_store.os, "open", side_effect=unsupported_tmpfile):
            state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)

        other = state_store.other_workers_conclusions_md_path(conclusions_dir)
        assert stat.S_IMODE(other.stat().st_mode) == 0o600, error_number


# --------------------------------------------------------------------------- sync_conclusions_after_cycle


def test_conclusions_after_cycle_publishes_changed_conclusions_atomically_and_skips_unchanged(
    tmp_path: Path,
) -> None:
    # copies own conclusions to the shared dir under the worker index
    shared, conclusions_dir = _dirs(tmp_path / "copy")
    own_path = state_store.conclusions_md_path(conclusions_dir)
    own_path.write_text("Promising lead: try X.\n", encoding="utf-8")

    state_store.sync_conclusions_after_cycle(shared, conclusions_dir, worker_index=2, pre_state=EMPTY_PRE)

    target = state_store.shared_worker_conclusions_path(shared, 2)
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == own_path.read_text(encoding="utf-8")

    # copies when same-size content changes
    shared, conclusions_dir = _dirs(tmp_path / "same-size")
    own_path = state_store.conclusions_md_path(conclusions_dir)
    own_path.write_text("Try A.\n", encoding="utf-8")
    pre = state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)
    own_path.write_text("Try B.\n", encoding="utf-8")

    state_store.sync_conclusions_after_cycle(shared, conclusions_dir, worker_index=1, pre_state=pre)

    assert state_store.shared_worker_conclusions_path(shared, 1).read_text(encoding="utf-8") == "Try B.\n"

    # the atomic publish leaves no temp file
    shared, conclusions_dir = _dirs(tmp_path / "atomic")
    state_store.conclusions_md_path(conclusions_dir).write_text("Promising lead.\n", encoding="utf-8")

    state_store.sync_conclusions_after_cycle(shared, conclusions_dir, worker_index=1, pre_state=NO_PRIOR)

    target = state_store.shared_worker_conclusions_path(shared, 1)
    assert target.read_text(encoding="utf-8") == "Promising lead.\n"
    assert not (target.parent / f".{target.name}.tmp").exists()

    # no-op if the content is unchanged
    shared, conclusions_dir = _dirs(tmp_path / "unchanged")
    state_store.conclusions_md_path(conclusions_dir).write_text("stable conclusion.\n", encoding="utf-8")
    pre = state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)

    state_store.sync_conclusions_after_cycle(shared, conclusions_dir, worker_index=1, pre_state=pre)

    assert not state_store.shared_worker_conclusions_path(shared, 1).exists()

    # no-op if the local file is missing
    shared, conclusions_dir = _dirs(tmp_path / "missing")

    state_store.sync_conclusions_after_cycle(shared, conclusions_dir, worker_index=1, pre_state=EMPTY_PRE)

    assert not state_store.shared_worker_conclusions_path(shared, 1).exists()


def test_conclusions_sync_preserves_source_mode_and_default_modes_with_private_temp(tmp_path: Path) -> None:
    # the after-cycle copy preserves the source mode despite a restrictive umask
    shared, conclusions_dir = _dirs(tmp_path / "source-mode", shared=False)
    own_path = state_store.conclusions_md_path(conclusions_dir)
    own_path.write_text("safe conclusion\n", encoding="utf-8")
    own_path.chmod(0o666)

    with _umask(0o077):
        state_store.sync_conclusions_after_cycle(shared, conclusions_dir, worker_index=1, pre_state=NO_PRIOR)

    target = state_store.shared_worker_conclusions_path(shared, 1)
    assert stat.S_IMODE(target.stat().st_mode) == 0o666

    # the atomic copy keeps the in-progress temp file private
    copy_dir = tmp_path / "atomic-copy"
    copy_dir.mkdir()
    observed_modes: list[int] = []

    class InspectingSource(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            temporary_files = list(copy_dir.glob(".worker_01.md.*.tmp"))
            if len(temporary_files) != 1:
                raise AssertionError(f"expected one temporary file, got {temporary_files}")
            observed_modes.append(stat.S_IMODE(temporary_files[0].stat().st_mode))
            return super().read(size)

    directory_fd = os.open(copy_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        state_store._atomic_copy_regular_file_at(
            directory_fd, "worker_01.md", InspectingSource(b"safe conclusion\n"), 0o666
        )
    finally:
        os.close(directory_fd)

    assert observed_modes
    assert set(observed_modes) == {0o600}
    assert stat.S_IMODE((copy_dir / "worker_01.md").stat().st_mode) == 0o666

    # created conclusion paths preserve the default requested modes
    shared, conclusions_dir = _dirs(tmp_path / "defaults", shared=False, local=False)

    with _umask(0):
        pre = state_store.sync_conclusions_before_cycle(shared, conclusions_dir, worker_index=1)
        state_store.conclusions_md_path(conclusions_dir).write_text("safe conclusion\n", encoding="utf-8")
        state_store.sync_conclusions_after_cycle(shared, conclusions_dir, worker_index=1, pre_state=pre)

    assert stat.S_IMODE(shared.stat().st_mode) == 0o777
    assert stat.S_IMODE(conclusions_dir.stat().st_mode) == 0o777
    assert stat.S_IMODE(state_store.shared_conclusions_dir(shared).stat().st_mode) == 0o777
    assert stat.S_IMODE((shared / LOCK).stat().st_mode) == 0o666
    other = state_store.other_workers_conclusions_md_path(conclusions_dir)
    assert stat.S_IMODE(other.stat().st_mode) in {0o600, 0o666}


def test_conclusions_after_cycle_never_writes_through_symlinks(tmp_path: Path) -> None:
    # ignores symlinked local conclusions
    root = tmp_path / "local"
    shared, conclusions_dir = _dirs(root)
    secret = root / "secret"
    secret.write_text("do not publish\n", encoding="utf-8")
    state_store.conclusions_md_path(conclusions_dir).symlink_to(secret)

    state_store.sync_conclusions_after_cycle(shared, conclusions_dir, worker_index=1, pre_state=NO_PRIOR)

    assert not state_store.shared_worker_conclusions_path(shared, 1).exists()
    assert secret.read_text(encoding="utf-8") == "do not publish\n"

    # never writes through symlinks in the shared dir (destination / fixed temp name)
    for name, symlink_fixed_temp in (("destination", False), ("fixed-temp", True)):
        root = tmp_path / name
        shared, conclusions_dir = _dirs(root, shared=False)
        state_store.conclusions_md_path(conclusions_dir).write_text("safe conclusion\n", encoding="utf-8")
        target = state_store.shared_worker_conclusions_path(shared, 1)
        target.parent.mkdir(parents=True)
        outside = root / "outside"
        outside.write_text("keep outside\n", encoding="utf-8")
        link = target.parent / f".{target.name}.tmp" if symlink_fixed_temp else target
        link.symlink_to(outside)

        state_store.sync_conclusions_after_cycle(shared, conclusions_dir, worker_index=1, pre_state=NO_PRIOR)

        assert outside.read_text(encoding="utf-8") == "keep outside\n", name
        assert link.is_symlink() == symlink_fixed_temp, name
        assert target.read_text(encoding="utf-8") == "safe conclusion\n", name

    # rejects a symlinked shared conclusions directory
    root = tmp_path / "shared-dir"
    shared, conclusions_dir = _dirs(root)
    outside = _symlink_outside_dir(root, state_store.shared_conclusions_dir(shared))
    state_store.conclusions_md_path(conclusions_dir).write_text("safe conclusion\n", encoding="utf-8")

    with pytest.raises(OSError):
        state_store.sync_conclusions_after_cycle(shared, conclusions_dir, worker_index=1, pre_state=NO_PRIOR)

    assert list(outside.iterdir()) == []

    # publishes to the pinned directory if the path is swapped mid-publish
    root = tmp_path / "swapped"
    shared, conclusions_dir = _dirs(root, shared=False)
    shared_conclusions = state_store.shared_conclusions_dir(shared)
    detached = shared / "detached-conclusions"
    outside = root / "outside"
    shared_conclusions.mkdir(parents=True)
    outside.mkdir()
    state_store.conclusions_md_path(conclusions_dir).write_text("safe conclusion\n", encoding="utf-8")
    real_open = state_store._open_conclusions_directory_at

    def open_then_swap(shared_fd: int, *, create: bool) -> int | None:
        directory_fd = real_open(shared_fd, create=create)
        assert directory_fd is not None
        shared_conclusions.rename(detached)
        shared_conclusions.symlink_to(outside, target_is_directory=True)
        return directory_fd

    with mock.patch.object(state_store, "_open_conclusions_directory_at", side_effect=open_then_swap):
        state_store.sync_conclusions_after_cycle(shared, conclusions_dir, worker_index=1, pre_state=NO_PRIOR)

    assert list(outside.iterdir()) == []
    assert (detached / "worker_01.md").read_text(encoding="utf-8") == "safe conclusion\n"


# --------------------------------------------------------------------------- append_memory.py --write-conclusions


def test_write_conclusions_accepts_valid_text_archives_prior_and_rejects_without_touching_files(
    tmp_path: Path,
) -> None:
    # accepts a valid paragraph (and exactly 300 words)
    for name, text in (("paragraph", "Promising lead: explore X. Avoid Y."), ("exactly-300-words", " ".join(["word"] * 300))):
        conclusions_md, history = tmp_path / f"{name}.md", tmp_path / f"{name}_history.jsonl"

        run = _write_conclusions(conclusions_md, history, text)

        assert run.returncode == 0, (name, run.stderr)
        assert conclusions_md.read_text(encoding="utf-8").strip() == text, name
        records = _records(history)
        assert len(records) == 1, name
        assert records[0]["gate_trigger"] == "b", name
        assert records[0]["gate_reason"] == "first cycle", name
        assert records[0]["new"].strip() == text, name
        assert records[0]["prior"] == "", name

    # archives the prior version to the history jsonl
    conclusions_md, history = tmp_path / "conclusions.md", tmp_path / "conclusions_history.jsonl"
    assert _write_conclusions(conclusions_md, history, "First version of conclusions.").returncode == 0

    run = _write_conclusions(
        conclusions_md, history, "Second version, replacing the first.",
        cycle_index=2, gate_trigger="a", gate_reason="cycle 2 contradicts prior",
    )

    assert run.returncode == 0
    records = _records(history)
    assert len(records) == 2
    assert "First version" in records[1]["prior"]
    assert "Second version" in records[1]["new"]

    # a rejection leaves the existing files byte-identical
    initial_md, initial_history = conclusions_md.read_bytes(), history.read_bytes()

    run = _write_conclusions(
        conclusions_md, history, " ".join(["word"] * 500),
        cycle_index=3, gate_trigger="d", gate_reason="should be rejected",
    )

    assert run.returncode != 0
    assert conclusions_md.read_bytes() == initial_md
    assert history.read_bytes() == initial_history


@pytest.mark.parametrize(
    ("text", "gate_trigger", "expected_stderr"),
    [
        pytest.param("First paragraph.\n\nSecond paragraph.", "b", "single paragraph", id="multi-paragraph"),
        pytest.param(" ".join(["word"] * 301), "b", "word count", id="over-300-words"),
        pytest.param("   \n  \t", "b", "empty", id="empty"),
        pytest.param("some valid paragraph", "z", "gate-trigger", id="invalid-gate-trigger"),
    ],
)
def test_write_conclusions_rejects_invalid_input_without_writing(
    tmp_path: Path, text: str, gate_trigger: str, expected_stderr: str
) -> None:
    conclusions_md, history = tmp_path / "conclusions.md", tmp_path / "conclusions_history.jsonl"

    run = _write_conclusions(conclusions_md, history, text, gate_trigger=gate_trigger)

    assert run.returncode != 0
    assert expected_stderr in run.stderr.lower()
    assert not conclusions_md.exists() and not history.exists()


# --------------------------------------------------------------------------- rubrics / idea_bundle


def test_rubric_helpers_validate_status_extract_ids_and_load_json(tmp_path: Path) -> None:
    # validate_rubric_status accepts matching ids
    status = {
        "attempt_id": "a1", "idea_id": "i1", "overall_pass": True,
        "rubrics": [{"id": "r1", "pass": True, "evidence": "ok"}, {"id": "r2", "pass": True, "evidence": "ok"}],
    }
    result = rubrics.validate_rubric_status(status, {"r1", "r2"}, "a1", "i1")
    assert result.valid and result.overall_pass
    assert result.error is None

    # validate_rubric_status rejects a mismatch
    status = {
        "attempt_id": "a1", "idea_id": "i1", "overall_pass": True,
        "rubrics": [{"id": "r1", "pass": True, "evidence": "ok"}],
    }
    result = rubrics.validate_rubric_status(status, {"r1", "r2"}, "a1", "i1")
    assert not result.valid
    assert "mismatch" in (result.error or "")

    # extract_target_rubric_ids from a list
    assert rubrics.extract_target_rubric_ids([{"id": "a"}, {"id": "b"}, {"description": "missing id"}]) == {"a", "b"}

    # load_json_dict handles invalid JSON
    p = tmp_path / "status.json"
    p.write_text("{invalid", encoding="utf-8")
    assert rubrics.load_json_dict(p) is None


REPAIR_CASES = [
    (
        "structural-trailing-commas",
        '{"text":"keep comma, } and ],", "items":[1, 2,],}',
        ({"text": "keep comma, } and ],", "items": [1, 2]}, True),
    ),
    ("valid-unchanged", '{"items":[1,2]}', ({"items": [1, 2]}, False)),
]
MALFORMED_JSON = [
    "{'items': [1, 2]}", "[,]", "{,}", '{"items":[,]}', '{"item":{,}}',
    '{"items":[NaN]}', '{"items":[Infinity]}', '{"items":[-Infinity]}',
]


def test_parse_agent_json_repairs_only_structural_trailing_commas() -> None:
    for name, raw, expected in REPAIR_CASES:
        assert idea_bundle.parse_agent_json(raw) == expected, name
    # rejects non-trailing-comma errors
    for malformed in MALFORMED_JSON:
        with pytest.raises(json.JSONDecodeError):
            idea_bundle.parse_agent_json(malformed)


def test_normalize_idea_bundle_and_target_rubrics_payload() -> None:
    # normalize_target_rubrics_payload accepts a dict with target_rubrics
    rubrics_payload = idea_bundle.normalize_target_rubrics_payload(
        {"target_rubrics": [{"id": "score-threshold", "description": "Score > 2.6", "type": "metric"}]}
    )
    assert rubrics_payload[0]["id"] == "score_threshold"
    assert rubrics_payload[0]["type"] == "metric"

    # normalize_idea_bundle returns the canonical payload
    normalized = idea_bundle.normalize_idea_bundle(
        {
            "idea_id": " Cycle 3 / Covariance ",
            "title": " Learned covariance ",
            "hypothesis": " Paths reveal useful structure. ",
            "implementation_plan": [" Add path. ", "", " Test it. "],
            "target_rubrics": [{"description": "Score > 1", "type": " Metric "}],
            "notes_for_agent": " Keep it anytime. ",
        },
        cycle_index=3,
    )
    assert normalized["idea_id"] == "cycle_3_covariance"
    assert normalized["implementation_plan"] == ["Add path.", "Test it."]
    assert normalized["target_rubrics"][0]["id"] == "score_1"
    assert normalized["notes_for_agent"] == "Keep it anytime."

    # preserves source_contains verification
    normalized = idea_bundle.normalize_idea_bundle(
        {
            "idea_id": "idea_0001",
            "title": "Verified source evidence",
            "hypothesis": "Source checks prove the requested change exists.",
            "implementation_plan": ["Add a solver."],
            "target_rubrics": [
                {
                    "id": "uses_solver",
                    "description": "The candidate invokes the solver.",
                    "type": "implementation",
                    "verification": {"type": "source_contains", "patterns": ["scipy.optimize.minimize"]},
                }
            ],
            "notes_for_agent": "",
        },
        cycle_index=1,
    )
    verification = normalized["target_rubrics"][0]["verification"]
    assert verification["type"] == "source_contains"
    assert verification["patterns"] == ["scipy.optimize.minimize"]

    # raises for a missing title
    raw = {
        "idea_id": "idea_0001",
        "hypothesis": "test",
        "implementation_plan": ["one step"],
        "target_rubrics": [{"id": "r", "description": "desc", "type": "metric"}],
        "notes_for_agent": "",
    }
    with pytest.raises(idea_bundle.IdeaBundleValidationError):
        idea_bundle.normalize_idea_bundle(raw, cycle_index=1)
