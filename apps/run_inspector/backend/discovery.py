"""Run discovery and normalization for investigation output directories."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

logger = logging.getLogger(__name__)

_SKIP_DIRS: set[str] = {
    ".venv",
    "venv",
    ".env",
    "node_modules",
    "__pycache__",
    ".git",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
}


def _walk_find(root: Path, target_name: str) -> Iterator[Path]:
    """Recursively find files named *target_name*, skipping junk directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in-place so os.walk won't descend into skipped dirs.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if target_name in filenames:
            yield Path(dirpath) / target_name

MARKER_FILES = (
    "outer_state.json",
    "ideas_history.jsonl",
    "attempts_history.jsonl",
    "parallel_summary.json",
)

WORKER_DIR_RE = re.compile(r"^worker_(\d+)$")
ROUND_DIR_RE = re.compile(r"^round_(\d+)$")
IDEA_PREFIX_RE = re.compile(r"^(\d+)_")
ATTEMPT_DIR_RE = re.compile(r"^(\d+)_attempt$")
PROGRAM_DIR_RE = re.compile(r"^program_(\d+)$")
SESSION_ID_RE = re.compile(r'"session_id":"([^"]+)"')


@dataclass(frozen=True)
class SessionHints:
    by_attempt_id: dict[str, str]
    by_idea_id: dict[str, str]


@dataclass(frozen=True)
class ProgramEval:
    program_idx: int
    sequence_idx: int
    attempt_name: str | None
    display_label: str
    dir_path: Path
    metrics: dict[str, Any] | None
    correct: dict[str, Any] | None
    rubric_status: dict[str, Any] | None


@dataclass(frozen=True)
class Idea:
    idea_id: str
    title: str
    dir_name: str
    path: Path
    order_index: int | None
    programs: list[ProgramEval]


@dataclass(frozen=True)
class Worker:
    worker_index: int
    worker_name: str
    path: Path
    synthetic: bool
    ideas: list[Idea]


@dataclass(frozen=True)
class DiscoveredRun:
    run_id: str
    path: Path
    task_name: str
    agent: str
    containerized: bool
    parallel: bool
    workers: list[Worker]
    best_combined_score: float | None
    session_hints: SessionHints
    prune_every_n_cycles: int
    prune_chapters: list[int]

    @property
    def idea_count(self) -> int:
        return sum(len(worker.ideas) for worker in self.workers)

    @property
    def program_eval_count(self) -> int:
        return sum(len(idea.programs) for worker in self.workers for idea in worker.ideas)


def idea_rel_path(run: DiscoveredRun, idea: Idea) -> str:
    """Return the idea path relative to the run root."""
    return idea.path.relative_to(run.path).as_posix()


def idea_node_id(run: DiscoveredRun, worker: Worker, idea: Idea) -> str:
    """Stable, URL-safe node ID for an idea in the inspector graph."""
    rel_path = idea_rel_path(run, idea)
    return f"idea:{worker.worker_index}:{quote(rel_path, safe='')}"


@dataclass
class _WorkerDescriptor:
    worker_index: int
    worker_name: str
    path: Path
    synthetic: bool = False


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _string_value(payload: dict[str, Any] | None, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _extract_index_from_name(name: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.match(name)
    if not match:
        return None
    try:
        parsed = int(match.group(1))
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _has_round_ideas(worker_dir: Path) -> bool:
    """Check if *worker_dir* contains ``round_NNN/ideas/`` subdirectories."""
    for child in worker_dir.iterdir():
        if child.is_dir() and ROUND_DIR_RE.match(child.name) and (child / "ideas").is_dir():
            return True
    return False


def _idea_order_key(idea_dir: Path) -> tuple[int, float, str]:
    order_index = _extract_index_from_name(idea_dir.name, IDEA_PREFIX_RE)
    if order_index is not None:
        return (order_index, idea_dir.stat().st_mtime, idea_dir.name)
    return (10_000_000, idea_dir.stat().st_mtime, idea_dir.name)


def _attempt_order_key(attempt_dir: Path) -> tuple[int, float, str]:
    order_index = _extract_index_from_name(attempt_dir.name, ATTEMPT_DIR_RE)
    if order_index is not None:
        return (order_index, attempt_dir.stat().st_mtime, attempt_dir.name)
    return (10_000_000, attempt_dir.stat().st_mtime, attempt_dir.name)


def _program_dirs_from(source_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for child in source_dir.iterdir():
        if not child.is_dir():
            continue
        match = PROGRAM_DIR_RE.match(child.name)
        if match is None:
            continue
        try:
            idx = int(match.group(1))
        except ValueError:
            continue
        if idx < 0:
            continue
        out.append((idx, child))
    return sorted(out, key=lambda item: item[0])


def _discover_programs_flattened(idea_dir: Path) -> list[ProgramEval]:
    attempt_dirs = [
        child
        for child in idea_dir.iterdir()
        if child.is_dir() and ATTEMPT_DIR_RE.match(child.name)
    ]

    if attempt_dirs:
        source_dirs = sorted(attempt_dirs, key=_attempt_order_key)
    else:
        source_dirs = [idea_dir]

    programs: list[ProgramEval] = []
    sequence_idx = 0
    for source_dir in source_dirs:
        attempt_name = None if source_dir == idea_dir else source_dir.name
        for program_idx, program_dir in _program_dirs_from(source_dir):
            sequence_idx += 1
            metrics = _load_json_dict(program_dir / "metrics.json")
            correct = _load_json_dict(program_dir / "correct.json")
            rubric_status = _load_json_dict(program_dir / "rubric_status.json")
            label = f"program_{program_idx}"
            if attempt_name is not None:
                label = f"{label} ({attempt_name})"
            programs.append(
                ProgramEval(
                    program_idx=program_idx,
                    sequence_idx=sequence_idx,
                    attempt_name=attempt_name,
                    display_label=label,
                    dir_path=program_dir,
                    metrics=metrics,
                    correct=correct,
                    rubric_status=rubric_status,
                )
            )

    return programs


def _discover_ideas_from_root(ideas_root: Path) -> list[Idea]:
    """Discover ideas from a single ``ideas/`` directory."""
    ideas: list[Idea] = []
    idea_dirs = sorted(
        [candidate for candidate in ideas_root.iterdir() if candidate.is_dir()],
        key=_idea_order_key,
    )

    for idea_dir in idea_dirs:
        idea_bundle = _load_json_dict(idea_dir / "idea_bundle.json")
        active_idea = _load_json_dict(idea_dir / "active_idea.json")

        idea_id = (
            _string_value(idea_bundle, "idea_id")
            or _string_value(active_idea, "idea_id")
            or idea_dir.name
        )
        title = (
            _string_value(idea_bundle, "title")
            or _string_value(active_idea, "title")
            or idea_id
        )

        ideas.append(
            Idea(
                idea_id=idea_id,
                title=title,
                dir_name=idea_dir.name,
                path=idea_dir,
                order_index=_extract_index_from_name(idea_dir.name, IDEA_PREFIX_RE),
                programs=_discover_programs_flattened(idea_dir),
            )
        )

    return ideas


def _discover_ideas(worker_dir: Path) -> list[Idea]:
    ideas_root = worker_dir / "ideas"
    if ideas_root.is_dir():
        return _discover_ideas_from_root(ideas_root)

    # Round-based isolation: collect ideas from all round_NNN/ subdirs.
    round_dirs = sorted(
        [
            child
            for child in worker_dir.iterdir()
            if child.is_dir() and ROUND_DIR_RE.match(child.name)
        ],
        key=lambda d: _extract_index_from_name(d.name, ROUND_DIR_RE) or 0,
    )

    all_ideas: list[Idea] = []
    for round_dir in round_dirs:
        round_ideas_root = round_dir / "ideas"
        if round_ideas_root.is_dir():
            all_ideas.extend(_discover_ideas_from_root(round_ideas_root))
    return all_ideas


def _discover_workers(run_dir: Path) -> tuple[list[_WorkerDescriptor], bool]:
    workers: list[_WorkerDescriptor] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        worker_index = _extract_index_from_name(child.name, WORKER_DIR_RE)
        if worker_index is None:
            continue
        if not (child / "ideas").is_dir() and not _has_round_ideas(child):
            continue
        workers.append(
            _WorkerDescriptor(
                worker_index=worker_index,
                worker_name=child.name,
                path=child,
                synthetic=False,
            )
        )

    if workers:
        workers.sort(key=lambda worker: worker.worker_index)
        return workers, True

    if (run_dir / "ideas").is_dir():
        return [
            _WorkerDescriptor(
                worker_index=1,
                worker_name="worker_01",
                path=run_dir,
                synthetic=True,
            )
        ], False

    return [], False


def _contains_markers(run_dir: Path) -> bool:
    return any((run_dir / marker).is_file() for marker in MARKER_FILES)


def _looks_like_run_dir(run_dir: Path) -> bool:
    if (run_dir / "ideas").is_dir():
        return True
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        if WORKER_DIR_RE.match(child.name) and (
            (child / "ideas").is_dir() or _has_round_ideas(child)
        ):
            return True
    return False


def _extract_session_hints(run_dir: Path) -> SessionHints:
    by_attempt_id: dict[str, str] = {}
    by_idea_id: dict[str, str] = {}

    for row in _load_jsonl_dicts(run_dir / "attempts_history.jsonl"):
        output_tail = row.get("output_tail")
        session_id: str | None = None
        if isinstance(output_tail, str):
            match = SESSION_ID_RE.search(output_tail)
            if match:
                session_id = match.group(1)
        if not session_id:
            continue

        attempt_id = row.get("attempt_id")
        if isinstance(attempt_id, str) and attempt_id.strip():
            by_attempt_id[attempt_id] = session_id

        idea_id = row.get("idea_id")
        if isinstance(idea_id, str) and idea_id.strip():
            by_idea_id[idea_id] = session_id

    return SessionHints(by_attempt_id=by_attempt_id, by_idea_id=by_idea_id)


def _extract_prune_info(run_dir: Path) -> tuple[int, list[int]]:
    """Return (prune_every_n_cycles, sorted list of archived chapter indices)."""
    prune_n = 0
    outer = _load_json_dict(run_dir / "outer_state.json")
    if isinstance(outer, dict):
        val = outer.get("prune_every_n_cycles")
        if isinstance(val, int) and val > 0:
            prune_n = val

    pruned_dir = run_dir / "_pruned"
    chapters: list[int] = []
    if pruned_dir.is_dir():
        for child in pruned_dir.iterdir():
            if child.is_dir() and (
                child.name.startswith("chapter_") or child.name.startswith("round_")
            ):
                try:
                    chapters.append(int(child.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue

    chapters = sorted(chapters)

    # Infer prune_every_n from the first archived round when the field is
    # missing from outer_state.json (older runs don't persist it).
    if prune_n == 0 and chapters:
        first_dir = pruned_dir / f"round_{chapters[0]:03d}"
        if not first_dir.is_dir():
            first_dir = pruned_dir / f"chapter_{chapters[0]:03d}"
        history = _load_jsonl_dicts(first_dir / "ideas_history.jsonl")
        cycles: set[int] = set()
        for rec in history:
            ci = rec.get("cycle_index")
            if isinstance(ci, int):
                cycles.add(ci)
        if cycles:
            prune_n = len(cycles)
            logger.info(
                "Inferred prune_every_n_cycles=%d from archived round %d",
                prune_n,
                chapters[0],
            )

    return prune_n, chapters


def _extract_best_score(run_dir: Path) -> float | None:
    outer = _load_json_dict(run_dir / "outer_state.json")
    if isinstance(outer, dict):
        score = outer.get("best_combined_score")
        if isinstance(score, (int, float)):
            return float(score)

    summary = _load_json_dict(run_dir / "parallel_summary.json")
    if isinstance(summary, dict):
        score = summary.get("best_combined_score")
        if isinstance(score, (int, float)):
            return float(score)

    return None


def _infer_containerized(run_dir: Path) -> bool:
    if any("docker" in part.lower() for part in run_dir.parts):
        return True

    checked = 0
    for wrapper_log in _walk_find(run_dir, "wrapper_output.log"):
        if not wrapper_log.is_file():
            continue
        checked += 1
        if checked > 4:
            break

        snippet = wrapper_log.read_text(encoding="utf-8", errors="replace")
        # Back-compat: older runs logged `containerize=true` and/or referenced the legacy
        # `--containerize` flag in output. Newer runs log Hydra-aligned keys.
        if (
            "container.enabled=true" in snippet
            or "containerize=true" in snippet
            or "--containerize" in snippet
        ):
            return True
    return False


def _discover_candidate_run_dirs(results_root: Path) -> list[Path]:
    marker_set = set(MARKER_FILES)
    candidates: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(results_root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        path = Path(dirpath)

        # Once a directory is identified as a run dir, every worker/round/
        # ideas/attempts/programs subdir lives below it — descending further
        # adds thousands of pointless os.walk steps per run.
        if _looks_like_run_dir(path):
            candidates.add(path)
            dirnames[:] = []
            continue

        if marker_set.intersection(filenames):
            candidates.add(path)

    # Promote worker sub-directories to their parent experiment directory.
    # When marker files only exist inside worker_XX/ dirs, each worker appears
    # as a separate candidate.  We want the parent experiment dir instead.
    filtered: set[Path] = set()
    for candidate in candidates:
        # Promote round_NNN/ inside worker_XX/ to the run dir.
        if ROUND_DIR_RE.match(candidate.name):
            parent = candidate.parent
            if WORKER_DIR_RE.match(parent.name):
                grandparent = parent.parent
                if _looks_like_run_dir(grandparent):
                    filtered.add(grandparent)
                    continue

        if WORKER_DIR_RE.match(candidate.name):
            parent = candidate.parent
            # Replace worker candidate with its parent if the parent looks
            # like a multi-worker run dir (i.e. has worker_XX/ideas/ children).
            if _looks_like_run_dir(parent):
                filtered.add(parent)
                continue
        filtered.add(candidate)

    return sorted(filtered)


@dataclass(frozen=True)
class RunStub:
    """Lightweight metadata for a run directory — no deep traversal."""

    run_id: str
    path: Path
    task_name: str
    agent: str
    best_combined_score: float | None


def discover_run_stubs(results_root: Path) -> list[RunStub]:
    """Fast discovery: find run directories and extract only basic metadata.

    This avoids the expensive per-worker/idea/program traversal that
    :func:`discover_single_run` performs.
    """
    root = results_root.expanduser().resolve()
    if not root.exists():
        return []

    stubs: list[RunStub] = []
    for run_dir in _discover_candidate_run_dirs(root):
        if not _looks_like_run_dir(run_dir):
            continue

        rel = run_dir.relative_to(root)
        rel_parts = rel.parts
        if "paper" in rel_parts or any(part.startswith("_merged_parallel_") for part in rel_parts):
            continue
        task_name = rel_parts[0] if rel_parts else run_dir.name
        agent = rel_parts[-1] if rel_parts else run_dir.name

        stubs.append(
            RunStub(
                run_id=rel.as_posix(),
                path=run_dir,
                task_name=task_name,
                agent=agent,
                best_combined_score=_extract_best_score(run_dir),
            )
        )

    stubs.sort(key=lambda s: s.run_id)
    return stubs


def discover_single_run(run_dir: Path, run_id: str, task_name: str, agent: str) -> DiscoveredRun | None:
    """Fully discover a single run directory (workers, ideas, programs, etc.)."""
    worker_descriptors, parallel = _discover_workers(run_dir)
    if not worker_descriptors:
        return None

    workers = [
        Worker(
            worker_index=worker.worker_index,
            worker_name=worker.worker_name,
            path=worker.path,
            synthetic=worker.synthetic,
            ideas=_discover_ideas(worker.path),
        )
        for worker in worker_descriptors
    ]

    prune_n, prune_chapters = _extract_prune_info(run_dir)

    return DiscoveredRun(
        run_id=run_id,
        path=run_dir,
        task_name=task_name,
        agent=agent,
        containerized=_infer_containerized(run_dir),
        parallel=parallel,
        workers=workers,
        best_combined_score=_extract_best_score(run_dir),
        session_hints=_extract_session_hints(run_dir),
        prune_every_n_cycles=prune_n,
        prune_chapters=prune_chapters,
    )


def discover_runs(results_root: Path) -> list[DiscoveredRun]:
    """Discover investigation run directories and normalize them for API consumption."""
    root = results_root.expanduser().resolve()
    if not root.exists():
        return []

    runs: list[DiscoveredRun] = []
    for stub in discover_run_stubs(root):
        run = discover_single_run(stub.path, stub.run_id, stub.task_name, stub.agent)
        if run is not None:
            runs.append(run)

    runs.sort(key=lambda run: run.run_id)
    return runs
