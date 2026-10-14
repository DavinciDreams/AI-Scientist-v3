"""Build graph nodes/edges and detail payloads from discovered runs."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import os
from pathlib import Path
from typing import Any, Literal

from .agentlens_link import build_agentlens_placeholder_url, build_agentlens_trace_file_url
from .discovery import DiscoveredRun, Idea, ProgramEval, Worker, idea_node_id, idea_rel_path
from .models import ArtifactRef, GraphEdge, GraphNode

from src.investigation.program_spec import (
    ProgramSpecError,
    default_program_spec,
    load_program_spec,
    resolve_entrypoint,
)


@dataclass(frozen=True)
class GraphBuildResult:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    details: dict[str, dict[str, Any]]


def _artifact_kind(path: Path) -> Literal["json", "code", "log", "data", "text", "other"]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {
        ".c",
        ".cpp",
        ".css",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".lua",
        ".m",
        ".mm",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".scala",
        ".sh",
        ".swift",
        ".toml",
        ".ts",
        ".tsx",
        ".yaml",
        ".yml",
    }:
        return "code"
    if suffix in {".log", ".md", ".txt"}:
        return "log"
    if suffix in {".npz", ".npy"}:
        return "data"
    return "other"


def _artifact_agentlens_url(
    *,
    path: Path,
    kind: str,
    agentlens_base_url: str | None,
) -> str | None:
    if kind != "log" or agentlens_base_url is None or not path.is_file():
        return None
    return build_agentlens_trace_file_url(
        base_url=agentlens_base_url,
        trace_file_path=path,
    )


def _to_artifact(
    run: DiscoveredRun,
    path: Path,
    name: str,
    *,
    agentlens_base_url: str | None = None,
) -> ArtifactRef:
    exists = path.is_file()
    kind = _artifact_kind(path)
    rel_path = path
    if path.is_absolute() and run.path in path.parents:
        rel_path = path.relative_to(run.path)
    size = path.stat().st_size if exists else None
    return ArtifactRef(
        name=name,
        path=rel_path.as_posix(),
        kind=kind,
        exists=exists,
        size_bytes=size,
        agentlens_url=_artifact_agentlens_url(
            path=path,
            kind=kind,
            agentlens_base_url=agentlens_base_url,
        ),
    )


def _score_from_program(program: ProgramEval) -> float | None:
    score = (program.metrics or {}).get("combined_score")
    return float(score) if isinstance(score, (int, float)) else None


def _correct_from_program(program: ProgramEval) -> bool | None:
    correct = (program.correct or {}).get("correct")
    return bool(correct) if isinstance(correct, bool) else None


def _entrypoint_path(program_dir: Path) -> Path:
    """Best-effort entrypoint resolution for inspector previews."""
    spec = None
    try:
        spec = load_program_spec(program_dir, required=False)
    except ProgramSpecError:
        spec = None

    if spec is not None:
        try:
            return resolve_entrypoint(program_dir, spec)
        except ProgramSpecError:
            return program_dir / "src" / "program.py"

    # Legacy runs.
    legacy = program_dir / "program.py"
    return legacy if legacy.exists() else (program_dir / "src" / "program.py")


def _matches_any(patterns: tuple[str, ...], relpath: str) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatchcase(relpath, pattern):
            return True
    return False


def _should_include_path(
    *,
    relpath: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> bool:
    if include and _matches_any(include, relpath):
        return True
    if exclude and _matches_any(exclude, relpath):
        return False
    return True


def _program_src_artifacts(
    *,
    run: DiscoveredRun,
    program: ProgramEval,
    agentlens_base_url: str,
) -> list[ArtifactRef]:
    src_dir = program.dir_path / "src"
    if not src_dir.is_dir():
        return []

    spec = None
    try:
        spec = load_program_spec(program.dir_path, required=False)
    except ProgramSpecError:
        spec = None
    if spec is None:
        spec = default_program_spec()

    include = spec.copy.include
    exclude = spec.copy.exclude

    artifacts: list[ArtifactRef] = []
    followlinks = False
    for root, dirnames, filenames in os.walk(src_dir, topdown=True, followlinks=followlinks):
        root_path = Path(root)
        rel_root = root_path.relative_to(program.dir_path).as_posix()

        # Avoid traversing huge directories that are excluded (only safe when includes are empty).
        if not include and exclude:
            for dirname in list(dirnames):
                candidate_rel = f"{rel_root}/{dirname}".lstrip("/")
                if _matches_any(exclude, f"{candidate_rel}/") or _matches_any(exclude, candidate_rel):
                    dirnames.remove(dirname)

        dirnames.sort()
        filenames.sort()

        for filename in filenames:
            src_path = root_path / filename
            relpath = f"{rel_root}/{filename}".lstrip("/")
            if not _should_include_path(relpath=relpath, include=include, exclude=exclude):
                continue

            artifacts.append(
                _to_artifact(
                    run,
                    src_path,
                    f"{program.display_label}/{relpath}",
                    agentlens_base_url=agentlens_base_url,
                )
            )

    return artifacts


def _latest_agent_log(idea: Idea) -> Path | None:
    logs = sorted(
        [
            item
            for item in idea.path.iterdir()
            if item.is_file() and item.name.startswith("agent_") and item.suffix == ".log"
        ],
        key=lambda item: item.name,
    )
    if not logs:
        return None
    return logs[-1]


def _idea_generation_roots(worker: Worker) -> list[Path]:
    """Find all ``_idea_generation`` directories for *worker*.

    Checks both the flat layout (``worker_XX/_idea_generation/``) and the
    round-isolation layout (``worker_XX/round_NNN/_idea_generation/``).
    """
    from .discovery import ROUND_DIR_RE

    roots: list[Path] = []
    direct = worker.path / "_idea_generation"
    if direct.is_dir():
        roots.append(direct)

    if worker.path.is_dir():
        for child in sorted(worker.path.iterdir(), key=lambda d: d.name):
            if child.is_dir() and ROUND_DIR_RE.match(child.name):
                round_gen = child / "_idea_generation"
                if round_gen.is_dir():
                    roots.append(round_gen)
    return roots


def _worker_idea_generation_logs(worker: Worker) -> list[Path]:
    logs: list[Path] = []
    for generation_root in _idea_generation_roots(worker):
        cycle_dirs = sorted(
            [item for item in generation_root.iterdir() if item.is_dir()],
            key=lambda item: item.name,
        )
        for cycle_dir in cycle_dirs:
            cycle_logs = sorted(
                [
                    item
                    for item in cycle_dir.iterdir()
                    if item.is_file() and item.name.startswith("idea_agent_") and item.suffix == ".log"
                ],
                key=lambda item: item.name,
            )
            logs.extend(cycle_logs)
    return logs


def _matching_idea_generation_log(worker: Worker, idea: Idea) -> Path | None:
    if idea.order_index is None:
        return None

    cycle_prefix = f"cycle_{idea.order_index:04d}"

    for generation_root in _idea_generation_roots(worker):
        cycle_dirs = sorted(
            [
                item
                for item in generation_root.iterdir()
                if item.is_dir() and (item.name == cycle_prefix or item.name.startswith(f"{cycle_prefix}_"))
            ],
            key=lambda item: item.name,
        )
        if not cycle_dirs:
            continue

        cycle_dir = cycle_dirs[-1]
        logs = sorted(
            [
                item
                for item in cycle_dir.iterdir()
                if item.is_file() and item.name.startswith("idea_agent_") and item.suffix == ".log"
            ],
            key=lambda item: item.name,
        )
        if logs:
            return logs[-1]

    return None


def _session_id_for_program(run: DiscoveredRun, idea: Idea, program: ProgramEval) -> str | None:
    rubric = program.rubric_status or {}
    attempt_id = rubric.get("attempt_id")
    if isinstance(attempt_id, str) and attempt_id in run.session_hints.by_attempt_id:
        return run.session_hints.by_attempt_id[attempt_id]

    if idea.idea_id in run.session_hints.by_idea_id:
        return run.session_hints.by_idea_id[idea.idea_id]

    return None


def _worker_node(worker: Worker) -> GraphNode:
    return GraphNode(
        id=f"worker:{worker.worker_index}",
        type="worker",
        label=worker.worker_name,
        data={
            "worker_index": worker.worker_index,
            "synthetic": worker.synthetic,
            "idea_count": len(worker.ideas),
        },
    )


def _idea_node(run: DiscoveredRun, worker: Worker, idea: Idea) -> GraphNode:
    rel_path = idea_rel_path(run, idea)
    return GraphNode(
        id=idea_node_id(run, worker, idea),
        type="idea",
        label=idea.title,
        data={
            "worker_index": worker.worker_index,
            "worker_name": worker.worker_name,
            "idea_id": idea.idea_id,
            "idea_dir_name": idea.dir_name,
            "idea_path": rel_path,
            "order_index": idea.order_index,
            "program_eval_count": len(idea.programs),
        },
    )


def _program_summary(run: DiscoveredRun, idea: Idea, program: ProgramEval) -> dict[str, Any]:
    def rel(path: Path) -> str:
        return path.relative_to(run.path).as_posix()

    return {
        "program_idx": program.program_idx,
        "sequence_idx": program.sequence_idx,
        "label": program.display_label,
        "attempt_name": program.attempt_name,
        "program_dir": rel(program.dir_path),
        "combined_score": _score_from_program(program),
        "correct": _correct_from_program(program),
        "session_id_hint": _session_id_for_program(run, idea, program),
        "artifacts": {
            "program_spec_toml": rel(program.dir_path / "program_spec.toml"),
            "entrypoint": rel(_entrypoint_path(program.dir_path)),
            "metrics_json": rel(program.dir_path / "metrics.json"),
            "correct_json": rel(program.dir_path / "correct.json"),
            "rubric_status_json": rel(program.dir_path / "rubric_status.json"),
            "stdout_log": rel(program.dir_path / "stdout.log"),
            "stderr_log": rel(program.dir_path / "stderr.log"),
        },
    }


def _worker_details(
    run: DiscoveredRun,
    worker: Worker,
    node: GraphNode,
    agentlens_base_url: str,
) -> dict[str, Any]:
    artifacts: list[ArtifactRef] = []
    idea_generation_logs = _worker_idea_generation_logs(worker)
    if not worker.synthetic:
        artifacts.append(
            _to_artifact(
                run,
                worker.path / "parallel_worker.log",
                "parallel_worker.log",
                agentlens_base_url=agentlens_base_url,
            )
        )
    artifacts.append(
        _to_artifact(
            run,
            worker.path / "latest_run_manifest.json",
            "latest_run_manifest.json",
            agentlens_base_url=agentlens_base_url,
        )
    )
    for idea_generation_log in idea_generation_logs:
        display_name = idea_generation_log.relative_to(worker.path).as_posix()
        artifacts.append(
            _to_artifact(
                run,
                idea_generation_log,
                display_name,
                agentlens_base_url=agentlens_base_url,
            )
        )

    source_log = worker.path / "parallel_worker.log"
    if not source_log.is_file() and idea_generation_logs:
        source_log = idea_generation_logs[-1]
    source_log_rel = source_log.relative_to(run.path).as_posix() if source_log.is_file() else None
    agentlens_url = build_agentlens_placeholder_url(
        base_url=agentlens_base_url,
        run_id=run.run_id,
        node_id=node.id,
        source_log=source_log_rel,
        session_id=None,
    )

    return {
        "node": node,
        "metadata": {
            "worker_index": worker.worker_index,
            "worker_name": worker.worker_name,
            "worker_path": worker.path.relative_to(run.path).as_posix(),
            "synthetic": worker.synthetic,
            "idea_count": len(worker.ideas),
            "program_eval_count": sum(len(idea.programs) for idea in worker.ideas),
        },
        "artifacts": artifacts,
        "agentlens_url": agentlens_url,
    }


def _idea_details(
    run: DiscoveredRun,
    worker: Worker,
    idea: Idea,
    node: GraphNode,
    agentlens_base_url: str,
) -> dict[str, Any]:
    artifacts = [
        _to_artifact(
            run,
            idea.path / "idea_bundle.json",
            "idea_bundle.json",
            agentlens_base_url=agentlens_base_url,
        ),
        _to_artifact(
            run,
            idea.path / "active_idea.json",
            "active_idea.json",
            agentlens_base_url=agentlens_base_url,
        ),
        _to_artifact(
            run,
            idea.path / "active_rubrics.json",
            "active_rubrics.json",
            agentlens_base_url=agentlens_base_url,
        ),
        _to_artifact(
            run,
            idea.path / "target_rubrics.json",
            "target_rubrics.json",
            agentlens_base_url=agentlens_base_url,
        ),
        _to_artifact(
            run,
            idea.path / "rubric_status.json",
            "rubric_status.json",
            agentlens_base_url=agentlens_base_url,
        ),
        _to_artifact(
            run,
            idea.path / "prompt.md",
            "prompt.md",
            agentlens_base_url=agentlens_base_url,
        ),
        _to_artifact(
            run,
            idea.path / "task.md",
            "task.md",
            agentlens_base_url=agentlens_base_url,
        ),
        _to_artifact(
            run,
            idea.path / "memory.md",
            "memory.md",
            agentlens_base_url=agentlens_base_url,
        ),
        _to_artifact(
            run,
            idea.path / "wrapper_output.log",
            "wrapper_output.log",
            agentlens_base_url=agentlens_base_url,
        ),
        _to_artifact(
            run,
            idea.path / "idea_generation_wrapper_output.log",
            "idea_generation_wrapper_output.log",
            agentlens_base_url=agentlens_base_url,
        ),
    ]

    for program in idea.programs:
        artifacts.append(
            _to_artifact(
                run,
                program.dir_path / "program_spec.toml",
                f"{program.display_label}/program_spec.toml",
                agentlens_base_url=agentlens_base_url,
            )
        )
        artifacts.extend(
            _program_src_artifacts(
                run=run,
                program=program,
                agentlens_base_url=agentlens_base_url,
            )
        )

        entrypoint = _entrypoint_path(program.dir_path)
        try:
            entrypoint.relative_to(program.dir_path / "src")
        except ValueError:
            artifacts.append(
                _to_artifact(
                    run,
                    entrypoint,
                    f"{program.display_label}/entrypoint{entrypoint.suffix}",
                    agentlens_base_url=agentlens_base_url,
                )
            )

    agent_log = _latest_agent_log(idea)
    idea_generation_log = _matching_idea_generation_log(worker, idea)
    if idea_generation_log is not None:
        display_name = idea_generation_log.relative_to(worker.path).as_posix()
        artifacts.append(
            _to_artifact(
                run,
                idea_generation_log,
                display_name,
                agentlens_base_url=agentlens_base_url,
            )
        )
    if agent_log is not None:
        artifacts.append(
            _to_artifact(
                run,
                agent_log,
                agent_log.name,
                agentlens_base_url=agentlens_base_url,
            )
        )

    source_log_path = idea_generation_log or agent_log
    source_log = source_log_path.relative_to(run.path).as_posix() if source_log_path is not None else None
    session_id = run.session_hints.by_idea_id.get(idea.idea_id)
    agentlens_url = build_agentlens_placeholder_url(
        base_url=agentlens_base_url,
        run_id=run.run_id,
        node_id=node.id,
        source_log=source_log,
        session_id=session_id,
    )

    return {
        "node": node,
        "metadata": {
            "worker_index": worker.worker_index,
            "worker_name": worker.worker_name,
            "idea_node_id": node.id,
            "idea_id": idea.idea_id,
            "title": idea.title,
            "idea_dir_name": idea.dir_name,
            "idea_path": idea_rel_path(run, idea),
            "program_eval_count": len(idea.programs),
            "programs": [_program_summary(run, idea, program) for program in idea.programs],
            "session_id_hint": session_id,
        },
        "artifacts": artifacts,
        "agentlens_url": agentlens_url,
    }


def build_graph(run: DiscoveredRun, *, agentlens_base_url: str) -> GraphBuildResult:
    """Build worker/idea graph nodes + detail payloads for one run."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    details: dict[str, dict[str, Any]] = {}

    edge_index = 0
    for worker in run.workers:
        worker_node = _worker_node(worker)
        nodes.append(worker_node)
        details[worker_node.id] = _worker_details(
            run=run,
            worker=worker,
            node=worker_node,
            agentlens_base_url=agentlens_base_url,
        )

        previous_node_id = worker_node.id
        for idea in worker.ideas:
            idea_node = _idea_node(run, worker, idea)
            nodes.append(idea_node)
            edge_index += 1
            edges.append(
                GraphEdge(
                    id=f"edge:{edge_index}",
                    source=previous_node_id,
                    target=idea_node.id,
                    kind="sequence",
                )
            )
            details[idea_node.id] = _idea_details(
                run=run,
                worker=worker,
                idea=idea,
                node=idea_node,
                agentlens_base_url=agentlens_base_url,
            )
            previous_node_id = idea_node.id

    return GraphBuildResult(nodes=nodes, edges=edges, details=details)
