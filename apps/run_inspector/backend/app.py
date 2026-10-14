"""FastAPI application for investigation run inspection."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Any

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .content_loader import load_text_file, resolve_path_under_run
from .discovery import DiscoveredRun, RunStub, discover_run_stubs, discover_single_run
from .graph_builder import build_graph
from .models import (
    FileContent,
    IdeaAncestors,
    IdeaTimeline,
    NodeDetail,
    ProgramDirDiff,
    ProgramDiff,
    RunGraph,
    RunPerformance,
    RunSummary,
)
from .performance_builder import build_run_performance
from .views_builder import (
    build_idea_ancestors,
    build_idea_timeline,
    build_program_dir_diff,
    build_program_diff,
)

_DEFAULT_RESULTS_ROOT = Path.cwd() / "results"
_DEFAULT_AGENTLENS_BASE_URL = "http://127.0.0.1:8787"


@dataclass(frozen=True)
class PreparedRun:
    run: DiscoveredRun
    summary: RunSummary
    graph: RunGraph
    performance: RunPerformance
    details: dict[str, NodeDetail]
    idea_timeline: IdeaTimeline


class RunStore:
    """In-memory cache with lazy per-run loading.

    On ``refresh()`` only lightweight run stubs are discovered (fast).
    Full per-run data (graph, performance, details, timeline) is built
    on first access and cached.
    """

    def __init__(self, *, results_root: Path, agentlens_base_url: str) -> None:
        self._results_root = results_root
        self._agentlens_base_url = agentlens_base_url
        self._stubs_by_id: dict[str, RunStub] = {}
        self._prepared_by_id: dict[str, PreparedRun] = {}

    def refresh(self) -> None:
        t0 = time.monotonic()
        self._stubs_by_id = {}
        self._prepared_by_id = {}
        for stub in discover_run_stubs(self._results_root):
            self._stubs_by_id[stub.run_id] = stub
        logger.info("Discovered %d run(s) in %.2fs (lightweight scan)", len(self._stubs_by_id), time.monotonic() - t0)

    def _ensure_prepared(self, run_id: str) -> PreparedRun:
        """Lazily discover and prepare a single run on first access."""
        prepared = self._prepared_by_id.get(run_id)
        if prepared is not None:
            return prepared

        stub = self._stubs_by_id.get(run_id)
        if stub is None:
            raise KeyError(run_id)

        logger.info("Preparing run %s (first access)...", run_id)
        t0 = time.monotonic()
        run = discover_single_run(stub.path, stub.run_id, stub.task_name, stub.agent)
        if run is None:
            raise KeyError(run_id)

        summary = RunSummary(
            run_id=run.run_id,
            path=str(run.path),
            task_name=run.task_name,
            agent=run.agent,
            containerized=run.containerized,
            parallel=run.parallel,
            worker_count=len(run.workers),
            idea_count=run.idea_count,
            program_eval_count=run.program_eval_count,
            best_combined_score=run.best_combined_score,
        )

        graph_result = build_graph(run, agentlens_base_url=self._agentlens_base_url)
        graph = RunGraph(run=summary, nodes=graph_result.nodes, edges=graph_result.edges)
        performance = build_run_performance(run)

        details: dict[str, NodeDetail] = {}
        for node_id, payload in graph_result.details.items():
            details[node_id] = NodeDetail(
                run_id=run.run_id,
                node_id=node_id,
                node=payload["node"],
                metadata=payload["metadata"],
                artifacts=payload["artifacts"],
                agentlens_url=payload["agentlens_url"],
            )

        idea_timeline = build_idea_timeline(run)

        prepared = PreparedRun(
            run=run,
            summary=summary,
            graph=graph,
            performance=performance,
            details=details,
            idea_timeline=idea_timeline,
        )
        self._prepared_by_id[run_id] = prepared
        logger.info("Prepared run %s in %.2fs (%d ideas, %d programs)", run_id, time.monotonic() - t0, run.idea_count, run.program_eval_count)
        return prepared

    def list_runs(self) -> list[RunSummary]:
        summaries: list[RunSummary] = []
        for run_id, stub in self._stubs_by_id.items():
            # If already prepared, return the full summary with counts.
            prepared = self._prepared_by_id.get(run_id)
            if prepared is not None:
                summaries.append(prepared.summary)
            else:
                # Lightweight summary — counts not yet known.
                summaries.append(
                    RunSummary(
                        run_id=stub.run_id,
                        path=str(stub.path),
                        task_name=stub.task_name,
                        agent=stub.agent,
                        best_combined_score=stub.best_combined_score,
                    )
                )
        return sorted(summaries, key=lambda s: s.run_id)

    def run_graph(self, run_id: str) -> RunGraph:
        return self._ensure_prepared(run_id).graph

    def node_detail(self, run_id: str, node_id: str) -> NodeDetail:
        prepared = self._ensure_prepared(run_id)
        detail = prepared.details.get(node_id)
        if detail is None:
            raise LookupError(node_id)
        return detail

    def run_performance(self, run_id: str) -> RunPerformance:
        return self._ensure_prepared(run_id).performance

    def idea_timeline(self, run_id: str) -> IdeaTimeline:
        return self._ensure_prepared(run_id).idea_timeline

    def run_path(self, run_id: str) -> Path:
        # Check prepared first, then stubs.
        prepared = self._prepared_by_id.get(run_id)
        if prepared is not None:
            return prepared.run.path
        stub = self._stubs_by_id.get(run_id)
        if stub is not None:
            return stub.path
        raise KeyError(run_id)

    def idea_ancestors(self, run_id: str, idea_id: str, *, worker_name: str | None = None) -> IdeaAncestors:
        prepared = self._ensure_prepared(run_id)
        return build_idea_ancestors(prepared.run, idea_id, worker_name=worker_name)

    def program_diff(self, run_id: str, from_path: str, to_path: str) -> ProgramDiff:
        prepared = self._ensure_prepared(run_id)
        return build_program_diff(prepared.run, from_path, to_path)

    def program_dir_diff(self, run_id: str, from_dir: str, to_dir: str) -> ProgramDirDiff:
        prepared = self._ensure_prepared(run_id)
        return build_program_dir_diff(prepared.run, from_dir, to_dir)


def _frontend_dist_default() -> Path:
    return Path(__file__).resolve().parents[1] / "frontend" / "dist"


def _configure_spa_routes(app: FastAPI, frontend_dist: Path) -> None:
    index_path = frontend_dist / "index.html"
    if not index_path.exists():
        return

    @app.get("/", include_in_schema=False)
    def root_index() -> FileResponse:
        return FileResponse(index_path)

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not found")

        candidate = (frontend_dist / full_path).resolve()
        try:
            candidate.relative_to(frontend_dist)
        except ValueError:
            return FileResponse(index_path)

        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_path)


def create_app(
    *,
    results_root: Path | None = None,
    frontend_dist: Path | None = None,
    agentlens_base_url: str | None = None,
) -> FastAPI:
    resolved_results_root = (
        (results_root or Path(os.environ.get("RUN_INSPECTOR_RESULTS_ROOT", _DEFAULT_RESULTS_ROOT))).expanduser().resolve()
    )
    resolved_frontend_dist = (
        (frontend_dist or Path(os.environ.get("RUN_INSPECTOR_FRONTEND_DIST", _frontend_dist_default()))).expanduser().resolve()
    )
    resolved_agentlens_base_url = agentlens_base_url or os.environ.get(
        "AGENTLENS_BASE_URL",
        _DEFAULT_AGENTLENS_BASE_URL,
    )
    assert resolved_agentlens_base_url is not None

    store = RunStore(
        results_root=resolved_results_root,
        agentlens_base_url=resolved_agentlens_base_url,
    )
    store.refresh()

    app = FastAPI(title="Run Inspector", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "results_root": str(resolved_results_root),
            "frontend_dist": str(resolved_frontend_dist),
        }

    @app.post("/api/refresh")
    def force_refresh() -> dict[str, bool]:
        store.refresh()
        return {"ok": True}

    @app.get("/api/runs", response_model=list[RunSummary])
    def list_runs() -> list[RunSummary]:
        return store.list_runs()

    @app.get("/api/runs/{run_id:path}/graph", response_model=RunGraph)
    def run_graph(run_id: str) -> RunGraph:
        normalized = run_id.strip("/")
        try:
            return store.run_graph(normalized)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {normalized}") from exc

    @app.get("/api/runs/{run_id:path}/nodes/{node_id}", response_model=NodeDetail)
    def node_detail(run_id: str, node_id: str) -> NodeDetail:
        normalized = run_id.strip("/")
        try:
            return store.node_detail(normalized, node_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {normalized}") from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown node_id: {node_id}") from exc

    @app.get("/api/runs/{run_id:path}/performance", response_model=RunPerformance)
    def run_performance(run_id: str) -> RunPerformance:
        normalized = run_id.strip("/")
        try:
            return store.run_performance(normalized)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {normalized}") from exc

    @app.get("/api/runs/{run_id:path}/ideas", response_model=IdeaTimeline)
    def idea_timeline(run_id: str) -> IdeaTimeline:
        normalized = run_id.strip("/")
        try:
            return store.idea_timeline(normalized)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {normalized}") from exc

    @app.get("/api/runs/{run_id:path}/idea-ancestors/{idea_id}", response_model=IdeaAncestors)
    def idea_ancestors(
        run_id: str,
        idea_id: str,
        worker_name: str | None = Query(default=None),
    ) -> IdeaAncestors:
        normalized = run_id.strip("/")
        try:
            return store.idea_ancestors(normalized, idea_id, worker_name=worker_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {normalized}") from exc

    @app.get("/api/runs/{run_id:path}/program-diff", response_model=ProgramDiff)
    def program_diff(
        run_id: str,
        from_path: str = Query(...),
        to_path: str = Query(...),
    ) -> ProgramDiff:
        normalized = run_id.strip("/")
        try:
            return store.program_diff(normalized, from_path, to_path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {normalized}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id:path}/program-dir-diff", response_model=ProgramDirDiff)
    def program_dir_diff(
        run_id: str,
        from_dir: str = Query(...),
        to_dir: str = Query(...),
    ) -> ProgramDirDiff:
        normalized = run_id.strip("/")
        try:
            return store.program_dir_diff(normalized, from_dir, to_dir)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {normalized}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/file", response_model=FileContent)
    def artifact_file(
        run_id: str = Query(...),
        path: str = Query(...),
    ) -> FileContent:
        normalized = run_id.strip("/")
        try:
            run_path = store.run_path(normalized)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {normalized}") from exc

        try:
            target = resolve_path_under_run(run_path, path)
            loaded = load_text_file(target)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IsADirectoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        rel = target.relative_to(run_path).as_posix()
        return FileContent(
            path=rel,
            size_bytes=loaded.size_bytes,
            truncated=loaded.truncated,
            text=loaded.text,
        )

    if resolved_frontend_dist.exists():
        _configure_spa_routes(app, resolved_frontend_dist)

    return app


app = create_app()
