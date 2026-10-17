"""FastAPI application for investigation run inspection."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .content_loader import load_text_file, resolve_path_under_run
from .control import (
    DEFAULT_PHASES,
    PIPELINE_PHASES,
    VALID_AGENTS,
    ControlConfig,
    ControlError,
    JobManager,
    create_task,
    list_studies,
    list_tasks,
    resolve_study,
)
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

    def run_id_for_path(self, path: Path) -> str | None:
        """The inspector run id whose investigation directory is ``path``, if any."""
        try:
            target = path.resolve()
        except OSError:
            return None
        for run_id, stub in self._stubs_by_id.items():
            if stub.path.resolve() == target:
                return run_id
        return None

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


def _require_same_origin(request: Request) -> None:
    """Refuse cross-site requests to endpoints that write or launch.

    The API allows any CORS origin for reads, so a page on another site could
    otherwise drive a local server that has launching enabled.
    """
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None:
        if fetch_site not in {"same-origin", "none"}:
            raise HTTPException(status_code=403, detail="cross-site request refused")
        return
    origin = request.headers.get("origin")
    if origin is not None and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(status_code=403, detail="cross-site request refused")


def _register_control_routes(app: FastAPI, store: RunStore, control: ControlConfig) -> None:
    jobs = JobManager(control)
    app.state.jobs = jobs
    guarded = [Depends(_require_same_origin)]

    def bad_request(exc: ControlError) -> HTTPException:
        return HTTPException(status_code=400, detail=str(exc))

    def require_launch() -> None:
        if not control.launch_enabled:
            raise HTTPException(status_code=403, detail="Launching is disabled on this server")

    @app.get("/api/control/config")
    def control_config() -> dict[str, Any]:
        return {
            "launch_enabled": control.launch_enabled,
            "can_create_tasks": control.launch_enabled and control.writable_task_root is not None,
            "max_running_jobs": control.max_running_jobs,
            "agents": list(VALID_AGENTS),
            "agent_locked": control.agent_locked,
            "phases": list(PIPELINE_PHASES),
            "default_phases": list(DEFAULT_PHASES),
            "presets": list(control.presets),
            "providers": list(control.providers),
            "reserved_keys": sorted(control.reserved_keys),
        }

    @app.get("/api/tasks")
    def tasks() -> list[dict[str, Any]]:
        return list_tasks(control)

    @app.post("/api/tasks", dependencies=guarded)
    def new_task(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        require_launch()
        try:
            return create_task(control, str(payload.get("name", "")), str(payload.get("brief", "")))
        except ControlError as exc:
            raise bad_request(exc) from exc

    @app.get("/api/studies")
    def studies() -> list[dict[str, Any]]:
        return list_studies(control.results_root, inspector_run_id=store.run_id_for_path)

    @app.get("/api/studies/{study_id:path}/paper.pdf", include_in_schema=False)
    def study_paper(study_id: str) -> FileResponse:
        try:
            study = resolve_study(control.results_root, study_id)
        except ControlError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        pdf = study / "paper" / "paper.pdf"
        try:
            pdf.resolve().relative_to(study)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail="paper not found") from exc
        if not pdf.is_file():
            raise HTTPException(status_code=404, detail="paper not found")
        filename = f"{study.parent.parent.name}-{study.name}.pdf"
        return FileResponse(
            pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    def job_request(payload: dict[str, Any]) -> dict[str, Any]:
        overrides = payload.get("overrides") or []
        if not isinstance(overrides, list) or not all(isinstance(o, str) for o in overrides):
            raise ControlError("overrides must be a list of key=value strings")
        resume = payload.get("resume_study_id")
        return {
            "kind": str(payload.get("kind", "run")),
            "task_path": str(payload.get("task_path", "")),
            "overrides": overrides,
            "dry_run": bool(payload.get("dry_run", False)),
            "resume_study_id": str(resume) if resume else None,
        }

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return jobs.list_jobs()

    @app.post("/api/jobs/preview")
    def preview_job(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            _task, argv = jobs.build_command(**job_request(payload), run_id="<job id>")
        except ControlError as exc:
            raise bad_request(exc) from exc
        return {"argv": argv}

    @app.post("/api/jobs", dependencies=guarded)
    def start_job(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        require_launch()
        try:
            return jobs.start(**job_request(payload))
        except ControlError as exc:
            raise bad_request(exc) from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
        try:
            return jobs.read_log(job_id, offset)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc

    @app.post("/api/jobs/{job_id}/cancel", dependencies=guarded)
    def cancel_job(job_id: str, force: bool = Query(default=False)) -> dict[str, Any]:
        try:
            return jobs.cancel(job_id, force=force)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
        except ControlError as exc:
            raise bad_request(exc) from exc


def create_app(
    *,
    results_root: Path | None = None,
    frontend_dist: Path | None = None,
    agentlens_base_url: str | None = None,
    control: ControlConfig | None = None,
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

    # Without an explicit control config the UI lists studies but cannot launch.
    _register_control_routes(
        app,
        store,
        control
        or ControlConfig(
            results_root=resolved_results_root,
            jobs_dir=resolved_results_root / ".jobs",
            task_roots=(),
        ),
    )

    if resolved_frontend_dist.exists():
        _configure_spa_routes(app, resolved_frontend_dist)

    return app


app = create_app()
