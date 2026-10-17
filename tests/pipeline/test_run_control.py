"""apps.run_inspector.backend.control: tasks, study status, jobs, and their endpoints."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.run_inspector.backend.app import create_app
from apps.run_inspector.backend.control import (
    BASE_RESERVED_KEYS,
    DEFAULT_PHASES,
    PIPELINE_PHASES,
    VALID_AGENTS,
    ControlConfig,
    ControlError,
    JobManager,
    TaskRoot,
    create_task,
    list_studies,
    list_tasks,
    summarize_events,
    validate_overrides,
)

SAME_ORIGIN = {"sec-fetch-site": "same-origin"}


def _event(event: str, timestamp: str, **fields: object) -> dict[str, object]:
    return {"timestamp": timestamp, "source": "pipeline", "event": event, "fields": fields}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _task(root: Path, name: str, brief: str = "# Pack circles\nMaximise radii.\n", *, scaffold: bool = True) -> Path:
    task_dir = root / name
    task_dir.mkdir(parents=True)
    (task_dir / "task.md").write_text(brief, encoding="utf-8")
    if scaffold:
        (task_dir / "evaluate.py").write_text("", encoding="utf-8")
    return task_dir


def _config(tmp_path: Path, *, prefix: tuple[str, ...] | None = None, **extra: object) -> ControlConfig:
    options: dict[str, object] = {
        "results_root": tmp_path / "results",
        "jobs_dir": tmp_path / "results" / ".jobs",
        "task_roots": (
            TaskRoot("examples", tmp_path / "examples"),
            TaskRoot("tasks", tmp_path / "tasks", writable=True),
        ),
        "launch_enabled": True,
        **extra,
    }
    if prefix is not None:
        options["command_prefix"] = prefix
    return ControlConfig(**options)  # type: ignore[arg-type]


def _wait_for(manager: JobManager, job_id: str, *, timeout: float = 30.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {manager.get(job_id)}")


def test_constants_match_the_pipeline() -> None:
    from src.investigation.config import VALID_AGENTS as investigation_agents
    from src.pipeline.config import PIPELINE_PHASES as pipeline_phases, PipelineExecutionConfig

    assert VALID_AGENTS == investigation_agents
    assert set(PIPELINE_PHASES) == set(pipeline_phases) | {"review", "revision"}
    assert ",".join(DEFAULT_PHASES) == PipelineExecutionConfig().phases


def test_agsci_inspect_forwards_launch_flags_to_the_backend() -> None:
    from src.agsci import inspector

    host = inspector.InspectorHost(python="py")
    args = inspector.parse_args(["dev", "--enable-launch", "--tasks-root", "tasks"])
    assert inspector.backend_command(host, args, dev=True)[-5:] == [
        "--enable-launch", "--tasks-root", "tasks", "--dev", "--reload",
    ]
    plain = inspector.backend_command(host, inspector.parse_args([]), dev=False)
    assert "--enable-launch" not in plain and "--tasks-root" not in plain


def test_summarize_events_tracks_phases_across_a_resumed_journal() -> None:
    events = [
        _event("pipeline_start", "t0", phases=["investigation", "ablations", "paper"],
               **{"agent.name": "claude", "agent.model": "opus"}),
        _event("phase_start", "t1", phase_name="investigation"),
        _event("phase_end", "t2", phase_name="investigation", status="completed", elapsed="9.0s"),
        _event("phase_start", "t3", phase_name="ablations"),
        _event("pipeline_end", "t4", status="failed", return_code=1),
        _event("pipeline_start", "t5", phases=["paper"]),
        _event("phase_start", "t6", phase_name="paper"),
    ]
    summary = summarize_events(events)
    assert summary["status"] == "running"
    assert summary["started_at"] == "t0"
    assert summary["agent"] == "claude" and summary["model"] == "opus"
    statuses = {phase["phase"]: phase["status"] for phase in summary["phases"]}
    assert statuses == {"investigation": "completed", "ablations": "interrupted", "paper": "running"}

    finished = summarize_events([*events, _event("pipeline_end", "t7", status="completed_degraded")])
    assert finished["status"] == "completed_degraded"
    assert finished["ended_at"] == "t7"
    assert summarize_events([])["status"] == "unknown"


def test_list_studies_and_paper_endpoint(tmp_path: Path) -> None:
    results = tmp_path / "results"
    study = results / "circpack" / "claude" / "20260101_000000"
    _write_jsonl(study / "pipeline_events.jsonl", [
        _event("pipeline_start", "2026-01-01T00:00:00Z", phases=list(DEFAULT_PHASES)),
        _event("pipeline_end", "2026-01-01T01:00:00Z", status="completed"),
    ])
    (study / "paper").mkdir()
    (study / "paper" / "paper.pdf").write_bytes(b"%PDF-1.4 test")
    (results / ".jobs" / "x" / "y").mkdir(parents=True)  # never a study
    (results / "circpack" / "claude" / "stray").mkdir()  # no envelope markers

    studies = list_studies(results)
    assert [s["study_id"] for s in studies] == ["circpack/claude/20260101_000000"]
    assert studies[0]["status"] == "completed"
    assert studies[0]["has_paper"] is True
    assert [p["status"] for p in studies[0]["phases"]] == ["pending"] * 6

    with TestClient(create_app(results_root=results, frontend_dist=tmp_path / "none")) as client:
        assert client.get("/api/studies").json()[0]["run_id"] == "20260101_000000"
        pdf = client.get("/api/studies/circpack/claude/20260101_000000/paper.pdf")
        assert pdf.status_code == 200
        assert pdf.headers["content-type"] == "application/pdf"
        assert pdf.content.startswith(b"%PDF")
        assert client.get("/api/studies/circpack/claude/../../x/paper.pdf").status_code == 404
        assert client.get("/api/studies/circpack/claude/stray/paper.pdf").status_code == 404


def test_tasks_are_listed_and_created_only_in_the_writable_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _task(tmp_path / "examples", "circpack")
    _task(tmp_path / "examples", "brief_only", "Just a brief.\n", scaffold=False)
    (tmp_path / "examples" / "no_brief").mkdir()

    tasks = {task["name"]: task for task in list_tasks(config)}
    assert set(tasks) == {"circpack", "brief_only"}
    assert tasks["circpack"]["summary"] == "Maximise radii."
    assert tasks["brief_only"]["summary"] == "Just a brief."
    assert tasks["circpack"]["has_scaffold"] is True
    assert tasks["brief_only"]["has_scaffold"] is False

    created = create_task(config, "my-task", "# Mine\nBody")
    assert created["source"] == "tasks"
    assert (tmp_path / "tasks" / "my-task" / "task.md").read_text(encoding="utf-8") == "# Mine\nBody\n"
    for name, brief in (("my-task", "again"), ("../escape", "x"), ("Upper", "x"), ("ok", "  ")):
        with pytest.raises(ControlError):
            create_task(config, name, brief)

    read_only = _config(tmp_path, task_roots=(TaskRoot("examples", tmp_path / "examples"),))
    with pytest.raises(ControlError, match="writable"):
        create_task(read_only, "other", "x")


def test_overrides_are_validated() -> None:
    reserved = BASE_RESERVED_KEYS | {"provider"}
    assert validate_overrides(["agent.name=codex", " ", "execution.phases=a,b"], reserved) == [
        "agent.name=codex",
        "execution.phases=a,b",
    ]
    for bad in ("--dry-run", "noequals", "+extra=1", "hydra.run.dir=x", "run_id=x",
                "paths.run_dir=/", "provider=other", "a.b=line\nbreak"):
        with pytest.raises(ControlError):
            validate_overrides([bad], reserved)


def test_build_command_for_run_doctor_and_resume(tmp_path: Path) -> None:
    task = _task(tmp_path / "examples", "circpack")
    (tmp_path / "results" / "circpack" / "pi" / "r1").mkdir(parents=True)
    manager = JobManager(_config(tmp_path, prefix=("agsci",), fixed_overrides=("paths.results_root=/r",)))

    _, argv = manager.build_command(kind="run", task_path=str(task), overrides=["a=1"],
                                    dry_run=True, resume_study_id=None, run_id=None)
    assert argv == ["agsci", "run", str(task.resolve()), "--dry-run", "a=1", "paths.results_root=/r"]

    _, argv = manager.build_command(kind="doctor", task_path=str(task), overrides=[],
                                    dry_run=True, resume_study_id=None, run_id="ignored")
    assert argv == ["agsci", "doctor", str(task.resolve()), "paths.results_root=/r"]

    _, argv = manager.build_command(kind="run", task_path=str(task), overrides=[], dry_run=False,
                                    resume_study_id="circpack/pi/r1", run_id="ignored")
    assert argv[-1] == f"paths.run_dir={(tmp_path / 'results' / 'circpack' / 'pi' / 'r1').resolve()}"

    for kwargs in (
        {"kind": "shell", "task_path": str(task)},
        {"kind": "run", "task_path": str(tmp_path)},
        {"kind": "run", "task_path": "examples/circpack"},
        {"kind": "run", "task_path": str(task), "resume_study_id": "circpack/pi/missing"},
    ):
        with pytest.raises(ControlError):
            manager.build_command(**{"overrides": [], "dry_run": False, "resume_study_id": None,
                                     "run_id": None, **kwargs})  # type: ignore[arg-type]


def test_job_runs_logs_and_records_its_run_id(tmp_path: Path) -> None:
    task = _task(tmp_path / "examples", "circpack")
    script = "import sys; print('argv', ' '.join(sys.argv[1:]), flush=True)"
    manager = JobManager(_config(tmp_path, prefix=(sys.executable, "-c", script)))

    job = manager.start(kind="run", task_path=str(task), overrides=["agent.name=pi"])
    assert job["run_id"] == job["job_id"]
    finished = _wait_for(manager, job["job_id"])
    assert finished["status"] == "succeeded" and finished["return_code"] == 0

    log = manager.read_log(job["job_id"], 0)
    assert f"argv run {task.resolve()} agent.name=pi run_id={job['job_id']}" in log["text"]
    assert manager.read_log(job["job_id"], log["next_offset"])["text"] == ""

    dry = manager.start(kind="run", task_path=str(task), overrides=[], dry_run=True)
    assert dry["run_id"] is None
    assert _wait_for(manager, dry["job_id"])["status"] == "succeeded"

    # A new manager (server restart) reloads finished jobs from disk.
    assert {j["job_id"] for j in JobManager(manager._config).list_jobs()} == {job["job_id"], dry["job_id"]}


def test_failed_cancelled_limited_and_interrupted_jobs(tmp_path: Path) -> None:
    task = _task(tmp_path / "examples", "circpack")
    failing = JobManager(_config(tmp_path / "a", prefix=(sys.executable, "-c", "raise SystemExit(3)"),
                                 task_roots=(TaskRoot("examples", tmp_path / "examples"),)))
    job = failing.start(kind="doctor", task_path=str(task), overrides=[])
    assert _wait_for(failing, job["job_id"])["return_code"] == 3
    assert failing.get(job["job_id"])["status"] == "failed"

    sleeper = JobManager(_config(tmp_path / "b", prefix=(sys.executable, "-c", "import time; time.sleep(60)"),
                                 task_roots=(TaskRoot("examples", tmp_path / "examples"),),
                                 max_running_jobs=1))
    job = sleeper.start(kind="run", task_path=str(task), overrides=[])
    with pytest.raises(ControlError, match="limit"):
        sleeper.start(kind="run", task_path=str(task), overrides=[])
    sleeper.cancel(job["job_id"], force=True)
    assert _wait_for(sleeper, job["job_id"])["status"] == "cancelled"
    with pytest.raises(ControlError):
        sleeper.cancel(job["job_id"])

    jobs_dir = tmp_path / "c" / "jobs"
    (jobs_dir / "old").mkdir(parents=True)
    (jobs_dir / "old" / "job.json").write_text(json.dumps({
        "job_id": "old", "kind": "run", "task_name": "t", "task_path": "/t", "argv": [],
        "dry_run": False, "run_id": "old", "resume_study_id": None, "status": "running", "pid": 1,
    }), encoding="utf-8")
    restarted = JobManager(_config(tmp_path / "c", jobs_dir=jobs_dir))
    assert restarted.get("old")["status"] == "interrupted"
    assert json.loads((jobs_dir / "old" / "job.json").read_text(encoding="utf-8"))["status"] == "interrupted"


def test_launch_endpoints_are_gated_and_same_origin_only(tmp_path: Path) -> None:
    task = _task(tmp_path / "examples", "circpack")
    script = "print('hello from job', flush=True)"
    enabled = _config(tmp_path, prefix=(sys.executable, "-c", script))
    app = create_app(results_root=tmp_path / "results", frontend_dist=tmp_path / "none", control=enabled)
    body = {"kind": "run", "task_path": str(task), "overrides": ["agent.name=pi"]}
    with TestClient(app) as client:
        config = client.get("/api/control/config").json()
        assert config["launch_enabled"] is True and config["can_create_tasks"] is True
        assert [t["name"] for t in client.get("/api/tasks").json()] == ["circpack"]

        preview = client.post("/api/jobs/preview", json=body).json()
        assert preview["argv"][-1] == "run_id=<job id>"
        assert client.post("/api/jobs/preview", json={**body, "overrides": ["bad"]}).status_code == 400

        assert client.post("/api/jobs", json=body, headers={"sec-fetch-site": "cross-site"}).status_code == 403
        assert client.post("/api/jobs", json=body, headers={"origin": "https://evil.example"}).status_code == 403
        created = client.post("/api/jobs", json=body, headers=SAME_ORIGIN)
        assert created.status_code == 200, created.text
        job_id = created.json()["job_id"]
        _wait_for(app.state.jobs, job_id)
        assert "hello from job" in client.get(f"/api/jobs/{job_id}/log").json()["text"]
        assert client.get("/api/jobs").json()[0]["job_id"] == job_id
        assert client.get("/api/jobs/missing").status_code == 404
        assert client.post(f"/api/jobs/{job_id}/cancel", headers=SAME_ORIGIN).status_code == 400

        made = client.post("/api/tasks", json={"name": "fresh", "brief": "# Fresh"}, headers=SAME_ORIGIN)
        assert made.status_code == 200 and made.json()["name"] == "fresh"

    disabled = create_app(results_root=tmp_path / "results", frontend_dist=tmp_path / "none",
                          control=_config(tmp_path, launch_enabled=False))
    with TestClient(disabled) as client:
        assert client.get("/api/control/config").json()["launch_enabled"] is False
        assert client.post("/api/jobs", json=body, headers=SAME_ORIGIN).status_code == 403
        assert client.post("/api/tasks", json={"name": "x", "brief": "y"}, headers=SAME_ORIGIN).status_code == 403
