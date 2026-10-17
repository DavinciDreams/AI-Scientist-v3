"""Control-plane support for the web UI: tasks, study status, and launched jobs.

The inspector itself stays read-only. Everything here that writes (creating a
task brief, launching ``agsci run``/``agsci doctor``, cancelling a job) is only
reachable when the app is created with ``ControlConfig(launch_enabled=True)``.

Jobs are plain child processes of the server. Each one owns a directory under
``jobs_dir`` holding ``job.json`` (metadata, rewritten on every state change)
and ``output.log`` (combined stdout/stderr). A server restart cannot re-attach
to a child, so jobs left ``running`` by a previous server are reported as
``interrupted``; their run envelopes can be continued with ``paths.run_dir``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Mirrors src.pipeline.config.PIPELINE_PHASES (plus the optional review and
# revision phases) and src.investigation.config.VALID_AGENTS. The inspector
# backend does not import the pipeline; tests keep these lists in sync.
PIPELINE_PHASES = (
    "task_drafting",
    "investigation",
    "ablations",
    "final_evaluation",
    "posthoc",
    "paper",
    "review",
    "revision",
)
DEFAULT_PHASES = PIPELINE_PHASES[:6]
VALID_AGENTS = ("claude", "codex", "cursor", "opencode", "gemini", "pi")

JOB_KINDS = ("run", "doctor")
# Keys the control plane derives itself; letting a request set them would
# break the job -> run-envelope link or escape the configured results root.
BASE_RESERVED_KEYS = frozenset({"task.dir", "run_id", "paths.run_dir", "paths.results_root"})

_OVERRIDE_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z0-9_@-]+)*$")
_TASK_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_OVERRIDES = 64
_MAX_OVERRIDE_LENGTH = 1024
_MAX_BRIEF_BYTES = 200_000
_EXCERPT_CHARS = 600
_EVENTS_TAIL_BYTES = 4_000_000
_LOG_CHUNK_BYTES = 256_000


class ControlError(ValueError):
    """A request the control plane refuses; the message is safe to show."""


@dataclass(frozen=True)
class TaskRoot:
    label: str
    path: Path
    writable: bool = False


@dataclass(frozen=True)
class ControlConfig:
    """Where tasks and jobs live, and how a job command is spelled."""

    results_root: Path
    jobs_dir: Path
    task_roots: tuple[TaskRoot, ...]
    # ``agsci`` itself; the subcommand and arguments follow.
    command_prefix: tuple[str, ...] = (sys.executable, "-m", "src.agsci.main")
    # Appended after the request's overrides, e.g. the results root the UI reads.
    fixed_overrides: tuple[str, ...] = ()
    cwd: Path | None = None
    launch_enabled: bool = False
    max_running_jobs: int = 2
    reserved_keys: frozenset[str] = BASE_RESERVED_KEYS
    # Shown instead of the agent picker when a wrapper pins the agent.
    agent_locked: str | None = None
    presets: tuple[str, ...] = ()
    providers: tuple[str, ...] = ()
    env_scrub: tuple[str, ...] = ()

    @property
    def writable_task_root(self) -> TaskRoot | None:
        return next((root for root in self.task_roots if root.writable), None)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def config_names(directory: Path) -> tuple[str, ...]:
    """Names of the ``*.yaml`` Hydra options in a config group directory."""
    if not directory.is_dir():
        return ()
    return tuple(sorted(path.stem for path in directory.glob("*.yaml")))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


# --------------------------------------------------------------------------- tasks


def list_tasks(config: ControlConfig) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for root in config.task_roots:
        if not root.path.is_dir():
            continue
        for task_dir in sorted(root.path.iterdir()):
            brief = task_dir / "task.md"
            if task_dir.name.startswith((".", "_")) or not brief.is_file():
                continue
            try:
                text = brief.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            tasks.append(
                {
                    "name": task_dir.name,
                    "path": str(task_dir.resolve()),
                    "source": root.label,
                    # A shipped scaffold carries its evaluator; a bare brief
                    # needs the experimental task-drafting phase.
                    "has_scaffold": (task_dir / "evaluate.py").is_file(),
                    "summary": _brief_summary(text),
                    "excerpt": text[:_EXCERPT_CHARS],
                }
            )
    return tasks


def _brief_summary(text: str) -> str:
    """The first content line of a brief; headings such as ``# Goal`` say little."""
    for line in text.splitlines():
        stripped = line.strip().lstrip("-*").strip()
        if stripped and not line.lstrip().startswith("#"):
            return stripped if len(stripped) <= 160 else stripped[:157].rstrip() + "…"
    return ""


def resolve_task(config: ControlConfig, task_path: str) -> Path:
    candidate = Path(task_path)
    for root in config.task_roots:
        if candidate.is_absolute() and _is_within(candidate, root.path):
            resolved = candidate.resolve()
            if resolved.parent == root.path.resolve() and (resolved / "task.md").is_file():
                return resolved
    raise ControlError(f"Unknown task: {task_path}")


def create_task(config: ControlConfig, name: str, brief: str) -> dict[str, Any]:
    root = config.writable_task_root
    if root is None:
        raise ControlError("No writable task directory is configured")
    if not _TASK_NAME_RE.match(name):
        raise ControlError("Task names use lowercase letters, digits, '-' and '_' (max 64)")
    if not brief.strip():
        raise ControlError("The task brief is empty")
    if len(brief.encode("utf-8")) > _MAX_BRIEF_BYTES:
        raise ControlError("The task brief is too large")
    task_dir = root.path / name
    try:
        root.path.mkdir(parents=True, exist_ok=True)
        task_dir.mkdir()
    except FileExistsError as exc:
        raise ControlError(f"Task {name!r} already exists") from exc
    (task_dir / "task.md").write_text(brief.rstrip() + "\n", encoding="utf-8")
    return next(task for task in list_tasks(config) if task["path"] == str(task_dir.resolve()))


# ------------------------------------------------------------------------- studies


def _read_events(path: Path) -> list[dict[str, Any]]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > _EVENTS_TAIL_BYTES:
                handle.seek(size - _EVENTS_TAIL_BYTES)
                handle.readline()
            raw = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("event"), str):
            events.append(parsed)
    return events


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold a ``pipeline_events.jsonl`` journal into per-phase status.

    A resumed run appends to the same journal, so a phase keeps the status of
    its latest start/end while the overall status follows the last pipeline.
    """
    phases: dict[str, dict[str, Any]] = {}
    selected: list[str] = []
    summary: dict[str, Any] = {
        "status": "unknown",
        "started_at": None,
        "ended_at": None,
        "agent": None,
        "model": None,
        "dry_run": False,
        "outcome": None,
    }
    for record in events:
        event = record["event"]
        raw_fields = record.get("fields")
        fields: dict[str, Any] = raw_fields if isinstance(raw_fields, dict) else {}
        timestamp = record.get("timestamp")
        if event == "pipeline_start":
            selected = [phase for phase in fields.get("phases") or [] if isinstance(phase, str)]
            summary.update(
                status="running",
                started_at=summary["started_at"] or timestamp,
                ended_at=None,
                agent=fields.get("agent.name") or summary["agent"],
                model=fields.get("agent.model") or summary["model"],
                dry_run=bool(fields.get("dry_run")),
                outcome=None,
            )
        elif event in {"phase_start", "phase_end"}:
            name = fields.get("phase_name")
            if not isinstance(name, str):
                continue
            entry = phases.setdefault(name, {"phase": name})
            if event == "phase_start":
                entry.update(status="running", started_at=timestamp, ended_at=None, reason=None)
            else:
                entry.update(
                    status=fields.get("status") or "completed",
                    ended_at=timestamp,
                    elapsed=fields.get("elapsed"),
                    reason=fields.get("reason"),
                    outcome=fields.get("outcome"),
                )
        elif event == "pipeline_end":
            summary.update(
                status=fields.get("status") or "completed",
                ended_at=timestamp,
                outcome=fields.get("outcome"),
            )
            for entry in phases.values():
                if entry.get("status") == "running":
                    entry["status"] = "interrupted"
    order = [phase for phase in PIPELINE_PHASES if phase in selected or phase in phases]
    summary["phases"] = [
        phases.get(phase, {"phase": phase, "status": "pending"}) for phase in order
    ]
    return summary


def _looks_like_envelope(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in ("pipeline_events.jsonl", "investigation", "task", "paper")
    )


def list_studies(
    results_root: Path,
    *,
    inspector_run_id: Callable[[Path], str | None] = lambda _path: None,
) -> list[dict[str, Any]]:
    """Pipeline run envelopes at ``<results_root>/<task>/<agent>/<run_id>``."""
    root = results_root.expanduser()
    studies: list[dict[str, Any]] = []
    if not root.is_dir():
        return studies

    def children(path: Path) -> list[Path]:
        try:
            return [
                child
                for child in path.iterdir()
                if child.is_dir() and not child.name.startswith((".", "_"))
            ]
        except OSError:
            return []

    for task_dir in children(root):
        for agent_dir in children(task_dir):
            for run_dir in children(agent_dir):
                if not _looks_like_envelope(run_dir):
                    continue
                summary = summarize_events(_read_events(run_dir / "pipeline_events.jsonl"))
                if summary["status"] == "unknown" and (run_dir / "paper" / "paper.pdf").is_file():
                    summary["status"] = "completed"
                studies.append(
                    {
                        "study_id": f"{task_dir.name}/{agent_dir.name}/{run_dir.name}",
                        "path": str(run_dir.resolve()),
                        "task_name": task_dir.name,
                        "agent_dir": agent_dir.name,
                        "run_id": run_dir.name,
                        "has_paper": (run_dir / "paper" / "paper.pdf").is_file(),
                        "inspector_run_id": inspector_run_id(run_dir / "investigation")
                        or inspector_run_id(run_dir),
                        "modified_at": datetime.fromtimestamp(
                            run_dir.stat().st_mtime, timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        **summary,
                    }
                )
    studies.sort(key=lambda study: study["started_at"] or study["modified_at"], reverse=True)
    return studies


def resolve_study(results_root: Path, study_id: str) -> Path:
    parts = [part for part in study_id.split("/") if part]
    if len(parts) != 3 or any(part in {".", ".."} or part.startswith(".") for part in parts):
        raise ControlError(f"Unknown study: {study_id}")
    candidate = results_root.joinpath(*parts)
    if not _is_within(candidate, results_root) or not candidate.is_dir():
        raise ControlError(f"Unknown study: {study_id}")
    return candidate.resolve()


# ---------------------------------------------------------------------------- jobs


@dataclass
class Job:
    job_id: str
    kind: str
    task_name: str
    task_path: str
    argv: list[str]
    dry_run: bool
    run_id: str | None
    resume_study_id: str | None
    status: str = "queued"  # queued | running | succeeded | failed | cancelled | interrupted
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    ended_at: str | None = None
    return_code: int | None = None
    pid: int | None = None
    cancel_requested: bool = False
    error: str | None = None

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("pid")
        return payload


def validate_overrides(overrides: list[str], reserved_keys: frozenset[str]) -> list[str]:
    if len(overrides) > _MAX_OVERRIDES:
        raise ControlError(f"At most {_MAX_OVERRIDES} overrides are accepted")
    cleaned: list[str] = []
    for override in overrides:
        override = override.strip()
        if not override:
            continue
        key, sep, value = override.partition("=")
        if not sep or not _OVERRIDE_KEY_RE.match(key):
            raise ControlError(f"Overrides must look like key=value: {override!r}")
        if key.startswith("hydra.") or key in reserved_keys:
            raise ControlError(f"{key} is managed by the control panel")
        if len(override) > _MAX_OVERRIDE_LENGTH or any(c in value for c in "\r\n\0"):
            raise ControlError(f"Invalid value for {key}")
        cleaned.append(override)
    return cleaned


class JobManager:
    def __init__(
        self,
        config: ControlConfig,
        *,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self._config = config
        self._popen = popen
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._load_existing()

    # -- persistence

    def _job_dir(self, job_id: str) -> Path:
        return self._config.jobs_dir / job_id

    def _log_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "output.log"

    def _save(self, job: Job) -> None:
        path = self._job_dir(job.job_id) / "job.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _load_existing(self) -> None:
        if not self._config.jobs_dir.is_dir():
            return
        for meta in self._config.jobs_dir.glob("*/job.json"):
            try:
                job = Job(**json.loads(meta.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                logger.warning("Skipping unreadable job record %s", meta)
                continue
            if job.status in {"queued", "running"}:
                job.status = "interrupted"
                job.ended_at = job.ended_at or _utc_now()
                job.error = "The control server restarted while this job was running."
                try:
                    self._save(job)
                except OSError:
                    pass
            self._jobs[job.job_id] = job

    # -- queries

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)
            return [job.public() for job in jobs]

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job.public()

    def running_count(self) -> int:
        with self._lock:
            return sum(job.status in {"queued", "running"} for job in self._jobs.values())

    def read_log(self, job_id: str, offset: int) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            status = self._jobs[job_id].status
        path = self._log_path(job_id)
        try:
            size = path.stat().st_size
        except OSError:
            return {"offset": 0, "next_offset": 0, "size": 0, "text": "", "status": status}
        offset = max(0, min(offset, size))
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read(_LOG_CHUNK_BYTES)
        return {
            "offset": offset,
            "next_offset": offset + len(chunk),
            "size": size,
            "text": chunk.decode("utf-8", errors="replace"),
            "status": status,
        }

    # -- commands

    def build_command(
        self,
        *,
        kind: str,
        task_path: str,
        overrides: list[str],
        dry_run: bool,
        resume_study_id: str | None,
        run_id: str | None,
    ) -> tuple[Path, list[str]]:
        if kind not in JOB_KINDS:
            raise ControlError(f"Unknown job kind: {kind}")
        task = resolve_task(self._config, task_path)
        argv = [*self._config.command_prefix, kind, str(task)]
        if kind == "run" and dry_run:
            argv.append("--dry-run")
        argv.extend(validate_overrides(overrides, self._config.reserved_keys))
        argv.extend(self._config.fixed_overrides)
        if kind == "run":
            if resume_study_id:
                study = resolve_study(self._config.results_root, resume_study_id)
                argv.append(f"paths.run_dir={study}")
            elif run_id:
                argv.append(f"run_id={run_id}")
        return task, argv

    def start(
        self,
        *,
        kind: str,
        task_path: str,
        overrides: list[str],
        dry_run: bool = False,
        resume_study_id: str | None = None,
    ) -> dict[str, Any]:
        if not self._config.launch_enabled:
            raise ControlError("Launching is disabled on this server")
        job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
        # A fresh run's id is the job id, so the run envelope is findable as
        # <results_root>/<task>/<agent>/<job_id> once the pipeline creates it.
        fresh_run = kind == "run" and not dry_run and not resume_study_id
        task, argv = self.build_command(
            kind=kind,
            task_path=task_path,
            overrides=overrides,
            dry_run=dry_run,
            resume_study_id=resume_study_id,
            run_id=job_id if fresh_run else None,
        )
        with self._lock:
            running = sum(job.status in {"queued", "running"} for job in self._jobs.values())
            if running >= self._config.max_running_jobs:
                raise ControlError(
                    f"{running} job(s) already running; the limit is {self._config.max_running_jobs}"
                )
            job = Job(
                job_id=job_id,
                kind=kind,
                task_name=task.name,
                task_path=str(task),
                argv=argv,
                dry_run=dry_run,
                run_id=job_id if fresh_run else None,
                resume_study_id=resume_study_id,
            )
            self._job_dir(job_id).mkdir(parents=True, exist_ok=False)
            self._jobs[job_id] = job
            self._save(job)
            try:
                process = self._spawn(job)
            except OSError as exc:
                job.status = "failed"
                job.ended_at = _utc_now()
                job.error = f"Could not start: {exc}"
                self._save(job)
                return job.public()
            job.status = "running"
            job.pid = process.pid
            job.started_at = _utc_now()
            self._processes[job_id] = process
            self._save(job)
        threading.Thread(
            target=self._wait, args=(job_id, process), name=f"job-{job_id}", daemon=True
        ).start()
        return job.public()

    def _spawn(self, job: Job) -> subprocess.Popen[bytes]:
        env = {key: value for key, value in os.environ.items() if key not in self._config.env_scrub}
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("NO_COLOR", "1")
        if self._config.cwd is not None:
            env["PYTHONPATH"] = str(self._config.cwd)
        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            # Own process group, so cancelling reaches the phase subprocesses.
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        log = self._log_path(job.job_id).open("ab")
        try:
            return self._popen(
                job.argv,
                cwd=str(self._config.cwd) if self._config.cwd else None,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
        finally:
            log.close()

    def _wait(self, job_id: str, process: subprocess.Popen[bytes]) -> None:
        return_code = process.wait()
        with self._lock:
            job = self._jobs[job_id]
            job.return_code = return_code
            job.ended_at = _utc_now()
            if job.cancel_requested:
                job.status = "cancelled"
            else:
                job.status = "succeeded" if return_code == 0 else "failed"
            self._processes.pop(job_id, None)
            try:
                self._save(job)
            except OSError:
                logger.exception("Could not record the end of job %s", job_id)

    def cancel(self, job_id: str, *, force: bool = False) -> dict[str, Any]:
        """Interrupt a job (SIGINT, so the pipeline cleans up); ``force`` kills it."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            process = self._processes.get(job_id)
            if process is None or job.status != "running":
                raise ControlError("This job is not running")
            job.cancel_requested = True
            self._save(job)
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL if force else signal.SIGINT)
                elif force:
                    process.kill()
                else:
                    process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            except (ProcessLookupError, OSError) as exc:
                logger.warning("Could not signal job %s: %s", job_id, exc)
            return job.public()
