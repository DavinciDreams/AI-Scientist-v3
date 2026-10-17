"""Single-command launcher for the run inspector web app."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.run_inspector.backend.app import create_app
from apps.run_inspector.backend.control import ControlConfig, TaskRoot, config_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inspector web UI")
    parser.add_argument("--results-root", default="results", help="Root directory containing investigation results")
    parser.add_argument("--host", default="127.0.0.1", help="Host bind address")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    parser.add_argument("--agentlens-base-url", default="http://127.0.0.1:8787", help="AgentLens base URL")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="API-only mode (does not require built frontend assets)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn autoreload",
    )
    parser.add_argument(
        "--enable-launch",
        action="store_true",
        help="Allow the UI to launch `agsci run`/`agsci doctor` jobs and create tasks",
    )
    parser.add_argument(
        "--tasks-root",
        default=None,
        help="Writable directory for task briefs created in the UI (next to examples/)",
    )
    parser.add_argument("--max-running-jobs", type=int, default=2, help="Concurrent job limit")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else (REPO_ROOT / resolved).resolve()


def control_config(args: argparse.Namespace, results_root: Path) -> ControlConfig:
    task_roots = [TaskRoot("examples", REPO_ROOT / "examples")]
    if args.tasks_root:
        task_roots.append(TaskRoot("tasks", _resolve(args.tasks_root), writable=True))
    return ControlConfig(
        results_root=results_root,
        jobs_dir=results_root / ".jobs",
        task_roots=tuple(task_roots),
        command_prefix=(sys.executable, "-m", "src.agsci.main"),
        fixed_overrides=(f"paths.results_root={results_root}",),
        cwd=REPO_ROOT,
        launch_enabled=args.enable_launch,
        max_running_jobs=args.max_running_jobs,
        presets=config_names(REPO_ROOT / "configs" / "preset"),
        providers=config_names(REPO_ROOT / "configs" / "provider"),
    )


def main() -> int:
    args = parse_args()
    results_root = _resolve(args.results_root)

    frontend_dist = None
    if not args.dev:
        candidate = (Path(__file__).resolve().parent / "frontend" / "dist").resolve()
        if candidate.exists():
            frontend_dist = candidate
        else:
            print(
                "[run_inspector] frontend dist missing at "
                f"{candidate}; start with --dev or build frontend (npm install && npm run build)",
                flush=True,
            )

    app = create_app(
        results_root=results_root,
        frontend_dist=frontend_dist,
        agentlens_base_url=args.agentlens_base_url,
        control=control_config(args, results_root),
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
