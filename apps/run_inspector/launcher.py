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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    results_root = Path(args.results_root).expanduser()
    if not results_root.is_absolute():
        results_root = (repo_root / results_root).resolve()

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
