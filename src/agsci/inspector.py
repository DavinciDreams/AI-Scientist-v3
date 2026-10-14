"""``agsci inspect``: build and serve the run inspector web UI.

Modes mirror the former ``run_inspector.sh``:

* ``build``  installs frontend dependencies and builds the static assets;
* ``start``  builds (unless ``--no-build``), then serves the API and built
  frontend from one backend;
* ``dev``    serves the API with autoreload next to the Vite dev server.

Every setting is a flag with a literal default; nothing is read from the
environment. The backend is the same ``apps/run_inspector/launcher.py`` the
manual instructions use, run under the interpreter that runs ``agsci``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "apps" / "run_inspector" / "frontend"
LAUNCHER_PATH = REPO_ROOT / "apps" / "run_inspector" / "launcher.py"
MODES = ("build", "start", "dev")
BACKEND_STOP_TIMEOUT_SECONDS = 10.0


def _probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _open_browser(url: str) -> None:
    if not webbrowser.open(url):
        _log(f"no browser opener found; open manually: {url}")


def _log(message: str) -> None:
    print(f"[agsci inspect] {message}", flush=True)


@dataclass(slots=True)
class InspectorHost:
    """Process and network primitives, replaceable in tests."""

    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen
    which: Callable[[str], str | None] = shutil.which
    python: str = sys.executable
    probe: Callable[[str], bool] = _probe
    open_browser: Callable[[str], None] = _open_browser
    sleep: Callable[[float], None] = time.sleep
    frontend_dir: Path = FRONTEND_DIR
    launcher_path: Path = LAUNCHER_PATH
    repo_root: Path = REPO_ROOT
    threads: list[threading.Thread] = field(default_factory=list)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agsci inspect",
        description="Build and serve the run inspector web UI.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=MODES,
        default="start",
        help="build: frontend assets only; start: build then serve (default); "
        "dev: API with reload plus the Vite dev server",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    parser.add_argument(
        "--backend-port", type=int, default=8765, help="API port (default: %(default)s)"
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=5173,
        help="Vite dev-server port, dev mode only (default: %(default)s)",
    )
    parser.add_argument(
        "--results-root", default="results", help="run results directory (default: %(default)s)"
    )
    parser.add_argument(
        "--agentlens-base-url",
        default="http://127.0.0.1:8787",
        help="AgentLens base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="do not open a browser once the server is up"
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="start mode: serve the existing frontend build instead of rebuilding it",
    )
    parser.add_argument(
        "--open-wait-seconds",
        type=int,
        default=60,
        help="how long start mode waits for the server before giving up on auto-open "
        "(default: %(default)s)",
    )
    return parser.parse_args(argv)


def npm_install_command(frontend_dir: Path) -> list[str]:
    return ["npm", "ci"] if (frontend_dir / "package-lock.json").is_file() else ["npm", "install"]


def backend_command(host: InspectorHost, args: argparse.Namespace, *, dev: bool) -> list[str]:
    command = [
        host.python,
        str(host.launcher_path),
        "--results-root",
        args.results_root,
        "--host",
        args.host,
        "--port",
        str(args.backend_port),
        "--agentlens-base-url",
        args.agentlens_base_url,
    ]
    if dev:
        command.extend(["--dev", "--reload"])
    return command


def install_frontend_deps(host: InspectorHost) -> int:
    """Install once; an existing ``node_modules`` is reused on repeat launches."""
    if (host.frontend_dir / "node_modules").is_dir():
        _log("frontend dependencies already installed (delete node_modules to refresh)")
        return 0
    _log("install frontend dependencies")
    return host.run(npm_install_command(host.frontend_dir), cwd=host.frontend_dir).returncode


def build_frontend(host: InspectorHost) -> int:
    returncode = install_frontend_deps(host)
    if returncode != 0:
        return returncode
    _log("build frontend")
    return host.run(["npm", "run", "build"], cwd=host.frontend_dir).returncode


def open_when_ready(
    host: InspectorHost, *, health_url: str, page_url: str, wait_seconds: int
) -> None:
    """Open the page from a background thread once the health endpoint answers."""

    def wait() -> None:
        for _ in range(max(wait_seconds, 1)):
            if host.probe(health_url):
                _log(f"auto-open browser: {page_url}")
                host.open_browser(page_url)
                return
            host.sleep(1.0)
        _log(f"timed out waiting for server; open manually: {page_url}")

    thread = threading.Thread(target=wait, name="agsci-inspect-open", daemon=True)
    host.threads.append(thread)
    thread.start()


def run_start(host: InspectorHost, args: argparse.Namespace) -> int:
    if args.no_build:
        if not (host.frontend_dir / "dist" / "index.html").is_file():
            print(
                "agsci inspect: --no-build but no frontend build exists; run: agsci inspect build",
                file=sys.stderr,
            )
            return 1
    else:
        returncode = build_frontend(host)
        if returncode != 0:
            return returncode
    base_url = f"http://{args.host}:{args.backend_port}"
    if not args.no_open:
        open_when_ready(
            host,
            health_url=f"{base_url}/api/healthz",
            page_url=f"{base_url}/",
            wait_seconds=args.open_wait_seconds,
        )
    _log(f"launch backend + built frontend on {base_url}")
    return host.run(backend_command(host, args, dev=False), cwd=host.repo_root).returncode


def run_dev(host: InspectorHost, args: argparse.Namespace) -> int:
    returncode = install_frontend_deps(host)
    if returncode != 0:
        return returncode
    _log(f"launch backend API in dev mode on http://{args.host}:{args.backend_port}")
    backend = host.popen(backend_command(host, args, dev=True), cwd=host.repo_root)
    try:
        frontend = ["npm", "run", "dev", "--", "--host", args.host, "--port", str(args.frontend_port)]
        if not args.no_open:
            frontend.append("--open")
        _log(f"launch frontend dev server on http://{args.host}:{args.frontend_port}")
        return host.run(frontend, cwd=host.frontend_dir).returncode
    finally:
        if backend.poll() is None:
            _log(f"stop backend pid={backend.pid}")
            backend.terminate()
            try:
                backend.wait(timeout=BACKEND_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                backend.kill()
                backend.wait()


def main(argv: list[str] | None = None, *, host: InspectorHost | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    host = host or InspectorHost()
    if host.which("npm") is None:
        print("agsci inspect: npm is required (install Node.js)", file=sys.stderr)
        return 1
    if not host.launcher_path.is_file():
        print(f"agsci inspect: launcher missing at {host.launcher_path}", file=sys.stderr)
        return 1
    try:
        if args.mode == "build":
            return build_frontend(host)
        if args.mode == "start":
            return run_start(host, args)
        return run_dev(host, args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
