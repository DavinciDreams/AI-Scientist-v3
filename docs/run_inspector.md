# Run Inspector Web UI

## Quick Start

From repo root, with the `.venv` from `uv sync`:

```bash
agsci inspect start
```

This builds the frontend, launches backend + UI at `http://127.0.0.1:8765`, and opens a
browser once `/api/healthz` answers (pass `--no-open` to skip that). An existing
`node_modules` is reused; delete it to refresh dependencies. `--no-build` serves the last
build without touching npm.

## Sections

- **Runs**: every pipeline run under the results root, with per-phase status read
  from its `pipeline_events.jsonl`, links to the inspector and the paper PDF, and
  **Continue…** to run more phases against an existing run directory.
- **Launch**: pick a task from `examples/` (or `--tasks-root`), choose agent, model,
  provider, preset, phases and budgets, see the exact `agsci` command, then run
  **Check setup** (`agsci doctor`), **Dry run**, or **Launch run**.
- **Jobs**: launched jobs with live output, exit status and Interrupt (SIGINT, so
  the pipeline cleans up) or Force kill.
- **Inspector**: the idea tree, performance plot and artifacts of one investigation.

Launching is off by default; the UI is then read-only. Enable it with:

```bash
agsci inspect start --enable-launch --tasks-root tasks
```

Jobs are child processes of the server, run with the `agsci` of the same
interpreter and `paths.results_root` set to `--results-root`. Their records and
output live in `<results-root>/.jobs/<job id>/`. Stopping the server interrupts
running jobs; resume those runs with **Continue…**. Writes and launches are
refused for cross-site requests. Anyone who can reach the port can launch, so
keep `--host 127.0.0.1` unless the server is behind authentication (see
[the Coolify deployment](coolify.md)).

## Modes

```bash
agsci inspect build   # install frontend deps and build the static assets only
agsci inspect start   # build, then serve API + built frontend (default)
agsci inspect dev     # API with autoreload next to the Vite dev server
```

In dev mode:

- frontend: `http://127.0.0.1:5173`
- backend API: `http://127.0.0.1:8765`

## Flags

Every setting is a flag with a literal default; nothing is read from the environment.

- `--results-root` (default `results`)
- `--host` (default `127.0.0.1`)
- `--backend-port` (default `8765`)
- `--frontend-port` (default `5173`, dev mode only)
- `--agentlens-base-url` (default `http://127.0.0.1:8787`)
- `--enable-launch` (let the UI launch jobs and create task briefs)
- `--tasks-root` (writable directory for task briefs created in the UI; default none)
- `--no-open` (never open a browser)
- `--no-build` (start mode: serve the existing `frontend/dist` without rebuilding)
- `--open-wait-seconds` (default `60`, start mode only)

The backend runs under the interpreter that runs `agsci`, so the `.venv` Python is used.

## Manual Launch (without `agsci`)

```bash
cd apps/run_inspector/frontend
npm install
npm run build

cd /path/to/agentic-scientist
python apps/run_inspector/launcher.py --results-root results --host 127.0.0.1 --port 8765
```
