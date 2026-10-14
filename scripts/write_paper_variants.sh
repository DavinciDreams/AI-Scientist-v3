#!/bin/bash
# Generic paper-writing wrapper for a variants-pipeline output bundle.
#
# Replaces the per-task write_paper_<task>_variants_v*.sh scripts: the task name
# is taken from $TASK_NAME or auto-detected from aggregated/<task>/run_summary.csv,
# so there is no hardcoded task. It creates the flat top-level aggregated symlinks
# and shells into the generic engine src/scripts/paper.sh.
#
# Required:
#   SRC_RESULTS_DIR   variants output dir (contains aggregated/<task>/ and a task bundle)
# Optional (with defaults):
#   TASK_NAME         auto-detected from aggregated/*/run_summary.csv if unset
#   TASK_DIR          task bundle; auto-detected from task/, _retrofitted_bundle/,
#                     or _extracted_bundle/ if unset
#   OUT_DIR           $SRC_RESULTS_DIR/paper/final_<ts>_standalone_<fmt>
#   PAPER_FORMAT      alife
#   PAPER_MIN_REFS    25
#   PAPER_MAX_REFS    30
#   AGENT             claude
#   AGENT_MODEL       claude-opus-5-5[1m] for Claude; agent default otherwise
#   AGENT_PROFILE     optional Codex CLI profile
#   CONTAINER_ENV_FILE  $REPO_ROOT/.env when present; empty disables
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -z "${SRC_RESULTS_DIR:-}" ]; then
  echo "[write_paper_variants] ERROR: SRC_RESULTS_DIR is required" >&2
  exit 1
fi

PAPER_FORMAT="${PAPER_FORMAT:-alife}"
PAPER_MIN_REFS="${PAPER_MIN_REFS:-25}"
PAPER_MAX_REFS="${PAPER_MAX_REFS:-30}"
AGENT="${AGENT:-claude}"
if [ "${AGENT_MODEL+x}" != x ]; then
  if [ "$AGENT" = "claude" ]; then
    AGENT_MODEL="claude-opus-5-5[1m]"
  else
    AGENT_MODEL=""
  fi
fi
AGENT_PROFILE="${AGENT_PROFILE:-}"
CONTAINER_ENV_FILE_WAS_SET="${CONTAINER_ENV_FILE+x}"
if [ "${CONTAINER_ENV_FILE+x}" != x ]; then
  if [ -f "$REPO_ROOT/.env" ]; then
    CONTAINER_ENV_FILE="$REPO_ROOT/.env"
  else
    CONTAINER_ENV_FILE=""
  fi
fi
if [ -n "$CONTAINER_ENV_FILE" ] && [ ! -f "$CONTAINER_ENV_FILE" ]; then
  echo "[write_paper_variants] ERROR: container env file does not exist: $CONTAINER_ENV_FILE" >&2
  exit 1
fi
OUT_DIR="${OUT_DIR:-$SRC_RESULTS_DIR/paper/final_$(date +%Y%m%d_%H%M%S)_standalone_${PAPER_FORMAT}}"

# Resolve through no-follow directory descriptors so a redirected aggregate root
# cannot influence task selection or later link publication.
TASK_NAME="$($PYTHON_BIN - "$REPO_ROOT" "$SRC_RESULTS_DIR/aggregated" "${TASK_NAME:-}" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
aggregated_root = Path(sys.argv[2])
requested_task_name = sys.argv[3] or None
sys.path.insert(0, str(repo_root))

from src.pipeline.glue import resolve_aggregated_task_name  # noqa: E402

try:
    print(resolve_aggregated_task_name(aggregated_root, requested_task_name))
except (OSError, RuntimeError) as exc:
    print(f"[write_paper_variants] ERROR: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
)"

is_usable_task_bundle() {
  [ -d "$1" ] || return 1
  "$PYTHON_BIN" - "$REPO_ROOT" "$1" >/dev/null 2>&1 <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
bundle_dir = Path(sys.argv[2])
sys.path.insert(0, str(repo_root))

from src.pipeline.safe_fs import validate_tree_symlinks  # noqa: E402
from src.investigation.program_spec import entrypoint_relpath_from_spec  # noqa: E402

try:
    validate_tree_symlinks(bundle_dir, role="paper task bundle")
except (OSError, RuntimeError):
    raise SystemExit(1)

if not (bundle_dir / "task.md").is_file():
    raise SystemExit(1)
if not (bundle_dir / "evaluate.py").is_file():
    raise SystemExit(1)
program_dir = bundle_dir / "program_0"
if not program_dir.is_dir():
    raise SystemExit(1)
if entrypoint_relpath_from_spec(program_dir, required=True) is None:
    raise SystemExit(1)
PY
}

if [ -n "${TASK_DIR:-}" ]; then
  if [ ! -d "$TASK_DIR" ]; then
    echo "[write_paper_variants] ERROR: TASK_DIR is not a directory: $TASK_DIR" >&2
    exit 1
  fi
else
  for candidate in \
    "$SRC_RESULTS_DIR/task" \
    "$SRC_RESULTS_DIR/_retrofitted_bundle" \
    "$SRC_RESULTS_DIR/_extracted_bundle"; do
    if is_usable_task_bundle "$candidate"; then
      TASK_DIR="$candidate"
      break
    fi
    if [ -d "$candidate" ]; then
      echo "[write_paper_variants] WARNING: ignoring incomplete task bundle: $candidate" >&2
    fi
  done
  if [ -z "${TASK_DIR:-}" ]; then
    echo "[write_paper_variants] ERROR: could not find a usable task bundle under" \
         "$SRC_RESULTS_DIR/{task,_retrofitted_bundle,_extracted_bundle}; set TASK_DIR explicitly" >&2
    exit 1
  fi
fi

echo "[write_paper_variants] source results: $SRC_RESULTS_DIR"
echo "[write_paper_variants] task name:      $TASK_NAME"
echo "[write_paper_variants] task dir:       $TASK_DIR"
echo "[write_paper_variants] output dir:     $OUT_DIR"

# Flat top-level links, created relative to a pinned no-follow directory.
"$PYTHON_BIN" - "$REPO_ROOT" "$SRC_RESULTS_DIR/aggregated" "$TASK_NAME" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
aggregated_root = Path(sys.argv[2])
task_name = sys.argv[3]
sys.path.insert(0, str(repo_root))

from src.pipeline.glue import link_aggregated_csvs_in_root  # noqa: E402

try:
    link_aggregated_csvs_in_root(aggregated_root, task_name, dry_run=False)
except (OSError, RuntimeError) as exc:
    print(f"[write_paper_variants] ERROR: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY

paper_args=(
  src/scripts/paper.sh
  --agent "$AGENT"
  --task-dir "$TASK_DIR"
  --results-dir "$SRC_RESULTS_DIR"
  --output-dir "$OUT_DIR"
  --standalone-paper
  --paper-format "$PAPER_FORMAT"
  --paper-min-refs "$PAPER_MIN_REFS"
  --paper-max-refs "$PAPER_MAX_REFS"
  --containerize
  --container-image agentic-scientist/agent-cli:latest
  --container-network host
)
if [ -n "$AGENT_MODEL" ]; then
  paper_args+=(--agent-model "$AGENT_MODEL")
fi
if [ -n "$AGENT_PROFILE" ]; then
  paper_args+=(--agent-profile "$AGENT_PROFILE")
fi
if [ -n "$CONTAINER_ENV_FILE" ] || [ "$CONTAINER_ENV_FILE_WAS_SET" = x ]; then
  paper_args+=(--container-env-file "$CONTAINER_ENV_FILE")
fi
bash "${paper_args[@]}"
