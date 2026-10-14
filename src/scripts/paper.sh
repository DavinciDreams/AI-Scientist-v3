#!/bin/bash
# Usage: bash src/scripts/paper.sh [--agent claude|codex|cursor|opencode|gemini|pi] [--agent-model <name>] [--reasoning-effort low|medium|high|xhigh] [--task-dir <path>] [--results-dir <path>] [--ablations-dir <path>] [--posthoc-dir <path>] [--final-evaluation-dir <path>|--no-final-evaluation] [--final-outcome-policy qualified_only|allow_not_qualified] [--global-memory-dir <path>] [--output-dir <path>] [--manifest-dir <path>] [--resume-existing] [--instruct <path>] [--venue-file <path>] [--contributions-file <path>] [--skills <skill_a,skill_b,...>] [--paper-format <venue>] [--paper-min-refs <n>] [--paper-max-refs <n>] [--paper-max-repair-rounds <n>] [--paper-review-rounds <n>] [--two-stage] [--log-format human|json] [--verbose] [--heartbeat-seconds <n>] [--event-transport direct|wrapper] [--containerize] [--container-image <name>] [--container-network <mode>] [--container-auth-root <path>] [--container-env-file <path>]

set -euo pipefail

export BASH_DEFAULT_TIMEOUT_MS=600000
export BASH_MAX_TIMEOUT_MS=7200000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/container_agent.sh"
source "$SCRIPT_DIR/lib/concept_figure.sh"
source "$SCRIPT_DIR/lib/paper_diagnostics.sh"
source "$SCRIPT_DIR/lib/paper_progress.sh"
source "$SCRIPT_DIR/lib/paper_figure_quality.sh"
source "$SCRIPT_DIR/lib/task_pathing.sh"

AGENT_LOG_MAX_BYTES=16777216
validate_agent_log_max_bytes "$AGENT_LOG_MAX_BYTES" || exit $?

AGENT="claude"
AGENT_MODEL=""
AGENT_PROFILE=""
AGENT_REASONING_EFFORT=""
TASK_DIR="examples/circpack"
RESULTS_DIR=""
ABLATIONS_DIR=""
POSTHOC_DIR=""
FINAL_EVALUATION_DIR=""
FINAL_EVALUATION_MODE="discover"
FINAL_OUTCOME_POLICY="qualified_only"
FINAL_OUTCOME_POLICY_SET=0
CONTRACT_TASK_DIR=""
INVESTIGATION_RESULTS_DIR=""
GLOBAL_MEMORY_DIR=""
OUTPUT_DIR=""
MANIFEST_DIR=""
RESUME_EXISTING=0
INSTRUCT_PATH=""
VENUE_SOURCE_PATH=""
VENUE_SOURCE_EXPLICIT=0
CONTRIBUTIONS_SOURCE_PATH=""
PAPER_SKILLS_CSV=""
PAPER_PROFILE="default"
PAPER_FORMAT="iclr"
PAPER_MIN_REFS=8
PAPER_MAX_REFS=15
PAPER_MAX_REPAIR_ROUNDS=10
PAPER_REVIEW_ROUNDS=0
TWO_STAGE=0
PAPER_LOG_FORMAT="human"
PAPER_LOG_VERBOSE=0
PAPER_HEARTBEAT_SECONDS=60
PAPER_EVENT_TRANSPORT="direct"
CONTAINERIZE=0
CONTAINER_IMAGE="agentic-scientist/agent-cli:latest"
CONTAINER_NETWORK="bridge"
CONTAINER_AUTH_ROOT="$HOME"
CONTAINER_ENV_FILE_SET=""
CONTAINER_ENV_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent|-a)
      AGENT="$2"
      shift 2
      ;;
    --agent-model|--model)
      AGENT_MODEL="$2"
      shift 2
      ;;
    --agent-profile|--profile)
      AGENT_PROFILE="$2"
      shift 2
      ;;
    --reasoning-effort)
      AGENT_REASONING_EFFORT="$2"
      shift 2
      ;;
    --task-dir|-t)
      TASK_DIR="$2"
      shift 2
      ;;
    --results-dir|-r)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --ablations-dir)
      ABLATIONS_DIR="$2"
      shift 2
      ;;
    --posthoc-dir)
      POSTHOC_DIR="$2"
      shift 2
      ;;
    --final-evaluation-dir)
      FINAL_EVALUATION_DIR="$2"
      shift 2
      ;;
    --final-outcome-policy)
      if [ "$#" -lt 2 ] || [ "$FINAL_OUTCOME_POLICY_SET" -eq 1 ]; then
        echo "--final-outcome-policy requires one value and may only be provided once" >&2
        exit 2
      fi
      case "$2" in
        qualified_only|allow_not_qualified) FINAL_OUTCOME_POLICY="$2" ;;
        *) echo "Invalid --final-outcome-policy: $2" >&2; exit 2 ;;
      esac
      FINAL_OUTCOME_POLICY_SET=1
      shift 2
      ;;
    --no-final-evaluation)
      FINAL_EVALUATION_MODE="omit"
      shift
      ;;
    --contract-task-dir)
      CONTRACT_TASK_DIR="$2"
      shift 2
      ;;
    --investigation-results-dir)
      INVESTIGATION_RESULTS_DIR="$2"
      shift 2
      ;;
    --global-memory-dir)
      GLOBAL_MEMORY_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --manifest-dir)
      MANIFEST_DIR="$2"
      shift 2
      ;;
    --resume-existing)
      RESUME_EXISTING=1
      shift
      ;;
    --instruct|-i)
      INSTRUCT_PATH="$2"
      shift 2
      ;;
    --venue-file)
      VENUE_SOURCE_PATH="$2"
      VENUE_SOURCE_EXPLICIT=1
      shift 2
      ;;
    --contributions-file)
      CONTRIBUTIONS_SOURCE_PATH="$2"
      shift 2
      ;;
    --skills)
      PAPER_SKILLS_CSV="$2"
      shift 2
      ;;
    --paper-min-refs)
      PAPER_MIN_REFS="$2"
      shift 2
      ;;
    --paper-max-refs)
      PAPER_MAX_REFS="$2"
      shift 2
      ;;
    --paper-profile)
      PAPER_PROFILE="$2"
      shift 2
      ;;
    --standalone-paper)
      PAPER_PROFILE="standalone"
      TWO_STAGE=1
      shift
      ;;
    --paper-format)
      PAPER_FORMAT="$2"
      shift 2
      ;;
    --paper-max-repair-rounds)
      PAPER_MAX_REPAIR_ROUNDS="$2"
      shift 2
      ;;
    --paper-review-rounds)
      PAPER_REVIEW_ROUNDS="$2"
      shift 2
      ;;
    --two-stage)
      TWO_STAGE=1
      shift
      ;;
    --log-format)
      PAPER_LOG_FORMAT="$2"
      shift 2
      ;;
    --verbose)
      PAPER_LOG_VERBOSE=1
      shift
      ;;
    --heartbeat-seconds)
      PAPER_HEARTBEAT_SECONDS="$2"
      shift 2
      ;;
    --event-transport)
      PAPER_EVENT_TRANSPORT="$2"
      shift 2
      ;;
    --containerize)
      CONTAINERIZE=1
      shift
      ;;
    --container-image)
      CONTAINER_IMAGE="$2"
      shift 2
      ;;
    --container-network)
      CONTAINER_NETWORK="$2"
      shift 2
      ;;
    --container-auth-root)
      CONTAINER_AUTH_ROOT="$2"
      shift 2
      ;;
    --container-env-file)
      CONTAINER_ENV_FILE="$2"
      CONTAINER_ENV_FILE_SET=x
      shift 2
      ;;
    claude|codex|cursor|opencode|gemini|pi)
      AGENT="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ -n "$AGENT_MODEL" ]; then
  AGENT_MODEL_NORMALIZED="$(printf '%s' "$AGENT_MODEL" | tr '[:upper:]' '[:lower:]')"
  case "$AGENT_MODEL_NORMALIZED" in
    none|null)
      AGENT_MODEL=""
      ;;
  esac
fi

if [ -z "$RESULTS_DIR" ]; then
  TASK_NAME="$(basename "${TASK_DIR%/}")"
  if [ -z "$TASK_NAME" ] || [ "$TASK_NAME" = "." ] || [ "$TASK_NAME" = "/" ]; then
    TASK_NAME="task"
  fi
  RESULTS_DIR="results/$TASK_NAME/$AGENT"
fi
if [ -z "$GLOBAL_MEMORY_DIR" ]; then
  GLOBAL_MEMORY_DIR="$RESULTS_DIR"
fi

if [[ "$TASK_DIR" != /* ]]; then
  TASK_DIR="$REPO_ROOT/$TASK_DIR"
fi
if [[ "$RESULTS_DIR" != /* ]]; then
  RESULTS_DIR="$REPO_ROOT/$RESULTS_DIR"
fi
if [ -n "$ABLATIONS_DIR" ] && [[ "$ABLATIONS_DIR" != /* ]]; then
  ABLATIONS_DIR="$REPO_ROOT/$ABLATIONS_DIR"
fi
if [ -n "$MANIFEST_DIR" ] && [[ "$MANIFEST_DIR" != /* ]]; then
  MANIFEST_DIR="$REPO_ROOT/$MANIFEST_DIR"
fi
if [ -n "$POSTHOC_DIR" ] && [[ "$POSTHOC_DIR" != /* ]]; then
  POSTHOC_DIR="$REPO_ROOT/$POSTHOC_DIR"
fi
if [ -n "$FINAL_EVALUATION_DIR" ] && [[ "$FINAL_EVALUATION_DIR" != /* ]]; then
  FINAL_EVALUATION_DIR="$REPO_ROOT/$FINAL_EVALUATION_DIR"
fi
if [ -n "$CONTRACT_TASK_DIR" ] && [[ "$CONTRACT_TASK_DIR" != /* ]]; then
  CONTRACT_TASK_DIR="$REPO_ROOT/$CONTRACT_TASK_DIR"
fi
if [ -n "$INVESTIGATION_RESULTS_DIR" ] && [[ "$INVESTIGATION_RESULTS_DIR" != /* ]]; then
  INVESTIGATION_RESULTS_DIR="$REPO_ROOT/$INVESTIGATION_RESULTS_DIR"
fi
if [[ "$GLOBAL_MEMORY_DIR" != /* ]]; then
  GLOBAL_MEMORY_DIR="$REPO_ROOT/$GLOBAL_MEMORY_DIR"
fi
if [ -n "$INSTRUCT_PATH" ] && [[ "$INSTRUCT_PATH" != /* ]]; then
  INSTRUCT_PATH="$REPO_ROOT/$INSTRUCT_PATH"
fi
if [ -n "$OUTPUT_DIR" ] && [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
fi
if [ -n "$VENUE_SOURCE_PATH" ] && [[ "$VENUE_SOURCE_PATH" != /* ]]; then
  VENUE_SOURCE_PATH="$REPO_ROOT/$VENUE_SOURCE_PATH"
fi
if [ -n "$CONTRIBUTIONS_SOURCE_PATH" ] && [[ "$CONTRIBUTIONS_SOURCE_PATH" != /* ]]; then
  CONTRIBUTIONS_SOURCE_PATH="$REPO_ROOT/$CONTRIBUTIONS_SOURCE_PATH"
fi
if [[ "$CONTAINER_AUTH_ROOT" == "~"* ]]; then
  CONTAINER_AUTH_ROOT="${CONTAINER_AUTH_ROOT/#\~/$HOME}"
fi
if [ -n "$CONTAINER_AUTH_ROOT" ] && [[ "$CONTAINER_AUTH_ROOT" != /* ]]; then
  CONTAINER_AUTH_ROOT="$REPO_ROOT/$CONTAINER_AUTH_ROOT"
fi
if [[ "$CONTAINER_ENV_FILE" == "~"* ]]; then
  CONTAINER_ENV_FILE="${CONTAINER_ENV_FILE/#\~/$HOME}"
fi
if [ -n "$CONTAINER_ENV_FILE" ] && [[ "$CONTAINER_ENV_FILE" != /* ]]; then
  CONTAINER_ENV_FILE="$REPO_ROOT/$CONTAINER_ENV_FILE"
fi
if [ "$CONTAINER_ENV_FILE_SET" != x ] && [ -f "$REPO_ROOT/.env" ]; then
  CONTAINER_ENV_FILE="$REPO_ROOT/.env"
fi
# Reference retrieval runs on the host, not inside the agent container, so the
# container env-file never reaches it. Export just the retrieval credentials from
# that file when the caller has not already set them; every other secret in the
# env-file deliberately stays out of the host environment.
load_host_reference_credentials() {
  local env_file="$1"
  local name value
  if [ -z "$env_file" ] || [ ! -f "$env_file" ]; then
    return 0
  fi
  for name in S2_API_KEY OPENALEX_EMAIL; do
    if [ -n "${!name:-}" ]; then
      continue
    fi
    value="$(sed -n "s/^[[:space:]]*\(export[[:space:]]\+\)\?${name}=//p" "$env_file" | tail -n 1)"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    if [ -n "$value" ]; then
      export "$name=$value"
    fi
  done
}
load_host_reference_credentials "$CONTAINER_ENV_FILE"

if [ -z "$CONTRIBUTIONS_SOURCE_PATH" ] && [ -f "$REPO_ROOT/contributions.md" ]; then
  CONTRIBUTIONS_SOURCE_PATH="$REPO_ROOT/contributions.md"
fi

if [ -z "$INSTRUCT_PATH" ]; then
  if [ "$PAPER_PROFILE" = "standalone" ]; then
    INSTRUCT_PATH="$REPO_ROOT/src/prompts/paper_standalone.md"
  else
    INSTRUCT_PATH="$REPO_ROOT/src/prompts/paper.md"
  fi
fi
REFERENCE_QUERY_PLAN_INSTRUCT_PATH="$REPO_ROOT/src/prompts/reference_query_plan.md"
PAPER_REFERENCE_GAPS_INSTRUCT_PATH="$REPO_ROOT/src/prompts/paper_reference_gaps.md"
POSTDRAFT_REFERENCE_QUERY_PLAN_INSTRUCT_PATH="$REPO_ROOT/src/prompts/reference_query_plan_postdraft.md"
PAPER_STRATEGY_INSTRUCT_PATH="$REPO_ROOT/src/prompts/paper_strategy.md"
PAPER_REVIEW_INSTRUCT_PATH="$REPO_ROOT/src/prompts/paper_review/review_json.md"
PAPER_EVIDENCE_AUDIT_INSTRUCT_PATH="$REPO_ROOT/src/prompts/paper_review/evidence_audit.md"
PAPER_FIGURE_QUALITY_REPAIR_INSTRUCT_PATH="$REPO_ROOT/src/prompts/paper_figure_quality_repair.md"
if [ -n "$AGENT_MODEL" ] && [ -z "${AGENT_MODEL//[[:space:]]/}" ]; then
  echo "--agent-model must be non-empty when provided" >&2
  exit 1
fi
if [ -n "$AGENT_PROFILE" ] && [ -z "${AGENT_PROFILE//[[:space:]]/}" ]; then
  echo "--agent-profile must be non-empty when provided" >&2
  exit 1
fi
if [ -n "$AGENT_PROFILE" ] && [ "$AGENT" != "codex" ]; then
  echo "--agent-profile is only supported with --agent codex" >&2
  exit 1
fi
validate_agent_reasoning_effort "$AGENT" "$AGENT_REASONING_EFFORT" || exit 1
export AGENT_REASONING_EFFORT

if [ -n "$FINAL_EVALUATION_DIR" ]; then
  if [ "$CONTAINERIZE" -ne 1 ]; then
    echo "Sealed final grounding requires containerized paper generation" >&2
    exit 1
  fi
  export AGENT_CONTAINER_MOUNT_DOCKER_SOCK=0
fi

if [ ! -d "$TASK_DIR" ]; then
  echo "Missing task directory: $TASK_DIR" >&2
  exit 1
fi
if [ -n "$POSTHOC_DIR" ] && { [ ! -f "$POSTHOC_DIR/ranking.json" ] || [ ! -d "$POSTHOC_DIR/analyses" ]; }; then
  echo "Invalid posthoc directory: $POSTHOC_DIR" >&2
  exit 1
fi
if [ ! -d "$RESULTS_DIR" ]; then
  echo "Missing results directory: $RESULTS_DIR" >&2
  exit 1
fi
if [ ! -f "$INSTRUCT_PATH" ]; then
  echo "Missing paper instruction template: $INSTRUCT_PATH" >&2
  exit 1
fi
# Resolve --paper-format against the venue catalog (configs/venue/<name>.yaml);
# its brief is the default venue file. An explicit --venue-file wins and
# --venue-file "" opts out entirely.
if ! VENUE_CATALOG_BRIEF="$("$PYTHON_BIN" - "$REPO_ROOT" "$PAPER_FORMAT" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from src.orchestrator.paper_venues import VenueSpecError, load_venue, venue_names  # noqa: E402

try:
    print(load_venue(sys.argv[2]).brief)
except VenueSpecError as exc:
    print(f"{exc}; declared venues: {', '.join(venue_names())}", file=sys.stderr)
    sys.exit(1)
PY
)"; then
  echo "--paper-format must name a venue under configs/venue (got: $PAPER_FORMAT)" >&2
  exit 1
fi
if [ "$VENUE_SOURCE_EXPLICIT" != "1" ]; then
  VENUE_SOURCE_PATH="$REPO_ROOT/$VENUE_CATALOG_BRIEF"
fi

if [ -n "$VENUE_SOURCE_PATH" ] && [ ! -f "$VENUE_SOURCE_PATH" ]; then
  echo "Missing venue file: $VENUE_SOURCE_PATH" >&2
  exit 1
fi
if [ -n "$CONTRIBUTIONS_SOURCE_PATH" ] && [ ! -f "$CONTRIBUTIONS_SOURCE_PATH" ]; then
  echo "Missing contributions file: $CONTRIBUTIONS_SOURCE_PATH" >&2
  exit 1
fi
if [ ! -f "$REFERENCE_QUERY_PLAN_INSTRUCT_PATH" ]; then
  echo "Missing reference query plan instruction template: $REFERENCE_QUERY_PLAN_INSTRUCT_PATH" >&2
  exit 1
fi
if [ ! -f "$PAPER_REFERENCE_GAPS_INSTRUCT_PATH" ]; then
  echo "Missing paper reference gaps instruction template: $PAPER_REFERENCE_GAPS_INSTRUCT_PATH" >&2
  exit 1
fi
if [ ! -f "$POSTDRAFT_REFERENCE_QUERY_PLAN_INSTRUCT_PATH" ]; then
  echo "Missing postdraft reference query plan instruction template: $POSTDRAFT_REFERENCE_QUERY_PLAN_INSTRUCT_PATH" >&2
  exit 1
fi
if [ "$TWO_STAGE" -eq 1 ] && [ ! -f "$PAPER_STRATEGY_INSTRUCT_PATH" ]; then
  echo "Missing paper strategy instruction template: $PAPER_STRATEGY_INSTRUCT_PATH" >&2
  exit 1
fi
if [[ "$PAPER_REVIEW_ROUNDS" =~ ^[0-9]+$ ]] && [ "$PAPER_REVIEW_ROUNDS" -gt 0 ] && [ ! -f "$PAPER_REVIEW_INSTRUCT_PATH" ]; then
  echo "Missing paper review instruction template: $PAPER_REVIEW_INSTRUCT_PATH" >&2
  exit 1
fi
if [ ! -f "$PAPER_EVIDENCE_AUDIT_INSTRUCT_PATH" ]; then
  echo "Missing paper evidence audit instruction template: $PAPER_EVIDENCE_AUDIT_INSTRUCT_PATH" >&2
  exit 1
fi
if [ ! -f "$PAPER_FIGURE_QUALITY_REPAIR_INSTRUCT_PATH" ]; then
  echo "Missing figure-quality repair instruction template: $PAPER_FIGURE_QUALITY_REPAIR_INSTRUCT_PATH" >&2
  exit 1
fi
if ! [[ "$PAPER_MIN_REFS" =~ ^[0-9]+$ ]] || [ "$PAPER_MIN_REFS" -le 0 ]; then
  echo "--paper-min-refs must be a positive integer (got: $PAPER_MIN_REFS)" >&2
  exit 1
fi
if ! [[ "$PAPER_MAX_REFS" =~ ^[0-9]+$ ]] || [ "$PAPER_MAX_REFS" -lt "$PAPER_MIN_REFS" ]; then
  echo "--paper-max-refs must be an integer >= --paper-min-refs (got: $PAPER_MAX_REFS)" >&2
  exit 1
fi
if ! [[ "$PAPER_MAX_REPAIR_ROUNDS" =~ ^[0-9]+$ ]]; then
  echo "--paper-max-repair-rounds must be a non-negative integer (got: $PAPER_MAX_REPAIR_ROUNDS)" >&2
  exit 1
fi
if ! [[ "$PAPER_REVIEW_ROUNDS" =~ ^[0-9]+$ ]]; then
  echo "--paper-review-rounds must be a non-negative integer (got: $PAPER_REVIEW_ROUNDS)" >&2
  exit 1
fi
if [[ "$PAPER_PROFILE" != "default" && "$PAPER_PROFILE" != "standalone" ]]; then
  echo "--paper-profile must be 'default' or 'standalone' (got: $PAPER_PROFILE)" >&2
  exit 1
fi
validate_paper_progress_settings
if [ "$CONTAINERIZE" -eq 1 ]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required when --containerize is enabled" >&2
    exit 1
  fi
  if [ ! -d "$CONTAINER_AUTH_ROOT" ]; then
    echo "Missing container auth root directory: $CONTAINER_AUTH_ROOT" >&2
    exit 1
  fi
  if [ -n "$CONTAINER_ENV_FILE" ] && [ ! -f "$CONTAINER_ENV_FILE" ]; then
    echo "Missing container env file: $CONTAINER_ENV_FILE" >&2
    exit 1
  fi
fi

TASK_SOURCE_DIR="$TASK_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$RESULTS_DIR/paper/final_${TIMESTAMP}"
fi
if [ -e "$OUTPUT_DIR" ]; then
  PAPER_OUTPUT_PREEXISTED=1
else
  PAPER_OUTPUT_PREEXISTED=0
  mkdir -p "$OUTPUT_DIR"
fi
if [ "$PAPER_OUTPUT_PREEXISTED" -eq 1 ] && [ "$RESUME_EXISTING" -ne 1 ]; then
  echo "Paper output directory already exists: $OUTPUT_DIR" >&2
  exit 1
fi
if [ "$PAPER_OUTPUT_PREEXISTED" -eq 1 ]; then
  paper_workspace_validation_args=(
    validate
    --output-dir "$OUTPUT_DIR"
    --task-dir "$TASK_SOURCE_DIR"
    --results-dir "$RESULTS_DIR"
    --global-memory-dir "$GLOBAL_MEMORY_DIR"
  )
  if [ -n "$ABLATIONS_DIR" ]; then
    paper_workspace_validation_args+=(--ablations-dir "$ABLATIONS_DIR")
  fi
  if ! "$PYTHON_BIN" -m src.pipeline.paper_workspace \
    "${paper_workspace_validation_args[@]}"; then
    echo "Refusing to resume unowned or unsafe paper output: $OUTPUT_DIR" >&2
    exit 1
  fi
else
  "$PYTHON_BIN" -m src.pipeline.paper_workspace init \
    --output-dir "$OUTPUT_DIR" \
    --task-dir "$TASK_SOURCE_DIR" \
    --results-dir "$RESULTS_DIR" \
    --global-memory-dir "$GLOBAL_MEMORY_DIR"
fi
paper_progress_initialize

trap paper_event_early_exit EXIT

# OpenAI is the default concept-figure backend. Gemini is selected automatically
# when only GEMINI_API_KEY is available; without either key, enhancement is skipped.
if [ -n "${OPENAI_API_KEY:-}" ] || { [ "$CONTAINERIZE" -eq 1 ] && [ -n "$CONTAINER_ENV_FILE" ] && env_file_has_nonempty_key "$CONTAINER_ENV_FILE" "OPENAI_API_KEY"; }; then
  PAPER_OPENAI_API_KEY_AVAILABLE="true"
  PAPER_OPENAI_API_KEY_CONTEXT="OPENAI_API_KEY is expected to be available in this run environment. Do not claim it is unavailable unless you re-check inside the active workspace or process environment."
else
  PAPER_OPENAI_API_KEY_AVAILABLE="false"
  PAPER_OPENAI_API_KEY_CONTEXT="OPENAI_API_KEY was not detected from the current shell or configured container env file before launch. Re-check inside the active workspace or process environment before deciding it is unavailable; the orchestrator will use Gemini when available or otherwise ship the deterministic export with --skip-enhance."
fi
export PAPER_OPENAI_API_KEY_AVAILABLE
export PAPER_OPENAI_API_KEY_CONTEXT

# GEMINI_API_KEY provides the automatic enhancement and audit fallback when
# OPENAI_API_KEY is unavailable; a missing key never blocks deterministic export.
if [ -n "${GEMINI_API_KEY:-}" ] || { [ "$CONTAINERIZE" -eq 1 ] && [ -n "$CONTAINER_ENV_FILE" ] && env_file_has_nonempty_key "$CONTAINER_ENV_FILE" "GEMINI_API_KEY"; }; then
  PAPER_GEMINI_API_KEY_AVAILABLE="true"
  PAPER_GEMINI_API_KEY_CONTEXT="GEMINI_API_KEY is available as the automatic concept-figure fallback when OPENAI_API_KEY is unavailable."
else
  PAPER_GEMINI_API_KEY_AVAILABLE="false"
  PAPER_GEMINI_API_KEY_CONTEXT="GEMINI_API_KEY was not detected before launch. This is not a blocker: OpenAI remains the default, and a no-key run can still ship the deterministic export."
fi
export PAPER_GEMINI_API_KEY_AVAILABLE
export PAPER_GEMINI_API_KEY_CONTEXT

SELECTED_PAPER_SKILLS="$("$PYTHON_BIN" - "$REPO_ROOT" "$PAPER_SKILLS_CSV" "$PAPER_PROFILE" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
override_csv = sys.argv[2]
paper_profile = sys.argv[3]

sys.path.insert(0, str(script_dir))
from src.investigation.skill_scope import (  # noqa: E402
    paper_writer_skill,
    resolve_skills_for_step,
    validate_skill_set,
)

skills_src = script_dir / "src" / "agents" / "skills"
selected_skills = resolve_skills_for_step("paper", override_csv, paper_profile=paper_profile)
validate_skill_set(skills_src, selected_skills)
paper_writer_skill(selected_skills)
print(",".join(selected_skills))
PY
)"
echo "[paper.sh] selected paper skills: $SELECTED_PAPER_SKILLS"
PAPER_WRITER_SKILL="$("$PYTHON_BIN" - "$REPO_ROOT" "$SELECTED_PAPER_SKILLS" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
selected_csv = sys.argv[2]

sys.path.insert(0, str(script_dir))
from src.investigation.skill_scope import paper_writer_skill, parse_skill_csv  # noqa: E402

print(paper_writer_skill(parse_skill_csv(selected_csv)))
PY
)"
echo "[paper.sh] paper writer skill: $PAPER_WRITER_SKILL"
CURRENT_PAPER_INSTRUCT_PATH="$INSTRUCT_PATH"
CURRENT_SELECTED_PAPER_SKILLS="$SELECTED_PAPER_SKILLS"

mkdir -p "$GLOBAL_MEMORY_DIR"
if [ -n "$CONTRACT_TASK_DIR" ] && [ "$CONTRACT_TASK_DIR" = "$TASK_SOURCE_DIR" ]; then
  TASK_DIR="$TASK_SOURCE_DIR"
else
  TASK_DIR="$(ensure_staged_task_copy "$REPO_ROOT" "$TASK_SOURCE_DIR" "$RESULTS_DIR")"
fi
PAPER_PREP_METADATA_PATH="$OUTPUT_DIR/paper_prep_${TIMESTAMP}.json"
paper_gate_start "preparation" "phase=workspace"
SOURCE_RESULTS_DIR="$(
  "$PYTHON_BIN" - "$REPO_ROOT" "$RESULTS_DIR" "$TIMESTAMP" "$PAPER_PREP_METADATA_PATH" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
timestamp = sys.argv[3]
metadata_path = Path(sys.argv[4])
merge_root = metadata_path.parent / f"_merged_parallel_{timestamp}"

sys.path.insert(0, str(script_dir))
from src.orchestrator.paper_data import prepare_results_dir_for_paper  # noqa: E402

paper_results_dir, prep = prepare_results_dir_for_paper(results_dir, merge_root=merge_root)
prep["requested_results_dir"] = str(results_dir.expanduser().resolve())
prep["effective_results_dir"] = str(paper_results_dir.expanduser().resolve())
metadata_path.write_text(json.dumps(prep, indent=2, sort_keys=True), encoding="utf-8")
print(str(paper_results_dir))
PY
)"

if [ "$SOURCE_RESULTS_DIR" != "$RESULTS_DIR" ]; then
  echo "[paper.sh] detected parallel worker layout; merged paper source at: $SOURCE_RESULTS_DIR"
fi
echo "[paper.sh] preparation metadata: $PAPER_PREP_METADATA_PATH"
if [ "$PAPER_OUTPUT_PREEXISTED" -eq 1 ]; then
  echo "[paper.sh] resuming existing paper output: $OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR/figures"
configure_cost_tracking "$OUTPUT_DIR" paper "$PAPER_LOG_FORMAT"
if [ "$PAPER_EVENT_TRANSPORT" = "wrapper" ]; then
  echo "Paper directory: $OUTPUT_DIR" >&3
fi
paper_gate_complete "preparation" "phase=workspace"
if [ -z "$VENUE_SOURCE_PATH" ]; then
  rm -f -- "$OUTPUT_DIR/VENUE.md"
fi
if [ "$RESUME_EXISTING" -eq 1 ] && [ -z "$POSTHOC_DIR" ] && [ "$FINAL_EVALUATION_MODE" != "omit" ]; then
  paper_gate_start "finalization" "phase=resume"
  set +e
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$RESULTS_DIR" "$GLOBAL_MEMORY_DIR" "$CONTAINERIZE" "$CONTAINER_IMAGE" "$CONTAINER_NETWORK" "$PAPER_REVIEW_ROUNDS" "$MANIFEST_DIR" "$FINAL_EVALUATION_DIR" "$SOURCE_RESULTS_DIR" "${INVESTIGATION_RESULTS_DIR:-$SOURCE_RESULTS_DIR}" "$CONTRACT_TASK_DIR" "$FINAL_OUTCOME_POLICY" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
results_dir = Path(sys.argv[3])
global_memory_dir = Path(sys.argv[4])
containerize = sys.argv[5] == "1"
container_image = sys.argv[6] if containerize else None
container_network = sys.argv[7]
paper_review_rounds = int(sys.argv[8])
manifest_dir = Path(sys.argv[9]) if sys.argv[9] else None
sys.path.insert(0, str(script_dir))

from src.orchestrator.paper_finalize import finalize_resumed_paper_if_valid  # noqa: E402
from src.orchestrator.paper_final_outcome import verify_resumed_final_outcome  # noqa: E402

try:
    verify_resumed_final_outcome(
        output_dir, final_evaluation_dir=Path(sys.argv[10]) if sys.argv[10] else None,
        source_results_dir=Path(sys.argv[11]), investigation_results_dir=Path(sys.argv[12]),
        task_dir=Path(sys.argv[13]) if sys.argv[13] else None,
        final_outcome_policy=sys.argv[14],
    )
    completed = finalize_resumed_paper_if_valid(
        output_dir=output_dir,
        results_dir=results_dir,
        global_memory_dir=global_memory_dir,
        container_image=container_image,
        container_network=container_network,
        paper_review_rounds=paper_review_rounds,
        manifest_dir=manifest_dir,
    )
except (Exception, SystemExit) as exc:
    print(f"[paper.sh] resumed-paper finalization failed: {exc}", file=sys.stderr)
    raise SystemExit(2)
raise SystemExit(0 if completed else 1)
PY
  resume_rc=$?
  set -e
  if [ "$resume_rc" -eq 0 ]; then
    paper_gate_complete "finalization" "phase=resume" "exit_code=0"
    echo "[paper.sh] resumed paper already validates; finalized without rerunning agent stages"
    exit 0
  fi
  if [ "$resume_rc" -gt 1 ]; then
    paper_gate_failed "finalization" "phase=resume" "exit_code=$resume_rc"
    exit "$resume_rc"
  fi
  paper_gate_finish "finalization" "skipped" "phase=resume" "reason=incomplete"
  echo "[paper.sh] resumed paper is incomplete or invalid; continuing agent stages"
elif [ "$RESUME_EXISTING" -eq 1 ]; then
  echo "[paper.sh] fresh posthoc input supplied; rerunning paper stages"
fi
if [ "$TWO_STAGE" -eq 1 ]; then
  : > "$OUTPUT_DIR/.paper_two_stage_enabled"
else
  rm -f "$OUTPUT_DIR/.paper_two_stage_enabled"
fi
ensure_workspace_dir_alias "$REPO_ROOT" "$OUTPUT_DIR" "task" "$TASK_DIR" >/dev/null
ensure_workspace_dir_alias "$REPO_ROOT" "$OUTPUT_DIR" "results" "$SOURCE_RESULTS_DIR" >/dev/null
if [ "$RESUME_EXISTING" -eq 1 ]; then
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR/global_memory" "$GLOBAL_MEMORY_DIR" <<'PY' >/dev/null
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
alias_path = Path(sys.argv[2])
target_dir = Path(sys.argv[3])
sys.path.insert(0, str(repo_root))

from src.pipeline.paper_memory import restore_materialized_paper_memory  # noqa: E402

print(restore_materialized_paper_memory(alias_path, target_dir))
PY
else
  ensure_workspace_dir_alias "$REPO_ROOT" "$OUTPUT_DIR" "global_memory" "$GLOBAL_MEMORY_DIR" >/dev/null
fi
# Campaign-level ablation assets remain optional paper enrichments.
if [ -n "$ABLATIONS_DIR" ] && [ -f "$ABLATIONS_DIR/README.md" ]; then
  cp "$ABLATIONS_DIR/README.md" "$OUTPUT_DIR/results_readme.md"
elif [ -f "$RESULTS_DIR/README.md" ]; then
  cp "$RESULTS_DIR/README.md" "$OUTPUT_DIR/results_readme.md"
elif [ -f "$SOURCE_RESULTS_DIR/README.md" ]; then
  cp "$SOURCE_RESULTS_DIR/README.md" "$OUTPUT_DIR/results_readme.md"
fi
PAPER_AGGREGATED_DIR=""
if [ -n "$ABLATIONS_DIR" ] && [ -d "$ABLATIONS_DIR/aggregated" ]; then
  PAPER_AGGREGATED_DIR="$ABLATIONS_DIR/aggregated"
elif [ -d "$RESULTS_DIR/aggregated" ]; then
  PAPER_AGGREGATED_DIR="$RESULTS_DIR/aggregated"
elif [ -d "$SOURCE_RESULTS_DIR/aggregated" ]; then
  PAPER_AGGREGATED_DIR="$SOURCE_RESULTS_DIR/aggregated"
fi
if [ -n "$PAPER_AGGREGATED_DIR" ]; then
  ensure_workspace_dir_alias "$REPO_ROOT" "$OUTPUT_DIR" "aggregated" "$PAPER_AGGREGATED_DIR" >/dev/null
fi
PAPER_EXTRA_TASKS_DIR=""
if [ -n "$ABLATIONS_DIR" ] && [ -d "$ABLATIONS_DIR/extra_tasks" ]; then
  PAPER_EXTRA_TASKS_DIR="$ABLATIONS_DIR/extra_tasks"
elif [ -d "$RESULTS_DIR/extra_tasks" ]; then
  PAPER_EXTRA_TASKS_DIR="$RESULTS_DIR/extra_tasks"
fi
if [ -n "$PAPER_EXTRA_TASKS_DIR" ]; then
  ensure_workspace_dir_alias "$REPO_ROOT" "$OUTPUT_DIR" "extra_tasks" "$PAPER_EXTRA_TASKS_DIR" >/dev/null
fi

PAPER_CONTAINER_MOUNT_ROOT="$OUTPUT_DIR"
PAPER_AGENT_CONTAINER_EXTRA_MOUNTS="${AGENT_CONTAINER_EXTRA_MOUNTS:-}"
PAPER_AGENT_WORKDIR_READONLY=0
if [ "$CONTAINERIZE" -eq 1 ]; then
  PAPER_CONTAINER_PLAN="$(
    "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$TASK_DIR" "$RESULTS_DIR" "$GLOBAL_MEMORY_DIR" "$PAPER_AGGREGATED_DIR" "$PAPER_EXTRA_TASKS_DIR" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
task_dir = Path(sys.argv[3])
results_dir = Path(sys.argv[4])
global_memory_dir = Path(sys.argv[5])
aggregated_dir = Path(sys.argv[6]) if sys.argv[6] else None
extra_tasks_dir = Path(sys.argv[7]) if sys.argv[7] else None

sys.path.insert(0, str(repo_root))
from src.orchestrator import state as state_store  # noqa: E402
from src.pipeline.paper_workspace import (  # noqa: E402
    paper_container_mount_plan,
    validate_existing_global_memory_files,
)

validate_existing_global_memory_files(global_memory_dir)
state_store.ensure_global_memory_files(global_memory_dir)
conclusions_history = state_store.conclusions_history_path(global_memory_dir)
if not conclusions_history.exists():
    state_store.write_private_text(conclusions_history, "")
plan = paper_container_mount_plan(
    output_dir=output_dir,
    task_dir=task_dir,
    results_dir=results_dir,
    global_memory_dir=global_memory_dir,
    aggregated_dir=aggregated_dir,
    extra_tasks_dir=extra_tasks_dir,
)
print(plan.mount_root)
print("\n".join(plan.readonly_mounts))
print("\n".join(plan.writable_mounts))
PY
  )"
  PAPER_CONTAINER_MOUNT_ROOT="${PAPER_CONTAINER_PLAN%%$'\n'*}"
  PAPER_CONTAINER_WRITABLE_MOUNTS="${PAPER_CONTAINER_PLAN#*$'\n'}"
  if [ -n "${AGENT_CONTAINER_EXTRA_MOUNTS:-}" ]; then
    echo "[paper.sh] ignoring ambient AGENT_CONTAINER_EXTRA_MOUNTS; paper mounts are policy-owned" >&2
  fi
  PAPER_AGENT_CONTAINER_EXTRA_MOUNTS="$PAPER_CONTAINER_WRITABLE_MOUNTS"
  PAPER_AGENT_WORKDIR_READONLY=1
fi

paper_gate_start "preparation" "phase=grounding"
"$PYTHON_BIN" - "$REPO_ROOT" "$TASK_DIR" "$RESULTS_DIR" "$GLOBAL_MEMORY_DIR" "$SOURCE_RESULTS_DIR" "$OUTPUT_DIR" "$PAPER_MIN_REFS" "$PAPER_MAX_REFS" "$PAPER_PROFILE" "$FINAL_EVALUATION_DIR" "$FINAL_EVALUATION_MODE" "$CONTRACT_TASK_DIR" "$INVESTIGATION_RESULTS_DIR" "$ABLATIONS_DIR" "$FINAL_OUTCOME_POLICY" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
task_dir = Path(sys.argv[2])
campaign_results_dir = Path(sys.argv[3])
global_memory_dir = Path(sys.argv[4])
source_results_dir = Path(sys.argv[5])
output_dir = Path(sys.argv[6])
paper_min_refs = int(sys.argv[7])
paper_max_refs = int(sys.argv[8])
paper_profile = sys.argv[9]
final_evaluation_dir = Path(sys.argv[10]) if sys.argv[10] else None
final_evaluation_mode = sys.argv[11]
contract_task_dir = Path(sys.argv[12]) if sys.argv[12] else None
investigation_results_dir = Path(sys.argv[13]) if sys.argv[13] else None
ablations_dir = Path(sys.argv[14]) if sys.argv[14] else None
final_outcome_policy = sys.argv[15]

sys.path.insert(0, str(script_dir))

from src.orchestrator.paper_data import (  # noqa: E402
    FinalEvaluationMode,
    build_paper_data,
    write_paper_data,
)
from src.orchestrator import state as state_store  # noqa: E402

paper_data = build_paper_data(
    source_results_dir,
    paper_profile=paper_profile,
    ablations_dir=ablations_dir,
    final_evaluation_dir=final_evaluation_dir,
    task_dir=contract_task_dir,
    investigation_results_dir=investigation_results_dir,
    final_evaluation_mode=FinalEvaluationMode(final_evaluation_mode),
    final_outcome_policy=final_outcome_policy,
)
if int(paper_data["totals"]["evaluated_programs"]) <= 0 and not paper_data.get("downstream_study"):
    raise SystemExit("No evaluated programs found in results directory; cannot write paper.")

state_store.ensure_histories(campaign_results_dir)
state_store.ensure_global_memory_files(global_memory_dir)
global_memory_jsonl_path = state_store.global_memory_jsonl_path(global_memory_dir)
global_memory_recent = state_store.tail_jsonl(global_memory_jsonl_path, max_lines=80)

paper_data_path = output_dir / "paper_data.json"
from src.orchestrator.paper_study_sources import stage_study_sources
stage_study_sources(paper_data, output_dir, investigation_results_dir or source_results_dir)
write_paper_data(paper_data, paper_data_path)

summary_lines = [
    "# Global Memory Summary",
    "",
    "Use as qualitative context only; do not use as quantitative grounding source.",
    "",
]
if not global_memory_recent:
    summary_lines.append("- No global memory entries found yet.")
else:
    summary_lines.append("## Recent Entries")
    summary_lines.append("")
    for entry in global_memory_recent[-20:]:
        if not isinstance(entry, dict):
            continue
        timestamp = str(entry.get("timestamp", "")).strip() or "unknown_ts"
        step = str(entry.get("step", "")).strip() or "unknown_step"
        scope = str(entry.get("scope", "")).strip() or "unknown_scope"
        summary = str(entry.get("summary", "")).strip() or "(missing summary)"
        evidence = str(entry.get("evidence", "")).strip() or "(missing evidence)"
        summary_lines.append(f"- {timestamp} | step={step} | scope={scope}")
        summary_lines.append(f"  - summary: {summary}")
        summary_lines.append(f"  - evidence: {evidence}")

(output_dir / "global_memory_summary.md").write_text(
    "\n".join(summary_lines).rstrip() + "\n",
    encoding="utf-8",
)
(output_dir / "global_memory_recent.json").write_text(
    json.dumps(global_memory_recent, indent=2, sort_keys=True),
    encoding="utf-8",
)

print(f"[paper.sh] prepared grounded inputs in {output_dir}")
PY
paper_gate_complete "preparation" "phase=grounding"

stage_venue_file() {
  local venue_source_path="$1"
  local paper_format="$2"
  local paper_min_refs="$3"
  local paper_max_refs="$4"
  "$PYTHON_BIN" - "$REPO_ROOT" "$venue_source_path" "$OUTPUT_DIR" "$paper_format" "$paper_min_refs" "$paper_max_refs" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
venue_source_path = Path(sys.argv[2]) if sys.argv[2] else None
output_dir = Path(sys.argv[3])
paper_format = sys.argv[4]
paper_min_refs = sys.argv[5]
paper_max_refs = sys.argv[6]

sys.path.insert(0, str(script_dir))
from src.orchestrator.paper_venues import page_budget_sentence  # noqa: E402
from src.investigation.paper_prompt_context import stage_optional_rendered_doc  # noqa: E402

stage_optional_rendered_doc(
    source_path=venue_source_path,
    workspace_dir=output_dir,
    workspace_name="VENUE.md",
    task_dir=output_dir / "task",
    results_dir=output_dir / "results",
    extra_replacements={
        "PAPER_FORMAT": paper_format,
        "PAPER_PAGE_BUDGET": page_budget_sentence(paper_format),
        "PAPER_DIR": ".",
        "PAPER_DATA_PATH": "./paper_data.json",
        "PAPER_BIB_PATH": "./references.bib",
        "PAPER_REFERENCES_PROVENANCE_PATH": "./references_provenance.json",
        "PAPER_FIGURES_DIR": "./figures",
        "PAPER_FIGURE_PATH": "./figures/run_progression.png",
        "PAPER_AUTHOR_BLOCK_PATH": "./paper_authors.tex",
        "PAPER_TEX_PATH": "./paper.tex",
        "PAPER_CLAIMS_PATH": "./claims_grounding.json",
        "PAPER_MIN_REFS": paper_min_refs,
        "PAPER_MAX_REFS": paper_max_refs,
        "GLOBAL_MEMORY_SUMMARY_PATH": "./global_memory_summary.md",
    },
)
PY
}

stage_contributions_file() {
  local contributions_source_path="$1"
  local paper_format="$2"
  local paper_min_refs="$3"
  local paper_max_refs="$4"
  if [ -z "$contributions_source_path" ]; then
    return 0
  fi
  "$PYTHON_BIN" - "$REPO_ROOT" "$contributions_source_path" "$OUTPUT_DIR" "$paper_format" "$paper_min_refs" "$paper_max_refs" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
contributions_source_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
paper_format = sys.argv[4]
paper_min_refs = sys.argv[5]
paper_max_refs = sys.argv[6]

sys.path.insert(0, str(script_dir))
from src.investigation.paper_prompt_context import stage_optional_rendered_doc  # noqa: E402

stage_optional_rendered_doc(
    source_path=contributions_source_path,
    workspace_dir=output_dir,
    workspace_name="contributions.md",
    task_dir=output_dir / "task",
    results_dir=output_dir / "results",
    extra_replacements={
        "PAPER_FORMAT": paper_format,
        "PAPER_DIR": ".",
        "PAPER_DATA_PATH": "./paper_data.json",
        "PAPER_BIB_PATH": "./references.bib",
        "PAPER_REFERENCES_PROVENANCE_PATH": "./references_provenance.json",
        "PAPER_FIGURES_DIR": "./figures",
        "PAPER_FIGURE_PATH": "./figures/run_progression.png",
        "PAPER_AUTHOR_BLOCK_PATH": "./paper_authors.tex",
        "PAPER_TEX_PATH": "./paper.tex",
        "PAPER_CLAIMS_PATH": "./claims_grounding.json",
        "PAPER_MIN_REFS": paper_min_refs,
        "PAPER_MAX_REFS": paper_max_refs,
        "GLOBAL_MEMORY_SUMMARY_PATH": "./global_memory_summary.md",
    },
)
PY
}

stage_author_block_file() {
  local paper_format="$1"
  local author_block_source_path="$REPO_ROOT/src/templates/paper_authors/${paper_format}.tex"
  if [ ! -f "$author_block_source_path" ]; then
    return 0
  fi
  "$PYTHON_BIN" - "$REPO_ROOT" "$author_block_source_path" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
author_block_source_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

sys.path.insert(0, str(script_dir))
from src.investigation.paper_prompt_context import stage_optional_rendered_doc  # noqa: E402

stage_optional_rendered_doc(
    source_path=author_block_source_path,
    workspace_dir=output_dir,
    workspace_name="paper_authors.tex",
    task_dir=output_dir / "task",
    results_dir=output_dir / "results",
)
PY
}

case "$AGENT" in
  claude) AGENT_CONFIG_DIR=".claude" ;;
  codex) AGENT_CONFIG_DIR=".codex" ;;
  cursor) AGENT_CONFIG_DIR=".cursor" ;;
  opencode) AGENT_CONFIG_DIR=".opencode" ;;
  gemini) AGENT_CONFIG_DIR=".gemini" ;;
  pi) AGENT_CONFIG_DIR=".pi/agent" ;;
  *)
    echo "Unsupported agent: $AGENT" >&2
    exit 1
    ;;
esac

FORMATTED_SKILLS_REL_DIR="$AGENT_CONFIG_DIR/skills"
INCLUDE_REVIEW_SKILL=0
if [ "$PAPER_REVIEW_ROUNDS" -gt 0 ]; then
  INCLUDE_REVIEW_SKILL=1
fi

stage_paper_skills() {
  local selected_skills_csv="$1"
  local main_prompt_path="$2"
  local include_review_skill="$3"
  stage_venue_file "$VENUE_SOURCE_PATH" "$PAPER_FORMAT" "$PAPER_MIN_REFS" "$PAPER_MAX_REFS"
  stage_contributions_file "$CONTRIBUTIONS_SOURCE_PATH" "$PAPER_FORMAT" "$PAPER_MIN_REFS" "$PAPER_MAX_REFS"
  stage_author_block_file "$PAPER_FORMAT"
  (
    cd "$OUTPUT_DIR"
    "$PYTHON_BIN" - "$REPO_ROOT" "$FORMATTED_SKILLS_REL_DIR" "$RESULTS_DIR" "$GLOBAL_MEMORY_DIR" "$PAPER_MIN_REFS" "$PAPER_MAX_REFS" "$PAPER_FORMAT" "$selected_skills_csv" "$main_prompt_path" "$PAPER_REVIEW_ROUNDS" "$include_review_skill" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
formatted_skills_dir = Path(sys.argv[2])
campaign_results_dir = Path(sys.argv[3])
global_memory_dir = Path(sys.argv[4])
paper_min_refs = sys.argv[5]
paper_max_refs = sys.argv[6]
paper_format = sys.argv[7]
selected_skills_csv = sys.argv[8]
main_prompt_path = Path(sys.argv[9])
paper_review_rounds = int(sys.argv[10])
include_review_skill = sys.argv[11] == "1"

sys.path.insert(0, str(script_dir))
from src.orchestrator import state as state_store  # noqa: E402
from src.orchestrator.paper_watermark import watermark_staged_templates  # noqa: E402
from src.investigation.skill_scope import (  # noqa: E402
    paper_writer_skill,
    parse_skill_csv,
    paper_templates_dir,
    stage_paper_templates,
    stage_skill_tree,
    validate_skill_set,
)
from src.investigation.paper_prompt_context import (  # noqa: E402
    author_block_context_text,
    contributions_context_text,
    paper_workspace_context_replacements,
    venue_context_text,
)
from src.investigation.files import load_instructions  # noqa: E402

skills_src = script_dir / "src" / "agents" / "skills"
if not skills_src.is_dir():
    raise SystemExit(f"Missing skills directory: {skills_src}")
selected_skills = parse_skill_csv(selected_skills_csv)
validate_skill_set(skills_src, selected_skills)
writer_skill = paper_writer_skill(selected_skills)
staged_skill_names = list(selected_skills)
if include_review_skill and "review-paper" not in staged_skill_names:
    staged_skill_names.append("review-paper")
selected_skill_set = set(staged_skill_names)
formatted_skills_dir.mkdir(parents=True, exist_ok=True)
available_skill_names = {entry.name for entry in skills_src.iterdir() if entry.is_dir()}
for existing_dir in formatted_skills_dir.iterdir():
    if (
        existing_dir.is_dir()
        and existing_dir.name in available_skill_names
        and existing_dir.name not in selected_skill_set
    ):
        shutil.rmtree(existing_dir)

state_store.ensure_histories(campaign_results_dir)
state_store.ensure_global_memory_files(global_memory_dir)
global_memory_jsonl = state_store.global_memory_jsonl_path(global_memory_dir)
global_memory_recent = state_store.tail_jsonl(global_memory_jsonl, max_lines=80)

task_dir_display = Path("./task")
results_dir_display = Path("./results")
strategy_enabled = (Path(".") / ".paper_two_stage_enabled").exists()
if strategy_enabled:
    strategy_context = (
        "Two-stage writing is enabled for this run.\n"
        "Read the strategy artifacts before drafting the paper:\n"
        "- Strategy memo: `./paper_strategy.md`\n"
        "- Strategy JSON: `./paper_strategy.json`\n"
        "Treat them as the framing contract for the draft unless they contradict grounded evidence, in which case fail clearly."
    )
else:
    strategy_context = (
        "No precomputed paper strategy is provided for this run. "
        "Infer the framing directly from grounded inputs."
    )

validation_feedback_json = Path("validation_feedback.json")
validation_feedback_md = Path("validation_feedback.md")
if validation_feedback_json.is_file() or validation_feedback_md.is_file():
    validation_feedback_context = (
        "Validation feedback from a previous draft is present.\n"
        "Read these files before editing:\n"
        "- `./validation_feedback.json`\n"
        "- `./validation_feedback.md`\n"
        "Fix every listed hard error first, keep grounded content stable when possible, and rerun validation after each repair pass until the hard errors are gone.\n"
        "Treat figure readability findings as soft feedback unless a validator explicitly reports them as hard errors."
    )
else:
    validation_feedback_context = (
        "No prior validation feedback is present. Produce a fresh grounded draft, run validation, and fix any hard errors before finishing."
    )

review_feedback_json = Path("paper_review_feedback.json")
review_feedback_md = Path("paper_review_feedback.md")
if review_feedback_json.is_file() or review_feedback_md.is_file():
    review_feedback_context = (
        "Reviewer feedback from a prior draft is present.\n"
        "Read these files before editing:\n"
        "- `./paper_review_feedback.json`\n"
        "- `./paper_review_feedback.md`\n"
        "Treat this as a manuscript-only review repair pass. Address every major comment seriously and consider each minor comment when it improves the paper without drifting from grounded artifacts.\n"
        "Do not run new experiments, rerun evaluations, or invent new quantitative results. Revise only the manuscript text, framing, claims grounding, captions, tables, references, and narrative using the existing artifacts.\n"
        "If a comment cannot be fully resolved without new experiments, narrow the claim, add caveats, or acknowledge the limitation explicitly. Keep grounded content stable unless the current artifacts force a stronger correction."
    )
elif paper_review_rounds > 0:
    review_feedback_context = (
        "No reviewer feedback is present yet. A separate reviewer gate may write major and minor comments before validation is fully cleared or after it passes, and the manuscript will re-enter a text-only repair pass."
    )
else:
    review_feedback_context = "Paper review gate is disabled for this run."

extra = {
    "PAPER_FORMAT": paper_format,
    "PAPER_DIR": ".",
    "PAPER_DATA_PATH": "./paper_data.json",
    "PAPER_BIB_PATH": "./references.bib",
    "PAPER_REFERENCES_PROVENANCE_PATH": "./references_provenance.json",
    "PAPER_FIGURES_DIR": "./figures",
    "PAPER_FIGURE_PATH": "./figures/run_progression.png",
    "PAPER_AUTHOR_BLOCK_PATH": "./paper_authors.tex",
    "PAPER_TEX_PATH": "./paper.tex",
    "PAPER_CLAIMS_PATH": "./claims_grounding.json",
    "PAPER_MIN_REFS": paper_min_refs,
    "PAPER_MAX_REFS": paper_max_refs,
    "PAPER_WRITER_SKILL": writer_skill,
    "PAPER_MAIN_PROMPT_PATH": main_prompt_path.name,
    "REFERENCE_QUERY_PLAN_PATH": "./reference_query_plan.md",
    "REFERENCE_QUERY_PLAN_JSON_PATH": "./reference_query_plan.json",
    "PAPER_STRATEGY_PATH": "./paper_strategy.md",
    "PAPER_STRATEGY_JSON_PATH": "./paper_strategy.json",
    "PAPER_STRATEGY_CONTEXT": strategy_context,
    "PAPER_VALIDATION_FEEDBACK_CONTEXT": validation_feedback_context,
    "PAPER_AUTHOR_BLOCK_CONTEXT": author_block_context_text("./paper_authors.tex" if (Path(".") / "paper_authors.tex").is_file() else None),
    "CONTRIBUTIONS_FILE_PATH": "./contributions.md",
    "CONTRIBUTIONS_CONTEXT": contributions_context_text("./contributions.md" if (Path(".") / "contributions.md").is_file() else None),
    "VENUE_FILE_PATH": "./VENUE.md",
    "VENUE_CONTEXT": venue_context_text("./VENUE.md" if (Path(".") / "VENUE.md").is_file() else None),
    "PAPER_REVIEW_CONTEXT": review_feedback_context,
    "PAPER_REVIEW_PACKET_PATH": "./paper_review_packet.md",
    "PAPER_REVIEW_JSON_PATH": "./paper_review.json",
    "PAPER_REVIEW_MD_PATH": "./paper_review.md",
    "PAPER_REVIEW_FEEDBACK_JSON_PATH": "./paper_review_feedback.json",
    "PAPER_REVIEW_FEEDBACK_MD_PATH": "./paper_review_feedback.md",
    "PAPER_EVIDENCE_AUDIT_PACKET_PATH": "./paper_evidence_audit_packet.md",
    "PAPER_EVIDENCE_AUDIT_RAW_OUTPUT_PATH": "./paper_evidence_audit_raw.txt",
    "PAPER_EVIDENCE_AUDIT_JSON_PATH": "./paper_evidence_audit.json",
    "PAPER_EVIDENCE_AUDIT_MD_PATH": "./paper_evidence_audit.md",
    "PAPER_EVIDENCE_AUDIT_FEEDBACK_JSON_PATH": "./paper_evidence_audit_feedback.json",
    "PAPER_EVIDENCE_AUDIT_FEEDBACK_MD_PATH": "./paper_evidence_audit_feedback.md",
    "PAPER_EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH": "./paper_evidence_audit_stage_feedback.json",
    "PAPER_EVIDENCE_AUDIT_STAGE_FEEDBACK_MD_PATH": "./paper_evidence_audit_stage_feedback.md",
    "SCIENTIFIC_EVIDENCE_VERIFICATION_REPORT_PATH": "./scientific-evidence/verification-report.json",
    "GLOBAL_MEMORY_MD_FILE": "./global_memory/memory.md",
    "GLOBAL_MEMORY_JSONL_FILE": "./global_memory/memory.jsonl",
    "GLOBAL_MEMORY_RECENT_JSON": json.dumps(global_memory_recent, indent=2, sort_keys=True),
    "GLOBAL_MEMORY_SUMMARY_PATH": "./global_memory_summary.md",
    "PAPER_OPENAI_API_KEY_AVAILABLE": os.environ.get("PAPER_OPENAI_API_KEY_AVAILABLE", "false"),
    "PAPER_OPENAI_API_KEY_CONTEXT": os.environ.get("PAPER_OPENAI_API_KEY_CONTEXT", ""),
    "PAPER_GEMINI_API_KEY_AVAILABLE": os.environ.get("PAPER_GEMINI_API_KEY_AVAILABLE", "false"),
    "PAPER_GEMINI_API_KEY_CONTEXT": os.environ.get("PAPER_GEMINI_API_KEY_CONTEXT", ""),
}
extra.update(paper_workspace_context_replacements(Path(".")))

for skill_name in staged_skill_names:
    src_dir = skills_src / skill_name
    dst_dir = formatted_skills_dir / src_dir.name
    stage_skill_tree(src_dir, dst_dir)
    if skill_name == writer_skill:
        stage_paper_templates(skills_src, dst_dir)
    src_skill = src_dir / "SKILL.md"
    if src_skill.is_file():
        rendered = load_instructions(src_skill, task_dir_display, results_dir_display, extra)
        (dst_dir / "SKILL.md").write_text(rendered, encoding="utf-8")

watermark_staged_templates(formatted_skills_dir / writer_skill)

# Copy format-specific TeX support files to the paper workspace root so pdflatex finds them.
# Keep paper.tex in the template directory; the writer uses it as a scaffold, not as an output.
fmt_dir = paper_templates_dir(skills_src) / paper_format
for pattern in ("*.sty", "*.bst", "*.cls", "*.png", "*.jpg", "*.jpeg", "*.pdf", "*.eps"):
    for support_file in fmt_dir.glob(pattern):
        shutil.copy2(support_file, Path("."))
PY
  )
}

stage_paper_skills "$CURRENT_SELECTED_PAPER_SKILLS" "$CURRENT_PAPER_INSTRUCT_PATH" "$INCLUDE_REVIEW_SKILL"

resolve_final_paper_stage_from_strategy() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$CURRENT_PAPER_INSTRUCT_PATH" "$CURRENT_SELECTED_PAPER_SKILLS" "$OUTPUT_DIR/paper_strategy.json" "$OUTPUT_DIR/paper_strategy_resolution.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

script_dir = Path(sys.argv[1])
current_prompt_path = Path(sys.argv[2])
current_selected_skills_csv = sys.argv[3]
strategy_json_path = Path(sys.argv[4])
resolution_path = Path(sys.argv[5])

sys.path.insert(0, str(script_dir))
from src.investigation.paper_prompt_family import resolve_final_paper_stage  # noqa: E402
from src.investigation.skill_scope import parse_skill_csv  # noqa: E402

strategy_payload = json.loads(strategy_json_path.read_text(encoding="utf-8"))
selected_prompt_family = str(strategy_payload.get("selected_prompt_family", "")).strip()
resolved = resolve_final_paper_stage(
    repo_root=script_dir,
    current_prompt_path=current_prompt_path,
    current_selected_skills=parse_skill_csv(current_selected_skills_csv),
    selected_prompt_family=selected_prompt_family,
    resolution_path=resolution_path,
)
print(str(resolved.prompt_path))
print(",".join(resolved.selected_skills))
print(resolved.writer_skill)
print(resolved.prompt_family)
PY

  CURRENT_PAPER_INSTRUCT_PATH="$("$PYTHON_BIN" - "$OUTPUT_DIR/paper_strategy_resolution.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

resolution_path = Path(sys.argv[1])
payload = json.loads(resolution_path.read_text(encoding="utf-8"))
print(str(payload["prompt_path"]))
PY
)"

  CURRENT_SELECTED_PAPER_SKILLS="$("$PYTHON_BIN" - "$OUTPUT_DIR/paper_strategy_resolution.json" <<'PY'
import json
import os
import sys
from pathlib import Path

resolution_path = Path(sys.argv[1])
payload = json.loads(resolution_path.read_text(encoding="utf-8"))
print(",".join(str(skill) for skill in payload["selected_skills"]))
PY
)"

  PAPER_WRITER_SKILL="$("$PYTHON_BIN" - "$OUTPUT_DIR/paper_strategy_resolution.json" <<'PY'
import json
import os
import sys
from pathlib import Path

resolution_path = Path(sys.argv[1])
payload = json.loads(resolution_path.read_text(encoding="utf-8"))
print(str(payload["writer_skill"]))
PY
)"

  CURRENT_PAPER_PROMPT_FAMILY="$("$PYTHON_BIN" - "$OUTPUT_DIR/paper_strategy_resolution.json" <<'PY'
import json
import sys
from pathlib import Path

resolution_path = Path(sys.argv[1])
payload = json.loads(resolution_path.read_text(encoding="utf-8"))
print(str(payload["prompt_family"]))
PY
)"

  stage_paper_skills "$CURRENT_SELECTED_PAPER_SKILLS" "$CURRENT_PAPER_INSTRUCT_PATH" "$INCLUDE_REVIEW_SKILL"
  echo "[paper.sh] resolved final paper family: $CURRENT_PAPER_PROMPT_FAMILY"
  echo "[paper.sh] resolved final paper prompt: $CURRENT_PAPER_INSTRUCT_PATH"
  echo "[paper.sh] resolved final paper skills: $CURRENT_SELECTED_PAPER_SKILLS"
}

render_prompt() {
  local template_path="$1"
  local main_prompt_path="$2"
  stage_venue_file "$VENUE_SOURCE_PATH" "$PAPER_FORMAT" "$PAPER_MIN_REFS" "$PAPER_MAX_REFS"
  stage_contributions_file "$CONTRIBUTIONS_SOURCE_PATH" "$PAPER_FORMAT" "$PAPER_MIN_REFS" "$PAPER_MAX_REFS"

  (
    cd "$OUTPUT_DIR"
    "$PYTHON_BIN" - "$REPO_ROOT" "$template_path" "$main_prompt_path" "$RESULTS_DIR" "$GLOBAL_MEMORY_DIR" "$PAPER_MIN_REFS" "$PAPER_MAX_REFS" "$PAPER_FORMAT" "$PAPER_WRITER_SKILL" "$PAPER_REVIEW_ROUNDS" <<'PY'
import json
import os
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
instruct_path = Path(sys.argv[2])
main_prompt_path = Path(sys.argv[3])
campaign_results_dir = Path(sys.argv[4])
global_memory_dir = Path(sys.argv[5])
paper_min_refs = sys.argv[6]
paper_max_refs = sys.argv[7]
paper_format = sys.argv[8]
paper_writer_skill = sys.argv[9]
paper_review_rounds = int(sys.argv[10])

sys.path.insert(0, str(script_dir))
from src.orchestrator import state as state_store  # noqa: E402
from src.investigation.paper_prompt_context import (  # noqa: E402
    author_block_context_text,
    contributions_context_text,
    paper_workspace_context_replacements,
    venue_context_text,
)
from src.investigation.files import load_instructions  # noqa: E402

state_store.ensure_histories(campaign_results_dir)
state_store.ensure_global_memory_files(global_memory_dir)
global_memory_jsonl = state_store.global_memory_jsonl_path(global_memory_dir)
global_memory_recent = state_store.tail_jsonl(global_memory_jsonl, max_lines=80)

task_dir_display = Path("./task")
results_dir_display = Path("./results")
strategy_enabled = Path(".paper_two_stage_enabled").exists()
strategy_ready = Path("paper_strategy.md").is_file() and Path("paper_strategy.json").is_file()
if strategy_enabled and strategy_ready:
    strategy_context = (
        "Two-stage writing is enabled for this run.\n"
        "Read the strategy artifacts before drafting the paper:\n"
        "- Strategy memo: `./paper_strategy.md`\n"
        "- Strategy JSON: `./paper_strategy.json`\n"
        "Treat them as the framing contract for the draft unless they contradict grounded evidence, in which case fail clearly."
    )
elif strategy_enabled:
    strategy_context = (
        "Two-stage writing is enabled for this run, but the framing artifacts have not been written yet.\n"
        "If you are the framing stage, produce `./paper_strategy.md` and `./paper_strategy.json`.\n"
        "If you are the final drafting stage and those files are still missing, fail clearly."
    )
else:
    strategy_context = (
        "No precomputed paper strategy is provided for this run. "
        "Infer the framing directly from grounded inputs."
    )

prompt_name = instruct_path.name
if prompt_name == "paper_strategy.md":
    validation_feedback_json = Path("strategy_validation_feedback.json")
    validation_feedback_md = Path("strategy_validation_feedback.md")
else:
    validation_feedback_json = Path("validation_feedback.json")
    validation_feedback_md = Path("validation_feedback.md")
if validation_feedback_json.is_file() or validation_feedback_md.is_file():
    validation_feedback_context = (
        "Validation feedback from a previous draft is present.\n"
        "Read these files before editing:\n"
        f"- `./{validation_feedback_json.name}`\n"
        f"- `./{validation_feedback_md.name}`\n"
        "Fix every listed hard error first, keep grounded content stable when possible, and rerun validation after each repair pass until the hard errors are gone.\n"
        "Treat figure readability findings as soft feedback unless a validator explicitly reports them as hard errors."
    )
else:
    validation_feedback_context = (
        "No prior validation feedback is present. Produce a fresh grounded draft, run validation, and fix any hard errors before finishing."
    )

review_feedback_json = Path("paper_review_feedback.json")
review_feedback_md = Path("paper_review_feedback.md")
if review_feedback_json.is_file() or review_feedback_md.is_file():
    review_feedback_context = (
        "Reviewer feedback from a prior draft is present.\n"
        "Read these files before editing:\n"
        "- `./paper_review_feedback.json`\n"
        "- `./paper_review_feedback.md`\n"
        "Treat this as a manuscript-only review repair pass. Address every major comment seriously and consider each minor comment when it improves the paper without drifting from grounded artifacts.\n"
        "Do not run new experiments, rerun evaluations, or invent new quantitative results. Revise only the manuscript text, framing, claims grounding, captions, tables, references, and narrative using the existing artifacts.\n"
        "If a comment cannot be fully resolved without new experiments, narrow the claim, add caveats, or acknowledge the limitation explicitly. Keep grounded content stable unless the current artifacts force a stronger correction."
    )
elif paper_review_rounds > 0:
    review_feedback_context = (
        "No reviewer feedback is present yet. A separate reviewer gate may write major and minor comments before validation is fully cleared or after it passes, and the manuscript will re-enter a text-only repair pass."
    )
else:
    review_feedback_context = "Paper review gate is disabled for this run."

extra = {
    "PAPER_FORMAT": paper_format,
    "PAPER_DIR": ".",
    "PAPER_DATA_PATH": "./paper_data.json",
    "PAPER_BIB_PATH": "./references.bib",
    "PAPER_REFERENCES_PROVENANCE_PATH": "./references_provenance.json",
    "PAPER_FIGURES_DIR": "./figures",
    "PAPER_FIGURE_PATH": "./figures/run_progression.png",
    "PAPER_AUTHOR_BLOCK_PATH": "./paper_authors.tex",
    "PAPER_TEX_PATH": "./paper.tex",
    "PAPER_CLAIMS_PATH": "./claims_grounding.json",
    "PAPER_MIN_REFS": paper_min_refs,
    "PAPER_MAX_REFS": paper_max_refs,
    "PAPER_WRITER_SKILL": paper_writer_skill,
    "PAPER_MAIN_PROMPT_PATH": main_prompt_path.name,
    "REFERENCE_QUERY_PLAN_PATH": "./reference_query_plan.md",
    "REFERENCE_QUERY_PLAN_JSON_PATH": "./reference_query_plan.json",
    "PAPER_STRATEGY_PATH": "./paper_strategy.md",
    "PAPER_STRATEGY_JSON_PATH": "./paper_strategy.json",
    "PAPER_STRATEGY_CONTEXT": strategy_context,
    "PAPER_VALIDATION_FEEDBACK_CONTEXT": validation_feedback_context,
    "PAPER_AUTHOR_BLOCK_CONTEXT": author_block_context_text("./paper_authors.tex" if Path("paper_authors.tex").is_file() else None),
    "CONTRIBUTIONS_FILE_PATH": "./contributions.md",
    "CONTRIBUTIONS_CONTEXT": contributions_context_text("./contributions.md" if Path("contributions.md").is_file() else None),
    "VENUE_FILE_PATH": "./VENUE.md",
    "VENUE_CONTEXT": venue_context_text("./VENUE.md" if Path("VENUE.md").is_file() else None),
    "PAPER_REVIEW_CONTEXT": review_feedback_context,
    "PAPER_REVIEW_PACKET_PATH": "./paper_review_packet.md",
    "PAPER_REVIEW_JSON_PATH": "./paper_review.json",
    "PAPER_REVIEW_MD_PATH": "./paper_review.md",
    "PAPER_REVIEW_FEEDBACK_JSON_PATH": "./paper_review_feedback.json",
    "PAPER_REVIEW_FEEDBACK_MD_PATH": "./paper_review_feedback.md",
    "PAPER_EVIDENCE_AUDIT_PACKET_PATH": "./paper_evidence_audit_packet.md",
    "PAPER_EVIDENCE_AUDIT_RAW_OUTPUT_PATH": "./paper_evidence_audit_raw.txt",
    "PAPER_EVIDENCE_AUDIT_JSON_PATH": "./paper_evidence_audit.json",
    "PAPER_EVIDENCE_AUDIT_MD_PATH": "./paper_evidence_audit.md",
    "PAPER_EVIDENCE_AUDIT_FEEDBACK_JSON_PATH": "./paper_evidence_audit_feedback.json",
    "PAPER_EVIDENCE_AUDIT_FEEDBACK_MD_PATH": "./paper_evidence_audit_feedback.md",
    "PAPER_EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH": "./paper_evidence_audit_stage_feedback.json",
    "PAPER_EVIDENCE_AUDIT_STAGE_FEEDBACK_MD_PATH": "./paper_evidence_audit_stage_feedback.md",
    "SCIENTIFIC_EVIDENCE_VERIFICATION_REPORT_PATH": "./scientific-evidence/verification-report.json",
    "GLOBAL_MEMORY_MD_FILE": "./global_memory/memory.md",
    "GLOBAL_MEMORY_JSONL_FILE": "./global_memory/memory.jsonl",
    "GLOBAL_MEMORY_RECENT_JSON": json.dumps(global_memory_recent, indent=2, sort_keys=True),
    "GLOBAL_MEMORY_SUMMARY_PATH": "./global_memory_summary.md",
    "PAPER_OPENAI_API_KEY_AVAILABLE": os.environ.get("PAPER_OPENAI_API_KEY_AVAILABLE", "false"),
    "PAPER_OPENAI_API_KEY_CONTEXT": os.environ.get("PAPER_OPENAI_API_KEY_CONTEXT", ""),
    "PAPER_GEMINI_API_KEY_AVAILABLE": os.environ.get("PAPER_GEMINI_API_KEY_AVAILABLE", "false"),
    "PAPER_GEMINI_API_KEY_CONTEXT": os.environ.get("PAPER_GEMINI_API_KEY_CONTEXT", ""),
}
extra.update(paper_workspace_context_replacements(Path(".")))

prompt = load_instructions(instruct_path, task_dir_display, results_dir_display, extra)
print(prompt)
PY
  )
}

run_prompt_stage() {
  local stage_name="$1"
  local prompt_text="$2"
  local prompt_path="$3"
  local log_path="$4"
  shift 4
  local -a expected_outputs=("$@")
  local stage_log_pipe="$OUTPUT_DIR/.${stage_name}_stream_${TIMESTAMP}.fifo"
  local stage_agent_exit_code=0
  local stage_tee_exit_code=0
  local stage_forced_success=0
  local expected_output=""
  local expected_output_path=""
  local agent_prompt_file=""
  local cost_invocation="paper:${TIMESTAMP}:${stage_name}:${log_path##*/}"

  if ! write_agent_prompt_file "$prompt_path" "$prompt_text"; then
    echo "[paper.sh] failed to stage prompt for '$stage_name': $prompt_path" >&2
    return 1
  fi
  # Rendered paper prompts can exceed Linux's per-argument limit; pass the
  # inspectable prompt artifact as a file instead of argv.
  agent_prompt_file="$(relative_agent_prompt_path "$prompt_path" "$OUTPUT_DIR")"
  ACTIVE_COST_LOG="$log_path"
  ACTIVE_COST_STAGE="$stage_name"
  ACTIVE_COST_INVOCATION="$cost_invocation"
  LOG_PIPE="$stage_log_pipe"
  mkfifo "$stage_log_pipe"
  bounded_agent_log_writer "$PYTHON_BIN" "$log_path" "$AGENT_LOG_MAX_BYTES" \
    < "$stage_log_pipe" &
  TEE_PID=$!

  (
    exec 3>&-
    AGENT_WORKDIR_READONLY="$PAPER_AGENT_WORKDIR_READONLY" \
    AGENT_CONTAINER_EXTRA_MOUNTS="$PAPER_AGENT_CONTAINER_EXTRA_MOUNTS" \
    AGENT_PROMPT_FILE="$agent_prompt_file" \
    run_agent_prompt \
      "$AGENT" \
      "$prompt_text" \
      "$AGENT_MODEL" \
      "$AGENT_PROFILE" \
      "$OUTPUT_DIR" \
      "$CONTAINERIZE" \
      "$CONTAINER_IMAGE" \
      "$CONTAINER_NETWORK" \
      "$CONTAINER_HOME_DIR" \
      "$CONTAINER_ENV_FILE" \
      "$PAPER_CONTAINER_MOUNT_ROOT"
  ) > "$stage_log_pipe" 2>&1 &
  AGENT_PID=$!

  set +e
  if [ -n "${AGENT_PID:-}" ]; then
    while kill -0 "$AGENT_PID" 2>/dev/null; do
      paper_maybe_heartbeat
      if stage_expected_outputs_ready "$OUTPUT_DIR" "${expected_outputs[@]}" \
        && stage_log_reports_success "$log_path"; then
        echo "[paper.sh] stage '$stage_name' produced required outputs and success result while agent still running; terminating lingering agent process tree" >&2
        terminate_pid_tree "$AGENT_PID" 0.2
        stage_forced_success=1
        stage_agent_exit_code=0
        break
      fi
      sleep 1
    done
    if [ "$stage_forced_success" -eq 0 ]; then
      wait "$AGENT_PID"
      stage_agent_exit_code=$?
    fi
  fi
  if [ -n "${TEE_PID:-}" ]; then
    wait "$TEE_PID"
    stage_tee_exit_code=$?
  fi
  set -e

  record_headless_cost_log \
    "$PYTHON_BIN" "$log_path" "$stage_name" \
    "$cost_invocation" \
    "$(headless_cost_status "$stage_agent_exit_code" "$INTERRUPTED")"
  ACTIVE_COST_LOG=""
  ACTIVE_COST_STAGE=""
  ACTIVE_COST_INVOCATION=""

  AGENT_PID=""
  TEE_PID=""
  if [ -p "$stage_log_pipe" ]; then
    rm -f "$stage_log_pipe"
  fi
  LOG_PIPE=""

  if [ "$INTERRUPTED" -eq 1 ]; then
    return 130
  fi
  if [ "$stage_agent_exit_code" -ne 0 ]; then
    return "$stage_agent_exit_code"
  fi
  if [ "$stage_tee_exit_code" -ne 0 ]; then
    return "$stage_tee_exit_code"
  fi
  if [ ! -s "$log_path" ]; then
    echo "[paper.sh] stage '$stage_name' returned success but produced an empty log: $log_path" >&2
    return 70
  fi
  for expected_output in "${expected_outputs[@]}"; do
    if [ -z "$expected_output" ]; then
      continue
    fi
    if [[ "$expected_output" = /* ]]; then
      expected_output_path="$expected_output"
    else
      expected_output_path="$OUTPUT_DIR/$expected_output"
    fi
    if [ ! -s "$expected_output_path" ]; then
      echo "[paper.sh] stage '$stage_name' returned success but required output is missing or empty: $expected_output_path" >&2
      return 71
    fi
  done
  return 0
}

stage_expected_outputs_ready() {
  local output_dir="$1"
  shift
  local expected_output=""
  local expected_output_path=""

  for expected_output in "$@"; do
    if [ -z "$expected_output" ]; then
      continue
    fi
    if [[ "$expected_output" = /* ]]; then
      expected_output_path="$expected_output"
    else
      expected_output_path="$output_dir/$expected_output"
    fi
    if [ ! -s "$expected_output_path" ]; then
      return 1
    fi
  done
  return 0
}

stage_log_reports_success() {
  local log_path="$1"
  if [ ! -s "$log_path" ]; then
    return 1
  fi
  if ! grep -F '"type":"result"' "$log_path" >/dev/null 2>&1; then
    return 1
  fi
  if ! grep -F '"subtype":"success"' "$log_path" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

write_stage_execution_feedback() {
  local stage_name="$1"
  local feedback_json_path="$2"
  local feedback_md_path="$3"
  local prompt_path="$4"
  local log_path="$5"
  local stage_return_code="$6"
  shift 6

  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$feedback_json_path" "$feedback_md_path" "$stage_name" "$stage_return_code" "$prompt_path" "$log_path" "$@" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
feedback_json_path = Path(sys.argv[3])
feedback_md_path = Path(sys.argv[4])
stage_name = sys.argv[5]
stage_return_code = int(sys.argv[6])
prompt_path = Path(sys.argv[7])
log_path = Path(sys.argv[8])
expected_outputs = list(sys.argv[9:])

sys.path.insert(0, str(script_dir))

from src.orchestrator.paper_feedback import write_stage_execution_feedback  # noqa: E402

write_stage_execution_feedback(
    output_dir=output_dir,
    feedback_json_path=feedback_json_path,
    feedback_md_path=feedback_md_path,
    stage_name=stage_name,
    stage_return_code=stage_return_code,
    prompt_path=prompt_path,
    log_path=log_path,
    expected_outputs=expected_outputs,
)
PY
}

write_stage_contract_feedback() {
  local stage_name="$1"
  local feedback_json_path="$2"
  local feedback_md_path="$3"
  local prompt_path="$4"
  local log_path="$5"
  local error_message="$6"
  shift 6

  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$feedback_json_path" "$feedback_md_path" "$stage_name" "$prompt_path" "$log_path" "$error_message" "$@" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
feedback_json_path = Path(sys.argv[3])
feedback_md_path = Path(sys.argv[4])
stage_name = sys.argv[5]
prompt_path = Path(sys.argv[6])
log_path = Path(sys.argv[7])
error_message = sys.argv[8]
expected_outputs = list(sys.argv[9:])

sys.path.insert(0, str(script_dir))

from src.orchestrator.paper_feedback import write_stage_contract_feedback  # noqa: E402

write_stage_contract_feedback(
    output_dir=output_dir,
    feedback_json_path=feedback_json_path,
    feedback_md_path=feedback_md_path,
    stage_name=stage_name,
    prompt_path=prompt_path,
    log_path=log_path,
    expected_outputs=expected_outputs,
    hard_errors=[error_message],
)
PY
}

run_prompt_stage_with_feedback() {
  local stage_name="$1"
  local prompt_text="$2"
  local prompt_path="$3"
  local log_path="$4"
  local feedback_json_path="$5"
  local feedback_md_path="$6"
  shift 6
  local -a expected_outputs=("$@")
  local stage_return_code=0

  set +e
  run_prompt_stage "$stage_name" "$prompt_text" "$prompt_path" "$log_path" "${expected_outputs[@]}"
  stage_return_code=$?
  set -e
  if [ "$stage_return_code" -eq 0 ]; then
    return 0
  fi
  if [ "$stage_return_code" -eq 130 ]; then
    return 130
  fi
  write_stage_execution_feedback \
    "$stage_name" \
    "$feedback_json_path" \
    "$feedback_md_path" \
    "$prompt_path" \
    "$log_path" \
    "$stage_return_code" \
    "${expected_outputs[@]}"
  echo "[paper.sh] stage '$stage_name' failed before validation; wrote feedback: $feedback_json_path" >&2
  return "$stage_return_code"
}

AGENT_PID=""
TEE_PID=""
LOG_PIPE=""
INTERRUPTED=0
CLEANUP_IN_PROGRESS=0
ACTIVE_READONLY_AUDIT_WORKSPACE=""
ACTIVE_COST_LOG=""
ACTIVE_COST_STAGE=""
ACTIVE_COST_INVOCATION=""
FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR=""
FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256=""
FIGURE_QUALITY_REPAIR_AGENT_QUIESCED=0
export GLOBAL_MEMORY_JSONL_FILE="$OUTPUT_DIR/global_memory/memory.jsonl"
export GLOBAL_MEMORY_MD_FILE="$OUTPUT_DIR/global_memory/memory.md"

cleanup_run_processes() {
  local agent_session_stopped=1
  if [ "${CLEANUP_IN_PROGRESS:-0}" -eq 1 ]; then
    return 0
  fi
  CLEANUP_IN_PROGRESS=1

  if [ -n "${AGENT_PID:-}" ] && kill -0 "$AGENT_PID" 2>/dev/null; then
    terminate_pid_tree "$AGENT_PID" 0.2
  fi
  AGENT_PID=""

  if [ -n "${TEE_PID:-}" ] && kill -0 "$TEE_PID" 2>/dev/null; then
    kill -TERM "$TEE_PID" 2>/dev/null || true
    wait "$TEE_PID" 2>/dev/null || true
  fi
  TEE_PID=""

  if [ "$INTERRUPTED" -eq 1 ] && [ -n "${ACTIVE_COST_LOG:-}" ]; then
    record_headless_cost_log \
      "$PYTHON_BIN" "$ACTIVE_COST_LOG" "$ACTIVE_COST_STAGE" \
      "$ACTIVE_COST_INVOCATION" interrupted
    ACTIVE_COST_LOG=""
    ACTIVE_COST_STAGE=""
    ACTIVE_COST_INVOCATION=""
  fi

  if [ "$CONTAINERIZE" -eq 1 ] && [ -n "${AGENT_CONTAINER_SESSION_NAME:-}" ]; then
    if [ -n "${FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR:-}" ]; then
      if paper_figure_quality_stop_agent_session \
        "$CONTAINERIZE" "$AGENT_CONTAINER_SESSION_NAME"; then
        FIGURE_QUALITY_REPAIR_AGENT_QUIESCED=1
      else
        agent_session_stopped=0
      fi
    elif ! stop_container_session_force "$AGENT_CONTAINER_SESSION_NAME"; then
      agent_session_stopped=0
    fi
  elif [ -n "${FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR:-}" ] \
    && [ "$FIGURE_QUALITY_REPAIR_AGENT_QUIESCED" -ne 1 ]; then
    agent_session_stopped=0
  fi

  if [ -n "${LOG_PIPE:-}" ] && [ -p "$LOG_PIPE" ]; then
    rm -f "$LOG_PIPE"
  fi
  LOG_PIPE=""
  if [ -n "${ACTIVE_READONLY_AUDIT_WORKSPACE:-}" ]; then
    cleanup_readonly_paper_audit_workspace "$ACTIVE_READONLY_AUDIT_WORKSPACE" || true
    ACTIVE_READONLY_AUDIT_WORKSPACE=""
  fi
  if [ -n "${FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR:-}" ] \
    && [ "$agent_session_stopped" -eq 1 ]; then
    if paper_figure_quality_restore_workspace \
      "$FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR" "$OUTPUT_DIR" \
      "$FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256"; then
      FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR=""
      FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256=""
    fi
  fi
  cleanup_concept_figure_state
}

on_interrupt() {
  trap - INT TERM
  INTERRUPTED=1
  echo "Interrupt received. Shutting down agent and log stream..." >&2
  cleanup_run_processes
  exit 130
}

cleanup_paper_run() {
  local exit_code=$?
  paper_event_on_exit "$exit_code"
  cleanup_run_processes || true
  finish_paper_diagnostic_sink || true
  return "$exit_code"
}

trap on_interrupt INT TERM
trap cleanup_paper_run EXIT

echo "Paper directory: $OUTPUT_DIR"
echo "Agent config directory: $OUTPUT_DIR/$AGENT_CONFIG_DIR"
if [ -n "$AGENT_MODEL" ]; then
  echo "Agent model override: $AGENT_MODEL"
fi
if [ -n "$AGENT_REASONING_EFFORT" ]; then
  echo "Agent reasoning effort override: $AGENT_REASONING_EFFORT"
fi
if [ "$TWO_STAGE" -eq 1 ]; then
  echo "Two-stage writing: enabled"
  echo "Strategy prompt template: $PAPER_STRATEGY_INSTRUCT_PATH"
else
  echo "Two-stage writing: disabled"
fi
echo "Agentic evidence audit gate: enabled"
echo "Evidence audit prompt template: $PAPER_EVIDENCE_AUDIT_INSTRUCT_PATH"
if [ "$PAPER_REVIEW_ROUNDS" -gt 0 ]; then
  echo "Paper review gate: enabled (planned review/repair cycles=$PAPER_REVIEW_ROUNDS)"
  echo "Review prompt template: $PAPER_REVIEW_INSTRUCT_PATH"
else
  echo "Paper review gate: disabled"
fi
if [ "$CONTAINERIZE" -eq 1 ]; then
  CONTAINER_HOME_DIR="$(prepare_container_home "$RESULTS_DIR")"
  seed_auth_into_home "$AGENT" "$CONTAINER_AUTH_ROOT" "$CONTAINER_HOME_DIR"
  echo "Containerized run: image=$CONTAINER_IMAGE network=$CONTAINER_NETWORK"
  echo "Container home directory: $CONTAINER_HOME_DIR"
  if [ -n "${AGENT_CONTAINER_SESSION_NAME:-}" ]; then
    echo "Container session: ${AGENT_CONTAINER_SESSION_NAME}"
  fi
  if [ -n "$CONTAINER_ENV_FILE" ]; then
    echo "Container env file: $CONTAINER_ENV_FILE"
  fi
else
  CONTAINER_HOME_DIR=""
fi

REFERENCE_QUERY_PLAN_LOG_FILE="$OUTPUT_DIR/reference_query_plan_agent_${AGENT}_${TIMESTAMP}.log"
REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_JSON_PATH="$OUTPUT_DIR/reference_query_plan_validation_feedback.json"
REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_MD_PATH="$OUTPUT_DIR/reference_query_plan_validation_feedback.md"
REFERENCE_RETRIEVAL_LOG_FILE="$OUTPUT_DIR/reference_retrieval.log"
DRAFT_REFERENCE_GAPS_LOG_FILE="$OUTPUT_DIR/draft_reference_gaps_agent_${AGENT}_${TIMESTAMP}.log"
DRAFT_REFERENCE_GAPS_VALIDATION_FEEDBACK_JSON_PATH="$OUTPUT_DIR/draft_reference_gaps_validation_feedback.json"
DRAFT_REFERENCE_GAPS_VALIDATION_FEEDBACK_MD_PATH="$OUTPUT_DIR/draft_reference_gaps_validation_feedback.md"
POSTDRAFT_REFERENCE_QUERY_PLAN_LOG_FILE="$OUTPUT_DIR/postdraft_reference_query_plan_agent_${AGENT}_${TIMESTAMP}.log"
POSTDRAFT_REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_JSON_PATH="$OUTPUT_DIR/postdraft_reference_query_plan_validation_feedback.json"
POSTDRAFT_REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_MD_PATH="$OUTPUT_DIR/postdraft_reference_query_plan_validation_feedback.md"
POSTDRAFT_REFERENCE_RETRIEVAL_LOG_FILE="$OUTPUT_DIR/postdraft_reference_retrieval.log"
POSTDRAFT_REFERENCE_ENRICHMENT_MARKER_PATH="$OUTPUT_DIR/.postdraft_reference_enrichment_complete"
STRATEGY_LOG_FILE="$OUTPUT_DIR/paper_strategy_agent_${AGENT}_${TIMESTAMP}.log"
PAPER_LOG_FILE="$OUTPUT_DIR/paper_agent_${AGENT}_${TIMESTAMP}.log"
REVIEW_LOG_FILE="$OUTPUT_DIR/paper_review_agent_${AGENT}_${TIMESTAMP}.log"
REVIEW_STAGE_FEEDBACK_JSON_PATH="$OUTPUT_DIR/paper_review_stage_feedback.json"
REVIEW_STAGE_FEEDBACK_MD_PATH="$OUTPUT_DIR/paper_review_stage_feedback.md"
REVIEW_PACKET_PATH="$OUTPUT_DIR/paper_review_packet.md"
REVIEW_JSON_PATH="$OUTPUT_DIR/paper_review.json"
REVIEW_MD_PATH="$OUTPUT_DIR/paper_review.md"
REVIEW_FEEDBACK_JSON_PATH="$OUTPUT_DIR/paper_review_feedback.json"
REVIEW_FEEDBACK_MD_PATH="$OUTPUT_DIR/paper_review_feedback.md"
EVIDENCE_AUDIT_LOG_FILE="$OUTPUT_DIR/paper_evidence_audit_agent_${AGENT}_${TIMESTAMP}.log"
EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH="$OUTPUT_DIR/paper_evidence_audit_stage_feedback.json"
EVIDENCE_AUDIT_STAGE_FEEDBACK_MD_PATH="$OUTPUT_DIR/paper_evidence_audit_stage_feedback.md"
EVIDENCE_AUDIT_PACKET_PATH="$OUTPUT_DIR/paper_evidence_audit_packet.md"
EVIDENCE_AUDIT_INPUT_HASHES_PATH="$OUTPUT_DIR/paper_evidence_audit_input_sha256.json"
EVIDENCE_AUDIT_RAW_OUTPUT_PATH="$OUTPUT_DIR/paper_evidence_audit_raw.txt"
EVIDENCE_AUDIT_JSON_PATH="$OUTPUT_DIR/paper_evidence_audit.json"
EVIDENCE_AUDIT_MD_PATH="$OUTPUT_DIR/paper_evidence_audit.md"
EVIDENCE_AUDIT_FEEDBACK_JSON_PATH="$OUTPUT_DIR/paper_evidence_audit_feedback.json"
EVIDENCE_AUDIT_FEEDBACK_MD_PATH="$OUTPUT_DIR/paper_evidence_audit_feedback.md"
STRATEGY_VALIDATION_FEEDBACK_JSON_PATH="$OUTPUT_DIR/strategy_validation_feedback.json"
STRATEGY_VALIDATION_FEEDBACK_MD_PATH="$OUTPUT_DIR/strategy_validation_feedback.md"

validate_reference_query_plan_with_feedback() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_JSON_PATH" "$REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_MD_PATH" <<'PY'
import json
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
feedback_json_path = Path(sys.argv[3])
feedback_md_path = Path(sys.argv[4])
plan_md_path = output_dir / "reference_query_plan.md"
plan_json_path = output_dir / "reference_query_plan.json"

sys.path.insert(0, str(script_dir))
from src.orchestrator.reference_query_plan import (  # noqa: E402
    write_reference_query_plan_validation_feedback,
)

feedback = write_reference_query_plan_validation_feedback(
    output_dir=output_dir,
    feedback_json_path=feedback_json_path,
    feedback_md_path=feedback_md_path,
    plan_md_path=plan_md_path,
    plan_json_path=plan_json_path,
)
hard_errors = feedback.get("hard_errors", [])
if isinstance(hard_errors, list) and hard_errors:
    print("[paper.sh] reference query plan validation errors:")
    for message in hard_errors:
        print(f"  - {message}")
    raise SystemExit(1)
print(f"[paper.sh] reference query plan ok: {plan_json_path}")
PY
}

validate_postdraft_reference_query_plan_with_feedback() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$POSTDRAFT_REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_JSON_PATH" "$POSTDRAFT_REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_MD_PATH" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
feedback_json_path = Path(sys.argv[3])
feedback_md_path = Path(sys.argv[4])
plan_md_path = output_dir / "postdraft_reference_query_plan.md"
plan_json_path = output_dir / "postdraft_reference_query_plan.json"

sys.path.insert(0, str(script_dir))
from src.orchestrator.reference_query_plan import (  # noqa: E402
    write_reference_query_plan_validation_feedback,
)

feedback = write_reference_query_plan_validation_feedback(
    output_dir=output_dir,
    feedback_json_path=feedback_json_path,
    feedback_md_path=feedback_md_path,
    plan_md_path=plan_md_path,
    plan_json_path=plan_json_path,
)
hard_errors = feedback.get("hard_errors", [])
if isinstance(hard_errors, list) and hard_errors:
    print("[paper.sh] postdraft reference query plan validation errors:")
    for message in hard_errors:
        print(f"  - {message}")
    raise SystemExit(1)
print(f"[paper.sh] postdraft reference query plan ok: {plan_json_path}")
PY
}

validate_draft_reference_gaps_with_feedback() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$DRAFT_REFERENCE_GAPS_VALIDATION_FEEDBACK_JSON_PATH" "$DRAFT_REFERENCE_GAPS_VALIDATION_FEEDBACK_MD_PATH" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
feedback_json_path = Path(sys.argv[3])
feedback_md_path = Path(sys.argv[4])
gaps_md_path = output_dir / "draft_reference_gaps.md"
gaps_json_path = output_dir / "draft_reference_gaps.json"

sys.path.insert(0, str(script_dir))
from src.orchestrator.paper_reference_enrichment import (  # noqa: E402
    write_draft_reference_gaps_validation_feedback,
)

feedback = write_draft_reference_gaps_validation_feedback(
    output_dir=output_dir,
    feedback_json_path=feedback_json_path,
    feedback_md_path=feedback_md_path,
    gaps_md_path=gaps_md_path,
    gaps_json_path=gaps_json_path,
)
hard_errors = feedback.get("hard_errors", [])
if isinstance(hard_errors, list) and hard_errors:
    print("[paper.sh] draft reference gap validation errors:")
    for message in hard_errors:
        print(f"  - {message}")
    raise SystemExit(1)
print(f"[paper.sh] draft reference gaps ok: {gaps_json_path}")
PY
}

retrieve_references_from_plan() {
  local query_plan_path="$1"
  "$PYTHON_BIN" - "$REPO_ROOT" "$TASK_DIR" "$RESULTS_DIR" "$GLOBAL_MEMORY_DIR" "$SOURCE_RESULTS_DIR" "$OUTPUT_DIR" "$PAPER_MIN_REFS" "$PAPER_MAX_REFS" "$PAPER_PROFILE" "$query_plan_path" "$FINAL_EVALUATION_DIR" "$FINAL_EVALUATION_MODE" "$CONTRACT_TASK_DIR" "$INVESTIGATION_RESULTS_DIR" "$ABLATIONS_DIR" "$FINAL_OUTCOME_POLICY" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
task_dir = Path(sys.argv[2])
campaign_results_dir = Path(sys.argv[3])
global_memory_dir = Path(sys.argv[4])
source_results_dir = Path(sys.argv[5])
output_dir = Path(sys.argv[6])
paper_min_refs = int(sys.argv[7])
paper_max_refs = int(sys.argv[8])
paper_profile = sys.argv[9]
query_plan_path = Path(sys.argv[10])
final_evaluation_dir = Path(sys.argv[11]) if sys.argv[11] else None
final_evaluation_mode = sys.argv[12]
contract_task_dir = Path(sys.argv[13]) if sys.argv[13] else None
investigation_results_dir = Path(sys.argv[14]) if sys.argv[14] else None
ablations_dir = Path(sys.argv[15]) if sys.argv[15] else None
final_outcome_policy = sys.argv[16]

sys.path.insert(0, str(script_dir))

from src.orchestrator.paper_data import FinalEvaluationMode, build_paper_data  # noqa: E402
from src.orchestrator.reference_query_plan import (  # noqa: E402
    load_reference_query_plan,
    reference_query_plan_metadata,
    reference_query_texts,
    validate_reference_query_plan_payload,
)
from src.orchestrator.semantic_scholar import (  # noqa: E402
    SemanticScholarError,
    extract_bibtex_key,
    is_builtin_reference_id,
    reference_outputs_exist,
    retrieve_bibtex_entries,
    semantic_scholar_key_forbidden_error,
    write_bibtex_file,
)

paper_data = build_paper_data(
    source_results_dir,
    paper_profile=paper_profile,
    ablations_dir=ablations_dir,
    final_evaluation_dir=final_evaluation_dir,
    task_dir=contract_task_dir,
    investigation_results_dir=investigation_results_dir,
    final_evaluation_mode=FinalEvaluationMode(final_evaluation_mode),
    final_outcome_policy=final_outcome_policy,
)
plan_payload = load_reference_query_plan(query_plan_path)
validate_reference_query_plan_payload(plan_payload)
queries = reference_query_texts(plan_payload)
query_plan_metadata = reference_query_plan_metadata(plan_payload)
print(
    f"[paper.sh] reference query plan ready: {len(queries)} queries",
    file=sys.stderr,
    flush=True,
)

s2_forbidden_error = semantic_scholar_key_forbidden_error(
    queries[0],
    timeout_seconds=20,
)
if s2_forbidden_error is not None:
    print(
        f"[paper.sh] Semantic Scholar key forbidden during health check; falling back to OpenAlex: {s2_forbidden_error}",
        file=sys.stderr,
        flush=True,
    )
    references = []
    provenance = []
    s2_error = s2_forbidden_error
else:
    try:
        references, provenance = retrieve_bibtex_entries(
            queries=queries,
            min_refs=paper_min_refs,
            max_refs=paper_max_refs,
        )
    except SemanticScholarError as exc:
        references = []
        provenance = []
        s2_error = str(exc)
    else:
        s2_error = None

references_bib_path = output_dir / "references.bib"
references_provenance_path = output_dir / "references_provenance.json"
openalex_bib_path = output_dir / "references.openalex.bib"
openalex_provenance_path = output_dir / "references.openalex.json"
scholarly_bib_path = output_dir / "references.scholarly.bib"
scholarly_provenance_path = output_dir / "references.scholarly.json"

def _references_are_builtin_only(rows: list[dict[str, object]]) -> bool:
    if not rows:
        return False
    return all(is_builtin_reference_id(row.get("id")) for row in rows)

def _reference_provenance_payload(
    refs: list[dict[str, object]],
    *,
    generated_at: object | None = None,
) -> dict[str, object]:
    return {
        "generated_at": generated_at if generated_at is not None else paper_data.get("generated_at"),
        "results_dir": str(source_results_dir.resolve()),
        "campaign_results_dir": str(campaign_results_dir.resolve()),
        "queries": queries,
        "reference_query_plan": query_plan_metadata,
        "count": len(refs),
        "references": refs,
    }

def _write_reference_outputs(
    refs: list[dict[str, object]],
    prov: list[dict[str, object]],
) -> None:
    write_bibtex_file(refs, references_bib_path)
    references_provenance_path.write_text(
        json.dumps(_reference_provenance_payload(prov), indent=2, sort_keys=True),
        encoding="utf-8",
    )

def _load_bibtex_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    entries: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("@") and current:
            entries.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append("\n".join(current).strip())
    rows: list[dict[str, object]] = []
    for entry in entries:
        key = extract_bibtex_key(entry)
        if key is None:
            continue
        rows.append({"citation_key": key, "bibtex": entry})
    return rows

def _load_provenance_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    rows = payload.get("references")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]

def _merge_reference_sets(*sources: tuple[list[dict[str, object]], list[dict[str, object]]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    merged_refs: list[dict[str, object]] = []
    merged_prov: list[dict[str, object]] = []
    seen_keys: set[str] = set()
    for bib_rows, prov_rows in sources:
        prov_by_key: dict[str, dict[str, object]] = {}
        for row in prov_rows:
            key = str(row.get("citation_key", "")).strip()
            if key:
                prov_by_key[key] = row
        for row in bib_rows:
            key = str(row.get("citation_key", "")).strip()
            bibtex = row.get("bibtex")
            if not key or key in seen_keys or not isinstance(bibtex, str) or not bibtex.strip():
                continue
            prov_row = dict(prov_by_key.get(key, {}))
            prov_row["citation_key"] = key
            prov_row["bibtex"] = bibtex.strip()
            merged_refs.append(prov_row)
            merged_prov.append({k: v for k, v in prov_row.items() if k != "bibtex"})
            seen_keys.add(key)
    return merged_refs, merged_prov

need_scholarly_fallback = (
    s2_error is not None
    or len(references) < paper_min_refs
    or _references_are_builtin_only(references)
)
openalex_error: str | None = None
if need_scholarly_fallback:
    if s2_error is not None:
        reason = s2_error
    elif len(references) < paper_min_refs:
        reason = (
            f"Semantic Scholar returned {len(references)} references"
            f" < required {paper_min_refs}"
        )
    else:
        reason = "Semantic Scholar returned builtin-only references"
    print(
        f"[paper.sh] attempting OpenAlex fallback: {reason}",
        file=sys.stderr,
        flush=True,
    )
    openalex_script = (
        script_dir
        / "src"
        / "agents"
        / "skills"
        / "retrieve-references"
        / "scripts"
        / "openalex_fetch.py"
    )
    if not openalex_script.is_file():
        openalex_error = f"Missing OpenAlex fallback script: {openalex_script}"
    else:
        openalex_search_limit = 10
        openalex_cmd = [
            sys.executable,
            str(openalex_script),
            "--min-refs",
            str(paper_min_refs),
            "--max-refs",
            str(paper_max_refs),
            "--search-limit",
            str(openalex_search_limit),
            "--allow-partial-output",
            "--out-bib",
            str(openalex_bib_path),
            "--out-json",
            str(openalex_provenance_path),
        ]
        for query in queries:
            openalex_cmd.extend(["--query", query])
        try:
            openalex_run = subprocess.run(
                openalex_cmd,
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=240,
                check=True,
            )
        except subprocess.TimeoutExpired:
            openalex_error = "OpenAlex fallback timed out after 240s"
        except subprocess.CalledProcessError as exc:
            stderr_tail = "\n".join((exc.stderr or "").splitlines()[-20:]).strip()
            stdout_tail = "\n".join((exc.stdout or "").splitlines()[-20:]).strip()
            openalex_error = stderr_tail or stdout_tail or f"OpenAlex fallback failed with code {exc.returncode}"
        else:
            if openalex_run.stderr.strip():
                print(openalex_run.stderr.strip(), file=sys.stderr, flush=True)
            if openalex_run.stdout.strip():
                print(openalex_run.stdout.strip(), file=sys.stderr, flush=True)
        if openalex_bib_path.is_file() and openalex_provenance_path.is_file():
            openalex_refs, openalex_prov = _merge_reference_sets(
                (_load_bibtex_rows(openalex_bib_path), _load_provenance_rows(openalex_provenance_path)),
            )
            if openalex_refs:
                print(
                    f"[paper.sh] OpenAlex contributed {len(openalex_refs)} references",
                    file=sys.stderr,
                    flush=True,
                )
                if len(openalex_refs) >= paper_min_refs:
                    _write_reference_outputs(
                        openalex_refs,
                        openalex_prov,
                    )

need_scholarly_fallback = not reference_outputs_exist(output_dir)
scholarly_error: str | None = None
if need_scholarly_fallback:
    if openalex_error is not None:
        reason = openalex_error
    elif s2_error is not None:
        reason = s2_error
    elif len(references) < paper_min_refs and not reference_outputs_exist(output_dir):
        reason = (
            f"OpenAlex returned no usable references after Semantic Scholar returned {len(references)}"
        )
    else:
        reason = "OpenAlex fallback did not produce references"
    print(
        f"[paper.sh] attempting scholarly fallback: {reason}",
        file=sys.stderr,
        flush=True,
    )
    scholarly_script = (
        script_dir
        / "src"
        / "agents"
        / "skills"
        / "retrieve-references"
        / "scripts"
        / "scholarly_fetch.py"
    )
    if not scholarly_script.is_file():
        scholarly_error = f"Missing scholarly fallback script: {scholarly_script}"
    else:
        scholarly_search_limit = 10
        scholarly_total_timeout_seconds = 300
        scholarly_process_timeout_seconds = scholarly_total_timeout_seconds + 60
        scholarly_progress_interval_seconds = 10
        openalex_existing_refs = _load_bibtex_rows(openalex_bib_path)
        openalex_existing_prov = _load_provenance_rows(openalex_provenance_path)
        scholarly_min_refs = max(1, paper_min_refs - len(openalex_existing_refs))
        scholarly_max_refs = max(1, paper_max_refs - len(openalex_existing_refs))
        scholarly_cmd = [
            sys.executable,
            str(scholarly_script),
            "--min-refs",
            str(scholarly_min_refs),
            "--max-refs",
            str(scholarly_max_refs),
            "--search-limit",
            str(scholarly_search_limit),
            "--total-timeout-seconds",
            str(scholarly_total_timeout_seconds),
            "--progress-interval-seconds",
            str(scholarly_progress_interval_seconds),
            "--allow-partial-output",
            "--out-bib",
            str(scholarly_bib_path),
            "--out-json",
            str(scholarly_provenance_path),
        ]
        for query in queries:
            scholarly_cmd.extend(["--query", query])
        try:
            scholarly_run = subprocess.run(
                scholarly_cmd,
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=scholarly_process_timeout_seconds,
                check=True,
            )
        except subprocess.TimeoutExpired:
            scholarly_error = (
                f"scholarly fallback timed out after {scholarly_process_timeout_seconds}s"
            )
        except subprocess.CalledProcessError as exc:
            stderr_tail = "\n".join((exc.stderr or "").splitlines()[-20:]).strip()
            stdout_tail = "\n".join((exc.stdout or "").splitlines()[-20:]).strip()
            scholarly_error = stderr_tail or stdout_tail or f"scholarly fallback failed with code {exc.returncode}"
        else:
            if scholarly_run.stderr.strip():
                print(scholarly_run.stderr.strip(), file=sys.stderr, flush=True)
            if scholarly_run.stdout.strip():
                print(scholarly_run.stdout.strip(), file=sys.stderr, flush=True)
        merged_refs, merged_prov = _merge_reference_sets(
            (openalex_existing_refs, openalex_existing_prov),
            (_load_bibtex_rows(scholarly_bib_path), _load_provenance_rows(scholarly_provenance_path)),
        )
        if merged_refs:
            print(
                f"[paper.sh] merged fallback references: {len(merged_refs)} total",
                file=sys.stderr,
                flush=True,
            )
            if len(merged_refs) >= paper_min_refs:
                _write_reference_outputs(merged_refs, merged_prov)

if not reference_outputs_exist(output_dir):
    if references and len(references) >= paper_min_refs:
        print(
            "[paper.sh] downstream fallbacks unavailable; using Semantic Scholar references",
            file=sys.stderr,
            flush=True,
        )
        _write_reference_outputs(references, provenance)
    else:
        if scholarly_error and openalex_error and s2_error:
            raise SystemExit(
                f"Reference retrieval failed. Semantic Scholar: {s2_error}; OpenAlex: {openalex_error}; scholarly: {scholarly_error}"
            )
        if scholarly_error and s2_error:
            raise SystemExit(
                f"Reference retrieval failed. Semantic Scholar: {s2_error}; scholarly: {scholarly_error}"
            )
        if scholarly_error and openalex_error:
            raise SystemExit(
                f"Reference retrieval failed. OpenAlex: {openalex_error}; scholarly: {scholarly_error}"
            )
        if scholarly_error:
            raise SystemExit(f"Reference retrieval failed via scholarly fallback: {scholarly_error}")
        if openalex_error:
            raise SystemExit(f"Reference retrieval failed via OpenAlex fallback: {openalex_error}")
        if s2_error:
            raise SystemExit(s2_error)
        raise SystemExit("Reference retrieval failed without producing references.bib")
PY
}

run_reference_retrieval_with_log() {
  : > "$REFERENCE_RETRIEVAL_LOG_FILE"
  printf '[paper.sh] reference providers: semantic_scholar=%s openalex=%s\n' \
    "$([ -n "${S2_API_KEY:-}" ] && printf keyed || printf anonymous)" \
    "$([ -n "${OPENALEX_EMAIL:-}" ] && printf polite-pool || printf anonymous)" \
    | tee -a "$REFERENCE_RETRIEVAL_LOG_FILE"
  set +e
  retrieve_references_from_plan "$OUTPUT_DIR/reference_query_plan.json" 2>&1 | tee "$REFERENCE_RETRIEVAL_LOG_FILE"
  local stage_rc=${PIPESTATUS[0]}
  set -e
  return "$stage_rc"
}

draft_reference_gaps_require_enrichment() {
  "$PYTHON_BIN" - "$OUTPUT_DIR/draft_reference_gaps.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if str(payload.get("status", "")).strip() == "ready" else 1)
PY
}

run_postdraft_reference_retrieval_with_log() {
  local base_bib_path="$OUTPUT_DIR/references.pre_postdraft.bib"
  local base_provenance_path="$OUTPUT_DIR/references.pre_postdraft.json"
  local incremental_bib_path="$OUTPUT_DIR/postdraft_references.bib"
  local incremental_provenance_path="$OUTPUT_DIR/postdraft_references_provenance.json"

  cp "$OUTPUT_DIR/references.bib" "$base_bib_path"
  cp "$OUTPUT_DIR/references_provenance.json" "$base_provenance_path"

  : > "$POSTDRAFT_REFERENCE_RETRIEVAL_LOG_FILE"
  set +e
  retrieve_references_from_plan "$OUTPUT_DIR/postdraft_reference_query_plan.json" 2>&1 | tee "$POSTDRAFT_REFERENCE_RETRIEVAL_LOG_FILE"
  local stage_rc=${PIPESTATUS[0]}
  set -e
  if [ "$stage_rc" -ne 0 ]; then
    cp "$base_bib_path" "$OUTPUT_DIR/references.bib"
    cp "$base_provenance_path" "$OUTPUT_DIR/references_provenance.json"
    return "$stage_rc"
  fi

  mv "$OUTPUT_DIR/references.bib" "$incremental_bib_path"
  mv "$OUTPUT_DIR/references_provenance.json" "$incremental_provenance_path"
  "$PYTHON_BIN" - "$REPO_ROOT" "$base_bib_path" "$base_provenance_path" "$incremental_bib_path" "$incremental_provenance_path" "$OUTPUT_DIR/references.bib" "$OUTPUT_DIR/references_provenance.json" "$SOURCE_RESULTS_DIR" "$RESULTS_DIR" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
existing_bib_path = Path(sys.argv[2])
existing_provenance_path = Path(sys.argv[3])
incremental_bib_path = Path(sys.argv[4])
incremental_provenance_path = Path(sys.argv[5])
output_bib_path = Path(sys.argv[6])
output_provenance_path = Path(sys.argv[7])
results_dir = Path(sys.argv[8])
campaign_results_dir = Path(sys.argv[9])

sys.path.insert(0, str(script_dir))
from src.orchestrator.paper_reference_enrichment import merge_reference_artifacts  # noqa: E402

payload = merge_reference_artifacts(
    existing_bib_path=existing_bib_path,
    existing_provenance_path=existing_provenance_path,
    incremental_bib_path=incremental_bib_path,
    incremental_provenance_path=incremental_provenance_path,
    output_bib_path=output_bib_path,
    output_provenance_path=output_provenance_path,
    results_dir=results_dir,
    campaign_results_dir=campaign_results_dir,
)
print(f"[paper.sh] merged postdraft references: {payload['count']} total", file=sys.stderr, flush=True)
PY
}

run_postdraft_reference_enrichment() {
  local draft_reference_gap_repair_round=0
  while true; do
    if [ "$draft_reference_gap_repair_round" -eq 0 ]; then
      DRAFT_REFERENCE_GAPS_PROMPT="$(render_prompt "$PAPER_REFERENCE_GAPS_INSTRUCT_PATH" "$CURRENT_PAPER_INSTRUCT_PATH")"
      DRAFT_REFERENCE_GAPS_PROMPT_PATH="$OUTPUT_DIR/draft_reference_gaps_prompt.md"
      CURRENT_DRAFT_REFERENCE_GAPS_LOG_PATH="$DRAFT_REFERENCE_GAPS_LOG_FILE"
      CURRENT_DRAFT_REFERENCE_GAPS_STAGE_NAME="draft_reference_gaps"
      echo "Draft reference gaps prompt file: $DRAFT_REFERENCE_GAPS_PROMPT_PATH"
      echo "Draft reference gaps log: $CURRENT_DRAFT_REFERENCE_GAPS_LOG_PATH"
    else
      DRAFT_REFERENCE_GAPS_PROMPT="$(render_prompt "$PAPER_REFERENCE_GAPS_INSTRUCT_PATH" "$CURRENT_PAPER_INSTRUCT_PATH")"
      DRAFT_REFERENCE_GAPS_PROMPT_PATH="$OUTPUT_DIR/draft_reference_gaps_repair_prompt_round_${draft_reference_gap_repair_round}.md"
      CURRENT_DRAFT_REFERENCE_GAPS_LOG_PATH="$OUTPUT_DIR/draft_reference_gaps_repair_agent_${AGENT}_${TIMESTAMP}_round_${draft_reference_gap_repair_round}.log"
      CURRENT_DRAFT_REFERENCE_GAPS_STAGE_NAME="draft_reference_gaps_repair_round_${draft_reference_gap_repair_round}"
      echo "[paper.sh] starting draft reference gap repair round $draft_reference_gap_repair_round / $PAPER_MAX_REPAIR_ROUNDS"
      echo "Draft reference gaps repair prompt file: $DRAFT_REFERENCE_GAPS_PROMPT_PATH"
      echo "Draft reference gaps repair log: $CURRENT_DRAFT_REFERENCE_GAPS_LOG_PATH"
    fi

    if run_prompt_stage_with_feedback \
      "$CURRENT_DRAFT_REFERENCE_GAPS_STAGE_NAME" \
      "$DRAFT_REFERENCE_GAPS_PROMPT" \
      "$DRAFT_REFERENCE_GAPS_PROMPT_PATH" \
      "$CURRENT_DRAFT_REFERENCE_GAPS_LOG_PATH" \
      "$DRAFT_REFERENCE_GAPS_VALIDATION_FEEDBACK_JSON_PATH" \
      "$DRAFT_REFERENCE_GAPS_VALIDATION_FEEDBACK_MD_PATH" \
      "draft_reference_gaps.md" \
      "draft_reference_gaps.json"; then
      :
    else
      stage_rc=$?
      if [ "$stage_rc" -eq 130 ]; then
        return 130
      fi
      if [ "$draft_reference_gap_repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
        echo "[paper.sh] exhausted draft reference gap repair rounds after $draft_reference_gap_repair_round attempts; see $DRAFT_REFERENCE_GAPS_VALIDATION_FEEDBACK_JSON_PATH" >&2
        return 1
      fi
      draft_reference_gap_repair_round=$((draft_reference_gap_repair_round + 1))
      continue
    fi

    if validate_draft_reference_gaps_with_feedback; then
      break
    fi
    if [ "$draft_reference_gap_repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
      echo "[paper.sh] exhausted draft reference gap repair rounds after $draft_reference_gap_repair_round attempts; see $DRAFT_REFERENCE_GAPS_VALIDATION_FEEDBACK_JSON_PATH" >&2
      return 1
    fi
    draft_reference_gap_repair_round=$((draft_reference_gap_repair_round + 1))
  done

  if ! draft_reference_gaps_require_enrichment; then
    echo "[paper.sh] draft gap analysis found no targeted citation gaps; skipping postdraft retrieval"
    touch "$POSTDRAFT_REFERENCE_ENRICHMENT_MARKER_PATH"
    return 0
  fi

  local postdraft_reference_query_plan_repair_round=0
  while true; do
    if [ "$postdraft_reference_query_plan_repair_round" -eq 0 ]; then
      POSTDRAFT_REFERENCE_QUERY_PLAN_PROMPT="$(render_prompt "$POSTDRAFT_REFERENCE_QUERY_PLAN_INSTRUCT_PATH" "$CURRENT_PAPER_INSTRUCT_PATH")"
      POSTDRAFT_REFERENCE_QUERY_PLAN_PROMPT_PATH="$OUTPUT_DIR/postdraft_reference_query_plan_prompt.md"
      CURRENT_POSTDRAFT_REFERENCE_QUERY_PLAN_LOG_PATH="$POSTDRAFT_REFERENCE_QUERY_PLAN_LOG_FILE"
      CURRENT_POSTDRAFT_REFERENCE_QUERY_PLAN_STAGE_NAME="postdraft_reference_query_plan"
      echo "Postdraft reference query plan prompt file: $POSTDRAFT_REFERENCE_QUERY_PLAN_PROMPT_PATH"
      echo "Postdraft reference query plan log: $CURRENT_POSTDRAFT_REFERENCE_QUERY_PLAN_LOG_PATH"
      echo "Postdraft reference retrieval log: $POSTDRAFT_REFERENCE_RETRIEVAL_LOG_FILE"
    else
      POSTDRAFT_REFERENCE_QUERY_PLAN_PROMPT="$(render_prompt "$POSTDRAFT_REFERENCE_QUERY_PLAN_INSTRUCT_PATH" "$CURRENT_PAPER_INSTRUCT_PATH")"
      POSTDRAFT_REFERENCE_QUERY_PLAN_PROMPT+="$(paper_diagnostics context postdraft_reference_query_plan)"
      POSTDRAFT_REFERENCE_QUERY_PLAN_PROMPT_PATH="$OUTPUT_DIR/postdraft_reference_query_plan_repair_prompt_round_${postdraft_reference_query_plan_repair_round}.md"
      CURRENT_POSTDRAFT_REFERENCE_QUERY_PLAN_LOG_PATH="$OUTPUT_DIR/postdraft_reference_query_plan_repair_agent_${AGENT}_${TIMESTAMP}_round_${postdraft_reference_query_plan_repair_round}.log"
      CURRENT_POSTDRAFT_REFERENCE_QUERY_PLAN_STAGE_NAME="postdraft_reference_query_plan_repair_round_${postdraft_reference_query_plan_repair_round}"
      echo "[paper.sh] starting postdraft reference query plan repair round $postdraft_reference_query_plan_repair_round / $PAPER_MAX_REPAIR_ROUNDS"
      echo "Postdraft reference query plan repair prompt file: $POSTDRAFT_REFERENCE_QUERY_PLAN_PROMPT_PATH"
      echo "Postdraft reference query plan repair log: $CURRENT_POSTDRAFT_REFERENCE_QUERY_PLAN_LOG_PATH"
    fi

    if run_prompt_stage_with_feedback \
      "$CURRENT_POSTDRAFT_REFERENCE_QUERY_PLAN_STAGE_NAME" \
      "$POSTDRAFT_REFERENCE_QUERY_PLAN_PROMPT" \
      "$POSTDRAFT_REFERENCE_QUERY_PLAN_PROMPT_PATH" \
      "$CURRENT_POSTDRAFT_REFERENCE_QUERY_PLAN_LOG_PATH" \
      "$POSTDRAFT_REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_JSON_PATH" \
      "$POSTDRAFT_REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_MD_PATH" \
      "postdraft_reference_query_plan.md" \
      "postdraft_reference_query_plan.json"; then
      :
    else
      stage_rc=$?
      if [ "$stage_rc" -eq 130 ]; then
        return 130
      fi
      paper_diagnostics reference postdraft_reference_query_plan "$postdraft_reference_query_plan_repair_round" || return 1
      if [ "$postdraft_reference_query_plan_repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
        echo "[paper.sh] exhausted postdraft reference query plan repair rounds after $postdraft_reference_query_plan_repair_round attempts; see $POSTDRAFT_REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_JSON_PATH" >&2
        return 1
      fi
      postdraft_reference_query_plan_repair_round=$((postdraft_reference_query_plan_repair_round + 1))
      continue
    fi

    if validate_postdraft_reference_query_plan_with_feedback; then
      paper_diagnostics reference postdraft_reference_query_plan "$postdraft_reference_query_plan_repair_round" || return 1
      break
    fi
    paper_diagnostics reference postdraft_reference_query_plan "$postdraft_reference_query_plan_repair_round" || return 1
    if [ "$postdraft_reference_query_plan_repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
      echo "[paper.sh] exhausted postdraft reference query plan repair rounds after $postdraft_reference_query_plan_repair_round attempts; see $POSTDRAFT_REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_JSON_PATH" >&2
      return 1
    fi
    postdraft_reference_query_plan_repair_round=$((postdraft_reference_query_plan_repair_round + 1))
  done

  if run_postdraft_reference_retrieval_with_log; then
    :
  else
    retrieval_rc=$?
    if [ "$retrieval_rc" -eq 130 ]; then
      return 130
    fi
    echo "[paper.sh] postdraft reference retrieval failed; see $POSTDRAFT_REFERENCE_RETRIEVAL_LOG_FILE" >&2
    return "$retrieval_rc"
  fi

  POSTDRAFT_CITATION_WEAVE_PROMPT="$(render_prompt "$CURRENT_PAPER_INSTRUCT_PATH" "$CURRENT_PAPER_INSTRUCT_PATH")"
  POSTDRAFT_CITATION_WEAVE_PROMPT_PATH="$OUTPUT_DIR/postdraft_citation_weave_prompt.md"
  POSTDRAFT_CITATION_WEAVE_LOG_PATH="$OUTPUT_DIR/postdraft_citation_weave_agent_${AGENT}_${TIMESTAMP}.log"
  if run_prompt_stage_with_feedback \
    "postdraft_citation_weave" \
    "$POSTDRAFT_CITATION_WEAVE_PROMPT" \
    "$POSTDRAFT_CITATION_WEAVE_PROMPT_PATH" \
    "$POSTDRAFT_CITATION_WEAVE_LOG_PATH" \
    "$VALIDATION_FEEDBACK_JSON_PATH" \
    "$VALIDATION_FEEDBACK_MD_PATH" \
    "paper.tex"; then
    touch "$POSTDRAFT_REFERENCE_ENRICHMENT_MARKER_PATH"
    return 0
  fi
  return $?
}

validate_strategy_with_feedback() {
  "$PYTHON_BIN" - "$OUTPUT_DIR" "$STRATEGY_VALIDATION_FEEDBACK_JSON_PATH" "$STRATEGY_VALIDATION_FEEDBACK_MD_PATH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output_dir = Path(sys.argv[1])
feedback_json_path = Path(sys.argv[2])
feedback_md_path = Path(sys.argv[3])
strategy_md = output_dir / "paper_strategy.md"
strategy_json = output_dir / "paper_strategy.json"

feedback: dict[str, object] = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "paper_dir": str(output_dir.resolve()),
    "ok": False,
    "hard_errors": [],
}

try:
    if not strategy_md.is_file():
        raise SystemExit(f"Missing required paper strategy memo: {strategy_md}")
    if not strategy_json.is_file():
        raise SystemExit(f"Missing required paper strategy JSON: {strategy_json}")
    if not strategy_md.read_text(encoding="utf-8").strip():
        raise SystemExit(f"Paper strategy memo is empty: {strategy_md}")

    try:
        payload = json.loads(strategy_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {strategy_json}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {strategy_json}")

    status = str(payload.get("status", "")).strip()
    if status not in {"ready", "fail"}:
        raise SystemExit("paper_strategy.json must set status to 'ready' or 'fail'")
    selected_prompt_family = str(payload.get("selected_prompt_family", "")).strip()
    if not selected_prompt_family:
        raise SystemExit("paper_strategy.json missing selected_prompt_family")

    if status == "fail":
        failure_reason = str(payload.get("failure_reason", "")).strip()
        if not failure_reason:
            raise SystemExit("paper_strategy.json status=fail requires non-empty failure_reason")
        raise SystemExit(f"paper strategy failed: {failure_reason}")

    paper_data_path = output_dir / "paper_data.json"
    try:
        paper_data = json.loads(paper_data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid paper data in {paper_data_path}: {exc}") from exc
    if not isinstance(paper_data, dict):
        raise SystemExit(f"Expected JSON object in {paper_data_path}")
    grounding_sources = paper_data.get("grounding_sources")
    if not isinstance(grounding_sources, dict):
        raise SystemExit("paper_data.json grounding_sources must be an object")
    allowed_source_ids = set(grounding_sources)

    required_text_fields = ["recommended_title", "central_question", "central_thesis", "narrative_rationale"]
    for field in required_text_fields:
        if not str(payload.get(field, "")).strip():
            raise SystemExit(f"paper_strategy.json missing non-empty field: {field}")

    supplementary_plan = payload.get("supplementary_plan")
    if not isinstance(supplementary_plan, dict):
        raise SystemExit("paper_strategy.json must contain a supplementary_plan object")
    if not str(supplementary_plan.get("purpose", "")).strip():
        raise SystemExit("paper_strategy.json supplementary_plan.purpose must be non-empty")
    planned_subsections = supplementary_plan.get("planned_subsections")
    if not isinstance(planned_subsections, list) or not planned_subsections:
        raise SystemExit(
            "paper_strategy.json supplementary_plan.planned_subsections must be a non-empty list"
        )
    for idx, row in enumerate(planned_subsections):
        if not isinstance(row, dict):
            raise SystemExit(
                f"paper_strategy.json supplementary_plan.planned_subsections[{idx}] must be an object"
            )
        for key in ("heading", "grounded_content"):
            if not str(row.get(key, "")).strip():
                raise SystemExit(
                    f"paper_strategy.json supplementary_plan.planned_subsections[{idx}].{key} must be non-empty"
                )
        source_ids = row.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise SystemExit(
                f"paper_strategy.json supplementary_plan.planned_subsections[{idx}].source_ids must be a non-empty list"
            )
        if not all(isinstance(item, str) and item.strip() for item in source_ids):
            raise SystemExit(
                f"paper_strategy.json supplementary_plan.planned_subsections[{idx}].source_ids entries must be non-empty strings"
            )
        unknown_source_ids = sorted(set(source_ids) - allowed_source_ids)
        if unknown_source_ids:
            raise SystemExit(
                "paper_strategy.json "
                f"supplementary_plan.planned_subsections[{idx}].source_ids "
                f"contains unknown grounding source ids: {unknown_source_ids}"
            )
    method_groups = payload.get("method_groups")
    if not isinstance(method_groups, list) or not method_groups:
        raise SystemExit("paper_strategy.json must contain non-empty method_groups list")
    for idx, row in enumerate(method_groups):
        if not isinstance(row, dict):
            raise SystemExit(f"paper_strategy.json method_groups[{idx}] must be an object")
        for key in ("name", "role", "include_reason"):
            if not str(row.get(key, "")).strip():
                raise SystemExit(f"paper_strategy.json method_groups[{idx}] missing non-empty field: {key}")

    for list_field in (
        "results_focus",
        "figure_plan",
        "figure_specs",
        "forbidden_headings",
        "forbidden_phrases",
        "notes_for_writer",
    ):
        value = payload.get(list_field)
        if not isinstance(value, list):
            raise SystemExit(f"paper_strategy.json field must be a list: {list_field}")

    figure_specs = payload.get("figure_specs", [])
    for idx, row in enumerate(figure_specs):
        if not isinstance(row, dict):
            raise SystemExit(f"paper_strategy.json figure_specs[{idx}] must be an object")
        preferred_generator = str(row.get("preferred_generator", "")).strip()
        for key in ("figure_id", "figure_type"):
            if not str(row.get(key, "")).strip():
                raise SystemExit(f"paper_strategy.json figure_specs[{idx}] missing non-empty field: {key}")
        if preferred_generator and preferred_generator not in {
            "generate-concept-figure",
            "matplotlib",
            "tikz",
            "native_latex",
            "manual",
        }:
            raise SystemExit(
                f"paper_strategy.json figure_specs[{idx}].preferred_generator has unsupported value: {preferred_generator}"
            )
        if (
            bool(row.get("required", False))
            and str(row.get("figure_type", "")).strip() == "diagram"
            and not preferred_generator
        ):
            raise SystemExit(
                f"paper_strategy.json figure_specs[{idx}] missing preferred_generator for required diagram"
            )
        for key in ("max_panels", "max_categories_per_panel", "max_series_per_panel"):
            value = row.get(key)
            if not isinstance(value, int) or value <= 0:
                raise SystemExit(f"paper_strategy.json figure_specs[{idx}].{key} must be positive int")
        allow_progression = row.get("allow_progression")
        if not isinstance(allow_progression, bool):
            raise SystemExit(
                f"paper_strategy.json figure_specs[{idx}].allow_progression must be bool"
            )
        required = row.get("required")
        if required is not None and not isinstance(required, bool):
            raise SystemExit(f"paper_strategy.json figure_specs[{idx}].required must be bool when provided")
except SystemExit as exc:
    feedback["hard_errors"] = [str(exc)]
else:
    feedback["ok"] = True

feedback_json_path.write_text(json.dumps(feedback, indent=2, sort_keys=True), encoding="utf-8")
lines = [
    "# Paper strategy validation feedback",
    "",
    f"- generated_at: {feedback['generated_at']}",
    f"- paper_dir: `{feedback['paper_dir']}`",
    f"- ok: `{feedback['ok']}`",
    "",
]
hard_errors = feedback.get("hard_errors", [])
if isinstance(hard_errors, list) and hard_errors:
    lines.append("## Hard errors")
    lines.extend(f"- {message}" for message in hard_errors)
    lines.append("")
feedback_md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

if isinstance(hard_errors, list) and hard_errors:
    print("[paper.sh] strategy validation errors:")
    for message in hard_errors:
        print(f"  - {message}")
    raise SystemExit(1)
print(f"[paper.sh] paper strategy ok: {strategy_json}")
PY
}

paper_gate_start "references" "phase=initial"
reference_query_plan_repair_round=0
while true; do
  if [ "$reference_query_plan_repair_round" -eq 0 ]; then
    REFERENCE_QUERY_PLAN_PROMPT="$(render_prompt "$REFERENCE_QUERY_PLAN_INSTRUCT_PATH" "$INSTRUCT_PATH")"
    REFERENCE_QUERY_PLAN_PROMPT_PATH="$OUTPUT_DIR/reference_query_plan_prompt.md"
    CURRENT_REFERENCE_QUERY_PLAN_LOG_PATH="$REFERENCE_QUERY_PLAN_LOG_FILE"
    CURRENT_REFERENCE_QUERY_PLAN_STAGE_NAME="reference_query_plan"
    echo "Reference query plan prompt file: $REFERENCE_QUERY_PLAN_PROMPT_PATH"
    echo "Reference query plan log: $CURRENT_REFERENCE_QUERY_PLAN_LOG_PATH"
    echo "Reference retrieval log: $REFERENCE_RETRIEVAL_LOG_FILE"
  else
    REFERENCE_QUERY_PLAN_PROMPT="$(render_prompt "$REFERENCE_QUERY_PLAN_INSTRUCT_PATH" "$INSTRUCT_PATH")"
    REFERENCE_QUERY_PLAN_PROMPT+="$(paper_diagnostics context reference_query_plan)"
    REFERENCE_QUERY_PLAN_PROMPT_PATH="$OUTPUT_DIR/reference_query_plan_repair_prompt_round_${reference_query_plan_repair_round}.md"
    CURRENT_REFERENCE_QUERY_PLAN_LOG_PATH="$OUTPUT_DIR/reference_query_plan_repair_agent_${AGENT}_${TIMESTAMP}_round_${reference_query_plan_repair_round}.log"
    CURRENT_REFERENCE_QUERY_PLAN_STAGE_NAME="reference_query_plan_repair_round_${reference_query_plan_repair_round}"
    echo "[paper.sh] starting reference query plan repair round $reference_query_plan_repair_round / $PAPER_MAX_REPAIR_ROUNDS"
    echo "Reference query plan repair prompt file: $REFERENCE_QUERY_PLAN_PROMPT_PATH"
    echo "Reference query plan repair log: $CURRENT_REFERENCE_QUERY_PLAN_LOG_PATH"
  fi

  if run_prompt_stage_with_feedback \
    "$CURRENT_REFERENCE_QUERY_PLAN_STAGE_NAME" \
    "$REFERENCE_QUERY_PLAN_PROMPT" \
    "$REFERENCE_QUERY_PLAN_PROMPT_PATH" \
    "$CURRENT_REFERENCE_QUERY_PLAN_LOG_PATH" \
    "$REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_JSON_PATH" \
    "$REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_MD_PATH" \
    "reference_query_plan.md" \
    "reference_query_plan.json"; then
    :
  else
    stage_rc=$?
    if [ "$stage_rc" -eq 130 ]; then
      exit 130
    fi
    paper_diagnostics reference reference_query_plan "$reference_query_plan_repair_round" || exit 1
    if [ "$reference_query_plan_repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
      echo "[paper.sh] exhausted reference query plan repair rounds after $reference_query_plan_repair_round attempts; see $REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_JSON_PATH" >&2
      exit 1
    fi
    reference_query_plan_repair_round=$((reference_query_plan_repair_round + 1))
    continue
  fi

  if validate_reference_query_plan_with_feedback; then
    paper_diagnostics reference reference_query_plan "$reference_query_plan_repair_round" || exit 1
    break
  fi
  paper_diagnostics reference reference_query_plan "$reference_query_plan_repair_round" || exit 1
  if [ "$reference_query_plan_repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
    echo "[paper.sh] exhausted reference query plan repair rounds after $reference_query_plan_repair_round attempts; see $REFERENCE_QUERY_PLAN_VALIDATION_FEEDBACK_JSON_PATH" >&2
    exit 1
  fi
  reference_query_plan_repair_round=$((reference_query_plan_repair_round + 1))
done

if run_reference_retrieval_with_log; then
  paper_gate_complete "references" "phase=initial" "repair_round=$reference_query_plan_repair_round"
else
  retrieval_rc=$?
  paper_gate_failed "references" "phase=initial" "exit_code=$retrieval_rc"
  if [ "$retrieval_rc" -eq 130 ]; then
    exit 130
  fi
  echo "[paper.sh] reference retrieval failed; see $REFERENCE_RETRIEVAL_LOG_FILE" >&2
  exit "$retrieval_rc"
fi

if [ "$TWO_STAGE" -eq 1 ]; then
  paper_gate_start "strategy"
  strategy_repair_round=0
  while true; do
    if [ "$strategy_repair_round" -eq 0 ]; then
      STRATEGY_PROMPT="$(render_prompt "$PAPER_STRATEGY_INSTRUCT_PATH" "$INSTRUCT_PATH")"
      STRATEGY_PROMPT_PATH="$OUTPUT_DIR/paper_strategy_prompt.md"
      CURRENT_STRATEGY_LOG_PATH="$STRATEGY_LOG_FILE"
      CURRENT_STRATEGY_STAGE_NAME="paper_strategy"
      echo "Strategy prompt file: $STRATEGY_PROMPT_PATH"
      echo "Strategy log: $CURRENT_STRATEGY_LOG_PATH"
    else
      STRATEGY_PROMPT="$(render_prompt "$PAPER_STRATEGY_INSTRUCT_PATH" "$INSTRUCT_PATH")"
      STRATEGY_PROMPT_PATH="$OUTPUT_DIR/paper_strategy_repair_prompt_round_${strategy_repair_round}.md"
      CURRENT_STRATEGY_LOG_PATH="$OUTPUT_DIR/paper_strategy_repair_agent_${AGENT}_${TIMESTAMP}_round_${strategy_repair_round}.log"
      CURRENT_STRATEGY_STAGE_NAME="paper_strategy_repair_round_${strategy_repair_round}"
      echo "[paper.sh] starting strategy repair round $strategy_repair_round / $PAPER_MAX_REPAIR_ROUNDS"
      echo "Strategy repair prompt file: $STRATEGY_PROMPT_PATH"
      echo "Strategy repair log: $CURRENT_STRATEGY_LOG_PATH"
    fi

    if run_prompt_stage_with_feedback \
      "$CURRENT_STRATEGY_STAGE_NAME" \
      "$STRATEGY_PROMPT" \
      "$STRATEGY_PROMPT_PATH" \
      "$CURRENT_STRATEGY_LOG_PATH" \
      "$STRATEGY_VALIDATION_FEEDBACK_JSON_PATH" \
      "$STRATEGY_VALIDATION_FEEDBACK_MD_PATH" \
      "paper_strategy.md" \
      "paper_strategy.json"; then
      :
    else
      stage_rc=$?
      if [ "$stage_rc" -eq 130 ]; then
        exit 130
      fi
      if [ "$strategy_repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
        echo "[paper.sh] exhausted strategy repair rounds after $strategy_repair_round attempts; see $STRATEGY_VALIDATION_FEEDBACK_JSON_PATH" >&2
        exit 1
      fi
      strategy_repair_round=$((strategy_repair_round + 1))
      continue
    fi

    if validate_strategy_with_feedback; then
      break
    fi
    if [ "$strategy_repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
      echo "[paper.sh] exhausted strategy repair rounds after $strategy_repair_round attempts; see $STRATEGY_VALIDATION_FEEDBACK_JSON_PATH" >&2
      exit 1
    fi
    strategy_repair_round=$((strategy_repair_round + 1))
  done
  resolve_final_paper_stage_from_strategy
  paper_gate_complete "strategy" "repair_round=$strategy_repair_round"
else
  paper_gate_start "strategy"
  paper_gate_finish "strategy" "skipped" "reason=single_stage"
fi

# ── Posthoc analysis ranking (optional) ─────────────────────────────────────
POSTHOC_RANKING_INSTRUCT_PATH="$REPO_ROOT/src/prompts/paper_posthoc_ranking.md"
POSTHOC_RANKING_LOG_FILE="$OUTPUT_DIR/posthoc_ranking_agent_${AGENT}_${TIMESTAMP}.log"
POSTHOC_RANKING_VALIDATION_FEEDBACK_JSON_PATH="$OUTPUT_DIR/posthoc_ranking_validation_feedback.json"
POSTHOC_RANKING_VALIDATION_FEEDBACK_MD_PATH="$OUTPUT_DIR/posthoc_ranking_validation_feedback.md"
HAS_POSTHOC_ANALYSES=0

validate_posthoc_ranking_with_feedback() {
  local validation_output=""
  local validation_rc=0
  set +e
  validation_output="$("$PYTHON_BIN" - "$REPO_ROOT" "$RESULTS_DIR" "$OUTPUT_DIR" "$POSTHOC_DIR" <<'PY' 2>&1
import json
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
posthoc_dir = Path(sys.argv[4]) if sys.argv[4] else None
sys.path.insert(0, str(script_dir))

from src.orchestrator.paper_data_posthoc import (  # noqa: E402
    collect_posthoc_analyses,
    validate_posthoc_ranking,
)

try:
    ranking = json.loads(
        (output_dir / "posthoc_ranking.json").read_text(encoding="utf-8")
    )
    if not isinstance(ranking, dict):
        raise ValueError("posthoc_ranking.json must contain a JSON object")
    analyses = collect_posthoc_analyses(results_dir, posthoc_dir=posthoc_dir)
    validate_posthoc_ranking(ranking, analyses)
except (OSError, json.JSONDecodeError, ValueError) as exc:
    raise SystemExit(str(exc)) from exc
PY
)"
  validation_rc=$?
  set -e
  if [ "$validation_rc" -eq 0 ]; then
    rm -f \
      "$POSTHOC_RANKING_VALIDATION_FEEDBACK_JSON_PATH" \
      "$POSTHOC_RANKING_VALIDATION_FEEDBACK_MD_PATH"
    return 0
  fi
  write_stage_contract_feedback \
    "$CURRENT_POSTHOC_RANKING_STAGE_NAME" \
    "$POSTHOC_RANKING_VALIDATION_FEEDBACK_JSON_PATH" \
    "$POSTHOC_RANKING_VALIDATION_FEEDBACK_MD_PATH" \
    "$CURRENT_POSTHOC_RANKING_PROMPT_PATH" \
    "$CURRENT_POSTHOC_RANKING_LOG_PATH" \
    "invalid posthoc ranking artifact: ${validation_output:-validation failed}" \
    "posthoc_ranking.json"
  echo "[paper.sh] invalid posthoc ranking artifact: ${validation_output:-validation failed}" >&2
  return "$validation_rc"
}

if [ "$TWO_STAGE" -eq 1 ] || [ -n "$POSTHOC_DIR" ]; then
  HAS_POSTHOC_ANALYSES="$("$PYTHON_BIN" - "$REPO_ROOT" "$RESULTS_DIR" "$POSTHOC_DIR" <<'PY'
import sys
from pathlib import Path
script_dir = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
posthoc_dir = Path(sys.argv[3]) if sys.argv[3] else None
sys.path.insert(0, str(script_dir))
from src.orchestrator.paper_data_posthoc import collect_posthoc_analyses  # noqa: E402
print(1 if collect_posthoc_analyses(results_dir, posthoc_dir=posthoc_dir) else 0)
PY
)"
fi

if [ "$HAS_POSTHOC_ANALYSES" -eq 1 ]; then
  echo "[paper.sh] posthoc analyses found; preparing ranking context"
  "$PYTHON_BIN" - "$REPO_ROOT" "$RESULTS_DIR" "$OUTPUT_DIR" "$POSTHOC_DIR" <<'PY'
import json
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
posthoc_dir = Path(sys.argv[4]) if sys.argv[4] else None
sys.path.insert(0, str(script_dir))
from src.orchestrator.paper_data_posthoc import collect_posthoc_analyses, copy_posthoc_figures  # noqa: E402

analyses = collect_posthoc_analyses(results_dir, posthoc_dir=posthoc_dir)
copy_posthoc_figures(analyses, output_dir / "figures" / "posthoc")
# Write context file consumed by render_prompt to build POSTHOC_ANALYSES_CONTEXT
(output_dir / "posthoc_analyses_context.json").write_text(
    json.dumps(analyses, indent=2, sort_keys=True), encoding="utf-8",
)
print(f"[paper.sh] prepared {len(analyses)} posthoc analyses for ranking")
PY

  posthoc_ranking_repair_round=0
  while true; do
    POSTHOC_RANKING_PROMPT="$(render_prompt "$POSTHOC_RANKING_INSTRUCT_PATH" "$INSTRUCT_PATH")"
    if [ "$posthoc_ranking_repair_round" -eq 0 ]; then
      CURRENT_POSTHOC_RANKING_STAGE_NAME="posthoc_ranking"
      CURRENT_POSTHOC_RANKING_PROMPT_PATH="$OUTPUT_DIR/posthoc_ranking_prompt.md"
      CURRENT_POSTHOC_RANKING_LOG_PATH="$POSTHOC_RANKING_LOG_FILE"
      echo "[paper.sh] running posthoc ranking stage"
    else
      CURRENT_POSTHOC_RANKING_STAGE_NAME="posthoc_ranking_repair_round_${posthoc_ranking_repair_round}"
      CURRENT_POSTHOC_RANKING_PROMPT_PATH="$OUTPUT_DIR/posthoc_ranking_repair_prompt_round_${posthoc_ranking_repair_round}.md"
      CURRENT_POSTHOC_RANKING_LOG_PATH="$OUTPUT_DIR/posthoc_ranking_repair_agent_${AGENT}_${TIMESTAMP}_round_${posthoc_ranking_repair_round}.log"
      POSTHOC_RANKING_PROMPT="${POSTHOC_RANKING_PROMPT}

# Contract repair
Read ./posthoc_ranking_validation_feedback.md and replace ./posthoc_ranking.json with a corrected artifact."
      echo "[paper.sh] starting posthoc ranking repair round $posthoc_ranking_repair_round / $PAPER_MAX_REPAIR_ROUNDS"
    fi

    rm -f "$OUTPUT_DIR/posthoc_ranking.json"
    if run_prompt_stage_with_feedback \
      "$CURRENT_POSTHOC_RANKING_STAGE_NAME" \
      "$POSTHOC_RANKING_PROMPT" \
      "$CURRENT_POSTHOC_RANKING_PROMPT_PATH" \
      "$CURRENT_POSTHOC_RANKING_LOG_PATH" \
      "$POSTHOC_RANKING_VALIDATION_FEEDBACK_JSON_PATH" \
      "$POSTHOC_RANKING_VALIDATION_FEEDBACK_MD_PATH" \
      "posthoc_ranking.json"; then
      :
    else
      stage_rc=$?
      if [ "$stage_rc" -eq 130 ]; then
        exit 130
      fi
      if [ "$posthoc_ranking_repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
        echo "[paper.sh] exhausted posthoc ranking repair rounds after $posthoc_ranking_repair_round attempts; see $POSTHOC_RANKING_VALIDATION_FEEDBACK_JSON_PATH" >&2
        exit 1
      fi
      posthoc_ranking_repair_round=$((posthoc_ranking_repair_round + 1))
      continue
    fi

    if validate_posthoc_ranking_with_feedback; then
      echo "[paper.sh] posthoc ranking completed"
      break
    fi
    if [ "$posthoc_ranking_repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
      echo "[paper.sh] exhausted posthoc ranking repair rounds after $posthoc_ranking_repair_round attempts; see $POSTHOC_RANKING_VALIDATION_FEEDBACK_JSON_PATH" >&2
      exit 1
    fi
    posthoc_ranking_repair_round=$((posthoc_ranking_repair_round + 1))
  done
fi

if [ "$HAS_POSTHOC_ANALYSES" -eq 1 ] && [ -s "$OUTPUT_DIR/posthoc_ranking.json" ]; then
  "$PYTHON_BIN" - "$REPO_ROOT" "$RESULTS_DIR" "$OUTPUT_DIR" "$POSTHOC_DIR" <<'PY'
import json
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
posthoc_dir = Path(sys.argv[4]) if sys.argv[4] else None
sys.path.insert(0, str(script_dir))
from src.orchestrator.paper_data_posthoc import (  # noqa: E402
    augment_paper_data_with_posthoc,
    collect_posthoc_analyses,
)
from src.orchestrator.paper_data import write_paper_data  # noqa: E402

paper_data_path = output_dir / "paper_data.json"
paper_data = json.loads(paper_data_path.read_text(encoding="utf-8"))
ranking = json.loads((output_dir / "posthoc_ranking.json").read_text(encoding="utf-8"))
analyses = collect_posthoc_analyses(results_dir, posthoc_dir=posthoc_dir)
augmented = augment_paper_data_with_posthoc(paper_data, ranking, analyses)
write_paper_data(augmented, paper_data_path)
print(f"[paper.sh] augmented paper_data.json with {len(augmented.get('posthoc_analyses', []))} ranked posthoc analyses")
PY
else
  echo "[paper.sh] no posthoc analyses to integrate"
fi

VALIDATION_FEEDBACK_JSON_PATH="$OUTPUT_DIR/validation_feedback.json"
VALIDATION_FEEDBACK_MD_PATH="$OUTPUT_DIR/validation_feedback.md"
REVISIONS_DIR="$OUTPUT_DIR/revisions"

initialize_concept_figure_step "paper.sh"

validate_paper_with_feedback() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$CONTAINERIZE" "$CONTAINER_IMAGE" "$CONTAINER_NETWORK" "$VALIDATION_FEEDBACK_JSON_PATH" "$VALIDATION_FEEDBACK_MD_PATH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

script_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
containerize = sys.argv[3] == "1"
container_image = sys.argv[4]
container_network = sys.argv[5]
feedback_json_path = Path(sys.argv[6])
feedback_md_path = Path(sys.argv[7])
sys.path.insert(0, str(script_dir))

from src.orchestrator.paper_feedback import write_validation_feedback  # noqa: E402
from src.orchestrator.paper_step import validate_and_compile_paper  # noqa: E402

report: dict[str, object] | None = None
try:
    report = validate_and_compile_paper(
        output_dir,
        container_image=container_image if containerize else None,
        container_network=container_network,
    )
except Exception as exc:
    report_path = output_dir / "validation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    feedback = write_validation_feedback(
        output_dir=output_dir,
        feedback_json_path=feedback_json_path,
        feedback_md_path=feedback_md_path,
        report=report,
        error=exc,
    )
else:
    feedback = write_validation_feedback(
        output_dir=output_dir,
        feedback_json_path=feedback_json_path,
        feedback_md_path=feedback_md_path,
        report=report,
    )

hard_errors = feedback.get("hard_errors", [])
soft_warnings = feedback.get("soft_warnings", [])
if isinstance(soft_warnings, list) and soft_warnings:
    print("[paper.sh] figure validation feedback:")
    for message in soft_warnings:
        print(f"  - {message}")
if isinstance(hard_errors, list) and hard_errors:
    print("[paper.sh] hard validation errors:")
    for message in hard_errors:
        print(f"  - {message}")
    raise SystemExit(1)
print(f"[paper.sh] paper validation ok: pdf={feedback['pdf_path']}")
PY
}

has_actionable_figure_quality_warnings() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$VALIDATION_FEEDBACK_JSON_PATH" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
feedback_path = Path(sys.argv[2])
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import has_actionable_figure_quality_warnings  # noqa: E402

if not feedback_path.is_file():
    raise SystemExit(1)
feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
warnings = feedback.get("soft_warnings", [])
if not isinstance(warnings, list):
    raise SystemExit(1)
raise SystemExit(0 if has_actionable_figure_quality_warnings(warnings) else 1)
PY
}

validation_failure_fingerprint() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$VALIDATION_FEEDBACK_JSON_PATH" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
feedback_path = Path(sys.argv[2])
paper_dir = Path(sys.argv[3])
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import validation_failure_fingerprint  # noqa: E402

try:
    fingerprint = validation_failure_fingerprint(feedback_path, paper_dir)
except RuntimeError:
    raise SystemExit(1)
if fingerprint is None:
    raise SystemExit(1)
print(fingerprint)
PY
}

write_validation_feedback_from_current_report() {
  local error_message="${1:-}"

  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$VALIDATION_FEEDBACK_JSON_PATH" "$VALIDATION_FEEDBACK_MD_PATH" "$error_message" <<'PY'
import json
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
feedback_json_path = Path(sys.argv[3])
feedback_md_path = Path(sys.argv[4])
error_message = sys.argv[5].strip()
report_path = output_dir / "validation_report.json"
sys.path.insert(0, str(script_dir))

from src.orchestrator.paper_feedback import write_validation_feedback  # noqa: E402

report = None
if report_path.is_file():
    report = json.loads(report_path.read_text(encoding="utf-8"))
write_validation_feedback(
    output_dir=output_dir,
    feedback_json_path=feedback_json_path,
    feedback_md_path=feedback_md_path,
    report=report,
    error=RuntimeError(error_message) if error_message else None,
)
PY
}

finalize_paper_evidence_audit() {
  "$PYTHON_BIN" "$REPO_ROOT/src/orchestrator/paper_evidence_audit.py" \
    finalize-audit \
    --log "$EVIDENCE_AUDIT_LOG_FILE" \
    --raw-output "$EVIDENCE_AUDIT_RAW_OUTPUT_PATH" \
    --audit-json "$EVIDENCE_AUDIT_JSON_PATH" \
    --markdown-output "$EVIDENCE_AUDIT_MD_PATH" \
    --feedback-json "$EVIDENCE_AUDIT_FEEDBACK_JSON_PATH" \
    --feedback-md "$EVIDENCE_AUDIT_FEEDBACK_MD_PATH" \
    --validation-report "$OUTPUT_DIR/validation_report.json" \
    --input-hashes "$EVIDENCE_AUDIT_INPUT_HASHES_PATH" \
    --refresh-evidence-script "$REPO_ROOT/src/agents/skills/scientific-evidence/scripts/scientific_evidence.py"
}

paper_evidence_audit_has_findings() {
  "$PYTHON_BIN" - "$EVIDENCE_AUDIT_FEEDBACK_JSON_PATH" "$EVIDENCE_AUDIT_JSON_PATH" <<'PY'
import json
import sys
from pathlib import Path

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    if not path.is_file():
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    verdict = str(payload.get("verdict", "")).lower() if isinstance(payload, dict) else ""
    finding_count = payload.get("finding_count") if isinstance(payload, dict) else 0
    if finding_count is None and isinstance(payload.get("findings"), list):
        finding_count = len(payload["findings"])
    try:
        parsed_count = int(finding_count or 0)
    except (TypeError, ValueError):
        parsed_count = 0
    if verdict == "fail" or parsed_count > 0:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

clear_paper_evidence_audit_artifacts() {
  rm -f \
    "$EVIDENCE_AUDIT_RAW_OUTPUT_PATH" \
    "$EVIDENCE_AUDIT_JSON_PATH" \
    "$EVIDENCE_AUDIT_MD_PATH" \
    "$EVIDENCE_AUDIT_FEEDBACK_JSON_PATH" \
    "$EVIDENCE_AUDIT_FEEDBACK_MD_PATH"
}

clear_stale_paper_evidence_audit_artifacts() {
  rm -f \
    "$EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH" \
    "$EVIDENCE_AUDIT_STAGE_FEEDBACK_MD_PATH" \
    "$EVIDENCE_AUDIT_PACKET_PATH" \
    "$EVIDENCE_AUDIT_INPUT_HASHES_PATH" \
    "$EVIDENCE_AUDIT_LOG_FILE"
  clear_paper_evidence_audit_artifacts
}

cleanup_readonly_paper_audit_workspace() {
  local readonly_dir="$1"
  if [ -z "$readonly_dir" ]; then
    return 0
  fi
  "$PYTHON_BIN" "$REPO_ROOT/src/orchestrator/paper_evidence_audit.py" \
    cleanup-readonly-workspace \
    --path "$readonly_dir" \
    --source-dir "$OUTPUT_DIR"
}

prepare_readonly_paper_audit_workspace() {
  local workspace_tag="$1"
  local output_variable="$2"
  if ! "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "${INVESTIGATION_RESULTS_DIR:-$SOURCE_RESULTS_DIR}" "${FINAL_EVALUATION_DIR:-}" "${FINAL_EVALUATION_MODE:-discover}" "${CONTRACT_TASK_DIR:-}" "${FINAL_OUTCOME_POLICY:-qualified_only}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from src.orchestrator.paper_study_sources import verify_study_sources
verify_study_sources(Path(sys.argv[2]), Path(sys.argv[3]))
if sys.argv[5] == "discover" and sys.argv[7] == "allow_not_qualified":
    from src.orchestrator.paper_final_outcome import verify_paper_final_outcome
    final_dir = Path(sys.argv[4]) if sys.argv[4] else Path(sys.argv[3]) / "final_evaluation"
    if sys.argv[4] or final_dir.exists():
        verify_paper_final_outcome(
            Path(sys.argv[2]), final_evaluation_dir=final_dir,
            investigation_results_dir=Path(sys.argv[3]),
            task_dir=Path(sys.argv[6]) if sys.argv[6] else None,
            final_outcome_policy=sys.argv[7],
        )
PY
  then
    return 1
  fi
  if [ -n "${ACTIVE_READONLY_AUDIT_WORKSPACE:-}" ]; then
    if cleanup_readonly_paper_audit_workspace "$ACTIVE_READONLY_AUDIT_WORKSPACE"; then
      ACTIVE_READONLY_AUDIT_WORKSPACE=""
    else
      echo "[paper.sh] refusing to replace an audit workspace whose cleanup failed: $ACTIVE_READONLY_AUDIT_WORKSPACE" >&2
      return 1
    fi
  fi
  local readonly_dir=""
  # Keep the transient read-only audit copy on the same shared filesystem as the
  # paper output, but outside OUTPUT_DIR itself. Putting it under OUTPUT_DIR would
  # make the source copy recursive; putting it under node-local /tmp makes
  # container mounts and post-failure debugging brittle. The helper validates
  # ownership and permissions before creating an isolated per-user workspace.
  if readonly_dir="$("$PYTHON_BIN" "$REPO_ROOT/src/orchestrator/paper_evidence_audit.py" \
    prepare-audit-workspace \
    --source-dir "$OUTPUT_DIR" \
    --workspace-tag "${TIMESTAMP}-${workspace_tag}" \
    --refresh-evidence-script "$REPO_ROOT/src/agents/skills/scientific-evidence/scripts/scientific_evidence.py" \
  )"; then
    ACTIVE_READONLY_AUDIT_WORKSPACE="$readonly_dir"
    printf -v "$output_variable" '%s' "$readonly_dir"
    return 0
  fi
  return 1
}

run_readonly_prompt_stage() {
  local stage_name="$1"
  local prompt_text="$2"
  local prompt_path="$3"
  local log_path="$4"
  local stage_work_dir="$5"
  local stage_log_pipe="$OUTPUT_DIR/.${stage_name}_stream_${TIMESTAMP}.fifo"
  local stage_agent_exit_code=0
  local stage_tee_exit_code=0
  local agent_prompt_file
  local cost_invocation="paper-readonly:${TIMESTAMP}:${stage_name}:${log_path##*/}"
  agent_prompt_file="$(basename "$prompt_path")"

  if [ "$CONTAINERIZE" -ne 1 ]; then
    echo "[paper.sh] evidence audit read-only stage requires --containerize; refusing native agent execution" >&2
    return 1
  fi

  if ! agent_prompt_file_matches_text "$stage_work_dir/$agent_prompt_file" "$prompt_text"; then
    echo "[paper.sh] read-only stage prompt failed integrity check: $stage_work_dir/$agent_prompt_file" >&2
    return 1
  fi
  ACTIVE_COST_LOG="$log_path"
  ACTIVE_COST_STAGE="$stage_name"
  ACTIVE_COST_INVOCATION="$cost_invocation"
  LOG_PIPE="$stage_log_pipe"
  mkfifo "$stage_log_pipe"
  bounded_agent_log_writer "$PYTHON_BIN" "$log_path" "$AGENT_LOG_MAX_BYTES" \
    < "$stage_log_pipe" &
  TEE_PID=$!

  (
    exec 3>&-
    export AGENT_WORKDIR_READONLY=1
    unset AGENT_CONTAINER_SESSION_NAME
    AGENT_PROMPT_FILE="$agent_prompt_file" \
    run_agent_prompt \
      "$AGENT" \
      "$prompt_text" \
      "$AGENT_MODEL" \
      "$AGENT_PROFILE" \
      "$stage_work_dir" \
      "$CONTAINERIZE" \
      "$CONTAINER_IMAGE" \
      "$CONTAINER_NETWORK" \
      "$CONTAINER_HOME_DIR" \
      "$CONTAINER_ENV_FILE" \
      "$RESULTS_DIR"
  ) > "$stage_log_pipe" 2>&1 &
  AGENT_PID=$!

  if [ -n "${AGENT_PID:-}" ]; then
    while kill -0 "$AGENT_PID" 2>/dev/null; do
      paper_maybe_heartbeat
      sleep 1
    done
    if wait "$AGENT_PID"; then
      stage_agent_exit_code=0
    else
      stage_agent_exit_code=$?
    fi
  fi
  if [ -n "${TEE_PID:-}" ]; then
    if wait "$TEE_PID"; then
      stage_tee_exit_code=0
    else
      stage_tee_exit_code=$?
    fi
  fi

  record_headless_cost_log \
    "$PYTHON_BIN" "$log_path" "$stage_name" \
    "$cost_invocation" \
    "$(headless_cost_status "$stage_agent_exit_code" "$INTERRUPTED")"
  ACTIVE_COST_LOG=""
  ACTIVE_COST_STAGE=""
  ACTIVE_COST_INVOCATION=""

  AGENT_PID=""
  TEE_PID=""
  if [ -p "$stage_log_pipe" ]; then
    rm -f "$stage_log_pipe"
  fi
  LOG_PIPE=""

  if [ "$INTERRUPTED" -eq 1 ]; then
    return 130
  fi
  if [ "$stage_agent_exit_code" -ne 0 ]; then
    return "$stage_agent_exit_code"
  fi
  if [ "$stage_tee_exit_code" -ne 0 ]; then
    return "$stage_tee_exit_code"
  fi
  if [ ! -s "$log_path" ]; then
    echo "[paper.sh] stage '$stage_name' returned success but produced an empty log: $log_path" >&2
    return 70
  fi
  return 0
}

run_agentic_evidence_audit_gate() {
  local audit_attempt="$1"
  local contract_attempt=1
  local max_contract_attempts=2
  local audit_prompt=""
  local audit_prompt_path=""
  local audit_log_path=""
  local audit_stage_name=""
  local readonly_workspace=""
  local audit_stage_rc=0
  local cleanup_rc=0
  local finalize_rc=0
  local finalize_output=""

  clear_stale_paper_evidence_audit_artifacts

  while true; do
    if [ "$audit_attempt" -eq 1 ] && [ "$contract_attempt" -eq 1 ]; then
      audit_prompt_path="$OUTPUT_DIR/paper_evidence_audit_prompt.md"
      audit_log_path="$EVIDENCE_AUDIT_LOG_FILE"
      audit_stage_name="paper_evidence_audit"
    elif [ "$contract_attempt" -eq 1 ]; then
      audit_prompt_path="$OUTPUT_DIR/paper_evidence_audit_prompt_round_${audit_attempt}.md"
      audit_log_path="$OUTPUT_DIR/paper_evidence_audit_agent_${AGENT}_${TIMESTAMP}_round_${audit_attempt}.log"
      audit_stage_name="paper_evidence_audit_round_${audit_attempt}"
    else
      audit_prompt_path="$OUTPUT_DIR/paper_evidence_audit_prompt_round_${audit_attempt}_contract_retry_${contract_attempt}.md"
      audit_log_path="$OUTPUT_DIR/paper_evidence_audit_agent_${AGENT}_${TIMESTAMP}_round_${audit_attempt}_contract_retry_${contract_attempt}.log"
      audit_stage_name="paper_evidence_audit_round_${audit_attempt}_contract_retry_${contract_attempt}"
    fi
    audit_prompt="$(render_prompt "$PAPER_EVIDENCE_AUDIT_INSTRUCT_PATH" "$CURRENT_PAPER_INSTRUCT_PATH")"
    if ! write_agent_prompt_file "$audit_prompt_path" "$audit_prompt"; then
      echo "[paper.sh] failed to stage evidence audit prompt: $audit_prompt_path" >&2
      return 1
    fi
    readonly_workspace=""
    if ! prepare_readonly_paper_audit_workspace "$audit_stage_name" readonly_workspace; then
      echo "[paper.sh] failed to prepare read-only evidence audit workspace for '$audit_stage_name'" >&2
      return 1
    fi
    if [ -z "$readonly_workspace" ] || [ ! -f "$readonly_workspace/paper_evidence_audit_packet.md" ]; then
      echo "[paper.sh] read-only evidence audit workspace is missing paper_evidence_audit_packet.md: ${readonly_workspace:-<empty>}" >&2
      if cleanup_readonly_paper_audit_workspace "$readonly_workspace"; then
        ACTIVE_READONLY_AUDIT_WORKSPACE=""
      fi
      return 1
    fi
    rm -f "$EVIDENCE_AUDIT_PACKET_PATH"
    install -m 0644 "$readonly_workspace/paper_evidence_audit_packet.md" "$EVIDENCE_AUDIT_PACKET_PATH"
    rm -f "$EVIDENCE_AUDIT_INPUT_HASHES_PATH"
    install -m 0644 "$readonly_workspace/.paper_evidence_audit_input_sha256.json" "$EVIDENCE_AUDIT_INPUT_HASHES_PATH"

    if run_readonly_prompt_stage \
      "$audit_stage_name" \
      "$audit_prompt" \
      "$audit_prompt_path" \
      "$audit_log_path" \
      "$readonly_workspace"; then
      audit_stage_rc=0
    else
      audit_stage_rc=$?
    fi
    if cleanup_readonly_paper_audit_workspace "$readonly_workspace"; then
      cleanup_rc=0
      ACTIVE_READONLY_AUDIT_WORKSPACE=""
    else
      cleanup_rc=$?
    fi
    if [ "$cleanup_rc" -ne 0 ]; then
      echo "[paper.sh] failed to clean read-only evidence audit workspace: $readonly_workspace" >&2
      set +e
      return 1
    fi
    if [ "$audit_stage_rc" -eq 130 ]; then
      set +e
      return 130
    fi
    if [ "$audit_log_path" != "$EVIDENCE_AUDIT_LOG_FILE" ] && [ -f "$audit_log_path" ]; then
      cp "$audit_log_path" "$EVIDENCE_AUDIT_LOG_FILE"
    fi

    if finalize_output="$(finalize_paper_evidence_audit 2>&1)"; then
      finalize_rc=0
    else
      finalize_rc=$?
    fi
    if [ "$finalize_rc" -eq 0 ]; then
      if [ "$audit_stage_rc" -ne 0 ]; then
        echo "[paper.sh] agentic evidence audit contract accepted despite agent exit code $audit_stage_rc" >&2
      fi
      rm -f "$EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH" "$EVIDENCE_AUDIT_STAGE_FEEDBACK_MD_PATH"
      write_validation_feedback_from_current_report
      echo "[paper.sh] agentic evidence audit passed: $EVIDENCE_AUDIT_JSON_PATH"
      return 0
    fi
    if [ "$finalize_rc" -eq 2 ]; then
      write_validation_feedback_from_current_report \
        "agentic evidence audit failed: see $EVIDENCE_AUDIT_FEEDBACK_JSON_PATH and $EVIDENCE_AUDIT_MD_PATH"
      echo "[paper.sh] agentic evidence audit found mismatches: $EVIDENCE_AUDIT_FEEDBACK_JSON_PATH" >&2
      # The outer loop intentionally captures status 2 to start a repair round.
      # finalize-audit is run under a temporary `set +e` block and then restores
      # errexit; a bare `return 2` here would terminate the script before the
      # caller can inspect the status.
      set +e
      return 2
    fi
    if [ "$audit_stage_rc" -ne 0 ]; then
      write_stage_execution_feedback \
        "$audit_stage_name" \
        "$EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH" \
        "$EVIDENCE_AUDIT_STAGE_FEEDBACK_MD_PATH" \
        "$audit_prompt_path" \
        "$audit_log_path" \
        "$audit_stage_rc"
      echo "[paper.sh] evidence audit stage '$audit_stage_name' failed and produced no valid audit contract; feedback: $EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH" >&2
      set +e
      return 1
    fi
    if [ "$finalize_rc" -ne 3 ]; then
      echo "[paper.sh] evidence audit finalization failed operationally: ${finalize_output:-exit $finalize_rc}" >&2
      set +e
      return 1
    fi

    write_stage_contract_feedback \
      "$audit_stage_name" \
      "$EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH" \
      "$EVIDENCE_AUDIT_STAGE_FEEDBACK_MD_PATH" \
      "$audit_prompt_path" \
      "$audit_log_path" \
      "evidence audit stage '$audit_stage_name' wrote an invalid audit artifact: ${finalize_output:-finalize-audit exited with code $finalize_rc}"
    echo "[paper.sh] invalid evidence audit output contract" >&2
    if [ -n "$finalize_output" ]; then
      printf '%s\n' "$finalize_output" >&2
    fi
    if [ "$contract_attempt" -ge "$max_contract_attempts" ]; then
      echo "[paper.sh] evidence audit artifact remained invalid after $contract_attempt attempts" >&2
      set +e
      return 3
    fi
    echo "[paper.sh] retrying evidence audit stage after contract failure; feedback: $EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH" >&2
    contract_attempt=$((contract_attempt + 1))
  done
}

save_review_revision_pdf() {
  local review_round="$1"
  local next_repair_round="$2"
  local source_pdf="$OUTPUT_DIR/paper.pdf"
  if [ ! -f "$source_pdf" ]; then
    echo "[paper.sh] review round $review_round requested repair but paper.pdf is missing; skipping revision snapshot" >&2
    return 0
  fi
  mkdir -p "$REVISIONS_DIR"
  local snapshot_path="$REVISIONS_DIR/review_round_$(printf '%02d' "$review_round")_before_repair_round_$(printf '%02d' "$next_repair_round").pdf"
  cp "$source_pdf" "$snapshot_path"
  echo "[paper.sh] saved pre-repair review snapshot: $snapshot_path"
}

save_audit_revision_pdf() {
  local next_repair_round="$1"
  local source_pdf="$OUTPUT_DIR/paper.pdf"
  if [ ! -f "$source_pdf" ]; then
    echo "[paper.sh] evidence audit requested repair but paper.pdf is missing; skipping revision snapshot" >&2
    return 0
  fi
  mkdir -p "$REVISIONS_DIR"
  local snapshot_path="$REVISIONS_DIR/evidence_audit_before_repair_round_$(printf '%02d' "$next_repair_round").pdf"
  cp "$source_pdf" "$snapshot_path"
  echo "[paper.sh] saved pre-repair evidence audit snapshot: $snapshot_path"
}

build_paper_review_packet() {
  "$PYTHON_BIN" "$REPO_ROOT/src/orchestrator/paper_review.py" \
    build-packet \
    --paper-dir "$OUTPUT_DIR" \
    --output "$REVIEW_PACKET_PATH"
}

finalize_paper_review() {
  "$PYTHON_BIN" "$REPO_ROOT/src/orchestrator/paper_review.py" \
    finalize-review \
    --input "$REVIEW_JSON_PATH" \
    --markdown-output "$REVIEW_MD_PATH" \
    --feedback-json "$REVIEW_FEEDBACK_JSON_PATH" \
    --feedback-md "$REVIEW_FEEDBACK_MD_PATH"
}

review_feedback_has_major_comments() {
  "$PYTHON_BIN" -c "import json, sys; from pathlib import Path; data = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); raise SystemExit(0 if int(data.get('major_comment_count', 0)) > 0 else 1)" "$REVIEW_FEEDBACK_JSON_PATH"
}

clear_review_feedback_artifacts() {
  rm -f "$REVIEW_FEEDBACK_JSON_PATH" "$REVIEW_FEEDBACK_MD_PATH" "$REVIEW_MD_PATH"
}

clear_stale_review_artifacts() {
  rm -f \
    "$REVIEW_STAGE_FEEDBACK_JSON_PATH" \
    "$REVIEW_STAGE_FEEDBACK_MD_PATH" \
    "$REVIEW_JSON_PATH"
  clear_review_feedback_artifacts
}

run_paper_review_gate() {
  local review_attempt="$1"
  local review_contract_attempt=1
  local max_review_contract_attempts=2
  local review_prompt_path=""
  local review_log_path=""
  local review_stage_name=""
  local review_prompt=""
  local review_stage_rc=0
  local finalize_rc=0
  local finalize_output=""
  local review_json_backup_path="$OUTPUT_DIR/.paper_review_previous.json"

  clear_stale_review_artifacts

  while true; do
    build_paper_review_packet >/dev/null
    review_prompt="$(render_prompt "$PAPER_REVIEW_INSTRUCT_PATH" "$CURRENT_PAPER_INSTRUCT_PATH")"
    if [ "$review_attempt" -eq 1 ] && [ "$review_contract_attempt" -eq 1 ]; then
      review_prompt_path="$OUTPUT_DIR/paper_review_prompt.md"
      review_log_path="$REVIEW_LOG_FILE"
      review_stage_name="paper_review"
    elif [ "$review_contract_attempt" -eq 1 ]; then
      review_prompt_path="$OUTPUT_DIR/paper_review_prompt_round_${review_attempt}.md"
      review_log_path="$OUTPUT_DIR/paper_review_agent_${AGENT}_${TIMESTAMP}_round_${review_attempt}.log"
      review_stage_name="paper_review_round_${review_attempt}"
    else
      review_prompt_path="$OUTPUT_DIR/paper_review_prompt_round_${review_attempt}_contract_retry_${review_contract_attempt}.md"
      review_log_path="$OUTPUT_DIR/paper_review_agent_${AGENT}_${TIMESTAMP}_round_${review_attempt}_contract_retry_${review_contract_attempt}.log"
      review_stage_name="paper_review_round_${review_attempt}_contract_retry_${review_contract_attempt}"
    fi

    rm -f "$review_json_backup_path"
    if [ -f "$REVIEW_JSON_PATH" ]; then
      mv "$REVIEW_JSON_PATH" "$review_json_backup_path"
    fi

    if run_prompt_stage_with_feedback \
      "$review_stage_name" \
      "$review_prompt" \
      "$review_prompt_path" \
      "$review_log_path" \
      "$REVIEW_STAGE_FEEDBACK_JSON_PATH" \
      "$REVIEW_STAGE_FEEDBACK_MD_PATH" \
      "paper_review.json"; then
      review_stage_rc=0
    else
      review_stage_rc=$?
    fi
    if [ "$review_stage_rc" -ne 0 ]; then
      if [ ! -s "$REVIEW_JSON_PATH" ] && [ -f "$review_json_backup_path" ]; then
        mv "$review_json_backup_path" "$REVIEW_JSON_PATH"
      fi
      set +e
      return "$review_stage_rc"
    fi
    rm -f "$review_json_backup_path"

    set +e
    finalize_output="$(finalize_paper_review 2>&1)"
    finalize_rc=$?
    set -e
    if [ "$finalize_rc" -eq 0 ]; then
      rm -f "$REVIEW_STAGE_FEEDBACK_JSON_PATH" "$REVIEW_STAGE_FEEDBACK_MD_PATH"
      if review_feedback_has_major_comments; then
        echo "[paper.sh] paper review complete with major comments: $REVIEW_FEEDBACK_JSON_PATH"
      else
        echo "[paper.sh] paper review complete with only minor comments (or none): $REVIEW_FEEDBACK_JSON_PATH"
      fi
      return 0
    fi

    write_stage_contract_feedback \
      "$review_stage_name" \
      "$REVIEW_STAGE_FEEDBACK_JSON_PATH" \
      "$REVIEW_STAGE_FEEDBACK_MD_PATH" \
      "$review_prompt_path" \
      "$review_log_path" \
      "review stage '$review_stage_name' wrote an invalid paper review artifact: ${finalize_output:-finalize-review exited with code $finalize_rc}" \
      "paper_review.json"
    echo "[paper.sh] invalid paper review artifact: $REVIEW_JSON_PATH" >&2
    if [ -n "$finalize_output" ]; then
      printf '%s\n' "$finalize_output" >&2
    fi
    if [ "$review_contract_attempt" -ge "$max_review_contract_attempts" ]; then
      clear_review_feedback_artifacts
      echo "[paper.sh] review artifact remained invalid after $review_contract_attempt attempts; continuing into manuscript repair without structured review feedback" >&2
      return 0
    fi
    echo "[paper.sh] retrying review stage after contract failure; feedback: $REVIEW_STAGE_FEEDBACK_JSON_PATH" >&2
    review_contract_attempt=$((review_contract_attempt + 1))
  done
}

FINAL_PAPER_PIPELINE_INCOMPLETE=0
FINAL_PAPER_PIPELINE_REASON=""

mark_incomplete_paper_pipeline() {
  local reason="$1"
  FINAL_PAPER_PIPELINE_INCOMPLETE=1
  if [ -z "$FINAL_PAPER_PIPELINE_REASON" ]; then
    FINAL_PAPER_PIPELINE_REASON="$reason"
  else
    FINAL_PAPER_PIPELINE_REASON="$FINAL_PAPER_PIPELINE_REASON; $reason"
  fi
  echo "[paper.sh] warning: $reason" >&2
  if [ ! -f "$VALIDATION_FEEDBACK_JSON_PATH" ] || [ ! -f "$VALIDATION_FEEDBACK_MD_PATH" ]; then
    write_validation_feedback_from_current_report "$reason"
  fi
}

repair_round=0
review_round=0
figure_quality_repair_attempted=0
figure_quality_repair_pending=0
figure_quality_repair_warning_reported=0
previous_validation_failure_fingerprint=""
if [ "$PAPER_REVIEW_ROUNDS" -eq 0 ]; then
  paper_gate_start "review"
  paper_gate_finish "review" "skipped" "reason=disabled"
fi
while true; do
  current_stage_is_figure_quality_repair=0
  figure_quality_repair_reverted=0
  validation_failure_repeated=0
  if [ "$figure_quality_repair_pending" -eq 1 ]; then
    PROMPT="$(render_prompt "$PAPER_FIGURE_QUALITY_REPAIR_INSTRUCT_PATH" "$CURRENT_PAPER_INSTRUCT_PATH")"
    CURRENT_PAPER_PROMPT_PATH="$OUTPUT_DIR/paper_figure_quality_repair_prompt.md"
    CURRENT_PAPER_LOG_PATH="$OUTPUT_DIR/paper_figure_quality_repair_agent_${AGENT}_${TIMESTAMP}.log"
    CURRENT_PAPER_STAGE_NAME="paper_figure_quality_repair"
    current_stage_is_figure_quality_repair=1
    figure_quality_repair_pending=0
    echo "[paper.sh] starting targeted figure-quality repair"
    echo "Figure-quality repair prompt file: $CURRENT_PAPER_PROMPT_PATH"
    echo "Figure-quality repair log: $CURRENT_PAPER_LOG_PATH"
  elif [ "$repair_round" -eq 0 ]; then
    PROMPT="$(render_prompt "$CURRENT_PAPER_INSTRUCT_PATH" "$CURRENT_PAPER_INSTRUCT_PATH")"
    CURRENT_PAPER_PROMPT_PATH="$OUTPUT_DIR/paper_prompt.md"
    CURRENT_PAPER_LOG_PATH="$PAPER_LOG_FILE"
    CURRENT_PAPER_STAGE_NAME="paper"
    echo "Final paper prompt file: $CURRENT_PAPER_PROMPT_PATH"
    echo "Final paper log: $CURRENT_PAPER_LOG_PATH"
  else
    PROMPT="$(render_prompt "$CURRENT_PAPER_INSTRUCT_PATH" "$CURRENT_PAPER_INSTRUCT_PATH")"
    CURRENT_PAPER_PROMPT_PATH="$OUTPUT_DIR/paper_repair_prompt_round_${repair_round}.md"
    CURRENT_PAPER_LOG_PATH="$OUTPUT_DIR/paper_repair_agent_${AGENT}_${TIMESTAMP}_round_${repair_round}.log"
    CURRENT_PAPER_STAGE_NAME="paper_repair_round_${repair_round}"
    echo "[paper.sh] starting repair round $repair_round / $PAPER_MAX_REPAIR_ROUNDS"
    echo "Repair prompt file: $CURRENT_PAPER_PROMPT_PATH"
    echo "Repair log: $CURRENT_PAPER_LOG_PATH"
  fi

  if [ "$current_stage_is_figure_quality_repair" -eq 1 ]; then
    paper_gate_start "repair" "phase=figure_quality" "round=$repair_round"
  elif [ "$repair_round" -eq 0 ]; then
    paper_gate_start "draft" "round=0"
  else
    paper_gate_start "repair" "phase=manuscript" "round=$repair_round"
  fi

  if run_prompt_stage_with_feedback \
    "$CURRENT_PAPER_STAGE_NAME" \
    "$PROMPT" \
    "$CURRENT_PAPER_PROMPT_PATH" \
    "$CURRENT_PAPER_LOG_PATH" \
    "$VALIDATION_FEEDBACK_JSON_PATH" \
    "$VALIDATION_FEEDBACK_MD_PATH" \
    "paper.tex"; then
    if [ "$current_stage_is_figure_quality_repair" -eq 1 ]; then
      if ! paper_figure_quality_complete_agent_session \
        "$CONTAINERIZE" "${AGENT_CONTAINER_SESSION_NAME:-}"; then
        echo "[paper.sh] could not stop the figure-quality agent session before promotion" >&2
        exit 1
      fi
      unset AGENT_CONTAINER_SESSION_NAME
      FIGURE_QUALITY_REPAIR_AGENT_QUIESCED=1
    fi
    if [ "$repair_round" -eq 0 ] && [ "$current_stage_is_figure_quality_repair" -eq 0 ]; then
      paper_gate_complete "draft" "round=0"
    elif [ "$current_stage_is_figure_quality_repair" -eq 1 ]; then
      paper_gate_complete "repair" "phase=figure_quality" "round=$repair_round"
    else
      paper_gate_complete "repair" "phase=manuscript" "round=$repair_round"
    fi
  else
    stage_rc=$?
    if [ "$repair_round" -eq 0 ] && [ "$current_stage_is_figure_quality_repair" -eq 0 ]; then
      paper_gate_failed "draft" "round=0" "exit_code=$stage_rc"
    elif [ "$current_stage_is_figure_quality_repair" -eq 1 ]; then
      paper_gate_failed "repair" "phase=figure_quality" "round=$repair_round" "exit_code=$stage_rc"
    else
      paper_gate_failed "repair" "phase=manuscript" "round=$repair_round" "exit_code=$stage_rc"
    fi
    if [ "$current_stage_is_figure_quality_repair" -eq 1 ]; then
      if ! paper_figure_quality_stop_agent_session \
        "$CONTAINERIZE" "${AGENT_CONTAINER_SESSION_NAME:-}"; then
        echo "[paper.sh] could not stop the failed figure-quality agent session before rollback" >&2
        exit 1
      fi
      unset AGENT_CONTAINER_SESSION_NAME
      FIGURE_QUALITY_REPAIR_AGENT_QUIESCED=1
      paper_figure_quality_restore_workspace \
        "$FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR" "$OUTPUT_DIR" \
        "$FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256"
      FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR=""
      FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256=""
      figure_quality_repair_reverted=1
      echo "[paper.sh] targeted figure-quality repair failed; restored the valid pre-repair paper" >&2
      if [ "$stage_rc" -eq 130 ]; then
        exit 130
      fi
    elif [ "$stage_rc" -eq 130 ]; then
      exit 130
    elif [ "$repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
      mark_incomplete_paper_pipeline "exhausted repair rounds after $repair_round attempts; see $VALIDATION_FEEDBACK_JSON_PATH"
      break
    else
      repair_round=$((repair_round + 1))
      continue
    fi
  fi

  if [ "$repair_round" -eq 0 ] && [ ! -f "$POSTDRAFT_REFERENCE_ENRICHMENT_MARKER_PATH" ]; then
    paper_gate_start "references" "phase=postdraft"
    set +e
    run_postdraft_reference_enrichment
    postdraft_enrichment_rc=$?
    set -e
    if [ "$postdraft_enrichment_rc" -eq 0 ]; then
      paper_gate_complete "references" "phase=postdraft"
    else
      paper_gate_failed "references" "phase=postdraft" "exit_code=$postdraft_enrichment_rc"
    fi
    if [ "$postdraft_enrichment_rc" -eq 130 ]; then
      exit 130
    fi
    if [ "$postdraft_enrichment_rc" -ne 0 ]; then
      exit "$postdraft_enrichment_rc"
    fi
  fi

  concept_figure_rc=0
  paper_gate_start "figures" "round=$repair_round"
  if [ "$figure_quality_repair_reverted" -eq 1 ]; then
    concept_figure_rc=0
  elif prepare_concept_figure_for_validation "$repair_round" "refresh"; then
    concept_figure_rc=0
  else
    concept_figure_rc=$?
  fi
  if [ "$concept_figure_rc" -eq 130 ]; then
    paper_gate_failed "figures" "round=$repair_round" "exit_code=130"
    exit 130
  fi
  if [ "$concept_figure_rc" -eq 0 ]; then
    paper_gate_complete "figures" "round=$repair_round"
  else
    paper_gate_failed "figures" "round=$repair_round" "exit_code=$concept_figure_rc"
  fi

  validation_stage_rc=0
  audit_stage_rc=0
  review_stage_rc=0

  paper_gate_start "validation" "round=$repair_round"
  set +e
  validate_paper_with_feedback
  validation_stage_rc=$?
  archive_paper_validation "$repair_round" || exit 1
  set -e
  if [ "$validation_stage_rc" -eq 130 ]; then
    if [ "$current_stage_is_figure_quality_repair" -eq 1 ] \
      && [ "$figure_quality_repair_reverted" -eq 0 ]; then
      paper_figure_quality_restore_workspace \
        "$FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR" "$OUTPUT_DIR" \
        "$FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256"
      FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR=""
      FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256=""
    fi
    exit 130
  fi
  if [ "$current_stage_is_figure_quality_repair" -eq 1 ] \
    && [ "$validation_stage_rc" -ne 0 ] \
    && [ "$figure_quality_repair_reverted" -eq 0 ]; then
    paper_figure_quality_restore_workspace \
      "$FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR" "$OUTPUT_DIR" \
      "$FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256"
    FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR=""
    FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256=""
    figure_quality_repair_reverted=1
    echo "[paper.sh] targeted figure-quality repair regressed validation; restored the valid pre-repair paper" >&2
    set +e
    validate_paper_with_feedback
    validation_stage_rc=$?
    archive_paper_validation "$repair_round" || exit 1
    set -e
  fi
  if [ "$validation_stage_rc" -eq 130 ]; then
    paper_gate_failed "validation" "round=$repair_round" "exit_code=130"
    exit 130
  fi
  if [ "$validation_stage_rc" -eq 0 ]; then
    paper_gate_complete "validation" "round=$repair_round"
  else
    paper_gate_failed "validation" "round=$repair_round" "exit_code=$validation_stage_rc"
  fi
  if [ "$validation_stage_rc" -ne 0 ]; then
    echo "[paper.sh] validation failed; continuing because validation is non-gating" >&2
    current_validation_failure_fingerprint=""
    set +e
    current_validation_failure_fingerprint="$(validation_failure_fingerprint)"
    fingerprint_rc=$?
    set -e
    if [ "$fingerprint_rc" -eq 0 ]; then
      if [ -n "$previous_validation_failure_fingerprint" ] \
        && [ "$current_validation_failure_fingerprint" = "$previous_validation_failure_fingerprint" ]; then
        validation_failure_repeated=1
      fi
      previous_validation_failure_fingerprint="$current_validation_failure_fingerprint"
    else
      previous_validation_failure_fingerprint=""
    fi
  else
    previous_validation_failure_fingerprint=""
  fi

  paper_gate_start "evidence_audit" "round=$((repair_round + 1))"
  set +e
  run_agentic_evidence_audit_gate "$((repair_round + 1))"
  audit_stage_rc=$?
  archive_paper_evidence_audit "$((repair_round + 1))" || exit 1
  set -e
  if [ "$audit_stage_rc" -eq 130 ]; then
    paper_gate_failed "evidence_audit" "round=$((repair_round + 1))" "exit_code=130"
    if [ "$current_stage_is_figure_quality_repair" -eq 1 ] \
      && [ "$figure_quality_repair_reverted" -eq 0 ]; then
      paper_figure_quality_restore_workspace \
        "$FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR" "$OUTPUT_DIR" \
        "$FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256"
      FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR=""
      FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256=""
    fi
    exit 130
  fi
  audit_has_findings=0
  if [ "$audit_stage_rc" -eq 2 ] || paper_evidence_audit_has_findings; then
    audit_has_findings=1
  fi
  if [ "$audit_stage_rc" -eq 0 ] && [ "$audit_has_findings" -eq 0 ]; then
    paper_gate_complete "evidence_audit" "round=$((repair_round + 1))"
  else
    paper_gate_failed "evidence_audit" "round=$((repair_round + 1))" "exit_code=$audit_stage_rc" "findings=$audit_has_findings"
  fi
  if [ "$current_stage_is_figure_quality_repair" -eq 1 ] \
    && [ "$figure_quality_repair_reverted" -eq 0 ] \
    && { [ "$audit_stage_rc" -ne 0 ] || [ "$audit_has_findings" -eq 1 ]; }; then
    paper_figure_quality_restore_workspace \
      "$FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR" "$OUTPUT_DIR" \
      "$FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256"
    FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR=""
    FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256=""
    figure_quality_repair_reverted=1
    echo "[paper.sh] targeted figure-quality repair did not pass the evidence audit; restored the valid pre-repair paper" >&2
    audit_stage_rc=0
    audit_has_findings=0
  elif [ "$current_stage_is_figure_quality_repair" -eq 1 ] \
    && [ "$figure_quality_repair_reverted" -eq 0 ]; then
    paper_figure_quality_discard_snapshot "$FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR"
    FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR=""
    FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256=""
  fi
  if [ "$audit_has_findings" -eq 1 ]; then
    if [ "$repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
      mark_incomplete_paper_pipeline "paper evidence audit reported revision findings after exhausting repair rounds; see $EVIDENCE_AUDIT_FEEDBACK_JSON_PATH"
      break
    fi
    authorize_concept_figure_repair
    echo "[paper.sh] evidence audit reported revision findings; starting repair round $((repair_round + 1)) / $PAPER_MAX_REPAIR_ROUNDS" >&2
    save_audit_revision_pdf "$((repair_round + 1))"
    repair_round=$((repair_round + 1))
    continue
  fi
  if [ "$audit_stage_rc" -eq 1 ]; then
    mark_incomplete_paper_pipeline "paper evidence audit failed operationally; see $EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH"
    break
  fi
  if [ "$audit_stage_rc" -eq 3 ]; then
    echo "[paper.sh] evidence audit output contract remained invalid; continuing because contract exhaustion is non-gating" >&2
  fi

  if [ "$review_round" -lt "$PAPER_REVIEW_ROUNDS" ]; then
    if [ "$validation_stage_rc" -ne 0 ]; then
      echo "[paper.sh] validation failed before review round $((review_round + 1)); continuing into review gate because review rounds are enabled" >&2
    fi
    paper_gate_start "review" "round=$((review_round + 1))"
    set +e
    run_paper_review_gate "$((review_round + 1))"
    review_stage_rc=$?
    set -e
    if [ "$review_stage_rc" -eq 130 ]; then
      paper_gate_failed "review" "round=$((review_round + 1))" "exit_code=130"
      exit 130
    fi
    if [ "$review_stage_rc" -ne 0 ]; then
      paper_gate_failed "review" "round=$((review_round + 1))" "exit_code=$review_stage_rc"
      mark_incomplete_paper_pipeline "paper review stage failed with exit code $review_stage_rc; see $REVIEW_STAGE_FEEDBACK_JSON_PATH"
      break
    fi
    paper_gate_complete "review" "round=$((review_round + 1))"
    review_round=$((review_round + 1))
    if [ "$repair_round" -ge "$PAPER_MAX_REPAIR_ROUNDS" ]; then
      mark_incomplete_paper_pipeline "exhausted repair rounds after $repair_round attempts; see $VALIDATION_FEEDBACK_JSON_PATH"
      break
    fi
    save_review_revision_pdf "$review_round" "$((repair_round + 1))"
    repair_round=$((repair_round + 1))
    continue
  fi

  if [ "$validation_stage_rc" -eq 0 ] && has_actionable_figure_quality_warnings; then
    if [ "$figure_quality_repair_attempted" -eq 0 ] && [ "$repair_round" -lt "$PAPER_MAX_REPAIR_ROUNDS" ]; then
      figure_quality_repair_attempted=1
      FIGURE_QUALITY_REPAIR_AGENT_QUIESCED=0
      if [ "$CONTAINERIZE" -ne 1 ]; then
        figure_quality_repair_warning_reported=1
        echo "[paper.sh] targeted figure-quality repair requires containerization for safe rollback; continuing without it" >&2
      elif paper_figure_quality_snapshot_workspace \
        "$OUTPUT_DIR" \
        FIGURE_QUALITY_REPAIR_SNAPSHOT_DIR \
        FIGURE_QUALITY_REPAIR_SNAPSHOT_SHA256 \
        "$RESULTS_DIR"; then
        figure_quality_repair_pending=1
        repair_round=$((repair_round + 1))
        echo "[paper.sh] actionable figure-quality warnings found; using repair round $repair_round / $PAPER_MAX_REPAIR_ROUNDS" >&2
        continue
      else
        figure_quality_repair_warning_reported=1
        echo "[paper.sh] could not snapshot the paper for optional figure-quality repair; continuing without it" >&2
      fi
    fi
    if [ "$figure_quality_repair_attempted" -eq 1 ] && [ "$figure_quality_repair_warning_reported" -eq 0 ]; then
      echo "[paper.sh] figure-quality warnings remain after one targeted repair; continuing because they are non-gating" >&2
      figure_quality_repair_warning_reported=1
    elif [ "$figure_quality_repair_attempted" -eq 0 ]; then
      echo "[paper.sh] figure-quality warnings remain but the configured repair budget is exhausted; continuing because they are non-gating" >&2
    fi
  fi

  if [ "$validation_stage_rc" -eq 0 ]; then
    break
  fi

  if [ "$validation_stage_rc" -ne 0 ]; then
    if [ "$validation_failure_repeated" -eq 1 ]; then
      mark_incomplete_paper_pipeline "paper validation errors and inputs were unchanged after repair; see $VALIDATION_FEEDBACK_JSON_PATH"
      break
    fi
    if [ "$repair_round" -lt "$PAPER_MAX_REPAIR_ROUNDS" ]; then
      echo "[paper.sh] paper validation failed; starting repair round $((repair_round + 1)) / $PAPER_MAX_REPAIR_ROUNDS" >&2
      repair_round=$((repair_round + 1))
      continue
    fi
    mark_incomplete_paper_pipeline "paper validation failed; see $VALIDATION_FEEDBACK_JSON_PATH"
  fi
  if [ "$audit_stage_rc" -eq 2 ]; then
    mark_incomplete_paper_pipeline "paper evidence audit reported revision findings; see $EVIDENCE_AUDIT_FEEDBACK_JSON_PATH"
  elif [ "$audit_stage_rc" -eq 1 ]; then
    mark_incomplete_paper_pipeline "paper evidence audit failed operationally; see $EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH"
  fi
  break
done

if [ "$FINAL_PAPER_PIPELINE_INCOMPLETE" -eq 1 ]; then
  paper_gate_start "finalization" "phase=pipeline"
  paper_gate_failed "finalization" "phase=pipeline" "exit_code=2" "reason=incomplete"
  echo "[paper.sh] completed with incomplete paper state: $FINAL_PAPER_PIPELINE_REASON" >&2
  exit 2
fi

paper_gate_start "finalization" "phase=pipeline"
"$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$RESULTS_DIR" "$GLOBAL_MEMORY_DIR" "$VALIDATION_FEEDBACK_JSON_PATH" "$VALIDATION_FEEDBACK_MD_PATH" "$MANIFEST_DIR" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
results_dir = Path(sys.argv[3])
global_memory_dir = Path(sys.argv[4])
validation_feedback_json_path = Path(sys.argv[5])
validation_feedback_md_path = Path(sys.argv[6])
manifest_dir = Path(sys.argv[7]) if sys.argv[7] else None
sys.path.insert(0, str(script_dir))

from src.orchestrator.paper_finalize import finalize_paper_artifacts  # noqa: E402

finalize_paper_artifacts(
    output_dir=output_dir,
    results_dir=results_dir,
    global_memory_dir=global_memory_dir,
    validation_feedback_json_path=validation_feedback_json_path,
    validation_feedback_md_path=validation_feedback_md_path,
    manifest_dir=manifest_dir,
)
PY
paper_gate_complete "finalization" "phase=pipeline" "exit_code=0"
if [ "$CONTAINERIZE" -eq 1 ] && [ -n "${AGENT_CONTAINER_SESSION_NAME:-}" ]; then
  complete_container_session "$AGENT_CONTAINER_SESSION_NAME" || true
fi
