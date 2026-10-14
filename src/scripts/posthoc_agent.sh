#!/bin/bash
# Usage: bash src/scripts/posthoc_agent.sh --agent <agent> [--reasoning-effort low|medium|high|xhigh] --work-dir <path> (--prompt <text> | --prompt-file <path>) [options]

set -euo pipefail

export BASH_DEFAULT_TIMEOUT_MS=600000
export BASH_MAX_TIMEOUT_MS=7200000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/container_agent.sh"
source "$SCRIPT_DIR/lib/task_pathing.sh"

AGENT_LOG_MAX_BYTES=16777216
POSTHOC_AGENT_ALLOWED_ENV_KEYS=""
validate_agent_log_max_bytes "$AGENT_LOG_MAX_BYTES" || exit $?

AGENT="claude"
AGENT_MODEL=""
AGENT_MODEL_NORMALIZED=""
AGENT_PROFILE=""
AGENT_REASONING_EFFORT=""
WORK_DIR=""
PROMPT=""
PROMPT_FILE=""
TASK_SOURCE_DIR=""
MAX_PROGRAM_EVALS=1
RUBRICS_FILE=""
ANALYSIS_ID=""
ATTEMPT_ID=""
STAGE_SKILLS=0
CONTAINERIZE=0
CONTAINER_IMAGE="agentic-scientist/agent-cli:latest"
CONTAINER_NETWORK="bridge"
CONTAINER_AUTH_ROOT="$HOME"
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
    --work-dir|-w)
      WORK_DIR="$2"
      shift 2
      ;;
    --prompt|-p)
      PROMPT="$2"
      shift 2
      ;;
    --prompt-file)
      PROMPT_FILE="$2"
      shift 2
      ;;
    --task-source-dir)
      TASK_SOURCE_DIR="$2"
      shift 2
      ;;
    --max-program-evals)
      MAX_PROGRAM_EVALS="$2"
      shift 2
      ;;
    --rubrics-file)
      RUBRICS_FILE="$2"
      shift 2
      ;;
    --analysis-id)
      ANALYSIS_ID="$2"
      shift 2
      ;;
    --attempt-id)
      ATTEMPT_ID="$2"
      shift 2
      ;;
    --stage-skills)
      STAGE_SKILLS=1
      shift
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
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

validate_agent_reasoning_effort "$AGENT" "$AGENT_REASONING_EFFORT" || exit 1
export AGENT_REASONING_EFFORT
if [ -z "$WORK_DIR" ]; then
  echo "--work-dir is required" >&2
  exit 1
fi
if [ -z "$PROMPT" ] && [ -z "$PROMPT_FILE" ]; then
  echo "--prompt or --prompt-file is required" >&2
  exit 1
fi

if [[ "$WORK_DIR" != /* ]]; then
  WORK_DIR="$REPO_ROOT/$WORK_DIR"
fi
configure_cost_tracking "$WORK_DIR" posthoc
if [ -n "$TASK_SOURCE_DIR" ] && [[ "$TASK_SOURCE_DIR" != /* ]]; then
  TASK_SOURCE_DIR="$REPO_ROOT/$TASK_SOURCE_DIR"
fi
PROMPT_FILE_CHECK=""
if [ -n "$PROMPT_FILE" ]; then
  if [[ "$PROMPT_FILE" = /* ]]; then
    PROMPT_FILE_CHECK="$PROMPT_FILE"
    if [[ "$PROMPT_FILE" == "$WORK_DIR/"* ]]; then
      PROMPT_FILE="${PROMPT_FILE#$WORK_DIR/}"
    fi
  else
    PROMPT_FILE_CHECK="$WORK_DIR/$PROMPT_FILE"
  fi
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
if [ -z "$CONTAINER_ENV_FILE" ] && [ -f "$REPO_ROOT/.env" ]; then
  CONTAINER_ENV_FILE="$REPO_ROOT/.env"
fi
if [ -n "$RUBRICS_FILE" ] && [[ "$RUBRICS_FILE" != /* ]]; then
  RUBRICS_FILE="$REPO_ROOT/$RUBRICS_FILE"
fi

if [ -n "$AGENT_MODEL" ]; then
  AGENT_MODEL_NORMALIZED="$(printf '%s' "$AGENT_MODEL" | tr '[:upper:]' '[:lower:]')"
  case "$AGENT_MODEL_NORMALIZED" in
    none|null) AGENT_MODEL="" ;;
  esac
fi
POSTHOC_PROVIDER_ENV_FILE=""
if [ "$CONTAINERIZE" -eq 1 ]; then
  POSTHOC_PROVIDER_ENV_FILE="$CONTAINER_ENV_FILE"
fi
POSTHOC_AGENT_ALLOWED_ENV_KEYS="$(
  agent_provider_allowed_env_keys "$AGENT" "$AGENT_MODEL_NORMALIZED" "$POSTHOC_PROVIDER_ENV_FILE"
)"
AGENT_ENV_ALLOWLIST_KEYS="$(agent_runtime_allowed_env_keys) $POSTHOC_AGENT_ALLOWED_ENV_KEYS"
if [ -n "$AGENT_PROFILE" ] && [ -z "${AGENT_PROFILE//[[:space:]]/}" ]; then
  echo "--agent-profile must be non-empty when provided" >&2
  exit 1
fi
if [ -n "$AGENT_PROFILE" ] && [ "$AGENT" != "codex" ]; then
  echo "--agent-profile is only supported with --agent codex" >&2
  exit 1
fi
if [ ! -d "$WORK_DIR" ]; then
  echo "Work directory does not exist: $WORK_DIR" >&2
  exit 1
fi
if [ -n "$PROMPT_FILE" ] && [ ! -f "$PROMPT_FILE_CHECK" ]; then
  echo "Prompt file does not exist: $PROMPT_FILE_CHECK" >&2
  exit 1
fi

if [ -n "$TASK_SOURCE_DIR" ]; then
  export TASK_SOURCE_DIR
fi

export MAX_PROGRAM_EVALS_PER_ATTEMPT="$MAX_PROGRAM_EVALS"
if [ -z "$ATTEMPT_ID" ]; then
  ATTEMPT_ID="attempt_$(date +%Y%m%d_%H%M%S)"
fi
export ATTEMPT_ID
export IDEA_ID="${ANALYSIS_ID:-analysis}"
export IDEA_DIR_NAME="${ANALYSIS_ID:-analysis}"
export ATTEMPT_DIR_NAME=""

if [ -n "$RUBRICS_FILE" ] && [ -f "$RUBRICS_FILE" ]; then
  cp "$RUBRICS_FILE" "$WORK_DIR/active_rubrics.json"
else
  printf '%s\n' '[]' > "$WORK_DIR/active_rubrics.json"
fi
if [ -f "$WORK_DIR/analysis_spec.json" ]; then
  cp "$WORK_DIR/analysis_spec.json" "$WORK_DIR/active_idea.json"
elif [ ! -f "$WORK_DIR/active_idea.json" ]; then
  printf '%s\n' '{}' > "$WORK_DIR/active_idea.json"
fi

if [ "$STAGE_SKILLS" -eq 1 ]; then
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
  (
    cd "$WORK_DIR"
    "$PYTHON_BIN" - "$REPO_ROOT" "$FORMATTED_SKILLS_REL_DIR" "$WORK_DIR" "$ATTEMPT_ID" "$IDEA_ID" "$IDEA_DIR_NAME" "$MAX_PROGRAM_EVALS" <<'PY'
import json
import shutil
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
formatted_skills_dir = Path(sys.argv[2])
results_dir = Path(sys.argv[3])
attempt_id = sys.argv[4]
idea_id = sys.argv[5]
idea_dir_name = sys.argv[6]
max_program_evals_per_attempt = sys.argv[7]

sys.path.insert(0, str(script_dir))
from src.investigation.skill_scope import skills_for_step, validate_skill_set
from src.investigation.files import load_instructions, relative_display_path

skills_src = script_dir / "src" / "agents" / "skills"
selected_skills = skills_for_step("run")
validate_skill_set(skills_src, selected_skills)

active_idea = Path("active_idea.json").read_text(encoding="utf-8").strip() or "{}"
active_rubrics = Path("active_rubrics.json").read_text(encoding="utf-8").strip() or "[]"

try:
    json.loads(active_idea)
except json.JSONDecodeError:
    active_idea = "{}"
try:
    json.loads(active_rubrics)
except json.JSONDecodeError:
    active_rubrics = "[]"

task_dir_display = Path(relative_display_path(results_dir / "task_bundle", base_dir=results_dir))
results_dir_display = Path(".")

extra = {
    "REPO_ROOT": str(script_dir),
    "ACTIVE_IDEA_JSON": active_idea,
    "ACTIVE_RUBRICS_JSON": active_rubrics,
    "ACTIVE_IDEA_FILE": "./active_idea.json",
    "ACTIVE_RUBRICS_FILE": "./active_rubrics.json",
    "ATTEMPT_ID": attempt_id,
    "IDEA_ID": idea_id,
    "IDEA_DIR_NAME": idea_dir_name,
    "ATTEMPT_DIR_NAME": "",
    "MAX_PROGRAM_EVALS_PER_ATTEMPT": max_program_evals_per_attempt,
    "GLOBAL_MEMORY_MD_FILE": "./memory.md",
    "GLOBAL_MEMORY_JSONL_FILE": "./memory.jsonl",
    "GLOBAL_MEMORY_RECENT_SUMMARY": "- none",
}

for skill_name in selected_skills:
    src_dir = skills_src / skill_name
    dst_dir = formatted_skills_dir / src_dir.name
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    src_skill = src_dir / "SKILL.md"
    if src_skill.is_file():
        rendered = load_instructions(src_skill, task_dir_display, results_dir_display, extra)
        (dst_dir / "SKILL.md").write_text(rendered, encoding="utf-8")
PY
  )
fi

LOG_FILE="$WORK_DIR/posthoc_agent_${AGENT}_$(date +%Y%m%d_%H%M%S).log"
LOG_PIPE="$WORK_DIR/.posthoc_stream_$(date +%Y%m%d_%H%M%S).fifo"
AGENT_PID=""
TEE_PID=""
INTERRUPTED=0
CLEANUP_IN_PROGRESS=0
SANITIZED_CONTAINER_ENV_FILE=""

cleanup_run_processes() {
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

  if [ "${INTERRUPTED:-0}" -eq 1 ] && [ -f "${LOG_FILE:-}" ]; then
    record_headless_cost_log \
      "$PYTHON_BIN" "$LOG_FILE" analysis \
      "posthoc:${ATTEMPT_ID}:${LOG_FILE##*/}" interrupted
  fi

  if [ "$CONTAINERIZE" -eq 1 ] && [ -n "${AGENT_CONTAINER_SESSION_NAME:-}" ]; then
    stop_container_session_force "$AGENT_CONTAINER_SESSION_NAME" || true
  fi

  if [ -n "${LOG_PIPE:-}" ] && [ -p "$LOG_PIPE" ]; then
    rm -f "$LOG_PIPE"
  fi
  LOG_PIPE=""

  if [ -n "${SANITIZED_CONTAINER_ENV_FILE:-}" ]; then
    rm -f "$SANITIZED_CONTAINER_ENV_FILE"
  fi
  SANITIZED_CONTAINER_ENV_FILE=""
}

on_interrupt() {
  trap - INT TERM
  INTERRUPTED=1
  cleanup_run_processes
  exit 130
}

trap on_interrupt INT TERM
trap cleanup_run_processes EXIT

if [ "$CONTAINERIZE" -eq 1 ]; then
  CONTAINER_HOME_BASE="$WORK_DIR"
  if [ -n "${AGENT_CONTAINER_MOUNT_ROOT:-}" ]; then
    CONTAINER_HOME_BASE="$AGENT_CONTAINER_MOUNT_ROOT"
  fi
  CONTAINER_HOME_DIR="$(prepare_container_home "$CONTAINER_HOME_BASE")"
  seed_auth_into_home "$AGENT" "$CONTAINER_AUTH_ROOT" "$CONTAINER_HOME_DIR"
  if [ -n "$CONTAINER_ENV_FILE" ] && [ -f "$CONTAINER_ENV_FILE" ]; then
    SANITIZED_CONTAINER_ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/posthoc-agent-env-XXXXXX")"
    sanitize_env_file_for_docker \
      "$CONTAINER_ENV_FILE" \
      "$SANITIZED_CONTAINER_ENV_FILE" \
      "$POSTHOC_AGENT_ALLOWED_ENV_KEYS"
    CONTAINER_ENV_FILE="$SANITIZED_CONTAINER_ENV_FILE"
  fi
else
  CONTAINER_HOME_DIR=""
fi

mkfifo "$LOG_PIPE"
bounded_agent_log_writer "$PYTHON_BIN" "$LOG_FILE" "$AGENT_LOG_MAX_BYTES" \
  < "$LOG_PIPE" &
TEE_PID=$!

if [ -n "$AGENT_REASONING_EFFORT" ]; then
  echo "Agent reasoning effort override: $AGENT_REASONING_EFFORT"
fi

(
  AGENT_PROMPT_FILE="$PROMPT_FILE" \
  run_agent_prompt \
    "$AGENT" \
    "$PROMPT" \
    "$AGENT_MODEL" \
    "$AGENT_PROFILE" \
    "$WORK_DIR" \
    "$CONTAINERIZE" \
    "$CONTAINER_IMAGE" \
    "$CONTAINER_NETWORK" \
    "$CONTAINER_HOME_DIR" \
    "$CONTAINER_ENV_FILE"
) > "$LOG_PIPE" 2>&1 &
AGENT_PID=$!

set +e
AGENT_EXIT_CODE=0
TEE_EXIT_CODE=0
if [ -n "${AGENT_PID:-}" ]; then
  wait "$AGENT_PID"
  AGENT_EXIT_CODE=$?
fi
if [ -n "${TEE_PID:-}" ]; then
  wait "$TEE_PID"
  TEE_EXIT_CODE=$?
fi
set -e

record_headless_cost_log \
  "$PYTHON_BIN" "$LOG_FILE" analysis \
  "posthoc:${ATTEMPT_ID}:${LOG_FILE##*/}" \
  "$(headless_cost_status "$AGENT_EXIT_CODE" "$INTERRUPTED")"

trap - INT TERM EXIT

if [ "$INTERRUPTED" -eq 1 ]; then
  exit 130
fi
if [ "$AGENT_EXIT_CODE" -ne 0 ]; then
  exit "$AGENT_EXIT_CODE"
fi
if [ "$TEE_EXIT_CODE" -ne 0 ]; then
  exit "$TEE_EXIT_CODE"
fi
