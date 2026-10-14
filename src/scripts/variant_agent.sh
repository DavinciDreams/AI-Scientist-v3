#!/bin/bash
# Usage: bash src/scripts/variant_agent.sh --agent <agent> [--reasoning-effort low|medium|high|xhigh] --work-dir <path> (--prompt <text> | --prompt-file <path>) [options]
#
# Shell runner for isolated variant authoring and rubric-verification stages.
# Builds on container_agent.sh for containerized execution.

set -euo pipefail

export BASH_DEFAULT_TIMEOUT_MS=600000    # 10 minutes
export BASH_MAX_TIMEOUT_MS=7200000       # 2 hours

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/container_agent.sh"
source "$SCRIPT_DIR/lib/task_pathing.sh"

AGENT_LOG_MAX_BYTES=16777216
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
VENV_DIR=""
MAX_PROGRAM_EVALS=2
RUBRICS_FILE=""
VARIANT_ID=""
ATTEMPT_ID=""
STAGE_SKILLS=0
SKILL_STEP="variant_edit"
CONTAINERIZE=0
CONTAINER_IMAGE="agentic-scientist/agent-cli:latest"
CONTAINER_NETWORK="bridge"
CONTAINER_AUTH_ROOT="$HOME"
CONTAINER_ENV_FILE=""
VARIANT_AGENT_ALLOWED_ENV_KEYS="ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_BEARER_TOKEN_BEDROCK AWS_REGION AWS_DEFAULT_REGION AWS_REGION_NAME CLAUDE_CODE_OAUTH_TOKEN CLAUDE_CODE_USE_BEDROCK CURSOR_API_KEY DEEPSEEK_API_KEY GEMINI_API_KEY GOOGLE_API_KEY MOONSHOT_API_KEY OPENAI_API_KEY OPENROUTER_API_KEY PI_CODING_AGENT_PROVIDER PI_CODING_AGENT_MODEL PI_CODING_AGENT_MODELS S2_API_KEY"
POSTHOC_ENV_POLICY=0

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
    --venv-dir)
      VENV_DIR="$2"
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
    --variant-id)
      VARIANT_ID="$2"
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
    --skill-step)
      SKILL_STEP="$2"
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
      shift 2
      ;;
    --posthoc-env-policy)
      POSTHOC_ENV_POLICY=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# --- validate ---
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

# Resolve paths
if [[ "$WORK_DIR" != /* ]]; then
  WORK_DIR="$REPO_ROOT/$WORK_DIR"
fi
configure_cost_tracking "$WORK_DIR" variants
if [ -n "$TASK_SOURCE_DIR" ] && [[ "$TASK_SOURCE_DIR" != /* ]]; then
  TASK_SOURCE_DIR="$REPO_ROOT/$TASK_SOURCE_DIR"
fi
if [ -n "$VENV_DIR" ] && [[ "$VENV_DIR" != /* ]]; then
  VENV_DIR="$REPO_ROOT/$VENV_DIR"
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
if [ "$POSTHOC_ENV_POLICY" -eq 1 ]; then
  VARIANT_PROVIDER_ENV_FILE=""
  if [ "$CONTAINERIZE" -eq 1 ]; then
    VARIANT_PROVIDER_ENV_FILE="$CONTAINER_ENV_FILE"
  fi
  VARIANT_AGENT_ALLOWED_ENV_KEYS="$(
    agent_provider_allowed_env_keys "$AGENT" "$AGENT_MODEL_NORMALIZED" "$VARIANT_PROVIDER_ENV_FILE"
  )"
  AGENT_ENV_ALLOWLIST_KEYS="$(agent_runtime_allowed_env_keys) $VARIANT_AGENT_ALLOWED_ENV_KEYS"
fi
if [ -n "$AGENT_PROFILE" ] && [ -z "${AGENT_PROFILE//[[:space:]]/}" ]; then
  echo "--agent-profile must be non-empty when provided" >&2
  exit 1
fi
if [ -n "$AGENT_PROFILE" ] && [ "$AGENT" != "codex" ]; then
  echo "--agent-profile is only supported with --agent codex" >&2
  exit 1
fi
case "$SKILL_STEP" in
  run|variant_edit|variant_verify) ;;
  *) echo "Unsupported --skill-step: $SKILL_STEP" >&2; exit 1 ;;
esac
if [ ! -d "$WORK_DIR" ]; then
  echo "Work directory does not exist: $WORK_DIR" >&2
  exit 1
fi
if [ -n "$PROMPT_FILE" ] && [ ! -f "$PROMPT_FILE_CHECK" ]; then
  echo "Prompt file does not exist: $PROMPT_FILE_CHECK" >&2
  exit 1
fi

safe_stage() {
  "$PYTHON_BIN" "$REPO_ROOT/src/investigation/safe_staging.py" "$@"
}
safe_stage_exec_log() {
  exec "$PYTHON_BIN" "$REPO_ROOT/src/investigation/safe_staging.py" exec-log "$@"
}

RETAINED_AGENT_LOG_MAX_BYTES=$((16 * 1024 * 1024))

HOST_RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/variant-agent.XXXXXXXX")"
_cleanup_host_runtime() {
  "$PYTHON_BIN" -c 'import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' \
    "$HOST_RUNTIME_DIR"
}
trap _cleanup_host_runtime EXIT

# Export TASK_SOURCE_DIR so evaluate.py can find the task venv's Python.
if [ -n "$TASK_SOURCE_DIR" ]; then
  export TASK_SOURCE_DIR
fi
# Export VENV_DIR so evaluate.py can prefer a specific venv over TASK_SOURCE_DIR.
if [ -n "$VENV_DIR" ]; then
  export VENV_DIR
fi

# Export variant metadata
export MAX_PROGRAM_EVALS_PER_ATTEMPT="$MAX_PROGRAM_EVALS"
if [ -z "$ATTEMPT_ID" ]; then
  ATTEMPT_ID="variant_${VARIANT_ID:-unknown}_$(date +%Y%m%d_%H%M%S)"
fi
export ATTEMPT_ID
export IDEA_ID="${VARIANT_ID:-variant}"
export IDEA_DIR_NAME="${VARIANT_ID:-variant}"
export ATTEMPT_DIR_NAME=""

# --- Stage rubrics into workspace ---
if [ -n "$RUBRICS_FILE" ] && [ -f "$RUBRICS_FILE" ]; then
  safe_stage copy-file "$WORK_DIR" "active_rubrics.json" "$RUBRICS_FILE"
else
  safe_stage write "$WORK_DIR" "active_rubrics.json" $'[]\n'
fi
# Variant spec serves as the "idea" for the agent
if [ -f "$WORK_DIR/../variant_spec.json" ]; then
  safe_stage copy-file "$WORK_DIR" "active_idea.json" "$WORK_DIR/../variant_spec.json"
elif ! safe_stage read "$WORK_DIR" "active_idea.json" >/dev/null 2>&1; then
  safe_stage write "$WORK_DIR" "active_idea.json" $'{}\n'
else
  safe_stage refresh-file "$WORK_DIR" "active_idea.json"
fi

# --- Stage skills into workspace (same pattern as run.sh) ---
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
  FORMATTED_SKILLS_STAGE_DIR="$HOST_RUNTIME_DIR/skills"
  mkdir "$FORMATTED_SKILLS_STAGE_DIR"
  (
    "$PYTHON_BIN" - "$REPO_ROOT" "$FORMATTED_SKILLS_STAGE_DIR" "$WORK_DIR" "$ATTEMPT_ID" "$IDEA_ID" "$IDEA_DIR_NAME" "" "$MAX_PROGRAM_EVALS" "$SKILL_STEP" <<'PY'
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
attempt_dir_name = sys.argv[7]
max_program_evals_per_attempt = sys.argv[8]
skill_step = sys.argv[9]

sys.path.insert(0, str(script_dir))
from src.investigation.files import extract_markdown_section, load_instructions
from src.investigation.safe_staging import read_text
from src.investigation.skill_scope import skills_for_step, validate_skill_set

skills_src = script_dir / "src" / "agents" / "skills"
if not skills_src.is_dir():
    raise SystemExit(f"Missing skills directory: {skills_src}")
selected_skills = skills_for_step(skill_step)
validate_skill_set(skills_src, selected_skills)

active_idea = read_text(results_dir, "active_idea.json").strip() or "{}"
active_rubrics = read_text(results_dir, "active_rubrics.json").strip() or "[]"

try:
    json.loads(active_idea)
except json.JSONDecodeError:
    active_idea = "{}"
try:
    json.loads(active_rubrics)
except json.JSONDecodeError:
    active_rubrics = "[]"

try:
    task_markdown = read_text(results_dir, "task.md")
except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
    task_goal = "(missing `task.md`)"
    task_rubrics = "(missing `task.md`)"
else:
    task_goal = extract_markdown_section(task_markdown, "Goal") or "(missing Goal section)"
    task_rubrics = (
        extract_markdown_section(task_markdown, "Rubrics")
        or "(missing Rubrics section)"
    )

extra = {
    "REPO_ROOT": str(script_dir),
    "TASK_DIR": ".",
    "RESULTS_DIR": ".",
    "TASK_MD_PATH": "task.md",
    "TASK_GOAL": task_goal,
    "TASK_RUBRICS": task_rubrics,
    "ACTIVE_IDEA_JSON": active_idea,
    "ACTIVE_RUBRICS_JSON": active_rubrics,
    "ACTIVE_IDEA_FILE": "./active_idea.json",
    "ACTIVE_RUBRICS_FILE": "./active_rubrics.json",
    "ATTEMPT_ID": attempt_id,
    "IDEA_ID": idea_id,
    "IDEA_DIR_NAME": idea_dir_name,
    "ATTEMPT_DIR_NAME": attempt_dir_name,
    "MAX_PROGRAM_EVALS_PER_ATTEMPT": max_program_evals_per_attempt,
    "GLOBAL_MEMORY_MD_FILE": "./memory.md",
    "GLOBAL_MEMORY_JSONL_FILE": "./memory.jsonl",
    "GLOBAL_MEMORY_RECENT_SUMMARY": "- none",
}

for skill_name in selected_skills:
    src_dir = skills_src / skill_name
    dst_dir = formatted_skills_dir / src_dir.name
    shutil.copytree(src_dir, dst_dir)

    src_skill = src_dir / "SKILL.md"
    if src_skill.is_file():
        rendered = load_instructions(
            src_skill, formatted_skills_dir, formatted_skills_dir, extra
        )
        (dst_dir / "SKILL.md").write_text(rendered, encoding="utf-8")
PY
  )
  safe_stage copy-tree "$WORK_DIR" "$FORMATTED_SKILLS_REL_DIR" "$FORMATTED_SKILLS_STAGE_DIR"
fi

# --- prepare container home ---
CONTAINER_HOME_DIR=""
CONTAINER_ENV_FILE_EFFECTIVE=""
if [ "$CONTAINERIZE" -eq 1 ]; then
  if [ -n "${AGENT_CONTAINER_SESSION_NAME:-}" ]; then
    CONTAINER_HOME_DIR="$(
      "$PYTHON_BIN" - "$AGENT_CONTAINER_SESSION_NAME" <<'PY'
import hashlib
import os
import sys
import tempfile
from pathlib import Path

key = sys.argv[1].encode()
default_root = Path(tempfile.gettempdir()) / f"variant-agent-session-homes-{os.getuid()}"
root = Path(os.environ.get("AGENT_VARIANT_SESSION_HOME_ROOT", default_root))
if not root.is_absolute():
    raise SystemExit(f"Container-home root must be absolute: {root}")
root.mkdir(mode=0o700, exist_ok=True)
if root.is_symlink() or not root.is_dir() or root.stat().st_uid != os.getuid():
    raise SystemExit(f"Unsafe container-home root: {root}")
root.chmod(0o700)
home = root / hashlib.sha256(key).hexdigest()
home.mkdir(mode=0o700, exist_ok=True)
if home.is_symlink() or not home.is_dir() or home.stat().st_uid != os.getuid():
    raise SystemExit(f"Unsafe container home: {home}")
home.chmod(0o700)
print(home)
PY
    )"
  else
    CONTAINER_HOME_DIR="$HOST_RUNTIME_DIR/home"
    mkdir "$CONTAINER_HOME_DIR"
  fi
  while IFS= read -r seed_path; do
    [ -n "$seed_path" ] || continue
    if chatgpt_subscription_enabled; then
      case "$AGENT:$seed_path" in
        pi:.pi/agent/auth.json|opencode:.config/opencode) continue ;;
      esac
    fi
    seed_source="$CONTAINER_AUTH_ROOT/$seed_path"
    if [ -f "$seed_source" ]; then
      safe_stage copy-file "$CONTAINER_HOME_DIR" "$seed_path" "$seed_source" \
        --allow-source-symlinks
    elif [ -d "$seed_source" ]; then
      safe_stage copy-tree "$CONTAINER_HOME_DIR" "$seed_path" "$seed_source" \
        --allow-source-symlinks
    else
      safe_stage remove-path "$CONTAINER_HOME_DIR" "$seed_path"
    fi
  done < <(seed_paths_for_agent "$AGENT")
  case "$AGENT" in
    pi) seed_managed_agent_files "$AGENT" "$CONTAINER_AUTH_ROOT" "$CONTAINER_HOME_DIR" ;;
    opencode)
      if chatgpt_subscription_enabled; then
        seed_managed_agent_files "$AGENT" "$CONTAINER_AUTH_ROOT" "$CONTAINER_HOME_DIR"
      fi
      ;;
  esac
  safe_stage ensure-directory "$CONTAINER_HOME_DIR" "$(agent_config_rel_dir "$AGENT")"

  if [ -n "$CONTAINER_ENV_FILE" ] && [ -f "$CONTAINER_ENV_FILE" ]; then
    CONTAINER_ENV_FILE_EFFECTIVE="$HOST_RUNTIME_DIR/container.env"
    sanitize_env_file_for_docker \
      "$CONTAINER_ENV_FILE" \
      "$CONTAINER_ENV_FILE_EFFECTIVE" \
      "$VARIANT_AGENT_ALLOWED_ENV_KEYS"
  fi
fi

# --- run agent ---
cd "$WORK_DIR"
safe_stage remove-path "$WORK_DIR" "variant_agent_failure.jsonl"

if [ -n "$AGENT_REASONING_EFFORT" ]; then
  echo "Agent reasoning effort override: $AGENT_REASONING_EFFORT"
fi

VARIANT_COST_TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/agentic-variant-cost-XXXXXX")"
VARIANT_AGENT_LOG="$VARIANT_COST_TEMP_DIR/agent_${AGENT}.jsonl"
VARIANT_RETAINED_LOG=""
VARIANT_LOG_PIPE="$VARIANT_COST_TEMP_DIR/agent.fifo"
VARIANT_COST_RECORDED=0
VARIANT_AGENT_PID=""
VARIANT_WRITER_PID=""

cleanup_variant_cost() {
  if [ -n "$VARIANT_AGENT_PID" ] && kill -0 "$VARIANT_AGENT_PID" 2>/dev/null; then
    terminate_pid_tree "$VARIANT_AGENT_PID" 0.2
    wait "$VARIANT_AGENT_PID" 2>/dev/null || true
  fi
  if [ -n "$VARIANT_WRITER_PID" ] && kill -0 "$VARIANT_WRITER_PID" 2>/dev/null; then
    kill -TERM "$VARIANT_WRITER_PID" 2>/dev/null || true
    wait "$VARIANT_WRITER_PID" 2>/dev/null || true
  fi
  if [ "$VARIANT_COST_RECORDED" -eq 0 ] && [ -f "$VARIANT_AGENT_LOG" ]; then
    record_headless_cost_log \
      "$PYTHON_BIN" "$VARIANT_AGENT_LOG" "$SKILL_STEP" \
      "variant:${SKILL_STEP}:${VARIANT_COST_TEMP_DIR##*/}" interrupted
  fi
  rm -f -- "$VARIANT_LOG_PIPE"
  rm -f -- "$VARIANT_AGENT_LOG"
  rm -f -- "$VARIANT_RETAINED_LOG"
  rmdir -- "$VARIANT_COST_TEMP_DIR" 2>/dev/null || true
}

trap 'cleanup_variant_cost; _cleanup_host_runtime' EXIT
trap 'exit 130' INT TERM
mkfifo "$VARIANT_LOG_PIPE"
bounded_agent_log_writer \
  "$PYTHON_BIN" "$VARIANT_AGENT_LOG" "$AGENT_LOG_MAX_BYTES" 1 \
  < "$VARIANT_LOG_PIPE" &
VARIANT_WRITER_PID=$!
set +e
( AGENT_PROMPT_FILE="$PROMPT_FILE" run_agent_prompt \
    "$AGENT" \
    "$PROMPT" \
    "$AGENT_MODEL" \
    "$AGENT_PROFILE" \
    "$WORK_DIR" \
    "$CONTAINERIZE" \
    "$CONTAINER_IMAGE" \
    "$CONTAINER_NETWORK" \
    "$CONTAINER_HOME_DIR" \
    "$CONTAINER_ENV_FILE_EFFECTIVE" \
    "$WORK_DIR" ) > "$VARIANT_LOG_PIPE" 2>&1 &
VARIANT_AGENT_PID=$!
wait "$VARIANT_AGENT_PID"
VARIANT_AGENT_EXIT_CODE=$?
VARIANT_AGENT_PID=""
wait "$VARIANT_WRITER_PID"
VARIANT_LOG_EXIT_CODE=$?
VARIANT_WRITER_PID=""
set -e
record_headless_cost_log \
  "$PYTHON_BIN" "$VARIANT_AGENT_LOG" "$SKILL_STEP" \
  "variant:${SKILL_STEP}:${VARIANT_COST_TEMP_DIR##*/}" \
  "$(headless_cost_status "$VARIANT_AGENT_EXIT_CODE")"
VARIANT_COST_RECORDED=1
if [ "$VARIANT_AGENT_EXIT_CODE" -ne 0 ] && [ "$VARIANT_AGENT_EXIT_CODE" -ne 130 ]; then
  VARIANT_RETAINED_LOG="$VARIANT_COST_TEMP_DIR/agent_failure.jsonl"
  if ! bounded_agent_log_writer \
    "$PYTHON_BIN" "$VARIANT_RETAINED_LOG" "$RETAINED_AGENT_LOG_MAX_BYTES" 1 \
    < "$VARIANT_AGENT_LOG"; then
    echo "[variants] failed to bound agent failure log" >&2
  elif ! safe_stage copy-file "$WORK_DIR" "variant_agent_failure.jsonl" "$VARIANT_RETAINED_LOG" \
    --maximum-bytes "$RETAINED_AGENT_LOG_MAX_BYTES" --mode 600; then
    echo "[variants] failed to retain agent failure log" >&2
  fi
fi
if [ "$VARIANT_LOG_EXIT_CODE" -ne 0 ]; then
  exit "$VARIANT_LOG_EXIT_CODE"
fi
exit "$VARIANT_AGENT_EXIT_CODE"
