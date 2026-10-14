#!/bin/bash

# Shared synchronous concept-figure orchestration for paper creation and edits.

CONCEPT_FIGURE_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

parse_concept_figure_seconds() {
  local raw="$1"
  if [[ ! "$raw" =~ ^[0-9]+$ ]] || [ "${#raw}" -gt 9 ]; then
    return 1
  fi
  printf '%d\n' "$((10#$raw))"
}

positive_concept_figure_seconds() {
  local parsed
  if parsed="$(parse_concept_figure_seconds "$1")" && [ "$parsed" -gt 0 ]; then
    printf '%s\n' "$parsed"
  else
    printf '%s\n' "$2"
  fi
}

# Autoillustrate's --mode presets (light | medium | full) trade runtime for
# quality; only full runs the panel-inspect and verify/repair loops. The paper
# workflow asks for full and steps down to medium or light when the remaining
# workflow deadline cannot afford the preset's minimum budget.
concept_figure_preferred_mode() {
  local mode
  mode="$(concept_figure_setting PAPER_CONCEPT_FIGURE_MODE full)"
  case "$mode" in
    light|medium|full) printf '%s\n' "$mode" ;;
    *)
      echo "[${CONCEPT_FIGURE_LOG_PREFIX:-paper.sh}] invalid PAPER_CONCEPT_FIGURE_MODE '$mode'; using full" >&2
      printf 'full\n'
      ;;
  esac
}

select_concept_figure_mode() {
  local preferred_mode="$1"
  local budget_seconds="$2"
  local full_mode_seconds medium_mode_seconds
  full_mode_seconds="$(
    positive_concept_figure_seconds \
      "${PAPER_CONCEPT_FIGURE_FULL_MODE_MIN_SECONDS:-3600}" 3600
  )"
  medium_mode_seconds="$(
    positive_concept_figure_seconds \
      "${PAPER_CONCEPT_FIGURE_MEDIUM_MODE_MIN_SECONDS:-1800}" 1800
  )"
  if [ "$preferred_mode" = "full" ] && [ "$budget_seconds" -ge "$full_mode_seconds" ]; then
    printf 'full\n'
  elif [ "$preferred_mode" != "light" ] \
    && [ "$budget_seconds" -ge "$medium_mode_seconds" ]; then
    printf 'medium\n'
  else
    printf 'light\n'
  fi
}

configure_concept_figure_budget() {
  local elapsed_seconds="${1:-${SECONDS:-0}}"
  CONCEPT_FIGURE_TIMEOUT_SECONDS="$(
    positive_concept_figure_seconds \
      "${PAPER_CONCEPT_FIGURE_TIMEOUT_SECONDS:-9000}" 9000
  )"
  CONCEPT_FIGURE_KILL_GRACE_SECONDS="$(
    positive_concept_figure_seconds \
      "${PAPER_CONCEPT_FIGURE_KILL_GRACE_SECONDS:-60}" 60
  )"
  CONCEPT_FIGURE_MODE_FLAGS=""
  CONCEPT_FIGURE_UNAVAILABLE_REASON="concept-figure private state unavailable"
  local preferred_mode
  preferred_mode="$(concept_figure_preferred_mode)"

  local workflow_timeout
  if workflow_timeout="$(
    parse_concept_figure_seconds "${PAPER_WORKFLOW_TIMEOUT_SECONDS:-}"
  )" && [ "$workflow_timeout" -gt 0 ]; then
    if ! elapsed_seconds="$(parse_concept_figure_seconds "$elapsed_seconds")"; then
      elapsed_seconds=0
    fi

    local reserve_seconds light_mode_seconds remaining_seconds
    reserve_seconds="$(
      positive_concept_figure_seconds \
        "${PAPER_CONCEPT_FIGURE_POST_RENDER_RESERVE_SECONDS:-300}" 300
    )"
    light_mode_seconds="$(
      positive_concept_figure_seconds \
        "${PAPER_CONCEPT_FIGURE_LIGHT_MODE_MIN_SECONDS:-300}" 300
    )"
    remaining_seconds=$((
      workflow_timeout - elapsed_seconds - reserve_seconds - CONCEPT_FIGURE_KILL_GRACE_SECONDS
    ))
    if [ "$remaining_seconds" -le 0 ]; then
      CONCEPT_FIGURE_UNAVAILABLE_REASON="concept-figure workflow deadline budget exhausted"
      return 1
    fi
    if [ "$CONCEPT_FIGURE_TIMEOUT_SECONDS" -gt "$remaining_seconds" ]; then
      CONCEPT_FIGURE_TIMEOUT_SECONDS="$remaining_seconds"
    fi
    if [ "$CONCEPT_FIGURE_TIMEOUT_SECONDS" -lt "$light_mode_seconds" ]; then
      CONCEPT_FIGURE_UNAVAILABLE_REASON="concept-figure workflow deadline budget exhausted"
      return 1
    fi
  fi
  CONCEPT_FIGURE_MODE_FLAGS="--mode $(
    select_concept_figure_mode "$preferred_mode" "$CONCEPT_FIGURE_TIMEOUT_SECONDS"
  )"
}

initialize_concept_figure_step() {
  CONCEPT_FIGURE_LOG_PREFIX="${1:-paper.sh}"
  CONCEPT_FIGURE_RUN_LOG="$OUTPUT_DIR/concept_figure_run.log"
  CONCEPT_FIGURE_MARKER_PATH="$OUTPUT_DIR/.concept_figure_complete"
  CONCEPT_FIGURE_ATTEMPTED=0
  CONCEPT_FIGURE_REQUEST_FINGERPRINT=""
  CONCEPT_FIGURE_FAILURE_ACTIVE=0
  CONCEPT_FIGURE_FAILURE_ID=""
  CONCEPT_FIGURE_FAILURE_REASON=""
  CONCEPT_FIGURE_REPAIR_AUTHORIZED=0
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_FIGURE_ID=""
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINGERPRINT=""
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_AUDIT_SHA256=""
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINDING_IDS=""
  CONCEPT_FIGURE_FALLBACK_ACTIVE=0
  CONCEPT_FIGURE_FALLBACK_FIGURE_ID=""
  CONCEPT_FIGURE_FALLBACK_FINGERPRINT=""
  CONCEPT_FIGURE_FALLBACK_AUDIT_SHA256=""
  CONCEPT_FIGURE_FALLBACK_FINDING_IDS=""
  CONCEPT_FIGURE_AVAILABLE=0
  CONCEPT_FIGURE_PRESERVE_SNAPSHOT=""
  CONCEPT_FIGURE_RENDERED_DIR=""
  CONCEPT_FIGURE_SCOPED_ENV_FILE=""
  CONCEPT_FIGURE_CONTAINER_HOME_DIR=""
  CONCEPT_FIGURE_RENDER_WORK_DIR=""
  if [ -d "$OUTPUT_DIR/figures" ] \
    && [ ! -L "$OUTPUT_DIR/figures" ] \
    && [ -n "${PYTHON_BIN:-}" ] \
    && [ -n "${REPO_ROOT:-}" ] \
    && ! recover_concept_figure_outputs; then
    concept_figure_warn "could not recover interrupted concept-figure publication; feature disabled"
    return 0
  fi
  if ! configure_concept_figure_budget; then
    concept_figure_warn "concept-figure deadline budget exhausted; feature disabled"
    return 0
  fi
  if ! CONCEPT_FIGURE_PRESERVE_SNAPSHOT="$(capture_concept_figure_preserve_snapshot)"; then
    CONCEPT_FIGURE_PRESERVE_SNAPSHOT=""
    concept_figure_warn "could not capture trusted concept-figure preserve snapshot"
  fi
  local rendered_dir
  if ! rendered_dir="$(mktemp -d "${TMPDIR:-/tmp}/agentic-concept-figure-XXXXXX")"; then
    concept_figure_warn "could not create private concept-figure state; feature disabled"
    return 0
  fi
  CONCEPT_FIGURE_RENDERED_DIR="$rendered_dir"
  if ! chmod 700 "$CONCEPT_FIGURE_RENDERED_DIR"; then
    concept_figure_warn "could not secure private concept-figure state; feature disabled"
    cleanup_concept_figure_state
    return 0
  fi
  CONCEPT_FIGURE_AVAILABLE=1
}

reset_concept_figure_attempt_state() {
  CONCEPT_FIGURE_ATTEMPTED=0
  CONCEPT_FIGURE_REQUEST_FINGERPRINT=""
  CONCEPT_FIGURE_FAILURE_ACTIVE=0
  CONCEPT_FIGURE_FAILURE_ID=""
  CONCEPT_FIGURE_FAILURE_REASON=""
  CONCEPT_FIGURE_REPAIR_AUTHORIZED=0
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_FIGURE_ID=""
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINGERPRINT=""
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_AUDIT_SHA256=""
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINDING_IDS=""
  CONCEPT_FIGURE_FALLBACK_ACTIVE=0
  CONCEPT_FIGURE_FALLBACK_FIGURE_ID=""
  CONCEPT_FIGURE_FALLBACK_FINGERPRINT=""
  CONCEPT_FIGURE_FALLBACK_AUDIT_SHA256=""
  CONCEPT_FIGURE_FALLBACK_FINDING_IDS=""
  rm -f -- "$CONCEPT_FIGURE_MARKER_PATH"
  if [ -n "${CONCEPT_FIGURE_RENDERED_DIR:-}" ]; then
    rm -f -- \
      "$CONCEPT_FIGURE_RENDERED_DIR/figure.png" \
      "$CONCEPT_FIGURE_RENDERED_DIR/figure.prompts.json" \
      "$CONCEPT_FIGURE_RENDERED_DIR/output_rel.txt"
  fi
}

concept_figure_log() {
  echo "[$CONCEPT_FIGURE_LOG_PREFIX] $*"
}

concept_figure_warn() {
  echo "[$CONCEPT_FIGURE_LOG_PREFIX] $*" >&2
}

cleanup_concept_figure_state() {
  if [ -n "${OUTPUT_DIR:-}" ] \
    && [ -e "$OUTPUT_DIR/figures/.concept-figure-transaction.json" ] \
    && [ -n "${PYTHON_BIN:-}" ] \
    && [ -n "${REPO_ROOT:-}" ]; then
    recover_concept_figure_outputs \
      || concept_figure_warn "could not reconcile interrupted concept-figure publication"
  fi
  cleanup_concept_figure_scoped_env
  cleanup_concept_figure_container_home
  cleanup_concept_figure_render_workspace
  local state_dir="${CONCEPT_FIGURE_RENDERED_DIR:-}"
  if [[ "$state_dir" == "${TMPDIR:-/tmp}/agentic-concept-figure-"* ]] && [ -d "$state_dir" ]; then
    rm -rf -- "$state_dir"
  fi
  CONCEPT_FIGURE_RENDERED_DIR=""
}

cleanup_concept_figure_render_workspace() {
  local work_dir="${CONCEPT_FIGURE_RENDER_WORK_DIR:-}"
  if [[ "$work_dir" == "${TMPDIR:-/tmp}/agentic-concept-work-"* ]] \
    && [ -d "$work_dir" ]; then
    if ! rm -rf -- "$work_dir"; then
      chmod -R u+rwX -- "$work_dir" 2>/dev/null || true
      rm -rf -- "$work_dir" \
        || concept_figure_warn "could not remove staged concept-figure workspace: $work_dir"
    fi
  fi
  CONCEPT_FIGURE_RENDER_WORK_DIR=""
}

prepare_concept_figure_render_workspace() {
  cleanup_concept_figure_render_workspace
  local work_dir
  work_dir="$(mktemp -d "${TMPDIR:-/tmp}/agentic-concept-work-XXXXXX")" \
    || return 1
  CONCEPT_FIGURE_RENDER_WORK_DIR="$work_dir"
  if ! chmod 700 "$work_dir" || ! mkdir "$work_dir/figures"; then
    cleanup_concept_figure_render_workspace
    return 1
  fi
  if ! "$PYTHON_BIN" - "$OUTPUT_DIR/paper.tex" "$work_dir/paper.tex" <<'PY'
import os
import shutil
import stat
import sys

source_path, destination_path = sys.argv[1:]
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(source_path, flags)
with os.fdopen(descriptor, "rb") as source:
    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
        raise RuntimeError("paper.tex must be a regular file")
    with open(destination_path, "xb") as destination:
        shutil.copyfileobj(source, destination)
PY
  then
    cleanup_concept_figure_render_workspace
    return 1
  fi
}

publish_concept_figure_outputs() {
  "$PYTHON_BIN" - \
    "$REPO_ROOT" "$OUTPUT_DIR" "$CONCEPT_FIGURE_RENDER_WORK_DIR" "$1" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
render_dir = Path(sys.argv[3])
figure_id = sys.argv[4]
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import publish_concept_figure_artifacts  # noqa: E402

publish_concept_figure_artifacts(paper_dir, render_dir, figure_id)
PY
}

commit_concept_figure_outputs() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import commit_concept_figure_artifact_publication  # noqa: E402

commit_concept_figure_artifact_publication(paper_dir)
PY
}

rollback_concept_figure_outputs() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import rollback_concept_figure_artifact_publication  # noqa: E402

rollback_concept_figure_artifact_publication(paper_dir)
PY
}

recover_concept_figure_outputs() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import recover_concept_figure_artifact_publication  # noqa: E402

recover_concept_figure_artifact_publication(paper_dir)
PY
}

publish_stashed_concept_figure_outputs() {
  "$PYTHON_BIN" - \
    "$REPO_ROOT" "$OUTPUT_DIR" "$CONCEPT_FIGURE_RENDERED_DIR" "$1" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
state_dir = Path(sys.argv[3])
figure_id = sys.argv[4]
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import publish_stashed_concept_figure_artifacts  # noqa: E402

publish_stashed_concept_figure_artifacts(paper_dir, state_dir, figure_id)
PY
}

concept_figure_provenance_matches() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$1" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
figure_id = sys.argv[3]
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import concept_figure_provenance_matches  # noqa: E402

raise SystemExit(0 if concept_figure_provenance_matches(paper_dir, figure_id) else 1)
PY
}

clear_concept_figure_provenance() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$1" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
figure_id = sys.argv[3]
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import clear_concept_figure_provenance  # noqa: E402

clear_concept_figure_provenance(paper_dir, figure_id)
PY
}

cleanup_concept_figure_container_home() {
  local home_dir="${CONCEPT_FIGURE_CONTAINER_HOME_DIR:-}"
  if [[ "$home_dir" == "${TMPDIR:-/tmp}/agentic-concept-home-"* ]] \
    && [ -d "$home_dir" ]; then
    if ! rm -rf -- "$home_dir"; then
      chmod -R u+rwX -- "$home_dir" 2>/dev/null || true
      rm -rf -- "$home_dir" \
        || concept_figure_warn "could not remove isolated concept-figure home: $home_dir"
    fi
  fi
  CONCEPT_FIGURE_CONTAINER_HOME_DIR=""
}

cleanup_concept_figure_scoped_env() {
  local env_file="${CONCEPT_FIGURE_SCOPED_ENV_FILE:-}"
  if [[ "$env_file" == "${TMPDIR:-/tmp}/agentic-concept-env-"* ]]; then
    rm -f -- "$env_file"
  fi
  CONCEPT_FIGURE_SCOPED_ENV_FILE=""
}

prepare_concept_figure_container_env() {
  cleanup_concept_figure_scoped_env
  if [ -z "$CONTAINER_ENV_FILE" ]; then
    return 0
  fi

  local env_file
  local subscription_extra_allowed_keys=""
  if [ "$AGENT" = "codex" ]; then
    subscription_extra_allowed_keys="HEADLESS_CAPACITY_RETRIES"
  fi
  env_file="$(mktemp "${TMPDIR:-/tmp}/agentic-concept-env-XXXXXX")" || return 1
  CONCEPT_FIGURE_SCOPED_ENV_FILE="$env_file"
  if ! sanitize_env_file_for_docker \
    "$CONTAINER_ENV_FILE" "$env_file" "$CONCEPT_FIGURE_ALLOWED_ENV_KEYS" \
    "$subscription_extra_allowed_keys" \
    || ! chmod 600 "$env_file"; then
    cleanup_concept_figure_scoped_env
    return 1
  fi
}

prepare_concept_figure_sandbox() {
  cleanup_concept_figure_container_home
  command -v docker >/dev/null 2>&1 || return 1
  [ -d "$CONTAINER_AUTH_ROOT" ] || return 1

  local home_dir
  home_dir="$(mktemp -d "${TMPDIR:-/tmp}/agentic-concept-home-XXXXXX")" \
    || return 1
  CONCEPT_FIGURE_CONTAINER_HOME_DIR="$home_dir"
  if ! chmod 700 "$home_dir"; then
    cleanup_concept_figure_container_home
    return 1
  fi
  if [ "${CONCEPT_FIGURE_SEED_AGENT_AUTH:-1}" -eq 1 ]; then
    if ! seed_concept_figure_auth "$AGENT" "$CONTAINER_AUTH_ROOT" "$home_dir"; then
      cleanup_concept_figure_container_home
      return 1
    fi
  elif ! mkdir -p "$home_dir/$(agent_config_rel_dir "$AGENT")"; then
    cleanup_concept_figure_container_home
    return 1
  fi
  if ! prepare_concept_figure_container_env; then
    cleanup_concept_figure_container_home
    cleanup_concept_figure_scoped_env
    return 1
  fi
}

concept_figure_auth_paths() {
  case "$1" in
    codex) printf '%s\n' ".codex/auth.json" ;;
    claude) printf '%s\n' ".claude/.credentials.json" ".claude/auth.json" ;;
    cursor) printf '%s\n' ".cursor/cli-config.json" ;;
    gemini)
      printf '%s\n' \
        ".gemini/google_accounts.json" \
        ".gemini/state.json" \
        ".gemini/installation_id"
      ;;
    *) return 1 ;;
  esac
}

seed_concept_figure_auth() {
  local agent="$1"
  local auth_root="$2"
  local home_dir="$3"
  local rel source destination
  mkdir -p "$home_dir/$(agent_config_rel_dir "$agent")" || return 1
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    source="$auth_root/$rel"
    [ -f "$source" ] && [ ! -L "$source" ] || continue
    destination="$home_dir/$rel"
    mkdir -p "$(dirname "$destination")" || return 1
    cp "$source" "$destination" || return 1
    chmod 600 "$destination" || return 1
  done < <(concept_figure_auth_paths "$agent")

  if [ "$agent" = "codex" ] && [ -n "${AGENT_PROFILE:-}" ]; then
    "${AGENT_PYTHON_BIN:-python3}" "$CONCEPT_FIGURE_LIB_DIR/stage_codex_profile.py" \
      "$auth_root/.codex" "$home_dir/.codex" "$AGENT_PROFILE" || return 1
  fi
}

is_safe_concept_output_rel() {
  local output_rel="$1"
  [[ "$output_rel" =~ ^figures/[A-Za-z0-9][A-Za-z0-9_-]{0,127}\.png$ ]]
}

is_safe_concept_artifact_rel() {
  local artifact_rel="$1"
  [[ "$artifact_rel" =~ ^figures/[A-Za-z0-9][A-Za-z0-9_-]{0,127}\.(png|prompts\.json)$ ]]
}

is_safe_workspace_figure() {
  local output_rel="$1"
  is_safe_concept_output_rel "$output_rel" \
    && [ -d "$OUTPUT_DIR/figures" ] \
    && [ ! -L "$OUTPUT_DIR/figures" ] \
    && [ -f "$OUTPUT_DIR/$output_rel" ] \
    && [ ! -L "$OUTPUT_DIR/$output_rel" ]
}

is_safe_concept_render_destination() {
  local output_rel="$1"
  is_safe_concept_output_rel "$output_rel" || return 1
  [ -d "$OUTPUT_DIR/figures" ] && [ ! -L "$OUTPUT_DIR/figures" ] || return 1

  local prompts_rel="${output_rel%.png}.prompts.json"
  local artifact_rel artifact_path
  for artifact_rel in "$output_rel" "$prompts_rel"; do
    artifact_path="$OUTPUT_DIR/$artifact_rel"
    [ ! -L "$artifact_path" ] || return 1
    if [ -e "$artifact_path" ] && [ ! -f "$artifact_path" ]; then
      return 1
    fi
  done
}

atomic_replace_path() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import os
import sys

os.replace(sys.argv[1], sys.argv[2])
PY
}

publish_concept_marker() {
  if [ -e "$CONCEPT_FIGURE_MARKER_PATH" ] && [ ! -f "$CONCEPT_FIGURE_MARKER_PATH" ] \
    && [ ! -L "$CONCEPT_FIGURE_MARKER_PATH" ]; then
    concept_figure_warn "refusing to replace non-file marker path: $CONCEPT_FIGURE_MARKER_PATH"
    return 1
  fi
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import publish_concept_figure_marker  # noqa: E402

publish_concept_figure_marker(paper_dir)
PY
}

add_concept_figure_secret_key() {
  case " ${CONCEPT_FIGURE_SECRET_KEYS} " in
    *" $1 "*) ;;
    *) CONCEPT_FIGURE_SECRET_KEYS="${CONCEPT_FIGURE_SECRET_KEYS:+${CONCEPT_FIGURE_SECRET_KEYS} }$1" ;;
  esac
}

add_concept_figure_runtime_key() {
  case " ${CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS} " in
    *" $1 "*) ;;
    *) CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS="${CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS:+${CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS} }$1" ;;
  esac
}

concept_figure_setting() {
  local key="$1"
  local fallback="${2:-}"
  local value="${!key-}"
  if [ -z "$value" ] && [ -n "${CONTAINER_ENV_FILE:-}" ]; then
    value="$(env_file_value "$CONTAINER_ENV_FILE" "$key" 2>/dev/null || true)"
  fi
  printf '%s\n' "${value:-$fallback}"
}

concept_figure_setting_is_true() {
  case "$(concept_figure_setting "$1")" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
  esac
  return 1
}

add_concept_figure_aws_credentials() {
  add_concept_figure_secret_key "AWS_ACCESS_KEY_ID"
  add_concept_figure_secret_key "AWS_SECRET_ACCESS_KEY"
  add_concept_figure_secret_key "AWS_SESSION_TOKEN"
  add_concept_figure_secret_key "AWS_BEARER_TOKEN_BEDROCK"
  add_concept_figure_runtime_key "AWS_REGION"
  add_concept_figure_runtime_key "AWS_DEFAULT_REGION"
}

concept_figure_secret_available() {
  local key="$1"
  local availability_flag="$2"
  [ "$availability_flag" = "true" ] \
    || { [ -n "${CONTAINER_ENV_FILE:-}" ] \
      && env_file_has_nonempty_key "$CONTAINER_ENV_FILE" "$key"; }
}

select_concept_figure_backend() {
  CONCEPT_FIGURE_BACKEND_FLAGS=""
  CONCEPT_FIGURE_SECRET_KEYS=""
  CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS=""
  CONCEPT_FIGURE_SEED_AGENT_AUTH=1
  if concept_figure_secret_available \
    OPENAI_API_KEY "${PAPER_OPENAI_API_KEY_AVAILABLE:-false}"; then
    CONCEPT_FIGURE_BACKEND_FLAGS="--enhance-model gpt-image-2 --audit-vlm gpt-5.5 --unit-test-vlm openai-single"
    add_concept_figure_secret_key "OPENAI_API_KEY"
    concept_figure_log "using OpenAI for concept-figure enhancement and audits"
  elif concept_figure_secret_available \
    GEMINI_API_KEY "${PAPER_GEMINI_API_KEY_AVAILABLE:-false}"; then
    CONCEPT_FIGURE_BACKEND_FLAGS="--enhance-model nanobanana-pro --audit-vlm gemini-3.1-pro --unit-test-vlm gemini-single"
    add_concept_figure_secret_key "GEMINI_API_KEY"
    concept_figure_log "OPENAI_API_KEY unavailable; using Gemini for concept-figure enhancement and audits"
  else
    CONCEPT_FIGURE_BACKEND_FLAGS="--skip-enhance --panel-inspects 0"
    concept_figure_log "no concept-figure VLM key available; using deterministic export (--skip-enhance)"
  fi

  case "$AGENT" in
    claude)
      add_concept_figure_runtime_key "HEADLESS_BILLING"
      add_concept_figure_aws_credentials
      add_concept_figure_runtime_key "CLAUDE_CODE_USE_BEDROCK"
      add_concept_figure_secret_key "ANTHROPIC_API_KEY"
      add_concept_figure_secret_key "ANTHROPIC_AUTH_TOKEN"
      add_concept_figure_secret_key "CLAUDE_CODE_OAUTH_TOKEN"
      ;;
    cursor)
      add_concept_figure_secret_key "CURSOR_API_KEY"
      add_concept_figure_runtime_key "CURSOR_CLI_BIN"
      ;;
    codex)
      add_concept_figure_runtime_key "HEADLESS_BILLING"
      add_concept_figure_runtime_key "HEADLESS_CAPACITY_RETRIES"
      add_concept_figure_secret_key "CODEX_API_KEY"
      add_concept_figure_secret_key "OPENAI_API_KEY"
      ;;
    gemini) add_concept_figure_secret_key "GEMINI_API_KEY" ;;
    opencode)
      CONCEPT_FIGURE_SEED_AGENT_AUTH=0
      case "$(concept_figure_setting PAPER_CONCEPT_FIGURE_AGENT_PROVIDER)" in
        anthropic) add_concept_figure_secret_key "ANTHROPIC_API_KEY" ;;
        gemini|google) add_concept_figure_secret_key "GEMINI_API_KEY" ;;
        openai) add_concept_figure_secret_key "OPENAI_API_KEY" ;;
      esac
      ;;
    pi)
      CONCEPT_FIGURE_SEED_AGENT_AUTH=0
      add_concept_figure_runtime_key "PI_CODING_AGENT_PROVIDER"
      add_concept_figure_runtime_key "PI_CODING_AGENT_MODEL"
      add_concept_figure_runtime_key "PI_CODING_AGENT_MODELS"
      case "$(concept_figure_setting PI_CODING_AGENT_PROVIDER amazon-bedrock)" in
        amazon-bedrock|bedrock) add_concept_figure_aws_credentials ;;
        openai) add_concept_figure_secret_key "OPENAI_API_KEY" ;;
        gemini|google) add_concept_figure_secret_key "GEMINI_API_KEY" ;;
        anthropic) add_concept_figure_secret_key "ANTHROPIC_API_KEY" ;;
      esac
      ;;
  esac

  CONCEPT_FIGURE_IMAGE_RUNTIME_KEYS="SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE NODE_EXTRA_CA_CERTS VENV_PY AUTOILLUSTRATE_PYTHON AUTOILLUSTRATE_CHROME AUTOILLUSTRATE_PRICE_FILE AUTOILLUSTRATE_HEADLESS_BIN"
  CONCEPT_FIGURE_SAFE_SETTING_KEYS="AUTOILLUSTRATE_CMD_TIMEOUT AUTOILLUSTRATE_SKIP_DRAWIO_SMOKE"
  CONCEPT_FIGURE_ALLOWED_ENV_KEYS="${CONCEPT_FIGURE_SECRET_KEYS}${CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS:+ ${CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS}} ${CONCEPT_FIGURE_SAFE_SETTING_KEYS}"
  CONCEPT_FIGURE_RUNTIME_KEYS="HOME PATH USER LOGNAME SHELL TMPDIR TMP TEMP LANG LANGUAGE LC_ALL LC_CTYPE TERM COLORTERM ${CONCEPT_FIGURE_IMAGE_RUNTIME_KEYS} ${CONCEPT_FIGURE_ALLOWED_ENV_KEYS}"
  CONCEPT_FIGURE_ENV_COMMANDS="for concept_env_key in \$(compgen -e); do case \" ${CONCEPT_FIGURE_RUNTIME_KEYS} \" in *\" \${concept_env_key} \"*) ;; *) unset \"\${concept_env_key}\" ;; esac; done; unset concept_env_key"
}

# Autoillustrate's own agent-model default moves between releases (0.4.4 switched
# to claude-opus-5-5, which needs Claude Code >= 2.1.280). Always pass --model
# for the agents where Autoillustrate keeps a literal default so the figure
# model is pinned here, not by the installed renderer. Other agents keep the
# Headless per-agent default.
concept_figure_default_model() {
  case "$1" in
    claude) printf 'claude-opus-5-5' ;;
    gemini) printf 'gemini-3.1-pro-preview' ;;
    *) printf '' ;;
  esac
}

build_concept_figure_command() {
  local output_rel="$1"
  local placement="$2"
  local instructions_name="$3"
  local metrics_name="${4:-${output_rel##*/}}"
  metrics_name="${metrics_name%.png}"
  local workdir_flag=""
  local reasoning_effort_flag=""
  local model_flag=""
  local profile_flag=""
  local probe_timeout=$((CONCEPT_FIGURE_TIMEOUT_SECONDS < 10 ? CONCEPT_FIGURE_TIMEOUT_SECONDS : 10))
  if [ -n "${AGENT_REASONING_EFFORT:-}" ]; then
    case "$AGENT_REASONING_EFFORT" in
      low|medium|high|xhigh)
        reasoning_effort_flag="--effort ${AGENT_REASONING_EFFORT}"
        ;;
      *)
        echo "invalid concept-figure reasoning effort: $AGENT_REASONING_EFFORT" >&2
        return 2
        ;;
    esac
  fi
  local concept_model="${AGENT_MODEL:-}"
  if [ -z "$concept_model" ]; then
    concept_model="$(concept_figure_default_model "$AGENT")"
  fi
  if [ -n "$concept_model" ]; then
    printf -v model_flag ' --model %q' "$concept_model"
  fi
  if [ -n "${AGENT_PROFILE:-}" ]; then
    printf -v profile_flag ' --profile %q' "$AGENT_PROFILE"
  fi
  printf -v workdir_flag ' --workdir %q' "./.autoillustrate/$metrics_name"
  printf -v CONCEPT_FIGURE_COMMAND '%s\n' \
    'set -o pipefail' \
    "$CONCEPT_FIGURE_ENV_COMMANDS" \
    'autoillustrate_bin="$(command -v autoillustrate || true)"' \
    '[ -n "$autoillustrate_bin" ] || { echo "autoillustrate CLI not found" >&2; exit 127; }' \
    "autoillustrate_args=(./paper.tex --output ./${output_rel} --placement ${placement} --agent ${AGENT}${model_flag}${profile_flag}${workdir_flag} ${reasoning_effort_flag} ${CONCEPT_FIGURE_BACKEND_FLAGS} ${CONCEPT_FIGURE_MODE_FLAGS})" \
    'timeout_bin="$(command -v timeout || command -v gtimeout || true)"' \
    'if [ -z "$timeout_bin" ]; then' \
    '  echo "no timeout command found; skipping non-gating concept figure" >&2' \
    '  exit 125' \
    'fi' \
    'printf "renderer executable=%s\n" "$autoillustrate_bin"' \
    'renderer_exit() { printf "renderer exit_code=%s\n" "$?"; }' \
    'trap renderer_exit EXIT' \
    'renderer_started=$SECONDS' \
    "autoillustrate_help=\"\$(\"\$timeout_bin\" -k 1s ${probe_timeout}s \"\$autoillustrate_bin\" --help 2>&1)\" || { renderer_rc=\$?; echo \"renderer capability probe failed\" >&2; exit \"\$renderer_rc\"; }" \
    'if [[ "$autoillustrate_help" =~ --prompt-file([^[:alnum:]_-]|$) ]]; then' \
    "  autoillustrate_args+=(--prompt-file ./${instructions_name})" \
    '  echo "renderer prompt_flag=--prompt-file"' \
    'elif [[ "$autoillustrate_help" =~ --prompt([^[:alnum:]_-]|$) ]]; then' \
    "  autoillustrate_prompt=\"\$(cat -- ./${instructions_name})\" || exit 125" \
    '  autoillustrate_args+=(--prompt "$autoillustrate_prompt")' \
    '  echo "renderer prompt_flag=--prompt"' \
    'else' \
    '  echo "renderer has no supported prompt flag" >&2; exit 125' \
    'fi' \
    'renderer_rc=0' \
    "renderer_remaining=\$(( ${CONCEPT_FIGURE_TIMEOUT_SECONDS} - SECONDS + renderer_started ))" \
    '[ "$renderer_remaining" -gt 0 ] || exit 124' \
    "\"\$timeout_bin\" -k ${CONCEPT_FIGURE_KILL_GRACE_SECONDS}s \"\${renderer_remaining}s\" \"\$autoillustrate_bin\" \"\${autoillustrate_args[@]}\" || renderer_rc=\$?" \
    'exit "$renderer_rc"'
}

record_autoillustrate_costs() {
  local figure_id="$1"
  local execution_id="$2"
  local status="$3"
  local metrics_path="$CONCEPT_FIGURE_RENDER_WORK_DIR/.autoillustrate/$execution_id/metrics/run_metrics.json"
  local -a print_args=()
  [ -f "$metrics_path" ] && [ ! -L "$metrics_path" ] || return 0
  if [ "${AGENTIC_COST_TERMINAL:-0}" = "1" ]; then
    print_args+=(--print-summary)
  fi
  local -a command=(
    "$PYTHON_BIN" -m src.investigation.cost_tracking autoillustrate
    --root "${AGENTIC_COST_ROOT:-$OUTPUT_DIR}" \
    --input "$metrics_path" \
    --phase paper \
    --stage concept_figure \
    --invocation "autoillustrate:$execution_id:$request_fingerprint" \
    --status "$status"
    "${print_args[@]}"
  )
  if [ "${AGENTIC_COST_OUTPUT_FD:-1}" = "3" ]; then
    PYTHONPATH="$CONCEPT_FIGURE_LIB_DIR/../../..${PYTHONPATH:+:$PYTHONPATH}" \
      "${command[@]}" >&3 || true
  else
    PYTHONPATH="$CONCEPT_FIGURE_LIB_DIR/../../..${PYTHONPATH:+:$PYTHONPATH}" \
      "${command[@]}" || true
  fi
}

persist_concept_figure_outcome() {
  local outcome="$1"
  local figure_id="$2"
  local reason="${3:-}"
  local private_state_rc=0
  if [ "$outcome" = "failure" ]; then
    if ! is_safe_concept_output_rel "figures/${figure_id}.png" || [ -z "$reason" ]; then
      private_state_rc=1
    else
      CONCEPT_FIGURE_FAILURE_ACTIVE=1
      CONCEPT_FIGURE_FAILURE_ID="$figure_id"
      CONCEPT_FIGURE_FAILURE_REASON="$reason"
    fi
    rm -f -- "$CONCEPT_FIGURE_MARKER_PATH"
  fi

  local persistence_rc=0
  if "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$outcome" "$figure_id" "$reason" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
outcome = sys.argv[3]
figure_id = sys.argv[4]
reason = sys.argv[5]
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import (  # noqa: E402
    record_concept_figure_failure,
    record_concept_figure_success,
)

if outcome == "failure":
    record_concept_figure_failure(paper_dir, figure_id, reason)
elif outcome == "success":
    record_concept_figure_success(paper_dir, figure_id)
else:
    raise RuntimeError(f"Unsupported concept figure outcome: {outcome}")
PY
  then
    persistence_rc=0
  else
    persistence_rc=$?
  fi
  if [ "$outcome" = "success" ] && [ "$persistence_rc" -eq 0 ]; then
    CONCEPT_FIGURE_FAILURE_ACTIVE=0
    CONCEPT_FIGURE_FAILURE_ID=""
    CONCEPT_FIGURE_FAILURE_REASON=""
  fi
  if [ "$private_state_rc" -ne 0 ]; then
    return "$private_state_rc"
  fi
  return "$persistence_rc"
}

reassert_concept_figure_failure() {
  [ "${CONCEPT_FIGURE_FAILURE_ACTIVE:-0}" -eq 1 ] || return 0
  local figure_id="${CONCEPT_FIGURE_FAILURE_ID:-}"
  local reason="${CONCEPT_FIGURE_FAILURE_REASON:-}"
  if ! is_safe_concept_output_rel "figures/${figure_id}.png" || [ -z "$reason" ]; then
    return 1
  fi
  if concept_figure_has_valid_fallback "$figure_id"; then
    clear_concept_figure_provenance "$figure_id" || return 1
    publish_concept_marker || return 1
    CONCEPT_FIGURE_FAILURE_ACTIVE=0
    CONCEPT_FIGURE_FAILURE_ID=""
    CONCEPT_FIGURE_FAILURE_REASON=""
    return 0
  fi
  persist_concept_figure_outcome failure "$figure_id" "$reason"
}

concept_figure_has_valid_fallback() {
  local figure_id="$1"
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$figure_id" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
figure_id = sys.argv[3]
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import has_valid_concept_figure_fallback  # noqa: E402

raise SystemExit(0 if has_valid_concept_figure_fallback(paper_dir, figure_id) else 1)
PY
}

concept_figure_repair_fallback_matches() {
  local figure_id="$1"
  local request_fingerprint="$2"
  local audit_feedback_sha256="$3"
  local audit_finding_ids="$4"
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$figure_id" "$request_fingerprint" \
    "$audit_feedback_sha256" "$audit_finding_ids" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
figure_id = sys.argv[3]
request_fingerprint = sys.argv[4]
audit_feedback_sha256 = sys.argv[5]
audit_finding_ids = sys.argv[6].split(",") if sys.argv[6] else []
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import (  # noqa: E402
    concept_figure_repair_fallback_matches,
)

raise SystemExit(
    0
    if concept_figure_repair_fallback_matches(
        paper_dir,
        figure_id,
        request_fingerprint,
        audit_feedback_sha256,
        audit_finding_ids,
    )
    else 1
)
PY
}

promote_concept_figure_repair_fallback() {
  local figure_id="$1"
  local request_fingerprint="$2"
  local audit_feedback_sha256="$3"
  local audit_finding_ids="$4"
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$figure_id" "$request_fingerprint" \
    "$audit_feedback_sha256" "$audit_finding_ids" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
figure_id = sys.argv[3]
request_fingerprint = sys.argv[4]
audit_feedback_sha256 = sys.argv[5]
audit_finding_ids = sys.argv[6].split(",") if sys.argv[6] else []
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import (  # noqa: E402
    promote_concept_figure_repair_fallback,
)

promote_concept_figure_repair_fallback(
    paper_dir,
    figure_id,
    request_fingerprint,
    audit_feedback_sha256,
    audit_finding_ids,
)
PY
}

resolve_concept_figure_repair_feedback_authorization() {
  local figure_id="$1"
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$figure_id" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
figure_id = sys.argv[3]
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import (  # noqa: E402
    authorize_concept_figure_repair_feedback,
)

feedback_sha256, finding_ids = authorize_concept_figure_repair_feedback(
    paper_dir, figure_id
)
print(f"{feedback_sha256}\t{','.join(finding_ids)}")
PY
}

authorize_concept_figure_repair() {
  [ "${CONCEPT_FIGURE_ATTEMPTED:-0}" -eq 1 ] || return 0
  [ -f "${CONCEPT_FIGURE_RENDERED_DIR:-}/figure.png" ] || return 0
  [ -f "${CONCEPT_FIGURE_RENDERED_DIR:-}/figure.prompts.json" ] || return 0
  [ -f "${CONCEPT_FIGURE_RENDERED_DIR:-}/output_rel.txt" ] || return 0
  [ -f "$CONCEPT_FIGURE_MARKER_PATH" ] || return 0
  if [ ! -f "$OUTPUT_DIR/paper_evidence_audit_feedback.json" ] \
    || [ -L "$OUTPUT_DIR/paper_evidence_audit_feedback.json" ]; then
    return 0
  fi

  local request_tsv figure_id output_rel placement request_fingerprint stashed_output_rel
  local authorization_tsv audit_feedback_sha256 audit_finding_ids
  request_tsv="$(resolve_concept_figure_request_identity)" || return 0
  [ -n "$request_tsv" ] && [ "$request_tsv" != "SKIP" ] || return 0
  IFS=$'\t' read -r figure_id output_rel placement request_fingerprint <<<"$request_tsv"
  stashed_output_rel="$(cat "$CONCEPT_FIGURE_RENDERED_DIR/output_rel.txt")"
  if [ "$output_rel" != "$stashed_output_rel" ] \
    || [ "$request_fingerprint" != "${CONCEPT_FIGURE_REQUEST_FINGERPRINT:-}" ]; then
    return 0
  fi
  authorization_tsv="$(
    resolve_concept_figure_repair_feedback_authorization "$figure_id"
  )" || return 0
  IFS=$'\t' read -r audit_feedback_sha256 audit_finding_ids <<<"$authorization_tsv"
  if [[ ! "$audit_feedback_sha256" =~ ^[a-f0-9]{64}$ ]] \
    || [ -z "$audit_finding_ids" ]; then
    return 0
  fi
  CONCEPT_FIGURE_REPAIR_AUTHORIZED=1
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_FIGURE_ID="$figure_id"
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINGERPRINT="$request_fingerprint"
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_AUDIT_SHA256="$audit_feedback_sha256"
  CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINDING_IDS="$audit_finding_ids"
  concept_figure_log "authorized evidence-audit repair for concept figure $figure_id"
}

resolve_concept_figure_request_identity() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
sys.path.insert(0, str(script_dir))

from src.orchestrator.paper_step import resolve_concept_figure_request  # noqa: E402

request = resolve_concept_figure_request(output_dir)
if not request:
    print("SKIP")
else:
    fingerprint = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    print(
        "\t".join(
            [
                request["figure_id"],
                request["output_rel"],
                request["placement"],
                fingerprint,
            ]
        )
    )
PY
}

capture_concept_figure_preserve_snapshot() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
paper_dir = Path(sys.argv[2])
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import (  # noqa: E402
    fingerprint_concept_figure_preserve_state,
)

fingerprint = fingerprint_concept_figure_preserve_state(paper_dir)
if fingerprint is None:
    raise SystemExit(1)
print(fingerprint)
PY
}

# Resolve the strategy request and run the renderer synchronously. Failures are
# non-gating so an existing placeholder can still compile.
run_concept_figure_step() {
  local request_tsv=""
  local resolver_rc=0
  if request_tsv="$(resolve_concept_figure_request_identity)"; then
    resolver_rc=0
  else
    resolver_rc=$?
  fi
  if [ "$resolver_rc" -ne 0 ]; then
    reset_concept_figure_attempt_state
    concept_figure_warn "concept-figure resolver failed (rc=$resolver_rc); skipping figure step (non-gating)"
    return 0
  fi
  if [ -z "$request_tsv" ] || [ "$request_tsv" = "SKIP" ]; then
    reset_concept_figure_attempt_state
    concept_figure_log "no generate-concept-figure spec in paper_strategy.json; skipping figure step"
    return 0
  fi

  local figure_id output_rel placement request_fingerprint
  IFS=$'\t' read -r figure_id output_rel placement request_fingerprint <<<"$request_tsv"
  if [ -z "$figure_id" ] || [ -z "$output_rel" ] || [ -z "$placement" ] \
    || [[ ! "$request_fingerprint" =~ ^[a-f0-9]{64}$ ]] \
    || ! is_safe_concept_output_rel "$output_rel"; then
    reset_concept_figure_attempt_state
    concept_figure_warn "concept-figure resolver returned an incomplete request ('$request_tsv'); skipping figure step"
    return 0
  fi
  local request_changed=0
  if [ "${CONCEPT_FIGURE_ATTEMPTED:-0}" -eq 1 ] \
    && [ -n "${CONCEPT_FIGURE_REQUEST_FINGERPRINT:-}" ] \
    && [ "${CONCEPT_FIGURE_REQUEST_FINGERPRINT:-}" != "$request_fingerprint" ]; then
    request_changed=1
  fi
  if [ "$request_changed" -eq 1 ]; then
    CONCEPT_FIGURE_REPAIR_AUTHORIZED=0
    CONCEPT_FIGURE_REPAIR_AUTHORIZED_FIGURE_ID=""
    CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINGERPRINT=""
    CONCEPT_FIGURE_REPAIR_AUTHORIZED_AUDIT_SHA256=""
    CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINDING_IDS=""
    CONCEPT_FIGURE_FALLBACK_ACTIVE=0
    CONCEPT_FIGURE_FALLBACK_FIGURE_ID=""
    CONCEPT_FIGURE_FALLBACK_FINGERPRINT=""
    CONCEPT_FIGURE_FALLBACK_AUDIT_SHA256=""
    CONCEPT_FIGURE_FALLBACK_FINDING_IDS=""
  fi
  if [ "${CONCEPT_FIGURE_FALLBACK_ACTIVE:-0}" -eq 1 ] \
    && [ "${CONCEPT_FIGURE_FALLBACK_FIGURE_ID:-}" = "$figure_id" ] \
    && [ "${CONCEPT_FIGURE_FALLBACK_FINGERPRINT:-}" = "$request_fingerprint" ] \
    && concept_figure_repair_fallback_matches \
      "$figure_id" \
      "$request_fingerprint" \
      "${CONCEPT_FIGURE_FALLBACK_AUDIT_SHA256:-}" \
      "${CONCEPT_FIGURE_FALLBACK_FINDING_IDS:-}"; then
    CONCEPT_FIGURE_ATTEMPTED=1
    CONCEPT_FIGURE_REQUEST_FINGERPRINT="$request_fingerprint"
    concept_figure_log "preserving authorized concept-figure fallback for $figure_id"
    return 0
  fi
  CONCEPT_FIGURE_FALLBACK_ACTIVE=0
  CONCEPT_FIGURE_FALLBACK_FIGURE_ID=""
  CONCEPT_FIGURE_FALLBACK_FINGERPRINT=""
  CONCEPT_FIGURE_FALLBACK_AUDIT_SHA256=""
  CONCEPT_FIGURE_FALLBACK_FINDING_IDS=""
  if [ "${CONCEPT_FIGURE_REPAIR_AUTHORIZED:-0}" -eq 1 ] \
    && [ "${CONCEPT_FIGURE_REPAIR_AUTHORIZED_FIGURE_ID:-}" = "$figure_id" ] \
    && [ "${CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINGERPRINT:-}" = "$request_fingerprint" ]; then
    CONCEPT_FIGURE_REPAIR_AUTHORIZED=0
    CONCEPT_FIGURE_REPAIR_AUTHORIZED_FIGURE_ID=""
    CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINGERPRINT=""
    local authorized_audit_sha256 authorized_finding_ids
    authorized_audit_sha256="${CONCEPT_FIGURE_REPAIR_AUTHORIZED_AUDIT_SHA256:-}"
    authorized_finding_ids="${CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINDING_IDS:-}"
    CONCEPT_FIGURE_REPAIR_AUTHORIZED_AUDIT_SHA256=""
    CONCEPT_FIGURE_REPAIR_AUTHORIZED_FINDING_IDS=""
    if promote_concept_figure_repair_fallback \
      "$figure_id" \
      "$request_fingerprint" \
      "$authorized_audit_sha256" \
      "$authorized_finding_ids"; then
      CONCEPT_FIGURE_FALLBACK_ACTIVE=1
      CONCEPT_FIGURE_FALLBACK_FIGURE_ID="$figure_id"
      CONCEPT_FIGURE_FALLBACK_FINGERPRINT="$request_fingerprint"
      CONCEPT_FIGURE_FALLBACK_AUDIT_SHA256="$authorized_audit_sha256"
      CONCEPT_FIGURE_FALLBACK_FINDING_IDS="$authorized_finding_ids"
      CONCEPT_FIGURE_ATTEMPTED=1
      CONCEPT_FIGURE_REQUEST_FINGERPRINT="$request_fingerprint"
      CONCEPT_FIGURE_FAILURE_ACTIVE=0
      CONCEPT_FIGURE_FAILURE_ID=""
      CONCEPT_FIGURE_FAILURE_REASON=""
      if ! publish_concept_marker; then
        CONCEPT_FIGURE_FALLBACK_ACTIVE=0
        CONCEPT_FIGURE_FALLBACK_FIGURE_ID=""
        CONCEPT_FIGURE_FALLBACK_FINGERPRINT=""
        CONCEPT_FIGURE_FALLBACK_AUDIT_SHA256=""
        CONCEPT_FIGURE_FALLBACK_FINDING_IDS=""
        concept_figure_warn "could not publish repaired concept-figure marker"
        return 0
      fi
      concept_figure_log "accepted evidence-audit fallback for concept figure $figure_id"
      return 0
    fi
    concept_figure_warn "rejected incomplete evidence-audit fallback for concept figure $figure_id"
  fi
  if [ "${CONCEPT_FIGURE_ATTEMPTED:-0}" -eq 1 ] \
    && [ "${CONCEPT_FIGURE_REQUEST_FINGERPRINT:-}" = "$request_fingerprint" ]; then
    return 0
  fi
  if ! is_safe_concept_render_destination "$output_rel"; then
    reset_concept_figure_attempt_state
    CONCEPT_FIGURE_ATTEMPTED=1
    CONCEPT_FIGURE_REQUEST_FINGERPRINT="$request_fingerprint"
    persist_concept_figure_outcome \
      failure "$figure_id" "unsafe concept-figure render destination: $output_rel" \
      || concept_figure_warn "could not persist unsafe concept-figure destination provenance"
    concept_figure_warn "refusing unsafe concept-figure render destination: $output_rel"
    return 0
  fi
  if ! configure_concept_figure_budget; then
    reset_concept_figure_attempt_state
    CONCEPT_FIGURE_ATTEMPTED=1
    CONCEPT_FIGURE_REQUEST_FINGERPRINT="$request_fingerprint"
    persist_concept_figure_outcome \
      failure "$figure_id" "$CONCEPT_FIGURE_UNAVAILABLE_REASON" \
      || concept_figure_warn "could not persist concept-figure deadline provenance"
    concept_figure_warn "concept-figure deadline budget exhausted; skipping render"
    return 0
  fi

  reset_concept_figure_attempt_state
  CONCEPT_FIGURE_ATTEMPTED=1
  CONCEPT_FIGURE_REQUEST_FINGERPRINT="$request_fingerprint"

  if ! prepare_concept_figure_render_workspace; then
    persist_concept_figure_outcome \
      failure "$figure_id" "could not create staged concept-figure workspace" \
      || concept_figure_warn "could not persist concept-figure workspace failure"
    concept_figure_warn "could not create staged concept-figure workspace; skipping"
    return 0
  fi

  local instructions_path
  instructions_path="$(mktemp "$CONCEPT_FIGURE_RENDER_WORK_DIR/.concept-figure-instructions-XXXXXX")" || {
    cleanup_concept_figure_render_workspace
    persist_concept_figure_outcome \
      failure "$figure_id" "could not create concept-figure instruction file" \
      || concept_figure_warn "could not persist concept-figure failure provenance"
    concept_figure_warn "could not create a safe concept-figure instruction file; skipping"
    return 0
  }
  if ! "$PYTHON_BIN" - \
    "$REPO_ROOT" "$OUTPUT_DIR" "$instructions_path" "$request_fingerprint" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
instructions_path = Path(sys.argv[3])
expected_fingerprint = sys.argv[4]
sys.path.insert(0, str(script_dir))

from src.orchestrator.paper_step import resolve_concept_figure_request  # noqa: E402

request = resolve_concept_figure_request(output_dir)
if not request:
    raise RuntimeError("Concept figure request disappeared while staging instructions")
fingerprint = hashlib.sha256(
    json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
if fingerprint != expected_fingerprint:
    raise RuntimeError("Concept figure request changed while staging instructions")
flags = os.O_WRONLY | os.O_TRUNC
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
descriptor = os.open(instructions_path, flags)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    handle.write(request["instructions"])
PY
  then
    rm -f -- "$instructions_path"
    cleanup_concept_figure_render_workspace
    persist_concept_figure_outcome \
      failure "$figure_id" "could not safely stage concept-figure instructions" \
      || concept_figure_warn "could not persist concept-figure failure provenance"
    concept_figure_warn "could not safely stage concept-figure instructions; skipping"
    return 0
  fi

  select_concept_figure_backend
  if ! prepare_concept_figure_sandbox; then
    rm -f -- "$instructions_path"
    cleanup_concept_figure_render_workspace
    persist_concept_figure_outcome \
      failure "$figure_id" "could not create isolated concept-figure container" \
      || concept_figure_warn "could not persist concept-figure environment failure"
    concept_figure_warn "could not create isolated concept-figure container; skipping"
    return 0
  fi

  concept_figure_log "generating concept figure '$figure_id' -> $output_rel (placement=$placement, mode=${CONCEPT_FIGURE_MODE_FLAGS#--mode }); streaming log: $CONCEPT_FIGURE_RUN_LOG"

  local figure_cmd
  local instructions_name="${instructions_path##*/}"
  local autoillustrate_execution_id="${figure_id}-${TIMESTAMP}-$$-${RANDOM}"
  build_concept_figure_command \
    "$output_rel" "$placement" "$instructions_name" "$autoillustrate_execution_id"
  figure_cmd="$CONCEPT_FIGURE_COMMAND"

  local figure_rc=0
  local log_temp_dir
  local log_temp
  log_temp_dir="$(mktemp -d "$OUTPUT_DIR/.concept-figure-run-XXXXXX")" || {
    rm -f -- "$instructions_path"
    cleanup_concept_figure_scoped_env
    cleanup_concept_figure_container_home
    cleanup_concept_figure_render_workspace
    persist_concept_figure_outcome \
      failure "$figure_id" "could not create concept-figure run log" \
      || concept_figure_warn "could not persist concept-figure failure provenance"
    concept_figure_warn "could not create a safe concept-figure log; skipping"
    return 0
  }
  log_temp="$log_temp_dir/output.log"
  local AGENT_CONTAINER_SECRET_PASSTHROUGH_KEYS="$CONCEPT_FIGURE_SECRET_KEYS"
  local AGENT_CONTAINER_PASSTHROUGH_KEYS="${CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS} ${CONCEPT_FIGURE_SAFE_SETTING_KEYS}"
  local AGENT_ENV_ALLOWLIST_KEYS="$CONCEPT_FIGURE_RUNTIME_KEYS"
  local AGENT_CONTAINER_SESSION_NAME=""
  local AGENT_CONTAINER_EXTRA_MOUNTS=""
  local AGENT_CONTAINER_MOUNT_DOCKER_SOCK="0"
  local AGENT_CONTAINER_DISABLE_HOST_GATEWAY="1"
  local AGENT_CONTAINER_GPUS=""
  local AGENT_WORKDIR_READONLY="0"
  local TASK_SOURCE_DIR=""
  (
    set +e
    run_command_prompt \
      "$AGENT" \
      "$figure_cmd" \
      "$CONCEPT_FIGURE_RENDER_WORK_DIR" \
      "1" \
      "$CONTAINER_IMAGE" \
      "bridge" \
      "$CONCEPT_FIGURE_CONTAINER_HOME_DIR" \
      "$CONCEPT_FIGURE_SCOPED_ENV_FILE" \
      "$CONCEPT_FIGURE_RENDER_WORK_DIR" \
      2>&1 \
      | "$PYTHON_BIN" "$CONCEPT_FIGURE_LIB_DIR/bounded_log.py" \
        --output "$log_temp" \
        --max-bytes "${AGENT_LOG_MAX_BYTES:-16777216}"
    local -a pipeline_status=("${PIPESTATUS[@]}")
    if [ "${pipeline_status[1]}" -ne 0 ]; then
      exit "${pipeline_status[1]}"
    fi
    exit "${pipeline_status[0]}"
  ) &
  AGENT_PID=$!
  if wait "$AGENT_PID"; then
    figure_rc=0
  else
    figure_rc=$?
  fi
  AGENT_PID=""
  if [ "$figure_rc" -eq 0 ]; then
    record_autoillustrate_costs "$figure_id" "$autoillustrate_execution_id" completed
  else
    record_autoillustrate_costs "$figure_id" "$autoillustrate_execution_id" failed
  fi
  cleanup_concept_figure_scoped_env
  cleanup_concept_figure_container_home
  rm -f -- "$instructions_path"
  if [ -e "$CONCEPT_FIGURE_RUN_LOG" ] && [ ! -f "$CONCEPT_FIGURE_RUN_LOG" ] \
    && [ ! -L "$CONCEPT_FIGURE_RUN_LOG" ]; then
    concept_figure_warn "refusing to replace non-file concept log path: $CONCEPT_FIGURE_RUN_LOG"
  elif atomic_replace_path "$log_temp" "$CONCEPT_FIGURE_RUN_LOG"; then
    log_temp=""
  else
    concept_figure_warn "could not publish concept-figure log: $CONCEPT_FIGURE_RUN_LOG"
  fi
  [ -z "$log_temp" ] || rm -f -- "$log_temp"
  rmdir "$log_temp_dir" 2>/dev/null || true

  if [ "$figure_rc" -eq 130 ]; then
    cleanup_concept_figure_render_workspace
    persist_concept_figure_outcome \
      failure "$figure_id" "generate-concept-figure was interrupted (rc=130)" \
      || concept_figure_warn "could not persist concept-figure failure provenance"
    return 130
  fi
  if [ "$figure_rc" -ne 0 ]; then
    cleanup_concept_figure_render_workspace
    persist_concept_figure_outcome \
      failure "$figure_id" "generate-concept-figure exited non-zero (rc=$figure_rc)" \
      || concept_figure_warn "could not persist concept-figure failure provenance"
    concept_figure_warn "concept figure pipeline exited non-zero (rc=$figure_rc); continuing without it (non-gating). See $CONCEPT_FIGURE_RUN_LOG"
    return 0
  fi
  if ! publish_concept_figure_outputs "$figure_id"; then
    cleanup_concept_figure_render_workspace
    rollback_concept_figure_outputs \
      || concept_figure_warn "could not recover failed concept-figure publication"
    persist_concept_figure_outcome \
      failure "$figure_id" "generate-concept-figure produced unsafe staged artifacts" \
      || concept_figure_warn "could not persist unsafe staged artifact provenance"
    concept_figure_warn "concept figure artifacts failed staged publication; continuing without them"
    return 0
  fi
  cleanup_concept_figure_render_workspace
  if ! is_safe_workspace_figure "$output_rel" || [ ! -s "$OUTPUT_DIR/$output_rel" ]; then
    rollback_concept_figure_outputs \
      || concept_figure_warn "could not roll back invalid concept-figure publication"
    persist_concept_figure_outcome \
      failure "$figure_id" "generate-concept-figure reported success but $output_rel was missing or empty" \
      || concept_figure_warn "could not persist concept-figure failure provenance"
    concept_figure_warn "concept figure pipeline reported success but $output_rel is missing/empty; continuing without it (non-gating). See $CONCEPT_FIGURE_RUN_LOG"
    return 0
  fi
  if ! cp "$OUTPUT_DIR/$output_rel" "$CONCEPT_FIGURE_RENDERED_DIR/figure.png" \
    || ! printf '%s\n' "$output_rel" > "$CONCEPT_FIGURE_RENDERED_DIR/output_rel.txt"; then
    rollback_concept_figure_outputs \
      || concept_figure_warn "could not roll back unstashed concept-figure publication"
    persist_concept_figure_outcome \
      failure "$figure_id" "could not safely stash rendered concept figure" \
      || concept_figure_warn "could not persist concept-figure failure provenance"
    concept_figure_warn "could not safely stash rendered concept figure; continuing without it (non-gating)"
    return 0
  fi
  local prompts_rel="${output_rel%.png}.prompts.json"
  if [ -f "$OUTPUT_DIR/$prompts_rel" ] && [ ! -L "$OUTPUT_DIR/$prompts_rel" ]; then
    if ! cp "$OUTPUT_DIR/$prompts_rel" "$CONCEPT_FIGURE_RENDERED_DIR/figure.prompts.json"; then
      rollback_concept_figure_outputs \
        || concept_figure_warn "could not roll back unstashed concept-figure prompt log"
      persist_concept_figure_outcome \
        failure "$figure_id" "could not safely stash concept-figure prompt log" \
        || concept_figure_warn "could not persist concept-figure failure provenance"
      concept_figure_warn "could not safely stash concept-figure prompt log; continuing without it (non-gating)"
      return 0
    fi
  fi

  if ! publish_concept_marker; then
    rollback_concept_figure_outputs \
      || concept_figure_warn "could not roll back unmarked concept-figure publication"
    persist_concept_figure_outcome \
      failure "$figure_id" "could not publish concept-figure completion marker" \
      || concept_figure_warn "could not persist concept-figure failure provenance"
    concept_figure_warn "could not publish concept-figure completion marker"
    return 0
  fi
  if ! persist_concept_figure_outcome success "$figure_id"; then
    rm -f -- "$CONCEPT_FIGURE_MARKER_PATH"
    rollback_concept_figure_outputs \
      || concept_figure_warn "could not roll back unverified concept-figure publication"
    persist_concept_figure_outcome \
      failure "$figure_id" "generate-concept-figure produced output without valid provenance" \
      || concept_figure_warn "could not persist concept-figure failure provenance"
    concept_figure_warn "concept figure output lacked valid provenance; continuing without it (non-gating). See $CONCEPT_FIGURE_RUN_LOG"
    return 0
  fi
  if ! commit_concept_figure_outputs; then
    if recover_concept_figure_outputs; then
      concept_figure_log "recovered concept-figure artifact commit for $output_rel"
      return 0
    fi
    rm -f -- "$CONCEPT_FIGURE_MARKER_PATH"
    concept_figure_warn "could not reconcile concept-figure artifact commit; preserving recovery journal"
    return 0
  fi
  concept_figure_log "concept figure ready: $output_rel"
  return 0
}

prepare_unavailable_concept_figure() {
  local request_tsv=""
  if ! request_tsv="$(resolve_concept_figure_request_identity)"; then
    reset_concept_figure_attempt_state
    concept_figure_warn "concept-figure resolver failed while private state is unavailable"
    return 0
  fi
  if [ -z "$request_tsv" ] || [ "$request_tsv" = "SKIP" ]; then
    reset_concept_figure_attempt_state
    return 0
  fi

  local figure_id output_rel placement request_fingerprint
  IFS=$'\t' read -r figure_id output_rel placement request_fingerprint <<<"$request_tsv"
  if [ -z "$figure_id" ] || [ -z "$output_rel" ] || [ -z "$placement" ] \
    || [[ ! "$request_fingerprint" =~ ^[a-f0-9]{64}$ ]] \
    || ! is_safe_concept_output_rel "$output_rel"; then
    reset_concept_figure_attempt_state
    concept_figure_warn "concept-figure resolver returned an incomplete request while private state is unavailable"
    return 0
  fi
  if [ "${CONCEPT_FIGURE_ATTEMPTED:-0}" -eq 1 ] \
    && [ "${CONCEPT_FIGURE_REQUEST_FINGERPRINT:-}" = "$request_fingerprint" ]; then
    return 0
  fi

  reset_concept_figure_attempt_state
  CONCEPT_FIGURE_ATTEMPTED=1
  CONCEPT_FIGURE_REQUEST_FINGERPRINT="$request_fingerprint"
  CONCEPT_FIGURE_FAILURE_ACTIVE=1
  CONCEPT_FIGURE_FAILURE_ID="$figure_id"
  CONCEPT_FIGURE_FAILURE_REASON="${CONCEPT_FIGURE_UNAVAILABLE_REASON:-concept-figure private state unavailable}"
}

# Restore the rendered asset if a later agent repair rewrites its placeholder.
reassert_concept_figure() {
  [ -f "$CONCEPT_FIGURE_RENDERED_DIR/figure.png" ] || return 0
  [ -f "$CONCEPT_FIGURE_RENDERED_DIR/output_rel.txt" ] || return 0
  local output_rel
  output_rel="$(cat "$CONCEPT_FIGURE_RENDERED_DIR/output_rel.txt")"
  if ! is_safe_concept_output_rel "$output_rel"; then
    concept_figure_warn "refusing unsafe stashed concept-figure path: $output_rel"
    return 1
  fi
  local figure_id="${output_rel#figures/}"
  figure_id="${figure_id%.png}"
  if [ "${CONCEPT_FIGURE_FALLBACK_ACTIVE:-0}" -eq 1 ] \
    && [ "${CONCEPT_FIGURE_FALLBACK_FIGURE_ID:-}" = "$figure_id" ] \
    && [ -n "${CONCEPT_FIGURE_FALLBACK_FINGERPRINT:-}" ] \
    && concept_figure_repair_fallback_matches \
      "$figure_id" \
      "$CONCEPT_FIGURE_FALLBACK_FINGERPRINT" \
      "${CONCEPT_FIGURE_FALLBACK_AUDIT_SHA256:-}" \
      "${CONCEPT_FIGURE_FALLBACK_FINDING_IDS:-}"; then
    concept_figure_log "preserving authorized concept-figure fallback for $figure_id"
    return 0
  fi
  CONCEPT_FIGURE_FALLBACK_ACTIVE=0
  CONCEPT_FIGURE_FALLBACK_FIGURE_ID=""
  CONCEPT_FIGURE_FALLBACK_FINGERPRINT=""
  CONCEPT_FIGURE_FALLBACK_AUDIT_SHA256=""
  CONCEPT_FIGURE_FALLBACK_FINDING_IDS=""
  [ -f "$CONCEPT_FIGURE_RENDERED_DIR/figure.prompts.json" ] || return 0
  local figure_matches=0
  if is_safe_workspace_figure "$output_rel" \
    && cmp -s "$CONCEPT_FIGURE_RENDERED_DIR/figure.png" "$OUTPUT_DIR/$output_rel"; then
    figure_matches=1
  fi
  local prompts_rel="${output_rel%.png}.prompts.json"
  local prompt_matches=0
  if [ -f "$OUTPUT_DIR/$prompts_rel" ] \
    && [ ! -L "$OUTPUT_DIR/$prompts_rel" ] \
    && cmp -s \
      "$CONCEPT_FIGURE_RENDERED_DIR/figure.prompts.json" \
      "$OUTPUT_DIR/$prompts_rel"; then
    prompt_matches=1
  fi
  local provenance_matches=0
  if concept_figure_provenance_matches "$figure_id"; then
    provenance_matches=1
  fi
  if [ "$figure_matches" -eq 1 ] \
    && [ "$prompt_matches" -eq 1 ] \
    && [ "$provenance_matches" -eq 1 ]; then
    persist_concept_figure_outcome success "$figure_id" || return 1
    publish_concept_marker || return 1
    return 0
  fi

  publish_stashed_concept_figure_outputs "$figure_id" || return 1
  if ! publish_concept_marker; then
    rollback_concept_figure_outputs || true
    return 1
  fi
  if ! persist_concept_figure_outcome success "$figure_id"; then
    rm -f -- "$CONCEPT_FIGURE_MARKER_PATH"
    rollback_concept_figure_outputs || true
    return 1
  fi
  if ! commit_concept_figure_outputs; then
    if ! recover_concept_figure_outputs; then
      rm -f -- "$CONCEPT_FIGURE_MARKER_PATH"
      return 1
    fi
  fi
  concept_figure_log "re-asserted orchestrator-rendered concept artifacts for $output_rel"
}

# Ensure paper.pdf reflects a newly rendered asset even when later validation
# stops before its own compile. This remains non-gating.
compile_paper_for_concept_figure() {
  [ -f "$CONCEPT_FIGURE_MARKER_PATH" ] || return 0
  [ -f "$OUTPUT_DIR/paper.tex" ] || return 0
  if "$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$CONTAINERIZE" "$CONTAINER_IMAGE" "$CONTAINER_NETWORK" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
containerize = sys.argv[3] == "1"
container_image = sys.argv[4]
container_network = sys.argv[5]
sys.path.insert(0, str(repo_root))

from src.orchestrator.paper_step import compile_latex  # noqa: E402

compile_latex(
    paper_dir=output_dir,
    container_image=container_image if containerize else None,
    container_network=container_network,
)
PY
  then
    concept_figure_log "compiled paper.pdf with the rendered concept figure"
  else
    concept_figure_warn "best-effort concept-figure compile failed (non-fatal); see validation feedback below"
  fi
}

prepare_concept_figure_for_validation() {
  local refresh_mode="${2:-refresh}"
  if [ "$refresh_mode" = "preserve" ]; then
    local current_preserve_snapshot=""
    if current_preserve_snapshot="$(capture_concept_figure_preserve_snapshot)" \
      && [ -n "${CONCEPT_FIGURE_PRESERVE_SNAPSHOT:-}" ] \
      && [ "$current_preserve_snapshot" = "$CONCEPT_FIGURE_PRESERVE_SNAPSHOT" ]; then
      concept_figure_log "figure refresh mode is preserve; trusted concept state is unchanged"
      return 0
    fi
    concept_figure_warn "concept state changed since the preserve snapshot; refreshing"
  fi
  if [ "${CONCEPT_FIGURE_AVAILABLE:-1}" -eq 0 ]; then
    prepare_unavailable_concept_figure
    if ! reassert_concept_figure_failure; then
      concept_figure_warn "could not record unavailable concept-figure provenance; continuing"
    fi
    return 0
  fi
  local concept_figure_rc=0
  if run_concept_figure_step; then
    concept_figure_rc=0
  else
    concept_figure_rc=$?
  fi
  if [ "$concept_figure_rc" -eq 130 ]; then
    return 130
  fi

  if ! reassert_concept_figure_failure; then
    concept_figure_warn "could not re-assert concept-figure failure provenance; continuing"
  fi
  if [ "${CONCEPT_FIGURE_FAILURE_ACTIVE:-0}" -eq 1 ]; then
    return 0
  fi
  if ! reassert_concept_figure; then
    concept_figure_warn "could not safely re-assert the rendered concept figure; continuing"
  fi
  compile_paper_for_concept_figure
  return 0
}
