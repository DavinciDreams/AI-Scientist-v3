#!/bin/bash

set -euo pipefail

AGENT_INNER_CMD=()
HEADLESS_NPX_SPEC="git+https://github.com/RobertTLange/headless-cli.git#93cd9b06b85f848af1308c41e018991b33907c5e"
case "${BASH_SOURCE[0]}" in
  */*) container_agent_lib_source_dir="${BASH_SOURCE[0]%/*}" ;;
  *) container_agent_lib_source_dir="." ;;
esac
CONTAINER_AGENT_LIB_DIR="$(cd "$container_agent_lib_source_dir" && pwd)"
source "$CONTAINER_AGENT_LIB_DIR/python_bin.sh"
PYTHON_BIN="$(resolve_python_bin)"
WRAPPER_PYTHON="$PYTHON_BIN"
export WRAPPER_PYTHON
AGENT_LOG_MIN_BYTES=1024
AGENT_LOG_MAX_ALLOWED_BYTES=1073741824

validate_agent_log_max_bytes() {
  local max_bytes="$1"
  if [[ ! "$max_bytes" =~ ^[1-9][0-9]*$ ]] \
    || [ "${#max_bytes}" -gt 10 ] \
    || [ "$max_bytes" -lt "$AGENT_LOG_MIN_BYTES" ] \
    || [ "$max_bytes" -gt "$AGENT_LOG_MAX_ALLOWED_BYTES" ]; then
    echo "AGENT_LOG_MAX_BYTES must be an integer from ${AGENT_LOG_MIN_BYTES} to ${AGENT_LOG_MAX_ALLOWED_BYTES}" >&2
    return 2
  fi
}

bounded_agent_log_writer() {
  local python_bin="$1"
  local output_path="$2"
  local max_bytes="$3"
  local passthrough="${4:-0}"
  local -a writer_args=(
    --output "$output_path"
    --max-bytes "$max_bytes"
  )

  validate_agent_log_max_bytes "$max_bytes" || return $?
  if [ "$passthrough" = "1" ]; then
    writer_args+=(--passthrough)
  fi
  local writer_rc=0
  "$python_bin" "$CONTAINER_AGENT_LIB_DIR/bounded_log.py" "${writer_args[@]}" || writer_rc=$?
  return "$writer_rc"
}

configure_cost_tracking() {
  local default_root="$1"
  local phase="$2"
  local output_format="${3:-human}"
  if [ -z "${AGENTIC_COST_ROOT:-}" ]; then
    export AGENTIC_COST_ROOT="$default_root"
    export AGENTIC_COST_TERMINAL="${AGENTIC_COST_TERMINAL:-1}"
  else
    export AGENTIC_COST_TERMINAL="${AGENTIC_COST_TERMINAL:-0}"
  fi
  export AGENTIC_COST_PHASE="$phase"
  export AGENTIC_COST_OUTPUT_FORMAT="${AGENTIC_COST_OUTPUT_FORMAT:-$output_format}"
}

record_headless_cost_log() {
  local python_bin="$1"
  local log_path="$2"
  local stage="$3"
  local invocation="$4"
  local status="$5"
  local -a print_args=()
  [ -n "${AGENTIC_COST_ROOT:-}" ] && [ -f "$log_path" ] || return 0
  if [ "${AGENTIC_COST_TERMINAL:-0}" = "1" ]; then
    print_args+=(--print-summary)
  fi
  local -a command=(
    "$python_bin" -m src.investigation.cost_tracking headless
    --root "$AGENTIC_COST_ROOT"
    --input "$log_path"
    --phase "${AGENTIC_COST_PHASE:-agent}"
    --stage "$stage"
    --invocation "$invocation"
    --agent "${AGENT:-unknown}"
    --status "$status"
    "${print_args[@]}"
  )
  if [ "${AGENTIC_COST_OUTPUT_FD:-1}" = "3" ]; then
    PYTHONPATH="$CONTAINER_AGENT_LIB_DIR/../../..${PYTHONPATH:+:$PYTHONPATH}" \
      "${command[@]}" >&3 || true
  else
    PYTHONPATH="$CONTAINER_AGENT_LIB_DIR/../../..${PYTHONPATH:+:$PYTHONPATH}" \
      "${command[@]}" || true
  fi
}

headless_cost_status() {
  local exit_code="$1"
  local interrupted="${2:-0}"
  if [ "$interrupted" -eq 1 ] || [ "$exit_code" -eq 130 ]; then
    echo interrupted
  elif [ "$exit_code" -eq 0 ]; then
    echo completed
  else
    echo failed
  fi
}

run_with_bounded_capture() {
  local output_path="$1"
  local max_bytes="$2"
  shift 2
  local -a pipeline_status

  "$@" 2>&1 \
    | bounded_agent_log_writer "$PYTHON_BIN" "$output_path" "$max_bytes" 1
  pipeline_status=("${PIPESTATUS[@]}")
  if [ "${pipeline_status[1]}" -ne 0 ]; then
    return "${pipeline_status[1]}"
  fi
  return "${pipeline_status[0]}"
}

validate_reasoning_effort() {
  local effort="${1:-}"
  case "$effort" in
    ""|low|medium|high|xhigh) return 0 ;;
    *)
      echo "--reasoning-effort must be one of: low, medium, high, xhigh (got: $effort)" >&2
      return 1
      ;;
  esac
}

validate_agent_reasoning_effort() {
  local agent="$1"
  local effort="${2:-}"
  validate_reasoning_effort "$effort" || return 1
  if [ "$agent" = "gemini" ] && [ -n "$effort" ]; then
    echo "--reasoning-effort is not supported for gemini" >&2
    return 1
  fi
}

chatgpt_subscription_enabled() {
  case "${AGENT_CHATGPT_SUBSCRIPTION:-0}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

chatgpt_subscription_allowed_env_keys() {
  printf '%s\n' \
    "PI_CODING_AGENT_PROVIDER PI_CODING_AGENT_MODEL PI_CODING_AGENT_MODELS"
}

agent_env_key_is_denied() {
  local expected="$1"
  local subscription_extra_allowed_keys="${2:-}"
  local key
  for key in ${AGENT_CONTAINER_ACTIVE_DENY_ENV_KEYS:-}; do
    [ "$key" = "$expected" ] && return 0
  done
  if chatgpt_subscription_enabled; then
    case " $(chatgpt_subscription_allowed_env_keys) $subscription_extra_allowed_keys " in
      *" $expected "*) ;;
      *) return 0 ;;
    esac
  fi
  return 1
}

agent_provider_allowed_env_keys() {
  local agent="$1"
  local normalized_model="${2:-}"
  local provider_env_file="${3:-}"
  local allowed_keys=""
  case "$agent" in
    claude)
      allowed_keys="HEADLESS_BILLING ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_BEARER_TOKEN_BEDROCK AWS_REGION AWS_DEFAULT_REGION AWS_REGION_NAME CLAUDE_CODE_OAUTH_TOKEN CLAUDE_CODE_USE_BEDROCK"
      ;;
    codex)
      allowed_keys="HEADLESS_BILLING HEADLESS_CAPACITY_RETRIES CODEX_API_KEY OPENAI_API_KEY"
      ;;
    cursor)
      allowed_keys="CURSOR_API_KEY"
      ;;
    gemini)
      allowed_keys="GEMINI_API_KEY GOOGLE_API_KEY"
      ;;
    pi|opencode)
      allowed_keys="PI_CODING_AGENT_PROVIDER PI_CODING_AGENT_MODEL PI_CODING_AGENT_MODELS"
      local provider_value=""
      local model_value=""
      local key value
      for key in PI_CODING_AGENT_PROVIDER PI_CODING_AGENT_MODEL; do
        value=""
        if [ -n "$provider_env_file" ]; then
          value="$(env_file_value "$provider_env_file" "$key" 2>/dev/null || true)"
        fi
        if [ -z "$value" ]; then
          value="${!key-}"
        fi
        value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
        case "$key" in
          PI_CODING_AGENT_PROVIDER) provider_value="$value" ;;
          PI_CODING_AGENT_MODEL) model_value="$value" ;;
        esac
      done

      local selected_provider=""
      local hint
      local provider_hints=("$provider_value")
      if [ -z "$provider_value" ]; then
        provider_hints=("$normalized_model" "$model_value")
      fi
      for hint in "${provider_hints[@]}"; do
        case "$hint" in
          zai|zai/*|z-ai|z-ai/*|glm-*) selected_provider="zai" ;;
          *amazon-bedrock*|*bedrock*|*aws*) selected_provider="bedrock" ;;
          *openrouter*) selected_provider="openrouter" ;;
          *deepseek*) selected_provider="deepseek" ;;
          *anthropic*|*claude*) selected_provider="anthropic" ;;
          *google*|*gemini*) selected_provider="google" ;;
          *moonshot*) selected_provider="moonshot" ;;
          *openai*|*gpt-*) selected_provider="openai" ;;
        esac
        [ -z "$selected_provider" ] || break
      done
      if [ -z "$selected_provider" ] && [ -z "$provider_value" ]; then
        selected_provider="openai"
      fi

      case "$selected_provider" in
        zai) allowed_keys+=" ZAI_API_KEY" ;;
        openrouter) allowed_keys+=" OPENROUTER_API_KEY" ;;
        deepseek) allowed_keys+=" DEEPSEEK_API_KEY" ;;
        anthropic) allowed_keys+=" ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN" ;;
        google) allowed_keys+=" GEMINI_API_KEY GOOGLE_API_KEY" ;;
        moonshot) allowed_keys+=" MOONSHOT_API_KEY" ;;
        bedrock)
          allowed_keys+=" AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_BEARER_TOKEN_BEDROCK AWS_REGION AWS_DEFAULT_REGION AWS_REGION_NAME"
          ;;
        openai)
          if ! chatgpt_subscription_enabled; then
            allowed_keys+=" OPENAI_API_KEY"
          fi
          ;;
      esac
      ;;
  esac
  printf '%s\n' "$allowed_keys"
}

agent_runtime_allowed_env_keys() {
  printf '%s\n' \
    "HOME PATH USER LOGNAME SHELL TMPDIR TMP TEMP LANG LANGUAGE LC_ALL LC_CTYPE TERM COLORTERM"
}

agent_streams_prompt_file() {
  case "$1" in
    claude|codex|gemini) return 0 ;;
    *) return 1 ;;
  esac
}

prompt_file_reference() {
  printf 'Read the complete task instructions from %s, then follow them exactly.' "$1"
}

AGENT_PYTHON_BIN="${AGENT_PYTHON_BIN:-$PYTHON_BIN}"

docker_sock_available() {
  [ -S /var/run/docker.sock ]
}

should_mount_docker_sock() {
  case "${AGENT_CONTAINER_MOUNT_DOCKER_SOCK:-0}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
  esac
  return 1
}

docker_sock_gid() {
  stat -c '%g' /var/run/docker.sock 2>/dev/null || echo ""
}

env_file_has_key() {
  local env_file="$1"
  local key="$2"
  [ -f "$env_file" ] && grep -Eq "^[[:space:]]*${key}=" "$env_file"
}

env_file_value() {
  local env_file="$1"
  local expected_key="$2"
  [ -f "$env_file" ] || return 1

  local line key value first_char last_char
  local matched_value=""
  local found=0
  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      key="${BASH_REMATCH[2]}"
      [ "$key" = "$expected_key" ] || continue
      value="${BASH_REMATCH[3]}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      if [ "${#value}" -ge 2 ]; then
        first_char="${value:0:1}"
        last_char="${value: -1}"
        if [ "$first_char" = "$last_char" ] \
          && { [ "$first_char" = "\"" ] || [ "$first_char" = "'" ]; }; then
          value="${value:1:${#value}-2}"
        fi
      fi
      matched_value="$value"
      found=1
    fi
  done < "$env_file"
  [ "$found" -eq 1 ] || return 1
  printf '%s\n' "$matched_value"
}

env_file_has_nonempty_key() {
  local value
  value="$(env_file_value "$1" "$2")" && [ -n "$value" ]
}

resolve_agent_capacity_retries() {
  local agent="$1"
  local provider_env_file="${2:-}"
  [ "$agent" = "codex" ] || return 0
  local value
  if [ "${HEADLESS_CAPACITY_RETRIES+x}" = "x" ]; then
    value="$HEADLESS_CAPACITY_RETRIES"
  elif ! value="$(env_file_value "$provider_env_file" HEADLESS_CAPACITY_RETRIES)"; then
    return 0
  fi
  case "$value" in
    0|1|2|3) printf '%s\n' "$value" ;;
    *)
      echo "[container_agent] HEADLESS_CAPACITY_RETRIES must be 0, 1, 2, or 3" >&2
      return 2
      ;;
  esac
}

append_passthrough_envs() {
  local keys_raw="${AGENT_CONTAINER_PASSTHROUGH_KEYS:-}"
  [ -n "$keys_raw" ] || return 0

  local key
  for key in $keys_raw; do
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      continue
    fi
    agent_env_key_is_denied "$key" && continue
    local value="${!key-}"
    if [ -n "${value:-}" ]; then
      docker_cmd+=(-e "$key")
    fi
  done
}

# Docker inherits these values from its own environment. Keeping values out of
# the command arguments prevents secrets from appearing in process listings.
append_secret_passthrough_envs() {
  local keys_raw="${AGENT_CONTAINER_SECRET_PASSTHROUGH_KEYS:-}"
  [ -n "$keys_raw" ] || return 0

  local key
  for key in $keys_raw; do
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      continue
    fi
    agent_env_key_is_denied "$key" && continue
    if [ -n "${!key-}" ]; then
      docker_cmd+=(-e "$key")
    fi
  done
}

append_codex_backup_env() {
  local -n backup_command="$1"
  local agent="$2"
  [ "$agent" = "codex" ] || return 0
  agent_env_key_is_denied CODEX_API_KEY && return 0
  if [ -n "${CODEX_API_KEY:-}" ]; then
    backup_command+=(-e CODEX_API_KEY)
  fi
}

append_pi_container_environment() {
  local agent="$1"
  local effective_env_file="$2"
  [ "$agent" = "pi" ] || return 0

  local key
  for key in PI_CODING_AGENT_PROVIDER PI_CODING_AGENT_MODEL PI_CODING_AGENT_MODELS; do
    [ -n "${!key-}" ] || continue
    if [ -n "$effective_env_file" ] && env_file_has_key "$effective_env_file" "$key"; then
      continue
    fi
    docker_cmd+=(-e "$key")
  done
}

restrict_agent_environment() {
  local allowed_keys="$1"
  local key
  while IFS= read -r key; do
    case " $allowed_keys " in
      *" $key "*) ;;
      *) unset "$key" 2>/dev/null || true ;;
    esac
  done < <(compgen -e)
}

append_extra_mounts() {
  local mounts_raw="${AGENT_CONTAINER_EXTRA_MOUNTS:-}"
  [ -n "$mounts_raw" ] || return 0

  local line
  while IFS= read -r line || [ -n "$line" ]; do
    # Trim whitespace.
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [ -z "$line" ]; then
      continue
    fi
    if [[ "$line" =~ ^# ]]; then
      continue
    fi
    docker_cmd+=(-v "$line")
  done <<< "$mounts_raw"
}

append_gpu_options() {
  local gpus_raw="${AGENT_CONTAINER_GPUS:-}"
  [ -n "$gpus_raw" ] || return 0

  docker_cmd+=(--gpus "$gpus_raw")
}

append_task_source_options() {
  local task_source_container="$1"
  local task_source_mount="$2"
  [ -n "$task_source_container" ] || return 0

  if [ -n "$task_source_mount" ]; then
    docker_cmd+=(-v "$task_source_mount")
  fi
  append_task_sealed_masks "$task_source_container"
  docker_cmd+=(-e "TASK_SOURCE_DIR=$task_source_container")
}

validate_task_sealed_relative_path() {
  local relative_path="$1"
  if [ -z "$relative_path" ] \
    || [[ "$relative_path" == /* ]] \
    || [[ "$relative_path" == *\\* ]] \
    || [[ "$relative_path" == *:* ]]; then
    echo "unsafe sealed task path: $relative_path" >&2
    return 1
  fi
  local component
  local -a components=()
  IFS='/' read -r -a components <<< "$relative_path"
  for component in "${components[@]}"; do
    if [ -z "$component" ] || [ "$component" = "." ] || [ "$component" = ".." ]; then
      echo "unsafe sealed task path: $relative_path" >&2
      return 1
    fi
  done
}

append_task_sealed_masks() {
  local task_source_container="$1"
  local relative_path
  while IFS= read -r relative_path || [ -n "$relative_path" ]; do
    [ -n "$relative_path" ] || continue
    validate_task_sealed_relative_path "$relative_path" || return 1
    docker_cmd+=(-v "/dev/null:${task_source_container}/${relative_path}:ro")
  done <<< "${AGENT_TASK_SEALED_FILES:-}"

  [ -n "${AGENT_TASK_SEALED_DIRS:-}" ] || return 0
  local empty_dir="${AGENT_TASK_SEAL_EMPTY_DIR:-}"
  if [[ "$empty_dir" != /* ]] || [ -L "$empty_dir" ] || [ ! -d "$empty_dir" ]; then
    echo "AGENT_TASK_SEAL_EMPTY_DIR must be an absolute, non-symlink directory" >&2
    return 1
  fi
  if find "$empty_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "AGENT_TASK_SEAL_EMPTY_DIR must be empty" >&2
    return 1
  fi
  while IFS= read -r relative_path || [ -n "$relative_path" ]; do
    [ -n "$relative_path" ] || continue
    validate_task_sealed_relative_path "$relative_path" || return 1
    docker_cmd+=(-v "${empty_dir}:${task_source_container}/${relative_path}:ro")
  done <<< "$AGENT_TASK_SEALED_DIRS"
}

append_workdir_writable_file_mount() {
  local workspace_root="$1"
  local workdir_readonly="$2"
  local source_path="${AGENT_CONTAINER_WRITABLE_FILE_SOURCE:-}"
  local relative_path="${AGENT_CONTAINER_WRITABLE_FILE_RELATIVE:-}"

  if [ -z "$source_path" ] && [ -z "$relative_path" ]; then
    return 0
  fi
  if [ "$workdir_readonly" != "1" ] \
    || [ -z "$source_path" ] \
    || [ -z "$relative_path" ]; then
    echo "writable workdir file requires a read-only workspace and both paths" >&2
    return 1
  fi
  validate_task_sealed_relative_path "$relative_path" || return 1
  if [[ "$source_path" != /* ]] \
    || [[ "$source_path" == *,* ]] \
    || [ -L "$source_path" ] \
    || [ ! -f "$source_path" ]; then
    echo "unsafe writable workdir file source" >&2
    return 1
  fi
  local canonical_source
  canonical_source="$(realpath --canonicalize-existing "$source_path")" || return 1
  if [ "$canonical_source" != "$source_path" ]; then
    echo "writable workdir file source must be canonical" >&2
    return 1
  fi
  docker_cmd+=(
    --mount
    "type=bind,src=${source_path},dst=${workspace_root}/${relative_path}"
  )
}

container_task_python() {
  local task_source_container="$1"
  if [ -n "${AGENT_CONTAINER_TASK_PYTHON:-}" ]; then
    printf '%s\n' "$AGENT_CONTAINER_TASK_PYTHON"
    return 0
  fi
  if [ -n "$task_source_container" ] \
    && [ -n "${TASK_SOURCE_DIR:-}" ] \
    && [ -f "$TASK_SOURCE_DIR/.agentic_runtime.json" ] \
    && [ ! -L "$TASK_SOURCE_DIR/.agentic_runtime.json" ] \
    && [ -f "$TASK_SOURCE_DIR/.venv/bin/python" ] \
    && [ ! -L "$TASK_SOURCE_DIR/.venv" ] \
    && [ ! -L "$TASK_SOURCE_DIR/.venv/bin/python" ]; then
    printf '%s/.venv/bin/python\n' "$task_source_container"
    return 0
  fi
  printf 'python\n'
}

normalize_positive_int() {
  local raw="$1"
  local fallback="$2"
  if [[ "$raw" =~ ^[0-9]+$ ]] && [ "$raw" -gt 0 ]; then
    echo "$raw"
    return 0
  fi
  echo "$fallback"
}

normalize_nonnegative_int() {
  local raw="$1"
  local fallback="$2"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    echo "$raw"
    return 0
  fi
  echo "$fallback"
}

write_agent_prompt_file() {
  local prompt_path="$1"
  local prompt_text="$2"
  local prompt_dir
  local prompt_tmp

  if [ -d "$prompt_path" ] && [ ! -L "$prompt_path" ]; then
    echo "[container_agent] prompt path is a directory: $prompt_path" >&2
    return 1
  fi

  prompt_dir="$(dirname "$prompt_path")"
  if ! prompt_tmp="$(mktemp "$prompt_dir/.agent-prompt-XXXXXX")"; then
    echo "[container_agent] failed to create temporary prompt in: $prompt_dir" >&2
    return 1
  fi
  if ! printf '%s\n' "$prompt_text" > "$prompt_tmp"; then
    rm -f -- "$prompt_tmp"
    echo "[container_agent] failed to write prompt file: $prompt_path" >&2
    return 1
  fi
  if [ -z "$AGENT_PYTHON_BIN" ]; then
    rm -f -- "$prompt_tmp"
    echo "[container_agent] python is required to install prompt files safely" >&2
    return 1
  fi
  if ! "$AGENT_PYTHON_BIN" - "$prompt_tmp" "$prompt_path" <<'PY'
import os
import sys

os.replace(sys.argv[1], sys.argv[2])
PY
  then
    rm -f -- "$prompt_tmp"
    echo "[container_agent] failed to install prompt file: $prompt_path" >&2
    return 1
  fi
  if [ ! -f "$prompt_path" ] || [ -L "$prompt_path" ]; then
    echo "[container_agent] prompt path is not a regular file: $prompt_path" >&2
    return 1
  fi
}

agent_prompt_file_matches_text() {
  local prompt_path="$1"
  local prompt_text="$2"

  if [ -z "$AGENT_PYTHON_BIN" ]; then
    echo "[container_agent] python is required to verify prompt files" >&2
    return 1
  fi
  printf '%s\n' "$prompt_text" | "$AGENT_PYTHON_BIN" -c '
import hashlib
import pathlib
import sys

prompt_path = pathlib.Path(sys.argv[1])
if prompt_path.is_symlink() or not prompt_path.is_file():
    raise SystemExit(1)
expected_digest = hashlib.sha256(sys.stdin.buffer.read()).digest()
actual_digest = hashlib.sha256(prompt_path.read_bytes()).digest()
raise SystemExit(0 if actual_digest == expected_digest else 1)
' "$prompt_path"
}

relative_agent_prompt_path() {
  local prompt_path="$1"
  local work_dir="$2"

  if [ -z "$AGENT_PYTHON_BIN" ]; then
    echo "[container_agent] python is required to relativize prompt paths" >&2
    return 1
  fi
  "$AGENT_PYTHON_BIN" - "$prompt_path" "$work_dir" <<'PY'
import os
import sys

prompt_path = os.path.normpath(sys.argv[1])
work_dir = os.path.normpath(sys.argv[2])
relative_path = os.path.relpath(prompt_path, work_dir)
if relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
    print(sys.argv[1])
else:
    print(relative_path)
PY
}

resolve_relative_subdir() {
  local mount_root="$1"
  local work_dir="$2"
  if [ -z "$AGENT_PYTHON_BIN" ]; then
    echo "__OUTSIDE__"
    return 0
  fi
  "$AGENT_PYTHON_BIN" - "$mount_root" "$work_dir" <<'PY'
from pathlib import Path
import sys

mount_root = Path(sys.argv[1]).resolve()
work_dir = Path(sys.argv[2]).resolve()
try:
    relative = work_dir.relative_to(mount_root)
except ValueError:
    print("__OUTSIDE__")
    raise SystemExit(0)

relative_str = relative.as_posix()
print(relative_str if relative_str else ".")
PY
}

wait_for_docker_daemon() {
  local max_attempts_raw="${AGENT_DOCKER_READY_RETRIES:-10}"
  local sleep_seconds_raw="${AGENT_DOCKER_READY_SLEEP_SECONDS:-2}"
  local max_attempts
  local sleep_seconds
  local attempt

  max_attempts="$(normalize_positive_int "$max_attempts_raw" 10)"
  sleep_seconds="$(normalize_nonnegative_int "$sleep_seconds_raw" 2)"

  for ((attempt = 1; attempt <= max_attempts; attempt += 1)); do
    if docker info >/dev/null 2>&1; then
      return 0
    fi
    if [ "$attempt" -lt "$max_attempts" ]; then
      sleep "$sleep_seconds"
    fi
  done
  return 1
}

docker_error_is_transient() {
  local docker_rc="$1"
  local output="$2"
  if [ "$docker_rc" -ne 125 ]; then
    return 1
  fi
  case "$output" in
    *"Internal Server Error"*)
      return 0
      ;;
    *"Cannot connect to the Docker daemon"*)
      return 0
      ;;
    *"is the docker daemon running"*)
      return 0
      ;;
    *"error during connect"*)
      return 0
      ;;
    *"connection refused"*)
      return 0
      ;;
    *"no route to host"*)
      return 0
      ;;
    *"/_ping"*)
      return 0
      ;;
    *"/version"*)
      return 0
      ;;
  esac
  return 1
}

sanitize_env_file_for_docker() {
  local src_file="$1"
  local dst_file="$2"
  local allowed_keys="${3-}"
  local subscription_extra_allowed_keys="${4:-}"
  local restrict_keys=0
  if [ "$#" -ge 3 ]; then
    restrict_keys=1
  fi

  : > "$dst_file" || return $?
  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" =~ ^[[:space:]]*$ ]]; then
      continue
    fi
    if [[ "$line" =~ ^[[:space:]]*# ]]; then
      continue
    fi
    if [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      local key="${BASH_REMATCH[2]}"
      local value="${BASH_REMATCH[3]}"
      local first_char
      local last_char
      if agent_env_key_is_denied "$key" "$subscription_extra_allowed_keys"; then
        continue
      fi
      if [ "$restrict_keys" -eq 1 ]; then
        case " $allowed_keys " in
          *" $key "*) ;;
          *) continue ;;
        esac
      fi

      # Trim surrounding whitespace around the value.
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"

      if [ "${#value}" -ge 2 ]; then
        first_char="${value:0:1}"
        last_char="${value: -1}"
        if [ "$first_char" = "$last_char" ] && { [ "$first_char" = "\"" ] || [ "$first_char" = "'" ]; }; then
          value="${value:1:${#value}-2}"
        fi
      fi

      printf '%s=%s\n' "$key" "$value" >> "$dst_file" || return $?
      continue
    fi

    # Keep unrecognized lines only for the legacy unrestricted mode.
    if [ "$restrict_keys" -eq 0 ]; then
      printf '%s\n' "$line" >> "$dst_file" || return $?
    fi
  done < "$src_file"
}

agent_config_rel_dir() {
  local agent="$1"
  case "$agent" in
    codex) echo ".codex" ;;
    claude) echo ".claude" ;;
    cursor) echo ".cursor" ;;
    opencode) echo ".config/opencode" ;;
    gemini) echo ".gemini" ;;
    pi) echo ".pi/agent" ;;
    *)
      echo "Unsupported agent: $agent" >&2
      return 1
      ;;
  esac
}

seed_paths_for_agent() {
  local agent="$1"
  case "$agent" in
    codex)
      cat <<'EOF'
.codex/auth.json
.codex/config.toml
EOF
      if [ -n "${AGENT_PROFILE:-}" ]; then
        printf '.codex/%s.config.toml\n.codex/%s.json\n' "$AGENT_PROFILE" "$AGENT_PROFILE"
      fi
      ;;
    claude)
      cat <<'EOF'
.claude/.credentials.json
.claude/auth.json
EOF
      ;;
    cursor)
      cat <<'EOF'
.cursor/cli-config.json
EOF
      ;;
    opencode)
      cat <<'EOF'
.config/opencode
EOF
      ;;
    gemini)
      cat <<'EOF'
.gemini/google_accounts.json
.gemini/settings.json
.gemini/state.json
.gemini/trustedFolders.json
.gemini/installation_id
EOF
      ;;
    pi)
      cat <<'EOF'
.pi/agent/auth.json
.pi/agent/settings.json
EOF
      ;;
    *)
      echo "Unsupported agent: $agent" >&2
      return 1
      ;;
  esac
}

prepare_container_home() {
  local workspace_dir="$1"
  local home_dir="$workspace_dir/.agent_home"
  mkdir -p "$home_dir"
  echo "$home_dir"
}

seed_auth_into_home() {
  local agent="$1"
  local auth_root="$2"
  local home_dir="$3"
  local rel
  local src
  local dst
  local seed_paths
  local config_dir
  local destination_parent

  mkdir -p "$home_dir" || return $?
  seed_paths="$(seed_paths_for_agent "$agent")" || return $?

  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    if chatgpt_subscription_enabled; then
      case "$agent:$rel" in
        pi:.pi/agent/auth.json|opencode:.config/opencode) continue ;;
      esac
    fi
    src="$auth_root/$rel"
    dst="$home_dir/$rel"
    if [ -f "$src" ]; then
      destination_parent="$(dirname "$dst")" || return $?
      mkdir -p "$destination_parent" || return $?
      cp "$src" "$dst" || return $?
    elif [ -d "$src" ]; then
      destination_parent="$(dirname "$dst")" || return $?
      mkdir -p "$destination_parent" || return $?
      cp -R "$src" "$dst" || return $?
    fi
  done <<< "$seed_paths"

  seed_managed_agent_files "$agent" "$auth_root" "$home_dir" || return $?

  config_dir="$(agent_config_rel_dir "$agent")" || return $?
  mkdir -p "$home_dir/$config_dir"
}

seed_managed_agent_files() {
  local agent="$1"
  local auth_root="$2"
  local home_dir="$3"

  if chatgpt_subscription_enabled; then
    if [ "$agent" = "opencode" ]; then
      "$PYTHON_BIN" "$CONTAINER_AGENT_LIB_DIR/../../investigation/safe_staging.py" \
        remove-path "$home_dir" ".config/opencode" || return $?
    fi
    case "$agent" in
      pi|opencode)
        PYTHONPATH="$CONTAINER_AGENT_LIB_DIR/../../.." \
          "$PYTHON_BIN" -m src.investigation.codex_subscription_auth \
          --agent "$agent" \
          --codex-auth "$auth_root/.codex/auth.json" \
          --home "$home_dir" || return $?
        ;;
      *)
        echo "ChatGPT subscription bridging is unsupported for agent: $agent" >&2
        return 2
        ;;
    esac
  fi

  if [ "$agent" = "pi" ]; then
    local pi_catalog="$CONTAINER_AGENT_LIB_DIR/../../agents/pi/models.json"
    if [ ! -f "$pi_catalog" ] || [ -L "$pi_catalog" ]; then
      echo "Pi model catalog is missing or unsafe: $pi_catalog" >&2
      return 1
    fi
    "$PYTHON_BIN" "$CONTAINER_AGENT_LIB_DIR/../../investigation/safe_staging.py" \
      copy-file "$home_dir" ".pi/agent/models.json" "$pi_catalog" \
      --maximum-bytes 1048576 --mode 600 || return $?
  fi
}

build_agent_inner_command() {
  local agent="$1"
  local prompt="$2"
  local agent_model="${3:-}"
  local agent_profile="${4:-}"
  local prompt_file="${AGENT_PROMPT_FILE:-}"
  local reasoning_effort="${AGENT_REASONING_EFFORT:-}"
  local use_container="${5:-0}"
  local -a headless_command=()

  agent_model="${agent_model#"${agent_model%%[![:space:]]*}"}"
  agent_model="${agent_model%"${agent_model##*[![:space:]]}"}"

  if [ -n "$agent_profile" ] && [ "$agent" != "codex" ]; then
    echo "[container_agent] agent profile is only supported for codex" >&2
    return 2
  fi
  if [ -n "$agent_profile" ] && [ -z "$agent_model" ]; then
    echo "[container_agent] agent model is required when agent profile is provided" >&2
    return 2
  fi
  validate_agent_reasoning_effort "$agent" "$reasoning_effort" || return 2
  case "$agent" in
    claude|codex|cursor|opencode|gemini|pi) ;;
    *)
      echo "Unsupported agent: $agent" >&2
      return 1
      ;;
  esac

  AGENT_INNER_CMD=(env HEADLESS_MODELS_DEV_CACHE=)
  if chatgpt_subscription_enabled; then
    AGENT_INNER_CMD+=(HEADLESS_BILLING=subscription)
  elif [ -n "${HEADLESS_BILLING:-}" ]; then
    AGENT_INNER_CMD+=("HEADLESS_BILLING=$HEADLESS_BILLING")
  fi
  if [ "$use_container" = "1" ]; then
    headless_command=(/usr/local/bin/headless)
  elif [ -n "${HEADLESS_BIN:-}" ]; then
    headless_command=("$HEADLESS_BIN")
  else
    headless_command=(npx -y "$HEADLESS_NPX_SPEC")
  fi
  AGENT_INNER_CMD+=("${headless_command[@]}" "$agent" --debug --usage --allow yolo)
  if [ -n "$agent_model" ]; then
    AGENT_INNER_CMD+=(--model "$agent_model")
  fi
  if [ -n "$agent_profile" ]; then
    AGENT_INNER_CMD+=(--profile "$agent_profile")
  fi
  if [ -n "$reasoning_effort" ]; then
    AGENT_INNER_CMD+=(--reasoning-effort "$reasoning_effort")
  fi
  if [ -n "$prompt_file" ] && agent_streams_prompt_file "$agent"; then
    AGENT_INNER_CMD+=(--prompt-file "$prompt_file")
  elif [ -n "$prompt_file" ]; then
    AGENT_INNER_CMD+=(--prompt "$(prompt_file_reference "$prompt_file")")
  else
    AGENT_INNER_CMD+=(--prompt "$prompt")
  fi
}

container_exists() {
  local container_name="$1"
  docker inspect "$container_name" >/dev/null 2>&1
}

container_is_running() {
  local container_name="$1"
  [ "$(docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null || true)" = "true" ]
}

kill_pid_tree() {
  local root_pid="$1"
  local signal_name="$2"
  local child_pid

  if [ -z "$root_pid" ]; then
    return 0
  fi
  if ! [[ "$root_pid" =~ ^[0-9]+$ ]]; then
    return 0
  fi

  for child_pid in $(pgrep -P "$root_pid" 2>/dev/null || true); do
    kill_pid_tree "$child_pid" "$signal_name"
  done

  kill "-$signal_name" "$root_pid" 2>/dev/null || true
}

terminate_pid_tree() {
  local root_pid="$1"
  local grace_seconds_raw="${2:-0.2}"
  local grace_seconds

  if [ -z "$root_pid" ]; then
    return 0
  fi
  if ! [[ "$root_pid" =~ ^[0-9]+$ ]]; then
    return 0
  fi
  if ! kill -0 "$root_pid" 2>/dev/null; then
    return 0
  fi

  grace_seconds="$grace_seconds_raw"
  if ! [[ "$grace_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    grace_seconds="0.2"
  fi
  kill_pid_tree "$root_pid" TERM
  sleep "$grace_seconds"
  kill_pid_tree "$root_pid" KILL
  wait "$root_pid" 2>/dev/null || true
}

complete_container_session() {
  local session_name="$1"
  local container_temp_dir=""
  if [ -n "${AGENTIC_CONTAINER_TEMP_ROOT:-}" ]; then
    container_temp_dir="$(
      "$PYTHON_BIN" "$CONTAINER_AGENT_LIB_DIR/container_temp.py" verify "$session_name"
    )" || container_temp_dir=""
  fi
  stop_container_session_force "$session_name" || return $?
  if [ -n "$container_temp_dir" ]; then
    "$PYTHON_BIN" "$CONTAINER_AGENT_LIB_DIR/container_temp.py" cleanup \
      "$session_name" "$container_temp_dir" || true
  fi
}

stop_container_session_force() {
  local session_name="$1"
  local max_stop_attempts_raw="${AGENT_DOCKER_STOP_RETRIES:-3}"
  local retry_sleep_raw="${AGENT_DOCKER_STOP_RETRY_SLEEP_SECONDS:-1}"
  local max_stop_attempts
  local retry_sleep
  local attempt
  local docker_rc=0
  local docker_err=""

  if [ -z "$session_name" ]; then
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1; then
    return 0
  fi

  max_stop_attempts="$(normalize_positive_int "$max_stop_attempts_raw" 3)"
  retry_sleep="$(normalize_nonnegative_int "$retry_sleep_raw" 1)"

  for ((attempt = 1; attempt <= max_stop_attempts; attempt += 1)); do
    if ! container_exists "$session_name"; then
      return 0
    fi

    set +e
    docker_err="$(docker rm -f "$session_name" 2>&1)"
    docker_rc=$?
    set -e

    if [ "$docker_rc" -eq 0 ]; then
      return 0
    fi
    case "$docker_err" in
      *"No such container"*)
        return 0
        ;;
    esac

    if [ "$attempt" -lt "$max_stop_attempts" ]; then
      sleep "$retry_sleep"
    fi
  done

  if [ -n "$docker_err" ]; then
    echo "[container_agent] failed to stop container session '$session_name': $docker_err" >&2
  else
    echo "[container_agent] failed to stop container session '$session_name'" >&2
  fi
  return 1
}

ensure_agent_session_container() {
  local agent="$1"
  local session_name="$2"
  local container_image="$3"
  local container_network="$4"
  local container_mount_root="$5"
  local container_home_dir="$6"
  local effective_env_file="$7"
  local task_source_container="${8:-}"
  local task_source_mount="${9:-}"
  local max_run_attempts_raw="${AGENT_DOCKER_RUN_RETRIES:-3}"
  local retry_sleep_seconds_raw="${AGENT_DOCKER_RUN_RETRY_SLEEP_SECONDS:-2}"
  local max_run_attempts
  local retry_sleep_seconds
  local attempt
  local docker_rc=0
  local run_output_dir=""
  local run_output_file=""
  local run_output=""
  local container_temp_dir=""

  if container_is_running "$session_name"; then
    if [ -n "${AGENTIC_CONTAINER_TEMP_ROOT:-}" ]; then
      "$PYTHON_BIN" "$CONTAINER_AGENT_LIB_DIR/container_temp.py" verify "$session_name" \
        >/dev/null || return $?
    fi
    return 0
  fi

  if container_exists "$session_name"; then
    docker rm -f "$session_name" >/dev/null 2>&1 || true
  fi

  max_run_attempts="$(normalize_positive_int "$max_run_attempts_raw" 3)"
  retry_sleep_seconds="$(normalize_nonnegative_int "$retry_sleep_seconds_raw" 2)"
  if [ -n "${AGENTIC_CONTAINER_TEMP_ROOT:-}" ]; then
    container_temp_dir="$("$PYTHON_BIN" "$CONTAINER_AGENT_LIB_DIR/container_temp.py" create)" \
      || return $?
  fi

    local -a docker_cmd=(
    docker run
    -d
    --rm
    --init
    --name "$session_name"
    --user "$(id -u):$(id -g)"
    --network "$container_network"
  )
  if should_mount_docker_sock && docker_sock_available; then
    local sock_gid
    sock_gid="$(docker_sock_gid)"
    if [ -n "$sock_gid" ]; then
      docker_cmd+=(--group-add "$sock_gid")
    fi
    docker_cmd+=(-v /var/run/docker.sock:/var/run/docker.sock)
  fi
  if [ -n "$effective_env_file" ]; then
    docker_cmd+=(
      --env-file "$effective_env_file"
    )
  fi
  append_gpu_options
  append_task_source_options "$task_source_container" "$task_source_mount"
  append_passthrough_envs
  append_secret_passthrough_envs
  append_codex_backup_env docker_cmd "$agent"
  docker_cmd+=(
    -e HOME=/home/agent-home
    -e USER=agent
    -e LOGNAME=agent
    -e "WRAPPER_PYTHON=${AGENT_CONTAINER_WRAPPER_PYTHON:-python}"
    -e "TASK_PYTHON=$(container_task_python "$task_source_container")"
  )
  append_extra_mounts
  docker_cmd+=(
    -v "$container_mount_root:/workspace"
    -v "$container_home_dir:/home/agent-home"
    -w /workspace
  )
  if [ "$agent" = "cursor" ]; then
    docker_cmd+=(-e CURSOR_CLI_BIN=/home/agent/.local/bin/agent)
  fi
  if [ "$agent" = "claude" ] \
    && [ -n "$effective_env_file" ] \
    && env_file_has_key "$effective_env_file" "AWS_ACCESS_KEY_ID" \
    && env_file_has_key "$effective_env_file" "AWS_SECRET_ACCESS_KEY" \
    && [ -z "${CLAUDE_CODE_USE_BEDROCK:-}" ] \
    && ! env_file_has_key "$effective_env_file" "CLAUDE_CODE_USE_BEDROCK"; then
    docker_cmd+=(
      -e CLAUDE_CODE_USE_BEDROCK=1
    )
  fi
  append_pi_container_environment "$agent" "$effective_env_file"
  if [ "$container_network" != "host" ] \
    && [ "${AGENT_CONTAINER_DISABLE_HOST_GATEWAY:-0}" != "1" ]; then
    docker_cmd+=(--add-host=host.docker.internal:host-gateway)
  fi
  if [ -n "$container_temp_dir" ]; then
    docker_cmd+=(
      --mount "type=bind,src=$container_temp_dir,dst=/agent-tmp"
      --label "ai.agentic-scientist.temp-path=$container_temp_dir"
      -e TMPDIR=/agent-tmp -e TMP=/agent-tmp -e TEMP=/agent-tmp
    )
  fi
  docker_cmd+=(
    "$container_image"
    sleep infinity
  )

  for ((attempt = 1; attempt <= max_run_attempts; attempt += 1)); do
    if ! wait_for_docker_daemon; then
      echo "[container_agent] docker daemon not ready while starting session '$session_name' (attempt $attempt/$max_run_attempts)" >&2
      docker_rc=125
      if [ "$attempt" -lt "$max_run_attempts" ]; then
        sleep "$retry_sleep_seconds"
        continue
      fi
      break
    fi

    run_output_dir="$(mktemp -d "${TMPDIR:-/tmp}/docker-run-output-XXXXXX")"
    run_output_file="$run_output_dir/output.log"
    set +e
    run_with_bounded_capture \
      "$run_output_file" "${AGENT_LOG_MAX_BYTES:-16777216}" "${docker_cmd[@]}"
    docker_rc=$?
    set -e

    if [ "$docker_rc" -eq 0 ]; then
      rm -f "$run_output_file"
      run_output_file=""
      rmdir "$run_output_dir"
      run_output_dir=""
      break
    fi

    if [ -f "$run_output_file" ]; then
      run_output="$(cat "$run_output_file")"
    else
      run_output=""
    fi
    rm -f "$run_output_file"
    run_output_file=""
    rmdir "$run_output_dir"
    run_output_dir=""

    if [ "$attempt" -lt "$max_run_attempts" ] && docker_error_is_transient "$docker_rc" "$run_output"; then
      echo "[container_agent] transient docker failure while starting session '$session_name' (attempt $attempt/$max_run_attempts, rc=$docker_rc); retrying in ${retry_sleep_seconds}s" >&2
      sleep "$retry_sleep_seconds"
      continue
    fi
    break
  done

  if [ -n "$run_output_file" ] && [ -f "$run_output_file" ]; then
    rm -f "$run_output_file"
  fi
  if [ -n "$run_output_dir" ] && [ -d "$run_output_dir" ]; then
    rmdir "$run_output_dir" 2>/dev/null || true
  fi

  return "$docker_rc"
}

run_agent_prompt() {
  local agent="$1"
  local prompt="$2"
  shift 2

  local agent_model=""
  local agent_profile=""
  local work_dir=""
  local use_container=""
  local container_image=""
  local container_network=""
  local container_home_dir=""
  local container_env_file=""
  local container_mount_root_arg=""
  local container_mount_root=""
  local container_session_name=""
  local subscription_extra_allowed_env_keys=""
  local workdir_readonly="${AGENT_WORKDIR_READONLY:-0}"

  if [ $# -lt 5 ]; then
    echo "[container_agent] run_agent_prompt requires at least 7 args: agent prompt work_dir use_container container_image container_network container_home_dir" >&2
    return 2
  fi

  # Backward compatibility:
  # - legacy signature (no agent_model/profile): work_dir use_container ...
  # - model signature: agent_model work_dir use_container ...
  # - profile signature: agent_model agent_profile work_dir use_container ...
  if [ "${2:-}" = "0" ] || [ "${2:-}" = "1" ]; then
    work_dir="$1"
    use_container="$2"
    container_image="$3"
    container_network="$4"
    container_home_dir="$5"
    container_env_file="${6:-}"
    container_mount_root_arg="${7:-}"
    container_session_name="${8:-${AGENT_CONTAINER_SESSION_NAME:-}}"
    subscription_extra_allowed_env_keys="${9:-}"
  elif [ "${3:-}" = "0" ] || [ "${3:-}" = "1" ]; then
    agent_model="$1"
    work_dir="$2"
    use_container="$3"
    container_image="$4"
    container_network="$5"
    container_home_dir="$6"
    container_env_file="${7:-}"
    container_mount_root_arg="${8:-}"
    container_session_name="${9:-${AGENT_CONTAINER_SESSION_NAME:-}}"
    subscription_extra_allowed_env_keys="${10:-}"
  else
    agent_model="$1"
    agent_profile="$2"
    work_dir="$3"
    use_container="$4"
    container_image="$5"
    container_network="$6"
    container_home_dir="$7"
    container_env_file="${8:-}"
    container_mount_root_arg="${9:-}"
    container_session_name="${10:-${AGENT_CONTAINER_SESSION_NAME:-}}"
    subscription_extra_allowed_env_keys="${11:-}"
  fi
  container_mount_root="$container_mount_root_arg"

  build_agent_inner_command "$agent" "$prompt" "$agent_model" "$agent_profile" "$use_container"

  _dispatch_inner_command \
    "$agent" "$work_dir" "$use_container" "$container_image" \
    "$container_network" "$container_home_dir" "$container_env_file" \
    "$container_mount_root" "$container_session_name" \
    "$subscription_extra_allowed_env_keys"
}

# Dispatch a prepared AGENT_INNER_CMD either inside the agent container or
# natively. AGENT_INNER_CMD is set by the caller — build_agent_inner_command for
# run_agent_prompt, or a literal command for run_command_prompt — so both paths
# inherit identical mount / auth / env wiring (the figure pipeline's claude
# sub-agents need the same seeded home + env-file as a normal agent stage).
#   args: agent work_dir use_container container_image container_network
#         container_home_dir container_env_file container_mount_root
#         container_session_name [subscription_extra_allowed_env_keys]
_dispatch_inner_command() {
  local agent="$1"
  local work_dir="$2"
  local use_container="$3"
  local container_image="$4"
  local container_network="$5"
  local container_home_dir="$6"
  local container_env_file="$7"
  local container_mount_root="$8"
  local container_session_name="$9"
  local subscription_extra_allowed_env_keys="${10:-}"
  local workdir_readonly="${AGENT_WORKDIR_READONLY:-0}"
  local AGENT_CONTAINER_ACTIVE_DENY_ENV_KEYS="${AGENT_CONTAINER_DENY_ENV_KEYS:-}"
  local -a AGENT_INNER_CMD=("${AGENT_INNER_CMD[@]}")
  local capacity_retries
  capacity_retries="$(resolve_agent_capacity_retries "$agent" "$container_env_file")" || return $?
  if [ -n "$capacity_retries" ]; then
    AGENT_INNER_CMD=(env "HEADLESS_CAPACITY_RETRIES=$capacity_retries" "${AGENT_INNER_CMD[@]}")
  fi

  if [ "$workdir_readonly" = "1" ] && [ -n "$container_session_name" ]; then
    echo "[container_agent] AGENT_WORKDIR_READONLY=1 requested; bypassing reusable container session for a one-shot read-only run" >&2
    container_session_name=""
  fi

  if [ "$use_container" = "1" ]; then
    local effective_env_file=""
    local temp_env_file=""
    local docker_rc=0
    local workspace_root="/workspace"
    local container_work_dir="$workspace_root"
    local relative_work_subdir=""
    local task_source_container=""
    local task_source_mount=""

    if [ -z "$container_mount_root" ] && [ -n "${AGENT_CONTAINER_MOUNT_ROOT:-}" ]; then
      container_mount_root="$AGENT_CONTAINER_MOUNT_ROOT"
    fi
    if [ -z "$container_mount_root" ]; then
      container_mount_root="$work_dir"
    fi

    if [ -n "$container_env_file" ]; then
      temp_env_file="$(mktemp "${TMPDIR:-/tmp}/agent-env-XXXXXX")"
      if chatgpt_subscription_enabled; then
        sanitize_env_file_for_docker \
          "$container_env_file" "$temp_env_file" \
          "$(chatgpt_subscription_allowed_env_keys) $subscription_extra_allowed_env_keys" \
          "$subscription_extra_allowed_env_keys"
      elif [ "${AGENT_CONTAINER_RESTRICT_ENV:-0}" = "1" ]; then
        sanitize_env_file_for_docker \
          "$container_env_file" "$temp_env_file" \
          "$(agent_provider_allowed_env_keys "$agent" "${AGENT_MODEL:-}" "$container_env_file")"
      else
        sanitize_env_file_for_docker "$container_env_file" "$temp_env_file"
      fi
      effective_env_file="$temp_env_file"
    fi

    relative_work_subdir="$(resolve_relative_subdir "$container_mount_root" "$work_dir")"
    if [ "$relative_work_subdir" = "__OUTSIDE__" ]; then
      echo "[container_agent] work directory '$work_dir' is outside mount root '$container_mount_root'; falling back to work directory mount" >&2
      container_mount_root="$work_dir"
      relative_work_subdir="."
    fi
    if [ "$relative_work_subdir" = "." ]; then
      container_work_dir="$workspace_root"
    else
      container_work_dir="$workspace_root/$relative_work_subdir"
    fi

    if [ -n "${TASK_SOURCE_DIR:-}" ]; then
      local task_source_subdir
      task_source_subdir="$(resolve_relative_subdir "$container_mount_root" "$TASK_SOURCE_DIR")"
      if [ "$task_source_subdir" = "." ]; then
        task_source_container="$workspace_root"
      elif [ "$task_source_subdir" = "__OUTSIDE__" ]; then
        task_source_container="/task-source"
        task_source_mount="$TASK_SOURCE_DIR:$task_source_container:ro"
      else
        task_source_container="$workspace_root/$task_source_subdir"
      fi
    fi

    if [ -n "$container_session_name" ]; then
      set +e
      ensure_agent_session_container \
        "$agent" \
        "$container_session_name" \
        "$container_image" \
        "$container_network" \
        "$container_mount_root" \
        "$container_home_dir" \
        "$effective_env_file" \
        "$task_source_container" \
        "$task_source_mount"
      docker_rc=$?
      set -e
      if [ "$docker_rc" -ne 0 ]; then
        if [ -n "$temp_env_file" ] && [ -f "$temp_env_file" ]; then
          rm -f "$temp_env_file"
        fi
        return "$docker_rc"
      fi

      # Inject env vars that may not have existed when the session container
      # was first started.
      local -a exec_cmd=(docker exec -w "$container_work_dir")
      exec_cmd+=(-e "WRAPPER_PYTHON=${AGENT_CONTAINER_WRAPPER_PYTHON:-python}")
      exec_cmd+=(-e "TASK_PYTHON=$(container_task_python "$task_source_container")")
      if [ -n "${AGENT_CONTAINER_PASSTHROUGH_KEYS:-}" ]; then
        local key
        for key in $AGENT_CONTAINER_PASSTHROUGH_KEYS; do
          if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            continue
          fi
          agent_env_key_is_denied "$key" && continue
          local value="${!key-}"
          if [ -n "${value:-}" ]; then
            exec_cmd+=(-e "$key")
          fi
        done
      fi
      if [ -n "${AGENT_CONTAINER_SECRET_PASSTHROUGH_KEYS:-}" ]; then
        local key
        for key in $AGENT_CONTAINER_SECRET_PASSTHROUGH_KEYS; do
          if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            continue
          fi
          agent_env_key_is_denied "$key" && continue
          if [ -n "${!key-}" ]; then
            exec_cmd+=(-e "$key")
          fi
        done
      fi
      append_codex_backup_env exec_cmd "$agent"
      if [ -n "$task_source_container" ]; then
        exec_cmd+=(-e "TASK_SOURCE_DIR=$task_source_container")
      fi
      if [ -n "${AGENTIC_CONTAINER_TEMP_ROOT:-}" ]; then
        exec_cmd+=(-e TMPDIR=/agent-tmp -e TMP=/agent-tmp -e TEMP=/agent-tmp)
      fi
      exec_cmd+=("$container_session_name" "${AGENT_INNER_CMD[@]}")
      set +e
      "${exec_cmd[@]}"
      docker_rc=$?
      set -e

      if [ -n "$temp_env_file" ] && [ -f "$temp_env_file" ]; then
        rm -f "$temp_env_file"
      fi
      return "$docker_rc"
    fi

    local max_run_attempts_raw="${AGENT_DOCKER_RUN_RETRIES:-3}"
    local retry_sleep_seconds_raw="${AGENT_DOCKER_RUN_RETRY_SLEEP_SECONDS:-2}"
    local max_run_attempts
    local retry_sleep_seconds
    local attempt
    local run_output_dir=""
    local run_output_file=""
    local run_output=""
    local container_temp_dir=""
    local temporary_container_name=""
    max_run_attempts="$(normalize_positive_int "$max_run_attempts_raw" 3)"
    retry_sleep_seconds="$(normalize_nonnegative_int "$retry_sleep_seconds_raw" 2)"
    if [ -n "${AGENTIC_CONTAINER_TEMP_ROOT:-}" ]; then
      container_temp_dir="$("$PYTHON_BIN" "$CONTAINER_AGENT_LIB_DIR/container_temp.py" create)" \
        || return $?
      temporary_container_name="agentic-$(id -u)-${container_temp_dir##*/}"
    fi

    local -a docker_cmd=(
      docker run
      --rm
      --init
      --user "$(id -u):$(id -g)"
      --network "$container_network"
    )
    if should_mount_docker_sock && docker_sock_available; then
      local sock_gid
      sock_gid="$(docker_sock_gid)"
      if [ -n "$sock_gid" ]; then
        docker_cmd+=(--group-add "$sock_gid")
      fi
      docker_cmd+=(-v /var/run/docker.sock:/var/run/docker.sock)
    fi
    if [ -n "$effective_env_file" ]; then
      docker_cmd+=(
        --env-file "$effective_env_file"
      )
    fi
    append_gpu_options
    append_task_source_options "$task_source_container" "$task_source_mount"
    append_passthrough_envs
    append_secret_passthrough_envs
    append_codex_backup_env docker_cmd "$agent"
    docker_cmd+=(
      -e HOME=/home/agent-home
      -e USER=agent
      -e LOGNAME=agent
      -e "WRAPPER_PYTHON=${AGENT_CONTAINER_WRAPPER_PYTHON:-python}"
      -e "TASK_PYTHON=$(container_task_python "$task_source_container")"
    )
    append_extra_mounts
    if [ "$workdir_readonly" = "1" ]; then
      docker_cmd+=(-v "$container_mount_root:$workspace_root:ro")
    else
      docker_cmd+=(-v "$container_mount_root:$workspace_root")
    fi
    append_workdir_writable_file_mount "$workspace_root" "$workdir_readonly"
    docker_cmd+=(
      -v "$container_home_dir:/home/agent-home"
      -w "$container_work_dir"
    )
    if [ "$agent" = "cursor" ]; then
      docker_cmd+=(-e CURSOR_CLI_BIN=/home/agent/.local/bin/agent)
    fi
    if [ "$agent" = "claude" ] \
      && [ -n "$effective_env_file" ] \
      && env_file_has_key "$effective_env_file" "AWS_ACCESS_KEY_ID" \
      && env_file_has_key "$effective_env_file" "AWS_SECRET_ACCESS_KEY" \
      && [ -z "${CLAUDE_CODE_USE_BEDROCK:-}" ] \
      && ! env_file_has_key "$effective_env_file" "CLAUDE_CODE_USE_BEDROCK"; then
      docker_cmd+=(
        -e CLAUDE_CODE_USE_BEDROCK=1
      )
    fi
    append_pi_container_environment "$agent" "$effective_env_file"
    if [ "$container_network" != "host" ] \
      && [ "${AGENT_CONTAINER_DISABLE_HOST_GATEWAY:-0}" != "1" ]; then
      docker_cmd+=(--add-host=host.docker.internal:host-gateway)
    fi
    if [ -n "$container_temp_dir" ]; then
      docker_cmd+=(
        --name "$temporary_container_name"
        --mount "type=bind,src=$container_temp_dir,dst=/agent-tmp"
        --label "ai.agentic-scientist.temp-path=$container_temp_dir"
        -e TMPDIR=/agent-tmp -e TMP=/agent-tmp -e TEMP=/agent-tmp
      )
    fi
    docker_cmd+=(
      "$container_image"
      "${AGENT_INNER_CMD[@]}"
    )

    for ((attempt = 1; attempt <= max_run_attempts; attempt += 1)); do
      if ! wait_for_docker_daemon; then
        echo "[container_agent] docker daemon not ready (attempt $attempt/$max_run_attempts)" >&2
        docker_rc=125
        if [ "$attempt" -lt "$max_run_attempts" ]; then
          sleep "$retry_sleep_seconds"
          continue
        fi
        break
      fi

      run_output_dir="$(mktemp -d "${TMPDIR:-/tmp}/docker-run-output-XXXXXX")"
      run_output_file="$run_output_dir/output.log"
      set +e
      run_with_bounded_capture \
        "$run_output_file" "${AGENT_LOG_MAX_BYTES:-16777216}" "${docker_cmd[@]}"
      docker_rc=$?
      set -e

      if [ "$docker_rc" -eq 0 ]; then
        rm -f "$run_output_file"
        run_output_file=""
        rmdir "$run_output_dir"
        run_output_dir=""
        break
      fi

      if [ -f "$run_output_file" ]; then
        run_output="$(cat "$run_output_file")"
      else
        run_output=""
      fi
      rm -f "$run_output_file"
      run_output_file=""
      rmdir "$run_output_dir"
      run_output_dir=""

      if [ "$attempt" -lt "$max_run_attempts" ] && docker_error_is_transient "$docker_rc" "$run_output"; then
        echo "[container_agent] transient docker failure (attempt $attempt/$max_run_attempts, rc=$docker_rc); retrying in ${retry_sleep_seconds}s" >&2
        sleep "$retry_sleep_seconds"
        continue
      fi
      break
    done

    if [ -n "$run_output_file" ] && [ -f "$run_output_file" ]; then
      rm -f "$run_output_file"
    fi
    if [ -n "$run_output_dir" ] && [ -d "$run_output_dir" ]; then
      rmdir "$run_output_dir" 2>/dev/null || true
    fi

    if [ -n "$temp_env_file" ] && [ -f "$temp_env_file" ]; then
      rm -f "$temp_env_file"
    fi
    if [ "$docker_rc" -eq 0 ] && [ -n "$container_temp_dir" ]; then
      "$PYTHON_BIN" "$CONTAINER_AGENT_LIB_DIR/container_temp.py" cleanup \
        "$temporary_container_name" "$container_temp_dir" || true
    fi
    return "$docker_rc"
  fi

  if [ "${AGENT_ENV_ALLOWLIST_KEYS+x}" = "x" ]; then
    (
      restrict_agent_environment "$AGENT_ENV_ALLOWLIST_KEYS"
      cd "$work_dir"
      "${AGENT_INNER_CMD[@]}"
    )
  else
    (
      cd "$work_dir"
      "${AGENT_INNER_CMD[@]}"
    )
  fi
}

# Run an arbitrary shell command in the same containerized (or native)
# environment a normal agent stage would get. Used by orchestrators that own a
# tool pipeline directly (e.g. the synchronous concept-figure step) instead of
# delegating it to an LLM turn. The command is executed via `bash -lc` so it
# sees the image's login PATH (drawio CLI, $VENV_PY, claude CLI).
#   args: agent command work_dir use_container container_image
#         container_network container_home_dir [container_env_file]
#         [container_mount_root] [container_session_name]
run_command_prompt() {
  local agent="$1"
  local command="$2"
  local work_dir="$3"
  local use_container="$4"
  local container_image="$5"
  local container_network="$6"
  local container_home_dir="$7"
  local container_env_file="${8:-}"
  local container_mount_root="${9:-}"
  local container_session_name="${10:-${AGENT_CONTAINER_SESSION_NAME:-}}"

  if [ -z "$command" ] || [ -z "$work_dir" ] || [ -z "$use_container" ]; then
    echo "[container_agent] run_command_prompt requires: agent command work_dir use_container container_image container_network container_home_dir" >&2
    return 2
  fi

  AGENT_INNER_CMD=(bash -lc "$command")

  _dispatch_inner_command \
    "$agent" "$work_dir" "$use_container" "$container_image" \
    "$container_network" "$container_home_dir" "$container_env_file" \
    "$container_mount_root" "$container_session_name"
}
