#!/bin/bash

set -euo pipefail

TASK_PROGRESS_COST_BASELINE_PREFIX="AGENTIC_SCIENTIST_TASK_COST_BASELINE "

task_progress_run_bounded() {
  local timeout_seconds="${TASK_PROGRESS_TIMEOUT_SECONDS:-5}"
  local timeout_bin
  if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || [ "$timeout_seconds" -gt 30 ]; then
    return 124
  fi
  timeout_bin="$(command -v timeout || command -v gtimeout || true)"
  [ -n "$timeout_bin" ] || return 124
  "$timeout_bin" --kill-after=1s "${timeout_seconds}s" "$@"
}

task_progress_capture_cost_baseline() {
  [ "${AGENTIC_TASK_PROGRESS_TRANSPORT:-direct}" != "wrapper" ] || return 0
  local cost_root="${AGENTIC_COST_ROOT:-${OUTPUT_DIR:-}}"
  local baseline
  [ -n "$cost_root" ] || { TASK_PROGRESS_COST_BASELINE_VALID=0; return 1; }
  if baseline="$(
    TASK_PROGRESS_TIMEOUT_SECONDS=1 task_progress_run_bounded \
      "$PYTHON_BIN" -m src.task_drafting.progress baseline --root "$cost_root" --strict
  )"; then
    TASK_PROGRESS_COST_BASELINE="$baseline"
    TASK_PROGRESS_COST_BASELINE_VALID=1
    export TASK_PROGRESS_COST_BASELINE
    return 0
  fi
  TASK_PROGRESS_COST_BASELINE_VALID=0
  return 1
}

task_progress_apply_cost_output() {
  local output="$1"
  local baseline=""
  local line
  [ -n "$output" ] || { TASK_PROGRESS_COST_BASELINE_VALID=0; return 1; }
  while IFS= read -r line; do
    if [[ "$line" == "$TASK_PROGRESS_COST_BASELINE_PREFIX"* ]]; then
      baseline="${line#"$TASK_PROGRESS_COST_BASELINE_PREFIX"}"
    else
      printf '%s\n' "$line"
    fi
  done <<< "$output"
  [ -n "$baseline" ] || { TASK_PROGRESS_COST_BASELINE_VALID=0; return 1; }
  TASK_PROGRESS_COST_BASELINE="$baseline"
  TASK_PROGRESS_COST_BASELINE_VALID=1
  export TASK_PROGRESS_COST_BASELINE
}

task_progress_initialize() {
  TASK_PROGRESS_SESSION_ID="${TASK_PROGRESS_SESSION_ID:-task-$(date +%Y%m%d_%H%M%S)-$$}"
  export TASK_PROGRESS_SESSION_ID
  if [ "${AGENTIC_TASK_PROGRESS_TRANSPORT:-direct}" = "wrapper" ]; then
    TASK_PROGRESS_COST_BASELINE="{}"
    TASK_PROGRESS_COST_BASELINE_VALID=1
  else
    TASK_PROGRESS_COST_BASELINE="{}"
    TASK_PROGRESS_COST_BASELINE_VALID=0
    task_progress_capture_cost_baseline 2>/dev/null || true
  fi
  export TASK_PROGRESS_COST_BASELINE
}

task_progress_emit() {
  local gate="$1"
  local state="$2"
  local label="$3"
  local include_cost=0
  local argument
  local -a emit_arguments
  shift 3
  if [ "$state" = "started" ]; then
    task_progress_capture_cost_baseline 2>/dev/null || true
  fi
  emit_arguments=("$@")
  for argument in "$@"; do
    [ "$argument" != "--include-cost" ] || include_cost=1
  done
  if [ "$include_cost" -eq 1 ] \
    && [ "${AGENTIC_TASK_PROGRESS_TRANSPORT:-direct}" != "wrapper" ] \
    && [ "${TASK_PROGRESS_COST_BASELINE_VALID:-0}" -ne 1 ]; then
    emit_arguments=()
    for argument in "$@"; do
      [ "$argument" = "--include-cost" ] || emit_arguments+=("$argument")
    done
  fi
  if [ "$include_cost" -eq 1 ] \
    && [ "${AGENTIC_TASK_PROGRESS_TRANSPORT:-direct}" != "wrapper" ] \
    && [ "${TASK_PROGRESS_COST_BASELINE_VALID:-0}" -eq 1 ]; then
    local output=""
    local emit_status=0
    output="$(
      task_progress_run_bounded "$PYTHON_BIN" -m src.task_drafting.progress emit \
        --gate "$gate" --state "$state" --label "$label" \
        "${emit_arguments[@]}" --return-cost-baseline
    )" || emit_status=$?
    task_progress_apply_cost_output "$output" || true
    [ "$emit_status" -eq 0 ] || TASK_PROGRESS_COST_BASELINE_VALID=0
    return 0
  fi
  task_progress_run_bounded "$PYTHON_BIN" -m src.task_drafting.progress emit \
    --gate "$gate" --state "$state" --label "$label" "${emit_arguments[@]}" || true
}

task_progress_reviewer_position() {
  case "$1" in
    a) printf '1' ;;
    b) printf '2' ;;
    c) printf '3' ;;
    *) printf '?' ;;
  esac
}

task_progress_json_field() {
  local path="$1"
  local field="$2"
  task_progress_run_bounded "$PYTHON_BIN" - "$path" "$field" <<'PY' || printf '[]\n'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get(sys.argv[2], [])
if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
    value = []
print(json.dumps(value[:32], ensure_ascii=False))
PY
}
