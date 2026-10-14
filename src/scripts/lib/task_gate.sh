#!/bin/bash

task_gate_prompt() {
  cat <<'EOF'

# Harness-owned validation gate
After each edit, run the complete acceptance check. Invoke:

    python "$TASK_GATE_CLIENT" acceptance-smoke

The production-entry check uses ordinary production device discovery and requires one completed
declared reset in each isolated plane. The service is fingerprint-bound, allowlisted, and does
not expose Docker or sealed final values.
Passing an in-session check is advisory; the harness repeats all gates after you exit. Address
every issue in the current repair prompt, then stop once their relevant gates pass. Do not spend
the bounded request budget rerunning unrelated gates; a refused request after the service closes
is operational, not a new scaffold failure.
EOF
}

write_task_gate_token() {
  local token_file="$1"
  "$PYTHON_BIN" - "$token_file" <<'PY'
import secrets
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(secrets.token_hex(32), encoding="ascii")
path.chmod(0o600)
PY
}

start_task_repair_gate() {
  local label="$1"
  local gate_root="$VALIDATION_ROOT/gate_${label}"
  local gate_socket
  local ready_file="$gate_root/ready"
  local token_file="$gate_root/token"
  local gate_log="$gate_root/service.log"
  local receipt_path="$gate_root/summary.json"
  local runtime_created_marker="$gate_root/runtime-created"
  local gate_wait
  local container_socket="/run/agentic-task-gate.sock"
  local client="$REPO_ROOT/src/task_drafting/standalone_client.py"
  if [ -n "${GATE_SERVICE_PID:-}" ] || [ -n "${GATE_SOCKET_DIRECTORY:-}" ]; then
    echo "Task repair gate already allocated" >&2
    return 4
  fi
  if ! mkdir -m 700 "$gate_root" || ! write_task_gate_token "$token_file"; then
    echo "Task repair gate private state setup failed" >&2
    return 4
  fi
  if ! GATE_SOCKET_DIRECTORY="$("$PYTHON_BIN" -m src.task_drafting.socket_runtime create)"; then
    echo "Task repair gate socket allocation failed" >&2
    return 4
  fi
  gate_socket="$GATE_SOCKET_DIRECTORY/s"
  GATE_SOCKET_IDENTITY=""
  local -a command=(
    "$PYTHON_BIN" -m src.task_drafting.service
    --socket "$gate_socket"
    --ready-file "$ready_file"
    --token-file "$token_file"
    --task-dir "$STAGED_TASK_DIR"
    --source-task-dir "$TASK_SOURCE_DIR"
    --container-image "$CONTAINER_IMAGE"
    --container-network "$CONTAINER_NETWORK"
    --timeout-seconds "$EVALUATOR_SMOKE_TIMEOUT_SECONDS"
    --production-entry-timeout-seconds "$PRODUCTION_ENTRY_TIMEOUT_SECONDS"
    --max-runtime-bytes "$TASK_DRAFT_MAX_RUNTIME_BYTES"
    --max-requests "$MAX_GATE_CHECKS_PER_REPAIR"
    --receipt-path "$receipt_path"
    --runtime-created-marker "$runtime_created_marker"
    --install-cache-dir "$RUNTIME_INSTALL_CACHE_ROOT"
  )
  if [ -n "${TASK_REPAIR_SNAPSHOT:-}" ]; then
    command+=(
      --repair-snapshot "$TASK_REPAIR_SNAPSHOT"
      --repair-transaction "$TASK_REPAIR_TRANSACTION"
      --repair-authorization "$TASK_REPAIR_AUTHORIZATION"
      --repair-session-id "$TASK_REPAIR_SESSION_ID"
    )
  fi

  TASK_GATE_TOKEN="$(<"$token_file")"
  TASK_GATE_SOCKET="$gate_socket"
  TASK_GATE_CLIENT="$client"
  TASK_GATE_TIMEOUT_SECONDS=$((
    2 * EVALUATOR_SMOKE_TIMEOUT_SECONDS
    + PRODUCTION_ENTRY_TIMEOUT_SECONDS
    + 30
  ))
  if [ "$TASK_GATE_TIMEOUT_SECONDS" -gt 7200 ]; then
    TASK_GATE_TIMEOUT_SECONDS=7200
  fi
  GATE_RUNTIME_CREATED_MARKER="$runtime_created_marker"
  TASK_GATE_EXTRA_MOUNTS=""
  if [ "$CONTAINERIZE" -eq 1 ]; then
    TASK_GATE_SOCKET="$container_socket"
    TASK_GATE_CLIENT="/tmp/agentic-task-draft-check.py"
    TASK_GATE_EXTRA_MOUNTS="$client:/tmp/agentic-task-draft-check.py:ro
$gate_socket:$container_socket"
    command+=(--containerize)
  fi
  export TASK_GATE_TOKEN TASK_GATE_SOCKET TASK_GATE_CLIENT TASK_GATE_TIMEOUT_SECONDS
  "${command[@]}" >"$gate_log" 2>&1 &
  GATE_SERVICE_PID=$!
  for gate_wait in $(seq 1 100); do
    if ! kill -0 "$GATE_SERVICE_PID" 2>/dev/null; then
      echo "Task repair gate failed to start; diagnostics: $gate_log" >&2
      abort_task_repair_gate_startup || true
      return 4
    fi
    if [ -S "$gate_socket" ] && [ -z "$GATE_SOCKET_IDENTITY" ]; then
      if ! GATE_SOCKET_IDENTITY="$("$PYTHON_BIN" -m src.task_drafting.socket_runtime identify \
        --directory "$GATE_SOCKET_DIRECTORY")"; then
        echo "Task repair gate socket identity check failed" >&2
        abort_task_repair_gate_startup || true
        return 4
      fi
    fi
    if [ -S "$gate_socket" ] && [ -f "$ready_file" ] && [ ! -L "$ready_file" ] \
      && [ "$(<"$ready_file")" = ready ]; then
      return 0
    fi
    sleep 0.05
  done
  echo "Task repair gate did not become ready; diagnostics: $gate_log" >&2
  abort_task_repair_gate_startup || true
  return 4
}

cleanup_task_gate_endpoint() {
  local socket_identity="${1:-}"
  local -a identity_args=()
  if [ -n "$socket_identity" ]; then
    identity_args=(--socket-identity "$socket_identity")
  fi
  unset TASK_GATE_TOKEN TASK_GATE_SOCKET TASK_GATE_CLIENT TASK_GATE_TIMEOUT_SECONDS
  TASK_GATE_EXTRA_MOUNTS=""
  if [ -n "${GATE_SOCKET_DIRECTORY:-}" ]; then
    if ! "$PYTHON_BIN" -m src.task_drafting.socket_runtime cleanup \
      --directory "$GATE_SOCKET_DIRECTORY" "${identity_args[@]}"; then
      echo "Task repair gate socket cleanup failed" >&2
      return 4
    fi
  fi
  GATE_SOCKET_DIRECTORY=""
  GATE_SOCKET_IDENTITY=""
}

abort_task_repair_gate_startup() {
  local poll
  if [ -n "${GATE_SERVICE_PID:-}" ]; then
    kill "$GATE_SERVICE_PID" 2>/dev/null || true
    for ((poll = 0; poll < 50; poll += 1)); do
      kill -0 "$GATE_SERVICE_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$GATE_SERVICE_PID" 2>/dev/null; then
      kill -KILL "$GATE_SERVICE_PID" 2>/dev/null || true
    fi
    wait "$GATE_SERVICE_PID" 2>/dev/null || true
  fi
  GATE_SERVICE_PID=""
  cleanup_task_gate_endpoint "${GATE_SOCKET_IDENTITY:-}"
}

stop_task_repair_gate() {
  if [ -n "${GATE_SERVICE_PID:-}" ] && kill -0 "$GATE_SERVICE_PID" 2>/dev/null; then
    kill "$GATE_SERVICE_PID" 2>/dev/null || true
  fi
  if [ -n "${GATE_SERVICE_PID:-}" ]; then
    wait "$GATE_SERVICE_PID" 2>/dev/null || true
  fi
  GATE_SERVICE_PID=""
  cleanup_task_gate_endpoint
}
