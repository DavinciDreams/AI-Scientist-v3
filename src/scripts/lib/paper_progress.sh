#!/bin/bash
# Shared trusted paper/revision progress and bounded diagnostic capture.

validate_paper_progress_settings() {
  if [[ "$PAPER_LOG_FORMAT" != "human" && "$PAPER_LOG_FORMAT" != "json" ]]; then
    echo "--log-format must be 'human' or 'json' (got: $PAPER_LOG_FORMAT)" >&2
    exit 1
  fi
  if [[ "$PAPER_LOG_VERBOSE" != "0" && "$PAPER_LOG_VERBOSE" != "1" ]]; then
    echo "PAPER_LOG_VERBOSE must be 0 or 1 (got: $PAPER_LOG_VERBOSE)" >&2
    exit 1
  fi
  if ! [[ "$PAPER_HEARTBEAT_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "--heartbeat-seconds must be a non-negative integer (got: $PAPER_HEARTBEAT_SECONDS)" >&2
    exit 1
  fi
  if [[ "$PAPER_EVENT_TRANSPORT" != "direct" && "$PAPER_EVENT_TRANSPORT" != "wrapper" ]]; then
    echo "--event-transport must be 'direct' or 'wrapper' (got: $PAPER_EVENT_TRANSPORT)" >&2
    exit 1
  fi
}

paper_progress_initialize() {
  PAPER_EVENT_BRIDGE="$REPO_ROOT/src/orchestrator/paper_event_bridge.py"
  PAPER_EVENT_JOURNAL="$OUTPUT_DIR/paper_events_${TIMESTAMP}.jsonl"
  PAPER_EVENT_SESSION_ID="${PAPER_EVENT_SESSION_ID:-paper-${TIMESTAMP}-$$}"
  PAPER_DIAGNOSTIC_LOG="$OUTPUT_DIR/paper_diagnostics_${TIMESTAMP}_$$.log"
  PAPER_CURRENT_GATE=""
  PAPER_LAST_GATE=""
  PAPER_LAST_GATE_STATE=""
  PAPER_LAST_HEARTBEAT_EPOCH=0

  exec 3>&1
  exec 1> >("$PYTHON_BIN" "$PAPER_EVENT_BRIDGE" --diagnostic-sink "$PAPER_DIAGNOSTIC_LOG") 2>&1
  PAPER_DIAGNOSTIC_SINK_PID="$!"
  export AGENTIC_COST_OUTPUT_FD=3
}

paper_diagnostic_sink_is_child() {
  local sink_pid="$1"
  local sink_ppid=""
  sink_ppid="$(ps -o ppid= -p "$sink_pid" 2>/dev/null)" || return 1
  sink_ppid="${sink_ppid//[[:space:]]/}"
  [ "$sink_ppid" = "$$" ]
}

finish_paper_diagnostic_sink() {
  local sink_pid="${PAPER_DIAGNOSTIC_SINK_PID:-}"
  local drain_ticks="${PAPER_DIAGNOSTIC_DRAIN_TICKS:-20}"
  if [ -z "$sink_pid" ]; then
    return 0
  fi
  if ! [[ "$drain_ticks" =~ ^[0-9]+$ ]] || [ "$drain_ticks" -gt 20 ]; then
    drain_ticks=20
  fi
  PAPER_DIAGNOSTIC_SINK_PID=""
  exec 1>&3 2>&1
  local tick=0
  while paper_diagnostic_sink_is_child "$sink_pid" && [ "$tick" -lt "$drain_ticks" ]; do
    sleep 0.1
    ((tick += 1))
  done
  if paper_diagnostic_sink_is_child "$sink_pid"; then
    kill "$sink_pid" 2>/dev/null || true
    tick=0
    while paper_diagnostic_sink_is_child "$sink_pid" && [ "$tick" -lt 10 ]; do
      sleep 0.1
      ((tick += 1))
    done
  fi
  if paper_diagnostic_sink_is_child "$sink_pid"; then
    kill -KILL "$sink_pid" 2>/dev/null || true
  fi
  local sink_status=0
  wait "$sink_pid" || sink_status=$?
  if [ "$sink_status" -ne 0 ]; then
    echo "[paper.sh] warning: paper diagnostic sink failed" >&2
    return 1
  fi
}

paper_event() {
  local gate="$1"
  local state="$2"
  shift 2
  local command=(
    "$PYTHON_BIN" "$PAPER_EVENT_BRIDGE"
    --journal "$PAPER_EVENT_JOURNAL"
    --format "$PAPER_LOG_FORMAT"
    --transport "$PAPER_EVENT_TRANSPORT"
    --session-id "$PAPER_EVENT_SESSION_ID"
    --gate "$gate"
    --state "$state"
  )
  local field=""
  for field in "$@"; do
    command+=(--field "$field")
  done
  if [ "$PAPER_LOG_VERBOSE" -eq 1 ]; then
    command+=(--verbose)
  fi
  if ! "${command[@]}" >&3; then
    echo "[paper.sh] warning: paper event bridge unavailable for $gate/$state" >&2
  fi
  return 0
}

paper_gate_start() {
  local gate="$1"
  shift
  PAPER_CURRENT_GATE="$gate"
  PAPER_LAST_GATE="$gate"
  PAPER_LAST_GATE_STATE=started
  PAPER_GATE_STARTED_SECONDS=$SECONDS
  PAPER_LAST_HEARTBEAT_EPOCH="$(date +%s)"
  paper_event "$gate" "started" "$@"
}

paper_revision_stage_gate() {
  case "$1" in
    *reference*|*query*) echo references ;;
    *strategy*) echo strategy ;;
    *review*) echo review ;;
    *figure*) echo figures ;;
    *) echo repair ;;
  esac
}

paper_gate_result() {
  local gate="$1"
  local exit_code="$2"
  shift 2
  if [ "$exit_code" -eq 0 ]; then
    paper_gate_complete "$gate" "$@"
  else
    paper_gate_failed "$gate" "exit_code=$exit_code" "$@"
  fi
}

paper_gate_finish() {
  local gate="$1"
  local state="$2"
  shift 2
  PAPER_LAST_GATE="$gate"
  PAPER_LAST_GATE_STATE="$state"
  paper_event "$gate" "$state" "elapsed_seconds=$((SECONDS - ${PAPER_GATE_STARTED_SECONDS:-SECONDS}))" "$@"
  if [ "$PAPER_CURRENT_GATE" = "$gate" ]; then
    PAPER_CURRENT_GATE=""
    PAPER_LAST_HEARTBEAT_EPOCH=0
  fi
}

paper_gate_complete() {
  local gate="$1"
  shift
  paper_gate_finish "$gate" "completed" "$@"
}

paper_gate_failed() {
  local gate="$1"
  shift
  paper_gate_finish "$gate" "failed" "diagnostic_log=$PAPER_DIAGNOSTIC_LOG" "$@"
}

paper_maybe_heartbeat() {
  if [ "$PAPER_HEARTBEAT_SECONDS" -eq 0 ] || [ -z "$PAPER_CURRENT_GATE" ]; then
    return 0
  fi
  local now
  now="$(date +%s)"
  if [ $((now - PAPER_LAST_HEARTBEAT_EPOCH)) -ge "$PAPER_HEARTBEAT_SECONDS" ]; then
    paper_event "$PAPER_CURRENT_GATE" "started" "heartbeat=true"
    PAPER_LAST_HEARTBEAT_EPOCH="$now"
  fi
}

paper_event_on_exit() {
  local exit_code="$1"
  if [ "$exit_code" -ne 0 ] && [ "${PAPER_LAST_GATE_STATE:-}" != failed ]; then
    local gate="${PAPER_CURRENT_GATE:-${PAPER_LAST_GATE:-finalization}}"
    paper_gate_failed "$gate" "exit_code=$exit_code" phase=workflow reason=host_workflow_stopped
  fi
  return 0
}

paper_event_early_exit() {
  local exit_code=$?
  paper_event_on_exit "$exit_code"
  finish_paper_diagnostic_sink || true
  return "$exit_code"
}
