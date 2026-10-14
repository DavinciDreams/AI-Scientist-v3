#!/usr/bin/env bash
# Qualification recovery stays provisional until production and alignment both pass.

clear_provisional_acceptance() {
  ACCEPTED_ALIGNMENT_REPORT=""
  ACCEPTED_QUALIFICATION_REPORT=""
  ACCEPTED_QUALIFICATION_RECEIPT=""
}

qualification_repair_checkpoint() {
  local message
  local remaining="$((QUALIFICATION_MAX_REPAIRS - qualification_repairs_used))"
  case "$1" in
    started) message="starting repair $((qualification_repairs_used + 1))" ;;
    production_passed) message="production search passed; independent alignment pending" ;;
    production_failed) message="production search failed; edits retained" ;;
    validation_failed) message="post-repair contract or smoke validation rejected; edits retained" ;;
    alignment_rejected) message="production search passed; independent alignment rejected; edits retained" ;;
    exhausted)
      message="repair budget exhausted; last failed gate: ${QUALIFICATION_LAST_FAILURE_STAGE:-production-search}"
      ;;
    *) return 1 ;;
  esac
  printf '[task.sh] Qualification repairs %s/%s (%s remaining): %s\n' \
    "$qualification_repairs_used" "$QUALIFICATION_MAX_REPAIRS" "$remaining" "$message" >&2
}

retain_rejected_qualification_context() {
  local transaction="$1"
  local context_index="${ACTIVE_QUALIFICATION_CONTEXT_INDEX:-${transaction##*/qualification_}}"
  context_index="${context_index%_transaction}"
  local output="$VALIDATION_ROOT/qualification_${context_index}_transaction.json"
  [ "${ACTIVE_QUALIFICATION_RETAINED_CONTEXT:-}" != "$output" ] || return 0
  local report
  local -a report_args=()
  for report in "${FAILED_QUALIFICATION_REPORT:-}" \
    "${QUALIFICATION_POST_VALIDATION_REPORT:-}" "${QUALIFICATION_ALIGNMENT_REPORT:-}"; do
    if [ -n "$report" ]; then
      report_args+=(--report "$report")
    fi
  done
  "$PYTHON_BIN" -m src.task_drafting.repair_context \
    --task-dir "$STAGED_TASK_DIR" \
    --transaction-dir "$transaction" \
    --output "$output" \
    --base-fingerprint "$ACTIVE_QUALIFICATION_BASE_FINGERPRINT" \
    --rejected-fingerprint "${after_fingerprint:-${ACTIVE_QUALIFICATION_REJECTED_FINGERPRINT:-}}" \
    "${report_args[@]}" || return $?
  ACTIVE_QUALIFICATION_RETAINED_CONTEXT="$output"
}

retain_pending_qualification_context() {
  local transaction="${ACTIVE_QUALIFICATION_TRANSACTION:-}"
  [ -n "$transaction" ] || return 0
  retain_rejected_qualification_context "$transaction"
}

repair_and_qualify_until_valid() {
  local report_path="$1"
  local repair_exit
  local qualification_exit

  while [ "$qualification_repairs_used" -lt "$QUALIFICATION_MAX_REPAIRS" ]; do
    clear_provisional_acceptance
    qualification_repair_checkpoint started
    if repair_qualification_transactionally "$report_path"; then
      :
    else
      repair_exit=$?
      if [ "$repair_exit" -eq 3 ] \
        && [ "$QUALIFICATION_REPAIR_FAILURE_STAGE" = "validation" ]; then
        QUALIFICATION_LAST_FAILURE_STAGE=post-repair-validation
        qualification_repair_checkpoint validation_failed
        continue
      fi
      if [ "$QUALIFICATION_REPAIR_FAILURE_STAGE" = "agent" ]; then
        task_progress_emit qualification failed "Qualification repair agent failed" \
          --verdict fail --failure-kind infrastructure_failure
      elif [ "$QUALIFICATION_REPAIR_FAILURE_STAGE" = "oscillation" ]; then
        echo "Qualification repair oscillation detected; repeated failed scaffold rejected" >&2
      else
        echo "Qualification repair failed post-repair validation" >&2
      fi
      return "$repair_exit"
    fi
    if qualify_with_infrastructure_retry; then
      qualification_repair_checkpoint production_passed
      return 0
    else
      qualification_exit=$?
    fi
    if [ "$qualification_exit" -ne 3 ]; then
      rollback_pending_qualification_repair || return 4
      return "$qualification_exit"
    fi
    if ! retain_pending_qualification_context; then
      rollback_pending_qualification_repair || return 4
      return 4
    fi
    report_path="$FAILED_QUALIFICATION_REPORT"
    QUALIFICATION_LAST_FAILURE_STAGE=production-search
    qualification_repair_checkpoint production_failed
  done
  qualification_repair_checkpoint exhausted
  rollback_pending_qualification_repair || return 4
  return 3
}

validate_accepted_qualification() {
  "$PYTHON_BIN" -m src.task_drafting.qualification_context \
    --task-dir "$STAGED_TASK_DIR" \
    --publication-task-dir "$TASK_DIR" \
    --report "$ACCEPTED_QUALIFICATION_REPORT" \
    --receipt "$ACCEPTED_QUALIFICATION_RECEIPT" \
    --max-runtime-bytes "$TASK_DRAFT_MAX_RUNTIME_BYTES"
}
