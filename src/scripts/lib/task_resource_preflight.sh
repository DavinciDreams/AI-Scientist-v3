#!/usr/bin/env bash

task_resource_preflight() {
  local attempt=1
  local report category
  while true; do
    task_progress_emit resource_preflight started "Drafting GPU allocation check" \
      --infrastructure-attempt "$attempt" \
      --max-infrastructure-attempts "$SMOKE_MAX_INFRASTRUCTURE_ATTEMPTS"
    if report="$("$PYTHON_BIN" -m src.task_drafting.resource_preflight)"; then
      task_progress_emit resource_preflight completed "Drafting GPU allocation check" --verdict pass
      return 0
    fi
    category="$("$PYTHON_BIN" -c '
import json, sys
try:
    category = json.loads(sys.argv[1]).get("category")
except (ValueError, AttributeError):
    category = None
allowed = {"gpu_allocation_busy", "gpu_allocation_quarantined", "gpu_allocation_invalid"}
print(category if category in allowed else "gpu_allocation_invalid")
' "$report")" || category=gpu_allocation_invalid
    if [ "$category" != gpu_allocation_busy ] \
      || [ "$attempt" -ge "$SMOKE_MAX_INFRASTRUCTURE_ATTEMPTS" ]; then
      task_progress_emit resource_preflight failed "Drafting GPU allocation: $category" \
        --verdict fail --failure-kind infrastructure_failure \
        --infrastructure-attempt "$attempt" \
        --max-infrastructure-attempts "$SMOKE_MAX_INFRASTRUCTURE_ATTEMPTS"
      echo "Drafting resource preflight failed: $category; agent generation skipped" >&2
      return 4
    fi
    task_progress_emit resource_preflight retrying "Drafting GPU allocation busy" \
      --infrastructure-attempt "$attempt" \
      --max-infrastructure-attempts "$SMOKE_MAX_INFRASTRUCTURE_ATTEMPTS"
    sleep "$attempt"
    attempt=$((attempt + 1))
  done
}
