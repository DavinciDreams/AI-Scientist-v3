#!/bin/bash

set -euo pipefail

require_managed_task_python() {
  local task_dir="$1"
  local task_python="$task_dir/.venv/bin/python"
  if [ -L "$task_python" ] || [ ! -f "$task_python" ] || [ ! -x "$task_python" ]; then
    echo "managed task runtime Python must be a regular executable: $task_python" >&2
    return 1
  fi
  printf '%s\n' "$task_python"
}

ensure_staged_task_copy() {
  local repo_root="$1"
  local source_task_dir="$2"
  local results_root="$3"
  "$PYTHON_BIN" - "$repo_root" "$source_task_dir" "$results_root" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
source_task_dir = Path(sys.argv[2])
results_root = Path(sys.argv[3])

sys.path.insert(0, str(repo_root))
from src.investigation.files import ensure_task_copy_in_results  # noqa: E402

print(str(ensure_task_copy_in_results(source_task_dir, results_root)))
PY
}

ensure_workspace_dir_alias() {
  local repo_root="$1"
  local workspace_dir="$2"
  local alias_name="$3"
  local target_dir="$4"
  if [ -z "$alias_name" ] || [[ "$alias_name" == *"/"* ]] || [[ "$alias_name" == "." ]] || [[ "$alias_name" == ".." ]]; then
    echo "Invalid workspace alias name: $alias_name" >&2
    return 1
  fi
  "$PYTHON_BIN" - "$repo_root" "$workspace_dir" "$alias_name" "$target_dir" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
workspace_dir = Path(sys.argv[2])
alias_name = sys.argv[3]
target_dir = Path(sys.argv[4])

sys.path.insert(0, str(repo_root))
from src.investigation.files import ensure_directory_alias  # noqa: E402

alias_path = workspace_dir / alias_name
print(str(ensure_directory_alias(alias_path, target_dir)))
PY
}
