#!/bin/bash

_absolute_executable_file() {
  local candidate="$1"
  local candidate_dir=""
  local candidate_name=""

  if [[ "$candidate" != /* ]]; then
    candidate="$PWD/$candidate"
  fi
  candidate_dir="${candidate%/*}"
  candidate_name="${candidate##*/}"
  if [[ -z "$candidate_dir" ]]; then
    candidate_dir="/"
  fi
  candidate_dir="$(cd "$candidate_dir" 2>/dev/null && pwd -P)" || return 1
  candidate="$candidate_dir/$candidate_name"
  [[ -f "$candidate" && -x "$candidate" ]] || return 1
  printf '%s\n' "$candidate"
}

resolve_python_bin() {
  local requested_python="${WRAPPER_PYTHON:-}"
  local resolved_python=""

  if [[ -n "$requested_python" ]]; then
    if [[ "$requested_python" == */* ]]; then
      if ! resolved_python="$(_absolute_executable_file "$requested_python")"; then
        echo "Python interpreter is not an executable file: $requested_python" >&2
        return 127
      fi
      printf '%s\n' "$resolved_python"
      return 0
    fi

    resolved_python="$(type -P "$requested_python" 2>/dev/null)" || {
      echo "Python interpreter not found: $requested_python" >&2
      return 127
    }
    if ! resolved_python="$(_absolute_executable_file "$resolved_python")"; then
      echo "Python interpreter is not an executable file: $requested_python" >&2
      return 127
    fi
    printf '%s\n' "$resolved_python"
    return 0
  fi

  if [[ -n "${REPO_ROOT:-}" ]]; then
    resolved_python="$(_absolute_executable_file "$REPO_ROOT/.venv/bin/python")" || true
    if [[ -n "$resolved_python" ]]; then
      printf '%s\n' "$resolved_python"
      return 0
    fi
  fi

  resolved_python="$(type -P python3 2>/dev/null)" || true
  if [[ -n "$resolved_python" ]]; then
    resolved_python="$(_absolute_executable_file "$resolved_python")" || true
    if [[ -n "$resolved_python" ]]; then
      printf '%s\n' "$resolved_python"
      return 0
    fi
  fi

  resolved_python="$(type -P python 2>/dev/null)" || true
  if [[ -n "$resolved_python" ]]; then
    resolved_python="$(_absolute_executable_file "$resolved_python")" || true
    if [[ -n "$resolved_python" ]]; then
      printf '%s\n' "$resolved_python"
      return 0
    fi
  fi

  echo "Python interpreter not found; set WRAPPER_PYTHON or install python3" >&2
  return 127
}
