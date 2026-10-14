#!/usr/bin/env bash

acquire_writer_lock() {
  local results_dir="$1"
  local lock_name="$2"
  local writer_identity="$3"
  local lock_dir
  local lock_path
  local active_identity
  local results_mode

  if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required for single-writer output protection" >&2
    return 1
  fi
  if [[ ! "$lock_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Writer lock name must be one safe path component: $lock_name" >&2
    return 2
  fi
  if [[ ! -d "$results_dir" || -L "$results_dir" ]]; then
    echo "Writer lock results directory must be a real directory: $results_dir" >&2
    return 2
  fi
  results_dir="$(realpath -e -- "$results_dir")"
  results_mode="$(stat -c %a -- "$results_dir")"
  if [[ "$(stat -c %u -- "$results_dir")" != "$(id -u)" ]]; then
    echo "Writer lock results directory must be user-owned: $results_dir" >&2
    return 2
  fi
  if (( (8#$results_mode & 022) != 0 )); then
    echo "Writer lock results directory must not be group- or other-writable: $results_dir" >&2
    return 2
  fi
  lock_dir="$results_dir/.writer-locks"
  if [[ -L "$lock_dir" ]]; then
    echo "Writer lock directory must not be a symbolic link: $lock_dir" >&2
    return 2
  fi
  if [[ ! -e "$lock_dir" ]]; then
    mkdir -m 700 -- "$lock_dir" 2>/dev/null || true
  fi
  if [[ ! -d "$lock_dir" || -L "$lock_dir" ]]; then
    echo "Writer lock path must be a real directory: $lock_dir" >&2
    return 2
  fi
  lock_dir="$(realpath -e -- "$lock_dir")"
  case "$lock_dir" in
    "$results_dir"/.writer-locks) ;;
    *)
      echo "Writer lock directory escaped results: $lock_dir" >&2
      return 2
      ;;
  esac
  if [[ "$(stat -c %u -- "$lock_dir")" != "$(id -u)" ]]; then
    echo "Writer lock directory must be private and user-owned: $lock_dir" >&2
    return 2
  fi
  chmod 700 -- "$lock_dir"
  if [[ "$(stat -c %a -- "$lock_dir")" != "700" ]]; then
    echo "Writer lock directory must be private and user-owned: $lock_dir" >&2
    return 2
  fi

  lock_path="$lock_dir/$lock_name.lock"
  if [[ -L "$lock_path" ]]; then
    echo "Writer lock file must not be a symbolic link: $lock_path" >&2
    return 2
  fi
  (umask 077; : >> "$lock_path")
  if [[ ! -f "$lock_path" || -L "$lock_path" ]]; then
    echo "Writer lock path must be a regular file: $lock_path" >&2
    return 2
  fi
  chmod 600 -- "$lock_path"
  exec {WRITER_LOCK_FD}<>"$lock_path"
  if ! flock -n "$WRITER_LOCK_FD"; then
    active_identity="$(head -n 1 -- "$lock_path" 2>/dev/null || true)"
    echo "Refusing duplicate active writer: contender=[$writer_identity] active=[${active_identity:-unknown}] results=$results_dir" >&2
    return 3
  fi
  : > "$lock_path"
  printf '%s\n' "$writer_identity" >&"$WRITER_LOCK_FD"
}
