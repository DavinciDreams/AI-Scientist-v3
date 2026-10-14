#!/bin/bash

paper_figure_quality_stop_agent_session() {
  local containerize="$1"
  local session_name="$2"

  if [ "$containerize" -ne 1 ]; then
    return 1
  fi
  if [ -z "$session_name" ]; then
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1 \
    || ! stop_container_session_force "$session_name"; then
    return 1
  fi
  paper_figure_quality_confirm_agent_session_removed "$session_name"
}

paper_figure_quality_complete_agent_session() {
  local containerize="$1"
  local session_name="$2"

  if [ "$containerize" -ne 1 ]; then
    return 1
  fi
  if [ -z "$session_name" ]; then
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1 \
    || ! complete_container_session "$session_name"; then
    return 1
  fi
  paper_figure_quality_confirm_agent_session_removed "$session_name"
}

paper_figure_quality_confirm_agent_session_removed() {
  local session_name="$1"
  local inspect_error
  local inspect_rc

  if inspect_error="$(docker inspect "$session_name" 2>&1 >/dev/null)"; then
    inspect_rc=0
  else
    inspect_rc=$?
  fi
  if [ "$inspect_rc" -eq 0 ]; then
    echo "Figure-quality agent container still exists after forced stop: $session_name" >&2
    return 1
  fi
  case "$inspect_error" in
    *"No such object"*|*"No such container"*)
      return 0
      ;;
  esac
  echo "Could not confirm figure-quality agent container removal: $inspect_error" >&2
  return 1
}

paper_figure_quality_output_identity() {
  local output_dir="$1"
  local output_parent
  local output_name

  if [ -z "$output_dir" ] || [ "$output_dir" = "/" ]; then
    return 1
  fi
  output_parent="$(cd "$(dirname -- "$output_dir")" && pwd -P)" || return 1
  output_name="$(basename -- "$output_dir")"
  if [ -z "$output_name" ] || [ "$output_name" = "." ] || [ "$output_name" = ".." ]; then
    return 1
  fi
  printf '%s/%s' "$output_parent" "$output_name"
}

paper_figure_quality_target_digest() {
  local output_identity="$1"
  printf '%s' "$output_identity" | sha256sum | cut -d' ' -f1
}

paper_figure_quality_effective_writable_root() {
  local output_dir="$1"
  local requested_root="$2"
  local output_identity
  local requested_identity

  output_identity="$(paper_figure_quality_output_identity "$output_dir")" || return 1
  requested_identity="$(realpath -e -- "$requested_root")" || return 1
  case "$output_identity" in
    "$requested_identity"|"$requested_identity"/*)
      printf '%s' "$requested_identity"
      ;;
    *)
      printf '%s' "$output_identity"
      ;;
  esac
}

paper_figure_quality_tree_digest() {
  local workspace_dir="$1"
  local digest_output

  digest_output="$(
    set -o pipefail
    tar \
      --sort=name \
      --format=posix \
      --acls \
      --selinux \
      --xattrs \
      --xattrs-include='*' \
      --pax-option=delete=atime,delete=ctime \
      -cf - \
      -C "$workspace_dir" . \
      | sha256sum
  )" || return 1
  printf '%s' "${digest_output%% *}"
}

paper_figure_quality_snapshot_workspace() {
  local output_dir="$1"
  local result_variable="$2"
  local digest_variable="$3"
  local writable_root="$4"
  local output_identity
  local effective_writable_root
  local resolved_snapshot_dir
  local resolved_writable_root
  local snapshot_dir
  local snapshot_digest

  output_identity="$(paper_figure_quality_output_identity "$output_dir")" || return 1
  if [ ! -d "$output_identity" ] || [ -L "$output_identity" ]; then
    return 1
  fi
  snapshot_dir="$(mktemp -d "${TMPDIR:-/tmp}/agentic-paper-figure-quality.XXXXXX")" \
    || return 1
  effective_writable_root="$(
    paper_figure_quality_effective_writable_root "$output_identity" "$writable_root"
  )" || {
    rm -rf -- "$snapshot_dir"
    return 1
  }
  resolved_writable_root="$(realpath -e -- "$effective_writable_root")" || {
    rm -rf -- "$snapshot_dir"
    return 1
  }
  resolved_snapshot_dir="$(realpath -e -- "$snapshot_dir")" || {
    rm -rf -- "$snapshot_dir"
    return 1
  }
  case "$resolved_snapshot_dir" in
    "$resolved_writable_root"|"$resolved_writable_root"/*)
      echo "Refusing to store figure-quality snapshot inside the agent-writable root" >&2
      rm -rf -- "$snapshot_dir"
      return 1
      ;;
  esac
  if ! mkdir "$snapshot_dir/workspace" \
    || ! touch "$snapshot_dir/.agentic-paper-figure-quality-snapshot" \
    || ! paper_figure_quality_target_digest "$output_identity" \
      > "$snapshot_dir/.target-sha256" \
    || ! cp -a -- "$output_identity/." "$snapshot_dir/workspace/"; then
    rm -rf -- "$snapshot_dir"
    return 1
  fi
  snapshot_digest="$(
    paper_figure_quality_tree_digest "$snapshot_dir/workspace"
  )" || {
    rm -rf -- "$snapshot_dir"
    return 1
  }
  printf -v "$result_variable" '%s' "$snapshot_dir"
  printf -v "$digest_variable" '%s' "$snapshot_digest"
}

paper_figure_quality_snapshot_is_valid() {
  local snapshot_dir="$1"
  [ -n "$snapshot_dir" ] \
    && [ -d "$snapshot_dir" ] \
    && [ ! -L "$snapshot_dir" ] \
    && [ -f "$snapshot_dir/.agentic-paper-figure-quality-snapshot" ] \
    && [ ! -L "$snapshot_dir/.agentic-paper-figure-quality-snapshot" ] \
    && [ -f "$snapshot_dir/.target-sha256" ] \
    && [ ! -L "$snapshot_dir/.target-sha256" ] \
    && [ -d "$snapshot_dir/workspace" ] \
    && [ ! -L "$snapshot_dir/workspace" ]
}

paper_figure_quality_restore_workspace() {
  local snapshot_dir="$1"
  local output_dir="$2"
  local expected_tree_digest="$3"
  local output_identity
  local expected_digest
  local actual_tree_digest
  local debris_dir
  local debris_workspace
  local replacement_dir
  local replacement_workspace

  if ! paper_figure_quality_snapshot_is_valid "$snapshot_dir"; then
    echo "Invalid figure-quality repair snapshot: $snapshot_dir" >&2
    return 1
  fi
  output_identity="$(paper_figure_quality_output_identity "$output_dir")" || return 1
  expected_digest="$(paper_figure_quality_target_digest "$output_identity")" || return 1
  if [ "$(< "$snapshot_dir/.target-sha256")" != "$expected_digest" ]; then
    echo "Figure-quality repair snapshot target mismatch: $output_identity" >&2
    return 1
  fi
  actual_tree_digest="$(
    paper_figure_quality_tree_digest "$snapshot_dir/workspace"
  )" || return 1
  if [ -z "$expected_tree_digest" ] || [ "$actual_tree_digest" != "$expected_tree_digest" ]; then
    echo "Figure-quality repair snapshot content changed unexpectedly" >&2
    return 1
  fi

  replacement_dir="$(
    mktemp -d "$(dirname -- "$output_identity")/.figure-quality-replacement.XXXXXX"
  )" || return 1
  replacement_workspace="$replacement_dir/workspace"
  if ! cp -a -- "$snapshot_dir/workspace" "$replacement_workspace"; then
    rm -rf -- "$replacement_dir"
    return 1
  fi
  actual_tree_digest="$(
    paper_figure_quality_tree_digest "$replacement_workspace"
  )" || actual_tree_digest=""
  if [ "$actual_tree_digest" != "$expected_tree_digest" ]; then
    rm -rf -- "$replacement_dir"
    echo "Figure-quality replacement failed integrity verification" >&2
    return 1
  fi

  debris_dir="$(mktemp -d "$(dirname -- "$output_identity")/.figure-quality-debris.XXXXXX")" \
    || {
      rm -rf -- "$replacement_dir"
      return 1
    }
  debris_workspace="$debris_dir/workspace"
  if [ -e "$output_identity" ] || [ -L "$output_identity" ]; then
    if ! mv -- "$output_identity" "$debris_workspace"; then
      rmdir "$debris_dir" || true
      rm -rf -- "$replacement_dir"
      return 1
    fi
  fi
  if ! mv -- "$replacement_workspace" "$output_identity"; then
    if [ -e "$debris_workspace" ] || [ -L "$debris_workspace" ]; then
      mv -- "$debris_workspace" "$output_identity" || true
    fi
    rmdir "$debris_dir" || true
    rm -rf -- "$replacement_dir"
    return 1
  fi
  if ! rm -rf -- "$debris_dir" "$replacement_dir" "$snapshot_dir"; then
    echo "Warning: restored paper but could not remove figure-quality snapshot debris" >&2
  fi
  return 0
}

paper_figure_quality_discard_snapshot() {
  local snapshot_dir="$1"

  if [ -z "$snapshot_dir" ]; then
    return 0
  fi
  if ! paper_figure_quality_snapshot_is_valid "$snapshot_dir"; then
    echo "Refusing to remove invalid figure-quality repair snapshot: $snapshot_dir" >&2
    return 1
  fi
  rm -rf -- "$snapshot_dir"
}
