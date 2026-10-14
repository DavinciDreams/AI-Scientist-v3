"""Ownership marker for safely resuming generated paper workspaces."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path


MARKER_NAME = ".agentic_paper_workspace.json"
SCHEMA_VERSION = 1
WRITABLE_GLOBAL_MEMORY_FILES = (
    "memory.md",
    "memory.jsonl",
    "conclusions_history.jsonl",
)
MAX_WRITABLE_GLOBAL_MEMORY_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class PaperContainerMountPlan:
    mount_root: Path
    readonly_mounts: tuple[str, ...]
    writable_mounts: tuple[str, ...]


def _resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def paper_container_mount_plan(
    *,
    output_dir: Path,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path,
    aggregated_dir: Path | None = None,
    extra_tasks_dir: Path | None = None,
) -> PaperContainerMountPlan:
    """Mount dependencies read-only while overlaying paper and memory writes."""
    output = output_dir.expanduser().resolve()
    task = task_dir.expanduser().resolve()
    results = results_dir.expanduser().resolve()
    global_memory = global_memory_dir.expanduser().resolve()
    required_dependencies = [task, results, global_memory]
    optional_dependencies = [
        ("aggregated", aggregated_dir.expanduser().resolve())
        if aggregated_dir is not None
        else None,
        ("extra_tasks", extra_tasks_dir.expanduser().resolve())
        if extra_tasks_dir is not None
        else None,
    ]

    legacy_layout = output.is_relative_to(results) and all(
        dependency.is_relative_to(results) for dependency in required_dependencies
    )
    run_root = output.parent
    phase_layout = (
        task.name == "task"
        and task.parent == run_root
        and results.name == "investigation"
        and results.parent == run_root
        and all(
            dependency.is_relative_to(run_root)
            for dependency in required_dependencies
        )
        and _valid_run_policy_marker(run_root / ".agentic_resume_policy.json")
    )
    mount_root = results if legacy_layout else run_root if phase_layout else None
    if mount_root is None:
        raise ValueError("Paper paths do not share a supported bounded container root")
    if mount_root == Path(mount_root.anchor) or mount_root == Path.home().resolve():
        raise ValueError(f"Refusing broad paper container mount root: {mount_root}")
    if (mount_root / ".git").exists():
        raise ValueError(f"Refusing repository-wide paper container mount root: {mount_root}")
    if output in required_dependencies:
        raise ValueError("Paper output must be distinct from its dependencies")

    _require_owned_real_directory(output)
    readonly_mounts = tuple(
        _external_dependency_mount(
            alias_name=alias_name,
            source=source,
            output=output,
            mount_root=mount_root,
        )
        for dependency in optional_dependencies
        if dependency is not None
        for alias_name, source in (dependency,)
        if not source.is_relative_to(mount_root)
    )
    writable_paths = [output]
    writable_paths.extend(global_memory / name for name in WRITABLE_GLOBAL_MEMORY_FILES)
    mounts: list[str] = []
    for source in writable_paths:
        if source != output:
            _require_owned_regular_file(source)
        source_text = str(source)
        if any(character in source_text for character in ("\n", "\r", ":")):
            raise ValueError(f"Paper path cannot be mounted safely: {source}")
        relative = source.relative_to(mount_root).as_posix()
        mounts.append(f"{source_text}:/workspace/{relative}:rw")
    return PaperContainerMountPlan(
        mount_root=mount_root,
        readonly_mounts=readonly_mounts,
        writable_mounts=tuple(mounts),
    )


def _external_dependency_mount(
    *, alias_name: str, source: Path, output: Path, mount_root: Path
) -> str:
    _require_owned_real_directory(source)
    source_text = str(source)
    if any(character in source_text for character in ("\n", "\r", ":")):
        raise ValueError(f"Paper dependency path cannot be mounted safely: {source}")
    output_container = f"/workspace/{output.relative_to(mount_root).as_posix()}"
    relative_target = os.path.relpath(source, start=output)
    container_target = posixpath.normpath(f"{output_container}/{relative_target}")
    protected_targets = (
        "/",
        "/bin",
        "/dev",
        "/etc",
        "/home/agent-home",
        "/lib",
        "/opt",
        "/proc",
        "/root",
        "/run",
        "/sys",
        "/usr",
        "/workspace",
    )
    if any(
        container_target == protected
        or container_target.startswith(f"{protected}/")
        for protected in protected_targets
    ):
        raise ValueError(
            f"External paper dependency has an unsafe container target: {alias_name}"
        )
    return f"{source_text}:{container_target}:ro"


def _valid_run_policy_marker(marker: Path) -> bool:
    try:
        marker_stat = marker.lstat()
    except (FileNotFoundError, OSError):
        return False
    if (
        not stat.S_ISREG(marker_stat.st_mode)
        or marker_stat.st_uid != os.getuid()
        or marker_stat.st_nlink != 1
    ):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("schema_version") == 1


def validate_existing_global_memory_files(global_memory_dir: Path) -> None:
    """Reject unsafe pre-existing files before memory initialization opens them."""
    memory_dir = global_memory_dir.expanduser().resolve()
    for name in WRITABLE_GLOBAL_MEMORY_FILES:
        path = memory_dir / name
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        _require_owned_regular_file(path)


def _require_owned_real_directory(path: Path) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Missing writable paper directory: {path}") from exc
    if not stat.S_ISDIR(path_stat.st_mode) or path_stat.st_uid != os.getuid():
        raise ValueError(f"Unsafe writable paper directory: {path}")


def _require_owned_regular_file(path: Path) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Missing regular writable paper path: {path}") from exc
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or path_stat.st_uid != os.getuid()
        or path_stat.st_nlink != 1
        or path_stat.st_size > MAX_WRITABLE_GLOBAL_MEMORY_BYTES
    ):
        raise ValueError(f"Unsafe writable paper file: {path}")


def marker_payload(
    *, output_dir: Path, task_dir: Path, results_dir: Path, global_memory_dir: Path
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "output_dir": _resolved(output_dir),
        "task_dir": _resolved(task_dir),
        "results_dir": _resolved(results_dir),
        "global_memory_dir": _resolved(global_memory_dir),
    }


def write_workspace_marker(
    *, output_dir: Path, task_dir: Path, results_dir: Path, global_memory_dir: Path
) -> Path:
    output_dir = output_dir.expanduser()
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(f"paper output must be a real directory: {output_dir}")
    marker_path = output_dir / MARKER_NAME
    if marker_path.is_symlink():
        raise ValueError(f"paper workspace marker must not be a symlink: {marker_path}")
    if marker_path.exists():
        raise ValueError(f"paper workspace marker already exists: {marker_path}")
    payload = marker_payload(
        output_dir=output_dir,
        task_dir=task_dir,
        results_dir=results_dir,
        global_memory_dir=global_memory_dir,
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{MARKER_NAME}.", dir=output_dir)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, marker_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return marker_path


def workspace_validation_errors(
    *,
    output_dir: Path,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path,
    ablations_dir: Path | None = None,
) -> list[str]:
    output_dir = output_dir.expanduser()
    if output_dir.is_symlink():
        return [f"paper output must not be a symlink: {output_dir}"]
    if not output_dir.is_dir():
        return [f"paper output is not a directory: {output_dir}"]

    errors: list[str] = []
    marker_path = output_dir / MARKER_NAME
    if marker_path.is_symlink() or not marker_path.is_file():
        errors.append(f"missing regular pipeline-owned marker: {marker_path}")
    else:
        try:
            actual = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid paper workspace marker: {exc}")
        else:
            expected = marker_payload(
                output_dir=output_dir,
                task_dir=task_dir,
                results_dir=results_dir,
                global_memory_dir=global_memory_dir,
            )
            if actual != expected:
                errors.append("paper workspace marker identity does not match this pipeline run")

    resolved_output_dir = output_dir.resolve()
    resolved_results_dir = results_dir.expanduser().resolve()
    resolved_task_dir = task_dir.expanduser().resolve()
    resolved_global_memory_dir = global_memory_dir.expanduser().resolve()
    resolved_ablations_dir = (
        ablations_dir.expanduser().resolve() if ablations_dir is not None else None
    )
    for path in output_dir.rglob("*"):
        if not path.is_symlink():
            continue
        relative_path = path.relative_to(output_dir)
        resolved_target = path.resolve()
        if relative_path == Path("global_memory"):
            target_allowed = resolved_target == resolved_global_memory_dir
        elif relative_path == Path("task"):
            target_allowed = resolved_target == resolved_task_dir or resolved_target.is_relative_to(
                resolved_results_dir
            )
        elif relative_path == Path("aggregated"):
            target_allowed = (
                resolved_target.is_dir()
                and not resolved_target.is_symlink()
                and resolved_target.stat().st_uid == os.getuid()
                and (
                    resolved_target.is_relative_to(resolved_results_dir)
                    or (
                        resolved_ablations_dir is not None
                        and resolved_target.is_relative_to(resolved_ablations_dir)
                    )
                )
            )
        elif relative_path == Path("extra_tasks"):
            target_allowed = (
                resolved_target.is_dir()
                and not resolved_target.is_symlink()
                and resolved_target.stat().st_uid == os.getuid()
                and (
                    resolved_target.is_relative_to(resolved_results_dir)
                    or (
                        resolved_ablations_dir is not None
                        and resolved_target.is_relative_to(resolved_ablations_dir)
                    )
                )
            )
        elif relative_path == Path("results"):
            merged_parallel_output = (
                resolved_target.parent == resolved_output_dir
                and resolved_target.name.startswith("_merged_parallel_")
                and resolved_target.is_dir()
                and not resolved_target.is_symlink()
                and resolved_target.stat().st_uid == os.getuid()
            )
            target_allowed = resolved_target.is_relative_to(
                resolved_results_dir
            ) or merged_parallel_output
        elif len(relative_path.parts) > 1 and relative_path.parts[0].startswith(
            "_merged_parallel_"
        ):
            target_allowed = resolved_target.is_relative_to(resolved_results_dir)
        else:
            target_allowed = False
        if not target_allowed:
            errors.append(
                f"unexpected symlink in paper workspace: {relative_path}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "validate"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--global-memory-dir", required=True, type=Path)
    parser.add_argument("--ablations-dir", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        write_workspace_marker(
            output_dir=args.output_dir,
            task_dir=args.task_dir,
            results_dir=args.results_dir,
            global_memory_dir=args.global_memory_dir,
        )
        return 0
    errors = workspace_validation_errors(
        output_dir=args.output_dir,
        task_dir=args.task_dir,
        results_dir=args.results_dir,
        global_memory_dir=args.global_memory_dir,
        ablations_dir=args.ablations_dir,
    )
    if errors:
        for error in errors:
            print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
