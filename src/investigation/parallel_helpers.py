"""Parallel worker planning, execution, and output aggregation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Union

from src.orchestrator import state as state_store

from .config import InvestigationConfig
from .container_gpus import (
    GPU_POOL_ENV,
    GPU_SELECTOR_ENV,
    discover_gpu_selector,
    gpu_pool_label,
    gpu_pool_size,
    settle_gpu_pool,
    validate_gpu_allocation,
    worker_gpu_selector,
)
from .constants import (
    CODEX_STREAM_DISCONNECT_MARKER,
    DEFAULT_INTERRUPT_GRACE_SECONDS,
    PARALLEL_WORKER_MEMORY_FILES,
    PARALLEL_WORKER_STATE_FILES,
)
from .event_logging import TERMINAL_WRITE_LOCK
from .files import confirm_prune_warning, evaluation_results_complete, resolve_path, utc_now
from .models import ParallelWorkerPlan, ParallelWorkerResult
from .program_spec import entrypoint_relpath_from_spec
from .task_failure import HOST_FAILURES

PopenFactory = Callable[..., subprocess.Popen[str]]
RegisterFn = Callable[[subprocess.Popen[str]], None]


def _stop_task_failure_workers(
    *, cfg: InvestigationConfig, worker_dirs: Iterable[Path], failure_code: int,
    terminate_all_active_processes_fn: Callable[..., list[int]],
    stop_container_session_fn: Callable[..., bool],
    build_container_session_name_fn: Callable[..., str],
    log_event_fn: LogFn, grace_seconds: int,
) -> None:
    """Stop sibling worker processes before removing their named containers."""
    def log_cleanup(event: str, **fields: Any) -> None:
        try:
            log_event_fn(event, **fields)
        except Exception:
            pass

    directories = list(worker_dirs)
    failure_category, failure_scope = HOST_FAILURES[failure_code]
    log_cleanup(
        "parallel_task_failure", failure_category=failure_category,
        failure_scope=failure_scope, action="stop_scheduling_and_cancel",
    )
    try:
        terminate_all_active_processes_fn(grace_seconds=grace_seconds)
    except Exception as exc:
        log_cleanup("parallel_task_failure_cleanup_failed", error_type=type(exc).__name__)
    if cfg.container.enabled:
        for worker_dir in directories:
            try:
                session = build_container_session_name_fn(
                    agent=cfg.agent.name, results_dir=worker_dir,
                )
                stop_container_session_fn(session, log_event_fn=log_event_fn)
            except Exception as exc:
                log_cleanup("parallel_task_failure_cleanup_failed", error_type=type(exc).__name__)


UnregisterFn = Callable[[subprocess.Popen[str]], None]
TerminateFn = Callable[..., None]
EnsureTextFn = Callable[[Optional[Union[str, bytes]]], str]
LogFn = Callable[..., None]
_ROUND_DIR_RE = re.compile(r"round_(\d+)")
_SEED_PROGRAM_RE = re.compile(r"paths\.seed_program=([^ ]+)")
_EVENT_RECORD_KEYS = {
    "timestamp",
    "source",
    "event",
    "phase",
    "worker_index",
    "session_id",
    "fields",
}
_MAX_EVENT_LINE_CHARS = 64 * 1024


def _worker_exec_env(plan: ParallelWorkerPlan | None = None) -> dict[str, str]:
    """Build env for parallel worker subprocess execution."""
    env = dict(os.environ)
    stream = getattr(sys, "stdout", None)
    stream_is_tty = bool(stream and hasattr(stream, "isatty") and stream.isatty())
    if stream_is_tty and "NO_COLOR" not in env and "FORCE_COLOR" not in env:
        # Worker stdout is piped in parallel mode; force ANSI so badges stay colored.
        env["FORCE_COLOR"] = "1"
    if plan is not None and plan.environment:
        env.update(plan.environment)
    return env


def settle_gpu_allocation(
    cfg: InvestigationConfig,
    environment: MutableMapping[str, str],
    *,
    discover: Callable[..., str] = discover_gpu_selector,
) -> str | None:
    """Settle ``AGENT_CONTAINER_GPUS`` for the run; returns the parallel pool, or ``None``.

    With ``container.gpu_count>0`` an explicit selector must name the GPUs the run needs
    and an unset one discovers them from ``nvidia-smi``. A single worker needs exactly
    ``gpu_count`` GPUs. ``execution.workers>1`` needs ``workers x gpu_count``: a pool the
    pipeline already exported as ``AGENT_CONTAINER_GPU_POOL`` is validated and reused;
    otherwise the settled selector becomes the pool and ``AGENT_CONTAINER_GPUS`` narrows
    to the first slice for this coordinating process.
    """
    gpu_count = cfg.container.gpu_count
    workers = cfg.execution.workers
    if gpu_count <= 0:
        return None
    if workers <= 1:
        settled = settle_gpu_pool(
            environment.get(GPU_SELECTOR_ENV), gpu_count, 1, discover=discover
        )
        assert settled is not None
        environment[GPU_SELECTOR_ENV] = settled[0]
        return None
    label = gpu_pool_label(gpu_count, workers)
    pool = environment.get(GPU_POOL_ENV)
    if pool is not None and pool.strip():
        validate_gpu_allocation(pool, gpu_pool_size(gpu_count, workers), label=label)
        return pool
    settled = settle_gpu_pool(
        environment.get(GPU_SELECTOR_ENV), gpu_count, workers, discover=discover
    )
    assert settled is not None
    pool, first_slice = settled
    environment[GPU_POOL_ENV] = pool
    environment[GPU_SELECTOR_ENV] = first_slice
    return pool


def worker_gpu_environment(
    cfg: InvestigationConfig,
    worker_index: int,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Environment overrides giving 1-based *worker_index* its own slice of the GPU pool."""
    source: Mapping[str, str] = os.environ if environment is None else environment
    pool = source.get(GPU_POOL_ENV)
    if cfg.container.gpu_count <= 0 or cfg.execution.workers <= 1 or not pool:
        return {}
    return {GPU_SELECTOR_ENV: worker_gpu_selector(pool, cfg.container.gpu_count, worker_index)}


def _worker_output_prefix(worker_index: int) -> str:
    """Render the per-worker streaming prefix."""
    return f"[worker_{worker_index:02d}] "


def _prefix_output(output: str, *, prefix: str) -> str:
    """Prefix each rendered line while keeping the raw captured output intact."""
    if not output or not prefix:
        return output
    return "".join(f"{prefix}{line}" for line in output.splitlines(keepends=True))


def _emit_worker_output(output: str, *, prefix: str = "") -> None:
    """Emit collected worker output without interleaving chunks."""
    if not output:
        return
    rendered = _prefix_output(output, prefix=prefix)
    with TERMINAL_WRITE_LOCK:
        if rendered.endswith("\n"):
            print(rendered, end="")
            return
        print(rendered)


def _valid_json_object_lines(output: str, *, worker_index: int) -> tuple[str, int]:
    """Keep validated worker event records; count private diagnostics."""
    valid_lines: list[str] = []
    suppressed = 0
    for line in output.splitlines(keepends=True):
        try:
            value = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except (json.JSONDecodeError, RecursionError, ValueError):
            suppressed += 1
            continue
        if not _valid_worker_event(value, worker_index=worker_index, line=line):
            suppressed += 1
            continue
        valid_lines.append(line)
    return "".join(valid_lines), suppressed


def _reject_json_constant(value: str) -> None:
    """Reject NaN and infinity tokens accepted by Python's permissive decoder."""
    raise ValueError(f"non-standard JSON constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _valid_worker_event(value: Any, *, worker_index: int, line: str) -> bool:
    if (
        len(line) > _MAX_EVENT_LINE_CHARS
        or not isinstance(value, dict)
        or set(value) != _EVENT_RECORD_KEYS
    ):
        return False
    return (
        value.get("source") == "wrapper"
        and value.get("worker_index") == worker_index
        and isinstance(value.get("timestamp"), str)
        and 0 < len(value["timestamp"]) <= 64
        and isinstance(value.get("event"), str)
        and 0 < len(value["event"]) <= 96
        and (value.get("phase") is None or isinstance(value.get("phase"), str))
        and isinstance(value.get("session_id"), str)
        and 0 < len(value["session_id"]) <= 160
        and isinstance(value.get("fields"), dict)
        and len(value["fields"]) <= 256
    )


def _stream_worker_output(
    stream: Any | None,
    *,
    ensure_text: EnsureTextFn,
    emit_output_fn: Callable[[str], None],
) -> str:
    """Stream process output line-by-line and return the raw combined text."""
    if stream is None:
        return ""

    chunks: list[str] = []
    try:
        for chunk in stream:
            text = ensure_text(chunk)
            if not text:
                continue
            chunks.append(text)
            emit_output_fn(text)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    return "".join(chunks)


def build_parallel_worker_command(
    *,
    cfg: InvestigationConfig,
    task_dir: Path,
    instruct_path: Path,
    results_dir: Path,
    global_memory_dir: Path,
    assigned_cycles: int,
    seed_program: Path | None = None,
    shared_memory_dir: Path | None = None,
    conclusions_dir: Path | None = None,
    worker_index: int = 0,
    resume: bool | None = None,
    global_cycle: int | None = None,
) -> list[str]:
    """Build parallel worker command using Hydra override syntax.

    ``global_cycle`` is the campaign-wide number of the worker's last assigned
    cycle; the worker reports its local cycles relative to it, over the
    campaign total.
    """
    cmd = [
        sys.executable,
        "-m",
        "src.investigation.cli",
        f"agent.name={cfg.agent.name}",
        f"paths.task_dir={task_dir}",
        f"paths.results_dir={results_dir}",
        f"paths.global_memory_dir={global_memory_dir}",
        f"paths.instruct={instruct_path}",
        f"execution.max_idea_cycles={assigned_cycles}",
        f"execution.run_timeout_seconds={cfg.execution.run_timeout_seconds}",
        f"execution.harness_evaluation_timeout_seconds={cfg.execution.harness_evaluation_timeout_seconds}",
        f"execution.idea_timeout_seconds={cfg.execution.idea_timeout_seconds}",
        f"execution.max_idea_generation_attempts={cfg.execution.max_idea_generation_attempts}",
        f"execution.run_timeout_grace_seconds={cfg.execution.run_timeout_grace_seconds}",
        f"execution.max_program_evals_per_attempt={cfg.execution.max_program_evals_per_attempt}",
        f"execution.harness_evaluated_attempts={'true' if cfg.execution.harness_evaluated_attempts else 'false'}",
        f"execution.gpu_evaluation={'true' if cfg.execution.gpu_evaluation else 'false'}",
        f"execution.isolate_worker_memory={'true' if cfg.execution.isolate_worker_memory else 'false'}",
        f"logging.format={cfg.logging.format}",
        f"logging.verbose={'true' if cfg.logging.verbose else 'false'}",
        f"logging.heartbeat_seconds={cfg.logging.heartbeat_seconds}",
        "logging.show_run_header=false",
        "execution.workers=1",
        f"execution.worker_index={worker_index}",
        # Worker subtrees are never checkpointed: the parent commits the run
        # root once every worker has joined.
        "versioning.enabled=false",
    ]
    if resume is not None:
        cmd.append(f"execution.resume={'true' if resume else 'false'}")
    if global_cycle is not None:
        cmd.append(f"execution.cycle_offset={global_cycle - assigned_cycles}")
        cmd.append(f"execution.total_cycles={cfg.execution.max_idea_cycles}")
    if cfg.agent.model:
        cmd.append(f"agent.model={cfg.agent.model}")
    if cfg.agent.profile:
        cmd.append(f"agent.profile={cfg.agent.profile}")
    if cfg.agent.reasoning_effort:
        cmd.append(f"agent.reasoning_effort={cfg.agent.reasoning_effort}")
    if cfg.agent.pool:
        cmd.append(f"agent.pool=[{','.join(cfg.agent.pool)}]")
    for agent_name, model in sorted(cfg.agent.models.items()):
        cmd.append(f"agent.models.{agent_name}={model}")
    if seed_program is not None:
        cmd.append(f"paths.seed_program={seed_program}")
    if shared_memory_dir is not None:
        cmd.append(f"paths.shared_memory_dir={shared_memory_dir}")
    if conclusions_dir is not None:
        cmd.append(f"paths.conclusions_dir={conclusions_dir}")
    if cfg.container.enabled:
        cmd.append("container.enabled=true")
        cmd.append(f"container.image={cfg.container.image}")
        cmd.append(f"container.network={cfg.container.network}")
        # Resolve ~ to absolute path — Hydra's parser rejects bare tildes.
        _auth_root = str(Path(cfg.container.auth_root).expanduser())
        cmd.append(f"container.auth_root={_auth_root}")
        if cfg.container.env_file.strip():
            cmd.append(f"container.env_file={cfg.container.env_file}")
    return cmd


def _command_flag_value(command: list[str], flag: str) -> str | None:
    """Extract a Hydra override value from the command list."""
    prefix = f"{flag}="
    for arg in command:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _parallel_worker_network_hint(plan: ParallelWorkerPlan, output: str) -> str | None:
    """Handle parallel worker network hint logic for this module."""
    if CODEX_STREAM_DISCONNECT_MARKER not in output:
        return None
    if _command_flag_value(plan.command, "container.enabled") != "true":
        return None
    if _command_flag_value(plan.command, "agent.name") != "codex":
        return None
    container_network = _command_flag_value(plan.command, "container.network")
    if container_network != "bridge":
        return None
    return (
        "[wrapper] hint=codex_container_bridge_egress "
        "Codex could not maintain stream transport to chatgpt.com from Docker bridge. "
        "Try container.network=host, and ensure HTTPS_PROXY/HTTP_PROXY/NO_PROXY are set "
        "in container.env_file when your server needs a proxy."
    )


def run_parallel_worker(
    plan: ParallelWorkerPlan,
    *,
    popen_factory: PopenFactory,
    register_subprocess: RegisterFn,
    unregister_subprocess: UnregisterFn,
    terminate_process_group: TerminateFn,
    ensure_text: EnsureTextFn,
    interrupt_grace_seconds: int = DEFAULT_INTERRUPT_GRACE_SECONDS,
) -> ParallelWorkerResult:
    """Run parallel worker."""
    plan.results_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        plan.log_path if plan.log_path is not None else plan.results_dir / "parallel_worker.log"
    )
    started_at = time.monotonic()
    proc: subprocess.Popen[str] | None = None
    stderr_thread: threading.Thread | None = None
    output_chunks: list[str] = []
    output_lock = threading.Lock()
    worker_log_format = _command_flag_value(plan.command, "logging.format")
    prefix = (
        "" if worker_log_format in {"human", "json"} else _worker_output_prefix(plan.worker_index)
    )
    suppressed_output_lines = 0

    def _record_output(text: str, *, emit: bool = True) -> None:
        nonlocal suppressed_output_lines
        if not text:
            return
        with output_lock:
            output_chunks.append(text)
        if emit:
            rendered = text
            if worker_log_format == "json":
                rendered, suppressed = _valid_json_object_lines(
                    text,
                    worker_index=plan.worker_index,
                )
                suppressed_output_lines += suppressed
            _emit_worker_output(rendered, prefix=prefix)

    try:
        proc = popen_factory(
            plan.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=_worker_exec_env(plan),
        )
        register_subprocess(proc)
        stderr_thread = threading.Thread(
            target=_stream_worker_output,
            kwargs={
                "stream": getattr(proc, "stderr", None),
                "ensure_text": ensure_text,
                "emit_output_fn": lambda text: _record_output(text, emit=False),
            },
            name=f"worker-{plan.worker_index:02d}-stderr",
        )
        stderr_thread.start()
        _stream_worker_output(
            proc.stdout,
            ensure_text=ensure_text,
            emit_output_fn=_record_output,
        )
        exit_code = proc.wait()
        stderr_thread.join()
    except KeyboardInterrupt:
        if proc is not None:
            terminate_process_group(proc, grace_seconds=interrupt_grace_seconds)
            _stream_worker_output(
                getattr(proc, "stdout", None),
                ensure_text=ensure_text,
                emit_output_fn=_record_output,
            )
            proc.wait()
            if stderr_thread is not None:
                stderr_thread.join()
        _record_output(
            "[wrapper] parallel worker interrupted; terminated process group.\n",
            emit=worker_log_format not in {"human", "json"},
        )
        exit_code = 130
    except OSError as exc:
        _record_output(
            f"[wrapper] failed to launch worker {plan.worker_index}: {exc}\n",
            emit=worker_log_format not in {"human", "json"},
        )
        exit_code = 127
    finally:
        if proc is not None:
            unregister_subprocess(proc)
        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=1)

    output = "".join(output_chunks)
    if exit_code != 0:
        hint = _parallel_worker_network_hint(plan, output)
        if hint:
            hint_text = hint if hint.endswith("\n") else f"{hint}\n"
            _record_output(hint_text, emit=worker_log_format not in {"human", "json"})
            output = "".join(output_chunks)

    state_store.write_private_text(log_path, output)
    return ParallelWorkerResult(
        plan=plan,
        exit_code=exit_code,
        log_path=log_path,
        elapsed_seconds=time.monotonic() - started_at,
        output_tail=output[-1200:],
        suppressed_output_lines=suppressed_output_lines,
    )


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read jsonl records."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def _positive_int(value: Any) -> int | None:
    """Handle positive int logic for this module."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed > 0 else None
    return None


def _map_cycle_index(local_cycle: Any, cycle_start: int) -> int | None:
    """Map cycle index."""
    local = _positive_int(local_cycle)
    if local is None:
        return None
    return cycle_start + local - 1


def _map_worker_cycle(plan: ParallelWorkerPlan, local_cycle: Any) -> int | None:
    """Map a worker-local cycle to its global campaign cycle."""
    local = _positive_int(local_cycle)
    if local is None:
        return None
    if plan.cycle_map is not None:
        return plan.cycle_map.get(local)
    return _map_cycle_index(local, plan.cycle_start)


def _cleanup_parallel_worker_state_files(
    worker_results: list[ParallelWorkerResult],
    *,
    log_event_fn: LogFn,
) -> None:
    """Clean up parallel worker state files."""
    for worker in sorted(worker_results, key=lambda item: item.plan.worker_index):
        removed: list[str] = []
        for rel in _worker_state_files_to_cleanup(worker.plan):
            candidate = worker.plan.results_dir / rel
            try:
                if candidate.is_file() or candidate.is_symlink():
                    candidate.unlink()
                    removed.append(rel)
            except OSError:
                continue
        if removed:
            log_event_fn(
                "parallel_worker_state_cleanup",
                worker=worker.plan.worker_index,
                removed=removed,
            )


def _worker_state_files_to_cleanup(plan: ParallelWorkerPlan) -> tuple[str, ...]:
    if _command_flag_value(plan.command, "execution.isolate_worker_memory") != "true":
        return PARALLEL_WORKER_STATE_FILES
    return tuple(
        path for path in PARALLEL_WORKER_STATE_FILES if path not in PARALLEL_WORKER_MEMORY_FILES
    )


def _normalized_best_manifest(
    manifest_raw: dict[str, Any] | None,
    *,
    seed_program_dir: Path,
) -> tuple[dict[str, Any], float] | None:
    """Normalize a correct best-manifest payload when it has a numeric score."""
    if not isinstance(manifest_raw, dict):
        return None
    if manifest_raw.get("correct") is not True:
        return None
    score_raw = manifest_raw.get("combined_score")
    if (
        isinstance(score_raw, bool)
        or not isinstance(score_raw, (int, float))
        or not math.isfinite(float(score_raw))
    ):
        return None

    score = float(score_raw)
    program_dir_raw = manifest_raw.get("program_dir")
    entrypoint_path_raw = manifest_raw.get("entrypoint_path")
    if not (isinstance(program_dir_raw, str) and program_dir_raw.strip()):
        program_path_raw = manifest_raw.get("program_path")
        if isinstance(program_path_raw, str) and program_path_raw.strip():
            program_path = resolve_path(program_path_raw)
            program_dir_raw = str(program_path.parent)
            entrypoint_path_raw = program_path.relative_to(program_path.parent).as_posix()

    program_dir = (
        str(resolve_path(program_dir_raw))
        if isinstance(program_dir_raw, str) and program_dir_raw.strip()
        else str(seed_program_dir)
    )
    entrypoint_path = (
        str(entrypoint_path_raw).strip()
        if isinstance(entrypoint_path_raw, str) and entrypoint_path_raw.strip()
        else (entrypoint_relpath_from_spec(Path(program_dir), required=False) or "src/program.py")
    )
    normalized = {
        "program_dir": program_dir,
        "entrypoint_path": entrypoint_path,
        "combined_score": score,
        "correct": True,
        "source_attempt_id": manifest_raw.get("source_attempt_id"),
        "updated_at": utc_now(),
    }
    if "parallel_worker_index" in manifest_raw:
        normalized["parallel_worker_index"] = manifest_raw.get("parallel_worker_index")
    return normalized, score


def _parallel_archive_dirs(results_dir: Path) -> list[Path]:
    """Return archived parallel round directories sorted by round index."""
    pruned_root = results_dir / "_pruned"
    if not pruned_root.is_dir():
        return []
    return sorted(
        [
            child
            for child in pruned_root.iterdir()
            if child.is_dir() and child.name.startswith("round_")
        ],
        key=lambda path: path.name,
    )


def _round_index_from_path(path_like: Any) -> int | None:
    """Extract round_NNN index from a results/log path-like value."""
    if not isinstance(path_like, (str, Path)):
        return None
    match = _ROUND_DIR_RE.search(str(path_like))
    if match is None:
        return None
    return int(match.group(1))


def _worker_entries_from_results(
    worker_results: list[ParallelWorkerResult],
) -> list[dict[str, Any]]:
    """Render worker-result metadata into summary entries."""
    return [
        {
            "worker_index": worker.plan.worker_index,
            "assigned_cycles": worker.plan.assigned_cycles,
            "cycle_start": worker.plan.cycle_start,
            "cycle_end": worker.plan.cycle_end,
            "cycle_map": worker.plan.cycle_map,
            "results_dir": str(worker.plan.results_dir),
            "log_path": str(worker.log_path),
            "exit_code": worker.exit_code,
            "elapsed_seconds": round(worker.elapsed_seconds, 3),
            "suppressed_output_lines": worker.suppressed_output_lines,
        }
        for worker in worker_results
    ]


def _sort_worker_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort summary worker entries deterministically by round then worker index."""
    return sorted(
        entries,
        key=lambda entry: (
            _round_index_from_path(entry.get("results_dir")) or -1,
            _positive_int(entry.get("worker_index")) or 0,
            _positive_int(entry.get("cycle_start")) or 0,
        ),
    )


def _archived_worker_entries(
    *,
    results_dir: Path,
    fallback_summary_raw: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Load worker summary entries for archived rounds."""
    entries: list[dict[str, Any]] = []
    fallback_workers = (
        fallback_summary_raw.get("workers", []) if isinstance(fallback_summary_raw, dict) else []
    )
    for archive_dir in _parallel_archive_dirs(results_dir):
        summary_raw = state_store.load_json(archive_dir / "parallel_summary.json")
        if isinstance(summary_raw, dict) and isinstance(summary_raw.get("workers"), list):
            entries.extend(worker for worker in summary_raw["workers"] if isinstance(worker, dict))
            continue

        archive_round = _round_index_from_path(archive_dir)
        if archive_round is None:
            continue
        for worker in fallback_workers:
            if not isinstance(worker, dict):
                continue
            if _round_index_from_path(worker.get("results_dir")) == archive_round:
                entries.append(dict(worker))
    return _sort_worker_entries(entries)


def _scan_program_manifest(
    program_dir: Path,
    *,
    seed_program_dir: Path,
) -> tuple[dict[str, Any], float] | None:
    """Inspect one evaluated program directory and return a normalized manifest."""
    if not evaluation_results_complete(program_dir):
        return None
    metrics_raw = state_store.load_json(program_dir / "metrics.json")
    correct_raw = state_store.load_json(program_dir / "correct.json")

    score_raw = None
    if isinstance(metrics_raw, dict):
        score_raw = metrics_raw.get("combined_score")
    if not isinstance(score_raw, (int, float)):
        return None

    correct_value = None
    if isinstance(correct_raw, dict):
        correct_value = correct_raw.get("correct")
    if correct_value is None and isinstance(metrics_raw, dict):
        correct_value = metrics_raw.get("correct")
    if correct_value is not True:
        return None

    manifest = {
        "program_dir": str(program_dir.resolve()),
        "entrypoint_path": entrypoint_relpath_from_spec(program_dir, required=False)
        or "src/program.py",
        "combined_score": float(score_raw),
        "correct": True,
        "source_attempt_id": None,
        "updated_at": utc_now(),
    }
    normalized = _normalized_best_manifest(manifest, seed_program_dir=seed_program_dir)
    return normalized


def _best_manifest_from_worker_round_dir(
    worker_round_dir: Path,
    *,
    seed_program_dir: Path,
) -> tuple[dict[str, Any], float] | None:
    """Reconstruct the best valid program manifest for one worker round."""
    ideas_dir = worker_round_dir / "ideas"
    if not ideas_dir.is_dir():
        return None

    best: tuple[dict[str, Any], float] | None = None
    for program_dir in ideas_dir.glob("*/program_*"):
        if not program_dir.is_dir():
            continue
        candidate = _scan_program_manifest(program_dir, seed_program_dir=seed_program_dir)
        if candidate is None:
            continue
        if best is None or candidate[1] > best[1]:
            best = candidate
    return best


def _best_manifest_from_completed_rounds(
    *,
    results_dir: Path,
    seed_program_dir: Path,
    max_round_index_exclusive: int | None = None,
    include_root_manifest: bool = True,
) -> tuple[dict[str, Any], float] | None:
    """Reconstruct the best valid manifest across archived/current completed rounds."""
    best = (
        _normalized_best_manifest(
            state_store.load_json(results_dir / state_store.BEST_MANIFEST_FILE),
            seed_program_dir=seed_program_dir,
        )
        if include_root_manifest
        else None
    )
    for worker_round_dir in sorted(results_dir.glob("worker_*/round_*")):
        round_index = _round_index_from_path(worker_round_dir)
        if (
            max_round_index_exclusive is not None
            and round_index is not None
            and round_index >= max_round_index_exclusive
        ):
            continue
        candidate = _best_manifest_from_worker_round_dir(
            worker_round_dir, seed_program_dir=seed_program_dir
        )
        if candidate is None:
            continue
        if best is None or candidate[1] > best[1]:
            best = candidate
    return best


def _extract_seed_program_from_worker_log(log_path: Path) -> Path | None:
    """Parse the next-round seed_program from a saved worker log."""
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _SEED_PROGRAM_RE.search(text)
    if match is None:
        return None
    return resolve_path(match.group(1))


def _resume_round_seed_map(
    *,
    results_dir: Path,
    next_round_index: int,
    last_successful_round_index: int,
    workers: int,
    default_seed: Path,
) -> dict[int, Path]:
    """Resolve per-worker seeds for the resumed round."""
    seeds: dict[int, Path] = {}
    for worker_index in range(1, workers + 1):
        next_log = (
            results_dir
            / f"worker_{worker_index:02d}"
            / f"round_{next_round_index:03d}"
            / "parallel_worker.log"
        )
        parsed_seed = _extract_seed_program_from_worker_log(next_log)
        if parsed_seed is not None:
            seeds[worker_index] = parsed_seed
            continue

        prior_round_dir = (
            results_dir / f"worker_{worker_index:02d}" / f"round_{last_successful_round_index:03d}"
        )
        reconstructed = _best_manifest_from_worker_round_dir(
            prior_round_dir, seed_program_dir=default_seed
        )
        if reconstructed is not None:
            seeds[worker_index] = resolve_path(reconstructed[0]["program_dir"])
            continue

        seeds[worker_index] = default_seed
    return seeds


def _parallel_resume_state_is_valid(
    resume_state: dict[str, Any],
    *,
    total_rounds: int,
    cycles_per_round: int,
    workers: int,
) -> bool:
    """Return whether persisted parallel resume-state matches current config."""
    return (
        _positive_int(resume_state.get("total_rounds")) == total_rounds
        and _positive_int(resume_state.get("cycles_per_round")) == cycles_per_round
        and _positive_int(resume_state.get("workers")) == workers
    )


def _validate_parallel_resume_memory_isolation(
    resume_state: dict[str, Any],
    *,
    isolate_worker_memory: bool,
) -> None:
    """Reject persisted campaigns that used a different memory-sharing mode."""
    persisted_value = resume_state.get("isolate_worker_memory", False)
    if not isinstance(persisted_value, bool):
        raise RuntimeError("parallel resume-state has invalid execution.isolate_worker_memory")
    if persisted_value != isolate_worker_memory:
        raise RuntimeError(
            "parallel resume-state execution.isolate_worker_memory does not match "
            "the current config"
        )


def _load_parallel_resume_checkpoint(results_dir: Path) -> dict[str, Any] | None:
    checkpoint_path = state_store.parallel_resume_state_path(results_dir)
    resume_state = state_store.load_parallel_resume_state(results_dir)
    if resume_state is None and checkpoint_path.exists():
        raise RuntimeError("parallel resume-state is invalid or unreadable")
    return resume_state


def _deserialize_worker_seeds(
    worker_seeds_raw: Any,
) -> dict[int, Path]:
    """Deserialize worker seed paths stored in resume-state."""
    if not isinstance(worker_seeds_raw, dict):
        return {}
    seeds: dict[int, Path] = {}
    for worker_key, seed_path in worker_seeds_raw.items():
        worker_index = _positive_int(worker_key)
        if worker_index is None or not isinstance(seed_path, str) or not seed_path.strip():
            continue
        seeds[worker_index] = resolve_path(seed_path)
    return seeds


def _serialize_worker_seeds(worker_seeds: dict[int, Path]) -> dict[str, str]:
    """Serialize worker seed mapping for persisted resume-state."""
    return {
        str(worker_index): str(seed.resolve())
        for worker_index, seed in sorted(worker_seeds.items())
    }


def _deserialize_worker_seed_scores(
    raw: Any,
) -> dict[int, float]:
    """Deserialize worker seed scores stored in resume-state."""
    if not isinstance(raw, dict):
        return {}
    scores: dict[int, float] = {}
    for key, value in raw.items():
        worker_index = _positive_int(key)
        if worker_index is not None and isinstance(value, (int, float)):
            scores[worker_index] = float(value)
    return scores


def _serialize_worker_seed_scores(scores: dict[int, float]) -> dict[str, float]:
    """Serialize worker seed scores for persisted resume-state."""
    return {str(worker_index): score for worker_index, score in sorted(scores.items())}


def _cleanup_stale_parallel_round_state(
    *,
    results_dir: Path,
    next_round_index: int,
    log_event_fn: LogFn,
) -> None:
    """Remove stale round artifacts that will be replayed on resume."""
    removed: list[str] = []
    for archive_dir in _parallel_archive_dirs(results_dir):
        archive_round = _round_index_from_path(archive_dir)
        if archive_round is not None and archive_round >= next_round_index:
            shutil.rmtree(archive_dir, ignore_errors=True)
            removed.append(str(archive_dir))

    shared_root = results_dir / "_shared_memory"
    if shared_root.is_dir():
        for shared_dir in sorted(shared_root.glob("round_*")):
            shared_round = _round_index_from_path(shared_dir)
            if shared_round is not None and shared_round >= next_round_index:
                shutil.rmtree(shared_dir, ignore_errors=True)
                removed.append(str(shared_dir))

    for worker_root in sorted(results_dir.glob("worker_*")):
        if not worker_root.is_dir():
            continue
        for round_dir in sorted(worker_root.glob("round_*")):
            round_index = _round_index_from_path(round_dir)
            if round_index is not None and round_index >= next_round_index:
                shutil.rmtree(round_dir, ignore_errors=True)
                removed.append(str(round_dir))

    if removed:
        log_event_fn(
            "parallel_resume_cleanup",
            next_round_index=next_round_index,
            removed=removed,
        )


def _restore_parallel_root_state_for_resume(
    *,
    results_dir: Path,
    global_memory_dir: Path,
    seed_program_dir: Path,
    next_round_index: int,
    best_manifest: dict[str, Any] | None,
    worker_entries: list[dict[str, Any]],
) -> None:
    """Hydrate the root run state to the last healthy round boundary."""
    state_store.ensure_histories(results_dir)
    state_store.ideas_history_path(results_dir).write_text("", encoding="utf-8")
    state_store.attempts_history_path(results_dir).write_text("", encoding="utf-8")
    state_store.idea_generation_failures_path(results_dir).write_text("", encoding="utf-8")

    completed_ideas: list[dict[str, Any]] = []
    completed_attempts: list[dict[str, Any]] = []
    completed_idea_failures: list[dict[str, Any]] = []
    last_idea_id = None
    current_cycle = 0
    for archive_dir in _parallel_archive_dirs(results_dir):
        round_index = _round_index_from_path(archive_dir)
        if round_index is None or round_index >= next_round_index:
            continue
        for record in _read_jsonl_records(archive_dir / state_store.IDEAS_HISTORY_FILE):
            completed_ideas.append(record)
            idea_id = record.get("idea_id")
            cycle_idx = _positive_int(record.get("cycle_index")) or 0
            if cycle_idx >= current_cycle:
                current_cycle = cycle_idx
                if isinstance(idea_id, str) and idea_id.strip():
                    last_idea_id = idea_id
        completed_attempts.extend(
            _read_jsonl_records(archive_dir / state_store.ATTEMPTS_HISTORY_FILE)
        )
        for record in _read_jsonl_records(archive_dir / state_store.IDEA_GENERATION_FAILURES_FILE):
            completed_idea_failures.append(record)
            generation_attempt = _positive_int(record.get("generation_attempt"))
            max_attempts = _positive_int(record.get("max_attempts"))
            if generation_attempt is not None and generation_attempt == max_attempts:
                cycle_idx = _positive_int(record.get("cycle_index")) or 0
                current_cycle = max(current_cycle, cycle_idx)

    manifest = best_manifest
    if manifest is None:
        manifest = {
            "program_dir": str(seed_program_dir),
            "entrypoint_path": entrypoint_relpath_from_spec(seed_program_dir, required=False)
            or "src/program.py",
            "combined_score": None,
            "correct": None,
            "source_attempt_id": "parallel_bootstrap",
            "updated_at": utc_now(),
        }
    state_store.save_best_manifest(results_dir, dict(manifest))

    outer_state = {
        "version": 1,
        "created_at": utc_now(),
        "current_cycle": current_cycle,
        "failure_streak": 0,
        "last_idea_id": last_idea_id,
        "best_program_dir": str(manifest.get("program_dir", seed_program_dir)),
        "best_combined_score": manifest.get("combined_score"),
        "best_correct": manifest.get("correct"),
        "prune_chapter": next_round_index,
    }
    state_store.save_outer_state(results_dir, outer_state)

    summary = {
        "created_at": utc_now(),
        "worker_count": len(worker_entries),
        "failed_worker_count": sum(1 for entry in worker_entries if entry.get("exit_code") != 0),
        "idea_count": len(completed_ideas),
        "attempt_count": len(completed_attempts),
        "idea_generation_failure_count": len(completed_idea_failures),
        "best_program_dir": manifest.get("program_dir"),
        "best_entrypoint_path": manifest.get("entrypoint_path"),
        "best_combined_score": manifest.get("combined_score"),
        "workers": _sort_worker_entries(worker_entries),
    }
    state_store.write_json(results_dir / "parallel_summary.json", summary)

    if next_round_index > 0:
        state_store.reset_parallel_global_memory_for_round(
            global_memory_dir,
            round_index=next_round_index - 1,
        )


def _seed_round_shared_memory(
    *,
    results_dir: Path,
    round_index: int,
    source_dir: Path,
) -> None:
    """Seed the next round's shared memory from the archived checkpoint."""
    target_dir = results_dir / "_shared_memory" / f"round_{round_index:03d}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname in (
        state_store.GLOBAL_MEMORY_MD_FILE,
        state_store.GLOBAL_MEMORY_JSONL_FILE,
    ):
        src = source_dir / fname
        if src.is_file():
            shutil.copy2(src, target_dir / fname)
    state_store.ensure_global_memory_files(target_dir)


def _persist_parallel_resume_state(
    *,
    results_dir: Path,
    total_rounds: int,
    cycles_per_round: int,
    workers: int,
    next_round_index: int,
    global_cycle_offset: int,
    worker_seeds: dict[int, Path],
    worker_seed_scores: dict[int, float],
    shared_memory_source: Path | None,
    isolate_worker_memory: bool,
) -> None:
    """Persist round-boundary resume state for future parallel resumes."""
    state_store.save_parallel_resume_state(
        results_dir,
        {
            "version": 1,
            "created_at": utc_now(),
            "mode": "parallel_pruned",
            "total_rounds": total_rounds,
            "cycles_per_round": cycles_per_round,
            "workers": workers,
            "next_round_index": next_round_index,
            "global_cycle_offset": global_cycle_offset,
            "completed_rounds": list(range(next_round_index)),
            "worker_seeds": _serialize_worker_seeds(worker_seeds),
            "worker_seed_scores": _serialize_worker_seed_scores(worker_seed_scores),
            "isolate_worker_memory": isolate_worker_memory,
            "shared_memory_source": (
                str(shared_memory_source.resolve()) if shared_memory_source is not None else None
            ),
            "resume_basis": "persisted",
        },
    )


def _load_or_reconstruct_parallel_resume_state(
    *,
    cfg: InvestigationConfig,
    results_dir: Path,
    global_memory_dir: Path,
    seed_program_dir: Path,
    total_rounds: int,
    cycles_per_round: int,
    workers: int,
    log_event_fn: LogFn,
) -> dict[str, Any]:
    """Load persisted parallel resume-state or reconstruct it for legacy runs."""
    persisted = _load_parallel_resume_checkpoint(results_dir)
    if persisted is not None:
        _validate_parallel_resume_memory_isolation(
            persisted,
            isolate_worker_memory=cfg.execution.isolate_worker_memory,
        )
    if persisted is not None and _parallel_resume_state_is_valid(
        persisted,
        total_rounds=total_rounds,
        cycles_per_round=cycles_per_round,
        workers=workers,
    ):
        next_round_index = _positive_int(persisted.get("next_round_index")) or 0
        global_cycle_offset = _positive_int(persisted.get("global_cycle_offset")) or 1
        worker_seeds = _deserialize_worker_seeds(persisted.get("worker_seeds"))
        if len(worker_seeds) != workers:
            raise RuntimeError("parallel resume-state is missing one or more worker seeds")
        worker_seed_scores = _deserialize_worker_seed_scores(persisted.get("worker_seed_scores"))

        shared_memory_source_raw = persisted.get("shared_memory_source")
        persisted_shared_memory_source = (
            resolve_path(shared_memory_source_raw)
            if isinstance(shared_memory_source_raw, str) and shared_memory_source_raw.strip()
            else None
        )
        root_summary_raw = state_store.load_json(results_dir / "parallel_summary.json")
        archived_workers = _archived_worker_entries(
            results_dir=results_dir,
            fallback_summary_raw=root_summary_raw if isinstance(root_summary_raw, dict) else None,
        )
        best = (
            _best_manifest_from_completed_rounds(
                results_dir=results_dir,
                seed_program_dir=seed_program_dir,
                max_round_index_exclusive=next_round_index,
                include_root_manifest=False,
            )
            if next_round_index < total_rounds
            else _normalized_best_manifest(
                state_store.load_json(results_dir / state_store.BEST_MANIFEST_FILE),
                seed_program_dir=seed_program_dir,
            )
        )
        if best is None:
            best = _best_manifest_from_completed_rounds(
                results_dir=results_dir,
                seed_program_dir=seed_program_dir,
            )
        if next_round_index < total_rounds:
            _cleanup_stale_parallel_round_state(
                results_dir=results_dir,
                next_round_index=next_round_index,
                log_event_fn=log_event_fn,
            )
            _restore_parallel_root_state_for_resume(
                results_dir=results_dir,
                global_memory_dir=global_memory_dir,
                seed_program_dir=seed_program_dir,
                next_round_index=next_round_index,
                best_manifest=best[0] if best is not None else None,
                worker_entries=archived_workers,
            )
        return {
            "next_round_index": next_round_index,
            "global_cycle_offset": global_cycle_offset,
            "worker_seeds": worker_seeds,
            "worker_seed_scores": worker_seed_scores,
            "shared_memory_source": persisted_shared_memory_source,
            "best_manifest": best[0] if best is not None else None,
            "worker_entries": archived_workers,
            "resume_basis": "persisted",
        }

    root_summary_raw = state_store.load_json(results_dir / "parallel_summary.json")
    summary_workers = (
        root_summary_raw.get("workers", []) if isinstance(root_summary_raw, dict) else []
    )
    round_status: dict[int, list[int]] = {}
    for worker in summary_workers:
        if not isinstance(worker, dict):
            continue
        round_index = _round_index_from_path(worker.get("results_dir"))
        exit_code = worker.get("exit_code")
        if round_index is None or not isinstance(exit_code, int):
            continue
        round_status.setdefault(round_index, []).append(exit_code)

    archived_rounds = _parallel_archive_dirs(results_dir)
    healthy_archived_rounds: list[int] = []
    for archive_dir in archived_rounds:
        round_index = _round_index_from_path(archive_dir)
        if round_index is None:
            continue
        exits = round_status.get(round_index)
        if exits is None:
            healthy_archived_rounds.append(round_index)
            continue
        if len(exits) >= workers and all(exit_code == 0 for exit_code in exits[:workers]):
            healthy_archived_rounds.append(round_index)

    last_successful_round_index = max(healthy_archived_rounds, default=-1)
    next_round_index = last_successful_round_index + 1
    global_cycle_offset = (next_round_index * cycles_per_round) + 1

    worker_seeds = (
        _resume_round_seed_map(
            results_dir=results_dir,
            next_round_index=next_round_index,
            last_successful_round_index=last_successful_round_index,
            workers=workers,
            default_seed=seed_program_dir,
        )
        if next_round_index > 0
        else {worker_index: seed_program_dir for worker_index in range(1, workers + 1)}
    )

    shared_memory_source: Path | None = None
    if last_successful_round_index >= 0:
        shared_memory_source = (
            results_dir / "_pruned" / f"round_{last_successful_round_index:03d}" / "_shared_memory"
        )
        if not shared_memory_source.is_dir():
            fallback = results_dir / "_shared_memory" / f"round_{last_successful_round_index:03d}"
            if fallback.is_dir():
                shared_memory_source = fallback
            else:
                raise RuntimeError(
                    "parallel resume could not find archived shared memory for the last "
                    f"successful round: {shared_memory_source}"
                )

    archived_workers = _archived_worker_entries(
        results_dir=results_dir,
        fallback_summary_raw=root_summary_raw if isinstance(root_summary_raw, dict) else None,
    )
    best = _best_manifest_from_completed_rounds(
        results_dir=results_dir,
        seed_program_dir=seed_program_dir,
        max_round_index_exclusive=next_round_index,
        include_root_manifest=False,
    )
    if next_round_index < total_rounds:
        _cleanup_stale_parallel_round_state(
            results_dir=results_dir,
            next_round_index=next_round_index,
            log_event_fn=log_event_fn,
        )
        archived_workers = _archived_worker_entries(
            results_dir=results_dir,
            fallback_summary_raw=root_summary_raw if isinstance(root_summary_raw, dict) else None,
        )
        _restore_parallel_root_state_for_resume(
            results_dir=results_dir,
            global_memory_dir=global_memory_dir,
            seed_program_dir=seed_program_dir,
            next_round_index=next_round_index,
            best_manifest=best[0] if best is not None else None,
            worker_entries=archived_workers,
        )

    # Legacy reconstruction cannot recover per-worker scores; start empty so
    # the first new round's result is always accepted.
    worker_seed_scores = _deserialize_worker_seed_scores(None)

    state_store.save_parallel_resume_state(
        results_dir,
        {
            "version": 1,
            "created_at": utc_now(),
            "mode": "parallel_pruned",
            "total_rounds": total_rounds,
            "cycles_per_round": cycles_per_round,
            "workers": workers,
            "next_round_index": next_round_index,
            "global_cycle_offset": global_cycle_offset,
            "completed_rounds": list(range(next_round_index)),
            "worker_seeds": _serialize_worker_seeds(worker_seeds),
            "worker_seed_scores": _serialize_worker_seed_scores(worker_seed_scores),
            "isolate_worker_memory": cfg.execution.isolate_worker_memory,
            "shared_memory_source": (
                str(shared_memory_source.resolve()) if shared_memory_source is not None else None
            ),
            "resume_basis": "legacy_reconstructed",
        },
    )
    return {
        "next_round_index": next_round_index,
        "global_cycle_offset": global_cycle_offset,
        "worker_seeds": worker_seeds,
        "worker_seed_scores": worker_seed_scores,
        "shared_memory_source": shared_memory_source,
        "best_manifest": best[0] if best is not None else None,
        "worker_entries": archived_workers,
        "resume_basis": "legacy_reconstructed",
    }


def build_parallel_campaign_summary(
    *,
    results_dir: Path,
    seed_program_dir: Path,
    worker_results: list[ParallelWorkerResult],
) -> dict[str, Any]:
    """Build the root campaign summary across archived rounds and current state."""
    archived_idea_count = 0
    archived_attempt_count = 0
    archived_idea_failure_count = 0
    for archive_dir in _parallel_archive_dirs(results_dir):
        archived_idea_count += len(
            _read_jsonl_records(archive_dir / state_store.IDEAS_HISTORY_FILE)
        )
        archived_attempt_count += len(
            _read_jsonl_records(archive_dir / state_store.ATTEMPTS_HISTORY_FILE)
        )
        archived_idea_failure_count += len(
            _read_jsonl_records(archive_dir / state_store.IDEA_GENERATION_FAILURES_FILE)
        )

    current_idea_count = len(_read_jsonl_records(results_dir / state_store.IDEAS_HISTORY_FILE))
    current_attempt_count = len(
        _read_jsonl_records(results_dir / state_store.ATTEMPTS_HISTORY_FILE)
    )
    current_idea_failure_count = len(
        _read_jsonl_records(results_dir / state_store.IDEA_GENERATION_FAILURES_FILE)
    )

    root_summary_raw = state_store.load_json(results_dir / "parallel_summary.json")
    worker_entries = _archived_worker_entries(
        results_dir=results_dir,
        fallback_summary_raw=root_summary_raw if isinstance(root_summary_raw, dict) else None,
    )
    worker_entries.extend(_worker_entries_from_results(worker_results))
    worker_entries = _sort_worker_entries(worker_entries)

    manifest_raw = state_store.load_json(results_dir / state_store.BEST_MANIFEST_FILE)
    normalized_best = _normalized_best_manifest(
        manifest_raw if isinstance(manifest_raw, dict) else None,
        seed_program_dir=seed_program_dir,
    )
    if normalized_best is not None:
        best_manifest, _best_score = normalized_best
    else:
        reconstructed_best = _best_manifest_from_completed_rounds(
            results_dir=results_dir,
            seed_program_dir=seed_program_dir,
        )
        if reconstructed_best is not None:
            best_manifest, _best_score = reconstructed_best
        else:
            best_manifest = {
                "program_dir": str(seed_program_dir),
                "entrypoint_path": entrypoint_relpath_from_spec(seed_program_dir, required=False)
                or "src/program.py",
                "combined_score": None,
                "correct": None,
                "source_attempt_id": "parallel_bootstrap",
                "updated_at": utc_now(),
            }

    return {
        "created_at": utc_now(),
        "worker_count": len(worker_entries),
        "failed_worker_count": sum(1 for entry in worker_entries if entry["exit_code"] != 0),
        "idea_count": archived_idea_count + current_idea_count,
        "attempt_count": archived_attempt_count + current_attempt_count,
        "idea_generation_failure_count": (archived_idea_failure_count + current_idea_failure_count),
        "best_program_dir": best_manifest.get("program_dir"),
        "best_entrypoint_path": best_manifest.get("entrypoint_path"),
        "best_combined_score": best_manifest.get("combined_score"),
        "workers": worker_entries,
    }


def aggregate_parallel_worker_outputs(
    *,
    results_dir: Path,
    seed_program_dir: Path,
    worker_results: list[ParallelWorkerResult],
    log_event_fn: LogFn,
) -> dict[str, Any]:
    """Handle aggregate parallel worker outputs logic for this module."""
    state_store.ensure_histories(results_dir)
    ideas_path = state_store.ideas_history_path(results_dir)
    attempts_path = state_store.attempts_history_path(results_dir)
    idea_failures_path = state_store.idea_generation_failures_path(results_dir)
    ideas_path.write_text("", encoding="utf-8")
    attempts_path.write_text("", encoding="utf-8")
    idea_failures_path.write_text("", encoding="utf-8")

    merged_ideas: list[dict[str, Any]] = []
    merged_attempts: list[dict[str, Any]] = []
    merged_idea_failures: list[dict[str, Any]] = []

    worker_entries: list[dict[str, Any]] = []
    existing_best = _normalized_best_manifest(
        state_store.load_json(results_dir / state_store.BEST_MANIFEST_FILE),
        seed_program_dir=seed_program_dir,
    )
    if existing_best is not None:
        best_manifest, best_score = existing_best
    else:
        best_manifest = None
        best_score = None

    for worker in sorted(worker_results, key=lambda item: item.plan.worker_index):
        plan = worker.plan
        worker_entries.append(
            {
                "worker_index": plan.worker_index,
                "assigned_cycles": plan.assigned_cycles,
                "cycle_start": plan.cycle_start,
                "cycle_end": plan.cycle_end,
                "cycle_map": plan.cycle_map,
                "results_dir": str(plan.results_dir),
                "log_path": str(worker.log_path),
                "exit_code": worker.exit_code,
                "elapsed_seconds": round(worker.elapsed_seconds, 3),
            }
        )

        for record in _read_jsonl_records(plan.results_dir / state_store.IDEAS_HISTORY_FILE):
            local_cycle = record.get("cycle_index")
            mapped_cycle = _map_worker_cycle(plan, local_cycle)
            merged = dict(record)
            if mapped_cycle is not None:
                merged["cycle_index"] = mapped_cycle
            merged["parallel_worker_index"] = plan.worker_index
            merged["parallel_worker_cycle_index"] = _positive_int(local_cycle)
            merged_ideas.append(merged)

        for record in _read_jsonl_records(plan.results_dir / state_store.ATTEMPTS_HISTORY_FILE):
            local_cycle = record.get("cycle_index")
            mapped_cycle = _map_worker_cycle(plan, local_cycle)
            merged = dict(record)
            if mapped_cycle is not None:
                merged["cycle_index"] = mapped_cycle
            merged["parallel_worker_index"] = plan.worker_index
            merged["parallel_worker_cycle_index"] = _positive_int(local_cycle)
            merged_attempts.append(merged)

        for record in _read_jsonl_records(
            plan.results_dir / state_store.IDEA_GENERATION_FAILURES_FILE
        ):
            local_cycle = record.get("cycle_index")
            mapped_cycle = _map_worker_cycle(plan, local_cycle)
            merged = dict(record)
            if mapped_cycle is not None:
                merged["cycle_index"] = mapped_cycle
            merged["parallel_worker_index"] = plan.worker_index
            merged["parallel_worker_cycle_index"] = _positive_int(local_cycle)
            merged_idea_failures.append(merged)

        normalized = _normalized_best_manifest(
            state_store.load_json(plan.results_dir / state_store.BEST_MANIFEST_FILE),
            seed_program_dir=seed_program_dir,
        )
        if normalized is None:
            continue
        candidate_manifest, score = normalized
        if best_score is None or score > best_score:
            best_score = score
            best_manifest = {
                **candidate_manifest,
                "parallel_worker_index": plan.worker_index,
                "updated_at": utc_now(),
            }

    merged_ideas.sort(
        key=lambda record: (
            _positive_int(record.get("cycle_index")) or 0,
            str(record.get("timestamp", "")),
            str(record.get("idea_id", "")),
        )
    )
    merged_attempts.sort(
        key=lambda record: (
            _positive_int(record.get("cycle_index")) or 0,
            _positive_int(record.get("attempt_number")) or 0,
            str(record.get("timestamp", "")),
            str(record.get("attempt_id", "")),
        )
    )
    merged_idea_failures.sort(
        key=lambda record: (
            _positive_int(record.get("cycle_index")) or 0,
            _positive_int(record.get("generation_attempt")) or 0,
            str(record.get("timestamp", "")),
        )
    )

    for record in merged_ideas:
        state_store.append_jsonl(ideas_path, record)
    for record in merged_attempts:
        state_store.append_jsonl(attempts_path, record)
    for record in merged_idea_failures:
        state_store.append_jsonl(idea_failures_path, record)

    if best_manifest is None:
        best_manifest = {
            "program_dir": str(seed_program_dir),
            "entrypoint_path": entrypoint_relpath_from_spec(seed_program_dir, required=False)
            or "src/program.py",
            "combined_score": None,
            "correct": None,
            "source_attempt_id": "parallel_bootstrap",
            "updated_at": utc_now(),
        }
    state_store.save_best_manifest(results_dir, best_manifest)

    outer_state = state_store.load_json(results_dir / state_store.OUTER_STATE_FILE)
    if not isinstance(outer_state, dict):
        outer_state = state_store.load_or_init_outer_state(
            results_dir, seed_program_dir, resume=False
        )
    completed_cycle = 0
    for record in merged_ideas:
        cycle_idx = _positive_int(record.get("cycle_index"))
        if cycle_idx and cycle_idx > completed_cycle:
            completed_cycle = cycle_idx
    for record in merged_idea_failures:
        generation_attempt = _positive_int(record.get("generation_attempt"))
        max_attempts = _positive_int(record.get("max_attempts"))
        if generation_attempt is not None and generation_attempt == max_attempts:
            cycle_idx = _positive_int(record.get("cycle_index"))
            if cycle_idx and cycle_idx > completed_cycle:
                completed_cycle = cycle_idx
    last_idea_id = merged_ideas[-1].get("idea_id") if merged_ideas else None
    prior_cycle = _positive_int(outer_state.get("current_cycle")) or 0
    outer_state["current_cycle"] = max(prior_cycle, completed_cycle)
    if last_idea_id is not None:
        outer_state["last_idea_id"] = last_idea_id
    outer_state["failure_streak"] = 0
    outer_state["best_program_dir"] = str(best_manifest.get("program_dir", seed_program_dir))
    outer_state["best_combined_score"] = best_manifest.get("combined_score")
    outer_state["best_correct"] = best_manifest.get("correct")
    state_store.save_outer_state(results_dir, outer_state)
    _cleanup_parallel_worker_state_files(worker_results, log_event_fn=log_event_fn)

    failed_workers = [entry for entry in worker_entries if entry["exit_code"] != 0]
    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "worker_count": len(worker_entries),
        "failed_worker_count": len(failed_workers),
        "idea_count": len(merged_ideas),
        "attempt_count": len(merged_attempts),
        "idea_generation_failure_count": len(merged_idea_failures),
        "best_program_dir": best_manifest.get("program_dir"),
        "best_entrypoint_path": best_manifest.get("entrypoint_path"),
        "best_combined_score": best_manifest.get("combined_score"),
        "workers": worker_entries,
    }
    state_store.write_json(results_dir / "parallel_summary.json", summary)
    return summary


def _run_single_parallel_round(
    *,
    cfg: InvestigationConfig,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path,
    instruct_path: Path,
    worker_seeds: dict[int, Path],
    round_cycle_budget: list[int],
    global_cycle_offset: int,
    round_index: int,
    is_first_round: bool,
    use_round_isolation: bool,
    build_parallel_worker_command_fn: Callable[..., list[str]],
    run_parallel_worker_fn: Callable[[ParallelWorkerPlan], ParallelWorkerResult],
    log_event_fn: LogFn,
    format_elapsed_fn: Callable[[float], str],
    terminate_all_active_processes_fn: Callable[..., list[int]],
    stop_container_session_fn: Callable[..., bool],
    build_container_session_name_fn: Callable[..., str],
    interrupt_grace_seconds: int,
) -> list[ParallelWorkerResult]:
    """Execute a single pruning round with dynamic one-cycle worker leases."""
    worker_root = results_dir
    round_total_cycles = sum(cycles for cycles in round_cycle_budget if cycles > 0)
    workers_active = min(cfg.execution.workers, round_total_cycles)

    # Create a shared memory directory for this round when using round isolation.
    # Workers sync from/to this directory before/after each cycle.
    shared_memory_for_round: Path | None = None
    if use_round_isolation:
        shared_memory_for_round = results_dir / "_shared_memory" / f"round_{round_index:03d}"
        shared_memory_for_round.mkdir(parents=True, exist_ok=True)
        state_store.ensure_global_memory_files(shared_memory_for_round)
        log_event_fn(
            "shared_memory_init",
            round_index=round_index,
            shared_memory_dir=str(shared_memory_for_round),
        )

    worker_results_dirs: dict[int, Path] = {}
    worker_global_memory_dirs: dict[int, Path] = {}
    worker_conclusions_dirs: dict[int, Path] = {}
    for worker_index in range(1, workers_active + 1):
        if use_round_isolation:
            # Each round gets its own isolated subdirectory so the container
            # only mounts the current round's data — the agent cannot see
            # previous rounds' ideas, attempts, or archived history files.
            worker_results_dir = (
                worker_root / f"worker_{worker_index:02d}" / f"round_{round_index:03d}"
            )
            if worker_results_dir.exists():
                shutil.rmtree(worker_results_dir)
                log_event_fn(
                    "parallel_worker_round_reset",
                    worker=worker_index,
                    round_index=round_index,
                    results_dir=worker_results_dir,
                )
            worker_global_memory_dir = worker_results_dir
            # Conclusions live alongside per-round memory in the same dir.
            worker_conclusions_dir = worker_results_dir
        else:
            worker_results_dir = worker_root / f"worker_{worker_index:02d}"
            worker_global_memory_dir = worker_results_dir
            # Workers synchronize local memory into the shared directory under
            # an fcntl lock. Conclusions remain worker-local because they are
            # overwritten rather than appended.
            worker_conclusions_dir = worker_results_dir

            if is_first_round:
                # Round 0: clean worker dirs (unless resume — but resume is not
                # supported for parallel, validated in config)
                if not cfg.execution.resume and worker_results_dir.exists():
                    shutil.rmtree(worker_results_dir)
                    log_event_fn(
                        "parallel_worker_results_reset",
                        worker=worker_index,
                        results_dir=worker_results_dir,
                    )
            else:
                # Round > 0: reset per-worker history files and outer_state so the
                # worker starts fresh, but keep best_program_manifest.json intact.
                for rel in (
                    state_store.OUTER_STATE_FILE,
                    state_store.IDEAS_HISTORY_FILE,
                    state_store.ATTEMPTS_HISTORY_FILE,
                    state_store.CONCLUSIONS_FILE,
                    state_store.CONCLUSIONS_HISTORY_FILE,
                    state_store.OTHER_WORKERS_CONCLUSIONS_FILE,
                ):
                    candidate = worker_results_dir / rel
                    if candidate.is_file():
                        candidate.unlink()

        worker_results_dirs[worker_index] = worker_results_dir
        worker_global_memory_dirs[worker_index] = worker_global_memory_dir
        worker_conclusions_dirs[worker_index] = worker_conclusions_dir

    queue_lock = threading.Lock()
    next_round_cycle = 0
    worker_lease_counts: dict[int, int] = {idx: 0 for idx in range(1, workers_active + 1)}
    worker_cycle_maps: dict[int, dict[int, int]] = {idx: {} for idx in range(1, workers_active + 1)}
    worker_logs: dict[int, list[Path]] = {idx: [] for idx in range(1, workers_active + 1)}
    stop_scheduling = False
    task_failures: list[ParallelWorkerResult] = []

    def _lease_next_cycle(worker_index: int) -> tuple[int, int] | None:
        nonlocal next_round_cycle, stop_scheduling
        with queue_lock:
            if stop_scheduling or next_round_cycle >= round_total_cycles:
                return None
            next_round_cycle += 1
            global_cycle = global_cycle_offset + next_round_cycle - 1
            worker_lease_counts[worker_index] += 1
            local_cycle = worker_lease_counts[worker_index]
            worker_cycle_maps[worker_index][local_cycle] = global_cycle
            return local_cycle, global_cycle

    def _handle_failed_lease(*, worker_index: int, local_cycle: int, global_cycle: int) -> None:
        nonlocal stop_scheduling
        if cfg.execution.parallel_fail_fast:
            with queue_lock:
                stop_scheduling = True
            log_event_fn(
                "parallel_worker_lease_failure",
                round_index=round_index,
                worker=worker_index,
                lease_index=local_cycle,
                global_cycle=global_cycle,
                action="stop_scheduling",
            )
            return
        log_event_fn(
            "parallel_worker_lease_failure_tolerated",
            round_index=round_index,
            worker=worker_index,
            lease_index=local_cycle,
            global_cycle=global_cycle,
            action="continue_scheduling",
        )

    def _run_worker_queue(worker_index: int) -> ParallelWorkerResult:
        nonlocal stop_scheduling
        worker_results_dir = worker_results_dirs[worker_index]
        worker_global_memory_dir = worker_global_memory_dirs[worker_index]
        worker_conclusions_dir = worker_conclusions_dirs[worker_index]
        worker_started_at = time.monotonic()
        seed_for_worker = worker_seeds.get(worker_index)
        lease_results: list[ParallelWorkerResult] = []

        while True:
            lease = _lease_next_cycle(worker_index)
            if lease is None:
                break
            local_cycle, global_cycle = lease
            lease_log_path = worker_results_dir / f"parallel_worker_cycle_{global_cycle:04d}.log"
            worker_command = build_parallel_worker_command_fn(
                cfg=cfg,
                task_dir=task_dir,
                instruct_path=instruct_path,
                results_dir=worker_results_dir,
                global_memory_dir=worker_global_memory_dir,
                assigned_cycles=local_cycle,
                seed_program=seed_for_worker,
                shared_memory_dir=(
                    None if cfg.execution.isolate_worker_memory else shared_memory_for_round
                ),
                conclusions_dir=worker_conclusions_dir,
                worker_index=worker_index,
                resume=(local_cycle > 1),
                global_cycle=global_cycle,
            )
            plan = ParallelWorkerPlan(
                worker_index=worker_index,
                assigned_cycles=1,
                cycle_start=global_cycle,
                cycle_end=global_cycle,
                results_dir=worker_results_dir,
                command=worker_command,
                cycle_map={local_cycle: global_cycle},
                lease_index=local_cycle,
                log_path=lease_log_path,
                environment=worker_gpu_environment(cfg, worker_index),
            )
            log_event_fn(
                "parallel_worker_lease",
                round_index=round_index,
                worker=worker_index,
                lease_index=local_cycle,
                local_cycle=local_cycle,
                global_cycle=global_cycle,
                results_dir=worker_results_dir,
            )
            lease_result = run_parallel_worker_fn(plan)
            lease_results.append(lease_result)
            if lease_result.exit_code in HOST_FAILURES:
                with queue_lock:
                    stop_scheduling = True
                    task_failures.append(lease_result)
                _stop_task_failure_workers(
                    failure_code=lease_result.exit_code,
                    cfg=cfg, worker_dirs=worker_results_dirs.values(),
                    terminate_all_active_processes_fn=terminate_all_active_processes_fn,
                    stop_container_session_fn=stop_container_session_fn,
                    build_container_session_name_fn=build_container_session_name_fn,
                    log_event_fn=log_event_fn, grace_seconds=interrupt_grace_seconds,
                )
                return lease_result
            worker_logs[worker_index].append(lease_result.log_path)
            log_event_fn(
                "parallel_worker_lease_complete",
                round_index=round_index,
                worker=worker_index,
                lease_index=local_cycle,
                global_cycle=global_cycle,
                exit=lease_result.exit_code,
                elapsed=format_elapsed_fn(lease_result.elapsed_seconds),
                log_path=lease_result.log_path,
            )
            if lease_result.exit_code != 0:
                _handle_failed_lease(
                    worker_index=worker_index,
                    local_cycle=local_cycle,
                    global_cycle=global_cycle,
                )
                break

        cycle_map = dict(worker_cycle_maps[worker_index])
        assigned_cycles = len(cycle_map)
        worker_log_path = worker_results_dir / "parallel_worker.log"
        output_parts: list[str] = []
        for log_path in worker_logs[worker_index]:
            if not log_path.is_file():
                continue
            output_parts.append(f"\n===== {log_path.name} =====\n")
            output_parts.append(log_path.read_text(encoding="utf-8", errors="replace"))
        output = "".join(output_parts)
        worker_log_path.parent.mkdir(parents=True, exist_ok=True)
        state_store.write_private_text(worker_log_path, output)
        exit_code = next((result.exit_code for result in lease_results if result.exit_code != 0), 0)
        global_cycles = sorted(cycle_map.values())
        if global_cycles:
            cycle_start = global_cycles[0]
            cycle_end = global_cycles[-1]
        else:
            cycle_start = 0
            cycle_end = 0
        final_plan = ParallelWorkerPlan(
            worker_index=worker_index,
            assigned_cycles=assigned_cycles,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            results_dir=worker_results_dir,
            command=lease_results[-1].plan.command if lease_results else [],
            cycle_map=cycle_map,
            log_path=worker_log_path,
        )
        return ParallelWorkerResult(
            plan=final_plan,
            exit_code=exit_code,
            log_path=worker_log_path,
            elapsed_seconds=time.monotonic() - worker_started_at,
            output_tail=output[-1200:],
            suppressed_output_lines=sum(result.suppressed_output_lines for result in lease_results),
        )

    log_event_fn(
        "parallel_round_start",
        round_index=round_index,
        workers_active=workers_active,
        scheduling="dynamic_queue",
    )
    for worker_index in range(1, workers_active + 1):
        log_event_fn(
            "parallel_worker_launch",
            round_index=round_index,
            worker=worker_index,
            assigned_cycles="dynamic",
            cycle_start=None,
            cycle_end=None,
            results_dir=worker_results_dirs[worker_index],
        )

    worker_results: list[ParallelWorkerResult] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers_active)) as pool:
            future_to_worker = {
                pool.submit(_run_worker_queue, worker_index): worker_index
                for worker_index in range(1, workers_active + 1)
            }
            for future in as_completed(future_to_worker):
                result = future.result()
                worker_results.append(result)
                log_event_fn(
                    "parallel_worker_complete",
                    round_index=round_index,
                    worker=result.plan.worker_index,
                    exit=result.exit_code,
                    elapsed=format_elapsed_fn(result.elapsed_seconds),
                    log_path=result.log_path,
                    suppressed_output_lines=result.suppressed_output_lines,
                )
    except KeyboardInterrupt:
        log_event_fn(
            "parallel_interrupt",
            action="terminate_active_processes",
            grace_seconds=interrupt_grace_seconds,
        )
        raw_terminated_pids = terminate_all_active_processes_fn(
            grace_seconds=interrupt_grace_seconds
        )
        terminated_pids = (
            [pid for pid in raw_terminated_pids if isinstance(pid, int) and pid > 0]
            if isinstance(raw_terminated_pids, list)
            else []
        )
        stopped_container_names: list[str] = []
        if cfg.container.enabled:
            for worker_index in range(1, workers_active + 1):
                session_name = build_container_session_name_fn(
                    agent=cfg.agent.name,
                    results_dir=worker_results_dirs[worker_index],
                )
                if stop_container_session_fn(session_name, log_event_fn=log_event_fn):
                    stopped_container_names.append(session_name)
        if stopped_container_names:
            log_event_fn(
                "parallel_interrupt_cleanup_targets",
                mode="containers",
                targets=",".join(stopped_container_names),
            )
        elif terminated_pids:
            log_event_fn(
                "parallel_interrupt_cleanup_targets",
                mode="processes",
                targets=",".join(str(pid) for pid in terminated_pids),
            )
        else:
            log_event_fn(
                "parallel_interrupt_cleanup_targets",
                mode="none",
                targets="",
            )
        raise
    except Exception:
        if task_failures:
            return task_failures
        raise

    return worker_results


def _seed_dir_from_manifest(
    manifest_raw: dict[str, Any] | None,
) -> Path | None:
    """Resolve the seed program directory from a worker best-manifest."""
    if not isinstance(manifest_raw, dict) or manifest_raw.get("correct") is not True:
        return None

    program_dir_raw = manifest_raw.get("program_dir")
    if isinstance(program_dir_raw, str) and program_dir_raw.strip():
        candidate = resolve_path(program_dir_raw)
        if candidate.is_dir():
            return candidate

    program_path_raw = manifest_raw.get("program_path")
    if isinstance(program_path_raw, str) and program_path_raw.strip():
        candidate = resolve_path(program_path_raw)
        if candidate.is_dir():
            return candidate
        if candidate.is_file():
            parent = candidate.parent
            if parent.name == "src" and parent.parent.is_dir():
                return parent.parent
            return parent
    return None


def _update_worker_seeds(
    worker_results: list[ParallelWorkerResult],
    *,
    current_seeds: dict[int, Path],
    current_seed_scores: dict[int, float],
    default_seed: Path,
    seed_program_dir: Path,
) -> tuple[dict[int, Path], dict[int, float]]:
    """Update worker seeds, only replacing when the new round's best beats the previous best."""
    seeds: dict[int, Path] = dict(current_seeds)
    scores: dict[int, float] = dict(current_seed_scores)
    for worker in worker_results:
        manifest_raw = state_store.load_json(
            worker.plan.results_dir / state_store.BEST_MANIFEST_FILE
        )
        normalized = _normalized_best_manifest(
            manifest_raw if isinstance(manifest_raw, dict) else None,
            seed_program_dir=seed_program_dir,
        )
        if normalized is not None:
            _manifest, new_score = normalized
            seed_dir = _seed_dir_from_manifest(manifest_raw)
            prev_score = scores.get(worker.plan.worker_index)
            if seed_dir is not None and (prev_score is None or new_score > prev_score):
                seeds[worker.plan.worker_index] = seed_dir
                scores[worker.plan.worker_index] = new_score
        elif worker.plan.worker_index not in seeds:
            seeds[worker.plan.worker_index] = default_seed
    return seeds, scores


def run_parallel_campaign(
    *,
    cfg: InvestigationConfig,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path,
    instruct_path: Path,
    seed_program_dir: Path,
    split_cycle_budget_fn: Callable[[int, int], list[int]],
    build_parallel_worker_command_fn: Callable[..., list[str]],
    run_parallel_worker_fn: Callable[[ParallelWorkerPlan], ParallelWorkerResult],
    aggregate_parallel_worker_outputs_fn: Callable[..., dict[str, Any]],
    log_event_fn: LogFn,
    format_elapsed_fn: Callable[[float], str],
    terminate_all_active_processes_fn: Callable[..., list[int]],
    stop_container_session_fn: Callable[..., bool],
    build_container_session_name_fn: Callable[..., str],
    interrupt_grace_seconds: int = DEFAULT_INTERRUPT_GRACE_SECONDS,
) -> int:
    """Run parallel campaign."""
    prune_n = cfg.execution.prune_every_n_cycles
    workers = cfg.execution.workers
    max_cycles = cfg.execution.max_idea_cycles
    if cfg.execution.resume and 0 < prune_n < max_cycles:
        try:
            persisted = _load_parallel_resume_checkpoint(results_dir)
            if persisted is not None:
                _validate_parallel_resume_memory_isolation(
                    persisted,
                    isolate_worker_memory=cfg.execution.isolate_worker_memory,
                )
        except RuntimeError as exc:
            log_event_fn("parallel_resume_failed", error=str(exc), results_dir=results_dir)
            return 2

    results_dir.mkdir(parents=True, exist_ok=True)
    global_memory_dir.mkdir(parents=True, exist_ok=True)
    state_store.ensure_global_memory_files(global_memory_dir)

    if prune_n == 0 or prune_n >= max_cycles:
        # No pruning — single-round behavior (original logic).
        return _run_single_round_campaign(
            cfg=cfg,
            task_dir=task_dir,
            results_dir=results_dir,
            global_memory_dir=global_memory_dir,
            instruct_path=instruct_path,
            seed_program_dir=seed_program_dir,
            split_cycle_budget_fn=split_cycle_budget_fn,
            build_parallel_worker_command_fn=build_parallel_worker_command_fn,
            run_parallel_worker_fn=run_parallel_worker_fn,
            aggregate_parallel_worker_outputs_fn=aggregate_parallel_worker_outputs_fn,
            log_event_fn=log_event_fn,
            format_elapsed_fn=format_elapsed_fn,
            terminate_all_active_processes_fn=terminate_all_active_processes_fn,
            stop_container_session_fn=stop_container_session_fn,
            build_container_session_name_fn=build_container_session_name_fn,
            interrupt_grace_seconds=interrupt_grace_seconds,
        )

    # Multi-round pruning campaign always uses round-based directories.
    # Containers determine whether that boundary is also a true filesystem
    # isolation boundary for the worker.
    use_round_isolation = True
    if not cfg.container.enabled:
        log_event_fn(
            "prune_warning",
            message=(
                "Pruning is enabled but the agent is NOT containerized. "
                "Workers will still have access to archived history and "
                "idea files from previous rounds, so pruning may be "
                "ineffective at resetting context. Use container.enabled=true "
                "for full round isolation."
            ),
        )
        confirm_prune_warning()

    cycles_per_round = prune_n
    total_rounds = math.ceil(max_cycles / cycles_per_round)

    worker_seeds: dict[int, Path] = {i: seed_program_dir for i in range(1, workers + 1)}
    worker_seed_scores: dict[int, float] = {}
    all_worker_results: list[ParallelWorkerResult] = []
    global_cycle_offset = 1
    start_round_index = 0
    resume_shared_memory_source: Path | None = None

    if cfg.execution.resume:
        try:
            resume_state = _load_or_reconstruct_parallel_resume_state(
                cfg=cfg,
                results_dir=results_dir,
                global_memory_dir=global_memory_dir,
                seed_program_dir=seed_program_dir,
                total_rounds=total_rounds,
                cycles_per_round=cycles_per_round,
                workers=workers,
                log_event_fn=log_event_fn,
            )
        except RuntimeError as exc:
            log_event_fn("parallel_resume_failed", error=str(exc), results_dir=results_dir)
            return 2

        start_round_index = min(
            _positive_int(resume_state.get("next_round_index")) or 0,
            total_rounds,
        )
        global_cycle_offset = _positive_int(resume_state.get("global_cycle_offset")) or 1
        worker_seeds = {
            worker_index: seed
            for worker_index, seed in dict(resume_state.get("worker_seeds", {})).items()
            if isinstance(worker_index, int)
        }
        worker_seed_scores = {
            worker_index: score
            for worker_index, score in dict(resume_state.get("worker_seed_scores", {})).items()
            if isinstance(worker_index, int) and isinstance(score, (int, float))
        }
        resume_shared_memory_source = resume_state.get("shared_memory_source")
        log_event_fn(
            "parallel_resume_loaded",
            next_round_index=start_round_index,
            global_cycle_offset=global_cycle_offset,
            resume_basis=resume_state.get("resume_basis", "unknown"),
            shared_memory_source=resume_shared_memory_source,
        )

    log_event_fn(
        "parallel_start",
        workers_requested=workers,
        workers_active=workers,
        max_idea_cycles=max_cycles,
        worker_root=results_dir,
        global_memory_dir=global_memory_dir,
        total_rounds=total_rounds,
        cycles_per_round=cycles_per_round,
        round_isolation=use_round_isolation,
        resume=cfg.execution.resume,
        start_round_index=start_round_index,
    )

    if start_round_index >= total_rounds:
        summary = build_parallel_campaign_summary(
            results_dir=results_dir,
            seed_program_dir=seed_program_dir,
            worker_results=[],
        )
        state_store.write_json(results_dir / "parallel_summary.json", summary)
        failed_count = int(summary.get("failed_worker_count", 0))
        log_event_fn(
            "parallel_end",
            workers=workers,
            failed_workers=failed_count,
            ideas=summary.get("idea_count", 0),
            attempts=summary.get("attempt_count", 0),
            summary_path=results_dir / "parallel_summary.json",
            total_rounds=total_rounds,
        )
        return 0 if failed_count == 0 else 2

    for round_idx in range(start_round_index, total_rounds):
        remaining = max_cycles - (round_idx * cycles_per_round)
        this_round_total = min(cycles_per_round, remaining)
        if this_round_total <= 0:
            break

        round_cycle_budget = split_cycle_budget_fn(this_round_total, workers)
        if (
            resume_shared_memory_source is not None
            and round_idx == start_round_index
            and round_idx > 0
        ):
            _seed_round_shared_memory(
                results_dir=results_dir,
                round_index=round_idx,
                source_dir=resume_shared_memory_source,
            )

        worker_results = _run_single_parallel_round(
            cfg=cfg,
            task_dir=task_dir,
            results_dir=results_dir,
            global_memory_dir=global_memory_dir,
            instruct_path=instruct_path,
            worker_seeds=worker_seeds,
            round_cycle_budget=round_cycle_budget,
            global_cycle_offset=global_cycle_offset,
            round_index=round_idx,
            is_first_round=(round_idx == 0),
            use_round_isolation=use_round_isolation,
            build_parallel_worker_command_fn=build_parallel_worker_command_fn,
            run_parallel_worker_fn=run_parallel_worker_fn,
            log_event_fn=log_event_fn,
            format_elapsed_fn=format_elapsed_fn,
            terminate_all_active_processes_fn=terminate_all_active_processes_fn,
            stop_container_session_fn=stop_container_session_fn,
            build_container_session_name_fn=build_container_session_name_fn,
            interrupt_grace_seconds=interrupt_grace_seconds,
        )
        all_worker_results.extend(worker_results)
        host_failure_code = next(
            (worker.exit_code for worker in worker_results if worker.exit_code in HOST_FAILURES), None,
        )
        if host_failure_code is not None:
            return host_failure_code
        resume_shared_memory_source = None

        # Update worker seeds BEFORE aggregate (which cleans up per-worker
        # state files including best_program_manifest.json).
        worker_seeds, worker_seed_scores = _update_worker_seeds(
            worker_results,
            current_seeds=worker_seeds,
            current_seed_scores=worker_seed_scores,
            default_seed=seed_program_dir,
            seed_program_dir=seed_program_dir,
        )

        # Aggregate this round's outputs (merges histories, updates best manifest)
        aggregate_parallel_worker_outputs_fn(
            results_dir=results_dir,
            seed_program_dir=seed_program_dir,
            worker_results=worker_results,
        )

        global_cycle_offset += this_round_total

        # Prune between rounds (not after the last round)
        is_last_round = round_idx == total_rounds - 1
        round_failed = any(worker.exit_code != 0 for worker in worker_results)
        if not is_last_round:
            state_store.archive_for_parallel_prune(results_dir, global_memory_dir, round_idx)
            # Archive shared memory contents alongside the pruned round data
            shared_archive: Path | None = None
            if use_round_isolation:
                shared_mem_dir = results_dir / "_shared_memory" / f"round_{round_idx:03d}"
                prune_archive = results_dir / "_pruned" / f"round_{round_idx:03d}"
                candidate_archive = prune_archive / "_shared_memory"
                if state_store.archive_memory_files_for_prune(
                    shared_mem_dir,
                    candidate_archive,
                ):
                    shared_archive = candidate_archive
                    # Archive the cross-worker conclusions subdir (one file per
                    # worker) so the next round's first cycle starts clean.
                    state_store.archive_shared_conclusions_for_prune(shared_mem_dir, shared_archive)
            if not round_failed:
                _persist_parallel_resume_state(
                    results_dir=results_dir,
                    total_rounds=total_rounds,
                    cycles_per_round=cycles_per_round,
                    workers=workers,
                    next_round_index=round_idx + 1,
                    global_cycle_offset=global_cycle_offset,
                    worker_seeds=worker_seeds,
                    worker_seed_scores=worker_seed_scores,
                    shared_memory_source=shared_archive,
                    isolate_worker_memory=cfg.execution.isolate_worker_memory,
                )
            log_event_fn(
                "parallel_prune",
                round_index=round_idx,
                archive_dir=str(results_dir / "_pruned" / f"round_{round_idx:03d}"),
            )
        elif not round_failed:
            _persist_parallel_resume_state(
                results_dir=results_dir,
                total_rounds=total_rounds,
                cycles_per_round=cycles_per_round,
                workers=workers,
                next_round_index=total_rounds,
                global_cycle_offset=global_cycle_offset,
                worker_seeds=worker_seeds,
                worker_seed_scores=worker_seed_scores,
                shared_memory_source=None,
                isolate_worker_memory=cfg.execution.isolate_worker_memory,
            )

    # Final campaign summary spans archived rounds plus the current root state.
    summary = build_parallel_campaign_summary(
        results_dir=results_dir,
        seed_program_dir=seed_program_dir,
        worker_results=all_worker_results,
    )
    state_store.write_json(results_dir / "parallel_summary.json", summary)

    # Store prune_every_n_cycles in outer_state so the inspector can detect it.
    outer = state_store.load_json(results_dir / state_store.OUTER_STATE_FILE)
    if isinstance(outer, dict):
        outer["prune_every_n_cycles"] = prune_n
        state_store.save_outer_state(results_dir, outer)

    failed_count = int(summary.get("failed_worker_count", 0))
    log_event_fn(
        "parallel_end",
        workers=workers,
        failed_workers=failed_count,
        ideas=summary.get("idea_count", 0),
        attempts=summary.get("attempt_count", 0),
        summary_path=results_dir / "parallel_summary.json",
        total_rounds=total_rounds,
    )
    return 0 if failed_count == 0 else 2


def _run_single_round_campaign(
    *,
    cfg: InvestigationConfig,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path,
    instruct_path: Path,
    seed_program_dir: Path,
    split_cycle_budget_fn: Callable[[int, int], list[int]],
    build_parallel_worker_command_fn: Callable[..., list[str]],
    run_parallel_worker_fn: Callable[[ParallelWorkerPlan], ParallelWorkerResult],
    aggregate_parallel_worker_outputs_fn: Callable[..., dict[str, Any]],
    log_event_fn: LogFn,
    format_elapsed_fn: Callable[[float], str],
    terminate_all_active_processes_fn: Callable[..., list[int]],
    stop_container_session_fn: Callable[..., bool],
    build_container_session_name_fn: Callable[..., str],
    interrupt_grace_seconds: int = DEFAULT_INTERRUPT_GRACE_SECONDS,
) -> int:
    """Run a non-pruning campaign with dynamic one-cycle worker leases."""
    worker_root = results_dir

    workers_active = min(cfg.execution.workers, cfg.execution.max_idea_cycles)
    for worker_index in range(1, workers_active + 1):
        worker_results_dir = worker_root / f"worker_{worker_index:02d}"
        if not cfg.execution.resume and worker_results_dir.exists():
            shutil.rmtree(worker_results_dir)
            log_event_fn(
                "parallel_worker_results_reset",
                worker=worker_index,
                results_dir=worker_results_dir,
            )

    log_event_fn(
        "parallel_start",
        workers_requested=cfg.execution.workers,
        workers_active=workers_active,
        max_idea_cycles=cfg.execution.max_idea_cycles,
        worker_root=worker_root,
        global_memory_dir=global_memory_dir,
        scheduling="dynamic_queue",
    )

    queue_lock = threading.Lock()
    next_global_cycle = 1
    worker_lease_counts: dict[int, int] = {idx: 0 for idx in range(1, workers_active + 1)}
    worker_cycle_maps: dict[int, dict[int, int]] = {idx: {} for idx in range(1, workers_active + 1)}
    worker_logs: dict[int, list[Path]] = {idx: [] for idx in range(1, workers_active + 1)}
    stop_scheduling = False
    task_failures: list[ParallelWorkerResult] = []

    def _lease_next_cycle(worker_index: int) -> tuple[int, int, int] | None:
        nonlocal next_global_cycle, stop_scheduling
        with queue_lock:
            if stop_scheduling or next_global_cycle > cfg.execution.max_idea_cycles:
                return None
            global_cycle = next_global_cycle
            next_global_cycle += 1
            worker_lease_counts[worker_index] += 1
            local_cycle = worker_lease_counts[worker_index]
            worker_cycle_maps[worker_index][local_cycle] = global_cycle
            return local_cycle, global_cycle, local_cycle

    def _handle_failed_lease(*, worker_index: int, local_cycle: int, global_cycle: int) -> None:
        nonlocal stop_scheduling
        if cfg.execution.parallel_fail_fast:
            with queue_lock:
                stop_scheduling = True
            log_event_fn(
                "parallel_worker_lease_failure",
                worker=worker_index,
                lease_index=local_cycle,
                global_cycle=global_cycle,
                action="stop_scheduling",
            )
            return
        log_event_fn(
            "parallel_worker_lease_failure_tolerated",
            worker=worker_index,
            lease_index=local_cycle,
            global_cycle=global_cycle,
            action="continue_scheduling",
        )

    def _run_worker_queue(worker_index: int) -> ParallelWorkerResult:
        nonlocal stop_scheduling
        worker_results_dir = worker_root / f"worker_{worker_index:02d}"
        worker_started_at = time.monotonic()
        lease_results: list[ParallelWorkerResult] = []

        while True:
            lease = _lease_next_cycle(worker_index)
            if lease is None:
                break
            local_cycle, global_cycle, command_max_cycles = lease
            lease_log_path = worker_results_dir / f"parallel_worker_cycle_{global_cycle:04d}.log"
            worker_command = build_parallel_worker_command_fn(
                cfg=cfg,
                task_dir=task_dir,
                instruct_path=instruct_path,
                results_dir=worker_results_dir,
                global_memory_dir=worker_results_dir,
                assigned_cycles=command_max_cycles,
                seed_program=seed_program_dir,
                shared_memory_dir=(
                    None if cfg.execution.isolate_worker_memory else global_memory_dir
                ),
                conclusions_dir=worker_results_dir,
                worker_index=worker_index,
                resume=(local_cycle > 1),
                global_cycle=global_cycle,
            )
            plan = ParallelWorkerPlan(
                worker_index=worker_index,
                assigned_cycles=1,
                cycle_start=global_cycle,
                cycle_end=global_cycle,
                results_dir=worker_results_dir,
                command=worker_command,
                cycle_map={local_cycle: global_cycle},
                lease_index=local_cycle,
                log_path=lease_log_path,
                environment=worker_gpu_environment(cfg, worker_index),
            )
            log_event_fn(
                "parallel_worker_lease",
                worker=worker_index,
                lease_index=local_cycle,
                local_cycle=local_cycle,
                global_cycle=global_cycle,
                results_dir=worker_results_dir,
            )
            lease_result = run_parallel_worker_fn(plan)
            lease_results.append(lease_result)
            if lease_result.exit_code in HOST_FAILURES:
                with queue_lock:
                    stop_scheduling = True
                    task_failures.append(lease_result)
                _stop_task_failure_workers(
                    failure_code=lease_result.exit_code,
                    cfg=cfg,
                    worker_dirs=(
                        worker_root / f"worker_{index:02d}"
                        for index in range(1, workers_active + 1)
                    ),
                    terminate_all_active_processes_fn=terminate_all_active_processes_fn,
                    stop_container_session_fn=stop_container_session_fn,
                    build_container_session_name_fn=build_container_session_name_fn,
                    log_event_fn=log_event_fn, grace_seconds=interrupt_grace_seconds,
                )
                return lease_result
            worker_logs[worker_index].append(lease_result.log_path)
            log_event_fn(
                "parallel_worker_lease_complete",
                worker=worker_index,
                lease_index=local_cycle,
                global_cycle=global_cycle,
                exit=lease_result.exit_code,
                elapsed=format_elapsed_fn(lease_result.elapsed_seconds),
                log_path=lease_result.log_path,
            )
            if lease_result.exit_code != 0:
                _handle_failed_lease(
                    worker_index=worker_index,
                    local_cycle=local_cycle,
                    global_cycle=global_cycle,
                )
                break

        cycle_map = dict(worker_cycle_maps[worker_index])
        assigned_cycles = len(cycle_map)
        worker_log_path = worker_results_dir / "parallel_worker.log"
        output_parts: list[str] = []
        for log_path in worker_logs[worker_index]:
            if not log_path.is_file():
                continue
            output_parts.append(f"\n===== {log_path.name} =====\n")
            output_parts.append(log_path.read_text(encoding="utf-8", errors="replace"))
        output = "".join(output_parts)
        worker_log_path.parent.mkdir(parents=True, exist_ok=True)
        state_store.write_private_text(worker_log_path, output)
        exit_code = next((result.exit_code for result in lease_results if result.exit_code != 0), 0)
        global_cycles = sorted(cycle_map.values())
        if global_cycles:
            cycle_start = global_cycles[0]
            cycle_end = global_cycles[-1]
        else:
            cycle_start = 0
            cycle_end = 0
        final_plan = ParallelWorkerPlan(
            worker_index=worker_index,
            assigned_cycles=assigned_cycles,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            results_dir=worker_results_dir,
            command=lease_results[-1].plan.command if lease_results else [],
            cycle_map=cycle_map,
            log_path=worker_log_path,
        )
        return ParallelWorkerResult(
            plan=final_plan,
            exit_code=exit_code,
            log_path=worker_log_path,
            elapsed_seconds=time.monotonic() - worker_started_at,
            output_tail=output[-1200:],
            suppressed_output_lines=sum(result.suppressed_output_lines for result in lease_results),
        )

    for worker_index in range(1, workers_active + 1):
        log_event_fn(
            "parallel_worker_launch",
            worker=worker_index,
            assigned_cycles="dynamic",
            cycle_start=None,
            cycle_end=None,
            results_dir=worker_root / f"worker_{worker_index:02d}",
        )

    worker_results: list[ParallelWorkerResult] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers_active)) as pool:
            future_to_worker = {
                pool.submit(_run_worker_queue, worker_index): worker_index
                for worker_index in range(1, workers_active + 1)
            }
            for future in as_completed(future_to_worker):
                result = future.result()
                worker_results.append(result)
                log_event_fn(
                    "parallel_worker_complete",
                    worker=result.plan.worker_index,
                    exit=result.exit_code,
                    elapsed=format_elapsed_fn(result.elapsed_seconds),
                    log_path=result.log_path,
                    suppressed_output_lines=result.suppressed_output_lines,
                )
    except KeyboardInterrupt:
        log_event_fn(
            "parallel_interrupt",
            action="terminate_active_processes",
            grace_seconds=interrupt_grace_seconds,
        )
        raw_terminated_pids = terminate_all_active_processes_fn(
            grace_seconds=interrupt_grace_seconds
        )
        terminated_pids = (
            [pid for pid in raw_terminated_pids if isinstance(pid, int) and pid > 0]
            if isinstance(raw_terminated_pids, list)
            else []
        )
        # Kill worker wrappers before removing named containers so workers cannot
        # recreate the same sessions during interrupt cleanup.
        stopped_container_names: list[str] = []
        if cfg.container.enabled:
            for worker_index in range(1, workers_active + 1):
                worker_results_dir = worker_root / f"worker_{worker_index:02d}"
                session_name = build_container_session_name_fn(
                    agent=cfg.agent.name,
                    results_dir=worker_results_dir,
                )
                if stop_container_session_fn(session_name, log_event_fn=log_event_fn):
                    stopped_container_names.append(session_name)
        if stopped_container_names:
            log_event_fn(
                "parallel_interrupt_cleanup_targets",
                mode="containers",
                targets=",".join(stopped_container_names),
            )
        elif terminated_pids:
            log_event_fn(
                "parallel_interrupt_cleanup_targets",
                mode="processes",
                targets=",".join(str(pid) for pid in terminated_pids),
            )
        else:
            log_event_fn(
                "parallel_interrupt_cleanup_targets",
                mode="none",
                targets="",
            )
        raise
    except Exception:
        if task_failures:
            return task_failures[0].exit_code
        raise

    host_failure_code = next(
        (worker.exit_code for worker in worker_results if worker.exit_code in HOST_FAILURES), None,
    )
    if host_failure_code is not None:
        return host_failure_code
    summary = aggregate_parallel_worker_outputs_fn(
        results_dir=results_dir,
        seed_program_dir=seed_program_dir,
        worker_results=worker_results,
    )
    failed_count = int(summary.get("failed_worker_count", 0))
    log_event_fn(
        "parallel_end",
        workers=len(worker_results),
        failed_workers=failed_count,
        ideas=summary.get("idea_count", 0),
        attempts=summary.get("attempt_count", 0),
        summary_path=results_dir / "parallel_summary.json",
    )
    return 0 if failed_count == 0 else 2
