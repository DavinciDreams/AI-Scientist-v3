"""Contract-bound final evaluator with once-only invocation semantics."""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import re
import shutil
import socket
import stat
import subprocess
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.pipeline.downstream_selection import resolve_downstream_program

from .broker_storage import broker_root, ensure_broker_root, reject_legacy_broker_directory
from .candidate_broker import (
    CandidateBroker,
    CandidateBrokerError,
    CandidateBrokerInfrastructureError,
    CandidateContainerCleanupError,
    ContainerMetadataCleanupError,
    read_container_id,
)

from src.investigation.task_contract import (
    TASK_CONTRACT_FILENAME,
    TaskContractError,
    load_task_contract,
    task_scaffold_fingerprint,
)
from src.investigation.bounded_process import (
    BoundedProcessError,
    BoundedProcessResult,
    DEFAULT_MAX_OUTPUT_BYTES,
    run_bounded_process,
)
from src.investigation.container_images import resolve_immutable_image
from src.investigation.container_gpus import (
    GPU_ALLOCATION_BUSY_CATEGORY,
    GPU_ALLOCATION_INVALID_CATEGORY,
    GPU_ALLOCATION_QUARANTINED_CATEGORY,
    GPU_ALLOCATION_LEASE_ROOT,
    GpuAllocationLease,
    GpuLeaseError,
    GpuLeaseQuarantined,
    GpuLeaseTimeout,
    GpuSelectorError,
    acquire_gpu_allocation_lease,
    bounded_docker_gpu_selector,
)
from src.investigation.evaluator_tree import (
    TASK_SNAPSHOT_METADATA_FILENAME,
    task_snapshot_metadata_present,
)
from src.investigation.reset_sequence import (
    declared_reset_sequence_matches,
    reset_sequence_diagnostic,
)

RunProcess = Callable[..., subprocess.CompletedProcess[str]]
ResolveContainerImage = Callable[[str], str]
RemoveNamedContainer = Callable[[str], None]

MAX_JSON_BYTES = 1024 * 1024
MAX_FINAL_RESULT_BYTES = 1024 * 1024 * 1024
MAX_FINAL_RESULT_ENTRIES = 8192
_READ_CHUNK_BYTES = 1024 * 1024
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PUBLIC_BUNDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FINAL_MANIFEST_V1_FIELDS = {
    "schema_version",
    "status",
    "started_at",
    "completed_at",
    "correct",
    "metrics",
    "declared_final_metrics",
    "task_tree_sha256",
    "evaluator_sha256",
    "program_tree_sha256",
    "results_tree_sha256",
}
_FINAL_MANIFEST_V2_FIELDS = _FINAL_MANIFEST_V1_FIELDS | {"outcome"}
_FINAL_MANIFEST_V3_FIELDS = _FINAL_MANIFEST_V2_FIELDS | {"evaluation_evidence", "compute_usage"}
_INVOCATION_FIELDS = {
    "schema_version",
    "status",
    "started_at",
    "task_tree_sha256",
    "evaluator_sha256",
    "program_tree_sha256",
}
_RETRYABLE_CONTAINER_LAUNCH_CODES = frozenset({125, 126, 127})
_FINAL_LAUNCH_ATTEMPTS = 3
_FINAL_CLEANUP_ATTEMPTS = 3
_LAUNCH_STDERR_EXCERPT_BYTES = 4096
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_SAFE_LAUNCH_EXCERPTS = {
    "daemon_unavailable": "Cannot connect to the container runtime daemon.",
    "image_unavailable": "The configured container image is unavailable.",
    "container_name_conflict": "The evaluator container name is already in use.",
    "cidfile_conflict": "The evaluator container ID file already exists.",
    "mount_failure": "The container runtime rejected an evaluator mount.",
    "permission_failure": "The container runtime reported a permission failure.",
    "resource_exhaustion": "The container runtime reported exhausted host resources.",
    "executable_failure": "The evaluator executable could not be started.",
    "unknown": "Container runtime stderr withheld; use its byte count and digest.",
}
_TRANSIENT_LAUNCH_CATEGORIES = frozenset(
    {
        "daemon_unavailable",
        "container_name_conflict",
        "cidfile_conflict",
        "resource_exhaustion",
    }
)


class FinalEvaluationError(RuntimeError):
    """Raised when final evaluation cannot run or cannot be trusted."""


class FinalEvaluationInfrastructureError(FinalEvaluationError):
    """Raised when host infrastructure prevents final evaluation."""


class GpuAllocationInvalidError(FinalEvaluationInfrastructureError):
    """Raised when a container's GPU selector is malformed or insufficient."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _remaining_final_evaluation_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise FinalEvaluationError("final evaluation deadline exceeded")
    return remaining


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FinalEvaluationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FinalEvaluationError(f"invalid {label}: {path}")
    try:
        if path.stat(follow_symlinks=False).st_size > MAX_JSON_BYTES:
            raise FinalEvaluationError(f"{label} exceeds size limit: {path}")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except FinalEvaluationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalEvaluationError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FinalEvaluationError(f"invalid {label}: {path}")
    return value


def _write_exclusive_json(path: Path, value: dict[str, Any], *, mode: int = 0o444) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_declared_file(task_dir: Path, relative: str, label: str) -> Path:
    candidate = task_dir / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise FinalEvaluationError(f"invalid {label}: {relative}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(task_dir):
        raise FinalEvaluationError(f"{label} escapes task directory")
    return resolved


def resolve_frozen_incumbent(investigation_dir: Path) -> tuple[Path, Path]:
    investigation_dir = investigation_dir.expanduser().resolve()
    try:
        downstream_program = resolve_downstream_program(investigation_dir)
    except ValueError as exc:
        raise FinalEvaluationError(f"invalid downstream selection: {exc}") from exc
    if downstream_program is not None:
        return downstream_program
    manifest = _load_json(investigation_dir / "best_program_manifest.json", "best-program manifest")
    raw_program_dir = manifest.get("program_dir")
    raw_entrypoint = manifest.get("entrypoint_path", "src/program.py")
    if not isinstance(raw_program_dir, str) or not isinstance(raw_entrypoint, str):
        raise FinalEvaluationError("best-program manifest has no frozen incumbent")
    program_dir = Path(raw_program_dir).expanduser().resolve()
    if not program_dir.is_relative_to(investigation_dir) or not program_dir.is_dir():
        raise FinalEvaluationError("frozen incumbent escapes investigation results")
    relative_entrypoint = Path(raw_entrypoint)
    if relative_entrypoint.is_absolute() or ".." in relative_entrypoint.parts:
        raise FinalEvaluationError("invalid frozen incumbent entrypoint")
    entrypoint = (program_dir / relative_entrypoint).resolve()
    if not entrypoint.is_relative_to(program_dir) or not entrypoint.is_file():
        raise FinalEvaluationError("invalid frozen incumbent entrypoint")
    return program_dir, entrypoint


def _tree_digest(
    root: Path,
    *,
    max_bytes: int | None = None,
    max_entries: int | None = None,
) -> str:
    if root.is_symlink() or not root.is_dir():
        raise FinalEvaluationError(f"invalid digest root: {root}")
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    total_bytes = 0
    file_sizes: dict[Path, int] = {}
    for index, path in enumerate(entries, start=1):
        if max_entries is not None and index > max_entries:
            raise FinalEvaluationError(f"result entry count exceeds size limit: {max_entries}")
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise FinalEvaluationError(f"digest tree contains symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise FinalEvaluationError(f"unsupported digest tree entry: {relative}")
        file_size = path.stat(follow_symlinks=False).st_size
        file_sizes[path] = file_size
        total_bytes += file_size
        if max_bytes is not None and total_bytes > max_bytes:
            raise FinalEvaluationError(f"result tree exceeds size limit: {max_bytes} bytes")

    hasher = hashlib.sha256()
    for path in entries:
        relative_bytes = path.relative_to(root).as_posix().encode()
        if path.is_dir():
            hasher.update(b"dir\0" + relative_bytes + b"\0")
            continue
        size = file_sizes[path]
        hasher.update(b"file\0" + relative_bytes + b"\0" + str(size).encode() + b"\0")
        with path.open("rb") as handle:
            remaining = size
            while remaining and (chunk := handle.read(min(_READ_CHUNK_BYTES, remaining))):
                hasher.update(chunk)
                remaining -= len(chunk)
            if remaining or handle.read(1):
                raise FinalEvaluationError(f"digest tree changed while hashing: {path}")
    return "sha256:" + hasher.hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK_BYTES):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _seal_results(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444, follow_symlinks=False)
    root.chmod(0o555)


def _validate_metrics(
    results_dir: Path, declared_names: tuple[str, ...]
) -> tuple[dict[str, Any], bool]:
    try:
        result_entries = {path.name for path in results_dir.iterdir()}
    except OSError as exc:
        raise FinalEvaluationError("final results directory is unavailable") from exc
    from src.investigation.evaluation_evidence import EVIDENCE_FILENAME, load_evaluation_evidence

    required = {"metrics.json", "correct.json"}
    if not required <= result_entries or result_entries - (required | {EVIDENCE_FILENAME}):
        raise FinalEvaluationError("final evaluator emitted undeclared result artifacts")
    try:
        load_evaluation_evidence(results_dir)
    except ValueError as exc:
        raise FinalEvaluationError(str(exc)) from exc
    metrics = _load_json(results_dir / "metrics.json", "final metrics")
    correctness = _load_json(results_dir / "correct.json", "final correctness")
    if set(correctness) != {"correct"} or not isinstance(correctness["correct"], bool):
        raise FinalEvaluationError(
            "final correctness must contain exactly one boolean field: correct"
        )
    if set(metrics) != set(declared_names):
        raise FinalEvaluationError(
            "final metric names must exactly match declared metrics: "
            f"declared={sorted(declared_names)!r}, actual={sorted(metrics)!r}"
        )
    for name in declared_names:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FinalEvaluationError(f"declared final metric is not numeric: {name}")
        if not math.isfinite(float(value)):
            raise FinalEvaluationError(f"declared final metric is not finite: {name}")
    return metrics, correctness["correct"]


def _expect_exact_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = expected - value.keys()
    unexpected = value.keys() - expected
    if missing:
        raise FinalEvaluationError(f"{label} missing fields: {sorted(missing)!r}")
    if unexpected:
        raise FinalEvaluationError(f"{label} has unexpected fields: {sorted(unexpected)!r}")


def _validate_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise FinalEvaluationError(f"invalid {label}")
    return value


def verify_final_manifest(
    output_dir: Path,
    *,
    task_dir: Path | None = None,
    program_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify a completed final artifact and, when supplied, its frozen inputs."""
    output_dir = output_dir.expanduser()
    if output_dir.is_symlink():
        raise FinalEvaluationError("final output directory must not be a symlink")
    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "final_manifest.json", "final manifest")
    schema_version = manifest.get("schema_version")
    if schema_version == 1:
        _expect_exact_fields(manifest, _FINAL_MANIFEST_V1_FIELDS, "final manifest")
    elif schema_version == 2:
        _expect_exact_fields(manifest, _FINAL_MANIFEST_V2_FIELDS, "final manifest")
    elif schema_version == 3:
        _expect_exact_fields(manifest, _FINAL_MANIFEST_V3_FIELDS, "final manifest")
    else:
        raise FinalEvaluationError("final manifest has an unsupported schema version")
    if manifest["status"] != "completed":
        raise FinalEvaluationError("final manifest is not a completed artifact")
    correct = manifest["correct"]
    if not isinstance(correct, bool):
        raise FinalEvaluationError("final manifest correct must be boolean")
    if schema_version == 1 and correct is not True:
        raise FinalEvaluationError("schema-v1 final manifest must report correct=true")
    if schema_version in (2, 3):
        expected_outcome = "qualified" if correct else "not_qualified"
        if manifest["outcome"] != expected_outcome:
            raise FinalEvaluationError("final manifest outcome does not match correct")
    for field in ("started_at", "completed_at"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise FinalEvaluationError(f"invalid final manifest {field}")
    declared_raw = manifest["declared_final_metrics"]
    if (
        not isinstance(declared_raw, list)
        or not declared_raw
        or any(not isinstance(name, str) or not name for name in declared_raw)
        or len(set(declared_raw)) != len(declared_raw)
    ):
        raise FinalEvaluationError("invalid declared_final_metrics")
    declared_names = tuple(declared_raw)
    if not isinstance(manifest["metrics"], dict):
        raise FinalEvaluationError("invalid final manifest metrics")
    for name in declared_names:
        value = manifest["metrics"].get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FinalEvaluationError(f"declared final metric is not numeric: {name}")
        if not math.isfinite(float(value)):
            raise FinalEvaluationError(f"declared final metric is not finite: {name}")
    if set(manifest["metrics"]) != set(declared_names):
        raise FinalEvaluationError("manifest metrics do not exactly match declared metrics")

    for field in (
        "task_tree_sha256",
        "evaluator_sha256",
        "program_tree_sha256",
        "results_tree_sha256",
    ):
        _validate_digest(manifest[field], field)
    results_dir = output_dir / "results"
    results_digest = _tree_digest(
        results_dir,
        max_bytes=MAX_FINAL_RESULT_BYTES,
        max_entries=MAX_FINAL_RESULT_ENTRIES,
    )
    if manifest["results_tree_sha256"] != results_digest:
        raise FinalEvaluationError("final results digest mismatch")
    result_metrics, result_correct = _validate_metrics(results_dir, declared_names)
    if result_metrics != manifest["metrics"]:
        raise FinalEvaluationError("final manifest metrics differ from sealed results")
    if result_correct is not correct:
        raise FinalEvaluationError("final manifest correct differs from sealed results")
    if schema_version == 3:
        from src.investigation.evaluation_evidence import evidence_metadata
        from src.investigation.evaluation_usage import validate_compute_usage

        try:
            expected_evidence = evidence_metadata(results_dir)
            # Locations may move when a complete run is copied for inspection.
            expected_evidence.pop("path", None)
            if manifest["evaluation_evidence"] != expected_evidence:
                raise ValueError("final manifest evidence differs from sealed results")
            validate_compute_usage(manifest["compute_usage"])
            if manifest["compute_usage"]["outcome"] != "completed":
                raise ValueError("completed final manifest has failed compute usage")
        except ValueError as exc:
            raise FinalEvaluationError(str(exc)) from exc

    invocation = _load_json(output_dir / "invocation.json", "final invocation")
    _expect_exact_fields(invocation, _INVOCATION_FIELDS, "final invocation")
    invocation_expected = {
        "schema_version": 1,
        "status": "started",
        "started_at": manifest["started_at"],
        "task_tree_sha256": manifest["task_tree_sha256"],
        "evaluator_sha256": manifest["evaluator_sha256"],
        "program_tree_sha256": manifest["program_tree_sha256"],
    }
    if invocation != invocation_expected:
        raise FinalEvaluationError("final invocation does not match final manifest")

    if task_dir is not None:
        resolved_task = task_dir.expanduser().resolve()
        try:
            task_digest = task_scaffold_fingerprint(resolved_task)
            contract = load_task_contract(resolved_task / TASK_CONTRACT_FILENAME)
        except TaskContractError as exc:
            raise FinalEvaluationError(f"invalid task contract: {exc}") from exc
        if task_digest != manifest["task_tree_sha256"]:
            raise FinalEvaluationError("final task digest mismatch")
        evaluator = _safe_declared_file(resolved_task, contract.evaluators.final, "final evaluator")
        if _file_digest(evaluator) != manifest["evaluator_sha256"]:
            raise FinalEvaluationError("final evaluator digest mismatch")
        contract_metrics = tuple(metric.name for metric in contract.metrics.final)
        if contract_metrics != declared_names:
            raise FinalEvaluationError("final manifest metrics differ from task contract")
        if schema_version == 3:
            usage = manifest["compute_usage"]
            if (usage["max_rpc_calls"] != contract.compute.final.max_candidate_rpc_calls
                    or usage["wall_clock_limit_seconds"] > contract.compute.final.wall_clock_seconds):
                raise FinalEvaluationError("final compute usage limits differ from task contract")
    if program_dir is not None:
        if _tree_digest(program_dir.expanduser().resolve()) != manifest["program_tree_sha256"]:
            raise FinalEvaluationError("final program digest mismatch")
    return manifest


def _evaluation_command(
    *,
    task_dir: Path,
    program_dir: Path,
    entrypoint: Path,
    evaluator: Path,
    results_dir: Path,
    python_bin: Path,
    container_image: str | None,
    gpu_count: int,
    cpu_cores: int = 1,
    memory_gb: int = 1,
    environment: Mapping[str, str],
    broker_dir: Path | None = None,
    candidate_editable_paths: tuple[str, ...] = (),
    sealed_task_paths: tuple[str, ...] = (),
    rpc_timeout_seconds: int = 21600,
    plane: str = "final",
    draft_smoke: bool = False,
    qualification: bool = False,
    evaluator_source_dir: Path | None = None,
    runtime_venv_dir: Path | None = None,
    public_artifact_bundle: str | None = None,
    container_name: str | None = None,
) -> list[str]:
    if plane not in {"search", "final"}:
        raise FinalEvaluationError("evaluation plane must be search or final")
    if not container_image:
        return [
            str(python_bin),
            str(evaluator),
            "--program_path",
            str(entrypoint),
            "--results_dir",
            str(results_dir),
        ]
    evaluator_root = evaluator_source_dir or task_dir
    try:
        evaluator_relative = evaluator.relative_to(evaluator_root)
    except ValueError as exc:
        raise FinalEvaluationError("evaluator escapes its declared source directory") from exc
    evaluator_mount = "/task" if evaluator_root == task_dir else "/evaluator-source"
    runtime_venv = runtime_venv_dir or task_dir / ".venv"
    runtime_mount = "/task/.venv" if runtime_venv_dir is None else "/task-venv"
    command = ["docker", "run"]
    if container_name is None:
        command.append("--rm")
    else:
        command.extend(("--name", container_name))
    command.extend(
        [
            "--cidfile",
        str(broker_dir / "evaluator.cid") if broker_dir is not None else "",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "1024",
        "--cpus",
        str(cpu_cores),
        "--memory",
        f"{memory_gb}g",
        "--memory-swap",
        f"{memory_gb}g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=1g",
        "--mount",
            f"type=bind,src={task_dir},dst=/task,readonly",
        ]
    )
    _mask_task_snapshot_metadata(command, task_dir, "/task")
    if evaluator_root != task_dir:
        command.extend(
            ("--mount", f"type=bind,src={evaluator_root},dst=/evaluator-source,readonly")
        )
        _mask_task_snapshot_metadata(command, evaluator_root, "/evaluator-source")
    if runtime_venv_dir is not None:
        command.extend(("--mount", f"type=bind,src={runtime_venv},dst=/task-venv,readonly"))
    command.extend(
        [
            "--mount",
            f"type=bind,src={Path(__file__).parent},dst=/broker-code,readonly",
            "--mount",
            f"type=bind,src={results_dir},dst=/results",
        ]
    )
    if broker_dir is None:
        raise FinalEvaluationError("candidate broker directory is required")
    command.extend(
        (
            "--mount",
            f"type=bind,src={broker_dir},dst=/broker",
            "--env",
            "AGENTIC_CANDIDATE_SOCKET=/broker/candidate.sock",
            "--env",
            f"AGENTIC_CANDIDATE_RPC_TIMEOUT={rpc_timeout_seconds}",
            "--env",
            f"TASK_EVALUATION_PLANE={plane}",
            "--env",
            f"AGENTIC_EVALUATOR_IMAGE_ID={container_image}",
            "--env",
            "TASK_SOURCE_DIR=/task",
            "--env",
            "HOME=/tmp",
            "--env",
            "XDG_CACHE_HOME=/tmp/.cache",
            "--env",
            "USER=agent",
            "--env",
            "LOGNAME=agent",
        )
    )
    if container_name is not None:
        command.extend(
            ("--env", "AGENTIC_EVALUATOR_STARTED_SOCKET=/broker/evaluator-start.sock")
        )
    if draft_smoke:
        command.extend(("--env", "AGENTIC_TASK_DRAFT_SMOKE=1"))
    if qualification:
        command.extend(("--env", "AGENTIC_TASK_QUALIFICATION=1"))
    public_artifacts = _public_artifact_root(environment, public_artifact_bundle, task_dir)
    if public_artifacts is not None:
        command.extend(
            (
                "--mount",
                f"type=bind,src={public_artifacts},dst=/public-artifacts/{public_artifacts.name},readonly",
                "--env",
                "AGENTIC_PUBLIC_ASSETS_DIR=/public-artifacts",
            )
        )
    masked_task_paths = {
        relative
        for relative in {"program_0", *candidate_editable_paths, *sealed_task_paths}
        if not relative.startswith("program_0/")
    }
    for relative in sorted(masked_task_paths):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise FinalEvaluationError("task mask path must be a safe relative path")
        source = task_dir / relative
        destination = f"/task/{relative}"
        if source.is_dir():
            command.extend(
                ("--tmpfs", f"{destination}:ro,noexec,nosuid,nodev,size=64k,nr_inodes=16")
            )
        elif source.is_file():
            command.extend(("--mount", f"type=bind,src=/dev/null,dst={destination},readonly"))
    if gpu_count:
        gpu_selector = _exact_gpu_selector(environment, gpu_count, offset=0)
        command.extend(["--gpus", gpu_selector])
    command.extend([container_image, f"{runtime_mount}/bin/python"])
    if container_name is not None:
        command.extend(("-I", "-S", "/broker-code/evaluator_launcher.py", f"{runtime_mount}/bin/python"))
    command.extend(
        [
            f"{evaluator_mount}/{evaluator_relative.as_posix()}",
            "--program_path",
            "/broker-code/candidate_proxy.py",
            "--results_dir",
            "/results",
        ]
    )
    return command


def _mask_task_snapshot_metadata(
    command: list[str], source_root: Path, container_root: str
) -> None:
    try:
        metadata_present = task_snapshot_metadata_present(source_root)
    except ValueError as exc:
        raise FinalEvaluationError(str(exc)) from exc
    if metadata_present:
        command.extend(
            (
                "--mount",
                "type=bind,src=/dev/null,"
                f"dst={container_root}/{TASK_SNAPSHOT_METADATA_FILENAME},readonly",
            )
        )


def _public_artifact_root(
    environment: Mapping[str, str], bundle: str | None, task_dir: Path
) -> Path | None:
    if bundle is None:
        return None
    if _PUBLIC_BUNDLE_RE.fullmatch(bundle) is None:
        raise FinalEvaluationError("final runtime declared an unsafe public artifact bundle")
    raw = environment.get("AGENTIC_PUBLIC_ASSETS_DIR", "").strip()
    if not raw:
        raise FinalEvaluationError("final runtime requires AGENTIC_PUBLIC_ASSETS_DIR")
    declared = Path(raw)
    if not declared.is_absolute() or declared.is_symlink() or ".." in declared.parts:
        raise FinalEvaluationError("public artifact root must be an absolute regular directory")
    current = Path("/")
    for component in declared.parts[1:]:
        current /= component
        if current.is_symlink():
            raise FinalEvaluationError("public artifact root must not cross symlinks")
    try:
        resolved = declared.resolve(strict=True)
        root_metadata = resolved.stat(follow_symlinks=False)
    except OSError as exc:
        raise FinalEvaluationError("public artifact root is unavailable") from exc
    if (
        resolved == Path("/")
        or resolved == Path.home().resolve()
        or task_dir.resolve().is_relative_to(resolved)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid not in {0, os.getuid()}
        or root_metadata.st_mode & 0o022
    ):
        raise FinalEvaluationError("public artifact root must be narrow and owner-controlled")
    family_root = resolved / bundle
    try:
        family_metadata = family_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise FinalEvaluationError("public artifact family bundle is unavailable") from exc
    if (
        family_root.is_symlink()
        or not stat.S_ISDIR(family_metadata.st_mode)
        or family_metadata.st_uid not in {0, os.getuid()}
    ):
        raise FinalEvaluationError("public artifact family bundle must be immutable")
    for path in (family_root, *family_root.rglob("*")):
        metadata = path.stat(follow_symlinks=False)
        if (
            path.is_symlink()
            or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
            or metadata.st_uid not in {0, os.getuid()}
            or metadata.st_mode & 0o222
        ):
            raise FinalEvaluationError("public artifact family bundle must be immutable")
    return family_root


def _exact_gpu_selector(environment: Mapping[str, str], gpu_count: int, *, offset: int) -> str:
    try:
        return bounded_docker_gpu_selector(
            environment.get("AGENT_CONTAINER_GPUS"),
            gpu_count,
            offset=offset,
        )
    except GpuSelectorError as exc:
        raise GpuAllocationInvalidError(GPU_ALLOCATION_INVALID_CATEGORY) from exc


def _force_remove_container(cid_path: Path) -> None:
    try:
        container_id = read_container_id(cid_path)
    except CandidateBrokerInfrastructureError as exc:
        raise FinalEvaluationError("invalid final evaluator container ID") from exc
    if container_id is None:
        return
    try:
        result = subprocess.run(
            ["docker", "rm", "-f", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FinalEvaluationError("could not remove final evaluator container") from exc
    if result.returncode != 0 and "no such container" not in result.stderr.lower():
        raise FinalEvaluationError("could not remove final evaluator container")
    try:
        cid_path.unlink(missing_ok=True)
    except OSError as exc:
        raise ContainerMetadataCleanupError(
            "evaluator container-id cleanup failed; recovery metadata retained"
        ) from exc


def _force_remove_named_container(container_name: str) -> None:
    try:
        result = subprocess.run(
            ["docker", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FinalEvaluationError("could not remove named final evaluator container") from exc
    if result.returncode != 0 and "no such container" not in result.stderr.lower():
        raise FinalEvaluationError("could not remove named final evaluator container")


def _final_evaluator_container_name(output_dir: Path) -> str:
    identity = hashlib.sha256(os.fsencode(output_dir.resolve())).hexdigest()[:24]
    return f"agentic-final-{os.getuid()}-{identity}"


class _EvaluatorLaunchWitness:
    _TOKEN = b"evaluator-started-v1"

    def __init__(self, broker_dir: Path) -> None:
        self.path = broker_dir / "evaluator-start.sock"
        self._socket: socket.socket | None = None

    def __enter__(self) -> "_EvaluatorLaunchWitness":
        try:
            self.path.parent.mkdir(mode=0o700)
        except FileExistsError:
            pass
        parent_metadata = self.path.parent.lstat()
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.getuid()
            or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        ):
            raise FinalEvaluationError("unsafe final evaluator launch witness directory")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            listener.bind(str(self.path))
            os.chmod(self.path, 0o600)
            listener.setblocking(False)
        except BaseException:
            listener.close()
            raise
        self._socket = listener
        return self

    def observed(self) -> bool:
        if self._socket is None:
            raise FinalEvaluationError("final evaluator launch witness is unavailable")
        try:
            payload = self._socket.recv(64)
        except BlockingIOError:
            return False
        except OSError as exc:
            raise FinalEvaluationError("final evaluator launch witness failed") from exc
        if payload != self._TOKEN:
            raise FinalEvaluationError("invalid final evaluator launch witness")
        return True

    def __exit__(self, *_args: object) -> None:
        if self._socket is not None:
            self._socket.close()
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise FinalEvaluationError("unsafe final evaluator launch witness path")
        self.path.unlink()
        try:
            self.path.parent.rmdir()
        except OSError:
            pass


def _recover_broker_directory(broker_dir: Path) -> None:
    """Reclaim retained broker containers before reusing a durable directory."""
    _reject_legacy_broker_directory(broker_dir)
    try:
        metadata = broker_dir.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise FinalEvaluationError("unsafe broker recovery directory")
    for name in ("candidate.cid", "evaluator.cid"):
        _force_remove_container(broker_dir / name)
    socket_path = broker_dir / "candidate.sock"
    try:
        socket_metadata = socket_path.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISSOCK(socket_metadata.st_mode) or socket_metadata.st_uid != os.getuid():
            raise FinalEvaluationError("unsafe broker recovery socket")
        socket_path.unlink()
    scoring_mask = broker_dir / "scoring-mask"
    if scoring_mask.exists() or scoring_mask.is_symlink():
        mask_metadata = scoring_mask.lstat()
        if (
            not stat.S_ISDIR(mask_metadata.st_mode)
            or mask_metadata.st_uid != os.getuid()
            or stat.S_IMODE(mask_metadata.st_mode) != 0o700
        ):
            raise FinalEvaluationError("unsafe broker scoring mask")
        shutil.rmtree(scoring_mask)
    try:
        broker_dir.rmdir()
    except OSError as exc:
        raise FinalEvaluationError("broker recovery directory contains unexpected entries") from exc


def _final_broker_directory(output_dir: Path) -> Path:
    identity = hashlib.sha256(os.fsencode(output_dir.resolve())).hexdigest()[:32]
    return _broker_recovery_root() / f"final-{identity}"


def _broker_recovery_root() -> Path:
    return broker_root()


def _ensure_broker_recovery_root() -> Path:
    try:
        return ensure_broker_root(_broker_recovery_root())
    except (OSError, ValueError) as exc:
        raise FinalEvaluationError(f"broker recovery root is unavailable: {exc}") from exc


def _reject_legacy_broker_directory(directory: Path) -> None:
    try:
        reject_legacy_broker_directory(directory)
    except (OSError, ValueError) as exc:
        raise FinalEvaluationError(str(exc)) from exc


@contextmanager
def _final_evaluation_lock(output_dir: Path) -> Any:
    if output_dir.is_symlink():
        raise FinalEvaluationError("final output directory must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = output_dir.stat(follow_symlinks=False)
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o022:
        raise FinalEvaluationError("final output directory must be owner-controlled")
    descriptor = os.open(
        output_dir / ".broker.lock",
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_uid != os.getuid()
            or lock_metadata.st_mode & 0o077
        ):
            raise FinalEvaluationError("unsafe final evaluation lock")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _candidate_command(
    *,
    task_dir: Path,
    program_dir: Path,
    entrypoint: Path,
    container_image: str,
    contract: Any,
    environment: Mapping[str, str],
    broker_dir: Path,
    plane: str = "final",
    runtime_venv_dir: Path | None = None,
) -> list[str]:
    if plane not in {"search", "final"}:
        raise FinalEvaluationError("candidate plane must be search or final")
    compute = getattr(contract.compute, plane)
    runtime_venv = runtime_venv_dir or task_dir / ".venv"
    scoring_packages = sorted((runtime_venv / "lib").glob("python*/site-packages/src/investigation"))
    if len(scoring_packages) != 1 or not scoring_packages[0].is_dir():
        raise FinalEvaluationError("candidate runtime lacks one maskable trusted scoring package")
    scoring_relative = scoring_packages[0].relative_to(runtime_venv).as_posix()
    command = [
        "docker",
        "run",
        "--cidfile",
        str(broker_dir / "candidate.cid"),
        "-i",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "1024",
        "--cpus",
        str(compute.cpu_cores),
        "--memory",
        f"{compute.memory_gb}g",
        "--memory-swap",
        f"{compute.memory_gb}g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=1g",
        "--mount",
        f"type=bind,src={program_dir},dst=/program,readonly",
        "--mount",
        f"type=bind,src={runtime_venv},dst=/task-venv,readonly",
        "--mount",
        f"type=bind,src={Path(__file__).parent},dst=/broker-code,readonly",
        "--mount",
        (f"type=bind,src={broker_dir / 'scoring-mask'},dst=/task-venv/{scoring_relative},readonly"),
    ]
    if compute.candidate_gpu_count:
        selector = _exact_gpu_selector(
            environment,
            compute.candidate_gpu_count,
            offset=compute.evaluator_gpu_count,
        )
        command.extend(("--gpus", selector))
    relative_entrypoint = entrypoint.relative_to(program_dir)
    command.extend(
        (
            container_image,
            "/task-venv/bin/python",
            "/broker-code/candidate_worker.py",
            f"/program/{relative_entrypoint.as_posix()}",
        )
    )
    return command


def _output_bytes(value: str | bytes | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="replace")


def _run_final_process(
    command: list[str],
    *,
    task_dir: Path,
    timeout_seconds: float,
    run: RunProcess,
) -> BoundedProcessResult:
    if run is subprocess.run:
        return run_bounded_process(
            command,
            cwd=task_dir,
            env=os.environ,
            timeout_seconds=timeout_seconds,
        )
    completed = run(
        command,
        cwd=str(task_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    stdout = _output_bytes(completed.stdout)
    stderr = _output_bytes(completed.stderr)
    if len(stdout) + len(stderr) > DEFAULT_MAX_OUTPUT_BYTES:
        raise FinalEvaluationError("final evaluator exceeded the combined output limit")
    return BoundedProcessResult(completed.returncode, stdout, stderr)


def _rollback_unstarted_final_attempt(
    *,
    invocation_path: Path,
    invocation: dict[str, Any],
    results_dir: Path,
) -> bool:
    expected_payload = (json.dumps(invocation, indent=2, sort_keys=True) + "\n").encode()
    parent_descriptor = -1
    results_descriptor = -1
    invocation_descriptor = -1
    try:
        parent_descriptor = os.open(
            invocation_path.parent,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        results_descriptor = os.open(
            results_dir.name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        invocation_descriptor = os.open(
            invocation_path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        results_metadata = os.fstat(results_descriptor)
        invocation_metadata = os.fstat(invocation_descriptor)
        if (
            results_metadata.st_uid != os.getuid()
            or invocation_metadata.st_uid != os.getuid()
            or not stat.S_ISREG(invocation_metadata.st_mode)
        ):
            return False
        with os.scandir(results_descriptor) as entries:
            if next(entries, None) is not None:
                return False
        payload = os.read(invocation_descriptor, len(expected_payload) + 1)
        if payload != expected_payload:
            return False
        os.rmdir(results_dir.name, dir_fd=parent_descriptor)
        os.unlink(invocation_path.name, dir_fd=parent_descriptor)
        return True
    except OSError:
        return False
    finally:
        for descriptor in (
            invocation_descriptor,
            results_descriptor,
            parent_descriptor,
        ):
            if descriptor >= 0:
                os.close(descriptor)


def _clear_launch_failure(path: Path) -> None:
    if not (path.exists() or path.is_symlink()):
        return
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise FinalEvaluationError("final launch diagnostic is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise FinalEvaluationError("unsafe final launch diagnostic")
    path.unlink()


def _sanitize_launch_stderr(stderr: bytes, category: str) -> tuple[str, bool]:
    excerpt = _SAFE_LAUNCH_EXCERPTS.get(category, _SAFE_LAUNCH_EXCERPTS["unknown"])
    return excerpt, len(stderr) > _LAUNCH_STDERR_EXCERPT_BYTES


def _classify_launch_failure(return_code: int, stderr_excerpt: str) -> str:
    message = stderr_excerpt.casefold()
    if any(
        marker in message
        for marker in (
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "error during connect",
            "connection refused",
        )
    ):
        return "daemon_unavailable"
    if any(
        marker in message
        for marker in ("no such image", "manifest unknown", "pull access denied")
    ):
        return "image_unavailable"
    if "container name" in message and "already in use" in message:
        return "container_name_conflict"
    if "cidfile" in message or "cid file" in message:
        return "cidfile_conflict"
    if any(
        marker in message
        for marker in (
            "invalid mount config",
            "mounts denied",
            "failed to mount",
            "error mounting",
            "error while creating mount source path",
        )
    ):
        return "mount_failure"
    if "permission denied" in message or "operation not permitted" in message:
        return "permission_failure"
    if any(
        marker in message
        for marker in (
            "cannot allocate memory",
            "no space left on device",
            "resource temporarily unavailable",
            "too many open files",
        )
    ):
        return "resource_exhaustion"
    if return_code in {126, 127} or any(
        marker in message
        for marker in ("executable file not found", "exec format error")
    ):
        return "executable_failure"
    return "unknown"


def _launch_attempt_diagnostic(
    attempt: int,
    completed: BoundedProcessResult,
) -> dict[str, Any]:
    classification_text = _ANSI_ESCAPE_RE.sub(
        "", completed.stderr.decode("utf-8", errors="replace")
    )
    category = _classify_launch_failure(completed.returncode, classification_text)
    stderr_excerpt, stderr_truncated = _sanitize_launch_stderr(completed.stderr, category)
    return {
        "attempt": attempt,
        "return_code": completed.returncode,
        "category": category,
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": "sha256:" + hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_bytes": len(completed.stderr),
        "stderr_sha256": "sha256:" + hashlib.sha256(completed.stderr).hexdigest(),
        "stderr_excerpt": stderr_excerpt,
        "stderr_truncated": stderr_truncated,
    }


def _write_launch_failure(
    path: Path,
    *,
    launch_attempts: list[dict[str, Any]],
) -> None:
    final_attempt = launch_attempts[-1]
    _write_exclusive_json(
        path,
        {
            "schema_version": 2,
            "status": "launch_failed",
            "failed_at": _utc_now(),
            "reason": "container_runtime_launch_failed",
            "return_code": final_attempt["return_code"],
            "attempts": len(launch_attempts),
            "stdout_bytes": final_attempt["stdout_bytes"],
            "stdout_sha256": final_attempt["stdout_sha256"],
            "stderr_bytes": final_attempt["stderr_bytes"],
            "stderr_sha256": final_attempt["stderr_sha256"],
            "launch_attempts": launch_attempts,
        },
        mode=0o400,
    )


def _write_cleanup_failure(path: Path) -> None:
    _write_exclusive_json(
        path,
        {
            "schema_version": 1,
            "status": "cleanup_failed",
            "failed_at": _utc_now(),
            "reason": "container_cleanup_failed",
            "attempts": _FINAL_CLEANUP_ATTEMPTS,
            "invocation_retained": True,
        },
        mode=0o400,
    )


def _launch_failure_message(
    launch_attempts: list[dict[str, Any]],
    diagnostic_path: Path,
) -> str:
    final_attempt = launch_attempts[-1]
    excerpt = str(final_attempt["stderr_excerpt"]).strip()
    if len(excerpt) > 512:
        excerpt = excerpt[:512] + "…"
    detail = excerpt or "no stderr was captured"
    return (
        "final evaluator container failed to start after "
        f"{len(launch_attempts)} attempts (return code {final_attempt['return_code']}, "
        f"category {final_attempt['category']}): {detail}; diagnostics: {diagnostic_path}"
    )


def _retryable_launch_failure(return_code: int, category: str) -> bool:
    if category in _TRANSIENT_LAUNCH_CATEGORIES or category == "image_unavailable":
        return True
    return category == "unknown" and return_code == 125


def _cleanup_evaluator_container(
    cid_path: Path,
    container_name: str,
    remove_named_container: RemoveNamedContainer,
) -> bool:
    cleanup_succeeded = True
    metadata_error: ContainerMetadataCleanupError | None = None
    try:
        _force_remove_container(cid_path)
    except ContainerMetadataCleanupError as exc:
        metadata_error = exc
    except FinalEvaluationError:
        cleanup_succeeded = False
    try:
        remove_named_container(container_name)
    except FinalEvaluationError:
        cleanup_succeeded = False
    if cleanup_succeeded and metadata_error is not None:
        raise metadata_error
    return cleanup_succeeded


def _retry_evaluator_cleanup(
    cid_path: Path,
    container_name: str,
    remove_named_container: RemoveNamedContainer,
    sleep: Callable[[float], None],
) -> bool:
    for retry in range(2):
        sleep(float(2**retry))
        if _cleanup_evaluator_container(cid_path, container_name, remove_named_container):
            return True
    return False


def run_final_evaluation(
    *,
    task_dir: Path,
    investigation_results_dir: Path,
    output_dir: Path,
    python_bin: Path,
    timeout_seconds: int = 21600,
    container_image: str | None = None,
    container_image_recovery_ref: str | None = None,
    environment: Mapping[str, str] = os.environ,
    run: RunProcess = subprocess.run,
    _allow_native_for_tests: bool = False,
    _resolve_container_image: ResolveContainerImage = resolve_immutable_image,
    _remove_named_container: RemoveNamedContainer = _force_remove_named_container,
    _sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Serialize final evaluation and recover retained broker containers first."""
    output_dir = Path(os.path.abspath(output_dir.expanduser()))
    with _final_evaluation_lock(output_dir):
        return _run_final_evaluation_locked(
            task_dir=task_dir,
            investigation_results_dir=investigation_results_dir,
            output_dir=output_dir,
            python_bin=python_bin,
            timeout_seconds=timeout_seconds,
            container_image=container_image,
            container_image_recovery_ref=container_image_recovery_ref,
            environment=environment,
            run=run,
            _allow_native_for_tests=_allow_native_for_tests,
            _resolve_container_image=_resolve_container_image,
            _remove_named_container=_remove_named_container,
            _sleep=_sleep,
        )


def _run_final_evaluation_locked(
    *,
    task_dir: Path,
    investigation_results_dir: Path,
    output_dir: Path,
    python_bin: Path,
    timeout_seconds: int = 21600,
    container_image: str | None = None,
    container_image_recovery_ref: str | None = None,
    environment: Mapping[str, str] = os.environ,
    run: RunProcess = subprocess.run,
    _allow_native_for_tests: bool = False,
    _resolve_container_image: ResolveContainerImage = resolve_immutable_image,
    _remove_named_container: RemoveNamedContainer = _force_remove_named_container,
    _sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run the contract's final evaluator once, or reuse a completed manifest."""
    if timeout_seconds < 1:
        raise FinalEvaluationError("final evaluation timeout must be positive")
    task_dir = task_dir.resolve()
    investigation_dir = investigation_results_dir.resolve()
    if output_dir.is_symlink():
        raise FinalEvaluationError("final output directory must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_stat = output_dir.stat(follow_symlinks=False)
    if output_stat.st_uid != os.getuid() or output_stat.st_mode & 0o022:
        raise FinalEvaluationError("final output directory must be owner-controlled")
    manifest_path = output_dir / "final_manifest.json"
    invocation_path = output_dir / "invocation.json"
    launch_failure_path = output_dir / "final_launch_failure.json"
    cleanup_failure_path = output_dir / "final_cleanup_failure.json"
    broker_root = _broker_recovery_root()
    broker_dir = _final_broker_directory(output_dir)
    _reject_legacy_broker_directory(broker_dir)
    if broker_root.exists() or broker_root.is_symlink():
        _ensure_broker_recovery_root()
        _recover_broker_directory(broker_dir)
    try:
        contract = load_task_contract(task_dir / TASK_CONTRACT_FILENAME)
        scaffold_digest = task_scaffold_fingerprint(task_dir)
    except TaskContractError as exc:
        raise FinalEvaluationError(f"invalid task contract: {exc}") from exc
    expected_reset_seeds = contract.protocol.final_reset_sequence
    public_artifact_bundle = contract.resources.public_artifact_bundle
    evaluator = _safe_declared_file(task_dir, contract.evaluators.final, "final evaluator")
    program_dir, entrypoint = resolve_frozen_incumbent(investigation_dir)
    program_digest = _tree_digest(program_dir)
    evaluator_digest = _file_digest(evaluator)
    if manifest_path.exists() or manifest_path.is_symlink():
        if not manifest_path.is_file():
            raise FinalEvaluationError("invalid existing final manifest")
        return verify_final_manifest(
            output_dir,
            task_dir=task_dir,
            program_dir=program_dir,
        )
    if invocation_path.exists() or invocation_path.is_symlink():
        raise FinalEvaluationError("final evaluation already started; refusing rerun")
    if not container_image and not _allow_native_for_tests:
        raise FinalEvaluationError("contracted final evaluation requires container isolation")

    results_dir = output_dir / "results"
    if results_dir.exists() or results_dir.is_symlink():
        raise FinalEvaluationError("final results directory already exists")
    if not container_image:
        resolved_python = python_bin.expanduser().resolve()
        if not resolved_python.is_file() or not os.access(resolved_python, os.X_OK):
            raise FinalEvaluationError(f"invalid final evaluator Python: {python_bin}")
    requested_container_image = container_image
    if container_image:
        try:
            container_image = _resolve_container_image(container_image)
        except ValueError as exc:
            raise FinalEvaluationError(f"final evaluator image preflight failed: {exc}") from exc
        _ensure_broker_recovery_root()
    _clear_launch_failure(launch_failure_path)
    _clear_launch_failure(cleanup_failure_path)
    if broker_dir.exists() or broker_dir.is_symlink():
        raise FinalEvaluationError("candidate broker directory already exists")
    evaluator_container_name = (
        _final_evaluator_container_name(output_dir) if container_image else None
    )
    command = _evaluation_command(
        task_dir=task_dir,
        program_dir=program_dir,
        entrypoint=entrypoint,
        evaluator=evaluator,
        results_dir=results_dir,
        python_bin=python_bin,
        container_image=container_image,
        gpu_count=contract.compute.final.evaluator_gpu_count,
        cpu_cores=contract.compute.final.cpu_cores,
        memory_gb=contract.compute.final.memory_gb,
        environment=environment,
        broker_dir=broker_dir,
        candidate_editable_paths=contract.candidate.editable_paths,
        sealed_task_paths=(
            contract.evaluators.search,
            contract.resources.search_root,
            *(
                artifact.path
                for artifact in contract.provenance.artifacts
                if artifact.role == "search_helper"
            ),
        ),
        rpc_timeout_seconds=min(timeout_seconds, contract.compute.final.wall_clock_seconds),
        public_artifact_bundle=public_artifact_bundle,
        container_name=evaluator_container_name,
    )
    candidate_command = None
    if container_image:
        if contract.candidate.interface != "python_rpc_v1:handle":
            raise FinalEvaluationError("final evaluation requires candidate python_rpc_v1:handle")
        if not (task_dir / ".venv" / "bin" / "python").is_file():
            raise FinalEvaluationError("final evaluation requires a managed task runtime")
        candidate_command = _candidate_command(
            task_dir=task_dir,
            program_dir=program_dir,
            entrypoint=entrypoint,
            container_image=container_image,
            contract=contract,
            environment=environment,
            broker_dir=broker_dir,
        )
    effective_timeout_seconds = min(timeout_seconds, contract.compute.final.wall_clock_seconds)
    evaluation_started = time.monotonic()
    evaluation_deadline = evaluation_started + effective_timeout_seconds
    max_attempts = _FINAL_LAUNCH_ATTEMPTS if candidate_command is not None else 1
    launch_attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        _remaining_final_evaluation_seconds(evaluation_deadline)
        try:
            gpu_count = contract.compute.final.gpu_count
            gpu_lease: GpuAllocationLease | None = (
                acquire_gpu_allocation_lease(
                    environment.get("AGENT_CONTAINER_GPUS"),
                    gpu_count,
                    lease_root=GPU_ALLOCATION_LEASE_ROOT,
                    deadline=evaluation_deadline,
                )
                if gpu_count
                else None
            )
            allocation = gpu_lease if gpu_lease is not None else nullcontext()
        except GpuLeaseTimeout as exc:
            raise FinalEvaluationInfrastructureError(GPU_ALLOCATION_BUSY_CATEGORY) from exc
        except GpuLeaseQuarantined as exc:
            raise FinalEvaluationInfrastructureError(
                GPU_ALLOCATION_QUARANTINED_CATEGORY
            ) from exc
        except GpuLeaseError as exc:
            raise FinalEvaluationInfrastructureError(GPU_ALLOCATION_INVALID_CATEGORY) from exc
        try:
            with allocation:
                if gpu_lease is not None and evaluation_deadline <= time.monotonic():
                    raise FinalEvaluationInfrastructureError(
                        GPU_ALLOCATION_BUSY_CATEGORY
                    )
                evaluator_launch_command = command
                candidate_launch_command = candidate_command
                if gpu_lease is not None and candidate_command is not None:
                    gpu_lease.prepare_runtime()
                    evaluator_launch_command = gpu_lease.wrap_command(command)
                    candidate_launch_command = gpu_lease.wrap_command(candidate_command)
                remaining_timeout = _remaining_final_evaluation_seconds(evaluation_deadline)
                invocation = {
                    "schema_version": 1,
                    "status": "started",
                    "started_at": _utc_now(),
                    "task_tree_sha256": scaffold_digest,
                    "evaluator_sha256": evaluator_digest,
                    "program_tree_sha256": program_digest,
                }
                results_dir.mkdir(mode=0o700)
                _write_exclusive_json(invocation_path, invocation)
                broker: CandidateBroker | None = None
                evaluator_started = False
                evaluator_cleanup_succeeded = True
                from src.investigation.evaluation_usage import record_compute_failure

                try:
                    if candidate_command is None:
                        with record_compute_failure(
                            output_dir, wall_clock_limit_seconds=effective_timeout_seconds,
                            started_at=evaluation_started,
                            max_rpc_calls=contract.compute.final.max_candidate_rpc_calls,
                        ):
                            completed = _run_final_process(
                                evaluator_launch_command, task_dir=task_dir,
                                timeout_seconds=_remaining_final_evaluation_seconds(evaluation_deadline),
                                run=run,
                            )
                    else:
                        assert candidate_launch_command is not None
                        broker = CandidateBroker(
                            candidate_launch_command,
                            broker_dir,
                            rpc_timeout_seconds=remaining_timeout,
                            reset_timeout_seconds=remaining_timeout,
                            max_rpc_calls=contract.compute.final.max_candidate_rpc_calls,
                            expected_reset_seeds=expected_reset_seeds,
                        )
                        with record_compute_failure(
                            output_dir, wall_clock_limit_seconds=effective_timeout_seconds,
                            started_at=evaluation_started,
                            max_rpc_calls=contract.compute.final.max_candidate_rpc_calls,
                            broker=broker,
                        ), broker:
                            with _EvaluatorLaunchWitness(broker_dir) as launch_witness:
                                try:
                                    completed = _run_final_process(
                                        evaluator_launch_command,
                                        task_dir=task_dir,
                                        timeout_seconds=_remaining_final_evaluation_seconds(
                                            evaluation_deadline
                                        ),
                                        run=run,
                                    )
                                finally:
                                    evaluator_cid = broker_dir / "evaluator.cid"
                                    assert evaluator_container_name is not None
                                    evaluator_cleanup_succeeded = False
                                    try:
                                        evaluator_cleanup_succeeded = _cleanup_evaluator_container(
                                            evaluator_cid,
                                            evaluator_container_name,
                                            _remove_named_container,
                                        )
                                    except ContainerMetadataCleanupError:
                                        evaluator_cleanup_succeeded = True
                                        raise
                                    evaluator_started = launch_witness.observed()
                except CandidateBrokerInfrastructureError as exc:
                    if gpu_lease is not None and isinstance(exc, CandidateContainerCleanupError):
                        gpu_lease.quarantine()
                    raise FinalEvaluationInfrastructureError(str(exc)) from exc
                except (
                    OSError,
                    subprocess.TimeoutExpired,
                    BoundedProcessError,
                    CandidateBrokerError,
                ) as exc:
                    raise FinalEvaluationError(
                        f"final evaluator could not complete: {exc}"
                    ) from exc
                finally:
                    if broker is not None and not evaluator_cleanup_succeeded:
                        assert evaluator_container_name is not None
                        try:
                            evaluator_cleanup_succeeded = _retry_evaluator_cleanup(
                                broker_dir / "evaluator.cid",
                                evaluator_container_name,
                                _remove_named_container,
                                _sleep,
                            )
                        except ContainerMetadataCleanupError as exc:
                            evaluator_cleanup_succeeded = True
                            raise FinalEvaluationInfrastructureError(str(exc)) from exc
                        except BaseException:
                            if gpu_lease is not None:
                                gpu_lease.quarantine()
                            raise
                    if not evaluator_cleanup_succeeded and gpu_lease is not None:
                        gpu_lease.quarantine()

                prestart_failure = (
                    broker is not None
                    and completed.returncode in _RETRYABLE_CONTAINER_LAUNCH_CODES
                    and broker.calls == 0
                    and not evaluator_started
                )
        except GpuLeaseQuarantined as exc:
            raise FinalEvaluationInfrastructureError(GPU_ALLOCATION_QUARANTINED_CATEGORY) from exc
        except GpuLeaseError as exc:
            raise FinalEvaluationInfrastructureError(GPU_ALLOCATION_INVALID_CATEGORY) from exc
        if prestart_failure:
            launch_attempts.append(_launch_attempt_diagnostic(attempt, completed))
            if not evaluator_cleanup_succeeded:
                _write_cleanup_failure(cleanup_failure_path)
                _write_launch_failure(
                    launch_failure_path,
                    launch_attempts=launch_attempts,
                )
                raise FinalEvaluationError(
                    "final evaluator cleanup could not be verified; invocation retained; "
                    f"cleanup diagnostics: {cleanup_failure_path}; "
                    + _launch_failure_message(launch_attempts, launch_failure_path)
                )
            rolled_back = _rollback_unstarted_final_attempt(
                invocation_path=invocation_path,
                invocation=invocation,
                results_dir=results_dir,
            )
            if not rolled_back:
                raise FinalEvaluationError(
                    f"final evaluator failed with return code {completed.returncode}"
                )
            final_launch_attempt = launch_attempts[-1]
            category = str(final_launch_attempt["category"])
            should_retry = _retryable_launch_failure(completed.returncode, category)
            if should_retry and attempt < max_attempts:
                retry_delay = float(2 ** (attempt - 1))
                if evaluation_deadline - time.monotonic() <= retry_delay:
                    _write_launch_failure(
                        launch_failure_path,
                        launch_attempts=launch_attempts,
                    )
                    raise FinalEvaluationError(
                        _launch_failure_message(launch_attempts, launch_failure_path)
                    )
                _sleep(retry_delay)
                if evaluation_deadline <= time.monotonic():
                    _write_launch_failure(
                        launch_failure_path,
                        launch_attempts=launch_attempts,
                    )
                    raise FinalEvaluationError(
                        _launch_failure_message(launch_attempts, launch_failure_path)
                    )
                if category == "image_unavailable":
                    assert requested_container_image is not None
                    assert container_image is not None
                    recovery_reference = (
                        container_image_recovery_ref or requested_container_image
                    )
                    try:
                        refreshed_image = _resolve_container_image(recovery_reference)
                    except ValueError:
                        if evaluation_deadline <= time.monotonic():
                            _write_launch_failure(
                                launch_failure_path,
                                launch_attempts=launch_attempts,
                            )
                            raise FinalEvaluationError(
                                _launch_failure_message(
                                    launch_attempts, launch_failure_path
                                )
                            )
                        continue
                    should_retry = refreshed_image == container_image
                if should_retry:
                    if evaluation_deadline <= time.monotonic():
                        _write_launch_failure(
                            launch_failure_path,
                            launch_attempts=launch_attempts,
                        )
                        raise FinalEvaluationError(
                            _launch_failure_message(
                                launch_attempts, launch_failure_path
                            )
                        )
                    continue
            _write_launch_failure(
                launch_failure_path,
                launch_attempts=launch_attempts,
            )
            raise FinalEvaluationError(
                _launch_failure_message(launch_attempts, launch_failure_path)
            )

        with record_compute_failure(
            output_dir, wall_clock_limit_seconds=effective_timeout_seconds,
            started_at=evaluation_started,
            max_rpc_calls=contract.compute.final.max_candidate_rpc_calls, broker=broker,
        ):
            if broker is not None and not evaluator_cleanup_succeeded:
                _write_cleanup_failure(cleanup_failure_path)
                raise FinalEvaluationError(
                    "final evaluator container cleanup failed; invocation retained; "
                    f"diagnostics: {cleanup_failure_path}"
                )
            if completed.returncode != 0:
                raise FinalEvaluationError(
                    f"final evaluator failed with return code {completed.returncode}"
                )
            if broker is not None:
                if broker.calls < 1:
                    raise FinalEvaluationError("final evaluator did not call the frozen candidate")
                if broker.calls_before_first_reset:
                    raise FinalEvaluationError(
                        "final evaluator called the candidate before establishing a declared seed"
                    )
                if not declared_reset_sequence_matches(
                    broker.reset_seeds,
                    expected_reset_seeds,
                ):
                    raise FinalEvaluationError(
                        reset_sequence_diagnostic(
                            broker.reset_seeds,
                            expected_reset_seeds,
                            context="Final evaluator",
                        )
                    )
                if broker.peer_disconnects:
                    raise FinalEvaluationError(
                        "final evaluator disconnected before receiving candidate responses"
                    )
        break
    with record_compute_failure(
        output_dir, wall_clock_limit_seconds=effective_timeout_seconds,
        started_at=evaluation_started,
        max_rpc_calls=contract.compute.final.max_candidate_rpc_calls, broker=broker,
    ):
        if _tree_digest(program_dir) != program_digest:
            raise FinalEvaluationError("frozen incumbent changed during final evaluation")
        try:
            current_task_digest = task_scaffold_fingerprint(task_dir)
        except TaskContractError as exc:
            raise FinalEvaluationError(f"task scaffold became invalid: {exc}") from exc
        if current_task_digest != scaffold_digest:
            raise FinalEvaluationError("task scaffold changed during final evaluation")
        declared_names = tuple(metric.name for metric in contract.metrics.final)
        metrics, correct = _validate_metrics(results_dir, declared_names)
        results_digest = _tree_digest(
            results_dir,
            max_bytes=MAX_FINAL_RESULT_BYTES,
            max_entries=MAX_FINAL_RESULT_ENTRIES,
        )
        _seal_results(results_dir)
    from src.investigation.evaluation_evidence import evidence_metadata
    from src.investigation.evaluation_usage import compute_usage

    retained_evidence = evidence_metadata(results_dir)
    retained_evidence.pop("path", None)
    manifest = {
        **invocation,
        "schema_version": 3,
        "status": "completed",
        "completed_at": _utc_now(),
        "correct": correct,
        "outcome": "qualified" if correct else "not_qualified",
        "metrics": metrics,
        "declared_final_metrics": list(declared_names),
        "results_tree_sha256": results_digest,
        "evaluation_evidence": retained_evidence,
        "compute_usage": compute_usage(
            wall_clock_limit_seconds=effective_timeout_seconds,
            elapsed_seconds=time.monotonic() - evaluation_started,
            max_rpc_calls=contract.compute.final.max_candidate_rpc_calls,
            rpc_calls=broker.calls if broker is not None else None,
            outcome="completed",
        ),
    }
    _write_exclusive_json(manifest_path, manifest)
    from src.investigation.safe_staging import remove_path

    remove_path(output_dir, "evaluation_failure.json")
    return verify_final_manifest(
        output_dir,
        task_dir=task_dir,
        program_dir=program_dir,
    )
