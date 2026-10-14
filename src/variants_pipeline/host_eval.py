"""Sandboxed evaluation for container-authored variant programs."""

from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import os
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.investigation.program_spec import entrypoint_relpath_from_spec
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
from src.investigation.evaluator_tree import managed_evaluator_tree_digest
from src.investigation.safe_staging import remove_path
from src.investigation.reset_sequence import (
    declared_reset_sequence_matches,
    reset_sequence_diagnostic,
)
from src.investigation.task_runtime import RUNTIME_MANIFEST_FILENAME
from src.investigation.task_contract import (
    TASK_CONTRACT_FILENAME,
    TaskContractError,
    load_task_contract,
)
from src.investigation.bounded_process import (
    ProcessOutputLimitError,
    ProcessTimeoutError,
    run_bounded_process,
)
from src.final_evaluation.candidate_broker import (
    CandidateBroker,
    CandidateBrokerError,
    CandidateBrokerInfrastructureError,
    CandidateContainerCleanupError,
    ContainerMetadataCleanupError,
)
from src.final_evaluation.runner import (
    FinalEvaluationError,
    _broker_recovery_root,
    _candidate_command,
    _evaluation_command,
    _force_remove_container,
    _ensure_broker_recovery_root,
    _recover_broker_directory,
    _reject_legacy_broker_directory,
)

from .sandbox_outputs import (
    MAX_SANDBOX_OUTPUT_BYTES,
    PROMOTABLE_RESULT_PATHS,
    canonicalize_results,
    extract_promotable_frame,
    validate_promotable_results,
)

logger = logging.getLogger(__name__)

_EVALUATOR_ARTIFACTS = (
    "_results.json",
    "analysis",
    "metrics.json",
    "correct.json",
    "evaluation_evidence.json",
    "rubric_status.json",
    "evaluation_manifest.json",
    "harness_evaluate_stdout.log",
    "harness_evaluate_stderr.log",
    "runs",
)
_SANDBOX_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NCCL_DEBUG",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)
_HOST_LIBRARY_MOUNTS = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
)


@dataclass(frozen=True)
class HostEvaluationResult:
    returncode: int
    correct: bool
    metrics: dict[str, Any]
    diagnostic: str = ""
    command: tuple[str, ...] = ()


class SandboxEvaluationError(RuntimeError):
    """A sealed evaluator process exited without promotable results."""

    def __init__(self, returncode: int, command: list[str], diagnostic: str) -> None:
        self.returncode = returncode
        self.command = tuple(command)
        self.diagnostic = diagnostic
        super().__init__(f"sandbox evaluation exited with code {returncode}")


class SandboxEvaluationTimeout(TimeoutError):
    """A sealed evaluator exceeded its wall-clock limit."""

    def __init__(self, timeout_seconds: int, command: list[str]) -> None:
        self.timeout_seconds = timeout_seconds
        self.command = tuple(command)
        super().__init__(f"sandbox evaluation timed out after {timeout_seconds}s")


@contextmanager
def _gpu_allocation_scope(
    gpu_lease: GpuAllocationLease | None,
) -> Iterator[None]:
    allocation = gpu_lease if gpu_lease is not None else nullcontext()
    with allocation:
        try:
            yield
        except CandidateContainerCleanupError:
            if gpu_lease is not None:
                gpu_lease.quarantine()
            raise


class _ForwardedSandboxSignal(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum


class SandboxTeardownError(RuntimeError):
    """A sealed evaluator could not be confirmed stopped."""


@dataclass(frozen=True)
class _SandboxLease:
    identity: str
    recovery_dir: Path


def _validate_search_result_schema(results_dir: Path, contract: Any) -> None:
    from src.investigation.evaluator_outputs import validate_evaluator_outputs

    validate_evaluator_outputs(results_dir, tuple(metric.name for metric in contract.metrics.search))


def _brokered_search_commands(
    *,
    task_source_dir: Path,
    evaluator_source_dir: Path,
    program_dir: Path,
    entrypoint: Path,
    results_dir: Path,
    broker_dir: Path,
    sandbox_image: str,
    contract: Any,
    environment: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    """Build isolated evaluator and candidate commands for contracted search."""
    evaluator = evaluator_source_dir / contract.evaluators.search
    evaluator_command = _evaluation_command(
        task_dir=task_source_dir,
        evaluator_source_dir=evaluator_source_dir,
        program_dir=program_dir,
        entrypoint=entrypoint,
        evaluator=evaluator,
        results_dir=results_dir,
        python_bin=task_source_dir / ".venv" / "bin" / "python",
        container_image=sandbox_image,
        gpu_count=contract.compute.search.evaluator_gpu_count,
        cpu_cores=contract.compute.search.cpu_cores,
        memory_gb=contract.compute.search.memory_gb,
        environment=environment,
        broker_dir=broker_dir,
        candidate_editable_paths=contract.candidate.editable_paths,
        public_artifact_bundle=contract.resources.public_artifact_bundle,
        rpc_timeout_seconds=contract.compute.search.wall_clock_seconds,
        plane="search",
    )
    candidate_command = _candidate_command(
        task_dir=task_source_dir,
        program_dir=program_dir,
        entrypoint=entrypoint,
        container_image=sandbox_image,
        contract=contract,
        environment=environment,
        broker_dir=broker_dir,
        plane="search",
    )
    return evaluator_command, candidate_command


def _evaluate_brokered_search_locked(
    *,
    task_source_dir: Path,
    evaluator_source_dir: Path,
    program_dir: Path,
    entrypoint: Path,
    python_bin: Path,
    sandbox_image: str,
    timeout_seconds: int,
    environment: Mapping[str, str],
    sandbox_gpus: str | None,
    contract: Any,
    lease: _SandboxLease | None = None,
) -> HostEvaluationResult:
    """Evaluate a contracted search candidate across a host-owned broker."""
    if contract.candidate.interface != "python_rpc_v1:handle":
        raise ValueError("contracted search requires candidate python_rpc_v1:handle")
    if not _uses_image_managed_runtime(
        task_source_dir=task_source_dir,
        evaluator_source_dir=evaluator_source_dir,
        python_bin=python_bin,
        sandbox_image=sandbox_image,
    ):
        raise ValueError("contracted search requires an image-managed task runtime")
    try:
        broker_root = _ensure_broker_recovery_root()
    except FinalEvaluationError as exc:
        raise ValueError("contracted search requires a private local broker directory") from exc
    effective_timeout = min(timeout_seconds, contract.compute.search.wall_clock_seconds)
    evaluation_started = time.monotonic()
    evaluation_deadline = evaluation_started + effective_timeout
    broker_environment = dict(environment)
    if contract.compute.search.gpu_count:
        try:
            broker_environment["AGENT_CONTAINER_GPUS"] = _bounded_gpu_selector(
                sandbox_gpus, contract.compute.search.gpu_count
            )
        except ValueError as exc:
            raise CandidateBrokerInfrastructureError(
                GPU_ALLOCATION_INVALID_CATEGORY
            ) from exc
    broker_dir = (
        _search_broker_directory(lease)
        if lease is not None
        else broker_root / f"agentic-search-{os.getuid()}-{secrets.token_hex(8)}"
    )
    _recover_broker_directory(broker_dir)
    manifest_path = program_dir / "evaluation_manifest.json"
    remove_path(program_dir, "evaluation_failure.json")
    with tempfile.TemporaryDirectory(prefix="agentic-search-results-") as temporary:
        results_dir = Path(temporary)
        evaluator_command, candidate_command = _brokered_search_commands(
            task_source_dir=task_source_dir,
            evaluator_source_dir=evaluator_source_dir,
            program_dir=program_dir,
            entrypoint=entrypoint,
            results_dir=results_dir,
            broker_dir=broker_dir,
            sandbox_image=sandbox_image,
            contract=contract,
            environment=broker_environment,
        )
        manifest: dict[str, Any] = {
            "producer": "sandbox_host_eval",
            "command": evaluator_command,
            "execution_boundary": "brokered_search",
            "program_id": program_dir.name,
            "started_at": _utc_now(),
        }
        _write_json(manifest_path, manifest)
        try:
            gpu_count = contract.compute.search.gpu_count
            gpu_lease: GpuAllocationLease | None = (
                acquire_gpu_allocation_lease(
                    sandbox_gpus,
                    gpu_count,
                    lease_root=GPU_ALLOCATION_LEASE_ROOT,
                    deadline=evaluation_deadline,
                )
                if gpu_count
                else None
            )
            with _gpu_allocation_scope(gpu_lease):
                remaining_timeout = evaluation_deadline - time.monotonic()
                if remaining_timeout <= 0:
                    if gpu_lease is not None:
                        raise CandidateBrokerInfrastructureError(
                            GPU_ALLOCATION_BUSY_CATEGORY
                        )
                    raise SandboxEvaluationTimeout(effective_timeout, evaluator_command)
                from src.investigation.evaluation_usage import record_compute_failure

                evaluator_launch = evaluator_command
                candidate_launch = candidate_command
                if gpu_lease is not None:
                    gpu_lease.prepare_runtime()
                    evaluator_launch = gpu_lease.wrap_command(evaluator_command)
                    candidate_launch = gpu_lease.wrap_command(candidate_command)
                broker = CandidateBroker(
                    candidate_launch,
                    broker_dir,
                    rpc_timeout_seconds=remaining_timeout,
                    reset_timeout_seconds=remaining_timeout,
                    max_rpc_calls=contract.compute.search.max_candidate_rpc_calls,
                    expected_reset_seeds=contract.protocol.search_reset_sequence,
                )
                with record_compute_failure(
                    program_dir, wall_clock_limit_seconds=effective_timeout,
                    started_at=evaluation_started,
                    max_rpc_calls=contract.compute.search.max_candidate_rpc_calls,
                    broker=broker,
                ), broker:
                    try:
                        process_timeout = evaluation_deadline - time.monotonic()
                        if process_timeout <= 0:
                            raise SandboxEvaluationTimeout(
                                effective_timeout, evaluator_command
                            )
                        try:
                            completed = run_bounded_process(
                                evaluator_launch,
                                cwd=task_source_dir,
                                env=os.environ,
                                timeout_seconds=process_timeout,
                            )
                        except ProcessTimeoutError:
                            raise
                        except OSError as exc:
                            raise CandidateBrokerInfrastructureError(
                                "search evaluator container runtime failed to launch"
                            ) from exc
                    finally:
                        try:
                            _force_remove_container(broker_dir / "evaluator.cid")
                        except ContainerMetadataCleanupError:
                            raise
                        except FinalEvaluationError as exc:
                            if gpu_lease is not None:
                                gpu_lease.quarantine()
                            raise CandidateBrokerInfrastructureError(
                                "search evaluator container cleanup failed; "
                                "recovery metadata retained"
                            ) from exc
                        except BaseException:
                            if gpu_lease is not None:
                                gpu_lease.quarantine()
                            raise
            with record_compute_failure(
                program_dir, wall_clock_limit_seconds=effective_timeout,
                started_at=evaluation_started,
                max_rpc_calls=contract.compute.search.max_candidate_rpc_calls,
                broker=broker,
            ):
                if completed.returncode in {125, 126, 127}:
                    raise CandidateBrokerInfrastructureError(
                        "search evaluator container runtime failed"
                    )
                if completed.returncode != 0:
                    diagnostic = _decode_sandbox_diagnostic(completed.stderr or completed.stdout)
                    raise SandboxEvaluationError(
                        completed.returncode,
                        evaluator_command,
                        diagnostic,
                    )
                if broker.calls < 1:
                    raise CandidateBrokerError("search evaluator did not call the frozen candidate")
                if broker.calls_before_first_reset:
                    raise CandidateBrokerError(
                        "search evaluator called the candidate before establishing a declared seed"
                    )
                if not declared_reset_sequence_matches(
                    broker.reset_seeds,
                    contract.protocol.search_reset_sequence,
                ):
                    raise CandidateBrokerError(
                        reset_sequence_diagnostic(
                            broker.reset_seeds,
                            contract.protocol.search_reset_sequence,
                            context="Search evaluator",
                        )
                    )
                if broker.peer_disconnects:
                    raise CandidateBrokerError(
                        "search evaluator disconnected before receiving candidate responses"
                    )
        except ProcessTimeoutError as exc:
            discard_evaluation_artifacts(program_dir)
            raise SandboxEvaluationTimeout(effective_timeout, evaluator_command) from exc
        except ProcessOutputLimitError as exc:
            discard_evaluation_artifacts(program_dir)
            raise SandboxEvaluationError(
                1, evaluator_command, "search evaluator exceeded the output limit"
            ) from exc
        except GpuLeaseTimeout as exc:
            discard_evaluation_artifacts(program_dir)
            raise CandidateBrokerInfrastructureError(GPU_ALLOCATION_BUSY_CATEGORY) from exc
        except GpuLeaseQuarantined as exc:
            discard_evaluation_artifacts(program_dir)
            raise CandidateBrokerInfrastructureError(
                GPU_ALLOCATION_QUARANTINED_CATEGORY
            ) from exc
        except GpuLeaseError as exc:
            discard_evaluation_artifacts(program_dir)
            raise CandidateBrokerInfrastructureError(GPU_ALLOCATION_INVALID_CATEGORY) from exc
        except CandidateBrokerInfrastructureError:
            discard_evaluation_artifacts(program_dir)
            raise
        except CandidateBrokerError as exc:
            discard_evaluation_artifacts(program_dir)
            raise SandboxEvaluationError(1, evaluator_command, str(exc)) from exc
        except SandboxEvaluationError:
            discard_evaluation_artifacts(program_dir)
            raise
        except BaseException:
            discard_evaluation_artifacts(program_dir)
            raise
        try:
            with record_compute_failure(
                program_dir, wall_clock_limit_seconds=effective_timeout,
                started_at=evaluation_started,
                max_rpc_calls=contract.compute.search.max_candidate_rpc_calls,
                broker=broker,
            ):
                _validate_search_result_schema(results_dir, contract)
                canonicalize_results(results_dir, program_dir.name, broker_environment)
                _validate_staged_results(results_dir, program_dir, broker_environment)
                _promote_results(results_dir, program_dir)
                metrics = _load_mapping(program_dir / "metrics.json")
                correct_payload = _load_mapping(program_dir / "correct.json")
                if not metrics or not correct_payload:
                    raise RuntimeError("brokered search produced invalid result artifacts")
                manifest.update(finished_at=_utc_now(), returncode=0)
                from src.investigation.evaluation_evidence import evidence_metadata
                from src.investigation.evaluation_usage import compute_usage

                manifest["evaluation_evidence"] = evidence_metadata(program_dir)
                manifest["compute_usage"] = compute_usage(
                    wall_clock_limit_seconds=effective_timeout,
                    elapsed_seconds=time.monotonic() - evaluation_started,
                    max_rpc_calls=contract.compute.search.max_candidate_rpc_calls,
                    rpc_calls=broker.calls,
                    outcome="completed",
                )
                _write_json(manifest_path, manifest)
        except BaseException:
            discard_evaluation_artifacts(program_dir)
            raise
    return HostEvaluationResult(
        returncode=0,
        correct=correct_payload.get("correct") is True,
        metrics=metrics,
        command=tuple(evaluator_command),
    )


def evaluate_variant_program(
    *,
    task_bundle_dir: Path,
    evaluator_source_dir: Path,
    task_source_dir: Path,
    program_index: int,
    python_bin: Path,
    sandbox_image: str,
    timeout_seconds: int,
    environment: Mapping[str, str] | None = None,
    sandbox_gpus: str | None = None,
) -> HostEvaluationResult:
    """Evaluate one candidate in a sealed, networkless Docker sandbox."""
    if os.getuid() == 0:
        raise ValueError("sealed variant evaluation requires a non-root host user")
    task_bundle_dir = task_bundle_dir.resolve()
    program_dir = task_bundle_dir / f"program_{program_index}"
    with _sandbox_lease(program_dir) as lease:
        if _recover_stale_sandbox(lease):
            discard_evaluation_artifacts(program_dir)
        return _evaluate_variant_program_locked(
            task_bundle_dir=task_bundle_dir,
            evaluator_source_dir=evaluator_source_dir,
            task_source_dir=task_source_dir,
            program_index=program_index,
            python_bin=python_bin,
            sandbox_image=sandbox_image,
            timeout_seconds=timeout_seconds,
            environment=environment,
            sandbox_gpus=sandbox_gpus,
            lease=lease,
        )


def _evaluate_variant_program_locked(
    *,
    task_bundle_dir: Path,
    evaluator_source_dir: Path,
    task_source_dir: Path,
    program_index: int,
    python_bin: Path,
    sandbox_image: str,
    timeout_seconds: int,
    lease: _SandboxLease,
    environment: Mapping[str, str] | None = None,
    sandbox_gpus: str | None = None,
) -> HostEvaluationResult:
    task_bundle_dir = task_bundle_dir.resolve()
    evaluator_source_dir = evaluator_source_dir.resolve()
    task_source_dir = task_source_dir.resolve()
    evaluate_py = evaluator_source_dir / "evaluate.py"
    if not evaluate_py.is_file():
        raise ValueError(f"missing sealed evaluator: {evaluate_py}")
    if not python_bin.is_file():
        raise ValueError(f"missing task Python: {python_bin}")
    if not sandbox_image.strip():
        raise ValueError("sandbox image is required")
    if os.getuid() == 0:
        raise ValueError("sealed variant evaluation requires a non-root host user")
    search_contract = None
    contract_path = task_source_dir / TASK_CONTRACT_FILENAME
    if contract_path.is_file():
        try:
            search_contract = load_task_contract(contract_path)
        except TaskContractError as exc:
            raise ValueError(f"invalid task contract for search evaluation: {exc}") from exc
        timeout_seconds = min(timeout_seconds, search_contract.compute.search.wall_clock_seconds)

    program_dir, entrypoint_rel = _validate_candidate(task_bundle_dir, program_index)
    if search_contract is not None:
        return _evaluate_brokered_search_locked(
            task_source_dir=task_source_dir,
            evaluator_source_dir=evaluator_source_dir,
            program_dir=program_dir,
            entrypoint=program_dir / entrypoint_rel,
            python_bin=python_bin,
            sandbox_image=sandbox_image,
            timeout_seconds=timeout_seconds,
            environment=environment or {},
            sandbox_gpus=sandbox_gpus,
            contract=search_contract,
            lease=lease,
        )
    results_dir = program_dir / ".harness_results"
    cid_path: Path | None = None
    container_name: str | None = None
    create_attempted = False
    create_completed = False
    try:
        results_dir.mkdir(mode=0o700)
        command, start_command, cid_path, container_name = _sandbox_command(
            task_source_dir=task_source_dir,
            evaluator_source_dir=evaluator_source_dir,
            program_dir=program_dir,
            entrypoint_rel=entrypoint_rel,
            python_bin=python_bin,
            sandbox_image=sandbox_image,
            environment=environment or {},
            sandbox_gpus=sandbox_gpus,
            lease=lease,
        )
        manifest_path = program_dir / "evaluation_manifest.json"
        manifest: dict[str, Any] = {
            "producer": "sandbox_host_eval",
            "command": command,
            "execution_boundary": "sandbox",
            "program_id": program_dir.name,
            "started_at": _utc_now(),
        }
        _write_json(manifest_path, manifest)
        create_attempted = True
        _create_sandbox(command)
        create_completed = True
    except BaseException as exc:
        cleanup_confirmed = container_name is None
        try:
            if container_name is not None:
                cleanup_confirmed = _stop_sandbox(None, container_name)
        finally:
            if create_attempted and not create_completed:
                cleanup_confirmed = False
            try:
                discard_evaluation_artifacts(program_dir)
            finally:
                if cid_path is not None and cleanup_confirmed:
                    _discard_cid_path(cid_path)
        if not cleanup_confirmed:
            assert cid_path is not None
            raise SandboxTeardownError(
                f"could not stop failed sandbox {container_name}; "
                f"recovery metadata retained at {cid_path.parent}"
            ) from exc
        raise
    assert cid_path is not None
    assert container_name is not None
    previous_handlers: dict[int, Any] = {}
    forwarded_signum: int | None = None
    process: subprocess.Popen[bytes] | None = None
    cleanup_confirmed = True

    def interrupt_sandbox(signum: int, _frame: object) -> None:
        nonlocal forwarded_signum
        if forwarded_signum is None:
            forwarded_signum = signum
        if process is not None:
            raise _ForwardedSandboxSignal(signum)

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, interrupt_sandbox)
        process = subprocess.Popen(
            start_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        if forwarded_signum is not None:
            raise _ForwardedSandboxSignal(forwarded_signum)
        archive_payload, diagnostic_payload = process.communicate(timeout=timeout_seconds)
        returncode = process.returncode
        if returncode != 0:
            cleanup_confirmed = _stop_sandbox(process, container_name)
    except subprocess.TimeoutExpired as exc:
        cleanup_confirmed = False
        try:
            cleanup_confirmed = _stop_sandbox(process, container_name)
        finally:
            discard_evaluation_artifacts(program_dir)
        if not cleanup_confirmed:
            raise SandboxTeardownError(
                f"could not stop timed-out sandbox {container_name}; "
                f"recovery metadata retained at {cid_path.parent}"
            ) from exc
        raise SandboxEvaluationTimeout(timeout_seconds, command) from exc
    except _ForwardedSandboxSignal as exc:
        cleanup_confirmed = False
        try:
            cleanup_confirmed = _stop_sandbox(process, container_name)
        finally:
            discard_evaluation_artifacts(program_dir)
        if not cleanup_confirmed:
            raise SandboxTeardownError(
                f"could not stop interrupted sandbox {container_name}; "
                f"recovery metadata retained at {cid_path.parent}"
            ) from exc
        raise SystemExit(128 + exc.signum) from None
    except BaseException as exc:
        cleanup_confirmed = False
        try:
            cleanup_confirmed = _stop_sandbox(process, container_name)
        finally:
            discard_evaluation_artifacts(program_dir)
        if not cleanup_confirmed:
            raise SandboxTeardownError(
                f"could not stop failed sandbox {container_name}; "
                f"recovery metadata retained at {cid_path.parent}"
            ) from exc
        raise
    finally:
        for restore_signum, previous_handler in previous_handlers.items():
            signal.signal(restore_signum, previous_handler)
        if cleanup_confirmed:
            _discard_cid_path(cid_path)

    diagnostic = _decode_sandbox_diagnostic(diagnostic_payload)
    if returncode != 0:
        if not cleanup_confirmed:
            discard_evaluation_artifacts(program_dir)
            raise SandboxTeardownError(
                f"could not stop failed sandbox {container_name}; "
                f"recovery metadata retained at {cid_path.parent}"
            )
        try:
            manifest.update(finished_at=_utc_now(), returncode=returncode)
            _write_json(manifest_path, manifest)
        finally:
            discard_evaluation_artifacts(program_dir)
        raise SandboxEvaluationError(returncode, command, diagnostic)
    try:
        extract_promotable_frame(archive_payload, results_dir)
        _validate_staged_results(results_dir, program_dir, environment or {})
        _promote_results(results_dir, program_dir)
        metrics = _load_mapping(program_dir / "metrics.json")
        correct_payload = _load_mapping(program_dir / "correct.json")
        if not metrics or not correct_payload:
            missing = [
                name
                for name in ("metrics.json", "correct.json")
                if not (program_dir / name).is_file()
            ]
            raise RuntimeError(
                f"sandbox evaluation of {program_dir.name} produced invalid artifacts"
                + (f": {', '.join(missing)}" if missing else "")
            )
        manifest.update(finished_at=_utc_now(), returncode=returncode)
        _write_json(manifest_path, manifest)
    except BaseException:
        discard_evaluation_artifacts(program_dir)
        raise
    return HostEvaluationResult(
        returncode=returncode,
        correct=correct_payload.get("correct") is True,
        metrics=metrics,
        diagnostic=diagnostic,
        command=tuple(command),
    )


def _decode_sandbox_diagnostic(payload: bytes | None) -> str:
    if not payload:
        return ""
    return payload[: 1024 * 1024].decode("ascii", errors="backslashreplace")


def _create_sandbox(command: list[str]) -> None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("could not create sealed evaluation sandbox") from exc
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout).strip()
        raise SandboxEvaluationError(result.returncode, command, diagnostic)


def resolve_host_python(*, task_source_dir: Path | None, venv_dir: Path | None) -> Path:
    """Find the venv launcher without resolving its interpreter symlink."""
    candidates: list[Path] = []
    if venv_dir is not None:
        candidates.extend((venv_dir / "bin" / "python", venv_dir / ".venv" / "bin" / "python"))
    if task_source_dir is not None:
        candidates.append(task_source_dir / ".venv" / "bin" / "python")
    for candidate in candidates:
        absolute_candidate = candidate.expanduser().absolute()
        if absolute_candidate.is_file():
            return absolute_candidate
    searched = ", ".join(str(path) for path in candidates) or "(no task path provided)"
    raise ValueError(f"task Python not found; searched: {searched}")


def _sandbox_command(
    *,
    task_source_dir: Path,
    evaluator_source_dir: Path,
    program_dir: Path,
    entrypoint_rel: str,
    python_bin: Path,
    sandbox_image: str,
    environment: Mapping[str, str],
    sandbox_gpus: str | None,
    lease: _SandboxLease,
) -> tuple[list[str], list[str], Path, str]:
    venv_dir = python_bin.parent.parent.absolute()
    image_managed_runtime = _uses_image_managed_runtime(
        task_source_dir=task_source_dir,
        evaluator_source_dir=evaluator_source_dir,
        python_bin=python_bin,
        sandbox_image=sandbox_image,
    )
    sandbox_env = {key: environment[key] for key in _SANDBOX_ENV_KEYS if key in environment}
    memory_limit = environment.get("VARIANT_EVAL_MEMORY_LIMIT", "64g")
    cpu_cores: int | None = None
    contract_path = task_source_dir / TASK_CONTRACT_FILENAME
    if contract_path.is_file():
        try:
            contract = load_task_contract(contract_path)
        except TaskContractError as exc:
            raise ValueError(f"invalid task contract for search sandbox: {exc}") from exc
        memory_limit = f"{contract.compute.search.memory_gb}g"
        cpu_cores = contract.compute.search.cpu_cores
        sandbox_env["AGENTIC_NUMERIC_PRECISION"] = contract.compute.search.precision
        if contract.compute.search.gpu_count == 0:
            sandbox_gpus = None
        else:
            sandbox_gpus = _bounded_gpu_selector(sandbox_gpus, contract.compute.search.gpu_count)
    if re.fullmatch(r"[1-9][0-9]*[mg]", memory_limit.lower()) is None:
        raise ValueError("VARIANT_EVAL_MEMORY_LIMIT must be a positive Docker memory size")
    sandbox_env.update(
        {
            "BASE_PROGRAM": environment.get("BASE_PROGRAM", ""),
            "AGENT_NAME": environment.get("AGENT_NAME", ""),
            "EVALUATION_ID": environment.get("EVALUATION_ID", ""),
            "HOME": "/tmp/home",
            "NUM_REPLICATIONS": environment.get("NUM_REPLICATIONS", "1"),
            "PATH": "/task-venv/bin:/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "TASK_SOURCE_DIR": "/task-source",
            "VARIANT_ID": environment.get("VARIANT_ID", ""),
            "VARIANT_KIND": environment.get("VARIANT_KIND", ""),
            "VENV_DIR": "/task-venv",
        }
    )
    command = [
        "docker",
        "create",
        "--rm",
        "--read-only",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--cap-add",
        "SETUID",
        "--cap-add",
        "SETGID",
        "--cap-add",
        "KILL",
        "--cap-add",
        "DAC_OVERRIDE",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "4096",
        "--memory",
        memory_limit,
        "--memory-swap",
        memory_limit,
        "--user",
        "0:0",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=16g",
        "--shm-size",
        "16g",
    ]
    if cpu_cores is not None:
        command.extend(("--cpus", str(cpu_cores)))
    if sandbox_gpus:
        command.extend(("--gpus", sandbox_gpus))
    for key, value in sorted(sandbox_env.items()):
        command.extend(("--env", f"{key}={value}"))
    command.extend(
        (
            "--volume",
            f"{task_source_dir}:/task-source:ro",
            "--volume",
            f"{evaluator_source_dir}:/evaluator-source:ro",
            "--volume",
            f"{venv_dir}:/task-venv:ro",
            "--volume",
            f"{program_dir}:/candidate:ro",
            "--volume",
            f"{Path(__file__).resolve().parents[1]}:/harness-source/src:ro",
            "--tmpfs",
            (
                "/results:rw,noexec,nosuid,nodev,size=144m,nr_inodes=20000,"
                f"uid={os.getuid()},gid={os.getgid()},mode=0777"
            ),
        )
    )
    for source_dir, mount_root in (
        (task_source_dir, "/task-source"),
        (evaluator_source_dir, "/evaluator-source"),
    ):
        for ignored_name in (".git", ".venv", "__pycache__", "results"):
            if (source_dir / ignored_name).is_dir():
                command.extend(
                    (
                        "--tmpfs",
                        f"{mount_root}/{ignored_name}:ro,noexec,nosuid,nodev,size=64k,nr_inodes=16",
                    )
                )
    _append_search_seals(
        command,
        task_source_dir=task_source_dir,
        evaluator_source_dir=evaluator_source_dir,
    )
    if not image_managed_runtime:
        for mount in _python_runtime_mounts(python_bin):
            command.extend(("--volume", f"{mount}:{mount}:ro"))
    command.extend(
        (
            sandbox_image,
            "/usr/local/bin/python3",
            "-I",
            "-S",
            "/harness-source/src/variants_pipeline/sandbox_runner.py",
            "--evaluator",
            "/evaluator-source/evaluate.py",
            "--evaluator-python",
            "/task-venv/bin/python",
            "--program",
            str(Path("/candidate") / entrypoint_rel),
            "--program-name",
            program_dir.name,
            "--results",
            "/results",
            "--evaluator-uid",
            str(os.getuid()),
            "--evaluator-gid",
            str(os.getgid()),
        )
    )
    lease.recovery_dir.mkdir(mode=0o700)
    _fsync_directory(lease.recovery_dir.parent)
    cid_path = lease.recovery_dir / "container.cid"
    container_name = f"agentic-eval-{os.getuid()}-{lease.identity}-{secrets.token_hex(6)}"
    name_path = lease.recovery_dir / "container.name"
    temporary_name_path = lease.recovery_dir / "container.name.tmp"
    descriptor = os.open(
        temporary_name_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, container_name.encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary_name_path, name_path)
    directory_descriptor = os.open(lease.recovery_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    command[3:3] = ["--cidfile", str(cid_path), "--name", container_name]
    start_command = ["docker", "start", "--attach", container_name]
    return command, start_command, cid_path, container_name


def _append_search_seals(
    command: list[str], *, task_source_dir: Path, evaluator_source_dir: Path
) -> None:
    """Overlay sealed final artifacts in every search-evaluation task mount."""
    contract_path = task_source_dir / TASK_CONTRACT_FILENAME
    if not contract_path.is_file():
        return
    try:
        contract = load_task_contract(contract_path)
    except TaskContractError as exc:
        raise ValueError(f"invalid task contract for search sandbox: {exc}") from exc
    for source_dir, mount_root in (
        (task_source_dir, "/task-source"),
        (evaluator_source_dir, "/evaluator-source"),
    ):
        if (source_dir / contract.resources.final_root).is_dir():
            command.extend(
                (
                    "--tmpfs",
                    f"{mount_root}/{contract.resources.final_root}:ro,noexec,nosuid,nodev,size=64k,nr_inodes=16",
                )
            )
        if (source_dir / contract.evaluators.final).is_file():
            command.extend(
                (
                    "--mount",
                    (
                        "type=bind,src=/dev/null,"
                        f"dst={mount_root}/{contract.evaluators.final},readonly"
                    ),
                )
            )


def _bounded_gpu_selector(raw_selector: str | None, gpu_count: int) -> str:
    try:
        return bounded_docker_gpu_selector(raw_selector, gpu_count)
    except GpuSelectorError as exc:
        raise ValueError(f"invalid contracted GPU search selector: {exc}") from exc


def _uses_image_managed_runtime(
    *,
    task_source_dir: Path,
    evaluator_source_dir: Path,
    python_bin: Path,
    sandbox_image: str,
) -> bool:
    manifest_path = task_source_dir / RUNTIME_MANIFEST_FILENAME
    expected_python = task_source_dir / ".venv" / "bin" / "python"
    uses_task_runtime = python_bin.absolute() == expected_python.absolute()
    if manifest_path.is_symlink():
        raise ValueError(f"unsafe task runtime manifest: {manifest_path}")
    if not manifest_path.exists():
        if uses_task_runtime:
            raise ValueError("managed task runtime manifest is missing")
        return False
    if not manifest_path.is_file():
        raise ValueError(f"unsafe task runtime manifest: {manifest_path}")
    if not uses_task_runtime:
        raise ValueError("managed task runtime Python must be task_dir/.venv/bin/python")
    if expected_python.is_symlink() or not expected_python.is_file():
        raise ValueError(f"unsafe managed task Python: {expected_python}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid task runtime manifest: {manifest_path}") from exc
    requirements_path = task_source_dir / "requirements.txt"
    if requirements_path.is_symlink() or (
        requirements_path.exists() and not requirements_path.is_file()
    ):
        raise ValueError(f"unsafe task requirements: {requirements_path}")
    requirements_bytes = requirements_path.read_bytes() if requirements_path.is_file() else b""
    requirements_sha256 = hashlib.sha256(requirements_bytes).hexdigest()
    evaluator_path = evaluator_source_dir / "evaluate.py"
    if evaluator_path.is_symlink() or not evaluator_path.is_file():
        raise ValueError(f"unsafe sealed evaluator: {evaluator_path}")
    evaluator_sha256 = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    evaluator_tree_sha256 = managed_evaluator_tree_digest(evaluator_source_dir)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported task runtime manifest")
    if manifest.get("image_id") != sandbox_image:
        raise ValueError("task runtime image does not match the sealed evaluation image")
    if manifest.get("requirements_sha256") != requirements_sha256:
        raise ValueError("task runtime requirements have changed since bootstrap")
    if manifest.get("evaluator_sha256") != evaluator_sha256:
        raise ValueError("task runtime evaluator has changed since bootstrap")
    if manifest.get("evaluator_tree_sha256") != evaluator_tree_sha256:
        raise ValueError("task runtime evaluator tree has changed since bootstrap")
    return True


def _python_runtime_mounts(python_bin: Path) -> tuple[Path, ...]:
    resolved_python = python_bin.resolve(strict=True)
    runtime_prefixes = [resolved_python.parent.parent]
    versions = set(re.findall(r"python(3\.\d+)", resolved_python.name))
    pyvenv_config = python_bin.parent.parent / "pyvenv.cfg"
    if pyvenv_config.is_file() and not pyvenv_config.is_symlink():
        for line in pyvenv_config.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if not separator:
                continue
            if key.strip() == "version":
                match = re.fullmatch(r"(3\.\d+)(?:\.\d+)?", value.strip())
                if match:
                    versions.add(match.group(1))
            elif key.strip() == "home" and value.strip():
                configured_home = Path(value.strip()).expanduser()
                if configured_home.is_absolute() and configured_home.is_dir():
                    runtime_prefixes.append(configured_home.resolve().parent)

    mounts = [*_external_python_symlink_targets(python_bin), resolved_python]
    for prefix in dict.fromkeys(runtime_prefixes):
        for version in versions:
            for library_root in (prefix / "lib", prefix / "lib64"):
                stdlib = library_root / f"python{version}"
                if stdlib.is_dir() and not stdlib.is_symlink():
                    mounts.append(stdlib)
                mounts.extend(
                    path.resolve()
                    for path in library_root.glob(f"libpython{version}.so*")
                    if path.is_file()
                )
    for library_dir in map(Path, _HOST_LIBRARY_MOUNTS):
        if library_dir.is_dir():
            mounts.append(library_dir)
    return tuple(dict.fromkeys(mounts))


def _external_python_symlink_targets(python_bin: Path) -> tuple[Path, ...]:
    venv_root = python_bin.parent.parent.absolute()
    current = python_bin.absolute()
    visited: set[Path] = set()
    targets: list[Path] = []
    while current.is_symlink():
        if current in visited:
            raise ValueError(f"task Python contains a symlink cycle: {python_bin}")
        visited.add(current)
        link_target = Path(os.readlink(current))
        if not link_target.is_absolute():
            link_target = current.parent / link_target
        current = Path(os.path.abspath(link_target))
        if not current.is_relative_to(venv_root) and current.is_file():
            targets.append(current)
    return tuple(targets)


def _promote_results(results_dir: Path, program_dir: Path) -> None:
    for source in sorted(results_dir.iterdir(), key=lambda path: path.name):
        destination = program_dir / source.name
        shutil.move(str(source), str(destination))
    results_dir.rmdir()


def _validate_staged_results(
    results_dir: Path,
    program_dir: Path,
    environment: Mapping[str, str],
) -> None:
    paths = list(results_dir.rglob("*"))
    unsafe = next(
        (path for path in paths if path.is_symlink() or not (path.is_file() or path.is_dir())),
        None,
    )
    if unsafe is not None:
        raise RuntimeError(f"sandbox produced unsafe output: {unsafe.relative_to(results_dir)}")
    unexpected = []
    for path in paths:
        relative = path.relative_to(results_dir).as_posix()
        expected_type = path.is_dir() if relative == "analysis" else path.is_file()
        if relative not in PROMOTABLE_RESULT_PATHS or not expected_type:
            unexpected.append(relative)
    unexpected.sort()
    if unexpected:
        raise RuntimeError(f"sandbox produced unexpected output: {', '.join(unexpected)}")
    output_bytes = sum(path.stat().st_size for path in paths if path.is_file())
    if output_bytes > MAX_SANDBOX_OUTPUT_BYTES:
        raise RuntimeError("sandbox output exceeds the 64 MiB promotion limit")
    collision = next(
        (
            source.name
            for source in results_dir.iterdir()
            if (program_dir / source.name).exists() or (program_dir / source.name).is_symlink()
        ),
        None,
    )
    if collision is not None:
        raise RuntimeError(f"sandbox attempted to replace candidate input: {collision}")
    validate_promotable_results(results_dir, environment, program_dir.name)


def discard_evaluation_artifacts(program_dir: Path) -> None:
    """Descriptor-safely remove all host-owned evaluation outputs."""
    for artifact in (".harness_results", *_EVALUATOR_ARTIFACTS):
        remove_path(program_dir, artifact)


def _discard_cid_path(cid_path: Path) -> None:
    for metadata_path in (cid_path, cid_path.parent / "container.name"):
        try:
            metadata_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove sandbox metadata %s: %s", metadata_path, exc)
    try:
        cid_path.parent.rmdir()
    except OSError as exc:
        logger.warning("Could not remove sandbox CID directory %s: %s", cid_path.parent, exc)
    else:
        _fsync_directory(cid_path.parent.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_container_name(name_path: Path, identity: str) -> str:
    descriptor = os.open(
        name_path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o022
        ):
            raise OSError(f"unsafe sandbox recovery metadata: {name_path}")
        raw_name = os.read(descriptor, 129)
    finally:
        os.close(descriptor)
    if len(raw_name) > 128:
        raise OSError(f"oversized sandbox recovery metadata: {name_path}")
    try:
        container_name = raw_name.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise OSError(f"non-ASCII sandbox recovery metadata: {name_path}") from exc
    expected_prefix = f"agentic-eval-{os.getuid()}-{identity}-"
    if (
        not container_name.startswith(expected_prefix)
        or re.fullmatch(rf"{re.escape(expected_prefix)}[0-9a-f]{{12}}", container_name) is None
    ):
        raise OSError(f"invalid sandbox recovery metadata: {name_path}")
    return container_name


def _force_remove_sandbox(container_name: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "rm", "--force", container_name],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.error("Could not force-remove sandbox %s: %s", container_name, exc)
        return False
    if result.returncode == 0 or "no such container" in result.stderr.lower():
        return True
    logger.error("Could not force-remove sandbox %s: %s", container_name, result.stderr.strip())
    return False


def _private_sandbox_root() -> Path:
    candidates: list[Path] = []
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir and Path(runtime_dir).is_absolute():
        candidates.append(Path(runtime_dir))
    candidates.extend((Path(f"/run/user/{os.getuid()}"), Path.home()))
    for parent in candidates:
        try:
            parent_metadata = parent.lstat()
        except OSError:
            continue
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.getuid()
            or parent_metadata.st_mode & 0o022
        ):
            continue
        root = parent / ".agentic-scientist-sandbox-eval"
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        metadata = root.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
        ):
            raise OSError(f"unsafe sandbox recovery root: {root}")
        return root
    raise OSError("no owner-private directory is available for sandbox recovery")


@contextmanager
def _sandbox_lease(program_dir: Path) -> Iterator[_SandboxLease]:
    identity = _sandbox_identity(program_dir)
    root = _private_sandbox_root()
    lock_path = root / f"{identity}.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
        ):
            raise OSError(f"unsafe sandbox evaluation lock: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield _SandboxLease(
            identity=identity,
            recovery_dir=root / f"{identity}.recovery",
        )
    finally:
        os.close(descriptor)


def _sandbox_identity(program_dir: Path) -> str:
    payload = socket.gethostname().encode("utf-8") + b"\0" + os.fsencode(program_dir.absolute())
    return hashlib.sha256(payload).hexdigest()[:24]


def _search_broker_directory(lease: _SandboxLease) -> Path:
    return _broker_recovery_root() / f"search-{lease.identity}"


def _recover_stale_sandbox(lease: _SandboxLease) -> bool:
    recovery_dir = lease.recovery_dir
    broker_dir = _search_broker_directory(lease)
    try:
        _reject_legacy_broker_directory(broker_dir)
    except FinalEvaluationError as exc:
        raise SandboxTeardownError(str(exc)) from exc
    broker_recovered = broker_dir.exists() or broker_dir.is_symlink()
    if broker_recovered:
        try:
            _recover_broker_directory(broker_dir)
        except FinalEvaluationError as exc:
            raise SandboxTeardownError(
                f"could not recover stale broker; metadata at {broker_dir}"
            ) from exc
    try:
        metadata = recovery_dir.lstat()
    except FileNotFoundError:
        return broker_recovered
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise OSError(f"unsafe stale sandbox metadata: {recovery_dir}")
    name_path = recovery_dir / "container.name"
    if not name_path.exists():
        shutil.rmtree(recovery_dir)
        return True
    container_name = _read_container_name(name_path, lease.identity)
    for attempt in range(5):
        if not _stop_sandbox(None, container_name):
            raise SandboxTeardownError(
                f"could not recover stale sandbox {container_name}; metadata at {recovery_dir}"
            )
        if attempt < 4:
            time.sleep(0.5)
    _discard_cid_path(recovery_dir / "container.cid")
    return True


def _stop_sandbox(
    process: subprocess.Popen[Any] | None,
    container_name: str,
) -> bool:
    process_stopped = True
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        else:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process_stopped = False
    container_removed = _force_remove_sandbox(container_name)
    return container_removed and process_stopped


def _validate_candidate(task_bundle_dir: Path, program_index: int) -> tuple[Path, str]:
    if program_index < 0:
        raise ValueError("program index must be non-negative")
    program_dir = task_bundle_dir / f"program_{program_index}"
    if program_dir.is_symlink() or not program_dir.is_dir():
        raise ValueError(f"missing candidate directory: {program_dir}")
    for artifact in _EVALUATOR_ARTIFACTS:
        artifact_path = program_dir / artifact
        if artifact_path.exists() or artifact_path.is_symlink():
            raise ValueError(f"candidate contains evaluator artifact: {artifact_path}")
    entrypoint_rel = entrypoint_relpath_from_spec(program_dir)
    if entrypoint_rel is None:
        raise ValueError(f"missing valid program_spec.toml: {program_dir}")
    entrypoint = program_dir / entrypoint_rel
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise ValueError(f"candidate entrypoint does not exist: {program_dir / entrypoint_rel}")
    return program_dir, entrypoint_rel


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
