"""Bounded production-entry checks for drafted evaluator topologies."""

from __future__ import annotations

import errno
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator, Literal, Mapping

from src.final_evaluation.candidate_broker import (
    CandidateBroker,
    CandidateBrokerError,
    CandidateBrokerInfrastructureError,
    CandidateContainerCleanupError,
    ContainerMetadataCleanupError,
)
from src.final_evaluation.runner import (
    FinalEvaluationError,
    FinalEvaluationInfrastructureError,
    _candidate_command,
    _evaluation_command,
)
from src.task_drafting.gate import sanitize_diagnostic
from src.investigation.bounded_process import (
    BoundedProcessError,
    BoundedProcessResult,
    ProcessOutputLimitError,
    ProcessTimeoutError,
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
    acquire_gpu_allocation_lease,
)
from src.investigation.program_spec import entrypoint_relpath_from_spec
from src.investigation.task_contract import TaskContractError, load_task_contract
from src.investigation.task_runtime import (
    _installation_environment,
    _prepare_task_broker_directory,
    _remove_container_before_deadline,
)


DEFAULT_PRODUCTION_CANARY_TIMEOUT_SECONDS = 300.0
_MAX_PRODUCTION_CANARY_ATTEMPTS = 2
_POLL_SECONDS = 0.01
_MAX_DIAGNOSTIC_BYTES = 4096

CanaryStatus = Literal["pass", "authored_failure", "infrastructure_failure"]
EvaluationPlane = Literal["search", "final"]


@dataclass(frozen=True, slots=True)
class ProductionCanaryResult:
    status: CanaryStatus
    plane: str
    category: str
    component: str
    stage: str
    diagnostic: str

    @classmethod
    def passed(cls, plane: str, *, diagnostic: str = "") -> ProductionCanaryResult:
        return cls(
            "pass",
            plane,
            "production_entry_passed",
            "harness",
            "production-entry",
            sanitize_diagnostic(diagnostic, max_bytes=_MAX_DIAGNOSTIC_BYTES),
        )


@dataclass(slots=True)
class _ProcessState:
    completed: threading.Event = field(default_factory=threading.Event)
    result: BoundedProcessResult | None = None
    error: BaseException | None = None


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


def _failure(
    plane: str,
    *,
    status: Literal["authored_failure", "infrastructure_failure"],
    category: str,
    component: str,
    diagnostic: str = "",
) -> ProductionCanaryResult:
    return ProductionCanaryResult(
        status,
        plane,
        category,
        component,
        "production-entry",
        sanitize_diagnostic(diagnostic, max_bytes=_MAX_DIAGNOSTIC_BYTES),
    )


def _plane_failure(
    plane: EvaluationPlane,
    suffix: str,
    *,
    infrastructure: bool,
    diagnostic: str = "",
) -> ProductionCanaryResult:
    return _failure(
        plane,
        status="infrastructure_failure" if infrastructure else "authored_failure",
        category=f"{plane}_production_{suffix}",
        component="harness" if infrastructure else f"{plane}_evaluator",
        diagnostic=diagnostic,
    )


def _attempt_diagnostic_entries(attempts: list[ProductionCanaryResult]) -> list[str]:
    entries = []
    for attempt, attempted_result in enumerate(attempts, start=1):
        detail = attempted_result.diagnostic
        if not detail:
            detail = "passed" if attempted_result.status == "pass" else "no diagnostic"
        entries.append(f"attempt {attempt} [{attempted_result.category}]: {detail}")
    return entries


def _bounded_diagnostics(entries: list[str]) -> str:
    if not entries:
        return ""
    entry_bytes = (_MAX_DIAGNOSTIC_BYTES - len(entries) + 1) // len(entries)
    return "\n".join(sanitize_diagnostic(entry, max_bytes=entry_bytes) for entry in entries)


def _with_diagnostics(
    result: ProductionCanaryResult,
    entries: list[str],
) -> ProductionCanaryResult:
    return replace(
        result,
        diagnostic=_bounded_diagnostics(entries),
    )


def _with_entry_outcome(
    failure: ProductionCanaryResult,
    entry: ProductionCanaryResult | None,
) -> ProductionCanaryResult:
    if entry is None:
        return failure
    entries = [f"entry_status={entry.status} entry_category={entry.category}"]
    if entry.diagnostic:
        entries.append(f"entry_diagnostic={entry.diagnostic}")
    if failure.diagnostic:
        entries.append(failure.diagnostic)
    return _with_diagnostics(failure, entries)


def _run_evaluator(
    state: _ProcessState,
    command: list[str],
    *,
    task_dir: Path,
    environment: Mapping[str, str],
    deadline: float,
) -> None:
    try:
        state.result = run_bounded_process(
            command,
            cwd=task_dir,
            env=environment,
            timeout_seconds=max(0.001, deadline - time.monotonic()),
        )
    except BaseException as exc:
        state.error = exc
    finally:
        state.completed.set()


def _stop_evaluator(evaluator_cid: Path, worker: threading.Thread, deadline: float) -> bool:
    _remove_container_before_deadline(evaluator_cid, deadline)
    # Docker may still create or fill its CID while the launch worker finishes.
    # Keep the file linked until its producer stops, then confirm final removal.
    worker.join(timeout=max(0.0, deadline - time.monotonic()))
    if worker.is_alive() or not _remove_container_before_deadline(evaluator_cid, deadline):
        return False
    try:
        evaluator_cid.unlink(missing_ok=True)
    except OSError as exc:
        error_code = errno.errorcode.get(exc.errno) if type(exc.errno) is int else None
        raise ContainerMetadataCleanupError(
            f"operation=evaluator_cid_unlink errno={error_code or 'unknown'}"
        ) from exc
    return True


def _process_failure(
    plane: EvaluationPlane,
    state: _ProcessState,
) -> ProductionCanaryResult:
    error = state.error
    if isinstance(error, ProcessOutputLimitError):
        return _plane_failure(plane, "entry_excessive_output", infrastructure=False)
    if isinstance(error, ProcessTimeoutError):
        return _plane_failure(plane, "entry_timeout", infrastructure=False)
    if isinstance(error, (OSError, BoundedProcessError)) or error is not None:
        return _plane_failure(plane, "process_runtime", infrastructure=True)
    result = state.result
    if result is None:
        return _plane_failure(plane, "process_runtime", infrastructure=True)
    diagnostic = result.stderr.decode("utf-8", errors="replace")
    if result.returncode in {125, 126, 127}:
        return _plane_failure(
            plane,
            "container_runtime",
            infrastructure=True,
            diagnostic=diagnostic,
        )
    if result.returncode != 0:
        return _plane_failure(
            plane,
            "entry_nonzero",
            infrastructure=False,
            diagnostic=diagnostic,
        )
    return _plane_failure(plane, "entry_missing_reset", infrastructure=False)


def _sealed_paths(contract: object, plane: EvaluationPlane) -> tuple[str, ...]:
    if plane == "search":
        return ()
    return (
        contract.evaluators.search,  # type: ignore[attr-defined]
        contract.resources.search_root,  # type: ignore[attr-defined]
        *(
            artifact.path
            for artifact in contract.provenance.artifacts  # type: ignore[attr-defined]
            if artifact.role == "search_helper"
        ),
    )


def run_production_entry_canary(
    *,
    task_dir: Path,
    source_task_dir: Path,
    container_image: str,
    plane: EvaluationPlane,
    caller_deadline: float,
    max_duration_seconds: float = DEFAULT_PRODUCTION_CANARY_TIMEOUT_SECONDS,
    environment: Mapping[str, str] = os.environ,
) -> ProductionCanaryResult:
    """Reach one completed declared reset in the ordinary production path, then stop."""
    now = time.monotonic()
    if (
        not math.isfinite(caller_deadline)
        or caller_deadline <= now
        or not math.isfinite(max_duration_seconds)
        or max_duration_seconds <= 0
    ):
        return _plane_failure(plane, "entry_timeout", infrastructure=False)
    deadline = min(caller_deadline, now + max_duration_seconds)
    cleanup_reserve = min(30.0, max(0.001, (deadline - now) * 0.1))
    execution_deadline = deadline - cleanup_reserve
    task_dir = task_dir.resolve()
    source_task_dir = source_task_dir.resolve()
    try:
        contract = load_task_contract(task_dir / "task_contract.json")
        entrypoint = entrypoint_relpath_from_spec(task_dir / "program_0")
        if entrypoint is None:
            return _plane_failure(plane, "entrypoint_invalid", infrastructure=False)
        image_id = resolve_immutable_image(container_image)
        compute = getattr(contract.compute, plane)
        evaluator = getattr(contract.evaluators, plane)
        declared_resets = (
            contract.protocol.search_reset_sequence
            if plane == "search"
            else contract.protocol.final_reset_sequence
        )
        expected_reset = declared_resets[0]
        broker_phase = "runtime-search" if plane == "search" else "final-smoke"
        broker_dir = _prepare_task_broker_directory(
            source_task_dir,
            broker_phase,
            deadline=deadline,
        )
        program_dir = task_dir / "program_0"
        entrypoint_path = program_dir / entrypoint
        with tempfile.TemporaryDirectory(prefix=f"agentic-task-{plane}-entry-") as output:
            evaluator_command = _evaluation_command(
                task_dir=task_dir,
                program_dir=program_dir,
                entrypoint=entrypoint_path,
                evaluator=task_dir / evaluator,
                results_dir=Path(output),
                python_bin=task_dir / ".venv/bin/python",
                container_image=image_id,
                gpu_count=compute.evaluator_gpu_count,
                cpu_cores=compute.cpu_cores,
                memory_gb=compute.memory_gb,
                environment=environment,
                broker_dir=broker_dir,
                candidate_editable_paths=contract.candidate.editable_paths,
                sealed_task_paths=_sealed_paths(contract, plane),
                public_artifact_bundle=contract.resources.public_artifact_bundle,
                rpc_timeout_seconds=max(1, math.ceil(execution_deadline - time.monotonic())),
                plane=plane,
                draft_smoke=False,
            )
            candidate_command = _candidate_command(
                task_dir=task_dir,
                program_dir=program_dir,
                entrypoint=entrypoint_path,
                container_image=image_id,
                contract=contract,
                environment=environment,
                broker_dir=broker_dir,
                plane=plane,
            )
            evaluator_cid = broker_dir / "evaluator.cid"
            state = _ProcessState()
            entry_result: ProductionCanaryResult | None = None
            try:
                with tempfile.TemporaryDirectory(
                    prefix="agentic-task-production-entry-home-"
                ) as home:
                    safe_environment = _installation_environment(
                        environment,
                        Path(home),
                        task_dir,
                    )
                    gpu_lease: GpuAllocationLease | None = (
                        acquire_gpu_allocation_lease(
                            environment.get("AGENT_CONTAINER_GPUS"),
                            compute.gpu_count,
                            lease_root=GPU_ALLOCATION_LEASE_ROOT,
                            deadline=execution_deadline,
                        )
                        if compute.gpu_count
                        else None
                    )
                    with _gpu_allocation_scope(gpu_lease):
                        remaining_timeout = execution_deadline - time.monotonic()
                        if remaining_timeout <= 0:
                            return _plane_failure(
                                plane,
                                (
                                    GPU_ALLOCATION_BUSY_CATEGORY
                                    if gpu_lease is not None
                                    else "entry_timeout"
                                ),
                                infrastructure=gpu_lease is not None,
                            )
                        if gpu_lease is not None:
                            gpu_lease.prepare_runtime()
                            evaluator_command = gpu_lease.wrap_command(evaluator_command)
                            candidate_command = gpu_lease.wrap_command(candidate_command)
                        with CandidateBroker(
                            candidate_command,
                            broker_dir,
                            rpc_timeout_seconds=remaining_timeout,
                            reset_timeout_seconds=remaining_timeout,
                            max_rpc_calls=compute.max_candidate_rpc_calls,
                            expected_reset_seeds=(expected_reset,),
                            completed_reset_limit=1,
                        ) as broker:
                            worker = threading.Thread(
                                target=_run_evaluator,
                                args=(state, evaluator_command),
                                kwargs={
                                    "task_dir": task_dir,
                                    "environment": safe_environment,
                                    "deadline": execution_deadline,
                                },
                                daemon=True,
                            )
                            evaluator_cleanup_attempted = False

                            def stop_evaluator() -> bool:
                                nonlocal evaluator_cleanup_attempted
                                try:
                                    return _stop_evaluator(evaluator_cid, worker, deadline)
                                except ContainerMetadataCleanupError:
                                    raise
                                except BaseException:
                                    if gpu_lease is not None:
                                        gpu_lease.quarantine()
                                    raise
                                finally:
                                    evaluator_cleanup_attempted = True

                            try:
                                worker.start()
                                while True:
                                    if broker.completed_reset_seeds:
                                        if broker.completed_reset_seeds != (expected_reset,):
                                            entry_result = _plane_failure(
                                                plane,
                                                "entry_reset_sequence",
                                                infrastructure=False,
                                            )
                                        else:
                                            entry_result = ProductionCanaryResult.passed(plane)
                                    elif state.completed.is_set():
                                        entry_result = _process_failure(plane, state)
                                    elif time.monotonic() >= execution_deadline:
                                        entry_result = _plane_failure(
                                            plane, "entry_timeout", infrastructure=False,
                                        )
                                    if entry_result is not None:
                                        if not stop_evaluator():
                                            if gpu_lease is not None:
                                                gpu_lease.quarantine()
                                            return _with_entry_outcome(
                                                _plane_failure(
                                                    plane, "container_cleanup", infrastructure=True,
                                                ),
                                                entry_result,
                                            )
                                        return entry_result
                                    state.completed.wait(
                                        min(
                                            _POLL_SECONDS,
                                            max(0.0, execution_deadline - time.monotonic()),
                                        )
                                    )
                            finally:
                                if not evaluator_cleanup_attempted:
                                    cleanup_verified = stop_evaluator()
                                    if not cleanup_verified and gpu_lease is not None:
                                        gpu_lease.quarantine()
            except GpuLeaseTimeout:
                return _plane_failure(
                    plane,
                    GPU_ALLOCATION_BUSY_CATEGORY,
                    infrastructure=True,
                )
            except GpuLeaseQuarantined:
                return _with_entry_outcome(
                    _plane_failure(
                        plane, GPU_ALLOCATION_QUARANTINED_CATEGORY, infrastructure=True,
                    ),
                    entry_result,
                )
            except GpuLeaseError:
                return _plane_failure(
                    plane,
                    GPU_ALLOCATION_INVALID_CATEGORY,
                    infrastructure=True,
                )
            except ContainerMetadataCleanupError as exc:
                return _with_entry_outcome(
                    _plane_failure(
                        plane, "harness_runtime", infrastructure=True, diagnostic=str(exc),
                    ),
                    entry_result,
                )
            except CandidateBrokerInfrastructureError as exc:
                return _with_entry_outcome(
                    _plane_failure(
                        plane, "candidate_runtime", infrastructure=True, diagnostic=str(exc),
                    ),
                    entry_result,
                )
            except CandidateBrokerError as exc:
                return _plane_failure(
                    plane,
                    "candidate_protocol",
                    infrastructure=False,
                    diagnostic=str(exc),
                )
            except OSError as exc:
                error_code = errno.errorcode.get(exc.errno) if type(exc.errno) is int else None
                return _with_entry_outcome(
                    _plane_failure(
                        plane, "harness_runtime", infrastructure=True,
                        diagnostic=(
                            "operation=evaluator_runtime_filesystem "
                            f"errno={error_code or 'unknown'}"
                        ),
                    ),
                    entry_result,
                )
    except CandidateBrokerInfrastructureError as exc:
        return _plane_failure(
            plane,
            "harness_runtime",
            infrastructure=True,
            diagnostic=str(exc),
        )
    except FinalEvaluationInfrastructureError as exc:
        return _plane_failure(
            plane,
            str(exc),
            infrastructure=True,
            diagnostic=str(exc),
        )
    except (TaskContractError, FinalEvaluationError, ValueError) as exc:
        return _plane_failure(
            plane,
            "command_invalid",
            infrastructure=False,
            diagnostic=str(exc),
        )
    except OSError:
        return _plane_failure(plane, "harness_runtime", infrastructure=True)


def run_production_entry_canaries(
    *,
    task_dir: Path,
    source_task_dir: Path,
    container_image: str,
    caller_deadline: float,
    max_duration_seconds: float = DEFAULT_PRODUCTION_CANARY_TIMEOUT_SECONDS,
    environment: Mapping[str, str] = os.environ,
) -> ProductionCanaryResult:
    """Require production entry to succeed for search and final planes."""
    recovered_entries: list[str] = []
    for plane in ("search", "final"):
        attempts: list[ProductionCanaryResult] = []
        for _ in range(_MAX_PRODUCTION_CANARY_ATTEMPTS):
            result = run_production_entry_canary(
                task_dir=task_dir,
                source_task_dir=source_task_dir,
                container_image=container_image,
                plane=plane,
                caller_deadline=caller_deadline,
                max_duration_seconds=max_duration_seconds,
                environment=environment,
            )
            attempts.append(result)
            if result.status != "infrastructure_failure" or time.monotonic() >= caller_deadline:
                break
        attempt_diagnostics = _attempt_diagnostic_entries(attempts) if len(attempts) > 1 else []
        if result.status != "pass":
            if recovered_entries:
                terminal_diagnostics = attempt_diagnostics or _attempt_diagnostic_entries(attempts)
                return _with_diagnostics(
                    result,
                    [*recovered_entries, *terminal_diagnostics],
                )
            if attempt_diagnostics:
                return _with_diagnostics(result, attempt_diagnostics)
            return result
        recovered_entries.extend(attempt_diagnostics)
    return ProductionCanaryResult.passed(
        "both",
        diagnostic=_bounded_diagnostics(recovered_entries),
    )
