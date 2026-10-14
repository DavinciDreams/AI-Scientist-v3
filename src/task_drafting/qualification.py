"""Harness-owned receipts for fully qualified drafted tasks."""

from __future__ import annotations

import json
import math
import os
import stat
import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from src.investigation.evaluator_tree import managed_evaluator_tree_digest
from src.investigation.container_images import resolve_immutable_image
from src.task_drafting.gate import sanitize_diagnostic
from src.investigation.task_alignment import task_tree_fingerprint
from src.investigation.runtime_output import TaskRuntimeOutputEvidence
from src.investigation.storage_failures import storage_failure_category, warn_storage_failure
from src.investigation.task_contract import TaskContractError, load_task_contract
from src.investigation.task_runtime import (
    TaskRuntimeEvaluationRequirement,
    TaskRuntimeFailureEvidence,
    TaskRuntimePreparationReport,
    TaskRuntimePreparationResult,
    _evaluator_digest,
    _requirements_digest,
    _tooling_digest,
    _write_runtime_manifest,
    discard_managed_task_runtime,
    prepare_task_runtime_report,
)
from src.investigation.task_runtime_cache import (
    TaskRuntimeInstallCache,
    TaskRuntimeInstallCacheDeadlineExceeded,
    TaskRuntimeInstallCacheKey,
    prepare_task_runtime_install_cache,
)

_SCHEMA_VERSION = 2
_VALIDATION_PROFILE = "production_search_v2"
_LEGACY_SCHEMA_VERSION = 1
_LEGACY_VALIDATION_PROFILE = "production_search_v1"
_HEX_DIGITS = frozenset("0123456789abcdef")
_MAX_QUALIFICATION_JSON_BYTES = 1024 * 1024


class TaskQualificationError(ValueError):
    """Raised when a task qualification receipt is unsafe, stale, or malformed."""


class TaskQualificationCacheUnavailable(TaskQualificationError):
    """Raised when a valid qualified receipt outlives its private install cache."""


class TaskQualificationRevalidationRequired(TaskQualificationError):
    """Raised when host/runtime inputs changed after a valid qualification."""


@dataclass(frozen=True, slots=True)
class TaskQualificationReceipt:
    publication_plan_sha256: str
    image_id: str
    evaluator_tree_sha256: str
    requirements_sha256: str
    tooling_sha256: str
    runtime_cache_key: str
    baseline_combined_score: float
    accelerator_model: str
    evaluator_gpu_count: int
    candidate_gpu_count: int
    baseline_correct: bool = True
    baseline_feasible: bool = True


def _safe_task_fingerprint(task_dir: Path) -> str:
    try:
        return task_tree_fingerprint(task_dir)
    except (OSError, ValueError):
        return ""


@dataclass(frozen=True, slots=True)
class TaskQualificationAttempt:
    status: Literal["pass", "authored_failure", "infrastructure_failure"]
    category: str
    stage: str
    diagnostic: str
    task_fingerprint: str
    elapsed_seconds: float
    receipt_path: str | None
    failure_evidence: TaskRuntimeFailureEvidence | None = None
    output_evidence: TaskRuntimeOutputEvidence | None = None
    compute_usage: dict[str, object] | None = None

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _read_owned_json(path: Path, *, maximum_bytes: int) -> object:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TaskQualificationError(f"could not open task qualification artifact: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
        ):
            raise TaskQualificationError("task qualification artifact is unsafe or too large")
        content = bytearray()
        while len(content) <= maximum_bytes:
            chunk = os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        after = os.fstat(descriptor)
        if (
            len(content) > maximum_bytes
            or (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_uid,
                before.st_gid,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_gid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or len(content) != after.st_size
        ):
            raise TaskQualificationError("task qualification artifact changed while reading")
    finally:
        os.close(descriptor)
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TaskQualificationError("task qualification artifact is not valid JSON") from exc


def _validated_payload(payload: object) -> TaskQualificationReceipt:
    if not isinstance(payload, dict):
        raise TaskQualificationError("task qualification receipt must be an object")
    if (
        payload.get("schema_version") == _LEGACY_SCHEMA_VERSION
        and payload.get("validation_profile") == _LEGACY_VALIDATION_PROFILE
    ):
        raise TaskQualificationRevalidationRequired(
            "task qualification receipt uses the legacy production-search profile"
        )
    expected = {
        "schema_version",
        "validation_profile",
        *TaskQualificationReceipt.__dataclass_fields__,
    }
    if frozenset(payload) != frozenset(expected):
        raise TaskQualificationError("task qualification receipt fields are invalid")
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise TaskQualificationError("task qualification receipt schema is unsupported")
    if payload["validation_profile"] != _VALIDATION_PROFILE:
        raise TaskQualificationError("task qualification validation profile is unsupported")
    for field in (
        "publication_plan_sha256",
        "evaluator_tree_sha256",
        "requirements_sha256",
        "tooling_sha256",
        "runtime_cache_key",
    ):
        if not _is_sha256(payload[field]):
            raise TaskQualificationError(f"task qualification {field} must be a SHA-256")
    image_id = payload["image_id"]
    if not (
        isinstance(image_id, str)
        and image_id.startswith("sha256:")
        and _is_sha256(image_id.removeprefix("sha256:"))
    ):
        raise TaskQualificationError("task qualification image_id must be immutable")
    score = payload["baseline_combined_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise TaskQualificationError("task qualification baseline score must be finite")
    baseline_correct = payload["baseline_correct"]
    if not isinstance(baseline_correct, bool):
        raise TaskQualificationError("task qualification baseline correctness must be boolean")
    baseline_feasible = payload["baseline_feasible"]
    if baseline_feasible is not True:
        raise TaskQualificationError("task qualification baseline feasibility must be true")
    accelerator_model = payload["accelerator_model"]
    if (
        not isinstance(accelerator_model, str)
        or not accelerator_model.strip()
        or len(accelerator_model.encode("utf-8")) > 256
        or any(character in accelerator_model for character in "\r\n\0")
    ):
        raise TaskQualificationError("task qualification accelerator model is invalid")
    for field in ("evaluator_gpu_count", "candidate_gpu_count"):
        count = payload[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TaskQualificationError(f"task qualification {field} must be non-negative")
    return TaskQualificationReceipt(
        publication_plan_sha256=payload["publication_plan_sha256"],
        image_id=image_id,
        evaluator_tree_sha256=payload["evaluator_tree_sha256"],
        requirements_sha256=payload["requirements_sha256"],
        tooling_sha256=payload["tooling_sha256"],
        runtime_cache_key=payload["runtime_cache_key"],
        baseline_combined_score=float(score),
        accelerator_model=accelerator_model,
        evaluator_gpu_count=payload["evaluator_gpu_count"],
        candidate_gpu_count=payload["candidate_gpu_count"],
        baseline_correct=baseline_correct,
        baseline_feasible=baseline_feasible,
    )


def write_task_qualification_receipt(path: Path, receipt: TaskQualificationReceipt) -> None:
    """Create one private receipt without following or replacing filesystem entries."""
    validated = _validated_payload(
        {
            "schema_version": _SCHEMA_VERSION,
            "validation_profile": _VALIDATION_PROFILE,
            **asdict(receipt),
        }
    )
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "validation_profile": _VALIDATION_PROFILE,
        **asdict(validated),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise TaskQualificationError(f"could not create task qualification receipt: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def qualification_receipt_from_runtime(
    task_dir: Path,
    *,
    publication_plan_sha256: str,
) -> TaskQualificationReceipt:
    """Bind a successful managed search runtime to its publishable scaffold."""
    manifest_path = task_dir / ".agentic_runtime.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise TaskQualificationError("qualified task is missing a managed runtime manifest")
    try:
        manifest = _read_owned_json(
            manifest_path,
            maximum_bytes=_MAX_QUALIFICATION_JSON_BYTES,
        )
        contract = load_task_contract(task_dir / "task_contract.json")
    except (OSError, TaskQualificationError, TaskContractError) as exc:
        raise TaskQualificationError("qualified task runtime metadata is invalid") from exc
    expected_manifest_fields = {
        "schema_version",
        "image_id",
        "baseline_combined_score",
        "baseline_correct",
        "baseline_feasible",
        "evaluator_sha256",
        "evaluator_tree_sha256",
        "gpu_evaluation",
        "cuda_visible_devices",
        "sandbox_gpus",
        "requirements_sha256",
        "tooling_sha256",
    }
    if not isinstance(manifest, dict) or frozenset(manifest) != frozenset(
        expected_manifest_fields
    ):
        raise TaskQualificationError("qualified task runtime manifest fields are invalid")
    cache_key = TaskRuntimeInstallCacheKey(
        image_id=manifest["image_id"],
        requirements_sha256=manifest["requirements_sha256"],
        tooling_sha256=manifest["tooling_sha256"],
    )
    compute = contract.compute.search
    return _validated_payload(
        {
            "schema_version": _SCHEMA_VERSION,
            "validation_profile": _VALIDATION_PROFILE,
            "publication_plan_sha256": publication_plan_sha256,
            "image_id": manifest["image_id"],
            "evaluator_tree_sha256": manifest["evaluator_tree_sha256"],
            "requirements_sha256": manifest["requirements_sha256"],
            "tooling_sha256": manifest["tooling_sha256"],
            "runtime_cache_key": cache_key.digest,
            "baseline_combined_score": manifest["baseline_combined_score"],
            "baseline_correct": manifest["baseline_correct"],
            "baseline_feasible": manifest["baseline_feasible"],
            "accelerator_model": compute.accelerator_model,
            "evaluator_gpu_count": compute.evaluator_gpu_count,
            "candidate_gpu_count": compute.candidate_gpu_count,
        }
    )


def load_task_qualification_receipt(
    path: Path,
    *,
    expected_publication_plan_sha256: str,
) -> TaskQualificationReceipt:
    """Load a receipt and bind it to the publication plan being activated."""
    if path.is_symlink() or not path.is_file():
        raise TaskQualificationError("task qualification receipt must be a regular file")
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise TaskQualificationError("task qualification receipt must be an owned regular file")
    payload = _read_owned_json(path, maximum_bytes=_MAX_QUALIFICATION_JSON_BYTES)
    receipt = _validated_payload(payload)
    if receipt.publication_plan_sha256 != expected_publication_plan_sha256:
        raise TaskQualificationError("task qualification receipt has a different publication plan")
    return receipt


def _storage_failure_attempt(
    error: BaseException, *, stage: str, task_dir: Path, elapsed_seconds: float = 0.0,
    task_fingerprint: str | None = None,
) -> TaskQualificationAttempt | None:
    category = storage_failure_category(error)
    if category is None:
        return None
    warn_storage_failure(error, operation=stage)
    return TaskQualificationAttempt(
        "infrastructure_failure", category, stage,
        f"Host storage failed during {stage}; {category}. No authored repair is warranted.",
        _safe_task_fingerprint(task_dir) if task_fingerprint is None else task_fingerprint,
        elapsed_seconds, None,
    )


def qualify_task_for_publication(
    *,
    task_dir: Path,
    publication_task_dir: Path,
    output_dir: Path,
    receipt_path: Path,
    timeout_seconds: int,
    container_image: str,
    container_network: str,
    install_cache_dir: Path,
    max_runtime_bytes: int,
    prepare: Callable[..., TaskRuntimePreparationReport] = prepare_task_runtime_report,
    baseline_results_dir: Path | None = None,
) -> TaskQualificationAttempt:
    """Run the full production search path and bind its runtime to a publication plan."""
    from src.task_drafting.publication import (
        TaskPublicationError,
        task_publication_plan_sha256,
    )

    task_dir = task_dir.resolve(strict=True)
    output_dir = output_dir.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=False)
    if receipt_path.parent != output_dir or receipt_path.exists() or receipt_path.is_symlink():
        raise TaskQualificationError("qualification receipt must be a new file in output_dir")
    try:
        contract = load_task_contract(task_dir / "task_contract.json")
        evaluator_tree_sha256 = managed_evaluator_tree_digest(task_dir)
        publication_plan_sha256 = task_publication_plan_sha256(
            task_dir=publication_task_dir,
            staged_task_dir=task_dir,
            max_runtime_bytes=max_runtime_bytes,
        )
    except (OSError, TaskContractError, TaskPublicationError, ValueError) as exc:
        storage_failure = _storage_failure_attempt(exc, stage="publication-plan", task_dir=task_dir)
        if storage_failure is not None:
            return storage_failure
        if isinstance(exc, OSError):
            raise
        return TaskQualificationAttempt(
            "authored_failure",
            "qualification_publication_plan",
            "publication-plan",
            sanitize_diagnostic(str(exc), max_bytes=8 * 1024),
            _safe_task_fingerprint(task_dir),
            0.0,
            None,
        )
    archive_options = (
        {"baseline_results_dir": baseline_results_dir} if baseline_results_dir is not None else {}
    )
    report = prepare(
        task_dir,
        timeout_seconds,
        containerize=True,
        container_image=container_image,
        container_network=container_network,
        evaluator_tree_sha256=evaluator_tree_sha256,
        gpu_evaluation=contract.compute.search.accelerator == "gpu",
        draft_smoke=False,
        install_cache_dir=install_cache_dir,
        evaluation_requirement=TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT,
        **archive_options,
    )
    if report.ready:
        try:
            current_plan_sha256 = task_publication_plan_sha256(
                task_dir=publication_task_dir,
                staged_task_dir=task_dir,
                max_runtime_bytes=max_runtime_bytes,
            )
            receipt = qualification_receipt_from_runtime(
                task_dir,
                publication_plan_sha256=publication_plan_sha256,
            )
        except (OSError, TaskContractError, TaskPublicationError, TaskQualificationError, ValueError) as exc:
            storage_failure = _storage_failure_attempt(
                exc, stage="production-search", task_dir=task_dir,
                elapsed_seconds=report.elapsed_seconds,
                task_fingerprint=report.task_fingerprint,
            )
            if storage_failure is not None:
                return storage_failure
            if isinstance(exc, OSError):
                raise
            return TaskQualificationAttempt(
                "authored_failure",
                "qualification_modified_scaffold",
                "production-search",
                sanitize_diagnostic(str(exc), max_bytes=8 * 1024),
                report.task_fingerprint,
                report.elapsed_seconds,
                None,
                compute_usage=report.compute_usage,
            )
        if current_plan_sha256 != publication_plan_sha256:
            return TaskQualificationAttempt(
                "authored_failure",
                "qualification_modified_scaffold",
                "production-search",
                "Full search qualification modified the publishable scaffold.",
                report.task_fingerprint,
                report.elapsed_seconds,
                None,
                compute_usage=report.compute_usage,
            )
        try:
            write_task_qualification_receipt(receipt_path, receipt)
        except (OSError, TaskQualificationError, TaskPublicationError) as exc:
            storage_failure = _storage_failure_attempt(
                exc, stage="qualification_receipt", task_dir=task_dir,
                elapsed_seconds=report.elapsed_seconds,
                task_fingerprint=report.task_fingerprint,
            )
            if storage_failure is not None:
                return storage_failure
            raise
        return TaskQualificationAttempt(
            "pass",
            "qualification_passed",
            "production-search",
            "",
            report.task_fingerprint,
            report.elapsed_seconds,
            str(receipt_path),
            compute_usage=report.compute_usage,
        )
    status: Literal["authored_failure", "infrastructure_failure"]
    if report.result in {
        TaskRuntimePreparationResult.AUTHORED_FAILURE,
        TaskRuntimePreparationResult.DEPENDENCY_FAILURE,
    }:
        status = "authored_failure"
    else:
        status = "infrastructure_failure"
    return TaskQualificationAttempt(
        status,
        report.failure_category or "qualification_runtime_failure",
        report.failure_stage or "production-search",
        report.diagnostic,
        report.task_fingerprint,
        report.elapsed_seconds,
        None,
        report.failure_evidence,
        report.output_evidence,
        compute_usage=report.compute_usage,
    )


def activate_qualified_task_runtime(
    task_dir: Path,
    *,
    container_image: str,
    gpu_evaluation: bool,
    timeout_seconds: float = 1800,
    base_environment: Mapping[str, str] = os.environ,
) -> TaskRuntimePreparationReport:
    """Materialize the exact qualified install without rerunning its full search baseline."""
    started_at = time.monotonic()
    task_dir = task_dir.resolve(strict=True)
    materialized = False
    try:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("qualified runtime activation timeout must be positive and finite")
        deadline = started_at + timeout_seconds
        from src.task_drafting.publication import (
            TaskPublicationError,
            validate_published_task_plan,
        )

        publication_sha256 = validate_published_task_plan(task_dir)
        receipt = load_task_qualification_receipt(
            task_dir / ".agentic_task_qualification.json",
            expected_publication_plan_sha256=publication_sha256,
        )
        contract = load_task_contract(task_dir / "task_contract.json")
        search_compute = contract.compute.search
        if (
            search_compute.accelerator_model != receipt.accelerator_model
            or search_compute.evaluator_gpu_count != receipt.evaluator_gpu_count
            or search_compute.candidate_gpu_count != receipt.candidate_gpu_count
            or (search_compute.accelerator == "gpu") is not gpu_evaluation
        ):
            raise TaskQualificationError("qualified task compute topology changed")
        image_id = resolve_immutable_image(container_image)
        evaluator_tree_sha256 = managed_evaluator_tree_digest(task_dir)
        requirements_sha256 = _requirements_digest(task_dir / "requirements.txt")
        tooling_sha256 = _tooling_digest()
        if (
            evaluator_tree_sha256 != receipt.evaluator_tree_sha256
            or requirements_sha256 != receipt.requirements_sha256
        ):
            raise TaskQualificationError("qualified runtime inputs changed")
        if image_id != receipt.image_id or tooling_sha256 != receipt.tooling_sha256:
            raise TaskQualificationRevalidationRequired(
                "qualified host runtime inputs changed"
            )
        cache_key = TaskRuntimeInstallCacheKey(
            image_id=image_id,
            requirements_sha256=requirements_sha256,
            tooling_sha256=tooling_sha256,
        )
        if cache_key.digest != receipt.runtime_cache_key:
            raise TaskQualificationError("qualified runtime cache key changed")
        cache_root = prepare_task_runtime_install_cache(task_dir, reset=False)
        cache = TaskRuntimeInstallCache(cache_root)
        if cache.lookup(cache_key, deadline=deadline) is None:
            raise TaskQualificationCacheUnavailable(
                "qualified runtime cache entry is unavailable"
            )
        discard_managed_task_runtime(task_dir)
        cache.materialize(
            cache_key,
            task_dir / ".venv",
            deadline=deadline,
        )
        materialized = True
        _require_activation_deadline(deadline)
        evaluator_sha256 = _evaluator_digest(task_dir / contract.evaluators.search)
        cuda_visible_devices = (
            base_environment.get("CUDA_VISIBLE_DEVICES") if gpu_evaluation else ""
        )
        sandbox_gpus = (
            base_environment.get("AGENT_CONTAINER_GPUS", "").strip() or None
            if gpu_evaluation
            else None
        )
        _write_runtime_manifest(
            task_dir,
            {
                "image_id": image_id,
                "baseline_combined_score": receipt.baseline_combined_score,
                "baseline_correct": receipt.baseline_correct,
                "baseline_feasible": receipt.baseline_feasible,
                "evaluator_sha256": evaluator_sha256,
                "evaluator_tree_sha256": evaluator_tree_sha256,
                "gpu_evaluation": gpu_evaluation,
                "cuda_visible_devices": cuda_visible_devices,
                "sandbox_gpus": sandbox_gpus,
                "requirements_sha256": requirements_sha256,
                "schema_version": 1,
                "tooling_sha256": tooling_sha256,
            },
        )
        _require_activation_deadline(deadline)
        return TaskRuntimePreparationReport(
            TaskRuntimePreparationResult.READY,
            None,
            None,
            "",
            _safe_task_fingerprint(task_dir),
            time.monotonic() - started_at,
        )
    except (
        OSError,
        TaskContractError,
        TaskPublicationError,
        TaskQualificationError,
        TaskRuntimeInstallCacheDeadlineExceeded,
        ValueError,
    ) as exc:
        if materialized:
            try:
                discard_managed_task_runtime(task_dir)
            except (OSError, ValueError):
                pass
        category = (
            "qualified_runtime_activation_timeout"
            if isinstance(exc, TaskRuntimeInstallCacheDeadlineExceeded)
            else (
                "qualified_runtime_cache_unavailable"
                if isinstance(exc, TaskQualificationCacheUnavailable)
                else (
                    "qualified_runtime_revalidation_required"
                    if isinstance(exc, TaskQualificationRevalidationRequired)
                    else "qualified_runtime_activation"
                )
            )
        )
        return TaskRuntimePreparationReport(
            TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE,
            "qualification_activation",
            storage_failure_category(exc) or category,
            (
                "Host storage failed during qualification activation"
                if storage_failure_category(exc) is not None
                else sanitize_diagnostic(str(exc), max_bytes=8 * 1024)
            ),
            _safe_task_fingerprint(task_dir),
            time.monotonic() - started_at,
        )


def _require_activation_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TaskRuntimeInstallCacheDeadlineExceeded(
            "qualified runtime activation deadline exceeded"
        )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Qualify one drafted task for publication")
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--publication-task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--container-network", required=True)
    parser.add_argument("--install-cache-dir", type=Path, required=True)
    parser.add_argument("--max-runtime-bytes", type=int, required=True)
    parser.add_argument("--baseline-results-dir", type=Path,
                        help="new external directory for validated production search artifacts")
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 604800:
        parser.error("--timeout-seconds must be from 1 to 604800")
    try:
        attempt = qualify_task_for_publication(
            task_dir=args.task_dir,
            publication_task_dir=args.publication_task_dir,
            output_dir=args.output_dir,
            receipt_path=args.receipt,
            timeout_seconds=args.timeout_seconds,
            container_image=args.container_image,
            container_network=args.container_network,
            install_cache_dir=args.install_cache_dir,
            max_runtime_bytes=args.max_runtime_bytes,
            baseline_results_dir=args.baseline_results_dir,
        )
    except Exception as exc:
        storage_category = storage_failure_category(exc)
        if storage_category is not None:
            warn_storage_failure(exc, operation="qualification")
        attempt = TaskQualificationAttempt(
            "infrastructure_failure",
            storage_category or "qualification_harness_failure",
            "production-search",
            storage_category or sanitize_diagnostic(str(exc), max_bytes=8 * 1024),
            "",
            0.0,
            None,
        )
    print(json.dumps(attempt.to_payload(), ensure_ascii=False, sort_keys=True))
    if attempt.status == "pass":
        return 0
    return 3 if attempt.status == "authored_failure" else 4


if __name__ == "__main__":
    raise SystemExit(_main())
