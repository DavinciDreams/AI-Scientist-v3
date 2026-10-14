"""Managed runtime preflight for host-sealed variant evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

from src.investigation import task_runtime
from src.investigation.evaluator_tree import managed_evaluator_tree_digest
from src.investigation.task_contract import load_task_contract

logger = logging.getLogger(__name__)
_RUNTIME_SETUP_SECONDS = 1800
_REVALIDATION_CATEGORIES = {
    "qualified_runtime_cache_unavailable",
    "qualified_runtime_revalidation_required",
}


def _verify_evaluator_seal(task_dir: Path, expected: str) -> None:
    if managed_evaluator_tree_digest(task_dir) != expected:
        raise ValueError("task evaluator tree changed before runtime preparation")


def prepare_evaluation_runtime(
    task_dir: Path,
    *,
    sandbox_image: str,
    container_network: str,
    evaluator_tree_sha256: str,
) -> bool:
    """Provision and validate the canonical task-local evaluation runtime."""
    try:
        _verify_evaluator_seal(task_dir, evaluator_tree_sha256)
        contract_path = task_dir / "task_contract.json"
        receipt_path = task_dir / ".agentic_task_qualification.json"
        for path in (contract_path, receipt_path):
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError(f"runtime authority must be a regular file: {path.name}")
        if contract_path.exists():
            contract = load_task_contract(contract_path)
            gpu_evaluation = contract.compute.search.accelerator == "gpu"
            if receipt_path.exists():
                from src.task_drafting.qualification import activate_qualified_task_runtime

                report = activate_qualified_task_runtime(
                    task_dir, container_image=sandbox_image, gpu_evaluation=gpu_evaluation,
                    timeout_seconds=_RUNTIME_SETUP_SECONDS,
                )
                _verify_evaluator_seal(task_dir, evaluator_tree_sha256)
                if report.ready:
                    return True
                if report.failure_category not in _REVALIDATION_CATEGORIES:
                    logger.error("Qualified runtime activation failed: %s", report.failure_category)
                    return False
            return task_runtime.prepare_task_runtime(
                task_dir,
                timeout_seconds=contract.compute.search.wall_clock_seconds + _RUNTIME_SETUP_SECONDS,
                gpu_evaluation=gpu_evaluation,
                containerize=True,
                container_image=sandbox_image,
                container_network=container_network,
                evaluator_tree_sha256=evaluator_tree_sha256,
            )
        if receipt_path.exists():
            raise ValueError("qualified runtime requires a task contract")
        return task_runtime.prepare_task_runtime(
            task_dir,
            containerize=True,
            container_image=sandbox_image,
            container_network=container_network,
            evaluator_tree_sha256=evaluator_tree_sha256,
        )
    except (OSError, ValueError) as exc:
        logger.error("Managed runtime preparation failed for %s: %s", task_dir, exc)
        return False
