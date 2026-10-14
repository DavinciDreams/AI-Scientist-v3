"""Stage 4: Materialize post-hoc analysis workspaces."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import clone_program_bundle, copy_task_files, write_json
from .models import AnalysisSelection, AnalysisSpec, is_valid_analysis_id
from .rubric_contract import build_rubric_contract, build_target_rubrics

logger = logging.getLogger(__name__)


def materialize_analysis_workspaces(
    *,
    analysis_ready_bundle_dir: Path,
    posthoc_run_dir: Path,
    selection: AnalysisSelection,
    agent: str,
    agent_model: str | None,
    max_attempts_per_analysis: int,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
) -> list[AnalysisSpec]:
    analysis_ready_bundle_dir = analysis_ready_bundle_dir.resolve()
    posthoc_run_dir = posthoc_run_dir.resolve()
    analyses_dir = posthoc_run_dir / "analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)
    invalid_ids = [
        candidate.analysis_id
        for candidate in selection.accepted
        if not is_valid_analysis_id(candidate.analysis_id)
    ]
    if invalid_ids:
        raise ValueError(f"Invalid analysis_id values: {invalid_ids}")
    accepted_ids = {candidate.analysis_id for candidate in selection.accepted}

    for existing_dir in analyses_dir.iterdir():
        if not existing_dir.is_dir():
            continue
        if existing_dir.name in accepted_ids:
            continue
        shutil.rmtree(existing_dir)

    specs: list[AnalysisSpec] = []
    for candidate in selection.accepted:
        analysis_dir = (analyses_dir / candidate.analysis_id).resolve()
        if not analysis_dir.is_relative_to(analyses_dir.resolve()):
            raise ValueError(f"Invalid analysis_id: {candidate.analysis_id!r}")
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)
        analysis_dir.mkdir(parents=True, exist_ok=True)

        task_bundle_dir = analysis_dir / "task_bundle"
        copy_task_files(analysis_ready_bundle_dir, task_bundle_dir)
        evaluator_path = task_bundle_dir / "evaluate.py"
        if evaluator_path.exists():
            evaluator_path.unlink()
        extraction_manifest_path = task_bundle_dir / "extraction_manifest.json"
        if extraction_manifest_path.exists():
            extraction_manifest_path.unlink()
        clone_program_bundle(
            analysis_ready_bundle_dir / "program_0",
            task_bundle_dir / "program_0",
        )
        if (analysis_ready_bundle_dir / "reference_program").is_dir():
            clone_program_bundle(
                analysis_ready_bundle_dir / "reference_program",
                task_bundle_dir / "reference_program",
            )
        _remove_task_bundle_symlinks(task_bundle_dir)

        spec = AnalysisSpec(
            analysis_id=candidate.analysis_id,
            title=candidate.title,
            family=candidate.family,
            analysis_dir=analysis_dir,
            hypothesis=candidate.hypothesis,
            relevance_reason=candidate.relevance_reason,
            feasibility_reason=candidate.feasibility_reason,
            required_artifacts=candidate.required_artifacts,
            requires_retrain=candidate.requires_retrain,
            implementation_plan=candidate.implementation_plan,
            expected_claim=candidate.expected_claim,
            expected_outputs=candidate.expected_outputs,
        )
        write_json(analysis_dir / "analysis_spec.json", spec.to_dict())

        rubric_contract = build_rubric_contract(candidate.analysis_id, candidate.family)
        write_json(analysis_dir / "rubric_contract.json", rubric_contract)
        write_json(analysis_dir / "target_rubrics.json", build_target_rubrics(rubric_contract))
        write_json(
            analysis_dir / "run_manifest.json",
            _build_run_manifest(
                spec,
                agent,
                agent_model,
                agent_profile,
                max_attempts_per_analysis,
                reasoning_effort,
            ),
        )

        _write_analysis_task_md(task_bundle_dir, spec, max_attempts_per_analysis=max_attempts_per_analysis)

        for staged_name in ("analysis_spec.json", "rubric_contract.json", "target_rubrics.json"):
            shutil.copy2(analysis_dir / staged_name, task_bundle_dir / staged_name)

        specs.append(spec)

    logger.info("Materialized %d analysis workspaces under %s", len(specs), analyses_dir)
    return specs


def _remove_task_bundle_symlinks(task_bundle_dir: Path) -> None:
    for path in sorted(
        task_bundle_dir.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if path.is_symlink():
            logger.warning("Removing symlink from analysis task bundle: %s", path)
            path.unlink()


def _build_run_manifest(
    spec: AnalysisSpec,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None,
    max_attempts_per_analysis: int,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "analysis_id": spec.analysis_id,
        "analysis_family": spec.family,
        "title": spec.title,
        "agent": agent,
        "agent_model": agent_model,
        "agent_profile": agent_profile,
        "reasoning_effort": reasoning_effort,
        "analysis_dir": str(spec.analysis_dir),
        "task_bundle_dir": str(spec.analysis_dir / "task_bundle"),
        "expected_outputs": spec.expected_outputs,
        "requires_retrain": spec.requires_retrain,
        "max_attempts_per_analysis": max_attempts_per_analysis,
        "status": "created",
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "attempt_count": 0,
        "selected_attempt": None,
    }


def _write_analysis_task_md(
    task_bundle_dir: Path,
    spec: AnalysisSpec,
    *,
    max_attempts_per_analysis: int,
) -> None:
    task_md = task_bundle_dir / "task.md"
    original_task_md = task_bundle_dir / "original_task.md"
    if task_md.exists():
        task_md.rename(original_task_md)

    lines: list[str] = [
        f"# Post-Hoc Analysis: {spec.title}\n\n",
        f"- **Analysis ID:** `{spec.analysis_id}`\n",
        f"- **Family:** `{spec.family}`\n",
        f"- **Max attempts:** {max_attempts_per_analysis}\n",
        f"- **Requires retrain:** {spec.requires_retrain}\n",
    ]
    if spec.hypothesis:
        lines.append(f"\n## Hypothesis\n{spec.hypothesis}\n")
    if spec.expected_claim:
        lines.append(f"\n## Expected Claim\n{spec.expected_claim}\n")
    if spec.required_artifacts:
        lines.append(
            "\n## Required Artifacts\n"
            + "\n".join(f"- `{item}`" for item in spec.required_artifacts)
            + "\n"
        )
    if spec.expected_outputs:
        lines.append(
            "\n## Expected Outputs\n"
            + "\n".join(f"- `{item}`" for item in spec.expected_outputs)
            + "\n"
        )
    if spec.implementation_plan:
        lines.append(
            "\n## Implementation Plan\n"
            + "\n".join(f"{index + 1}. {step}" for index, step in enumerate(spec.implementation_plan))
            + "\n"
        )
    lines.append(
        "\n## Execution Rules\n"
        "Use `task_bundle/program_0` as a read-only artifact-ready base. "
        "This post-hoc pipeline is analysis-only: do not retrain, fine-tune, call `evaluate.py`, or create new program variants. "
        "Focus on analysis scripts that inspect existing logs, metrics, saved artifacts, and checkpoints when present. "
        "Repairs happen in the next `attempt_<n>` directory, not by creating a new analysis workspace.\n"
    )
    if original_task_md.exists():
        lines.append(
            "\n## Original Task Context\n"
            "Refer to `original_task.md` for the original search-time task and evaluator constraints.\n"
        )

    task_md.write_text("".join(lines), encoding="utf-8")
