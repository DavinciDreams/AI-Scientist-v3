"""Top-level orchestrator for the standalone post-hoc analysis pipeline."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.investigation.runtime import log_event

from .common import read_json
from .bootstrap import bootstrap_analysis_bundle
from .config import PosthocPipelineConfig
from .execute import execute_analysis_agents
from .extract import extract_bundle, restore_verified_evidence
from .generate import _parse_posthoc_bundle, generate_analysis_candidates
from .materialize import materialize_analysis_workspaces
from .models import (
    AnalysisCandidate,
    AnalysisRunResult,
    AnalysisSelection,
    AnalysisSpec,
    normalize_expected_outputs,
    normalize_implementation_plan,
    normalize_string_list,
)
from .progress import emit_pipeline_summary, stage_event
from .publish import aggregate_and_publish_valid_analyses
from .select import validate_and_select_analyses

logger = logging.getLogger(__name__)

ALL_STAGES = {0, 1, 2, 3, 4, 5, 6}


def run_posthoc_pipeline(
    cfg: PosthocPipelineConfig,
    stages: set[int] | None = None,
) -> int:
    if stages is None:
        stages = set(ALL_STAGES)

    pipeline_started_at = time.monotonic()

    task_dir = Path(cfg.paths.task_dir).expanduser().resolve()
    results_dir = Path(cfg.paths.results_dir).expanduser().resolve()
    evidence_dir = (
        Path(cfg.paths.evidence_dir).expanduser().resolve() if cfg.paths.evidence_dir else None
    )
    output_dir = _resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_dir = output_dir / "generation"
    generation_dir.mkdir(parents=True, exist_ok=True)
    log_event(
        "pipeline_start",
        task=task_dir.name,
        run_id=cfg.run_id,
        phases=sorted(stages),
        stages=sorted(stages),
        agent=cfg.agent.name,
        model=cfg.agent.model,
        output_dir=output_dir,
        show_run_header=cfg.logging.show_run_header,
    )

    agent_kwargs: dict[str, Any] = {
        "agent": cfg.agent.name,
        "agent_model": cfg.agent.model,
        "agent_profile": cfg.agent.profile,
        "reasoning_effort": cfg.agent.reasoning_effort,
        "task_source_dir": task_dir,
        "containerize": cfg.container.enabled,
        "container_image": cfg.container.image,
        "container_network": cfg.container.network,
        "container_auth_root": Path(cfg.container.auth_root).expanduser(),
        "container_env_file": Path(cfg.container.env_file) if cfg.container.env_file else None,
    }

    bundle_dir = output_dir / "_extracted_bundle"
    analysis_ready_bundle_dir = output_dir / "_analysis_ready_bundle"
    bootstrap_report: dict[str, Any] | None = None
    candidates: list[AnalysisCandidate] = []
    selection = AnalysisSelection(accepted=[], rejected=[])
    analysis_specs: list[AnalysisSpec] = []
    results: list[AnalysisRunResult] = []

    if 0 in stages:
        with stage_event(0, mode="run") as stage_fields:
            logger.info("=== Stage 0: Extract bundle ===")
            bundle_dir = extract_bundle(
                task_dir=task_dir,
                results_dir=results_dir,
                evidence_dir=evidence_dir,
                output_dir=output_dir,
            )
            stage_fields["bundle_dir"] = bundle_dir
    else:
        missing_bundle = False
        with stage_event(0, mode="load") as stage_fields:
            logger.info("=== Stage 0: Skipped (using existing bundle at %s) ===", bundle_dir)
            missing_bundle = not bundle_dir.is_dir()
            stage_fields.update(skipped=True, bundle_dir=bundle_dir)
            if missing_bundle:
                logger.error("No extracted bundle found at %s. Run stage 0 first.", bundle_dir)
                stage_fields.update(status="failed", reason="missing_extracted_bundle")
        if missing_bundle:
            emit_pipeline_summary(
                status="failed",
                stages=stages,
                candidates=candidates,
                selection=selection,
                analysis_specs=analysis_specs,
                results=results,
                output_dir=output_dir,
                started_at=pipeline_started_at,
            )
            return 1

    if 1 in stages:
        with stage_event(1, mode="run") as stage_fields:
            logger.info("=== Stage 1: Bootstrap analysis-ready bundle ===")
            bootstrap_report = bootstrap_analysis_bundle(
                bundle_dir=bundle_dir,
                output_bundle_dir=analysis_ready_bundle_dir,
                timeout_seconds=cfg.execution.bootstrap_timeout_seconds,
                timeout_grace_seconds=cfg.execution.analysis_timeout_grace_seconds,
                max_attempts=cfg.execution.bootstrap_max_attempts,
                primary_metric_key=cfg.execution.primary_metric_key,
                primary_metric_tolerance=cfg.execution.primary_metric_tolerance,
                retrain_if_checkpoint_missing=cfg.execution.retrain_if_checkpoint_missing,
                checkpoint_mode=cfg.execution.checkpoint_mode,
                **agent_kwargs,
            )
            logger.info("Bootstrap report: %s", json.dumps(bootstrap_report, indent=2))
            stage_fields.update(
                bootstrap_ready=bootstrap_report.get("bootstrap_ready"),
                checkpoint_mode=cfg.execution.checkpoint_mode,
            )
    else:
        missing_analysis_bundle = False
        with stage_event(1, mode="load") as stage_fields:
            logger.info(
                "=== Stage 1: Skipped (using existing analysis-ready bundle at %s) ===",
                analysis_ready_bundle_dir,
            )
            missing_analysis_bundle = not analysis_ready_bundle_dir.is_dir()
            if missing_analysis_bundle:
                logger.error(
                    "No analysis-ready bundle found at %s. Run stage 1 first.",
                    analysis_ready_bundle_dir,
                )
                stage_fields.update(
                    status="failed",
                    reason="missing_analysis_ready_bundle",
                    bundle_dir=analysis_ready_bundle_dir,
                )
            else:
                bootstrap_report = read_json(output_dir / "bootstrap_report.json")
                stage_fields.update(
                    skipped=True,
                    bootstrap_ready=(
                        bootstrap_report.get("bootstrap_ready")
                        if isinstance(bootstrap_report, dict)
                        else None
                    ),
                )
        if missing_analysis_bundle:
            emit_pipeline_summary(
                status="failed",
                stages=stages,
                candidates=candidates,
                selection=selection,
                analysis_specs=analysis_specs,
                results=results,
                output_dir=output_dir,
                started_at=pipeline_started_at,
            )
            return 1

    restore_verified_evidence(bundle_dir, analysis_ready_bundle_dir)

    if 2 in stages:
        with stage_event(2, mode="run") as stage_fields:
            logger.info("=== Stage 2: Generate analysis candidates ===")
            candidates = generate_analysis_candidates(
                bundle_dir=analysis_ready_bundle_dir,
                generation_dir=generation_dir,
                timeout_seconds=cfg.execution.generation_timeout_seconds,
                timeout_grace_seconds=cfg.execution.analysis_timeout_grace_seconds,
                **agent_kwargs,
            )
            logger.info("Generated %d analysis candidates", len(candidates))
            stage_fields["candidate_count"] = len(candidates)
    else:
        with stage_event(2, mode="load") as stage_fields:
            candidates = _load_candidates_from_disk(generation_dir)
            logger.info("Loaded %d analysis candidates from disk", len(candidates))
            stage_fields.update(skipped=True, candidate_count=len(candidates))

    if 3 in stages:
        with stage_event(3, mode="run") as stage_fields:
            logger.info("=== Stage 3: Select analysis candidates ===")
            selection = validate_and_select_analyses(
                candidates=candidates,
                bundle_dir=analysis_ready_bundle_dir,
                generation_dir=generation_dir,
                timeout_seconds=cfg.execution.selection_timeout_seconds,
                timeout_grace_seconds=cfg.execution.analysis_timeout_grace_seconds,
                max_selected_analyses=cfg.execution.max_selected_analyses,
                **agent_kwargs,
            )
            logger.info(
                "Selected %d analyses (%d rejected)",
                len(selection.accepted),
                len(selection.rejected),
            )
            stage_fields.update(
                selected_count=len(selection.accepted),
                rejected_count=len(selection.rejected),
            )
    else:
        with stage_event(3, mode="load") as stage_fields:
            selection = _load_selection_from_disk(generation_dir, candidates)
            logger.info("Loaded selection from disk: %d accepted", len(selection.accepted))
            stage_fields.update(
                skipped=True,
                selected_count=len(selection.accepted),
                rejected_count=len(selection.rejected),
            )

    if 4 in stages:
        with stage_event(4, mode="run") as stage_fields:
            logger.info("=== Stage 4: Materialize analysis workspaces ===")
            analysis_specs = materialize_analysis_workspaces(
                analysis_ready_bundle_dir=analysis_ready_bundle_dir,
                posthoc_run_dir=output_dir,
                selection=selection,
                agent=cfg.agent.name,
                agent_model=cfg.agent.model,
                agent_profile=cfg.agent.profile,
                reasoning_effort=cfg.agent.reasoning_effort,
                max_attempts_per_analysis=cfg.execution.max_attempts_per_analysis,
            )
            logger.info("Materialized %d analysis workspaces", len(analysis_specs))
            stage_fields["analysis_count"] = len(analysis_specs)
    else:
        with stage_event(4, mode="load") as stage_fields:
            analysis_specs = _load_analysis_specs_from_disk(
                output_dir,
                accepted_analysis_ids=[candidate.analysis_id for candidate in selection.accepted],
            )
            logger.info("Loaded %d analysis specs from disk", len(analysis_specs))
            stage_fields.update(skipped=True, analysis_count=len(analysis_specs))

    if 5 in stages:
        with stage_event(5, mode="run") as stage_fields:
            logger.info("=== Stage 5: Execute analysis agents ===")
            results = execute_analysis_agents(
                analysis_specs=analysis_specs,
                posthoc_run_dir=output_dir,
                timeout_seconds=cfg.execution.analysis_timeout_seconds,
                timeout_grace_seconds=cfg.execution.analysis_timeout_grace_seconds,
                max_attempts_per_analysis=cfg.execution.max_attempts_per_analysis,
                checkpoint_mode=cfg.execution.checkpoint_mode,
                **agent_kwargs,
            )
            logger.info("Executed %d analyses", len(results))
            stage_fields.update(
                analysis_count=len(results),
                completed_count=sum(result.status == "completed" for result in results),
                rubric_passed_count=sum(result.rubric_passed for result in results),
            )
    else:
        with stage_event(5, mode="load") as stage_fields:
            results = _load_results_from_disk(
                output_dir,
                accepted_analysis_ids=[candidate.analysis_id for candidate in selection.accepted],
            )
            logger.info("Loaded %d analysis results from disk", len(results))
            stage_fields.update(
                skipped=True,
                analysis_count=len(results),
                completed_count=sum(result.status == "completed" for result in results),
                rubric_passed_count=sum(result.rubric_passed for result in results),
            )

    ranking: dict[str, Any] | None = None
    if 6 in stages:
        with stage_event(6, mode="run") as stage_fields:
            logger.info("=== Stage 6: Aggregate and Publish Valid Analyses ===")
            ranking = aggregate_and_publish_valid_analyses(
                output_dir=output_dir,
                analysis_specs=analysis_specs,
                results=results,
                checkpoint_mode=cfg.execution.checkpoint_mode,
            )
            stage_fields["ranking_valid"] = (
                ranking.get("valid") if isinstance(ranking, dict) else None
            )
    else:
        with stage_event(6, mode="skip") as stage_fields:
            logger.info("=== Stage 6: Skipped ===")
            stage_fields["skipped"] = True

    exit_code = 0
    if 6 in stages and ranking is not None:
        if (
            not ranking.get("valid")
            and bootstrap_report is not None
            and bootstrap_report.get("bootstrap_ready") is False
        ):
            logger.error(
                "Post-hoc pipeline produced no valid analyses and bootstrap was not ready: %s",
                bootstrap_report.get("failure_reason"),
            )
            exit_code = 1

    if exit_code == 0:
        logger.info("Post-hoc pipeline finished (stages: %s)", sorted(stages))
    else:
        logger.error("Post-hoc pipeline failed (stages: %s)", sorted(stages))
    emit_pipeline_summary(
        status="completed" if exit_code == 0 else "failed",
        stages=stages,
        candidates=candidates,
        selection=selection,
        analysis_specs=analysis_specs,
        results=results,
        output_dir=output_dir,
        started_at=pipeline_started_at,
    )
    return exit_code


def _load_candidates_from_disk(generation_dir: Path) -> list[AnalysisCandidate]:
    bundle_path = generation_dir / "posthoc_bundle.json"
    if not bundle_path.exists():
        return []
    raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return []
    _, candidates = _parse_posthoc_bundle(raw)
    return candidates


def _load_selection_from_disk(
    generation_dir: Path,
    candidates: list[AnalysisCandidate],
) -> AnalysisSelection:
    selection_path = generation_dir / "posthoc_selection.json"
    if not selection_path.exists():
        return AnalysisSelection(accepted=candidates, rejected=[])

    raw = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return AnalysisSelection(accepted=candidates, rejected=[])

    def _entry_id(entry: object) -> str:
        if not isinstance(entry, dict):
            return ""
        candidate_id = entry.get("id", entry.get("analysis_id"))
        return candidate_id if isinstance(candidate_id, str) else ""

    accepted_ids: set[str] = set()
    for entry in raw.get("accepted", []):
        entry_id = _entry_id(entry)
        if entry_id:
            accepted_ids.add(entry_id)

    rejected_map: dict[str, str] = {}
    for entry in raw.get("rejected", []):
        if not isinstance(entry, dict):
            continue
        entry_id = _entry_id(entry)
        if entry_id:
            rejected_map[entry_id] = str(entry.get("reason", ""))
    accepted = [candidate for candidate in candidates if candidate.analysis_id in accepted_ids]
    rejected = [
        (candidate, rejected_map[candidate.analysis_id])
        for candidate in candidates
        if candidate.analysis_id in rejected_map
    ]
    return AnalysisSelection(accepted=accepted, rejected=rejected)


def _load_analysis_specs_from_disk(
    output_dir: Path,
    accepted_analysis_ids: list[str] | None = None,
) -> list[AnalysisSpec]:
    analyses_dir = output_dir / "analyses"
    if not analyses_dir.is_dir():
        return []

    specs_by_id: dict[str, AnalysisSpec] = {}
    for analysis_dir in sorted(analyses_dir.iterdir()):
        if not analysis_dir.is_dir():
            continue
        spec_path = analysis_dir / "analysis_spec.json"
        if not spec_path.exists():
            continue
        try:
            data = json.loads(spec_path.read_text(encoding="utf-8"))
            spec = AnalysisSpec(
                analysis_id=data["analysis_id"],
                title=data.get("title", ""),
                family=data.get("family", ""),
                analysis_dir=analysis_dir,
                hypothesis=data.get("hypothesis", ""),
                relevance_reason=data.get("relevance_reason", ""),
                feasibility_reason=data.get("feasibility_reason", ""),
                required_artifacts=normalize_string_list(data.get("required_artifacts", [])),
                requires_retrain=bool(data.get("requires_retrain", False)),
                implementation_plan=normalize_implementation_plan(
                    data.get("implementation_plan", [])
                ),
                expected_claim=data.get("expected_claim", ""),
                expected_outputs=normalize_expected_outputs(data.get("expected_outputs", [])),
            )
            specs_by_id[spec.analysis_id] = spec
        except (json.JSONDecodeError, KeyError):
            continue
    if accepted_analysis_ids is None:
        return [specs_by_id[analysis_id] for analysis_id in sorted(specs_by_id)]
    return [
        specs_by_id[analysis_id]
        for analysis_id in accepted_analysis_ids
        if analysis_id in specs_by_id
    ]


def _load_results_from_disk(
    output_dir: Path,
    accepted_analysis_ids: list[str] | None = None,
) -> list[AnalysisRunResult]:
    analyses_dir = output_dir / "analyses"
    if not analyses_dir.is_dir():
        return []

    results_by_id: dict[str, AnalysisRunResult] = {}
    for analysis_dir in sorted(analyses_dir.iterdir()):
        result_path = analysis_dir / "analysis_result.json"
        if not result_path.exists():
            continue
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        attempts = [attempt for attempt in raw.get("attempts", []) if isinstance(attempt, dict)]
        from .models import AttemptInspection

        result = AnalysisRunResult(
            analysis_id=raw["analysis_id"],
            status=raw.get("status", "failed"),
            attempt_count=int(raw.get("attempt_count", 0)),
            exit_code=int(raw.get("exit_code", 1)),
            selected_attempt=raw.get("selected_attempt"),
            rubric_passed=bool(raw.get("rubric_passed", False)),
            required_outputs_present=bool(raw.get("required_outputs_present", False)),
            guardrail_passed=raw.get("guardrail_passed"),
            output=str(raw.get("output", "")),
            produced_files=dict(raw.get("produced_files", {})),
            attempts=[
                AttemptInspection(
                    attempt_dir=Path(attempt["attempt_dir"]),
                    attempt_id=attempt["attempt_id"],
                    overall_pass=bool(attempt.get("overall_pass", False)),
                    rubric_error=attempt.get("rubric_error"),
                    required_outputs_present=bool(attempt.get("required_outputs_present", False)),
                    missing_outputs=list(attempt.get("missing_outputs", [])),
                    manifest_valid=bool(attempt.get("manifest_valid", False)),
                    manifest_error=attempt.get("manifest_error"),
                    guardrail_passed=attempt.get("guardrail_passed"),
                    checkpoint_validated=bool(attempt.get("checkpoint_validated", False)),
                    manifest=attempt.get("manifest"),
                    rubric_status=attempt.get("rubric_status"),
                )
                for attempt in attempts
            ],
        )
        results_by_id[result.analysis_id] = result
    if accepted_analysis_ids is None:
        return [results_by_id[analysis_id] for analysis_id in sorted(results_by_id)]
    return [
        results_by_id[analysis_id]
        for analysis_id in accepted_analysis_ids
        if analysis_id in results_by_id
    ]


def _resolve_output_dir(cfg: PosthocPipelineConfig) -> Path:
    if cfg.paths.output_dir:
        return Path(cfg.paths.output_dir).expanduser().resolve()
    task_name = Path(cfg.paths.task_dir).name
    return (
        Path(cfg.paths.results_dir).expanduser().resolve()
        / "posthoc"
        / f"{task_name}_{cfg.agent.name}_{cfg.run_id}"
    )
