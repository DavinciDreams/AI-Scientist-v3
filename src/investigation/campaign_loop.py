"""Campaign loop that drives idea generation and run attempts."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.orchestrator import rubrics, state as state_store

logger = logging.getLogger(__name__)
from .agent_selection import resolve_agent_model, resolve_cycle_agent
from .config import InvestigationConfig
from .files import confirm_prune_warning
from .idea_generation import IdeaArtifactsError, load_idea_artifacts
from .idea_retry_feedback import sanitize_validation_feedback, validation_feedback_from_output
from .models import IdeaGenerationResult, PaperGenerationResult, RunAttemptResult
from .script_runners import copy_output_log_no_follow, write_output_log_no_follow
from .task_failure import (
    AGENT_LOG_FAILURE, INFRASTRUCTURE_EXIT_CODE, TASK_CONTRACT_FAILURE, failure_details,
)


BestProgramCandidate = tuple[
    Path, str, float, bool
]  # (program_dir, entrypoint_rel, score, correct)


def _correct_incumbent_score(
    best_manifest: dict[str, Any],
    *,
    harness_evaluated_attempts: bool,
) -> float | None:
    """Return a score gate only for a scientifically correct incumbent."""
    score = best_manifest.get("combined_score")
    if (
        not harness_evaluated_attempts
        or best_manifest.get("correct") is not True
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        return None
    return float(score)


def run_cycles_and_finalize(
    *,
    cfg: InvestigationConfig,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path,
    shared_memory_dir: Path | None = None,
    conclusions_dir: Path | None = None,
    worker_index: int = 0,
    instruct_path: Path,
    idea_instruct_path: Path,
    paper_instruct_path: Path,
    paper_contributions_path: Path | None,
    containerize: bool,
    container_image: str,
    container_network: str,
    container_auth_root: Path,
    container_env_file: Path | None,
    container_session_name: str | None,
    container_mount_root: Path | None,
    outer_state: dict[str, Any],
    best_manifest: dict[str, Any],
    best_program_dir: Path,
    start_cycle: int,
    max_program_evals_per_attempt: int,
    main_started_at: float,
    log_event_fn: Callable[..., None],
    log_step_start_fn: Callable[..., float],
    log_step_end_fn: Callable[..., None],
    format_elapsed_fn: Callable[[float], str],
    run_idea_generation_fn: Callable[..., IdeaGenerationResult],
    run_single_attempt_fn: Callable[..., RunAttemptResult],
    run_paper_generation_fn: Callable[..., PaperGenerationResult],
    reconcile_legacy_idea_local_memory_fn: Callable[..., None],
    find_best_program_in_experiment_fn: Callable[[Path], BestProgramCandidate | None],
    read_json_fn: Callable[[Path], dict[str, Any] | None],
    utc_now_fn: Callable[[], str],
    checkpoint_fn: Callable[..., None] = lambda **_kwargs: None,
) -> int:
    """Execute campaign cycles and finalize optional paper output."""
    agent_model = resolve_agent_model(cfg.agent, cfg.agent.name)
    agent_profile = (
        str(cfg.agent.profile).strip()
        if isinstance(cfg.agent.profile, str) and cfg.agent.profile.strip()
        else None
    )
    reasoning_effort = cfg.agent.reasoning_effort
    finalize_on_exit = cfg.versioning.finalize_on_exit
    if conclusions_dir is None:
        conclusions_dir = global_memory_dir
    state_store.ensure_conclusions_files(conclusions_dir)
    prune_n = cfg.execution.prune_every_n_cycles
    if prune_n > 0 and not containerize:
        log_event_fn(
            "prune_warning",
            message=(
                "Pruning is enabled but the agent is NOT containerized. "
                "The agent will still have access to archived history and "
                "idea files from previous chapters, so pruning may be "
                "ineffective at resetting context."
            ),
        )
        confirm_prune_warning()

    for cycle_idx in range(start_cycle, cfg.execution.max_idea_cycles + 1):
        # Parallel workers number cycles locally; events also carry the
        # campaign-wide position so progress output reads "Cycle 2/3".
        cycle_position = {
            "global_cycle": cfg.execution.cycle_offset + cycle_idx,
            "total_cycles": cfg.execution.total_cycles or cfg.execution.max_idea_cycles,
        }
        assignments_before = outer_state.get("agent_cycle_assignments")
        selected_before = (
            assignments_before.get(str(cycle_idx)) if isinstance(assignments_before, dict) else None
        )
        cycle_agent = resolve_cycle_agent(cfg, outer_state, cycle_index=cycle_idx)
        assignments_after = outer_state.get("agent_cycle_assignments")
        selected_after = (
            assignments_after.get(str(cycle_idx)) if isinstance(assignments_after, dict) else None
        )
        if selected_after != selected_before:
            state_store.save_outer_state(results_dir, outer_state)

        # -- Idea pruning: reset context at chapter boundaries --
        if prune_n > 0 and cycle_idx > start_cycle and (cycle_idx - 1) % prune_n == 0:
            chapter = (cycle_idx - 1) // prune_n
            log_event_fn(
                "prune_start",
                cycle=cycle_idx,
                chapter=chapter,
                prune_every_n_cycles=prune_n,
            )
            state_store.archive_for_prune(results_dir, global_memory_dir, chapter)
            outer_state["prune_chapter"] = chapter + 1
            outer_state["prune_every_n_cycles"] = prune_n
            state_store.save_outer_state(results_dir, outer_state)
            log_event_fn(
                "prune_complete",
                cycle=cycle_idx,
                new_chapter=chapter + 1,
                archive_dir=str(results_dir / "_pruned" / f"chapter_{chapter:03d}"),
            )

        # -- Shared memory sync: pull shared → local before cycle --
        memory_pre_state: dict[str, int] | None = None
        conclusions_pre_state: state_store.ConclusionsSyncState | None = None
        if shared_memory_dir is not None:
            sync_memory_files = shared_memory_dir.resolve() != global_memory_dir.resolve()
            if sync_memory_files:
                memory_pre_state = state_store.sync_memory_before_cycle(
                    shared_memory_dir, global_memory_dir
                )
                logger.info(
                    "Synced shared memory → local before cycle %d (jsonl_lines=%d, md_size=%d)",
                    cycle_idx,
                    memory_pre_state["jsonl_lines"],
                    memory_pre_state["md_size"],
                )
            conclusions_pre_state = state_store.sync_conclusions_before_cycle(
                shared_memory_dir, conclusions_dir, worker_index
            )
            logger.info(
                "Synced shared conclusions → local before cycle %d (worker=%d, own_size=%d)",
                cycle_idx,
                worker_index,
                conclusions_pre_state["own_conclusions_size"],
            )

        cycle_started_at = time.monotonic()
        log_event_fn(
            "cycle_start",
            cycle=cycle_idx,
            **cycle_position,
            start_ts_utc=utc_now_fn(),
            agent=cycle_agent.agent,
            agent_model=cycle_agent.model,
            reasoning_effort=reasoning_effort,
        )
        idea_bundle: dict[str, Any] | None = None
        idea_result: IdeaGenerationResult | None = None
        idea_workspace: Path | None = None
        validation_feedback = ""
        for generation_attempt in range(1, cfg.execution.max_idea_generation_attempts + 1):
            idea_staging_dir = (
                results_dir
                / "_idea_generation"
                / f"cycle_{cycle_idx:04d}_attempt_{generation_attempt:02d}_{time.time_ns()}"
            )
            idea_generation_started_at = log_step_start_fn(
                "idea_generation",
                cycle=cycle_idx,
                **cycle_position,
                generation_attempt=generation_attempt,
                output_dir=idea_staging_dir,
                agent=cycle_agent.agent,
                agent_model=cycle_agent.model,
                reasoning_effort=reasoning_effort,
            )
            try:
                idea_result = run_idea_generation_fn(
                    agent=cycle_agent.agent,
                    agent_model=cycle_agent.model,
                    agent_profile=agent_profile if cycle_agent.agent == "codex" else None,
                    reasoning_effort=reasoning_effort,
                    task_dir=task_dir,
                    results_dir=results_dir,
                    global_memory_dir=global_memory_dir,
                    conclusions_dir=conclusions_dir,
                    worker_index=worker_index,
                    cycle_index=cycle_idx,
                    output_dir=idea_staging_dir,
                    instruct_path=idea_instruct_path,
                    timeout_seconds=cfg.execution.idea_timeout_seconds,
                    timeout_grace_seconds=cfg.execution.run_timeout_grace_seconds,
                    harness_evaluated_attempts=cfg.execution.harness_evaluated_attempts,
                    containerize=containerize,
                    container_image=container_image,
                    container_network=container_network,
                    container_auth_root=container_auth_root,
                    container_env_file=container_env_file,
                    container_session_name=container_session_name,
                    container_mount_root=container_mount_root,
                    **({"validation_feedback": validation_feedback} if validation_feedback else {}),
                )
            except OSError as exc:
                idea_result = IdeaGenerationResult(
                    output=f"idea generation launch failed: {exc}",
                    exit_code=1,
                    idea_dir=idea_staging_dir,
                )
            if idea_result.infrastructure_failure_category == AGENT_LOG_FAILURE:
                failure_record = {
                    "cycle_index": cycle_idx, "generation_attempt": generation_attempt,
                    "exit_code": INFRASTRUCTURE_EXIT_CODE,
                    "failure_category": AGENT_LOG_FAILURE, "failure_scope": "infrastructure",
                }
                outer_state.update(failure_category=AGENT_LOG_FAILURE, failure_scope="infrastructure")
                try:
                    state_store.append_jsonl(
                        state_store.idea_generation_failures_path(results_dir), failure_record,
                    )
                    state_store.save_outer_state(results_dir, outer_state)
                except OSError:
                    pass
                try:
                    log_event_fn("campaign_failed", **failure_record)
                    checkpoint_fn(
                        label=AGENT_LOG_FAILURE, phase="investigation", event="campaign_failed",
                        cycle=cycle_idx, final=finalize_on_exit,
                    )
                except Exception:
                    pass
                return INFRASTRUCTURE_EXIT_CODE
            idea_workspace = idea_result.idea_dir or idea_staging_dir
            idea_log_path = idea_workspace / "wrapper_output.log"
            if not idea_log_path.is_file():
                write_output_log_no_follow(idea_result.output, idea_log_path)
            log_step_end_fn(
                "idea_generation",
                idea_generation_started_at,
                cycle=cycle_idx,
                generation_attempt=generation_attempt,
                exit=idea_result.exit_code,
                idea_workspace=idea_workspace,
                log_path=idea_log_path,
            )

            failure_reason = ""
            failure_detail = ""
            failure_path: Path | None = None
            validation_feedback = ""
            if idea_result.exit_code != 0:
                failure_reason = "idea_generation_exit_nonzero"
                validation_feedback = validation_feedback_from_output(idea_result.output)
            else:
                harness_incumbent_score = _correct_incumbent_score(
                    best_manifest,
                    harness_evaluated_attempts=cfg.execution.harness_evaluated_attempts,
                )
                try:
                    artifacts = load_idea_artifacts(
                        idea_workspace,
                        cycle_idx,
                        read_json_fn,
                        require_harness_verification=cfg.execution.harness_evaluated_attempts,
                        harness_incumbent_score=harness_incumbent_score,
                    )
                except IdeaArtifactsError as exc:
                    failure_reason = exc.reason
                    failure_detail = exc.detail
                    failure_path = exc.path
                    validation_feedback = sanitize_validation_feedback(str(exc))
                else:
                    idea_bundle = artifacts.bundle
                    break

            failure_record = {
                "timestamp": utc_now_fn(),
                "cycle_index": cycle_idx,
                "generation_attempt": generation_attempt,
                "max_attempts": cfg.execution.max_idea_generation_attempts,
                "agent": cycle_agent.agent,
                "agent_model": cycle_agent.model,
                "reasoning_effort": reasoning_effort,
                "exit_code": idea_result.exit_code,
                "reason": failure_reason,
                "detail": failure_detail or None,
                "path": str(failure_path) if failure_path else None,
                "workspace": str(idea_workspace),
                "log_path": str(idea_log_path),
            }
            state_store.append_jsonl(
                state_store.idea_generation_failures_path(results_dir),
                failure_record,
            )
            log_event_fn(
                "idea_generation_failed",
                cycle=cycle_idx,
                generation_attempt=generation_attempt,
                max_attempts=cfg.execution.max_idea_generation_attempts,
                exit=idea_result.exit_code,
                reason=failure_reason,
                error=failure_detail or None,
                path=failure_path,
                log_path=idea_log_path,
                output_tail=idea_result.output[-1200:] or None,
            )
            if generation_attempt < cfg.execution.max_idea_generation_attempts:
                log_event_fn(
                    "idea_generation_retry_scheduled",
                    cycle=cycle_idx,
                    next_generation_attempt=generation_attempt + 1,
                )

        if idea_bundle is None or idea_result is None or idea_workspace is None:
            failure_streak = int(outer_state.get("failure_streak", 0)) + 1
            outer_state["current_cycle"] = cycle_idx - 1
            outer_state["failure_streak"] = failure_streak
            state_store.save_outer_state(results_dir, outer_state)
            log_event_fn(
                "idea_generation_skipped",
                cycle=cycle_idx,
                attempts_used=cfg.execution.max_idea_generation_attempts,
                failure_streak=failure_streak,
            )
            log_event_fn(
                "cycle_end",
                cycle=cycle_idx,
                **cycle_position,
                idea_id=None,
                idea_success=False,
                skipped=True,
                attempts_used=0,
                idea_generation_attempts=cfg.execution.max_idea_generation_attempts,
                failure_streak=failure_streak,
                elapsed=format_elapsed_fn(time.monotonic() - cycle_started_at),
            )
            if shared_memory_dir is not None:
                if memory_pre_state is not None:
                    state_store.sync_memory_after_cycle(
                        shared_memory_dir, global_memory_dir, memory_pre_state
                    )
                if conclusions_pre_state is not None:
                    state_store.sync_conclusions_after_cycle(
                        shared_memory_dir,
                        conclusions_dir,
                        worker_index,
                        conclusions_pre_state,
                    )
            log_event_fn(
                "campaign_failed",
                reason="idea_generation_attempts_exhausted",
                cycle=cycle_idx,
                failure_streak=failure_streak,
            )
            checkpoint_fn(
                label="final" if finalize_on_exit else f"campaign failed c{cycle_idx:03d}",
                phase="investigation",
                event="campaign_failed",
                cycle=cycle_idx,
                final=finalize_on_exit,
            )
            return 2

        idea_id = str(idea_bundle["idea_id"])
        idea_title = idea_bundle.get("title") if isinstance(idea_bundle.get("title"), str) else None
        target_ids = rubrics.extract_target_rubric_ids(idea_bundle["target_rubrics"])
        idea_dir_name = state_store.idea_dir_name(
            cycle_index=cycle_idx,
            idea_title=idea_title,
            idea_id=idea_id,
        )
        idea_dir = state_store.idea_dir(results_dir, idea_dir_name)
        idea_dir.mkdir(parents=True, exist_ok=True)
        reconcile_legacy_idea_local_memory_fn(
            results_dir=results_dir,
            cycle_index=cycle_idx,
            idea_id=idea_id,
            canonical_idea_dir_name=idea_dir_name,
        )
        idea_file = idea_dir / "idea_bundle.json"
        rubrics_file = idea_dir / "target_rubrics.json"
        state_store.write_json(idea_file, idea_bundle)
        state_store.write_json(rubrics_file, idea_bundle["target_rubrics"])
        canonical_idea_log_path = idea_dir / "idea_generation_wrapper_output.log"
        if idea_log_path.is_file():
            copy_output_log_no_follow(idea_log_path, canonical_idea_log_path)
        else:
            write_output_log_no_follow(idea_result.output, canonical_idea_log_path)

        state_store.append_jsonl(
            state_store.ideas_history_path(results_dir),
            {
                "timestamp": utc_now_fn(),
                "cycle_index": cycle_idx,
                "idea_id": idea_id,
                "idea_dir_name": idea_dir_name,
                "title": idea_bundle.get("title"),
                "target_rubrics": idea_bundle.get("target_rubrics", []),
                "agent": cycle_agent.agent,
                "agent_model": cycle_agent.model,
                "reasoning_effort": reasoning_effort,
            },
        )

        outer_state["current_cycle"] = cycle_idx
        outer_state["last_idea_id"] = idea_id
        outer_state["failure_streak"] = 0
        state_store.save_outer_state(results_dir, outer_state)
        checkpoint_fn(
            label=f"idea c{cycle_idx:03d} {idea_id}",
            phase="investigation",
            event="idea_drafted",
            cycle=cycle_idx,
            idea_id=idea_id,
            agent=cycle_agent.agent,
        )

        log_event_fn(
            "idea_bundle_loaded",
            cycle=cycle_idx,
            idea_id=idea_id,
            idea_title=idea_title,
            idea_dir=idea_dir,
            target_rubric_count=len(target_ids),
            target_rubric_ids=target_ids,
        )

        attempt_number = 1
        attempt_id = f"{idea_id}_attempt_{attempt_number:02d}"
        attempt_started_at = log_step_start_fn(
            "attempt",
            cycle=cycle_idx,
            **cycle_position,
            idea_id=idea_id,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            seed_program_dir=best_program_dir,
            agent=cycle_agent.agent,
            agent_model=cycle_agent.model,
        )

        attempt_result = run_single_attempt_fn(
            agent=cycle_agent.agent,
            agent_model=cycle_agent.model,
            agent_profile=agent_profile if cycle_agent.agent == "codex" else None,
            reasoning_effort=reasoning_effort,
            task_dir=task_dir,
            results_dir=results_dir,
            global_memory_dir=global_memory_dir,
            conclusions_dir=conclusions_dir,
            worker_index=worker_index,
            seed_program_dir=best_program_dir,
            idea_file=idea_file,
            rubrics_file=rubrics_file,
            attempt_id=attempt_id,
            idea_dir_name=idea_dir_name,
            instruct_path=instruct_path,
            timeout_seconds=cfg.execution.run_timeout_seconds,
            timeout_grace_seconds=cfg.execution.run_timeout_grace_seconds,
            max_program_evals_per_attempt=max_program_evals_per_attempt,
            harness_evaluated_attempts=cfg.execution.harness_evaluated_attempts,
            gpu_evaluation=cfg.execution.gpu_evaluation,
            harness_evaluation_timeout_seconds=cfg.execution.harness_evaluation_timeout_seconds,
            containerize=containerize,
            container_image=container_image,
            container_network=container_network,
            container_auth_root=container_auth_root,
            container_env_file=container_env_file,
            container_session_name=container_session_name,
            container_mount_root=container_mount_root,
        )

        output_tail = attempt_result.output[-1200:] if attempt_result.output else ""
        attempt_workspace = (
            attempt_result.experiment_dir
            if attempt_result.experiment_dir is not None
            else state_store.idea_dir(results_dir, idea_dir_name)
        )
        attempt_log_path = attempt_workspace / "wrapper_output.log"
        failure_category = (
            TASK_CONTRACT_FAILURE
            if attempt_result.task_failure_category == TASK_CONTRACT_FAILURE
            else AGENT_LOG_FAILURE
            if attempt_result.infrastructure_failure_category == AGENT_LOG_FAILURE
            else None
        )
        if failure_category is not None:
            failure_code, failure_scope = failure_details(failure_category)
            failure = {
                "timestamp": utc_now_fn(), "cycle_index": cycle_idx,
                "idea_id": idea_id, "attempt_id": attempt_id,
                "attempt_number": attempt_number, "attempt_status": "fail",
                "run_exit_code": failure_code,
                "failure_category": failure_category, "failure_scope": failure_scope,
                "best_updated": False, "wrapper_output_log": str(attempt_log_path),
            }
            outer_state.update(failure_category=failure_category, failure_scope=failure_scope)
            # A secondary filesystem/checkpoint error must not erase trusted terminal status.
            try:
                state_store.append_jsonl(state_store.attempts_history_path(results_dir), failure)
                state_store.save_outer_state(results_dir, outer_state)
                state_store.write_private_text(attempt_log_path, attempt_result.output)
            except OSError:
                pass
            try:
                log_step_end_fn("attempt", attempt_started_at, **failure)
                log_event_fn("campaign_failed", **failure)
                checkpoint_fn(
                    label=failure_category, phase="investigation",
                    event="campaign_failed", cycle=cycle_idx, idea_id=idea_id,
                    final=finalize_on_exit,
                )
            except Exception:
                pass
            return failure_code

        attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
        state_store.write_private_text(attempt_log_path, attempt_result.output)

        status_data: dict[str, Any] | None = None
        validation = rubrics.RubricStatusValidation(
            valid=False,
            overall_pass=False,
            error="no experiment directory found",
            status=None,
        )
        if attempt_result.experiment_dir is not None:
            status_data = rubrics.load_json_dict(
                attempt_result.experiment_dir / "rubric_status.json"
            )
            validation = rubrics.validate_rubric_status(
                status_data,
                target_ids,
                expected_attempt_id=attempt_id,
                expected_idea_id=idea_id,
            )

        best_updated = False
        previous_best_score: float | None = None
        new_best_score: float | None = None
        current_best = best_manifest.get("combined_score")
        best_score_before = (
            float(current_best)
            if best_manifest.get("correct") is True
            and isinstance(current_best, (int, float))
            and not isinstance(current_best, bool)
            else None
        )
        evaluated_score: float | None = None
        evaluated_correct: bool | None = None
        if attempt_result.experiment_dir is not None:
            candidate = find_best_program_in_experiment_fn(attempt_result.experiment_dir)
            if candidate is not None:
                candidate_dir, candidate_entrypoint, candidate_score, candidate_correct = candidate
                evaluated_score = candidate_score
                evaluated_correct = candidate_correct
                if candidate_correct and (
                    best_score_before is None or candidate_score > best_score_before
                ):
                    previous_best_score = best_score_before
                    new_best_score = candidate_score
                    best_program_dir = candidate_dir
                    best_manifest = {
                        "program_dir": str(candidate_dir),
                        "entrypoint_path": candidate_entrypoint,
                        "combined_score": candidate_score,
                        "correct": True,
                        "source_attempt_id": attempt_id,
                        "updated_at": utc_now_fn(),
                    }
                    state_store.save_best_manifest(results_dir, best_manifest)
                    outer_state["best_program_dir"] = str(candidate_dir)
                    outer_state["best_combined_score"] = candidate_score
                    outer_state["best_correct"] = True
                    best_updated = True

        run_failed = attempt_result.exit_code != 0
        attempt_pass = (not run_failed) and validation.valid and validation.overall_pass
        idea_success = attempt_pass
        attempt_status = "pass" if attempt_pass else "fail"
        outer_state["failure_streak"] = 0

        attempt_record = {
            "timestamp": utc_now_fn(),
            "cycle_index": cycle_idx,
            "idea_id": idea_id,
            "idea_dir_name": idea_dir_name,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "run_exit_code": attempt_result.exit_code,
            "attempt_status": attempt_status,
            "experiment_dir": (
                str(attempt_result.experiment_dir) if attempt_result.experiment_dir else None
            ),
            "wrapper_output_log": str(attempt_log_path),
            "rubric_status_valid": validation.valid,
            "rubric_overall_pass": validation.overall_pass,
            "rubric_error": validation.error,
            "failure_streak_after": 0,
            "best_updated": best_updated,
            "output_tail": output_tail,
            "agent": cycle_agent.agent,
            "agent_model": cycle_agent.model,
            "reasoning_effort": reasoning_effort,
        }
        metrics: dict[str, Any] | None = None
        if isinstance(status_data, dict):
            metrics = status_data.get("metrics")
            if isinstance(metrics, dict):
                attempt_record["metrics"] = metrics

        state_store.append_jsonl(state_store.attempts_history_path(results_dir), attempt_record)
        state_store.save_outer_state(results_dir, outer_state)
        checkpoint_fn(
            label=f"evaluation c{cycle_idx:03d} a{attempt_number:02d}",
            phase="investigation",
            event="evaluation_complete",
            cycle=cycle_idx,
            idea_id=idea_id,
            agent=cycle_agent.agent,
        )

        log_step_end_fn(
            "attempt",
            attempt_started_at,
            cycle=cycle_idx,
            idea_id=idea_id,
            attempt_id=attempt_id,
            exit=attempt_result.exit_code,
            experiment_dir=attempt_result.experiment_dir,
            rubric_valid=validation.valid,
            rubric_overall_pass=validation.overall_pass,
            attempt_status=attempt_status,
            failure_streak_after=0,
            log_path=attempt_log_path,
        )
        if isinstance(metrics, dict):
            log_event_fn(
                "attempt_metrics",
                attempt_id=attempt_id,
                combined_score=metrics.get("combined_score"),
                correct=metrics.get("correct"),
            )
        if validation.error and not validation.valid:
            log_event_fn(
                "rubric_validation_error",
                attempt_id=attempt_id,
                error=validation.error,
            )
        if best_updated:
            log_event_fn(
                "best_program_updated",
                attempt_id=attempt_id,
                source_attempt_id=best_manifest.get("source_attempt_id"),
                previous_best_score=previous_best_score,
                new_best_score=new_best_score,
                best_program_dir=best_program_dir,
                best_entrypoint_path=best_manifest.get("entrypoint_path"),
            )
        metrics_score = metrics.get("combined_score") if isinstance(metrics, dict) else None
        combined_score = (
            metrics_score if isinstance(metrics_score, (int, float)) else evaluated_score
        )
        metrics_correct = metrics.get("correct") if isinstance(metrics, dict) else None
        correct = metrics_correct if isinstance(metrics_correct, bool) else evaluated_correct
        score_delta = (
            float(combined_score) - best_score_before
            if isinstance(combined_score, (int, float)) and best_score_before is not None
            else None
        )
        rubric_results = status_data.get("rubrics") if isinstance(status_data, dict) else None
        attempt_failure_reason = (
            None
            if attempt_status == "pass"
            else "process"
            if run_failed
            else "rubric_report"
            if not validation.valid
            else "rubrics"
        )
        log_event_fn(
            "attempt_summary",
            cycle=cycle_idx,
            **cycle_position,
            idea_id=idea_id,
            idea_title=idea_title,
            attempt_id=attempt_id,
            elapsed=format_elapsed_fn(time.monotonic() - attempt_started_at),
            exit=attempt_result.exit_code,
            combined_score=combined_score,
            score_delta=score_delta,
            best_score=best_manifest.get("combined_score"),
            best_updated=best_updated,
            correct=correct,
            rubric_valid=validation.valid,
            rubric_overall_pass=validation.overall_pass,
            rubric_results=rubric_results,
            rubric_error=validation.error,
            attempt_status=attempt_status,
            failure_reason=attempt_failure_reason,
            log_path=attempt_log_path,
        )
        if idea_success:
            log_event_fn(
                "idea_complete",
                idea_id=idea_id,
                cycle=cycle_idx,
                attempts_used=attempt_number,
            )
        else:
            log_event_fn(
                "idea_incomplete",
                idea_id=idea_id,
                cycle=cycle_idx,
                attempts_used=attempt_number,
                next_action="generate_new_idea",
            )

        log_event_fn(
            "cycle_end",
            cycle=cycle_idx,
            **cycle_position,
            idea_id=idea_id,
            idea_success=idea_success,
            attempts_used=attempt_number,
            failure_streak=outer_state.get("failure_streak", 0),
            elapsed=format_elapsed_fn(time.monotonic() - cycle_started_at),
        )

        # -- Shared memory sync: push local → shared after cycle --
        if shared_memory_dir is not None:
            if memory_pre_state is not None:
                state_store.sync_memory_after_cycle(
                    shared_memory_dir, global_memory_dir, memory_pre_state
                )
                logger.info(
                    "Synced local memory → shared after cycle %d",
                    cycle_idx,
                )
            if conclusions_pre_state is not None:
                state_store.sync_conclusions_after_cycle(
                    shared_memory_dir,
                    conclusions_dir,
                    worker_index,
                    conclusions_pre_state,
                )
                logger.info(
                    "Synced local conclusions → shared after cycle %d (worker=%d)",
                    cycle_idx,
                    worker_index,
                )

    if cfg.paper.enabled:
        paper_output_dir = (
            results_dir
            / "paper"
            / f"final_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
        )
        paper_started_at = log_step_start_fn(
            "paper_generation",
            output_dir=paper_output_dir,
            refs_min=cfg.paper.min_refs,
            refs_max=cfg.paper.max_refs,
        )
        paper_result = run_paper_generation_fn(
            agent=cfg.agent.name,
            agent_model=agent_model,
            agent_profile=agent_profile,
            reasoning_effort=reasoning_effort,
            task_dir=task_dir,
            results_dir=results_dir,
            global_memory_dir=global_memory_dir,
            output_dir=paper_output_dir,
            instruct_path=paper_instruct_path,
            contributions_path=paper_contributions_path,
            timeout_seconds=cfg.paper.timeout_seconds,
            timeout_grace_seconds=cfg.paper.timeout_grace_seconds,
            paper_min_refs=cfg.paper.min_refs,
            paper_max_refs=cfg.paper.max_refs,
            paper_profile=cfg.paper.profile,
            two_stage=cfg.paper.two_stage,
            paper_review_rounds=cfg.paper.review_rounds,
            paper_max_repair_rounds=cfg.paper.max_repair_rounds,
            final_ai_review=cfg.paper.ai_review,
            containerize=containerize,
            container_image=container_image,
            container_network=container_network,
            container_auth_root=container_auth_root,
            container_env_file=container_env_file,
            container_session_name=container_session_name,
            container_mount_root=container_mount_root,
        )
        paper_workspace = paper_result.paper_dir if paper_result.paper_dir else paper_output_dir
        paper_workspace.mkdir(parents=True, exist_ok=True)
        paper_log_path = paper_workspace / "wrapper_output.log"
        state_store.write_private_text(paper_log_path, paper_result.output)
        log_step_end_fn(
            "paper_generation",
            paper_started_at,
            exit=paper_result.exit_code,
            paper_workspace=paper_workspace,
            log_path=paper_log_path,
        )

        if paper_result.exit_code != 0:
            log_event_fn(
                "paper_generation_failed",
                exit=paper_result.exit_code,
                log_path=paper_log_path,
                tail_chars=1200,
                output_tail=paper_result.output[-1200:],
            )
            checkpoint_fn(
                label="final" if finalize_on_exit else "paper failed",
                phase="paper",
                event="paper_generation_failed",
                final=finalize_on_exit,
            )
            return 2

        latest_manifest = results_dir / "paper" / "latest_manifest.json"
        if not latest_manifest.exists():
            log_event_fn(
                "paper_generation_error",
                reason="missing_latest_manifest",
                path=latest_manifest,
            )
            checkpoint_fn(
                label="final" if finalize_on_exit else "paper failed",
                phase="paper",
                event="paper_generation_error",
                final=finalize_on_exit,
            )
            return 2
        log_event_fn("paper_generation_complete", paper_workspace=paper_workspace)
        checkpoint_fn(label="paper", phase="paper", event="paper_generation_complete")

    log_event_fn(
        "finished",
        cycles_completed=outer_state.get("current_cycle", 0),
        max_idea_cycles=cfg.execution.max_idea_cycles,
        best_program_dir=best_manifest.get("program_dir"),
        best_entrypoint_path=best_manifest.get("entrypoint_path"),
        best_score=best_manifest.get("combined_score"),
        best_correct=best_manifest.get("correct"),
        total_runtime_elapsed=format_elapsed_fn(time.monotonic() - main_started_at),
    )
    checkpoint_fn(
        label="final" if finalize_on_exit else "campaign finished",
        phase="investigation",
        event="finished",
        final=finalize_on_exit,
    )
    return 0
