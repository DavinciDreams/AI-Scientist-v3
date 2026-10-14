"""Build run-level performance payload across ideas and programs."""

from __future__ import annotations

from pathlib import Path

from .discovery import DiscoveredRun, ProgramEval, idea_node_id, idea_rel_path
from .models import IdeaPerformance, ProgramPerformance, RunPerformance

from src.investigation.program_spec import ProgramSpecError, load_program_spec, resolve_entrypoint


def _program_score(program: ProgramEval) -> float | None:
    score = (program.metrics or {}).get("combined_score")
    return float(score) if isinstance(score, (int, float)) else None


def _program_correct(program: ProgramEval) -> bool | None:
    correct = (program.correct or {}).get("correct")
    return bool(correct) if isinstance(correct, bool) else None


def _program_timestamp(program: ProgramEval) -> float | None:
    try:
        return program.dir_path.stat().st_mtime
    except OSError:
        return None


def _program_code_path(run: DiscoveredRun, program: ProgramEval) -> str | None:
    program_dir = program.dir_path

    path: Path
    spec = None
    try:
        spec = load_program_spec(program_dir, required=False)
    except ProgramSpecError:
        spec = None

    if spec is not None:
        try:
            path = resolve_entrypoint(program_dir, spec)
        except ProgramSpecError:
            path = program_dir / "src" / "program.py"
    else:
        # Legacy runs.
        legacy = program_dir / "program.py"
        path = legacy if legacy.exists() else (program_dir / "src" / "program.py")
    try:
        return Path(path).relative_to(run.path).as_posix()
    except ValueError:
        return None


def build_run_performance(run: DiscoveredRun) -> RunPerformance:
    ideas: list[IdeaPerformance] = []
    programs: list[ProgramPerformance] = []

    max_idea_score: float | None = None
    max_program_score: float | None = None

    for worker in run.workers:
        for idea in worker.ideas:
            node_id = idea_node_id(run, worker, idea)
            idea_path = idea_rel_path(run, idea)
            best_score: float | None = None
            best_label: str | None = None
            best_correct: bool | None = None
            scored_count = 0
            correct_count = 0
            incorrect_count = 0

            for program in idea.programs:
                score = _program_score(program)
                correct = _program_correct(program)
                timestamp = _program_timestamp(program)
                code_path = _program_code_path(run, program)

                if score is not None:
                    scored_count += 1
                    if max_program_score is None or score > max_program_score:
                        max_program_score = score
                    if best_score is None or score > best_score:
                        best_score = score
                        best_label = program.display_label
                        best_correct = correct

                if correct is True:
                    correct_count += 1
                elif correct is False:
                    incorrect_count += 1

                programs.append(
                    ProgramPerformance(
                        worker_index=worker.worker_index,
                        worker_name=worker.worker_name,
                        idea_node_id=node_id,
                        idea_id=idea.idea_id,
                        idea_title=idea.title,
                        idea_dir_name=idea.dir_name,
                        idea_path=idea_path,
                        idea_order_index=idea.order_index,
                        program_label=program.display_label,
                        program_idx=program.program_idx,
                        sequence_idx=program.sequence_idx,
                        attempt_name=program.attempt_name,
                        program_code_path=code_path,
                        timestamp_unix=timestamp,
                        combined_score=score,
                        correct=correct,
                    )
                )

            if best_score is not None and (max_idea_score is None or best_score > max_idea_score):
                max_idea_score = best_score

            ideas.append(
                IdeaPerformance(
                    worker_index=worker.worker_index,
                    worker_name=worker.worker_name,
                    idea_node_id=node_id,
                    idea_id=idea.idea_id,
                    idea_title=idea.title,
                    idea_dir_name=idea.dir_name,
                    idea_path=idea_path,
                    idea_order_index=idea.order_index,
                    program_count=len(idea.programs),
                    scored_program_count=scored_count,
                    correct_program_count=correct_count,
                    incorrect_program_count=incorrect_count,
                    best_combined_score=best_score,
                    best_program_label=best_label,
                    best_correct=best_correct,
                )
            )

    return RunPerformance(
        run_id=run.run_id,
        ideas=ideas,
        programs=programs,
        max_idea_combined_score=max_idea_score,
        max_program_combined_score=max_program_score,
    )
