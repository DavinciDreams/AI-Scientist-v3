"""Human and JSON terminal renderers for structured experiment events."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TextIO

from .logo import render_logo
from .paper_events import is_paper_heartbeat, paper_gate_name, paper_gate_qualifier
from .review_summary import format_review_result, format_revision_review
from .terminal_ui import (
    ERROR_SUFFIXES as _ERROR_SUFFIXES,
    PROGRESS_LABELS as _PROGRESS_LABELS,
    TERMINAL_WRITE_LOCK,
    colors_enabled,
    style,
    terminal_text,
)


def _terminal_columns(stream: TextIO) -> int | None:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty) or not isatty():
        return None
    fileno = getattr(stream, "fileno", None)
    if not callable(fileno):
        return None
    try:
        return max(1, os.get_terminal_size(fileno()).columns)
    except (OSError, ValueError):
        return None


def _cycle_number(fields: dict[str, Any]) -> Any:
    """Campaign-wide cycle number; parallel workers also carry a local ``cycle``."""
    return fields.get("global_cycle", fields.get("cycle"))


class HumanRenderer:
    """Render domain-focused experiment status for people."""

    def __init__(
        self,
        stream: TextIO,
        *,
        verbose: bool,
        results_dir: Path,
        worker_index: int,
        transient_progress: bool = False,
    ) -> None:
        self.stream = stream
        self.verbose = verbose
        self.results_dir = results_dir
        self.worker_index = worker_index
        self.colors_enabled = colors_enabled(stream)
        # When the logger redraws a spinner in place, the spinner already names the
        # running step, so permanent "started" lines would only duplicate the
        # completion line that follows. Redirected output keeps them: without a
        # spinner they are the only sign that a step began.
        self.transient_progress = transient_progress

    def render(self, record: dict[str, Any]) -> None:
        event = str(record["event"])
        fields = record["fields"]
        lines = self._event_lines(event, fields)
        if not lines:
            return
        prefix = f"W{self.worker_index:02d} · " if self.worker_index else ""
        with TERMINAL_WRITE_LOCK:
            for line in lines:
                self.stream.write(f"{prefix}{line}\n" if line else "\n")
            self.stream.flush()

    def _event_lines(self, event: str, fields: dict[str, Any]) -> list[str]:
        generic = self._generic_event_lines(event, fields)
        if generic is not None:
            return generic
        if event == "run_hydra_config":
            return self._verbose_event(event, fields) if self.verbose else []
        if event == "run_config":
            if not fields.get("show_run_header", True):
                return []
            return [*self._logo(), "", *self._run_header(fields)]
        if event == "cycle_start":
            return [
                "",
                self._styled(
                    f"Cycle {_cycle_number(fields)}/{fields.get('total_cycles')}",
                    "phase",
                ),
            ]
        if event == "step_start":
            return self._step_start(fields)
        if event == "step_end":
            return self._step_end(fields)
        if event == "idea_bundle_loaded":
            return [
                "  Idea: "
                + terminal_text(fields.get("idea_title") or fields.get("idea_id") or "unknown"),
                f"  Target rubrics: {fields.get('target_rubric_count', 0)}",
            ]
        if event == "attempt_summary":
            return self._attempt_summary(fields)
        if event == "parallel_start":
            lines = [
                f"Agentic Scientist · parallel · {fields.get('max_idea_cycles', '?')} cycles",
                f"Workers: {fields.get('workers_active', '?')}",
            ]
            if self.verbose:
                lines.append(
                    f"Artifacts: {terminal_text(fields.get('worker_root', self.results_dir))}"
                )
            return lines
        if event == "parallel_worker_launch":
            return [f"  W{int(fields.get('worker', 0)):02d} · started"]
        if event == "parallel_worker_complete":
            worker = int(fields.get("worker", 0))
            exit_code = fields.get("exit", 0)
            status = "complete" if exit_code == 0 else f"FAILED · exit {exit_code}"
            return [f"  W{worker:02d} · {status} · {fields.get('elapsed', 'unknown')}"]
        if event == "parallel_end":
            failures = fields.get("failed_workers", 0)
            status = "complete" if failures in (0, None) else f"failed workers: {failures}"
            return [
                f"Parallel run {status} · {fields.get('ideas', 0)} ideas · "
                f"{fields.get('attempts', 0)} attempts"
            ]
        if event == "heartbeat":
            if fields.get("label"):
                current = fields.get("current")
                total = fields.get("total")
                position = (
                    f"{current}/{total} · " if current is not None and total is not None else ""
                )
                return [
                    self._styled(
                        f"{position}{terminal_text(fields['label'])} "
                        f"{fields.get('elapsed')} · still running",
                        "running",
                    )
                ]
            return [
                f"Cycle {_cycle_number(fields)}/{fields.get('total_cycles')} · "
                f"{_PROGRESS_LABELS.get(str(fields.get('step')), 'Working…')} "
                f"{fields.get('elapsed')} · still running"
            ]
        if event == "cycle_end":
            return [f"Cycle {_cycle_number(fields)} complete  ·  {fields.get('elapsed', 'unknown')}"]
        if event == "finished":
            return self._finished(fields)
        if event == "event_journal_unavailable":
            return [
                self._styled(
                    "WARNING: structured event journal unavailable; terminal logging continues",
                    "warning",
                )
            ]
        if event == "cost_summary":
            message = terminal_text(fields.get("message", "Tokens unavailable"))
            if fields.get("pipeline_elapsed"):
                message += f" · Time {terminal_text(fields['pipeline_elapsed'])}"
            return [
                self._styled(message, "detail")
            ]
        if event == "task_drafting_checkpoint":
            return self._task_drafting_checkpoint(fields)
        if event == "preparation_step":
            label = terminal_text(fields.get("label", "Preparing task"))
            state = fields.get("state")
            if state == "started":
                if self.transient_progress:
                    return []
                return [self._styled(f"  {label}…", "running")]
            outcome = "FAILED" if state == "failed" else "complete"
            elapsed = fields.get("elapsed")
            suffix = f" · {terminal_text(elapsed)}" if elapsed else ""
            return [self._styled(
                f"  {label} · {outcome}{suffix}",
                "failure" if state == "failed" else "success",
            )]
        if event == "idea_incomplete":
            return ["  Idea incomplete · continuing with a new idea"]
        if event.endswith(_ERROR_SUFFIXES) or event in {"interrupted", "campaign_failed"}:
            return self._error(event, fields)
        return self._verbose_event(event, fields) if self.verbose else []

    def _task_drafting_checkpoint(self, fields: dict[str, Any]) -> list[str]:
        label = terminal_text(
            fields.get("label") or str(fields.get("gate", "task drafting")).replace("_", " ")
        )
        state = str(fields.get("state", "completed"))
        verdict = str(fields.get("verdict", "")).upper()
        elapsed = fields.get("elapsed")
        if state == "started":
            if self.transient_progress:
                return []
            return [self._styled(f"  {label}…", "running")]
        outcome = verdict or {
            "completed": "complete",
            "failed": "FAILED",
            "retrying": "retrying",
            "skipped": "skipped",
        }.get(state, state)
        suffix = f" · {terminal_text(outcome)}"
        if elapsed:
            suffix += f" · {terminal_text(elapsed)}"
        semantic = (
            "success"
            if state == "completed" and verdict != "FAIL"
            else "failure"
            if state == "failed" or verdict == "FAIL"
            else "warning"
        )
        lines = [self._styled(f"  {label}{suffix}", semantic)]
        for key, prefix in (
            ("failed_dimensions", "Failed"),
            ("fixed", "Fixed"),
            ("still_failed", "Still failing"),
            ("regressed", "Regressed"),
        ):
            values = fields.get(key)
            if isinstance(values, list) and values:
                rendered = ", ".join(terminal_text(value, limit=80) for value in values[:12])
                lines.append(f"    {prefix}: {rendered}")
        cost_message = fields.get("cost_message")
        if state != "started" and cost_message:
            rendered_cost = terminal_text(cost_message)
            if fields.get("pipeline_elapsed"):
                rendered_cost += f" · Time {terminal_text(fields['pipeline_elapsed'])}"
            lines.append(self._styled(rendered_cost, "detail"))
        return lines

    def _generic_event_lines(
        self,
        event: str,
        fields: dict[str, Any],
    ) -> list[str] | None:
        if event == "pipeline_start":
            task = terminal_text(fields.get("task", "unknown"))
            run_id = terminal_text(fields.get("run_id", ""))
            phases = fields.get("phases", [])
            phase_text = (
                ", ".join(str(item) for item in phases) if isinstance(phases, list) else phases
            )
            lines: list[str] = []
            if fields.get("show_run_header") is True:
                lines.extend(self._logo())
                if fields.get("investigation_selected"):
                    lines.extend(["", *self._run_header(fields)])
                lines.append("")
            lines.extend(
                [
                    self._styled(
                        f"Pipeline · {task}" + (f" · run {run_id}" if run_id else ""),
                        "phase",
                    ),
                    self._styled(f"Phases: {terminal_text(phase_text)}", "detail"),
                ]
            )
            return lines
        if event in {"phase_start", "stage_start"}:
            label = terminal_text(
                fields.get("label") or fields.get("phase_name") or fields.get("stage")
            )
            current = fields.get("current")
            total = fields.get("total")
            position = f" · {current}/{total}" if current is not None and total is not None else ""
            return [self._styled(f"{label}{position}", "phase")]
        if event in {"variant_start", "analysis_start"}:
            noun = "Variant" if event == "variant_start" else "Analysis"
            name = terminal_text(
                fields.get("title") or fields.get("name") or fields.get("id") or "unknown"
            )
            current = fields.get("current", "?")
            total = fields.get("total", "?")
            return [self._styled(f"  {noun} {current}/{total} · {name}", "running")]
        if event in {"phase_end", "stage_end"}:
            label = terminal_text(
                fields.get("label") or fields.get("phase_name") or fields.get("stage")
            )
            status = terminal_text(fields.get("status", "completed"))
            outcome = terminal_text(str(fields.get("outcome", "")).replace("_", " "))
            elapsed = fields.get("elapsed")
            suffix = f" · {outcome}" if outcome else ""
            if elapsed:
                suffix += f" · {elapsed}"
            semantic = (
                "success"
                if status in {"completed", "passed", "success"}
                and outcome != "not qualified"
                else "warning"
                if status in {"degraded", "skipped"} or outcome == "not qualified"
                else "failure"
            )
            return [self._styled(f"{label} · {status}{suffix}", semantic)]
        if event == "review_result":
            return [self._styled(format_review_result(fields), "detail")]
        if event == "revision_result":
            return [self._styled(format_revision_review(fields), "detail")]
        if event == "variant_result":
            return self._item_result(
                fields, noun="Variant", count_key="eval_count", count_label="evals"
            )
        if event == "analysis_result":
            return self._item_result(
                fields,
                noun="Analysis",
                count_key="attempt_count",
                count_label="attempts",
            )
        if event in {"variants_summary", "analyses_summary"}:
            noun = "Variants" if event == "variants_summary" else "Analyses"
            return [
                self._styled(
                    f"{noun} · "
                    f"{fields.get('passed', 0)}/{fields.get('total', 0)} passed · "
                    f"{fields.get('failed', 0)} failed · {fields.get('timeout', 0)} timed out",
                    "success" if fields.get("failed", 0) == 0 else "warning",
                )
            ]
        if event == "pipeline_summary":
            status = fields.get("status", "completed")
            return [
                self._styled(
                    "Analyses · "
                    f"{fields.get('rubric_passed_count', 0)}/"
                    f"{fields.get('analysis_count', 0)} passed rubric · "
                    f"{fields.get('selected_count', 0)} selected",
                    "success" if status == "completed" else "warning",
                )
            ]
        if event == "paper_gate":
            if is_paper_heartbeat(fields.get("state"), fields):
                return []
            gate = paper_gate_name(fields.get("gate", "paper"))
            qualifier = paper_gate_qualifier(fields)
            state = terminal_text(fields.get("state", "started"))
            if state == "started" and self.transient_progress:
                return []
            glyph = {"started": "⠋", "completed": "✓", "failed": "✗", "skipped": "–"}.get(
                state,
                "·",
            )
            semantic = {
                "started": "running",
                "completed": "success",
                "failed": "failure",
                "skipped": "detail",
            }.get(state, "detail")
            line = f"[PAPER] {glyph} {gate} · {state}"
            if qualifier:
                line = f"{line} ({qualifier})"
            if state in {"completed", "failed"} and fields.get("elapsed"):
                line = f"{line} · {terminal_text(fields['elapsed'])}"
            return [self._styled(line, semantic)]
        if event == "pipeline_end":
            status = terminal_text(fields.get("status", "completed"))
            outcome = terminal_text(str(fields.get("outcome", "")).replace("_", " "))
            elapsed = fields.get("elapsed")
            suffix = f" · {outcome}" if outcome else ""
            if elapsed:
                suffix += f" · {elapsed}"
            semantic = (
                "warning"
                if outcome == "not qualified" or status == "completed_degraded"
                else "success"
                if status == "completed"
                else "failure"
            )
            return ["", self._styled(f"Pipeline finished · {status}{suffix}", semantic)]
        return None

    def _logo(self) -> list[str]:
        return render_logo(
            colors=self.colors_enabled,
            columns=_terminal_columns(self.stream),
        )

    def _item_result(
        self,
        fields: dict[str, Any],
        *,
        noun: str,
        count_key: str,
        count_label: str,
    ) -> list[str]:
        name = terminal_text(
            fields.get("title") or fields.get("name") or fields.get("id") or "unknown"
        )
        status = terminal_text(fields.get("status", "unknown"))
        passed = fields.get("rubric_passed") is True
        rubric = "PASS" if passed else "FAIL"
        count = fields.get(count_key, 0)
        semantic = "success" if passed and status == "completed" else "failure"
        return [
            self._styled(
                f"  {noun}: {name} · {status} · {count} {count_label} · rubric {rubric}",
                semantic,
            )
        ]

    def _styled(self, text: str, semantic: str) -> str:
        return style(text, semantic, enabled=self.colors_enabled)

    def _run_header(self, fields: dict[str, Any]) -> list[str]:
        task_dir = Path(str(fields.get("paths.task_dir", "task")))
        cycles = fields.get("execution.max_idea_cycles", "?")
        agent = fields.get("agent.name", "unknown")
        model = fields.get("agent.model")
        reasoning_effort = fields.get("agent.reasoning_effort")
        evaluations = fields.get("execution.max_program_evals_per_attempt", "?")
        workers = fields.get("execution.workers", 1)
        runtime = "Docker" if fields.get("container.enabled") else "native"
        worker_label = "worker" if workers == 1 else "workers"
        agent_line = f"Agent: {terminal_text(agent)}"
        if model not in (None, "none"):
            agent_line += f" / {terminal_text(model)}"
        if reasoning_effort not in (None, "", "none"):
            agent_line += f" / {terminal_text(reasoning_effort)}"
        lines = [
            f"Agentic Scientist · {terminal_text(task_dir.name)} · {cycles} cycles",
            agent_line,
            f"Config: {evaluations} evaluations/idea · {workers} {worker_label} · {runtime}",
            f"Results: {terminal_text(fields.get('paths.results_dir', self.results_dir))}",
        ]
        return lines

    def _step_start(self, fields: dict[str, Any]) -> list[str]:
        label = _PROGRESS_LABELS.get(str(fields.get("step")))
        if label:
            if self.transient_progress:
                return []
            return [self._styled(f"  {label}", "running")]
        return self._verbose_event("step_start", fields) if self.verbose else []

    def _step_end(self, fields: dict[str, Any]) -> list[str]:
        if fields.get("step") not in {"idea_generation", "task_draft", "paper_generation"}:
            return []
        status = "done" if fields.get("exit", 0) == 0 else f"failed (exit {fields.get('exit')})"
        semantic = "success" if fields.get("exit", 0) == 0 else "failure"
        return [
            self._styled(
                f"  {status}  ·  {fields.get('elapsed', 'unknown')}",
                semantic,
            )
        ]

    def _attempt_summary(self, fields: dict[str, Any]) -> list[str]:
        lines = [f"  Experiment finished  ·  {fields.get('elapsed', 'unknown')}"]
        score = fields.get("combined_score")
        best = fields.get("best_score")
        if isinstance(score, (int, float)):
            score_line = f"  Score: {score:.2f}"
            delta = fields.get("score_delta")
            if isinstance(delta, (int, float)):
                arrow = "▲" if delta >= 0 else "▼"
                score_line += f"  {arrow} {delta:+.2f}"
            if isinstance(best, (int, float)):
                score_line += f"                 Best: {best:.2f}"
            lines.append(self._styled(score_line, "score"))
        else:
            lines.append("  Score: unavailable")

        correct = fields.get("correct")
        correctness = "PASS" if correct is True else "FAIL" if correct is False else "UNKNOWN"
        lines.append(f"  Correctness: {correctness}")
        rubrics = fields.get("rubric_results")
        if isinstance(rubrics, list) and rubrics:
            passed = sum(
                1 for item in rubrics if isinstance(item, dict) and item.get("pass") is True
            )
            lines.append(f"  Rubrics: {passed}/{len(rubrics)} passed")
            for item in rubrics:
                if not isinstance(item, dict):
                    continue
                state = "PASS" if item.get("pass") is True else "FAIL"
                detail = f"    {state}  {terminal_text(item.get('id', 'unknown'), limit=120)}"
                evidence = item.get("evidence")
                if self.verbose and evidence:
                    detail += f" — {terminal_text(evidence, limit=200)}"
                lines.append(self._styled(detail, "success" if state == "PASS" else "failure"))
        else:
            lines.append("  Rubrics: unavailable")

        status = fields.get("attempt_status")
        reason = fields.get("failure_reason")
        if status == "pass":
            lines.append(self._styled("  Outcome: PASSED", "success"))
        elif reason == "process":
            lines.append(
                self._styled(
                    f"  Outcome: FAILED — agent process exited {fields.get('exit')}",
                    "failure",
                )
            )
        elif not fields.get("rubric_valid", False):
            lines.append(self._styled("  Outcome: FAILED — rubric report invalid", "failure"))
        else:
            lines.append(self._styled("  Outcome: FAILED — rubric checks did not pass", "failure"))
        if self.verbose:
            lines.extend(self._path_lines(fields))
        return lines

    def _finished(self, fields: dict[str, Any]) -> list[str]:
        score = fields.get("best_score")
        score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "unavailable"
        return [
            "",
            self._styled(
                f"Finished · {fields.get('cycles_completed', 0)} cycles · best score {score_text}",
                "success",
            ),
            f"Runtime: {fields.get('total_runtime_elapsed', 'unknown')}",
        ]

    def _error(self, event: str, fields: dict[str, Any]) -> list[str]:
        reason = fields.get("reason") or fields.get("error") or fields.get("message") or event
        lines = [self._styled(f"ERROR: {terminal_text(reason, limit=300)}", "failure")]
        path_lines = self._path_lines(fields, absolute=self.verbose)
        if path_lines:
            lines.extend(path_lines)
        return lines

    def _path_lines(self, fields: dict[str, Any], *, absolute: bool = True) -> list[str]:
        lines: list[str] = []
        for key in ("path", "log_path", "output_dir", "experiment_dir"):
            raw = fields.get(key)
            if raw in (None, "none", ""):
                continue
            path = Path(str(raw))
            if not absolute:
                try:
                    path = path.relative_to(self.results_dir)
                except ValueError:
                    path = Path("<external>") / path.name
            lines.append(f"  {key}: {terminal_text(path)}")
        return lines

    def _verbose_event(self, event: str, fields: dict[str, Any]) -> list[str]:
        details = " ".join(
            f"{terminal_text(key, limit=80)}={terminal_text(value)}"
            for key, value in fields.items()
        )
        return [f"{event}: {details}".rstrip()]


class JsonRenderer:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream

    def render(self, record: dict[str, Any]) -> None:
        with TERMINAL_WRITE_LOCK:
            self.stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            self.stream.flush()
