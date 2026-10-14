"""EventLogger renderers, journals, heartbeats, cost summaries, preparation milestones and logo."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.investigation import runtime as runtime_module
from src.investigation.cost_tracking import record_events
from src.investigation.event_logging import EventLogger, format_pipeline_elapsed
from src.investigation.logo import render_logo
from src.investigation.terminal_ui import terminal_width

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
GET_TERMINAL_SIZE = "src.investigation.event_logging.shutil.get_terminal_size"


class TtyStream(io.StringIO):
    def isatty(self) -> bool:
        return True


def _sub(tmp_path: Path, name: str) -> Path:
    """Fresh results directory for one section of a merged test (isolates journals and cost roots)."""
    path = tmp_path / name
    path.mkdir()
    return path


def _logger(
    tmp_path: Path,
    stream: io.StringIO | None = None,
    *,
    log_format: str = "human",
    verbose: bool = False,
    heartbeat_seconds: int = 0,
    clock: list[float] | None = None,
    **kwargs: Any,
) -> EventLogger:
    """EventLogger with a fixed session/timestamp; `clock` pins monotonic time and disables the thread."""
    if clock is not None:
        kwargs["monotonic_fn"] = lambda: clock[0]
        kwargs.setdefault("start_progress_thread", False)
    return EventLogger(
        log_format=log_format,
        verbose=verbose,
        heartbeat_seconds=heartbeat_seconds,
        results_dir=tmp_path,
        worker_index=kwargs.pop("worker_index", 0),
        stream=stream if stream is not None else io.StringIO(),
        session_id="session-test",
        timestamp_fn=lambda: "2026-08-17T10:00:00.000Z",
        **kwargs,
    )


def _records(tmp_path: Path, name: str = "events.jsonl") -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / name).read_text(encoding="utf-8").splitlines()]


def _cost_record(results_dir: Path, event_id: str, stage: str, **fields: Any) -> None:
    record = {
        "event_id": event_id,
        "invocation_id": event_id,
        "phase": "paper",
        "stage": stage,
        "source": "headless",
        "agent": "codex",
        "model": "gpt-5.6-sol",
        "status": "completed",
        "billing_basis": "reported",
        "usage": {},
    }
    record.update(fields)
    record_events(results_dir, [record])


def _cost_summary(tmp_path: Path) -> dict[str, Any]:
    return next(record for record in _records(tmp_path) if record["event"] == "cost_summary")


def _log_record(name: str, msg: str, *, checkpoint_stored: bool = False) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name, level=logging.WARNING, pathname=__file__, lineno=1, msg=msg, args=(), exc_info=None
    )
    if checkpoint_stored:
        record.checkpoint_stored = True
    return record


def _emit_run_config(logger: EventLogger, **overrides: Any) -> None:
    fields = {
        "agent.name": "codex",
        "paths.task_dir": "task",
        "execution.max_idea_cycles": 1,
        "execution.max_program_evals_per_attempt": 1,
        "execution.workers": 1,
        "container.enabled": False,
    }
    fields.update(overrides)
    logger.emit("run_config", **fields)


@contextmanager
def _restored_root_logger():
    """Snapshot root handlers/level inside the test body (pytest swaps stderr/handlers per phase)."""
    root = logging.getLogger()
    original_handlers, original_level = root.handlers[:], root.level
    try:
        yield root
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)


def _patched_stderr(monkeypatch) -> io.StringIO:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    return stream


@pytest.fixture
def fresh_cost_root(monkeypatch):
    monkeypatch.setattr(runtime_module, "_INHERITED_COST_ROOT", None)
    monkeypatch.setattr(runtime_module, "_MANAGED_COST_ROOT", None)


def _configure_event_logging(results_dir: Path) -> None:
    runtime_module.configure_event_logging(
        log_format="human", verbose=False, heartbeat_seconds=0, results_dir=results_dir,
        worker_index=0, journal_enabled=False,
    )


# ---------------------------------------------------------------------------
# Standard logging / formatter
# ---------------------------------------------------------------------------


FORMATTER_CASES = [
    (
        "checkpoint-label-drops-module-prefix",
        "src.investigation.run_checkpoint",
        "Checkpoint 'paper phase' excluded 3 sensitive path(s)",
        True,
        "[CHECKPOINT STORED] Checkpoint 'paper phase'",
        ["[WARNING]", "src.investigation.run_checkpoint"],
    ),
    (
        "label-cannot-be-claimed-by-other-logger",
        "untrusted.extension",
        "not a checkpoint",
        True,
        "[WARNING] untrusted.extension: not a checkpoint",
        ["[CHECKPOINT STORED]"],
    ),
    (
        "warning-keeps-severity-and-module",
        "src.pipeline.cli",
        "ordinary warning",
        False,
        "[WARNING] src.pipeline.cli: ordinary warning",
        [],
    ),
]


def test_pipeline_elapsed_units_and_standard_log_formatter() -> None:
    # format_pipeline_elapsed uses unambiguous units
    for seconds, expected in [(100.0, "1m 40s"), (7_740.0, "2h 9m"), (93_784.9, "26h 3m")]:
        assert format_pipeline_elapsed(seconds) == expected, seconds

    # StandardLogFormatter labels checkpoint records and keeps severity/module for everything else
    for case, name, msg, stored, expected, absent in FORMATTER_CASES:
        rendered = runtime_module.StandardLogFormatter().format(
            _log_record(name, msg, checkpoint_stored=stored)
        )

        assert expected in rendered, case
        for text in absent:
            assert text not in rendered, case


def test_configure_standard_logging_handler_policy(monkeypatch) -> None:
    # one process-wide format
    with _restored_root_logger() as root:
        runtime_module.configure_standard_logging(level=logging.WARNING)

        assert root.level == logging.WARNING
        assert any(isinstance(h, runtime_module.StandardLogHandler) for h in root.handlers)

    # replaces a preconfigured plain terminal handler
    with _restored_root_logger() as root, monkeypatch.context() as m:
        terminal_stream = _patched_stderr(m)
        preconfigured = logging.StreamHandler()
        root.handlers.append(preconfigured)
        runtime_module.configure_standard_logging(level=logging.WARNING)

        root.handle(
            _log_record(
                "src.investigation.run_checkpoint",
                "Checkpoint 'paper phase' excluded 1 sensitive path(s)",
                checkpoint_stored=True,
            )
        )

        assert preconfigured not in root.handlers
        assert sum(isinstance(h, runtime_module.StandardLogHandler) for h in root.handlers) == 1
        assert terminal_stream.getvalue().count("[CHECKPOINT STORED]") == 1

    # respects a filtered terminal handler
    with _restored_root_logger() as root, monkeypatch.context() as m:
        terminal_stream = _patched_stderr(m)
        filtered = logging.StreamHandler()
        filtered.addFilter(lambda _record: False)
        root.handlers.append(filtered)
        root.setLevel(logging.ERROR)
        runtime_module.configure_standard_logging(level=logging.INFO)

        root.handle(_log_record("src.pipeline.cli", "private diagnostic"))

        assert filtered in root.handlers
        assert root.level == logging.ERROR
        assert terminal_stream.getvalue() == ""

    # respects a formatted terminal handler
    with _restored_root_logger() as root, monkeypatch.context() as m:
        terminal_stream = _patched_stderr(m)
        formatted = logging.StreamHandler()
        formatted.setLevel(logging.ERROR)
        formatted.setFormatter(logging.Formatter("REDACTED: %(levelname)s"))
        root.handlers.append(formatted)
        root.setLevel(logging.ERROR)

        runtime_module.configure_standard_logging(level=logging.INFO)

        assert formatted in root.handlers
        assert root.level == logging.ERROR
        assert not any(
            isinstance(h, runtime_module.StandardLogHandler)
            and getattr(h, "stream", None) is terminal_stream
            for h in root.handlers
        )


def test_standard_log_starts_after_active_spinner_line(tmp_path: Path, monkeypatch) -> None:
    spinner_stream, diagnostic_stream, clock = TtyStream(), io.StringIO(), [10.0]
    logger = _logger(tmp_path, spinner_stream, heartbeat_seconds=60, clock=clock)
    monkeypatch.setattr(runtime_module, "_EVENT_LOGGER", logger)
    logger.start_progress("attempt", started_at=10.0)
    clock[0] = 11.0
    logger.poll_progress()
    handler = runtime_module.StandardLogHandler(diagnostic_stream)
    handler.setFormatter(runtime_module.StandardLogFormatter())

    handler.handle(
        _log_record(
            "src.investigation.run_checkpoint",
            "Checkpoint 'evaluation c002 a01' excluded 1 sensitive path(s)",
            checkpoint_stored=True,
        )
    )

    assert spinner_stream.getvalue().endswith("\r\x1b[2K\n")
    assert diagnostic_stream.getvalue().startswith("20")
    assert "[CHECKPOINT STORED]" in diagnostic_stream.getvalue()
    logger.close()


# ---------------------------------------------------------------------------
# Human / JSON renderers and journal
# ---------------------------------------------------------------------------


def test_human_run_header_rendering(tmp_path: Path) -> None:
    # compact run header without repeated paths
    results = _sub(tmp_path, "compact")
    stream = io.StringIO()
    logger = _logger(results, stream)
    logger.emit(
        "run_hydra_config", **{"paths.task_dir": "/very/long/task", "execution.max_idea_cycles": "20"}
    )
    _emit_run_config(
        logger,
        **{
            "agent.name": "pi",
            "agent.model": "deepseek-v4-flash",
            "agent.reasoning_effort": "high",
            "paths.task_dir": "/very/long/bayesian_optimization",
            "paths.results_dir": results,
            "execution.max_idea_cycles": 20,
            "execution.max_program_evals_per_attempt": 4,
            "container.enabled": True,
        },
    )

    rendered = stream.getvalue()
    assert rendered.index("████") < rendered.index("Agentic Scientist")
    assert "Agentic Scientist · bayesian_optimization · 20 cycles" in rendered
    assert "Agent: pi / deepseek-v4-flash / high" in rendered
    assert "4 evaluations/idea · 1 worker · Docker" in rendered
    assert f"Results: {results}" in rendered
    assert "Artifacts:" not in rendered
    assert "/very/long/task" not in rendered
    assert "run_hydra_config" not in rendered

    # managed child keeps the run_config event without rendering the header
    child = _sub(tmp_path, "child")
    stream = io.StringIO()
    _emit_run_config(
        _logger(child, stream),
        show_run_header=False,
        **{"execution.max_idea_cycles": 2, "container.enabled": True},
    )

    assert stream.getvalue() == ""
    records = _records(child)
    assert records[-1]["event"] == "run_config"
    assert records[-1]["fields"]["show_run_header"] is False

    # header escapes untrusted agent fields
    stream = io.StringIO()
    _emit_run_config(
        _logger(_sub(tmp_path, "escape"), stream),
        **{
            "agent.name": "codex\nforged",
            "agent.model": "model\x1b]8;;https://example.com\x07link",
            "agent.reasoning_effort": "high\nforged",
        },
    )

    rendered = stream.getvalue()
    assert "codex\\nforged" in rendered
    assert "high\\nforged" in rendered
    assert "\x1b" not in rendered
    assert "\x07" not in rendered

    # header omits unspecified reasoning effort
    stream = io.StringIO()
    _emit_run_config(_logger(_sub(tmp_path, "effort"), stream), **{"agent.model": "gpt-5.6-sol"})

    assert "Agent: codex / gpt-5.6-sol\n" in stream.getvalue()


def test_human_renderer_cycle_rubrics_and_verbose_evidence(tmp_path: Path) -> None:
    # cycle, idea, score and every rubric (evidence hidden when not verbose)
    stream = io.StringIO()
    logger = _logger(tmp_path, stream)

    logger.emit("cycle_start", cycle=2, total_cycles=20)
    logger.emit("step_start", step="idea_generation", cycle=2, status="started")
    logger.emit(
        "step_end", step="idea_generation", cycle=2, status="completed", elapsed="3m 53s", exit=0
    )
    logger.emit(
        "idea_bundle_loaded",
        cycle=2,
        idea_id="ensemble_ts",
        idea_title="Ensemble Surrogate with Thompson Sampling",
        target_rubric_count=3,
    )
    logger.emit("step_start", step="attempt", cycle=2, status="started")
    logger.emit(
        "attempt_summary",
        cycle=2,
        elapsed="1h 32m",
        exit=0,
        combined_score=93.61,
        score_delta=1.75,
        best_score=93.61,
        correct=True,
        rubric_valid=True,
        rubric_results=[
            {"id": "held_out_generalization", "pass": True, "evidence": "0.93"},
            {"id": "noise_robustness", "pass": False, "evidence": "below threshold"},
            {"id": "wall_clock_overhead", "pass": True, "evidence": "12 seconds"},
        ],
        attempt_status="fail",
        failure_reason="rubrics",
    )
    logger.emit("cycle_end", cycle=2, total_cycles=20, elapsed="1h 36m")

    rendered = stream.getvalue()
    for text in (
        "Cycle 2/20",
        "Generating idea",
        "3m 53s",
        "Idea: Ensemble Surrogate with Thompson Sampling",
        "Score: 93.61  ▲ +1.75",
        "Best: 93.61",
        "Correctness: PASS",
        "Rubrics: 2/3 passed",
        "PASS  held_out_generalization",
        "FAIL  noise_robustness",
        "PASS  wall_clock_overhead",
        "Outcome: FAILED — rubric checks did not pass",
    ):
        assert text in rendered
    assert "below threshold" not in rendered

    # verbose renderer includes paths and rubric evidence
    stream = io.StringIO()
    log_path = tmp_path / "ideas" / "idea-1" / "wrapper_output.log"
    _logger(tmp_path, stream, verbose=True).emit(
        "attempt_summary",
        cycle=1,
        elapsed="2.0s",
        exit=0,
        combined_score=1.0,
        best_score=1.0,
        correct=True,
        rubric_valid=True,
        rubric_results=[{"id": "score", "pass": True, "evidence": "score exceeded 0.9"}],
        attempt_status="pass",
        log_path=log_path,
    )

    assert "score exceeded 0.9" in stream.getvalue()
    assert str(log_path) in stream.getvalue()


def test_json_renderer_records_redaction_and_non_finite_numbers(tmp_path: Path) -> None:
    # one parseable record per line
    stream = io.StringIO()
    _logger(tmp_path, stream, log_format="json").emit(
        "cycle_start", cycle=3, total_cycles=8, output_dir=tmp_path / "cycle"
    )

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "event": "cycle_start",
        "fields": {"cycle": 3, "output_dir": str(tmp_path / "cycle"), "total_cycles": 8},
        "phase": None,
        "session_id": "session-test",
        "source": "wrapper",
        "timestamp": "2026-08-17T10:00:00.000Z",
        "worker_index": 0,
    }

    # raw output tail is redacted from the console but retained in the journal
    redacted = _sub(tmp_path, "redacted")
    stream = io.StringIO()
    _logger(redacted, stream, log_format="json").emit(
        "paper_generation_failed", output_tail="token=secret", exit=1
    )

    assert "output_tail" not in json.loads(stream.getvalue())["fields"]
    assert _records(redacted)[0]["fields"]["output_tail"] == "token=secret"

    # non-finite numbers normalize to null
    stream = io.StringIO()
    _logger(tmp_path, stream, log_format="json").emit(
        "attempt_summary", combined_score=float("nan"), best_score=float("inf")
    )

    record = json.loads(
        stream.getvalue(),
        parse_constant=lambda value: pytest.fail(f"non-standard JSON constant: {value}"),
    )
    assert record["fields"]["combined_score"] is None
    assert record["fields"]["best_score"] is None


def test_event_journal_permissions_symlink_guard_and_log_format_validation(tmp_path: Path) -> None:
    # sessions append to the journal with private permissions
    journal_dir = _sub(tmp_path, "journal")
    event_path = journal_dir / "events.jsonl"
    event_path.touch(mode=0o644)
    event_path.chmod(0o644)
    first = _logger(journal_dir, log_format="json")
    first.emit("first", value=1)
    first.close()
    second = EventLogger(
        log_format="json", verbose=False, heartbeat_seconds=0, results_dir=journal_dir, worker_index=0,
        stream=io.StringIO(), session_id="session-two",
        timestamp_fn=lambda: "2026-08-17T10:00:01.000Z",
    )
    second.emit("second", value=2)
    second.close()

    records = _records(journal_dir)
    assert [record["event"] for record in records] == ["first", "second"]
    assert [record["session_id"] for record in records] == ["session-test", "session-two"]
    assert os.stat(event_path).st_mode & 0o777 == 0o600

    # an unsafe (symlinked) journal warns once and terminal logging continues
    unsafe_dir = _sub(tmp_path, "unsafe")
    target = unsafe_dir / "elsewhere.log"
    target.write_text("untouched\n", encoding="utf-8")
    (unsafe_dir / "events.jsonl").symlink_to(target)
    stream = io.StringIO()
    logger = _logger(unsafe_dir, stream, log_format="json")

    logger.emit("first")
    logger.emit("second")

    events = [json.loads(line)["event"] for line in stream.getvalue().splitlines()]
    assert events == ["event_journal_unavailable", "first", "second"]
    assert target.read_text(encoding="utf-8") == "untouched\n"

    # unknown log_format is rejected
    with pytest.raises(ValueError, match="log_format"):
        EventLogger(
            log_format="legacy", verbose=False, heartbeat_seconds=0, results_dir=tmp_path, worker_index=0
        )


# ---------------------------------------------------------------------------
# Progress / heartbeat
# ---------------------------------------------------------------------------


def test_redirected_progress_heartbeat_global_cycle_and_interrupt(tmp_path: Path) -> None:
    # redirected progress emits a durable heartbeat and stops cleanly
    durable = _sub(tmp_path, "durable")
    stream, clock = io.StringIO(), [100.0]
    logger = _logger(durable, stream, heartbeat_seconds=60, clock=clock)

    logger.start_progress("attempt", started_at=100.0, cycle=4, total_cycles=20)
    clock[0] = 161.0
    logger.poll_progress()
    logger.stop_progress("attempt")
    before = stream.getvalue()
    clock[0] = 230.0
    logger.poll_progress()

    assert "Cycle 4/20 · Running experiment… 1m 1s · still running" in before
    assert stream.getvalue() == before
    heartbeat = _records(durable)[-1]
    assert heartbeat["event"] == "heartbeat"
    assert heartbeat["fields"]["elapsed_seconds"] == 61.0

    # parallel workers render the global cycle rather than the local one
    stream, clock = io.StringIO(), [100.0]
    logger = _logger(_sub(tmp_path, "global"), stream, heartbeat_seconds=60, clock=clock, worker_index=2)

    # Worker 2's first local cycle is campaign cycle 2 of 3.
    logger.emit("cycle_start", cycle=1, global_cycle=2, total_cycles=3)
    logger.start_progress("attempt", started_at=100.0, cycle=1, global_cycle=2, total_cycles=3)
    clock[0] = 161.0
    logger.poll_progress()
    logger.stop_progress("attempt")
    logger.emit("cycle_end", cycle=1, global_cycle=2, total_cycles=3, elapsed="1m 1s")

    rendered = stream.getvalue()
    assert "W02 · Cycle 2/3\n" in rendered
    assert "W02 · Cycle 2/3 · Running experiment… 1m 1s · still running" in rendered
    assert "W02 · Cycle 2 complete  ·  1m 1s" in rendered
    assert "Cycle 1/" not in rendered

    # an interrupt cancels active progress
    stream, clock = io.StringIO(), [0.0]
    logger = _logger(_sub(tmp_path, "interrupt"), stream, heartbeat_seconds=1, clock=clock)
    logger.start_progress("attempt", started_at=0.0, cycle=1, total_cycles=1)

    logger.emit("interrupted", action="cleanup")
    before = stream.getvalue()
    clock[0] = 10.0
    logger.poll_progress()

    assert stream.getvalue() == before


def test_task_drafting_checkpoint_heartbeat_context(tmp_path: Path) -> None:
    # a retry updates the active heartbeat context
    retry = _sub(tmp_path, "retry")
    clock = [100.0]
    logger = _logger(retry, heartbeat_seconds=60, clock=clock)

    logger.emit(
        "task_drafting_checkpoint", gate="smoke", state="started", label="Evaluator smoke",
        attempt=2, max_attempts=4,
    )
    clock[0] = 130.0
    logger.emit(
        "task_drafting_checkpoint", gate="smoke", state="retrying",
        label="Evaluator smoke · infrastructure retry",
        infrastructure_attempt=2, max_infrastructure_attempts=3,
    )
    clock[0] = 161.0
    logger.poll_progress()

    heartbeat = _records(retry)[-1]
    assert heartbeat["event"] == "heartbeat"
    assert heartbeat["fields"]["gate"] == "smoke"
    assert heartbeat["fields"]["attempt"] == 2
    assert heartbeat["fields"]["infrastructure_attempt"] == 2
    assert heartbeat["fields"]["max_infrastructure_attempts"] == 3
    assert heartbeat["fields"]["elapsed_seconds"] == 61.0

    # nested completion restores the outer heartbeat; the outer completion stops it
    nested = _sub(tmp_path, "nested")
    clock = [100.0]
    logger = _logger(nested, heartbeat_seconds=60, clock=clock)

    def checkpoint(gate: str, state: str, label: str) -> None:
        logger.emit("task_drafting_checkpoint", gate=gate, state=state, label=label)

    checkpoint("scaffold", "started", "Scaffold generation")
    clock[0] = 110.0
    checkpoint("scaffold", "started", "Scaffold agent attempt 1/2")
    checkpoint("scaffold", "retrying", "Scaffold agent attempt 1/2")
    checkpoint("scaffold", "started", "Scaffold agent attempt 2/2")
    checkpoint("qualification", "retrying", "Unrelated stale retry")
    checkpoint("qualification", "completed", "Unrelated stale terminal")
    checkpoint("scaffold", "completed", "Scaffold agent attempt 2/2")
    clock[0] = 161.0
    logger.poll_progress()

    heartbeat = _records(nested)[-1]
    assert heartbeat["event"] == "heartbeat"
    assert heartbeat["fields"]["gate"] == "scaffold"
    assert heartbeat["fields"]["label"] == "Scaffold generation"
    assert heartbeat["fields"]["elapsed_seconds"] == 61.0

    checkpoint("scaffold", "completed", "Scaffold generation")
    before_poll = (nested / "events.jsonl").read_text(encoding="utf-8")
    clock[0] = 230.0
    logger.poll_progress()
    assert (nested / "events.jsonl").read_text(encoding="utf-8") == before_poll


def test_background_progress_thread_stops_on_close(tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = _logger(tmp_path, stream, log_format="json", heartbeat_seconds=1, progress_tick_seconds=0.01)
    logger.start_progress("attempt", started_at=time.monotonic() - 2, cycle=1, total_cycles=1)

    deadline = time.monotonic() + 1
    while '"event": "heartbeat"' not in stream.getvalue() and time.monotonic() < deadline:
        time.sleep(0.01)
    logger.close()
    rendered = stream.getvalue()
    time.sleep(0.03)

    assert '"event": "heartbeat"' in rendered
    assert stream.getvalue() == rendered


def test_human_renderer_escapes_controls_external_paths_and_soft_errors(tmp_path: Path) -> None:
    # terminal control characters are escaped
    stream = io.StringIO()
    logger = _logger(tmp_path, stream)

    logger.emit(
        "idea_bundle_loaded",
        idea_title="forged\nline\x1b]8;;https://example.com\x07link‮evil",
        target_rubric_count=1,
    )
    logger.emit(
        "attempt_summary", elapsed="1s", exit=0, rubric_valid=True,
        rubric_results=[{"id": "rubric\rname\x1b[2J", "pass": True}], attempt_status="pass",
    )

    rendered = stream.getvalue()
    assert "forged\\nline" in rendered
    assert "rubric\\rname" in rendered
    assert "\\u202e" in rendered
    assert "‮" not in rendered
    assert "\x1b" not in rendered
    assert "\x07" not in rendered

    # non-verbose external error paths do not disclose the absolute path
    stream = io.StringIO()
    _logger(tmp_path, stream).emit(
        "config_error", reason="missing task", path="/home/private/project/task.md"
    )

    assert "/home/private/project" not in stream.getvalue()
    assert "<external>/task.md" in stream.getvalue()

    # an expected incomplete idea is not rendered as a system error
    stream = io.StringIO()
    _logger(tmp_path, stream).emit("idea_incomplete", idea_id="test", next_action="generate_new_idea")

    assert "ERROR" not in stream.getvalue()
    assert "Idea incomplete · continuing with a new idea" in stream.getvalue()


# ---------------------------------------------------------------------------
# TTY styling
# ---------------------------------------------------------------------------


def test_human_tty_colors_spinner_refresh_and_no_color(tmp_path: Path) -> None:
    # semantic colors and spinner frames under FORCE_COLOR
    stream, clock = TtyStream(), [10.0]
    with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=True):
        logger = _logger(_sub(tmp_path, "colors"), stream, heartbeat_seconds=60, clock=clock)
        logger.emit("stage_start", stage="execute", label="Execute variants")
        logger.emit("variant_start", title="Adaptive restart", current=1, total=3)
        clock[0] = 11.0
        logger.poll_progress()
        logger.emit(
            "variant_result", title="Adaptive restart", status="completed", rubric_passed=True, eval_count=2
        )

    rendered = stream.getvalue()
    assert "\x1b[1;36m" in rendered
    assert "\x1b[33m" in rendered
    assert any(frame in rendered for frame in SPINNER_FRAMES)
    assert "\x1b[32m" in rendered

    # TTY progress refreshes in place without a permanent heartbeat line
    stream, clock = TtyStream(), [10.0]
    # Pin the environment so an ambient FORCE_COLOR/TERM does not wrap the frame in ANSI.
    with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True):
        logger = _logger(_sub(tmp_path, "refresh"), stream, heartbeat_seconds=60, clock=clock)
        logger.start_progress("idea_generation", started_at=10.0, cycle=1, total_cycles=2)
        clock[0] = 11.0
        logger.poll_progress()
        logger.stop_progress("idea_generation")

    rendered = stream.getvalue()
    assert "\r\x1b[2K⠋ 1/2 · Generating idea… 1s" in rendered
    assert "still running\n" not in rendered
    logger.close()

    # NO_COLOR disables ANSI but keeps human lines
    stream = TtyStream()
    with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True):
        _logger(_sub(tmp_path, "no-color"), stream).emit(
            "stage_start", stage="publish", label="Publish analyses"
        )

    assert "Publish analyses" in stream.getvalue()
    assert "\x1b[" not in stream.getvalue()


def test_human_tty_spinner_fits_terminal_width(tmp_path: Path) -> None:
    cases = [
        ("wide-glyphs", 40, 0, "Attribution 測定測定測定測定測定測定測定測定測定測定測定測定測定測定測定", True),
        ("narrow", 10, 1, "Attribution of the headline score", False),
    ]
    for case, columns, worker_index, title, check_suffix in cases:
        stream, clock = TtyStream(), [0.0]
        with (
            patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True),
            patch(GET_TERMINAL_SIZE, return_value=os.terminal_size((columns, 24))),
        ):
            logger = _logger(
                _sub(tmp_path, case), stream, heartbeat_seconds=60, clock=clock, worker_index=worker_index
            )
            logger.emit("analysis_start", current=1, total=3, title=title)
            clock[0] = 1.0
            logger.poll_progress()

        spinner_frame = stream.getvalue().rsplit("\x1b[2K", 1)[-1]
        assert terminal_width(spinner_frame) <= columns, case
        if check_suffix:
            assert spinner_frame.endswith("1s"), case


# ---------------------------------------------------------------------------
# Pipeline renderer / headers / logo
# ---------------------------------------------------------------------------


PIPELINE_HEADER_CASES: list[tuple[str, dict[str, Any], bool]] = [
    (
        "without-investigation-omits-investigation-config",
        {"phases": ["paper", "review"], "investigation_selected": False, "show_run_header": True},
        True,
    ),
    (
        "header-suppressed-keeps-pipeline-status",
        {"phases": ["investigation"], "investigation_selected": True, "show_run_header": False},
        False,
    ),
    ("fail-closed-without-ownership", {"phases": ["investigation"]}, False),
]


def test_pipeline_renderer_headers_outcomes_and_banner_ownership(tmp_path: Path) -> None:
    # generic pipeline events render named outcomes
    generic = _sub(tmp_path, "generic")
    stream = io.StringIO()
    logger = _logger(
        generic, stream, source="pipeline", renderer="pipeline", journal_name="pipeline_events.jsonl"
    )

    logger.emit(
        "pipeline_start",
        task="circpack",
        phases=["task_drafting", "investigation", "final_evaluation", "paper"],
        investigation_selected=True,
        show_run_header=True,
        **{
            "agent.name": "codex",
            "agent.model": "gpt-5.6-sol",
            "agent.reasoning_effort": "high",
            "paths.task_dir": "/run/task",
            "paths.results_dir": "/results/circpack/codex/run-123",
            "execution.max_idea_cycles": 5,
            "execution.max_program_evals_per_attempt": 1,
            "execution.workers": 1,
            "container.enabled": True,
        },
    )
    logger.emit("phase_start", phase_name="task_drafting", label="Task drafting")
    logger.emit("phase_end", phase_name="task_drafting", status="completed", elapsed="2m")
    logger.emit("phase_start", phase_name="investigation", label="Investigation")
    logger.emit("phase_end", phase_name="investigation", status="completed", elapsed="3m 1s")
    logger.emit("phase_start", phase_name="final_evaluation", label="Final evaluation")
    logger.emit(
        "phase_end", phase_name="final_evaluation", label="Final evaluation", status="completed",
        outcome="not_qualified", elapsed="51.8s",
    )
    logger.emit("pipeline_end", status="completed", outcome="not_qualified", elapsed="9m 4s")

    rendered = stream.getvalue()
    assert (
        rendered.index("████")
        < rendered.index("Agentic Scientist · task · 5 cycles")
        < rendered.index("Pipeline · circpack")
        < rendered.index("Task drafting")
    )
    assert "Agent: codex / gpt-5.6-sol / high" in rendered
    assert "Config: 1 evaluations/idea · 1 worker · Docker" in rendered
    assert "Results: /results/circpack/codex/run-123" in rendered
    assert "Investigation" in rendered
    assert "Final evaluation · completed · not qualified · 51.8s" in rendered
    assert "completed" in rendered
    assert "Pipeline finished · completed · not qualified · 9m 4s" in rendered
    assert {record["source"] for record in _records(generic, "pipeline_events.jsonl")} == {"pipeline"}

    # pipeline_start header: logo only with explicit ownership; never the investigation config
    for case, fields, shows_logo in PIPELINE_HEADER_CASES:
        stream = io.StringIO()
        _logger(tmp_path, stream).emit("pipeline_start", task="circpack", **fields)

        rendered = stream.getvalue()
        assert ("████" in rendered) is shows_logo, case
        if not shows_logo:
            assert "AGSCI" not in rendered, case
        assert "Agentic Scientist ·" not in rendered, case
        assert "Pipeline · circpack" in rendered, case

    # managed phase pipelines do not repeat the parent banner
    stream = io.StringIO()
    logger = _logger(tmp_path, stream)

    logger.emit(
        "pipeline_start", task="circpack", phases=["investigation", "ablations", "posthoc"],
        show_run_header=True,
    )
    logger.emit("pipeline_start", task="circpack", phases=[0, 1, 2], show_run_header=False)
    logger.emit("pipeline_start", task="circpack", phases=[0, 1], show_run_header=False)

    rendered = stream.getvalue()
    assert rendered.count(" █████╗") == 1
    assert rendered.count("Pipeline · circpack") == 3


def test_narrow_tty_uses_compact_gradient_logo(tmp_path: Path) -> None:
    class FdTtyStream(TtyStream):
        def fileno(self) -> int:
            return 99

    stream = FdTtyStream()
    with (
        patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=True),
        patch(
            "src.investigation.event_renderers.os.get_terminal_size",
            return_value=os.terminal_size((86, 24)),
        ),
    ):
        _logger(tmp_path, stream).emit(
            "pipeline_start", task="circpack", phases=["investigation"], show_run_header=True
        )

    rendered = stream.getvalue()
    assert " ███   ███  ████   ███  ███" in rendered
    assert "█████╗" not in rendered
    assert "\x1b[38;5;202m" in rendered
    assert "\x1b[38;5;75m" in rendered


def test_render_logo_variants_and_gradient_endpoints() -> None:
    # plain logo is a compact six-line wordmark
    lines = render_logo(colors=False)

    assert len(lines) == 6
    assert all(lines)
    assert max(terminal_width(line) for line in lines) <= 96
    assert all("\x1b" not in line for line in lines)

    # colored logo (full and compact) matches the shinka gradient endpoints
    for case, columns in (("full", None), ("compact", 86)):
        lines = render_logo(colors=True) if columns is None else render_logo(colors=True, columns=columns)

        assert lines[0].startswith("\x1b[38;5;202m"), case
        assert lines[-1].startswith("\x1b[38;5;75m"), case
        assert all(line.endswith("\x1b[0m") for line in lines), case

    # minimal colored logo keeps the gradient endpoints
    minimal = render_logo(colors=True, columns=26)

    assert minimal[0].startswith("\x1b[38;5;202mA")
    assert "\x1b[38;5;75mI" in minimal[0]
    assert minimal[0].endswith("\x1b[0m")

    # compact block logo is used below full width
    lines = render_logo(colors=False, columns=86)

    assert len(lines) == 5
    assert lines[0] == " ███   ███  ████   ███  ███"
    assert lines[-1] == "█   █  ███  ████   ███  ███"
    assert max(terminal_width(line) for line in lines) == 27

    # full logo is used at the exact width boundary
    lines = render_logo(colors=False, columns=87)

    assert len(lines) == 6
    assert max(terminal_width(line) for line in lines) == 87

    # minimal logo is used below the compact width
    assert render_logo(colors=False, columns=26) == ["AGSCI"]


# ---------------------------------------------------------------------------
# Review / paper gate events
# ---------------------------------------------------------------------------


def test_review_result_and_paper_gate_rendering_fields(tmp_path: Path) -> None:
    # review result rendering with and without score/decision
    for overall_score, decision, expected in (
        (3, "Reject", "Review result · score 3/10 · REJECT"),
        (None, None, "Review result · score unavailable · decision unavailable"),
    ):
        stream = io.StringIO()
        _logger(tmp_path, stream).emit("review_result", overall_score=overall_score, decision=decision)

        assert expected in stream.getvalue(), expected

    # a paper gate outcome carrying the heartbeat field still renders
    stream = io.StringIO()
    _logger(tmp_path, stream).emit(
        "paper_gate", gate="strategy", state="failed", paper_fields={"heartbeat": True}
    )

    assert "Strategy · failed" in stream.getvalue()

    # nested collection qualifiers are ignored
    stream = io.StringIO()
    _logger(tmp_path, stream).emit(
        "paper_gate", gate="preparation", state="started",
        paper_fields={"phase": ["private"], "round": {"secret": "value"}},
    )

    assert stream.getvalue() == "[PAPER] ⠋ Preparation · started\n"

    # scalar nested qualifiers are rendered
    for gate, paper_fields, qualifier in (
        ("preparation", {"phase": "grounding"}, "grounding"),
        ("repair", {"round": 2}, "round=2"),
    ):
        stream = io.StringIO()
        _logger(tmp_path, stream).emit("paper_gate", gate=gate, state="started", paper_fields=paper_fields)

        assert qualifier in stream.getvalue(), gate

    # a paper gate heartbeat is journaled without rendering
    journaled = _sub(tmp_path, "journaled")
    stream = io.StringIO()
    logger = _logger(journaled, stream, heartbeat_seconds=60, start_progress_thread=False)

    logger.emit("paper_gate", gate="strategy", state="started")
    before_heartbeat = stream.getvalue()
    logger.emit("paper_gate", gate="strategy", state="started", paper_fields={"heartbeat": True})

    assert _records(journaled)[-1]["fields"]["paper_fields"] == {"heartbeat": True}
    assert stream.getvalue() == before_heartbeat


def test_paper_gate_tty_spinner_and_heartbeat_elapsed(tmp_path: Path) -> None:
    # a paper gate event drives the spinner and renders a named outcome
    stream, clock = TtyStream(), [10.0]
    logger = _logger(_sub(tmp_path, "spinner"), stream, heartbeat_seconds=60, clock=clock)

    logger.emit("paper_gate", gate="evidence_audit", state="started")
    clock[0] = 11.0
    logger.poll_progress()
    logger.emit("paper_gate", gate="evidence_audit", state="completed")

    rendered = stream.getvalue()
    assert "Evidence Audit · started" not in rendered
    assert "Writing · evidence audit… 1s" in rendered
    assert "Evidence Audit · completed" in rendered

    # a paper gate heartbeat preserves the progress elapsed time
    stream, clock = TtyStream(), [10.0]
    with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True):
        logger = _logger(_sub(tmp_path, "elapsed"), stream, heartbeat_seconds=60, clock=clock)
        logger.emit("paper_gate", gate="strategy", state="started")
        clock[0] = 20.0
        logger.emit("paper_gate", gate="strategy", state="started", paper_fields={"heartbeat": True})
        clock[0] = 25.0
        logger.poll_progress()

    assert "Writing · strategy… 15s" in stream.getvalue()


def test_paper_gate_cost_summaries(tmp_path: Path) -> None:
    # a heartbeat preserves the cost baseline
    baseline = _sub(tmp_path, "baseline")
    logger = _logger(baseline)
    logger.emit("paper_gate", gate="strategy", state="started")
    _cost_record(baseline, "strategy-cost-before-heartbeat", "strategy", reported_cost_usd="0.40")
    logger.emit("paper_gate", gate="strategy", state="started", paper_fields={"heartbeat": True})
    _cost_record(baseline, "strategy-cost-after-heartbeat", "strategy", reported_cost_usd="0.20")
    logger.emit("paper_gate", gate="strategy", state="completed")

    assert float(_cost_summary(baseline)["fields"]["reported_cost_usd"]) == pytest.approx(0.60)

    # the checkpoint delta spans a skipped gate and omits a zero-cost gate
    skipped = _sub(tmp_path, "skipped")
    logger = _logger(skipped)
    logger.emit("paper_gate", gate="strategy", state="started")
    _cost_record(skipped, "strategy-cost", "strategy", reported_cost_usd="0.40")
    logger.emit("paper_gate", gate="strategy", state="completed")
    _cost_record(skipped, "between-gates-cost", "ranking", reported_cost_usd="0.10")
    logger.emit("paper_gate", gate="review", state="started")
    logger.emit("paper_gate", gate="review", state="skipped")
    logger.emit("paper_gate", gate="draft", state="started")
    _cost_record(skipped, "draft-cost-after-skip", "draft", reported_cost_usd="0.20")
    logger.emit("paper_gate", gate="draft", state="completed")
    logger.emit("paper_gate", gate="validation", state="started")
    logger.emit("paper_gate", gate="validation", state="completed")

    costs = [record for record in _records(skipped) if record["event"] == "cost_summary"]
    assert [record["fields"]["label"] for record in costs] == ["Strategy", "Draft"]
    assert [record["fields"]["reported_cost_usd"] for record in costs] == ["0.4", "0.3"]
    assert costs[-1]["fields"]["run_total_cost_usd"] == "0.7"

    # usage-only activity stays visible
    usage_only = _sub(tmp_path, "usage-only")
    stream = io.StringIO()
    logger = _logger(usage_only, stream)
    logger.emit("paper_gate", gate="references", state="started")
    _cost_record(
        usage_only, "unpriced-reference-cost", "references", model="unknown",
        billing_basis="usage_only", usage={"input_tokens": 100},
    )

    logger.emit("paper_gate", gate="references", state="completed")

    assert "Tokens · References · no usage · run 0 · 1 unreported" in stream.getvalue()


def test_paper_gate_durations_in_human_and_json(tmp_path: Path) -> None:
    # a completed gate reports duration, delta, run total and pipeline time
    completed_dir = _sub(tmp_path, "completed")
    stream, clock = io.StringIO(), [10.0]
    logger = _logger(completed_dir, stream, clock=clock, pipeline_started_at=0.0)
    logger.emit("paper_gate", gate="draft", state="started", paper_fields={"round": 1})
    _cost_record(
        completed_dir, "draft-cost", "draft", billing_basis="estimated", estimated_cost_usd="0.42",
        cost_basis="api-list-price-estimate", pricing_source="models.dev", pricing_status="priced",
        usage={"input_tokens": 1_200, "output_tokens": 350},
    )
    clock[0] = 100.0

    logger.emit("paper_gate", gate="draft", state="completed", paper_fields={"round": 1})

    rendered = stream.getvalue()
    assert "[PAPER] ✓ Draft · completed (round=1) · 1m 30s" in rendered
    assert "Tokens · Draft · 350 output · run 350 · Time 1m 40s" in rendered
    completed = next(
        record for record in _records(completed_dir)
        if record["event"] == "paper_gate" and record["fields"]["state"] == "completed"
    )
    assert completed["fields"]["elapsed"] == "1m 30s"
    assert completed["fields"]["elapsed_seconds"] == 90.0

    # a failed gate reports cost and duration
    failed_dir = _sub(tmp_path, "failed")
    stream, clock = io.StringIO(), [20.0]
    logger = _logger(failed_dir, stream, clock=clock)
    logger.emit("paper_gate", gate="evidence_audit", state="started")
    _cost_record(
        failed_dir, "failed-audit-cost", "evidence_audit", status="failed", reported_cost_usd="0.25",
        usage={"output_tokens": 80},
    )
    clock[0] = 25.0

    logger.emit("paper_gate", gate="evidence_audit", state="failed")

    assert "[PAPER] ✗ Evidence Audit · failed · 5s" in stream.getvalue()
    assert "Tokens · Evidence Audit · 80 output · run 80" in stream.getvalue()

    # the JSON renderer includes the computed duration
    stream, clock = io.StringIO(), [5.0]
    logger = _logger(_sub(tmp_path, "json"), stream, log_format="json", clock=clock)
    logger.emit("paper_gate", gate="figures", state="started")
    clock[0] = 70.0

    logger.emit("paper_gate", gate="figures", state="completed")

    record = json.loads(stream.getvalue().splitlines()[-1])
    assert record["fields"]["elapsed"] == "1m 5s"
    assert record["fields"]["elapsed_seconds"] == 65.0


# ---------------------------------------------------------------------------
# Cost summaries
# ---------------------------------------------------------------------------


def test_stage_end_cost_summaries(tmp_path: Path) -> None:
    # stage_end emits the running cost delta to terminal and journal
    running = _sub(tmp_path, "running")
    stream = io.StringIO()
    logger = _logger(running, stream)
    logger.emit("stage_start", stage="draft", label="Draft paper")
    _cost_record(
        running, "draft-cost", "draft", agent="claude", model="claude-opus", reported_cost_usd="0.42",
        usage={"input_tokens": 20, "output_tokens": 1_500},
    )

    logger.emit("stage_end", stage="draft", label="Draft paper", status="completed")

    assert "Tokens · Draft paper · 1.5k output · run 1.5k" in stream.getvalue()
    fields = _cost_summary(running)["fields"]
    assert fields["output_tokens"] == 1_500
    assert fields["run_output_tokens"] == 1_500
    assert fields["unreported_usage_events"] == 0
    assert fields["reported_cost_usd"] == "0.42"
    assert fields["run_reported_cost_usd"] == "0.42"

    # the parent pipeline cost summary includes cumulative runtime
    parent = _sub(tmp_path, "parent")
    stream, clock = io.StringIO(), [0.0]
    logger = EventLogger(
        log_format="human", verbose=False, heartbeat_seconds=0, results_dir=parent, worker_index=0,
        stream=stream, source="pipeline", renderer="pipeline", pipeline_started_at=0.0,
        monotonic_fn=lambda: clock[0],
    )
    logger.emit("phase_start", phase_name="task_drafting", label="Task drafting")
    _cost_record(
        parent, "draft-cost", "task_drafting", phase="task", billing_basis="estimated",
        estimated_cost_usd="0.44", cost_basis="api-list-price-estimate", pricing_source="models.dev",
        pricing_status="priced", usage={"output_tokens": 4_400},
    )
    clock[0] = 93_784.9

    logger.emit("phase_end", phase_name="task_drafting", label="Task drafting")

    assert "Tokens · Task drafting · 4.4k output · run 4.4k · Time 26h 3m" in stream.getvalue()
    fields = _cost_summary(parent)["fields"]
    assert fields["pipeline_elapsed_seconds"] == 93_784.9
    assert fields["pipeline_elapsed"] == "26h 3m"

    # a nested cost summary omits the pipeline runtime
    nested = _sub(tmp_path, "nested")
    stream = io.StringIO()
    logger = _logger(nested, stream)
    logger.emit("stage_start", stage="draft", label="Draft paper")
    _cost_record(nested, "nested-cost", "draft", agent="claude", model="claude-opus", reported_cost_usd="0.42")

    logger.emit("stage_end", stage="draft", label="Draft paper")

    cost_line = next(line for line in stream.getvalue().splitlines() if line.startswith("Tokens ·"))
    assert "Time" not in cost_line

    # stage_end derives the structured run total for a legacy summary
    legacy = _sub(tmp_path, "legacy")
    stream = io.StringIO()
    logger = _logger(legacy, stream)
    logger.emit("stage_start", stage="draft", label="Draft paper")
    metrics = legacy / "metrics"
    metrics.mkdir()
    (metrics / "cost_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reported_cost_usd": "0.42",
                "estimated_events": 0,
                "usage_only_events": 0,
                "unknown_events": 0,
                "usage": {"output_tokens": 42},
            }
        ),
        encoding="utf-8",
    )

    logger.emit("stage_end", stage="draft", label="Draft paper", status="completed")

    assert "Tokens · Draft paper · 42 output · run 42" in stream.getvalue()
    assert _cost_summary(legacy)["fields"]["run_total_cost_usd"] == "0.42"

    # stage_end includes estimated cost in terminal and journal
    estimated = _sub(tmp_path, "estimated")
    stream = io.StringIO()
    logger = _logger(estimated, stream)
    logger.emit("stage_start", stage="idea_generation", label="Idea generation")
    _cost_record(
        estimated, "idea-estimate", "idea_generation", phase="idea", provider="openai",
        billing_basis="estimated", reported_cost_usd=None, estimated_cost_usd="1.59",
        cost_basis="api-list-price-estimate", pricing_source="models.dev", pricing_status="priced",
        usage={"input_tokens": 40, "output_tokens": 1_590},
    )

    logger.emit("stage_end", stage="idea_generation", label="Idea generation", status="completed")

    assert "Tokens · Idea generation · 1.6k output · run 1.6k" in stream.getvalue()
    fields = _cost_summary(estimated)["fields"]
    assert fields["estimated_cost_usd"] == "1.59"
    assert fields["total_cost_usd"] == "1.59"
    assert fields["run_total_cost_usd"] == "1.59"

    # a corrupt cost summary never breaks scientific event logging
    corrupt = _sub(tmp_path, "corrupt")
    metrics = corrupt / "metrics"
    metrics.mkdir()
    (metrics / "cost_summary.json").write_text(
        '{"reported_cost_usd": "not-a-number", "usage_only_events": "bad"}\n', encoding="utf-8"
    )
    stream = io.StringIO()
    logger = _logger(corrupt, stream)

    logger.emit("stage_start", stage="draft")
    logger.emit("stage_end", stage="draft", status="completed")

    assert "draft · completed" in stream.getvalue()
    assert "Tokens ·" not in stream.getvalue()


def test_run_level_cost_summaries_for_finished_worker_and_parallel_end(tmp_path: Path) -> None:
    # the finished event emits the final run cost
    finished = _sub(tmp_path, "finished")
    stream = io.StringIO()
    logger = _logger(finished, stream)
    _cost_record(
        finished, "final-cost", "attempt", phase="run", agent="claude", model="claude-opus",
        reported_cost_usd="1.25", usage={"output_tokens": 12_500},
    )

    logger.emit("finished", elapsed="1m")

    assert "Tokens · final run · 12.5k output · run 12.5k" in stream.getvalue()

    # a worker logger does not report the shared global cost delta
    worker = _sub(tmp_path, "worker")
    stream = io.StringIO()
    logger = _logger(worker, stream, worker_index=1, start_progress_thread=False)
    _cost_record(
        worker, "other-worker-cost", "attempt", phase="run", agent="claude", model="claude",
        reported_cost_usd="0.25",
    )

    logger.emit("stage_end", stage="attempt")
    logger.close()

    assert "Tokens ·" not in stream.getvalue()

    # parallel_end reports the aggregate worker cost
    parallel = _sub(tmp_path, "parallel")
    stream = io.StringIO()
    _cost_record(
        parallel, "prior-cost", "draft", phase="task", agent="claude", model="claude",
        reported_cost_usd="1", usage={"output_tokens": 1_000},
    )
    logger = _logger(parallel, stream)
    logger.emit("parallel_start", workers=2)
    _cost_record(
        parallel, "parallel-cost", "attempt", phase="run", agent="claude", model="claude",
        reported_cost_usd="0.75", usage={"output_tokens": 750},
    )

    logger.emit("parallel_end", workers=2)
    logger.close()

    assert "Tokens · parallel campaign · 750 output · run 1.8k" in stream.getvalue()


# ---------------------------------------------------------------------------
# configure_event_logging / inherited pipeline start
# ---------------------------------------------------------------------------


def test_configure_event_logging_cost_roots_and_inherited_pipeline_start(
    tmp_path: Path, monkeypatch, fresh_cost_root
) -> None:
    # sequential top-level runs get distinct cost roots
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.delenv("AGENTIC_COST_ROOT", raising=False)

    _configure_event_logging(first)
    assert os.environ["AGENTIC_COST_ROOT"] == str(first.absolute())
    runtime_module.shutdown_event_logging()
    assert "AGENTIC_COST_ROOT" not in os.environ
    _configure_event_logging(second)
    assert os.environ["AGENTIC_COST_ROOT"] == str(second.absolute())
    runtime_module.shutdown_event_logging()

    # only a valid pipeline start is inherited from the environment
    for value, expected in (
        ("12.5", 12.5), ("not-a-number", None), ("nan", None), ("inf", None), ("-1", None),
    ):
        with monkeypatch.context() as m:
            m.setenv(runtime_module.PIPELINE_STARTED_AT_ENV, value)

            _configure_event_logging(tmp_path / f"inherit-{value}")

            assert runtime_module._EVENT_LOGGER is not None, value
            assert runtime_module._EVENT_LOGGER.pipeline_started_at == expected, value
            runtime_module.shutdown_event_logging()

    # future monotonic or wall-clock values are rejected
    for value in ("100.001", "1800000000"):
        with monkeypatch.context() as m:
            m.setenv(runtime_module.PIPELINE_STARTED_AT_ENV, value)
            m.setattr(runtime_module.time, "monotonic", lambda: 100.0)

            assert runtime_module._inherited_pipeline_started_at() is None, value


# ---------------------------------------------------------------------------
# Started lines: TTY spinner vs redirected output
# ---------------------------------------------------------------------------


def _emit_started_and_completed_steps(logger: EventLogger, clock: list[float]) -> None:
    logger.emit("step_start", step="idea_generation", status="started", cycle=1, total_cycles=2)
    logger.start_progress("idea_generation", started_at=clock[0], cycle=1, total_cycles=2)
    clock[0] += 41.0
    logger.poll_progress()
    logger.stop_progress("idea_generation")
    logger.emit("step_end", step="idea_generation", status="completed", exit=0, elapsed="41s")
    logger.emit("task_drafting_checkpoint", gate="design", state="started", label="Generate task design")
    clock[0] += 5.0
    logger.poll_progress()
    logger.emit(
        "task_drafting_checkpoint", gate="design", state="completed", label="Generate task design",
        elapsed="4m 57s",
    )
    logger.emit("preparation_step", step="smoke", label="Search smoke", state="started")
    clock[0] += 5.0
    logger.poll_progress()
    logger.emit("preparation_step", step="smoke", label="Search smoke", state="completed")
    logger.emit("paper_gate", gate="references", state="started", phase="initial")
    clock[0] += 5.0
    logger.poll_progress()
    logger.emit("paper_gate", gate="references", state="completed", phase="initial")


def test_started_lines_tty_spinner_vs_redirected_output(tmp_path: Path) -> None:
    # the TTY spinner replaces permanent started lines
    tty_dir = _sub(tmp_path, "tty")
    stream: io.StringIO = TtyStream()
    clock = [10.0]
    with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True):
        logger = _logger(tty_dir, stream, heartbeat_seconds=60, clock=clock)
        _emit_started_and_completed_steps(logger, clock)
        logger.close()

    permanent = [
        line for line in stream.getvalue().replace("\r\x1b[2K", "\n").split("\n")
        if line and not line.startswith(tuple(SPINNER_FRAMES))
    ]
    assert permanent == [
        "  done  ·  41s",
        "  Generate task design · complete · 4m 57s",
        "  Search smoke · complete",
        "[PAPER] ✓ References · completed (initial) · 5s",
    ]
    assert "⠋ 1/2 · Generating idea… 41s" in stream.getvalue()
    assert "Generate task design 5s" in stream.getvalue()
    assert "Writing · references (initial)… 5s" in stream.getvalue()
    started = [
        record["fields"].get("state", record["fields"].get("status"))
        for record in _records(tty_dir)
        if record["event"] in {"step_start", "task_drafting_checkpoint", "preparation_step", "paper_gate"}
    ]
    assert started.count("started") == 4

    # redirected output keeps permanent started lines
    stream, clock = io.StringIO(), [10.0]
    with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True):
        logger = _logger(_sub(tmp_path, "redirected"), stream, heartbeat_seconds=60, clock=clock)
        _emit_started_and_completed_steps(logger, clock)
        logger.close()

    rendered = stream.getvalue()
    assert "\r" not in rendered
    assert "  Generating idea…\n" in rendered
    assert "  Generate task design…\n" in rendered
    assert "  Search smoke…\n" in rendered
    assert "  Generate task design · complete · 4m 57s\n" in rendered
    assert "[PAPER] ⠋ References · started (initial)\n" in rendered
    assert "[PAPER] ✓ References · completed (initial) · 5s\n" in rendered


# ---------------------------------------------------------------------------
# Preparation milestones (shared terminal feedback, no cost accounting)
# ---------------------------------------------------------------------------


def _prep(logger: EventLogger, step: str, label: str, state: str, **fields: Any) -> None:
    logger.emit("preparation_step", step=step, label=label, state=state, **fields)


def test_preparation_milestones_console_variants(tmp_path: Path) -> None:
    # default console shows milestones and a durable heartbeat
    default = _sub(tmp_path, "default")
    stream, clock = io.StringIO(), [0.0]
    logger = _logger(default, stream, heartbeat_seconds=60, clock=clock)
    _prep(logger, "runtime_copy", "Copying runtime", "started")
    clock[0] = 61.0
    logger.poll_progress()
    _prep(logger, "runtime_copy", "Copying runtime", "completed", elapsed="1m 1s", elapsed_seconds=61.0)

    assert "Copying runtime…" in stream.getvalue()
    assert "Copying runtime 1m 1s · still running" in stream.getvalue()
    assert "Copying runtime · complete · 1m 1s" in stream.getvalue()
    records = _records(default)
    assert [record["event"] for record in records] == ["preparation_step", "heartbeat", "preparation_step"]
    assert records[1]["fields"]["step"] == "runtime_copy"
    assert all(record["phase"] is None for record in records)
    logger.close()

    # disabled heartbeats retain the milestones
    disabled = _sub(tmp_path, "disabled")
    stream, clock = io.StringIO(), [0.0]
    logger = _logger(disabled, stream, heartbeat_seconds=0, clock=clock)
    _prep(logger, "smoke", "Search smoke", "started")
    clock[0] = 120.0
    logger.poll_progress()
    _prep(logger, "smoke", "Search smoke", "completed")
    assert "Search smoke…" in stream.getvalue()
    assert "Search smoke · complete" in stream.getvalue()
    assert not any(record["event"] == "heartbeat" for record in _records(disabled))
    logger.close()

    # the JSON console preserves structured milestones and heartbeat
    json_dir = _sub(tmp_path, "json")
    stream, clock = io.StringIO(), [0.0]
    logger = _logger(json_dir, stream, log_format="json", heartbeat_seconds=60, clock=clock)
    _prep(logger, "smoke", "Search smoke", "started")
    clock[0] = 61.0
    logger.poll_progress()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [record["event"] for record in records] == ["preparation_step", "heartbeat"]
    assert records == _records(json_dir)
    logger.close()

    # a TTY refreshes in place and still journals the heartbeat
    tty_dir = _sub(tmp_path, "tty")
    stream, clock = TtyStream(), [0.0]
    logger = _logger(tty_dir, stream, heartbeat_seconds=60, clock=clock)
    _prep(logger, "runtime_copy", "Copying runtime", "started")
    clock[0] = 61.0
    logger.poll_progress()
    assert "\r\x1b[2K" in stream.getvalue()
    assert "still running" not in stream.getvalue()
    assert _records(tty_dir)[-1]["event"] == "heartbeat"
    logger.close()


def test_preparation_heartbeat_nesting_outcomes_and_cost_isolation(tmp_path: Path, monkeypatch) -> None:
    # nested preparation restores the outer step, then the campaign heartbeat
    nested = _sub(tmp_path, "nested")
    clock = [0.0]
    logger = _logger(nested, heartbeat_seconds=60, clock=clock)
    logger.start_progress("attempt", started_at=0.0, label="Investigation")
    clock[0] = 10.0
    _prep(logger, "smoke", "Search smoke", "started")
    clock[0] = 20.0
    _prep(logger, "runtime_copy", "Copying runtime", "started")
    _prep(logger, "runtime_copy", "Copying runtime", "completed")
    clock[0] = 71.0
    logger.poll_progress()
    assert _records(nested)[-1]["fields"]["label"] == "Search smoke"
    assert _records(nested)[-1]["fields"]["elapsed_seconds"] == 61.0
    _prep(logger, "smoke", "Search smoke", "completed")
    logger.poll_progress()
    assert _records(nested)[-1]["fields"]["label"] == "Investigation"
    assert _records(nested)[-1]["fields"]["elapsed_seconds"] == 71.0
    logger.close()

    # a completed/failed outcome stops the heartbeat
    for state in ("completed", "failed"):
        stream, clock = io.StringIO(), [0.0]
        logger = _logger(_sub(tmp_path, f"outcome-{state}"), stream, heartbeat_seconds=60, clock=clock)
        _prep(logger, "smoke", "Search smoke", "started")
        _prep(logger, "smoke", "Search smoke", state)
        before = stream.getvalue()
        clock[0] = 61.0
        logger.poll_progress()
        assert stream.getvalue() == before, state
        assert ("FAILED" if state == "failed" else "complete") in before, state
        logger.close()

    # terminal events discard a suspended preparation step
    for event in ("phase_end", "interrupted", "campaign_failed"):
        terminal = _sub(tmp_path, f"terminal-{event}")
        clock = [0.0]
        logger = _logger(terminal, heartbeat_seconds=60, clock=clock)
        logger.start_progress("attempt", started_at=0.0, label="Investigation")
        _prep(logger, "smoke", "Search smoke", "started")
        logger.emit(event)
        _prep(logger, "smoke", "Search smoke", "failed")
        clock[0] = 61.0
        logger.poll_progress()
        assert not any(record["event"] == "heartbeat" for record in _records(terminal)), event
        logger.close()

    # preparation does not capture costs
    costs = _sub(tmp_path, "costs")
    stream, clock = io.StringIO(), [0.0]
    logger = _logger(costs, stream, heartbeat_seconds=60, clock=clock)

    def unexpected_cost_read() -> None:
        pytest.fail("Preparation changed cost accounting")

    monkeypatch.setattr(logger, "_cost_snapshot", unexpected_cost_read)
    monkeypatch.setattr(logger, "_capture_task_checkpoint_cost_baseline", unexpected_cost_read)
    for state in ("started", "completed"):
        _prep(logger, "smoke", "Search smoke", state)
    assert [record["event"] for record in _records(costs)] == ["preparation_step", "preparation_step"]
    assert "Search smoke" in stream.getvalue()
    logger.close()
