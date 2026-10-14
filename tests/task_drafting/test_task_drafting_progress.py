"""Task-drafting progress: transport records, cost fields, human renderer, shell helpers."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
from decimal import Decimal
from pathlib import Path

import pytest

import src.investigation.event_logging as event_logging_module
from src.investigation.event_logging import EventLogger
from src.task_drafting.progress import (
    TRANSPORT_PREFIX,
    build_cost_fields,
    build_record,
    emit_event,
    extract_transport_events,
    main as progress_main,
)

ROOT = Path(__file__).resolve().parents[2]


def _cost_summary(
    *, reported: str = "0", estimated: str = "0", estimated_events: int = 0,
    output_tokens: int | None = 0,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "reported_cost_usd": reported,
        "estimated_cost_usd": estimated,
        "total_cost_usd": format(Decimal(reported) + Decimal(estimated), "f"),
        "estimated_events": estimated_events,
        "usage_only_events": 0,
        "unknown_events": 0,
    }
    if output_tokens is not None:
        summary["usage"] = {"output_tokens": output_tokens}
    return summary


def _write_cost_summary(root: Path, **kwargs) -> None:
    metrics = root / "metrics"
    metrics.mkdir(exist_ok=True)
    (metrics / "cost_summary.json").write_text(json.dumps(_cost_summary(**kwargs)), encoding="utf-8")


def _human_logger(tmp_path: Path, stream: io.StringIO, **kwargs) -> EventLogger:
    return EventLogger(
        log_format="human", verbose=False, heartbeat_seconds=0, results_dir=tmp_path,
        worker_index=0, stream=stream, start_progress_thread=False, **kwargs,
    )


def _cost_lines(stream: io.StringIO) -> list[str]:
    return [line for line in stream.getvalue().splitlines() if line.startswith("Tokens ·")]


def _checkpoint(logger: EventLogger, gate: str, state: str, label: str, **fields) -> None:
    logger.emit("task_drafting_checkpoint", gate=gate, state=state, label=label, **fields)


# --- transport ------------------------------------------------------------------------------


_CLI_TYPED_FIELDS: list[tuple[str, list[str], dict[str, object]]] = [
    (
        "typed-failure-kind",
        [
            "--gate", "qualification", "--state", "failed",
            "--label", "Full production search qualification",
            "--verdict", "fail", "--failure-kind", "infrastructure_failure",
        ],
        {"verdict": "fail", "failure_kind": "infrastructure_failure"},
    ),
    (
        "infrastructure-retry-attempt",
        [
            "--gate", "smoke", "--state", "retrying",
            "--label", "Evaluator smoke tests · infrastructure retry",
            "--infrastructure-attempt", "2", "--max-infrastructure-attempts", "3",
        ],
        {"infrastructure_attempt": 2, "max_infrastructure_attempts": 3},
    ),
]


def test_transport_round_trips_typed_records_and_rejects_untrusted_or_foreign_sessions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # test_wrapper_transport_round_trips_validated_task_progress
    stream = io.StringIO()
    emit_event(
        gate="design_review",
        state="completed",
        fields={"round": 2, "current": 1, "total": 3, "verdict": "pass"},
        transport="wrapper",
        stream=stream,
        session_id="task-test",
        auth_token="task-secret",
        timestamp_fn=lambda: "2026-08-29T08:00:00.000Z",
    )
    encoded = stream.getvalue()
    assert encoded.startswith(TRANSPORT_PREFIX)
    retained, records = extract_transport_events(
        encoded, expected_session_id="task-test", auth_token="task-secret"
    )
    assert retained == ""
    expected = {
        "timestamp": "2026-08-29T08:00:00.000Z",
        "source": "task_drafting",
        "event": "task_drafting_checkpoint",
        "session_id": "task-test",
        "gate": "design_review",
        "state": "completed",
        "fields": {"current": 1, "round": 2, "total": 3, "verdict": "pass"},
    }
    assert len(records) == 1
    assert records[0] == build_record(
        gate="design_review",
        state="completed",
        fields={"round": 2, "current": 1, "total": 3, "verdict": "pass"},
        session_id="task-test",
        auth_token="task-secret",
        timestamp_fn=lambda: "2026-08-29T08:00:00.000Z",
    )
    assert {key: records[0][key] for key in expected} == expected

    # test_progress_cli_emits_typed_fields
    for name, arguments, expected_fields in _CLI_TYPED_FIELDS:
        with monkeypatch.context() as patch:
            patch.setenv("AGENTIC_TASK_PROGRESS_TRANSPORT", "wrapper")
            patch.setenv("TASK_PROGRESS_SESSION_ID", "task-test")
            patch.delenv("TASK_PROGRESS_AUTH_TOKEN", raising=False)
            assert progress_main(["emit", *arguments]) == 0, name
        retained, records = extract_transport_events(
            capsys.readouterr().out, expected_session_id="task-test"
        )
        assert retained == "", name
        for key, value in expected_fields.items():
            assert records[0]["fields"][key] == value, name

    # test_transport_rejects_untrusted_or_oversized_records
    forged = {
        "timestamp": "2026-08-29T08:00:00.000Z",
        "source": "task_drafting",
        "event": "task_drafting_checkpoint",
        "session_id": "task-test",
        "gate": "not-a-gate",
        "state": "completed",
        "fields": {},
    }
    output = f"before\n{TRANSPORT_PREFIX}{json.dumps(forged)}\nafter\n"
    retained, records = extract_transport_events(output)
    assert records == []
    assert retained == output

    # test_transport_rejects_record_from_another_task_session
    stream = io.StringIO()
    emit_event(
        gate="scaffold",
        state="completed",
        fields={"label": "Forged scaffold"},
        transport="wrapper",
        stream=stream,
        session_id="agent-controlled-session",
        auth_token="agent-secret",
    )
    retained, records = extract_transport_events(
        stream.getvalue(), expected_session_id="host-controlled-session", auth_token="host-secret"
    )
    assert retained == stream.getvalue()
    assert records == []

    # test_resource_gate_is_supported_by_terminal_transport
    record = build_record(
        gate="resource_preflight", state="failed", fields={"label": "GPU allocation"}, session_id="test"
    )
    assert record["gate"] == "resource_preflight"


# --- cost fields ----------------------------------------------------------------------------


def test_cost_fields_report_task_delta_cumulative_run_cost_and_clamp_negative_deltas() -> None:
    # test_cost_fields_report_task_delta_and_cumulative_run_cost
    summary = {
        "reported_cost_usd": "1.25",
        "estimated_cost_usd": "2.50",
        "total_cost_usd": "3.75",
        "estimated_events": 3,
        "usage_only_events": 1,
        "unknown_events": 0,
        "usage": {"input_tokens": 90_000, "output_tokens": 3_750},
        "unreported_usage_events": 1,
    }
    baseline = {
        "reported_cost_usd": "0.25",
        "estimated_cost_usd": "0.50",
        "estimated_events": 1,
        "usage_only_events": 0,
        "unknown_events": 0,
        "output_tokens": 750,
        "unreported_usage_events": 0,
    }
    fields = build_cost_fields(summary, baseline)
    assert fields["task_cost_usd"] == "3.00"
    assert fields["run_cost_usd"] == "3.75"
    assert fields["task_reported_cost_usd"] == "1.00"
    assert fields["task_estimated_cost_usd"] == "2.00"
    assert fields["task_usage_only_events"] == 1
    assert fields["task_output_tokens"] == 3_000
    assert fields["run_output_tokens"] == 3_750
    assert fields["task_unreported_usage_events"] == 1
    assert fields["cost_message"] == "Tokens · task drafting · 3.0k output · run 3.8k · 1 unreported"

    # test_cost_fields_clamp_negative_deltas
    fields = build_cost_fields(
        _cost_summary(output_tokens=None),
        {
            "reported_cost_usd": Decimal("1"),
            "estimated_cost_usd": Decimal("1"),
            "estimated_events": 1,
            "usage_only_events": 1,
            "unknown_events": 1,
            "output_tokens": 10,
            "unreported_usage_events": 1,
        },
    )
    assert fields["task_cost_usd"] == "0.00"
    assert fields["task_usage_only_events"] == 0
    assert fields["task_output_tokens"] == 0
    assert fields["task_unreported_usage_events"] == 0


# --- human renderer -------------------------------------------------------------------------


def _subdir(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


def test_human_renderer_shows_rolling_task_costs_only_at_agent_checkpoints(tmp_path: Path) -> None:
    # test_human_renderer_shows_checkpoint_and_cumulative_cost
    root = _subdir(tmp_path, "cumulative")
    stream = io.StringIO()
    clock = [0.0]
    logger = _human_logger(root, stream, pipeline_started_at=0.0, monotonic_fn=lambda: clock[0])
    logger.emit("step_start", step="task_draft")
    _write_cost_summary(root, estimated="0.42", estimated_events=1, output_tokens=420)

    clock[0] = 93_784.9
    _checkpoint(
        logger, "design_review", "completed", "Design review round 2 · reviewer 1/3",
        verdict="pass", elapsed="2m 4s", cost_checkpoint=True,
        cost_message="forged subprocess cost", pipeline_elapsed="99:99:99",
        pipeline_elapsed_seconds=999_999,
    )
    logger.close()

    rendered = stream.getvalue()
    assert "Design review round 2 · reviewer 1/3 · PASS · 2m 4s" in rendered
    assert "Tokens · task drafting · 420 output · run 420 · Time 26h 3m" in rendered
    assert "forged subprocess cost" not in rendered

    # test_human_renderer_reports_rolling_task_checkpoint_deltas
    root = _subdir(tmp_path, "rolling")
    stream = io.StringIO()
    logger = _human_logger(root, stream)
    logger.emit("step_start", step="task_draft")

    _checkpoint(logger, "agent_call", "started", "First agent call")
    _write_cost_summary(root, estimated="0.40", estimated_events=1, output_tokens=400)
    _checkpoint(logger, "agent_call", "completed", "First agent call", cost_checkpoint=True)

    _write_cost_summary(root, estimated="0.50", estimated_events=2, output_tokens=500)
    _checkpoint(logger, "design_review", "started", "Second agent call")
    _write_cost_summary(root, estimated="0.64", estimated_events=3, output_tokens=640)
    _checkpoint(logger, "design_review", "completed", "Second agent call", cost_checkpoint=True)

    _checkpoint(logger, "design_repair", "started", "No-cost repair")
    _checkpoint(logger, "design_repair", "completed", "No-cost repair", cost_checkpoint=True)
    logger.emit("step_end", step="task_draft")
    logger.close()

    assert _cost_lines(stream) == [
        "Tokens · task drafting · 400 output · run 400",
        "Tokens · task drafting · 140 output · run 640",
        "Tokens · task drafting · 0 output · run 640",
        "Tokens · task draft · 640 output · run 640",
    ]
    records = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    checkpoint_costs = [
        record["fields"]["task_estimated_cost_usd"]
        for record in records
        if record["event"] == "task_drafting_checkpoint"
        and "task_estimated_cost_usd" in record["fields"]
    ]
    assert checkpoint_costs == ["0.40", "0.14", "0.00"]

    # test_human_renderer_omits_cost_at_non_agent_checkpoint
    root = _subdir(tmp_path, "non-agent")
    _write_cost_summary(root, reported="0.25", output_tokens=None)
    stream = io.StringIO()
    logger = _human_logger(root, stream)
    _checkpoint(logger, "design_schema", "completed", "Design schema")
    logger.close()
    assert "Design schema · complete" in stream.getvalue()
    assert "Tokens ·" not in stream.getvalue()

    # test_human_renderer_keeps_cost_visible_for_unsuccessful_checkpoint
    for state in ("failed", "retrying"):
        root = _subdir(tmp_path, f"unsuccessful-{state}")
        stream = io.StringIO()
        logger = _human_logger(root, stream)
        logger.emit("step_start", step="task_draft")
        _write_cost_summary(root, reported="0.40", output_tokens=400)
        _checkpoint(logger, "agent_call", "completed", "Previous agent call", cost_checkpoint=True)
        _checkpoint(logger, "design_repair", "started", "Design repair")
        _write_cost_summary(root, reported="0.65", output_tokens=650)
        _checkpoint(logger, "design_repair", state, "Design repair", cost_checkpoint=True, synthetic=True)
        logger.close()
        assert "Tokens · task drafting · 250 output · run 650" in stream.getvalue(), state


def test_stalled_or_stale_cost_storage_never_blocks_or_advances_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # test_stalled_cost_storage_does_not_block_checkpoint_rendering
    root = _subdir(tmp_path, "stalled")
    release = threading.Event()
    stream = io.StringIO()
    logger = _human_logger(root, stream)

    def stalled_summary(_root: Path) -> dict[str, object]:
        release.wait(5)
        return {}

    started = time.monotonic()
    with monkeypatch.context() as patch:
        patch.setattr("src.investigation.event_logging.load_summary", stalled_summary)
        _checkpoint(logger, "design_review", "completed", "Design review", cost_checkpoint=True)
    elapsed = time.monotonic() - started
    release.set()
    logger.close()

    assert elapsed < 1.5
    assert "Design review · complete" in stream.getvalue()
    assert "Tokens ·" not in stream.getvalue()

    # test_stale_cost_summary_is_not_rendered_or_advanced
    root = _subdir(tmp_path, "stale")
    stream = io.StringIO()
    logger = _human_logger(root, stream)
    logger.emit("step_start", step="task_draft")
    _write_cost_summary(root, estimated="0.40", estimated_events=1, output_tokens=400)
    _checkpoint(logger, "agent_call", "completed", "First agent call", cost_checkpoint=True)

    _write_cost_summary(root, estimated="0.64", estimated_events=2, output_tokens=640)
    release = threading.Event()
    original_load_summary = event_logging_module.load_summary

    def stalled_original(summary_root: Path) -> dict[str, object]:
        release.wait(5)
        return original_load_summary(summary_root)

    with monkeypatch.context() as patch:
        patch.setattr(event_logging_module, "load_summary", stalled_original)
        _checkpoint(logger, "design_review", "completed", "Delayed agent call", cost_checkpoint=True)
        release.set()
        assert logger._task_cost_reader is not None
        logger._task_cost_reader.join(1)
    _checkpoint(logger, "design_review", "started", "Recovered agent call")
    _write_cost_summary(root, estimated="0.70", estimated_events=3, output_tokens=700)
    _checkpoint(logger, "design_review", "completed", "Recovered agent call", cost_checkpoint=True)
    logger.close()

    assert _cost_lines(stream) == [
        "Tokens · task drafting · 400 output · run 400",
        "Tokens · task drafting · 60 output · run 700",
    ]


# --- shell transport (task_progress.sh) -----------------------------------------------------


def _write_summary_file(path: Path, estimated: str, estimated_events: int, output_tokens: int) -> None:
    path.write_text(
        json.dumps(_cost_summary(estimated=estimated, estimated_events=estimated_events, output_tokens=output_tokens)),
        encoding="utf-8",
    )


def _run_progress_shell(
    script: str, *args: str, env: dict[str, str] | None = None, timeout: int = 10, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script, "bash", str(ROOT), *args],
        capture_output=True, text=True, timeout=timeout, cwd=ROOT,
        env=os.environ | (env or {}), check=check,
    )


def test_shell_progress_drops_hung_telemetry_and_reports_rolling_direct_deltas(
    tmp_path: Path,
) -> None:
    # test_shell_progress_helper_drops_hung_telemetry
    root = _subdir(tmp_path, "hung")
    fake_python = root / "hung-python"
    fake_python.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    fake_python.chmod(0o755)
    script = """
source "$1/src/scripts/lib/task_progress.sh"
PYTHON_BIN="$2"
TASK_PROGRESS_TIMEOUT_SECONDS=1
task_progress_emit workflow started "Task workflow"
"""
    started = time.monotonic()
    result = _run_progress_shell(script, str(fake_python), timeout=5, check=False)
    assert result.returncode == 0
    assert time.monotonic() - started < 4

    # test_direct_progress_reports_rolling_task_checkpoint_deltas
    root = _subdir(tmp_path, "rolling")
    metrics = root / "metrics"
    metrics.mkdir()
    first_summary = root / "first-summary.json"
    second_start_summary = root / "second-start-summary.json"
    second_summary = root / "second-summary.json"
    _write_summary_file(metrics / "cost_summary.json", "0", 0, 0)
    _write_summary_file(first_summary, "0.40", 1, 400)
    _write_summary_file(second_start_summary, "0.50", 2, 500)
    _write_summary_file(second_summary, "0.64", 3, 640)
    script = """
source "$1/src/scripts/lib/task_progress.sh"
PYTHON_BIN="$2"
export AGENTIC_TASK_PROGRESS_TRANSPORT=direct
export AGENTIC_COST_ROOT="$3"
OUTPUT_DIR="$3"
task_progress_initialize
task_progress_emit agent_call started "First agent call"
cp -- "$4" "$3/metrics/cost_summary.json"
task_progress_emit agent_call completed "First agent call" --include-cost
cp -- "$5" "$3/metrics/cost_summary.json"
task_progress_emit design_review started "Second agent call"
cp -- "$6" "$3/metrics/cost_summary.json"
task_progress_emit design_review completed "Second agent call" --include-cost
"""
    result = _run_progress_shell(
        script, sys.executable, str(root), str(first_summary), str(second_start_summary),
        str(second_summary),
    )
    cost_lines = [line for line in result.stdout.splitlines() if line.startswith("Tokens ·")]
    assert cost_lines == [
        "Tokens · task drafting · 400 output · run 400",
        "Tokens · task drafting · 140 output · run 640",
    ]
    assert "AGENTIC_SCIENTIST_TASK_COST_BASELINE" not in result.stdout

    # test_direct_progress_suppresses_delta_after_baseline_refresh_failure
    root = _subdir(tmp_path, "baseline-failure")
    metrics = root / "metrics"
    metrics.mkdir()
    first_summary = root / "first-summary.json"
    second_summary = root / "second-summary.json"
    _write_summary_file(metrics / "cost_summary.json", "0", 0, 0)
    _write_summary_file(first_summary, "0.40", 1, 400)
    _write_summary_file(second_summary, "0.64", 2, 640)
    flaky_python = root / "flaky-python"
    flaky_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "src.task_drafting.progress" ] \
    && [ "${3:-}" = "baseline" ]; then
  count=0
  [ ! -f "$BASELINE_COUNT_FILE" ] || read -r count < "$BASELINE_COUNT_FILE"
  count=$((count + 1))
  printf '%s\n' "$count" > "$BASELINE_COUNT_FILE"
  [ "$count" -ne 2 ] || exit 1
fi
exec "$REAL_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    flaky_python.chmod(0o755)
    script = """
source "$1/src/scripts/lib/task_progress.sh"
PYTHON_BIN="$2"
export AGENTIC_TASK_PROGRESS_TRANSPORT=direct
export AGENTIC_COST_ROOT="$3"
OUTPUT_DIR="$3"
task_progress_initialize
task_progress_emit agent_call started "First agent call"
cp -- "$4" "$3/metrics/cost_summary.json"
task_progress_emit agent_call completed "First agent call" --include-cost
task_progress_emit design_review started "Second agent call"
cp -- "$5" "$3/metrics/cost_summary.json"
task_progress_emit design_review completed "Second agent call" --include-cost
"""
    result = _run_progress_shell(
        script, str(flaky_python), str(root), str(first_summary), str(second_summary),
        env={"BASELINE_COUNT_FILE": str(root / "baseline-count"), "REAL_PYTHON": sys.executable},
    )
    cost_lines = [line for line in result.stdout.splitlines() if line.startswith("Tokens ·")]
    assert cost_lines == ["Tokens · task drafting · 240 output · run 640"]


# --- resource preflight (task_resource_preflight.sh) ----------------------------------------


_PREFLIGHT_CASES = [
    ("gpu_allocation_quarantined", 4, 1),
    ("gpu_allocation_invalid", 4, 1),
    ("gpu_allocation_busy gpu_allocation_busy gpu_allocation_busy", 4, 3),
    ("gpu_allocation_busy passed", 0, 2),
    ("passed", 0, 1),
]


def test_preflight_retries_only_busy_allocation_and_precedes_design_agent(tmp_path: Path) -> None:
    # test_preflight_retries_only_busy_allocation
    for index, (categories, expected_exit, expected_attempts) in enumerate(_PREFLIGHT_CASES):
        root = _subdir(tmp_path, f"case-{index}")
        result = subprocess.run(
            ["bash", "-c", '''
set -eu
source "$HELPER"
PYTHON_BIN=fake_python
SMOKE_MAX_INFRASTRUCTURE_ATTEMPTS=3
fake_python() {
  if [ "$1" != -m ]; then "$REAL_PYTHON" "$@"; return; fi
  local count=0 category
  [ ! -f attempts ] || count=$(cat attempts)
  count=$((count + 1))
  printf '%s' "$count" > attempts
  read -ra values <<< "$CATEGORIES"
  category=${values[$((count - 1))]}
  printf '{"category":"%s"}\n' "$category"
  [ "$category" = passed ]
}
task_progress_emit() { printf 'event %s %s\n' "$1" "$2"; }
sleep() { :; }
if task_resource_preflight; then printf 'design started\n'; else exit "$?"; fi
'''],
            cwd=root,
            env=os.environ | {
                "HELPER": str(ROOT / "src/scripts/lib/task_resource_preflight.sh"),
                "REAL_PYTHON": sys.executable,
                "CATEGORIES": categories,
            },
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == expected_exit, (categories, result.stderr)
        assert int((root / "attempts").read_text()) == expected_attempts, categories
        assert ("design started" in result.stdout) == (expected_exit == 0), categories
        if expected_exit:
            assert "event resource_preflight failed" in result.stdout, categories

    # test_resource_check_precedes_design_agent
    script = (ROOT / "src/scripts/task.sh").read_text()
    assert script.index("task_resource_preflight || exit 4") < script.index("run_design_workflow\n")
