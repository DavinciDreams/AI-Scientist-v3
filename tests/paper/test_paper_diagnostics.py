"""Paper diagnostics archives, audit-hash stability, and the paper event bridge."""

from __future__ import annotations

import io
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import src.orchestrator.paper_event_bridge as paper_event_bridge
from src.orchestrator.paper_diagnostics import (
    archive_round,
    main,
    reference_repair_context,
    record_reference_round,
)
from src.orchestrator.paper_event_bridge import (
    TRANSPORT_PREFIX,
    emit_event,
    extract_transport_events,
    stream_private_diagnostics,
)
from src.orchestrator.paper_evidence_audit import audit_input_sha256, validation_input_sha256
from src.orchestrator.reference_query_plan import VALID_QUERY_CATEGORIES

ROOT = Path(__file__).resolve().parents[2]
STAGE = "postdraft_reference_query_plan"
TIMESTAMP = "2026-08-17T10:00:00.000Z"


# ── reference repair diagnostics ──


def write_failure(root: Path, error: str = "invalid category") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{STAGE}.json").write_text('{"category":"wrong"}')
    (root / f"{STAGE}.md").write_text("Prior plan")
    (root / f"{STAGE}_validation_feedback.json").write_text(
        json.dumps({"hard_errors": [error], "generated_at": "changes each invocation"})
    )


def test_reference_repair_receives_previous_plan_and_repeated_invalid_plan_stops_and_archives(
    tmp_path: Path,
) -> None:
    write_failure(tmp_path)
    # the repair context carries the previous plan and exact errors
    context = reference_repair_context(tmp_path, STAGE)
    assert '{"category":"wrong"}' in context
    assert "invalid category" in context
    assert "baseline_comparator" in context
    # a repeated invalid plan stops and archives both rounds
    assert not record_reference_round(tmp_path, STAGE, 0)
    assert record_reference_round(tmp_path, STAGE, 1)
    archive = tmp_path / f"paper_diagnostics/{STAGE}/round_0000"
    assert (archive / f"attempt_001/{STAGE}.json").is_file()
    assert (tmp_path / f"{STAGE}.json").is_file()


def test_changed_error_or_plan_permits_next_repair_and_new_session_does_not_inherit_failure(
    tmp_path: Path,
) -> None:
    # a changed error or plan permits the next reference repair
    root = tmp_path / "changed"
    write_failure(root)
    record_reference_round(root, STAGE, 0)
    write_failure(root, "new error")
    assert not record_reference_round(root, STAGE, 1)
    (root / f"{STAGE}.json").write_text("changed")
    assert not record_reference_round(root, STAGE, 2)
    # a new reference session does not inherit the repeat failure
    root = tmp_path / "new_session"
    write_failure(root)
    record_reference_round(root, STAGE, 0)
    assert not record_reference_round(root, STAGE, 0)


def test_archive_round_preserves_each_validation_result_and_refuses_symlink_destination(
    tmp_path: Path,
) -> None:
    # multiple validations in one round preserve each result
    root = tmp_path / "rounds"
    root.mkdir()
    report = root / "validation_report.json"
    report.write_text("first")
    first = archive_round(root, "validation", 0, [report.name])
    report.write_text("second")
    second = archive_round(root, "validation", 0, [report.name])
    assert (first / report.name).read_text() == "first"
    assert (second / report.name).read_text() == "second"
    assert report.read_text() == "second"
    # the archive refuses a symlink destination
    root = tmp_path / "symlinked"
    root.mkdir()
    (root / "paper_diagnostics").symlink_to(root / "outside")
    with pytest.raises(ValueError, match="symlink"):
        archive_round(root, "validation", 0, [])


def test_reference_prompt_lists_every_supported_category() -> None:
    prompt = Path("src/prompts/reference_query_plan_postdraft.md").read_text()
    assert all(f"`{category}`" in prompt for category in VALID_QUERY_CATEGORIES)


def test_diagnostic_cli_dispatches_actions_and_reports_repeated_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # every action dispatches successfully
    for action in ("context", "reference", "archive"):
        root = tmp_path / action
        write_failure(root)
        with monkeypatch.context() as m:
            m.setattr(sys, "argv", ["diagnostics", action, str(root), STAGE, "0"])
            assert main() == 0, action
    # a repeated failure is reported with exit code 3
    root = tmp_path / "repeated"
    write_failure(root)
    record_reference_round(root, STAGE, 0)
    monkeypatch.setattr(sys, "argv", ["diagnostics", "reference", str(root), STAGE, "1"])
    assert main() == 3


@pytest.mark.parametrize("script", ["paper.sh", "paper_edit.sh"])
def test_shell_reference_loop_repairs_with_feedback_then_stops_identical_failure(
    tmp_path: Path, script: str
) -> None:
    source = (ROOT / "src/scripts" / script).read_text()
    for stage in ("reference_query_plan", STAGE):
        output_dir = tmp_path / stage
        output_dir.mkdir()
        match = re.search(rf"(?m)^ *(?:local )?{stage}_repair_round=0$", source)
        assert match is not None
        lines = source[match.start():].splitlines()
        indentation = len(lines[0]) - len(lines[0].lstrip())
        end = next(index for index, line in enumerate(lines) if line == " " * indentation + "done")
        loop = textwrap.dedent("\n".join(lines[: end + 1])).replace("local ", "", 1)
        prefix = stage.upper()
        # Execute the production shell loop with local fake agent/validator functions.
        command = f'''
source "{ROOT}/src/scripts/lib/paper_diagnostics.sh"
REPO_ROOT="{ROOT}"
PYTHON_BIN="{sys.executable}"
OUTPUT_DIR="{output_dir}"
PAPER_MAX_REPAIR_ROUNDS=10
{prefix}_VALIDATION_FEEDBACK_JSON_PATH="$OUTPUT_DIR/{stage}_validation_feedback.json"
{prefix}_VALIDATION_FEEDBACK_MD_PATH="$OUTPUT_DIR/{stage}_validation_feedback.md"
render_prompt() {{ echo "base prompt"; }}
run_prompt_stage_with_feedback() {{
  printf '%s' "$2" > "$3"
  echo '{{"category":"invalid"}}' > "$OUTPUT_DIR/{stage}.json"
  echo 'previous plan' > "$OUTPUT_DIR/{stage}.md"
  echo called >> "$OUTPUT_DIR/calls"
}}
validate_{stage}_with_feedback() {{
  echo '{{"hard_errors":["category must be baseline_comparator"]}}' > "${prefix}_VALIDATION_FEEDBACK_JSON_PATH"
  return 1
}}
run_loop() {{
{loop}
}}
run_loop
'''
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=15)
        assert result.returncode == 1, result.stderr
        assert (output_dir / "calls").read_text().splitlines() == ["called", "called"]
        repair = (output_dir / f"{stage}_repair_prompt_round_1.md").read_text()
        assert '"category":"invalid"' in repair
        assert "category must be baseline_comparator" in repair
        assert "unchanged after repair" in result.stdout


# ── archiving must not perturb audit input hashes ──


def _write_paper(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "paper.tex").write_text("A grounded result.\n")
    (root / "paper_data.json").write_text('{"score":2.5}\n')
    (root / "claims_grounding.json").write_text("{}\n")
    (root / "validation_report.json").write_text('{"ok":true}\n')
    (root / "paper_evidence_audit.json").write_text('{"verdict":"pass"}\n')
    (root / "paper_evidence_audit_raw.txt").write_text("AUDIT_VERDICT: PASS\n")


def test_archiving_paper_gates_and_cost_ledger_updates_do_not_invalidate_input_hashes(
    tmp_path: Path,
) -> None:
    # archiving either paper gate keeps the audit and validation hashes
    for stage in ("validation", "evidence_audit"):
        root = tmp_path / stage
        _write_paper(root)
        before_audit = audit_input_sha256(root)
        before_validation = validation_input_sha256(root)
        archive_round(
            root, stage, 0,
            ["validation_report.json", "paper_evidence_audit.json", "paper_evidence_audit_raw.txt"],
        )
        assert audit_input_sha256(root) == before_audit, stage
        assert validation_input_sha256(root) == before_validation, stage
    # cost ledger updates do not invalidate the audit hashes
    root = tmp_path / "ledger"
    _write_paper(root)
    metrics = root / "metrics"
    metrics.mkdir()
    ledger = metrics / "cost_events.jsonl"
    summary = metrics / "cost_summary.json"
    ledger.write_text('{"cost": 1}\n')
    summary.write_text('{"total": 1}\n')
    before = audit_input_sha256(root)
    assert not any("cost_events" in key or "cost_summary" in key for key in before)
    ledger.write_text('{"cost": 1}\n{"cost": 2}\n')
    summary.write_text('{"total": 3}\n')
    assert audit_input_sha256(root) == before


def test_resolved_diagnostic_targets_stay_out_of_hashes_while_real_evidence_changes_invalidate(
    tmp_path: Path,
) -> None:
    # resolved diagnostic targets do not reintroduce archive hashes
    for index, resolved_path in enumerate((
        ".",
        "paper_diagnostics",
        "paper_diagnostics/evidence_audit/round_0000/attempt_001/paper_evidence_audit_raw.txt",
    )):
        root = tmp_path / f"resolved_{index}"
        _write_paper(root)
        evidence = root / "scientific-evidence"
        evidence.mkdir()
        (evidence / "verification-report.json").write_text(
            json.dumps({"results": [{"resolved_path": resolved_path}]})
        )
        before = audit_input_sha256(root)
        archive_round(root, "evidence_audit", 0, ["paper_evidence_audit_raw.txt"])
        assert audit_input_sha256(root) == before, resolved_path
    # real evidence changes still invalidate the audit after archiving
    for index, evidence_path in enumerate(("paper_data.json", "evidence/paper_diagnostics/data.csv")):
        root = tmp_path / f"evidence_{index}"
        _write_paper(root)
        evidence = root / evidence_path
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text('{"score":2.5}\n')
        archive_round(root, "evidence_audit", 0, ["paper_evidence_audit_raw.txt"])
        before = audit_input_sha256(root)
        evidence.write_text('{"score":0.0}\n')
        assert audit_input_sha256(root) != before, evidence_path


# ── paper event bridge ──


def _emit(tmp_path: Path, gate: str, state: str, fields: dict | None = None, **overrides) -> str:
    stream = overrides.pop("stream", io.StringIO())
    kwargs = dict(
        gate=gate, state=state, fields=fields or {}, output_format="human", transport="direct",
        journal_path=tmp_path / "paper_events.jsonl", session_id="paper-test", stream=stream,
        timestamp_fn=lambda: TIMESTAMP,
    )
    kwargs.update(overrides)
    emit_event(**kwargs)
    return stream.getvalue()


def test_direct_human_events_are_concise_terminal_safe_and_render_sanitized_qualifiers(
    tmp_path: Path,
) -> None:
    # concise and terminal safe
    rendered = _emit(tmp_path, "draft", "started", {"idea": "forged\nline\x1b[2J", "round": 1})
    assert rendered.startswith("[PAPER] ⠋ Draft · started")
    assert "\x1b" not in rendered
    assert "forged\nline" not in rendered
    # collection qualifiers are ignored
    rendered = _emit(
        tmp_path, "preparation", "started", {"phase": ["private"], "round": {"secret": "value"}}
    )
    assert rendered == "[PAPER] ⠋ Preparation · started\n"
    # scalar gate qualifiers are rendered
    for gate, fields, qualifier in (
        ("preparation", {"phase": "grounding"}, "grounding"),
        ("repair", {"round": 2}, "round=2"),
    ):
        assert qualifier in _emit(
            tmp_path, gate, "started", fields, journal_path=tmp_path / f"{gate}.jsonl"
        )
    # verbose display fields are sanitized
    rendered = _emit(
        tmp_path, "review", "failed",
        {"reason": "bad\nartifact\x1b]8;;https://example.com\x07‮evil", "round": 2},
        verbose=True,
    )
    assert "reason=bad\\nartifact" in rendered
    assert "round=2" in rendered
    assert "\\u202e" in rendered
    assert "‮" not in rendered
    assert "\x1b" not in rendered
    assert "\x07" not in rendered


def test_direct_human_heartbeat_is_journaled_without_rendering_but_outcomes_still_render(
    tmp_path: Path,
) -> None:
    # heartbeat is journaled without rendering
    rendered = _emit(tmp_path, "strategy", "started", {"heartbeat": True})
    records = [json.loads(line) for line in (tmp_path / "paper_events.jsonl").read_text().splitlines()]
    assert records[0]["fields"] == {"heartbeat": True}
    assert rendered == ""
    # an outcome with a heartbeat field still renders
    assert "Strategy · failed" in _emit(
        tmp_path, "strategy", "failed", {"heartbeat": True}, journal_path=tmp_path / "outcome.jsonl"
    )


def test_direct_human_event_honors_no_color(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("FORCE_COLOR", "1")
    colored = _emit(tmp_path, "figures", "completed", journal_path=tmp_path / "colored.jsonl")
    assert "\x1b[32m✓\x1b[0m" in colored

    monkeypatch.setenv("NO_COLOR", "1")
    plain = _emit(tmp_path, "figures", "failed", journal_path=tmp_path / "plain.jsonl")
    assert "[PAPER] ✗ Figures · failed" in plain
    assert "\x1b" not in plain


def test_json_and_wrapper_transport_records_are_strict_and_extractable(tmp_path: Path) -> None:
    # direct json event is strict json
    rendered = _emit(
        tmp_path, "validation", "completed", {"exit_code": 0, "score": float("nan")},
        output_format="json",
    )
    record = json.loads(
        rendered, parse_constant=lambda value: pytest.fail(f"non-standard JSON constant: {value}")
    )
    assert record["event"] == "paper_gate"
    assert record["gate"] == "validation"
    assert record["state"] == "completed"
    assert record["fields"] == {"exit_code": 0, "score": None}
    # wrapper transport emits a prefixed json record
    line = _emit(
        tmp_path, "evidence_audit", "started", {"round": 3}, transport="wrapper",
        journal_path=tmp_path / "wrapper.jsonl",
    ).strip()
    assert line.startswith(TRANSPORT_PREFIX)
    record = json.loads(line.removeprefix(TRANSPORT_PREFIX))
    assert record["gate"] == "evidence_audit"
    assert record["fields"]["round"] == 3
    # transport extraction removes sentinels and rejects non-strict json
    valid_record = {
        "timestamp": TIMESTAMP, "source": "paper", "event": "paper_gate",
        "session_id": "paper-test", "gate": "draft", "state": "completed", "fields": {"round": 1},
    }
    output = (
        f"{TRANSPORT_PREFIX}{json.dumps(valid_record)}\n"
        f'{TRANSPORT_PREFIX}{{"score": NaN}}\n'
        f'{TRANSPORT_PREFIX}{{"timestamp":"x","source":"paper",'
        '"event":"paper_gate","session_id":"s","gate":"draft",'
        '"gate":"draft","state":"started","fields":{}}\n'
        "Paper directory: /tmp/paper_final\n"
    )
    cleaned_output, records = extract_transport_events(output)
    assert cleaned_output == "Paper directory: /tmp/paper_final\n"
    assert records == [valid_record]


def test_event_journal_is_private_appends_and_rejects_symlink(tmp_path: Path) -> None:
    # private and appends
    journal_path = tmp_path / "paper_events.jsonl"
    journal_path.touch(mode=0o644)
    journal_path.chmod(0o644)
    for state in ("started", "completed"):
        _emit(tmp_path, "references", state)
    records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert [record["state"] for record in records] == ["started", "completed"]
    assert os.stat(journal_path).st_mode & 0o777 == 0o600
    # rejects a symlink
    root = tmp_path / "symlinked"
    root.mkdir()
    target = root / "target.jsonl"
    target.write_text("untouched\n", encoding="utf-8")
    (root / "paper_events.jsonl").symlink_to(target)
    with pytest.raises(OSError):
        _emit(root, "finalization", "started", output_format="json")
    assert target.read_text(encoding="utf-8") == "untouched\n"


def test_diagnostic_sink_is_private_bounded_and_rejects_symlink_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # private sink
    log_path = tmp_path / "paper.log"
    stream_private_diagnostics(log_path, io.BytesIO(b"private diagnostics\n"))
    assert log_path.read_bytes() == b"private diagnostics\n"
    assert os.stat(log_path).st_mode & 0o777 == 0o600
    # rejects a symlinked parent
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(OSError):
        stream_private_diagnostics(linked_directory / "escaped.log", io.BytesIO(b"must not escape\n"))
    assert not (real_directory / "escaped.log").exists()
    # bounded but drains the source
    monkeypatch.setattr(paper_event_bridge, "_DIAGNOSTIC_LOG_MAX_BYTES", 96)
    monkeypatch.setattr(paper_event_bridge, "_DIAGNOSTIC_LOG_TAIL_BYTES", 24)
    payload = b"prefix\n" + (b"x" * 256) + b"\ntail"
    source = io.BytesIO(payload)
    log_path = tmp_path / "bounded.log"

    stream_private_diagnostics(log_path, source)

    captured = log_path.read_bytes()
    assert len(captured) <= 96
    assert paper_event_bridge._DIAGNOSTIC_TRUNCATION_MARKER in captured
    assert captured.startswith(b"prefix\n")
    assert captured.endswith(b"\ntail")
    assert source.tell() == len(payload)


@pytest.mark.parametrize("drain_ticks", ["0", "invalid", "86400"])
def test_diagnostic_sink_wait_is_bounded_when_writer_lingers(drain_ticks: str, tmp_path: Path) -> None:
    source = Path("src/scripts/lib/paper_progress.sh").read_text(encoding="utf-8")
    function_start = source.index("paper_diagnostic_sink_is_child() {")
    finish_start = source.index("finish_paper_diagnostic_sink() {", function_start)
    function_end = source.index("\n}\n", finish_start) + len("\n}\n")
    function_source = source[function_start:function_end]
    bridge = Path("src/orchestrator/paper_event_bridge.py").resolve()
    log_path = tmp_path / "paper.log"
    script = f"""
set -euo pipefail
PAPER_DIAGNOSTIC_DRAIN_TICKS="{drain_ticks}"
exec 3>&1
exec 1> >(
  "{sys.executable}" "{bridge}" \
    --diagnostic-sink "{log_path}"
) 2>&1
PAPER_DIAGNOSTIC_SINK_PID="$!"
{function_source}
sleep 30 &
lingering_writer="$!"
finish_paper_diagnostic_sink || true
kill "$lingering_writer" 2>/dev/null || true
wait "$lingering_writer" 2>/dev/null || true
echo finished
"""
    process = subprocess.Popen(
        ["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, _stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    assert process.returncode == 0
    assert stdout.endswith("finished\n")


def test_paper_script_instruments_every_major_gate() -> None:
    source = Path("src/scripts/paper.sh").read_text(encoding="utf-8")
    for gate in (
        "preparation", "strategy", "draft", "references", "figures", "validation",
        "evidence_audit", "review", "repair", "finalization",
    ):
        assert f'paper_gate_start "{gate}"' in source
    for fragment in (
        "--log-format", "--heartbeat-seconds", "--event-transport",
        'echo "Paper directory: $OUTPUT_DIR"',
    ):
        assert fragment in source
    source += Path("src/scripts/lib/paper_progress.sh").read_text(encoding="utf-8")
    for fragment in (
        'PAPER_DIAGNOSTIC_LOG="$OUTPUT_DIR/paper_diagnostics_',
        '--diagnostic-sink "$PAPER_DIAGNOSTIC_LOG"',
        'exec 1> >("$PYTHON_BIN" "$PAPER_EVENT_BRIDGE"',
        'PAPER_DIAGNOSTIC_SINK_PID="$!"',
        'wait "$sink_pid"',
        "finish_paper_diagnostic_sink",
        'if ! "${command[@]}" >&3',
        "exec 3>&-",
        "paper_maybe_heartbeat",
        "exit 130",
        "exit 2",
    ):
        assert fragment in source
