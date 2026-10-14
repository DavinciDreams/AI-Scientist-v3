"""Alignment reports and fingerprints, alignment history, and cumulative prompt rendering."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.investigation.program_spec import render_default_program_spec_toml
from src.investigation.task_alignment import (
    REQUIRED_ALIGNMENT_DIMENSIONS,
    TaskAlignmentError,
    _main,
    load_task_alignment_report,
    task_alignment_fingerprints,
    task_tree_fingerprint,
)
from src.task_drafting import alignment_history
from src.task_drafting.alignment_history import (
    MAX_PROMPT_BYTES,
    capture_alignment_report,
    render_alignment_history,
)
from tests.task_drafting.test_task_contract import _valid_contract

ROOT = Path(__file__).resolve().parents[2]
BASE = "sha256:" + "a" * 64
REPAIRED = "sha256:" + "b" * 64


def _write_scaffold(root: Path) -> dict[str, str]:
    (root / "task.md").write_text("# Canonical task\n", encoding="utf-8")
    (root / "capture.py").write_text("CAPTURE = True\n", encoding="utf-8")
    (root / "evaluate.py").write_text("SEARCH = True\n", encoding="utf-8")
    (root / "evaluate_final.py").write_text("FINAL = True\n", encoding="utf-8")
    (root / "program_0/src").mkdir(parents=True)
    (root / "program_0/program_spec.toml").write_text(
        render_default_program_spec_toml(), encoding="utf-8"
    )
    (root / "program_0/src/program.py").write_text("BASELINE = True\n", encoding="utf-8")
    (root / "resources/search").mkdir(parents=True)
    (root / "resources/search/dev.json").write_text("{}\n", encoding="utf-8")
    (root / "resources/final").mkdir(parents=True)
    (root / "resources/final/test.json").write_text("{}\n", encoding="utf-8")
    (root / "task_contract.json").write_text(json.dumps(_valid_contract(root)), encoding="utf-8")
    return task_alignment_fingerprints(root)


def _report(fingerprints: dict[str, str], *, failed: str | None = None) -> dict:
    dimensions = []
    for dimension in REQUIRED_ALIGNMENT_DIMENSIONS:
        verdict = "fail" if dimension == failed else "pass"
        dimensions.append(
            {
                "id": dimension,
                "verdict": verdict,
                "evidence": [
                    {"path": "task_contract.json", "line": 1, "detail": f"Evidence for {dimension}"}
                ],
                "remediation": "Use the required upstream evaluator." if verdict == "fail" else "",
            }
        )
    return {
        "schema_version": 1,
        "verdict": "fail" if failed else "pass",
        "summary": "Aligned scaffold." if failed is None else "Alignment failed.",
        "fingerprints": fingerprints,
        "dimensions": dimensions,
    }


def _load(root: Path, payload: dict, **kwargs):
    report_path = root / "report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_task_alignment_report(report_path, scaffold_dir=root, **kwargs)


def _scaffold_dir(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    _write_scaffold(root)
    return root


# --- load_task_alignment_report -------------------------------------------------------------


def test_load_task_alignment_report_accepts_pass_returns_failures_and_rejects_hidden_evidence(
    tmp_path: Path,
) -> None:
    # test_load_task_alignment_report_accepts_complete_pass
    root = tmp_path / "pass"
    root.mkdir()
    report = _load(root, _report(_write_scaffold(root)))
    assert report.verdict == "pass"
    assert report.failed_dimensions == ()

    # test_load_task_alignment_report_returns_failed_dimensions
    root = tmp_path / "fail"
    root.mkdir()
    report = _load(root, _report(_write_scaffold(root), failed="search_final_isolation"))
    assert report.verdict == "fail"
    assert report.failed_dimensions == ("search_final_isolation",)

    # test_load_task_alignment_report_rejects_evidence_hidden_from_judge
    scaffold = tmp_path / "scaffold"
    scaffold.mkdir()
    fingerprints = _write_scaffold(scaffold)
    visible = tmp_path / "visible"
    visible.mkdir()
    (visible / "task_contract.json").write_bytes((scaffold / "task_contract.json").read_bytes())
    payload = _report(fingerprints)
    payload["dimensions"][0]["evidence"][0]["path"] = "resources/final/test.json"
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TaskAlignmentError, match="safe relative file"):
        load_task_alignment_report(
            report_path, scaffold_dir=scaffold, evidence_scaffold_dir=visible
        )


def _pop_dimension(payload: dict, root: Path) -> dict:
    payload["dimensions"].pop()
    return payload


def _append_to(relative: str):
    def mutate(payload: dict, root: Path) -> dict:
        with (root / relative).open("a", encoding="utf-8") as handle:
            handle.write("\n# changed\n")
        return payload

    return mutate


def _evidence_path(unsafe: str):
    def mutate(payload: dict, root: Path) -> dict:
        payload["dimensions"][0]["evidence"][0]["path"] = unsafe
        return payload

    return mutate


def _rewrite_evaluator(payload: dict, root: Path) -> dict:
    (root / "evaluate.py").write_text("print('changed')\n", encoding="utf-8")
    return payload


def _pass_with_failed_dimension(payload: dict, root: Path) -> dict:
    return {**_report(payload["fingerprints"], failed="compute_realism"), "verdict": "pass"}


def _evidence_past_end(payload: dict, root: Path) -> dict:
    payload["dimensions"][0]["evidence"][0]["line"] = 999
    return payload


def _unknown_field(payload: dict, root: Path) -> dict:
    payload["dimensions"][0]["unexpected"] = True
    return payload


@pytest.mark.parametrize(
    "mutate,match",
    [
        pytest.param(_pop_dimension, "dimensions must contain exactly", id="missing-dimension"),
        pytest.param(_rewrite_evaluator, "fingerprints do not match", id="stale-evaluator"),
        pytest.param(_append_to("capture.py"), "fingerprints do not match", id="stale-capture"),
        pytest.param(
            _append_to("program_0/program_spec.toml"), "fingerprints do not match", id="stale-spec"
        ),
        pytest.param(
            _append_to("program_0/src/program.py"), "fingerprints do not match", id="stale-baseline"
        ),
        pytest.param(_evidence_path("../task.md"), "safe relative file", id="evidence-parent"),
        pytest.param(_evidence_path("/tmp/task.md"), "safe relative file", id="evidence-absolute"),
        pytest.param(_evidence_path("task.md/child"), "safe relative file", id="evidence-child"),
        pytest.param(_pass_with_failed_dimension, "top-level verdict", id="pass-with-failure"),
        pytest.param(_evidence_past_end, "evidence line exceeds file length", id="evidence-line"),
        pytest.param(_unknown_field, "unexpected field", id="unknown-field"),
    ],
)
def test_load_task_alignment_report_rejects_invalid_report_or_stale_scaffold(
    tmp_path: Path, mutate, match: str
) -> None:
    payload = mutate(_report(_write_scaffold(tmp_path)), tmp_path)
    with pytest.raises(TaskAlignmentError, match=match):
        _load(tmp_path, payload)


# --- fingerprints and CLI -------------------------------------------------------------------


def test_task_tree_fingerprint_detects_mutations_ignores_runtime_and_root_prompt_and_deadline(
    tmp_path: Path,
) -> None:
    # test_task_tree_fingerprint_detects_any_scaffold_mutation
    root = _scaffold_dir(tmp_path, "mutation")
    candidate = root / "program_0" / "program.py"
    candidate.write_text("BASELINE = 1\n", encoding="utf-8")
    baseline = task_tree_fingerprint(root)
    candidate.write_text("BASELINE = 2\n", encoding="utf-8")
    assert task_tree_fingerprint(root) != baseline

    # test_task_tree_fingerprint_ignores_managed_runtime
    root = _scaffold_dir(tmp_path, "runtime")
    baseline = task_tree_fingerprint(root)
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to("/usr/bin/python3")
    (root / ".agentic_runtime.json").write_text("{}\n", encoding="utf-8")
    assert task_tree_fingerprint(root) == baseline

    # test_task_tree_fingerprint_ignores_only_root_repair_prompt
    root = _scaffold_dir(tmp_path, "prompt")
    baseline = task_tree_fingerprint(root)
    (root / ".agentic_repair_prompt.md").write_text("repair instructions\n")
    assert task_tree_fingerprint(root) == baseline
    (root / "program_0" / ".agentic_repair_prompt.md").write_text("candidate-owned content\n")
    assert task_tree_fingerprint(root) != baseline

    # test_task_tree_fingerprint_respects_expired_deadline
    root = _scaffold_dir(tmp_path, "deadline")
    with pytest.raises(TaskAlignmentError, match="deadline exceeded"):
        task_tree_fingerprint(root, deadline=0.0)


def test_fingerprints_reject_symlinked_files_and_entrypoints(tmp_path: Path) -> None:
    # test_task_tree_fingerprint_rejects_symlinks
    root = _scaffold_dir(tmp_path, "tree")
    (root / "unsafe").symlink_to(root / "task.md")
    with pytest.raises(TaskAlignmentError, match="symlink"):
        task_tree_fingerprint(root)

    # test_task_alignment_fingerprints_rejects_symlinked_entrypoint
    root = _scaffold_dir(tmp_path, "entrypoint")
    entrypoint = root / "program_0/src/program.py"
    target = root / "program_0/src/target.py"
    entrypoint.replace(target)
    entrypoint.symlink_to(target.name)
    with pytest.raises(TaskAlignmentError, match="symlink"):
        task_alignment_fingerprints(root)


def test_alignment_cli_reports_unsafe_scaffold_and_filesystem_errors_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # test_fingerprint_cli_reports_unsafe_scaffold_without_traceback
    root = _scaffold_dir(tmp_path, "unsafe")
    capture = root / "capture.py"
    capture.unlink()
    capture.symlink_to("/etc/passwd")
    for command in ("fingerprints", "tree-fingerprint"):
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", ["task_alignment", command, "--scaffold", str(root)])
            assert _main() == 2, command
        output = capsys.readouterr()
        assert "invalid scaffold:" in output.err, command
        assert "Traceback" not in output.err, command

    # test_fingerprint_cli_reports_filesystem_errors_without_traceback
    def fail(_scaffold: Path) -> str:
        raise OSError("permission denied")

    for command, target in (
        ("fingerprints", "task_alignment_fingerprints"),
        ("tree-fingerprint", "task_tree_fingerprint"),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(f"src.investigation.task_alignment.{target}", fail)
            patch.setattr(sys, "argv", ["task_alignment", command, "--scaffold", str(tmp_path)])
            assert _main() == 2, command
        output = capsys.readouterr()
        assert "invalid scaffold: permission denied" in output.err, command
        assert "Traceback" not in output.err, command

    # test_validate_cli_reports_filesystem_errors_without_traceback
    def fail_load(*_args: object, **_kwargs: object) -> object:
        raise OSError("permission denied")

    monkeypatch.setattr("src.investigation.task_alignment.load_task_alignment_report", fail_load)
    monkeypatch.setattr(
        sys,
        "argv",
        ["task_alignment", "validate", "--scaffold", str(tmp_path), "--report", str(tmp_path / "report.json")],
    )
    assert _main() == 2
    output = capsys.readouterr()
    assert "invalid alignment report: permission denied" in output.out
    assert "Traceback" not in output.out + output.err


# --- alignment history: cumulative findings survive rolled-back repairs ---------------------


def _history_report(path: Path, *, remediation: str = "Calibrate both planes", verdict="fail"):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "verdict": verdict,
                "fingerprints": {"declared_scaffold": BASE},
                "dimensions": [
                    {
                        "id": "execution_feasibility",
                        "verdict": verdict,
                        "remediation": remediation if verdict == "fail" else "",
                        "evidence": [
                            {
                                "path": "evaluate_final.py",
                                "line": 35,
                                "detail": "Final retains 875000 updates",
                            }
                        ],
                    }
                ],
                "output_evidence": "RAW WORKER STDERR MUST NEVER REACH JUDGE",
            }
        )
    )
    return path


def test_capture_accepts_normalized_reports_and_marks_clipped_findings(tmp_path: Path) -> None:
    # test_capture_accepts_normalized_canonical_report
    for field in ("verdict", "dimension_id", "dimension_verdict"):
        root = tmp_path / f"normalized-{field}"
        scaffold = root / "task"
        scaffold.mkdir(parents=True)
        payload = _report(_write_scaffold(scaffold), failed="execution_feasibility")
        if field == "verdict":
            payload["verdict"] = " fail "
        else:
            key = "id" if field == "dimension_id" else "verdict"
            for dimension in payload["dimensions"]:
                dimension[key] = " " + dimension[key] + " "
        source = root / "report.json"
        source.write_text(json.dumps(payload))
        assert load_task_alignment_report(source, scaffold_dir=scaffold).verdict == "fail", field

        capture_alignment_report(source, root / "alignment_0.json", task_fingerprint=BASE)
        text = render_alignment_history(root, current_fingerprint=BASE)
        assert '"historical_verdict": "fail"' in text, field
        assert '"id": "execution_feasibility"' in text, field

    # test_capture_marks_clipped_findings_in_stored_and_rendered_history
    for field in ("remediation", "evidence", "path", "detail"):
        root = tmp_path / f"clipped-{field}"
        root.mkdir()
        source = _history_report(root / "report.json")
        payload = json.loads(source.read_text())
        finding = payload["dimensions"][0]
        if field == "remediation":
            finding[field] = "a" * 1100
        elif field == "evidence":
            finding[field] *= 4
        else:
            finding["evidence"][0][field] = "b" * 600
        source.write_text(json.dumps(payload))

        output = root / "alignment_0.json"
        capture_alignment_report(source, output, task_fingerprint=BASE)
        assert json.loads(output.read_text())["findings"][0]["truncated"] is True, field
        text = render_alignment_history(root, current_fingerprint=BASE)
        assert '"truncated": true' in text, field


def test_history_retains_earlier_findings_redacts_secrets_and_never_overwrites(tmp_path):
    # test_earlier_findings_survive_later_pass_and_changed_fingerprint
    root = tmp_path / "survive"
    root.mkdir()
    source = _history_report(root / "task_alignment.json")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    capture_alignment_report(source, root / "alignment_0.json", task_fingerprint=BASE)
    _history_report(source, verdict="pass")
    capture_alignment_report(source, root / "alignment_1.json", task_fingerprint=REPAIRED)
    text = render_alignment_history(root, current_fingerprint=BASE)
    assert "Calibrate both planes" in text
    assert "875000" in text
    assert digest in text
    assert BASE in text and REPAIRED in text
    assert "never acceptance authority" in text
    assert "revalidate" in text
    assert "RAW WORKER STDERR" not in text
    assert "source_diff" not in text

    # test_redacts_findings_and_reread_history
    root = tmp_path / "redact"
    root.mkdir()
    secret = "Authorization: Bearer private-value"
    source = _history_report(root / "report.json", remediation=secret)
    capture_alignment_report(source, root / "alignment_0.json", task_fingerprint=BASE)
    text = render_alignment_history(root, current_fingerprint=BASE)
    assert "private-value" not in text
    assert "redacted" in text

    # test_existing_history_is_never_overwritten
    root = tmp_path / "overwrite"
    root.mkdir()
    source = _history_report(root / "report.json")
    output = root / "alignment_0.json"
    capture_alignment_report(source, output, task_fingerprint=BASE)
    original = output.read_bytes()
    _history_report(source, remediation="different")
    with pytest.raises(ValueError):
        capture_alignment_report(source, output, task_fingerprint=BASE)
    assert output.read_bytes() == original


_MALFORMED_REPORT_FIELDS = [
    {"verdict": []},
    {"fingerprints": []},
    {"fingerprints": {}},
    {"dimensions": None},
    {"dimensions": [None]},
    {"dimensions": [{"id": "unknown", "verdict": "fail"}]},
    {"dimensions": [{"id": "execution_feasibility", "verdict": []}]},
    {"dimensions": [{"id": "execution_feasibility", "verdict": "fail", "evidence": 1}]},
    {"verdict": "pass"},
]


def test_capture_rejects_symlinks_invalid_reports_and_malformed_fields(tmp_path):
    # test_rejects_symlink_inputs_outputs_and_parents
    for kind in ("source", "destination", "parent"):
        root = tmp_path / f"symlink-{kind}"
        root.mkdir()
        real = _history_report(root / "real.json")
        source, output = real, root / "alignment_0.json"
        if kind == "source":
            source = root / "task_alignment.json"
            source.symlink_to(real)
        elif kind == "destination":
            output.symlink_to(real)
        else:
            linked = root / "linked"
            linked.symlink_to(root, target_is_directory=True)
            output = linked / "alignment_0.json"
        original = real.read_bytes()
        with pytest.raises((OSError, ValueError)):
            capture_alignment_report(source, output, task_fingerprint=BASE)
        assert real.read_bytes() == original, kind

    # test_invalid_reports_never_create_history
    for index, contents in enumerate(["not json", "[]", '{"schema_version":99}', "x" * 1048577]):
        root = tmp_path / f"invalid-{index}"
        root.mkdir()
        source = root / "report.json"
        source.write_text(contents)
        with pytest.raises(ValueError):
            capture_alignment_report(source, root / "alignment_0.json", task_fingerprint=BASE)
        assert not (root / "alignment_0.json").exists(), contents[:20]

    # test_rejects_malformed_report_fields
    for index, change in enumerate(_MALFORMED_REPORT_FIELDS):
        root = tmp_path / f"malformed-{index}"
        root.mkdir()
        source = _history_report(root / "report.json")
        source.write_text(json.dumps(json.loads(source.read_text()) | change))
        with pytest.raises(ValueError):
            capture_alignment_report(source, root / "alignment_0.json", task_fingerprint=BASE)


_MALFORMED_PROVENANCE = [
    {"report_sha256": "invalid"},
    {"reviewed_task_fingerprint": None},
    {"historical_verdict": []},
    {"findings": None},
]


def test_render_history_bounds_large_reports_and_ignores_missing_or_malformed_records(tmp_path):
    # test_large_history_preserves_each_dimension_and_report_hash
    root = tmp_path / "large"
    root.mkdir()
    hashes = []
    for index in range(16):
        source = _history_report(root / "report.json")
        payload = json.loads(source.read_text())
        payload["dimensions"] = [
            dict(payload["dimensions"][0], id=name, remediation=f"fix{index} {name} " + "日本語" * 4000)
            for name in REQUIRED_ALIGNMENT_DIMENSIONS
        ]
        source.write_text(json.dumps(payload))
        hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())
        capture_alignment_report(source, root / f"alignment_{index}.json", task_fingerprint=BASE)
    text = render_alignment_history(root, current_fingerprint=BASE)
    assert len(text.encode()) <= MAX_PROMPT_BYTES
    assert all(digest in text for digest in hashes)
    assert all(name in text for name in REQUIRED_ALIGNMENT_DIMENSIONS)
    assert "truncated" in text

    # test_missing_or_malformed_history_is_empty
    root = tmp_path / "malformed"
    root.mkdir()
    assert render_alignment_history(root / "missing", current_fingerprint=BASE) == ""
    (root / "alignment_0.json").write_text("[]")
    (root / "alignment_1.json").symlink_to(root / "missing")
    assert render_alignment_history(root, current_fingerprint=BASE) == ""

    # test_malformed_stored_provenance_is_ignored
    for index, change in enumerate(_MALFORMED_PROVENANCE):
        root = tmp_path / f"provenance-{index}"
        root.mkdir()
        source = _history_report(root / "report.json")
        output = root / "alignment_0.json"
        capture_alignment_report(source, output, task_fingerprint=BASE)
        output.write_text(json.dumps(json.loads(output.read_text()) | change))
        assert render_alignment_history(root, current_fingerprint=BASE) == "", change


def test_storage_failures_map_to_infrastructure_without_exposing_paths(
    tmp_path, monkeypatch, capsys
):
    # test_cli_maps_storage_failure_without_exposing_paths
    def fail(*args, **kwargs):
        raise OSError(errno.EDQUOT, "secret path")

    with monkeypatch.context() as patch:
        patch.setattr(
            sys,
            "argv",
            ["alignment_history", "--report", "unused", "--output", "unused", "--task-fingerprint", BASE],
        )
        patch.setattr(alignment_history, "capture_alignment_report", fail)
        with pytest.raises(SystemExit) as result:
            alignment_history._main()
    assert result.value.code == 4
    assert "secret path" not in capsys.readouterr().err

    # test_failed_write_preserves_primary_storage_error
    source = _history_report(tmp_path / "report.json")

    def fail_sync(fd):
        raise OSError(errno.EDQUOT, "quota")

    def fail_cleanup(*args, **kwargs):
        raise OSError(errno.EIO, "cleanup")

    monkeypatch.setattr(os, "fsync", fail_sync)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(OSError) as result:
        capture_alignment_report(source, tmp_path / "alignment_0.json", task_fingerprint=BASE)
    assert result.value.errno == errno.EDQUOT


# --- prior findings reach repair agents and independent review after expansion --------------


def test_prior_findings_are_rendered_for_each_consumer_and_only_validated_reports_are_captured(
    tmp_path: Path,
) -> None:
    # test_prior_findings_are_rendered_for_each_consumer
    script = (ROOT / "src/scripts/task.sh").read_text()
    for function in (
        "render_alignment_prompt", "render_alignment_repair_prompt",
        "render_qualification_repair_prompt",
    ):
        root = tmp_path / function
        root.mkdir()
        task = root / "task"
        task.mkdir()
        (task / "task.md").write_text("Canonical task")
        history = root / "history"
        history.mkdir()
        report = root / "report.json"
        prior: dict[str, Any] = {
            "schema_version": 1,
            "verdict": "fail",
            "summary": "Earlier defect",
            "fingerprints": {"declared_scaffold": BASE},
            "dimensions": [
                {
                    "id": sorted(REQUIRED_ALIGNMENT_DIMENSIONS)[0],
                    "verdict": "fail",
                    "remediation": "Retain {{TASK_DIR}} baseline budget correction",
                    "evidence": [{"path": "evaluate.py", "line": 1, "detail": "controls starved"}],
                }
            ],
        }
        report.write_text(json.dumps(prior))
        capture_alignment_report(report, history / "alignment_0.json", task_fingerprint=BASE)
        prior["dimensions"][0]["remediation"] = "Latest finding only"
        report.write_text(json.dumps(prior))
        template = root / "template.md"
        template.write_text("Review the current scaffold.")
        args = {
            "render_alignment_prompt": [
                str(template), str(report), "{}", str(task), "0", str(task / "task.md"),
            ],
            "render_alignment_repair_prompt": [str(report)],
            "render_qualification_repair_prompt": [str(report), BASE],
        }[function]
        body = script.split(f"{function}() {{", 1)[1].split("\nPY\n}", 1)[0] + "\nPY"
        result = subprocess.run(
            ["bash", "-c", f'set -eu\n{function}() {{{body}\n}}\n{function} "$@"', "bash", *args],
            cwd=ROOT,
            env=os.environ
            | {
                "PYTHON_BIN": sys.executable,
                "REPO_ROOT": str(ROOT),
                "VALIDATION_ROOT": str(history),
                "OUTPUT_DIR": str(root),
                "STAGED_TASK_DIR": str(task),
                "TASK_DIR": str(task),
                "TASK_DRAFT_MAX_RUNTIME_BYTES": "1024",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "Retain {{TASK_DIR}} baseline budget correction" in result.stdout, function
        assert "Historical verdicts are never acceptance authority" in result.stdout, function
        assert '"report_sha256":' in result.stdout, function
        if function == "render_alignment_prompt":
            assert "Rejected repair history" not in result.stdout

    # test_only_validated_reports_are_captured
    validation = script.index('      --report "$REPORT_PATH" > "$ALIGNMENT_DIAGNOSTIC"')
    capture = script.index('"$PYTHON_BIN" -m src.task_drafting.alignment_history')
    assert capture > validation
    assert '--task-fingerprint "$AFTER_REVIEW"' in script[capture:]
