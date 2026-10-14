"""Rejected repair history preserves useful edits without exposing runtime artifacts or judge text."""

from __future__ import annotations

import base64
import errno
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import src.investigation.task_runtime as runtime
from src.task_drafting import repair_context
from src.task_drafting.qualification import qualify_task_for_publication
from src.task_drafting.qualification_context import qualification_failure_projection
from src.task_drafting.repair_context import capture_rejected_repair, render_repair_history
from tests.task_drafting.test_scaffold_diagnostic_contract import PROMPTS, render_qualification_prompt
from tests.runtime.test_task_runtime import _write_contract_task

ROOT = Path(__file__).resolve().parents[2]
BASE = "sha256:" + "a" * 64


def _trees(root: Path) -> tuple[Path, Path, Path]:
    task = root / "task"
    transaction = root / "transaction"
    (transaction / "files").mkdir(parents=True)
    task.mkdir()
    history = root / "history"
    history.mkdir()
    return task, transaction, history


def capture(
    task: Path,
    transaction: Path,
    history: Path,
    *,
    reports: list[Path] | None = None,
    rejected_fingerprint: str | None = None,
) -> dict:
    output = history / "qualification_0_transaction.json"
    capture_rejected_repair(
        task,
        transaction,
        output,
        base_fingerprint=BASE,
        reports=reports if reports is not None else [],
        rejected_fingerprint=rejected_fingerprint,
    )
    return json.loads(output.read_text())


def _write_record(history: Path, index: int, **fields) -> None:
    record = {"schema_version": 1, "base_fingerprint": BASE, **fields}
    (history / f"qualification_{index}_transaction.json").write_text(json.dumps(record))


def _shell_function(name: str) -> str:
    script = (ROOT / "src/scripts/task.sh").read_text()
    body = script.split(name + "() {", 1)[1].split("\n}\n", 1)[0]
    return name + "() {" + body + "\n}\n"


def test_capture_retains_edit_and_report_and_reuses_fingerprints_without_rehashing(
    tmp_path: Path, monkeypatch
) -> None:
    # test_retains_timeout_edit_and_qualification_report
    task, transaction, history = _trees(tmp_path / "edit")
    (transaction / "files/evaluate.py").write_text("timeout = 90\n")
    (task / "evaluate.py").write_text("timeout = 300\n")
    report = tmp_path / "qualification.json"
    report.write_text('{"category":"missing_second_reset","rpc_calls":760}')
    record = capture(task, transaction, history, reports=[report], rejected_fingerprint=BASE)
    assert "-timeout = 90" in record["diff"]
    assert "+timeout = 300" in record["diff"]
    assert record["base_fingerprint"] == BASE
    assert record["rejected_fingerprint"].startswith("sha256:")
    assert "missing_second_reset" in record["reports"][0]["text"]
    assert "760" in record["reports"][0]["text"]

    # test_history_reuses_fingerprints_without_reading_resource_contents
    from src.investigation import task_alignment

    task, transaction, history = _trees(tmp_path / "reuse")
    with monkeypatch.context() as patch:
        patch.setattr(
            task_alignment,
            "task_tree_fingerprint",
            lambda *_a, **_k: pytest.fail("unexpected resource fingerprint pass"),
        )
        record = capture(task, transaction, history, rejected_fingerprint=BASE)
    assert record["rejected_fingerprint"] == BASE

    # test_missing_rejected_fingerprint_is_unavailable_without_rehashing
    task, transaction, history = _trees(tmp_path / "missing")
    assert capture(task, transaction, history)["rejected_fingerprint"] is None


def test_capture_excludes_sensitive_paths_and_filenames_from_diff(tmp_path: Path) -> None:
    # test_excludes_resources_credentials_binary_and_symlinks
    task, transaction, history = _trees(tmp_path / "paths")
    for name in ["resources/private.py", ".venv/secret.py", "data/observations.py"]:
        target = task / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("do_not_disclose = 1\n")
    (task / ".env").write_text("password=private\n")
    (task / "binary.py").write_bytes(b"\x00binary")
    (task / "credential.py").write_text('api_key = "sk-proj-' + "a" * 32 + '"\n')
    (task / "linked.py").symlink_to(task / "resources/private.py")
    record = capture(task, transaction, history)
    assert "do_not_disclose" not in record["diff"]
    assert "sk-proj-" not in record["diff"]
    assert "binary" not in record["diff"]
    assert record["rejected_fingerprint"] is None
    assert record["omitted_files"] >= 3

    # test_secret_in_filename_is_excluded_from_diff_headers
    task, transaction, history = _trees(tmp_path / "filename")
    name = "Authorization: Bearer private-file.py"
    (task / name).write_text("value = 1\n")
    record = capture(task, transaction, history)
    assert "private-file" not in record["diff"]


_SECRETS = [
    'password = "tiny"',
    "Authorization: Bearer short-value",
    "https://user:tiny@example.test",
    'token = "x"',
]


def test_capture_redacts_report_secrets_marks_missing_reports_and_hides_sensitive_text(
    tmp_path: Path,
) -> None:
    # test_report_secret_is_redacted_and_missing_report_is_marked
    task, transaction, history = _trees(tmp_path / "report")
    report = tmp_path / "secret.json"
    report.write_text('{"api_key":"sk-proj-' + "a" * 32 + '"}')
    record = capture(task, transaction, history, reports=[report, tmp_path / "missing"])
    assert "sk-proj-" not in record["reports"][0]["text"]
    assert record["reports"][1] == {"unavailable": True}

    # test_sensitive_source_headers_reports_and_history_are_not_disclosed
    for index, secret in enumerate(_SECRETS):
        task, transaction, history = _trees(tmp_path / f"secret-{index}")
        (task / "evaluate.py").write_text(secret)
        report = tmp_path / f"report-{index}.json"
        report.write_text(secret)
        record = capture(task, transaction, history, reports=[report])
        assert secret not in json.dumps(record), secret
        assert record["diff"] == "", secret
        assert "redacted-sensitive-text" in record["reports"][0]["text"], secret
        _write_record(history, 1, diff=secret, reports=[{"text": secret}])
        assert secret not in render_repair_history(history, current_fingerprint=BASE), secret


def test_diff_and_history_prompt_stay_bounded_with_report_findings_prioritized(
    tmp_path: Path,
) -> None:
    # test_diff_is_capped_in_bytes_and_marks_truncation
    task, transaction, history = _trees(tmp_path / "diff")
    (task / "evaluate.py").write_text("value = '日本語'\n" * 8000)
    record = capture(task, transaction, history)
    assert len(record["diff"].encode()) <= 64 * 1024
    assert record["truncated"] is True

    # test_large_history_prompt_stays_bounded
    task, transaction, history = _trees(tmp_path / "large")
    (task / "evaluate.py").write_text("value = '日本語'\n" * 8000)
    for index in range(8):
        capture_rejected_repair(
            task, transaction, history / f"qualification_{index}_transaction.json",
            base_fingerprint=BASE, reports=[],
        )
    prompt = render_repair_history(history, current_fingerprint=BASE)
    assert len(prompt.encode()) <= 64 * 1024
    assert '"prompt_truncated": true' in prompt

    # test_unicode_heavy_reports_shrink_alongside_source_diffs
    _, _, history = _trees(tmp_path / "unicode")
    for index in range(8):
        _write_record(history, index, diff="", reports=[{"text": "日本語" * 1400}] * 3)
    prompt = render_repair_history(history, current_fingerprint=BASE)
    assert len(prompt.encode()) <= 64 * 1024
    assert '"prompt_truncated": true' in prompt

    # test_report_findings_take_priority_over_large_source_diffs
    _, _, history = _trees(tmp_path / "priority")
    report = "context " * 450 + " important final finding"
    for index in range(8):
        _write_record(history, index, diff="changed source\n" * 4000, reports=[{"text": report}])
    prompt = render_repair_history(history, current_fingerprint=BASE)
    assert prompt.count("important final finding") == 8
    assert len(prompt.encode()) <= 64 * 1024


def test_history_renders_untrusted_deltas_and_ignores_missing_or_malformed_records(
    tmp_path: Path,
) -> None:
    # test_next_repair_sees_multiple_deltas_as_untrusted_not_current_state
    task, transaction, history = _trees(tmp_path / "deltas")
    (task / "evaluate.py").write_text("timeout = 300\n")
    capture(task, transaction, history)
    (task / "evaluate.py").write_text("reset_count = 2\n")
    capture_rejected_repair(
        task, transaction, history / "qualification_1_transaction.json",
        base_fingerprint=BASE, reports=[],
    )
    text = render_repair_history(history, current_fingerprint=BASE)
    assert "untrusted" in text.lower()
    assert "timeout = 300" in text
    assert "reset_count = 2" in text
    assert "current_task_fingerprint" in text
    assert "These edits were rolled back" not in text
    assert "may remain in the provisional tree" in text
    assert len(text.encode()) <= 64 * 1024

    # test_missing_history_is_empty
    assert render_repair_history(tmp_path / "missing", current_fingerprint=BASE) == ""

    # test_malformed_and_symlink_history_records_are_ignored
    task, transaction, history = _trees(tmp_path / "malformed")
    (history / "qualification_0_transaction.json").write_text("not JSON")
    (history / "qualification_1_transaction.json").write_text('{"schema_version":99}')
    (history / "qualification_2_transaction.json").write_text('{"schema_version":1,"reports":3}')
    (history / "qualification_3_transaction.json").symlink_to(task / "missing")
    assert render_repair_history(history, current_fingerprint=BASE) == ""


def test_capture_refuses_symlinks_and_preserves_quota_errors_in_library_and_cli(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # test_refuses_symlink_output_and_parent
    task, transaction, history = _trees(tmp_path / "symlink")
    target = tmp_path / "untouched"
    target.write_text("safe")
    (history / "qualification_0_transaction.json").symlink_to(target)
    with pytest.raises(ValueError):
        capture(task, transaction, history)
    assert target.read_text() == "safe"
    alias = tmp_path / "alias"
    alias.symlink_to(history, target_is_directory=True)
    with pytest.raises(ValueError):
        capture(task, transaction, alias)

    # test_history_cleanup_does_not_replace_primary_quota_error
    task, transaction, history = _trees(tmp_path / "quota")

    def fail_sync(fd):
        raise OSError(errno.EDQUOT, "quota")

    original = Path.unlink

    def cleanup(path, *args, **kwargs):
        if path.name.startswith(".repair-context-"):
            raise OSError(errno.EIO, "cleanup")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_sync)
        patch.setattr(Path, "unlink", cleanup)
        with pytest.raises(OSError) as error:
            capture(task, transaction, history)
    assert error.value.errno == errno.EDQUOT

    # test_history_cli_reports_quota_as_infrastructure
    monkeypatch.setattr(sys, "argv", [
        "repair_context", "--task-dir", "/unused", "--transaction-dir", "/unused",
        "--output", "/unused", "--base-fingerprint", BASE,
    ])

    def fail(*args, **kwargs):
        raise OSError(errno.EDQUOT, "secret-path")

    monkeypatch.setattr(repair_context, "capture_rejected_repair", fail)
    with pytest.raises(SystemExit) as exit_info:
        repair_context._main()
    assert exit_info.value.code == 4
    diagnostic = capsys.readouterr().err
    assert "storage_quota_exceeded" in diagnostic
    assert "secret-path" not in diagnostic


def test_shell_rollback_retains_delta_and_survives_history_write_failure(tmp_path: Path) -> None:
    # test_real_rollback_retains_delta_and_next_prompt_receives_it
    from src.investigation.task_alignment import task_tree_fingerprint
    from src.investigation.task_repair_transaction import create_tree_repair_transaction

    task = tmp_path / "task"
    task.mkdir()
    (task / "evaluate.py").write_text("timeout = 90\n")
    (task / "task.md").write_text("Implement optimizer")
    fingerprint = task_tree_fingerprint(task)
    transaction = tmp_path / "qualification_0_transaction"
    create_tree_repair_transaction(task, transaction)
    (task / "evaluate.py").write_text("timeout = 300\n")
    history = tmp_path / "history"
    history.mkdir()
    report = tmp_path / "failed.json"
    report.write_text('{"category":"missing_reset"}')
    functions = [
        _shell_function(name)
        for name in ("restore_repair_transaction", "render_qualification_repair_prompt")
    ]
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -eu\nsource "$REPO_ROOT/src/scripts/lib/task_qualification.sh"\n'
            + "\n".join(functions)
            + "\ndiscard_staged_runtime_for_repair() { :; }\n"
            + 'restore_repair_transaction "$ACTIVE_QUALIFICATION_TRANSACTION"\n'
            + 'render_qualification_repair_prompt "$FAILED_QUALIFICATION_REPORT" "$ACTIVE_QUALIFICATION_BASE_FINGERPRINT"',
        ],
        cwd=ROOT,
        env=os.environ
        | {
            "REPO_ROOT": str(ROOT),
            "PYTHON_BIN": sys.executable,
            "STAGED_TASK_DIR": str(task),
            "TASK_DIR": str(task),
            "OUTPUT_DIR": str(tmp_path),
            "VALIDATION_ROOT": str(history),
            "FAILED_QUALIFICATION_REPORT": str(report),
            "ACTIVE_QUALIFICATION_TRANSACTION": str(transaction),
            "ACTIVE_QUALIFICATION_BASE_FINGERPRINT": fingerprint,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (task / "evaluate.py").read_text() == "timeout = 90\n"
    assert "timeout = 300" in result.stdout
    assert "current_task_fingerprint" in result.stdout
    assert not transaction.exists()

    # test_history_write_failure_does_not_prevent_rollback
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -eu\n"
            + _shell_function("restore_repair_transaction")
            + "retain_rejected_qualification_context() { return 4; }\n"
            + "discard_staged_runtime_for_repair() { :; }\n"
            + "fake_python() { printf '%s\\n' \"$*\"; }\n"
            + "PYTHON_BIN=fake_python\nSTAGED_TASK_DIR=task\n"
            + "ACTIVE_QUALIFICATION_TRANSACTION=transaction\n"
            + "ACTIVE_QUALIFICATION_BASE_FINGERPRINT=fingerprint\n"
            + "restore_repair_transaction transaction",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 4
    assert "task_repair_transaction restore" in result.stdout
    assert "task_repair_transaction commit" in result.stdout


# --- real child stderr reaches repair context, never the alignment judge --------------------


def test_child_failure_reaches_repair_without_judge_text_and_opaque_prompt_requests_instrumentation(
    tmp_path, monkeypatch,
):
    # test_child_failure_reaches_qualification_and_repair_without_judge_text
    fragment = (PROMPTS / "shared/worker_diagnostics.md").read_text()
    recipe = "\n".join(re.findall(r"```python\n(.*?)\n```", fragment, re.DOTALL))
    for index, message in enumerate(
        ["ValueError: invalid worker configuration", "password=secret"]
    ):
        root = tmp_path / f"child-{index}"
        root.mkdir()
        task = _write_contract_task(root)
        # Execute repository-owned recipe with a fixed test child, never a generated task.
        child = "import sys; print(sys.argv[1], file=sys.stderr); sys.exit(1)"
        program = recipe + "\n" + "\n".join([
            "import subprocess",
            f"child = subprocess.Popen([sys.executable, '-c', {child!r}, {message!r}],",
            "                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)",
            "capture = BoundedWorkerStderr(child.stderr)",
            "code = child.wait(timeout=5)",
            "snapshot = capture.close()",
            "sys.stderr.write(snapshot['stderr'])",
            "sys.exit(code)",
        ])

        def prepare_runtime(*args, _task=task, _program=program, **kwargs):
            result = runtime._run_checked(
                [sys.executable, "-c", _program], task_dir=_task,
                deadline=time.monotonic() + 10, environment=dict(os.environ),
                run=subprocess.run, stage="baseline_validation",
            )
            assert result.status is runtime._CheckedCommandStatus.COMMAND_FAILED
            return runtime.TaskRuntimePreparationResult.AUTHORED_FAILURE

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "prepare_task_runtime_result", prepare_runtime)
            report = runtime.prepare_task_runtime_report(task)
            patch.setattr(
                "src.task_drafting.publication.task_publication_plan_sha256",
                lambda **kwargs: "1" * 64,
            )
            attempt = qualify_task_for_publication(
                task_dir=task, publication_task_dir=task, output_dir=root,
                receipt_path=root / "receipt.json", timeout_seconds=30,
                container_image="unused", container_network="none",
                install_cache_dir=root / "cache",
                max_runtime_bytes=1024, prepare=lambda *args, **kwargs: report,
            )
            assert attempt.status == "authored_failure", message
            assert attempt.output_evidence.task_fingerprint == report.task_fingerprint
            report_path = root / "qualification_0.json"
            report_path.write_text(json.dumps(attempt.to_payload()))
            history = root / "history"
            history.mkdir()
            (history / "qualification_0_transaction.json").write_text(json.dumps({
                "schema_version": 1, "base_fingerprint": report.task_fingerprint,
                "rejected_fingerprint": report.task_fingerprint,
                "diff": "+ stderr=subprocess.PIPE\n", "reports": [{
                    "text": report_path.read_text(), "sha256": "a" * 64,
                }],
            }))
            prompt = render_qualification_prompt(
                patch, report_path, history, report.task_fingerprint
            )
        encoded = re.search(r"Qualification diagnostic \(base64\): (\S+)", prompt).group(1)
        diagnostic = json.loads(base64.b64decode(encoded))
        expected = message if not message.startswith("password=") else "<redacted-sensitive-text>"
        assert expected in diagnostic["output_evidence"]["stderr"], message
        assert expected in prompt, message  # Retained, inert rejected-repair history.
        assert "password=secret" not in prompt
        projection = qualification_failure_projection(
            report_path, current_fingerprint=report.task_fingerprint,
        )
        assert expected not in json.dumps(projection), message
        assert projection["untrusted_diagnostic_omitted"], message

    # test_opaque_qualification_repair_requests_causal_instrumentation
    prompt = render_qualification_prompt(monkeypatch)
    assert "instrument the suppressing wrapper" in prompt
    assert "does not prove a timeout" in prompt
    assert "production-entry stops after the first reset" in prompt
    assert "rolled-back diagnostic changes" in prompt
    assert "do not assume a rollback occurred" in prompt
