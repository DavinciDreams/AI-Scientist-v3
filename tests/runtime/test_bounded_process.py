"""Bounded subprocess execution and the sanitized output/timeout evidence built from it."""
from __future__ import annotations

import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

import src.investigation.bounded_process as bounded_process_module
import src.investigation.task_runtime as runtime
from src.investigation import script_runners
from src.investigation.models import IdeaGenerationResult
from src.investigation.bounded_process import (
    ProcessCancelledError,
    ProcessCleanupWitness,
    ProcessOutputLimitError,
    ProcessTimeoutError,
    run_bounded_process,
)
from src.investigation.runtime_output import output_limit_evidence, timeout_evidence
from src.investigation.task_contract import load_task_contract
from src.task_drafting.qualification import qualify_task_for_publication
from tests.runtime.test_task_runtime import _fake_broker, _stub_commands, _write_contract_task

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_DIAGNOSTICS = REPO_ROOT / "src/prompts/shared/worker_diagnostics.md"
_STAGE = "baseline_validation"


# --- helpers --------------------------------------------------------------------------------


def _run_python(source: str, tmp_path: Path, *, timeout_seconds: float = 2, max_output_bytes: int = 1024):
    return run_bounded_process(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=os.environ,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )


def _tracked_popen(monkeypatch: pytest.MonkeyPatch, *, require_no_shell: bool = False) -> list:
    created: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def tracked(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        if require_no_shell:
            assert kwargs.get("shell") is False
        process = real_popen(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", tracked)
    return created


def _assert_pipes_closed(created: list) -> None:
    assert created
    assert all(process.stdout is not None and process.stdout.closed for process in created)
    assert all(process.stderr is not None and process.stderr.closed for process in created)


def _reap(created: list) -> None:
    for process in created:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)


def _fail_teardown(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("injected teardown failure")


@contextmanager
def _diagnostic_handler():
    handler = runtime._RuntimeReportDiagnosticHandler()
    runtime.logger.addHandler(handler)
    try:
        yield handler
    finally:
        runtime.logger.removeHandler(handler)


def _checked(task_dir: Path, run, **kwargs):
    return runtime._run_checked(
        ["evaluator"], task_dir=task_dir, deadline=time.monotonic() + 30, environment={}, run=run, **kwargs
    )


def _limit_evidence(error: ProcessOutputLimitError, *, stage=_STAGE, fingerprint="", environment=None):
    return output_limit_evidence(
        error, stage=stage, task_fingerprint=fingerprint, environment=environment or {}
    )


def _timeout_evidence(error):
    return timeout_evidence(error, stage=_STAGE, environment={})


# --- run_bounded_process --------------------------------------------------------------------


def test_captures_immutable_output_returns_nonzero_exit_and_never_invokes_shell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created = _tracked_popen(monkeypatch, require_no_shell=True)

    # Former: captures stdout and stderr as immutable bytes.
    result = _run_python("import sys; sys.stdout.buffer.write(b'out'); sys.stderr.buffer.write(b'err')", tmp_path)

    assert result.returncode == 0
    assert result.stdout == b"out"
    assert result.stderr == b"err"
    with pytest.raises((AttributeError, TypeError)):
        result.stdout = b"changed"

    # Former: nonzero exit is returned without raising.
    result = _run_python("raise SystemExit(7)", tmp_path)

    assert result.returncode == 7
    assert result.stdout == b""
    assert result.stderr == b""

    # Former: never invokes the shell (and closes its pipes).
    assert _run_python("pass", tmp_path).returncode == 0
    _assert_pipes_closed(created)


def test_output_flood_hits_combined_cap_quickly_even_when_process_exits_immediately(tmp_path: Path) -> None:
    # Former: output flood hits the combined cap quickly without the payload in the error.
    secret = "SMOKE_PAYLOAD_MUST_NOT_LEAK"
    started = time.monotonic()

    with pytest.raises(ProcessOutputLimitError) as caught:
        _run_python(
            f"import os\nchunk = {secret.encode()!r} * 1024\nwhile True:\n    os.write(1, chunk)\n    os.write(2, chunk)\n",
            tmp_path,
            timeout_seconds=10,
            max_output_bytes=4096,
        )

    assert time.monotonic() - started < 3
    assert secret not in str(caught.value)

    # Former: output limit is enforced when the process exits immediately.
    with pytest.raises(ProcessOutputLimitError):
        _run_python("import os; os.write(1, b'x' * 4097)", tmp_path, max_output_bytes=4096)


def test_output_limit_retains_bounded_stream_evidence_and_preserves_small_other_stream(tmp_path: Path) -> None:
    # Former: output limit retains bounded stream evidence (stdout, stderr).
    for stream in ("stdout", "stderr"):
        descriptor = 1 if stream == "stdout" else 2
        with pytest.raises(ProcessOutputLimitError) as caught:
            _run_python(f"import os; os.write({descriptor}, b'private output' * 1024)", tmp_path, max_output_bytes=5000)

        error = caught.value
        assert error.max_output_bytes == 5000, stream
        assert error.observed_output_bytes > 5000, stream
        assert len(getattr(error, stream)) == 4096, stream
        assert len(error.stdout) <= 4096, stream
        assert len(error.stderr) <= 4096, stream
        assert "private output" not in str(error), stream
        assert "private output" not in repr(error), stream

    # Former: output limit preserves the small stream when the other stream floods.
    with pytest.raises(ProcessOutputLimitError) as caught:
        _run_python(
            "import os, time; os.write(2, b'error component'); time.sleep(0.05); os.write(1, b'x' * 8192)",
            tmp_path,
            max_output_bytes=5000,
        )
    assert caught.value.stderr == b"error component"
    assert len(caught.value.stdout) == 4096
    assert caught.value.observed_output_bytes > 5000


def test_output_limit_keeps_payload_private_and_retains_triggering_and_interrupted_streams(tmp_path: Path) -> None:
    # Former: output limit legacy constructor keeps the message payload private.
    error = ProcessOutputLimitError("private console content")
    assert "private console content" not in str(error)
    assert "private console content" not in repr(error)

    # Former: output limit retains the triggering stream when the capture is full.
    with pytest.raises(ProcessOutputLimitError) as caught:
        _run_python(
            "import os, time; os.write(1, b'x' * 4096); time.sleep(0.05); os.write(2, b'triggering error')",
            tmp_path,
            max_output_bytes=4096,
        )
    assert caught.value.stdout == b"x" * 4096
    assert caught.value.stderr == b"triggering error"
    assert caught.value.observed_output_bytes == 4096 + len(b"triggering error")

    # Former: output limit marks the interrupted small stream as truncated.
    with pytest.raises(ProcessOutputLimitError) as caught:
        _run_python(
            "import os, time; os.write(1, b'partial-secret'); time.sleep(0.05); "
            "os.write(2, b'x' * 8192); time.sleep(60)",
            tmp_path,
            max_output_bytes=5000,
        )
    assert caught.value.stdout == b"partial-secret"
    assert caught.value.stdout_truncated
    assert caught.value.stderr_truncated


def test_timeout_terminates_process_and_covers_pipe_drain_after_parent_exits(tmp_path: Path) -> None:
    # Former: timeout terminates the process without the payload in the error.
    secret = "TIMEOUT_PAYLOAD_MUST_NOT_LEAK"
    started = time.monotonic()

    with pytest.raises(ProcessTimeoutError) as caught:
        _run_python(f"import sys, time; print({secret!r}, flush=True); time.sleep(60)", tmp_path, timeout_seconds=0.1)

    assert time.monotonic() - started < 3
    assert secret not in str(caught.value)

    # Former: timeout covers the pipe drain after the parent exits.
    with pytest.raises(ProcessTimeoutError):
        _run_python(
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print('parent exited', flush=True)\n",
            tmp_path,
            timeout_seconds=0.1,
        )


def test_timeout_retains_bounded_output_and_keeps_payload_private(tmp_path: Path) -> None:
    # Former: timeout retains bounded output without the payload in the error.
    with pytest.raises(ProcessTimeoutError) as caught:
        run_bounded_process(
            [sys.executable, "-c", "import os,time; os.write(1,b'private stdout\\n'); "
             "os.write(2,b'worker failed\\n'+b'x'*5000); time.sleep(60)"],
            cwd=tmp_path, env=os.environ, timeout_seconds=0.3,
        )
    error = caught.value
    assert error.stdout == b"private stdout\n"
    assert error.stderr.startswith(b"worker failed\n")
    assert len(error.stderr) == 4096
    assert error.observed_output_bytes == len(b"private stdout\nworker failed\n") + 5000
    assert error.stdout_truncated and error.stderr_truncated
    assert "worker failed" not in str(error)
    assert "private stdout" not in repr(error)

    # Former: timeout constructor keeps the legacy message private.
    error = ProcessTimeoutError("private console content", stderr=b"x" * 5000)
    assert "private console content" not in str(error)
    assert "private console content" not in repr(error)
    assert len(error.stderr) == 4096
    assert error.stderr_truncated


def test_rejects_invalid_timeout_output_cap_and_string_command(tmp_path: Path) -> None:
    # Former: rejects an invalid timeout.
    for timeout_seconds in (0, -1, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            _run_python("pass", tmp_path, timeout_seconds=timeout_seconds)

    # Former: rejects an invalid output cap.
    for max_output_bytes in (0, -1, math.nan, math.inf, 1.5, True):
        with pytest.raises(ValueError):
            run_bounded_process(
                [sys.executable, "-c", "pass"],
                cwd=tmp_path,
                env=os.environ,
                timeout_seconds=1,
                max_output_bytes=cast(Any, max_output_bytes),
            )

    # Former: rejects a string command instead of argv.
    with pytest.raises(ValueError):
        run_bounded_process(f"{sys.executable} -c pass", cwd=tmp_path, env=os.environ, timeout_seconds=1)


def test_child_umask_secures_files_without_changing_parent_umask(tmp_path: Path) -> None:
    output = tmp_path / "child-created"
    original_umask = os.umask(0o002)
    try:
        result = run_bounded_process(
            [sys.executable, "-c", f"from pathlib import Path; Path({str(output)!r}).touch()"],
            cwd=tmp_path,
            env=os.environ,
            timeout_seconds=2,
            child_umask=0o077,
        )
    finally:
        observed_umask = os.umask(original_umask)
    assert result.returncode == 0
    assert output.stat().st_mode & 0o777 == 0o600
    assert observed_umask == 0o002


def test_retained_output_limit_exceptions_do_not_retain_open_pipes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created = _tracked_popen(monkeypatch)
    caught_errors: list[ProcessOutputLimitError] = []
    for _ in range(20):
        with pytest.raises(ProcessOutputLimitError) as caught:
            _run_python("import os; os.write(1, b'x' * 4097)", tmp_path, max_output_bytes=4096)
        caught_errors.append(caught.value)

    assert len(caught_errors) == 20
    _assert_pipes_closed(created)


@pytest.mark.parametrize("primary", ["timeout", "output_limit"])
def test_teardown_failure_does_not_mask_primary_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, primary) -> None:
    created = _tracked_popen(monkeypatch)
    monkeypatch.setattr(bounded_process_module, "_stop_and_reap", _fail_teardown)
    try:
        if primary == "timeout":
            with pytest.raises(ProcessTimeoutError):
                _run_python("import time; time.sleep(60)", tmp_path, timeout_seconds=0.05)
        else:
            with pytest.raises(ProcessOutputLimitError):
                _run_python("import os; os.write(1, b'x' * 4097)", tmp_path, max_output_bytes=4096)
    finally:
        _reap(created)

    if primary == "timeout":
        _assert_pipes_closed(created)


def test_setsid_descendant_cannot_block_deadline_or_keep_local_pipes_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created = _tracked_popen(monkeypatch)
    escaped_pid_path = tmp_path / "escaped.pid"
    started = time.monotonic()
    try:
        with pytest.raises(ProcessTimeoutError):
            _run_python(
                "import pathlib, subprocess, sys, time\n"
                "child = subprocess.Popen(\n"
                "    [sys.executable, '-c', 'import time; time.sleep(60)'],\n"
                "    stdout=sys.stdout, stderr=sys.stderr, start_new_session=True,\n"
                ")\n"
                f"pathlib.Path({str(escaped_pid_path)!r}).write_text(str(child.pid))\n"
                "time.sleep(60)\n",
                tmp_path,
                timeout_seconds=0.2,
            )
        assert escaped_pid_path.is_file()
    finally:
        if escaped_pid_path.is_file():
            try:
                os.kill(int(escaped_pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
            except ProcessLookupError:
                pass

    assert time.monotonic() - started < 3
    _assert_pipes_closed(created)


def test_cancellation_reaps_running_controller(tmp_path: Path) -> None:
    cancelled = threading.Event()
    timer = threading.Timer(0.2, cancelled.set)
    witness = ProcessCleanupWitness()
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(ProcessCancelledError):
            run_bounded_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                cwd=tmp_path, env=os.environ, timeout_seconds=60,
                cancel_event=cancelled, cleanup_witness=witness,
            )
    finally:
        timer.join()
    assert witness.reaped
    assert time.monotonic() - started < 5


# --- output / timeout evidence sanitization -------------------------------------------------


_TRUNCATION_CREDENTIALS = [
    b"abcdefghijklmnopqrstuvwx", b"https://user:password@example.invalid/resource", b'password="short"'
]
_SHORT_CREDENTIALS = [b"password=short", b'password="x"', b"'token': 'x'", b"api_key=x", b"secret=x"]
_RECOGNIZABLE_CREDENTIALS = [
    (b"api_key=sk-proj-abcdefghijklmnop", {}),
    (b"https://user:password@example.invalid/error", {}),
    (b"credentials: short-secret-value", {"SERVICE_TOKEN": "short-secret-value"}),
    (b"-----BEGIN PRIVATE KEY-----\nprivate-key-content", {}),
]


def test_output_evidence_discards_incomplete_credentials_redacts_complete_ones_and_bounds_controls() -> None:
    # Former: truncated output discards an incomplete credential line.
    for prefix_size in (4070, 4088, 4095):
        for credential in _TRUNCATION_CREDENTIALS:
            case = (prefix_size, credential)
            safe_prefix = b"component started\n"
            payload = safe_prefix + b"x" * (prefix_size - len(safe_prefix)) + credential + b"z" * 64
            evidence = _limit_evidence(
                ProcessOutputLimitError(stdout=payload, stderr=payload),
                environment={"SERVICE_TOKEN": "abcdefghijklmnopqrstuvwx"},
            )
            assert evidence.stdout == "component started\\x0a", case
            assert evidence.stderr == evidence.stdout, case
            assert evidence.stdout_truncated, case
            assert evidence.stderr_truncated, case

    # Former: output redacts short bare and quoted credentials.
    for credential in _SHORT_CREDENTIALS:
        evidence = _limit_evidence(ProcessOutputLimitError(stdout=credential), stage="dependency_install")
        assert evidence.stdout.startswith("<redacted-sensitive-text"), credential

    # Former: output evidence redacts recognizable credentials.
    for payload, environment in _RECOGNIZABLE_CREDENTIALS:
        evidence = _limit_evidence(
            ProcessOutputLimitError(stdout=payload, stderr=payload),
            stage="dependency_install",
            fingerprint="sha256:" + "a" * 64,
            environment=environment,
        )
        assert evidence.stdout.startswith("<redacted-sensitive-text"), payload
        assert evidence.stderr.startswith("<redacted-sensitive-text"), payload

    # Former: output evidence bounds expanding control characters.
    evidence = _limit_evidence(
        ProcessOutputLimitError(stdout=b"\x00" * 4096, stderr="é".encode() * 4096),
        fingerprint="sha256:" + "a" * 64,
    )
    assert len(evidence.stdout.encode()) <= 4096
    assert len(evidence.stderr.encode()) <= 4096
    assert "\x00" not in evidence.stdout


def test_timeout_evidence_discards_stdout_redacts_credentials_bounds_output_and_keeps_authored_deadline(
    tmp_path: Path,
) -> None:
    # Former: timeout evidence discards stdout and incomplete credentials (bounded, subprocess bytes/text).
    stderr = b"worker failed\nAuthorization: Bearer partial"
    for kind in ("bounded", "subprocess_bytes", "subprocess_text"):
        error = (
            ProcessTimeoutError(stderr=stderr, stderr_truncated=True, stdout=b"private")
            if kind == "bounded"
            else subprocess.TimeoutExpired(
                ["evaluator"], 1, output=b"private", stderr=stderr.decode() if kind == "subprocess_text" else stderr
            )
        )
        evidence = _timeout_evidence(error)
        assert evidence.failure_category == "timeout", kind
        assert evidence.stderr == "worker failed\\x0a", kind
        assert evidence.stderr_truncated, kind
        assert evidence.stdout == "", kind

    # Former: timeout evidence redacts complete credentials.
    for payload in (b"Authorization: Bearer private\n", b"password=x\n"):
        assert _timeout_evidence(ProcessTimeoutError(stderr=payload)).stderr == "<redacted-sensitive-text>", payload

    # Former: timeout evidence sanitizes controls and bounds compatible output.
    evidence = _timeout_evidence(subprocess.TimeoutExpired(["evaluator"], 1, stderr=b"\x00\x1b[31m\n" * 2000))
    assert len(evidence.stderr.encode()) <= 4096
    assert "\x00" not in evidence.stderr and "\x1b" not in evidence.stderr

    # Former: a silent timeout keeps empty evidence.
    evidence = _timeout_evidence(subprocess.TimeoutExpired(["evaluator"], 1))
    assert evidence.stderr == ""
    assert evidence.observed_output_bytes == 0

    # Former: timeout evidence does not override the task-authored deadline.
    report = runtime.runtime_preparation_report_from_result(
        runtime.TaskRuntimePreparationResult.AUTHORED_FAILURE,
        task_dir=tmp_path,
        output_evidence=_timeout_evidence(ProcessTimeoutError(stderr=b"worker failed\n")),
    )
    assert report.failure_category == "runtime_baseline_validation"
    assert report.failure_stage == "baseline_validation"


# --- runtime report integration -------------------------------------------------------------


@pytest.mark.parametrize("stage", ["baseline_validation", "dependency_install"])
def test_overflow_report_preserves_sanitized_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage) -> None:
    task_dir = _write_contract_task(tmp_path)

    def overflow(*_args: object, **_kwargs: object) -> None:
        raise ProcessOutputLimitError(
            max_output_bytes=65536,
            observed_output_bytes=65537,
            stdout=b"stdout component\x1b[31m\x00\n" + b"x" * 5000,
            stderr=b"Authorization: Bearer private-test-credential",
        )

    def fail_runtime(*_args: object, **_kwargs: object) -> runtime.TaskRuntimePreparationResult:
        result = _checked(task_dir, subprocess.run, stage=stage)
        assert result.status is runtime._CheckedCommandStatus.COMMAND_FAILED
        return runtime.TaskRuntimePreparationResult.AUTHORED_FAILURE

    monkeypatch.setattr(runtime, "run_bounded_process", overflow)
    monkeypatch.setattr(runtime, "prepare_task_runtime_result", fail_runtime)
    report = runtime.prepare_task_runtime_report(task_dir)

    assert report.failure_stage == stage
    assert report.failure_category == "output_limit_exceeded"
    assert "failed to launch" not in report.diagnostic
    assert "console" in report.diagnostic
    assert "RPC" in report.diagnostic
    evidence = report.output_evidence
    assert evidence is not None
    assert evidence.stage == stage
    assert evidence.task_fingerprint == report.task_fingerprint
    assert evidence.max_output_bytes == 65536
    assert evidence.observed_output_bytes == 65537
    assert evidence.stdout.startswith("stdout component")
    assert len(evidence.stdout.encode()) <= 4096
    assert "\x1b" not in evidence.stdout
    assert "\x00" not in evidence.stdout
    assert "private-test-credential" not in evidence.stderr
    assert report.failure_evidence is None

    output_dir = tmp_path / "validation"
    output_dir.mkdir()
    monkeypatch.setattr("src.task_drafting.publication.task_publication_plan_sha256", lambda **_kwargs: "1" * 64)
    attempt = qualify_task_for_publication(
        task_dir=task_dir,
        publication_task_dir=task_dir,
        output_dir=output_dir,
        receipt_path=output_dir / "receipt.json",
        timeout_seconds=30,
        container_image="image",
        container_network="none",
        install_cache_dir=tmp_path / "cache",
        max_runtime_bytes=1024,
        prepare=lambda *_args, **_kwargs: report,
    )
    assert attempt.category == "output_limit_exceeded"
    assert attempt.output_evidence == evidence
    assert "private-test-credential" not in json.dumps(attempt.to_payload())


def test_overflow_fingerprints_once_after_runtime_teardown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preparing = False
    fingerprint_calls = 0

    def fingerprint(_task_dir: Path) -> str:
        nonlocal fingerprint_calls
        assert not preparing, "Do not hash the scaffold while the evaluator broker is alive"
        fingerprint_calls += 1
        return "sha256:" + "a" * 64

    def overflow(*_args: object, **_kwargs: object) -> None:
        raise ProcessOutputLimitError(stdout=b"output")

    def fail_runtime(*_args: object, **_kwargs: object) -> runtime.TaskRuntimePreparationResult:
        nonlocal preparing
        preparing = True
        _checked(tmp_path, subprocess.run)
        preparing = False
        return runtime.TaskRuntimePreparationResult.AUTHORED_FAILURE

    monkeypatch.setattr(runtime, "_safe_task_fingerprint", fingerprint)
    monkeypatch.setattr(runtime, "run_bounded_process", overflow)
    monkeypatch.setattr(runtime, "prepare_task_runtime_result", fail_runtime)
    report = runtime.prepare_task_runtime_report(tmp_path)
    assert fingerprint_calls == 1
    assert report.output_evidence is not None
    assert report.output_evidence.task_fingerprint == report.task_fingerprint


def test_failed_validation_nonzero_command_and_late_completion_retain_sanitized_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Former: failed validation retains sanitized stderr (exit 0, 1, -9).
    for returncode in (0, 1, -9):
        process = subprocess.CompletedProcess(
            ["evaluator"], returncode, "candidate payload must not be retained",
            "episode=2 operation=train signal=SIGKILL\nAuthorization: Bearer private-credential",
        )
        with _diagnostic_handler() as handler:
            runtime._log_failed_evaluation(process, environment={})
        evidence = handler.output_evidence()
        assert evidence is not None, returncode
        assert evidence.failure_category == "evaluation_failed", returncode
        assert evidence.stdout == "", returncode
        assert "private-credential" not in evidence.stderr, returncode
        report = runtime.runtime_preparation_report_from_result(
            runtime.TaskRuntimePreparationResult.AUTHORED_FAILURE, task_dir=tmp_path, output_evidence=evidence
        )
        assert report.failure_category == "evaluation_failed", returncode

    # Former: a nonzero command attaches output evidence.
    def failing(*_args, **_kwargs):
        return subprocess.CompletedProcess(["evaluator"], 1, "", "episode=1 operation=train timeout=rpc\n")

    with _diagnostic_handler() as handler:
        result = _checked(tmp_path, failing, stage=_STAGE)
    assert result.status is runtime._CheckedCommandStatus.COMMAND_FAILED
    assert "operation=train" in handler.output_evidence().stderr

    # Former: a completed command exceeding the deadline retains stderr.
    now = [1.0]

    def late_result(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        now[0] = 11.0
        return subprocess.CompletedProcess(["evaluator"], 0, "private stdout", "worker failed\n")

    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    with _diagnostic_handler() as handler, pytest.raises(runtime.TaskRuntimeDeadlineExceeded):
        runtime._run_checked(
            ["evaluator"], task_dir=tmp_path, deadline=10.0, execution_deadline=10.0,
            environment={}, run=late_result, stage=_STAGE,
        )
    evidence = handler.output_evidence()
    assert evidence is not None
    assert evidence.stderr == "worker failed\\x0a"
    assert evidence.stdout == ""
    assert evidence.failure_category == "timeout"


@pytest.mark.parametrize("failure", ["reset_sequence", "feasibility"])
def test_exit_zero_contracted_validation_failure_retains_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure) -> None:
    task = _write_contract_task(tmp_path)
    contract = load_task_contract(task / "task_contract.json")
    _stub_commands(monkeypatch, _fake_broker(reset_seeds=() if failure == "reset_sequence" else (1,)))
    monkeypatch.setattr(runtime, "_validated_contracted_baseline_score", lambda *_a, **_k: None)

    def exit_zero_with_stderr(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, "private stdout", "episode=2 operation=train timeout=rpc\n")

    with _diagnostic_handler() as handler:
        result = runtime._run_contracted_baseline(
            task_dir=task, runtime_venv=task / ".venv", image_id="sha256:" + "a" * 64,
            contract=contract, entrypoint="main.py", deadline=time.monotonic() + 60,
            environment={}, run=exit_zero_with_stderr,
        )
    assert result.status is runtime._CheckedCommandStatus.COMMAND_FAILED
    assert "operation=train" in handler.output_evidence().stderr
    assert handler.output_evidence().stdout == ""


def test_timeout_report_preserves_classification_and_cleanup_even_when_logging_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Former: timeout report preserves classification and cleanup (compatible x cleanup_succeeded).
    for compatible in (True, False):
        for cleanup_succeeded in (True, False):
            case = (compatible, cleanup_succeeded)
            case_dir = tmp_path / f"report-{compatible}-{cleanup_succeeded}"
            case_dir.mkdir()
            cid_path = case_dir / "container.cid"
            cid_path.write_text("container-id")

            def timeout(*_args: object, compatible=compatible, **_kwargs: object) -> subprocess.CompletedProcess[str]:
                if compatible:
                    raise subprocess.TimeoutExpired(["evaluator"], 1, stderr=b"worker failed\n")
                raise ProcessTimeoutError(stderr=b"worker failed\n", stderr_truncated=True)

            def fail_runtime(
                *_args: object, timeout=timeout, compatible=compatible, cleanup_succeeded=cleanup_succeeded,
                case_dir=case_dir, cid_path=cid_path, **_kwargs: object,
            ) -> runtime.TaskRuntimePreparationResult:
                expected_error = (
                    runtime.TaskRuntimeDeadlineExceeded
                    if cleanup_succeeded
                    else runtime._TaskRuntimeCleanupDeadlineExceeded
                )
                with pytest.raises(expected_error):
                    _checked(case_dir, timeout if compatible else subprocess.run, stage=_STAGE, cid_path=cid_path)
                return runtime.TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE

            with monkeypatch.context() as m:
                m.setattr(runtime, "run_bounded_process", timeout)
                m.setattr(runtime, "prepare_task_runtime_result", fail_runtime)
                m.setattr(runtime, "_remove_container_before_deadline", lambda *_a, value=cleanup_succeeded: value)
                report = runtime.prepare_task_runtime_report(case_dir)
            assert report.failure_stage == "runtime_preparation", case
            assert report.failure_category == "runtime_bootstrap", case
            assert report.output_evidence is not None, case
            assert report.output_evidence.stderr == "worker failed\\x0a", case
            assert report.output_evidence.task_fingerprint == report.task_fingerprint, case
            assert cid_path.exists() is not cleanup_succeeded, case

    # Former: a timeout log failure cannot skip container cleanup.
    for cleanup_succeeded in (True, False):
        case_dir = tmp_path / f"log-{cleanup_succeeded}"
        case_dir.mkdir()
        cid_path = case_dir / "container.cid"
        cid_path.write_text("container-id")
        cleanups: list[Path] = []

        def cleanup(path: Path, deadline: float, cleanups=cleanups, value=cleanup_succeeded) -> bool:
            cleanups.append(path)
            return value

        def broken_log(*_args: object, **_kwargs: object) -> None:
            raise OSError("log storage failed")

        def timeout_without_log(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            raise ProcessTimeoutError(stderr=b"worker failed\n")

        expected = (
            runtime.TaskRuntimeDeadlineExceeded if cleanup_succeeded else runtime._TaskRuntimeCleanupDeadlineExceeded
        )
        with monkeypatch.context() as m:
            m.setattr(runtime.logger, "error", broken_log)
            m.setattr(runtime, "_remove_container_before_deadline", cleanup)
            with pytest.raises(expected):
                _checked(case_dir, timeout_without_log, stage=_STAGE, cid_path=cid_path)
        assert cleanups == [cid_path], cleanup_succeeded
        assert cid_path.exists() is not cleanup_succeeded, cleanup_succeeded


# --- trusted worker diagnostics recipe (src/prompts/shared/worker_diagnostics.md) -----------


def _recipe_source() -> str:
    return "\n".join(re.findall(r"```python\n(.*?)\n```", WORKER_DIAGNOSTICS.read_text(), re.DOTALL))


@pytest.fixture
def recipe() -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(_recipe_source(), str(WORKER_DIAGNOSTICS), "exec"), namespace)
    return namespace


def run_worker(recipe, code):
    process = subprocess.Popen([sys.executable, "-c", code], stderr=subprocess.PIPE)
    capture = recipe["BoundedWorkerStderr"](process.stderr)
    try:
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        result = capture.close()
    return process.returncode, result


def _piped_capture(recipe):
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb")
    return recipe["BoundedWorkerStderr"](stream), stream, write_fd


def test_crashed_silent_and_oversized_line_workers_report_exact_stderr_without_automatic_emission(recipe, capsys: pytest.CaptureFixture[str]) -> None:
    # Former: a crashed worker preserves complete stderr without automatic emission.
    code, evidence = run_worker(recipe, "import sys; print('baseline broke', file=sys.stderr); exit(3)")
    assert code == 3
    assert evidence["stderr"] == "baseline broke\n"
    assert evidence["observed_bytes"] == evidence["retained_bytes"] == 15
    assert not evidence["truncated"]
    assert evidence["capture_error"] is None
    assert capsys.readouterr().err == ""

    # Former: a silent failure explicitly has no stderr.
    code, evidence = run_worker(recipe, "exit(9)")
    assert code == 9
    assert evidence["stderr"] == ""
    assert evidence["observed_bytes"] == 0
    assert evidence["capture_error"] is None

    # Former: oversized and incomplete lines are discarded whole.
    _, evidence = run_worker(
        recipe, "import os; os.write(2, b'X' * 20000 + b'secret\\nvalid\\nAuthorization: hidden')"
    )
    assert evidence["stderr"] == "valid\n"
    assert evidence["truncated"]


def test_large_invalid_and_credential_bearing_output_is_tail_bounded_without_deadlock(recipe) -> None:
    # Former: large output drains without deadlock and retains the complete tail.
    _, evidence = run_worker(recipe, "import os; os.write(2, b'noise\\n' * 100000 + b'cause\\n')")
    assert evidence["stderr"].endswith("cause\n")
    assert evidence["stderr"].startswith("noise\n")
    assert evidence["retained_bytes"] <= 4096
    assert evidence["observed_bytes"] == 600006
    assert evidence["truncated"]

    # Former: invalid UTF-8 remains bounded.
    _, evidence = run_worker(recipe, "import os; os.write(2, b'\\xff' * 4000 + b'\\n')")
    assert len(evidence["stderr"].encode("utf-8")) <= 4096
    assert evidence["retained_bytes"] <= 4096

    # Former: tail capture keeps credentials whole for host redaction.
    _, captured = run_worker(recipe, "import os; os.write(2, b'noise\\n' * 1000 + b'password=hidden\\nfailed\\n')")
    assert "password=hidden\n" in captured["stderr"]
    evidence = _limit_evidence(ProcessOutputLimitError(stderr=captured["stderr"].encode()))
    assert "hidden" not in evidence.stderr
    assert evidence.stderr.startswith("<redacted-sensitive-text")


def test_real_timeout_keeps_output_after_caller_terminates_worker(recipe) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import os,time; os.write(2,b'before hang\\n'); time.sleep(30)"],
        stderr=subprocess.PIPE,
    )
    capture = recipe["BoundedWorkerStderr"](process.stderr)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.2)
    finally:
        process.kill()
        process.wait(timeout=5)
        evidence = capture.close()
    assert evidence["stderr"] == "before hang\n"


def test_piped_capture_closes_reader_on_open_writer_silent_writer_and_read_errors(recipe, monkeypatch: pytest.MonkeyPatch) -> None:
    # Former: an open inherited pipe does not leave the reader alive.
    capture, stream, write_fd = _piped_capture(recipe)
    try:
        os.write(write_fd, b"complete\npartial")
        started = time.monotonic()
        evidence = capture.close()
        assert time.monotonic() - started < 1
        assert not capture.reader.is_alive()
        assert stream.closed
        assert evidence["stderr"] == "complete\n"
        assert evidence["truncated"]
        assert evidence["capture_error"] == "reader_stopped_before_eof"
        assert capture.close() == evidence
    finally:
        os.close(write_fd)

    # Former: a silent open writer marks the capture incomplete.
    capture, _stream, write_fd = _piped_capture(recipe)
    try:
        evidence = capture.close()
        assert evidence["stderr"] == ""
        assert evidence["truncated"]
        assert evidence["capture_error"] == "reader_stopped_before_eof"
    finally:
        os.close(write_fd)

    # Former: a reader error is explicit and still closes the reader.
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb")

    def failed_read(*_args):
        raise OSError("private device detail")

    monkeypatch.setattr(os, "read", failed_read)
    os.write(write_fd, b"wake reader\n")
    capture = recipe["BoundedWorkerStderr"](stream)
    try:
        evidence = capture.close()
        assert evidence["capture_error"] == "stderr_capture_failed"
        assert evidence["truncated"]
        assert "private" not in repr(evidence)
        assert stream.closed
        assert not capture.reader.is_alive()
    finally:
        os.close(write_fd)


def test_worker_exception_report_is_bounded_and_keeps_short_actionable_messages(recipe, capsys: pytest.CaptureFixture[str]) -> None:
    # Former: worker exception report is bounded and omits the exception message.
    recipe["report_worker_exception"](RuntimeError("secret" * 100000))
    diagnostic = capsys.readouterr().err
    assert "RuntimeError" in diagnostic
    assert "secret" not in diagnostic
    assert len(diagnostic.encode()) <= 256

    # Former: worker exception report preserves an actionable short message.
    recipe["report_worker_exception"](ValueError("shape mismatch: expected 8, got 3"))
    assert "shape mismatch: expected 8, got 3" in capsys.readouterr().err


def test_nested_worker_parent_forwards_failure_evidence(recipe) -> None:
    parent = _recipe_source() + "\n" + "\n".join([
        "import subprocess",
        "child_code = \"import sys; print('nested worker broke',file=sys.stderr); exit(4)\"",
        "child = subprocess.Popen([sys.executable,'-c',child_code], stderr=subprocess.PIPE)",
        "capture = BoundedWorkerStderr(child.stderr)",
        "code = child.wait(timeout=2)",
        "evidence = capture.close()",
        "if code: sys.stderr.write(evidence['stderr'])",
        "sys.exit(code)",
    ])
    code, evidence = run_worker(recipe, parent)
    assert code == 4
    assert evidence["stderr"] == "nested worker broke\n"


# --- script runner output bounding ----------------------------------------------------------

_OUTPUT_DIR_SCRIPT = """#!/bin/bash
set -euo pipefail
output_dir=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output-dir" ]; then
    output_dir="$2"
    shift 2
    continue
  fi
  shift
done
mkdir -p "$output_dir"
"""


def _ensure_text(value):
    return value.decode() if isinstance(value, bytes) else value or ""


def _run_idea(tmp_path: Path, output_dir: Path, **overrides: Any) -> IdeaGenerationResult:
    options: dict[str, Any] = {
        "agent": "pi",
        "task_dir": tmp_path / "task",
        "results_dir": tmp_path / "results",
        "global_memory_dir": None,
        "cycle_index": 1,
        "output_dir": output_dir,
        "instruct_path": tmp_path / "idea.md",
        "timeout_seconds": 10,
        "script_path": tmp_path / "idea.sh",
        "popen_factory": subprocess.Popen,
        "register_subprocess": lambda _proc: None,
        "unregister_subprocess": lambda _proc: None,
        "terminate_process_group": lambda _proc, **_kwargs: None,
        "container_execution_env": lambda **_kwargs: os.environ.copy(),
        "ensure_text": _ensure_text,
    }
    return script_runners.run_idea_generation(**(options | overrides))


def _piped_stdout(data: bytes = b"") -> tuple[BinaryIO, int | None]:
    """Readable binary stream; when ``data`` is given the write end is closed, else it is returned open."""
    read_fd, write_fd = os.pipe()
    if data:
        os.write(write_fd, data)
        os.close(write_fd)
        return os.fdopen(read_fd, "rb", buffering=0), None
    return os.fdopen(read_fd, "rb", buffering=0), write_fd


def test_script_output_quota_bounds_unpublished_memory_and_idea_log_while_retaining_marker_and_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script_runners, "SCRIPT_OUTPUT_LOG_MAX_BYTES", 4096)

    # Former: unpublished script output is bounded in memory.
    output, exit_code = script_runners._execute_script_command(
        cmd=["bash", "-c", "printf 'discarded-prefix\\n'; head -c 20000 /dev/zero | tr '\\0' z; printf '\\nretained-tail\\n'"],
        timeout_seconds=10,
        timeout_grace_seconds=1,
        script_label="unpublished",
        missing_output="missing\n",
        popen_factory=subprocess.Popen,
        register_subprocess=lambda _proc: None,
        unregister_subprocess=lambda _proc: None,
        terminate_process_group=lambda _proc, **_kwargs: None,
        ensure_text=_ensure_text,
        env=os.environ.copy(),
    )

    assert exit_code == 0
    assert len(output.encode("utf-8")) <= script_runners.SCRIPT_RESULT_TAIL_BYTES
    assert "output truncated after exceeding 4096 bytes" in output
    assert output.endswith("\nretained-tail\n")

    # Former: idea output log has a hard quota and retains the marker and tail.
    script_path = tmp_path / "over-quota-output.sh"
    script_path.write_text(
        _OUTPUT_DIR_SCRIPT + "printf 'discarded-prefix\\n'\nhead -c 20000 /dev/zero | tr '\\0' z\n"
        "printf '\\nretained-tail\\n' >&2\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "quota-idea"

    result = _run_idea(tmp_path, output_dir, script_path=script_path)

    log_bytes = (output_dir / "wrapper_output.log").read_bytes()
    assert len(log_bytes) <= 4096
    assert b"output truncated after exceeding 4096 bytes" in log_bytes
    assert log_bytes.endswith(b"\nretained-tail\n")
    assert "output truncated after exceeding 4096 bytes" in result.output
    assert result.output.endswith("\nretained-tail\n")


def test_idea_generation_streams_full_output_returns_bounded_tail_and_forwards_harness_flag(tmp_path: Path) -> None:
    # Former: idea generation streams the full output and returns a bounded tail.
    script_path = tmp_path / "large-output.sh"
    script_path.write_text(
        _OUTPUT_DIR_SCRIPT + "printf 'full-output-start\\n'\nhead -c 200000 /dev/zero | tr '\\0' x\n"
        "printf '\\nfull-output-end\\n' >&2\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "idea-attempt"

    result = _run_idea(tmp_path, output_dir, script_path=script_path)

    full_output = (output_dir / "wrapper_output.log").read_text(encoding="utf-8")
    assert result.exit_code == 0
    assert len(result.output.encode("utf-8")) <= script_runners.SCRIPT_RESULT_TAIL_BYTES
    assert "full-output-start" not in result.output
    assert result.output.endswith("\nfull-output-end\n")
    assert full_output.startswith("full-output-start\n")
    assert full_output.endswith("\nfull-output-end\n")
    assert len(full_output) > len(result.output)

    # Former: idea generation forwards the harness flag.
    script_path = tmp_path / "assert-harness.sh"
    script_path.write_text(
        '#!/bin/bash\nset -euo pipefail\nfor argument in "$@"; do\n'
        '  if [ "$argument" = "--harness-evaluated-attempt" ]; then\n    exit 0\n  fi\ndone\nexit 1\n',
        encoding="utf-8",
    )

    result = _run_idea(tmp_path, tmp_path / "harness-attempt", harness_evaluated_attempts=True, script_path=script_path)

    assert result.exit_code == 0


def test_fake_idea_processes_handle_timeout_symlinked_output_dir_and_inherited_open_pipe(tmp_path: Path) -> None:
    # Former: a streamed idea timeout keeps the log and a bounded error tail.
    terminated: list[object] = []
    stdout, _ = _piped_stdout(b"before-timeout\n" + b"y" * 10000)

    class TimedOutProcess:
        args = ["bash"]
        returncode = None
        stdout: BinaryIO

        def poll(self) -> None:
            return None

    TimedOutProcess.stdout = stdout
    output_dir = tmp_path / "timed-out-idea"
    result = _run_idea(
        tmp_path,
        output_dir,
        timeout_seconds=1,
        timeout_grace_seconds=2,
        popen_factory=cast(script_runners.PopenFactory, lambda *_a, **_k: TimedOutProcess()),
        terminate_process_group=lambda proc, **_kwargs: terminated.append(proc),
    )

    full_output = (output_dir / "wrapper_output.log").read_text(encoding="utf-8")
    assert result.exit_code == 124
    assert terminated
    assert len(result.output.encode("utf-8")) <= script_runners.SCRIPT_RESULT_TAIL_BYTES
    assert "before-timeout" in result.output
    assert "timed out after 1 seconds" in result.output
    assert full_output.startswith("before-timeout\n")
    assert full_output.endswith(result.output)

    # Former: idea log publication rejects a symlinked output directory.
    output_dir = tmp_path / "symlinked-idea"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    class CompletedProcess:
        returncode = 0
        stdout = io.BytesIO(b"trusted output\n")

        def wait(self, timeout: int | None = None) -> int:
            return 0

    def symlinking_popen_factory(*_args: object, **_kwargs: object) -> CompletedProcess:
        output_dir.symlink_to(outside_dir, target_is_directory=True)
        return CompletedProcess()

    with pytest.raises(OSError):
        _run_idea(tmp_path, output_dir, popen_factory=cast(script_runners.PopenFactory, symlinking_popen_factory))

    assert not (outside_dir / "wrapper_output.log").exists()

    # Former: an exited process with an inherited open pipe completes promptly.
    stdout, write_fd = _piped_stdout()

    class ExitedProcess:
        args = ["bash"]
        returncode = 0
        stdout: BinaryIO

        def poll(self) -> int:
            return 0

    ExitedProcess.stdout = stdout
    assert write_fd is not None
    output_dir = tmp_path / "detached-pipe-idea"
    started_at = time.monotonic()
    try:
        result = _run_idea(
            tmp_path, output_dir, popen_factory=cast(script_runners.PopenFactory, lambda *_a, **_k: ExitedProcess())
        )
    finally:
        os.close(write_fd)

    assert time.monotonic() - started_at < 1.0
    assert result.exit_code == 0
    assert (output_dir / "wrapper_output.log").read_bytes() == b""


def test_swapped_temporary_path_cannot_replace_idea_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text("secret-must-not-leak", encoding="utf-8")

    class CompletedProcess:
        args = ["bash"]
        returncode = 0
        stdout: BinaryIO

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(script_runners, "temporary_directory", lambda: tmp_path)
    real_temporary_file = tempfile.TemporaryFile

    def visible_temporary_file(*, mode: str, prefix: str = "", dir: str | Path | None = None) -> BinaryIO:
        if not prefix.endswith("-wrapper-output-"):
            return cast(BinaryIO, real_temporary_file(mode=mode, prefix=prefix, dir=dir))
        return cast(BinaryIO, tempfile.NamedTemporaryFile(mode=mode, prefix=prefix, dir=dir, delete=True))

    def popen_factory(*_args: object, **_kwargs: object) -> CompletedProcess:
        visible_path = next(tmp_path.glob(".*-wrapper-output-*"))
        visible_path.unlink()
        visible_path.symlink_to(secret_path)
        process = CompletedProcess()
        process.stdout, _ = _piped_stdout(b"trusted child output\n")
        return process

    monkeypatch.setattr(script_runners.tempfile, "TemporaryFile", visible_temporary_file)
    output_dir = tmp_path / "swapped-temp-idea"
    result = _run_idea(tmp_path, output_dir, popen_factory=cast(script_runners.PopenFactory, popen_factory))

    published_output = (output_dir / "wrapper_output.log").read_text(encoding="utf-8")
    assert result.output == "trusted child output\n"
    assert published_output == "trusted child output\n"
    assert "secret-must-not-leak" not in result.output
    assert "secret-must-not-leak" not in published_output
    assert secret_path.read_text(encoding="utf-8") == "secret-must-not-leak"


def test_output_log_publication_rejects_symlink_source_and_content_over_hard_quota(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Former: canonical log copy rejects a symlink source.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text("secret-must-not-copy", encoding="utf-8")
    source_path = source_dir / "wrapper_output.log"
    source_path.symlink_to(secret_path)
    destination_path = tmp_path / "destination" / "idea_generation_wrapper_output.log"

    with pytest.raises(OSError):
        script_runners.copy_output_log_no_follow(source_path, destination_path)

    assert not destination_path.exists()

    # Former: output log publisher rejects content over the hard quota.
    monkeypatch.setattr(script_runners, "SCRIPT_OUTPUT_LOG_MAX_BYTES", 1024)
    destination_path = tmp_path / "destination" / "wrapper_output.log"

    with pytest.raises(OSError, match="publication quota"):
        script_runners.write_output_log_no_follow("x" * 1025, destination_path)

    assert not destination_path.exists()
