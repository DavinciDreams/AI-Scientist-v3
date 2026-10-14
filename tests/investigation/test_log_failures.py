"""Private task/logger failure receipts, runner cleanup, and host-side failure handling."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.investigation import bounded_process, main_entry, script_runners, task_failure
from src.investigation import cli as investigation_cli
from src.investigation.config import (
    AgentConfig,
    ContainerConfig,
    ExecutionConfig,
    InvestigationConfig,
    PaperConfig,
    PathsConfig,
)
from src.investigation.evaluation_artifact_cleanup import ExperimentDirectoryGuard
from src.investigation.evaluator_outputs import EvaluatorOutputContractError
from src.investigation.harness_eval import SealedHarnessConfig, _run_sealed_evaluation

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_COMMON: dict[str, Any] = dict(
    agent="codex", timeout_seconds=1, popen_factory=subprocess.Popen,
    register_subprocess=lambda proc: None, unregister_subprocess=lambda proc: None,
    terminate_process_group=lambda *a, **kw: None,
    container_execution_env=lambda **kw: {}, ensure_text=str,
)


def _run_attempt(tmp_path: Path, **overrides: Any):
    tmp_path.mkdir(parents=True, exist_ok=True)
    idea = tmp_path / "idea.json"
    idea.write_text(json.dumps({"idea_id": "idea"}))
    arguments = dict(
        _RUNNER_COMMON, task_dir=tmp_path / "task", results_dir=tmp_path / "results",
        global_memory_dir=None, seed_program_dir=tmp_path / "seed", idea_file=idea,
        rubrics_file=tmp_path / "rubrics", attempt_id="attempt", idea_dir_name="idea",
        instruct_path=None, harness_evaluated_attempts=True, containerize=True,
        script_path=Path("src/scripts/run.sh"),
    )
    arguments.update(overrides)
    return script_runners.run_single_attempt(**arguments)


def _run_idea(tmp_path: Path):
    return script_runners.run_idea_generation(
        **_RUNNER_COMMON, task_dir=tmp_path / "task", results_dir=tmp_path / "results",
        global_memory_dir=None, cycle_index=3, worker_index=2, output_dir=tmp_path / "output",
        instruct_path=tmp_path / "instruction", script_path=Path("src/scripts/idea.sh"),
    )


def _run_task(tmp_path: Path):
    return script_runners.run_task_drafting(
        **_RUNNER_COMMON, task_dir=tmp_path / "task", output_dir=tmp_path / "output",
        instruct_path=tmp_path / "instruct.md", script_path=Path("src/scripts/task.sh"),
    )


# --------------------------------------------------------------------------- receipts


def test_private_receipt_roundtrip_empty_invalid_schema_and_cli() -> None:
    # private receipt roundtrip
    with tempfile.TemporaryFile() as receipt:
        task_failure.write_task_failure(receipt.fileno(), "attempt", "idea")
        assert task_failure.read_task_failure(receipt.fileno(), "attempt", "idea") == (
            task_failure.TASK_CONTRACT_FAILURE
        )
        with pytest.raises(task_failure.TaskFailureControlError):
            task_failure.read_task_failure(receipt.fileno(), "other", "idea")
    with tempfile.TemporaryFile() as control:
        task_failure.write_log_failure(control.fileno(), "attempt", "idea")
        assert task_failure.read_task_failure(control.fileno(), "attempt", "idea") == (
            task_failure.AGENT_LOG_FAILURE
        )

    # empty private receipt means no task failure
    with tempfile.TemporaryFile() as receipt:
        assert task_failure.read_task_failure(receipt.fileno(), "attempt", "idea") is None

    # invalid private receipt aborts as control error
    for payload in (b"null", b"{}", b"x" * 4097, b'{"schema_version":'):
        with tempfile.TemporaryFile() as receipt:
            receipt.write(payload)
            receipt.flush()
            with pytest.raises(task_failure.TaskFailureControlError):
                task_failure.read_task_failure(receipt.fileno(), "attempt", "idea")

    # receipt schema version must be exact
    for replacement in ('true', '1, "schema_version": 1'):
        with tempfile.TemporaryFile() as receipt:
            task_failure.write_task_failure(receipt.fileno(), "attempt", "idea")
            raw = os.pread(receipt.fileno(), 4096, 0).replace(
                b'"schema_version": 1', f'"schema_version": {replacement}'.encode()
            )
            receipt.seek(0)
            receipt.write(raw)
            receipt.truncate()
            receipt.flush()
            with pytest.raises(task_failure.TaskFailureControlError):
                task_failure.read_task_failure(receipt.fileno(), "attempt", "idea")

    # log failure CLI writes parent private descriptor
    with tempfile.TemporaryFile() as control:
        assert task_failure.main([
            "--log-failure-fd", str(control.fileno()), "--attempt-id", "attempt", "--idea-id", "idea",
        ]) == 0
        assert b'"failure_scope": "infrastructure"' in os.pread(control.fileno(), 4096, 0)


def test_sealed_contract_failure_preserves_manifest_and_private_receipt(tmp_path, monkeypatch):
    from src.variants_pipeline import host_eval

    program = tmp_path / "program_1"
    program.mkdir()

    def evaluate(**kwargs):
        (program / "evaluation_failure.json").write_text(json.dumps({"compute_usage": {"rpc_calls": 2}}))
        raise EvaluatorOutputContractError("password=do-not-leak")

    monkeypatch.setattr(host_eval, "evaluate_variant_program", evaluate)
    with tempfile.TemporaryFile() as receipt:
        code = _run_sealed_evaluation(
            experiment_dir=tmp_path, attempt_id="attempt", idea_id="idea",
            python_bin="python", timeout_seconds=10,
            config=SealedHarnessConfig(tmp_path, "image", None, {}),
            task_failure_fd=receipt.fileno(),
        )
        assert code == task_failure.TASK_CONTRACT_EXIT_CODE
        assert task_failure.read_task_failure(receipt.fileno(), "attempt", "idea")
    manifest = json.loads((program / "evaluation_manifest.json").read_text())
    assert manifest["failure_category"] == task_failure.TASK_CONTRACT_FAILURE
    assert manifest["failure_scope"] == "task"
    assert manifest["attempt_id"] == "attempt"
    assert (program / "evaluation_failure.json").is_file()
    assert "do-not-leak" not in (program / "harness_evaluate_stderr.log").read_text()


# --------------------------------------------------------------------------- script runners


def test_run_single_attempt_trusts_only_private_control_receipt(tmp_path, monkeypatch):
    # runner reads only private control receipt
    for trusted in (False, True):
        observed = []

        def execute(**kwargs):
            descriptor = int(kwargs["env"][task_failure.TASK_FAILURE_FD_ENV])
            assert kwargs["pass_fds"] == (descriptor,)
            observed.append(descriptor)
            if trusted:
                task_failure.write_task_failure(descriptor, "attempt", "idea")
            return '{"failure_category":"evaluator_output_contract_invalid"}', 86

        with monkeypatch.context() as m:
            m.setattr(script_runners, "_execute_script_command", execute)
            result = _run_attempt(tmp_path / f"trusted_{trusted}")
        assert result.task_failure_category == (task_failure.TASK_CONTRACT_FAILURE if trusted else None), trusted
        with pytest.raises(OSError):
            os.fstat(observed[0])

    # legacy attempt has private logging failure channel
    def execute_legacy(**kwargs):
        assert task_failure.TASK_FAILURE_FD_ENV in kwargs["env"]
        return "", 0

    with monkeypatch.context() as m:
        m.setattr(script_runners, "_execute_script_command", execute_legacy)
        assert _run_attempt(tmp_path / "legacy", containerize=False).task_failure_category is None

    # confirmed failure does not depend on public artifact access
    for public_access in ("stale_experiment_dir", "capture_error"):
        def execute_confirmed(**kwargs):
            descriptor = int(kwargs["env"][task_failure.TASK_FAILURE_FD_ENV])
            task_failure.write_task_failure(descriptor, "attempt", "idea")
            if public_access == "capture_error":
                raise OSError(errno.ESTALE, "capture finalization failed")
            return "diagnostic", 86

        def fail(*args):
            raise OSError(errno.ESTALE, "stale public directory")

        with monkeypatch.context() as m:
            m.setattr(script_runners, "_execute_script_command", execute_confirmed)
            if public_access == "stale_experiment_dir":
                m.setattr(ExperimentDirectoryGuard, "open_experiment", fail)
            result = _run_attempt(tmp_path / public_access)
        assert result.task_failure_category == task_failure.TASK_CONTRACT_FAILURE, public_access

    # control creation failure closes directory guard
    guard_root = tmp_path / "guard"
    guard_root.mkdir()
    guard = ExperimentDirectoryGuard.prepare(guard_root / "results", "idea")
    descriptors = (guard._results_fd, guard._ideas_fd)

    def fail_tempfile(**kwargs):
        raise OSError(errno.ENOSPC, "quota")

    with monkeypatch.context() as m:
        m.setattr(ExperimentDirectoryGuard, "prepare", lambda *args: guard)
        m.setattr(script_runners.tempfile, "TemporaryFile", fail_tempfile)
        with pytest.raises(OSError):
            _run_attempt(guard_root)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


_LOGGER_RUNNERS = [
    (_run_idea, "idea_generation_cycle_3_worker_2", "idea_generation", "idea"),
    (_run_task, "task_drafting", "task_drafting", "task"),
]


def test_runner_logger_failure_uses_private_channel(tmp_path, monkeypatch):
    # logger failure requires private receipt (run_single_attempt)
    for containerize in (False, True):
        for authenticated in (False, True):
            name = f"attempt_c{containerize}_a{authenticated}"

            def execute_attempt(**kwargs):
                if authenticated:
                    fd = int(kwargs["env"][task_failure.TASK_FAILURE_FD_ENV])
                    task_failure.write_log_failure(fd, "attempt", "idea")
                return '{"failure_category":"agent_log_persistence_failed"}', 4

            with monkeypatch.context() as m:
                m.setattr(script_runners, "_execute_script_command", execute_attempt)
                result = _run_attempt(tmp_path / name, containerize=containerize)
            assert result.task_failure_category is None, name
            assert result.infrastructure_failure_category == (
                task_failure.AGENT_LOG_FAILURE if authenticated else None
            ), name

    # runner logger failure uses private channel (idea generation and task drafting)
    for runner, attempt_id, idea_id, label in _LOGGER_RUNNERS:
        for capture_error in (False, True):
            for trusted in (False, True):
                name = f"{label}_capture{capture_error}_trusted{trusted}"

                def execute_runner(**kwargs):
                    if trusted:
                        descriptor = int(kwargs["env"][task_failure.TASK_FAILURE_FD_ENV])
                        assert kwargs["pass_fds"] == (descriptor,)
                        task_failure.write_log_failure(descriptor, attempt_id, idea_id)
                    if capture_error:
                        raise OSError(errno.ESTALE, "capture finalization failed")
                    return "agent_log_persistence_failed", 4

                root = tmp_path / name
                root.mkdir()
                with monkeypatch.context() as m:
                    m.setattr(script_runners, "_execute_script_command", execute_runner)
                    if capture_error and not trusted:
                        with pytest.raises(OSError):
                            runner(root)
                        continue
                    result = runner(root)
                assert result.infrastructure_failure_category == (
                    task_failure.AGENT_LOG_FAILURE if trusted else None
                ), name


# --------------------------------------------------------------------------- shell scripts


def test_agent_subshell_closes_host_control_descriptor_and_idea_receipt_survives_cleanup():
    # agent subshell closes host control descriptor (run.sh and idea.sh)
    for script in ("run.sh", "idea.sh"):
        source = (_REPO_ROOT / "src/scripts" / script).read_text()
        launch = source.index('mkfifo "$LOG_PIPE"')
        agent = source[launch:source.index("AGENT_PID=$!", launch)]
        assert "exec {AGENTIC_TASK_FAILURE_FD}>&-" in agent, script
        assert "unset AGENTIC_TASK_FAILURE_FD" in agent, script
        if script == "idea.sh":
            receipt = source.index("--log-failure-fd", launch)
            assert receipt < source.index("record_headless_cost_log", launch)
            assert receipt < source.index("cleanup_idea_security_paths", launch)

    # agent descriptor is closed but parent can still write
    with tempfile.TemporaryFile() as receipt:
        descriptor = receipt.fileno()
        script = (
            "( exec {AGENTIC_TASK_FAILURE_FD}>&-; unset AGENTIC_TASK_FAILURE_FD; "
            f"if echo forged >&{descriptor}; then exit 1; fi ); "
            f"echo trusted >&{descriptor}"
        )
        subprocess.run(
            ["bash", "-c", script], check=True, capture_output=True,
            pass_fds=(descriptor,), env={task_failure.TASK_FAILURE_FD_ENV: str(descriptor)},
        )
        assert os.pread(descriptor, 100, 0) == b"trusted\n"

    # idea logger receipt survives shell cleanup failure
    source = (_REPO_ROOT / "src/scripts/idea.sh").read_text()
    start = source.index('if [ "$INTERRUPTED" -eq 1 ] || [ "$AGENT_EXIT_CODE" -eq 130 ]; then')
    block = source[start:source.index("record_headless_cost_log", start)]
    with tempfile.TemporaryFile() as control:
        result = subprocess.run(
            ["bash", "-c", "trap 'exit 9' EXIT\n" + block],
            env={
                **os.environ, "PYTHON_BIN": sys.executable, "INTERRUPTED": "0",
                "AGENT_EXIT_CODE": "0", "TEE_EXIT_CODE": "4", "CYCLE_INDEX": "3",
                "WORKER_INDEX": "2", task_failure.TASK_FAILURE_FD_ENV: str(control.fileno()),
            },
            pass_fds=(control.fileno(),), capture_output=True,
        )
        assert result.returncode == 9
        assert task_failure.read_task_failure(
            control.fileno(), "idea_generation_cycle_3_worker_2", "idea_generation",
        ) == task_failure.AGENT_LOG_FAILURE


@pytest.mark.parametrize("failed_echo", [False, True])
def test_run_shell_reports_logger_failure_before_cleanup(failed_echo):
    script = (_REPO_ROOT / "src/scripts/run.sh").read_text()
    block = 'if [ "$TEE_EXIT_CODE" -ne 0 ]; then' + script.split(
        'if [ "$TEE_EXIT_CODE" -ne 0 ]; then', 1,
    )[1].split("\nrecord_headless_cost_log", 1)[0]
    for interrupted in (0, 1):
        for agent_exit in (17, 130):
            name = f"interrupted={interrupted} agent_exit={agent_exit}"
            with tempfile.TemporaryFile() as control:
                env = {
                    **os.environ, "PYTHON_BIN": sys.executable, "TEE_EXIT_CODE": "4",
                    "AGENT_EXIT_CODE": str(agent_exit), "INTERRUPTED": str(interrupted),
                    "ATTEMPT_ID": "attempt", "IDEA_ID": "idea",
                    task_failure.TASK_FAILURE_FD_ENV: str(control.fileno()),
                }
                result = subprocess.run(
                    ["bash", "-c", "set -euo pipefail\n" + ("echo() { return 1; }\n" if failed_echo else "") + block],
                    env=env, pass_fds=(control.fileno(),), capture_output=True, text=True, timeout=5,
                )
                assert result.returncode == 0, (name, result.stderr)
                if not failed_echo:
                    assert f"agent_exit={agent_exit} logger_exit=4" in result.stderr, name
                assert task_failure.read_task_failure(control.fileno(), "attempt", "idea") == (
                    None if interrupted or agent_exit == 130 else task_failure.AGENT_LOG_FAILURE
                ), name


# --------------------------------------------------------------------------- run_single_attempt cleanup


def _cli_attempt(*, results_dir: Path, idea_dir_name: str):
    return investigation_cli.run_single_attempt(
        agent="codex", task_dir=results_dir.parent / "task", results_dir=results_dir,
        seed_program_dir=results_dir.parent / "task" / "program_0",
        idea_file=results_dir.parent / "idea.json", rubrics_file=results_dir.parent / "rubrics.json",
        attempt_id="idea_0001_attempt_01", idea_dir_name=idea_dir_name,
        instruct_path=None, timeout_seconds=30,
    )


def _fake_proc(stdout: str) -> Mock:
    proc = Mock()
    proc.communicate.return_value = (stdout, "")
    proc.returncode = 0
    return proc


def test_run_single_attempt_cleans_unmanaged_evaluator_runs_and_ignores_forged_marker(tmp_path: Path) -> None:
    # cleans unmanaged evaluator runs
    results_dir = tmp_path / "a" / "results"
    experiment_dir = results_dir / "ideas" / "01_cleanup-smoke"
    run_dir = experiment_dir / ".dev_eval_program_1" / "runs" / "run_000_case"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
    proc = _fake_proc(f"Experiment directory: {experiment_dir}\n")
    with patch("src.investigation.cli.subprocess.Popen", return_value=proc):
        result = _cli_attempt(results_dir=results_dir, idea_dir_name="01_cleanup-smoke")
    assert not (experiment_dir / ".dev_eval_program_1" / "runs").exists()
    assert "unmanaged_evaluation_outputs_removed" in result.output

    # ignores forged experiment directory marker
    results_dir = tmp_path / "b" / "results"
    expected_dir = results_dir / "ideas" / "01_trusted"
    expected_dir.mkdir(parents=True)
    with patch("src.investigation.cli.subprocess.Popen", return_value=_fake_proc("Experiment directory: /tmp/forged\n")):
        result = _cli_attempt(results_dir=results_dir, idea_dir_name="01_trusted")
    assert result.experiment_dir == expected_dir.absolute()


def test_run_single_attempt_rejects_traversal_symlink_and_swapped_results_ancestor(tmp_path: Path) -> None:
    # rejects traversing idea directory name
    with (
        patch("src.investigation.cli.subprocess.Popen") as popen_mock,
        pytest.raises(ValueError, match="idea directory name"),
    ):
        _cli_attempt(results_dir=tmp_path / "a" / "results", idea_dir_name="../outside")
    popen_mock.assert_not_called()

    # rejects symlinked experiment directory
    root = tmp_path / "b"
    results_dir = root / "results"
    ideas_dir = results_dir / "ideas"
    ideas_dir.mkdir(parents=True)
    external_dir = root / "external"
    external_dir.mkdir()
    expected_dir = ideas_dir / "01_symlink"
    expected_dir.symlink_to(external_dir, target_is_directory=True)
    with (
        patch("src.investigation.cli.subprocess.Popen") as popen_mock,
        pytest.raises(NotADirectoryError, match="experiment directory"),
    ):
        _cli_attempt(results_dir=results_dir, idea_dir_name="01_symlink")
    popen_mock.assert_not_called()
    assert expected_dir.is_symlink()
    assert external_dir.is_dir()

    # rejects results ancestor swapped during agent
    root = tmp_path / "c"
    results_dir = root / "results"
    (results_dir / "ideas" / "01_swap").mkdir(parents=True)
    moved_results = root / "moved_results"
    external_results = root / "external_results"
    external_run = external_results / "ideas" / "01_swap" / ".dev_eval_program_1" / "runs" / "run_000_case"
    external_run.mkdir(parents=True)
    (external_run / "config.json").write_text("external", encoding="utf-8")
    (external_run / "run_summary.json").write_text("external", encoding="utf-8")
    fake_proc = Mock()
    fake_proc.returncode = 0

    def swap_results(*, timeout):
        results_dir.rename(moved_results)
        results_dir.symlink_to(external_results, target_is_directory=True)
        return "finished\n", ""

    fake_proc.communicate.side_effect = swap_results
    with (
        patch("src.investigation.cli.subprocess.Popen", return_value=fake_proc),
        pytest.raises(RuntimeError, match="changed during agent execution"),
    ):
        _cli_attempt(results_dir=results_dir, idea_dir_name="01_swap")
    assert (external_run / "config.json").read_text(encoding="utf-8") == "external"


# --------------------------------------------------------------------------- process cleanup witness


def test_interruption_witness_requires_reaped_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for teardown_fails in (False, True):
        children: list[subprocess.Popen[bytes]] = []
        real_popen = subprocess.Popen
        witness = bounded_process.ProcessCleanupWitness()
        interruption = KeyboardInterrupt()

        def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        def interrupt(*args: object) -> None:
            raise interruption

        with monkeypatch.context() as m:
            m.setattr(subprocess, "Popen", popen)
            m.setattr(bounded_process.os, "set_blocking", interrupt)
            if teardown_fails:
                m.setattr(bounded_process, "_stop_and_reap", lambda *a: None)
            try:
                with pytest.raises(KeyboardInterrupt) as raised:
                    bounded_process.run_bounded_process(
                        [sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path,
                        env=os.environ, timeout_seconds=5, cleanup_witness=witness,
                    )
                assert raised.value is interruption, teardown_fails
                assert witness.reaped is not teardown_fails, teardown_fails
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=2)


# --------------------------------------------------------------------------- drafting logger failure


_DRAFT_CASES = [
    (task_failure.AGENT_LOG_FAILURE, 4, 87), (None, 86, 2), (None, 87, 2),
]


def test_draft_logger_failure_stops_before_host_log_or_ideas(tmp_path, monkeypatch):
    for stale_host_write in (False, True):
        for category, raw_exit, expected_exit in _DRAFT_CASES:
            name = f"stale={stale_host_write} category={category} raw_exit={raw_exit}"
            root = tmp_path / f"stale{stale_host_write}_exit{raw_exit}"
            task_dir = root / "task"
            task_dir.mkdir(parents=True)
            (task_dir / "task.md").write_text(
                "# Goal\n- optimize objective\n\n# Rubrics\n- use restarts\n", encoding="utf-8"
            )
            (task_dir / "goal.md").write_text("legacy", encoding="utf-8")
            instruct = root / "instruct.md"
            instruct.write_text("template", encoding="utf-8")
            cfg = InvestigationConfig(
                agent=AgentConfig(name="codex"),
                paths=PathsConfig(task_dir=str(task_dir), results_dir=str(root / "results"), instruct=str(instruct)),
                execution=ExecutionConfig(
                    max_idea_cycles=2, run_timeout_seconds=5, run_timeout_grace_seconds=1,
                    max_program_evals_per_attempt=4, workers=1, resume=False, draft_task_if_missing=True,
                ),
                paper=PaperConfig(enabled=False),
                container=ContainerConfig(),
            )
            host_write = Mock(
                side_effect=(
                    OSError(errno.ESTALE, "shared output unavailable")
                    if stale_host_write and category is not None else None
                ),
            )

            with monkeypatch.context() as m:
                def failed_draft(**kwargs):
                    m.setattr(main_entry.state_store, "write_private_text", host_write)
                    return investigation_cli.TaskDraftResult(
                        output="logger persistence failed", exit_code=raw_exit,
                        draft_dir=kwargs["output_dir"], infrastructure_failure_category=category,
                    )

                with (
                    patch("src.investigation.cli.load_config", return_value=cfg),
                    patch("src.investigation.cli.run_task_drafting", side_effect=failed_draft) as draft,
                    patch("src.investigation.cli.run_idea_generation") as ideas,
                    patch("src.investigation.cli.run_single_attempt") as attempts,
                ):
                    assert investigation_cli.main() == expected_exit, name
            draft.assert_called_once()
            if category is not None:
                host_write.assert_not_called()
            else:
                host_write.assert_called_once()
            ideas.assert_not_called()
            attempts.assert_not_called()
