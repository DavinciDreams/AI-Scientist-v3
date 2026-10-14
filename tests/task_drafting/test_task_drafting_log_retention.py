"""Bounded agent-log retention and task.sh's rejection of symlinked drafted artifacts."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "src" / "scripts"
TASK_SCRIPT = SCRIPTS / "task.sh"
BOUNDED_LOG_SCRIPT = SCRIPTS / "lib" / "bounded_log.py"
CONTAINER_AGENT_SCRIPT = SCRIPTS / "lib" / "container_agent.sh"
SHELL_AGENT_RUNNERS = ("task.sh", "idea.sh", "run.sh", "posthoc_agent.sh", "paper.sh", "paper_edit.sh")


def _bounded_log(tmp_path: Path, agent_output: bytes, max_bytes: str, *flags: str):
    tmp_path.mkdir(exist_ok=True)
    output_path = tmp_path / "agent.log"
    result = subprocess.run(
        [sys.executable, str(BOUNDED_LOG_SCRIPT), "--output", str(output_path),
         "--max-bytes", max_bytes, *flags],
        input=agent_output,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result, output_path.read_bytes()


def test_bounded_log_preserves_stream_under_quota_and_truncates_passthrough(tmp_path: Path) -> None:
    # preserves the complete stream under quota
    agent_output = b"complete agent output\n"
    _, retained = _bounded_log(tmp_path / "under_quota", agent_output, "1024")
    assert retained == agent_output

    # passthrough retains a bounded copy
    agent_output = b"stream-start\n" + (b"x" * 8192) + b"\nstream-tail\n"
    result, retained = _bounded_log(tmp_path / "passthrough", agent_output, "2048", "--passthrough")
    assert result.stdout == agent_output
    assert len(retained) <= 2048
    assert b"output truncated after exceeding 2048 bytes" in retained
    assert retained.endswith(b"\nstream-tail\n")


def test_agent_log_quota_validation_and_shell_runners_share_bounded_writer() -> None:
    # quota validation enforces the supported range
    command = (
        f"source '{CONTAINER_AGENT_SCRIPT}'; "
        "validate_agent_log_max_bytes 1024; "
        "validate_agent_log_max_bytes 1073741824; "
        "! validate_agent_log_max_bytes 1023; "
        "! validate_agent_log_max_bytes 1073741825; "
        "! validate_agent_log_max_bytes 999999999999999999999999"
    )
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=10,
        env=os.environ | {"WRAPPER_PYTHON": sys.executable},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # shell-owned agent logs use the shared bounded writer
    for script_name in SHELL_AGENT_RUNNERS:
        script_text = (SCRIPTS / script_name).read_text(encoding="utf-8")
        assert "bounded_agent_log_writer" in script_text, script_name
        assert 'tee "$LOG_FILE"' not in script_text, script_name
        assert 'tee "$log_path"' not in script_text, script_name

    assert 'tee "$run_output_file"' not in CONTAINER_AGENT_SCRIPT.read_text(encoding="utf-8")
    concept_text = (SCRIPTS / "lib" / "concept_figure.sh").read_text(encoding="utf-8")
    assert '"$CONCEPT_FIGURE_LIB_DIR/bounded_log.py"' in concept_text
    variant_text = (SCRIPTS / "variant_agent.sh").read_text(encoding="utf-8")
    assert 'validate_agent_log_max_bytes "$AGENT_LOG_MAX_BYTES"' in variant_text


def _run_malicious_draft(tmp_path: Path, attack: str, **extra_env: str) -> subprocess.CompletedProcess:
    """Run task.sh with a fake headless agent that misbehaves only during the draft phase."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text("Draft a test task.\n", encoding="utf-8")
    fake_headless = tmp_path / "headless"
    fake_headless.write_text(
        '#!/bin/bash\nset -euo pipefail\n'
        'if [[ "${TASK_AGENT_PHASE:-}" != "draft" ]]; then\n'
        '  exec "${WRAPPER_PYTHON}" "${TASK_REPO_ROOT}/tests/fake_task_draft_agent.py"\n'
        f"fi\n{attack}\n",
        encoding="utf-8",
    )
    fake_headless.chmod(0o755)
    env = os.environ | {
        "TASK_ALLOW_NATIVE_DRAFT_FOR_TESTS": "1",
        "WRAPPER_PYTHON": sys.executable,
        "HEADLESS_BIN": str(fake_headless),
        **extra_env,
    }
    result = subprocess.run(
        ["bash", str(TASK_SCRIPT), "--agent", "pi", "--task-dir", str(task_dir),
         "--output-dir", str(tmp_path / "draft")],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    return result


@pytest.mark.parametrize(
    ("attack", "message", "leaked"),
    [
        pytest.param(
            'mkdir -p "${DRAFT_CODE_ROOT}"\n'
            'ln -s /etc/passwd "${DRAFT_CAPTURE_PATH}"\n'
            "printf 'print(1)\\n' > \"${DRAFT_EVALUATE_PATH}\"\n"
            "printf 'schema_version = 1\\ncode_root = \"src\"\\n' > \"${DRAFT_PROGRAM_SPEC_PATH}\"\n"
            "printf 'print(1)\\n' > \"${DRAFT_ENTRYPOINT_PATH}\"",
            "Unsafe drafted artifact: capture.py",
            "capture.py",
            id="required-output",
        ),
        pytest.param(
            'mkdir -p "$(dirname "${DRAFT_CAPTURE_PATH}")"\n'
            "printf 'print(1)\\n' > \"${DRAFT_CAPTURE_PATH}\"\n"
            "printf 'print(1)\\n' > \"${DRAFT_EVALUATE_PATH}\"\n"
            'ln -s "${OUTSIDE_DIR}" "${DRAFT_PROGRAM_DIR}"',
            "Unsafe drafted program directory: program_0",
            "program_spec.toml",
            id="program-directory",
        ),
        pytest.param(
            'rm -rf ./task\nln -s "${OUTSIDE_DIR}" ./task',
            "Unsafe staged task directory",
            None,
            id="staged-task-root",
        ),
    ],
)
def test_task_draft_rejects_symlinked_outputs_and_symlinks_into_outside_directories(
    tmp_path: Path, attack: str, message: str, leaked: str | None
) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    result = _run_malicious_draft(tmp_path, attack, OUTSIDE_DIR=str(outside_dir))

    assert message in result.stderr
    if leaked == "capture.py":
        # The symlinked required output is never copied into the staged task.
        assert not (tmp_path / "task" / "capture.py").exists()
    elif leaked is None:
        assert list(outside_dir.iterdir()) == []
    else:
        assert not (outside_dir / leaked).exists()
