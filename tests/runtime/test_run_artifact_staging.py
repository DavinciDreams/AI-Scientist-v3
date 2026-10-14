from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.program_testutils import make_program_dir
from tests.e2e.test_agent_execution_pipeline import (
    _script_path,
    _write_fake_agent,
    _write_idea_and_rubrics,
)
from tests.task_drafting.test_task_contract import _write_scaffold


@pytest.mark.parametrize("has_manual_report", [False, True])
def test_contracted_run_stages_only_search_evaluator_helpers(
    tmp_path: Path, has_manual_report: bool
) -> None:
    task_dir = tmp_path / "task"
    _write_scaffold(task_dir)
    make_program_dir(task_dir / "program_0", content="def solve():\n    return 1\n")
    (task_dir / "capture.py").write_text("# capture stub\n", encoding="utf-8")
    report = task_dir / "manual_validation.json"
    if has_manual_report:
        report.write_text('{"final": "private validation sentinel"}\n')
    (task_dir / "evaluate.py").write_text(
        "from shared_search import SEARCH_VALUE\n",
        encoding="utf-8",
    )
    (task_dir / "evaluate_final.py").write_text(
        "from final_secret_helper import FINAL_VALUE\n",
        encoding="utf-8",
    )
    shared_search = task_dir / "shared_search.py"
    shared_search.write_text(
        "from parent_search import SEARCH_VALUE\n",
        encoding="utf-8",
    )
    parent_search = task_dir / "parent_search.py"
    parent_search.write_text("SEARCH_VALUE = 1\n", encoding="utf-8")
    (task_dir / "final_secret_helper.py").write_text(
        "from parent_final_secret import FINAL_VALUE\n",
        encoding="utf-8",
    )
    (tmp_path / "parent_final_secret.py").write_text(
        "FINAL_VALUE = 'sealed sentinel'\n",
        encoding="utf-8",
    )
    contract_path = task_dir / "task_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for helper in (shared_search, parent_search):
        contract["provenance"]["artifacts"].append(
            {
                "role": "search_helper",
                "path": helper.name,
                "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
                "source": "test search evaluator helper",
            }
        )
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    staged_task_dir = results_dir / "task"
    shutil.copytree(task_dir, staged_task_dir)
    (staged_task_dir / "evaluate.py").write_text(
        "from final_secret_helper import FINAL_VALUE\n",
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_agent(bin_dir, "headless")
    fake_docker = bin_dir / "docker"
    docker_log = tmp_path / "docker.log"
    fake_docker.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\0" "$@" >> "{docker_log}"\n'
        'if [ "$1" = "info" ]; then exit 0; fi\n'
        'if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then\n'
        '  printf "sha256:%064d\\n" 0\n'
        "  exit 0\n"
        "fi\n"
        "exit 42\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    env_file = tmp_path / "container.env"
    env_file.write_text("", encoding="utf-8")
    idea_file, rubrics_file = _write_idea_and_rubrics(tmp_path)

    env = os.environ | {
        "AGENT_DOCKER_READY_RETRIES": "1",
        "AGENT_DOCKER_RUN_RETRIES": "1",
        "HEADLESS_BIN": str(bin_dir / "headless"),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WRAPPER_PYTHON": sys.executable,
    }
    run = subprocess.run(
        [
            "bash",
            str(_script_path("run.sh")),
            "--agent",
            "codex",
            "--task-dir",
            str(task_dir),
            "--results-dir",
            str(results_dir),
            "--idea-file",
            str(idea_file),
            "--idea-dir-name",
            "artifact_staging",
            "--rubrics-file",
            str(rubrics_file),
            "--attempt-id",
            "idea_0001_attempt_01",
            "--containerize",
            "--container-auth-root",
            str(auth_dir),
            "--container-env-file",
            str(env_file),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )

    assert run.returncode != 0, run.stdout + run.stderr
    experiment_dir = results_dir / "ideas" / "artifact_staging"
    assert (experiment_dir / "shared_search.py").is_file()
    assert (experiment_dir / "parent_search.py").is_file()
    assert not (experiment_dir / "evaluate_final.py").exists()
    assert not (experiment_dir / "final_secret_helper.py").exists()
    assert not (experiment_dir / "parent_final_secret.py").exists()
    assert not (staged_task_dir / "evaluate_final.py").exists()
    assert not (staged_task_dir / "final_secret_helper.py").exists()
    assert not (staged_task_dir / "resources" / "final").exists()
    assert not (staged_task_dir / "manual_validation.json").exists()
    assert not (experiment_dir / "manual_validation.json").exists()
    if has_manual_report:
        assert "private validation sentinel" in report.read_text()
    docker_arguments = docker_log.read_bytes().split(b"\0")
    canonical_mount = f"{task_dir}:/task-source:ro".encode()
    assert canonical_mount not in docker_arguments
    assert f"{results_dir}:/workspace".encode() in docker_arguments
    search_source_mounts = [
        argument
        for argument in docker_arguments
        if argument.startswith(b"/tmp/agentic-task-search-")
        and argument.endswith(b":/task-source:ro")
    ]
    assert len(search_source_mounts) == 1
    assert b"TASK_SOURCE_DIR=/task-source" in docker_arguments
    search_source_path = Path(
        search_source_mounts[0].decode().removesuffix(":/task-source:ro")
    )
    assert not search_source_path.exists()
    assert not any(argument.startswith(b"/dev/null:/task-source/") for argument in docker_arguments)
