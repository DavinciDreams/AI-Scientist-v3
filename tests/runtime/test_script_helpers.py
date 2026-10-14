"""Agent wrapper shell scripts and their Python support modules (skills, publication guard)."""

from __future__ import annotations

import json
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import src.investigation.safe_staging as safe_staging
from src.investigation.content_publication import ContentPublicationGuard
from src.investigation.safe_staging import atomic_copy_tree, atomic_write_bytes
from src.investigation.skill_scope import (
    SKILLS_BY_STEP,
    paper_writer_skill,
    parse_skill_csv,
    resolve_skills_for_step,
    skills_for_step,
    stage_skill_tree,
    validate_skill_set,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "src" / "scripts"
PYTHON_HELPER = SCRIPTS / "lib" / "python_bin.sh"
CONTAINER_HELPER = SCRIPTS / "lib" / "container_agent.sh"
TASK_PATHING_HELPER = SCRIPTS / "lib" / "task_pathing.sh"
CONCEPT_FIGURE_HELPER = SCRIPTS / "lib" / "concept_figure.sh"
AGENT_SCRIPTS = (
    "idea.sh", "paper.sh", "paper_edit.sh", "posthoc_agent.sh", "run.sh", "task.sh",
    "variant_agent.sh",
)


def _bash(script: str, env: dict[str, str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-c", script], check=False, capture_output=True, text=True,
        env={**os.environ, **env}, **kwargs,
    )


def _executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# task_pathing.sh
# ---------------------------------------------------------------------------


def _managed_task_python(task_dir: Path) -> subprocess.CompletedProcess[str]:
    return _bash(
        'source "$TASK_PATHING_HELPER"; require_managed_task_python "$TASK_DIR"',
        {"TASK_PATHING_HELPER": str(TASK_PATHING_HELPER), "TASK_DIR": str(task_dir)},
    )


def test_managed_task_python_returns_canonical_executable_and_rejects_unsafe_runtimes(
    tmp_path: Path,
) -> None:
    # a canonical regular executable is returned
    task_python = _executable(tmp_path / "ok/task/.venv/bin/python", "#!/bin/sh\n")
    result = _managed_task_python(tmp_path / "ok/task")
    assert result.returncode == 0
    assert result.stdout.strip() == str(task_python)

    # missing, directory, non-executable, and symlinked runtimes are rejected
    for kind in ("missing", "directory", "non_executable", "symlink"):
        task_dir = tmp_path / kind / "task"
        task_python = task_dir / ".venv/bin/python"
        task_python.parent.mkdir(parents=True)
        if kind == "directory":
            task_python.mkdir()
        elif kind == "non_executable":
            task_python.write_text("#!/bin/sh\n", encoding="utf-8")
        elif kind == "symlink":
            task_python.symlink_to(_executable(tmp_path / kind / "python", "#!/bin/sh\n"))
        result = _managed_task_python(task_dir)
        assert result.returncode != 0, kind
        assert "managed task runtime Python must be a regular executable" in result.stderr, kind


# ---------------------------------------------------------------------------
# python_bin.sh resolver
# ---------------------------------------------------------------------------


def _resolve_python(
    tmp_path: Path, *, repo_root: Path | None = None, wrapper_python: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = {
        "PATH": str(tmp_path / "bin"),
        "REPO_ROOT": str(repo_root or tmp_path / "repo"),
        "PYTHON_HELPER": str(PYTHON_HELPER),
    }
    if wrapper_python is not None:
        environment["WRAPPER_PYTHON"] = wrapper_python
    return _bash('source "$PYTHON_HELPER"; resolve_python_bin', environment)


def test_resolver_honors_explicit_and_relative_wrapper_python(tmp_path: Path) -> None:
    # an explicit WRAPPER_PYTHON is honoured and executable
    explicit_python = _executable(tmp_path / "python dir/python")
    result = _resolve_python(tmp_path, wrapper_python=str(explicit_python))
    assert result.returncode == 0
    assert result.stdout.strip() == str(explicit_python)
    executed = _bash(
        'python_bin="$(source "$PYTHON_HELPER"; resolve_python_bin)"; "$python_bin"',
        {"PATH": str(tmp_path / "bin"), "PYTHON_HELPER": str(PYTHON_HELPER),
         "REPO_ROOT": str(tmp_path / "repo"), "WRAPPER_PYTHON": str(explicit_python)},
    )
    assert executed.returncode == 0

    # a relative WRAPPER_PYTHON is canonicalised
    relative_python = _executable(tmp_path / "relative/python")
    result = _bash(
        'cd "$WORK_DIR"; source "$PYTHON_HELPER"; resolve_python_bin',
        {"PATH": str(tmp_path / "bin"), "PYTHON_HELPER": str(PYTHON_HELPER),
         "REPO_ROOT": str(tmp_path / "repo"), "WORK_DIR": str(tmp_path),
         "WRAPPER_PYTHON": "relative/python"},
    )
    assert result.returncode == 0
    assert result.stdout.strip() == str(relative_python)


def test_resolver_prefers_virtualenv_then_python3_then_python_and_fails_clearly(
    tmp_path: Path,
) -> None:
    # without any interpreter the failure is clear
    result = _resolve_python(tmp_path)
    assert result.returncode == 127
    assert "Python interpreter not found" in result.stderr

    # python3 is preferred over python on PATH
    python3 = _executable(tmp_path / "bin/python3")
    _executable(tmp_path / "bin/python")
    python3_result = _resolve_python(tmp_path)
    python3.unlink()
    python_result = _resolve_python(tmp_path)
    assert python3_result.stdout.strip() == str(python3)
    assert python_result.stdout.strip() == str(tmp_path / "bin/python")

    # the repository virtualenv is preferred over PATH
    repo_root = tmp_path / "repo"
    venv_python = _executable(repo_root / ".venv/bin/python")
    _executable(tmp_path / "bin/python3")
    result = _resolve_python(tmp_path, repo_root=repo_root)
    assert result.returncode == 0
    assert result.stdout.strip() == str(venv_python)


def test_resolver_rejects_invalid_explicit_interpreters(tmp_path: Path) -> None:
    for kind in ("missing", "directory", "non_executable"):
        invalid_python = tmp_path / f"{kind}-python"
        if kind == "directory":
            invalid_python.mkdir()
        elif kind == "non_executable":
            invalid_python.write_text("#!/bin/sh\n", encoding="utf-8")
        result = _resolve_python(tmp_path, wrapper_python=str(invalid_python))
        assert result.returncode == 127, kind
        assert "Python interpreter is not an executable file" in result.stderr, kind


def test_wrapper_scripts_share_resolver_and_keep_repository_python_in_child_shells(
    tmp_path: Path,
) -> None:
    # all wrapper scripts use the shared Python resolver
    for script_name in AGENT_SCRIPTS:
        script = (SCRIPTS / script_name).read_text(encoding="utf-8")
        assert 'source "$SCRIPT_DIR/lib/container_agent.sh"' in script, script_name
        assert 'PYTHON_BIN="${WRAPPER_PYTHON:-python}"' not in script, script_name
        assert "$($PYTHON_BIN " not in script, script_name
    container_helper = CONTAINER_HELPER.read_text(encoding="utf-8")
    assert 'source "$CONTAINER_AGENT_LIB_DIR/python_bin.sh"' in container_helper
    assert 'PYTHON_BIN="$(resolve_python_bin)"' in container_helper
    assert 'AGENT_PYTHON_BIN="${AGENT_PYTHON_BIN:-$PYTHON_BIN}"' in container_helper

    # source order and child shells keep the repository Python
    repo_root = tmp_path / "repo"
    venv_python = _executable(repo_root / ".venv/bin/python")
    environment = os.environ.copy()
    for name in ("WRAPPER_PYTHON", "AGENT_PYTHON_BIN", "REPO_ROOT"):
        environment.pop(name, None)
    environment.update({
        "CONTAINER_HELPER": str(CONTAINER_HELPER),
        "TASK_PATHING_HELPER": str(TASK_PATHING_HELPER),
        "TEST_REPO_ROOT": str(repo_root),
    })
    result = subprocess.run(
        ["/bin/bash", "-c", """
REPO_ROOT="$TEST_REPO_ROOT"
source "$CONTAINER_HELPER"
source "$TASK_PATHING_HELPER"
printf "%s\\n" "$PYTHON_BIN" "$AGENT_PYTHON_BIN" "$WRAPPER_PYTHON"
/bin/bash -c 'source "$CONTAINER_HELPER"; printf "%s\\n" "$PYTHON_BIN"'
"""],
        check=False, capture_output=True, text=True, env=environment,
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == [str(venv_python)] * 4


# ---------------------------------------------------------------------------
# --reasoning-effort handling and concept-figure command construction
# ---------------------------------------------------------------------------


def _run_script(script_name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPTS / script_name), *arguments],
        cwd=REPO_ROOT, check=False, capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize("script_name", AGENT_SCRIPTS)
def test_agent_script_validates_exports_and_logs_reasoning_effort(script_name: str) -> None:
    # an invalid reasoning effort is rejected
    completed = _run_script(script_name, "--reasoning-effort", "max")
    assert completed.returncode == 1
    assert (
        "--reasoning-effort must be one of: low, medium, high, xhigh (got: max)"
        in completed.stderr
    )

    # reasoning effort is not supported for gemini
    completed = _run_script(script_name, "--agent", "gemini", "--reasoning-effort", "high")
    assert completed.returncode == 1
    assert "--reasoning-effort is not supported for gemini" in completed.stderr

    # the override is exported and logged
    script = (SCRIPTS / script_name).read_text(encoding="utf-8")
    assert 'AGENT_REASONING_EFFORT=""' in script
    assert "${AGENT_REASONING_EFFORT:-" not in script
    assert "export AGENT_REASONING_EFFORT" in script
    assert "Agent reasoning effort override: $AGENT_REASONING_EFFORT" in script


def _build_concept_figure_command(
    reasoning_effort: str, *, agent: str = "codex", model: str = "", profile: str = "",
) -> str:
    command = f"""
source '{CONCEPT_FIGURE_HELPER}'
AGENT='{agent}'
AGENT_MODEL='{model}'
AGENT_PROFILE='{profile}'
AGENT_REASONING_EFFORT='{reasoning_effort}'
CONCEPT_FIGURE_ENV_COMMANDS=':'
CONCEPT_FIGURE_BACKEND_FLAGS='--skip-enhance'
CONCEPT_FIGURE_MODE_FLAGS=''
CONCEPT_FIGURE_KILL_GRACE_SECONDS=60
CONCEPT_FIGURE_TIMEOUT_SECONDS=600
build_concept_figure_command figures/concept.png after-title instructions.md
printf '%s' "$CONCEPT_FIGURE_COMMAND"
"""
    completed = subprocess.run(
        ["bash", "-c", command], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    return completed.stdout


def test_concept_figure_forwards_effort_model_profile_and_uses_deterministic_workdir() -> None:
    # reasoning effort, model, and profile are forwarded to autoillustrate
    assert "--effort high" in _build_concept_figure_command("high")
    command = _build_concept_figure_command("high", model="gpt-5.5-codex", profile="research")
    assert "--model gpt-5.5-codex" in command
    assert "--profile research" in command

    # an unspecified reasoning effort is omitted and the workdir is deterministic
    command = _build_concept_figure_command("")
    assert "--effort" not in command
    assert "--workdir ./.autoillustrate/concept" in command


def test_concept_figure_leaves_claude_billing_to_headless_routes_cost_and_rejects_unvalidated_effort() -> None:
    # Claude billing selection is left to headless
    assert "HEADLESS_CLAUDE_AUTH=api" not in _build_concept_figure_command("", agent="claude")

    # standalone paper routes autoillustrate cost to the terminal
    paper = (SCRIPTS / "paper.sh").read_text(encoding="utf-8")
    progress = (SCRIPTS / "lib" / "paper_progress.sh").read_text(encoding="utf-8")
    helper = CONCEPT_FIGURE_HELPER.read_text(encoding="utf-8")
    assert 'source "$SCRIPT_DIR/lib/paper_progress.sh"' in paper
    assert "\npaper_progress_initialize\n" in paper
    assert "export AGENTIC_COST_OUTPUT_FD=3" in progress
    assert "print_args+=(--print-summary)" in helper

    # an unvalidated reasoning effort is rejected
    command = f"""
source '{CONCEPT_FIGURE_HELPER}'
AGENT=codex
AGENT_REASONING_EFFORT='high; echo injected'
build_concept_figure_command figures/concept.png after-title instructions.md
"""
    completed = subprocess.run(
        ["bash", "-c", command], cwd=REPO_ROOT, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert "invalid concept-figure reasoning effort" in completed.stderr


# ---------------------------------------------------------------------------
# variant_agent.sh workspace safety (no Docker or live agents)
# ---------------------------------------------------------------------------

_ENV_FILE_CAPTURING_DOCKER = (
    "#!/usr/bin/env bash\n"
    'while [ "$#" -gt 0 ]; do\n'
    '  if [ "$1" = "--env-file" ]; then cp "$2" "$CAPTURED_ENV"; fi\n'
    "  shift\n"
    "done\n"
    "exit 0\n"
)
_CALL_LOGGING_DOCKER = '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$DOCKER_CALLS"\nexit 0\n'


def _work_dir(tmp_path: Path, name: str = "variant") -> Path:
    path = tmp_path / name / "task_bundle"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    return _work_dir(tmp_path)


def _fake_docker(tmp_path: Path, body: str) -> None:
    _executable(tmp_path / "bin" / "docker", body)


def _run_variant_agent(
    tmp_path: Path,
    work_dir: Path,
    *,
    agent: str,
    stage_skills: bool = False,
    containerize: bool = False,
    container_env_file: Path | None = None,
    container_auth_root: Path | None = None,
    agent_model: str | None = None,
    posthoc_env_policy: bool = False,
    extra_environment: dict[str, str] | None = None,
    agent_script: str = "exit 0",
    interrupt_after: float | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    executable = _executable(
        bin_dir / "headless", f"#!/usr/bin/env bash\nset -euo pipefail\n{agent_script}\n"
    )
    rubrics = tmp_path / "rubrics.json"
    rubrics.write_text("[]\n", encoding="utf-8")
    command = [
        "bash", str(SCRIPTS / "variant_agent.sh"), "--agent", agent,
        "--work-dir", str(work_dir), "--prompt", "stop", "--rubrics-file", str(rubrics),
    ]
    if stage_skills:
        command.extend(("--stage-skills", "--skill-step", "variant_edit"))
    if containerize:
        command.append("--containerize")
    if container_env_file is not None:
        command.extend(("--container-env-file", str(container_env_file)))
    if container_auth_root is not None:
        command.extend(("--container-auth-root", str(container_auth_root)))
    if agent_model is not None:
        command.extend(("--agent-model", agent_model))
    if posthoc_env_policy:
        command.append("--posthoc-env-policy")
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["HEADLESS_BIN"] = str(executable)
    environment["WRAPPER_PYTHON"] = sys.executable
    environment.update(extra_environment or {})
    if interrupt_after is None:
        return subprocess.run(command, capture_output=True, text=True, timeout=20, env=environment)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
    )
    time.sleep(interrupt_after)
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=20)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def test_staging_does_not_follow_workspace_symlinks_and_manages_failure_logs(
    tmp_path: Path,
) -> None:
    # staging does not follow workspace symlinks for active rubrics or idea
    for name, protected_content, staged_content in [
        ("active_rubrics.json", "protected\n", "[]\n"),
        ("active_idea.json", '{"secret": true}\n', "{}\n"),
    ]:
        work_dir = _work_dir(tmp_path, name)
        protected_file = tmp_path / f"protected-{name}"
        protected_file.write_text(protected_content, encoding="utf-8")
        (work_dir / name).symlink_to(protected_file)
        result = _run_variant_agent(tmp_path, work_dir, agent="claude")
        assert result.returncode == 0, result.stdout + result.stderr
        assert protected_file.read_text(encoding="utf-8") == protected_content, name
        assert not (work_dir / name).is_symlink(), name
        assert (work_dir / name).read_text(encoding="utf-8") == staged_content, name
        assert not list(work_dir.glob("variant_agent_*.log")), name

    # a failed variant retains a bounded, private agent log
    work_dir = _work_dir(tmp_path, "failed")
    result = _run_variant_agent(
        tmp_path, work_dir, agent="claude",
        agent_script=(
            "printf '%s\\n' '{\"type\":\"error\",\"message\":\"retrofit failed\"}'\nexit 1"
        ),
    )
    assert result.returncode == 1
    retained = work_dir / "variant_agent_failure.jsonl"
    assert retained.read_text(encoding="utf-8") == '{"type":"error","message":"retrofit failed"}\n'
    assert retained.stat().st_mode & 0o777 == 0o600

    # a successful variant removes a stale failure log
    work_dir = _work_dir(tmp_path, "stale")
    (work_dir / "variant_agent_failure.jsonl").write_text("stale\n", encoding="utf-8")
    result = _run_variant_agent(tmp_path, work_dir, agent="claude")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (work_dir / "variant_agent_failure.jsonl").exists()


def test_interrupted_variant_records_partial_headless_cost(tmp_path: Path, work_dir: Path) -> None:
    cost_root = tmp_path / "cost-root"
    result = _run_variant_agent(
        tmp_path, work_dir, agent="claude",
        extra_environment={"AGENTIC_COST_ROOT": str(cost_root)},
        agent_script=(
            "printf '%s\\n' '{\"type\":\"result\",\"total_cost_usd\":0.33}'\nsleep 30"
        ),
        interrupt_after=2.0,
    )

    assert result.returncode == 130
    summary = json.loads((cost_root / "metrics" / "cost_summary.json").read_text(encoding="utf-8"))
    assert summary["reported_cost_usd"] == "0.33"
    assert not list(work_dir.glob("variant_agent_*.log"))


def test_variant_container_env_file_is_filtered(tmp_path: Path) -> None:
    _fake_docker(tmp_path, _ENV_FILE_CAPTURING_DOCKER)
    for name, agent, agent_model, posthoc, source_env, expected in [
        (
            "only_agent_credentials", "claude", None, False,
            "OPENROUTER_API_KEY=safe-key\nDEEPSEEK_API_KEY=deepseek-key\n"
            "AWS_BEARER_TOKEN_BEDROCK=bedrock-token\nHOSTILE_TOKEN=hostile-token\n"
            "DOCKER_HOST=tcp://127.0.0.1:2375\nAGENT_CONTAINER_EXTRA_MOUNTS=/:/host:rw\n",
            "OPENROUTER_API_KEY=safe-key\nDEEPSEEK_API_KEY=deepseek-key\n"
            "AWS_BEARER_TOKEN_BEDROCK=bedrock-token\n",
        ),
        (
            "posthoc_selected_provider_only", "pi", "openrouter/deepseek/deepseek-v4-flash", True,
            "OPENROUTER_API_KEY=openrouter-key\nDEEPSEEK_API_KEY=deepseek-key\n"
            "S2_API_KEY=paper-only-key\nOPENAI_API_KEY=openai-key\n",
            "OPENROUTER_API_KEY=openrouter-key\n",
        ),
        (
            "posthoc_provider_from_env_file", "pi", None, True,
            "PI_CODING_AGENT_PROVIDER=anthropic\n"
            'PI_CODING_AGENT_MODELS={"providers":["openrouter"]}\n'
            "ANTHROPIC_API_KEY=anthropic-key\nOPENAI_API_KEY=wrong-provider-key\n",
            "PI_CODING_AGENT_PROVIDER=anthropic\n"
            'PI_CODING_AGENT_MODELS={"providers":["openrouter"]}\n'
            "ANTHROPIC_API_KEY=anthropic-key\n",
        ),
    ]:
        work_dir = _work_dir(tmp_path, name)
        source_env_file = tmp_path / f"{name}-source.env"
        source_env_file.write_text(source_env, encoding="utf-8")
        captured_env = tmp_path / f"{name}-captured.env"
        result = _run_variant_agent(
            tmp_path, work_dir, agent=agent, agent_model=agent_model, posthoc_env_policy=posthoc,
            containerize=True, container_env_file=source_env_file,
            extra_environment={"CAPTURED_ENV": str(captured_env)},
        )
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr
        assert captured_env.read_text(encoding="utf-8") == expected, name


def test_posthoc_variant_env_policy_restricts_native_agent_environment(tmp_path: Path) -> None:
    # provider-selected credentials only
    for name, agent_model, environment, agent_script in [
        (
            "bedrock_provider", None,
            {"PI_CODING_AGENT_PROVIDER": "amazon-bedrock",
             "PI_CODING_AGENT_MODEL": "anthropic.claude-3-sonnet",
             "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
             "OPENAI_API_KEY": "wrong-provider-key"},
            'test "${AWS_BEARER_TOKEN_BEDROCK:-}" = bedrock-token\ntest -z "${OPENAI_API_KEY:-}"',
        ),
        (
            "bedrock_model", "amazon-bedrock/anthropic.claude-3-sonnet",
            {"AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
             "ANTHROPIC_API_KEY": "wrong-provider-key"},
            'test "${AWS_BEARER_TOKEN_BEDROCK:-}" = bedrock-token\n'
            'test -z "${ANTHROPIC_API_KEY:-}"',
        ),
        (
            "native_openrouter", "openrouter/deepseek/deepseek-v4-flash",
            {"OPENROUTER_API_KEY": "openrouter-key", "S2_API_KEY": "paper-only-key"},
            'test "${OPENROUTER_API_KEY:-}" = openrouter-key\ntest -z "${S2_API_KEY:-}"',
        ),
    ]:
        result = _run_variant_agent(
            tmp_path, _work_dir(tmp_path, name), agent="pi", agent_model=agent_model,
            posthoc_env_policy=True, extra_environment=environment, agent_script=agent_script,
        )
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr

    # the policy supports the default model
    result = _run_variant_agent(
        tmp_path, _work_dir(tmp_path, "default"), agent="codex", posthoc_env_policy=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_host_staging_preserves_protected_and_unmanaged_workspace_content(tmp_path: Path) -> None:
    # skill staging does not follow a workspace ancestor symlink
    work_dir = _work_dir(tmp_path, "skills")
    protected_config = tmp_path / "protected-codex"
    sentinel = protected_config / "skills" / "sentinel.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("protected\n", encoding="utf-8")
    (work_dir / ".codex").symlink_to(protected_config, target_is_directory=True)
    result = _run_variant_agent(tmp_path, work_dir, agent="codex", stage_skills=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.read_text(encoding="utf-8") == "protected\n"
    assert not (work_dir / ".codex").is_symlink()
    rendered_skill = work_dir / ".codex" / "skills" / "edit-program" / "SKILL.md"
    assert rendered_skill.is_file()
    rendered_text = rendered_skill.read_text(encoding="utf-8")
    assert str(work_dir) not in rendered_text
    assert "src.investigation.program_clone" in rendered_text
    assert "mkdir -p ./program_3" not in rendered_text

    # a preserved idea is detached from a host hardlink
    work_dir = _work_dir(tmp_path, "hardlink")
    protected_file = tmp_path / "protected-idea.json"
    protected_file.write_text('{"idea": true}\n', encoding="utf-8")
    os.link(protected_file, work_dir / "active_idea.json")
    result = _run_variant_agent(
        tmp_path, work_dir, agent="claude",
        agent_script="printf 'agent-write\\n' > active_idea.json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert protected_file.read_text(encoding="utf-8") == '{"idea": true}\n'
    assert (work_dir / "active_idea.json").read_text(encoding="utf-8") == "agent-write\n"

    # unmanaged workspace files are preserved
    work_dir = _work_dir(tmp_path, "unmanaged")
    settings = work_dir / ".codex" / "settings.toml"
    settings.parent.mkdir()
    settings.write_text("approval = 'never'\n", encoding="utf-8")
    custom_module = work_dir / "custom_package" / "custom.py"
    custom_module.parent.mkdir()
    custom_module.write_text("VALUE = 1\n", encoding="utf-8")
    result = _run_variant_agent(tmp_path, work_dir, agent="codex", stage_skills=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert settings.read_text(encoding="utf-8") == "approval = 'never'\n"
    assert custom_module.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_container_home_is_staged_outside_workspace_and_refreshes_credentials(
    tmp_path: Path,
) -> None:
    _fake_docker(tmp_path, _CALL_LOGGING_DOCKER)

    # the container home is staged outside the agent workspace
    work_dir = _work_dir(tmp_path, "home")
    protected_home = tmp_path / "protected-home"
    protected_home.mkdir()
    sentinel = protected_home / "sentinel.txt"
    sentinel.write_text("protected\n", encoding="utf-8")
    (work_dir / ".agent_home").symlink_to(protected_home, target_is_directory=True)
    docker_calls = tmp_path / "docker-calls-home.txt"
    result = _run_variant_agent(
        tmp_path, work_dir, agent="codex", containerize=True,
        extra_environment={"DOCKER_CALLS": str(docker_calls)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.read_text(encoding="utf-8") == "protected\n"
    assert (work_dir / ".agent_home").is_symlink()
    assert f"{work_dir}/.agent_home" not in docker_calls.read_text(encoding="utf-8")

    # a reusable container home refreshes credentials
    work_dir = _work_dir(tmp_path, "reusable")
    auth_file = tmp_path / "auth-root" / ".codex" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    real_auth_file = tmp_path / "managed-auth.json"
    real_auth_file.write_text('{"token": "first"}\n', encoding="utf-8")
    auth_file.symlink_to(real_auth_file)
    docker_calls = tmp_path / "docker-calls-reusable.txt"
    environment = {
        "AGENT_CONTAINER_SESSION_NAME": f"test-{tmp_path.name}",
        "AGENT_VARIANT_SESSION_HOME_ROOT": str(tmp_path / "session-homes"),
        "DOCKER_CALLS": str(docker_calls),
    }

    def run() -> subprocess.CompletedProcess[str]:
        return _run_variant_agent(
            tmp_path, work_dir, agent="codex", containerize=True,
            container_auth_root=tmp_path / "auth-root", extra_environment=environment,
        )

    first = run()
    assert first.returncode == 0, first.stdout + first.stderr
    mount = re.search(r"-v ([^:\n]+):/home/agent-home", docker_calls.read_text(encoding="utf-8"))
    assert mount is not None
    persistent_auth = Path(mount.group(1)) / ".codex" / "auth.json"
    assert persistent_auth.read_text(encoding="utf-8") == '{"token": "first"}\n'

    real_auth_file.write_text('{"token": "second"}\n', encoding="utf-8")
    second = run()
    assert second.returncode == 0, second.stdout + second.stderr
    assert persistent_auth.read_text(encoding="utf-8") == '{"token": "second"}\n'


# ---------------------------------------------------------------------------
# safe_staging primitives
# ---------------------------------------------------------------------------


def test_atomic_staging_resists_symlink_races_restrictive_old_trees_and_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # atomic writes never go through a racing symlink
    work_dir = tmp_path / "task_bundle"
    work_dir.mkdir()
    destination = work_dir / "active_rubrics.json"
    protected_file = tmp_path / "protected.json"
    protected_file.write_text("protected\n", encoding="utf-8")
    stop = threading.Event()

    def race_destination() -> None:
        while not stop.is_set():
            try:
                destination.unlink(missing_ok=True)
                destination.symlink_to(protected_file)
            except FileExistsError:
                pass

    racer = threading.Thread(target=race_destination)
    racer.start()
    try:
        for _ in range(25):
            atomic_write_bytes(work_dir, "active_rubrics.json", b"[]\n")
    finally:
        stop.set()
        racer.join()
    assert protected_file.read_text(encoding="utf-8") == "protected\n"

    # tree publication survives restrictive old content
    work_dir = tmp_path / "restrictive" / "task_bundle"
    old_tree = work_dir / ".codex" / "skills"
    blocked = old_tree / "blocked"
    blocked.mkdir(parents=True)
    (blocked / "old.txt").write_text("old\n", encoding="utf-8")
    blocked.chmod(0)
    new_tree = tmp_path / "new-skills"
    new_tree.mkdir()
    (new_tree / "new.txt").write_text("new\n", encoding="utf-8")
    try:
        atomic_copy_tree(work_dir, ".codex/skills", new_tree)
    finally:
        for blocked_path in (work_dir / ".codex").glob("*/blocked"):
            blocked_path.chmod(0o700)
    assert (old_tree / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert not list((work_dir / ".codex").glob(".host-staging-old-*"))

    # tree copies retry partial writes
    work_dir = tmp_path / "partial" / "task_bundle"
    work_dir.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    expected = b"0123456789" * 1_000
    (source / "data.bin").write_bytes(expected)
    real_write = os.write

    def partial_write(file_descriptor: int, content: bytes | memoryview) -> int:
        return real_write(file_descriptor, content[: max(1, len(content) // 2)])

    monkeypatch.setattr(safe_staging.os, "write", partial_write)
    atomic_copy_tree(work_dir, "skills", source)
    assert (work_dir / "skills" / "data.bin").read_bytes() == expected


# ---------------------------------------------------------------------------
# skill_scope
# ---------------------------------------------------------------------------

_PAPER_SKILLS = (
    "retrieve-references", "generate-concept-figure", "scientific-evidence", "write-paper",
    "validate-paper", "memory-log",
)
_STANDALONE_PAPER_SKILLS = tuple(
    "write-paper-standalone" if skill == "write-paper" else skill for skill in _PAPER_SKILLS
)


def test_skill_step_mapping_validation_and_csv_resolution() -> None:
    # expected step mapping
    for step, kwargs, expected in [
        ("idea", {}, ("create-idea", "memory-log")),
        ("run", {}, ("edit-program", "evaluate-program", "verify-rubrics", "memory-log")),
        ("variant_edit", {}, ("edit-program", "memory-log")),
        ("variant_verify", {}, ("verify-rubrics", "memory-log")),
        ("task", {}, ("draft-task", "memory-log")),
        ("paper", {}, _PAPER_SKILLS),
        ("paper", {"paper_profile": "standalone"}, _STANDALONE_PAPER_SKILLS),
    ]:
        assert skills_for_step(step, **kwargs) == expected, (step, kwargs)

    # all mapped skills exist on disk
    skills_root = REPO_ROOT / "src" / "agents" / "skills"
    assert skills_root.is_dir()
    for step in sorted(SKILLS_BY_STEP):
        assert SKILLS_BY_STEP[step], step
        validate_skill_set(skills_root, SKILLS_BY_STEP[step])

    # unknown steps or profiles raise
    for kwargs, match in [
        ({"step": "unknown"}, "Unknown step"),
        ({"step": "paper", "paper_profile": "bogus"}, "Unknown paper profile"),
    ]:
        with pytest.raises(ValueError, match=match):
            skills_for_step(**kwargs)

    # CSV parsing deduplicates and strips
    assert parse_skill_csv(
        " retrieve-references, write-paper-standalone ,validate-paper,write-paper-standalone "
    ) == ("retrieve-references", "write-paper-standalone", "validate-paper")

    # resolution prefers the override CSV, then the profile default
    override = "retrieve-references,scientific-evidence,write-paper-standalone,validate-paper,memory-log"
    assert resolve_skills_for_step("paper", override) == tuple(override.split(","))
    assert (
        resolve_skills_for_step("paper", "", paper_profile="standalone")
        == _STANDALONE_PAPER_SKILLS
    )

    # exactly one paper-writing skill is required
    assert paper_writer_skill(
        ("retrieve-references", "write-paper-standalone", "validate-paper")
    ) == "write-paper-standalone"
    for skills in (
        ("retrieve-references", "validate-paper", "memory-log"),
        ("write-paper", "write-paper-standalone", "validate-paper"),
    ):
        with pytest.raises(ValueError, match="exactly one paper-writing skill"):
            paper_writer_skill(skills)


def _skill_source(root: Path) -> Path:
    src = root / "skills" / "generate-concept-figure"
    pack = src / "drawio-figure" / ".git" / "objects" / "pack"
    pack.mkdir(parents=True)
    (src / "SKILL.md").write_text("skill\n", encoding="utf-8")
    (src / "drawio-figure" / "README.md").write_text("readme\n", encoding="utf-8")
    pack_file = pack / "pack-abc.pack"
    pack_file.write_bytes(b"pack")
    pack_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return src


def test_stage_skill_tree_replaces_read_only_files_and_prunes_nested_git(tmp_path: Path) -> None:
    # restaging over read-only files replaces them
    root = tmp_path / "read-only"
    src = _skill_source(root)
    read_only = src / "drawio-figure" / "fixed.bin"
    read_only.write_bytes(b"v1")
    read_only.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    dst = root / "workspace" / ".claude" / "skills" / "generate-concept-figure"
    stage_skill_tree(src, dst)
    staged = dst / "drawio-figure" / "fixed.bin"
    assert staged.read_bytes() == b"v1"
    assert not os.access(staged, os.W_OK)
    read_only.chmod(stat.S_IWUSR | stat.S_IRUSR)
    read_only.write_bytes(b"v2")
    read_only.chmod(stat.S_IRUSR)
    stage_skill_tree(src, dst)
    assert staged.read_bytes() == b"v2"
    assert (dst / "SKILL.md").read_text(encoding="utf-8") == "skill\n"

    # nested git directories are skipped and pruned
    root = tmp_path / "git"
    src = _skill_source(root)
    dst = root / "workspace" / ".claude" / "skills" / "generate-concept-figure"
    stale = dst / "drawio-figure" / ".git" / "objects" / "pack" / "pack-old.pack"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    stale.chmod(stat.S_IRUSR)
    stage_skill_tree(src, dst)
    assert (dst / "drawio-figure" / "README.md").is_file()
    assert not (dst / "drawio-figure" / ".git").exists()
    stage_skill_tree(src, dst)
    assert not (dst / "drawio-figure" / ".git").exists()


# ---------------------------------------------------------------------------
# content_publication guard
# ---------------------------------------------------------------------------


def test_publication_guard_flags_only_unpublished_sealed_content(tmp_path: Path) -> None:
    for name, public_bytes, sealed_bytes, violates in [
        ("already_public_content", b"shared bytes", b"shared bytes", False),
        ("unpublished_sealed_content", b"known bytes", b"private bytes", True),
    ]:
        root = tmp_path / name
        public_file = root / "public" / "shared.bin"
        sealed_file = root / "sealed" / "answer.bin"
        copied_file = root / "packaged" / "copy.bin"
        for path, content in ((public_file, public_bytes), (sealed_file, sealed_bytes)):
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
        guard = ContentPublicationGuard.capture(
            sealed_files=(sealed_file,), public_files=(public_file,)
        )

        copied_file.parent.mkdir()
        copied_file.write_bytes(sealed_bytes)

        assert guard.first_violation((copied_file,)) == (copied_file if violates else None), name
