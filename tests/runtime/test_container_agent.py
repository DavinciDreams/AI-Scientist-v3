"""container_agent.sh wiring: prompt routing, docker invocation, env passthrough, sessions."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "src/scripts/lib/container_agent.sh"
HEADLESS_AGENTS = ("claude", "codex", "cursor", "gemini", "opencode", "pi")
STDIN_PROMPT_AGENTS = {"claude", "codex", "gemini"}
HEADLESS_PREFIX = ["env", "HEADLESS_MODELS_DEV_CACHE=", "/usr/local/bin/headless"]
YOLO = ["--debug", "--usage", "--allow", "yolo"]
PRINT_INNER = "printf '%s\\n' \"${AGENT_INNER_CMD[@]}\"\n"
ARGS_DOCKER = "printf '%s\\n' \"$@\" > \"$DOCKER_ARGS_FILE\"\n"
CODEX_RUN = "run_agent_prompt codex 'How are you doing' {ws} 1 'test-image:latest' 'bridge' {home}"

q = shlex.quote


def _sh(
    script: str,
    *,
    pre: str = "",
    strict: bool = False,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: int = 15,
) -> subprocess.CompletedProcess[str]:
    """Run `script` in bash after sourcing the helper; `pre` runs before the source line."""
    body = ("set -euo pipefail\n" if strict else "") + pre + f"source {q(str(HELPER))}\n" + script
    return subprocess.run(
        ["/bin/bash", "-c", body],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
    )


def _ok(script: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
    run = _sh(script, **kwargs)  # type: ignore[arg-type]
    assert run.returncode == 0, run.stdout + run.stderr
    return run


def _write_exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _exports(**values: object) -> str:
    return "".join(f"export {key}={q(str(value))}\n" for key, value in values.items())


# ---------------------------------------------------------------------------
# pi container environment / prompt-path helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exports", "env_file_body", "expected"),
    [
        pytest.param("", None, [], id="no_overrides_delegate_to_headless"),
        pytest.param(
            "export PI_CODING_AGENT_PROVIDER=custom PI_CODING_AGENT_MODEL=model\n",
            None,
            ["-e", "PI_CODING_AGENT_PROVIDER", "-e", "PI_CODING_AGENT_MODEL"],
            id="explicit_env_forwarded_by_name",
        ),
        pytest.param(
            "",
            "PI_CODING_AGENT_PROVIDER=custom-provider\nPI_CODING_AGENT_MODEL=custom-model\n",
            [],
            id="env_file_defaults_deferred",
        ),
    ],
)
def test_pi_container_environment(
    tmp_path: Path, exports: str, env_file_body: str | None, expected: list[str]
) -> None:
    env_file = ""
    if env_file_body is not None:
        (tmp_path / ".env").write_text(env_file_body, encoding="utf-8")
        env_file = str(tmp_path / ".env")
    run = _ok(
        "docker_cmd=()\n"
        f"{exports}"
        f"append_pi_container_environment pi {q(env_file)}\n"
        "if [ ${#docker_cmd[@]} -gt 0 ]; then printf '%s\\n' \"${docker_cmd[@]}\"; fi\n"
    )
    assert run.stdout.splitlines() == expected


@pytest.mark.parametrize(
    ("prompt_path", "output_dir", "expected"),
    [
        ("/tmp/paper[review]/prompt.md", "/tmp/paper[review]", "prompt.md"),
        ("/tmp/paper-results//prompt.md", "/tmp/paper-results/", "prompt.md"),
        ("/tmp/paper-results/sub/../prompt.md", "/tmp/paper-results", "prompt.md"),
        ("/tmp/paper-results-copy/prompt.md", "/tmp/paper-results", "/tmp/paper-results-copy/prompt.md"),
        ("/tmp/paper-results/../secret.md", "/tmp/paper-results", "/tmp/paper-results/../secret.md"),
    ],
)
def test_prompt_path_relativization_is_component_aware(
    prompt_path: str, output_dir: str, expected: str
) -> None:
    run = _ok(f"relative_agent_prompt_path {q(prompt_path)} {q(output_dir)}")
    assert run.stdout == f"{expected}\n"


# ---------------------------------------------------------------------------
# prompt file writing / verification
# ---------------------------------------------------------------------------


def test_write_agent_prompt_file_replaces_untrusted_file_and_directory_symlinks(tmp_path: Path) -> None:
    for target_kind in ("file", "directory"):
        prompt_file = tmp_path / target_kind / "prompt.md"
        prompt_file.parent.mkdir()
        if target_kind == "file":
            target = tmp_path / target_kind / "protected.txt"
            target.write_text("protected\n", encoding="utf-8")
            prompt_file.symlink_to(target)
        else:
            target = tmp_path / target_kind / "external"
            target.mkdir()
            prompt_file.symlink_to(target, target_is_directory=True)

        _ok(f"write_agent_prompt_file {q(str(prompt_file))} 'trusted prompt'")

        assert not prompt_file.is_symlink(), target_kind
        assert prompt_file.read_text(encoding="utf-8") == "trusted prompt\n", target_kind
        if target_kind == "file":
            assert target.read_text(encoding="utf-8") == "protected\n"
        else:
            assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    ("content", "symlink", "expected_rc"),
    [
        pytest.param("stale prompt\n", False, 1, id="stale_content_rejected"),
        pytest.param("trusted prompt\n", False, 0, id="exact_regular_file_accepted"),
        pytest.param("trusted prompt\n", True, 1, id="symlink_rejected"),
    ],
)
def test_agent_prompt_file_matches_text(
    tmp_path: Path, content: str, symlink: bool, expected_rc: int
) -> None:
    prompt_file = tmp_path / "prompt.md"
    if symlink:
        target = tmp_path / "target.md"
        target.write_text(content, encoding="utf-8")
        prompt_file.symlink_to(target)
    else:
        prompt_file.write_text(content, encoding="utf-8")

    run = _sh(f"agent_prompt_file_matches_text {q(str(prompt_file))} 'trusted prompt'")
    assert run.returncode == expected_rc, run.stderr


# ---------------------------------------------------------------------------
# env passthrough / env-file sanitising
# ---------------------------------------------------------------------------


PASSTHROUGH_CASES = [
    (
        "variable_name_without_secret_value",
        {"TEST_CONTAINER_SECRET": "must-not-appear", "AGENT_CONTAINER_PASSTHROUGH_KEYS": "TEST_CONTAINER_SECRET"},
        ["-e", "TEST_CONTAINER_SECRET"],
    ),
    (
        "subscription_uses_positive_allowlist",
        {
            "AGENT_CHATGPT_SUBSCRIPTION": "1",
            "AGENT_CONTAINER_PASSTHROUGH_KEYS": "HF_TOKEN PI_CODING_AGENT_MODEL",
            "HF_TOKEN": "must-not-pass",
            "PI_CODING_AGENT_MODEL": "gpt-5",
        },
        ["-e", "PI_CODING_AGENT_MODEL"],
    ),
]
SANITIZE_CASES = [
    (
        "deny_list_excludes_openai_api_key",
        "OPENAI_API_KEY=api-secret\nDEEPSEEK_API_KEY=deepseek-secret\nSAFE_SETTING=yes\n",
        "AGENT_CONTAINER_ACTIVE_DENY_ENV_KEYS=OPENAI_API_KEY\n",
        "",
        ["DEEPSEEK_API_KEY=deepseek-secret", "SAFE_SETTING=yes"],
    ),
    (
        "subscription_strict_mode_drops_all_entries",
        "OPENAI_API_KEY\nS2_API_KEY=sensitive\nSAFE_SETTING=yes\n",
        "export AGENT_CHATGPT_SUBSCRIPTION=1\n",
        " ''",
        [],
    ),
]


def test_passthrough_envs_forward_names_only_and_env_file_sanitising_drops_denied_keys(tmp_path: Path) -> None:
    # append_passthrough_envs
    for name, extra_env, expected in PASSTHROUGH_CASES:
        run = _ok(
            "docker_cmd=(); append_passthrough_envs; printf '%s\\n' \"${docker_cmd[@]}\"",
            env=os.environ | extra_env,
            cwd=REPO_ROOT,
        )
        assert run.stdout.splitlines() == expected, name
        assert "must-not" not in run.stdout + run.stderr, name
    # sanitize_env_file_for_docker
    for name, source_body, setup, trailing, expected in SANITIZE_CASES:
        source = tmp_path / f"{name}.env"
        destination = tmp_path / f"{name}.destination.env"
        source.write_text(source_body, encoding="utf-8")
        _ok(
            f"{setup}sanitize_env_file_for_docker {q(str(source))} {q(str(destination))}{trailing}\n",
            strict=True,
            cwd=REPO_ROOT,
        )
        assert destination.read_text(encoding="utf-8").splitlines() == expected, name


# ---------------------------------------------------------------------------
# native pi routing through headless
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["small_prompt", "large_prompt_file", "headless_bin_only"])
def test_pi_native_prompt_routes_through_headless(tmp_path: Path, mode: str) -> None:
    bin_dir = tmp_path / "bin"
    captured_args = tmp_path / "captured-args.txt"
    captured_stdin = tmp_path / "captured-stdin.txt"
    fake_headless = _write_exe(
        bin_dir / "headless",
        "printf '%s\\n' \"$@\" > \"$CAPTURE_FILE\"\ncat > \"$CAPTURE_STDIN\"\n",
    )
    env = os.environ | {
        "HEADLESS_BIN": str(fake_headless),
        "CAPTURE_FILE": str(captured_args),
        "CAPTURE_STDIN": str(captured_stdin),
    }
    if mode != "headless_bin_only":
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
    tail = f"\"\" \"\" {q(str(tmp_path))} 0 unused bridge {q(str(tmp_path / 'home'))}"

    if mode == "large_prompt_file":
        large_prompt = "x" * 150_000
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text(large_prompt, encoding="utf-8")
        script = (
            f"prompt_text=\"$(cat {q(str(prompt_file))})\"\n"
            f"AGENT_PROMPT_FILE={q(str(prompt_file))} run_agent_prompt pi \"$prompt_text\" {tail}"
        )
        expected = [
            "pi",
            *YOLO,
            "--prompt",
            f"Read the complete task instructions from {prompt_file}, then follow them exactly.",
        ]
    else:
        prompt = "small prompt" if mode == "small_prompt" else "prompt"
        script = f"run_agent_prompt pi {q(prompt)} {tail}"
        expected = ["pi", *YOLO, "--prompt", prompt]

    _ok(script, env=env)
    args = captured_args.read_text(encoding="utf-8").splitlines()
    if mode == "headless_bin_only":
        assert args[:6] == expected[:6]
    else:
        assert args == expected
    if mode == "large_prompt_file":
        assert large_prompt not in args
        assert captured_stdin.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# build_agent_inner_command
# ---------------------------------------------------------------------------


INNER_ARGV_CASES = [
    (
        "codex_profile_forwarded",
        "codex prompt test-model custom 1",
        [*HEADLESS_PREFIX, "codex", *YOLO, "--model", "test-model", "--profile", "custom", "--prompt", "prompt"],
    ),
    ("claude_default_api_auth_precedence", "claude prompt '' '' 1", [*HEADLESS_PREFIX, "claude", *YOLO, "--prompt", "prompt"]),
    ("pi_headless_yolo", "pi prompt '' '' 1", [*HEADLESS_PREFIX, "pi", *YOLO, "--prompt", "prompt"]),
]


def test_inner_command_exact_argv_and_claude_prompt_file_keeps_large_prompt_out_of_argv() -> None:
    # exact argv per agent
    for name, call, expected in INNER_ARGV_CASES:
        run = _ok(f"build_agent_inner_command {call}\n{PRINT_INNER}")
        assert run.stdout.splitlines() == expected, name
    # a 200 KB prompt must go via AGENT_PROMPT_FILE, never inline argv (MAX_ARG_STRLEN / E2BIG)
    measure = (
        "maxlen=0\n"
        'for a in "${AGENT_INNER_CMD[@]}"; do (( ${#a} > maxlen )) && maxlen=${#a}; done\n'
    )
    run = _ok(
        "big=\"$(head -c 200000 < /dev/zero | tr '\\0' 'X')\"\n"
        "AGENT_PROMPT_FILE='paper_repair_prompt.md' "
        "build_agent_inner_command claude \"$big\" 'claude-opus-4-6[1m]' ''\n"
        f"{measure}echo \"MAXLEN_FILE=$maxlen\"\n"
        "build_agent_inner_command claude \"$big\" 'claude-opus-4-6[1m]' ''\n"
        f"{measure}echo \"MAXLEN_INLINE=$maxlen\"\n",
        strict=True,
    )
    values = dict(line.split("=", 1) for line in run.stdout.splitlines() if line.startswith("MAXLEN_"))
    assert int(values["MAXLEN_FILE"]) < 128 * 1024, run.stdout
    assert int(values["MAXLEN_INLINE"]) >= 200000, run.stdout


@pytest.mark.parametrize("agent", HEADLESS_AGENTS)
def test_every_agent_uses_headless_yolo(agent: str) -> None:
    reasoning_effort = "" if agent == "gemini" else "high"
    run = _ok(
        f"export AGENT_REASONING_EFFORT={q(reasoning_effort)}\n"
        f"build_agent_inner_command {agent} prompt test-model '' 1\n{PRINT_INNER}"
    )
    args = run.stdout.splitlines()
    assert args[:4] == [*HEADLESS_PREFIX, agent]
    assert "--usage" in args
    assert "--debug" in args
    assert "--json" not in args
    assert args[args.index("--allow") + 1] == "yolo"


def test_headless_binary_selection_model_default_wrapper_python_and_idea_script_pins() -> None:
    # container runtime ignores host HEADLESS_BIN override
    run = _ok(
        "export HEADLESS_BIN=/workspace/untrusted-headless\n"
        f"build_agent_inner_command codex prompt test-model '' 1\n{PRINT_INNER}"
    )
    assert run.stdout.splitlines()[2] == "/usr/local/bin/headless"
    assert "/workspace/untrusted-headless" not in run.stdout
    # native default uses the pinned headless revision
    run = _ok(f"build_agent_inner_command codex prompt '' '' 0\n{PRINT_INNER}")
    assert run.stdout.splitlines()[:6] == [
        "env",
        "HEADLESS_MODELS_DEV_CACHE=",
        "npx",
        "-y",
        "git+https://github.com/RobertTLange/headless-cli.git#93cd9b06b85f848af1308c41e018991b33907c5e",
        "codex",
    ]
    # unset model defers to the headless default
    run = _ok(f"build_agent_inner_command codex prompt '' '' 1\n{PRINT_INNER}")
    assert "--model" not in run.stdout.splitlines()
    # container agent uses the wrapper python without PATH fallback
    run = _sh(
        f"test \"$AGENT_PYTHON_BIN\" = {q(sys.executable)}",
        env={"PATH": "/nonexistent", "WRAPPER_PYTHON": sys.executable},
    )
    assert run.returncode == 0, run.stderr
    # idea script has no native pi override
    idea_script = (REPO_ROOT / "src/scripts/idea.sh").read_text(encoding="utf-8")
    assert "PI_CODING_AGENT_TOOLS" not in idea_script


def test_claude_billing_left_to_headless_and_codex_profile_requires_explicit_model(tmp_path: Path) -> None:
    # modeled claude headless leaves billing selection to headless
    captured = tmp_path / "auth-mode.txt"
    fake_headless = _write_exe(
        tmp_path / "headless", "printf '%s' \"${HEADLESS_CLAUDE_AUTH:-}\" > \"$CAPTURE_AUTH_MODE\"\n"
    )
    _ok(
        f"export HEADLESS_BIN={q(str(fake_headless))}\n"
        "build_agent_inner_command claude prompt claude-opus-4-8 ''\n"
        '"${AGENT_INNER_CMD[@]}"\n',
        env=os.environ | {"ANTHROPIC_API_KEY": "test-api-key", "CAPTURE_AUTH_MODE": str(captured)},
        cwd=tmp_path,
    )
    assert captured.read_text(encoding="utf-8") == ""
    # codex profile requires an explicit model (missing or blank)
    for model in ("", "   "):
        run = _sh(f"build_agent_inner_command codex prompt {q(model)} custom\n{PRINT_INNER}")
        assert run.returncode == 2, repr(model)
        assert "agent model is required when agent profile is provided" in run.stderr, repr(model)


@pytest.mark.parametrize("agent", HEADLESS_AGENTS)
def test_large_prompt_file_reaches_headless_without_large_argv(tmp_path: Path, agent: str) -> None:
    bin_dir = tmp_path / "bin"
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_bytes(b"X" * 200_000)
    captured_args = tmp_path / "args.txt"
    captured_stdin = tmp_path / "stdin.txt"
    fake_headless = _write_exe(
        bin_dir / "headless",
        "set -euo pipefail\n"
        "max_arg_len=0\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"${#arg}\" -gt \"$max_arg_len\" ]; then max_arg_len=${#arg}; fi\n"
        "done\n"
        "printf 'MAX_ARG_LEN=%s\\n' \"$max_arg_len\" > \"$CAPTURE_ARGS\"\n"
        "printf '%s\\n' \"$@\" >> \"$CAPTURE_ARGS\"\n"
        "cat > \"$CAPTURE_STDIN\"\n",
    )
    reasoning_effort = "" if agent == "gemini" else "high"
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CAPTURE_ARGS": str(captured_args),
        "CAPTURE_STDIN": str(captured_stdin),
    }

    _ok(
        _exports(
            AGENT_PROMPT_FILE=prompt_file,
            HEADLESS_BIN=fake_headless,
            AGENT_REASONING_EFFORT=reasoning_effort,
        )
        + f"build_agent_inner_command {q(agent)} 'INLINE_SENTINEL' 'test-model' ''\n"
        '"${AGENT_INNER_CMD[@]}"\n',
        strict=True,
        env=env,
        cwd=tmp_path,
    )

    captured = captured_args.read_text(encoding="utf-8")
    assert int(captured.splitlines()[0].split("=", 1)[1]) < 128 * 1024
    assert "INLINE_SENTINEL" not in captured
    assert captured_stdin.read_bytes() == b""
    expected = [agent, *YOLO, "--model", "test-model"]
    if reasoning_effort:
        expected += ["--reasoning-effort", reasoning_effort]
    if agent in STDIN_PROMPT_AGENTS:
        expected += ["--prompt-file", str(prompt_file)]
    else:
        expected += [
            "--prompt",
            f"Read the complete task instructions from {prompt_file}, then follow them exactly.",
        ]
    assert captured.splitlines()[1:] == expected


# ---------------------------------------------------------------------------
# headless cost log recording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("via_fd3", [False, True], ids=["stdout", "cost_output_fd3"])
def test_record_headless_cost_log(tmp_path: Path, via_fd3: bool) -> None:
    cost_root = tmp_path / "run"
    log_path = tmp_path / "agent.jsonl"
    output_tokens = 50 if via_fd3 else 2
    payload = json.dumps(
        {
            "type": "result",
            "total_cost_usd": 0.5 if via_fd3 else 0.75,
            "usage": {"input_tokens": 10, "output_tokens": output_tokens},
        }
    )
    if via_fd3:
        redirect = "exec 3>&1\nexec 1>/dev/null\nexport AGENTIC_COST_OUTPUT_FD=3\n"
        write_log = f"printf '%s\\n' {q(payload)} > {q(str(log_path))}\n"
        invocation = "paper-draft-fd3"
    else:
        redirect = "export AGENTIC_COST_STAGE=draft\nexport AGENTIC_COST_INVOCATION=paper-draft-1\n"
        write_log = (
            f"printf '%s\\n' {q(payload)} | bounded_agent_log_writer \"$PYTHON_BIN\" {q(str(log_path))} 4096\n"
        )
        invocation = "paper-draft-1"

    run = _ok(
        f"{redirect}"
        f"export AGENTIC_COST_ROOT={q(str(cost_root))}\n"
        "export AGENTIC_COST_PHASE=paper\nexport AGENTIC_COST_TERMINAL=1\nexport AGENT=claude\n"
        f"{write_log}"
        f"record_headless_cost_log \"$PYTHON_BIN\" {q(str(log_path))} draft {invocation} completed\n",
        strict=True,
        cwd=REPO_ROOT,
    )

    assert f"Tokens · paper/draft · {output_tokens} output · run {output_tokens}" in run.stdout
    if not via_fd3:
        summary = json.loads((cost_root / "metrics/cost_summary.json").read_text(encoding="utf-8"))
        assert summary["reported_cost_usd"] == "0.75"
        assert summary["by_phase"]["paper"]["event_count"] == 1


# ---------------------------------------------------------------------------
# docker invocation wiring (fake docker on PATH)
# ---------------------------------------------------------------------------


class _DockerSandbox:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.workspace = root / "workspace"
        self.container_home = root / "container-home"
        self.bin_dir = root / "bin"
        self.args_file = root / "docker_args.txt"
        for directory in (self.workspace, self.container_home, self.bin_dir):
            directory.mkdir(parents=True)

    def codex_run(self, *, workspace: Path | None = None, extra: str = "") -> str:
        ws = q(str(workspace or self.workspace))
        return CODEX_RUN.format(ws=ws, home=q(str(self.container_home))) + extra + "\n"

    def run(
        self,
        script: str,
        *,
        docker_body: str = ARGS_DOCKER,
        exports: dict[str, object] | None = None,
        timeout: int = 15,
    ) -> subprocess.CompletedProcess[str]:
        _write_exe(self.bin_dir / "docker", docker_body)
        pre = (
            f"export PATH=\"{self.bin_dir}:$PATH\"\n"
            + _exports(WRAPPER_PYTHON=sys.executable, DOCKER_ARGS_FILE=self.args_file, **(exports or {}))
        )
        return _ok(script, pre=pre, strict=True, timeout=timeout)

    def args(self) -> list[str]:
        assert self.args_file.is_file()
        return self.args_file.read_text(encoding="utf-8").splitlines()


@pytest.fixture()
def dock(tmp_path: Path) -> _DockerSandbox:
    return _DockerSandbox(tmp_path)


def test_run_agent_prompt_mounts_seed_home_env_file_results_root_and_task_source(dock: _DockerSandbox) -> None:
    # mounts seed home at a stable path
    dock.run(dock.codex_run())
    args = dock.args()
    assert "HOME=/home/agent-home" in args
    assert f"{dock.container_home}:/home/agent-home" in args
    assert f"{dock.container_home}:/home/agent" not in args
    # forwards env file to docker
    env_file = dock.root / ".env"
    env_file.write_text("AWS_REGION=us-east-1\n", encoding="utf-8")
    dock.run(dock.codex_run(extra=f" {q(str(env_file))}"))
    args = dock.args()
    env_idx = args.index("--env-file")
    assert env_idx + 1 < len(args)
    assert args[env_idx + 1].strip()
    # mounts results root and uses nested workdir
    results_root = dock.root / "results" / "codex"
    workspace = results_root / "worker_01" / "_idea_generation" / "cycle_0001"
    workspace.mkdir(parents=True)
    dock.run(dock.codex_run(workspace=workspace, extra=f" '' {q(str(results_root))}"))
    args = dock.args()
    assert f"{results_root}:/workspace" in args
    assert f"{workspace}:/workspace" not in args
    workdir_idx = args.index("-w")
    assert workdir_idx + 1 < len(args)
    assert args[workdir_idx + 1] == "/workspace/worker_01/_idea_generation/cycle_0001"
    # mounts external task source
    task_source = dock.root / "examples" / "circpack"
    results_root = dock.root / "results" / "circpack"
    workspace = results_root / "attempt_1"
    task_source.mkdir(parents=True)
    workspace.mkdir(parents=True)
    dock.run(
        dock.codex_run(workspace=workspace, extra=f" '' {q(str(results_root))}"),
        exports={"TASK_SOURCE_DIR": task_source},
    )
    args = dock.args()
    assert f"{task_source}:/task-source:ro" in args
    assert "TASK_SOURCE_DIR=/task-source" in args


RETRYING_DOCKER = """set -euo pipefail
if [ "${1:-}" = "info" ]; then exit 0; fi
if [ "${1:-}" = "run" ]; then
  count=0
  if [ -f "$DOCKER_ATTEMPT_FILE" ]; then count="$(cat "$DOCKER_ATTEMPT_FILE")"; fi
  count=$((count + 1))
  printf '%s\\n' "$count" > "$DOCKER_ATTEMPT_FILE"
  if [ "$count" -eq 1 ]; then
    echo "docker: request returned 500 Internal Server Error for API route and version http://%2Ftmp%2Fdocker.sock/_ping" >&2
    exit 125
  fi
fi
printf '%s\\n' "$@" > "$DOCKER_ARGS_FILE"
"""


def test_run_agent_prompt_sanitizes_quoted_env_values_and_retries_transient_docker_failure(
    dock: _DockerSandbox,
) -> None:
    # sanitizes quoted env values for docker
    env_file = dock.root / ".env"
    env_capture = dock.root / "docker_env_capture.txt"
    env_file.write_text(
        'CURSOR_API_KEY="abc123"\nAWS_REGION=\'us-east-1\'\nUNQUOTED=value\n', encoding="utf-8"
    )
    dock.run(
        dock.codex_run(extra=f" {q(str(env_file))}"),
        docker_body=ARGS_DOCKER
        + "while [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = \"--env-file\" ]; then cp \"$2\" \"$DOCKER_ENV_CAPTURE\"; break; fi\n"
        "  shift\n"
        "done\n",
        exports={"DOCKER_ENV_CAPTURE": env_capture},
    )
    assert dock.args_file.is_file()
    docker_env = env_capture.read_text(encoding="utf-8")
    for line in ("CURSOR_API_KEY=abc123", "AWS_REGION=us-east-1", "UNQUOTED=value"):
        assert line in docker_env
    assert 'CURSOR_API_KEY="abc123"' not in docker_env
    assert "AWS_REGION='us-east-1'" not in docker_env
    # retries a transient docker failure
    dock.args_file.unlink()
    attempt_file = dock.root / "docker_attempts.txt"
    run = dock.run(
        dock.codex_run(),
        docker_body=RETRYING_DOCKER,
        exports={
            "DOCKER_ATTEMPT_FILE": attempt_file,
            "AGENT_DOCKER_RUN_RETRIES": 2,
            "AGENT_DOCKER_RUN_RETRY_SLEEP_SECONDS": 0,
            "AGENT_DOCKER_READY_RETRIES": 1,
            "AGENT_DOCKER_READY_SLEEP_SECONDS": 0,
        },
    )
    assert attempt_file.read_text(encoding="utf-8").strip() == "2"
    assert "transient docker failure" in run.stdout + run.stderr
    assert dock.args_file.is_file()


SESSION_DOCKER = """set -euo pipefail
printf '%s|%s\\n' "$1" "$*" >> "$DOCKER_CALLS_FILE"
cmd="$1"
shift || true
case "$cmd" in
  info) exit 0 ;;
  inspect)
    if [ "${1:-}" = "-f" ]; then name="${3:-}"; else name="${1:-}"; fi
    if [ -f "$DOCKER_STATE_DIR/$name" ]; then
      if [ "${1:-}" = "-f" ]; then echo true; fi
      exit 0
    fi
    exit 1
    ;;
  run)
    name=''
    prev=''
    for token in "$@"; do
      if [ "$prev" = "--name" ]; then name="$token"; break; fi
      prev="$token"
    done
    if [ -n "$name" ]; then touch "$DOCKER_STATE_DIR/$name"; fi
    echo container-id
    exit 0
    ;;
  exec) echo ok; exit 0 ;;
  rm)
    if [ "${1:-}" = "-f" ]; then rm -f "$DOCKER_STATE_DIR/${2:-}" || true; exit 0; fi
    ;;
esac
exit 0
"""


def test_run_agent_prompt_session_reuse_codex_profile_exec_and_non_codex_profile_rejection(
    dock: _DockerSandbox,
) -> None:
    # reuses the session container
    results_root = dock.root / "results" / "codex"
    workspace = results_root / "worker_01" / "_idea_generation" / "cycle_0001"
    container_home = results_root / ".agent_home"
    state_dir = dock.root / "state"
    calls_file = dock.root / "docker_calls.txt"
    for directory in (workspace, container_home, state_dir):
        directory.mkdir(parents=True)
    dock.container_home = container_home

    dock.run(
        dock.codex_run(workspace=workspace)
        + dock.codex_run(workspace=workspace).replace("How are you doing", "How are you doing again"),
        docker_body=SESSION_DOCKER,
        exports={
            "DOCKER_CALLS_FILE": calls_file,
            "DOCKER_STATE_DIR": state_dir,
            "AGENT_CONTAINER_SESSION_NAME": "codex-worker-01",
            "AGENT_CONTAINER_MOUNT_ROOT": results_root,
            "TEST_CONTAINER_SECRET": "must-not-appear",
            "AGENT_CONTAINER_PASSTHROUGH_KEYS": "TEST_CONTAINER_SECRET",
        },
        timeout=20,
    )

    calls = calls_file.read_text(encoding="utf-8").splitlines()
    run_calls = [line for line in calls if line.startswith("run|")]
    exec_calls = [line for line in calls if line.startswith("exec|")]
    assert len(run_calls) == 1, "\n".join(calls)
    assert len(exec_calls) == 2, "\n".join(calls)
    assert "--name codex-worker-01" in run_calls[0]
    assert "-e TEST_CONTAINER_SECRET" in run_calls[0]
    assert "sleep infinity" in run_calls[0]
    assert "-e TEST_CONTAINER_SECRET" in exec_calls[0]
    assert "must-not-appear" not in "\n".join(calls)
    assert "-w /workspace/worker_01/_idea_generation/cycle_0001" in exec_calls[0]
    assert "codex-worker-01 env HEADLESS_MODELS_DEV_CACHE= /usr/local/bin/headless codex" in exec_calls[0]
    assert "codex-worker-01 codex exec" not in exec_calls[0]
    # adds the codex profile to the session exec
    dock.container_home = dock.root / "container-home"
    dock.run(
        f"run_agent_prompt codex 'How are you doing' custom-model custom {q(str(dock.workspace))} 1 "
        f"'test-image:latest' 'bridge' {q(str(dock.container_home))}\n"
    )
    args = dock.args()
    assert "/usr/local/bin/headless" in args
    assert "codex" in args
    assert args[args.index("--profile") + 1] == "custom"
    assert args[args.index("--model") + 1] == "custom-model"
    # rejects a non-codex profile
    run = _sh(
        f"run_agent_prompt claude 'How are you doing' '' custom {q(str(dock.workspace))} 0 "
        f"'test-image:latest' 'bridge' {q(str(dock.container_home))}\n",
        strict=True,
    )
    output = run.stdout + run.stderr
    assert run.returncode == 2, output
    assert "agent profile is only supported for codex" in output


def test_run_command_prompt_runs_command_in_container_or_natively(dock: _DockerSandbox) -> None:
    # runs the command in the container
    dock.run(
        f"run_command_prompt claude 'echo hello-figure' {q(str(dock.workspace))} 1 "
        f"'test-image:latest' 'bridge' {q(str(dock.container_home))}\n"
    )
    args = dock.args()
    for expected in (
        "HOME=/home/agent-home",
        f"{dock.container_home}:/home/agent-home",
        "test-image:latest",
        "bash",
        "-lc",
        "echo hello-figure",
    ):
        assert expected in args
    # runs the command natively without a container
    marker = dock.root / "marker.txt"
    _ok(
        f"run_command_prompt claude 'pwd > \"{marker}\"' {q(str(dock.workspace))} 0 "
        f"'' 'bridge' {q(str(dock.container_home))}\n",
        pre=_exports(WRAPPER_PYTHON=sys.executable),
        strict=True,
    )
    assert marker.read_text(encoding="utf-8").strip() == str(dock.workspace.resolve())


# ---------------------------------------------------------------------------
# stop_container_session_force
# ---------------------------------------------------------------------------

STOP_RETRY_DOCKER = """set -euo pipefail
cmd="${1:-}"
shift || true
case "$cmd" in
  inspect)
    if [ -f "$DOCKER_STATE_DIR/${1:-}" ]; then exit 0; fi
    exit 1
    ;;
  rm)
    if [ "${1:-}" = "-f" ]; then
      count=0
      if [ -f "$DOCKER_ATTEMPT_FILE" ]; then count="$(cat "$DOCKER_ATTEMPT_FILE")"; fi
      count=$((count + 1))
      printf '%s\\n' "$count" > "$DOCKER_ATTEMPT_FILE"
      if [ "$count" -eq 1 ]; then echo "daemon busy" >&2; exit 1; fi
      rm -f "$DOCKER_STATE_DIR/${2:-}"
      exit 0
    fi
    ;;
esac
exit 0
"""

STOP_MISSING_DOCKER = """set -euo pipefail
printf '%s\\n' "$*" >> "$DOCKER_CALLS_FILE"
if [ "${1:-}" = "inspect" ]; then exit 1; fi
if [ "${1:-}" = "rm" ]; then echo "unexpected rm" >&2; exit 2; fi
exit 0
"""


def test_stop_container_session_force_retries_then_succeeds_and_treats_missing_as_success(
    dock: _DockerSandbox,
) -> None:
    # retries then succeeds
    state_dir = dock.root / "state"
    state_dir.mkdir()
    attempt_file = dock.root / "stop_attempts.txt"
    (state_dir / "codex-worker-01").write_text("", encoding="utf-8")
    dock.run(
        "stop_container_session_force 'codex-worker-01'\n",
        docker_body=STOP_RETRY_DOCKER,
        exports={
            "DOCKER_STATE_DIR": state_dir,
            "DOCKER_ATTEMPT_FILE": attempt_file,
            "AGENT_DOCKER_STOP_RETRIES": 2,
            "AGENT_DOCKER_STOP_RETRY_SLEEP_SECONDS": 0,
        },
    )
    assert attempt_file.read_text(encoding="utf-8").strip() == "2"
    assert not (state_dir / "codex-worker-01").exists()
    # treats a missing container as success
    calls_file = dock.root / "docker_calls.txt"
    dock.run(
        "stop_container_session_force 'codex-worker-missing'\n",
        docker_body=STOP_MISSING_DOCKER,
        exports={"DOCKER_CALLS_FILE": calls_file},
    )
    calls = calls_file.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("inspect codex-worker-missing") for line in calls)
    assert not any(line.startswith("rm -f") for line in calls), "\n".join(calls)
