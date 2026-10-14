"""Billing policy plumbing in the shell helpers (container_agent.sh, task.sh, concept_figure.sh)."""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "src/scripts/lib/container_agent.sh"
CONCEPT_HELPER = ROOT / "src/scripts/lib/concept_figure.sh"
_PRINT_INNER_CMD = 'printf "%s\\n" "${AGENT_INNER_CMD[@]}"'


def _shell(command: str, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(HELPER))}\n{command}"],
        env=os.environ | environment, capture_output=True, text=True, timeout=10,
    )


def _sanitize(tmp_path: Path, source_text: str, **environment: str) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source, destination = tmp_path / "source.env", tmp_path / "filtered.env"
    source.write_text(source_text)
    result = _shell(
        f"sanitize_env_file_for_docker {shlex.quote(str(source))} "
        f'{shlex.quote(str(destination))} "$(agent_provider_allowed_env_keys codex model)"',
        **environment,
    )
    assert result.returncode == 0, result.stderr
    return destination.read_text()


def _concept_shell(agent: str, body: str, **environment: str) -> subprocess.CompletedProcess[str]:
    command = (
        f"source {shlex.quote(str(CONCEPT_HELPER))}\nAGENT={agent}\nconcept_figure_log() {{ :; }}\n"
        f"select_concept_figure_backend\n{body}"
    )
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=5,
                            env={"PATH": os.environ["PATH"], **environment})
    assert result.returncode == 0, result.stderr
    return result


def test_billing_policy_reaches_headless_and_subscription_bridge_overrides_api_preference() -> None:
    # the billing policy reaches headless without forcing legacy auth
    for agent in ("codex", "claude"):
        result = _shell(f"build_agent_inner_command {agent} prompt model '' 1\n{_PRINT_INNER_CMD}",
                        HEADLESS_BILLING="subscription")
        assert result.returncode == 0, result.stderr
        assert "HEADLESS_BILLING=subscription" in result.stdout.splitlines(), agent
        assert "HEADLESS_CLAUDE_AUTH=api" not in result.stdout, agent

    # the subscription bridge overrides an api billing preference
    result = _shell(f"build_agent_inner_command pi prompt model '' 1\n{_PRINT_INNER_CMD}",
                    HEADLESS_BILLING="api", AGENT_CHATGPT_SUBSCRIPTION="1")
    assert "HEADLESS_BILLING=subscription" in result.stdout.splitlines()
    assert "HEADLESS_BILLING=api" not in result.stdout


def test_provider_env_file_transports_billing_and_backup_key_only_without_subscription_bridge(tmp_path: Path) -> None:
    # billing policy and the codex backup key are transported
    filtered = _sanitize(tmp_path / "plain", "CODEX_API_KEY=test-only\nHEADLESS_BILLING=auto\nUNRELATED=private\n")
    assert filtered == "CODEX_API_KEY=test-only\nHEADLESS_BILLING=auto\n"

    # the subscription bridge never transports backup keys
    filtered = _sanitize(tmp_path / "bridge", "CODEX_API_KEY=test-only\nOPENAI_API_KEY=test-only\nHEADLESS_BILLING=api\n",
                         AGENT_CHATGPT_SUBSCRIPTION="1")
    assert filtered == ""


def test_host_codex_backup_key_is_forwarded_by_name_only():
    for agent, subscription, expected in [
        ("codex", "0", ["-e", "CODEX_API_KEY"]),
        ("claude", "0", []),
        ("codex", "1", []),
    ]:
        result = _shell(
            f"docker_cmd=(); append_codex_backup_env docker_cmd {agent}\n"
            'if [ "${#docker_cmd[@]}" -gt 0 ]; then printf "%s\\n" "${docker_cmd[@]}"; fi',
            CODEX_API_KEY="must-not-appear-in-arguments", AGENT_CHATGPT_SUBSCRIPTION=subscription,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == expected, (agent, subscription)


def test_headless_terminal_billing_exit_stops_all_drafting_retries() -> None:
    source = (ROOT / "src/scripts/task.sh").read_text()
    body = source.split("run_logged_agent_step() {", 1)[1].split("\n}\n", 1)[0]
    for agent in ("codex", "claude"):
        command = f'''set -euo pipefail
AGENT={agent}
PYTHON_BIN=true
AGENT_TRANSIENT_MAX_ATTEMPTS=3
LOG_FILE=/unused
run_logged_agent_step_once() {{
  printf 'called\\n'
  LAST_AGENT_PROCESS_EXIT=78
  LAST_AGENT_LOGGER_EXIT=0
  return 78
}}
agent_transient_retry_delay_seconds() {{ printf 0; }}
run_logged_agent_step() {{{body}
}}
run_logged_agent_step label prompt file || true
printf continued
'''
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=5)
        assert result.returncode == 78, (agent, result.stderr)
        assert result.stdout == "called\n", agent


def test_concept_runtime_preserves_subscription_policy_and_codex_api_key() -> None:
    # an explicit subscription policy is preserved for every agent
    for agent in ("codex", "claude"):
        result = _concept_shell(agent, 'eval "$CONCEPT_FIGURE_ENV_COMMANDS"\nprintf \'%s\' "${HEADLESS_BILLING:-missing}"',
                                HEADLESS_BILLING="subscription")
        assert result.stdout == "subscription", agent

    # codex preserves the invocation api key
    result = _concept_shell("codex", 'eval "$CONCEPT_FIGURE_ENV_COMMANDS"\n[ -n "${CODEX_API_KEY:-}" ] && printf present',
                            CODEX_API_KEY="test-secret")
    assert result.stdout == "present"


def test_concept_claude_keeps_oauth_and_bedrock_backup() -> None:
    body = """printf '%s\\n' "$CONCEPT_FIGURE_SEED_AGENT_AUTH"
eval "$CONCEPT_FIGURE_ENV_COMMANDS"
for key in CLAUDE_CODE_OAUTH_TOKEN AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION; do
  [ -n "${!key-}" ] && printf '%s\\n' "$key"
done
"""
    for legacy_backend in ("0", "1"):
        result = _concept_shell(
            "claude", body, CLAUDE_CODE_USE_BEDROCK=legacy_backend, CLAUDE_CODE_OAUTH_TOKEN="test-oauth",
            AWS_ACCESS_KEY_ID="test-id", AWS_SECRET_ACCESS_KEY="test-secret", AWS_REGION="us-east-1",
        )
        assert result.stdout.splitlines() == [
            "1", "CLAUDE_CODE_OAUTH_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION",
        ], legacy_backend
