"""Container auth seeding: seed_auth_into_home, ChatGPT subscription bridging, pi catalog, image pins."""

from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.investigation.codex_subscription_auth import (
    SubscriptionAuthError,
    stage_subscription_auth,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "src/scripts/lib/container_agent.sh"
VARIANT_AGENT = REPO_ROOT / "src/scripts/variant_agent.sh"
CATALOG = REPO_ROOT / "src/agents/pi/models.json"
DOCKERFILE = REPO_ROOT / "src/docker/agent-cli/Dockerfile"
REFRESH_SENTINEL = "agentic-scientist-access-token-only"
# (agent, auth path relative to the home directory, provider key)
AGENT_AUTH_PATHS: list[tuple[str, str, str]] = [
    ("pi", ".pi/agent/auth.json", "openai-codex"),
    ("opencode", ".local/share/opencode/auth.json", "openai"),
]

q = shlex.quote


def _sh(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    body = f"set -euo pipefail\nsource {q(str(HELPER))}\n{script}"
    return subprocess.run(
        ["bash", "-c", body], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=15
    )


def _ok(script: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
    run = _sh(script, **kwargs)  # type: ignore[arg-type]
    assert run.returncode == 0, run.stdout + run.stderr
    return run


def _jwt(payload: dict[str, object]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def _write_codex_auth(
    path: Path, *, auth_mode: str = "chatgpt", expires: int = 2_000_000_000
) -> dict[str, str]:
    tokens = {
        "access_token": _jwt(
            {"exp": expires, "https://api.openai.com/auth": {"chatgpt_account_id": "account-from-claim"}}
        ),
        "refresh_token": "refresh-from-codex",
        "id_token": "unused-id-token",
        "account_id": "account-from-file",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"auth_mode": auth_mode, "tokens": tokens}), encoding="utf-8")
    path.chmod(0o600)
    return tokens


def _seed_into_prepared_home(agent: str, auth_root: Path, workspace: Path, *, prefix: str = "") -> Path:
    workspace.mkdir(parents=True)
    run = _ok(
        f"home_dir=\"$(prepare_container_home {q(str(workspace))})\"\n"
        f"{prefix}seed_auth_into_home {agent} {q(str(auth_root))} \"$home_dir\"\n"
        "printf '%s\\n' \"$home_dir\"\n"
    )
    return Path(run.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# seed_auth_into_home (shell)
# ---------------------------------------------------------------------------


def test_seed_auth_into_home_copies_only_the_agents_credentials_and_allowlists_subscription_env(
    tmp_path: Path,
) -> None:
    # codex auth copied without other agent dirs
    auth_root = tmp_path / "codex-auth"
    codex_files = {
        "auth.json": '{"token":"t"}\n',
        "config.toml": 'model = "gpt-5"\n',
        "custom.config.toml": 'model_provider = "custom"\n',
        "custom.json": '{"models":[]}\n',
    }
    (auth_root / ".codex").mkdir(parents=True)
    for name, body in codex_files.items():
        (auth_root / ".codex" / name).write_text(body, encoding="utf-8")
    (auth_root / ".claude").mkdir(parents=True)
    (auth_root / ".claude/settings.json").write_text('{"unused":true}\n', encoding="utf-8")

    home_dir = _seed_into_prepared_home(
        "codex", auth_root, tmp_path / "codex-workspace", prefix="AGENT_PROFILE=custom "
    )

    for name, body in codex_files.items():
        assert (home_dir / ".codex" / name).is_file()
        assert (home_dir / ".codex" / name).read_text(encoding="utf-8") == body
    assert not (home_dir / ".claude").exists()

    # claude oauth copied without settings
    auth_root = tmp_path / "claude-auth"
    claude_dir = auth_root / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / ".credentials.json").write_text('{"claudeAiOauth":{}}\n', encoding="utf-8")
    (claude_dir / "auth.json").write_text('{"token":"legacy"}\n', encoding="utf-8")
    (claude_dir / "settings.json").write_text(
        '{"hooks":{"SessionStart":[{"hooks":[{"command":"bash ~/.claude/hooks/moonside_hook.sh"}]}]}}\n',
        encoding="utf-8",
    )

    home_dir = _seed_into_prepared_home("claude", auth_root, tmp_path / "claude-workspace")

    assert (home_dir / ".claude/.credentials.json").is_file()
    assert (home_dir / ".claude/auth.json").is_file()
    assert not (home_dir / ".claude/settings.json").exists()

    # subscription provider allowlist excludes OPENAI_API_KEY
    run = _ok(
        "agent_provider_allowed_env_keys opencode openai/gpt-5.6-sol ''",
        env=os.environ | {"AGENT_CHATGPT_SUBSCRIPTION": "1"},
    )
    assert "OPENAI_API_KEY" not in run.stdout.split()


def test_subscription_seed_bridges_codex_auth_and_pi_home_receives_versioned_catalog(tmp_path: Path) -> None:
    # subscription seed bridges codex auth into each agent's native schema
    for agent, relative_path, provider in AGENT_AUTH_PATHS:
        auth_root = tmp_path / agent / "auth"
        home = tmp_path / agent / "home"
        _write_codex_auth(auth_root / ".codex/auth.json")

        _ok(
            "export AGENT_CHATGPT_SUBSCRIPTION=1\n"
            f"seed_auth_into_home {agent} {q(str(auth_root))} {q(str(home))}\n"
        )

        auth_path = home / relative_path
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
        assert auth[provider]["type"] == "oauth", agent
        assert auth[provider]["refresh"] == REFRESH_SENTINEL, agent
        assert auth_path.stat().st_mode & 0o777 == 0o600, agent

    # pi home receives the versioned catalog (fresh home, then symlinked .pi dir)
    for name, pi_dir_is_symlink in (("fresh_home", False), ("symlinked_pi_dir", True)):
        root = tmp_path / name
        auth_root = root / "auth"
        home_dir = root / "home"
        auth_root.mkdir(parents=True)
        victim = root / "victim" / "models.json"
        if pi_dir_is_symlink:
            home_dir.mkdir()
            victim.parent.mkdir()
            victim.write_text("do not replace", encoding="utf-8")
            (home_dir / ".pi").symlink_to(victim.parent, target_is_directory=True)

        _ok(f"seed_auth_into_home pi {q(str(auth_root))} {q(str(home_dir))}\n")

        assert (home_dir / ".pi/agent/models.json").read_bytes() == CATALOG.read_bytes(), name
        if pi_dir_is_symlink:
            assert victim.read_text(encoding="utf-8") == "do not replace"
            assert not (home_dir / ".pi").is_symlink()


@pytest.mark.parametrize(
    ("agent", "model", "auth_path", "provider", "expects_catalog"),
    [
        ("opencode", "openai/gpt-5.6-sol", ".local/share/opencode/auth.json", "openai", False),
        ("pi", "openai-codex/gpt-5.6-sol", ".pi/agent/auth.json", "openai-codex", True),
    ],
)
def test_variant_agent_executes_managed_subscription_seeding(
    tmp_path: Path, agent: str, model: str, auth_path: str, provider: str, expects_catalog: bool
) -> None:
    auth_root = tmp_path / "auth"
    _write_codex_auth(auth_root / ".codex/auth.json")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    rubrics = tmp_path / "rubrics.json"
    rubrics.write_text("[]\n", encoding="utf-8")
    env_file = tmp_path / "agent.env"
    env_file.write_text("OPENAI_API_KEY=must-not-pass\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for executable in ("docker", "headless"):
        path = fake_bin / executable
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    session_homes = tmp_path / "session-homes"
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "WRAPPER_PYTHON": sys.executable,
        "HEADLESS_BIN": str(fake_bin / "headless"),
        "AGENT_CONTAINER_SESSION_NAME": f"subscription-{agent}",
        "AGENT_VARIANT_SESSION_HOME_ROOT": str(session_homes),
        "AGENT_CHATGPT_SUBSCRIPTION": "1",
        "AGENT_CHATGPT_SUBSCRIPTION_REQUIRED_EXPIRY": "2000000000",
    }

    result = subprocess.run(
        [
            "bash", str(VARIANT_AGENT),
            "--agent", agent,
            "--agent-model", model,
            "--work-dir", str(work_dir),
            "--prompt", "smoke",
            "--rubrics-file", str(rubrics),
            "--containerize",
            "--container-auth-root", str(auth_root),
            "--container-env-file", str(env_file),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    [home] = list(session_homes.iterdir())
    auth = json.loads((home / auth_path).read_text(encoding="utf-8"))
    assert set(auth) == {provider}
    assert auth[provider]["type"] == "oauth"
    assert auth[provider]["refresh"] == REFRESH_SENTINEL
    assert (home / ".pi/agent/models.json").exists() is expects_catalog


# ---------------------------------------------------------------------------
# src.investigation.codex_subscription_auth.stage_subscription_auth
# ---------------------------------------------------------------------------


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


@pytest.fixture()
def codex_auth(tmp_path: Path) -> Path:
    return tmp_path / "auth-root/.codex/auth.json"


def test_stages_codex_chatgpt_tokens_in_native_schema_replacing_existing_credentials(
    tmp_path: Path, codex_auth: Path
) -> None:
    tokens = _write_codex_auth(codex_auth)
    # stages tokens in each agent's native schema
    for agent, relative_path, provider in AGENT_AUTH_PATHS:
        home = tmp_path / f"home-{agent}"
        home.mkdir()

        stage_subscription_auth(agent, codex_auth, home)

        staged_path = home / relative_path
        assert json.loads(staged_path.read_text(encoding="utf-8")) == {
            provider: {
                "type": "oauth",
                "refresh": REFRESH_SENTINEL,
                "access": tokens["access_token"],
                "expires": 2_000_000_000_000,
                "accountId": "account-from-file",
            }
        }, agent
        assert staged_path.stat().st_mode & 0o777 == 0o600, agent
        assert staged_path.parent.stat().st_mode & 0o777 == 0o700, agent

    # replaces provider oauth and discards unrelated credentials
    home = tmp_path / "home-pi-existing"
    home.mkdir()
    native_auth = home / ".pi/agent/auth.json"
    native_auth.parent.mkdir(parents=True)
    refreshed = {
        "type": "oauth",
        "refresh": "rotated-refresh",
        "access": "rotated-access",
        "expires": 2_100_000_000_000,
        "accountId": "account-from-file",
    }
    native_auth.write_text(
        json.dumps({"openai-codex": refreshed, "deepseek": {"type": "api_key", "key": "preserve-me"}}),
        encoding="utf-8",
    )
    native_auth.chmod(0o600)

    stage_subscription_auth("pi", codex_auth, home)

    staged = json.loads(native_auth.read_text(encoding="utf-8"))
    assert staged["openai-codex"]["access"] != refreshed["access"]
    assert staged["openai-codex"]["refresh"] == REFRESH_SENTINEL
    assert "deepseek" not in staged

    # replaces an api key for the subscription provider
    home = tmp_path / "home-opencode-existing"
    home.mkdir()
    native_auth = home / ".local/share/opencode/auth.json"
    native_auth.parent.mkdir(parents=True)
    native_auth.write_text(json.dumps({"openai": {"type": "api", "key": "must-not-survive"}}), encoding="utf-8")
    native_auth.chmod(0o600)

    stage_subscription_auth("opencode", codex_auth, home)

    staged = json.loads(native_auth.read_text(encoding="utf-8"))
    assert staged["openai"]["type"] == "oauth"
    assert "key" not in staged["openai"]


def test_rejects_unsafe_or_insufficient_codex_auth_and_keeps_fixed_launch_deadline(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rejects non-ChatGPT codex auth
    codex_auth = tmp_path / "apikey/.codex/auth.json"
    _write_codex_auth(codex_auth, auth_mode="apikey")
    with pytest.raises(SubscriptionAuthError, match="ChatGPT"):
        stage_subscription_auth("pi", codex_auth, home)
    assert not (home / ".pi").exists()
    # rejects symlinked codex auth
    actual = tmp_path / "actual.json"
    _write_codex_auth(actual)
    codex_auth = tmp_path / "symlinked/.codex/auth.json"
    codex_auth.parent.mkdir(parents=True)
    codex_auth.symlink_to(actual)
    with pytest.raises(SubscriptionAuthError, match="safe regular file"):
        stage_subscription_auth("opencode", codex_auth, home)
    assert not (home / ".local").exists()
    # rejects codex auth readable by other users
    codex_auth = tmp_path / "public/.codex/auth.json"
    _write_codex_auth(codex_auth)
    codex_auth.chmod(0o644)
    with pytest.raises(SubscriptionAuthError, match="private"):
        stage_subscription_auth("pi", codex_auth, home)
    assert os.stat(codex_auth).st_mode & 0o077
    # rejects an access token that cannot cover the matrix job
    codex_auth = tmp_path / "short-lived/.codex/auth.json"
    _write_codex_auth(codex_auth, expires=int(time.time()) + 60)
    with pytest.raises(SubscriptionAuthError, match="will not cover"):
        stage_subscription_auth("pi", codex_auth, home)
    # a fixed launch deadline does not move between stages
    codex_auth = tmp_path / "fixed-deadline/.codex/auth.json"
    deadline = int(time.time()) + 73 * 60 * 60
    _write_codex_auth(codex_auth, expires=deadline + 60)
    monkeypatch.setenv("AGENT_CHATGPT_SUBSCRIPTION_REQUIRED_EXPIRY", str(deadline))

    stage_subscription_auth("pi", codex_auth, home)
    monkeypatch.setattr(time, "time", lambda: deadline)
    stage_subscription_auth("pi", codex_auth, home)


# ---------------------------------------------------------------------------
# pi model catalog / agent-cli image pins
# ---------------------------------------------------------------------------


def test_catalog_models_and_agent_cli_image_pins() -> None:
    # catalog registers global bedrock opus 5
    [model] = json.loads(CATALOG.read_text(encoding="utf-8"))["providers"]["amazon-bedrock"]["models"]
    assert model["id"] == "global.anthropic.claude-opus-5"
    assert model["name"] == "Claude Opus 5 (Global)"
    assert model["reasoning"] is True
    assert model["thinkingLevelMap"]["xhigh"] == "xhigh"
    assert model["cost"] == {"input": 5, "output": 25, "cacheRead": 0.5, "cacheWrite": 6.25}
    assert model["contextWindow"] == 1_000_000
    assert model["maxTokens"] == 128_000
    # catalog registers subscription gpt-5.6-sol
    [model] = json.loads(CATALOG.read_text(encoding="utf-8"))["providers"]["openai-codex"]["models"]
    assert model["id"] == "gpt-5.6-sol"
    assert model["api"] == "openai-codex-responses"
    assert model["baseUrl"] == "https://chatgpt.com/backend-api"
    assert model["reasoning"] is True
    assert model["contextWindow"] == 272_000
    assert model["maxTokens"] == 128_000
    # agent-cli dependencies are reproducibly pinned
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    for pin in (
        "@openai/codex@0.153.4",
        "@anthropic-ai/claude-code@2.1.285",
        "@earendil-works/pi-coding-agent@0.85.1",
        "opencode-ai@1.18.29",
    ):
        assert pin in dockerfile
