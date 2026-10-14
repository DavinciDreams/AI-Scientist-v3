"""Private scratch storage, container temp leaves, bounded log sinks, and temp artifact cleanup."""
from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.investigation.evaluation_artifact_cleanup as cleanup_module
from src.investigation import temp_storage
from src.investigation.evaluation_artifact_cleanup import cleanup_unmanaged_evaluation_outputs
from src.investigation.runtime import container_execution_env
from src.investigation.task_runtime_cache import (
    prepare_task_runtime_install_cache,
    task_runtime_install_cache_root,
)
from src.investigation.temp_cleanup import validate_temp_directory, validate_temp_file
from src.scripts.lib import bounded_log, container_temp

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "src/scripts"
HELPER = SCRIPTS / "lib/container_agent.sh"
ENV_NAMES = (
    "AGENTIC_TEMP_ROOT", "AGENTIC_TEMP_DIR", "AGENTIC_CACHE_ROOT",
    "AGENTIC_CONTAINER_TEMP_ROOT", "TMPDIR", "TMP", "TEMP",
)
_STATEFUL_DOCKER = '''#!/usr/bin/env python3
import os, pathlib, sys
state = pathlib.Path(os.environ["DOCKER_STATE"])
if os.environ.get("DOCKER_CALLS"):
    with open(os.environ["DOCKER_CALLS"], "a") as output:
        output.write(sys.argv[1] + "\\n")
if sys.argv[1] == "inspect" and state.exists():
    print(state.read_text())
elif sys.argv[1] == "rm" and os.environ.get("DOCKER_STOP_FAIL") != "1":
    state.unlink(missing_ok=True)
elif sys.argv[1] == "inspect":
    print("Error: No such container: session", file=sys.stderr)
    sys.exit(1)
else:
    sys.exit(1)
'''


# --- helpers --------------------------------------------------------------------------------


def _script(name: str) -> str:
    return (SCRIPTS / name).read_text()


def _bash_function(source: str, name: str) -> str:
    """The complete ``name() { ... }`` definition extracted from a shell script."""
    start = source.index(f"{name}() {{")
    return source[start:source.index("\n}\n", start) + 3]


def _bash(command: str, *, env=None, args=(), timeout=10, check=False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command, *args], env=env, capture_output=True, text=True, timeout=timeout, check=check
    )


def _fake_docker(bin_dir: Path, script: str) -> Path:
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(script)
    docker.chmod(0o755)
    return docker


def _inspect(leaf: Path) -> dict:
    return {
        "Name": "/session",
        "Config": {"Labels": {container_temp.LABEL: str(leaf)}},
        "Mounts": [{"Destination": "/agent-tmp", "Source": str(leaf), "Type": "bind", "RW": True}],
    }


def _run_shell(scratch: Path, script: str) -> subprocess.CompletedProcess[str]:
    return _bash(
        f"source {shlex.quote(str(HELPER))}\n" + script,
        env=os.environ | {"AGENTIC_CONTAINER_TEMP_ROOT": str(scratch)},
        timeout=None,
    )


def _private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700)
    return path


def local_spools(tmp_path, monkeypatch):
    original = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kw: original(prefix="spool-", dir=tmp_path))


def input_stream(monkeypatch, data):
    stream = io.BytesIO(data)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=stream))
    return stream


def _notices(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().err.splitlines()]


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    return _private_dir(tmp_path / "container scratch")


# --- temp_storage (isolated environment) ----------------------------------------------------


class TestTempStorage:
    @pytest.fixture(autouse=True)
    def _isolated_environment(self, monkeypatch):
        monkeypatch.setattr(os, "environ", os.environ.copy())
        monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
        for name in ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(temp_storage, "LOCAL_SSD", Path("/nonexistent-local-ssd"))

    @staticmethod
    @contextlib.contextmanager
    def _fresh(monkeypatch, root: Path | None = None):
        """A section with its own environment copy, optionally pointing AGENTIC_TEMP_ROOT at ``root``."""
        with monkeypatch.context() as m:
            m.setattr(os, "environ", os.environ.copy())
            m.setattr(tempfile, "tempdir", tempfile.tempdir)
            for name in ENV_NAMES:
                m.delenv(name, raising=False)
            if root is not None:
                m.setenv("AGENTIC_TEMP_ROOT", str(root))
            yield m

    def test_explicit_root_creates_private_storage_isolating_invocations_and_sharing_cache(
        self, tmp_path, monkeypatch
    ):
        # explicit root creates private isolated storage
        monkeypatch.setenv("AGENTIC_TEMP_ROOT", str(tmp_path))
        storage = temp_storage.initialize_temp_storage()
        assert storage.base == tmp_path
        assert storage.temporary != storage.containers
        assert not storage.containers.is_relative_to(storage.temporary)
        for path in (storage.temporary, storage.cache, storage.containers):
            assert stat.S_IMODE(path.stat().st_mode) == 0o700
        assert os.environ["TMPDIR"] == str(storage.temporary)
        assert temp_storage.initialize_temp_storage() == storage
        # invocations isolated, cache shared
        for name in ENV_NAMES[1:]:
            monkeypatch.delenv(name, raising=False)
        second = temp_storage.initialize_temp_storage()
        assert storage.temporary != second.temporary
        assert storage.containers != second.containers
        assert storage.cache == second.cache
        # runtime cache resolves configured root and reuses across phases
        with self._fresh(monkeypatch, _private_dir(tmp_path / "phases")) as m:
            task = tmp_path / "task"
            first = prepare_task_runtime_install_cache(task, reset=False)
            assert first.is_relative_to(temp_storage.runtime_cache_directory())
            for name in ENV_NAMES[1:]:
                m.delenv(name, raising=False)
            assert task_runtime_install_cache_root(task) == first

    def test_base_selection_precedence_and_preview(self, tmp_path, monkeypatch):
        # preview never creates namespace
        preview = _private_dir(tmp_path / "preview")
        with self._fresh(monkeypatch, preview):
            assert str(preview) in temp_storage.preview_temp_storage()
            assert list(preview.iterdir()) == []
            assert "AGENTIC_TEMP_DIR" not in os.environ
        # explicit TMPDIR selected
        with self._fresh(monkeypatch) as m:
            m.setenv("TMPDIR", str(tmp_path))
            assert temp_storage.initialize_temp_storage().base == tmp_path
        # default system temp used without SSD
        with self._fresh(monkeypatch) as m:
            m.setattr(temp_storage, "_system_temporary_base", lambda: tmp_path)
            assert temp_storage.initialize_temp_storage().base == tmp_path
        # available local SSD selected
        with self._fresh(monkeypatch) as m:
            m.setattr(temp_storage, "LOCAL_SSD", tmp_path)
            m.setattr(temp_storage, "_is_scratch_mount", lambda path: path == tmp_path)
            assert temp_storage.initialize_temp_storage().base == tmp_path
        # explicit root overrides TMPDIR
        with self._fresh(monkeypatch, tmp_path) as m:
            m.setenv("TMPDIR", "/does-not-exist")
            assert temp_storage.initialize_temp_storage().base == tmp_path
        # sticky shared base supported
        sticky = _private_dir(tmp_path / "sticky")
        sticky.chmod(0o1777)
        with self._fresh(monkeypatch, sticky):
            assert temp_storage.initialize_temp_storage().base == sticky

    def test_invalid_explicit_roots_rejected(self, tmp_path, monkeypatch):
        for value in ("relative", "", "/missing-agentic-test-root", "/tmp\ninjected"):
            with self._fresh(monkeypatch) as m:
                m.setenv("AGENTIC_TEMP_ROOT", value)
                with pytest.raises((OSError, ValueError)):
                    temp_storage.initialize_temp_storage()
        # symlink base rejected
        link = tmp_path / "link"
        link.symlink_to(tmp_path, target_is_directory=True)
        with self._fresh(monkeypatch, link), pytest.raises(ValueError, match="symlink"):
            temp_storage.initialize_temp_storage()
        # public existing namespace rejected
        public = _private_dir(tmp_path / "public")
        (public / f"agentic-scientist-{os.getuid()}").mkdir(mode=0o755)
        with self._fresh(monkeypatch, public), pytest.raises(ValueError, match="private"):
            temp_storage.initialize_temp_storage()
        # unwritable explicit setting rejected
        with self._fresh(monkeypatch, tmp_path) as m:
            m.setattr(os, "access", lambda *args: False)
            with pytest.raises(ValueError, match="writable"):
                temp_storage.initialize_temp_storage()
        # foreign namespace owner rejected
        real_fstat = os.fstat

        def foreign_owner(descriptor):
            metadata = real_fstat(descriptor)
            return SimpleNamespace(st_uid=metadata.st_uid + 1, st_mode=metadata.st_mode)

        with self._fresh(monkeypatch, tmp_path) as m:
            m.setattr(os, "fstat", foreign_owner)
            with pytest.raises(ValueError, match="owned"):
                temp_storage.initialize_temp_storage()
        # non-sticky shared ancestor rejected
        parent = tmp_path / "shared"
        parent.mkdir(mode=0o777)
        parent.chmod(0o777)
        base = _private_dir(parent / "scratch")
        with self._fresh(monkeypatch, base), pytest.raises(ValueError, match="unsafe writable"):
            temp_storage.initialize_temp_storage()

    def test_inherited_state_mismatches_rejected(self, tmp_path, monkeypatch):
        # inherited unrelated directory rejected
        with self._fresh(monkeypatch, tmp_path) as m:
            temp_storage.initialize_temp_storage()
            m.setenv("AGENTIC_TEMP_DIR", str(tmp_path))
            with pytest.raises(ValueError, match="inherited"):
                temp_storage.initialize_temp_storage()
        # partial inherited state rejected
        with self._fresh(monkeypatch) as m:
            m.setenv("AGENTIC_TEMP_DIR", str(tmp_path))
            with pytest.raises(ValueError, match="inherited"):
                temp_storage.initialize_temp_storage()
        # inherited directory permission change rejected
        with self._fresh(monkeypatch, tmp_path):
            temp_storage.initialize_temp_storage().temporary.chmod(0o755)
            with pytest.raises(ValueError, match="private"):
                temp_storage.initialize_temp_storage()

    def test_execution_environment_inherits_initialized_scratch_or_keeps_legacy_paths(
        self, tmp_path, monkeypatch
    ):
        # legacy direct helpers keep active run paths
        assert temp_storage.temporary_directory() == Path("/tmp")
        assert temp_storage.runtime_cache_directory() == Path("/tmp")
        # legacy execution environment does not initialize scratch
        with self._fresh(monkeypatch) as m:
            m.setenv("TMPDIR", "/legacy-configured-temp")
            env = container_execution_env(containerize=True, container_session_name=None, container_mount_root=None)
            assert env["TMPDIR"] == "/legacy-configured-temp"
            assert "AGENTIC_TEMP_DIR" not in env
            assert "AGENTIC_CACHE_ROOT" not in env
        # direct execution environment inherits initialized scratch
        for containerize in (True, False):
            root = _private_dir(tmp_path / f"containerize-{containerize}")
            with self._fresh(monkeypatch, root):
                env = container_execution_env(
                    containerize=containerize, container_session_name=None, container_mount_root=None
                )
                assert env["TMPDIR"] == str(temp_storage.temporary_directory()), containerize
                assert env["AGENTIC_TEMP_DIR"] == env["TMPDIR"], containerize
                assert env["AGENTIC_CACHE_ROOT"] == str(temp_storage.runtime_cache_directory()), containerize
                assert Path(env["AGENTIC_CONTAINER_TEMP_ROOT"]).is_relative_to(root), containerize
                assert not Path(env["AGENTIC_CONTAINER_TEMP_ROOT"]).is_relative_to(env["TMPDIR"]), containerize

    @pytest.mark.parametrize("root", ["missing", "relative"], ids=["init-failure", "unsafe-configuration"])
    def test_logger_storage_failure_still_drains_and_saves_shared_log(self, tmp_path, monkeypatch, root):
        monkeypatch.setenv("AGENTIC_TEMP_ROOT", str(tmp_path / "missing") if root == "missing" else root)
        stream = input_stream(monkeypatch, b"output" * 50000)
        target = tmp_path / "agent.log"
        assert bounded_log.retain_bounded_stream(target, 2048) == 4
        assert stream.tell() == 300000
        if root == "missing":
            assert 0 < target.stat().st_size <= 2048

    def test_cli_returns_four_paths_or_rejects_invalid_base_without_path_output(self, tmp_path, monkeypatch):
        # CLI returns four paths and storage diagnostic
        monkeypatch.setenv("AGENTIC_TEMP_ROOT", str(tmp_path))
        result = subprocess.run(
            [sys.executable, str(Path(temp_storage.__file__))],
            capture_output=True, text=True, check=True, env=os.environ,
        )
        lines = result.stdout.splitlines()
        assert len(lines) == 4
        assert lines[0] == str(tmp_path)
        assert all(Path(line).is_dir() for line in lines)
        assert "free_bytes=" in result.stderr
        # CLI rejects invalid base without path output
        monkeypatch.setenv("AGENTIC_TEMP_ROOT", "/does-not-exist")
        result = subprocess.run(
            [sys.executable, str(Path(temp_storage.__file__))], capture_output=True, text=True, env=os.environ
        )
        assert result.returncode == 1
        assert result.stdout == ""
        assert "initialization failed" in result.stderr


# --- container_temp (python) ----------------------------------------------------------------


def test_private_leaves_are_isolated_verified_and_cleaned_only_when_container_absent(scratch, monkeypatch):
    # private leaves are isolated
    first = container_temp.create(scratch)
    second = container_temp.create(scratch)
    assert first != second
    assert first.parent == second.parent == scratch
    assert first.stat().st_mode & 0o777 == 0o700
    # existing session requires matching private mount
    assert container_temp.verify(scratch, "session", _inspect(first)) == first
    for bad in [{}, _inspect(scratch), _inspect(first) | {"Name": "/other"}]:
        with pytest.raises(ValueError):
            container_temp.verify(scratch, "session", bad)
    # cleanup requires confirmed container absence
    (first / "output").write_text("temporary")
    monkeypatch.setattr(container_temp, "inspect_container", lambda _: _inspect(first))
    assert not container_temp.cleanup(scratch, "session", first)
    assert first.exists()
    monkeypatch.setattr(container_temp, "inspect_container", lambda _: None)
    assert container_temp.cleanup(scratch, "session", first)
    assert not first.exists()
    # cleanup does not follow agent symlinks
    recovery = scratch.parent / "private-recovery"
    recovery.mkdir()
    secret = recovery / "diagnostic"
    secret.write_text("preserve")
    (second / "link").symlink_to(recovery, target_is_directory=True)
    assert container_temp.cleanup(scratch, "session", second)
    assert secret.read_text() == "preserve"
    # private root under sticky shared ancestor is supported
    scratch.parent.chmod(0o1777)
    assert container_temp.create(scratch).is_dir()


def test_invalid_roots_and_uncertain_docker_failures_rejected(scratch, monkeypatch):
    leaf = container_temp.create(scratch)
    # uncertain docker failure cannot authorize cleanup
    for stderr in ("daemon unavailable", "permission denied"):
        with monkeypatch.context() as m:
            m.setattr(
                container_temp.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", stderr)
            )
            with pytest.raises(ValueError, match="Docker inspect failed"):
                container_temp.cleanup(scratch, "session", leaf)
        assert leaf.exists(), stderr
    # invalid roots rejected: symlink, comma, missing
    link = scratch.parent / "link"
    link.symlink_to(scratch, target_is_directory=True)
    comma = _private_dir(scratch.parent / "bad,root")
    for root in (link, comma, scratch / "missing"):
        with pytest.raises((ValueError, OSError)):
            container_temp.create(root)
    # private root under foreign-owned ancestor is rejected
    real_lstat = Path.lstat

    def foreign_parent(path: Path):
        metadata = real_lstat(path)
        if path == scratch.parent:
            return SimpleNamespace(st_uid=os.getuid() + 1, st_mode=metadata.st_mode)
        return metadata

    with monkeypatch.context() as m:
        m.setattr(Path, "lstat", foreign_parent)
        with pytest.raises(ValueError, match="ancestor"):
            container_temp.create(scratch)
    # public root rejected
    scratch.chmod(0o755)
    with pytest.raises((ValueError, OSError)):
        container_temp.create(scratch)
    scratch.chmod(0o700)
    # private root under non-sticky writable ancestor is rejected
    scratch.parent.chmod(0o777)
    with pytest.raises(ValueError, match="ancestor"):
        container_temp.create(scratch)


# --- container_temp (container_agent.sh) ----------------------------------------------------


def test_fresh_and_persistent_containers_override_host_temp_with_private_leaf(scratch: Path) -> None:
    # fresh container overrides host temp and preserves socket mount
    result = _run_shell(scratch, '''
wait_for_docker_daemon() { return 0; }
run_with_bounded_capture() { shift 2; printf '%s\\n' "$@"; }
should_mount_docker_sock() { return 1; }
export TMPDIR=/tmp AGENT_CONTAINER_PASSTHROUGH_KEYS='TMPDIR TMP TEMP'
export TMP=/host/private TEMP=/host/private
export AGENT_CONTAINER_EXTRA_MOUNTS='/tmp/socket:/tmp/socket'
AGENT_INNER_CMD=(true)
_dispatch_inner_command codex /workspace 1 image host /home/agent '' /workspace ''
''')
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    mount = next(arg for arg in args if arg.startswith("type=bind,src="))
    assert mount.startswith(f"type=bind,src={scratch}/agent-tmp-")
    assert mount.endswith(",dst=/agent-tmp")
    for key in ("TMPDIR", "TMP", "TEMP"):
        assert f"{key}=/agent-tmp" in args
        assert args.index(f"{key}=/agent-tmp") > args.index(key)
    assert "/tmp/socket:/tmp/socket" in args
    # persistent container exec overrides passthrough
    result = _run_shell(scratch, '''
ensure_agent_session_container() { return 0; }
docker() { printf '%s\\n' "$@"; }
export TMPDIR=/tmp TMP=/host/private TEMP=/host/private
export AGENT_CONTAINER_PASSTHROUGH_KEYS='TMPDIR TMP TEMP'
AGENT_INNER_CMD=(true)
_dispatch_inner_command codex /workspace 1 image host /home/agent '' /workspace session
''')
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    for key in ("TMPDIR", "TMP", "TEMP"):
        assert f"{key}=/agent-tmp" in args
        assert args.index(f"{key}=/agent-tmp") > args.index(key)
    # session start and reuse share exact mount
    result = _run_shell(scratch, '''
container_is_running() { return 1; }
container_exists() { return 1; }
wait_for_docker_daemon() { return 0; }
should_mount_docker_sock() { return 1; }
run_with_bounded_capture() { shift 2; printf '%s\\n' "$@"; }
export TMPDIR=/tmp AGENT_CONTAINER_PASSTHROUGH_KEYS=TMPDIR
ensure_agent_session_container codex session image host /workspace /home/agent ''
''')
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    mount = next(arg for arg in args if arg.startswith("type=bind,src="))
    leaf = Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/agent-tmp"))
    assert leaf.parent == scratch
    assert leaf.exists()
    assert f"{container_temp.LABEL}={leaf}" in args
    assert args.index("TMPDIR=/agent-tmp") > args.index("TMPDIR")
    assert container_temp.verify(scratch, "session", _inspect(leaf)) == leaf


def test_legacy_sessions_and_fresh_scratch_cleanup_follow_container_state(scratch, monkeypatch):
    # unconfigured running session keeps legacy behavior
    result = _run_shell(scratch, '''
unset AGENTIC_CONTAINER_TEMP_ROOT
container_is_running() { return 0; }
docker() { echo 'unexpected Docker call' >&2; return 1; }
ensure_agent_session_container codex session image host /workspace /home/agent ''
''')
    assert result.returncode == 0, result.stderr
    assert list(scratch.iterdir()) == []
    # running legacy session is rejected
    legacy_bin = scratch.parent / "legacy-bin"
    _fake_docker(legacy_bin, "#!/bin/sh\nprintf '%s\\n' '[{\"Name\":\"/session\"}]'\n")
    with monkeypatch.context() as m:
        m.setenv("PATH", f"{legacy_bin}:{os.environ['PATH']}")
        result = _run_shell(scratch, '''
container_is_running() { return 0; }
ensure_agent_session_container codex session image host /workspace /home/agent ''
''')
    assert result.returncode != 0
    assert "scratch" in result.stderr
    # fresh scratch cleanup only after successful container removal
    inspection_log = scratch.parent / "inspection.log"
    fake_bin = scratch.parent / "bin"
    _fake_docker(
        fake_bin,
        "#!/bin/sh\n"
        f"printf '%s\\n' inspect >> {shlex.quote(str(inspection_log))}\n"
        "echo 'Error: No such container: removed' >&2\nexit 1\n",
    )
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    for run_status in (0, 7):
        root = _private_dir(scratch / f"status-{run_status}")
        inspection_log.unlink(missing_ok=True)
        result = _run_shell(root, f'''
wait_for_docker_daemon() {{ return 0; }}
run_with_bounded_capture() {{ return {run_status}; }}
should_mount_docker_sock() {{ return 1; }}
AGENT_INNER_CMD=(true)
_dispatch_inner_command codex /workspace 1 image host /home/agent '' /workspace ''
''')
        assert result.returncode == run_status, result.stderr
        assert bool(list(root.iterdir())) == bool(run_status), run_status
        assert inspection_log.exists() == (run_status == 0), run_status


@pytest.mark.parametrize("teardown", ["complete", "failed", "stop_failed"])
def test_persistent_scratch_removed_only_for_completed_stopped_session(scratch, monkeypatch, teardown):
    leaf = container_temp.create(scratch)
    (leaf / "cache").write_text("agent temporary data")
    state = scratch.parent / "docker-state.json"
    state.write_text(json.dumps([_inspect(leaf)]))
    log = scratch.parent / "docker-calls.log"
    fake_bin = scratch.parent / "bin"
    _fake_docker(fake_bin, _STATEFUL_DOCKER)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("DOCKER_STATE", str(state))
    monkeypatch.setenv("DOCKER_CALLS", str(log))
    monkeypatch.setenv("DOCKER_STOP_FAIL", "1" if teardown == "stop_failed" else "0")
    monkeypatch.setenv("AGENT_DOCKER_STOP_RETRIES", "1")
    function = "stop_container_session_force" if teardown == "failed" else "complete_container_session"
    result = _run_shell(scratch, f"{function} session\n")
    assert result.returncode == (1 if teardown == "stop_failed" else 0), result.stderr
    assert leaf.exists() == (teardown != "complete")
    calls = log.read_text().splitlines()
    if teardown == "complete":
        assert calls[0] == "inspect"
        assert calls[-2:] == ["rm", "inspect"]


@pytest.mark.parametrize("script", ["task.sh", "run.sh"])
def test_drafting_and_attempt_private_temps_use_configured_storage(scratch: Path, script: str) -> None:
    source = _script(script)
    templates = {
        "task.sh": ["agentic-task-provider-env", "agentic-task-validation", "agentic-task-recovery"],
        "run.sh": ["agentic-task-search", "agentic-run-provider-env"],
    }
    captured_root = "TASK_TEMP_ROOT" if script == "task.sh" else "SCRIPT_TEMP_ROOT"
    initialization = re.search(rf"^{captured_root}=.*$", source, re.MULTILINE)
    assert initialization
    for prefix in templates[script]:
        match = re.search(r"mktemp(?: -d)? ([^\n]*" + prefix + r"[^\n]*?)\)", source)
        assert match, prefix
        command = "mktemp" + (" -d" if "mktemp -d" in match[0] else "") + " " + match[1]
        result = _bash(
            initialization[0] + "\n" + command, env=os.environ | {"TMPDIR": str(scratch)}, check=True, timeout=None
        )
        assert Path(result.stdout.strip()).parent == scratch


@pytest.mark.parametrize("agent_status", [0, 7, 130])
def test_attempt_status_handoff_preserves_reusable_session_for_caller(scratch, agent_status):
    source = _script("run.sh")
    start = source.index("cleanup_search_only_task_source\ncleanup_contracted_container_env\n")
    end = source.index('if [ "$HARNESS_EVALUATED_ATTEMPT" -eq 1 ]; then', start)
    status_handoff = source[start:end]
    cleanup_marker = scratch / "unexpected-session-stop"
    result = _run_shell(scratch, f'''
cleanup_run_processes() {{ touch {shlex.quote(str(cleanup_marker))}; }}
cleanup_search_only_task_source() {{ :; }}
cleanup_contracted_container_env() {{ :; }}
trap cleanup_run_processes EXIT
AGENT_EXIT_CODE={agent_status}
INTERRUPTED={int(agent_status == 130)}
TEE_EXIT_CODE=0
HARNESS_EVALUATED_ATTEMPT=0
{status_handoff}
''')
    assert result.returncode == agent_status, result.stderr
    assert not cleanup_marker.exists()


@pytest.mark.parametrize("script", ["paper.sh", "paper_edit.sh"])
@pytest.mark.parametrize("outcome", ["success", "failed", "stop_failed"])
@pytest.mark.parametrize("phase", ["figure", "finalization"])
def test_paper_teardown_preserves_only_failed_scratch(tmp_path, script, outcome, phase):
    root = _private_dir(tmp_path / "scratch")
    leaf = container_temp.create(root)
    (leaf / "build-output").write_text("temporary figure build")
    state = tmp_path / "state.json"
    state.write_text(json.dumps([_inspect(leaf)]))
    fake_bin = tmp_path / "bin"
    _fake_docker(fake_bin, _STATEFUL_DOCKER)
    source = _script(script)
    if phase == "figure":
        branches = re.findall(
            r"if ! paper_figure_quality_\w+_agent_session \\\n.*?FIGURE_QUALITY_REPAIR_AGENT_QUIESCED=1",
            source, re.DOTALL,
        )
        marker = "before rollback" if outcome == "failed" else "before promotion"
        branch = next(branch for branch in branches if marker in branch)
    else:
        branch = source[source.rindex("paper_gate_complete "):]
    command = "\n".join([
        "set -euo pipefail",
        f"source {shlex.quote(str(HELPER))}",
        f"source {shlex.quote(str(SCRIPTS / 'lib/paper_figure_quality.sh'))}",
        "CONTAINERIZE=1",
        "AGENT_CONTAINER_SESSION_NAME=session",
        'paper_gate_complete() { return "$FINAL_GATE_STATUS"; }',
        "trap 'stop_container_session_force session || true' EXIT",
        branch,
    ])
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHON_BIN": sys.executable,
        "DOCKER_STATE": str(state),
        "DOCKER_STOP_FAIL": "1" if outcome == "stop_failed" else "0",
        "AGENT_DOCKER_STOP_RETRIES": "1",
        "AGENTIC_CONTAINER_TEMP_ROOT": str(root),
        "FINAL_GATE_STATUS": "1" if outcome == "failed" else "0",
    }
    result = _bash(command, env=env, timeout=None)
    failure = outcome == ("stop_failed" if phase == "figure" else "failed")
    assert result.returncode == (1 if failure else 0), result.stderr
    assert leaf.exists() == (outcome != "success")
    assert state.exists() == (outcome == "stop_failed")


# --- shell cleanup of relocated private temp artifacts --------------------------------------

_TASK_CLEANUP_CASES = (
    ("DESIGN_PRIVATE_ROOT", "agentic-task-design-private-", "directory"),
    ("HARNESS_VALIDATION_ROOT", "agentic-task-validation-", "directory"),
    ("HARNESS_RECOVERY_ROOT", "agentic-task-recovery-", "directory"),
    ("DRAFT_CONTAINER_ENV_FILE", "agentic-task-provider-env-", "file"),
)


@pytest.mark.parametrize("variable,prefix,kind", _TASK_CLEANUP_CASES)
@pytest.mark.parametrize("placement", ["scratch", "outside", "symlink", "public"])
def test_task_cleanup_respects_captured_private_root(tmp_path, variable, prefix, kind, placement):
    scratch = _private_dir(tmp_path / "scratch with spaces")
    target = (tmp_path if placement == "outside" else scratch) / f"{prefix}trusted"
    if placement == "symlink":
        original = _private_dir(tmp_path / "preserved-original")
        (original / "content").write_text("preserve")
        target.symlink_to(original, target_is_directory=True)
    elif kind == "directory":
        target.mkdir(mode=0o700)
        (target / "content").write_text("private")
    else:
        target.write_text("private")
        target.chmod(0o600)
    if placement == "public":
        target.chmod(0o777)
    source = _script("task.sh")
    functions = "\n".join(
        _bash_function(source, name) for name in ("cleanup_design_private_root", "cleanup_run_processes")
    )
    script = functions + '''
PYTHON_BIN="$1" TASK_TEMP_ROOT="$2" REPO_ROOT="$5"
CONTAINERIZE=0 CLEANUP_IN_PROGRESS=0 VALIDATION_PUBLISHED=1
stop_task_log_writer() { :; }
stop_task_repair_gate() { :; }
export TMPDIR="$4"
printf -v "$3" '%s' "$6"
cleanup_run_processes
'''
    result = _bash(
        script,
        args=("bash", sys.executable, str(scratch), variable, str(tmp_path / "changed-environment"),
              str(REPO_ROOT), str(target)),
        env=os.environ,
    )
    assert result.returncode == 0, result.stderr
    assert (target.exists() or target.is_symlink()) == (placement != "scratch")
    if placement == "symlink":
        assert (original / "content").read_text() == "preserve"


_RELOCATED_CLEANUP_CASES = [
    ("run.sh", "cleanup_search_only_task_source", "SEARCH_ONLY_TASK_SOURCE_DIR", "agentic-task-search-test", "directory"),
    ("run.sh", "cleanup_contracted_container_env", "CONTRACTED_CONTAINER_ENV_FILE", "agentic-run-provider-env-test", "file"),
    ("idea.sh", "cleanup_idea_security_paths", "CONTRACTED_CONTAINER_ENV_FILE", "agentic-idea-provider-env-test", "file"),
]


@pytest.mark.parametrize("script,function,variable,name,kind", _RELOCATED_CLEANUP_CASES)
@pytest.mark.parametrize("scenario", ["safe", "safe_readonly", "outside", "symlink", "wrong_type"])
def test_relocated_shell_cleanup_validates_and_removes_only_its_own_temporary_artifact(
    tmp_path, script, function, variable, name, kind, scenario
):
    root = _private_dir(tmp_path / "private scratch with spaces")
    path = (tmp_path if scenario == "outside" else root) / name
    if scenario == "symlink":
        target = tmp_path / "preserved-target"
        target.write_text("keep")
        path.symlink_to(target)
    elif (kind == "directory") != (scenario == "wrong_type"):
        path.mkdir(mode=0o700)
        (path / "content").write_text("temporary data")
        if scenario == "safe_readonly":
            path.chmod(0o555)
    else:
        path.write_text("provider configuration")
        path.chmod(0o600)
    command = "\n".join([
        "set -euo pipefail", _bash_function(_script(script), function),
        f"{variable}={shlex.quote(str(path))}", "TMPDIR=/changed-after-creation", function,
    ])
    result = _bash(
        command,
        env=os.environ | {"PYTHON_BIN": sys.executable, "REPO_ROOT": str(REPO_ROOT),
                          "SCRIPT_TEMP_ROOT": str(root), "TMPDIR": str(root)},
        timeout=None,
    )
    if scenario in {"safe", "safe_readonly"}:
        assert result.returncode == 0, result.stderr
        assert not path.exists()
    else:
        assert path.exists() or path.is_symlink()
    if scenario == "symlink":
        assert target.read_text() == "keep"


def test_cleanup_validators_accept_readonly_shadow_and_reject_missing_wrong_type_and_symlink_parent(
    tmp_path: Path,
) -> None:
    # shared directory validator accepts read-only search shadow
    root = tmp_path / "agentic-task-search-readonly"
    root.mkdir(mode=0o555)
    assert validate_temp_directory(root, tmp_path, "agentic-task-search-")
    # cleanup validator handles missing artifact
    for validate in (validate_temp_file, validate_temp_directory):
        assert not validate(tmp_path / "agentic-absent", tmp_path, "agentic-"), validate.__name__
    # cleanup validator rejects wrong type
    directory = tmp_path / "agentic-artifact-dir"
    directory.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="must be"):
        validate_temp_file(directory, tmp_path, "agentic-")
    file = tmp_path / "agentic-artifact-file"
    file.write_text("private")
    file.chmod(0o600)
    with pytest.raises(ValueError, match="must be"):
        validate_temp_directory(file, tmp_path, "agentic-")
    # cleanup validator rejects symlink parent
    link = tmp_path / "alias"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="normalized"):
        validate_temp_file(link / "agentic-artifact", link, "agentic-")


# --- bounded_log: a lost log sink must not break the producer -------------------------------


@pytest.mark.parametrize("operation", ["open", "write", "flush", "close"])
@pytest.mark.parametrize("number", [errno.ESTALE, errno.EDQUOT, errno.ENOSPC])
def test_failed_shared_sink_retains_bounded_local_log_and_drains(tmp_path, monkeypatch, capsys, operation, number):
    local_spools(tmp_path, monkeypatch)
    shared = tmp_path / "shared.log"
    original = getattr(bounded_log, "_open_log", None)

    class BrokenStream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            if name == operation:
                def fail(*args, **kwargs):
                    if name == "close":
                        self.stream.close()
                    raise OSError(number, "private-path-and-token")
                return fail
            return getattr(self.stream, name)

    def open_log(path):
        if path == shared:
            if operation == "open":
                raise OSError(number, "private-path-and-token")
            return BrokenStream(original(path))
        return original(path)

    monkeypatch.setattr(bounded_log, "_open_log", open_log, raising=False)
    data = b"prefix\n" + b"x" * (2 * 1024 * 1024) + b"\ntail\n"
    stream = input_stream(monkeypatch, data)
    assert bounded_log.retain_bounded_stream(shared, 2048) == 4
    assert stream.tell() == len(data)
    logs = list(tmp_path.glob("spool-*/*.log"))
    assert len(logs) == 1
    retained = logs[0].read_bytes()
    assert len(retained) <= 2048
    assert retained.startswith(b"prefix\n") and retained.endswith(b"\ntail\n")
    assert stat.S_IMODE(logs[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(logs[0].parent.stat().st_mode) == 0o700
    diagnostic = capsys.readouterr().err
    assert "infrastructure_failure" in diagnostic
    assert "private-path-and-token" not in diagnostic
    assert len(diagnostic) < 2048
    notices = [json.loads(line) for line in diagnostic.splitlines()]
    assert notices[-1]["local_spool_status"] == "retained"


def test_successful_shared_sink_removes_own_spool_and_tolerates_symlink_or_passthrough_failure(
    tmp_path, monkeypatch
):
    # success removes only own local spool
    with monkeypatch.context() as m:
        root = tmp_path / "success"
        root.mkdir()
        local_spools(root, m)
        unrelated = root / "keep.log"
        unrelated.write_text("keep")
        shared = root / "log"
        input_stream(m, b"complete")
        assert bounded_log.retain_bounded_stream(shared, 2048) == 0
        assert shared.read_bytes() == b"complete"
        assert not list(root.glob("spool-*"))
        assert unrelated.read_text() == "keep"
    # shared symlink is not overwritten
    with monkeypatch.context() as m:
        root = tmp_path / "symlink"
        root.mkdir()
        local_spools(root, m)
        target = root / "target"
        target.write_text("keep")
        link = root / "log"
        link.symlink_to(target)
        stream = input_stream(m, b"complete")
        assert bounded_log.retain_bounded_stream(link, 2048) == 4
        assert target.read_text() == "keep" and stream.tell() == 8
    # passthrough failure still drains to bounded logs
    with monkeypatch.context() as m:
        root = tmp_path / "passthrough"
        root.mkdir()
        local_spools(root, m)

        class BrokenOutput:
            def write(self, data):
                raise BrokenPipeError(errno.EPIPE, "closed")

        m.setattr(sys, "stdout", SimpleNamespace(buffer=BrokenOutput()))
        data = b"prefix" + b"x" * (2 * 1024 * 1024) + b"tail"
        stream = input_stream(m, data)
        shared = root / "log"
        assert bounded_log.retain_bounded_stream(shared, 2048, passthrough=True) == 4
        assert stream.tell() == len(data)
        assert shared.read_bytes().endswith(b"tail")
        assert shared.stat().st_size <= 2048


def test_local_spool_failures_are_reported_without_private_details(tmp_path, monkeypatch, capsys):
    # both log sinks failing still drains input
    with monkeypatch.context() as m:
        root = tmp_path / "both"
        root.mkdir()
        local_spools(root, m)

        def fail(path):
            raise OSError(errno.ENOSPC, "private")

        m.setattr(bounded_log, "_open_log", fail, raising=False)
        data = b"x" * (2 * 1024 * 1024)
        stream = input_stream(m, data)
        assert bounded_log.retain_bounded_stream(root / "log", 2048) == 4
        assert stream.tell() == len(data)
        diagnostic = capsys.readouterr().err
        assert "private" not in diagnostic
        notices = [json.loads(line) for line in diagnostic.splitlines()]
        assert notices[-1]["local_spool_status"] == "unavailable"
        assert all("local_spool" not in notice for notice in notices)
    # fallback path reported before interrupted input
    with monkeypatch.context() as m:
        root = tmp_path / "interrupted"
        root.mkdir()
        local_spools(root, m)

        class InterruptedInput:
            def read(self, size):
                notices = _notices(capsys)
                assert notices[-1]["local_spool_status"] == "in_progress"
                assert Path(notices[-1]["local_spool"]).is_file()
                raise KeyboardInterrupt

        m.setattr(sys, "stdin", SimpleNamespace(buffer=InterruptedInput()))
        with pytest.raises(KeyboardInterrupt):
            bounded_log.retain_bounded_stream(root / "missing-parent" / "log", 2048)
    capsys.readouterr()
    # local write failure reports partial log
    with monkeypatch.context() as m:
        root = tmp_path / "partial"
        root.mkdir()
        local_spools(root, m)
        original = bounded_log._LogSink.write

        def write(sink, *args):
            if sink.name == "local":
                sink._fail("write", OSError(errno.ENOSPC, "private"))
            else:
                original(sink, *args)

        m.setattr(bounded_log._LogSink, "write", write)
        input_stream(m, b"complete")
        assert bounded_log.retain_bounded_stream(root / "log", 2048) == 4
        notices = _notices(capsys)
        assert notices[-1]["local_spool_status"] == "partial"
        assert Path(notices[-1]["local_spool"]).is_file()


def test_subprocess_producer_finishes_when_shared_log_cannot_open(tmp_path):
    producer = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 8388608 + b'tail')"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    command = f'''
import tempfile
from pathlib import Path
from src.scripts.lib import bounded_log
original = tempfile.mkdtemp
tempfile.mkdtemp = lambda **kw: original(prefix="spool-", dir={str(tmp_path)!r})
raise SystemExit(bounded_log.retain_bounded_stream(Path({str(tmp_path / 'missing/log')!r}), 2048))
'''
    logger = subprocess.Popen(
        [sys.executable, "-c", command], stdin=producer.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    producer.stdout.close()
    try:
        _, diagnostic = logger.communicate(timeout=10)
        assert producer.wait(timeout=5) == 0, producer.stderr.read()
        assert logger.returncode == 4, diagnostic
        log, = tmp_path.glob("spool-*/*.log")
        assert log.read_bytes().endswith(b"tail")
        assert log.stat().st_size <= 2048
    finally:
        for process in (producer, logger):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        producer.stderr.close()


def test_task_logger_failure_stops_retries_scaffold_acceptance_and_never_waits_for_stuck_writer():
    source = _script("task.sh")
    # logger failure suppresses provider retry even when agent failed
    command = f'''set -euo pipefail
PYTHON_BIN=true
TASK_AGENT_TRANSIENT_MAX_ATTEMPTS=3
AGENT_TRANSIENT_MAX_ATTEMPTS=3
LOG_FILE=/unused
calls=0
run_logged_agent_step_once() {{
  calls=$((calls + 1))
  printf 'called\n'
  LAST_AGENT_PROCESS_EXIT=17
  LAST_AGENT_LOGGER_EXIT=4
  return 4
}}
agent_transient_retry_delay_seconds() {{ printf 0; }}
{_bash_function(source, "run_logged_agent_step")}
rc=0
run_logged_agent_step label prompt file || rc=$?
printf 'continued'
'''
    result = _bash(command, timeout=5)
    assert result.returncode == 4, result.stderr
    assert result.stdout == "called\n"
    # scaffold cannot accept retained artifacts after logger failure
    functions = "\n".join(
        _bash_function(source, name) for name in ("run_logged_agent_step", "draft_scaffold_until_complete")
    )
    command = f'''set -euo pipefail
PROMPT=prompt DRAFT_MAX_ATTEMPTS=3 OUTPUT_DIR=/unused CONTAINER_HOME_DIR=/unused
STAGED_TASK_DIR=/unused
task_progress_emit() {{ :; }}
inspect_staged_scaffold() {{ echo unexpected-inspection >&2; return 0; }}
run_logged_agent_step_once() {{
  echo called
  LAST_AGENT_PROCESS_EXIT=17 LAST_AGENT_LOGGER_EXIT=4
  return 4
}}
{functions}
draft_scaffold_until_complete
echo continued
'''
    result = _bash(command, timeout=5)
    assert result.returncode == 4, result.stderr
    assert result.stdout == "called\n"
    assert "unexpected-inspection" not in result.stderr
    # task logger cleanup never waits for stuck writer
    command = f'''set -euo pipefail
LOG_WRITER_PID=12345
kill() {{ echo "kill $*"; return 0; }}
sleep() {{ :; }}
wait() {{ echo unexpected-blocking-wait; }}
{_bash_function(source, "stop_task_log_writer")}
stop_task_log_writer
echo "writer=$LOG_WRITER_PID"
'''
    result = _bash(command, timeout=5)
    assert result.returncode == 0, result.stderr
    assert "kill -TERM 12345" in result.stdout
    assert "kill -KILL 12345" in result.stdout
    assert "unexpected-blocking-wait" not in result.stdout
    assert result.stdout.endswith("writer=\n")


@pytest.mark.parametrize("script_name", ["task.sh", "run.sh"])
def test_child_cancellation_takes_precedence_over_logger_failure(script_name):
    source = _script(script_name)
    if script_name == "task.sh":
        body = source.split("run_logged_agent_step_once() {", 1)[1].split("\n}\n", 1)[0]
        body = body.split('"$(headless_cost_status "$agent_exit_code" "$INTERRUPTED")"', 1)[1]
        command = f"INTERRUPTED=0 agent_exit_code=130 writer_exit_code=4\nstep() {{{body}\n}}\nstep\n"
    else:
        body = source.split("cleanup_contracted_container_env\ntrap - INT TERM EXIT\n", 1)[1]
        body = body.split('if [ "$AGENT_EXIT_CODE" -ne 0 ]', 1)[0]
        command = f"INTERRUPTED=0 AGENT_EXIT_CODE=130 TEE_EXIT_CODE=4\n{body}"
    result = _bash(command, timeout=5)
    assert result.returncode == 130, result.stderr


# --- unmanaged evaluation artifact cleanup --------------------------------------------------


def _make_evaluator_runs(root: Path) -> tuple[Path, Path]:
    runs_dir = root / "runs"
    run_dir = runs_dir / "run_000_case_seed_0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_summary.json").write_text('{"outcome":"ok"}', encoding="utf-8")
    (run_dir / "program_stdout.txt").write_text("output", encoding="utf-8")
    return runs_dir, run_dir


def test_cleanup_removes_recognized_unmanaged_run_bundles_and_keeps_unrecognized_siblings(tmp_path: Path) -> None:
    # cleanup removes recognized unmanaged run bundles
    experiment_dir = tmp_path / "idea"
    scratch_dir = experiment_dir / ".dev_eval_program_1"
    scratch_runs, scratch_run = _make_evaluator_runs(scratch_dir)
    (scratch_dir / "metrics.json").write_text("{}", encoding="utf-8")
    program_dir = experiment_dir / "program_1"
    program_runs, program_run = _make_evaluator_runs(program_dir)
    source = program_dir / "src" / "program.py"
    source.parent.mkdir()
    source.write_text("print('candidate')", encoding="utf-8")

    report = cleanup_unmanaged_evaluation_outputs(experiment_dir)

    assert report.removed_roots == (scratch_run,)
    assert report.removed_bytes > 0
    assert not scratch_runs.exists()
    assert program_runs.is_dir()
    assert program_run.is_dir()
    assert (scratch_dir / "metrics.json").is_file()
    assert source.is_file()

    # cleanup preserves unrecognized siblings in scratch runs
    experiment_dir = tmp_path / "siblings"
    runs_dir, evaluator_run = _make_evaluator_runs(experiment_dir / ".dev_eval_program_1")
    unrelated_run = runs_dir / "training_checkpoint"
    unrelated_run.mkdir()
    (unrelated_run / "weights.bin").write_bytes(b"keep")

    report = cleanup_unmanaged_evaluation_outputs(experiment_dir)

    assert report.removed_roots == (evaluator_run,)
    assert not evaluator_run.exists()
    assert (unrelated_run / "weights.bin").read_bytes() == b"keep"


def test_cleanup_preserves_managed_symlinked_undermarked_and_swapped_runs(tmp_path, monkeypatch):
    # cleanup preserves managed and unrecognized runs
    experiment_dir = tmp_path / "managed"
    managed_runs, _ = _make_evaluator_runs(experiment_dir / ".evaluation_outputs" / "program_1" / "attempt_a")
    unrelated_runs, _ = _make_evaluator_runs(experiment_dir / "analysis")

    report = cleanup_unmanaged_evaluation_outputs(experiment_dir)

    assert report.removed_roots == ()
    assert report.removed_bytes == 0
    assert managed_runs.is_dir()
    assert unrelated_runs.is_dir()

    # cleanup does not follow symlinked run directories
    experiment_dir = tmp_path / "symlinked"
    experiment_dir.mkdir()
    external_runs, _ = _make_evaluator_runs(tmp_path / "external")
    symlinked_runs = experiment_dir / ".dev_eval_program_1" / "runs"
    symlinked_runs.parent.mkdir()
    symlinked_runs.symlink_to(external_runs, target_is_directory=True)

    report = cleanup_unmanaged_evaluation_outputs(experiment_dir)

    assert report.removed_roots == ()
    assert symlinked_runs.is_symlink()
    assert external_runs.is_dir()

    # cleanup requires multiple evaluator markers
    experiment_dir = tmp_path / "undermarked"
    runs_dir = experiment_dir / "program_1" / "runs"
    run_dir = runs_dir / "run_000"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")

    report = cleanup_unmanaged_evaluation_outputs(experiment_dir)

    assert report.removed_roots == ()
    assert runs_dir.is_dir()

    # cleanup preserves directory swapped after validation
    experiment_dir = tmp_path / "swapped"
    runs_dir, evaluator_run = _make_evaluator_runs(experiment_dir / ".dev_eval_program_1")
    saved_run = runs_dir / "saved_verified_run"
    real_rename = os.rename
    swapped = False

    def swap_before_rename(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal swapped
        if not swapped and src == evaluator_run.name:
            swapped = True
            real_rename(src, saved_run.name, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
            replacement = runs_dir / evaluator_run.name
            replacement.mkdir()
            (replacement / "config.json").write_text("replacement", encoding="utf-8")
            (replacement / "run_summary.json").write_text("replacement", encoding="utf-8")
        return real_rename(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(cleanup_module.os, "rename", swap_before_rename)

    report = cleanup_unmanaged_evaluation_outputs(experiment_dir)

    assert report.removed_roots == ()
    assert len(report.failures) == 1
    assert (evaluator_run / "config.json").read_text(encoding="utf-8") == "replacement"
    assert saved_run.is_dir()
