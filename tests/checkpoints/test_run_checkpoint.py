"""write_checkpoint / checkpoint_git / checkpoint_paths: core, security, size limits, root safety."""

import os
import subprocess
import tempfile
import zlib
from pathlib import Path

import pytest

import src.investigation.checkpoint_git as checkpoint_git_module
import src.investigation.checkpoint_paths as checkpoint_paths_module
import src.investigation.run_checkpoint as run_checkpoint_module
from src.investigation.checkpoint_git import (
    _quarantine_identity,
    _resolve_git_executable,
    base_argv,
    git_environment,
)
from src.investigation.checkpoint_paths import descriptor_aliases, pinned_checkpoint_paths
from src.investigation.run_checkpoint import (
    CheckpointResult,
    CheckpointSettings,
    checkpoint_git_dir,
    describe_checkpoint,
    write_checkpoint,
)

TOKEN = "sk-proj-abcdefghijklmnopqrstuvwxyz012345"
SECRET_LINE = f"OPENAI_API_KEY={TOKEN}\n"


def _settings(**overrides) -> CheckpointSettings:
    base = dict(git_dir_name=".agentic_history", timeout_seconds=60)
    return CheckpointSettings(**{**base, **overrides})


def _run_dir(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    (run_root / "ideas" / "01" / "program_0" / "src").mkdir(parents=True)
    (run_root / "ideas" / "01" / "idea_bundle.json").write_text("{}\n", encoding="utf-8")
    (run_root / "outer_state.json").write_text('{"current_cycle": 1}\n', encoding="utf-8")
    return run_root


def _bare_run(tmp_path: Path, *files: str) -> Path:
    """A run dir holding only `metadata.txt` plus any extra small files."""
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    for name in files:
        (run_root / name).write_text("ok\n", encoding="utf-8")
    return run_root


def _git(run_root: Path, *args: str, git_dir: Path | None = None) -> str:
    git_dir = git_dir or checkpoint_git_dir(run_root, ".agentic_history")
    proc = subprocess.run(
        ["git", f"--git-dir={git_dir}", f"--work-tree={run_root}", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(run_root),
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _tracked(run_root: Path, git_dir: Path | None = None) -> list[str]:
    output = _git(run_root, "ls-tree", "-r", "--name-only", "HEAD", git_dir=git_dir)
    return sorted(path for path in output.splitlines() if path)


def _head(run_root: Path) -> str:
    return _git(run_root, "rev-parse", "HEAD").strip()


def _checkpoint(
    run_root: Path,
    *,
    label: str | None = None,
    phase: str = "investigation",
    settings: CheckpointSettings | None = None,
    cycle: int | None = None,
    idea_id: str | None = None,
    event: str | None = None,
    final: bool = False,
    pin_run_root: bool = False,
) -> CheckpointResult:
    return write_checkpoint(
        run_root=run_root,
        label=label if label is not None else ("final" if final else "cycle 1"),
        phase=phase,
        settings=settings if settings is not None else _settings(),
        cycle=cycle,
        idea_id=idea_id,
        event=event,
        final=final,
        pin_run_root=pin_run_root,
    )


# --------------------------------------------------------------------------- core behaviour


def test_exclusion_warnings_are_marked_stored_only_when_the_checkpoint_succeeds(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    # exclusion messages are marked as stored checkpoints
    with caplog.at_level("WARNING"):
        run_checkpoint_module._warn_exclusions(
            label="paper phase",
            sensitive=(".agent_home",),
            oversize=("large.bin",),
            max_file_bytes=100,
        )

    assert len(caplog.records) == 2
    assert all(getattr(record, "checkpoint_stored", False) for record in caplog.records)

    # a failed checkpoint does not claim exclusions were stored
    caplog.clear()
    run_root = _run_dir(tmp_path)
    (run_root / "agent.log").write_text(SECRET_LINE, encoding="utf-8")
    monkeypatch.setattr(
        run_checkpoint_module,
        "_create_quarantined_commit",
        lambda *_args, **_kwargs: (None, "commit failed"),
    )

    with caplog.at_level("WARNING"):
        result = _checkpoint(run_root)

    assert result.status == "degraded"
    assert not any(getattr(record, "checkpoint_stored", False) for record in caplog.records)


def test_sidecar_lives_outside_absolute_relative_and_symlinked_work_trees(
    tmp_path: Path, monkeypatch
) -> None:
    # repository metadata lives outside the work tree
    run_root = _run_dir(tmp_path / "absolute")

    result = _checkpoint(run_root)

    assert result.status == "committed"
    assert not (run_root / ".git").exists()
    assert not (run_root / ".agentic_history").exists()
    git_dir = checkpoint_git_dir(run_root, ".agentic_history")
    assert git_dir.parent == run_root.parent
    assert not git_dir.is_relative_to(run_root)
    assert "worktree" not in (git_dir / "config").read_text(encoding="utf-8")

    # a symlinked run root keeps the sidecar outside the resolved tree
    resolved_root = _run_dir(tmp_path / "tree")
    linked_root = tmp_path / "linked-run"
    linked_root.symlink_to(resolved_root, target_is_directory=True)

    result = _checkpoint(linked_root)

    git_dir = checkpoint_git_dir(resolved_root, ".agentic_history")
    assert result.status == "committed"
    assert not git_dir.is_relative_to(resolved_root)
    assert (git_dir / "HEAD").is_file()

    # a relative run root uses absolute git paths
    relative_base = tmp_path / "relative"
    relative_base.mkdir()
    monkeypatch.chdir(relative_base)
    run_root = _run_dir(Path("."))

    result = _checkpoint(run_root)

    assert result.status == "committed"
    assert (checkpoint_git_dir(run_root.absolute(), ".agentic_history") / "HEAD").is_file()

    # parent segments cannot place the sidecar inside the resolved tree
    segments_base = tmp_path / "segments"
    segments_base.mkdir()
    monkeypatch.chdir(segments_base)
    (segments_base / "child").mkdir()
    (segments_base / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    run_root = Path("child/..")

    result = _checkpoint(run_root)

    git_dir = checkpoint_git_dir(segments_base, ".agentic_history")
    assert result.status == "committed"
    assert not git_dir.is_relative_to(segments_base)
    assert not any("agentic_history" in path for path in _tracked(run_root))


_UNSAFE_GIT_DIR_NAMES = ["../escape", "unsafe name", "line\nbreak", "unsafe*name", "unsafe:name"]


def test_sidecar_resolver_rejects_root_and_unsafe_names_before_creating_anything(
    tmp_path: Path,
) -> None:
    # the resolver rejects the filesystem root
    with pytest.raises(ValueError, match="filesystem root"):
        checkpoint_git_dir(Path("/"), ".agentic_history")

    # the resolver rejects unsafe sidecar names
    run_root = _run_dir(tmp_path)
    for git_dir_name in _UNSAFE_GIT_DIR_NAMES:
        with pytest.raises(ValueError, match="letters, digits, dot, underscore, or hyphen"):
            checkpoint_git_dir(run_root, git_dir_name)

    # a checkpoint degrades before creating an unsafe sidecar, pinned or not
    for pin_run_root in (False, True):
        result = _checkpoint(
            run_root,
            label="unsafe",
            settings=CheckpointSettings(git_dir_name="unsafe*name"),
            pin_run_root=pin_run_root,
        )

        assert result.status == "degraded", pin_run_root
        assert "letters, digits, dot, underscore, or hyphen" in str(result.error), pin_run_root
        assert list(tmp_path.glob(".run*")) == [], pin_run_root


def test_tampered_sidecars_are_rejected_or_ignored(tmp_path: Path) -> None:
    # an agent-created in-tree sidecar cannot suppress or enter history
    run_root = _run_dir(tmp_path / "in-tree")
    _checkpoint(run_root)
    legacy = run_root / ".agentic_history"
    legacy.mkdir()
    (legacy / "config").write_text('[filter "unsafe"]\n', encoding="utf-8")
    (run_root / "outer_state.json").write_text('{"current_cycle": 2}\n', encoding="utf-8")

    result = _checkpoint(run_root)

    assert result.status == "committed"
    assert not any(path.startswith(".agentic_history") for path in _tracked(run_root))
    assert (legacy / "config").read_text(encoding="utf-8") == '[filter "unsafe"]\n'

    # a symlinked sidecar is rejected without touching its target
    run_root = _run_dir(tmp_path / "symlinked")
    external = tmp_path / "external"
    external.mkdir()
    checkpoint_git_dir(run_root, ".agentic_history").symlink_to(external, target_is_directory=True)

    result = _checkpoint(run_root)

    assert result.status == "degraded"
    assert "not a real directory" in str(result.error)
    assert list(external.iterdir()) == []


def test_failures_after_the_first_commit_never_replace_or_block_history(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = _run_dir(tmp_path)
    _checkpoint(run_root)
    original_head = _head(run_root)
    (run_root / "new.txt").write_text("unscanned\n", encoding="utf-8")

    # a failed staged diff cannot advance the final checkpoint
    original_run_git = checkpoint_git_module.run_git

    def fail_diff(git_dir, root, settings, *args, **kwargs):
        if args and args[0] == "diff":
            return subprocess.CompletedProcess(args, 2, "", "simulated diff failure")
        return original_run_git(git_dir, root, settings, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(checkpoint_git_module, "run_git", fail_diff)
        result = _checkpoint(run_root, final=True)

    assert result.status == "degraded"
    assert _head(run_root) == original_head

    # a HEAD probe failure does not replace history
    with monkeypatch.context() as m:
        m.setattr(
            run_checkpoint_module,
            "_head_commit",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("probe failed")),
        )
        result = _checkpoint(run_root)

    assert result.status == "degraded"
    assert _head(run_root) == original_head

    # a main index lock cannot turn a commit into degraded
    git_dir = checkpoint_git_dir(run_root, ".agentic_history")
    (git_dir / "index.lock").write_text("occupied\n", encoding="utf-8")

    result = _checkpoint(run_root)

    assert result.status == "committed"
    assert result.commit == _head(run_root)


@pytest.mark.parametrize("failing_command", ["read-tree", "tag", "repack", "lock-cleanup"])
def test_post_ref_failure_still_reports_committed(
    tmp_path: Path, monkeypatch, failing_command: str
) -> None:
    run_root = _run_dir(tmp_path)
    original_run_git = checkpoint_git_module.run_git

    def time_out_maintenance(git_dir, root, settings, *args, **kwargs):
        is_index_refresh = args and args[0] == "read-tree" and kwargs.get("quarantine") is None
        if (args and args[0] == failing_command and failing_command != "read-tree") or (
            failing_command == "read-tree" and is_index_refresh
        ):
            raise subprocess.TimeoutExpired(args, settings.timeout_seconds)
        return original_run_git(git_dir, root, settings, *args, **kwargs)

    monkeypatch.setattr(checkpoint_git_module, "run_git", time_out_maintenance)
    monkeypatch.setattr(run_checkpoint_module, "_run_git", time_out_maintenance)
    if failing_command == "lock-cleanup":
        original_unlink = Path.unlink

        def fail_lock_cleanup(path: Path, *args, **kwargs):
            if path.name == "checkpoint.lock":
                raise PermissionError("simulated cleanup failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_lock_cleanup)

    result = _checkpoint(run_root, final=True)

    assert result.status == "committed"
    assert result.commit == _head(run_root)


def test_quarantine_identity_is_per_root_and_stale_quarantines_are_reaped(tmp_path: Path) -> None:
    # quarantine identity distinguishes same-named roots
    first_root = tmp_path / "first" / "run"
    second_root = tmp_path / "second" / "run"
    first_root.mkdir(parents=True)
    second_root.mkdir(parents=True)

    with pinned_checkpoint_paths(first_root, git_dir_name=".agentic_history") as first:
        first_alias = str(first.git_dir)
        first_identity = _quarantine_identity(first.git_dir)
    with pinned_checkpoint_paths(second_root, git_dir_name=".agentic_history") as second:
        second_alias = str(second.git_dir)
        second_identity = _quarantine_identity(second.git_dir)

    assert first_alias == second_alias
    assert first_identity != second_identity

    # a stale external quarantine is reaped
    run_root = _run_dir(tmp_path)
    identity = _quarantine_identity(checkpoint_git_dir(run_root, ".agentic_history"))
    stale = Path(tempfile.gettempdir()) / f"agentic-checkpoint-quarantine-{identity}-stale"
    stale.mkdir(mode=0o700)
    (stale / "secret").write_text("credential\n", encoding="utf-8")

    result = _checkpoint(run_root)

    assert result.status == "committed"
    assert not stale.exists()


def test_repeat_checkpoints_track_unchanged_deleted_and_newly_sensitive_files(
    tmp_path: Path,
) -> None:
    run_root = _run_dir(tmp_path)
    log_path = run_root / "agent.log"
    log_path.write_text("safe output\n", encoding="utf-8")
    _checkpoint(run_root)

    # a second checkpoint without changes makes no commit
    result = _checkpoint(run_root)

    assert result.status == "unchanged"
    assert _git(run_root, "rev-list", "--count", "HEAD").strip() == "1"

    # a file that becomes sensitive is removed from the current tree
    log_path.write_text(SECRET_LINE, encoding="utf-8")

    result = _checkpoint(run_root)

    assert result.status == "committed"
    assert result.skipped_sensitive == ("agent.log",)
    assert "agent.log" not in _tracked(run_root)
    assert _git(run_root, "show", "HEAD^:agent.log") == "safe output\n"

    # a deleted file is staged as a removal
    (run_root / "outer_state.json").unlink()

    result = _checkpoint(run_root)

    assert result.status == "committed"
    assert "outer_state.json" not in _tracked(run_root)

    # a colon in the run path supports repeat checkpoints
    run_root = _run_dir(tmp_path / "run:group")

    first = _checkpoint(run_root)
    (run_root / "new.txt").write_text("new\n", encoding="utf-8")
    second = _checkpoint(run_root)

    assert first.status == "committed"
    assert second.status == "committed"


def test_agent_gitignore_fifos_and_hostile_global_config_cannot_hide_output(
    tmp_path: Path, monkeypatch
) -> None:
    # a hostile global git config is neutralized
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".gitignore").write_text("*.json\n", encoding="utf-8")
    (fake_home / ".gitconfig").write_text(
        f"[commit]\n\tgpgsign = true\n[core]\n\texcludesFile = {fake_home / '.gitignore'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(fake_home))
    run_root = _run_dir(tmp_path)
    # an agent-authored .gitignore cannot hide experiment output
    source = run_root / "ideas" / "01" / "program_0" / "src"
    (source / ".gitignore").write_text("results/\neval-artifacts/\n", encoding="utf-8")
    (source / "results").mkdir()
    (source / "results" / "metrics.json").write_text('{"score": 1.0}\n', encoding="utf-8")
    # a fifo is skipped without failing the checkpoint
    os.mkfifo(run_root / "ideas" / "01" / ".agent_stream.fifo")

    result = _checkpoint(run_root)

    tracked = _tracked(run_root)
    assert result.status == "committed"
    assert "outer_state.json" in tracked
    assert "ideas/01/program_0/src/results/metrics.json" in tracked
    assert not any("fifo" in path for path in tracked)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_unreadable_file_still_commits_the_readable_ones(tmp_path: Path) -> None:
    run_root = _run_dir(tmp_path)
    secret = run_root / "unreadable.txt"
    secret.write_text("nope\n", encoding="utf-8")
    secret.chmod(0o000)

    result = _checkpoint(run_root)

    assert result.status == "committed"
    assert "outer_state.json" in _tracked(run_root)
    assert result.error is not None and "unreadable" in result.error


def test_deadline_is_set_once_and_honored_by_scans_and_git_commands(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = _run_dir(tmp_path)

    # write_checkpoint sets one monotonic deadline
    observed_deadlines: list[float | None] = []

    def stop_after_preflight(_git_dir, *, root, settings):
        assert root == run_root
        observed_deadlines.append(settings._deadline)
        raise OSError("stop after deadline observation")

    with monkeypatch.context() as m:
        m.setattr(run_checkpoint_module.time, "monotonic", lambda: 100.0)
        m.setattr(run_checkpoint_module, "_ensure_repository", stop_after_preflight)
        result = _checkpoint(run_root, settings=_settings(timeout_seconds=60))

    assert result.status == "degraded"
    assert observed_deadlines == [160.0]

    # the aggregate scan honors the checkpoint deadline
    with monkeypatch.context() as m:
        m.setattr(checkpoint_paths_module.time, "monotonic", lambda: float("inf"))
        result = _checkpoint(run_root, settings=_settings(max_total_bytes=1024))

    assert result.status == "degraded"
    assert "timed out" in str(result.error)

    # git commands use the remaining whole-checkpoint deadline
    git_dir = tmp_path / "sidecar"
    git_dir.mkdir()
    observed_timeouts: list[float] = []
    clock = iter((100.0, 110.0))

    def fake_run(argv, **kwargs):
        observed_timeouts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    with monkeypatch.context() as m:
        m.setattr(checkpoint_git_module.time, "monotonic", lambda: next(clock))
        m.setattr(checkpoint_git_module.subprocess, "run", fake_run)
        settings = run_checkpoint_module.replace(_settings(timeout_seconds=60), _deadline=160.0)
        checkpoint_git_module.run_git(git_dir, run_root, settings, "status")
        checkpoint_git_module.run_git(git_dir, run_root, settings, "status")

    assert observed_timeouts == [60.0, 50.0]


def test_cross_device_object_promotion_uses_bounded_temporary_copy(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "objects" / "target"
    destination.parent.mkdir()
    source.write_bytes(b"object")
    original_link = os.link

    def cross_device_once(existing, target, *args, **kwargs):
        if Path(existing) == source:
            raise OSError(18, "cross-device")
        return original_link(existing, target, *args, **kwargs)

    monkeypatch.setattr(checkpoint_git_module.os, "link", cross_device_once)

    created = checkpoint_git_module._promote_object(
        source, destination, settings=_settings(timeout_seconds=10)
    )

    assert created is True
    assert destination.read_bytes() == b"object"
    assert list(destination.parent.glob(".*.promote-*")) == []


def _trailer(run_root: Path, key: str) -> str:
    return _git(run_root, "log", "-1", f"--format=%(trailers:key={key},valueonly)").strip()


def test_commit_identity_dates_and_trailers_are_pinned_and_unforgeable(tmp_path: Path) -> None:
    run_root = _run_dir(tmp_path)

    _checkpoint(run_root, phase="ablations", cycle=3, idea_id="idea_x", event="idea_drafted")

    # commit identity and dates are pinned
    author, committer, author_date, commit_date = _git(
        run_root, "log", "-1", "--format=%an <%ae>%n%cn <%ce>%n%ad%n%cd", "--date=iso"
    ).splitlines()
    assert (
        author == committer == "Agentic Scientist Checkpoint <checkpoint@agentic-scientist.invalid>"
    )
    assert author_date == commit_date
    assert author_date.endswith("+0000")
    # trailers are machine parseable
    assert _trailer(run_root, "Checkpoint-Phase") == "ablations"
    assert _trailer(run_root, "Checkpoint-Cycle") == "3"

    # trailers cannot be forged by a multiline idea id
    (run_root / "new.txt").write_text("new\n", encoding="utf-8")

    _checkpoint(run_root, idea_id="real\nCheckpoint-Phase: forged")

    assert _trailer(run_root, "Checkpoint-Phase") == "investigation"


def test_final_checkpoints_include_environments_and_tags_are_sequenced(tmp_path: Path) -> None:
    # a final checkpoint includes environments but never credentials
    run_root = _run_dir(tmp_path / "envs")
    (run_root / "task" / ".venv").mkdir(parents=True)
    (run_root / "task" / ".venv" / "x.so").write_text("lib\n", encoding="utf-8")
    (run_root / "ideas" / "01" / "__pycache__").mkdir()
    (run_root / "ideas" / "01" / "__pycache__" / "m.pyc").write_text("b\n", encoding="utf-8")
    (run_root / ".agent_home").mkdir()
    (run_root / ".agent_home" / "auth.json").write_text("secret\n", encoding="utf-8")

    _checkpoint(run_root)
    assert not any(".venv" in path for path in _tracked(run_root))

    result = _checkpoint(run_root, final=True)

    tracked = _tracked(run_root)
    assert result.status == "committed"
    assert "task/.venv/x.so" in tracked
    assert "ideas/01/__pycache__/m.pyc" in tracked
    assert not any("agent_home" in path for path in tracked)

    # tags are sequenced; a final checkpoint commits and tags even with no changes
    run_root = _run_dir(tmp_path / "tags")

    first = _checkpoint(run_root, label="idea c001")
    final = _checkpoint(run_root, final=True)

    assert first.tag == "checkpoint/0001-idea-c001"
    assert final.status == "committed"
    assert final.tag == "checkpoint/0002-final"
    assert _git(run_root, "rev-list", "--count", "HEAD").strip() == "2"

    # tags can be disabled
    (run_root / "ideas" / "01" / "note.md").write_text("hi\n", encoding="utf-8")
    second = _checkpoint(run_root, label="idea c002", settings=_settings(tag_checkpoints=False))

    assert second.status == "committed"
    assert second.tag is None


def test_disabled_settings_missing_root_and_missing_git_do_not_raise(
    tmp_path: Path, monkeypatch
) -> None:
    # disabled settings take no checkpoint
    run_root = _run_dir(tmp_path)

    result = _checkpoint(run_root, settings=_settings(enabled=False))

    assert result.status == "disabled"
    assert not checkpoint_git_dir(run_root, ".agentic_history").exists()

    # a missing run root is disabled rather than raising
    result = _checkpoint(tmp_path / "absent")

    assert result.status == "disabled"
    assert result.error is not None

    # a missing git binary degrades
    monkeypatch.setattr(checkpoint_git_module, "_GIT_EXECUTABLE", str(tmp_path / "missing-git"))

    result = _checkpoint(run_root)

    assert result.status == "degraded"
    assert result.error is not None


def test_busy_lock_degrades_while_a_stale_lock_is_reclaimed(tmp_path: Path) -> None:
    # a busy lock degrades instead of blocking
    run_root = _run_dir(tmp_path / "busy")
    git_dir = checkpoint_git_dir(run_root, ".agentic_history")
    git_dir.mkdir(mode=0o700)
    (git_dir / "checkpoint.lock").write_text(f"{os.getpid()}\n", encoding="utf-8")

    result = _checkpoint(run_root, settings=_settings(timeout_seconds=1))

    assert result.status == "degraded"
    assert result.error == "checkpoint lock busy"

    # a stale lock from a dead process is reclaimed
    run_root = _run_dir(tmp_path / "stale")
    git_dir = checkpoint_git_dir(run_root, ".agentic_history")
    git_dir.mkdir(mode=0o700)
    dead_pid = subprocess.run(["true"], check=False)
    (git_dir / "checkpoint.lock").write_text("2147483646\n", encoding="utf-8")
    assert dead_pid.returncode == 0

    result = _checkpoint(run_root, settings=_settings(timeout_seconds=2))

    assert result.status == "committed"


_DESCRIBE_CASES = [
    ("scope", {}, ["paper phase", ".agentic_history", "full run directory"]),
    ("no-envs", {"final_include_environments": False}, ["without derived environments"]),
]


def test_describe_checkpoint_names_scope_and_policy_and_sanitizes_the_path(
    tmp_path: Path,
) -> None:
    # the preview names the sidecar, scope and environment policy
    for name, overrides, expected in _DESCRIBE_CASES:
        preview = describe_checkpoint(
            run_root=tmp_path / "run",
            label="paper phase",
            settings=_settings(**overrides),
            final=True,
        )

        assert all(fragment in preview for fragment in expected), name

    # the preview sanitizes the sidecar path
    sensitive = describe_checkpoint(run_root=tmp_path / TOKEN, label="start", settings=_settings())
    control = describe_checkpoint(
        run_root=tmp_path / "line\nbreak", label="start", settings=_settings()
    )

    assert "sk-proj" not in sensitive
    assert "<redacted-path:" in sensitive
    assert "\n" not in control
    assert "\\x0a" in control


# --------------------------------------------------------------------------- secret exclusion


def test_agent_homes_and_common_credential_files_are_never_committed(tmp_path: Path) -> None:
    run_root = _run_dir(tmp_path)
    # nested agent homes
    for relative in (".agent_home", "variants/v1/.agent_home", "posthoc/a/attempt_1/.agent_home"):
        home = run_root / relative
        home.mkdir(parents=True)
        (home / "auth.json").write_text('{"token": "secret"}\n', encoding="utf-8")
    # common credential files
    credential_files = (
        ".env",
        "task/.docker/config.json",
        "task/.git-credentials",
        "task/.env.local",
        "task/.config/gh/hosts.yml",
        "task/.kaggle/kaggle.json",
        "task/.npmrc",
        "task/.pypirc",
        "task/.netrc",
        "task/.ssh/id_ed25519",
        "task/.aws/credentials",
        "task/service-account.json",
    )
    for relative in credential_files:
        path = run_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("secret-value\n", encoding="utf-8")
    (run_root / "task/.env.example").write_text("API_KEY=\n", encoding="utf-8")

    _checkpoint(run_root)

    tracked = _tracked(run_root)
    assert not any("agent_home" in path for path in tracked)
    assert not any(path in tracked for path in credential_files)
    assert "task/.env.example" in tracked


def test_secret_text_and_sensitive_symlinks_are_excluded_and_reported(tmp_path: Path) -> None:
    # secret-bearing text is excluded and reported
    run_root = _run_dir(tmp_path / "text")
    (run_root / "agent.log").write_text(SECRET_LINE, encoding="utf-8")
    (run_root / "metrics.json").write_text('{"completion_tokens": 42}\n', encoding="utf-8")

    result = _checkpoint(run_root)

    assert result.skipped_sensitive == ("agent.log",)
    assert "agent.log" not in _tracked(run_root)
    assert "metrics.json" in _tracked(run_root)
    assert "Checkpoint-Skipped-Sensitive: 1" in _git(run_root, "log", "-1", "--format=%B")

    # a sensitive symlink name and target are excluded
    run_root = _run_dir(tmp_path / "links")
    (run_root / ".env").symlink_to(TOKEN)
    (run_root / "latest").symlink_to(TOKEN)

    result = _checkpoint(run_root)

    assert result.skipped_sensitive == (".env", "latest")
    assert ".env" not in _tracked(run_root)
    assert "latest" not in _tracked(run_root)


_PROVIDER_TOKENS = (
    "github_pat_abcdefghijklmnopqrstuvwxyz012345",
    "glpat-abcdefghijklmnopqrstuvwxyz012345",
    "hf_abcdefghijklmnopqrstuvwxyz012345",
    "xoxb-1234567890-abcdefghijklmnopqrstuvwxyz",
    "AIzaabcdefghijklmnopqrstuvwxyz0123456789",
    "eyJabcdefghijk.abcdefghijklmnop.abcdefghijklmnop",
)


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        pytest.param(
            "settings.toml", b'password = "correct-horse-battery-staple"\n', id="assignment"
        ),
        pytest.param("settings.bin", b'password="correct.horse.battery.staple"\n', id="dots"),
        pytest.param("settings.bin", b'\0password="correcthorsebattery"\n', id="nul-prefix"),
        pytest.param("settings.bin", b'password="real-example-password-7xQ9"\n', id="hyphens"),
        pytest.param("settings.bin", b'password="$uperSecret12345"\n', id="symbols"),
        pytest.param(
            "settings.bin",
            b"AWS_SECRET_ACCESS_KEY=abcdefghijklmnopqrstuvwxyz0123456789ABCD\n",
            id="aws",
        ),
        *[
            pytest.param("agent.out", f"token={token}\n".encode(), id=token[:6])
            for token in _PROVIDER_TOKENS
        ],
        pytest.param("tls.key", b"opaque-key-data\n", id="key-file"),
        *[
            pytest.param(name, f"payload={TOKEN}\n".encode(), id=name)
            for name in ("artifact.xml", "artifact.bin", "artifact.out")
        ],
        pytest.param("nul.log", f"\0payload={TOKEN}\0".encode(), id="nul-blob"),
    ],
)
def test_secret_bearing_content_is_excluded(tmp_path: Path, relative: str, content: bytes) -> None:
    run_root = _run_dir(tmp_path)
    (run_root / relative).write_bytes(content)

    result = _checkpoint(run_root)

    assert result.skipped_sensitive == (relative,)
    assert relative not in _tracked(run_root)


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        pytest.param("template.env.example", "OPENAI_API_KEY=${OPENAI_API_KEY}\n", id="expansion"),
        pytest.param("template.env.example", "OPENAI_API_KEY=placeholder_value\n", id="placeholder"),
        pytest.param("template.env.example", "GOOGLE_API_KEY=your-api-key-here\n", id="your-key"),
        pytest.param("template.env.example", 'password="a_secure_password"\n', id="generic-pw"),
        pytest.param("source.py", "password = config.password\n", id="attribute"),
        pytest.param("source.py", "password: Optional[str]\n", id="annotation"),
        pytest.param("source.py", "api_key = placeholder_value\n", id="name"),
        pytest.param("source.py", "client_secret = argparse.ArgumentParser()\n", id="call"),
        pytest.param("source.py", "access_token = response.access_token\n", id="chained"),
        pytest.param(
            "ca.pem",
            "-----BEGIN CERTIFICATE-----\npublic-ca-data\n-----END CERTIFICATE-----\n",
            id="certificate",
        ),
    ],
)
def test_secret_scanner_keeps_templates_and_non_literals(
    tmp_path: Path, relative: str, content: str
) -> None:
    run_root = _run_dir(tmp_path)
    (run_root / relative).write_text(content, encoding="utf-8")

    result = _checkpoint(run_root)

    assert result.status == "committed"
    assert result.skipped_sensitive == ()
    assert relative in _tracked(run_root)


def test_token_bearing_filenames_are_excluded_and_redacted(tmp_path: Path) -> None:
    # a token-bearing filename is excluded and redacted
    run_root = _run_dir(tmp_path / "sensitive")
    (run_root / TOKEN).write_text("safe body\n", encoding="utf-8")

    result = _checkpoint(run_root)

    assert result.skipped_sensitive[0].startswith("<redacted-path:")
    assert TOKEN not in str(result)
    assert TOKEN not in _tracked(run_root)
    git_dir = checkpoint_git_dir(run_root, ".agentic_history")
    assert not any("pathspec" in path.name for path in git_dir.iterdir())

    # a token-bearing oversize filename is redacted
    run_root = _run_dir(tmp_path / "oversize")
    (run_root / TOKEN).write_text("xx", encoding="utf-8")

    result = _checkpoint(run_root, settings=_settings(max_file_bytes=1))

    assert any(path.startswith("<redacted-path:") for path in result.skipped_sensitive)
    assert TOKEN not in str(result)


def test_secret_scan_uses_the_exact_staged_blob_and_failed_commits_leave_nothing_staged(
    tmp_path: Path, monkeypatch
) -> None:
    # the secret scan uses the exact staged blob
    run_root = _run_dir(tmp_path / "staged")
    output = run_root / "agent.log"
    output.write_text("safe output\n", encoding="utf-8")
    original_scan = run_checkpoint_module._scan_staged_index

    def mutate_after_stage(*args, **kwargs):
        output.write_text(SECRET_LINE, encoding="utf-8")
        return original_scan(*args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(run_checkpoint_module, "_scan_staged_index", mutate_after_stage)
        result = _checkpoint(run_root)

    assert result.status == "committed"
    assert _git(run_root, "show", "HEAD:agent.log") == "safe output\n"

    # a failed commit leaves no staged content for the retry
    run_root = _run_dir(tmp_path / "retry")
    output = run_root / "agent.log"
    output.write_text("safe output\n", encoding="utf-8")

    with monkeypatch.context() as m:
        m.setattr(
            run_checkpoint_module,
            "_create_quarantined_commit",
            lambda *args, **kwargs: (None, "simulated commit failure"),
        )
        failed = _checkpoint(run_root)
    output.write_text(SECRET_LINE, encoding="utf-8")
    retried = _checkpoint(run_root)

    assert failed.status == "degraded"
    assert retried.status == "committed"
    assert retried.skipped_sensitive == ("agent.log",)
    assert "agent.log" not in _tracked(run_root)


def test_rejected_secret_blob_is_not_promoted_to_sidecar(tmp_path: Path) -> None:
    run_root = _run_dir(tmp_path)
    secret = SECRET_LINE.encode()
    (run_root / "agent.log").write_bytes(secret)

    result = _checkpoint(run_root)

    git_dir = checkpoint_git_dir(run_root, ".agentic_history")
    object_id = subprocess.run(
        ["git", f"--git-dir={git_dir}", "hash-object", "--stdin"],
        input=secret,
        capture_output=True,
        check=True,
    ).stdout.decode().strip()
    present = subprocess.run(
        ["git", f"--git-dir={git_dir}", "cat-file", "-e", object_id],
        capture_output=True,
        check=False,
    )
    assert result.status == "committed"
    assert present.returncode != 0


_CAP8 = _settings(max_file_bytes=8)


def test_nested_repositories_are_excluded_without_degrading_or_size_reports(
    tmp_path: Path,
) -> None:
    # a nested repository is excluded without degrading
    run_root = _run_dir(tmp_path / "nested")
    nested = run_root / "embedded"
    nested.mkdir()
    subprocess.run(["git", "-C", str(nested), "init", "-q"], check=True)
    (nested / "data.txt").write_text("nested\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(nested), "add", "data.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(nested), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "-q", "-m", "nested"],
        check=True,
    )

    result = _checkpoint(run_root)

    assert result.status == "committed"
    assert "embedded" in result.skipped_sensitive
    assert not any(path.startswith("embedded") for path in _tracked(run_root))

    # the final cap never reports nested git objects
    run_root = tmp_path / "objects" / "run"
    objects = run_root / "nested" / ".git" / "objects" / "aa"
    objects.mkdir(parents=True)
    (objects / "large-object").write_bytes(b"x" * 64)
    (run_root / "metadata.txt").write_text("ok\n", encoding="utf-8")

    result = _checkpoint(run_root, final=True, settings=_CAP8)

    assert result.status == "committed"
    assert result.skipped_oversize == ()
    assert "nested" in result.skipped_sensitive
    assert not any(path.startswith("nested") for path in _tracked(run_root))


def test_degraded_errors_redact_secrets_and_escape_control_characters(
    tmp_path: Path, monkeypatch
) -> None:
    # a non-git exception redacts a recognizable secret
    run_root = _run_dir(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(
            run_checkpoint_module,
            "_ensure_private_git_dir",
            lambda *_args: (_ for _ in ()).throw(OSError(f"cannot open {TOKEN}")),
        )
        result = _checkpoint(run_root)

    assert result.status == "degraded"
    assert TOKEN not in str(result.error)
    assert str(result.error).startswith("<redacted-sensitive-text:")

    # a degraded checkpoint error escapes control characters
    run_root = tmp_path / "line\nbreak"
    run_root.mkdir()
    sidecar = tmp_path / ".line\nbreak.agentic_history"
    outside = tmp_path / "outside"
    outside.mkdir()
    sidecar.symlink_to(outside, target_is_directory=True)

    result = _checkpoint(run_root, phase="pipeline", final=True, pin_run_root=True)

    assert result.status == "degraded"
    assert "\n" not in str(result.error)
    assert "\\x0a" in str(result.error)


# --------------------------------------------------------------------------- size limits

_UNSAFE_OVERSIZE_NAMES = [
    ("control", "line\nbreak.log", r"line\x0abreak.log"),
    ("unicode-format", "rtl\u202ebreak.log", r"rtl\u202ebreak.log"),
]


def test_oversize_files_are_skipped_and_reported_with_literal_escaped_names(
    tmp_path: Path,
) -> None:
    # an oversize file is skipped and recorded
    run_root = _run_dir(tmp_path / "recorded")
    (run_root / "huge.log").write_text("x" * 4096, encoding="utf-8")

    result = _checkpoint(run_root, settings=_settings(max_file_bytes=1024))

    assert result.skipped_oversize == ("huge.log",)
    assert "huge.log" not in _tracked(run_root)
    assert "Checkpoint-Skipped-Oversize: 1" in _git(run_root, "log", "-1", "--format=%B")

    # the oversize pathspec treats metacharacters literally
    run_root = _bare_run(tmp_path / "meta", "huge1.log")
    (run_root / "huge[1].log").write_bytes(b"x" * 64)

    result = _checkpoint(run_root, settings=_CAP8)

    assert result.skipped_oversize == ("huge[1].log",)
    assert "huge[1].log" not in _tracked(run_root)
    assert "huge1.log" in _tracked(run_root)

    # the oversize report escapes unsafe characters
    for case, name, reported in _UNSAFE_OVERSIZE_NAMES:
        run_root = _bare_run(tmp_path / case, "metadata.txt")
        (run_root / name).write_bytes(b"x" * 64)

        result = _checkpoint(run_root, settings=_CAP8)

        assert result.skipped_oversize == (reported,), case


def test_final_environment_policy_respects_size_caps_and_removes_disabled_runtimes(
    tmp_path: Path,
) -> None:
    # final environment inclusion still enforces the per-file cap
    run_root = tmp_path / "cap" / "run"
    environment = run_root / "task" / ".venv"
    environment.mkdir(parents=True)
    (environment / "small.txt").write_text("small\n", encoding="utf-8")
    (environment / "large.bin").write_bytes(b"x" * 64)

    result = _checkpoint(run_root, final=True, settings=_CAP8)

    assert result.skipped_oversize == ("task/.venv/large.bin",)
    assert "task/.venv/small.txt" in _tracked(run_root)
    assert "task/.venv/large.bin" not in _tracked(run_root)

    # the total cap excludes final environments when they are disabled
    run_root = tmp_path / "disabled" / "run"
    (run_root / "task" / ".venv").mkdir(parents=True)
    (run_root / "task" / ".venv" / "large.bin").write_bytes(b"x" * 64)
    (run_root / "metadata").write_bytes(b"12345678")

    result = _checkpoint(
        run_root,
        final=True,
        settings=_settings(max_total_bytes=8, final_include_environments=False),
    )

    assert result.status == "committed"
    assert "task/.venv/large.bin" not in _tracked(run_root)

    # the final environment policy removes a previously tracked runtime
    run_root = tmp_path / "removed" / "run"
    runtime = run_root / "task" / ".venv"
    runtime.mkdir(parents=True)
    (runtime / "large.bin").write_bytes(b"x" * 64)
    first = _checkpoint(run_root, final=True, settings=_settings(final_include_environments=True))

    second = _checkpoint(
        run_root,
        final=True,
        settings=_settings(max_total_bytes=8, final_include_environments=False),
    )

    assert first.status == "committed"
    assert second.status == "committed"
    assert "task/.venv/large.bin" not in _tracked(run_root)


def test_files_that_become_oversize_are_removed_from_the_current_tree(tmp_path: Path) -> None:
    # a file that grows past the cap is removed
    run_root = _bare_run(tmp_path / "grown", "metadata.txt", "output.log")
    _checkpoint(run_root, settings=_CAP8)
    (run_root / "output.log").write_bytes(b"x" * 64)

    result = _checkpoint(run_root, settings=_CAP8)

    assert result.status == "committed"
    assert result.skipped_oversize == ("output.log",)
    assert "output.log" not in _tracked(run_root)

    # a new cap removes an unchanged oversize symlink
    run_root = _bare_run(tmp_path / "symlink")
    (run_root / "latest").symlink_to("x" * 64)

    _checkpoint(run_root, settings=_settings(max_file_bytes=0))
    result = _checkpoint(run_root, settings=_CAP8)

    assert result.status == "committed"
    assert result.skipped_oversize == ("latest",)
    assert "latest" not in _tracked(run_root)


_STAGED_BLOB_CASES = [
    ("grows-after-scan", b"small\n", b"x" * 64),
    ("shrinks-after-prefilter", b"x" * 64, b"ok\n"),
]


def test_size_cap_checks_the_exact_staged_blob_without_quarantining_obvious_oversize(
    tmp_path: Path, monkeypatch
) -> None:
    # the size cap checks the exact staged blob
    for case, initial, mutated in _STAGED_BLOB_CASES:
        run_root = _bare_run(tmp_path / case, "metadata.txt")
        output = run_root / "output.log"
        output.write_bytes(initial)
        original_pathspec = run_checkpoint_module._write_pathspec

        def mutate_before_staging(*args, _output=output, _mutated=mutated, **kwargs):
            _output.write_bytes(_mutated)
            return original_pathspec(*args, **kwargs)

        with monkeypatch.context() as m:
            m.setattr(run_checkpoint_module, "_write_pathspec", mutate_before_staging)
            result = _checkpoint(run_root, settings=_CAP8)

        assert result.skipped_oversize == ("output.log",), case
        assert "output.log" not in _tracked(run_root), case

    # an obviously oversize blob is not written to quarantine
    run_root = _bare_run(tmp_path / "obvious", "metadata.txt")
    (run_root / "large.bin").write_bytes(os.urandom(256 * 1024))
    original_run_git = run_checkpoint_module._run_git
    quarantine_bytes: list[int] = []

    def observe_add(git_dir, root, settings, *args, **kwargs):
        result = original_run_git(git_dir, root, settings, *args, **kwargs)
        if args and args[0] == "add":
            objects = kwargs["quarantine"].objects
            quarantine_bytes.append(
                sum(path.stat().st_size for path in objects.rglob("*") if path.is_file())
            )
        return result

    with monkeypatch.context() as m:
        m.setattr(run_checkpoint_module, "_run_git", observe_add)
        result = _checkpoint(run_root, settings=_CAP8)

    assert result.skipped_oversize == ("large.bin",)
    assert quarantine_bytes[0] < 4096


def test_total_size_cap_is_checked_before_git_and_against_the_staged_tree(
    tmp_path: Path, monkeypatch
) -> None:
    # the total cap allows the exact boundary
    run_root = _bare_run(tmp_path / "boundary")
    (run_root / "payload.bin").write_bytes(b"12345678")

    result = _checkpoint(run_root, settings=_settings(max_total_bytes=8))

    assert result.status == "committed"

    # the total cap degrades before git staging
    run_root = _run_dir(tmp_path / "preflight")
    (run_root / "large.bin").write_bytes(b"x" * 64)

    def reject_git(*_args, **_kwargs):
        raise AssertionError("aggregate preflight must run before Git")

    with monkeypatch.context() as m:
        m.setattr(run_checkpoint_module, "_ensure_repository", reject_git)
        result = _checkpoint(run_root, settings=_settings(max_total_bytes=32))

    assert result.status == "degraded"
    assert result.error == "checkpoint byte budget exceeded (32 bytes)"

    # the total cap rechecks the exact staged tree
    run_root = _bare_run(tmp_path / "recheck")
    payload = run_root / "payload.bin"
    payload.write_bytes(b"12345678")
    original_total = run_checkpoint_module._checkpoint_total_bytes

    def grow_after_preflight(*args, **kwargs):
        observed = original_total(*args, **kwargs)
        payload.write_bytes(b"x" * 64)
        return observed

    with monkeypatch.context() as m:
        m.setattr(run_checkpoint_module, "_checkpoint_total_bytes", grow_after_preflight)
        result = _checkpoint(run_root, settings=_settings(max_total_bytes=8))

    assert result.status == "degraded"
    assert result.error == "checkpoint staged byte budget exceeded (8 bytes)"
    assert not checkpoint_git_dir(run_root, ".agentic_history").joinpath("refs/heads/master").exists()


def test_total_size_cap_excludes_per_file_skips_and_sensitive_paths(tmp_path: Path) -> None:
    # files skipped by the per-file cap do not count
    run_root = _run_dir(tmp_path / "per-file")
    (run_root / "large.bin").write_bytes(b"x" * 64)

    result = _checkpoint(run_root, settings=_settings(max_file_bytes=32, max_total_bytes=1024))

    assert result.status == "committed"
    assert result.skipped_oversize == ("large.bin",)

    # sensitive paths do not count
    run_root = _run_dir(tmp_path / "sensitive")
    (run_root / ".env").write_bytes(b"x" * 2048)

    result = _checkpoint(run_root, settings=_settings(max_total_bytes=1024))

    assert result.status == "committed"
    assert ".env" in result.skipped_sensitive


# --------------------------------------------------------------------------- root / path safety


def test_git_helpers_resolve_descriptors_executables_and_a_minimal_environment(
    tmp_path: Path, monkeypatch
) -> None:
    # descriptor_aliases rejects an embedded lookalike
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert descriptor_aliases(Path(f"/proc/self/fd/{descriptor}/work")) == {descriptor}
        assert descriptor_aliases(Path(f"/tmp/user/proc/self/fd/{descriptor}/work")) == set()
    finally:
        os.close(descriptor)

    # the git child environment is a minimal allowlist
    monkeypatch.setenv("OPENAI_API_KEY", TOKEN)
    monkeypatch.setenv("LD_PRELOAD", "/tmp/inject.so")
    monkeypatch.setenv("PATH", str(tmp_path))

    environment = git_environment()

    assert set(environment) == {
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_TERMINAL_PROMPT",
        "LANG",
        "LC_ALL",
        "PATH",
    }
    assert environment["PATH"] != str(tmp_path)
    assert Path(base_argv(tmp_path / "git", tmp_path / "run")[0]).is_absolute()

    # the git executable resolves from a nonstandard launch path
    custom_bin = tmp_path / "custom-bin"
    custom_bin.mkdir()
    custom_git = custom_bin / "git"
    custom_git.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    custom_git.chmod(0o755)

    assert _resolve_git_executable(str(custom_bin)) == str(custom_git.resolve())


def test_pinned_run_root_cannot_be_redirected_after_validation(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "outer_state.json").write_text("{}\n", encoding="utf-8")
    moved_root = tmp_path / "moved-run"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "host-secret.txt").write_text("secret\n", encoding="utf-8")
    real_checkpoint = run_checkpoint_module._checkpoint

    def swap_then_checkpoint(**kwargs):
        run_root.rename(moved_root)
        run_root.symlink_to(outside, target_is_directory=True)
        return real_checkpoint(**kwargs)

    monkeypatch.setattr(run_checkpoint_module, "_checkpoint", swap_then_checkpoint)

    result = _checkpoint(run_root, pin_run_root=True)

    assert result.status == "committed", result.error
    tracked = _tracked(moved_root, git_dir=tmp_path / ".run.agentic_history")
    assert "outer_state.json" in tracked
    assert "host-secret.txt" not in tracked


def test_parent_checkpoint_excludes_nested_phase_sidecars_even_when_created_late(
    tmp_path: Path, monkeypatch
) -> None:
    # a nested phase sidecar is excluded
    bundle = tmp_path / "bundle"
    child = bundle / "variants"
    child.mkdir(parents=True)
    (child / "variant.json").write_text("{}\n", encoding="utf-8")
    child_result = _checkpoint(child, label="ablations phase", phase="ablations")
    (bundle / "paper.txt").write_text("paper\n", encoding="utf-8")

    parent_result = _checkpoint(bundle, label="paper phase", phase="paper")

    assert child_result.status == "committed"
    assert parent_result.status == "committed"
    assert "paper.txt" in _tracked(bundle)
    assert not any(path.startswith(".variants.agentic_history/") for path in _tracked(bundle))

    # a sidecar created after the prescan is excluded
    bundle = tmp_path / "late-bundle"
    bundle.mkdir()
    (bundle / "paper.txt").write_text("paper\n", encoding="utf-8")
    real_scan = run_checkpoint_module.scan_sensitive_paths

    def scan_then_create(*args, **kwargs):
        scanned = real_scan(*args, **kwargs)
        objects = bundle / ".variants.agentic_history" / "objects" / "aa"
        objects.mkdir(parents=True)
        objects.joinpath("leaked").write_bytes(zlib.compress(TOKEN.encode()))
        return scanned

    monkeypatch.setattr(run_checkpoint_module, "scan_sensitive_paths", scan_then_create)

    result = _checkpoint(bundle, label="paper phase", phase="paper")

    assert result.status == "committed"
    assert not any(path.startswith(".variants.agentic_history/") for path in _tracked(bundle))


def test_pinned_checkpoint_debug_log_omits_root_path(tmp_path: Path, caplog) -> None:
    run_root = tmp_path / f"{TOKEN}\nline"
    run_root.mkdir()
    (run_root / "artifact.txt").write_text("safe\n", encoding="utf-8")

    with caplog.at_level("DEBUG", logger="src.investigation.checkpoint_git"):
        result = _checkpoint(run_root, phase="pipeline", final=True, pin_run_root=True)

    assert result.status == "committed"
    assert "sk-proj" not in caplog.text
    assert all("\n" not in record.message for record in caplog.records)


def test_checkpoint_ignores_ambient_git_environment(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "artifact.txt").write_text("safe\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    poisoned = {
        "GIT_DIR": str(outside / "repo"),
        "GIT_WORK_TREE": str(outside),
        "GIT_INDEX_FILE": str(outside / "index"),
        "GIT_OBJECT_DIRECTORY": str(outside / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(outside / "alternate"),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": str(outside / "hooks"),
        "GIT_TEMPLATE_DIR": str(outside / "templates"),
        "OPENAI_API_KEY": TOKEN,
        "AWS_SECRET_ACCESS_KEY": "host-secret",
        "LD_PRELOAD": str(outside / "inject.so"),
        "LD_LIBRARY_PATH": str(outside),
        "DYLD_INSERT_LIBRARIES": str(outside / "inject.dylib"),
        "PATH": str(outside),
    }
    for key, value in poisoned.items():
        monkeypatch.setenv(key, value)

    result = _checkpoint(run_root, phase="pipeline", final=True, pin_run_root=True)

    assert result.status == "committed", result.error
    monkeypatch.undo()
    assert _tracked(run_root) == ["artifact.txt"]
    assert list(outside.iterdir()) == []
