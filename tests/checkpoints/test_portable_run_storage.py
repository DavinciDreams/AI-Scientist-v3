"""CheckpointArchive scratch<->durable transfer and portable storage path helpers."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

import src.pipeline.portable_run_storage as portable_storage_module
from src.investigation.run_checkpoint import CheckpointSettings, checkpoint_git_dir, write_checkpoint
from src.pipeline.portable_run_storage import CheckpointArchive
from src.pipeline.portable_storage_paths import (
    ArchiveError,
    RunIdentity,
    atomic_no_replace_supported,
    build_pipeline_command,
    scratch_results_root,
    select_scratch_base,
)

IDENTITY = RunIdentity(task="world_models", agent="codex", run_id="world_models_70189_run")
PIPELINE_ARGV = ["python", "-m", "src.pipeline.cli"]


@dataclass
class _Env:
    """A scratch run dir plus a durable root, with helpers to checkpoint and archive it."""

    local_results: Path
    durable_results: Path

    @property
    def local_run(self) -> Path:
        return IDENTITY.run_root(self.local_results)

    @property
    def durable_run(self) -> Path:
        return IDENTITY.run_root(self.durable_results)

    def write(self, name: str, content: str = "{}\n") -> Path:
        path = self.local_run / name
        path.write_text(content, encoding="utf-8")
        return path

    def checkpoint(self, label: str = "cycle 1") -> str:
        result = write_checkpoint(
            run_root=self.local_run,
            label=label,
            phase="investigation",
            settings=CheckpointSettings(timeout_seconds=30),
            run_id=IDENTITY.run_id,
        )
        assert result.status == "committed"
        assert result.commit is not None
        return result.commit

    def archive(self, local_results: Path | None = None) -> CheckpointArchive:
        return CheckpointArchive(
            identity=IDENTITY,
            local_results_root=local_results or self.local_results,
            durable_results_root=self.durable_results,
        )

    def backups(self) -> list[Path]:
        return list(self.local_run.parent.glob(f".{IDENTITY.run_id}.interrupted.*"))


def _make_env(base: Path) -> _Env:
    env = _Env(base / "scratch" / "results", base / "durable")
    env.local_run.mkdir(parents=True)
    return env


@pytest.fixture
def env(tmp_path: Path) -> _Env:
    return _make_env(tmp_path)


@pytest.fixture
def archived(env: _Env) -> tuple[_Env, CheckpointArchive, str]:
    """A run with `state.json` checkpointed and archived once."""
    env.write("state.json")
    commit = env.checkpoint()
    archive = env.archive()
    archive.archive_latest()
    return env, archive, commit


# --------------------------------------------------------------------------- scratch selection


def test_archive_policy_atomic_probe_and_scratch_results_root(tmp_path: Path, env: _Env) -> None:
    # the portable archive fallback uses the safe checkpoint policy
    archive = env.archive()

    assert archive.settings.max_total_bytes == 16 * 1024**3
    assert archive.settings.final_include_environments is False

    # the atomic no-replace probe succeeds on a local filesystem and leaves nothing behind
    probe = tmp_path / "probe"
    probe.mkdir()

    assert atomic_no_replace_supported(probe) is True
    assert list(probe.iterdir()) == []

    # the scratch results root is stable and owner-private
    base = tmp_path / "base"
    base.mkdir()
    first = scratch_results_root(base=base, repo_root=tmp_path / "repo")
    second = scratch_results_root(base=base, repo_root=tmp_path / "repo")

    assert first == second
    assert first.name == "results"
    assert first.parent.parent.stat().st_uid == os.getuid()
    assert first.parent.parent.stat().st_mode & 0o077 == 0


_UNSAFE_SCRATCH_ROOTS = [
    ("relative", "AGENTIC_SCRATCH_ROOT must be absolute"),
    ("{tmp}/unsafe\npath", "control characters"),
]


def test_select_scratch_prefers_explicit_root_rejects_unsafe_ones_and_falls_back(
    tmp_path: Path,
) -> None:
    # an explicit root wins
    explicit = tmp_path / "explicit"

    selected = select_scratch_base(
        environ={"AGENTIC_SCRATCH_ROOT": str(explicit), "TMPDIR": str(tmp_path / "tmpdir")},
        localssd=tmp_path / "localssd",
    )

    assert selected == explicit.resolve()

    # unavailable local SSD falls back to the system temp
    unavailable = tmp_path / "not-a-directory"
    unavailable.write_text("blocked\n", encoding="utf-8")

    selected = select_scratch_base(
        environ={}, localssd=unavailable / "localssd", system_temp=tmp_path / "system-temp"
    )

    assert selected == (tmp_path / "system-temp").resolve()

    # unsafe explicit roots are rejected
    for root, message in _UNSAFE_SCRATCH_ROOTS:
        with pytest.raises(ArchiveError, match=message):
            select_scratch_base(
                environ={"AGENTIC_SCRATCH_ROOT": root.format(tmp=tmp_path)},
                localssd=tmp_path / "localssd",
            )


# --------------------------------------------------------------------------- archive_latest


def test_archive_materializes_only_safe_content_and_replays_deletions(env: _Env) -> None:
    # only safe checkpoint content is materialized
    env.write("metrics.json", '{"score": 1}\n')
    env.write(".env", "OPENAI_API_KEY=secret\n")
    (env.local_run / ".agent_home").mkdir()
    env.write(".agent_home/auth.json", "secret\n")
    stale = env.write("stale.txt", "old\n")
    commit = env.checkpoint()
    archive = env.archive()

    assert env.archive().archive_latest() == commit

    assert (env.durable_run / "metrics.json").read_text(encoding="utf-8") == '{"score": 1}\n'
    assert not (env.durable_run / ".env").exists()
    assert not (env.durable_run / ".agent_home").exists()
    marker = json.loads((env.durable_run / ".agentic_portable_archive.json").read_text())
    assert marker["commit"] == commit
    assert marker["run_id"] == IDENTITY.run_id

    # deletions are materialized incrementally
    stale.unlink()
    env.write("current.txt", "new\n")
    commit = env.checkpoint(label="cycle 2")

    assert archive.archive_latest() == commit

    assert not (env.durable_run / "stale.txt").exists()
    assert (env.durable_run / "current.txt").read_text(encoding="utf-8") == "new\n"


def test_archive_removes_untracked_residue_but_rejects_symlinked_git_internals(
    tmp_path: Path, archived
) -> None:
    env, _archive, _commit = archived

    # untracked durable residue is removed
    residue = env.durable_run / "nested-repository"
    residue.mkdir()
    subprocess.run(["git", "init", "-q", str(residue)], check=True)
    (residue / "secret.txt").write_text("must disappear\n", encoding="utf-8")

    env.archive().archive_latest()

    assert not residue.exists()

    # symlinked managed git internals are rejected
    durable_git = checkpoint_git_dir(env.durable_run, ".agentic_history")
    pack = durable_git / "objects" / "pack"
    pack.rename(durable_git / "objects" / "pack-backup")
    pack.symlink_to(tmp_path / "outside", target_is_directory=True)

    with pytest.raises(ArchiveError, match="Git internals are unsafe"):
        env.archive().archive_latest()


def test_first_archive_retries_after_marker_or_sidecar_identity_failure(
    tmp_path: Path, env: _Env, monkeypatch
) -> None:
    # retry after the marker publication fails
    env.write("state.json")
    commit = env.checkpoint()
    archive = env.archive()
    real_write_marker = archive._write_marker
    monkeypatch.setattr(
        archive, "_write_marker", lambda _commit: (_ for _ in ()).throw(OSError("NFS"))
    )

    with pytest.raises(ArchiveError, match="archive I/O failed"):
        archive.archive_latest()

    monkeypatch.setattr(archive, "_write_marker", real_write_marker)
    assert env.archive().archive_latest() == commit

    # retry after the sidecar identity write fails
    env = _make_env(tmp_path / "identity")
    env.write("state.json")
    commit = env.checkpoint()
    archive = env.archive()
    real_write_json = portable_storage_module._write_json_file
    monkeypatch.setattr(
        portable_storage_module,
        "_write_json_file",
        lambda *_args: (_ for _ in ()).throw(OSError("NFS")),
    )

    with pytest.raises(ArchiveError, match="archive I/O failed"):
        archive.archive_latest()

    monkeypatch.setattr(portable_storage_module, "_write_json_file", real_write_json)
    assert env.archive().archive_latest() == commit


def test_markerless_partial_archive_replays_and_reserved_marker_path_is_rejected(
    tmp_path: Path, env: _Env
) -> None:
    # a markerless partial archive replays from the intact scratch
    env.write("state.json")
    env.checkpoint()
    archive = env.archive()
    durable_git = checkpoint_git_dir(env.durable_run, ".agentic_history")
    env.durable_run.mkdir(parents=True)
    durable_git.mkdir(mode=0o700)
    portable_storage_module._write_json_file(
        durable_git / "portable-archive-identity.json", archive._identity_payload()
    )

    recovered = env.archive().recover_marker()

    assert recovered
    assert (env.durable_run / "state.json").is_file()
    assert (env.durable_run / ".agentic_portable_archive.json").is_file()

    # a checkpoint containing the control marker is rejected
    env = _make_env(tmp_path / "reserved")
    env.write(".agentic_portable_archive.json")
    env.checkpoint()

    with pytest.raises(ArchiveError, match="reserved portable archive path"):
        env.archive().archive_latest()


# --------------------------------------------------------------------------- restore


def test_restore_rejects_symlinked_marker_and_foreign_scratch_path(tmp_path: Path, archived) -> None:
    # a symlinked archive marker is rejected
    env, archive, _commit = archived
    marker = env.durable_run / ".agentic_portable_archive.json"
    marker.unlink()
    marker.symlink_to(tmp_path / "outside.json")

    with pytest.raises(ArchiveError, match="marker is unsafe"):
        archive.restore()

    # a marker for a different scratch path is rejected
    env = _Env(tmp_path / "first" / "results", tmp_path / "other-durable")
    env.local_run.mkdir(parents=True)
    env.write("state.json")
    env.checkpoint()
    env.archive().archive_latest()

    with pytest.raises(ArchiveError, match="same scratch root"):
        env.archive(local_results=tmp_path / "second" / "results").restore()


def test_restore_preserves_the_interrupted_tree_and_keeps_only_the_latest_backup(
    tmp_path: Path, env: _Env, monkeypatch
) -> None:
    # the interrupted local tree is backed up and the marker commit is recovered
    env.write("state.json", '{"cycle": 1}\n')
    commit = env.checkpoint()
    archive = env.archive()
    archive.archive_latest()
    env.write("state.json", "interrupted\n")
    env.write("untracked.txt", "preserve me\n")

    assert archive.restore() == commit

    assert (env.local_run / "state.json").read_text(encoding="utf-8") == '{"cycle": 1}\n'
    assert not (env.local_run / "untracked.txt").exists()
    backups = env.backups()
    assert len(backups) == 1
    assert (backups[0] / "untracked.txt").read_text(encoding="utf-8") == "preserve me\n"

    # a repeated restore keeps only the latest interrupted backup
    env.write("interrupted.txt", "again\n")
    archive.restore()

    backups = env.backups()
    assert len(backups) == 1
    assert (backups[0] / "interrupted.txt").read_text(encoding="utf-8") == "again\n"

    # a failed restore also keeps only the latest interrupted backup
    env = _make_env(tmp_path / "failed")
    env.write("state.json")
    env.checkpoint()
    archive = env.archive()
    archive.archive_latest()
    monkeypatch.setattr(
        archive, "_transfer_commit", lambda **_kwargs: (_ for _ in ()).throw(OSError("NFS"))
    )

    for _attempt in range(2):
        with pytest.raises(OSError, match="NFS"):
            archive.restore()

    backups = env.backups()
    assert len(backups) == 1
    assert (backups[0] / "state.json").is_file()


# --------------------------------------------------------------------------- pipeline command


def test_pipeline_command_uses_results_root_for_fresh_runs_and_anchors_resume(
    tmp_path: Path,
) -> None:
    # a fresh run passes the (quoted) results root
    for case, results_name in (("plain", "results"), ("spaces-quoted", "results with spaces")):
        results = tmp_path / results_name

        command = build_pipeline_command(
            PIPELINE_ARGV, identity=IDENTITY, results_root=results, resume=False
        )

        assert command[-1] == f'paths.results_root="{results}"', case

    # a resume anchors the investigation run dir
    command = build_pipeline_command(
        PIPELINE_ARGV, identity=IDENTITY, results_root=tmp_path / "results", resume=True
    )

    assert command[-2:] == [
        "investigation.execution.resume=true",
        f'paths.run_dir="{IDENTITY.run_root(tmp_path / "results")}"',
    ]
