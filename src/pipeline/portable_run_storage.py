"""Portable scratch execution with durable, credential-safe checkpoint archives."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

from src.pipeline.safe_fs import UnsafeGluePathError
from src.pipeline.portable_storage_paths import ArchiveError, RunIdentity
from src.pipeline.portable_restore_state import RestoreBackup
from src.pipeline.portable_storage_io import (
    ensure_real_directory as _ensure_real_directory,
    read_small_json as _read_small_json,
    tree_size as _tree_size,
    validate_owned_directory as _validate_owned_directory,
    write_json_file as _write_json_file,
    write_text_file as _write_text_file,
)
from src.investigation.checkpoint_git import (
    ensure_repository,
    git_error,
    run_git,
)
from src.investigation.checkpoint_lock import ensure_private_git_dir
from src.investigation.checkpoint_security import redact_sensitive_text
from src.investigation.run_checkpoint import (
    CheckpointSettings,
    checkpoint_git_dir,
    write_checkpoint,
)

_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_MARKER_NAME = ".agentic_portable_archive.json"
_IDENTITY_NAME = "portable-archive-identity.json"
_ARCHIVE_VERSION = 1


class CheckpointArchive:
    """Mirror reachable safe checkpoints and their materialized trees."""

    def __init__(
        self,
        *,
        identity: RunIdentity,
        local_results_root: Path,
        durable_results_root: Path,
        timeout_seconds: int = 600,
    ) -> None:
        self.identity = identity
        self.local_results_root = Path(local_results_root).resolve()
        self.durable_results_root = Path(durable_results_root).resolve()
        self.settings = CheckpointSettings(
            max_total_bytes=16 * 1024**3,
            timeout_seconds=timeout_seconds,
            final_include_environments=False,
        )
        self._last_verified_commit: str | None = None

    @property
    def local_run_root(self) -> Path:
        return self.identity.run_root(self.local_results_root)

    @property
    def durable_run_root(self) -> Path:
        return self.identity.run_root(self.durable_results_root)

    def archive_latest(self) -> str | None:
        """Publish the current reachable source HEAD, or return None while busy."""
        try:
            return self._archive_latest()
        except ArchiveError:
            raise
        except (OSError, subprocess.SubprocessError, UnsafeGluePathError) as exc:
            detail = redact_sensitive_text(str(exc))
            raise ArchiveError(f"archive I/O failed: {type(exc).__name__}: {detail}") from exc

    def _archive_latest(self) -> str | None:
        source_git = checkpoint_git_dir(self.local_run_root, self.settings.git_dir_name)
        if not (source_git / "HEAD").is_file() or (source_git / "checkpoint.lock").exists():
            return None
        commit = self._head_commit(source_git, self.local_run_root)
        if commit == self._last_verified_commit:
            return commit
        self._reject_reserved_source_path(source_git, commit)
        marker = self._read_marker(required=False)
        if marker is not None and marker["local_results_root"] != str(self.local_results_root):
            raise ArchiveError("portable archive belongs to a different scratch root")
        self._ensure_durable_destination(marker)
        durable_git = checkpoint_git_dir(self.durable_run_root, self.settings.git_dir_name)
        if self._archive_is_current(durable_git, commit, marker):
            self._last_verified_commit = commit
            return commit
        self._transfer_commit(
            source_git=source_git,
            destination_git=durable_git,
            destination_root=self.durable_run_root,
            commit=commit,
        )
        self._write_marker(commit)
        self._last_verified_commit = commit
        return commit

    def restore(self) -> str:
        """Restore the last fully materialized archive to its stable scratch path."""
        marker = self._read_marker(required=True)
        assert marker is not None
        if marker["local_results_root"] != str(self.local_results_root):
            raise ArchiveError(
                "resume requires the same scratch root used by the archived run; "
                f"set AGENTIC_SCRATCH_ROOT for {marker['local_results_root']}"
            )
        commit = marker["commit"]
        durable_git = checkpoint_git_dir(self.durable_run_root, self.settings.git_dir_name)
        if not durable_git.is_dir():
            raise ArchiveError("portable archive Git sidecar is missing")

        self._require_restore_capacity()
        local_git = checkpoint_git_dir(self.local_run_root, self.settings.git_dir_name)
        backup = RestoreBackup(run_root=self.local_run_root, git_dir=local_git)
        preserved_token = backup.prepare()
        self._ensure_run_directory(self.local_run_root, allow_existing_empty=True)
        self._transfer_commit(
            source_git=durable_git,
            destination_git=local_git,
            destination_root=self.local_run_root,
            commit=commit,
            source_is_managed=True,
        )
        backup.complete(preserved_token)
        return commit

    def archived_local_results_root(self) -> Path:
        marker = self._read_marker(required=True)
        assert marker is not None
        return Path(marker["local_results_root"])

    def recoverable_local_results_root(self) -> Path | None:
        durable_git = checkpoint_git_dir(self.durable_run_root, self.settings.git_dir_name)
        try:
            identity = _read_small_json(durable_git / _IDENTITY_NAME)
        except (ArchiveError, FileNotFoundError):
            return None
        expected = {
            "version": _ARCHIVE_VERSION,
            "task": self.identity.task,
            "agent": self.identity.agent,
            "run_id": self.identity.run_id,
        }
        if not isinstance(identity, dict) or any(
            identity.get(key) != value for key, value in expected.items()
        ):
            return None
        local_root = identity.get("local_results_root")
        if not isinstance(local_root, str) or not Path(local_root).is_absolute():
            return None
        return Path(local_root)

    def recover_marker(self) -> str:
        """Complete publication interrupted after durable materialization."""
        durable_git = checkpoint_git_dir(self.durable_run_root, self.settings.git_dir_name)
        try:
            self._validate_managed_git(durable_git)
            self._write_safe_git_config(durable_git)
            commit = self._head_commit(durable_git, self.durable_run_root)
            if self._worktree_matches(durable_git, commit):
                self._write_marker(commit)
                return commit
        except (ArchiveError, FileNotFoundError):
            pass
        replayed_commit = self.archive_latest()
        if replayed_commit is None:
            raise ArchiveError("markerless durable archive cannot be recovered")
        return replayed_commit

    def source_has_final_checkpoint(self) -> bool:
        source_git = checkpoint_git_dir(self.local_run_root, self.settings.git_dir_name)
        if not (source_git / "HEAD").is_file():
            return False
        result = run_git(
            source_git,
            self.local_run_root,
            self.settings,
            "log",
            "-1",
            "--format=%s",
        )
        return result.returncode == 0 and result.stdout.strip() == "checkpoint: final"

    def current_source_commit(self) -> str | None:
        source_git = checkpoint_git_dir(self.local_run_root, self.settings.git_dir_name)
        if not (source_git / "HEAD").is_file():
            return None
        return self._head_commit(source_git, self.local_run_root)

    def checkpoint_final_state(self) -> str:
        """Create a gating final safe checkpoint before the allocation exits."""
        if not self.local_run_root.is_dir():
            raise ArchiveError(f"run directory was not created: {self.local_run_root}")
        result = write_checkpoint(
            run_root=self.local_run_root,
            label="portable archive final",
            phase="pipeline",
            event="portable_archive_final",
            settings=self.settings,
            run_id=self.identity.run_id,
            agent=self.identity.agent,
            final=True,
            pin_run_root=True,
        )
        if result.status != "committed" or result.commit is None:
            raise ArchiveError(f"final safe checkpoint failed: {result.error or result.status}")
        return result.commit

    def _head_commit(self, git_dir: Path, root: Path) -> str:
        result = run_git(git_dir, root, self.settings, "rev-parse", "--verify", "HEAD")
        if result.returncode != 0:
            raise ArchiveError(git_error("resolve archive source", result))
        commit = result.stdout.strip()
        if _COMMIT_RE.fullmatch(commit) is None:
            raise ArchiveError("checkpoint sidecar returned an invalid commit")
        return commit

    def _transfer_commit(
        self,
        *,
        source_git: Path,
        destination_git: Path,
        destination_root: Path,
        commit: str,
        source_is_managed: bool = False,
    ) -> None:
        if source_is_managed:
            self._validate_managed_git(source_git)
            self._write_safe_git_config(source_git)
        self._prepare_managed_git(destination_git, destination_root)
        for arguments, stage in (
            (("fetch", "--no-tags", "--force", str(source_git), commit), "fetch archive"),
            (("symbolic-ref", "HEAD", "refs/heads/master"), "select archive branch"),
            (("update-ref", "refs/heads/master", commit), "advance archive ref"),
            (("clean", "-ffdx", "-e", _MARKER_NAME), "clean archive worktree"),
            (("reset", "--hard", commit), "materialize archive"),
        ):
            result = run_git(destination_git, destination_root, self.settings, *arguments)
            if result.returncode != 0:
                raise ArchiveError(git_error(stage, result))

    def _ensure_durable_destination(self, marker: dict[str, str] | None) -> None:
        self._ensure_run_directory(self.durable_run_root, allow_existing_empty=True)
        entries = [entry.name for entry in os.scandir(self.durable_run_root)]
        durable_git = checkpoint_git_dir(self.durable_run_root, self.settings.git_dir_name)
        recoverable = durable_git.is_dir() and self._managed_git_identity_matches(durable_git)
        if marker is None and entries and not recoverable:
            raise ArchiveError(f"refusing unmanaged durable run directory: {self.durable_run_root}")

    def _ensure_run_directory(self, run_root: Path, *, allow_existing_empty: bool) -> None:
        root = (
            self.durable_results_root
            if run_root == self.durable_run_root
            else self.local_results_root
        )
        _ensure_real_directory(root)
        current = root
        for component in (self.identity.task, self.identity.agent, self.identity.run_id):
            current = current / component
            if current.exists():
                _validate_owned_directory(current)
            else:
                current.mkdir(mode=0o700)
        if not allow_existing_empty and any(current.iterdir()):
            raise ArchiveError(f"run directory is not empty: {current}")

    def _read_marker(self, *, required: bool) -> dict[str, str] | None:
        marker_path = self.durable_run_root / _MARKER_NAME
        try:
            raw = _read_small_json(marker_path)
        except FileNotFoundError:
            if required:
                raise ArchiveError(f"portable archive marker is missing: {marker_path}")
            return None
        except ArchiveError as exc:
            raise ArchiveError("portable archive marker is unsafe") from exc
        expected = {
            "version": _ARCHIVE_VERSION,
            "task": self.identity.task,
            "agent": self.identity.agent,
            "run_id": self.identity.run_id,
        }
        if not isinstance(raw, dict) or any(
            raw.get(key) != value for key, value in expected.items()
        ):
            raise ArchiveError("portable archive marker does not match this run")
        commit = raw.get("commit")
        local_root = raw.get("local_results_root")
        if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
            raise ArchiveError("portable archive marker has an invalid commit")
        if (
            not isinstance(local_root, str)
            or any(ord(character) < 32 or ord(character) == 127 for character in local_root)
            or not Path(local_root).is_absolute()
        ):
            raise ArchiveError("portable archive marker has an invalid scratch root")
        if str(Path(local_root).resolve()) != local_root:
            raise ArchiveError("portable archive marker scratch root is not canonical")
        return {key: str(value) for key, value in raw.items()}

    def _reject_reserved_source_path(self, source_git: Path, commit: str) -> None:
        result = run_git(
            source_git,
            self.local_run_root,
            self.settings,
            "ls-tree",
            "-r",
            "--name-only",
            "-z",
            commit,
        )
        if result.returncode != 0:
            raise ArchiveError(git_error("inspect archive tree", result))
        if _MARKER_NAME in result.stdout.split("\0"):
            raise ArchiveError(
                f"checkpoint contains reserved portable archive path: {_MARKER_NAME}"
            )

    def _archive_is_current(
        self,
        durable_git: Path,
        commit: str,
        marker: dict[str, str] | None,
    ) -> bool:
        if marker is None or marker["commit"] != commit:
            return False
        try:
            self._validate_managed_git(durable_git)
            self._write_safe_git_config(durable_git)
        except (ArchiveError, FileNotFoundError):
            return False
        head = self._head_commit(durable_git, self.durable_run_root)
        if head != commit:
            return False
        return self._worktree_matches(durable_git, commit)

    def _worktree_matches(self, durable_git: Path, commit: str) -> bool:
        diff = run_git(
            durable_git,
            self.durable_run_root,
            self.settings,
            "diff",
            "--quiet",
            commit,
            "--",
        )
        untracked = run_git(
            durable_git,
            self.durable_run_root,
            self.settings,
            "clean",
            "-nffdx",
            "-e",
            _MARKER_NAME,
        )
        return diff.returncode == 0 and untracked.returncode == 0 and not untracked.stdout.strip()

    def _prepare_managed_git(self, git_dir: Path, root: Path) -> None:
        if git_dir.exists() and not self._managed_git_identity_matches(git_dir):
            _validate_owned_directory(git_dir, require_private=True)
            with os.scandir(git_dir) as entries:
                has_entries = any(entries)
            if has_entries:
                raise ArchiveError(f"refusing unmanaged archive sidecar: {git_dir}")
        ensure_private_git_dir(git_dir)
        identity_path = git_dir / _IDENTITY_NAME
        if not identity_path.exists():
            _write_json_file(identity_path, self._identity_payload())
        self._validate_git_internals(git_dir, allow_uninitialized=True)
        ensure_repository(git_dir, root=root, settings=self.settings)
        self._validate_git_internals(git_dir, allow_uninitialized=False)
        self._write_safe_git_config(git_dir)

    def _validate_managed_git(self, git_dir: Path) -> None:
        ensure_private_git_dir(git_dir)
        if not self._managed_git_identity_matches(git_dir):
            raise ArchiveError(f"archive sidecar identity is invalid: {git_dir}")
        self._validate_git_internals(git_dir, allow_uninitialized=False)

    def _validate_git_internals(self, git_dir: Path, *, allow_uninitialized: bool) -> None:
        head = git_dir / "HEAD"
        if not head.exists():
            entries = {entry.name for entry in os.scandir(git_dir)}
            if allow_uninitialized and entries <= {_IDENTITY_NAME}:
                return
            raise ArchiveError(f"managed Git internals are unsafe: {git_dir}")
        for path, expected_directory in (
            (head, False),
            (git_dir / "config", False),
            (git_dir / "objects", True),
            (git_dir / "refs", True),
            (git_dir / "refs" / "heads", True),
            (git_dir / "refs" / "tags", True),
        ):
            metadata = path.lstat()
            expected = (
                stat.S_ISDIR(metadata.st_mode)
                if expected_directory
                else stat.S_ISREG(metadata.st_mode)
            )
            if not expected or metadata.st_uid != os.getuid():
                raise ArchiveError(f"managed Git internals are unsafe: {path}")
        self._validate_git_tree(git_dir)

    def _validate_git_tree(self, git_dir: Path) -> None:
        pending = [git_dir]
        while pending:
            directory = pending.pop()
            for entry in os.scandir(directory):
                metadata = entry.stat(follow_symlinks=False)
                if metadata.st_uid != os.getuid():
                    raise ArchiveError(f"managed Git internals are unsafe: {entry.path}")
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(Path(entry.path))
                elif not stat.S_ISREG(metadata.st_mode):
                    raise ArchiveError(f"managed Git internals are unsafe: {entry.path}")

    def _managed_git_identity_matches(self, git_dir: Path) -> bool:
        try:
            identity = _read_small_json(git_dir / _IDENTITY_NAME)
        except (ArchiveError, FileNotFoundError):
            return False
        return identity == self._identity_payload()

    def _identity_payload(self) -> dict[str, object]:
        return {
            "version": _ARCHIVE_VERSION,
            "task": self.identity.task,
            "agent": self.identity.agent,
            "run_id": self.identity.run_id,
            "local_results_root": str(self.local_results_root),
        }

    def _write_safe_git_config(self, git_dir: Path) -> None:
        config = (
            "[core]\n"
            "\trepositoryformatversion = 0\n"
            "\tfilemode = true\n"
            "\tbare = false\n"
            "\tlogallrefupdates = true\n"
        )
        config_path = git_dir / "config"
        try:
            if config_path.read_text(encoding="utf-8") == config:
                return
        except (OSError, UnicodeError):
            pass
        _write_text_file(config_path, config)

    def _write_marker(self, commit: str) -> None:
        payload = {
            "version": _ARCHIVE_VERSION,
            "task": self.identity.task,
            "agent": self.identity.agent,
            "run_id": self.identity.run_id,
            "commit": commit,
            "local_results_root": str(self.local_results_root),
        }
        _write_json_file(self.durable_run_root / _MARKER_NAME, payload)

    def _require_restore_capacity(self) -> None:
        required = _tree_size(self.durable_run_root)
        durable_git = checkpoint_git_dir(self.durable_run_root, self.settings.git_dir_name)
        required += _tree_size(durable_git)
        available = shutil.disk_usage(self.local_results_root).free
        if required > available:
            raise ArchiveError(
                f"insufficient scratch space to restore checkpoint: "
                f"need {required} bytes, have {available}"
            )
