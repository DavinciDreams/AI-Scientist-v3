"""Crash-safe preservation of interrupted scratch state during restore."""

from __future__ import annotations

import os
import re
import secrets
import shutil
from pathlib import Path

from src.pipeline.portable_storage_io import (
    fsync_directory,
    read_small_json,
    validate_owned_directory,
    write_json_file,
)
from src.pipeline.portable_storage_paths import ArchiveError
from src.pipeline.safe_fs import rename_directory_no_replace

_TOKEN_RE = re.compile(r"[0-9a-f]{12}\Z")


class RestoreBackup:
    """Keep the newest genuine pre-restore state across failed retries."""

    def __init__(self, *, run_root: Path, git_dir: Path) -> None:
        self.run_root = run_root
        self.git_dir = git_dir
        self.parent = run_root.parent
        self.state_path = self.parent / f".{run_root.name}.portable-restore.json"

    def prepare(self) -> str:
        self.parent.mkdir(parents=True, exist_ok=True)
        validate_owned_directory(self.parent)
        state = self._read_state()
        if state is None:
            token = secrets.token_hex(6)
            self._write_state(token, "preparing")
            self._preserve(token)
            self._write_state(token, "preserved")
            return token

        token, stage = state
        if stage == "preparing":
            self._preserve(token)
            self._write_state(token, "preserved")
        else:
            self._discard_partial_restore()
        return token

    def complete(self, token: str) -> None:
        state = self._read_state()
        if state != (token, "preserved"):
            raise ArchiveError("portable restore transaction changed unexpectedly")
        self.state_path.unlink()
        fsync_directory(self.parent)
        self._prune(keep_token=token)

    def _preserve(self, token: str) -> None:
        for source, destination_name in self._backup_pairs(token):
            if not source.exists():
                continue
            validate_owned_directory(source)
            descriptor = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                rename_directory_no_replace(
                    descriptor,
                    source.name,
                    destination_name,
                    role="interrupted scratch run",
                )
            finally:
                os.close(descriptor)

    def _discard_partial_restore(self) -> None:
        for path in (self.run_root, self.git_dir):
            if not path.exists():
                continue
            validate_owned_directory(path)
            shutil.rmtree(path)

    def _prune(self, *, keep_token: str) -> None:
        keep_names = {name for _path, name in self._backup_pairs(keep_token)}
        prefixes = (
            f".{self.run_root.name}.interrupted.",
            f"{self.git_dir.name}.interrupted.",
        )
        for entry in os.scandir(self.parent):
            if entry.name in keep_names or not entry.name.startswith(prefixes):
                continue
            path = Path(entry.path)
            validate_owned_directory(path)
            shutil.rmtree(path)

    def _backup_pairs(self, token: str) -> tuple[tuple[Path, str], ...]:
        return (
            (self.run_root, f".{self.run_root.name}.interrupted.{token}"),
            (self.git_dir, f"{self.git_dir.name}.interrupted.{token}"),
        )

    def _read_state(self) -> tuple[str, str] | None:
        try:
            raw = read_small_json(self.state_path)
        except FileNotFoundError:
            return None
        if not isinstance(raw, dict):
            raise ArchiveError("portable restore transaction is invalid")
        token = raw.get("token")
        stage = raw.get("stage")
        if (
            not isinstance(token, str)
            or _TOKEN_RE.fullmatch(token) is None
            or stage not in {"preparing", "preserved"}
        ):
            raise ArchiveError("portable restore transaction is invalid")
        return token, stage

    def _write_state(self, token: str, stage: str) -> None:
        write_json_file(self.state_path, {"token": token, "stage": stage})
