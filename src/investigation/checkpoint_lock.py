"""Owner-private sidecar creation and process-level checkpoint locking."""

from __future__ import annotations

import logging
import os
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .checkpoint_security import redact_sensitive_text

logger = logging.getLogger(__name__)

_LOCK_FILENAME = "checkpoint.lock"
_LOCK_POLL_SECONDS = 0.1
_LOCK_MAX_WAIT_SECONDS = 30


def ensure_private_git_dir(git_dir: Path) -> None:
    """Create a private real directory; refuse symlink or ownership substitution."""
    try:
        os.mkdir(git_dir, 0o700)
    except FileExistsError:
        metadata = git_dir.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError(f"checkpoint git dir is not a real directory: {git_dir}")
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise OSError(f"checkpoint git dir is not owner-private: {git_dir}")


@contextmanager
def checkpoint_lock(git_dir: Path, *, timeout_seconds: float) -> Iterator[bool]:
    """Serialize checkpoints against concurrent writers on the same sidecar."""
    lock_path = git_dir / _LOCK_FILENAME
    deadline = time.monotonic() + min(_LOCK_MAX_WAIT_SECONDS, max(0.0, timeout_seconds))
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if not _reclaim_stale_lock(lock_path) and time.monotonic() >= deadline:
                yield False
                return
            time.sleep(_LOCK_POLL_SECONDS)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        descriptor = None
        yield True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(
                "Checkpoint lock cleanup skipped: %s", redact_sensitive_text(str(exc))
            )


def _reclaim_stale_lock(lock_path: Path) -> bool:
    try:
        holder = int(lock_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(holder, 0)
    except ProcessLookupError:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            return False
        logger.warning("Reclaimed stale checkpoint lock from dead pid %d", holder)
        return True
    except OSError:
        return False
    return False
