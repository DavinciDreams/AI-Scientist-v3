"""Short private candidate socket paths outside login-managed IPC storage."""
from pathlib import Path
import os
import stat


def broker_root() -> Path:
    return Path("/tmp") / f"agentic-scientist-broker-{os.getuid()}"


def legacy_broker_root() -> Path:
    return Path("/dev/shm") / f"agentic-scientist-broker-{os.getuid()}"


def _validate_root(root: Path) -> None:
    metadata = root.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700):
        raise ValueError("unsafe broker recovery root")


def ensure_broker_root(root: Path) -> Path:
    parent = Path("/tmp").lstat()
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != 0
            or (parent.st_mode & 0o022 and not parent.st_mode & stat.S_ISVTX)):
        raise ValueError("unsafe broker temporary directory")
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _validate_root(root)
    return root


def reject_legacy_broker_directory(directory: Path) -> None:
    """Never abandon an interrupted old runtime when adopting the new location."""
    legacy_root = legacy_broker_root()
    try:
        _validate_root(legacy_root)
    except FileNotFoundError:
        return
    legacy = legacy_root / directory.name
    try:
        legacy.lstat()
    except FileNotFoundError:
        return
    raise ValueError(
        f"legacy broker recovery metadata remains at {legacy}; "
        "verify and clean up the old runtime before retrying"
    )
