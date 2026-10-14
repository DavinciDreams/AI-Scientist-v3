"""Private agent scratch mounts; host controls remain outside the mounted leaf."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any


LABEL = "ai.agentic-scientist.temp-path"


def validate_directory(path: Path) -> None:
    if not path.is_absolute() or any(c in str(path) for c in ",\n\r"):
        raise ValueError("container scratch needs an absolute path without commas or newlines")
    for part in (*reversed(path.parents), path):
        metadata = part.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("container scratch paths cannot contain symlinks")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("container scratch paths must contain only directories")
        mode = stat.S_IMODE(metadata.st_mode)
        if metadata.st_uid not in (0, os.getuid()) or (
            mode & 0o022 and not mode & stat.S_ISVTX
        ):
            raise ValueError("container scratch has an unsafe writable or foreign-owned ancestor")
    if path != path.resolve():
        raise ValueError("container scratch paths must be normalized")
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("container scratch must be an owned directory")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("container scratch directories must have permissions 0700")


def create(root: Path) -> Path:
    validate_directory(root)
    return Path(tempfile.mkdtemp(prefix="agent-tmp-", dir=root))


def validate_leaf(root: Path, leaf: Path) -> None:
    validate_directory(root)
    if leaf.parent != root or not leaf.name.startswith("agent-tmp-"):
        raise ValueError("container scratch is outside the configured private root")
    validate_directory(leaf)


def verify(root: Path, name: str, details: dict[str, Any]) -> Path:
    config = details.get("Config") or {}
    labels = config.get("Labels") if isinstance(config, dict) else None
    raw = labels.get(LABEL) if isinstance(labels, dict) else None
    if not isinstance(raw, str) or details.get("Name") != f"/{name}":
        raise ValueError("existing container has no matching scratch identity; restart explicitly")
    leaf = Path(raw)
    validate_leaf(root, leaf)
    mounts = [m for m in details.get("Mounts", []) if m.get("Destination") == "/agent-tmp"]
    if len(mounts) != 1 or not (
        mounts[0].get("Type") == "bind"
        and mounts[0].get("Source") == str(leaf)
        and mounts[0].get("RW") is True
    ):
        raise ValueError("existing container scratch mount does not match its private identity")
    return leaf


def inspect_container(name: str) -> dict[str, Any] | None:
    result = subprocess.run(
        ["docker", "inspect", "--type", "container", name],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if result.returncode:
        if "No such container:" in result.stderr or "No such object:" in result.stderr:
            return None
        raise ValueError("cannot confirm container scratch lifecycle: Docker inspect failed")
    details = json.loads(result.stdout)
    if not isinstance(details, list) or len(details) != 1 or not isinstance(details[0], dict):
        raise ValueError("cannot confirm container scratch lifecycle: malformed Docker inspect")
    return details[0]


def cleanup(root: Path, name: str, leaf: Path) -> bool:
    validate_leaf(root, leaf)
    if inspect_container(name) is not None:
        return False
    if not shutil.rmtree.avoids_symlink_attacks:
        return False
    shutil.rmtree(leaf)
    return True


def main() -> int:
    try:
        root = Path(os.environ["AGENTIC_CONTAINER_TEMP_ROOT"])
        action = sys.argv[1]
        if action == "create":
            print(create(root))
        elif action == "verify":
            name = sys.argv[2]
            print(verify(root, name, inspect_container(name) or {}))
        elif action == "cleanup":
            if not cleanup(root, sys.argv[2], Path(sys.argv[3])):
                print("[container_agent] retaining scratch: container still present", file=sys.stderr)
        else:
            raise ValueError("unknown container scratch action")
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"[container_agent] scratch: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
