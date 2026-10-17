"""Stage the source on the host-visible data mount before serving the inspector."""

import os
from pathlib import Path
import shutil


def main() -> None:
    data = Path("/data/sakanascientist")
    # The pipeline rejects scratch ancestors owned by a different unprivileged
    # user. Keep the existing host group for operator access, without group write.
    os.chown(data, os.getuid(), data.stat().st_gid)
    data.chmod(0o750)
    for ipc in (Path("/tmp/agentic-scientist-broker-0"), Path("/dev/shm/agentic-task-gates-0")):
        ipc.mkdir(mode=0o700, exist_ok=True)
        if ipc.is_symlink() or ipc.stat().st_uid != os.getuid():
            raise ValueError(f"Unsafe IPC directory: {ipc}")
        ipc.chmod(0o700)
    revision = os.environ["SOURCE_REVISION"]
    if not revision or any(c not in "0123456789abcdef-" for c in revision):
        raise ValueError("SOURCE_REVISION must be an immutable hexadecimal source identifier")
    source = data / "releases" / revision
    if not source.exists():
        staging = source.with_name(revision + "-staging")
        shutil.copytree("/opt/source", staging, ignore=shutil.ignore_patterns(".venv"))
        staging.rename(source)
    for name in ("results", "tmp", "tasks", "jobs", "secrets", "home"):
        (data / name).mkdir(parents=True, exist_ok=True)
    (data / "secrets").chmod(0o700)
    (data / "home").chmod(0o700)
    env_file = data / "secrets" / "zai.env"
    key = os.environ.get("ZAI_API_KEY", "")
    if any(c in key for c in "\r\n\x00"):
        raise ValueError("Invalid ZAI_API_KEY")
    fd = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(f"ZAI_API_KEY={key}\nPI_CODING_AGENT_PROVIDER=zai\n")
    # A fresh link selects the deployed source without deleting prior releases.
    current = data / "current"
    if current.is_symlink():
        current.unlink()
    current.symlink_to(source, target_is_directory=True)
    os.chdir(source)
    os.environ["PYTHONPATH"] = str(source)
    os.execvp("python", ["python", "deploy/coolify/server.py"])


if __name__ == "__main__":
    main()
