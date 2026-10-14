"""Fence abandoned GPU runtimes, terminate their launchers, and verify container removal."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

from .bounded_process import BoundedProcessError, BoundedProcessResult, run_bounded_process
from .gpu_owner_process import (
    ProcessIdentity, ProcessIdentityError, current_identity, process_is_alive, terminate_process,
)
from .gpu_runtime_journal import (
    CONTAINER_ID_PATTERN, DAEMON_PATTERN, Journal, LaunchRecord, RuntimeJournalError, open_runtime,
)

LOCAL_DOCKER_HOST = "unix:///var/run/docker.sock"


class RuntimeRecoveryError(RuntimeError):
    """Recovery must retain quarantine because ownership or shutdown is unverified."""


def _docker(arguments: list[str], deadline: float) -> BoundedProcessResult:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeRecoveryError("GPU recovery deadline elapsed")
    executable = shutil.which("docker")
    if executable is None:
        raise RuntimeRecoveryError("GPU recovery Docker executable is unavailable")
    environment = dict(os.environ)
    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        environment.pop(key, None)
    return run_bounded_process(
        [executable, "--host", LOCAL_DOCKER_HOST, *arguments], cwd=Path("/"), env=environment,
        timeout_seconds=min(10.0, remaining), max_output_bytes=128 * 1024,
    )


def _daemon_id(deadline: float) -> str:
    result = _docker(["info", "--format", "{{.ID}}"], deadline)
    value = result.stdout.decode("ascii", errors="replace").strip()
    if result.returncode or DAEMON_PATTERN.fullmatch(value) is None:
        raise RuntimeRecoveryError("GPU recovery Docker daemon identity is unverified")
    return value


def _absent(result: BoundedProcessResult) -> bool:
    message = result.stderr.decode("utf-8", errors="replace").lower()
    return result.returncode != 0 and ("no such container" in message or "no such object" in message)


def _inspect_container(identity: str, deadline: float) -> dict[str, object] | None:
    template = '{"id":{{json .Id}},"labels":{{json .Config.Labels}}}'
    result = _docker(["inspect", "--type", "container", "--format", template, identity], deadline)
    if _absent(result):
        return None
    try:
        value = json.loads(result.stdout)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RuntimeRecoveryError("GPU recovery container inspection is unverified") from exc
    if result.returncode or not isinstance(value, dict):
        raise RuntimeRecoveryError("GPU recovery container inspection failed")
    return value


def _remove_container(identity: str, deadline: float) -> None:
    result = _docker(["rm", "--force", identity], deadline)
    if result.returncode and not _absent(result):
        raise RuntimeRecoveryError("GPU recovery container removal failed")


def _clean_container(journal: Journal, launch_id: str, launch: LaunchRecord, deadline: float) -> None:
    if launch.creator is None:
        if launch.container_id is not None or launch.starter is not None:
            raise RuntimeRecoveryError("GPU ownership lacks its container creation process")
        return
    container = _inspect_container(launch.container_id or launch.name, deadline)
    if container is None:
        if launch.container_id is None:
            raise RuntimeRecoveryError("GPU container creation completion remains unverified")
        return
    identity = container.get("id")
    labels = container.get("labels")
    if (
        not isinstance(identity, str) or CONTAINER_ID_PATTERN.fullmatch(identity) is None
        or (launch.container_id is not None and identity != launch.container_id)
        or not isinstance(labels, dict)
        or labels.get("agentic-scientist.lease") != journal.record.token
        or labels.get("agentic-scientist.launch") != launch_id
    ):
        raise RuntimeRecoveryError("GPU recovery container ownership does not match")
    if launch.container_id is None:
        launch.container_id = identity
        launch.create_finished = True
        journal.save()
    _remove_container(identity, deadline)
    if _inspect_container(identity, deadline) is not None:
        raise RuntimeRecoveryError("GPU recovery container removal remains unverified")


def reclaim_runtime(
    root: Path, token: str, *, deadline: float, releasing_owner: ProcessIdentity | None = None,
) -> tuple[str, ...]:
    """Caller holds the complete old allocation; a live foreign owner cannot be reclaimed."""
    try:
        with open_runtime(root, token, deadline=deadline) as journal:
            record = journal.record
            if record.state == "recovered":
                return record.devices
            if releasing_owner is not None:
                if releasing_owner != current_identity() or releasing_owner != record.owner:
                    raise RuntimeRecoveryError("GPU recovery release owner does not match")
            elif process_is_alive(record.owner):
                raise RuntimeRecoveryError("GPU runtime owner is still alive")
            record.state = "revoked"
            journal.save()
            # A revoked journal fences even children not yet registered after a fork.
            # Stop supervisors first, before their registered create/start subprocesses.
            processes = [entry.supervisor for entry in record.launches.values()]
            processes += [
                identity for entry in record.launches.values()
                for identity in (entry.creator, entry.starter) if identity is not None
            ]
            for identity in dict.fromkeys(processes):
                terminate_process(identity, deadline=deadline)
            if any(process_is_alive(identity) for identity in processes):
                raise RuntimeRecoveryError("GPU launch process shutdown remains unverified")
            if any(entry.creator is not None for entry in record.launches.values()):
                if record.daemon_id is None or _daemon_id(deadline) != record.daemon_id:
                    raise RuntimeRecoveryError("GPU recovery Docker daemon differs from launch")
                for launch_id, launch in record.launches.items():
                    _clean_container(journal, launch_id, launch, deadline)
                if _daemon_id(deadline) != record.daemon_id:
                    raise RuntimeRecoveryError("GPU recovery Docker daemon changed during cleanup")
            record.state = "recovered"
            journal.save()
            return record.devices
    except (RuntimeJournalError, ProcessIdentityError, BoundedProcessError, OSError) as exc:
        raise RuntimeRecoveryError("GPU runtime shutdown could not be verified") from exc
