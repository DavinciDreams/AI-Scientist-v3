"""Private ownership journals and a durable fence for managed GPU launchers."""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .gpu_owner_process import ProcessIdentity, ProcessIdentityError, current_identity

MAX_JOURNAL_BYTES = 4 * 1024 * 1024
MAX_LAUNCHES = 4096
TOKEN_PATTERN = re.compile(r"[a-f0-9]{32}\Z")
CONTAINER_ID_PATTERN = re.compile(r"[a-f0-9]{64}\Z")
NAME_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
DAEMON_PATTERN = re.compile(r"[a-zA-Z0-9:_-]{1,128}\Z")
DEVICE_PATTERN = re.compile(r"(?:GPU-[a-fA-F0-9-]+|MIG-[a-zA-Z0-9/.-]+)\Z")


class RuntimeJournalError(RuntimeError):
    """Ownership or its durable launch fence could not be established."""


@dataclass
class LaunchRecord:
    name: str
    supervisor: ProcessIdentity
    creator: ProcessIdentity | None = None
    starter: ProcessIdentity | None = None
    container_id: str | None = None
    create_finished: bool = False
    create_command_sha256: str | None = None
    start_command_sha256: str | None = None

    def fields(self) -> dict[str, object]:
        return {
            "name": self.name,
            "supervisor": self.supervisor.fields(),
            "creator": None if self.creator is None else self.creator.fields(),
            "starter": None if self.starter is None else self.starter.fields(),
            "container_id": self.container_id,
            "create_finished": self.create_finished,
            "create_command_sha256": self.create_command_sha256,
            "start_command_sha256": self.start_command_sha256,
        }


@dataclass
class RuntimeRecord:
    token: str
    owner: ProcessIdentity
    devices: tuple[str, ...]
    state: str = "active"
    daemon_id: str | None = None
    launches: dict[str, LaunchRecord] = field(default_factory=dict)

    def fields(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "token": self.token,
            "owner": self.owner.fields(),
            "devices": list(self.devices),
            "state": self.state,
            "daemon_id": self.daemon_id,
            "launches": {key: launch.fields() for key, launch in self.launches.items()},
        }


def _match(value: object, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RuntimeJournalError("GPU ownership record has an invalid identifier")
    return value


def _identity(value: object) -> ProcessIdentity:
    try:
        identity = ProcessIdentity.from_fields(value)
    except (ProcessIdentityError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeJournalError("GPU ownership process identity is invalid") from exc
    if identity.uid != os.getuid():
        raise RuntimeJournalError("GPU ownership belongs to another user")
    return identity


def _decode(data: bytes, token: str) -> RuntimeRecord:
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RuntimeJournalError("GPU ownership record is unreadable") from exc
    if (
        not isinstance(value, dict) or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.keys() != {"schema_version", "token", "owner", "devices", "state", "daemon_id", "launches"}
    ):
        raise RuntimeJournalError("GPU ownership schema is invalid")
    if value.get("token") != token or value.get("state") not in ("active", "revoked", "recovered"):
        raise RuntimeJournalError("GPU ownership binding or state is invalid")
    devices = value.get("devices")
    if not isinstance(devices, list) or not 1 <= len(devices) <= 16:
        raise RuntimeJournalError("GPU ownership device set is invalid")
    for device in devices:
        if len(_match(device, DEVICE_PATTERN)) > 160:
            raise RuntimeJournalError("GPU ownership device identifier exceeds limits")
    if len({device.casefold() for device in devices}) != len(devices):
        raise RuntimeJournalError("GPU ownership device set contains duplicates")
    owner = _identity(value.get("owner"))
    daemon_id = value.get("daemon_id")
    if daemon_id is not None:
        daemon_id = _match(daemon_id, DAEMON_PATTERN)
    launches = value.get("launches")
    if not isinstance(launches, dict) or len(launches) > MAX_LAUNCHES:
        raise RuntimeJournalError("GPU ownership launch inventory exceeds limits")
    parsed: dict[str, LaunchRecord] = {}
    for launch_id, raw in launches.items():
        _match(launch_id, TOKEN_PATTERN)
        if (
            not isinstance(raw, dict) or type(raw.get("create_finished")) is not bool
            or raw.keys() != {
                "name", "supervisor", "creator", "starter", "container_id", "create_finished",
                "create_command_sha256", "start_command_sha256",
            }
        ):
            raise RuntimeJournalError("GPU launch receipt is invalid")
        container_id = raw.get("container_id")
        if container_id is not None:
            container_id = _match(container_id, CONTAINER_ID_PATTERN)
        command_hashes = [raw.get(key) for key in ("create_command_sha256", "start_command_sha256")]
        for digest in command_hashes:
            if digest is not None:
                _match(digest, CONTAINER_ID_PATTERN)
        identities = [
            _identity(raw.get(key)) if raw.get(key) is not None else None
            for key in ("supervisor", "creator", "starter")
        ]
        if identities[0] is None or any(
            identity is not None and identity.boot_id != owner.boot_id for identity in identities
        ):
            raise RuntimeJournalError("GPU launch process binding is invalid")
        if (
            (raw["create_finished"] and (identities[1] is None or container_id is None))
            or (container_id is not None and not raw["create_finished"])
            or (identities[1] is not None and command_hashes[0] is None)
            or (identities[2] is not None and command_hashes[1] is None)
        ):
            raise RuntimeJournalError("GPU launch phase receipt is inconsistent")
        if identities[2] is not None and (container_id is None or not raw["create_finished"]):
            raise RuntimeJournalError("GPU launch lacks an acknowledged container identity")
        parsed[launch_id] = LaunchRecord(
            _match(raw.get("name"), NAME_PATTERN), identities[0], identities[1], identities[2],
            container_id, raw["create_finished"],
            *command_hashes,
        )
    return RuntimeRecord(token, owner, tuple(devices), value["state"], daemon_id, parsed)


def _validate_metadata(metadata: os.stat_result, *, directory: bool = False) -> None:
    correct_kind = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if (
        not correct_kind or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077
        or (not directory and metadata.st_nlink != 1)
    ):
        raise RuntimeJournalError("GPU ownership storage must be private and owner-controlled")


def _open_root(root: Path) -> int:
    if not root.is_absolute():
        raise RuntimeJournalError("GPU ownership storage must be absolute")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        _validate_metadata(os.fstat(descriptor), directory=True)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _write(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        count = os.pwrite(descriptor, data[offset:], offset)
        if count <= 0 or count > len(data) - offset:
            raise RuntimeJournalError("GPU ownership write did not progress")
        offset += count
    os.ftruncate(descriptor, len(data))
    os.fsync(descriptor)


@dataclass
class Journal:
    record: RuntimeRecord
    root: Path
    root_fd: int
    descriptor: int

    def revalidate(self) -> None:
        pairs = (
            (os.fstat(self.root_fd), os.stat(self.root, follow_symlinks=False), True),
            (os.fstat(self.descriptor), os.stat(
                f"runtime-{self.record.token}.json", dir_fd=self.root_fd, follow_symlinks=False,
            ), False),
        )
        for opened, linked, directory in pairs:
            _validate_metadata(opened, directory=directory)
            _validate_metadata(linked, directory=directory)
            if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
                raise RuntimeJournalError("GPU ownership storage changed during use")

    def save(self) -> None:
        self.revalidate()
        data = json.dumps(self.record.fields(), sort_keys=True).encode("ascii") + b"\n"
        if len(data) > MAX_JOURNAL_BYTES:
            raise RuntimeJournalError("GPU ownership record exceeds storage limit")
        _decode(data, self.record.token)
        _write(self.descriptor, data)


def create_runtime(root: Path, devices: tuple[str, ...]) -> str:
    """Create durable ownership before exposing any managed launch command."""
    token = uuid.uuid4().hex
    root_fd = descriptor = None
    try:
        root_fd = _open_root(root)
        descriptor = os.open(
            f"runtime-{token}.json", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            | os.O_CLOEXEC, 0o600, dir_fd=root_fd,
        )
        record = RuntimeRecord(token, current_identity(), devices)
        Journal(record, root, root_fd, descriptor).save()
        os.fsync(root_fd)
        return token
    except (OSError, ProcessIdentityError) as exc:
        raise RuntimeJournalError("GPU ownership could not be persisted") from exc
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            if root_fd is not None:
                os.close(root_fd)


@contextmanager
def open_runtime(root: Path, token: str, *, deadline: float) -> Iterator[Journal]:
    """Lock the stable journal inode; callers explicitly save before external actions."""
    _match(token, TOKEN_PATTERN)
    if not math.isfinite(deadline):
        raise RuntimeJournalError("GPU ownership deadline is invalid")
    root_fd = descriptor = None
    try:
        root_fd = _open_root(root)
        descriptor = os.open(
            f"runtime-{token}.json", os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=root_fd,
        )
        _validate_metadata(os.fstat(descriptor))
        while True:
            if time.monotonic() >= deadline:
                raise RuntimeJournalError("GPU ownership fence remained busy until deadline")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        data = os.pread(descriptor, MAX_JOURNAL_BYTES + 1, 0)
        if len(data) > MAX_JOURNAL_BYTES or len(data) != os.fstat(descriptor).st_size:
            raise RuntimeJournalError("GPU ownership record exceeds storage limit")
        journal = Journal(_decode(data, token), root, root_fd, descriptor)
        journal.revalidate()
        yield journal
    except OSError as exc:
        raise RuntimeJournalError("GPU ownership storage operation failed") from exc
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            if root_fd is not None:
                os.close(root_fd)
