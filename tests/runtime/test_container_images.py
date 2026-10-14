"""Immutable image resolution and private lease tags (src.investigation.container_images)."""

from __future__ import annotations

import re
import subprocess
from typing import Any, Callable

import pytest

from src.investigation.container_images import (
    ContainerImageInfrastructureError,
    ContainerImageResolutionError,
    acquire_image_lease,
    resolve_immutable_image,
)

IMAGE_ID = "sha256:" + "a" * 64


def _time_out(*args, **kwargs):
    del args, kwargs
    raise subprocess.TimeoutExpired(["docker"], 30)


def _missing(command, **kwargs):
    del kwargs
    return subprocess.CompletedProcess(command, 1, "", "No such image")


RESOLUTION_FAILURE_CASES: list[tuple[str, Callable, type[Exception]]] = [
    ("transient-inspection-failure", _time_out, ContainerImageInfrastructureError),
    ("missing-image", _missing, ContainerImageResolutionError),
]


def test_image_resolution_types_failures() -> None:
    for _name, run, error in RESOLUTION_FAILURE_CASES:
        with pytest.raises(error):
            resolve_immutable_image("agent:latest", run=run)


class DockerRunner:
    def __init__(self, *, remove_error: str | None = None, remove_timeout: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.images: dict[str, str] = {}
        self.remove_error = remove_error
        self.remove_timeout = remove_timeout
        self.inspect_error: str | None = None

    def __call__(self, command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        if command[:4] == ["docker", "image", "inspect", "--format={{.Id}}"]:
            if self.inspect_error is not None:
                return subprocess.CompletedProcess(command, 1, "", self.inspect_error)
            reference = command[4]
            if reference == "agent:latest":
                return subprocess.CompletedProcess(command, 0, IMAGE_ID + "\n", "")
            if reference not in self.images:
                return subprocess.CompletedProcess(command, 1, "", "No such image")
            return subprocess.CompletedProcess(command, 0, self.images[reference] + "\n", "")
        if command[:3] == ["docker", "image", "tag"]:
            self.images[command[4]] = command[3]
        if command[:3] == ["docker", "image", "rm"]:
            if self.remove_timeout:
                raise subprocess.TimeoutExpired(command, 30)
            if self.remove_error is not None:
                return subprocess.CompletedProcess(command, 1, "", self.remove_error)
            self.images.pop(command[3], None)
        return subprocess.CompletedProcess(command, 0, "", "")


def _lease(runner: DockerRunner, *, lease_scope: str = "run", role: str = "agent"):
    return acquire_image_lease("agent:latest", lease_scope=lease_scope, role=role, run=runner)


def test_image_lease_private_tags_release_and_verification() -> None:
    # uses a safe private tag and releases it
    runner = DockerRunner()
    lease = _lease(runner, lease_scope="/tmp/run with unsafe spaces")
    assert lease.image_id == IMAGE_ID
    assert re.fullmatch(r"agentic-scientist/pipeline-lease:\d+-[0-9a-f]{16}-agent-[0-9a-f]{32}", lease.tag)
    assert runner.calls[2] == ["docker", "image", "tag", IMAGE_ID, lease.tag]
    lease.release()
    assert runner.calls[-1] == ["docker", "image", "rm", lease.tag]

    # same scope acquires independent lease tags
    runner = DockerRunner()
    first = _lease(runner, role="sandbox")
    second = _lease(runner, role="sandbox")
    assert first.tag != second.tag
    first.release()
    second.verify()
    second.release()

    # verification rejects a disappeared tag
    runner = DockerRunner()
    lease = _lease(runner)
    runner.images.pop(lease.tag)
    with pytest.raises(ValueError, match="lease tag is unavailable"):
        lease.verify()


# (name, runner kwargs, mutate(runner, lease) or None, error match, tag remains after failed release)
RELEASE_FAILURE_CASES: list[tuple[str, dict[str, Any], Callable | None, str, bool]] = [
    (
        "refuses-to-remove-retargeted-tag",
        {},
        lambda runner, lease: runner.images.__setitem__(lease.tag, "sha256:" + "b" * 64),
        "different image",
        True,
    ),
    ("cleanup-failure-is-reported", {"remove_error": "daemon unavailable"}, None, "could not release Docker image lease", False),
    ("cleanup-timeout-is-reported", {"remove_timeout": True}, None, "timed out", False),
    (
        "daemon-failure-is-not-a-missing-tag",
        {},
        lambda runner, lease: setattr(runner, "inspect_error", "Cannot connect to the Docker daemon"),
        "Docker image inspection failed",
        True,
    ),
]


def test_image_lease_release_failures() -> None:
    for name, runner_kwargs, mutate, match, tag_remains in RELEASE_FAILURE_CASES:
        runner = DockerRunner(**runner_kwargs)
        lease = _lease(runner)
        if mutate is not None:
            mutate(runner, lease)

        with pytest.raises(ValueError, match=match):
            lease.release()
        if tag_remains:
            assert lease.tag in runner.images, name
