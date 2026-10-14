"""Docker image resolution helpers shared by orchestration phases."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_LEASE_ROLE_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,31}\Z")
_LEASE_REPOSITORY = "agentic-scientist/pipeline-lease"
_DOCKER_COMMAND_TIMEOUT_SECONDS = 30

ImageCommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class ContainerImageResolutionError(ValueError):
    """Raised when an image reference cannot resolve to the requested image."""


class ContainerImageInfrastructureError(ValueError):
    """Raised when Docker cannot reliably inspect an image reference."""


@dataclass(slots=True)
class DockerImageLease:
    """Temporary local tag that keeps one immutable image reachable."""

    image_id: str
    tag: str
    _run: ImageCommandRunner = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    def verify(self) -> None:
        current_id = _inspect_optional_image(self.tag, run=self._run)
        if current_id is None:
            raise ValueError("Docker image lease tag is unavailable")
        if current_id != self.image_id:
            raise ValueError("Docker image lease tag points to a different image")

    def release(self) -> None:
        if self._released:
            return
        _remove_owned_tag(self.tag, self.image_id, run=self._run)
        self._released = True


def acquire_image_lease(
    image_reference: str,
    *,
    lease_scope: str,
    role: str,
    run: ImageCommandRunner = subprocess.run,
) -> DockerImageLease:
    """Pin an image ID behind a private, invocation-owned run-scoped tag."""
    if _LEASE_ROLE_PATTERN.fullmatch(role) is None:
        raise ValueError(f"invalid Docker image lease role: {role!r}")
    image_id = resolve_immutable_image(image_reference, run=run)
    scope_digest = hashlib.sha256(os.fsencode(lease_scope)).hexdigest()[:16]
    tag = (
        f"{_LEASE_REPOSITORY}:{os.getuid()}-{scope_digest}-{role}-"
        f"{secrets.token_hex(16)}"
    )
    if _inspect_optional_image(tag, run=run) is not None:
        raise ValueError("generated Docker image lease tag already exists")
    try:
        tagged = run(
            ["docker", "image", "tag", image_id, tag],
            capture_output=True,
            text=True,
            check=False,
            timeout=_DOCKER_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Docker image lease acquisition timed out") from exc
    except OSError as exc:
        raise ValueError(f"could not acquire Docker image lease: {exc}") from exc
    if tagged.returncode != 0:
        detail = tagged.stderr.strip() or "docker image tag failed"
        raise ValueError(f"could not acquire Docker image lease: {detail[:512]}")
    return DockerImageLease(image_id=image_id, tag=tag, _run=run)


def _remove_owned_tag(
    tag: str,
    image_id: str,
    *,
    run: ImageCommandRunner,
) -> None:
    current_id = _inspect_optional_image(tag, run=run)
    if current_id is None:
        return
    if current_id != image_id:
        raise ValueError("Docker image lease tag points to a different image")
    try:
        result = run(
            ["docker", "image", "rm", tag],
            capture_output=True,
            text=True,
            check=False,
            timeout=_DOCKER_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Docker image lease cleanup timed out") from exc
    except OSError as exc:
        raise ValueError(f"could not release Docker image lease: {exc}") from exc
    if result.returncode != 0 and "no such image" not in result.stderr.lower():
        detail = result.stderr.strip() or "docker image rm failed"
        raise ValueError(f"could not release Docker image lease: {detail[:512]}")


def _inspect_optional_image(
    image_reference: str,
    *,
    run: ImageCommandRunner,
) -> str | None:
    try:
        result = run(
            ["docker", "image", "inspect", "--format={{.Id}}", image_reference],
            capture_output=True,
            text=True,
            check=False,
            timeout=_DOCKER_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Docker image inspection timed out") from exc
    except OSError as exc:
        raise ValueError(f"Docker image inspection failed: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "docker image inspect failed"
        if "no such image" in detail.lower():
            return None
        raise ValueError(f"Docker image inspection failed: {detail[:512]}")
    image_id = result.stdout.strip()
    if _IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise ValueError(f"Docker returned an invalid image ID for lease tag: {image_id!r}")
    return image_id


def resolve_immutable_image(
    image_reference: str,
    *,
    run: ImageCommandRunner = subprocess.run,
) -> str:
    """Verify a local Docker image and return its immutable image ID."""
    image_reference = image_reference.strip()
    if not image_reference:
        raise ContainerImageResolutionError("Docker image reference is required")
    try:
        result = run(
            ["docker", "image", "inspect", "--format={{.Id}}", image_reference],
            capture_output=True,
            text=True,
            check=False,
            timeout=_DOCKER_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContainerImageInfrastructureError("Docker image inspection timed out") from exc
    except OSError as exc:
        raise ContainerImageInfrastructureError(
            f"Docker image inspection failed: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "docker image inspect failed"
        error_type = (
            ContainerImageResolutionError
            if "no such image" in detail.lower()
            else ContainerImageInfrastructureError
        )
        raise error_type(f"Docker image {image_reference!r} is unavailable: {detail}")
    image_id = result.stdout.strip()
    if _IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise ContainerImageInfrastructureError(
            f"Docker returned an invalid image ID for {image_reference!r}: {image_id!r}"
        )
    if _IMAGE_ID_PATTERN.fullmatch(image_reference) and image_id != image_reference:
        raise ContainerImageResolutionError(
            f"Docker resolved {image_reference!r} to unexpected image ID {image_id!r}"
        )
    return image_id
