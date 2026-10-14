"""Recover trusted scalar settings needed to resume a durable pipeline run."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import stat

import yaml

from src.pipeline.safe_fs import UnsafeGluePathError, ensure_directory, open_directory

_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_MAX_CONFIG_BYTES = 1024 * 1024
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
RESUME_POLICY_FILENAME = ".agentic_resume_policy.json"
TASK_HANDOFF_FILENAME = ".agentic_task_handoff"
INVESTIGATION_CLAIM_FILENAME = ".agentic_investigation_claim"
_RESUME_POLICY_VERSION = 1


class ResumeConfigError(RuntimeError):
    """Raised when durable resume metadata cannot be trusted or interpreted."""


def _read_bounded(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(descriptor, min(64 * 1024, _MAX_CONFIG_BYTES + 1 - total)):
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_CONFIG_BYTES:
            raise ResumeConfigError("durable Hydra config is too large")
    return b"".join(chunks)


def _read_owned_file_at(directory_fd: int, filename: str, *, role: str) -> bytes:
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ResumeConfigError(f"{role} is unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_size > _MAX_CONFIG_BYTES
        ):
            raise ResumeConfigError(f"{role} is unsafe")
        payload = _read_bounded(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    compared_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(payload) != before.st_size or any(
        getattr(before, field) != getattr(after, field) for field in compared_fields
    ):
        raise ResumeConfigError(f"{role} changed while reading")
    return payload


def _read_legacy_config_at(run_fd: int) -> Mapping[str, object]:
    try:
        hydra_fd = os.open(".hydra", _DIRECTORY_FLAGS, dir_fd=run_fd)
        try:
            payload = _read_owned_file_at(
                hydra_fd,
                "config.yaml",
                role="durable Hydra config",
            )
        finally:
            os.close(hydra_fd)
    except (FileNotFoundError, OSError) as error:
        raise ResumeConfigError("durable Hydra config is missing or unsafe") from error
    try:
        config = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise ResumeConfigError("durable Hydra config is invalid YAML") from error
    if not isinstance(config, Mapping):
        raise ResumeConfigError("durable Hydra config must be a mapping")
    return config


def _validated_cycle_budget(value: object, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000:
        raise ResumeConfigError(f"{role} is invalid")
    return value


def _read_resume_policy_at(run_fd: int) -> int:
    payload = _read_owned_file_at(
        run_fd,
        RESUME_POLICY_FILENAME,
        role="trusted resume policy",
    )
    try:
        policy = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ResumeConfigError("trusted resume policy is invalid JSON") from error
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "max_idea_cycles"}:
        raise ResumeConfigError("trusted resume policy has an invalid schema")
    if policy["schema_version"] != _RESUME_POLICY_VERSION:
        raise ResumeConfigError("trusted resume policy has an unsupported version")
    return _validated_cycle_budget(
        policy["max_idea_cycles"],
        role="trusted resume policy cycle budget",
    )


def _entry_exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def read_run_idea_cycle_budget(run_dir: Path) -> int | None:
    """Return the idea-cycle budget a run directory recorded, or None if it recorded none.

    The trusted resume policy wins. Runs that predate it may carry a legacy
    envelope-level Hydra config; a new-layout run without a policy, or an empty
    run directory, recorded nothing. Unsafe or corrupt metadata is an error.
    """
    try:
        with open_directory(run_dir, role="resume run") as run_fd:
            try:
                return _read_resume_policy_at(run_fd)
            except FileNotFoundError:
                pass
            if _entry_exists_at(run_fd, "investigation") or not _entry_exists_at(
                run_fd, ".hydra"
            ):
                return None
            config = _read_legacy_config_at(run_fd)
    except (FileNotFoundError, UnsafeGluePathError, OSError) as error:
        raise ResumeConfigError("resume run is missing or unsafe") from error
    return _legacy_cycle_budget(config)


def _read_config(run_dir: Path) -> Mapping[str, object] | int:
    try:
        with open_directory(run_dir, role="durable resume run") as run_fd:
            try:
                return _read_resume_policy_at(run_fd)
            except FileNotFoundError:
                pass
            try:
                os.stat("investigation", dir_fd=run_fd, follow_symlinks=False)
            except FileNotFoundError:
                return _read_legacy_config_at(run_fd)
            raise ResumeConfigError("trusted resume policy is missing for new-layout run")
    except (FileNotFoundError, UnsafeGluePathError, OSError) as error:
        raise ResumeConfigError("durable resume run is missing or unsafe") from error


def write_resume_policy(run_dir: Path, *, max_idea_cycles: int) -> None:
    """Publish the immutable resume budget outside the agent-writable workspace."""
    budget = _validated_cycle_budget(
        max_idea_cycles,
        role="trusted resume policy cycle budget",
    )
    payload = json.dumps(
        {"schema_version": _RESUME_POLICY_VERSION, "max_idea_cycles": budget},
        separators=(",", ":"),
    ).encode() + b"\n"
    try:
        with ensure_directory(run_dir, role="resume policy run envelope") as run_fd:
            run_stat = os.fstat(run_fd)
            if run_stat.st_uid != os.getuid():
                raise ResumeConfigError("resume policy run envelope is not owned by this user")
            if stat.S_IMODE(run_stat.st_mode) != 0o700:
                os.fchmod(run_fd, 0o700)
            try:
                descriptor = os.open(
                    RESUME_POLICY_FILENAME,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=run_fd,
                )
            except FileExistsError:
                if _read_resume_policy_at(run_fd) != budget:
                    raise ResumeConfigError("trusted resume policy does not match this run")
            else:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.fsync(run_fd)
    except ResumeConfigError:
        raise
    except (UnsafeGluePathError, OSError) as error:
        raise ResumeConfigError("could not safely publish trusted resume policy") from error


def _create_empty_marker_at(run_fd: int, filename: str, *, role: str) -> None:
    try:
        marker_fd = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=run_fd,
        )
    except FileExistsError as error:
        raise ResumeConfigError(f"{role} already exists") from error
    except OSError as error:
        raise ResumeConfigError(f"could not safely create {role}") from error
    try:
        os.fsync(marker_fd)
    finally:
        os.close(marker_fd)
    os.fsync(run_fd)


def publish_task_handoff(run_dir: Path) -> None:
    """Record successful task-only completion outside the agent workspace."""
    try:
        with open_directory(run_dir, role="task handoff run envelope") as run_fd:
            if os.fstat(run_fd).st_uid != os.getuid():
                raise ResumeConfigError("task handoff run envelope is not owned by this user")
            _read_resume_policy_at(run_fd)
            _create_empty_marker_at(
                run_fd,
                TASK_HANDOFF_FILENAME,
                role="trusted task handoff",
            )
    except ResumeConfigError:
        raise
    except (FileNotFoundError, UnsafeGluePathError, OSError) as error:
        raise ResumeConfigError("could not safely publish trusted task handoff") from error


def claim_task_handoff_for_investigation(
    run_dir: Path,
    *,
    max_idea_cycles: int,
) -> None:
    """Atomically consume the one supported existing-workspace handoff."""
    expected_budget = _validated_cycle_budget(
        max_idea_cycles,
        role="trusted resume policy cycle budget",
    )
    try:
        with open_directory(run_dir, role="investigation claim run envelope") as run_fd:
            if os.fstat(run_fd).st_uid != os.getuid():
                raise ResumeConfigError("investigation claim run envelope is not owned by this user")
            if _read_resume_policy_at(run_fd) != expected_budget:
                raise ResumeConfigError("trusted resume policy does not match this run")
            handoff = _read_owned_file_at(
                run_fd,
                TASK_HANDOFF_FILENAME,
                role="trusted task handoff",
            )
            if handoff:
                raise ResumeConfigError("trusted task handoff is invalid")
            _create_empty_marker_at(
                run_fd,
                INVESTIGATION_CLAIM_FILENAME,
                role="trusted investigation claim",
            )
    except ResumeConfigError:
        raise
    except FileNotFoundError as error:
        raise ResumeConfigError("trusted task handoff is missing") from error
    except (UnsafeGluePathError, OSError) as error:
        raise ResumeConfigError("could not safely claim trusted task handoff") from error


def _cycle_value(config: Mapping[str, object], section: str) -> int | None:
    section_value = config.get(section)
    if not isinstance(section_value, Mapping):
        return None
    value = section_value.get("max_idea_cycles")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000:
        raise ResumeConfigError("durable idea-cycle budget is invalid")
    return value


def read_idea_cycle_budget(
    durable_results_root: Path,
    task: str,
    agent: str,
    run_id: str,
) -> int:
    """Return the original idea-cycle budget for one durable run."""
    if not durable_results_root.is_absolute():
        raise ResumeConfigError("durable results root must be absolute")
    for label, component_value in (
        ("task", task),
        ("agent", agent),
        ("run ID", run_id),
    ):
        if not _COMPONENT.fullmatch(component_value):
            raise ResumeConfigError(f"invalid {label}")
    run_dir = durable_results_root / task / agent / run_id
    config = _read_config(run_dir)
    if isinstance(config, int):
        return config
    return _legacy_cycle_budget(config)


def _legacy_cycle_budget(config: Mapping[str, object]) -> int:
    values = {
        budget
        for section in ("investigation", "execution")
        if (budget := _cycle_value(config, section)) is not None
    }
    if len(values) != 1:
        raise ResumeConfigError("durable idea-cycle budget is missing or inconsistent")
    return values.pop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--durable-results-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        budget = read_idea_cycle_budget(
            args.durable_results_root,
            args.task,
            args.agent,
            args.run_id,
        )
    except ResumeConfigError as error:
        parser.exit(2, f"resume_config: {error}\n")
    print(budget)


if __name__ == "__main__":
    main()
