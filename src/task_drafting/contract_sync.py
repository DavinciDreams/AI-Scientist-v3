"""Mechanical synchronization of already-declared task-contract digests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from src.investigation.task_contract import (
    MAX_TASK_CONTRACT_BYTES,
    TaskContractError,
    safe_contract_relative_path,
)


_PACKAGE_COMPONENT = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_PINNED_VERSION = (
    r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*"
    r"(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?"
    r"(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?"
)
_PINNED_REQUIREMENT = re.compile(
    rf"^(?P<name>{_PACKAGE_COMPONENT}"
    rf"(?:\[{_PACKAGE_COMPONENT}(?:,{_PACKAGE_COMPONENT})*\])?)"
    rf"==(?P<version>{_PINNED_VERSION})$"
)
_PACKAGE_SEPARATOR = re.compile(r"[-_.]+")
_MAX_DEPENDENCIES = 64
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "canonical_task",
    "method",
    "candidate",
    "evaluators",
    "resources",
    "data",
    "metrics",
    "upstream",
    "provenance",
    "compute",
    "protocol",
    "feedback",
    "publication",
}


class ContractSyncError(ValueError):
    """Raised when digest synchronization cannot be performed without semantic edits."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractSyncError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_contract(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractSyncError("task contract must be a regular file")
    if path.stat().st_size > MAX_TASK_CONTRACT_BYTES:
        raise ContractSyncError("task contract is too large")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicates
        )
    except ContractSyncError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractSyncError("task contract is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ContractSyncError("task contract must be an object")
    missing = _TOP_LEVEL_FIELDS - payload.keys()
    unexpected = payload.keys() - _TOP_LEVEL_FIELDS
    if missing or unexpected:
        raise ContractSyncError(
            f"task contract fields mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if payload.get("schema_version") != 1:
        raise ContractSyncError("task contract schema_version must be 1")
    return payload


def _object(value: object, label: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ContractSyncError(f"{label} must contain exactly {sorted(fields)}")
    return value


def _safe_path(task_dir: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractSyncError(f"{label} must be a safe relative path")
    try:
        relative = PurePosixPath(safe_contract_relative_path(value, label))
    except TaskContractError as exc:
        raise ContractSyncError(str(exc)) from exc
    current = task_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ContractSyncError(f"{label} must not contain symlinks")
    if not current.is_file():
        raise ContractSyncError(f"{label} must identify an existing regular file")
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def _normalized_dependency(name: str, version: str) -> tuple[str, str]:
    base_name, separator, extras_value = name.partition("[")
    normalized_name = _PACKAGE_SEPARATOR.sub("-", base_name).lower()
    if separator:
        extras = extras_value.removesuffix("]").split(",")
        normalized_extras = sorted({_PACKAGE_SEPARATOR.sub("-", extra).lower() for extra in extras})
        normalized_name += f"[{','.join(normalized_extras)}]"
    return normalized_name, version


def _declared_dependencies(payload: dict[str, Any]) -> list[dict[str, str]]:
    provenance = _object(payload["provenance"], "provenance", {"dependencies", "artifacts"})
    dependencies = provenance["dependencies"]
    if not isinstance(dependencies, list) or len(dependencies) > _MAX_DEPENDENCIES:
        raise ContractSyncError("provenance.dependencies must be a bounded array")
    parsed: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(dependencies):
        dependency = _object(
            value,
            f"provenance.dependencies[{index}]",
            {"name", "version"},
        )
        name = dependency["name"]
        version = dependency["version"]
        if (
            not isinstance(name, str)
            or not isinstance(version, str)
            or _PINNED_REQUIREMENT.fullmatch(f"{name}=={version}") is None
        ):
            raise ContractSyncError("provenance.dependencies must contain pinned packages")
        key = _normalized_dependency(name, version)
        if key in seen:
            raise ContractSyncError("provenance.dependencies must not contain duplicates")
        seen.add(key)
        parsed.append({"name": name, "version": version})
    return parsed


def _requirement_keys(path: Path, task_dir: Path) -> set[tuple[str, str]]:
    if path.parent.resolve(strict=True) != task_dir or path.is_symlink() or not path.is_file():
        raise ContractSyncError("requirements.txt must be a regular scaffold file")
    if path.stat().st_size > MAX_TASK_CONTRACT_BYTES:
        raise ContractSyncError("requirements.txt is too large")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractSyncError("requirements.txt must be readable UTF-8") from exc
    keys: set[tuple[str, str]] = set()
    rows = 0
    for line in lines:
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        rows += 1
        match = _PINNED_REQUIREMENT.fullmatch(requirement)
        if rows > _MAX_DEPENDENCIES or match is None:
            raise ContractSyncError("requirements.txt must contain bounded pinned packages")
        key = _normalized_dependency(match["name"], match["version"])
        if key in keys:
            raise ContractSyncError("requirements.txt must not contain duplicate packages")
        keys.add(key)
    return keys


def _refresh_declared_digests(payload: dict[str, Any], task_dir: Path, expected_task: Path) -> bool:
    canonical = _object(payload["canonical_task"], "canonical_task", {"path", "sha256"})
    if canonical["path"] != "task.md":
        raise ContractSyncError("canonical_task.path must remain task.md")
    provenance = _object(payload["provenance"], "provenance", {"dependencies", "artifacts"})
    artifacts = provenance["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ContractSyncError("provenance.artifacts must be a non-empty array")
    changed = False
    task_digest = _sha256(expected_task)
    if canonical["sha256"] != task_digest:
        canonical["sha256"] = task_digest
        changed = True
    seen_paths: set[str] = set()
    for index, value in enumerate(artifacts):
        artifact = _object(
            value,
            f"provenance.artifacts[{index}]",
            {"role", "path", "sha256", "source"},
        )
        path_value = artifact["path"]
        if not isinstance(path_value, str) or path_value in seen_paths:
            raise ContractSyncError("provenance artifact paths must be unique strings")
        seen_paths.add(path_value)
        declared = _safe_path(task_dir, path_value, f"provenance.artifacts[{index}].path")
        digest = _sha256(declared)
        if artifact["sha256"] != digest:
            artifact["sha256"] = digest
            changed = True
    return changed


def synchronize_declared_digests(contract_path: Path, *, canonical_task: Path) -> bool:
    """Refresh only canonical/artifact SHA fields for paths already declared in the contract."""
    task_dir = contract_path.parent.resolve(strict=True)
    expected_task = task_dir / "task.md"
    if canonical_task.resolve(strict=True) != expected_task.resolve(strict=True):
        raise ContractSyncError("canonical task must be the scaffold task.md")
    payload = _load_contract(contract_path)
    changed = _refresh_declared_digests(payload, task_dir, expected_task)
    if changed:
        _atomic_write_json(contract_path, payload)
    return changed


def declared_dependencies_match_requirements(
    contract_path: Path, *, requirements_path: Path
) -> bool:
    """Return whether the strict requirements lock matches declared dependencies."""
    task_dir = contract_path.parent.resolve(strict=True)
    payload = _load_contract(contract_path)
    declared = {
        _normalized_dependency(dependency["name"], dependency["version"])
        for dependency in _declared_dependencies(payload)
    }
    if requirements_path.exists():
        required = _requirement_keys(requirements_path, task_dir)
    else:
        required = set()
    return declared == required


def retained_declared_dependencies(
    requirements_path: Path, *, allowed_dependencies: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Derive the unique declared dependency subset retained by a repaired lock."""
    task_dir = requirements_path.parent.resolve(strict=True)
    allowed = _declared_dependencies(
        {"provenance": {"dependencies": allowed_dependencies, "artifacts": []}}
    )
    allowed_keys = {
        _normalized_dependency(dependency["name"], dependency["version"]) for dependency in allowed
    }
    retained_keys = _requirement_keys(requirements_path, task_dir)
    if not retained_keys.issubset(allowed_keys):
        raise ContractSyncError("repair may only remove pinned dependencies")
    return [
        dependency
        for dependency in allowed
        if _normalized_dependency(dependency["name"], dependency["version"]) in retained_keys
    ]


def synchronize_declared_dependency_removals(
    contract_path: Path,
    *,
    requirements_path: Path,
    canonical_task: Path,
    allowed_dependencies: list[dict[str, str]],
) -> bool:
    """Remove only previously declared pins absent from the repaired requirements lock."""
    task_dir = contract_path.parent.resolve(strict=True)
    expected_task = task_dir / "task.md"
    if canonical_task.resolve(strict=True) != expected_task.resolve(strict=True):
        raise ContractSyncError("canonical task must be the scaffold task.md")
    payload = _load_contract(contract_path)
    current_dependencies = _declared_dependencies(payload)
    retained_dependencies = retained_declared_dependencies(
        requirements_path,
        allowed_dependencies=allowed_dependencies,
    )
    if current_dependencies not in (allowed_dependencies, retained_dependencies):
        raise ContractSyncError("repair changed declared dependencies directly")
    changed = retained_dependencies != current_dependencies
    payload["provenance"]["dependencies"] = retained_dependencies
    changed = _refresh_declared_digests(payload, task_dir, expected_task) or changed
    if changed:
        _atomic_write_json(contract_path, payload)
    return changed
