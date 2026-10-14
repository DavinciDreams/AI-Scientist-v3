"""Harness-enforced mutation boundary for task-scaffold repair agents."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.task_drafting.contract_sync import (
    ContractSyncError,
    declared_dependencies_match_requirements,
    retained_declared_dependencies,
    synchronize_declared_dependency_removals,
    synchronize_declared_digests,
)

from .program_spec import ProgramSpecError, load_program_spec, resolve_entrypoint
from .task_contract import MAX_TASK_CONTRACT_BYTES, TaskContractError, load_task_contract

_SCHEMA_VERSION = 6
_DEPENDENCY_JOURNAL_SCHEMA_VERSION = 1
_IMMUTABLE_CONTRACT = "immutable"
_SYNCHRONIZED_CONTRACT_DIGESTS = "synchronized-digests"
_SYNCHRONIZED_DEPENDENCY_REMOVALS = "synchronized-dependency-removals"


class TaskRepairScopeError(ValueError):
    """A repair changed data outside its harness-owned mutation boundary."""


def create_repair_snapshot(task_dir: Path, output_path: Path) -> None:
    """Record task files and immutable contract fields before a scoped repair."""
    _create_repair_snapshot(task_dir, output_path, contract_policy=_IMMUTABLE_CONTRACT)


def create_digest_synchronized_repair_snapshot(task_dir: Path, output_path: Path) -> None:
    """Record a repair scope that permits only harness-synchronized contract digests."""
    _create_repair_snapshot(
        task_dir,
        output_path,
        contract_policy=_SYNCHRONIZED_CONTRACT_DIGESTS,
    )


def create_dependency_removal_repair_snapshot(task_dir: Path, output_path: Path) -> None:
    """Record a repair scope permitting harness-owned removal of dependency declarations."""
    _create_repair_snapshot(
        task_dir,
        output_path,
        contract_policy=_SYNCHRONIZED_DEPENDENCY_REMOVALS,
    )


def _create_repair_snapshot(
    task_dir: Path,
    output_path: Path,
    *,
    contract_policy: str,
) -> None:
    files = _task_files(task_dir)
    contract = _load_contract(task_dir / "task_contract.json")
    mutable_paths = list(_mutable_paths(task_dir))
    if contract_policy in {
        _SYNCHRONIZED_CONTRACT_DIGESTS,
        _SYNCHRONIZED_DEPENDENCY_REMOVALS,
    }:
        mutable_paths.append("task_contract.json")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "task_dir": str(task_dir.resolve()),
        "files": files,
        "contract_invariants": contract,
        "contract_policy": contract_policy,
        "mutable_paths": sorted(mutable_paths),
    }
    _atomic_write_json(output_path, payload)


def validate_repair_snapshot(task_dir: Path, snapshot_path: Path) -> tuple[str, ...]:
    """Reject out-of-scope mutations and protected contract changes."""
    snapshot = _load_snapshot(snapshot_path)
    return _validate_repair_snapshot(task_dir, snapshot_path, snapshot)


def validate_dependency_removal_repair(task_dir: Path, snapshot_path: Path) -> tuple[str, ...]:
    """Apply host-authorized removal rules without rewriting the bound snapshot."""
    snapshot = _load_snapshot(snapshot_path)
    if snapshot["contract_policy"] not in {
        _SYNCHRONIZED_CONTRACT_DIGESTS, _SYNCHRONIZED_DEPENDENCY_REMOVALS,
    }:
        raise TaskRepairScopeError("dependency authorization requires a synchronized repair scope")
    snapshot["contract_policy"] = _SYNCHRONIZED_DEPENDENCY_REMOVALS
    return _validate_repair_snapshot(task_dir, snapshot_path, snapshot)


def _validate_repair_snapshot(
    task_dir: Path, snapshot_path: Path, snapshot: dict[str, Any],
) -> tuple[str, ...]:
    if snapshot["task_dir"] != str(task_dir.resolve()):
        raise TaskRepairScopeError("repair snapshot belongs to a different task directory")
    before = snapshot["files"]
    after = _task_files(task_dir)
    changed = tuple(
        sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    )
    mutable_paths = tuple(snapshot["mutable_paths"])
    disallowed = tuple(path for path in changed if path not in mutable_paths)
    if disallowed:
        raise TaskRepairScopeError("repair changed out-of-scope path(s): " + ", ".join(disallowed))
    contract = _load_contract(task_dir / "task_contract.json")
    contract_policy = snapshot["contract_policy"]
    if contract_policy == _SYNCHRONIZED_CONTRACT_DIGESTS:
        _validate_synchronized_contract_digests(snapshot["contract_invariants"], contract)
        try:
            synchronize_declared_digests(
                task_dir / "task_contract.json",
                canonical_task=task_dir / "task.md",
            )
        except ContractSyncError as exc:
            raise TaskRepairScopeError("repair left an invalid declared contract digest") from exc
        try:
            dependencies_match = declared_dependencies_match_requirements(
                task_dir / "task_contract.json",
                requirements_path=task_dir / "requirements.txt",
            )
        except ContractSyncError as exc:
            raise TaskRepairScopeError("repair left invalid dependency declarations") from exc
        if not dependencies_match:
            raise TaskRepairScopeError(
                "dependency changes require a runtime_dependencies repair diagnostic"
            )
    elif contract_policy == _SYNCHRONIZED_DEPENDENCY_REMOVALS:
        journal_path = _dependency_journal_path(snapshot_path)
        synchronized_dependencies = _load_dependency_journal(
            journal_path,
            snapshot_path=snapshot_path,
        )
        original_contract = snapshot["contract_invariants"]
        if synchronized_dependencies is None:
            _validate_synchronized_contract_digests(original_contract, contract)
            try:
                synchronized_dependencies = retained_declared_dependencies(
                    task_dir / "requirements.txt",
                    allowed_dependencies=original_contract["provenance"]["dependencies"],
                )
            except (ContractSyncError, KeyError, TypeError) as exc:
                message = str(exc)
                if "only remove pinned dependencies" in message:
                    raise TaskRepairScopeError(message) from exc
                raise TaskRepairScopeError(
                    "repair left invalid synchronized dependency removals"
                ) from exc
            _atomic_write_json(
                journal_path,
                {
                    "schema_version": _DEPENDENCY_JOURNAL_SCHEMA_VERSION,
                    "snapshot_sha256": _sha256(snapshot_path),
                    "dependencies": synchronized_dependencies,
                },
            )
        else:
            synchronized_contract = deepcopy(original_contract)
            synchronized_contract["provenance"]["dependencies"] = synchronized_dependencies
            try:
                _validate_synchronized_contract_digests(original_contract, contract)
            except TaskRepairScopeError:
                _validate_synchronized_contract_digests(synchronized_contract, contract)
        try:
            synchronize_declared_dependency_removals(
                task_dir / "task_contract.json",
                requirements_path=task_dir / "requirements.txt",
                canonical_task=task_dir / "task.md",
                allowed_dependencies=snapshot["contract_invariants"]["provenance"]["dependencies"],
            )
        except (ContractSyncError, KeyError, TypeError) as exc:
            message = str(exc)
            if "only remove pinned dependencies" in message:
                raise TaskRepairScopeError(message) from exc
            raise TaskRepairScopeError(
                "repair left invalid synchronized dependency removals"
            ) from exc
        synchronized_contract = _load_contract(task_dir / "task_contract.json")
        retained_dependencies = synchronized_contract["provenance"]["dependencies"]
        if retained_dependencies != synchronized_dependencies:
            raise TaskRepairScopeError("repair changed synchronized dependencies")
    elif contract != snapshot["contract_invariants"]:
        raise TaskRepairScopeError("repair changed immutable task-contract fields")
    return changed


def _validate_synchronized_contract_digests(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    before_semantics = deepcopy(before)
    after_semantics = deepcopy(after)
    try:
        before_semantics["canonical_task"]["sha256"] = "<digest>"
        after_semantics["canonical_task"]["sha256"] = "<digest>"
        before_artifacts = before_semantics["provenance"]["artifacts"]
        after_artifacts = after_semantics["provenance"]["artifacts"]
        for artifact in before_artifacts:
            artifact["sha256"] = "<digest>"
        for artifact in after_artifacts:
            artifact["sha256"] = "<digest>"
    except (KeyError, TypeError):
        raise TaskRepairScopeError("repair changed task-contract semantics") from None
    if _canonical_json(before_semantics) != _canonical_json(after_semantics):
        raise TaskRepairScopeError("repair changed task-contract semantics")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def contract_policy_for_smoke_diagnostic(diagnostic: str) -> str:
    """Select the narrow repair contract policy for a typed smoke diagnostic."""
    try:
        payload = json.loads(diagnostic)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    if isinstance(payload, dict) and payload.get("category") == "runtime_dependencies":
        return _SYNCHRONIZED_DEPENDENCY_REMOVALS
    return _SYNCHRONIZED_CONTRACT_DIGESTS


def _dependency_journal_path(snapshot_path: Path) -> Path:
    return snapshot_path.parent / f".{snapshot_path.name}.dependencies.json"


def _load_dependency_journal(path: Path, *, snapshot_path: Path) -> list[dict[str, str]] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_TASK_CONTRACT_BYTES:
        raise TaskRepairScopeError("unsafe dependency-removal repair journal")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskRepairScopeError("invalid dependency-removal repair journal") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "snapshot_sha256", "dependencies"}
        or payload.get("schema_version") != _DEPENDENCY_JOURNAL_SCHEMA_VERSION
        or payload.get("snapshot_sha256") != _sha256(snapshot_path)
        or not isinstance(payload.get("dependencies"), list)
    ):
        raise TaskRepairScopeError("invalid dependency-removal repair journal")
    return payload["dependencies"]


def quarantine_generated_bytecode(
    task_dir: Path,
    quarantine_dir: Path,
    manifest_path: Path | None = None,
    accepted_caches: frozenset[tuple[str, bytes]] | None = None,
) -> tuple[str, ...]:
    """Delegate verified cache quarantine while preserving the scope error API."""
    from .task_repair_bytecode import (
        TaskRepairBytecodeError,
        quarantine_generated_bytecode as quarantine,
    )

    try:
        return quarantine(task_dir, quarantine_dir, manifest_path, accepted_caches)
    except TaskRepairBytecodeError as exc:
        raise TaskRepairScopeError(str(exc)) from exc


def _parse_cache_metadata(values: list[str]) -> frozenset[tuple[str, bytes]]:
    from .task_repair_bytecode import TaskRepairBytecodeError, parse_cache_metadata

    try:
        return parse_cache_metadata(values)
    except TaskRepairBytecodeError as exc:
        raise TaskRepairScopeError(str(exc)) from exc


def _task_files(task_dir: Path) -> dict[str, str]:
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise TaskRepairScopeError(f"unsafe task directory: {task_dir}")
    files: dict[str, str] = {}
    for path in sorted(task_dir.rglob("*")):
        relative = path.relative_to(task_dir)
        if path.is_symlink():
            raise TaskRepairScopeError(f"repair scope contains symlink: {relative.as_posix()}")
        if path.is_file():
            files[relative.as_posix()] = _sha256(path)
        elif not path.is_dir():
            raise TaskRepairScopeError(f"repair scope contains special path: {relative.as_posix()}")
    return files


def _mutable_paths(task_dir: Path) -> tuple[str, ...]:
    try:
        contract = load_task_contract(task_dir / "task_contract.json")
        program_dir = task_dir / "program_0"
        program_spec = load_program_spec(program_dir)
        if program_spec is None:  # pragma: no cover - required=True
            raise TaskRepairScopeError("missing program specification during repair")
        entrypoint = resolve_entrypoint(program_dir, program_spec)
        entrypoint_relative = entrypoint.relative_to(task_dir.resolve()).as_posix()
    except (TaskContractError, ProgramSpecError, OSError, ValueError) as exc:
        raise TaskRepairScopeError("invalid scaffold metadata during repair") from exc
    return tuple(
        sorted(
            {
                contract.evaluators.search,
                contract.evaluators.final,
                "requirements.txt",
                entrypoint_relative,
            }
        )
    )


def _load_contract(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TaskRepairScopeError(f"invalid task contract during repair: {path}")
    if path.stat().st_size > MAX_TASK_CONTRACT_BYTES:
        raise TaskRepairScopeError("task contract is too large during repair")

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TaskRepairScopeError(f"duplicate JSON key in task contract: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_without_duplicates,
        )
    except TaskRepairScopeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskRepairScopeError(f"invalid task contract during repair: {path}") from exc
    if not isinstance(value, dict):
        raise TaskRepairScopeError("task contract must be an object during repair")
    return value


def _load_snapshot(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TaskRepairScopeError(f"unsafe repair snapshot: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskRepairScopeError(f"invalid repair snapshot: {path}") from exc
    if (
        not isinstance(value, dict)
        or value.keys()
        != {
            "schema_version",
            "task_dir",
            "files",
            "contract_invariants",
            "contract_policy",
            "mutable_paths",
        }
        or value.get("schema_version") != _SCHEMA_VERSION
        or not isinstance(value.get("task_dir"), str)
        or not isinstance(value.get("files"), dict)
        or not all(
            isinstance(key, str) and isinstance(digest, str)
            for key, digest in value["files"].items()
        )
        or not isinstance(value.get("contract_invariants"), dict)
        or value.get("contract_policy")
        not in {
            _IMMUTABLE_CONTRACT,
            _SYNCHRONIZED_CONTRACT_DIGESTS,
            _SYNCHRONIZED_DEPENDENCY_REMOVALS,
        }
        or not isinstance(value.get("mutable_paths"), list)
        or not all(isinstance(item, str) for item in value["mutable_paths"])
        or len(set(value["mutable_paths"])) != len(value["mutable_paths"])
    ):
        raise TaskRepairScopeError("invalid repair snapshot schema")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("snapshot", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--task-dir", type=Path, required=True)
        subparser.add_argument("--snapshot", type=Path, required=True)
        if command == "snapshot":
            subparser.add_argument(
                "--contract-policy",
                choices=(
                    _IMMUTABLE_CONTRACT,
                    _SYNCHRONIZED_CONTRACT_DIGESTS,
                    _SYNCHRONIZED_DEPENDENCY_REMOVALS,
                ),
                default=_IMMUTABLE_CONTRACT,
            )
    quarantine_parser = subparsers.add_parser("quarantine-bytecode")
    quarantine_parser.add_argument("--task-dir", type=Path, required=True)
    quarantine_parser.add_argument("--quarantine-dir", type=Path, required=True)
    quarantine_parser.add_argument("--manifest", type=Path)
    quarantine_parser.add_argument(
        "--accepted-cache",
        action="append",
        default=[],
        metavar="TAG:MAGIC_HEX",
    )
    policy_parser = subparsers.add_parser("policy")
    policy_parser.add_argument("--diagnostic", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            if args.contract_policy == _SYNCHRONIZED_DEPENDENCY_REMOVALS:
                create_dependency_removal_repair_snapshot(args.task_dir, args.snapshot)
            elif args.contract_policy == _SYNCHRONIZED_CONTRACT_DIGESTS:
                create_digest_synchronized_repair_snapshot(args.task_dir, args.snapshot)
            else:
                create_repair_snapshot(args.task_dir, args.snapshot)
        elif args.command == "validate":
            validate_repair_snapshot(args.task_dir, args.snapshot)
        elif args.command == "quarantine-bytecode":
            accepted = _parse_cache_metadata(args.accepted_cache)
            quarantined = quarantine_generated_bytecode(
                args.task_dir,
                args.quarantine_dir,
                args.manifest,
                accepted,
            )
            if quarantined:
                print("quarantined generated bytecode: " + ", ".join(quarantined))
        else:
            print(contract_policy_for_smoke_diagnostic(args.diagnostic))
    except TaskRepairScopeError as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
