"""Strict task-contract parsing and deterministic scaffold fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .task_contract_types import (
    Artifact,
    Candidate,
    CanonicalTask,
    Compute,
    ComputePlane,
    DataRoles,
    DEFAULT_MAX_CANDIDATE_RPC_CALLS,
    Dependency,
    Evaluators,
    Feedback,
    HARD_MAX_CANDIDATE_RPC_CALLS,
    Method,
    Metric,
    Metrics,
    MIN_COMPUTE_WALL_CLOCK_SECONDS,
    Protocol,
    Provenance,
    Publication,
    ResourceRoots,
    SearchProxy,
    TaskContract,
    Upstream,
)

TASK_CONTRACT_FILENAME = "task_contract.json"
TASK_CONTRACT_SCHEMA_VERSION = 1
MAX_TASK_CONTRACT_BYTES = 2 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_FINAL_ROLE_NAMES = frozenset({"final", "test", "sealed_test", "private_test"})
_FLOATING_REVISIONS = frozenset({"head", "latest", "main", "master", "trunk"})
_SAFE_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@-]*$")
_PUBLIC_BUNDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class TaskContractError(ValueError):
    """Raised when a task contract or its declared artifacts are invalid."""


def prune_generated_bytecode(task_dir: Path) -> None:
    """Remove agent-created bytecode without traversing harness runtime state."""
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise TaskContractError(f"unsafe task directory for bytecode pruning: {task_dir}")
    for directory, directory_names, file_names in os.walk(task_dir, topdown=True):
        current = Path(directory)
        if current == task_dir:
            directory_names[:] = [name for name in directory_names if name != ".venv"]
        for name in list(directory_names):
            path = current / name
            if name != "__pycache__":
                continue
            if path.is_symlink() or not path.is_dir():
                raise TaskContractError(f"unsafe runtime cache path: {path}")
            for entry in path.rglob("*"):
                if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
                    raise TaskContractError(f"unsafe runtime cache entry: {entry}")
            shutil.rmtree(path)
            directory_names.remove(name)
        for name in file_names:
            if not name.endswith((".pyc", ".pyo")):
                continue
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise TaskContractError(f"unsafe runtime bytecode path: {path}")
            path.unlink()


def _mapping(value: object, label: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TaskContractError(f"{label} must be an object")
    missing = fields - value.keys()
    unexpected = value.keys() - fields
    if missing:
        raise TaskContractError(f"{label} missing field(s): {', '.join(sorted(missing))}")
    if unexpected:
        raise TaskContractError(f"{label} has unexpected field(s): {', '.join(sorted(unexpected))}")
    return value


def _mapping_with_optional(
    value: object,
    label: str,
    required_fields: set[str],
    optional_fields: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TaskContractError(f"{label} must be an object")
    missing = required_fields - value.keys()
    unexpected = value.keys() - required_fields - optional_fields
    if missing:
        raise TaskContractError(f"{label} missing field(s): {', '.join(sorted(missing))}")
    if unexpected:
        raise TaskContractError(f"{label} has unexpected field(s): {', '.join(sorted(unexpected))}")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskContractError(f"{label} must be a non-empty string")
    return value.strip()


def safe_contract_relative_path(value: object, label: str) -> str:
    """Validate contract path syntax, including a leaf Python package initializer."""
    raw = _string(value, label)
    components = raw.split("/")
    ordinary_components = components[:-1] if components[-1] == "__init__.py" else components
    if "\\" in raw or any(
        not _SAFE_PATH_COMPONENT_RE.fullmatch(part) for part in ordinary_components
    ):
        raise TaskContractError(f"{label} must be a safe POSIX relative path")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or raw in {"", "."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise TaskContractError(f"{label} must be a safe relative path")
    return path.as_posix()


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TaskContractError(f"{label} must be a non-empty array")
    items = tuple(_string(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(items) != len(set(items)):
        raise TaskContractError(f"{label} must not contain duplicates")
    return items


def _optional_string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TaskContractError(f"{label} must be an array")
    items = tuple(_string(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(items) != len(set(items)):
        raise TaskContractError(f"{label} must not contain duplicates")
    return items


def _seed_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TaskContractError(f"{label} must be a non-empty array")
    seeds: list[int] = []
    for index, seed in enumerate(value):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise TaskContractError(f"{label}[{index}] must be a non-negative integer")
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise TaskContractError(f"{label} must not contain duplicates")
    return tuple(seeds)


def _reset_sequence(
    value: object,
    *,
    label: str,
    statistical_seeds: tuple[int, ...],
    max_candidate_rpc_calls: int,
    plane: str,
) -> tuple[int, ...]:
    sequence = _reset_seed_tuple(value, label, max_candidate_rpc_calls, plane)
    if set(sequence) != set(statistical_seeds):
        seed_field = "search_seeds" if plane == "search" else "final_seeds"
        raise TaskContractError(f"{label} must cover exactly {seed_field}")
    return sequence


def _smoke_reset_sequence(
    value: object,
    *,
    label: str,
    production_sequence: tuple[int, ...],
    max_candidate_rpc_calls: int,
    plane: str,
) -> tuple[int, ...]:
    sequence = _reset_seed_tuple(value, label, max_candidate_rpc_calls, plane)
    if any(seed != production_sequence[0] for seed in sequence):
        raise TaskContractError(f"{label} must contain only the first production reset seed")
    return sequence


def _reset_seed_tuple(
    value: object,
    label: str,
    max_candidate_rpc_calls: int,
    plane: str,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TaskContractError(f"{label} must be a non-empty array")
    for index, seed in enumerate(value):
        if type(seed) is not int or seed < 0:
            raise TaskContractError(f"{label}[{index}] must be a non-negative integer")
    if len(value) > max_candidate_rpc_calls:
        raise TaskContractError(f"{label} exceeds compute.{plane}.max_candidate_rpc_calls")
    return tuple(value)


def _parse_compute_plane(value: object, label: str) -> ComputePlane:
    data = _mapping_with_optional(
        value,
        label,
        {
            "accelerator",
            "accelerator_model",
            "gpu_count",
            "evaluator_gpu_count",
            "candidate_gpu_count",
            "cpu_cores",
            "memory_gb",
            "precision",
            "wall_clock_seconds",
        },
        {"max_candidate_rpc_calls", "execution_mode"},
    )
    execution_mode = _string(data.get("execution_mode", "isolated_v1"), f"{label}.execution_mode")
    if execution_mode != "isolated_v1":
        raise TaskContractError(f"{label}.execution_mode must be 'isolated_v1'")
    accelerator = _string(data["accelerator"], f"{label}.accelerator").lower()
    if accelerator not in {"cpu", "gpu"}:
        raise TaskContractError(f"{label}.accelerator must be 'cpu' or 'gpu'")
    gpu_count = data["gpu_count"]
    evaluator_gpu_count = data["evaluator_gpu_count"]
    candidate_gpu_count = data["candidate_gpu_count"]
    cpu_cores = data["cpu_cores"]
    memory_gb = data["memory_gb"]
    wall_clock_seconds = data["wall_clock_seconds"]
    max_candidate_rpc_calls = data.get("max_candidate_rpc_calls", DEFAULT_MAX_CANDIDATE_RPC_CALLS)
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
        raise TaskContractError(f"{label}.gpu_count must be a non-negative integer")
    for field_name, count in (
        ("evaluator_gpu_count", evaluator_gpu_count),
        ("candidate_gpu_count", candidate_gpu_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TaskContractError(f"{label}.{field_name} must be a non-negative integer")
    if isinstance(cpu_cores, bool) or not isinstance(cpu_cores, int) or cpu_cores < 1:
        raise TaskContractError(f"{label}.cpu_cores must be a positive integer")
    if isinstance(memory_gb, bool) or not isinstance(memory_gb, int) or memory_gb < 1:
        raise TaskContractError(f"{label}.memory_gb must be a positive integer")
    if (
        isinstance(wall_clock_seconds, bool)
        or not isinstance(wall_clock_seconds, int)
        or wall_clock_seconds < MIN_COMPUTE_WALL_CLOCK_SECONDS
    ):
        raise TaskContractError(
            f"{label}.wall_clock_seconds must be an integer of at least "
            f"{MIN_COMPUTE_WALL_CLOCK_SECONDS} seconds"
        )
    if (
        isinstance(max_candidate_rpc_calls, bool)
        or not isinstance(max_candidate_rpc_calls, int)
        or not 1 <= max_candidate_rpc_calls <= HARD_MAX_CANDIDATE_RPC_CALLS
    ):
        raise TaskContractError(
            f"{label}.max_candidate_rpc_calls must be an integer between 1 and "
            f"{HARD_MAX_CANDIDATE_RPC_CALLS}"
        )
    if (accelerator == "cpu" and gpu_count != 0) or (accelerator == "gpu" and gpu_count < 1):
        raise TaskContractError(f"{label}.gpu_count is inconsistent with accelerator")
    if evaluator_gpu_count + candidate_gpu_count != gpu_count:
        raise TaskContractError(
            f"{label} evaluator_gpu_count + candidate_gpu_count must equal gpu_count"
        )
    accelerator_model = _string(data["accelerator_model"], f"{label}.accelerator_model")
    if accelerator == "cpu" and accelerator_model.lower() != "none":
        raise TaskContractError(f"{label}.accelerator_model must be 'none' for CPU")
    if accelerator == "gpu" and accelerator_model.lower() == "none":
        raise TaskContractError(f"{label}.accelerator_model must identify the GPU model")
    return ComputePlane(
        accelerator,
        accelerator_model,
        gpu_count,
        evaluator_gpu_count,
        candidate_gpu_count,
        cpu_cores,
        memory_gb,
        _string(data["precision"], f"{label}.precision"),
        wall_clock_seconds,
        max_candidate_rpc_calls,
        execution_mode,
    )


def _parse_search_proxies(value: object) -> tuple[SearchProxy, ...]:
    if not isinstance(value, list):
        raise TaskContractError("protocol.search_proxies must be an array")
    result: list[SearchProxy] = []
    for index, item in enumerate(value):
        label = f"protocol.search_proxies[{index}]"
        data = _mapping(item, label, {"name", "preserves", "omits"})
        result.append(
            SearchProxy(
                _string(data["name"], f"{label}.name"),
                _string(data["preserves"], f"{label}.preserves"),
                _string(data["omits"], f"{label}.omits"),
            )
        )
    if len({proxy.name for proxy in result}) != len(result):
        raise TaskContractError("protocol.search_proxies names must be unique")
    return tuple(result)


def _parse_metrics(value: object, label: str) -> tuple[Metric, ...]:
    if not isinstance(value, list) or not value:
        raise TaskContractError(f"{label} must be a non-empty array")
    result: list[Metric] = []
    for index, item in enumerate(value):
        metric_label = f"{label}[{index}]"
        data = _mapping(item, metric_label, {"name", "data_role"})
        result.append(
            Metric(
                _string(data["name"], f"{metric_label}.name"),
                _string(data["data_role"], f"{metric_label}.data_role"),
            )
        )
    if len({metric.name for metric in result}) != len(result):
        raise TaskContractError(f"{label} metric names must be unique")
    return tuple(result)


def _paths_overlap(first: str, second: str) -> bool:
    first_parts = PurePosixPath(first).parts
    second_parts = PurePosixPath(second).parts
    shared = min(len(first_parts), len(second_parts))
    return first_parts[:shared] == second_parts[:shared]


def _path_is_within(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return len(path_parts) >= len(root_parts) and path_parts[: len(root_parts)] == root_parts


def parse_task_contract(value: object) -> TaskContract:
    """Parse a JSON-compatible value into a strict schema-version 1 contract."""
    data = _mapping(
        value,
        "task contract",
        {
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
        },
    )
    version = data["schema_version"]
    if isinstance(version, bool) or version != TASK_CONTRACT_SCHEMA_VERSION:
        raise TaskContractError(
            f"unsupported schema_version={version!r}; expected {TASK_CONTRACT_SCHEMA_VERSION}"
        )

    canonical_data = _mapping(data["canonical_task"], "canonical_task", {"path", "sha256"})
    canonical = CanonicalTask(
        safe_contract_relative_path(canonical_data["path"], "canonical_task.path"),
        _string(canonical_data["sha256"], "canonical_task.sha256").lower(),
    )
    if not _SHA256_RE.fullmatch(canonical.sha256):
        raise TaskContractError("canonical_task.sha256 must be a lowercase SHA-256 digest")
    if canonical.path != "task.md":
        raise TaskContractError("canonical_task.path must be task.md")

    method_data = _mapping(data["method"], "method", {"class", "description"})
    method = Method(
        _string(method_data["class"], "method.class"),
        _string(method_data["description"], "method.description"),
    )
    candidate_data = _mapping(data["candidate"], "candidate", {"interface", "editable_paths"})
    editable_values = _string_tuple(candidate_data["editable_paths"], "candidate.editable_paths")
    candidate = Candidate(
        _string(candidate_data["interface"], "candidate.interface"),
        tuple(
            safe_contract_relative_path(path, f"candidate.editable_paths[{index}]")
            for index, path in enumerate(editable_values)
        ),
    )
    if candidate.interface != "python_rpc_v1:handle":
        raise TaskContractError("candidate.interface must be python_rpc_v1:handle")
    if any(not _path_is_within(path, "program_0") for path in candidate.editable_paths):
        raise TaskContractError("candidate editable paths must be inside program_0")

    evaluator_data = _mapping(data["evaluators"], "evaluators", {"search", "final"})
    evaluators = Evaluators(
        safe_contract_relative_path(evaluator_data["search"], "evaluators.search"),
        safe_contract_relative_path(evaluator_data["final"], "evaluators.final"),
    )
    if evaluators.search == evaluators.final:
        raise TaskContractError("search and final evaluator paths must differ")
    if evaluators.search != "evaluate.py":
        raise TaskContractError("evaluators.search must be evaluate.py")
    if evaluators.final != "evaluate_final.py":
        raise TaskContractError("evaluators.final must be evaluate_final.py")

    resource_data = _mapping(
        data["resources"],
        "resources",
        {"search_root", "final_root", "public_artifact_bundle"},
    )
    raw_bundle = resource_data["public_artifact_bundle"]
    if raw_bundle is not None and (
        not isinstance(raw_bundle, str) or _PUBLIC_BUNDLE_RE.fullmatch(raw_bundle) is None
    ):
        raise TaskContractError("resources.public_artifact_bundle is invalid")
    resources = ResourceRoots(
        safe_contract_relative_path(resource_data["search_root"], "resources.search_root"),
        safe_contract_relative_path(resource_data["final_root"], "resources.final_root"),
        raw_bundle,
    )
    if (resources.search_root, resources.final_root) != (
        "resources/search",
        "resources/final",
    ):
        raise TaskContractError(
            "resource roots must be resources/search and resources/final"
        )
    if _paths_overlap(resources.search_root, resources.final_root):
        raise TaskContractError("search and final resource roots must not overlap")
    if _path_is_within(evaluators.search, resources.final_root):
        raise TaskContractError("search evaluator must not be inside the final resource root")
    if _path_is_within(evaluators.final, resources.search_root):
        raise TaskContractError("final evaluator must not be inside the search resource root")
    for editable_path in candidate.editable_paths:
        if _paths_overlap(editable_path, resources.final_root) or editable_path == evaluators.final:
            raise TaskContractError(
                f"candidate editable path must not overlap sealed final artifacts: {editable_path}"
            )

    role_data = _mapping(data["data"], "data", {"search_roles", "final_roles"})
    roles = DataRoles(
        _string_tuple(role_data["search_roles"], "data.search_roles"),
        _string_tuple(role_data["final_roles"], "data.final_roles"),
    )
    for role in (*roles.search_roles, *roles.final_roles):
        if not _ROLE_RE.fullmatch(role):
            raise TaskContractError(f"invalid data role: {role!r}")
    forbidden = _FINAL_ROLE_NAMES.intersection(roles.search_roles)
    if forbidden:
        raise TaskContractError(
            f"search data roles must not include final roles: {sorted(forbidden)}"
        )
    if set(roles.search_roles).intersection(roles.final_roles):
        raise TaskContractError("search and final data roles must be distinct")

    metric_data = _mapping(data["metrics"], "metrics", {"search", "final"})
    metrics = Metrics(
        _parse_metrics(metric_data["search"], "metrics.search"),
        _parse_metrics(metric_data["final"], "metrics.final"),
    )
    if "combined_score" not in {metric.name for metric in metrics.search}:
        raise TaskContractError("metrics.search must declare combined_score")
    for metric in metrics.search:
        if metric.data_role not in roles.search_roles or metric.data_role in _FINAL_ROLE_NAMES:
            raise TaskContractError(f"search metric {metric.name!r} must use a search data role")
    for metric in metrics.final:
        if metric.data_role not in roles.final_roles:
            raise TaskContractError(f"final metric {metric.name!r} must use a final data role")

    upstream_value = data["upstream"]
    if not isinstance(upstream_value, list):
        raise TaskContractError("upstream must be an array")
    upstream: list[Upstream] = []
    for index, item in enumerate(upstream_value):
        label = f"upstream[{index}]"
        item_data = _mapping(
            item,
            label,
            {"repository", "revision", "package_version", "license", "entrypoints", "patches"},
        )
        repository = _string(item_data["repository"], f"{label}.repository")
        revision = _string(item_data["revision"], f"{label}.revision")
        revision_lower = revision.lower()
        if (
            revision_lower in _FLOATING_REVISIONS
            or revision_lower.startswith("refs/heads/")
            or revision_lower.startswith("origin/")
        ):
            raise TaskContractError(f"{label}.revision must identify a pinned revision")
        package_version_value = item_data["package_version"]
        package_version = (
            None
            if package_version_value is None
            else _string(package_version_value, f"{label}.package_version")
        )
        upstream.append(
            Upstream(
                repository,
                revision,
                package_version,
                _string(item_data["license"], f"{label}.license"),
                _string_tuple(item_data["entrypoints"], f"{label}.entrypoints"),
                tuple(
                    safe_contract_relative_path(path, f"{label}.patches[{patch_index}]")
                    for patch_index, path in enumerate(
                        _optional_string_list(item_data["patches"], f"{label}.patches")
                    )
                ),
            )
        )

    provenance_data = _mapping(data["provenance"], "provenance", {"dependencies", "artifacts"})
    dependency_values = provenance_data["dependencies"]
    if not isinstance(dependency_values, list):
        raise TaskContractError("provenance.dependencies must be an array")
    dependencies: list[Dependency] = []
    for index, item in enumerate(dependency_values):
        label = f"provenance.dependencies[{index}]"
        item_data = _mapping(item, label, {"name", "version"})
        dependencies.append(
            Dependency(
                _string(item_data["name"], f"{label}.name"),
                _string(item_data["version"], f"{label}.version"),
            )
        )
    artifact_values = provenance_data["artifacts"]
    if not isinstance(artifact_values, list):
        raise TaskContractError("provenance.artifacts must be an array")
    artifacts: list[Artifact] = []
    for index, item in enumerate(artifact_values):
        label = f"provenance.artifacts[{index}]"
        item_data = _mapping(item, label, {"role", "path", "sha256", "source"})
        digest = _string(item_data["sha256"], f"{label}.sha256").lower()
        if not _SHA256_RE.fullmatch(digest):
            raise TaskContractError(f"{label}.sha256 must be a lowercase SHA-256 digest")
        artifacts.append(
            Artifact(
                _string(item_data["role"], f"{label}.role"),
                safe_contract_relative_path(item_data["path"], f"{label}.path"),
                digest,
                _string(item_data["source"], f"{label}.source"),
            )
        )
    artifact_paths = [artifact.path for artifact in artifacts]
    if len(artifact_paths) != len(set(artifact_paths)):
        raise TaskContractError("provenance.artifacts paths must be unique")
    declared_patch_paths = {patch for item in upstream for patch in item.patches}
    missing_patch_hashes = declared_patch_paths - set(artifact_paths)
    if missing_patch_hashes:
        raise TaskContractError(
            "upstream patches must have provenance artifact checksums: "
            + ", ".join(sorted(missing_patch_hashes))
        )
    provenance = Provenance(tuple(dependencies), tuple(artifacts))

    compute_data = _mapping(data["compute"], "compute", {"search", "final"})
    compute = Compute(
        _parse_compute_plane(compute_data["search"], "compute.search"),
        _parse_compute_plane(compute_data["final"], "compute.final"),
    )

    protocol_data = _mapping_with_optional(
        data["protocol"],
        "protocol",
        {
            "search_seeds",
            "final_seeds",
            "final_reset_sequence",
            "baselines",
            "promotion_rule",
            "search_proxies",
        },
        {
            "search_reset_sequence",
            "search_smoke_reset_sequence",
            "final_smoke_reset_sequence",
        },
    )
    search_seeds = _seed_tuple(protocol_data["search_seeds"], "protocol.search_seeds")
    final_seeds = _seed_tuple(protocol_data["final_seeds"], "protocol.final_seeds")
    search_reset_sequence = _reset_sequence(
        protocol_data.get("search_reset_sequence", list(search_seeds)),
        label="protocol.search_reset_sequence",
        statistical_seeds=search_seeds,
        max_candidate_rpc_calls=compute.search.max_candidate_rpc_calls,
        plane="search",
    )
    final_reset_sequence = _reset_sequence(
        protocol_data["final_reset_sequence"],
        label="protocol.final_reset_sequence",
        statistical_seeds=final_seeds,
        max_candidate_rpc_calls=compute.final.max_candidate_rpc_calls,
        plane="final",
    )
    protocol = Protocol(
        search_seeds=search_seeds,
        final_seeds=final_seeds,
        search_reset_sequence=search_reset_sequence,
        search_smoke_reset_sequence=_smoke_reset_sequence(
            protocol_data.get("search_smoke_reset_sequence", [search_reset_sequence[0]]),
            label="protocol.search_smoke_reset_sequence",
            production_sequence=search_reset_sequence,
            max_candidate_rpc_calls=compute.search.max_candidate_rpc_calls,
            plane="search",
        ),
        final_reset_sequence=final_reset_sequence,
        final_smoke_reset_sequence=_smoke_reset_sequence(
            protocol_data.get("final_smoke_reset_sequence", [final_reset_sequence[0]]),
            label="protocol.final_smoke_reset_sequence",
            production_sequence=final_reset_sequence,
            max_candidate_rpc_calls=compute.final.max_candidate_rpc_calls,
            plane="final",
        ),
        baselines=_string_tuple(protocol_data["baselines"], "protocol.baselines"),
        promotion_rule=_string(protocol_data["promotion_rule"], "protocol.promotion_rule"),
        search_proxies=_parse_search_proxies(protocol_data["search_proxies"]),
    )
    if set(search_seeds).intersection(final_seeds):
        raise TaskContractError("protocol search and final seeds must be disjoint")
    feedback_data = _mapping(data["feedback"], "feedback", {"search_visible", "final_sealed"})
    feedback = Feedback(
        _string_tuple(feedback_data["search_visible"], "feedback.search_visible"),
        _string_tuple(feedback_data["final_sealed"], "feedback.final_sealed"),
    )
    leaked_feedback = set(feedback.search_visible).intersection(feedback.final_sealed)
    if leaked_feedback:
        raise TaskContractError(
            f"feedback search-visible and final-sealed declarations overlap: {sorted(leaked_feedback)}"
        )
    publication_data = _mapping(
        data["publication"],
        "publication",
        {"search_is_proxy", "requires_final_evaluation", "required_evidence"},
    )
    for name in ("search_is_proxy", "requires_final_evaluation"):
        if not isinstance(publication_data[name], bool):
            raise TaskContractError(f"publication.{name} must be a boolean")
    publication = Publication(
        publication_data["search_is_proxy"],
        publication_data["requires_final_evaluation"],
        _string_tuple(publication_data["required_evidence"], "publication.required_evidence"),
    )
    return TaskContract(
        version,
        canonical,
        method,
        candidate,
        evaluators,
        resources,
        roles,
        metrics,
        tuple(upstream),
        provenance,
        compute,
        protocol,
        feedback,
        publication,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TaskContractError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_task_contract(contract_path: Path) -> TaskContract:
    """Load a contract and verify its canonical task digest."""
    if contract_path.is_symlink():
        raise TaskContractError(f"task contract must not be a symlink: {contract_path}")
    try:
        if contract_path.stat().st_size > MAX_TASK_CONTRACT_BYTES:
            raise TaskContractError(
                f"task contract exceeds {MAX_TASK_CONTRACT_BYTES} byte harness limit"
            )
        raw = contract_path.read_text(encoding="utf-8")
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except TaskContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise TaskContractError(f"failed to load {contract_path}: {exc}") from exc
    contract = parse_task_contract(value)
    task_path = _resolve_declared_file(contract_path.parent, contract.canonical_task.path)
    actual_digest = hashlib.sha256(task_path.read_bytes()).hexdigest()
    if actual_digest != contract.canonical_task.sha256:
        raise TaskContractError("canonical task digest mismatch")
    for artifact in contract.provenance.artifacts:
        artifact_path = _resolve_declared_file(contract_path.parent, artifact.path)
        actual_artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_artifact_digest != artifact.sha256:
            raise TaskContractError(f"provenance artifact digest mismatch: {artifact.path}")
    declared_artifacts = {artifact.path for artifact in contract.provenance.artifacts}
    frozen_resource_files: set[str] = set()
    for root_relative in (contract.resources.search_root, contract.resources.final_root):
        _reject_symlink_components(contract_path.parent, root_relative)
        root = contract_path.parent / root_relative
        if not root.is_dir():
            raise TaskContractError(f"declared resource root is not a directory: {root_relative}")
        for path in root.rglob("*"):
            relative = path.relative_to(contract_path.parent).as_posix()
            if path.is_symlink():
                raise TaskContractError(f"declared resource contains a symlink: {relative}")
            if path.is_file() and path.name != ".agentic_resource_manifest.json":
                frozen_resource_files.add(relative)
    missing_artifacts = frozen_resource_files - declared_artifacts
    if missing_artifacts:
        raise TaskContractError(
            "frozen resource files missing provenance checksums: "
            + ", ".join(sorted(missing_artifacts))
        )
    return contract


def _resolve_declared_file(task_dir: Path, relative_path: str) -> Path:
    _reject_symlink_components(task_dir, relative_path)
    path = task_dir / relative_path
    if not path.is_file():
        raise TaskContractError(f"declared artifact is not a file: {relative_path}")
    return path


def _reject_symlink_components(task_dir: Path, relative_path: str) -> None:
    current = task_dir
    for part in PurePosixPath(relative_path).parts:
        current /= part
        if current.is_symlink():
            raise TaskContractError(f"declared artifact must not contain symlinks: {relative_path}")


def _hash_file(hasher: Any, task_dir: Path, path: Path) -> None:
    relative = path.relative_to(task_dir).as_posix().encode()
    content = path.read_bytes()
    hasher.update(b"file\0" + relative + b"\0" + str(len(content)).encode() + b"\0" + content)


def task_scaffold_fingerprint(task_dir: Path) -> str:
    """Hash the contract, canonical task, evaluators, and both declared resource trees."""
    contract_path = task_dir / TASK_CONTRACT_FILENAME
    contract = load_task_contract(contract_path)
    hasher = hashlib.sha256()
    for relative in (
        TASK_CONTRACT_FILENAME,
        contract.canonical_task.path,
        contract.evaluators.search,
        contract.evaluators.final,
    ):
        _hash_file(hasher, task_dir, _resolve_declared_file(task_dir, relative))
    for root_relative in (contract.resources.search_root, contract.resources.final_root):
        _reject_symlink_components(task_dir, root_relative)
        root = task_dir / root_relative
        if not root.is_dir():
            raise TaskContractError(f"declared resource root is not a directory: {root_relative}")
        hasher.update(b"dir\0" + root_relative.encode() + b"\0")
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(task_dir).as_posix()):
            relative = path.relative_to(task_dir).as_posix()
            if path.name == ".agentic_resource_manifest.json":
                continue
            if path.is_symlink():
                raise TaskContractError(f"declared resource contains a symlink: {relative}")
            if path.is_dir():
                hasher.update(b"dir\0" + relative.encode() + b"\0")
            elif path.is_file():
                _hash_file(hasher, task_dir, path)
            else:
                raise TaskContractError(f"unsupported declared resource: {relative}")
    return "sha256:" + hasher.hexdigest()
