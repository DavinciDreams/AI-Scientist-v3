"""Bounded multi-error diagnostics for generated task contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .task_contract import MAX_TASK_CONTRACT_BYTES, TaskContractError, safe_contract_relative_path
from .task_contract_types import DEFAULT_MAX_CANDIDATE_RPC_CALLS, HARD_MAX_CANDIDATE_RPC_CALLS

_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PACKAGE_COMPONENT = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_PINNED_VERSION = (
    r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*"
    r"(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?"
    r"(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?"
)
_PINNED_REQUIREMENT_RE = re.compile(
    rf"^(?P<name>{_PACKAGE_COMPONENT}"
    rf"(?:\[{_PACKAGE_COMPONENT}(?:,{_PACKAGE_COMPONENT})*\])?)"
    rf"==(?P<version>{_PINNED_VERSION})$"
)
_PACKAGE_NAME_SEPARATOR_RE = re.compile(r"[-_.]+")
_MAX_FINDINGS = 64
_MAX_ITEMS = 64
_MAX_ARTIFACTS = 256
MAX_DEPENDENCY_ENTRIES = 256
_TOP_FIELDS = {
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
_UPSTREAM_FIELDS = {
    "repository",
    "revision",
    "package_version",
    "license",
    "entrypoints",
    "patches",
}
_COMPUTE_REQUIRED_FIELDS = {
    "accelerator",
    "accelerator_model",
    "gpu_count",
    "evaluator_gpu_count",
    "candidate_gpu_count",
    "cpu_cores",
    "memory_gb",
    "precision",
    "wall_clock_seconds",
}
_COMPUTE_OPTIONAL_FIELDS = {"max_candidate_rpc_calls", "execution_mode"}
_PROTOCOL_REQUIRED_FIELDS = {
    "search_seeds",
    "final_seeds",
    "final_reset_sequence",
    "baselines",
    "promotion_rule",
    "search_proxies",
}
_PROTOCOL_OPTIONAL_FIELDS = {
    "search_reset_sequence",
    "search_smoke_reset_sequence",
    "final_smoke_reset_sequence",
}
_METRIC_FIELDS = {"name", "data_role"}
_DEPENDENCY_FIELDS = {"name", "version"}
_FEEDBACK_FIELDS = {"search_visible", "final_sealed"}
_PUBLICATION_FIELDS = {
    "search_is_proxy",
    "requires_final_evaluation",
    "required_evidence",
}


class _Findings:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.omitted = 0

    def add(self, message: str) -> None:
        if message in self.items:
            return
        if len(self.items) < _MAX_FINDINGS:
            self.items.append(message)
        else:
            self.omitted += 1

    def result(self) -> tuple[str, ...]:
        if self.omitted:
            self.items.append(f"{self.omitted} additional bounded lint finding(s) omitted")
        return tuple(self.items)


def _mapping(value: object) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _safe_path(value: object) -> str | None:
    try:
        return safe_contract_relative_path(value, "path")
    except TaskContractError:
        return None


def _path_is_within(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return len(path_parts) >= len(root_parts) and path_parts[: len(root_parts)] == root_parts


def _object_fields(
    findings: _Findings, value: object, label: str, expected: set[str]
) -> dict[str, Any] | None:
    mapping = _mapping(value)
    if mapping is None:
        findings.add(f"{label} must be an object")
        return None
    missing = expected - mapping.keys()
    if missing:
        findings.add(f"{label} is missing required field(s): {', '.join(sorted(missing))}")
    unexpected_count = sum(key not in expected for key in mapping)
    if unexpected_count:
        findings.add(
            f"{label} has {unexpected_count} unexpected field(s); allowed fields: "
            + ", ".join(sorted(expected))
        )
    return mapping


def _object_fields_with_optional(
    findings: _Findings,
    value: object,
    label: str,
    required: set[str],
    optional: set[str],
) -> dict[str, Any] | None:
    mapping = _mapping(value)
    if mapping is None:
        findings.add(f"{label} must be an object")
        return None
    missing = required - mapping.keys()
    if missing:
        findings.add(f"{label} is missing required field(s): {', '.join(sorted(missing))}")
    allowed = required | optional
    unexpected_count = sum(key not in allowed for key in mapping)
    if unexpected_count:
        findings.add(
            f"{label} has {unexpected_count} unexpected field(s); allowed fields: "
            + ", ".join(sorted(allowed))
        )
    return mapping


def _items(
    findings: _Findings, value: object, label: str, *, limit: int = _MAX_ITEMS
) -> list[object]:
    if not isinstance(value, list):
        findings.add(f"{label} must be an array")
        return []
    if len(value) > limit:
        findings.add(f"{label} has {len(value) - limit} item(s) omitted from bounded lint")
    return value[:limit]


def _normalized_dependency_pin(name: str, version: str) -> str:
    base_name, separator, extras_value = name.partition("[")
    normalized_name = _PACKAGE_NAME_SEPARATOR_RE.sub("-", base_name).lower()
    if separator:
        extras = extras_value.removesuffix("]").split(",")
        normalized_extras = sorted(
            {_PACKAGE_NAME_SEPARATOR_RE.sub("-", extra).lower() for extra in extras}
        )
        normalized_name += f"[{','.join(normalized_extras)}]"
    return f"{normalized_name}=={version}"


def _requirement_pins(path: Path, findings: _Findings) -> set[str] | None:
    if path.is_symlink() or not path.is_file():
        findings.add("requirements.txt must be a safe regular file")
        return None
    if path.stat().st_size > MAX_TASK_CONTRACT_BYTES:
        findings.add(f"requirements.txt exceeds {MAX_TASK_CONTRACT_BYTES} byte harness limit")
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        findings.add("requirements.txt must be readable UTF-8")
        return None
    pins: set[str] = set()
    requirement_rows = 0
    for line in lines:
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        requirement_rows += 1
        if requirement_rows > MAX_DEPENDENCY_ENTRIES:
            findings.add(f"requirements.txt exceeds {MAX_DEPENDENCY_ENTRIES} pinned entries")
            return None
        match = _PINNED_REQUIREMENT_RE.fullmatch(requirement)
        if match is None:
            findings.add("requirements.txt must contain only pinned package==version entries")
            return None
        pins.add(_normalized_dependency_pin(match["name"], match["version"]))
    return pins


def task_contract_preflight_findings(
    contract_path: Path,
    *,
    requirements_path: Path | None = None,
    require_explicit_rpc_budgets: bool = False,
) -> tuple[str, ...]:
    """Return independent actionable findings without stopping at the first error."""
    findings = _Findings()
    try:
        if contract_path.stat().st_size > MAX_TASK_CONTRACT_BYTES:
            findings.add(f"task contract exceeds {MAX_TASK_CONTRACT_BYTES} byte harness limit")
            return findings.result()
        value = json.loads(contract_path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        return ()
    contract = _object_fields(findings, value, "task contract", _TOP_FIELDS)
    if contract is None:
        return findings.result()

    canonical = _object_fields(
        findings, contract.get("canonical_task"), "canonical_task", {"path", "sha256"}
    )
    if canonical is not None and _text(canonical.get("path")) != "task.md":
        findings.add("canonical_task.path must be task.md")

    _object_fields(findings, contract.get("method"), "method", {"class", "description"})

    candidate = _object_fields(
        findings, contract.get("candidate"), "candidate", {"interface", "editable_paths"}
    )
    if candidate is not None:
        if _text(candidate.get("interface")) != "python_rpc_v1:handle":
            findings.add("candidate.interface must be python_rpc_v1:handle")
        for index, value in enumerate(
            _items(findings, candidate.get("editable_paths"), "candidate.editable_paths")
        ):
            path = _safe_path(value)
            if path is None:
                findings.add(
                    f"candidate.editable_paths[{index}] must be a safe POSIX relative path"
                )
            elif not _path_is_within(path, "program_0"):
                findings.add(f"candidate.editable_paths[{index}] must be inside program_0")

    roles = _object_fields(findings, contract.get("data"), "data", {"search_roles", "final_roles"})
    if roles is not None:
        for plane in ("search_roles", "final_roles"):
            for index, value in enumerate(_items(findings, roles.get(plane), f"data.{plane}")):
                role = _text(value)
                if role is None or not _ROLE_RE.fullmatch(role):
                    findings.add(f"data.{plane}[{index}] must be a snake_case role")

    metrics = _object_fields(findings, contract.get("metrics"), "metrics", {"search", "final"})
    if metrics is not None:
        for plane in ("search", "final"):
            label = f"metrics.{plane}"
            for index, metric in enumerate(_items(findings, metrics.get(plane), label)):
                _object_fields(findings, metric, f"{label}[{index}]", _METRIC_FIELDS)

    evaluators = _object_fields(
        findings, contract.get("evaluators"), "evaluators", {"search", "final"}
    )
    if evaluators is not None:
        if _text(evaluators.get("search")) != "evaluate.py":
            findings.add("evaluators.search must be evaluate.py")
        if _text(evaluators.get("final")) != "evaluate_final.py":
            findings.add("evaluators.final must be evaluate_final.py")
    resources = _object_fields(
        findings,
        contract.get("resources"),
        "resources",
        {"search_root", "final_root", "public_artifact_bundle"},
    )
    if resources is not None and (
        _text(resources.get("search_root")) != "resources/search"
        or _text(resources.get("final_root")) != "resources/final"
    ):
        findings.add("resource roots must be resources/search and resources/final")

    artifact_paths: set[str] = set()
    artifact_cross_reference_complete = True
    dependency_pins: set[str] = set()
    dependencies_value = None
    dependencies_complete = True
    provenance = _object_fields(
        findings, contract.get("provenance"), "provenance", {"dependencies", "artifacts"}
    )
    if provenance is not None:
        dependencies_value = provenance.get("dependencies")
        dependencies_complete = (
            isinstance(dependencies_value, list)
            and len(dependencies_value) <= MAX_DEPENDENCY_ENTRIES
        )
        for index, dependency in enumerate(
            _items(
                findings, dependencies_value, "provenance.dependencies",
                limit=MAX_DEPENDENCY_ENTRIES,
            )
        ):
            dependency_data = _object_fields(
                findings,
                dependency,
                f"provenance.dependencies[{index}]",
                _DEPENDENCY_FIELDS,
            )
            if dependency_data is None:
                dependencies_complete = False
                continue
            name = _text(dependency_data.get("name"))
            version = _text(dependency_data.get("version"))
            if (
                name is None
                or version is None
                or _PINNED_REQUIREMENT_RE.fullmatch(f"{name}=={version}") is None
            ):
                findings.add(
                    f"provenance.dependencies[{index}] must name a pip-installable "
                    "package and version"
                )
                dependencies_complete = False
                continue
            dependency_pins.add(_normalized_dependency_pin(name, version))
        artifact_values = provenance.get("artifacts")
        artifact_cross_reference_complete = (
            isinstance(artifact_values, list) and len(artifact_values) <= _MAX_ARTIFACTS
        )
        for index, artifact in enumerate(_items(findings, artifact_values, "provenance.artifacts",
                                                 limit=_MAX_ARTIFACTS)):
            artifact_data = _object_fields(
                findings,
                artifact,
                f"provenance.artifacts[{index}]",
                {"role", "path", "sha256", "source"},
            )
            if artifact_data is not None:
                path = _safe_path(artifact_data.get("path"))
                if path is not None:
                    artifact_paths.add(path)

    patch_paths: set[str] = set()
    upstream_value = contract.get("upstream")
    patch_cross_reference_complete = (
        isinstance(upstream_value, list) and len(upstream_value) <= _MAX_ITEMS
    )
    for index, item in enumerate(_items(findings, upstream_value, "upstream")):
        label = f"upstream[{index}]"
        item_data = _object_fields(findings, item, label, _UPSTREAM_FIELDS)
        if item_data is None:
            continue
        patch_values = item_data.get("patches")
        if not isinstance(patch_values, list) or len(patch_values) > _MAX_ITEMS:
            patch_cross_reference_complete = False
        for patch_index, value in enumerate(_items(findings, patch_values, f"{label}.patches")):
            path = _safe_path(value)
            if path is None:
                findings.add(f"{label}.patches[{patch_index}] must be a safe POSIX relative path")
            else:
                patch_paths.add(path)
    missing_hash_count = len(patch_paths - artifact_paths)
    if artifact_cross_reference_complete and patch_cross_reference_complete and missing_hash_count:
        findings.add(
            f"{missing_hash_count} upstream patch path(s) lack matching provenance artifact checksums"
        )

    rpc_budgets: dict[str, int] = {}
    compute = _object_fields(findings, contract.get("compute"), "compute", {"search", "final"})
    if compute is not None:
        for plane in ("search", "final"):
            label = f"compute.{plane}"
            plane_data = _object_fields_with_optional(
                findings,
                compute.get(plane),
                label,
                _COMPUTE_REQUIRED_FIELDS,
                _COMPUTE_OPTIONAL_FIELDS,
            )
            if (
                require_explicit_rpc_budgets
                and plane_data is not None
                and "max_candidate_rpc_calls" not in plane_data
            ):
                findings.add(f"{label} is missing required field(s): max_candidate_rpc_calls")
            accelerator = _text(plane_data.get("accelerator")) if plane_data else None
            if accelerator is not None and accelerator.lower() not in {"cpu", "gpu"}:
                findings.add(f"{label}.accelerator must be 'cpu' or 'gpu'")
            if plane_data is not None:
                execution_mode = _text(plane_data.get("execution_mode", "isolated_v1"))
                if execution_mode != "isolated_v1":
                    findings.add(f"{label}.execution_mode must be 'isolated_v1'")
            rpc_budget = plane_data.get("max_candidate_rpc_calls") if plane_data else None
            if rpc_budget is None:
                rpc_budgets[plane] = DEFAULT_MAX_CANDIDATE_RPC_CALLS
            elif (
                isinstance(rpc_budget, bool)
                or not isinstance(rpc_budget, int)
                or not 1 <= rpc_budget <= HARD_MAX_CANDIDATE_RPC_CALLS
            ):
                findings.add(
                    f"{label}.max_candidate_rpc_calls must be an integer between 1 and "
                    f"{HARD_MAX_CANDIDATE_RPC_CALLS}"
                )
            else:
                rpc_budgets[plane] = rpc_budget

    protocol = _object_fields_with_optional(
        findings,
        contract.get("protocol"),
        "protocol",
        _PROTOCOL_REQUIRED_FIELDS,
        _PROTOCOL_OPTIONAL_FIELDS,
    )
    if protocol is not None:
        seed_sets: dict[str, set[int]] = {}
        for field in ("search_seeds", "final_seeds"):
            label = f"protocol.{field}"
            raw_seeds = protocol.get(field)
            seeds = _items(findings, raw_seeds, label)
            if not seeds:
                findings.add(f"{label} must be a non-empty array")
            valid_seeds: list[int] = []
            for index, seed in enumerate(seeds):
                if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                    findings.add(f"{label}[{index}] must be a non-negative integer")
                else:
                    valid_seeds.append(seed)
            if isinstance(raw_seeds, list) and len(raw_seeds) <= _MAX_ITEMS:
                if len(valid_seeds) == len(raw_seeds):
                    if len(valid_seeds) != len(set(valid_seeds)):
                        findings.add(f"{label} must not contain duplicates")
                    seed_sets[field] = set(valid_seeds)
        if seed_sets.get("search_seeds", set()).intersection(seed_sets.get("final_seeds", set())):
            findings.add("protocol search and final seeds must be disjoint")

        production_sequences: dict[str, list[int]] = {}
        for plane, field, seed_field in (
            ("search", "search_reset_sequence", "search_seeds"),
            ("final", "final_reset_sequence", "final_seeds"),
        ):
            label = f"protocol.{field}"
            raw_sequence = protocol.get(field)
            if field == "search_reset_sequence" and field not in protocol:
                raw_sequence = protocol.get("search_seeds")
            if not isinstance(raw_sequence, list):
                findings.add(f"{label} must be an array")
                continue
            if not raw_sequence:
                findings.add(f"{label} must be a non-empty array")
                continue
            rpc_budget = rpc_budgets.get(plane)
            if rpc_budget is not None and len(raw_sequence) > rpc_budget:
                findings.add(f"{label} exceeds compute.{plane}.max_candidate_rpc_calls")
            valid_sequence: list[int] = []
            for index, seed in enumerate(raw_sequence):
                if type(seed) is not int or seed < 0:
                    findings.add(f"{label}[{index}] must be a non-negative integer")
                else:
                    valid_sequence.append(seed)
            if len(valid_sequence) != len(raw_sequence):
                continue
            production_sequences[plane] = valid_sequence
            if seed_field in seed_sets and set(valid_sequence) != seed_sets[seed_field]:
                findings.add(f"{label} must cover exactly {seed_field}")

        for plane, field in (
            ("search", "search_smoke_reset_sequence"),
            ("final", "final_smoke_reset_sequence"),
        ):
            production_sequence = production_sequences.get(plane)
            raw_sequence = protocol.get(field)
            if field not in protocol:
                if not production_sequence:
                    continue
                raw_sequence = [production_sequence[0]]
            label = f"protocol.{field}"
            if not isinstance(raw_sequence, list):
                findings.add(f"{label} must be an array")
                continue
            if not raw_sequence:
                findings.add(f"{label} must be a non-empty array")
                continue
            rpc_budget = rpc_budgets.get(plane)
            if rpc_budget is not None and len(raw_sequence) > rpc_budget:
                findings.add(f"{label} exceeds compute.{plane}.max_candidate_rpc_calls")
            valid_smoke_sequence: list[int] = []
            for index, seed in enumerate(raw_sequence):
                if type(seed) is not int or seed < 0:
                    findings.add(f"{label}[{index}] must be a non-negative integer")
                else:
                    valid_smoke_sequence.append(seed)
            if (
                production_sequence
                and len(valid_smoke_sequence) == len(raw_sequence)
                and any(seed != production_sequence[0] for seed in valid_smoke_sequence)
            ):
                findings.add(f"{label} must contain only the first production reset seed")

        raw_baselines = protocol.get("baselines")
        baselines = _items(findings, raw_baselines, "protocol.baselines")
        if not baselines:
            findings.add("protocol.baselines must be a non-empty array")
        valid_baselines: list[str] = []
        for index, baseline in enumerate(baselines):
            normalized_baseline = _text(baseline)
            if normalized_baseline is None:
                findings.add(f"protocol.baselines[{index}] must be a non-empty string")
            else:
                valid_baselines.append(normalized_baseline)
        if (
            isinstance(raw_baselines, list)
            and len(raw_baselines) <= _MAX_ITEMS
            and len(valid_baselines) == len(raw_baselines)
            and len(valid_baselines) != len(set(valid_baselines))
        ):
            findings.add("protocol.baselines must not contain duplicates")
        if _text(protocol.get("promotion_rule")) is None:
            findings.add("protocol.promotion_rule must be a non-empty string")
        raw_proxies = protocol.get("search_proxies")
        proxies = _items(findings, raw_proxies, "protocol.search_proxies")
        proxy_names: list[str] = []
        all_proxy_names_valid = True
        for index, proxy in enumerate(proxies):
            label = f"protocol.search_proxies[{index}]"
            proxy_data = _object_fields(findings, proxy, label, {"name", "preserves", "omits"})
            if proxy_data is None:
                all_proxy_names_valid = False
                continue
            for field in ("name", "preserves", "omits"):
                if _text(proxy_data.get(field)) is None:
                    findings.add(f"{label}.{field} must be a non-empty string")
            name = _text(proxy_data.get("name"))
            if name is None:
                all_proxy_names_valid = False
            else:
                proxy_names.append(name)
        if (
            isinstance(raw_proxies, list)
            and len(raw_proxies) <= _MAX_ITEMS
            and all_proxy_names_valid
            and len(proxy_names) != len(set(proxy_names))
        ):
            findings.add("protocol.search_proxies names must be unique")

    _object_fields(findings, contract.get("feedback"), "feedback", _FEEDBACK_FIELDS)
    _object_fields(findings, contract.get("publication"), "publication", _PUBLICATION_FIELDS)
    if requirements_path is not None and provenance is not None and dependencies_complete:
        requirement_pins = _requirement_pins(requirements_path, findings)
        if requirement_pins is not None and requirement_pins != dependency_pins:
            requirements_only = sorted(requirement_pins - dependency_pins)
            contract_only = sorted(dependency_pins - requirement_pins)
            findings.add(
                "requirements.txt and provenance.dependencies must contain the same pinned "
                f"packages; requirements-only={requirements_only} contract-only={contract_only}"
            )
    return findings.result()
