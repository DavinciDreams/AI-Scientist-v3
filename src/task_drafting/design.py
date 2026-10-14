"""Strict, task-general design and design-review contracts."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


_SAFE_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@-]*$")


DESIGN_DIMENSIONS = (
    "goal_and_method_fidelity",
    "upstream_grounding",
    "search_proxy_fidelity",
    "final_evaluation_fidelity",
    "search_final_isolation",
    "metrics_and_rubrics",
    "candidate_editability",
    "compute_realism",
    "execution_feasibility",
    "reproducibility",
    "publication_scope",
)
MAX_DESIGN_ARTIFACT_BYTES = 1024 * 1024
MAX_DESIGN_PATCH_OPERATIONS = 64
IMMUTABLE_DESIGN_PATHS = ("/candidate/interface",)


class TaskDesignError(ValueError):
    """Raised when a generated task design or review violates its contract."""


class TaskDesignAdjudicationRequired(TaskDesignError):
    """Raised when a minority failure needs an independent tie-break."""


@dataclass(frozen=True)
class TaskDesign:
    payload: dict[str, Any]
    fingerprint: str


@dataclass(frozen=True)
class TaskDesignReview:
    verdict: str
    failed_dimensions: tuple[str, ...]
    payload: dict[str, Any]


@dataclass(frozen=True)
class TaskDesignAdjudication:
    payload: dict[str, Any]


@dataclass(frozen=True)
class TaskDesignReviewProgress:
    fixed: tuple[str, ...]
    still_failed: tuple[str, ...]
    regressed: tuple[str, ...]

    @property
    def improved(self) -> bool:
        return bool(self.fixed) and not self.regressed


@dataclass(frozen=True)
class TaskDesignDeltaReview:
    verdict: str
    fixed_dimensions: tuple[str, ...]
    regressed_dimensions: tuple[str, ...]
    payload: dict[str, Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskDesignError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TaskDesignError(f"{label} must be a regular file")
    if path.stat().st_size > MAX_DESIGN_ARTIFACT_BYTES:
        raise TaskDesignError(f"{label} exceeds 1 MiB")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except TaskDesignError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskDesignError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise TaskDesignError(f"{label} must be a JSON object")
    return payload


def _exact_fields(value: object, label: str, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskDesignError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise TaskDesignError(f"{label} fields mismatch: missing={missing}, extra={extra}")
    return value


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskDesignError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _text_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise TaskDesignError(f"{label} must be a non-empty array")
    normalized = [_nonempty_text(item, f"{label}[]") for item in value]
    if len(set(normalized)) != len(normalized):
        raise TaskDesignError(f"{label} must contain unique values")
    return normalized


def _candidate_editable_paths(value: object) -> list[str]:
    paths = _text_list(value, "candidate.editable_paths")
    for raw in paths:
        path = PurePosixPath(raw)
        if (
            path.is_absolute()
            or "\\" in raw
            or any(not _SAFE_PATH_COMPONENT_RE.fullmatch(part) for part in raw.split("/"))
            or any(part in {"", ".", ".."} for part in path.parts)
            or not path.parts
            or path.parts[0] != "program_0"
            or len(path.parts) < 2
            or path.parts[1] == "program_spec.toml"
        ):
            raise TaskDesignError(
                "candidate.editable_paths must contain safe relative paths inside program_0"
            )
    return paths


def _possibly_empty_text_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TaskDesignError(f"{label} must be an array")
    normalized = [_nonempty_text(item, f"{label}[]") for item in value]
    if len(set(normalized)) != len(normalized):
        raise TaskDesignError(f"{label} must contain unique values")
    return normalized


def _seed_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise TaskDesignError(f"{label} must be a non-empty array")
    if any(type(seed) is not int or seed < 0 for seed in value):
        raise TaskDesignError(f"{label} must contain non-negative integers")
    if len(set(value)) != len(value):
        raise TaskDesignError(f"{label} must contain unique values")
    return value


def _validate_upstream(value: object) -> None:
    if not isinstance(value, list):
        raise TaskDesignError("upstream must be an array")
    repositories: set[str] = set()
    for index, item in enumerate(value):
        row = _exact_fields(
            item,
            f"upstream[{index}]",
            {"repository", "revision", "license", "entrypoints"},
        )
        repository = _nonempty_text(row["repository"], f"upstream[{index}].repository")
        if repository in repositories:
            raise TaskDesignError("upstream repositories must be unique")
        repositories.add(repository)
        _nonempty_text(row["revision"], f"upstream[{index}].revision")
        _nonempty_text(row["license"], f"upstream[{index}].license")
        _text_list(row["entrypoints"], f"upstream[{index}].entrypoints")


def _validate_plane(value: object, label: str) -> list[int]:
    plane = _exact_fields(
        value,
        label,
        {"resources", "seeds", "budget", "baselines", "metrics", "correctness", "smoke"},
    )
    _text_list(plane["resources"], f"{label}.resources")
    seeds = _seed_list(plane["seeds"], f"{label}.seeds")
    _text_list(plane["baselines"], f"{label}.baselines")
    _text_list(plane["metrics"], f"{label}.metrics")
    _text_list(plane["correctness"], f"{label}.correctness")
    _nonempty_text(plane["smoke"], f"{label}.smoke")
    budget = _exact_fields(
        plane["budget"],
        f"{label}.budget",
        {"unit", "training_steps", "wall_clock_seconds", "hardware", "precision"},
    )
    _nonempty_text(budget["unit"], f"{label}.budget.unit")
    _nonempty_text(budget["hardware"], f"{label}.budget.hardware")
    _nonempty_text(budget["precision"], f"{label}.budget.precision")
    for field in ("training_steps", "wall_clock_seconds"):
        if type(budget[field]) is not int or budget[field] < 1:
            raise TaskDesignError(f"{label}.budget.{field} must be a positive integer")
    return seeds


def _validate_design(payload: dict[str, Any], canonical_task: Path) -> None:
    _exact_fields(
        payload,
        "task design",
        {
            "schema_version",
            "canonical_task",
            "method",
            "upstream",
            "candidate",
            "planes",
            "feedback",
            "publication",
        },
    )
    if payload["schema_version"] != 1:
        raise TaskDesignError("task design schema_version must be 1")
    canonical = _exact_fields(payload["canonical_task"], "canonical_task", {"path", "sha256"})
    if canonical["path"] != "task.md":
        raise TaskDesignError("canonical_task.path must be task.md")
    digest = hashlib.sha256(canonical_task.read_bytes()).hexdigest()
    if canonical["sha256"] != digest:
        raise TaskDesignError("canonical task hash does not match task.md")
    method = _exact_fields(payload["method"], "method", {"class", "claim"})
    _nonempty_text(method["class"], "method.class")
    _nonempty_text(method["claim"], "method.claim")
    _validate_upstream(payload["upstream"])
    candidate = _exact_fields(
        payload["candidate"],
        "candidate",
        {"interface", "editable_paths", "state_semantics"},
    )
    interface = _nonempty_text(candidate["interface"], "candidate.interface")
    if interface != "python_rpc_v1:handle":
        raise TaskDesignError("candidate.interface must be python_rpc_v1:handle")
    _candidate_editable_paths(candidate["editable_paths"])
    _nonempty_text(candidate["state_semantics"], "candidate.state_semantics")
    planes = _exact_fields(payload["planes"], "planes", {"search", "final"})
    search_seeds = _validate_plane(planes["search"], "planes.search")
    final_seeds = _validate_plane(planes["final"], "planes.final")
    if set(search_seeds).intersection(final_seeds):
        raise TaskDesignError("search and final seeds must be disjoint")
    feedback = _exact_fields(payload["feedback"], "feedback", {"search_visible", "final_sealed"})
    visible = set(_text_list(feedback["search_visible"], "feedback.search_visible"))
    sealed = set(_text_list(feedback["final_sealed"], "feedback.final_sealed"))
    if visible.intersection(sealed):
        raise TaskDesignError("search-visible and final-sealed feedback must be disjoint")
    publication = _exact_fields(payload["publication"], "publication", {"required_evidence"})
    _text_list(publication["required_evidence"], "publication.required_evidence")


def load_task_design(path: Path, *, canonical_task: Path) -> TaskDesign:
    payload = _load_json(path, "task design")
    _validate_design(payload, canonical_task)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return TaskDesign(payload=payload, fingerprint=hashlib.sha256(canonical.encode()).hexdigest())


def _task_design_from_payload(payload: dict[str, Any], canonical_task: Path) -> TaskDesign:
    _validate_design(payload, canonical_task)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return TaskDesign(payload=payload, fingerprint=hashlib.sha256(canonical.encode()).hexdigest())


def _decode_json_pointer(path: object, *, allow_protected: bool = False) -> list[str]:
    if not isinstance(path, str) or not path.startswith("/"):
        raise TaskDesignError("task design patch cannot replace the document root")
    tokens = path[1:].split("/")
    decoded: list[str] = []
    for token in tokens:
        index = 0
        value = ""
        while index < len(token):
            if token[index] != "~":
                value += token[index]
                index += 1
                continue
            if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
                raise TaskDesignError(f"invalid JSON pointer escape: {path}")
            value += "~" if token[index + 1] == "0" else "/"
            index += 2
        decoded.append(value)
    if not decoded or not decoded[0]:
        raise TaskDesignError("task design patch cannot replace the document root")
    if not allow_protected and decoded[0] in {"schema_version", "canonical_task"}:
        raise TaskDesignError(f"task design patch cannot modify {decoded[0]}")
    return decoded


def _array_index(token: str, length: int, path: str) -> int:
    if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
        raise TaskDesignError(f"invalid array index in JSON pointer: {path}")
    index = int(token)
    if index >= length:
        raise TaskDesignError(f"task design path does not exist: {path}")
    return index


def _resolve_pointer(root: object, tokens: list[str], path: str) -> object:
    target = root
    for token in tokens:
        if isinstance(target, dict) and token in target:
            target = target[token]
        elif isinstance(target, list):
            target = target[_array_index(token, len(target), path)]
        else:
            raise TaskDesignError(f"task design path does not exist: {path}")
    return target


def _replace_pointer(root: object, tokens: list[str], path: str, value: object) -> None:
    parent = _resolve_pointer(root, tokens[:-1], path)
    leaf = tokens[-1]
    if isinstance(parent, dict) and leaf in parent:
        parent[leaf] = value
    elif isinstance(parent, list):
        parent[_array_index(leaf, len(parent), path)] = value
    else:
        raise TaskDesignError(f"task design patch path does not exist: {path}")


def apply_task_design_patch(
    path: Path,
    *,
    parent: TaskDesign,
    canonical_task: Path,
    repair_review: TaskDesignReview,
) -> TaskDesign:
    """Apply a bounded replace-only patch to a validated parent design."""
    patch = _load_json(path, "task design patch")
    _exact_fields(
        patch,
        "task design patch",
        {"schema_version", "parent_fingerprint", "operations"},
    )
    if patch["schema_version"] != 1:
        raise TaskDesignError("task design patch schema_version must be 1")
    if patch["parent_fingerprint"] != parent.fingerprint:
        raise TaskDesignError("task design patch parent fingerprint does not match design")
    operations = patch["operations"]
    if not isinstance(operations, list) or not operations:
        raise TaskDesignError("task design patch operations must be a non-empty array")
    if len(operations) > MAX_DESIGN_PATCH_OPERATIONS:
        raise TaskDesignError(f"task design patch exceeds {MAX_DESIGN_PATCH_OPERATIONS} operations")
    candidate = deepcopy(parent.payload)
    if repair_review.verdict != "fail":
        raise TaskDesignError("task design repair review must fail")
    repair_path_field = (
        "repair_paths" if repair_review.payload["schema_version"] == 3 else "design_paths"
    )
    allowed_paths = tuple(
        _decode_json_pointer(repair_path, allow_protected=True)
        for row in repair_review.payload["dimensions"]
        if row["verdict"] == "fail"
        for repair_path in row.get(repair_path_field, [])
    )
    if not allowed_paths:
        raise TaskDesignError("task design repair review has no failed repair paths")
    seen_paths: set[str] = set()
    for index, operation in enumerate(operations):
        row = _exact_fields(operation, f"operations[{index}]", {"op", "path", "value"})
        if row["op"] != "replace":
            raise TaskDesignError("task design patch operations must use replace")
        pointer = row["path"]
        if not isinstance(pointer, str):
            raise TaskDesignError(f"operations[{index}].path must be a string")
        if pointer in seen_paths:
            raise TaskDesignError(f"duplicate task design patch path: {pointer}")
        seen_paths.add(pointer)
        tokens = _decode_json_pointer(pointer)
        if any(_paths_overlap(pointer, immutable) for immutable in IMMUTABLE_DESIGN_PATHS):
            raise TaskDesignError(f"task design patch path is immutable: {pointer}")
        if not any(tokens[: len(allowed)] == allowed for allowed in allowed_paths):
            raise TaskDesignError(
                f"task design patch path is outside failed repair paths: {pointer}"
            )
        _replace_pointer(candidate, tokens, pointer, deepcopy(row["value"]))
    repaired = _task_design_from_payload(candidate, canonical_task)
    if repaired.fingerprint == parent.fingerprint:
        raise TaskDesignError("task design patch must change the design")
    return repaired


def load_task_design_review(
    path: Path, *, design: TaskDesign, require_schema_version: int | None = None
) -> TaskDesignReview:
    payload = _load_json(path, "task design review")
    _exact_fields(
        payload,
        "task design review",
        {"schema_version", "verdict", "summary", "design_fingerprint", "dimensions"},
    )
    schema_version = payload["schema_version"]
    if schema_version not in {1, 2, 3}:
        raise TaskDesignError("task design review schema_version must be 1, 2, or 3")
    if require_schema_version is not None and schema_version != require_schema_version:
        raise TaskDesignError(f"task design review schema_version must be {require_schema_version}")
    if payload["design_fingerprint"] != design.fingerprint:
        raise TaskDesignError("task design review fingerprint does not match design")
    _nonempty_text(payload["summary"], "task design review summary")
    dimensions = payload["dimensions"]
    if not isinstance(dimensions, list):
        raise TaskDesignError("task design review dimensions must be an array")
    rows: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(dimensions):
        fields = {"id", "verdict", "evidence", "remediation"}
        if schema_version >= 2:
            fields.update({"design_paths", "task_evidence"})
        if schema_version == 3:
            fields.add("repair_paths")
        row = _exact_fields(value, f"dimensions[{index}]", fields)
        identifier = _nonempty_text(row["id"], f"dimensions[{index}].id")
        if identifier in rows:
            raise TaskDesignError(f"duplicate design dimension: {identifier}")
        rows[identifier] = row
    if set(rows) != set(DESIGN_DIMENSIONS):
        raise TaskDesignError("design review must contain exactly the required dimensions")
    failed: list[str] = []
    for identifier in DESIGN_DIMENSIONS:
        row = rows[identifier]
        verdict = row["verdict"]
        if verdict not in {"pass", "fail"}:
            raise TaskDesignError(f"dimension {identifier} verdict must be pass or fail")
        _text_list(row["evidence"], f"dimension {identifier} evidence")
        if schema_version >= 2:
            design_paths = _text_list(row["design_paths"], f"dimension {identifier} design_paths")
            _text_list(row["task_evidence"], f"dimension {identifier} task_evidence")
            for design_path in design_paths:
                tokens = _decode_json_pointer(design_path, allow_protected=True)
                try:
                    _resolve_pointer(design.payload, tokens, design_path)
                except TaskDesignError as exc:
                    raise TaskDesignError(
                        f"dimension {identifier} design path does not exist: {design_path}"
                    ) from exc
        repair_paths: list[str] = []
        if schema_version == 3:
            repair_paths = _possibly_empty_text_list(
                row["repair_paths"], f"dimension {identifier} repair_paths"
            )
            if verdict == "pass" and repair_paths:
                raise TaskDesignError(
                    f"passed dimension {identifier} must not have repair_paths"
                )
            for repair_path in repair_paths:
                try:
                    tokens = _decode_json_pointer(repair_path)
                except TaskDesignError as exc:
                    raise TaskDesignError(
                        f"dimension {identifier} repair path is not writable: {repair_path}"
                    ) from exc
                if any(
                    _paths_overlap(repair_path, immutable)
                    for immutable in IMMUTABLE_DESIGN_PATHS
                ):
                    raise TaskDesignError(
                        f"dimension {identifier} repair path is immutable: {repair_path}"
                    )
                try:
                    _resolve_pointer(design.payload, tokens, repair_path)
                except TaskDesignError as exc:
                    raise TaskDesignError(
                        f"dimension {identifier} repair path does not exist: {repair_path}"
                    ) from exc
                if not any(
                    repair_path == design_path or repair_path.startswith(f"{design_path}/")
                    for design_path in design_paths
                ):
                    raise TaskDesignError(
                        f"dimension {identifier} repair path lacks matching design evidence: "
                        f"{repair_path}"
                    )
        remediation = str(row["remediation"]).strip()
        if verdict == "fail":
            failed.append(identifier)
            if not remediation:
                raise TaskDesignError(f"failed dimension {identifier} requires remediation")
            if schema_version == 3 and not repair_paths:
                raise TaskDesignError(f"failed dimension {identifier} requires repair_paths")
        elif remediation:
            raise TaskDesignError(f"passed dimension {identifier} must not have remediation")
        elif schema_version == 3 and repair_paths:
            raise TaskDesignError(f"passed dimension {identifier} must not have repair_paths")
    expected = "fail" if failed else "pass"
    if payload["verdict"] != expected:
        raise TaskDesignError("design review verdict does not match dimensions")
    return TaskDesignReview(expected, tuple(failed), payload)


def _common_repair_scopes(rows: list[dict[str, Any]]) -> list[str]:
    common_scopes: list[str] = []
    for index, first in enumerate(rows):
        for second in rows[index + 1 :]:
            for first_path in first["repair_paths"]:
                for second_path in second["repair_paths"]:
                    if _paths_overlap(first_path, second_path):
                        common_scopes.append(
                            first_path if first_path.startswith(f"{second_path}/") else second_path
                        )
    return list(dict.fromkeys(common_scopes))


def _narrow_scope_alternatives(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    remediation_by_path: dict[str, str] = {}
    for row in rows:
        for path in row["repair_paths"]:
            remediation_by_path.setdefault(path, row["remediation"])
    return [
        {"repair_path": path, "remediation": remediation_by_path[path]}
        for path in sorted(remediation_by_path, key=lambda path: (-path.count("/"), path))
    ]


def task_design_scope_alternatives(
    reviews: list[TaskDesignReview],
    *,
    adjudication: TaskDesignAdjudication | None = None,
) -> dict[str, object]:
    """Return ordered, non-authorizing alternatives for unanimous scope conflicts."""
    if len(reviews) != 3:
        raise TaskDesignError("exactly three task design reviews are required")
    fingerprints = {review.payload["design_fingerprint"] for review in reviews}
    if len(fingerprints) != 1:
        raise TaskDesignError("task design reviews refer to different designs")
    review_rows = [{row["id"]: row for row in review.payload["dimensions"]} for review in reviews]
    decisions = adjudication.payload["decisions"] if adjudication is not None else []
    dimensions = []
    for identifier in DESIGN_DIMENSIONS:
        rows = [rows_by_id[identifier] for rows_by_id in review_rows]
        if all(row["verdict"] == "fail" for row in rows) and not _common_repair_scopes(rows):
            upheld = [
                row
                for row in decisions
                if row["verdict"] == "uphold" and row["canonical_dimension"] == identifier
            ]
            source_rows = [
                review_rows[row["source_review_index"]][row["source_dimension"]]
                for row in upheld
            ]
            dimensions.append(
                {
                    "id": identifier,
                    "alternatives": _narrow_scope_alternatives(rows),
                    "preserved_repair_paths": list(
                        dict.fromkeys(
                            path
                            for source_row, decision in zip(source_rows, upheld, strict=True)
                            for path in _common_repair_scopes([source_row, decision])
                        )
                    ),
                    "preserved_remediations": list(
                        dict.fromkeys(row["remediation"] for row in upheld)
                    ),
                }
            )
    return {
        "schema_version": 1,
        "design_fingerprint": fingerprints.pop(),
        "dimensions": dimensions,
    }


def task_design_adjudication_request(
    reviews: list[TaskDesignReview],
) -> dict[str, Any]:
    """Describe the minority findings that require a source-bound adjudication."""
    if len(reviews) != 3:
        raise TaskDesignError("exactly three task design reviews are required")
    fingerprints = {review.payload["design_fingerprint"] for review in reviews}
    if len(fingerprints) != 1:
        raise TaskDesignError("task design reviews refer to different designs")
    if {review.payload["schema_version"] for review in reviews} != {3}:
        raise TaskDesignError("task design adjudication requires schema version 3 reviews")
    review_rows = [
        {row["id"]: row for row in review.payload["dimensions"]} for review in reviews
    ]
    minority_findings = []
    for identifier in DESIGN_DIMENSIONS:
        failed = [
            (index, rows[identifier])
            for index, rows in enumerate(review_rows)
            if rows[identifier]["verdict"] == "fail"
        ]
        if len(failed) == 1:
            source_index, source_row = failed[0]
            minority_findings.append(
                {
                    "source_review_index": source_index,
                    "source_dimension": identifier,
                    "source_repair_paths": source_row["repair_paths"],
                }
            )
    request: dict[str, Any] = {
        "schema_version": 1,
        "design_fingerprint": fingerprints.pop(),
        "source_review_fingerprints": [
            hashlib.sha256(
                json.dumps(
                    review.payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            for review in reviews
        ],
        "minority_findings": minority_findings,
    }
    canonical = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    request["request_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return request


def load_task_design_adjudication(
    path: Path,
    *,
    design: TaskDesign,
    reviews: list[TaskDesignReview],
) -> TaskDesignAdjudication:
    """Load an adjudication bound to one ordered blind-review set."""
    payload = _load_json(path, "task design adjudication")
    _exact_fields(
        payload,
        "task design adjudication",
        {
            "schema_version",
            "design_fingerprint",
            "request_fingerprint",
            "source_review_fingerprints",
            "decisions",
        },
    )
    if payload["schema_version"] != 1:
        raise TaskDesignError("task design adjudication schema_version must be 1")
    expected = task_design_adjudication_request(reviews)
    if not expected["minority_findings"]:
        raise TaskDesignError("task design adjudication was not requested")
    if payload["design_fingerprint"] != design.fingerprint:
        raise TaskDesignError("task design adjudication fingerprint does not match design")
    if (
        payload["request_fingerprint"] != expected["request_fingerprint"]
        or payload["source_review_fingerprints"] != expected["source_review_fingerprints"]
    ):
        raise TaskDesignError("task design adjudication does not match source reviews")
    expected_findings = {
        (row["source_review_index"], row["source_dimension"]): row
        for row in expected["minority_findings"]
    }
    decisions = payload["decisions"]
    if not isinstance(decisions, list):
        raise TaskDesignError("task design adjudication decisions must be an array")
    seen: set[tuple[int, str]] = set()
    decision_keys: list[tuple[int, str]] = []
    review_rows = [{row["id"]: row for row in review.payload["dimensions"]} for review in reviews]
    for index, value in enumerate(decisions):
        row = _exact_fields(
            value,
            f"adjudication decisions[{index}]",
            {
                "source_review_index",
                "source_dimension",
                "verdict",
                "canonical_dimension",
                "evidence",
                "design_paths",
                "task_evidence",
                "repair_paths",
                "remediation",
            },
        )
        source_index = row["source_review_index"]
        if type(source_index) is not int or not 0 <= source_index < len(reviews):
            raise TaskDesignError("task design adjudication source review index is invalid")
        source_dimension = _nonempty_text(
            row["source_dimension"], "task design adjudication source dimension"
        )
        key = (source_index, source_dimension)
        if key not in expected_findings:
            raise TaskDesignError("task design adjudication contains an unexpected decision")
        if key in seen:
            raise TaskDesignError("task design adjudication contains a duplicate decision")
        seen.add(key)
        decision_keys.append(key)
        canonical_dimension = _nonempty_text(
            row["canonical_dimension"], "task design adjudication canonical dimension"
        )
        if canonical_dimension not in DESIGN_DIMENSIONS:
            raise TaskDesignError("task design adjudication canonical dimension is invalid")
        verdict = row["verdict"]
        if verdict not in {"uphold", "reject"}:
            raise TaskDesignError("task design adjudication verdict must be uphold or reject")
        _text_list(row["evidence"], "task design adjudication evidence")
        design_paths = _text_list(row["design_paths"], "task design adjudication design paths")
        _text_list(row["task_evidence"], "task design adjudication task evidence")
        repair_paths = _possibly_empty_text_list(
            row["repair_paths"], "task design adjudication repair paths"
        )
        for pointer in design_paths:
            tokens = _decode_json_pointer(pointer, allow_protected=True)
            try:
                _resolve_pointer(design.payload, tokens, pointer)
            except TaskDesignError as exc:
                raise TaskDesignError(
                    f"task design adjudication path does not exist: {pointer}"
                ) from exc
        for pointer in repair_paths:
            try:
                tokens = _decode_json_pointer(pointer)
            except TaskDesignError as exc:
                raise TaskDesignError(
                    f"task design adjudication repair path is not writable: {pointer}"
                ) from exc
            try:
                _resolve_pointer(design.payload, tokens, pointer)
            except TaskDesignError as exc:
                raise TaskDesignError(
                    f"task design adjudication repair path does not exist: {pointer}"
                ) from exc
        remediation = str(row["remediation"]).strip()
        if verdict == "uphold":
            if not repair_paths or not remediation:
                raise TaskDesignError(
                    "upheld task design adjudication requires repair paths and remediation"
                )
            source_row = review_rows[source_index][source_dimension]
            if not _common_repair_scopes([source_row, row]):
                raise TaskDesignError(
                    "task design adjudication repair paths do not overlap the source finding"
                )
        elif repair_paths or remediation:
            raise TaskDesignError(
                "rejected task design adjudication cannot authorize repair"
            )
    if decision_keys != list(expected_findings):
        raise TaskDesignError("task design adjudication decisions do not match minority findings")
    return TaskDesignAdjudication(payload)


def aggregate_task_design_reviews(
    reviews: list[TaskDesignReview],
    *,
    adjudication: TaskDesignAdjudication | None = None,
    allow_unanimous_scope_fallback: bool = False,
) -> TaskDesignReview:
    """Combine three blind reviews, adjudicating minority failure findings."""
    if len(reviews) != 3:
        raise TaskDesignError("exactly three task design reviews are required")
    fingerprints = {review.payload["design_fingerprint"] for review in reviews}
    if len(fingerprints) != 1:
        raise TaskDesignError("task design reviews refer to different designs")
    schema_versions = {review.payload["schema_version"] for review in reviews}
    if schema_versions != {3}:
        raise TaskDesignError("task design review aggregation requires schema version 3")
    review_rows = [{row["id"]: row for row in review.payload["dimensions"]} for review in reviews]
    failed_rows_by_dimension = {
        identifier: [
            row[identifier] for row in review_rows if row[identifier]["verdict"] == "fail"
        ]
        for identifier in DESIGN_DIMENSIONS
    }
    for identifier, failed_rows in failed_rows_by_dimension.items():
        if len(failed_rows) < 2 or _common_repair_scopes(failed_rows):
            continue
        if not allow_unanimous_scope_fallback or len(failed_rows) != len(reviews):
            raise TaskDesignError(f"failed dimension {identifier} has no majority repair path")
    minority_failures = [
        identifier
        for identifier in DESIGN_DIMENSIONS
        if len(failed_rows_by_dimension[identifier]) == 1
    ]
    if minority_failures and adjudication is None:
        raise TaskDesignAdjudicationRequired(
            "minority task design failures require adjudication: "
            + ", ".join(minority_failures)
        )
    if not minority_failures and adjudication is not None:
        raise TaskDesignError("task design adjudication was not requested")
    if adjudication is not None:
        expected_request = task_design_adjudication_request(reviews)
        if (
            adjudication.payload["design_fingerprint"] not in fingerprints
            or adjudication.payload["request_fingerprint"]
            != expected_request["request_fingerprint"]
            or adjudication.payload["source_review_fingerprints"]
            != expected_request["source_review_fingerprints"]
        ):
            raise TaskDesignError("task design adjudication does not match source reviews")
    decisions = adjudication.payload["decisions"] if adjudication is not None else []
    upheld_by_dimension = {
        identifier: [
            row
            for row in decisions
            if row["verdict"] == "uphold" and row["canonical_dimension"] == identifier
        ]
        for identifier in DESIGN_DIMENSIONS
    }
    failed: list[str] = []
    dimensions: list[dict[str, Any]] = []
    used_scope_fallback = False
    for identifier in DESIGN_DIMENSIONS:
        rows = [rows_by_id[identifier] for rows_by_id in review_rows]
        failed_rows = failed_rows_by_dimension[identifier]
        adjudicated_failures = upheld_by_dimension[identifier]
        verdict = "fail" if len(failed_rows) >= 2 or adjudicated_failures else "pass"
        evidence_rows = [row for row in rows if row["verdict"] == verdict]
        if verdict == "pass" and not evidence_rows:
            evidence_rows = rows
        adjudicated_source_rows = [
            review_rows[row["source_review_index"]][row["source_dimension"]]
            for row in adjudicated_failures
        ]
        if verdict == "fail":
            evidence_rows = [
                *(failed_rows if len(failed_rows) >= 2 else []),
                *adjudicated_source_rows,
                *adjudicated_failures,
            ]
        source = {
            "id": identifier,
            "verdict": verdict,
            "evidence": list(
                dict.fromkeys(item for row in evidence_rows for item in row["evidence"])
            ),
            "design_paths": list(
                dict.fromkeys(item for row in evidence_rows for item in row["design_paths"])
            ),
            "task_evidence": list(
                dict.fromkeys(item for row in evidence_rows for item in row["task_evidence"])
            ),
            "repair_paths": [],
            "remediation": "",
        }
        if verdict == "fail":
            failed.append(identifier)
            majority_repair_paths: list[str] = []
            majority_remediation: list[str] = []
            if len(failed_rows) >= 2:
                majority_repair_paths = _common_repair_scopes(failed_rows)
                if majority_repair_paths:
                    majority_remediation = [row["remediation"] for row in failed_rows]
                else:
                    selected = _narrow_scope_alternatives(failed_rows)[0]
                    majority_repair_paths = [selected["repair_path"]]
                    majority_remediation = [selected["remediation"]]
                    used_scope_fallback = True
            adjudicated_repair_paths = [
                path
                for source_row, decision in zip(
                    adjudicated_source_rows, adjudicated_failures, strict=True
                )
                for path in _common_repair_scopes([source_row, decision])
            ]
            source["repair_paths"] = list(
                dict.fromkeys([*majority_repair_paths, *adjudicated_repair_paths])
            )
            source["remediation"] = " ".join(
                dict.fromkeys(
                    [
                        *majority_remediation,
                        *(row["remediation"] for row in adjudicated_failures),
                    ]
                )
            )
        dimensions.append(source)
    payload = {
        "schema_version": 3,
        "verdict": "fail" if failed else "pass",
        "summary": (
            "Consensus of independent task-design reviews with unanimous narrow-scope fallback."
            if used_scope_fallback
            else "Consensus of independent task-design reviews with minority-failure adjudication."
            if minority_failures
            else "Consensus of independent task-design reviews."
        ),
        "design_fingerprint": fingerprints.pop(),
        "dimensions": dimensions,
    }
    return TaskDesignReview(payload["verdict"], tuple(failed), payload)


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _changed_design_paths(parent: object, candidate: object, path: str = "") -> tuple[str, ...]:
    if isinstance(parent, dict) and isinstance(candidate, dict):
        changed: list[str] = []
        for key in sorted(parent.keys() | candidate.keys()):
            child_path = f"{path}/{_json_pointer_token(key)}"
            if key not in parent or key not in candidate:
                changed.append(child_path)
            else:
                changed.extend(_changed_design_paths(parent[key], candidate[key], child_path))
        return tuple(changed)
    if isinstance(parent, list) and isinstance(candidate, list):
        if len(parent) != len(candidate):
            return (path or "/",)
        return tuple(
            changed
            for index, (parent_item, candidate_item) in enumerate(zip(parent, candidate))
            for changed in _changed_design_paths(parent_item, candidate_item, f"{path}/{index}")
        )
    return () if type(parent) is type(candidate) and parent == candidate else (path or "/",)


def task_design_changed_paths(
    parent: TaskDesign,
    candidate: TaskDesign,
) -> tuple[str, ...]:
    """Return deterministic leaf JSON pointers changed between two validated designs."""
    return _changed_design_paths(parent.payload, candidate.payload)


def _paths_overlap(first: str, second: str) -> bool:
    return first == second or first.startswith(f"{second}/") or second.startswith(f"{first}/")


def compare_task_design_reviews(
    parent: TaskDesignReview,
    candidate: TaskDesignReview,
    *,
    parent_design: TaskDesign,
    candidate_design: TaskDesign,
) -> TaskDesignReviewProgress:
    """Describe whether a candidate strictly reduces the parent's failures."""
    parent_failed = set(parent.failed_dimensions)
    candidate_failed = set(candidate.failed_dimensions)
    candidate_rows = {row["id"]: row for row in candidate.payload["dimensions"]}
    changed_paths = _changed_design_paths(parent_design.payload, candidate_design.payload)
    candidate_cited_paths = {
        identifier: tuple(
            dict.fromkeys(
                [
                    *row.get("design_paths", []),
                    *row.get("repair_paths", []),
                ]
            )
        )
        for identifier, row in candidate_rows.items()
    }
    causally_regressed = {
        identifier
        for identifier in candidate_failed - parent_failed
        if (
            not (cited_paths := candidate_cited_paths[identifier])
            or any(
                _paths_overlap(changed_path, cited_path)
                for changed_path in changed_paths
                for cited_path in cited_paths
            )
        )
    }
    return TaskDesignReviewProgress(
        fixed=tuple(
            identifier
            for identifier in DESIGN_DIMENSIONS
            if identifier in parent_failed - candidate_failed
        ),
        still_failed=tuple(
            identifier
            for identifier in DESIGN_DIMENSIONS
            if identifier in parent_failed & candidate_failed
        ),
        regressed=tuple(
            identifier for identifier in DESIGN_DIMENSIONS if identifier in causally_regressed
        ),
    )


def _dimension_ids(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TaskDesignError(f"{label} must be an array")
    identifiers = tuple(_nonempty_text(item, f"{label}[]") for item in value)
    if len(set(identifiers)) != len(identifiers):
        raise TaskDesignError(f"{label} must contain unique values")
    unknown = set(identifiers) - set(DESIGN_DIMENSIONS)
    if unknown:
        raise TaskDesignError(f"{label} contains unknown dimensions: {sorted(unknown)}")
    return identifiers


def load_task_design_delta_review(
    path: Path, *, parent: TaskDesign, candidate: TaskDesign
) -> TaskDesignDeltaReview:
    """Load a regression-specific comparison between parent and candidate designs."""
    payload = _load_json(path, "task design delta review")
    _exact_fields(
        payload,
        "task design delta review",
        {
            "schema_version",
            "verdict",
            "summary",
            "parent_fingerprint",
            "candidate_fingerprint",
            "fixed_dimensions",
            "regressions",
        },
    )
    schema_version = payload["schema_version"]
    if schema_version not in {1, 2}:
        raise TaskDesignError("task design delta review schema_version must be 1 or 2")
    if payload["parent_fingerprint"] != parent.fingerprint:
        raise TaskDesignError("task design delta parent fingerprint does not match")
    if payload["candidate_fingerprint"] != candidate.fingerprint:
        raise TaskDesignError("task design delta candidate fingerprint does not match")
    changed_paths = _changed_design_paths(parent.payload, candidate.payload)
    _nonempty_text(payload["summary"], "task design delta review summary")
    fixed = _dimension_ids(payload["fixed_dimensions"], "fixed_dimensions")
    regressions = payload["regressions"]
    if not isinstance(regressions, list):
        raise TaskDesignError("task design delta regressions must be an array")
    regressed: list[str] = []
    for index, value in enumerate(regressions):
        regression_fields = {"dimension", "design_paths", "evidence", "remediation"}
        if schema_version == 2:
            regression_fields.add("repair_paths")
        row = _exact_fields(value, f"regressions[{index}]", regression_fields)
        identifier = _nonempty_text(row["dimension"], f"regressions[{index}].dimension")
        if identifier not in DESIGN_DIMENSIONS:
            raise TaskDesignError(f"unknown regressed dimension: {identifier}")
        if identifier in regressed:
            raise TaskDesignError(f"duplicate regressed dimension: {identifier}")
        regressed.append(identifier)
        for design_path in _text_list(row["design_paths"], f"regressions[{index}].design_paths"):
            try:
                _resolve_pointer(
                    candidate.payload,
                    _decode_json_pointer(design_path, allow_protected=True),
                    design_path,
                )
            except TaskDesignError as exc:
                raise TaskDesignError(
                    f"regression design path does not exist: {design_path}"
                ) from exc
            if not any(_paths_overlap(design_path, changed) for changed in changed_paths):
                raise TaskDesignError(
                    f"regression design path was not changed by the candidate: {design_path}"
                )
        if schema_version == 2:
            for repair_path in _text_list(
                row["repair_paths"], f"regressions[{index}].repair_paths"
            ):
                try:
                    _resolve_pointer(
                        candidate.payload,
                        _decode_json_pointer(repair_path, allow_protected=True),
                        repair_path,
                    )
                except TaskDesignError as exc:
                    raise TaskDesignError(
                        f"regression repair path does not exist: {repair_path}"
                    ) from exc
        _text_list(row["evidence"], f"regressions[{index}].evidence")
        _nonempty_text(row["remediation"], f"regressions[{index}].remediation")
    expected = "fail" if regressed else "pass"
    if payload["verdict"] != expected:
        raise TaskDesignError("task design delta verdict does not match regressions")
    return TaskDesignDeltaReview(expected, fixed, tuple(regressed), payload)


def confirm_task_design_delta(
    delta: TaskDesignDeltaReview,
    progress: TaskDesignReviewProgress,
    candidate_review: TaskDesignReview,
) -> None:
    """Require the delta auditor to account for every consensus-claimed fix.

    The full review identifies candidate failures; the delta audit independently
    decides whether the repair caused them.  A disputed regression may therefore
    remain as a failed, quarantined frontier finding without blocking subsequent
    repair.
    """
    fixed = set(delta.fixed_dimensions)
    regressed = set(delta.regressed_dimensions)
    claimed = set(progress.fixed)
    claimed_regressed = set(progress.regressed)
    candidate_rows = {row["id"]: row for row in candidate_review.payload["dimensions"]}
    if fixed - claimed or not claimed <= fixed | regressed:
        raise TaskDesignError("task design delta fixed dimensions do not match consensus progress")
    if claimed_regressed and delta.payload["schema_version"] < 2:
        raise TaskDesignError(
            "task design delta schema_version 1 cannot authorize a repair frontier"
        )
    delta_rows = {row["dimension"]: row for row in delta.payload["regressions"]}
    for identifier in claimed_regressed & regressed:
        if delta_rows[identifier]["repair_paths"] != candidate_rows[identifier]["repair_paths"]:
            raise TaskDesignError("task design delta repair paths do not match candidate consensus")


def _main() -> int:
    from .design_cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
