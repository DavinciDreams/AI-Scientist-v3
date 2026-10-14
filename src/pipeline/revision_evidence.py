"""Evidence-retention checks for review-driven manuscript revisions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from src.orchestrator.paper_evidence_audit import audit_input_sha256
from src.orchestrator.paper_scope_audit import validate_method_scope_audit

from .revision_publication import _read_regular

_MAX_JSON_BYTES = 32 * 1024 * 1024
_VOLATILE_KEYS = {
    "completed_at",
    "created_at",
    "generated_at",
    "started_at",
    "timestamp",
    "updated_at",
    "verified_at",
}
_LOCATION_KEYS = {"path", "source_csv"}
_ABSOLUTE_LOCATION = "<absolute-location>"
_POSTHOC_EDITORIAL_FIELDS = {
    "key_finding_for_paper",
    "relevance_justification",
    "recommended_usage",
    "relevance_to_thesis",
}
_LEGACY_EVIDENCE_LABELS = {
    "all_programs": "program inventory",
    "task_names": "task inventory",
    "cross_task_results": "task inventory",
    "grounding_sources": "grounding inventory",
    "ablation_studies": "ablation inventory",
    "posthoc_analyses": "posthoc inventory",
    "final_evaluation": "final evaluation inventory",
    "downstream_study": "downstream study context",
}
_COMPLETION_MANIFEST_AUDIT_PATH = "evidence-source/latest_manifest.json"
AuditHashesMatch = Callable[[object, dict[str, str]], bool]


def validate_revision_evidence(source_dir: Path, candidate_dir: Path) -> None:
    """Reject a revision that loses staged evidence or lacks passing audits."""
    _validate_semantic_evidence(source_dir, candidate_dir)
    _validate_passing_audits(candidate_dir, _candidate_audit_hashes_match)


def validate_published_revision_evidence(source_dir: Path, candidate_dir: Path) -> None:
    """Validate a promoted paper while ignoring only publication metadata churn."""
    _validate_semantic_evidence(source_dir, candidate_dir)
    _validate_passing_audits(candidate_dir, _published_audit_hashes_match)


def _validate_semantic_evidence(source_dir: Path, candidate_dir: Path) -> None:
    source = _immutable_paper_evidence(
        _read_json_object(source_dir / "paper_data.json", "source paper data")
    )
    candidate = _immutable_paper_evidence(
        _read_json_object(candidate_dir / "paper_data.json", "revision paper data")
    )

    if not _contains_normalized(candidate, source):
        for field, label in _LEGACY_EVIDENCE_LABELS.items():
            if field in source and not _contains_normalized(candidate.get(field), source[field]):
                raise ValueError(f"revision loses or changes {label}")
        raise ValueError("revision loses or changes paper data evidence")


def _immutable_paper_evidence(paper_data: dict[str, Any]) -> dict[str, Any]:
    """Separate direct posthoc editorial annotations from retained evidence."""
    analyses = paper_data.get("posthoc_analyses")
    if not isinstance(analyses, list):
        return paper_data
    return {
        **paper_data,
        "posthoc_analyses": [
            {key: value for key, value in entry.items() if key not in _POSTHOC_EDITORIAL_FIELDS}
            if isinstance(entry, dict)
            else entry
            for entry in analyses
        ],
    }


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(_read_regular(path, max_bytes=_MAX_JSON_BYTES).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _normalize(
    value: object,
    *,
    key: str | None = None,
    location_container: bool = False,
) -> object:
    if isinstance(value, dict):
        return {
            item_key: _normalize(
                item_value,
                key=item_key,
                location_container=key == "artifacts",
            )
            for item_key, item_value in value.items()
            if item_key not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [
            _normalize(item, key=key, location_container=location_container) for item in value
        ]
    is_location = location_container or key in _LOCATION_KEYS or bool(
        key and key.endswith(("_dir", "_path"))
    )
    if isinstance(value, str) and is_location and os.path.isabs(value):
        return _ABSOLUTE_LOCATION
    return value


def _contains_normalized(candidate: object, required: object) -> bool:
    return _contains(_normalize(candidate), _normalize(required))


def _contains(candidate: object, required: object) -> bool:
    if isinstance(required, dict):
        return isinstance(candidate, dict) and all(
            key in candidate and _contains(candidate[key], value)
            for key, value in required.items()
        )
    if isinstance(required, list):
        if not isinstance(candidate, list) or len(candidate) < len(required):
            return False
        ordered_required = sorted(required, key=_specificity, reverse=True)
        matched_required = [-1] * len(candidate)

        def assign(required_index: int, seen: set[int]) -> bool:
            required_item = ordered_required[required_index]
            for candidate_index, candidate_item in enumerate(candidate):
                if candidate_index in seen or not _contains(candidate_item, required_item):
                    continue
                seen.add(candidate_index)
                previous = matched_required[candidate_index]
                if previous == -1 or assign(previous, seen):
                    matched_required[candidate_index] = required_index
                    return True
            return False

        for required_index in range(len(ordered_required)):
            if not assign(required_index, set()):
                return False
        return True
    return candidate == required


def _specificity(value: object) -> int:
    if isinstance(value, dict):
        return 1 + sum(_specificity(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_specificity(item) for item in value)
    return 1


def _candidate_audit_hashes(hashes: dict[str, str]) -> dict[str, str]:
    """Exclude the completion manifest written by the host after the final audit."""
    return {
        path: digest
        for path, digest in hashes.items()
        if path != _COMPLETION_MANIFEST_AUDIT_PATH
    }


def _candidate_audit_hashes_match(recorded: object, current: dict[str, str]) -> bool:
    return isinstance(recorded, dict) and _candidate_audit_hashes(recorded) == (
        _candidate_audit_hashes(current)
    )


def _published_audit_hashes(hashes: dict[str, str]) -> dict[str, str]:
    return {
        path: digest
        for path, digest in hashes.items()
        if path != _COMPLETION_MANIFEST_AUDIT_PATH
        and not path.startswith("evidence-source/ai_review.")
        and not path.startswith("evidence-source/revisions/")
    }


def _published_audit_hashes_match(recorded: object, current: dict[str, str]) -> bool:
    return isinstance(recorded, dict) and _published_audit_hashes(recorded) == (
        _published_audit_hashes(current)
    )


def _validate_passing_audits(candidate_dir: Path, hashes_match: AuditHashesMatch) -> None:
    for name in ("method_scope_audit.json", "method_scope_audit.md"):
        path = candidate_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"revision lacks a valid method-scope audit: {name}")
    try:
        method_scope = validate_method_scope_audit(candidate_dir)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("could not verify revision method-scope audit") from exc
    if method_scope.get("ok") is not True:
        raise ValueError("revision lacks a valid method-scope audit")

    audit = _read_json_object(
        candidate_dir / "paper_evidence_audit.json",
        "agentic paper-evidence audit",
    )
    try:
        expected_hashes = audit_input_sha256(candidate_dir)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("could not verify current agentic evidence audit") from exc
    audit_passes = (
        audit.get("verdict") == "pass"
        and audit.get("finding_count") == 0
        and audit.get("findings") == []
        and bool(expected_hashes)
        and hashes_match(audit.get("input_sha256"), expected_hashes)
    )

    report = _read_json_object(candidate_dir / "validation_report.json", "validation report")
    report_audit = report.get("agentic_evidence_audit")
    report_passes = (
        report.get("ok") is True
        and isinstance(report_audit, dict)
        and report_audit.get("checked") is True
        and report_audit.get("verdict") == "pass"
        and report_audit.get("finding_count") == 0
        and report_audit.get("findings") == []
        and hashes_match(report_audit.get("input_sha256"), expected_hashes)
    )
    if not audit_passes or not report_passes:
        raise ValueError("revision lacks a current passing agentic evidence audit")
