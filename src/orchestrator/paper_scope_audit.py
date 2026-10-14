"""Validation helpers for edit-mode method scope audits."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def validate_method_scope_audit(output_dir: Path) -> dict[str, Any]:
    """Validate the edit-mode scope-audit artifacts against the manuscript."""
    output_dir = output_dir.expanduser().resolve()
    paper_tex_path = output_dir / "paper.tex"
    audit_json_path = output_dir / "method_scope_audit.json"
    audit_md_path = output_dir / "method_scope_audit.md"
    edit_context_json_path = output_dir / "paper_edit_context.json"

    hard_errors: list[str] = []
    soft_warnings: list[str] = []
    required_paper_text_hits: list[str] = []
    forbidden_text_hits: list[str] = []

    if not audit_json_path.is_file():
        hard_errors.append(f"missing required edit-scope artifact: {audit_json_path}")
    if not audit_md_path.is_file():
        hard_errors.append(f"missing required edit-scope artifact: {audit_md_path}")
    if not paper_tex_path.is_file():
        hard_errors.append(f"missing manuscript required for scope audit validation: {paper_tex_path}")

    if hard_errors:
        return {
            "ok": False,
            "hard_errors": hard_errors,
            "soft_warnings": soft_warnings,
            "required_paper_text_hits": required_paper_text_hits,
            "forbidden_text_hits": forbidden_text_hits,
        }

    try:
        audit_payload = json.loads(audit_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        hard_errors.append(f"method_scope_audit.json is not valid JSON: {exc}")
        return {
            "ok": False,
            "hard_errors": hard_errors,
            "soft_warnings": soft_warnings,
            "required_paper_text_hits": required_paper_text_hits,
            "forbidden_text_hits": forbidden_text_hits,
        }

    if not isinstance(audit_payload, dict):
        hard_errors.append("method_scope_audit.json must contain a top-level object")
        return {
            "ok": False,
            "hard_errors": hard_errors,
            "soft_warnings": soft_warnings,
            "required_paper_text_hits": required_paper_text_hits,
            "forbidden_text_hits": forbidden_text_hits,
        }

    required_list_fields = (
        "shared_method_elements",
        "scoped_method_elements",
        "claims_requiring_scoping",
    )
    for field_name in required_list_fields:
        value = audit_payload.get(field_name)
        if not isinstance(value, list):
            hard_errors.append(f"method_scope_audit.json field '{field_name}' must be a list")

    manuscript_focus = audit_payload.get("manuscript_focus")
    if not isinstance(manuscript_focus, dict):
        hard_errors.append("method_scope_audit.json field 'manuscript_focus' must be an object")
    else:
        for field_name in ("shared_method_summary", "scope_strategy"):
            value = manuscript_focus.get(field_name)
            if not isinstance(value, str) or not value.strip():
                hard_errors.append(
                    f"method_scope_audit.json field 'manuscript_focus.{field_name}' must be a non-empty string"
                )

    shared_method_elements = audit_payload.get("shared_method_elements", [])
    if isinstance(shared_method_elements, list):
        if not shared_method_elements:
            hard_errors.append("method_scope_audit.json must list at least one shared_method_element")
        elif not all(isinstance(item, str) and item.strip() for item in shared_method_elements):
            hard_errors.append("every shared_method_elements entry must be a non-empty string")

    scoped_method_elements = audit_payload.get("scoped_method_elements", [])
    if isinstance(scoped_method_elements, list):
        for idx, item in enumerate(scoped_method_elements):
            if not isinstance(item, dict):
                hard_errors.append(f"scoped_method_elements[{idx}] must be an object")
                continue
            for field_name in ("element", "scope", "rationale"):
                value = item.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    hard_errors.append(
                        f"scoped_method_elements[{idx}].{field_name} must be a non-empty string"
                    )
            evidence_paths = item.get("evidence_paths")
            if not isinstance(evidence_paths, list) or not evidence_paths or not all(
                isinstance(path, str) and path.strip() for path in evidence_paths
            ):
                hard_errors.append(
                    f"scoped_method_elements[{idx}].evidence_paths must be a list of non-empty strings"
                )

    claims_requiring_scoping = audit_payload.get("claims_requiring_scoping", [])
    normalized_required: list[str] = []
    normalized_forbidden: list[str] = []
    if isinstance(claims_requiring_scoping, list):
        for idx, item in enumerate(claims_requiring_scoping):
            if not isinstance(item, dict):
                hard_errors.append(f"claims_requiring_scoping[{idx}] must be an object")
                continue
            for field_name in ("claim", "scope", "reason"):
                value = item.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    hard_errors.append(
                        f"claims_requiring_scoping[{idx}].{field_name} must be a non-empty string"
                    )
            for field_name, sink in (
                ("required_paper_text", normalized_required),
                ("forbidden_unscoped_text", normalized_forbidden),
            ):
                values = item.get(field_name)
                if not isinstance(values, list) or not values:
                    hard_errors.append(
                        f"claims_requiring_scoping[{idx}].{field_name} must be a non-empty list of strings"
                    )
                    continue
                if not all(isinstance(value, str) and value.strip() for value in values):
                    hard_errors.append(
                        f"claims_requiring_scoping[{idx}].{field_name} entries must be non-empty strings"
                    )
                    continue
                sink.extend(_normalize_text(value) for value in values)

    markdown_text = audit_md_path.read_text(encoding="utf-8", errors="replace").strip()
    if not markdown_text:
        hard_errors.append("method_scope_audit.md must not be empty")

    paper_text = paper_tex_path.read_text(encoding="utf-8", errors="replace")
    normalized_paper = _normalize_text(paper_text)

    for raw_text, normalized_text in zip(
        _flatten_claim_texts(claims_requiring_scoping, "required_paper_text"),
        normalized_required,
        strict=False,
    ):
        if normalized_text not in normalized_paper:
            hard_errors.append(
                "paper.tex does not include required scoped text from method_scope_audit.json: "
                f"{raw_text}"
            )
        else:
            required_paper_text_hits.append(raw_text)

    for raw_text, normalized_text in zip(
        _flatten_claim_texts(claims_requiring_scoping, "forbidden_unscoped_text"),
        normalized_forbidden,
        strict=False,
    ):
        if normalized_text in normalized_paper:
            hard_errors.append(
                "paper.tex still contains forbidden unscoped text from method_scope_audit.json: "
                f"{raw_text}"
            )
            forbidden_text_hits.append(raw_text)

    staged_program_count = _count_staged_program_inputs(edit_context_json_path)
    if staged_program_count > 0 and not claims_requiring_scoping:
        soft_warnings.append(
            "method_scope_audit.json declares no claims_requiring_scoping despite staged program inputs; "
            "confirm that the implementations are truly manuscript-compatible without additional scoping"
        )
    if staged_program_count > 1 and not scoped_method_elements:
        soft_warnings.append(
            "method_scope_audit.json declares no scoped_method_elements despite multiple staged implementations; "
            "confirm that there are no setting-specific mechanisms to separate"
        )

    return {
        "ok": not hard_errors,
        "hard_errors": hard_errors,
        "soft_warnings": soft_warnings,
        "required_paper_text_hits": required_paper_text_hits,
        "forbidden_text_hits": forbidden_text_hits,
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _flatten_claim_texts(claims: Any, field_name: str) -> list[str]:
    if not isinstance(claims, list):
        return []
    flattened: list[str] = []
    for item in claims:
        if not isinstance(item, dict):
            continue
        values = item.get(field_name)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                flattened.append(value.strip())
    return flattened


def _count_staged_program_inputs(edit_context_json_path: Path) -> int:
    if not edit_context_json_path.is_file():
        return 0
    try:
        payload = json.loads(edit_context_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0
    edit_rows = payload.get("edit_programs", [])
    old_rows = payload.get("old_programs", [])
    count = 0
    if isinstance(edit_rows, list):
        count += sum(1 for item in edit_rows if isinstance(item, dict))
    if isinstance(old_rows, list):
        count += sum(1 for item in old_rows if isinstance(item, dict))
    return count
