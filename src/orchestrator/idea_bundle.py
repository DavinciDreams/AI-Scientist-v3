"""Validation and normalization utilities for idea bundles."""

from __future__ import annotations

import json
import re
from typing import Any

NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]+")
MULTI_UNDERSCORE_RE = re.compile(r"_+")
MAX_SOURCE_PATTERN_CHARS = 1024


class IdeaBundleValidationError(RuntimeError):
    """Raised when idea bundle artifacts are missing required structure."""


def parse_agent_json(text: str) -> tuple[Any, bool]:
    """Parse agent-authored JSON and report whether safe repair was required."""

    def reject_constant(constant: str) -> None:
        raise json.JSONDecodeError(f"Invalid JSON constant {constant}", text, 0)

    def strict_loads(candidate: str) -> Any:
        return json.loads(candidate, parse_constant=reject_constant)

    try:
        return strict_loads(text), False
    except json.JSONDecodeError as original_error:
        repaired: list[str] = []
        in_string = False
        escaped = False
        removed_comma = False
        for index, character in enumerate(text):
            if in_string:
                repaired.append(character)
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
                repaired.append(character)
                continue
            if character == ",":
                previous_index = index - 1
                while previous_index >= 0 and text[previous_index].isspace():
                    previous_index -= 1
                next_index = index + 1
                while next_index < len(text) and text[next_index].isspace():
                    next_index += 1
                if (
                    previous_index >= 0
                    and text[previous_index] not in "[{,:"
                    and next_index < len(text)
                    and text[next_index] in "]}"
                ):
                    removed_comma = True
                    continue
            repaired.append(character)
        if not removed_comma:
            raise original_error
        try:
            return strict_loads("".join(repaired)), True
        except json.JSONDecodeError:
            raise original_error from None


def _slugify(text: str, *, fallback: str = "rubric") -> str:
    """Handle slugify logic for this module."""
    clean = NON_ALNUM_RE.sub("_", text.strip().lower())
    clean = MULTI_UNDERSCORE_RE.sub("_", clean).strip("_")
    return clean or fallback


def without_target_rubric_verification(payload: Any) -> Any:
    """Copy an agent payload without optional static-verification metadata."""

    def strip_rubrics(raw_rubrics: list[Any]) -> list[Any]:
        stripped: list[Any] = []
        for rubric in raw_rubrics:
            if not isinstance(rubric, dict):
                stripped.append(rubric)
                continue
            normalized_rubric = dict(rubric)
            normalized_rubric.pop("verification", None)
            stripped.append(normalized_rubric)
        return stripped

    if isinstance(payload, list):
        return strip_rubrics(payload)
    if isinstance(payload, dict) and isinstance(payload.get("target_rubrics"), list):
        normalized_payload = dict(payload)
        normalized_payload["target_rubrics"] = strip_rubrics(payload["target_rubrics"])
        return normalized_payload
    return payload


def _normalize_source_contains_verification(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("type") != "source_contains":
        raise IdeaBundleValidationError(
            f"target_rubrics[{index}].verification must use type source_contains"
        )
    patterns = raw.get("patterns")
    if not isinstance(patterns, list) or not 1 <= len(patterns) <= 8:
        raise IdeaBundleValidationError(
            f"target_rubrics[{index}].verification.patterns must contain 1 to 8 strings"
        )
    normalized_patterns: list[str] = []
    for pattern in patterns:
        if (
            not isinstance(pattern, str)
            or not pattern.strip()
            or len(pattern) > MAX_SOURCE_PATTERN_CHARS
        ):
            raise IdeaBundleValidationError(
                f"target_rubrics[{index}].verification.patterns must be non-empty strings "
                f"up to {MAX_SOURCE_PATTERN_CHARS} characters"
            )
        normalized_patterns.append(pattern.strip())
    return {"type": "source_contains", "patterns": normalized_patterns}


def normalize_target_rubrics(raw: Any) -> list[dict[str, Any]]:
    """Normalize target rubrics."""
    if not isinstance(raw, list):
        raise IdeaBundleValidationError("target_rubrics must be a list")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise IdeaBundleValidationError(f"target_rubrics[{idx}] must be an object")

        desc = item.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise IdeaBundleValidationError(
                f"target_rubrics[{idx}].description must be non-empty string"
            )

        rubric_type = item.get("type")
        if not isinstance(rubric_type, str) or not rubric_type.strip():
            rubric_type = "implementation"

        rubric_id = item.get("id")
        if not isinstance(rubric_id, str) or not rubric_id.strip():
            rubric_id = _slugify(desc)
        rubric_id = _slugify(rubric_id)

        dedupe = rubric_id
        suffix = 2
        while dedupe in seen_ids:
            dedupe = f"{rubric_id}_{suffix}"
            suffix += 1
        seen_ids.add(dedupe)

        normalized_rubric: dict[str, Any] = {
            "id": dedupe,
            "description": desc.strip(),
            "type": rubric_type.strip().lower(),
        }
        verification = item.get("verification")
        if verification is not None:
            if normalized_rubric["type"] != "implementation":
                raise IdeaBundleValidationError(
                    f"target_rubrics[{idx}].verification is only supported for implementation rubrics"
                )
            normalized_rubric["verification"] = _normalize_source_contains_verification(
                verification, idx
            )
        normalized.append(normalized_rubric)

    if not normalized:
        raise IdeaBundleValidationError("target_rubrics must not be empty")
    return normalized


def normalize_target_rubrics_payload(raw: Any) -> list[dict[str, Any]]:
    """Normalize target rubrics payload."""
    if isinstance(raw, dict):
        maybe = raw.get("target_rubrics")
        if isinstance(maybe, list):
            return normalize_target_rubrics(maybe)
    return normalize_target_rubrics(raw)


def normalize_idea_bundle(raw: Any, cycle_index: int) -> dict[str, Any]:
    """Normalize idea bundle."""
    if not isinstance(raw, dict):
        raise IdeaBundleValidationError("idea bundle must be a JSON object")

    idea_id = raw.get("idea_id")
    if not isinstance(idea_id, str) or not idea_id.strip():
        idea_id = f"idea_{cycle_index:04d}"
    idea_id = _slugify(idea_id, fallback=f"idea_{cycle_index:04d}")

    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        raise IdeaBundleValidationError("title must be non-empty string")

    hypothesis = raw.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise IdeaBundleValidationError("hypothesis must be non-empty string")

    plan = raw.get("implementation_plan")
    if not isinstance(plan, list) or not plan:
        raise IdeaBundleValidationError("implementation_plan must be a non-empty list")
    plan_items = [str(item).strip() for item in plan if str(item).strip()]
    if not plan_items:
        raise IdeaBundleValidationError("implementation_plan must contain non-empty items")

    target_rubrics = normalize_target_rubrics(raw.get("target_rubrics"))

    notes = raw.get("notes_for_agent")
    if not isinstance(notes, str):
        notes = ""

    return {
        "idea_id": idea_id,
        "title": title.strip(),
        "hypothesis": hypothesis.strip(),
        "implementation_plan": plan_items,
        "target_rubrics": target_rubrics,
        "notes_for_agent": notes.strip(),
    }
