"""Reference query plan helpers for paper generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_QUERY_CATEGORIES = frozenset(
    {
        "core_method",
        "baseline_comparator",
        "benchmark_context",
        "architectural_shift",
        "result_driven_theme",
    }
)
MAX_REFERENCE_QUERY_COUNT = 10


def utc_now() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def load_reference_query_plan(path: Path) -> dict[str, Any]:
    """Load a reference query plan JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def validate_reference_query_plan_payload(
    payload: dict[str, Any],
    *,
    max_queries: int = MAX_REFERENCE_QUERY_COUNT,
) -> None:
    """Validate a reference query plan payload."""
    status = str(payload.get("status", "")).strip()
    if status not in {"ready", "fail"}:
        raise ValueError("reference_query_plan.json must set status to 'ready' or 'fail'")

    if status == "fail":
        failure_reason = str(payload.get("failure_reason", "")).strip()
        if not failure_reason:
            raise ValueError("reference_query_plan.json status=fail requires non-empty failure_reason")
        return

    if not str(payload.get("central_topic", "")).strip():
        raise ValueError("reference_query_plan.json missing non-empty central_topic")

    query_rows = payload.get("queries")
    if not isinstance(query_rows, list) or not query_rows:
        raise ValueError("reference_query_plan.json must contain non-empty queries list")
    if len(query_rows) > max_queries:
        raise ValueError(
            f"reference_query_plan.json queries exceeds max {max_queries}: {len(query_rows)}"
        )

    seen_queries: set[str] = set()
    for idx, row in enumerate(query_rows):
        if not isinstance(row, dict):
            raise ValueError(f"reference_query_plan.json queries[{idx}] must be an object")
        query = " ".join(str(row.get("query", "")).split()).strip()
        rationale = " ".join(str(row.get("rationale", "")).split()).strip()
        category = str(row.get("category", "")).strip()
        if not query:
            raise ValueError(f"reference_query_plan.json queries[{idx}] missing non-empty query")
        if len(query) > 200:
            raise ValueError(f"reference_query_plan.json queries[{idx}].query exceeds 200 chars")
        if not rationale:
            raise ValueError(f"reference_query_plan.json queries[{idx}] missing non-empty rationale")
        if category not in VALID_QUERY_CATEGORIES:
            allowed = ", ".join(sorted(VALID_QUERY_CATEGORIES))
            raise ValueError(
                f"reference_query_plan.json queries[{idx}].category must be one of: {allowed}"
            )
        key = query.lower()
        if key in seen_queries:
            raise ValueError(f"reference_query_plan.json queries[{idx}] duplicates query: {query}")
        seen_queries.add(key)

    for list_field in (
        "core_methods",
        "comparators",
        "benchmarks",
        "architectural_shifts",
        "result_driven_themes",
        "forbidden_query_patterns",
    ):
        value = payload.get(list_field, [])
        if not isinstance(value, list):
            raise ValueError(f"reference_query_plan.json field must be a list: {list_field}")


def reference_query_texts(payload: dict[str, Any]) -> list[str]:
    """Extract ordered query texts from a validated plan payload."""
    query_rows = payload.get("queries", [])
    out: list[str] = []
    for row in query_rows:
        if not isinstance(row, dict):
            continue
        query = " ".join(str(row.get("query", "")).split()).strip()
        if query:
            out.append(query)
    return out


def reference_query_plan_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return compact metadata from a validated query plan for provenance files."""
    return {
        "status": str(payload.get("status", "")).strip(),
        "central_topic": str(payload.get("central_topic", "")).strip(),
        "core_methods": payload.get("core_methods", []),
        "comparators": payload.get("comparators", []),
        "benchmarks": payload.get("benchmarks", []),
        "architectural_shifts": payload.get("architectural_shifts", []),
        "result_driven_themes": payload.get("result_driven_themes", []),
        "forbidden_query_patterns": payload.get("forbidden_query_patterns", []),
        "queries": payload.get("queries", []),
    }


def write_reference_query_plan_validation_feedback(
    *,
    output_dir: Path,
    feedback_json_path: Path,
    feedback_md_path: Path,
    plan_md_path: Path,
    plan_json_path: Path,
    max_queries: int = MAX_REFERENCE_QUERY_COUNT,
) -> dict[str, Any]:
    """Validate reference query plan artifacts and persist feedback."""
    feedback: dict[str, Any] = {
        "generated_at": utc_now(),
        "paper_dir": str(output_dir.resolve()),
        "ok": False,
        "hard_errors": [],
    }
    try:
        if not plan_md_path.is_file():
            raise ValueError(f"Missing required reference query plan memo: {plan_md_path}")
        if not plan_json_path.is_file():
            raise ValueError(f"Missing required reference query plan JSON: {plan_json_path}")
        if not plan_md_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Reference query plan memo is empty: {plan_md_path}")
        payload = load_reference_query_plan(plan_json_path)
        validate_reference_query_plan_payload(payload, max_queries=max_queries)
        if str(payload.get("status", "")).strip() == "fail":
            raise ValueError(
                f"reference query planning failed: {str(payload.get('failure_reason', '')).strip()}"
            )
    except ValueError as exc:
        feedback["hard_errors"] = [str(exc)]
    else:
        feedback["ok"] = True

    feedback_json_path.write_text(json.dumps(feedback, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Reference query plan validation feedback",
        "",
        f"- generated_at: `{feedback['generated_at']}`",
        f"- paper_dir: `{feedback['paper_dir']}`",
        f"- ok: `{feedback['ok']}`",
        "",
    ]
    hard_errors = feedback.get("hard_errors", [])
    if isinstance(hard_errors, list) and hard_errors:
        lines.append("## Hard errors")
        lines.extend(f"- {message}" for message in hard_errors)
        lines.append("")
    feedback_md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return feedback
