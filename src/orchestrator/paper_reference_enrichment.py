"""Post-draft citation enrichment helpers for paper generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.orchestrator.semantic_scholar import extract_bibtex_key

VALID_GAP_PRIORITIES = frozenset({"high", "medium", "low"})
VALID_CITATION_INTENTS = frozenset(
    {
        "background",
        "positioning",
        "comparator",
        "benchmark_context",
    }
)


def utc_now() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def load_draft_reference_gaps(path: Path) -> dict[str, Any]:
    """Load a draft reference gaps JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def validate_draft_reference_gaps_payload(payload: dict[str, Any]) -> None:
    """Validate a draft reference gap payload."""
    status = str(payload.get("status", "")).strip()
    if status not in {"ready", "no_gaps", "fail"}:
        raise ValueError("draft_reference_gaps.json must set status to 'ready', 'no_gaps', or 'fail'")

    summary = " ".join(str(payload.get("summary", "")).split()).strip()
    if not summary:
        raise ValueError("draft_reference_gaps.json missing non-empty summary")

    rewrite_targets = payload.get("rewrite_targets", [])
    if not isinstance(rewrite_targets, list):
        raise ValueError("draft_reference_gaps.json field must be a list: rewrite_targets")

    missing_literatures = payload.get("missing_literatures", [])
    if not isinstance(missing_literatures, list):
        raise ValueError("draft_reference_gaps.json field must be a list: missing_literatures")

    if status == "fail":
        failure_reason = str(payload.get("failure_reason", "")).strip()
        if not failure_reason:
            raise ValueError("draft_reference_gaps.json status=fail requires non-empty failure_reason")
        return

    if status == "no_gaps":
        if rewrite_targets:
            raise ValueError("draft_reference_gaps.json status=no_gaps requires empty rewrite_targets")
        return

    if not rewrite_targets:
        raise ValueError("draft_reference_gaps.json status=ready requires non-empty rewrite_targets")

    seen_target_ids: set[str] = set()
    for idx, row in enumerate(rewrite_targets):
        if not isinstance(row, dict):
            raise ValueError(f"draft_reference_gaps.json rewrite_targets[{idx}] must be an object")
        target_id = str(row.get("target_id", "")).strip()
        section = " ".join(str(row.get("section", "")).split()).strip()
        priority = str(row.get("priority", "")).strip()
        citation_intent = str(row.get("citation_intent", "")).strip()
        reason = " ".join(str(row.get("reason", "")).split()).strip()
        anchor_text = " ".join(str(row.get("anchor_text", "")).split()).strip()
        search_topics = row.get("search_topics", [])

        if not target_id:
            raise ValueError(f"draft_reference_gaps.json rewrite_targets[{idx}] missing non-empty target_id")
        if target_id in seen_target_ids:
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}] duplicates target_id: {target_id}"
            )
        seen_target_ids.add(target_id)
        if not section:
            raise ValueError(f"draft_reference_gaps.json rewrite_targets[{idx}] missing non-empty section")
        if priority not in VALID_GAP_PRIORITIES:
            allowed = ", ".join(sorted(VALID_GAP_PRIORITIES))
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}].priority must be one of: {allowed}"
            )
        if citation_intent not in VALID_CITATION_INTENTS:
            allowed = ", ".join(sorted(VALID_CITATION_INTENTS))
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}].citation_intent must be one of: {allowed}"
            )
        if not reason:
            raise ValueError(f"draft_reference_gaps.json rewrite_targets[{idx}] missing non-empty reason")
        if not anchor_text:
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}] missing non-empty anchor_text"
            )
        if not isinstance(search_topics, list) or not search_topics:
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}] must contain non-empty search_topics"
            )
        clean_topics = [" ".join(str(topic).split()).strip() for topic in search_topics]
        if any(not topic for topic in clean_topics):
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}] contains empty search topic"
            )


def write_draft_reference_gaps_validation_feedback(
    *,
    output_dir: Path,
    feedback_json_path: Path,
    feedback_md_path: Path,
    gaps_md_path: Path,
    gaps_json_path: Path,
) -> dict[str, Any]:
    """Validate draft reference gaps artifacts and persist feedback."""
    feedback: dict[str, Any] = {
        "generated_at": utc_now(),
        "paper_dir": str(output_dir.resolve()),
        "ok": False,
        "hard_errors": [],
    }
    try:
        if not gaps_md_path.is_file():
            raise ValueError(f"Missing required draft reference gaps memo: {gaps_md_path}")
        if not gaps_json_path.is_file():
            raise ValueError(f"Missing required draft reference gaps JSON: {gaps_json_path}")
        if not gaps_md_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Draft reference gaps memo is empty: {gaps_md_path}")
        payload = load_draft_reference_gaps(gaps_json_path)
        validate_draft_reference_gaps_payload(payload)
        if str(payload.get("status", "")).strip() == "fail":
            raise ValueError(
                f"draft reference gap analysis failed: {str(payload.get('failure_reason', '')).strip()}"
            )
    except ValueError as exc:
        feedback["hard_errors"] = [str(exc)]
    else:
        feedback["ok"] = True

    feedback_json_path.write_text(json.dumps(feedback, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Draft Reference Gaps Validation Feedback",
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


def _load_bibtex_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    entries: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("@") and current:
            entries.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append("\n".join(current).strip())

    rows: list[dict[str, str]] = []
    for entry in entries:
        key = extract_bibtex_key(entry)
        if key is None:
            continue
        rows.append({"citation_key": key, "bibtex": entry})
    return rows


def _load_provenance_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_provenance_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_provenance_payload(path)
    rows = payload.get("references")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _merge_query_lists(*query_lists: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for queries in query_lists:
        for query in queries:
            clean = " ".join(str(query).split()).strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(clean)
    return merged


def _merge_reference_query_plans(*payloads: dict[str, Any]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for payload in payloads:
        metadata = payload.get("reference_query_plan")
        if isinstance(metadata, dict) and metadata:
            plans.append(metadata)
        legacy_list = payload.get("reference_query_plans")
        if isinstance(legacy_list, list):
            plans.extend(row for row in legacy_list if isinstance(row, dict) and row)
    return plans


def _merge_reference_rows(
    *sources: tuple[list[dict[str, str]], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged_refs: list[dict[str, Any]] = []
    merged_prov: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for bib_rows, prov_rows in sources:
        prov_by_key: dict[str, dict[str, Any]] = {}
        for row in prov_rows:
            key = str(row.get("citation_key", "")).strip()
            if key:
                prov_by_key[key] = row
        for row in bib_rows:
            key = str(row.get("citation_key", "")).strip()
            bibtex = row.get("bibtex")
            if not key or key in seen_keys or not isinstance(bibtex, str) or not bibtex.strip():
                continue
            prov_row = dict(prov_by_key.get(key, {}))
            prov_row["citation_key"] = key
            prov_row["bibtex"] = bibtex.strip()
            merged_refs.append(prov_row)
            merged_prov.append({k: v for k, v in prov_row.items() if k != "bibtex"})
            seen_keys.add(key)
    return merged_refs, merged_prov


def merge_reference_artifacts(
    *,
    existing_bib_path: Path,
    existing_provenance_path: Path,
    incremental_bib_path: Path,
    incremental_provenance_path: Path,
    output_bib_path: Path,
    output_provenance_path: Path,
    results_dir: Path,
    campaign_results_dir: Path,
) -> dict[str, Any]:
    """Merge existing and incremental reference artifacts and write canonical outputs."""
    existing_payload = _load_provenance_payload(existing_provenance_path)
    incremental_payload = _load_provenance_payload(incremental_provenance_path)
    merged_refs, merged_prov = _merge_reference_rows(
        (_load_bibtex_rows(existing_bib_path), _load_provenance_rows(existing_provenance_path)),
        (_load_bibtex_rows(incremental_bib_path), _load_provenance_rows(incremental_provenance_path)),
    )
    if not merged_refs:
        raise ValueError("No BibTeX entries available to merge.")

    output_bib_path.write_text(
        "\n\n".join(str(row["bibtex"]).strip() for row in merged_refs) + "\n",
        encoding="utf-8",
    )
    payload = {
        "generated_at": incremental_payload.get("generated_at")
        or existing_payload.get("generated_at")
        or utc_now(),
        "results_dir": str(results_dir.resolve()),
        "campaign_results_dir": str(campaign_results_dir.resolve()),
        "queries": _merge_query_lists(
            existing_payload.get("queries", []),
            incremental_payload.get("queries", []),
        ),
        "reference_query_plan": incremental_payload.get("reference_query_plan")
        or existing_payload.get("reference_query_plan", {}),
        "reference_query_plans": _merge_reference_query_plans(existing_payload, incremental_payload),
        "count": len(merged_prov),
        "references": merged_prov,
    }
    output_provenance_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
