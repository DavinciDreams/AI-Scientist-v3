"""Helpers for staging optional paper-context docs and building prompt context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .files import load_instructions


def _render_posthoc_analyses_context(workspace_dir: Path) -> str:
    """Render posthoc analyses context from the workspace copy when present."""
    posthoc_context_path = workspace_dir / "posthoc_analyses_context.json"
    if not posthoc_context_path.is_file():
        return "No posthoc analyses are available for this run."

    posthoc_analyses = json.loads(posthoc_context_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for analysis in posthoc_analyses:
        lines.append(f"## {analysis.get('title', analysis.get('analysis_id', 'unknown'))}")
        lines.append(f"- **ID**: `{analysis.get('analysis_id', '')}`")
        lines.append(f"- **Family**: {analysis.get('family', '')}")
        summary = analysis.get("summary_text", "").strip()
        if summary:
            lines.append(f"\n### Summary\n{summary}")
        rubric = analysis.get("rubric", {})
        if rubric:
            lines.append("\n### Interestingness Rubric (self-assessed, treat as noisy context)")
            for key, value in rubric.items():
                lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def _render_draft_reference_gaps_context(workspace_dir: Path) -> str:
    """Render post-draft reference gap context from workspace JSON when present."""
    gaps_path = workspace_dir / "draft_reference_gaps.json"
    if not gaps_path.is_file():
        return "No post-draft reference gaps have been identified for this run."

    try:
        payload = json.loads(gaps_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "Draft reference gap artifact is unreadable; ignore it unless the file is repaired."
    if not isinstance(payload, dict):
        return "Draft reference gap artifact is unreadable; ignore it unless the file is repaired."

    lines = [f"Status: `{payload.get('status', 'unknown')}`"]
    summary = str(payload.get("summary", "")).strip()
    if summary:
        lines.append(f"Summary: {summary}")
    missing_literatures = payload.get("missing_literatures", [])
    if isinstance(missing_literatures, list) and missing_literatures:
        lines.append("Missing literatures:")
        lines.extend(f"- {topic}" for topic in missing_literatures)
    rewrite_targets = payload.get("rewrite_targets", [])
    if isinstance(rewrite_targets, list) and rewrite_targets:
        lines.append("Rewrite targets:")
        for row in rewrite_targets:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- "
                + " | ".join(
                    value
                    for value in (
                        str(row.get("target_id", "")).strip(),
                        str(row.get("section", "")).strip(),
                        str(row.get("priority", "")).strip(),
                        str(row.get("citation_intent", "")).strip(),
                    )
                    if value
                )
            )
            reason = str(row.get("reason", "")).strip()
            if reason:
                lines.append(f"  reason: {reason}")
            anchor_text = str(row.get("anchor_text", "")).strip()
            if anchor_text:
                lines.append(f"  anchor: {anchor_text}")
            search_topics = row.get("search_topics", [])
            if isinstance(search_topics, list) and search_topics:
                lines.append("  search_topics: " + ", ".join(str(topic).strip() for topic in search_topics))
    return "\n".join(lines)


def paper_workspace_context_replacements(workspace_dir: Path) -> dict[str, str]:
    """Build prompt replacements derived from optional paper-workspace artifacts."""
    workspace_dir = workspace_dir.expanduser()
    aggregated_dir = workspace_dir / "aggregated"
    results_readme_path = workspace_dir / "results_readme.md"
    contributions_path = workspace_dir / "contributions.md"
    return {
        "POSTHOC_ANALYSES_CONTEXT": _render_posthoc_analyses_context(workspace_dir),
        "POSTHOC_RANKING_PATH": "./posthoc_ranking.json",
        "DRAFT_REFERENCE_GAPS_PATH": "./draft_reference_gaps.md",
        "DRAFT_REFERENCE_GAPS_JSON_PATH": "./draft_reference_gaps.json",
        "DRAFT_REFERENCE_GAPS_CONTEXT": _render_draft_reference_gaps_context(workspace_dir),
        "POSTDRAFT_REFERENCE_QUERY_PLAN_PATH": "./postdraft_reference_query_plan.md",
        "POSTDRAFT_REFERENCE_QUERY_PLAN_JSON_PATH": "./postdraft_reference_query_plan.json",
        "CONTRIBUTIONS_FILE_PATH": "./contributions.md",
        "CONTRIBUTIONS_CONTEXT": contributions_context_text(
            "./contributions.md" if contributions_path.is_file() else None
        ),
        "PAPER_AGGREGATED_DIR": "./aggregated"
        if aggregated_dir.exists() or aggregated_dir.is_symlink()
        else "",
        "RESULTS_README": (
            results_readme_path.read_text(encoding="utf-8").strip()
            if results_readme_path.is_file()
            else "(No campaign README available — rely on paper_data.json for variant and task metadata.)"
        ),
    }


def venue_context_text(venue_file_path: str | Path | None) -> str:
    """Return prompt text describing venue-brief availability."""
    if venue_file_path is None:
        return (
            "No venue-specific brief is provided for this run. "
            "Rely on the selected paper format, grounded artifacts, and references only."
        )

    venue_path = venue_file_path if isinstance(venue_file_path, str) else venue_file_path.as_posix()
    return (
        "A venue-specific brief is provided for this run.\n"
        f"Read `{venue_path}` before planning or drafting.\n"
        "Treat venue hard constraints as requirements and softer framing guidance as preferences "
        "when they remain consistent with grounded evidence.\n"
        "Do not invent claims, experiments, or positioning just to satisfy the venue brief."
    )


def contributions_context_text(contributions_file_path: str | Path | None) -> str:
    """Return prompt text describing optional contributions-scaffold availability."""
    if contributions_file_path is None:
        return (
            "No contributions scaffold is provided for this run. "
            "Infer the paper's claimed contributions directly from grounded artifacts."
        )

    contributions_path = (
        contributions_file_path
        if isinstance(contributions_file_path, str)
        else contributions_file_path.as_posix()
    )
    return (
        "A contributions scaffold is provided for this run.\n"
        f"Read `{contributions_path}` before drafting or repairing the paper.\n"
        "Use it as a base for contribution framing, structure, or wording only when it remains "
        "consistent with grounded artifacts.\n"
        "Do not treat it as quantitative grounding, citation grounding, or permission to claim "
        "results not supported by `paper_data.json` and the references artifacts."
    )


def author_block_context_text(author_block_path: str | Path | None) -> str:
    """Return prompt text describing canonical author-block availability."""
    if author_block_path is None:
        return (
            "No canonical author-block file is provided for this run. "
            "Preserve author metadata from the selected template and venue guidance."
        )

    author_path = (
        author_block_path if isinstance(author_block_path, str) else author_block_path.as_posix()
    )
    return (
        "A canonical author-block file is provided for this run.\n"
        f"Read `{author_path}` before drafting or repairing `paper.tex`.\n"
        "Treat its contents as a hard contract for the `\\author{...}` block.\n"
        "Keep `\\input{paper_authors.tex}` inside `\\author{...}` when possible; "
        "otherwise copy the exact file contents without renaming, reordering, anonymizing, "
        "or replacing authors or affiliations."
    )


def stage_optional_rendered_doc(
    *,
    source_path: Path | None,
    workspace_dir: Path,
    workspace_name: str,
    task_dir: Path,
    results_dir: Path,
    extra_replacements: Mapping[str, str] | None = None,
) -> Path | None:
    """Synchronize an optional markdown/template file into the workspace."""
    workspace_dir = workspace_dir.expanduser()
    destination = workspace_dir / workspace_name
    if source_path is None:
        destination.unlink(missing_ok=True)
        return None

    resolved_source = source_path.expanduser().resolve()
    if not resolved_source.is_file():
        raise FileNotFoundError(f"Missing optional prompt context file: {resolved_source}")

    workspace_dir.mkdir(parents=True, exist_ok=True)
    rendered = load_instructions(
        resolved_source,
        task_dir,
        results_dir,
        dict(extra_replacements or {}),
    )
    destination.write_text(rendered, encoding="utf-8")
    return destination
