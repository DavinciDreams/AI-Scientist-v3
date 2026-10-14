"""Paper-review packet building and review artifact validation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from src.orchestrator.paper_supplement import split_manuscript_and_supplement
from src.orchestrator.paper_final_outcome import final_outcome_summary

_MAX_TEX_CHARS = 30000
_MAX_JSON_CHARS = 12000

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ReviewIssue(BaseModel):
    """Single actionable review issue."""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyText
    section: NonEmptyText
    issue: NonEmptyText
    evidence: list[NonEmptyText] = Field(default_factory=list)
    requested_change: NonEmptyText


class ReviewScore(BaseModel):
    """Reviewer score payload."""

    model_config = ConfigDict(extra="forbid")

    value: int = Field(ge=1, le=10)
    label: NonEmptyText


class ReviewPayload(BaseModel):
    """Structured paper review contract."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pass", "revise"]
    summary: NonEmptyText
    strengths: list[NonEmptyText] = Field(default_factory=list)
    major_comments: list[ReviewIssue] = Field(default_factory=list)
    minor_comments: list[ReviewIssue] = Field(default_factory=list)
    questions: list[NonEmptyText] = Field(default_factory=list)
    score: ReviewScore


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"_json_error": f"Invalid JSON in {path.name}: {exc}"}


def _json_error_summary(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict) and isinstance(payload.get("_json_error"), str):
        return {"available": False, "error": payload["_json_error"]}
    return None


def _truncate(text: str, *, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return f"{stripped[:max_chars].rstrip()}\n\n[truncated after {max_chars} chars]"


def _json_excerpt(payload: Any, *, max_chars: int = _MAX_JSON_CHARS) -> str:
    return _truncate(json.dumps(payload, indent=2, sort_keys=True), max_chars=max_chars)


def _paper_data_summary(payload: Any) -> dict[str, Any]:
    error_summary = _json_error_summary(payload)
    if error_summary is not None:
        return error_summary
    if not isinstance(payload, dict):
        return {"available": False}

    totals = payload.get("totals")
    idea_titles = payload.get("idea_titles")
    grounding_sources = payload.get("grounding_sources")
    ablations = payload.get("ablation_studies")
    if isinstance(grounding_sources, dict):
        grounding_source_count = len(grounding_sources)
    elif isinstance(grounding_sources, list):
        grounding_source_count = len(grounding_sources)
    else:
        grounding_source_count = 0

    summary = {
        "available": True,
        "paper_profile": payload.get("paper_profile", "default"),
        "downstream_study": payload.get("downstream_study"),
        "final_evaluation": final_outcome_summary(payload),
        "totals": totals if isinstance(totals, dict) else {},
        "idea_titles_preview": idea_titles[:10] if isinstance(idea_titles, list) else [],
        "grounding_source_count": grounding_source_count,
        "ablation_count": len(ablations) if isinstance(ablations, list) else 0,
    }
    return summary


def _references_summary(payload: Any) -> dict[str, Any]:
    error_summary = _json_error_summary(payload)
    if error_summary is not None:
        return error_summary
    if not isinstance(payload, dict):
        return {"available": False}

    references = payload.get("references")
    preview: list[dict[str, Any]] = []
    if isinstance(references, list):
        for row in references[:10]:
            if not isinstance(row, dict):
                continue
            preview.append(
                {
                    "id": row.get("id"),
                    "citation_key": row.get("citation_key"),
                    "title": row.get("title"),
                }
            )
    return {
        "available": True,
        "count": len(references) if isinstance(references, list) else 0,
        "queries": payload.get("queries") if isinstance(payload.get("queries"), list) else [],
        "preview": preview,
    }


def _validation_summary(payload: Any) -> dict[str, Any]:
    error_summary = _json_error_summary(payload)
    if error_summary is not None:
        return error_summary
    if not isinstance(payload, dict):
        return {"available": False}

    feedback_payload = payload.get("feedback")
    feedback: dict[str, Any] = feedback_payload if isinstance(feedback_payload, dict) else {}
    return {
        "available": True,
        "paper_profile": payload.get("paper_profile"),
        "pdf_path": payload.get("pdf_path"),
        "hard_error_count": len(feedback.get("errors", []))
        if isinstance(feedback.get("errors"), list)
        else 0,
        "figure_warning_count": len(feedback.get("figure_validation_warnings", []))
        if isinstance(feedback.get("figure_validation_warnings"), list)
        else 0,
        "figure_error_count": len(feedback.get("figure_validation_errors", []))
        if isinstance(feedback.get("figure_validation_errors"), list)
        else 0,
    }


def build_review_packet(paper_dir: Path, *, output_path: Path | None = None) -> Path:
    """Build a compact review packet from paper-stage artifacts."""
    paper_dir = paper_dir.expanduser().resolve()
    output_path = (output_path or (paper_dir / "paper_review_packet.md")).expanduser().resolve()

    paper_tex = paper_dir / "paper.tex"
    if not paper_tex.is_file():
        raise FileNotFoundError(f"Missing required manuscript artifact: {paper_tex}")

    packet_lines = [
        "# Paper review packet",
        "",
        f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- paper_dir: `{paper_dir}`",
        "",
        "Use this packet as the default review entrypoint. Read the raw artifacts only when this packet is insufficient.",
        "",
        "## Expected review focus",
        "",
        "- unsupported quantitative claims or weak claim-grounding links",
        "- inconsistencies between paper.tex, claims_grounding.json, and validation artifacts",
        "- unclear scientific framing, overclaiming, or missing caveats",
        "- figure or results descriptions that are hard to interpret",
        "- Supplementary Information completeness, relevance, and grounding",
        "",
        "## Artifact checklist",
        "",
    ]
    for name in [
        "paper.tex",
        "claims_grounding.json",
        "paper_data.json",
        "references.bib",
        "references_provenance.json",
        "validation_report.json",
        "paper_strategy.json",
        "figure_manifest.json",
        "paper.pdf",
    ]:
        path = paper_dir / name
        state = "present" if path.exists() else "missing"
        packet_lines.append(f"- `{name}` — `{state}`")

    paper_data = _read_json(paper_dir / "paper_data.json")
    claims_grounding = _read_json(paper_dir / "claims_grounding.json")
    references = _read_json(paper_dir / "references_provenance.json")
    validation_report = _read_json(paper_dir / "validation_report.json")
    strategy_payload = _read_json(paper_dir / "paper_strategy.json")
    figure_manifest = _read_json(paper_dir / "figure_manifest.json")

    packet_lines.extend(
        [
            "",
            "## paper_data summary",
            "",
            "```json",
            _json_excerpt(_paper_data_summary(paper_data)),
            "```",
            "",
            "## claims_grounding excerpt",
            "",
            "```json",
            _json_excerpt(claims_grounding),
            "```",
            "",
            "## references provenance summary",
            "",
            "```json",
            _json_excerpt(_references_summary(references)),
            "```",
            "",
            "## validation report summary",
            "",
            "```json",
            _json_excerpt(_validation_summary(validation_report)),
            "```",
        ]
    )

    if strategy_payload is not None:
        packet_lines.extend(
            [
                "",
                "## paper_strategy excerpt",
                "",
                "```json",
                _json_excerpt(strategy_payload),
                "```",
            ]
        )
    if figure_manifest is not None:
        packet_lines.extend(
            [
                "",
                "## figure_manifest excerpt",
                "",
                "```json",
                _json_excerpt(figure_manifest),
                "```",
            ]
        )

    paper_tex_text = _read_text(paper_tex) or ""
    manuscript_text, supplement_excerpt = split_manuscript_and_supplement(paper_tex_text)
    packet_lines.extend(
        [
            "",
            "## manuscript source excerpt",
            "",
            "```tex",
            _truncate(manuscript_text, max_chars=_MAX_TEX_CHARS),
            "```",
            "",
        ]
    )
    if supplement_excerpt is not None:
        packet_lines.extend(
            [
                "## supplementary information excerpt",
                "",
                "```tex",
                _truncate(supplement_excerpt, max_chars=_MAX_TEX_CHARS),
                "```",
                "",
            ]
        )
    output_path.write_text("\n".join(packet_lines), encoding="utf-8")
    return output_path


def validate_review_payload(review_json_path: Path) -> ReviewPayload:
    """Load and validate the structured review JSON."""
    review_json_path = review_json_path.expanduser().resolve()
    try:
        payload = json.loads(review_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {review_json_path}: {exc}") from exc
    review = ReviewPayload.model_validate(payload)
    if review.status == "pass" and review.major_comments:
        raise ValueError("review status 'pass' cannot include major_comments")
    if review.status == "revise" and not review.major_comments:
        raise ValueError("review status 'revise' must include at least one major_comment")
    for issue in review.major_comments:
        if not issue.evidence:
            raise ValueError(f"major comment {issue.id!r} must include evidence")
    seen_issue_ids: set[str] = set()
    for issue in [*review.major_comments, *review.minor_comments]:
        if issue.id in seen_issue_ids:
            raise ValueError(f"duplicate review issue id: {issue.id}")
        seen_issue_ids.add(issue.id)
    return review


def render_review_markdown(review: ReviewPayload) -> str:
    """Render a human-readable markdown review."""
    lines = [
        "# Paper review",
        "",
        f"- status: `{review.status}`",
        f"- score: `{review.score.value}: {review.score.label}`",
        "",
        "## Summary",
        review.summary,
        "",
        "## Strengths",
    ]
    if review.strengths:
        lines.extend(f"- {item}" for item in review.strengths)
    else:
        lines.append("- None recorded.")
    lines.append("")

    def _append_issue_block(title: str, issues: list[ReviewIssue]) -> None:
        lines.append(f"## {title}")
        if not issues:
            lines.append("- None.")
            lines.append("")
            return
        for idx, issue in enumerate(issues, start=1):
            lines.append(f"### {idx}. {issue.id} — {issue.section}")
            lines.append(f"- issue: {issue.issue}")
            lines.append(f"- requested_change: {issue.requested_change}")
            if issue.evidence:
                lines.append("- evidence:")
                lines.extend(f"  - {item}" for item in issue.evidence)
            lines.append("")

    _append_issue_block("Major comments", review.major_comments)
    _append_issue_block("Minor comments", review.minor_comments)
    lines.append("## Questions")
    if review.questions:
        lines.extend(f"- {item}" for item in review.questions)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_review_feedback(
    review: ReviewPayload,
    *,
    review_json_path: Path,
    review_markdown_path: Path,
    feedback_json_path: Path,
    feedback_md_path: Path,
) -> dict[str, Any]:
    """Write compact review feedback artifacts for the next repair pass."""
    major_summaries = [
        {
            "id": issue.id,
            "section": issue.section,
            "issue": issue.issue,
            "requested_change": issue.requested_change,
            "evidence": issue.evidence,
        }
        for issue in review.major_comments
    ]
    minor_summaries = [
        {
            "id": issue.id,
            "section": issue.section,
            "issue": issue.issue,
            "requested_change": issue.requested_change,
            "evidence": issue.evidence,
        }
        for issue in review.minor_comments
    ]
    feedback: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": len(review.major_comments) == 0,
        "status": review.status,
        "summary": review.summary,
        "major_comment_count": len(review.major_comments),
        "minor_comment_count": len(review.minor_comments),
        "review_json_path": str(review_json_path.resolve()),
        "review_markdown_path": str(review_markdown_path.resolve()),
        "major_comments": major_summaries,
        "minor_comments": minor_summaries,
    }
    feedback_json_path.write_text(json.dumps(feedback, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Paper review feedback",
        "",
        f"- generated_at: `{feedback['generated_at']}`",
        f"- ok: `{feedback['ok']}`",
        f"- status: `{feedback['status']}`",
        f"- major_comment_count: `{feedback['major_comment_count']}`",
        f"- minor_comment_count: `{feedback['minor_comment_count']}`",
        f"- review_json_path: `{review_json_path.resolve()}`",
        f"- review_markdown_path: `{review_markdown_path.resolve()}`",
        "",
        "## Summary",
        review.summary,
        "",
        "## Major comments",
    ]
    if major_summaries:
        for item in major_summaries:
            lines.append(f"- `{item['id']}` ({item['section']}): {item['issue']}")
            lines.append(f"  - requested_change: {item['requested_change']}")
            if item["evidence"]:
                lines.append("  - evidence:")
                lines.extend(f"    - {entry}" for entry in item["evidence"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Minor comments"])
    if minor_summaries:
        for item in minor_summaries:
            lines.append(f"- `{item['id']}` ({item['section']}): {item['issue']}")
            lines.append(f"  - requested_change: {item['requested_change']}")
            if item["evidence"]:
                lines.append("  - evidence:")
                lines.extend(f"    - {entry}" for entry in item["evidence"])
    else:
        lines.append("- None.")
    lines.append("")
    feedback_md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return feedback


def finalize_review(
    *,
    review_json_path: Path,
    markdown_output_path: Path,
    feedback_json_path: Path,
    feedback_md_path: Path,
) -> bool:
    """Validate the review JSON and write derived markdown/feedback artifacts.

    Returns True when major comments are present.
    """
    review = validate_review_payload(review_json_path)
    markdown_output_path.write_text(render_review_markdown(review), encoding="utf-8")
    write_review_feedback(
        review,
        review_json_path=review_json_path,
        review_markdown_path=markdown_output_path,
        feedback_json_path=feedback_json_path,
        feedback_md_path=feedback_md_path,
    )
    return len(review.major_comments) > 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper review helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    packet_parser = subparsers.add_parser("build-packet", help="Build paper_review_packet.md")
    packet_parser.add_argument("--paper-dir", required=True, help="Paper workspace directory")
    packet_parser.add_argument("--output", required=True, help="Output packet path")

    finalize_parser = subparsers.add_parser(
        "finalize-review",
        help="Validate paper_review.json and derive markdown/feedback artifacts",
    )
    finalize_parser.add_argument("--input", required=True, help="Path to paper_review.json")
    finalize_parser.add_argument("--markdown-output", required=True, help="Path to paper_review.md")
    finalize_parser.add_argument(
        "--feedback-json", required=True, help="Path to paper_review_feedback.json"
    )
    finalize_parser.add_argument(
        "--feedback-md", required=True, help="Path to paper_review_feedback.md"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-packet":
            packet_path = build_review_packet(Path(args.paper_dir), output_path=Path(args.output))
            print(packet_path)
            return 0
        if args.command == "finalize-review":
            finalize_review(
                review_json_path=Path(args.input),
                markdown_output_path=Path(args.markdown_output),
                feedback_json_path=Path(args.feedback_json),
                feedback_md_path=Path(args.feedback_md),
            )
            return 0
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
