"""Grounded framing for downstream studies without an investigation winner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.downstream_selection import load_downstream_selection


def merge_study_context(paper_data: dict[str, Any], investigation_dir: Path) -> None:
    """Attach only host-verified selection facts, independently of final evidence."""
    selection = load_downstream_selection(investigation_dir)
    if selection is None:
        return
    context = {
        key: selection[key]
        for key in (
            "selection_kind", "investigation_outcome", "combined_score", "baseline_score",
            "source_attempt_id", "source_program_dir", "program_dir", "entrypoint_path",
            "program_tree_sha256", "source_tree_sha256", "baseline_program_dir",
            "baseline_tree_sha256", "attempt_counts",
        )
    }
    baseline = context["baseline_score"] if context["baseline_program_dir"] is not None else None
    context["baseline_score"] = baseline
    context["search_score_delta"] = (
        context["combined_score"] - baseline if baseline is not None else None
    )
    context["final_evaluation"] = None
    context["limitations"] = [
        "Investigation found no qualifying winner; selection used search evidence only.",
        "Unscored attempts are missing performance evidence, not measured losses.",
        "Search scores and sealed final metrics belong to different evaluation scopes.",
        "An unqualified final outcome may reflect validity or other criteria; do not infer "
        "that failure means only an inferior score. Interpret metrics within those limitations.",
    ]
    if baseline is None:
        context["limitations"].append(
            "A valid scored baseline is unavailable; no baseline performance comparison is supported."
        )
    paper_data["downstream_study"] = context
    grounding = paper_data.setdefault("grounding_sources", {})
    facts = {
        "selection_kind": context["selection_kind"],
        "investigation_outcome": context["investigation_outcome"],
        "search_score": context["combined_score"],
        "baseline_score": baseline,
        "search_score_delta": context["search_score_delta"],
        **{f"attempts.{name}": value for name, value in context["attempt_counts"].items()},
    }
    for name, value in facts.items():
        if value is None:
            continue
        source_id = f"study.{name}"
        grounding[source_id] = {
            "id": source_id,
            "type": "study_context",
            "value": value,
            "evaluation_scope": "search",
            "program_tree_sha256": context["program_tree_sha256"],
        }


def merge_study_final_outcome(paper_data: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Preserve qualification status without treating it as a validity diagnosis."""
    context = paper_data.get("downstream_study")
    if not isinstance(context, dict):
        return
    outcome = {
        "status": manifest["status"],
        "correct": manifest["correct"],
        "outcome": manifest.get("outcome", "qualified"),
    }
    context["final_evaluation"] = outcome
    for name, value in outcome.items():
        source_id = f"study.final.{name}"
        paper_data["grounding_sources"][source_id] = {
            "id": source_id, "type": "study_context", "value": value,
            "evaluation_scope": "sealed_final",
            "program_tree_sha256": manifest["program_tree_sha256"],
            "results_tree_sha256": manifest["results_tree_sha256"],
        }
