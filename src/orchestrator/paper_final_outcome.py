"""Publication policy and immutable grounding for final qualification outcomes."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.pipeline.safe_fs import open_directory
from src.investigation.pinned_files import read_regular_at


class FinalOutcomePolicy(StrEnum):
    QUALIFIED_ONLY = "qualified_only"
    ALLOW_NOT_QUALIFIED = "allow_not_qualified"


def final_outcome_summary(paper_data: dict[str, Any]) -> dict[str, Any] | None:
    final = paper_data.get("final_evaluation")
    if not isinstance(final, dict):
        return None
    return {key: final[key] for key in ("status", "correct", "outcome", "limitations")
            if key in final}


def ground_final_outcome(paper_data: dict[str, Any]) -> None:
    """Expose verified qualification facts independently of scientific metrics."""
    final = paper_data["final_evaluation"]
    if final["correct"] is not True:
        final["limitations"] = [
            "Final evaluation completed but did not qualify; search success does not "
            "establish final qualification.",
            "Non-qualification may reflect performance, uncertainty, validity, adaptivity, "
            "or other criteria; report only causes supported by evaluator evidence.",
            "Retained metrics do not independently establish claims whose qualification "
            "or validity checks failed. Missing gate diagnostics remain unknown.",
        ]
    grounding = paper_data.setdefault("grounding_sources", {})
    for name in ("status", "correct", "outcome"):
        source_id = f"final.qualification.{name}"
        grounding[source_id] = {
            "id": source_id, "type": "final_qualification", "value": final[name],
            "evaluation_scope": "sealed_final",
            "manifest_path": final["manifest_path"],
            "program_tree_sha256": final["program_tree_sha256"],
            "results_tree_sha256": final["results_tree_sha256"],
        }


def verify_paper_final_outcome(
    output_dir: Path,
    *,
    final_evaluation_dir: Path,
    investigation_results_dir: Path,
    task_dir: Path | None = None,
    final_outcome_policy: FinalOutcomePolicy = FinalOutcomePolicy.QUALIFIED_ONLY,
) -> None:
    """Recheck opted-in evidence against host sources before each manuscript audit."""
    from src.orchestrator.paper_data import _merge_final_evaluation
    from src.orchestrator.paper_study_context import merge_study_context

    expected: dict[str, Any] = {}
    merge_study_context(expected, investigation_results_dir)
    _merge_final_evaluation(
        expected, final_evaluation_dir, task_dir=task_dir,
        investigation_results_dir=investigation_results_dir,
        final_outcome_policy=FinalOutcomePolicy(final_outcome_policy),
    )
    if "final_evaluation" not in expected:
        raise ValueError("Required paper final evidence is missing")
    actual = _read_paper_data(output_dir)
    if not isinstance(actual, dict):
        raise ValueError("Paper final evidence must contain an object")
    actual_final = actual.get("final_evaluation")
    legacy_study = (
        final_outcome_policy == FinalOutcomePolicy.QUALIFIED_ONLY
        and "downstream_study" in expected
        and isinstance(actual_final, dict)
        and "limitations" not in actual_final
    )
    expected_final = dict(expected["final_evaluation"])
    if legacy_study:
        expected_final.pop("limitations", None)
    if actual_final != expected_final:
        raise ValueError("Paper final evidence differs from verified final evaluation")
    actual_grounding = actual.get("grounding_sources")
    if not isinstance(actual_grounding, dict):
        raise ValueError("Paper final evidence grounding is missing")
    final_grounding = {
        key: value for key, value in actual_grounding.items()
        if key.startswith(("final.", "study.final."))
    }
    expected_grounding = {
        key: value for key, value in expected["grounding_sources"].items()
        if key.startswith(("final.", "study.final."))
    }
    # Legacy papers predate the observation locator; their complete embedded
    # evidence is still checked above, and added authoritative IDs stay forbidden.
    if not any(key.startswith("final.evidence.") for key in final_grounding):
        expected_grounding.pop("final.evidence.episodes", None)
    # Existing qualified and default no-winner papers predate these extra facts.
    # Opted-in negative winner papers must retain the complete new grounding.
    if (expected_final["correct"] is True or legacy_study) and not any(
        key.startswith("final.qualification.") for key in final_grounding
    ):
        expected_grounding = {
            key: value for key, value in expected_grounding.items()
            if not key.startswith("final.qualification.")
        }
    if final_grounding != expected_grounding:
        raise ValueError("Paper final evidence grounding differs from verified final evaluation")


def _read_paper_data(output_dir: Path) -> Any:
    with open_directory(output_dir, role="paper workspace") as output_fd:
        return json.loads(read_regular_at(output_fd, "paper_data.json",
                                          max_bytes=32 * 1024 * 1024))


def verify_resumed_final_outcome(
    output_dir: Path,
    *,
    source_results_dir: Path,
    investigation_results_dir: Path,
    final_evaluation_dir: Path | None = None,
    task_dir: Path | None = None,
    final_outcome_policy: FinalOutcomePolicy = FinalOutcomePolicy.QUALIFIED_ONLY,
) -> None:
    """Apply current policy before fast-finalizing an already audited manuscript."""
    try:
        paper_data = _read_paper_data(output_dir)
    except FileNotFoundError:
        return  # Incomplete workspaces cannot fast-finalize; preparation follows.
    final_dir = final_evaluation_dir or source_results_dir / "final_evaluation"
    has_final = isinstance(paper_data, dict) and paper_data.get("final_evaluation") is not None
    if final_evaluation_dir is not None or final_dir.exists() or has_final:
        verify_paper_final_outcome(
            output_dir, final_evaluation_dir=final_dir,
            investigation_results_dir=investigation_results_dir, task_dir=task_dir,
            final_outcome_policy=final_outcome_policy,
        )
