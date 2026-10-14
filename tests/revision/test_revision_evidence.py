"""Revision evidence validation: inventories, audits, and completion-manifest metadata."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest

from src.orchestrator.paper_evidence_audit import audit_input_sha256
from src.orchestrator.paper_manifest import latest_manifest_lock, write_manifest_atomically
from src.pipeline.revision_evidence import (
    validate_published_revision_evidence,
    validate_revision_evidence,
)
from tests.revision_test_support import refresh_evidence_audit


def _paper_data() -> dict[str, Any]:
    return {
        "generated_at": "2026-01-01T00:00:00Z",
        "paper_profile": "default",
        "presentation_contract": {
            "variant_unit": "idea",
            "run_unit": "program",
            "internal_fields": ["correct", "eligible"],
        },
        "results_dir": "/source/run/investigation",
        "all_programs": [
            {
                "attempt_id": "idea_01_attempt_01",
                "idea_id": "idea_01",
                "program_id": "program_1",
                "eval_index": 1,
                "combined_score": 2.5,
                "correct": True,
                "program_dir": "/source/run/program_1",
                "completed_at": "2026-01-01T00:00:00Z",
            }
        ],
        "best_correct": {
            "program_id": "program_1",
            "combined_score": 2.5,
            "program_dir": "/source/run/program_1",
        },
        "best_any": {
            "program_id": "program_1",
            "combined_score": 2.5,
            "program_dir": "/source/run/program_1",
        },
        "totals": {"ideas": 1, "attempts": 1, "evaluated_programs": 1, "correct_programs": 1},
        "top_programs": [
            {
                "rank": 1,
                "program_id": "program_1",
                "combined_score": 2.5,
                "program_dir": "/source/run/program_1",
            }
        ],
        "improvement_trace": [{"eval_index": 1, "program_id": "program_1", "combined_score": 2.5}],
        "idea_titles": ["Primary method"],
        "task_names": ["primary", "held_out"],
        "cross_task_results": {
            "held_out": {"control": {"variant_id": "control", "mean": 2.4, "n": 3}}
        },
        "grounding_sources": {
            "result.top_program_01": {"value": 2.5},
            "ablation.study_main.variant_control.score": {"value": 2.5},
        },
        "ablation_studies": [
            {
                "study_id": "main",
                "variants": [
                    {"variant_id": "control", "metric_summaries": {"score": {"mean": 2.5}}},
                    {"variant_id": "without_search"},
                ],
            }
        ],
        "posthoc_analyses": [
            {"analysis_id": "precision_check", "summary_text": "Precision is adequate."}
        ],
        "final_evaluation": {
            "metrics": {"score": 2.5},
            "program_tree_sha256": "sha256:program",
            "task_tree_sha256": "sha256:task",
        },
        "evidence_sources": {
            "investigation": {"status": "included", "path": "/source/run/investigation"},
            "ablations": {"status": "included", "path": "/source/run/ablations"},
        },
    }


def _write_source(path: Path, data: dict[str, Any] | None = None) -> None:
    path.mkdir(parents=True)
    (path / "paper_data.json").write_text(json.dumps(data or _paper_data()), encoding="utf-8")


def _write_candidate(path: Path, data: dict[str, Any] | None = None) -> None:
    """Complete, audited candidate workspace (also imported by other test modules)."""
    path.mkdir(parents=True)
    (path / "paper.tex").write_text("\\title{Revised}\n", encoding="utf-8")
    (path / "claims_grounding.json").write_text("{}\n", encoding="utf-8")
    (path / "references.bib").write_text("% refs\n", encoding="utf-8")
    (path / "references_provenance.json").write_text("[]\n", encoding="utf-8")
    (path / "paper_data.json").write_text(json.dumps(data or _paper_data()), encoding="utf-8")
    (path / "method_scope_audit.json").write_text(
        json.dumps(
            {
                "manuscript_focus": {
                    "shared_method_summary": "Shared method.",
                    "scope_strategy": "Keep each claim scoped.",
                },
                "shared_method_elements": ["Shared search procedure."],
                "scoped_method_elements": [],
                "claims_requiring_scoping": [],
            }
        ),
        encoding="utf-8",
    )
    (path / "method_scope_audit.md").write_text("# Method scope audit\n", encoding="utf-8")
    audit = {
        "verdict": "pass",
        "summary": "All claims are supported.",
        "finding_count": 0,
        "findings": [],
        "input_sha256": audit_input_sha256(path),
    }
    (path / "paper_evidence_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (path / "validation_report.json").write_text(
        json.dumps({"ok": True, "agentic_evidence_audit": {"checked": True, **audit}}),
        encoding="utf-8",
    )


def _pair(
    tmp_path: Path,
    source_data: dict[str, Any] | None = None,
    candidate_data: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    source = tmp_path / "paper"
    candidate = tmp_path / "attempt_01"
    _write_source(source, source_data)
    _write_candidate(candidate, candidate_data)
    return source, candidate


def _edited_pair(tmp_path: Path, edit_source, edit_candidate) -> tuple[Path, Path]:
    source_data = _paper_data()
    edit_source(source_data)
    candidate_data = deepcopy(source_data)
    edit_candidate(candidate_data)
    return _pair(tmp_path, source_data, candidate_data)


def _resolved_report(candidate: Path, resolved_path: str) -> None:
    evidence = candidate / "scientific-evidence"
    evidence.mkdir()
    (evidence / "verification-report.json").write_text(
        json.dumps({"results": [{"resolved_path": resolved_path}]})
    )


def test_revision_evidence_accepts_complete_candidate_additions_and_new_evidence(
    tmp_path: Path,
) -> None:
    # a complete, current candidate is accepted
    validate_revision_evidence(*_pair(tmp_path / "complete"))

    # additions and location/timestamp changes are allowed
    candidate_data = deepcopy(_paper_data())
    candidate_data["all_programs"][0]["program_dir"] = "/isolated/copy/program_1"
    candidate_data["all_programs"][0]["completed_at"] = "2026-02-02T00:00:00Z"
    candidate_data["generated_at"] = "2026-02-02T00:00:00Z"
    candidate_data["evidence_sources"]["ablations"]["path"] = "/isolated/copy/ablations"
    candidate_data["all_programs"][0]["new_metadata"] = "allowed"
    candidate_data["grounding_sources"]["new.claim"] = {"value": 3.0}

    validate_revision_evidence(*_pair(tmp_path / "additions", None, candidate_data))

    # a candidate may add a final evaluation the source lacked
    source_data = deepcopy(_paper_data())
    source_data.pop("final_evaluation")

    validate_revision_evidence(*_pair(tmp_path / "new_final", source_data))

    # overlapping records are matched without a false loss
    source, candidate = _edited_pair(
        tmp_path / "overlap",
        lambda data: data.update(
            posthoc_analyses=[{"analysis_id": "precision"}, {"summary_text": "supported"}]
        ),
        lambda data: data.update(
            posthoc_analyses=[
                {"analysis_id": "precision", "summary_text": "supported"},
                {"analysis_id": "precision"},
            ]
        ),
    )

    validate_revision_evidence(source, candidate)


_POSTHOC_EDITORIAL_CHANGES = [
    ("key_finding_for_paper", "Metric rankings agree.", "Arithmetic checks agree."),
    ("relevance_justification", "Duplicates the audit.", "Supports small score differences."),
    ("recommended_usage", "skip", "appendix"),
    ("relevance_to_thesis", "low", "medium"),
]


def test_revision_evidence_accepts_posthoc_editorial_changes(tmp_path: Path) -> None:
    for validator in (validate_revision_evidence, validate_published_revision_evidence):
        for field, original, revised in _POSTHOC_EDITORIAL_CHANGES:
            source, candidate = _edited_pair(
                tmp_path / validator.__name__ / field,
                lambda data: data["posthoc_analyses"][0].__setitem__(field, original),
                lambda data: data["posthoc_analyses"][0].__setitem__(field, revised),
            )

            validator(source, candidate)


def test_revision_evidence_preserves_editorial_names_outside_posthoc_entries(tmp_path: Path) -> None:
    for location in ("root", "raw_data", "nested_inventory", "other_inventory"):

        def target(data: dict[str, Any], _location: str = location) -> dict[str, Any]:
            if _location == "root":
                return data
            if _location == "raw_data":
                return data["posthoc_analyses"][0].setdefault("raw_data", {})
            if _location == "nested_inventory":
                return data["grounding_sources"].setdefault("posthoc_analyses", [{}])[0]
            return data["all_programs"][0]

        for field, original, revised in _POSTHOC_EDITORIAL_CHANGES:
            source, candidate = _edited_pair(
                tmp_path / location / field,
                lambda data: target(data).__setitem__(field, original),
                lambda data: target(data).__setitem__(field, revised),
            )

            with pytest.raises(ValueError, match="revision loses or changes"):
                validate_revision_evidence(source, candidate)


_POSTHOC_EVIDENCE_CHANGES = [
    ("raw_data", {"score": 2.5}, {"score": 0.0}),
    ("run_analysis_code", "print(2.5)", "print(0.0)"),
    ("source_sha256", "source-digest", "other-digest"),
    ("figure_paths", ["figures/audit.png"], []),
    ("rank", 1, 2),
]


def test_revision_evidence_rejects_posthoc_evidence_change_alongside_editorial_change(
    tmp_path: Path,
) -> None:
    # evidence fields may not change together with editorial fields
    for field, original, revised in _POSTHOC_EVIDENCE_CHANGES:
        source, candidate = _edited_pair(
            tmp_path / field,
            lambda data: data["posthoc_analyses"][0].update(recommended_usage="skip", **{field: original}),
            lambda data: data["posthoc_analyses"][0].update(
                recommended_usage="appendix", **{field: revised}
            ),
        )

        with pytest.raises(ValueError, match="posthoc inventory"):
            validate_revision_evidence(source, candidate)

    # arbitrary absolute-looking text is not treated as a location change
    source, candidate = _edited_pair(
        tmp_path / "absolute_text",
        lambda data: data["posthoc_analyses"][0].update(summary_text="/supported conclusion"),
        lambda data: data["posthoc_analyses"][0].update(summary_text="/changed conclusion"),
    )

    with pytest.raises(ValueError, match="posthoc inventory"):
        validate_revision_evidence(source, candidate)


TAMPER_CASES = [
    (
        "stale_agentic_audit",
        lambda candidate: (candidate / "paper.tex").write_text(
            "\\title{Changed after audit}\n", encoding="utf-8"
        ),
        "current passing agentic evidence audit",
    ),
    (
        "invalid_method_scope_audit",
        lambda candidate: (candidate / "method_scope_audit.md").write_text("", encoding="utf-8"),
        "method-scope audit",
    ),
]


def test_revision_evidence_requires_current_audit_and_rejects_tampered_workspace(
    tmp_path: Path,
) -> None:
    # a posthoc editorial change written after the audit requires a new audit
    source_data = _paper_data()
    source_data["posthoc_analyses"][0]["recommended_usage"] = "skip"
    source, candidate = _pair(tmp_path / "stale_editorial", source_data, source_data)
    candidate_data = deepcopy(source_data)
    candidate_data["posthoc_analyses"][0]["recommended_usage"] = "appendix"
    (candidate / "paper_data.json").write_text(json.dumps(candidate_data), encoding="utf-8")

    with pytest.raises(ValueError, match="current passing agentic evidence audit"):
        validate_revision_evidence(source, candidate)

    # tampered candidate workspaces are rejected
    for name, tamper, expected_error in TAMPER_CASES:
        source, candidate = _pair(tmp_path / name)
        tamper(candidate)

        with pytest.raises(ValueError, match=expected_error):
            validate_revision_evidence(source, candidate)


def _clear(field: str):
    empty: dict[str, Any] | list[Any] = {} if field in {"grounding_sources", "final_evaluation"} else []
    return lambda data: data.__setitem__(field, empty)


CORRUPTION_CASES: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    # Inventory loss.
    ("lose_programs", _clear("all_programs"), "program inventory"),
    ("lose_tasks", _clear("task_names"), "task inventory"),
    ("lose_grounding", _clear("grounding_sources"), "grounding inventory"),
    ("lose_ablations", _clear("ablation_studies"), "ablation inventory"),
    ("lose_posthoc", _clear("posthoc_analyses"), "posthoc inventory"),
    ("lose_final_evaluation", _clear("final_evaluation"), "final evaluation inventory"),
    # Partial semantic corruption inside inventories.
    (
        "program_score",
        lambda data: data["all_programs"][0].update(combined_score=0.0),
        "program inventory",
    ),
    (
        "cross_task_mean",
        lambda data: data["cross_task_results"]["held_out"]["control"].update(mean=0.0),
        "task inventory",
    ),
    (
        "grounding_value",
        lambda data: data["grounding_sources"]["result.top_program_01"].update(value=0.0),
        "grounding inventory",
    ),
    (
        "ablation_mean",
        lambda data: data["ablation_studies"][0]["variants"][0]["metric_summaries"]["score"].update(
            mean=0.0
        ),
        "ablation inventory",
    ),
    (
        "posthoc_summary",
        lambda data: data["posthoc_analyses"][0].update(summary_text=""),
        "posthoc inventory",
    ),
    # Corruption outside the legacy inventories.
    (
        "best_correct",
        lambda data: data["best_correct"].update(combined_score=0.0),
        "paper data evidence",
    ),
    (
        "top_programs",
        lambda data: data["top_programs"][0].update(combined_score=0.0),
        "paper data evidence",
    ),
    (
        "improvement_trace",
        lambda data: data["improvement_trace"][0].update(combined_score=0.0),
        "paper data evidence",
    ),
    ("totals", lambda data: data["totals"].update(evaluated_programs=0), "paper data evidence"),
    (
        "evidence_sources",
        lambda data: data["evidence_sources"]["ablations"].update(status="missing"),
        "paper data evidence",
    ),
    ("paper_profile", lambda data: data.update(paper_profile="other"), "paper data evidence"),
    (
        "presentation_contract",
        lambda data: data["presentation_contract"].update(run_unit="attempt"),
        "paper data evidence",
    ),
]


def test_revision_evidence_rejects_candidate_corruption(tmp_path: Path) -> None:
    for name, corrupt, expected_error in CORRUPTION_CASES:
        candidate_data = deepcopy(_paper_data())
        corrupt(candidate_data)
        source, candidate = _pair(tmp_path / name, None, candidate_data)

        with pytest.raises(ValueError, match=expected_error):
            validate_revision_evidence(source, candidate)


# --- completion manifest written after the audit ----------------------------------------


def test_revision_accepts_completion_manifest_written_after_audit(tmp_path: Path) -> None:
    for initial_manifest in (None, '{"status":"draft"}\n'):
        for resolved_path in (None, "."):
            case = f"{'draft' if initial_manifest else 'none'}-{'resolved' if resolved_path else 'plain'}"
            source, candidate = _pair(tmp_path / case)
            if resolved_path is not None:
                _resolved_report(candidate, resolved_path)
            manifest = candidate / "latest_manifest.json"
            if initial_manifest is not None:
                manifest.write_text(initial_manifest)
            refresh_evidence_audit(candidate)

            with latest_manifest_lock(candidate):
                write_manifest_atomically(manifest, {"status": "completed"})

            validate_revision_evidence(source, candidate)


CHANGED_PATHS = [
    "paper.tex",
    "claims_grounding.json",
    "evidence/latest_manifest.json",
    "ai_review.json",
    "revisions/snapshot/metrics.json",
]


def test_candidate_audit_still_rejects_changes_outside_completion_manifest(tmp_path: Path) -> None:
    # any audited file changed after the audit is rejected, with or without a resolved report
    for index, changed_path in enumerate(CHANGED_PATHS):
        for resolved_path in (None, "."):
            case = f"{index}-{'resolved' if resolved_path else 'plain'}"
            source, candidate = _pair(tmp_path / case)
            if resolved_path is not None:
                _resolved_report(candidate, resolved_path)
            changed = candidate / changed_path
            changed.parent.mkdir(parents=True, exist_ok=True)
            changed.write_text("{}\n")
            refresh_evidence_audit(candidate)

            (candidate / "latest_manifest.json").write_text('{"status":"completed"}\n')
            changed.write_text('{"changed":true}\n')

            with pytest.raises(ValueError, match="current passing agentic evidence audit"):
                validate_revision_evidence(source, candidate)

    # a nested publication lock inside a resolved workspace stays audited evidence
    source, candidate = _pair(tmp_path / "nested_lock")
    _resolved_report(candidate, ".")
    nested_lock = candidate / "evidence" / ".latest_manifest.lock"
    nested_lock.parent.mkdir()
    nested_lock.write_text("original evidence\n")
    refresh_evidence_audit(candidate)

    with latest_manifest_lock(candidate):
        write_manifest_atomically(candidate / "latest_manifest.json", {"status": "completed"})
    nested_lock.write_text("changed evidence\n")

    with pytest.raises(ValueError, match="current passing agentic evidence audit"):
        validate_revision_evidence(source, candidate)
