"""Paper review packets/payloads, validation feedback artifacts, scope audit and finalization."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.orchestrator import paper_finalize
from src.orchestrator import state as state_store
from src.orchestrator.paper_feedback import (
    write_stage_contract_feedback,
    write_stage_execution_feedback,
    write_validation_feedback,
)
from src.orchestrator.paper_finalize import finalize_paper_artifacts
from src.orchestrator.paper_review import (
    build_review_packet,
    finalize_review,
    validate_review_payload,
)
from src.orchestrator.paper_scope_audit import validate_method_scope_audit

PAPER_SH = Path("src/scripts/paper.sh")
PAPER_DATA = {
    "paper_profile": "default",
    "totals": {"evaluated_programs": 3},
    "idea_titles": ["idea a", "idea b"],
    "grounding_sources": {"s1": {"id": "s1", "type": "metric", "value": 1.0}},
}
CLAIMS = {"claims": [{"claim": "x", "source_ids": ["s1"], "citation_keys": []}]}
MAJOR = {
    "id": "claim-gap", "section": "Results", "issue": "A score claim is not grounded.",
    "evidence": ["paper.tex states score 5.1 but claims_grounding omits it"],
    "requested_change": "Ground or remove the claim.",
}
MINOR = {
    "id": "tighten-opening", "section": "Introduction",
    "issue": "The opening paragraph is denser than it needs to be.", "evidence": [],
    "requested_change": "Split the first paragraph for readability.",
}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, data: object) -> Path:
    return _write(path, json.dumps(data))


def _review(status: str = "revise", major=(MAJOR,), minor=(MINOR,), score=(5, "borderline_accept")) -> dict:
    return {
        "status": status, "summary": "Needs one grounded revision.",
        "strengths": ["Clear motivation."], "major_comments": list(major),
        "minor_comments": list(minor), "questions": ["Was this metric averaged across seeds?"],
        "score": {"value": score[0], "label": score[1]},
    }


def _packet(root: Path, tex: str) -> str:
    _write(root / "paper.tex", tex)
    return build_review_packet(root, output_path=root / "paper_review_packet.md").read_text(
        encoding="utf-8"
    )


# ── review packet ──


def test_build_review_packet_includes_expected_sections_and_survives_invalid_optional_json(
    tmp_path: Path,
) -> None:
    # includes expected sections
    root = tmp_path / "sections"
    _write_json(root / "claims_grounding.json", CLAIMS)
    _write_json(root / "paper_data.json", PAPER_DATA)
    text = _packet(root, "\\section{Introduction}\nHello world\n")
    assert "# Paper review packet" in text
    assert "## claims_grounding excerpt" in text
    assert "## manuscript source excerpt" in text
    assert "paper.tex" in text
    assert '"grounding_source_count": 1' in text
    # survives invalid optional json
    root = tmp_path / "invalid"
    _write(root / "claims_grounding.json", "{not valid json")
    text = _packet(root, "\\section{Introduction}\nHello world\n")
    assert "Invalid JSON in claims_grounding.json" in text


def test_build_review_packet_supplement_excerpt_handling(tmp_path: Path) -> None:
    # includes the supplement beyond the manuscript excerpt
    marker = "SI robustness result unique to the manuscript tail."
    text = _packet(
        tmp_path / "long",
        "A" * 31_000 + "\n\\appendix\n\\section{Supplementary Information}\n" + marker + "\n",
    )
    manuscript_excerpt, supplement_excerpt = text.split(
        "## supplementary information excerpt", maxsplit=1
    )
    assert marker not in manuscript_excerpt
    assert "\\section{Supplementary Information}" in supplement_excerpt
    assert marker in supplement_excerpt
    assert "Supplementary Information completeness, relevance, and grounding" in text
    # preserves supplement listing content
    marker = "implementation_setting = reproducible_value"
    text = _packet(
        tmp_path / "listing",
        "\\section{Introduction}\nMain text.\n\\appendix\n\\section{Supplementary Information}\n"
        f"\\subsection{{Implementation}}\n\\begin{{lstlisting}}\n{marker}\n\\end{{lstlisting}}\n",
    )
    assert marker in text
    # does not duplicate a short supplement
    marker = "SI marker present exactly once."
    text = _packet(
        tmp_path / "short",
        "\\section{Introduction}\nMain paper.\n\\appendix\n\\section{Supplementary Information}\n" + marker,
    )
    assert text.count(marker) == 1
    # keeps later appendix sections with the supplement
    marker = "Later appendix evidence remains reviewable."
    text = _packet(
        tmp_path / "later",
        "\\section{Introduction}\nMain paper.\n\\appendix\n\\section{Supplementary Information}\n"
        "SI content.\n\\section{Additional Proofs}\n" + marker + "\n\\end{document}\n",
    )
    _, supplement_excerpt = text.split("## supplementary information excerpt", maxsplit=1)
    assert marker in supplement_excerpt


# ── review payload ──


INCONSISTENT_REVIEW_PAYLOADS = [
    (
        "pass_with_major_comments",
        _review(status="pass", score=(7, "accept")), "cannot include major_comments",
    ),
    (
        "revise_without_major_comments",
        _review(major=(), score=(6, "weak_accept")), "must include at least one major_comment",
    ),
    (
        "major_comment_without_evidence",
        _review(major=({**MAJOR, "id": "missing-proof", "evidence": []},), minor=(),
                score=(4, "weak_reject")),
        "must include evidence",
    ),
    (
        "duplicate_issue_ids",
        _review(major=({**MAJOR, "id": "duplicate-id"},), minor=({**MINOR, "id": "duplicate-id"},),
                score=(4, "weak_reject")),
        "duplicate review issue id",
    ),
]


def test_finalize_review_writes_feedback_and_validate_review_payload_rejects_inconsistent_payloads(
    tmp_path: Path,
) -> None:
    # finalize_review writes feedback and markdown
    review_json = _write_json(tmp_path / "paper_review.json", _review())
    review_md = tmp_path / "paper_review.md"
    feedback_json = tmp_path / "paper_review_feedback.json"
    feedback_md = tmp_path / "paper_review_feedback.md"

    has_major_comments = finalize_review(
        review_json_path=review_json, markdown_output_path=review_md,
        feedback_json_path=feedback_json, feedback_md_path=feedback_md,
    )

    assert has_major_comments
    assert review_md.is_file()
    assert feedback_json.is_file()
    assert feedback_md.is_file()
    persisted = json.loads(feedback_json.read_text(encoding="utf-8"))
    assert persisted["ok"] is False
    assert persisted["major_comment_count"] == 1
    assert persisted["minor_comment_count"] == 1
    feedback_md_text = feedback_md.read_text(encoding="utf-8")
    assert "claim-gap" in feedback_md_text
    assert "tighten-opening" in feedback_md_text

    # validate_review_payload rejects inconsistent payloads
    for name, payload, message in INCONSISTENT_REVIEW_PAYLOADS:
        review_json = _write_json(tmp_path / name / "paper_review.json", payload)
        with pytest.raises(ValueError, match=message):
            validate_review_payload(review_json)


def test_review_cli_smoke_builds_packet_and_finalizes_feedback(tmp_path: Path) -> None:
    _write(tmp_path / "paper.tex", "\\section{Results}\nA grounded result appears here.\n")
    _write_json(tmp_path / "claims_grounding.json", CLAIMS)
    _write_json(tmp_path / "paper_data.json", {**PAPER_DATA, "totals": {"evaluated_programs": 1}})
    _write_json(tmp_path / "references_provenance.json", {"queries": ["query"], "references": []})
    _write_json(
        tmp_path / "validation_report.json",
        {"feedback": {"errors": [], "figure_validation_warnings": []}},
    )
    packet_path = tmp_path / "paper_review_packet.md"
    review_json = tmp_path / "paper_review.json"
    review_md = tmp_path / "paper_review.md"
    feedback_json = tmp_path / "paper_review_feedback.json"
    feedback_md = tmp_path / "paper_review_feedback.md"

    build_packet = subprocess.run(
        [sys.executable, "src/orchestrator/paper_review.py", "build-packet",
         "--paper-dir", str(tmp_path), "--output", str(packet_path)],
        check=False, capture_output=True, text=True,
    )
    assert build_packet.returncode == 0, build_packet.stderr
    assert packet_path.is_file()
    assert "# Paper review packet" in packet_path.read_text(encoding="utf-8")

    _write_json(review_json, _review(major=({**MAJOR, "id": "grounding-gap"},), minor=(),
                                     score=(4, "weak_reject")))
    finalize = subprocess.run(
        [sys.executable, "src/orchestrator/paper_review.py", "finalize-review",
         "--input", str(review_json), "--markdown-output", str(review_md),
         "--feedback-json", str(feedback_json), "--feedback-md", str(feedback_md)],
        check=False, capture_output=True, text=True,
    )
    assert finalize.returncode == 0, finalize.stderr
    assert review_md.is_file()
    assert feedback_json.is_file()
    assert feedback_md.is_file()
    persisted = json.loads(feedback_json.read_text(encoding="utf-8"))
    assert persisted["major_comment_count"] == 1
    assert persisted["ok"] is False


# ── validation / stage feedback ──


def _validation_feedback(root: Path, report: dict, **kwargs) -> tuple[dict, dict, str]:
    """Returns (feedback, persisted json, markdown text)."""
    root.mkdir(parents=True, exist_ok=True)
    feedback_json_path = root / "validation_feedback.json"
    feedback_md_path = root / "validation_feedback.md"
    feedback = write_validation_feedback(
        output_dir=root, feedback_json_path=feedback_json_path,
        feedback_md_path=feedback_md_path, report=report, **kwargs,
    )
    return (
        feedback,
        json.loads(feedback_json_path.read_text(encoding="utf-8")),
        feedback_md_path.read_text(encoding="utf-8"),
    )


def _stage_dir(root: Path, prompt: str, log: str) -> tuple[Path, Path, Path]:
    output_dir = root / "paper"
    output_dir.mkdir(parents=True)
    return (
        output_dir,
        _write(output_dir / f"{prompt}_prompt.md", f"{prompt}\n"),
        _write(output_dir / f"{log}.log", "first line\nsecond line\nboom\n"),
    )


def test_write_stage_execution_and_contract_feedback_record_failure_details(
    tmp_path: Path,
) -> None:
    # stage execution feedback records missing required outputs
    output_dir, prompt_path, log_path = _stage_dir(tmp_path / "missing", "paper", "paper")
    feedback_json_path = output_dir / "validation_feedback.json"
    feedback_md_path = output_dir / "validation_feedback.md"

    feedback = write_stage_execution_feedback(
        output_dir=output_dir, feedback_json_path=feedback_json_path,
        feedback_md_path=feedback_md_path, stage_name="paper", stage_return_code=71,
        prompt_path=prompt_path, log_path=log_path,
        expected_outputs=["paper.tex", "claims_grounding.json"],
    )

    assert feedback["ok"] is False
    assert feedback["failure_kind"] == "stage_execution"
    assert feedback["stage_name"] == "paper"
    assert "did not leave all required outputs non-empty" in " ; ".join(feedback["hard_errors"])
    assert len(feedback["expected_outputs"]) == 2
    assert feedback_json_path.is_file()
    assert feedback_md_path.is_file()
    persisted = json.loads(feedback_json_path.read_text(encoding="utf-8"))
    assert persisted["stage_return_code"] == 71
    assert len(persisted["expected_outputs"]) == 2
    assert "missing_or_empty" in feedback_md_path.read_text(encoding="utf-8")

    # stage execution feedback records a nonzero exit and the log tail
    output_dir, prompt_path, log_path = _stage_dir(
        tmp_path / "nonzero", "paper_strategy", "paper_strategy"
    )
    feedback_md_path = output_dir / "strategy_validation_feedback.md"

    feedback = write_stage_execution_feedback(
        output_dir=output_dir, feedback_json_path=output_dir / "strategy_validation_feedback.json",
        feedback_md_path=feedback_md_path, stage_name="paper_strategy", stage_return_code=17,
        prompt_path=prompt_path, log_path=log_path,
        expected_outputs=["paper_strategy.md", "paper_strategy.json"],
    )

    assert "exited with code 17" in " ; ".join(feedback["hard_errors"])
    assert feedback["log_tail"] == ["first line", "second line", "boom"]
    md_text = feedback_md_path.read_text(encoding="utf-8")
    assert "## Log tail" in md_text
    assert "boom" in md_text

    # stage contract feedback records invalid artifact details
    output_dir, prompt_path, log_path = _stage_dir(
        tmp_path / "contract", "paper_review", "paper_review"
    )
    _write(output_dir / "paper_review.json", '{"status": "pass"}\n')
    feedback_json_path = output_dir / "paper_review_stage_feedback.json"
    feedback_md_path = output_dir / "paper_review_stage_feedback.md"

    feedback = write_stage_contract_feedback(
        output_dir=output_dir, feedback_json_path=feedback_json_path,
        feedback_md_path=feedback_md_path, stage_name="paper_review",
        prompt_path=prompt_path, log_path=log_path, expected_outputs=["paper_review.json"],
        hard_errors=["review JSON failed validation: missing score"],
    )

    assert feedback["ok"] is False
    assert feedback["failure_kind"] == "stage_contract"
    assert "missing score" in " ; ".join(feedback["hard_errors"])
    assert len(feedback["expected_outputs"]) == 1
    persisted = json.loads(feedback_json_path.read_text(encoding="utf-8"))
    assert persisted["failure_kind"] == "stage_contract"
    assert "review JSON failed validation" in feedback_md_path.read_text(encoding="utf-8")


def test_write_validation_feedback_includes_page_budget_supplement_errors_and_extra_hard_errors(
    tmp_path: Path,
) -> None:
    # page budget details
    root = tmp_path / "pages"
    feedback, persisted, md_text = _validation_feedback(
        root,
        {
            "pdf_path": str(root / "paper.pdf"),
            "feedback": {"figure_validation_warnings": ["png looks cramped"]},
            "page_validation": {
                "checked": True, "paper_format": "alife", "min_body_pages": 8,
                "min_last_body_page_fill_ratio": 0.7, "max_body_pages": 8,
                "exclude_references": True, "total_pages": 7, "reference_start_page": 6,
                "body_pages": 5, "last_body_page_number": 6, "last_body_page_fill_ratio": 0.18,
                "last_body_page_counted": False, "supplement_start_page": 7, "supplement_pages": 1,
                "errors": [
                    "alife page minimum not met: body uses 5 pages (min 8 excluding references)"
                ],
            },
        },
        error=RuntimeError("page-limit validation failed: alife page minimum not met"),
    )
    assert feedback["ok"] is False
    assert feedback["page_validation"]["body_pages"] == 5
    assert feedback["page_validation"]["supplement_pages"] == 1
    assert "page-limit validation failed" in feedback["hard_errors"][0]
    assert "body_pages" in persisted["page_validation"]
    for fragment in (
        "## Page budget", "min_body_pages", "last_body_page_fill_ratio", "supplement_start_page",
        "png looks cramped",
    ):
        assert fragment in md_text

    # supplementary information errors
    feedback, _, md_text = _validation_feedback(
        tmp_path / "supplement",
        {
            "ok": False,
            "feedback": {"figure_validation_warnings": []},
            "supplementary_information_validation": {
                "checked": True, "appendix_present": False, "heading_present": False,
                "placed_after_references": False, "visible_character_count": 0,
                "substantive_element_count": 0,
                "hard_errors": ["paper.tex missing Supplementary Information heading"],
            },
        },
    )
    assert feedback["ok"] is False
    assert "missing Supplementary Information heading" in " ; ".join(feedback["hard_errors"])
    assert feedback["supplementary_information_validation"]["checked"] is True
    assert "## Supplementary Information validation" in md_text

    # extra hard errors mark the feedback not ok
    root = tmp_path / "extra"
    feedback, persisted, md_text = _validation_feedback(
        root, {"pdf_path": str(root / "paper.pdf")},
        extra_hard_errors=["method scope audit failed"],
    )
    assert feedback["ok"] is False
    assert feedback["hard_errors"] == ["method scope audit failed"]
    assert persisted["ok"] is False
    assert "method scope audit failed" in md_text


def test_write_validation_feedback_surfaces_dangling_evidence_links(tmp_path: Path) -> None:
    feedback, persisted, md_text = _validation_feedback(
        tmp_path,
        {
            "feedback": {"figure_validation_warnings": []},
            "evidence_validation": {
                "checked": True,
                "bundle_path": str(tmp_path / "scientific-evidence"),
                "bundle_present": True,
                "verification_report_path": str(
                    tmp_path / "scientific-evidence/verification-report.json"
                ),
                "summary": {
                    "total_references": 2, "resolved_references": 1, "anchored_references": 0,
                    "dangling_references": 1, "manifests_checked": 0,
                },
                "dangling": [{
                    "subject_path": "paper.tex", "subject_kind": "text-artifact",
                    "raw_reference": "@evidence[[results/ghost.json#acc]]",
                    "target": "results/ghost.json", "locator": "acc",
                    "messages": [
                        "Target did not resolve to a file path, URI, or known ledger id."
                    ],
                }],
                "paper_token_count": 2,
                "hard_errors": [
                    "scientific-evidence verifier found 1 dangling @evidence[[...]] link(s)."
                ],
            },
        },
        error=RuntimeError("scientific-evidence validation failed: dangling link in paper.tex"),
    )
    assert feedback["ok"] is False
    evidence = feedback["evidence_validation"]
    assert evidence["checked"] is True
    assert evidence["bundle_present"] is True
    assert evidence["summary"]["dangling_references"] == 1
    assert len(evidence["dangling"]) == 1
    assert evidence["dangling"][0]["subject_path"] == "paper.tex"
    assert evidence["dangling"][0]["raw_reference"] == "@evidence[[results/ghost.json#acc]]"
    assert "Target did not resolve" in evidence["dangling"][0]["messages"][0]
    assert persisted["evidence_validation"]["summary"]["dangling_references"] == 1
    for fragment in (
        "## Evidence validation", "### Dangling `@evidence[[...]]` links to fix", "paper.tex",
        "@evidence[[results/ghost.json#acc]]",
    ):
        assert fragment in md_text


def _audit_report(finding: dict, **extra) -> dict:
    return {
        "feedback": {"figure_validation_warnings": []},
        "agentic_evidence_audit": {
            "checked": True, "verdict": "fail", "summary": "One claim is wrong.",
            "finding_count": 1, "findings": [finding],
        },
        **extra,
    }


def test_write_validation_feedback_includes_agentic_audit_findings_and_preserves_report_error(
    tmp_path: Path,
) -> None:
    # agentic evidence audit findings
    feedback, persisted, md_text = _validation_feedback(
        tmp_path / "findings",
        _audit_report({
            "id": "results-overclaim-1", "section": "Results",
            "claim": "The paper says the method wins across tasks.",
            "issue": "claims_grounding.json only supports one benchmark-specific comparison.",
            "evidence": "paper.tex Results paragraph 2; claims_grounding.json claim[3]",
            "fix": "Narrow the sentence to the supported benchmark.",
        }),
        error=RuntimeError("agentic evidence audit failed"),
    )
    assert feedback["ok"] is False
    audit = feedback["agentic_evidence_audit"]
    assert audit["checked"] is True
    assert audit["verdict"] == "fail"
    assert audit["finding_count"] == 1
    assert audit["findings"][0]["id"] == "results-overclaim-1"
    assert persisted["agentic_evidence_audit"]["finding_count"] == 1
    for fragment in (
        "## Agentic evidence audit", "### Audit findings to fix", "results-overclaim-1",
        "Narrow the sentence",
    ):
        assert fragment in md_text

    # the report error is preserved when the audit failure is added
    feedback, _, md_text = _validation_feedback(
        tmp_path / "preserved",
        _audit_report(
            {"id": "numeric-range-1", "section": "Results", "claim": "range claim",
             "issue": "range excludes one grounded value",
             "evidence": "claims_grounding.json claim[12]", "fix": "Widen the range."},
            ok=False, error="figure validation failed: missing required figure",
        ),
        error=RuntimeError("agentic evidence audit failed"),
    )
    assert feedback["ok"] is False
    assert "agentic evidence audit failed" in feedback["hard_errors"]
    assert "figure validation failed: missing required figure" in feedback["hard_errors"]
    for fragment in ("agentic evidence audit failed", "figure validation failed", "numeric-range-1"):
        assert fragment in md_text


# ── method scope audit ──


def test_scope_audit_rejects_empty_scoped_evidence_paths(tmp_path: Path) -> None:
    _write(tmp_path / "paper.tex", "Shared method.\n")
    _write(tmp_path / "method_scope_audit.md", "Scoped audit.\n")
    _write_json(
        tmp_path / "method_scope_audit.json",
        {
            "shared_method_elements": ["Shared method."],
            "scoped_method_elements": [{
                "element": "Specialized repair", "scope": "One implementation",
                "rationale": "Not shared", "evidence_paths": [],
            }],
            "claims_requiring_scoping": [],
            "manuscript_focus": {
                "shared_method_summary": "Shared method.",
                "scope_strategy": "Keep specialized repair scoped.",
            },
        },
    )
    report = validate_method_scope_audit(tmp_path)
    assert report["ok"] is False
    assert report["hard_errors"] == [
        "scoped_method_elements[0].evidence_paths must be a list of non-empty strings"
    ]


# ── paper finalization and fast resume ──


def _write_resumable_workspace(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True)
    for filename in paper_finalize.RESUMABLE_PAPER_ARTIFACTS:
        (output_dir / filename).write_text("content", encoding="utf-8")
    return output_dir


def _write_passing_audit(output_dir: Path) -> None:
    _write_json(
        output_dir / "paper_evidence_audit.json",
        {"verdict": "pass", "finding_count": 0, "findings": [],
         "input_sha256": paper_finalize.audit_input_sha256(output_dir)},
    )


def _write_stale_audit(output_dir: Path) -> None:
    _write_passing_audit(_write_resumable_workspace(output_dir))
    _write(output_dir / "paper.tex", "changed")


def _resume(root: Path, **overrides: Any) -> bool:
    kwargs: dict[str, Any] = dict(
        output_dir=root / "paper", results_dir=root / "results",
        global_memory_dir=root / "global_memory", container_image=None,
        container_network="bridge", paper_review_rounds=0,
    )
    kwargs.update(overrides)
    return paper_finalize.finalize_resumed_paper_if_valid(**kwargs)


def _finalize_workspace(root: Path, profile: str) -> tuple[Path, Path, Path]:
    output_dir = root / "paper"
    results_dir = root / "results"
    global_memory_dir = root / "global_memory"
    output_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    state_store.ensure_global_memory_files(global_memory_dir)
    _write(output_dir / "paper.tex", "\\section{Test}\n")
    (output_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    _write(output_dir / "paper_prompt.md", "# prompt\n")
    _write_json(output_dir / "paper_data.json", {"paper_profile": profile})
    _write(output_dir / "references.bib", "% refs\n")
    _write(output_dir / "references_provenance.json", "{}")
    _write(output_dir / "claims_grounding.json", '{"claims": []}')
    return output_dir, results_dir, global_memory_dir


def _finalize(output_dir: Path, results_dir: Path, global_memory_dir: Path, **kwargs) -> None:
    finalize_paper_artifacts(
        output_dir=output_dir, results_dir=results_dir, global_memory_dir=global_memory_dir,
        validation_feedback_json_path=output_dir / "validation_feedback.json",
        validation_feedback_md_path=output_dir / "validation_feedback.md", **kwargs,
    )


def test_paper_shell_resume_and_audit_gates_distinguish_operational_failures() -> None:
    script = PAPER_SH.read_text(encoding="utf-8")
    # resume distinguishes operational failure
    assert "resume_rc=$?" in script
    assert (
        'if [ "$resume_rc" -gt 1 ]; then\n'
        '    paper_gate_failed "finalization" "phase=resume" "exit_code=$resume_rc"\n'
        '    exit "$resume_rc"'
    ) in script
    # operational audit failure is gated but contract exhaustion is not
    for fragment in (
        'if [ "$validation_stage_rc" -eq 0 ]; then\n    break',
        'if [ "$audit_stage_rc" -eq 1 ]; then\n'
        '    mark_incomplete_paper_pipeline "paper evidence audit failed operationally;',
        'if [ "$audit_stage_rc" -eq 3 ]; then\n'
        '    echo "[paper.sh] evidence audit output contract remained invalid;',
        'if [ "$finalize_rc" -ne 3 ]; then\n'
        '      echo "[paper.sh] evidence audit finalization failed operationally:',
    ):
        assert fragment in script


SKIPPED_RESUME_WORKSPACES = [
    ("incomplete_workspace", lambda out: None, 0),
    ("requires_audit", _write_resumable_workspace, 0),
    ("preserves_configured_review_gate", _write_resumable_workspace, 1),
    (
        "preserves_audit_findings",
        lambda out: _write_json(_write_resumable_workspace(out) / "paper_evidence_audit.json",
                                {"verdict": "fail", "finding_count": 1}),
        0,
    ),
    ("rejects_stale_audit", _write_stale_audit, 0),
]


def test_finalize_resumed_paper_if_valid_skips_without_validating(tmp_path: Path) -> None:
    for name, prepare, review_rounds in SKIPPED_RESUME_WORKSPACES:
        root = tmp_path / name
        prepare(root / "paper")
        with patch.object(paper_finalize, "validate_and_compile_paper") as validate:
            assert _resume(root, paper_review_rounds=review_rounds) is False, name
        validate.assert_not_called()


def test_paper_audit_freshness_propagates_io_failure_and_treats_malformed_report_as_stale(
    tmp_path: Path,
) -> None:
    _write_passing_audit(tmp_path)
    # input-hash I/O failure propagates
    with (
        patch.object(paper_finalize, "audit_input_sha256", side_effect=PermissionError("denied")),
        pytest.raises(PermissionError),
    ):
        paper_finalize._paper_audit_is_current_and_passing(tmp_path)
    # a malformed verifier report is treated as stale
    _write(tmp_path / "scientific-evidence" / "verification-report.json", "not json")
    assert paper_finalize._paper_audit_is_current_and_passing(tmp_path) is False


def test_finalize_resumed_paper_if_valid_revalidates_finalizes_and_rechecks_hashes(
    tmp_path: Path,
) -> None:
    # revalidates and finalizes
    root = tmp_path / "finalizes"
    output_dir = _write_resumable_workspace(root / "paper")
    _write_passing_audit(output_dir)
    with (
        patch.object(paper_finalize, "validate_and_compile_paper", return_value={"ok": True}) as validate,
        patch.object(paper_finalize, "write_validation_feedback") as feedback,
        patch.object(paper_finalize, "finalize_paper_artifacts") as finalize,
    ):
        completed = _resume(root, container_image="image@sha256:digest", container_network="host")

    assert completed is True
    persisted_report = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert persisted_report["agentic_evidence_audit"]["verdict"] == "pass"
    validate.assert_called_once_with(
        output_dir, container_image="image@sha256:digest", container_network="host"
    )
    finalize.assert_called_once_with(
        output_dir=output_dir, results_dir=root / "results",
        global_memory_dir=root / "global_memory",
        validation_feedback_json_path=output_dir / "validation_feedback.json",
        validation_feedback_md_path=output_dir / "validation_feedback.md",
        manifest_dir=None,
    )
    feedback.assert_called_once()

    # rechecks hashes after validation
    root = tmp_path / "rechecks"
    output_dir = _write_resumable_workspace(root / "paper")
    _write_passing_audit(output_dir)

    def mutate_input(*args: object, **kwargs: object) -> dict[str, bool]:
        (output_dir / "paper.tex").write_text("changed", encoding="utf-8")
        return {"ok": True}

    with (
        patch.object(paper_finalize, "validate_and_compile_paper", side_effect=mutate_input),
        patch.object(paper_finalize, "finalize_paper_artifacts") as finalize,
    ):
        assert _resume(root) is False
    finalize.assert_not_called()


def test_finalize_resumed_paper_if_valid_propagates_infrastructure_and_finalization_errors(
    tmp_path: Path,
) -> None:
    # infrastructure error
    root = tmp_path / "infrastructure"
    _write_passing_audit(_write_resumable_workspace(root / "paper"))
    with (
        patch.object(
            paper_finalize, "validate_and_compile_paper",
            side_effect=paper_finalize.PaperValidationInfrastructureError("docker unavailable"),
        ),
        pytest.raises(paper_finalize.PaperValidationInfrastructureError, match="docker unavailable"),
    ):
        _resume(root, container_image="image@sha256:digest", container_network="host")
    # finalization error
    root = tmp_path / "finalization"
    _write_passing_audit(_write_resumable_workspace(root / "paper"))
    with (
        patch.object(paper_finalize, "validate_and_compile_paper", return_value={"ok": True}),
        patch.object(paper_finalize, "write_validation_feedback"),
        patch.object(
            paper_finalize, "finalize_paper_artifacts",
            side_effect=RuntimeError("manifest write failed"),
        ),
        pytest.raises(RuntimeError, match="manifest write failed"),
    ):
        _resume(root)


def test_finalize_paper_artifacts_tolerates_missing_validation_report(tmp_path: Path) -> None:
    output_dir, results_dir, global_memory_dir = _finalize_workspace(tmp_path, "standalone")

    _finalize(output_dir, results_dir, global_memory_dir, manifest_dir=output_dir)

    manifest = json.loads((output_dir / "latest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_report"] is None
    assert manifest["paper_profile"] == "standalone"
    assert manifest["supplementary_information"]["embedded"] is False
    assert manifest["supplementary_information"]["status"] == "missing_validation"
    memory_lines = (global_memory_dir / "memory.jsonl").read_text(encoding="utf-8").splitlines()
    assert memory_lines
    assert json.loads(memory_lines[-1])["step"] == "paper"


def test_finalize_paper_artifacts_reflects_validation_report_supplement_status(
    tmp_path: Path,
) -> None:
    # keeps the validation report when present
    output_dir, results_dir, global_memory_dir = _finalize_workspace(tmp_path / "validated", "default")
    _write_json(
        output_dir / "validation_report.json",
        {
            "feedback": {},
            "supplementary_information_validation": {
                "checked": True, "appendix_present": True, "heading_present": True,
                "placed_after_references": True, "pre_bibliography_clearpage_present": True,
                "clearpage_present": True, "visible_character_count": 150,
                "substantive_element_count": 1, "rendered_checked": True,
                "rendered_heading_present": True, "rendered_visible_character_count": 150,
                "hard_errors": [],
            },
        },
    )
    _write(output_dir / "paper_evidence_audit.json", '{"verdict": "fail"}')

    _finalize(output_dir, results_dir, global_memory_dir)

    manifest = json.loads((results_dir / "paper" / "latest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_report"] == str((output_dir / "validation_report.json").resolve())
    assert manifest["paper_evidence_audit_json"] == str(
        (output_dir / "paper_evidence_audit.json").resolve()
    )
    assert manifest["supplementary_information"]["embedded"] is True
    assert manifest["supplementary_information"]["status"] == "validated"
    assert manifest["supplementary_information"]["validation"]["hard_errors"] == []

    # failed supplement validation is marked not embedded
    output_dir, results_dir, global_memory_dir = _finalize_workspace(tmp_path / "failed", "default")
    _write_json(
        output_dir / "validation_report.json",
        {"supplementary_information_validation": {
            "checked": True, "heading_present": False, "rendered_checked": True,
            "rendered_heading_present": False,
            "hard_errors": ["missing Supplementary Information heading"],
        }},
    )

    _finalize(output_dir, results_dir, global_memory_dir)

    manifest = json.loads((results_dir / "paper" / "latest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["supplementary_information"]["embedded"] is False
    assert manifest["supplementary_information"]["status"] == "validation_failed"
