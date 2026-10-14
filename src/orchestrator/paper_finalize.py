"""Finalize paper artifacts after the paper pipeline completes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.orchestrator import state as state_store
from src.orchestrator.paper_evidence_audit import audit_input_sha256
from src.orchestrator.paper_feedback import write_validation_feedback
from src.orchestrator.paper_manifest import latest_manifest_lock, write_manifest_atomically
from src.orchestrator.paper_step import (
    PaperValidationInfrastructureError,
    validate_and_compile_paper,
)


def _supplement_manifest_state(
    validation: dict[str, object] | None,
) -> tuple[bool, str]:
    if validation is None:
        return False, "missing_validation"

    hard_errors = validation.get("hard_errors")
    boolean_checks = (
        "checked",
        "appendix_present",
        "heading_present",
        "placed_after_references",
        "pre_bibliography_clearpage_present",
        "clearpage_present",
        "rendered_checked",
        "rendered_heading_present",
    )
    counts = (
        ("visible_character_count", 100),
        ("substantive_element_count", 1),
        ("rendered_visible_character_count", 100),
    )
    checks_passed = all(validation.get(key) is True for key in boolean_checks)
    counts_passed = True
    for key, minimum in counts:
        value = validation.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            counts_passed = False
            break
    if hard_errors == [] and checks_passed and counts_passed:
        return True, "validated"
    return False, "validation_failed"


RESUMABLE_PAPER_ARTIFACTS = (
    "paper.tex",
    "references.bib",
    "claims_grounding.json",
    "paper_data.json",
)


def _paper_audit_is_current_and_passing(output_dir: Path) -> bool:
    """Return whether a passing audit covers the exact current manuscript inputs."""
    path = output_dir / "paper_evidence_audit.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if str(payload.get("verdict", "")).strip().lower() != "pass":
        return False
    findings = payload.get("findings")
    if not isinstance(findings, list) or findings:
        return False
    finding_count = payload.get("finding_count")
    if not isinstance(finding_count, int) or isinstance(finding_count, bool):
        return False
    if finding_count != 0:
        return False
    try:
        expected_hashes = audit_input_sha256(output_dir)
    except (FileNotFoundError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return False
    return bool(expected_hashes) and payload.get("input_sha256") == expected_hashes


def _restore_passing_audit_report(output_dir: Path, report: dict[str, object]) -> None:
    audit_payload = json.loads(
        (output_dir / "paper_evidence_audit.json").read_text(encoding="utf-8")
    )
    report["agentic_evidence_audit"] = {
        "checked": True,
        "verdict": audit_payload["verdict"],
        "summary": audit_payload.get("summary", ""),
        "finding_count": audit_payload["finding_count"],
        "findings": audit_payload["findings"],
        "input_sha256": audit_payload["input_sha256"],
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def finalize_resumed_paper_if_valid(
    *,
    output_dir: Path,
    results_dir: Path,
    global_memory_dir: Path,
    container_image: str | None,
    container_network: str,
    paper_review_rounds: int,
    manifest_dir: Path | None = None,
) -> bool:
    """Revalidate and finalize an already complete paper workspace."""
    if paper_review_rounds > 0:
        return False
    if not all((output_dir / filename).is_file() for filename in RESUMABLE_PAPER_ARTIFACTS):
        return False
    if not _paper_audit_is_current_and_passing(output_dir):
        return False
    try:
        report = validate_and_compile_paper(
            output_dir,
            container_image=container_image,
            container_network=container_network,
        )
    except PaperValidationInfrastructureError:
        raise
    except RuntimeError as exc:
        print(f"[paper.sh] resumed paper is not yet valid: {exc}")
        return False
    if report.get("ok") is not True:
        return False
    if not _paper_audit_is_current_and_passing(output_dir):
        return False
    _restore_passing_audit_report(output_dir, report)

    feedback_json_path = output_dir / "validation_feedback.json"
    feedback_md_path = output_dir / "validation_feedback.md"
    write_validation_feedback(
        output_dir=output_dir,
        feedback_json_path=feedback_json_path,
        feedback_md_path=feedback_md_path,
        report=report,
    )
    state_store.ensure_global_memory_files(global_memory_dir)
    finalize_paper_artifacts(
        output_dir=output_dir,
        results_dir=results_dir,
        global_memory_dir=global_memory_dir,
        validation_feedback_json_path=feedback_json_path,
        validation_feedback_md_path=feedback_md_path,
        manifest_dir=manifest_dir,
    )
    return True


def finalize_paper_artifacts(
    *,
    output_dir: Path,
    results_dir: Path,
    global_memory_dir: Path,
    validation_feedback_json_path: Path,
    validation_feedback_md_path: Path,
    manifest_dir: Path | None = None,
) -> None:
    """Persist final manifest and fallback paper memory entry after the paper pipeline completes."""
    report_path = output_dir / "validation_report.json"
    report: dict[str, object] = {}
    if report_path.exists():
        parsed_report = json.loads(report_path.read_text(encoding="utf-8"))
        report = parsed_report if isinstance(parsed_report, dict) else {}
    else:
        print(f"[paper.sh] warning: missing validation report during finalization: {report_path}")

    raw_feedback = report.get("feedback")
    feedback = raw_feedback if isinstance(raw_feedback, dict) else {}
    figure_feedback = feedback.get("figure_validation_errors", [])
    if isinstance(figure_feedback, list) and figure_feedback:
        print("[paper.sh] final figure validation feedback:")
        for message in figure_feedback:
            print(f"  - {message}")

    raw_supplement_validation = report.get("supplementary_information_validation")
    supplement_validation = (
        raw_supplement_validation if isinstance(raw_supplement_validation, dict) else None
    )
    supplement_embedded, supplement_status = _supplement_manifest_state(supplement_validation)

    global_memory_jsonl_path = state_store.global_memory_jsonl_path(global_memory_dir)
    if not global_memory_jsonl_path.exists():
        raise SystemExit(f"Missing required global memory artifact: {global_memory_jsonl_path}")

    entries: list[dict[str, object]] = []
    for line in global_memory_jsonl_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)

    if not any(entry.get("step") == "paper" for entry in entries):
        supplement_evidence = (
            "paper.tex (validated embedded Supplementary Information)"
            if supplement_embedded
            else "paper.tex (Supplementary Information validation incomplete)"
        )
        fallback_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": "paper",
            "scope": "paper",
            "summary": "paper artifacts generated (auto-healed by paper.sh)",
            "evidence": (f"{supplement_evidence} + claims_grounding.json + validation_report.json"),
            "next_action": "publish latest_manifest.json",
        }
        state_store.append_jsonl(global_memory_jsonl_path, fallback_entry)
        global_memory_md_path = state_store.global_memory_md_path(global_memory_dir)
        with global_memory_md_path.open("a", encoding="utf-8") as handle:
            handle.write(state_store.format_memory_markdown_entry(fallback_entry))
        print("[paper.sh] warning: appended fallback paper-step memory entry")

    figure_manifest_path = output_dir / "figure_manifest.json"
    audit_json_path = output_dir / "paper_evidence_audit.json"
    audit_md_path = output_dir / "paper_evidence_audit.md"
    audit_feedback_json_path = output_dir / "paper_evidence_audit_feedback.json"
    audit_feedback_md_path = output_dir / "paper_evidence_audit_feedback.md"
    latest_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_profile": json.loads(
            (output_dir / "paper_data.json").read_text(encoding="utf-8")
        ).get("paper_profile", "default"),
        "paper_dir": str(output_dir.resolve()),
        "paper_tex": str((output_dir / "paper.tex").resolve()),
        "paper_pdf": str((output_dir / "paper.pdf").resolve()),
        "supplementary_information": {
            "embedded": supplement_embedded,
            "status": supplement_status,
            "paper_tex": str((output_dir / "paper.tex").resolve()),
            "paper_pdf": str((output_dir / "paper.pdf").resolve()),
            "validation": supplement_validation,
        },
        "paper_prompt": str((output_dir / "paper_prompt.md").resolve()),
        "paper_data": str((output_dir / "paper_data.json").resolve()),
        "references_bib": str((output_dir / "references.bib").resolve()),
        "references_provenance": str((output_dir / "references_provenance.json").resolve()),
        "claims_grounding": str((output_dir / "claims_grounding.json").resolve()),
        "figure_manifest": str(figure_manifest_path.resolve())
        if figure_manifest_path.exists()
        else None,
        "paper_evidence_audit_json": str(audit_json_path.resolve())
        if audit_json_path.exists()
        else None,
        "paper_evidence_audit_md": str(audit_md_path.resolve()) if audit_md_path.exists() else None,
        "paper_evidence_audit_feedback_json": (
            str(audit_feedback_json_path.resolve()) if audit_feedback_json_path.exists() else None
        ),
        "paper_evidence_audit_feedback_md": (
            str(audit_feedback_md_path.resolve()) if audit_feedback_md_path.exists() else None
        ),
        "validation_feedback_json": (
            str(validation_feedback_json_path.resolve())
            if validation_feedback_json_path.exists()
            else None
        ),
        "validation_feedback_md": (
            str(validation_feedback_md_path.resolve())
            if validation_feedback_md_path.exists()
            else None
        ),
        "validation_report": str(report_path.resolve()) if report_path.exists() else None,
    }

    paper_root = manifest_dir if manifest_dir is not None else results_dir / "paper"
    paper_root.mkdir(parents=True, exist_ok=True)
    with latest_manifest_lock(paper_root):
        write_manifest_atomically(paper_root / "latest_manifest.json", latest_manifest)
