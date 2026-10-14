"""Paper-stage execution feedback helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_TAIL_LINES = 40


def _write_feedback_files(
    *,
    feedback: dict[str, Any],
    feedback_json_path: Path,
    feedback_md_path: Path,
    markdown_lines: list[str],
) -> dict[str, Any]:
    feedback_json_path.write_text(json.dumps(feedback, indent=2, sort_keys=True), encoding="utf-8")
    feedback_md_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")
    return feedback


def _append_unique(messages: list[str], message: object) -> None:
    text = str(message).strip()
    if text and text not in messages:
        messages.append(text)


def _validation_report_hard_errors(report: dict[str, Any] | None) -> list[str]:
    """Return top-level validation failure messages that repair prompts must see."""
    if not isinstance(report, dict):
        return []

    messages: list[str] = []
    if report.get("ok") is False and report.get("error"):
        _append_unique(messages, report["error"])
        return messages

    if report.get("ok") is not False:
        return messages

    for block_name in (
        "figure_validation",
        "page_validation",
        "evidence_validation",
        "reference_validation",
        "standalone_language_validation",
        "venue_validation",
        "supplementary_information_validation",
    ):
        block = report.get(block_name)
        if not isinstance(block, dict):
            continue
        for key in ("hard_errors", "errors"):
            values = block.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                _append_unique(messages, value)
    return messages


def write_validation_feedback(
    *,
    output_dir: Path,
    feedback_json_path: Path,
    feedback_md_path: Path,
    report: dict[str, Any] | None = None,
    error: BaseException | None = None,
    extra_hard_errors: list[str] | None = None,
    extra_soft_warnings: list[str] | None = None,
    auxiliary_checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist validation-loop feedback, including structured page-budget details."""
    output_dir = output_dir.expanduser().resolve()
    feedback_json_path = feedback_json_path.expanduser().resolve()
    feedback_md_path = feedback_md_path.expanduser().resolve()

    page_validation = report.get("page_validation") if isinstance(report, dict) else None
    if not isinstance(page_validation, dict):
        page_validation = {}

    figure_feedback = (
        report.get("feedback", {}).get("figure_validation_warnings", [])
        if isinstance(report, dict)
        else []
    )
    if not isinstance(figure_feedback, list):
        figure_feedback = []

    evidence_validation = report.get("evidence_validation") if isinstance(report, dict) else None
    if not isinstance(evidence_validation, dict):
        evidence_validation = {}

    agentic_evidence_audit = (
        report.get("agentic_evidence_audit") if isinstance(report, dict) else None
    )
    if not isinstance(agentic_evidence_audit, dict):
        agentic_evidence_audit = {}

    supplementary_validation = (
        report.get("supplementary_information_validation") if isinstance(report, dict) else None
    )
    if not isinstance(supplementary_validation, dict):
        supplementary_validation = {}

    hard_errors: list[str] = []
    if error is not None:
        _append_unique(hard_errors, error)
    for message in _validation_report_hard_errors(report if isinstance(report, dict) else None):
        _append_unique(hard_errors, message)
    if extra_hard_errors:
        for message in extra_hard_errors:
            _append_unique(hard_errors, message)
    report_ok = (
        report.get("ok")
        if isinstance(report, dict) and isinstance(report.get("ok"), bool)
        else None
    )
    feedback_ok = error is None and report_ok is not False and not hard_errors
    pdf_path = report.get("pdf_path") if isinstance(report, dict) else None
    soft_warnings = [str(message) for message in figure_feedback]
    if extra_soft_warnings:
        soft_warnings.extend(str(message) for message in extra_soft_warnings)

    raw_evidence_summary = evidence_validation.get("summary")
    evidence_summary: dict[str, Any] = (
        raw_evidence_summary if isinstance(raw_evidence_summary, dict) else {}
    )
    evidence_dangling = evidence_validation.get("dangling", [])
    if not isinstance(evidence_dangling, list):
        evidence_dangling = []
    evidence_hard = evidence_validation.get("hard_errors", [])
    if not isinstance(evidence_hard, list):
        evidence_hard = []

    audit_findings = agentic_evidence_audit.get("findings", [])
    if not isinstance(audit_findings, list):
        audit_findings = []

    feedback: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_dir": str(output_dir),
        "ok": feedback_ok,
        "hard_errors": hard_errors,
        "soft_warnings": soft_warnings,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "evidence_validation": {
            "checked": bool(evidence_validation.get("checked", False)),
            "bundle_path": evidence_validation.get("bundle_path"),
            "bundle_present": bool(evidence_validation.get("bundle_present", False)),
            "verification_report_path": evidence_validation.get("verification_report_path"),
            "paper_token_count": evidence_summary.get(
                "total_references", evidence_validation.get("paper_token_count", 0)
            ),
            "summary": {
                "total_references": evidence_summary.get("total_references", 0),
                "resolved_references": evidence_summary.get("resolved_references", 0),
                "anchored_references": evidence_summary.get("anchored_references", 0),
                "dangling_references": evidence_summary.get("dangling_references", 0),
                "manifests_checked": evidence_summary.get("manifests_checked", 0),
            },
            "dangling": [
                {
                    "subject_path": item.get("subject_path") if isinstance(item, dict) else None,
                    "subject_kind": item.get("subject_kind") if isinstance(item, dict) else None,
                    "raw_reference": item.get("raw_reference") if isinstance(item, dict) else None,
                    "target": item.get("target") if isinstance(item, dict) else None,
                    "locator": item.get("locator") if isinstance(item, dict) else None,
                    "messages": (
                        [str(message) for message in item.get("messages", [])]
                        if isinstance(item, dict) and isinstance(item.get("messages"), list)
                        else []
                    ),
                }
                for item in evidence_dangling
            ],
            "errors": [str(message) for message in evidence_hard],
        },
        "agentic_evidence_audit": {
            "checked": bool(agentic_evidence_audit.get("checked", False)),
            "verdict": agentic_evidence_audit.get("verdict"),
            "summary": agentic_evidence_audit.get("summary"),
            "finding_count": int(agentic_evidence_audit.get("finding_count", 0) or 0),
            "findings": [
                {
                    "id": item.get("id") if isinstance(item, dict) else None,
                    "section": item.get("section") if isinstance(item, dict) else None,
                    "claim": item.get("claim") if isinstance(item, dict) else None,
                    "issue": item.get("issue") if isinstance(item, dict) else None,
                    "evidence": item.get("evidence") if isinstance(item, dict) else None,
                    "fix": item.get("fix") if isinstance(item, dict) else None,
                }
                for item in audit_findings
            ],
        },
        "supplementary_information_validation": {
            "checked": bool(supplementary_validation.get("checked", False)),
            "appendix_present": bool(supplementary_validation.get("appendix_present", False)),
            "heading_present": bool(supplementary_validation.get("heading_present", False)),
            "placed_after_references": bool(
                supplementary_validation.get("placed_after_references", False)
            ),
            "pre_bibliography_clearpage_present": bool(
                supplementary_validation.get("pre_bibliography_clearpage_present", False)
            ),
            "clearpage_present": bool(
                supplementary_validation.get("clearpage_present", False)
            ),
            "visible_character_count": int(
                supplementary_validation.get("visible_character_count", 0) or 0
            ),
            "substantive_element_count": int(
                supplementary_validation.get("substantive_element_count", 0) or 0
            ),
            "rendered_checked": bool(supplementary_validation.get("rendered_checked", False)),
            "rendered_heading_present": bool(
                supplementary_validation.get("rendered_heading_present", False)
            ),
            "rendered_visible_character_count": int(
                supplementary_validation.get("rendered_visible_character_count", 0) or 0
            ),
            "errors": [str(message) for message in supplementary_validation.get("hard_errors", [])]
            if isinstance(supplementary_validation.get("hard_errors", []), list)
            else [],
        },
        "page_validation": {
            "checked": bool(page_validation.get("checked", False)),
            "paper_format": page_validation.get("paper_format"),
            "min_body_pages": page_validation.get("min_body_pages"),
            "min_last_body_page_fill_ratio": page_validation.get("min_last_body_page_fill_ratio"),
            "max_body_pages": page_validation.get("max_body_pages"),
            "exclude_references": bool(page_validation.get("exclude_references", False)),
            "total_pages": page_validation.get("total_pages"),
            "reference_start_page": page_validation.get("reference_start_page"),
            "supplement_start_page": page_validation.get("supplement_start_page"),
            "supplement_pages": page_validation.get("supplement_pages", 0),
            "body_pages": page_validation.get("body_pages"),
            "last_body_page_number": page_validation.get("last_body_page_number"),
            "last_body_page_fill_ratio": page_validation.get("last_body_page_fill_ratio"),
            "last_body_page_counted": page_validation.get("last_body_page_counted"),
            "errors": [str(message) for message in page_validation.get("errors", [])]
            if isinstance(page_validation.get("errors", []), list)
            else [],
        },
    }
    if auxiliary_checks:
        feedback["auxiliary_checks"] = auxiliary_checks

    lines = [
        "# Paper validation feedback",
        "",
        f"- generated_at: {feedback['generated_at']}",
        f"- paper_dir: `{feedback['paper_dir']}`",
        f"- ok: `{feedback['ok']}`",
        "",
    ]
    if hard_errors:
        lines.append("## Hard errors")
        lines.extend(f"- {message}" for message in hard_errors)
        lines.append("")

    if feedback["supplementary_information_validation"]["checked"]:
        supplement_block = feedback["supplementary_information_validation"]
        lines.extend(
            [
                "## Supplementary Information validation",
                f"- appendix_present: `{supplement_block['appendix_present']}`",
                f"- heading_present: `{supplement_block['heading_present']}`",
                f"- placed_after_references: `{supplement_block['placed_after_references']}`",
                f"- pre_bibliography_clearpage_present: `{supplement_block['pre_bibliography_clearpage_present']}`",
                f"- clearpage_present: `{supplement_block['clearpage_present']}`",
                f"- visible_character_count: `{supplement_block['visible_character_count']}`",
                f"- substantive_element_count: `{supplement_block['substantive_element_count']}`",
                f"- rendered_checked: `{supplement_block['rendered_checked']}`",
                f"- rendered_heading_present: `{supplement_block['rendered_heading_present']}`",
                f"- rendered_visible_character_count: `{supplement_block['rendered_visible_character_count']}`",
                "",
            ]
        )

    if feedback["page_validation"]["checked"]:
        page_block = feedback["page_validation"]
        lines.extend(
            [
                "## Page budget",
                f"- paper_format: `{page_block['paper_format']}`",
                f"- body_pages: `{page_block['body_pages']}`",
                f"- min_body_pages: `{page_block['min_body_pages']}`",
                f"- min_last_body_page_fill_ratio: `{page_block['min_last_body_page_fill_ratio']}`",
                f"- max_body_pages: `{page_block['max_body_pages']}`",
                f"- exclude_references: `{page_block['exclude_references']}`",
                f"- total_pages: `{page_block['total_pages']}`",
                f"- reference_start_page: `{page_block['reference_start_page']}`",
                f"- supplement_start_page: `{page_block['supplement_start_page']}`",
                f"- supplement_pages: `{page_block['supplement_pages']}`",
                f"- last_body_page_number: `{page_block['last_body_page_number']}`",
                f"- last_body_page_fill_ratio: `{page_block['last_body_page_fill_ratio']}`",
                f"- last_body_page_counted: `{page_block['last_body_page_counted']}`",
            ]
        )
        page_errors = page_block.get("errors", [])
        if isinstance(page_errors, list) and page_errors:
            lines.extend(f"- page_error: {message}" for message in page_errors)
        lines.append("")

    evidence_block = feedback["evidence_validation"]
    if (
        evidence_block["checked"]
        or evidence_block["dangling"]
        or evidence_block["errors"]
        or evidence_block["bundle_present"]
    ):
        lines.append("## Evidence validation")
        lines.extend(
            [
                f"- bundle_path: `{evidence_block['bundle_path']}`",
                f"- bundle_present: `{evidence_block['bundle_present']}`",
                f"- verification_report: `{evidence_block['verification_report_path']}`",
                f"- total_references: `{evidence_block['summary']['total_references']}`",
                f"- resolved_references: `{evidence_block['summary']['resolved_references']}`",
                f"- anchored_references: `{evidence_block['summary']['anchored_references']}`",
                f"- dangling_references: `{evidence_block['summary']['dangling_references']}`",
                f"- manifests_checked: `{evidence_block['summary']['manifests_checked']}`",
            ]
        )
        if evidence_block["dangling"]:
            lines.append("")
            lines.append("### Dangling `@evidence[[...]]` links to fix")
            for item in evidence_block["dangling"]:
                subject = item.get("subject_path") or "<unknown-subject>"
                token = item.get("raw_reference") or item.get("target") or "<unknown-token>"
                detail = "; ".join(item.get("messages") or []) or "dangling link"
                lines.append(f"- `{subject}` {token} -> {detail}")
        if evidence_block["errors"]:
            lines.append("")
            lines.append("### Evidence hard errors")
            lines.extend(f"- {message}" for message in evidence_block["errors"])
        lines.append("")

    audit_block = feedback["agentic_evidence_audit"]
    if audit_block["checked"] or audit_block["findings"]:
        lines.append("## Agentic evidence audit")
        lines.extend(
            [
                f"- checked: `{audit_block['checked']}`",
                f"- verdict: `{audit_block['verdict']}`",
                f"- finding_count: `{audit_block['finding_count']}`",
            ]
        )
        if audit_block.get("summary"):
            lines.append(f"- summary: {audit_block['summary']}")
        if audit_block["findings"]:
            lines.append("")
            lines.append("### Audit findings to fix")
            for item in audit_block["findings"]:
                lines.append(f"- `{item.get('id')}` ({item.get('section')}): {item.get('issue')}")
                if item.get("claim"):
                    lines.append(f"  - claim: {item['claim']}")
                if item.get("evidence"):
                    lines.append(f"  - evidence: {item['evidence']}")
                if item.get("fix"):
                    lines.append(f"  - requested_fix: {item['fix']}")
        lines.append("")

    if auxiliary_checks:
        lines.append("## Auxiliary checks")
        for check_name, payload in auxiliary_checks.items():
            if not isinstance(payload, dict):
                lines.append(f"- {check_name}: `{payload}`")
                continue
            lines.append(f"- {check_name}.ok: `{payload.get('ok')}`")
            for message in payload.get("hard_errors", []) or []:
                lines.append(f"- {check_name}.hard_error: {message}")
            for message in payload.get("soft_warnings", []) or []:
                lines.append(f"- {check_name}.soft_warning: {message}")
        lines.append("")

    soft_warnings = feedback.get("soft_warnings", [])
    if isinstance(soft_warnings, list) and soft_warnings:
        lines.append("## Soft warnings")
        lines.extend(f"- {message}" for message in soft_warnings)
        lines.append("")

    return _write_feedback_files(
        feedback=feedback,
        feedback_json_path=feedback_json_path,
        feedback_md_path=feedback_md_path,
        markdown_lines=lines,
    )


def write_stage_contract_feedback(
    *,
    output_dir: Path,
    feedback_json_path: Path,
    feedback_md_path: Path,
    stage_name: str,
    prompt_path: Path,
    log_path: Path,
    expected_outputs: list[str],
    hard_errors: list[str],
) -> dict[str, Any]:
    """Persist actionable feedback for a stage that wrote an invalid artifact contract."""
    output_dir = output_dir.expanduser().resolve()
    prompt_path = prompt_path.expanduser().resolve()
    log_path = log_path.expanduser().resolve()
    feedback_json_path = feedback_json_path.expanduser().resolve()
    feedback_md_path = feedback_md_path.expanduser().resolve()

    expected_output_records: list[dict[str, Any]] = []
    for raw_path in expected_outputs:
        resolved = _resolve_output_path(output_dir, raw_path).expanduser().resolve()
        exists = resolved.exists()
        size = resolved.stat().st_size if exists and resolved.is_file() else None
        expected_output_records.append(
            {
                "path": str(resolved),
                "source": raw_path,
                "exists": exists,
                "is_file": resolved.is_file(),
                "non_empty": bool(size and size > 0),
                "size_bytes": size,
            }
        )

    persisted_hard_errors = list(hard_errors)
    persisted_hard_errors.append(f"inspect the stage log at: {log_path}")
    if not prompt_path.is_file():
        persisted_hard_errors.append(f"stage prompt file is missing: {prompt_path}")

    log_tail = _read_log_tail(log_path)
    feedback: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_dir": str(output_dir),
        "ok": False,
        "failure_kind": "stage_contract",
        "stage_name": stage_name,
        "prompt_path": str(prompt_path),
        "log_path": str(log_path),
        "hard_errors": persisted_hard_errors,
        "soft_warnings": [],
        "expected_outputs": expected_output_records,
        "log_tail": log_tail,
    }

    lines = [
        "# Paper stage contract feedback",
        "",
        f"- generated_at: `{feedback['generated_at']}`",
        f"- paper_dir: `{feedback['paper_dir']}`",
        f"- ok: `{feedback['ok']}`",
        f"- failure_kind: `{feedback['failure_kind']}`",
        f"- stage_name: `{stage_name}`",
        f"- prompt_path: `{prompt_path}`",
        f"- log_path: `{log_path}`",
        "",
        "## Hard errors",
    ]
    lines.extend(f"- {message}" for message in persisted_hard_errors)
    lines.append("")

    if expected_output_records:
        lines.append("## Required outputs")
        for record in expected_output_records:
            state = "ok" if record["non_empty"] else "missing_or_empty"
            lines.append(f"- `{record['path']}` — `{state}`")
        lines.append("")

    if log_tail:
        lines.extend(
            [
                "## Log tail",
                "```text",
                *log_tail,
                "```",
                "",
            ]
        )

    return _write_feedback_files(
        feedback=feedback,
        feedback_json_path=feedback_json_path,
        feedback_md_path=feedback_md_path,
        markdown_lines=lines,
    )


def _resolve_output_path(output_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return output_dir / path


def _read_log_tail(log_path: Path, *, max_lines: int = _LOG_TAIL_LINES) -> list[str]:
    if not log_path.is_file():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines <= 0:
        return []
    return lines[-max_lines:]


def write_stage_execution_feedback(
    *,
    output_dir: Path,
    feedback_json_path: Path,
    feedback_md_path: Path,
    stage_name: str,
    stage_return_code: int,
    prompt_path: Path,
    log_path: Path,
    expected_outputs: list[str],
) -> dict[str, Any]:
    """Persist actionable feedback for a paper stage that failed before validation."""
    output_dir = output_dir.expanduser().resolve()
    prompt_path = prompt_path.expanduser().resolve()
    log_path = log_path.expanduser().resolve()
    feedback_json_path = feedback_json_path.expanduser().resolve()
    feedback_md_path = feedback_md_path.expanduser().resolve()

    expected_output_records: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    for raw_path in expected_outputs:
        resolved = _resolve_output_path(output_dir, raw_path).expanduser().resolve()
        exists = resolved.exists()
        size = resolved.stat().st_size if exists and resolved.is_file() else None
        record = {
            "path": str(resolved),
            "source": raw_path,
            "exists": exists,
            "is_file": resolved.is_file(),
            "non_empty": bool(size and size > 0),
            "size_bytes": size,
        }
        expected_output_records.append(record)
        if not record["non_empty"]:
            missing_outputs.append(str(resolved))

    hard_errors: list[str] = []
    if stage_return_code == 70:
        hard_errors.append(f"stage '{stage_name}' produced an empty log: {log_path}")
    elif stage_return_code == 71:
        if missing_outputs:
            hard_errors.append(
                f"stage '{stage_name}' did not leave all required outputs non-empty: {', '.join(missing_outputs)}"
            )
        else:
            hard_errors.append(
                f"stage '{stage_name}' reported missing required outputs, but none were detected after the fact"
            )
    else:
        hard_errors.append(f"stage '{stage_name}' exited with code {stage_return_code}")

    hard_errors.append(f"inspect the stage log at: {log_path}")
    if not prompt_path.is_file():
        hard_errors.append(f"stage prompt file is missing: {prompt_path}")

    log_tail = _read_log_tail(log_path)
    feedback: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_dir": str(output_dir),
        "ok": False,
        "failure_kind": "stage_execution",
        "stage_name": stage_name,
        "stage_return_code": stage_return_code,
        "prompt_path": str(prompt_path),
        "log_path": str(log_path),
        "hard_errors": hard_errors,
        "soft_warnings": [],
        "expected_outputs": expected_output_records,
        "log_tail": log_tail,
    }
    lines = [
        "# Paper stage execution feedback",
        "",
        f"- generated_at: `{feedback['generated_at']}`",
        f"- paper_dir: `{feedback['paper_dir']}`",
        f"- ok: `{feedback['ok']}`",
        f"- failure_kind: `{feedback['failure_kind']}`",
        f"- stage_name: `{stage_name}`",
        f"- stage_return_code: `{stage_return_code}`",
        f"- prompt_path: `{prompt_path}`",
        f"- log_path: `{log_path}`",
        "",
        "## Hard errors",
    ]
    lines.extend(f"- {message}" for message in hard_errors)
    lines.append("")

    if expected_output_records:
        lines.append("## Required outputs")
        for record in expected_output_records:
            state = "ok" if record["non_empty"] else "missing_or_empty"
            lines.append(f"- `{record['path']}` — `{state}`")
        lines.append("")

    if log_tail:
        lines.extend(
            [
                "## Log tail",
                "```text",
                *log_tail,
                "```",
                "",
            ]
        )

    return _write_feedback_files(
        feedback=feedback,
        feedback_json_path=feedback_json_path,
        feedback_md_path=feedback_md_path,
        markdown_lines=lines,
    )
