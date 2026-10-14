#!/bin/bash

paper_diagnostics() {
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m src.orchestrator.paper_diagnostics "$1" "$OUTPUT_DIR" "${@:2}"
}

archive_paper_validation() {
  paper_diagnostics archive validation "$1" \
    validation_report.json validation_feedback.json validation_feedback.md
}

archive_paper_evidence_audit() {
  paper_diagnostics archive evidence_audit "$1" \
    paper_evidence_audit.json paper_evidence_audit.md \
    paper_evidence_audit_feedback.json paper_evidence_audit_feedback.md \
    paper_evidence_audit_stage_feedback.json paper_evidence_audit_stage_feedback.md \
    paper_evidence_audit_input_sha256.json paper_evidence_audit_raw.txt
}
