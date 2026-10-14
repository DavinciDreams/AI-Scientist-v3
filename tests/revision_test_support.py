from __future__ import annotations

import json
from pathlib import Path

from src.orchestrator.paper_evidence_audit import audit_input_sha256
from src.pipeline.config import PipelineConfig
from src.pipeline.paths import PipelinePaths


def revision_config(tmp_path: Path) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.paths.run_dir = str(tmp_path / "run")
    cfg.paths.paper_dir = str(tmp_path / "run" / "paper")
    cfg.run_id = "test-run"
    return cfg


def review_payload() -> dict[str, object]:
    return {
        "Summary": "A useful paper.",
        "Strengths": ["Clear method."],
        "Weaknesses": ["The limitations are too vague."],
        "Questions": ["Which settings are covered?"],
        "Limitations": ["Only one benchmark family."],
        "Overall": 5,
        "Decision": "Reject",
    }


def refresh_evidence_audit(workspace: Path) -> dict[str, str]:
    hashes = audit_input_sha256(workspace)
    audit = {
        "verdict": "pass",
        "summary": "All claims supported.",
        "finding_count": 0,
        "findings": [],
        "input_sha256": hashes,
    }
    (workspace / "paper_evidence_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (workspace / "validation_report.json").write_text(
        json.dumps({"ok": True, "agentic_evidence_audit": {"checked": True, **audit}}),
        encoding="utf-8",
    )
    return hashes


def write_workspace(workspace: Path, *, title: str) -> None:
    workspace.mkdir(parents=True)
    (workspace / "paper.tex").write_text(f"\\title{{{title}}}\n", encoding="utf-8")
    (workspace / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    (workspace / "claims_grounding.json").write_text("{}\n", encoding="utf-8")
    (workspace / "references.bib").write_text("% refs\n", encoding="utf-8")
    (workspace / "references_provenance.json").write_text("[]\n", encoding="utf-8")
    (workspace / "paper_data.json").write_text(
        json.dumps(
            {
                "all_programs": [{"attempt_id": "attempt_01", "program_id": "program_1"}],
                "task_names": ["primary"],
                "grounding_sources": {"result.top_program_01": {"value": 1.0}},
                "ablation_studies": [{"study_id": "main", "variants": []}],
                "posthoc_analyses": [{"analysis_id": "check"}],
                "final_evaluation": {"metrics": {"score": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    (workspace / "method_scope_audit.json").write_text(
        json.dumps(
            {
                "manuscript_focus": {
                    "shared_method_summary": "Shared method.",
                    "scope_strategy": "Scope all claims.",
                },
                "shared_method_elements": ["Shared method."],
                "scoped_method_elements": [],
                "claims_requiring_scoping": [],
            }
        ),
        encoding="utf-8",
    )
    (workspace / "method_scope_audit.md").write_text("# Method scope\n", encoding="utf-8")
    (workspace / "paper_prompt.md").write_text("original prompt\n", encoding="utf-8")
    (workspace / "latest_manifest.json").write_text(
        json.dumps(
            {
                "paper_dir": str(workspace.resolve()),
                "paper_tex": str((workspace / "paper.tex").resolve()),
                "paper_pdf": str((workspace / "paper.pdf").resolve()),
                "claims_grounding": str((workspace / "claims_grounding.json").resolve()),
                "references_bib": str((workspace / "references.bib").resolve()),
                "references_provenance": str((workspace / "references_provenance.json").resolve()),
                "validation_report": str((workspace / "validation_report.json").resolve()),
            }
        ),
        encoding="utf-8",
    )
    refresh_evidence_audit(workspace)


def write_original_snapshot(paths: PipelinePaths, *, title: str = "Before") -> Path:
    snapshot = paths.paper_dir / "revisions" / "revision_01" / "original"
    write_workspace(snapshot, title=title)
    return snapshot
