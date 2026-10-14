#!/usr/bin/env python3
"""Validate figure manifest artifacts and basic figure readability budgets."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.orchestrator.paper_step import read_json, validate_figure_manifest
from src.orchestrator.tex_graphics import read_tex_file_bounded


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate figure manifest artifacts.")
    parser.add_argument("--paper-dir", default=".")
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    paper_dir = Path(args.paper_dir).resolve()
    tex_path = paper_dir / "paper.tex"
    paper_data_path = paper_dir / "paper_data.json"
    references_path = paper_dir / "references_provenance.json"

    if not tex_path.exists():
        raise SystemExit(f"Missing required artifact: {tex_path}")

    tex_text = read_tex_file_bounded(tex_path)
    paper_data = read_json(paper_data_path)
    references = read_json(references_path)
    grounding_sources = paper_data.get("grounding_sources")
    if not isinstance(grounding_sources, dict):
        raise SystemExit("paper_data missing grounding_sources object.")
    allowed_source_ids = set(str(key) for key in grounding_sources)
    refs = references.get("references")
    if isinstance(refs, list):
        for row in refs:
            if isinstance(row, dict):
                source_id = row.get("id")
                if isinstance(source_id, str) and source_id.strip():
                    allowed_source_ids.add(source_id)

    report = validate_figure_manifest(
        paper_dir,
        tex_text=tex_text,
        allowed_source_ids=allowed_source_ids,
    )
    if not bool(report.get("checked", False)):
        print("No figures referenced or required; figure validation skipped.")
        return 0

    hard_errors = report.get("hard_errors", [])
    if not isinstance(hard_errors, list):
        raise SystemExit("figure_manifest validation returned invalid hard_errors payload")
    soft_warnings = report.get("soft_warnings", [])
    if not isinstance(soft_warnings, list):
        raise SystemExit("figure_manifest validation returned invalid soft_warnings payload")
    if hard_errors:
        raise SystemExit("Figure validation failed: " + "; ".join(str(error) for error in hard_errors))
    if soft_warnings:
        print("Figure validation warnings: " + "; ".join(str(message) for message in soft_warnings))

    print(
        f"Validated {int(report.get('figure_count', 0))} figures with manifest and readability budgets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
