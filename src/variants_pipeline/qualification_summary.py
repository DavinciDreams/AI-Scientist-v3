"""Publish rubric qualification separately from evaluator correctness."""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from src.investigation.safe_staging import atomic_write_bytes, read_text

from .rubric_contract import build_rubric_contract, canonicalize_variant_rubric_status


def summary_qualification(csv_path: Path) -> dict[str, str]:
    """Reconcile a structured evaluation with its canonical rubric, or mark unknown."""
    program = csv_path.parent.parent
    unknown = {"overall_pass": ""}
    match = re.fullmatch(r"program_(\d+)", program.name)
    if match is None:
        return unknown
    try:
        rows = list(csv.DictReader(io.StringIO(read_text(program, "analysis/run_summary.csv"))))
        status = json.loads(read_text(program, "rubric_status.json"))
        correctness = json.loads(read_text(program, "correct.json"))
        metrics = json.loads(read_text(program, "metrics.json"))
        if len(rows) != 1 or not all(isinstance(v, dict) for v in (status, correctness, metrics)):
            return unknown
        row = rows[0]
        variant_id = row["variant_id"]
        correct = correctness.get("correct")
        if not isinstance(correct, bool):
            return unknown
        attempt_id = f"{variant_id}_attempt_{int(match[1]):02d}"
        if (
            status.get("variant_id") != variant_id
            or status.get("idea_id") != variant_id
            or status.get("attempt_id") != attempt_id
        ):
            return unknown
        contract = build_rubric_contract(
            variant_id, row.get("variant_kind", ""), row.get("base_program", "best"),
        )
        required = [r["id"] for r in contract["rubrics"]]
        canonical, _ = canonicalize_variant_rubric_status(
            status, variant_id=variant_id, idea_id=variant_id, attempt_id=attempt_id,
            required_rubric_ids=required, combined_score=metrics.get("combined_score"),
            correct=correct,
        )
        if (
            canonical is None
            or status.get("overall_pass") is not canonical["overall_pass"]
            or status.get("metrics") != canonical["metrics"]
        ):
            return unknown
        return {"overall_pass": str(canonical["overall_pass"])}
    except (OSError, ValueError, UnicodeError, KeyError, csv.Error):
        return unknown


def publish_summary_qualification(program: Path) -> None:
    """Update a saved summary after the host has finalized rubric qualification."""
    relative = "analysis/run_summary.csv"
    try:
        reader = csv.DictReader(io.StringIO(read_text(program, relative)))
    except FileNotFoundError:
        return
    rows = list(reader)
    if not reader.fieldnames or len(rows) != 1:
        return
    row = rows[0]
    row.update(summary_qualification(program / relative))
    columns = list(dict.fromkeys([*reader.fieldnames, "overall_pass"]))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerow(row)
    atomic_write_bytes(program, relative, stream.getvalue().encode())
