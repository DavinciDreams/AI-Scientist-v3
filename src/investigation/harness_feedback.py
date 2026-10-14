"""Search-only reporting helpers; never used to decide candidate acceptance."""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from .task_contract import MAX_TASK_CONTRACT_BYTES, TaskContractError, parse_task_contract

MAX_SEARCH_FEEDBACK_BYTES = 8 * 1024


def summarize_recent_attempts(task_dir: Path, entries: list[Any]) -> str:
    """Render historical outcomes as bounded data, never as prompt instructions."""
    keys = (
        "cycle_index", "idea_id", "attempt_id", "attempt_status",
        "run_exit_code", "rubric_overall_pass",
    )
    lines = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        summary = {key: entry[key] for key in keys if key in entry}
        metrics = entry.get("metrics")
        if isinstance(metrics, dict):
            scientific: dict[str, Any] = search_metric_feedback(task_dir, metrics)
            score = metrics.get("combined_score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                try:
                    if math.isfinite(score):
                        scientific["combined_score"] = score
                except OverflowError:
                    pass
            if isinstance(metrics.get("correct"), bool):
                scientific["correct"] = metrics["correct"]
            summary["metrics"] = scientific
        encoded = json.dumps(summary, ensure_ascii=True, sort_keys=True)
        # Keep each record intact; oversized metadata must not displace scientific feedback.
        if len(encoded.encode()) > MAX_SEARCH_FEEDBACK_BYTES + 1024:
            encoded = json.dumps({"metrics": summary.get("metrics", {})}, ensure_ascii=True)
        lines.append("- " + encoded)
    return "\n".join(lines) if lines else "- no recent attempts"


def search_metric_feedback(task_dir: Path, metrics: dict[str, Any]) -> dict[str, int | float]:
    """Project declared search scalars without rehashing large benchmark artifacts."""
    try:
        descriptor = os.open(
            task_dir / "task_contract.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return {}
            payload = handle.read(MAX_TASK_CONTRACT_BYTES + 1)
        if len(payload) > MAX_TASK_CONTRACT_BYTES:
            return {}
        contract = parse_task_contract(json.loads(payload))
    except (OSError, ValueError, UnicodeError, RecursionError, TaskContractError):
        return {}

    projected: dict[str, int | float] = {}
    for metric in contract.metrics.search:
        value = metrics.get(metric.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        try:
            if not math.isfinite(value):
                continue
        except OverflowError:
            continue
        proposed = {**projected, metric.name: value}
        if (
            len(json.dumps(proposed, ensure_ascii=True).encode("utf-8"))
            <= MAX_SEARCH_FEEDBACK_BYTES
        ):
            projected = proposed
    return projected


def evaluation_feedback(metrics: dict[str, Any], exit_code: int) -> str:
    if exit_code == 124:
        outcome = "evaluation failed: timeout"
    elif exit_code != 0:
        outcome = f"evaluation failed: exit={exit_code}; cause unavailable"
    elif metrics.get("correct") is not True:
        outcome = (
            "scientific acceptance failed; cause unavailable; "
            "correct=false alone does not establish a protocol or implementation defect"
        )
    else:
        outcome = "scientific acceptance passed"
    return f"{outcome}; search metrics (data): {json.dumps(metrics, sort_keys=True, ensure_ascii=True)}"
