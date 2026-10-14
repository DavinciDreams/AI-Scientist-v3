"""Validated review summaries for pipeline events and terminal output."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Literal, Mapping

from .pinned_files import absolute_path, open_directory_tree_no_follow, read_regular_at

ReviewDecision = Literal["Accept", "Reject"]
_REVIEW_MAX_BYTES = 4 * 1024 * 1024


def parse_review_summary(
    payload: Mapping[str, object],
) -> tuple[int | float | None, ReviewDecision | None]:
    raw_score = payload.get("Overall")
    overall_score: int | float | None = None
    if isinstance(raw_score, int) and not isinstance(raw_score, bool):
        overall_score = raw_score if 1 <= raw_score <= 10 else None
    elif isinstance(raw_score, float) and math.isfinite(raw_score):
        overall_score = raw_score if 1 <= raw_score <= 10 else None
    raw_decision = payload.get("Decision")
    decision: ReviewDecision | None = None
    if raw_decision == "Accept":
        decision = "Accept"
    elif raw_decision == "Reject":
        decision = "Reject"
    return overall_score, decision


def _read_review_summary(path: Path | None) -> tuple[int | float | None, ReviewDecision | None]:
    if path is None:
        return None, None
    try:
        path = absolute_path(path)
        directory_fd = open_directory_tree_no_follow(path.parent)
        try:
            content = read_regular_at(directory_fd, path.name, max_bytes=_REVIEW_MAX_BYTES)
        finally:
            os.close(directory_fd)
        payload = json.loads(content)
        if isinstance(payload, dict):
            return parse_review_summary(payload)
    except (OSError, ValueError, UnicodeError, RecursionError, RuntimeError):
        pass
    return None, None


def revision_review_fields(
    before_review_path: Path | None,
    after_review_path: Path | None,
) -> dict[str, int | float | str | None]:
    """Return optional diagnostic fields without making review display a gate."""
    before_score, before_decision = _read_review_summary(before_review_path)
    after_score, after_decision = _read_review_summary(after_review_path)
    return {
        "before_overall_score": before_score,
        "after_overall_score": after_score,
        "before_decision": before_decision,
        "after_decision": after_decision,
        "overall_score_delta": (
            after_score - before_score
            if before_score is not None and after_score is not None
            else None
        ),
    }


def _score_text(score: int | float | None) -> str:
    return f"{score:g}/10" if score is not None else "unavailable"


def format_review_result(fields: Mapping[str, object]) -> str:
    score, decision = parse_review_summary(
        {"Overall": fields.get("overall_score"), "Decision": fields.get("decision")}
    )
    decision_text = decision.upper() if decision else "decision unavailable"
    return f"Review result · score {_score_text(score)} · {decision_text}"


def format_revision_review(fields: Mapping[str, object]) -> str:
    before_score, before_decision = parse_review_summary(
        {"Overall": fields.get("before_overall_score"), "Decision": fields.get("before_decision")}
    )
    after_score, after_decision = parse_review_summary(
        {"Overall": fields.get("after_overall_score"), "Decision": fields.get("after_decision")}
    )
    delta = (
        f" ({after_score - before_score:+g})"
        if before_score is not None and after_score is not None
        else ""
    )
    before_text = before_decision.upper() if before_decision else "unavailable"
    after_text = after_decision.upper() if after_decision else "unavailable"
    return (
        f"Revision review · overall {_score_text(before_score)} → {_score_text(after_score)}{delta}"
        f" · decision {before_text} → {after_text}"
    )
