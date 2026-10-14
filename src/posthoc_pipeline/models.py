"""Data models for the post-hoc pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any


_ANALYSIS_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_OUTPUT_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
_OUTPUT_SUFFIXES = frozenset({".csv", ".json", ".md", ".pdf", ".png", ".svg", ".tex", ".txt"})
_IMAGE_OUTPUT_TOKENS = (
    "boxplot",
    "chart",
    "curve",
    "dendrogram",
    "figure",
    "heatmap",
    "histogram",
    "plot",
    "scatter",
    "violin",
)


def is_valid_analysis_id(value: str) -> bool:
    """Return whether an analysis ID is safe as one directory component."""
    return bool(_ANALYSIS_ID_PATTERN.fullmatch(value))


def normalize_string_list(value: Any) -> list[str]:
    """Normalize list-like fields without exploding strings into characters."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def normalize_expected_outputs(value: Any) -> list[str]:
    """Convert LLM prose output descriptions into deterministic safe paths."""
    normalized: list[str] = []
    for index, item in enumerate(normalize_string_list(value), start=1):
        path = Path(item)
        if (
            not path.is_absolute()
            and ".." not in path.parts
            and path.as_posix() == item
            and len(item) <= 240
            and all(len(part) <= 120 for part in path.parts)
            and _OUTPUT_PATH_PATTERN.fullmatch(item)
            and path.suffix.lower() in _OUTPUT_SUFFIXES
        ):
            normalized.append(item)
            continue
        lower_item = item.lower()
        if any(token in lower_item for token in _IMAGE_OUTPUT_TOKENS):
            suffix = ".png"
        elif "table" in lower_item or "result" in lower_item:
            suffix = ".csv"
        else:
            suffix = ".md"
        slug = re.sub(r"[^a-z0-9]+", "_", lower_item).strip("_")[:96]
        normalized.append(f"outputs/{index:02d}_{slug or 'artifact'}{suffix}")
    return normalized


def normalize_implementation_plan(value: Any) -> list[str]:
    """Normalize implementation plans emitted as strings, lists, or char arrays."""

    if value is None:
        return []
    if isinstance(value, str):
        return _split_implementation_plan_text(value)
    if isinstance(value, list):
        raw_items = [str(item) for item in value if str(item)]
        if _looks_like_character_plan(raw_items):
            return _split_implementation_plan_text("".join(raw_items))
        return [item.strip() for item in raw_items if item.strip()]
    return _split_implementation_plan_text(str(value))


def _looks_like_character_plan(items: list[str]) -> bool:
    if len(items) < 12:
        return False
    single_char_items = sum(1 for item in items if len(item) == 1)
    return single_char_items / len(items) >= 0.85


def _split_implementation_plan_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    line_steps: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        if line:
            line_steps.append(line)
    if len(line_steps) > 1:
        return line_steps

    compact = re.sub(r"\s+", " ", text)
    numbered_steps = [
        re.sub(r"^\d+[.)]\s+", "", part).strip()
        for part in re.split(r"\s+(?=\d+[.)]\s)", compact)
    ]
    numbered_steps = [step for step in numbered_steps if step]
    if len(numbered_steps) > 1:
        return numbered_steps

    return [compact]


@dataclass
class AnalysisCandidate:
    """A proposed post-hoc analysis candidate."""

    analysis_id: str
    title: str
    family: str
    hypothesis: str
    relevance_reason: str
    feasibility_reason: str
    required_artifacts: list[str]
    requires_retrain: bool
    implementation_plan: list[str]
    expected_claim: str
    expected_outputs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.analysis_id,
            "title": self.title,
            "family": self.family,
            "hypothesis": self.hypothesis,
            "relevance_reason": self.relevance_reason,
            "feasibility_reason": self.feasibility_reason,
            "required_artifacts": self.required_artifacts,
            "requires_retrain": self.requires_retrain,
            "implementation_plan": self.implementation_plan,
            "expected_claim": self.expected_claim,
            "expected_outputs": self.expected_outputs,
        }


@dataclass
class AnalysisSelection:
    """Selection result for post-hoc analysis candidates."""

    accepted: list[AnalysisCandidate]
    rejected: list[tuple[AnalysisCandidate, str]]


@dataclass
class AnalysisSpec:
    """Specification for a materialized analysis workspace."""

    analysis_id: str
    title: str
    family: str
    analysis_dir: Path
    hypothesis: str = ""
    relevance_reason: str = ""
    feasibility_reason: str = ""
    required_artifacts: list[str] = field(default_factory=list)
    requires_retrain: bool = False
    implementation_plan: list[str] = field(default_factory=list)
    expected_claim: str = ""
    expected_outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "title": self.title,
            "family": self.family,
            "analysis_dir": str(self.analysis_dir),
            "hypothesis": self.hypothesis,
            "relevance_reason": self.relevance_reason,
            "feasibility_reason": self.feasibility_reason,
            "required_artifacts": self.required_artifacts,
            "requires_retrain": self.requires_retrain,
            "implementation_plan": self.implementation_plan,
            "expected_claim": self.expected_claim,
            "expected_outputs": self.expected_outputs,
        }


@dataclass
class AttemptInspection:
    """Inspection result for a single analysis attempt directory."""

    attempt_dir: Path
    attempt_id: str
    overall_pass: bool
    rubric_error: str | None
    required_outputs_present: bool
    missing_outputs: list[str]
    manifest_valid: bool
    manifest_error: str | None
    guardrail_passed: bool | None = None
    checkpoint_validated: bool = False
    manifest: dict[str, Any] | None = None
    rubric_status: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_dir": str(self.attempt_dir),
            "attempt_id": self.attempt_id,
            "overall_pass": self.overall_pass,
            "rubric_error": self.rubric_error,
            "required_outputs_present": self.required_outputs_present,
            "missing_outputs": self.missing_outputs,
            "manifest_valid": self.manifest_valid,
            "manifest_error": self.manifest_error,
            "guardrail_passed": self.guardrail_passed,
            "checkpoint_validated": self.checkpoint_validated,
            "manifest": self.manifest,
            "rubric_status": self.rubric_status,
        }


@dataclass
class AnalysisRunResult:
    """Result of executing a single analysis workspace."""

    analysis_id: str
    status: str
    attempt_count: int
    exit_code: int
    selected_attempt: str | None
    rubric_passed: bool
    required_outputs_present: bool
    guardrail_passed: bool | None
    output: str
    produced_files: dict[str, bool]
    attempts: list[AttemptInspection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "exit_code": self.exit_code,
            "selected_attempt": self.selected_attempt,
            "rubric_passed": self.rubric_passed,
            "required_outputs_present": self.required_outputs_present,
            "guardrail_passed": self.guardrail_passed,
            "output": self.output,
            "produced_files": self.produced_files,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }
