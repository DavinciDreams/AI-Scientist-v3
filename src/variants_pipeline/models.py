"""Data models for the variants pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VariantCandidate:
    """A proposed baseline or ablation candidate from the generation stage."""

    variant_id: str
    title: str
    kind: str  # "baseline" | "ablation"
    base_program: str  # "best" | "reference"
    hypothesis: str
    relevance_reason: str
    feasibility_reason: str
    target_components: list[str]
    implementation_plan: list[str]
    expected_effect: str
    description_body: str = ""
    reference_dir: Optional[Path] = None  # folder-based baseline source directory


@dataclass
class VariantSelection:
    """Result of the selection stage: accepted candidates + rejections."""

    accepted: list[VariantCandidate]
    rejected: list[tuple[VariantCandidate, str]]  # (candidate, rejection_reason)
    controls: list[str]  # e.g. ["control_best", "baseline_reference"]


@dataclass
class VariantSpec:
    """Specification for a materialized variant workspace."""

    variant_id: str
    title: str
    kind: str  # "control" | "baseline" | "ablation"
    base_program: str
    variant_dir: Path
    hypothesis: str = ""
    relevance_reason: str = ""
    feasibility_reason: str = ""
    target_components: list[str] = field(default_factory=list)
    implementation_plan: list[str] = field(default_factory=list)
    expected_effect: str = ""
    description_body: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "variant_id": self.variant_id,
            "title": self.title,
            "kind": self.kind,
            "base_program": self.base_program,
            "variant_dir": str(self.variant_dir),
            "hypothesis": self.hypothesis,
            "relevance_reason": self.relevance_reason,
            "feasibility_reason": self.feasibility_reason,
            "target_components": self.target_components,
            "implementation_plan": self.implementation_plan,
            "expected_effect": self.expected_effect,
            "description_body": self.description_body,
        }


@dataclass
class VariantRunResult:
    """Result of executing a single variant agent."""

    variant_id: str
    status: str  # "completed" | "failed" | "timeout"
    eval_count: int
    rubric_passed: bool
    exit_code: int
    output: str
    analysis_files: dict[str, bool]  # filename -> exists
