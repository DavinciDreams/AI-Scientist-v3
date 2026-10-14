"""Pydantic models for run inspector API responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

NodeType = Literal["worker", "idea", "program_eval"]


class RunSummary(BaseModel):
    run_id: str
    path: str
    task_name: str
    agent: str
    containerized: bool | None = None
    parallel: bool | None = None
    worker_count: int | None = None
    idea_count: int | None = None
    program_eval_count: int | None = None
    best_combined_score: float | None = None


class GraphNode(BaseModel):
    id: str
    type: NodeType
    label: str
    data: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    kind: str


class RunGraph(BaseModel):
    run: RunSummary
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class IdeaPerformance(BaseModel):
    worker_index: int
    worker_name: str
    idea_node_id: str
    idea_id: str
    idea_title: str
    idea_dir_name: str
    idea_path: str
    idea_order_index: int | None = None
    program_count: int
    scored_program_count: int
    correct_program_count: int
    incorrect_program_count: int
    best_combined_score: float | None = None
    best_program_label: str | None = None
    best_correct: bool | None = None


class ProgramPerformance(BaseModel):
    worker_index: int
    worker_name: str
    idea_node_id: str
    idea_id: str
    idea_title: str
    idea_dir_name: str
    idea_path: str
    idea_order_index: int | None = None
    program_label: str
    program_idx: int
    sequence_idx: int
    attempt_name: str | None = None
    program_code_path: str | None = None
    timestamp_unix: float | None = None
    combined_score: float | None = None
    correct: bool | None = None


class RunPerformance(BaseModel):
    run_id: str
    ideas: list[IdeaPerformance] = Field(default_factory=list)
    programs: list[ProgramPerformance] = Field(default_factory=list)
    max_idea_combined_score: float | None = None
    max_program_combined_score: float | None = None


class ArtifactRef(BaseModel):
    name: str
    path: str
    kind: Literal["json", "code", "log", "data", "text", "other"]
    exists: bool
    size_bytes: int | None = None
    agentlens_url: str | None = None


class NodeDetail(BaseModel):
    run_id: str
    node_id: str
    node: GraphNode
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    agentlens_url: str | None = None


class FileContent(BaseModel):
    path: str
    size_bytes: int
    truncated: bool
    text: str


# ---------------------------------------------------------------------------
# Idea timeline models
# ---------------------------------------------------------------------------


class IdeaAttemptSummary(BaseModel):
    attempt_id: str | None = None
    attempt_number: int | None = None
    attempt_status: str | None = None
    rubric_overall_pass: bool = False
    best_updated: bool = False
    metrics: dict[str, Any] | None = None
    timestamp: str | None = None


class IdeaTimelineEntry(BaseModel):
    node_id: str | None = None
    idea_id: str
    idea_path: str | None = None
    cycle_index: int | None = None
    title: str = ""
    timestamp: str | None = None
    hypothesis: str | None = None
    implementation_plan: list[str] | None = None
    notes_for_agent: str | None = None
    target_rubrics: list[dict[str, Any]] = Field(default_factory=list)
    num_attempts: int = 0
    best_score: float | None = None
    any_pass: bool = False
    rubric_summary: dict[str, int] | None = None
    attempts: list[IdeaAttemptSummary] = Field(default_factory=list)
    worker_name: str | None = None
    prune_chapter: int | None = None


class PruneEvent(BaseModel):
    chapter: int
    cycle_index: int
    timestamp: str | None = None


class IdeaTimeline(BaseModel):
    run_id: str
    ideas: list[IdeaTimelineEntry] = Field(default_factory=list)
    prune_events: list[PruneEvent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tree / diff models
# ---------------------------------------------------------------------------


class ProgramLocation(BaseModel):
    path: str


class AncestorEntry(BaseModel):
    idea_id: str
    title: str = ""
    cycle_index: int | None = None
    worker_name: str | None = None
    program_info: ProgramLocation | None = None
    programs: list["SiblingProgramEntry"] = Field(default_factory=list)


class SiblingProgramEntry(BaseModel):
    """A program from one of the current idea's attempts (for intra-cycle diff)."""

    label: str
    attempt_name: str | None = None
    program_info: ProgramLocation | None = None


class IdeaAncestors(BaseModel):
    idea_id: str
    target_program: ProgramLocation | None = None
    ancestors: list[AncestorEntry] = Field(default_factory=list)
    sibling_programs: list[SiblingProgramEntry] = Field(default_factory=list)


class DiffLine(BaseModel):
    type: Literal["context", "addition", "removal", "hunk_header"]
    content: str
    from_line: int | None = None
    to_line: int | None = None


class DiffStats(BaseModel):
    additions: int = 0
    deletions: int = 0
    hunks: int = 0


class ProgramDiff(BaseModel):
    lines: list[DiffLine] = Field(default_factory=list)
    stats: DiffStats = Field(default_factory=DiffStats)


class ProgramDirDiffFile(BaseModel):
    relpath: str
    status: Literal["added", "removed", "modified", "unchanged"]
    from_path: str
    to_path: str
    from_exists: bool = False
    to_exists: bool = False
    is_binary: bool = False
    from_size_bytes: int | None = None
    to_size_bytes: int | None = None


class ProgramDirDiffStats(BaseModel):
    added: int = 0
    removed: int = 0
    modified: int = 0
    unchanged: int = 0


class ProgramDirDiff(BaseModel):
    files: list[ProgramDirDiffFile] = Field(default_factory=list)
    stats: ProgramDirDiffStats = Field(default_factory=ProgramDirDiffStats)
