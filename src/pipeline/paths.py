"""Single source of truth for every path the orchestrator touches.

``derive_paths`` is pure (no I/O). Each derived path mirrors the corresponding
stage's own default so that stage N+1 reads exactly where stage N wrote:

* run envelope = ``<results_root>/<task>/<agent>/<run_id>``
* investigation = ``<run envelope>/investigation``
* ablations = ``<run envelope>/ablations``
* final evaluation = ``<run envelope>/investigation/final_evaluation``
* posthoc = ``<run envelope>/posthoc``
* paper = ``<run envelope>/paper``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig, _generate_run_id
from .investigation_layout import investigation_workspace


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved paths shared across the pipeline stages."""

    task_name: str
    run_id: str
    source_task_dir: Path
    task_dir: Path
    investigation_results_dir: Path
    ablations_output_dir: Path
    ablations_aggregated_task_dir: Path
    final_output_dir: Path
    posthoc_output_dir: Path
    paper_task_dir: Path
    paper_output_dir: Path

    @property
    def run_dir(self) -> Path:
        return self.investigation_results_dir

    @property
    def investigation_dir(self) -> Path:
        return self.investigation_workspace_dir

    @property
    def task_drafting_dir(self) -> Path:
        return self.run_dir / "task_drafting"

    @property
    def task_drafting_output_dir(self) -> Path:
        return self.task_drafting_dir

    @property
    def ablations_dir(self) -> Path:
        return self.ablations_output_dir

    @property
    def final_dir(self) -> Path:
        return self.final_output_dir

    @property
    def posthoc_dir(self) -> Path:
        return self.posthoc_output_dir

    @property
    def paper_dir(self) -> Path:
        return self.paper_output_dir

    @property
    def revision_dir(self) -> Path:
        return self.run_dir / "revision"

    @property
    def investigation_workspace_dir(self) -> Path:
        """Agent-writable investigation state inside the checkpointed run envelope."""
        return investigation_workspace(self.investigation_results_dir)


def derive_paths(cfg: PipelineConfig, *, task_name: str | None = None) -> PipelinePaths:
    """Derive every inter-stage path from the config. Pure; performs no I/O."""
    source_task_dir = Path(cfg.task.dir).expanduser()
    task_name = task_name or source_task_dir.name
    run_id = cfg.run_id or _generate_run_id()
    agent = cfg.agent.name

    if cfg.paths.run_dir:
        investigation = Path(cfg.paths.run_dir).expanduser()
    else:
        investigation = Path(cfg.paths.results_root).expanduser() / task_name / agent / run_id

    if cfg.paths.ablations_dir:
        ablations = Path(cfg.paths.ablations_dir).expanduser()
    else:
        ablations = investigation / "ablations"

    if cfg.paths.paper_dir:
        paper_out = Path(cfg.paths.paper_dir).expanduser()
    else:
        paper_out = investigation / "paper"

    if cfg.paths.posthoc_dir:
        posthoc = Path(cfg.paths.posthoc_dir).expanduser()
    else:
        posthoc = investigation / "posthoc"

    final_output = (
        Path(cfg.paths.final_evaluation_dir).expanduser() if cfg.paths.final_evaluation_dir else investigation / "final_evaluation"
    )

    return PipelinePaths(
        task_name=task_name,
        run_id=run_id,
        source_task_dir=source_task_dir,
        task_dir=investigation / "task",
        investigation_results_dir=investigation,
        ablations_output_dir=ablations,
        ablations_aggregated_task_dir=ablations / "aggregated" / task_name,
        final_output_dir=final_output,
        posthoc_output_dir=posthoc,
        paper_task_dir=investigation / "task",
        paper_output_dir=paper_out,
    )
