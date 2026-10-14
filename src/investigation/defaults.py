"""Path and default-value helpers for wrapper execution."""

from __future__ import annotations

from pathlib import Path

from .files import resolve_path


def _repo_root() -> Path:
    # defaults.py lives at <repo>/src/investigation/defaults.py
    """Handle repo root logic for this module."""
    return Path(__file__).resolve().parents[2]


def default_instruction_path(task_dir: Path) -> Path:
    """Return default instruction path."""
    repo_prompt = resolve_path(_repo_root() / "src" / "prompts" / "general.md")
    if repo_prompt.is_file():
        return repo_prompt
    return task_dir / "goal.md"


def default_idea_instruction_path() -> Path:
    """Return default idea instruction path."""
    return resolve_path(_repo_root() / "src" / "prompts" / "idea.md")


def default_draft_instruction_path() -> Path:
    """Return default draft instruction path."""
    return resolve_path(_repo_root() / "src" / "prompts" / "draft_task.md")


def default_paper_instruction_path() -> Path:
    """Return default paper instruction path."""
    return resolve_path(_repo_root() / "src" / "prompts" / "paper.md")


def default_results_path(task_dir: Path, agent: str) -> Path:
    """Return default results path."""
    task_name = task_dir.name.strip() or "task"
    return resolve_path(Path("results") / task_name / agent)


def normalize_agent_results_dir(results_dir: Path, agent: str) -> Path:
    """Ensure an agent segment exists once for shared parallel artifacts.

    Checks all path components (not just the leaf) so that paths like
    ``results/<task>/<agent>/<run_id>`` are recognised as already containing
    the agent and are not appended a second time.
    """
    if agent in results_dir.parts:
        return results_dir
    return results_dir / agent


def split_cycle_budget(total_cycles: int, workers: int) -> list[int]:
    """Split cycle budget."""
    base = total_cycles // workers
    remainder = total_cycles % workers
    return [base + (1 if idx < remainder else 0) for idx in range(workers)]
