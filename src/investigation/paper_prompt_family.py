"""Resolve strategy-selected paper prompt families into executable prompt settings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .skill_scope import PAPER_WRITER_SKILLS, paper_writer_skill


@dataclass(frozen=True)
class PaperPromptFamilyConfig:
    """Canonical configuration for one prompt family."""

    prompt_family: str
    prompt_relpath: Path
    writer_skill: str


@dataclass(frozen=True)
class FinalPaperStage:
    """Resolved final-stage prompt configuration."""

    prompt_family: str
    prompt_path: Path
    writer_skill: str
    selected_skills: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize into a JSON-friendly payload."""
        return {
            "prompt_family": self.prompt_family,
            "prompt_path": str(self.prompt_path),
            "writer_skill": self.writer_skill,
            "selected_skills": list(self.selected_skills),
        }


PROMPT_FAMILY_CONFIGS: dict[str, PaperPromptFamilyConfig] = {
    "paper": PaperPromptFamilyConfig(
        prompt_family="paper",
        prompt_relpath=Path("src") / "prompts" / "paper.md",
        writer_skill="write-paper",
    ),
    "paper_detailed_grouped": PaperPromptFamilyConfig(
        prompt_family="paper_detailed_grouped",
        prompt_relpath=Path("src") / "prompts" / "paper_detailed_grouped.md",
        writer_skill="write-paper",
    ),
    "paper_standalone": PaperPromptFamilyConfig(
        prompt_family="paper_standalone",
        prompt_relpath=Path("src") / "prompts" / "paper_standalone.md",
        writer_skill="write-paper-standalone",
    ),
}


def infer_prompt_family_from_path(prompt_path: Path, *, repo_root: Path) -> str | None:
    """Infer a prompt family from a canonical repo prompt path."""
    repo_root = repo_root.expanduser().resolve()
    resolved_prompt = prompt_path.expanduser().resolve()
    for family, config in PROMPT_FAMILY_CONFIGS.items():
        if resolved_prompt == (repo_root / config.prompt_relpath).resolve():
            return family
    return None


def _resolve_family_config(selected_prompt_family: str) -> PaperPromptFamilyConfig:
    config = PROMPT_FAMILY_CONFIGS.get(selected_prompt_family)
    if config is None:
        available = ", ".join(sorted(PROMPT_FAMILY_CONFIGS))
        raise ValueError(
            f"Unknown selected_prompt_family '{selected_prompt_family}'. Available families: {available}"
        )
    return config


def _retarget_paper_skills(
    current_selected_skills: Sequence[str],
    *,
    writer_skill: str,
) -> tuple[str, ...]:
    """Replace the current writer skill while preserving the rest of the selection."""
    current_writer = paper_writer_skill(current_selected_skills)
    if current_writer == writer_skill:
        return tuple(current_selected_skills)

    retargeted: list[str] = []
    inserted = False
    for skill in current_selected_skills:
        if skill in PAPER_WRITER_SKILLS:
            if not inserted:
                retargeted.append(writer_skill)
                inserted = True
            continue
        retargeted.append(skill)

    if not inserted:
        retargeted.append(writer_skill)
    return tuple(retargeted)


def resolve_final_paper_stage(
    *,
    repo_root: Path,
    current_prompt_path: Path,
    current_selected_skills: Sequence[str],
    selected_prompt_family: str,
    resolution_path: Path | None = None,
) -> FinalPaperStage:
    """Resolve the final paper prompt and skill selection from strategy output."""
    repo_root = repo_root.expanduser().resolve()
    current_prompt_path = current_prompt_path.expanduser().resolve()
    selected_prompt_family = selected_prompt_family.strip()

    config = _resolve_family_config(selected_prompt_family)
    current_writer_skill = paper_writer_skill(current_selected_skills)
    current_prompt_family = infer_prompt_family_from_path(current_prompt_path, repo_root=repo_root)
    prompt_path = (
        current_prompt_path
        if (
            current_prompt_family == selected_prompt_family
            or (current_prompt_family is None and current_writer_skill == config.writer_skill)
        )
        else (repo_root / config.prompt_relpath).resolve()
    )
    selected_skills = _retarget_paper_skills(
        current_selected_skills,
        writer_skill=config.writer_skill,
    )
    resolved = FinalPaperStage(
        prompt_family=config.prompt_family,
        prompt_path=prompt_path,
        writer_skill=config.writer_skill,
        selected_skills=selected_skills,
    )
    if resolution_path is not None:
        resolution_path = resolution_path.expanduser().resolve()
        resolution_path.parent.mkdir(parents=True, exist_ok=True)
        resolution_path.write_text(
            json.dumps(resolved.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return resolved
