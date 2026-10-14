"""Step-scoped skill selection for agent execution scripts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Sequence

SKILLS_BY_STEP: dict[str, tuple[str, ...]] = {
    "idea": (
        "create-idea",
        "memory-log",
    ),
    "run": (
        "edit-program",
        "evaluate-program",
        "verify-rubrics",
        "memory-log",
    ),
    "run_harness_eval": (
        "edit-program",
        "memory-log",
    ),
    "variant_edit": (
        "edit-program",
        "memory-log",
    ),
    "variant_verify": (
        "verify-rubrics",
        "memory-log",
    ),
    "task": (
        "draft-task",
        "memory-log",
    ),
    "paper": (
        "retrieve-references",
        "generate-concept-figure",
        "scientific-evidence",
        "write-paper",
        "validate-paper",
        "memory-log",
    ),
}

PAPER_SKILLS_BY_PROFILE: dict[str, tuple[str, ...]] = {
    "default": SKILLS_BY_STEP["paper"],
    "standalone": (
        "retrieve-references",
        "generate-concept-figure",
        "scientific-evidence",
        "write-paper-standalone",
        "validate-paper",
        "memory-log",
    ),
}

PAPER_WRITER_SKILLS = frozenset({"write-paper", "write-paper-standalone"})
#: The writer skill whose ``templates/`` tree every paper writer skill shares.
PAPER_TEMPLATES_SKILL = "write-paper"


def skills_for_step(step: str, *, paper_profile: str = "default") -> tuple[str, ...]:
    """Return configured skill names for a workflow step."""
    if step == "paper":
        selected = PAPER_SKILLS_BY_PROFILE.get(paper_profile)
        if selected is None:
            available_profiles = ", ".join(sorted(PAPER_SKILLS_BY_PROFILE))
            raise ValueError(
                f"Unknown paper profile '{paper_profile}'. Available profiles: {available_profiles}"
            )
        return selected
    selected = SKILLS_BY_STEP.get(step)
    if selected is None:
        available = ", ".join(sorted(SKILLS_BY_STEP))
        raise ValueError(f"Unknown step '{step}'. Available steps: {available}")
    return selected


def parse_skill_csv(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated skill selection string."""
    selected: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        skill = part.strip()
        if not skill:
            continue
        if skill in seen:
            continue
        seen.add(skill)
        selected.append(skill)
    if not selected:
        raise ValueError("Selected skill set must not be empty")
    return tuple(selected)


def resolve_skills_for_step(
    step: str,
    override_csv: str | None = None,
    *,
    paper_profile: str = "default",
) -> tuple[str, ...]:
    """Return the default step skill set or a parsed CLI override."""
    if override_csv is None or not override_csv.strip():
        return skills_for_step(step, paper_profile=paper_profile)
    return parse_skill_csv(override_csv)


def paper_writer_skill(selected: Sequence[str]) -> str:
    """Return the single paper-writing skill used by the paper step."""
    writers = [skill for skill in selected if skill in PAPER_WRITER_SKILLS]
    if len(writers) != 1:
        allowed = ", ".join(sorted(PAPER_WRITER_SKILLS))
        raise ValueError(
            "Paper skill selection must include exactly one paper-writing skill "
            f"({allowed}); got: {', '.join(selected) or '(empty)'}"
        )
    return writers[0]


def validate_skill_set(skills_src_dir: Path, selected: Sequence[str]) -> None:
    """Validate selected skills exist on disk."""
    if not selected:
        raise ValueError("Selected skill set must not be empty")

    missing = [skill for skill in selected if not (skills_src_dir / skill).is_dir()]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"Missing configured skills for selection in {skills_src_dir}: {missing_text}"
        )


_STAGING_IGNORED_DIRS = frozenset({".git"})


def _ignore_staging_dirs(directory: str, entries: list[str]) -> set[str]:
    return {name for name in entries if name in _STAGING_IGNORED_DIRS}


def _copy_replacing(src: str, dst: str, *, follow_symlinks: bool = True) -> str:
    """Copy ``src`` over ``dst`` even when a previous staging left ``dst`` read-only."""
    try:
        os.unlink(dst)
    except FileNotFoundError:
        pass
    return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)


def paper_templates_dir(skills_src: Path) -> Path:
    """Return the checked-in paper templates shared by every writer skill."""
    return skills_src / PAPER_TEMPLATES_SKILL / "templates"


def stage_paper_templates(skills_src: Path, staged_writer_skill_dir: Path) -> None:
    """Copy the shared paper templates into a staged writer skill.

    Only ``write-paper`` carries the templates in the repository; the other
    writer skills receive the same tree at staging time so their ``SKILL.md``
    can keep referring to ``templates/<format>/``.
    """
    stage_skill_tree(paper_templates_dir(skills_src), staged_writer_skill_dir / "templates")


def stage_skill_tree(src_dir: Path, dst_dir: Path) -> None:
    """Copy a skill into an agent workspace, tolerating an earlier staging of the same skill.

    Nested ``.git`` directories (for example a vendored tool checkout inside a skill)
    are skipped: agents never need them, and git stores pack
    files read-only, so re-staging over a previous copy would fail with EACCES. Stale
    ``.git`` copies left by earlier stagings are removed from the destination.
    """
    shutil.copytree(
        src_dir,
        dst_dir,
        dirs_exist_ok=True,
        ignore=_ignore_staging_dirs,
        copy_function=_copy_replacing,
    )
    for root, dirs, _files in os.walk(dst_dir):
        stale = [name for name in dirs if name in _STAGING_IGNORED_DIRS]
        for name in stale:
            shutil.rmtree(Path(root) / name)
            dirs.remove(name)
