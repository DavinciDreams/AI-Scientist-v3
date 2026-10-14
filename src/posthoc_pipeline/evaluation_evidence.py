"""Bounded discovery of evaluator-authored observations in variant workspaces."""

from itertools import islice
from pathlib import Path
import re

from src.investigation.evaluation_evidence import EVIDENCE_FILENAME, load_evaluation_evidence

_MAX_DIRECTORY_ENTRIES = 512
_MAX_PROGRAMS = 512


def _directories(root: Path) -> list[Path]:
    if root.is_symlink():
        raise ValueError(f"Evaluation evidence directory contains symlink: {root}")
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError(f"Evaluation evidence directory is invalid: {root}")
    entries = list(islice(root.iterdir(), _MAX_DIRECTORY_ENTRIES + 1))
    if len(entries) > _MAX_DIRECTORY_ENTRIES:
        raise ValueError("Evaluation evidence directory entries exceed safety limit")
    return sorted(entries)


def variant_evaluation_evidence(source_root: Path) -> list[Path]:
    """Find only the fixed evidence filename in non-symlinked program directories."""
    candidates: list[Path] = []
    programs = 0
    for variant in _directories(source_root / "variants"):
        if variant.is_symlink():
            raise ValueError(f"Evaluation evidence variant contains symlink: {variant}")
        if not variant.is_dir():
            continue
        for program in _directories(variant / "task_bundle"):
            if not re.fullmatch(r"program_\d+", program.name):
                continue
            if program.is_symlink():
                raise ValueError(f"Evaluation evidence program contains symlink: {program}")
            if not program.is_dir():
                continue
            programs += 1
            if programs > _MAX_PROGRAMS:
                raise ValueError("Evaluation evidence programs exceed safety limit")
            if load_evaluation_evidence(program) is not None:
                candidates.append(program / EVIDENCE_FILENAME)
    return candidates
