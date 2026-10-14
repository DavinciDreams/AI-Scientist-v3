"""Helpers for reconciling and merging idea-local memory files."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from src.orchestrator import state as state_store


def reconcile_legacy_idea_local_memory(
    *,
    results_dir: Path,
    cycle_index: int,
    idea_id: str,
    canonical_idea_dir_name: str,
    log_event_fn: Callable[..., None],
) -> None:
    """Move/merge local memory from fallback idea-id dir into canonical title dir."""
    fallback_idea_dir_name = state_store.idea_dir_name(
        cycle_index=cycle_index,
        idea_title=None,
        idea_id=idea_id,
    )
    if fallback_idea_dir_name == canonical_idea_dir_name:
        return

    fallback_dir = state_store.idea_dir(results_dir, fallback_idea_dir_name)
    canonical_dir = state_store.idea_dir(results_dir, canonical_idea_dir_name)
    canonical_memory = canonical_dir / "memory.md"

    def _merge_local_memory_from(source_dir: Path) -> tuple[bool, bool, bool]:
        """Handle merge local memory from logic for this module."""
        source_memory = source_dir / "memory.md"
        moved_memory = False
        merged_memory = False
        removed_orphan_dir = False

        if source_memory.is_file():
            canonical_dir.mkdir(parents=True, exist_ok=True)
            if canonical_memory.exists():
                source_text = source_memory.read_text(encoding="utf-8")
                existing_text = canonical_memory.read_text(encoding="utf-8")
                if source_text.strip() and source_text not in existing_text:
                    separator = "" if existing_text.endswith("\n") else "\n"
                    with canonical_memory.open("a", encoding="utf-8") as handle:
                        handle.write(f"{separator}\n{source_text}")
                    merged_memory = True
                source_memory.unlink(missing_ok=True)
            else:
                shutil.move(str(source_memory), str(canonical_memory))
                moved_memory = True

        if source_dir.is_dir() and not any(source_dir.iterdir()):
            source_dir.rmdir()
            removed_orphan_dir = True
        return moved_memory, merged_memory, removed_orphan_dir

    moved_memory, merged_memory, removed_orphan_dir = _merge_local_memory_from(fallback_dir)
    if moved_memory or merged_memory or removed_orphan_dir:
        log_event_fn(
            "idea_memory_reconciled",
            cycle=cycle_index,
            idea_id=idea_id,
            from_dir=fallback_dir,
            to_dir=canonical_dir,
            moved_memory=moved_memory,
            merged_memory=merged_memory,
            removed_orphan_dir=removed_orphan_dir,
        )

    ideas_root = results_dir / "ideas"
    cycle_prefix = f"{cycle_index:02d}_"
    if not ideas_root.is_dir():
        return

    for candidate_dir in sorted(ideas_root.iterdir()):
        if not candidate_dir.is_dir():
            continue
        if candidate_dir.name in {canonical_idea_dir_name, fallback_idea_dir_name}:
            continue
        if not candidate_dir.name.startswith(cycle_prefix):
            continue

        moved_alt, merged_alt, removed_alt = _merge_local_memory_from(candidate_dir)
        if moved_alt or merged_alt or removed_alt:
            log_event_fn(
                "idea_memory_reconciled",
                cycle=cycle_index,
                idea_id=idea_id,
                from_dir=candidate_dir,
                to_dir=canonical_dir,
                moved_memory=moved_alt,
                merged_memory=merged_alt,
                removed_orphan_dir=removed_alt,
            )
