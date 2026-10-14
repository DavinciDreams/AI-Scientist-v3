"""Keep raw dataset downloads outside task snapshots and agent mounts."""
from __future__ import annotations

import os
from pathlib import Path


def require_no_legacy_data_cache(task_dir: Path) -> None:
    if os.path.lexists(task_dir / '.data-cache'):
        raise ValueError(
            'legacy .data-cache must be moved outside task trees before preparation or launch; '
            'start fresh results if a previous task projection contained raw data'
        )


def external_data_cache(cache: Path, *, task_dirs: tuple[Path, ...]) -> Path:
    cache = cache.expanduser().resolve()
    for task_dir in task_dirs:
        task_dir = task_dir.resolve()
        require_no_legacy_data_cache(task_dir)
        if cache.is_relative_to(task_dir) or task_dir.is_relative_to(cache):
            raise ValueError('raw data cache must be outside all task trees')
    for parent in (cache, *cache.parents):
        if any(os.path.lexists(parent / marker)
               for marker in ('task.md', 'task_contract.json', '.git')):
            raise ValueError('raw data cache must be outside task trees and Git checkouts')
    return cache
