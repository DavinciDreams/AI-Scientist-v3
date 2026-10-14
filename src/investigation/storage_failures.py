"""Classify host storage errors without trusting exception text or evaluator output."""

from __future__ import annotations

import errno
import json
import sys

_STORAGE_CATEGORIES = {
    errno.ESTALE: "storage_stale_handle",
    errno.EDQUOT: "storage_quota_exceeded",
    errno.ENOSPC: "storage_space_exhausted",
}


def storage_failure_category(error: BaseException) -> str | None:
    """Inspect typed causes/contexts; rollback wrappers must not hide quota failures."""
    pending = [error]
    seen: set[int] = set()
    while pending and len(seen) < 64:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in _STORAGE_CATEGORIES:
            return _STORAGE_CATEGORIES[current.errno]
        for linked in (current.__context__, current.__cause__):
            if linked is not None:
                pending.append(linked)
    return None


def warn_storage_failure(error: BaseException, *, operation: str) -> None:
    """Last-resort stderr diagnostic; omit paths, exception text, and event payloads."""
    category = storage_failure_category(error)
    if category is None:
        return
    fields = {
        "status": "infrastructure_failure",
        "failure_category": category,
        "operation": operation[:80],
    }
    try:
        print(json.dumps(fields, ensure_ascii=True, sort_keys=True), file=sys.stderr, flush=True)
    except OSError:
        pass  # No recursive persistence attempt when even stderr is unavailable.
