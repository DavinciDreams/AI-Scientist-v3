"""Manifest path rebasing for revision publication."""

from __future__ import annotations

import os
from pathlib import Path


def rebase_manifest(value: object, *, old_root: Path, new_root: Path) -> object:
    """Rebase absolute paths rooted in a revision workspace to its publication root."""
    old_root = Path(os.path.abspath(old_root)).resolve()
    new_root = Path(os.path.abspath(new_root)).resolve()
    if isinstance(value, dict):
        return {
            key: rebase_manifest(item, old_root=old_root, new_root=new_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rebase_manifest(item, old_root=old_root, new_root=new_root) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                raw_relative = path.relative_to(old_root)
            except ValueError:
                raw_relative = None
            if raw_relative is not None and ".." in raw_relative.parts:
                raise ValueError(f"manifest path escapes revision root: {value}")
            lexical_path = Path(os.path.abspath(path))
            try:
                lexical_path.relative_to(old_root)
            except ValueError:
                return value
            resolved_path = lexical_path.resolve()
            try:
                relative = resolved_path.relative_to(old_root)
            except ValueError as exc:
                raise ValueError(f"manifest path escapes revision root: {value}") from exc
            rebased = (new_root / relative).resolve()
            if not rebased.is_relative_to(new_root):
                raise ValueError(f"manifest path escapes publication root: {value}")
            return str(rebased)
    return value
