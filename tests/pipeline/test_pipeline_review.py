"""Compatibility shim: the review-phase tests now live in ``tests/pipeline/test_final_ai_review.py``.

Kept as a module so other test files can keep importing its helpers.
"""

from __future__ import annotations

from pathlib import Path

from tests.pipeline.test_final_ai_review import _cfg, _no_subprocess, _write_completed_run
from tests.pipeline.test_final_ai_review import _write_published_paper as _write_published_paper_dir

__all__ = ["_cfg", "_no_subprocess", "_write_completed_run", "_write_published_paper"]


def _write_published_paper(bundle: Path, paper_dir: Path) -> None:
    """Original two-argument signature; ``bundle`` is unused, as before."""
    del bundle
    _write_published_paper_dir(paper_dir)
