"""Private prompt-file staging for agent subprocesses."""

from __future__ import annotations

import tempfile
from pathlib import Path


def stage_agent_prompt(work_dir: Path, filename: str, prompt: str) -> str:
    """Atomically stage *prompt* mode 0600 and return its relative name."""
    if not filename or Path(filename).name != filename:
        raise ValueError("prompt filename must be a basename")
    prompt_path = work_dir / filename
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=work_dir,
            prefix=f".{filename}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(prompt)
        temporary_path.replace(prompt_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return filename
