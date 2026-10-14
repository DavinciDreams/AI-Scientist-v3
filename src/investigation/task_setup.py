"""Run an optional task-level setup.py before the main orchestration loop."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_task_setup(
    task_dir: Path,
    extra_task_args: dict[str, Any],
    timeout_seconds: int = 120,
) -> bool:
    """Run ``setup.py`` in *task_dir* if it exists, passing *extra_task_args* as JSON.

    Returns ``True`` on success (or when no setup.py is present).
    Returns ``False`` on failure or timeout.
    """
    setup_script = task_dir / "setup.py"
    if not setup_script.is_file():
        logger.debug("No setup.py in %s — skipping task setup", task_dir)
        return True

    args_json = json.dumps(extra_task_args)
    cmd = [sys.executable, str(setup_script), "--args", args_json]
    logger.info("Running task setup: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(task_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        logger.error("Task setup timed out after %ds in %s", timeout_seconds, task_dir)
        return False
    except OSError as exc:
        logger.error("Task setup failed to launch in %s: %s", task_dir, exc)
        return False

    if result.stdout.strip():
        logger.info("task setup stdout: %s", result.stdout.strip())
    if result.stderr.strip():
        logger.warning("task setup stderr: %s", result.stderr.strip())

    if result.returncode != 0:
        logger.error(
            "Task setup exited with code %d in %s", result.returncode, task_dir
        )
        return False

    logger.info("Task setup completed successfully in %s", task_dir)
    return True
