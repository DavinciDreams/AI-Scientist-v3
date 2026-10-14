"""Retire verified scratch only after successful reusable-container sessions."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.scripts.lib import container_temp


@dataclass(frozen=True)
class ContainerScratch:
    root: Path
    name: str
    leaf: Path

    def cleanup_removed_container(self) -> None:
        try:
            if not container_temp.cleanup(self.root, self.name, self.leaf):
                logging.warning("Retaining session scratch: container removal unconfirmed")
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
            logging.warning("Retaining session scratch: cleanup could not be verified")


def capture_container_scratch(name: str | None) -> ContainerScratch | None:
    root_value = os.environ.get("AGENTIC_CONTAINER_TEMP_ROOT")
    if not name or not root_value:
        return None
    root = Path(root_value)
    try:
        details = container_temp.inspect_container(name)
        leaf = container_temp.verify(root, name, details or {})
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        logging.warning("Retaining session scratch: container identity could not be verified")
        return None
    return ContainerScratch(root, name, leaf)
