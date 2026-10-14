"""
Output capture utilities for stdout/stderr with real-time file streaming.
"""

import os
import sys
from typing import Optional, Tuple, TextIO


class OutputCapture:
    """Captures stdout/stderr, streaming to files in real-time with optional console tee."""

    def __init__(self, results_dir: str, tee: bool = True):
        """
        Args:
            results_dir: Directory to write stdout.log and stderr.log
            tee: If True, also write to original console streams
        """
        self.results_dir = results_dir
        self.tee = tee
        self._stdout_orig: Optional[TextIO] = None
        self._stderr_orig: Optional[TextIO] = None
        self._stdout_file: Optional[TextIO] = None
        self._stderr_file: Optional[TextIO] = None
        self.stdout_path = os.path.join(results_dir, "stdout.log")
        self.stderr_path = os.path.join(results_dir, "stderr.log")

    def __enter__(self):
        os.makedirs(self.results_dir, exist_ok=True)

        # Open log files with line buffering for real-time writes
        self._stdout_file = open(self.stdout_path, "w", buffering=1)
        self._stderr_file = open(self.stderr_path, "w", buffering=1)

        # Save original streams
        self._stdout_orig = sys.stdout
        self._stderr_orig = sys.stderr

        # Replace with tee streams
        sys.stdout = _TeeStream(self._stdout_orig, self._stdout_file, self.tee)
        sys.stderr = _TeeStream(self._stderr_orig, self._stderr_file, self.tee)

        return self

    def __exit__(self, *args):
        # Restore original streams
        sys.stdout = self._stdout_orig
        sys.stderr = self._stderr_orig

        # Close log files
        if self._stdout_file:
            self._stdout_file.close()
        if self._stderr_file:
            self._stderr_file.close()

    @property
    def stdout(self) -> str:
        """Read captured stdout from file."""
        if os.path.exists(self.stdout_path):
            with open(self.stdout_path, "r") as f:
                return f.read()
        return ""

    @property
    def stderr(self) -> str:
        """Read captured stderr from file."""
        if os.path.exists(self.stderr_path):
            with open(self.stderr_path, "r") as f:
                return f.read()
        return ""


class _TeeStream:
    """Stream that writes to both console and file in real-time."""

    def __init__(self, console: Optional[TextIO], file: TextIO, tee: bool):
        self.console = console
        self.file = file
        self.tee = tee

    def write(self, data):
        # Always write to file immediately
        self.file.write(data)
        self.file.flush()

        # Optionally tee to console
        if self.tee and self.console:
            self.console.write(data)
            self.console.flush()

    def flush(self):
        self.file.flush()
        if self.tee and self.console:
            self.console.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def save_logs(results_dir: str, stdout: str, stderr: str) -> Tuple[str, str]:
    """Save captured stdout/stderr to log files (legacy helper)."""
    stdout_path = os.path.join(results_dir, "stdout.log")
    stderr_path = os.path.join(results_dir, "stderr.log")

    with open(stdout_path, "w") as f:
        f.write(stdout)

    with open(stderr_path, "w") as f:
        f.write(stderr)

    return stdout_path, stderr_path
