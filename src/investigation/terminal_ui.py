"""Dependency-free semantic terminal styling for structured pipeline events."""

from __future__ import annotations

import os
import threading
import unicodedata
from typing import Any, Mapping, TextIO

RESET = "\033[0m"
STYLES: Mapping[str, str] = {
    "phase": "\033[1;36m",
    "running": "\033[33m",
    "success": "\033[32m",
    "failure": "\033[31m",
    "warning": "\033[33m",
    "detail": "\033[2m",
    "score": "\033[36m",
}
SPINNER_FRAMES = tuple("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
ERROR_SUFFIXES = ("_error", "_failed", "_warning")
PROGRESS_LABELS = {
    "idea_generation": "Generating idea…",
    "attempt": "Running experiment…",
    "task_draft": "Drafting task…",
    "paper_generation": "Generating paper…",
}
TERMINAL_WRITE_LOCK = threading.RLock()


def colors_enabled(stream: TextIO) -> bool:
    """Return whether semantic ANSI styles should be written to the stream."""
    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    force_color = os.environ.get("FORCE_COLOR")
    if force_color is not None:
        return force_color != "0"
    isatty = getattr(stream, "isatty", None)
    return callable(isatty) and bool(isatty())


def style(text: str, semantic: str, *, enabled: bool) -> str:
    """Apply one semantic style without changing the underlying text."""
    code = STYLES.get(semantic)
    if not enabled or code is None or not text:
        return text
    return f"{code}{text}{RESET}"


def terminal_text(value: Any, *, limit: int = 240) -> str:
    """Escape terminal control characters in untrusted human-facing fields."""
    rendered: list[str] = []
    rendered_length = 0
    for character in str(value):
        codepoint = ord(character)
        if character == "\n":
            replacement = "\\n"
        elif character == "\r":
            replacement = "\\r"
        elif character == "\t":
            replacement = " "
        elif codepoint < 32 or 127 <= codepoint <= 159:
            replacement = f"\\x{codepoint:02x}"
        elif unicodedata.category(character) in {"Cf", "Cs", "Zl", "Zp"}:
            replacement = (
                f"\\u{codepoint:04x}" if codepoint <= 0xFFFF else f"\\U{codepoint:08x}"
            )
        else:
            replacement = character
        remaining = limit - rendered_length
        if len(replacement) > remaining:
            rendered.append(replacement[:remaining])
            return "".join(rendered) + "…"
        rendered.append(replacement)
        rendered_length += len(replacement)
        if rendered_length == limit:
            return "".join(rendered) + "…"
    return "".join(rendered)


def terminal_width(value: str) -> int:
    """Return the terminal-column width of already-sanitized text."""
    return sum(
        0
        if unicodedata.combining(character)
        else 2
        if unicodedata.east_asian_width(character) in {"F", "W"}
        else 1
        for character in value
    )


def truncate_terminal_text(value: Any, *, columns: int) -> str:
    """Sanitize and truncate text so it occupies at most ``columns`` cells."""
    if columns < 1:
        return ""
    rendered = terminal_text(value)
    if terminal_width(rendered) <= columns:
        return rendered
    if columns == 1:
        return "…"
    kept: list[str] = []
    used = 0
    for character in rendered:
        width = terminal_width(character)
        if used + width > columns - 1:
            break
        kept.append(character)
        used += width
    return "".join(kept) + "…"
