"""Terminal wordmark rendering for Agentic Scientist."""

from __future__ import annotations

from .terminal_ui import terminal_width

_WORDMARK = (
    " █████╗  ██████╗     ███████╗ ██████╗██╗███████╗███╗   ██╗████████╗██╗███████╗████████╗",
    "██╔══██╗██╔════╝     ██╔════╝██╔════╝██║██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝╚══██╔══╝",
    "███████║██║  ███╗    ███████╗██║     ██║█████╗  ██╔██╗ ██║   ██║   ██║███████╗   ██║",
    "██╔══██║██║   ██║    ╚════██║██║     ██║██╔══╝  ██║╚██╗██║   ██║   ██║╚════██║   ██║",
    "██║  ██║╚██████╔╝    ███████║╚██████╗██║███████╗██║ ╚████║   ██║   ██║███████║   ██║",
    "╚═╝  ╚═╝ ╚═════╝     ╚══════╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝   ╚═╝",
)
_COMPACT_WORDMARK = (
    " ███   ███  ████   ███  ███",
    "█   █ █     █     █      █ ",
    "█████ █ ███  ███  █      █ ",
    "█   █ █   █     █ █      █ ",
    "█   █  ███  ████   ███  ███",
)
_MINIMAL_WORDMARK = "AGSCI"
_GRADIENT_START = (255, 100, 50)
_GRADIENT_END = (100, 200, 255)
_RESET = "\033[0m"


def _ansi_256(rgb: tuple[int, int, int]) -> int:
    red, green, blue = (int(channel * 5 / 255) for channel in rgb)
    return 16 + 36 * red + 6 * green + blue


def _gradient_color(index: int, count: int) -> int:
    fraction = index / max(1, count - 1)
    channels = tuple(
        int(start + fraction * (end - start))
        for start, end in zip(_GRADIENT_START, _GRADIENT_END, strict=True)
    )
    return _ansi_256((channels[0], channels[1], channels[2]))


def render_logo(*, colors: bool, columns: int | None = None) -> list[str]:
    """Return the AG Scientist terminal wordmark."""
    full_width = max(terminal_width(line) for line in _WORDMARK)
    compact_width = max(terminal_width(line) for line in _COMPACT_WORDMARK)
    if columns is not None and columns < compact_width:
        if not colors:
            return [_MINIMAL_WORDMARK]
        rendered = "".join(
            f"\033[38;5;{_gradient_color(index, len(_MINIMAL_WORDMARK))}m{character}"
            for index, character in enumerate(_MINIMAL_WORDMARK)
        )
        return [rendered + _RESET]
    wordmark = _COMPACT_WORDMARK if columns is not None and columns < full_width else _WORDMARK
    if not colors:
        return list(wordmark)
    return [
        f"\033[38;5;{_gradient_color(index, len(wordmark))}m{line}{_RESET}"
        for index, line in enumerate(wordmark)
    ]
