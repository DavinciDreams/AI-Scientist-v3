"""Lightweight parsing helpers for LaTeX graphics dependencies."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from src.orchestrator.paper_supplement import (
    mask_false_conditionals,
    mask_tex_verbatim,
    strip_tex_comments,
)

FIGURE_INCLUDE_RE = re.compile(r"\\includegraphics\*?\s*(?:\[[^\]]*\]\s*)?\{([^}]+)\}")
DYNAMIC_GRAPHICS_PATH_RE = re.compile(r"\\csname\s*graphicspath\s*\\endcsname")
GRAPHICS_EXTENSION_STATE_RE = re.compile(
    r"\\(?:DeclareGraphicsExtensions|Gin@extensions)(?![A-Za-z@])"
    r"|\\csname\s*(?:DeclareGraphicsExtensions|Gin@extensions)\s*\\endcsname"
)
TEX_INPUT_RE = re.compile(
    r"\\(?:input|include)(?![A-Za-z@])\s*"
    r"(?:\{(?P<braced>[^{}]+)\}|(?P<unbraced>[^\s%{}]+))"
)
TEX_LOCAL_SUPPORT_RE = re.compile(
    r"\\(?P<command>usepackage|RequirePackage|documentclass|LoadClass)"
    r"(?![A-Za-z@])(?:\s*\[[^\]]*\])?\s*\{(?P<targets>[^{}]+)\}"
)
GRAPHICS_SUFFIXES = (
    ".eps",
    ".jb2",
    ".jbig2",
    ".jpeg",
    ".jpg",
    ".mps",
    ".pdf",
    ".png",
    ".svg",
)
PDFLATEX_GRAPHICS_SUFFIXES = (
    ".pdf",
    ".png",
    ".jpg",
    ".mps",
    ".jpeg",
    ".jbig2",
    ".jb2",
)
MAX_GRAPHICS_TEX_CHARS = 16 * 1024 * 1024
MAX_GRAPHICS_REFERENCE_COUNT = 256
MAX_GRAPHICS_SEARCH_PATH_COUNT = 32
MAX_GRAPHICS_PATH_CHARS = 1024
MAX_GRAPHICS_CANDIDATE_COUNT = 16 * 1024
MAX_GRAPHICS_CANDIDATE_CHARS = 16 * 1024 * 1024


@dataclass(frozen=True)
class GraphicsReferenceResolution:
    """One static LaTeX graphics reference and its rendered workspace path."""

    raw_path: str
    resolved_path: str | None
    explicitly_png: bool


def read_tex_file_bounded(path: Path) -> str:
    """Read a regular TeX file without following its leaf symlink or exceeding the limit."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Unable to open regular TeX file: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"TeX input is not a regular file: {path}")
        if metadata.st_size > MAX_GRAPHICS_TEX_CHARS:
            raise ValueError("Paper TeX exceeds the graphics resolution limit")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            data = handle.read(MAX_GRAPHICS_TEX_CHARS + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(data) > MAX_GRAPHICS_TEX_CHARS:
        raise ValueError("Paper TeX exceeds the graphics resolution limit")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"TeX input is not valid UTF-8: {path}") from exc


def extract_graphics_paths(tex_text: str) -> list[str]:
    """Extract rendered includegraphics paths from LaTeX."""
    return list(iter_graphics_paths(tex_text))


def iter_graphics_paths(tex_text: str) -> Iterator[str]:
    """Yield rendered includegraphics paths from LaTeX."""
    visible_tex = _visible_tex_commands(tex_text)
    for match in FIGURE_INCLUDE_RE.finditer(visible_tex):
        if path := match.group(1).strip():
            yield path


def iter_graphics_search_paths(tex_text: str) -> Iterator[str]:
    """Yield static directories declared by rendered ``graphicspath`` commands."""
    for declaration in iter_graphics_search_path_declarations(tex_text):
        yield from declaration


def iter_graphics_search_path_declarations(
    tex_text: str,
) -> Iterator[tuple[str, ...]]:
    """Yield static ``graphicspath`` declarations with linear-time parsing."""
    visible_tex = _visible_tex_commands(tex_text)
    if DYNAMIC_GRAPHICS_PATH_RE.search(visible_tex):
        raise ValueError("Dynamic graphicspath command construction")
    command = r"\graphicspath"
    cursor = 0
    while (command_start := visible_tex.find(command, cursor)) >= 0:
        command_end = command_start + len(command)
        next_command = visible_tex.find(command, command_end)
        scan_end = next_command if next_command >= 0 else len(visible_tex)
        cursor = scan_end
        if command_end < len(visible_tex) and (
            visible_tex[command_end].isalpha() or visible_tex[command_end] == "@"
        ):
            continue
        declaration_start = _skip_whitespace(visible_tex, command_end)
        declaration_end = _graphics_path_declaration_end(visible_tex, declaration_start, scan_end)
        if declaration_end is None:
            raise ValueError("Dynamic or malformed graphicspath declaration")
        entries: list[str] = []
        position = declaration_start + 1
        while True:
            position = _skip_whitespace(visible_tex, position)
            if visible_tex[position] == "}":
                break
            entry_end = visible_tex.find("}", position + 1)
            path = visible_tex[position + 1 : entry_end].strip()
            if path:
                entries.append(path)
            position = entry_end + 1
        cursor = declaration_end
        if entries:
            yield tuple(entries)


def _graphics_path_declaration_end(
    tex_text: str,
    start: int,
    scan_end: int,
) -> int | None:
    if start >= scan_end or tex_text[start] != "{":
        return None
    position = start + 1
    entry_count = 0
    while True:
        position = _skip_whitespace(tex_text, position)
        if position >= scan_end:
            return None
        if tex_text[position] == "}":
            return position + 1 if entry_count else None
        if tex_text[position] != "{":
            return None
        entry_end = tex_text.find("}", position + 1, scan_end)
        if entry_end < 0 or "{" in tex_text[position + 1 : entry_end]:
            return None
        entry_count += 1
        position = entry_end + 1


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def iter_tex_import_paths(tex_text: str) -> Iterator[str]:
    """Yield static local input/include targets from LaTeX."""
    visible_tex = _visible_tex_commands(tex_text)
    for match in TEX_INPUT_RE.finditer(visible_tex):
        if path := (match.group("braced") or match.group("unbraced")).strip():
            yield path


def iter_tex_support_paths(tex_text: str) -> Iterator[str]:
    """Yield local package and class dependency candidates from LaTeX."""
    visible_tex = _visible_tex_commands(tex_text)
    for match in TEX_LOCAL_SUPPORT_RE.finditer(visible_tex):
        suffix = ".cls" if "class" in match.group("command").casefold() else ".sty"
        for raw_target in match.group("targets").split(","):
            target = Path(raw_target.strip())
            if not target.name:
                continue
            yield (target if target.suffix else target.with_suffix(suffix)).as_posix()


def _visible_tex_commands(tex_text: str) -> str:
    without_comments = strip_tex_comments(tex_text)
    without_verbatim = mask_tex_verbatim(without_comments)
    return mask_false_conditionals(without_verbatim)


def candidate_graphics_paths(raw_path: str) -> set[str]:
    """Normalize includegraphics references to comparable relative paths."""
    path = Path(raw_path)
    if path.suffix:
        return {path.as_posix()}
    return {(path.with_suffix(extension)).as_posix() for extension in GRAPHICS_SUFFIXES}


def ordered_candidate_graphics_paths(raw_path: str) -> tuple[str, ...]:
    """Return graphics candidates in deterministic TeX extension order."""
    path = Path(raw_path)
    if path.suffix:
        return (path.as_posix(),)
    return tuple(
        (path.with_suffix(extension)).as_posix()
        for extension in PDFLATEX_GRAPHICS_SUFFIXES
    )


def resolve_graphics_references(
    tex_text: str,
    workspace_root: Path,
) -> list[GraphicsReferenceResolution]:
    """Resolve static graphics references using bounded, no-follow TeX search order."""
    if len(tex_text) > MAX_GRAPHICS_TEX_CHARS:
        raise ValueError("Paper TeX exceeds the graphics resolution limit")
    if GRAPHICS_EXTENSION_STATE_RE.search(_visible_tex_commands(tex_text)):
        raise ValueError("Paper declares or mutates a custom graphics extension order")

    search_paths: tuple[Path, ...] | None = None
    for raw_declaration in iter_graphics_search_path_declarations(tex_text):
        if len(raw_declaration) > MAX_GRAPHICS_SEARCH_PATH_COUNT:
            raise ValueError(
                f"Paper declares more than {MAX_GRAPHICS_SEARCH_PATH_COUNT} graphics search paths"
            )
        declaration = tuple(_validated_graphics_search_path(path) for path in raw_declaration)
        if search_paths is not None and declaration != search_paths:
            raise ValueError("Paper declares conflicting graphics search paths")
        search_paths = declaration

    candidate_count = 0
    candidate_chars = 0

    def consume_candidate(candidate: str) -> None:
        nonlocal candidate_count, candidate_chars
        candidate_count += 1
        candidate_chars += len(candidate)
        if (
            candidate_count > MAX_GRAPHICS_CANDIDATE_COUNT
            or candidate_chars > MAX_GRAPHICS_CANDIDATE_CHARS
        ):
            raise ValueError("Paper figure references exceed the graphics expansion limit")

    resolutions: list[GraphicsReferenceResolution] = []
    for reference_count, raw_path in enumerate(iter_graphics_paths(tex_text), start=1):
        if reference_count > MAX_GRAPHICS_REFERENCE_COUNT:
            raise ValueError(
                f"Paper references more than {MAX_GRAPHICS_REFERENCE_COUNT} figure assets"
            )
        if len(raw_path) > MAX_GRAPHICS_PATH_CHARS:
            raise ValueError("Paper figure reference exceeds the graphics path limit")

        resolved_path: str | None = None
        for candidate in ordered_candidate_graphics_paths(raw_path):
            consume_candidate(candidate)
            candidate_path = _validated_figure_candidate(candidate)
            if candidate_path.parts[0] == "figures":
                if _path_entry_exists(workspace_root, candidate_path):
                    _require_regular_file_nofollow(workspace_root, candidate_path)
                    resolved_path = candidate
                    break
                continue
            if not search_paths:
                continue
            if _path_entry_exists(workspace_root, candidate_path):
                raise ValueError(
                    "Unsafe figure reference shadows graphicspath asset in paper root: "
                    f"{candidate!r}"
                )
            for search_path in search_paths:
                resolved_candidate = (search_path / candidate_path).as_posix()
                consume_candidate(resolved_candidate)
                resolved_candidate_path = _validated_figure_candidate(resolved_candidate)
                if not _path_entry_exists(workspace_root, resolved_candidate_path):
                    continue
                _require_regular_file_nofollow(workspace_root, resolved_candidate_path)
                resolved_path = resolved_candidate
                break
            if resolved_path is not None:
                break

        resolutions.append(
            GraphicsReferenceResolution(
                raw_path=raw_path,
                resolved_path=resolved_path,
                explicitly_png=Path(raw_path).suffix.casefold() == ".png",
            )
        )
    return resolutions


def _validated_graphics_search_path(raw_path: str) -> Path:
    if len(raw_path) > MAX_GRAPHICS_PATH_CHARS:
        raise ValueError("Paper graphics search path exceeds the graphics path limit")
    path = Path(raw_path)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "figures"
        or not raw_path.endswith("/")
        or "#" in raw_path
        or any(part in {"", ".", ".."} or "\\" in part for part in path.parts)
    ):
        raise ValueError(f"Unsafe graphicspath in paper.tex: {raw_path!r}")
    return path


def _validated_figure_candidate(candidate: str) -> Path:
    path = Path(candidate)
    if (
        path.is_absolute()
        or not path.parts
        or "#" in candidate
        or any(part in {"", ".", ".."} or "\\" in part for part in path.parts)
        or path.suffix.casefold() not in GRAPHICS_SUFFIXES
    ):
        raise ValueError(f"Unsafe figure reference in paper.tex: {candidate!r}")
    return path


def _path_entry_exists(workspace_root: Path, relative_path: Path) -> bool:
    return os.path.lexists(workspace_root / relative_path)


def _require_regular_file_nofollow(workspace_root: Path, relative_path: Path) -> None:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current_descriptor = os.open(workspace_root, directory_flags)
        descriptors.append(current_descriptor)
        for component in relative_path.parts[:-1]:
            current_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=current_descriptor,
            )
            descriptors.append(current_descriptor)
        file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        file_descriptor = os.open(
            relative_path.parts[-1],
            file_flags,
            dir_fd=current_descriptor,
        )
        try:
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                raise ValueError(
                    f"Unsafe figure asset in paper workspace: {relative_path.as_posix()!r}"
                )
        finally:
            os.close(file_descriptor)
    except OSError as exc:
        raise ValueError(
            f"Unsafe figure asset in paper workspace: {relative_path.as_posix()!r}"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
