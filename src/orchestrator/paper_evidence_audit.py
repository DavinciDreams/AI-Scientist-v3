"""Agentic evidence-audit packet building, parsing, and finalization helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from src.orchestrator.paper_final_outcome import final_outcome_summary

from src.orchestrator.paper_supplement import (
    mask_false_conditionals,
    mask_tex_verbatim,
    split_manuscript_and_supplement,
    strip_tex_comments,
)
from src.orchestrator.paper_watermark import strip_managed_block
from src.orchestrator.tex_graphics import (
    GRAPHICS_SUFFIXES,
    candidate_graphics_paths,
    iter_graphics_paths,
    iter_graphics_search_path_declarations,
    iter_tex_import_paths,
    iter_tex_support_paths,
)

_MAX_TEX_CHARS = 30000
_MAX_JSON_CHARS = 14000
_VERDICT_PREFIX = "AUDIT_VERDICT: "
_SUMMARY_PREFIX = "AUDIT_SUMMARY: "
_FINDING_COUNT_PREFIX = "AUDIT_FINDING_COUNT: "
_FINDING_PREFIX = "AUDIT_FINDING: "
_FINDING_KEYS = ("id", "section", "claim", "issue", "evidence", "fix")
_AUDIT_WORKSPACE_ROOT_PREFIX = ".paper_evidence_audit_workspaces-"
_AUDIT_WORKSPACE_PARENT_PREFIX = "agentic-evidence-audit-"
_AUDIT_WORKSPACE_MARKER = ".managed-paper-evidence-audit"
_AUDIT_WORKSPACE_MARKER_CONTENT = "agentic-scientist managed paper evidence audit\n"
_WORKSPACE_TAG_RE = re.compile(r"[A-Za-z0-9_.-]+")
_PROMPT_ARTIFACT_RE = re.compile(r".+_prompt(?:_round_\d+(?:_contract_retry_\d+)?)?\.md$")
AUDIT_INPUT_PATHS = (
    "paper.tex",
    "claims_grounding.json",
    "paper_data.json",
    "references.bib",
    "references_provenance.json",
    "figure_manifest.json",
    "scientific-evidence/verification-report.json",
)
VALIDATION_INPUT_PATHS = AUDIT_INPUT_PATHS + (
    "VENUE.md",
    "paper_authors.tex",
    "paper_strategy.md",
    "paper_strategy.json",
    "method_scope_audit.md",
    "method_scope_audit.json",
)
AUDIT_INPUT_HASH_MANIFEST = ".paper_evidence_audit_input_sha256.json"
_MAX_AUDIT_LOG_LINE_BYTES = 4 * 1024 * 1024
_MAX_AUDIT_CONTRACT_CHARS = 128 * 1024
_MAX_AUDIT_TEX_BYTES = 16 * 1024 * 1024
_MAX_VALIDATION_METADATA_BYTES = 16 * 1024 * 1024
_MAX_VALIDATION_METADATA_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_VALIDATION_BIBLIOGRAPHY_STYLE_COUNT = 16
_MAX_VALIDATION_BIBLIOGRAPHY_STYLE_BYTES = 16 * 1024 * 1024
_MAX_VALIDATION_BIBLIOGRAPHY_STYLE_TARGET_CHARS = 1024
_MAX_AUDIT_TEX_SOURCE_COUNT = 256
_MAX_AUDIT_FIGURE_COUNT = 256
_MAX_AUDIT_GRAPHICS_PATH_COUNT = 32
_MAX_AUDIT_GRAPHICS_PATH_CHARS = 1024
_MAX_AUDIT_FIGURE_REFERENCE_CHARS = 1024
_MAX_AUDIT_FIGURE_CANDIDATE_COUNT = 16 * 1024
_MAX_AUDIT_FIGURE_CANDIDATE_CHARS = 16 * 1024 * 1024
_MAX_AUDIT_FIGURE_BYTES = 512 * 1024 * 1024
_AUDIT_FIGURE_SUFFIXES = set(GRAPHICS_SUFFIXES)
_BIBLIOGRAPHY_STYLE_RE = re.compile(
    r"\\bibliographystyle(?![A-Za-z@])\s*\{(?P<target>[^{}]+)\}"
)
_TEX_CONDITIONAL_TOKEN_RE = re.compile(r"\\(?:if[A-Za-z@]*|else|fi)\b")
_MAX_VALIDATION_TEX_CONDITIONAL_TOKENS = 1024
_VOLATILE_VERIFICATION_FIELDS = {"root_dir", "verified_at"}
_AGENT_SKILL_DIR_PREFIXES = (
    (".claude", "skills"),
    (".codex", "skills"),
    (".cursor", "skills"),
    (".opencode", "skills"),
    (".gemini", "skills"),
    (".pi", "agent", "skills"),
)
_AGENT_NAMESPACE_NAMES = {prefix[0] for prefix in _AGENT_SKILL_DIR_PREFIXES}
_VERIFICATION_SUMMARY_KEYS = (
    "total_references",
    "resolved_references",
    "anchored_references",
    "dangling_references",
    "manifests_checked",
)
_EVIDENCE_SCAN_EXTENSIONS = {
    ".bash", ".cfg", ".conf", ".csv", ".ini", ".js", ".json", ".jsonl",
    ".jsx", ".md", ".py", ".r", ".R", ".rst", ".sh", ".sql", ".tex",
    ".toml", ".ts", ".tsv", ".tsx", ".txt", ".yaml", ".yml", ".zsh",
}
_EVIDENCE_SKIP_DIR_NAMES = {
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "__pycache__", "scientific-evidence",
}
_EVIDENCE_GENERATED_NAMES = {
    "validation_feedback.json", "validation_feedback.md", "validation_report.json",
    "verification-report.json", ".latest_manifest.lock",
}

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AuditFinding(BaseModel):
    """Single evidence/claim mismatch finding."""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyText
    section: NonEmptyText
    claim: NonEmptyText
    issue: NonEmptyText
    evidence: NonEmptyText
    fix: NonEmptyText


class AuditPayload(BaseModel):
    """Structured result parsed from the agent's raw audit response."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["pass", "fail"]
    summary: NonEmptyText
    findings: list[AuditFinding] = Field(default_factory=list, max_length=20)


class AuditContractError(ValueError):
    """Raised when an audit agent response violates the line-oriented contract."""


def validation_input_sha256(paper_dir: Path) -> dict[str, str]:
    """Hash canonical validation inputs without unrelated workspace churn."""
    return _paper_input_sha256(paper_dir, VALIDATION_INPUT_PATHS)


def _paper_input_sha256(
    paper_dir: Path,
    input_paths: tuple[str, ...],
) -> dict[str, str]:
    workspace_root = paper_dir.expanduser().resolve()
    hashes: dict[str, str] = {}
    remaining_metadata_bytes = _MAX_VALIDATION_METADATA_TOTAL_BYTES
    for relative_path in input_paths:
        relative = Path(relative_path)
        per_file_limit = (
            _MAX_AUDIT_TEX_BYTES
            if relative_path == "paper.tex"
            else _MAX_VALIDATION_METADATA_BYTES
        )
        data = _read_bounded_workspace_file_nofollow(
            workspace_root,
            relative,
            byte_limit=min(per_file_limit, remaining_metadata_bytes),
            description=f"validation input {relative_path}",
        )
        if data is None:
            continue
        remaining_metadata_bytes -= len(data)
        if relative_path == "paper.tex":
            # The orchestrator-managed watermark block is regenerated before every
            # compile and is not manuscript content; keep audit hashes independent of it.
            data = strip_managed_block(data.decode("utf-8")).encode("utf-8")
        if relative_path == "scientific-evidence/verification-report.json":
            payload = json.loads(data.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {workspace_root / relative}")
            stable_payload = {
                key: value
                for key, value in payload.items()
                if key not in _VOLATILE_VERIFICATION_FIELDS
            }
            data = json.dumps(
                stable_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        hashes[relative_path] = hashlib.sha256(data).hexdigest()

    tex_sources = _paper_tex_sources(workspace_root)
    for tex_relative_path, source_bytes in tex_sources:
        if tex_relative_path != Path("paper.tex"):
            hashes[f"tex-source/{tex_relative_path.as_posix()}"] = hashlib.sha256(
                source_bytes
            ).hexdigest()
    remaining_style_bytes = _MAX_VALIDATION_BIBLIOGRAPHY_STYLE_BYTES
    for style_path in _referenced_bibliography_style_paths(workspace_root, tex_sources):
        style_data = _read_bounded_workspace_file_nofollow(
            workspace_root,
            style_path,
            byte_limit=min(_MAX_VALIDATION_METADATA_BYTES, remaining_style_bytes),
            description=f"bibliography style {style_path}",
        )
        if style_data is not None:
            remaining_style_bytes -= len(style_data)
            hashes[f"compile-input/{style_path.as_posix()}"] = hashlib.sha256(
                style_data
            ).hexdigest()

    figure_bytes = 0
    manifest_figures, prompt_logs = _manifest_validator_input_paths(workspace_root)
    figure_paths = set(_referenced_figure_paths(workspace_root, tex_sources))
    figure_paths.update(manifest_figures)
    for figure_path in sorted(figure_paths, key=lambda path: path.as_posix()):
        result = _hash_figure_nofollow(
            workspace_root,
            figure_path,
            byte_limit=_MAX_AUDIT_FIGURE_BYTES - figure_bytes,
        )
        if result is None:
            continue
        figure_digest, bytes_read = result
        figure_bytes += bytes_read
        hashes[f"figure-asset/{figure_path.as_posix()}"] = figure_digest
    for prompt_log_path in prompt_logs:
        prompt_data = _read_bounded_workspace_file_nofollow(
            workspace_root,
            prompt_log_path,
            byte_limit=min(_MAX_VALIDATION_METADATA_BYTES, remaining_metadata_bytes),
            description=f"concept figure prompt log {prompt_log_path}",
        )
        if prompt_data is not None:
            remaining_metadata_bytes -= len(prompt_data)
            hashes[f"validator-input/{prompt_log_path.as_posix()}"] = hashlib.sha256(
                prompt_data
            ).hexdigest()
    return hashes


def _manifest_validator_input_paths(
    workspace_root: Path,
) -> tuple[set[Path], list[Path]]:
    manifest_data = _read_bounded_workspace_file_nofollow(
        workspace_root,
        Path("figure_manifest.json"),
        byte_limit=_MAX_VALIDATION_METADATA_BYTES,
        description="validation input figure_manifest.json",
    )
    if manifest_data is None:
        return set(), []
    payload = _json_object_from_bytes(manifest_data, "figure_manifest.json")
    rows = payload.get("figures")
    if not isinstance(rows, list):
        return set(), []
    if len(rows) > _MAX_AUDIT_FIGURE_COUNT:
        raise ValueError(
            f"Figure manifest contains more than {_MAX_AUDIT_FIGURE_COUNT} figures"
        )

    figure_paths: set[Path] = set()
    prompt_logs: set[Path] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        files = row.get("files")
        nested_png = files.get("png") if isinstance(files, dict) else None
        nested_pdf = files.get("pdf") if isinstance(files, dict) else None
        raw_paths = (
            row.get("path_png", nested_png),
            row.get("path_pdf", nested_pdf),
        )
        for raw_path_value in raw_paths:
            raw_path = raw_path_value.strip() if isinstance(raw_path_value, str) else ""
            if not raw_path:
                continue
            path = _validated_figure_candidate(raw_path)
            if len(path.parts) < 2 or path.parts[0] != "figures":
                continue
            figure_paths.add(path)
        prompt_log_value = row.get("prompt_log_path")
        figure_id_value = row.get("figure_id")
        if isinstance(prompt_log_value, str) and isinstance(figure_id_value, str):
            raw_prompt_path = prompt_log_value.strip()
            expected = f"figures/{figure_id_value}.prompts.json"
            if raw_prompt_path == expected:
                prompt_logs.add(Path(raw_prompt_path))
    return figure_paths, sorted(prompt_logs, key=lambda path: path.as_posix())


def _json_object_from_bytes(data: bytes, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _referenced_bibliography_style_paths(
    workspace_root: Path,
    tex_sources: list[tuple[Path, bytes]],
) -> list[Path]:
    styles: set[Path] = set()
    candidates: set[Path] = set()
    for source_path, source_bytes in tex_sources:
        if source_path.suffix.casefold() != ".tex":
            continue
        text = mask_false_conditionals(
            mask_tex_verbatim(strip_tex_comments(source_bytes.decode("utf-8")))
        )
        for match in _BIBLIOGRAPHY_STYLE_RE.finditer(text):
            raw_target = match.group("target").strip()
            if len(raw_target) > _MAX_VALIDATION_BIBLIOGRAPHY_STYLE_TARGET_CHARS:
                raise ValueError("Bibliography style target is too long")
            if "\\" in raw_target or "#" in raw_target:
                raise ValueError(f"Dynamic bibliography style target: {raw_target!r}")
            target = Path(raw_target)
            if target.is_absolute() or any(part in {"", ".", ".."} for part in target.parts):
                raise ValueError(f"Unsafe bibliography style target: {raw_target!r}")
            if not target.suffix:
                target = target.with_suffix(".bst")
            candidate = Path(os.path.abspath(workspace_root / target))
            try:
                relative = candidate.relative_to(workspace_root)
            except ValueError as exc:
                raise ValueError(
                    f"Bibliography style escapes the paper workspace: {raw_target!r}"
                ) from exc
            candidates.add(relative)
            if len(candidates) > _MAX_VALIDATION_BIBLIOGRAPHY_STYLE_COUNT:
                raise ValueError(
                    "Paper references more than "
                    f"{_MAX_VALIDATION_BIBLIOGRAPHY_STYLE_COUNT} bibliography styles"
                )
            if _workspace_regular_file_exists(workspace_root, relative):
                styles.add(relative)
    return sorted(styles, key=lambda path: path.as_posix())


def audit_input_sha256(paper_dir: Path) -> dict[str, str]:
    """Hash the current evidence-audit inputs without loading them wholesale."""
    paper_dir = paper_dir.expanduser().resolve()
    hashes = _paper_input_sha256(paper_dir, AUDIT_INPUT_PATHS)
    for path in _evidence_source_paths(paper_dir):
        relative_path = path.relative_to(paper_dir).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[f"evidence-source/{relative_path}"] = digest.hexdigest()
    for path in _resolved_evidence_directories(paper_dir):
        relative_path = path.relative_to(paper_dir).as_posix()
        hashes[f"evidence-tree/{relative_path}"] = _directory_sha256(
            path,
            workspace_root=paper_dir,
        )
    return hashes


def _referenced_figure_paths(
    paper_dir: Path,
    tex_sources: list[tuple[Path, bytes]],
) -> list[Path]:
    search_paths: tuple[Path, ...] | None = None
    for _, source_bytes in tex_sources:
        tex_text = source_bytes.decode("utf-8")
        for raw_declaration in iter_graphics_search_path_declarations(tex_text):
            if len(raw_declaration) > _MAX_AUDIT_GRAPHICS_PATH_COUNT:
                raise ValueError(
                    "Paper declares more than "
                    f"{_MAX_AUDIT_GRAPHICS_PATH_COUNT} graphics search paths"
                )
            declaration = tuple(
                _validated_graphics_search_path(raw_path) for raw_path in raw_declaration
            )
            if search_paths is not None and declaration != search_paths:
                raise ValueError("Paper declares conflicting graphics search paths")
            search_paths = declaration

    candidates: dict[str, bool] = {}
    reference_count = 0
    candidate_count = 0
    candidate_chars = 0

    def consume_candidate_budget(candidate: str) -> None:
        nonlocal candidate_count, candidate_chars
        candidate_count += 1
        candidate_chars += len(candidate)
        if (
            candidate_count > _MAX_AUDIT_FIGURE_CANDIDATE_COUNT
            or candidate_chars > _MAX_AUDIT_FIGURE_CANDIDATE_CHARS
        ):
            raise ValueError("Paper figure references exceed the audit expansion limit")

    def add_candidate(candidate: str, *, support_file: bool) -> None:
        candidates[candidate] = candidates.get(candidate, True) and support_file

    for source_path, source_bytes in tex_sources:
        tex_text = source_bytes.decode("utf-8")
        support_file = source_path.suffix.casefold() in {".cls", ".sty"}
        for raw_path in iter_graphics_paths(tex_text):
            reference_count += 1
            if reference_count > _MAX_AUDIT_FIGURE_COUNT:
                raise ValueError(
                    f"Paper references more than {_MAX_AUDIT_FIGURE_COUNT} figure assets"
                )
            if len(raw_path) > _MAX_AUDIT_FIGURE_REFERENCE_CHARS:
                raise ValueError("Paper figure reference exceeds the audit path limit")
            if ("#" in raw_path or "\\" in raw_path) and support_file:
                continue
            for candidate in sorted(candidate_graphics_paths(raw_path)):
                candidate_path = _validated_figure_candidate(candidate)
                consume_candidate_budget(candidate)
                if candidate_path.parts and candidate_path.parts[0] == "figures":
                    add_candidate(candidate, support_file=support_file)
                    continue
                if search_paths:
                    if _workspace_regular_file_exists(paper_dir, candidate_path):
                        raise ValueError(
                            "Unsafe figure reference shadows graphicspath asset in "
                            f"paper root: {candidate!r}"
                        )
                    for search_path in search_paths:
                        resolved_candidate = (search_path / candidate_path).as_posix()
                        consume_candidate_budget(resolved_candidate)
                        resolved_path = _validated_figure_candidate(resolved_candidate)
                        if _workspace_regular_file_exists(paper_dir, resolved_path):
                            add_candidate(
                                resolved_candidate,
                                support_file=support_file,
                            )
                            break
                    continue
                add_candidate(candidate, support_file=support_file)
    paths: list[Path] = []
    for candidate, support_file in sorted(candidates.items()):
        path = _validated_figure_candidate(candidate)
        if (len(path.parts) < 2 or path.parts[0] != "figures") and not support_file:
            raise ValueError(f"Unsafe figure reference in paper.tex: {candidate!r}")
        if support_file and not _workspace_regular_file_exists(paper_dir, path):
            continue
        paths.append(path)
    return paths


def _validated_graphics_search_path(raw_path: str) -> Path:
    if len(raw_path) > _MAX_AUDIT_GRAPHICS_PATH_CHARS:
        raise ValueError("Paper graphics search path exceeds the audit path limit")
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
        or path.suffix.casefold() not in _AUDIT_FIGURE_SUFFIXES
    ):
        raise ValueError(f"Unsafe figure reference in paper.tex: {candidate!r}")
    return path


def _workspace_regular_file_exists(workspace_root: Path, relative_path: Path) -> bool:
    descriptor = _open_regular_workspace_file_nofollow(
        workspace_root,
        relative_path,
        description=f"figure asset {relative_path}",
    )
    if descriptor is None:
        return False
    os.close(descriptor)
    return True


def _paper_tex_sources(paper_dir: Path) -> list[tuple[Path, bytes]]:
    workspace_root = paper_dir.resolve()
    pending = [Path("paper.tex")]
    queued = set(pending)
    visited: set[Path] = set()
    sources: list[tuple[Path, bytes]] = []
    total_bytes = 0
    conditional_tokens = 0
    while pending:
        relative_path = pending.pop()
        if relative_path in visited:
            continue
        visited.add(relative_path)
        data = _read_workspace_file_nofollow(
            workspace_root,
            relative_path,
            byte_limit=_MAX_AUDIT_TEX_BYTES - total_bytes,
        )
        if data is None:
            continue
        total_bytes += len(data)
        text = data.decode("utf-8")
        visible_conditionals = mask_tex_verbatim(strip_tex_comments(text))
        conditional_tokens += len(_TEX_CONDITIONAL_TOKEN_RE.findall(visible_conditionals))
        if conditional_tokens > _MAX_VALIDATION_TEX_CONDITIONAL_TOKENS:
            raise ValueError("Paper TeX exceeds the conditional parsing limit")
        sources.append((relative_path, data))
        dependency_targets = chain(
            iter_tex_import_paths(text),
            iter_tex_support_paths(text),
        )
        for raw_target in dependency_targets:
            if "\\" in raw_target or "#" in raw_target:
                raise ValueError(f"Dynamic TeX import target: {raw_target!r}")
            target = Path(raw_target)
            if target.is_absolute():
                raise ValueError(f"TeX import escapes the paper workspace: {raw_target!r}")
            for base in (Path(), relative_path.parent):
                candidate = Path(os.path.abspath(workspace_root / base / target))
                try:
                    candidate_relative = candidate.relative_to(workspace_root)
                except ValueError as exc:
                    raise ValueError(
                        f"TeX import escapes the paper workspace: {raw_target!r}"
                    ) from exc
                if (
                    candidate_relative.parts
                    and candidate_relative.parts[0] in _AGENT_NAMESPACE_NAMES
                ):
                    raise ValueError(f"Paper must not import agent files: {raw_target!r}")
                variants = [candidate_relative]
                if not candidate_relative.suffix:
                    variants.append(candidate_relative.with_suffix(".tex"))
                for variant in variants:
                    if variant in queued:
                        continue
                    queued.add(variant)
                    if len(queued) > _MAX_AUDIT_TEX_SOURCE_COUNT:
                        raise ValueError(
                            "Paper imports more than "
                            f"{_MAX_AUDIT_TEX_SOURCE_COUNT} TeX sources"
                        )
                    pending.append(variant)
    return sources


def _read_workspace_file_nofollow(
    workspace_root: Path,
    relative_path: Path,
    *,
    byte_limit: int,
) -> bytes | None:
    return _read_bounded_workspace_file_nofollow(
        workspace_root,
        relative_path,
        byte_limit=byte_limit,
        description=f"TeX source {relative_path}",
    )


def _read_bounded_workspace_file_nofollow(
    workspace_root: Path,
    relative_path: Path,
    *,
    byte_limit: int,
    description: str,
) -> bytes | None:
    descriptor = _open_regular_workspace_file_nofollow(
        workspace_root,
        relative_path,
        description=description,
    )
    if descriptor is None:
        return None
    try:
        if os.fstat(descriptor).st_size > byte_limit:
            raise ValueError(f"{description} exceeds {byte_limit} bytes")
        data = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, byte_limit - len(data) + 1)):
            data.extend(chunk)
            if len(data) > byte_limit:
                raise ValueError(f"{description} exceeds {byte_limit} bytes")
        return bytes(data)
    finally:
        os.close(descriptor)


def _hash_figure_nofollow(
    paper_dir: Path,
    relative_path: Path,
    *,
    byte_limit: int,
) -> tuple[str, int] | None:
    file_fd = _open_regular_workspace_file_nofollow(
        paper_dir,
        relative_path,
        description=f"figure asset {relative_path}",
    )
    if file_fd is None:
        return None
    try:
        metadata = os.fstat(file_fd)
        if metadata.st_size > byte_limit:
            raise ValueError(
                f"Referenced figure assets exceed {_MAX_AUDIT_FIGURE_BYTES} bytes"
            )
        digest = hashlib.sha256()
        bytes_read = 0
        while chunk := os.read(file_fd, min(1024 * 1024, byte_limit - bytes_read + 1)):
            bytes_read += len(chunk)
            if bytes_read > byte_limit:
                raise ValueError(
                    f"Referenced figure assets exceed {_MAX_AUDIT_FIGURE_BYTES} bytes"
                )
            digest.update(chunk)
        return digest.hexdigest(), bytes_read
    finally:
        os.close(file_fd)


def _open_regular_workspace_file_nofollow(
    workspace_root: Path,
    relative_path: Path,
    *,
    description: str,
) -> int | None:
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} or "\\" in part for part in relative_path.parts)
    ):
        raise ValueError(f"Unsafe workspace path: {relative_path}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    nonblocking = getattr(os, "O_NONBLOCK", 0)
    directory_fd: int | None = None
    opened_directory_fds: list[int] = []
    file_fd: int | None = None
    try:
        directory_fd = os.open(workspace_root, directory_flags | close_on_exec)
        opened_directory_fds.append(directory_fd)
        for part in relative_path.parts[:-1]:
            directory_fd = os.open(
                part,
                directory_flags | nofollow | close_on_exec,
                dir_fd=directory_fd,
            )
            opened_directory_fds.append(directory_fd)
        file_fd = os.open(
            relative_path.name,
            os.O_RDONLY | nofollow | close_on_exec | nonblocking,
            dir_fd=directory_fd,
        )
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(file_fd)
            raise ValueError(f"{description} is not a regular file")
        return file_fd
    except FileNotFoundError:
        return None
    except OSError as exc:
        if file_fd is not None:
            os.close(file_fd)
        raise ValueError(f"Could not safely read {description}: {exc}") from exc
    finally:
        for fd in reversed(opened_directory_fds):
            os.close(fd)


def _evidence_source_paths(paper_dir: Path) -> list[Path]:
    sources: list[Path] = []
    for path in paper_dir.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(paper_dir)
        if any(part in _EVIDENCE_SKIP_DIR_NAMES for part in relative.parts):
            continue
        if _is_agent_skill_path(relative):
            continue
        if _is_generated_audit_artifact(relative):
            continue
        if path.name.endswith(".prov.json") or path.suffix in _EVIDENCE_SCAN_EXTENSIONS or path.name in {"Dockerfile", "Makefile"}:
            sources.append(path)
    ledger_path = paper_dir / "scientific-evidence" / "ledger.jsonl"
    if ledger_path.is_file() and not ledger_path.is_symlink():
        sources.append(ledger_path)
    sources.extend(_resolved_evidence_files(paper_dir))
    return sorted(set(sources))


def _is_generated_audit_artifact(relative_path: Path) -> bool:
    if relative_path.parts and relative_path.parts[0] == "paper_diagnostics":
        return True
    if relative_path.as_posix() == "scientific-evidence/verification-report.json":
        return True
    # Orchestrator-owned cost bookkeeping is appended while agents (including the
    # audit itself) run; hashing it lets the audit stale its own snapshot.
    if relative_path.as_posix() in (
        "metrics/cost_events.jsonl",
        "metrics/cost_summary.json",
    ):
        return True
    if len(relative_path.parts) != 1:
        return False
    if (
        relative_path.name.startswith("paper_events_")
        and relative_path.suffix == ".jsonl"
    ):
        return True
    return bool(
        relative_path.name in _EVIDENCE_GENERATED_NAMES
        or relative_path.name == AUDIT_INPUT_HASH_MANIFEST
        or relative_path.name.startswith("paper_evidence_audit")
        or _PROMPT_ARTIFACT_RE.fullmatch(relative_path.name)
    )


def _resolved_evidence_files(paper_dir: Path) -> list[Path]:
    """Return local verifier-resolved files, including arbitrary binary targets."""
    return [path for path in _resolved_evidence_targets(paper_dir) if path.is_file()]


def _resolved_evidence_directories(paper_dir: Path) -> list[Path]:
    """Return local verifier-resolved directories whose full trees affect the audit."""
    return [path for path in _resolved_evidence_targets(paper_dir) if path.is_dir()]


def _resolved_evidence_targets(paper_dir: Path) -> list[Path]:
    report = _read_json(paper_dir / "scientific-evidence" / "verification-report.json")
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        return []
    root = paper_dir.resolve()
    resolved_targets: list[Path] = []
    for result in report["results"]:
        if not isinstance(result, dict):
            continue
        raw_path = result.get("resolved_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        candidate = Path(raw_path)
        if candidate.is_absolute():
            continue
        candidate = (root / candidate).resolve()
        try:
            resolved_relative = candidate.relative_to(root)
        except ValueError:
            continue
        if _is_generated_audit_artifact(resolved_relative):
            continue
        if candidate.exists():
            resolved_targets.append(candidate)
    return sorted(set(resolved_targets))


def _is_generated_directory_hash_artifact(relative_path: Path) -> bool:
    return (
        relative_path == Path("latest_manifest.json")
        or _is_generated_audit_artifact(relative_path)
    )


def _directory_sha256(directory: Path, *, workspace_root: Path) -> str:
    """Hash a resolved evidence directory without following external symlinks."""
    digest = hashlib.sha256()
    visited_directories: set[tuple[int, int]] = set()

    def hash_record(kind: bytes, relative_path: Path, payload: bytes = b"") -> None:
        relative = os.fsencode(relative_path.as_posix())
        digest.update(kind)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def hash_file(path: Path, relative_path: Path) -> None:
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        hash_record(b"F", relative_path, file_digest.digest())

    def hash_directory(path: Path, relative_prefix: Path) -> None:
        directory_stat = path.stat()
        identity = (directory_stat.st_dev, directory_stat.st_ino)
        if identity in visited_directories:
            hash_record(b"C", relative_prefix)
            return
        visited_directories.add(identity)
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            child_relative_path = relative_prefix / child.name
            try:
                workspace_relative = child.relative_to(workspace_root)
            except ValueError:
                workspace_relative = child_relative_path
            if _is_generated_directory_hash_artifact(workspace_relative):
                continue
            child_stat = child.lstat()
            if child.is_symlink():
                link_target = Path(os.readlink(child))
                if link_target.is_absolute():
                    continue
                try:
                    resolved = child.resolve(strict=True)
                    resolved_relative = resolved.relative_to(workspace_root)
                except (FileNotFoundError, RuntimeError, ValueError):
                    continue
                hash_record(b"L", child_relative_path, os.fsencode(link_target))
                if _is_generated_directory_hash_artifact(resolved_relative):
                    continue
                if resolved.is_file():
                    hash_file(resolved, child_relative_path)
                elif resolved.is_dir():
                    hash_directory(resolved, child_relative_path)
                continue
            if stat.S_ISDIR(child_stat.st_mode):
                hash_record(b"D", child_relative_path)
                hash_directory(child, child_relative_path)
                continue
            if stat.S_ISREG(child_stat.st_mode):
                hash_file(child, child_relative_path)

    hash_directory(directory, Path("."))
    return digest.hexdigest()


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"_json_error": f"Invalid JSON in {path.name}: {exc}"}


def _json_error_summary(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict) and isinstance(payload.get("_json_error"), str):
        return {"available": False, "error": payload["_json_error"]}
    return None


def _truncate(text: str, *, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return f"{stripped[:max_chars].rstrip()}\n\n[truncated after {max_chars} chars]"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _json_excerpt(payload: Any, *, max_chars: int = _MAX_JSON_CHARS) -> str:
    return _truncate(json.dumps(payload, indent=2, sort_keys=True), max_chars=max_chars)


def _paper_data_summary(payload: Any) -> dict[str, Any]:
    error_summary = _json_error_summary(payload)
    if error_summary is not None:
        return error_summary
    if not isinstance(payload, dict):
        return {"available": False}

    grounding_sources = payload.get("grounding_sources")
    claims = payload.get("claims")
    return {
        "available": True,
        "paper_profile": payload.get("paper_profile", "default"),
        "downstream_study": payload.get("downstream_study"),
        "final_evaluation": final_outcome_summary(payload),
        "grounding_source_count": len(grounding_sources)
        if isinstance(grounding_sources, dict)
        else 0,
        "claim_count": len(claims) if isinstance(claims, list) else 0,
        "totals": payload.get("totals") if isinstance(payload.get("totals"), dict) else {},
    }


def _evidence_verification_summary(payload: Any) -> dict[str, Any]:
    error_summary = _json_error_summary(payload)
    if error_summary is not None:
        return error_summary
    if not isinstance(payload, dict):
        return {"available": False}

    summary = payload.get("summary")
    results = payload.get("results")
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(results, list):
        results = []
    preview: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        if row.get("status") != "dangling":
            continue
        preview.append(
            {
                "subject_path": row.get("subject_path"),
                "raw_reference": row.get("raw_reference"),
                "target": row.get("target"),
                "messages": row.get("messages") if isinstance(row.get("messages"), list) else [],
            }
        )
        if len(preview) >= 10:
            break
    return {
        "available": True,
        "summary": {
            "total_references": summary.get("total_references", 0),
            "resolved_references": summary.get("resolved_references", 0),
            "anchored_references": summary.get("anchored_references", 0),
            "dangling_references": summary.get("dangling_references", 0),
            "manifests_checked": summary.get("manifests_checked", 0),
        },
        "dangling_preview": preview,
    }


def _validation_summary(payload: Any) -> dict[str, Any]:
    error_summary = _json_error_summary(payload)
    if error_summary is not None:
        return error_summary
    if not isinstance(payload, dict):
        return {"available": False}

    evidence_validation = payload.get("evidence_validation")
    if not isinstance(evidence_validation, dict):
        evidence_validation = {}
    evidence_summary = evidence_validation.get("summary")
    if not isinstance(evidence_summary, dict):
        evidence_summary = {}
    return {
        "available": True,
        "ok": payload.get("ok"),
        "pdf_path": payload.get("pdf_path"),
        "evidence_validation": {
            "checked": bool(evidence_validation.get("checked", False)),
            "bundle_present": bool(evidence_validation.get("bundle_present", False)),
            "paper_token_count": evidence_validation.get("paper_token_count", 0),
            "total_references": evidence_summary.get("total_references", 0),
            "resolved_references": evidence_summary.get("resolved_references", 0),
            "anchored_references": evidence_summary.get("anchored_references", 0),
            "dangling_references": evidence_summary.get("dangling_references", 0),
        },
    }


def _symlink_points_within(root_dir: Path, symlink_path: Path) -> bool:
    target_path = (symlink_path.parent / os.readlink(symlink_path)).resolve(strict=False)
    try:
        target_path.relative_to(root_dir)
    except ValueError:
        return False
    return True


def _is_agent_skill_path(relative_path: Path) -> bool:
    return any(
        relative_path.parts[: len(prefix)] == prefix
        for prefix in _AGENT_SKILL_DIR_PREFIXES
    )


def _ignore_external_symlinks(root_dir: Path):
    resolved_root = root_dir.resolve()

    def _ignore(current_dir: str, names: list[str]) -> set[str]:
        current_path = Path(current_dir)
        ignored: set[str] = set()
        for name in names:
            candidate = current_path / name
            try:
                relative_candidate = candidate.relative_to(resolved_root)
            except ValueError:
                ignored.add(name)
                continue
            if _is_agent_skill_path(relative_candidate):
                ignored.add(name)
                continue
            if (
                current_path == resolved_root
                and name in _AGENT_NAMESPACE_NAMES
                and candidate.is_symlink()
            ):
                ignored.add(name)
                continue
            if not candidate.is_symlink():
                try:
                    mode = candidate.lstat().st_mode
                except FileNotFoundError:
                    ignored.add(name)
                    continue
                if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
                    ignored.add(name)
                continue
            if Path(os.readlink(candidate)).is_absolute():
                ignored.add(name)
                continue
            if _symlink_points_within(resolved_root, candidate):
                continue
            ignored.add(name)
        return ignored

    return _ignore


def _make_tree_readonly(root_dir: Path) -> None:
    for path in [root_dir, *root_dir.rglob("*")]:
        if path.is_symlink():
            continue
        current_mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, current_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def remove_readonly_workspace(root_dir: Path) -> None:
    """Remove a read-only audit workspace by restoring owner write bits first."""
    root_dir = Path(os.path.abspath(root_dir.expanduser()))
    if not root_dir.exists():
        return
    if root_dir.is_symlink():
        raise ValueError(f"Read-only audit workspace must not be a symlink: {root_dir}")
    for path in sorted(root_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        try:
            current_mode = stat.S_IMODE(path.stat().st_mode)
            os.chmod(path, current_mode | stat.S_IWUSR)
        except FileNotFoundError:
            continue
    if not root_dir.is_symlink():
        current_mode = stat.S_IMODE(root_dir.stat().st_mode)
        os.chmod(root_dir, current_mode | stat.S_IWUSR | stat.S_IXUSR)
    shutil.rmtree(root_dir)


def _audit_workspace_root(source_dir: Path) -> Path:
    source_dir = Path(os.path.abspath(source_dir.expanduser()))
    return source_dir.parent / f"{_AUDIT_WORKSPACE_ROOT_PREFIX}{os.getuid()}"


def _require_trusted_directory(
    path: Path,
    *,
    label: str,
    require_owner: bool = True,
    private: bool = False,
) -> os.stat_result:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise ValueError(f"{label} must be a non-symlink directory: {path}")
    if require_owner and path_stat.st_uid != os.getuid():
        raise ValueError(f"{label} must be owned by uid {os.getuid()}: {path}")
    if private and stat.S_IMODE(path_stat.st_mode) != 0o700:
        raise ValueError(f"{label} must have mode 0700: {path}")
    return path_stat


def _require_trusted_output_parent(source_dir: Path) -> Path:
    source_dir = Path(os.path.abspath(source_dir.expanduser()))
    output_parent = source_dir.parent
    parent_stat = _require_trusted_directory(
        output_parent,
        label="Paper output parent",
        require_owner=False,
    )
    if parent_stat.st_uid == os.getuid():
        writable_by_others = parent_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        if writable_by_others and not parent_stat.st_mode & stat.S_ISVTX:
            output_parent.chmod(stat.S_IMODE(parent_stat.st_mode) | stat.S_ISVTX)
    elif parent_stat.st_uid != 0:
        raise ValueError(
            f"Paper output parent must be owned by uid {os.getuid()} or root: {output_parent}"
        )
    elif parent_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH) and not (
        parent_stat.st_mode & stat.S_ISVTX
    ):
        raise ValueError(
            f"Root-owned paper output parent is group/other-writable without sticky bit: {output_parent}"
        )
    _require_trusted_directory(source_dir, label="Paper output directory")
    return output_parent


def _ensure_audit_workspace_root(source_dir: Path) -> Path:
    output_parent = _require_trusted_output_parent(source_dir)
    audit_root = _audit_workspace_root(source_dir)
    try:
        audit_root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _require_trusted_directory(audit_root, label="Audit workspace root", private=True)
    if audit_root.parent != output_parent:
        raise ValueError(f"Audit workspace root escaped the paper output parent: {audit_root}")
    return audit_root


def prepare_audit_workspace(
    source_dir: Path,
    *,
    workspace_tag: str,
    evidence_script_path: Path | None = None,
) -> Path:
    """Create one managed read-only audit copy beside a paper output directory."""
    if not _WORKSPACE_TAG_RE.fullmatch(workspace_tag):
        raise ValueError(f"Invalid audit workspace tag: {workspace_tag!r}")
    source_dir = Path(os.path.abspath(source_dir.expanduser()))
    audit_root = _ensure_audit_workspace_root(source_dir)
    parent_prefix = f"{_AUDIT_WORKSPACE_PARENT_PREFIX}{os.getuid()}-{workspace_tag}-"
    workspace_parent = Path(tempfile.mkdtemp(prefix=parent_prefix, dir=audit_root))
    marker_path = workspace_parent / _AUDIT_WORKSPACE_MARKER
    readonly_dir = workspace_parent / "workspace"
    try:
        marker_path.write_text(_AUDIT_WORKSPACE_MARKER_CONTENT, encoding="utf-8")
        marker_path.chmod(0o400)
        return prepare_readonly_workspace(
            source_dir,
            readonly_dir,
            evidence_script_path=evidence_script_path,
            packet_output_path=readonly_dir / "paper_evidence_audit_packet.md",
        )
    except Exception:
        if readonly_dir.exists() and not readonly_dir.is_symlink():
            remove_readonly_workspace(readonly_dir)
        if marker_path.exists() and not marker_path.is_symlink():
            marker_path.unlink()
        workspace_parent.rmdir()
        raise


def cleanup_audit_workspace(readonly_dir: Path, *, source_dir: Path) -> None:
    """Remove one validated managed audit copy while preserving its per-user root."""
    source_dir = Path(os.path.abspath(source_dir.expanduser()))
    readonly_dir = Path(os.path.abspath(readonly_dir.expanduser()))
    expected_root = _ensure_audit_workspace_root(source_dir)
    workspace_parent = readonly_dir.parent
    expected_prefix = f"{_AUDIT_WORKSPACE_PARENT_PREFIX}{os.getuid()}-"
    if (
        readonly_dir.name != "workspace"
        or workspace_parent.parent != expected_root
        or not workspace_parent.name.startswith(expected_prefix)
    ):
        raise ValueError(f"Path is not a managed audit workspace: {readonly_dir}")
    _require_trusted_directory(workspace_parent, label="Audit workspace parent", private=True)
    if readonly_dir.is_symlink():
        raise ValueError(f"Read-only audit workspace must not be a symlink: {readonly_dir}")
    marker_path = workspace_parent / _AUDIT_WORKSPACE_MARKER
    try:
        marker_stat = marker_path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Path is not a managed audit workspace: {readonly_dir}") from exc
    if (
        not stat.S_ISREG(marker_stat.st_mode)
        or marker_stat.st_uid != os.getuid()
        or stat.S_IMODE(marker_stat.st_mode) != 0o400
        or marker_path.read_text(encoding="utf-8") != _AUDIT_WORKSPACE_MARKER_CONTENT
    ):
        raise ValueError(f"Invalid managed audit workspace marker: {marker_path}")
    if not readonly_dir.exists():
        marker_path.unlink()
        workspace_parent.rmdir()
        return
    _require_trusted_directory(readonly_dir, label="Read-only audit workspace")
    remove_readonly_workspace(readonly_dir)
    marker_path.unlink()
    workspace_parent.rmdir()


def _is_usable_verification_report(path: Path, *, expected_root: Path) -> bool:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_name") != "scientific-evidence.verification-report":
        return False
    if not all(isinstance(payload.get(key), str) and payload[key] for key in ("verified_at", "root_dir")):
        return False
    if Path(payload["root_dir"]).expanduser().resolve() != expected_root.resolve():
        return False
    summary = payload.get("summary")
    results = payload.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        return False
    if any(
        not isinstance(summary.get(key), int)
        or isinstance(summary.get(key), bool)
        or summary[key] < 0
        for key in _VERIFICATION_SUMMARY_KEYS
    ):
        return False
    statuses: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            return False
        required_strings = ("subject_path", "subject_kind", "raw_reference", "target", "status")
        if any(not isinstance(result.get(key), str) or not result[key] for key in required_strings):
            return False
        status_value = result["status"]
        if status_value not in {"resolved", "anchored", "dangling"}:
            return False
        if result["subject_kind"] not in {"text-artifact", "manifest", "ledger-entry", "other"}:
            return False
        if any(
            key in result and (not isinstance(result[key], str) or not result[key])
            for key in ("locator", "resolved_path")
        ):
            return False
        if "hash_checked" in result and not isinstance(result["hash_checked"], bool):
            return False
        messages = result.get("messages")
        if not isinstance(messages, list) or any(not isinstance(message, str) for message in messages):
            return False
        statuses.append(status_value)
    return (
        summary["total_references"] == len(results)
        and summary["resolved_references"] == statuses.count("resolved")
        and summary["anchored_references"] == statuses.count("anchored")
        and summary["dangling_references"] == statuses.count("dangling")
    )


def _refresh_evidence_verification_report(
    destination_dir: Path, evidence_script_path: Path
) -> None:
    destination_dir = destination_dir.resolve()
    evidence_dir = destination_dir / "scientific-evidence"
    try:
        evidence_stat = evidence_dir.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(evidence_stat.st_mode) or evidence_dir.is_symlink():
        raise ValueError(f"Scientific-evidence directory must be a real directory: {evidence_dir}")
    try:
        evidence_dir.resolve().relative_to(destination_dir)
    except ValueError as exc:
        raise ValueError(f"Scientific-evidence directory escaped paper workspace: {evidence_dir}") from exc
    output_path = evidence_dir / "verification-report.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".verification-report.",
        suffix=".json",
        dir=evidence_dir,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(evidence_script_path),
                "verify",
                "--root-dir",
                str(destination_dir),
                "--output",
                str(temporary_path),
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=300,
        )
        details = "\n".join(
            part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
        ).strip()
        if completed.returncode not in {0, 1}:
            raise ValueError(
                "Failed to refresh scientific-evidence verification report in readonly "
                f"workspace: verifier exited {completed.returncode}. {details}"
            )
        temporary_stat = temporary_path.lstat()
        if not stat.S_ISREG(temporary_stat.st_mode) or temporary_stat.st_uid != os.getuid():
            raise ValueError(f"Verifier output is not a trusted regular file: {temporary_path}")
        if not _is_usable_verification_report(
            temporary_path, expected_root=destination_dir
        ):
            raise ValueError(
                "Failed to refresh scientific-evidence verification report in readonly "
                f"workspace: {details or f'verifier exited {completed.returncode} without a usable report'}"
            )
        payload = _read_json(temporary_path)
        assert isinstance(payload, dict)
        summary = payload["summary"]
        assert isinstance(summary, dict)
        has_dangling = summary["dangling_references"] > 0
        if (completed.returncode == 1) != has_dangling:
            raise ValueError(
                "Failed to refresh scientific-evidence verification report in readonly "
                "workspace: verifier exit status disagrees with dangling reference count"
            )
        os.replace(temporary_path, output_path)
        if completed.returncode == 1:
            print(
                "Warning: scientific-evidence verifier returned non-zero after writing "
                f"a usable report; continuing with audit packet generation. {details}",
                file=sys.stderr,
            )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def prepare_readonly_workspace(
    source_dir: Path,
    destination_dir: Path,
    *,
    evidence_script_path: Path | None = None,
    packet_output_path: Path | None = None,
) -> Path:
    """Copy a paper workspace to a read-only directory without dereferencing external symlinks."""
    source_dir = source_dir.expanduser().resolve()
    destination_dir = destination_dir.expanduser().resolve()
    if evidence_script_path is not None:
        evidence_script_path = evidence_script_path.expanduser().resolve()
        if not evidence_script_path.is_file():
            raise FileNotFoundError(
                f"Missing scientific-evidence verifier script: {evidence_script_path}"
            )
    if packet_output_path is not None:
        packet_output_path = packet_output_path.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing paper workspace directory: {source_dir}")
    if (
        destination_dir == source_dir
        or source_dir in destination_dir.parents
        or destination_dir in source_dir.parents
    ):
        raise ValueError("Read-only audit source and destination must not contain each other")
    if destination_dir.exists():
        remove_readonly_workspace(destination_dir)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_dir,
        destination_dir,
        symlinks=True,
        ignore=_ignore_external_symlinks(source_dir),
    )
    if evidence_script_path is not None:
        _refresh_evidence_verification_report(destination_dir, evidence_script_path)
    if packet_output_path is not None:
        build_audit_packet(destination_dir, output_path=packet_output_path)
    (destination_dir / AUDIT_INPUT_HASH_MANIFEST).write_text(
        json.dumps(audit_input_sha256(destination_dir), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _make_tree_readonly(destination_dir)
    return destination_dir


def build_audit_packet(paper_dir: Path, *, output_path: Path | None = None) -> Path:
    """Build a compact packet for the read-only evidence audit agent."""
    paper_dir = paper_dir.expanduser().resolve()
    output_path = (
        (output_path or (paper_dir / "paper_evidence_audit_packet.md")).expanduser().resolve()
    )

    paper_tex = paper_dir / "paper.tex"
    if not paper_tex.is_file():
        raise FileNotFoundError(f"Missing required manuscript artifact: {paper_tex}")

    claims_grounding = _read_json(paper_dir / "claims_grounding.json")
    paper_data = _read_json(paper_dir / "paper_data.json")
    validation_report = _read_json(paper_dir / "validation_report.json")
    evidence_verification = _read_json(
        paper_dir / "scientific-evidence" / "verification-report.json"
    )
    references_provenance = _read_json(paper_dir / "references_provenance.json")

    packet_lines = [
        "# Agentic evidence audit packet",
        "",
        f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- paper_dir: `{paper_dir}`",
        "",
        "Use this packet as the default entrypoint. Read raw artifacts only when this packet is insufficient.",
        "The goal is narrow: check whether the paper's visible claims remain consistent with claims_grounding.json, paper_data.json, and the scientific-evidence trail.",
        "",
        "## Expected audit focus",
        "",
        "- visible claims in `paper.tex` that overstate, contradict, or drift from `claims_grounding.json`",
        "- claims grounded to the wrong source ids, or source ids that do not support the visible wording",
        "- mismatches between visible manuscript text and the resolved/anchored evidence trail",
        "- cases where the evidence trail exists but does not justify the paper's wording or numeric precision",
        "- contradictions between `paper.tex`, `claims_grounding.json`, `paper_data.json`, and `scientific-evidence/verification-report.json`",
        "- every visible claim in Supplementary Information, including tables and figures, under the same grounding standard as the main paper",
        "",
        "## Artifact checklist",
        "",
    ]
    for name in [
        "paper.tex",
        "claims_grounding.json",
        "paper_data.json",
        "validation_report.json",
        "scientific-evidence/verification-report.json",
        "references_provenance.json",
    ]:
        path = paper_dir / name
        state = "present" if path.exists() else "missing"
        packet_lines.append(f"- `{name}` — `{state}`")

    paper_tex_text = _read_text(paper_tex) or ""
    manuscript_text, supplement_excerpt = split_manuscript_and_supplement(paper_tex_text)
    packet_lines.extend(
        [
            "",
            "## validation report summary",
            "",
            "```json",
            _json_excerpt(_validation_summary(validation_report)),
            "```",
            "",
            "## scientific-evidence verification summary",
            "",
            "```json",
            _json_excerpt(_evidence_verification_summary(evidence_verification)),
            "```",
            "",
            "## claims_grounding excerpt",
            "",
            "```json",
            _json_excerpt(claims_grounding),
            "```",
            "",
            "## paper_data summary",
            "",
            "```json",
            _json_excerpt(_paper_data_summary(paper_data)),
            "```",
            "",
            "## references provenance summary",
            "",
            "```json",
            _json_excerpt(references_provenance),
            "```",
            "",
            "## manuscript source excerpt",
            "",
            "```tex",
            _truncate(manuscript_text, max_chars=_MAX_TEX_CHARS),
            "```",
            "",
        ]
    )
    if supplement_excerpt is not None:
        packet_lines.extend(
            [
                "## supplementary information excerpt",
                "",
                "```tex",
                _truncate(supplement_excerpt, max_chars=_MAX_TEX_CHARS),
                "```",
                "",
            ]
        )
    _atomic_write_text(output_path, "\n".join(packet_lines))
    return output_path


def extract_audit_response(log_path: Path) -> str:
    """Extract the final assistant text payload from a stream-json log."""
    log_path = log_path.expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing audit log: {log_path}")

    latest_candidate: tuple[int, str] | None = None
    latest_recovered_fail: tuple[int, str] | None = None
    for line_number, raw_line_text in _iter_bounded_log_lines(log_path):
        line = raw_line_text.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            recovered = _recover_fail_contract_from_malformed_line(line_number, line)
            if recovered is not None:
                latest_recovered_fail = recovered
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "result" and payload.get("subtype") == "success":
            result_text = payload.get("result")
            if (
                isinstance(result_text, str)
                and result_text.strip()
                and _VERDICT_PREFIX in result_text
            ):
                latest_candidate = (line_number, result_text.strip())

        if payload.get("type") in {"message_end", "turn_end"}:
            message_text = _assistant_message_text(payload.get("message"))
            if message_text is not None and _VERDICT_PREFIX in message_text:
                latest_candidate = (line_number, message_text)

        item = payload.get("item")
        if (
            payload.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
        ):
            item_text = item.get("text")
            if (
                isinstance(item_text, str)
                and item_text.strip()
                and _VERDICT_PREFIX in item_text
            ):
                latest_candidate = (line_number, item_text.strip())
    candidates = [
        candidate
        for candidate in (latest_candidate, latest_recovered_fail)
        if candidate is not None and _VERDICT_PREFIX in candidate[1]
    ]
    if not candidates:
        raise AuditContractError(
            f"Audit log did not contain a parseable success result: {log_path}"
        )
    return max(candidates, key=lambda candidate: candidate[0])[1]


def _iter_bounded_log_lines(log_path: Path) -> Iterator[tuple[int, str]]:
    with log_path.open("rb") as handle:
        line_number = 0
        while True:
            raw_line = handle.readline(_MAX_AUDIT_LOG_LINE_BYTES + 1)
            if not raw_line:
                return
            current_line_number = line_number
            line_number += 1
            if len(raw_line) > _MAX_AUDIT_LOG_LINE_BYTES and not raw_line.endswith(b"\n"):
                while raw_line and not raw_line.endswith(b"\n"):
                    raw_line = handle.readline(_MAX_AUDIT_LOG_LINE_BYTES + 1)
                continue
            yield current_line_number, raw_line.decode("utf-8", errors="replace")


def _assistant_message_text(raw_message: object) -> str | None:
    if not isinstance(raw_message, dict) or raw_message.get("role") != "assistant":
        return None
    content = raw_message.get("content")
    if not isinstance(content, list):
        return None
    text_parts = [
        str(part["text"]).strip()
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
        and str(part["text"]).strip()
    ]
    return "\n".join(text_parts) if text_parts else None


def _recover_fail_contract_from_malformed_line(
    line_number: int,
    raw_line: str,
) -> tuple[int, str] | None:
    """Recover only conservative FAIL contracts from malformed terminal output."""
    if _VERDICT_PREFIX not in raw_line:
        return None
    normalized = raw_line
    for _ in range(2):
        normalized = normalized.replace("\\r\\n", "\n").replace("\\n", "\n")
    contract_start = normalized.rfind(_VERDICT_PREFIX)
    if contract_start < 0:
        return None

    verdict: str | None = None
    summary: str | None = None
    finding_count: int | None = None
    findings: list[str] = []
    for raw_contract_line in normalized[contract_start:].splitlines():
        line = raw_contract_line.strip()
        if line.startswith(_VERDICT_PREFIX):
            match = re.match(r"AUDIT_VERDICT:\s*(PASS|FAIL)\b", line)
            verdict = match.group(1) if match else None
        elif line.startswith(_SUMMARY_PREFIX):
            summary = line[len(_SUMMARY_PREFIX) :].strip()
        elif line.startswith(_FINDING_COUNT_PREFIX):
            match = re.match(r"AUDIT_FINDING_COUNT:\s*(\d+)\b", line)
            finding_count = int(match.group(1)) if match else None
        elif line.startswith(_FINDING_PREFIX):
            findings.append(line.split('"}]', 1)[0].strip())

    if verdict != "FAIL" or not summary or finding_count is None:
        return None
    candidate = "\n".join(
        [
            f"{_VERDICT_PREFIX}{verdict}",
            f"{_SUMMARY_PREFIX}{summary}",
            f"{_FINDING_COUNT_PREFIX}{finding_count}",
            *findings,
        ]
    )
    try:
        parse_audit_response(candidate)
    except ValueError:
        return None
    return line_number, candidate


def parse_audit_response(text: str) -> AuditPayload:
    """Parse the regex-stable raw audit contract emitted by the audit agent."""
    if len(text) > _MAX_AUDIT_CONTRACT_CHARS:
        raise ValueError("Audit response exceeds the maximum contract size")
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
    verdict_raw: str | None = None
    summary: str | None = None
    declared_finding_count: int | None = None
    findings: list[AuditFinding] = []

    for line in lines:
        if line.startswith(_VERDICT_PREFIX):
            verdict_raw = line[len(_VERDICT_PREFIX) :].strip().upper()
            continue
        if line.startswith(_SUMMARY_PREFIX):
            summary = line[len(_SUMMARY_PREFIX) :].strip()
            continue
        if line.startswith(_FINDING_COUNT_PREFIX):
            raw_count = line[len(_FINDING_COUNT_PREFIX) :].strip()
            if not raw_count.isdigit():
                raise ValueError(f"Invalid AUDIT_FINDING_COUNT line: {line}")
            declared_finding_count = int(raw_count)
            continue
        if line.startswith(_FINDING_PREFIX):
            findings.append(_parse_finding_line(line))

    if verdict_raw not in {"PASS", "FAIL"}:
        raise ValueError("Missing or invalid AUDIT_VERDICT line")
    if not summary:
        raise ValueError("Missing AUDIT_SUMMARY line")
    if declared_finding_count is None:
        raise ValueError("Missing AUDIT_FINDING_COUNT line")
    if declared_finding_count != len(findings):
        raise ValueError(
            f"AUDIT_FINDING_COUNT={declared_finding_count} does not match parsed findings={len(findings)}"
        )
    if verdict_raw == "PASS" and findings:
        raise ValueError("AUDIT_VERDICT PASS cannot include AUDIT_FINDING lines")
    if verdict_raw == "FAIL" and not findings:
        raise ValueError("AUDIT_VERDICT FAIL must include at least one AUDIT_FINDING line")

    payload = AuditPayload(
        verdict="pass" if verdict_raw == "PASS" else "fail",
        summary=summary,
        findings=findings,
    )
    return payload


def _parse_finding_line(line: str) -> AuditFinding:
    body = line[len(_FINDING_PREFIX) :].strip()
    raw_parts = [part.strip() for part in body.split(" || ")]
    parsed: dict[str, str] = {}
    for part in raw_parts:
        if "=" not in part:
            raise ValueError(f"Invalid AUDIT_FINDING field (missing '='): {part}")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in _FINDING_KEYS:
            raise ValueError(f"Unknown AUDIT_FINDING field {key!r}")
        if key in parsed:
            raise ValueError(f"Duplicate AUDIT_FINDING field {key!r}")
        parsed[key] = value
    missing = [key for key in _FINDING_KEYS if key not in parsed]
    if missing:
        raise ValueError(f"AUDIT_FINDING missing required fields: {', '.join(missing)}")
    return AuditFinding(**parsed)


def render_audit_markdown(audit: AuditPayload) -> str:
    """Render a human-readable markdown audit summary."""
    lines = [
        "# Agentic evidence audit",
        "",
        f"- verdict: `{audit.verdict}`",
        f"- finding_count: `{len(audit.findings)}`",
        "",
        "## Summary",
        audit.summary,
        "",
        "## Findings",
    ]
    if not audit.findings:
        lines.append("- None.")
        lines.append("")
        return "\n".join(lines)
    for idx, finding in enumerate(audit.findings, start=1):
        lines.extend(
            [
                f"### {idx}. {finding.id} — {finding.section}",
                f"- claim: {finding.claim}",
                f"- issue: {finding.issue}",
                f"- evidence: {finding.evidence}",
                f"- requested_fix: {finding.fix}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_audit_feedback(
    audit: AuditPayload,
    *,
    audit_json_path: Path,
    audit_markdown_path: Path,
    feedback_json_path: Path,
    feedback_md_path: Path,
    raw_output_path: Path,
) -> dict[str, Any]:
    """Write compact machine-readable audit feedback for the paper repair loop."""
    findings = [finding.model_dump(mode="json") for finding in audit.findings]
    feedback: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": audit.verdict == "pass",
        "verdict": audit.verdict,
        "summary": audit.summary,
        "finding_count": len(audit.findings),
        "audit_json_path": str(audit_json_path.resolve()),
        "audit_markdown_path": str(audit_markdown_path.resolve()),
        "raw_output_path": str(raw_output_path.resolve()),
        "findings": findings,
    }
    feedback_json_path.write_text(json.dumps(feedback, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Agentic evidence audit feedback",
        "",
        f"- generated_at: `{feedback['generated_at']}`",
        f"- ok: `{feedback['ok']}`",
        f"- verdict: `{feedback['verdict']}`",
        f"- finding_count: `{feedback['finding_count']}`",
        f"- audit_json_path: `{audit_json_path.resolve()}`",
        f"- audit_markdown_path: `{audit_markdown_path.resolve()}`",
        f"- raw_output_path: `{raw_output_path.resolve()}`",
        "",
        "## Summary",
        audit.summary,
        "",
        "## Findings",
    ]
    if findings:
        for item in findings:
            lines.append(f"- `{item['id']}` ({item['section']}): {item['issue']}")
            lines.append(f"  - claim: {item['claim']}")
            lines.append(f"  - evidence: {item['evidence']}")
            lines.append(f"  - requested_fix: {item['fix']}")
    else:
        lines.append("- None.")
    lines.append("")
    feedback_md_path.write_text("\n".join(lines), encoding="utf-8")
    return feedback


def _merge_audit_into_validation_report(
    validation_report_path: Path, audit: AuditPayload
) -> dict[str, Any]:
    report: dict[str, Any]
    if validation_report_path.is_file():
        try:
            loaded = json.loads(validation_report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {validation_report_path}: {exc}") from exc
        report = loaded if isinstance(loaded, dict) else {}
    else:
        report = {}

    report["agentic_evidence_audit"] = {
        "checked": True,
        "verdict": audit.verdict,
        "summary": audit.summary,
        "finding_count": len(audit.findings),
        "findings": [finding.model_dump(mode="json") for finding in audit.findings],
    }
    validation_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def finalize_audit(
    *,
    log_path: Path,
    raw_output_path: Path,
    audit_json_path: Path,
    markdown_output_path: Path,
    feedback_json_path: Path,
    feedback_md_path: Path,
    validation_report_path: Path,
    input_hashes_path: Path | None = None,
    evidence_script_path: Path | None = None,
) -> bool:
    """Parse the audit log, persist artifacts, and merge into validation_report.

    Returns True when the audit found at least one mismatch.
    """
    try:
        raw_output = extract_audit_response(log_path)
        audit = parse_audit_response(raw_output)
    except ValueError as exc:
        raise AuditContractError(str(exc)) from exc
    raw_output_path.write_text(raw_output.rstrip() + "\n", encoding="utf-8")
    if evidence_script_path is not None:
        _refresh_evidence_verification_report(
            validation_report_path.parent,
            evidence_script_path.expanduser().resolve(),
        )
    current_input_hashes = audit_input_sha256(validation_report_path.parent)
    if input_hashes_path is not None:
        loaded_hashes = json.loads(input_hashes_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_hashes, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in loaded_hashes.items()
        ):
            raise ValueError(f"Invalid audit input hash manifest: {input_hashes_path}")
        input_hashes = loaded_hashes
        if input_hashes != current_input_hashes:
            raise ValueError("Paper inputs changed after the read-only audit snapshot was prepared")
    else:
        input_hashes = current_input_hashes
    audit_payload = audit.model_dump(mode="json")
    audit_payload["finding_count"] = len(audit.findings)
    audit_payload["input_sha256"] = input_hashes
    markdown_output_path.write_text(render_audit_markdown(audit), encoding="utf-8")
    write_audit_feedback(
        audit,
        audit_json_path=audit_json_path,
        audit_markdown_path=markdown_output_path,
        feedback_json_path=feedback_json_path,
        feedback_md_path=feedback_md_path,
        raw_output_path=raw_output_path,
    )
    _merge_audit_into_validation_report(validation_report_path, audit)
    report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    report["agentic_evidence_audit"]["input_sha256"] = input_hashes
    validation_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _atomic_write_text(
        audit_json_path,
        json.dumps(audit_payload, indent=2, sort_keys=True),
    )
    return audit.verdict == "fail"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper evidence-audit helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    packet_parser = subparsers.add_parser(
        "build-packet", help="Build paper_evidence_audit_packet.md"
    )
    packet_parser.add_argument("--paper-dir", required=True, help="Paper workspace directory")
    packet_parser.add_argument("--output", required=True, help="Output packet path")

    readonly_parser = subparsers.add_parser(
        "prepare-readonly-workspace",
        help="Copy a paper workspace to a read-only directory for the evidence audit agent",
    )
    readonly_parser.add_argument(
        "--source-dir", required=True, help="Source paper workspace directory"
    )
    readonly_parser.add_argument(
        "--output-dir", required=True, help="Destination read-only workspace directory"
    )
    readonly_parser.add_argument(
        "--refresh-evidence-script",
        help="Optional path to scientific_evidence.py; when provided, the copied workspace gets a fresh verification-report.json before being marked read-only",
    )
    readonly_parser.add_argument(
        "--packet-output",
        help="Optional packet path inside the copied workspace; when provided, build-packet runs against the refreshed copy before it becomes read-only",
    )

    audit_workspace_parser = subparsers.add_parser(
        "prepare-audit-workspace",
        help="Create a managed read-only evidence-audit workspace beside a paper output directory",
    )
    audit_workspace_parser.add_argument(
        "--source-dir", required=True, help="Source paper workspace directory"
    )
    audit_workspace_parser.add_argument(
        "--workspace-tag", required=True, help="Safe label included in the temporary directory name"
    )
    audit_workspace_parser.add_argument(
        "--refresh-evidence-script",
        help="Optional path to scientific_evidence.py used to refresh verification-report.json",
    )

    cleanup_parser = subparsers.add_parser(
        "cleanup-readonly-workspace",
        help="Remove a managed read-only evidence-audit workspace",
    )
    cleanup_parser.add_argument(
        "--path", required=True, help="Read-only workspace directory to remove"
    )
    cleanup_parser.add_argument(
        "--source-dir", required=True, help="Paper output directory that owns the workspace"
    )

    finalize_parser = subparsers.add_parser(
        "finalize-audit",
        help="Parse the audit log, validate the output contract, and derive audit artifacts",
    )
    finalize_parser.add_argument("--log", required=True, help="Path to the agent stream-json log")
    finalize_parser.add_argument(
        "--raw-output", required=True, help="Path to extracted raw audit output"
    )
    finalize_parser.add_argument("--audit-json", required=True, help="Path to parsed audit JSON")
    finalize_parser.add_argument("--markdown-output", required=True, help="Path to audit markdown")
    finalize_parser.add_argument(
        "--feedback-json", required=True, help="Path to audit feedback JSON"
    )
    finalize_parser.add_argument(
        "--feedback-md", required=True, help="Path to audit feedback markdown"
    )
    finalize_parser.add_argument(
        "--validation-report",
        required=True,
        help="Path to validation_report.json that will receive the merged audit block",
    )
    finalize_parser.add_argument(
        "--input-hashes",
        help="Hash manifest copied from the exact read-only audit snapshot",
    )
    finalize_parser.add_argument(
        "--refresh-evidence-script",
        help="Refresh verification-report.json in the source workspace before comparing hashes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-packet":
            packet_path = build_audit_packet(Path(args.paper_dir), output_path=Path(args.output))
            print(packet_path)
            return 0
        if args.command == "prepare-readonly-workspace":
            readonly_dir = prepare_readonly_workspace(
                Path(args.source_dir),
                Path(args.output_dir),
                evidence_script_path=Path(args.refresh_evidence_script)
                if args.refresh_evidence_script
                else None,
                packet_output_path=Path(args.packet_output) if args.packet_output else None,
            )
            print(readonly_dir)
            return 0
        if args.command == "prepare-audit-workspace":
            readonly_dir = prepare_audit_workspace(
                Path(args.source_dir),
                workspace_tag=args.workspace_tag,
                evidence_script_path=Path(args.refresh_evidence_script)
                if args.refresh_evidence_script
                else None,
            )
            print(readonly_dir)
            return 0
        if args.command == "cleanup-readonly-workspace":
            cleanup_audit_workspace(Path(args.path), source_dir=Path(args.source_dir))
            return 0
        if args.command == "finalize-audit":
            found_mismatch = finalize_audit(
                log_path=Path(args.log),
                raw_output_path=Path(args.raw_output),
                audit_json_path=Path(args.audit_json),
                markdown_output_path=Path(args.markdown_output),
                feedback_json_path=Path(args.feedback_json),
                feedback_md_path=Path(args.feedback_md),
                validation_report_path=Path(args.validation_report),
                input_hashes_path=Path(args.input_hashes) if args.input_hashes else None,
                evidence_script_path=Path(args.refresh_evidence_script)
                if args.refresh_evidence_script
                else None,
            )
            return 2 if found_mismatch else 0
    except AuditContractError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
