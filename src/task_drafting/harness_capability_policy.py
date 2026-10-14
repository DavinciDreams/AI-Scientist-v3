"""Fail fast when drafted evaluators depend on unavailable harness inputs."""

from __future__ import annotations

import ast
import os
import stat
from dataclasses import dataclass
from pathlib import Path


_MAX_EVALUATOR_BYTES = 2 * 1024 * 1024
_RESERVED_HARNESS_PREFIX = "AGENTIC_HARNESS_"
_UNAVAILABLE_PROXY_ACCESSORS = frozenset({"candidate_peak_accelerator_memory_gib"})


@dataclass(frozen=True)
class HarnessCapabilityFinding:
    component: str
    diagnostic: str


def unavailable_harness_environment_findings(
    task_dir: Path,
    evaluators: tuple[tuple[str, str], ...],
) -> tuple[HarnessCapabilityFinding, ...]:
    """Return findings for reserved environment inputs and unsupported proxy accessors."""
    findings: list[HarnessCapabilityFinding] = []
    root_fd = os.open(task_dir.absolute(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component, relative in evaluators:
            source = _read_regular_source(root_fd, relative)
            if source is None:
                continue
            for line, name in _reserved_environment_occurrences(source, relative):
                findings.append(
                    HarnessCapabilityFinding(
                        component,
                        f"{relative}:{line}: `{name}` is reserved for harness-owned data but "
                        "is not supplied to task evaluators; correctness and metrics must use "
                        "evaluator-owned inputs",
                    )
                )
    finally:
        os.close(root_fd)
    return tuple(findings)


def _read_regular_source(root_fd: int, relative: str) -> bytes | None:
    try:
        file_fd = os.open(relative, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    except OSError:
        return None
    try:
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_EVALUATOR_BYTES:
            return None
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(file_fd, min(1024 * 1024, _MAX_EVALUATOR_BYTES + 1 - observed)):
            chunks.append(chunk)
            observed += len(chunk)
            if observed > _MAX_EVALUATOR_BYTES:
                return None
        return b"".join(chunks)
    finally:
        os.close(file_fd)


def _reserved_environment_occurrences(
    source: bytes, relative: str
) -> tuple[tuple[int, str], ...]:
    try:
        tree = ast.parse(source, filename=relative)
    except (SyntaxError, UnicodeError, ValueError, RecursionError):
        return ()
    docstrings = _docstring_nodes(tree)
    occurrences = {
        (getattr(node, "lineno", 1), node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and id(node) not in docstrings
        and isinstance(node.value, str)
        and node.value.startswith(_RESERVED_HARNESS_PREFIX)
    }
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            name = node.args[1].value
        if name in _UNAVAILABLE_PROXY_ACCESSORS:
            occurrences.add((getattr(node, "lineno", 1), name))
    return tuple(sorted(occurrences))


def _docstring_nodes(tree: ast.AST) -> set[int]:
    nodes: set[int] = set()
    for owner in ast.walk(tree):
        body = getattr(owner, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            nodes.add(id(first.value))
    return nodes
