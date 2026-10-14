"""Fail-fast lint for retired shared-filesystem RPC names."""

from __future__ import annotations

import ast
import json
import os
import stat
from pathlib import Path

from src.investigation.program_spec import entrypoint_relpath_from_spec


_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_FILES = 1024
_MAX_TREE_ENTRIES = 4096
_MAX_TREE_DEPTH = 64
_MAX_FINDINGS = 64
_RESERVED_NAMES = frozenset({"EXCHANGE_DIR", "AGENTIC_CANDIDATE_EXCHANGE"})


class _TraversalLimit(ValueError):
    pass


def _display_path(relative: str) -> str:
    return json.dumps(relative, ensure_ascii=True)[1:-1]


def _source_finding(relative: str, line: int, name: str) -> str:
    return (
        f"{_display_path(relative)}:{line}: isolated candidate RPC cannot use `{name}` because "
        "evaluator and candidate share no filesystem; send finite inline JSON, using bounded "
        "chunks below the 64 MiB message limit when needed"
    )


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


def _parse_source(source: bytes, relative: str) -> ast.AST:
    return ast.parse(source, filename=relative)


def _reserved_occurrences(
    source: bytes, relative: str
) -> tuple[tuple[int, str], ...] | None:
    try:
        tree = _parse_source(source, relative)
    except (SyntaxError, UnicodeError, ValueError, RecursionError):
        return None
    docstrings = _docstring_nodes(tree)
    occurrences: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        name: str | None = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.arg):
            name = node.arg
        elif isinstance(node, ast.alias):
            name = node.name
        elif isinstance(node, ast.keyword):
            name = node.arg
        elif isinstance(node, ast.Constant) and id(node) not in docstrings:
            if isinstance(node.value, str):
                name = node.value
            elif isinstance(node.value, bytes):
                try:
                    name = node.value.decode("ascii")
                except UnicodeDecodeError:
                    pass
        if name in _RESERVED_NAMES:
            occurrences.add((getattr(node, "lineno", 1), name))
    return tuple(sorted(occurrences))


def _read_source(
    parent_fd: int,
    name: str,
    metadata: os.stat_result,
    remaining_bytes: int,
) -> tuple[bytes | None, str | None]:
    try:
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError:
        return None, "isolated RPC source is unreadable"
    try:
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            return None, "isolated RPC source changed during validation"
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(file_fd, min(1024 * 1024, remaining_bytes + 1 - observed)):
            chunks.append(chunk)
            observed += len(chunk)
            if observed > remaining_bytes:
                return None, "source_bytes_exceeded"
        if os.fstat(file_fd).st_size != opened.st_size:
            return None, "isolated RPC source changed during validation"
        return b"".join(chunks), None
    finally:
        os.close(file_fd)


def isolated_rpc_transport_findings(task_dir: Path) -> tuple[str, ...]:
    """Return bounded findings for retired shared-filesystem RPC names."""
    findings: list[str] = []
    source_files = 0
    total_bytes = 0
    tree_entries = 0
    declared_entrypoint = entrypoint_relpath_from_spec(task_dir / "program_0", required=False)
    declared_source = f"program_0/{declared_entrypoint}" if declared_entrypoint else None

    def add(message: str) -> None:
        if len(findings) < _MAX_FINDINGS:
            findings.append(message)

    def inspect_source(
        parent_fd: int,
        name: str,
        relative: str,
        metadata: os.stat_result,
    ) -> None:
        nonlocal source_files, total_bytes
        source_files += 1
        if source_files > _MAX_SOURCE_FILES:
            raise _TraversalLimit(
                f"isolated RPC source contains more than {_MAX_SOURCE_FILES} Python files"
            )
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            add(
                f"{_display_path(relative)}: isolated RPC source must be a regular "
                "non-symlink file"
            )
            return
        aggregate_remaining = _MAX_SOURCE_BYTES - total_bytes
        read_limit = min(_MAX_SOURCE_FILE_BYTES, aggregate_remaining)
        source, error = _read_source(
            parent_fd,
            name,
            metadata,
            read_limit,
        )
        if error == "source_bytes_exceeded":
            if _MAX_SOURCE_FILE_BYTES <= aggregate_remaining:
                raise _TraversalLimit(
                    f"{_display_path(relative)}: isolated RPC source exceeds the "
                    f"{_MAX_SOURCE_FILE_BYTES} byte per-file validation limit"
                )
            raise _TraversalLimit(
                f"isolated RPC source exceeds the {_MAX_SOURCE_BYTES} byte validation limit"
            )
        if error:
            add(f"{_display_path(relative)}: {error}")
            return
        assert source is not None
        total_bytes += len(source)
        occurrences = _reserved_occurrences(source, relative)
        if occurrences is None:
            add(f"{_display_path(relative)}: isolated RPC source could not be parsed safely")
            return
        for line, reserved_name in occurrences:
            add(_source_finding(relative, line, reserved_name))
            if len(findings) == _MAX_FINDINGS:
                return

    def visit_program(directory_fd: int, prefix: str, depth: int) -> None:
        nonlocal tree_entries
        if len(findings) == _MAX_FINDINGS:
            return
        if depth > _MAX_TREE_DEPTH:
            raise _TraversalLimit(
                f"isolated RPC source tree exceeds the {_MAX_TREE_DEPTH} directory depth limit"
            )
        names: list[str] = []
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                tree_entries += 1
                if tree_entries > _MAX_TREE_ENTRIES:
                    raise _TraversalLimit(
                        f"isolated RPC source tree contains more than {_MAX_TREE_ENTRIES} entries"
                    )
                names.append(entry.name)
        for name in sorted(names):
            if len(findings) == _MAX_FINDINGS:
                return
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = f"{prefix}/{name}"
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(child_fd)
                    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                        raise _TraversalLimit(
                            "isolated RPC source tree changed during validation"
                        )
                    visit_program(child_fd, relative, depth + 1)
                finally:
                    os.close(child_fd)
            elif relative == declared_source or name.endswith(".py"):
                inspect_source(directory_fd, name, relative, metadata)

    try:
        root_fd = os.open(task_dir.absolute(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return ("isolated RPC task root is unreadable",)
    try:
        for evaluator in ("evaluate.py", "evaluate_final.py"):
            try:
                metadata = os.stat(evaluator, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            inspect_source(root_fd, evaluator, evaluator, metadata)
        try:
            metadata = os.stat("program_0", dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return tuple(findings)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            add("program_0: isolated RPC source root must be a regular non-symlink directory")
            return tuple(findings)
        program_fd = os.open(
            "program_0",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        try:
            opened = os.fstat(program_fd)
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise _TraversalLimit("isolated RPC source tree changed during validation")
            visit_program(program_fd, "program_0", 1)
        finally:
            os.close(program_fd)
    except _TraversalLimit as exc:
        return (str(exc),)
    except OSError:
        return ("isolated RPC source tree is unreadable",)
    finally:
        os.close(root_fd)
    return tuple(findings)
