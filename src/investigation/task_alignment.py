"""Validation for independent agent reviews of drafted task scaffolds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .program_spec import ProgramSpecError, load_program_spec, resolve_entrypoint
from .task_contract import TaskContractError, task_scaffold_fingerprint

REQUIRED_ALIGNMENT_DIMENSIONS = (
    "goal_and_method_fidelity",
    "upstream_grounding",
    "search_proxy_fidelity",
    "final_evaluation_fidelity",
    "search_final_isolation",
    "metrics_and_rubrics",
    "candidate_editability",
    "compute_realism",
    "execution_feasibility",
    "reproducibility",
    "publication_scope",
)

FINGERPRINTED_FILES = (
    "task.md",
    "capture.py",
    "task_contract.json",
    "evaluate.py",
    "evaluate_final.py",
    "program_0/program_spec.toml",
)
_HARNESS_RUNTIME_ROOT_ENTRIES = frozenset(
    {".venv", ".agentic_runtime.json", ".agentic_repair_prompt.md"}
)
_MAX_FINGERPRINT_ENTRIES = 100_000
_MAX_FINGERPRINT_BYTES = 64 * 1024 * 1024 * 1024
_FINGERPRINT_CHUNK_BYTES = 1024 * 1024


class TaskAlignmentError(ValueError):
    """Raised when an agentic task-alignment report is malformed or stale."""


@dataclass(frozen=True)
class TaskAlignmentReport:
    """Validated task-alignment decision."""

    verdict: str
    summary: str
    failed_dimensions: tuple[str, ...]
    payload: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_components(path: Path, root: Path) -> None:
    current = path
    while current != root:
        if current.is_symlink():
            raise TaskAlignmentError(f"scaffold contains a symlink: {current.relative_to(root)}")
        current = current.parent


def task_alignment_fingerprints(scaffold_dir: Path) -> dict[str, str]:
    """Fingerprint the task inputs whose semantics the judge reviewed."""
    fingerprints: dict[str, str] = {}
    for relative in FINGERPRINTED_FILES:
        path = scaffold_dir / relative
        if path.is_symlink() or not path.is_file():
            raise TaskAlignmentError(f"missing safe scaffold file for alignment: {relative}")
        fingerprints[relative] = _sha256(path)
    try:
        program_dir = scaffold_dir / "program_0"
        spec = load_program_spec(program_dir, required=True)
        assert spec is not None
        declared_entrypoint = program_dir / spec.code_root / spec.entrypoint.path
        _reject_symlink_components(declared_entrypoint, scaffold_dir)
        entrypoint = resolve_entrypoint(program_dir, spec)
        relative_entrypoint = entrypoint.relative_to(scaffold_dir.resolve()).as_posix()
    except (ProgramSpecError, ValueError) as exc:
        raise TaskAlignmentError(f"invalid candidate program specification: {exc}") from exc
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise TaskAlignmentError(
            f"missing safe scaffold file for alignment: {relative_entrypoint}"
        )
    fingerprints[relative_entrypoint] = _sha256(entrypoint)
    try:
        fingerprints["declared_scaffold"] = task_scaffold_fingerprint(scaffold_dir)
    except TaskContractError as exc:
        raise TaskAlignmentError(f"invalid declared scaffold: {exc}") from exc
    return fingerprints


def task_tree_fingerprint(
    scaffold_dir: Path, *, deadline: float | None = None,
    omitted_root_entries: frozenset[str] = frozenset(),
) -> str:
    """Hash every regular scaffold entry and reject indirect or special files."""
    if scaffold_dir.is_symlink() or not scaffold_dir.is_dir():
        raise TaskAlignmentError(f"unsafe scaffold directory: {scaffold_dir}")
    digest = hashlib.sha256()
    entry_count = 0
    total_bytes = 0

    def check_deadline() -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TaskAlignmentError("scaffold fingerprint deadline exceeded")

    def visit(directory: Path) -> None:
        nonlocal entry_count, total_bytes
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            check_deadline()
            entry_count += 1
            if entry_count > _MAX_FINGERPRINT_ENTRIES:
                raise TaskAlignmentError("scaffold exceeds fingerprint entry limit")
            relative_path = path.relative_to(scaffold_dir)
            relative = relative_path.as_posix()
            if len(relative_path.parts) == 1 and path.name in (
                _HARNESS_RUNTIME_ROOT_ENTRIES | omitted_root_entries
            ):
                continue
            if path.is_symlink():
                raise TaskAlignmentError(f"scaffold contains a symlink: {relative}")
            if path.is_dir():
                digest.update(b"dir\0" + relative.encode() + b"\0")
                visit(path)
                continue
            if not path.is_file():
                raise TaskAlignmentError(f"scaffold contains a special file: {relative}")
            size = path.stat(follow_symlinks=False).st_size
            total_bytes += size
            if total_bytes > _MAX_FINGERPRINT_BYTES:
                raise TaskAlignmentError("scaffold exceeds fingerprint byte limit")
            digest.update(
                b"file\0"
                + relative.encode()
                + b"\0"
                + str(size).encode()
                + b"\0"
            )
            observed_size = 0
            with path.open("rb") as handle:
                while chunk := handle.read(_FINGERPRINT_CHUNK_BYTES):
                    check_deadline()
                    observed_size += len(chunk)
                    digest.update(chunk)
            if observed_size != size:
                raise TaskAlignmentError(f"scaffold file changed while hashing: {relative}")

    visit(scaffold_dir)
    return "sha256:" + digest.hexdigest()


def _require_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskAlignmentError(f"{field} must be an object")
    return value


def _require_fields(value: dict[str, Any], field: str, expected: set[str]) -> None:
    missing = expected - value.keys()
    unexpected = value.keys() - expected
    if missing:
        raise TaskAlignmentError(f"{field} missing field(s): {', '.join(sorted(missing))}")
    if unexpected:
        raise TaskAlignmentError(
            f"{field} has unexpected field(s): {', '.join(sorted(unexpected))}"
        )


def _safe_evidence_path(raw: object, scaffold_dir: Path) -> tuple[str, Path]:
    value = str(raw).strip()
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or value.endswith("/"):
        raise TaskAlignmentError("evidence path must be a safe relative file")
    target = scaffold_dir.joinpath(*path.parts)
    if target.is_symlink() or not target.is_file():
        raise TaskAlignmentError("evidence path must be a safe relative file")
    return value, target


def _validate_evidence(value: object, dimension: str, scaffold_dir: Path) -> None:
    if not isinstance(value, list) or not value:
        raise TaskAlignmentError(f"dimension {dimension} requires non-empty evidence")
    for index, raw in enumerate(value):
        evidence = _require_object(raw, f"dimension {dimension} evidence[{index}]")
        _require_fields(
            evidence, f"dimension {dimension} evidence[{index}]", {"path", "line", "detail"}
        )
        _, target = _safe_evidence_path(evidence.get("path"), scaffold_dir)
        line = evidence.get("line")
        if type(line) is not int or line < 1:
            raise TaskAlignmentError(f"dimension {dimension} evidence line must be >= 1")
        try:
            line_count = len(target.read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeError) as exc:
            raise TaskAlignmentError("evidence path must be a readable UTF-8 file") from exc
        if line > line_count:
            raise TaskAlignmentError(f"dimension {dimension} evidence line exceeds file length")
        if not str(evidence.get("detail", "")).strip():
            raise TaskAlignmentError(f"dimension {dimension} evidence detail is required")


def _validate_dimensions(value: object, scaffold_dir: Path) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TaskAlignmentError("dimensions must be a list")
    rows: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        row = _require_object(raw, f"dimensions[{index}]")
        _require_fields(row, f"dimensions[{index}]", {"id", "verdict", "evidence", "remediation"})
        identifier = str(row.get("id", "")).strip()
        if identifier in rows:
            raise TaskAlignmentError(f"duplicate alignment dimension: {identifier}")
        rows[identifier] = row
    if set(rows) != set(REQUIRED_ALIGNMENT_DIMENSIONS):
        raise TaskAlignmentError(
            "dimensions must contain exactly the required alignment dimensions"
        )

    failed: list[str] = []
    for identifier in REQUIRED_ALIGNMENT_DIMENSIONS:
        row = rows[identifier]
        verdict = str(row.get("verdict", "")).strip()
        if verdict not in {"pass", "fail"}:
            raise TaskAlignmentError(f"dimension {identifier} verdict must be pass or fail")
        _validate_evidence(row.get("evidence"), identifier, scaffold_dir)
        remediation = str(row.get("remediation", "")).strip()
        if verdict == "fail":
            failed.append(identifier)
            if not remediation:
                raise TaskAlignmentError(f"failed dimension {identifier} requires remediation")
        elif remediation:
            raise TaskAlignmentError(f"passed dimension {identifier} must not include remediation")
    return tuple(failed)


def load_task_alignment_report(
    report_path: Path,
    *,
    scaffold_dir: Path,
    evidence_scaffold_dir: Path | None = None,
) -> TaskAlignmentReport:
    """Load a strict, current alignment report and derive its failed dimensions."""
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskAlignmentError(f"invalid task alignment report: {report_path}") from exc
    report = _require_object(payload, "task alignment report")
    _require_fields(
        report,
        "task alignment report",
        {"schema_version", "verdict", "summary", "fingerprints", "dimensions"},
    )
    if report.get("schema_version") != 1:
        raise TaskAlignmentError("task alignment report schema_version must be 1")
    summary = " ".join(str(report.get("summary", "")).split())
    if not summary:
        raise TaskAlignmentError("task alignment report requires a summary")
    verdict = str(report.get("verdict", "")).strip()
    if verdict not in {"pass", "fail"}:
        raise TaskAlignmentError("task alignment report verdict must be pass or fail")
    fingerprints = _require_object(report.get("fingerprints"), "fingerprints")
    if fingerprints != task_alignment_fingerprints(scaffold_dir):
        raise TaskAlignmentError("task alignment report fingerprints do not match scaffold")
    failed = _validate_dimensions(
        report.get("dimensions"), evidence_scaffold_dir or scaffold_dir
    )
    expected_verdict = "fail" if failed else "pass"
    if verdict != expected_verdict:
        raise TaskAlignmentError("top-level verdict does not match dimension verdicts")
    return TaskAlignmentReport(
        verdict=verdict,
        summary=summary,
        failed_dimensions=failed,
        payload=report,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate drafted-task alignment reports")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fingerprints_parser = subparsers.add_parser("fingerprints")
    fingerprints_parser.add_argument("--scaffold", type=Path, required=True)
    tree_parser = subparsers.add_parser("tree-fingerprint")
    tree_parser.add_argument("--scaffold", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--scaffold", type=Path, required=True)
    validate_parser.add_argument("--evidence-scaffold", type=Path)
    validate_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command in {"fingerprints", "tree-fingerprint"}:
        try:
            if args.command == "fingerprints":
                result = json.dumps(task_alignment_fingerprints(args.scaffold), sort_keys=True)
            else:
                result = task_tree_fingerprint(args.scaffold)
        except (OSError, TaskAlignmentError) as exc:
            print(f"invalid scaffold: {exc}", file=sys.stderr)
            return 2
        print(result)
        return 0
    try:
        report = load_task_alignment_report(
            args.report,
            scaffold_dir=args.scaffold,
            evidence_scaffold_dir=args.evidence_scaffold,
        )
    except (OSError, TaskAlignmentError) as exc:
        print(f"invalid alignment report: {exc}")
        return 2
    print(json.dumps({"verdict": report.verdict, "failed": report.failed_dimensions}))
    return 0 if report.verdict == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(_main())
