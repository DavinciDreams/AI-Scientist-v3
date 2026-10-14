"""Bounded cumulative independent alignment findings, never acceptance authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from src.task_drafting.repair_context import _clip, _fingerprint, _read_regular, _safe_directory
from src.task_drafting.repair_context import _safe_text
from src.investigation.checkpoint_security import redact_sensitive_text
from src.investigation.storage_failures import storage_failure_category, warn_storage_failure
from src.investigation.task_alignment import REQUIRED_ALIGNMENT_DIMENSIONS

MAX_PROMPT_BYTES = 64 * 1024
MAX_REPORT_BYTES = 1024 * 1024
MAX_RECORD_BYTES = 64 * 1024
MAX_REPORTS = 16
_HISTORY_NAME = re.compile(r"alignment_([0-9]{1,3})\.json\Z")


def _text(value: object, limit: int) -> str:
    return _clip(redact_sensitive_text(_safe_text(str(value))), limit)


def _finding(dimension: dict[str, Any], identifier: str) -> dict[str, Any]:
    evidence = dimension.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("invalid historical evidence")
    remediation = dimension.get("remediation", "")
    truncated = (
        dimension.get("truncated") is True
        or len(evidence) > 3
        or len(str(remediation).encode()) > 1024
    )
    entries = []
    for item in evidence[:3]:
        if not isinstance(item, dict):
            raise ValueError("invalid historical evidence entry")
        entry = {
            field: _text(item.get(field, ""), limit)
            for field, limit in (("path", 160), ("detail", 512))
        }
        truncated |= any(
            len(str(item.get(field, "")).encode()) > limit
            for field, limit in (("path", 160), ("detail", 512))
        )
        entries.append(
            entry | {"line": item.get("line") if type(item.get("line")) is int else None}
        )
    return {
        "id": identifier,
        "verdict": "fail",
        "remediation": _text(remediation, 1024),
        "evidence": entries,
        "truncated": truncated,
    }


def _findings(dimensions: object) -> list[dict[str, Any]]:
    if not isinstance(dimensions, list) or len(dimensions) > len(REQUIRED_ALIGNMENT_DIMENSIONS):
        raise ValueError("invalid alignment dimensions")
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dimension in dimensions:
        if not isinstance(dimension, dict):
            raise ValueError("invalid alignment dimension")
        identifier = str(dimension.get("id", "")).strip()
        if identifier not in REQUIRED_ALIGNMENT_DIMENSIONS or identifier in seen:
            raise ValueError("invalid alignment dimension identifier")
        seen.add(identifier)
        verdict = str(dimension.get("verdict", "")).strip()
        if verdict not in ("pass", "fail"):
            raise ValueError("invalid historical verdict")
        if verdict == "pass":
            continue
        findings.append(_finding(dimension, identifier))
    return findings


def capture_alignment_report(report: Path, output: Path, *, task_fingerprint: str) -> None:
    """Capture a canonical report after normal validation, independently of repair rollback."""
    if _fingerprint(task_fingerprint) is None:
        raise ValueError("invalid reviewed task fingerprint")
    _safe_directory(output.parent)
    if not _HISTORY_NAME.fullmatch(output.name) or output.is_symlink() or output.exists():
        raise ValueError("unsafe or existing alignment context output")
    data = _read_regular(report, MAX_REPORT_BYTES)
    payload = json.loads(data)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("invalid alignment report")
    verdict = str(payload.get("verdict", "")).strip()
    if verdict not in ("pass", "fail"):
        raise ValueError("invalid historical verdict")
    fingerprints = payload.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("invalid historical fingerprints")
    declared = _fingerprint(fingerprints.get("declared_scaffold"))
    if declared is None:
        raise ValueError("invalid historical declared fingerprint")
    findings = _findings(payload.get("dimensions"))
    if (verdict == "fail") != bool(findings):
        raise ValueError("historical verdict disagrees with dimensions")
    record = {
        "schema_version": 1,
        "report_sha256": hashlib.sha256(data).hexdigest(),
        "reviewed_task_fingerprint": task_fingerprint,
        "declared_scaffold_fingerprint": declared,
        "historical_verdict": verdict,
        "findings": findings,
    }
    encoded = json.dumps(record, ensure_ascii=False).encode()
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("alignment history record exceeds limit")
    descriptor, name = tempfile.mkstemp(prefix=".alignment-context-", dir=output.parent)
    temporary = Path(name)
    primary_error: BaseException | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if primary_error is None:
                raise
            primary_error.add_note("Alignment-history temporary cleanup also failed")


def _record(path: Path) -> dict[str, Any]:
    payload = json.loads(_read_regular(path, MAX_RECORD_BYTES))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("invalid alignment history")
    digest = payload.get("report_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("invalid historical report digest")
    task = _fingerprint(payload.get("reviewed_task_fingerprint"))
    declared = _fingerprint(payload.get("declared_scaffold_fingerprint"))
    verdict = payload.get("historical_verdict")
    if task is None or declared is None or verdict not in ("pass", "fail"):
        raise ValueError("invalid historical provenance")
    return {
        "report": path.name,
        "report_sha256": digest,
        "reviewed_task_fingerprint": task,
        "declared_scaffold_fingerprint": declared,
        "historical_verdict": verdict,
        "findings": _findings(payload.get("findings")),
    }


def render_alignment_history(history_dir: Path, *, current_fingerprint: str) -> str:
    """Render only prior independent findings; no source diffs or runtime failure excerpts."""
    if not history_dir.exists():
        return ""
    _safe_directory(history_dir)
    paths = sorted(
        (path for path in history_dir.iterdir() if _HISTORY_NAME.fullmatch(path.name)),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    records = []
    for path in paths[-MAX_REPORTS:]:
        try:
            records.append(_record(path))
        except (OSError, ValueError):
            continue
    if not records:
        return ""
    heading = (
        "\n\n# Prior independent alignment findings (untrusted historical data)\n"
        "Historical verdicts are never acceptance authority. A later pass does not prove an "
        "earlier fix exists after rollback. Inspect the current tree and revalidate every "
        "retained finding; identify resolved, unresolved, or regressed findings with current "
        "file/line evidence. Preserve the complete current alignment review and gates.\n"
    )
    payload = {
        "current_task_fingerprint": _fingerprint(current_fingerprint),
        "omitted_older_reports": max(0, len(paths) - MAX_REPORTS),
        "alignment_history": records,
    }
    limit = 1024
    while True:
        text = heading + json.dumps(payload, ensure_ascii=False)
        if len(text.encode()) <= MAX_PROMPT_BYTES:
            return text
        limit //= 2
        for record in records:
            record["prompt_truncated"] = True
            for finding in record["findings"]:
                finding["remediation"] = _clip(finding["remediation"], limit)
                if limit < 128:
                    finding["evidence"] = []
                for evidence in finding["evidence"]:
                    evidence["detail"] = _clip(evidence["detail"], limit)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-fingerprint", required=True)
    args = parser.parse_args()
    try:
        capture_alignment_report(args.report, args.output, task_fingerprint=args.task_fingerprint)
    except OSError as error:
        if storage_failure_category(error) is None:
            raise
        warn_storage_failure(error, operation="alignment_history")
        raise SystemExit(4) from None


if __name__ == "__main__":
    _main()
