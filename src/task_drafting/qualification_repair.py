"""Qualification repair failure history and oscillation detection."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from src.task_drafting.qualification_calibration import file_sha256

_HEX_DIGITS = frozenset("0123456789abcdef")


class QualificationRepairOscillation(ValueError):
    """Raised when a repair reproduces a scaffold already known to fail."""


def _is_fingerprint(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in _HEX_DIGITS for character in value.removeprefix("sha256:"))
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"state must be a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("state must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, object], *, create: bool = False) -> None:
    if path.is_symlink() or (create and path.exists()):
        raise ValueError(f"refusing unsafe state path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_history(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"schema_version": 1, "failures": []}
    payload = _read_json(path)
    failures = payload.get("failures")
    if (
        set(payload) != {"schema_version", "failures"}
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or not isinstance(failures, list)
    ):
        raise ValueError("qualification repair history is invalid")
    for failure in failures:
        if not (
            isinstance(failure, dict)
            and set(failure) == {"category", "evidence_sha256", "fingerprint"}
            and isinstance(failure["category"], str)
            and bool(failure["category"])
            and len(failure["category"].encode("utf-8")) <= 256
            and _is_sha256(failure["evidence_sha256"])
            and _is_fingerprint(failure["fingerprint"])
        ):
            raise ValueError("qualification repair history entry is invalid")
    return payload


def record_failed_fingerprint(
    history_path: Path,
    *,
    fingerprint: str,
    category: str,
    evidence_sha256: str,
) -> None:
    """Retain a unique failed scaffold fingerprint and its evidence digest."""
    if not _is_fingerprint(fingerprint):
        raise ValueError("fingerprint must use sha256:<64 lowercase hex>")
    if not _is_sha256(evidence_sha256):
        raise ValueError("evidence_sha256 must be a lowercase SHA-256")
    if not category or len(category.encode("utf-8")) > 256:
        raise ValueError("category must be a short non-empty string")
    payload = _load_history(history_path)
    failures = payload["failures"]
    assert isinstance(failures, list)
    if any(item["fingerprint"] == fingerprint for item in failures):
        return
    failures.append(
        {
            "category": category,
            "evidence_sha256": evidence_sha256,
            "fingerprint": fingerprint,
        }
    )
    _write_json(history_path, payload)


def reject_repeated_candidate(
    history_path: Path,
    *,
    fingerprint: str,
    error_report: Path,
) -> None:
    """Emit a typed failure when a candidate repeats a known failed state."""
    if not _is_fingerprint(fingerprint):
        raise ValueError("fingerprint must use sha256:<64 lowercase hex>")
    payload = _load_history(history_path)
    failures = payload["failures"]
    assert isinstance(failures, list)
    if not any(item["fingerprint"] == fingerprint for item in failures):
        return
    report: dict[str, object] = {
        "category": "qualification_repair_oscillation",
        "diagnostic": (
            "Qualification repair reproduced a previously failed scaffold fingerprint."
        ),
        "elapsed_seconds": 0.0,
        "receipt_path": None,
        "schema_version": 1,
        "stage": "qualification-repair",
        "status": "authored_failure",
        "task_fingerprint": fingerprint,
    }
    _write_json(error_report, report, create=True)
    raise QualificationRepairOscillation(report["diagnostic"])


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record-failure")
    record.add_argument("--history", type=Path, required=True)
    record.add_argument("--fingerprint", required=True)
    record.add_argument("--category", required=True)
    record.add_argument("--evidence", type=Path, required=True)
    check = subparsers.add_parser("check-candidate")
    check.add_argument("--history", type=Path, required=True)
    check.add_argument("--fingerprint", required=True)
    check.add_argument("--error-report", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "record-failure":
            record_failed_fingerprint(
                args.history,
                fingerprint=args.fingerprint,
                category=args.category,
                evidence_sha256=file_sha256(args.evidence),
            )
        else:
            reject_repeated_candidate(
                args.history,
                fingerprint=args.fingerprint,
                error_report=args.error_report,
            )
    except QualificationRepairOscillation as exc:
        print(str(exc))
        return 3
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
