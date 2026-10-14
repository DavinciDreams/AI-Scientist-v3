"""Harness-owned evidence for qualification-driven task calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_HEX_DIGITS = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _require_sha256(value: str, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_fingerprint(value: str, name: str) -> str:
    if not value.startswith("sha256:") or not _is_sha256(value.removeprefix("sha256:")):
        raise ValueError(f"{name} must be a sha256: task fingerprint")
    return value


def file_sha256(path: Path) -> str:
    """Hash one regular, non-symlink evidence file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def create_calibration_amendment(
    path: Path,
    *,
    original_fingerprint: str,
    calibrated_fingerprint: str,
    accepted_design_sha256: str,
    qualification_failure_sha256: str,
) -> None:
    """Create an immutable-design calibration record for a provisional repair."""
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "status": "provisional",
        "original_design_immutable": True,
        "original_fingerprint": _require_fingerprint(
            original_fingerprint, "original_fingerprint"
        ),
        "calibrated_fingerprint": _require_fingerprint(
            calibrated_fingerprint, "calibrated_fingerprint"
        ),
        "evidence_sha256": {
            "accepted_design": _require_sha256(
                accepted_design_sha256, "accepted_design_sha256"
            ),
            "qualification_failure": _require_sha256(
                qualification_failure_sha256, "qualification_failure_sha256"
            ),
        },
    }
    _write_json(path, payload, create=True)


def accept_calibration_amendment(
    path: Path,
    *,
    alignment_report_sha256: str,
    qualification_report_sha256: str,
    qualification_receipt_sha256: str,
) -> None:
    """Bind a provisional calibration to both successful downstream gates."""
    payload = _read_json(path)
    evidence = payload.get("evidence_sha256")
    if (
        payload.get("schema_version") != _SCHEMA_VERSION
        or payload.get("status") != "provisional"
        or payload.get("original_design_immutable") is not True
        or not isinstance(payload.get("original_fingerprint"), str)
        or not isinstance(payload.get("calibrated_fingerprint"), str)
        or not payload["original_fingerprint"].startswith("sha256:")
        or not payload["calibrated_fingerprint"].startswith("sha256:")
        or not _is_sha256(payload["original_fingerprint"].removeprefix("sha256:"))
        or not _is_sha256(payload["calibrated_fingerprint"].removeprefix("sha256:"))
        or not isinstance(evidence, dict)
        or set(evidence) != {"accepted_design", "qualification_failure"}
        or not all(_is_sha256(value) for value in evidence.values())
    ):
        raise ValueError("provisional calibration amendment is invalid")
    evidence.update(
        {
            "alignment_report": _require_sha256(
                alignment_report_sha256, "alignment_report_sha256"
            ),
            "qualification_report": _require_sha256(
                qualification_report_sha256, "qualification_report_sha256"
            ),
            "qualification_receipt": _require_sha256(
                qualification_receipt_sha256, "qualification_receipt_sha256"
            ),
        }
    )
    payload["status"] = "accepted"
    _write_json(path, payload)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--path", type=Path, required=True)
    create.add_argument("--original-fingerprint", required=True)
    create.add_argument("--calibrated-fingerprint", required=True)
    create.add_argument("--accepted-design", type=Path, required=True)
    create.add_argument("--qualification-failure", type=Path, required=True)

    accept = subparsers.add_parser("accept")
    accept.add_argument("--path", type=Path, required=True)
    accept.add_argument("--alignment-report", type=Path, required=True)
    accept.add_argument("--qualification-report", type=Path, required=True)
    accept.add_argument("--qualification-receipt", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "create":
            create_calibration_amendment(
                args.path,
                original_fingerprint=args.original_fingerprint,
                calibrated_fingerprint=args.calibrated_fingerprint,
                accepted_design_sha256=file_sha256(args.accepted_design),
                qualification_failure_sha256=file_sha256(args.qualification_failure),
            )
        else:
            accept_calibration_amendment(
                args.path,
                alignment_report_sha256=file_sha256(args.alignment_report),
                qualification_report_sha256=file_sha256(args.qualification_report),
                qualification_receipt_sha256=file_sha256(args.qualification_receipt),
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
