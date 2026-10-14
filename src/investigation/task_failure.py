"""Bounded host-only control channel for evaluator-contract and logger failures."""

from __future__ import annotations

import argparse
import json
import os
import stat
from types import MappingProxyType

TASK_CONTRACT_EXIT_CODE = 86
TASK_CONTRACT_FAILURE = "evaluator_output_contract_invalid"
INFRASTRUCTURE_EXIT_CODE = 87
AGENT_LOG_FAILURE = "agent_log_persistence_failed"
HOST_FAILURES = MappingProxyType({
    TASK_CONTRACT_EXIT_CODE: (TASK_CONTRACT_FAILURE, "task"),
    INFRASTRUCTURE_EXIT_CODE: (AGENT_LOG_FAILURE, "infrastructure"),
})
TASK_FAILURE_FD_ENV = "AGENTIC_TASK_FAILURE_FD"
MAX_RECEIPT_BYTES = 4096


class TaskFailureControlError(RuntimeError):
    """The private host channel is corrupt, not a candidate/scientific failure."""


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TaskFailureControlError("invalid host failure control receipt")
        result[key] = value
    return result


def _receipt(attempt_id: str, idea_id: str, category: str) -> dict[str, object]:
    _, scope = failure_details(category)
    return {
        "schema_version": 1, "attempt_id": attempt_id, "idea_id": idea_id,
        "failure_category": category, "failure_scope": scope,
    }


def failure_details(category: str) -> tuple[int, str]:
    for code, (known_category, scope) in HOST_FAILURES.items():
        if category == known_category:
            return code, scope
    raise ValueError("unknown host failure category")


def write_task_failure(descriptor: int, attempt_id: str, idea_id: str) -> None:
    _write_failure(descriptor, attempt_id, idea_id, TASK_CONTRACT_FAILURE)


def write_log_failure(descriptor: int, attempt_id: str, idea_id: str) -> None:
    _write_failure(descriptor, attempt_id, idea_id, AGENT_LOG_FAILURE)


def _write_failure(descriptor: int, attempt_id: str, idea_id: str, category: str) -> None:
    """Write only fixed metadata, never evaluator/candidate exception text."""
    raw = json.dumps(_receipt(attempt_id, idea_id, category), sort_keys=True).encode()
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("task failure identity exceeds control-channel bound")
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise ValueError("task failure control descriptor must be a regular file")
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("task failure control write did not progress")
        view = view[written:]


def read_task_failure(descriptor: int, attempt_id: str, idea_id: str) -> str | None:
    """Accept only an exact receipt from the runner's anonymous private descriptor."""
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RECEIPT_BYTES:
        raise TaskFailureControlError("invalid host failure control receipt")
    raw = os.pread(descriptor, MAX_RECEIPT_BYTES + 1, 0)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise TaskFailureControlError("invalid host failure control receipt")
    if not raw:
        return None
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_fields)
    except (ValueError, UnicodeError, RecursionError):
        raise TaskFailureControlError("invalid host failure control receipt") from None
    if isinstance(payload, dict) and type(payload.get("schema_version")) is int:
        for category, _ in HOST_FAILURES.values():
            if payload == _receipt(attempt_id, idea_id, category):
                return category
    raise TaskFailureControlError("invalid host failure control receipt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report host logger failure privately")
    parser.add_argument("--log-failure-fd", type=int, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--idea-id", required=True)
    args = parser.parse_args(argv)
    if args.log_failure_fd < 3:
        parser.error("logger failure requires a private descriptor")
    write_log_failure(args.log_failure_fd, args.attempt_id, args.idea_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
