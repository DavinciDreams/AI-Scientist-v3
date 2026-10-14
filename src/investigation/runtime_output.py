"""Private, sanitized console-output evidence for runtime preparation failures."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Mapping

from .bounded_process import (
    DEFAULT_MAX_OUTPUT_BYTES, MAX_OUTPUT_EVIDENCE_BYTES,
    ProcessOutputLimitError, ProcessTimeoutError,
)
from .checkpoint_security import contains_secret, redact_sensitive_text

_TRANSPORT_CREDENTIAL = re.compile(
    r"(?i)(?:\bauthorization\s*:\s*(?:bearer|basic)\s+\S+|"
    r"\b[a-z][a-z0-9+.-]*://[^\s/@]+@)"
)
_SENSITIVE_ENV_NAME = re.compile(r"(?i)AUTH|CREDENTIAL|KEY|PASSWORD|PROXY|SECRET|TOKEN")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(?:api[_-]?key|api[_-]?token|auth(?:orization)?|credentials?|"
    r"password|secret|token|access[_-]?token|client[_-]?secret)"
    r"[\"']?\s*[:=]\s*\S+"
)


@dataclass(frozen=True, slots=True)
class TaskRuntimeOutputEvidence:
    """Repair-only text and harness metadata; never send text to alignment judges."""

    stage: str
    task_fingerprint: str
    max_output_bytes: int
    observed_output_bytes: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    failure_category: str = "output_limit_exceeded"


def contains_sensitive_output(text: str, environment: Mapping[str, str]) -> bool:
    """Detect diagnostic credentials, including short literals and credential URLs."""
    return bool(
        _TRANSPORT_CREDENTIAL.search(text)
        or _CREDENTIAL_ASSIGNMENT.search(text)
        or any(
            secret and _SENSITIVE_ENV_NAME.search(name) and secret in text
            for name, secret in environment.items()
        )
        or contains_secret((text.encode("utf-8", errors="replace"),))
    )


def _sanitize_output(value: bytes, environment: Mapping[str, str]) -> str:
    text = value.decode("utf-8", errors="replace")
    if contains_sensitive_output(text, environment):
        return "<redacted-sensitive-text>"
    safe = redact_sensitive_text(text)
    return safe.encode("utf-8")[:MAX_OUTPUT_EVIDENCE_BYTES].decode("utf-8", errors="ignore")


def _complete_lines(prefix: bytes) -> bytes:
    """Discard a truncated final line whose credentials may be only partially captured."""
    return prefix[: prefix.rfind(b"\n") + 1]


def failed_evaluation_evidence(
    stderr: str, *, stage: str, environment: Mapping[str, str],
) -> TaskRuntimeOutputEvidence:
    """Only stderr, sanitized before truncation; never retain evaluator data stdout."""
    payload = stderr.encode("utf-8", errors="replace")
    return TaskRuntimeOutputEvidence(
        stage=stage, task_fingerprint="", max_output_bytes=MAX_OUTPUT_EVIDENCE_BYTES,
        observed_output_bytes=len(payload), stdout="", stderr=_sanitize_output(payload, environment),
        stderr_truncated=len(payload) > MAX_OUTPUT_EVIDENCE_BYTES,
        failure_category="evaluation_failed",
    )


def output_limit_evidence(
    error: ProcessOutputLimitError,
    *,
    stage: str,
    task_fingerprint: str,
    environment: Mapping[str, str],
) -> TaskRuntimeOutputEvidence:
    return TaskRuntimeOutputEvidence(
        stage=stage,
        task_fingerprint=task_fingerprint,
        max_output_bytes=error.max_output_bytes,
        observed_output_bytes=error.observed_output_bytes,
        stdout=_sanitize_output(
            _complete_lines(error.stdout) if error.stdout_truncated else error.stdout,
            environment,
        ),
        stderr=_sanitize_output(
            _complete_lines(error.stderr) if error.stderr_truncated else error.stderr,
            environment,
        ),
        stdout_truncated=error.stdout_truncated,
        stderr_truncated=error.stderr_truncated,
    )


def timeout_evidence(
    error: ProcessTimeoutError | subprocess.TimeoutExpired,
    *,
    stage: str,
    environment: Mapping[str, str],
) -> TaskRuntimeOutputEvidence:
    """Retain stderr only; interrupted capture cannot establish a complete final line."""
    if isinstance(error, ProcessTimeoutError):
        stderr = error.stderr
        truncated = error.stderr_truncated
        observed = error.observed_output_bytes
        output_limit = error.max_output_bytes
    else:
        payload = error.stderr or b""
        stderr = payload.encode("utf-8", errors="replace") if isinstance(payload, str) else payload
        observed = len(stderr)
        stderr = stderr[:MAX_OUTPUT_EVIDENCE_BYTES]
        truncated = True
        output_limit = DEFAULT_MAX_OUTPUT_BYTES
    return TaskRuntimeOutputEvidence(
        stage=stage, task_fingerprint="", max_output_bytes=output_limit,
        observed_output_bytes=observed, stdout="",
        stderr=_sanitize_output(_complete_lines(stderr) if truncated else stderr, environment),
        stderr_truncated=truncated, failure_category="timeout",
    )
