"""Credential-path and staged-blob policy for run checkpoints."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

SENSITIVE_DIRECTORY_NAMES = frozenset(
    {
        ".agent_home",
        ".aws",
        ".azure",
        ".docker",
        ".gnupg",
        ".kaggle",
        ".kube",
        ".ssh",
    }
)
SENSITIVE_PATH_SUFFIXES = (
    (".cache", "huggingface"),
    (".config", "gcloud"),
    (".config", "gh"),
    (".config", "huggingface"),
)
SENSITIVE_FILE_NAMES = frozenset(
    {
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "application_default_credentials.json",
        "auth.json",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "kaggle.json",
        "service-account.json",
        "token",
        "token.json",
    }
)
SENSITIVE_FILE_SUFFIXES = frozenset({".key", ".p12", ".pfx"})
SAFE_ENV_TEMPLATE_SUFFIXES = (".example", ".sample", ".template")

_SCAN_OVERLAP_BYTES = 512
_UNIVERSAL_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(
        rb"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}|sk_live_[A-Za-z0-9]{16,}|"
        rb"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        rb"glpat-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|"
        rb"xox[baprs]-[A-Za-z0-9-]{20,}|AIza[0-9A-Za-z_-]{30,})\b"
    ),
    re.compile(rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
)
_ASSIGNMENT_RE = re.compile(
    rb"(?i)(?:api[_-]?key|api[_-]?token|auth[_-]?token|access[_-]?token|"
    rb"refresh[_-]?token|client[_-]?secret|secret[_-]?key|aws[_-]?secret[_-]?access[_-]?key|"
    rb"password)"
    rb"[\"']?\s*[:=]\s*(?:[\"'](?P<quoted>[^\"'\r\n]{12,})[\"']|"
    rb"(?P<bare>[^\s,#;]{12,}))"
)
_PLACEHOLDER_VALUES = frozenset(
    {
        b"a_secure_password",
        b"changeme",
        b"dummy",
        b"example",
        b"placeholder_value",
        b"redacted",
        b"your-api-key-here",
    }
)
_SOURCE_REFERENCE_RE = re.compile(
    rb"(?i)^(?:args|cfg|config|response|self|settings)\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$"
)
_SOURCE_CALL_RE = re.compile(rb"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\(.*\)$")
_TYPE_EXPRESSION_RE = re.compile(rb"^[A-Za-z_]\w*\[[^\]]+\]$")
_TEMPLATE_EXPRESSION_RE = re.compile(
    rb"^(?:\$[A-Z_][A-Z0-9_]*|\$\{[A-Z_][A-Z0-9_]*\}|\{\{[^{}]+\}\})$"
)


def is_sensitive_path(path: str) -> bool:
    """Return whether a Git path names a conventional credential location."""
    relative = PurePosixPath(path)
    lowered_parts = tuple(part.lower() for part in relative.parts)
    if any(part in SENSITIVE_DIRECTORY_NAMES for part in lowered_parts):
        return True
    if any(
        lowered_parts[index : index + len(suffix)] == suffix
        for suffix in SENSITIVE_PATH_SUFFIXES
        for index in range(len(lowered_parts) - len(suffix) + 1)
    ):
        return True
    name = relative.name.lower()
    if name in SENSITIVE_FILE_NAMES or relative.suffix.lower() in SENSITIVE_FILE_SUFFIXES:
        return True
    return name == ".env" or (
        name.startswith(".env.") and not name.endswith(SAFE_ENV_TEMPLATE_SUFFIXES)
    )


def scan_sensitive_paths(
    root: Path,
    *,
    skipped_directory_names: frozenset[str],
    checkpoint_git_dir_name: str | None = None,
    deadline: float | None = None,
) -> tuple[str, ...]:
    """Find conventional credential paths without reading file contents."""
    sensitive: list[str] = []
    stack = [root]
    while stack:
        if deadline is not None and time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired("checkpoint sensitive-path scan", 0)
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        current_relative = current.relative_to(root)
        if current_relative.parts and any(
            entry.name == ".git" and entry.is_dir(follow_symlinks=False) for entry in entries
        ):
            sensitive.append(current_relative.as_posix())
            continue
        for entry in entries:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("checkpoint sensitive-path scan", 0)
            relative = Path(entry.path).relative_to(root)
            relative_name = relative.as_posix()
            if is_sensitive_path(relative_name) or contains_secret(
                (relative_name.encode("utf-8", errors="surrogateescape"),)
            ):
                sensitive.append(relative_name)
            elif entry.is_dir(follow_symlinks=False):
                if _is_checkpoint_sidecar(entry.name, checkpoint_git_dir_name):
                    sensitive.append(relative_name)
                elif entry.name not in skipped_directory_names:
                    stack.append(Path(entry.path))
    return tuple(sorted(sensitive))


def _is_checkpoint_sidecar(name: str, git_dir_name: str | None) -> bool:
    return bool(
        git_dir_name
        and name.startswith(".")
        and name != git_dir_name
        and name.endswith(git_dir_name)
    )


def contains_secret(chunks: Iterable[bytes]) -> bool:
    """Inspect exact Git-blob bytes without treating NUL or suffixes as safe."""
    detector = SecretDetector()
    for chunk in chunks:
        if detector.feed(chunk):
            return True
    return False


def report_sensitive_path(path: str) -> str:
    """Redact a pathname when the pathname itself contains credential material."""
    encoded = path.encode("utf-8", errors="surrogateescape")
    if not contains_secret((encoded,)):
        return "".join(_display_character(character) for character in path)
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"<redacted-path:{digest}>"


def _display_character(character: str) -> str:
    codepoint = ord(character)
    if character.isprintable():
        return character
    if codepoint <= 0xFF:
        return f"\\x{codepoint:02x}"
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04x}"
    return f"\\U{codepoint:08x}"


def redact_sensitive_text(value: str) -> str:
    """Replace an observable diagnostic when it contains recognizable credentials."""
    encoded = value.encode("utf-8", errors="surrogateescape")
    if not contains_secret((encoded,)):
        return "".join(_display_character(character) for character in value)
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"<redacted-sensitive-text:{digest}>"


class SecretDetector:
    """Incrementally recognize secrets while callers continue draining a blob."""

    def __init__(self) -> None:
        self._overlap = b""

    def feed(self, chunk: bytes) -> bool:
        candidate = self._overlap + chunk
        found = any(pattern.search(candidate) for pattern in _UNIVERSAL_SECRET_PATTERNS)
        found = found or _contains_literal_assignment(candidate)
        self._overlap = candidate[-_SCAN_OVERLAP_BYTES:]
        return found


def _contains_literal_assignment(content: bytes) -> bool:
    for match in _ASSIGNMENT_RE.finditer(content):
        value = match.group("quoted") or match.group("bare")
        if value.lower() in _PLACEHOLDER_VALUES:
            continue
        if _TEMPLATE_EXPRESSION_RE.fullmatch(value):
            continue
        if any(
            pattern.fullmatch(value)
            for pattern in (_SOURCE_REFERENCE_RE, _SOURCE_CALL_RE, _TYPE_EXPRESSION_RE)
        ):
            continue
        return True
    return False
