#!/usr/bin/env python3
"""Stage one Codex profile and its optional model catalog into a private home."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat

MAX_PROFILE_BYTES = 1024 * 1024
MAX_CATALOG_BYTES = 16 * 1024 * 1024
PROFILE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,255}$")
CATALOG_ASSIGNMENT = re.compile(
    r"^(\s*model_catalog_json\s*=\s*)(\"(?:[^\"\\]|\\.)*\"|'[^']*')(\s*(?:#.*)?)$",
    re.MULTILINE,
)
MODEL_PROVIDER_ASSIGNMENT = re.compile(
    r"^\s*model_provider\s*=\s*(\"(?:[^\"\\]|\\.)*\"|'[^']*')\s*(?:#.*)?$",
    re.MULTILINE,
)
PROVIDER_HEADER = re.compile(
    r"^\s*\[\s*model_providers\.(\"(?:[^\"\\]|\\.)*\"|'[^']*'|[A-Za-z0-9_-]+)\s*\]\s*(?:#.*)?$"
)
PROVIDER_VALUE = re.compile(
    r"^\s*([A-Za-z0-9_-]+)\s*=\s*(\"(?:[^\"\\]|\\.)*\"|'[^']*'|true|false|[-+]?[0-9][0-9_]*)\s*(?:#.*)?$"
)
SAFE_PROVIDER_KEYS = (
    "name",
    "base_url",
    "env_key",
    "wire_api",
    "request_max_retries",
    "stream_max_retries",
    "stream_idle_timeout_ms",
    "requires_openai_auth",
)


def read_regular_file(path: Path, limit: int, label: str) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ValueError(f"{label} must be a bounded regular file: {path}")
        with os.fdopen(os.dup(descriptor), encoding="utf-8") as source:
            return source.read()
    finally:
        os.close(descriptor)


def write_private_file(path: Path, content: str) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
        destination.write(content)


def parse_catalog_path(value: str) -> str:
    if value.startswith("'"):
        return value[1:-1]
    parsed = json.loads(value)
    if not isinstance(parsed, str):
        raise ValueError("Codex profile model_catalog_json must be a quoted path")
    return parsed


def parse_toml_name(value: str) -> str:
    if value.startswith(("'", '"')):
        return parse_catalog_path(value)
    return value


def catalog_source(source_home: Path, configured_path: str) -> Path:
    source_home = source_home.resolve(strict=True)
    configured = Path(configured_path)
    candidate = configured if configured.is_absolute() else source_home / configured
    candidate = Path(os.path.abspath(candidate))
    try:
        relative_parts = candidate.relative_to(source_home).parts
    except ValueError as error:
        raise ValueError("Codex model catalog must remain inside the Codex home") from error

    current = source_home
    for part in relative_parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Codex model catalog path must not contain symlinks")
    return candidate


def sanitized_toml_value(value: str) -> str:
    if value.startswith(("'", '"')):
        return json.dumps(parse_catalog_path(value))
    if value in {"true", "false"}:
        return value
    return str(int(value.replace("_", "")))


def selected_provider_config(source_home: Path, profile_content: str) -> str | None:
    provider_assignment = MODEL_PROVIDER_ASSIGNMENT.search(profile_content)
    if not provider_assignment:
        return None
    provider_name = parse_toml_name(provider_assignment.group(1))

    try:
        base_content = read_regular_file(
            source_home / "config.toml",
            MAX_PROFILE_BYTES,
            "Codex base configuration",
        )
    except FileNotFoundError:
        return None
    lines = [f"[model_providers.{json.dumps(provider_name)}]"]
    in_selected_provider = False
    for line in base_content.splitlines():
        if line.lstrip().startswith("["):
            header = PROVIDER_HEADER.fullmatch(line)
            in_selected_provider = bool(
                header and parse_toml_name(header.group(1)) == provider_name
            )
            continue
        if not in_selected_provider:
            continue
        assignment = PROVIDER_VALUE.fullmatch(line)
        if assignment and assignment.group(1) in SAFE_PROVIDER_KEYS:
            lines.append(f"{assignment.group(1)} = {sanitized_toml_value(assignment.group(2))}")
    return "\n".join(lines) + "\n" if len(lines) > 1 else None


def stage_profile(source_home: Path, destination_home: Path, profile: str) -> None:
    if not PROFILE_NAME.fullmatch(profile):
        raise ValueError("invalid Codex profile name")

    profile_name = f"{profile}.config.toml"
    profile_content = read_regular_file(
        source_home / profile_name,
        MAX_PROFILE_BYTES,
        "Codex profile",
    )
    provider_config = selected_provider_config(source_home, profile_content)
    assignment = CATALOG_ASSIGNMENT.search(profile_content)
    if assignment:
        catalog_path = catalog_source(
            source_home,
            parse_catalog_path(assignment.group(2)),
        )
        catalog_name = catalog_path.name
        if not catalog_name or catalog_name == profile_name:
            raise ValueError("Codex model catalog has an unsafe destination name")
        catalog_content = read_regular_file(
            catalog_path,
            MAX_CATALOG_BYTES,
            "Codex model catalog",
        )
        remote_path = f"/home/agent-home/.codex/{catalog_name}"
        replacement = f"{assignment.group(1)}{json.dumps(remote_path)}{assignment.group(3)}"
        profile_content = (
            profile_content[: assignment.start()]
            + replacement
            + profile_content[assignment.end() :]
        )
        write_private_file(destination_home / catalog_name, catalog_content)

    if provider_config:
        write_private_file(destination_home / "config.toml", provider_config)
    write_private_file(destination_home / profile_name, profile_content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_home", type=Path)
    parser.add_argument("destination_home", type=Path)
    parser.add_argument("profile")
    args = parser.parse_args()
    stage_profile(args.source_home, args.destination_home, args.profile)


if __name__ == "__main__":
    main()
