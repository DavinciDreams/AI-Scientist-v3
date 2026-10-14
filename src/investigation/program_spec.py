"""Program variant specification (program_spec.toml) helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the Python 3.10 task env
    import tomli as tomllib

PROGRAM_SPEC_FILENAME = "program_spec.toml"
PROGRAM_SPEC_SCHEMA_VERSION = 1


class ProgramSpecError(RuntimeError):
    """Raised when a program_spec.toml is missing or invalid."""


@dataclass(frozen=True)
class ProgramEntrypoint:
    path: str
    language: str
    type: str  # "script" | "module" (module reserved for future)


@dataclass(frozen=True)
class ProgramCopySpec:
    exclude: tuple[str, ...]
    include: tuple[str, ...]


@dataclass(frozen=True)
class ProgramIntegrationSpec:
    kind: str
    notes: str
    extra: dict[str, Any]


@dataclass(frozen=True)
class ProgramSpec:
    schema_version: int
    code_root: str
    entrypoint: ProgramEntrypoint
    copy: ProgramCopySpec
    integration: ProgramIntegrationSpec


def default_program_spec() -> ProgramSpec:
    """Return the canonical v1 default spec."""
    return ProgramSpec(
        schema_version=PROGRAM_SPEC_SCHEMA_VERSION,
        code_root="src",
        entrypoint=ProgramEntrypoint(
            path="program.py",
            language="python",
            type="script",
        ),
        copy=ProgramCopySpec(
            exclude=(
                "**/__pycache__/**",
                "**/*.pyc",
                ".venv/**",
                "node_modules/**",
                ".git/**",
                ".mypy_cache/**",
                ".pytest_cache/**",
                ".ruff_cache/**",
                ".tox/**",
                ".nox/**",
                "dist/**",
                "build/**",
            ),
            include=(),
        ),
        integration=ProgramIntegrationSpec(kind="generic", notes="", extra={}),
    )


def render_default_program_spec_toml() -> str:
    """Render the canonical v1 default specification as TOML."""
    spec = default_program_spec()
    quote = lambda value: json.dumps(value, ensure_ascii=False)
    lines = [
        f"schema_version = {spec.schema_version}",
        f"code_root = {quote(spec.code_root)}",
        "",
        "[entrypoint]",
        f"path = {quote(spec.entrypoint.path)}",
        f"language = {quote(spec.entrypoint.language)}",
        f"type = {quote(spec.entrypoint.type)}",
        "",
        "[copy]",
        "exclude = [",
        *(f"  {quote(pattern)}," for pattern in spec.copy.exclude),
        "]",
        "include = [",
        *(f"  {quote(pattern)}," for pattern in spec.copy.include),
        "]",
        "",
        "[integration]",
        f"kind = {quote(spec.integration.kind)}",
        f"notes = {quote(spec.integration.notes)}",
        "",
    ]
    return "\n".join(lines)


def program_spec_path(program_dir: Path) -> Path:
    return program_dir / PROGRAM_SPEC_FILENAME


def _expect_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ProgramSpecError(f"{label} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ProgramSpecError(f"{label} must be non-empty")
    return stripped


def _expect_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramSpecError(f"{label} must be an integer")
    return value


def _expect_str_list(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProgramSpecError(f"{label} must be an array of strings")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ProgramSpecError(f"{label}[{idx}] must be a string")
        s = item.strip()
        if not s:
            continue
        out.append(s)
    return tuple(out)


def _ensure_safe_relpath(raw: str, *, label: str) -> str:
    path = Path(raw)
    if path.is_absolute():
        raise ProgramSpecError(f"{label} must be a relative path (got absolute path)")
    if any(part in ("..",) for part in path.parts):
        raise ProgramSpecError(f"{label} must not contain '..'")
    if raw.strip() in (".", ""):
        raise ProgramSpecError(f"{label} must not be '.' or empty")
    return path.as_posix()


def load_program_spec(program_dir: Path, *, required: bool = True) -> ProgramSpec | None:
    """Load and validate program_spec.toml from a program directory."""
    spec_path = program_spec_path(program_dir)
    if not spec_path.exists():
        if required:
            raise ProgramSpecError(f"Missing {PROGRAM_SPEC_FILENAME} at {spec_path}")
        return None
    if not spec_path.is_file():
        raise ProgramSpecError(f"{spec_path} must be a file")

    try:
        data = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ProgramSpecError(f"Failed to parse {spec_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProgramSpecError(f"{spec_path} must contain a top-level TOML table")

    schema_version = _expect_int(data.get("schema_version"), label="schema_version")
    if schema_version != PROGRAM_SPEC_SCHEMA_VERSION:
        raise ProgramSpecError(
            f"Unsupported schema_version={schema_version} (expected {PROGRAM_SPEC_SCHEMA_VERSION})"
        )

    code_root_raw = _expect_str(data.get("code_root"), label="code_root")
    code_root = _ensure_safe_relpath(code_root_raw, label="code_root")
    if code_root != "src":
        raise ProgramSpecError("code_root must be 'src' for schema_version=1")

    entrypoint = data.get("entrypoint")
    if not isinstance(entrypoint, dict):
        raise ProgramSpecError("entrypoint must be a TOML table")
    entry_path = _ensure_safe_relpath(
        _expect_str(entrypoint.get("path"), label="entrypoint.path"),
        label="entrypoint.path",
    )
    entry_language = _expect_str(entrypoint.get("language"), label="entrypoint.language")
    entry_type = _expect_str(entrypoint.get("type"), label="entrypoint.type").lower()
    if entry_type not in ("script", "module"):
        raise ProgramSpecError("entrypoint.type must be 'script' or 'module'")

    copy_spec = data.get("copy")
    if not isinstance(copy_spec, dict):
        raise ProgramSpecError("copy must be a TOML table")
    exclude = _expect_str_list(copy_spec.get("exclude"), label="copy.exclude")
    include = _expect_str_list(copy_spec.get("include"), label="copy.include")

    integration = data.get("integration")
    if not isinstance(integration, dict):
        raise ProgramSpecError("integration must be a TOML table")
    kind = _expect_str(integration.get("kind"), label="integration.kind")
    notes = str(integration.get("notes") or "")
    extra = {k: v for k, v in integration.items() if k not in ("kind", "notes")}

    return ProgramSpec(
        schema_version=schema_version,
        code_root=code_root,
        entrypoint=ProgramEntrypoint(path=entry_path, language=entry_language, type=entry_type),
        copy=ProgramCopySpec(exclude=exclude, include=include),
        integration=ProgramIntegrationSpec(kind=kind, notes=notes, extra=extra),
    )


def resolve_entrypoint(program_dir: Path, spec: ProgramSpec) -> Path:
    """Resolve the entrypoint absolute path and ensure code_root stays under program_dir and the entrypoint stays under code_root."""
    resolved_program_dir = program_dir.resolve()
    code_root_dir = (program_dir / spec.code_root).resolve()
    try:
        code_root_dir.relative_to(resolved_program_dir)
    except ValueError as exc:
        raise ProgramSpecError(
            f"code_root escapes program_dir: {code_root_dir} is not under {resolved_program_dir}"
        ) from exc
    entrypoint_path = (code_root_dir / spec.entrypoint.path).resolve()
    try:
        entrypoint_path.relative_to(code_root_dir)
    except ValueError as exc:
        raise ProgramSpecError(
            f"Entrypoint escapes code_root: {entrypoint_path} is not under {code_root_dir}"
        ) from exc
    return entrypoint_path


def entrypoint_relpath_from_spec(program_dir: Path, *, required: bool = True) -> str | None:
    """Best-effort compute entrypoint_path relative to program_dir from program_spec.toml."""
    try:
        spec = load_program_spec(program_dir, required=required)
        if spec is None:
            return None
        entrypoint_abs = resolve_entrypoint(program_dir, spec)
    except (ProgramSpecError, ValueError):
        return None
    if not entrypoint_abs.is_file():
        return None
    return entrypoint_abs.relative_to(program_dir.resolve()).as_posix()
