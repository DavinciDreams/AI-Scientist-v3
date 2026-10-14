from __future__ import annotations

from pathlib import Path


def write_program_spec_toml(
    program_dir: Path,
    *,
    entrypoint: str = "program.py",
    language: str = "python",
) -> Path:
    """Write a minimal v1 program_spec.toml to program_dir and return its path."""
    program_dir.mkdir(parents=True, exist_ok=True)
    spec_path = program_dir / "program_spec.toml"
    spec_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'code_root = "src"',
                "",
                "[entrypoint]",
                f'path = "{entrypoint}"',
                f'language = "{language}"',
                'type = "script"',
                "",
                "[copy]",
                "exclude = []",
                "include = []",
                "",
                "[integration]",
                'kind = "generic"',
                'notes = ""',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return spec_path


def write_program_entrypoint(
    program_dir: Path,
    content: str,
    *,
    entrypoint: str = "program.py",
) -> Path:
    """Write <program_dir>/src/<entrypoint> and return its path."""
    src_dir = program_dir / "src"
    target = src_dir / entrypoint
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def make_program_dir(
    program_dir: Path,
    *,
    content: str = "print('ok')\n",
    entrypoint: str = "program.py",
    language: str = "python",
) -> Path:
    """Create a minimal program_<idx> directory with spec + src entrypoint."""
    write_program_spec_toml(program_dir, entrypoint=entrypoint, language=language)
    return write_program_entrypoint(program_dir, content, entrypoint=entrypoint)

