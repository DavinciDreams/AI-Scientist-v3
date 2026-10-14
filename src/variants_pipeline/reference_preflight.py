"""Check comparison inputs before adapting an external reference."""

from pathlib import Path

from src.investigation.safe_staging import read_text


def validate_reference_workspace(
    task_bundle: Path, variant_id: str, base_program: str,
) -> None:
    """Reject external references whose required comparison inputs are absent."""
    if base_program != "reference" or variant_id == "baseline_reference":
        return
    try:
        control_spec = read_text(task_bundle, "control_program/program_spec.toml")
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError(f"Reference {variant_id!r} requires control_program") from exc
    if not control_spec.strip():
        raise ValueError(f"Reference {variant_id!r} requires control_program")
    for filename in ("README.md", "README.rst", "README.txt", "README", "reference_performance.md"):
        try:
            if read_text(task_bundle, f"program_0/{filename}").strip():
                return
        except (OSError, ValueError, UnicodeError):
            continue
    raise ValueError(
        f"Reference {variant_id!r} requires documentation in program_0/ "
        "(README or reference_performance.md) for the performance comparison"
    )
