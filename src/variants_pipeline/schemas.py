"""CSV schema validation for variant analysis files (stdlib csv only)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from .constants import ANALYSIS_CSV_SCHEMAS

logger = logging.getLogger(__name__)


def validate_csv_schema(
    csv_path: Path,
    expected_columns: list[str],
) -> tuple[bool, str | None]:
    """Validate that a CSV file exists and has the required columns.

    Returns (passed, error_message).  *error_message* is ``None`` on success.
    """
    if not csv_path.exists():
        return False, f"File does not exist: {csv_path}"

    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return False, f"File is empty (no header row): {csv_path}"
    except OSError as exc:
        return False, f"Cannot read {csv_path}: {exc}"

    header_set = set(header)
    expected_set = set(expected_columns)
    missing = expected_set - header_set
    if missing:
        return False, (
            f"Missing columns in {csv_path.name}: {sorted(missing)}. "
            f"Found: {header}"
        )
    return True, None


def validate_all_analysis_files(
    analysis_dir: Path,
) -> tuple[bool, list[str]]:
    """Validate all required analysis CSV files in a variant's analysis/ dir.

    Returns (all_passed, list_of_error_messages).
    An empty error list means all validations passed.
    """
    errors: list[str] = []

    if not analysis_dir.is_dir():
        return False, [f"analysis directory does not exist: {analysis_dir}"]

    for filename, expected_columns in ANALYSIS_CSV_SCHEMAS.items():
        csv_path = analysis_dir / filename
        passed, error = validate_csv_schema(csv_path, expected_columns)
        if not passed:
            errors.append(error)  # type: ignore[arg-type]

    return len(errors) == 0, errors
