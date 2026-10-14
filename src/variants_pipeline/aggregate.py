"""Post-execution aggregation: merge per-variant analysis CSVs into combined files.

Agents may have added or removed columns depending on the variant, so this
module takes a conservative union-of-columns approach: every column that appears
in *any* variant's CSV is included, and missing values are filled with empty
strings.

Results are written to per-task subdirectories under ``aggregated/``
(e.g. ``aggregated/ed_conv/run_summary.csv``).

Two extra-task layouts are supported:

1. **Structured** (pipeline-generated): ``extra_tasks/{task}/variants/{id}/task_bundle/program_*/analysis/*.csv``
2. **Flat** (manually added): ``extra_tasks/{task}/{variant_id}/*.csv``
   CSVs live directly inside each variant subdirectory.  ``evaluation_id`` and
   ``variant_id`` columns are injected automatically from the directory name.
"""

from __future__ import annotations

import csv
import logging
import re
import shutil
import tempfile
from pathlib import Path

from .constants import ANALYSIS_CSV_SCHEMAS
from .qualification_summary import summary_qualification

logger = logging.getLogger(__name__)

# The CSV files we attempt to aggregate (same set that the rubric contract checks).
_CSV_FILENAMES = list(ANALYSIS_CSV_SCHEMAS.keys())

# Extra filenames recognized only in the nested-rep extra-task layout
# (``{task}/{algorithm}/rep_<N>/<csv>``). These are not part of the rubric
# contract — they are raw per-replication timeseries dumps.
_NESTED_REP_CSV_FILENAMES = ["run_timeseries.csv"]

_REP_DIR_RE = re.compile(r"^rep_(.+)$")


def aggregate_variant_csvs(
    variants_run_dir: Path,
    original_task_name: str = "original",
) -> dict[str, dict[str, Path | None]]:
    """Aggregate analysis CSVs across all variant workspaces and extra tasks.

    Scans ``variants_run_dir/variants/*/task_bundle/program_*/analysis/`` for
    each CSV filename defined in :data:`ANALYSIS_CSV_SCHEMAS`.  When
    ``extra_tasks/`` subdirectories exist, they are also scanned.

    Results are written to per-task subdirectories under ``aggregated/``
    (e.g. ``aggregated/ed_conv/run_summary.csv``).

    Parameters
    ----------
    variants_run_dir:
        The top-level output directory for this pipeline run (contains
        ``variants/`` subdirectory and optionally ``extra_tasks/``).
    original_task_name:
        Label for the original/primary task (used as the subdirectory name
        under ``aggregated/`` for rows from ``variants/``).

    Returns
    -------
    dict mapping each task name to a dict of CSV filename -> aggregated
    file path (or ``None`` if no data was found for that file type).
    """
    # Build list of (variants_dir, task_name) pairs to scan.
    sources: list[tuple[Path, str]] = []

    primary_variants = variants_run_dir / "variants"
    if primary_variants.is_dir():
        sources.append((primary_variants, original_task_name))

    # Flat extra tasks: extra_tasks/{task}/{variant_id}/*.csv (no variants/ dir).
    flat_sources: list[tuple[Path, str]] = []
    # Nested-rep extra tasks: extra_tasks/{task}/{algorithm}/rep_<N>/*.csv.
    nested_rep_sources: list[tuple[Path, str]] = []

    extra_tasks_root = variants_run_dir / "extra_tasks"
    if extra_tasks_root.is_dir():
        for task_dir in sorted(extra_tasks_root.iterdir()):
            if not task_dir.is_dir():
                continue
            task_variants = task_dir / "variants"
            if task_variants.is_dir():
                sources.append((task_variants, task_dir.name))
            elif _is_nested_rep_extra_task(task_dir):
                nested_rep_sources.append((task_dir, task_dir.name))
            elif _is_flat_extra_task(task_dir):
                flat_sources.append((task_dir, task_dir.name))

    if not sources and not flat_sources and not nested_rep_sources:
        logger.warning(
            "No variants directory found at %s; skipping aggregation",
            variants_run_dir,
        )
        return {}

    aggregated_root = variants_run_dir / "aggregated"

    results: dict[str, dict[str, Path | None]] = {}

    # Structured sources (primary task + pipeline-generated extra tasks).
    for variants_dir, task_name in sources:
        task_output_dir = aggregated_root / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        task_results: dict[str, Path | None] = {}
        for csv_filename in _CSV_FILENAMES:
            csv_paths = _find_analysis_csvs(variants_dir, csv_filename)

            if not csv_paths:
                logger.info(
                    "No %s files found for task %s; skipping",
                    csv_filename,
                    task_name,
                )
                task_results[csv_filename] = None
                continue

            out_path = task_output_dir / csv_filename
            row_count = _merge_csvs(
                [
                    (p, summary_qualification(p) if csv_filename == "run_summary.csv" else {})
                    for p in csv_paths
                ],
                out_path,
            )
            logger.info(
                "Aggregated %s/%s: %d files, %d total rows -> %s",
                task_name,
                csv_filename,
                len(csv_paths),
                row_count,
                out_path,
            )
            task_results[csv_filename] = out_path

        if variants_dir == primary_variants:
            _sync_primary_compatibility_files(
                aggregated_root=aggregated_root,
                task_results=task_results,
            )

        results[task_name] = task_results

    # Flat extra tasks (manually added results).
    for task_dir, task_name in flat_sources:
        task_output_dir = aggregated_root / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        flat_task_results: dict[str, Path | None] = {}
        for csv_filename in _CSV_FILENAMES:
            csv_entries = _find_flat_csvs(task_dir, csv_filename)

            if not csv_entries:
                logger.info(
                    "No %s files found for flat task %s; skipping",
                    csv_filename,
                    task_name,
                )
                flat_task_results[csv_filename] = None
                continue

            out_path = task_output_dir / csv_filename
            row_count = _merge_csvs(csv_entries, out_path)
            logger.info(
                "Aggregated (flat) %s/%s: %d files, %d total rows -> %s",
                task_name,
                csv_filename,
                len(csv_entries),
                row_count,
                out_path,
            )
            flat_task_results[csv_filename] = out_path

        results[task_name] = flat_task_results

    # Nested-rep extra tasks: {task}/{algorithm}/rep_<N>/<csv>.
    for task_dir, task_name in nested_rep_sources:
        task_output_dir = aggregated_root / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        nested_task_results: dict[str, Path | None] = {}
        for csv_filename in _NESTED_REP_CSV_FILENAMES:
            csv_entries = _find_nested_rep_csvs(task_dir, csv_filename)

            if not csv_entries:
                logger.info(
                    "No %s files found for nested-rep task %s; skipping",
                    csv_filename,
                    task_name,
                )
                nested_task_results[csv_filename] = None
                continue

            out_path = task_output_dir / csv_filename
            row_count = _merge_csvs(csv_entries, out_path)
            logger.info(
                "Aggregated (nested-rep) %s/%s: %d files, %d total rows -> %s",
                task_name,
                csv_filename,
                len(csv_entries),
                row_count,
                out_path,
            )
            nested_task_results[csv_filename] = out_path

        results[task_name] = nested_task_results

    return results


def _sync_primary_compatibility_files(
    *,
    aggregated_root: Path,
    task_results: dict[str, Path | None],
) -> None:
    """Mirror primary-task CSVs at aggregated/*.csv for existing consumers."""
    aggregated_root.mkdir(parents=True, exist_ok=True)
    for csv_filename, source_path in task_results.items():
        compatibility_path = aggregated_root / csv_filename
        if source_path is None:
            compatibility_path.unlink(missing_ok=True)
            continue

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=aggregated_root,
                prefix=f".{csv_filename}.",
                delete=False,
            ) as tmp:
                temp_path = Path(tmp.name)
            shutil.copy2(source_path, temp_path)
            temp_path.replace(compatibility_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def _is_flat_extra_task(task_dir: Path) -> bool:
    """Return True if *task_dir* contains flat variant subdirectories with CSVs.

    A flat extra task has the layout ``{task_dir}/{variant_id}/*.csv`` — no
    ``variants/`` or ``task_bundle/`` nesting.
    """
    for child in task_dir.iterdir():
        if child.name.startswith("_"):
            continue
        if child.is_dir() and any(child.glob("*.csv")):
            return True
    return False


def _is_nested_rep_extra_task(task_dir: Path) -> bool:
    """Return True if *task_dir* uses the nested ``{algorithm}/rep_<N>/<csv>`` layout.

    Matches when at least one algorithm subdirectory contains at least one
    ``rep_<something>`` directory holding a recognized nested-rep CSV.
    """
    for algo_dir in task_dir.iterdir():
        if not algo_dir.is_dir() or algo_dir.name.startswith("_"):
            continue
        for rep_dir in algo_dir.iterdir():
            if not rep_dir.is_dir():
                continue
            if not _REP_DIR_RE.match(rep_dir.name):
                continue
            for csv_name in _NESTED_REP_CSV_FILENAMES:
                if (rep_dir / csv_name).is_file():
                    return True
    return False


def _find_nested_rep_csvs(
    task_dir: Path,
    csv_filename: str,
) -> list[tuple[Path, dict[str, str]]]:
    """Find *csv_filename* under ``{task_dir}/{algorithm}/rep_<N>/``.

    Each rep is treated as a separate variant row. The following columns are
    injected from directory names: ``algorithm``, ``replication``,
    ``variant_id`` (``"{algorithm}__rep_{N}"``) and ``evaluation_id``
    (same value as ``variant_id``).
    """
    entries: list[tuple[Path, dict[str, str]]] = []
    for algo_dir in sorted(task_dir.iterdir()):
        if not algo_dir.is_dir() or algo_dir.name.startswith("_"):
            continue
        for rep_dir in sorted(algo_dir.iterdir()):
            if not rep_dir.is_dir():
                continue
            m = _REP_DIR_RE.match(rep_dir.name)
            if not m:
                continue
            csv_path = rep_dir / csv_filename
            if not csv_path.is_file():
                continue
            rep_label = m.group(1)
            variant_id = f"{algo_dir.name}__rep_{rep_label}"
            entries.append((
                csv_path,
                {
                    "evaluation_id": variant_id,
                    "variant_id": variant_id,
                    "algorithm": algo_dir.name,
                    "replication": rep_label,
                },
            ))
    return entries


def _find_flat_csvs(
    task_dir: Path,
    csv_filename: str,
) -> list[tuple[Path, dict[str, str]]]:
    """Find *csv_filename* in flat variant subdirectories.

    Looks for ``{task_dir}/{variant_id}/{csv_filename}`` and returns entries
    suitable for :func:`_merge_csvs` — each with ``evaluation_id`` and
    ``variant_id`` injected from the directory name.
    """
    entries: list[tuple[Path, dict[str, str]]] = []
    for variant_dir in sorted(task_dir.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name.startswith("_"):
            continue
        csv_path = variant_dir / csv_filename
        if csv_path.is_file():
            variant_id = variant_dir.name
            entries.append((
                csv_path,
                {"evaluation_id": variant_id, "variant_id": variant_id},
            ))
    return entries


def _find_analysis_csvs(variants_dir: Path, csv_filename: str) -> list[Path]:
    """Find the latest *csv_filename* per variant under program analysis dirs.

    Looks for ``variants/<variant_id>/task_bundle/program_*/analysis/<csv_filename>``
    where program index > 0 (program_0 is the unmodified seed).

    For each variant, only the **highest-numbered** program directory that
    contains the requested CSV is included.  Earlier programs are intermediate
    attempts by the agent and their results should not be aggregated.
    """
    paths: list[Path] = []
    for variant_dir in sorted(variants_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        task_bundle = variant_dir / "task_bundle"
        if not task_bundle.is_dir():
            continue
        # Collect all program_N dirs that have the CSV, pick the highest N.
        # If only program_0 exists (e.g. control_best), include it; otherwise
        # prefer the highest agent-created program (program_1+).
        best_idx = -1
        best_csv: Path | None = None
        for prog_dir in task_bundle.iterdir():
            if not prog_dir.is_dir():
                continue
            m = re.match(r"^program_(\d+)$", prog_dir.name)
            if not m:
                continue
            idx = int(m.group(1))
            csv_path = prog_dir / "analysis" / csv_filename
            if csv_path.is_file() and idx > best_idx:
                best_idx = idx
                best_csv = csv_path
        if best_csv is not None:
            paths.append(best_csv)
    return paths


def _merge_csvs(
    csv_entries: list[tuple[Path, dict[str, str]]],
    output_path: Path,
) -> int:
    """Merge multiple CSV files into one, using the union of all columns.

    Each entry is ``(csv_path, extra_columns)`` where *extra_columns* are
    key-value pairs injected into every row from that file.

    Columns are ordered as: columns from the first file (preserving order),
    then any additional columns from subsequent files (sorted alphabetically).

    Missing values are written as empty strings.

    Returns the total number of data rows written.
    """
    if not csv_entries:
        output_path.write_text("", encoding="utf-8")
        return 0

    # First pass: collect all column names preserving order from files.
    ordered_columns: list[str] = []
    seen_columns: set[str] = set()

    # Include any extra column keys first (preserves ordering when present).
    for _, extra_cols in csv_entries:
        for col in extra_cols:
            if col not in seen_columns:
                ordered_columns.append(col)
                seen_columns.add(col)

    for csv_path, _ in csv_entries:
        headers = _read_csv_headers(csv_path)
        for col in headers:
            if col not in seen_columns:
                ordered_columns.append(col)
                seen_columns.add(col)

    if not ordered_columns:
        logger.warning("No columns found in any CSV; writing empty file to %s", output_path)
        output_path.write_text("", encoding="utf-8")
        return 0

    # Second pass: read all rows and write combined file.
    total_rows = 0
    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=ordered_columns, extrasaction="ignore")
        writer.writeheader()

        for csv_path, extra_cols in csv_entries:
            try:
                with open(csv_path, newline="", encoding="utf-8") as in_f:
                    reader = csv.DictReader(in_f)
                    for row in reader:
                        filled_row = {col: row.get(col, "") for col in ordered_columns}
                        filled_row.update(extra_cols)
                        writer.writerow(filled_row)
                        total_rows += 1
            except Exception:
                logger.exception("Error reading %s; skipping this file", csv_path)

    return total_rows


def _read_csv_headers(csv_path: Path) -> list[str]:
    """Read just the header row from a CSV file."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            return [h.strip() for h in headers]
    except Exception:
        logger.exception("Error reading headers from %s", csv_path)
        return []
