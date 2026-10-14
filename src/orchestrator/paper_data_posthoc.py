"""Discovery, loading, and integration of posthoc analyses for paper generation."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

POSTHOC_RELEVANCE_VALUES = frozenset({"high", "medium", "low"})
POSTHOC_USAGE_VALUES = frozenset({"main_results", "supporting", "appendix", "skip"})
MAX_POSTHOC_FILE_BYTES = 5 * 1024 * 1024
MAX_POSTHOC_ANALYSIS_BYTES = 20 * 1024 * 1024
MAX_POSTHOC_CONTEXT_BYTES = 50 * 1024 * 1024
MAX_POSTHOC_ANALYSES = 16
MAX_POSTHOC_DATA_FILES = 32
MAX_POSTHOC_CSV_ROWS = 10_000
MAX_POSTHOC_CSV_COLUMNS = 100
MAX_POSTHOC_CSV_CELL_CHARS = 10_000


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file, returning None on missing or invalid data."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_POSTHOC_FILE_BYTES:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_text(path: Path) -> str | None:
    """Read a text file, returning None on missing or error."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_POSTHOC_FILE_BYTES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def discover_posthoc_session(
    results_dir: Path,
    posthoc_dir: Path | None = None,
) -> Path | None:
    """Find the posthoc analysis directory containing a valid ranking.json.

    Expects the layout ``results_dir/posthoc/{ranking.json, analyses/}``.
    Returns the ``posthoc/`` directory, or ``None`` if it does not exist.
    """
    if posthoc_dir is not None:
        if posthoc_dir.expanduser().is_symlink():
            return None
        explicit = posthoc_dir.expanduser().resolve()
        return explicit if _is_posthoc_session(explicit) else None

    resolved_results = results_dir.expanduser().resolve()
    unresolved_posthoc_root = resolved_results / "posthoc"
    if unresolved_posthoc_root.is_symlink():
        return None
    posthoc_root = unresolved_posthoc_root.resolve()
    if not posthoc_root.is_relative_to(resolved_results):
        return None
    stable_manifest = _read_json(posthoc_root / "latest_manifest.json")
    if stable_manifest is not None:
        output_dir = stable_manifest.get("output_dir")
        if isinstance(output_dir, str) and output_dir:
            raw_output = Path(output_dir).expanduser()
            candidate = (
                raw_output.resolve()
                if raw_output.is_absolute()
                else (posthoc_root / raw_output).resolve()
            )
            if candidate.is_relative_to(posthoc_root) and _is_posthoc_session(candidate):
                return candidate
    return posthoc_root if _is_posthoc_session(posthoc_root) else None


def _is_posthoc_session(path: Path) -> bool:
    ranking_path = path / "ranking.json"
    analyses_path = path / "analyses"
    return (
        path.is_dir()
        and not path.is_symlink()
        and ranking_path.is_file()
        and not ranking_path.is_symlink()
        and analyses_path.is_dir()
        and not analyses_path.is_symlink()
    )


def _validate_path_component(component: str, name: str) -> str:
    """Validate an untrusted ranking.json path component."""
    if not component or component in {".", ".."} or "/" in component or "\\" in component:
        raise ValueError(f"Invalid {name} in ranking.json: {component!r}")
    return component


def _resolve_attempt_dir(posthoc_root: Path, analysis_id: str, selected_attempt: str) -> Path:
    """Resolve the attempt directory structurally, rejecting traversal from ranking.json."""
    analyses_root = (posthoc_root / "analyses").resolve()
    safe_analysis_id = _validate_path_component(analysis_id, "analysis_id")
    safe_selected_attempt = _validate_path_component(selected_attempt, "selected_attempt")
    attempt_dir = (analyses_root / safe_analysis_id / safe_selected_attempt).resolve()
    try:
        attempt_dir.relative_to(analyses_root)
    except ValueError as exc:
        raise ValueError("Resolved attempt directory escapes posthoc analyses root") from exc
    unresolved_analysis = analyses_root / safe_analysis_id
    unresolved_attempt = unresolved_analysis / safe_selected_attempt
    if unresolved_analysis.is_symlink() or unresolved_attempt.is_symlink():
        raise ValueError("Attempt directory must not be a symlink")
    return attempt_dir


def _collect_png_paths(directory: Path) -> list[str]:
    """List PNG files in a directory as absolute path strings."""
    if not directory.is_dir():
        return []
    return sorted(
        str(p.resolve())
        for p in directory.iterdir()
        if p.is_file()
        and not p.is_symlink()
        and p.suffix.lower() == ".png"
        and p.stat().st_size <= MAX_POSTHOC_FILE_BYTES
    )


def _resolve_declared_file(directory: Path, relpath: object) -> Path | None:
    if not isinstance(relpath, str) or not relpath:
        return None
    raw_path = Path(relpath)
    if raw_path.is_absolute():
        return None
    root = directory.resolve()
    unresolved = root / raw_path
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(root):
        return None
    if _path_has_symlink(root, raw_path) or not resolved.is_file():
        return None
    if resolved.stat().st_size > MAX_POSTHOC_FILE_BYTES:
        return None
    return resolved


def _path_has_symlink(root: Path, relpath: Path) -> bool:
    current = root
    for component in relpath.parts:
        current /= component
        if current.is_symlink():
            return True
    return False


def _manifest_output_paths(manifest: dict[str, Any], key: str) -> list[object]:
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        return []
    values = outputs.get(key, [])
    return values if isinstance(values, list) else []


def _collect_manifest_figures(directory: Path, manifest: dict[str, Any]) -> list[str]:
    figures: list[str] = []
    for relpath in _manifest_output_paths(manifest, "figures")[:MAX_POSTHOC_DATA_FILES]:
        path = _resolve_declared_file(directory, relpath)
        if path is not None and path.suffix.lower() == ".png":
            figures.append(str(path))
    return sorted(figures)


def _collect_manifest_data(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    total_bytes = 0
    for relpath in _manifest_output_paths(manifest, "data_files")[:MAX_POSTHOC_DATA_FILES]:
        path = _resolve_declared_file(directory, relpath)
        if path is None or path.suffix.lower() not in {".json", ".csv"}:
            continue
        total_bytes += path.stat().st_size
        if total_bytes > MAX_POSTHOC_ANALYSIS_BYTES:
            logger.warning("Posthoc analysis data exceeds byte budget; skipping %s", path)
            break
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                with path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if len(reader.fieldnames or []) > MAX_POSTHOC_CSV_COLUMNS:
                        continue
                    rows: list[dict[str, str | None]] = []
                    for row_index, row in enumerate(reader):
                        if row_index >= MAX_POSTHOC_CSV_ROWS:
                            rows = []
                            break
                        if any(len(str(value)) > MAX_POSTHOC_CSV_CELL_CHARS for value in row.values()):
                            rows = []
                            break
                        rows.append(row)
                    payload = rows
        except (OSError, UnicodeDecodeError, ValueError, csv.Error, RecursionError):
            continue
        data[str(relpath)] = payload
    return data


# Infrastructure / control JSON files that are NOT raw analysis data.
# These are either loaded into dedicated fields (e.g. rubric) or are
# pipeline bookkeeping and should not appear in ``raw_data``.
_CONTROL_JSON_NAMES: frozenset[str] = frozenset(
    {
        "active_idea.json",
        "active_rubrics.json",
        "analysis_manifest.json",
        "analysis_spec.json",
        "attempt_result.json",
        "interestingness_rubric.json",
        "rubric_contract.json",
        "rubric_status.json",
        "target_rubrics.json",
    }
)


def _collect_data_files(directory: Path) -> dict[str, Any]:
    """Read raw-data JSON files in *directory*, skipping control/metadata files.

    Returns a dict mapping ``filename_stem`` → parsed JSON content.
    """
    if not directory.is_dir():
        return {}
    data: dict[str, Any] = {}
    total_bytes = 0
    file_count = 0
    for p in sorted(directory.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        if p.name in _CONTROL_JSON_NAMES:
            continue
        file_count += 1
        total_bytes += p.stat().st_size
        if file_count > MAX_POSTHOC_DATA_FILES or total_bytes > MAX_POSTHOC_ANALYSIS_BYTES:
            break
        payload = _read_json(p)
        if payload is not None:
            data[p.stem] = payload
    return data


def collect_posthoc_analyses(
    results_dir: Path,
    posthoc_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load all valid posthoc analyses from the results directory.

    Returns a list of dicts, each containing:
    - ``analysis_id``, ``title``, ``family``
    - ``summary_text``: the full markdown summary
    - ``rubric``: the parsed interestingness_rubric.json dict
    - ``figure_paths``: list of absolute PNG paths in the attempt dir
    - ``attempt_dir``: resolved path to the selected attempt
    """
    resolved = results_dir.expanduser().resolve()
    posthoc_root = discover_posthoc_session(resolved, posthoc_dir=posthoc_dir)
    if posthoc_root is None:
        return []

    ranking = _read_json(posthoc_root / "ranking.json")
    if ranking is None:
        return []

    valid_entries = ranking.get("valid")
    if not isinstance(valid_entries, list):
        return []

    analyses: list[dict[str, Any]] = []
    context_bytes = 0
    for entry in valid_entries[:MAX_POSTHOC_ANALYSES]:
        if not isinstance(entry, dict):
            continue
        analysis_id = entry.get("analysis_id")
        selected_attempt = entry.get("selected_attempt")
        if not isinstance(analysis_id, str) or not isinstance(selected_attempt, str):
            continue

        try:
            attempt_dir = _resolve_attempt_dir(posthoc_root, analysis_id, selected_attempt)
        except ValueError:
            continue
        if not attempt_dir.is_dir():
            continue

        summary_text = _read_text(attempt_dir / "analysis_summary.md")
        if not summary_text:
            continue

        rubric = _read_json(attempt_dir / "interestingness_rubric.json") or {}
        manifest_path = attempt_dir / "analysis_manifest.json"
        manifest_present = manifest_path.exists() or manifest_path.is_symlink()
        manifest = _read_json(manifest_path) or {}
        figure_paths = _collect_manifest_figures(attempt_dir, manifest)
        if not manifest_present:
            figure_paths = _collect_png_paths(attempt_dir)
        raw_data = _collect_manifest_data(attempt_dir, manifest)
        if not manifest_present:
            raw_data = _collect_data_files(attempt_dir)
        run_analysis_code = _read_text(attempt_dir / "run_analysis.py") or ""

        analysis = {
            "analysis_id": analysis_id,
            "title": entry.get("title", analysis_id),
            "family": entry.get("family", ""),
            "summary_text": summary_text,
            "rubric": rubric,
            "figure_paths": figure_paths,
            "raw_data": raw_data,
            "run_analysis_code": run_analysis_code,
            "attempt_dir": str(attempt_dir),
        }
        analysis_bytes = len(json.dumps(analysis, ensure_ascii=False).encode("utf-8"))
        if analysis_bytes > MAX_POSTHOC_ANALYSIS_BYTES:
            logger.warning("Posthoc analysis %s exceeds context budget; skipping", analysis_id)
            continue
        context_bytes += analysis_bytes
        if context_bytes > MAX_POSTHOC_CONTEXT_BYTES:
            logger.warning("Posthoc context exceeds global byte budget; stopping collection")
            break
        analyses.append(analysis)

    return analyses


def copy_posthoc_figures(analyses: list[dict[str, Any]], dest_dir: Path) -> None:
    """Skip pre-computed PNGs — the paper writer generates its own figures from raw data.

    Logs a warning for each analysis that has pre-existing PNGs so the user
    knows they are intentionally not copied.
    """
    for analysis in analyses:
        analysis_id = analysis.get("analysis_id", "")
        figure_paths = analysis.get("figure_paths", [])
        if figure_paths:
            logger.warning(
                "Posthoc analysis %s has %d pre-computed PNG(s) — skipping copy; "
                "the paper writer will generate figures from raw data instead.",
                analysis_id,
                len(figure_paths),
            )


def validate_posthoc_ranking(
    ranking_result: dict[str, Any], analyses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate and rank every collected analysis exactly once."""
    errors: list[str] = []
    expected_ids: list[str] = []
    for index, analysis in enumerate(analyses):
        analysis_id = analysis.get("analysis_id")
        if not isinstance(analysis_id, str) or not analysis_id.strip():
            errors.append(f"analyses[{index}].analysis_id must be a non-empty string")
            continue
        expected_ids.append(analysis_id)
    duplicate_source_ids = sorted(
        analysis_id
        for analysis_id in set(expected_ids)
        if expected_ids.count(analysis_id) > 1
    )
    if duplicate_source_ids:
        errors.append(f"collected analyses contain duplicate IDs: {duplicate_source_ids}")

    rationale = ranking_result.get("ranking_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("ranking_rationale must be a non-empty string")

    ranked_entries = ranking_result.get("ranked_analyses")
    if not isinstance(ranked_entries, list):
        errors.append("ranked_analyses must be a list")
        raise ValueError("Invalid posthoc ranking: " + "; ".join(errors))

    ranked_ids: list[str] = []
    ranks: list[int] = []
    validated_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(ranked_entries):
        if not isinstance(entry, dict):
            errors.append(f"ranked_analyses[{index}] must be an object")
            continue
        validated_entries.append(entry)
        analysis_id = entry.get("analysis_id")
        if not isinstance(analysis_id, str) or not analysis_id.strip():
            errors.append(
                f"ranked_analyses[{index}].analysis_id must be a non-empty string"
            )
        else:
            ranked_ids.append(analysis_id)

        rank = entry.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            errors.append(f"ranked_analyses[{index}].rank must be an integer")
        else:
            ranks.append(rank)

        relevance = entry.get("relevance_to_thesis")
        if relevance not in POSTHOC_RELEVANCE_VALUES:
            errors.append(
                f"ranked_analyses[{index}].relevance_to_thesis must be one of "
                f"{sorted(POSTHOC_RELEVANCE_VALUES)}"
            )
        usage = entry.get("recommended_usage")
        if usage not in POSTHOC_USAGE_VALUES:
            errors.append(
                f"ranked_analyses[{index}].recommended_usage must be one of "
                f"{sorted(POSTHOC_USAGE_VALUES)}"
            )
        for field in ("relevance_justification", "key_finding_for_paper"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"ranked_analyses[{index}].{field} must be a non-empty string")

    duplicate_ranked_ids = sorted(
        analysis_id
        for analysis_id in set(ranked_ids)
        if ranked_ids.count(analysis_id) > 1
    )
    if duplicate_ranked_ids:
        errors.append(f"duplicate analysis IDs: {duplicate_ranked_ids}")
    expected_id_set = set(expected_ids)
    ranked_id_set = set(ranked_ids)
    unknown_ids = sorted(ranked_id_set - expected_id_set)
    missing_ids = sorted(expected_id_set - ranked_id_set)
    if unknown_ids:
        errors.append(f"unknown analysis IDs: {unknown_ids}")
    if missing_ids:
        errors.append(f"missing analysis IDs: {missing_ids}")
    if sorted(ranks) != list(range(1, len(expected_ids) + 1)):
        errors.append(
            f"ranks must be the contiguous integers 1..{len(expected_ids)} exactly once"
        )

    if errors:
        raise ValueError("Invalid posthoc ranking: " + "; ".join(errors))
    return sorted(validated_entries, key=lambda entry: entry["rank"])


def augment_paper_data_with_posthoc(
    paper_data: dict[str, Any],
    ranking_result: dict[str, Any],
    analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Add ranked posthoc analyses without changing existing evidence.

    ``ranking_result`` is the parsed ``posthoc_ranking.json`` produced by the
    ranking agent.  ``analyses`` is the output of ``collect_posthoc_analyses``.
    Existing inventory entries, totals, and grounding records remain authoritative.
    """
    analyses_by_id = {a["analysis_id"]: a for a in analyses}

    ranked_entries = validate_posthoc_ranking(ranking_result, analyses)

    existing_posthoc = paper_data.get("posthoc_analyses")
    posthoc_list = list(existing_posthoc) if isinstance(existing_posthoc, list) else []
    existing_ids = {
        item.get("analysis_id")
        for item in posthoc_list
        if isinstance(item, dict) and isinstance(item.get("analysis_id"), str)
    }
    for entry in ranked_entries:
        aid = entry["analysis_id"]
        if aid in existing_ids:
            continue
        source = analyses_by_id[aid]
        posthoc_list.append(
            {
                "rank": entry.get("rank"),
                "analysis_id": aid,
                "title": source.get("title", aid),
                "family": source.get("family", ""),
                "summary_text": source.get("summary_text", ""),
                "rubric": source.get("rubric", {}),
                "relevance_to_thesis": entry.get("relevance_to_thesis", ""),
                "relevance_justification": entry.get("relevance_justification", ""),
                "recommended_usage": entry.get("recommended_usage", ""),
                "key_finding_for_paper": entry.get("key_finding_for_paper", ""),
                "figure_paths": source.get("figure_paths", []),
                "raw_data": source.get("raw_data", {}),
                "run_analysis_code": source.get("run_analysis_code", ""),
            }
        )

    paper_data["posthoc_analyses"] = posthoc_list

    # Update totals
    totals = paper_data.get("totals")
    if isinstance(totals, dict):
        totals.setdefault("posthoc_analyses_total", len(analyses))
        totals.setdefault("posthoc_analyses_ranked", len(posthoc_list))

    # Register grounding sources
    grounding_sources = paper_data.get("grounding_sources")
    if isinstance(grounding_sources, dict):
        for item in posthoc_list:
            if not isinstance(item, dict):
                continue
            aid = item.get("analysis_id")
            if not isinstance(aid, str) or not aid:
                continue
            grounding_sources.setdefault(
                f"posthoc.{aid}.summary",
                {
                    "id": f"posthoc.{aid}.summary",
                    "type": "posthoc_analysis",
                    "analysis_id": aid,
                    "title": item.get("title", ""),
                    "family": item.get("family", ""),
                },
            )
            # Register each raw data file as a grounding source.
            raw_data = item.get("raw_data")
            if not isinstance(raw_data, dict):
                continue
            for data_stem in raw_data:
                if not isinstance(data_stem, str) or not data_stem:
                    continue
                source_id = f"posthoc.{aid}.data.{data_stem}"
                grounding_sources.setdefault(
                    source_id,
                    {
                        "id": source_id,
                        "type": "posthoc_data",
                        "analysis_id": aid,
                        "data_file": data_stem,
                    },
                )

    return paper_data
