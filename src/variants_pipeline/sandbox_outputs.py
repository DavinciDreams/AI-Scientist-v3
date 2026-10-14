"""Canonicalize bounded evaluator output before host promotion."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.investigation.evaluation_evidence import (
    EVIDENCE_FILENAME, evidence_timeseries, load_evaluation_evidence,
)

from .constants import ANALYSIS_CSV_SCHEMAS

MAX_SANDBOX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_SANDBOX_ENTRIES = 20_000
MAX_ANALYSIS_ROWS = 200_000
_MAX_SANDBOX_DEPTH = 8
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_JSON_DEPTH = 8
_MAX_METRICS = 5_000
_MAX_FIELD_LENGTH = 512
_SAFE_LABEL = re.compile(r"[A-Za-z0-9_.:/|\-]{1,160}\Z")
_PROTOCOL_MAGIC = b"AGENTIC_RESULTS_V1\n"
_MAX_PROTOCOL_BYTES = MAX_SANDBOX_OUTPUT_BYTES + 4096
PROMOTABLE_RESULT_FILES = frozenset(
    {
        "_results.json",
        "metrics.json",
        "correct.json",
        EVIDENCE_FILENAME,
        "rubric_status.json",
        "harness_evaluate_stdout.log",
        "harness_evaluate_stderr.log",
    }
)
PROMOTABLE_RESULT_PATHS = (
    PROMOTABLE_RESULT_FILES | {"analysis"} | {f"analysis/{name}" for name in ANALYSIS_CSV_SCHEMAS}
)


def canonicalize_results(
    results_dir: Path,
    program_name: str,
    environment: Mapping[str, str],
) -> None:
    """Replace evaluator-authored analysis with trusted canonical analysis."""
    _validate_bounded_tree(results_dir)
    load_evaluation_evidence(results_dir)
    metrics = _load_required_mapping(results_dir / "metrics.json")
    correct_payload = _load_required_mapping(results_dir / "correct.json")
    if not isinstance(correct_payload.get("correct"), bool):
        raise RuntimeError("sandbox correct.json must contain a boolean 'correct'")

    evaluation_id = _trusted_label(environment, "EVALUATION_ID")
    variant_id = _trusted_label(environment, "VARIANT_ID")
    analysis_tmp = Path(tempfile.mkdtemp(prefix=".canonical-analysis-", dir=results_dir))
    try:
        _write_canonical_analysis(
            analysis_tmp,
            results_dir,
            program_name,
            environment,
            evaluation_id,
            variant_id,
            metrics,
            correct_payload["correct"],
        )
        analysis_dir = results_dir / "analysis"
        if analysis_dir.exists() or analysis_dir.is_symlink():
            _remove_tree(analysis_dir)
        os.replace(analysis_tmp, analysis_dir)
    except BaseException:
        _remove_tree(analysis_tmp)
        raise


def validate_promotable_results(
    results_dir: Path,
    environment: Mapping[str, str],
    program_name: str,
) -> None:
    """Validate the exact host-visible output contract and its contents."""
    _validate_bounded_tree(results_dir)
    load_evaluation_evidence(results_dir)
    paths = list(results_dir.rglob("*"))
    unexpected: list[str] = []
    for path in paths:
        relative = path.relative_to(results_dir).as_posix()
        expected_type = path.is_dir() if relative == "analysis" else path.is_file()
        if relative not in PROMOTABLE_RESULT_PATHS or not expected_type:
            unexpected.append(relative)
    if unexpected:
        raise RuntimeError(f"sandbox produced unexpected output: {', '.join(sorted(unexpected))}")

    metrics = _load_required_mapping(results_dir / "metrics.json")
    correct = _load_required_mapping(results_dir / "correct.json").get("correct")
    if not isinstance(correct, bool):
        raise RuntimeError("sandbox correct.json must contain a boolean 'correct'")
    numeric_metrics = dict(_numeric_metrics(metrics))
    combined_score = numeric_metrics.get("combined_score")
    if combined_score is None:
        raise RuntimeError("sandbox metrics.json must contain finite combined_score")

    evaluation_id = _trusted_label(environment, "EVALUATION_ID")
    variant_id = _trusted_label(environment, "VARIANT_ID")
    summaries = _read_csv(results_dir / "analysis/run_summary.csv", "run_summary.csv", 1)
    if len(summaries) != 1:
        raise RuntimeError("canonical run_summary.csv must contain one row")
    summary = summaries[0]
    expected_summary = {
        "evaluation_id": evaluation_id,
        "variant_id": variant_id,
        "variant_kind": _trusted_label(environment, "VARIANT_KIND"),
        "base_program": _trusted_label(environment, "BASE_PROGRAM"),
        "agent_name": _trusted_label(environment, "AGENT_NAME"),
        "correct": str(correct),
        "combined_score": str(combined_score),
        "overall_pass": "",
        "primary_metric_name": "combined_score",
        "primary_metric_value": str(combined_score),
        "program_dir": _safe_label(program_name, "program name"),
        "results_dir": "analysis",
    }
    if summary != expected_summary:
        raise RuntimeError("canonical run_summary.csv does not match trusted metadata")

    metric_rows = _read_csv(
        results_dir / "analysis/metric_values.csv", "metric_values.csv", _MAX_METRICS
    )
    expected_metric_rows = [
        {
            "evaluation_id": evaluation_id,
            "variant_id": variant_id,
            "metric_name": name,
            "metric_value": str(value),
            "is_primary_metric": str(name == "combined_score"),
        }
        for name, value in _numeric_metrics(metrics)
    ]
    if metric_rows != expected_metric_rows:
        raise RuntimeError("canonical metric_values.csv does not match metrics.json")

    timeseries_rows = _read_csv(
        results_dir / "analysis/timeseries.csv", "timeseries.csv", MAX_ANALYSIS_ROWS
    )
    for row in timeseries_rows:
        if row["variant_id"] != variant_id:
            raise RuntimeError("canonical timeseries.csv variant identity mismatch")
        if row["evaluation_id"] != evaluation_id:
            raise RuntimeError("canonical timeseries.csv evaluation identity mismatch")
        _safe_label(row["series_name"], "series name")
        step = _finite_float(row["step"])
        if step < 0 or not step.is_integer():
            raise RuntimeError("canonical timeseries.csv contains an invalid step")
        _finite_float(row["value"])


def copy_promotable_results(source: Path, destination: Path) -> None:
    """Copy only the fixed promotion allowlist to an empty directory."""
    if any(destination.iterdir()):
        raise RuntimeError("promotion directory must be empty")
    for name in sorted(PROMOTABLE_RESULT_FILES):
        path = source / name
        if path.is_file() and not path.is_symlink():
            shutil.copy2(path, destination / name)
    shutil.copytree(source / "analysis", destination / "analysis")


def build_promotable_frame(source: Path) -> bytes:
    """Build an uncompressed, fixed-path promotion frame."""
    files = [
        (relative, source / relative)
        for relative in sorted(PROMOTABLE_RESULT_PATHS)
        if (source / relative).is_file() and not (source / relative).is_symlink()
    ]
    output = bytearray(_PROTOCOL_MAGIC)
    output.extend(struct.pack("!H", len(files)))
    total_bytes = 0
    for relative, path in files:
        name = relative.encode("ascii")
        content = path.read_bytes()
        total_bytes += len(content)
        if total_bytes > MAX_SANDBOX_OUTPUT_BYTES:
            raise RuntimeError("sandbox output exceeds the 64 MiB promotion limit")
        output.extend(struct.pack("!H", len(name)))
        output.extend(name)
        output.extend(struct.pack("!Q", len(content)))
        output.extend(content)
    if len(output) > _MAX_PROTOCOL_BYTES:
        raise RuntimeError("sandbox promotion frame exceeds its limit")
    return bytes(output)


def extract_promotable_frame(payload: bytes, destination: Path) -> None:
    """Decode one exact bounded frame without archive metadata or compression."""
    if len(payload) > _MAX_PROTOCOL_BYTES or not payload.startswith(_PROTOCOL_MAGIC):
        raise RuntimeError("sandbox produced an invalid promotion frame")
    cursor = len(_PROTOCOL_MAGIC)
    file_count, cursor = _read_frame_integer(payload, cursor, "!H")
    if file_count > len(PROMOTABLE_RESULT_PATHS):
        raise RuntimeError("sandbox produced an invalid promotion frame")
    total_bytes = 0
    seen: set[str] = set()
    for _ in range(file_count):
        name_length, cursor = _read_frame_integer(payload, cursor, "!H")
        name_bytes, cursor = _read_frame_bytes(payload, cursor, name_length)
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise RuntimeError("sandbox produced an invalid promotion frame") from exc
        content_length, cursor = _read_frame_integer(payload, cursor, "!Q")
        content, cursor = _read_frame_bytes(payload, cursor, content_length)
        if name not in PROMOTABLE_RESULT_PATHS or name in seen:
            raise RuntimeError("sandbox produced an invalid promotion frame")
        seen.add(name)
        total_bytes += content_length
        if total_bytes > MAX_SANDBOX_OUTPUT_BYTES:
            raise RuntimeError("sandbox output exceeds the 64 MiB promotion limit")
        destination_path = destination / name
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(content)
    if cursor != len(payload):
        raise RuntimeError("sandbox produced an invalid promotion frame")


def _read_frame_integer(payload: bytes, cursor: int, format_: str) -> tuple[int, int]:
    size = struct.calcsize(format_)
    raw, cursor = _read_frame_bytes(payload, cursor, size)
    return int(struct.unpack(format_, raw)[0]), cursor


def _read_frame_bytes(payload: bytes, cursor: int, length: int) -> tuple[bytes, int]:
    end = cursor + length
    if end > len(payload):
        raise RuntimeError("sandbox produced an invalid promotion frame")
    return payload[cursor:end], end


def _write_canonical_analysis(
    analysis_dir: Path,
    results_dir: Path,
    program_name: str,
    environment: Mapping[str, str],
    evaluation_id: str,
    variant_id: str,
    metrics: Mapping[str, Any],
    correct: bool,
) -> None:
    numeric_metrics = _numeric_metrics(metrics)
    combined_score = dict(numeric_metrics).get("combined_score")
    if combined_score is None:
        raise RuntimeError("sandbox metrics.json must contain finite combined_score")
    _write_csv(
        analysis_dir / "run_summary.csv",
        ANALYSIS_CSV_SCHEMAS["run_summary.csv"],
        [
            {
                "evaluation_id": evaluation_id,
                "variant_id": variant_id,
                "variant_kind": _trusted_label(environment, "VARIANT_KIND"),
                "base_program": _trusted_label(environment, "BASE_PROGRAM"),
                "agent_name": _trusted_label(environment, "AGENT_NAME"),
                "correct": correct,
                "combined_score": combined_score,
                "overall_pass": "",
                "primary_metric_name": "combined_score",
                "primary_metric_value": combined_score,
                "program_dir": _safe_label(program_name, "program name"),
                "results_dir": "analysis",
            }
        ],
    )
    _write_csv(
        analysis_dir / "metric_values.csv",
        ANALYSIS_CSV_SCHEMAS["metric_values.csv"],
        [
            {
                "evaluation_id": evaluation_id,
                "variant_id": variant_id,
                "metric_name": name,
                "metric_value": value,
                "is_primary_metric": name == "combined_score",
            }
            for name, value in numeric_metrics
        ],
    )
    _write_csv(
        analysis_dir / "timeseries.csv",
        ANALYSIS_CSV_SCHEMAS["timeseries.csv"],
        _validated_timeseries(results_dir, evaluation_id, variant_id),
    )


def _validated_timeseries(
    results_dir: Path, evaluation_id: str, variant_id: str
) -> Iterable[dict[str, Any]]:
    evidence_rows = evidence_timeseries(results_dir, evaluation_id, variant_id)
    yield from evidence_rows
    row_count = len(evidence_rows)
    sources = sorted(results_dir.rglob("analysis/timeseries.csv"))
    for source in sources:
        relative = source.relative_to(results_dir)
        scratch_name = "/".join(relative.parts[:-2])
        rows = _read_csv(source, "timeseries.csv", MAX_ANALYSIS_ROWS)
        for row in rows:
            row_count += 1
            if row_count > MAX_ANALYSIS_ROWS:
                raise RuntimeError("sandbox timeseries exceeds the row limit")
            series_name = _safe_label(row["series_name"], "series name")
            if evidence_rows and not scratch_name and series_name.startswith("candidate/seed_"):
                # The fixed evidence artifact is the source for this reserved namespace.
                continue
            if scratch_name:
                series_name = _safe_label(
                    f"{_safe_label(scratch_name, 'scratch name')}/{series_name}",
                    "series name",
                )
            step = _finite_float(row["step"])
            if step < 0 or not step.is_integer():
                raise RuntimeError("sandbox timeseries contains an invalid step")
            value = _optional_timeseries_float(row["value"])
            if value is None:
                continue
            yield {
                "evaluation_id": evaluation_id,
                "variant_id": variant_id,
                "series_name": series_name,
                "step": int(step),
                "value": value,
            }


def _numeric_metrics(
    payload: Mapping[str, Any], prefix: str = "", depth: int = 0
) -> list[tuple[str, float]]:
    if depth > _MAX_JSON_DEPTH:
        raise RuntimeError("sandbox metrics.json exceeds the nesting limit")
    metrics: list[tuple[str, float]] = []
    for key, value in payload.items():
        if not isinstance(key, str):
            raise RuntimeError("sandbox metrics.json contains a non-string key")
        name = _safe_label(f"{prefix}.{key}" if prefix else key, "metric name")
        if isinstance(value, bool) or value is None or isinstance(value, str):
            continue
        if isinstance(value, (int, float)):
            metrics.append((name, _finite_float(value)))
        elif isinstance(value, dict):
            metrics.extend(_numeric_metrics(value, name, depth + 1))
        elif isinstance(value, list):
            continue
        else:
            raise RuntimeError("sandbox metrics.json contains an invalid value")
        if len(metrics) > _MAX_METRICS:
            raise RuntimeError("sandbox metrics.json exceeds the metric limit")
    metrics.sort(key=lambda item: (item[0] != "combined_score", item[0]))
    return metrics


def _validate_bounded_tree(root: Path) -> None:
    entry_count = 0
    total_bytes = 0
    pending = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_count += 1
                relative = Path(entry.path).relative_to(root)
                if entry_count > MAX_SANDBOX_ENTRIES:
                    raise RuntimeError("sandbox output exceeds the entry limit")
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise RuntimeError(f"sandbox produced unsafe output: {relative}")
                if stat.S_ISDIR(metadata.st_mode):
                    if depth + 1 > _MAX_SANDBOX_DEPTH:
                        raise RuntimeError(f"sandbox output exceeds the depth limit: {relative}")
                    pending.append((Path(entry.path), depth + 1))
                elif stat.S_ISREG(metadata.st_mode):
                    total_bytes += metadata.st_size
                    if total_bytes > MAX_SANDBOX_OUTPUT_BYTES:
                        raise RuntimeError("sandbox output exceeds the 64 MiB promotion limit")
                else:
                    raise RuntimeError(f"sandbox produced unsafe output: {relative}")


def _read_csv(path: Path, schema_name: str, max_rows: int) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames != ANALYSIS_CSV_SCHEMAS[schema_name]:
                raise RuntimeError(f"sandbox produced invalid analysis CSV: {path.name}")
            rows: list[dict[str, str]] = []
            for row in reader:
                if len(rows) >= max_rows or None in row:
                    raise RuntimeError(f"sandbox produced invalid analysis CSV: {path.name}")
                if any(len(value or "") > _MAX_FIELD_LENGTH for value in row.values()):
                    raise RuntimeError(f"sandbox analysis field exceeds limit: {path.name}")
                rows.append({key: value or "" for key, value in row.items()})
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RuntimeError(f"cannot read sandbox analysis CSV: {path.name}") from exc


def _load_required_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise RuntimeError(f"sandbox {path.name} exceeds the JSON size limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"sandbox produced invalid {path.name}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"sandbox produced invalid {path.name}")
    return payload


def _trusted_label(environment: Mapping[str, str], key: str) -> str:
    return _safe_label(environment.get(key) or "unknown", key.lower())


def _safe_label(value: str, kind: str) -> str:
    if not _SAFE_LABEL.fullmatch(value):
        raise RuntimeError(f"sandbox produced invalid {kind}")
    return value


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("sandbox produced a non-numeric analysis value") from exc
    if not math.isfinite(number):
        raise RuntimeError("sandbox produced a non-finite analysis value")
    return number


def _optional_timeseries_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        boolean_value = value.strip().casefold()
        if boolean_value == "true":
            return 1.0
        if boolean_value == "false":
            return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("sandbox produced a non-numeric analysis value") from exc
    if math.isnan(number):
        return None
    if not math.isfinite(number):
        raise RuntimeError("sandbox produced a non-finite analysis value")
    return number


def _write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _remove_tree(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
