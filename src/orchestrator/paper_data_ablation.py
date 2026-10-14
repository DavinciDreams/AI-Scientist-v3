"""Ablation-bundle parsing for paper_data.json."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paper_evaluation_evidence import ablation_evaluation_evidence


_IDENTIFIER_RE = re.compile(r"[^a-z0-9]+")


def utc_now() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        compact = value.strip()
        if not compact:
            return None
        try:
            return float(compact)
        except ValueError:
            return None
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        compact = value.strip()
        if not compact:
            return None
        try:
            return int(compact)
        except ValueError:
            return None
    return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        compact = value.strip().lower()
        if compact in {"true", "1", "yes", "y"}:
            return True
        if compact in {"false", "0", "no", "n"}:
            return False
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _slug(value: str) -> str:
    compact = _IDENTIFIER_RE.sub("_", value.strip().lower()).strip("_")
    return compact or "unknown"


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = 0.0
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "n": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def _best_by_score(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(
        entries,
        key=lambda item: (
            float(item.get("combined_score", float("-inf"))),
            -int(item.get("eval_index", 10**9)),
        ),
    )


def _stats(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(values),
        "max": max(values),
    }


def _float_column_values(rows: list[dict[str, str]], column_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _safe_float(row.get(column_name))
        if value is not None:
            values.append(value)
    return values


def _cross_task_variant_metadata(results_dir: Path, task_name: str) -> dict[str, dict[str, Any]]:
    """Load optional variant metadata from extra_tasks/<task>/<variant>/variant_spec.json."""
    task_root = results_dir / "extra_tasks" / task_name
    metadata: dict[str, dict[str, Any]] = {}
    if not task_root.is_dir():
        return metadata

    for variant_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
        spec = _read_json(variant_dir / "variant_spec.json")
        if not isinstance(spec, dict):
            continue
        variant_id = str(spec.get("variant_id") or variant_dir.name).strip()
        if not variant_id:
            continue
        metadata[variant_id] = spec
    return metadata


def _summarize_cross_task_run_summary(
    *,
    rows: list[dict[str, str]],
    task_name: str,
    csv_path: Path,
    variant_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Summarize per-variant stats from aggregated/<task>/run_summary.csv."""
    task_payload: dict[str, Any] = {}
    grounding_sources: dict[str, dict[str, Any]] = {}
    task_key = _slug(task_name)

    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        variant_id = (row.get("variant_id") or "").strip()
        if not variant_id:
            continue
        by_variant[variant_id].append(row)

    for variant_id, variant_rows in by_variant.items():
        primary_metric_name = (variant_rows[0].get("primary_metric_name") or "").strip()
        metric_name = primary_metric_name or "combined_score"
        variant_kind = (variant_rows[0].get("variant_kind") or "").strip() or None
        if variant_kind is None:
            variant_kind = str(variant_metadata.get(variant_id, {}).get("kind") or "").strip() or None
        values = _float_column_values(variant_rows, "primary_metric_value")
        if not values:
            metric_name = "combined_score"
            values = _float_column_values(variant_rows, "combined_score")
        if not values:
            continue
        stats = _stats(values)
        correct_count = sum(1 for row in variant_rows if _safe_bool(row.get("correct")) is True)
        task_payload[variant_id] = {
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            "metric_name": metric_name,
            "n": int(stats["n"]),
            "correct_count": correct_count,
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "max": stats.get("max"),
        }
        variant_key = _slug(variant_id)
        metric_key = _slug(metric_name)
        base = f"result.{task_key}.{variant_key}"
        for field in ("mean", "std", "min", "max"):
            value = stats.get(field)
            if value is None:
                continue
            grounding_sources[f"{base}.{metric_key}.{field}"] = {
                "id": f"{base}.{metric_key}.{field}",
                "type": "metric",
                "value": float(value),
                "task_name": task_name,
                "variant_id": variant_id,
                "variant_kind": variant_kind,
                "metric_name": metric_name,
                "summary_field": field,
                "source_csv": str(csv_path),
            }
        grounding_sources[f"{base}.{metric_key}.n"] = {
            "id": f"{base}.{metric_key}.n",
            "type": "count",
            "value": int(stats["n"]),
            "task_name": task_name,
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            "metric_name": metric_name,
            "source_csv": str(csv_path),
        }

    return task_payload, grounding_sources


def _summarize_cross_task_metric_values(
    *,
    rows: list[dict[str, str]],
    task_name: str,
    csv_path: Path,
    task_payload: dict[str, Any],
    variant_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Register summaries for every numeric metric in metric_values.csv."""
    values_by_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        variant_id = str(row.get("variant_id", "")).strip()
        metric_name = str(row.get("metric_name", "")).strip()
        metric_value = _safe_float(row.get("metric_value"))
        if (
            variant_id
            and metric_name
            and metric_value is not None
            and math.isfinite(metric_value)
        ):
            values_by_metric[(variant_id, metric_name)].append(metric_value)

    grounding_sources: dict[str, dict[str, Any]] = {}
    source_identities: dict[str, tuple[str, str]] = {}
    task_key = _slug(task_name)
    for (variant_id, metric_name), values in sorted(values_by_metric.items()):
        try:
            stats = {
                "n": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        except (OverflowError, statistics.StatisticsError):
            continue
        if any(
            not math.isfinite(float(value))
            for field, value in stats.items()
            if field != "n"
        ):
            continue
        variant_kind = None
        variant_payload = task_payload.get(variant_id)
        if isinstance(variant_payload, dict):
            variant_kind = variant_payload.get("variant_kind")
        if variant_kind is None:
            variant_kind = variant_metadata.get(variant_id, {}).get("kind")
        base = f"result.{task_key}.{_slug(variant_id)}.{_slug(metric_name)}"
        identity = (variant_id, metric_name)
        previous_identity = source_identities.setdefault(base, identity)
        if previous_identity != identity:
            raise ValueError(
                "metric grounding source ID collision: "
                f"{base!r} maps to both {previous_identity!r} and {identity!r}"
            )
        for field in ("mean", "std", "min", "max"):
            value = stats.get(field)
            if value is None:
                continue
            source_id = f"{base}.{field}"
            grounding_sources[source_id] = {
                "id": source_id,
                "type": "metric",
                "value": float(value),
                "task_name": task_name,
                "variant_id": variant_id,
                "variant_kind": variant_kind,
                "metric_name": metric_name,
                "summary_field": field,
                "source_csv": str(csv_path),
            }
        count_source_id = f"{base}.n"
        grounding_sources[count_source_id] = {
            "id": count_source_id,
            "type": "count",
            "value": int(stats["n"]),
            "task_name": task_name,
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            "metric_name": metric_name,
            "source_csv": str(csv_path),
        }
    return grounding_sources


def _merge_cross_task_grounding_sources(
    target: dict[str, dict[str, Any]],
    incoming: dict[str, dict[str, Any]],
    *,
    preserve_existing: bool,
) -> None:
    """Merge sources while rejecting lossy identifier normalization."""
    identity_fields = ("task_name", "variant_id", "metric_name")
    for source_id, payload in incoming.items():
        existing = target.get(source_id)
        if existing is None:
            target[source_id] = payload
            continue
        existing_identity = tuple(existing.get(field) for field in identity_fields)
        incoming_identity = tuple(payload.get(field) for field in identity_fields)
        if existing_identity != incoming_identity:
            raise ValueError(
                "metric grounding source ID collision: "
                f"{source_id!r} maps to both {existing_identity!r} "
                f"and {incoming_identity!r}"
            )
        if not preserve_existing:
            target[source_id] = payload


def _preferred_cross_task_metric_name(metric_names: set[str]) -> str | None:
    """Choose a reader-facing primary metric from run_timeseries metric names."""
    if not metric_names:
        return None
    preferred_order = (
        "eval/episode_reward",
        "episode_return",
        "eval_return",
        "reward",
        "return",
    )
    normalized = {name.strip(): name.strip() for name in metric_names if name.strip()}
    for candidate in preferred_order:
        if candidate in normalized:
            return normalized[candidate]
    return sorted(normalized)[0] if normalized else None


def _summarize_cross_task_run_timeseries(
    *,
    rows: list[dict[str, str]],
    task_name: str,
    csv_path: Path,
    variant_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Summarize per-variant stats from aggregated/<task>/run_timeseries.csv."""
    task_payload: dict[str, Any] = {}
    grounding_sources: dict[str, dict[str, Any]] = {}
    task_key = _slug(task_name)

    # Track the terminal value for each (variant, replication, metric).
    terminal_values: dict[tuple[str, str, str], tuple[int, float]] = {}
    metric_names_by_variant: dict[str, set[str]] = defaultdict(set)
    variant_kind_by_id: dict[str, str | None] = {}

    for row in rows:
        algorithm = str(row.get("algorithm", "")).strip()
        variant_id = algorithm or str(row.get("variant_id", "")).strip()
        if not variant_id:
            continue
        replication = str(row.get("replication", "")).strip() or str(row.get("seed", "")).strip() or "0"
        metric_name = str(row.get("metric", "")).strip()
        step = _safe_int(row.get("step"))
        value = _safe_float(row.get("value"))
        if not metric_name or step is None or value is None:
            continue

        metric_names_by_variant[variant_id].add(metric_name)
        if variant_id not in variant_kind_by_id:
            variant_kind_by_id[variant_id] = (
                str(variant_metadata.get(variant_id, {}).get("kind") or "").strip() or None
            )
        key = (variant_id, replication, metric_name)
        current = terminal_values.get(key)
        if current is None or step >= current[0]:
            terminal_values[key] = (step, value)

    for variant_id in sorted(metric_names_by_variant):
        selected_metric_name = _preferred_cross_task_metric_name(metric_names_by_variant[variant_id])
        if selected_metric_name is None:
            continue
        values = [
            value
            for (current_variant_id, _replication, current_metric_name), (_step, value) in terminal_values.items()
            if current_variant_id == variant_id and current_metric_name == selected_metric_name
        ]
        if not values:
            continue
        stats = _stats(values)
        variant_kind = variant_kind_by_id.get(variant_id)
        task_payload[variant_id] = {
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            "metric_name": selected_metric_name,
            "n": int(stats["n"]),
            # RL cross-task timeseries lacks a correctness concept; preserve the field
            # for schema compatibility while using the completed replication count.
            "correct_count": int(stats["n"]),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "source_layout": "run_timeseries_terminal",
        }
        variant_key = _slug(variant_id)
        metric_key = _slug(selected_metric_name)
        base = f"result.{task_key}.{variant_key}"
        for field in ("mean", "std", "min", "max"):
            value = stats.get(field)
            if value is None:
                continue
            grounding_sources[f"{base}.{metric_key}.{field}"] = {
                "id": f"{base}.{metric_key}.{field}",
                "type": "metric",
                "value": float(value),
                "task_name": task_name,
                "variant_id": variant_id,
                "variant_kind": variant_kind,
                "metric_name": selected_metric_name,
                "summary_field": field,
                "source_csv": str(csv_path),
                "source_layout": "run_timeseries_terminal",
            }
        grounding_sources[f"{base}.{metric_key}.n"] = {
            "id": f"{base}.{metric_key}.n",
            "type": "count",
            "value": int(stats["n"]),
            "task_name": task_name,
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            "metric_name": selected_metric_name,
            "source_csv": str(csv_path),
            "source_layout": "run_timeseries_terminal",
        }

    return task_payload, grounding_sources


def collect_cross_task_results(results_dir: Path) -> dict[str, Any]:
    """Scan aggregated/<task>/run_summary.csv for every task and emit per-variant stats.

    Returns a dict with:
      - task_names: sorted list of task names discovered
      - cross_task_results: {task: {variant_id: {variant_kind, metric_name, stats...}}}
      - grounding_sources: {result.<task>.<variant>.<stat>: payload}
    """
    resolved = results_dir.expanduser().resolve()
    aggregated_root = resolved / "aggregated"
    task_names: list[str] = []
    cross_task_results: dict[str, dict[str, Any]] = {}
    grounding_sources: dict[str, dict[str, Any]] = {}
    if not aggregated_root.is_dir():
        return {"task_names": [], "cross_task_results": {}, "grounding_sources": {}}

    for task_dir in sorted(p for p in aggregated_root.iterdir() if p.is_dir()):
        task_name = task_dir.name
        variant_metadata = _cross_task_variant_metadata(resolved, task_name)
        task_payload: dict[str, Any] = {}
        task_sources: dict[str, dict[str, Any]] = {}
        summary_csv_path = task_dir / "run_summary.csv"
        summary_rows = _read_csv_rows(summary_csv_path)
        if summary_rows:
            task_payload, task_sources = _summarize_cross_task_run_summary(
                rows=summary_rows,
                task_name=task_name,
                csv_path=summary_csv_path,
                variant_metadata=variant_metadata,
            )
        else:
            timeseries_csv_path = task_dir / "run_timeseries.csv"
            timeseries_rows = _read_csv_rows(timeseries_csv_path)
            if timeseries_rows:
                task_payload, task_sources = _summarize_cross_task_run_timeseries(
                    rows=timeseries_rows,
                    task_name=task_name,
                    csv_path=timeseries_csv_path,
                    variant_metadata=variant_metadata,
                )

        metric_values_path = task_dir / "metric_values.csv"
        metric_sources = _summarize_cross_task_metric_values(
            rows=_read_csv_rows(metric_values_path),
            task_name=task_name,
            csv_path=metric_values_path,
            task_payload=task_payload,
            variant_metadata=variant_metadata,
        )

        if task_payload:
            task_names.append(task_name)
            cross_task_results[task_name] = task_payload
        _merge_cross_task_grounding_sources(
            grounding_sources,
            task_sources,
            preserve_existing=False,
        )
        _merge_cross_task_grounding_sources(
            grounding_sources,
            metric_sources,
            preserve_existing=True,
        )

    return {
        "task_names": task_names,
        "cross_task_results": cross_task_results,
        "grounding_sources": grounding_sources,
    }


def has_ablation_bundle(results_dir: Path) -> bool:
    """Return whether the results dir looks like a variants-with-reps bundle."""
    resolved = results_dir.expanduser().resolve()
    return (resolved / "variants").is_dir() and (resolved / "aggregated" / "run_summary.csv").is_file()


def has_multitask_ablation_bundle(results_dir: Path) -> bool:
    """Return whether the results dir declares multiple canonical task bundles."""
    resolved = results_dir.expanduser().resolve()
    manifest = _read_json(resolved / "paper_bundle_manifest.json")
    return bool(_iter_multitask_bundle_dirs(resolved, manifest))


def _presentation_contract(paper_profile: str) -> dict[str, Any]:
    if paper_profile == "standalone":
        return {
            "variant_unit": "model_variant",
            "run_unit": "seeded_run",
            "internal_fields": ["correct", "eligible"],
            "forbidden_visible_terms": [
                "eligible",
                "correct and eligible",
                "correctness rate",
                "worker_01",
                "worker_02",
                "combined_score",
                "rubric",
            ],
            "preferred_terms": {
                "idea": "model variant",
                "ideas": "model variants",
                "program": "seeded run",
                "programs": "seeded runs",
                "implementation": "configuration",
                "implementations": "configurations",
            },
            "threshold_rate_rule": (
                "If a thresholded rate appears in the visible paper, keep one reader-facing "
                "meaning per figure, such as target attainment, successful-run fraction, or "
                "non-collapse rate. Do not reuse a single label for different task-specific thresholds."
            ),
        }
    return {
        "variant_unit": "idea",
        "run_unit": "program",
        "internal_fields": ["correct", "eligible"],
    }


def _choose_anchor_variant(variants: list[dict[str, Any]]) -> str | None:
    if not variants:
        return None
    by_id = {str(item.get("variant_id")): item for item in variants if item.get("variant_id")}
    if "control_best" in by_id:
        return "control_best"
    for item in variants:
        if str(item.get("kind", "")).strip().lower() == "control":
            return str(item["variant_id"])
    for item in variants:
        if str(item.get("kind", "")).strip().lower() == "baseline":
            return str(item["variant_id"])
    return str(variants[0].get("variant_id")) if variants else None


def build_ablation_paper_data(
    results_dir: Path,
    *,
    paper_profile: str = "default",
    study_id_override: str | None = None,
    study_title_override: str | None = None,
) -> dict[str, Any]:
    """Build paper_data.json from a variants-with-reps ablation bundle."""
    resolved = results_dir.expanduser().resolve()
    pipeline_summary = _read_json(resolved / "pipeline_summary.json") or {}
    pipeline_variant_rows = pipeline_summary.get("variants")
    pipeline_variants: dict[str, dict[str, Any]] = {}
    if isinstance(pipeline_variant_rows, list):
        for item in pipeline_variant_rows:
            if not isinstance(item, dict):
                continue
            variant_id = item.get("variant_id")
            if isinstance(variant_id, str) and variant_id.strip():
                pipeline_variants[variant_id.strip()] = item

    variant_dirs = sorted(path for path in (resolved / "variants").iterdir() if path.is_dir())
    run_rows = _read_csv_rows(resolved / "aggregated" / "run_summary.csv")
    metric_rows = _read_csv_rows(resolved / "aggregated" / "metric_values.csv")

    runs_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metrics_by_variant: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    primary_metric_flags: dict[str, dict[str, bool]] = defaultdict(dict)
    all_programs: list[dict[str, Any]] = []

    for eval_index, row in enumerate(run_rows, start=1):
        variant_id = str(row.get("variant_id", "")).strip()
        if not variant_id:
            continue
        run_id = str(row.get("run_id", "")).strip() or f"run_{eval_index:03d}"
        evaluation_id = str(row.get("evaluation_id", "")).strip() or None
        program_dir = str(row.get("program_dir", "")).strip() or None
        results_dir_value = str(row.get("results_dir", "")).strip() or None
        primary_metric_name = str(row.get("primary_metric_name", "")).strip() or "combined_score"
        primary_metric_value = _safe_float(row.get("primary_metric_value"))
        combined_score = _safe_float(row.get("combined_score"))
        if combined_score is None:
            combined_score = primary_metric_value
        if combined_score is None:
            continue
        correct = _safe_bool(row.get("correct"))
        overall_pass = _safe_bool(row.get("overall_pass"))
        program_idx = len(runs_by_variant[variant_id])
        evaluation_evidence = ablation_evaluation_evidence(resolved, variant_id, program_dir)
        run_payload = {
            "run_id": run_id,
            "evaluation_id": evaluation_id,
            "variant_id": variant_id,
            "variant_kind": str(row.get("variant_kind", "")).strip() or None,
            "base_program": str(row.get("base_program", "")).strip() or None,
            "agent_name": str(row.get("agent_name", "")).strip() or None,
            "correct": correct,
            "overall_pass": overall_pass,
            "combined_score": combined_score,
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": primary_metric_value,
            "program_dir": program_dir,
            "results_dir": results_dir_value,
            "eval_index": eval_index,
            "evaluation_evidence": evaluation_evidence,
        }
        runs_by_variant[variant_id].append(run_payload)
        all_programs.append(
            {
                "idea_id": variant_id,
                "idea_dir_name": variant_id,
                "attempt_id": f"{variant_id}_attempt_01",
                "attempt_dir_name": "variant_bundle",
                "program_id": run_id,
                "program_idx": program_idx,
                "combined_score": combined_score,
                "correct": correct,
                "eligible": (correct is True),
                "program_dir": program_dir,
                "eval_index": eval_index,
                "variant_id": variant_id,
                "variant_kind": str(row.get("variant_kind", "")).strip() or None,
                "run_id": run_id,
                "evaluation_id": evaluation_id,
                "results_dir": results_dir_value,
                "source_layout": "ablation_bundle",
                "evaluation_evidence": evaluation_evidence,
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": primary_metric_value,
            }
        )

    for row in metric_rows:
        variant_id = str(row.get("variant_id", "")).strip()
        metric_name = str(row.get("metric_name", "")).strip()
        metric_value = _safe_float(row.get("metric_value"))
        if not variant_id or not metric_name or metric_value is None:
            continue
        metrics_by_variant[variant_id][metric_name].append(metric_value)
        primary_flag = _safe_bool(row.get("is_primary_metric"))
        if primary_flag is True:
            primary_metric_flags[variant_id][metric_name] = True
        elif metric_name not in primary_metric_flags[variant_id]:
            primary_metric_flags[variant_id][metric_name] = False

    variants: list[dict[str, Any]] = []
    idea_titles: list[str] = []
    for variant_dir in variant_dirs:
        variant_id = variant_dir.name
        variant_spec = _read_json(variant_dir / "variant_spec.json") or {}
        run_manifest = _read_json(variant_dir / "run_manifest.json") or {}
        rubric_contract = _read_json(variant_dir / "rubric_contract.json") or {}
        pipeline_variant = pipeline_variants.get(variant_id, {})
        variant_runs = runs_by_variant.get(variant_id, [])
        metric_summaries: dict[str, dict[str, Any]] = {}
        for metric_name, values in sorted(metrics_by_variant.get(variant_id, {}).items()):
            summary = _summary(values)
            if summary is None:
                continue
            metric_summaries[metric_name] = {
                **summary,
                "is_primary_metric": bool(primary_metric_flags.get(variant_id, {}).get(metric_name, False)),
            }

        title = None
        for candidate in (
            variant_spec.get("title"),
            pipeline_variant.get("title"),
            run_manifest.get("title"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                title = candidate.strip()
                break
        if title is not None:
            idea_titles.append(title)

        kind = None
        for candidate in (
            variant_spec.get("kind"),
            pipeline_variant.get("kind"),
            run_manifest.get("variant_kind"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                kind = candidate.strip()
                break

        selected_primary_metric_name: str | None = None
        if variant_runs:
            first_metric_name = str(variant_runs[0].get("primary_metric_name") or "").strip()
            selected_primary_metric_name = first_metric_name or None
        if selected_primary_metric_name is None:
            for metric_name, is_primary in primary_metric_flags.get(variant_id, {}).items():
                if is_primary:
                    selected_primary_metric_name = metric_name
                    break
        if selected_primary_metric_name is None and metric_summaries:
            selected_primary_metric_name = next(iter(metric_summaries.keys()))

        if selected_primary_metric_name and selected_primary_metric_name not in metric_summaries:
            values = [
                float(run["primary_metric_value"])
                for run in variant_runs
                if _safe_float(run.get("primary_metric_value")) is not None
            ]
            summary = _summary(values)
            if summary is not None:
                metric_summaries[selected_primary_metric_name] = {
                    **summary,
                    "is_primary_metric": True,
                }
        primary_metric_summary = (
            metric_summaries.get(selected_primary_metric_name)
            if selected_primary_metric_name
            else None
        )
        correct_run_count = sum(1 for run in variant_runs if run.get("correct") is True)
        overall_pass_count = sum(1 for run in variant_runs if run.get("overall_pass") is True)

        variant_payload = {
            "variant_id": variant_id,
            "title": title,
            "kind": kind,
            "status": run_manifest.get("status") or pipeline_variant.get("status"),
            "rubric_passed": pipeline_variant.get("rubric_passed"),
            "eval_count": run_manifest.get("eval_count", pipeline_variant.get("eval_count")),
            "exit_code": pipeline_variant.get("exit_code"),
            "variant_dir": str(variant_dir),
            "task_bundle_dir": str(variant_dir / "task_bundle") if (variant_dir / "task_bundle").exists() else None,
            "primary_metric_name": selected_primary_metric_name,
            "primary_metric_summary": primary_metric_summary,
            "metric_summaries": metric_summaries,
            "run_count": len(variant_runs),
            "correct_run_count": correct_run_count,
            "overall_pass_count": overall_pass_count,
            "variant_spec": variant_spec,
            "run_manifest": run_manifest,
            "rubric_contract": rubric_contract,
            "target_rubrics": _read_json(variant_dir / "target_rubrics.json"),
            "pipeline_variant_summary": pipeline_variant,
            "runs": variant_runs,
            "artifacts": {
                "variant_spec": str(variant_dir / "variant_spec.json") if (variant_dir / "variant_spec.json").exists() else None,
                "run_manifest": str(variant_dir / "run_manifest.json") if (variant_dir / "run_manifest.json").exists() else None,
                "rubric_contract": str(variant_dir / "rubric_contract.json") if (variant_dir / "rubric_contract.json").exists() else None,
                "target_rubrics": str(variant_dir / "target_rubrics.json") if (variant_dir / "target_rubrics.json").exists() else None,
                "aggregated_run_summary": str(resolved / "aggregated" / "run_summary.csv"),
                "aggregated_metric_values": str(resolved / "aggregated" / "metric_values.csv"),
                "aggregated_timeseries": str(resolved / "aggregated" / "timeseries.csv") if (resolved / "aggregated" / "timeseries.csv").exists() else None,
            },
        }
        variants.append(variant_payload)

    variants.sort(key=lambda item: str(item.get("variant_id", "")))
    idea_titles = list(dict.fromkeys(title for title in idea_titles if title))

    anchor_variant_id = _choose_anchor_variant(variants)
    anchor_variant = next((item for item in variants if item.get("variant_id") == anchor_variant_id), None)
    comparisons: list[dict[str, Any]] = []
    if anchor_variant is not None:
        anchor_metric_name = anchor_variant.get("primary_metric_name")
        anchor_summary = anchor_variant.get("primary_metric_summary")
        anchor_mean = None
        if isinstance(anchor_summary, dict):
            anchor_mean = _safe_float(anchor_summary.get("mean"))
        for variant in variants:
            variant_id = variant.get("variant_id")
            if not isinstance(variant_id, str) or variant_id == anchor_variant_id:
                continue
            variant_summary = variant.get("primary_metric_summary")
            variant_mean = None
            if isinstance(variant_summary, dict):
                variant_mean = _safe_float(variant_summary.get("mean"))
            if anchor_mean is None or variant_mean is None:
                continue
            metric_name = str(variant.get("primary_metric_name") or anchor_metric_name or "combined_score")
            comparisons.append(
                {
                    "comparison_id": f"{anchor_variant_id}__vs__{variant_id}",
                    "baseline_variant_id": anchor_variant_id,
                    "variant_id": variant_id,
                    "metric_name": metric_name,
                    "baseline_mean": anchor_mean,
                    "variant_mean": variant_mean,
                    "mean_delta": variant_mean - anchor_mean,
                }
            )

    all_programs.sort(key=lambda row: int(row["eval_index"]))
    correct_programs = [row for row in all_programs if row.get("correct") is True]
    top_programs = sorted(
        correct_programs,
        key=lambda row: (
            -float(row["combined_score"]),
            int(row["eval_index"]),
        ),
    )[:10]
    best_correct = _best_by_score(correct_programs)
    best_any = _best_by_score(all_programs)
    improvement_trace: list[dict[str, Any]] = []

    study_id = _slug(study_id_override or resolved.name)
    completed_variants = sum(1 for item in variants if item.get("status") == "completed")
    failed_variants = sum(1 for item in variants if item.get("status") == "failed")
    timed_out_variants = sum(1 for item in variants if item.get("status") == "timeout")
    totals = {
        "ideas": len(variants),
        "attempts": len(variants),
        "evaluated_programs": len(all_programs),
        "correct_programs": len(correct_programs),
        "ablation_studies": 1,
        "ablation_variants": len(variants),
        "ablation_runs": len(all_programs),
        "completed_ablation_variants": completed_variants,
        "failed_ablation_variants": failed_variants,
        "timed_out_ablation_variants": timed_out_variants,
    }

    grounding_sources: dict[str, dict[str, Any]] = {}

    def register_source(source_id: str, payload: dict[str, Any]) -> None:
        grounding_sources[source_id] = {"id": source_id, **payload}

    if isinstance(best_correct, dict):
        register_source(
            "result.best_correct_score",
            {
                "type": "metric",
                "value": float(best_correct["combined_score"]),
                "program_dir": best_correct.get("program_dir"),
                "idea_id": best_correct.get("idea_id"),
                "attempt_id": best_correct.get("attempt_id"),
                "program_id": best_correct.get("program_id"),
            },
        )

    if isinstance(best_any, dict):
        register_source(
            "result.best_any_score",
            {
                "type": "metric",
                "value": float(best_any["combined_score"]),
                "program_dir": best_any.get("program_dir"),
                "idea_id": best_any.get("idea_id"),
                "attempt_id": best_any.get("attempt_id"),
                "program_id": best_any.get("program_id"),
            },
        )

    register_source("result.total_ideas", {"type": "count", "value": len(variants)})
    register_source("result.total_attempts", {"type": "count", "value": len(variants)})
    register_source("result.total_evaluated_programs", {"type": "count", "value": len(all_programs)})
    register_source("result.total_correct_programs", {"type": "count", "value": len(correct_programs)})
    register_source("result.total_ablation_variants", {"type": "count", "value": len(variants)})
    register_source("result.total_ablation_runs", {"type": "count", "value": len(all_programs)})

    for idx, row in enumerate(top_programs, start=1):
        register_source(
            f"result.top_program_{idx:02d}",
            {
                "type": "metric",
                "value": float(row["combined_score"]),
                "program_dir": row.get("program_dir"),
                "idea_id": row.get("idea_id"),
                "attempt_id": row.get("attempt_id"),
                "program_id": row.get("program_id"),
            },
        )

    for variant in variants:
        variant_id = str(variant.get("variant_id") or "")
        if not variant_id:
            continue
        metric_name = str(variant.get("primary_metric_name") or "combined_score")
        metric_key = _slug(metric_name)
        variant_key = _slug(variant_id)
        summary = variant.get("primary_metric_summary")
        if isinstance(summary, dict):
            for field in ("mean", "std", "min", "max"):
                value = _safe_float(summary.get(field))
                if value is None:
                    continue
                register_source(
                    f"ablation.study_{study_id}.variant_{variant_key}.{metric_key}.{field}",
                    {
                        "type": "metric",
                        "value": value,
                        "study_id": study_id,
                        "variant_id": variant_id,
                        "metric_name": metric_name,
                        "summary_field": field,
                    },
                )
            count_value = _safe_int(summary.get("n"))
            if count_value is not None:
                register_source(
                    f"ablation.study_{study_id}.variant_{variant_key}.{metric_key}.n",
                    {
                        "type": "count",
                        "value": count_value,
                        "study_id": study_id,
                        "variant_id": variant_id,
                        "metric_name": metric_name,
                    },
                )

    for comparison in comparisons:
        baseline_variant_id = str(comparison.get("baseline_variant_id") or "")
        variant_id = str(comparison.get("variant_id") or "")
        metric_name = str(comparison.get("metric_name") or "combined_score")
        delta = _safe_float(comparison.get("mean_delta"))
        if not baseline_variant_id or not variant_id or delta is None:
            continue
        register_source(
            (
                f"ablation.study_{study_id}.compare_{_slug(baseline_variant_id)}"
                f"_vs_{_slug(variant_id)}.{_slug(metric_name)}.mean_delta"
            ),
            {
                "type": "metric",
                "value": delta,
                "study_id": study_id,
                "baseline_variant_id": baseline_variant_id,
                "variant_id": variant_id,
                "metric_name": metric_name,
                "summary_field": "mean_delta",
            },
        )

    return {
        "generated_at": utc_now(),
        "paper_profile": paper_profile,
        "presentation_contract": _presentation_contract(paper_profile),
        "results_dir": str(resolved),
        "best_correct": best_correct,
        "best_any": best_any,
        "totals": totals,
        "top_programs": top_programs,
        "improvement_trace": improvement_trace,
        "all_programs": all_programs,
        "idea_titles": idea_titles,
        "grounding_sources": grounding_sources,
        "ablation_studies": [
            {
                "study_id": study_id,
                "title": str(study_title_override or pipeline_summary.get("title") or resolved.name),
                "source_dir": str(resolved),
                "pipeline_summary": pipeline_summary,
                "primary_metric_name": (
                    anchor_variant.get("primary_metric_name")
                    if isinstance(anchor_variant, dict)
                    else None
                ),
                "anchor_variant_id": anchor_variant_id,
                "variant_count": len(variants),
                "run_count": len(all_programs),
                "variants": variants,
                "comparisons": comparisons,
            }
        ],
    }



def _is_within_bundle_root(bundle_root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(bundle_root)
    except ValueError:
        return False
    return True


def _resolve_multitask_bundle_dir(
    bundle_root: Path,
    *,
    task_name: str,
    task_entry: dict[str, Any],
) -> Path | None:
    bundle_root = bundle_root.expanduser().resolve()
    role = str(task_entry.get("role") or "").strip()
    if role == "primary_root_bundle" and has_ablation_bundle(bundle_root):
        return bundle_root

    task_dir = task_entry.get("task_dir")
    if isinstance(task_dir, str) and task_dir.strip():
        candidate = (bundle_root / task_dir.strip()).resolve()
        if _is_within_bundle_root(bundle_root, candidate) and has_ablation_bundle(candidate):
            return candidate

    fallback = (bundle_root / "extra_tasks" / task_name).resolve()
    if _is_within_bundle_root(bundle_root, fallback) and has_ablation_bundle(fallback):
        return fallback
    return None


def _iter_multitask_bundle_dirs(
    bundle_root: Path,
    manifest: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any], Path]]:
    tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
    if not isinstance(tasks, dict):
        return []

    resolved_entries: list[tuple[str, dict[str, Any], Path]] = []
    seen_dirs: set[Path] = set()
    for task_name_raw, task_entry_raw in tasks.items():
        if not isinstance(task_name_raw, str) or not isinstance(task_entry_raw, dict):
            continue
        task_name = task_name_raw.strip()
        if not task_name:
            continue
        bundle_dir = _resolve_multitask_bundle_dir(
            bundle_root,
            task_name=task_name,
            task_entry=task_entry_raw,
        )
        if bundle_dir is None or bundle_dir in seen_dirs:
            continue
        seen_dirs.add(bundle_dir)
        resolved_entries.append((task_name, task_entry_raw, bundle_dir))
    return resolved_entries


def _merge_unique_titles(existing: list[str], additions: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            str(title).strip()
            for title in [*existing, *additions]
            if isinstance(title, str) and str(title).strip()
        )
    )


def build_multitask_ablation_paper_data(
    results_dir: Path,
    *,
    paper_profile: str = "default",
) -> dict[str, Any]:
    """Build one paper_data payload from multiple canonical task bundles."""
    resolved = results_dir.expanduser().resolve()
    manifest = _read_json(resolved / "paper_bundle_manifest.json") or {}
    task_entries = _iter_multitask_bundle_dirs(resolved, manifest)
    if not task_entries:
        return build_ablation_paper_data(resolved, paper_profile=paper_profile)

    studies: list[dict[str, Any]] = []
    all_programs: list[dict[str, Any]] = []
    idea_titles: list[str] = []
    grounding_sources: dict[str, dict[str, Any]] = {}
    completed_variants = 0
    failed_variants = 0
    timed_out_variants = 0

    for task_name, task_entry_raw, bundle_dir in task_entries:
        role = str(task_entry_raw.get("role") or "supporting_task").strip() or "supporting_task"
        task_payload = build_ablation_paper_data(
            bundle_dir,
            paper_profile=paper_profile,
            study_id_override=task_name,
        )
        task_studies = task_payload.get("ablation_studies")
        if not isinstance(task_studies, list) or not task_studies:
            continue
        study = dict(task_studies[0])
        study_id = str(study.get("study_id") or task_name).strip() or task_name
        study["task_name"] = task_name
        study["task_role"] = role
        study["manifest_entry"] = task_entry_raw
        studies.append(study)

        idea_titles = _merge_unique_titles(
            idea_titles,
            [str(title) for title in task_payload.get("idea_titles", []) if isinstance(title, str)],
        )

        for row in task_payload.get("all_programs", []):
            if not isinstance(row, dict):
                continue
            payload = dict(row)
            variant_id = str(payload.get("variant_id") or payload.get("idea_id") or "").strip()
            run_id = str(payload.get("run_id") or payload.get("program_id") or "").strip()
            namespaced_prefix = f"{study_id}:{variant_id}" if variant_id else study_id
            payload["study_id"] = study_id
            payload["task_name"] = task_name
            payload["task_role"] = role
            payload["idea_id"] = namespaced_prefix
            payload["idea_dir_name"] = namespaced_prefix
            if run_id:
                payload["attempt_id"] = f"{namespaced_prefix}:{run_id}"
                payload["program_id"] = f"{namespaced_prefix}:{run_id}"
            else:
                payload["attempt_id"] = namespaced_prefix
                payload["program_id"] = namespaced_prefix
            all_programs.append(payload)

        task_sources = task_payload.get("grounding_sources")
        if isinstance(task_sources, dict):
            for source_id, payload in task_sources.items():
                if not isinstance(payload, dict):
                    continue
                source_key = str(source_id)
                if source_key.startswith("ablation."):
                    grounding_sources[source_key] = dict(payload)

        task_totals = task_payload.get("totals")
        if isinstance(task_totals, dict):
            completed_variants += int(task_totals.get("completed_ablation_variants", 0) or 0)
            failed_variants += int(task_totals.get("failed_ablation_variants", 0) or 0)
            timed_out_variants += int(task_totals.get("timed_out_ablation_variants", 0) or 0)

    all_programs.sort(key=lambda row: (str(row.get("study_id") or ""), int(row.get("eval_index", 10**9))))
    for eval_index, row in enumerate(all_programs, start=1):
        row["eval_index"] = eval_index

    correct_programs = [row for row in all_programs if row.get("correct") is True]
    studies.sort(key=lambda row: (str(row.get("task_role") != "primary_root_bundle"), str(row.get("study_id") or "")))

    totals = {
        "ideas": sum(int(study.get("variant_count") or 0) for study in studies),
        "attempts": sum(int(study.get("variant_count") or 0) for study in studies),
        "evaluated_programs": len(all_programs),
        "correct_programs": len(correct_programs),
        "ablation_studies": len(studies),
        "ablation_variants": sum(int(study.get("variant_count") or 0) for study in studies),
        "ablation_runs": sum(int(study.get("run_count") or 0) for study in studies),
        "completed_ablation_variants": completed_variants,
        "failed_ablation_variants": failed_variants,
        "timed_out_ablation_variants": timed_out_variants,
    }

    grounding_sources["result.total_ideas"] = {
        "id": "result.total_ideas",
        "type": "count",
        "value": totals["ideas"],
    }
    grounding_sources["result.total_attempts"] = {
        "id": "result.total_attempts",
        "type": "count",
        "value": totals["attempts"],
    }
    grounding_sources["result.total_evaluated_programs"] = {
        "id": "result.total_evaluated_programs",
        "type": "count",
        "value": totals["evaluated_programs"],
    }
    grounding_sources["result.total_correct_programs"] = {
        "id": "result.total_correct_programs",
        "type": "count",
        "value": totals["correct_programs"],
    }
    grounding_sources["result.total_ablation_variants"] = {
        "id": "result.total_ablation_variants",
        "type": "count",
        "value": totals["ablation_variants"],
    }
    grounding_sources["result.total_ablation_runs"] = {
        "id": "result.total_ablation_runs",
        "type": "count",
        "value": totals["ablation_runs"],
    }
    grounding_sources["result.total_ablation_studies"] = {
        "id": "result.total_ablation_studies",
        "type": "count",
        "value": totals["ablation_studies"],
    }

    for study in studies:
        study_id = str(study.get("study_id") or "").strip()
        if not study_id:
            continue
        grounding_sources[f"ablation.study_{study_id}.variant_count"] = {
            "id": f"ablation.study_{study_id}.variant_count",
            "type": "count",
            "value": int(study.get("variant_count") or 0),
            "study_id": study_id,
        }
        grounding_sources[f"ablation.study_{study_id}.run_count"] = {
            "id": f"ablation.study_{study_id}.run_count",
            "type": "count",
            "value": int(study.get("run_count") or 0),
            "study_id": study_id,
        }

    return {
        "generated_at": utc_now(),
        "paper_profile": paper_profile,
        "presentation_contract": _presentation_contract(paper_profile),
        "results_dir": str(resolved),
        "best_correct": None,
        "best_any": None,
        "totals": totals,
        "top_programs": [],
        "improvement_trace": [],
        "all_programs": all_programs,
        "idea_titles": idea_titles,
        "grounding_sources": grounding_sources,
        "ablation_studies": studies,
        "paper_bundle_manifest": manifest,
    }
