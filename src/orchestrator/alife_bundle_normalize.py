"""Normalize an ALife paper bundle into task-scoped canonical paper inputs.

This module implements the "option 2A" layout discussed for mixed-task paper
bundles:

- the bundle root becomes a canonical ``ed_conv`` ablation bundle
- each extra task becomes its own canonical task bundle under ``extra_tasks/``
- RL tasks with only ``run_timeseries.csv`` gain derived ``run_summary.csv`` and
  ``metric_values.csv`` using a fixed final-step-per-run reduction rule
- pointer docs are written so paper agents know where the full timeseries live
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

README_NOTE_START = "<!-- BEGIN CANONICAL_PAPER_BUNDLE_NOTE -->"
README_NOTE_END = "<!-- END CANONICAL_PAPER_BUNDLE_NOTE -->"
PRIMARY_TASK = "ed_conv"
CLASSIFICATION_EXTRA_TASK = "ed_mnist"
RL_TASKS = ("brax_ant", "brax_halfcheetah", "brax_humanoid", "craftax")
PRIMARY_METRIC_PREFERENCES = (
    "eval/episode_reward",
    "episode_reward",
    "reward",
)
RUN_SUMMARY_FIELDNAMES = [
    "run_id",
    "evaluation_id",
    "variant_id",
    "variant_kind",
    "base_program",
    "agent_name",
    "correct",
    "combined_score",
    "overall_pass",
    "primary_metric_name",
    "primary_metric_value",
    "program_dir",
    "results_dir",
]
METRIC_VALUES_FIELDNAMES = [
    "run_id",
    "evaluation_id",
    "variant_id",
    "metric_name",
    "metric_value",
    "is_primary_metric",
]


@dataclass(frozen=True)
class NormalizationOutputs:
    bundle_root: Path
    root_canonical_ready: bool
    extra_task_ready: dict[str, bool]
    report_path: Path
    manifest_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_text_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _copy_binary_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _ensure_symlink(dst: Path, target: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    target_rel = Path(os.path.relpath(target, start=dst.parent))
    if dst.is_symlink():
        if dst.resolve() == target.resolve():
            return
        dst.unlink()
    elif dst.exists():
        return
    dst.symlink_to(target_rel, target_is_directory=True)


def _render_summary(values: list[float]) -> dict[str, float | int]:
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


def _replace_marked_block(text: str, replacement: str) -> str:
    block = f"{README_NOTE_START}\n{replacement.rstrip()}\n{README_NOTE_END}"
    if README_NOTE_START in text and README_NOTE_END in text:
        start = text.index(README_NOTE_START)
        end = text.index(README_NOTE_END) + len(README_NOTE_END)
        updated = text[:start] + block + text[end:]
    else:
        sep = "\n\n" if text.strip() else ""
        updated = text.rstrip() + sep + block + "\n"
    return updated


def _write_root_readme_note(bundle_root: Path) -> None:
    readme_path = bundle_root / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    note = """## Canonical paper-writing layout

- The bundle root is normalized as the canonical `ed_conv` paper bundle.
- Root comparative summaries live at `aggregated/run_summary.csv`, `aggregated/metric_values.csv`, and `aggregated/timeseries.csv`.
- Task-sharded originals remain under `aggregated/<task>/...`.
- Canonical extra-task bundles live under `extra_tasks/<task>/`.
- For full training curves, inspect `extra_tasks/<task>/aggregated/run_timeseries.csv` for RL tasks and `extra_tasks/ed_mnist/aggregated/timeseries.csv` for MNIST.
- Use `paper_bundle_manifest.json` for the machine-readable map of task roles and timeseries pointers.
"""
    readme_path.write_text(_replace_marked_block(existing, note), encoding="utf-8")


def _task_pointer_readme(*, task_name: str, aggregated_dir: Path, timeseries_name: str) -> None:
    text = f"""# Aggregated artifacts for `{task_name}`

This directory is normalized for paper writing.

- Canonical per-run summary: `run_summary.csv`
- Canonical per-metric long table: `metric_values.csv`
- Full training curves: `{timeseries_name}`

Reduction rule for derived summaries:
- choose the largest `step` for each `(variant_id, run_id, metric)`
- treat that final value as the paper-facing per-run metric
- compute across-run summaries from those final values only
"""
    (aggregated_dir / "README.md").write_text(text, encoding="utf-8")


def _pipeline_variant_row(
    *,
    variant_id: str,
    title: str,
    kind: str,
    status: str,
    eval_count: int,
    has_run_summary: bool,
    has_metric_values: bool,
    has_timeseries: bool,
) -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "kind": kind,
        "title": title,
        "status": status,
        "rubric_passed": has_run_summary,
        "eval_count": eval_count,
        "exit_code": 0 if status == "completed" else 1,
        "analysis_files": {
            "run_summary.csv": has_run_summary,
            "metric_values.csv": has_metric_values,
            "timeseries.csv": has_timeseries,
            "rubric_status.json": False,
        },
    }


def _write_pipeline_summary(task_root: Path, variant_rows: list[dict[str, Any]]) -> None:
    completed = sum(1 for row in variant_rows if row.get("status") == "completed")
    payload = {
        "total_variants": len(variant_rows),
        "total_completed": completed,
        "total_failed": len(variant_rows) - completed,
        "total_timeout": 0,
        "rubric_passed": sum(1 for row in variant_rows if row.get("rubric_passed") is True),
        "variants": variant_rows,
    }
    _write_json(task_root / "pipeline_summary.json", payload)


def _rewrite_classification_run_summary(
    rows: list[dict[str, str]],
    *,
    bundle_root: Path,
    task_name: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        payload: dict[str, Any] = dict(row)
        variant_id = str(payload.get("variant_id", "")).strip()
        run_id = str(payload.get("run_id", "")).strip()
        if not variant_id or not run_id:
            continue
        if task_name == PRIMARY_TASK:
            variant_root = bundle_root / "variants" / variant_id / "task_bundle" / "program_1"
        else:
            variant_root = (
                bundle_root / "extra_tasks" / task_name / "variants" / variant_id / "task_bundle" / "program_1"
            )
        payload["program_dir"] = str((variant_root / "src").resolve())
        payload["results_dir"] = str((variant_root / run_id).resolve())
        normalized.append(payload)
    return normalized


def _ensure_root_primary_bundle(bundle_root: Path, report_lines: list[str]) -> None:
    source_dir = bundle_root / "aggregated" / PRIMARY_TASK
    target_dir = bundle_root / "aggregated"
    run_rows = _rewrite_classification_run_summary(
        _read_csv_rows(source_dir / "run_summary.csv"),
        bundle_root=bundle_root,
        task_name=PRIMARY_TASK,
    )
    metric_rows = _read_csv_rows(source_dir / "metric_values.csv")
    _write_csv(target_dir / "run_summary.csv", RUN_SUMMARY_FIELDNAMES, run_rows)
    _write_csv(target_dir / "metric_values.csv", METRIC_VALUES_FIELDNAMES, metric_rows)
    if (source_dir / "timeseries.csv").is_file():
        _copy_binary_file(source_dir / "timeseries.csv", target_dir / "timeseries.csv")
    _task_pointer_readme(task_name=PRIMARY_TASK, aggregated_dir=target_dir, timeseries_name="timeseries.csv")
    report_lines.append(
        f"- Root canonical bundle now uses `{PRIMARY_TASK}` summaries at `aggregated/run_summary.csv` and `aggregated/metric_values.csv`."
    )


def _ensure_ed_mnist_bundle(bundle_root: Path, report_lines: list[str]) -> None:
    task_root = bundle_root / "extra_tasks" / CLASSIFICATION_EXTRA_TASK
    sharded_root = bundle_root / "aggregated" / CLASSIFICATION_EXTRA_TASK
    aggregated_dir = task_root / "aggregated"
    run_rows = _rewrite_classification_run_summary(
        _read_csv_rows(sharded_root / "run_summary.csv"),
        bundle_root=bundle_root,
        task_name=CLASSIFICATION_EXTRA_TASK,
    )
    metric_rows = _read_csv_rows(sharded_root / "metric_values.csv")
    _write_csv(aggregated_dir / "run_summary.csv", RUN_SUMMARY_FIELDNAMES, run_rows)
    _write_csv(aggregated_dir / "metric_values.csv", METRIC_VALUES_FIELDNAMES, metric_rows)
    if (sharded_root / "timeseries.csv").is_file():
        _copy_binary_file(sharded_root / "timeseries.csv", aggregated_dir / "timeseries.csv")
    _task_pointer_readme(
        task_name=CLASSIFICATION_EXTRA_TASK,
        aggregated_dir=aggregated_dir,
        timeseries_name="timeseries.csv",
    )
    retro_task_md = task_root / "_retrofitted_bundle" / "task.md"
    if retro_task_md.is_file():
        _copy_text_file(retro_task_md, task_root / "task.md")
    variant_rows: list[dict[str, Any]] = []
    pipeline = _read_json(task_root / "pipeline_summary.json")
    if pipeline is None:
        for variant_dir in sorted((task_root / "variants").iterdir()):
            if not variant_dir.is_dir():
                continue
            spec = _read_json(variant_dir / "variant_spec.json") or {}
            variant_id = variant_dir.name
            title = str(spec.get("title") or variant_id)
            kind = str(spec.get("kind") or "unknown")
            eval_count = sum(1 for row in run_rows if row.get("variant_id") == variant_id)
            variant_rows.append(
                _pipeline_variant_row(
                    variant_id=variant_id,
                    title=title,
                    kind=kind,
                    status="completed" if eval_count > 0 else "failed",
                    eval_count=eval_count,
                    has_run_summary=eval_count > 0,
                    has_metric_values=bool(metric_rows),
                    has_timeseries=(aggregated_dir / "timeseries.csv").is_file(),
                )
            )
        _write_pipeline_summary(task_root, variant_rows)
    report_lines.append(
        "- `extra_tasks/ed_mnist/` now has `task.md`, canonical aggregated files, and task-local paper pointers."
    )


@dataclass(frozen=True)
class _RlRunMetric:
    variant_id: str
    run_id: str
    evaluation_id: str
    metric_name: str
    metric_value: float
    step: int


def _safe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        compact = value.strip()
        if not compact:
            return None
        try:
            return int(float(compact))
        except ValueError:
            return None
    return None


def _safe_float(value: Any) -> float | None:
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


def _infer_primary_metric(metric_names: list[str]) -> str:
    for preferred in PRIMARY_METRIC_PREFERENCES:
        if preferred in metric_names:
            return preferred
    counts = Counter(metric_names)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _derive_rl_tables(task_root: Path, task_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timeseries_path = task_root / "aggregated" / "run_timeseries.csv"
    rows = _read_csv_rows(timeseries_path)
    if not rows:
        return [], []

    latest_by_key: dict[tuple[str, str, str], _RlRunMetric] = {}
    metric_names: list[str] = []
    for row in rows:
        variant_id = str(row.get("algorithm") or row.get("variant_id") or "").strip()
        if not variant_id:
            continue
        replication = row.get("replication")
        run_id = f"rep_{replication}" if str(replication).strip() else str(row.get("run") or row.get("seed") or "").strip()
        if not run_id:
            continue
        metric_name = str(row.get("metric") or "").strip()
        metric_value = _safe_float(row.get("value"))
        step = _safe_int(row.get("step"))
        if not metric_name or metric_value is None or step is None:
            continue
        evaluation_id = str(row.get("evaluation_id") or f"{variant_id}__{run_id}").strip()
        key = (variant_id, run_id, metric_name)
        current = latest_by_key.get(key)
        if current is None or step >= current.step:
            latest_by_key[key] = _RlRunMetric(
                variant_id=variant_id,
                run_id=run_id,
                evaluation_id=evaluation_id,
                metric_name=metric_name,
                metric_value=metric_value,
                step=step,
            )
        metric_names.append(metric_name)

    if not latest_by_key:
        return [], []

    primary_metric = _infer_primary_metric(metric_names)
    metric_rows: list[dict[str, Any]] = []
    run_summary_rows: list[dict[str, Any]] = []

    spec_by_variant: dict[str, dict[str, Any]] = {}
    for variant_dir in sorted(path for path in task_root.iterdir() if path.is_dir() and path.name != "variants"):
        spec = _read_json(variant_dir / "variant_spec.json")
        if spec is not None:
            spec_by_variant[variant_dir.name] = spec

    grouped: dict[tuple[str, str], list[_RlRunMetric]] = defaultdict(list)
    for entry in latest_by_key.values():
        grouped[(entry.variant_id, entry.run_id)].append(entry)

    for (variant_id, run_id), metrics in sorted(grouped.items()):
        spec = spec_by_variant.get(variant_id, {})
        primary_row = next((row for row in metrics if row.metric_name == primary_metric), None)
        if primary_row is None:
            primary_row = sorted(metrics, key=lambda item: item.metric_name)[0]
        rep_dir = task_root / variant_id / run_id
        variant_dir = task_root / variant_id
        for metric in sorted(metrics, key=lambda item: item.metric_name):
            metric_rows.append(
                {
                    "run_id": run_id,
                    "evaluation_id": metric.evaluation_id,
                    "variant_id": variant_id,
                    "metric_name": metric.metric_name,
                    "metric_value": metric.metric_value,
                    "is_primary_metric": "True" if metric.metric_name == primary_row.metric_name else "False",
                }
            )
        run_summary_rows.append(
            {
                "run_id": run_id,
                "evaluation_id": primary_row.evaluation_id,
                "variant_id": variant_id,
                "variant_kind": str(spec.get("kind") or "unknown"),
                "base_program": str(spec.get("base_program") or "best"),
                "agent_name": "unknown",
                "correct": "True",
                "combined_score": primary_row.metric_value,
                "overall_pass": "True",
                "primary_metric_name": primary_row.metric_name,
                "primary_metric_value": primary_row.metric_value,
                "program_dir": str(variant_dir.resolve()),
                "results_dir": str(rep_dir.resolve()),
            }
        )

    return run_summary_rows, metric_rows


def _write_rl_variant_files(task_root: Path, task_name: str) -> list[dict[str, Any]]:
    variants_dir = task_root / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    aggregated_dir = task_root / "aggregated"
    root_timeseries = task_root.parent.parent / "aggregated" / task_name / "run_timeseries.csv"
    if root_timeseries.is_file():
        _copy_binary_file(root_timeseries, aggregated_dir / "run_timeseries.csv")
    run_summary_rows, metric_rows = _derive_rl_tables(task_root, task_name)
    _write_csv(aggregated_dir / "run_summary.csv", RUN_SUMMARY_FIELDNAMES, run_summary_rows)
    _write_csv(aggregated_dir / "metric_values.csv", METRIC_VALUES_FIELDNAMES, metric_rows)
    _task_pointer_readme(task_name=task_name, aggregated_dir=aggregated_dir, timeseries_name="run_timeseries.csv")

    variant_rows: list[dict[str, Any]] = []
    run_count_by_variant = Counter(str(row.get("variant_id")) for row in run_summary_rows)
    for variant_dir in sorted(path for path in task_root.iterdir() if path.is_dir() and path.name != "variants"):
        variant_spec = _read_json(variant_dir / "variant_spec.json")
        if variant_spec is None:
            continue
        variant_id = variant_dir.name
        _ensure_symlink(variants_dir / variant_id, variant_dir)
        run_manifest = {
            "variant_id": variant_id,
            "variant_kind": str(variant_spec.get("kind") or "unknown"),
            "base_program": str(variant_spec.get("base_program") or "best"),
            "title": str(variant_spec.get("title") or variant_id),
            "agent": "unknown",
            "agent_model": None,
            "variant_dir": str(variant_dir.resolve()),
            "task_bundle_dir": None,
            "expected_analysis_files": [
                "../aggregated/run_summary.csv",
                "../aggregated/metric_values.csv",
                "../aggregated/run_timeseries.csv",
            ],
            "status": "completed" if run_count_by_variant.get(variant_id, 0) > 0 else "failed",
            "created_at": None,
            "started_at": None,
            "completed_at": None,
            "eval_count": int(run_count_by_variant.get(variant_id, 0)),
        }
        rubric_contract = {
            "contract_version": 1,
            "variant_id": variant_id,
            "variant_kind": str(variant_spec.get("kind") or "unknown"),
            "rubrics": [
                {
                    "id": "run_timeseries_present",
                    "description": "A raw run_timeseries.csv exists for every replication.",
                    "type": "structural",
                },
                {
                    "id": "derived_final_step_summary_present",
                    "description": "Canonical run_summary.csv and metric_values.csv were derived using final-step-per-run reduction.",
                    "type": "structural",
                },
                {
                    "id": "paper_analysis_ready",
                    "description": "Task-scoped derived summaries and raw timeseries are both available for paper writing.",
                    "type": "semantic",
                },
            ],
        }
        _write_json(variant_dir / "run_manifest.json", run_manifest)
        _write_json(variant_dir / "rubric_contract.json", rubric_contract)
        variant_rows.append(
            _pipeline_variant_row(
                variant_id=variant_id,
                title=str(variant_spec.get("title") or variant_id),
                kind=str(variant_spec.get("kind") or "unknown"),
                status="completed" if run_count_by_variant.get(variant_id, 0) > 0 else "failed",
                eval_count=int(run_count_by_variant.get(variant_id, 0)),
                has_run_summary=run_count_by_variant.get(variant_id, 0) > 0,
                has_metric_values=bool(metric_rows),
                has_timeseries=(aggregated_dir / "run_timeseries.csv").is_file(),
            )
        )
    _write_pipeline_summary(task_root, variant_rows)
    return variant_rows


def _write_manifest(bundle_root: Path) -> Path:
    payload = {
        "generated_at": _utc_now(),
        "layout_version": 1,
        "primary_task": PRIMARY_TASK,
        "classification_precursor_task": CLASSIFICATION_EXTRA_TASK,
        "rl_summary_reduction": {
            "selector": "final_step_per_(variant_id,run_id,metric)",
            "aggregation": "paper tools may aggregate across runs; derived run_summary rows keep one final value per run",
        },
        "tasks": {
            PRIMARY_TASK: {
                "role": "primary_root_bundle",
                "task_dir": "task/",
                "aggregated_dir": "aggregated/",
                "timeseries_path": "aggregated/timeseries.csv",
            },
            CLASSIFICATION_EXTRA_TASK: {
                "role": "classification_precursor",
                "task_dir": "extra_tasks/ed_mnist/",
                "aggregated_dir": "extra_tasks/ed_mnist/aggregated/",
                "timeseries_path": "extra_tasks/ed_mnist/aggregated/timeseries.csv",
            },
            **{
                task_name: {
                    "role": "rl_supporting_task",
                    "task_dir": f"extra_tasks/{task_name}/",
                    "aggregated_dir": f"extra_tasks/{task_name}/aggregated/",
                    "timeseries_path": f"extra_tasks/{task_name}/aggregated/run_timeseries.csv",
                }
                for task_name in RL_TASKS
            },
        },
    }
    path = bundle_root / "paper_bundle_manifest.json"
    _write_json(path, payload)
    return path


def _write_report(bundle_root: Path, report_lines: list[str]) -> Path:
    path = bundle_root / "bundle_normalization_report.md"
    body = [
        "# Bundle normalization report",
        "",
        f"- generated_at: {_utc_now()}",
        f"- bundle_root: `{bundle_root}`",
        "- normalization_target: option 2A (root `ed_conv` canonical bundle + task-scoped extra bundles)",
        "",
        "## Findings and progress",
        "",
        *report_lines,
        "",
        "## Agent pointers",
        "",
        "- Start with `README.md`, `task/task.md`, `campaign.md`, and `paper_bundle_manifest.json`.",
        "- Use root `aggregated/` for the primary `ed_conv` ablation study.",
        "- Use `extra_tasks/<task>/aggregated/` for task-local canonical summaries.",
        "- For full curves, inspect `timeseries.csv` for classification tasks and `run_timeseries.csv` for RL tasks.",
    ]
    path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")
    return path


def normalize_alife_bundle(bundle_root: Path) -> NormalizationOutputs:
    root = bundle_root.expanduser().resolve()
    report_lines: list[str] = []
    _ensure_root_primary_bundle(root, report_lines)
    _ensure_ed_mnist_bundle(root, report_lines)

    extra_task_ready: dict[str, bool] = {CLASSIFICATION_EXTRA_TASK: True}
    for task_name in RL_TASKS:
        task_root = root / "extra_tasks" / task_name
        variant_rows = _write_rl_variant_files(task_root, task_name)
        report_lines.append(
            f"- `extra_tasks/{task_name}/` gained canonical `variants/`, `aggregated/run_summary.csv`, `aggregated/metric_values.csv`, and `aggregated/run_timeseries.csv` ({len(variant_rows)} variants)."
        )
        extra_task_ready[task_name] = bool(variant_rows)

    _write_root_readme_note(root)
    report_lines.append("- Root `README.md` now contains canonical paper-writing pointers and timeseries locations.")
    manifest_path = _write_manifest(root)
    report_lines.append("- Wrote `paper_bundle_manifest.json` with task roles and timeseries pointers.")
    report_path = _write_report(root, report_lines)
    return NormalizationOutputs(
        bundle_root=root,
        root_canonical_ready=True,
        extra_task_ready=extra_task_ready,
        report_path=report_path,
        manifest_path=manifest_path,
    )
