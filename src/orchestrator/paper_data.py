"""Paper data assembly helpers from run artifacts and memory."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.final_evaluation.runner import (
    FinalEvaluationError,
    resolve_frozen_incumbent,
    verify_final_manifest,
)
from src.orchestrator.paper_data_ablation import (
    build_ablation_paper_data,
    build_multitask_ablation_paper_data,
    collect_cross_task_results,
    has_ablation_bundle,
    has_multitask_ablation_bundle,
)
from src.investigation.files import evaluation_results_complete
from src.orchestrator.paper_evaluation_evidence import paper_evaluation_evidence
from src.orchestrator.paper_study_context import merge_study_context, merge_study_final_outcome
from src.orchestrator.paper_final_outcome import FinalOutcomePolicy, ground_final_outcome


class FinalEvaluationMode(StrEnum):
    """Control whether paper grounding may discover sealed final evidence."""

    DISCOVER = "discover"
    OMIT = "omit"


def _merge_cross_task_results(paper_data: dict[str, Any], results_dir: Path) -> None:
    """Merge aggregated/<task>/run_summary.csv stats into paper_data in-place."""
    cross = collect_cross_task_results(results_dir)
    task_names = cross.get("task_names") or []
    cross_results = cross.get("cross_task_results") or {}
    cross_sources = cross.get("grounding_sources") or {}
    if task_names:
        existing = paper_data.get("task_names")
        merged = list(existing) if isinstance(existing, list) else []
        for name in task_names:
            if name not in merged:
                merged.append(name)
        paper_data["task_names"] = merged
    if cross_results:
        paper_data["cross_task_results"] = cross_results
    if cross_sources:
        grounding = paper_data.get("grounding_sources")
        if not isinstance(grounding, dict):
            grounding = {}
            paper_data["grounding_sources"] = grounding
        for key, payload in cross_sources.items():
            grounding.setdefault(key, payload)


def _merge_final_evaluation(
    paper_data: dict[str, Any],
    final_evaluation_dir: Path,
    *,
    task_dir: Path | None,
    investigation_results_dir: Path | None,
    final_outcome_policy: FinalOutcomePolicy = FinalOutcomePolicy.QUALIFIED_ONLY,
) -> None:
    """Register completed sealed-final metrics as publication grounding."""
    final_evaluation_dir = final_evaluation_dir.expanduser()
    if final_evaluation_dir.is_symlink():
        raise ValueError(
            f"Final evaluation directory must not be a symlink: {final_evaluation_dir}"
        )
    final_evaluation_dir = final_evaluation_dir.resolve()
    manifest_path = final_evaluation_dir / "final_manifest.json"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return
    try:
        program_dir = None
        if investigation_results_dir is not None:
            program_dir, _ = resolve_frozen_incumbent(investigation_results_dir.resolve())
        elif isinstance(paper_data.get("downstream_study"), dict):
            program_dir = Path(paper_data["downstream_study"]["program_dir"])
        manifest = verify_final_manifest(
            final_evaluation_dir,
            task_dir=task_dir.resolve() if task_dir is not None else None,
            program_dir=program_dir,
        )
    except FinalEvaluationError as exc:
        raise ValueError(
            f"Invalid completed final evaluation manifest: {manifest_path}: {exc}"
        ) from exc
    if (manifest["correct"] is not True and "downstream_study" not in paper_data
            and final_outcome_policy is not FinalOutcomePolicy.ALLOW_NOT_QUALIFIED):
        raise ValueError(f"Final evaluation did not qualify for publication: {manifest_path}")
    merge_study_final_outcome(paper_data, manifest)
    paper_data["final_evaluation"] = {
        "status": manifest.get("status", "completed"),
        "correct": manifest["correct"],
        "outcome": manifest.get("outcome", "qualified"),
        "metrics": manifest["metrics"],
        "task_tree_sha256": manifest["task_tree_sha256"],
        "evaluator_sha256": manifest["evaluator_sha256"],
        "program_tree_sha256": manifest["program_tree_sha256"],
        "results_tree_sha256": manifest["results_tree_sha256"],
        "manifest_path": str(manifest_path),
        "evaluation_evidence": paper_evaluation_evidence(
            final_evaluation_dir, final_evaluation_dir / "results",
        ),
    }
    ground_final_outcome(paper_data)
    grounding = paper_data.get("grounding_sources")
    if not isinstance(grounding, dict):
        grounding = {}
        paper_data["grounding_sources"] = grounding
    evidence = paper_data["final_evaluation"]["evaluation_evidence"]
    if evidence["status"] == "available":
        source_id = "final.evidence.episodes"
        grounding[source_id] = {
            "id": source_id,
            "type": "evaluation_evidence",
            "evaluation_scope": "sealed_final",
            "path": evidence["path"],
            "sha256": evidence["sha256"],
            "json_pointer": "/episodes",
            "episode_count": evidence["episode_count"],
            "candidate_diagnostics_trust": evidence["candidate_diagnostics_trust"],
            "manifest_path": str(manifest_path),
            "program_tree_sha256": manifest["program_tree_sha256"],
            "results_tree_sha256": manifest["results_tree_sha256"],
        }
    normalized_metric_names: set[str] = set()
    for name, value in manifest["metrics"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        safe_name = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
        if not safe_name or safe_name in normalized_metric_names:
            raise ValueError("Final metric names collide after grounding-ID normalization")
        normalized_metric_names.add(safe_name)
        source_id = f"final.metric.{safe_name}"
        grounding[source_id] = {
            "id": source_id,
            "type": "metric",
            "value": value,
            "evaluation_scope": "sealed_final",
            "manifest_path": str(manifest_path),
            "results_tree_sha256": manifest["results_tree_sha256"],
        }


PROGRAM_DIR_RE = re.compile(r"^program_(\d+)$")
WORKER_DIR_RE = re.compile(r"^worker_\d+$")
VALID_PAPER_PROFILES = {"default", "standalone"}
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


def utc_now() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    """Handle safe float logic for this module."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_int(value: Any) -> int | None:
    """Handle safe int logic for this module."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON data from disk when available."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def discover_idea_titles(results_dir: Path) -> list[str]:
    """Discover idea titles."""
    ideas_root = results_dir / "ideas"
    if not ideas_root.exists():
        return []

    titles: list[str] = []
    for idea_dir in sorted(path for path in ideas_root.iterdir() if path.is_dir()):
        idea_bundle = read_json(idea_dir / "idea_bundle.json")
        if not isinstance(idea_bundle, dict):
            continue
        title = idea_bundle.get("title")
        if isinstance(title, str):
            clean = title.strip()
            if clean:
                titles.append(clean)
    return titles


def _reference_query_lines(section_text: str) -> list[str]:
    """Extract task-grounded query candidates from a goal or rubric section."""
    queries: list[str] = []
    for raw_line in section_text.splitlines():
        compact = " ".join(raw_line.split()).strip()
        if not compact or compact.startswith("#"):
            continue
        compact = LIST_PREFIX_RE.sub("", compact).strip()
        if compact:
            queries.append(compact)
    return queries


def build_reference_queries(
    *,
    task_goal: str,
    task_rubrics: str,
    idea_titles: list[str],
) -> list[str]:
    """Build reference queries."""
    seed_queries: list[str] = []
    seed_queries.extend(_reference_query_lines(task_goal))
    seed_queries.extend(_reference_query_lines(task_rubrics))
    seed_queries.extend(title.strip() for title in idea_titles if isinstance(title, str))

    seen: set[str] = set()
    queries: list[str] = []
    for query in seed_queries:
        compact = " ".join(query.split())
        if not compact:
            continue
        compact_l = compact.lower()
        if compact_l.startswith("(missing "):
            continue
        if "missing goal section" in compact_l or "missing rubrics section" in compact_l:
            continue
        if len(compact) > 200:
            compact = compact[:200].rstrip()
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(compact)
    return queries


def _discover_program_dirs(attempt_dir: Path) -> list[tuple[int, Path]]:
    """Discover program dirs."""
    out: list[tuple[int, Path]] = []
    for child in attempt_dir.iterdir():
        if not child.is_dir():
            continue
        match = PROGRAM_DIR_RE.match(child.name)
        if not match:
            continue
        out.append((int(match.group(1)), child))
    out.sort(key=lambda item: item[0])
    return out


def _discover_attempt_dirs(idea_dir: Path) -> list[Path]:
    """Discover legacy attempt dirs or synthesize a single root attempt."""
    attempt_dirs = sorted(
        path for path in idea_dir.iterdir() if path.is_dir() and path.name.endswith("_attempt")
    )
    if attempt_dirs:
        return attempt_dirs

    has_root_programs = any(_discover_program_dirs(idea_dir))
    has_root_status = (idea_dir / "rubric_status.json").is_file()
    if has_root_programs or has_root_status:
        return [idea_dir]
    return []


def _best_by_score(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Handle best by score logic for this module."""
    if not entries:
        return None
    return max(
        entries,
        key=lambda item: (
            float(item.get("combined_score", float("-inf"))),
            -int(item.get("eval_index", 10**9)),
        ),
    )


def _reindex_program_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return shallow-copied program rows with contiguous eval indices."""
    reindexed: list[dict[str, Any]] = []
    for eval_index, row in enumerate(rows, start=1):
        payload = dict(row)
        payload["eval_index"] = eval_index
        reindexed.append(payload)
    return reindexed


def _update_program_summary_fields(paper_data: dict[str, Any]) -> dict[str, Any]:
    """Recompute score-bearing paper fields from all_programs."""
    rows_raw = paper_data.get("all_programs")
    if not isinstance(rows_raw, list):
        return paper_data

    all_programs = [dict(row) for row in rows_raw if isinstance(row, dict)]
    all_programs.sort(key=lambda row: int(row.get("eval_index", 10**9)))
    all_programs = _reindex_program_rows(all_programs)
    correct_programs = [row for row in all_programs if row.get("correct") is True]
    top_programs = sorted(
        correct_programs,
        key=lambda row: (
            -float(row["combined_score"]),
            int(row["eval_index"]),
        ),
    )[:10]
    improvement_trace: list[dict[str, Any]] = []
    running_best = float("-inf")
    for row in correct_programs:
        score = float(row["combined_score"])
        if score <= running_best:
            continue
        running_best = score
        improvement_trace.append(
            {
                "id": f"result.improvement_{len(improvement_trace) + 1:03d}",
                "step_index": int(row["eval_index"]),
                "x_index": int(row["eval_index"]) - 1,
                "idea_id": row["idea_id"],
                "attempt_id": row["attempt_id"],
                "program_id": row["program_id"],
                "program_idx": int(row["program_idx"]),
                "combined_score": score,
                "program_dir": row["program_dir"],
            }
        )

    best_correct = _best_by_score(correct_programs)
    best_any = _best_by_score(all_programs)

    paper_data["all_programs"] = all_programs
    paper_data["best_correct"] = best_correct
    paper_data["best_any"] = best_any
    paper_data["top_programs"] = top_programs
    paper_data["improvement_trace"] = improvement_trace

    totals = paper_data.get("totals")
    if isinstance(totals, dict):
        totals["evaluated_programs"] = len(all_programs)
        totals["correct_programs"] = len(correct_programs)

    grounding_sources = paper_data.get("grounding_sources")
    if not isinstance(grounding_sources, dict):
        return paper_data

    for source_id in list(grounding_sources):
        if (
            source_id
            in {
                "result.best_correct_score",
                "result.best_any_score",
                "result.total_evaluated_programs",
                "result.total_correct_programs",
            }
            or source_id.startswith("result.top_program_")
            or source_id.startswith("result.improvement_")
        ):
            del grounding_sources[source_id]

    def register_source(source_id: str, payload: dict[str, Any]) -> None:
        """Register source."""
        grounding_sources[source_id] = {
            "id": source_id,
            **payload,
        }

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

    register_source(
        "result.total_evaluated_programs", {"type": "count", "value": len(all_programs)}
    )
    register_source(
        "result.total_correct_programs", {"type": "count", "value": len(correct_programs)}
    )

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

    for row in improvement_trace:
        register_source(
            str(row["id"]),
            {
                "type": "metric",
                "value": row["combined_score"],
                "program_dir": row.get("program_dir"),
                "idea_id": row.get("idea_id"),
                "attempt_id": row.get("attempt_id"),
                "program_id": row.get("program_id"),
            },
        )

    return paper_data


def _iter_idea_dirs(results_dir: Path) -> list[Path]:
    """Handle iter idea dirs logic for this module."""
    ideas_root = results_dir / "ideas"
    if not ideas_root.is_dir():
        return []
    return sorted(path for path in ideas_root.iterdir() if path.is_dir())


def has_evaluated_programs(results_dir: Path) -> bool:
    """Check whether evaluated programs."""
    resolved = results_dir.expanduser().resolve()
    if has_multitask_ablation_bundle(resolved) or has_ablation_bundle(resolved):
        return True
    for idea_dir in _iter_idea_dirs(resolved):
        for attempt_dir in _discover_attempt_dirs(idea_dir):
            for _, program_dir in _discover_program_dirs(attempt_dir):
                if not evaluation_results_complete(program_dir):
                    continue
                metrics = read_json(program_dir / "metrics.json")
                if not isinstance(metrics, dict):
                    continue
                if _safe_float(metrics.get("combined_score")) is not None:
                    return True
    return False


def _worker_results_dirs(results_dir: Path) -> list[Path]:
    """Handle worker results dirs logic for this module."""
    if not results_dir.exists():
        return []
    workers: list[Path] = []
    for child in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        if not WORKER_DIR_RE.match(child.name):
            continue
        if (child / "ideas").is_dir():
            workers.append(child)
    return workers


def _find_single_nested_ablation_bundle(results_dir: Path) -> Path | None:
    """Return the single direct-child ablation bundle when the layout is unambiguous."""
    if not results_dir.exists():
        return None
    matches = [
        child.resolve()
        for child in sorted(path for path in results_dir.iterdir() if path.is_dir())
        if has_ablation_bundle(child)
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _merge_nested_ablation_payload(
    paper_data: dict[str, Any],
    *,
    nested_bundle_dir: Path,
    paper_profile: str,
) -> dict[str, Any]:
    """Augment normal paper data with ablation-only fields from a nested bundle."""
    if has_multitask_ablation_bundle(nested_bundle_dir):
        ablation_data = build_multitask_ablation_paper_data(
            nested_bundle_dir, paper_profile=paper_profile
        )
    else:
        ablation_data = build_ablation_paper_data(nested_bundle_dir, paper_profile=paper_profile)

    ablation_studies = ablation_data.get("ablation_studies")
    if isinstance(ablation_studies, list) and ablation_studies:
        paper_data["ablation_studies"] = ablation_studies

    idea_titles = paper_data.get("idea_titles")
    ablation_titles = ablation_data.get("idea_titles")
    if isinstance(idea_titles, list) and isinstance(ablation_titles, list):
        paper_data["idea_titles"] = list(
            dict.fromkeys(
                str(title).strip()
                for title in [*idea_titles, *ablation_titles]
                if isinstance(title, str) and str(title).strip()
            )
        )

    totals = paper_data.get("totals")
    ablation_totals = ablation_data.get("totals")
    if isinstance(totals, dict) and isinstance(ablation_totals, dict):
        for key in (
            "ablation_studies",
            "ablation_variants",
            "ablation_runs",
            "completed_ablation_variants",
            "failed_ablation_variants",
            "timed_out_ablation_variants",
        ):
            value = ablation_totals.get(key)
            if isinstance(value, int):
                totals[key] = value

    grounding_sources = paper_data.get("grounding_sources")
    ablation_sources = ablation_data.get("grounding_sources")
    if isinstance(grounding_sources, dict) and isinstance(ablation_sources, dict):
        for source_id, payload in ablation_sources.items():
            source_key = str(source_id)
            if source_key.startswith("ablation.") or source_key in {
                "result.total_ablation_studies",
                "result.total_ablation_variants",
                "result.total_ablation_runs",
            }:
                grounding_sources[source_key] = payload

    base_rows = paper_data.get("all_programs")
    ablation_rows = ablation_data.get("all_programs")
    if isinstance(base_rows, list) and isinstance(ablation_rows, list):
        merged_rows = [dict(row) for row in base_rows if isinstance(row, dict)]
        merged_rows.extend(dict(row) for row in ablation_rows if isinstance(row, dict))
        paper_data["all_programs"] = _reindex_program_rows(merged_rows)
        _update_program_summary_fields(paper_data)

    return paper_data


def _link_or_copy_dir(src: Path, dst: Path) -> str:
    """Materialize a directory via symlink when possible, otherwise copy it."""
    try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
        return "symlinked"
    except OSError:
        shutil.copytree(src, dst)
        return "copied"


def _ensure_default_global_memory(results_dir: Path) -> None:
    """Ensure default global memory."""
    memory_md = results_dir / "memory.md"
    memory_json = results_dir / "memory.jsonl"
    if not memory_md.exists():
        memory_md.write_text("# Global Memory Log\n\n", encoding="utf-8")
    if not memory_json.exists():
        memory_json.write_text("", encoding="utf-8")


def prepare_results_dir_for_paper(
    results_dir: Path,
    *,
    merge_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Prepare results dir for paper."""
    resolved = results_dir.expanduser().resolve()
    if has_evaluated_programs(resolved):
        return resolved, {
            "mode": "direct",
            "source_results_dir": str(resolved),
            "paper_results_dir": str(resolved),
        }

    worker_dirs = [path for path in _worker_results_dirs(resolved) if has_evaluated_programs(path)]
    if not worker_dirs:
        return resolved, {
            "mode": "direct",
            "source_results_dir": str(resolved),
            "paper_results_dir": str(resolved),
        }

    if merge_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        merge_root = resolved / "paper" / f"_merged_parallel_{stamp}"
    merged_root = merge_root.expanduser().resolve()
    if merged_root.exists():
        raise FileExistsError(f"Merged paper results directory already exists: {merged_root}")

    merged_ideas_root = merged_root / "ideas"
    merged_ideas_root.mkdir(parents=True, exist_ok=False)

    copied_global_memory = False
    for rel_path in ("memory.md", "memory.jsonl"):
        src = resolved / rel_path
        dst = merged_root / rel_path
        if src.is_file():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            copied_global_memory = True
    _ensure_default_global_memory(merged_root)
    (merged_root / "ideas_history.jsonl").write_text("", encoding="utf-8")
    (merged_root / "attempts_history.jsonl").write_text("", encoding="utf-8")

    used_names: set[str] = set()
    linked_idea_dirs = 0
    copied_idea_dirs = 0
    for worker_dir in worker_dirs:
        for idea_dir in _iter_idea_dirs(worker_dir):
            base_name = f"{worker_dir.name}__{idea_dir.name}"
            dst_name = base_name
            suffix = 1
            while dst_name in used_names:
                suffix += 1
                dst_name = f"{base_name}_{suffix:02d}"
            used_names.add(dst_name)
            dst = merged_ideas_root / dst_name
            mode = _link_or_copy_dir(idea_dir, dst)
            if mode == "symlinked":
                linked_idea_dirs += 1
            else:
                copied_idea_dirs += 1

    nested_ablation_bundle = _find_single_nested_ablation_bundle(resolved)
    nested_ablation_bundle_mode = "absent"
    if nested_ablation_bundle is not None:
        nested_bundle_dst = merged_root / nested_ablation_bundle.name
        nested_ablation_bundle_mode = _link_or_copy_dir(nested_ablation_bundle, nested_bundle_dst)

    campaign_asset_modes: dict[str, str] = {}
    for rel_path in ("aggregated", "extra_tasks"):
        src = resolved / rel_path
        if src.is_dir():
            campaign_asset_modes[rel_path] = _link_or_copy_dir(src, merged_root / rel_path)

    return merged_root, {
        "mode": "parallel_merged",
        "source_results_dir": str(resolved),
        "paper_results_dir": str(merged_root),
        "worker_count": len(worker_dirs),
        "linked_idea_dirs": linked_idea_dirs,
        "copied_idea_dirs": copied_idea_dirs,
        "copied_global_memory": copied_global_memory,
        "nested_ablation_bundle_dir": (
            str(nested_ablation_bundle) if nested_ablation_bundle is not None else None
        ),
        "nested_ablation_bundle_mode": nested_ablation_bundle_mode,
        "campaign_asset_modes": campaign_asset_modes,
    }


def _presentation_contract(paper_profile: str) -> dict[str, Any]:
    """Return profile-specific writing guidance embedded in paper_data.json."""
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


def build_paper_data(
    results_dir: Path,
    *,
    paper_profile: str = "default",
    ablations_dir: Path | None = None,
    final_evaluation_dir: Path | None = None,
    task_dir: Path | None = None,
    investigation_results_dir: Path | None = None,
    final_evaluation_mode: FinalEvaluationMode = FinalEvaluationMode.DISCOVER,
    final_outcome_policy: FinalOutcomePolicy = FinalOutcomePolicy.QUALIFIED_ONLY,
) -> dict[str, Any]:
    """Build paper data."""
    final_outcome_policy = FinalOutcomePolicy(final_outcome_policy)
    if paper_profile not in VALID_PAPER_PROFILES:
        raise ValueError(
            f"paper_profile must be one of {sorted(VALID_PAPER_PROFILES)}, got {paper_profile!r}"
        )
    resolved = results_dir.expanduser().resolve()
    resolved_final_dir: Path | None = None
    if final_evaluation_mode is FinalEvaluationMode.DISCOVER:
        resolved_final_dir = (
            final_evaluation_dir.expanduser()
            if final_evaluation_dir is not None
            else resolved / "final_evaluation"
        )
    if has_multitask_ablation_bundle(resolved):
        paper_data = build_multitask_ablation_paper_data(resolved, paper_profile=paper_profile)
        merge_study_context(paper_data, investigation_results_dir or resolved)
        if resolved_final_dir is not None:
            _merge_final_evaluation(
                paper_data,
                resolved_final_dir,
                task_dir=task_dir,
                investigation_results_dir=investigation_results_dir,
                final_outcome_policy=final_outcome_policy,
            )
        return paper_data
    if has_ablation_bundle(resolved):
        ablation_data = build_ablation_paper_data(resolved, paper_profile=paper_profile)
        _merge_cross_task_results(ablation_data, resolved)
        merge_study_context(ablation_data, investigation_results_dir or resolved)
        if resolved_final_dir is not None:
            _merge_final_evaluation(
                ablation_data,
                resolved_final_dir,
                task_dir=task_dir,
                investigation_results_dir=investigation_results_dir,
                final_outcome_policy=final_outcome_policy,
            )
        return ablation_data
    ideas_root = resolved / "ideas"

    all_programs: list[dict[str, Any]] = []
    idea_count = 0
    attempt_count = 0
    eval_index = 0

    for idea_dir in (
        sorted(path for path in ideas_root.iterdir() if path.is_dir())
        if ideas_root.exists()
        else []
    ):
        idea_count += 1
        idea_bundle = read_json(idea_dir / "idea_bundle.json")
        idea_id_default = idea_dir.name
        if isinstance(idea_bundle, dict):
            idea_id_raw = idea_bundle.get("idea_id")
            if isinstance(idea_id_raw, str) and idea_id_raw.strip():
                idea_id_default = idea_id_raw.strip()

        for attempt_dir in _discover_attempt_dirs(idea_dir):
            attempt_count += 1

            root_status = read_json(attempt_dir / "rubric_status.json")
            attempt_id_default = (
                attempt_dir.name if attempt_dir != idea_dir else f"{idea_id_default}_attempt_01"
            )
            if isinstance(root_status, dict):
                maybe_attempt = root_status.get("attempt_id")
                if isinstance(maybe_attempt, str) and maybe_attempt.strip():
                    attempt_id_default = maybe_attempt.strip()

            for program_idx, program_dir in _discover_program_dirs(attempt_dir):
                if not evaluation_results_complete(program_dir):
                    continue
                metrics = read_json(program_dir / "metrics.json")
                if not isinstance(metrics, dict):
                    continue
                score = _safe_float(metrics.get("combined_score"))
                if score is None:
                    continue

                correctness = read_json(program_dir / "correct.json")
                correct = correctness.get("correct") if isinstance(correctness, dict) else None
                eval_index += 1
                all_programs.append(
                    {
                        "idea_id": idea_id_default,
                        "idea_dir_name": idea_dir.name,
                        "attempt_id": attempt_id_default,
                        "attempt_dir_name": (
                            attempt_dir.name if attempt_dir != idea_dir else "idea_root"
                        ),
                        "program_id": program_dir.name,
                        "program_idx": program_idx,
                        "combined_score": score,
                        "correct": correct,
                        "eligible": (correct is True),
                        "program_dir": str(program_dir),
                        "evaluation_evidence": (
                            {"status": "unavailable", "reason": "linked_idea_directory"}
                            if idea_dir.is_symlink()
                            else paper_evaluation_evidence(resolved, program_dir)
                        ),
                        "eval_index": eval_index,
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

    improvement_trace: list[dict[str, Any]] = []
    running_best = float("-inf")
    for row in correct_programs:
        score = float(row["combined_score"])
        if score <= running_best:
            continue
        running_best = score
        improvement_trace.append(
            {
                "id": f"result.improvement_{len(improvement_trace) + 1:03d}",
                "step_index": int(row["eval_index"]),
                "x_index": int(row["eval_index"]) - 1,
                "idea_id": row["idea_id"],
                "attempt_id": row["attempt_id"],
                "program_id": row["program_id"],
                "program_idx": int(row["program_idx"]),
                "combined_score": score,
                "program_dir": row["program_dir"],
            }
        )

    best_correct = _best_by_score(correct_programs)
    best_any = _best_by_score(all_programs)

    grounding_sources: dict[str, dict[str, Any]] = {}

    def register_source(source_id: str, payload: dict[str, Any]) -> None:
        """Register source."""
        grounding_sources[source_id] = {
            "id": source_id,
            **payload,
        }

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

    register_source("result.total_ideas", {"type": "count", "value": idea_count})
    register_source("result.total_attempts", {"type": "count", "value": attempt_count})
    register_source(
        "result.total_evaluated_programs", {"type": "count", "value": len(all_programs)}
    )
    register_source(
        "result.total_correct_programs", {"type": "count", "value": len(correct_programs)}
    )

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

    for row in improvement_trace:
        register_source(
            str(row["id"]),
            {
                "type": "metric",
                "value": row["combined_score"],
                "program_dir": row.get("program_dir"),
                "idea_id": row.get("idea_id"),
                "attempt_id": row.get("attempt_id"),
                "program_id": row.get("program_id"),
            },
        )

    paper_data = {
        "generated_at": utc_now(),
        "paper_profile": paper_profile,
        "presentation_contract": _presentation_contract(paper_profile),
        "results_dir": str(resolved),
        "best_correct": best_correct,
        "best_any": best_any,
        "totals": {
            "ideas": idea_count,
            "attempts": attempt_count,
            "evaluated_programs": len(all_programs),
            "correct_programs": len(correct_programs),
        },
        "top_programs": top_programs,
        "improvement_trace": improvement_trace,
        "all_programs": all_programs,
        "idea_titles": discover_idea_titles(resolved),
        "grounding_sources": grounding_sources,
        "ablation_studies": [],
    }

    nested_bundle_dir = _find_single_nested_ablation_bundle(resolved)
    if nested_bundle_dir is not None:
        paper_data = _merge_nested_ablation_payload(
            paper_data,
            nested_bundle_dir=nested_bundle_dir,
            paper_profile=paper_profile,
        )

    if ablations_dir is not None:
        resolved_ablations = ablations_dir.expanduser()
        evidence_sources = paper_data.setdefault("evidence_sources", {})
        if resolved_ablations.is_symlink():
            evidence_sources["ablations"] = {
                "status": "degraded",
                "path": str(resolved_ablations),
                "reason": "symlinked directory rejected",
            }
        elif not resolved_ablations.is_dir():
            evidence_sources["ablations"] = {
                "status": "missing",
                "path": str(resolved_ablations),
            }
        elif has_ablation_bundle(resolved_ablations) or has_multitask_ablation_bundle(
            resolved_ablations
        ):
            paper_data = _merge_nested_ablation_payload(
                paper_data,
                nested_bundle_dir=resolved_ablations.resolve(),
                paper_profile=paper_profile,
            )
            evidence_sources["ablations"] = {
                "status": "included",
                "path": str(resolved_ablations.resolve()),
            }
        else:
            evidence_sources["ablations"] = {
                "status": "degraded",
                "path": str(resolved_ablations.resolve()),
                "reason": "no validated ablation bundle",
            }

    paper_data.setdefault("evidence_sources", {})["investigation"] = {
        "status": "included",
        "path": str(resolved),
    }

    _merge_cross_task_results(paper_data, resolved)
    merge_study_context(paper_data, investigation_results_dir or resolved)
    if resolved_final_dir is not None:
        _merge_final_evaluation(
            paper_data,
            resolved_final_dir,
            task_dir=task_dir,
            investigation_results_dir=investigation_results_dir,
            final_outcome_policy=final_outcome_policy,
        )
    return paper_data


def write_paper_data(paper_data: dict[str, Any], path: Path) -> None:
    """Write paper data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(paper_data, indent=2, sort_keys=True), encoding="utf-8")


def generate_results_plot(results_dir: Path, output_path: Path) -> Path:
    """Generate results plot."""
    paper_data = build_paper_data(results_dir)
    points = paper_data.get("all_programs")
    if not isinstance(points, list) or not points:
        raise ValueError("No evaluated program scores found to plot for paper writing.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values: list[int] = []
    y_values: list[float] = []
    running_best_values: list[float] = []
    current_best = float("-inf")

    for row in points:
        if not isinstance(row, dict):
            continue
        score = _safe_float(row.get("combined_score"))
        x_index = _safe_int(row.get("eval_index"))
        if score is None or x_index is None:
            continue
        x_values.append(x_index)
        y_values.append(score)
        if row.get("correct") is True:
            current_best = max(current_best, score)
        running_best_values.append(current_best if current_best > float("-inf") else score)

    if not x_values:
        raise ValueError("No evaluated program scores found to plot for paper writing.")

    fig_width = min(16.0, max(8.0, len(x_values) * 0.22))
    fig, ax = plt.subplots(figsize=(fig_width, 5.6), constrained_layout=True)
    ax.plot(x_values, y_values, marker="o", linewidth=1.2, markersize=3.8, label="combined_score")
    ax.plot(
        x_values,
        running_best_values,
        linestyle="--",
        linewidth=1.4,
        label="running_best_correct",
    )

    ax.set_xlabel("evaluation index (within this run)", fontsize=14)
    ax.set_ylabel("combined_score", fontsize=14)
    ax.set_title(f"Within-Run Score Progression ({results_dir.name})", fontsize=16)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=11)

    out = output_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
