"""paper_data: result layouts, ablation bundles, cross-task stats, posthoc analyses, ALIFE normalisation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.orchestrator import paper_data
from src.orchestrator.alife_bundle_normalize import normalize_alife_bundle
from src.orchestrator.paper_data_ablation import collect_cross_task_results, has_ablation_bundle
from src.orchestrator.paper_data_posthoc import (
    _CONTROL_JSON_NAMES,
    _collect_data_files,
    _resolve_attempt_dir,
    augment_paper_data_with_posthoc,
    collect_posthoc_analyses,
    copy_posthoc_figures,
    discover_posthoc_session,
    validate_posthoc_ranking,
)
from tests.program_testutils import make_program_dir

RUN_SUMMARY_HEADER = (
    "variant_id,variant_kind,combined_score,primary_metric_name,primary_metric_value,correct\n"
)
METRIC_VALUES_HEADER = "evaluation_id,variant_id,metric_name,metric_value,is_primary_metric\n"
RL_TIMESERIES_HEADER = (
    "evaluation_id,variant_id,algorithm,replication,step,experiment,run,seed,metric,value"
)
IDEA_BUNDLE = {
    "title": "Demo Idea",
    "target_rubrics": [{"id": "r", "type": "metric", "description": "d"}],
    "hypothesis": "h",
    "implementation_plan": ["p"],
    "notes_for_agent": "",
}
ABLATION_VARIANTS = (
    ("control_best", "Control (best program)", "control", (0.55, 0.57)),
    ("ablation_no_deep_aux_mlp", "Single-layer linear auxiliary classifiers", "ablation", (0.52, 0.54)),
)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_program_result(
    root: Path,
    idea_dir_name: str,
    *,
    attempt_dir_name: str | None = "01_attempt",
    program_dir_name: str = "program_0",
    score: float = 1.0,
    correct: bool = True,
    idea_id: str = "i1",
    attempt_id: str = "a1",
) -> None:
    idea_dir = root / "ideas" / idea_dir_name
    attempt = idea_dir / attempt_dir_name if attempt_dir_name else idea_dir
    program_dir = attempt / program_dir_name
    program_dir.mkdir(parents=True, exist_ok=True)
    _write_json(program_dir / "metrics.json", {"combined_score": score})
    _write_json(program_dir / "correct.json", {"correct": correct})
    make_program_dir(program_dir, content="print('ok')\n")
    _write_json(
        attempt / "rubric_status.json",
        {"attempt_id": attempt_id, "idea_id": idea_id, "overall_pass": True, "rubrics": []},
    )
    _write_json(idea_dir / "idea_bundle.json", {**IDEA_BUNDLE, "idea_id": idea_id})


def _write_ablation_bundle(root: Path) -> None:
    run_summary = [
        "run_id,evaluation_id,variant_id,variant_kind,base_program,agent_name,correct,"
        "combined_score,overall_pass,primary_metric_name,primary_metric_value,program_dir,results_dir"
    ]
    metric_values = ["run_id,evaluation_id,variant_id,metric_name,metric_value,is_primary_metric"]
    timeseries = ["run_id,evaluation_id,variant_id,series_name,step,value"]
    summary_variants = []
    for variant_id, title, kind, runs in ABLATION_VARIANTS:
        summary_variants.append(
            {
                "variant_id": variant_id, "kind": kind, "title": title, "status": "completed",
                "rubric_passed": True, "eval_count": 1, "exit_code": 0,
                "analysis_files": {
                    "run_summary.csv": True, "metric_values.csv": True, "timeseries.csv": True,
                },
            }
        )
        variant_dir = root / "variants" / variant_id
        _write_json(
            variant_dir / "variant_spec.json",
            {"variant_id": variant_id, "title": title, "kind": kind,
             "hypothesis": "demo hypothesis", "implementation_plan": ["demo change"]},
        )
        _write_json(
            variant_dir / "run_manifest.json",
            {"variant_id": variant_id, "variant_kind": kind, "title": title,
             "status": "completed", "eval_count": 1},
        )
        _write_json(variant_dir / "rubric_contract.json", {"variant_id": variant_id, "rubrics": []})
        _write_json(
            variant_dir / "target_rubrics.json",
            [{"id": "analysis_files_present", "description": "demo"}],
        )
        bundle = variant_dir / "task_bundle" / "program_1"
        _write_text(bundle / "src" / "program.py", "print('ok')\n")
        for rep, accuracy in enumerate(runs):
            rep_dir = bundle / f"rep_{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"rep_{rep},{variant_id}_eval_1,{variant_id}"
            run_summary.append(
                f"{prefix},{kind},best,unknown,True,{accuracy},True,test_accuracy,{accuracy},"
                f"{bundle / 'src'},{rep_dir}"
            )
            metric_values.append(f"{prefix},test_accuracy,{accuracy},True")
            metric_values.append(f"{prefix},training_time,{100.0 + rep},False")
            timeseries.append(f"{prefix},lr,1,0.001")
    _write_json(
        root / "pipeline_summary.json",
        {"total_variants": 2, "total_completed": 2, "total_failed": 0, "total_timeout": 0,
         "rubric_passed": 2, "variants": summary_variants},
    )
    for name, lines in (
        ("run_summary.csv", run_summary),
        ("metric_values.csv", metric_values),
        ("timeseries.csv", timeseries),
    ):
        _write_text(root / "aggregated" / name, "\n".join(lines) + "\n")


def _write_multitask_ablation_bundle(root: Path) -> None:
    _write_ablation_bundle(root)
    _write_ablation_bundle(root / "extra_tasks" / "ed_mnist")
    _write_json(
        root / "paper_bundle_manifest.json",
        {
            "primary_task": "ed_conv",
            "tasks": {
                "ed_conv": {
                    "role": "primary_root_bundle", "task_dir": "task/",
                    "aggregated_dir": "aggregated/",
                    "timeseries_path": "aggregated/timeseries.csv",
                },
                "ed_mnist": {
                    "role": "classification_precursor", "task_dir": "extra_tasks/ed_mnist/",
                    "aggregated_dir": "extra_tasks/ed_mnist/aggregated/",
                    "timeseries_path": "extra_tasks/ed_mnist/aggregated/timeseries.csv",
                },
            },
        },
    )


def _write_variant_spec(root: Path, task: str, variant_id: str, **spec: str) -> None:
    _write_json(
        root / "extra_tasks" / task / variant_id / "variant_spec.json",
        {"variant_id": variant_id, **spec},
    )


def _write_final_manifest_stub(root: Path, name: str = "final_evaluation") -> Path:
    (root / "ideas").mkdir(parents=True, exist_ok=True)
    final_dir = root / name
    _write_text(final_dir / "final_manifest.json", "{}\n")
    return final_dir


# ── build_paper_data: layouts and ablation bundles ──


def test_build_paper_data_collects_totals_sources_and_profile_contracts(tmp_path: Path) -> None:
    # collects totals and sources
    _write_program_result(tmp_path, "01_demo", score=2.5)
    data = paper_data.build_paper_data(tmp_path)
    assert data["totals"]["ideas"] == 1
    assert data["totals"]["attempts"] == 1
    assert data["totals"]["evaluated_programs"] == 1
    assert "result.total_ideas" in data["grounding_sources"]
    assert "result.best_correct_score" in data["grounding_sources"]
    assert data["idea_titles"] == ["Demo Idea"]

    # standalone profile includes presentation contract
    data = paper_data.build_paper_data(tmp_path, paper_profile="standalone")
    assert data["paper_profile"] == "standalone"
    contract = data["presentation_contract"]
    assert contract["variant_unit"] == "model_variant"
    assert contract["run_unit"] == "seeded_run"
    assert "eligible" in contract["forbidden_visible_terms"]
    assert "correctness rate" in contract["forbidden_visible_terms"]
    assert contract["preferred_terms"]["program"] == "seeded run"
    assert "reader-facing meaning per figure" in contract["threshold_rate_rule"]

    # rejects unknown profile
    with pytest.raises(ValueError, match="paper_profile must be one of"):
        paper_data.build_paper_data(tmp_path, paper_profile="bogus")


def test_build_paper_data_supports_root_level_idea_layout(tmp_path: Path) -> None:
    _write_program_result(
        tmp_path, "01_demo", attempt_dir_name=None, program_dir_name="program_1", score=2.75,
        idea_id="idea_0001", attempt_id="idea_0001_attempt_01",
    )
    data = paper_data.build_paper_data(tmp_path)
    assert data["totals"]["ideas"] == 1
    assert data["totals"]["attempts"] == 1
    assert data["totals"]["evaluated_programs"] == 1
    assert data["all_programs"][0]["attempt_id"] == "idea_0001_attempt_01"
    assert data["all_programs"][0]["attempt_dir_name"] == "idea_root"


def test_build_paper_data_supports_ablation_bundle_layout(tmp_path: Path) -> None:
    _write_ablation_bundle(tmp_path)
    data = paper_data.build_paper_data(tmp_path, paper_profile="standalone")
    assert data["paper_profile"] == "standalone"
    assert data["totals"]["ideas"] == 2
    assert data["totals"]["attempts"] == 2
    assert data["totals"]["evaluated_programs"] == 4
    assert data["totals"]["ablation_variants"] == 2
    assert data["totals"]["ablation_runs"] == 4
    assert len(data["all_programs"]) == 4
    assert len(data["ablation_studies"]) == 1
    assert paper_data.has_evaluated_programs(tmp_path)
    assert "Control (best program)" in data["idea_titles"]
    assert "Single-layer linear auxiliary classifiers" in data["idea_titles"]
    assert "result.total_ablation_variants" in data["grounding_sources"]
    assert "result.total_ablation_runs" in data["grounding_sources"]
    study = data["ablation_studies"][0]
    assert study["anchor_variant_id"] == "control_best"
    assert study["variant_count"] == 2
    assert study["run_count"] == 4
    assert len(study["comparisons"]) == 1
    ablation_variant = next(
        item for item in study["variants"] if item["variant_id"] == "ablation_no_deep_aux_mlp"
    )
    assert ablation_variant["primary_metric_summary"]["mean"] == pytest.approx(0.53)
    study_id = study["study_id"]
    assert (
        f"ablation.study_{study_id}.variant_control_best.test_accuracy.mean"
        in data["grounding_sources"]
    )
    assert (
        f"ablation.study_{study_id}.compare_control_best_vs_ablation_no_deep_aux_mlp"
        ".test_accuracy.mean_delta"
    ) in data["grounding_sources"]


def test_build_paper_data_merges_nested_or_sibling_ablation_bundle_and_skips_ambiguous_ones(
    tmp_path: Path,
) -> None:
    # augments mixed layout with nested ablation bundle
    root = tmp_path / "nested"
    _write_program_result(root, "01_demo", score=0.2)
    _write_ablation_bundle(root / "variants_with_reps")
    data = paper_data.build_paper_data(root, paper_profile="standalone")
    assert data["totals"]["ideas"] == 1
    assert data["totals"]["attempts"] == 1
    assert data["totals"]["evaluated_programs"] == 5
    assert data["totals"]["correct_programs"] == 5
    assert data["totals"]["ablation_studies"] == 1
    assert data["totals"]["ablation_variants"] == 2
    assert data["totals"]["ablation_runs"] == 4
    assert len(data["all_programs"]) == 5
    assert data["best_any"]["combined_score"] == 0.57
    assert "Demo Idea" in data["idea_titles"]
    assert "Control (best program)" in data["idea_titles"]
    assert len(data["ablation_studies"]) == 1
    assert data["ablation_studies"][0]["source_dir"] == str((root / "variants_with_reps").resolve())
    assert data["top_programs"][0]["combined_score"] == 0.57
    assert data["grounding_sources"]["result.best_any_score"]["value"] == 0.57
    assert "result.total_ablation_variants" in data["grounding_sources"]
    assert "result.total_ablation_runs" in data["grounding_sources"]
    assert any(source_id.startswith("ablation.study_") for source_id in data["grounding_sources"])

    # merges explicit sibling ablation bundle
    sibling = tmp_path / "sibling"
    investigation = sibling / "investigation"
    _write_program_result(investigation, "01_demo", score=0.2)
    _write_ablation_bundle(sibling / "ablations")
    data = paper_data.build_paper_data(
        investigation, paper_profile="standalone", ablations_dir=sibling / "ablations"
    )
    assert data["totals"]["evaluated_programs"] == 5
    assert data["totals"]["ablation_studies"] == 1
    assert len(data["ablation_studies"]) == 1

    # skips ambiguous nested ablation bundles
    root = tmp_path / "ambiguous"
    _write_program_result(root, "01_demo", score=0.2)
    _write_ablation_bundle(root / "variants_with_reps")
    _write_ablation_bundle(root / "other_variants_with_reps")
    data = paper_data.build_paper_data(root, paper_profile="standalone")
    assert data["totals"]["ideas"] == 1
    assert data["totals"]["attempts"] == 1
    assert data["totals"]["evaluated_programs"] == 1
    assert data["totals"]["correct_programs"] == 1
    assert "ablation_variants" not in data["totals"]
    assert "ablation_runs" not in data["totals"]
    assert len(data["all_programs"]) == 1
    assert data["ablation_studies"] == []
    assert "result.total_ablation_variants" not in data["grounding_sources"]
    assert "result.total_ablation_runs" not in data["grounding_sources"]


def test_build_paper_data_supports_multitask_ablation_manifest(tmp_path: Path) -> None:
    _write_multitask_ablation_bundle(tmp_path)
    data = paper_data.build_paper_data(tmp_path, paper_profile="standalone")
    assert data["paper_profile"] == "standalone"
    assert data["totals"]["ablation_studies"] == 2
    assert data["totals"]["ablation_variants"] == 4
    assert data["totals"]["ablation_runs"] == 8
    assert data["totals"]["evaluated_programs"] == 8
    assert len(data["ablation_studies"]) == 2
    assert {study["study_id"] for study in data["ablation_studies"]} == {"ed_conv", "ed_mnist"}
    assert data["best_any"] is None
    assert data["best_correct"] is None
    assert data["top_programs"] == []
    assert data["improvement_trace"] == []
    assert paper_data.has_evaluated_programs(tmp_path)
    assert "result.total_ablation_studies" in data["grounding_sources"]
    for study_id in ("ed_conv", "ed_mnist"):
        assert (
            f"ablation.study_{study_id}.variant_control_best.test_accuracy.mean"
            in data["grounding_sources"]
        )
    assert {row["study_id"] for row in data["all_programs"]} == {"ed_conv", "ed_mnist"}
    assert any(row["idea_id"].startswith("ed_mnist:") for row in data["all_programs"])


def test_manifest_without_bundles_and_incomplete_evaluation_manifest_are_not_evaluated(
    tmp_path: Path,
) -> None:
    # manifest without real bundles is not treated as evaluated
    root = tmp_path / "manifest_only"
    _write_json(
        root / "paper_bundle_manifest.json",
        {"tasks": {"ed_conv": {"role": "primary_root_bundle", "task_dir": "task/"}}},
    )
    assert paper_data.has_evaluated_programs(root) is False
    payload = paper_data.build_paper_data(root, paper_profile="standalone")
    assert payload["totals"]["evaluated_programs"] == 0
    assert payload["ablation_studies"] == []

    # incomplete evaluation manifest is not treated as evaluated
    root = tmp_path / "incomplete"
    program = root / "ideas" / "01_idea" / "program_1"
    _write_text(program / "metrics.json", '{"combined_score": 1.0}')
    (program / "evaluation_manifest.json").write_text("{}")
    assert paper_data.has_evaluated_programs(root) is False
    payload = paper_data.build_paper_data(root, paper_profile="standalone")
    assert payload["totals"]["evaluated_programs"] == 0


def test_build_reference_queries_dedupes_trims_and_stays_task_grounded() -> None:
    queries = paper_data.build_reference_queries(
        task_goal=(
            "# Goal\n- Compare error diffusion on MNIST and CIFAR-10\n"
            "- Focus on biologically plausible credit assignment\n"
        ),
        task_rubrics=(
            "# Rubrics\n- Keep claims task-scoped\n"
            "- Highlight transfer from fully connected to convolutional models\n"
        ),
        idea_titles=["  Dual-stream routing  ", "Dual-stream routing", "Auxiliary local heads"],
    )
    assert len(queries) >= 4
    assert len(queries) == len({item.lower() for item in queries})
    assert "Compare error diffusion on MNIST and CIFAR-10" in queries
    assert "Focus on biologically plausible credit assignment" in queries
    assert "Auxiliary local heads" in queries
    assert "optimization algorithm benchmarking" not in queries
    assert "scientific method empirical evaluation investigation" not in queries


# ── prepare_results_dir_for_paper ──


def test_prepare_results_dir_for_paper_keeps_direct_results_and_merges_parallel_workers(
    tmp_path: Path,
) -> None:
    # keeps direct results
    direct = tmp_path / "direct"
    _write_program_result(direct, "01_direct", score=1.5)
    prepared_dir, meta = paper_data.prepare_results_dir_for_paper(direct)
    assert prepared_dir == direct.resolve()
    assert meta["mode"] == "direct"
    assert paper_data.has_evaluated_programs(prepared_dir)

    # merges parallel workers
    root = tmp_path / "parallel"
    _write_program_result(root / "worker_01", "01_alpha", score=1.0)
    _write_program_result(root / "worker_02", "01_alpha", score=2.0)
    (root / "memory.md").write_text("# Global Memory Log\n\n", encoding="utf-8")
    _write_json(
        root / "memory.jsonl", {"timestamp": "2026-02-18T00:00:00+00:00", "step": "attempt"}
    )
    _write_text(
        root / "aggregated" / "ed_extra" / "run_summary.csv",
        RUN_SUMMARY_HEADER + "X,baseline,1.0,eval,10.0,true\n",
    )
    merge_root = root / "paper" / "_merged_parallel_test"

    prepared_dir, meta = paper_data.prepare_results_dir_for_paper(root, merge_root=merge_root)

    assert prepared_dir == merge_root.resolve()
    assert meta["mode"] == "parallel_merged"
    merged_ideas = sorted(p.name for p in (prepared_dir / "ideas").iterdir() if p.is_dir())
    assert len(merged_ideas) == 2
    assert all(name.startswith("worker_") for name in merged_ideas)
    merged_data = paper_data.build_paper_data(prepared_dir)
    assert merged_data["totals"]["ideas"] == 2
    assert merged_data["totals"]["evaluated_programs"] == 2
    assert (prepared_dir / "aggregated" / "ed_extra" / "run_summary.csv").is_file()
    assert "ed_extra" in merged_data.get("cross_task_results", {})
    assert merged_data.get("task_names") == ["ed_extra"]
    assert (prepared_dir / "memory.md").is_file()
    assert (prepared_dir / "memory.jsonl").is_file()


def test_prepare_results_dir_for_paper_merges_parallel_workers_with_nested_ablation_bundles(
    tmp_path: Path,
) -> None:
    # nested single-task ablation bundle
    root = tmp_path / "single"
    _write_program_result(root / "worker_01", "01_alpha", score=1.0)
    _write_program_result(root / "worker_02", "01_alpha", score=2.0)
    _write_ablation_bundle(root / "variants_with_reps")
    merge_root = root / "paper" / "_merged_parallel_test"

    prepared_dir, meta = paper_data.prepare_results_dir_for_paper(root, merge_root=merge_root)

    assert prepared_dir == merge_root.resolve()
    assert meta["mode"] == "parallel_merged"
    assert meta["nested_ablation_bundle_dir"] == str((root / "variants_with_reps").resolve())
    assert meta["nested_ablation_bundle_mode"] in {"symlinked", "copied"}
    assert (prepared_dir / "variants_with_reps").exists()
    merged_data = paper_data.build_paper_data(prepared_dir, paper_profile="standalone")
    assert merged_data["totals"]["ideas"] == 2
    assert merged_data["totals"]["evaluated_programs"] == 6
    assert merged_data["totals"]["ablation_studies"] == 1
    assert merged_data["totals"]["ablation_variants"] == 2
    assert merged_data["totals"]["ablation_runs"] == 4
    assert len(merged_data["ablation_studies"]) == 1

    # nested multitask bundle keeps the ablation-study grounding source
    root = tmp_path / "multitask"
    _write_program_result(root / "worker_01", "01_alpha", score=1.0)
    _write_program_result(root / "worker_02", "01_beta", score=2.0)
    _write_multitask_ablation_bundle(root / "variants_with_reps")
    prepared_dir, _ = paper_data.prepare_results_dir_for_paper(
        root, merge_root=root / "paper" / "_merged_parallel_test"
    )
    payload = paper_data.build_paper_data(prepared_dir, paper_profile="standalone")
    assert payload["totals"]["ablation_studies"] == 2
    assert payload["grounding_sources"]["result.total_ablation_studies"]["value"] == 2


# ── cross-task results ──


def test_cross_task_collects_per_variant_stats_and_primary_metric_values(tmp_path: Path) -> None:
    # per-task, per-variant stats from run_summary.csv
    root = tmp_path / "stats"
    for task, variants in (
        ("ed_brax_ant", [("ED_PPO", 1.0), ("ED_PPO", 3.0), ("BP_PPO", 2.0)]),
        ("ed_craftax", [("DFA", 5.0)]),
    ):
        rows = "".join(f"{vid},baseline,{score},eval_reward,{score},true\n" for vid, score in variants)
        _write_text(root / "aggregated" / task / "run_summary.csv", RUN_SUMMARY_HEADER + rows)

    result = collect_cross_task_results(root)

    assert result["task_names"] == ["ed_brax_ant", "ed_craftax"]
    ant = result["cross_task_results"]["ed_brax_ant"]
    assert ant["ED_PPO"]["n"] == 2
    assert ant["ED_PPO"]["mean"] == pytest.approx(2.0)
    assert ant["BP_PPO"]["n"] == 1
    sources = result["grounding_sources"]
    assert "result.ed_brax_ant.ed_ppo.eval_reward.mean" in sources
    assert "result.ed_craftax.dfa.eval_reward.n" in sources
    assert sources["result.ed_brax_ant.ed_ppo.eval_reward.n"]["value"] == 2

    # run_summary primary metric values win under the primary metric name
    root = tmp_path / "primary"
    task_dir = root / "aggregated" / "ed_metric"
    _write_text(
        task_dir / "run_summary.csv",
        RUN_SUMMARY_HEADER + "X,baseline,0.1,eval_reward,10.0,true\nX,baseline,0.3,eval_reward,30.0,true\n",
    )
    _write_text(
        task_dir / "metric_values.csv",
        METRIC_VALUES_HEADER
        + "x_1,X,eval_reward,999.0,true\n"
        "x_1,X,components.terminal_regret,0.1,false\n"
        "x_2,X,components.terminal_regret,0.3,false\n",
    )

    result = collect_cross_task_results(root)

    metric = result["cross_task_results"]["ed_metric"]["X"]
    assert metric["metric_name"] == "eval_reward"
    assert metric["mean"] == pytest.approx(20.0)
    sources = result["grounding_sources"]
    assert sources["result.ed_metric.x.components_terminal_regret.mean"]["value"] == 0.2
    primary_source = sources["result.ed_metric.x.eval_reward.mean"]
    assert primary_source["value"] == 20.0
    assert primary_source["source_csv"] == str(task_dir / "run_summary.csv")


def test_cross_task_metric_only_tasks_use_variant_metadata_and_skip_non_finite_values(
    tmp_path: Path,
) -> None:
    # metric-only task with variant metadata
    root = tmp_path / "metadata"
    _write_variant_spec(root, "metric_only", "Control", kind="control")
    _write_text(
        root / "aggregated" / "metric_only" / "metric_values.csv",
        METRIC_VALUES_HEADER + "c_1,Control,terminal_regret,0.25,false\n",
    )
    result = collect_cross_task_results(root)
    assert result["task_names"] == []
    source = result["grounding_sources"]["result.metric_only.control.terminal_regret.mean"]
    assert source["value"] == 0.25
    assert source["variant_kind"] == "control"

    # non-finite metric values are skipped
    root = tmp_path / "finite"
    _write_text(
        root / "aggregated" / "finite" / "metric_values.csv",
        METRIC_VALUES_HEADER
        + "x_1,X,valid,2.0,false\nx_2,X,valid,4.0,false\n"
        "x_3,X,not_finite,nan,false\nx_4,X,not_finite,inf,false\n"
        "x_5,X,huge,1e308,false\nx_6,X,huge,-1e308,false\n",
    )
    sources = collect_cross_task_results(root)["grounding_sources"]
    assert sources["result.finite.x.valid.mean"]["value"] == 3.0
    assert sources["result.finite.x.huge.mean"]["value"] == 0.0
    assert not any("not_finite" in source_id for source_id in sources)


SLUG_COLLISION_CASES = [
    (
        "metric_slugs",
        {"collision/metric_values.csv": METRIC_VALUES_HEADER
         + "x_1,X,loss/a,1.0,false\nx_2,X,loss.a,2.0,false\n"},
    ),
    (
        "run_summary_vs_metric_slug",
        {"collision/run_summary.csv": RUN_SUMMARY_HEADER + "X,baseline,1.0,loss/a,1.0,true\n",
         "collision/metric_values.csv": METRIC_VALUES_HEADER + "x_1,X,loss.a,2.0,false\n"},
    ),
    (
        "task_slugs",
        {"task-a/metric_values.csv": METRIC_VALUES_HEADER + "x_1,X,reward,1.0,false\n",
         "task_a/metric_values.csv": METRIC_VALUES_HEADER + "x_1,X,reward,2.0,false\n"},
    ),
]


def test_cross_task_rejects_grounding_source_slug_collisions(tmp_path: Path) -> None:
    for name, files in SLUG_COLLISION_CASES:
        root = tmp_path / name
        for relative, text in files.items():
            _write_text(root / "aggregated" / relative, text)
        with pytest.raises(ValueError, match="metric grounding source ID collision"):
            collect_cross_task_results(root)


def _rl_timeseries(task: str, metric: str, series: dict[str, list[list[tuple[int, int]]]]) -> str:
    lines = [RL_TIMESERIES_HEADER]
    run = 0
    for variant, reps in series.items():
        for rep, values in enumerate(reps):
            for step, value in values:
                lines.append(
                    f"{variant}__rep_{rep},{variant}__rep_{rep},{variant},{rep},{step},{task},"
                    f"run_{run},{rep},{metric},{value}"
                )
            run += 1
    return "\n".join(lines) + "\n"


def test_cross_task_collects_rl_stats_from_run_timeseries(tmp_path: Path) -> None:
    _write_variant_spec(
        tmp_path, "brax_ant", "BP_PPO", title="PPO with Backpropagation Backbone", kind="baseline"
    )
    _write_variant_spec(
        tmp_path, "brax_ant", "ED_PPO", title="PPO with Error Diffusion Backbone", kind="control"
    )
    _write_text(
        tmp_path / "aggregated" / "brax_ant" / "run_timeseries.csv",
        _rl_timeseries(
            "ant_ppo", "eval/episode_reward",
            {"BP_PPO": [[(0, 100), (10, 200)], [(0, 110), (10, 220)]],
             "ED_PPO": [[(0, 300), (10, 500)], [(0, 320), (10, 700)]]},
        ),
    )

    result = collect_cross_task_results(tmp_path)

    assert result["task_names"] == ["brax_ant"]
    ant = result["cross_task_results"]["brax_ant"]
    assert sorted(ant) == ["BP_PPO", "ED_PPO"]
    assert ant["BP_PPO"]["variant_kind"] == "baseline"
    assert ant["ED_PPO"]["variant_kind"] == "control"
    assert ant["BP_PPO"]["metric_name"] == "eval/episode_reward"
    assert ant["ED_PPO"]["n"] == 2
    assert ant["BP_PPO"]["mean"] == pytest.approx(210.0)
    assert ant["ED_PPO"]["mean"] == pytest.approx(600.0)
    assert ant["ED_PPO"]["source_layout"] == "run_timeseries_terminal"
    sources = result["grounding_sources"]
    assert "result.brax_ant.bp_ppo.eval_episode_reward.mean" in sources
    assert "result.brax_ant.ed_ppo.eval_episode_reward.n" in sources


def test_build_paper_data_merges_cross_task_stats_from_run_summary_and_run_timeseries(
    tmp_path: Path,
) -> None:
    # run_summary.csv cross-task results merge into build_paper_data
    root = tmp_path / "summary"
    (root / "ideas").mkdir(parents=True)
    _write_text(
        root / "aggregated" / "ed_extra" / "run_summary.csv",
        RUN_SUMMARY_HEADER + "X,baseline,1.0,eval,1.0,true\nX,baseline,2.0,eval,2.0,true\n",
    )
    data = paper_data.build_paper_data(root)
    assert data.get("task_names") == ["ed_extra"]
    assert "ed_extra" in data.get("cross_task_results", {})
    assert "result.ed_extra.x.eval.mean" in data["grounding_sources"]

    # RL run_timeseries.csv cross-task stats merge into build_paper_data
    root = tmp_path / "timeseries"
    (root / "ideas").mkdir(parents=True)
    _write_variant_spec(root, "craftax", "DFA", kind="baseline")
    _write_text(
        root / "aggregated" / "craftax" / "run_timeseries.csv",
        _rl_timeseries(
            "craftax", "episode_return", {"DFA": [[(1, 10), (2, 20)], [(1, 30), (2, 40)]]}
        ),
    )
    data = paper_data.build_paper_data(root)
    assert data.get("task_names") == ["craftax"]
    assert "craftax" in data.get("cross_task_results", {})
    assert data["cross_task_results"]["craftax"]["DFA"]["mean"] == pytest.approx(30.0)
    assert "result.craftax.dfa.episode_return.mean" in data["grounding_sources"]


# ── final evaluation wiring ──


def test_build_paper_data_registers_sealed_final_metrics_and_rejects_non_qualifying(
    tmp_path: Path,
) -> None:
    final_dir = _write_final_manifest_stub(tmp_path, "custom-final")

    # registers sealed final metrics
    manifest = {
        "status": "completed", "correct": True, "metrics": {"held_out_accuracy": 0.83},
        "program_tree_sha256": "sha256:" + "1" * 64, "task_tree_sha256": "sha256:" + "2" * 64,
        "evaluator_sha256": "sha256:" + "3" * 64, "results_tree_sha256": "sha256:" + "4" * 64,
    }
    with patch("src.orchestrator.paper_data.verify_final_manifest", return_value=manifest) as verify:
        data = paper_data.build_paper_data(tmp_path, final_evaluation_dir=final_dir)
    assert data["final_evaluation"]["metrics"]["held_out_accuracy"] == 0.83
    assert data["final_evaluation"]["task_tree_sha256"] == "sha256:" + "2" * 64
    assert data["grounding_sources"]["final.metric.held_out_accuracy"]["value"] == 0.83
    verify.assert_called_once_with(final_dir.resolve(), task_dir=None, program_dir=None)

    # rejects non-qualifying final metrics
    manifest = {
        "schema_version": 2, "status": "completed", "outcome": "not_qualified",
        "correct": False, "metrics": {"held_out_accuracy": 0.83},
    }
    with (
        patch("src.orchestrator.paper_data.verify_final_manifest", return_value=manifest),
        pytest.raises(ValueError, match="did not qualify"),
    ):
        paper_data.build_paper_data(tmp_path, final_evaluation_dir=final_dir)


def test_build_paper_data_omits_final_evaluation_explicitly_and_rejects_unverified_manifest(
    tmp_path: Path,
) -> None:
    # can explicitly omit final evaluation
    root = tmp_path / "omit"
    _write_final_manifest_stub(root)
    with patch(
        "src.orchestrator.paper_data.verify_final_manifest",
        side_effect=AssertionError("omitted final evaluation must not be read"),
    ):
        data = paper_data.build_paper_data(
            root, final_evaluation_mode=paper_data.FinalEvaluationMode.OMIT
        )
    assert "final_evaluation" not in data

    # rejects unverified final manifest
    root = tmp_path / "unverified"
    (root / "ideas").mkdir(parents=True)
    _write_json(
        root / "final_evaluation" / "final_manifest.json",
        {"status": "completed", "metrics": {"score": 1.0}},
    )
    with pytest.raises(ValueError, match="Invalid completed final evaluation"):
        paper_data.build_paper_data(root)


# ── posthoc analyses ──


def _make_analysis(
    posthoc_root: Path,
    analysis_id: str,
    attempt: str = "attempt_2",
    *,
    family: str = "",
    png_names: list[str] | None = None,
    data_files: dict[str, Any] | None = None,
    run_analysis_code: str | None = None,
) -> dict[str, Any]:
    """Create a mock analysis directory and return its ranking.json 'valid' entry."""
    attempt_dir = posthoc_root / "analyses" / analysis_id / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    _write_text(attempt_dir / "analysis_summary.md", "# Summary\nSome findings.")
    _write_json(attempt_dir / "interestingness_rubric.json", {"overall": "medium"})
    for name in png_names or ["plot.png"]:
        (attempt_dir / name).write_bytes(b"\x89PNG fake")
    for fname, content in (data_files or {}).items():
        _write_json(attempt_dir / fname, content)
    if run_analysis_code is not None:
        _write_text(attempt_dir / "run_analysis.py", run_analysis_code)
    remote = f"/remote/machine/analyses/{analysis_id}"
    return {
        "analysis_id": analysis_id,
        "title": analysis_id.replace("_", " ").title(),
        "family": family or "test_family",
        "selected_attempt": attempt,
        "status": "completed",
        "attempt_count": 2,
        # Absolute paths from a different machine must be ignored.
        "analysis_dir": remote,
        "selected_attempt_dir": f"{remote}/{attempt}",
        "analysis_summary_path": f"{remote}/{attempt}/analysis_summary.md",
        "interestingness_rubric_path": f"{remote}/{attempt}/interestingness_rubric.json",
    }


def _make_session(tmp_path: Path, specs: list[dict[str, str]] | None = None) -> Path:
    """Flat posthoc directory: tmp_path/posthoc/{ranking.json, analyses/}."""
    posthoc_root = tmp_path / "posthoc"
    (posthoc_root / "analyses").mkdir(parents=True)
    specs = specs or [{"analysis_id": "alpha", "family": "fam_a"}, {"analysis_id": "beta", "family": "fam_b"}]
    entries = [_make_analysis(posthoc_root, spec["analysis_id"], family=spec.get("family", "")) for spec in specs]
    _write_json(posthoc_root / "ranking.json", {"valid": entries, "rejected": [], "selected_analysis_id": None})
    return posthoc_root


def _single_analysis_session(
    tmp_path: Path, analysis_id: str, root_name: str = "posthoc", **kwargs: Any
) -> tuple[Path, Path]:
    """Posthoc root holding one analysis; returns (posthoc_root, attempt_dir)."""
    posthoc_root = tmp_path / root_name
    entry = _make_analysis(posthoc_root, analysis_id, **kwargs)
    _write_json(posthoc_root / "ranking.json", {"valid": [entry], "rejected": []})
    return posthoc_root, posthoc_root / "analyses" / analysis_id / "attempt_2"


def _session_dir(path: Path) -> Path:
    (path / "analyses").mkdir(parents=True)
    _write_json(path / "ranking.json", {"valid": []})
    return path


INCOMPLETE_POSTHOC_LAYOUTS = [
    ("missing_posthoc_dir", lambda root: None),
    ("empty_posthoc_dir", lambda root: (root / "posthoc").mkdir()),
    ("missing_ranking_json", lambda root: (root / "posthoc" / "analyses").mkdir(parents=True)),
    ("missing_analyses_dir", lambda root: _write_json(root / "posthoc" / "ranking.json", {"valid": []})),
    ("nested_session_layout_unsupported", lambda root: _session_dir(root / "posthoc" / "session_001")),
]


class TestDiscoverPosthocSession:
    def test_found_and_incomplete_layouts_are_not_discovered(self, tmp_path: Path) -> None:
        # found
        _make_session(tmp_path / "found")
        result = discover_posthoc_session(tmp_path / "found")
        assert result is not None
        assert result.name == "posthoc"
        # incomplete layouts are not discovered
        for name, layout in INCOMPLETE_POSTHOC_LAYOUTS:
            root = tmp_path / name
            root.mkdir()
            layout(root)
            assert discover_posthoc_session(root) is None, name

    def test_accepts_explicit_run_directory_and_rejects_symlinked_roots(self, tmp_path: Path) -> None:
        # accepts explicit posthoc run directory
        root = tmp_path / "explicit"
        nested = _session_dir(root / "producer" / "posthoc" / "session_001")
        assert discover_posthoc_session(root, posthoc_dir=nested) == nested.resolve()
        # rejects symlinked posthoc roots
        root = tmp_path / "symlinked"
        outside = _session_dir(root / "outside")
        implicit_link = root / "results" / "posthoc"
        implicit_link.parent.mkdir()
        implicit_link.symlink_to(outside, target_is_directory=True)
        explicit_link = root / "explicit_link"
        explicit_link.symlink_to(outside, target_is_directory=True)
        assert discover_posthoc_session(root / "results") is None
        assert discover_posthoc_session(root, posthoc_dir=explicit_link) is None

    def test_stable_latest_manifest_resolution(self, tmp_path: Path) -> None:
        # resolves stable latest manifest
        root = tmp_path / "resolves"
        nested = _session_dir(root / "posthoc" / "session_001")
        _write_json(
            root / "posthoc" / "latest_manifest.json",
            {"manifest_version": 1, "output_dir": str(nested)},
        )
        assert discover_posthoc_session(root) == nested.resolve()
        # rejects stable manifest outside posthoc root
        root = tmp_path / "outside_root"
        outside = _session_dir(root / "outside")
        _write_json(
            root / "posthoc" / "latest_manifest.json",
            {"manifest_version": 1, "output_dir": str(outside)},
        )
        assert discover_posthoc_session(root) is None
        # stable manifest precedes legacy flat layout
        root = tmp_path / "legacy"
        legacy = _make_session(root, [{"analysis_id": "legacy"}])
        current = _session_dir(legacy / "session_current")
        _write_json(
            legacy / "latest_manifest.json", {"manifest_version": 1, "output_dir": str(current)}
        )
        assert discover_posthoc_session(root) == current.resolve()


UNRESOLVABLE_RANKING_ENTRIES = [
    ("missing_attempt", {"analysis_id": "missing", "selected_attempt": "attempt_99"}),
    ("escapes_root", {"analysis_id": "..", "selected_attempt": "escaped/attempt_2"}),
]
UNLOADABLE_DATA_FILES = [
    ("over_size_limit", "data/large.csv", "x\n" + "1\n" * (3 * 1024 * 1024)),
    ("large_json_integer", "data.json", '{"value": ' + "9" * 5000 + "}"),
]


class TestCollectPosthocAnalyses:
    def test_collects_valid_analyses_figures_and_empty_raw_data(self, tmp_path: Path) -> None:
        # valid session
        root = tmp_path / "valid"
        _make_session(root)
        analyses = collect_posthoc_analyses(root)
        assert [a["analysis_id"] for a in analyses] == ["alpha", "beta"]
        assert "# Summary" in analyses[0]["summary_text"]
        assert isinstance(analyses[0]["rubric"], dict)
        assert len(analyses[0]["figure_paths"]) == 1
        assert analyses[0]["figure_paths"][0].endswith("plot.png")
        # Paths resolve locally, not to the remote machine recorded in ranking.json.
        assert str(root) in analyses[0]["attempt_dir"]
        # interestingness_rubric.json is a control file and is excluded from raw data
        assert len(analyses) == 2
        for analysis in analyses:
            assert analysis["raw_data"] == {}
            assert analysis["run_analysis_code"] == ""
        # empty when no posthoc
        empty = tmp_path / "empty"
        empty.mkdir()
        assert collect_posthoc_analyses(empty) == []
        # multiple pngs
        root = tmp_path / "multi"
        _single_analysis_session(
            root, "multi_fig", png_names=["fig_a.png", "fig_b.png", "fig_c.png"]
        )
        analyses = collect_posthoc_analyses(root)
        assert len(analyses) == 1
        assert len(analyses[0]["figure_paths"]) == 3

    def test_skips_unresolvable_ranking_entries_and_missing_summary(self, tmp_path: Path) -> None:
        # unresolvable ranking entries are skipped
        for name, bad_entry in UNRESOLVABLE_RANKING_ENTRIES:
            root = tmp_path / name
            session = _make_session(root, [{"analysis_id": "good"}])
            _write_text(root / "escaped" / "attempt_2" / "analysis_summary.md", "# Escaped")
            ranking = json.loads((session / "ranking.json").read_text(encoding="utf-8"))
            ranking["valid"].append({**bad_entry, "title": "Bad", "family": "fam"})
            _write_json(session / "ranking.json", ranking)
            analyses = collect_posthoc_analyses(root)
            assert [a["analysis_id"] for a in analyses] == ["good"], name
        # missing summary is skipped
        root = tmp_path / "no_summary"
        session = _make_session(root, [{"analysis_id": "no_summary"}])
        (session / "analyses" / "no_summary" / "attempt_2" / "analysis_summary.md").unlink()
        assert collect_posthoc_analyses(root) == []

    def test_collects_manifest_declared_data_and_rejects_paths_outside_attempt(
        self, tmp_path: Path
    ) -> None:
        # collects manifest-declared nested json and csv
        root = tmp_path / "nested"
        posthoc_root, attempt_dir = _single_analysis_session(
            root, "nested_data", "posthoc_run", png_names=[]
        )
        _write_json(attempt_dir / "data" / "details.json", {"score": 0.91})
        _write_json(attempt_dir / "secrets.json", {"token": "must-not-leak"})
        _write_text(attempt_dir / "data" / "curve.csv", "step,value\n1,0.5\n2,0.9\n")
        (attempt_dir / "figures").mkdir()
        (attempt_dir / "figures" / "curve.png").write_bytes(b"\x89PNG fake")
        _write_json(
            attempt_dir / "analysis_manifest.json",
            {"outputs": {"figures": ["figures/curve.png"],
                         "data_files": ["data/details.json", "data/curve.csv"]}},
        )

        analyses = collect_posthoc_analyses(root, posthoc_dir=posthoc_root)

        assert analyses[0]["raw_data"]["data/details.json"] == {"score": 0.91}
        assert analyses[0]["raw_data"]["data/curve.csv"] == [
            {"step": "1", "value": "0.5"}, {"step": "2", "value": "0.9"},
        ]
        assert "secrets" not in analyses[0]["raw_data"]
        assert analyses[0]["figure_paths"] == [str((attempt_dir / "figures" / "curve.png").resolve())]

        # rejects manifest data paths outside the attempt
        root = tmp_path / "unsafe"
        posthoc_root, attempt_dir = _single_analysis_session(
            root, "unsafe_data", "posthoc_run", png_names=[]
        )
        outside = root / "outside.json"
        _write_json(outside, {"secret": True})
        (attempt_dir / "linked.json").symlink_to(outside)
        outside_dir = root / "outside_dir"
        _write_json(outside_dir / "nested.json", {"secret": "nested"})
        (attempt_dir / "linked_dir").symlink_to(outside_dir, target_is_directory=True)
        _write_json(
            attempt_dir / "analysis_manifest.json",
            {"outputs": {"data_files": [
                "../../../../outside.json", str(outside), "linked.json", "linked_dir/nested.json",
            ]}},
        )
        analyses = collect_posthoc_analyses(root, posthoc_dir=posthoc_root)
        assert analyses[0]["raw_data"] == {}

    def test_skips_unloadable_data_files_and_invalid_manifest_disables_legacy_scan(
        self, tmp_path: Path
    ) -> None:
        # unloadable manifest data files are skipped
        for name, file_name, text in UNLOADABLE_DATA_FILES:
            root = tmp_path / name
            posthoc_root, attempt_dir = _single_analysis_session(
                root, "large_data", "posthoc_run", png_names=[]
            )
            _write_text(attempt_dir / file_name, text)
            _write_json(attempt_dir / "analysis_manifest.json", {"outputs": {"data_files": [file_name]}})
            analyses = collect_posthoc_analyses(root, posthoc_dir=posthoc_root)
            assert analyses[0]["raw_data"] == {}, name
        # invalid existing manifest does not enable legacy data scan
        for index, manifest_text in enumerate(["{}", "{malformed"]):
            root = tmp_path / f"invalid_manifest_{index}"
            posthoc_root, attempt_dir = _single_analysis_session(
                root, "invalid_manifest", "posthoc_run", png_names=[],
                data_files={"secrets.json": {"token": "must-not-leak"}},
            )
            _write_text(attempt_dir / "analysis_manifest.json", manifest_text)
            analyses = collect_posthoc_analyses(root, posthoc_dir=posthoc_root)
            assert analyses[0]["raw_data"] == {}, manifest_text

    def test_raw_data_and_run_analysis_code_collection(self, tmp_path: Path) -> None:
        # raw data and code collected
        root = tmp_path / "with_code"
        _single_analysis_session(
            root, "metrics_analysis",
            data_files={
                "per_class_metrics.json": {"overall_accuracy": 0.61, "num_classes": 10},
                "dynamics_summary.json": {"best_val_accuracy": 0.60},
            },
            run_analysis_code="import torch\nprint('analysis')",
        )
        analyses = collect_posthoc_analyses(root)
        assert len(analyses) == 1
        assert "per_class_metrics" in analyses[0]["raw_data"]
        assert "dynamics_summary" in analyses[0]["raw_data"]
        assert analyses[0]["raw_data"]["per_class_metrics"]["overall_accuracy"] == 0.61
        assert "import torch" in analyses[0]["run_analysis_code"]
        # run_analysis.py missing
        root = tmp_path / "no_code"
        _single_analysis_session(root, "no_code", data_files={"metrics.json": {"val": 1}})
        analyses = collect_posthoc_analyses(root)
        assert analyses[0]["run_analysis_code"] == ""
        assert "metrics" in analyses[0]["raw_data"]


def test_resolve_attempt_dir_rejects_path_segments(tmp_path: Path) -> None:
    (tmp_path / "posthoc" / "analyses").mkdir(parents=True)
    for analysis_id, attempt, message in (
        ("..", "attempt_2", "Invalid analysis_id"),
        ("alpha", "nested/attempt_2", "Invalid selected_attempt"),
    ):
        with pytest.raises(ValueError, match=message):
            _resolve_attempt_dir(tmp_path / "posthoc", analysis_id, attempt)


def test_copy_posthoc_figures_warns_and_skips_pngs_only_when_figures_exist(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # warns and skips pngs
    _make_session(tmp_path)
    analyses = collect_posthoc_analyses(tmp_path)
    dest = tmp_path / "output" / "figures" / "posthoc"
    with caplog.at_level("WARNING"):
        copy_posthoc_figures(analyses, dest)
    assert not dest.exists()
    assert "alpha" in caplog.text
    assert "skipping copy" in caplog.text
    # no warning when no figures
    caplog.clear()
    with caplog.at_level("WARNING"):
        copy_posthoc_figures([{"analysis_id": "empty", "figure_paths": []}], Path("/tmp/unused"))
    assert "skipping copy" not in caplog.text


def _base_paper_data() -> dict[str, Any]:
    return {
        "totals": {"evaluated_programs": 10},
        "grounding_sources": {"result.best_correct_score": {"id": "result.best_correct_score"}},
        "posthoc_analyses": [],
    }


def _analyses() -> list[dict[str, Any]]:
    return [
        {
            "analysis_id": "alpha", "title": "Alpha Analysis", "family": "fam_a",
            "summary_text": "Alpha findings.", "rubric": {"overall": 4.0},
            "figure_paths": ["/tmp/alpha/plot.png"],
            "raw_data": {"per_class_metrics": {"overall_accuracy": 0.61}},
            "run_analysis_code": "print('alpha')", "attempt_dir": "/tmp/alpha",
        },
        {
            "analysis_id": "beta", "title": "Beta Analysis", "family": "fam_b",
            "summary_text": "Beta findings.", "rubric": {"overall": 3.5},
            "figure_paths": [],
            "raw_data": {"dynamics_summary": {"best_val_accuracy": 0.60}},
            "run_analysis_code": "", "attempt_dir": "/tmp/beta",
        },
    ]


def _ranking_result() -> dict[str, Any]:
    return {
        "ranked_analyses": [
            {
                "analysis_id": "beta", "rank": 1, "relevance_to_thesis": "high",
                "relevance_justification": "Directly supports central claim.",
                "recommended_usage": "main_results",
                "key_finding_for_paper": "Beta is essential.",
            },
            {
                "analysis_id": "alpha", "rank": 2, "relevance_to_thesis": "medium",
                "relevance_justification": "Supplementary evidence.",
                "recommended_usage": "supporting",
                "key_finding_for_paper": "Alpha adds depth.",
            },
        ],
        "ranking_rationale": "Beta is more relevant to the thesis.",
    }


RANKING_CONTRACT_VIOLATIONS = [
    ("coverage", lambda r: r["ranked_analyses"].pop(), "missing analysis IDs.*alpha"),
    (
        "duplicate_ids", lambda r: r["ranked_analyses"][1].update(analysis_id="beta"),
        "duplicate analysis IDs.*beta",
    ),
    (
        "non_contiguous_ranks", lambda r: r["ranked_analyses"][1].update(rank=3),
        "ranks must be the contiguous integers",
    ),
    ("rationale", lambda r: r.update(ranking_rationale=""), "ranking_rationale"),
    *(
        (
            f"invalid_{field}",
            (lambda f, v: lambda r: r["ranked_analyses"][0].update({f: v}))(field, value),
            field,
        )
        for field, value in (
            ("relevance_to_thesis", "critical"), ("recommended_usage", "discard"),
            ("relevance_justification", ""), ("key_finding_for_paper", "  "),
        )
    ),
]


class TestAugmentPaperData:
    def test_augmentation_ranks_analyses_registers_sources_and_preserves_existing_evidence(
        self,
    ) -> None:
        # ranks analyses and registers totals and sources
        result = augment_paper_data_with_posthoc(_base_paper_data(), _ranking_result(), _analyses())
        posthoc = result["posthoc_analyses"]
        assert len(posthoc) == 2
        assert posthoc[0]["analysis_id"] == "beta"
        assert posthoc[0]["rank"] == 1
        assert posthoc[0]["relevance_justification"] == "Directly supports central claim."
        assert posthoc[1]["analysis_id"] == "alpha"
        assert posthoc[1]["rank"] == 2
        assert posthoc[0]["raw_data"] == {"dynamics_summary": {"best_val_accuracy": 0.60}}
        assert posthoc[0]["run_analysis_code"] == ""
        assert posthoc[1]["raw_data"] == {"per_class_metrics": {"overall_accuracy": 0.61}}
        assert posthoc[1]["run_analysis_code"] == "print('alpha')"
        assert result["totals"]["posthoc_analyses_total"] == 2
        assert result["totals"]["posthoc_analyses_ranked"] == 2
        gs = result["grounding_sources"]
        assert "result.best_correct_score" in gs
        assert "posthoc.beta.summary" in gs
        assert "posthoc.alpha.summary" in gs
        assert gs["posthoc.beta.summary"]["type"] == "posthoc_analysis"
        assert gs["posthoc.alpha.data.per_class_metrics"]["type"] == "posthoc_data"
        assert gs["posthoc.alpha.data.per_class_metrics"]["data_file"] == "per_class_metrics"
        assert gs["posthoc.beta.data.dynamics_summary"]["type"] == "posthoc_data"

        # existing posthoc evidence is preserved
        base_paper_data = _base_paper_data()
        existing = {
            "analysis_id": "beta", "rank": 7, "summary_text": "Authoritative source summary.",
            "relevance_justification": "Authoritative source justification.",
            "key_finding_for_paper": "Authoritative source finding.",
        }
        base_paper_data["posthoc_analyses"] = [existing]
        base_paper_data["totals"]["posthoc_analyses_total"] = 1
        base_paper_data["grounding_sources"]["posthoc.beta.summary"] = {
            "id": "posthoc.beta.summary", "type": "authoritative_source",
        }
        expected_existing = dict(existing)

        result = augment_paper_data_with_posthoc(base_paper_data, _ranking_result(), _analyses())

        assert result["posthoc_analyses"][0] == expected_existing
        assert result["posthoc_analyses"][1]["analysis_id"] == "alpha"
        assert result["totals"]["posthoc_analyses_total"] == 1
        assert result["grounding_sources"]["posthoc.beta.summary"]["type"] == "authoritative_source"

    def test_malformed_existing_raw_data_does_not_create_grounding_sources(self) -> None:
        for raw_data in (None, 7, "raw", ["metric"], {"": 1, 7: 2}):
            base_paper_data = _base_paper_data()
            base_paper_data["posthoc_analyses"] = [{"analysis_id": "beta", "raw_data": raw_data}]
            result = augment_paper_data_with_posthoc(base_paper_data, _ranking_result(), _analyses())
            assert not any(
                s.startswith("posthoc.beta.data.") for s in result["grounding_sources"]
            ), repr(raw_data)

    def test_unknown_analysis_id_and_empty_ranking_rejected(self) -> None:
        # unknown analysis id in ranking rejected
        ranking = {"ranked_analyses": [
            {"analysis_id": "nonexistent", "rank": 1}, {"analysis_id": "alpha", "rank": 2},
        ]}
        with pytest.raises(ValueError, match="unknown analysis IDs"):
            augment_paper_data_with_posthoc(_base_paper_data(), ranking, _analyses())
        # empty ranking rejected
        with pytest.raises(ValueError, match="ranked_analyses must be a list"):
            augment_paper_data_with_posthoc(_base_paper_data(), {}, _analyses())

    def test_ranking_contract_violations_are_rejected(self) -> None:
        for name, mutate, message in RANKING_CONTRACT_VIOLATIONS:
            ranking_result = _ranking_result()
            mutate(ranking_result)
            with pytest.raises(ValueError, match=message) as excinfo:
                validate_posthoc_ranking(ranking_result, _analyses())
            assert excinfo.type is ValueError, name

    def test_ranking_accepts_appendix_and_sorts_by_rank(self) -> None:
        ranking_result = _ranking_result()
        ranking_result["ranked_analyses"][1]["recommended_usage"] = "appendix"
        ranking_result["ranked_analyses"].reverse()
        ranked = validate_posthoc_ranking(ranking_result, _analyses())
        assert [entry["analysis_id"] for entry in ranked] == ["beta", "alpha"]
        assert ranked[1]["recommended_usage"] == "appendix"


class TestCollectDataFiles:
    def test_collects_data_json_only_and_excludes_control_files(self, tmp_path: Path) -> None:
        # collects data json only
        root = tmp_path / "data"
        _write_json(root / "per_class_metrics.json", {"accuracy": 0.6})
        _write_json(root / "dynamics_summary.json", {"epochs": 250})
        _write_json(root / "analysis_spec.json", {"spec": True})
        _write_json(root / "interestingness_rubric.json", {"overall": 4.0})
        result = _collect_data_files(root)
        assert set(result) == {"per_class_metrics", "dynamics_summary"}
        assert result["per_class_metrics"] == {"accuracy": 0.6}
        # excludes all control files
        root = tmp_path / "control"
        for name in _CONTROL_JSON_NAMES:
            _write_json(root / name, {"control": True})
        _write_json(root / "real_data.json", {"val": 42})
        assert list(_collect_data_files(root)) == ["real_data"]

    def test_empty_nonexistent_and_mixed_dirs(self, tmp_path: Path) -> None:
        # empty and nonexistent dirs
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _collect_data_files(empty) == {}
        assert _collect_data_files(tmp_path / "nope") == {}
        # skips non-json and invalid json
        root = tmp_path / "mixed"
        _write_json(root / "data.json", {"val": 1})
        _write_text(root / "summary.md", "text")
        (root / "plot.png").write_bytes(b"\x89PNG")
        (root / "bad.json").write_text("not json {{{", encoding="utf-8")
        assert list(_collect_data_files(root)) == ["data"]


# ── ALIFE bundle normalisation ──

RUN_SUMMARY_FIELDS = [
    "run_id", "evaluation_id", "variant_id", "variant_kind", "base_program", "agent_name",
    "correct", "combined_score", "overall_pass", "primary_metric_name", "primary_metric_value",
    "program_dir", "results_dir",
]
METRIC_VALUES_FIELDS = [
    "run_id", "evaluation_id", "variant_id", "metric_name", "metric_value", "is_primary_metric",
]
RL_TASKS = ("brax_ant", "brax_halfcheetah", "brax_humanoid", "craftax")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_variant(path: Path, *, title: str, kind: str) -> None:
    _write_json(
        path / "variant_spec.json",
        {"variant_id": path.name, "title": title, "kind": kind, "base_program": "best"},
    )


def _write_alife_task(
    root: Path, task: str, combined: float, metric: float, series: float, stale: str
) -> None:
    aggregated = root / "aggregated" / task
    base = {"run_id": "rep_0", "evaluation_id": "control_best_eval_1", "variant_id": "control_best"}
    _write_csv(
        aggregated / "run_summary.csv", RUN_SUMMARY_FIELDS,
        [{**base, "variant_kind": "control", "base_program": "best", "agent_name": "unknown",
          "correct": "True", "combined_score": combined, "overall_pass": "True",
          "primary_metric_name": "test_accuracy", "primary_metric_value": metric,
          "program_dir": f"/stale/{stale}program_dir", "results_dir": f"/stale/{stale}results_dir"}],
    )
    _write_csv(
        aggregated / "metric_values.csv", METRIC_VALUES_FIELDS,
        [{**base, "metric_name": "test_accuracy", "metric_value": metric, "is_primary_metric": "True"}],
    )
    (aggregated / "timeseries.csv").write_text(f"step,val\n0,{series}\n", encoding="utf-8")


def _rl_rows(task: str, variant: str, seed: int, values: tuple[float, float]) -> list[dict[str, object]]:
    return [
        {"evaluation_id": f"{variant}__rep_0", "variant_id": variant, "algorithm": variant,
         "replication": 0, "step": step, "experiment": task, "run": f"{task}_seed{seed}",
         "seed": seed, "metric": "eval/episode_reward", "value": value}
        for step, value in zip((0, 10), values)
    ]


def _build_alife_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Test bundle\n", encoding="utf-8")
    _write_text(root / "task" / "task.md", "# Goal\n- test\n\n# Rubrics\n- test\n")
    _make_variant(root / "variants" / "control_best", title="Control", kind="control")
    _write_json(
        root / "pipeline_summary.json",
        {"variants": [{"variant_id": "control_best", "kind": "control", "title": "Control",
                       "status": "completed", "rubric_passed": True}]},
    )
    _write_alife_task(root, "ed_conv", 0.61, 0.62, 0.1, "")
    _write_text(
        root / "extra_tasks" / "ed_mnist" / "_retrofitted_bundle" / "task.md",
        "# Goal\n- mnist\n\n# Rubrics\n- mnist\n",
    )
    _make_variant(
        root / "extra_tasks" / "ed_mnist" / "variants" / "control_best",
        title="MNIST Control", kind="control",
    )
    _write_alife_task(root, "ed_mnist", 0.95, 0.96, 0.9, "mnist_")
    for task_name in RL_TASKS:
        task_root = root / "extra_tasks" / task_name
        _write_text(task_root / "task.md", f"# {task_name}\n")
        _make_variant(task_root / "ED_PPO", title="ED PPO", kind="control")
        _make_variant(task_root / "BP_PPO", title="BP PPO", kind="baseline")
        rows = _rl_rows(task_name, "ED_PPO", 0, (1.0, 3.5)) + _rl_rows(task_name, "BP_PPO", 1, (2.0, 4.0))
        if task_name == "craftax":
            _make_variant(task_root / "DFA", title="DFA", kind="baseline")
            rows += _rl_rows(task_name, "DFA", 2, (0.5, 0.75))
        _write_csv(root / "aggregated" / task_name / "run_timeseries.csv", list(rows[0]), rows)


def test_normalize_alife_bundle_creates_canonical_task_bundles(tmp_path: Path) -> None:
    bundle_root = tmp_path / "alife_data"
    _build_alife_fixture(bundle_root)

    outputs = normalize_alife_bundle(bundle_root)
    outputs = normalize_alife_bundle(bundle_root)  # idempotent

    assert outputs.root_canonical_ready is True
    assert outputs.extra_task_ready["ed_mnist"] is True
    assert outputs.extra_task_ready["brax_ant"] is True
    for sub in ("", "extra_tasks/ed_mnist", "extra_tasks/brax_ant", "extra_tasks/craftax"):
        assert has_ablation_bundle(bundle_root / sub) is True

    root_run_summary = list(csv.DictReader((bundle_root / "aggregated" / "run_summary.csv").open()))
    assert len(root_run_summary) == 1
    assert root_run_summary[0]["program_dir"].endswith("variants/control_best/task_bundle/program_1/src")

    mnist_task_md = (bundle_root / "extra_tasks" / "ed_mnist" / "task.md").read_text(encoding="utf-8")
    assert "# Goal" in mnist_task_md
    assert "# Rubrics" in mnist_task_md

    brax = bundle_root / "extra_tasks" / "brax_ant"
    brax_rows = list(csv.DictReader((brax / "aggregated" / "run_summary.csv").open()))
    assert {row["variant_id"] for row in brax_rows} == {"ED_PPO", "BP_PPO"}
    assert {row["primary_metric_value"] for row in brax_rows} == {"3.5", "4.0"}
    assert (brax / "variants" / "ED_PPO").is_symlink()
    assert (brax / "aggregated" / "run_timeseries.csv").is_file()
    brax_manifest = json.loads((brax / "ED_PPO" / "run_manifest.json").read_text(encoding="utf-8"))
    assert all((brax / "ED_PPO" / rel).exists() for rel in brax_manifest["expected_analysis_files"])

    manifest = json.loads((bundle_root / "paper_bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["primary_task"] == "ed_conv"
    assert manifest["tasks"]["craftax"]["timeseries_path"].endswith("run_timeseries.csv")

    readme_text = (bundle_root / "README.md").read_text(encoding="utf-8")
    assert "Canonical paper-writing layout" in readme_text
    assert "paper_bundle_manifest.json" in readme_text
    assert "option 2A" in (bundle_root / "bundle_normalization_report.md").read_text(encoding="utf-8")
