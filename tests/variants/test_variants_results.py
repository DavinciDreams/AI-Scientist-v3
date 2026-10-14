"""Tests for variants_pipeline aggregate (CSV merging) and readme (overview assembly)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.orchestrator.paper_data_ablation import has_ablation_bundle
from src.variants_pipeline.aggregate import (
    _find_analysis_csvs,
    _find_flat_csvs,
    _is_flat_extra_task,
    _merge_csvs,
    _read_csv_headers,
    aggregate_variant_csvs,
)
from src.variants_pipeline.models import VariantSpec
from src.variants_pipeline import readme as readme_mod
from src.variants_pipeline.readme import (
    _assemble_readme,
    _build_readme_prompt,
    _build_status_map,
    _discover_tasks,
    _fallback_summary,
    _fallback_task_summary,
    _load_flat_variant_specs,
    _parse_agent_summaries,
)

# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path | None) -> tuple[list[str], list[dict[str, str]]]:
    assert path is not None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _variant_csv(root: Path, variant: str, program: str, name: str, headers: list[str], rows) -> Path:
    """Write <root>/variants/<variant>/task_bundle/<program>/analysis/<name>."""
    path = root / "variants" / variant / "task_bundle" / program / "analysis" / name
    _write_csv(path, headers, rows)
    return path


_EVAL_ROW = (["evaluation_id", "variant_id"], [{"evaluation_id": "e1", "variant_id": "control_best"}])
_METRIC_ROW = (
    ["evaluation_id", "variant_id", "metric_name", "metric_value"],
    [{"evaluation_id": "e1", "variant_id": "control_best", "metric_name": "acc", "metric_value": "0.95"}],
)


def test_read_csv_headers_and_find_analysis_csvs(tmp_path: Path) -> None:
    # read_csv_headers
    _write_csv(tmp_path / "test.csv", ["a", "b", "c"], [{"a": "1", "b": "2", "c": "3"}])
    assert _read_csv_headers(tmp_path / "test.csv") == ["a", "b", "c"]
    assert _read_csv_headers(tmp_path / "nope.csv") == []

    # picks the highest program_N per variant and ignores non-program dirs
    root = tmp_path / "multi"
    _variant_csv(root, "ablation_foo", "program_0", "run_summary.csv", ["a"], [{"a": "seed"}])
    _variant_csv(root, "ablation_foo", "program_1", "run_summary.csv", ["a"], [{"a": "1"}])
    _variant_csv(root, "control_best", "program_2", "run_summary.csv", ["a"], [{"a": "2"}])
    _variant_csv(root, "control_best", "other", "run_summary.csv", ["a"], [{"a": "x"}])  # not program_N
    paths = _find_analysis_csvs(root / "variants", "run_summary.csv")
    assert sorted(p.parent.parent.name for p in paths) == ["program_1", "program_2"]

    # single variant: program_0 only / highest of many
    for name, programs, expected in (("program_0_only", [0], "program_0"), ("highest_of_many", [1, 2, 3], "program_3")):
        root = tmp_path / name
        for i in programs:
            _variant_csv(root, "baseline_dfa", f"program_{i}", "run_summary.csv", ["a"], [{"a": str(i)}])
        paths = _find_analysis_csvs(root / "variants", "run_summary.csv")
        assert [p.parent.parent.name for p in paths] == [expected], name

    # empty variants dir
    (tmp_path / "empty" / "variants").mkdir(parents=True)
    assert _find_analysis_csvs(tmp_path / "empty" / "variants", "run_summary.csv") == []


_MERGE_CASES = [
    # (name, (headers, row) first, (headers, row) second, expected headers or None, expected row subsets)
    (
        "uniform_columns",
        (["x", "y"], {"x": "1", "y": "2"}), (["x", "y"], {"x": "3", "y": "4"}),
        ["x", "y"], [{"x": "1"}, {"x": "3"}],
    ),
    (
        "different_columns",
        (["a", "b"], {"a": "1", "b": "2"}), (["b", "c"], {"b": "3", "c": "4"}),
        None, [{"a": "1", "c": ""}, {"a": "", "c": "4"}],
    ),
    (
        "disjoint_columns",
        (["x"], {"x": "1"}), (["y"], {"y": "2"}),
        None, [{"x": "1", "y": ""}, {"x": "", "y": "2"}],
    ),
]


def test_merge_csvs_column_union_empty_list_and_corrupt_file(tmp_path: Path) -> None:
    # column union across inputs
    for name, first, second, expected_headers, expected_rows in _MERGE_CASES:
        root = tmp_path / name
        root.mkdir()
        csv1, csv2 = root / "a.csv", root / "b.csv"
        _write_csv(csv1, first[0], [first[1]])
        _write_csv(csv2, second[0], [second[1]])
        out = root / "merged.csv"
        assert _merge_csvs([(csv1, {}), (csv2, {})], out) == 2, name
        headers, rows = _read_csv(out)
        if expected_headers is not None:
            assert headers == expected_headers, name
        assert set(first[0]) | set(second[0]) <= set(headers), name
        for row, expected in zip(rows, expected_rows):
            for key, value in expected.items():
                assert row[key] == value, name

    # empty input list
    assert _merge_csvs([], tmp_path / "merged.csv") == 0

    # corrupt file is skipped
    good, bad = tmp_path / "good.csv", tmp_path / "bad.csv"
    _write_csv(good, ["a"], [{"a": "1"}])
    bad.write_bytes(b"\x00\x01\x02\xff\xfe")
    assert _merge_csvs([(good, {}), (bad, {})], tmp_path / "merged_corrupt.csv") >= 1


def test_aggregate_variant_csvs_merges_variants_and_handles_missing_dir(tmp_path: Path) -> None:
    # full aggregation across variants with differing columns
    root = tmp_path / "full"
    _variant_csv(
        root, "control_best", "program_1", "run_summary.csv",
        ["evaluation_id", "variant_id", "correct"],
        [{"evaluation_id": "e1", "variant_id": "control_best", "correct": "true"}],
    )
    _variant_csv(root, "control_best", "program_1", "metric_values.csv", *_METRIC_ROW)
    _variant_csv(
        root, "ablation_x", "program_1", "run_summary.csv",
        ["evaluation_id", "variant_id", "extra_col"],
        [{"evaluation_id": "e2", "variant_id": "ablation_x", "extra_col": "foo"}],
    )

    result = aggregate_variant_csvs(root)

    assert "original" in result
    headers, rows = _read_csv(result["original"]["run_summary.csv"])
    assert "task_name" not in headers
    assert {"evaluation_id", "variant_id", "correct", "extra_col"} <= set(headers)
    assert len(rows) == 2
    _, mv_rows = _read_csv(result["original"]["metric_values.csv"])
    assert len(mv_rows) == 1
    assert result["original"]["timeseries.csv"] is None

    # no variants dir
    (tmp_path / "empty").mkdir()
    assert aggregate_variant_csvs(tmp_path / "empty") == {}


def test_aggregate_primary_compatibility_mirror_symlink_replacement_and_per_task_subdirs(tmp_path: Path) -> None:
    # per-task aggregates are written and mirrored at the root for paper-data detection
    root = tmp_path / "mirror"
    _variant_csv(root, "control_best", "program_1", "run_summary.csv", *_EVAL_ROW)
    _variant_csv(root, "control_best", "program_1", "metric_values.csv", *_METRIC_ROW)
    result = aggregate_variant_csvs(root, original_task_name="cifar10")
    aggregated = root / "aggregated"
    assert (aggregated / "run_summary.csv").read_text() == (aggregated / "cifar10" / "run_summary.csv").read_text()
    assert (aggregated / "metric_values.csv").read_text() == (aggregated / "cifar10" / "metric_values.csv").read_text()
    assert has_ablation_bundle(root)
    headers, _ = _read_csv(result["cifar10"]["run_summary.csv"])
    assert "task_name" not in headers

    # a symlink at the compatibility destination is replaced, not followed
    root = tmp_path / "symlink"
    _variant_csv(root, "control_best", "program_1", "run_summary.csv", *_EVAL_ROW)
    aggregated_dir = root / "aggregated"
    aggregated_dir.mkdir()
    symlink_target = root / "outside.csv"
    symlink_target.write_text("do not overwrite\n")
    (aggregated_dir / "run_summary.csv").symlink_to(symlink_target)
    aggregate_variant_csvs(root, original_task_name="cifar10")
    compatibility_path = aggregated_dir / "run_summary.csv"
    assert symlink_target.read_text() == "do not overwrite\n"
    assert not compatibility_path.is_symlink()
    assert compatibility_path.read_text() == (aggregated_dir / "cifar10" / "run_summary.csv").read_text()

    # per-task subdirectories for extra tasks
    root = tmp_path / "per_task"
    _variant_csv(root, "control_best", "program_1", "run_summary.csv", *_EVAL_ROW)
    for task in ("mnist", "fashion"):
        _variant_csv(
            root / "extra_tasks" / task, "control_best", "program_1", "run_summary.csv",
            ["evaluation_id", "variant_id"], [{"evaluation_id": "e2", "variant_id": "control_best"}],
        )
    result = aggregate_variant_csvs(root, original_task_name="cifar10")
    assert set(result) == {"cifar10", "mnist", "fashion"}
    for task_name in ("cifar10", "mnist", "fashion"):
        csv_path = result[task_name]["run_summary.csv"]
        assert csv_path is not None and csv_path.parent.name == task_name
        headers, rows = _read_csv(csv_path)
        assert "task_name" not in headers
        assert len(rows) == 1


_FLAT_LAYOUT_CASES = [
    ("variant_with_csv", {"BP_PPO/run_summary.csv": "csv"}, True),
    ("empty", {}, False),
    ("ignores_underscore_dirs", {"_internal/run_summary.csv": "csv"}, False),
    ("no_csvs", {"BP_PPO/notes.txt": "hello"}, False),
]


def test_is_flat_extra_task_and_find_flat_csvs(tmp_path: Path) -> None:
    for name, layout, expected in _FLAT_LAYOUT_CASES:
        task_dir = tmp_path / name
        task_dir.mkdir()
        for rel, content in layout.items():
            path = task_dir / rel
            if content == "csv":
                _write_csv(path, ["a"], [{"a": "1"}])
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
        assert _is_flat_extra_task(task_dir) is expected, name

    # flat CSV discovery injects ids and skips underscore dirs
    task_dir = tmp_path / "ed_mujoco_ant"
    _write_csv(task_dir / "BP_PPO" / "run_summary.csv", ["score"], [{"score": "100"}])
    _write_csv(task_dir / "variant_x" / "run_summary.csv", ["score"], [{"score": "200"}])
    _write_csv(task_dir / "_internal" / "run_summary.csv", ["a"], [{"a": "1"}])
    entries = _find_flat_csvs(task_dir, "run_summary.csv")
    assert len(entries) == 2
    paths = {e[0].parent.name: e[1] for e in entries}
    assert paths["BP_PPO"] == {"evaluation_id": "BP_PPO", "variant_id": "BP_PPO"}
    assert paths["variant_x"] == {"evaluation_id": "variant_x", "variant_id": "variant_x"}
    assert _find_flat_csvs(task_dir, "timeseries.csv") == []


def test_aggregate_flat_extra_tasks_with_primary_without_primary_and_mixed(tmp_path: Path) -> None:
    # flat extra task alongside a structured primary
    root = tmp_path / "with_primary"
    _variant_csv(
        root, "v1", "program_1", "run_summary.csv",
        ["evaluation_id", "variant_id", "correct"], [{"evaluation_id": "e1", "variant_id": "v1", "correct": "true"}],
    )
    _write_csv(
        root / "extra_tasks" / "mujoco_ant" / "BP_PPO" / "run_summary.csv",
        ["score", "correct"], [{"score": "6000", "correct": "true"}],
    )
    result = aggregate_variant_csvs(root, original_task_name="primary")
    assert "primary" in result and "mujoco_ant" in result
    _, rows = _read_csv(result["mujoco_ant"]["run_summary.csv"])
    assert rows == [{"evaluation_id": "BP_PPO", "variant_id": "BP_PPO", "score": "6000", "correct": "true"}]

    # multiple flat variants without a primary
    root = tmp_path / "without_primary"
    task_dir = root / "extra_tasks" / "mujoco_humanoid"
    _write_csv(task_dir / "BP_PPO" / "run_summary.csv", ["score"], [{"score": "3600"}])
    _write_csv(task_dir / "ED_variant" / "run_summary.csv", ["score"], [{"score": "4200"}])
    result = aggregate_variant_csvs(root)
    assert set(result) == {"mujoco_humanoid"}
    _, rows = _read_csv(result["mujoco_humanoid"]["run_summary.csv"])
    by_variant = {r["variant_id"]: r for r in rows}
    assert by_variant["BP_PPO"]["score"] == "3600"
    assert by_variant["ED_variant"]["score"] == "4200"

    # mixed structured and flat extra tasks
    root = tmp_path / "mixed"
    _variant_csv(
        root / "extra_tasks" / "mnist", "v1", "program_1", "run_summary.csv",
        ["evaluation_id", "variant_id"], [{"evaluation_id": "e1", "variant_id": "v1"}],
    )
    _write_csv(root / "extra_tasks" / "mujoco_ant" / "BP_PPO" / "run_summary.csv", ["score"], [{"score": "6000"}])
    result = aggregate_variant_csvs(root)
    assert "mnist" in result and "mujoco_ant" in result
    assert result["mujoco_ant"]["run_summary.csv"] is not None


# ---------------------------------------------------------------------------
# readme
# ---------------------------------------------------------------------------


def _make_spec(variant_id="v1", title="Test Variant", kind="ablation", hypothesis="Tests the hypothesis.") -> VariantSpec:
    return VariantSpec(
        variant_id=variant_id, title=title, kind=kind, base_program="best",
        variant_dir=Path("/tmp/fake"), hypothesis=hypothesis,
    )


_CTRL = [_make_spec("ctrl", "Control", "control")]
_CTRL_SUMMARY = {"ctrl": "The control."}
_TASKS_WITH_EXTRA = [
    {"name": "ed_conv", "data_path": "variants/", "goal": "Classify CIFAR-10.", "role": "primary"},
    {"name": "ed_mnist", "data_path": "extra_tasks/ed_mnist/variants/", "goal": "Classify MNIST.", "role": "cross-task"},
]


def _ctrl_readme(*args, **kwargs) -> str:
    return _assemble_readme("Task.", _CTRL, _CTRL_SUMMARY, {}, *args, **kwargs)


def test_assemble_readme_structure_kind_ordering_status_and_separators():
    # basic structure
    specs = [_make_spec("ctrl", "Control", "control", "The control.")]
    status_map = {"ctrl": {"status": "completed", "rubric_passed": True}}
    readme = _assemble_readme("Task summary.", specs, {"ctrl": "This is the control program."}, status_map)
    for needle in (
        "# Variants Experiment Overview", "Task summary.", "## Variants", "### Control (`ctrl`)",
        "**Kind**: control", "**Status**: completed", "**Rubric**: passed", "This is the control program.",
        "**Workspace**: `variants/ctrl/`", "## Directory Guide", "| Path | Contents |",
        "aggregated/{task}/run_summary.csv", "aggregated/{task}/metric_values.csv",
        "aggregated/{task}/timeseries.csv", "pipeline_summary.json", "variant_spec.json",
    ):
        assert needle in readme
    assert "## Evaluated Tasks" not in readme
    assert "extra_tasks/{task}/{variant}/" not in readme

    # kinds are ordered control, baseline, ablation
    specs = [_make_spec("abl", "Ablation", "ablation"), _make_spec("ctrl", "Control", "control"),
             _make_spec("base", "Baseline", "baseline")]
    readme = _assemble_readme("Task.", specs, {s.variant_id: f"Summary for {s.variant_id}" for s in specs}, {})
    assert readme.index("### Control (`ctrl`)") < readme.index("### Baseline (`base`)") < readme.index("### Ablation (`abl`)")

    # missing status defaults to unknown; a failed rubric is reported
    for name, status_map, status_line in (
        ("missing_status_defaults", {}, "**Status**: unknown"),
        ("rubric_failed", {"v1": {"status": "failed", "rubric_passed": False}}, "**Status**: failed"),
    ):
        readme = _assemble_readme("Task.", [_make_spec("v1", "V1", "ablation")], {"v1": "Summary."}, status_map)
        assert status_line in readme, name
        assert "**Rubric**: failed" in readme, name

    # no separator after the last variant
    readme = _assemble_readme("Task.", [_make_spec("v1", "V1", "ablation")], {"v1": "S."}, {})
    lines = readme.split("\n")
    guide_idx = next(i for i, line in enumerate(lines) if "## Directory Guide" in line)
    workspace_idx = next(i for i in range(len(lines) - 1, -1, -1) if "**Workspace**" in lines[i])
    assert "---" not in lines[workspace_idx + 1 : guide_idx]


def test_assemble_readme_evaluated_tasks_section():
    # with an extra cross-task
    readme = _ctrl_readme(_TASKS_WITH_EXTRA)
    for needle in ("## Evaluated Tasks", "### ed_conv (primary)", "Classify CIFAR-10.", "### ed_mnist (cross-task)",
                   "Classify MNIST.", "extra_tasks/{task}/variants/{id}/"):
        assert needle in readme

    # primary-only task info omits the extra_tasks guide row
    readme = _ctrl_readme([{"name": "ed_conv", "data_path": "variants/", "goal": "", "role": "primary"}])
    assert "extra_tasks/{task}" not in readme

    # flat tasks are labelled "extra"
    task_info = [
        {"name": "ed_conv", "data_path": "variants/", "goal": "", "role": "primary"},
        {"name": "ed_mujoco_humanoid", "data_path": "extra_tasks/ed_mujoco_humanoid/", "goal": "Train Humanoid.", "role": "flat"},
    ]
    readme = _ctrl_readme(task_info)
    assert "### ed_mujoco_humanoid (extra)" in readme
    assert "Train Humanoid." in readme


def test_assemble_readme_flat_task_sections_ordering_fallback_and_sorting():
    # flat task section is rendered with workspaces and guide row
    flat_tasks = {
        "ed_mujoco_humanoid": [
            {"variant_id": "BP_PPO", "title": "Backprop PPO Baseline", "kind": "baseline", "hypothesis": "Standard PPO baseline."},
            {"variant_id": "ED_PPO", "title": "Error Diffusion + PPO", "kind": "control", "hypothesis": "ED combined with PPO."},
        ],
    }
    readme = _ctrl_readme(flat_tasks=flat_tasks)
    for needle in (
        "## Extra Task: ed_mujoco_humanoid", "### Error Diffusion + PPO (`ED_PPO`)", "**Kind**: control",
        "ED combined with PPO.", "### Backprop PPO Baseline (`BP_PPO`)",
        "`extra_tasks/ed_mujoco_humanoid/BP_PPO/`", "`extra_tasks/ed_mujoco_humanoid/ED_PPO/`",
        "extra_tasks/{task}/{variant}/",
    ):
        assert needle in readme

    # kind ordering within a flat task
    flat_tasks = {
        "ed_mujoco_ant": [
            {"variant_id": "ES", "title": "ES", "kind": "baseline", "hypothesis": "ES."},
            {"variant_id": "ED_PPO", "title": "ED", "kind": "control", "hypothesis": "ED."},
            {"variant_id": "BP_PPO", "title": "BP", "kind": "baseline", "hypothesis": "BP."},
        ],
    }
    readme = _ctrl_readme(flat_tasks=flat_tasks)
    assert readme.index("### ED (`ED_PPO`)") < readme.index("### BP (`BP_PPO`)") < readme.index("### ES (`ES`)")

    # fallback hypothesis when none is given
    readme = _ctrl_readme(flat_tasks={"ed_craftax": [{"variant_id": "BP_PPO", "title": "BP PPO", "kind": "baseline"}]})
    assert "BP PPO (baseline variant)." in readme

    # multiple flat tasks are sorted by name
    flat_tasks = {
        "ed_mujoco_ant": [{"variant_id": "BP_PPO", "title": "BP", "kind": "baseline", "hypothesis": "BP ant."}],
        "ed_mujoco_humanoid": [{"variant_id": "BP_PPO", "title": "BP", "kind": "baseline", "hypothesis": "BP humanoid."}],
    }
    readme = _ctrl_readme(flat_tasks=flat_tasks)
    assert "## Extra Task: ed_mujoco_ant" in readme
    assert "## Extra Task: ed_mujoco_humanoid" in readme
    assert readme.index("ed_mujoco_ant") < readme.index("ed_mujoco_humanoid")


def test_fallback_summary_and_fallback_task_summary(tmp_path: Path):
    # variant fallback summary
    assert _fallback_summary(_make_spec(hypothesis="Tests whether X improves Y.")) == "Tests whether X improves Y."
    result = _fallback_summary(_make_spec(hypothesis=""))
    assert "Test Variant" in result and "ablation" in result

    # task fallback summary: first paragraph / missing / empty task.md
    for name, content, expected in (
        ("first_paragraph", "# My Task\nOptimize the thing.\n\nMore details.", "Optimize the thing."),
        ("missing_task_md", None, "Variants experiment."),
        ("empty_task_md", "", "Variants experiment."),
    ):
        task_dir = tmp_path / name
        task_dir.mkdir()
        if content is not None:
            (task_dir / "task.md").write_text(content)
        assert _fallback_task_summary(task_dir) == expected, name


def test_parse_agent_summaries_valid_and_invalid(tmp_path: Path):
    # valid
    valid = tmp_path / "valid"
    valid.mkdir()
    data = {"task_summary": "Optimize X.", "variant_summaries": {"v1": "Summary for v1."}}
    (valid / "variant_summaries.json").write_text(json.dumps(data))
    result = _parse_agent_summaries(valid)
    assert result is not None
    assert result["task_summary"] == "Optimize X."
    assert result["variant_summaries"]["v1"] == "Summary for v1."

    # missing / malformed / non-dict
    for name, content in (("missing", None), ("malformed", "not json{"), ("non_dict", "[1,2,3]")):
        root = tmp_path / name
        root.mkdir()
        if content is not None:
            (root / "variant_summaries.json").write_text(content)
        assert _parse_agent_summaries(root) is None, name


def test_build_readme_prompt_with_and_without_task_md(tmp_path: Path):
    with_task = tmp_path / "with_task"
    with_task.mkdir()
    (with_task / "task.md").write_text("# Task\nOptimize accuracy on CIFAR-10.")
    prompt = _build_readme_prompt([_make_spec("v1", "Ablation No X", "ablation", "Remove X.")], with_task)
    for needle in ("Optimize accuracy on CIFAR-10", "v1", "Ablation No X", "Remove X."):
        assert needle in prompt

    without_task = tmp_path / "without_task"
    without_task.mkdir()
    assert "variant_summaries.json" in _build_readme_prompt([_make_spec()], without_task)


def test_build_status_map_valid_and_empty():
    summary = {"variants": [
        {"variant_id": "v1", "status": "completed", "rubric_passed": True},
        {"variant_id": "v2", "status": "failed", "rubric_passed": False},
    ]}
    result = _build_status_map(summary)
    assert result["v1"]["status"] == "completed"
    assert result["v1"]["rubric_passed"] is True
    assert result["v2"]["status"] == "failed"
    for name, summary in (("none", None), ("missing_key", {"total_variants": 0}), ("non_list", {"variants": "bad"})):
        assert _build_status_map(summary) == {}, name


def _dirs(root: Path, task_name: str = "task") -> tuple[Path, Path]:
    task_dir = root / task_name
    task_dir.mkdir(parents=True)
    output_dir = root / "output"
    output_dir.mkdir()
    return task_dir, output_dir


def test_discover_tasks_explicit_name_primary_only_extra_and_flat(tmp_path: Path):
    # explicit primary name
    task_dir, output_dir = _dirs(tmp_path / "explicit")
    assert _discover_tasks(output_dir, task_dir, primary_task_name="ed_conv")[0]["name"] == "ed_conv"

    # primary only, goal parsed from task.md
    task_dir, output_dir = _dirs(tmp_path / "primary", "ed_conv")
    (task_dir / "task.md").write_text("# Goal\n- Optimize accuracy on CIFAR-10.\n")
    tasks = _discover_tasks(output_dir, task_dir)
    assert len(tasks) == 1
    assert tasks[0]["name"] == "ed_conv"
    assert tasks[0]["data_path"] == "variants/"
    assert tasks[0]["role"] == "primary"
    assert tasks[0]["goal"] == "Optimize accuracy on CIFAR-10."

    # structured extra tasks, sorted, incomplete ones skipped
    task_dir, output_dir = _dirs(tmp_path / "extra", "ed_conv")
    mnist_dir = output_dir / "extra_tasks" / "ed_mnist"
    (mnist_dir / "variants").mkdir(parents=True)
    (mnist_dir / "_retrofitted_bundle").mkdir()
    (mnist_dir / "_retrofitted_bundle" / "task.md").write_text("## Goal\n- Classify MNIST digits.\n")
    (output_dir / "extra_tasks" / "ed_fashion" / "variants").mkdir(parents=True)
    (output_dir / "extra_tasks" / "incomplete").mkdir()  # no variants/ and no variant subdirs: skipped
    tasks = _discover_tasks(output_dir, task_dir)
    assert [t["name"] for t in tasks] == ["ed_conv", "ed_fashion", "ed_mnist"]
    assert tasks[1]["data_path"] == "extra_tasks/ed_fashion/variants/"
    assert tasks[1]["role"] == "cross-task"
    assert tasks[2]["goal"] == "Classify MNIST digits."

    # flat and structured extra tasks together
    task_dir, output_dir = _dirs(tmp_path / "flat")
    (output_dir / "extra_tasks" / "ed_mnist" / "variants").mkdir(parents=True)
    flat = output_dir / "extra_tasks" / "ed_mujoco_humanoid"
    (flat / "BP_PPO").mkdir(parents=True)
    (flat / "task.md").write_text("## Goal\n- Train Humanoid-v4.\n")
    (flat / "BP_PPO" / "variant_spec.json").write_text('{"variant_id": "BP_PPO"}')
    (flat / "BP_PPO" / "run_summary.csv").write_text("col1\nval1\n")
    tasks = _discover_tasks(output_dir, task_dir)
    assert [t["role"] for t in tasks] == ["primary", "cross-task", "flat"]
    flat_task = tasks[2]
    assert flat_task["name"] == "ed_mujoco_humanoid"
    assert flat_task["data_path"] == "extra_tasks/ed_mujoco_humanoid/"
    assert flat_task["goal"] == "Train Humanoid-v4."


_README_FLAT_LAYOUT_CASES = [
    ("variant_spec", {"BP_PPO/variant_spec.json": '{"variant_id": "BP_PPO"}'}, True),
    ("csv", {"BP_PPO/run_summary.csv": "col\nval\n"}, True),
    ("empty", {}, False),
    ("only_files", {"task.md": "# Task"}, False),
    ("ignores_underscore_dirs", {"_internal/variant_spec.json": '{"variant_id": "x"}'}, False),
    ("subdir_without_csv_or_spec", {"BP_PPO/notes.txt": "just notes"}, False),
]


def test_readme_is_flat_extra_task(tmp_path: Path):
    for name, layout, expected in _README_FLAT_LAYOUT_CASES:
        root = tmp_path / name
        root.mkdir()
        for rel, content in layout.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(content)
        assert readme_mod._is_flat_extra_task(root) is expected, name


_FLAT_SPEC_SKIP_CASES = [
    ("underscore_dir", "_internal/variant_spec.json", '{"variant_id": "x"}'),
    ("missing_spec", "BP_PPO/run_summary.csv", "col\nval\n"),
    ("malformed_json", "BP_PPO/variant_spec.json", "not json{"),
    ("non_dict_json", "BP_PPO/variant_spec.json", "[1,2,3]"),
]


def test_load_flat_variant_specs_reads_valid_and_skips_invalid(tmp_path: Path):
    valid = tmp_path / "valid"
    for name in ["BP_PPO", "ED_PPO"]:
        (valid / name).mkdir(parents=True)
        (valid / name / "variant_spec.json").write_text(json.dumps({"variant_id": name, "title": f"{name} title"}))
    specs = _load_flat_variant_specs(valid)
    assert {s["variant_id"] for s in specs} == {"BP_PPO", "ED_PPO"}
    assert len(specs) == 2

    for name, rel, content in _FLAT_SPEC_SKIP_CASES:
        root = tmp_path / name
        (root / rel).parent.mkdir(parents=True)
        (root / rel).write_text(content)
        assert _load_flat_variant_specs(root) == [], name
