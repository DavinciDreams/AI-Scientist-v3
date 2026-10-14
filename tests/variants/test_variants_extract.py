"""Tests for variants_pipeline extract (bundle provenance), custom_baselines and select."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from src.variants_pipeline.custom_baselines import (
    _load_folder_baseline,
    _parse_baseline_markdown,
    _prettify_name,
    _sanitize_folder_name,
    load_custom_baselines,
)
from src.variants_pipeline.extract import BundleValidationError, _apply_baseline_override, extract_bundle
from src.variants_pipeline.models import VariantCandidate
from src.variants_pipeline.select import _apply_hard_caps

_SPEC_TOML = (
    'schema_version = 1\ncode_root = "src"\n\n'
    '[entrypoint]\npath = "program.py"\nlanguage = "python"\ntype = "script"\n\n'
    "[copy]\nexclude = []\ninclude = []\n\n"
    '[integration]\nkind = "generic"\n'
)


# ---------------------------------------------------------------------------
# extract helpers
# ---------------------------------------------------------------------------


def _create_program_dir(path: Path, entrypoint_content: str = "print('hello')\n") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "program_spec.toml").write_text(_SPEC_TOML)
    (path / "src").mkdir(exist_ok=True)
    (path / "src" / "program.py").write_text(entrypoint_content)


def _create_minimal_bundle(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _create_program_dir(bundle_dir / "program_0", "print('best')\n")
    _create_program_dir(bundle_dir / "reference_program", "print('seed')\n")
    (bundle_dir / "evaluate.py").write_text("# evaluator\n")
    (bundle_dir / "task.md").write_text("# Task\n")
    (bundle_dir / "extraction_manifest.json").write_text("{}")


def _write_best_manifest(results_dir: Path, source_program: Path, score: int) -> None:
    (results_dir / "best_program_manifest.json").write_text(
        json.dumps(
            {"program_dir": str(source_program.resolve()), "entrypoint_path": "src/program.py", "combined_score": score}
        ),
        encoding="utf-8",
    )


def _create_extraction_sources(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    task_dir = root / "task"
    task_dir.mkdir()
    _create_program_dir(task_dir / "program_0", "print('seed')\n")
    (task_dir / "evaluate.py").write_text("# evaluator\n", encoding="utf-8")
    (task_dir / "task.md").write_text("# Task\n", encoding="utf-8")
    results_dir = root / "results"
    source_program = results_dir / "best-program"
    _create_program_dir(source_program, "print('best')\n")
    _write_best_manifest(results_dir, source_program, 1)
    return task_dir, results_dir, source_program


def _sources(root: Path):
    """(task_dir, results_dir, source_program, extract_bundle kwargs) built under ``root``."""
    task_dir, results_dir, source_program = _create_extraction_sources(root)
    kwargs = {"task_dir": task_dir, "results_dir": results_dir, "output_dir": root / "output"}
    return task_dir, results_dir, source_program, kwargs


def _fake_extract(*, task_dir: Path, results_dir: Path, output_dir: Path) -> None:
    _create_minimal_bundle(output_dir)
    (output_dir / "extraction_manifest.json").write_text(
        json.dumps(
            {
                "source_task_dir": str(task_dir.resolve()),
                "source_results_dir": str(results_dir.resolve()),
                "source_best_program_dir": str((results_dir / "best-program").resolve()),
                "entrypoint_path": "src/program.py",
                "combined_score": 1,
                "correct": None,
                "source_attempt_id": None,
                "parallel_worker_index": None,
            }
        ),
        encoding="utf-8",
    )


def _create_evaluated_program(parent: Path, name: str, *, content: str, score: float) -> Path:
    program_dir = parent / name
    _create_program_dir(program_dir, content)
    (program_dir / "metrics.json").write_text(json.dumps({"combined_score": score}), encoding="utf-8")
    (program_dir / "correct.json").write_text(json.dumps({"correct": True}), encoding="utf-8")
    return program_dir


@pytest.fixture()
def mock_extract():
    with patch("src.variants_pipeline.extract.extract_best_program_bundle", side_effect=_fake_extract) as mock:
        yield mock


# ---------------------------------------------------------------------------
# _apply_baseline_override
# ---------------------------------------------------------------------------


def test_apply_baseline_override_replaces_reference_creates_missing_and_rejects_nonexistent(tmp_path: Path):
    override_dir = tmp_path / "override_program"
    _create_program_dir(override_dir, "print('custom baseline')\n")
    (override_dir / "src" / "utils.py").write_text("# utilities\n")

    # replaces the reference program and preserves the rest of the bundle
    bundle_dir = tmp_path / "bundle_replace"
    _create_minimal_bundle(bundle_dir)
    _apply_baseline_override(bundle_dir, override_dir)
    ref = bundle_dir / "reference_program" / "src"
    assert (ref / "program.py").read_text() == "print('custom baseline')\n"
    assert (ref / "utils.py").read_text() == "# utilities\n"
    assert (bundle_dir / "evaluate.py").read_text() == "# evaluator\n"
    assert (bundle_dir / "task.md").read_text() == "# Task\n"
    assert (bundle_dir / "program_0" / "src" / "program.py").read_text() == "print('best')\n"

    # works when there is no prior reference program
    bundle_dir = tmp_path / "bundle_no_reference"
    _create_minimal_bundle(bundle_dir)
    shutil.rmtree(bundle_dir / "reference_program")
    _apply_baseline_override(bundle_dir, override_dir)
    assert (bundle_dir / "reference_program" / "src" / "program.py").read_text() == "print('custom baseline')\n"

    # raises on a nonexistent override dir
    bundle_dir = tmp_path / "bundle_missing_override"
    _create_minimal_bundle(bundle_dir)
    with pytest.raises(BundleValidationError):
        _apply_baseline_override(bundle_dir, tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# extract_bundle
# ---------------------------------------------------------------------------


def test_extract_applies_override_only_when_requested(tmp_path: Path, mock_extract):
    # override is applied after extraction
    _, _, _, kwargs = _sources(tmp_path / "with_override")
    override = tmp_path / "override_program"
    _create_program_dir(override, "print('override baseline')\n")
    result = extract_bundle(**kwargs, baseline_program_override=override)
    assert (result / "reference_program" / "src" / "program.py").read_text() == "print('override baseline')\n"

    # no override by default; manifest records both program dirs
    _, _, _, kwargs = _sources(tmp_path / "default")
    result = extract_bundle(**kwargs)
    assert (result / "reference_program" / "src" / "program.py").read_text() == "print('seed')\n"
    manifest = json.loads((result / "extraction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reference_program_dir"] == str(result / "reference_program")
    assert manifest["extracted_program_dir"] == str(result / "program_0")
    assert Path(manifest["reference_program_dir"]).is_dir()
    assert Path(manifest["extracted_program_dir"]).is_dir()


def test_extract_reuses_complete_bundle_and_ignores_runtime_manifest(tmp_path: Path, mock_extract):
    task_dir, _, _, kwargs = _sources(tmp_path)
    bundle_dir = extract_bundle(**kwargs)
    # reuses a complete bundle with matching provenance
    assert extract_bundle(**kwargs) == bundle_dir
    assert mock_extract.call_count == 1
    # a runtime manifest in the task dir does not invalidate the existing bundle
    (task_dir / ".agentic_runtime.json").write_text('{"schema_version": 1}', encoding="utf-8")
    assert extract_bundle(**kwargs) == bundle_dir
    assert mock_extract.call_count == 1
    assert not (bundle_dir / ".agentic_runtime.json").exists()


def test_extract_rejects_reuse_after_source_changes(tmp_path: Path, mock_extract):
    # best manifest changes during extraction
    _, _, source_program, kwargs = _sources(tmp_path / "during")

    def change_selection_during_extract(*, task_dir: Path, results_dir: Path, output_dir: Path) -> None:
        _fake_extract(task_dir=task_dir, results_dir=results_dir, output_dir=output_dir)
        _write_best_manifest(results_dir, source_program, 2)

    mock_extract.side_effect = change_selection_during_extract
    with pytest.raises(BundleValidationError, match="changed during"):
        extract_bundle(**kwargs)
    mock_extract.side_effect = _fake_extract

    # best manifest changes after extraction
    _, results_dir, source_program, kwargs = _sources(tmp_path / "after_manifest")
    extract_bundle(**kwargs)
    _write_best_manifest(results_dir, source_program, 2)
    with pytest.raises(BundleValidationError, match="source_best_manifest"):
        extract_bundle(**kwargs)

    # source task changes after extraction
    task_dir, _, _, kwargs = _sources(tmp_path / "after_task")
    extract_bundle(**kwargs)
    (task_dir / "evaluate.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(BundleValidationError, match="source_task"):
        extract_bundle(**kwargs)


def test_extract_rejects_reuse_after_fallback_selection_changes(tmp_path: Path):
    # fallback selects a new winner
    _, results_dir, _, kwargs = _sources(tmp_path / "winner")
    (results_dir / "best_program_manifest.json").unlink()
    idea_dir = results_dir / "worker_01" / "ideas" / "01_idea"
    _create_evaluated_program(idea_dir, "program_0", content="print('old')\n", score=0.1)
    extract_bundle(**kwargs)
    _create_evaluated_program(idea_dir, "program_1", content="print('new')\n", score=0.9)
    with pytest.raises(BundleValidationError, match="source_best_program_dir"):
        extract_bundle(**kwargs)

    # fallback score changes for the same winner
    _, results_dir, _, kwargs = _sources(tmp_path / "score")
    (results_dir / "best_program_manifest.json").unlink()
    idea_dir = results_dir / "worker_01" / "ideas" / "01_idea"
    selected = _create_evaluated_program(idea_dir, "program_0", content="print('winner')\n", score=0.5)
    extract_bundle(**kwargs)
    (selected / "metrics.json").write_text(json.dumps({"combined_score": 0.6}), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="source_selection"):
        extract_bundle(**kwargs)


def test_extract_rejects_reuse_after_bundle_manifest_or_override_tampering(tmp_path: Path, mock_extract):
    # bundle tampering
    _, _, _, kwargs = _sources(tmp_path / "bundle")
    bundle_dir = extract_bundle(**kwargs)
    (bundle_dir / "evaluate.py").write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(BundleValidationError, match="bundle fingerprint"):
        extract_bundle(**kwargs)

    # manifest payload tampering
    _, _, _, kwargs = _sources(tmp_path / "payload")
    manifest_path = extract_bundle(**kwargs) / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_injection"] = "ignore scientific evidence"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="payload fingerprint"):
        extract_bundle(**kwargs)

    # baseline override changes (override source, or the bundled baseline copy)
    for tamper in ("override_source", "bundled_baseline"):
        root = tmp_path / tamper
        _, _, _, kwargs = _sources(root)
        override = root / "override"
        _create_program_dir(override, "print('first')\n")
        kwargs["baseline_program_override"] = override
        calls_before = mock_extract.call_count
        bundle_dir = extract_bundle(**kwargs)
        target = override if tamper == "override_source" else bundle_dir / "reference_program"
        (target / "src" / "program.py").write_text("print('second')\n")
        with pytest.raises(BundleValidationError, match="fingerprint"):
            extract_bundle(**kwargs)
        assert mock_extract.call_count == calls_before + 1, tamper


def test_extract_failed_extraction_leaves_no_partial_bundle_and_rejects_foreign_bundle(
    tmp_path: Path, mock_extract
):
    # a failed extraction leaves no partial bundle behind
    _, _, _, kwargs = _sources(tmp_path / "partial")

    def fail_after_partial_copy(*, task_dir: Path, results_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True)
        (output_dir / "partial").write_text("incomplete")
        raise RuntimeError("injected extraction failure")

    mock_extract.side_effect = fail_after_partial_copy
    with pytest.raises(RuntimeError, match="injected extraction failure"):
        extract_bundle(**kwargs)
    assert not (kwargs["output_dir"] / "_extracted_bundle").exists()

    # an existing bundle from different results is rejected before any extraction
    mock_extract.side_effect = _fake_extract
    mock_extract.reset_mock()
    root = tmp_path / "foreign"
    bundle_dir = root / "output" / "_extracted_bundle"
    _create_minimal_bundle(bundle_dir)
    (bundle_dir / "extraction_manifest.json").write_text(
        json.dumps(
            {
                "source_task_dir": str((root / "task").resolve()),
                "source_results_dir": str((root / "other-results").resolve()),
            }
        )
    )
    with pytest.raises(BundleValidationError, match="source_results_dir"):
        extract_bundle(task_dir=root / "task", results_dir=root / "results", output_dir=root / "output")
    mock_extract.assert_not_called()


# ---------------------------------------------------------------------------
# custom_baselines
# ---------------------------------------------------------------------------


def _write_md(path: Path, frontmatter: str, body: str = "") -> Path:
    path.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")
    return path


def _make_folder_baseline(folder: Path, **extra_files: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "src").mkdir(exist_ok=True)
    (folder / "src" / "program.py").write_text("print('hello')\n")
    for name, content in extra_files.items():
        (folder / name).write_text(content, encoding="utf-8")
    return folder


def test_parse_baseline_markdown_minimal_full_body_and_prefix(tmp_path: Path):
    # minimal frontmatter
    c = _parse_baseline_markdown(_write_md(tmp_path / "minimal.md", "variant_id: baseline_dfa"))
    assert c.variant_id == "baseline_dfa"
    assert c.title == "baseline_dfa"
    assert c.kind == "baseline"
    assert c.base_program == "best"
    assert c.hypothesis == "" and c.expected_effect == ""
    assert c.target_components == [] and c.implementation_plan == []
    assert c.description_body == ""

    # full frontmatter
    fm = (
        "variant_id: baseline_dfa\ntitle: Direct Feedback Alignment\nkind: baseline\nbase_program: best\n"
        "hypothesis: DFA uses random feedback\nexpected_effect: Lower accuracy\n"
        "relevance_reason: Tests feedback alignment\nfeasibility_reason: Simple to implement\n"
        "target_components:\n  - backward_pass\n  - weight_updates\n"
        "implementation_plan:\n  - Replace backward pass\n  - Keep forward pass\n"
    )
    c = _parse_baseline_markdown(_write_md(tmp_path / "full.md", fm, "\n## Description\nDFA is great.\n"))
    assert c.variant_id == "baseline_dfa"
    assert c.title == "Direct Feedback Alignment"
    assert c.hypothesis == "DFA uses random feedback"
    assert c.target_components == ["backward_pass", "weight_updates"]
    assert c.implementation_plan == ["Replace backward pass", "Keep forward pass"]
    assert c.description_body == "## Description\nDFA is great."

    # description body is the markdown after the frontmatter
    body = "\n# My Baseline\n\nSome detailed description.\n\n## Section\nMore text.\n"
    c = _parse_baseline_markdown(_write_md(tmp_path / "body.md", "variant_id: baseline_test", body))
    assert "# My Baseline" in c.description_body
    assert "## Section" in c.description_body

    # baseline_ prefix is added when missing and kept when present
    for raw in ("dfa", "baseline_dfa"):
        md = _write_md(tmp_path / f"prefix_{raw}.md", f"variant_id: {raw}")
        assert _parse_baseline_markdown(md).variant_id == "baseline_dfa", raw


_BAD_VARIANT_ID_CASES = [
    ("no_variant_id", "title: Something", "Missing required 'variant_id'"),
    ("empty_variant_id", "variant_id: ", "Missing required 'variant_id'"),
    ("special_characters", "variant_id: baseline-dfa!", "Invalid variant_id"),
    ("starts_with_number", "variant_id: 123abc", "Invalid variant_id"),
    ("uppercase", "variant_id: Baseline_DFA", "Invalid variant_id"),
]
_MISSING_FRONTMATTER_CASES = [
    ("no_delimiters", "Just some markdown without frontmatter.\n", "does not contain YAML frontmatter"),
    ("no_closing", "---\nvariant_id: baseline_dfa\nno closing\n", "no closing '---' delimiter"),
]


def test_parse_baseline_markdown_rejects_bad_variant_id_and_missing_frontmatter(tmp_path: Path):
    for name, frontmatter, match in _BAD_VARIANT_ID_CASES:
        with pytest.raises(ValueError, match=match):
            _parse_baseline_markdown(_write_md(tmp_path / f"{name}.md", frontmatter))
    for name, content, match in _MISSING_FRONTMATTER_CASES:
        md = tmp_path / f"{name}.md"
        md.write_text(content)
        with pytest.raises(ValueError, match=match):
            _parse_baseline_markdown(md)


def test_load_custom_baselines_files_duplicates_and_missing_paths(tmp_path: Path):
    # two files
    f1 = _write_md(tmp_path / "a.md", "variant_id: baseline_dfa")
    f2 = _write_md(tmp_path / "b.md", "variant_id: baseline_backprop")
    candidates = load_custom_baselines([f1, f2])
    assert {c.variant_id for c in candidates} == {"baseline_dfa", "baseline_backprop"}
    assert len(candidates) == 2

    # duplicate variant_id across files
    f2 = _write_md(tmp_path / "b_dup.md", "variant_id: baseline_dfa")
    with pytest.raises(ValueError, match="Duplicate variant_id"):
        load_custom_baselines([f1, f2])

    # nonexistent file / folder
    for name in ("does_not_exist.md", "does_not_exist"):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_custom_baselines([tmp_path / name])


def test_sanitize_folder_name_and_prettify_name():
    for raw, expected in (
        ("my-baseline", "my_baseline"),
        ("my baseline", "my_baseline"),
        ("DFA_Baseline", "dfa_baseline"),
        ("baseline@v2!", "baselinev2"),
        ("a--b__c", "a_b_c"),
    ):
        assert _sanitize_folder_name(raw) == expected, raw
    for raw in ("direct_feedback_alignment", "direct-feedback-alignment"):
        assert _prettify_name(raw) == "Direct Feedback Alignment", raw


def test_load_folder_baseline_code_only_readme_baseline_md_and_naming(tmp_path: Path):
    # code only
    folder = _make_folder_baseline(tmp_path / "code_only" / "dfa")
    c = _load_folder_baseline(folder)
    assert c.variant_id == "baseline_dfa"
    assert c.kind == "baseline"
    assert c.base_program == "reference"
    assert c.reference_dir == folder
    assert c.description_body == ""

    # README becomes the description body
    folder = _make_folder_baseline(tmp_path / "readme" / "dfa", **{"README.md": "# DFA\nA reference implementation.\n"})
    c = _load_folder_baseline(folder)
    assert "# DFA" in c.description_body
    assert "A reference implementation." in c.description_body

    # baseline.md overrides metadata but base_program stays reference
    baseline_md = (
        "---\nvariant_id: baseline_custom_dfa\ntitle: Custom DFA Title\n"
        "hypothesis: DFA uses random feedback\n---\n\n## Override description\n"
    )
    folder = _make_folder_baseline(tmp_path / "override" / "dfa", **{"baseline.md": baseline_md})
    c = _load_folder_baseline(folder)
    assert c.variant_id == "baseline_custom_dfa"
    assert c.title == "Custom DFA Title"
    assert c.hypothesis == "DFA uses random feedback"
    assert c.base_program == "reference"  # stays reference even if baseline.md says best
    assert c.reference_dir == folder
    assert "Override description" in c.description_body

    # naming: auto prefix, already prefixed, title prettified
    for folder_name, variant_id, title in (
        ("dfa_algorithm", "baseline_dfa_algorithm", "Dfa Algorithm"),
        ("baseline_dfa", "baseline_dfa", "Baseline Dfa"),
        ("direct_feedback_alignment", "baseline_direct_feedback_alignment", "Direct Feedback Alignment"),
    ):
        c = _load_folder_baseline(_make_folder_baseline(tmp_path / "naming" / folder_name))
        assert c.variant_id == variant_id, folder_name
        assert c.title == title, folder_name


def test_load_custom_baselines_mixed_files_and_folders_and_duplicates(tmp_path: Path):
    # mixed files and folders
    md_file = _write_md(tmp_path / "bp.md", "variant_id: baseline_backprop")
    folder = _make_folder_baseline(tmp_path / "dfa")
    candidates = load_custom_baselines([md_file, folder])
    assert len(candidates) == 2
    assert {c.variant_id for c in candidates} == {"baseline_backprop", "baseline_dfa"}
    folder_candidates = [c for c in candidates if c.reference_dir is not None]
    assert [c.variant_id for c in folder_candidates] == ["baseline_dfa"]

    # duplicate ids across a file and a folder
    md_file = _write_md(tmp_path / "a.md", "variant_id: baseline_dfa")
    with pytest.raises(ValueError, match="Duplicate variant_id"):
        load_custom_baselines([md_file, folder])


# ---------------------------------------------------------------------------
# select._apply_hard_caps
# ---------------------------------------------------------------------------


def _make_candidate(variant_id: str, kind: str = "ablation") -> VariantCandidate:
    return VariantCandidate(
        variant_id=variant_id, title=f"Test {variant_id}", kind=kind, base_program="best",
        hypothesis="test", relevance_reason="test", feasibility_reason="test",
        target_components=["comp"], implementation_plan=["step"], expected_effect="test",
    )


def _caps(candidates, decisions=None):
    if decisions is None:
        decisions = {c.variant_id: (True, "ok") for c in candidates}
    return _apply_hard_caps(candidates=candidates, llm_decisions=decisions, max_ablations=3, max_extra_baselines=3)


def test_apply_hard_caps_zero_all_accepted_llm_rejection_per_kind_and_mixed():
    # zero candidates
    sel = _caps([], {})
    assert sel.accepted == [] and sel.rejected == []
    assert "control_best" in sel.controls

    # all accepted
    sel = _caps([_make_candidate("a1"), _make_candidate("a2")])
    assert len(sel.accepted) == 2 and len(sel.rejected) == 0

    # LLM rejection is carried through with its reason
    sel = _caps([_make_candidate("a1"), _make_candidate("a2")], {"a1": (True, "good"), "a2": (False, "not relevant")})
    assert len(sel.accepted) == 1 and len(sel.rejected) == 1
    assert sel.rejected[0][1] == "not relevant"

    # per-kind cap
    for kind in ("ablation", "baseline"):
        sel = _caps([_make_candidate(f"{kind[0]}{i}", kind) for i in range(5)])
        assert len(sel.accepted) == 3 and len(sel.rejected) == 2, kind
        if kind == "ablation":
            assert all("cap" in reason for _, reason in sel.rejected)

    # mixed kinds are capped independently
    candidates = [_make_candidate(f"a{i}", "ablation") for i in range(4)] + [
        _make_candidate(f"b{i}", "baseline") for i in range(4)
    ]
    sel = _caps(candidates)
    assert len([c for c in sel.accepted if c.kind == "ablation"]) == 3
    assert len([c for c in sel.accepted if c.kind == "baseline"]) == 3
    assert len(sel.rejected) == 2
