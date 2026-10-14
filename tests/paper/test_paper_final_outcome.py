"""Final-evaluation outcome policy, negative-result study context, evidence grounding and source staging."""

from __future__ import annotations

import errno
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.final_evaluation.runner import FinalEvaluationError
from src.orchestrator.paper_data import FinalEvaluationMode, build_paper_data
from src.orchestrator.paper_evidence_audit import _paper_data_summary as audit_summary
from src.orchestrator.paper_final_outcome import (
    FinalOutcomePolicy,
    verify_paper_final_outcome,
    verify_resumed_final_outcome,
)
from src.orchestrator.paper_review import _paper_data_summary as review_summary
from src.pipeline.downstream_selection import _tree_digest
from src.pipeline.revision_evidence import _validate_semantic_evidence

ROOT = Path(__file__).resolve().parents[2]
PAPER_SCRIPTS = ("paper.sh", "paper_edit.sh")
TREE_HASHES = ("program_tree_sha256", "task_tree_sha256", "evaluator_sha256", "results_tree_sha256")
EPISODES = [{
    "seed": 7, "score": 2.4, "correct": True, "duration_seconds": 0.1,
    "solution": {"comparisons": {"starter": {"difference": 0.1}}},
    "candidate_diagnostics": {"iterations": 3},
}]
SOURCE_ID = "final.evidence.episodes"
BUNDLE_KINDS = ("ablation", "multitask_ablation")


def _sha(digit: int | str) -> str:
    return "sha256:" + str(digit) * 64


def _script(name: str) -> str:
    return (ROOT / "src/scripts" / name).read_text()


def _heredoc(text: str, start: int) -> str:
    start = text.index("<<'PY'\n", start) + len("<<'PY'\n")
    return text[start:text.index("\nPY\n", start)]


def _write_paper_data(directory: Path, data: dict) -> Path:
    directory.mkdir(exist_ok=True)
    (directory / "paper_data.json").write_text(json.dumps(data))
    return directory


def _patch_bundle(bundle: str):
    return (
        patch("src.orchestrator.paper_data.has_multitask_ablation_bundle",
              return_value=bundle == "multitask_ablation"),
        patch("src.orchestrator.paper_data.has_ablation_bundle", return_value=True),
        patch(f"src.orchestrator.paper_data.build_{bundle}_paper_data",
              return_value={"grounding_sources": {}}),
    )


# ── explicit publication policy for unsuccessful final evidence ──


NOT_QUALIFIED_METRICS = [
    {"adaptivity_gate_pass": 0.0, "combined_score": 0.622218},
    {"bbob_delta_lb_vs_starter": -0.0258536, "combined_score": 0.922779},
]


@pytest.fixture
def winner(tmp_path):
    investigation = tmp_path / "investigation"
    investigation.mkdir()
    final = tmp_path / "final"
    final.mkdir()
    (final / "final_manifest.json").write_text("{}")
    manifest = {
        "status": "completed", "outcome": "not_qualified", "correct": False,
        "metrics": dict(NOT_QUALIFIED_METRICS[0]),
        **{name: _sha(i) for i, name in enumerate(TREE_HASHES, 1)},
    }
    with (
        patch("src.orchestrator.paper_study_context.load_downstream_selection", return_value=None),
        patch("src.orchestrator.paper_data.resolve_frozen_incumbent",
              return_value=(tmp_path / "program", {})),
        patch("src.orchestrator.paper_data.verify_final_manifest", return_value=manifest) as verify,
    ):
        yield investigation, final, manifest, verify


def build(winner, **kwargs):
    investigation, final, _, _ = winner
    return build_paper_data(
        investigation, investigation_results_dir=investigation, final_evaluation_dir=final,
        final_outcome_policy=FinalOutcomePolicy.ALLOW_NOT_QUALIFIED, **kwargs,
    )


def test_opt_in_preserves_final_failure_without_fabricating_no_winner_and_summaries_disclose_it(
    winner,
):
    # opt-in preserves the final failure for every metric family
    for metrics in NOT_QUALIFIED_METRICS:
        winner[2]["metrics"] = dict(metrics)
        data = build(winner)
        assert "downstream_study" not in data
        assert data["final_evaluation"]["correct"] is False
        assert data["final_evaluation"]["outcome"] == "not_qualified"
        assert data["final_evaluation"]["metrics"] == metrics
        assert "validity" in " ".join(data["final_evaluation"]["limitations"])
        assert data["grounding_sources"]["final.qualification.correct"]["value"] is False
        assert data["grounding_sources"]["final.qualification.outcome"]["value"] == "not_qualified"
    # summaries disclose the failed qualification
    winner[2]["metrics"] = dict(NOT_QUALIFIED_METRICS[0])
    data = build(winner)
    for summarize in (audit_summary, review_summary):
        outcome = summarize(data)["final_evaluation"]
        assert outcome["outcome"] == "not_qualified"
        assert outcome["limitations"] == data["final_evaluation"]["limitations"]


def test_opt_in_applies_to_ablation_data(winner):
    for bundle in BUNDLE_KINDS:
        patches = _patch_bundle(bundle)
        with patches[0], patches[1], patches[2]:
            assert build(winner)["final_evaluation"]["correct"] is False, bundle


def test_unknown_policy_is_rejected_and_opt_in_still_verifies_final_evidence(winner):
    # unknown policy is rejected
    with pytest.raises(ValueError, match="FinalOutcomePolicy"):
        build_paper_data(winner[0], final_outcome_policy="anything")
    # opt-in still verifies final evidence
    winner[3].side_effect = FinalEvaluationError("results digest mismatch")
    with pytest.raises(ValueError, match="digest mismatch"):
        build(winner)


def test_audit_rejects_author_changes_to_final_evidence(winner, tmp_path):
    kwargs = dict(final_evaluation_dir=winner[1], investigation_results_dir=winner[0],
                  final_outcome_policy=FinalOutcomePolicy.ALLOW_NOT_QUALIFIED)
    for change in ("drop", "qualify", "metric", "grounding", "limitations", "forged_grounding"):
        data = build(winner)
        paper = _write_paper_data(tmp_path / f"paper_{change}", data)
        verify_paper_final_outcome(paper, **kwargs)
        if change == "drop":
            del data["final_evaluation"]
        elif change == "qualify":
            data["final_evaluation"].update(correct=True, outcome="qualified")
        elif change == "metric":
            data["final_evaluation"]["metrics"]["combined_score"] = 1.0
        elif change == "grounding":
            del data["grounding_sources"]["final.qualification.outcome"]
        elif change == "forged_grounding":
            data["grounding_sources"]["final.metric.fabricated_superiority"] = {
                "id": "final.metric.fabricated_superiority", "value": 1.0,
                "evaluation_scope": "sealed_final",
            }
        else:
            data["final_evaluation"]["limitations"] = []
        _write_paper_data(paper, data)
        with pytest.raises(ValueError, match="final evidence"):
            verify_paper_final_outcome(paper, **kwargs)


def test_revision_cannot_erase_final_qualification(winner, tmp_path):
    original = build(winner)
    source = _write_paper_data(tmp_path / "paper", original)
    revision = _write_paper_data(tmp_path / "revision", original)
    _validate_semantic_evidence(source, revision)
    original["final_evaluation"]["correct"] = True
    _write_paper_data(revision, original)
    with pytest.raises(ValueError, match="final evaluation inventory"):
        _validate_semantic_evidence(source, revision)


def test_fast_resume_requires_opt_in_again_and_cannot_omit_unavailable_original_final(
    winner, tmp_path,
):
    paper = _write_paper_data(tmp_path / "paper", build(winner))
    # fast resume requires opt-in again
    kwargs = dict(final_evaluation_dir=winner[1], source_results_dir=winner[0],
                  investigation_results_dir=winner[0])
    with pytest.raises(ValueError, match="did not qualify"):
        verify_resumed_final_outcome(paper, **kwargs)
    verify_resumed_final_outcome(
        paper, final_outcome_policy=FinalOutcomePolicy.ALLOW_NOT_QUALIFIED, **kwargs,
    )
    # fast resume cannot omit the unavailable original final
    with pytest.raises(ValueError, match="final evidence"):
        verify_resumed_final_outcome(paper, source_results_dir=winner[0],
                                     investigation_results_dir=winner[0])


# ── fallback studies retain negative outcomes ──


@pytest.fixture
def study(tmp_path: Path):
    investigation = tmp_path / "investigation"
    investigation.mkdir()
    final = tmp_path / "final_evaluation"
    final.mkdir()
    (final / "final_manifest.json").write_text("{}")
    frozen_program = tmp_path / "frozen"
    selection = {
        "selection_kind": "non_winner_candidate",
        "investigation_outcome": "completed_no_winner",
        "combined_score": 0.809734,
        "baseline_score": 0.811173,
        "baseline_program_dir": str(investigation / "baseline"),
        "baseline_tree_sha256": _sha(5),
        "source_tree_sha256": _sha(6),
        "source_attempt_id": "attempt_20",
        "source_program_dir": str(investigation / "attempt_20" / "program_1"),
        "program_dir": str(frozen_program),
        "entrypoint_path": str(frozen_program / "main.py"),
        "program_tree_sha256": _sha(1),
        "attempt_counts": {"total": 20, "scored": 1, "unscored": 19},
    }
    manifest = {
        "status": "completed", "outcome": "not_qualified", "correct": False,
        "metrics": {"held_out_score": 0.8},
        "program_tree_sha256": selection["program_tree_sha256"],
        "task_tree_sha256": _sha(2), "evaluator_sha256": _sha(3), "results_tree_sha256": _sha(4),
    }
    with (
        patch("src.orchestrator.paper_study_context.load_downstream_selection",
              return_value=selection) as load,
        patch("src.orchestrator.paper_data.resolve_frozen_incumbent",
              return_value=(frozen_program, {})),
        patch("src.orchestrator.paper_data.verify_final_manifest",
              return_value=manifest) as verify,
    ):
        yield investigation, final, selection, manifest, load, verify


def _build_study(investigation, **kwargs):
    return build_paper_data(investigation, investigation_results_dir=investigation, **kwargs)


def test_fallback_paper_preserves_negative_final_and_verifies_against_frozen_selection(study):
    investigation, final, selection, manifest, _, verify = study
    # preserves the negative final and attempt denominators
    data = _build_study(investigation, final_evaluation_dir=final)
    context = data["downstream_study"]
    assert context["selection_kind"] == "non_winner_candidate"
    assert context["baseline_program_dir"] == selection["baseline_program_dir"]
    assert context["source_tree_sha256"] == selection["source_tree_sha256"]
    assert context["search_score_delta"] == pytest.approx(-0.001439)
    assert context["attempt_counts"] == {"total": 20, "scored": 1, "unscored": 19}
    assert context["final_evaluation"]["outcome"] == "not_qualified"
    assert data["final_evaluation"]["correct"] is False
    assert data["grounding_sources"]["study.attempts.unscored"]["value"] == 19
    assert data["grounding_sources"]["study.search_score_delta"]["value"] < 0
    assert data["grounding_sources"]["final.metric.held_out_score"]["value"] == 0.8
    assert "validity" in " ".join(context["limitations"])
    verify.assert_called_once_with(final.resolve(), task_dir=None,
                                   program_dir=Path(selection["program_dir"]))
    # final must match the frozen selection even without an explicit investigation dir
    verify.reset_mock()
    build_paper_data(investigation, final_evaluation_dir=final)
    verify.assert_called_once_with(final.resolve(), task_dir=None,
                                   program_dir=Path(selection["program_dir"]))


def test_fallback_context_survives_final_omission_and_is_discovered_from_results_directory(study):
    investigation, _, selection, _, load, verify = study
    # context survives final omission
    selection["selection_kind"] = "baseline"
    data = _build_study(investigation, final_evaluation_mode=FinalEvaluationMode.OMIT)
    assert data["downstream_study"]["selection_kind"] == "baseline"
    assert data["downstream_study"]["final_evaluation"] is None
    verify.assert_not_called()
    # context is discovered from the results directory
    selection["selection_kind"] = "non_winner_candidate"
    load.reset_mock()
    data = build_paper_data(investigation, final_evaluation_mode=FinalEvaluationMode.OMIT)
    assert "downstream_study" in data
    load.assert_called_once_with(investigation)


def test_invalid_baseline_is_not_a_grounded_comparison(study):
    investigation, _, selection, _, _, _ = study
    selection["baseline_program_dir"] = None
    selection["baseline_tree_sha256"] = None
    data = build_paper_data(investigation, final_evaluation_mode=FinalEvaluationMode.OMIT)
    context = data["downstream_study"]
    assert context["baseline_score"] is None
    assert context["search_score_delta"] is None
    assert "study.baseline_score" not in data["grounding_sources"]
    assert "study.search_score_delta" not in data["grounding_sources"]
    assert any("baseline" in note and "unavailable" in note for note in context["limitations"])


def test_fallback_context_is_attached_to_ablation_paper_bundles(study):
    investigation, final, _, _, _, _ = study
    for bundle_kind in BUNDLE_KINDS:
        patches = _patch_bundle(bundle_kind)
        with patches[0], patches[1], patches[2]:
            data = _build_study(investigation, final_evaluation_dir=final)
        assert data["downstream_study"]["final_evaluation"]["outcome"] == "not_qualified", bundle_kind


def test_fallback_rejects_not_qualified_winner_unverified_final_and_stale_selection(study):
    investigation, final, selection, _, load, verify = study
    # a winner still rejects a not-qualified final
    load.return_value = None
    with pytest.raises(ValueError, match="did not qualify"):
        _build_study(investigation, final_evaluation_dir=final)
    load.return_value = selection
    # fallback does not admit an unverified final
    verify.side_effect = FinalEvaluationError("final results digest mismatch")
    with pytest.raises(ValueError, match="digest mismatch"):
        _build_study(investigation, final_evaluation_dir=final)
    # stale selection blocks the paper even without a final
    load.side_effect = ValueError("downstream selection digest mismatch")
    with pytest.raises(ValueError, match="selection digest mismatch"):
        _build_study(investigation, final_evaluation_mode=FinalEvaluationMode.OMIT)


def test_fallback_supports_missing_final_and_qualified_final_keeps_no_winner_outcome(study):
    investigation, final, _, manifest, _, _ = study
    # tasks without a final evaluation
    data = _build_study(investigation, final_evaluation_dir=investigation / "missing")
    assert data["downstream_study"]["final_evaluation"] is None
    # a qualified final does not rewrite the no-winner search outcome
    manifest.update(correct=True, outcome="qualified")
    context = _build_study(investigation, final_evaluation_dir=final)["downstream_study"]
    assert context["investigation_outcome"] == "completed_no_winner"
    assert context["final_evaluation"]["outcome"] == "qualified"


def test_summaries_include_study_context_and_revision_cannot_drop_or_change_it(tmp_path: Path):
    # review and audit summaries include the study context
    context: dict[str, Any] = {"investigation_outcome": "completed_no_winner"}
    for summarize in (review_summary, audit_summary):
        assert summarize({"downstream_study": context})["downstream_study"] == context
    # revision cannot drop or change the negative study context
    context = {"investigation_outcome": "completed_no_winner",
               "attempt_counts": {"total": 20, "scored": 1, "unscored": 19}}
    original = {"downstream_study": context}
    source = _write_paper_data(tmp_path / "source", original)
    candidate = _write_paper_data(tmp_path / "candidate", original)
    _validate_semantic_evidence(source, candidate)
    for revised in ({}, {"downstream_study": {"investigation_outcome": "completed"}}):
        _write_paper_data(candidate, revised)
        with pytest.raises(ValueError, match="downstream study context"):
            _validate_semantic_evidence(source, candidate)


def test_legacy_no_winner_paper_can_resume_with_verified_negative_final(study, tmp_path):
    investigation, final, _, _, _, _ = study
    data = _build_study(investigation, final_evaluation_dir=final)
    del data["final_evaluation"]["limitations"]
    data["grounding_sources"] = {
        key: value for key, value in data["grounding_sources"].items()
        if not key.startswith("final.qualification.")
    }
    paper = _write_paper_data(tmp_path / "paper", data)
    verify_resumed_final_outcome(
        paper, final_evaluation_dir=final, source_results_dir=investigation,
        investigation_results_dir=investigation,
    )


def test_paper_prompts_use_study_context_for_negative_results():
    for prompt in (
        "paper_strategy.md", "paper.md", "paper_standalone.md", "paper_detailed_grouped.md",
        "paper_edit.md", "paper_edit_standalone.md", "paper_revision.md",
        "paper_review/evidence_audit.md", "paper_review/review_json.md",
    ):
        text = (ROOT / "src" / "prompts" / prompt).read_text()
        assert "downstream_study" in text, prompt
        assert "negative" in text.lower(), prompt
        assert "unscored" in text.lower(), prompt


def test_paper_preparation_accepts_verified_baseline_without_generated_evaluations():
    for script_name in PAPER_SCRIPTS:
        script = _script(script_name)
        start = script.index('if int(paper_data["totals"]["evaluated_programs"])')
        guard = script[start:script.index("\n\n", start)]
        data = {"totals": {"evaluated_programs": 0},
                "downstream_study": {"selection_kind": "baseline"}}
        exec(compile(guard, script_name, "exec"), {"paper_data": data})
        del data["downstream_study"]
        with pytest.raises(SystemExit, match="No evaluated programs"):
            exec(compile(guard, script_name, "exec"), {"paper_data": data})


# ── structured final observations have immutable host-owned claim sources ──


@pytest.fixture(params=[True, False], ids=["qualified", "not_qualified"])
def final_run(tmp_path, request):
    investigation = tmp_path / "investigation"
    investigation.mkdir()
    final = tmp_path / "final"
    results = final / "results"
    results.mkdir(parents=True)
    (results / "evaluation_evidence.json").write_text(json.dumps({
        "schema_version": 1, "episodes": EPISODES,
    }))
    (final / "final_manifest.json").write_text("{}")
    manifest = {
        "status": "completed", "correct": request.param,
        "outcome": "qualified" if request.param else "not_qualified",
        "metrics": {"combined_score": 2.4},
        **{name: _sha(i) for i, name in enumerate(TREE_HASHES, 1)},
    }
    with (
        patch("src.orchestrator.paper_data.resolve_frozen_incumbent",
              return_value=(tmp_path / "program", {})),
        patch("src.orchestrator.paper_data.verify_final_manifest", return_value=manifest),
    ):
        yield investigation, final


def build_final(final_run):
    investigation, final = final_run
    return build_paper_data(
        investigation, investigation_results_dir=investigation, final_evaluation_dir=final,
        final_outcome_policy=FinalOutcomePolicy.ALLOW_NOT_QUALIFIED,
    )


def verify_final(final_run, tmp_path, data):
    paper = _write_paper_data(tmp_path / "paper", data)
    verify_paper_final_outcome(
        paper, investigation_results_dir=final_run[0], final_evaluation_dir=final_run[1],
        final_outcome_policy=FinalOutcomePolicy.ALLOW_NOT_QUALIFIED,
    )


def test_observations_have_a_canonical_source_and_legacy_or_unavailable_observations_verify(
    final_run, tmp_path,
):
    # observations have a canonical source without duplicating records
    data = build_final(final_run)
    final = data["final_evaluation"]
    evidence = final["evaluation_evidence"]
    assert data["grounding_sources"][SOURCE_ID] == {
        "id": SOURCE_ID, "type": "evaluation_evidence", "evaluation_scope": "sealed_final",
        "path": evidence["path"], "sha256": evidence["sha256"],
        "json_pointer": "/episodes", "episode_count": 1,
        "candidate_diagnostics_trust": "unverified",
        "manifest_path": final["manifest_path"],
        "program_tree_sha256": final["program_tree_sha256"],
        "results_tree_sha256": final["results_tree_sha256"],
    }
    assert evidence["episodes"] == EPISODES
    verify_final(final_run, tmp_path, data)
    # a legacy paper without the new source still verifies
    data = build_final(final_run)
    del data["grounding_sources"][SOURCE_ID]
    verify_final(final_run, tmp_path, data)
    # unavailable observations do not get a source
    (final_run[1] / "results/evaluation_evidence.json").unlink()
    data = build_final(final_run)
    assert SOURCE_ID not in data["grounding_sources"]
    verify_final(final_run, tmp_path, data)


def test_author_cannot_change_or_invent_evidence_sources_and_legacy_observations_still_verify(
    final_run, tmp_path,
):
    # the author cannot change the evidence source
    for field, value in (
        ("path", "/tmp/author-evidence.json"), ("sha256", _sha(0)),
        ("json_pointer", "/episodes/0/candidate_diagnostics"),
        ("candidate_diagnostics_trust", "verified"), ("episode_count", 99),
        ("results_tree_sha256", _sha(0)), ("evaluation_scope", "search"),
    ):
        data = build_final(final_run)
        data["grounding_sources"][SOURCE_ID][field] = value
        with pytest.raises(ValueError, match="grounding differs"):
            verify_final(final_run, tmp_path, data)
    # legacy compatibility does not allow invented sources
    for invented_id in ("final.metric.paired_comparisons", "final.evidence.fake"):
        data = build_final(final_run)
        del data["grounding_sources"][SOURCE_ID]
        data["grounding_sources"][invented_id] = {"id": invented_id, "value": EPISODES}
        with pytest.raises(ValueError, match="grounding differs"):
            verify_final(final_run, tmp_path, data)
    # legacy compatibility still verifies the underlying observations
    data = build_final(final_run)
    del data["grounding_sources"][SOURCE_ID]
    data["final_evaluation"]["evaluation_evidence"]["episodes"][0]["score"] = 100
    with pytest.raises(ValueError, match="final evidence differs"):
        verify_final(final_run, tmp_path, data)


# ── shell entrypoints forward the final-outcome policy ──


@pytest.mark.parametrize("script", PAPER_SCRIPTS)
def test_script_rejects_invalid_policy_before_preparation(script):
    for args in (
        ["--final-outcome-policy"],
        ["--final-outcome-policy", "unknown"],
        ["--final-outcome-policy", "qualified_only", "--final-outcome-policy", "allow_not_qualified"],
    ):
        result = subprocess.run(["bash", str(ROOT / "src/scripts" / script), *args],
                                capture_output=True, text=True)
        assert result.returncode != 0, args
        assert "--final-outcome-policy" in result.stderr, args


@pytest.mark.parametrize("script", PAPER_SCRIPTS)
def test_every_paper_data_build_receives_explicit_policy(script, tmp_path):
    text = _script(script)
    for phase in ("preparation", "references"):
        start = text.index('retrieve_references_from_plan() {') if phase == "references" else 0
        build_start = text.index("paper_data = build_paper_data(", start)
        start = text.rfind("<<'PY'\n", 0, build_start) + len("<<'PY'\n")
        end = text.index("\n)", build_start) + 2
        block = text[start:end]
        args = ["-", str(ROOT), str(tmp_path), str(tmp_path), str(tmp_path), str(tmp_path),
                str(tmp_path), "8", "15", "default"]
        if phase == "references":
            args.append(str(tmp_path / "query.json"))
        args.extend([str(tmp_path / "final"), "discover", str(tmp_path), str(tmp_path), "",
                     "allow_not_qualified"])
        with patch.object(sys, "argv", args), patch(
            "src.orchestrator.paper_data.build_paper_data", return_value={}
        ) as build_mock:
            exec(compile(block, script, "exec"), {})
        assert build_mock.call_args.kwargs["final_outcome_policy"] == "allow_not_qualified", phase


def test_fast_resume_checks_final_policy_before_finalization(tmp_path):
    text = _script("paper.sh")
    block = _heredoc(text, text.index('paper_gate_start "finalization" "phase=resume"'))
    args = ["-", str(ROOT), str(tmp_path), str(tmp_path), str(tmp_path), "0", "", "bridge",
            "0", "", str(tmp_path / "final"), str(tmp_path), str(tmp_path), "", "qualified_only"]
    with (
        patch.object(sys, "argv", args),
        patch("src.orchestrator.paper_final_outcome.verify_resumed_final_outcome",
              side_effect=ValueError("Final evaluation did not qualify")) as verify,
        patch("src.orchestrator.paper_finalize.finalize_resumed_paper_if_valid") as finalize,
        pytest.raises(SystemExit) as failure,
    ):
        exec(compile(block, "paper.sh", "exec"), {})
    assert failure.value.code == 2
    assert verify.call_args.kwargs["final_outcome_policy"] == "qualified_only"
    finalize.assert_not_called()


# ── fallback implementation sources stay readable through paper, audit and revision ──


def _make_sources(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    program = root / "program"
    baseline = root / "baseline"
    for source_root, text in ((program, "selected = True\n"), (baseline, "selected = False\n")):
        source_root.mkdir()
        (source_root / "main.py").write_text(text)
        (source_root / "pkg").mkdir()
        (source_root / "pkg" / "metrics.json").write_text('{"code_resource": true}')
    (baseline / "metrics.json").write_text('{"combined_score": 1}')
    selection = {
        "program_dir": str(program), "program_tree_sha256": _tree_digest(program),
        "baseline_program_dir": str(baseline), "baseline_tree_sha256": _tree_digest(baseline),
    }
    paper = root / "paper"
    paper.mkdir()
    return paper, {"downstream_study": dict(selection)}, selection


@pytest.fixture
def sources(tmp_path):
    return _make_sources(tmp_path)


def _stage(paper, data, selection):
    from src.orchestrator.paper_study_sources import stage_study_sources

    with patch("src.orchestrator.paper_study_sources.load_downstream_selection",
               return_value=selection):
        stage_study_sources(data, paper, paper.parent / "investigation")


def _verify_sources(paper, selection):
    from src.orchestrator.paper_study_sources import verify_study_sources

    with patch("src.orchestrator.paper_study_sources.load_downstream_selection",
               return_value=selection):
        verify_study_sources(paper, paper.parent / "investigation")


def _disable_exclusive_rename(monkeypatch):
    from src.pipeline import safe_fs

    monkeypatch.setattr(safe_fs, "_renameat2_no_replace", lambda *a, **k: False)
    monkeypatch.setattr(safe_fs, "_renameatx_np_no_replace", lambda *a, **k: False)


@pytest.fixture
def unsupported_source_rename(monkeypatch):
    _disable_exclusive_rename(monkeypatch)


def test_stages_code_with_anchored_exclusions_preaudit_accepts_it_and_absent_sources_are_skipped(
    tmp_path,
):
    # stages code with anchored evaluation exclusions (and re-staging is idempotent)
    paper, data, selection = _make_sources(tmp_path / "staged")
    _stage(paper, data, selection)
    paths = data["downstream_study"]["source_evidence"]
    assert (paper / paths["program"] / "main.py").read_text() == "selected = True\n"
    assert (paper / paths["baseline"] / "main.py").read_text() == "selected = False\n"
    assert not (paper / paths["baseline"] / "metrics.json").exists()
    assert (paper / paths["baseline"] / "pkg/metrics.json").exists()
    _stage(paper, data, selection)
    # preaudit accepts unchanged sources and context
    paper, data, selection = _make_sources(tmp_path / "accepted")
    _stage(paper, data, selection)
    _write_paper_data(paper, data)
    _verify_sources(paper, selection)
    # an unavailable baseline does not create baseline sources
    paper, data, selection = _make_sources(tmp_path / "no_baseline")
    selection.update(baseline_program_dir=None, baseline_tree_sha256=None)
    data["downstream_study"].update(selection)
    _stage(paper, data, selection)
    assert set(data["downstream_study"]["source_evidence"]) == {"program"}
    assert not (paper / "source_evidence/downstream/baseline").exists()
    # a winner needs no source staging
    from src.orchestrator.paper_study_sources import stage_study_sources

    winner = tmp_path / "winner"
    winner.mkdir()
    stage_study_sources({}, winner, winner / "investigation")
    assert list(winner.iterdir()) == []


def test_sources_survive_readonly_audit_and_both_revision_trees(sources):
    from src.orchestrator.paper_evidence_audit import prepare_readonly_workspace
    from src.investigation.paper_revision_workspace import (
        seed_revision_source_evidence, snapshot_previous_paper,
    )

    paper, data, selection = sources
    _stage(paper, data, selection)
    _write_paper_data(paper, data)
    readonly = paper.parent / "readonly"
    prepare_readonly_workspace(paper, readonly)
    revision = paper.parent / "revision"
    revision.mkdir()
    snapshot = revision / "previous_paper"
    snapshot_previous_paper(paper, snapshot)
    seed_revision_source_evidence(snapshot, revision)
    _stage(revision, data, selection)
    for root in (readonly, snapshot, revision):
        for role, text in (("program", "selected = True\n"), ("baseline", "selected = False\n")):
            path = data["downstream_study"]["source_evidence"][role]
            assert (root / path / "main.py").read_text() == text


def test_rejects_symlinked_destination_without_writing_outside(tmp_path):
    for index, relative in enumerate((
        "source_evidence", "source_evidence/downstream", "source_evidence/downstream/program",
    )):
        paper, data, selection = _make_sources(tmp_path / f"case_{index}")
        outside = paper.parent / "outside"
        outside.mkdir()
        target = paper / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError):
            _stage(paper, data, selection)
        assert list(outside.iterdir()) == [], relative


def test_rejects_changed_existing_sources_and_sources_changed_after_selection(tmp_path):
    # changed existing sources are rejected without overwriting
    paper, data, selection = _make_sources(tmp_path / "existing")
    _stage(paper, data, selection)
    target = paper / "source_evidence/downstream/program/main.py"
    target.write_text("changed\n")
    with pytest.raises(ValueError, match="differ"):
        _stage(paper, data, selection)
    assert target.read_text() == "changed\n"
    # a source changed after selection verification is rejected
    paper, data, selection = _make_sources(tmp_path / "after_selection")
    (Path(selection["program_dir"]) / "main.py").write_text("changed\n")
    with pytest.raises(ValueError, match="digest"):
        _stage(paper, data, selection)


def test_bounds_source_copy(sources, monkeypatch):
    import src.orchestrator.paper_study_sources as module

    paper, data, selection = sources
    for limit in ("MAX_SOURCE_BYTES", "MAX_SOURCE_ENTRIES"):
        with monkeypatch.context() as m:
            m.setattr(module, limit, 1)
            with pytest.raises(ValueError, match="limit"):
                _stage(paper, data, selection)
        assert not (paper / "source_evidence/downstream").exists(), limit


def test_revision_script_stages_after_seeding_and_persists_relative_paths(sources):
    paper, data, selection = sources
    data_path = _write_paper_data(paper, data) / "paper_data.json"
    script = _script("paper_edit.sh")
    block = _heredoc(script, script.index("\nseed_edit_workspace\n"))
    with (
        patch.object(sys, "argv", ["-", str(ROOT), str(paper), str(paper.parent / "investigation")]),
        patch("src.orchestrator.paper_study_sources.load_downstream_selection",
              return_value=selection),
    ):
        exec(compile(block, "paper_edit.sh", "exec"), {})
    paths = json.loads(data_path.read_text())["downstream_study"]["source_evidence"]
    assert (paper / paths["program"] / "main.py").is_file()


def test_preaudit_rejects_author_changes_without_recreating_sources(tmp_path):
    for mutation in ("code", "missing_code", "missing_context", "source_path",
                     "digest", "score", "attempts"):
        paper, data, selection = _make_sources(tmp_path / mutation)
        selection.update(combined_score=0.8, attempt_counts={"total": 20, "scored": 1, "unscored": 19})
        data["downstream_study"].update(selection)
        _stage(paper, data, selection)
        source = paper / "source_evidence/downstream/program/main.py"
        if mutation == "code":
            source.write_text("invented_method = True\n")
        elif mutation == "missing_code":
            source.unlink()
        elif mutation == "missing_context":
            del data["downstream_study"]
        elif mutation == "source_path":
            data["downstream_study"]["source_evidence"]["program"] = "other"
        elif mutation == "digest":
            data["downstream_study"]["program_tree_sha256"] = "changed"
        elif mutation == "score":
            data["downstream_study"]["combined_score"] = 1.0
        else:
            data["downstream_study"]["attempt_counts"] = {"total": 20, "scored": 20, "unscored": 0}
        _write_paper_data(paper, data)
        with pytest.raises(ValueError):
            _verify_sources(paper, selection)
        if mutation == "missing_code":
            assert not source.exists()


@pytest.mark.parametrize("script_name", PAPER_SCRIPTS)
def test_production_preaudit_hook_checks_source_before_readonly_copy(sources, script_name):
    paper, data, selection = sources
    _stage(paper, data, selection)
    _write_paper_data(paper, data)
    (paper / "source_evidence/downstream/program/main.py").write_text("changed\n")
    script = _script(script_name)
    function_start = script.index("\nprepare_readonly_paper_audit_workspace()")
    start = script.index("<<'PY'\n", function_start) + len("<<'PY'\n")
    end = script.index("\nPY\n", start)
    assert end < script.index("prepare-audit-workspace", function_start)
    with (
        patch.object(sys, "argv", ["-", str(ROOT), str(paper), str(paper.parent / "investigation")]),
        patch("src.orchestrator.paper_study_sources.load_downstream_selection",
              return_value=selection),
        pytest.raises(ValueError, match="differ"),
    ):
        exec(compile(script[start:end], script_name, "exec"), {})


@pytest.mark.parametrize("script_name", PAPER_SCRIPTS)
def test_shell_preaudit_stops_after_verification_failure(script_name):
    script = _script(script_name)
    start = script.index("\nprepare_readonly_paper_audit_workspace()")
    end = script.index("\n}\n", start) + len("\n}\n")
    for error_mode in ("-e", "+e"):
        command = f"""
set {error_mode}
{script[start:end]}
PYTHON_BIN=false
REPO_ROOT=unused
OUTPUT_DIR=unused
SOURCE_RESULTS_DIR=unused
if prepare_readonly_paper_audit_workspace audit output; then
  exit 99
fi
echo rejected
"""
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
        assert result.returncode == 0, error_mode
        assert result.stdout == "rejected\n", error_mode


def test_stages_reuses_verified_sources_and_retains_partial_publication_without_exclusive_rename(
    tmp_path, monkeypatch,
):
    from src.orchestrator import paper_study_sources as module

    _disable_exclusive_rename(monkeypatch)
    # stages and reuses verified sources without an exclusive rename
    paper, data, selection = _make_sources(tmp_path / "reuse")
    _stage(paper, data, selection)
    target = paper / "source_evidence/downstream/program/main.py"
    before = target.stat()
    _stage(paper, data, selection)
    assert target.read_text() == "selected = True\n"
    assert target.stat() == before
    # a partial source publication is retained and rejected on retry
    paper, data, selection = _make_sources(tmp_path / "partial")
    original = module._write_tree
    writes = 0

    def interrupt_publication(directory_fd, tree):
        nonlocal writes
        writes += 1
        if writes == 1:
            return original(directory_fd, tree)
        original(directory_fd, {"program": (None, 0o700)})
        raise OSError(errno.ENOSPC, "injected full filesystem")

    monkeypatch.setattr(module, "_write_tree", interrupt_publication)
    with pytest.raises(ValueError, match="injected full filesystem"):
        _stage(paper, data, selection)
    partial = paper / "source_evidence/downstream/program"
    before = partial.stat()
    monkeypatch.setattr(module, "_write_tree", original)
    with pytest.raises(ValueError, match="differ"):
        _stage(paper, data, selection)
    assert partial.stat() == before
    assert "source_evidence" not in data["downstream_study"]


def test_source_destination_replaced_during_copy_or_before_open_is_rejected(tmp_path, monkeypatch):
    from src.orchestrator import paper_study_sources as module

    _disable_exclusive_rename(monkeypatch)
    # destination replaced during the copy
    for replacement in ("directory", "symlink"):
        paper, data, selection = _make_sources(tmp_path / f"during_{replacement}")
        destination = paper / "source_evidence/downstream"
        original = module._write_tree
        writes = 0

        def replace_destination(directory_fd, tree, destination=destination,
                                replacement=replacement, original=original):
            nonlocal writes
            writes += 1
            original(directory_fd, tree)
            if writes == 2:
                displaced = destination.with_name("displaced")
                destination.rename(displaced)
                if replacement == "directory":
                    destination.mkdir()
                else:
                    destination.symlink_to(displaced, target_is_directory=True)

        with monkeypatch.context() as m:
            m.setattr(module, "_write_tree", replace_destination)
            with pytest.raises(ValueError, match="changed"):
                _stage(paper, data, selection)
        assert "source_evidence" not in data["downstream_study"], replacement
    # destination replaced before open is never written
    paper, data, selection = _make_sources(tmp_path / "before_open")
    destination = paper / "source_evidence/downstream"
    original_open = module.open_relative_directory_no_follow

    def replace_before_open(parent_fd, path):
        if path == Path("downstream") and destination.exists():
            destination.rename(destination.with_name("displaced"))
            destination.mkdir()
        return original_open(parent_fd, path)

    monkeypatch.setattr(module, "open_relative_directory_no_follow", replace_before_open)
    with pytest.raises(ValueError, match="changed before copying"):
        _stage(paper, data, selection)
    assert list(destination.iterdir()) == []


def test_competing_publisher_unrelated_rename_failure_and_incomplete_copy_are_rejected(
    tmp_path, monkeypatch,
):
    from src.orchestrator import paper_study_sources as module
    from src.pipeline import safe_fs

    # a competing source publisher cannot replace the reserved directory
    paper, data, selection = _make_sources(tmp_path / "competing")
    original_mkdir = module.os.mkdir
    destination = paper / "source_evidence/downstream"

    def reserve_first(path, *args, **kwargs):
        if path == "downstream":
            original_mkdir(path, *args, **kwargs)
            (destination / "keep").write_text("other publisher")
        return original_mkdir(path, *args, **kwargs)

    with monkeypatch.context() as m:
        _disable_exclusive_rename(m)
        m.setattr(module.os, "mkdir", reserve_first)
        with pytest.raises(ValueError, match="File exists"):
            _stage(paper, data, selection)
    assert (destination / "keep").read_text() == "other publisher"
    assert set(destination.iterdir()) == {destination / "keep"}

    # an unrelated source rename failure does not use the fallback
    paper, data, selection = _make_sources(tmp_path / "unrelated")

    def fail(*args, **kwargs):
        raise safe_fs.UnsafeGluePathError("injected I/O failure")

    with monkeypatch.context() as m:
        m.setattr(safe_fs, "_renameat2_no_replace", fail)
        with pytest.raises(ValueError, match="injected I/O failure"):
            _stage(paper, data, selection)
    assert not (paper / "source_evidence/downstream").exists()

    # the source copy must match the expected inventory before publication
    paper, data, selection = _make_sources(tmp_path / "incomplete")
    original_write = module._write_tree
    writes = 0

    def incomplete_copy(directory_fd, tree):
        nonlocal writes
        writes += 1
        original_write(directory_fd, tree if writes == 1 else {"program": (None, 0o700)})

    with monkeypatch.context() as m:
        _disable_exclusive_rename(m)
        m.setattr(module, "_write_tree", incomplete_copy)
        with pytest.raises(ValueError, match="differ"):
            _stage(paper, data, selection)
    assert "source_evidence" not in data["downstream_study"]
