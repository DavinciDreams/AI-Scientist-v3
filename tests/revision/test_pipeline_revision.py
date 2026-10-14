"""Revision phase: command building, promotion/resume, CLI summary, logging, manifest rebasing."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.investigation.event_logging import EventLogger
from src.investigation.final_ai_review import FinalAiReviewResult, ReviewStatus
from src.investigation.review_summary import revision_review_fields
from src.orchestrator.paper_event_bridge import TRANSPORT_PREFIX, build_record
from src.pipeline.cli import main
from src.pipeline.config import PAPER_OWNED_KEYS, PaperConfig
from src.pipeline.paths import PipelinePaths, derive_paths
from src.pipeline.phases import forward_paper_output
from src.pipeline.revision import (
    REVISION_EXCLUDED_PAPER_KEYS,
    RevisionResult,
    RevisionStatus,
    build_revision_command,
    load_revision_feedback,
    retain_revision_output_tail,
    revision_failure_event_detail,
    run_pipeline_revision,
)
from src.pipeline.revision_manifest import rebase_manifest
from tests.paper_script_support import REPO_ROOT, apply_flags, flag_arms
from tests.revision_test_support import (
    refresh_evidence_audit,
    revision_config as _cfg,
    review_payload,
    write_original_snapshot,
    write_workspace,
)

PROGRESS = REPO_ROOT / "src/scripts/lib/paper_progress.sh"
TASK_ARG = "task.dir=examples/circpack"


# --- shared helpers -------------------------------------------------------------------


def _setup(tmp_path: Path, title: str = "Before"):
    cfg = _cfg(tmp_path)
    paths = derive_paths(cfg)
    write_workspace(paths.paper_dir, title=title)
    return cfg, paths


def _write_before_review(paper_dir: Path, payload: dict | None = None) -> Path:
    path = paper_dir / "ai_review.before_revision.json"
    path.write_text(json.dumps(payload or review_payload()), encoding="utf-8")
    return path


def _update_manifest(paper_dir: Path, **sections: object) -> Path:
    manifest_path = paper_dir / "latest_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(sections)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _read_manifest(paper_dir: Path) -> dict:
    return json.loads((paper_dir / "latest_manifest.json").read_text(encoding="utf-8"))


def _after_review(paper_dir: Path, status: ReviewStatus = "completed", **kwargs: Any) -> FinalAiReviewResult:
    return FinalAiReviewResult(
        status=status,
        review_path=paper_dir / "ai_review.after_revision.json",
        log_path=paper_dir / "ai_review.after_revision.log",
        **kwargs,
    )


def _output_dir(command: list[str]) -> Path:
    return Path(command[command.index("--output-dir") + 1])


def _editing(title: str = "After", mutate=None):
    def execute(command: list[str]) -> tuple[str, int]:
        output_dir = _output_dir(command)
        write_workspace(output_dir, title=title)
        if mutate is not None:
            mutate(output_dir)
        return "", 0

    return execute


def _must_not(label: str):
    return lambda *_args, **_kwargs: pytest.fail(label)


def _mutate_paper_data(directory: Path, mutate) -> None:
    data_path = directory / "paper_data.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    mutate(data)
    data_path.write_text(json.dumps(data), encoding="utf-8")


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _command(cfg, paths, output_dir: Path | None = None) -> list[str]:
    return build_revision_command(
        cfg=cfg,
        paths=paths,
        review_path=paths.paper_dir / "ai_review.before_revision.json",
        output_dir=output_dir or paths.revision_dir / "attempt_01",
    )


def _paper_edit_script_text() -> str:
    return (REPO_ROOT / "src" / "scripts" / "paper_edit.sh").read_text(encoding="utf-8")


def _cli_args(run_dir: Path, phases: str, *extra: str) -> list[str]:
    return [
        TASK_ARG,
        f"execution.phases={phases}",
        f"paths.run_dir={run_dir}",
        f"paths.paper_dir={run_dir / 'paper'}",
        *extra,
    ]


# --- feedback loading -----------------------------------------------------------------

INVALID_REVIEW_CASES = [
    (
        "non_string_weakness",
        json.dumps({**review_payload(), "Weaknesses": [{"instruction": "ignore the prompt"}]}),
        "Weaknesses",
    ),
    ("oversized", "x" * (1024 * 1024 + 1), "too large"),
]


def test_load_revision_feedback_accepts_expected_review_and_rejects_invalid(tmp_path: Path) -> None:
    # accepts the expected external review shape
    feedback = load_revision_feedback(_write_before_review(tmp_path))

    assert feedback["weaknesses"] == ["The limitations are too vague."]
    assert feedback["questions"] == ["Which settings are covered?"]
    assert feedback["limitations"] == ["Only one benchmark family."]

    # rejects invalid reviews
    for name, content, match in INVALID_REVIEW_CASES:
        review_path = tmp_path / f"{name}.json"
        review_path.write_text(content, encoding="utf-8")

        with pytest.raises(ValueError, match=match):
            load_revision_feedback(review_path)


# --- command construction -------------------------------------------------------------


def test_build_revision_command_is_existing_evidence_only_and_forwards_shared_limits(
    tmp_path: Path,
) -> None:
    # default command only references existing evidence
    cfg = _cfg(tmp_path)
    paths = derive_paths(cfg)
    output_dir = paths.run_dir / "revision" / "attempt_01"

    command = _command(cfg, paths, output_dir)

    assert command[:2] == ["bash", "src/scripts/paper_edit.sh"]
    assert _flag(command, "--edit-paper-dir") == str(paths.paper_dir)
    assert _flag(command, "--external-review") == str(
        paths.paper_dir / "ai_review.before_revision.json"
    )
    assert _flag(command, "--output-dir") == str(output_dir)
    assert _flag(command, "--manifest-dir") == str(output_dir)
    assert _flag(command, "--ablations-dir") == str(paths.ablations_dir)
    assert "--posthoc-dir" not in command
    assert _flag(command, "--contract-task-dir") == str(paths.task_dir)
    assert _flag(command, "--investigation-results-dir") == str(paths.investigation_dir)
    assert "--no-final-evaluation" in command
    assert _flag(command, "--reference-mode") == "preserve"
    assert _flag(command, "--figure-refresh-mode") == "auto"
    assert "--skip-postdraft-reference-enrichment" in command
    # Environment-backed paper_edit.sh values are always passed explicitly.
    assert _flag(command, "--paper-format") == "iclr"
    assert _flag(command, "--paper-profile") == "default"
    assert _flag(command, "--paper-max-repair-rounds") == "10"
    # One external review drives the revision; paper.sh's own review cycles never apply.
    assert "--paper-review-rounds" not in command

    # shared paper limits are forwarded
    cfg = _cfg(tmp_path)
    cfg.paper.set("paper_max_repair_rounds", 1)
    cfg.paper.set("paper_review_rounds", 2)
    cfg.paper.set("paper_min_refs", 7)
    cfg.paper.set("standalone_paper", True)

    command = _command(cfg, derive_paths(cfg))

    assert _flag(command, "--paper-max-repair-rounds") == "1"
    assert _flag(command, "--paper-min-refs") == "7"
    # The pipeline default for max refs differs from paper.sh's, so it is forwarded.
    assert _flag(command, "--paper-max-refs") == "30"
    assert "--paper-review-rounds" not in command
    assert command.index("--paper-profile") < command.index("--standalone-paper")
    assert _flag(command, "--final-outcome-policy") == "qualified_only"


def test_revision_command_flags_match_paper_edit_script(tmp_path: Path) -> None:
    # the generated flags are accepted by paper_edit.sh's argument parser
    cfg = _cfg(tmp_path)
    cfg.paper.set("paper_max_repair_rounds", 3)
    cfg.paper.set("skills", "paper_writer")
    cfg.paper.set("two_stage", True)

    state = apply_flags(_command(cfg, derive_paths(cfg))[2:], _paper_edit_script_text())

    assert state["PAPER_MAX_REPAIR_ROUNDS"] == "3"
    assert state["PAPER_REVIEW_ROUNDS"] == "0"

    # the exclusion list must name exactly the mirrored keys paper_edit.sh lacks
    edit_flags = {arm.long_flag[2:].replace("-", "_") for arm in flag_arms(_paper_edit_script_text())}
    paper_flags = {arm.long_flag[2:].replace("-", "_") for arm in flag_arms()}
    forwardable = paper_flags - PAPER_OWNED_KEYS - PaperConfig.rewritten - REVISION_EXCLUDED_PAPER_KEYS

    assert forwardable <= edit_flags, sorted(forwardable - edit_flags)
    assert (REVISION_EXCLUDED_PAPER_KEYS - edit_flags) == {"contributions_file"}
    assert "paper_review_rounds" in edit_flags


PUBLISHED_EVIDENCE_CASES = [
    ("final_evaluation", {"final_evaluation": {"metrics": {"score": 1.0}}}, "--final-evaluation-dir", "final_dir"),
    ("posthoc", {"posthoc_analyses": [{"analysis_id": "precision"}]}, "--posthoc-dir", "posthoc_dir"),
]


def test_build_revision_command_includes_only_published_evidence(tmp_path: Path) -> None:
    for name, paper_data, flag, attr in PUBLISHED_EVIDENCE_CASES:
        cfg = _cfg(tmp_path / name)
        paths = derive_paths(cfg)
        paths.paper_dir.mkdir(parents=True)
        (paths.paper_dir / "paper_data.json").write_text(json.dumps(paper_data), encoding="utf-8")

        command = _command(cfg, paths)

        assert _flag(command, flag) == str(getattr(paths, attr)), name
        if attr == "final_dir":
            assert "--no-final-evaluation" not in command, name


def test_revision_forwards_pipeline_logging_flags(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.logging.format = "json"
    cfg.logging.verbose = True
    cfg.logging.heartbeat_seconds = 7

    command = build_revision_command(
        cfg=cfg,
        paths=derive_paths(cfg),
        review_path=tmp_path / "review.json",
        output_dir=tmp_path / "revision",
    )

    assert _flag(command, "--event-transport") == "wrapper"
    assert _flag(command, "--log-format") == "json"
    assert _flag(command, "--heartbeat-seconds") == "7"
    assert "--verbose" in command


# --- run_pipeline_revision: promotion, review, failure ----------------------------------


def test_run_pipeline_revision_promotes_then_reviews(tmp_path: Path) -> None:
    cfg, paths = _setup(tmp_path)
    merged_results = paths.paper_dir / "_merged_parallel_test"
    (merged_results / "aggregated").mkdir(parents=True)
    (merged_results / "extra_tasks").mkdir()
    for name, target in (
        ("results", merged_results),
        ("aggregated", merged_results / "aggregated"),
        ("extra_tasks", merged_results / "extra_tasks"),
    ):
        (paths.paper_dir / name).symlink_to(target, target_is_directory=True)
    before_review = _write_before_review(paths.paper_dir)
    calls: list[str] = []
    edit = _editing()

    def execute(command: list[str]) -> tuple[str, int]:
        calls.append("edit")
        return edit(command)

    def review(**kwargs: object) -> FinalAiReviewResult:
        calls.append("review")
        assert kwargs["artifact_stem"] == "ai_review.after_revision"
        assert kwargs["manifest_key"] == "after_revision"
        return _after_review(paths.paper_dir)

    result = run_pipeline_revision(cfg=cfg, paths=paths, execute_command=execute, review_runner=review)

    assert result == RevisionResult(
        status="completed",
        revision_dir=paths.revision_dir / "attempt_01",
        before_review_path=before_review,
        after_review_path=paths.paper_dir / "ai_review.after_revision.json",
        error=None,
    )
    assert calls == ["edit", "review"]
    assert (paths.paper_dir / "paper.tex").read_text(encoding="utf-8") == "\\title{After}\n"
    assert (paths.paper_dir / "results").resolve() == merged_results.resolve()
    assert (paths.paper_dir / "aggregated").resolve() == (merged_results / "aggregated").resolve()


def test_revision_promotes_and_reviews_changed_posthoc_editorial_annotations(tmp_path: Path) -> None:
    cfg, paths = _setup(tmp_path)
    _write_before_review(paths.paper_dir)
    data_path = paths.paper_dir / "paper_data.json"
    source_data = json.loads(data_path.read_text(encoding="utf-8"))
    source_data["posthoc_analyses"][0].update(
        recommended_usage="skip",
        relevance_to_thesis="low",
        key_finding_for_paper="The two metrics preserve ordering.",
        relevance_justification="The broader artifact audit already covers this check.",
        raw_data={"score": 2.5},
    )
    data_path.write_text(json.dumps(source_data), encoding="utf-8")
    refresh_evidence_audit(paths.paper_dir)
    reviews: list[Path] = []

    def execute(command: list[str]) -> tuple[str, int]:
        output_dir = _output_dir(command)
        write_workspace(output_dir, title="Revised")
        candidate_data = json.loads(json.dumps(source_data))
        candidate_data["posthoc_analyses"][0].update(
            recommended_usage="appendix",
            relevance_to_thesis="medium",
            key_finding_for_paper="Saved metric rankings are consistent.",
            relevance_justification="Useful arithmetic assurance for small score differences.",
        )
        (output_dir / "paper_data.json").write_text(json.dumps(candidate_data), encoding="utf-8")
        refresh_evidence_audit(output_dir)
        return "", 0

    def review(**_kwargs: object) -> FinalAiReviewResult:
        reviews.append(paths.paper_dir)
        return _after_review(paths.paper_dir)

    result = run_pipeline_revision(cfg=cfg, paths=paths, execute_command=execute, review_runner=review)

    assert result.status == "completed"
    assert reviews == [paths.paper_dir]
    published_data = json.loads(data_path.read_text(encoding="utf-8"))
    assert published_data["posthoc_analyses"][0]["recommended_usage"] == "appendix"
    assert published_data["posthoc_analyses"][0]["raw_data"] == {"score": 2.5}


def test_revision_attempt_reuse_and_evidence_safety(tmp_path: Path) -> None:
    # a complete, evidence-safe attempt is reused without editing again
    cfg, paths = _setup(tmp_path / "reuse")
    _write_before_review(paths.paper_dir)
    reusable = paths.revision_dir / "attempt_01"
    write_workspace(reusable, title="Reusable")
    calls: list[str] = []

    def review(**_kwargs: object) -> FinalAiReviewResult:
        calls.append("review")
        return _after_review(paths.paper_dir)

    result = run_pipeline_revision(
        cfg=cfg,
        paths=paths,
        execute_command=_must_not("safe attempt must be reused"),
        review_runner=review,
    )

    assert result.status == "completed"
    assert result.revision_dir == reusable
    assert calls == ["review"]
    assert "Reusable" in (paths.paper_dir / "paper.tex").read_text(encoding="utf-8")

    # an evidence-losing attempt is skipped and the next attempt is created
    cfg, paths = _setup(tmp_path / "skip")
    _write_before_review(paths.paper_dir)
    rejected = paths.revision_dir / "attempt_01"
    write_workspace(rejected, title="Evidence-losing")
    _mutate_paper_data(rejected, lambda data: data.update(ablation_studies=[]))
    edited_outputs: list[Path] = []

    def execute(command: list[str]) -> tuple[str, int]:
        edited_outputs.append(_output_dir(command))
        return _editing("Safe retry")(command)

    result = run_pipeline_revision(
        cfg=cfg,
        paths=paths,
        execute_command=execute,
        review_runner=lambda **_kwargs: _after_review(paths.paper_dir),
    )

    assert result.status == "completed"
    assert edited_outputs == [paths.revision_dir / "attempt_02"]
    assert result.revision_dir == paths.revision_dir / "attempt_02"

    # a newly produced evidence-losing attempt is never promoted or reviewed
    cfg, paths = _setup(tmp_path / "unsafe")
    _write_before_review(paths.paper_dir)

    result = run_pipeline_revision(
        cfg=cfg,
        paths=paths,
        execute_command=_editing(
            "Unsafe",
            lambda output: _mutate_paper_data(output, lambda data: data.update(posthoc_analyses=[])),
        ),
        review_runner=_must_not("unsafe revision must not be reviewed"),
    )

    assert result.status == "failed"
    assert result.error is not None and "posthoc inventory" in result.error
    assert "Before" in (paths.paper_dir / "paper.tex").read_text(encoding="utf-8")


def test_run_pipeline_revision_degrades_when_rereview_fails_or_throws(tmp_path: Path) -> None:
    # a failed re-review keeps the promoted paper
    cfg, paths = _setup(tmp_path / "failed")
    _write_before_review(paths.paper_dir)
    failed_review = FinalAiReviewResult(
        status="failed",
        review_path=None,
        log_path=paths.paper_dir / "ai_review.after_revision.log",
        error="review unavailable",
    )

    result = run_pipeline_revision(
        cfg=cfg, paths=paths, execute_command=_editing(), review_runner=lambda **_kwargs: failed_review
    )

    assert result.status == "degraded"
    assert result.error == "review unavailable"
    assert (paths.paper_dir / "paper.tex").read_text(encoding="utf-8") == "\\title{After}\n"
    assert _read_manifest(paths.paper_dir)["revision"]["rereview_status"] == "failed"

    # a throwing re-review is recorded in the manifest
    cfg, paths = _setup(tmp_path / "throwing")
    _write_before_review(paths.paper_dir)

    def throwing_review(**_kwargs: object) -> FinalAiReviewResult:
        raise RuntimeError("review launch failed")

    result = run_pipeline_revision(
        cfg=cfg, paths=paths, execute_command=_editing(), review_runner=throwing_review
    )

    assert result.status == "degraded"
    assert result.error == "review launch failed"
    manifest = _read_manifest(paths.paper_dir)
    assert manifest["revision"]["rereview_status"] == "failed"
    assert manifest["revision"]["rereview_error"] == "review launch failed"


def test_revision_only_archives_legacy_review_under_before_name(tmp_path: Path) -> None:
    cfg, paths = _setup(tmp_path)
    legacy_review = paths.paper_dir / "ai_review.json"
    legacy_review.write_text(json.dumps(review_payload()), encoding="utf-8")
    (paths.paper_dir / "ai_review.log").write_text("review log\n", encoding="utf-8")
    manifest_path = _update_manifest(
        paths.paper_dir,
        ai_review={
            "status": "completed",
            "review_json": str(legacy_review.resolve()),
            "log": str((paths.paper_dir / "ai_review.log").resolve()),
        },
    )

    def execute(command: list[str]) -> tuple[str, int]:
        write_workspace(_output_dir(command), title="After")
        return "paper_edit.sh: PAPER_LOG_FORMAT: unbound variable\n", 2

    result = run_pipeline_revision(
        cfg=cfg, paths=paths, execute_command=execute, review_runner=_must_not("review must not run")
    )

    named_review = paths.paper_dir / "ai_review.before_revision.json"
    assert result.before_review_path == named_review
    assert result.error == (
        "paper revision command failed with exit code 2: "
        "paper_edit.sh: PAPER_LOG_FORMAT: unbound variable"
    )
    assert json.loads(named_review.read_text(encoding="utf-8")) == review_payload()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["before_revision"]["review_json"] == str(named_review.resolve())


FAILURE_DETAIL_PATHS = [
    ("secret", "/tmp/\x1b[31msecret\nOPENAI_API_KEY=sk-abcdefghijklmnopqrstuv\r" + "x" * 2000),
    ("plain", "/tmp/\x1b[31mplain\npath\r" + "x" * 2000),
]


def test_revision_failure_output_is_redacted_and_sanitized(tmp_path: Path) -> None:
    # a failed revision command's output is redacted in the result error
    cfg, paths = _setup(tmp_path)
    _write_before_review(paths.paper_dir)

    result = run_pipeline_revision(
        cfg=cfg,
        paths=paths,
        execute_command=lambda _command: ("API_KEY=sk-proj-1234567890abcdefghijklmnop\n", 1),
        review_runner=_must_not("review must not run"),
    )

    assert result.error is not None
    assert "sk-proj" not in result.error
    assert "<redacted-sensitive-text:" in result.error

    # output is redacted before the retained tail is bounded
    output = retain_revision_output_tail("", "PASSWORD=" + "A" * 1500 + "\n")

    assert "A" not in output
    assert output.startswith("<redacted-sensitive-text:")

    # failure event details redact secrets and sanitize control characters in paths
    for name, path in FAILURE_DETAIL_PATHS:
        detail = revision_failure_event_detail(
            {"gate": "evidence_audit", "paper_fields": {"diagnostic_log": path}}
        )

        assert "\x1b" not in detail, name
        assert "\n" not in detail and "\r" not in detail, name
        assert "sk-abcdefghijklmnopqrstuv" not in detail, name
        assert len(detail) <= 1200, name


# --- run_pipeline_revision: resume of a promoted revision -------------------------------


def _promoted_state(paths, *, with_snapshot: bool = True, after_revision: dict | None = None):
    """Paper already promoted from attempt_01; returns (revision_dir, before_review, snapshot)."""
    revision_dir = paths.revision_dir / "attempt_01"
    write_workspace(revision_dir, title="After")
    snapshot = write_original_snapshot(paths) if with_snapshot else None
    before_review = _write_before_review(paths.paper_dir)
    revision = {"status": "promoted", "source_dir": str(revision_dir)}
    if snapshot is not None:
        revision["original_snapshot"] = str(snapshot)
    if after_revision is not None:
        revision["before_review"] = str(before_review)
        _update_manifest(paths.paper_dir, after_revision=after_revision)
    _update_manifest(paths.paper_dir, revision=revision)
    return revision_dir, before_review, snapshot


RESUME_REREVIEW_CASES = [
    ("failed_rereview", {"status": "failed", "review_json": None}),
    ("missing_after_artifact", {"status": "completed", "review_json": None}),
]


def test_promoted_revision_resume_rereviews_without_editing_and_never_recreates_before_review(
    tmp_path: Path,
) -> None:
    # a promoted revision whose re-review is incomplete is re-reviewed without editing again
    for name, after_revision in RESUME_REREVIEW_CASES:
        cfg, paths = _setup(tmp_path / name, title="After")
        revision_dir, _, _ = _promoted_state(paths, after_revision=after_revision)
        calls: list[str] = []

        def review(**_kwargs: object) -> FinalAiReviewResult:
            calls.append("review")
            after_path = paths.paper_dir / "ai_review.after_revision.json"
            after_path.write_text(json.dumps(review_payload()), encoding="utf-8")
            return _after_review(paths.paper_dir)

        result = run_pipeline_revision(
            cfg=cfg,
            paths=paths,
            execute_command=_must_not("revision must not run again"),
            review_runner=review,
        )

        assert result.status == "completed", name
        assert result.revision_dir == revision_dir, name
        assert calls == ["review"], name

    # a missing immutable before-review is never recreated from the latest review
    cfg, paths = _setup(tmp_path / "missing_before", title="After")
    revision_dir = paths.revision_dir / "attempt_01"
    write_workspace(revision_dir, title="After")
    (paths.paper_dir / "ai_review.json").write_text(json.dumps({"Summary": "after"}), encoding="utf-8")
    _update_manifest(
        paths.paper_dir,
        revision={"status": "promoted", "source_dir": str(revision_dir)},
        before_revision={
            "status": "completed",
            "review_json": str(paths.paper_dir / "ai_review.before_revision.json"),
        },
    )

    with pytest.raises(ValueError, match="immutable initial AI review"):
        run_pipeline_revision(
            cfg=cfg,
            paths=paths,
            execute_command=_must_not("revision must not run"),
            review_runner=_must_not("review must not run"),
        )

    assert not (paths.paper_dir / "ai_review.before_revision.json").exists()


def _corrupt_published_evidence(paths: PipelinePaths, revision_dir: Path, snapshot: Path) -> None:
    _mutate_paper_data(
        paths.paper_dir,
        lambda data: data["grounding_sources"]["result.top_program_01"].update(value=0.0),
    )


RESUME_CORRUPTION_CASES: list[tuple[str, bool, Callable[[PipelinePaths, Path, Path], object], str]] = [
    ("missing_original_snapshot", False, lambda *_: None, "original_snapshot"),
    (
        "candidate_corruption",
        True,
        lambda paths, revision_dir, snapshot: _mutate_paper_data(
            revision_dir,
            lambda data: data["grounding_sources"]["result.top_program_01"].update(value=0.0),
        ),
        "grounding inventory",
    ),
    (
        "invalid_original_snapshot",
        True,
        lambda paths, revision_dir, snapshot: (snapshot / "paper_data.json").write_text(
            "[]", encoding="utf-8"
        ),
        "source paper data must contain a JSON object",
    ),
    ("published_evidence", True, _corrupt_published_evidence, "grounding inventory"),
    (
        "published_audit",
        True,
        lambda paths, *_: (paths.paper_dir / "paper.tex").write_text(
            "\\title{Corrupted}\n", encoding="utf-8"
        ),
        "current passing agentic evidence audit",
    ),
    (
        "published_workspace",
        True,
        lambda paths, *_: (paths.paper_dir / "paper.pdf").unlink(),
        "missing regular file",
    ),
]


def test_promoted_revision_resume_rejects_corruption(tmp_path: Path) -> None:
    for name, with_snapshot, corrupt, expected_error in RESUME_CORRUPTION_CASES:
        cfg, paths = _setup(tmp_path / name, title="After")
        revision_dir, _, snapshot = _promoted_state(paths, with_snapshot=with_snapshot)
        corrupt(paths, revision_dir, snapshot)

        with pytest.raises(ValueError, match=expected_error):
            run_pipeline_revision(
                cfg=cfg,
                paths=paths,
                execute_command=_must_not("revision must not run"),
                review_runner=_must_not("review must not run"),
            )


# --- pipeline CLI: phase ordering and replay -------------------------------------------


def test_pipeline_runs_revision_after_initial_review_and_replays_completed_review(
    monkeypatch, tmp_path: Path
) -> None:
    # the revision phase runs after the named initial review
    run_dir = tmp_path / "fresh" / "run"
    paper_dir = run_dir / "paper"
    write_workspace(paper_dir, title="Before")
    calls: list[str] = []

    def initial_review(**kwargs: object) -> FinalAiReviewResult:
        calls.append("review")
        assert kwargs["artifact_stem"] == "ai_review.before_revision"
        assert kwargs["manifest_key"] == "before_revision"
        return FinalAiReviewResult(
            status="completed",
            review_path=paper_dir / "ai_review.before_revision.json",
            log_path=paper_dir / "ai_review.before_revision.log",
        )

    def revision(**_kwargs: object) -> RevisionResult:
        calls.append("revision")
        return RevisionResult(
            status="completed",
            revision_dir=run_dir / "revision" / "attempt_01",
            before_review_path=paper_dir / "ai_review.before_revision.json",
            after_review_path=paper_dir / "ai_review.after_revision.json",
        )

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.cli._pin_evaluation_images", lambda *_args: None)
        m.setattr("src.pipeline.cli.run_pipeline_review", initial_review)
        m.setattr("src.pipeline.cli.run_pipeline_revision", revision)

        assert main(_cli_args(run_dir, "review,revision")) == 0
    assert calls == ["review", "revision"]

    # a replay reuses the completed initial review before promotion instead of rerunning it
    run_dir = tmp_path / "replay" / "run"
    paper_dir = run_dir / "paper"
    revision_dir = run_dir / "revision" / "attempt_01"
    write_workspace(paper_dir, title="After")
    write_workspace(revision_dir, title="After")
    before_review = _write_before_review(paper_dir)
    _update_manifest(
        paper_dir,
        before_revision={"status": "completed", "review_json": str(before_review.resolve())},
    )
    with monkeypatch.context() as m:
        m.setattr("src.pipeline.cli._pin_evaluation_images", lambda *_args: None)
        m.setattr(
            "src.pipeline.cli.run_pipeline_review",
            _must_not("completed replay must not rerun initial review"),
        )
        m.setattr(
            "src.pipeline.cli.run_pipeline_revision",
            lambda **_kwargs: RevisionResult(
                status="completed",
                revision_dir=revision_dir,
                before_review_path=before_review,
                after_review_path=paper_dir / "ai_review.after_revision.json",
            ),
        )

        assert main(_cli_args(run_dir, "review,revision")) == 0
    assert json.loads(before_review.read_text(encoding="utf-8")) == review_payload()


def test_revision_dry_run_reports_paper_review_revision_order(monkeypatch, caplog, tmp_path: Path) -> None:
    monkeypatch.setattr("src.pipeline.cli.run_glue", lambda *_args, **_kwargs: None)

    with caplog.at_level("INFO"):
        rc = main(
            [
                TASK_ARG,
                "run_id=20260101_000000",
                "execution.phases=investigation,paper,review,revision",
                f"paths.results_root={tmp_path / 'results'}",
                "--dry-run",
            ]
        )

    assert rc == 0
    rendered = "\n".join(record.message for record in caplog.records)
    assert rendered.index("paper.sh") < rendered.index("Review preview")
    assert rendered.index("Review preview") < rendered.index("Revision preview")
    assert "paper_edit.sh" in rendered
    assert "ai_review.after_revision.json" in rendered


# --- pipeline CLI: revision summary ----------------------------------------------------


def _summary_args(run: Path, *extra: str) -> list[str]:
    return _cli_args(run, "revision", "logging.heartbeat_seconds=0", *extra)


def _prepare_cli_revision(run: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    paper = run / "paper"
    write_workspace(paper, title="Before")
    _write_before_review(paper, {**review_payload(), "Overall": 4})
    _update_manifest(
        paper,
        before_revision={
            "status": "completed",
            "review_json": str(paper / "ai_review.before_revision.json"),
        },
    )
    monkeypatch.setattr("src.pipeline.cli._pin_evaluation_images", lambda *_: None)
    monkeypatch.setattr("src.pipeline.cli._pipeline_checkpoint", lambda *_, **__: frozenset())
    return paper


def _events(run: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run / "pipeline_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _revision_result(run: Path, paper: Path, status: RevisionStatus, **kwargs: Any) -> RevisionResult:
    return RevisionResult(
        status=status,
        revision_dir=run / "revision" / "attempt_01",
        before_review_path=paper / "ai_review.before_revision.json",
        after_review_path=paper / "ai_review.after_revision.json",
        **kwargs,
    )


def test_revision_comparison_prints_once_for_fresh_completion_and_cached_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    paper = _prepare_cli_revision(run, monkeypatch)
    calls: list[str] = []

    def edit(command: list[str], **_kwargs: object) -> int:
        calls.append("edit")
        write_workspace(_output_dir(command), title="After")
        return 0

    def review(**kwargs: object) -> FinalAiReviewResult:
        calls.append("review")
        assert kwargs["artifact_stem"] == "ai_review.after_revision"
        after = paper / "ai_review.after_revision.json"
        after.write_text(json.dumps({**review_payload(), "Overall": 3}), encoding="utf-8")
        _update_manifest(paper, after_revision={"status": "completed", "review_json": str(after)})
        return _after_review(paper)

    monkeypatch.setattr("src.pipeline.cli.run_cmd", edit)
    monkeypatch.setattr("src.pipeline.cli.run_pipeline_review", review)
    for _ in range(2):
        assert main(_summary_args(run)) == 0
        output = capsys.readouterr().out
        assert output.count("Revision review ·") == 1
        assert "overall 4/10 → 3/10 (-1) · decision REJECT → REJECT" in output
    assert calls == ["edit", "review"]
    summaries = [event["fields"] for event in _events(run) if event["event"] == "revision_result"]
    assert len(summaries) == 2
    for fields in summaries:
        assert fields["before_overall_score"] == 4
        assert fields["after_overall_score"] == 3
        assert fields["before_decision"] == fields["after_decision"] == "Reject"
        assert fields["overall_score_delta"] == -1
        assert fields["revision_dir"] == str(run / "revision" / "attempt_01")
        assert fields["before_review_path"] == str(paper / "ai_review.before_revision.json")
        assert fields["after_review_path"] == str(paper / "ai_review.after_revision.json")


def test_revision_summary_hides_stale_or_dry_run_comparison_and_tolerates_missing_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # an unsuccessful revision never displays a stale after-review
    for status in ("failed", "degraded"):
        run = tmp_path / status / "run"
        with monkeypatch.context() as m:
            paper = _prepare_cli_revision(run, m)
            (paper / "ai_review.after_revision.json").write_text(
                json.dumps({"Overall": 10, "Decision": "Accept"}), encoding="utf-8"
            )
            result = _revision_result(run, paper, status, error="injected review failure")
            m.setattr("src.pipeline.cli.run_pipeline_revision", lambda **_: result)

            assert main(_summary_args(run)) == 0, status

        assert "Revision review ·" not in capsys.readouterr().out, status
        events = _events(run)
        assert not any(event["event"] == "revision_result" for event in events), status
        assert any(
            event["event"] == "phase_end" and event["fields"]["status"] == "degraded" for event in events
        ), status

    # a completed revision with missing summary fields remains successful
    run = tmp_path / "missing" / "run"
    with monkeypatch.context() as m:
        paper = _prepare_cli_revision(run, m)
        (paper / "ai_review.after_revision.json").write_text("{}", encoding="utf-8")
        result = _revision_result(run, paper, "completed")
        m.setattr("src.pipeline.cli.run_pipeline_revision", lambda **_: result)

        assert main(_summary_args(run)) == 0

    assert "overall 4/10 → unavailable · decision REJECT → unavailable" in capsys.readouterr().out
    summary = next(event for event in _events(run) if event["event"] == "revision_result")
    assert summary["fields"]["after_overall_score"] is None
    assert summary["fields"]["overall_score_delta"] is None

    # a dry run launches no revision and prints no comparison
    run = tmp_path / "dry" / "run"
    with monkeypatch.context() as m:
        _prepare_cli_revision(run, m)
        m.setattr("src.pipeline.cli.run_pipeline_revision", _must_not("dry run launched revision"))

        assert main(_summary_args(run, "--dry-run")) == 0

    assert "Revision review ·" not in capsys.readouterr().out


def test_pipeline_revision_failed_event_reaches_revision_result(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    paper_dir = run_dir / "paper"
    write_workspace(paper_dir, title="Before")
    _write_before_review(paper_dir)
    _update_manifest(
        paper_dir,
        before_revision={
            "status": "completed",
            "review_json": str(paper_dir / "ai_review.before_revision.json"),
        },
    )
    results: list[RevisionResult] = []

    def execute(command, **kwargs):
        event = build_record(
            gate="evidence_audit",
            state="failed",
            session_id="test",
            fields={"diagnostic_log": str(_output_dir(command) / "paper_diagnostics.log")},
        )
        kwargs["output_line_fn"](TRANSPORT_PREFIX + json.dumps(event))
        return 17

    def revision(**kwargs):
        result = run_pipeline_revision(**kwargs)
        results.append(result)
        return result

    monkeypatch.setattr("src.pipeline.cli._pin_evaluation_images", lambda *_args: None)
    monkeypatch.setattr("src.pipeline.cli.run_cmd", execute)
    monkeypatch.setattr("src.pipeline.cli.run_pipeline_revision", revision)
    main(_cli_args(run_dir, "revision"))

    assert len(results) == 1
    assert results[0].status == "failed"
    error = results[0].error
    assert error is not None
    assert "Evidence Audit" in error
    assert "paper_diagnostics.log" in error
    assert "code 17" in error


# --- review summary rendering ----------------------------------------------------------


def _write_review(path: Path, score: object, decision: object) -> Path:
    path.write_text(json.dumps({"Overall": score, "Decision": decision}), encoding="utf-8")
    return path


def _logger(tmp_path: Path, log_format: str) -> tuple[EventLogger, io.StringIO]:
    stream = io.StringIO()
    logger = EventLogger(
        log_format=log_format,
        verbose=False,
        heartbeat_seconds=0,
        results_dir=tmp_path,
        worker_index=0,
        stream=stream,
        start_progress_thread=False,
    )
    return logger, stream


COMPARISON_RENDER_CASES = [
    (4, 3, "Reject", "overall 4/10 → 3/10 (-1) · decision REJECT → REJECT"),
    (4, 7, "Accept", "overall 4/10 → 7/10 (+3) · decision REJECT → ACCEPT"),
    (4, 4, "Reject", "overall 4/10 → 4/10 (+0) · decision REJECT → REJECT"),
    (4.5, 5, "Accept", "overall 4.5/10 → 5/10 (+0.5) · decision REJECT → ACCEPT"),
]


def test_revision_review_comparison_renders_human_and_json_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the human renderer prints the comparison without colour
    with monkeypatch.context() as m:
        m.setenv("NO_COLOR", "1")
        m.setenv("FORCE_COLOR", "1")
        for before, after, decision, expected in COMPARISON_RENDER_CASES:
            fields = revision_review_fields(
                _write_review(tmp_path / "before.json", before, "Reject"),
                _write_review(tmp_path / "after.json", after, decision),
            )
            logger, stream = _logger(tmp_path, "human")
            logger.emit("revision_result", **fields)
            assert stream.getvalue().strip() == f"Revision review · {expected}", expected
            assert "\x1b" not in stream.getvalue(), expected

    # the json event keeps the numeric comparison
    fields = revision_review_fields(
        _write_review(tmp_path / "before.json", 4, "Reject"),
        _write_review(tmp_path / "after.json", 3, "Reject"),
    )
    logger, stream = _logger(tmp_path, "json")
    logger.emit("revision_result", **fields)
    assert json.loads(stream.getvalue())["fields"] == {
        "before_overall_score": 4,
        "after_overall_score": 3,
        "before_decision": "Reject",
        "after_decision": "Reject",
        "overall_score_delta": -1,
    }

    # the renderer handles a legacy event and an invalid decision
    logger, stream = _logger(tmp_path, "human")
    logger.emit("revision_result", after_overall_score=7, after_decision=["\x1b[31mAccept\n"])
    assert stream.getvalue().strip() == (
        "Revision review · overall unavailable → 7/10 · decision unavailable → unavailable"
    )


INVALID_SCORES: list[object] = [None, True, "7", 0, 11, float("nan"), float("inf"), {}, []]
UNUSABLE_ARTIFACTS = [
    ("malformed", b"{"),
    ("invalid_encoding", b"\xff"),
    ("array", b"[]"),
    ("missing_fields", b"{}"),
    ("too_deep", b"[" * 2000),
]


def test_revision_summary_tolerates_invalid_scores_unusable_and_unsafe_artifacts(
    tmp_path: Path,
) -> None:
    # invalid scores are dropped while the decision is kept
    for score in INVALID_SCORES:
        fields = revision_review_fields(None, _write_review(tmp_path / "after.json", score, "Accept"))
        assert fields["after_overall_score"] is None, repr(score)
        assert fields["overall_score_delta"] is None, repr(score)
        assert fields["after_decision"] == "Accept", repr(score)

    # unusable artifacts yield no fields at all
    for name, payload in UNUSABLE_ARTIFACTS:
        after = tmp_path / f"{name}.json"
        after.write_bytes(payload)
        fields = revision_review_fields(tmp_path / "missing.json", after)
        assert all(value is None for value in fields.values()), name

    # unsafe artifacts are never read
    for unsafe in ("file_symlink", "parent_symlink", "fifo", "oversize"):
        case_dir = tmp_path / unsafe
        case_dir.mkdir()
        safe = _write_review(case_dir / "safe.json", 9, "Accept")
        path = case_dir / "unsafe.json"
        if unsafe == "file_symlink":
            path.symlink_to(safe)
        elif unsafe == "parent_symlink":
            parent = case_dir / "linked"
            parent.symlink_to(case_dir, target_is_directory=True)
            path = parent / "safe.json"
        elif unsafe == "fifo":
            os.mkfifo(path)
        else:
            path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
        assert all(value is None for value in revision_review_fields(None, path).values()), unsafe


# --- shell progress / logging path -----------------------------------------------------


def _progress_env(tmp_path: Path, **overrides: str | None) -> dict[str, str]:
    env: dict[str, str | None] = {
        **os.environ,
        "REPO_ROOT": str(REPO_ROOT),
        "OUTPUT_DIR": str(tmp_path),
        "PYTHON_BIN": sys.executable,
        "TIMESTAMP": "test",
        "PAPER_LOG_FORMAT": "human",
        "PAPER_LOG_VERBOSE": "0",
        "PAPER_HEARTBEAT_SECONDS": "0",
        "PAPER_EVENT_TRANSPORT": "direct",
        "NO_COLOR": "1",
    }
    env.update(overrides)
    return {key: value for key, value in env.items() if value is not None}


def _run_bash(script: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10, env=env)


def _paper_events(tmp_path: Path) -> list[dict]:
    return [
        json.loads(row)
        for row in next(tmp_path.glob("paper_events_*.jsonl")).read_text().splitlines()
    ]


def run_progress(tmp_path, *, transport="direct", output_format="human", exit_code=0, environment=None):
    script = f'''
set -euo pipefail
source "{PROGRESS}"
paper_progress_initialize
trap paper_event_early_exit EXIT
paper_gate_start repair round=2 phase=revision
echo noisy-private-agent-output
if [ "$TEST_EXIT_CODE" -eq 0 ]; then
  paper_gate_complete repair round=2 phase=revision
fi
exit "$TEST_EXIT_CODE"
'''
    env = _progress_env(
        tmp_path,
        PAPER_LOG_FORMAT=output_format,
        PAPER_EVENT_TRANSPORT=transport,
        TEST_EXIT_CODE=str(exit_code),
        **(environment or {}),
    )
    return _run_bash(script, env)


def _case_dir(tmp_path: Path, name: str) -> Path:
    case_dir = tmp_path / name
    case_dir.mkdir()
    return case_dir


def test_standalone_revision_progress_hides_diagnostics_and_wrapper_transport_forwards_events(
    tmp_path,
):
    # standalone progress hides private diagnostics in both output formats
    for output_format in ("human", "json"):
        case_dir = _case_dir(tmp_path, output_format)
        result = run_progress(case_dir, output_format=output_format)
        assert result.returncode == 0, result.stderr
        assert "noisy-private" not in result.stdout, output_format
        assert "\x1b[" not in result.stdout, output_format
        logs = list(case_dir.glob("paper_diagnostics_*.log"))
        assert "noisy-private-agent-output" in logs[0].read_text(), output_format
        events = _paper_events(case_dir)
        assert [event["state"] for event in events] == ["started", "completed"], output_format
        assert events[-1]["fields"]["elapsed_seconds"] >= 0, output_format
        if output_format == "json":
            assert len([json.loads(row) for row in result.stdout.splitlines()]) == 2
        else:
            assert "Repair" in result.stdout and "completed" in result.stdout
            assert re.search(r"completed.*[0-9]+\.[0-9]s", result.stdout)

    # the wrapper transport is consumed by the existing pipeline forwarder
    result = run_progress(_case_dir(tmp_path, "wrapper"), transport="wrapper")
    events = []
    retained = forward_paper_output(
        result.stdout, event_fn=lambda name, **fields: events.append((name, fields))
    )
    assert retained == ""
    assert [fields["state"] for _, fields in events] == ["started", "completed"]
    assert events[-1][1]["paper_fields"]["round"] == 2


def test_revision_progress_preserves_failure_and_cancellation(tmp_path):
    for exit_code in (1, 130):
        case_dir = _case_dir(tmp_path, str(exit_code))
        result = run_progress(case_dir, exit_code=exit_code)
        assert result.returncode == exit_code
        events = _paper_events(case_dir)
        assert events[-1]["state"] == "failed", exit_code
        assert events[-1]["fields"]["exit_code"] == exit_code


def shell_function(name):
    source = (REPO_ROOT / "src/scripts/paper_edit.sh").read_text()
    return name + "() {" + source.split(name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}"


def test_paper_entrypoints_share_progress_document_logging_flags_and_close_trusted_fd():
    # both paper entrypoints source the shared progress library and accept the logging flags
    for name in ("paper.sh", "paper_edit.sh"):
        source = (REPO_ROOT / "src/scripts" / name).read_text()
        assert 'source "$SCRIPT_DIR/lib/paper_progress.sh"' in source, name
        assert "paper_progress_initialize" in source, name
        for flag in ("--log-format)", "--verbose)", "--heartbeat-seconds)", "--event-transport)"):
            assert flag in source, (name, flag)

    # paper_edit.sh --help documents the logging flags
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "src/scripts/paper_edit.sh"), "--help"],
        text=True,
        capture_output=True,
        timeout=5,
        env=os.environ | {"WRAPPER_PYTHON": sys.executable},
    )
    assert result.returncode == 0
    for flag in ("--log-format", "--verbose", "--heartbeat-seconds", "--event-transport"):
        assert flag in result.stdout, flag

    # the read-only audit stage closes the trusted event descriptor before running the agent
    audit = shell_function("run_readonly_evidence_audit_stage")
    assert audit.index("exec 3>&-") < audit.index("run_agent_prompt")


COLOR_POLICY_CASES = [("human", None, True), ("human", "1", False), ("json", None, False)]


def test_revision_color_policy(tmp_path):
    for index, (output_format, no_color, expected) in enumerate(COLOR_POLICY_CASES):
        result = run_progress(
            _case_dir(tmp_path, f"case{index}"),
            output_format=output_format,
            environment={"FORCE_COLOR": "1", "TERM": "xterm", "NO_COLOR": no_color},
        )
        assert ("\x1b[" in result.stdout) is expected, (output_format, no_color)


def test_revision_late_host_failure_keeps_one_specific_failure(tmp_path):
    script = f'''
set -euo pipefail
source "{PROGRESS}"
paper_progress_initialize
trap paper_event_early_exit EXIT
paper_gate_start references phase=revision_agent
if [ "$TEST_LAST_STATE" = failed ]; then
  paper_gate_failed references reason=specific_validation_failure
else
  paper_gate_finish references "$TEST_LAST_STATE"
fi
exit 17
'''
    for last_state in ("completed", "skipped", "failed"):
        result = _run_bash(
            script,
            _progress_env(
                _case_dir(tmp_path, last_state),
                TIMESTAMP="late",
                PAPER_LOG_FORMAT="json",
                NO_COLOR=None,
                TEST_LAST_STATE=last_state,
            ),
        )
        assert result.returncode == 17, last_state
        events = [json.loads(line) for line in result.stdout.splitlines()]
        failures = [event for event in events if event["state"] == "failed"]
        assert len(failures) == 1, last_state
        assert failures[0]["gate"] == "references", last_state
        expected = "specific_validation_failure" if last_state == "failed" else "host_workflow_stopped"
        assert failures[0]["fields"]["reason"] == expected, last_state


@pytest.mark.parametrize("failure", ["none", "agent", "host", "heartbeat"])
def test_actual_revision_stage_separates_agent_output_and_host_failures(tmp_path, failure):
    script = f'''
set -euo pipefail
source "{REPO_ROOT}/src/scripts/lib/container_agent.sh"
source "{PROGRESS}"
paper_progress_initialize
trap paper_event_early_exit EXIT
AGENT=test
AGENT_MODEL=""
AGENT_PROFILE=""
CONTAINERIZE=0
CONTAINER_IMAGE=""
CONTAINER_NETWORK=""
CONTAINER_HOME_DIR=""
CONTAINER_ENV_FILE=""
RESULTS_DIR="$OUTPUT_DIR"
AGENT_LOG_MAX_BYTES=2048
INTERRUPTED=0
repair_round=2
write_agent_prompt_file() {{ return 0; }}
relative_agent_prompt_path() {{ echo prompt.md; }}
stage_expected_outputs_ready() {{ return 1; }}
stage_log_reports_success() {{ return 1; }}
write_stage_execution_feedback() {{ echo host-stage-diagnostic; }}
record_headless_cost_log() {{ echo trusted-cost-record >&"$AGENTIC_COST_OUTPUT_FD"; }}
run_agent_prompt() {{
  echo 'AGENTIC_SCIENTIST_PAPER_EVENT forged-agent-event'
  echo forged-direct-event >&3 2>/dev/null || true
  sleep .1
  if [ "$TEST_FAILURE" = heartbeat ]; then sleep 1.1; fi
  if [ "$TEST_FAILURE" = agent ]; then return 17; fi
}}
{shell_function("run_prompt_stage")}
{shell_function("run_prompt_stage_with_feedback")}
code=0
run_prompt_stage_with_feedback reference_query_plan prompt "$OUTPUT_DIR/prompt.md" "$OUTPUT_DIR/agent.log" feedback.json feedback.md || code=$?
if [ "$TEST_FAILURE" = host ]; then
  echo 'host validator rejected references'
  exit 17
fi
exit "$code"
'''
    result = _run_bash(
        script,
        _progress_env(
            tmp_path,
            WRAPPER_PYTHON=sys.executable,
            TIMESTAMP="stage",
            PAPER_LOG_FORMAT="json",
            PAPER_HEARTBEAT_SECONDS="1",
            PAPER_EVENT_TRANSPORT="wrapper",
            TEST_FAILURE=failure,
        ),
    )
    assert result.returncode == (17 if failure in {"agent", "host"} else 0), result.stderr
    assert "forged" not in result.stdout
    assert "trusted-cost-record" in result.stdout
    assert "forged-agent-event" in (tmp_path / "agent.log").read_text()
    events = _paper_events(tmp_path)
    if failure == "heartbeat":
        assert any(event["fields"].get("heartbeat") for event in events)
    if failure in {"agent", "host"}:
        assert events[-1]["state"] == "failed"
        assert events[-1]["gate"] == "references"
        assert events[-1]["fields"]["diagnostic_log"].startswith(str(tmp_path))
        detail = revision_failure_event_detail(
            {"gate": events[-1]["gate"], "paper_fields": events[-1]["fields"]}
        )
        assert "References" in detail and "diagnostics" in detail
        assert "forged" not in detail


# --- manifest rebasing -----------------------------------------------------------------


def test_rebase_manifest_normalizes_contained_path_and_rejects_escapes(tmp_path: Path) -> None:
    # a contained path is normalized and rebased
    old_root = tmp_path / "contained" / "revision"
    new_root = tmp_path / "contained" / "paper"
    (old_root / "nested").mkdir(parents=True)
    new_root.mkdir()

    rebased = rebase_manifest(
        {"artifact": str(old_root / "nested" / "." / "artifact.txt")},
        old_root=old_root,
        new_root=new_root,
    )

    assert rebased == {"artifact": str(new_root / "nested" / "artifact.txt")}

    # lexical and symlink-resolved escapes are rejected
    for escape in ("lexical_parent", "symlink_resolved"):
        case_dir = tmp_path / escape
        old_root = case_dir / "revision"
        new_root = case_dir / "paper"
        old_root.mkdir(parents=True)
        new_root.mkdir()
        if escape == "lexical_parent":
            artifact = old_root / ".." / "outside.txt"
        else:
            outside = case_dir / "outside"
            outside.mkdir()
            (old_root / "linked").symlink_to(outside, target_is_directory=True)
            artifact = old_root / "linked" / "artifact.txt"

        with pytest.raises(ValueError, match="escapes revision root"):
            rebase_manifest({"artifact": str(artifact)}, old_root=old_root, new_root=new_root)
