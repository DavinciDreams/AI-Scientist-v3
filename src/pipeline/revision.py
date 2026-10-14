"""Review-driven manuscript revision and safe publication helpers."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from src.orchestrator.paper_manifest import latest_manifest_lock, write_manifest_atomically
from src.investigation.checkpoint_security import redact_sensitive_text
from src.investigation.final_ai_review import FinalAiReviewResult
from src.investigation.paper_events import paper_gate_label

from .config import PipelineConfig
from .paths import PipelinePaths
from .phases import _container_flags, _paper_logging_flags, _venue_flags, paper_flags_for
from .revision_publication import (
    _atomic_copy_file,
    _read_regular,
    _validate_revision_workspace,
    initial_review_must_be_preserved,
    promote_revision_workspace,
    recover_incomplete_promotion,
)
from .revision_evidence import validate_published_revision_evidence, validate_revision_evidence

_MAX_REVIEW_BYTES = 1024 * 1024
_FEEDBACK_FIELDS = {
    "Summary": "summary",
    "Strengths": "strengths",
    "Weaknesses": "weaknesses",
    "Questions": "questions",
    "Limitations": "limitations",
}
RevisionStatus = Literal["completed", "degraded", "failed"]
ExecuteRevisionCommand = Callable[[list[str]], tuple[str, int]]
_MAX_FAILURE_DETAIL_CHARS = 1200


def _workspace_alias_targets(paths: PipelinePaths) -> dict[str, tuple[Path, ...]]:
    results_targets = [paths.investigation_dir]
    try:
        paper_entries = list(paths.paper_dir.iterdir())
    except OSError:
        paper_entries = []
    for entry in paper_entries:
        try:
            mode = entry.lstat().st_mode
        except OSError:
            continue
        if entry.name.startswith("_merged_parallel_") and stat.S_ISDIR(mode):
            results_targets.append(entry)
    aggregate_targets = tuple(target / "aggregated" for target in results_targets)
    extra_task_targets = tuple(target / "extra_tasks" for target in results_targets)
    return {
        "ablations": (paths.ablations_dir,),
        "aggregated": (paths.ablations_dir / "aggregated", *aggregate_targets),
        "extra_tasks": (paths.ablations_dir / "extra_tasks", *extra_task_targets),
        "global_memory": (paths.investigation_dir,),
        "posthoc": (paths.posthoc_dir,),
        "results": tuple(results_targets),
        "task": (paths.task_dir,),
    }


def retain_revision_output_tail(current: str, line: str) -> str:
    """Append one redacted child-output line to the bounded diagnostic tail."""
    redacted = redact_sensitive_text(line.rstrip("\r\n"))
    return (current + redacted + "\n")[-_MAX_FAILURE_DETAIL_CHARS:]


def revision_failure_event_detail(fields: dict[str, object]) -> str:
    """Summarize trusted host progress metadata without opening agent diagnostics."""
    detail = f"{paper_gate_label(fields.get('gate', 'revision'), fields)} failed"
    nested = fields.get("paper_fields")
    diagnostic_log = nested.get("diagnostic_log") if isinstance(nested, dict) else None
    if isinstance(diagnostic_log, str) and diagnostic_log:
        detail += f"; diagnostics: {diagnostic_log}"
    return redact_sensitive_text(detail)[:_MAX_FAILURE_DETAIL_CHARS]


@dataclass(frozen=True)
class RevisionResult:
    status: RevisionStatus
    revision_dir: Path | None
    before_review_path: Path
    after_review_path: Path | None
    error: str | None = None


def describe_pipeline_revision(*, cfg: PipelineConfig, paths: PipelinePaths) -> str:
    """Describe the managed edit and re-review without mutating artifacts."""
    command = build_revision_command(
        cfg=cfg,
        paths=paths,
        review_path=paths.paper_dir / "ai_review.before_revision.json",
        output_dir=paths.revision_dir / "attempt_01",
    )
    return (
        f"managed review-driven paper edit ({' '.join(command[:2])}) followed by "
        "ai_review.after_revision.json; existing evidence only"
    )


def load_revision_feedback(path: Path) -> dict[str, object]:
    """Load the bounded subset of an external review used as revision feedback."""
    raw = _read_regular(path, max_bytes=_MAX_REVIEW_BYTES)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid AI review JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("AI review must contain a JSON object")

    feedback: dict[str, object] = {}
    for source_name, target_name in _FEEDBACK_FIELDS.items():
        value = payload.get(source_name, "" if source_name == "Summary" else [])
        if source_name == "Summary":
            if not isinstance(value, str):
                raise ValueError("Summary must be a string")
            feedback[target_name] = value
            continue
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{source_name} must be a list of strings")
        feedback[target_name] = value
    return feedback


#: Mirrored paper.sh keys the revision command never forwards to paper_edit.sh.
#: The revision is already driven by one external review, so paper.sh's own
#: review/repair cycles do not apply; ``contributions_file`` is a drafting-only
#: flag paper_edit.sh does not accept. Every other shared key (repair budget,
#: profile, format, reference limits, skills, ...) reaches both scripts alike.
REVISION_EXCLUDED_PAPER_KEYS: frozenset[str] = frozenset(
    {"paper_review_rounds", "contributions_file"}
)


def build_revision_command(
    *,
    cfg: PipelineConfig,
    paths: PipelinePaths,
    review_path: Path,
    output_dir: Path,
) -> list[str]:
    """Build the existing-evidence-only paper edit command."""
    command = [
        "bash",
        "src/scripts/paper_edit.sh",
        "--agent",
        cfg.agent.name,
    ]
    if cfg.agent.model:
        command += ["--agent-model", cfg.agent.model]
    if cfg.agent.profile is not None:
        command += ["--agent-profile", cfg.agent.profile]
    if cfg.agent.reasoning_effort is not None:
        command += ["--reasoning-effort", cfg.agent.reasoning_effort]
    command += [
        "--task-dir",
        str(paths.task_dir),
        "--results-dir",
        str(paths.investigation_dir),
        "--investigation-results-dir",
        str(paths.investigation_dir),
        "--contract-task-dir",
        str(paths.task_dir),
        "--ablations-dir",
        str(paths.ablations_dir),
        "--global-memory-dir",
        str(paths.investigation_dir),
        "--edit-paper-dir",
        str(paths.paper_dir),
        "--output-dir",
        str(output_dir),
        "--manifest-dir",
        str(output_dir),
        "--external-review",
        str(review_path),
        "--reference-mode",
        "preserve",
        "--figure-refresh-mode",
        "auto",
        "--skip-postdraft-reference-enrichment",
    ]
    command += paper_flags_for(cfg.paper, exclude=REVISION_EXCLUDED_PAPER_KEYS)
    command += ["--final-outcome-policy", cfg.paper.final_outcome_policy]
    source_paper_data = _load_source_paper_data(paths.paper_dir)
    if source_paper_data.get("posthoc_analyses") not in (None, {}, []):
        command += ["--posthoc-dir", str(paths.posthoc_dir)]
    if source_paper_data.get("final_evaluation") not in (None, {}, []):
        command += ["--final-evaluation-dir", str(paths.final_dir)]
    else:
        command.append("--no-final-evaluation")
    command += _venue_flags(cfg)
    command += _paper_logging_flags(cfg)
    command += _container_flags(cfg)
    return command


def _load_source_paper_data(paper_dir: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            _read_regular(paper_dir / "paper_data.json", max_bytes=32 * 1024 * 1024).decode(
                "utf-8"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _initial_review_path(paper_dir: Path) -> Path:
    named = paper_dir / "ai_review.before_revision.json"
    if named.is_file() and not named.is_symlink():
        return named
    if initial_review_must_be_preserved(paper_dir):
        raise ValueError("immutable initial AI review is missing or invalid")
    legacy = paper_dir / "ai_review.json"
    if legacy.is_file() and not legacy.is_symlink():
        named_log = paper_dir / "ai_review.before_revision.log"
        legacy_log = paper_dir / "ai_review.log"
        _atomic_copy_file(legacy, named)
        if legacy_log.is_file() and not legacy_log.is_symlink():
            _atomic_copy_file(legacy_log, named_log)
        manifest_path = paper_dir / "latest_manifest.json"
        try:
            manifest = json.loads(
                _read_regular(manifest_path, max_bytes=1024 * 1024).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("published paper manifest is invalid") from exc
        if not isinstance(manifest, dict):
            raise ValueError("published paper manifest must contain an object")
        manifest["before_revision"] = {
            "status": "completed",
            "review_json": str(named.resolve()),
            "log": str(named_log.resolve()),
        }
        with latest_manifest_lock(paper_dir):
            write_manifest_atomically(manifest_path, manifest)
        return named
    raise ValueError("completed initial AI review is unavailable")


def _next_attempt_dir(revision_root: Path) -> Path:
    for attempt in range(1, 100):
        candidate = revision_root / f"attempt_{attempt:02d}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise ValueError("revision attempt limit reached")


def _reusable_attempt(source_dir: Path, revision_root: Path) -> Path | None:
    try:
        entries = sorted(revision_root.iterdir(), key=lambda path: path.name)
    except FileNotFoundError:
        return None
    except OSError:
        return None
    for candidate in entries:
        if not candidate.name.startswith("attempt_") or len(candidate.name) != len("attempt_00"):
            continue
        if not candidate.name[-2:].isdigit():
            continue
        try:
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            _validate_revision_workspace(candidate)
            validate_revision_evidence(source_dir, candidate)
        except (OSError, UnicodeError, ValueError, RecursionError):
            continue
        return candidate
    return None


def _promoted_original_snapshot(
    paper_dir: Path,
    revision_state: dict[str, object],
) -> Path:
    snapshot_value = revision_state.get("original_snapshot")
    if not isinstance(snapshot_value, str) or not snapshot_value:
        raise ValueError("promoted revision manifest is missing original_snapshot")
    snapshot = Path(os.path.abspath(Path(snapshot_value).expanduser()))
    paper_root = Path(os.path.abspath(paper_dir.expanduser()))
    snapshot_root = paper_root / "revisions"
    if not snapshot.is_relative_to(snapshot_root):
        raise ValueError("promoted revision original_snapshot is outside the paper revision root")
    current = snapshot_root
    try:
        if stat.S_ISLNK(paper_root.lstat().st_mode):
            raise ValueError("promoted revision original_snapshot contains a symlink")
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError("promoted revision original_snapshot contains a symlink")
        for part in snapshot.relative_to(snapshot_root).parts:
            current /= part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("promoted revision original_snapshot contains a symlink")
    except FileNotFoundError as exc:
        raise ValueError("promoted revision original_snapshot is missing") from exc
    if not snapshot.is_dir():
        raise ValueError("promoted revision original_snapshot is not a directory")
    return snapshot


def run_pipeline_revision(
    *,
    cfg: PipelineConfig,
    paths: PipelinePaths,
    execute_command: ExecuteRevisionCommand,
    review_runner: Callable[..., FinalAiReviewResult],
) -> RevisionResult:
    """Edit, validate, promote, and re-review the published manuscript."""
    recover_incomplete_promotion(paths.paper_dir)
    before_review_path = _initial_review_path(paths.paper_dir)
    load_revision_feedback(before_review_path)
    manifest_path = paths.paper_dir / "latest_manifest.json"
    try:
        current_manifest = json.loads(
            _read_regular(manifest_path, max_bytes=1024 * 1024).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("published paper manifest is invalid") from exc
    if not isinstance(current_manifest, dict):
        raise ValueError("published paper manifest must contain an object")
    revision_state = current_manifest.get("revision")
    if isinstance(revision_state, dict) and revision_state.get("status") == "promoted":
        original_snapshot = _promoted_original_snapshot(paths.paper_dir, revision_state)
        _validate_revision_workspace(paths.paper_dir)
        validate_published_revision_evidence(original_snapshot, paths.paper_dir)

        source_value = revision_state.get("source_dir")
        if not isinstance(source_value, str):
            raise ValueError("promoted revision manifest is missing source_dir")
        revision_dir = Path(os.path.abspath(Path(source_value).expanduser()))
        revision_root = paths.revision_dir.expanduser().resolve()
        if not revision_dir.is_relative_to(revision_root):
            raise ValueError("promoted revision source is outside the revision root")
        _validate_revision_workspace(revision_dir)
        validate_revision_evidence(original_snapshot, revision_dir)
        after_state = current_manifest.get("after_revision")
        if isinstance(after_state, dict) and after_state.get("status") == "completed":
            after_value = after_state.get("review_json")
            expected_after = (paths.paper_dir / "ai_review.after_revision.json").resolve()
            if isinstance(after_value, str) and Path(after_value).resolve() == expected_after:
                try:
                    after_payload = json.loads(
                        _read_regular(expected_after, max_bytes=_MAX_REVIEW_BYTES).decode("utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    after_payload = None
                if isinstance(after_payload, dict):
                    return RevisionResult(
                        status="completed",
                        revision_dir=revision_dir,
                        before_review_path=before_review_path,
                        after_review_path=expected_after,
                    )
        return _run_after_revision_review(
            cfg=cfg,
            paths=paths,
            revision_dir=revision_dir,
            before_review_path=before_review_path,
            review_runner=review_runner,
        )

    reusable_dir = _reusable_attempt(paths.paper_dir, paths.revision_dir)
    if reusable_dir is None:
        revision_dir = _next_attempt_dir(paths.revision_dir)
        command = build_revision_command(
            cfg=cfg,
            paths=paths,
            review_path=before_review_path,
            output_dir=revision_dir,
        )
        output_tail, return_code = execute_command(command)
        if return_code != 0:
            detail = _last_failure_line(output_tail)
            error = f"paper revision command failed with exit code {return_code}"
            if detail:
                error = f"{error}: {detail}"
            return RevisionResult(
                status="failed",
                revision_dir=revision_dir,
                before_review_path=before_review_path,
                after_review_path=None,
                error=error,
            )
        try:
            _validate_revision_workspace(revision_dir)
            validate_revision_evidence(paths.paper_dir, revision_dir)
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            return RevisionResult(
                status="failed",
                revision_dir=revision_dir,
                before_review_path=before_review_path,
                after_review_path=None,
                error=f"paper revision evidence validation failed: {exc}",
            )
    else:
        revision_dir = reusable_dir

    promote_revision_workspace(
        paper_dir=paths.paper_dir,
        revision_dir=revision_dir,
        before_review_path=before_review_path,
        workspace_alias_targets=_workspace_alias_targets(paths),
    )
    return _run_after_revision_review(
        cfg=cfg,
        paths=paths,
        revision_dir=revision_dir,
        before_review_path=before_review_path,
        review_runner=review_runner,
    )


def _last_failure_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""
    redacted = redact_sensitive_text(lines[-1])
    return redacted[-_MAX_FAILURE_DETAIL_CHARS:]


def _run_after_revision_review(
    *,
    cfg: PipelineConfig,
    paths: PipelinePaths,
    revision_dir: Path,
    before_review_path: Path,
    review_runner: Callable[..., FinalAiReviewResult],
) -> RevisionResult:
    try:
        after_review = review_runner(
            cfg=cfg,
            paths=paths,
            artifact_stem="ai_review.after_revision",
            manifest_key="after_revision",
        )
    except Exception as exc:
        after_review = FinalAiReviewResult(
            status="failed",
            review_path=None,
            log_path=paths.paper_dir / "ai_review.after_revision.log",
            error=str(exc),
        )
    _record_rereview_status(paths.paper_dir, after_review)
    if after_review.status == "failed":
        return RevisionResult(
            status="degraded",
            revision_dir=revision_dir,
            before_review_path=before_review_path,
            after_review_path=None,
            error=after_review.error or "second AI review failed",
        )
    return RevisionResult(
        status="completed",
        revision_dir=revision_dir,
        before_review_path=before_review_path,
        after_review_path=after_review.review_path,
    )


def _record_rereview_status(paper_dir: Path, review: FinalAiReviewResult) -> None:
    manifest_path = paper_dir / "latest_manifest.json"
    with latest_manifest_lock(paper_dir):
        try:
            manifest = json.loads(
                _read_regular(manifest_path, max_bytes=1024 * 1024).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("published paper manifest is invalid") from exc
        revision_state = manifest.get("revision") if isinstance(manifest, dict) else None
        if not isinstance(manifest, dict) or not isinstance(revision_state, dict):
            raise ValueError("published paper manifest is missing revision state")
        revision_state["rereview_status"] = review.status
        revision_state["after_review"] = (
            str(review.review_path.resolve()) if review.review_path is not None else None
        )
        if review.error:
            revision_state["rereview_error"] = review.error
        else:
            revision_state.pop("rereview_error", None)
        write_manifest_atomically(manifest_path, manifest)
