"""Final AI review: ``run_final_ai_review`` core, publication recovery, and the pipeline review phase."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest

from src.final_evaluation.runner import FinalEvaluationError
from src.investigation.final_ai_review import (
    FinalAiReviewResult,
    ReviewStatus,
    final_ai_review_environment,
    read_paper_manifest,
    run_final_ai_review,
)
from src.pipeline import portable_publication
from src.pipeline.cli import main
from src.pipeline.config import PipelineConfig
from src.pipeline.paths import derive_paths
from src.pipeline.portable_publication import replace_directory_portably
from src.pipeline.review import describe_pipeline_review, run_pipeline_review
from tests.revision_test_support import refresh_evidence_audit, write_workspace

Execute = Callable[..., tuple[str, int]]
ACCEPT_REVIEW = '{"Overall": 7, "Decision": "Accept"}\n'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_manifest(manifest_path: Path, paper_pdf: Path) -> None:
    manifest_path.write_text(
        json.dumps({"paper_pdf": str(paper_pdf.resolve())}), encoding="utf-8"
    )


def _repo_root(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    (repo_root / "src").mkdir()
    (repo_root / "src" / "aireview.py").write_text("# reviewer\n", encoding="utf-8")
    return repo_root


class _Env:
    """A nested final paper (results/paper/final_test) plus a reviewer repo root."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.results_dir = tmp_path / "results"
        self.paper_dir = self.results_dir / "paper" / "final_test"
        self.paper_dir.mkdir(parents=True)
        (self.paper_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")
        self.manifest_path = self.results_dir / "paper" / "latest_manifest.json"
        _write_manifest(self.manifest_path, self.paper_dir / "paper.pdf")
        self.repo_root = _repo_root(tmp_path)

    def run(self, execute: Execute, *, uv: str = "uv", **kwargs: Any) -> FinalAiReviewResult:
        return run_final_ai_review(
            repo_root=self.repo_root,
            paper_dir=self.paper_dir,
            results_dir=self.results_dir,
            execute_command=execute,
            uv_executable=uv,
            **kwargs,
        )

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


@pytest.fixture
def env(tmp_path: Path) -> _Env:
    return _Env(tmp_path)


def _reviewer(review: str = ACCEPT_REVIEW, log: str = "review json\n") -> Execute:
    """Execute stub that writes ``review`` to the ``--output`` path and returns ``log``."""

    def execute(command: list[str], *_args: object) -> tuple[str, int]:
        Path(command[command.index("--output") + 1]).write_text(review, encoding="utf-8")
        return log, 0

    return execute


def _reviewed_pdf(command: list[str]) -> Path:
    return Path(command[command.index("--script") + 2])


def _latest_alias_manifest(paper_dir: Path, *, review_json: bool, status: str) -> dict:
    return {
        "log": str((paper_dir / "ai_review.log").resolve()),
        "review_json": str((paper_dir / "ai_review.json").resolve()) if review_json else None,
        "status": status,
    }


class _Recorder:
    executed = False

    def __call__(self, *_args: object, **_kwargs: object) -> tuple[str, int]:
        self.executed = True
        return "", 0


# ---------------------------------------------------------------------------
# run_final_ai_review
# ---------------------------------------------------------------------------

BAD_SUMMARY_CASES = [
    ("invalid-fields", '{"Overall": true, "Decision": "Maybe"}\n', None),
    ("oversized-score", json.dumps({"Overall": 10**1000, "Decision": "Accept"}) + "\n", "Accept"),
]


def test_run_final_ai_review_publishes_artifact_and_keeps_bad_summary_fields_nonfatal(
    tmp_path: Path,
) -> None:
    # publishes the artifact and annotates the manifest
    env = _Env(tmp_path / "publish")
    result = env.run(_reviewer(), uv="/usr/bin/uv")

    assert result.status == "completed"
    assert result.review_path == env.paper_dir / "ai_review.json"
    assert result.overall_score == 7
    assert result.decision == "Accept"
    assert result.log_path.read_text(encoding="utf-8") == "review json\n"
    assert env.manifest()["ai_review"] == _latest_alias_manifest(
        env.paper_dir, review_json=True, status="completed"
    )

    # bad summary fields are non-fatal
    for name, review, expected_decision in BAD_SUMMARY_CASES:
        result = _Env(tmp_path / name).run(_reviewer(review))

        assert result.status == "completed", name
        assert result.overall_score is None, name
        assert result.decision == expected_decision, name


def test_run_final_ai_review_preserves_named_artifacts_and_refreshes_latest_aliases(
    env: _Env,
) -> None:
    reviews = iter(
        [
            ('{"Overall": 6, "Decision": "Reject"}\n', "before log\n"),
            ('{"Overall": 8, "Decision": "Accept"}\n', "after log\n"),
        ]
    )

    def execute(command: list[str], *_args: object) -> tuple[str, int]:
        review, log = next(reviews)
        Path(command[command.index("--output") + 1]).write_text(review, encoding="utf-8")
        return log, 0

    before = env.run(
        execute, artifact_stem="ai_review.before_revision", manifest_key="before_revision"
    )
    after = env.run(
        execute, artifact_stem="ai_review.after_revision", manifest_key="after_revision"
    )

    assert before.review_path == env.paper_dir / "ai_review.before_revision.json"
    assert before.log_path == env.paper_dir / "ai_review.before_revision.log"
    assert after.review_path == env.paper_dir / "ai_review.after_revision.json"
    assert after.log_path == env.paper_dir / "ai_review.after_revision.log"
    assert json.loads(before.review_path.read_text(encoding="utf-8"))["Overall"] == 6
    assert before.log_path.read_text(encoding="utf-8") == "before log\n"
    assert json.loads(after.review_path.read_text(encoding="utf-8"))["Overall"] == 8
    assert after.log_path.read_text(encoding="utf-8") == "after log\n"
    assert (env.paper_dir / "ai_review.json").read_bytes() == after.review_path.read_bytes()
    assert (env.paper_dir / "ai_review.log").read_bytes() == after.log_path.read_bytes()
    manifest = env.manifest()
    assert manifest["before_revision"]["review_json"] == str(before.review_path.resolve())
    assert manifest["after_revision"]["review_json"] == str(after.review_path.resolve())
    assert manifest["ai_review"] == _latest_alias_manifest(
        env.paper_dir, review_json=True, status="completed"
    )


UNSAFE_ARTIFACT_NAMES = [
    ("../review", "before_revision"),
    ("nested/review", "before_revision"),
    (".hidden", "before_revision"),
    ("ai_review..before_revision", "before_revision"),
    ("ai_review.before_revision", "before-revision"),
]


def test_run_final_ai_review_rejects_unsafe_artifact_names_before_writing(env: _Env) -> None:
    for artifact_stem, manifest_key in UNSAFE_ARTIFACT_NAMES:
        with pytest.raises(ValueError, match="artifact_stem|manifest_key"):
            env.run(lambda *_args: ("", 0), artifact_stem=artifact_stem, manifest_key=manifest_key)

        assert not any(env.paper_dir.glob("*.log")), (artifact_stem, manifest_key)
        assert not any(env.paper_dir.glob("*.json")), (artifact_stem, manifest_key)


def test_named_review_failure_refreshes_latest_log_and_clears_stale_latest_json(
    tmp_path: Path,
) -> None:
    # records the named failure and refreshes the latest log alias
    env = _Env(tmp_path / "failure")
    result = env.run(
        lambda *_args: ("review failed upstream\n", 2),
        artifact_stem="ai_review.after_revision",
        manifest_key="after_revision",
    )

    assert result.status == "failed"
    assert result.log_path == env.paper_dir / "ai_review.after_revision.log"
    assert "review failed upstream" in result.log_path.read_text(encoding="utf-8")
    assert (env.paper_dir / "ai_review.log").read_bytes() == result.log_path.read_bytes()
    manifest = env.manifest()
    assert manifest["after_revision"] == {
        "log": str(result.log_path.resolve()),
        "review_json": None,
        "status": "failed",
    }
    assert manifest["ai_review"] == _latest_alias_manifest(
        env.paper_dir, review_json=False, status="failed"
    )

    # a named failure clears the stale latest json left by an earlier named success
    env = _Env(tmp_path / "stale")
    env.run(
        _reviewer('{"Overall": 6, "Decision": "Reject"}\n', "before\n"),
        artifact_stem="ai_review.before_revision",
        manifest_key="before_revision",
    )
    assert (env.paper_dir / "ai_review.json").is_file()

    result = env.run(
        lambda *_args: ("after failed\n", 2),
        artifact_stem="ai_review.after_revision",
        manifest_key="after_revision",
    )

    assert result.status == "failed"
    assert not (env.paper_dir / "ai_review.json").exists()


def test_run_final_ai_review_records_process_failures_as_failed(tmp_path: Path) -> None:
    # a non-zero review process is recorded without raising
    env = _Env(tmp_path / "process")
    result = env.run(lambda *_args: ("request failed\n", 1))

    assert result.status == "failed"
    assert result.review_path is None
    assert "request failed" in result.log_path.read_text(encoding="utf-8")
    assert env.manifest()["ai_review"]["status"] == "failed"
    assert env.manifest()["ai_review"]["review_json"] is None

    # a successful process that did not create ai_review.json
    result = _Env(tmp_path / "missing-json").run(lambda *_args: ("", 0))

    assert result.status == "failed"
    assert result.error == "review command did not create ai_review.json"

    # a missing uv executable
    env = _Env(tmp_path / "missing-uv")
    result = env.run(lambda *_args: ("", 0), uv="")

    assert result.status == "failed"
    assert result.error == "uv executable is not available"
    assert env.manifest()["ai_review"]["status"] == "failed"


def test_run_final_ai_review_replaces_symlinked_artifacts_without_touching_targets(
    tmp_path: Path,
) -> None:
    # replaces a symlinked ai_review.json without writing through it
    env = _Env(tmp_path / "json")
    victim = env.tmp_path / "victim.json"
    victim.write_text('{"safe": true}\n', encoding="utf-8")
    (env.paper_dir / "ai_review.json").symlink_to(victim)

    result = env.run(_reviewer('{"Decision": "Accept"}\n', "reviewed\n"))

    assert result.status == "completed"
    assert victim.read_text(encoding="utf-8") == '{"safe": true}\n'
    assert not (env.paper_dir / "ai_review.json").is_symlink()

    # replaces a symlinked log before recording a failure
    env = _Env(tmp_path / "log")
    victim = env.tmp_path / "victim.log"
    victim.write_text("do not replace\n", encoding="utf-8")
    log_path = env.paper_dir / "ai_review.log"
    log_path.symlink_to(victim)

    result = env.run(lambda *_args: ("", 0), uv="")

    assert result.status == "failed"
    assert victim.read_text(encoding="utf-8") == "do not replace\n"
    assert not log_path.is_symlink()
    assert "uv executable is not available" in log_path.read_text(encoding="utf-8")


def test_run_final_ai_review_guards_against_manifest_and_paper_root_swaps(tmp_path: Path) -> None:
    # does not annotate a manifest that moved on to a newer PDF during the review
    env = _Env(tmp_path / "newer")
    newer_pdf = env.results_dir / "paper" / "final_newer" / "paper.pdf"

    def execute_newer(command: list[str], *_args: object) -> tuple[str, int]:
        Path(command[command.index("--output") + 1]).write_text("{}\n", encoding="utf-8")
        env.manifest_path.write_text(json.dumps({"paper_pdf": str(newer_pdf)}), encoding="utf-8")
        return "{}\n", 0

    result = env.run(execute_newer)

    assert result.status == "failed"
    assert result.error == "latest paper manifest no longer points to the reviewed PDF"
    assert "ai_review" not in env.manifest()
    assert result.error in result.log_path.read_text(encoding="utf-8")

    # stays on the pinned directories when the paper root is swapped for a symlink
    env = _Env(tmp_path / "swap")
    (env.paper_dir / "paper.pdf").write_bytes(b"SAFE-PDF")
    pinned_root = env.results_dir / "paper-pinned"
    outside_root = env.tmp_path / "outside-paper"

    def execute_swap(command: list[str], *_args: object) -> tuple[str, int]:
        (env.results_dir / "paper").rename(pinned_root)
        outside_final = outside_root / env.paper_dir.name
        outside_final.mkdir(parents=True)
        (outside_final / "paper.pdf").write_bytes(b"EVIL-PDF")
        (env.results_dir / "paper").symlink_to(outside_root, target_is_directory=True)
        assert _reviewed_pdf(command).read_bytes() == b"SAFE-PDF"
        Path(command[command.index("--output") + 1]).write_text("{}\n", encoding="utf-8")
        return "reviewed\n", 0

    result = env.run(execute_swap)

    assert result.status == "completed"
    assert (pinned_root / env.paper_dir.name / "ai_review.json").is_file()
    assert not (outside_root / env.paper_dir.name / "ai_review.json").exists()


def test_run_final_ai_review_rejects_oversized_inputs_and_symlinked_paper_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an oversized manifest is rejected before execution
    env = _Env(tmp_path / "manifest")
    env.manifest_path.write_bytes(b" " * (1024 * 1024 + 1))
    execute = _Recorder()

    with pytest.raises(ValueError, match="manifest is too large"):
        env.run(execute)

    assert execute.executed is False

    # an oversized PDF is rejected before execution and leaves no temporary files
    env = _Env(tmp_path / "pdf")
    with (env.paper_dir / "paper.pdf").open("r+b") as stream:
        stream.truncate(100 * 1024 * 1024 + 1)
    temporary_root = env.tmp_path / "review-temp"
    temporary_root.mkdir()
    monkeypatch.setattr("src.investigation.final_ai_review.tempfile.tempdir", str(temporary_root))
    execute = _Recorder()

    result = env.run(execute)

    assert result.status == "failed"
    assert result.error is not None and "PDF exceeds" in result.error
    assert execute.executed is False
    assert list(temporary_root.iterdir()) == []

    # a symlinked paper root is rejected
    root = tmp_path / "symlink"
    results_dir = root / "results"
    results_dir.mkdir(parents=True)
    outside_root = root / "outside-paper"
    paper_dir = outside_root / "final"
    paper_dir.mkdir(parents=True)
    (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    _write_manifest(outside_root / "latest_manifest.json", paper_dir / "paper.pdf")
    (results_dir / "paper").symlink_to(outside_root, target_is_directory=True)

    with pytest.raises(ValueError, match="paper root must not be a symlink"):
        run_final_ai_review(
            repo_root=_repo_root(root),
            paper_dir=results_dir / "paper" / "final",
            results_dir=results_dir,
            execute_command=lambda *_args, **_kwargs: ("", 0),
            uv_executable="uv",
        )


def test_read_paper_manifest_does_not_follow_check_open_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "latest_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    target = tmp_path / "target.json"
    target.write_text('{"paper_pdf": "victim"}\n', encoding="utf-8")
    real_open = os.open

    def swap_before_open(path, flags, *args, dir_fd=None, **kwargs):
        if dir_fd is not None and path == manifest_path.name:
            manifest_path.unlink()
            manifest_path.symlink_to(target)
        return real_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr("src.investigation.final_ai_review.os.open", swap_before_open)

    with pytest.raises(OSError):
        read_paper_manifest(manifest_path)


# ---------------------------------------------------------------------------
# final_ai_review_environment
# ---------------------------------------------------------------------------

_UV_RESOLVER_ENV = {
    "NETRC": "/run/secrets/reviewer.netrc",
    "UV_CONFIG_FILE": "/etc/agentic-scientist/uv.toml",
    "UV_DEFAULT_INDEX": "https://packages.example.test/simple",
    "UV_EXTRA_INDEX_URL": "https://fallback.example.test/simple",
    "UV_FIND_LINKS": "/opt/wheels",
    "UV_INDEX_INTERNAL_PASSWORD": "index-password",
    "UV_INDEX_INTERNAL_USERNAME": "index-user",
    "UV_INDEX_STRATEGY": "first-index",
    "UV_KEYRING_PROVIDER": "subprocess",
    "UV_OFFLINE": "true",
    "UV_SYSTEM_CERTS": "true",
}

ENVIRONMENT_CASES = [
    (
        "excludes-unrelated-credentials",
        {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "openai-secret",
            "GITHUB_TOKEN": "github-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "UV_INDEX_INTERNAL_TOKEN": "unsupported-index-secret",
        },
        {"PATH": "/usr/bin", "OPENAI_API_KEY": "openai-secret"},
    ),
    ("preserves-uv-resolver-settings", _UV_RESOLVER_ENV, _UV_RESOLVER_ENV),
]


def test_final_ai_review_environment_filters_process_env_and_scopes_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # filters the process environment
    for name, environment, expected in ENVIRONMENT_CASES:
        assert final_ai_review_environment(environment) == expected, name

    # loads only OpenAI values from an env file
    env_file = tmp_path / "container.env"
    env_file.write_text(
        'OPENAI_API_KEY="file-openai-secret" # reviewer credential\n'
        "OPENAI_BASE_URL=https://openai.example.test/v1\n"
        "AWS_SECRET_ACCESS_KEY=aws-secret\n"
        "PYTHONPATH=/tmp/untrusted\n",
        encoding="utf-8",
    )

    assert final_ai_review_environment({"PATH": "/usr/bin"}, env_file=env_file) == {
        "OPENAI_API_KEY": "file-openai-secret",
        "OPENAI_BASE_URL": "https://openai.example.test/v1",
        "PATH": "/usr/bin",
    }

    # rejects a symlinked env file
    target = tmp_path / "target.env"
    target.write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    linked_env_file = tmp_path / "linked.env"
    linked_env_file.symlink_to(target)

    with pytest.raises(OSError):
        final_ai_review_environment({}, env_file=linked_env_file)

    # run_final_ai_review loads the scoped env file without forwarding it on the command line
    env = _Env(tmp_path / "review")
    scoped_env_file = env.tmp_path / "container.env"
    scoped_env_file.write_text(
        "OPENAI_API_KEY=secret-value\nAWS_SECRET_ACCESS_KEY=aws-secret\n", encoding="utf-8"
    )
    observed_command: list[str] = []
    observed_environment: dict[str, str] = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def execute(command: list[str], _log: Path, environment: dict[str, str]) -> tuple[str, int]:
        observed_command.extend(command)
        observed_environment.update(environment)
        Path(command[command.index("--output") + 1]).write_text("{}\n", encoding="utf-8")
        return "{}\n", 0

    env.run(execute, env_file=scoped_env_file)

    assert observed_command[:2] == ["uv", "run"]
    assert "--env-file" not in observed_command
    assert "secret-value" not in " ".join(observed_command)
    assert observed_environment["OPENAI_API_KEY"] == "secret-value"
    assert "AWS_SECRET_ACCESS_KEY" not in observed_environment


# ---------------------------------------------------------------------------
# recovery of a pending portable publication before reading the paper
# ---------------------------------------------------------------------------


def _interrupt_publication(
    monkeypatch: pytest.MonkeyPatch, current: Path, replacement: Path
) -> None:
    """Leave a prepared-but-unrenamed ``replace_directory_portably`` behind."""
    real_rename = portable_publication._rename_entry
    monkeypatch.setattr(
        portable_publication,
        "_rename_entry",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("prepared interruption")),
    )
    with pytest.raises(RuntimeError, match="prepared interruption"):
        replace_directory_portably(current, replacement)
    monkeypatch.setattr(portable_publication, "_rename_entry", real_rename)


def _execute_expecting_after(command: list[str], *_args: object) -> tuple[str, int]:
    assert _reviewed_pdf(command).read_bytes().endswith(b"after\n")
    Path(command[command.index("--output") + 1]).write_text(ACCEPT_REVIEW, encoding="utf-8")
    return "reviewed\n", 0


def test_review_recovers_pending_publication_before_reading_paper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # direct review: the paper directory itself is the pending publication
    direct = tmp_path / "direct"
    run_dir = direct / "run"
    run_dir.mkdir(parents=True)
    paper_dir = run_dir / "paper"
    replacement_dir = run_dir / ".paper.revision-review"
    write_workspace(paper_dir, title="Before")
    write_workspace(replacement_dir, title="After")
    (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.4\nbefore\n")
    (replacement_dir / "paper.pdf").write_bytes(b"%PDF-1.4\nafter\n")
    for directory in (paper_dir, replacement_dir):
        _write_manifest(directory / "latest_manifest.json", paper_dir / "paper.pdf")
        refresh_evidence_audit(directory)
    _interrupt_publication(monkeypatch, paper_dir, replacement_dir)

    result = run_final_ai_review(
        repo_root=_repo_root(direct),
        paper_dir=paper_dir,
        paper_root=paper_dir,
        results_dir=run_dir,
        execute_command=_execute_expecting_after,
        uv_executable="uv",
    )

    assert result.status == "completed"
    assert not (run_dir / ".paper.revision-publication.json").exists()

    # nested review: the paper root (parent of the reviewed paper) is the pending publication
    nested = tmp_path / "nested"
    run_dir = nested / "run"
    run_dir.mkdir(parents=True)
    paper_root = run_dir / "paper"
    replacement_root = run_dir / ".paper.revision-review"
    paper_dir = paper_root / "final_test"
    replacement_dir = replacement_root / "final_test"
    write_workspace(paper_root, title="Before")
    write_workspace(replacement_root, title="After")
    for directory, title in ((paper_dir, "before"), (replacement_dir, "after")):
        directory.mkdir()
        (directory / "paper.pdf").write_bytes(f"%PDF-1.4\n{title}\n".encode())
        _write_manifest(directory / "latest_manifest.json", paper_dir / "paper.pdf")
    for root in (paper_root, replacement_root):
        _write_manifest(root / "latest_manifest.json", paper_dir / "paper.pdf")
        refresh_evidence_audit(root)
    _interrupt_publication(monkeypatch, paper_root, replacement_root)

    result = run_final_ai_review(
        repo_root=_repo_root(nested),
        paper_dir=paper_dir,
        paper_root=paper_root,
        results_dir=run_dir,
        execute_command=_execute_expecting_after,
        uv_executable="uv",
    )

    assert result.status == "completed"
    assert not (run_dir / ".paper.revision-publication.json").exists()


# ---------------------------------------------------------------------------
# pipeline review phase (src.pipeline.review + cli integration)
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path) -> tuple[PipelineConfig, Path]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    cfg = PipelineConfig()
    cfg.task.dir = "examples/circpack"
    cfg.run_id = "test-run"
    cfg.paths.run_dir = str(tmp_path / "run")
    cfg.paths.paper_dir = str(tmp_path / "run" / "paper")
    return cfg, repo_root


def _write_published_paper(paper_dir: Path) -> None:
    paper_dir.mkdir(parents=True)
    (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    _write_manifest(paper_dir / "latest_manifest.json", paper_dir / "paper.pdf")


def _write_completed_run(run_dir: Path) -> None:
    task_dir = run_dir / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.md").write_text("# Task\n", encoding="utf-8")
    investigation = run_dir / "investigation"
    program_dir = investigation / "best_program"
    entrypoint = program_dir / "src" / "program.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("print('ok')\n", encoding="utf-8")
    (investigation / "outer_state.json").write_text('{"current_cycle": 1}', encoding="utf-8")
    (investigation / "best_program_manifest.json").write_text(
        json.dumps(
            {
                "program_dir": str(program_dir),
                "entrypoint_path": "src/program.py",
                "combined_score": 1.0,
                "correct": True,
            }
        ),
        encoding="utf-8",
    )


def _no_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called in dry-run")

    monkeypatch.setattr("src.pipeline.phases.subprocess.run", _boom)
    monkeypatch.setattr("src.pipeline.cli.run_glue", lambda *args, **kwargs: None)


class _ReviewSpy:
    calls = 0

    def __call__(self, **_kwargs: object) -> None:
        self.calls += 1
        raise AssertionError("review must not run")


def _review_result(paper_dir: Path, status: ReviewStatus = "completed", **kwargs: Any) -> FinalAiReviewResult:
    return FinalAiReviewResult(
        status=status,
        review_path=None if status == "failed" else paper_dir / "ai_review.json",
        log_path=paper_dir / "ai_review.log",
        **kwargs,
    )


def _review_only_args(tmp_path: Path, paper_dir: Path) -> list[str]:
    return [
        "execution.phases=review",
        f"paths.run_dir={tmp_path / 'bundle'}",
        f"paths.paper_dir={paper_dir}",
    ]


REVIEW_GATE_CASES = [
    ("not-qualified", lambda *_args: "not_qualified", 0, "WARNING", "Skipping review"),
    (
        "unverifiable",
        lambda *_args: (_ for _ in ()).throw(FinalEvaluationError("tampered result")),
        2,
        "ERROR",
        "Could not verify required final evaluation",
    ),
]


@pytest.mark.usefixtures("stub_image_leases")
class TestPipelineReviewPhase:
    def test_run_pipeline_review_describes_and_forwards_configuration(self, tmp_path: Path) -> None:
        # the preview describes the managed publication
        cfg, repo_root = _cfg(tmp_path / "preview")
        paths = derive_paths(cfg)

        description = describe_pipeline_review(cfg=cfg, paths=paths, repo_root=repo_root)

        assert "managed src/aireview.py phase" in description
        assert str(paths.paper_dir / "paper.pdf") in description
        assert str(paths.paper_dir / "ai_review.json") in description
        assert "restricted environment" in description
        assert "manifest publication enabled" in description

        # reuses the secure reviewer with the configured env file
        cfg, repo_root = _cfg(tmp_path / "env")
        cfg.container.env_file = "review.env"
        paths = derive_paths(cfg)
        expected = _review_result(paths.paper_dir)

        with patch("src.pipeline.review.run_final_ai_review", return_value=expected) as review:
            result = run_pipeline_review(cfg=cfg, paths=paths, repo_root=repo_root)

        assert result is expected
        kwargs = review.call_args.kwargs
        assert kwargs["paper_dir"] == paths.paper_dir
        assert kwargs["results_dir"] == paths.run_dir
        assert kwargs["paper_root"] == paths.paper_dir
        assert kwargs["env_file"] == repo_root / "review.env"
        assert callable(kwargs["execute_command"])

        # an empty env file setting disables env-file loading even when .env exists
        cfg, repo_root = _cfg(tmp_path / "empty")
        cfg.container.env_file = ""
        (repo_root / ".env").write_text("OPENAI_API_KEY=test\n", encoding="utf-8")
        paths = derive_paths(cfg)
        expected = _review_result(paths.paper_dir, "failed", error="test")

        with patch("src.pipeline.review.run_final_ai_review", return_value=expected) as review:
            run_pipeline_review(cfg=cfg, paths=paths, repo_root=repo_root)

        assert review.call_args.kwargs["env_file"] is None

        # named artifacts are forwarded
        cfg, repo_root = _cfg(tmp_path / "named")
        paths = derive_paths(cfg)
        expected = FinalAiReviewResult(
            status="completed",
            review_path=paths.paper_dir / "ai_review.before_revision.json",
            log_path=paths.paper_dir / "ai_review.before_revision.log",
        )

        with patch("src.pipeline.review.run_final_ai_review", return_value=expected) as review:
            result = run_pipeline_review(
                cfg=cfg,
                paths=paths,
                repo_root=repo_root,
                artifact_stem="ai_review.before_revision",
                manifest_key="before_revision",
            )

        assert result is expected
        assert review.call_args.kwargs["artifact_stem"] == "ai_review.before_revision"
        assert review.call_args.kwargs["manifest_key"] == "before_revision"

    def test_managed_review_uses_configured_timeouts(self, tmp_path: Path) -> None:
        cfg, repo_root = _cfg(tmp_path)
        cfg.review.timeout_seconds = 900
        cfg.review.timeout_grace_seconds = 15
        paths = derive_paths(cfg)

        def invoke_executor(**kwargs):
            return kwargs["execute_command"](
                ["uv", "run", "review"], paths.paper_dir / "ai_review.log", {"PATH": "/usr/bin"}
            )

        with (
            patch("src.pipeline.review.run_final_ai_review", side_effect=invoke_executor),
            patch(
                "src.pipeline.review.script_runners.execute_managed_command",
                return_value=("ok", 0),
            ) as execute,
        ):
            result = run_pipeline_review(cfg=cfg, paths=paths, repo_root=repo_root)

        assert result == ("ok", 0)
        assert execute.call_args.kwargs["timeout_seconds"] == 900
        assert execute.call_args.kwargs["timeout_grace_seconds"] == 15
        assert execute.call_args.kwargs["env"] == {"PATH": "/usr/bin"}

    def test_dry_run_prints_review_after_paper_and_accepts_explicit_output_outside_run(
        self, monkeypatch, caplog, tmp_path
    ) -> None:
        _no_subprocess(monkeypatch)

        # the full dry run prints the review phase after the paper phase
        with caplog.at_level("INFO"):
            rc = main(
                [
                    "task.dir=examples/circpack",
                    "run_id=20260101_000000",
                    "execution.phases=investigation,ablations,posthoc,paper,review",
                    f"paths.run_dir={tmp_path / 'investigation'}",
                    "--dry-run",
                ]
            )

        assert rc == 0
        logged = "\n".join(record.message for record in caplog.records)
        assert "paper.sh" in logged
        assert "src/aireview.py" in logged
        assert logged.index("paper.sh") < logged.index("src/aireview.py")
        assert "$ uv run --script" not in logged

        # paper + review accept an explicit paper directory outside the run directory
        caplog.clear()
        run_dir = tmp_path / "run"
        _write_completed_run(run_dir)

        with caplog.at_level("INFO"):
            rc = main(
                [
                    "execution.phases=paper,review",
                    f"paths.run_dir={run_dir}",
                    f"paths.paper_dir={tmp_path / 'outside' / 'paper'}",
                    "--dry-run",
                ]
            )

        assert rc == 0

    def test_review_only_failure_is_degraded_success(self, monkeypatch, tmp_path, caplog) -> None:
        for failure in ("failed-result", "runtime-exception"):
            case_dir = tmp_path / failure
            paper_dir = case_dir / "bundle" / "paper"
            _write_published_paper(paper_dir)
            calls: list[Path] = []

            def fake_review(*, paths, _failure=failure, **_kwargs):
                calls.append(paths.paper_dir)
                if _failure == "runtime-exception":
                    raise RuntimeError("review worker crashed")
                return _review_result(paper_dir, "failed", error="OpenAI request failed")

            caplog.clear()
            with monkeypatch.context() as m:
                m.setattr("src.pipeline.cli.run_pipeline_review", fake_review)
                with caplog.at_level("WARNING"):
                    rc = main(_review_only_args(case_dir, paper_dir))

            assert rc == 0, failure
            assert calls == [paper_dir], failure
            assert any("Review phase degraded" in record.message for record in caplog.records), failure

    def test_review_only_gates_on_required_final_evaluation(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        for name, verified_outcome, expected_rc, level, message in REVIEW_GATE_CASES:
            case_dir = tmp_path / name
            paper_dir = case_dir / "bundle" / "paper"
            _write_published_paper(paper_dir)
            spy = _ReviewSpy()

            caplog.clear()
            with monkeypatch.context() as m:
                m.setattr("src.pipeline.cli.task_requires_final_evaluation", lambda *_args: True)
                m.setattr("src.pipeline.cli._verified_final_outcome", verified_outcome)
                m.setattr("src.pipeline.cli.run_pipeline_review", spy)
                with caplog.at_level(level):
                    rc = main(_review_only_args(case_dir, paper_dir))

            assert rc == expected_rc, name
            assert spy.calls == 0, name
            assert any(message in record.message for record in caplog.records), name

    def test_review_runs_only_after_successful_paper(self, monkeypatch, tmp_path) -> None:
        # a successful paper phase is followed by the review, whose result is logged as an event
        bundle = tmp_path / "ok" / "bundle"
        _write_completed_run(bundle)
        paper_dir = bundle / "paper"
        calls: list[str] = []

        def fake_run_cmd(argv, **_kwargs):
            assert "src/scripts/paper.sh" in argv
            calls.append("paper")
            _write_published_paper(paper_dir)
            return 0

        def fake_review(**_kwargs):
            calls.append("review")
            return _review_result(paper_dir, overall_score=3, decision="Reject")

        with monkeypatch.context() as m:
            m.setattr("src.pipeline.cli.run_cmd", fake_run_cmd)
            m.setattr("src.pipeline.cli.run_glue", lambda *args, **kwargs: None)
            m.setattr("src.pipeline.cli.run_pipeline_review", fake_review)
            rc = main(
                ["execution.phases=paper,review", f"paths.run_dir={bundle}", f"paths.paper_dir={paper_dir}"]
            )

        assert rc == 0
        assert calls == ["paper", "review"]
        events = [
            json.loads(line)
            for line in (bundle / "pipeline_events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        review_event = next(event for event in events if event["event"] == "review_result")
        assert review_event["fields"]["overall_score"] == 3
        assert review_event["fields"]["decision"] == "Reject"

        # a failed paper phase skips the review
        bundle = tmp_path / "failed" / "bundle"
        _write_completed_run(bundle)
        spy = _ReviewSpy()

        with monkeypatch.context() as m:
            m.setattr("src.pipeline.cli.run_cmd", lambda *args, **kwargs: 7)
            m.setattr("src.pipeline.cli.run_glue", lambda *args, **kwargs: None)
            m.setattr("src.pipeline.cli.run_pipeline_review", spy)
            rc = main(
                [
                    "execution.phases=paper,review",
                    "execution.fail_fast=false",
                    f"paths.run_dir={bundle}",
                    f"paths.paper_dir={bundle / 'paper'}",
                ]
            )

        assert rc == 7
        assert spy.calls == 0

    def test_review_rejects_paper_root_symlink_created_by_paper_phase(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        bundle = tmp_path / "bundle"
        _write_completed_run(bundle)
        paper_dir = bundle / "paper"
        outside_root = tmp_path / "outside-paper"

        def fake_run_cmd(*_args, **_kwargs):
            _write_published_paper(outside_root)
            paper_dir.symlink_to(outside_root, target_is_directory=True)
            return 0

        monkeypatch.setattr("src.pipeline.cli.run_cmd", fake_run_cmd)
        monkeypatch.setattr("src.pipeline.cli.run_glue", lambda *args, **kwargs: None)
        spy = _ReviewSpy()
        monkeypatch.setattr("src.pipeline.cli.run_pipeline_review", spy)

        with caplog.at_level("WARNING"):
            rc = main(
                ["execution.phases=paper,review", f"paths.run_dir={bundle}", f"paths.paper_dir={paper_dir}"]
            )

        assert rc == 0
        assert spy.calls == 0
        assert any("Review phase degraded" in record.message for record in caplog.records)
