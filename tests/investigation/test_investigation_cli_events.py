"""Investigation CLI contracts: image pinning, paper step, live task/paper events, receipts, scratch cleanup."""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.investigation import cli as investigation_cli
from src.investigation import script_runners
from src.investigation.config import (
    AgentConfig,
    ContainerConfig,
    ExecutionConfig,
    InvestigationConfig,
    PaperConfig,
    PathsConfig,
)
from src.investigation.models import (
    IdeaGenerationResult,
    PaperGenerationResult,
    RunAttemptResult,
    TaskDraftResult,
)
from src.scripts.lib import container_temp
from src.task_drafting.progress import build_record
from tests.program_testutils import make_program_dir
from tests.task_drafting.test_task_contract import _write_scaffold

IMAGE_ID = "sha256:" + "ab" * 32
_DOCKER_PATCHES = (
    ("src.investigation.cli.shutil.which", {"return_value": "/usr/bin/docker"}),
    ("src.investigation.cli._probe_docker_available", {"return_value": (True, "")}),
)


def _task_dir(root: Path) -> Path:
    task_dir = root / "task"
    make_program_dir(task_dir / "program_0", content="def solve():\n    return 1\n")
    (task_dir / "capture.py").write_text("", encoding="utf-8")
    (task_dir / "evaluate.py").write_text("", encoding="utf-8")
    (task_dir / "task.md").write_text(
        "# Goal\n- optimize objective\n\n# Rubrics\n- score threshold\n", encoding="utf-8"
    )
    (task_dir / "goal.md").write_text("legacy", encoding="utf-8")
    return task_dir


def _container_overrides(tmp_path: Path, task_dir: Path, image: str, *extra: str) -> list[str]:
    return [
        "agent.name=claude",
        f"paths.task_dir={task_dir}",
        f"paths.results_dir={tmp_path / 'results'}",
        "container.enabled=true",
        f"container.image={image}",
        f"container.auth_root={tmp_path}",
        *extra,
    ]


@contextlib.contextmanager
def _docker_available():
    with contextlib.ExitStack() as stack:
        for target, kwargs in _DOCKER_PATCHES:
            stack.enter_context(patch(target, **kwargs))
        yield


# --------------------------------------------------------------------------- image pinning


@dataclass
class _FakeLease:
    image_id: str
    tag: str
    released: bool = False
    verified: int = field(default=0)

    def verify(self) -> None:
        self.verified += 1

    def release(self) -> None:
        self.released = True


def test_pinned_image_lease_is_managed_handed_to_every_step_and_unresolvable_is_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pin and release manage one lease
    acquired: list[tuple[str, str, str]] = []
    leases: list[_FakeLease] = []

    def acquire_many(image_reference: str, *, lease_scope: str, role: str) -> _FakeLease:
        acquired.append((image_reference, lease_scope, role))
        leases.append(_FakeLease(IMAGE_ID, f"lease:{len(leases)}"))
        return leases[-1]

    with monkeypatch.context() as m:
        m.setattr(investigation_cli, "acquire_image_lease", acquire_many)
        assert investigation_cli._pin_container_image("agent:latest", "/runs/a") == IMAGE_ID
        assert acquired == [("agent:latest", "/runs/a", "agent")]
        # Re-pinning releases the previous lease; releasing twice is harmless.
        investigation_cli._pin_container_image("agent:latest", "/runs/a")
        assert leases[0].released and not leases[1].released
        investigation_cli._release_container_image()
        investigation_cli._release_container_image()
        assert leases[1].released

    # main hands the pinned image id to every step
    root = tmp_path / "pinned"
    root.mkdir()
    task_dir = _task_dir(root)
    results_dir = root / "results"
    lease = _FakeLease(IMAGE_ID, "lease:agent")
    acquired.clear()

    def acquire_one(image_reference: str, *, lease_scope: str, role: str) -> _FakeLease:
        acquired.append((image_reference, lease_scope, role))
        return lease

    overrides = _container_overrides(
        root, task_dir, "agentic-scientist/agent-cli:test",
        "execution.max_idea_cycles=1", "execution.max_idea_generation_attempts=1",
    )
    with (
        monkeypatch.context() as m,
        _docker_available(),
        patch("src.investigation.cli._docker_image_exists", return_value=(True, "")),
        patch("src.investigation.cli.prepare_task_runtime", return_value=True),
        patch(
            "src.investigation.cli.run_idea_generation",
            return_value=IdeaGenerationResult(output="no idea", exit_code=1, idea_dir=None),
        ) as idea_mock,
    ):
        m.setattr(investigation_cli, "acquire_image_lease", acquire_one)
        exit_code = investigation_cli.main(overrides=overrides)
    # The stubbed idea step fails, so the loop exits non-zero; what matters is the pinning.
    assert exit_code == 2
    assert acquired == [("agentic-scientist/agent-cli:test", str(results_dir.resolve()), "agent")]
    assert idea_mock.call_count == 1
    assert idea_mock.call_args.kwargs["container_image"] == IMAGE_ID
    assert lease.released

    # unresolvable image is a config error
    root = tmp_path / "unresolvable"
    root.mkdir()
    task_dir = _task_dir(root)

    def acquire_missing(image_reference: str, *, lease_scope: str, role: str) -> _FakeLease:
        raise ValueError(f"no such image: {image_reference}")

    overrides = _container_overrides(root, task_dir, "agentic-scientist/agent-cli:missing")
    with (
        monkeypatch.context() as m,
        _docker_available(),
        patch("src.investigation.cli.run_idea_generation") as idea_mock,
    ):
        m.setattr(investigation_cli, "acquire_image_lease", acquire_missing)
        assert investigation_cli.main(overrides=overrides) == 2
    idea_mock.assert_not_called()


# --------------------------------------------------------------------------- paper step


def _paper_env(root: Path) -> SimpleNamespace:
    """Task dir, instruct files, and a successful idea workspace + experiment for one cycle."""
    root.mkdir(parents=True, exist_ok=True)
    env = SimpleNamespace(
        root=root,
        task_dir=_task_dir(root),
        results_dir=root / "results",
        instruct=root / "instruct.md",
        paper_instruct=root / "paper_instruct.md",
        exp=root / "exp_success",
        idea_workspace=root / "idea_workspace",
    )
    env.instruct.write_text("template", encoding="utf-8")
    env.paper_instruct.write_text("paper template", encoding="utf-8")
    rubrics = [{"id": "score_threshold", "description": "score > 2.6", "type": "metric"}]
    env.idea_workspace.mkdir(parents=True)
    (env.idea_workspace / "idea_bundle.json").write_text(
        json.dumps({
            "idea_id": "idea_0001", "title": "idea", "hypothesis": "hyp",
            "implementation_plan": ["step"], "target_rubrics": rubrics, "notes_for_agent": "notes",
        }),
        encoding="utf-8",
    )
    (env.idea_workspace / "target_rubrics.json").write_text(json.dumps(rubrics), encoding="utf-8")
    make_program_dir(env.exp / "program_1", content="# candidate\n")
    (env.exp / "program_1" / "metrics.json").write_text('{"combined_score": 2.8}', encoding="utf-8")
    (env.exp / "program_1" / "correct.json").write_text('{"correct": true}', encoding="utf-8")
    (env.exp / "rubric_status.json").write_text(
        json.dumps({
            "attempt_id": "idea_0001_attempt_01", "idea_id": "idea_0001", "overall_pass": True,
            "rubrics": [{"id": "score_threshold", "pass": True, "evidence": "ok"}],
        }),
        encoding="utf-8",
    )
    return env


def _paper_cfg(
    env: SimpleNamespace,
    *,
    write_paper: bool = True,
    workers: int = 1,
    paper_instruct: bool = True,
    paper_contributions: Path | None = None,
    global_memory_dir: Path | None = None,
) -> InvestigationConfig:
    return InvestigationConfig(
        agent=AgentConfig(name="codex"),
        paths=PathsConfig(
            task_dir=str(env.task_dir),
            results_dir=str(env.results_dir),
            global_memory_dir=str(global_memory_dir) if global_memory_dir else None,
            instruct=str(env.instruct),
            paper_instruct=str(env.paper_instruct) if paper_instruct else None,
            paper_contributions=str(paper_contributions) if paper_contributions else None,
        ),
        execution=ExecutionConfig(
            max_idea_cycles=1, run_timeout_seconds=5, run_timeout_grace_seconds=1,
            max_program_evals_per_attempt=4, workers=workers, resume=False, draft_task_if_missing=False,
        ),
        paper=PaperConfig(
            enabled=write_paper, timeout_seconds=1800, timeout_grace_seconds=30, min_refs=8, max_refs=15,
        ),
        container=ContainerConfig(),
    )


def _paper_success(results_dir: Path, check=None):
    paper_dir = results_dir / "paper" / "final_test"

    def side_effect(**kwargs: object) -> PaperGenerationResult:
        if check is not None:
            check(kwargs)
        paper_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "paper" / "latest_manifest.json").write_text("{}", encoding="utf-8")
        return PaperGenerationResult(output=f"Paper directory: {paper_dir}\n", exit_code=0, paper_dir=paper_dir)

    return paper_dir, side_effect


@contextlib.contextmanager
def _single_cycle(env: SimpleNamespace, cfg: InvestigationConfig, paper):
    with (
        patch("src.investigation.cli.load_config", return_value=cfg),
        patch(
            "src.investigation.cli.run_idea_generation",
            return_value=IdeaGenerationResult(
                output=f"Idea directory: {env.idea_workspace}\n", exit_code=0, idea_dir=env.idea_workspace
            ),
        ),
        patch(
            "src.investigation.cli.run_single_attempt",
            return_value=RunAttemptResult(
                output=f"Experiment directory: {env.exp}\n", exit_code=0, experiment_dir=env.exp
            ),
        ),
        patch("src.investigation.cli.run_paper_generation", **paper) as paper_mock,
    ):
        yield paper_mock


def test_main_parallel_campaign_routes_pruning_and_runs_paper_only_after_success(tmp_path: Path) -> None:
    # main routes single-worker pruning through the parallel campaign
    paper_env = _paper_env(tmp_path / "pruning")
    cfg = _paper_cfg(paper_env, write_paper=False, paper_instruct=False)
    cfg.execution.max_idea_cycles = 2
    cfg.execution.prune_every_n_cycles = 1
    cfg.container.enabled = False
    with (
        patch("src.investigation.cli.load_config", return_value=cfg),
        patch("src.investigation.cli.run_task_setup", return_value=True),
        patch("src.investigation.cli._stop_container_session") as stop_mock,
        patch("src.investigation.cli.run_parallel_campaign", return_value=0) as parallel_mock,
        patch("src.investigation.cli.run_idea_generation") as idea_mock,
    ):
        assert investigation_cli.main() == 0
    parallel_mock.assert_called_once()
    assert parallel_mock.call_args.kwargs["results_dir"] == paper_env.results_dir.resolve()
    assert parallel_mock.call_args.kwargs["global_memory_dir"] == paper_env.results_dir.resolve()
    stop_mock.assert_not_called()
    idea_mock.assert_not_called()

    # main runs paper after parallel campaign when enabled
    paper_env = _paper_env(tmp_path / "paper_after_parallel")
    cfg = _paper_cfg(paper_env, workers=2)
    cfg.paper.ai_review = "openai"
    parallel_results_dir = (paper_env.results_dir / "codex").resolve()

    def check(kwargs: dict) -> None:
        assert kwargs["results_dir"] == parallel_results_dir
        assert kwargs["final_ai_review"] == "openai"

    paper_dir, side_effect = _paper_success(parallel_results_dir, check)
    with (
        patch("src.investigation.cli.load_config", return_value=cfg),
        patch("src.investigation.cli.run_parallel_campaign", return_value=0) as parallel_mock,
        patch("src.investigation.cli.run_paper_generation", side_effect=side_effect) as paper_mock,
    ):
        assert investigation_cli.main() == 0
    parallel_mock.assert_called_once()
    assert paper_mock.call_count == 1
    assert (paper_dir / "wrapper_output.log").is_file()

    # main skips paper after parallel failure
    paper_env = _paper_env(tmp_path / "parallel_failure")
    cfg = _paper_cfg(paper_env, workers=2)
    with (
        patch("src.investigation.cli.load_config", return_value=cfg),
        patch("src.investigation.cli.run_parallel_campaign", return_value=2) as parallel_mock,
        patch("src.investigation.cli.run_paper_generation") as paper_mock,
    ):
        assert investigation_cli.main() == 2
    parallel_mock.assert_called_once()
    paper_mock.assert_not_called()


def test_main_paper_step_runs_when_enabled_and_fails_on_missing_contributions_or_step_failure(
    tmp_path: Path,
) -> None:
    # main runs paper step when enabled
    paper_env = _paper_env(tmp_path / "enabled")
    contributions = paper_env.root / "contributions.md"
    contributions.write_text("- contribution one\n", encoding="utf-8")
    shared_memory = paper_env.root / "shared_memory"
    cfg = _paper_cfg(paper_env, paper_contributions=contributions, global_memory_dir=shared_memory)
    cfg.paper.ai_review = "openai"
    paper_dir, side_effect = _paper_success(paper_env.results_dir)
    stdout = io.StringIO()
    with _single_cycle(paper_env, cfg, {"side_effect": side_effect}) as paper_mock:
        with contextlib.redirect_stdout(stdout):
            assert investigation_cli.main() == 0
    assert paper_mock.call_count == 1
    assert paper_mock.call_args.kwargs["global_memory_dir"] == shared_memory.resolve()
    assert paper_mock.call_args.kwargs["final_ai_review"] == "openai"
    assert paper_mock.call_args.kwargs["contributions_path"] == contributions.resolve()
    assert (paper_dir / "wrapper_output.log").is_file()
    rendered = stdout.getvalue()
    assert "Generating paper" in rendered
    assert "done" in rendered
    assert "Finished" in rendered

    # main fails fast when paper contributions file is missing
    paper_env = _paper_env(tmp_path / "missing_contributions")
    missing = paper_env.root / "missing_contributions.md"
    cfg = _paper_cfg(paper_env, paper_contributions=missing)
    with (
        patch("src.investigation.cli.load_config", return_value=cfg),
        patch("src.investigation.cli.run_idea_generation") as idea_mock,
        patch("src.investigation.cli.log_event") as log_mock,
    ):
        assert investigation_cli.main() == 2
    idea_mock.assert_not_called()
    assert any(
        call.args and call.args[0] == "config_error"
        and call.kwargs.get("reason") == "missing_paper_contributions_file"
        and call.kwargs.get("path") == missing.resolve()
        for call in log_mock.call_args_list
    )

    # main fails when paper step fails
    paper_env = _paper_env(tmp_path / "paper_failure")
    cfg = _paper_cfg(paper_env, paper_instruct=False)
    cfg.logging.format = "json"
    failed = PaperGenerationResult(
        output="paper failed", exit_code=1, paper_dir=paper_env.results_dir / "paper" / "final_test"
    )
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), _single_cycle(paper_env, cfg, {"return_value": failed}):
        assert investigation_cli.main() == 2
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert records
    assert "paper_generation_failed" in {record["event"] for record in records}


# --------------------------------------------------------------------------- live paper / task events


def test_run_paper_generation_reemits_transport_events() -> None:
    record = {
        "timestamp": "2026-08-17T10:00:00.000Z", "source": "paper", "event": "paper_gate",
        "session_id": "paper-test", "gate": "review", "state": "completed", "fields": {"round": 2},
    }
    fake_proc = Mock()
    fake_proc.communicate.return_value = (
        "AGENTIC_SCIENTIST_PAPER_EVENT " + json.dumps(record) + "\nPaper directory: /tmp/paper_final\n", "",
    )
    fake_proc.returncode = 0
    with (
        patch("src.investigation.cli.subprocess.Popen", return_value=fake_proc) as popen_mock,
        patch("src.investigation.cli.log_event") as log_event_mock,
    ):
        result = investigation_cli.run_paper_generation(
            agent="codex", task_dir=Path("/tmp/task"), results_dir=Path("/tmp/results"),
            output_dir=Path("/tmp/results/paper/final"), instruct_path=Path("/tmp/paper.md"),
            timeout_seconds=60,
        )
    assert result.output == "Paper directory: /tmp/paper_final\n"
    assert result.paper_dir == Path("/tmp/paper_final")
    command = popen_mock.call_args.args[0]
    assert "--event-transport" in command
    assert command[command.index("--heartbeat-seconds") + 1] == "0"
    log_event_mock.assert_called_once_with(
        "paper_gate", gate="review", state="completed", paper_session_id="paper-test",
        paper_timestamp="2026-08-17T10:00:00.000Z", paper_fields={"round": 2},
    )


def _task_records(*specs: tuple[str, str, dict]) -> list[dict]:
    return [
        build_record(
            gate=gate, state=state, session_id="host-task-test", auth_token="host-task-secret",
            timestamp_fn=lambda: "2026-08-29T08:00:00.000Z", fields=fields,
        )
        for gate, state, fields in specs
    ]


@contextlib.contextmanager
def _task_drafting_transport(records: list[dict], *, trailer: str = "", stderr: str = "", returncode: int = 0):
    """Drive ``run_task_drafting`` with scripted transport events; yields (log_event mock, popen mock)."""
    fake_proc = Mock()
    fake_proc.communicate.return_value = (
        "".join("AGENTIC_SCIENTIST_TASK_EVENT " + json.dumps(record) + "\n" for record in records) + trailer,
        stderr,
    )
    fake_proc.returncode = returncode
    with (
        patch("src.investigation.cli.subprocess.Popen", return_value=fake_proc) as popen_mock,
        patch("src.investigation.cli.log_event") as log_event_mock,
        patch("src.investigation.script_runners.uuid.uuid4") as uuid_mock,
        patch("src.investigation.script_runners.secrets.token_hex", return_value="host-task-secret"),
    ):
        uuid_mock.return_value.hex = "host-task-test"
        yield log_event_mock, popen_mock


def _run_task_drafting(**extra):
    return investigation_cli.run_task_drafting(
        agent="codex", task_dir=Path("/tmp/task"), output_dir=Path("/tmp/results/task-draft"),
        instruct_path=Path("/tmp/draft.md"), timeout_seconds=60, **extra,
    )


def test_run_task_drafting_facade_reemits_events_and_synthesizes_terminal_checkpoints() -> None:
    # task drafting facade accepts every core option
    injected_dependencies = {
        "script_path", "popen_factory", "register_subprocess", "unregister_subprocess",
        "terminate_process_group", "container_execution_env", "ensure_text", "task_event_fn",
    }
    facade_parameters = set(inspect.signature(investigation_cli.run_task_drafting).parameters)
    core_parameters = set(inspect.signature(script_runners.run_task_drafting).parameters)
    assert facade_parameters == core_parameters - injected_dependencies

    # run_task_drafting re-emits transport events
    records = _task_records((
        "design_review", "completed",
        {"label": "Design review round 1 · reviewer 2/3", "verdict": "pass",
         "cost_checkpoint": True, "task_cost_usd": "0.42"},
    ))
    with _task_drafting_transport(records, trailer="Draft complete\n") as (log_event_mock, popen_mock):
        result = _run_task_drafting(
            smoke_max_infrastructure_attempts=5, agent_transient_max_attempts=3,
            max_idea_cycles=7, max_program_evals_per_attempt=2,
        )
    assert result.output == "Draft complete\n"
    environment = popen_mock.call_args.kwargs["env"]
    command = popen_mock.call_args.args[0]
    expected_flags = {
        "--qualification-max-repairs": "4",
        "--agent-transient-max-attempts": "3",
        "--smoke-max-infrastructure-attempts": "5",
        "--qualification-max-infrastructure-attempts": "3",
        "--production-entry-timeout-seconds": "300",
        "--qualification-timeout-seconds": "60",
        "--max-runtime-bytes": "17179869184",
        "--max-idea-cycles": "7",
        "--max-program-evals-per-attempt": "2",
    }
    for flag, value in expected_flags.items():
        assert command[command.index(flag) + 1] == value, flag
    assert environment["AGENTIC_TASK_PROGRESS_TRANSPORT"] == "wrapper"
    assert environment["TASK_PROGRESS_SESSION_ID"] == "host-task-test"
    assert environment["TASK_PROGRESS_AUTH_TOKEN"] == "host-task-secret"
    log_event_mock.assert_called_once_with(
        "task_drafting_checkpoint", gate="design_review", state="completed",
        task_drafting_session_id="host-task-test", task_drafting_timestamp="2026-08-29T08:00:00.000Z",
        label="Design review round 1 · reviewer 2/3", verdict="pass", cost_checkpoint=True,
    )

    # closes active checkpoint after script failure
    records = _task_records(("publication", "started", {"label": "Publishing task"}))
    with _task_drafting_transport(records, trailer="publication failed\n", returncode=2) as (log_event_mock, _):
        def reject_synthetic_event(_event: str, **fields: object) -> None:
            if fields.get("synthetic"):
                raise BrokenPipeError("closed progress stream")

        log_event_mock.side_effect = reject_synthetic_event
        result = _run_task_drafting()
    assert result.exit_code == 2
    assert log_event_mock.call_count == 2
    failed = log_event_mock.call_args_list[-1]
    assert failed.args == ("task_drafting_checkpoint",)
    assert failed.kwargs["gate"] == "publication"
    assert failed.kwargs["state"] == "failed"
    assert failed.kwargs["cost_checkpoint"] is True
    assert failed.kwargs["synthetic"] is True

    # preserves retry context for synthetic failure
    records = _task_records(
        ("smoke", "started", {"label": "Evaluator smoke", "attempt": 2, "max_attempts": 4}),
        ("smoke", "retrying", {"label": "Evaluator smoke · infrastructure retry",
                               "infrastructure_attempt": 2, "max_infrastructure_attempts": 3}),
    )
    with _task_drafting_transport(records, stderr="runtime disappeared\n", returncode=4) as (log_event_mock, _):
        result = _run_task_drafting()
    assert result.exit_code == 4
    assert log_event_mock.call_count == 3
    terminal = log_event_mock.call_args_list[-1].kwargs
    assert terminal["state"] == "failed"
    assert terminal["gate"] == "smoke"
    assert terminal["attempt"] == 2
    assert terminal["infrastructure_attempt"] == 2
    assert terminal["max_infrastructure_attempts"] == 3
    assert terminal["synthetic"] is True

    # restores outer checkpoint after nested events
    records = _task_records(*(
        (gate, state, {"label": label})
        for gate, state, label in (
            ("scaffold", "started", "Scaffold generation"),
            ("scaffold", "started", "Scaffold agent attempt 1/2"),
            ("scaffold", "retrying", "Scaffold agent attempt 1/2"),
            ("scaffold", "started", "Scaffold agent attempt 2/2"),
            ("qualification", "retrying", "Unrelated stale retry"),
            ("qualification", "completed", "Unrelated stale terminal"),
            ("scaffold", "completed", "Scaffold agent attempt 2/2"),
        )
    ))
    with _task_drafting_transport(records, stderr="agent transport failed\n", returncode=75) as (log_event_mock, _):
        _run_task_drafting()
    assert log_event_mock.call_count == len(records) + 1
    synthetic_calls = [call for call in log_event_mock.call_args_list if call.kwargs.get("synthetic")]
    assert len(synthetic_calls) == 1
    terminal = synthetic_calls[0].kwargs
    assert terminal["state"] == "failed"
    assert terminal["gate"] == "scaffold"
    assert terminal["label"] == "Scaffold generation"
    assert terminal["synthetic"] is True

    # accepts typed terminal qualification failure
    records = _task_records(
        ("qualification", "started", {"label": "Full production search qualification"}),
        ("qualification", "failed", {"label": "Full production search qualification",
                                     "verdict": "fail", "failure_kind": "infrastructure_failure"}),
    )
    with _task_drafting_transport(records, returncode=4) as (log_event_mock, _):
        result = _run_task_drafting()
    assert result.exit_code == 4
    assert log_event_mock.call_count == 2
    terminal = log_event_mock.call_args_list[-1]
    assert terminal.kwargs["state"] == "failed"
    assert terminal.kwargs["verdict"] == "fail"
    assert terminal.kwargs["failure_kind"] == "infrastructure_failure"
    assert "synthetic" not in terminal.kwargs


# --------------------------------------------------------------------------- task drafting at main()


def test_drafting_does_not_reuse_campaign_container_session(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text("# Task\n", encoding="utf-8")
    results_dir = tmp_path / "results"
    instruct_path, idea_instruct_path, draft_instruct_path, paper_instruct_path = (
        tmp_path / name for name in ("instruct.md", "idea.md", "draft.md", "paper.md")
    )
    for path in (instruct_path, idea_instruct_path, draft_instruct_path, paper_instruct_path):
        path.write_text("instructions\n", encoding="utf-8")
    cfg = InvestigationConfig(
        agent=AgentConfig(name="pi"),
        paths=PathsConfig(task_dir=str(task_dir), results_dir=str(results_dir), instruct=str(instruct_path)),
        execution=ExecutionConfig(max_idea_cycles=1, workers=1, draft_task_if_missing=True),
        container=ContainerConfig(enabled=True, image="test/agent:latest", auth_root=str(tmp_path)),
    )

    def finish_draft(**_: object) -> TaskDraftResult:
        (task_dir / "capture.py").write_text("", encoding="utf-8")
        (task_dir / "evaluate.py").write_text("", encoding="utf-8")
        make_program_dir(task_dir / "program_0")
        return TaskDraftResult(
            output="draft complete\n", exit_code=0, draft_dir=results_dir / "_task_drafting" / "draft"
        )

    with (
        patch("src.investigation.cli.load_config", return_value=cfg),
        _docker_available(),
        patch("src.investigation.cli._build_container_session_name", return_value="campaign-session"),
        patch("src.investigation.cli._stop_container_session", return_value=True),
        patch("src.investigation.cli.default_idea_instruction_path", return_value=idea_instruct_path),
        patch("src.investigation.cli.default_draft_instruction_path", return_value=draft_instruct_path),
        patch("src.investigation.cli.default_paper_instruction_path", return_value=paper_instruct_path),
        patch("src.investigation.cli.run_task_setup", return_value=True),
        patch("src.investigation.cli.prepare_task_runtime", return_value=True),
        patch("src.investigation.cli.run_task_drafting", side_effect=finish_draft) as draft_mock,
        patch(
            "src.investigation.cli.run_idea_generation",
            return_value=IdeaGenerationResult(output="no idea\n", exit_code=1, idea_dir=None),
        ) as idea_mock,
    ):
        assert investigation_cli.main() == 2
    draft_kwargs = draft_mock.call_args.kwargs
    assert draft_kwargs["container_session_name"] is None
    assert draft_kwargs["container_mount_root"] is None
    idea_kwargs = idea_mock.call_args.kwargs
    assert idea_kwargs["container_session_name"] == "campaign-session"
    assert idea_kwargs["container_mount_root"] == results_dir.resolve()


def test_task_only_requires_existing_contracted_task_validation_receipt(tmp_path: Path) -> None:
    for draft_if_missing in (False, True):
        root = tmp_path / f"draft_{draft_if_missing}"
        root.mkdir()
        task_dir = root / "task"
        _write_scaffold(task_dir)
        (task_dir / "capture.py").write_text("", encoding="utf-8")
        make_program_dir(task_dir / "program_0")
        cfg = InvestigationConfig()
        cfg.paths.task_dir = str(task_dir)
        cfg.paths.results_dir = str(root / "results")
        cfg.execution.task_only = True
        cfg.execution.draft_task_if_missing = draft_if_missing
        cfg.container.enabled = True
        cfg.container.image = "test/task:latest"
        cfg.paper.enabled = False
        with (
            patch.dict(os.environ),
            patch("src.investigation.cli.load_config", return_value=cfg),
            _docker_available(),
            patch("src.investigation.cli._stop_container_session"),
            patch("src.investigation.cli.prepare_task_runtime", return_value=True) as runtime,
            patch(
                "src.investigation.cli.run_task_drafting",
                return_value=TaskDraftResult(output="", exit_code=0, draft_dir=root),
            ) as drafting,
            patch("src.investigation.cli.log_event") as events,
        ):
            assert investigation_cli.main() == 2, draft_if_missing
        runtime.assert_not_called()
        if draft_if_missing:
            drafting.assert_called_once()
            assert any(call.args == ("task_draft_incomplete",) for call in events.call_args_list)
        else:
            drafting.assert_not_called()
            events.assert_any_call(
                "config_error", reason="missing_task_scaffold_files",
                missing_scaffold=[str(task_dir / ".agentic_task_validation.json")],
                hint="use execution.draft_task_if_missing=true to auto-draft",
            )


# --------------------------------------------------------------------------- container scratch cleanup


def _scratch_session(root: Path, monkeypatch: pytest.MonkeyPatch):
    """Verified container scratch leaf plus a fake stoppable ``session`` container; returns (leaf, state, details)."""
    root.mkdir(mode=0o700, parents=True)
    leaf = container_temp.create(root)
    (leaf / "build-output").write_text("temporary build output")
    details = {
        "Name": "/session",
        "Config": {"Labels": {container_temp.LABEL: str(leaf)}},
        "Mounts": [{"Destination": "/agent-tmp", "Source": str(leaf), "Type": "bind", "RW": True}],
    }
    state = {"present": True, "inspect_calls": 0}

    def inspect(name):
        assert name == "session"
        state["inspect_calls"] += 1
        return details if state["present"] else None

    def stop(name):
        assert name == "session"
        state["present"] = False
        return True

    monkeypatch.setenv("AGENTIC_CONTAINER_TEMP_ROOT", str(root))
    monkeypatch.setattr(container_temp, "inspect_container", inspect)
    monkeypatch.setattr(investigation_cli, "_ACTIVE_CONTAINER_SESSION_NAME", "session")
    monkeypatch.setattr(investigation_cli, "_stop_container_session", stop)
    monkeypatch.setattr(investigation_cli, "shutdown_event_logging", lambda: None)
    monkeypatch.setattr(investigation_cli, "main", lambda **kwargs: 0)
    return leaf, state, details


def test_run_cli_removes_verified_scratch_only_after_successful_session_teardown(tmp_path, monkeypatch):
    # success removes verified scratch after session teardown
    with monkeypatch.context() as m:
        leaf, state, _ = _scratch_session(tmp_path / "success", m)
        assert investigation_cli.run_cli() == 0
        assert not state["present"]
        assert not leaf.exists()

    # interrupted scratch inspection still stops container
    with monkeypatch.context() as m:
        leaf, state, _ = _scratch_session(tmp_path / "interrupted_inspection", m)

        def interrupt(name):
            raise KeyboardInterrupt

        m.setattr(container_temp, "inspect_container", interrupt)
        assert investigation_cli.run_cli() == 0
        assert not state["present"]
        assert leaf.exists()

    # legacy session skips scratch inspection
    with monkeypatch.context() as m:
        leaf, state, _ = _scratch_session(tmp_path / "legacy", m)
        m.delenv("AGENTIC_CONTAINER_TEMP_ROOT")
        assert investigation_cli.run_cli() == 0
        assert leaf.exists()
        assert state["inspect_calls"] == 0


def test_run_cli_preserves_scratch_after_unsuccessful_run_or_unverified_teardown(tmp_path, monkeypatch):
    def raising(error):
        def main(**kwargs):
            raise error
        return main

    # unsuccessful run preserves scratch
    for outcome in ("failed", "exception", "interrupted"):
        with monkeypatch.context() as m:
            leaf, state, _ = _scratch_session(tmp_path / outcome, m)
            if outcome == "failed":
                m.setattr(investigation_cli, "main", lambda **kwargs: 3)
                assert investigation_cli.run_cli() == 3
            elif outcome == "exception":
                m.setattr(investigation_cli, "main", raising(RuntimeError("run failed")))
                with pytest.raises(RuntimeError, match="run failed"):
                    investigation_cli.run_cli()
            else:
                m.setattr(investigation_cli, "main", raising(KeyboardInterrupt()))
                m.setattr(investigation_cli, "_terminate_all_active_processes", lambda **kwargs: [])
                assert investigation_cli.run_cli() == 130
            assert leaf.exists(), outcome
            assert state["inspect_calls"] == 0, outcome

    # unverified scratch is preserved
    for scenario in ("container_still_present", "unknown_metadata", "inspection_failure", "uncertain_removal"):
        with monkeypatch.context() as m:
            leaf, _, details = _scratch_session(tmp_path / scenario, m)
            if scenario == "container_still_present":
                m.setattr(investigation_cli, "_stop_container_session", lambda name: True)
            elif scenario == "unknown_metadata":
                details["Config"] = {}
            elif scenario == "inspection_failure":
                def fail(name):
                    raise ValueError("Docker unavailable")
                m.setattr(container_temp, "inspect_container", fail)
            else:
                replies = iter([details, ValueError("Docker unavailable")])

                def inspect(name):
                    reply = next(replies)
                    if isinstance(reply, Exception):
                        raise reply
                    return reply
                m.setattr(container_temp, "inspect_container", inspect)
            assert investigation_cli.run_cli() == 0, scenario
            assert leaf.exists(), scenario
