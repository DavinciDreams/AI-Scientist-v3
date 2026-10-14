#!/usr/bin/env python3
"""Investigation CLI: config loading, helpers, parallel workers, run_cli and main."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from tests.program_testutils import make_program_dir
from tests.task_drafting.test_task_contract import _valid_contract

from src.investigation import cli as investigation_cli
from src.investigation import runtime as runtime_module
from src.investigation.config import (
    AgentConfig,
    ContainerConfig,
    ExecutionConfig,
    InvestigationConfig,
    PaperConfig,
    PathsConfig,
)
from src.investigation.config_loader import _dict_to_config
from src.investigation.main_entry import _enable_contracted_harness_mode
from src.investigation.task_runtime import (
    TaskRuntimePreparationReport,
    TaskRuntimePreparationResult,
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
CLI = "src.investigation.cli"
HEADLESS_PREFIX = [
    "env",
    "HEADLESS_MODELS_DEV_CACHE=",
    "npx",
    "-y",
    "git+https://github.com/RobertTLange/headless-cli.git#93cd9b06b85f848af1308c41e018991b33907c5e",
]
WORKER_MODULE = [sys.executable, "-m", "src.investigation.cli"]
IDEA_FAILED = investigation_cli.IdeaGenerationResult(output="idea failed", exit_code=1, idea_dir=None)
METRIC_RUBRIC = {"id": "score_threshold", "description": "score > 2.6", "type": "metric"}


class FakeStreamingProcess:
    def __init__(self, output: str, *, returncode: int = 0) -> None:
        self.stdout = io.StringIO(output)
        self.stderr = io.StringIO()
        self.returncode = returncode
        self.pid = 1234

    def wait(self) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode


def _attr(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _fake_proc(output: str, returncode: int = 0) -> Mock:
    proc = Mock()
    proc.communicate.return_value = (output, "")
    proc.returncode = returncode
    return proc


def _worker_plan(root: Path, *command_tail: str) -> investigation_cli.ParallelWorkerPlan:
    return investigation_cli.ParallelWorkerPlan(
        worker_index=1,
        assigned_cycles=1,
        cycle_start=1,
        cycle_end=1,
        results_dir=root / "worker_01",
        command=[*WORKER_MODULE, *command_tail],
    )


def _worker_result(plan: investigation_cli.ParallelWorkerPlan, **extra: Any) -> investigation_cli.ParallelWorkerResult:
    return investigation_cli.ParallelWorkerResult(
        plan=plan,
        exit_code=0,
        log_path=plan.log_path or plan.results_dir / "worker.log",
        elapsed_seconds=0.0,
        output_tail="",
        **extra,
    )


def _campaign_setup(root: Path, **execution: Any) -> tuple[Path, Path, InvestigationConfig]:
    task_dir = root / "task"
    task_dir.mkdir(parents=True)
    make_program_dir(task_dir / "program_0", content="def solve():\n    return 0\n")
    cfg = InvestigationConfig(execution=ExecutionConfig(resume=False, **execution))
    return task_dir, root / "results", cfg


def _run_campaign(cfg: InvestigationConfig, root: Path, task_dir: Path, results_dir: Path) -> int:
    return investigation_cli.run_parallel_campaign(
        cfg=cfg,
        task_dir=task_dir,
        results_dir=results_dir,
        global_memory_dir=results_dir,
        instruct_path=root / "instruct.md",
        seed_program_dir=task_dir / "program_0",
    )


def _attempt_kwargs(root: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        agent="codex",
        task_dir=root / "task",
        results_dir=root / "results",
        seed_program_dir=root / "task/program_0",
        idea_file=root / "idea.json",
        rubrics_file=root / "rubrics.json",
        attempt_id="idea_0001_attempt_01",
        idea_dir_name="01_timeout-smoke",
        attempt_dir_name="01_attempt",
        instruct_path=None,
        timeout_seconds=30,
        timeout_grace_seconds=7,
        max_program_evals_per_attempt=1,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Helpers and config loading
# ---------------------------------------------------------------------------


def test_load_config_defaults_seed_program_field_and_removed_preset_group() -> None:
    # load_config defaults include the extended timeout budget
    cfg = investigation_cli.load_config([])

    expected = {
        "execution.run_timeout_seconds": 21600,
        "execution.idea_timeout_seconds": 3600,
        "execution.max_idea_generation_attempts": 2,
        "execution.run_timeout_grace_seconds": 30,
        "paper.timeout_seconds": 7200,
        "paper.timeout_grace_seconds": 30,
        "paper.min_refs": 8,
        "paper.max_refs": 15,
        "execution.workers": 1,
        "paper.enabled": False,
        "paper.profile": "default",
        "paper.two_stage": False,
        "paper.ai_review": "disabled",
        "execution.draft_task_if_missing": False,
        "execution.task_only": False,
        "container.enabled": False,
        "container.image": investigation_cli.DEFAULT_CONTAINER_IMAGE,
        "container.network": investigation_cli.DEFAULT_CONTAINER_NETWORK,
        "container.env_file": "",
    }
    assert {key: _attr(cfg, key) for key in expected} == expected

    # _dict_to_config keeps the seed_program field
    cfg = _dict_to_config({"paths": {"task_dir": "examples/circpack", "seed_program": "/tmp/seed.py"}})
    assert cfg.paths.seed_program == "/tmp/seed.py"

    # load_config rejects the removed preset group
    from hydra.errors import ConfigCompositionException

    with pytest.raises(ConfigCompositionException):
        investigation_cli.load_config(["preset=docker"])


LOAD_CONFIG_OVERRIDE_CASES: list[tuple[str, list[str], dict[str, Any]]] = [
    (
        "later_cli_overrides_win",
        ["agent.name=codex", "agent.name=gemini", "execution.max_idea_cycles=7"],
        {"agent.name": "gemini", "execution.max_idea_cycles": 7},
    ),
    ("task_draft", ["execution.draft_task_if_missing=true"], {"execution.draft_task_if_missing": True}),
    ("task_only", ["execution.task_only=true"], {"execution.task_only": True}),
    ("agent_model", ["agent.model=gpt-5.2"], {"agent.model": "gpt-5.2"}),
    ("reasoning_effort", ["agent.reasoning_effort=xhigh"], {"agent.reasoning_effort": "xhigh"}),
    (
        "codex_profile",
        ["agent.name=codex", "agent.profile=custom", "agent.model=custom-model"],
        {"agent.profile": "custom"},
    ),
    (
        "pool_and_model_map",
        ["agent.pool=[codex,claude]", "agent.models.codex=gpt-5.2", "agent.models.claude=claude-opus-4-6"],
        {"agent.pool": ["codex", "claude"], "agent.models": {"codex": "gpt-5.2", "claude": "claude-opus-4-6"}},
    ),
    (
        "ensemble_results_dir",
        ["agent.pool=[codex,claude]", "run_id=test_run"],
        {"paths.results_dir": "results/circpack/ensemble/test_run"},
    ),
    (
        "idea_generation_recovery",
        ["execution.idea_timeout_seconds=42", "execution.max_idea_generation_attempts=3"],
        {"execution.idea_timeout_seconds": 42, "execution.max_idea_generation_attempts": 3},
    ),
    (
        "parallel_write_paper",
        ["execution.workers=2", "paper.enabled=true"],
        {"execution.workers": 2, "paper.enabled": True},
    ),
    ("openai_final_review", ["paper.ai_review=openai"], {"paper.ai_review": "openai"}),
    (
        "all_valid_keys",
        ["agent.name=claude", "execution.max_idea_cycles=5", "paper.min_refs=2", "paper.max_refs=10", "container.image=test:latest"],
        {"agent.name": "claude", "execution.max_idea_cycles": 5},
    ),
    (
        "two_stage_paper",
        ["paper.enabled=true", "paper.two_stage=true"],
        {"paper.enabled": True, "paper.two_stage": True},
    ),
    (
        "standalone_paper_flag",
        ["--standalone-paper"],
        {
            "paper.enabled": True,
            "paper.two_stage": True,
            "paper.profile": "standalone",
            "paths.paper_instruct": "src/prompts/paper_standalone.md",
        },
    ),
]


def test_load_config_accepts_overrides() -> None:
    for name, overrides, expected in LOAD_CONFIG_OVERRIDE_CASES:
        cfg = investigation_cli.load_config(overrides)
        assert {key: _attr(cfg, key) for key in expected} == expected, name


LOAD_CONFIG_REJECTED_CASES: list[tuple[str, list[str]]] = [
    ("invalid_reasoning_effort", ["agent.reasoning_effort=max"]),
    ("profile_without_model", ["agent.name=codex", "agent.profile=custom"]),
    ("empty_profile", ["agent.name=codex", "agent.profile=''"]),
    ("profile_for_non_codex", ["agent.name=claude", "agent.profile=custom"]),
    ("invalid_pool_entry", ["agent.pool=[codex,bogus]"]),
    ("duplicate_pool_entries", ["agent.pool=[codex,codex]"]),
    ("invalid_model_map_key", ["agent.models.bogus=gpt-test"]),
    ("null_model_map_value", ["agent.models.codex=null"]),
    ("non_string_model_map_value", ["agent.models.codex=123"]),
    ("non_positive_timeout_grace", ["execution.run_timeout_grace_seconds=0"]),
    ("non_positive_idea_timeout", ["execution.idea_timeout_seconds=0"]),
    ("non_positive_idea_attempts", ["execution.max_idea_generation_attempts=0"]),
    ("non_positive_workers", ["execution.workers=0"]),
    ("unknown_final_review", ["paper.ai_review=unknown"]),
    ("paper_ref_bounds", ["paper.min_refs=10", "paper.max_refs=9"]),
    ("unknown_top_level_key", ["+bogus_key=1"]),
    ("typo_in_execution_group", ["+execution.max_idea_cyles=5"]),
    ("typo_in_agent_group", ["+agent.nme=claude"]),
]
GEMINI_REASONING_EFFORT_CASES: list[tuple[str, list[str]]] = [
    ("gemini_agent", ["agent.name=gemini", "agent.reasoning_effort=high"]),
    ("gemini_pool_entry", ["agent.pool=[codex,gemini]", "agent.reasoning_effort=high"]),
]


def test_load_config_rejects_invalid_overrides_with_exit_2_and_gemini_reasoning_effort() -> None:
    # invalid overrides exit with code 2
    for name, overrides in LOAD_CONFIG_REJECTED_CASES:
        with pytest.raises(SystemExit) as ctx:
            investigation_cli.load_config(overrides)
        assert ctx.value.code == 2, name

    # reasoning effort is rejected for gemini (as agent or pool entry)
    for name, overrides in GEMINI_REASONING_EFFORT_CASES:
        with pytest.raises(SystemExit) as ctx:
            investigation_cli.load_config(overrides)
        assert "not supported for gemini" in str(getattr(ctx.value, "message", ctx.value)), name


NORMALIZE_RESULTS_DIR_CASES: list[tuple[str, str, str]] = [
    ("agent_is_leaf", "results/circpack/claude", "results/circpack/claude"),
    ("agent_is_intermediate_segment", "results/circpack/claude/20260224_153042", "results/circpack/claude/20260224_153042"),
    ("agent_absent", "results/circpack/20260224_153042", "results/circpack/20260224_153042/claude"),
    ("user_path_containing_agent", "/custom/claude/output", "/custom/claude/output"),
    ("user_path_without_agent", "/custom/output", "/custom/output/claude"),
]


def test_results_path_helpers_and_normalize_agent_results_dir(tmp_path: Path) -> None:
    # default_results_path uses task dir name and agent
    task_dir = tmp_path / "examples" / "kissing"
    task_dir.mkdir(parents=True)
    actual = investigation_cli.default_results_path(task_dir, "codex")
    assert actual == investigation_cli.resolve_path(Path("results") / "kissing" / "codex")

    # relative_display_path uses dot prefix inside and parent segments outside
    base = tmp_path / "base"
    assert investigation_cli.relative_display_path(base / "nested" / "file.txt", base_dir=base) == "./nested/file.txt"
    assert investigation_cli.relative_display_path(tmp_path / "other" / "file.txt", base_dir=base) == "../other/file.txt"

    # state path helpers build the idea/attempt hierarchy
    results_dir = Path("/tmp/results")
    store = investigation_cli.state_store
    assert store.idea_dir(results_dir, "01_fast_restarts") == results_dir / "ideas" / "01_fast_restarts"
    assert (
        store.attempt_dir(results_dir, "01_fast_restarts", "01_attempt")
        == results_dir / "ideas" / "01_fast_restarts" / "01_attempt"
    )

    # normalize_agent_results_dir appends the agent exactly once
    for name, given, expected in NORMALIZE_RESULTS_DIR_CASES:
        assert investigation_cli.normalize_agent_results_dir(Path(given), "claude") == Path(expected), name

    # resolve_defaults then normalize does not double-append
    from src.investigation.config import resolve_defaults

    cfg = InvestigationConfig(
        agent=AgentConfig(name="claude"), paths=PathsConfig(task_dir="examples/circpack", results_dir=None)
    )
    resolve_defaults(cfg)

    assert cfg.paths.results_dir is not None
    parallel_dir = investigation_cli.normalize_agent_results_dir(Path(cfg.paths.results_dir), cfg.agent.name)
    assert parallel_dir.parts.count("claude") == 1, f"Expected 'claude' exactly once in {parallel_dir}"


def test_ensure_task_copy_in_results_and_directory_alias(tmp_path: Path) -> None:
    # ensure_task_copy_in_results copies once
    root = tmp_path / "once"
    task_dir = root / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.md").write_text("# Goal\nx\n", encoding="utf-8")
    results_dir = root / "results"

    staged = investigation_cli.ensure_task_copy_in_results(task_dir, results_dir)
    assert (staged / "task.md").is_file()

    (task_dir / "late_change.txt").write_text("new", encoding="utf-8")
    staged_again = investigation_cli.ensure_task_copy_in_results(task_dir, results_dir)
    assert staged == staged_again
    assert not (staged_again / "late_change.txt").exists()

    # ensure_task_copy_in_results respects the task ignore file
    root = tmp_path / "ignore"
    task_dir = root / "task"
    (task_dir / "program_0" / "src").mkdir(parents=True)
    (task_dir / "Kamon").mkdir(parents=True)
    (task_dir / ".venv").mkdir()
    (task_dir / "program_0" / "results_smoke").mkdir()
    (task_dir / "task.md").write_text("# Goal\nx\n", encoding="utf-8")
    (task_dir / "Kamon" / "large.jpg").write_text("large", encoding="utf-8")
    (task_dir / ".venv" / "python").write_text("python", encoding="utf-8")
    (task_dir / "program_0" / "src" / "_results.json").write_text("{}", encoding="utf-8")
    (task_dir / "program_0" / "results_smoke" / "metrics.json").write_text("{}", encoding="utf-8")
    (task_dir / ".agentic_task_ignore").write_text(
        "Kamon/\n.venv/\nprogram_*/src/_results.json\nprogram_*/results*/\n", encoding="utf-8"
    )

    staged = investigation_cli.ensure_task_copy_in_results(task_dir, root / "results")

    assert (staged / "task.md").is_file()
    for ignored in ("Kamon", ".venv", "program_0/src/_results.json", "program_0/results_smoke"):
        assert not (staged / ignored).exists()

    # ensure_task_copy_in_results rejects a pre-existing task symlink
    root = tmp_path / "symlink"
    task_dir = root / "source-task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.md").write_text("# Goal\nx\n", encoding="utf-8")
    results_dir = root / "results"
    results_dir.mkdir()
    (results_dir / "task").symlink_to(root, target_is_directory=True)

    with pytest.raises(FileExistsError, match="Unsafe staged task symlink"):
        investigation_cli.ensure_task_copy_in_results(task_dir, results_dir)

    # ensure_directory_alias creates a relative symlink or keeps an existing target
    root = tmp_path / "alias"
    workspace = root / "workspace"
    target = root / "results" / "task"
    workspace.mkdir(parents=True)
    target.mkdir(parents=True)

    alias = investigation_cli.ensure_directory_alias(workspace / "task", target)
    assert alias.is_symlink()
    assert alias.resolve() == target.resolve()

    same = investigation_cli.ensure_directory_alias(target, target)
    assert same.is_dir()
    assert not same.is_symlink()


def test_build_parallel_worker_command_forwards_flags_leased_cycles_and_agent_pool(tmp_path: Path) -> None:
    # keeps seed program and agent flags
    cfg = InvestigationConfig(
        agent=AgentConfig(name="codex", model="gpt-5.2"),
        paths=PathsConfig(task_dir="examples/circpack"),
        execution=ExecutionConfig(
            max_idea_cycles=4, run_timeout_seconds=123, run_timeout_grace_seconds=9, max_program_evals_per_attempt=2
        ),
        container=ContainerConfig(
            enabled=True,
            image="agentic-scientist/agent-cli:test",
            network="host",
            auth_root="~/.auth",
            env_file="/tmp/agent.env",
        ),
    )

    cmd = investigation_cli.build_parallel_worker_command(
        cfg=cfg,
        task_dir=Path("/tmp/task"),
        instruct_path=Path("/tmp/instruct.md"),
        results_dir=Path("/tmp/results/worker_01"),
        global_memory_dir=Path("/tmp/results"),
        assigned_cycles=4,
        seed_program=Path("/tmp/seed.py"),
    )

    for expected in (
        "paths.seed_program=/tmp/seed.py",
        "agent.model=gpt-5.2",
        "execution.idea_timeout_seconds=3600",
        "execution.max_idea_generation_attempts=2",
    ):
        assert expected in cmd
    assert "execution.cycle_offset=0" not in cmd
    assert "execution.total_cycles=4" not in cmd

    # reports the leased global cycle
    cfg = InvestigationConfig(
        paths=PathsConfig(task_dir="examples/circpack"), execution=ExecutionConfig(max_idea_cycles=3, workers=2)
    )

    # Second lease of worker 2 ran campaign cycle 3: local cycle 2 -> global 3.
    cmd = investigation_cli.build_parallel_worker_command(
        cfg=cfg,
        task_dir=Path("/tmp/task"),
        instruct_path=Path("/tmp/instruct.md"),
        results_dir=Path("/tmp/results/worker_02"),
        global_memory_dir=Path("/tmp/results"),
        assigned_cycles=2,
        worker_index=2,
        resume=True,
        global_cycle=3,
    )

    for expected in ("execution.max_idea_cycles=2", "execution.cycle_offset=1", "execution.total_cycles=3"):
        assert expected in cmd

    # forwards agent pool and models
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    instruct = tmp_path / "instruct.md"
    instruct.write_text("template", encoding="utf-8")
    cfg = InvestigationConfig(
        agent=AgentConfig(
            name="codex",
            model="fallback-model",
            pool=["codex", "claude"],
            models={"codex": "gpt-5.2", "claude": "claude-opus-4-6"},
        ),
        execution=ExecutionConfig(run_timeout_seconds=5, run_timeout_grace_seconds=1),
    )

    cmd = investigation_cli.build_parallel_worker_command(
        cfg=cfg,
        task_dir=task_dir,
        instruct_path=instruct,
        results_dir=tmp_path / "results",
        global_memory_dir=tmp_path / "memory",
        assigned_cycles=2,
        worker_index=1,
    )

    for expected in ("agent.pool=[codex,claude]", "agent.models.codex=gpt-5.2", "agent.models.claude=claude-opus-4-6"):
        assert expected in cmd


OUTPUT_DIR_PARSERS: list[tuple[str, Any, str]] = [
    ("experiment", investigation_cli.parse_experiment_dir, "Experiment"),
    ("idea", investigation_cli.parse_idea_dir, "Idea"),
    ("paper", investigation_cli.parse_paper_dir, "Paper"),
]


def test_load_instructions_output_dir_markers_and_best_program_selection(tmp_path: Path) -> None:
    # load_instructions injects core and extra placeholders
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text("# Goal\n- optimize\n\n# Rubrics\n- do x\n", encoding="utf-8")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    template = tmp_path / "template.md"
    template.write_text(
        "GOAL={{TASK_GOAL}}\nRUBRICS={{TASK_RUBRICS}}\nTD={{TASK_DIR}}\nRD={{RESULTS_DIR}}\nA={{ATTEMPT_ID}}\n",
        encoding="utf-8",
    )

    rendered = investigation_cli.load_instructions(template, task_dir, results_dir, {"ATTEMPT_ID": "attempt_01"})

    for needle in ("GOAL=- optimize", "RUBRICS=- do x", f"TD={task_dir}", f"RD={results_dir}", "A=attempt_01"):
        assert needle in rendered

    # parse_*_dir markers extract the last match
    for name, parser, label in OUTPUT_DIR_PARSERS:
        output = f"line\n{label} directory: /tmp/x_1\nmore\n{label} directory: /tmp/x_2\n"
        parsed = parser(output)
        assert parsed is not None and parsed.name == "x_2", name

    # find_best_program_in_experiment prefers the highest correct score
    exp = tmp_path / "exp"
    for name, score, correct in (("program_0", 1.0, True), ("program_1", 3.0, False), ("program_2", 2.5, True)):
        program = exp / name
        program.mkdir(parents=True)
        make_program_dir(program, content="x")
        (program / "metrics.json").write_text(json.dumps({"combined_score": score}), encoding="utf-8")
        (program / "correct.json").write_text(json.dumps({"correct": correct}), encoding="utf-8")

    best = investigation_cli.find_best_program_in_experiment(exp)

    assert best is not None
    assert best[0].name == "program_2"
    assert best[1] == "src/program.py"
    assert best[2] == pytest.approx(2.5)
    assert best[3]


# ---------------------------------------------------------------------------
# Agent command construction
# ---------------------------------------------------------------------------


INVALID_AGENT_COMMAND_CASES: list[tuple[str, str, dict[str, str], str]] = [
    ("gemini_reasoning_effort", "gemini", {"reasoning_effort": "high"}, "not supported for gemini"),
    ("codex_profile_without_model", "codex", {"agent_profile": "custom"}, "agent_model is required"),
]


def test_build_agent_command_routes_through_headless_forwards_model_and_rejects_invalid_combinations() -> None:
    # every agent is routed through headless
    for agent in ("claude", "codex", "cursor", "gemini", "opencode", "pi"):
        command = investigation_cli._build_agent_command(agent, "/private/prompt.md")
        expected_prefix = [*HEADLESS_PREFIX, agent]
        assert command[: len(expected_prefix)] == expected_prefix, agent
        assert command[command.index("--allow") + 1] == "yolo", agent
        assert "--usage" in command and "--debug" in command, agent
        assert "--json" not in command and "--model" not in command, agent
        if agent in {"claude", "codex", "gemini"}:
            assert command[command.index("--prompt-file") + 1] == "/private/prompt.md", agent
        else:
            assert command[command.index("--prompt") + 1] == (
                "Read the complete task instructions from /private/prompt.md, then follow them exactly."
            ), agent

    # claude billing selection is left to headless
    command = investigation_cli._build_agent_command("claude", "prompt", agent_model="claude-opus-4-8")
    assert command[:6] == [*HEADLESS_PREFIX, "claude"]

    # model and effort are forwarded to headless
    command = investigation_cli._build_agent_command(
        "pi", "/private/prompt.md", agent_model="test-model", reasoning_effort="high"
    )
    assert command == [
        *HEADLESS_PREFIX,
        "pi",
        "--debug",
        "--usage",
        "--allow",
        "yolo",
        "--model",
        "test-model",
        "--reasoning-effort",
        "high",
        "--prompt",
        "Read the complete task instructions from /private/prompt.md, then follow them exactly.",
    ]

    # invalid combinations are rejected
    for name, agent, kwargs, message in INVALID_AGENT_COMMAND_CASES:
        with pytest.raises(ValueError, match=message):
            investigation_cli._build_agent_command(agent, "prompt", **kwargs)


# ---------------------------------------------------------------------------
# Logging and process bookkeeping
# ---------------------------------------------------------------------------


LOG_STEP_CASES: list[tuple[str, Any, str]] = [
    (
        "step_end",
        lambda: investigation_cli.log_step_end("idea_generation", 10.0),
        "[wrapper] step_end step=idea_generation status=completed elapsed=1.5s",
    ),
    (
        "step_start",
        lambda: investigation_cli.log_step_start("idea_generation", cycle=1),
        "[wrapper] step_start step=idea_generation status=started cycle=1",
    ),
]


def test_log_rendering_elapsed_units_field_order_phase_badge_and_separator() -> None:
    # format_elapsed uses readable units
    assert [investigation_cli.format_elapsed(v) for v in (0.04, 1.26, 61.2, 3723.6)] == [
        "0.0s",
        "1.3s",
        "1m 1s",
        "1h 2m 4s",
    ]

    # log_event renders fields in call order
    runtime_module.reset_log_render_state()
    with patch("builtins.print") as print_mock:
        investigation_cli.log_event("sample", b=2, a=1)

    print_mock.assert_called_once()
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z \[wrapper\] sample b=2 a=1$", print_mock.call_args.args[0]
    )

    # log_step_* include the phase badge for the idea step
    for name, call, expected in LOG_STEP_CASES:
        runtime_module.reset_log_render_state()
        with (
            patch("src.investigation.runtime.time.monotonic", return_value=11.5),
            patch("builtins.print") as print_mock,
        ):
            call()

        print_mock.assert_called_once()
        rendered = ANSI_ESCAPE_RE.sub("", print_mock.call_args.args[0])
        assert "[IDEA] " in rendered, name
        assert expected in rendered, name
        assert "phase=idea" in rendered, name

    # log_event inserts a separator between consecutive events
    runtime_module.reset_log_render_state()
    with patch("builtins.print") as print_mock:
        investigation_cli.log_event("first")
        investigation_cli.log_event("second")

    assert print_mock.call_count == 2
    first_rendered = print_mock.call_args_list[0].args[0]
    second_rendered = print_mock.call_args_list[1].args[0]
    assert "\n" not in first_rendered
    assert second_rendered.startswith("-" * 96 + "\n")
    assert "[wrapper] second" in second_rendered


def test_terminate_active_processes_and_stop_container_session_retry_semantics() -> None:
    # _terminate_all_active_processes targets registered children
    fake_proc_a, fake_proc_b = Mock(), Mock()
    with investigation_cli._ACTIVE_SUBPROCESSES_LOCK:
        investigation_cli._ACTIVE_SUBPROCESSES.clear()
    try:
        investigation_cli._register_active_subprocess(fake_proc_a)
        investigation_cli._register_active_subprocess(fake_proc_b)
        with patch(f"{CLI}._terminate_process_group") as terminate_mock:
            investigation_cli._terminate_all_active_processes(grace_seconds=5)

        assert terminate_mock.call_count == 2
        assert {call.args[0] for call in terminate_mock.call_args_list} == {fake_proc_a, fake_proc_b}
        assert all(call.kwargs.get("grace_seconds") == 5 for call in terminate_mock.call_args_list)
    finally:
        with investigation_cli._ACTIVE_SUBPROCESSES_LOCK:
            investigation_cli._ACTIVE_SUBPROCESSES.clear()

    # stop_container_session retries then succeeds
    first = Mock(returncode=1, stderr="daemon busy", stdout="")
    second = Mock(returncode=0, stderr="", stdout="")
    log_event_mock = Mock()

    with (
        patch("src.investigation.runtime.shutil.which", return_value="/usr/bin/docker"),
        patch("src.investigation.runtime.subprocess.run", side_effect=[first, second]) as run_mock,
        patch("src.investigation.runtime.time.sleep") as sleep_mock,
        patch.dict(
            "src.investigation.runtime.os.environ",
            {"AGENT_DOCKER_STOP_RETRIES": "2", "AGENT_DOCKER_STOP_RETRY_SLEEP_SECONDS": "0"},
            clear=False,
        ),
    ):
        stopped = runtime_module.stop_container_session("session-worker-01", log_event_fn=log_event_mock)

    assert stopped
    assert run_mock.call_count == 2
    sleep_mock.assert_called_once_with(0.0)
    log_event_mock.assert_any_call("container_session_stopped", session="session-worker-01", attempts=2)

    # stop_container_session with a missing container returns False without a failure event
    missing = Mock(returncode=1, stderr="Error: No such container: missing", stdout="")
    log_event_mock = Mock()

    with (
        patch("src.investigation.runtime.shutil.which", return_value="/usr/bin/docker"),
        patch("src.investigation.runtime.subprocess.run", return_value=missing) as run_mock,
    ):
        stopped = runtime_module.stop_container_session("missing", log_event_fn=log_event_mock)

    assert not stopped
    run_mock.assert_called_once()
    assert not any(
        call.args and call.args[0] == "container_session_stop_failed" for call in log_event_mock.mock_calls
    )


# ---------------------------------------------------------------------------
# Parallel campaign and workers
# ---------------------------------------------------------------------------


def test_run_parallel_campaign_resets_stale_worker_dir_and_interrupt_stops_container_sessions(tmp_path: Path) -> None:
    # an existing worker results dir is reset
    root = tmp_path / "reset"
    task_dir, results_dir, cfg = _campaign_setup(root, max_idea_cycles=1, workers=1)
    stale_dir = results_dir / "worker_01" / "ideas" / "01_stale"
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "marker.txt").write_text("stale", encoding="utf-8")

    with (
        patch(f"{CLI}.build_parallel_worker_command", return_value=["echo", "ok"]),
        patch(f"{CLI}.run_parallel_worker", side_effect=_worker_result),
        patch(
            f"{CLI}.aggregate_parallel_worker_outputs",
            return_value={"failed_worker_count": 0, "idea_count": 0, "attempt_count": 0},
        ),
    ):
        exit_code = _run_campaign(cfg, root, task_dir, results_dir)

    assert exit_code == 0
    assert not stale_dir.exists()

    # an interrupt stops worker container sessions after terminating processes
    root = tmp_path / "interrupt"
    task_dir, results_dir, cfg = _campaign_setup(root, max_idea_cycles=1, workers=1)
    cfg.agent = AgentConfig(name="codex")
    cfg.container = ContainerConfig(enabled=True)
    cleanup_order: list[str] = []

    def _stop_session(*_args: object, **_kwargs: object) -> bool:
        cleanup_order.append("stop")
        return True

    def _terminate_processes(*_args: object, **_kwargs: object) -> list[int]:
        cleanup_order.append("terminate")
        return [4321]

    with (
        patch(f"{CLI}.log_event") as log_event_mock,
        patch(f"{CLI}.build_parallel_worker_command", return_value=["echo", "ok"]),
        patch(f"{CLI}.run_parallel_worker", side_effect=KeyboardInterrupt),
        patch(f"{CLI}.aggregate_parallel_worker_outputs"),
        patch(f"{CLI}._build_container_session_name", return_value="session_worker_01") as build_session_name_mock,
        patch(f"{CLI}._stop_container_session", side_effect=_stop_session) as stop_container_session_mock,
        patch(f"{CLI}._terminate_all_active_processes", side_effect=_terminate_processes) as terminate_processes_mock,
    ):
        with pytest.raises(KeyboardInterrupt):
            _run_campaign(cfg, root, task_dir, results_dir)

    build_session_name_mock.assert_called_once_with(agent="codex", results_dir=results_dir / "worker_01")
    stop_container_session_mock.assert_called_once_with("session_worker_01", log_event_fn=log_event_mock)
    terminate_processes_mock.assert_called_once_with(grace_seconds=investigation_cli.DEFAULT_INTERRUPT_GRACE_SECONDS)
    assert cleanup_order == ["terminate", "stop"]
    log_event_mock.assert_any_call("parallel_interrupt_cleanup_targets", mode="containers", targets="session_worker_01")


def test_run_parallel_campaign_uses_dynamic_worker_leases(tmp_path: Path) -> None:
    task_dir, results_dir, cfg = _campaign_setup(tmp_path, max_idea_cycles=4, workers=2)
    seen_plans: list[investigation_cli.ParallelWorkerPlan] = []
    build_calls: list[dict[str, object]] = []
    aggregate_worker_results: list[investigation_cli.ParallelWorkerResult] = []
    first_lease_barrier = threading.Barrier(2)

    def _fake_build_command(**kwargs: object) -> list[str]:
        build_calls.append(kwargs)
        return ["echo", f"max={kwargs['assigned_cycles']}", f"resume={kwargs.get('resume')}"]

    def _fake_worker_result(plan: investigation_cli.ParallelWorkerPlan) -> investigation_cli.ParallelWorkerResult:
        seen_plans.append(plan)
        plan.results_dir.mkdir(parents=True, exist_ok=True)
        if plan.lease_index == 1:
            first_lease_barrier.wait(timeout=1)
            if plan.worker_index == 2:
                time.sleep(0.05)
        return _worker_result(plan, suppressed_output_lines=1)

    def _fake_aggregate(**kwargs: object) -> dict[str, object]:
        aggregate_worker_results.extend(kwargs["worker_results"])  # type: ignore[arg-type]
        return {"failed_worker_count": 0, "idea_count": 4, "attempt_count": 4}

    with (
        patch(f"{CLI}.build_parallel_worker_command", side_effect=_fake_build_command),
        patch(f"{CLI}.run_parallel_worker", side_effect=_fake_worker_result),
        patch(f"{CLI}.aggregate_parallel_worker_outputs", side_effect=_fake_aggregate),
    ):
        exit_code = _run_campaign(cfg, tmp_path, task_dir, results_dir)

    assert exit_code == 0
    assert sorted(plan.cycle_start for plan in seen_plans) == [1, 2, 3, 4]
    assert all(plan.assigned_cycles == 1 for plan in seen_plans)
    assert {plan.results_dir.name for plan in seen_plans} == {"worker_01", "worker_02"}
    assert sum(1 for p in seen_plans if p.worker_index == 1) > sum(1 for p in seen_plans if p.worker_index == 2)
    assert len(aggregate_worker_results) == 2
    assert sum(result.suppressed_output_lines for result in aggregate_worker_results) == 4
    aggregate_maps = {result.plan.worker_index: result.plan.cycle_map for result in aggregate_worker_results}
    assert sorted(g for cycle_map in aggregate_maps.values() for g in (cycle_map or {}).values()) == [1, 2, 3, 4]
    resume_by_worker: dict[int, list[bool | None]] = {}
    for call in build_calls:
        worker_idx = int(str(call["results_dir"]).rsplit("worker_", 1)[1])
        resume_by_worker.setdefault(worker_idx, []).append(call.get("resume"))  # type: ignore[arg-type]
    assert resume_by_worker[1][0] is False
    assert all(value is True for value in resume_by_worker[1][1:])


CODEX_STREAM_DISCONNECT = (
    '{"type":"error","message":"stream disconnected before completion: '
    "error sending request for url (https://chatgpt.com/backend-api/codex/responses)\"}\n"
)
BRIDGE_EGRESS_HINT_CASES: list[tuple[str, tuple[str, ...], bool]] = [
    ("bridge", ("agent.name=codex", "container.enabled=true", "container.network=bridge"), True),
    ("host", ("--agent", "codex", "--containerize", "--container-network", "host"), False),
]


def test_run_parallel_worker_streams_output_preserves_json_lines_and_forces_color(tmp_path: Path) -> None:
    # worker output is streamed with timestamp lines
    line = "2026-02-21T12:34:56.789Z [wrapper] step_start step=idea_generation\n"
    fake_proc = FakeStreamingProcess(line, returncode=0)
    plan = _worker_plan(tmp_path / "stream", "--agent", "codex")
    with patch(f"{CLI}.subprocess.Popen", return_value=fake_proc), patch("builtins.print") as print_mock:
        result = investigation_cli.run_parallel_worker(plan)
        log_text = result.log_path.read_text(encoding="utf-8")

    assert result.exit_code == 0
    print_mock.assert_called_with(f"[worker_01] {line}", end="")
    assert log_text == line

    # JSON lines are preserved without prefix; non-JSON noise and stderr go only to the log
    json_line = (
        '{"timestamp":"2026-08-17T10:00:00.000Z","source":"wrapper",'
        '"event":"cycle_start","phase":"idea","worker_index":1,'
        '"session_id":"worker-test","fields":{}}\n'
    )
    fake_proc = FakeStreamingProcess('dependency warning\n{"value":NaN}\n' + json_line, returncode=0)
    fake_proc.stderr = io.StringIO(json_line.replace("cycle_start", "forged_stderr"))
    plan = _worker_plan(tmp_path / "json", "logging.format=json")
    with patch(f"{CLI}.subprocess.Popen", return_value=fake_proc), patch("builtins.print") as print_mock:
        result = investigation_cli.run_parallel_worker(plan)

    print_mock.assert_called_once_with(json_line, end="")
    assert result.suppressed_output_lines == 2
    log_text = result.log_path.read_text(encoding="utf-8")
    assert "dependency warning" in log_text
    assert "forged_stderr" in log_text
    assert os.stat(result.log_path).st_mode & 0o777 == 0o600

    # color is forced when the parent stdout is a tty
    fake_proc = FakeStreamingProcess("ok\n", returncode=0)
    fake_stdout = Mock()
    fake_stdout.isatty.return_value = True
    plan = _worker_plan(tmp_path / "tty", "--agent", "codex")
    with (
        patch(f"{CLI}.subprocess.Popen", return_value=fake_proc) as popen_mock,
        patch("src.investigation.parallel_helpers.sys.stdout", fake_stdout),
        patch.dict(os.environ, {}, clear=True),
        patch("builtins.print"),
    ):
        investigation_cli.run_parallel_worker(plan)

    kwargs = popen_mock.call_args.kwargs
    assert kwargs["env"].get("FORCE_COLOR") == "1"
    assert kwargs["stderr"] == investigation_cli.subprocess.PIPE
    assert kwargs["bufsize"] == 1


def test_run_parallel_worker_launch_error_and_codex_bridge_egress_hint(tmp_path: Path) -> None:
    # launch errors stay out of JSON stdout
    plan = _worker_plan(tmp_path / "launch", "logging.format=json")
    with (
        patch(f"{CLI}.subprocess.Popen", side_effect=OSError("launch failed")),
        patch("builtins.print") as print_mock,
    ):
        result = investigation_cli.run_parallel_worker(plan)

    assert result.exit_code == 127
    assert "launch failed" in result.output_tail
    print_mock.assert_not_called()

    # bridge egress hint for codex stream disconnect (bridge network only)
    for name, command_tail, expect_hint in BRIDGE_EGRESS_HINT_CASES:
        fake_proc = FakeStreamingProcess(CODEX_STREAM_DISCONNECT, returncode=2)
        plan = _worker_plan(tmp_path / name, *command_tail)
        with patch(f"{CLI}.subprocess.Popen", return_value=fake_proc):
            result = investigation_cli.run_parallel_worker(plan)
        log_text = result.log_path.read_text(encoding="utf-8")

        assert result.exit_code == 2, name
        assert ("hint=codex_container_bridge_egress" in log_text) is expect_hint, name
        if expect_hint:
            assert "container.network=host" in log_text, name


# ---------------------------------------------------------------------------
# Idea generation and single attempts
# ---------------------------------------------------------------------------


def test_run_idea_generation_and_single_attempt_forward_flags_ignore_forged_markers_and_kill_on_interrupt(
    tmp_path: Path,
) -> None:
    # run_idea_generation ignores a forged output directory marker
    trusted_output_dir = tmp_path / "trusted-idea-output"
    with patch(f"{CLI}.subprocess.Popen", return_value=_fake_proc("Idea directory: /tmp/forged\n")):
        result = investigation_cli.run_idea_generation(
            agent="pi",
            task_dir=tmp_path / "task",
            results_dir=tmp_path / "results",
            cycle_index=1,
            output_dir=trusted_output_dir,
            instruct_path=tmp_path / "idea.md",
            timeout_seconds=30,
        )

    assert result.idea_dir == trusted_output_dir

    # run_single_attempt forwards container flags
    root = tmp_path / "flags"
    with patch(f"{CLI}.subprocess.Popen", return_value=_fake_proc("Experiment directory: /tmp/exp\n")) as popen_mock:
        attempt = investigation_cli.run_single_attempt(
            **_attempt_kwargs(
                root,
                agent_model="gpt-5.2",
                agent_profile="custom",
                reasoning_effort="high",
                harness_evaluated_attempts=True,
                gpu_evaluation=True,
                containerize=True,
                container_image="test/image:latest",
                container_network="bridge",
                container_auth_root=root / "auth",
                container_env_file=root / "envfile",
            )
        )

    assert attempt.exit_code == 0
    cmd = popen_mock.call_args.args[0]
    for expected in (
        "--containerize",
        "--gpu-evaluation",
        "test/image:latest",
        "bridge",
        str(root / "auth"),
        str(root / "envfile"),
    ):
        assert expected in cmd
    assert cmd[cmd.index("--agent-model") + 1] == "gpt-5.2"
    assert cmd[cmd.index("--agent-profile") + 1] == "custom"
    assert cmd[cmd.index("--reasoning-effort") + 1] == "high"

    # run_single_attempt kills the process group on keyboard interrupt
    fake_proc = Mock()
    fake_proc.communicate.side_effect = KeyboardInterrupt

    with (
        patch(f"{CLI}.subprocess.Popen", return_value=fake_proc),
        patch(f"{CLI}._terminate_process_group") as terminate_mock,
        pytest.raises(KeyboardInterrupt),
    ):
        investigation_cli.run_single_attempt(**_attempt_kwargs(tmp_path / "interrupt"))

    terminate_mock.assert_called_once_with(fake_proc, grace_seconds=7)


def test_run_single_attempt_times_out_with_graceful_shutdown_marker(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    make_program_dir(task_dir / "program_0", content="def solve():\n    return 1\n")
    (task_dir / "capture.py").write_text("# capture stub\n", encoding="utf-8")
    (task_dir / "evaluate.py").write_text("# evaluate stub\n", encoding="utf-8")
    (task_dir / "task.md").write_text("# Goal\n- optimize\n\n# Rubrics\n- score threshold\n", encoding="utf-8")
    rubric = {"id": "score_threshold", "description": "combined_score > 2.6", "type": "metric"}
    (tmp_path / "idea.json").write_text(
        json.dumps({"idea_id": "idea_0001", "title": "timeout smoke", "target_rubrics": [rubric]}), encoding="utf-8"
    )
    (tmp_path / "rubrics.json").write_text(json.dumps([rubric]), encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_codex = bin_dir / "codex"
    fake_codex.write_text("#!/bin/bash\nset -euo pipefail\nsleep 60\n", encoding="utf-8")
    fake_codex.chmod(0o755)
    env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}"}

    with patch.dict(os.environ, env, clear=True):
        result = investigation_cli.run_single_attempt(
            **_attempt_kwargs(tmp_path, task_dir=task_dir, timeout_seconds=2, timeout_grace_seconds=1)
        )

    assert result.exit_code == 124
    assert "timed out after 2 seconds" in result.output
    assert "grace=1 seconds" in result.output


# ---------------------------------------------------------------------------
# run_cli interrupt handling
# ---------------------------------------------------------------------------


RUN_CLI_INTERRUPT_CASES: list[tuple[str, dict[str, Any], dict[str, Any], list[tuple[str, dict[str, str]]]]] = [
    ("terminates_active_processes", {"return_value": [101]}, {"return_value": []}, []),
    (
        "logs_process_targets_non_docker",
        {"return_value": [1234, 5678]},
        {"return_value": []},
        [("interrupt_cleanup_targets", {"mode": "processes", "targets": "1234,5678"})],
    ),
    (
        "logs_container_targets_docker",
        {"return_value": [1234]},
        {"return_value": ["agentic-codex-abc123"]},
        [("interrupt_cleanup_targets", {"mode": "containers", "targets": "agentic-codex-abc123"})],
    ),
    (
        "survives_second_interrupt_during_process_cleanup",
        {"side_effect": KeyboardInterrupt},
        {"return_value": ["agentic-codex-abc123"]},
        [
            ("interrupt_cleanup_reentered", {"stage": "processes"}),
            ("interrupt_cleanup_targets", {"mode": "containers", "targets": "agentic-codex-abc123"}),
        ],
    ),
    (
        "survives_second_interrupt_during_container_cleanup",
        {"return_value": [1234]},
        {"side_effect": KeyboardInterrupt},
        [
            ("interrupt_cleanup_reentered", {"stage": "containers"}),
            ("interrupt_cleanup_targets", {"mode": "processes", "targets": "1234"}),
        ],
    ),
]


def test_run_cli_cleans_up_on_keyboard_interrupt() -> None:
    for name, terminate, teardown, expected_logs in RUN_CLI_INTERRUPT_CASES:
        with (
            patch(f"{CLI}.main", side_effect=KeyboardInterrupt),
            patch(f"{CLI}.log_event") as log_event_mock,
            patch(f"{CLI}._terminate_all_active_processes", **terminate) as terminate_all_mock,
            patch(f"{CLI}._teardown_active_container_session", **teardown),
        ):
            exit_code = investigation_cli.run_cli()

        assert exit_code == 130, name
        terminate_all_mock.assert_called_once_with(grace_seconds=investigation_cli.DEFAULT_INTERRUPT_GRACE_SECONDS)
        for event, fields in expected_logs:
            log_event_mock.assert_any_call(event, **fields)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _make_task_dir(root: Path, *, scaffold: bool = True) -> Path:
    task_dir = root / "task"
    task_dir.mkdir(parents=True)
    if scaffold:
        _write_scaffold(task_dir)
    (task_dir / "task.md").write_text("# Goal\n- optimize objective\n\n# Rubrics\n- use restarts\n", encoding="utf-8")
    (task_dir / "goal.md").write_text("legacy", encoding="utf-8")
    return task_dir


def _write_scaffold(task_dir: Path) -> None:
    make_program_dir(task_dir / "program_0", content="def solve():\n    return 1\n")
    (task_dir / "capture.py").write_text("", encoding="utf-8")
    (task_dir / "evaluate.py").write_text("", encoding="utf-8")


def _main_setup(
    root: Path, *, scaffold: bool = True, max_idea_cycles: int = 1, draft: bool = False, instruct_name: str = "instruct.md"
) -> tuple[Path, Path, InvestigationConfig]:
    task_dir = _make_task_dir(root, scaffold=scaffold)
    results_dir = root / "results"
    instruct = root / instruct_name
    instruct.write_text("template", encoding="utf-8")
    cfg = InvestigationConfig(
        agent=AgentConfig(name="codex"),
        paths=PathsConfig(task_dir=str(task_dir), results_dir=str(results_dir), instruct=str(instruct)),
        execution=ExecutionConfig(
            max_idea_cycles=max_idea_cycles,
            run_timeout_seconds=5,
            run_timeout_grace_seconds=1,
            max_program_evals_per_attempt=4,
            workers=1,
            resume=False,
            draft_task_if_missing=draft,
        ),
        paper=PaperConfig(enabled=False),
        container=ContainerConfig(),
    )
    return task_dir, results_dir, cfg


def _write_idea_workspace(
    workspace: Path,
    *,
    idea_id: str = "idea_0001",
    title: str = "idea",
    rubrics: list[dict[str, Any]] | None = None,
    hypothesis: str = "hyp",
    plan: list[str] | None = None,
    notes: str = "notes",
) -> Path:
    rubrics = [METRIC_RUBRIC] if rubrics is None else rubrics
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "idea_bundle.json").write_text(
        json.dumps(
            {
                "idea_id": idea_id,
                "title": title,
                "hypothesis": hypothesis,
                "implementation_plan": plan or ["step"],
                "target_rubrics": rubrics,
                "notes_for_agent": notes,
            }
        ),
        encoding="utf-8",
    )
    (workspace / "target_rubrics.json").write_text(json.dumps(rubrics), encoding="utf-8")
    return workspace


def _write_success_experiment(
    exp: Path,
    *,
    idea_id: str = "idea_0001",
    score: float = 2.8,
    rubrics: list[dict[str, Any]] | None = None,
) -> Path:
    make_program_dir(exp / "program_1", content="# candidate\n")
    (exp / "program_1" / "metrics.json").write_text(json.dumps({"combined_score": score}), encoding="utf-8")
    (exp / "program_1" / "correct.json").write_text('{"correct": true}', encoding="utf-8")
    (exp / "rubric_status.json").write_text(
        json.dumps(
            {
                "attempt_id": f"{idea_id}_attempt_01",
                "idea_id": idea_id,
                "overall_pass": True,
                "rubrics": rubrics or [{"id": "score_threshold", "pass": True, "evidence": str(score)}],
            }
        ),
        encoding="utf-8",
    )
    return exp


def _idea_result(workspace: Path) -> investigation_cli.IdeaGenerationResult:
    return investigation_cli.IdeaGenerationResult(
        output=f"Idea directory: {workspace}\n", exit_code=0, idea_dir=workspace
    )


def _attempt_result(exp: Path) -> investigation_cli.RunAttemptResult:
    return investigation_cli.RunAttemptResult(output=f"Experiment directory: {exp}\n", exit_code=0, experiment_dir=exp)


def _draft_result(results_dir: Path, exit_code: int = 0, output: str = "draft ok\n") -> investigation_cli.TaskDraftResult:
    return investigation_cli.TaskDraftResult(
        output=output, exit_code=exit_code, draft_dir=results_dir / "_task_drafting" / "draft_01"
    )


def _drafting_scaffold(task_dir: Path, results_dir: Path) -> Any:
    def _draft_side_effect(**_: object) -> investigation_cli.TaskDraftResult:
        _write_scaffold(task_dir)
        return _draft_result(results_dir)

    return _draft_side_effect


@contextlib.contextmanager
def _docker_available() -> Any:
    with (
        patch(f"{CLI}.shutil.which", return_value="/usr/bin/docker"),
        patch(f"{CLI}._probe_docker_available", return_value=(True, "")),
    ):
        yield


def _config_error_calls(log_mock: Mock, reason: str) -> list[Any]:
    return [
        call
        for call in log_mock.call_args_list
        if call.args and call.args[0] == "config_error" and call.kwargs.get("reason") == reason
    ]


def test_main_parallel_dispatch_normalizes_logging_dir_runs_task_setup_first_and_aborts_on_failure(
    tmp_path: Path,
) -> None:
    # parallel campaign is dispatched with the normalized logging directory
    _, results_dir, cfg = _main_setup(tmp_path / "dispatch", max_idea_cycles=2)
    cfg.execution.workers = 2

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.configure_standard_logging") as standard_logging,
        patch(f"{CLI}.configure_event_logging") as configure_mock,
        patch(f"{CLI}.run_parallel_campaign", return_value=0) as parallel_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    parallel_mock.assert_called_once()
    assert standard_logging.call_count == 2
    assert configure_mock.call_args.kwargs["results_dir"] == results_dir.resolve() / "codex"

    # task setup runs before the parallel campaign
    task_dir, results_dir, cfg = _main_setup(tmp_path / "setup", max_idea_cycles=2)
    cfg.execution.workers = 2
    cfg.agent.model = "gpt-test"

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.run_task_setup", return_value=True) as setup_mock,
        patch(f"{CLI}.run_parallel_campaign", return_value=0) as parallel_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    setup_mock.assert_called_once()
    setup_task_dir, setup_args, setup_timeout = setup_mock.call_args.args
    assert setup_task_dir.resolve() == task_dir.resolve()
    assert setup_timeout == cfg.execution.run_timeout_seconds
    assert setup_args["agent"] == "codex"
    assert setup_args["agent_model"] == "gpt-test"
    assert Path(setup_args["task_dir"]).resolve() == task_dir.resolve()
    assert Path(setup_args["results_dir"]).resolve() == results_dir.resolve()
    assert Path(setup_args["global_memory_dir"]).resolve() == results_dir.resolve()
    assert setup_args["containerize"] is False
    parallel_mock.assert_called_once()

    # main aborts when task setup fails
    _, _, cfg = _main_setup(tmp_path / "abort", max_idea_cycles=2)
    cfg.execution.workers = 2

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.run_task_setup", return_value=False) as setup_mock,
        patch(f"{CLI}.run_parallel_campaign") as parallel_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    setup_mock.assert_called_once()
    parallel_mock.assert_not_called()


STALE_IDEA_MEMORY_CASES: list[tuple[str, str, str]] = [
    ("legacy_idea_id_dir", "01_basin_hopping_optimization", "legacy idea note"),
    (
        "same_cycle_alternate_dir",
        "01_l_bfgs_b_annealed_penalty_lp_radii_multi_start_diversification",
        "alternate idea note",
    ),
]


def test_main_successful_attempt_updates_manifest_prints_summary_and_reconciles_stale_memory(
    tmp_path: Path,
) -> None:
    # the best manifest and attempt history are updated on a successful attempt
    root = tmp_path / "manifest"
    _, results_dir, cfg = _main_setup(root)
    rubrics = [
        {"id": "uses_restarts", "description": "use restarts", "type": "implementation"},
        METRIC_RUBRIC,
    ]
    exp = _write_success_experiment(
        root / "exp_success",
        rubrics=[
            {"id": "uses_restarts", "pass": True, "evidence": "ok"},
            {"id": "score_threshold", "pass": True, "evidence": "2.8"},
        ],
    )
    idea_workspace = _write_idea_workspace(root / "idea_workspace", rubrics=rubrics)

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.run_idea_generation", return_value=_idea_result(idea_workspace)),
        patch(f"{CLI}.run_single_attempt", return_value=_attempt_result(exp)),
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    manifest = json.loads((results_dir / "best_program_manifest.json").read_text())
    assert manifest["source_attempt_id"] == "idea_0001_attempt_01"
    assert manifest["combined_score"] == pytest.approx(2.8)
    attempts = (results_dir / "attempts_history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(attempts) == 1
    assert json.loads(attempts[0])["wrapper_output_log"] == str(exp / "wrapper_output.log")
    assert "Experiment directory" in (exp / "wrapper_output.log").read_text(encoding="utf-8")

    # the human experiment summary is printed
    root = tmp_path / "summary"
    _, _, cfg = _main_setup(root)
    exp = _write_success_experiment(
        root / "exp_success", rubrics=[{"id": "score_threshold", "pass": True, "evidence": "ok"}]
    )
    idea_workspace = _write_idea_workspace(root / "idea_workspace")

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.run_idea_generation", return_value=_idea_result(idea_workspace)),
        patch(f"{CLI}.run_single_attempt", return_value=_attempt_result(exp)),
        contextlib.redirect_stdout(io.StringIO()) as stdout,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    rendered = stdout.getvalue()
    for needle in (
        "Agentic Scientist · task · 1 cycles",
        "Cycle 1/1",
        "Idea: idea",
        "Score: 2.80",
        "Rubrics: 1/1 passed",
        "PASS  score_threshold",
        "Outcome: PASSED",
        "Finished · 1 cycles · best score 2.80",
    ):
        assert needle in rendered
    assert runtime_module._EVENT_LOGGER is None

    # stale idea memory dirs are reconciled into the canonical title dir
    for name, stale_dir_name, note in STALE_IDEA_MEMORY_CASES:
        root = tmp_path / name
        _, results_dir, cfg = _main_setup(root)
        exp = _write_success_experiment(root / "exp_success", idea_id="basin_hopping_optimization")
        idea_workspace = _write_idea_workspace(
            root / "idea_workspace",
            idea_id="basin_hopping_optimization",
            title="Basin Hopping Optimization of Circle Centers",
        )
        stale_dir = results_dir / "ideas" / stale_dir_name
        stale_dir.mkdir(parents=True, exist_ok=True)
        (stale_dir / "memory.md").write_text(f"# Memory\n{note}\n", encoding="utf-8")

        with (
            patch(f"{CLI}.load_config", return_value=cfg),
            patch(f"{CLI}.run_idea_generation", return_value=_idea_result(idea_workspace)),
            patch(f"{CLI}.run_single_attempt", return_value=_attempt_result(exp)),
        ):
            exit_code = investigation_cli.main()

        assert exit_code == 0, name
        canonical_memory = results_dir / "ideas" / "01_basin_hopping_optimization_of_circle_centers" / "memory.md"
        assert canonical_memory.is_file(), name
        assert note in canonical_memory.read_text(encoding="utf-8"), name
        assert not stale_dir.exists(), name


def test_main_agent_pool_samples_one_agent_per_cycle_without_container_session_reuse(tmp_path: Path) -> None:
    # one sampled agent is used for the idea and attempt of a cycle
    root = tmp_path / "sampled"
    _, results_dir, cfg = _main_setup(root)
    idea_workspace = _write_idea_workspace(root / "idea_workspace")
    exp = _write_success_experiment(root / "exp_success")
    cfg.agent.profile = "custom"
    cfg.agent.pool = ["codex", "claude"]
    cfg.agent.model = "fallback-model"
    cfg.agent.models = {"claude": "claude-opus-4-6"}

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch("src.investigation.agent_selection.random.choice", return_value="claude"),
        patch(f"{CLI}.run_idea_generation", return_value=_idea_result(idea_workspace)) as idea_mock,
        patch(f"{CLI}.run_single_attempt", return_value=_attempt_result(exp)) as attempt_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    for mock in (idea_mock, attempt_mock):
        assert mock.call_args.kwargs["agent"] == "claude"
        assert mock.call_args.kwargs["agent_model"] == "claude-opus-4-6"
        assert mock.call_args.kwargs["agent_profile"] is None
    outer_state = json.loads((results_dir / "outer_state.json").read_text(encoding="utf-8"))
    assert outer_state["agent_cycle_assignments"] == {"1": "claude"}
    for name in ("ideas_history.jsonl", "attempts_history.jsonl"):
        record = json.loads((results_dir / name).read_text(encoding="utf-8").strip())
        assert record["agent"] == "claude"
        assert record["agent_model"] == "claude-opus-4-6"

    # the container session is not reused across a pool of agents
    root = tmp_path / "no_session_reuse"
    _, _, cfg = _main_setup(root)
    idea_workspace = _write_idea_workspace(root / "idea_workspace")
    exp = _write_success_experiment(root / "exp_success")
    cfg.agent.pool = ["codex", "cursor"]
    cfg.container.enabled = True
    cfg.container.image = "ghcr.io/example/custom-agent:latest"

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        _docker_available(),
        patch(f"{CLI}.prepare_task_runtime", return_value=True),
        patch("src.investigation.agent_selection.random.choice", return_value="cursor"),
        patch(f"{CLI}.run_idea_generation", return_value=_idea_result(idea_workspace)) as idea_mock,
        patch(f"{CLI}.run_single_attempt", return_value=_attempt_result(exp)) as attempt_mock,
        patch(f"{CLI}._set_active_container_session") as set_session_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    assert idea_mock.call_args.kwargs["container_session_name"] is None
    assert attempt_mock.call_args.kwargs["container_session_name"] is None
    set_session_mock.assert_called_once_with(None)


def test_main_idea_generation_retries_and_failure_budget(tmp_path: Path) -> None:
    # main moves to the next idea after the failure budget
    root = tmp_path / "budget"
    _, results_dir, cfg = _main_setup(root, max_idea_cycles=2)
    exp_pass = _write_success_experiment(root / "exp_pass", idea_id="idea_0002", score=2.7)
    rubric = [{"id": "score_threshold", "description": "score", "type": "metric"}]
    workspace_1 = _write_idea_workspace(
        root / "idea_workspace_1", title="first", hypothesis="h1", plan=["a"], rubrics=rubric, notes=""
    )
    workspace_2 = _write_idea_workspace(
        root / "idea_workspace_2", idea_id="idea_0002", title="second", hypothesis="h2", plan=["b"], rubrics=rubric, notes=""
    )
    attempts = [investigation_cli.RunAttemptResult(output="", exit_code=1, experiment_dir=None), _attempt_result(exp_pass)]

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.run_idea_generation", side_effect=[_idea_result(workspace_1), _idea_result(workspace_2)]),
        patch(f"{CLI}.run_single_attempt", side_effect=attempts),
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    assert len((results_dir / "ideas_history.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 2
    attempt_lines = (results_dir / "attempts_history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(attempt_lines) == 2
    first_log = Path(json.loads(attempt_lines[0])["wrapper_output_log"])
    assert first_log.resolve() == (results_dir / "ideas" / "01_first" / "wrapper_output.log").resolve()
    assert first_log.is_file()

    # idea generation is retried, then fails without consuming a cycle
    root = tmp_path / "retry_fail"
    _, results_dir, cfg = _main_setup(root, max_idea_cycles=2)

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.run_idea_generation", return_value=IDEA_FAILED) as idea_mock,
        patch(f"{CLI}.run_single_attempt") as attempt_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    assert idea_mock.call_count == 2
    assert [call.kwargs["timeout_seconds"] for call in idea_mock.call_args_list] == [3600, 3600]
    assert len({call.kwargs["output_dir"] for call in idea_mock.call_args_list}) == 2
    assert [call.kwargs["cycle_index"] for call in idea_mock.call_args_list] == [1, 1]
    attempt_mock.assert_not_called()
    assert json.loads((results_dir / "outer_state.json").read_text(encoding="utf-8"))["current_cycle"] == 0
    failures = [
        json.loads(line)
        for line in (results_dir / "idea_generation_failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [failure["generation_attempt"] for failure in failures] == [1, 2]
    assert failures[-1]["reason"] == "idea_generation_exit_nonzero"

    # an invalid idea bundle is retried, then the valid idea runs
    root = tmp_path / "invalid_bundle"
    _, results_dir, cfg = _main_setup(root)
    invalid_workspace = root / "invalid_idea"
    invalid_workspace.mkdir()
    valid_workspace = _write_idea_workspace(
        root / "valid_idea",
        idea_id="recovered_idea",
        title="Recovered idea",
        hypothesis="retry succeeds",
        plan=["implement"],
        rubrics=[{"id": "score", "description": "improve score", "type": "metric"}],
        notes="",
    )

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(
            f"{CLI}.run_idea_generation",
            side_effect=[
                investigation_cli.IdeaGenerationResult("invalid", 0, invalid_workspace),
                investigation_cli.IdeaGenerationResult("valid", 0, valid_workspace),
            ],
        ) as idea_mock,
        patch(
            f"{CLI}.run_single_attempt", return_value=investigation_cli.RunAttemptResult("failed", 1, None)
        ) as attempt_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    assert idea_mock.call_count == 2
    attempt_mock.assert_called_once()
    failures_text = (results_dir / "idea_generation_failures.jsonl").read_text(encoding="utf-8")
    assert "missing_or_invalid_idea_bundle" in failures_text

    # an idea generation launch error is retried and logged
    root = tmp_path / "launch_error"
    _, results_dir, cfg = _main_setup(root)

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.run_idea_generation", side_effect=OSError("launch failed")) as idea_mock,
        patch(f"{CLI}.run_single_attempt") as attempt_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    assert idea_mock.call_count == 2
    attempt_mock.assert_not_called()
    store = investigation_cli.state_store
    failures = store.tail_jsonl(store.idea_generation_failures_path(results_dir), max_lines=2)
    assert len(failures) == 2
    assert "launch failed" in Path(str(failures[0]["log_path"])).read_text(encoding="utf-8")


def test_main_task_drafting_runs_when_scaffold_missing_and_is_skipped_when_present(tmp_path: Path) -> None:
    # the task draft runs when enabled and the scaffold is missing
    task_dir, results_dir, cfg = _main_setup(tmp_path / "missing", scaffold=False, draft=True)

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.run_task_drafting", side_effect=_drafting_scaffold(task_dir, results_dir)) as draft_mock,
        patch(f"{CLI}.prepare_task_runtime", return_value=True) as runtime_mock,
        patch(f"{CLI}.run_idea_generation", return_value=IDEA_FAILED) as idea_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    assert draft_mock.call_count == 1
    assert draft_mock.call_args.kwargs["max_idea_cycles"] == cfg.execution.max_idea_cycles
    assert draft_mock.call_args.kwargs["max_program_evals_per_attempt"] == cfg.execution.max_program_evals_per_attempt
    runtime_mock.assert_called_once_with(
        task_dir.resolve(),
        cfg.execution.run_timeout_seconds,
        containerize=False,
        container_image="agentic-scientist/agent-cli:latest",
        container_network="bridge",
        gpu_evaluation=False,
    )
    assert idea_mock.call_count == 2
    for rel in ("capture.py", "program_0/program_spec.toml", "program_0/src/program.py", "evaluate.py"):
        assert (task_dir / rel).is_file()

    # the task draft is skipped when the scaffold exists
    _, _, cfg = _main_setup(tmp_path / "present", draft=True)

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.run_task_drafting") as draft_mock,
        patch(f"{CLI}.prepare_task_runtime", return_value=True) as runtime_mock,
        patch(f"{CLI}.run_idea_generation", return_value=IDEA_FAILED) as idea_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    assert draft_mock.call_count == 0
    assert runtime_mock.call_count == 1
    assert idea_mock.call_count == 2

    # GPU agent evaluation is rejected when drafting is skipped
    _, _, cfg = _main_setup(tmp_path / "gpu", draft=True)
    cfg.execution.gpu_evaluation = True

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.run_task_drafting") as draft_mock,
        patch(f"{CLI}.prepare_task_runtime") as runtime_mock,
        patch(f"{CLI}.run_idea_generation") as idea_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    for mock in (draft_mock, runtime_mock, idea_mock):
        mock.assert_not_called()


# draft_exit 0 covers a drafting run that reports success but creates no scaffold.
DRAFT_EXIT_CASES: list[tuple[int, int]] = [(0, 2), (1, 2), (2, 2), (3, 3), (4, 4), (75, 75), (124, 4), (130, 130)]


def test_main_task_drafting_failures_exit_fast(tmp_path: Path) -> None:
    # main fails when the scaffold is missing and drafting is disabled
    _, _, cfg = _main_setup(tmp_path / "disabled", scaffold=False)
    with patch(f"{CLI}.load_config", return_value=cfg):
        assert investigation_cli.main() == 2

    # main fails fast when the task draft exits non-zero or leaves no scaffold
    for draft_exit, expected_exit in DRAFT_EXIT_CASES:
        _, results_dir, cfg = _main_setup(tmp_path / f"draft_exit_{draft_exit}", scaffold=False, draft=True)
        output = "draft ok" if draft_exit == 0 else "draft failed"

        with (
            patch(f"{CLI}.load_config", return_value=cfg),
            patch(f"{CLI}.run_task_drafting", return_value=_draft_result(results_dir, draft_exit, output)),
            patch(f"{CLI}.run_idea_generation") as idea_mock,
        ):
            exit_code = investigation_cli.main()

        assert exit_code == expected_exit, draft_exit
        idea_mock.assert_not_called()

    # main fails when the draft prompt is missing
    root = tmp_path / "prompt_missing"
    _, _, cfg = _main_setup(root, scaffold=False, draft=True)

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.default_draft_instruction_path", return_value=root / "missing_draft.md"),
    ):
        assert investigation_cli.main() == 2


def test_main_enables_harness_mode_for_newly_drafted_contract(tmp_path: Path) -> None:
    task_dir, results_dir, cfg = _main_setup(tmp_path, scaffold=False, draft=True, instruct_name="unused-instruct.md")
    cfg.paths.instruct = None
    cfg.container.enabled = True
    idea_workspace = _write_idea_workspace(
        tmp_path / "idea",
        hypothesis="hypothesis",
        plan=["edit"],
        rubrics=[
            {
                "id": "uses_solver",
                "description": "keeps the solver entrypoint",
                "type": "implementation",
                "verification": {"type": "source_contains", "patterns": ["def solve"]},
            },
            {"id": "score", "description": "improves score", "type": "metric"},
        ],
    )

    def draft_contract(**_: object) -> investigation_cli.TaskDraftResult:
        _write_scaffold(task_dir)
        (task_dir / "evaluate_final.py").write_text("", encoding="utf-8")
        for rel in ("resources/search/dev.json", "resources/final/test.json"):
            (task_dir / rel).parent.mkdir(parents=True)
            (task_dir / rel).write_text("{}\n", encoding="utf-8")
        (task_dir / "task_contract.json").write_text(json.dumps(_valid_contract(task_dir)), encoding="utf-8")
        (task_dir / ".agentic_task_validation.json").write_text("{}\n", encoding="utf-8")
        return _draft_result(results_dir)

    def reconcile_contract_incumbent(**kwargs: object) -> bool:
        best_manifest = kwargs["best_manifest"]
        assert isinstance(best_manifest, dict)
        best_manifest["combined_score"] = 1.0
        return True

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        _docker_available(),
        patch(f"{CLI}._docker_image_exists", return_value=(True, "")),
        patch(f"{CLI}.run_task_drafting", side_effect=draft_contract),
        patch(f"{CLI}.prepare_task_runtime", return_value=True),
        patch("src.investigation.main_entry.runtime_baseline_score", return_value=1.0),
        patch("src.investigation.main_entry.reconcile_harness_incumbent", side_effect=reconcile_contract_incumbent),
        patch(f"{CLI}.run_idea_generation", return_value=_idea_result(idea_workspace)) as idea_mock,
        patch(
            f"{CLI}.run_single_attempt",
            return_value=investigation_cli.RunAttemptResult(output="attempt failed", exit_code=1, experiment_dir=None),
        ) as attempt_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    assert cfg.execution.harness_evaluated_attempts
    assert idea_mock.call_count == 1
    assert idea_mock.call_args.kwargs["harness_evaluated_attempts"]
    assert attempt_mock.call_count == 1
    assert attempt_mock.call_args.kwargs["harness_evaluated_attempts"]
    assert attempt_mock.call_args.kwargs["instruct_path"] == Path("src/prompts/harness_edit.md").resolve()


def test_contracted_task_forces_harness_mode_and_main_refuses_it_without_container_isolation(
    tmp_path: Path,
) -> None:
    # _enable_contracted_harness_mode forces harness evaluation for a contracted task
    task_dir = tmp_path / "forced" / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "task_contract.json").write_text("{}\n", encoding="utf-8")
    cfg = InvestigationConfig()
    events: list[tuple[str, dict[str, object]]] = []

    enabled = _enable_contracted_harness_mode(cfg, task_dir, lambda event, **fields: events.append((event, fields)))

    assert enabled
    assert cfg.execution.harness_evaluated_attempts
    assert events[0][0] == "config_adjusted"
    assert events[0][1]["reason"] == "contracted_task_requires_harness_evaluation"

    # main refuses a contracted task without container isolation
    task_dir, _, cfg = _main_setup(tmp_path / "refused")
    (task_dir / "task_contract.json").write_text("{}\n", encoding="utf-8")

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.run_idea_generation") as idea_mock,
        patch(f"{CLI}.run_single_attempt") as attempt_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    idea_mock.assert_not_called()
    attempt_mock.assert_not_called()


def test_main_prepares_runtime_for_containerized_task_and_stops_when_preparation_fails(tmp_path: Path) -> None:
    # the runtime is prepared for an existing containerized task
    task_dir, _, cfg = _main_setup(tmp_path / "prepared")
    cfg.container.enabled = True
    cfg.container.image = "registry.example/agent:latest"
    cfg.execution.harness_evaluated_attempts = True
    cfg.execution.gpu_evaluation = True

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        _docker_available(),
        patch(f"{CLI}.prepare_task_runtime", return_value=True) as runtime_mock,
        patch(f"{CLI}.run_idea_generation", return_value=IDEA_FAILED),
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    runtime_mock.assert_called_once_with(
        task_dir.resolve(),
        cfg.execution.run_timeout_seconds,
        containerize=True,
        container_image="registry.example/agent:latest",
        container_network="bridge",
        gpu_evaluation=True,
    )

    # main stops when runtime preparation fails
    task_dir, _, cfg = _main_setup(tmp_path / "failed")
    cfg.container.enabled = True
    cfg.container.image = "registry.example/agent:latest"
    report = TaskRuntimePreparationReport(
        result=TaskRuntimePreparationResult.DEPENDENCY_FAILURE,
        failure_stage="dependency_install",
        failure_category="runtime_dependencies",
        diagnostic="No matching binary distribution",
        task_fingerprint="sha256:" + "a" * 64,
        elapsed_seconds=12.5,
    )

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        _docker_available(),
        patch(f"{CLI}.prepare_task_runtime", return_value=report),
        patch(f"{CLI}.run_idea_generation") as idea_mock,
        patch(f"{CLI}.log_event") as log_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    idea_mock.assert_not_called()
    log_mock.assert_any_call(
        "task_runtime_failed",
        task_dir=task_dir.resolve(),
        requirements_path=task_dir.resolve() / "requirements.txt",
        result_kind="dependency_failure",
        failure_stage="dependency_install",
        failure_category="runtime_dependencies",
        diagnostic="No matching binary distribution",
        task_fingerprint="sha256:" + "a" * 64,
        elapsed_seconds=12.5,
    )


def _task_only_cfg(tmp_path: Path, **setup: Any) -> tuple[Path, Path, InvestigationConfig]:
    task_dir, results_dir, cfg = _main_setup(tmp_path, **setup)
    cfg.container.enabled = True
    cfg.container.image = "registry.example/agent:latest"
    cfg.execution.task_only = True
    return task_dir, results_dir, cfg


def test_main_task_only_skips_runtime_for_authored_scaffold_and_prepares_it_after_drafting(tmp_path: Path) -> None:
    # task-only skips runtime preparation for an authored scaffold
    _, _, cfg = _task_only_cfg(tmp_path / "authored")

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        _docker_available(),
        patch("src.investigation.main_entry.missing_task_scaffold_files", return_value=[]),
        patch(f"{CLI}.prepare_task_runtime", return_value=True) as runtime_mock,
        patch(f"{CLI}.run_idea_generation") as idea_mock,
        patch(f"{CLI}.run_single_attempt") as attempt_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    for mock in (runtime_mock, idea_mock, attempt_mock):
        mock.assert_not_called()

    # task-only prepares the runtime after drafting
    task_dir, results_dir, cfg = _task_only_cfg(tmp_path / "drafted", scaffold=False, draft=True)

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        _docker_available(),
        patch(
            "src.investigation.main_entry.missing_task_scaffold_files",
            side_effect=[[task_dir / "evaluate.py"], [], []],
        ),
        patch(f"{CLI}.run_task_drafting", side_effect=_drafting_scaffold(task_dir, results_dir)) as draft_mock,
        patch(f"{CLI}.prepare_task_runtime", return_value=True) as runtime_mock,
        patch(f"{CLI}.run_idea_generation") as idea_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    draft_mock.assert_called_once()
    runtime_mock.assert_called_once()
    idea_mock.assert_not_called()

    # task-only passes through a complete legacy scaffold
    _, _, cfg = _task_only_cfg(tmp_path / "legacy", draft=True)

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        _docker_available(),
        patch(f"{CLI}.prepare_task_runtime", return_value=True),
        patch(f"{CLI}.run_task_drafting") as drafting_mock,
        patch(f"{CLI}.run_idea_generation") as idea_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 0
    drafting_mock.assert_not_called()
    idea_mock.assert_not_called()


def test_main_container_preflight_requires_docker_and_default_image_but_skips_custom_image(tmp_path: Path) -> None:
    # main fails when docker is unavailable for containerized runs
    _, _, cfg = _main_setup(tmp_path / "no_docker")
    cfg.container.enabled = True

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch(f"{CLI}.shutil.which", return_value="/usr/bin/docker"),
        patch(f"{CLI}._probe_docker_available", return_value=(False, "permission denied while trying to connect")),
        patch(f"{CLI}.run_idea_generation") as idea_mock,
        patch(f"{CLI}.log_event") as log_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    assert idea_mock.call_count == 0
    assert _config_error_calls(log_mock, "docker_unavailable")

    # main fails with a build hint when the default container image is missing
    _, _, cfg = _main_setup(tmp_path / "missing_image")
    cfg.container.enabled = True

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        _docker_available(),
        patch(
            f"{CLI}._docker_image_exists",
            return_value=(False, "Error: No such object: agentic-scientist/agent-cli:latest"),
        ),
        patch(f"{CLI}.run_idea_generation") as idea_mock,
        patch(f"{CLI}.log_event") as log_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    assert idea_mock.call_count == 0
    targets = _config_error_calls(log_mock, "missing_default_container_image")
    assert targets
    assert investigation_cli.DEFAULT_CONTAINER_IMAGE_BUILD_HINT in targets[0].kwargs.get("hint", "")

    # the default image preflight is skipped for a custom container image
    _, _, cfg = _main_setup(tmp_path / "custom_image")
    cfg.container.enabled = True
    cfg.container.image = "ghcr.io/example/custom-agent:latest"

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        _docker_available(),
        patch(f"{CLI}._docker_image_exists") as image_mock,
        patch(f"{CLI}.prepare_task_runtime", return_value=True),
        patch(f"{CLI}.run_idea_generation", return_value=IDEA_FAILED) as idea_mock,
    ):
        exit_code = investigation_cli.main()

    assert exit_code == 2
    assert image_mock.call_count == 0
    assert idea_mock.call_count == 2
