"""Host preparation progress: scoped emission, runtime step tracking, and CLI rendering."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.investigation import cli as investigation_cli
from src.investigation import preparation_progress as progress
from src.investigation import task_runtime
from tests.program_testutils import make_program_dir


@pytest.fixture
def events(monkeypatch):
    captured = []
    monkeypatch.setattr(progress, "log_event", lambda event, **fields: captured.append({"event": event, **fields}))
    return captured


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str, str]]]:
    """Step/state pairs emitted inside an active ``preparation_progress`` scope."""
    recorded: list[tuple[str, str]] = []

    def record(event: str, **fields: object) -> None:
        assert event == "preparation_step"
        recorded.append((str(fields["step"]), str(fields["state"])))

    monkeypatch.setattr(progress, "log_event", record)
    with progress.preparation_progress():
        yield recorded


# --------------------------------------------------------------------------- preparation_progress


def test_preparation_progress_scoping_elapsed_failures_and_label_validation(monkeypatch, events):
    # steps outside preparation emit nothing
    with progress.preparation_step("runtime_copy"):
        pass
    assert events == []

    # nested preparation restores outer scope
    with progress.preparation_progress():
        with progress.preparation_step("runtime_cache"):
            with progress.preparation_progress():
                with progress.preparation_step("dependency_install"):
                    pass
            with progress.preparation_step("baseline_evaluation"):
                pass
    with progress.preparation_step("runtime_copy"):
        pass
    assert [(event["step"], event["state"]) for event in events] == [
        ("runtime_cache", "started"), ("dependency_install", "started"),
        ("dependency_install", "completed"), ("baseline_evaluation", "started"),
        ("baseline_evaluation", "completed"), ("runtime_cache", "completed"),
    ]

    # progress reports elapsed time
    events.clear()
    ticks = iter([100.0, 165.0])
    with monkeypatch.context() as m:
        m.setattr(progress.time, "monotonic", lambda: next(ticks))
        with progress.preparation_progress(), progress.preparation_step("runtime_copy"):
            pass
    assert events[-1]["elapsed_seconds"] == 65.0
    assert events[-1]["label"] == "Copying prepared runtime into task"

    # status failure is not reported as completed
    events.clear()
    with progress.preparation_progress(), progress.preparation_step("search_smoke") as step:
        step.fail()
    assert events[-1]["state"] == "failed"

    # exception reports failure without exposing details
    for error in (RuntimeError("private diagnostic"), KeyboardInterrupt()):
        events.clear()
        with pytest.raises(type(error)) as caught:
            with progress.preparation_progress(), progress.preparation_step("final_smoke"):
                raise error
        assert caught.value is error, repr(error)
        assert events[-1]["state"] == "failed", repr(error)
        assert "private diagnostic" not in str(events), repr(error)
        with progress.preparation_step("runtime_copy"):
            pass
        assert len(events) == 2, repr(error)

    # broken progress sink does not replace operation outcome
    def unavailable(*args, **kwargs):
        raise OSError("terminal unavailable")

    with monkeypatch.context() as m:
        m.setattr(progress, "log_event", unavailable)
        with progress.preparation_progress(), progress.preparation_step("runtime_copy"):
            result = 42
    assert result == 42

    # unknown labels are rejected before emission
    events.clear()
    with pytest.raises(ValueError, match="Unknown preparation step"):
        with progress.preparation_progress(), progress.preparation_step("untrusted\nlabel"):
            pass
    assert events == []


# --------------------------------------------------------------------------- task_runtime steps


_COMMAND_CASES = [
    (0, task_runtime._CheckedCommandStatus.SUCCEEDED, "completed"),
    (1, task_runtime._CheckedCommandStatus.COMMAND_FAILED, "failed"),
    (125, task_runtime._CheckedCommandStatus.INFRASTRUCTURE_FAILED, "failed"),
]


def test_command_progress_preserves_failure_classification_timeout_and_cid_cleanup(
    tmp_path: Path, steps: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    # command progress preserves failure classification
    for returncode, status, outcome in _COMMAND_CASES:
        steps.clear()
        completed = subprocess.CompletedProcess(["pip"], returncode, "", "")
        result = task_runtime._run_checked_with_progress(
            ["pip"], task_dir=tmp_path, deadline=time.monotonic() + 60,
            run=lambda *_args, **_kwargs: completed, environment={}, stage="dependency_install",
        )
        assert result.status is status, returncode
        assert result.process is completed, returncode
        assert steps == [("dependency_install", "started"), ("dependency_install", outcome)], returncode

    # command progress preserves timeout and cid cleanup
    steps.clear()
    cid_path = tmp_path / "runtime.cid"
    cid_path.write_text("a" * 64)
    removed: list[Path] = []

    def timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("pip", 1)

    def remove(path: Path, deadline: float) -> bool:
        removed.append(path)
        return True

    monkeypatch.setattr(task_runtime, "_remove_container_before_deadline", remove)
    with pytest.raises(task_runtime.TaskRuntimeDeadlineExceeded):
        task_runtime._run_checked_with_progress(
            ["pip"], task_dir=tmp_path, deadline=time.monotonic() + 60,
            run=timeout, environment={}, stage="dependency_install", cid_path=cid_path,
        )
    assert steps == [("dependency_install", "started"), ("dependency_install", "failed")]
    assert removed == [cid_path]
    assert not cid_path.exists()


def test_runtime_progress_tracks_cache_build_and_copy(
    tmp_path: Path, steps: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    make_program_dir(task_dir / "program_0", content="print('baseline')\n")
    (task_dir / "evaluate.py").write_text("")
    (task_dir / "requirements.txt").write_text("example==1.0\n")
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    image_id = "sha256:" + "a" * 64
    monkeypatch.setattr(
        task_runtime, "acquire_runtime_build_lease", lambda **_kwargs: SimpleNamespace(release=lambda: None)
    )

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["docker", "image", "inspect", "--format={{.Id}}"]:
            return subprocess.CompletedProcess(command, 0, image_id + "\n", "")
        if "venv" in command:
            mount = next(value for value in command if value.endswith(":/runtime:rw"))
            python = Path(mount.removesuffix(":/runtime:rw")) / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("")
        if "/task/evaluate.py" in command:
            mount = next(value for value in command if value.endswith(":/results:rw"))
            results = Path(mount.removesuffix(":/results:rw"))
            (results / "metrics.json").write_text('{"combined_score": 1.0}')
            (results / "correct.json").write_text('{"correct": true}')
        return subprocess.CompletedProcess(command, 0, "", "")

    def prepare() -> task_runtime.TaskRuntimePreparationResult:
        return task_runtime.prepare_task_runtime_result(
            task_dir, containerize=True, container_image=image_id, run=fake_run, install_cache_dir=cache,
        )

    assert prepare() is task_runtime.TaskRuntimePreparationResult.READY
    assert [step for step, outcome in steps if outcome == "started"] == [
        "runtime_cache", "runtime_capacity", "runtime_cache", "runtime_create", "tooling_build",
        "tooling_install", "dependency_install", "runtime_publish", "baseline_evaluation", "runtime_copy",
    ]
    assert all(outcome != "failed" for _, outcome in steps)
    steps.clear()
    assert prepare() is task_runtime.TaskRuntimePreparationResult.READY
    assert [step for step, outcome in steps if outcome == "started"] == [
        "runtime_cache", "baseline_evaluation", "runtime_copy",
    ]


# --------------------------------------------------------------------------- agsci investigate rendering


def test_runtime_preparation_steps_are_reported(tmp_path: Path, capsys) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    make_program_dir(task_dir / "program_0", content="def solve():\n    return 1\n")
    (task_dir / "capture.py").write_text("# capture stub\n", encoding="utf-8")
    (task_dir / "evaluate.py").write_text("# evaluate stub\n", encoding="utf-8")
    (task_dir / "task.md").write_text("# Goal\n- optimize\n", encoding="utf-8")
    overrides = [
        "agent.name=claude",
        f"paths.task_dir={task_dir}",
        f"paths.results_dir={tmp_path / 'results'}",
        "container.enabled=true",
        "container.image=agentic-scientist/agent-cli:test",
        f"container.auth_root={tmp_path}",
        "logging.heartbeat_seconds=0",
        "execution.max_idea_cycles=1",
        "execution.max_idea_generation_attempts=1",
    ]

    def prepare(*args: object, **kwargs: object) -> bool:
        with progress.preparation_step("baseline_evaluation"):
            pass
        return True

    with (
        patch("src.investigation.cli.shutil.which", return_value="/usr/bin/docker"),
        patch("src.investigation.cli._probe_docker_available", return_value=(True, "")),
        patch("src.investigation.cli._docker_image_exists", return_value=(True, "")),
        patch("src.investigation.cli.prepare_task_runtime", side_effect=prepare),
        patch(
            "src.investigation.cli.run_idea_generation",
            return_value=investigation_cli.IdeaGenerationResult(output="no idea", exit_code=1, idea_dir=None),
        ),
    ):
        investigation_cli.main(overrides=overrides)
    assert "Evaluating starting candidate · complete" in capsys.readouterr().out
