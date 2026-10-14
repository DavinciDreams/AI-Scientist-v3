"""Isolated evaluator runner (src.investigation.evaluate_program) and the GRPO example evaluator."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path
from typing import Any, Callable

import pytest

from src.investigation import evaluate_program
from src.investigation.evaluate_program import run_evaluation
from src.investigation.files import find_best_program_in_experiment
from src.investigation.program_clone import clone_program_dir
from tests.program_testutils import make_program_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
GRPO_DIR = REPO_ROOT / "examples" / "grpo"


def _evaluator(*, score: str = "1.0", extra_args: str = "", before: str = "", after: str = "") -> str:
    """Source for a minimal evaluator that publishes metrics/correct summaries."""
    return (
        "from pathlib import Path\nimport argparse\nimport json\n"
        "p = argparse.ArgumentParser()\np.add_argument('--program_path')\np.add_argument('--results_dir')\n"
        + extra_args
        + "a = p.parse_args()\nout = Path(a.results_dir)\nprogram = Path(a.program_path)\n"
        + before
        + f"(out / 'metrics.json').write_text(json.dumps({{'combined_score': {score}}}))\n"
        "(out / 'correct.json').write_text(json.dumps({'correct': True}))\n"
        + after
    )


def _program_dir(root: Path) -> Path:
    """A candidate program_1 under root (root is the experiment directory)."""
    make_program_dir(root / "program_1", content="print('candidate')\n")
    return root / "program_1"


@pytest.fixture
def program_dir(tmp_path: Path) -> Path:
    return _program_dir(tmp_path)


def _write_evaluator(program_dir: Path, source: str) -> Path:
    evaluator = program_dir.parent / "evaluate.py"
    evaluator.write_text(source, encoding="utf-8")
    return evaluator


def _run(program_dir: Path, source: str | None = None, **kwargs: Any) -> Any:
    evaluator = program_dir.parent / "evaluate.py"
    if source is not None:
        _write_evaluator(program_dir, source)
    return run_evaluation(evaluator=evaluator, program_dir=program_dir, python_executable=sys.executable, **kwargs)


def _cli(program_dir: Path, *extra: str) -> list[str]:
    return [
        sys.executable, "-I", "-m", "src.investigation.evaluate_program",
        "--evaluator", str(program_dir.parent / "evaluate.py"),
        "--program-dir", str(program_dir),
        *extra,
    ]


def test_evaluation_outputs_and_runtime_stay_outside_program_and_canonical_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # outputs stay outside the successor clone; secrets and python injection do not reach the evaluator
    root = tmp_path / "clone"
    program_dir = _program_dir(root)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-evaluator")
    monkeypatch.setenv("PYTHONHOME", "/agent/python")
    monkeypatch.setenv("PYTHONPATH", "/agent/site-packages")
    monkeypatch.setenv("VIRTUAL_ENV", "/agent/venv")
    source = _evaluator(
        score="float(a.num_replications)",
        extra_args="p.add_argument('--num_replications', type=int, required=True)\n",
        before=textwrap.dedent(
            """
            import os
            (out / "runs" / "run_000").mkdir(parents=True)
            (out / "runs" / "run_000" / "transcript.jsonl").write_text("large")
            (out / "environment.json").write_text(json.dumps({
                "secret_present": "OPENAI_API_KEY" in os.environ,
                "home": os.environ["HOME"],
                "python_injection": sorted(
                    key for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV") if key in os.environ
                ),
            }))
            """
        ),
    )
    result = _run(program_dir, source, evaluator_args=("--num_replications", "5"))
    assert result.returncode == 0
    assert json.loads((program_dir / "metrics.json").read_text())["combined_score"] == 5.0
    assert json.loads((program_dir / "correct.json").read_text())["correct"] is True
    assert not (program_dir / "runs").exists()
    assert (result.output_dir / "runs" / "run_000" / "transcript.jsonl").is_file()
    environment = json.loads((result.output_dir / "environment.json").read_text())
    assert environment["secret_present"] is False
    assert Path(environment["home"]).parent == result.runtime_dir
    assert result.runtime_dir.parent.parent.name == ".evaluation_runtime"
    assert not (result.output_dir / "home").exists()
    assert not (result.output_dir / "tmp").exists()
    assert environment["python_injection"] == []
    assert json.loads((program_dir / "evaluation_manifest.json").read_text())["program_id"] == "program_1"
    successor = root / "program_2"
    clone_program_dir(src_dir=program_dir, dst_dir=successor)
    assert (successor / "src" / "program.py").is_file()
    for name in ("metrics.json", "correct.json", "runs"):
        assert not (successor / name).exists()
    assert find_best_program_in_experiment(root) is not None
    (program_dir / "evaluation_manifest.json").write_text("{}", encoding="utf-8")
    assert find_best_program_in_experiment(root) is None

    # the evaluator runtime stays outside the canonical results directory
    program_dir = _program_dir(tmp_path / "runtime")
    source = _evaluator(
        before=textwrap.dedent(
            """
            import os
            results_dir = out.resolve()
            unexpected = sorted(path.name for path in results_dir.iterdir())
            if unexpected:
                raise RuntimeError(f"contaminated results directory: {unexpected}")
            home = Path(os.environ["HOME"]).resolve()
            temp = Path(os.environ["TMPDIR"]).resolve()
            if results_dir in (home, *home.parents) or results_dir in (temp, *temp.parents):
                raise RuntimeError("runtime state is nested inside canonical results")
            """
        )
    )
    result = _run(program_dir, source)
    assert result.returncode == 0
    assert json.loads((program_dir / "correct.json").read_text())["correct"] is True
    assert sorted(path.name for path in result.output_dir.iterdir()) == ["correct.json", "metrics.json"]


def _attempt_path(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    return str(args[0]).startswith("attempt_")


def _attempt_child(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    return str(args[1]).startswith("attempt_")


def _attempt_stat(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    return str(args[0]).startswith("attempt_") and kwargs.get("dir_fd") is not None


_ALLOCATION_FAILURES: list[tuple[str, str | None, str, Callable[..., bool], int, type[Exception], str, int, int]] = [
    # name, owner, attribute, matches, nth, error, match, outputs_left, runtime_left
    ("runtime_allocation_failure", "os", "mkdir", _attempt_path, 2, PermissionError, "injected failure", 0, 0),
    ("runtime_open_failure", None, "_open_child_directory", _attempt_child, 2, PermissionError, "injected failure", 0, 0),
    ("output_identity_failure", "os", "stat", _attempt_stat, 1, RuntimeError, "Could not clean up partial evaluation allocation", 1, 0),
    ("runtime_identity_failure", "os", "stat", _attempt_stat, 2, RuntimeError, "Could not clean up partial evaluation allocation", 0, 1),
]


def test_allocation_failures_clean_up_only_verified_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, owner, attribute, matches, nth, error, match, outputs_left, runtime_left in _ALLOCATION_FAILURES:
        root = tmp_path / name
        program_dir = _program_dir(root)
        _write_evaluator(program_dir, "raise AssertionError('must not run')\n")
        target = evaluate_program.os if owner == "os" else evaluate_program
        original = getattr(target, attribute)
        calls = 0

        def failing(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            if matches(args, kwargs):
                calls += 1
                if calls == nth:
                    raise PermissionError("injected failure")
            return original(*args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(target, attribute, failing)
            with pytest.raises(error, match=match):
                _run(program_dir)
        assert len(list((root / ".evaluation_outputs/program_1").iterdir())) == outputs_left, name
        assert len(list((root / ".evaluation_runtime/program_1").iterdir())) == runtime_left, name


def test_retries_use_unique_outputs_recover_interrupted_publication_and_preserve_completed_results(tmp_path: Path) -> None:
    # a failed evaluation can retry in a unique output directory
    program_dir = _program_dir(tmp_path / "retry")
    first = _run(program_dir, "raise SystemExit(1)\n")
    assert first.returncode == 1
    assert not (program_dir / "metrics.json").exists()
    second = _run(program_dir, _evaluator())
    assert second.returncode == 0
    assert first.output_dir != second.output_dir

    # a retry recovers an interrupted result publication
    program_dir = _program_dir(tmp_path / "interrupted")
    (program_dir / "metrics.json").write_text("partial")
    partial_stat = (program_dir / "metrics.json").stat()
    (program_dir / ".evaluation_pending").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_id": "abandoned",
                "owned_files": {"metrics.json": {"device": partial_stat.st_dev, "inode": partial_stat.st_ino}},
            }
        )
    )
    (program_dir / ".evaluation_stage_abandoned_metrics.json").write_text("partial")
    result = _run(program_dir, _evaluator())
    assert result.returncode == 0
    assert not (program_dir / ".evaluation_pending").exists()
    assert not list(program_dir.glob(".evaluation_stage_*"))
    assert json.loads((program_dir / "metrics.json").read_text())["combined_score"] == 1.0

    # a rerun preserves an existing completed evaluation
    program_dir = _program_dir(tmp_path / "completed")
    (program_dir / "metrics.json").write_text('{"combined_score": 7.0}')
    (program_dir / "correct.json").write_text('{"correct": true}')
    (program_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "producer": "isolated_evaluator",
                "schema_version": 1,
                "output_dir": "../.evaluation_outputs/program_1/prior",
                "returncode": 0,
                "published_summaries": ["metrics.json", "correct.json"],
            }
        )
    )
    with pytest.raises(FileExistsError, match="already exists"):
        _run(program_dir, _evaluator())
    assert json.loads((program_dir / "metrics.json").read_text())["combined_score"] == 7.0
    assert not (program_dir / ".evaluation_pending").exists()


def test_evaluation_terminates_process_group_on_timeout_and_background_descendants_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a timeout terminates the process group gracefully
    program_dir = _program_dir(tmp_path / "timeout")
    source = (
        "import atexit, os, time\nfrom pathlib import Path\n"
        "atexit.register(lambda: Path(os.environ['HOME'], 'graceful_cleanup').write_text('done'))\n"
        "time.sleep(60)\n"
    )
    result = _run(program_dir, source, timeout_seconds=0.5)
    assert result.returncode == 124
    assert (result.runtime_dir / "home" / "graceful_cleanup").read_text() == "done"
    assert not (program_dir / "metrics.json").exists()

    # a successful evaluation terminates background descendants
    program_dir = _program_dir(tmp_path / "descendants")
    monkeypatch.setattr(evaluate_program, "_PROCESS_TERMINATION_GRACE_SECONDS", 0.1)
    child_code = (
        "import signal, sys, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "Path(sys.argv[1]).write_text('ready')\ntime.sleep(2)\nPath(sys.argv[2]).write_text('survived')\n"
    )
    source = _evaluator(
        before=(
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}, out / 'ready', out / 'survived'], start_new_session=True)\n"
            "deadline = time.monotonic() + 2\n"
            "while not (out / 'ready').exists() and time.monotonic() < deadline:\n    time.sleep(0.01)\n"
        )
    )
    result = _run(program_dir, source)
    time.sleep(1.2)
    assert result.returncode == 0
    assert not (result.output_dir / "survived").exists()


def test_result_publication_rejects_failed_mutated_or_invalid_evaluations(tmp_path: Path) -> None:
    # a nonzero evaluation does not publish valid-looking summaries
    program_dir = _program_dir(tmp_path / "nonzero")
    result = _run(program_dir, _evaluator(score="99.0", after="raise SystemExit(7)\n"))
    assert result.returncode == 7
    assert not (program_dir / "metrics.json").exists()
    assert not (program_dir / "correct.json").exists()

    # self-modifying candidate results are rejected
    program_dir = _program_dir(tmp_path / "mutated")
    source = _evaluator(score="99.0", before="program.write_text(program.read_text() + '# changed during evaluation\\n')\n")
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        _run(program_dir, source)
    assert not (program_dir / "metrics.json").exists()
    assert not (program_dir / "correct.json").exists()

    # candidate runtime outputs do not count as source mutation
    program_dir = _program_dir(tmp_path / "runtime-outputs")
    result = _run(program_dir, _evaluator(before="(program.parent / '_results.json').write_text(json.dumps({'runtime': True}))\n"))
    assert result.returncode == 0
    assert (program_dir / "src" / "_results.json").is_file()

    # invalid success summaries are rejected
    program_dir = _program_dir(tmp_path / "invalid")
    with pytest.raises(RuntimeError, match="combined_score must be numeric"):
        _run(program_dir, _evaluator(score="'invalid'"))
    assert not (program_dir / "metrics.json").exists()


def test_cli_forwards_external_termination_to_evaluator(program_dir: Path) -> None:
    _write_evaluator(
        program_dir,
        "import argparse, os, time\nfrom pathlib import Path\n"
        "p = argparse.ArgumentParser()\np.add_argument('--program_path')\np.add_argument('--results_dir')\n"
        "a = p.parse_args()\nPath(a.results_dir, 'pid.txt').write_text(str(os.getpid()))\ntime.sleep(60)\n",
    )
    process = subprocess.Popen(_cli(program_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pid_path = None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        matches = list((program_dir.parent / ".evaluation_outputs").glob("program_1/*/pid.txt"))
        if matches:
            pid_path = matches[0]
            break
        time.sleep(0.02)
    assert pid_path is not None
    evaluator_pid = int(pid_path.read_text())

    os.kill(process.pid, signal.SIGTERM)
    time.sleep(0.01)
    try:
        os.kill(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    assert process.wait(timeout=10) == 143
    try:
        os.kill(evaluator_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("evaluator survived wrapper termination")


def test_cli_forwards_replication_count_and_reports_allocated_paths(tmp_path: Path) -> None:
    # the replication count is forwarded and allocated paths are printed
    program_dir = _program_dir(tmp_path / "replications")
    _write_evaluator(
        program_dir,
        _evaluator(score="a.num_replications", extra_args="p.add_argument('--num_replications', type=int, required=True)\n"),
    )
    completed = subprocess.run(_cli(program_dir, "--num-replications", "5"), capture_output=True, text=True, timeout=10, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "evaluation outputs:" in completed.stdout
    assert "evaluation runtime:" in completed.stdout
    assert json.loads((program_dir / "metrics.json").read_text())["combined_score"] == 5

    # allocated paths are reported when summary validation fails
    program_dir = _program_dir(tmp_path / "invalid-summary")
    _write_evaluator(program_dir, "pass\n")
    completed = subprocess.run(_cli(program_dir), capture_output=True, text=True, timeout=10, check=False)
    assert completed.returncode == 1
    assert "evaluation outputs:" in completed.stdout
    assert "evaluation runtime:" in completed.stdout
    assert "Evaluator did not produce a safe summary" in completed.stderr


# --- GRPO example evaluator (examples/grpo/evaluate.py, direct in-container execution) ----------


@pytest.fixture
def grpo_evaluate():
    previous_capture = sys.modules.pop("capture", None)
    sys.path.insert(0, str(GRPO_DIR))
    try:
        spec = importlib.util.spec_from_file_location("grpo_evaluate_under_test", GRPO_DIR / "evaluate.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.pop(0)
        sys.modules.pop("capture", None)
        if previous_capture is not None:
            sys.modules["capture"] = previous_capture


@pytest.fixture
def task_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake task directory with a model snapshot and grpo_utils sources."""
    task = tmp_path / "task"
    (task / "resources" / "Qwen2.5-Math-1.5B").mkdir(parents=True)
    (task / "resources" / "Qwen2.5-Math-1.5B" / "config.json").write_text("{}")
    (task / "grpo_utils" / "src" / "grpo_utils").mkdir(parents=True)
    (task / "grpo_utils" / "src" / "grpo_utils" / "__init__.py").write_text("MARK = 'ok'\n")
    monkeypatch.setenv("TASK_SOURCE_DIR", str(task))
    return task


def _grpo_program(root: Path, body: str) -> Path:
    return make_program_dir(root / "program_1", content=body)


RESULT_WRITING_PROGRAM = """
import json, os
import grpo_utils
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.path.isdir(os.environ["GRPO_MODEL_PATH"])
print("training", flush=True)
print("epoch 1 eval/accuracy 0.5", flush=True)
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "_results.json"), "w") as f:
    json.dump({"final_accuracy": 0.5, "logs": [{"train/loss": 1.0, "eval/accuracy": 0.5}]}, f)
"""


def test_grpo_successful_run_scores_accuracy_and_missing_snapshot_fails_before_running(grpo_evaluate, task_tree, tmp_path, capsys) -> None:
    # a successful run scores accuracy and keeps the program directory pristine
    program = _grpo_program(tmp_path / "ok", RESULT_WRITING_PROGRAM)
    results = tmp_path / "ok" / "results"
    metrics, correct, message = grpo_evaluate.run_program(str(program), str(results))
    assert correct is True
    assert message == "Training completed successfully."
    assert metrics["combined_score"] == 0.5
    assert metrics["public"] == {"final_accuracy": 0.5, "num_epochs": 1, "final_eval_accuracy": 0.5, "final_train_loss": 1.0}
    assert json.loads((results / "correct.json").read_text()) == {"correct": True}
    assert json.loads((results / "metrics.json").read_text())["combined_score"] == 0.5
    assert not (program.parent / "_results.json").exists()
    console = capsys.readouterr().out
    assert "epoch 1 eval/accuracy 0.5" in console
    assert "training\n" not in console
    assert (results / "program_stdout.log").read_text() == "training\nepoch 1 eval/accuracy 0.5\n"

    # a missing model snapshot fails before running the program
    (task_tree / "resources" / "Qwen2.5-Math-1.5B" / "config.json").unlink()
    program = _grpo_program(tmp_path / "missing", "raise SystemExit('must not run')\n")
    results = tmp_path / "missing" / "results"
    metrics, correct, message = grpo_evaluate.run_program(str(program), str(results))
    assert correct is False
    assert metrics["combined_score"] == 0.0
    assert "download_model.py" in message
    assert json.loads((results / "correct.json").read_text())["error"] == message


_GRPO_FAILURES = [
    # name, body, expected_message, stderr_contains, kwargs
    ("exit_code_with_output_tail", "print('boom')\nraise SystemExit(3)\n", "Program exited with code 3", "boom", {}),
    ("missing_results", "print('no results')\n", "_results.json", None, {}),
    ("timeout", "import time\ntime.sleep(60)\n", "Program timed out after 1s", None, {"timeout_seconds": 1}),
]


def test_grpo_program_failures_are_recorded_and_console_echo_stays_within_budget(grpo_evaluate, task_tree, tmp_path, capsys) -> None:
    # program failures are recorded with their cause
    for name, body, expected_message, stderr_contains, kwargs in _GRPO_FAILURES:
        root = tmp_path / name
        program = _grpo_program(root, body)
        metrics, correct, message = grpo_evaluate.run_program(str(program), str(root / "results"), **kwargs)
        assert correct is False, name
        assert expected_message in message, name
        assert metrics["combined_score"] == 0.0, name
        captured = capsys.readouterr()
        if stderr_contains is not None:
            assert stderr_contains in captured.err, name

    # console echo stays within budget while the full log is kept
    root = tmp_path / "budget"
    body = "for i in range(5000):\n    print(f'epoch {i} ' + 'x' * 100)\nraise SystemExit(0)\n"
    grpo_evaluate.run_program(str(_grpo_program(root, body)), str(root / "results"))
    assert len(capsys.readouterr().out) < grpo_evaluate.CONSOLE_ECHO_BUDGET_BYTES + 2048
    assert (root / "results" / "program_stdout.log").stat().st_size > 500_000


_MALFORMED_GRPO_RESULTS = [
    ({"logs": []}, "final_accuracy not found"),
    ({"final_accuracy": "0.5", "logs": []}, "must be a number"),
    ({"final_accuracy": float("nan"), "logs": []}, "NaN"),
    ({"final_accuracy": 1.5, "logs": []}, "outside valid range"),
    ({"final_accuracy": 0.5}, "logs not found"),
    ({"final_accuracy": 0.5, "logs": {}}, "Logs must be a list"),
]


def test_grpo_helpers_resolve_environment_task_dir_layout_and_validate_results(grpo_evaluate, task_tree, tmp_path, monkeypatch) -> None:
    # the program environment targets task resources
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    env = grpo_evaluate.program_environment(tmp_path / "scratch")
    assert env["PYTHONPATH"] == str(task_tree / "grpo_utils" / "src")
    assert env["GRPO_MODEL_PATH"] == str(task_tree / "resources" / "Qwen2.5-Math-1.5B")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_HOME"] == str(tmp_path / "scratch" / "hf")

    # the task dir falls back to the evaluator directory
    with monkeypatch.context() as context:
        context.setenv("TASK_SOURCE_DIR", str(tmp_path / "not-a-task"))
        assert grpo_evaluate.task_dir() == GRPO_DIR

    # malformed results are rejected
    for parsed, reason in _MALFORMED_GRPO_RESULTS:
        correct, message = grpo_evaluate.validate_result(parsed)
        assert correct is False, reason
        assert reason in message, reason

    # a program layout without a spec uses the entrypoint directory
    program = tmp_path / "loose" / "program.py"
    program.parent.mkdir()
    program.write_text("")
    assert grpo_evaluate.resolve_program_layout(str(program)) == (program.parent, "program.py")


def test_grpo_validation_split_cli_reads_archived_train_file(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_data_module = types.ModuleType("torch.utils.data")
    torch_data_module.DataLoader = object  # type: ignore[attr-defined]
    torch_data_module.Dataset = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "torch.utils", types.ModuleType("torch.utils"))
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_data_module)

    data_path = GRPO_DIR / "grpo_utils" / "src" / "grpo_utils" / "data.py"
    spec = importlib.util.spec_from_file_location("grpo_data_under_test", data_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.TRAIN_PATH.name == "_train.jsonl"
    assert module.TRAIN_PATH.exists()
