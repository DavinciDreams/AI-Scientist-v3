"""Evaluator for the GRPO training task.

Runs the candidate ``program.py`` with this Python on the GPUs visible to the
process, reads the ``_results.json`` the program writes next to itself, and saves
``metrics.json`` + ``correct.json`` to ``--results_dir``.

Evaluations run offline: ``uv run examples/grpo/download_model.py`` fetches the
Qwen2.5-Math-1.5B weights into ``resources/`` once, and the program receives that
snapshot through ``GRPO_MODEL_PATH``.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any

from capture import OutputCapture

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
# Training takes 1-2 hours; the program is killed after this many seconds.
PROGRAM_TIMEOUT_SECONDS = 150 * 60
# The full program output goes to program_stdout.log in the results dir; only progress
# and error lines are echoed to the console, within this budget, because the harness
# caps the evaluator's console output.
PROGRAM_LOG_NAME = "program_stdout.log"
CONSOLE_ECHO_BUDGET_BYTES = 24 * 1024
CONSOLE_ECHO_PATTERN = re.compile(
    r"epoch|eval/|train/|accuracy|reward|Error|Traceback|Exception|CUDA|GPU", re.IGNORECASE
)


def task_dir() -> Path:
    """The task directory: ``TASK_SOURCE_DIR`` when the harness sets it, else this file's."""
    source = os.environ.get("TASK_SOURCE_DIR", "").strip()
    if source and (Path(source) / "grpo_utils" / "src").is_dir():
        return Path(source).resolve()
    return Path(__file__).resolve().parent


def model_dir() -> Path:
    return task_dir() / "resources" / MODEL_NAME.split("/")[-1]


def grpo_utils_src() -> Path:
    return task_dir() / "grpo_utils" / "src"


# ---------------------------------------------------------------------------
# Program layout
# ---------------------------------------------------------------------------
def resolve_program_layout(program_path: str) -> tuple[Path, str]:
    """Return ``(program_root, entrypoint_rel)``.

    *program_root* is the closest ancestor holding ``program_spec.toml`` (or the
    entrypoint's directory) and *entrypoint_rel* the entrypoint path inside it.
    """
    entrypoint = Path(program_path).expanduser().resolve()
    for parent in [entrypoint.parent, *entrypoint.parents]:
        if (parent / "program_spec.toml").is_file():
            return parent, entrypoint.relative_to(parent).as_posix()
    return entrypoint.parent, entrypoint.name


# ---------------------------------------------------------------------------
# Results reading and validation
# ---------------------------------------------------------------------------
def read_results_json(program_dir: Path) -> dict[str, Any]:
    """Read the ``_results.json`` the program writes (``final_accuracy``, ``logs``)."""
    results_path = program_dir / "_results.json"
    if not results_path.exists():
        return {}
    with open(results_path) as f:
        return json.load(f)


def validate_result(parsed: dict[str, Any]) -> tuple[bool, str]:
    """Validate the experiment result."""
    if "final_accuracy" not in parsed:
        return False, "final_accuracy not found in results."

    accuracy = parsed["final_accuracy"]

    if not isinstance(accuracy, (int, float)):
        return False, f"Accuracy must be a number, got {type(accuracy).__name__}"

    if accuracy != accuracy:  # NaN check
        return False, "Accuracy is NaN"

    if accuracy < 0 or accuracy > 1:
        return False, f"Accuracy {accuracy:.4f} is outside valid range [0, 1]"

    if "logs" not in parsed:
        return False, "logs not found in results."

    if not isinstance(parsed["logs"], list):
        return False, f"Logs must be a list, got {type(parsed['logs']).__name__}"

    return True, "Training completed successfully."


def aggregate_metrics(parsed: dict[str, Any]) -> dict[str, Any]:
    """Aggregate training results into metrics dict."""
    accuracy = parsed["final_accuracy"]
    logs = parsed["logs"]

    train_accuracies = [log["train/accuracy"] for log in logs if "train/accuracy" in log]
    eval_accuracies = [log["eval/accuracy"] for log in logs if "eval/accuracy" in log]
    train_losses = [log["train/loss"] for log in logs if "train/loss" in log]

    public_metrics = {
        "final_accuracy": float(accuracy),
        "num_epochs": len(logs),
    }
    if train_accuracies:
        public_metrics["final_train_accuracy"] = train_accuracies[-1]
    if eval_accuracies:
        public_metrics["final_eval_accuracy"] = eval_accuracies[-1]
    if train_losses:
        public_metrics["final_train_loss"] = train_losses[-1]

    return {
        "combined_score": float(accuracy),
        "public": public_metrics,
        "private": {"training_logs": logs},
    }


def save_json_results(
    results_dir: Path, metrics: dict[str, Any], correct: bool, error: str | None
) -> None:
    """Write metrics.json and correct.json to results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    correct_payload: dict[str, Any] = {"correct": bool(correct)}
    if error:
        correct_payload["error"] = str(error)
    with open(results_dir / "correct.json", "w", encoding="utf-8") as f:
        json.dump(correct_payload, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Program execution
# ---------------------------------------------------------------------------
def program_environment(scratch_dir: Path) -> dict[str, str]:
    """Environment for the program: offline model snapshot, grpo_utils, scratch caches."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        PYTHONPATH=str(grpo_utils_src()),
        PYTHONUNBUFFERED="1",
        GRPO_MODEL_PATH=str(model_dir()),
        HF_HUB_OFFLINE="1",
        HF_HOME=str(scratch_dir / "hf"),
        XDG_CACHE_HOME=str(scratch_dir / "cache"),
        TORCHINDUCTOR_CACHE_DIR=str(scratch_dir / "torch" / "inductor"),
        TRITON_CACHE_DIR=str(scratch_dir / "triton"),
        VLLM_USE_V1="0",
    )
    return env


_child: subprocess.Popen | None = None


def _kill_child() -> None:
    if _child is not None and _child.poll() is None:
        try:
            os.killpg(_child.pid, signal.SIGKILL)
        except OSError:
            pass


def _sigterm_handler(signum: int, frame: Any) -> None:
    _kill_child()
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _sigterm_handler)


def run_program(
    program_path: str,
    results_dir: str,
    *,
    timeout_seconds: int = PROGRAM_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], bool, str | None]:
    """Run the program from a scratch copy and evaluate its ``_results.json``.

    Returns (metrics, correct, message).
    """
    global _child
    results_path = Path(results_dir).resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    default_metrics = {"combined_score": 0.0, "error": "GRPO evaluation failed"}

    def fail(message: str) -> tuple[dict[str, Any], bool, str]:
        print(f"[grpo-eval] Error: {message}")
        print(f"[grpo-eval] Error: {message}", file=sys.stderr)
        save_json_results(results_path, default_metrics, False, message)
        return default_metrics, False, message

    if not (model_dir() / "config.json").is_file():
        return fail(
            f"Model snapshot missing at {model_dir()}; run "
            "`uv run examples/grpo/download_model.py` once before evaluating."
        )
    if not grpo_utils_src().is_dir():
        return fail(f"grpo_utils sources missing at {grpo_utils_src()}")

    program_root, entrypoint_rel = resolve_program_layout(program_path)
    scratch_dir = Path(tempfile.mkdtemp(prefix="grpo-eval-"))
    try:
        run_root = scratch_dir / "program"
        shutil.copytree(
            program_root,
            run_root,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "results"),
        )
        print(f"[grpo-eval] Program root: {program_root}")
        print(f"[grpo-eval] Entrypoint: {entrypoint_rel}")
        print(f"[grpo-eval] Scratch dir: {scratch_dir}")
        print(f"[grpo-eval] Model: {model_dir()}")

        timed_out = threading.Event()

        def on_timeout() -> None:
            timed_out.set()
            _kill_child()

        _child = subprocess.Popen(
            [sys.executable, "-u", entrypoint_rel],
            cwd=run_root,
            env=program_environment(scratch_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timer = threading.Timer(timeout_seconds, on_timeout)
        timer.start()
        program_log = results_path / PROGRAM_LOG_NAME
        echoed_bytes = 0
        tail: list[str] = []
        try:
            assert _child.stdout is not None
            with open(program_log, "w", encoding="utf-8") as log:
                for line in _child.stdout:
                    log.write(line)
                    tail = (tail + [line])[-40:]
                    if CONSOLE_ECHO_PATTERN.search(line) and len(line) < 400:
                        echoed_bytes += len(line)
                        if echoed_bytes <= CONSOLE_ECHO_BUDGET_BYTES:
                            print(line, end="", flush=True)
            returncode = _child.wait()
        except KeyboardInterrupt:
            _kill_child()
            raise
        finally:
            timer.cancel()
            _child = None

        print(f"[grpo-eval] Full program output: {program_log}")
        if timed_out.is_set():
            return fail(f"Program timed out after {timeout_seconds}s")
        if returncode != 0:
            print("[grpo-eval] Last program output lines:", file=sys.stderr)
            print("".join(tail)[-4000:], file=sys.stderr)
            return fail(f"Program exited with code {returncode}")

        parsed = read_results_json(run_root / Path(entrypoint_rel).parent)
        if parsed.get("final_accuracy") is None:
            return fail("Program did not write _results.json with final_accuracy")

        correct, message = validate_result(parsed)
        metrics = aggregate_metrics(parsed)
        print(f"[grpo-eval] Validation {'passed' if correct else 'FAILED'}: {message}")
        print(f"[grpo-eval] Combined score (accuracy): {metrics['combined_score']:.4f}")
        save_json_results(results_path, metrics, correct, None if correct else message)
        return metrics, correct, message
    except Exception as exc:  # noqa: BLE001 - the evaluator must always write results
        print(traceback.format_exc(), file=sys.stderr)
        return fail(f"GRPO evaluation failed: {exc}")
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(program_path: str, results_dir: str, verbose: bool = True) -> None:
    program_path = os.path.abspath(program_path)
    os.makedirs(results_dir, exist_ok=True)

    with OutputCapture(results_dir, tee=verbose):
        print(f"Evaluating program: {program_path}")
        print(f"Saving results to: {results_dir}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

        metrics, correct, message = run_program(program_path, results_dir)

        if correct:
            print("Evaluation completed successfully.")
        else:
            print(f"Evaluation failed: {message}")

        print("Metrics:")
        for key, value in metrics.items():
            if key == "private":
                print(f"  {key}: <omitted>")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO training evaluator")
    parser.add_argument(
        "--program_path",
        type=str,
        default="program_0/src/program.py",
        help="Path to training script to evaluate",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, stdout.log, stderr.log)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (only save to logs)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir, verbose=not parsed_args.quiet)
