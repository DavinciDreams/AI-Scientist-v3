"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import os
import sys
import json
import argparse
import traceback
import importlib.util
import multiprocessing as mp
import numpy as np
from queue import Empty
from typing import Tuple, Optional, List, Dict, Any

from capture import OutputCapture

PROGRAM_TIMEOUT_SECONDS = 5 * 60


def load_function_from_file(program_path: str, fn_name: str):
    """Dynamically load a function from a Python file."""
    # Ensure sibling imports work when the candidate entrypoint lives under src/.
    program_dir = os.path.dirname(os.path.abspath(program_path))
    if program_dir and program_dir not in sys.path:
        sys.path.insert(0, program_dir)

    spec = importlib.util.spec_from_file_location("program_module", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load python module spec from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["program_module"] = module
    spec.loader.exec_module(module)
    return getattr(module, fn_name)


def _run_solution_worker(
    program_path: str, kwargs: Dict[str, Any], result_queue: mp.Queue
) -> None:
    """Runs user solution in a child process and sends back result/error payload."""
    try:
        run_packing = load_function_from_file(program_path, "solution")
        result_queue.put({"ok": True, "result": run_packing(**kwargs)})
    except Exception as e:
        result_queue.put(
            {
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def run_solution_with_timeout(
    program_path: str,
    kwargs: Dict[str, Any],
    timeout_seconds: int = PROGRAM_TIMEOUT_SECONDS,
):
    """Executes imported solution with a hard timeout."""
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_run_solution_worker, args=(program_path, kwargs, result_queue))
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(1)
        raise TimeoutError(
            f"Timed out after {timeout_seconds} seconds while running imported program."
        )

    if proc.exitcode not in (0, None):
        raise RuntimeError(f"Imported program exited with code {proc.exitcode}.")

    try:
        payload = result_queue.get_nowait()
    except Empty as e:
        raise RuntimeError("Imported program finished without returning any result.") from e

    if payload.get("ok"):
        return payload["result"]

    child_error = payload.get("error", "Unknown child process error")
    child_trace = payload.get("traceback", "")
    raise RuntimeError(f"{child_error}\n{child_trace}".strip())


def format_centers_string(centers: np.ndarray) -> str:
    """Formats circle centers into a multi-line string for display."""
    return "\n".join(
        [
            f"  centers[{i}] = ({x_coord:.4f}, {y_coord:.4f})"
            for i, (x_coord, y_coord) in enumerate(centers)
        ]
    )


def adapted_validate_packing(
    run_output: Tuple[np.ndarray, np.ndarray, float],
    atol=1e-6,
) -> Tuple[bool, Optional[str]]:
    """
    Validates circle packing results based on the output of 'run_packing'.

    Args:
        run_output: Tuple (centers, radii, reported_sum) from run_packing.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    centers, radii, reported_sum = run_output
    msg = "The circles are placed correctly. There are no overlaps or any circles outside the unit square."
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    n_expected = 26
    if centers.shape != (n_expected, 2):
        msg = (
            f"Centers shape incorrect. Expected ({n_expected}, 2), got {centers.shape}"
        )
        return False, msg
    if radii.shape != (n_expected,):
        msg = f"Radii shape incorrect. Expected ({n_expected},), got {radii.shape}"
        return False, msg

    if np.any(radii < 0):
        negative_indices = np.where(radii < 0)[0]
        msg = f"Negative radii found for circles at indices: {negative_indices}"
        return False, msg

    if not np.isclose(np.sum(radii), reported_sum, atol=atol):
        msg = (
            f"Sum of radii ({np.sum(radii):.6f}) does not match "
            f"reported ({reported_sum:.6f})"
        )
        return False, msg

    for i in range(n_expected):
        x, y = centers[i]
        r = radii[i]
        is_outside = (
            x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol
        )
        if is_outside:
            msg = (
                f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) is outside unit square."
            )
            return False, msg

    for i in range(n_expected):
        for j in range(i + 1, n_expected):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                msg = (
                    f"Circles {i} & {j} overlap. Dist: {dist:.4f}, "
                    f"Sum Radii: {(radii[i] + radii[j]):.4f}"
                )
                return False, msg
    return True, msg


def get_circle_packing_kwargs(run_index: int) -> Dict[str, Any]:
    """Provides keyword arguments for circle packing runs (none needed)."""
    return {}


def aggregate_circle_packing_metrics(
    results: List[Tuple[np.ndarray, np.ndarray, float]], results_dir: str
) -> Dict[str, Any]:
    """
    Aggregates metrics for circle packing. Assumes num_runs=1.
    Saves extra.npz with detailed packing information.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    centers, radii, reported_sum = results[0]

    public_metrics = {
        "centers_str": format_centers_string(centers),
        "num_circles": centers.shape[0],
    }
    private_metrics = {
        "reported_sum_of_radii": float(reported_sum),
    }
    metrics = {
        "combined_score": float(reported_sum),
        "public": public_metrics,
        "private": private_metrics,
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(
            extra_file,
            centers=centers,
            radii=radii,
            reported_sum=reported_sum,
        )
        print(f"Detailed packing data saved to {extra_file}")
    except Exception as e:
        print(f"Error saving extra.npz: {e}")
        metrics["extra_npz_save_error"] = str(e)

    return metrics


def main(program_path: str, results_dir: str, verbose: bool = True):
    """Runs the circle packing evaluation and saves results to JSON."""
    os.makedirs(results_dir, exist_ok=True)

    metrics = {}
    correct = False
    error_msg = None

    # Capture output during execution (streams to files in real-time)
    with OutputCapture(results_dir, tee=verbose) as capture:
        print(f"Evaluating program: {program_path}")
        print(f"Saving results to: {results_dir}")

        try:
            # Run imported program with hard timeout
            print(
                f"[1/4] Running imported program with timeout={PROGRAM_TIMEOUT_SECONDS}s..."
            )
            result = run_solution_with_timeout(
                program_path=program_path,
                kwargs=get_circle_packing_kwargs(0),
                timeout_seconds=PROGRAM_TIMEOUT_SECONDS,
            )
            centers, radii, reported_sum = result
            print(
                f"[1/4] Solution returned: {len(centers)} circles, sum_radii={reported_sum:.6f}"
            )

            # Validate
            print(f"[2/4] Validating packing constraints...")
            correct, error_msg = adapted_validate_packing(result)
            if correct:
                print(f"[2/4] Validation passed: {error_msg}")
            else:
                print(f"[2/4] Validation FAILED: {error_msg}")

            # Aggregate metrics
            print(f"[3/4] Aggregating metrics...")
            metrics = aggregate_circle_packing_metrics([result], results_dir)
            print(f"[3/4] Combined score: {metrics.get('combined_score', 0.0):.6f}")

        except Exception as e:
            error_msg = f"Execution error: {e}"
            print(f"[ERROR] {error_msg}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            metrics = {"combined_score": 0.0, "error": error_msg}

        print(f"Saved {capture.stdout_path}")
        print(f"Saved {capture.stderr_path}")

        # Save metrics.json
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved {metrics_path}")

        # Save correct.json
        correct_path = os.path.join(results_dir, "correct.json")
        with open(correct_path, "w") as f:
            json.dump({"correct": correct, "message": error_msg}, f, indent=2)
        print(f"Saved {correct_path}")

        if correct:
            print("Evaluation and Validation completed successfully.")
        else:
            print(f"Evaluation or Validation failed: {error_msg}")

        print("Metrics:")
        for key, value in metrics.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: <string_too_long_to_display>")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Circle packing evaluator (saves metrics.json + correct.json)"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'solution')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, stdout.log, stderr.log)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Tee captured output to console (default: True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (only save to logs)",
    )
    parsed_args = parser.parse_args()
    verbose = not parsed_args.quiet
    main(parsed_args.program_path, parsed_args.results_dir, verbose=verbose)
