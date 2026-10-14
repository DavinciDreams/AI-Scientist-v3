"""Stage 1: Prepare an analysis-ready bundle with checkpoint guardrails."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.investigation.agent_prompt import stage_agent_prompt

from .common import (
    compare_metric_guardrail,
    copy_selected_runtime_artifacts,
    find_checkpoint_candidates,
    infer_primary_metric,
    load_extraction_manifest,
    read_json,
    resolve_program_entrypoint,
    validate_checkpoint_loadable,
    write_json,
)

logger = logging.getLogger(__name__)

PopenFactory = Callable[..., subprocess.Popen[str]]
RegisterFn = Callable[[subprocess.Popen[str]], None]
UnregisterFn = Callable[[subprocess.Popen[str]], None]
TerminateFn = Callable[..., None]
EnsureTextFn = Callable[[Optional[str | bytes]], str]
ContainerEnvFn = Callable[..., dict[str, str]]

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "posthoc_bootstrap.md"


class BootstrapError(RuntimeError):
    """Raised when bootstrap cannot produce a valid checkpoint-ready bundle."""


def bootstrap_analysis_bundle(
    *,
    bundle_dir: Path,
    output_bundle_dir: Path,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: int,
    timeout_grace_seconds: int,
    max_attempts: int,
    primary_metric_key: str | None,
    primary_metric_tolerance: float,
    retrain_if_checkpoint_missing: bool,
    task_source_dir: Path | None = None,
    containerize: bool = False,
    container_image: str = "",
    container_network: str = "bridge",
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    container_session_name: str | None = None,
    container_mount_root: Path | None = None,
    script_path: Path | None = None,
    popen_factory: PopenFactory | None = None,
    register_subprocess: RegisterFn | None = None,
    unregister_subprocess: UnregisterFn | None = None,
    terminate_process_group: TerminateFn | None = None,
    container_execution_env: ContainerEnvFn | None = None,
    ensure_text: EnsureTextFn | None = None,
    checkpoint_mode: str = "required",
) -> dict[str, Any]:
    """Create an analysis-ready bundle with checkpoint metadata and guardrails."""
    bundle_dir = bundle_dir.resolve()
    output_bundle_dir = output_bundle_dir.resolve()

    manifest = load_extraction_manifest(bundle_dir)
    source_program_dir = Path(manifest["source_best_program_dir"]).expanduser().resolve()

    baseline_metrics = _load_baseline_metrics(source_program_dir, manifest)
    baseline_results = read_json(source_program_dir / "_results.json") if source_program_dir.is_dir() else None
    baseline_primary = infer_primary_metric(
        metrics=baseline_metrics,
        results=baseline_results,
        requested_key=primary_metric_key,
    )

    attempt_reports: list[dict[str, Any]] = []
    final_promoted: dict[str, Any] | None = None
    final_checkpoint_report: dict[str, Any] | None = None
    final_current_primary: dict[str, Any] | None = None
    final_guardrail: dict[str, Any] | None = None
    final_bootstrap_agent_report: dict[str, Any] | None = None
    final_rerun_report: dict[str, Any] | None = None
    final_status = "degraded"
    final_reason: str | None = None

    promoted = _reset_bootstrap_bundle(bundle_dir, output_bundle_dir, source_program_dir)
    final_promoted = promoted
    target_program_dir = output_bundle_dir / "program_0"
    checkpoint_candidates = find_checkpoint_candidates(target_program_dir)
    checkpoint_report = _checkpoint_report(target_program_dir, checkpoint_candidates)
    initial_current_primary: dict[str, Any] | None = None
    initial_guardrail: dict[str, Any] | None = None
    initial_error_text: str | None = None

    checkpoint_is_usable = (
        checkpoint_report["exists"] and checkpoint_report["load_error"] is None
    )
    if checkpoint_is_usable or checkpoint_mode == "optional":
        initial_current_primary, initial_guardrail, initial_error_text = _validate_bootstrap_bundle(
            target_program_dir=target_program_dir,
            baseline_primary=baseline_primary,
            primary_metric_key=primary_metric_key,
            primary_metric_tolerance=primary_metric_tolerance,
        )
        if initial_error_text is None:
            final_status = "completed"
            final_checkpoint_report = checkpoint_report
            final_current_primary = initial_current_primary
            final_guardrail = initial_guardrail
        else:
            final_reason = initial_error_text
    else:
        final_reason = checkpoint_report["load_error"]

    if (
        final_status != "completed"
        and checkpoint_mode == "required"
        and retrain_if_checkpoint_missing
    ):
        for attempt_index in range(1, max_attempts + 1):
            promoted = _reset_bootstrap_bundle(bundle_dir, output_bundle_dir, source_program_dir)
            target_program_dir = output_bundle_dir / "program_0"
            bootstrap_agent_report: dict[str, Any] | None = None
            rerun_report: dict[str, Any] | None = None
            checkpoint_report = {"exists": False, "paths": [], "selected_path": None, "load_error": None}
            attempt_current_primary: dict[str, Any] | None = None
            attempt_guardrail: dict[str, Any] | None = None
            attempt_error_text: str | None = None

            try:
                bootstrap_agent_report = _run_bootstrap_agent(
                    bundle_dir=output_bundle_dir,
                    agent=agent,
                    agent_model=agent_model,
                    agent_profile=agent_profile,
                    reasoning_effort=reasoning_effort,
                    timeout_seconds=timeout_seconds,
                    timeout_grace_seconds=timeout_grace_seconds,
                    task_source_dir=task_source_dir,
                    containerize=containerize,
                    container_image=container_image,
                    container_network=container_network,
                    container_auth_root=container_auth_root,
                    container_env_file=container_env_file,
                    container_session_name=container_session_name,
                    container_mount_root=container_mount_root,
                    script_path=script_path,
                    popen_factory=popen_factory,
                    register_subprocess=register_subprocess,
                    unregister_subprocess=unregister_subprocess,
                    terminate_process_group=terminate_process_group,
                    container_execution_env=container_execution_env,
                    ensure_text=ensure_text,
                )
                bootstrap_agent_report = _persist_bootstrap_process_report(
                    report_root=output_bundle_dir.parent,
                    attempt_index=attempt_index,
                    report=bootstrap_agent_report,
                    prefix="bootstrap_agent",
                )
                final_bootstrap_agent_report = bootstrap_agent_report
                if bootstrap_agent_report["exit_code"] != 0:
                    attempt_error_text = (
                        f"Bootstrap agent exited with code {bootstrap_agent_report['exit_code']}"
                    )
                    raise BootstrapError(attempt_error_text)
                rerun_report = _rerun_official_evaluator(
                    bundle_dir=output_bundle_dir,
                    task_source_dir=task_source_dir,
                )
                rerun_report = _persist_bootstrap_process_report(
                    report_root=output_bundle_dir.parent,
                    attempt_index=attempt_index,
                    report=rerun_report,
                    prefix="bootstrap_evaluator",
                )
                checkpoint_candidates = find_checkpoint_candidates(target_program_dir)
                checkpoint_report = _checkpoint_report(target_program_dir, checkpoint_candidates)
                (
                    attempt_current_primary,
                    attempt_guardrail,
                    attempt_error_text,
                ) = _validate_bootstrap_bundle(
                    target_program_dir=target_program_dir,
                    baseline_primary=baseline_primary,
                    primary_metric_key=primary_metric_key,
                    primary_metric_tolerance=primary_metric_tolerance,
                )
                if checkpoint_report["load_error"] is not None and attempt_error_text is None:
                    attempt_error_text = checkpoint_report["load_error"]
                if not checkpoint_report["exists"] and attempt_error_text is None:
                    attempt_error_text = "bootstrap completed but no checkpoint was found"
            except BootstrapError as exc:
                attempt_error_text = str(exc)

            attempt_report = {
                "attempt_index": attempt_index,
                "promoted_runtime_artifacts": promoted,
                "bootstrap_agent": bootstrap_agent_report,
                "rerun": rerun_report,
                "checkpoint": checkpoint_report,
                "current_primary_metric": attempt_current_primary,
                "guardrail": attempt_guardrail,
                "status": "completed" if attempt_error_text is None else "failed",
                "error": attempt_error_text,
            }
            attempt_reports.append(attempt_report)

            if attempt_error_text is None:
                final_status = "completed"
                final_reason = None
                final_promoted = promoted
                final_checkpoint_report = checkpoint_report
                final_current_primary = attempt_current_primary
                final_guardrail = attempt_guardrail
                final_bootstrap_agent_report = bootstrap_agent_report
                final_rerun_report = rerun_report
                break

            final_reason = attempt_error_text

    if final_status != "completed":
        final_promoted = _reset_bootstrap_bundle(bundle_dir, output_bundle_dir, source_program_dir)
        target_program_dir = output_bundle_dir / "program_0"
        checkpoint_candidates = find_checkpoint_candidates(target_program_dir)
        final_checkpoint_report = _checkpoint_report(target_program_dir, checkpoint_candidates)
        final_current_primary, final_guardrail, _ = _validate_bootstrap_bundle(
            target_program_dir=target_program_dir,
            baseline_primary=baseline_primary,
            primary_metric_key=primary_metric_key,
            primary_metric_tolerance=primary_metric_tolerance,
        )

    report: dict[str, Any] = {
        "stage": "bootstrap",
        "status": final_status,
        "bootstrapped_at": datetime.now(timezone.utc).isoformat(),
        "source_bundle_dir": str(bundle_dir),
        "analysis_ready_bundle_dir": str(output_bundle_dir),
        "source_program_dir": str(source_program_dir),
        "promoted_runtime_artifacts": final_promoted,
        "baseline_primary_metric": baseline_primary,
        "current_primary_metric": final_current_primary,
        "guardrail": final_guardrail,
        "checkpoint": final_checkpoint_report,
        "bootstrap_agent": final_bootstrap_agent_report,
        "rerun": final_rerun_report,
        "retry_attempts": attempt_reports,
        "bootstrap_ready": final_status == "completed",
        "failure_reason": final_reason,
        "retrain_if_checkpoint_missing": retrain_if_checkpoint_missing,
        "bootstrap_max_attempts": max_attempts,
        "checkpoint_mode": checkpoint_mode,
    }
    write_json(output_bundle_dir.parent / "bootstrap_report.json", report)
    logger.info("Bootstrap report written to %s", output_bundle_dir.parent / "bootstrap_report.json")
    return report


def _reset_bootstrap_bundle(
    bundle_dir: Path,
    output_bundle_dir: Path,
    source_program_dir: Path,
) -> dict[str, Any]:
    if output_bundle_dir.exists():
        shutil.rmtree(output_bundle_dir)
    shutil.copytree(bundle_dir, output_bundle_dir, symlinks=True)
    evidence_dir = output_bundle_dir / "evidence"
    if evidence_dir.is_symlink():
        evidence_dir.unlink()
    elif evidence_dir.is_dir():
        shutil.rmtree(evidence_dir)
    evidence_manifest = output_bundle_dir / "evidence_manifest.json"
    if evidence_manifest.exists() or evidence_manifest.is_symlink():
        evidence_manifest.unlink()

    target_program_dir = output_bundle_dir / "program_0"
    promoted: dict[str, Any] = {"files": [], "directories": [], "checkpoints": []}
    if source_program_dir.is_dir():
        promoted = copy_selected_runtime_artifacts(source_program_dir, target_program_dir)
    return promoted


def _validate_bootstrap_bundle(
    *,
    target_program_dir: Path,
    baseline_primary: dict[str, Any],
    primary_metric_key: str | None,
    primary_metric_tolerance: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    current_correct = read_json(target_program_dir / "correct.json")
    if not current_correct or current_correct.get("correct") is not True:
        return None, None, "analysis-ready bundle failed correctness validation"

    current_metrics = read_json(target_program_dir / "metrics.json")
    current_results = read_json(target_program_dir / "_results.json")
    if current_metrics is None:
        return None, None, "analysis-ready bundle is missing metrics.json after bootstrap"

    current_primary = infer_primary_metric(
        metrics=current_metrics,
        results=current_results,
        requested_key=primary_metric_key,
    )
    guardrail = compare_metric_guardrail(
        baseline=baseline_primary,
        candidate=current_primary,
        tolerance=primary_metric_tolerance,
    )
    if not guardrail["passed"]:
        reason = guardrail["reason"] or "primary metric regressed beyond tolerance"
        return current_primary, guardrail, f"bootstrap guardrail failed: {reason}"
    return current_primary, guardrail, None


def _load_baseline_metrics(
    source_program_dir: Path,
    extraction_manifest: dict[str, Any],
) -> dict[str, Any]:
    metrics = read_json(source_program_dir / "metrics.json") if source_program_dir.is_dir() else None
    if metrics:
        return metrics

    combined_score = extraction_manifest.get("combined_score")
    if isinstance(combined_score, (int, float)):
        return {"combined_score": float(combined_score)}
    raise BootstrapError("Unable to determine baseline metrics for bootstrap guardrail")


def _checkpoint_report(program_dir: Path, candidates: list[Path]) -> dict[str, Any]:
    if not candidates:
        return {
            "exists": False,
            "paths": [],
            "selected_path": None,
            "load_error": None,
        }

    selected = max(candidates, key=lambda path: path.stat().st_size)
    loadable, load_error = validate_checkpoint_loadable(selected)
    return {
        "exists": True,
        "paths": [path.relative_to(program_dir).as_posix() for path in candidates],
        "selected_path": selected.relative_to(program_dir).as_posix(),
        "load_error": None if loadable else load_error,
    }


def _run_bootstrap_agent(
    *,
    bundle_dir: Path,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: int,
    timeout_grace_seconds: int,
    task_source_dir: Path | None,
    containerize: bool,
    container_image: str,
    container_network: str,
    container_auth_root: Path | None,
    container_env_file: Path | None,
    container_session_name: str | None,
    container_mount_root: Path | None,
    script_path: Path | None,
    popen_factory: PopenFactory | None,
    register_subprocess: RegisterFn | None,
    unregister_subprocess: UnregisterFn | None,
    terminate_process_group: TerminateFn | None,
    container_execution_env: ContainerEnvFn | None,
    ensure_text: EnsureTextFn | None,
) -> dict[str, Any]:
    prompt = _build_bootstrap_prompt(bundle_dir)
    prompt_filename = stage_agent_prompt(bundle_dir, "bootstrap_prompt.md", prompt)

    if popen_factory is None:
        popen_factory = subprocess.Popen
    if register_subprocess is None:
        register_subprocess = lambda process: None  # noqa: E731
    if unregister_subprocess is None:
        unregister_subprocess = lambda process: None  # noqa: E731
    if terminate_process_group is None:
        from src.investigation.runtime import terminate_active_process_group

        terminate_process_group = terminate_active_process_group
    if ensure_text is None:
        ensure_text = (
            lambda text: text
            if isinstance(text, str)
            else (text.decode("utf-8", errors="replace") if text else "")
        )
    if container_execution_env is None:
        from src.investigation.runtime import container_execution_env as _cee

        container_execution_env = _cee
    if script_path is None:
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "posthoc_agent.sh"

    cmd = [
        "bash",
        str(script_path),
        "--agent",
        agent,
        "--work-dir",
        str(bundle_dir),
        "--prompt-file",
        prompt_filename,
    ]
    if task_source_dir is not None:
        cmd.extend(["--task-source-dir", str(task_source_dir)])
    if agent_model:
        cmd.extend(["--agent-model", agent_model])
    if agent_profile:
        cmd.extend(["--agent-profile", agent_profile])
    if reasoning_effort:
        cmd.extend(["--reasoning-effort", reasoning_effort])
    if containerize:
        cmd.append("--containerize")
        cmd.extend(["--container-image", container_image])
        cmd.extend(["--container-network", container_network])
        auth_root = container_auth_root if container_auth_root is not None else Path.home()
        cmd.extend(["--container-auth-root", str(auth_root)])
        if container_env_file is not None:
            cmd.extend(["--container-env-file", str(container_env_file)])

    env = container_execution_env(
        containerize=containerize,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
    )

    report = _execute_bootstrap_command(
        cmd=cmd,
        timeout_seconds=timeout_seconds,
        timeout_grace_seconds=timeout_grace_seconds,
        popen_factory=popen_factory,
        register_subprocess=register_subprocess,
        unregister_subprocess=unregister_subprocess,
        terminate_process_group=terminate_process_group,
        ensure_text=ensure_text,
        env=env,
        script_path=script_path,
    )
    report["prompt_path"] = str(bundle_dir / prompt_filename)
    return report


def _build_bootstrap_prompt(bundle_dir: Path) -> str:
    if _PROMPT_TEMPLATE_PATH.exists():
        template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        template = _DEFAULT_BOOTSTRAP_PROMPT

    program_source = ""
    entrypoint = resolve_program_entrypoint(bundle_dir / "program_0")
    if entrypoint.exists():
        program_source = entrypoint.read_text(encoding="utf-8")

    evaluator_source = ""
    evaluate_path = bundle_dir / "evaluate.py"
    if evaluate_path.exists():
        evaluator_source = evaluate_path.read_text(encoding="utf-8")

    task_md = ""
    task_path = bundle_dir / "task.md"
    if task_path.exists():
        task_md = task_path.read_text(encoding="utf-8")

    results_text = ""
    results_path = bundle_dir / "program_0" / "_results.json"
    if results_path.exists():
        results_text = results_path.read_text(encoding="utf-8")

    replacements = {
        "{{TASK_MD}}": task_md,
        "{{PROGRAM_SOURCE}}": program_source,
        "{{EVALUATOR_SOURCE}}": evaluator_source,
        "{{RESULTS_JSON}}": results_text,
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def _rerun_official_evaluator(
    *,
    bundle_dir: Path,
    task_source_dir: Path | None,
) -> dict[str, Any]:
    program_dir = bundle_dir / "program_0"
    entrypoint = resolve_program_entrypoint(program_dir)
    evaluate_path = bundle_dir / "evaluate.py"
    if not evaluate_path.is_file():
        raise BootstrapError(f"Missing evaluate.py in {bundle_dir}")

    env = os.environ.copy()
    if task_source_dir is not None:
        env["TASK_SOURCE_DIR"] = str(task_source_dir)

    cmd = [
        os.environ.get("WRAPPER_PYTHON", sys.executable),
        str(evaluate_path),
        "--program_path",
        str(entrypoint),
        "--results_dir",
        str(program_dir),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(bundle_dir),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise BootstrapError(
            f"Bootstrap evaluator rerun failed with exit code {result.returncode}"
        )
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _execute_bootstrap_command(
    *,
    cmd: list[str],
    timeout_seconds: int,
    timeout_grace_seconds: int,
    popen_factory: PopenFactory,
    register_subprocess: RegisterFn,
    unregister_subprocess: UnregisterFn,
    terminate_process_group: TerminateFn,
    ensure_text: EnsureTextFn,
    env: dict[str, str],
    script_path: Path,
) -> dict[str, Any]:
    started_at = time.monotonic()
    proc: subprocess.Popen[str] | None = None
    stdout_text = ""
    stderr_text = ""
    exit_code = 1
    timed_out = False
    try:
        proc = popen_factory(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        register_subprocess(proc)
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        stdout_text = ensure_text(stdout)
        stderr_text = ensure_text(stderr)
        exit_code = proc.returncode if proc.returncode is not None else 1
    except FileNotFoundError:
        exit_code = 127
        stderr_text = "[posthoc] bootstrap agent script not found\n"
    except KeyboardInterrupt:
        if proc is not None:
            terminate_process_group(proc, grace_seconds=timeout_grace_seconds)
        raise
    except subprocess.TimeoutExpired:
        timed_out = True
        elapsed_seconds = time.monotonic() - started_at
        if proc is not None:
            terminate_process_group(proc, grace_seconds=timeout_grace_seconds)
            stdout, stderr = proc.communicate()
            stdout_text = ensure_text(stdout)
            stderr_text = ensure_text(stderr)
        stderr_text += (
            f"\n[posthoc] bootstrap agent timed out after {timeout_seconds} seconds; "
            f"grace={timeout_grace_seconds} seconds; elapsed={elapsed_seconds:.1f}s.\n"
        )
        exit_code = 124
    finally:
        if proc is not None:
            unregister_subprocess(proc)

    return {
        "cmd": cmd,
        "script_path": str(script_path),
        "exit_code": exit_code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "timed_out": timed_out,
    }


def _persist_bootstrap_process_report(
    *,
    report_root: Path,
    attempt_index: int,
    report: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    attempt_dir = report_root / "bootstrap_attempts" / f"attempt_{attempt_index:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = attempt_dir / f"{prefix}_stdout.log"
    stderr_path = attempt_dir / f"{prefix}_stderr.log"
    stdout_text = str(report.get("stdout", ""))
    stderr_text = str(report.get("stderr", ""))
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")

    persisted = dict(report)
    persisted["stdout_path"] = str(stdout_path)
    persisted["stderr_path"] = str(stderr_path)
    return persisted


_DEFAULT_BOOTSTRAP_PROMPT = """You are preparing the extracted best program for post-hoc analysis.

Modify `program_0` only as needed so that:
1. Training/evaluation preserves the current task behavior and correctness.
2. The program saves a reusable best checkpoint into the program directory.
3. `_results.json` records the checkpoint path and any checkpoint metadata needed for later analysis.
4. Optional training history needed for analysis is also saved.

Rules:
- Do not change the task objective or scoring logic.
- Preserve the current model behavior as closely as possible.
- Prefer minimal, localized edits to the training program.
- The official evaluator will be rerun once after your edits.

Task:
```md
{{TASK_MD}}
```

Evaluator:
```python
{{EVALUATOR_SOURCE}}
```

Program:
```python
{{PROGRAM_SOURCE}}
```

Existing `_results.json`:
```json
{{RESULTS_JSON}}
```
"""
