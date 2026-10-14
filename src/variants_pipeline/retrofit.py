"""Prepare analysis bundles, preserving brokered candidates without agent edits."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from src.investigation.safe_staging import (
    atomic_write_bytes,
    read_text as safe_read_text,
    remove_path,
)
from src.investigation.evaluator_tree import evaluator_tree_digest

from .agent_prompt import stage_agent_prompt
from .constants import ANALYSIS_CSV_SCHEMAS
from .retrofit_preservation import (
    brokered_preservation_metadata,
    copy_brokered_bundle,
    preserved_program_digest,
)

logger = logging.getLogger(__name__)

PopenFactory = Callable[..., subprocess.Popen[str]]
RegisterFn = Callable[[subprocess.Popen[str]], None]
UnregisterFn = Callable[[subprocess.Popen[str]], None]
TerminateFn = Callable[..., None]
EnsureTextFn = Callable[[Optional[str | bytes]], str]
ContainerEnvFn = Callable[..., dict[str, str]]

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "variant_retrofit.md"
_RETROFIT_COMPLETE_FILENAME = ".retrofit_complete.json"
_MAX_EVALUATOR_BYTES = 16 * 1024 * 1024


def retrofit_bundle_complete(
    bundle_dir: Path,
    source_dir: Path,
    *,
    task_source_dir: Path | None = None,
) -> bool:
    """Verify the evaluator seal and any required brokered-program preservation."""
    try:
        marker = json.loads(
            safe_read_text(
                bundle_dir,
                _RETROFIT_COMPLETE_FILENAME,
                maximum_bytes=4096,
            )
        )
        evaluator = safe_read_text(
            bundle_dir,
            "evaluate.py",
            maximum_bytes=_MAX_EVALUATOR_BYTES,
        ).encode()
        source_digest = evaluator_tree_digest(source_dir)
        preservation = brokered_preservation_metadata(source_dir, task_source_dir)
        if preservation is not None:
            copied_digest = preserved_program_digest(bundle_dir, preservation)
            if not isinstance(marker, dict) or not (
                marker.get("schema_version") == 2
                and all(marker.get(key) == value for key, value in preservation.items())
                and marker.get("copied_program_tree_sha256") == copied_digest
            ):
                return False
        elif not isinstance(marker, dict) or marker.get("schema_version") != 1:
            return False
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(marker, dict)
        and marker.get("evaluator_sha256") == hashlib.sha256(evaluator).hexdigest()
        and marker.get("source_tree_sha256") == source_digest
    )


def retrofit_program(
    *,
    bundle_dir: Path,
    output_bundle_dir: Path,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: int,
    timeout_grace_seconds: int,
    reference_evaluator_source: str = "",
    containerize: bool = False,
    container_image: str = "",
    container_network: str = "bridge",
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    task_source_dir: Path | None = None,
    container_session_name: str | None = None,
    container_mount_root: Path | None = None,
    script_path: Path | None = None,
    popen_factory: PopenFactory | None = None,
    register_subprocess: RegisterFn | None = None,
    unregister_subprocess: UnregisterFn | None = None,
    terminate_process_group: TerminateFn | None = None,
    container_execution_env: ContainerEnvFn | None = None,
    ensure_text: EnsureTextFn | None = None,
) -> dict[str, Any]:
    """Copy brokered programs unchanged; instrument only legacy executables.

    Parameters
    ----------
    reference_evaluator_source:
        Optional read-only evaluator context from another task.  The agent may
        use it to match program-output conventions but cannot modify the
        host-sealed evaluator.

    Returns a retrofit report dict.
    """
    from src.investigation.script_runners import _execute_script_command

    bundle_dir = bundle_dir.resolve()
    output_bundle_dir = output_bundle_dir.resolve()
    report = {
        "stage": "retrofit",
        "bundle_dir": str(bundle_dir),
        "output_bundle_dir": str(output_bundle_dir),
        "exit_code": 1,
        "agent": agent,
        "agent_model": agent_model,
        "agent_profile": agent_profile,
        "reasoning_effort": reasoning_effort,
    }
    try:
        preservation = brokered_preservation_metadata(bundle_dir, task_source_dir)
        sealed_source_digest = evaluator_tree_digest(bundle_dir)
        sealed_evaluator = safe_read_text(
            bundle_dir,
            "evaluate.py",
            maximum_bytes=_MAX_EVALUATOR_BYTES,
        ).encode()
    except (OSError, ValueError, UnicodeError) as error:
        report["error"] = f"Cannot seal evaluator: {error}"
        return report

    # --- copy bundle ---
    if output_bundle_dir.exists():
        shutil.rmtree(output_bundle_dir)
    if preservation is not None:
        copy_brokered_bundle(bundle_dir, output_bundle_dir)
    else:
        shutil.copytree(bundle_dir, output_bundle_dir, symlinks=True)
    remove_path(output_bundle_dir, _RETROFIT_COMPLETE_FILENAME)
    logger.info("Copied bundle to %s", output_bundle_dir)

    if preservation is not None:
        try:
            copied_digest = preserved_program_digest(output_bundle_dir, preservation)
            if evaluator_tree_digest(bundle_dir) != sealed_source_digest:
                raise ValueError("Sealed evaluator source changed during retrofit")
            if brokered_preservation_metadata(bundle_dir, task_source_dir) != preservation:
                raise ValueError("Brokered task or selected program changed during retrofit")
            marker = {
                "schema_version": 2,
                "evaluator_sha256": hashlib.sha256(sealed_evaluator).hexdigest(),
                "source_tree_sha256": sealed_source_digest,
                "copied_program_tree_sha256": copied_digest,
                **preservation,
            }
            atomic_write_bytes(
                output_bundle_dir,
                _RETROFIT_COMPLETE_FILENAME,
                (json.dumps(marker, sort_keys=True) + "\n").encode(),
            )
        except (OSError, ValueError, UnicodeError) as error:
            report["error"] = f"Cannot preserve brokered program: {error}"
            return report
        report.update(exit_code=0, mode="brokered_preserve", program_tree_sha256=copied_digest)
        logger.info("Preserved brokered candidate; host evaluator owns analysis outputs")
        return report

    # --- build prompt ---
    prompt = _build_retrofit_prompt(
        output_bundle_dir,
        reference_evaluator_source=reference_evaluator_source,
    )
    try:
        prompt_filename = stage_agent_prompt(
            output_bundle_dir, "variant_retrofit_prompt.md", prompt
        )
    except OSError as error:
        logger.error("Failed to stage retrofit prompt: %s", error)
        report["error"] = f"Failed to stage prompt: {error}"
        return report

    # --- resolve defaults ---
    if popen_factory is None:
        popen_factory = subprocess.Popen
    if register_subprocess is None:
        register_subprocess = lambda p: None  # noqa: E731
    if unregister_subprocess is None:
        unregister_subprocess = lambda p: None  # noqa: E731
    if terminate_process_group is None:
        from src.investigation.runtime import terminate_active_process_group

        terminate_process_group = terminate_active_process_group
    if ensure_text is None:
        ensure_text = lambda t: (
            t if isinstance(t, str) else (t.decode("utf-8", errors="replace") if t else "")
        )  # noqa: E731
    if container_execution_env is None:
        from src.investigation.runtime import container_execution_env as _cee

        container_execution_env = _cee

    # --- resolve script path ---
    if script_path is None:
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "variant_agent.sh"

    model = agent_model.strip() if isinstance(agent_model, str) else ""
    profile = agent_profile.strip() if isinstance(agent_profile, str) else ""
    effort = reasoning_effort.strip() if isinstance(reasoning_effort, str) else ""
    cmd = [
        "bash",
        str(script_path),
        "--agent",
        agent,
        "--work-dir",
        str(output_bundle_dir),
        "--prompt-file",
        prompt_filename,
        "--skill-step",
        "variant_edit",
    ]
    if task_source_dir is not None:
        cmd.extend(["--task-source-dir", str(task_source_dir)])
    if model:
        cmd.extend(["--agent-model", model])
    if profile:
        cmd.extend(["--agent-profile", profile])
    if effort:
        cmd.extend(["--reasoning-effort", effort])
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
        sanitize_ambient_controls=True,
    )

    logger.info("Running retrofit agent (%s) on %s", agent, output_bundle_dir)
    restore_error: OSError | ValueError | None = None
    try:
        output, exit_code = _execute_script_command(
            cmd=cmd,
            timeout_seconds=timeout_seconds,
            timeout_grace_seconds=timeout_grace_seconds,
            script_label="variant_retrofit",
            missing_output="[variants] variant_agent.sh not found\n",
            popen_factory=popen_factory,
            register_subprocess=register_subprocess,
            unregister_subprocess=unregister_subprocess,
            terminate_process_group=terminate_process_group,
            ensure_text=ensure_text,
            env=env,
        )
    finally:
        try:
            remove_path(output_bundle_dir, _RETROFIT_COMPLETE_FILENAME)
            atomic_write_bytes(output_bundle_dir, "evaluate.py", sealed_evaluator)
        except (OSError, ValueError) as error:
            restore_error = error

    if restore_error is not None:
        logger.error("Failed to restore sealed evaluator: %s", restore_error)
        report["error"] = f"Failed to restore sealed evaluator: {restore_error}"
        return report

    try:
        source_unchanged = evaluator_tree_digest(bundle_dir) == sealed_source_digest
    except (OSError, ValueError) as error:
        report["error"] = f"Cannot verify sealed evaluator source: {error}"
        return report
    if not source_unchanged:
        report["error"] = "Sealed evaluator source changed during retrofit"
        return report

    report["exit_code"] = exit_code

    if exit_code != 0:
        logger.warning("Retrofit agent exited with code %d", exit_code)
        report["error"] = f"Agent exited with code {exit_code}"
    else:
        marker = {
            "schema_version": 1,
            "evaluator_sha256": hashlib.sha256(sealed_evaluator).hexdigest(),
            "source_tree_sha256": sealed_source_digest,
        }
        try:
            atomic_write_bytes(
                output_bundle_dir,
                _RETROFIT_COMPLETE_FILENAME,
                (json.dumps(marker, sort_keys=True) + "\n").encode(),
            )
        except (OSError, ValueError) as error:
            report["exit_code"] = 1
            report["error"] = f"Failed to publish retrofit completion: {error}"
            return report
        logger.info("Retrofit agent completed successfully")

    return report


def _build_retrofit_prompt(
    bundle_dir: Path,
    reference_evaluator_source: str = "",
) -> str:
    """Build the prompt for the retrofit agent."""
    # Load template
    if _PROMPT_TEMPLATE_PATH.exists():
        template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        template = _DEFAULT_RETROFIT_PROMPT

    # Read program source
    program_dir = bundle_dir / "program_0"
    program_source = ""
    if program_dir.is_dir():
        # Find entrypoint
        spec_path = program_dir / "program_spec.toml"
        if spec_path.exists():
            try:
                from src.investigation.program_spec import load_program_spec, resolve_entrypoint

                spec = load_program_spec(program_dir)
                if spec is not None:
                    entrypoint = resolve_entrypoint(program_dir, spec)
                    program_source = entrypoint.read_text(encoding="utf-8")
            except Exception:
                logger.warning("Could not load program spec, using fallback")

    # Read evaluator source
    evaluator_source = ""
    eval_path = bundle_dir / "evaluate.py"
    if eval_path.exists():
        evaluator_source = eval_path.read_text(encoding="utf-8")

    # Read task description
    task_description = ""
    task_md_path = bundle_dir / "task.md"
    if task_md_path.exists():
        task_description = task_md_path.read_text(encoding="utf-8")

    # Read extraction manifest
    manifest_text = ""
    manifest_path = bundle_dir / "extraction_manifest.json"
    if manifest_path.exists():
        manifest_text = manifest_path.read_text(encoding="utf-8")

    # Build CSV schema description
    schema_desc = _format_csv_schemas()

    replacements = {
        "{{PROGRAM_SOURCE}}": program_source,
        "{{EVALUATOR_SOURCE}}": evaluator_source,
        "{{EXTRACTION_MANIFEST}}": manifest_text,
        "{{CSV_SCHEMAS}}": schema_desc,
        "{{TASK_DESCRIPTION}}": task_description,
        "{{REFERENCE_EVALUATOR_SOURCE}}": reference_evaluator_source,
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)

    return result


def _format_csv_schemas() -> str:
    """Format the CSV schema definitions for inclusion in the prompt."""
    lines = []
    for filename, columns in ANALYSIS_CSV_SCHEMAS.items():
        lines.append(f"### {filename}")
        lines.append(f"Required columns: {', '.join(columns)}")
        lines.append("")
    return "\n".join(lines)


_DEFAULT_RETROFIT_PROMPT = """You are preparing a program for final evaluation in a scientific comparison study.

## Step 0: Understand the Task and Plan Evaluation Metrics

Read this task description (written for the search/investigation phase, not final evaluation):
```
{{TASK_DESCRIPTION}}
```

Before reading code, identify observational metrics already available during execution. Respect all task constraints and sealed data boundaries. Report missing evidence without adding experiments.

## Step 1: Read Code, Finalize Metrics

Read the program and evaluator below. Identify existing measurements and preserve the complete execution protocol.

## Step 2: Implement Changes

Modify the training program in `program_0/` (the main program.py entrypoint) to:

1. **Log training metrics per step/epoch during the training loop.** Store in memory, write to timeseries.csv after training.

2. **Write CSV analysis files** after training completes, under an `analysis/` subdirectory:

{{CSV_SCHEMAS}}

   Timeseries values must be finite numeric values. Omit observations whose values are unavailable; never write NaN or infinity. Encode boolean observations as numeric `1` or `0`, never Python `True`/`False` strings.

3. Do not change stopping conditions, seeds, or compute budgets. Preserve sampling, search schedules, initialization, and all existing outputs.

4. Log available evaluation metrics identified in Steps 0-1; report unavailable evidence.

## Rules
- You are primarily modifying program.py (the training program)
- Do not modify `evaluate.py`; it is a host-sealed scientific protocol. Add paper-ready measurements in the program outputs instead.
- Keep task.md and all protocol constraints unchanged
- Add only observational logging; do not add training, solver, or candidate RPC calls
- Do not swallow program, evaluator, or CSV-writing errors
- Minimize performance overhead: accumulate in memory, write once at the end
- Use only Python stdlib (csv module) for writing CSVs
- Read variant metadata from env vars: VARIANT_ID, VARIANT_KIND, BASE_PROGRAM, EVALUATION_ID

## Evaluator Source (read-only)
```python
{{EVALUATOR_SOURCE}}
```

## Current Program Source (primary file to modify)
```python
{{PROGRAM_SOURCE}}
```
"""
