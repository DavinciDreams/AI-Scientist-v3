"""Stage 8: Generate a variants overview README for the paper-writing agent.

Builds a ``variants_readme.md`` with deterministic structure and agent-written
per-variant summaries.  If the agent call fails, summaries fall back to the
hypothesis text from variant specs.

Supports two extra-task layouts:

* **Structured** – pipeline-generated tasks with a ``variants/`` subdirectory
  and full ``task_bundle/program_*/analysis/`` nesting.
* **Flat** – manually added tasks where each variant is a direct subdirectory
  containing CSVs and an optional ``variant_spec.json``.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from .agent_prompt import stage_agent_prompt
from .models import VariantSpec

logger = logging.getLogger(__name__)

PopenFactory = Callable[..., subprocess.Popen[str]]
RegisterFn = Callable[[subprocess.Popen[str]], None]
UnregisterFn = Callable[[subprocess.Popen[str]], None]
TerminateFn = Callable[..., None]
EnsureTextFn = Callable[[Optional[str | bytes]], str]
ContainerEnvFn = Callable[..., dict[str, str]]

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "variant_readme.md"

_KIND_ORDER = {"control": 0, "baseline": 1, "ablation": 2}

_SUMMARIES_FILENAME = "variant_summaries.json"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_variants_readme(
    *,
    variant_specs: list[VariantSpec],
    output_dir: Path,
    task_dir: Path,
    primary_task_name: str | None = None,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: int,
    timeout_grace_seconds: int,
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
) -> Path:
    """Generate ``variants_readme.md`` in *output_dir*.

    Returns the path to the written file.
    """
    output_dir = output_dir.resolve()

    # Load pipeline artifacts
    pipeline_summary = _load_json(output_dir / "pipeline_summary.json")
    status_map = _build_status_map(pipeline_summary)
    task_info = _discover_tasks(
        output_dir,
        task_dir,
        primary_task_name=primary_task_name,
    )

    # Discover flat extra task variant specs
    flat_tasks: dict[str, list[dict[str, Any]]] = {}
    extra_tasks_root = output_dir / "extra_tasks"
    if extra_tasks_root.is_dir():
        for t in task_info:
            if t.get("role") == "flat":
                flat_specs = _load_flat_variant_specs(extra_tasks_root / t["name"])
                if flat_specs:
                    flat_tasks[t["name"]] = flat_specs
                    logger.info(
                        "Loaded %d flat variant specs for %s",
                        len(flat_specs),
                        t["name"],
                    )

    # Call agent for summaries
    agent_output = _run_agent_summaries(
        variant_specs=variant_specs,
        task_dir=task_dir,
        output_dir=output_dir,
        agent=agent,
        agent_model=agent_model,
        agent_profile=agent_profile,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        timeout_grace_seconds=timeout_grace_seconds,
        containerize=containerize,
        container_image=container_image,
        container_network=container_network,
        container_auth_root=container_auth_root,
        container_env_file=container_env_file,
        task_source_dir=task_source_dir,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
        script_path=script_path,
        popen_factory=popen_factory,
        register_subprocess=register_subprocess,
        unregister_subprocess=unregister_subprocess,
        terminate_process_group=terminate_process_group,
        container_execution_env=container_execution_env,
        ensure_text=ensure_text,
        flat_tasks=flat_tasks,
    )

    # Build summaries dict, falling back per-variant when needed
    if agent_output is not None:
        task_summary = agent_output.get("task_summary", "")
        variant_summaries = agent_output.get("variant_summaries", {})
    else:
        task_summary = ""
        variant_summaries = {}

    if not task_summary:
        task_summary = _fallback_task_summary(task_dir)

    summaries: dict[str, str] = {}
    for spec in variant_specs:
        agent_summary = variant_summaries.get(spec.variant_id, "")
        summaries[spec.variant_id] = agent_summary if agent_summary else _fallback_summary(spec)

    # Assemble and write
    readme = _assemble_readme(
        task_summary, variant_specs, summaries, status_map, task_info, flat_tasks
    )
    # Write as README.md — paper.sh automatically picks this up via RESULTS_README
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    logger.info("Variants README written to %s", readme_path)
    return readme_path


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------


def _run_agent_summaries(
    *,
    variant_specs: list[VariantSpec],
    task_dir: Path,
    output_dir: Path,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None,
    reasoning_effort: str | None = None,
    timeout_seconds: int,
    timeout_grace_seconds: int,
    containerize: bool,
    container_image: str,
    container_network: str,
    container_auth_root: Path | None,
    container_env_file: Path | None,
    task_source_dir: Path | None,
    container_session_name: str | None,
    container_mount_root: Path | None,
    script_path: Path | None,
    popen_factory: PopenFactory | None,
    register_subprocess: RegisterFn | None,
    unregister_subprocess: UnregisterFn | None,
    terminate_process_group: TerminateFn | None,
    container_execution_env: ContainerEnvFn | None,
    ensure_text: EnsureTextFn | None,
    flat_tasks: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any] | None:
    """Call an LLM agent to produce per-variant summaries.

    Returns the parsed JSON dict, or ``None`` on failure.
    """
    from src.investigation.script_runners import _execute_script_command

    prompt = _build_readme_prompt(variant_specs, task_dir, flat_tasks)

    work_dir = output_dir / "_readme_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        prompt_filename = stage_agent_prompt(work_dir, "variant_readme_prompt.md", prompt)
    except OSError as error:
        logger.warning(
            "Failed to stage README prompt; falling back to deterministic summaries: %s",
            error,
        )
        return None

    # Resolve defaults (same pattern as select.py)
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
        str(work_dir),
        "--prompt-file",
        prompt_filename,
        "--skill-step",
        "variant_verify",
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

    logger.info("Running README summarization agent (%s)", agent)
    output, exit_code = _execute_script_command(
        cmd=cmd,
        timeout_seconds=timeout_seconds,
        timeout_grace_seconds=timeout_grace_seconds,
        script_label="variant_readme",
        missing_output="[variants] variant_agent.sh not found\n",
        popen_factory=popen_factory,
        register_subprocess=register_subprocess,
        unregister_subprocess=unregister_subprocess,
        terminate_process_group=terminate_process_group,
        ensure_text=ensure_text,
        env=env,
    )

    if exit_code != 0:
        logger.warning(
            "README agent exited with code %d; falling back to hypothesis summaries",
            exit_code,
        )
        return None

    return _parse_agent_summaries(work_dir)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_readme_prompt(
    variant_specs: list[VariantSpec],
    task_dir: Path,
    flat_tasks: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Build the prompt for the summarization agent."""
    if _PROMPT_TEMPLATE_PATH.exists():
        template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        template = _DEFAULT_PROMPT

    task_md = _read_task_md(task_dir)

    variants_json = json.dumps(
        [
            {
                "variant_id": s.variant_id,
                "title": s.title,
                "kind": s.kind,
                "base_program": s.base_program,
                "hypothesis": s.hypothesis,
                "target_components": s.target_components,
                "expected_effect": s.expected_effect,
            }
            for s in variant_specs
        ],
        indent=2,
    )

    # Append flat extra task specs so the agent can summarise them too
    flat_section = ""
    if flat_tasks:
        flat_section = "\n\n## Extra Task Variants (manually added)\n"
        flat_section += json.dumps(flat_tasks, indent=2)

    replacements = {
        "{{TASK_MD}}": task_md,
        "{{VARIANTS_JSON}}": variants_json + flat_section,
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


# ---------------------------------------------------------------------------
# Parsing and fallback
# ---------------------------------------------------------------------------


def _parse_agent_summaries(work_dir: Path) -> dict[str, Any] | None:
    """Parse ``variant_summaries.json`` written by the agent."""
    path = work_dir / _SUMMARIES_FILENAME
    if not path.exists():
        logger.warning("Agent did not write %s", _SUMMARIES_FILENAME)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse %s: %s", _SUMMARIES_FILENAME, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("%s is not a JSON object", _SUMMARIES_FILENAME)
        return None
    return data


def _fallback_summary(spec: VariantSpec) -> str:
    """Return hypothesis text or a generic fallback."""
    if spec.hypothesis:
        return spec.hypothesis
    return f"{spec.title} ({spec.kind} variant)."


def _fallback_task_summary(task_dir: Path) -> str:
    """Extract first non-empty paragraph from task.md as a fallback summary."""
    task_md = _read_task_md(task_dir)
    for line in task_md.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return "Variants experiment."


# ---------------------------------------------------------------------------
# README assembly (pure function)
# ---------------------------------------------------------------------------

_KIND_SORT_KEY = {"control": 0, "baseline": 1, "ablation": 2}


def _assemble_readme(
    task_summary: str,
    variant_specs: list[VariantSpec],
    summaries: dict[str, str],
    status_map: dict[str, dict[str, Any]],
    task_info: list[dict[str, str]] | None = None,
    flat_tasks: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Assemble the final markdown README.  Pure function, no I/O.

    Parameters
    ----------
    flat_tasks:
        Mapping of flat task name → list of variant spec dicts loaded from
        ``variant_spec.json`` files.
    """
    flat_tasks = flat_tasks or {}
    lines: list[str] = []

    lines.append("# Variants Experiment Overview")
    lines.append("")
    lines.append(task_summary)
    lines.append("")

    # Tasks section
    if task_info:
        lines.append("## Evaluated Tasks")
        lines.append("")
        for t in task_info:
            role = t.get("role", "cross-task")
            if role == "primary":
                role_label = "(primary)"
            elif role == "flat":
                role_label = "(extra)"
            else:
                role_label = "(cross-task)"
            lines.append(f"### {t['name']} {role_label}")
            if t.get("goal"):
                lines.append(t["goal"])
            lines.append("")
            lines.append(f"**Results**: `{t['data_path']}`")
            lines.append("")

    lines.append("## Variants")
    lines.append("")

    sorted_specs = sorted(
        variant_specs,
        key=lambda s: (_KIND_SORT_KEY.get(s.kind, 9), s.variant_id),
    )

    for i, spec in enumerate(sorted_specs):
        status_info = status_map.get(spec.variant_id, {})
        status = status_info.get("status", "unknown")
        rubric = "passed" if status_info.get("rubric_passed") else "failed"

        lines.append(f"### {spec.title} (`{spec.variant_id}`)")
        lines.append(f"**Kind**: {spec.kind} | **Status**: {status} | **Rubric**: {rubric}")
        lines.append("")
        lines.append(summaries.get(spec.variant_id, _fallback_summary(spec)))
        lines.append("")
        lines.append(f"**Workspace**: `variants/{spec.variant_id}/`")
        if i < len(sorted_specs) - 1:
            lines.append("")
            lines.append("---")
        lines.append("")

    # Flat extra task variants
    if flat_tasks:
        for task_name, flat_specs in sorted(flat_tasks.items()):
            lines.append(f"## Extra Task: {task_name}")
            lines.append("")
            sorted_flat = sorted(
                flat_specs,
                key=lambda s: (
                    _KIND_SORT_KEY.get(s.get("kind", ""), 9),
                    s.get("variant_id", ""),
                ),
            )
            for j, fspec in enumerate(sorted_flat):
                vid = fspec.get("variant_id", "unknown")
                title = fspec.get("title", vid)
                kind = fspec.get("kind", "unknown")
                hypothesis = fspec.get("hypothesis", f"{title} ({kind} variant).")

                lines.append(f"### {title} (`{vid}`)")
                lines.append(f"**Kind**: {kind}")
                lines.append("")
                lines.append(hypothesis)
                lines.append("")
                lines.append(f"**Workspace**: `extra_tasks/{task_name}/{vid}/`")
                if j < len(sorted_flat) - 1:
                    lines.append("")
                    lines.append("---")
                lines.append("")

    lines.append("## Directory Guide")
    lines.append("")
    lines.append("| Path | Contents |")
    lines.append("|------|----------|")
    lines.append(
        "| `aggregated/{task}/run_summary.csv` | Per-task variant scores and replications |"
    )
    lines.append("| `aggregated/{task}/metric_values.csv` | Per-task metrics in long format |")
    lines.append("| `aggregated/{task}/timeseries.csv` | Per-task epoch-level training curves |")
    lines.append(
        "| `generation/variant_selection.json` | Accepted/rejected proposals with reasons |"
    )
    lines.append("| `pipeline_summary.json` | Per-variant execution status and rubric results |")
    lines.append("| `variants/{id}/variant_spec.json` | Hypothesis, target components, plan |")
    lines.append(
        "| `variants/{id}/task_bundle/analysis/` | Per-variant analysis CSVs (primary task) |"
    )
    has_structured_extra = task_info and any(t.get("role") == "cross-task" for t in task_info)
    has_flat_extra = bool(flat_tasks)
    if has_structured_extra:
        lines.append(
            "| `extra_tasks/{task}/variants/{id}/` | Per-variant workspace for extra tasks |"
        )
    if has_flat_extra:
        lines.append(
            "| `extra_tasks/{task}/{variant}/` | Per-variant results for manually added tasks |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_tasks(
    output_dir: Path,
    task_dir: Path,
    primary_task_name: str | None = None,
) -> list[dict[str, str]]:
    """Discover the primary task and any extra (cross-task) tasks.

    Returns a list of dicts with ``name``, ``data_path``, ``goal``, and
    ``role`` keys.  Role is one of ``"primary"``, ``"cross-task"``, or
    ``"flat"``.
    """
    tasks: list[dict[str, str]] = []

    # Primary task
    primary_name = primary_task_name or task_dir.name or "original"
    tasks.append(
        {
            "name": primary_name,
            "data_path": "variants/",
            "goal": _extract_task_goal(task_dir),
            "role": "primary",
        }
    )

    # Extra tasks from cross-task generalization (stage 6) or manual flat dirs
    extra_tasks_root = output_dir / "extra_tasks"
    if extra_tasks_root.is_dir():
        for task_subdir in sorted(extra_tasks_root.iterdir()):
            if not task_subdir.is_dir():
                continue
            if (task_subdir / "variants").is_dir():
                # Structured layout
                extra_task_dir = task_subdir / "_retrofitted_bundle"
                if not extra_task_dir.is_dir():
                    extra_task_dir = task_subdir
                tasks.append(
                    {
                        "name": task_subdir.name,
                        "data_path": f"extra_tasks/{task_subdir.name}/variants/",
                        "goal": _extract_task_goal(extra_task_dir),
                        "role": "cross-task",
                    }
                )
            elif _is_flat_extra_task(task_subdir):
                # Flat layout (manually added)
                tasks.append(
                    {
                        "name": task_subdir.name,
                        "data_path": f"extra_tasks/{task_subdir.name}/",
                        "goal": _extract_task_goal(task_subdir),
                        "role": "flat",
                    }
                )

    return tasks


def _is_flat_extra_task(task_dir: Path) -> bool:
    """Return True if *task_dir* looks like a flat extra task.

    A flat extra task has at least one non-underscore-prefixed subdirectory
    containing a ``variant_spec.json`` or any CSV file.
    """
    for child in task_dir.iterdir():
        if child.name.startswith("_") or not child.is_dir():
            continue
        if (child / "variant_spec.json").is_file():
            return True
        if any(child.glob("*.csv")):
            return True
    return False


def _load_flat_variant_specs(
    task_dir: Path,
) -> list[dict[str, Any]]:
    """Load ``variant_spec.json`` from each variant subdir in a flat task.

    Returns a list of dicts (raw JSON), one per variant.  Subdirs without
    a ``variant_spec.json`` are skipped.
    """
    specs: list[dict[str, Any]] = []
    for child in sorted(task_dir.iterdir()):
        if child.name.startswith("_") or not child.is_dir():
            continue
        spec_path = child / "variant_spec.json"
        if not spec_path.is_file():
            continue
        try:
            data = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load %s: %s", spec_path, exc)
            continue
        if isinstance(data, dict):
            specs.append(data)
    return specs


def _extract_task_goal(task_dir: Path) -> str:
    """Extract the first goal bullet from ``task.md``."""
    task_md_path = task_dir / "task.md"
    if not task_md_path.exists():
        return ""
    try:
        text = task_md_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    in_goal = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("# goal") or stripped.lower().startswith("## goal"):
            in_goal = True
            continue
        if in_goal and stripped.startswith("- "):
            # Return the first bullet, trimmed
            goal = stripped.lstrip("- ").strip()
            # Truncate to first sentence if very long
            if len(goal) > 200:
                dot = goal.find(". ", 0, 200)
                if dot > 0:
                    goal = goal[: dot + 1]
                else:
                    goal = goal[:200] + "..."
            return goal
        if in_goal and stripped.startswith("#"):
            break  # next section
    return ""


def _build_status_map(
    pipeline_summary: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Extract per-variant status from ``pipeline_summary.json``."""
    if not pipeline_summary:
        return {}
    variants_list = pipeline_summary.get("variants")
    if not isinstance(variants_list, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for entry in variants_list:
        if not isinstance(entry, dict):
            continue
        vid = entry.get("variant_id")
        if isinstance(vid, str):
            result[vid] = entry
    return result


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning ``None`` on failure."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_task_md(task_dir: Path) -> str:
    """Read ``task.md`` from *task_dir*."""
    task_md_path = task_dir / "task.md"
    if task_md_path.exists():
        return task_md_path.read_text(encoding="utf-8")
    return ""


# ---------------------------------------------------------------------------
# Default prompt (used if template file is missing)
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT = """You are writing concise summaries for a variants experiment README.

## Task Description
{{TASK_MD}}

## Variant Specifications
{{VARIANTS_JSON}}

## Instructions

Write a JSON file `variant_summaries.json` to your working directory:

```json
{
  "task_summary": "One sentence describing what this task optimizes.",
  "variant_summaries": {
    "<variant_id>": "2-3 sentences describing what this variant IS."
  }
}
```

Rules:
- task_summary: One sentence about the optimization objective.
- Per-variant: Describe WHAT the variant changes, not results.
- Controls: what the control represents.
- Ablations: which component is removed/disabled and what hypothesis it tests.
- Baselines: what alternative approach is used.
- Do NOT mention results, scores, or outcomes.
- Include ALL variant_ids.
"""
