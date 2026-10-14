"""Stage 2: LLM-driven variant candidate generation.

Uses an LLM agent to propose zero or more baseline/ablation candidates based
on the extracted bundle's task description and program diff.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from .agent_prompt import stage_agent_prompt
from .models import VariantCandidate

logger = logging.getLogger(__name__)

PopenFactory = Callable[..., subprocess.Popen[str]]
RegisterFn = Callable[[subprocess.Popen[str]], None]
UnregisterFn = Callable[[subprocess.Popen[str]], None]
TerminateFn = Callable[..., None]
EnsureTextFn = Callable[[Optional[str | bytes]], str]
ContainerEnvFn = Callable[..., dict[str, str]]

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "variant_generate.md"


def generate_variant_candidates(
    *,
    bundle_dir: Path,
    generation_dir: Path,
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
) -> list[VariantCandidate]:
    """Use an LLM agent to propose baseline/ablation candidates.

    Writes ``generation/change_summary.json`` and ``generation/variant_bundle.json``.
    Returns an empty list when the agent decides no relevant variants exist.
    """
    from src.investigation.script_runners import _execute_script_command

    bundle_dir = bundle_dir.resolve()
    generation_dir = generation_dir.resolve()
    generation_dir.mkdir(parents=True, exist_ok=True)

    # --- build prompt ---
    prompt = _build_generation_prompt(bundle_dir)
    try:
        prompt_filename = stage_agent_prompt(generation_dir, "variant_generate_prompt.md", prompt)
    except OSError as error:
        logger.error("Failed to stage variant generation prompt: %s", error)
        return []

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
        str(generation_dir),
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

    logger.info("Running variant generation agent (%s)", agent)
    output, exit_code = _execute_script_command(
        cmd=cmd,
        timeout_seconds=timeout_seconds,
        timeout_grace_seconds=timeout_grace_seconds,
        script_label="variant_generate",
        missing_output="[variants] variant_agent.sh not found\n",
        popen_factory=popen_factory,
        register_subprocess=register_subprocess,
        unregister_subprocess=unregister_subprocess,
        terminate_process_group=terminate_process_group,
        ensure_text=ensure_text,
        env=env,
    )

    if exit_code != 0:
        logger.warning("Generation agent exited with code %d", exit_code)

    # --- parse variant_bundle.json ---
    bundle_path = generation_dir / "variant_bundle.json"
    if not bundle_path.exists():
        logger.warning("variant_bundle.json not found at %s", bundle_path)
        return []

    raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    decision, candidates = _parse_variant_bundle(raw)

    # Write change summary
    summary_path = generation_dir / "change_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "decision": decision,
                "summary": raw.get("summary", ""),
                "num_candidates": len(candidates),
                "exit_code": exit_code,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Generation complete: decision=%s, %d candidates", decision, len(candidates))
    return candidates


def _build_generation_prompt(bundle_dir: Path) -> str:
    """Build the prompt for the generation agent."""
    if _PROMPT_TEMPLATE_PATH.exists():
        template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        template = _DEFAULT_GENERATION_PROMPT

    # Read task.md
    task_md = ""
    task_md_path = bundle_dir / "task.md"
    if task_md_path.exists():
        task_md = task_md_path.read_text(encoding="utf-8")

    # Read extraction manifest
    manifest_text = ""
    manifest_path = bundle_dir / "extraction_manifest.json"
    if manifest_path.exists():
        manifest_text = manifest_path.read_text(encoding="utf-8")

    # Compute diff between program_0 and reference_program
    program_diff = _compute_program_diff(bundle_dir)

    # Read evaluator summary
    evaluator_summary = ""
    eval_path = bundle_dir / "evaluate.py"
    if eval_path.exists():
        evaluator_summary = eval_path.read_text(encoding="utf-8")

    replacements = {
        "{{TASK_MD}}": task_md,
        "{{EXTRACTION_MANIFEST}}": manifest_text,
        "{{PROGRAM_DIFF}}": program_diff,
        "{{EVALUATOR_SUMMARY}}": evaluator_summary,
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def _compute_program_diff(bundle_dir: Path) -> str:
    """Compute a unified diff between reference_program and program_0."""
    ref_dir = bundle_dir / "reference_program"
    best_dir = bundle_dir / "program_0"

    if not ref_dir.is_dir() or not best_dir.is_dir():
        return "(no reference program available for diff)"

    try:
        result = subprocess.run(
            ["diff", "-ruN", str(ref_dir), str(best_dir)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        diff_text = result.stdout
        if not diff_text:
            return "(programs are identical)"
        # Truncate very long diffs
        max_chars = 50_000
        if len(diff_text) > max_chars:
            diff_text = diff_text[:max_chars] + "\n... (diff truncated)"
        return diff_text
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "(diff unavailable)"


def _parse_variant_bundle(
    raw: dict[str, Any],
) -> tuple[str, list[VariantCandidate]]:
    """Validate and parse ``variant_bundle.json``.

    Returns ``(decision, candidates)``.
    """
    decision = raw.get("decision", "no_relevant_variants")
    if decision == "no_relevant_variants":
        return decision, []

    candidates: list[VariantCandidate] = []
    for entry in raw.get("candidates", []):
        try:
            candidate = VariantCandidate(
                variant_id=str(entry["variant_id"]),
                title=str(entry["title"]),
                kind=str(entry["kind"]),
                base_program=str(entry.get("base_program", "best")),
                hypothesis=str(entry.get("hypothesis", "")),
                relevance_reason=str(entry.get("relevance_reason", "")),
                feasibility_reason=str(entry.get("feasibility_reason", "")),
                target_components=[str(c) for c in entry.get("target_components", [])],
                implementation_plan=[str(s) for s in entry.get("implementation_plan", [])],
                expected_effect=str(entry.get("expected_effect", "")),
            )
            candidates.append(candidate)
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed candidate entry: %s", exc)
            continue

    return decision, candidates


_DEFAULT_GENERATION_PROMPT = """You are a variant-generation agent for scientific ablation studies.

## Task Description
{{TASK_MD}}

## Extraction Manifest
{{EXTRACTION_MANIFEST}}

## Program Diff (best vs reference)
{{PROGRAM_DIFF}}

## Evaluator
{{EVALUATOR_SUMMARY}}

## Instructions
Propose zero or a few baselines/ablations for comparison against the best method.
Write your output to `variant_bundle.json` in the current directory.

The JSON must follow this schema:
{
  "decision": "propose_variants" or "no_relevant_variants",
  "summary": "brief explanation",
  "candidates": [
    {
      "variant_id": "snake_case_id",
      "title": "Human-readable title",
      "kind": "baseline" or "ablation",
      "base_program": "best",
      "hypothesis": "What we expect to learn",
      "relevance_reason": "Why this comparison matters",
      "feasibility_reason": "Why an agent can implement this",
      "target_components": ["component_to_change"],
      "implementation_plan": ["step 1", "step 2"],
      "expected_effect": "Expected impact on metrics"
    }
  ]
}

Rules:
- Do NOT meet a quota; zero proposals is valid and expected if no good comparisons exist
- Each candidate must be single-change in scope
- Each must be implementable by one agent in one isolated workspace
- Focus on paper-ready comparisons that help explain the best method's contributions
"""
