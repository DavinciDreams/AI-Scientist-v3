"""Stage 3: LLM-driven variant selection and validation.

A separate LLM call reviews the proposals from the generation stage and filters
them based on relevance, feasibility, and single-change scope.  Hard caps are
enforced programmatically after the LLM filtering.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

from .agent_prompt import stage_agent_prompt
from .constants import MAX_ABLATIONS, MAX_EXTRA_BASELINES
from .models import VariantCandidate, VariantSelection

logger = logging.getLogger(__name__)

PopenFactory = Callable[..., subprocess.Popen[str]]
RegisterFn = Callable[[subprocess.Popen[str]], None]
UnregisterFn = Callable[[subprocess.Popen[str]], None]
TerminateFn = Callable[..., None]
EnsureTextFn = Callable[[Optional[str | bytes]], str]
ContainerEnvFn = Callable[..., dict[str, str]]

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "variant_select.md"


def validate_and_select_variants(
    *,
    candidates: list[VariantCandidate],
    bundle_dir: Path,
    generation_dir: Path,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: int,
    timeout_grace_seconds: int,
    max_ablations: int = MAX_ABLATIONS,
    max_extra_baselines: int = MAX_EXTRA_BASELINES,
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
) -> VariantSelection:
    """Review and filter variant proposals via LLM + hard caps.

    Writes ``generation/variant_selection.json``.
    """
    generation_dir = generation_dir.resolve()
    generation_dir.mkdir(parents=True, exist_ok=True)

    if not candidates:
        logger.info("No candidates to select from; returning empty selection")
        selection = VariantSelection(accepted=[], rejected=[], controls=["control_best"])
        _write_selection_json(generation_dir, selection)
        return selection

    # --- run LLM validation ---
    llm_decisions = _run_llm_validation(
        candidates=candidates,
        bundle_dir=bundle_dir,
        generation_dir=generation_dir,
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
    )

    # --- apply hard caps ---
    selection = _apply_hard_caps(
        candidates=candidates,
        llm_decisions=llm_decisions,
        max_ablations=max_ablations,
        max_extra_baselines=max_extra_baselines,
    )

    _write_selection_json(generation_dir, selection)
    logger.info(
        "Selection complete: %d accepted, %d rejected",
        len(selection.accepted),
        len(selection.rejected),
    )
    return selection


def _run_llm_validation(
    *,
    candidates: list[VariantCandidate],
    bundle_dir: Path,
    generation_dir: Path,
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
) -> dict[str, tuple[bool, str]]:
    """Run the LLM validation agent.

    Returns a dict mapping ``variant_id`` to ``(accepted, reason)``.
    """
    from src.investigation.script_runners import _execute_script_command

    prompt = _build_selection_prompt(candidates, bundle_dir)
    try:
        prompt_filename = stage_agent_prompt(generation_dir, "variant_select_prompt.md", prompt)
    except OSError as error:
        logger.warning(
            "Failed to stage variant selection prompt; rejecting unvalidated candidates: %s",
            error,
        )
        return {
            candidate.variant_id: (False, "rejected (prompt staging failed)")
            for candidate in candidates
        }

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

    logger.info("Running selection validation agent (%s)", agent)
    output, exit_code = _execute_script_command(
        cmd=cmd,
        timeout_seconds=timeout_seconds,
        timeout_grace_seconds=timeout_grace_seconds,
        script_label="variant_select",
        missing_output="[variants] variant_agent.sh not found\n",
        popen_factory=popen_factory,
        register_subprocess=register_subprocess,
        unregister_subprocess=unregister_subprocess,
        terminate_process_group=terminate_process_group,
        ensure_text=ensure_text,
        env=env,
    )

    if exit_code != 0:
        logger.warning("Selection agent exited with code %d; accepting all candidates", exit_code)
        return {c.variant_id: (True, "accepted (agent failed)") for c in candidates}

    # Parse variant_selection.json written by agent
    selection_path = generation_dir / "variant_selection.json"
    if not selection_path.exists():
        logger.warning("variant_selection.json not found; accepting all candidates")
        return {c.variant_id: (True, "accepted (no selection output)") for c in candidates}

    try:
        raw = json.loads(selection_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to parse variant_selection.json; accepting all")
        return {c.variant_id: (True, "accepted (parse error)") for c in candidates}

    decisions: dict[str, tuple[bool, str]] = {}
    for entry in raw.get("decisions", []):
        vid = entry.get("variant_id", "")
        accepted = entry.get("accepted", True)
        reason = entry.get("reason", "")
        decisions[vid] = (accepted, reason)

    # Any candidates not in decisions are accepted by default.
    for c in candidates:
        if c.variant_id not in decisions:
            decisions[c.variant_id] = (True, "accepted (not evaluated by agent)")

    return decisions


def _apply_hard_caps(
    *,
    candidates: list[VariantCandidate],
    llm_decisions: dict[str, tuple[bool, str]],
    max_ablations: int,
    max_extra_baselines: int,
) -> VariantSelection:
    """Apply LLM decisions and enforce hard caps."""
    accepted: list[VariantCandidate] = []
    rejected: list[tuple[VariantCandidate, str]] = []

    ablation_count = 0
    baseline_count = 0

    for candidate in candidates:
        is_accepted, reason = llm_decisions.get(candidate.variant_id, (True, "no decision"))

        if not is_accepted:
            rejected.append((candidate, reason))
            continue

        # Enforce hard caps
        if candidate.kind == "ablation":
            if ablation_count >= max_ablations:
                rejected.append((candidate, f"ablation cap reached ({max_ablations})"))
                continue
            ablation_count += 1
        elif candidate.kind == "baseline":
            if baseline_count >= max_extra_baselines:
                rejected.append((candidate, f"baseline cap reached ({max_extra_baselines})"))
                continue
            baseline_count += 1

        accepted.append(candidate)

    controls = ["control_best"]
    return VariantSelection(accepted=accepted, rejected=rejected, controls=controls)


def _build_selection_prompt(
    candidates: list[VariantCandidate],
    bundle_dir: Path,
) -> str:
    """Build the prompt for the selection validation agent."""
    if _PROMPT_TEMPLATE_PATH.exists():
        template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        template = _DEFAULT_SELECTION_PROMPT

    # Read task.md
    task_md = ""
    task_md_path = bundle_dir / "task.md"
    if task_md_path.exists():
        task_md = task_md_path.read_text(encoding="utf-8")

    candidates_json = json.dumps(
        [
            {
                "variant_id": c.variant_id,
                "title": c.title,
                "kind": c.kind,
                "hypothesis": c.hypothesis,
                "relevance_reason": c.relevance_reason,
                "feasibility_reason": c.feasibility_reason,
                "target_components": c.target_components,
                "implementation_plan": c.implementation_plan,
                "expected_effect": c.expected_effect,
            }
            for c in candidates
        ],
        indent=2,
    )

    # Compute program diff so the selector can judge whether ablations
    # target innovations (components new in the best program).
    program_diff = _compute_program_diff(bundle_dir)

    replacements = {
        "{{TASK_MD}}": task_md,
        "{{PROGRAM_DIFF}}": program_diff,
        "{{CANDIDATES_JSON}}": candidates_json,
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
        max_chars = 50_000
        if len(diff_text) > max_chars:
            diff_text = diff_text[:max_chars] + "\n... (diff truncated)"
        return diff_text
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "(diff unavailable)"


def _write_selection_json(
    generation_dir: Path,
    selection: VariantSelection,
) -> None:
    """Write the final ``variant_selection.json``."""
    data = {
        "controls": selection.controls,
        "accepted": [
            {
                "variant_id": c.variant_id,
                "title": c.title,
                "kind": c.kind,
                "hypothesis": c.hypothesis,
                "source": "custom" if c.description_body else "generated",
            }
            for c in selection.accepted
        ],
        "rejected": [
            {
                "variant_id": c.variant_id,
                "title": c.title,
                "kind": c.kind,
                "reason": reason,
            }
            for c, reason in selection.rejected
        ],
        "num_accepted": len(selection.accepted),
        "num_rejected": len(selection.rejected),
    }
    path = generation_dir / "variant_selection.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


_DEFAULT_SELECTION_PROMPT = """You are reviewing proposed ablation/baseline variants for a scientific comparison.

## Task Description
{{TASK_MD}}

## Proposed Candidates
{{CANDIDATES_JSON}}

## Instructions
Evaluate each candidate and write `variant_selection.json` with your decisions:

{
  "decisions": [
    {
      "variant_id": "...",
      "accepted": true/false,
      "reason": "why accepted or rejected"
    }
  ]
}

Accept candidates that are:
- Directly relevant to understanding the best method's contributions
- Implementable by one agent in one isolated workspace
- Single-change in scope (one modification axis only)
- Useful for a paper comparison

Reject candidates that are:
- Filler or quota-driven
- Hyperparameter tuning sweeps
- Speculative improvements rather than controls
- Multi-change (modifying more than one component)
- Generic baselines with no task-specific rationale
- Dependent on new datasets, services, or large refactors

Zero accepted proposals is a valid outcome.
"""
