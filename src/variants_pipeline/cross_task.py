"""Stage 6: Cross-task generalization.

For each extra task folder with an already pipeline-compatible, host-sealed
evaluator, the pipeline runs two phases:

1. **Adapt control** — the ``control_best`` program is adapted to the new task
   in isolation, producing ``_adapted_control/``.  This establishes *how* to
   integrate with the new task's evaluator.
2. **Adapt variants** — every other variant receives:
   - ``program_0/``  = the adapted control (already working on the new task)
   - ``original_variant_program/`` = the variant's best program from the
     original task (the algorithmic changes to port)

   The agent's job is to apply the variant's changes on top of the adapted
   control.

Results are stored under ``extra_tasks/{task_name}/``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from src.investigation.files import evaluation_results_complete
from src.investigation.program_clone import clone_program_dir

from .materialize import (
    _build_run_manifest,
    _copy_task_files,
)
from .models import VariantRunResult, VariantSpec
from .rubric_contract import build_rubric_contract, build_target_rubrics

logger = logging.getLogger(__name__)

_CONTROL_BEST_ID = "control_best"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_and_execute_cross_task(
    *,
    extra_task_paths: list[Path],
    variants_run_dir: Path,
    variant_specs: list[VariantSpec],
    timeout_seconds: int,
    evaluation_timeout_seconds: int | None = None,
    timeout_grace_seconds: int,
    retrofit_timeout_seconds: int,
    max_evals_per_variant: int,
    num_replications: int,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    task_source_dir: Path | None = None,
    containerize: bool = False,
    container_image: str = "",
    container_network: str = "bridge",
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    sandbox_image: str | None = None,
) -> dict[str, list[VariantRunResult]]:
    """Prepare and execute cross-task adaptation for every extra task.

    For each extra task, first runs the retrofit agent on the extra task's
    ``program_0`` to add analysis outputs.  The raw evaluator must already be
    variants-pipeline compatible and remains host-sealed.  Then proceeds with
    the two-phase control-first adaptation.

    Returns a mapping from task name to list of run results.
    """
    from .execute import execute_variant_agents
    from .retrofit import (
        retrofit_bundle_complete,
        retrofit_program,
    )
    from .runtime import prepare_evaluation_runtime
    all_results: dict[str, list[VariantRunResult]] = {}
    if not containerize:
        raise ValueError("cross-task agent execution requires container isolation")
    resolved_extra_task_paths = [
        path.expanduser().resolve() for path in extra_task_paths
    ]
    evaluator_digests = {
        path: _managed_evaluator_digest(path)
        for path in resolved_extra_task_paths
    }
    runtime_image = sandbox_image or container_image
    for extra_task_path in resolved_extra_task_paths:
        if not prepare_evaluation_runtime(
            extra_task_path,
            sandbox_image=runtime_image,
            container_network=container_network,
            evaluator_tree_sha256=evaluator_digests[extra_task_path],
        ):
            raise RuntimeError(
                f"cross-task {extra_task_path.name!r} managed runtime preflight failed"
            )
    for extra_task_path in resolved_extra_task_paths:
        if _managed_evaluator_digest(extra_task_path) != evaluator_digests[extra_task_path]:
            raise RuntimeError(
                f"cross-task {extra_task_path.name!r} evaluator changed during preflight"
            )

    original_task_md_text = _find_original_task_md(variants_run_dir)
    reference_evaluator_raw = _read_retrofitted_evaluator(variants_run_dir)
    # Format as a prompt section (empty string when no reference available)
    if reference_evaluator_raw:
        reference_evaluator = (
            "## Reference Evaluator (read-only context)\n\n"
            "The host-sealed evaluator below illustrates the program-output "
            "conventions expected by the pipeline:\n"
            "- Forwarding `--seed` and `--results_dir` to the program\n"
            "- Supporting multiple replications (running the program N times "
            "with different seeds, writing results to separate directories)\n"
            "- Appropriate timeout handling\n"
            "- Producing `metrics.json` and `correct.json` in the results dir\n\n"
            "Do not modify evaluate.py. Cross-task support requires the raw extra-task "
            "evaluator to already implement the pipeline CLI contract.\n\n"
            "```python\n"
            f"{reference_evaluator_raw}\n"
            "```\n"
        )
    else:
        reference_evaluator = ""

    # Separate control_best from the rest
    control_spec: VariantSpec | None = None
    other_specs: list[VariantSpec] = []
    for spec in variant_specs:
        if spec.variant_id == _CONTROL_BEST_ID:
            control_spec = spec
        else:
            other_specs.append(spec)
    if variant_specs and control_spec is None:
        raise RuntimeError("cross-task adaptation requires control_best")

    def _run_agents(
        specs: list[VariantSpec],
        run_dir: Path,
        task_src: Path,
        evaluator_source_digest: str,
    ) -> list[VariantRunResult]:
        return execute_variant_agents(
            variant_specs=specs,
            variants_run_dir=run_dir,
            agent=agent,
            agent_model=agent_model,
            agent_profile=agent_profile,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            evaluation_timeout_seconds=evaluation_timeout_seconds,
            timeout_grace_seconds=timeout_grace_seconds,
            max_evals_per_variant=max_evals_per_variant,
            num_replications=num_replications,
            task_source_dir=task_src,
            venv_dir=task_src,
            containerize=containerize,
            container_image=container_image,
            container_network=container_network,
            container_auth_root=container_auth_root,
            container_env_file=container_env_file,
            sandbox_image=sandbox_image,
            evaluator_source_dir=task_src,
            evaluator_source_digest=evaluator_source_digest,
        )

    for extra_task_path in resolved_extra_task_paths:
        if _managed_evaluator_digest(extra_task_path) != evaluator_digests[extra_task_path]:
            raise RuntimeError(
                f"cross-task {extra_task_path.name!r} evaluator changed before adaptation"
            )
        task_name = extra_task_path.name
        cross_task_dir = variants_run_dir / "extra_tasks" / task_name
        cross_task_dir.mkdir(parents=True, exist_ok=True)
        task_evaluator_digest = evaluator_digests[extra_task_path]

        # ------------------------------------------------------------------
        # Phase 0: retrofit only the extra task program and analysis outputs.
        # ------------------------------------------------------------------
        retrofitted_extra = cross_task_dir / "_retrofitted_bundle"
        if not retrofit_bundle_complete(
            retrofitted_extra, extra_task_path, task_source_dir=extra_task_path,
        ):
            logger.info(
                "Cross-task '%s' phase 0: retrofitting extra task", task_name,
            )
            retrofit_report = retrofit_program(
                bundle_dir=extra_task_path,
                output_bundle_dir=retrofitted_extra,
                agent=agent,
                agent_model=agent_model,
                agent_profile=agent_profile,
                reasoning_effort=reasoning_effort,
                timeout_seconds=retrofit_timeout_seconds,
                timeout_grace_seconds=timeout_grace_seconds,
                reference_evaluator_source=reference_evaluator,
                task_source_dir=extra_task_path,
                containerize=containerize,
                container_image=container_image,
                container_network=container_network,
                container_auth_root=container_auth_root,
                container_env_file=container_env_file,
            )
            if retrofit_report.get("exit_code") != 0 or not retrofit_bundle_complete(
                retrofitted_extra,
                extra_task_path,
                task_source_dir=extra_task_path,
            ):
                raise RuntimeError(f"cross-task {task_name!r} retrofit failed")
        else:
            logger.info(
                "Cross-task '%s': reusing existing retrofitted bundle at %s",
                task_name,
                retrofitted_extra,
            )

        # ------------------------------------------------------------------
        # Phase 1: adapt control_best to the new task
        # ------------------------------------------------------------------
        adapted_control_dir: Path | None = None
        ctrl_results: list[VariantRunResult] = []
        ctrl_specs: list[VariantSpec] = []

        if control_spec is not None:
            logger.info(
                "Cross-task '%s' phase 1: adapting control_best", task_name,
            )
            ctrl_specs = _prepare_control_workspace(
                control_spec=control_spec,
                extra_task_path=extra_task_path,
                retrofitted_extra=retrofitted_extra,
                cross_task_dir=cross_task_dir,
                original_task_md_text=original_task_md_text,
                agent=agent,
                agent_model=agent_model,
                agent_profile=agent_profile,
                reasoning_effort=reasoning_effort,
                max_evals_per_variant=max_evals_per_variant,
            )
            if ctrl_specs:
                ctrl_results = _run_agents(
                    ctrl_specs,
                    cross_task_dir,
                    extra_task_path,
                    task_evaluator_digest,
                )
                if len(ctrl_results) != 1 or not ctrl_results[0].rubric_passed:
                    raise RuntimeError(
                        f"cross-task {task_name!r} adapted control failed verification"
                    )
                # Find best adapted program
                ctrl_bundle = ctrl_specs[0].variant_dir / "task_bundle"
                adapted_control_dir = _find_best_program_dir(ctrl_bundle)
                if adapted_control_dir is None:
                    raise RuntimeError(
                        f"cross-task {task_name!r} adapted control has no evaluated program"
                    )
            else:
                raise RuntimeError(
                    f"cross-task {task_name!r} control_best has no evaluated program"
                )

        # ------------------------------------------------------------------
        # Phase 2: adapt each variant using the adapted control as seed
        # ------------------------------------------------------------------
        logger.info(
            "Cross-task '%s' phase 2: adapting %d variants",
            task_name,
            len(other_specs),
        )

        if other_specs:
            if adapted_control_dir is None:
                raise RuntimeError(
                    f"cross-task {task_name!r} has no verified adapted control"
                )
            variant_cross_specs = _prepare_variant_workspaces(
                variant_specs=other_specs,
                extra_task_path=extra_task_path,
                retrofitted_extra=retrofitted_extra,
                cross_task_dir=cross_task_dir,
                adapted_control_dir=adapted_control_dir,
                original_task_md_text=original_task_md_text,
                agent=agent,
                agent_model=agent_model,
                agent_profile=agent_profile,
                reasoning_effort=reasoning_effort,
                max_evals_per_variant=max_evals_per_variant,
            )
        else:
            variant_cross_specs = []

        if variant_cross_specs:
            variant_results = _run_agents(
                variant_cross_specs,
                cross_task_dir,
                extra_task_path,
                task_evaluator_digest,
            )
        else:
            variant_results = []

        # Combine control + variant results
        combined: list[VariantRunResult] = []
        combined.extend(ctrl_results)
        combined.extend(variant_results)
        all_results[task_name] = combined

        passed = sum(1 for r in combined if r.rubric_passed)
        logger.info(
            "Cross-task '%s': %d passed, %d failed out of %d",
            task_name,
            passed,
            len(combined) - passed,
            len(combined),
        )

    return all_results


def _managed_evaluator_digest(task_path: Path) -> str:
    from src.investigation.evaluator_tree import managed_evaluator_tree_digest

    try:
        return managed_evaluator_tree_digest(task_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cross-task {task_path.name!r} evaluator preflight failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Phase 1: adapt control_best
# ---------------------------------------------------------------------------


def _prepare_control_workspace(
    *,
    control_spec: VariantSpec,
    extra_task_path: Path,
    retrofitted_extra: Path,
    cross_task_dir: Path,
    original_task_md_text: str,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    max_evals_per_variant: int,
) -> list[VariantSpec]:
    """Create a workspace for adapting control_best to the extra task.

    Returns a single-element list (or empty if control has no evaluated
    program).
    """
    task_name = extra_task_path.name
    orig_bundle = control_spec.variant_dir / "task_bundle"
    best_program = _find_best_program_dir(orig_bundle)
    if best_program is None:
        return []

    variant_dir = cross_task_dir / "variants" / _CONTROL_BEST_ID
    task_bundle_dir = variant_dir / "task_bundle"

    _setup_task_bundle(
        task_bundle_dir=task_bundle_dir,
        retrofitted_extra=retrofitted_extra,
        program_src=best_program,
        original_task_md_text=original_task_md_text,
    )

    # Write control-specific adaptation task.md
    _write_control_cross_task_md(
        task_bundle_dir=task_bundle_dir,
        task_name=task_name,
        max_evals=max_evals_per_variant,
    )

    return _finalize_variant(
        variant_dir=variant_dir,
        task_bundle_dir=task_bundle_dir,
        orig_spec=control_spec,
        task_name=task_name,
        agent=agent,
        agent_model=agent_model,
        agent_profile=agent_profile,
        reasoning_effort=reasoning_effort,
    )


# ---------------------------------------------------------------------------
# Phase 2: adapt variants using adapted control
# ---------------------------------------------------------------------------


def _prepare_variant_workspaces(
    *,
    variant_specs: list[VariantSpec],
    extra_task_path: Path,
    retrofitted_extra: Path,
    cross_task_dir: Path,
    adapted_control_dir: Path,
    original_task_md_text: str,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    max_evals_per_variant: int,
) -> list[VariantSpec]:
    """Create cross-task workspaces for non-control variants."""
    if not adapted_control_dir.is_dir():
        raise ValueError("adapted control directory is missing")
    task_name = extra_task_path.name
    specs: list[VariantSpec] = []

    for orig_spec in variant_specs:
        orig_bundle = orig_spec.variant_dir / "task_bundle"
        best_program = _find_best_program_dir(orig_bundle)
        if best_program is None:
            logger.info(
                "Cross-task '%s': skipping variant '%s' (no evaluated program)",
                task_name,
                orig_spec.variant_id,
            )
            continue

        variant_dir = cross_task_dir / "variants" / orig_spec.variant_id
        task_bundle_dir = variant_dir / "task_bundle"

        _setup_task_bundle(
            task_bundle_dir=task_bundle_dir,
            retrofitted_extra=retrofitted_extra,
            program_src=adapted_control_dir,
            original_task_md_text=original_task_md_text,
        )

        # Clone the variant's original best program as a reference
        orig_variant_dst = task_bundle_dir / "original_variant_program"
        if orig_variant_dst.exists():
            shutil.rmtree(orig_variant_dst)
        clone_program_dir(src_dir=best_program, dst_dir=orig_variant_dst)

        # Write variant-specific adaptation task.md
        _write_variant_cross_task_md(
            task_bundle_dir=task_bundle_dir,
            orig_spec=orig_spec,
            task_name=task_name,
            has_adapted_control=True,
            max_evals=max_evals_per_variant,
        )

        new_specs = _finalize_variant(
            variant_dir=variant_dir,
            task_bundle_dir=task_bundle_dir,
            orig_spec=orig_spec,
            task_name=task_name,
            agent=agent,
            agent_model=agent_model,
            agent_profile=agent_profile,
            reasoning_effort=reasoning_effort,
        )
        specs.extend(new_specs)

    return specs


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _setup_task_bundle(
    *,
    task_bundle_dir: Path,
    retrofitted_extra: Path,
    program_src: Path,
    original_task_md_text: str,
) -> None:
    """Set up a cross-task task_bundle directory.

    Copies task files from the **retrofitted** extra task bundle (so that
    evaluate.py is variants-pipeline compatible), clones the seed program as
    program_0, and writes source/original task.md files.
    """
    # 1. Copy task-level files from the retrofitted extra task bundle
    _copy_task_files(retrofitted_extra, task_bundle_dir)

    # 2. Clone seed program as program_0 (respects program_spec.toml excludes,
    #    strips metrics.json / correct.json / logs automatically).
    program_dst = task_bundle_dir / "program_0"
    if program_dst.exists():
        shutil.rmtree(program_dst)
    clone_program_dir(src_dir=program_src, dst_dir=program_dst)

    # 3. Save original task.md as source_task.md
    (task_bundle_dir / "source_task.md").write_text(
        original_task_md_text, encoding="utf-8",
    )

    # 4. Rename the retrofitted task.md to original_task.md
    copied_task_md = task_bundle_dir / "task.md"
    original_task_md = task_bundle_dir / "original_task.md"
    if copied_task_md.is_file():
        copied_task_md.rename(original_task_md)
    elif (retrofitted_extra / "task.md").is_file():
        shutil.copy2(retrofitted_extra / "task.md", original_task_md)


def _finalize_variant(
    *,
    variant_dir: Path,
    task_bundle_dir: Path,
    orig_spec: VariantSpec,
    task_name: str,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None,
    reasoning_effort: str | None = None,
) -> list[VariantSpec]:
    """Write metadata files and return the VariantSpec."""
    variant_dir.mkdir(parents=True, exist_ok=True)

    # For cross-task, even control_best must create program_1+ (it's adapting,
    # not just evaluating), so override kind="control" -> "baseline".
    cross_task_kind = "baseline" if orig_spec.kind == "control" else orig_spec.kind

    spec = VariantSpec(
        variant_id=orig_spec.variant_id,
        title=f"Cross-task: {orig_spec.title} -> {task_name}",
        kind=cross_task_kind,
        base_program=orig_spec.base_program,
        variant_dir=variant_dir,
        hypothesis=orig_spec.hypothesis,
        relevance_reason=orig_spec.relevance_reason,
        feasibility_reason=orig_spec.feasibility_reason,
        target_components=orig_spec.target_components,
        implementation_plan=orig_spec.implementation_plan,
        expected_effect=orig_spec.expected_effect,
        description_body=orig_spec.description_body,
    )

    (variant_dir / "variant_spec.json").write_text(
        json.dumps(spec.to_dict(), indent=2), encoding="utf-8",
    )

    rubric_contract = build_rubric_contract(
        orig_spec.variant_id, cross_task_kind, base_program=orig_spec.base_program,
    )
    (variant_dir / "rubric_contract.json").write_text(
        json.dumps(rubric_contract, indent=2), encoding="utf-8",
    )
    (variant_dir / "target_rubrics.json").write_text(
        json.dumps(build_target_rubrics(rubric_contract), indent=2), encoding="utf-8",
    )

    manifest = _build_run_manifest(
        spec, agent, agent_model, agent_profile, reasoning_effort
    )
    (variant_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )

    shutil.copy2(
        variant_dir / "rubric_contract.json",
        task_bundle_dir / "rubric_contract.json",
    )

    logger.info("Cross-task '%s': materialized variant '%s'", task_name, spec.variant_id)
    return [spec]


def _find_best_program_dir(task_bundle_dir: Path) -> Path | None:
    """Find the highest-numbered ``program_N`` with ``metrics.json``.

    Returns ``None`` if no evaluated program exists.
    """
    best_idx = -1
    best_dir: Path | None = None

    if not task_bundle_dir.is_dir():
        return None

    for entry in task_bundle_dir.iterdir():
        if not entry.is_dir():
            continue
        m = re.match(r"^program_(\d+)$", entry.name)
        if not m:
            continue
        idx = int(m.group(1))
        if (
            (entry / "metrics.json").is_file()
            and evaluation_results_complete(entry)
            and idx > best_idx
        ):
            best_idx = idx
            best_dir = entry

    return best_dir


def _read_retrofitted_evaluator(variants_run_dir: Path) -> str:
    """Read the retrofitted evaluate.py from the original task.

    This is provided as a reference when retrofitting extra tasks so the
    agent can follow the same conventions (replications, ``--seed``,
    ``--results_dir``, timeout handling, etc.).
    """
    for candidate in [
        variants_run_dir / "_retrofitted_bundle" / "evaluate.py",
        variants_run_dir / "_extracted_bundle" / "evaluate.py",
    ]:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return ""


def _find_original_task_md(variants_run_dir: Path) -> str:
    """Retrieve the original task.md text from the retrofitted bundle or variants."""
    for candidate in [
        variants_run_dir / "_retrofitted_bundle" / "task.md",
        variants_run_dir / "_extracted_bundle" / "task.md",
    ]:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")

    variants_dir = variants_run_dir / "variants"
    if variants_dir.is_dir():
        for vdir in sorted(variants_dir.iterdir()):
            orig = vdir / "task_bundle" / "original_task.md"
            if orig.is_file():
                return orig.read_text(encoding="utf-8")

    return "(Original task description not available.)"


# ---------------------------------------------------------------------------
# Task.md writers
# ---------------------------------------------------------------------------


def _write_control_cross_task_md(
    task_bundle_dir: Path,
    task_name: str,
    *,
    max_evals: int = 2,
) -> None:
    """Write task.md for the control_best cross-task adaptation."""
    sections: list[str] = [
        "# Cross-Task Adaptation: Control (best program)\n\n",
        f"- **Variant ID:** `{_CONTROL_BEST_ID}`\n",
        f"- **Target task:** {task_name}\n",
        f"- **Max program attempts:** {max_evals}\n",
        "\n## Context\n\n"
        "`program_0/` contains the **control (best) program** developed for a "
        "different task. Your job is to adapt it to work with the current task "
        f"(**{task_name}**).\n\n"
        "- `source_task.md` describes the **original task** the program was built for.\n"
        "- `original_task.md` describes the **target task** you need to adapt to.\n"
        "- `evaluate.py` is the evaluator for the **target task**.\n",
        "\n## What To Do\n\n"
        "1. Read `original_task.md` to understand the target task requirements.\n"
        "2. Read `source_task.md` to understand what `program_0` was designed for.\n"
        "3. Study `evaluate.py` to understand how the target task evaluates programs.\n"
        "4. Create `program_1` by cloning `program_0` and adapting it:\n"
        "   - Adapt data loading, input/output formats, and hyperparameters "
        "to match the target task\n"
        "   - Ensure compatibility with the target task's `evaluate.py`\n"
        "5. Evaluate `program_1` and verify rubrics.\n"
        "6. If rubrics fail, create the next `program_<idx>` as a repair.\n\n"
        f"You may create up to **{max_evals}** program attempts "
        f"(`program_1` through `program_{max_evals}`). "
        "If a program passes all rubrics, stop immediately.\n",
        "\n## Task References\n\n"
        "- `source_task.md` — the original task this program was built for\n"
        "- `original_task.md` — the target task you are adapting to\n",
    ]

    (task_bundle_dir / "task.md").write_text("".join(sections), encoding="utf-8")


def _write_variant_cross_task_md(
    task_bundle_dir: Path,
    orig_spec: VariantSpec,
    task_name: str,
    *,
    has_adapted_control: bool,
    max_evals: int = 2,
) -> None:
    """Write task.md for a non-control variant cross-task adaptation."""
    sections: list[str] = [
        f"# Cross-Task Adaptation: {orig_spec.title}\n\n",
        f"- **Variant ID:** `{orig_spec.variant_id}`\n",
        f"- **Original kind:** {orig_spec.kind}\n",
        f"- **Target task:** {task_name}\n",
        f"- **Max program attempts:** {max_evals}\n",
    ]

    if has_adapted_control:
        sections.append(
            "\n## Context\n\n"
            "`program_0/` contains the **control program already adapted** to the "
            f"target task (**{task_name}**). It is a working baseline on this task.\n\n"
            "`original_variant_program/` contains the **variant program** from the "
            "original task. It implements the algorithmic changes described below.\n\n"
            "Your job is to **port the variant's changes** onto `program_0/` so that "
            "the variant runs on the new task.\n\n"
            "- `source_task.md` describes the **original task** both programs were "
            "built for.\n"
            "- `original_task.md` describes the **target task**.\n"
            "- `evaluate.py` is the evaluator for the **target task**.\n"
        )
        sections.append(
            "\n## What To Do\n\n"
            "1. Study `program_0/` to understand how the control was adapted to the "
            "new task.\n"
            "2. Study `original_variant_program/` to understand the variant's "
            "algorithmic changes relative to the original control.\n"
            "3. Create `program_1` by cloning `program_0/` and applying the "
            "variant's changes:\n"
            "   - **Port the algorithmic changes** from `original_variant_program/` "
            "onto the adapted control\n"
            "   - Keep the task-adaptation work from `program_0/` (data loading, "
            "I/O formats, etc.) intact\n"
            "   - Ensure compatibility with `evaluate.py`\n"
            "4. Evaluate `program_1` and verify rubrics.\n"
            "5. If rubrics fail, create the next `program_<idx>` as a repair.\n\n"
            f"You may create up to **{max_evals}** program attempts "
            f"(`program_1` through `program_{max_evals}`). "
            "If a program passes all rubrics, stop immediately.\n"
        )
    else:
        # Fallback: no adapted control available
        sections.append(
            "\n## Context\n\n"
            "`program_0/` contains the variant program from the original task. "
            f"Your job is to adapt it to the target task (**{task_name}**).\n\n"
            "- `source_task.md` — the original task.\n"
            "- `original_task.md` — the target task.\n"
            "- `evaluate.py` — evaluator for the target task.\n"
        )
        sections.append(
            "\n## What To Do\n\n"
            "1. Read `original_task.md` and `source_task.md`.\n"
            "2. Create `program_1` by adapting `program_0` to the new task.\n"
            "3. Evaluate and verify rubrics.\n\n"
            f"You may create up to **{max_evals}** program attempts. "
            "Stop if all rubrics pass.\n"
        )

    if orig_spec.hypothesis:
        sections.append(
            "\n## Original Variant Information\n\n"
            f"**Hypothesis:** {orig_spec.hypothesis}\n"
        )
        if orig_spec.expected_effect:
            sections.append(f"\n**Expected effect:** {orig_spec.expected_effect}\n")
        if orig_spec.description_body:
            sections.append(f"\n{orig_spec.description_body}\n")

    sections.append(
        "\n## Task References\n\n"
        "- `source_task.md` — the original task\n"
        "- `original_task.md` — the target task you are adapting to\n"
    )

    (task_bundle_dir / "task.md").write_text("".join(sections), encoding="utf-8")
