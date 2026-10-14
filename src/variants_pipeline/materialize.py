"""Stage 4: Materialize variant workspaces.

Creates ``variants/<variant_id>/`` directories, each containing a task bundle,
variant spec, rubric contract, and run manifest.  All variants are cloned from
the **retrofitted** bundle so that CSV analysis infrastructure is already in
place.
"""

from __future__ import annotations

import filecmp
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.investigation.constants import TASK_COPY_EXCLUDE_DIRS
from src.investigation.program_clone import clone_program_dir
from src.investigation.evaluation_evidence import EVIDENCE_FILENAME
from src.investigation.program_spec import load_program_spec, resolve_entrypoint
from src.investigation.safe_staging import remove_path
from src.investigation.files import (
    load_task_copy_ignore_patterns,
    task_copy_path_is_ignored,
)

from .models import VariantSelection, VariantSpec
from .reference_preflight import validate_reference_workspace
from .rubric_contract import build_rubric_contract, build_target_rubrics

logger = logging.getLogger(__name__)

# Files from previous runs that should NOT be copied into variant workspaces.
_TASK_COPY_EXCLUDE_FILES: set[str] = {
    "correct.json",
    EVIDENCE_FILENAME,
    "evaluation_manifest.json",
    "metrics.json",
    "rubric_status.json",
    "_results.json",
    "extraction_manifest.json",
}

_TASK_ROOT_EXCLUDE_FILES: set[str] = {
    ".agentic_runtime.json",
}

# File suffixes for log files that should not be copied.
_TASK_COPY_EXCLUDE_SUFFIXES: tuple[str, ...] = (
    ".log",
    ".out",
    ".err",
)
_RESERVED_VARIANT_IDS = {"baseline_reference", "control_best"}


def materialize_variant_workspaces(
    *,
    bundle_dir: Path,
    retrofitted_bundle_dir: Path,
    variants_run_dir: Path,
    selection: VariantSelection | None = None,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    max_evals_per_variant: int = 2,
    custom_only: bool = False,
    required_controls: set[str] | None = None,
) -> list[VariantSpec]:
    """Create all variant directories from the selection.

    Parameters
    ----------
    custom_only:
        When ``True``, skip ``control_best`` and ``baseline_reference``
        creation.  Only the accepted candidates (typically custom baselines)
        are materialized.
    required_controls:
        Controls that must still be materialized in ``custom_only`` mode.

    Returns a list of :class:`VariantSpec` for every materialized variant.
    """
    bundle_dir = bundle_dir.resolve()
    retrofitted_bundle_dir = retrofitted_bundle_dir.resolve()
    variants_run_dir = variants_run_dir.resolve()
    accepted = selection.accepted if selection is not None else []
    reserved_ids = sorted(
        candidate.variant_id
        for candidate in accepted
        if candidate.variant_id in _RESERVED_VARIANT_IDS
    )
    if reserved_ids:
        raise ValueError(f"Candidates use reserved variant IDs: {', '.join(reserved_ids)}")
    variants_dir = variants_run_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    specs: list[VariantSpec] = []
    required_controls = required_controls or set()
    unsupported_controls = required_controls - {"control_best"}
    if unsupported_controls:
        raise ValueError(f"Unsupported required controls: {sorted(unsupported_controls)}")

    if not custom_only or "control_best" in required_controls:
        # --- create control_best ---
        control_spec = _materialize_single_variant(
            variant_id="control_best",
            title="Control (best program)",
            kind="control",
            base_program="best",
            source_program_dir=retrofitted_bundle_dir / "program_0",
            retrofitted_bundle_dir=retrofitted_bundle_dir,
            variant_dir=variants_dir / "control_best",
            spec_data={
                "hypothesis": "Baseline control using the best program as-is.",
                "expected_effect": "Establishes the reference metrics.",
            },
            agent=agent,
            agent_model=agent_model,
            agent_profile=agent_profile,
            reasoning_effort=reasoning_effort,
            max_evals_per_variant=max_evals_per_variant,
        )
        specs.append(control_spec)

        # --- conditionally create baseline_reference ---
        if not custom_only and _reference_differs_from_best(bundle_dir):
            ref_program_dir = retrofitted_bundle_dir / "reference_program"
            if not ref_program_dir.is_dir():
                # Fall back to original bundle's reference_program
                ref_program_dir = bundle_dir / "reference_program"

            if ref_program_dir.is_dir():
                ref_spec = _materialize_single_variant(
                    variant_id="baseline_reference",
                    title="Baseline (reference / seed program)",
                    kind="baseline",
                    base_program="reference",
                    source_program_dir=ref_program_dir,
                    retrofitted_bundle_dir=retrofitted_bundle_dir,
                    variant_dir=variants_dir / "baseline_reference",
                    spec_data={
                        "hypothesis": "Compare the improved best program against the original seed.",
                        "expected_effect": "Shows improvement over the starting point.",
                    },
                    agent=agent,
                    agent_model=agent_model,
                    agent_profile=agent_profile,
                    reasoning_effort=reasoning_effort,
                    max_evals_per_variant=max_evals_per_variant,
                )
                specs.append(ref_spec)
    else:
        logger.info("custom_only mode: skipping control_best")
    if custom_only:
        logger.info("custom_only mode: skipping baseline_reference")

    # --- create accepted candidate variants ---
    for candidate in accepted:
        candidate_spec = _materialize_single_variant(
            variant_id=candidate.variant_id,
            title=candidate.title,
            kind=candidate.kind,
            base_program=candidate.base_program,
            source_program_dir=retrofitted_bundle_dir / "program_0",
            retrofitted_bundle_dir=retrofitted_bundle_dir,
            variant_dir=variants_dir / candidate.variant_id,
            spec_data={
                "hypothesis": candidate.hypothesis,
                "relevance_reason": candidate.relevance_reason,
                "feasibility_reason": candidate.feasibility_reason,
                "target_components": candidate.target_components,
                "implementation_plan": candidate.implementation_plan,
                "expected_effect": candidate.expected_effect,
                "description_body": candidate.description_body,
            },
            agent=agent,
            agent_model=agent_model,
            agent_profile=agent_profile,
            reasoning_effort=reasoning_effort,
            max_evals_per_variant=max_evals_per_variant,
            reference_dir=candidate.reference_dir,
        )
        specs.append(candidate_spec)

    logger.info("Materialized %d variant workspaces under %s", len(specs), variants_dir)
    return specs


def _materialize_single_variant(
    *,
    variant_id: str,
    title: str,
    kind: str,
    base_program: str,
    source_program_dir: Path,
    retrofitted_bundle_dir: Path,
    variant_dir: Path,
    spec_data: dict[str, Any],
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    max_evals_per_variant: int = 2,
    reference_dir: Path | None = None,
) -> VariantSpec:
    """Create one variant directory with all required artifacts."""
    variant_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy task-level files into task_bundle/
    task_bundle_dir = variant_dir / "task_bundle"
    _copy_task_files(retrofitted_bundle_dir, task_bundle_dir)

    # 2. Set up program_0 in the task_bundle
    program_dst = task_bundle_dir / "program_0"
    if program_dst.exists():
        shutil.rmtree(program_dst)

    if reference_dir is not None:
        # Folder-based baseline: copy reference code directly
        _copy_reference_as_program_0(reference_dir, program_dst)
        # Also copy control_best as control_program/ for the agent to compare
        control_dst = task_bundle_dir / "control_program"
        if control_dst.exists():
            shutil.rmtree(control_dst)
        clone_program_dir(src_dir=source_program_dir, dst_dir=control_dst)
        logger.info(
            "Folder baseline '%s': reference code -> program_0, control -> control_program",
            variant_id,
        )
    else:
        # Standard flow: clone from control_best
        clone_program_dir(src_dir=source_program_dir, dst_dir=program_dst)

    # 2b. Remove stale result/log files that clone_program_dir may have copied
    _remove_stale_files(program_dst)
    validate_reference_workspace(task_bundle_dir, variant_id, base_program)

    # 3. Write variant_spec.json
    spec = VariantSpec(
        variant_id=variant_id,
        title=title,
        kind=kind,
        base_program=base_program,
        variant_dir=variant_dir,
        hypothesis=spec_data.get("hypothesis", ""),
        relevance_reason=spec_data.get("relevance_reason", ""),
        feasibility_reason=spec_data.get("feasibility_reason", ""),
        target_components=spec_data.get("target_components", []),
        implementation_plan=spec_data.get("implementation_plan", []),
        expected_effect=spec_data.get("expected_effect", ""),
        description_body=spec_data.get("description_body", ""),
    )
    spec_path = variant_dir / "variant_spec.json"
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    # 4. Write rubric_contract.json and target_rubrics.json
    rubric_contract = build_rubric_contract(variant_id, kind, base_program=base_program)
    (variant_dir / "rubric_contract.json").write_text(
        json.dumps(rubric_contract, indent=2), encoding="utf-8"
    )
    target_rubrics = build_target_rubrics(rubric_contract)
    (variant_dir / "target_rubrics.json").write_text(
        json.dumps(target_rubrics, indent=2), encoding="utf-8"
    )

    # 5. Write run_manifest.json
    manifest = _build_run_manifest(
        spec, agent, agent_model, agent_profile, reasoning_effort
    )
    (variant_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # 6. Replace task.md with variant-specific instructions
    _write_variant_task_md(task_bundle_dir, spec, max_evals=max_evals_per_variant)

    # 7. Stage rubric_contract.json into task_bundle so the agent can see it
    rubric_contract_src = variant_dir / "rubric_contract.json"
    if rubric_contract_src.exists():
        shutil.copy2(rubric_contract_src, task_bundle_dir / "rubric_contract.json")

    logger.info("Materialized variant: %s (%s)", variant_id, kind)
    return spec


def _copy_reference_as_program_0(reference_dir: Path, program_dst: Path) -> None:
    """Copy a reference implementation folder as ``program_0``.

    Uses ``shutil.copytree`` (not ``clone_program_dir``) because reference
    folders may not have ``program_spec.toml``.  Excludes pipeline metadata
    (``baseline.md``) and common clutter directories.

    A default ``program_spec.toml`` is generated if missing so that the agent
    can use ``clone_program_dir`` later to clone ``program_0`` -> ``program_1``.
    """
    ignore = shutil.ignore_patterns(
        "baseline.md",
        "__pycache__",
        "*.pyc",
        ".git",
        ".venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    )
    shutil.copytree(reference_dir, program_dst, ignore=ignore)

    # Ensure program_spec.toml exists for downstream clone_program_dir
    spec_path = program_dst / "program_spec.toml"
    if not spec_path.is_file():
        _write_default_program_spec(program_dst)
        logger.info("Generated default program_spec.toml in %s", program_dst)


def _write_default_program_spec(program_dir: Path) -> None:
    """Write a minimal ``program_spec.toml`` for a reference implementation.

    Creates the ``src/`` directory if needed and places a placeholder
    ``program.py`` so the spec is valid.
    """
    src_dir = program_dir / "src"
    src_dir.mkdir(exist_ok=True)
    entrypoint = src_dir / "program.py"
    if not entrypoint.exists():
        entrypoint.write_text(
            "# Placeholder entrypoint — see reference code in parent directory\n",
            encoding="utf-8",
        )

    spec_content = """\
schema_version = 1
code_root = "src"

[entrypoint]
path = "program.py"
language = "python"
type = "script"

[copy]
exclude = [
    "**/__pycache__/**",
    "**/*.pyc",
    ".venv/**",
    "node_modules/**",
    ".git/**",
    ".mypy_cache/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
]
include = []

[integration]
kind = "generic"
notes = "Auto-generated for folder-based reference implementation"
"""
    (program_dir / "program_spec.toml").write_text(spec_content, encoding="utf-8")


def _remove_stale_files(program_dir: Path) -> None:
    """Remove result/log files from a cloned program directory.

    ``clone_program_dir`` copies everything allowed by ``program_spec.toml``,
    which may include stale outputs from a previous run (metrics, logs, etc.).
    Walk the tree and delete anything that matches the exclusion rules.
    """
    for path in list(program_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in _TASK_COPY_EXCLUDE_FILES or path.name.endswith(
            _TASK_COPY_EXCLUDE_SUFFIXES
        ):
            path.unlink()
            logger.debug("Removed stale file from program clone: %s", path)


def _copy_task_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy task-level files (evaluate.py, capture.py, etc.) into *dst_dir*.

    Excludes program directories and common cache/venv directories.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    remove_path(dst_dir, ".agentic_runtime.json")

    ignore_patterns = load_task_copy_ignore_patterns(src_dir)
    for entry in sorted(src_dir.iterdir()):
        name = entry.name

        if task_copy_path_is_ignored(
            name, is_dir=entry.is_dir(), patterns=ignore_patterns
        ):
            continue

        # Skip program directories
        if name.startswith("program_") or name == "reference_program":
            continue
        # Skip excluded directories
        if entry.is_dir() and name in TASK_COPY_EXCLUDE_DIRS:
            continue
        # Skip result/metric files from previous runs
        if name in _TASK_COPY_EXCLUDE_FILES or name in _TASK_ROOT_EXCLUDE_FILES:
            continue
        # Skip log files (.out/.err, .log, etc.)
        if name.endswith(_TASK_COPY_EXCLUDE_SUFFIXES):
            continue

        dst = dst_dir / name
        if entry.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(entry, dst, symlinks=True)
        else:
            shutil.copy2(entry, dst)


def _build_run_manifest(
    spec: VariantSpec,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Build the initial ``run_manifest.json``."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "variant_id": spec.variant_id,
        "variant_kind": spec.kind,
        "base_program": spec.base_program,
        "title": spec.title,
        "agent": agent,
        "agent_model": agent_model,
        "agent_profile": agent_profile,
        "reasoning_effort": reasoning_effort,
        "variant_dir": str(spec.variant_dir),
        "task_bundle_dir": str(spec.variant_dir / "task_bundle"),
        "expected_analysis_files": [
            "analysis/run_summary.csv",
            "analysis/metric_values.csv",
            "analysis/timeseries.csv",
            "rubric_status.json",
        ],
        "status": "created",
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "eval_count": 0,
    }


def _write_variant_task_md(
    task_bundle_dir: Path, spec: VariantSpec, *, max_evals: int = 2
) -> None:
    """Rename the original ``task.md`` and write a variant-specific one.

    The original task description is preserved as ``original_task.md`` for
    reference.  The new ``task.md`` contains variant-specific instructions so
    the agent knows exactly what change to make.
    """
    task_md = task_bundle_dir / "task.md"
    original_task_md = task_bundle_dir / "original_task.md"

    # Rename the original
    if task_md.exists():
        task_md.rename(original_task_md)

    sections: list[str] = []
    sections.append(f"# Variant: {spec.title}\n\n")
    sections.append(f"- **Variant ID:** `{spec.variant_id}`\n")
    sections.append(f"- **Kind:** {spec.kind}\n")
    sections.append(f"- **Base program:** {spec.base_program}\n")
    sections.append(f"- **Max program attempts:** {max_evals}\n")

    if spec.description_body:
        sections.append(f"\n## Detailed Description\n{spec.description_body}\n")

    if spec.hypothesis:
        sections.append(f"\n## Hypothesis\n{spec.hypothesis}\n")
    if spec.expected_effect:
        sections.append(f"\n## Expected Effect\n{spec.expected_effect}\n")
    if spec.target_components:
        components = ", ".join(f"`{c}`" for c in spec.target_components)
        sections.append(f"\n## Target Components\n{components}\n")
    if spec.implementation_plan:
        steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(spec.implementation_plan))
        sections.append(f"\n## Implementation Plan\n{steps}\n")

    if spec.kind == "control":
        sections.append(
            "\n## What To Do\n"
            "Evaluate `program_0` as-is. Do NOT modify the program.\n"
        )
    elif spec.kind == "ablation":
        sections.append(
            "\n## What To Do\n"
            "Create `program_1` by cloning `program_0` and applying ONLY the "
            "ablation described above. Do NOT evaluate `program_0` first — go "
            "straight to creating and evaluating `program_1`.\n\n"
            f"You may create up to **{max_evals}** program attempts "
            f"(`program_1` through `program_{max_evals}`). "
            "If a program passes all rubrics, stop immediately — do not use "
            "remaining attempts. If it fails, create the next `program_<idx>` "
            "as a repair, fixing only the rubric failures.\n"
        )
    elif spec.variant_id == "baseline_reference":
        sections.append(
            "\n## What To Do\n"
            "`program_0/` contains the original seed for this task. "
            "Clone it into `program_1` and evaluate it unchanged. "
            "Preserve its algorithm and parameters to measure improvement over "
            "the starting point. Report available evaluator evidence and its "
            "limitations; no external-reference adaptation audit is required.\n"
        )
    elif spec.kind == "baseline" and spec.base_program == "reference":
        # Folder-based baseline: adapt reference implementation
        sections.append(
            "\n## What To Do\n"
            "`program_0/` contains a **reference implementation** from an external source. "
            "Your task is to **adapt** it to integrate with our evaluation harness "
            "for a fair, apples-to-apples comparison.\n\n"
            "The **control program** is available at `control_program/`. Study it to understand:\n"
            "- How `evaluate.py` calls and evaluates the program\n"
            "- The expected function signatures and metric reporting format\n"
            "- Data loading API and file structure conventions\n\n"
            "### Step 1: Hyperparameter Audit\n\n"
            "Before making any code changes, compare the hyperparameters of the "
            "reference implementation (`program_0/`) with those of the control "
            "program (`control_program/`). Write your findings to "
            "`hyperparameter_comparison.md` as a table with these columns:\n\n"
            "| Parameter | Reference value | Control value | Decision | Reason |\n"
            "|-----------|----------------|---------------|----------|--------|\n\n"
            "For each parameter, the **Decision** should be one of:\n"
            "- **align**: This is an experimental condition (not part of the "
            "algorithm's contribution) that should match the control for a fair "
            "comparison\n"
            "- **preserve**: This is part of the algorithm's design and must be "
            "kept as-is\n\n"
            "The guiding question is: *is this difference an unfair advantage/"
            "disadvantage, or is it part of how the algorithm works?*\n\n"
            "### Step 2: Adapt the Implementation\n\n"
            "Create `program_1` by adapting the reference implementation. "
            "Use the hyperparameter audit to guide your decisions:\n"
            "1. Keep the **core algorithm** from `program_0/` unchanged — "
            "the scientific contribution must be preserved faithfully\n"
            "2. Adapt the **interface** to match what `evaluate.py` expects "
            "(use `control_program/` as reference for function signatures, "
            "metric reporting, and file structure)\n"
            "3. **Align** all parameters you marked as *align* in the audit\n"
            "4. **Preserve** all parameters you marked as *preserve* — "
            "changing these can silently degrade performance and produce "
            "misleading comparisons\n\n"
            "### Step 3: Performance Sanity Check\n\n"
            "After evaluating `program_1`, compare its performance against the "
            "expected performance documented in `program_0/` (e.g. README). "
            "If you observe a **large degradation**, revisit your audit — one of "
            "your *align* decisions may have been wrong. Revert any change that "
            "is not strictly required for `evaluate.py` to run, and re-evaluate.\n\n"
            "Do NOT evaluate `program_0` first — go straight to creating and evaluating "
            "`program_1`.\n\n"
            f"You may create up to **{max_evals}** program attempts "
            f"(`program_1` through `program_{max_evals}`). "
            "If a program passes all rubrics, stop immediately — do not use "
            "remaining attempts. If it fails, create the next `program_<idx>` "
            "as a repair, fixing only the rubric failures.\n"
        )
    elif spec.kind == "baseline":
        sections.append(
            "\n## What To Do\n"
            "Create `program_1` by cloning `program_0` and implementing the "
            "simpler approach described above. Do NOT evaluate `program_0` first "
            "— go straight to creating and evaluating `program_1`.\n\n"
            f"You may create up to **{max_evals}** program attempts "
            f"(`program_1` through `program_{max_evals}`). "
            "If a program passes all rubrics, stop immediately — do not use "
            "remaining attempts. If it fails, create the next `program_<idx>` "
            "as a repair, fixing only the rubric failures.\n"
        )

    if original_task_md.exists():
        sections.append(
            "\n## Original Task Context\n"
            "See `original_task.md` for the full task description and domain rubrics.\n"
        )

    task_md.write_text("".join(sections), encoding="utf-8")
    logger.debug("Wrote variant task.md for %s", spec.variant_id)


def _reference_differs_from_best(bundle_dir: Path) -> bool:
    """Check if ``reference_program/`` differs materially from ``program_0/``.

    Compares entrypoint source files.  Returns ``False`` if
    ``reference_program/`` does not exist.
    """
    ref_dir = bundle_dir / "reference_program"
    best_dir = bundle_dir / "program_0"

    if not ref_dir.is_dir():
        return False
    if not best_dir.is_dir():
        return False

    try:
        ref_spec = load_program_spec(ref_dir, required=False)
        best_spec = load_program_spec(best_dir, required=False)

        if ref_spec is None or best_spec is None:
            # Can't compare specs; assume they differ if both dirs exist.
            return True

        ref_entry = resolve_entrypoint(ref_dir, ref_spec)
        best_entry = resolve_entrypoint(best_dir, best_spec)

        return not filecmp.cmp(str(ref_entry), str(best_entry), shallow=False)
    except Exception:
        # On any error, assume they differ.
        return True
