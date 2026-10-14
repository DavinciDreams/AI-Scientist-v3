"""Configuration dataclasses for the variants pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.investigation.config import (
    PROFILE_MODEL_ERROR,
    GEMINI_REASONING_EFFORT_ERROR,
    VALID_REASONING_EFFORTS,
    AgentConfig,
    ContainerConfig,
    LoggingConfig,
)

logger = logging.getLogger(__name__)

ALL_STAGES = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8})


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_stages(stages_str: str) -> set[int]:
    """Parse a stage specification like ``"0,1,2"``, ``"0-3"``, or ``"2,4-5"``.

    Lives here rather than in ``cli.py`` so ``execution.stages`` is validated
    alongside every other config key, whether it arrived from the config file
    or from the ``--stages`` flag.
    """
    result: set[int] = set()
    for part in str(stages_str).split(","):
        part = part.strip()
        try:
            if "-" in part:
                low, high = part.split("-", 1)
                result.update(range(int(low), int(high) + 1))
            else:
                result.add(int(part))
        except ValueError:
            raise ValueError(f"Invalid stage specification: {stages_str!r}") from None
    invalid = sorted(result - ALL_STAGES)
    if invalid:
        # A range like "0-99" yields a long list; show enough to identify the
        # mistake without burying the valid range that follows it.
        shown = ", ".join(str(stage) for stage in invalid[:5])
        if len(invalid) > 5:
            shown += f", ... ({len(invalid)} in total)"
        raise ValueError(
            f"Invalid stage numbers: {shown}. Valid: {min(ALL_STAGES)}-{max(ALL_STAGES)}"
        )
    return result


@dataclass
class VariantsPathsConfig:
    """Path configuration for the variants pipeline."""

    task_dir: str = field(
        default="",
        metadata={"help": "Path to the original task directory"},
    )
    results_dir: str = field(
        default="",
        metadata={"help": "Path to the completed scientist run results directory"},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for variants run (auto-computed if omitted)"},
    )
    baseline_program_override: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to an alternative program directory to use as the baseline "
                "reference instead of task_dir/program_0.  Only experiment files "
                "(program.py, etc.) are taken from this directory; task-level files "
                "(task.md, evaluate.py) still come from task_dir."
            ),
        },
    )
    custom_baselines: list[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Comma-separated paths to custom baseline .md files (with YAML "
                "frontmatter) or folders containing reference implementations.  "
                "These baselines bypass LLM generation/selection and are always "
                "included in the variant run."
            ),
        },
    )
    extra_tasks: list[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Paths to extra task folders for cross-task generalization.  "
                "Each variant's best program will be adapted to each extra task.  "
                "Results are stored under extra_tasks/{task_name}/variants/."
            ),
        },
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Stable task name when task_dir is a run-local snapshot"},
    )


@dataclass
class VariantsExecutionConfig:
    """Execution settings for the variants pipeline."""

    stages: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Stage subset to run, e.g. '0,1' or '0-3'; null runs all stages. "
                "The --stages CLI flag is shorthand for this key."
            )
        },
    )
    custom_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Run only custom baselines, skipping control_best and baseline_reference. "
                "The --custom-only CLI flag is shorthand for this key."
            )
        },
    )
    max_ablations: int = field(
        default=3,
        metadata={"help": "Maximum number of ablation variants"},
    )
    max_extra_baselines: int = field(
        default=3,
        metadata={"help": "Maximum number of extra baseline variants"},
    )
    max_evals_per_variant: int = field(
        default=2,
        metadata={"help": "Max evaluations per variant (1 attempt + 1 repair)"},
    )
    num_replications: int = field(
        default=1,
        metadata={
            "help": "Number of parallel replications per evaluation (default: 1, single run)"
        },
    )
    sandbox_image: Optional[str] = field(
        default=None,
        metadata={"help": "Docker image reference resolved to an immutable ID before stage 5"},
    )
    variant_timeout_seconds: int = field(
        default=7200,
        metadata={"help": "Per-variant agent timeout in seconds"},
    )
    evaluation_timeout_seconds: Optional[int] = field(
        default=None,
        metadata={"help": "Host scoring deadline; null uses variant_timeout_seconds"},
    )
    variant_timeout_grace_seconds: int = field(
        default=30,
        metadata={"help": "Grace period after variant timeout"},
    )
    generation_timeout_seconds: int = field(
        default=1800,
        metadata={"help": "Variant generation LLM timeout in seconds"},
    )
    retrofit_timeout_seconds: int = field(
        default=1800,
        metadata={"help": "Retrofit LLM timeout in seconds"},
    )
    selection_timeout_seconds: int = field(
        default=1800,
        metadata={"help": "Selection LLM timeout in seconds"},
    )
    cross_task_timeout_seconds: int = field(
        default=7200,
        metadata={"help": "Per-variant agent timeout for cross-task adaptation in seconds"},
    )
    readme_timeout_seconds: int = field(
        default=600,
        metadata={"help": "Variant README generation LLM timeout in seconds"},
    )


@dataclass
class VariantsPipelineConfig:
    """Top-level configuration for the variants pipeline."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    paths: VariantsPathsConfig = field(default_factory=VariantsPathsConfig)
    execution: VariantsExecutionConfig = field(default_factory=VariantsExecutionConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    run_id: str = field(default_factory=_generate_run_id)


def validate_variants_config(cfg: VariantsPipelineConfig) -> list[str]:
    """Validate the config, returning a list of error messages (empty = valid)."""
    errors: list[str] = []

    if not cfg.paths.task_dir:
        errors.append("paths.task_dir is required")
    if not cfg.paths.results_dir:
        errors.append("paths.results_dir is required")
    if cfg.paths.task_name is not None and (
        not cfg.paths.task_name
        or cfg.paths.task_name in {".", ".."}
        or "/" in cfg.paths.task_name
        or "\\" in cfg.paths.task_name
        or "\0" in cfg.paths.task_name
    ):
        errors.append("paths.task_name must be one path component")
    if cfg.execution.stages is not None:
        try:
            parse_stages(cfg.execution.stages)
        except ValueError as exc:
            errors.append(f"execution.stages: {exc}")
    if cfg.agent.profile is not None and not str(cfg.agent.profile).strip():
        errors.append("agent.profile must be non-empty when provided")
    if cfg.agent.profile is not None and cfg.agent.name != "codex":
        errors.append("agent.profile is only supported when agent.name='codex'")
    if cfg.agent.profile is not None and not str(cfg.agent.model or "").strip():
        errors.append(PROFILE_MODEL_ERROR)
    if (
        cfg.agent.reasoning_effort is not None
        and cfg.agent.reasoning_effort not in VALID_REASONING_EFFORTS
    ):
        errors.append(f"agent.reasoning_effort must be one of {VALID_REASONING_EFFORTS}")
    if cfg.agent.name == "gemini" and cfg.agent.reasoning_effort is not None:
        errors.append(GEMINI_REASONING_EFFORT_ERROR)
    if cfg.logging.format not in {"human", "json"}:
        errors.append("logging.format must be one of ('human', 'json')")
    if type(cfg.logging.heartbeat_seconds) is not int:
        errors.append("logging.heartbeat_seconds must be an integer")
    elif cfg.logging.heartbeat_seconds < 0:
        errors.append("logging.heartbeat_seconds must be >= 0")
    if cfg.execution.max_ablations < 0:
        errors.append("execution.max_ablations must be >= 0")
    if cfg.execution.max_extra_baselines < 0:
        errors.append("execution.max_extra_baselines must be >= 0")
    if cfg.execution.max_evals_per_variant < 1:
        errors.append("execution.max_evals_per_variant must be >= 1")
    evaluation_timeout = cfg.execution.evaluation_timeout_seconds
    if evaluation_timeout is not None and (
        type(evaluation_timeout) is not int or evaluation_timeout < 1
    ):
        errors.append("execution.evaluation_timeout_seconds must be a positive integer or null")
    if cfg.execution.num_replications < 1:
        errors.append("execution.num_replications must be >= 1")
    if cfg.execution.sandbox_image is not None and not str(cfg.execution.sandbox_image).strip():
        errors.append("execution.sandbox_image must be non-empty when provided")

    if cfg.paths.baseline_program_override:
        from pathlib import Path

        override_path = Path(cfg.paths.baseline_program_override).expanduser()
        if not override_path.is_dir():
            errors.append(
                f"paths.baseline_program_override is not a directory: "
                f"{cfg.paths.baseline_program_override}"
            )

    for cb_path_str in cfg.paths.custom_baselines:
        from pathlib import Path

        cb_path = Path(cb_path_str).expanduser()
        if not (cb_path.is_file() or cb_path.is_dir()):
            errors.append(f"paths.custom_baselines: path not found: {cb_path_str}")

    for et_path_str in cfg.paths.extra_tasks:
        from pathlib import Path

        et_path = Path(et_path_str).expanduser()
        if not et_path.is_dir():
            errors.append(f"paths.extra_tasks: not a directory: {et_path_str}")

    return errors
