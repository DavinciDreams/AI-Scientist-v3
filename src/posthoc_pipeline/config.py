"""Configuration dataclasses for the post-hoc pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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

VALID_CHECKPOINT_MODES = ("required", "optional")


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class PosthocPathsConfig:
    """Path configuration for the post-hoc pipeline."""

    task_dir: str = field(
        default="",
        metadata={"help": "Path to the original task directory"},
    )
    results_dir: str = field(
        default="",
        metadata={"help": "Path to the completed scientist run results directory"},
    )
    evidence_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional completed ablations bundle staged as evidence"},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for a single post-hoc pipeline run"},
    )


@dataclass
class PosthocExecutionConfig:
    """Execution settings for the post-hoc pipeline."""

    bootstrap_timeout_seconds: int = field(
        default=3600,
        metadata={"help": "Bootstrap stage timeout in seconds"},
    )
    bootstrap_max_attempts: int = field(
        default=3,
        metadata={"help": "Maximum bootstrap repair/rerun attempts before degrading"},
    )
    generation_timeout_seconds: int = field(
        default=1800,
        metadata={"help": "Analysis generation stage timeout in seconds"},
    )
    selection_timeout_seconds: int = field(
        default=1800,
        metadata={"help": "Analysis selection stage timeout in seconds"},
    )
    analysis_timeout_seconds: int = field(
        default=3600,
        metadata={"help": "Per-analysis execution timeout in seconds"},
    )
    analysis_timeout_grace_seconds: int = field(
        default=30,
        metadata={"help": "Grace period after bootstrap/generation/analysis timeout"},
    )
    max_selected_analyses: int = field(
        default=3,
        metadata={"help": "Maximum number of analyses to materialize and run"},
    )
    max_attempts_per_analysis: int = field(
        default=3,
        metadata={"help": "Maximum number of attempts per selected analysis"},
    )
    primary_metric_key: Optional[str] = field(
        default=None,
        metadata={"help": "Preferred primary metric key for bootstrap guardrails"},
    )
    primary_metric_tolerance: float = field(
        default=0.01,
        metadata={"help": "Allowed primary-metric regression during bootstrap"},
    )
    retrain_if_checkpoint_missing: bool = field(
        default=True,
        metadata={"help": "Whether bootstrap may rerun evaluation when no checkpoint exists"},
    )
    checkpoint_mode: str = field(
        default="required",
        metadata={"help": "Whether post-hoc analyses require a reusable checkpoint"},
    )


@dataclass
class PosthocPipelineConfig:
    """Top-level configuration for the post-hoc pipeline."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    paths: PosthocPathsConfig = field(default_factory=PosthocPathsConfig)
    execution: PosthocExecutionConfig = field(default_factory=PosthocExecutionConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    run_id: str = field(default_factory=_generate_run_id)


def validate_posthoc_config(cfg: PosthocPipelineConfig) -> list[str]:
    """Validate the config, returning a list of error messages."""
    errors: list[str] = []

    if not cfg.paths.task_dir:
        errors.append("paths.task_dir is required")
    if not cfg.paths.results_dir:
        errors.append("paths.results_dir is required")
    if cfg.paths.evidence_dir:
        evidence_dir = Path(cfg.paths.evidence_dir).expanduser()
        if evidence_dir.is_symlink() or not evidence_dir.is_dir():
            errors.append(f"paths.evidence_dir is not a real directory: {evidence_dir}")
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
        errors.append(
            f"logging.format must be one of ('human', 'json'), got {cfg.logging.format!r}"
        )
    if type(cfg.logging.heartbeat_seconds) is not int:
        errors.append("logging.heartbeat_seconds must be an integer")
    elif cfg.logging.heartbeat_seconds < 0:
        errors.append("logging.heartbeat_seconds must be >= 0")

    if cfg.execution.bootstrap_timeout_seconds <= 0:
        errors.append("execution.bootstrap_timeout_seconds must be > 0")
    if cfg.execution.bootstrap_max_attempts < 1:
        errors.append("execution.bootstrap_max_attempts must be >= 1")
    if cfg.execution.generation_timeout_seconds <= 0:
        errors.append("execution.generation_timeout_seconds must be > 0")
    if cfg.execution.selection_timeout_seconds <= 0:
        errors.append("execution.selection_timeout_seconds must be > 0")
    if cfg.execution.analysis_timeout_seconds <= 0:
        errors.append("execution.analysis_timeout_seconds must be > 0")
    if cfg.execution.analysis_timeout_grace_seconds <= 0:
        errors.append("execution.analysis_timeout_grace_seconds must be > 0")
    if cfg.execution.max_selected_analyses < 0:
        errors.append("execution.max_selected_analyses must be >= 0")
    if cfg.execution.max_attempts_per_analysis < 1:
        errors.append("execution.max_attempts_per_analysis must be >= 1")
    if (
        cfg.execution.primary_metric_key is not None
        and not cfg.execution.primary_metric_key.strip()
    ):
        errors.append("execution.primary_metric_key must be non-empty when provided")
    if cfg.execution.primary_metric_tolerance < 0:
        errors.append("execution.primary_metric_tolerance must be >= 0")
    if cfg.execution.checkpoint_mode not in VALID_CHECKPOINT_MODES:
        errors.append(f"execution.checkpoint_mode must be one of {VALID_CHECKPOINT_MODES}")

    return errors
