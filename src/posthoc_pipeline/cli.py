"""CLI entry point for the post-hoc pipeline.

Usage::

    python -m src.posthoc_pipeline.cli \
        paths.task_dir=/path/to/task \
        paths.results_dir=/path/to/results

    python -m src.posthoc_pipeline.cli --stages 0,1
    python -m src.posthoc_pipeline.cli --stages 2-4

Stage numbers:
    0 = extract_bundle
    1 = bootstrap_analysis_bundle
    2 = generate_analysis_candidates
    3 = validate_and_select_analyses
    4 = materialize_analysis_workspaces
    5 = execute_analysis_agents
    6 = aggregate_and_publish_valid_analyses
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.investigation.runtime import configure_event_logging, log_event, shutdown_event_logging

from .config import (
    PosthocExecutionConfig,
    PosthocPathsConfig,
    PosthocPipelineConfig,
    validate_posthoc_config,
)
from .pipeline import ALL_STAGES, _resolve_output_dir, run_posthoc_pipeline

logger = logging.getLogger(__name__)

_CONFIGS_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs")


def _parse_stages(stages_str: str) -> set[int]:
    result: set[int] = set()
    for part in stages_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            for stage in range(int(lo), int(hi) + 1):
                result.add(stage)
        else:
            result.add(int(part))

    invalid = result - ALL_STAGES
    if invalid:
        valid_text = f"{min(ALL_STAGES)}-{max(ALL_STAGES)}"
        raise ValueError(f"Invalid stage numbers: {invalid}. Valid: {valid_text}")
    return result


def load_posthoc_config(overrides: list[str] | None = None) -> PosthocPipelineConfig:
    """Load the post-hoc pipeline config using the Hydra Compose API."""
    with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base="1.3"):
        raw_cfg = compose(
            config_name="posthoc_config",
            overrides=overrides or [],
        )

    cfg_dict = OmegaConf.to_container(raw_cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    from src.investigation.config import AgentConfig, ContainerConfig, LoggingConfig
    from .config import _generate_run_id

    agent_raw = cfg_dict.get("agent", {})
    paths_raw = cfg_dict.get("paths", {})
    execution_raw = cfg_dict.get("execution", {})
    container_raw = cfg_dict.get("container", {})
    logging_raw = cfg_dict.get("logging", {})
    run_id = cfg_dict.get("run_id") or _generate_run_id()

    return PosthocPipelineConfig(
        agent=AgentConfig(
            name=agent_raw.get("name", "claude"),
            model=agent_raw.get("model"),
            profile=agent_raw.get("profile"),
            reasoning_effort=agent_raw.get("reasoning_effort"),
        ),
        paths=PosthocPathsConfig(
            task_dir=str(paths_raw.get("task_dir", "")),
            results_dir=str(paths_raw.get("results_dir", "")),
            evidence_dir=paths_raw.get("evidence_dir"),
            output_dir=paths_raw.get("output_dir"),
        ),
        execution=PosthocExecutionConfig(
            bootstrap_timeout_seconds=execution_raw.get("bootstrap_timeout_seconds", 3600),
            bootstrap_max_attempts=execution_raw.get("bootstrap_max_attempts", 3),
            generation_timeout_seconds=execution_raw.get("generation_timeout_seconds", 1800),
            selection_timeout_seconds=execution_raw.get("selection_timeout_seconds", 1800),
            analysis_timeout_seconds=execution_raw.get("analysis_timeout_seconds", 3600),
            analysis_timeout_grace_seconds=execution_raw.get("analysis_timeout_grace_seconds", 30),
            max_selected_analyses=execution_raw.get("max_selected_analyses", 3),
            max_attempts_per_analysis=execution_raw.get("max_attempts_per_analysis", 3),
            primary_metric_key=execution_raw.get("primary_metric_key"),
            primary_metric_tolerance=execution_raw.get("primary_metric_tolerance", 0.01),
            retrain_if_checkpoint_missing=execution_raw.get("retrain_if_checkpoint_missing", True),
            checkpoint_mode=execution_raw.get("checkpoint_mode", "required"),
        ),
        container=ContainerConfig(
            enabled=container_raw.get("enabled", False),
            image=container_raw.get("image", "agentic-scientist/agent-cli:latest"),
            network=container_raw.get("network", "bridge"),
            auth_root=str(container_raw.get("auth_root", "~")),
            env_file=str(container_raw.get("env_file", "")),
        ),
        logging=LoggingConfig(
            format=str(logging_raw.get("format", "human")),
            verbose=bool(logging_raw.get("verbose", False)),
            heartbeat_seconds=logging_raw.get("heartbeat_seconds", 60),
            show_run_header=bool(logging_raw.get("show_run_header", True)),
        ),
        run_id=str(run_id),
    )


def _parse_cli_arguments(argv: list[str]) -> tuple[set[int] | None, list[str]]:
    """Split convenience flags from Hydra overrides."""
    stages: set[int] | None = None
    hydra_args: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--stages":
            if index + 1 >= len(argv):
                raise ValueError("--stages requires a stage list")
            stages = _parse_stages(argv[index + 1])
            index += 2
        elif argument.startswith("--stages="):
            stages = _parse_stages(argument.split("=", 1)[1])
            index += 1
        elif argument == "--log-format":
            if index + 1 >= len(argv):
                raise ValueError("--log-format requires human or json")
            hydra_args.append(f"logging.format={argv[index + 1]}")
            index += 2
        elif argument.startswith("--log-format="):
            hydra_args.append(f"logging.format={argument.split('=', 1)[1]}")
            index += 1
        elif argument == "--heartbeat-seconds":
            if index + 1 >= len(argv):
                raise ValueError("--heartbeat-seconds requires a non-negative integer")
            hydra_args.append(f"logging.heartbeat_seconds={argv[index + 1]}")
            index += 2
        elif argument.startswith("--heartbeat-seconds="):
            hydra_args.append(f"logging.heartbeat_seconds={argument.split('=', 1)[1]}")
            index += 1
        elif argument == "--verbose":
            hydra_args.append("logging.verbose=true")
            index += 1
        else:
            hydra_args.append(argument)
            index += 1
    return stages, hydra_args


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if argv is None:
        argv = sys.argv[1:]

    try:
        stages, hydra_args = _parse_cli_arguments(argv)
    except ValueError as exc:
        logger.error("Argument error: %s", exc)
        return 1

    cfg = load_posthoc_config(overrides=hydra_args)

    errors = validate_posthoc_config(cfg)
    if errors:
        for error in errors:
            logger.error("Config error: %s", error)
        return 1

    logging.getLogger().setLevel(logging.INFO if cfg.logging.verbose else logging.WARNING)

    if stages is not None:
        logger.info("Starting post-hoc pipeline (stages: %s)", sorted(stages))
    else:
        logger.info("Starting post-hoc pipeline (all stages)")

    configure_event_logging(
        log_format=cfg.logging.format,
        verbose=cfg.logging.verbose,
        heartbeat_seconds=cfg.logging.heartbeat_seconds,
        results_dir=_resolve_output_dir(cfg),
        worker_index=0,
        source="posthoc",
        renderer="posthoc",
        journal_name="events.jsonl",
    )
    try:
        return run_posthoc_pipeline(cfg, stages=stages)
    except KeyboardInterrupt:
        log_event("pipeline_end", status="interrupted", return_code=130)
        return 130
    except Exception as exc:
        log_event(
            "pipeline_end",
            status="failed",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        shutdown_event_logging()


if __name__ == "__main__":
    raise SystemExit(main())
