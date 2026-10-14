"""CLI entry point for the variants pipeline.

Usage::

    # Run the full pipeline:
    python -m src.variants_pipeline.cli \\
        paths.task_dir=/path/to/original/task \\
        paths.results_dir=/path/to/completed/results \\
        container.enabled=true

    # Run only specific stages:
    python -m src.variants_pipeline.cli --stages 0,1 ...   # extract + retrofit
    python -m src.variants_pipeline.cli --stages 2,3 ...   # generate + select
    python -m src.variants_pipeline.cli --stages 4 ...     # materialize only
    python -m src.variants_pipeline.cli --stages 5 ...     # execute only
    python -m src.variants_pipeline.cli --stages 0-3 ...   # extract through select

    # Inject custom baselines (always accepted, bypass generation/selection):
    python -m src.variants_pipeline.cli \\
        paths.custom_baselines='[/path/to/dfa.md,/path/to/bp.md]' ...

    # Run ONLY custom baselines (skip control_best and baseline_reference):
    python -m src.variants_pipeline.cli --custom-only \\
        paths.custom_baselines='[/path/to/dfa_folder]' ...

Stage numbers:
    0 = extract_bundle
    1 = retrofit_program
    2 = generate_variant_candidates
    3 = validate_and_select_variants
    4 = materialize_variant_workspaces
    5 = execute_variant_agents
    6 = cross_task_generalization
    7 = aggregate_results

Uses the Hydra Compose API (same pattern as ``src/investigation/config_loader.py``).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.final_evaluation.candidate_broker import CandidateBrokerInfrastructureError
from src.investigation.config import LoggingConfig
from src.investigation.runtime import (
    configure_event_logging,
    format_elapsed,
    log_event,
    shutdown_event_logging,
)

from .config import (
    ALL_STAGES,
    VariantsPipelineConfig,
    parse_stages,
    validate_variants_config,
)
from .pipeline import run_variants_pipeline

logger = logging.getLogger(__name__)

_CONFIGS_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs")

def _stages_override(spec: str) -> str:
    """Quote a stage spec so Hydra keeps ``0,1`` as one string, not a sweep."""
    escaped = spec.replace("\\", "\\\\").replace("'", "\\'")
    return f"execution.stages='{escaped}'"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if argv is None:
        argv = sys.argv[1:]

    # --stages / --custom-only are shorthand for the config keys of the same
    # name, so they become Hydra overrides rather than state carried beside the
    # config. Callers that already speak Hydra (the pipeline orchestrator) set
    # execution.stages / execution.custom_only directly.
    hydra_args: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--stages" and i + 1 < len(argv):
            hydra_args.append(_stages_override(argv[i + 1]))
            i += 2
        elif argv[i].startswith("--stages="):
            hydra_args.append(_stages_override(argv[i].split("=", 1)[1]))
            i += 1
        elif argv[i] == "--custom-only":
            hydra_args.append("execution.custom_only=true")
            i += 1
        elif argv[i] == "--log-format":
            if i + 1 >= len(argv):
                logger.error("--log-format requires human or json")
                return 1
            hydra_args.append(f"logging.format={argv[i + 1]}")
            i += 2
        elif argv[i].startswith("--log-format="):
            hydra_args.append(f"logging.format={argv[i].split('=', 1)[1]}")
            i += 1
        elif argv[i] == "--verbose":
            hydra_args.append("logging.verbose=true")
            i += 1
        elif argv[i] == "--heartbeat-seconds":
            if i + 1 >= len(argv):
                logger.error("--heartbeat-seconds requires a non-negative integer")
                return 1
            hydra_args.append(f"logging.heartbeat_seconds={argv[i + 1]}")
            i += 2
        elif argv[i].startswith("--heartbeat-seconds="):
            hydra_args.append(f"logging.heartbeat_seconds={argv[i].split('=', 1)[1]}")
            i += 1
        else:
            hydra_args.append(argv[i])
            i += 1

    cfg = load_variants_config(overrides=hydra_args)
    stages: set[int] | None = None
    custom_only = bool(cfg.execution.custom_only)

    errors = validate_variants_config(cfg)
    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
        return 1
    if cfg.execution.stages is not None:
        stages = parse_stages(cfg.execution.stages)

    logging.getLogger().setLevel(logging.INFO if cfg.logging.verbose else logging.WARNING)

    custom_baseline_paths: list[Path] | None = None
    if cfg.paths.custom_baselines:
        custom_baseline_paths = [Path(p).expanduser().resolve() for p in cfg.paths.custom_baselines]

    from .pipeline import _resolve_output_dir

    output_dir = _resolve_output_dir(cfg)
    configure_event_logging(
        log_format=cfg.logging.format,
        verbose=cfg.logging.verbose,
        heartbeat_seconds=cfg.logging.heartbeat_seconds,
        results_dir=output_dir,
        worker_index=0,
        source="variants",
        renderer="variants",
        journal_name="events.jsonl",
    )
    started_at = time.monotonic()
    selected_stages = sorted(stages if stages is not None else ALL_STAGES)
    log_event(
        "pipeline_start",
        task=cfg.paths.task_name or Path(cfg.paths.task_dir).name or "variants",
        run_id=cfg.run_id,
        phases=selected_stages,
        output_dir=output_dir,
        custom_only=custom_only,
        show_run_header=cfg.logging.show_run_header,
    )
    exit_code = 1
    try:
        exit_code = run_variants_pipeline(
            cfg,
            stages=stages,
            custom_baseline_paths=custom_baseline_paths,
            custom_only=custom_only,
            log_event_fn=log_event,
        )
        return exit_code
    except CandidateBrokerInfrastructureError as exc:
        logger.error("Variants infrastructure failure: %s", exc)
        exit_code = 4
        return exit_code
    except ValueError as exc:
        logger.error("Variants preflight failed: %s", exc)
        return 1
    finally:
        log_event(
            "pipeline_end",
            status="completed" if exit_code == 0 else "failed",
            exit_code=exit_code,
            elapsed=format_elapsed(time.monotonic() - started_at),
            stages=selected_stages,
        )
        shutdown_event_logging()


def load_variants_config(
    overrides: list[str] | None = None,
) -> VariantsPipelineConfig:
    """Load the variants pipeline config using Hydra Compose API."""
    with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base="1.3"):
        raw_cfg = compose(
            config_name="variants_config",
            overrides=overrides or [],
        )

    # Convert to structured config
    cfg_dict = OmegaConf.to_container(raw_cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    from src.investigation.config import AgentConfig, ContainerConfig
    from .config import VariantsExecutionConfig, VariantsPathsConfig

    agent_raw = cfg_dict.get("agent", {})
    paths_raw = cfg_dict.get("paths", {})
    execution_raw = cfg_dict.get("execution", {})
    container_raw = cfg_dict.get("container", {})
    logging_raw = cfg_dict.get("logging", {})

    return VariantsPipelineConfig(
        agent=AgentConfig(
            name=agent_raw.get("name", "claude"),
            model=agent_raw.get("model"),
            profile=agent_raw.get("profile"),
            reasoning_effort=agent_raw.get("reasoning_effort"),
        ),
        paths=VariantsPathsConfig(
            task_dir=str(paths_raw.get("task_dir", "")),
            task_name=(None if paths_raw.get("task_name") is None else str(paths_raw["task_name"])),
            results_dir=str(paths_raw.get("results_dir", "")),
            output_dir=paths_raw.get("output_dir"),
            baseline_program_override=paths_raw.get("baseline_program_override"),
            custom_baselines=[str(p) for p in (paths_raw.get("custom_baselines") or [])],
            extra_tasks=[str(p) for p in (paths_raw.get("extra_tasks") or [])],
        ),
        execution=VariantsExecutionConfig(
            stages=(
                None if execution_raw.get("stages") is None else str(execution_raw["stages"])
            ),
            custom_only=bool(execution_raw.get("custom_only", False)),
            max_ablations=execution_raw.get("max_ablations", 3),
            max_extra_baselines=execution_raw.get("max_extra_baselines", 3),
            max_evals_per_variant=execution_raw.get("max_evals_per_variant", 2),
            variant_timeout_seconds=execution_raw.get("variant_timeout_seconds", 7200),
            evaluation_timeout_seconds=execution_raw.get("evaluation_timeout_seconds"),
            variant_timeout_grace_seconds=execution_raw.get("variant_timeout_grace_seconds", 30),
            generation_timeout_seconds=execution_raw.get("generation_timeout_seconds", 1800),
            retrofit_timeout_seconds=execution_raw.get("retrofit_timeout_seconds", 1800),
            selection_timeout_seconds=execution_raw.get("selection_timeout_seconds", 1800),
            readme_timeout_seconds=execution_raw.get("readme_timeout_seconds", 600),
            num_replications=execution_raw.get("num_replications", 1),
            sandbox_image=(
                None
                if execution_raw.get("sandbox_image") is None
                else str(execution_raw["sandbox_image"])
            ),
            cross_task_timeout_seconds=execution_raw.get("cross_task_timeout_seconds", 7200),
        ),
        container=ContainerConfig(
            enabled=container_raw.get("enabled", True),
            image=container_raw.get("image", "agentic-scientist/agent-cli:latest"),
            network=container_raw.get("network", "bridge"),
            auth_root=str(container_raw.get("auth_root", "~")),
            env_file=str(container_raw.get("env_file", "")),
        ),
        logging=LoggingConfig(
            format=str(logging_raw.get("format", "human")),
            verbose=bool(logging_raw.get("verbose", False)),
            heartbeat_seconds=int(logging_raw.get("heartbeat_seconds", 60)),
            show_run_header=bool(logging_raw.get("show_run_header", True)),
        ),
        run_id=cfg_dict.get("run_id", ""),
    )


if __name__ == "__main__":
    raise SystemExit(main())
