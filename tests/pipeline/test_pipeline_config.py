"""Pipeline config loading/validation, investigation config loading, resume policy, paper.sh mirror."""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest
import yaml
from hydra.errors import ConfigCompositionException

from src.investigation.config import (
    AgentConfig,
    ConfigValidationError,
    ContainerConfig,
    ExecutionConfig,
    InvestigationConfig,
    LoggingConfig,
    PathsConfig,
    PaperConfig as InvestigationPaperConfig,
    config_to_flat_dict,
    has_hydra_override,
    should_use_harness_instruction,
)
from src.investigation.config_loader import _dict_to_config, load_config
from src.investigation.parallel_helpers import build_parallel_worker_command
from src.pipeline import portable_publication
from src.pipeline.cli import (
    _normalize_logging_flags,
    _recover_pending_revision_publication,
    config_error_message,
    load_pipeline_config,
)
from src.pipeline.config import (
    PAPER_OWNED_KEYS,
    PIPELINE_PHASES,
    PaperConfig,
    PipelineConfig,
    describe_pipeline_config,
    flatten_mapping,
    parse_pipeline_phases,
    stage_baseline,
    validate_investigation_resume_state,
    validate_pipeline_config,
)
from src.pipeline.paths import derive_paths
from src.pipeline.portable_publication import replace_directory_portably
from src.pipeline.resume_config import (
    INVESTIGATION_CLAIM_FILENAME,
    ResumeConfigError,
    claim_task_handoff_for_investigation,
    publish_task_handoff,
    read_idea_cycle_budget,
    read_run_idea_cycle_budget,
    write_resume_policy,
)
from src.posthoc_pipeline.cli import load_posthoc_config
from src.variants_pipeline.cli import load_variants_config
from tests.paper_script_support import (
    PAPER_SCRIPT,
    REPO_ROOT,
    environment_backed_vars,
    flag_arms,
    script_defaults,
)
from tests.revision_test_support import write_workspace

UNKNOWN_HOME = "~agentic_scientist_user_that_does_not_exist"
PDF_BYTES = b"%PDF-1.4\n"


def _set(cfg: Any, key: str, value: object) -> None:
    head, dot, tail = key.partition(".")
    setattr(getattr(cfg, head) if dot else cfg, tail if dot else head, value)


def _get(cfg: Any, key: str) -> object:
    node = cfg
    for part in key.split("."):
        node = getattr(node, part)
    return node


@functools.cache
def _base_cfg() -> PipelineConfig:
    """The composed circpack config, loaded once; callers deep-copy before mutating."""
    return load_pipeline_config(["task.dir=examples/circpack"])


def _valid_cfg(**overrides: object) -> PipelineConfig:
    cfg = copy.deepcopy(_base_cfg())
    for key, value in overrides.items():
        _set(cfg, key, value)
    return cfg


def _config_errors(overrides: list[str]) -> list[str]:
    """Load-time schema errors or, when the config loads, validator errors."""
    try:
        cfg = load_pipeline_config(overrides)
    except ConfigCompositionException as exc:
        return [config_error_message(exc)]
    return validate_pipeline_config(cfg)


def _write_investigation_resume_state(results_dir: Path) -> None:
    program_dir = results_dir / "best_program"
    entrypoint = program_dir / "src" / "program.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("", encoding="utf-8")
    (results_dir / "task").mkdir()
    (results_dir / "outer_state.json").write_text('{"current_cycle": 1}', encoding="utf-8")
    (results_dir / "best_program_manifest.json").write_text(
        json.dumps(
            {
                "program_dir": str(program_dir),
                "entrypoint_path": "src/program.py",
                "combined_score": 1.0,
                "correct": True,
            }
        ),
        encoding="utf-8",
    )


def _publish_paper(
    paper_dir: Path,
    *,
    pdf: bool = True,
    manifest_pdf: str | None = None,
    before_revision: dict[str, str] | None = None,
) -> Path:
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper_pdf = paper_dir / "paper.pdf"
    if pdf:
        paper_pdf.write_bytes(PDF_BYTES)
    manifest: dict[str, object] = {"paper_pdf": manifest_pdf or str(paper_pdf.resolve())}
    if before_revision is not None:
        manifest["before_revision"] = before_revision
    (paper_dir / "latest_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return paper_pdf


def _anchored_cfg(phases: str, run_dir: Path, paper_dir: Path) -> PipelineConfig:
    return _valid_cfg(
        **{
            "execution.phases": phases,
            "paths.run_dir": str(run_dir),
            "paths.paper_dir": str(paper_dir),
        }
    )


# --- pipeline config: loading ---------------------------------------------------


def test_load_pipeline_config_defaults_match_dataclass_and_yaml() -> None:
    # Composed defaults carry every documented value.
    cfg = load_pipeline_config()

    assert cfg.logging.format == "human"
    assert cfg.logging.verbose is False
    assert cfg.logging.heartbeat_seconds == 60
    assert cfg.agent.name == "claude"
    assert cfg.agent.model is None
    assert cfg.agent.profile is None
    assert cfg.agent.reasoning_effort is None
    assert cfg.ablations.num_replications == 5
    assert cfg.ablations.max_evals_per_variant == 5
    assert cfg.investigation.resume is False
    assert cfg.investigation.draft_task_if_missing is False
    assert cfg.investigation.gpu_evaluation is False
    assert cfg.paper.format == "iclr"
    assert cfg.paper.min_refs == 25
    assert cfg.paper.max_refs == 30
    assert cfg.review.timeout_seconds == 1800
    assert cfg.review.timeout_grace_seconds == 30
    assert cfg.final.timeout_seconds == 21600
    assert cfg.execution.phases == "task_drafting,investigation,ablations,final_evaluation,posthoc,paper"
    assert cfg.posthoc.max_selected_analyses == 3
    assert cfg.container.enabled is True
    assert cfg.task_drafting.timeout_seconds == 57600
    expected_task_drafting = {
        "workflow": "robust",
        "design_max_repairs": 8,
        "contract_max_repairs": 3,
        "smoke_max_repairs": 3,
        "smoke_max_infrastructure_attempts": 3,
        "alignment_max_repairs": 10,
        "qualification_max_repairs": 4,
        "agent_transient_max_attempts": 4,
        "qualification_max_infrastructure_attempts": 3,
        "production_entry_timeout_seconds": 300,
        "max_gate_checks_per_repair": 12,
        "max_runtime_bytes": 16 * 1024 * 1024 * 1024,
    }
    for key, value in expected_task_drafting.items():
        assert cfg.investigation.get(f"task_drafting.{key}") == value, key
    # Versioning block is declared in the YAML (load raises when missing entirely).
    assert cfg.versioning.enabled is True
    assert cfg.versioning.git_dir_name == ".agentic_history"
    assert cfg.versioning.max_file_bytes == 0
    assert cfg.versioning.max_total_bytes == 16 * 1024**3
    assert cfg.versioning.final_include_environments is False
    assert cfg.versioning.finalize_on_exit is True

    # The YAML restates nothing: composing it with no overrides is the dataclass.
    assert load_pipeline_config([]) == PipelineConfig()

    # A default-constructed config matches the composed phase defaults.
    default = PipelineConfig()
    assert default.investigation.max_idea_cycles == 3
    assert default.investigation.workers == 1
    assert default.ablations.num_replications == 5
    assert default.posthoc.container_network == "bridge"
    assert default.paper.format == "iclr"
    assert default.paper.standalone is True
    assert default.investigation.forwarded == {
        "execution.max_idea_cycles": 3,
        "execution.max_program_evals_per_attempt": 1,
    }


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param(
            ["logging.format=json", "logging.verbose=true", "logging.heartbeat_seconds=15"],
            {"logging.format": "json", "logging.verbose": True, "logging.heartbeat_seconds": 15},
            id="logging",
        ),
        pytest.param(
            [
                "ablations.execution.num_replications=2",
                "investigation.execution.resume=true",
                "investigation.execution.draft_task_if_missing=true",
                "paper.paper_format=neurips",
                "execution.phases=paper",
            ],
            {
                "ablations.num_replications": 2,
                "investigation.resume": True,
                "investigation.draft_task_if_missing": True,
                "paper.format": "neurips",
                "execution.phases": "paper",
            },
            id="stage-overrides",
        ),
        pytest.param(
            ["agent.name=codex", "agent.profile=custom", "agent.model=custom-model"],
            {"agent.profile": "custom"},
            id="agent-profile",
        ),
        pytest.param(
            ["agent.reasoning_effort=xhigh"], {"agent.reasoning_effort": "xhigh"}, id="reasoning-effort"
        ),
        pytest.param(
            ["review.timeout_seconds=900", "review.timeout_grace_seconds=15"],
            {"review.timeout_seconds": 900, "review.timeout_grace_seconds": 15},
            id="review-timeouts",
        ),
        pytest.param(
            [
                "investigation.execution.max_idea_generation_attempts=1",
                "investigation.execution.idea_timeout_seconds=600",
                "investigation.execution.run_timeout_seconds=600",
            ],
            {
                "investigation.max_idea_generation_attempts": 1,
                "investigation.idea_timeout_seconds": 600,
                "investigation.run_timeout_seconds": 600,
            },
            id="investigation-cli-bounds",
        ),
        pytest.param(
            ["investigation.execution.harness_evaluated_attempts=true"],
            {"investigation.harness_evaluated_attempts": True},
            id="harness-evaluated-attempts",
        ),
        pytest.param(
            [
                "investigation.execution.harness_evaluated_attempts=true",
                "investigation.execution.gpu_evaluation=true",
            ],
            {"investigation.gpu_evaluation": True},
            id="gpu-evaluation",
        ),
        pytest.param(
            ["agent.name=codex", "agent.model=456", "agent.profile=123"],
            {"agent.model": "456", "agent.profile": "123"},
            id="numeric-agent-profile-normalized",
        ),
        pytest.param(
            ["container.image=123", "container.network=456"],
            {"container.image": "123", "container.network": "456"},
            id="container-scalars-normalized",
        ),
        pytest.param(
            [
                "paths.run_dir=123",
                "paths.ablations_dir=456",
                "paths.posthoc_dir=654",
                "paths.paper_dir=789",
                "run_id=20260101",
            ],
            {
                "paths.run_dir": "123",
                "paths.ablations_dir": "456",
                "paths.posthoc_dir": "654",
                "paths.paper_dir": "789",
                "run_id": "20260101",
            },
            id="optional-string-scalars-normalized",
        ),
        pytest.param(["container.env_file=''"], {"container.env_file": ""}, id="empty-env-file"),
    ],
)
def test_load_pipeline_config_overrides(overrides: list[str], expected: dict[str, object]) -> None:
    cfg = load_pipeline_config(overrides)

    assert {key: _get(cfg, key) for key in expected} == expected


@pytest.mark.parametrize(
    ("override", "match", "message"),
    [
        ("paths.investigation_results_dir=/tmp/old-run", None, None),
        ("container.env_file=null", "container.env_file", None),
        ("investigation.max_idea_cyles=1", "investigation.max_idea_cyles", None),
        ("task.generation_timeout_seconds=5", "task.generation_timeout_seconds", None),
        ("final.timeout_seconds=abc", "final.timeout_seconds", "could not be converted to Integer"),
        ("investigation.execution.max_idea_cyles=1", "investigation.execution.max_idea_cyles", None),
        ("preset=docker", None, None),
        ("investigation/preset=docker", None, None),
        ("investigation.max_idea_cycles=1", "investigation.max_idea_cycles", None),
        ("task_drafting.design_max_repairs=1", "task_drafting.design_max_repairs", None),
        ("ablations.steps=0-3", "ablations.steps", None),
        ("posthoc.container_network=host", "posthoc.container_network", None),
        ("paper.format=iclr", "paper.format", None),
        ("paper.standalone=false", "paper.standalone", None),
    ],
)
def test_load_pipeline_config_rejects_unknown_or_malformed_keys(
    override: str, match: str | None, message: str | None
) -> None:
    with pytest.raises(ConfigCompositionException, match=match) as excinfo:
        load_pipeline_config([override])
    if message is not None:
        assert message in config_error_message(excinfo.value)


BOOLEAN_KEYS = (
    "investigation.execution.resume",
    "investigation.execution.draft_task_if_missing",
    "ablations.execution.custom_only",
    "paper.standalone_paper",
    "container.enabled",
    "execution.dry_run",
    "execution.fail_fast",
)


def test_load_pipeline_config_rejects_appended_keys_and_invalid_scalars() -> None:
    # Appended (+/++) keys are rejected.
    for override in ("+preset=docker", "+investigation.execution.extra=1", "++foo=1"):
        with pytest.raises(ValueError, match="cannot be appended"):
            load_pipeline_config([override])

    # A stage key with the wrong scalar type is a validation error.
    errors = _config_errors(["task.dir=examples/circpack", "investigation.execution.max_idea_cycles=abc"])
    assert "investigation.execution.max_idea_cycles must be an integer" in errors

    # Invalid boolean words: schema keys fail at load; composed stage keys fail validation.
    for key in BOOLEAN_KEYS:
        errors = _config_errors(["task.dir=examples/circpack", f"{key}='sometimes'"])
        assert any(
            key in error and ("not a valid bool" in error or "must be a boolean" in error)
            for error in errors
        ), key


def test_load_pipeline_config_coerces_boolean_words_and_logging_flags(monkeypatch) -> None:
    # Interpolated "false" parses as a boolean.
    monkeypatch.setenv("PIPELINE_FALSE", "false")
    false_value = "${oc.env:PIPELINE_FALSE}"
    cfg = load_pipeline_config([f"{key}={false_value}" for key in BOOLEAN_KEYS])

    assert cfg.investigation.resume is False
    assert cfg.investigation.draft_task_if_missing is False
    assert cfg.ablations.custom_only is False
    assert cfg.paper.standalone is False
    assert cfg.container.enabled is False
    assert cfg.execution.dry_run is False
    assert cfg.execution.fail_fast is False

    # OmegaConf boolean words ("yes") are accepted.
    for key in ("investigation.execution.resume", "ablations.execution.custom_only", "container.enabled"):
        cfg = load_pipeline_config([f"{key}=yes"])
        head, _, tail = key.partition(".")
        section = getattr(cfg, head)
        value = section.get(tail) if hasattr(section, "get") else getattr(section, tail)
        assert value is True, key

    # Logging convenience flags normalize to overrides.
    assert _normalize_logging_flags(
        ["--log-format", "json", "--verbose", "--heartbeat-seconds=15"]
    ) == ["logging.format=json", "logging.verbose=true", "logging.heartbeat_seconds=15"]


def test_describe_pipeline_config_and_stage_baseline_cover_every_key() -> None:
    # describe_pipeline_config lists every leaf key.
    rows = {row.key: row for row in describe_pipeline_config()}

    assert rows["task.dir"].default == "examples/circpack"
    assert rows["container.enabled"].default == "True"
    assert rows["container.network"].default == "host"
    assert rows["execution.phases"].type_name == "str"
    assert rows["paths.run_dir"].default == "null"
    assert rows["run_id"].help
    assert "task.generation_timeout_seconds" not in rows
    assert rows["investigation.execution.isolate_worker_memory"].default == "False"
    assert "stage default: 20" in rows["investigation.execution.max_idea_cycles"].help
    assert "set by the pipeline" in rows["investigation.paths.task_dir"].help
    assert rows["paper.paper_review_rounds"].default == "0"

    # stage_baseline matches each stage loader (a stage adding a key unnoticed would break forwarding).
    variants = flatten_mapping(dataclasses.asdict(load_variants_config([])))
    assert variants == flatten_mapping(dataclasses.asdict(load_variants_config([]))) | stage_baseline(
        "variants_config"
    )

    posthoc = flatten_mapping(dataclasses.asdict(load_posthoc_config([])))
    posthoc_baseline = stage_baseline("posthoc_config")
    posthoc_baseline.pop("run_id")  # the loader mints one when null
    assert {key: posthoc[key] for key in posthoc_baseline} == posthoc_baseline

    investigation = config_to_flat_dict(load_config(["paths.task_dir=examples/circpack"]))
    investigation_baseline = stage_baseline("config")
    for derived in ("paths.task_dir", "paths.results_dir", "paths.global_memory_dir", "run_id"):
        investigation_baseline.pop(derived, None)
    for key, value in investigation_baseline.items():
        if isinstance(value, (list, dict)):
            continue
        assert investigation[key] == ("null" if value is None else str(value)), key


# --- pipeline config: phases ----------------------------------------------------


PHASE_CASES = [
    (
        "investigation,ablations,posthoc,paper,review",
        {"investigation", "ablations", "posthoc", "paper", "review"},
    ),
    ("posthoc,paper", {"posthoc", "paper"}),
    ("ablations,paper", {"ablations", "paper"}),
    ("paper", {"paper"}),
    ("review", {"review"}),
    ("revision", {"revision"}),
    ("investigation,paper", {"investigation", "paper"}),
    ("task", {"task_drafting"}),
]


def test_parse_pipeline_phases_and_phase_order() -> None:
    for spec, expected in PHASE_CASES:
        assert parse_pipeline_phases(spec) == expected, spec
    for spec in ("", "1", "variants", "investigation,unknown"):
        with pytest.raises(ValueError):
            parse_pipeline_phases(spec)
    assert PIPELINE_PHASES == (
        "task_drafting",
        "investigation",
        "ablations",
        "final_evaluation",
        "posthoc",
        "paper",
        "review",
        "revision",
    )


# --- pipeline config: validation ------------------------------------------------


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("investigation.task_drafting.workflow=unknown", "workflow must be robust or legacy"),
        ("investigation.task_drafting.design_max_repairs=9", "design_max_repairs"),
        ("investigation.task_drafting.contract_max_repairs=4", "contract_max_repairs"),
        ("investigation.task_drafting.smoke_max_repairs=4", "smoke_max_repairs"),
        ("investigation.task_drafting.smoke_max_infrastructure_attempts=4", "smoke_max_infrastructure_attempts"),
        ("investigation.task_drafting.smoke_max_infrastructure_attempts=true", "smoke_max_infrastructure_attempts"),
        ("investigation.task_drafting.smoke_max_infrastructure_attempts=1.5", "smoke_max_infrastructure_attempts"),
        ("investigation.task_drafting.alignment_max_repairs=11", "alignment_max_repairs"),
        ("investigation.task_drafting.qualification_max_repairs=9", "qualification_max_repairs"),
        ("investigation.task_drafting.agent_transient_max_attempts=5", "agent_transient_max_attempts"),
        (
            "investigation.task_drafting.qualification_max_infrastructure_attempts=4",
            "qualification_max_infrastructure_attempts",
        ),
        ("investigation.task_drafting.production_entry_timeout_seconds=29", "production_entry_timeout_seconds"),
        ("investigation.task_drafting.max_gate_checks_per_repair=0", "max_gate_checks_per_repair"),
        ("investigation.task_drafting.max_runtime_bytes=0", "max_runtime_bytes"),
        ("investigation.task_drafting.max_runtime_bytes=68719476737", "max_runtime_bytes"),
        ("investigation.task_drafting.max_runtime_bytes=true", "max_runtime_bytes"),
        ("investigation.task_drafting.max_runtime_bytes=1.5", "max_runtime_bytes"),
    ],
)
def test_task_drafting_limits_are_validated(override: str, expected: str) -> None:
    assert any(expected in error for error in _config_errors(["task.dir=examples/circpack", override]))


def test_task_drafting_boundary_values_and_versioning_limits() -> None:
    # Boundary values of task_drafting limits are accepted.
    for override, forbidden in (
        ("investigation.task_drafting.design_max_repairs=8", "design_max_repairs"),
        ("investigation.task_drafting.alignment_max_repairs=10", "alignment_max_repairs"),
        ("investigation.task_drafting.contract_max_repairs=3", None),
    ):
        errors = validate_pipeline_config(load_pipeline_config(["task.dir=examples/circpack", override]))
        assert errors == [] if forbidden is None else not any(forbidden in error for error in errors), override

    # Versioning limits are validated.
    for override, message in (
        ("versioning.git_dir_name=../escape", "one non-empty path component"),
        ("versioning.git_dir_name='unsafe name'", "letters, digits, dot, underscore, or hyphen"),
        ("versioning.git_dir_name='unsafe*name'", "letters, digits, dot, underscore, or hyphen"),
        ("versioning.max_file_bytes=-1", "versioning.max_file_bytes must be >= 0"),
        ("versioning.max_total_bytes=-1", "versioning.max_total_bytes must be >= 0"),
        ("versioning.timeout_seconds=0", "versioning.timeout_seconds must be > 0"),
    ):
        cfg = load_pipeline_config(["task.dir=examples/circpack", override])
        assert any(message in error for error in validate_pipeline_config(cfg)), override


ACCEPTED_CASES: list[tuple[str, dict[str, object]]] = [
    ("defaults", {}),
    ("legacy-task-alias", {"execution.phases": "task", "investigation.draft_task_if_missing": False}),
    (
        "task-drafting-allows-parallel-workers",
        {"investigation.draft_task_if_missing": True, "investigation.workers": 2},
    ),
    (
        "disabled-container-needs-no-auth-root",
        {"execution.phases": "investigation", "container.enabled": False, "container.auth_root": ""},
    ),
    ("empty-venue-file", {"paper.venue_file": ""}),
    ("alife-format", {"paper.format": "alife"}),
    (
        "gpu-evaluation-with-gpus",
        {"investigation.harness_evaluated_attempts": True, "investigation.gpu_evaluation": True, "container.gpu_count": 2},
    ),
]


def test_validate_accepts(tmp_path: Path) -> None:
    for name, overrides in ACCEPTED_CASES:
        assert validate_pipeline_config(_valid_cfg(**overrides)) == [], name

    # An absolute, existing venue file is accepted.
    venue_file = tmp_path / "venue.md"
    venue_file.write_text("venue", encoding="utf-8")
    assert validate_pipeline_config(_valid_cfg(**{"paper.venue_file": str(venue_file)})) == []


EXACT_ERROR_CASES: list[tuple[str, Mapping[str, object], str]] = [
    (
        "gpu-without-harness",
        {"investigation.gpu_evaluation": True},
        "investigation.execution.gpu_evaluation requires investigation.execution.harness_evaluated_attempts=true",
    ),
    (
        "gpu-without-gpus",
        {"investigation.harness_evaluated_attempts": True, "investigation.gpu_evaluation": True},
        "investigation.execution.gpu_evaluation requires container.gpu_count >= 1",
    ),
    *[
        (f"gpu-count-{value!r}", {"container.gpu_count": value}, "container.gpu_count must be a non-negative integer")
        for value in (-1, True, "2", 1.5)
    ],
    ("empty-profile", {"agent.name": "codex", "agent.profile": "  "}, "agent.profile must be non-empty when provided"),
    (
        "profile-for-non-codex",
        {"agent.name": "claude", "agent.profile": "custom"},
        "agent.profile is only supported when agent.name='codex'",
    ),
    (
        "profile-without-model",
        {"agent.name": "codex", "agent.profile": "custom"},
        "agent.model is required when agent.profile is provided",
    ),
    (
        "profile-with-blank-model",
        {"agent.name": "codex", "agent.model": "   ", "agent.profile": "custom"},
        "agent.model is required when agent.profile is provided",
    ),
    *[
        (f"ablations-{field}-{value!r}", {f"ablations.{field}": value}, f"ablations.execution.{field} must be an integer")
        for field, value in [
            ("num_replications", 1.5),
            ("num_replications", True),
            ("num_replications", "5"),
            ("max_evals_per_variant", 2.5),
            ("max_evals_per_variant", False),
            ("max_evals_per_variant", "5"),
        ]
    ],
    *[
        (
            f"investigation-{field}-{value!r}",
            {f"investigation.{field}": value},
            f"investigation.execution.{field} must be an integer",
        )
        for field, value in [
            ("max_idea_cycles", 1.5),
            ("max_idea_cycles", True),
            ("max_idea_cycles", "3"),
            ("workers", 1.5),
            ("workers", False),
            ("workers", "2"),
            ("prune_every_n_cycles", 0.5),
            ("prune_every_n_cycles", True),
            ("prune_every_n_cycles", "0"),
            ("max_program_evals_per_attempt", 1.5),
            ("max_program_evals_per_attempt", True),
            ("max_program_evals_per_attempt", "1"),
            ("max_idea_generation_attempts", 1.5),
            ("max_idea_generation_attempts", True),
            ("max_idea_generation_attempts", "1"),
            ("idea_timeout_seconds", 600.5),
            ("idea_timeout_seconds", True),
            ("idea_timeout_seconds", "600"),
            ("run_timeout_seconds", 600.5),
            ("run_timeout_seconds", True),
            ("run_timeout_seconds", "600"),
        ]
    ],
]


def test_validate_reports_exact_error() -> None:
    for name, overrides, message in EXACT_ERROR_CASES:
        assert message in validate_pipeline_config(_valid_cfg(**overrides)), name


ERROR_FRAGMENT_CASES: list[tuple[str, Mapping[str, object], str]] = [
    ("bad-agent", {"agent.name": "bogus"}, "agent.name"),
    ("gemini-reasoning-effort", {"agent.name": "gemini", "agent.reasoning_effort": "high"}, "not supported for gemini"),
    *[
        (f"run-id-{run_id!r}", {"run_id": run_id}, "run_id must be one non-empty path component")
        for run_id in ("", ".", "..", "/tmp/escape", "nested/run", "a\\b")
    ],
    ("missing-task-dir", {"task.dir": "examples/does_not_exist"}, "task.dir"),
    ("task-dir-unknown-home", {"task.dir": f"{UNKNOWN_HOME}/task"}, "task.dir could not be expanded"),
    *[
        (f"task-dir-{task_dir!r}", {"task.dir": task_dir}, "task.dir must identify one named directory")
        for task_dir in (".", "..", "examples/circpack/..", "/")
    ],
    *[
        (f"paths-{field}-unknown-home", {f"paths.{field}": f"{UNKNOWN_HOME}/output"}, f"paths.{field} could not be expanded")
        for field in ("results_root", "run_dir", "ablations_dir", "paper_dir")
    ],
    *[(f"auth-root-{root!r}", {"container.auth_root": root}, "container.auth_root") for root in ("", "/no/such/auth/root")],
    ("auth-root-unknown-home", {"container.auth_root": UNKNOWN_HOME}, "could not be expanded"),
    ("env-file-unknown-home", {"container.env_file": f"{UNKNOWN_HOME}/.env"}, "container.env_file could not be expanded"),
    *[(f"paper-format-{value!r}", {"paper.format": value}, "paper.paper_format") for value in ("", "neurips")],
    *[(f"paper-min-refs-{value!r}", {"paper.min_refs": value}, "paper.paper_min_refs") for value in (0, -1, 1.5, True, "25")],
    *[(f"paper-max-refs-{value!r}", {"paper.max_refs": value}, "paper.paper_max_refs") for value in (30.5, False, "30")],
    ("paper-min-gt-max", {"paper.min_refs": 40}, "min_refs"),
    ("paper-venue-missing", {"paper.venue_file": "missing-venue.md"}, "paper.venue_file"),
    *[
        (f"review-{field}-zero", {"execution.phases": "review", f"review.{field}": 0}, f"review.{field}")
        for field in ("timeout_seconds", "timeout_grace_seconds")
    ],
    ("ablations-without-anchor", {"execution.phases": "ablations,paper"}, "paths.run_dir"),
    ("paper-without-anchor", {"execution.phases": "paper"}, "paths.run_dir"),
    (
        "paper-anchor-missing",
        {"execution.phases": "paper", "paths.run_dir": "/no/such/dir/anywhere"},
        "paths.run_dir is not a directory",
    ),
    ("paper-revision-without-review", {"execution.phases": "paper,revision"}, "paper and revision requires review"),
    *[(f"review-only-needs-{key}", {"execution.phases": "review"}, key) for key in ("paths.run_dir", "paths.paper_dir")],
    *[
        (f"resume-anchor-{anchor!r}", {"investigation.resume": True, "paths.run_dir": anchor}, "paths.run_dir")
        for anchor in (None, "/no/such/investigation/run")
    ],
]


def test_validate_reports_error_mentioning() -> None:
    for name, overrides, fragment in ERROR_FRAGMENT_CASES:
        assert any(fragment in error for error in validate_pipeline_config(_valid_cfg(**overrides))), name


def test_validate_auth_root_and_task_dir_check_real_directories(monkeypatch, tmp_path: Path) -> None:
    # container.auth_root rejects a regular file and a symlink.
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}", encoding="utf-8")
    (tmp_path / "auth").mkdir()
    auth_link = tmp_path / "auth-link"
    auth_link.symlink_to(tmp_path / "auth", target_is_directory=True)
    for auth_root in (auth_file, auth_link):
        errors = validate_pipeline_config(_valid_cfg(**{"container.auth_root": str(auth_root)}))
        assert any("container.auth_root" in error for error in errors), auth_root

    # task.dir checks the expanded basename.
    _base_cfg()  # load before HOME changes
    monkeypatch.setenv("HOME", "/")
    errors = validate_pipeline_config(_valid_cfg(**{"task.dir": "~"}))
    assert any("task.dir must identify one named directory" in error for error in errors)


UNSELECTED_PHASE_CASES: list[tuple[str, str, dict[str, object], bool, str | None]] = [
    (
        "ablations",
        "investigation",
        {"ablations.num_replications": "invalid", "ablations.max_evals_per_variant": False},
        False,
        None,
    ),
    (
        "paper",
        "investigation",
        {"paper.format": "neurips", "paper.min_refs": 0, "paper.max_refs": -1, "paper.venue_file": "missing-venue.md"},
        False,
        None,
    ),
    (
        "posthoc",
        "paper",
        {"posthoc.max_selected_analyses": "invalid", "paths.ablations_dir": "/tmp/nonexistent-paper-anchor"},
        False,
        "posthoc.",
    ),
    (
        "investigation-counts",
        "paper",
        {
            "investigation.max_idea_cycles": "invalid",
            "investigation.workers": False,
            "investigation.prune_every_n_cycles": 1.5,
            "investigation.max_program_evals_per_attempt": "invalid",
            "investigation.max_idea_generation_attempts": False,
            "investigation.idea_timeout_seconds": "invalid",
            "investigation.run_timeout_seconds": 0,
        },
        True,
        None,
    ),
    (
        "investigation-resume",
        "paper",
        {"investigation.resume": True, "investigation.workers": 2, "investigation.prune_every_n_cycles": 0},
        True,
        None,
    ),
]


def test_rules_do_not_block_unselected_phase(tmp_path: Path) -> None:
    for name, phases, overrides, anchored, forbidden_prefix in UNSELECTED_PHASE_CASES:
        cfg = _valid_cfg(**{"execution.phases": phases, **overrides})
        if anchored:
            run_dir = tmp_path / name
            _write_investigation_resume_state(run_dir / "investigation")
            cfg.paths.run_dir = str(run_dir)

        errors = validate_pipeline_config(cfg)

        if forbidden_prefix is None:
            assert errors == [], name
        else:
            assert not any(error.startswith(forbidden_prefix) for error in errors), name


# --- pipeline config: contracted paper publication --------------------------------


def test_contracted_publication_requires_final_phase_unless_optional(monkeypatch, tmp_path: Path) -> None:
    # A JSON contract and a malformed (directory) contract node both require the final phase.
    for name, contract_is_dir in (("json-contract", False), ("malformed-contract-node", True)):
        task_dir = tmp_path / name / "contracted"
        if contract_is_dir:
            (task_dir / "task_contract.json").mkdir(parents=True)
        else:
            task_dir.mkdir(parents=True)
            (task_dir / "task_contract.json").write_text("{}\n", encoding="utf-8")
        cfg = _valid_cfg(**{"task.dir": str(task_dir), "execution.phases": "paper"})
        assert "contracted paper phase requires final phase" in validate_pipeline_config(cfg), name

    # Optional final publication allows paper without final.
    task_dir = tmp_path / "optional" / "contracted"
    task_dir.mkdir(parents=True)
    (task_dir / "task_contract.json").write_text("{}\n", encoding="utf-8")
    cfg = _valid_cfg(**{"task.dir": str(task_dir), "execution.phases": "paper"})
    publication = type("Publication", (), {"requires_final_evaluation": False})()
    contract = type("Contract", (), {"publication": publication})()
    with monkeypatch.context() as m:
        m.setattr("src.pipeline.config.load_task_contract", lambda _path: contract)
        assert "contracted paper phase requires final phase" not in validate_pipeline_config(cfg)


# --- pipeline config: review / revision anchors ------------------------------------


REVIEW_LAYOUT_CASES: list[tuple[str, str | None]] = [
    ("published", None),
    ("outside_run", None),
    ("dir_loop", "paths.paper_dir"),
    ("pdf_loop", "review paper PDF is not a regular file"),
    ("stale_manifest", "does not point to the review paper PDF"),
]

REVISION_REVIEW_CASES: list[tuple[str, str | None, str | None]] = [
    ("missing", None, None),
    ("failed", "failed", '{"Weaknesses": []}\n'),
    ("completed", "completed", json.dumps({"Summary": "summary", "Weaknesses": []})),
]


def test_validate_review_and_revision_only_paper_layouts(tmp_path: Path) -> None:
    # review-only: the published paper layout must be a real, current PDF.
    for layout, expected in REVIEW_LAYOUT_CASES:
        root = tmp_path / "review" / layout
        run_dir = root / "run"
        run_dir.mkdir(parents=True)
        paper_dir = root / "outside-paper" if layout == "outside_run" else run_dir / "paper"
        if layout == "dir_loop":
            paper_dir.symlink_to(paper_dir)
        elif layout == "pdf_loop":
            paper_dir.mkdir()
            paper_pdf = paper_dir / "paper.pdf"
            paper_pdf.symlink_to(paper_pdf)
            _publish_paper(paper_dir, pdf=False, manifest_pdf=str(paper_pdf))
        elif layout == "stale_manifest":
            paper_dir.mkdir()
            _publish_paper(paper_dir, manifest_pdf=str((paper_dir / "older.pdf").resolve()))
        else:
            _publish_paper(paper_dir)

        errors = validate_pipeline_config(_anchored_cfg("review", run_dir, paper_dir))

        if expected is None:
            assert errors == [], layout
        else:
            assert any(expected in error for error in errors), layout

    # revision-only: requires a completed initial AI review.
    for name, status, review_payload in REVISION_REVIEW_CASES:
        run_dir = tmp_path / "revision" / name / "run"
        paper_dir = run_dir / "paper"
        before_revision = None
        if status is not None:
            assert review_payload is not None
            review_path = paper_dir / "ai_review.before_revision.json"
            paper_dir.mkdir(parents=True)
            review_path.write_text(review_payload, encoding="utf-8")
            before_revision = {"status": status, "review_json": str(review_path.resolve())}
        _publish_paper(paper_dir, before_revision=before_revision)

        errors = validate_pipeline_config(_anchored_cfg("revision", run_dir, paper_dir))

        if status == "completed":
            assert errors == [], name
        else:
            assert any("completed initial AI review" in error for error in errors), name


def _interrupt_rename(monkeypatch: pytest.MonkeyPatch, paper_dir: Path, replacement_dir: Path, *, on_call: int) -> None:
    """Replace paper_dir with replacement_dir but raise on the on_call-th rename, then restore."""
    real_rename = portable_publication._rename_entry
    calls = 0

    def interrupt(parent_fd: int, source: str, destination: str) -> None:
        nonlocal calls
        calls += 1
        if calls == on_call:
            raise RuntimeError("injected restart")
        real_rename(parent_fd, source, destination)

    monkeypatch.setattr(portable_publication, "_rename_entry", interrupt)
    with pytest.raises(RuntimeError, match="injected restart"):
        replace_directory_portably(paper_dir, replacement_dir)
    monkeypatch.setattr(portable_publication, "_rename_entry", real_rename)


def test_pending_revision_publication_recovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # revision resume preflight reaches pending publication recovery
    run_dir = tmp_path / "revision" / "run"
    paper_dir = run_dir / "paper"
    replacement_dir = run_dir / ".paper.revision-resume"
    run_dir.mkdir(parents=True)
    write_workspace(paper_dir, title="Before")
    write_workspace(replacement_dir, title="After")
    review_path = paper_dir / "ai_review.before_revision.json"
    (replacement_dir / review_path.name).write_text('{"Summary": "review"}\n', encoding="utf-8")
    (replacement_dir / "latest_manifest.json").write_text(
        json.dumps(
            {
                "paper_pdf": str((paper_dir / "paper.pdf").resolve()),
                "before_revision": {"status": "completed", "review_json": str(review_path.resolve())},
            }
        ),
        encoding="utf-8",
    )
    _interrupt_rename(monkeypatch, paper_dir, replacement_dir, on_call=2)
    assert not paper_dir.exists()
    cfg = _anchored_cfg("revision", run_dir, paper_dir)

    assert validate_pipeline_config(cfg) == []
    assert _recover_pending_revision_publication(derive_paths(cfg), {"revision"})
    assert (paper_dir / "paper.tex").read_text(encoding="utf-8") == "\\title{After}\n"
    assert validate_pipeline_config(cfg) == []

    # paper / review consumers recover a prepared revision before consuming the paper
    for phase in ("paper", "review"):
        run_dir = tmp_path / phase / "run"
        paper_dir = run_dir / "paper"
        replacement_dir = run_dir / ".paper.revision-review-resume"
        run_dir.mkdir(parents=True)
        write_workspace(paper_dir, title="Before")
        write_workspace(replacement_dir, title="After")
        for directory in (paper_dir, replacement_dir):
            (directory / "latest_manifest.json").write_text(
                json.dumps({"paper_pdf": str((paper_dir / "paper.pdf").resolve())}), encoding="utf-8"
            )
        _interrupt_rename(monkeypatch, paper_dir, replacement_dir, on_call=1)
        cfg = _anchored_cfg(phase, run_dir, paper_dir)

        if phase == "review":
            assert validate_pipeline_config(cfg) == []
        assert _recover_pending_revision_publication(derive_paths(cfg), {phase}), phase
        assert (paper_dir / "paper.tex").read_text(encoding="utf-8") == "\\title{After}\n", phase

    # review-only blocks a malformed pending publication journal
    run_dir = tmp_path / "malformed" / "run"
    paper_dir = run_dir / "paper"
    paper_dir.mkdir(parents=True)
    (run_dir / ".paper.revision-publication.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="publication journal"):
        _recover_pending_revision_publication(derive_paths(_anchored_cfg("review", run_dir, paper_dir)), {"review"})


# --- pipeline config: investigation anchors and resume state -----------------------


def test_downstream_phase_anchors_require_investigation_workspace(tmp_path: Path) -> None:
    # paper anchor requires a completed investigation
    run_dir = tmp_path / "paper"
    run_dir.mkdir()
    errors = validate_pipeline_config(_valid_cfg(**{"execution.phases": "paper", "paths.run_dir": str(run_dir)}))
    assert any("investigation workspace" in error for error in errors)

    # ablations anchor allows a legacy run without task snapshot
    run_dir = tmp_path / "ablations-legacy"
    run_dir.mkdir()
    errors = validate_pipeline_config(_valid_cfg(**{"execution.phases": "ablations", "paths.run_dir": str(run_dir)}))
    assert not any("investigation task snapshot" in error for error in errors)

    # ablations / final_evaluation reject an envelope (empty or legacy flat) without a workspace
    for phase in ("ablations", "final_evaluation"):
        for name, legacy_flat in (("empty-envelope", False), ("legacy-flat-state", True)):
            run_dir = tmp_path / f"{phase}-{name}"
            run_dir.mkdir()
            if legacy_flat:
                _write_investigation_resume_state(run_dir)

            errors = validate_pipeline_config(_valid_cfg(**{"execution.phases": phase, "paths.run_dir": str(run_dir)}))

            assert any(
                "investigation workspace" in error and str(run_dir / "investigation") in error and "not a directory" in error
                for error in errors
            ), (phase, name)


def _empty_workspace(run_dir: Path) -> None:
    (run_dir / "investigation").mkdir()


def _symlinked_workspace(run_dir: Path) -> None:
    _write_investigation_resume_state(run_dir / "outside")
    (run_dir / "investigation").symlink_to(run_dir / "outside", target_is_directory=True)


def _symlinked_state_file(run_dir: Path) -> None:
    workspace = run_dir / "investigation"
    workspace.mkdir()
    (run_dir / "external-state.json").write_text('{"current_cycle": 1}', encoding="utf-8")
    (workspace / "outer_state.json").symlink_to(run_dir / "external-state.json")
    (workspace / "best_program_manifest.json").write_text("{}", encoding="utf-8")


def _empty_state_objects(run_dir: Path) -> None:
    workspace = run_dir / "investigation"
    workspace.mkdir()
    (workspace / "outer_state.json").write_text("{}", encoding="utf-8")
    (workspace / "best_program_manifest.json").write_text("{}", encoding="utf-8")


def _valid_workspace(run_dir: Path) -> None:
    _write_investigation_resume_state(run_dir / "investigation")


def _legacy_manifest_without_entrypoint(run_dir: Path) -> None:
    workspace = run_dir / "investigation"
    _write_investigation_resume_state(workspace)
    (workspace / "best_program_manifest.json").write_text(
        json.dumps({"program_dir": str(workspace / "best_program")}), encoding="utf-8"
    )


RESUME_STATE_CASES: list[tuple[str, Callable[[Path], None], str | None, list[list[str]] | None]] = [
    ("missing-state-files", _empty_workspace, "investigation", [["outer_state.json"], ["best_program_manifest.json"]]),
    ("symlinked-workspace", _symlinked_workspace, "investigation", [["not a directory"]]),
    ("symlinked-state-file", _symlinked_state_file, None, [["outer_state.json", "missing or invalid"]]),
    ("empty-state-objects", _empty_state_objects, None, [["current_cycle"], ["program_dir"]]),
    ("valid-state", _valid_workspace, "investigation", None),
    ("legacy-flat-state", _write_investigation_resume_state, None, [["investigation workspace is not a directory"]]),
    ("legacy-manifest-without-entrypoint", _legacy_manifest_without_entrypoint, "investigation", None),
]


def test_validate_investigation_resume_state(tmp_path: Path) -> None:
    for name, build, phases, expected in RESUME_STATE_CASES:
        run_dir = tmp_path / name
        run_dir.mkdir()
        build(run_dir)
        overrides: dict[str, object] = {"investigation.resume": True, "paths.run_dir": str(run_dir)}
        if phases is not None:
            overrides["execution.phases"] = phases

        errors = validate_pipeline_config(_valid_cfg(**overrides))

        if expected is None:
            assert errors == [], name
        else:
            for needles in expected:
                assert any(all(needle in error for needle in needles) for error in errors), (name, needles)

    # A resume manifest pointing at an unknown home is rejected without crashing.
    for manifest_key in ("program_path", "program_dir"):
        results_dir = tmp_path / f"unknown-home-{manifest_key}"
        results_dir.mkdir()
        (results_dir / "outer_state.json").write_text('{"current_cycle": 1}', encoding="utf-8")
        (results_dir / "best_program_manifest.json").write_text(
            json.dumps({manifest_key: f"{UNKNOWN_HOME}/program", "entrypoint_path": "src/program.py"}),
            encoding="utf-8",
        )
        assert any(
            "no usable program_dir" in error for error in validate_investigation_resume_state(results_dir)
        ), manifest_key


def test_validate_investigation_resume_mode_checks(tmp_path: Path) -> None:
    # Pruned / parallel investigation resume requires multiple rounds.
    _write_investigation_resume_state(tmp_path / "investigation")
    for workers, prune_every_n_cycles, message in (
        (2, 0, "parallel investigation resume"),
        (2, 3, "parallel investigation resume"),
        (2, 4, "parallel investigation resume"),
        (1, 3, "investigation resume requires"),
    ):
        cfg = _valid_cfg(
            **{
                "paths.run_dir": str(tmp_path),
                "investigation.resume": True,
                "investigation.workers": workers,
                "investigation.max_idea_cycles": 3,
                "investigation.prune_every_n_cycles": prune_every_n_cycles,
            }
        )
        assert any(message in error for error in validate_pipeline_config(cfg)), (workers, prune_every_n_cycles)

    # An invalid investigation count skips the resume-mode check.
    errors = validate_pipeline_config(_valid_cfg(**{"investigation.resume": True, "investigation.workers": 0}))
    assert "investigation.execution.workers must be >= 1" in errors
    assert not any("investigation resume requires" in error for error in errors)


def test_validate_posthoc_anchors_are_best_effort(tmp_path: Path) -> None:
    # posthoc without investigation is best-effort
    errors = validate_pipeline_config(_valid_cfg(**{"execution.phases": "posthoc", "paths.run_dir": None}))
    assert not any("paths.run_dir" in error for error in errors)

    # paper allows explicit posthoc without a consumable analysis (partial ranking / degraded rerun)
    for name, degraded in (("partial-ranking", False), ("degraded-rerun", True)):
        posthoc = tmp_path / name / "posthoc"
        if degraded:
            attempt = posthoc / "analyses" / "analysis_a" / "attempt_1"
            attempt.mkdir(parents=True)
            (attempt / "analysis_summary.md").write_text("# Finding\n", encoding="utf-8")
            (posthoc / "ranking.json").write_text(
                '{"valid":[{"analysis_id":"analysis_a","selected_attempt":"attempt_1"}]}\n', encoding="utf-8"
            )
            (posthoc / "integration_status.json").write_text('{"status":"degraded"}\n', encoding="utf-8")
        else:
            (posthoc / "analyses").mkdir(parents=True)
            (posthoc / "ranking.json").write_text('{"valid": []}\n', encoding="utf-8")
        cfg = _valid_cfg(**{"execution.phases": "ablations,paper", "paths.posthoc_dir": str(posthoc)})

        assert not any("posthoc" in error for error in validate_pipeline_config(cfg)), name


# --- investigation config: config_to_flat_dict ---------------------------------------


FLAT_DICT_CASES: list[tuple[str, Callable[[], InvestigationConfig], dict[str, str]]] = [
    (
        "agent",
        lambda: InvestigationConfig(
            agent=AgentConfig(name="codex", model="custom-model", profile="custom", reasoning_effort="high")
        ),
        {
            "agent.name": "codex",
            "agent.model": "custom-model",
            "agent.profile": "custom",
            "agent.reasoning_effort": "high",
        },
    ),
    (
        "none-as-null",
        lambda: InvestigationConfig(agent=AgentConfig(name="claude", model=None)),
        {"agent.model": "null", "agent.profile": "null", "agent.reasoning_effort": "null"},
    ),
    (
        "bools",
        lambda: InvestigationConfig(execution=ExecutionConfig(resume=True), paper=InvestigationPaperConfig(enabled=False)),
        {"execution.resume": "True", "paper.enabled": "False"},
    ),
    (
        "ints",
        lambda: InvestigationConfig(execution=ExecutionConfig(workers=4, max_idea_cycles=10)),
        {"execution.workers": "4", "execution.max_idea_cycles": "10"},
    ),
    (
        "container",
        lambda: InvestigationConfig(container=ContainerConfig(enabled=True, image="my-image:latest", network="host")),
        {"container.enabled": "True", "container.image": "my-image:latest", "container.network": "host"},
    ),
    (
        "paths",
        lambda: InvestigationConfig(paths=PathsConfig(task_dir="examples/kissing", results_dir="/tmp/results")),
        {"paths.task_dir": "examples/kissing", "paths.results_dir": "/tmp/results"},
    ),
    (
        "versioning",
        lambda: load_config([]),
        {
            "versioning.enabled": "True",
            "versioning.git_dir_name": ".agentic_history",
            "versioning.finalize_on_exit": "True",
        },
    ),
]


def test_config_to_flat_dict_covers_sections_and_renders_scalars() -> None:
    # Every section and the plain run_id field are flattened.
    flat = config_to_flat_dict(InvestigationConfig())

    assert {"agent", "paths", "execution", "paper", "container"} <= {k.split(".")[0] for k in flat if "." in k}
    assert flat["run_id"]  # plain str field, not a nested dataclass
    for key in (
        "max_idea_cycles",
        "workers",
        "resume",
        "draft_task_if_missing",
        "max_program_evals_per_attempt",
        "harness_evaluated_attempts",
        "gpu_evaluation",
        "run_timeout_seconds",
        "run_timeout_grace_seconds",
        "prune_every_n_cycles",
        "parallel_fail_fast",
    ):
        assert f"execution.{key}" in flat, key

    # Scalars render as hydra-style strings.
    for name, make_cfg, expected in FLAT_DICT_CASES:
        flat = config_to_flat_dict(make_cfg())
        assert {key: flat[key] for key in expected} == expected, name


# --- investigation config: load_config -------------------------------------------


LOAD_CONFIG_CASES: list[tuple[str, list[str], dict[str, object]]] = [
    (
        "defaults",
        [],
        {
            "logging.format": "human",
            "logging.verbose": False,
            "logging.heartbeat_seconds": 60,
            "logging.show_run_header": True,
            "versioning.enabled": True,
            "versioning.run_root": None,
            "versioning.git_dir_name": ".agentic_history",
            "versioning.max_file_bytes": 0,
            "versioning.max_total_bytes": 16 * 1024**3,
            "versioning.timeout_seconds": 600,
            "versioning.tag_checkpoints": True,
            "versioning.final_include_environments": False,
            "versioning.finalize_on_exit": True,
        },
    ),
    ("harness-evaluated-attempts", ["execution.harness_evaluated_attempts=true"], {"execution.harness_evaluated_attempts": True}),
    (
        "gpu-evaluation",
        ["execution.harness_evaluated_attempts=true", "execution.gpu_evaluation=true"],
        {"execution.gpu_evaluation": True},
    ),
    ("fail-fast", ["execution.parallel_fail_fast=true"], {"execution.parallel_fail_fast": True}),
    (
        "worker-cycle-position",
        ["execution.cycle_offset=2", "execution.total_cycles=6"],
        {"execution.cycle_offset": 2, "execution.total_cycles": 6},
    ),
    ("log-format-equals", ["--log-format=json"], {"logging.format": "json"}),
    ("log-format-space", ["--log-format", "json"], {"logging.format": "json"}),
    ("verbose-flag", ["--verbose"], {"logging.verbose": True}),
    (
        "logging-overrides",
        [
            "logging.format=json",
            "logging.verbose=true",
            "logging.heartbeat_seconds=15",
            "logging.show_run_header=false",
        ],
        {
            "logging.format": "json",
            "logging.verbose": True,
            "logging.heartbeat_seconds": 15,
            "logging.show_run_header": False,
        },
    ),
    (
        "versioning-overrides",
        [
            "versioning.enabled=false",
            "versioning.max_file_bytes=1024",
            "versioning.max_total_bytes=4096",
            "versioning.tag_checkpoints=false",
            "versioning.final_include_environments=true",
        ],
        {
            "versioning.enabled": False,
            "versioning.max_file_bytes": 1024,
            "versioning.max_total_bytes": 4096,
            "versioning.tag_checkpoints": False,
            "versioning.final_include_environments": True,
        },
    ),
]


def test_load_config_applies_overrides_and_dict_to_config_keeps_flags() -> None:
    # _dict_to_config keeps execution flags.
    for execution, flag in (
        ({"harness_evaluated_attempts": True}, "harness_evaluated_attempts"),
        ({"harness_evaluated_attempts": True, "gpu_evaluation": True}, "gpu_evaluation"),
        ({"parallel_fail_fast": True}, "parallel_fail_fast"),
    ):
        assert getattr(_dict_to_config({"execution": execution}).execution, flag) is True, flag

    # load_config applies overrides and convenience flags.
    for name, arguments, expected in LOAD_CONFIG_CASES:
        cfg = load_config(arguments)
        assert {key: _get(cfg, key) for key in expected} == expected, name


LOAD_CONFIG_REJECT_CASES: list[tuple[str, list[str], str]] = [
    ("gpu-without-harness", ["execution.gpu_evaluation=true"], "requires"),
    (
        "gpu-when-drafting-is-noop",
        [
            "paths.task_dir=examples/circpack",
            "execution.draft_task_if_missing=true",
            "execution.gpu_evaluation=true",
        ],
        "requires",
    ),
    ("negative-cycle-offset", ["execution.cycle_offset=-1"], "must be >= 0"),
    ("negative-total-cycles", ["execution.total_cycles=-1"], "must be >= 0"),
    ("runtime-limit-bool", ["task_drafting.max_runtime_bytes=true"], "max_runtime_bytes"),
    ("runtime-limit-float", ["task_drafting.max_runtime_bytes=1.5"], "max_runtime_bytes"),
    ("logging-format", ["logging.format=legacy"], "logging.format"),
    ("logging-heartbeat", ["logging.heartbeat_seconds=-1"], "logging.heartbeat_seconds"),
    ("unknown-versioning-key", ["+versioning.max_file_byte=1"], "Unknown keys in 'versioning'"),
    ("git-dir-escape", ["versioning.git_dir_name=../escape"], "one non-empty path component"),
    ("git-dir-empty", ["versioning.git_dir_name=''"], "one non-empty path component"),
    ("git-dir-space", ["versioning.git_dir_name='unsafe name'"], "letters, digits, dot, underscore, or hyphen"),
    ("git-dir-glob", ["versioning.git_dir_name='unsafe*name'"], "letters, digits, dot, underscore, or hyphen"),
    ("negative-max-file-bytes", ["versioning.max_file_bytes=-1"], "max_file_bytes must be >= 0"),
    ("zero-timeout", ["versioning.timeout_seconds=0"], "timeout_seconds must be > 0"),
]


def test_load_config_rejects_invalid_values() -> None:
    for name, overrides, message in LOAD_CONFIG_REJECT_CASES:
        with pytest.raises(ConfigValidationError) as error:
            load_config(overrides)
        assert message in error.value.message, name


def test_parallel_worker_command_and_harness_instruction_selection(tmp_path: Path) -> None:
    # The parallel worker command propagates logging and gpu configuration.
    for name, cfg, expected in (
        (
            "logging",
            InvestigationConfig(logging=LoggingConfig(format="json", verbose=True, heartbeat_seconds=15)),
            ["logging.format=json", "logging.verbose=true", "logging.heartbeat_seconds=15", "logging.show_run_header=false"],
        ),
        (
            "gpu-evaluation",
            InvestigationConfig(execution=ExecutionConfig(harness_evaluated_attempts=True, gpu_evaluation=True)),
            ["execution.gpu_evaluation=true"],
        ),
    ):
        command = build_parallel_worker_command(
            cfg=cfg,
            task_dir=tmp_path / "task",
            instruct_path=tmp_path / "instruct.md",
            results_dir=tmp_path / "results" / "worker_01",
            global_memory_dir=tmp_path / "results",
            assigned_cycles=1,
            worker_index=1,
        )
        for override in expected:
            assert override in command, (name, override)

    # Harness mode selects the harness instruction only for the default instruction path.
    for name, instruct, overrides, has_override, uses_harness in (
        (
            "default-instruction-path",
            "src/prompts/general.md",
            ["agent.name=codex", "execution.harness_evaluated_attempts=true"],
            False,
            True,
        ),
        (
            "explicit-override",
            "custom.md",
            ["agent.name=codex", "execution.harness_evaluated_attempts=true", "paths.instruct=custom.md"],
            True,
            False,
        ),
        ("programmatic-path", "custom.md", [], False, False),
    ):
        cfg = InvestigationConfig(
            paths=PathsConfig(instruct=instruct), execution=ExecutionConfig(harness_evaluated_attempts=True)
        )
        cfg.hydra_overrides = overrides

        assert has_hydra_override(cfg, "paths.instruct") is has_override, name
        assert should_use_harness_instruction(cfg) is uses_harness, name


# --- resume policy (src.pipeline.resume_config) --------------------------------------


def _write_hydra_config(run_dir: Path, content: str) -> None:
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text(content, encoding="utf-8")


def _budget(root: Path) -> int:
    return read_idea_cycle_budget(root, "world_models", "codex", "safe-run")


def _run_root(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """A fresh results root and its world_models/codex/safe-run run directory."""
    root = tmp_path / name
    return root, root / "world_models" / "codex" / "safe-run"


def test_idea_cycle_budget_reads_trusted_policy_or_legacy_envelope(monkeypatch, tmp_path: Path) -> None:
    # The trusted policy wins over the workspace hydra config (matching or not).
    for name, workspace_cycles in (("matching-workspace", 20), ("trusted-policy-wins", 999)):
        root, run_dir = _run_root(tmp_path, name)
        write_resume_policy(run_dir, max_idea_cycles=20)
        _write_hydra_config(run_dir / "investigation", f"execution:\n  max_idea_cycles: {workspace_cycles}\n")
        assert _budget(root) == 20, name

    # Publication does not require hard links.
    root, run_dir = _run_root(tmp_path, "no-hard-links")
    with monkeypatch.context() as m:
        m.setattr(
            "src.pipeline.resume_config.os.link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unsupported")),
        )
        write_resume_policy(run_dir, max_idea_cycles=20)
    assert _budget(root) == 20

    # A legacy envelope hydra config is read.
    root, run_dir = _run_root(tmp_path, "legacy-envelope")
    _write_hydra_config(run_dir, "execution:\n  max_idea_cycles: 20\n")
    assert _budget(root) == 20

    # read_run_idea_cycle_budget prefers the trusted policy ...
    run_dir = tmp_path / "run-policy"
    write_resume_policy(run_dir, max_idea_cycles=7)
    _write_hydra_config(run_dir / "investigation", "execution:\n  max_idea_cycles: 999\n")
    assert read_run_idea_cycle_budget(run_dir) == 7

    # ... and reads the legacy envelope config.
    run_dir = tmp_path / "run-legacy"
    _write_hydra_config(run_dir, "investigation:\n  max_idea_cycles: 12\n")
    assert read_run_idea_cycle_budget(run_dir) == 12


def test_idea_cycle_budget_rejects_unsafe_or_inconsistent_layouts(tmp_path: Path) -> None:
    # The new layout without a trusted policy does not fall back to the workspace.
    root, run_dir = _run_root(tmp_path, "no-policy")
    _write_hydra_config(run_dir / "investigation", "execution:\n  max_idea_cycles: 20\n")
    with pytest.raises(ResumeConfigError, match="trusted resume policy"):
        _budget(root)

    # Inconsistent pipeline and investigation budgets are rejected.
    root, run_dir = _run_root(tmp_path, "inconsistent")
    _write_hydra_config(run_dir, "investigation:\n  max_idea_cycles: 20\nexecution:\n  max_idea_cycles: 2\n")
    with pytest.raises(ResumeConfigError, match="missing or inconsistent"):
        _budget(root)

    # A symlinked .hydra directory is rejected.
    root, run_dir = _run_root(tmp_path, "symlinked-hydra")
    run_dir.mkdir(parents=True)
    external = root / "external"
    external.mkdir()
    (external / "config.yaml").write_text("execution:\n  max_idea_cycles: 20\n", encoding="utf-8")
    (run_dir / ".hydra").symlink_to(external, target_is_directory=True)
    with pytest.raises(ResumeConfigError, match="missing or unsafe"):
        _budget(root)

    # read_run_idea_cycle_budget is None without a policy or legacy config.
    for name, new_layout in (("new-layout-without-policy", True), ("empty-run-directory", False)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        if new_layout:
            (run_dir / "investigation").mkdir()
        assert read_run_idea_cycle_budget(run_dir) is None, name

    # A corrupt policy is rejected.
    run_dir = tmp_path / "corrupt-policy"
    run_dir.mkdir()
    (run_dir / ".agentic_resume_policy.json").write_text("{", encoding="utf-8")
    with pytest.raises(ResumeConfigError, match="invalid JSON"):
        read_run_idea_cycle_budget(run_dir)

    # A missing run directory is rejected.
    with pytest.raises(ResumeConfigError, match="missing or unsafe"):
        read_run_idea_cycle_budget(tmp_path / "absent")


def test_resume_policy_republication_and_task_handoff_claim_are_one_shot(tmp_path: Path) -> None:
    # Conflicting republication is rejected.
    _root, run_dir = _run_root(tmp_path, "republish")
    write_resume_policy(run_dir, max_idea_cycles=20)
    with pytest.raises(ResumeConfigError, match="does not match"):
        write_resume_policy(run_dir, max_idea_cycles=21)

    # The task handoff claim requires a matching policy and is one-shot.
    _root, run_dir = _run_root(tmp_path, "handoff")
    write_resume_policy(run_dir, max_idea_cycles=20)
    publish_task_handoff(run_dir)

    with pytest.raises(ResumeConfigError, match="does not match"):
        claim_task_handoff_for_investigation(run_dir, max_idea_cycles=21)
    assert not (run_dir / INVESTIGATION_CLAIM_FILENAME).exists()

    claim_task_handoff_for_investigation(run_dir, max_idea_cycles=20)

    with pytest.raises(ResumeConfigError, match="already exists"):
        claim_task_handoff_for_investigation(run_dir, max_idea_cycles=20)


# --- configs/paper_config.yaml mirrors src/scripts/paper.sh ------------------------------
# paper.sh takes CLI flags, so the config mirrors its surface; both directions are checked
# by parsing paper.sh itself (a flag missing from the config is unreachable, a stale default
# makes the diff baseline wrong).

PAPER_CONFIG = REPO_ROOT / "configs" / "paper_config.yaml"


def _config_keys() -> dict[str, object]:
    raw = yaml.safe_load(PAPER_CONFIG.read_text(encoding="utf-8"))
    return {key: value for key, value in raw.items() if key != "defaults"}


def _flag_to_key(flag: str) -> str:
    return flag[2:].replace("-", "_")


def test_paper_config_mirrors_every_paper_script_flag() -> None:
    keys = _config_keys()
    arms = flag_arms()

    # Every long flag has a config key.
    missing = [arm.long_flag for arm in arms if _flag_to_key(arm.long_flag) not in keys]
    assert missing == [], f"paper.sh flags without a configs/paper_config.yaml key: {missing}"

    # Every config key has a flag.
    extra = sorted(set(keys) - {_flag_to_key(arm.long_flag) for arm in arms})
    assert extra == [], f"configs/paper_config.yaml keys with no paper.sh flag: {extra}"

    # Presence flags are false by default.
    for arm in arms:
        if not arm.takes_value:
            assert keys[_flag_to_key(arm.long_flag)] is False, arm.long_flag

    # Non-owned value keys carry paper.sh's own default (``${VAR:-x}`` read as ``x``).
    defaults = script_defaults()
    mismatches: list[str] = []
    for arm in arms:
        key = _flag_to_key(arm.long_flag)
        if not arm.takes_value or key in PAPER_OWNED_KEYS:
            continue
        script_default = defaults[arm.value_vars[0]]
        mirrored = keys[key]
        mirrored_text = "" if mirrored is None else str(mirrored)
        if mirrored_text != script_default:
            mismatches.append(f"{key}: config={mirrored!r} script={script_default!r}")
    assert mismatches == [], "\n".join(mismatches)


def test_paper_config_owned_forwarded_and_rewritten_keys() -> None:
    keys = _config_keys()

    # Owned keys exist in the mirror.
    assert PAPER_OWNED_KEYS <= set(keys)
    assert tuple(sorted(PAPER_OWNED_KEYS)) == PaperConfig.owned

    # A flag is either forwarded generically, owned, or resolved by the builder.
    reachable = set(keys) - PAPER_OWNED_KEYS - set(PaperConfig.rewritten)
    assert {"paper_review_rounds", "paper_max_repair_rounds", "skills", "instruct"} <= reachable
    assert PaperConfig.rewritten == {"venue_file", "final_outcome_policy"}
    assert PAPER_SCRIPT.is_file()

    # Flag defaults are literals; an env-backed flag would have to be in always_forwarded.
    env_vars = environment_backed_vars()
    env_backed_keys = {
        _flag_to_key(arm.long_flag)
        for arm in flag_arms()
        if arm.takes_value and set(arm.value_vars) & env_vars
    }
    assert env_backed_keys == set()
    assert env_backed_keys - PAPER_OWNED_KEYS <= PaperConfig.always_forwarded
    assert not PaperConfig.always_forwarded & PaperConfig.rewritten
