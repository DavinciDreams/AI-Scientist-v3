"""GPU selector arithmetic, and how pipeline and parallel workers settle container.gpu_count."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pipeline.cli import GPU_COMPUTE_PHASES, load_pipeline_config, prepare_gpu_allocation
from src.investigation import cli as investigation_cli, parallel_helpers
from src.investigation.config import ContainerConfig, ExecutionConfig, InvestigationConfig
from src.investigation.config_loader import load_config
from src.investigation.container_gpus import (
    GpuSelectorError, bounded_docker_gpu_selector, discover_gpu_selector, gpu_pool_label,
    gpu_pool_size, settle_gpu_pool, validate_gpu_allocation, worker_gpu_selector,
)
from src.investigation.models import ParallelWorkerPlan
from tests.program_testutils import make_program_dir

POOL = "AGENT_CONTAINER_GPU_POOL"
GPUS = "AGENT_CONTAINER_GPUS"
POOL_UUIDS = tuple(f"GPU-{index:08x}-0000-0000-0000-000000000000" for index in range(1, 9))
UUIDS = POOL_UUIDS[:4]


def _selector(*devices: str) -> str:
    joined = "device=" + ",".join(devices)
    return f'"{joined}"' if len(devices) > 1 else joined


def _no_discovery(gpu_count: int, **_: object) -> str:
    raise AssertionError(f"discovery must not run (count={gpu_count})")


def _discover(calls: list, *, labels: bool = False):
    def discover(gpu_count: int, **kwargs: object) -> str:
        calls.append((gpu_count, kwargs.get("label")) if labels else gpu_count)
        return _selector(*POOL_UUIDS[:gpu_count])
    return discover


# --------------------------------------------------------------------------- selector helpers


def test_bounded_docker_gpu_selector_slices_preserves_shape_and_rejects_invalid_input() -> None:
    # slices preserve the selector shape
    for name, raw, count, offset, expected in [
        ("single-shape", '"device=GPU-a1,GPU-b2"', 1, 0, "device=GPU-a1"),
        ("quotes-multiple", "device=0,1", 2, 0, '"device=0,1"'),
        ("offset", '"device=0,1,2"', 2, 1, '"device=1,2"'),
        ("mig", '"device=MIG-GPU-a/1/2,MIG-GPU-b/3/4"', 2, 0, '"device=MIG-GPU-a/1/2,MIG-GPU-b/3/4"'),
    ]:
        assert bounded_docker_gpu_selector(raw, count, offset=offset) == expected, name
    # malformed values are rejected
    for selector in [
        '"device=0,1', 'device=0,1"', "device=0,0", "device=0,,1", "device=0;touch /tmp/pwned", "all",
    ]:
        with pytest.raises(GpuSelectorError):
            bounded_docker_gpu_selector(selector, 2)
    # insufficient devices are rejected
    with pytest.raises(GpuSelectorError, match="requires 2 GPUs but selector exposes 1"):
        bounded_docker_gpu_selector("device=0", 2)
    # invalid bounds are rejected
    for gpu_count, offset in [(0, 0), (-1, 0), (1, -1)]:
        with pytest.raises(GpuSelectorError):
            bounded_docker_gpu_selector("device=0", gpu_count, offset=offset)


def test_gpu_pool_size_and_label_multiply_the_count_by_the_workers() -> None:
    assert gpu_pool_size(2, 1) == 2
    assert gpu_pool_size(2, 4) == 8
    assert gpu_pool_size(0, 4) == 0
    assert gpu_pool_label(2, 1) == "container.gpu_count=2"
    assert gpu_pool_label(2, 4) == "container.gpu_count=2 x 4 workers (8 GPUs)"
    for gpu_count, workers in ((-1, 1), (2, 0), (True, 1), (2, "2")):
        with pytest.raises(GpuSelectorError):
            gpu_pool_size(gpu_count, workers)  # type: ignore[arg-type]


def test_settle_gpu_pool_requires_unset_selector_for_cpu_and_discovers_or_matches_the_pool_exactly() -> None:
    # CPU runs require an unset selector
    assert settle_gpu_pool(None, 0, 4, discover=_no_discovery) is None
    assert settle_gpu_pool("  ", 0, 1, discover=_no_discovery) is None
    with pytest.raises(GpuSelectorError, match=r"container.gpu_count=0 x 4 workers \(0 GPUs\)"):
        settle_gpu_pool(_selector(POOL_UUIDS[0]), 0, 4, discover=_no_discovery)
    # the whole pool is discovered and the first slice returned
    calls: list[tuple[int, object]] = []
    assert settle_gpu_pool("", 2, 4, discover=_discover(calls, labels=True)) == (
        _selector(*POOL_UUIDS), _selector(*POOL_UUIDS[:2]),
    )
    assert calls == [(8, "container.gpu_count=2 x 4 workers (8 GPUs)")]
    # an explicit pool must match exactly
    assert settle_gpu_pool(_selector(*POOL_UUIDS[:4]), 2, 2, discover=_no_discovery) == (
        _selector(*POOL_UUIDS[:4]), _selector(*POOL_UUIDS[:2]),
    )
    assert settle_gpu_pool(_selector(*POOL_UUIDS[:2]), 2, 1, discover=_no_discovery) == (
        _selector(*POOL_UUIDS[:2]), _selector(*POOL_UUIDS[:2]),
    )
    for exposed in (2, 8):
        with pytest.raises(
            GpuSelectorError,
            match=rf"container.gpu_count=2 x 2 workers \(4 GPUs\) but AGENT_CONTAINER_GPUS exposes {exposed} GPUs",
        ):
            settle_gpu_pool(_selector(*POOL_UUIDS[:exposed]), 2, 2, discover=_no_discovery)
    with pytest.raises(GpuSelectorError, match="stable GPU UUID"):
        settle_gpu_pool('"device=0,1,2,3"', 2, 2, discover=_no_discovery)


def test_worker_gpu_selector_slices_the_pool_by_worker_index() -> None:
    pool = _selector(*POOL_UUIDS)
    assert worker_gpu_selector(pool, 2, 1) == _selector(*POOL_UUIDS[:2])
    assert worker_gpu_selector(pool, 2, 4) == _selector(*POOL_UUIDS[6:8])
    assert worker_gpu_selector(pool, 1, 8) == _selector(POOL_UUIDS[7])
    with pytest.raises(GpuSelectorError, match="requires 10 GPUs but selector exposes 8"):
        worker_gpu_selector(pool, 2, 5)
    with pytest.raises(GpuSelectorError, match="positive"):
        worker_gpu_selector(pool, 2, 0)


# --------------------------------------------------------------------------- validate / discover


def test_validate_gpu_allocation_accepts_exact_stable_uuids_and_rejects_mismatches() -> None:
    # zero GPUs require an unset selector
    for raw in [None, "", "   "]:
        assert validate_gpu_allocation(raw, 0) == ()
    with pytest.raises(GpuSelectorError, match="container.gpu_count=0"):
        validate_gpu_allocation(_selector(UUIDS[0]), 0)
    # exact stable UUID allocations are accepted
    for count in [1, 2, 4]:
        assert validate_gpu_allocation(_selector(*UUIDS[:count]), count) == UUIDS[:count]
        assert validate_gpu_allocation('"device=' + UUIDS[0] + '"', 1) == UUIDS[:1]
    # mismatched allocations are rejected
    for raw, count, message in [
        (None, 2, "requires AGENT_CONTAINER_GPUS"),
        ("", 4, "requires AGENT_CONTAINER_GPUS"),
        (_selector(*UUIDS[:3]), 4, "exposes 3 GPUs"),
        (_selector(*UUIDS), 2, "exposes 4 GPUs"),
        (_selector("0", "1"), 2, "stable GPU UUID"),
        (_selector(UUIDS[0], UUIDS[0]), 2, "duplicate"),
        (_selector(UUIDS[0].replace("1", "a"), UUIDS[0].replace("1", "A")), 2, "duplicate"),
        ("all", 1, "device=<ids>"),
        ('"device=' + UUIDS[0], 1, "unbalanced"),
        (_selector(UUIDS[0], ""), 2, "empty or padded"),
    ]:
        with pytest.raises(GpuSelectorError, match=message):
            validate_gpu_allocation(raw, count)
    # invalid counts are rejected
    for bad_count in [-1, True, "2"]:
        with pytest.raises(GpuSelectorError, match="non-negative integer"):
            validate_gpu_allocation(None, bad_count)  # type: ignore[arg-type]


def _fake_nvidia_smi(tmp_path: Path, stdout: str, *, status: int = 0) -> str:
    script = tmp_path / "nvidia-smi"
    script.write_text(
        f"#!{sys.executable}\nimport sys\nsys.stdout.write({stdout!r})\n"
        f"sys.stderr.write('smi failed')\nraise SystemExit({status})\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return str(script)


@pytest.mark.parametrize(("count", "reported"), [(1, 1), (1, 4), (2, 4), (4, 4)])
def test_discovery_takes_the_first_reported_gpus(tmp_path: Path, count: int, reported: int) -> None:
    smi = _fake_nvidia_smi(tmp_path, "\n".join(UUIDS[:reported]) + "\n")
    selector = discover_gpu_selector(count, nvidia_smi=smi)
    assert selector == _selector(*UUIDS[:count])
    assert validate_gpu_allocation(selector, count) == UUIDS[:count]


def test_discovery_rejects_unusable_nvidia_smi_output_and_requires_nvidia_smi(tmp_path: Path) -> None:
    for stdout, status, count, message in [
        ("\n".join(UUIDS[:2]), 0, 4, "reports 2 GPUs"),
        ("", 0, 1, "reports 0 GPUs"),
        ("No devices were found", 6, 1, "status 6"),
        ("0\n1\n", 0, 2, "unexpected GPU identifier"),
        (f"{UUIDS[0]}\n{UUIDS[0]}\n", 0, 2, "duplicate"),
    ]:
        smi = _fake_nvidia_smi(tmp_path, stdout, status=status)
        with pytest.raises(GpuSelectorError, match=message):
            discover_gpu_selector(count, nvidia_smi=smi)
    # nvidia-smi and a positive count are required
    with pytest.raises(GpuSelectorError, match="not installed"):
        discover_gpu_selector(1, nvidia_smi=str(tmp_path / "missing-nvidia-smi"))
    with pytest.raises(GpuSelectorError, match="positive"):
        discover_gpu_selector(0, nvidia_smi=str(tmp_path / "missing-nvidia-smi"))


# --------------------------------------------------------------------------- pipeline preflight


def _pipeline(*overrides: str):
    """The grpo preset (two GPUs); explicit overrides win over the preset."""
    return load_pipeline_config(["preset=grpo", "task.dir=examples/grpo", *overrides])


def _four_gpu_config():
    return _pipeline("container.gpu_count=4")


def _cpu_config():
    return _pipeline("container.gpu_count=0", "investigation.execution.gpu_evaluation=false")


def _parallel_config(workers: int):
    return _pipeline(f"investigation.execution.workers={workers}")


def test_preflight_discovers_validates_or_keeps_selectors() -> None:
    # discovery runs when the selector is unset
    cfg = _four_gpu_config()
    calls: list[int] = []
    environment: dict[str, str] = {}
    assert prepare_gpu_allocation(cfg, {"task_drafting", "paper"}, environment, discover=_discover(calls)) is None
    assert calls == [4]
    assert environment == {GPUS: _selector(*UUIDS), "VARIANT_EVAL_SANDBOX_GPUS": _selector(*UUIDS)}
    # discovery failures are reported
    def discover(gpu_count: int, **_: object) -> str:
        raise GpuSelectorError(f"container.gpu_count={gpu_count} but nvidia-smi reports 1 GPUs")

    environment = {}
    error = prepare_gpu_allocation(cfg, {"investigation"}, environment, discover=discover)
    assert error == "container.gpu_count=4 but nvidia-smi reports 1 GPUs"
    assert environment == {}
    # an explicit selector skips discovery and must match exactly
    environment = {GPUS: _selector(*UUIDS)}
    assert prepare_gpu_allocation(cfg, {"investigation"}, environment, discover=_no_discovery) is None
    assert environment == {GPUS: _selector(*UUIDS)}
    environment = {GPUS: _selector(*UUIDS[:2])}
    assert (
        prepare_gpu_allocation(cfg, {"final_evaluation"}, environment, discover=_no_discovery)
        == "container.gpu_count=4 but AGENT_CONTAINER_GPUS exposes 2 GPUs"
    )
    # an existing ablation selector is kept
    cfg = _pipeline()
    environment = {"VARIANT_EVAL_SANDBOX_GPUS": _selector(UUIDS[3])}
    assert prepare_gpu_allocation(cfg, {"ablations"}, environment, discover=_discover([])) is None
    assert environment[GPUS] == _selector(*UUIDS[:2])
    assert environment["VARIANT_EVAL_SANDBOX_GPUS"] == _selector(UUIDS[3])


def test_cpu_presets_and_downstream_only_phases_skip_the_allocation_check() -> None:
    # a CPU configuration never discovers and rejects exported GPUs for compute phases
    cfg = _cpu_config()
    assert cfg.container.gpu_count == 0
    assert cfg.investigation.get("execution.gpu_evaluation") is False
    assert prepare_gpu_allocation(cfg, {"investigation"}, {}, discover=_no_discovery) is None
    exported = {GPUS: _selector(UUIDS[0])}
    for phase in sorted(GPU_COMPUTE_PHASES):
        error = prepare_gpu_allocation(cfg, {phase}, dict(exported), discover=_no_discovery)
        assert error is not None and "container.gpu_count=0" in error
    for phase in ("posthoc", "paper", "review", "revision"):
        assert prepare_gpu_allocation(cfg, {phase}, dict(exported), discover=_no_discovery) is None
    # downstream-only phases skip the check entirely
    cfg = _four_gpu_config()
    assert prepare_gpu_allocation(cfg, {"posthoc", "paper"}, {}, discover=_no_discovery) is None
    assert prepare_gpu_allocation(cfg, set(), {}, discover=_no_discovery) is None


def test_parallel_workers_widen_the_pool_only_for_investigation_and_slice_it() -> None:
    # parallel workers multiply the discovered pool and slice it
    calls: list[tuple[int, object]] = []
    environment: dict[str, str] = {}
    assert prepare_gpu_allocation(
        _parallel_config(2), {"investigation"}, environment, discover=_discover(calls, labels=True),
    ) is None
    assert calls == [(4, "container.gpu_count=2 x 2 workers (4 GPUs)")]
    assert environment == {
        POOL: _selector(*UUIDS), GPUS: _selector(*UUIDS[:2]), "VARIANT_EVAL_SANDBOX_GPUS": _selector(*UUIDS[:2]),
    }
    # an explicit pool must name exactly one slice per worker
    cfg = _parallel_config(2)
    environment = {GPUS: _selector(*UUIDS)}
    assert prepare_gpu_allocation(cfg, {"investigation"}, environment, discover=_no_discovery) is None
    assert environment == {POOL: _selector(*UUIDS), GPUS: _selector(*UUIDS[:2])}
    for exposed in (UUIDS[:2], UUIDS[:3]):
        environment = {GPUS: _selector(*exposed)}
        error = prepare_gpu_allocation(cfg, {"investigation"}, environment, discover=_no_discovery)
        assert error == (
            "container.gpu_count=2 x 2 workers (4 GPUs) but AGENT_CONTAINER_GPUS exposes "
            f"{len(exposed)} GPUs"
        )
        assert POOL not in environment
    # workers only widen the pool when the investigation phase runs
    single_calls: list[int] = []
    environment = {}
    assert prepare_gpu_allocation(
        _parallel_config(2), {"final_evaluation"}, environment, discover=_discover(single_calls),
    ) is None
    assert single_calls == [2]
    assert environment == {GPUS: _selector(*UUIDS[:2]), "VARIANT_EVAL_SANDBOX_GPUS": _selector(*UUIDS[:2])}
    # a single worker leaves the allocation untouched
    environment = {GPUS: _selector(*UUIDS[:2])}
    assert prepare_gpu_allocation(_parallel_config(1), {"investigation"}, environment, discover=_no_discovery) is None
    assert environment == {GPUS: _selector(*UUIDS[:2])}


# --------------------------------------------------------------------------- parallel investigation workers


def _cfg(*, workers: int = 2, gpu_count: int = 2, **execution: object) -> InvestigationConfig:
    return InvestigationConfig(
        execution=ExecutionConfig(max_idea_cycles=4, workers=workers, **execution),  # type: ignore[arg-type]
        container=ContainerConfig(enabled=True, gpu_count=gpu_count),
    )


def test_settle_gpu_allocation_reuses_validates_or_discovers_the_pool() -> None:
    # the pool is unused without GPUs or parallel workers
    for cfg in (_cfg(workers=1), _cfg(gpu_count=0)):
        environment = {GPUS: _selector(*UUIDS[:2])}
        assert parallel_helpers.settle_gpu_allocation(cfg, environment, discover=_no_discovery) is None
        assert environment == {GPUS: _selector(*UUIDS[:2])}
        assert parallel_helpers.worker_gpu_environment(cfg, 1, environment) == {}
    # a single worker discovers or validates its own GPUs
    calls: list[int] = []
    environment = {}
    assert parallel_helpers.settle_gpu_allocation(_cfg(workers=1), environment, discover=_discover(calls)) is None
    assert calls == [2]
    assert environment == {GPUS: _selector(*UUIDS[:2])}
    with pytest.raises(GpuSelectorError, match="container.gpu_count=2 but AGENT_CONTAINER_GPUS exposes 3"):
        parallel_helpers.settle_gpu_allocation(
            _cfg(workers=1), {GPUS: _selector(*UUIDS[:3])}, discover=_no_discovery
        )
    # a pipeline-exported pool is validated and reused
    environment = {POOL: _selector(*UUIDS), GPUS: _selector(*UUIDS[:2])}
    assert parallel_helpers.settle_gpu_allocation(_cfg(), environment) == _selector(*UUIDS)
    assert environment == {POOL: _selector(*UUIDS), GPUS: _selector(*UUIDS[:2])}
    with pytest.raises(
        GpuSelectorError,
        match=r"container.gpu_count=2 x 2 workers \(4 GPUs\) but AGENT_CONTAINER_GPUS exposes 3",
    ):
        parallel_helpers.settle_gpu_allocation(_cfg(), {POOL: _selector(*UUIDS[:3])})
    # a standalone campaign settles the pool from the selector or discovery
    environment = {GPUS: _selector(*UUIDS)}
    assert parallel_helpers.settle_gpu_allocation(_cfg(), environment, discover=_no_discovery) == _selector(*UUIDS)
    assert environment == {POOL: _selector(*UUIDS), GPUS: _selector(*UUIDS[:2])}
    calls = []
    environment = {}
    assert parallel_helpers.settle_gpu_allocation(_cfg(), environment, discover=_discover(calls)) == (
        _selector(*UUIDS)
    )
    assert calls == [4]
    assert environment[GPUS] == _selector(*UUIDS[:2])
    with pytest.raises(GpuSelectorError, match="exposes 2 GPUs"):
        parallel_helpers.settle_gpu_allocation(_cfg(), {GPUS: _selector(*UUIDS[:2])}, discover=_no_discovery)


def test_each_worker_receives_its_own_slice_and_exec_env_applies_plan_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # each worker receives its own slice
    environment = {POOL: _selector(*UUIDS)}
    assert parallel_helpers.worker_gpu_environment(_cfg(), 1, environment) == {GPUS: _selector(*UUIDS[:2])}
    assert parallel_helpers.worker_gpu_environment(_cfg(), 2, environment) == {GPUS: _selector(*UUIDS[2:])}
    assert parallel_helpers.worker_gpu_environment(_cfg(), 1, {}) == {}
    with pytest.raises(GpuSelectorError, match="requires 6 GPUs but selector exposes 4"):
        parallel_helpers.worker_gpu_environment(_cfg(), 3, environment)
    # the worker exec env applies the plan overrides
    monkeypatch.setenv(GPUS, _selector(*UUIDS[:2]))
    plan = ParallelWorkerPlan(
        worker_index=2, assigned_cycles=1, cycle_start=1, cycle_end=1,
        results_dir=Path("/nonexistent/worker_02"), command=["echo"],
        environment={GPUS: _selector(*UUIDS[2:])},
    )
    assert parallel_helpers._worker_exec_env(plan)[GPUS] == _selector(*UUIDS[2:])
    assert parallel_helpers._worker_exec_env()[GPUS] == _selector(*UUIDS[:2])


@pytest.mark.parametrize("prune_every_n_cycles", [0, 2])
def test_parallel_campaign_plans_carry_their_worker_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prune_every_n_cycles: int
) -> None:
    monkeypatch.setenv(POOL, _selector(*UUIDS))
    task_dir = tmp_path / "task"
    make_program_dir(task_dir / "program_0", content="def solve():\n    return 0\n")
    results_dir = tmp_path / "results"
    seen: list[ParallelWorkerPlan] = []

    def _fake_worker_result(plan: ParallelWorkerPlan) -> investigation_cli.ParallelWorkerResult:
        seen.append(plan)
        plan.results_dir.mkdir(parents=True, exist_ok=True)
        return investigation_cli.ParallelWorkerResult(
            plan=plan, exit_code=0, log_path=plan.log_path or plan.results_dir / "worker.log",
            elapsed_seconds=0.0, output_tail="",
        )

    with (
        patch("src.investigation.cli.build_parallel_worker_command", return_value=["echo", "ok"]),
        patch("src.investigation.cli.run_parallel_worker", side_effect=_fake_worker_result),
        patch(
            "src.investigation.cli.aggregate_parallel_worker_outputs",
            return_value={"failed_worker_count": 0, "idea_count": 0, "attempt_count": 0},
        ),
    ):
        exit_code = investigation_cli.run_parallel_campaign(
            cfg=_cfg(prune_every_n_cycles=prune_every_n_cycles), task_dir=task_dir, results_dir=results_dir,
            global_memory_dir=results_dir, instruct_path=tmp_path / "instruct.md",
            seed_program_dir=task_dir / "program_0",
        )

    assert exit_code == 0
    assert seen
    expected = {1: {GPUS: _selector(*UUIDS[:2])}, 2: {GPUS: _selector(*UUIDS[2:])}}
    for plan in seen:
        assert plan.environment == expected[plan.worker_index]


def test_the_loader_reads_container_gpu_count_from_overrides() -> None:
    cfg = load_config(["container.gpu_count=2", "execution.workers=2"])
    assert cfg.container.gpu_count == 2
    assert load_config([]).container.gpu_count == 0
