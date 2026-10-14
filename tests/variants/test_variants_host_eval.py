"""Host-side sandbox evaluation, brokered search, teardown recovery, and sandbox outputs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.final_evaluation.candidate_broker import (
    CandidateBrokerInfrastructureError,
    CandidateContainerCleanupError,
)
from src.final_evaluation.runner import FinalEvaluationError
from src.investigation.bounded_process import ProcessTimeoutError
from src.investigation.container_gpus import GpuLeaseTimeout
from src.investigation.evaluator_outputs import EvaluatorOutputContractError
from src.investigation.evaluator_tree import managed_evaluator_tree_digest
from src.investigation.task_contract import load_task_contract
from src.variants_pipeline import host_eval, sandbox_outputs, sandbox_runner
from src.variants_pipeline.host_eval import (
    SandboxEvaluationTimeout,
    SandboxTeardownError,
    _bounded_gpu_selector,
    _brokered_search_commands,
    _evaluate_brokered_search_locked,
    discard_evaluation_artifacts,
    evaluate_variant_program,
    resolve_host_python,
)
from src.variants_pipeline.sandbox_outputs import (
    build_promotable_frame,
    canonicalize_results,
    copy_promotable_results,
    extract_promotable_frame,
    validate_promotable_results,
)
from src.variants_pipeline.sandbox_runner import _sanitize_diagnostic, _stream_bounded_output
from tests.program_testutils import make_program_dir
from tests.final_evaluation.test_final_evaluation import _write_contract_task

IMAGE = "sha256:" + "a" * 64
ENVIRONMENT = {
    "AGENT_NAME": "pi",
    "BASE_PROGRAM": "program_2",
    "EVALUATION_ID": "eval-1",
    "VARIANT_ID": "control_best",
    "VARIANT_KIND": "control",
}
_TS_HEADER = "evaluation_id,variant_id,series_name,step,value\n"


@pytest.fixture(autouse=True)
def _mock_sandbox_creation():
    with patch("src.variants_pipeline.host_eval._create_sandbox") as create:
        yield create


@pytest.fixture
def recovery_root(tmp_path: Path) -> Iterator[Path]:
    root = tmp_path / "recovery"
    root.mkdir(mode=0o700)
    with patch("src.variants_pipeline.host_eval._private_sandbox_root", return_value=root):
        yield root


# ── helpers: legacy single-container sandbox ──────────────────────


class _CompletedSandbox:
    def __init__(self, command: list[str], *, correct: bool = True, returncode: int = 0) -> None:
        self.command = command
        self._returncode = returncode
        self.returncode = returncode
        with tempfile.TemporaryDirectory() as temporary:
            temporary_dir = Path(temporary)
            (temporary_dir / "_results.json").write_text('{"score": 1.25}')
            (temporary_dir / "metrics.json").write_text('{"combined_score": 1.25}')
            (temporary_dir / "correct.json").write_text(json.dumps({"correct": correct}))
            sandbox_env = {
                command[index + 1].split("=", 1)[0]: command[index + 1].split("=", 1)[1]
                for index, value in enumerate(command)
                if value == "--env"
            }
            canonicalize_results(temporary_dir, "program_1", sandbox_env)
            self.archive_payload = build_promotable_frame(temporary_dir)

    def communicate(self, timeout: int | None = None):
        del timeout
        return self.archive_payload, None

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        return self._returncode

    def poll(self) -> int | None:
        return self._returncode


class _InvalidFrameSandbox(_CompletedSandbox):
    def __init__(self, command: list[str]) -> None:
        super().__init__(command)
        self.archive_payload = b"not a valid promotion frame"


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create (task_bundle, task_source, evaluator_source) with a sealed evaluator."""
    task_bundle, task_source, evaluator_source = (
        tmp_path / name for name in ("task_bundle", "task_source", "retrofitted")
    )
    for directory in (task_bundle, task_source, evaluator_source):
        directory.mkdir(parents=True)
    make_program_dir(task_bundle / "program_1")
    (evaluator_source / "evaluate.py").write_text("# sealed evaluator\n")
    return task_bundle, task_source, evaluator_source


def _evaluate_program(task_bundle: Path, task_source: Path, evaluator_source: Path, **overrides: Any) -> host_eval.HostEvaluationResult:
    kwargs: dict[str, Any] = dict(
        task_bundle_dir=task_bundle, evaluator_source_dir=evaluator_source,
        task_source_dir=task_source, program_index=1, python_bin=Path("/usr/bin/python3"),
        sandbox_image="sha256:test", timeout_seconds=10,
    )
    kwargs.update(overrides)
    return evaluate_variant_program(**kwargs)


def _popen(sandbox_factory):
    return patch(
        "src.variants_pipeline.host_eval.subprocess.Popen",
        side_effect=lambda command, **_: sandbox_factory(command),
    )


def _evaluate(
    tmp_path: Path,
    *,
    correct: bool = True,
    returncode: int = 0,
    python_bin: Path = Path("/usr/bin/python3"),
    managed_image: str | None = None,
    managed_evaluator_sha256: str | None = None,
    managed_evaluator_tree_sha256: str | None = None,
    write_managed_manifest: bool = True,
    sandbox_image: str | None = None,
    environment: dict[str, str] | None = None,
    include_ignored_roots: bool = False,
    force_remove: bool = True,
    create_error: BaseException | None = None,
):
    task_bundle, task_source, evaluator_source = _workspace(tmp_path)
    if include_ignored_roots:
        for source in (task_source, evaluator_source):
            for ignored_name in (".git", ".venv", "__pycache__", "results"):
                (source / ignored_name).mkdir(parents=True)
    if managed_image is not None:
        python_bin = task_source / ".venv/bin/python"
        python_bin.parent.mkdir(parents=True)
        python_bin.write_text("copied runtime\n", encoding="utf-8")
        if write_managed_manifest:
            (task_source / ".agentic_runtime.json").write_text(json.dumps({
                "image_id": managed_image,
                "evaluator_sha256": managed_evaluator_sha256
                or hashlib.sha256((evaluator_source / "evaluate.py").read_bytes()).hexdigest(),
                "evaluator_tree_sha256": managed_evaluator_tree_sha256
                or managed_evaluator_tree_digest(evaluator_source),
                "requirements_sha256": (
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                ),
                "schema_version": 1,
            }), encoding="utf-8")

    commands: list[list[str]] = []

    def create(command: list[str]) -> None:
        commands.append(command)
        if create_error is not None:
            raise create_error

    def launch(command: list[str], **kwargs):
        del kwargs
        assert command[:3] == ["docker", "start", "--attach"]
        return _CompletedSandbox(commands[0], correct=correct, returncode=returncode)

    with (
        patch("src.variants_pipeline.host_eval._create_sandbox", side_effect=create),
        patch("src.variants_pipeline.host_eval._force_remove_sandbox", return_value=force_remove),
        patch("src.variants_pipeline.host_eval.subprocess.Popen", side_effect=launch),
    ):
        result = _evaluate_program(
            task_bundle, task_source, evaluator_source, python_bin=python_bin,
            sandbox_image=sandbox_image or managed_image or "sha256:test",
            environment=environment or {"NCCL_DEBUG": "INFO", "VARIANT_ID": "ablation"},
            sandbox_gpus="all",
        )
    return task_bundle, result, commands[0]


def _assert_staging_discarded(program_dir: Path) -> None:
    assert not (program_dir / ".harness_results").exists()
    assert not (program_dir / "evaluation_manifest.json").exists()


# ── helpers: brokered contracted search ───────────────────────────


def _contract_workspace(
    tmp_path: Path, *, gpu: bool = False, metrics: list[str] | None = None,
    reset_sequence: list[int] | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    task_source = _write_contract_task(tmp_path)
    contract_path = task_source / "task_contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if gpu:
        payload["compute"]["search"].update(
            accelerator="gpu", accelerator_model="H100", gpu_count=1, evaluator_gpu_count=1,
            candidate_gpu_count=0,
        )
    if metrics is not None:
        payload["metrics"]["search"] = [{"name": n, "data_role": "development"} for n in metrics]
    if reset_sequence is not None:
        payload["protocol"]["search_reset_sequence"] = reset_sequence
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    program_dir = tmp_path / "bundle" / "program_1"
    make_program_dir(program_dir)
    return task_source, load_task_contract(contract_path), program_dir


def _brokered(task_source: Path, contract, program_dir: Path, **overrides):
    kwargs = dict(
        task_source_dir=task_source, evaluator_source_dir=task_source, program_dir=program_dir,
        entrypoint=program_dir / "src" / "program.py",
        python_bin=task_source / ".venv" / "bin" / "python", sandbox_image=IMAGE,
        timeout_seconds=10, environment={}, sandbox_gpus=None, contract=contract,
    )
    kwargs.update(overrides)
    return _evaluate_brokered_search_locked(**kwargs)


def _managed():
    return patch("src.variants_pipeline.host_eval._uses_image_managed_runtime", return_value=True)


class _FakeBroker:
    calls = 2
    calls_before_first_reset = 0
    reset_seeds: tuple[int, ...] = (1,)
    peer_disconnects = 0

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _Lease:
    active = False
    quarantined = False

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *_args: object) -> None:
        self.active = False

    def prepare_runtime(self) -> None:
        assert self.active

    def wrap_command(self, command: list[str]) -> list[str]:
        assert self.active
        return command

    def quarantine(self) -> None:
        assert self.active
        self.quarantined = True


def _evaluator_writing(metrics: dict, correct: dict, commands: list[list[str]] | None = None):
    def run_evaluator(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        if commands is not None:
            commands.append(command)
        mount = next(
            v for v in command if v.startswith("type=bind,src=") and v.endswith(",dst=/results")
        )
        results = Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/results"))
        (results / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (results / "correct.json").write_text(json.dumps(correct), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    return run_evaluator


# ── legacy sandbox command and runtime mounts ─────────────────────


def test_evaluator_sandbox_is_sealed_masks_unsealed_roots_and_bounds_memory(tmp_path) -> None:
    # The evaluator runs in a sealed, networkless sandbox and writes a host manifest.
    task_bundle, result, command = _evaluate(tmp_path / "sealed")

    assert result.correct is True
    assert result.returncode == 0
    assert result.metrics["combined_score"] == 1.25
    assert command[:2] == ["docker", "create"]
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "--gpus" in command and command[command.index("--gpus") + 1] == "all"
    assert any(value.endswith(":/task-source:ro") for value in command)
    assert any(value.endswith(":/evaluator-source:ro") for value in command)
    assert any(value.endswith(":/candidate:ro") for value in command)
    assert not any(value.endswith(":/promoted:rw") for value in command)
    assert not any(value.endswith(":/results:rw") for value in command)
    assert "--tmpfs" in command
    assert any(value.startswith("/results:rw,") and "size=144m" in value for value in command)
    assert "/harness-source/src/variants_pipeline/sandbox_runner.py" in command
    assert "/usr/local/bin/python3" in command
    assert "-S" in command
    assert "DAC_OVERRIDE" in command
    assert "--memory" in command and command[command.index("--memory") + 1] == "64g"
    assert not any("CLAUDE_CODE_OAUTH_TOKEN" in value for value in command)
    assert "/usr:/usr:ro" not in command
    resolved_python = Path("/usr/bin/python3").resolve()
    assert f"{resolved_python}:{resolved_python}:ro" in command
    manifest = json.loads((task_bundle / "program_1" / "evaluation_manifest.json").read_text())
    assert manifest["producer"] == "sandbox_host_eval"
    assert manifest["execution_boundary"] == "sandbox"

    # Unsealed root directories are masked with tmpfs mounts.
    _, _, command = _evaluate(tmp_path / "masked", include_ignored_roots=True)
    tmpfs_targets = {
        command[index + 1].split(":", 1)[0]
        for index, argument in enumerate(command)
        if argument == "--tmpfs"
    }
    assert {
        f"/{root}/{name}"
        for root in ("task-source", "evaluator-source")
        for name in (".git", ".venv", "__pycache__", "results")
    } <= tmpfs_targets

    # A bounded memory override is accepted.
    _, _, command = _evaluate(
        tmp_path / "memory",
        environment={"VARIANT_EVAL_MEMORY_LIMIT": "96g", "VARIANT_ID": "control"},
    )
    assert command[command.index("--memory") + 1] == "96g"
    assert command[command.index("--memory-swap") + 1] == "96g"


def test_evaluator_runtime_mounts_follow_resolved_python_bounded_venv_and_managed_image(
    tmp_path: Path,
) -> None:
    # The resolved task python runtime is mounted.
    root = tmp_path / "resolved"
    runtime_root = root / "cpython-3.13"
    runtime_python = runtime_root / "bin" / "python3.13"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("runtime\n", encoding="utf-8")
    (runtime_root / "lib" / "python3.13").mkdir(parents=True)
    runtime_alias = runtime_python.with_name("python3")
    runtime_alias.symlink_to(runtime_python.name)
    task_python = root / "task-venv" / "bin" / "python"
    task_python.parent.mkdir(parents=True)
    task_python.symlink_to("python3")
    task_python.with_name("python3").symlink_to(runtime_alias)

    _, _, command = _evaluate(root, python_bin=task_python)

    assert f"{runtime_python}:{runtime_python}:ro" in command
    assert f"{runtime_alias}:{runtime_alias}:ro" in command
    runtime_stdlib = runtime_root / "lib" / "python3.13"
    assert f"{runtime_stdlib}:{runtime_stdlib}:ro" in command
    assert not any("python3.10" in value for value in command)

    # Copied venv runtime mounts are bounded to the declared home.
    root = tmp_path / "copied"
    runtime_root = root / "cpython-3.14"
    runtime_home = runtime_root / "bin"
    runtime_home.mkdir(parents=True)
    runtime_stdlib = runtime_root / "lib" / "python3.14"
    runtime_stdlib.mkdir(parents=True)
    task_venv = root / "copied-venv"
    task_python = task_venv / "bin" / "python"
    task_python.parent.mkdir(parents=True)
    task_python.write_text("copied runtime\n", encoding="utf-8")
    (task_venv / "pyvenv.cfg").write_text(
        f"home = {runtime_home}\nversion = 3.14.1\nexecutable = /home/operator/secrets/python\n",
        encoding="utf-8",
    )

    _, _, command = _evaluate(root, python_bin=task_python)

    assert f"{runtime_stdlib}:{runtime_stdlib}:ro" in command
    assert not any("/home/operator" in value for value in command)
    assert not any(value in {"/:/:ro", "/usr:/usr:ro"} for value in command)

    # An image-managed runtime does not mount the host stdlib.
    _, _, command = _evaluate(tmp_path / "managed", managed_image=IMAGE)

    assert not any("/usr/lib" in value for value in command)
    assert not any("libpython" in value for value in command)
    assert any(":/task-venv:ro" in value for value in command)
    assert command[command.index("--evaluator-python") + 1] == "/task-venv/bin/python"
    assert IMAGE in command

    # resolve_host_python preserves the venv symlink.
    task_source = tmp_path / "task"
    python = task_source / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to("/usr/bin/python3")
    assert resolve_host_python(task_source_dir=task_source, venv_dir=None) == python.absolute()


def test_evaluator_admission_rejects_managed_runtime_drift_and_root_identity(
    tmp_path, recovery_root,
) -> None:
    # A runtime admission failure removes the cid directory.
    with pytest.raises(ValueError, match="managed task runtime manifest is missing"):
        _evaluate(tmp_path / "missing_manifest", managed_image=IMAGE, write_managed_manifest=False)
    assert not list(recovery_root.glob("*.recovery"))

    # Image, evaluator, and evaluator-tree drift are rejected.
    drift_cases: list[tuple[str, dict[str, Any], str]] = [
        ("image", {"sandbox_image": "sha256:" + "b" * 64}, "does not match"),
        ("evaluator", {"managed_evaluator_sha256": "0" * 64}, "evaluator has changed"),
        ("tree", {"managed_evaluator_tree_sha256": "0" * 64}, "evaluator tree has changed"),
    ]
    for name, overrides, match in drift_cases:
        with pytest.raises(ValueError, match=match):
            _evaluate(tmp_path / name, managed_image=IMAGE, **overrides)

    # A root host identity is rejected.
    with (
        patch("src.variants_pipeline.host_eval.os.getuid", return_value=0),
        pytest.raises(ValueError, match="non-root host user"),
    ):
        _evaluate(tmp_path / "root")


def test_manifest_write_and_cleanup_errors_discard_staging_without_masking_outcomes(
    tmp_path, recovery_root,
) -> None:
    # An artifact cleanup error does not skip cid cleanup.
    with (
        patch(
            "src.variants_pipeline.host_eval._write_json",
            side_effect=PermissionError("manifest denied"),
        ),
        patch(
            "src.variants_pipeline.host_eval.discard_evaluation_artifacts",
            side_effect=PermissionError("artifact cleanup denied"),
        ),
        pytest.raises(PermissionError, match="artifact cleanup denied"),
    ):
        _evaluate(tmp_path / "cleanup")

    assert not list(recovery_root.glob("*.recovery"))

    # An initial manifest write error discards staging.
    root = tmp_path / "initial"
    with (
        patch(
            "src.variants_pipeline.host_eval._write_json",
            side_effect=PermissionError("manifest denied"),
        ),
        pytest.raises(PermissionError, match="manifest denied"),
    ):
        _evaluate(root)

    _assert_staging_discarded(root / "task_bundle" / "program_1")

    # A cid cleanup error does not mask success.
    original_rmdir = Path.rmdir

    def fail_cid_rmdir(path: Path) -> None:
        if path.name.endswith(".recovery"):
            raise PermissionError("cid cleanup denied")
        original_rmdir(path)

    try:
        with patch.object(Path, "rmdir", fail_cid_rmdir):
            _, result, _ = _evaluate(tmp_path / "cid")
    finally:
        for recovery_dir in recovery_root.glob("*.recovery"):
            original_rmdir(recovery_dir)

    assert result.returncode == 0

    # A final manifest write error discards staging and results.
    original_write_json = host_eval._write_json

    def fail_final_write(path: Path, payload: dict[str, object]) -> None:
        if "finished_at" in payload:
            raise PermissionError("manifest denied")
        original_write_json(path, payload)

    for name, returncode in (("promoted_results", 0), ("nonzero_exit", 7)):
        root = tmp_path / f"final_{name}"
        with (
            patch("src.variants_pipeline.host_eval._write_json", side_effect=fail_final_write),
            pytest.raises(PermissionError, match="manifest denied"),
        ):
            _evaluate(root, returncode=returncode)

        program_dir = root / "task_bundle" / "program_1"
        for artifact in (
            ".harness_results", "metrics.json", "correct.json", "evaluation_manifest.json",
        ):
            assert not (program_dir / artifact).exists(), name


# ── brokered contracted search ────────────────────────────────────


def test_contracted_search_uses_disjoint_brokered_mounts_and_bypasses_legacy_container(
    tmp_path: Path, _mock_sandbox_creation: MagicMock
) -> None:
    # Evaluator and candidate containers get disjoint mounts.
    root = tmp_path / "mounts"
    root.mkdir()
    task_source = _write_contract_task(root)
    evaluator_source = root / "sealed-evaluator"
    evaluator_source.mkdir()
    (evaluator_source / "evaluate.py").write_text("# sealed search evaluator\n", encoding="utf-8")
    program_dir = root / "bundle" / "program_1"
    make_program_dir(program_dir)
    results_dir = root / "trusted-results"
    results_dir.mkdir()
    contract = load_task_contract(task_source / "task_contract.json")

    evaluator, candidate = _brokered_search_commands(
        task_source_dir=task_source, evaluator_source_dir=evaluator_source,
        program_dir=program_dir, entrypoint=program_dir / "src" / "program.py",
        results_dir=results_dir, broker_dir=root / "broker", sandbox_image=IMAGE,
        contract=contract, environment={},
    )

    evaluator_rendered = " ".join(evaluator)
    candidate_rendered = " ".join(candidate)
    assert f"src={task_source},dst=/task,readonly" in evaluator_rendered
    assert f"src={evaluator_source},dst=/evaluator-source,readonly" in evaluator_rendered
    assert f"src={results_dir},dst=/results" in evaluator_rendered
    assert "AGENTIC_CANDIDATE_SOCKET=/broker/candidate.sock" in evaluator_rendered
    assert str(program_dir) not in evaluator_rendered
    assert "/broker-code/candidate_proxy.py" in evaluator_rendered

    assert f"src={program_dir},dst=/program,readonly" in candidate_rendered
    assert f"src={task_source / '.venv'},dst=/task-venv,readonly" in candidate_rendered
    assert f"src={task_source},dst=/task,readonly" not in candidate_rendered
    assert str(results_dir) not in candidate_rendered
    assert "candidate.sock" not in candidate_rendered
    assert "exchange" not in candidate_rendered
    assert str(evaluator_source) not in candidate_rendered
    assert "/var/run/docker.sock" not in candidate_rendered
    assert "--network none" in candidate_rendered
    assert "src/investigation,readonly" in candidate_rendered

    # A contracted task routes around the legacy single container.
    root = tmp_path / "routing"
    root.mkdir()
    task_source = _write_contract_task(root)
    python_bin = task_source / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("managed runtime\n", encoding="utf-8")
    task_bundle = root / "bundle"
    make_program_dir(task_bundle / "program_1")
    expected = host_eval.HostEvaluationResult(0, True, {"development_score": 0.5})

    with patch(
        "src.variants_pipeline.host_eval._evaluate_brokered_search_locked", return_value=expected,
    ) as brokered:
        result = host_eval._evaluate_variant_program_locked(
            task_bundle_dir=task_bundle, evaluator_source_dir=task_source,
            task_source_dir=task_source, program_index=1, python_bin=python_bin,
            sandbox_image=IMAGE, timeout_seconds=60, environment={}, sandbox_gpus=None,
            lease=host_eval._SandboxLease("test", root / "recovery"),
        )

    assert result == expected
    brokered.assert_called_once()
    assert not _mock_sandbox_creation.called


def test_contracted_search_promotes_only_contract_conformant_evaluator_results(tmp_path) -> None:
    # Only evaluator results are promoted, with broker options from the contract.
    task_source, contract, program_dir = _contract_workspace(
        tmp_path / "promoted", metrics=["combined_score"], reset_sequence=[1, 1],
    )
    candidate_commands: list[list[str]] = []
    broker_options: list[dict[str, object]] = []
    evaluator_commands: list[list[str]] = []

    class RecordingBroker(_FakeBroker):
        reset_seeds = (1, 1)

        def __init__(self, command: list[str], *_: object, **kwargs: object) -> None:
            candidate_commands.append(command)
            broker_options.append(kwargs)

    with (
        patch("src.variants_pipeline.host_eval.CandidateBroker", RecordingBroker),
        patch(
            "src.variants_pipeline.host_eval.run_bounded_process",
            side_effect=_evaluator_writing(
                {"combined_score": 0.75}, {"correct": True}, evaluator_commands,
            ),
        ),
        patch("src.variants_pipeline.host_eval._force_remove_container"),
        _managed(),
    ):
        result = _brokered(
            task_source, contract, program_dir, environment={"VARIANT_ID": "isolated"},
        )

    assert result.correct is True
    assert result.metrics == {"combined_score": 0.75}
    assert len(evaluator_commands) == len(candidate_commands) == 1
    assert broker_options[0]["max_rpc_calls"] == contract.compute.search.max_candidate_rpc_calls
    assert broker_options[0]["expected_reset_seeds"] == contract.protocol.search_reset_sequence
    candidate_rendered = " ".join(candidate_commands[0])
    assert str(program_dir) in candidate_rendered
    assert "/results" not in candidate_rendered
    assert not (program_dir / ".harness_results").exists()
    assert (program_dir / "metrics.json").is_file()

    # Result schema drift is rejected and nothing is promoted.
    for index, (metrics, correct, message) in enumerate((
        (
            {"combined_score": 0.75},
            {"correct": True},
            "search result metric names must exactly match the task contract",
        ),
        (
            {"combined_score": 0.75, "required_score": 0.5},
            {"correct": True, "unexpected": True},
            "correct.json must contain exactly one boolean key: correct",
        ),
        (
            {"combined_score": float("inf"), "required_score": 0.5},
            {"correct": True},
            "search result metrics must be finite numbers",
        ),
    )):
        task_source, contract, program_dir = _contract_workspace(
            tmp_path / f"drift_{index}", metrics=["combined_score", "required_score"],
        )
        with (
            patch("src.variants_pipeline.host_eval.CandidateBroker", _FakeBroker),
            patch(
                "src.variants_pipeline.host_eval.run_bounded_process",
                side_effect=_evaluator_writing(metrics, correct),
            ),
            patch("src.variants_pipeline.host_eval._force_remove_container"),
            _managed(),
            pytest.raises(EvaluatorOutputContractError, match=message),
        ):
            _brokered(task_source, contract, program_dir)

        assert not (program_dir / "metrics.json").exists(), message


def test_contracted_search_classifies_timeouts_broker_and_evaluator_runtime_failures(
    tmp_path: Path,
) -> None:
    # Evaluator timeouts keep their taxonomy and the broker is exited.
    task_source, contract, program_dir = _contract_workspace(tmp_path / "timeout")
    broker_exited = False

    class ExitTrackingBroker(_FakeBroker):
        def __exit__(self, *_args: object) -> None:
            nonlocal broker_exited
            broker_exited = True

    with (
        patch("src.variants_pipeline.host_eval.CandidateBroker", ExitTrackingBroker),
        patch(
            "src.variants_pipeline.host_eval.run_bounded_process",
            side_effect=ProcessTimeoutError("deadline exceeded"),
        ),
        patch("src.variants_pipeline.host_eval._force_remove_container") as cleanup,
        _managed(),
        pytest.raises(SandboxEvaluationTimeout),
    ):
        _brokered(task_source, contract, program_dir)

    cleanup.assert_called_once()
    assert broker_exited is True
    failure = json.loads((program_dir / "evaluation_failure.json").read_text())
    assert failure["outcome"] == "timeout"
    assert failure["wall_clock_limit_seconds"] == 10
    assert not (program_dir / "metrics.json").exists()

    # Broker infrastructure failures propagate.
    task_source, contract, program_dir = _contract_workspace(tmp_path / "broker")

    class FailedBroker(_FakeBroker):
        def __enter__(self):
            raise CandidateBrokerInfrastructureError("docker launch unavailable")

    with (
        patch("src.variants_pipeline.host_eval.CandidateBroker", FailedBroker),
        _managed(),
        pytest.raises(CandidateBrokerInfrastructureError, match="docker launch unavailable"),
    ):
        _brokered(task_source, contract, program_dir)

    # Evaluator runtime failures are classified as infrastructure and quarantine the lease.
    for failure_site in ("launch", "removal", "removal_interrupt", "broker_cleanup"):
        task_source, contract, program_dir = _contract_workspace(
            tmp_path / f"runtime_{failure_site}", gpu=True,
        )
        lease = _Lease()

        class Broker(_FakeBroker):
            calls = 1

            def __exit__(self, *_args: object, failure_site=failure_site) -> None:
                if failure_site == "broker_cleanup":
                    raise CandidateContainerCleanupError(
                        "candidate container cleanup failed: daemon unavailable; "
                        "recovery metadata retained"
                    )

        run_effect = OSError("docker unavailable") if failure_site == "launch" else None
        remove_effect = {
            "removal": FinalEvaluationError("could not remove final evaluator container"),
            "removal_interrupt": KeyboardInterrupt(),
        }.get(failure_site)
        expected_error = (
            KeyboardInterrupt if failure_site == "removal_interrupt"
            else CandidateBrokerInfrastructureError
        )
        with (
            patch("src.variants_pipeline.host_eval.CandidateBroker", Broker),
            patch(
                "src.variants_pipeline.host_eval.acquire_gpu_allocation_lease", return_value=lease,
            ),
            patch("src.variants_pipeline.host_eval.run_bounded_process", side_effect=run_effect),
            patch(
                "src.variants_pipeline.host_eval._force_remove_container",
                side_effect=remove_effect,
            ),
            _managed(),
            pytest.raises(expected_error),
        ):
            _brokered(task_source, contract, program_dir, sandbox_gpus="device=GPU-aaaa")

        assert lease.quarantined is (
            failure_site in {"removal", "removal_interrupt", "broker_cleanup"}
        ), failure_site


def test_contracted_gpu_search_allocation_selector_and_budget_handling(tmp_path: Path) -> None:
    # The bounded GPU selector preserves docker CSV quoting for multiple devices.
    assert _bounded_gpu_selector('"device=GPU-a1,GPU-b2"', 2) == '"device=GPU-a1,GPU-b2"'

    # A busy allocation is classified as infrastructure.
    task_source, contract, program_dir = _contract_workspace(tmp_path / "busy", gpu=True)
    with (
        _managed(),
        patch(
            "src.variants_pipeline.host_eval.acquire_gpu_allocation_lease",
            side_effect=GpuLeaseTimeout("GPU allocation remained busy until the deadline: GPU-aaaa"),
        ),
        pytest.raises(CandidateBrokerInfrastructureError, match="gpu_allocation_busy"),
    ):
        _brokered(task_source, contract, program_dir, sandbox_gpus="device=GPU-aaaa")

    # A malformed selector is classified as infrastructure.
    task_source, contract, program_dir = _contract_workspace(tmp_path / "malformed", gpu=True)
    with (
        _managed(),
        pytest.raises(CandidateBrokerInfrastructureError, match="gpu_allocation_invalid"),
    ):
        _brokered(task_source, contract, program_dir, sandbox_gpus='"device=GPU-aaaa')

    # Waiting for a GPU consumes the evaluation budget.
    task_source, contract, program_dir = _contract_workspace(tmp_path / "budget", gpu=True)
    clock = [100.0]
    broker_timeouts: list[float] = []
    process_timeouts: list[float] = []

    class Broker(_FakeBroker):
        def __init__(self, *_args: object, rpc_timeout_seconds: float, **_kwargs: object) -> None:
            broker_timeouts.append(rpc_timeout_seconds)

        def __enter__(self):
            clock[0] = 105.0
            return self

    def acquire(*_args: object, deadline: float, **_kwargs: object) -> _Lease:
        assert deadline == pytest.approx(110.0)
        clock[0] = 104.0
        return _Lease()

    def fail_launch(*_args: object, timeout_seconds: float, **_kwargs: object) -> None:
        process_timeouts.append(timeout_seconds)
        raise OSError("stop after observing timeout")

    with (
        patch("src.variants_pipeline.host_eval.time.monotonic", side_effect=lambda: clock[0]),
        patch("src.variants_pipeline.host_eval.acquire_gpu_allocation_lease", side_effect=acquire),
        patch("src.variants_pipeline.host_eval.CandidateBroker", Broker),
        patch("src.variants_pipeline.host_eval.run_bounded_process", side_effect=fail_launch),
        patch("src.variants_pipeline.host_eval._force_remove_container"),
        _managed(),
        pytest.raises(CandidateBrokerInfrastructureError),
    ):
        _brokered(task_source, contract, program_dir, sandbox_gpus="device=GPU-aaaa")

    assert broker_timeouts == pytest.approx([6.0])
    assert process_timeouts == pytest.approx([5.0])

    # Admission exhaustion is infrastructure.
    task_source, contract, program_dir = _contract_workspace(tmp_path / "exhausted", gpu=True)
    clock = [100.0]

    def acquire_until_deadline(*_args: object, deadline: float, **_kwargs: object) -> _Lease:
        assert deadline == pytest.approx(110.0)
        clock[0] = deadline
        return _Lease()

    with (
        patch("src.variants_pipeline.host_eval.time.monotonic", side_effect=lambda: clock[0]),
        patch(
            "src.variants_pipeline.host_eval.acquire_gpu_allocation_lease",
            side_effect=acquire_until_deadline,
        ),
        _managed(),
        pytest.raises(CandidateBrokerInfrastructureError, match="gpu_allocation_busy"),
    ):
        _brokered(task_source, contract, program_dir, sandbox_gpus="device=GPU-aaaa")


def test_sandbox_lease_recovers_stale_sandboxes_and_serializes_same_program(
    tmp_path, recovery_root,
) -> None:
    # The search lease discovers and recovers a retained broker directory.
    lease_root = tmp_path / "lease"
    identity = hashlib.sha256(str(lease_root).encode()).hexdigest()[:24]
    lease = host_eval._SandboxLease(identity, lease_root / "recovery")
    broker_dir = host_eval._search_broker_directory(lease)
    broker_dir.mkdir(mode=0o700)

    def recover(path: Path) -> None:
        assert path == broker_dir
        path.rmdir()

    with patch(
        "src.variants_pipeline.host_eval._recover_broker_directory", side_effect=recover
    ) as recovery:
        assert host_eval._recover_stale_sandbox(lease) is True

    recovery.assert_called_once_with(broker_dir)
    assert not broker_dir.exists()

    # The next evaluation recovers a stale named sandbox.
    task_bundle, task_source, evaluator_source = _workspace(tmp_path / "stale")
    program_dir = task_bundle / "program_1"
    identity = host_eval._sandbox_identity(program_dir)
    stale_dir = recovery_root / f"{identity}.recovery"
    stale_dir.mkdir(mode=0o700)
    stale_name = f"agentic-eval-{os.getuid()}-{identity}-{'a' * 12}"
    stale_name_path = stale_dir / "container.name"
    stale_name_path.write_text(stale_name, encoding="ascii")
    stale_name_path.chmod(0o600)
    (program_dir / ".harness_results").mkdir()
    (program_dir / "evaluation_manifest.json").write_text("{}", encoding="utf-8")

    with (
        _popen(_CompletedSandbox),
        patch("src.variants_pipeline.host_eval.time.sleep"),
        patch("src.variants_pipeline.host_eval._stop_sandbox", return_value=True) as stop,
    ):
        result = _evaluate_program(task_bundle, task_source, evaluator_source)

    assert result.returncode == 0
    assert stop.call_count == 5
    stop.assert_called_with(None, stale_name)
    assert not stale_dir.exists()

    # The same-program sandbox lease serializes evaluations.
    program_dir = tmp_path / "program_1"
    program_dir.mkdir()
    acquired = threading.Event()

    def acquire_second_lease() -> None:
        with host_eval._sandbox_lease(program_dir):
            acquired.set()

    with host_eval._sandbox_lease(program_dir):
        thread = threading.Thread(target=acquire_second_lease)
        thread.start()
        assert acquired.wait(timeout=0.1) is False
    thread.join(timeout=2)

    assert thread.is_alive() is False
    assert acquired.is_set()


# ── sandbox runner diagnostics ────────────────────────────────────


def test_sandbox_runner_sanitizes_diagnostics_caps_output_and_terminates_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Terminal controls are escaped.
    assert _sanitize_diagnostic(b"ok\n\x1b]52;clipboard\x07") == "ok\n\\u001b]52;clipboard\\u0007"

    # The diagnostic cap applies after sanitization.
    class _NoisyProcess:
        stdout = io.BytesIO(b"\x1b" * (1024 * 1024 + 1))

    diagnostic = io.StringIO()
    with monkeypatch.context() as m:
        m.setattr("src.variants_pipeline.sandbox_runner.sys.stderr", diagnostic)
        _stream_bounded_output(_NoisyProcess())  # type: ignore[arg-type]

    output = diagnostic.getvalue()
    marker = "\n[evaluator output truncated after 1 MiB]\n"
    assert output.endswith(marker)
    assert len(output.removesuffix(marker).encode("ascii")) == 1024 * 1024
    assert "\x1b" not in output

    # Evaluator completion terminates a child holding stdout.
    child_code = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "print('parent complete', flush=True)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def terminate_process_group(evaluator_pid: int) -> None:
        try:
            os.killpg(evaluator_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    started_at = time.monotonic()
    with patch.object(
        sandbox_runner, "_terminate_remaining_processes", side_effect=terminate_process_group,
    ):
        returncode = sandbox_runner._wait_for_evaluator(process)

    assert returncode == 0
    assert time.monotonic() - started_at < 2


# ── results promotion, failures, and teardown ─────────────────────


def test_evaluator_rejects_agent_authored_artifacts_nonzero_exit_and_invalid_frames(
    tmp_path: Path,
) -> None:
    # Agent-authored evaluator artifacts are rejected.
    for name, artifact in (("metrics", "metrics.json"), ("raw_runs", "runs")):
        task_bundle, task_source, evaluator_source = _workspace(tmp_path / name)
        target = task_bundle / "program_1" / artifact
        if artifact.endswith(".json"):
            target.write_text("{}")
        else:
            target.mkdir()
        with pytest.raises(ValueError, match="evaluator artifact"):
            _evaluate_program(task_bundle, task_source, evaluator_source)

    # A non-zero sandbox exit is reported and staging discarded.
    root = tmp_path / "nonzero"
    with pytest.raises(RuntimeError, match="exited with code 7"):
        _evaluate(root, returncode=7)
    _assert_staging_discarded(root / "task_bundle" / "program_1")

    # Unexpected sandbox output is rejected and cleaned.
    task_bundle, task_source, evaluator_source = _workspace(tmp_path / "frame")
    with _popen(_InvalidFrameSandbox), pytest.raises(RuntimeError, match="invalid promotion frame"):
        _evaluate_program(task_bundle, task_source, evaluator_source)

    program_dir = task_bundle / "program_1"
    _assert_staging_discarded(program_dir)
    assert not (program_dir / "analysis").exists()


def test_evaluator_promotes_analysis_outputs_and_trusts_correct_json(tmp_path: Path) -> None:
    # correct.json is authoritative even on a zero exit.
    _, result, _ = _evaluate(tmp_path / "incorrect", correct=False)
    assert result.returncode == 0
    assert result.correct is False

    # Analysis outputs are promoted.
    task_bundle, task_source, evaluator_source = _workspace(tmp_path / "promoted")
    with _popen(_CompletedSandbox):
        _evaluate_program(task_bundle, task_source, evaluator_source)

    program_dir = task_bundle / "program_1"
    assert (program_dir / "_results.json").is_file()
    assert sorted(path.name for path in (program_dir / "analysis").iterdir()) == [
        "metric_values.csv", "run_summary.csv", "timeseries.csv",
    ]


def test_failed_teardown_retains_recovery_metadata(tmp_path, recovery_root) -> None:
    # Failed removal, ambiguous creation, and interrupted teardown all retain metadata.
    for name, overrides, stop_interrupt, expected in (
        ("start_failure_removal_fails", {"returncode": 7, "force_remove": False}, False,
         SandboxTeardownError),
        ("ambiguous_create_failure",
         {"create_error": subprocess.TimeoutExpired(["docker", "create"], 60)}, False,
         SandboxTeardownError),
        ("interrupted_teardown",
         {"create_error": subprocess.TimeoutExpired(["docker", "create"], 60)}, True,
         KeyboardInterrupt),
    ):
        for stale in recovery_root.glob("*.recovery"):
            shutil.rmtree(stale)
        stop = (
            patch("src.variants_pipeline.host_eval._stop_sandbox", side_effect=KeyboardInterrupt)
            if stop_interrupt else nullcontext()
        )
        raises = (
            pytest.raises(expected, match="recovery metadata retained")
            if expected is SandboxTeardownError else pytest.raises(expected)
        )
        with stop, raises:
            _evaluate(tmp_path / name, **overrides)

        recovery_dirs = list(recovery_root.glob("*.recovery"))
        assert len(recovery_dirs) == 1, name
        assert (recovery_dirs[0] / "container.name").is_file(), name

    # A timeout cleans staging even when stopping the sandbox fails.
    class _TimedOutSandbox(_CompletedSandbox):
        def communicate(self, timeout: int | None = None):
            raise subprocess.TimeoutExpired(self.command, timeout or 0)

        def poll(self) -> int | None:
            return None

    task_bundle, task_source, evaluator_source = _workspace(tmp_path / "timeout")
    with (
        _popen(_TimedOutSandbox),
        patch("src.variants_pipeline.host_eval._stop_sandbox", return_value=False),
        pytest.raises(SandboxTeardownError, match="recovery metadata retained"),
    ):
        _evaluate_program(task_bundle, task_source, evaluator_source)

    program_dir = task_bundle / "program_1"
    _assert_staging_discarded(program_dir)
    identity = host_eval._sandbox_identity(program_dir)
    assert (recovery_root / f"{identity}.recovery" / "container.name").is_file()


def test_interrupt_and_sigterm_stop_sandbox_and_clean_staging(tmp_path: Path) -> None:
    # KeyboardInterrupt stops the sandbox and cleans staging.
    class _InterruptedSandbox(_CompletedSandbox):
        def communicate(self, timeout: int | None = None):
            raise KeyboardInterrupt

        def poll(self) -> int | None:
            return None

    task_bundle, task_source, evaluator_source = _workspace(tmp_path / "interrupt")
    with (
        _popen(_InterruptedSandbox),
        patch("src.variants_pipeline.host_eval._stop_sandbox") as stop_sandbox,
        pytest.raises(KeyboardInterrupt),
    ):
        _evaluate_program(task_bundle, task_source, evaluator_source)

    stop_sandbox.assert_called_once()
    _assert_staging_discarded(task_bundle / "program_1")

    # SIGTERM stops the sandbox before exiting.
    class _SignaledSandbox(_CompletedSandbox):
        def communicate(self, timeout: int | None = None):
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("SIGTERM handler did not interrupt communicate")

        def poll(self) -> int | None:
            return None

    task_bundle, task_source, evaluator_source = _workspace(tmp_path / "sigterm")
    with (
        _popen(_SignaledSandbox),
        patch("src.variants_pipeline.host_eval._stop_sandbox") as stop_sandbox,
        pytest.raises(SystemExit) as raised,
    ):
        _evaluate_program(task_bundle, task_source, evaluator_source)

    assert raised.value.code == 128 + signal.SIGTERM
    stop_sandbox.assert_called_once()
    _assert_staging_discarded(task_bundle / "program_1")


def test_cleanup_discards_symlinks_and_read_only_trees_without_touching_targets(tmp_path):
    # A symlinked staging directory is unlinked without changing its target.
    target = tmp_path / "outside.txt"
    target.write_text("preserved", encoding="utf-8")
    target.chmod(0o640)
    program_dir = tmp_path / "program_1"
    program_dir.mkdir()
    link = program_dir / ".harness_results"
    link.symlink_to(target)

    discard_evaluation_artifacts(program_dir)

    assert not link.exists()
    assert not link.is_symlink()
    assert target.read_text(encoding="utf-8") == "preserved"
    assert target.stat().st_mode & 0o777 == 0o640

    # Read-only trees with nested symlinks or files are removed.
    for nested_entry in ("symlink", "file"):
        root = tmp_path / nested_entry
        root.mkdir()
        target = root / "outside.txt"
        target.write_text("preserved", encoding="utf-8")
        target.chmod(0o640)
        program_dir = root / "program_1"
        tree = program_dir / ".harness_results"
        nested = tree / "nested"
        nested.mkdir(parents=True)
        if nested_entry == "symlink":
            (nested / "link").symlink_to(target)
        else:
            (nested / "result.json").write_text("{}", encoding="utf-8")
        nested.chmod(0o500)

        try:
            discard_evaluation_artifacts(program_dir)
        finally:
            if nested.exists():
                nested.chmod(0o700)

        assert not tree.exists(), nested_entry
        assert target.read_text(encoding="utf-8") == "preserved", nested_entry
        assert target.stat().st_mode & 0o777 == 0o640, nested_entry


# ── sandbox_outputs: canonicalization and promotion frames ────────


def _results(tmp_path: Path, timeseries: dict[str, str] | None = None) -> Path:
    """Results dir with trusted summaries and optional timeseries.csv files per sub-layout."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "metrics.json").write_text(
        json.dumps({"combined_score": 4.5, "nested": {"loss": 0.25}}), encoding="utf-8",
    )
    (results_dir / "correct.json").write_text(json.dumps({"correct": True}), encoding="utf-8")
    for relative, rows in (timeseries or {}).items():
        analysis_dir = results_dir / relative / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "timeseries.csv").write_text(_TS_HEADER + rows, encoding="utf-8")
    return results_dir


def test_canonicalization_replaces_forged_analysis_and_validates_timeseries(tmp_path) -> None:
    # Evaluator-authored analysis is always replaced with trusted identity.
    results_dir = _results(tmp_path / "forged")
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "run_summary.csv").write_text(
        "evaluation_id,variant_id\nforged,forged\n", encoding="utf-8"
    )

    canonicalize_results(results_dir, "program_1", ENVIRONMENT)

    rows = list(csv.DictReader((analysis_dir / "run_summary.csv").open()))
    assert rows[0]["evaluation_id"] == "eval-1"
    assert rows[0]["variant_id"] == "control_best"
    assert rows[0]["program_dir"] == "program_1"
    assert rows[0]["results_dir"] == "analysis"

    # Timeseries are canonicalized with trusted identity across layouts.
    for name, timeseries, expected in (
        ("header_only_without_scratch", {}, []),
        ("flat", {".": "forged,forged,loss,1,0.5\n"}, [("loss", "1", "0.5")]),
        (
            "missing_values_omitted",
            {"runs/run_000": "forged,forged,epsilon,1,nan\nforged,forged,loss,1,0.5\n"},
            [("runs/run_000/loss", "1", "0.5")],
        ),
        (
            "booleans_binary",
            {".": "forged,forged,is_valid,1,True\nforged,forged,is_warmup,2,false\n"},
            [("is_valid", "1", "1.0"), ("is_warmup", "2", "0.0")],
        ),
        (
            "scratch_layouts_merged",
            {"runs/run_000": "forged,forged,loss,2,0.125\n", "rep_1": "forged,forged,loss,2,0.125\n"},
            [("rep_1/loss", "2", "0.125"), ("runs/run_000/loss", "2", "0.125")],
        ),
    ):
        results_dir = _results(tmp_path / name, timeseries)

        canonicalize_results(results_dir, "program_1", ENVIRONMENT)

        text = (results_dir / "analysis/timeseries.csv").read_text()
        rows = list(csv.DictReader(io.StringIO(text)))
        assert all(
            (r["evaluation_id"], r["variant_id"]) == ("eval-1", "control_best") for r in rows
        ), name
        assert sorted((r["series_name"], r["step"], r["value"]) for r in rows) == expected, name
        if not expected:
            assert text == _TS_HEADER, name

    # Invalid timeseries are rejected.
    for name, relative, row, match in (
        ("name", "rep_0", "forged,forged,=FORMULA(),2.5,nan\n", "invalid series name"),
        *[
            (value, ".", f"forged,forged,loss,1,{value}\n", "analysis value")
            for value in ("inf", "-inf", "not-a-number")
        ],
    ):
        results_dir = _results(tmp_path / "invalid" / name, {relative: row})
        with pytest.raises(RuntimeError, match=match):
            canonicalize_results(results_dir, "program_1", ENVIRONMENT)


def test_promotion_frames_are_validated_bounded_and_tamper_evident(tmp_path, monkeypatch) -> None:
    # Promoted analysis is semantically validated against trusted metadata and metrics.
    results_dir = _results(tmp_path / "validated")
    canonicalize_results(results_dir, "program_1", ENVIRONMENT)

    promoted_dir = tmp_path / "promoted"
    promoted_dir.mkdir()
    copy_promotable_results(results_dir, promoted_dir)
    summary = promoted_dir / "analysis/run_summary.csv"
    summary.write_text(summary.read_text().replace("eval-1", "forged", 1))
    with pytest.raises(RuntimeError, match="trusted metadata"):
        validate_promotable_results(promoted_dir, ENVIRONMENT, "program_1")

    promoted_metrics = tmp_path / "promoted_metrics"
    promoted_metrics.mkdir()
    copy_promotable_results(results_dir, promoted_metrics)
    metric_values = promoted_metrics / "analysis/metric_values.csv"
    metric_values.write_text(metric_values.read_text().replace("4.5,True", "999.0,True"))
    with pytest.raises(RuntimeError, match="does not match metrics.json"):
        validate_promotable_results(promoted_metrics, ENVIRONMENT, "program_1")

    # The promotion frame round-trips and rejects surrounding bytes.
    results_dir = _results(tmp_path / "frame")
    promoted_dir = tmp_path / "frame" / "promoted"
    promoted_dir.mkdir()
    canonicalize_results(results_dir, "program_1", ENVIRONMENT)

    frame = build_promotable_frame(results_dir)
    extract_promotable_frame(frame, promoted_dir)
    validate_promotable_results(promoted_dir, ENVIRONMENT, "program_1")

    for index, forged in enumerate((b"prefix" + frame, frame + b"trailing")):
        with pytest.raises(RuntimeError, match="invalid promotion frame"):
            extract_promotable_frame(forged, tmp_path / f"rejected_{index}")

    # The raw output size limit is enforced before consolidation.
    results_dir = _results(tmp_path / "large")
    (results_dir / "large.bin").write_bytes(b"12345")
    monkeypatch.setattr(sandbox_outputs, "MAX_SANDBOX_OUTPUT_BYTES", 4)
    with pytest.raises(RuntimeError, match="64 MiB"):
        canonicalize_results(results_dir, "program_1", ENVIRONMENT)
