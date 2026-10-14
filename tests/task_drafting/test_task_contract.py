"""Task contract loading, lint, digest sync, scaffold files, and evaluator output contract."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from collections.abc import Callable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import pytest

from src.investigation import task_runtime
from src.investigation.evaluator_outputs import (
    EvaluatorOutputContractError,
    validate_evaluator_outputs,
)
from src.investigation.files import (
    missing_task_scaffold_files,
    read_task_sections,
    task_copy_path_is_ignored,
)
from src.investigation.task_contract import (
    MAX_TASK_CONTRACT_BYTES,
    TaskContractError,
    load_task_contract,
    prune_generated_bytecode,
    task_scaffold_fingerprint,
)
from src.investigation.task_contract_lint import task_contract_preflight_findings as _lint
from src.task_drafting.contract_sync import ContractSyncError, synchronize_declared_digests
from tests.final_evaluation.test_evaluation_evidence import evidence

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_contract(task_dir: Path) -> dict[str, object]:
    task_path = task_dir / "task.md"
    return {
        "schema_version": 1,
        "canonical_task": {"path": "task.md", "sha256": _sha256(task_path)},
        "method": {"class": "learned_optimizer", "description": "Meta-trained update rule."},
        "candidate": {
            "interface": "python_rpc_v1:handle",
            "editable_paths": ["program_0/src"],
        },
        "evaluators": {"search": "evaluate.py", "final": "evaluate_final.py"},
        "resources": {
            "search_root": "resources/search",
            "final_root": "resources/final",
            "public_artifact_bundle": None,
        },
        "data": {
            "search_roles": ["development", "validation"],
            "final_roles": ["final"],
        },
        "metrics": {
            "search": [{"name": "combined_score", "data_role": "validation"}],
            "final": [{"name": "test_score", "data_role": "final"}],
        },
        "upstream": [
            {
                "repository": "https://github.com/example/project",
                "revision": "v1.2.3",
                "package_version": "1.2.3",
                "license": "Apache-2.0",
                "entrypoints": ["project.train:main"],
                "patches": [],
            }
        ],
        "provenance": {
            "dependencies": [{"name": "numpy", "version": "2.1.0"}],
            "artifacts": [
                {
                    "role": "development",
                    "path": "resources/search/dev.json",
                    "sha256": _sha256(task_dir / "resources" / "search" / "dev.json"),
                    "source": "generated fixture",
                },
                {
                    "role": "final",
                    "path": "resources/final/test.json",
                    "sha256": _sha256(task_dir / "resources" / "final" / "test.json"),
                    "source": "generated fixture",
                },
            ],
        },
        "compute": {
            "search": {
                "accelerator": "gpu",
                "accelerator_model": "H100",
                "gpu_count": 1,
                "evaluator_gpu_count": 1,
                "candidate_gpu_count": 0,
                "cpu_cores": 4,
                "memory_gb": 32,
                "precision": "float32",
                "wall_clock_seconds": 600,
                "max_candidate_rpc_calls": 120_012,
            },
            "final": {
                "accelerator": "gpu",
                "accelerator_model": "H100",
                "gpu_count": 4,
                "evaluator_gpu_count": 2,
                "candidate_gpu_count": 2,
                "cpu_cores": 8,
                "memory_gb": 128,
                "precision": "float32",
                "wall_clock_seconds": 3600,
                "max_candidate_rpc_calls": 500_020,
            },
        },
        "protocol": {
            "search_seeds": [11, 12],
            "final_seeds": [21, 22],
            "final_reset_sequence": [21, 22],
            "baselines": ["Adam", "ELO"],
            "promotion_rule": "Freeze the highest validation score before final evaluation.",
            "search_proxies": [
                {
                    "name": "short-horizon",
                    "preserves": "optimizer ranking",
                    "omits": "long-horizon stability",
                }
            ],
        },
        "feedback": {
            "search_visible": ["development metrics", "validation aggregate"],
            "final_sealed": ["final examples", "per-instance final metrics"],
        },
        "publication": {
            "search_is_proxy": True,
            "requires_final_evaluation": True,
            "required_evidence": ["frozen artifact hashes", "sealed final metrics"],
        },
    }


def _write_scaffold(task_dir: Path) -> Path:
    task_dir.mkdir()
    (task_dir / "task.md").write_text("# Canonical task\n", encoding="utf-8")
    (task_dir / "evaluate.py").write_text("SEARCH = True\n", encoding="utf-8")
    (task_dir / "evaluate_final.py").write_text("FINAL = True\n", encoding="utf-8")
    (task_dir / "resources" / "search").mkdir(parents=True)
    (task_dir / "resources" / "search" / "dev.json").write_text("{}\n", encoding="utf-8")
    (task_dir / "resources" / "final").mkdir(parents=True)
    (task_dir / "resources" / "final" / "test.json").write_text("{}\n", encoding="utf-8")
    contract_path = task_dir / "task_contract.json"
    contract_path.write_text(json.dumps(_valid_contract(task_dir)), encoding="utf-8")
    return contract_path


def _reset(contract_path: Path) -> Path:
    """Rewrite the contract of an existing scaffold back to its valid form."""
    contract_path.write_text(json.dumps(_valid_contract(contract_path.parent)), encoding="utf-8")
    return contract_path


def _rewrite(contract_path: Path, mutate: Callable[..., object]) -> dict:
    """Load the contract JSON, apply `mutate` in place, and write it back."""
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    mutate(payload)
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _artifact(relative: str, task_dir: Path | None = None, **fields: object) -> dict:
    sha256 = _sha256(task_dir / relative) if task_dir is not None else "0" * 64
    return {"role": "development", "path": relative, "sha256": sha256, "source": "fixture",
            **fields}


def _requirements(contract_path: Path, text: str) -> Path:
    path = contract_path.parent / "requirements.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _sync(contract_path: Path) -> bool:
    return synchronize_declared_digests(
        contract_path, canonical_task=contract_path.parent / "task.md"
    )


@pytest.fixture
def contract_path(tmp_path: Path) -> Path:
    return _write_scaffold(tmp_path / "task")


# ---------------------------------------------------------------------------
# Strict contract loading
# ---------------------------------------------------------------------------


def test_loads_valid_contract_variants(contract_path: Path) -> None:
    # a valid contract loads and the canonical task digest is verified
    contract = load_task_contract(contract_path)
    assert contract.schema_version == 1
    assert contract.canonical_task.path == "task.md"
    assert contract.compute.final.gpu_count == 4
    assert contract.compute.search.max_candidate_rpc_calls == 120_012
    assert contract.compute.final.max_candidate_rpc_calls == 500_020
    assert contract.metrics.search[0].data_role == "validation"
    assert contract.candidate.editable_paths == ("program_0/src",)
    assert contract.protocol.final_seeds == (21, 22)
    assert contract.protocol.search_reset_sequence == (11, 12)
    assert contract.protocol.search_smoke_reset_sequence == (11,)
    assert contract.protocol.final_reset_sequence == (21, 22)
    assert contract.protocol.final_smoke_reset_sequence == (21,)

    # explicit repeated reset sequences are accepted
    _rewrite(_reset(contract_path), lambda data: data["protocol"].update(
        search_reset_sequence=[11] * 9 + [12] * 9, search_smoke_reset_sequence=[11, 11, 11],
        final_reset_sequence=[21] * 9 + [22] * 9, final_smoke_reset_sequence=[21, 21],
    ))
    contract = load_task_contract(contract_path)
    assert contract.protocol.search_reset_sequence == (11,) * 9 + (12,) * 9
    assert contract.protocol.search_smoke_reset_sequence == (11, 11, 11)
    assert contract.protocol.final_reset_sequence == (21,) * 9 + (22,) * 9
    assert contract.protocol.final_smoke_reset_sequence == (21, 21)

    # smoke defaults follow the first production reset, not statistical seed order
    _rewrite(_reset(contract_path), lambda data: data["protocol"].update(
        search_reset_sequence=[12, 11, 12], final_reset_sequence=[22, 21, 22]
    ))
    contract = load_task_contract(contract_path)
    assert contract.protocol.search_smoke_reset_sequence == (12,)
    assert contract.protocol.final_smoke_reset_sequence == (22,)

    # a legacy contract uses the safe default candidate RPC budget
    _rewrite(_reset(contract_path), lambda data: (
        data["compute"]["search"].pop("max_candidate_rpc_calls"),
        data["compute"]["final"].pop("max_candidate_rpc_calls"),
    ))
    contract = load_task_contract(contract_path)
    assert contract.compute.search.max_candidate_rpc_calls == 100_000
    assert contract.compute.final.max_candidate_rpc_calls == 100_000

    # a self-contained contract without upstream or dependencies loads
    _rewrite(_reset(contract_path), lambda data: (
        data.update(upstream=[]), data["provenance"].update(dependencies=[])
    ))
    contract = load_task_contract(contract_path)
    assert contract.upstream == ()
    assert contract.provenance.dependencies == ()

    # a non-package upstream without package_version loads
    _rewrite(_reset(contract_path), lambda data: data["upstream"][0].update(package_version=None))
    assert load_task_contract(contract_path).upstream[0].package_version is None


def _budget_bound(plane: str, field: str, value: list[int]) -> Callable[[dict], object]:
    return lambda data: (
        data["compute"][plane].update(max_candidate_rpc_calls=2),
        data["protocol"].update({field: value}),
    )


_CPU_COMPUTE = {
    "accelerator": "cpu", "accelerator_model": "none", "gpu_count": 1,
    "evaluator_gpu_count": 1, "candidate_gpu_count": 0, "cpu_cores": 4, "memory_gb": 16,
    "precision": "float64", "wall_clock_seconds": 120,
}
_NO_GPU_COMPUTE = {
    "accelerator": "gpu", "accelerator_model": "H100", "gpu_count": 0,
    "evaluator_gpu_count": 0, "candidate_gpu_count": 0, "cpu_cores": 4, "memory_gb": 16,
    "precision": "float32", "wall_clock_seconds": 120,
}

_INVALID_CONTRACTS: list[tuple[Callable[[dict[str, Any]], object], str]] = [
    (lambda data: data.update(schema_version=2), "schema_version"),
    (lambda data: data.update(extra=True), "unexpected field"),
    (lambda data: data["method"].update(extra=True), "unexpected field"),
    (lambda data: data["candidate"].update(editable_paths=["../private"]), "relative path"),
    (lambda data: data["candidate"].update(editable_paths=["resources/final"]), "editable path"),
    (lambda data: data["canonical_task"].update(path="../task.md"), "relative path"),
    (lambda data: data["evaluators"].update(final="/tmp/final.py"), "relative path"),
    (lambda data: data["evaluators"].update(search="search.py"), "evaluate.py"),
    (lambda data: data["evaluators"].update(final="final.py"), "evaluate_final.py"),
    (lambda data: data["resources"].update(final_root="resources/search/final"), "resource roots"),
    (lambda data: data["evaluators"].update(final="evaluate.py"), "must differ"),
    (lambda data: data["evaluators"].update(search="resources/final/evaluate.py"), "evaluate.py"),
    (lambda data: data["data"].update(search_roles=["development", "final"]), "final"),
    (
        lambda data: data["metrics"].update(
            search=[{"name": "combined_score", "data_role": "final"}]
        ),
        "search metric",
    ),
    (
        lambda data: data["metrics"].update(
            search=[{"name": "proxy_score", "data_role": "validation"}]
        ),
        "combined_score",
    ),
    (lambda data: data["upstream"][0].update(revision="main"), "pinned"),
    (lambda data: data["upstream"][0].update(revision="refs/heads/dev"), "pinned"),
    (lambda data: data["compute"].update(search=_CPU_COMPUTE), "gpu_count"),
    (lambda data: data["compute"]["search"].update(wall_clock_seconds=119), "at least 120 seconds"),
    (lambda data: data["protocol"].update(search_seeds=[]), "search_seeds"),
    (lambda data: data["protocol"].update(final_seeds=[1, 1]), "final_seeds"),
    (lambda data: data["protocol"].update(baselines=[]), "baselines"),
    (
        lambda data: data["feedback"].update(
            search_visible=["development metrics", "final examples"]
        ),
        "feedback",
    ),
    (lambda data: data["publication"].update(required_evidence=[]), "required_evidence"),
    (lambda data: data["compute"].update(final=_NO_GPU_COMPUTE), "gpu_count"),
    # reset sequence relationships
    (lambda data: data["protocol"].update(search_reset_sequence=[11]), "cover exactly search_seeds"),
    (
        lambda data: data["protocol"].update(search_reset_sequence=[11, 12, 13]),
        "cover exactly search_seeds",
    ),
    (lambda data: data["protocol"].update(final_reset_sequence=[21]), "cover exactly final_seeds"),
    (
        lambda data: data["protocol"].update(final_reset_sequence=[21, 22, 23]),
        "cover exactly final_seeds",
    ),
    (lambda data: data["protocol"].update(search_smoke_reset_sequence=[]), "non-empty array"),
    (
        lambda data: data["protocol"].update(search_smoke_reset_sequence=[12]),
        "first production reset seed",
    ),
    (lambda data: data["protocol"].update(final_smoke_reset_sequence=[]), "non-empty array"),
    (
        lambda data: data["protocol"].update(final_smoke_reset_sequence=[22]),
        "first production reset seed",
    ),
    # reset sequences bounded by the plane RPC budget
    (
        _budget_bound("search", "search_reset_sequence", [11, 12, 11]),
        "protocol.search_reset_sequence.*max_candidate_rpc_calls",
    ),
    (
        _budget_bound("search", "search_smoke_reset_sequence", [11, 11, 11]),
        "protocol.search_smoke_reset_sequence.*max_candidate_rpc_calls",
    ),
    (
        _budget_bound("final", "final_reset_sequence", [21, 22, 21]),
        "protocol.final_reset_sequence.*max_candidate_rpc_calls",
    ),
    (
        _budget_bound("final", "final_smoke_reset_sequence", [21, 21, 21]),
        "protocol.final_smoke_reset_sequence.*max_candidate_rpc_calls",
    ),
    # unsafe candidate RPC budgets
    *(
        (lambda data, v=v: data["compute"]["search"].update(max_candidate_rpc_calls=v),
         "max_candidate_rpc_calls")
        for v in (True, 0, -1, 10_000_001)
    ),
    # paths that cannot be safely mounted
    *(
        (lambda data, u=u: data["resources"].update(final_root=u), "safe POSIX")
        for u in ("resources/final\nleak", "resources/final\tleak", "a:b")
    ),
    # frozen package initializer exception keeps unsafe components rejected
    *(
        (lambda data, r=r: data["provenance"]["artifacts"][0].update(path=r),
         "safe POSIX relative path")
        for r in (
            "resources/final/__init__.py/module.py",
            "resources/final/_package/__init__.py",
            "resources/final/.hidden/__init__.py",
            "resources/final/../__init__.py",
            "resources/final//__init__.py",
            "resources/final/package\\__init__.py",
            "resources/final/package/_private.py",
            "resources/final/package/__init__.py/",
        )
    ),
]


def test_rejects_invalid_contracts(contract_path: Path) -> None:
    # `index` (visible in the failure traceback) identifies the offending case.
    for index, (mutate, match) in enumerate(_INVALID_CONTRACTS):
        _rewrite(_reset(contract_path), mutate)
        with pytest.raises(TaskContractError, match=match):
            load_task_contract(contract_path)


def test_rejects_corrupt_files_stale_digests_and_runtime_caches(tmp_path: Path) -> None:
    # a stale canonical task digest is rejected
    contract_path = _write_scaffold(tmp_path / "stale")
    (contract_path.parent / "task.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="digest mismatch"):
        load_task_contract(contract_path)

    # duplicate JSON fields are rejected
    contract_path = _write_scaffold(tmp_path / "duplicate")
    contract_path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(TaskContractError, match="duplicate JSON field"):
        load_task_contract(contract_path)

    # a missing canonical task is rejected
    contract_path = _write_scaffold(tmp_path / "missing")
    (contract_path.parent / "task.md").unlink()
    with pytest.raises(TaskContractError, match="declared artifact is not a file"):
        load_task_contract(contract_path)

    # a redirected canonical task path is rejected
    contract_path = _write_scaffold(tmp_path / "redirected")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "task.md").write_text("outside\n", encoding="utf-8")
    (contract_path.parent / "linked").symlink_to(outside, target_is_directory=True)
    _rewrite(contract_path, lambda data: data.update(
        canonical_task={"path": "linked/task.md", "sha256": _sha256(outside / "task.md")}
    ))
    with pytest.raises(TaskContractError, match="must be task.md"):
        load_task_contract(contract_path)

    # an oversized contract is rejected before JSON parsing
    path = tmp_path / "task_contract.json"
    path.write_bytes(b"{" + b" " * MAX_TASK_CONTRACT_BYTES + b"}")
    with pytest.raises(TaskContractError, match="exceeds"):
        load_task_contract(path)

    # the JSON integer guard is normalised to a contract error
    path.write_text("{" + '"value":' + "1" * 5000 + "}", encoding="utf-8")
    with pytest.raises(TaskContractError, match="failed to load"):
        load_task_contract(path)

    # runtime bytecode in frozen resources is rejected
    contract_path = _write_scaffold(tmp_path / "bytecode")
    cache_file = contract_path.parent / "resources/search/__pycache__/adapter.cpython-312.pyc"
    cache_file.parent.mkdir()
    cache_file.write_bytes(b"derived bytecode")
    with pytest.raises(TaskContractError, match="missing provenance checksums"):
        load_task_contract(contract_path)

    # a runtime cache symlink is rejected
    contract_path = _write_scaffold(tmp_path / "cache-link")
    (contract_path.parent / "resources/search/__pycache__").symlink_to(
        contract_path.parent / "resources/final", target_is_directory=True
    )
    with pytest.raises(TaskContractError, match="symlink"):
        load_task_contract(contract_path)


def test_bytecode_pruning_preserves_runtime_environment_and_rejects_cache_symlink(
    tmp_path: Path,
) -> None:
    # generated bytecode is pruned but the runtime environment is preserved
    task_dir = tmp_path / "task"
    resource_cache = task_dir / "resources/search/__pycache__/adapter.pyc"
    runtime_cache = task_dir / ".venv/lib/__pycache__/installed.pyc"
    resource_cache.parent.mkdir(parents=True)
    runtime_cache.parent.mkdir(parents=True)
    resource_cache.write_bytes(b"generated")
    runtime_cache.write_bytes(b"installed")
    prune_generated_bytecode(task_dir)
    assert not resource_cache.exists()
    assert runtime_cache.read_bytes() == b"installed"

    # a cache symlink is rejected
    task_dir = tmp_path / "linked"
    cache_parent = task_dir / "resources/search"
    outside = tmp_path / "outside"
    cache_parent.mkdir(parents=True)
    outside.mkdir()
    (cache_parent / "__pycache__").symlink_to(outside, target_is_directory=True)
    with pytest.raises(TaskContractError, match="unsafe runtime cache path"):
        prune_generated_bytecode(task_dir)


def test_fingerprint_covers_resource_trees_and_rejects_symlinks(tmp_path: Path) -> None:
    # the fingerprint covers contract, task, evaluators, and resource trees
    contract_path = _write_scaffold(tmp_path / "covered")
    task_scaffold_fingerprint(contract_path.parent)
    (contract_path.parent / "resources" / "search" / "dev.json").write_text(
        '{"changed": true}\n', encoding="utf-8"
    )
    with pytest.raises(TaskContractError, match="artifact digest mismatch"):
        task_scaffold_fingerprint(contract_path.parent)

    # symlinks in declared resources are rejected
    for kind in ("resource_link", "resource_parent"):
        task_dir = _write_scaffold(tmp_path / kind).parent
        if kind == "resource_link":
            (task_dir / "resources" / "search" / "link").symlink_to(task_dir / "task.md")
        else:
            (task_dir / "resources").rename(task_dir / "real_resources")
            (task_dir / "resources").symlink_to(task_dir / "real_resources", target_is_directory=True)
        with pytest.raises(TaskContractError, match="symlink"):
            task_scaffold_fingerprint(task_dir)


def test_frozen_python_package_initializer_is_verified_and_accepted_by_contract_gate(
    tmp_path: Path,
) -> None:
    from src.task_drafting.checks import GateSettings, run_gate
    from tests.task_drafting.test_task_alignment import _write_scaffold as _write_gate_scaffold

    # the contract verifies a frozen package initializer
    contract_path = _write_scaffold(tmp_path / "task")
    relative = "resources/final/package/__init__.py"
    package_file = contract_path.parent / relative
    package_file.parent.mkdir()
    package_file.write_text("PACKAGE = 'frozen'\n")
    _rewrite(contract_path, lambda data: data["provenance"]["artifacts"].append(
        _artifact(relative, contract_path.parent, role="final")
    ))
    assert load_task_contract(contract_path).provenance.artifacts[-1].path == relative
    package_file.write_text("PACKAGE = 'modified'\n")
    with pytest.raises(TaskContractError, match="artifact digest mismatch"):
        load_task_contract(contract_path)

    # the contract gate accepts a frozen evaluator package
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    _write_gate_scaffold(gate_dir)
    gate_contract = gate_dir / "task_contract.json"
    relative = "resources/search/evaluator/__init__.py"
    initializer = gate_dir / relative
    initializer.parent.mkdir()
    initializer.write_text("PACKAGE = 'frozen'\n")
    _rewrite(gate_contract, lambda data: data["provenance"]["artifacts"].insert(
        0, _artifact(relative)
    ))
    report = run_gate("contract", GateSettings(
        task_dir=gate_dir, source_task_dir=gate_dir, containerize=False,
        container_image="unused", container_network="none", timeout_seconds=10,
    ))
    assert report.status == "pass", report.diagnostic


def test_contract_v1_scaffold_requirements_are_opt_in_and_auto_detected(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    (task_dir / "program_0" / "src").mkdir(parents=True)
    (task_dir / "capture.py").write_text("", encoding="utf-8")
    (task_dir / "evaluate.py").write_text("", encoding="utf-8")
    (task_dir / "program_0" / "src" / "main.py").write_text("", encoding="utf-8")
    (task_dir / "program_0" / "program_spec.toml").write_text(
        'schema_version = 1\ncode_root = "src"\n[entrypoint]\npath = "main.py"\n'
        'language = "python"\ntype = "script"\n[copy]\ninclude = []\nexclude = []\n'
        '[integration]\nkind = "python"\n',
        encoding="utf-8",
    )

    assert missing_task_scaffold_files(task_dir) == []
    assert missing_task_scaffold_files(task_dir, require_contract_v1=True) == [
        task_dir / "task_contract.json",
        task_dir / "evaluate_final.py",
        task_dir / ".agentic_task_validation.json",
    ]

    (task_dir / "task_contract.json").write_text("{}", encoding="utf-8")
    assert missing_task_scaffold_files(task_dir) == [task_dir / "evaluate_final.py"]
    (task_dir / "evaluate_final.py").write_text("", encoding="utf-8")
    assert missing_task_scaffold_files(task_dir) == []
    assert missing_task_scaffold_files(task_dir, require_contract_v1=True) == [
        task_dir / ".agentic_task_validation.json",
    ]


# ---------------------------------------------------------------------------
# Preflight lint
# ---------------------------------------------------------------------------


def test_preflight_lint_normalizes_like_parser_and_reports_protocol_and_schema_errors(
    contract_path: Path,
) -> None:
    # lint matches strict parser normalisation
    def pad(data: dict) -> None:
        data["canonical_task"]["path"] = " task.md "
        data["candidate"]["interface"] = " python_rpc_v1:handle "
        data["candidate"]["editable_paths"] = [" program_0 "]
        data["evaluators"] = {"search": " evaluate.py ", "final": " evaluate_final.py "}
        data["resources"] = {
            "search_root": " resources/search ", "final_root": " resources/final ",
            "public_artifact_bundle": None,
        }
        data["data"]["search_roles"] = [" development ", " validation "]
        data["compute"]["search"]["accelerator"] = " GPU "

    _rewrite(contract_path, pad)
    load_task_contract(contract_path)
    assert _lint(contract_path) == ()

    # the explicit RPC budget requirement is opt-in
    _rewrite(_reset(contract_path), lambda data: (
        data["compute"]["search"].pop("max_candidate_rpc_calls"),
        data["compute"]["final"].pop("max_candidate_rpc_calls"),
    ))
    assert _lint(contract_path, require_explicit_rpc_budgets=False) == ()
    assert _lint(contract_path, require_explicit_rpc_budgets=True) == (
        "compute.search is missing required field(s): max_candidate_rpc_calls",
        "compute.final is missing required field(s): max_candidate_rpc_calls",
    )

    # all protocol shape errors are reported
    _rewrite(_reset(contract_path), lambda data: data["protocol"].update(
        final_reset_sequence=[{"op": "reset", "seed": data["protocol"]["final_seeds"][0]}],
        baselines=[{"name": "reference"}],
        promotion_rule={"metric": "combined_score"},
        search_proxies=[{"name": "proxy", "preserves": ["objective"], "omits": ["final_data"]}],
    ))
    findings = _lint(contract_path)
    assert "protocol.final_reset_sequence[0] must be a non-negative integer" in findings
    assert "protocol.baselines[0] must be a non-empty string" in findings
    assert "protocol.promotion_rule must be a non-empty string" in findings
    assert "protocol.search_proxies[0].preserves must be a non-empty string" in findings
    assert "protocol.search_proxies[0].omits must be a non-empty string" in findings

    # protocol relationship errors are reported
    proxy = {"name": "proxy", "preserves": "objective", "omits": "final_data"}
    _rewrite(_reset(contract_path), lambda data: (
        data["protocol"].update(
            search_seeds=[1, 1], final_seeds=[1, 2], search_reset_sequence=[1, 3],
            search_smoke_reset_sequence=[2], final_reset_sequence=[3] * 17,
            final_smoke_reset_sequence=[2, 3], baselines=["reference", "reference"],
            search_proxies=[proxy, proxy],
        ),
        data["compute"]["final"].update(max_candidate_rpc_calls=16),
    ))
    findings = _lint(contract_path)
    assert "protocol.search_seeds must not contain duplicates" in findings
    assert "protocol search and final seeds must be disjoint" in findings
    assert "protocol.search_reset_sequence must cover exactly search_seeds" in findings
    assert (
        "protocol.search_smoke_reset_sequence must contain only the first production reset seed"
        in findings
    )
    assert "protocol.final_reset_sequence exceeds compute.final.max_candidate_rpc_calls" in findings
    assert "protocol.final_reset_sequence must cover exactly final_seeds" in findings
    assert (
        "protocol.final_smoke_reset_sequence must contain only the first production reset seed"
        in findings
    )
    assert "protocol.baselines must not contain duplicates" in findings
    assert "protocol.search_proxies names must be unique" in findings

    # explicit repeated reset sequences are accepted
    _rewrite(_reset(contract_path), lambda data: data["protocol"].update(
        search_reset_sequence=[11] * 9 + [12] * 9, search_smoke_reset_sequence=[11, 11],
        final_reset_sequence=[21] * 9 + [22] * 9, final_smoke_reset_sequence=[21, 21],
    ))
    assert _lint(contract_path) == ()

    # a missing optional smoke sequence error is not invented
    _rewrite(_reset(contract_path), lambda data: data["protocol"].update(final_reset_sequence=[]))
    findings = _lint(contract_path)
    assert "protocol.final_reset_sequence must be a non-empty array" in findings
    assert not any("final_smoke_reset_sequence" in finding for finding in findings)

    # strict path rejections are preserved
    for path in ("program_0//src", "program_0/./src"):
        _rewrite(_reset(contract_path), lambda data, path=path: data["candidate"].update(
            editable_paths=[path]
        ))
        assert any("safe POSIX relative path" in finding for finding in _lint(contract_path)), path

    # missing fields are reported across schema branches
    _rewrite(_reset(contract_path), lambda data: data.update(
        method={}, metrics={}, feedback={}, publication={},
        provenance={"dependencies": [{}], "artifacts": []},
    ))
    findings = _lint(contract_path)
    for label in ("method", "metrics", "feedback", "publication"):
        assert any(f"{label} is missing required field(s)" in finding for finding in findings)
    assert any("provenance.dependencies[0] is missing" in finding for finding in findings)


def test_preflight_lint_bounds_untrusted_diagnostics_and_malformed_json(
    tmp_path: Path, contract_path: Path,
) -> None:
    # independent generated contract errors are all reported
    contract = {
        "schema_version": 1,
        "canonical_task": {"path": "./task.md", "sha256": "0" * 64},
        "candidate": {"editable_paths": ["program_*/src"]},
        "data": {"search_roles": ["development data"], "final_roles": ["final"]},
        "upstream": [{
            "repository": "fixture", "revision": "v1", "package_version": "1", "license": "MIT",
            "entrypoints": ["main"], "patches": ["evaluate.py", "descriptive patch text"],
            "compatibility_justification": "unsupported",
        }],
        "provenance": {"artifacts": []},
        "compute": {"search": {"accelerator": "NVIDIA H100"}, "final": {"accelerator": "gpu"}},
        "protocol": {
            "search_seeds": [1], "final_seeds": [2], "baselines": ["fixture"],
            "promotion_rule": "promote",
            "search_proxies": [{"name": "proxy", "preserves": "", "omits": "cost"}],
        },
    }
    path = tmp_path / "task_contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    findings = _lint(path)
    assert "canonical_task.path must be task.md" in findings
    assert any("candidate.editable_paths[0]" in finding for finding in findings)
    assert "data.search_roles[0] must be a snake_case role" in findings
    assert any("unexpected field(s)" in finding for finding in findings)
    assert any("upstream[0].patches[1]" in finding for finding in findings)
    assert any("patch path(s)" in finding and "checksums" in finding for finding in findings)
    assert "compute.search.accelerator must be 'cpu' or 'gpu'" in findings
    assert "protocol.search_proxies[0].preserves must be a non-empty string" in findings

    # deep and huge-integer JSON are handled
    path.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")
    assert _lint(path) in ((), ("task contract must be an object",))
    path.write_text("1" * 5000, encoding="utf-8")
    assert _lint(path) == ()
    path.write_bytes(b"{" + b" " * MAX_TASK_CONTRACT_BYTES + b"}")
    assert any("exceeds" in finding for finding in _lint(path))

    # an incomplete patch cross-reference is suppressed
    artifact = _artifact("resources/search/fixture.txt", role="fixture")
    _rewrite(contract_path, lambda data: (
        data["upstream"][0].update(patches=["evaluate.py"]),
        data["provenance"].update(artifacts=[artifact] * 128 + [{**artifact, "path": "evaluate.py"}]),
    ))
    assert not any("lack matching provenance" in finding for finding in _lint(contract_path))

    # untrusted contract diagnostics are bounded
    malicious_key = "DO_NOT_REFLECT_SECRET_VALUE"
    _rewrite(_reset(contract_path), lambda data: (
        data.update({f"{malicious_key}_{index}": index for index in range(200)}),
        data["provenance"].update(artifacts=[{} for _ in range(100)]),
    ))
    findings = _lint(contract_path)
    assert len(findings) <= 65
    assert malicious_key not in "\n".join(findings)
    assert any("omitted" in finding for finding in findings)


def test_preflight_lint_compares_requirements_with_dependencies(contract_path: Path) -> None:
    # requirement mismatches, non-installable identities, and extras
    for name, requirements, dependencies, expected in [
        (
            "requirement_mismatch", "example-package==1.2.3\n",
            [{"name": "example-package", "version": "1.2.3"},
             {"name": "python", "version": "3.11.11"}],
            "requirements.txt and provenance.dependencies must contain the same pinned packages; "
            "requirements-only=[] contract-only=['python==3.11.11']",
        ),
        (
            "noninstallable_identity", "example-package==1.2.3\n",
            [{"name": "Python runtime", "version": "3.11"}],
            "provenance.dependencies[0] must name a pip-installable package and version",
        ),
        (
            "extras_preserved", "example-package[foo]==1.2.3\n",
            [{"name": "example-package[bar]", "version": "1.2.3"}],
            "requirements.txt and provenance.dependencies must contain the same pinned packages; "
            "requirements-only=['example-package[foo]==1.2.3'] "
            "contract-only=['example-package[bar]==1.2.3']",
        ),
    ]:
        requirements_path = _requirements(contract_path, requirements)
        _rewrite(_reset(contract_path), lambda data, dependencies=dependencies: (
            data["provenance"].update(dependencies=dependencies)
        ))
        assert _lint(contract_path, requirements_path=requirements_path) == (expected,), name

    # malformed pip identities are rejected
    for name, version in [
        ("foo-", "1"), ("foo[a,,b]", "1"), ("foo[-]", "1"),
        ("example-package", "1..2"), ("example-package", "1!"), ("example-package", "1-"),
        ("example-package", "1_"),
    ]:
        requirements_path = _requirements(contract_path, f"{name}=={version}\n")
        _rewrite(_reset(contract_path), lambda data, name=name, version=version: (
            data["provenance"].update(dependencies=[{"name": name, "version": version}])
        ))
        findings = _lint(contract_path, requirements_path=requirements_path)
        assert any("pip-installable package" in finding for finding in findings), (name, version)

    # the requirements file size is bounded
    _reset(contract_path)
    requirements_path = contract_path.parent / "requirements.txt"
    requirements_path.write_bytes(b"x" * (MAX_TASK_CONTRACT_BYTES + 1))
    assert _lint(contract_path, requirements_path=requirements_path) == (
        f"requirements.txt exceeds {MAX_TASK_CONTRACT_BYTES} byte harness limit",
    )


def test_preflight_lint_bounds_dependency_inventories(contract_path: Path) -> None:
    # requirement rows are bounded
    for name, rows, exact in [
        ("duplicate_rows", "example-package==1.2.3\n" * 257, True),
        ("distinct_rows", "".join(f"dependency-{index}==1.0\n" for index in range(257)), False),
    ]:
        findings = _lint(contract_path, requirements_path=_requirements(contract_path, rows))
        if exact:
            assert findings == ("requirements.txt exceeds 256 pinned entries",), name
        else:
            assert "requirements.txt exceeds 256 pinned entries" in findings, name

    # complete large dependency locks are accepted
    for count in (65, 89, 128, 256):
        dependencies = [{"name": f"package-{index}", "version": "1.0"} for index in range(count)]
        _rewrite(_reset(contract_path), lambda data, dependencies=dependencies: (
            data["provenance"].update(dependencies=dependencies)
        ))
        requirements_path = _requirements(
            contract_path, "".join(f"{item['name']}=={item['version']}\n" for item in dependencies)
        )
        assert _lint(contract_path, requirements_path=requirements_path) == (), count

    # the dependency inventory is still bounded
    _rewrite(_reset(contract_path), lambda data: data["provenance"].update(
        dependencies=[{"name": f"package-{index}", "version": "1.0"} for index in range(257)]
    ))
    assert _lint(contract_path) == (
        "provenance.dependencies has 1 item(s) omitted from bounded lint",
    )

    # the dependency limit does not compare a truncated list
    _rewrite(_reset(contract_path), lambda data: (
        data["provenance"].update(
            dependencies=[{"name": f"dependency-{i}", "version": "1.0"} for i in range(257)]
        ),
        data.update(upstream=data["upstream"] * 65),
    ))
    requirements_path = _requirements(contract_path, "different==1.0\n")
    findings = _lint(contract_path, requirements_path=requirements_path)
    assert "provenance.dependencies has 1 item(s) omitted from bounded lint" in findings
    assert "upstream has 1 item(s) omitted from bounded lint" in findings
    assert not any("same pinned packages" in finding for finding in findings)
    assert len(findings) <= 65


def test_large_artifact_inventories_keep_diagnostics_bounded(contract_path: Path) -> None:
    # a reference cache inventory fits bounded artifact lint
    contract = _rewrite(contract_path, lambda data: data["provenance"]["artifacts"].extend(
        _artifact(f"resources/search/reference-{i}.json", source="frozen reference")
        for i in range(80)
    ))
    assert not any("provenance.artifacts has" in finding for finding in _lint(contract_path))
    contract["provenance"]["artifacts"].extend([{}] * 256)
    contract_path.write_text(json.dumps(contract))
    assert any("provenance.artifacts has" in finding for finding in _lint(contract_path))

    # inventory errors keep the diagnostic count bounded
    for field, count in [("dependencies", 257), ("artifacts", 256)]:
        _rewrite(_reset(contract_path), lambda data, field=field, count=count: (
            data["provenance"].update({field: [{}] * count})
        ))
        findings = _lint(contract_path)
        assert len(findings) == 65, field
        assert findings[-1].endswith("additional bounded lint finding(s) omitted"), field


# Large manual-task inventories retain complete, bounded preflight validation.


def _inventory(contract_path: Path, count: int) -> dict:
    contract = json.loads(contract_path.read_text())
    artifacts = contract["provenance"]["artifacts"]
    for index in range(count - len(artifacts)):
        relative = f"resources/search/reference-{index}.json"
        (contract_path.parent / relative).write_text("{}\n")
        artifacts.append(_artifact(relative, contract_path.parent, source="frozen reference"))
    return contract


def test_manual_reference_inventories_pass_complete_bounded_preflight(
    tmp_path: Path, contract_path: Path,
) -> None:
    # manual reference inventories pass complete preflight
    for count in (155, 256):
        contract_path.write_text(json.dumps(_inventory(_reset(contract_path), count)))
        assert len(load_task_contract(contract_path).provenance.artifacts) == count, count
        assert _lint(contract_path, require_explicit_rpc_budgets=True) == (), count

    # artifact fields beyond the old limit are checked
    for index in (128, 154, 255):
        contract = _inventory(_reset(contract_path), 256)
        del contract["provenance"]["artifacts"][index]["sha256"]
        contract_path.write_text(json.dumps(contract))
        assert _lint(contract_path) == (
            f"provenance.artifacts[{index}] is missing required field(s): sha256",
        ), index

    # a patch checksum can appear at the end of a large inventory
    contract = _inventory(_reset(contract_path), 256)
    contract["upstream"][0]["patches"] = [contract["provenance"]["artifacts"][-1]["path"]]
    contract_path.write_text(json.dumps(contract))
    assert _lint(contract_path) == ()

    # an unlisted patch checksum is rejected at 155 and skipped once truncated at 257
    for name, count, expected in [
        ("large_rejects_unlisted_patch", 155,
         "1 upstream patch path(s) lack matching provenance artifact checksums"),
        ("oversized_skips_truncated_comparison", 257,
         "provenance.artifacts has 1 item(s) omitted from bounded lint"),
    ]:
        contract = _inventory(_reset(contract_path), count)
        contract["upstream"][0]["patches"] = ["resources/search/unlisted.py"]
        contract_path.write_text(json.dumps(contract))
        assert _lint(contract_path) == (expected,), name

    # a declared package initializer patch is accepted (fresh scaffold: no residual references)
    contract_path = _write_scaffold(tmp_path / "patch")
    relative = "resources/search/package/__init__.py"
    initializer = contract_path.parent / relative
    initializer.parent.mkdir()
    initializer.write_text("PACKAGE = 'patched'\n")
    _rewrite(contract_path, lambda data: (
        data["upstream"][0].update(patches=[relative]),
        data["provenance"]["artifacts"].append(
            _artifact(relative, contract_path.parent, source="frozen Python package patch")
        ),
    ))
    load_task_contract(contract_path)
    assert _lint(contract_path) == ()


# ---------------------------------------------------------------------------
# Declared digest synchronisation
# ---------------------------------------------------------------------------


def test_synchronize_declared_digests_refreshes_artifacts_and_preserves_semantics(
    tmp_path: Path,
) -> None:
    # existing artifact digests are refreshed
    contract_path = _write_scaffold(tmp_path / "refresh")
    resource = contract_path.parent / "resources/search/dev.json"
    resource.write_text('{"changed": true}\n', encoding="utf-8")
    assert _sync(contract_path) is True
    contract = load_task_contract(contract_path)
    artifact = next(item for item in contract.provenance.artifacts if item.path.endswith("dev.json"))
    assert artifact.sha256 == _sha256(resource)

    # synchronisation does not change contract semantics
    contract_path = _write_scaffold(tmp_path / "semantics")
    before = json.loads(contract_path.read_text(encoding="utf-8"))
    _sync(contract_path)
    after = json.loads(contract_path.read_text(encoding="utf-8"))
    before["canonical_task"]["sha256"] = after["canonical_task"]["sha256"]
    for old, new in zip(before["provenance"]["artifacts"], after["provenance"]["artifacts"]):
        old["sha256"] = new["sha256"]
    assert after == before

    # a package initializer is accepted and refreshed on either plane
    for plane in ("search", "final"):
        contract_path = _write_scaffold(tmp_path / f"initializer-{plane}")
        relative = f"resources/{plane}/evaluator/__init__.py"
        initializer = contract_path.parent / relative
        initializer.parent.mkdir()
        initializer.write_text("PACKAGE = 'frozen'\n")
        _rewrite(contract_path, lambda data, relative=relative, plane=plane: (
            data["provenance"]["artifacts"].insert(0, _artifact(relative, role=plane))
        ))
        assert _sync(contract_path), plane
        assert load_task_contract(contract_path).provenance.artifacts[0].sha256 == _sha256(initializer)
        assert not _sync(contract_path), plane


def test_synchronize_declared_digests_rejects_unsafe_paths_symlinks_and_unknown_fields(
    tmp_path: Path, contract_path: Path,
) -> None:
    # unsafe or symlinked artifact paths are rejected
    for unsafe in ("../outside", "/etc/passwd", "resources/link"):
        if unsafe == "resources/link":
            (contract_path.parent / "resources/link").symlink_to(contract_path.parent / "task.md")
        _rewrite(_reset(contract_path), lambda data, unsafe=unsafe: (
            data["provenance"]["artifacts"][0].update(path=unsafe)
        ))
        with pytest.raises(ContractSyncError):
            _sync(contract_path)

    # unknown contract fields are rejected
    _rewrite(_reset(contract_path), lambda data: data.update(unexpected=True))
    with pytest.raises(ContractSyncError, match="unexpected"):
        _sync(contract_path)

    # unsafe initializer spellings are rejected without rewriting the contract
    # (normalization cannot hide invalid spelling behind a missing file)
    (contract_path.parent / "resources/search/__init__.py").write_text("PACKAGE = 'frozen'\n")
    for relative in (
        "resources/search/__init__.py/module.py",
        "resources/search/_package/__init__.py",
        "resources/search/.hidden/__init__.py",
        "resources/search/../__init__.py",
        "resources/search//__init__.py",
        "resources/search/./__init__.py",
        "resources/search/package\\__init__.py",
        "resources/search/package/_private.py",
        "resources/search/package/__init__.py/",
        "/resources/search/package/__init__.py",
    ):
        _rewrite(_reset(contract_path), lambda data, relative=relative: (
            data["provenance"]["artifacts"][0].update(path=relative)
        ))
        before = contract_path.read_bytes()
        with pytest.raises(ContractSyncError, match="safe .*relative path"):
            _sync(contract_path)
        assert contract_path.read_bytes() == before, relative

    # symlinked initializer components are rejected
    for link_component in ("package", "package/__init__.py"):
        linked = _write_scaffold(tmp_path / f"linked-{link_component.replace('/', '-')}")
        external = linked.parent / "external"
        external.mkdir()
        (external / "__init__.py").write_text("PACKAGE = 'outside'\n")
        link = linked.parent / "resources/search" / link_component
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(external if link_component == "package" else external / "__init__.py")
        _rewrite(linked, lambda data: data["provenance"]["artifacts"][0].update(
            path="resources/search/package/__init__.py"
        ))
        with pytest.raises(ContractSyncError, match="must not contain symlinks"):
            _sync(linked)


# ---------------------------------------------------------------------------
# Task copy ignore patterns and task-agnostic src tree (src.investigation.files)
# ---------------------------------------------------------------------------


def _ignored(path: str, pattern: str, *, is_dir: bool = False) -> bool:
    return task_copy_path_is_ignored(path, is_dir=is_dir, patterns=(pattern,))


def test_task_copy_ignore_patterns_and_bounded_globstar_matching(monkeypatch) -> None:
    # ignore pattern semantics
    for path, pattern, is_dir, expected in [
        ("result.json", "/*.json", False, True),
        ("nested/result.json", "/*.json", False, False),
        ("data/result.json", "/data/*.json", False, True),
        ("data/nested/result.json", "/data/*.json", False, False),
        ("cache-one", "/cache*/", True, True),
        ("nested/cache-one", "/cache*/", True, False),
        ("runtime/manifest.json", "/runtime", False, True),
        ("foo/result.txt", "/foo/**/*.txt", False, True),
        ("foo/one/result.txt", "/foo/**/*.txt", False, True),
        ("foo/one/two/result.txt", "/foo/**/*.txt", False, True),
        ("nested/foo/one/result.txt", "/foo/**/*.txt", False, False),
        ("nested/result.json", "*.json", False, True),
        *((path, pattern, is_dir, False)
          for pattern in ("/", "///", "/./")
          for path, is_dir in (("data", True), ("data/result.json", False))),
    ]:
        assert _ignored(path, pattern, is_dir=is_dir) is expected, (path, pattern, is_dir)

    # root globstar matching has bounded state growth
    calls = 0

    def counted_match(name: str, pattern: str) -> bool:
        nonlocal calls
        calls += 1
        return fnmatchcase(name, pattern)

    monkeypatch.setattr("src.investigation.files.fnmatch.fnmatchcase", counted_match)
    path = "/".join(["segment"] * 60)
    pattern = "/" + "/".join(["**"] * 60 + ["never"])
    assert not _ignored(path, pattern)
    assert calls <= 61


# The shipped example tasks (examples/<name>) and whether their task.md carries a Rubrics
# section; circpack ships its rubrics in rubrics_example.json instead, and
# circpack_task_drafting is a bare task.md brief.
EXAMPLE_TASKS = {"circpack": False, "grpo": True}


def test_src_is_task_agnostic_and_examples_ship_a_readable_task_md() -> None:
    # src contains no bundled example task implementations
    assert not list((REPO_ROOT / "src" / "task_harnesses").glob("*.py"))
    for task_name in sorted(EXAMPLE_TASKS):
        assert not (REPO_ROOT / "src" / task_name).exists(), task_name
        assert not list((REPO_ROOT / "src").rglob(f"{task_name}.py")), task_name

    # every shipped example ships a readable task.md
    for task_name in sorted(EXAMPLE_TASKS):
        task_dir = REPO_ROOT / "examples" / task_name
        assert task_dir.is_dir(), task_name
        assert not task_dir.is_symlink(), task_name
        assert (task_dir / "task.md").is_file(), task_name
        goal, rubrics, task_md = read_task_sections(task_dir)
        assert task_md == task_dir / "task.md", task_name
        assert not goal.startswith("(missing"), task_name
        assert rubrics.startswith("(missing") is not EXAMPLE_TASKS[task_name], task_name


# ---------------------------------------------------------------------------
# Evaluator output contract: qualification rejects outputs search cannot consume
# ---------------------------------------------------------------------------


def _evaluator_outputs(tmp_path: Path):
    from tests.final_evaluation.test_final_evaluation import _write_contract_task

    tmp_path.mkdir(parents=True, exist_ok=True)
    task = _write_contract_task(tmp_path)
    contract = load_task_contract(task / "task_contract.json")
    root = tmp_path / "results"
    root.mkdir()
    (root / "metrics.json").write_text(json.dumps({m.name: 0.5 for m in contract.metrics.search}))
    (root / "correct.json").write_text('{"correct": false}')
    (root / "qualification.json").write_text('{"schema_version": 1, "feasible": true}')
    return root, contract


def _qualify(root, contract):
    return task_runtime._validated_contracted_baseline_score(
        root, contract, requirement=task_runtime.TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT,
    )


def test_qualification_validates_optional_evidence(tmp_path):
    # valid or absent evidence preserves the feasible, rejected baseline
    for index, payload in enumerate([None, evidence()]):
        root, contract = _evaluator_outputs(tmp_path / f"valid-{index}")
        if payload is not None:
            (root / "evaluation_evidence.json").write_text(json.dumps(payload))
        assert _qualify(root, contract) == 0.5, index

    # malformed optional evidence is rejected
    for index, payload in enumerate([
        {"schema_version": 1, "aggregate_seed_summaries": [], "plane": "search"},
        {"schema_version": 1, "episodes": []},
        {"schema_version": 1, "episodes": evidence()["episodes"] * 2},
        {"schema_version": 1, "episodes": [{"seed": 1}]},
        {"schema_version": 1, "episodes": [dict(evidence()["episodes"][0], score=float("nan"))]},
    ]):
        root, contract = _evaluator_outputs(tmp_path / f"malformed-{index}")
        (root / "evaluation_evidence.json").write_text(json.dumps(payload))
        assert _qualify(root, contract) is None, index

    # unsafe evidence cannot qualify
    for kind in ("symlink", "oversized"):
        root, contract = _evaluator_outputs(tmp_path / f"unsafe-{kind}")
        path = root / "evaluation_evidence.json"
        if kind == "symlink":
            path.symlink_to(root / "metrics.json")
        else:
            with path.open("wb") as stream:
                stream.truncate(16 * 1024 * 1024 + 1)
        assert _qualify(root, contract) is None, kind

    # host and qualification use the same typed output contract
    from src.variants_pipeline.host_eval import _validate_search_result_schema

    root, contract = _evaluator_outputs(tmp_path / "host")
    (root / "evaluation_evidence.json").write_text('{"schema_version":1,"plane":"search"}')
    assert _qualify(root, contract) is None
    with pytest.raises(EvaluatorOutputContractError):
        _validate_search_result_schema(root, contract)


def test_qualification_storage_failures_propagate_and_metrics_are_read_once(
    tmp_path, monkeypatch,
):
    # evidence storage failures are not authored defects
    original_open = os.open
    for number in (errno.ESTALE, errno.EDQUOT, errno.ENOSPC):
        root, contract = _evaluator_outputs(tmp_path / f"errno-{number}")

        def fail_open(path, *args, number=number, **kwargs):
            if Path(path).name == "evaluation_evidence.json":
                raise OSError(number, "private-path")
            return original_open(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "open", fail_open)
            with pytest.raises(OSError) as error:
                _qualify(root, contract)
        assert error.value.errno == number

    # qualification reuses validated metrics instead of re-reading them
    original_read_text = Path.read_text
    root, contract = _evaluator_outputs(tmp_path / "reuse")

    def fail_second_read(path, *args, **kwargs):
        if path.name in {"metrics.json", "correct.json"}:
            raise OSError(errno.ESTALE, "unexpected second read")
        return original_read_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", fail_second_read)
        assert _qualify(root, contract) == 0.5

    # a feasibility storage failure propagates
    root, contract = _evaluator_outputs(tmp_path / "feasibility")

    def fail_qualification(path, *args, **kwargs):
        if path.name == "qualification.json":
            raise OSError(errno.EDQUOT, "quota")
        return original_read_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", fail_qualification)
        with pytest.raises(OSError):
            _qualify(root, contract)

    # a retained correctness storage failure propagates
    def fail(*args, **kwargs):
        raise OSError(errno.ESTALE, "private")

    monkeypatch.setattr(Path, "read_text", fail)
    with pytest.raises(OSError):
        task_runtime._validated_evaluation_correctness(tmp_path)


def test_host_rejects_invalid_required_outputs_and_non_regular_files(tmp_path):
    # invalid required output contents are rejected
    for index, (filename, raw) in enumerate([
        ("metrics.json", '{"combined_score":0.5,"extra":1}'),
        ("metrics.json", '{"combined_score":true}'),
        ("metrics.json", '{"combined_score":Infinity}'),
        ("metrics.json", '{"combined_score":' + "9" * 400 + '}'),
        ("metrics.json", '{"combined_score":1,"combined_score":2}'),
        ("metrics.json", '[]'),
        ("metrics.json", '{broken'),
        ("correct.json", '{"correct":1}'),
        ("correct.json", '{"correct":true,"extra":"private"}'),
    ]):
        root, _ = _evaluator_outputs(tmp_path / f"invalid-{index}")
        (root / filename).write_text(raw)
        with pytest.raises(EvaluatorOutputContractError):
            validate_evaluator_outputs(root, ("combined_score",))

    # required output files must be bounded regular files
    for kind in ("missing", "symlink", "directory", "oversized", "fifo"):
        root = tmp_path / f"file-{kind}"
        root.mkdir()
        path = root / "metrics.json"
        if kind == "symlink":
            path.symlink_to(root / "absent")
        elif kind == "directory":
            path.mkdir()
        elif kind == "oversized":
            with path.open("wb") as stream:
                stream.truncate(1024 * 1024 + 1)
        elif kind == "fifo":
            os.mkfifo(path)
        with pytest.raises(EvaluatorOutputContractError):
            validate_evaluator_outputs(root, ("combined_score",))
