"""Scoped repair snapshots, transactions, bytecode quarantine, and host gate authorization."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import _imp
import importlib.util
import json
import os
import py_compile
import struct
import sys
from pathlib import Path

import pytest
import src.investigation.task_repair_bytecode as repair_bytecode
import src.investigation.task_repair_scope as repair_scope
import src.investigation.task_repair_transaction as repair_transaction
from src.investigation import task_repair_authorization as authorization_module
from src.investigation.program_spec import render_default_program_spec_toml
from src.investigation.task_alignment import task_tree_fingerprint
from src.investigation.task_contract import MAX_TASK_CONTRACT_BYTES
from src.investigation.task_repair_authorization import (
    DependencyRepairExecutor,
    validate_authorized_repair,
)
from src.investigation.task_repair_scope import (
    TaskRepairScopeError,
    contract_policy_for_smoke_diagnostic,
    create_dependency_removal_repair_snapshot,
    create_digest_synchronized_repair_snapshot,
    create_repair_snapshot,
    quarantine_generated_bytecode,
    validate_repair_snapshot,
)
from src.investigation.task_repair_transaction import (
    commit_repair_transaction,
    create_authored_tree_repair_transaction,
    create_repair_transaction,
    create_tree_repair_transaction,
    restore_repair_transaction,
    settle_repair_transaction,
)
from src.task_drafting import service
from src.task_drafting.contract_sync import synchronize_declared_digests
from src.task_drafting.gate import TaskGateReport


def _write_task(root: Path) -> Path:
    task = root / "task"
    (task / "program_0" / "src").mkdir(parents=True)
    (task / "resources" / "search").mkdir(parents=True)
    (task / "resources" / "final").mkdir()
    for relative, content in {
        "task.md": "# task\n",
        "evaluate.py": "# search\n",
        "evaluate_final.py": "# final\n",
        "program_0/src/program.py": "def handle(request): return request\n",
        "resources/search/data.json": "{}\n",
        "resources/final/data.json": "{}\n",
        "program_0/program_spec.toml": render_default_program_spec_toml(),
        "requirements.txt": "fixture==1.0.0\n",
    }.items():
        (task / relative).write_text(content, encoding="utf-8")
    digest = lambda relative: hashlib.sha256((task / relative).read_bytes()).hexdigest()
    plane = {
        "accelerator": "cpu",
        "accelerator_model": "none",
        "gpu_count": 0,
        "evaluator_gpu_count": 0,
        "candidate_gpu_count": 0,
        "cpu_cores": 1,
        "memory_gb": 2,
        "precision": "float64",
        "wall_clock_seconds": 120,
    }
    contract = {
        "schema_version": 1,
        "canonical_task": {"path": "task.md", "sha256": digest("task.md")},
        "method": {"class": "fixed", "description": "fixed method"},
        "candidate": {"interface": "python_rpc_v1:handle", "editable_paths": ["program_0/src"]},
        "evaluators": {"search": "evaluate.py", "final": "evaluate_final.py"},
        "resources": {
            "search_root": "resources/search",
            "final_root": "resources/final",
            "public_artifact_bundle": None,
        },
        "data": {"search_roles": ["development"], "final_roles": ["final"]},
        "metrics": {
            "search": [{"name": "combined_score", "data_role": "development"}],
            "final": [{"name": "combined_score", "data_role": "final"}],
        },
        "upstream": [
            {
                "repository": "https://github.com/example/project",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "package_version": "1.0.0",
                "license": "MIT",
                "entrypoints": ["fixture:main"],
                "patches": [],
            }
        ],
        "provenance": {
            "dependencies": [{"name": "fixture", "version": "1.0.0"}],
            "artifacts": [
                {
                    "role": role,
                    "path": f"resources/{directory}/data.json",
                    "sha256": digest(f"resources/{directory}/data.json"),
                    "source": "generated fixture",
                }
                for role, directory in (("development", "search"), ("final", "final"))
            ],
        },
        "compute": {"search": plane, "final": plane},
        "protocol": {
            "search_seeds": [1],
            "final_seeds": [2],
            "final_reset_sequence": [2],
            "baselines": ["fixture"],
            "promotion_rule": "Freeze before final evaluation.",
            "search_proxies": [{"name": "small", "preserves": "interface", "omits": "scale"}],
        },
        "feedback": {"search_visible": ["aggregate"], "final_sealed": ["instances"]},
        "publication": {
            "search_is_proxy": True,
            "requires_final_evaluation": True,
            "required_evidence": ["hashes"],
        },
    }
    (task / "task_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    return task


_SNAPSHOTS = {
    "published": create_repair_snapshot,
    "digests": create_digest_synchronized_repair_snapshot,
    "dependencies": create_dependency_removal_repair_snapshot,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(task: Path) -> dict:
    return json.loads((task / "task_contract.json").read_text(encoding="utf-8"))


def _write_contract(task: Path, contract: dict) -> None:
    (task / "task_contract.json").write_text(json.dumps(contract), encoding="utf-8")


def _declare_artifact(task: Path, role: str, relative: str) -> None:
    contract = _contract(task)
    contract["provenance"]["artifacts"].append(
        {"role": role, "path": relative, "sha256": _sha256(task / relative), "source": "fixture"}
    )
    _write_contract(task, contract)


def _add_unused_dependency(task: Path, *, declare_lock: bool = False) -> Path:
    requirements = task / "requirements.txt"
    requirements.write_text("fixture==1.0.0\nunused==2.0.0\n", encoding="utf-8")
    contract = _contract(task)
    contract["provenance"]["dependencies"].append({"name": "unused", "version": "2.0.0"})
    _write_contract(task, contract)
    if declare_lock:
        _declare_artifact(task, "dependency_lock", "requirements.txt")
    return requirements


def _snapshot(task: Path, kind: str = "published") -> Path:
    snapshot = task.parent / "snapshot.json"
    _SNAPSHOTS[kind](task, snapshot)
    return snapshot


def _dependency_transaction(task: Path) -> tuple[Path, Path]:
    snapshot = _snapshot(task, "dependencies")
    transaction = task.parent / "transaction"
    create_repair_transaction(task, snapshot, transaction)
    return snapshot, transaction


def _evaluator_digest(task: Path) -> str:
    artifacts = _contract(task)["provenance"]["artifacts"]
    return next(item for item in artifacts if item["path"] == "evaluate.py")["sha256"]


def _assert_scope_rejected(task: Path, snapshot: Path, match: str, case: str) -> None:
    """Validate ``task`` against ``snapshot`` expecting rejection; ``case`` names the scenario."""
    with pytest.raises(TaskRepairScopeError, match=match):
        validate_repair_snapshot(task, snapshot)
    assert case


# --- snapshot scope -------------------------------------------------------------------------


def _weaken_method(task: Path) -> None:
    contract = _contract(task)
    contract["method"]["class"] = "weakened"
    _write_contract(task, contract)


def _relabel_provenance(task: Path) -> None:
    contract = _contract(task)
    contract["provenance"]["artifacts"][1]["role"] = "search_helper"
    _write_contract(task, contract)


def _create_runtime(task: Path) -> None:
    runtime_python = task / ".venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("tampered\n", encoding="utf-8")


def _write_unsafe(relative: str):
    return lambda task: (task / relative).write_text("unsafe\n", encoding="utf-8")


_OUT_OF_SCOPE_MUTATIONS = [
    ("task.md", _write_unsafe("task.md")),
    ("task_config.toml", _write_unsafe("task_config.toml")),
    ("final-resource", _write_unsafe("resources/final/data.json")),
    ("helper.py", _write_unsafe("helper.py")),
    ("immutable-contract", _weaken_method),
    ("provenance-relabel", _relabel_provenance),
    ("runtime-environment", _create_runtime),
]


def test_published_scope_allows_repair_artifacts_and_rejects_out_of_scope_changes(
    tmp_path: Path,
) -> None:
    # test_allows_published_repair_artifacts
    task = _write_task(tmp_path / "allowed")
    snapshot = _snapshot(task)
    (task / "evaluate.py").write_text("# repaired\n", encoding="utf-8")
    (task / "program_0/src/program.py").write_text("# baseline repair\n", encoding="utf-8")
    changed = validate_repair_snapshot(task, snapshot)
    assert "evaluate.py" in changed
    assert "program_0/src/program.py" in changed
    # test_rejects_out_of_scope_changes
    for name, mutate in _OUT_OF_SCOPE_MUTATIONS:
        task = _write_task(tmp_path / name)
        snapshot = _snapshot(task)
        mutate(task)
        _assert_scope_rejected(task, snapshot, "out-of-scope", name)
    # test_rejects_symlink_introduced_by_repair
    task = _write_task(tmp_path / "symlink")
    snapshot = _snapshot(task)
    (task / "resources/search/link").symlink_to(task / "resources/final/data.json")
    _assert_scope_rejected(task, snapshot, "symlink", "symlink")


def test_digest_synchronized_scope_accepts_harness_and_pending_digest_refresh(
    tmp_path: Path,
) -> None:
    # test_allows_harness_synchronized_contract_digests
    task = _write_task(tmp_path / "harness")
    _declare_artifact(task, "search_evaluator", "evaluate.py")
    snapshot = _snapshot(task, "digests")
    (task / "evaluate.py").write_text("# repaired\n", encoding="utf-8")
    assert synchronize_declared_digests(task / "task_contract.json", canonical_task=task / "task.md")
    assert validate_repair_snapshot(task, snapshot) == ("evaluate.py", "task_contract.json")
    # test_digest_synchronized_scope_allows_pending_harness_refresh
    task = _write_task(tmp_path / "pending")
    _declare_artifact(task, "search_evaluator", "evaluate.py")
    snapshot = _snapshot(task, "digests")
    (task / "evaluate.py").write_text("# repaired\n", encoding="utf-8")
    assert validate_repair_snapshot(task, snapshot) == ("evaluate.py",)
    assert _evaluator_digest(task) == _sha256(task / "evaluate.py")
    # test_digest_synchronized_scope_refreshes_intermediate_gate_digest
    task = _write_task(tmp_path / "intermediate")
    _declare_artifact(task, "search_evaluator", "evaluate.py")
    snapshot = _snapshot(task, "digests")
    (task / "evaluate.py").write_text("# first repair\n", encoding="utf-8")
    synchronize_declared_digests(task / "task_contract.json", canonical_task=task / "task.md")
    (task / "evaluate.py").write_text("# final repair\n", encoding="utf-8")
    validate_repair_snapshot(task, snapshot)
    assert _evaluator_digest(task) == _sha256(task / "evaluate.py")


def _set_contract_field(task: Path, field: str, value: object) -> None:
    contract = _contract(task)
    if field == "schema_version":
        contract[field] = value
    else:
        contract["compute"]["search"][field] = value
    _write_contract(task, contract)


def _duplicate_keys(task: Path) -> None:
    contract_path = task / "task_contract.json"
    contract_path.write_text(
        contract_path.read_text(encoding="utf-8").replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1
        ),
        encoding="utf-8",
    )


def _oversize_contract(task: Path) -> None:
    (task / "task_contract.json").write_text(" " * (MAX_TASK_CONTRACT_BYTES + 1), encoding="utf-8")


def _clear_dependencies(task: Path) -> None:
    contract = _contract(task)
    contract["provenance"]["dependencies"] = []
    _write_contract(task, contract)


_JSON_TYPE_CASES = [("gpu_count", 0, False), ("schema_version", 1, True), ("schema_version", 1, 1.0)]

_CONTRACT_TAMPERING = [
    ("duplicate-keys", "digests", _duplicate_keys, "duplicate JSON key"),
    ("oversized", "digests", _oversize_contract, "too large"),
    ("digests-semantic-change", "digests", _weaken_method, "semantic"),
    ("agent-contract-edit", "dependencies", _clear_dependencies, "semantic"),
]


def test_narrow_scopes_reject_contract_tampering_and_json_type_changes(tmp_path: Path) -> None:
    # test_digest_synchronized_scope_uses_json_type_strict_semantics
    for index, (field, original, replacement) in enumerate(_JSON_TYPE_CASES):
        task = _write_task(tmp_path / f"type-{index}")
        _set_contract_field(task, field, original)
        snapshot = _snapshot(task, "digests")
        _set_contract_field(task, field, replacement)
        _assert_scope_rejected(task, snapshot, "semantic", f"{field}:{original!r}->{replacement!r}")
    # test_narrow_scopes_reject_contract_tampering
    for name, kind, mutate, match in _CONTRACT_TAMPERING:
        task = _write_task(tmp_path / name)
        snapshot = _snapshot(task, kind)
        mutate(task)
        _assert_scope_rejected(task, snapshot, match, name)
    # test_digest_synchronized_scope_rejects_dependency_removal
    task = _write_task(tmp_path / "dependency-removal")
    requirements = _add_unused_dependency(task)
    snapshot = _snapshot(task, "digests")
    requirements.write_text("fixture==1.0.0\n", encoding="utf-8")
    _assert_scope_rejected(task, snapshot, "runtime_dependencies", "dependency-removal")


def test_dependency_removal_scope_synchronizes_pins_and_recovers_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # test_dependency_removal_scope_synchronizes_unused_pins_and_is_replay_safe
    task = _write_task(tmp_path / "replay")
    requirements = _add_unused_dependency(task, declare_lock=True)
    snapshot = _snapshot(task, "dependencies")
    requirements.write_text("fixture==1.0.0\n", encoding="utf-8")

    assert validate_repair_snapshot(task, snapshot) == ("requirements.txt",)
    synchronized = _contract(task)
    assert synchronized["provenance"]["dependencies"] == [{"name": "fixture", "version": "1.0.0"}]
    lock = next(a for a in synchronized["provenance"]["artifacts"] if a["path"] == "requirements.txt")
    assert lock["sha256"] == _sha256(requirements)

    validate_repair_snapshot(task, snapshot)
    assert _contract(task)["provenance"]["dependencies"] == [{"name": "fixture", "version": "1.0.0"}]

    # test_dependency_removal_scope_recovers_after_contract_publication_interruption
    task = _write_task(tmp_path / "interrupted")
    requirements = _add_unused_dependency(task)
    snapshot = _snapshot(task, "dependencies")
    requirements.write_text("fixture==1.0.0\n", encoding="utf-8")
    real_synchronize = repair_scope.synchronize_declared_dependency_removals

    def interrupt_after_contract_publication(published_contract_path: Path, **kwargs) -> bool:
        real_synchronize(published_contract_path, **kwargs)
        raise OSError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(
            repair_scope,
            "synchronize_declared_dependency_removals",
            interrupt_after_contract_publication,
        )
        with pytest.raises(OSError, match="simulated interruption"):
            validate_repair_snapshot(task, snapshot)

    validate_repair_snapshot(task, snapshot)
    assert _contract(task)["provenance"]["dependencies"] == [{"name": "fixture", "version": "1.0.0"}]


_SMOKE_POLICIES = [
    ("runtime_dependencies", "synchronized-dependency-removals"),
    ("runtime_baseline_validation", "synchronized-digests"),
    ("search_nonzero_exit", "synchronized-digests"),
    (None, "synchronized-digests"),
]


def test_dependency_removal_scope_rejects_expansion_and_smoke_category_selects_policy(
    tmp_path: Path,
) -> None:
    # test_dependency_removal_scope_rejects_dependency_expansion
    for index, requirements in enumerate(
        ["fixture==1.0.0\nnew-package==3.0.0\n", "fixture==1.1.0\n"]
    ):
        task = _write_task(tmp_path / f"expansion-{index}")
        snapshot = _snapshot(task, "dependencies")
        (task / "requirements.txt").write_text(requirements, encoding="utf-8")
        _assert_scope_rejected(task, snapshot, "remove pinned dependencies", requirements)
    # test_smoke_category_selects_narrow_contract_policy
    for category, expected in _SMOKE_POLICIES:
        diagnostic = {"schema_version": 1, "category": category}
        assert contract_policy_for_smoke_diagnostic(json.dumps(diagnostic)) == expected, category


# --- repair transactions --------------------------------------------------------------------


def test_dependency_repair_transaction_restores_mutable_new_and_parentless_files(
    tmp_path: Path,
) -> None:
    # test_dependency_repair_transaction_restores_every_mutable_file
    task = _write_task(tmp_path / "mutable")
    (task / "requirements.txt").write_text("fixture==1.0.0\nunused==2.0.0\n", encoding="utf-8")
    snapshot = _snapshot(task, "dependencies")
    before = {
        relative: (task / relative).read_bytes()
        for relative in (
            "evaluate.py",
            "evaluate_final.py",
            "program_0/src/program.py",
            "requirements.txt",
            "task_contract.json",
        )
    }
    transaction = task.parent / "transaction"
    create_repair_transaction(task, snapshot, transaction)
    for relative in before:
        (task / relative).write_text(f"changed {relative}\n", encoding="utf-8")

    restore_repair_transaction(task, transaction)
    restore_repair_transaction(task, transaction)

    assert {relative: (task / relative).read_bytes() for relative in before} == before
    commit_repair_transaction(task, transaction)
    assert not transaction.exists()

    # test_dependency_repair_transaction_removes_new_mutable_file
    task = _write_task(tmp_path / "new-file")
    requirements = task / "requirements.txt"
    requirements.unlink()
    _clear_dependencies(task)
    _, transaction = _dependency_transaction(task)
    requirements.write_text("fixture==1.0.0\n", encoding="utf-8")
    restore_repair_transaction(task, transaction)
    assert not requirements.exists()

    # test_dependency_repair_transaction_restores_deleted_entrypoint_parent
    task = _write_task(tmp_path / "parent")
    _, transaction = _dependency_transaction(task)
    entrypoint = task / "program_0/src/program.py"
    expected = entrypoint.read_bytes()
    entrypoint.unlink()
    entrypoint.parent.rmdir()
    restore_repair_transaction(task, transaction)
    assert entrypoint.read_bytes() == expected


def test_dependency_repair_transaction_recovers_interrupted_create_restore_and_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # test_dependency_repair_transaction_recovers_interrupted_create
    task = _write_task(tmp_path / "create")
    snapshot = _snapshot(task, "dependencies")
    transaction = task.parent / "transaction"
    real_replace = repair_transaction.os.replace

    def interrupt_publication(source: Path, destination: Path) -> None:
        if Path(destination) == transaction:
            raise OSError("simulated interrupted create")
        real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(repair_transaction.os, "replace", interrupt_publication)
        with pytest.raises(OSError, match="simulated interrupted create"):
            create_repair_transaction(task, snapshot, transaction)

    create_repair_transaction(task, snapshot, transaction)
    restore_repair_transaction(task, transaction)
    commit_repair_transaction(task, transaction)

    # test_dependency_repair_transaction_resumes_interrupted_restore
    task = _write_task(tmp_path / "restore")
    _, transaction = _dependency_transaction(task)
    evaluator = task / "evaluate.py"
    final_evaluator = task / "evaluate_final.py"
    expected_evaluator = evaluator.read_bytes()
    expected_final = final_evaluator.read_bytes()
    evaluator.write_text("changed search\n", encoding="utf-8")
    final_evaluator.write_text("changed final\n", encoding="utf-8")
    real_atomic_copy = repair_transaction._atomic_copy
    copied = 0

    def interrupt_after_first_copy(source: Path, destination: Path, mode: int) -> None:
        nonlocal copied
        real_atomic_copy(source, destination, mode)
        copied += 1
        if copied == 1:
            raise OSError("simulated interrupted restore")

    with monkeypatch.context() as patch:
        patch.setattr(repair_transaction, "_atomic_copy", interrupt_after_first_copy)
        with pytest.raises(OSError, match="simulated interrupted restore"):
            restore_repair_transaction(task, transaction)

    restore_repair_transaction(task, transaction)

    assert evaluator.read_bytes() == expected_evaluator
    assert final_evaluator.read_bytes() == expected_final

    # test_dependency_repair_transaction_recovers_interrupted_commit
    task = _write_task(tmp_path / "commit")
    _, transaction = _dependency_transaction(task)
    real_discard = repair_transaction._discard_committed_transaction
    interrupted = False

    def interrupt_cleanup(path: Path) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise OSError("simulated interrupted commit")
        real_discard(path)

    with monkeypatch.context() as patch:
        patch.setattr(repair_transaction, "_discard_committed_transaction", interrupt_cleanup)
        with pytest.raises(OSError, match="simulated interrupted commit"):
            commit_repair_transaction(task, transaction)

    commit_repair_transaction(task, transaction)
    assert not transaction.exists()


def test_dependency_repair_transaction_identity_ignores_journal_and_rejects_symlinked_target(
    tmp_path: Path,
) -> None:
    # test_dependency_repair_transaction_identity_ignores_mutable_journal
    task = _write_task(tmp_path / "identity")
    requirements = _add_unused_dependency(task)
    snapshot, transaction = _dependency_transaction(task)
    requirements.write_text("fixture==1.0.0\n", encoding="utf-8")
    validate_repair_snapshot(task, snapshot)

    create_repair_transaction(task, snapshot, transaction)

    restore_repair_transaction(task, transaction)
    commit_repair_transaction(task, transaction)

    # test_dependency_repair_transaction_rejects_symlinked_restore_target
    task = _write_task(tmp_path / "symlink")
    _, transaction = _dependency_transaction(task)
    evaluator = task / "evaluate.py"
    evaluator.unlink()
    evaluator.symlink_to(task / "task.md")
    with pytest.raises(TaskRepairScopeError, match="unsafe repair transaction target"):
        restore_repair_transaction(task, transaction)


def test_tree_repair_transaction_restores_changed_deleted_added_symlinked_and_special_files(
    tmp_path: Path,
) -> None:
    # test_tree_repair_transaction_restores_changed_deleted_and_added_files
    task = _write_task(tmp_path / "tree")
    transaction = task.parent / "transaction"
    before = repair_transaction._tree_files(task)
    create_tree_repair_transaction(task, transaction)
    (task / "evaluate.py").write_text("changed\n", encoding="utf-8")
    (task / "program_0/src/program.py").unlink()
    helper = task / "program_0/src/helpers/new.py"
    helper.parent.mkdir()
    helper.write_text("new\n", encoding="utf-8")

    restore_repair_transaction(task, transaction)

    assert repair_transaction._tree_files(task) == before
    assert not helper.exists()

    # test_tree_repair_transaction_removes_added_symlink_without_following_it
    task = _write_task(tmp_path / "symlink")
    transaction = task.parent / "transaction"
    victim = task.parent / "victim.txt"
    victim.write_text("preserve\n", encoding="utf-8")
    create_tree_repair_transaction(task, transaction)
    (task / "added-link").symlink_to(victim)
    restore_repair_transaction(task, transaction)
    assert victim.read_text(encoding="utf-8") == "preserve\n"
    assert not (task / "added-link").exists()

    # test_tree_repair_transaction_replaces_special_file_at_expected_path
    task = _write_task(tmp_path / "fifo")
    transaction = task.parent / "transaction"
    evaluator = task / "evaluate.py"
    expected = evaluator.read_bytes()
    create_tree_repair_transaction(task, transaction)
    evaluator.unlink()
    os.mkfifo(evaluator)
    restore_repair_transaction(task, transaction)
    assert evaluator.read_bytes() == expected


def test_tree_transaction_rejects_oversized_manifest_and_authored_tree_handles_runtime_and_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # test_tree_transaction_rejects_oversized_manifest_without_partial_state
    task = _write_task(tmp_path / "oversized")
    transaction = task.parent / "transaction"
    creating = task.parent / ".transaction.creating"
    with monkeypatch.context() as patch:
        patch.setattr(repair_transaction, "_MAX_MANIFEST_BYTES", 256)
        with pytest.raises(TaskRepairScopeError, match="manifest is too large"):
            create_tree_repair_transaction(task, transaction)
    assert not transaction.exists()
    assert not creating.exists()

    # test_authored_tree_transaction_discards_managed_runtime_before_capture
    task = _write_task(tmp_path / "runtime")
    (task / ".venv/lib").mkdir(parents=True)
    (task / ".venv/lib64").symlink_to("lib", target_is_directory=True)
    (task / ".agentic_runtime.json").write_text("{}\n", encoding="utf-8")
    evaluator = task / "evaluate.py"
    expected = evaluator.read_bytes()
    transaction = task.parent / "transaction"

    create_authored_tree_repair_transaction(task, transaction)
    evaluator.write_text("changed\n", encoding="utf-8")
    restore_repair_transaction(task, transaction)

    assert evaluator.read_bytes() == expected
    assert not (task / ".venv").exists()
    assert not (task / ".agentic_runtime.json").exists()

    # test_authored_tree_transaction_rejects_authored_symlink
    task = _write_task(tmp_path / "symlink")
    protected = task.parent / "protected.py"
    protected.write_text("protected\n", encoding="utf-8")
    evaluator = task / "evaluate.py"
    evaluator.unlink()
    evaluator.symlink_to(protected)
    with pytest.raises(TaskRepairScopeError, match="tree contains symlink"):
        create_authored_tree_repair_transaction(task, task.parent / "transaction")
    assert protected.read_text(encoding="utf-8") == "protected\n"


def test_settle_repair_transaction_rolls_back_uncommitted_or_finishes_linearized_commit(
    tmp_path: Path,
) -> None:
    # test_settle_repair_transaction_rolls_back_uncommitted_repair
    task = _write_task(tmp_path / "rollback")
    transaction = task.parent / "transaction"
    evaluator = task / "evaluate.py"
    expected = evaluator.read_bytes()
    create_tree_repair_transaction(task, transaction)
    evaluator.write_text("provisional\n", encoding="utf-8")
    (task / ".venv/lib").mkdir(parents=True)
    (task / ".venv/lib64").symlink_to("lib", target_is_directory=True)

    assert settle_repair_transaction(task, transaction) == "rolled_back"
    assert evaluator.read_bytes() == expected
    assert not (task / ".venv").exists()

    # test_settle_repair_transaction_finishes_linearized_commit
    task = _write_task(tmp_path / "commit")
    transaction = task.parent / "transaction"
    evaluator = task / "evaluate.py"
    create_tree_repair_transaction(task, transaction)
    evaluator.write_text("accepted\n", encoding="utf-8")
    committing, _ = repair_transaction._commit_marker_path(transaction)
    os.replace(transaction, committing)

    assert settle_repair_transaction(task, transaction) == "committed"
    assert evaluator.read_text(encoding="utf-8") == "accepted\n"
    assert not committing.exists()


# --- generated bytecode quarantine ----------------------------------------------------------


def _compile_evaluator(task: Path) -> Path:
    bytecode = Path(importlib.util.cache_from_source(str(task / "evaluate.py")))
    py_compile.compile(str(task / "evaluate.py"), cfile=str(bytecode), doraise=True)
    return bytecode


def test_quarantine_removes_generated_bytecode_and_accepts_configured_foreign_cache(
    tmp_path: Path,
) -> None:
    # test_removes_generated_bytecode_before_repair_validation
    task = _write_task(tmp_path / "generated")
    snapshot = _snapshot(task)
    cache = task / "__pycache__"
    cache.mkdir()
    bytecode = _compile_evaluator(task)
    (task / "evaluate.py").write_text("# repaired\n", encoding="utf-8")
    _compile_evaluator(task)

    quarantine = task.parent / "quarantine"
    manifest = task.parent / "manifest.json"
    quarantined = quarantine_generated_bytecode(task, quarantine, manifest)

    assert validate_repair_snapshot(task, snapshot) == ("evaluate.py",)
    assert not cache.exists()
    assert quarantined == (bytecode.relative_to(task).as_posix(),)
    quarantined_bytecode = quarantine / bytecode.relative_to(task)
    assert quarantined_bytecode.is_file()
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["schema_version"] == 1
    assert manifest_payload["artifacts"] == [
        {
            "magic": quarantined_bytecode.read_bytes()[:4].hex(),
            "path": bytecode.relative_to(task).as_posix(),
            "sha256": _sha256(quarantined_bytecode),
            "size": quarantined_bytecode.stat().st_size,
        }
    ]

    # test_accepts_hash_bytecode_from_configured_foreign_interpreter
    task = _write_task(tmp_path / "foreign")
    source = task / "evaluate.py"
    cache = task / "__pycache__"
    cache.mkdir()
    foreign_tag = "cpython-999"
    foreign_magic = b"\x42\x0d\x0d\x0a"
    source_hash = _imp.source_hash(int.from_bytes(foreign_magic, "little"), source.read_bytes())
    bytecode = cache / f"evaluate.{foreign_tag}.pyc"
    bytecode.write_bytes(foreign_magic + struct.pack("<I", 3) + source_hash + b"payload")

    quarantined = quarantine_generated_bytecode(
        task, task.parent / "quarantine", accepted_caches=frozenset({(foreign_tag, foreign_magic)})
    )
    assert quarantined == (f"__pycache__/{bytecode.name}",)


def test_quarantine_rejects_unexpected_corrupted_or_mutated_cache_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # test_rejects_non_bytecode_hidden_in_cache_directory
    task = _write_task(tmp_path / "hidden")
    cache = task / "__pycache__"
    cache.mkdir()
    (cache / "payload.py").write_text("unsafe\n", encoding="utf-8")
    with pytest.raises(TaskRepairScopeError, match="unexpected repair cache artifact"):
        quarantine_generated_bytecode(task, task.parent / "quarantine")

    # test_rejects_bytecode_for_untrusted_interpreter
    for corruption in ("magic", "tag"):
        task = _write_task(tmp_path / f"untrusted-{corruption}")
        bytecode = _compile_evaluator(task)
        if corruption == "magic":
            payload = bytearray(bytecode.read_bytes())
            payload[0] ^= 0xFF
            bytecode.write_bytes(payload)
        else:
            bytecode.rename(
                bytecode.with_name(
                    bytecode.name.replace(str(sys.implementation.cache_tag), "cpython-999")
                )
            )
        with pytest.raises(TaskRepairScopeError, match="unexpected repair cache artifact"):
            quarantine_generated_bytecode(task, task.parent / "quarantine")

    # test_cache_validation_finishes_before_deleting_artifacts
    task = _write_task(tmp_path / "validation-first")
    cache = task / "__pycache__"
    cache.mkdir()
    bytecode = _compile_evaluator(task)
    (cache / "payload.pyc").write_bytes(b"unsafe")
    with pytest.raises(TaskRepairScopeError, match="unexpected repair cache artifact"):
        quarantine_generated_bytecode(task, task.parent / "quarantine")
    assert bytecode.is_file()

    # test_rejects_cache_mutated_during_quarantine
    task = _write_task(tmp_path / "mutated")
    bytecode = _compile_evaluator(task)
    real_replace = repair_bytecode.os.replace

    def mutate_after_replace(source: Path, destination: Path) -> None:
        real_replace(source, destination)
        if Path(source).name == "__pycache__":
            moved = Path(destination) / bytecode.name
            moved.write_bytes(moved.read_bytes() + b"changed")

    with monkeypatch.context() as patch:
        patch.setattr(repair_bytecode.os, "replace", mutate_after_replace)
        with pytest.raises(TaskRepairScopeError, match="changed while"):
            quarantine_generated_bytecode(task, task.parent / "quarantine")


# --- host gate authorization for dependency failures ----------------------------------------


def _make_repair(root: Path):
    task = _write_task(root)
    private = task.parent / "private"
    private.mkdir(mode=0o700)
    snapshot = private / "snapshot.json"
    create_digest_synchronized_repair_snapshot(task, snapshot)
    transaction = private / "transaction"
    create_repair_transaction(task, snapshot, transaction)
    return task, snapshot, transaction, private / "authorization.json", "session-1"


def _failure(task: Path, operation: str = "runtime") -> TaskGateReport:
    return TaskGateReport.authored_failure(
        gate=operation, component="runtime", stage="bootstrap", category="runtime_dependencies",
        scaffold_fingerprint=task_tree_fingerprint(task), exit_code=3,
        elapsed_seconds=0.1, diagnostic="No matching distribution found",
    )


def _executor(repair, run=None):
    task, snapshot, transaction, authorization, session = repair
    return DependencyRepairExecutor(
        task, snapshot, transaction, authorization, session,
        run or (lambda operation: _failure(task, operation)),
    )


def test_authorized_repair_transitions_preserve_snapshot_receipt_and_chain(tmp_path):
    # test_capability_repair_transitions_without_mutating_original_snapshot
    repair = _make_repair(tmp_path / "transition")
    task, snapshot, _, authorization, _ = repair
    snapshot_before = snapshot.read_bytes()
    (task / "evaluate.py").write_text("# repaired environment allowlist\n")
    report = _executor(repair)("runtime")
    assert authorization.is_file()
    assert "Harness authorized dependency removal" in report.diagnostic
    (task / "requirements.txt").write_text("")
    validate_authorized_repair(*repair)
    assert _contract(task)["provenance"]["dependencies"] == []
    assert snapshot.read_bytes() == snapshot_before
    validate_authorized_repair(*repair)

    # test_repeated_gate_preserves_authorization_after_removal
    repair = _make_repair(tmp_path / "repeated")
    task, _, _, authorization, _ = repair
    gate = _executor(repair)
    gate("runtime")
    original = authorization.read_bytes()
    (task / "requirements.txt").write_text("")
    gate("runtime")
    assert authorization.read_bytes() == original
    validate_authorized_repair(*repair)

    # test_earlier_chain_transaction_can_bind_new_attempt
    repair = _make_repair(tmp_path / "chain")
    task, snapshot, transaction, authorization, session = repair
    (task / "evaluate.py").write_text("# previous valid repair\n")
    create_digest_synchronized_repair_snapshot(task, snapshot)
    _executor(repair)("runtime")
    assert authorization.exists()
    (task / "requirements.txt").write_text("")
    validate_authorized_repair(task, snapshot, transaction, authorization, session)


def test_authorization_survives_gate_runtime_initial_dependency_scope_and_cli_validate(tmp_path):
    # test_gate_created_runtime_does_not_prevent_authorization
    repair = _make_repair(tmp_path / "runtime")
    task, *_ = repair

    def run(operation):
        (task / ".venv").mkdir()
        (task / ".venv/python").symlink_to("/usr/bin/python")
        return _failure(task, operation)

    assert "Harness authorized" in _executor(repair, run)("runtime").diagnostic

    # test_initial_dependency_repair_retains_existing_authorization
    repair = _make_repair(tmp_path / "initial")
    task, snapshot, _, authorization, _ = repair
    create_dependency_removal_repair_snapshot(task, snapshot)
    result = _executor(repair)("runtime")
    assert result.category == "runtime_dependencies"
    assert not authorization.exists()
    (task / "requirements.txt").write_text("")
    validate_authorized_repair(*repair)

    # test_cli_authorized_removal_synchronizes_contract
    repair = _make_repair(tmp_path / "cli")
    task, snapshot, transaction, authorization, session = repair
    _executor(repair)("runtime")
    (task / "requirements.txt").write_text("")
    assert authorization_module.main([
        "validate", "--task-dir", str(task), "--snapshot", str(snapshot),
        "--transaction-dir", str(transaction), "--authorization", str(authorization),
        "--session-id", session,
    ]) == 0
    assert _contract(task)["provenance"]["dependencies"] == []


def test_gate_withholds_authorization_without_receipt_for_edits_or_stale_fingerprint(tmp_path):
    # test_missing_authorization_keeps_original_scope
    repair = _make_repair(tmp_path / "missing")
    task, *_ = repair
    (task / "requirements.txt").write_text("")
    with pytest.raises(TaskRepairScopeError, match="runtime_dependencies"):
        validate_authorized_repair(*repair)

    # test_pre_authorization_edits_rejected
    for edit in ("dependencies", "contract", "resource"):
        repair = _make_repair(tmp_path / f"edit-{edit}")
        task, _, _, authorization, _ = repair
        if edit == "dependencies":
            (task / "requirements.txt").write_text("")
        elif edit == "contract":
            payload = _contract(task)
            payload["method"]["description"] = "different science"
            _write_contract(task, payload)
        else:
            (task / "resources/search/data.json").write_text("different")
        report = _executor(repair)("runtime")
        assert report.category == "repair_authorization_failed", edit
        assert not authorization.exists(), edit

    # test_stale_gate_fingerprint_does_not_authorize
    repair = _make_repair(tmp_path / "stale")
    task, _, _, authorization, _ = repair
    report = _failure(task)
    (task / "evaluate.py").write_text("# later edit\n")
    result = _executor(repair, lambda operation: report)("runtime")
    assert result.category == "repair_authorization_failed"
    assert not authorization.exists()


def test_gate_withholds_authorization_for_infrastructure_status_publication_failure_and_expansion(
    tmp_path, monkeypatch
):
    # test_runtime_failure_infrastructure_status_does_not_authorize
    repair = _make_repair(tmp_path / "infrastructure")
    task, _, _, authorization, _ = repair
    report = replace(_failure(task), status="infrastructure_failure", exit_code=4)
    assert _executor(repair, lambda operation: report)("runtime") == report
    assert not authorization.exists()

    # test_receipt_publication_failure_never_announces_authorization
    repair = _make_repair(tmp_path / "publication")

    def fail(*args):
        raise TaskRepairScopeError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(authorization_module, "_publish_authorization", fail)
        report = _executor(repair)("runtime")
    assert report.category == "repair_authorization_failed"
    assert "Harness authorized" not in report.diagnostic

    # test_authorization_does_not_allow_additions_or_version_changes
    for index, requirements in enumerate(["fixture==2.0.0\n", "fixture==1.0.0\nnew==1.0.0\n"]):
        repair = _make_repair(tmp_path / f"expansion-{index}")
        task, *_ = repair
        _executor(repair)("runtime")
        (task / "requirements.txt").write_text(requirements)
        with pytest.raises(TaskRepairScopeError, match="only remove pinned dependencies"):
            validate_authorized_repair(*repair)


_RECEIPT_BINDINGS = [
    ("session_id", "different"), ("snapshot_sha256", "0" * 64),
    ("transaction_sha256", "0" * 64), ("task_dir", "/different"),
    ("scaffold_fingerprint", "invalid"),
]


def test_receipt_with_broken_bindings_plain_gate_summary_or_symlink_is_rejected(tmp_path):
    # test_receipt_bindings_rejected
    for field, value in _RECEIPT_BINDINGS:
        repair = _make_repair(tmp_path / f"binding-{field}")
        _, _, _, authorization, _ = repair
        _executor(repair)("runtime")
        payload = json.loads(authorization.read_text())
        payload[field] = value
        authorization.write_text(json.dumps(payload))
        with pytest.raises(TaskRepairScopeError, match="authorization"):
            validate_authorized_repair(*repair)

    # test_ordinary_gate_summary_cannot_authorize
    repair = _make_repair(tmp_path / "summary")
    task, _, _, authorization, _ = repair
    authorization.write_text(json.dumps(_failure(task).to_payload()))
    authorization.chmod(0o600)
    with pytest.raises(TaskRepairScopeError, match="authorization"):
        validate_authorized_repair(*repair)

    # test_symlink_receipt_rejected
    repair = _make_repair(tmp_path / "symlink")
    _, _, _, authorization, _ = repair
    target = tmp_path / "unrelated"
    target.write_text("unchanged")
    authorization.symlink_to(target)
    with pytest.raises(TaskRepairScopeError, match="authorization"):
        validate_authorized_repair(*repair)
    assert target.read_text() == "unchanged"


def test_authorization_parent_symlink_or_public_mode_is_rejected(tmp_path):
    # test_private_parent_symlink_rejected
    repair = _make_repair(tmp_path / "symlink")
    task, snapshot, transaction, authorization, session = repair
    link = tmp_path / "link"
    link.symlink_to(authorization.parent, target_is_directory=True)
    with pytest.raises(TaskRepairScopeError, match="authorization"):
        validate_authorized_repair(task, snapshot, transaction, link / authorization.name, session)

    # test_public_authorization_parent_rejected
    repair = _make_repair(tmp_path / "public")
    _, _, _, authorization, _ = repair
    authorization.parent.chmod(0o755)
    with pytest.raises(TaskRepairScopeError, match="host-private"):
        validate_authorized_repair(*repair)


def _service_arguments(repair):
    task, snapshot, transaction, authorization, session = repair
    token = authorization.parent / "token"
    token.write_text("fixture-token")
    return [
        "service", "--socket", str(authorization.parent / "gate.sock"),
        "--token-file", str(token), "--task-dir", str(task),
        "--source-task-dir", str(task), "--container-image", "fixture-image",
        "--container-network", "none", "--timeout-seconds", "10",
        "--production-entry-timeout-seconds", "10", "--max-runtime-bytes", "1024",
        "--max-requests", "1", "--receipt-path", str(authorization.parent / "receipt"),
        "--runtime-created-marker", str(authorization.parent / "marker"),
    ]


def test_service_rejects_partial_repair_arguments_and_wraps_gate_with_private_authorization(
    tmp_path, monkeypatch
):
    # test_service_partial_repair_arguments_rejected
    repair = _make_repair(tmp_path / "partial")
    with monkeypatch.context() as patch:
        patch.setattr("sys.argv", _service_arguments(repair) + ["--repair-session-id", "session"])
        with pytest.raises(SystemExit) as error:
            service._main()
    assert error.value.code == 2

    # test_service_wraps_gate_with_private_authorization
    repair = _make_repair(tmp_path / "wrapped")
    task, snapshot, transaction, authorization, session = repair
    monkeypatch.setattr("sys.argv", _service_arguments(repair) + [
        "--repair-snapshot", str(snapshot), "--repair-transaction", str(transaction),
        "--repair-authorization", str(authorization), "--repair-session-id", session,
    ])
    monkeypatch.setattr(service, "run_gate", lambda operation, settings: _failure(task, operation))
    monkeypatch.setattr(service.signal, "signal", lambda *args: None)
    observed = []

    class Server:
        def __init__(self, socket, *, executor, **kwargs):
            self.executor = executor

        def serve(self):
            observed.append(self.executor("runtime"))
            assert authorization.is_file()

        def close(self):
            pass

    monkeypatch.setattr(service, "GateServer", Server)
    assert service._main() == 0
    assert "Harness authorized" in observed[0].diagnostic


def test_authored_scan_rejects_swapped_files_enforces_byte_budget_and_matches_fingerprint(
    tmp_path, monkeypatch
):
    # test_authored_scan_rejects_file_swapped_before_open
    original_open = os.open
    for replacement in ("symlink", "fifo"):
        task, *_ = _make_repair(tmp_path / f"swap-{replacement}")
        swapped = False

        def swap(path, flags, *args, _task=task, _replacement=replacement, **kwargs):
            nonlocal swapped
            if path == "evaluate.py" and not swapped:
                swapped = True
                target = _task / "evaluate.py"
                target.unlink()
                if _replacement == "symlink":
                    target.symlink_to("/dev/zero")
                else:
                    os.mkfifo(target)
            return original_open(path, flags, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(authorization_module.os, "open", swap)
            with pytest.raises(TaskRepairScopeError, match="authored"):
                authorization_module._authored_state(task)

    # test_authored_scan_enforces_read_byte_budget
    task, *_ = _make_repair(tmp_path / "budget")
    with monkeypatch.context() as patch:
        patch.setattr(authorization_module, "_MAX_AUTHORED_BYTES", 10, raising=False)
        with pytest.raises(TaskRepairScopeError, match="limit"):
            authorization_module._authored_state(task)

    # test_authored_scan_digest_matches_gate_fingerprint
    task, *_ = _make_repair(tmp_path / "fingerprint")
    files, fingerprint = authorization_module._authored_state(task)
    assert fingerprint == task_tree_fingerprint(task)
    assert "evaluate.py" in files
