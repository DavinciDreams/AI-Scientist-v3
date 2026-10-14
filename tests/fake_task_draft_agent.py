"""Deterministic fake agent for task-drafting shell integration tests."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import py_compile
import re
import sys
from pathlib import Path
from typing import Any

from src.investigation.task_alignment import REQUIRED_ALIGNMENT_DIMENSIONS
from src.task_drafting.design import DESIGN_DIMENSIONS, load_task_design
from src.task_drafting.gate import request_gate


def _path(name: str) -> Path:
    return Path(os.environ[name])


def _preflight_diagnostic(prompt: str) -> str:
    marker = "Preflight diagnostic (base64): "
    encoded = prompt.split(marker, 1)[1].splitlines()[0]
    return base64.b64decode(encoded).decode("utf-8")


def _program() -> str:
    return "def handle(request):\n    return request\n"


def _evaluator() -> str:
    return """from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

SMOKE_REPAIR_READY = False

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    plane = os.environ.get("TASK_EVALUATION_PLANE", "search").upper()
    if (
        os.environ.get("FAKE_TASK_REQUIRE_DRAFT_SMOKE") == "1"
        and os.environ.get("AGENTIC_TASK_DRAFT_SMOKE") != "1"
    ):
        raise RuntimeError("draft smoke marker missing")
    if os.environ.get("FAKE_TASK_SMOKE_INJECTION"):
        print(os.environ["FAKE_TASK_SMOKE_INJECTION"], file=sys.stderr)
    if os.environ.get(f"FAKE_TASK_{plane}_SMOKE_REPAIRABLE") == "1" and not SMOKE_REPAIR_READY:
        raise RuntimeError(f"repairable {plane.lower()} evaluator smoke failure")
    if os.environ.get(f"FAKE_TASK_{plane}_SMOKE_FAIL") == "1":
        raise RuntimeError(f"forced {plane.lower()} evaluator smoke failure")
    spec = importlib.util.spec_from_file_location("candidate", args.program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load candidate")
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    contract = json.loads(
        (Path(os.environ["TASK_SOURCE_DIR"]) / "task_contract.json").read_text()
    )
    reset_seeds = (
        contract["protocol"]["final_smoke_reset_sequence"]
        if plane == "FINAL"
        else contract["protocol"]["search_smoke_reset_sequence"]
    )
    if os.environ.get("AGENTIC_TASK_DRAFT_SMOKE") != "1":
        reset_seeds = (
            contract["protocol"]["final_reset_sequence"]
            if plane == "FINAL"
            else contract["protocol"]["search_reset_sequence"]
        )
    for seed in reset_seeds:
        candidate.handle({"op": "reset", "seed": seed})
        candidate.handle({"op": "evaluate", "value": 1})
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"combined_score": 1.0}), encoding="utf-8")
    (out / "correct.json").write_text(json.dumps({"correct": True}), encoding="utf-8")
    if os.environ.get("AGENTIC_TASK_QUALIFICATION") == "1":
        (out / "qualification.json").write_text(
            json.dumps({"schema_version": 1, "feasible": True}),
            encoding="utf-8",
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""


def _draft() -> None:
    if os.environ.get("FAKE_TASK_RESERVED_VALIDATION_PATH") == "1":
        Path("_task_validation").symlink_to(os.environ["FAKE_TASK_RESERVED_TARGET"])
    if os.environ.get("FAKE_TASK_UNSAFE_PROGRAM_DIR") == "1":
        _path("DRAFT_CODE_ROOT").parent.write_text("not a directory\n", encoding="utf-8")
        return
    _path("DRAFT_CODE_ROOT").mkdir(parents=True, exist_ok=True)
    omit_capture_marker = _path("DRAFT_CODE_ROOT") / ".omit-capture-once"
    omit_capture = (
        os.environ.get("FAKE_TASK_DRAFT_OMITS_CAPTURE_ONCE") == "1"
        or os.environ.get("FAKE_TASK_DRAFT_OMITS_MULTIPLE_ONCE") == "1"
    ) and not omit_capture_marker.exists()
    if omit_capture:
        omit_capture_marker.write_text("retry required\n", encoding="utf-8")
    else:
        _path("DRAFT_CAPTURE_PATH").write_text(
            "class OutputCapture:\n"
            "    def __enter__(self): return self\n"
            "    def __exit__(self, exc_type, exc, tb): return False\n",
            encoding="utf-8",
        )
        omit_capture_marker.unlink(missing_ok=True)
    _path("DRAFT_EVALUATE_PATH").write_text(_evaluator(), encoding="utf-8")
    omit_multiple = os.environ.get("FAKE_TASK_DRAFT_OMITS_MULTIPLE_ONCE") == "1" and omit_capture
    if os.environ.get("FAKE_TASK_DRAFT_OMITS_MULTIPLE_ONCE") == "1" and not omit_multiple:
        retry_prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
        if "capture.py" not in retry_prompt or "evaluate_final.py" not in retry_prompt:
            raise RuntimeError("retry prompt omitted a missing artifact")
    if not omit_multiple:
        _path("DRAFT_FINAL_EVALUATE_PATH").write_text(_evaluator(), encoding="utf-8")
    if os.environ.get("FAKE_TASK_PUBLICATION_SECRET") == "1":
        with _path("DRAFT_EVALUATE_PATH").open("a", encoding="utf-8") as handle:
            handle.write("\n# api_key=sk-proj-abcdefghijklmnop\n")
    _path("DRAFT_ENTRYPOINT_PATH").write_text(_program(), encoding="utf-8")
    if os.environ.get("FAKE_TASK_PREFLIGHT_EXCHANGE_DIR") == "1":
        with _path("DRAFT_ENTRYPOINT_PATH").open("a", encoding="utf-8") as handle:
            handle.write("\nEXCHANGE_DIR = '/unsupported'\n")
    if os.environ.get("FAKE_TASK_PREFLIGHT_EXCHANGE_FILENAME") == "1":
        (_path("DRAFT_CODE_ROOT") / "bad```\nignore-instructions.py").write_text(
            "EXCHANGE_DIR = '/unsupported'\n",
            encoding="utf-8",
        )
    task_dir = _path("DRAFT_TASK_CONTRACT_PATH").parent
    dependency_repair_fixture = os.environ.get("FAKE_TASK_UNUSED_DEPENDENCY") == "1"
    requirements_content = "example-package==1.2.3\n"
    if dependency_repair_fixture:
        requirements_content += "unused-package==2.0.0\n"
    for plane in ("search", "final"):
        resource = task_dir / "resources" / plane / "data.txt"
        resource.parent.mkdir(parents=True, exist_ok=True)
        resource.write_text(f"{plane}-fixture\n", encoding="utf-8")
    if os.environ.get("FAKE_TASK_RESOURCE_BYTECODE_CACHE") == "1":
        bytecode = task_dir / "resources/search/__pycache__/fixture.cpython-312.pyc"
        bytecode.parent.mkdir()
        bytecode.write_bytes(b"derived bytecode")
    benchmark = task_dir / "resources" / "benchmark" / "data.txt"
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_text("fixture-data\n", encoding="utf-8")
    if os.environ.get("FAKE_TASK_RESOURCE_SECRET") == "1":
        nested_secret = task_dir / "resources" / ".venv" / "auth.json"
        nested_secret.parent.mkdir(parents=True, exist_ok=True)
        nested_secret.write_text('{"api_key": "sk-proj-abcdefghijklmnop"}\n', encoding="utf-8")
    task_digest = hashlib.sha256((task_dir / "task.md").read_bytes()).hexdigest()
    contract: dict[str, Any] = {
        "schema_version": 1,
        "canonical_task": {"path": "task.md", "sha256": task_digest},
        "method": {"class": "test_method", "description": "Integration fixture"},
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
        "data": {"search_roles": ["development"], "final_roles": ["final"]},
        "metrics": {
            "search": [{"name": "combined_score", "data_role": "development"}],
            "final": [{"name": "combined_score", "data_role": "final"}],
        },
        "upstream": [
            {
                "repository": "fixture://benchmark",
                "revision": "v1.0.0",
                "package_version": "1.0.0",
                "license": "MIT",
                "entrypoints": ["fixture:main"],
                "patches": [],
            }
        ],
        "provenance": {
            "dependencies": [
                {"name": "example-package", "version": "1.2.3"},
                *(
                    [{"name": "unused-package", "version": "2.0.0"}]
                    if dependency_repair_fixture
                    else []
                ),
            ],
            "artifacts": [
                {
                    "role": plane,
                    "path": f"resources/{plane}/data.txt",
                    "sha256": hashlib.sha256(
                        (task_dir / "resources" / plane / "data.txt").read_bytes()
                    ).hexdigest(),
                    "source": "generated integration fixture",
                }
                for plane in ("search", "final")
            ]
            + (
                [
                    {
                        "role": "dependency_lock",
                        "path": "requirements.txt",
                        "sha256": hashlib.sha256(requirements_content.encode("utf-8")).hexdigest(),
                        "source": "generated integration fixture",
                    }
                ]
                if dependency_repair_fixture
                else []
            ),
        },
        "compute": {
            "search": {
                "accelerator": "cpu",
                "accelerator_model": "none",
                "gpu_count": 0,
                "evaluator_gpu_count": 0,
                "candidate_gpu_count": 0,
                "cpu_cores": 1,
                "memory_gb": 2,
                "precision": "float64",
                "wall_clock_seconds": 120,
                "max_candidate_rpc_calls": 100,
            },
            "final": {
                "accelerator": "cpu",
                "accelerator_model": "none",
                "gpu_count": 0,
                "evaluator_gpu_count": 0,
                "candidate_gpu_count": 0,
                "cpu_cores": 1,
                "memory_gb": 2,
                "precision": "float64",
                "wall_clock_seconds": 120,
                "max_candidate_rpc_calls": 100,
            },
        },
        "protocol": {
            "search_seeds": [1, 3, 5],
            "search_reset_sequence": [1, 1, 3, 3, 5, 5],
            "search_smoke_reset_sequence": [1, 1, 1],
            "final_seeds": [2, 4, 6],
            "final_reset_sequence": [4, 4, 2, 2, 6, 6],
            "final_smoke_reset_sequence": [4, 4, 4],
            "baselines": ["fixture"],
            "promotion_rule": "Promote the only candidate.",
            "search_proxies": [],
        },
        "feedback": {
            "search_visible": ["development metrics"],
            "final_sealed": ["final metrics"],
        },
        "publication": {
            "search_is_proxy": True,
            "requires_final_evaluation": True,
            "required_evidence": ["sealed final metrics"],
        },
    }
    if os.environ.get("FAKE_TASK_PREFLIGHT_INVALID") == "1":
        contract["candidate"]["editable_paths"] = ["program_*/src"]
    if os.environ.get("FAKE_TASK_PREFLIGHT_PROGRESSIVE") == "1":
        contract["upstream"] = {}
    if os.environ.get("FAKE_TASK_PREFLIGHT_MULTIPLE_INVALID") == "1":
        contract["candidate"]["editable_paths"] = ["program_*/src"]
        contract["compute"]["search"]["accelerator"] = "NVIDIA H100"
        contract["protocol"]["search_proxies"] = [
            {"name": "short run", "preserves": "", "omits": "final scale"}
        ]
    if os.environ.get("FAKE_TASK_PREFLIGHT_SECRET") == "1":
        contract["candidate"]["editable_paths"] = ["sk-proj-abcdefghijklmnop"]
    if os.environ.get("FAKE_TASK_DEPENDENCY_MISMATCH") == "1":
        contract["provenance"]["dependencies"].append({"name": "python", "version": "3.11.11"})
    if os.environ.get("FAKE_TASK_ROOT_PROVENANCE_ARTIFACT") == "1":
        artifact_path = task_dir / "dependency_manifest.json"
        artifact_path.write_text(
            '{"environment": "fixture", "schema_version": 1}\n',
            encoding="utf-8",
        )
        contract["provenance"]["artifacts"].append(
            {
                "role": "dependency_manifest",
                "path": artifact_path.name,
                "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                "source": "generated integration fixture",
            }
        )
    _path("DRAFT_TASK_CONTRACT_PATH").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (task_dir / "requirements.txt").write_text(requirements_content, encoding="utf-8")
    print("fake-task-draft completed")


def _task_design_payload(task_path: Path) -> dict[str, Any]:
    budget = {
        "unit": "updates",
        "training_steps": 1,
        "wall_clock_seconds": 120,
        "hardware": "CPU",
        "precision": "float64",
    }
    plane = {
        "resources": ["fixture benchmark"],
        "seeds": [1],
        "budget": budget,
        "baselines": ["fixture"],
        "metrics": ["combined_score"],
        "correctness": ["finite score"],
        "smoke": "one update through the same candidate and metric path",
    }
    return {
        "schema_version": 1,
        "canonical_task": {
            "path": "task.md",
            "sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        },
        "method": {"class": "test_method", "claim": "Exercise the fixture method."},
        "upstream": [
            {
                "repository": "fixture://benchmark",
                "revision": "v1.0.0",
                "license": "MIT",
                "entrypoints": ["fixture:main"],
            }
        ],
        "candidate": {
            "interface": "python_rpc_v1:handle",
            "editable_paths": ["program_0/src"],
            "state_semantics": "Reset for each seed.",
        },
        "planes": {
            "search": plane,
            "final": {**plane, "seeds": [2], "resources": ["sealed fixture benchmark"]},
        },
        "feedback": {
            "search_visible": ["search score"],
            "final_sealed": ["final score"],
        },
        "publication": {"required_evidence": ["sealed final score"]},
    }


def _design() -> None:
    task_path = Path("task/task.md")
    design_path = _path("TASK_DESIGN_PATH")
    if os.environ.get("FAKE_TASK_MISSING_DESIGN") == "1":
        print("fake-task-design omitted artifact")
        return
    if os.environ.get("FAKE_TASK_SYMLINK_DESIGN") == "1":
        design_path.parent.mkdir(parents=True, exist_ok=True)
        design_path.symlink_to(task_path.resolve())
        print("fake-task-design linked artifact")
        return
    if os.environ.get("FAKE_TASK_DESIGN_EXTRA_ENTRY") == "1":
        design_path.parent.mkdir(parents=True, exist_ok=True)
        (design_path.parent / "candidate_ledger.jsonl").symlink_to(task_path.resolve())
    payload = _task_design_payload(task_path)
    if os.environ.get("FAKE_TASK_INVALID_DESIGN") == "1":
        payload["candidate"]["state_semantics"] = {"reset": "for each seed"}
    design_path.parent.mkdir(parents=True, exist_ok=True)
    design_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("fake-task-design completed")


def _validate_design(phase: str) -> None:
    design_path = _path("TASK_DESIGN_PATH")
    task_path = Path("task.md")
    design = load_task_design(design_path, canonical_task=task_path)
    required_repairs = int(os.environ.get("FAKE_TASK_DESIGN_FAILS", "0"))
    match = re.search(r"_(\d+)", phase)
    if match is None:
        raise RuntimeError(f"invalid design-review phase: {phase}")
    round_index = int(match.group(1))
    if (
        os.environ.get("FAKE_TASK_MUTATES_CHANGED_PATHS") == "1"
        and round_index > 0
        and Path("changed_paths.json").is_file()
    ):
        Path("changed_paths.json").write_text(
            '{"schema_version": 1, "paths": []}\n', encoding="utf-8"
        )
    if (
        os.environ.get("FAKE_TASK_REVIEW_MUTATES_PARENT") == "1"
        and Path("parent_review.json").is_file()
    ):
        Path("parent_review.json").write_text("{}\n", encoding="utf-8")
    should_fail = round_index < required_repairs
    scope_disagreement = (
        os.environ.get("FAKE_TASK_REPAIR_SCOPE_DISAGREES_ONCE") == "1" and round_index == 0
    )
    persistent_scope_disagreement = (
        os.environ.get("FAKE_TASK_REPAIR_SCOPE_DISAGREES_ALWAYS") == "1"
        or os.environ.get("FAKE_TASK_SCOPE_FALLBACK_REQUIRES_ROTATION") == "1"
    ) and round_index == 0
    if scope_disagreement:
        should_fail = "_c_" not in phase
    if persistent_scope_disagreement:
        should_fail = True
    rotation_pending = (
        os.environ.get("FAKE_TASK_SCOPE_FALLBACK_REQUIRES_ROTATION") == "1"
        and round_index > 0
        and design.payload["upstream"][0]["license"] != "Scope repaired."
    )
    if rotation_pending:
        should_fail = True
    regressed = (
        design.payload["method"]["claim"] == "Regressed design."
        and os.environ.get("FAKE_TASK_DELTA_ONLY_REGRESSION") != "1"
    )
    disagreement = os.environ.get("FAKE_TASK_DESIGN_REVIEW_DISAGREES") == "1" and phase.startswith(
        "design_validate_0_b_"
    )
    fallback_with_adjudication = os.environ.get("FAKE_TASK_FALLBACK_WITH_ADJUDICATION") == "1"
    cross_dimension_dissent = (
        os.environ.get("FAKE_TASK_CROSS_DIMENSION_DISSENT") == "1" or fallback_with_adjudication
    ) and round_index == 0
    focused_drift = os.environ.get("FAKE_TASK_FOCUSED_REVIEW_DRIFT") == "1" and round_index == 1
    full_gate_regression = (
        os.environ.get("FAKE_TASK_FULL_GATE_REGRESSES") == "1" and round_index == 2
    )
    full_gate_unchanged_failure = (
        os.environ.get("FAKE_TASK_FULL_GATE_UNCHANGED_FAILURE") == "1" and round_index == 2
    )
    full_gate_broad_latent_failure = (
        os.environ.get("FAKE_TASK_FULL_GATE_BROAD_LATENT_FAILURE") == "1" and round_index == 2
    )
    full_gate_overlapping_latent_failure = (
        os.environ.get("FAKE_TASK_FULL_GATE_OVERLAPPING_LATENT_FAILURE") == "1" and round_index == 2
    )
    full_gate_same_dimension_new_finding = (
        os.environ.get("FAKE_TASK_FULL_GATE_SAME_DIMENSION_NEW_FINDING") == "1" and round_index == 2
    )
    full_gate_worsens_existing_failure = os.environ.get(
        "FAKE_TASK_FULL_GATE_WORSENS_EXISTING_FAILURE"
    ) == "1" and round_index in {0, 2}
    full_gate_retained_and_new_regression = os.environ.get(
        "FAKE_TASK_FULL_GATE_RETAINED_AND_NEW_REGRESSION"
    ) == "1" and round_index in {0, 2}
    full_gate_latent_and_direct_regression = os.environ.get(
        "FAKE_TASK_FULL_GATE_LATENT_AND_DIRECT_REGRESSION"
    ) == "1" and round_index == 2
    partial_repair_failure = (
        os.environ.get("FAKE_TASK_DESIGN_TWO_FAILURES") == "1" and round_index <= 1
    )
    frontier_regression = (
        os.environ.get("FAKE_TASK_REPAIR_REQUIRES_FRONTIER") == "1"
        and round_index > 0
        and design.payload["planes"]["search"]["budget"]["training_steps"] > 1
        and design.payload["upstream"][0]["license"] != "Frontier repaired."
    )
    frontier_expansion = (
        os.environ.get("FAKE_TASK_FRONTIER_EXPANDS_FAILURES") == "1"
        and round_index > 0
        and design.payload["planes"]["search"]["budget"]["training_steps"] > 1
    )
    mixed_latent_indirect = (
        os.environ.get("FAKE_TASK_MIXED_LATENT_INDIRECT_REGRESSIONS") == "1"
        and round_index > 0
        and design.payload["planes"]["search"]["budget"]["training_steps"] > 1
        and design.payload["upstream"][0]["license"] != "Frontier repaired."
    )
    dimensions = []
    for identifier in DESIGN_DIMENSIONS:
        unrelated_drift = focused_drift and identifier == "reproducibility"
        failed = (
            (should_fail and identifier == "compute_realism")
            or ((regressed or disagreement) and identifier == "reproducibility")
            or (
                cross_dimension_dissent and identifier == "search_proxy_fidelity" and "_a_" in phase
            )
            or (
                cross_dimension_dissent
                and identifier == "search_final_isolation"
                and ("_b_" in phase or "_adjudicator_" in phase)
            )
            or unrelated_drift
            or (full_gate_regression and identifier == "reproducibility")
            or (full_gate_unchanged_failure and identifier == "reproducibility")
            or (full_gate_broad_latent_failure and identifier == "reproducibility")
            or (full_gate_overlapping_latent_failure and identifier == "reproducibility")
            or (
                full_gate_same_dimension_new_finding
                and identifier in {"compute_realism", "reproducibility"}
            )
            or (full_gate_worsens_existing_failure and identifier == "reproducibility")
            or (
                full_gate_retained_and_new_regression
                and (
                    identifier == "reproducibility"
                    or (round_index == 2 and identifier == "publication_scope")
                )
            )
            or (
                full_gate_latent_and_direct_regression
                and identifier in {"metrics_and_rubrics", "reproducibility"}
            )
            or (partial_repair_failure and identifier == "reproducibility")
            or (frontier_regression and identifier == "reproducibility")
            or (frontier_expansion and identifier in {"reproducibility", "publication_scope"})
            or (mixed_latent_indirect and identifier in {"reproducibility", "publication_scope"})
        )
        dimensions.append(
            {
                "id": identifier,
                "verdict": "fail" if failed else "pass",
                "evidence": [f"Fixture evidence for {identifier}."],
                "design_paths": (
                    ["/planes/search/budget/training_steps", "/upstream"]
                    if full_gate_broad_latent_failure and identifier == "reproducibility"
                    else ["/planes/search/budget/training_steps"]
                    if full_gate_overlapping_latent_failure and identifier == "reproducibility"
                    else ["/planes/search/budget/training_steps"]
                    if full_gate_retained_and_new_regression and round_index == 2
                    else ["/upstream"]
                    if full_gate_latent_and_direct_regression
                    and identifier == "metrics_and_rubrics"
                    else ["/planes/search/budget/training_steps"]
                    if full_gate_latent_and_direct_regression
                    else ["/planes/search/budget/training_steps"]
                    if (frontier_regression or frontier_expansion or mixed_latent_indirect)
                    and identifier in {"reproducibility", "publication_scope"}
                    else ["/upstream"]
                    if unrelated_drift or full_gate_unchanged_failure
                    else ["/method", "/planes/search/budget"]
                ),
                "task_evidence": ["task.md:1: fixture goal"],
                "repair_paths": (
                    ["/upstream"]
                    if unrelated_drift
                    else (
                        ["/candidate/interface"]
                        if fallback_with_adjudication and identifier == "search_proxy_fidelity"
                        else ["/upstream/0/license"]
                        if (frontier_regression or frontier_expansion or mixed_latent_indirect)
                        and identifier == "reproducibility"
                        else ["/upstream/0/license"]
                        if full_gate_same_dimension_new_finding and identifier == "compute_realism"
                        else ["/method/claim"]
                        if full_gate_same_dimension_new_finding
                        else ["/publication/required_evidence/0"]
                        if (frontier_expansion or mixed_latent_indirect)
                        and identifier == "publication_scope"
                        else ["/planes/search/budget/training_steps"]
                        if full_gate_overlapping_latent_failure
                        else ["/planes/search/budget/training_steps"]
                        if full_gate_retained_and_new_regression and round_index == 2
                        else ["/upstream/0/license"]
                        if full_gate_latent_and_direct_regression
                        and identifier == "metrics_and_rubrics"
                        else ["/planes/search/budget/training_steps"]
                        if full_gate_latent_and_direct_regression
                        else ["/planes/search/budget"]
                        if persistent_scope_disagreement and "_a_" in phase
                        else ["/upstream/0/license"]
                        if (persistent_scope_disagreement and "_b_" in phase)
                        or full_gate_unchanged_failure
                        or full_gate_broad_latent_failure
                        else ["/candidate/interface"]
                        if persistent_scope_disagreement
                        else ["/method/claim"]
                        if scope_disagreement and "_scope_" not in phase and "_b_" in phase
                        else ["/planes/search/budget"]
                        if scope_disagreement
                        else ["/planes/search/budget", "/method/claim"]
                    )
                    if failed
                    else []
                ),
                "remediation": "Increase the workload." if failed else "",
            }
        )
    for row in dimensions:
        if row["verdict"] == "fail":
            row["design_paths"] = list(
                dict.fromkeys([*row["design_paths"], *row["repair_paths"]])
            )
    failed_any = any(row["verdict"] == "fail" for row in dimensions)
    report: dict[str, Any] = {
        "schema_version": 3,
        "verdict": "fail" if failed_any else "pass",
        "summary": "Fixture design review.",
        "design_fingerprint": design.fingerprint,
        "dimensions": dimensions,
    }
    if (
        os.environ.get("FAKE_TASK_DESIGN_REVIEW_INVALID_ONCE") == "1"
        and phase == "design_validate_0_a_attempt_0"
    ):
        report["schema_version"] = 2
        for row in report["dimensions"]:
            row.pop("repair_paths")
    _path("TASK_DESIGN_REVIEW_PATH").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("fake-task-design-validation completed")


def _adjudicate_design() -> None:
    request = json.loads(Path("adjudication_request.json").read_text(encoding="utf-8"))
    fallback_with_adjudication = os.environ.get("FAKE_TASK_FALLBACK_WITH_ADJUDICATION") == "1"
    cross_dimension_dissent = (
        os.environ.get("FAKE_TASK_CROSS_DIMENSION_DISSENT") == "1" or fallback_with_adjudication
    )
    decisions = []
    for finding in request["minority_findings"]:
        source_dimension = finding["source_dimension"]
        uphold = cross_dimension_dissent and source_dimension == "search_proxy_fidelity"
        canonical_dimension = (
            "compute_realism"
            if uphold and fallback_with_adjudication
            else "search_final_isolation"
            if uphold
            else source_dimension
        )
        repair_paths = finding["source_repair_paths"] if uphold else []
        decisions.append(
            {
                "source_review_index": finding["source_review_index"],
                "source_dimension": source_dimension,
                "verdict": "uphold" if uphold else "reject",
                "canonical_dimension": canonical_dimension,
                "evidence": ["Fixture adjudication evidence."],
                "design_paths": [repair_paths[0] if repair_paths else "/method/claim"],
                "task_evidence": ["task.md:1: fixture goal"],
                "repair_paths": repair_paths,
                "remediation": "Repair the canonical blocker." if uphold else "",
            }
        )
    _path("TASK_DESIGN_REVIEW_PATH").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "design_fingerprint": request["design_fingerprint"],
                "request_fingerprint": request["request_fingerprint"],
                "source_review_fingerprints": request["source_review_fingerprints"],
                "decisions": decisions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("fake-task-design-adjudication completed")


def _repair_design_structure() -> None:
    path = _path("TASK_DESIGN_PATH")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        os.environ.get("FAKE_TASK_MISSING_DESIGN") == "1"
        or os.environ.get("FAKE_TASK_SYMLINK_DESIGN") == "1"
    ):
        prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
        if "task design must be a regular file" not in prompt:
            raise RuntimeError("missing-design diagnostic absent from repair prompt")
        payload = _task_design_payload(Path("task.md"))
    if os.environ.get("FAKE_TASK_INVALID_DESIGN") == "1":
        if isinstance(payload["candidate"]["state_semantics"], dict):
            prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
            expected = "candidate.state_semantics must be a non-empty string"
            if expected not in prompt:
                raise RuntimeError("structural design diagnostic missing from repair prompt")
            if os.environ.get("FAKE_TASK_INVALID_DESIGN_REPAIR_NOOP") != "1":
                payload["candidate"]["state_semantics"] = "Reset for each seed."
    payload["planes"]["search"]["budget"]["training_steps"] += 1
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("fake-task-design-repair completed")


def _repair_design_patch(phase: str) -> None:
    design_path = _path("TASK_DESIGN_PATH")
    design = load_task_design(design_path, canonical_task=Path("task.md"))
    match = re.search(r"design_repair_(\d+)", phase)
    if match is None:
        raise RuntimeError(f"invalid design-repair phase: {phase}")
    repair_round = int(match.group(1))
    if os.environ.get("FAKE_TASK_REQUIRE_FOCUSED_FEEDBACK") == "1" and repair_round >= 2:
        prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
        feedback = json.loads(prompt.split("```json\n", 1)[1].split("\n```", 1)[0])
        if any(
            row.get("id") == "reproducibility" for row in feedback["review"]["dimensions"]
        ) or any(
            row.get("id") == "reproducibility"
            for row in feedback.get("previous_candidate", {}).get("candidate_findings", [])
        ):
            raise RuntimeError("focused repair retained an unrelated candidate finding")
    if os.environ.get("FAKE_TASK_INVALID_PATCH_ONCE") == "1" and "_format_1" in phase:
        prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
        if "parent fingerprint does not match" not in prompt:
            raise RuntimeError("semantic patch diagnostic missing from next repair prompt")
    operations: list[dict[str, object]] = [
        {
            "op": "replace",
            "path": "/planes/search/budget/training_steps",
            "value": design.payload["planes"]["search"]["budget"]["training_steps"] + 1,
        }
    ]
    if os.environ.get("FAKE_TASK_NOOP_PATCH_ONCE") == "1" and phase == "design_repair_0":
        operations[0]["value"] = design.payload["planes"]["search"]["budget"]["training_steps"]
    if os.environ.get("FAKE_TASK_FULL_GATE_UNCHANGED_FAILURE") == "1" and repair_round > 0:
        operations = [
            {
                "op": "replace",
                "path": "/upstream/0/license",
                "value": "Repaired license.",
            }
        ]
    if os.environ.get("FAKE_TASK_FULL_GATE_BROAD_LATENT_FAILURE") == "1" and repair_round > 0:
        if design.payload["planes"]["search"]["budget"]["training_steps"] <= 1:
            raise RuntimeError("repair did not continue from the latent-finding candidate")
        operations = [
            {
                "op": "replace",
                "path": "/upstream/0/license",
                "value": "Repaired broad latent finding.",
            }
        ]
    if os.environ.get("FAKE_TASK_FULL_GATE_OVERLAPPING_LATENT_FAILURE") == "1" and repair_round > 0:
        if design.payload["planes"]["search"]["budget"]["training_steps"] <= 1:
            raise RuntimeError("repair did not continue from the overlapping latent candidate")
        operations = [
            {
                "op": "replace",
                "path": "/planes/search/budget/training_steps",
                "value": design.payload["planes"]["search"]["budget"]["training_steps"] + 1,
            }
        ]
    if (
        os.environ.get("FAKE_TASK_FULL_GATE_LATENT_AND_DIRECT_REGRESSION") == "1"
        and repair_round > 0
    ):
        prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
        feedback = json.loads(prompt.split("```json\n", 1)[1].split("\n```", 1)[0])
        failed_paths = {
            path
            for row in feedback["review"]["dimensions"]
            if row["verdict"] == "fail"
            for path in row["repair_paths"]
        }
        if "/upstream/0/license" not in failed_paths:
            raise RuntimeError("latent full-gate finding was not rebound to the safe parent")
        operations = [
            {
                "op": "replace",
                "path": "/planes/search/budget/training_steps",
                "value": design.payload["planes"]["search"]["budget"]["training_steps"] + 1,
            },
            {
                "op": "replace",
                "path": "/upstream/0/license",
                "value": "Latent finding repaired from safe parent.",
            },
        ]
    if os.environ.get("FAKE_TASK_FULL_GATE_SAME_DIMENSION_NEW_FINDING") == "1" and repair_round > 0:
        prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
        feedback = json.loads(prompt.split("```json\n", 1)[1].split("\n```", 1)[0])
        failed_paths = {
            path
            for row in feedback["review"]["dimensions"]
            if row["verdict"] == "fail"
            for path in row["repair_paths"]
        }
        if failed_paths != {"/method/claim", "/planes/search/budget"}:
            raise RuntimeError("repair did not retain the parent authorization")
        operations = [
            {
                "op": "replace",
                "path": "/planes/search/budget/training_steps",
                "value": 2,
            },
            {
                "op": "replace",
                "path": "/method/claim",
                "value": "Fresh full-gate regression avoided.",
            },
        ]
    if (
        os.environ.get("FAKE_TASK_REPAIR_REQUIRES_FRONTIER") == "1"
        or os.environ.get("FAKE_TASK_MIXED_LATENT_INDIRECT_REGRESSIONS") == "1"
    ) and repair_round > 0:
        if design.payload["planes"]["search"]["budget"]["training_steps"] <= 1:
            raise RuntimeError("repair did not continue from the quarantined frontier")
        operations = [
            {
                "op": "replace",
                "path": "/upstream/0/license",
                "value": "Frontier repaired.",
            }
        ]
    if os.environ.get("FAKE_TASK_SCOPE_FALLBACK_REQUIRES_ROTATION") == "1" and repair_round > 0:
        operations = [
            {
                "op": "replace",
                "path": "/upstream/0/license",
                "value": "Scope repaired.",
            }
        ]
    regression_rounds = int(os.environ.get("FAKE_TASK_DESIGN_REGRESSION_ROUNDS", "0"))
    regresses_once = repair_round == 0 and (
        os.environ.get("FAKE_TASK_DESIGN_REGRESSES_ONCE") == "1"
        or os.environ.get("FAKE_TASK_DELTA_ONLY_REGRESSION") == "1"
    )
    if regresses_once or repair_round < regression_rounds:
        operations.append({"op": "replace", "path": "/method/claim", "value": "Regressed design."})
    _path("TASK_DESIGN_PATCH_PATH").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parent_fingerprint": (
                    "0" * 64
                    if (
                        os.environ.get("FAKE_TASK_INVALID_PATCH_ALWAYS") == "1"
                        or (
                            os.environ.get("FAKE_TASK_INVALID_PATCH_ONCE") == "1"
                            and phase == "design_repair_0"
                        )
                    )
                    else design.fingerprint
                ),
                "operations": operations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("fake-task-design-patch completed")


def _validate_design_delta(phase: str) -> None:
    parent = load_task_design(Path("parent_task_design.json"), canonical_task=Path("task.md"))
    candidate = load_task_design(Path("task_design.json"), canonical_task=Path("task.md"))
    parent_review = json.loads(Path("parent_review.json").read_text(encoding="utf-8"))
    candidate_review = json.loads(Path("candidate_review.json").read_text(encoding="utf-8"))
    parent_failed = {row["id"] for row in parent_review["dimensions"] if row["verdict"] == "fail"}
    candidate_failed = {
        row["id"] for row in candidate_review["dimensions"] if row["verdict"] == "fail"
    }
    candidate_rows = {row["id"]: row for row in candidate_review["dimensions"]}
    invalid_once = os.environ.get("FAKE_TASK_DELTA_INVALID_ONCE") == "1"
    first_attempt = phase.endswith("_attempt_0")
    if invalid_once and not first_attempt:
        prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
        if "task design delta fixed dimensions do not match consensus progress" not in prompt:
            raise RuntimeError("delta validation diagnostic missing from retry prompt")
    if os.environ.get("FAKE_TASK_DELTA_MUTATES_INPUT") == "1":
        Path("parent_review.json").write_text("{}\n", encoding="utf-8")
    regressed = candidate.payload["method"]["claim"] == "Regressed design."
    worsened_existing_failure = (
        os.environ.get("FAKE_TASK_FULL_GATE_WORSENS_EXISTING_FAILURE") == "1"
        and candidate.payload["planes"]["search"]["budget"]["training_steps"]
        > parent.payload["planes"]["search"]["budget"]["training_steps"]
    )
    frontier_regression = (
        os.environ.get("FAKE_TASK_REPAIR_REQUIRES_FRONTIER") == "1"
        and candidate.payload["planes"]["search"]["budget"]["training_steps"] > 1
        and candidate.payload["upstream"][0]["license"] != "Frontier repaired."
    )
    frontier_expansion = (
        os.environ.get("FAKE_TASK_FRONTIER_EXPANDS_FAILURES") == "1"
        and candidate.payload["planes"]["search"]["budget"]["training_steps"] > 1
    )
    mixed_latent_indirect = (
        os.environ.get("FAKE_TASK_MIXED_LATENT_INDIRECT_REGRESSIONS") == "1"
        and candidate.payload["planes"]["search"]["budget"]["training_steps"] > 1
        and candidate.payload["upstream"][0]["license"] != "Frontier repaired."
    )
    fresh_full_gate_regression = (
        os.environ.get("FAKE_TASK_FULL_GATE_SAME_DIMENSION_NEW_FINDING") == "1"
        and "reproducibility" in candidate_failed
    )
    full_gate_regression = (
        os.environ.get("FAKE_TASK_FULL_GATE_REGRESSES") == "1"
        and "reproducibility" in candidate_failed
        and candidate.payload["planes"]["search"]["budget"]["training_steps"]
        > parent.payload["planes"]["search"]["budget"]["training_steps"]
    )
    retained_and_new_regression = (
        os.environ.get("FAKE_TASK_FULL_GATE_RETAINED_AND_NEW_REGRESSION") == "1"
        and {"reproducibility", "publication_scope"} <= candidate_failed
        and candidate.payload["planes"]["search"]["budget"]["training_steps"]
        > parent.payload["planes"]["search"]["budget"]["training_steps"]
    )
    latent_and_direct_regression = (
        os.environ.get("FAKE_TASK_FULL_GATE_LATENT_AND_DIRECT_REGRESSION") == "1"
        and {"metrics_and_rubrics", "reproducibility"} <= candidate_failed
        and candidate.payload["planes"]["search"]["budget"]["training_steps"]
        > parent.payload["planes"]["search"]["budget"]["training_steps"]
    )
    regressions = []
    if (
        regressed
        or worsened_existing_failure
        or frontier_regression
        or frontier_expansion
        or mixed_latent_indirect
        or fresh_full_gate_regression
        or full_gate_regression
        or retained_and_new_regression
        or latent_and_direct_regression
    ):
        regressions.append(
            {
                "dimension": "reproducibility",
                "design_paths": [
                    "/planes/search/budget/training_steps"
                    if fresh_full_gate_regression
                    or full_gate_regression
                    or retained_and_new_regression
                    or latent_and_direct_regression
                    or frontier_regression
                    or mixed_latent_indirect
                    or worsened_existing_failure
                    else "/method/claim"
                ],
                "repair_paths": candidate_rows["reproducibility"]["repair_paths"],
                "evidence": ["Fixture candidate contains a deliberate regression."],
                "remediation": "Restore the parent claim.",
            }
        )
    if frontier_expansion:
        regressions.append(
            {
                "dimension": "publication_scope",
                "design_paths": ["/planes/search/budget/training_steps"],
                "repair_paths": candidate_rows["publication_scope"]["repair_paths"],
                "evidence": ["Fixture candidate contains a second deliberate regression."],
                "remediation": "Repair the publication evidence.",
            }
        )
    if retained_and_new_regression:
        regressions.append(
            {
                "dimension": "publication_scope",
                "design_paths": ["/planes/search/budget/training_steps"],
                "repair_paths": candidate_rows["publication_scope"]["repair_paths"],
                "evidence": ["Fixture candidate contains a new direct regression."],
                "remediation": "Restore the parent publication behavior.",
            }
        )
    _path("TASK_DESIGN_REVIEW_PATH").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "verdict": "fail" if regressions else "pass",
                "summary": "Fixture delta review.",
                "parent_fingerprint": parent.fingerprint,
                "candidate_fingerprint": candidate.fingerprint,
                "fixed_dimensions": (
                    []
                    if invalid_once and first_attempt
                    else sorted(parent_failed - candidate_failed)
                ),
                "regressions": regressions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("fake-task-design-delta completed")


def _validate(phase: str) -> None:
    fingerprints = json.loads(os.environ["TASK_ALIGNMENT_FINGERPRINTS"])
    round_index = int(phase.split("_", 2)[1])
    if os.environ.get("FAKE_TASK_JUDGE_REQUIRE_ISOLATION") == "1":
        forbidden = [key for key in os.environ if key.startswith("DRAFT_")]
        if forbidden:
            raise RuntimeError(f"judge inherited draft environment: {forbidden}")
        task_source = Path(os.environ["TASK_SOURCE_DIR"])
        if (
            Path.cwd().name != f"round_{round_index}"
            or not task_source.is_dir()
            or task_source.parent == Path.cwd()
        ):
            raise RuntimeError("judge did not receive an isolated round workspace")
        prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
        if str(task_source) not in prompt:
            raise RuntimeError("judge prompt does not identify the curated task source")
        if Path(".codex").exists():
            raise RuntimeError("judge inherited the drafting agent configuration")
        Path("judge_isolation_verified").write_text("ok\n", encoding="utf-8")
    if os.environ.get("FAKE_TASK_JUDGE_REQUIRE_CURATED") == "1":
        task_source = Path(os.environ["TASK_SOURCE_DIR"])
        prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
        forbidden_paths = [
            task_source / ".venv",
            task_source / ".agentic_runtime.json",
        ]
        visible = [path.name for path in forbidden_paths if path.exists() or path.is_symlink()]
        if visible:
            raise RuntimeError(f"judge received sealed task artifacts: {visible}")
        if "agent_authored_scaffold" not in prompt:
            raise RuntimeError("judge prompt lacks harness attestation")
        if "passed in native test topology" not in prompt:
            raise RuntimeError("native judge prompt overstates smoke topology")
    if os.environ.get("FAKE_TASK_JUDGE_REQUIRE_LEGACY_FINAL") == "1":
        final_root = Path(os.environ["TASK_SOURCE_DIR"]) / "resources" / "final"
        if not final_root.is_dir():
            raise RuntimeError("legacy judge cannot inspect final resources")
    required_repairs = int(os.environ.get("FAKE_TASK_VALIDATION_FAILS", "0"))
    marker = Path(os.environ["TASK_SOURCE_DIR"]) / "semantic-repair.marker"
    completed_repairs = int(marker.read_text(encoding="utf-8")) if marker.is_file() else 0
    should_fail = completed_repairs < required_repairs
    failed_id = "search_final_isolation"
    dimensions = [
        {
            "id": identifier,
            "verdict": "fail" if should_fail and identifier == failed_id else "pass",
            "evidence": [
                {
                    "path": "task_contract.json",
                    "line": 1,
                    "detail": f"Fixture evidence for {identifier}",
                }
            ],
            "remediation": (
                "Separate the final fixture." if should_fail and identifier == failed_id else ""
            ),
        }
        for identifier in REQUIRED_ALIGNMENT_DIMENSIONS
    ]
    report = {
        "schema_version": 1,
        "verdict": "fail" if should_fail else "pass",
        "summary": "Fixture scaffold needs repair." if should_fail else "Fixture scaffold aligns.",
        "fingerprints": fingerprints,
        "dimensions": dimensions,
    }
    malformed_attempts = int(os.environ.get("FAKE_TASK_INVALID_ALIGNMENT_REPORTS", "0"))
    validation_attempt = int(phase.rsplit("_", 1)[1]) if "_attempt_" in phase else 0
    missing_attempts = int(os.environ.get("FAKE_TASK_MISSING_ALIGNMENT_REPORTS", "0"))
    if validation_attempt < missing_attempts:
        print("fake-task-validation omitted report")
        return
    if validation_attempt < malformed_attempts:
        report["dimensions"][-1]["evidence"][0]["line"] = 10_000
    path = Path(os.environ["TASK_ALIGNMENT_REPORT_PATH"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if os.environ.get("FAKE_TASK_JUDGE_PRECREATES_CANONICALIZATION_LOG") == "1":
        Path("canonicalization_1.log").symlink_to(
            os.environ["FAKE_TASK_RESERVED_TARGET"]
        )
    if os.environ.get("FAKE_TASK_JUDGE_MUTATES") == "1":
        (Path(os.environ["TASK_SOURCE_DIR"]) / "judge-mutation.txt").write_text(
            "unsafe\n", encoding="utf-8"
        )
    print("fake-task-validation completed")


def _repair_preflight(phase: str) -> None:
    prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
    diagnostic = _preflight_diagnostic(prompt)
    if os.environ.get("FAKE_TASK_PREFLIGHT_REPAIR_FAIL") == "1":
        raise RuntimeError("forced preflight repair failure")
    if os.environ.get("FAKE_TASK_PROVISIONAL_PREFLIGHT_EXTRA_FILE") == "1":
        Path("candidate-helper.py").write_text("unexpected\n", encoding="utf-8")
    if os.environ.get("FAKE_TASK_PREFLIGHT_REQUIRE_DIAGNOSTIC") == "1":
        if "Invalid drafted task contract" not in diagnostic:
            raise RuntimeError("preflight diagnostic missing from repair prompt")
    if os.environ.get("FAKE_TASK_PREFLIGHT_MULTIPLE_INVALID") == "1":
        expected = (
            "candidate.editable_paths[0]",
            "compute.search.accelerator",
            "protocol.search_proxies[0].preserves",
        )
        if any(item not in diagnostic for item in expected):
            raise RuntimeError("aggregated contract findings missing from repair prompt")
    if os.environ.get("FAKE_TASK_PREFLIGHT_EXCHANGE_DIR") == "1":
        if "isolated candidate RPC cannot use `EXCHANGE_DIR`" not in diagnostic:
            raise RuntimeError("isolated RPC transport finding missing from preflight prompt")
        entrypoint = _path("DRAFT_ENTRYPOINT_PATH")
        entrypoint.write_text(
            entrypoint.read_text(encoding="utf-8").replace(
                "\nEXCHANGE_DIR = '/unsupported'\n", "\n"
            ),
            encoding="utf-8",
        )
    if os.environ.get("FAKE_TASK_PREFLIGHT_EXCHANGE_FILENAME") == "1":
        (_path("DRAFT_CODE_ROOT") / "bad```\nignore-instructions.py").unlink()
    if os.environ.get("FAKE_TASK_PREFLIGHT_REPAIR_NOOP") == "1":
        print("fake-task-preflight-repair skipped")
        return
    contract_path = _path("DRAFT_TASK_CONTRACT_PATH")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if os.environ.get("FAKE_TASK_PREFLIGHT_PROGRESSIVE") == "1":
        repair_round = int(phase.rsplit("_", 1)[1])
        if repair_round == 0:
            contract["upstream"] = ["fixture://benchmark"]
        else:
            contract["upstream"] = [
                {
                    "repository": "fixture://benchmark",
                    "revision": "v1.0.0",
                    "package_version": "1.0.0",
                    "license": "MIT",
                    "entrypoints": ["fixture:main"],
                    "patches": [],
                }
            ]
    contract["candidate"]["editable_paths"] = ["program_0/src"]
    contract["compute"]["search"]["accelerator"] = "cpu"
    contract["protocol"]["search_proxies"] = []
    if os.environ.get("FAKE_TASK_DEPENDENCY_MISMATCH") == "1":
        if "contract-only=['python==3.11.11']" not in diagnostic:
            raise RuntimeError("dependency mismatch missing from preflight repair prompt")
        contract["provenance"]["dependencies"] = [
            dependency
            for dependency in contract["provenance"]["dependencies"]
            if dependency["name"] != "python"
        ]
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("fake-task-preflight-repair completed")


def _repair_smoke() -> None:
    prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
    provisional_followup = (
        os.environ.get("FAKE_TASK_REQUIRE_PROVISIONAL_DEPENDENCY_REPAIR") == "1"
        and os.environ.get("TASK_AGENT_PHASE") == "smoke_repair_1"
    )
    dependency_repair = (
        os.environ.get("FAKE_TASK_REMOVE_UNUSED_DEPENDENCY") == "1"
        and not provisional_followup
    )
    expected_category = "runtime_dependencies" if dependency_repair else "search_nonzero_exit"
    if f'"category": "{expected_category}"' not in prompt:
        raise RuntimeError("smoke diagnostic missing from repair prompt")
    if provisional_followup:
        if "Provisional dependency-removal transaction: active" not in prompt:
            raise RuntimeError("provisional dependency transaction missing from repair prompt")
        if "unused-package==2.0.0" in Path("requirements.txt").read_text(encoding="utf-8"):
            raise RuntimeError("provisional dependency repair restarted from original scaffold")
    remove_on_search = os.environ.get("FAKE_TASK_REMOVE_DEPENDENCY_ON_SEARCH_FAILURE") == "1"
    if dependency_repair or remove_on_search:
        requirements = Path("requirements.txt")
        if "unused-package==2.0.0" not in requirements.read_text(encoding="utf-8"):
            raise RuntimeError("dependency repair did not start from the rollback frontier")
        retained = [
            line
            for line in requirements.read_text(encoding="utf-8").splitlines()
            if not line.startswith("unused-package==")
        ]
        requirements.write_text("\n".join(retained) + "\n", encoding="utf-8")
        print("fake-task-smoke-dependency-repair completed")
        return
    injection = os.environ.get("FAKE_TASK_SMOKE_INJECTION", "")
    if os.environ.get("FAKE_TASK_SMOKE_REQUIRE_TYPED") == "1" and injection in prompt:
        raise RuntimeError("untrusted smoke output leaked into repair prompt")
    if os.environ.get("FAKE_TASK_REPAIR_USE_GATE") == "1":
        before = request_gate(
            Path(os.environ["TASK_GATE_SOCKET"]),
            token=os.environ["TASK_GATE_TOKEN"],
            operation="search-smoke",
        )
        if before.status != "authored_failure":
            raise RuntimeError(f"repair gate did not reproduce failure: {before.status}")
    evaluator_path = _path("DRAFT_EVALUATE_PATH")
    evaluator = evaluator_path.read_text(encoding="utf-8")
    evaluator_path.write_text(
        evaluator.replace("SMOKE_REPAIR_READY = False", "SMOKE_REPAIR_READY = True"),
        encoding="utf-8",
    )
    if provisional_followup and os.environ.get("FAKE_TASK_FAIL_PROVISIONAL_FOLLOWUP") == "1":
        raise RuntimeError("provisional follow-up repair failed")
    if os.environ.get("FAKE_TASK_REPAIR_USE_GATE") == "1":
        after = request_gate(
            Path(os.environ["TASK_GATE_SOCKET"]),
            token=os.environ["TASK_GATE_TOKEN"],
            operation="search-smoke",
        )
        if after.status != "pass":
            raise RuntimeError(f"repair gate did not accept fix: {after.status}")
    if os.environ.get("FAKE_TASK_SMOKE_OUT_OF_SCOPE") == "1":
        Path("task.md").write_text("# weakened by repair\n", encoding="utf-8")
    if os.environ.get("FAKE_TASK_SMOKE_RUNTIME_TAMPER") == "1":
        runtime_python = Path(".venv/bin/python")
        runtime_python.parent.mkdir(parents=True)
        runtime_python.write_text("tampered runtime\n", encoding="utf-8")
    if os.environ.get("FAKE_TASK_SMOKE_WRITES_BYTECODE") == "1":
        py_compile.compile(str(evaluator_path), doraise=True)
    print("fake-task-smoke-repair completed")


def main() -> int:
    phase = os.environ.get("TASK_AGENT_PHASE", "draft")
    if capture_path := os.environ.get("FAKE_TASK_PROGRESS_ENV_CAPTURE"):
        with Path(capture_path).open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        key: os.environ.get(key)
                        for key in (
                            "AGENTIC_TASK_PROGRESS_TRANSPORT",
                            "TASK_PROGRESS_AUTH_TOKEN",
                            "TASK_PROGRESS_COST_BASELINE",
                            "TASK_PROGRESS_SESSION_ID",
                        )
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    if phase == "design":
        _design()
    elif phase.startswith("design_validate_") and "_adjudicator_" in phase:
        _adjudicate_design()
    elif phase.startswith("design_validate_"):
        _validate_design(phase)
    elif phase.startswith("design_delta_"):
        _validate_design_delta(phase)
    elif phase.startswith("design_compile_repair_"):
        _repair_design_structure()
    elif phase.startswith("design_repair_"):
        _repair_design_patch(phase)
    elif phase == "draft":
        _draft()
    elif phase.startswith("validate_"):
        _validate(phase)
    elif phase.startswith("preflight_repair_"):
        _repair_preflight(phase)
    elif phase.startswith("smoke_repair_"):
        _repair_smoke()
    elif phase.startswith("repair_"):
        if os.environ.get("FAKE_TASK_REPAIR_REQUIRE_DIAGNOSTIC") == "1":
            prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
            if "search_final_isolation" not in prompt or "Separate the final fixture" not in prompt:
                raise RuntimeError("alignment diagnostic missing from repair prompt")
        if os.environ.get("FAKE_TASK_ALIGNMENT_BREAKS_ENTRYPOINT_ONCE") == "1":
            entrypoint = _path("DRAFT_ENTRYPOINT_PATH")
            if phase == "repair_0":
                entrypoint.unlink()
            else:
                prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
                if "structurally invalid canonical candidate boundary" not in prompt:
                    raise RuntimeError("canonical entrypoint feedback missing from repair prompt")
                entrypoint.parent.mkdir(parents=True, exist_ok=True)
                entrypoint.write_text(_program(), encoding="utf-8")
        if (
            os.environ.get("FAKE_TASK_ALIGNMENT_SYMLINKS_ENTRYPOINT") == "1"
            and phase == "repair_0"
        ):
            entrypoint = _path("DRAFT_ENTRYPOINT_PATH")
            entrypoint.unlink()
            entrypoint.symlink_to(os.environ["FAKE_TASK_RESERVED_TARGET"])
        if os.environ.get("FAKE_TASK_ALIGNMENT_DELETES_CAPTURE_ONCE") == "1":
            capture = _path("DRAFT_CAPTURE_PATH")
            if phase == "repair_0":
                capture.unlink()
            else:
                prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
                encoded = prompt.split("Validation diagnostic (base64): ", 1)[1].splitlines()[
                    0
                ]
                diagnostic = base64.b64decode(encoded).decode("utf-8")
                if (
                    "Unsafe drafted artifact:" not in diagnostic
                    or "capture.py" not in diagnostic
                ):
                    raise RuntimeError("capture canonicalization diagnostic missing from retry prompt")
        if marker_path := os.environ.get("FAKE_TASK_ALIGNMENT_BREAKS_FINAL_SMOKE_ONCE"):
            attempt_marker = Path(marker_path)
            final_evaluator = _path("DRAFT_FINAL_EVALUATE_PATH")
            if not attempt_marker.exists():
                attempt_marker.write_text("failed once\n", encoding="utf-8")
                final_evaluator.write_text(
                    _evaluator().replace(
                        "def main() -> int:\n",
                        "def main() -> int:\n    raise RuntimeError('alignment regression')\n",
                    ),
                    encoding="utf-8",
                )
            else:
                prompt = Path(os.environ["AGENT_PROMPT_FILE"]).read_text(encoding="utf-8")
                encoded = prompt.split("Validation diagnostic (base64): ", 1)[1].splitlines()[0]
                if "final_nonzero_exit" not in base64.b64decode(encoded).decode("utf-8"):
                    raise RuntimeError("alignment smoke diagnostic missing from retry prompt")
                final_evaluator.write_text(_evaluator(), encoding="utf-8")
        marker = Path("semantic-repair.marker")
        completed_repairs = int(marker.read_text(encoding="utf-8")) if marker.is_file() else 0
        marker.write_text(f"{completed_repairs + 1}\n", encoding="utf-8")
        print("fake-task-repair completed")
    else:
        raise RuntimeError(f"unknown task agent phase: {phase}")
    if os.environ.get("FAKE_TASK_AGENT_LARGE_LOG") == "1":
        print("agent-stream-start")
        sys.stdout.write("x" * 8192)
        print("\nagent-stream-tail")
    if os.environ.get("FAKE_TASK_PATCH_VALID_EXIT_NONZERO") == "1" and phase == "design_repair_0":
        return 9
    if os.environ.get("FAKE_TASK_REVIEW_VALID_EXIT_NONZERO") == "1" and phase.startswith(
        "design_validate_"
    ):
        return 9
    if (
        os.environ.get("FAKE_TASK_STRUCTURE_VALID_EXIT_NONZERO") == "1"
        and phase == "design_compile_repair_0"
    ):
        return 9
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
