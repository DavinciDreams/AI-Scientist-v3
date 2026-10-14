from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from src.investigation.task_contract import load_task_contract
from tests.e2e.test_agent_execution_pipeline import (
    _script_path,
    _write_fake_task_draft_agent,
    _write_task_md_only,
)


def _run_robust_draft(
    root: Path, *, design_max_repairs: str | None = None, **environment: str
) -> subprocess.CompletedProcess[str]:
    task_dir = _write_task_md_only(root)
    output_dir = root / "draft_out"
    bin_dir = root / "bin"
    bin_dir.mkdir()
    _write_fake_task_draft_agent(bin_dir, "headless")
    env = os.environ | {
        "HEADLESS_BIN": str(bin_dir / "headless"),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WRAPPER_PYTHON": sys.executable,
        "TASK_ALLOW_NATIVE_DRAFT_FOR_TESTS": "1",
        "TASK_ALLOW_NATIVE_REPAIR_FOR_TESTS": "1",
        **environment,
    }
    return subprocess.run(
        [
            "bash",
            str(_script_path("task.sh")),
            "--agent",
            "codex",
            "--task-dir",
            str(task_dir),
            "--output-dir",
            str(output_dir),
            "--workflow",
            "robust",
            *(["--design-max-repairs", design_max_repairs] if design_max_repairs else []),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def test_design_prompt_rendering_does_not_import_pipeline_or_pass_payload_in_argv() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        private_root = root / "private"
        private_root.mkdir()
        poison_root = root / "poison"
        poison_root.mkdir()
        (poison_root / "hydra.py").write_text(
            'raise RuntimeError("pipeline import attempted")\n', encoding="utf-8"
        )
        payload_path = root / "feedback.json"
        payload_path.write_text(json.dumps({"diagnostic": "x" * 200_000}), encoding="utf-8")
        script = r"""
source "$1/src/scripts/lib/task_design.sh"
PYTHON_BIN="$2"
REPO_ROOT="$1"
DESIGN_PRIVATE_ROOT="$3"
payload="$(<"$4")"
render_design_prompt \
  "$1/src/prompts/repair_task_design.md" \
  "./task.md" "./task_design.json" "" "fixture-fingerprint" \
  "$payload" "" "" "" "./task_design_patch.json"
"""

        result = subprocess.run(
            [
                "bash",
                "-c",
                script,
                "bash",
                str(Path(__file__).parents[2]),
                sys.executable,
                str(private_root),
                str(payload_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ | {"PYTHONPATH": str(poison_root)},
        )

        assert result.returncode == 0, result.stderr
        assert "pipeline import attempted" not in result.stderr
        assert "x" * 1000 in result.stdout
        assert len(result.stdout) > 200_000


def test_design_prompt_renders_campaign_capacity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        private_root = root / "private"
        private_root.mkdir()
        template = root / "template.md"
        template.write_text(
            "cycles={{MAX_IDEA_CYCLES}} evals={{MAX_PROGRAM_EVALS_PER_ATTEMPT}}\n",
            encoding="utf-8",
        )
        script = r"""
source "$1/src/scripts/lib/task_design.sh"
PYTHON_BIN="$2"
REPO_ROOT="$1"
DESIGN_PRIVATE_ROOT="$3"
MAX_IDEA_CYCLES=7
MAX_PROGRAM_EVALS_PER_ATTEMPT=2
render_design_prompt "$4" "./task.md" "./task_design.json"
"""

        result = subprocess.run(
            [
                "bash",
                "-c",
                script,
                "bash",
                str(Path(__file__).parents[2]),
                sys.executable,
                str(private_root),
                str(template),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout == "cycles=7 evals=2\n\n"


def test_retry_feedback_compacts_previous_candidate_findings() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        review_path = root / "review.json"
        progress_path = root / "progress.json"
        review_path.write_text(
            json.dumps(
                {
                    "verdict": "fail",
                    "dimensions": [
                        {"id": "compute_realism", "verdict": "fail", "evidence": ["current"]},
                        {"id": "reproducibility", "verdict": "pass", "evidence": ["current"]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        progress_path.write_text(
            json.dumps(
                {
                    "fixed": ["compute_realism"],
                    "still_failed": ["reproducibility"],
                    "regressed": ["publication_scope"],
                    "improved": False,
                    "diagnostic": "candidate rejected",
                    "candidate_findings": [
                        {
                            "id": "reproducibility",
                            "verdict": "fail",
                            "design_paths": ["/upstream"],
                            "evidence": ["e" * 10_000],
                            "remediation": "r" * 10_000,
                            "untrusted_extra": "must not propagate",
                        }
                    ],
                    "delta_regressions": [
                        {
                            "dimension": "compute_realism",
                            "design_paths": ["/planes/search/budget"],
                            "evidence": ["delta evidence"],
                            "remediation": "restore realistic compute",
                        }
                    ],
                    "internal_detail": "must not propagate",
                }
            ),
            encoding="utf-8",
        )
        script = r"""
source "$1/src/scripts/lib/task_design.sh"
PYTHON_BIN="$2"
design_review_failure_payload "$3" 2 "$4"
"""

        result = subprocess.run(
            [
                "bash",
                "-c",
                script,
                "bash",
                str(Path(__file__).parents[2]),
                sys.executable,
                str(review_path),
                str(progress_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        previous = payload["previous_candidate"]
        assert set(previous) == {
            "fixed",
            "still_failed",
            "regressed",
            "improved",
            "diagnostic",
            "candidate_findings",
            "delta_regressions",
        }
        finding = previous["candidate_findings"][0]
        assert set(finding) == {"id", "verdict", "design_paths", "evidence", "remediation"}
        assert len(finding["evidence"][0]) < 1_000
        assert len(finding["remediation"]) < 2_000
        regression = previous["delta_regressions"][0]
        assert set(regression) == {"dimension", "design_paths", "evidence", "remediation"}
        assert regression["dimension"] == "compute_realism"
        assert payload["protected_dimensions"] == [
            {
                "id": "reproducibility",
                "design_paths": [],
                "evidence": ["current"],
            }
        ]


def test_repair_memory_retains_distinct_causes_on_the_same_path() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        memory_path = root / "repair_memory.json"
        progress_paths = []
        for index, remediation in enumerate(
            ("Preserve fixed batch size.", "Preserve the rollout cadence.")
        ):
            progress_path = root / f"progress_{index}.json"
            progress_path.write_text(
                json.dumps(
                    {
                        "fixed": [],
                        "still_failed": ["reproducibility"],
                        "regressed": ["compute_realism"],
                        "improved": False,
                        "candidate_findings": [
                            {
                                "id": "compute_realism",
                                "design_paths": ["/planes/search/resources/2"],
                                "evidence": [f"Distinct cause {index}."],
                                "remediation": remediation,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            progress_paths.append(progress_path)
        script = r"""
source "$1/src/scripts/lib/task_design.sh"
PYTHON_BIN="$2"
update_design_repair_memory "$3" "$4"
update_design_repair_memory "$3" "$5"
"""

        result = subprocess.run(
            [
                "bash",
                "-c",
                script,
                "bash",
                str(Path(__file__).parents[2]),
                sys.executable,
                str(memory_path),
                *(str(path) for path in progress_paths),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        assert [row["remediation"] for row in memory["constraints"]] == [
            "Preserve fixed batch size.",
            "Preserve the rollout cadence.",
        ]


def test_changed_path_manifest_generation_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output_path = root / "changed_paths.json"
        script = r"""
source "$1/src/scripts/lib/task_design.sh"
PYTHON_BIN=/bin/false
REPO_ROOT="$1"
TASK_SOURCE_DIR="$2"
DESIGN_PRIVATE_ROOT="$2"
write_design_change_manifest "$3" "$4" "$5"
"""

        result = subprocess.run(
            [
                "bash",
                "-c",
                script,
                "bash",
                str(Path(__file__).parents[2]),
                str(root),
                str(root / "parent.json"),
                str(root / "candidate.json"),
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0
        assert "Could not derive task design changed paths" in result.stderr
        assert not output_path.exists()


def test_design_is_repaired_and_accepted_before_scaffold_generation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        progress_environment = root / "agent-progress-environment.jsonl"

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_PROGRESS_ENV_CAPTURE=str(progress_environment),
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "independent task design passed in round 2" in output
        assert "[TASK] Generate task design · completed" in output
        assert "[TASK] Design review round 0 · reviewer 1/1 · FAIL" in output
        assert "[TASK] Design decision round 0 · FAIL" in output
        assert "Failed: compute_realism" in output
        assert "[TASK] Semantic design repair 1/8 · completed" in output
        assert "Fixed: compute_realism" in output
        assert "[TASK] Scaffold generation · completed" in output
        assert "[TASK] Contract validation · PASS" in output
        assert "[TASK] Evaluator smoke tests · PASS" in output
        assert "[TASK] Task publication · PASS" in output
        assert "Tokens · task drafting ·" in output
        captured_environments = [
            json.loads(line) for line in progress_environment.read_text().splitlines()
        ]
        assert captured_environments
        assert all(
            record == {
                "AGENTIC_TASK_PROGRESS_TRANSPORT": None,
                "TASK_PROGRESS_AUTH_TOKEN": None,
                "TASK_PROGRESS_COST_BASELINE": None,
                "TASK_PROGRESS_SESSION_ID": None,
            }
            for record in captured_environments
        )
        design_root = root / "draft_out/_task_design"
        assert (design_root / "task_design.json").is_file()
        assert (design_root / "accepted_review.json").is_file()
        assert (design_root / "repair_0/task_design.json").is_file()
        assert (design_root / "review_0_a/task_design_review.json").is_file()
        assert not (design_root / "review_0_b").exists()
        assert not (design_root / "review_0_c").exists()
        assert not (design_root / "review_0_adjudicator").exists()
        assert json.loads((design_root / "consensus_0.json").read_text()) == json.loads(
            (design_root / "review_0_a/task_design_review.json").read_text()
        )
        assert (design_root / "review_1_a/task_design_review.json").is_file()
        assert not (design_root / "review_1_b").exists()
        assert not (design_root / "review_1_c").exists()
        assert (design_root / "review_1_a/parent_design.json").is_file()
        changed_paths = json.loads(
            (design_root / "review_1_a/changed_paths.json").read_text(encoding="utf-8")
        )
        assert changed_paths["schema_version"] == 1
        assert changed_paths["paths"]
        focused_prompt = (design_root / "review_1_a/validate_prompt.md").read_text(encoding="utf-8")
        assert "./parent_design.json" in focused_prompt
        assert "./changed_paths.json" in focused_prompt
        assert (design_root / "full_review_2.json").is_file()
        assert (design_root / "candidate_ledger.jsonl").is_file()
        task_dir = root / "task"
        assert (task_dir / "evaluate.py").is_file()
        assert not (task_dir / "task_design.json").exists()


def test_review_prompt_prioritizes_runnable_blockers_and_schema_valid_repairs() -> None:
    prompt = (
        Path(__file__).parents[2] / "src/prompts/validate_task_design.md"
    ).read_text(encoding="utf-8")

    assert "candidate.interface" in prompt
    assert "python_rpc_v1:handle" in prompt
    assert "candidate.state_semantics" in prompt
    assert "non-negative integers" in prompt
    assert "not itself a leakage failure" in prompt
    assert "Do not require an extrapolation formula" in prompt
    assert "block scaffold generation or make evaluation invalid" in " ".join(prompt.split())


def test_focused_review_rejects_parent_review_mutation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_REVIEW_MUTATES_PARENT="1",
            design_max_repairs="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "Task design judge modified the parent review" in output


def test_incomplete_scaffold_is_retried_with_missing_artifact_feedback() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(root, FAKE_TASK_DRAFT_OMITS_CAPTURE_ONCE="1")

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "Missing required drafted artifact: capture.py" in output
        assert "Unsafe drafted artifact" not in output
        assert (root / "task/capture.py").is_file()
        assert len(list((root / "draft_out").glob("draft_agent_*.log"))) == 2


def test_focused_reviewer_cannot_mutate_changed_path_manifest() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_MUTATES_CHANGED_PATHS="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "modified the changed-path manifest" in output


def test_noop_design_patch_is_retried_before_review() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_NOOP_PATCH_ONCE="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "task design patch must change the design" in output
        assert "design_repair_0_format_1" in output


def test_incomplete_scaffold_reports_all_missing_artifacts_in_one_retry() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(root, FAKE_TASK_DRAFT_OMITS_MULTIPLE_ONCE="1")

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "Missing required drafted artifact: capture.py" in output
        assert "Missing required drafted artifact: evaluate_final.py" in output
        retry_prompt = (root / "draft_out/draft_retry_2_prompt.md").read_text()
        assert "capture.py" in retry_prompt
        assert "evaluate_final.py" in retry_prompt
        assert len(list((root / "draft_out").glob("draft_agent_*.log"))) == 2


def test_unsafe_program_directory_is_fatal_without_retry() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(root, FAKE_TASK_UNSAFE_PROGRAM_DIR="1")

        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "Unsafe drafted program directory: program_0" in output
        assert "Missing required drafted artifact" not in output
        assert len(list((root / "draft_out").glob("draft_agent_*.log"))) == 1


def test_dependency_mismatch_is_repaired_during_preflight() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(root, FAKE_TASK_DEPENDENCY_MISMATCH="1")

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "contract-only=['python==3.11.11']" in output
        assert list((root / "draft_out").glob("preflight_repair_0_agent_codex_*.log"))
        contract = json.loads((root / "task/task_contract.json").read_text())
        assert contract["provenance"]["dependencies"] == [
            {"name": "example-package", "version": "1.2.3"}
        ]


def test_publication_preserves_arbitrary_root_provenance_artifact() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(root, FAKE_TASK_ROOT_PROVENANCE_ARTIFACT="1")

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        artifact = root / "task/dependency_manifest.json"
        assert artifact.read_text(encoding="utf-8") == (
            '{"environment": "fixture", "schema_version": 1}\n'
        )
        contract = load_task_contract(root / "task/task_contract.json")
        assert any(item.path == artifact.name for item in contract.provenance.artifacts)
        assert not (root / "draft_out/task/.venv").exists()
        assert not (root / "draft_out/task/.agentic_runtime.json").exists()


def test_failed_design_stops_before_scaffold_agent() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(root, FAKE_TASK_DESIGN_FAILS="9")

        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "Task design still failed after 8 semantic repair attempt(s)" in output
        assert not (root / "task/evaluate.py").exists()


def test_structurally_invalid_design_is_repaired_with_validator_feedback() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(root, FAKE_TASK_INVALID_DESIGN="1")

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "candidate.state_semantics must be a non-empty string" in output
        assert "JSONDecodeError" not in output
        assert (root / "task/evaluate.py").is_file()


def test_valid_structural_repair_survives_nonzero_transport_exit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_INVALID_DESIGN="1",
            FAKE_TASK_STRUCTURE_VALID_EXIT_NONZERO="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "using valid retained design" in output


def test_structural_repairs_do_not_consume_semantic_repair_budget() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_INVALID_DESIGN="1",
            FAKE_TASK_DESIGN_FAILS="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        design_root = root / "draft_out/_task_design"
        assert (design_root / "structure_repair_0/task_design.json").is_file()
        assert (design_root / "repair_0/task_design.json").is_file()
    assert "independent task design passed in round 2" in output


def test_regressing_design_candidate_is_rejected_and_repaired_from_best() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_DESIGN_REGRESSES_ONCE="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "candidate rejected" in output
        design_root = root / "draft_out/_task_design"
        ledger = [
            __import__("json").loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        assert any(row.get("regressed") == ["reproducibility"] for row in ledger)
        accepted = __import__("json").loads(
            (design_root / "task_design.json").read_text(encoding="utf-8")
        )
        assert accepted["method"]["claim"] != "Regressed design."


def test_indirect_regression_continues_from_quarantined_repair_frontier() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_REPAIR_REQUIRES_FRONTIER="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        design_root = root / "draft_out/_task_design"
        accepted = json.loads((design_root / "task_design.json").read_text())
        assert accepted["planes"]["search"]["budget"]["training_steps"] == 2
        assert accepted["upstream"][0]["license"] == "Frontier repaired."
        ledger = [
            json.loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        assert any(row["event"] == "frontier_advanced" for row in ledger)


def test_mixed_latent_and_indirect_regressions_advance_quarantined_frontier() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_MIXED_LATENT_INDIRECT_REGRESSIONS="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        design_root = root / "draft_out/_task_design"
        accepted = json.loads((design_root / "task_design.json").read_text())
        assert accepted["planes"]["search"]["budget"]["training_steps"] == 2
        assert accepted["upstream"][0]["license"] == "Frontier repaired."
        ledger = [
            json.loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        frontier = next(row for row in ledger if row["event"] == "frontier_advanced")
        assert frontier["consensus_regressed"] == ["reproducibility", "publication_scope"]
        assert frontier["regressed"] == ["reproducibility"]


def test_repair_budget_exhaustion_keeps_partial_frontier_quarantined() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_REPAIR_REQUIRES_FRONTIER="1",
            design_max_repairs="1",
        )

        assert result.returncode != 0
        design_root = root / "draft_out/_task_design"
        accepted = json.loads((design_root / "task_design.json").read_text())
        frontier = json.loads((design_root / "repair_frontier.json").read_text())
        assert accepted["planes"]["search"]["budget"]["training_steps"] == 1
        assert accepted["upstream"][0]["license"] != "Frontier repaired."
        assert frontier["planes"]["search"]["budget"]["training_steps"] == 2
        assert frontier["upstream"][0]["license"] != "Frontier repaired."


def test_partial_frontier_cannot_expand_the_failure_count() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FRONTIER_EXPANDS_FAILURES="1",
            design_max_repairs="1",
        )

        assert result.returncode != 0
        design_root = root / "draft_out/_task_design"
        accepted = json.loads((design_root / "task_design.json").read_text())
        assert accepted["planes"]["search"]["budget"]["training_steps"] == 1
        ledger = [
            json.loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        assert not any(row["event"] == "frontier_advanced" for row in ledger)


def test_focused_repair_omits_unrelated_previous_candidate_findings() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_DESIGN_REGRESSION_ROUNDS="3",
            FAKE_TASK_REQUIRE_FOCUSED_FEEDBACK="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert output.count("candidate rejected") == 3
        focused_prompt = (root / "draft_out/_task_design/repair_2/repair_prompt.md").read_text(
            encoding="utf-8"
        )
        assert '"mode": "focused"' in focused_prompt
        feedback = json.loads(focused_prompt.split("```json\n", 1)[1].split("\n```", 1)[0])
        assert all(row["id"] != "reproducibility" for row in feedback["review"]["dimensions"])
        assert any(
            row["dimension"] == "reproducibility"
            for row in feedback["repair_memory"]["constraints"]
        )


def test_focused_review_ignores_unrelated_candidate_review_drift() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FOCUSED_REVIEW_DRIFT="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        design_root = root / "draft_out/_task_design"
        raw = json.loads((design_root / "review_1_a/task_design_review.json").read_text())
        consensus = json.loads((design_root / "consensus_1.json").read_text())
        raw_verdicts = {row["id"]: row["verdict"] for row in raw["dimensions"]}
        consensus_verdicts = {row["id"]: row["verdict"] for row in consensus["dimensions"]}
        assert raw_verdicts["reproducibility"] == "fail"
        assert consensus_verdicts["reproducibility"] == "fail"
        assert (design_root / "full_review_2.json").is_file()


def test_failed_full_gate_does_not_replace_last_known_best_design() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FULL_GATE_REGRESSES="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "candidate rejected in round 2" in output
        design_root = root / "draft_out/_task_design"
        accepted = json.loads((design_root / "task_design.json").read_text())
        assert accepted["planes"]["search"]["budget"]["training_steps"] == 2
        ledger = [
            json.loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        rejection = next(row for row in ledger if row["event"] == "rejected")
        assert rejection["regressed"] == ["reproducibility"]


def test_failed_full_gate_advances_nonregressing_repair_frontier() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FULL_GATE_UNCHANGED_FAILURE="1",
            design_max_repairs="2",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        design_root = root / "draft_out/_task_design"
        repair_prompt = (design_root / "review_3_a/validate_prompt.md").read_text()
        assert 'Target dimension IDs: `["reproducibility"]`' in repair_prompt
        repaired = json.loads((design_root / "task_design.json").read_text())
        assert repaired["planes"]["search"]["budget"]["training_steps"] == 2
        assert repaired["upstream"][0]["license"] == "Repaired license."
        ledger = [
            json.loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        assert any(row["event"] == "full_review" for row in ledger)
        assert not any(row["event"] == "rejected" for row in ledger)
        assert sum(row["event"] == "promoted" for row in ledger) == 2


def test_failed_full_gate_can_reclassify_broad_evidence_as_latent() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FULL_GATE_BROAD_LATENT_FAILURE="1",
            design_max_repairs="2",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        design_root = root / "draft_out/_task_design"
        repaired = json.loads((design_root / "task_design.json").read_text())
        assert repaired["planes"]["search"]["budget"]["training_steps"] == 2
        assert repaired["upstream"][0]["license"] == "Repaired broad latent finding."
        ledger = [
            json.loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        assert any(row["event"] == "full_review" for row in ledger)
        assert not any(row["event"] == "rejected" for row in ledger)


def test_latent_full_gate_finding_never_publishes_without_repair_budget() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FULL_GATE_BROAD_LATENT_FAILURE="1",
            design_max_repairs="1",
        )

        assert result.returncode != 0
        design_root = root / "draft_out/_task_design"
        canonical = json.loads((design_root / "task_design.json").read_text())
        frontier = json.loads((design_root / "repair_frontier.json").read_text())
        assert canonical["planes"]["search"]["budget"]["training_steps"] == 1
        assert frontier["planes"]["search"]["budget"]["training_steps"] == 2
        assert not (design_root / "accepted_review.json").exists()


def test_delta_audit_can_reclassify_overlapping_full_gate_finding() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FULL_GATE_OVERLAPPING_LATENT_FAILURE="1",
            design_max_repairs="2",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        design_root = root / "draft_out/_task_design"
        accepted = json.loads((design_root / "task_design.json").read_text())
        assert accepted["planes"]["search"]["budget"]["training_steps"] == 3
        ledger = [
            json.loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        assert not any(row["event"] == "rejected" for row in ledger)
        promoted = next(row for row in ledger if row["event"] == "promoted")
        assert promoted["consensus_regressed"] == ["reproducibility"]
        assert promoted["regressed"] == []
        assert promoted["improved"] is True


def test_overlapping_latent_finding_never_publishes_without_repair_budget() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FULL_GATE_OVERLAPPING_LATENT_FAILURE="1",
            design_max_repairs="1",
        )

        assert result.returncode != 0
        design_root = root / "draft_out/_task_design"
        canonical = json.loads((design_root / "task_design.json").read_text())
        frontier = json.loads((design_root / "repair_frontier.json").read_text())
        assert canonical["planes"]["search"]["budget"]["training_steps"] == 1
        assert frontier["planes"]["search"]["budget"]["training_steps"] == 2
        assert not (design_root / "accepted_review.json").exists()


def test_new_direct_full_gate_finding_repairs_from_retained_parent() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FULL_GATE_SAME_DIMENSION_NEW_FINDING="1",
            design_max_repairs="2",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "candidate rejected in round 2" in output
        design_root = root / "draft_out/_task_design"
        second_parent = json.loads((design_root / "repair_1/task_design.json").read_text())
        assert second_parent["planes"]["search"]["budget"]["training_steps"] == 1
        accepted = json.loads((design_root / "task_design.json").read_text())
        assert accepted["method"]["claim"] == "Fresh full-gate regression avoided."
        ledger = [
            json.loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        assert any(row["event"] == "rejected" for row in ledger)


def test_fresh_full_gate_candidate_remains_unpublished_without_repair_budget() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FULL_GATE_SAME_DIMENSION_NEW_FINDING="1",
            design_max_repairs="1",
        )

        assert result.returncode != 0
        design_root = root / "draft_out/_task_design"
        accepted = json.loads((design_root / "task_design.json").read_text())
        assert accepted["planes"]["search"]["budget"]["training_steps"] == 1
        assert not (design_root / "repair_frontier.json").exists()
        ledger = [
            json.loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        assert any(row["event"] == "rejected" for row in ledger)
        assert not (design_root / "accepted_review.json").exists()


def test_failed_full_gate_audits_worsened_existing_failure() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FULL_GATE_WORSENS_EXISTING_FAILURE="1",
            design_max_repairs="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode != 0, output
        design_root = root / "draft_out/_task_design"
        ledger = [
            json.loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        assert any(row["event"] == "full_review" for row in ledger)
        assert any(row["event"] == "delta_review" for row in ledger)
        assert any(row["event"] == "frontier_continued" for row in ledger)
        assert not any(row["event"] == "promoted" for row in ledger)


def test_retained_failure_cannot_mask_new_direct_regression() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FULL_GATE_RETAINED_AND_NEW_REGRESSION="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "candidate rejected in round 2" in output
        design_root = root / "draft_out/_task_design"
        second_parent = json.loads((design_root / "repair_1/task_design.json").read_text())
        assert second_parent["planes"]["search"]["budget"]["training_steps"] == 1
        ledger = [
            json.loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        rejection = next(row for row in ledger if row["event"] == "rejected")
        assert "publication_scope" in rejection["regressed"]


def test_rejected_candidate_rebinds_unchanged_full_gate_failure_to_safe_parent() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_FULL_GATE_LATENT_AND_DIRECT_REGRESSION="1",
            design_max_repairs="3",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "candidate rejected in round 2" in output
        design_root = root / "draft_out/_task_design"
        repaired = json.loads((design_root / "task_design.json").read_text())
        assert repaired["upstream"][0]["license"] == "Latent finding repaired from safe parent."
        repair_prompt = (design_root / "repair_1/repair_prompt.md").read_text()
        feedback = json.loads(repair_prompt.split("```json\n", 1)[1].split("\n```", 1)[0])
        failed = {
            row["id"]: row["repair_paths"]
            for row in feedback["review"]["dimensions"]
            if row["verdict"] == "fail"
        }
        assert failed["compute_realism"]
        assert failed["metrics_and_rubrics"] == ["/upstream/0/license"]
        assert any(
            row["dimension"] == "reproducibility"
            for row in feedback["previous_candidate"]["delta_regressions"]
        )


def test_delta_only_regression_is_checked_by_full_review() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_DELTA_ONLY_REGRESSION="1",
            design_max_repairs="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "all parent failures cleared; running full design review" in output
        design_root = root / "draft_out/_task_design"
        ledger = [
            __import__("json").loads(line)
            for line in (design_root / "candidate_ledger.jsonl").read_text().splitlines()
        ]
        assert not (design_root / "delta_1").exists()
        assert not any(row["event"] == "delta_review" for row in ledger)
        assert (design_root / "full_review_2.json").is_file()
        assert any(row["event"] == "promoted" for row in ledger)
        assert not any(row["event"] == "rejected" for row in ledger)
        assert not (design_root / "repair_1").exists()


def test_invalid_delta_artifact_is_retried_with_failure_feedback() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_DESIGN_TWO_FAILURES="1",
            FAKE_TASK_DELTA_INVALID_ONCE="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "task design delta fixed dimensions do not match consensus progress" in output
        assert "design_delta_1_attempt_1" in output


def test_delta_input_mutation_rejects_candidate_without_overwriting_best() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_DESIGN_TWO_FAILURES="1",
            FAKE_TASK_DELTA_MUTATES_INPUT="1",
            design_max_repairs="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "modified the parent review" in output
        best = __import__("json").loads(
            (root / "draft_out/_task_design/task_design.json").read_text(encoding="utf-8")
        )
        assert best["planes"]["search"]["budget"]["training_steps"] == 1


def test_missing_design_artifact_is_recreated_by_repair_agent() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(root, FAKE_TASK_MISSING_DESIGN="1")

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "task design must be a regular file" in output
        assert "JSONDecodeError" not in output
        assert (root / "task/evaluate.py").is_file()


def test_symlinked_design_is_repaired_without_overwriting_link_target() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        task_dir = root / "task"
        original_task = b"# Goal\n- optimize objective\n\n# Rubrics\n- score threshold\n"

        result = _run_robust_draft(root, FAKE_TASK_SYMLINK_DESIGN="1")

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert (task_dir / "task.md").read_bytes() == original_task
        design_path = root / "draft_out/_task_design/task_design.json"
        assert design_path.is_file()
        assert not design_path.is_symlink()


def test_design_agent_cannot_preseed_run_local_control_artifacts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(root, FAKE_TASK_DESIGN_EXTRA_ENTRY="1")

        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "Unexpected initial task-design entries" in output
        assert "optimize objective" in (root / "task/task.md").read_text(encoding="utf-8")


def test_invalid_review_contract_is_retried_with_failure_feedback() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(root, FAKE_TASK_DESIGN_REVIEW_INVALID_ONCE="1")

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "schema_version must be 3" in output
        assert "design_validate_0_a_attempt_1" in output


def test_valid_review_survives_nonzero_agent_transport_exit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(root, FAKE_TASK_REVIEW_VALID_EXIT_NONZERO="1")

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "validating retained review artifact" in output


def test_invalid_semantic_patch_is_retried_without_spending_semantic_budget() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_INVALID_PATCH_ONCE="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        design_root = root / "draft_out/_task_design"
        assert "parent fingerprint does not match" in output
        assert (design_root / "repair_0_format_1/task_design_patch.json").is_file()
        semantic_repairs = [
            row
            for row in map(
                json.loads, (design_root / "candidate_ledger.jsonl").read_text().splitlines()
            )
            if row["event"] in {"promoted", "rejected"}
        ]
        assert len(semantic_repairs) == 1


def test_format_retry_exhaustion_is_not_reported_as_semantic_exhaustion() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_INVALID_PATCH_ALWAYS="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "remained invalid after 3 format attempt(s)" in output
        assert "semantic repair attempt(s)" not in output


def test_valid_patch_survives_nonzero_agent_transport_exit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_DESIGN_FAILS="1",
            FAKE_TASK_PATCH_VALID_EXIT_NONZERO="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "validating retained patch artifact" in output


def test_structural_repair_exhaustion_preserves_validator_diagnostic() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        result = _run_robust_draft(
            root,
            FAKE_TASK_INVALID_DESIGN="1",
            FAKE_TASK_INVALID_DESIGN_REPAIR_NOOP="1",
        )

        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "candidate.state_semantics must be a non-empty string" in output
        assert (
            "Task design remained structurally invalid after 3 structural repair attempt(s)"
            in output
        )
        assert "JSONDecodeError" not in output
        assert not (root / "task/evaluate.py").exists()
