from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from src.investigation.evaluation_evidence import load_evaluation_evidence, evidence_metadata


def evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "episodes": [
            {
                "seed": 7,
                "score": 2.5,
                "correct": True,
                "duration_seconds": 0.1,
                "solution": [[0.5, 0.5, 0.25]],
                "candidate_diagnostics": {"objective_calls": 12},
                "candidate_timeseries": [{"series_name": "loss", "step": 0, "value": 0.5}],
            }
        ],
    }


def write_evidence(root: Path, payload: Any) -> Path:
    path = root / "evaluation_evidence.json"
    path.write_text(json.dumps(payload))
    return path


INVALID_EPISODE_CASES = [
    ("seed", True),
    ("score", float("nan")),
    ("correct", 1),
    ("duration_seconds", -1),
    ("duration_seconds", float("inf")),
    ("solution", {"bad": float("nan")}),
    ("candidate_diagnostics", {"foo": "not numeric"}),
    ("candidate_timeseries", [{"series_name": "../bad", "step": 0, "value": 1}]),
    ("candidate_timeseries", [{"series_name": "loss", "step": 0.5, "value": 1}]),
    ("path", "/etc/passwd"),
]


def test_evidence_roundtrip_and_invalid_payloads_rejected(tmp_path: Path) -> None:
    # roundtrip and missing status
    root = tmp_path / "roundtrip"
    root.mkdir()
    assert load_evaluation_evidence(root) is None
    assert evidence_metadata(root) == {"status": "unavailable"}
    write_evidence(root, evidence())
    assert load_evaluation_evidence(root) == evidence()
    metadata = evidence_metadata(root)
    assert metadata["status"] == "available"
    assert metadata["episode_count"] == 1
    assert str(metadata["sha256"]).startswith("sha256:")

    # invalid episode rejected
    root = tmp_path / "invalid"
    root.mkdir()
    for field, value in INVALID_EPISODE_CASES:
        payload = evidence()
        payload["episodes"][0][field] = value
        write_evidence(root, payload)
        with pytest.raises(ValueError):
            load_evaluation_evidence(root)

    # duplicate seeds rejected
    root = tmp_path / "duplicate"
    root.mkdir()
    payload = evidence()
    payload["episodes"] *= 2
    write_evidence(root, payload)
    with pytest.raises(ValueError, match="seed"):
        load_evaluation_evidence(root)

    # nested evidence limit
    root = tmp_path / "nested"
    root.mkdir()
    payload = evidence()
    nested: Any = 1
    for _ in range(20):
        nested = [nested]
    payload["episodes"][0]["solution"] = nested
    write_evidence(root, payload)
    with pytest.raises(ValueError, match="depth"):
        load_evaluation_evidence(root)

    # evidence symlink rejected
    root = tmp_path / "symlink"
    root.mkdir()
    target = root / "other.json"
    target.write_text(json.dumps(evidence()))
    (root / "evaluation_evidence.json").symlink_to(target)
    with pytest.raises(ValueError):
        load_evaluation_evidence(root)

    # evidence size limit
    root = tmp_path / "size"
    root.mkdir()
    path = root / "evaluation_evidence.json"
    with path.open("wb") as stream:
        stream.truncate(16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="limit"):
        load_evaluation_evidence(root)


def test_evidence_survives_search_promotion_with_trusted_ids(tmp_path: Path) -> None:
    from src.variants_pipeline.sandbox_outputs import (
        canonicalize_results,
        validate_promotable_results,
        build_promotable_frame,
        extract_promotable_frame,
    )
    from tests.variants.test_variants_sandbox_outputs import ENVIRONMENT

    source = tmp_path / "source"
    source.mkdir()
    (source / "metrics.json").write_text('{"combined_score": 2.5}')
    (source / "correct.json").write_text('{"correct": true}')
    write_evidence(source, evidence())
    canonicalize_results(source, "program_1", ENVIRONMENT)
    validate_promotable_results(source, ENVIRONMENT, "program_1")
    target = tmp_path / "target"
    target.mkdir()
    extract_promotable_frame(build_promotable_frame(source), target)
    assert load_evaluation_evidence(target) == evidence()
    with (target / "analysis/timeseries.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["evaluation_id"] == "eval-1"
    assert rows[0]["series_name"] == "candidate/seed_7/loss"
    canonicalize_results(target, "program_1", ENVIRONMENT)
    with (target / "analysis/timeseries.csv").open() as stream:
        assert list(csv.DictReader(stream)) == rows


def test_final_metrics_and_manifest_seal_evidence_and_host_usage(tmp_path: Path) -> None:
    import subprocess
    import sys
    from tests.final_evaluation.test_final_evaluation import (
        _write_contract_task,
        _write_investigation,
        run_final_evaluation,
    )
    from src.final_evaluation.runner import _validate_metrics, verify_final_manifest, FinalEvaluationError

    # final metrics allow evidence but reject extra files
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "metrics.json").write_text('{"score": 2.5}')
    (metrics_dir / "correct.json").write_text('{"correct": true}')
    write_evidence(metrics_dir, evidence())
    assert _validate_metrics(metrics_dir, ("score",)) == ({"score": 2.5}, True)
    (metrics_dir / "arbitrary.json").write_text("{}")
    with pytest.raises(FinalEvaluationError, match="undeclared"):
        _validate_metrics(metrics_dir, ("score",))

    # final manifest seals evidence and host usage
    task = _write_contract_task(tmp_path)
    investigation, _ = _write_investigation(tmp_path)
    output = tmp_path / "final"

    def evaluate(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(argv, **kwargs)
        write_evidence(output / "results", evidence())
        return result

    result = run_final_evaluation(
        task_dir=task,
        investigation_results_dir=investigation,
        output_dir=output,
        python_bin=Path(sys.executable),
        run=evaluate,
    )
    assert result["evaluation_evidence"]["episode_count"] == 1
    assert result["compute_usage"]["rpc_calls"] is None
    assert result["compute_usage"]["elapsed_seconds"] >= 0
    assert verify_final_manifest(output)["evaluation_evidence"] == result["evaluation_evidence"]
    manifest_path = output / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.chmod(0o600)
    for field, value in [
        ("outcome", "timeout"),
        ("rpc_calls", 999999999),
        ("max_rpc_calls", 999999999),
    ]:
        damaged = json.loads(json.dumps(manifest))
        damaged["compute_usage"][field] = value
        manifest_path.write_text(json.dumps(damaged))
        with pytest.raises(FinalEvaluationError, match="compute|RPC"):
            verify_final_manifest(output, task_dir=task)
    manifest_path.write_text(json.dumps(manifest))
    path = output / "results/evaluation_evidence.json"
    path.chmod(0o600)
    path.write_text("{}")
    with pytest.raises(FinalEvaluationError, match="digest"):
        verify_final_manifest(output)
