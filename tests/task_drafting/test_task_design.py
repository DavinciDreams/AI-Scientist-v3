"""Task-design loading, review, repair, aggregation, delta audit, latent rebinding and shell focus."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from src.task_drafting.design import (
    DESIGN_DIMENSIONS,
    TaskDesign,
    TaskDesignAdjudicationRequired,
    TaskDesignDeltaReview,
    TaskDesignError,
    TaskDesignReview,
    TaskDesignReviewProgress,
    aggregate_task_design_reviews,
    apply_task_design_patch,
    compare_task_design_reviews,
    confirm_task_design_delta,
    load_task_design,
    load_task_design_adjudication,
    load_task_design_delta_review,
    load_task_design_review,
    task_design_adjudication_request,
    task_design_scope_alternatives,
)
from src.task_drafting.design_cli import main as design_cli_main
from src.task_drafting.latent_review import rebind_latent_failures
from tests.fake_task_draft_agent import _task_design_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DESIGN_SH = REPO_ROOT / "src/scripts/lib/task_design.sh"
BUDGET = "/planes/search/budget"
UPHOLD_ISOLATION: dict[str, Any] = dict(
    source_review_index=0,
    source_dimension="search_proxy_fidelity",
    verdict="uphold",
    canonical_dimension="search_final_isolation",
)


# --------------------------------------------------------------------------- payload builders


def _design(task_path: Path) -> dict[str, Any]:
    budget = {
        "unit": "outer updates",
        "training_steps": 160,
        "wall_clock_seconds": 3600,
        "hardware": "4x GPU",
        "precision": "bfloat16",
    }
    plane = {
        "resources": ["pinned upstream benchmark"],
        "seeds": [1],
        "budget": budget,
        "baselines": ["published baseline"],
        "metrics": ["held-out loss"],
        "correctness": ["finite metrics", "completed training"],
        "smoke": "one update through the same candidate and metric path",
    }
    return {
        "schema_version": 1,
        "canonical_task": {
            "path": "task.md",
            "sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        },
        "method": {"class": "learned optimization", "claim": "Improve held-out loss."},
        "upstream": [
            {
                "repository": "https://example.test/upstream.git",
                "revision": "0123456789abcdef",
                "license": "Apache-2.0",
                "entrypoints": ["train.py"],
            }
        ],
        "candidate": {
            "interface": "python_rpc_v1:handle",
            "editable_paths": ["program_0/src"],
            "state_semantics": "Reset all candidate state for every declared seed.",
        },
        "planes": {
            "search": plane,
            "final": {**plane, "seeds": [2], "resources": ["sealed held-out benchmark"]},
        },
        "feedback": {
            "search_visible": ["search metrics"],
            "final_sealed": ["final identities", "final metrics"],
        },
        "publication": {"required_evidence": ["sealed final evaluation"]},
    }


def _review(fingerprint: str, failed: str | None = None) -> dict[str, Any]:
    """Schema-1 review failing at most one dimension."""
    dimensions = [
        {
            "id": identifier,
            "verdict": "fail" if identifier == failed else "pass",
            "evidence": ["Concrete task/design evidence."],
            "remediation": "Fix the design." if identifier == failed else "",
        }
        for identifier in DESIGN_DIMENSIONS
    ]
    return {
        "schema_version": 1,
        "verdict": "fail" if failed else "pass",
        "summary": "Reviewed design.",
        "design_fingerprint": fingerprint,
        "dimensions": dimensions,
    }


def _grounded_review(
    fingerprint: str,
    failed: str | None = None,
    *,
    path: str = BUDGET,
    design_path: str | None = None,
) -> dict[str, Any]:
    """Schema-3 review whose failed row repairs ``path`` with evidence at ``design_path``."""
    payload = _review(fingerprint, failed=failed)
    payload["schema_version"] = 3
    for row in payload["dimensions"]:
        row["design_paths"] = list(dict.fromkeys([design_path or path, path]))
        row["task_evidence"] = ["task.md:1: Task."]
        row["repair_paths"] = [path] if row["verdict"] == "fail" else []
    return payload


def _row(payload: dict[str, Any], dimension: str) -> dict[str, Any]:
    return next(row for row in payload["dimensions"] if row["id"] == dimension)


def _set_review_failure(payload: dict[str, Any], dimension: str, path: str) -> None:
    _row(payload, dimension).update(
        {
            "verdict": "fail",
            "remediation": f"Repair {dimension}.",
            "design_paths": [path],
            "repair_paths": [path],
        }
    )
    payload["verdict"] = "fail"


def _adjudication_decision(
    *,
    source_review_index: int,
    source_dimension: str,
    verdict: str,
    canonical_dimension: str,
    path: str = BUDGET,
) -> dict[str, Any]:
    upheld = verdict == "uphold"
    return {
        "source_review_index": source_review_index,
        "source_dimension": source_dimension,
        "verdict": verdict,
        "canonical_dimension": canonical_dimension,
        "evidence": ["Independent adjudication evidence."],
        "design_paths": [path],
        "task_evidence": ["task.md:1: Task."],
        "repair_paths": [path] if upheld else [],
        "remediation": "Repair the upheld blocker." if upheld else "",
    }


def _regression(
    dimension: str, path: str, repair_paths: list[str] | None = None, *, legacy: bool = False
) -> dict[str, Any]:
    row = {
        "dimension": dimension,
        "design_paths": [path],
        "repair_paths": [path] if repair_paths is None else repair_paths,
        "evidence": [f"The candidate regressed {dimension}."],
        "remediation": f"Repair {dimension}.",
    }
    if legacy:
        del row["repair_paths"]
    return row


# --------------------------------------------------------------------------- loaders / fixtures


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def task_path(tmp_path: Path) -> Path:
    path = tmp_path / "task.md"
    path.write_text("# Goal\nTask.\n", encoding="utf-8")
    return path


def _load_design(
    tmp_path: Path,
    task_path: Path,
    payload: dict[str, Any] | None = None,
    name: str = "task_design.json",
) -> TaskDesign:
    design_path = _write_json(tmp_path / name, payload or _design(task_path))
    return load_task_design(design_path, canonical_task=task_path)


@pytest.fixture
def design(tmp_path: Path, task_path: Path) -> TaskDesign:
    return _load_design(tmp_path, task_path)


def _change_claim(payload: dict[str, Any]) -> None:
    payload["method"]["claim"] = "Changed claim."


def _candidate_design(
    tmp_path: Path, task_path: Path, mutate: Callable[[dict[str, Any]], None] = _change_claim
) -> TaskDesign:
    payload = _design(task_path)
    mutate(payload)
    return _load_design(tmp_path, task_path, payload, "candidate_design.json")


def _load_review(
    tmp_path: Path,
    design: TaskDesign,
    payload: dict[str, Any],
    *,
    name: str = "review.json",
    schema: int | None = None,
) -> TaskDesignReview:
    return load_task_design_review(
        _write_json(tmp_path / name, payload), design=design, require_schema_version=schema
    )


def _load_reviews(
    tmp_path: Path, design: TaskDesign, payloads: list[dict[str, Any]]
) -> list[TaskDesignReview]:
    return [
        _load_review(tmp_path, design, payload, name=f"review_{index}.json", schema=3)
        for index, payload in enumerate(payloads)
    ]


def _grounded_reviews(
    tmp_path: Path, design: TaskDesign, cases: list[tuple[str | None, str]]
) -> list[TaskDesignReview]:
    return _load_reviews(
        tmp_path,
        design,
        [_grounded_review(design.fingerprint, failed, path=path) for failed, path in cases],
    )


def _repair_review(tmp_path: Path, design: TaskDesign, path: str) -> TaskDesignReview:
    payload = _grounded_review(design.fingerprint, "compute_realism", path=path)
    return _load_review(tmp_path, design, payload, name="repair_review.json", schema=3)


def _patch(tmp_path: Path, parent: TaskDesign, path: str, value: object, **extra: Any) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "parent_fingerprint": parent.fingerprint,
        "operations": [{"op": "replace", "path": path, "value": value}],
    }
    payload.update(extra)
    return _write_json(tmp_path / "task_design_patch.json", payload)


def _write_adjudication(
    tmp_path: Path, reviews: list[TaskDesignReview], decisions: list[dict[str, Any]]
) -> Path:
    request = task_design_adjudication_request(reviews)
    payload = {
        "schema_version": 1,
        "design_fingerprint": request["design_fingerprint"],
        "request_fingerprint": request["request_fingerprint"],
        "source_review_fingerprints": request["source_review_fingerprints"],
        "decisions": decisions,
    }
    return _write_json(tmp_path / "adjudication.json", payload)


def _adjudicate(tmp_path, design, reviews, decisions):
    path = _write_adjudication(tmp_path, reviews, decisions)
    return load_task_design_adjudication(path, design=design, reviews=reviews)


def _delta(
    tmp_path: Path,
    parent: TaskDesign,
    candidate: TaskDesign,
    *,
    verdict: str,
    fixed: list[str],
    regressions: list[dict[str, Any]],
    schema: int = 2,
) -> TaskDesignDeltaReview:
    payload = {
        "schema_version": schema,
        "verdict": verdict,
        "summary": "Delta audit.",
        "parent_fingerprint": parent.fingerprint,
        "candidate_fingerprint": candidate.fingerprint,
        "fixed_dimensions": fixed,
        "regressions": regressions,
    }
    path = _write_json(tmp_path / "delta.json", payload)
    return load_task_design_delta_review(path, parent=parent, candidate=candidate)


def _resolve(payload: Any, pointer: str) -> Any:
    for token in pointer.strip("/").split("/"):
        payload = payload[int(token)] if isinstance(payload, list) else payload[token]
    return payload


# --------------------------------------------------------------------------- load_task_design


def test_load_task_design_accepts_complete_and_upstream_free_designs(tmp_path: Path) -> None:
    # accepts a complete task-general design
    task_path = tmp_path / "task.md"
    task_path.write_text("# Goal\nTrain the requested method.\n", encoding="utf-8")

    design = _load_design(tmp_path, task_path)

    assert design.payload["method"]["class"] == "learned optimization"
    assert len(design.fingerprint) == 64

    # allows no upstream when the task names none
    task_dir = tmp_path / "self_contained"
    task_dir.mkdir()
    task_path = task_dir / "task.md"
    task_path.write_text("# Goal\nSolve a self-contained mathematical task.\n", encoding="utf-8")
    payload = _design(task_path)
    payload["upstream"] = []

    assert _load_design(task_dir, task_path, payload).payload["upstream"] == []


def _mutate_task_after_hash(payload: dict[str, Any], task_path: Path) -> None:
    task_path.write_text("# Goal\nChanged.\n", encoding="utf-8")


def _overlap_seeds(payload: dict[str, Any], task_path: Path) -> None:
    payload["planes"]["final"]["seeds"] = [1]


def _custom_interface(payload: dict[str, Any], task_path: Path) -> None:
    payload["candidate"]["interface"] = "custom:run"


def _editable(path: str) -> Callable[[dict[str, Any], Path], None]:
    def mutate(payload: dict[str, Any], task_path: Path) -> None:
        payload["candidate"]["editable_paths"] = [path]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(_mutate_task_after_hash, "canonical task hash", id="task-hash-mismatch"),
        pytest.param(_overlap_seeds, "seeds must be disjoint", id="overlapping-seeds"),
        pytest.param(_custom_interface, "python_rpc_v1:handle", id="noncanonical-interface"),
        *[
            pytest.param(_editable(path), "inside program_0", id=f"editable:{path}")
            for path in (
                "candidate/solver.py",
                "program_0/../candidate/solver.py",
                "/program_0/src",
                "program_0//src",
                "program_0/src/file:name.py",
                "program_0/src/*.py",
                "program_0/src/",
                "program_0",
                "program_0/program_spec.toml",
            )
        ],
    ],
)
def test_load_task_design_rejects_invalid_designs(
    tmp_path: Path, task_path: Path, mutate: Callable[..., None], message: str
) -> None:
    payload = _design(task_path)
    mutate(payload, task_path)

    with pytest.raises(TaskDesignError, match=message):
        _load_design(tmp_path, task_path, payload)


# --------------------------------------------------------------------------- load_task_design_review


def test_load_task_design_review_binds_verdict_and_accepts_grounded_schemas(
    tmp_path: Path, design: TaskDesign
) -> None:
    # binds the verdict to the design
    review = _load_review(tmp_path, design, _review(design.fingerprint, failed="compute_realism"))
    assert review.verdict == "fail"
    assert review.failed_dimensions == ("compute_realism",)

    # accepts grounded v2 findings
    payload = _review(design.fingerprint, failed="compute_realism")
    payload["schema_version"] = 2
    for row in payload["dimensions"]:
        row["design_paths"] = [BUDGET]
        row["task_evidence"] = ["task.md:1: Task."]
    assert _load_review(tmp_path, design, payload, name="v2.json").failed_dimensions == (
        "compute_realism",
    )

    # accepts v3 repair scope
    payload = _grounded_review(
        design.fingerprint, "compute_realism", path=BUDGET, design_path="/planes/search/metrics/0"
    )
    review = _load_review(tmp_path, design, payload, name="v3.json", schema=3)
    assert _row(review.payload, "compute_realism")["repair_paths"] == [BUDGET]


def test_load_task_design_review_rejects_stale_fingerprint_and_ungrounded_schema(
    tmp_path: Path, design: TaskDesign
) -> None:
    # rejects a stale fingerprint
    with pytest.raises(TaskDesignError, match="fingerprint"):
        _load_review(tmp_path, design, _review("0" * 64))

    # the robust review gate can require the grounded v2 schema
    with pytest.raises(TaskDesignError, match="schema_version must be 2"):
        _load_review(tmp_path, design, _review(design.fingerprint), name="v1.json", schema=2)


@pytest.mark.parametrize(
    ("failed", "overrides", "message"),
    [
        ("compute_realism", {"repair_paths": []}, "failed dimension compute_realism requires repair_paths"),
        (None, {"repair_paths": ["/method/claim"]}, "passed dimension compute_realism must not have repair_paths"),
        ("compute_realism", {"repair_paths": ["/canonical_task/sha256"]}, "repair path is not writable"),
        ("compute_realism", {"repair_paths": ["/candidate/interface"]}, "repair path is immutable"),
        ("compute_realism", {"repair_paths": ["/candidate"]}, "repair path is immutable"),
        ("compute_realism", {"repair_paths": ["/missing"]}, "repair path does not exist"),
        (
            "compute_realism",
            {"design_paths": ["/method/claim"], "repair_paths": [BUDGET]},
            "repair path lacks matching design evidence",
        ),
    ],
)
def test_load_task_design_review_rejects_invalid_v3_repair_scope(
    tmp_path: Path, design: TaskDesign, failed: str | None, overrides: dict[str, Any], message: str
) -> None:
    payload = _grounded_review(design.fingerprint, failed)
    _row(payload, "compute_realism").update(overrides)

    with pytest.raises(TaskDesignError, match=message):
        _load_review(tmp_path, design, payload, schema=3)


# --------------------------------------------------------------------------- apply_task_design_patch


@pytest.mark.parametrize(
    ("pointer", "value"),
    [
        ("/planes/search/budget/wall_clock_seconds", 7200),
        pytest.param("/upstream/0/revision", "fedcba9876543210", id="bounded-array-index"),
    ],
)
def test_apply_task_design_patch_changes_only_declared_paths(
    tmp_path: Path, task_path: Path, design: TaskDesign, pointer: str, value: object
) -> None:
    candidate = apply_task_design_patch(
        _patch(tmp_path, design, pointer, value),
        parent=design,
        canonical_task=task_path,
        repair_review=_repair_review(tmp_path, design, pointer),
    )

    assert _resolve(candidate.payload, pointer) == value
    assert candidate.payload["planes"]["final"] == design.payload["planes"]["final"]
    assert candidate.payload["method"] == design.payload["method"]


def test_task_design_patch_rejects_immutable_ancestor_unauthorized_scope_and_unsafe_operations(
    tmp_path: Path, task_path: Path, design: TaskDesign
) -> None:
    # cannot replace an immutable interface ancestor
    payload = _grounded_review(design.fingerprint, "candidate_editability", path="/candidate")
    repair_review = TaskDesignReview("fail", ("candidate_editability",), payload)
    replacement = {**design.payload["candidate"], "state_semantics": "Changed reset semantics."}
    with pytest.raises(TaskDesignError, match="patch path is immutable"):
        apply_task_design_patch(
            _patch(tmp_path, design, "/candidate", replacement),
            parent=design,
            canonical_task=task_path,
            repair_review=repair_review,
        )

    # requires failed repair authority covering the patched path
    for name, failed, message in [
        ("outside-repair-paths", "compute_realism", "outside failed repair paths"),
        ("passing-repair-review", None, "repair review must fail"),
    ]:
        payload = _grounded_review(design.fingerprint, failed, path=BUDGET, design_path="/method/claim")
        review = _load_review(tmp_path, design, payload, name=f"{name}.json", schema=3)
        with pytest.raises(TaskDesignError, match=message):
            apply_task_design_patch(
                _patch(tmp_path, design, "/method/claim", "Unrelated."),
                parent=design,
                canonical_task=task_path,
                repair_review=review,
            )

    # rejects unsafe operations
    repair_review = _repair_review(tmp_path, design, "/method")
    unsafe_mutations: list[tuple[dict[str, Any], str]] = [
        ({"parent_fingerprint": "0" * 64}, "parent fingerprint"),
        ({"operations": [{"op": "replace", "path": "", "value": {}}]}, "root"),
        (
            {"operations": [{"op": "replace", "path": "/canonical_task/sha256", "value": "0" * 64}]},
            "canonical_task",
        ),
        ({"operations": [{"op": "remove", "path": "/method/claim", "value": None}]}, "replace"),
    ]
    for mutation, message in unsafe_mutations:
        with pytest.raises(TaskDesignError, match=message):
            apply_task_design_patch(
                _patch(tmp_path, design, "/method/claim", "Better.", **mutation),
                parent=design,
                canonical_task=task_path,
                repair_review=repair_review,
            )


# --------------------------------------------------------------------------- aggregation / adjudication


def test_aggregate_task_design_reviews_majority_minority_and_review_count(
    tmp_path: Path, design: TaskDesign
) -> None:
    # uses a three-review majority
    reviews = _grounded_reviews(
        tmp_path, design, [(None, BUDGET), ("compute_realism", BUDGET), ("compute_realism", BUDGET)]
    )
    aggregate = aggregate_task_design_reviews(reviews)
    assert aggregate.failed_dimensions == ("compute_realism",)
    assert aggregate.payload["schema_version"] == 3
    assert aggregate.payload["summary"] == "Consensus of independent task-design reviews."

    # requires adjudication for a minority failure
    reviews = _grounded_reviews(
        tmp_path, design, [("search_proxy_fidelity", BUDGET), (None, BUDGET), (None, BUDGET)]
    )
    with pytest.raises(TaskDesignAdjudicationRequired, match="search_proxy_fidelity"):
        aggregate_task_design_reviews(reviews)

    # requires exactly three reviews
    reviews = _grounded_reviews(tmp_path, design, [(None, BUDGET), ("compute_realism", BUDGET)])
    with pytest.raises(TaskDesignError, match="exactly three"):
        aggregate_task_design_reviews(reviews)


def test_adjudication_reassigns_minority_findings_and_combines_repair_scope(
    tmp_path: Path, design: TaskDesign
) -> None:
    # reassigns a noncanonical minority finding
    reviews = _grounded_reviews(
        tmp_path, design, [("search_proxy_fidelity", BUDGET), (None, BUDGET), (None, BUDGET)]
    )
    adjudication = _adjudicate(
        tmp_path, design, reviews, [_adjudication_decision(**UPHOLD_ISOLATION)]
    )
    aggregate = aggregate_task_design_reviews(reviews, adjudication=adjudication)
    assert aggregate.failed_dimensions == ("search_final_isolation",)
    failed_row = next(row for row in aggregate.payload["dimensions"] if row["verdict"] == "fail")
    assert failed_row["repair_paths"] == [BUDGET]
    assert "adjudication" in aggregate.payload["summary"]

    # an adjudicated repair is combined with the canonical majority
    reviews = _grounded_reviews(
        tmp_path,
        design,
        [
            ("search_proxy_fidelity", "/candidate/state_semantics"),
            ("search_final_isolation", BUDGET),
            ("search_final_isolation", "/planes/search/budget/training_steps"),
        ],
    )
    adjudication = _adjudicate(
        tmp_path,
        design,
        reviews,
        [_adjudication_decision(**UPHOLD_ISOLATION, path="/candidate/state_semantics")],
    )
    aggregate = aggregate_task_design_reviews(reviews, adjudication=adjudication)
    assert _row(aggregate.payload, "search_final_isolation")["repair_paths"] == [
        "/planes/search/budget/training_steps",
        "/candidate/state_semantics",
    ]

    # an adjudicated repair is combined with the unanimous scope fallback
    payloads = [
        _grounded_review(design.fingerprint, "search_final_isolation", path=path)
        for path in (
            "/planes/search/resources/0",
            "/planes/search/metrics/0",
            "/planes/search/correctness/0",
        )
    ]
    _set_review_failure(payloads[0], "search_proxy_fidelity", "/candidate/state_semantics")
    reviews = _load_reviews(tmp_path, design, payloads)
    adjudication = _adjudicate(
        tmp_path,
        design,
        reviews,
        [_adjudication_decision(**UPHOLD_ISOLATION, path="/candidate/state_semantics")],
    )
    aggregate = aggregate_task_design_reviews(
        reviews, adjudication=adjudication, allow_unanimous_scope_fallback=True
    )
    isolation = _row(aggregate.payload, "search_final_isolation")
    assert "/candidate/state_semantics" in isolation["repair_paths"]
    assert len(isolation["repair_paths"]) == 2
    alternatives = task_design_scope_alternatives(reviews, adjudication=adjudication)
    isolation_alternatives = _row(alternatives, "search_final_isolation")
    assert isolation_alternatives["preserved_repair_paths"] == ["/candidate/state_semantics"]
    assert isolation_alternatives["preserved_remediations"] == ["Repair the upheld blocker."]


def test_adjudication_is_bound_to_source_reviews_and_validates_decisions(
    tmp_path: Path, design: TaskDesign
) -> None:
    # bound to ordered source reviews
    reviews = _grounded_reviews(
        tmp_path, design, [("search_proxy_fidelity", BUDGET), (None, BUDGET), (None, BUDGET)]
    )
    adjudication_path = _write_adjudication(
        tmp_path,
        reviews,
        [_adjudication_decision(**{**UPHOLD_ISOLATION, "verdict": "reject"})],
    )
    with pytest.raises(TaskDesignError, match="source reviews"):
        load_task_design_adjudication(
            adjudication_path, design=design, reviews=[reviews[1], reviews[0], reviews[2]]
        )
    changed_payload = _grounded_review(design.fingerprint)
    changed_payload["summary"] = "Different valid review."
    changed = _load_review(tmp_path, design, changed_payload, name="review_1.json", schema=3)
    with pytest.raises(TaskDesignError, match="source reviews"):
        load_task_design_adjudication(
            adjudication_path, design=design, reviews=[reviews[0], changed, reviews[2]]
        )

    # rejected without minority findings
    reviews = _grounded_reviews(tmp_path, design, [(None, BUDGET)] * 3)
    with pytest.raises(TaskDesignError, match="was not requested"):
        _adjudicate(tmp_path, design, reviews, [])

    # allows protected design evidence paths
    reviews = _grounded_reviews(
        tmp_path,
        design,
        [("search_proxy_fidelity", "/method/claim"), (None, "/method/claim"), (None, "/method/claim")],
    )
    decision = _adjudication_decision(
        **{**UPHOLD_ISOLATION, "canonical_dimension": "search_proxy_fidelity"}, path="/method/claim"
    )
    decision["design_paths"] = ["/canonical_task"]
    adjudication = _adjudicate(tmp_path, design, reviews, [decision])
    assert adjudication.payload["decisions"][0]["design_paths"] == ["/canonical_task"]


def test_aggregate_repair_scope_requires_majority_or_unanimous_narrow_fallback(
    tmp_path: Path, design: TaskDesign
) -> None:
    # a failed majority intersects repair authority
    reviews = _grounded_reviews(
        tmp_path,
        design,
        [
            ("compute_realism", BUDGET),
            ("compute_realism", "/planes/search/budget/training_steps"),
            (None, "/candidate/interface"),
        ],
    )
    row = _row(aggregate_task_design_reviews(reviews).payload, "compute_realism")
    assert row["design_paths"] == [BUDGET, "/planes/search/budget/training_steps"]
    assert row["repair_paths"] == ["/planes/search/budget/training_steps"]

    # a unanimous failure uses one narrow-scope fallback
    proposed_paths = ("/planes/search/resources/0", "/planes/search/metrics", "/candidate/state_semantics")
    reviews = _grounded_reviews(
        tmp_path, design, [("search_proxy_fidelity", path) for path in proposed_paths]
    )
    aggregate = aggregate_task_design_reviews(reviews, allow_unanimous_scope_fallback=True)
    row = _row(aggregate.payload, "search_proxy_fidelity")
    assert row["repair_paths"] == ["/planes/search/resources/0"]
    assert set(row["repair_paths"]) != set(proposed_paths)
    assert "unanimous narrow-scope fallback" in aggregate.payload["summary"]

    # the unanimous scope fallback selects one equally deep path
    proposals = (
        ("/planes/search/resources/0",),
        ("/planes/search/metrics/0", "/planes/search/correctness/0"),
        ("/candidate/state_semantics",),
    )
    payloads = []
    for paths in proposals:
        payload = _grounded_review(design.fingerprint, "search_proxy_fidelity", path=paths[0])
        _row(payload, "search_proxy_fidelity").update(
            {"design_paths": list(paths), "repair_paths": list(paths)}
        )
        payloads.append(payload)
    aggregate = aggregate_task_design_reviews(
        _load_reviews(tmp_path, design, payloads), allow_unanimous_scope_fallback=True
    )
    failed_row = next(row for row in aggregate.payload["dimensions"] if row["verdict"] == "fail")
    assert failed_row["repair_paths"] == ["/planes/search/correctness/0"]

    # rejects failures without a majority repair path
    no_majority_cases: list[tuple[str, list[tuple[str | None, str]]]] = [
        (
            "majority-scope-conflict-precedes-minority-adjudication",
            [
                ("compute_realism", "/method/claim"),
                ("compute_realism", BUDGET),
                ("search_proxy_fidelity", "/candidate/state_semantics"),
            ],
        ),
        (
            "failure-without-majority-repair-authority",
            [
                ("compute_realism", "/method/claim"),
                ("compute_realism", BUDGET),
                (None, "/candidate/interface"),
            ],
        ),
    ]
    for name, cases in no_majority_cases:
        reviews = _grounded_reviews(tmp_path, design, cases)
        with pytest.raises(TaskDesignError, match="no majority repair path"):
            aggregate_task_design_reviews(reviews)
        with pytest.raises(TaskDesignError, match="no majority repair path"):
            aggregate_task_design_reviews(reviews, allow_unanimous_scope_fallback=True)


# --------------------------------------------------------------------------- design CLI


def _cli_argv(command: str, tmp_path: Path, review_paths: list[Path], *extra: str) -> list[str]:
    return [
        "design",
        command,
        "--design",
        str(tmp_path / "task_design.json"),
        "--task",
        str(tmp_path / "task.md"),
        *(argument for path in review_paths for argument in ("--review", str(path))),
        *extra,
    ]


def test_aggregate_reviews_cli_emits_majority_and_uses_adjudication(
    tmp_path: Path, design: TaskDesign, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # emits the v3 majority
    review_paths = [
        _write_json(
            tmp_path / f"majority_review_{index}.json",
            _grounded_review(design.fingerprint, "compute_realism" if index > 0 else None),
        )
        for index in range(3)
    ]
    monkeypatch.setattr(
        sys, "argv", _cli_argv("aggregate-reviews", tmp_path, review_paths, "--require-schema-version", "3")
    )
    exit_code = design_cli_main()
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["schema_version"] == 3
    assert output["verdict"] == "fail"

    # requests, then uses, adjudication
    review_paths = [
        _write_json(tmp_path / f"review_{index}.json", _grounded_review(design.fingerprint, failed))
        for index, failed in enumerate(("search_proxy_fidelity", None, None))
    ]
    base_argv = _cli_argv("aggregate-reviews", tmp_path, review_paths, "--require-schema-version", "3")
    monkeypatch.setattr(sys, "argv", base_argv)

    assert design_cli_main() == 4
    assert "requires adjudication" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", _cli_argv("adjudication-request", tmp_path, review_paths))
    assert design_cli_main() == 0
    request = json.loads(capsys.readouterr().out)
    adjudication_path = _write_json(
        tmp_path / "adjudication.json",
        {
            "schema_version": 1,
            "design_fingerprint": request["design_fingerprint"],
            "request_fingerprint": request["request_fingerprint"],
            "source_review_fingerprints": request["source_review_fingerprints"],
            "decisions": [
                _adjudication_decision(
                    source_review_index=0,
                    source_dimension="search_proxy_fidelity",
                    verdict="reject",
                    canonical_dimension="search_proxy_fidelity",
                )
            ],
        },
    )
    monkeypatch.setattr(sys, "argv", [*base_argv, "--adjudication", str(adjudication_path)])

    assert design_cli_main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["verdict"] == "pass"
    assert "adjudication" in output["summary"]


# --------------------------------------------------------------------------- compare_task_design_reviews


def _bump_wall_clock(payload: dict[str, Any]) -> None:
    payload["planes"]["search"]["budget"]["wall_clock_seconds"] = 7200


@pytest.mark.parametrize(
    ("mutate", "candidate_review", "improved", "regressed"),
    [
        pytest.param(None, None, True, (), id="strict-monotonic-progress-v1"),
        pytest.param(
            _bump_wall_clock,
            {"failed": "reproducibility", "path": "/candidate/state_semantics"},
            True,
            (),
            id="ignores-new-finding-on-unchanged-paths",
        ),
        pytest.param(
            _change_claim,
            {"failed": "reproducibility", "path": BUDGET, "design_path": "/method"},
            False,
            ("reproducibility",),
            id="rejects-new-failure-on-changed-ancestor",
        ),
    ],
)
def test_compare_task_design_reviews(
    tmp_path: Path,
    task_path: Path,
    design: TaskDesign,
    mutate: Callable[[dict[str, Any]], None] | None,
    candidate_review: dict[str, Any] | None,
    improved: bool,
    regressed: tuple[str, ...],
) -> None:
    if mutate is None:
        candidate_design = design
        parent_payload = _review(design.fingerprint, failed="compute_realism")
        candidate_payload = _review(design.fingerprint)
    else:
        assert candidate_review is not None
        candidate_design = _candidate_design(tmp_path, task_path, mutate)
        parent_payload = _grounded_review(design.fingerprint, "compute_realism")
        candidate_payload = _grounded_review(candidate_design.fingerprint, **candidate_review)
    parent = _load_review(tmp_path, design, parent_payload, name="parent.json")
    candidate = _load_review(tmp_path, candidate_design, candidate_payload, name="candidate.json")

    progress = compare_task_design_reviews(
        parent, candidate, parent_design=design, candidate_design=candidate_design
    )

    assert progress.improved is improved
    assert progress.fixed == ("compute_realism",)
    assert progress.regressed == regressed


# --------------------------------------------------------------------------- delta reviews


def test_delta_review_binds_regressions_to_changed_paths(
    tmp_path: Path, task_path: Path, design: TaskDesign
) -> None:
    candidate = _candidate_design(tmp_path, task_path)

    # records a regression inside a still-failing dimension
    delta = _delta(
        tmp_path,
        design,
        candidate,
        verdict="fail",
        fixed=["compute_realism"],
        regressions=[_regression("reproducibility", "/method/claim")],
    )
    assert delta.verdict == "fail"
    assert delta.regressed_dimensions == ("reproducibility",)

    # rejects a regression on an unchanged path
    with pytest.raises(TaskDesignError, match="was not changed by the candidate"):
        _delta(
            tmp_path,
            design,
            candidate,
            verdict="fail",
            fixed=[],
            regressions=[_regression("reproducibility", "/planes/search/seeds")],
        )


def _add_entrypoint(payload: dict[str, Any]) -> None:
    payload["upstream"][0]["entrypoints"] = ["train.py", "invalid smoke claim"]


@pytest.mark.parametrize(
    ("mutate", "delta", "candidate_review", "claimed", "message"),
    [
        pytest.param(
            _change_claim,
            dict(
                schema=1,
                verdict="fail",
                fixed=["compute_realism"],
                regressions=[_regression("reproducibility", "/method/claim", legacy=True)],
            ),
            {"failed": "reproducibility", "path": "/method/claim"},
            TaskDesignReviewProgress(("compute_realism",), (), ("reproducibility",)),
            "cannot authorize a repair frontier",
            id="legacy-v1-cannot-authorize-frontier",
        ),
        pytest.param(
            _change_claim,
            dict(verdict="pass", fixed=[], regressions=[]),
            {},
            TaskDesignReviewProgress(("compute_realism",), (), ()),
            "fixed dimensions",
            id="must-confirm-every-consensus-fix",
        ),
        pytest.param(
            _change_claim,
            dict(verdict="pass", fixed=["compute_realism"], regressions=[]),
            {"failed": "reproducibility", "path": "/method/claim"},
            TaskDesignReviewProgress(("compute_realism",), (), ("reproducibility",)),
            None,
            id="can-dispute-consensus-regression",
        ),
        pytest.param(
            _change_claim,
            dict(verdict="pass", fixed=["compute_realism"], regressions=[]),
            {"failed": "reproducibility", "path": "/upstream/0/license", "design_path": "/method/claim"},
            TaskDesignReviewProgress(("compute_realism",), (), ("reproducibility",)),
            None,
            id="can-reclassify-disjoint-repair-scope-as-latent",
        ),
        pytest.param(
            _change_claim,
            dict(
                verdict="fail",
                fixed=["compute_realism"],
                regressions=[_regression("reproducibility", "/method/claim", repair_paths=["/upstream"])],
            ),
            {"failed": "reproducibility", "path": "/method/claim"},
            TaskDesignReviewProgress(("compute_realism",), (), ("reproducibility",)),
            "repair paths do not match",
            id="repair-paths-must-match-candidate-consensus",
        ),
        pytest.param(
            _add_entrypoint,
            dict(
                verdict="fail",
                fixed=["compute_realism", "reproducibility"],
                regressions=[_regression("upstream_grounding", "/upstream/0/entrypoints/1")],
            ),
            {},
            TaskDesignReviewProgress(("upstream_grounding", "compute_realism", "reproducibility"), (), ()),
            None,
            id="can-dispute-consensus-fix-as-regression",
        ),
    ],
)
def test_confirm_task_design_delta(
    tmp_path: Path,
    task_path: Path,
    design: TaskDesign,
    mutate: Callable[[dict[str, Any]], None],
    delta: dict[str, Any],
    candidate_review: dict[str, Any],
    claimed: TaskDesignReviewProgress,
    message: str | None,
) -> None:
    candidate = _candidate_design(tmp_path, task_path, mutate)
    delta_review = _delta(tmp_path, design, candidate, **delta)
    review = _load_review(
        tmp_path,
        candidate,
        _grounded_review(candidate.fingerprint, **candidate_review),
        name="candidate_review.json",
        schema=3,
    )

    if message is None:
        confirm_task_design_delta(delta_review, claimed, review)
    else:
        with pytest.raises(TaskDesignError, match=message):
            confirm_task_design_delta(delta_review, claimed, review)


# --------------------------------------------------------------------------- latent review rebinding


def _latent_review(fingerprint: str, failed: dict[str, str]) -> TaskDesignReview:
    dimensions = []
    for identifier in DESIGN_DIMENSIONS:
        path = failed.get(identifier, "/stable")
        verdict = "fail" if identifier in failed else "pass"
        dimensions.append(
            {
                "id": identifier,
                "verdict": verdict,
                "evidence": [f"Evidence for {identifier}."],
                "design_paths": [path],
                "task_evidence": ["task.md:1: Task."],
                "repair_paths": [path] if verdict == "fail" else [],
                "remediation": f"Repair {identifier}." if verdict == "fail" else "",
            }
        )
    verdict = "fail" if failed else "pass"
    payload = {
        "schema_version": 3,
        "verdict": verdict,
        "summary": "Fixture review.",
        "design_fingerprint": fingerprint,
        "dimensions": dimensions,
    }
    return TaskDesignReview(verdict, tuple(failed), payload)


def _mix_changed_evidence(parent_review: TaskDesignReview, candidate_review: TaskDesignReview) -> None:
    _row(candidate_review.payload, "metrics_and_rubrics")["design_paths"] = ["/stable", "/changed"]


def _exceed_size_limit(parent_review: TaskDesignReview, candidate_review: TaskDesignReview) -> None:
    _row(parent_review.payload, "compute_realism")["evidence"] = ["p" * 600_000]
    _row(candidate_review.payload, "metrics_and_rubrics")["evidence"] = ["c" * 600_000]


def test_rebind_latent_failures_authorizes_only_unchanged_nonregression_findings() -> None:
    # authorizes only unchanged, non-regression findings
    parent = TaskDesign({"stable": "old", "stable_new": "old", "changed": "old"}, "parent")
    candidate = TaskDesign({"stable": "old", "stable_new": "old", "changed": "new"}, "candidate")
    parent_review = _latent_review(parent.fingerprint, {"compute_realism": "/stable"})
    candidate_review = _latent_review(
        candidate.fingerprint,
        {
            "compute_realism": "/stable_new",
            "metrics_and_rubrics": "/stable",
            "reproducibility": "/changed",
            "publication_scope": "/changed",
        },
    )

    rebound = rebind_latent_failures(
        parent=parent,
        candidate=candidate,
        parent_review=parent_review,
        candidate_review=candidate_review,
        delta_review=TaskDesignDeltaReview("fail", ("compute_realism",), ("reproducibility",), {}),
        consensus_regressions=(),
    )

    assert rebound is not None
    assert rebound.payload["design_fingerprint"] == parent.fingerprint
    assert rebound.failed_dimensions == ("metrics_and_rubrics", "compute_realism")
    rows = {row["id"]: row for row in rebound.payload["dimensions"]}
    assert rows["compute_realism"] == _row(parent_review.payload, "compute_realism")
    assert rows["metrics_and_rubrics"]["repair_paths"] == ["/stable"]
    assert rows["reproducibility"]["verdict"] == "pass"
    assert rows["publication_scope"]["verdict"] == "pass"

    # returns None without safe latent findings
    cases: list[tuple[str, dict[str, str], dict[str, str], dict[str, str], dict[str, str],
                      TaskDesignDeltaReview, tuple[str, ...],
                      Callable[[TaskDesignReview, TaskDesignReview], None] | None]] = [
        (
            "no-safe-latent-findings",
            {"changed": "old"},
            {"changed": "new"},
            {},
            {"reproducibility": "/changed"},
            TaskDesignDeltaReview("fail", (), ("reproducibility",), {}),
            (),
            None,
        ),
        (
            "consensus-regression-overrides-disputing-delta",
            {"stable": "old"},
            {"stable": "old"},
            {},
            {"metrics_and_rubrics": "/stable"},
            TaskDesignDeltaReview("pass", (), (), {}),
            ("metrics_and_rubrics",),
            None,
        ),
        (
            "mixed-changed-and-unchanged-evidence",
            {"stable": "old", "changed": "old"},
            {"stable": "old", "changed": "new"},
            {},
            {"metrics_and_rubrics": "/stable"},
            TaskDesignDeltaReview("pass", (), (), {}),
            (),
            _mix_changed_evidence,
        ),
        (
            "synthesized-review-above-contract-size-limit",
            {"stable": "old", "stable_new": "old"},
            {"stable": "old", "stable_new": "old"},
            {"compute_realism": "/stable"},
            {"metrics_and_rubrics": "/stable_new"},
            TaskDesignDeltaReview("pass", (), (), {}),
            (),
            _exceed_size_limit,
        ),
    ]
    for name, parent_payload, candidate_payload, parent_failed, candidate_failed, delta, consensus, tweak in cases:
        parent = TaskDesign(parent_payload, "parent")
        candidate = TaskDesign(candidate_payload, "candidate")
        parent_review = _latent_review(parent.fingerprint, parent_failed)
        candidate_review = _latent_review(candidate.fingerprint, candidate_failed)
        if tweak is not None:
            tweak(parent_review, candidate_review)

        rebound = rebind_latent_failures(
            parent=parent,
            candidate=candidate,
            parent_review=parent_review,
            candidate_review=candidate_review,
            delta_review=delta,
            consensus_regressions=consensus,
        )

        assert rebound is None, name


# --------------------------------------------------------------------------- task_design.sh focus


def _run_shell_function(
    function: str, payload: dict[str, Any], argument: str | None = None
) -> dict[str, Any]:
    invocation = function if argument is None else f'{function} "$3"'
    script = f'source "$1/src/scripts/lib/task_design.sh"\nPYTHON_BIN="$2"\n{invocation}\n'
    args = ["bash", "-c", script, "bash", str(REPO_ROOT), sys.executable]
    if argument is not None:
        args.append(argument)
    result = subprocess.run(
        args, input=json.dumps(payload), capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _focus(payload: dict[str, Any], history: list[str]) -> dict[str, Any]:
    return _run_shell_function("focus_design_repair_payload", payload, json.dumps(history))


def _focused_ids(payloads: list[dict[str, Any]]) -> list[str]:
    history: list[str] = []
    for payload in payloads:
        history.append(_focus(payload, history)["review"]["dimensions"][0]["id"])
    return history


def _failing(*identifiers: str) -> dict[str, Any]:
    return {"dimensions": [{"id": identifier, "verdict": "fail"} for identifier in identifiers]}


def _previous(**fields: Any) -> dict[str, Any]:
    return {"fixed": [], "still_failed": [], "regressed": [], "delta_regressions": [], **fields}


THREE = ("metrics_and_rubrics", "execution_feasibility", "reproducibility")


@pytest.mark.parametrize(
    ("payload", "attempts", "expected"),
    [
        pytest.param({"review": _failing(*THREE)}, 3, list(THREE), id="rotates-across-persistent-failures"),
        pytest.param(
            {
                "review": _failing("reproducibility", "metrics_and_rubrics", "execution_feasibility"),
                "previous_candidate": _previous(
                    still_failed=list(THREE), regressed=["execution_feasibility"]
                ),
            },
            3,
            ["execution_feasibility", "metrics_and_rubrics", "reproducibility"],
            id="prioritizes-regression-then-rotates",
        ),
        pytest.param(
            {"review": _failing("reproducibility", "compute_realism", "metrics_and_rubrics")},
            3,
            ["metrics_and_rubrics", "compute_realism", "reproducibility"],
            id="stable-canonical-dimension-order",
        ),
        pytest.param(
            {
                "review": _failing("reproducibility", "metrics_and_rubrics"),
                "previous_candidate": _previous(
                    fixed=["compute_realism"], still_failed=["metrics_and_rubrics", "reproducibility"]
                ),
            },
            4,
            ["metrics_and_rubrics", "reproducibility", "metrics_and_rubrics", "reproducibility"],
            id="excludes-fixed-or-removed-dimensions",
        ),
    ],
)
def test_focus_rotation(payload: dict[str, Any], attempts: int, expected: list[str]) -> None:
    assert _focused_ids([{**payload, "attempt": attempt} for attempt in range(attempts)]) == expected


def test_changing_regressions_cannot_starve_persistent_failures() -> None:
    payloads = [
        {
            "review": _failing(*THREE),
            "previous_candidate": _previous(still_failed=list(THREE), regressed=[regressed]),
        }
        for regressed in THREE
    ]

    assert _focused_ids(payloads) == list(THREE)


def test_focus_prioritizes_worsened_failure_and_retains_new_regression_as_constraint() -> None:
    # prioritizes a worsened parent failure
    payload = {
        "review": _failing("search_proxy_fidelity", "metrics_and_rubrics"),
        "previous_candidate": _previous(
            fixed=["search_proxy_fidelity"],
            still_failed=["metrics_and_rubrics"],
            regressed=["metrics_and_rubrics"],
            candidate_findings=[
                {"id": "search_proxy_fidelity", "remediation": "stale"},
                {"id": "metrics_and_rubrics", "remediation": "align metrics"},
            ],
            delta_regressions=[{"dimension": "metrics_and_rubrics", "remediation": "align metrics"}],
        ),
    }
    focused = _run_shell_function("focus_design_repair_payload", payload)
    assert focused["review"]["dimensions"] == [{"id": "metrics_and_rubrics", "verdict": "fail"}]
    assert focused["previous_candidate"]["regressed"] == ["metrics_and_rubrics"]
    assert focused["previous_candidate"]["candidate_findings"] == [
        {"id": "metrics_and_rubrics", "remediation": "align metrics"}
    ]

    # retains a new regression as a negative constraint
    payload = {
        "review": _failing("search_proxy_fidelity"),
        "previous_candidate": _previous(
            still_failed=["search_proxy_fidelity"],
            regressed=["reproducibility"],
            candidate_findings=[{"id": "reproducibility", "remediation": "unrelated review drift"}],
            delta_regressions=[
                {"dimension": "reproducibility", "remediation": "do not weaken the pinned revision"}
            ],
        ),
    }
    focused = _run_shell_function("focus_design_repair_payload", payload)
    assert focused["review"]["dimensions"][0]["id"] == "search_proxy_fidelity"
    assert "candidate_findings" not in focused["previous_candidate"]
    assert focused["previous_candidate"]["regressed"] == ["reproducibility"]
    assert focused["previous_candidate"]["delta_regressions"][0]["remediation"] == (
        "do not weaken the pinned revision"
    )

    # a review-only rebound preserves the focus history
    source = TASK_DESIGN_SH.read_text(encoding="utf-8")
    rebound = source.split('best_review="$latent_parent_review"', 1)[1].split(
        "append_design_ledger", 1
    )[0]
    assert 'focused_repair_history="[]"' not in rebound


def test_task_design_sh_focused_authorization_and_private_root(tmp_path: Path) -> None:
    # focused authorization rejects the other failed dimension
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    task_path = task_dir / "task.md"
    task_path.write_text("# Goal\n- optimize objective\n\n# Rubrics\n- score threshold\n", encoding="utf-8")
    design = _load_design(tmp_path, task_path, _task_design_payload(task_path))
    review = _grounded_review(design.fingerprint)
    review["verdict"] = "fail"
    review["summary"] = "Two independently repairable failures."
    for identifier, path in (("compute_realism", BUDGET), ("reproducibility", "/method/claim")):
        _row(review, identifier).update(
            {"verdict": "fail", "repair_paths": [path], "remediation": "Repair this dimension."}
        )
    review_path = _write_json(tmp_path / "review.json", review)
    feedback_path = _write_json(
        tmp_path / "feedback.json", {"mode": "focused", "review": _failing("compute_realism")}
    )
    authorization_path = tmp_path / "focused_review.json"
    script = (
        'source "$1/src/scripts/lib/task_design.sh"\nPYTHON_BIN="$2"\nfeedback="$(<"$3")"\n'
        'write_focused_design_review "$4" "$feedback" "$5"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script, "bash", str(REPO_ROOT), sys.executable, str(feedback_path),
         str(review_path), str(authorization_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    authorization = load_task_design_review(authorization_path, design=design, require_schema_version=3)
    assert authorization.failed_dimensions == ("compute_realism",)

    with pytest.raises(TaskDesignError, match="outside failed repair paths"):
        apply_task_design_patch(
            _patch(tmp_path, design, "/method/claim", "Changed."),
            parent=design,
            canonical_task=task_path,
            repair_review=authorization,
        )

    # the design workflow's private agent homes must use the configured host scratch
    scratch = tmp_path / "scratch with spaces"
    scratch.mkdir(mode=0o700)
    output = tmp_path / "output"
    output.mkdir()
    script = """
set -eu
source "$1/src/scripts/lib/task_design.sh"
REPO_ROOT="$1" OUTPUT_DIR="$2" TASK_DRAFTING_WORKFLOW=robust
task_progress_emit() { :; }
render_design_prompt() { return 1; }
if run_design_workflow; then
  echo 'unexpected workflow continuation' >&2
  exit 1
fi
printf '%s\\n' "$DESIGN_PRIVATE_ROOT"
"""
    result = subprocess.run(
        ["bash", "-c", script, "bash", str(REPO_ROOT), str(output)],
        env=os.environ | {"TMPDIR": str(scratch)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    private = Path(result.stdout.strip())
    assert private.parent == scratch
    assert private.stat().st_mode & 0o777 == 0o700
    assert "Could not render initial task design prompt" in result.stderr
