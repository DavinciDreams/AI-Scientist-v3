"""Execute the repository-maintained recipes delivered to scaffold authors and repair agents."""

from __future__ import annotations

import base64
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.final_evaluation.candidate_broker import MAX_RPC_BYTES
from src.task_drafting.harness_capability_policy import unavailable_harness_environment_findings
from src.task_drafting.prompt_contract import expand_evidence_contract
from tests.task_drafting.test_scaffold_diagnostic_contract import (
    PROMPTS,
    render_prompt,
    render_qualification_prompt,
    render_untrusted_task,
)

ENVIRONMENT_MARKER = "{{CANDIDATE_ENVIRONMENT_RECIPE}}"
EVALUATORS = (("search", "evaluate.py"),)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(params=["draft_task.md", "repair_task.md", "qualification_repair"])
def recipe_prompt(request, monkeypatch) -> str:
    if request.param == "qualification_repair":
        return render_qualification_prompt(monkeypatch)
    return expand_evidence_contract((PROMPTS / request.param).read_text())


@pytest.fixture
def recipes(recipe_prompt) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    # Only repository-maintained prompt code is executed, never task or candidate text.
    for block in re.findall(r"```python\n(.*?)\n```", recipe_prompt, re.DOTALL):
        exec(compile(block, "scaffold_recipe", "exec"), namespace)
    return namespace


@pytest.fixture
def schedule(recipe_prompt, recipes) -> Any:
    match = re.search(r"```python\n(def run_declared_episodes\(.*?)\n```", recipe_prompt, re.DOTALL)
    assert match is not None, "Scaffold authors need the executable scheduling recipe"
    return recipes["run_declared_episodes"]


@pytest.fixture(params=[
    "draft_task.md", "repair_task.md", "repair_task_smoke.md", "qualification_repair",
])
def environment_recipe(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, Any]:
    prompt = render_prompt(request.param, tmp_path, monkeypatch, "# Goal\nTest candidate environment\n")
    assert ENVIRONMENT_MARKER not in prompt
    match = re.search(r"```python\n(def candidate_subprocess_settings\(.*?)\n```", prompt, re.DOTALL)
    assert match is not None, "Scaffold authors need the executable environment recipe"
    source = match.group(1)
    namespace: dict[str, Any] = {}
    exec(compile(source, "candidate_environment_recipe", "exec"), namespace)
    return source, namespace["candidate_subprocess_settings"]


@dataclass
class ClockedCandidate:
    response_seconds: float
    now: float = 2.0  # Startup already consumed part of the shared plane budget.
    reset_delay: float = 0.4
    resets: list[int] = field(default_factory=list)
    timeouts: int = 0

    def clock(self) -> float:
        return self.now

    def rpc(self, request: dict[str, Any], *, timeout: float) -> dict[str, float]:
        assert timeout > 0
        resetting = request["op"] == "reset"
        if resetting:
            self.resets.append(request["seed"])
        delay = self.reset_delay if resetting else self.response_seconds
        self.now += min(delay, timeout)
        if delay > timeout:
            self.timeouts += 1
            raise TimeoutError("bounded candidate response expired")
        return {"ok": True} if resetting else {"value": 1.0}


TWO_SEEDS = [{"seed": 3}, {"seed": 7}]


def episode_groups() -> list[dict[str, Any]]:
    # Auxiliary diagnostic branches are declared episodes, including repeated seeds.
    return [
        {"seed": seed, "role": role}
        for role, seeds in [("scored", [3, 7]), ("diagnostic", [3, 3, 7, 7])]
        for seed in seeds
    ]


def run_episodes(schedule, candidate, episodes, *, evaluate, rpc=None, record_failure=None,
                 worst_case=lambda episode: -1.0, **overrides):
    """Run the scheduling recipe with the shared fixture plane budget."""
    kwargs = dict(hard_deadline=15.0, finish_reserve=2.0, reset_timeout=0.5,
                  episode_overhead=0.25, rpc_cap=1.0)
    kwargs.update(overrides)
    return schedule(
        list(episodes), clock=candidate.clock, rpc=rpc or candidate.rpc, evaluate=evaluate,
        worst_case=worst_case, record_failure=record_failure or (lambda failure: None), **kwargs,
    )


def evaluate_episode(episode: dict[str, Any], call: Any, deadline: float) -> float:
    for _ in range(6):
        call({"op": "step", "role": episode["role"]})
    return 1.0


def run_recipe(schedule: Any, candidate: ClockedCandidate) -> tuple[list[float], bool]:
    def evaluate_with_cleanup(episode: dict[str, Any], call: Any, deadline: float) -> float:
        try:
            return evaluate_episode(episode, call, deadline)
        finally:
            candidate.now += 0.25

    return run_episodes(schedule, candidate, episode_groups(), evaluate=evaluate_with_cleanup)


# --------------------------------------------------------------------------- scheduling recipe


def test_scheduling_recipe_budgets_fast_slow_and_blocked_candidates(schedule: Any) -> None:
    # fast responses complete all scored and diagnostic episodes
    candidate = ClockedCandidate(response_seconds=0.05)
    outcomes, complete = run_recipe(schedule, candidate)
    assert complete
    assert outcomes == [1.0] * 6
    assert candidate.resets == [3, 7, 3, 3, 7, 7]

    # slow or timed-out replies preserve later resets
    for response_seconds in (0.9, 2.0):
        candidate = ClockedCandidate(response_seconds=response_seconds)
        outcomes, complete = run_recipe(schedule, candidate)
        assert candidate.resets == [3, 7, 3, 3, 7, 7], response_seconds
        assert not complete, response_seconds
        assert len(outcomes) == 6, response_seconds
        assert -1.0 in outcomes, response_seconds
        assert all(math.isfinite(value) for value in outcomes), response_seconds
        assert math.isfinite(sum(outcomes) / 6), response_seconds
        assert candidate.timeouts > 0, response_seconds
        # Output writing and cleanup retain their two seconds.
        assert candidate.now <= 13.0, response_seconds

    # every declared reset still runs with failure contributions
    failing_configs: list[tuple[str, dict[str, Any]]] = [
        ("timed-out-resets", dict(response_seconds=0.05, reset_delay=0.8)),
        ("no-budget-for-work", dict(response_seconds=0.05, now=8.5, reset_delay=0.5)),
    ]
    for name, config in failing_configs:
        candidate = ClockedCandidate(**config)
        outcomes, complete = run_recipe(schedule, candidate)
        assert candidate.resets == [3, 7, 3, 3, 7, 7], name
        assert outcomes == [-1.0] * 6, name
        assert not complete, name
        assert candidate.now <= 13.0, name

    # successful work can use its full budget and reserved cleanup
    candidate = ClockedCandidate(response_seconds=0.125, reset_delay=0.25)

    def evaluate_with_cleanup(episode: dict[str, Any], call: Any, deadline: float) -> float:
        value = evaluate_episode(episode, call, deadline)
        candidate.now += 0.25
        return value

    outcomes, complete = run_episodes(
        schedule, candidate, episode_groups()[:1], evaluate=evaluate_with_cleanup,
        hard_deadline=4.25, finish_reserve=1.0,
    )
    assert outcomes == [1.0]
    assert complete
    assert candidate.now == 3.25

    # setup cannot restart the plane clock
    candidate = ClockedCandidate(response_seconds=0.05, now=12.0)
    with pytest.raises(TimeoutError, match="reset reservations"):
        run_recipe(schedule, candidate)
    assert not candidate.resets


def test_sequential_shutdown_reserves_and_invalid_resets_preserve_later_resets(schedule) -> None:
    # sequential shutdown reserves preserve later resets and output
    for response_seconds in (0.05, 100.0):
        candidate = ClockedCandidate(response_seconds=response_seconds)
        candidate_close_seconds = 1.0 + 1.0  # Terminate wait, then kill/reap wait.
        baseline_cancel_seconds = 1.0 + 1.0
        dispatch_and_bookkeeping_seconds = 0.25
        episode_overhead = (
            candidate_close_seconds + baseline_cancel_seconds + dispatch_and_bookkeeping_seconds
        )
        cleanups: list[int] = []

        def evaluate(episode: dict[str, Any], call: Any, deadline: float,
                     candidate=candidate, cleanups=cleanups) -> float:
            try:
                while candidate.clock() < deadline:
                    call({"op": "train"})
                return 1.0
            finally:
                candidate.now += candidate_close_seconds
                candidate.now += baseline_cancel_seconds
                candidate.now += dispatch_and_bookkeeping_seconds
                cleanups.append(episode["seed"])

        outcomes, complete = run_episodes(
            schedule, candidate, episode_groups(), evaluate=evaluate,
            hard_deadline=40.0, episode_overhead=episode_overhead,
        )
        assert candidate.resets == cleanups == [3, 7, 3, 3, 7, 7], response_seconds
        assert len(outcomes) == len(cleanups), response_seconds
        assert all(math.isfinite(value) for value in outcomes), response_seconds
        assert candidate.now <= 38.0, response_seconds
        if response_seconds > 1.0:
            assert not complete
            assert outcomes == [-1.0] * 6

    # invalid reset retains penalty and remaining resets
    for response in ({}, {"ok": False}, {"ok": 1}, {"ok": True, "x": 1}, None):
        candidate = ClockedCandidate(response_seconds=0.05)
        failures: list[dict[str, str]] = []

        def rpc(request, *, timeout, candidate=candidate, response=response):
            candidate.rpc(request, timeout=timeout)
            return response

        outcomes, complete = run_episodes(
            schedule, candidate, TWO_SEEDS, rpc=rpc,
            evaluate=lambda *args: pytest.fail("invalid reset reached scoring"),
            record_failure=failures.append,
        )
        assert candidate.resets == [3, 7], response
        assert outcomes == [-1.0, -1.0], response
        assert not complete, response
        assert (
            failures
            == [{"component": "candidate", "category": "invalid_reset_response", "operation": "reset"}]
            * 2
        ), response


def test_scheduler_preserves_primary_failure_through_cleanup_and_sink_errors(schedule, recipes):
    # scheduler preserves train failure when cleanup calls rpc
    failures = []
    candidate = ClockedCandidate(response_seconds=0.05)

    def rpc(request, *, timeout):
        if request["op"] == "train":
            raise TimeoutError("private candidate payload")
        return candidate.rpc(request, timeout=timeout)

    def evaluate(episode, call, deadline):
        try:
            call({"op": "train"})
        finally:
            call({"op": "cleanup"})

    outcomes, complete = run_episodes(
        schedule, candidate, TWO_SEEDS, rpc=rpc, evaluate=evaluate, record_failure=failures.append,
    )
    assert failures == [{"component": "candidate", "category": "timeout",
                         "operation": "train"}] * 2
    assert candidate.resets == [3, 7]
    assert outcomes == [-1.0, -1.0] and not complete

    # cleanup timeout cannot replace baseline failure outside rpc
    failures = []
    candidate = ClockedCandidate(response_seconds=0.05)

    def cleanup_rpc(request, *, timeout):
        if request["op"] == "cleanup":
            raise TimeoutError("secondary cleanup timeout")
        return candidate.rpc(request, timeout=timeout)

    def evaluate_baseline(episode, call, deadline):
        try:
            recipes["fail_required_baseline"]("baseline_exit", lambda: None, lambda value: None)
        finally:
            call({"op": "cleanup"})

    outcomes, complete = run_episodes(
        recipes["run_declared_episodes"], candidate, TWO_SEEDS, rpc=cleanup_rpc,
        evaluate=evaluate_baseline, record_failure=failures.append,
    )
    assert failures == [{"component": "baseline", "category": "baseline_exit",
                         "operation": "baseline_worker"}] * 2
    assert candidate.resets == [3, 7]
    assert outcomes == [-1.0, -1.0] and not complete

    # episode diagnostic sink failure preserves remaining resets
    candidate = ClockedCandidate(response_seconds=0.05)

    def evaluate_raising(episode, call, deadline):
        raise recipes["RequiredBaselineFailure"]("baseline_exit")

    def failed_sink(failure):
        raise OSError("log storage unavailable")

    outcomes, complete = run_episodes(
        recipes["run_declared_episodes"], candidate, TWO_SEEDS, evaluate=evaluate_raising,
        record_failure=failed_sink,
    )
    assert candidate.resets == [3, 7]
    assert outcomes == [-1.0, -1.0] and not complete


# --------------------------------------------------------------------------- resource budgets


def test_cpu_budget_covers_parallel_threads_and_rejects_invalid_parameters(recipes):
    # cpu seconds cover parallel threads without extending wall deadline
    assert recipes["cpu_limit_seconds"](7200, 16) == 115200
    assert recipes["cpu_limit_seconds"](1.2, 2) == 3

    # cpu budget rejects invalid parameters
    for wall, threads in [(0, 2), (float("inf"), 2), (60, 0), (60, True)]:
        with pytest.raises(ValueError):
            recipes["cpu_limit_seconds"](wall, threads)


def plane_budget(wall_seconds):
    return dict(
        wall_seconds=wall_seconds, startup_seconds=2, episode_count=2, reset_seconds=1,
        cleanup_seconds=1, method_count=3, work_seconds=4, finish_seconds=2,
    )


def test_joint_profile_check_preserves_workload_and_rejects_unproductive_profiles(recipes):
    # joint profile check preserves declared workload
    profiles = dict(search=plane_budget(36), final=plane_budget(40))
    assert recipes["require_joint_workload_budget"](**profiles) == {"search": 36, "final": 36}

    # joint profile check rejects final even when search fits
    with pytest.raises(ValueError, match="final.*productive work"):
        recipes["require_joint_workload_budget"](search=plane_budget(100), final=plane_budget(20))

    # workload rejects nonproductive or nonfinite allowance
    for work_seconds in (0, -1, float("nan"), float("inf"), True):
        profile = plane_budget(100)
        profile["work_seconds"] = work_seconds
        with pytest.raises(ValueError):
            recipes["require_joint_workload_budget"](search=profile, final=plane_budget(100))


def test_method_deadlines_and_productive_baseline_calibration(recipes):
    # slow candidate cannot spend baseline allowances
    candidate = ClockedCandidate(response_seconds=100)
    budgets = []
    for remaining in [2, 1, 0]:
        deadline = recipes["reserved_method_deadline"](
            now=candidate.clock(), episode_deadline=17, methods_remaining=remaining,
            work_seconds=4, cleanup_seconds=1,
        )
        budgets.append(deadline - candidate.clock())
        if remaining == 2:
            with pytest.raises(TimeoutError):
                candidate.rpc({"op": "train"}, timeout=deadline - candidate.clock())
        else:
            candidate.now = deadline
        candidate.now += 1
    assert budgets == [4, 4, 4]
    assert candidate.now == 17

    # method preflight rejects insufficient baseline reserve
    with pytest.raises(TimeoutError, match="matched method"):
        recipes["reserved_method_deadline"](
            now=2, episode_deadline=16, methods_remaining=2, work_seconds=4, cleanup_seconds=1,
        )

    # productive baseline accepts measured successful work
    recipes["require_productive_baseline"](1.1, minimum_seconds=1)

    # penalty or short baseline is not productive calibration
    for measured in (0, 0.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="productive baseline"):
            recipes["require_productive_baseline"](measured, minimum_seconds=1)


# --------------------------------------------------------------------------- required baselines


def test_guard_required_baseline_skips_work_bounds_diagnostics_and_keeps_results(recipes):
    # failed required baseline skips work, not required resets
    candidate = ClockedCandidate(response_seconds=0.05)
    cancelled = []
    failures = []

    def evaluate(episode, call, deadline):
        guarded = recipes["guard_required_baseline"](
            call, lambda: "baseline_exit_signal", lambda: cancelled.append(episode["seed"]),
            failures.append,
        )
        guarded({"op": "train"})
        return 1.0

    outcomes, complete = run_episodes(
        recipes["run_declared_episodes"], candidate, TWO_SEEDS, evaluate=evaluate,
    )
    assert candidate.resets == [3, 7]
    assert cancelled == [3, 7]
    assert failures == ["baseline_exit_signal", "baseline_exit_signal"]
    assert outcomes == [-1.0, -1.0]
    assert not complete
    assert candidate.now < 4

    # successful baseline preserves candidate result
    guarded = recipes["guard_required_baseline"](
        lambda request: {"ok": True}, lambda: None,
        lambda: pytest.fail("unexpected cancel"), lambda category: pytest.fail("unexpected error"),
    )
    assert guarded({"op": "train"}) == {"ok": True}

    # baseline failure after bounded call is not success
    failed = []
    guarded = recipes["guard_required_baseline"](
        lambda request: failed.append(True),
        lambda: "baseline_exit_signal" if failed else None,
        lambda: None,
        lambda category: None,
    )
    with pytest.raises(recipes["RequiredBaselineFailure"], match="required baseline failed"):
        guarded({"op": "train"})

    # baseline diagnostic is bounded before recording
    for category in ("x" * 81, "unsafe\nmessage", 42):
        recorded = []
        guarded = recipes["guard_required_baseline"](
            lambda request: pytest.fail("unexpected work"), lambda category=category: category,
            lambda: None, recorded.append,
        )
        with pytest.raises(recipes["RequiredBaselineFailure"]):
            guarded({"op": "train"})
        assert recorded == ["invalid_baseline_diagnostic"], category


def test_required_baseline_failure_keeps_penalty_category_and_cleanup(recipes):
    # baseline failure preserves runtime penalty separate from measured evidence
    candidate = ClockedCandidate(response_seconds=0.05)
    start = candidate.clock()
    penalty = {"score": 0.0, "runtime_seconds": 300.0}

    def evaluate(*args):
        raise recipes["RequiredBaselineFailure"]("baseline_exit")

    outcomes, complete = run_episodes(
        recipes["run_declared_episodes"], candidate, [{"seed": 3}],
        rpc=lambda request, **kwargs: (candidate.rpc(request, **kwargs), {"ok": True})[1],
        evaluate=evaluate, worst_case=lambda episode: dict(penalty),
    )
    evidence = recipes["make_evaluation_evidence"]([
        {
            "seed": 3,
            "score": outcomes[0]["score"],
            "correct": complete,
            "duration_seconds": candidate.clock() - start,
        }
    ])
    assert outcomes == [penalty]
    assert evidence["episodes"][0]["duration_seconds"] == pytest.approx(0.4)
    assert not complete

    # baseline failure is not a timeout and retains its category before cleanup
    def cleanup():
        raise RuntimeError("secondary private cleanup error")

    with pytest.raises(recipes["RequiredBaselineFailure"]) as caught:
        recipes["fail_required_baseline"]("baseline_exit", cleanup, lambda category: None)
    assert not isinstance(caught.value, TimeoutError)
    failure = recipes["describe_episode_failure"](caught.value, "cleanup")
    assert failure == {"component": "baseline", "category": "baseline_exit",
                       "operation": "baseline_worker"}
    assert "secondary private" not in str(caught.value)

    # diagnostic callback failure does not skip cleanup
    cancelled = []

    def broken_record(category):
        raise OSError("private logger failure")

    with pytest.raises(recipes["RequiredBaselineFailure"]) as caught:
        recipes["fail_required_baseline"]("baseline_exit", lambda: cancelled.append(True), broken_record)
    assert cancelled == [True]
    assert caught.value.category == "baseline_exit"


def test_collect_required_baseline_deadlines_failures_and_cleanup_errors(recipes):
    # baseline collection success uses remaining deadline
    candidate = ClockedCandidate(response_seconds=0.05)
    result_value = {"loss": 0.2}
    waits = []

    def result(*, timeout):
        waits.append(timeout)
        candidate.now += 1
        return result_value

    collected = recipes["collect_required_baseline"](
        result, deadline=5, clock=candidate.clock,
        cancel_work=lambda: pytest.fail("unexpected cancel"),
        record_failure=lambda category: pytest.fail("unexpected failure"),
    )
    assert collected is result_value
    assert waits == [3]

    # baseline collection failure after successful poll preserves resets
    for error, category in [
        (RuntimeError("private worker output"), "baseline_result_failure"),
        (TimeoutError("private worker output"), "baseline_timeout"),
    ]:
        candidate = ClockedCandidate(response_seconds=0.05)
        failures = []
        cancelled = []

        def evaluate(episode, call, deadline, candidate=candidate, error=error,
                     failures=failures, cancelled=cancelled):
            guarded = recipes["guard_required_baseline"](
                call, lambda: None, lambda: None, failures.append
            )
            guarded({"op": "train"})

            def poll_result(*, timeout):
                assert 0 < timeout <= deadline - candidate.clock()
                raise error

            return recipes["collect_required_baseline"](
                poll_result, deadline=deadline, clock=candidate.clock,
                cancel_work=lambda: cancelled.append(episode["seed"]),
                record_failure=failures.append,
            )

        outcomes, complete = run_episodes(
            recipes["run_declared_episodes"], candidate, TWO_SEEDS, evaluate=evaluate,
        )
        assert candidate.resets == cancelled == [3, 7], category
        assert failures == [category, category], category
        assert outcomes == [-1.0, -1.0], category
        assert not complete, category

    # baseline collection with expired deadline never waits
    failures = []
    cancelled = []
    with pytest.raises(recipes["RequiredBaselineFailure"], match="required baseline failed"):
        recipes["collect_required_baseline"](
            lambda **kwargs: pytest.fail("deadline expired"), deadline=2, clock=lambda: 2,
            cancel_work=lambda: cancelled.append(True), record_failure=failures.append,
        )
    assert cancelled == [True]
    assert failures == ["baseline_timeout"]

    # baseline collection rejects result after work deadline
    candidate = ClockedCandidate(response_seconds=0.05)
    failures = []

    def late_result(*, timeout):
        candidate.now += timeout + .1
        return 1.0

    with pytest.raises(recipes["RequiredBaselineFailure"], match="required baseline failed"):
        recipes["collect_required_baseline"](
            late_result, deadline=5, clock=candidate.clock, cancel_work=lambda: None,
            record_failure=failures.append,
        )
    assert failures == ["baseline_timeout"]

    # unexpected baseline collection error propagates
    def broken_result(*, timeout):
        raise ValueError("programming error")

    with pytest.raises(ValueError, match="programming error"):
        recipes["collect_required_baseline"](
            broken_result, deadline=5, clock=lambda: 2,
            cancel_work=lambda: pytest.fail("unexpected cancel"),
            record_failure=lambda category: pytest.fail("unexpected failure"),
        )

    # baseline cleanup failure preserves the primary diagnostic (poll and collect entry points)
    def cancel():
        raise RuntimeError("private cleanup output")

    def failing_result(*, timeout):
        raise RuntimeError("private worker output")

    for entry in ("poll", "collect"):
        failures = []
        with pytest.raises(recipes["RequiredBaselineFailure"], match="required baseline failed") as caught:
            if entry == "poll":
                recipes["guard_required_baseline"](
                    lambda request: pytest.fail("unexpected call"), lambda: "baseline_result_failure",
                    cancel, failures.append,
                )({"op": "train"})
            else:
                recipes["collect_required_baseline"](
                    failing_result, deadline=5, clock=lambda: 2, cancel_work=cancel,
                    record_failure=failures.append,
                )
        assert failures == ["baseline_result_failure"], entry
        assert caught.value.__notes__ == ["baseline cleanup failed"], entry
        assert caught.value.__suppress_context__, entry


# --------------------------------------------------------------------------- feedback interventions


def test_feedback_intervention_certifies_only_genuine_feedback_dependence(recipes):
    # feedback dependence needs no performance improvement
    calls = []

    def replay(*, seed, history):
        calls.append(seed)
        return [history[0]["value"]]

    assert recipes["feedback_intervention"](
        replay, seed=3, history_a=[{"value": 1}], history_b=[{"value": 2}],
    )
    assert calls == [3, 3]

    # varying open-loop sequence fails matched feedback check
    def open_loop(*, seed, history):
        # Varies across steps; identical seeded state ignores all feedback.
        return [seed + len(history)]

    assert not recipes["feedback_intervention"](
        open_loop, seed=3, history_a=[{"value": 1}], history_b=[{"value": 2}],
    )

    # numeric serialization changes are not feedback responses
    for left, right in [(1, 1.0), (0.0, -0.0)]:
        suggestions = iter(([left], [right]))
        assert not recipes["feedback_intervention"](
            lambda **kwargs: next(suggestions), seed=3,
            history_a=[{"value": 1}], history_b=[{"value": 2}],
        ), (left, right)

    # identical histories cannot certify feedback dependence
    for name, history_b in [("identical", [{"value": 1}]),
                            ("numerically-equivalent", [{"value": 1.0}])]:
        with pytest.raises(ValueError, match="distinct feedback"):
            recipes["feedback_intervention"](
                lambda **kwargs: pytest.fail("vacuous intervention"),
                seed=3, history_a=[{"value": 1}], history_b=history_b,
            )


# --------------------------------------------------------------------------- RPC and content boundaries


def test_state_transfer_rpc_budget_counts_and_validation(recipes):
    # rpc budget covers complete state transfer transcript
    for episodes, steps, expected in [(2, 1536, 3384), (6, 4096, 25512)]:
        for chunks in (1, 32):
            calls = []

            def export_state(calls=calls, chunks=chunks):
                calls.append("export_start")
                calls.extend(["export_chunk"] * chunks)

            def import_state(calls=calls, chunks=chunks):
                calls.append("import_start")
                calls.extend(["import_chunk"] * chunks)
                calls.append("import_commit")

            for _ in range(episodes):
                calls.append("reset")
                export_state()
                calls.extend(["outer_step"] * steps)
                export_state()
                import_state()
                calls.extend(["checkpoint_rollout"] * 9)
                import_state()
                calls.extend(["control_rollout"] * 12)

            counts = {"reset": 1, "outer_step": steps, "checkpoint_rollout": 9, "control_rollout": 12}
            counts.update(recipes["state_transfer_rpc_counts"](exports=2, imports=2, max_chunks=chunks))
            budget = episodes * sum(counts.values())
            assert Counter(calls) == {
                operation: episodes * count for operation, count in counts.items()
            }, (episodes, chunks)
            assert budget == len(calls), (episodes, chunks)
            if chunks == 32:
                assert budget == expected, (episodes, chunks)

    # no state transfers add no calls
    assert sum(recipes["state_transfer_rpc_counts"](exports=0, imports=0, max_chunks=1).values()) == 0

    # state transfer budget rejects invalid counts
    for exports, imports, chunks in [
        (-1, 1, 1), (1, -1, 1), (1, 1, 0), (True, 1, 1), (1, True, 1), (1, 1, True), (1, 1, 1.5),
    ]:
        with pytest.raises(ValueError):
            recipes["state_transfer_rpc_counts"](exports=exports, imports=imports, max_chunks=chunks)


def test_bounded_json_line_enforces_envelope_wire_limit_and_finite_values(recipes):
    # complete JSON envelope, not raw tensor, controls payload limit
    request = {"tensor": base64.b64encode(b"x" * 24).decode(), "shape": [24]}
    size = len(json.dumps(request, separators=(",", ":")).encode()) + 1
    assert len(recipes["bounded_json_line"](request, size + 1)) == size
    with pytest.raises(ValueError, match="wire limit"):
        recipes["bounded_json_line"](request, size)
    with pytest.raises(ValueError, match="wire limit"):
        recipes["bounded_json_line"](request, 25)

    # default wire limit is unchanged 64 MiB
    assert recipes["bounded_json_line"].__defaults__ == (MAX_RPC_BYTES,)

    # wire serialization rejects nonfinite values
    with pytest.raises(ValueError):
        recipes["bounded_json_line"]({"score": float("nan")})


def test_document_overlap_weighted_aggregation_and_required_controls(recipes):
    # distinct documents are disjoint; changed ids do not hide same normalized content
    recipes["require_disjoint_documents"](["training text"], ["held out text"])
    with pytest.raises(ValueError, match="overlap"):
        recipes["require_disjoint_documents"](["A document\nwith text"], ["A  document with text"])

    # failed controls keep original aggregation weights
    assert recipes["fixed_weight_score"]([1.0, None], [.5, .5], -1.0) == 0.0
    assert recipes["fixed_weight_score"]([1.0, 1.0], [.5, .5], -1.0) == 1.0

    # invalid aggregation is not renormalized
    for scores, weights in [([1], [.5]), ([1, 2], [1]), ([1], [-1])]:
        with pytest.raises(ValueError):
            recipes["fixed_weight_score"](scores, weights, -1.0)

    # completed required controls are accepted; a missing required control is not success
    recipes["require_controls"]({"random", "adaptive"}, {"random", "adaptive"})
    with pytest.raises(ValueError, match="required control"):
        recipes["require_controls"]({"random", "adaptive"}, {"random"})


# --------------------------------------------------------------------------- environment recipe


def test_environment_recipe_filters_parent_environment_and_passes_capability_guard(
    environment_recipe, tmp_path
):
    source, settings = environment_recipe

    # recipe retains interpreter, devices and thread limits
    parent = {
        "TASK_PYTHON": "/task/.venv/bin/python",
        "PATH": "/task/.venv/bin:/usr/bin",
        "LD_LIBRARY_PATH": "/opt/cuda/lib64",
        "CUDA_VISIBLE_DEVICES": "2",
        "NVIDIA_VISIBLE_DEVICES": "GPU-assigned",
        "OMP_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "3",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "LANG": "C.UTF-8",
    }
    interpreter, environment = settings(parent)
    assert interpreter == parent["TASK_PYTHON"]
    assert {key: environment[key] for key in parent if key != "TASK_PYTHON"} == {
        key: value for key, value in parent.items() if key != "TASK_PYTHON"
    }
    assert "TASK_PYTHON" not in environment

    # recipe discards credentials, control paths and python overrides
    excluded = {
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AGENTIC_HARNESS_TOKEN",
        "AGENTIC_CANDIDATE_TOKEN", "AGENTIC_CANDIDATE_SOCKET", "TASK_SOURCE_DIR",
        "AGENTIC_TASK_QUALIFICATION", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
        "AWS_SECRET_ACCESS_KEY", "CUDA_SECRET", "NVIDIA_AUTH_TOKEN", "OMP_TOKEN",
    }
    parent = dict.fromkeys(excluded, "must-not-be-forwarded") | {"HOME": "/private/home"}
    _, environment = settings(parent)
    assert not excluded.intersection(environment)
    assert environment["HOME"] == "/nonexistent"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert parent["HOME"] == "/private/home"

    # recipe preserves CPU device mask and defaults to current python
    interpreter, environment = settings({"CUDA_VISIBLE_DEVICES": ""})
    assert interpreter == sys.executable
    assert environment["CUDA_VISIBLE_DEVICES"] == ""

    # recipe passes capability guard without weakening reserved namespace
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text(source)
    evaluators = (("search_evaluator", "evaluate.py"),)
    assert not unavailable_harness_environment_findings(tmp_path, evaluators)

    evaluator.write_text('import os\nvalue = os.environ.get("AGENTIC_HARNESS_TOKEN")\n')
    assert unavailable_harness_environment_findings(tmp_path, evaluators)


def test_untrusted_task_text_cannot_expand_environment_recipe(tmp_path):
    rendered = render_untrusted_task(tmp_path, ENVIRONMENT_MARKER)
    assert "def candidate_subprocess_settings" not in rendered


# --------------------------------------------------------------------------- harness capability policy


def test_harness_capability_policy_rejects_only_unavailable_proxy_memory_accessors(tmp_path):
    # documentation and the supported handle op are not rejected
    for index, source in enumerate([
        '# proxy.candidate_peak_accelerator_memory_gib()\n"documentation only"',
        '"""candidate_peak_accelerator_memory_gib is unavailable."""',
        "message = 'candidate_peak_accelerator_memory_gib'",
        "response = proxy.handle({'op': 'metadata'})",
    ]):
        task_dir = tmp_path / f"accepted_{index}"
        task_dir.mkdir()
        (task_dir / "evaluate.py").write_text(source)
        assert not unavailable_harness_environment_findings(task_dir, EVALUATORS), source

    # unavailable proxy memory accessor is rejected
    for index, source in enumerate([
        "memory = proxy.candidate_peak_accelerator_memory_gib()",
        "reader = getattr(proxy, 'candidate_peak_accelerator_memory_gib', None)",
    ]):
        task_dir = tmp_path / f"rejected_{index}"
        task_dir.mkdir()
        (task_dir / "evaluate.py").write_text(source)
        findings = unavailable_harness_environment_findings(task_dir, EVALUATORS)
        assert len(findings) == 1, source
        assert "candidate_peak_accelerator_memory_gib" in findings[0].diagnostic, source
        assert "not supplied" in findings[0].diagnostic, source
