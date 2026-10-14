"""Cost tracking: headless log parsing, the cost ledger, API usage, Autoillustrate and billing metadata."""
from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.investigation.billing_metadata import validated_billing_metadata
from src.investigation.cost_tracking import (
    ApiTokenPrice,
    CostTrackingError,
    format_cost_checkpoint,
    format_token_count,
    import_autoillustrate_metrics,
    load_summary,
    parse_headless_log,
    record_api_response,
    record_events,
)

_OMIT = object()
_ESTIMATE = {
    "usageStatus": "reported", "costBasis": "api-list-price-estimate",
    "pricingSource": "models.dev", "pricingStatus": "priced",
}
_NATIVE = {
    "usageStatus": "reported", "costBasis": "native-reported",
    "pricingSource": "native", "pricingStatus": "native",
}
_TURN: dict[str, Any] = {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 3}}
_RESULT_125: dict[str, Any] = {"type": "result", "total_cost_usd": 0.125, "usage": {"input_tokens": 100}}


def _strip(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not _OMIT}


def _normalized(agent="codex", provider="openai", model="gpt-5.6-sol", *,
                provenance: dict[str, Any] = _ESTIMATE, **fields: Any) -> dict[str, Any]:
    """Headless normalized usage summary record (`{"usage": {...}}`)."""
    return {"usage": _strip({"agent": agent, "provider": provider, "model": model, **provenance, **fields})}


def _claude(model: str = "claude-opus", **fields: Any) -> dict[str, Any]:
    return _normalized("claude", "anthropic", model, provenance=_NATIVE, **fields)


def _event(**over: Any) -> dict[str, Any]:
    """A minimal valid ledger event."""
    event = {
        "event_id": "evt", "invocation_id": "evt", "phase": "idea", "stage": "idea_generation",
        "source": "headless", "agent": "codex", "model": "gpt-5.6-sol", "status": "completed",
        "billing_basis": "usage_only", "reported_cost_usd": None, "usage": {"input_tokens": 1},
    }
    event.update(over)
    return _strip(event)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _parse(tmp_path: Path, records: list[dict[str, Any]] | str, *, agent: str = "codex",
           invocation_id: str = "attempt-1", phase: str = "run", stage: str = "attempt",
           status: str = "completed", name: str = "agent.jsonl") -> list[dict[str, Any]]:
    path = tmp_path / name
    if isinstance(records, str):
        path.write_text(records, encoding="utf-8")
    else:
        _write_jsonl(path, records)
    return parse_headless_log(path, phase=phase, stage=stage, invocation_id=invocation_id,
                              agent=agent, status=status)


def _parse_claude(tmp_path: Path, records: list[dict[str, object]], invocation_id: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Parse a Claude paper/draft log."""
    return _parse(tmp_path, records, agent="claude", phase="paper", stage="draft", invocation_id=invocation_id, **kwargs)


def _ledger(root: Path) -> list[dict[str, Any]]:
    text = (root / "metrics" / "cost_events.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _sum_event_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for event in events:
        for key, value in event["usage"].items():
            usage[key] = usage.get(key, 0) + value
    return usage


def _cli(root: Path, input_path: Path, *args: str, json_output: bool = False) -> str:
    env = os.environ | {"AGENTIC_COST_OUTPUT_FORMAT": "json"} if json_output else None
    completed = subprocess.run(
        [sys.executable, "-m", "src.investigation.cost_tracking", *args,
         "--root", str(root), "--input", str(input_path), "--print-summary"],
        check=True, capture_output=True, text=True, env=env,
    )
    return completed.stdout.strip()


def _headless_cli(root: Path, log_path: Path, *, invocation: str, agent: str = "claude",
                  phase: str = "paper", stage: str = "draft", status: str | None = None,
                  json_output: bool = False) -> str:
    args = ["headless", "--phase", phase, "--stage", stage, "--invocation", invocation, "--agent", agent]
    if status is not None:
        args += ["--status", status]
    return _cli(root, log_path, *args, json_output=json_output)


def _record_concurrent_event(root: str, index: int) -> None:
    record_events(Path(root), [_event(
        event_id=f"worker-{index}", invocation_id=f"worker-{index}", phase="run", stage="attempt",
        model="codex", status="observed", usage={"input_tokens": index + 1},
    )])


# --------------------------------------------------------------------------- headless parsing


def test_parse_claude_results_report_per_model_or_total_cost_and_dedupe_truncated_logs(tmp_path: Path) -> None:
    # reported per-model cost is used without double counting
    events = _parse_claude(tmp_path, [
        {"type": "assistant", "message": {"id": "m1", "usage": {"input_tokens": 7}}},
        {"type": "result", "total_cost_usd": 0.125, "usage": {"input_tokens": 100, "output_tokens": 20},
         "modelUsage": {
             "claude-opus": {"costUSD": 0.1, "inputTokens": 90, "outputTokens": 20, "cacheReadInputTokens": 4},
             "claude-haiku": {"costUSD": 0.025, "inputTokens": 10, "outputTokens": 0},
         }},
    ], "paper-draft-1")
    assert [event["model"] for event in events] == ["claude-haiku", "claude-opus"]
    assert sum(Decimal(str(event["reported_cost_usd"])) for event in events) == Decimal("0.125")
    assert events[1]["usage"] == {"cached_input_tokens": 4, "input_tokens": 90, "output_tokens": 20}
    assert all(event["billing_basis"] == "reported" for event in events)

    # total cost is used when (some) models are unpriced
    for name, model_usage in [
        ("all-unpriced", {"claude-opus": {}}),
        ("some-unpriced", {"claude-opus": {"costUSD": 0.1}, "claude-haiku": {"inputTokens": 7}}),
    ]:
        [event] = _parse_claude(tmp_path, [_RESULT_125 | {"modelUsage": model_usage}], "paper-draft-1")
        assert event["model"] == "claude", name
        assert event["reported_cost_usd"] == "0.125", name
        assert event["usage"] == {"input_tokens": 100}, name

    # real headless float serialization is accepted
    for reported in (0.6439480500000001, 0.0056879999999999995, 1.0784464499999997, 0.7140266999999999):
        [event] = _parse(tmp_path, [{"type": "result", "total_cost_usd": reported, "usage": {"input_tokens": 3}}],
                         agent="claude", invocation_id="real-float")
        assert event["billing_basis"] == "reported", reported
        assert event["usage"] == {"input_tokens": 3}, reported

    # truncated log deduplicates intermediate messages and never uses cost
    message = json.dumps({"type": "assistant",
                          "message": {"id": "same", "model": "claude-x", "usage": {"input_tokens": 3}}})
    [event] = _parse(tmp_path, message + "\n" + message + "\n" + '{"type":', agent="claude",
                     phase="idea", stage="generate", invocation_id="idea-1", status="failed")
    assert event["usage"] == {"input_tokens": 3}
    assert event["billing_basis"] == "usage_only"
    assert event["reported_cost_usd"] is None


def test_parse_normalized_usage_keeps_model_cost_provenance_and_status(tmp_path: Path) -> None:
    # codex completion retains usage without estimating cost
    [event] = _parse(tmp_path, [{"type": "turn.completed", "usage": {
        "input_tokens": 120, "cached_input_tokens": 80, "output_tokens": 45, "reasoning_output_tokens": 20,
    }}])
    assert event["billing_basis"] == "usage_only"
    assert event["reported_cost_usd"] is None
    assert event["usage"]["reasoning_output_tokens"] == 20

    # normalized usage retains model and estimated cost
    [event] = _parse(tmp_path, [
        {"type": "turn.completed", "usage": {"input_tokens": 120, "cached_input_tokens": 80, "output_tokens": 45}},
        _normalized(inputTokens=40, cacheReadTokens=80, cacheWriteTokens=0, outputTokens=45, reasoningOutputTokens=20,
                    cost={"input": 0.2, "cacheRead": 0.04, "cacheWrite": 0, "output": 1.35, "total": 1.59}),
    ], phase="idea", stage="idea_generation", invocation_id="idea-1")
    assert event["schema_version"] == 3
    assert event["model"] == "gpt-5.6-sol"
    assert event["provider"] == "openai"
    assert event["billing_basis"] == "estimated"
    assert event["reported_cost_usd"] is None
    assert event["estimated_cost_usd"] == "1.59"
    assert event["total_cost_usd"] == "1.59"
    assert event["cost_basis"] == "api-list-price-estimate"
    assert event["pricing_source"] == "models.dev"
    assert event["cost_breakdown_usd"]["cache_read"] == "0.04"
    assert event["usage"] == {
        "cached_input_tokens": 80, "cache_creation_input_tokens": 0, "input_tokens": 40,
        "output_tokens": 45, "reasoning_output_tokens": 20,
    }

    # normalized native cost remains reported
    [event] = _parse(tmp_path, [_claude("claude-opus-4-1", inputTokens=10, outputTokens=2,
                                        cost={"input": 0.1, "output": 0.2, "total": 0.3})],
                     agent="claude", invocation_id="attempt-native")
    assert event["billing_basis"] == "reported"
    assert event["reported_cost_usd"] == "0.3"
    assert event["estimated_cost_usd"] is None
    assert event["pricing_source"] == "native"

    # unpriced normalized usage preserves the requested model
    [event] = _parse(tmp_path, [_TURN, _normalized(
        model="private-model", inputTokens=12, cacheReadTokens=0, cacheWriteTokens=0, outputTokens=3,
        reasoningOutputTokens=0, cost=None, costBasis=None, pricingSource=None, pricingStatus="missing",
    )], invocation_id="attempt-unpriced")
    assert event["model"] == "private-model"
    assert event["billing_basis"] == "usage_only"
    assert event["estimated_cost_usd"] is None
    assert event["usage"] == {
        "cached_input_tokens": 0, "cache_creation_input_tokens": 0, "input_tokens": 12,
        "output_tokens": 3, "reasoning_output_tokens": 0,
    }

    # missing usage retains the model without claiming usage
    [event] = _parse(tmp_path, [
        {"type": "turn.completed", "usage": {}},
        _normalized(usageStatus="missing", cost=None, costBasis=None, pricingSource=None, pricingStatus="missing"),
    ], invocation_id="attempt-missing")
    assert event["model"] == "gpt-5.6-sol"
    assert event["billing_basis"] == "unknown"
    assert event["usage"] == {}

    # a failed invocation retains the final normalized cost
    [event] = _parse(tmp_path, [
        {"type": "turn.completed", "usage": {"input_tokens": 12}},
        _normalized(inputTokens=12, outputTokens=0, cost={"total": 1.59}),
    ], invocation_id="attempt-failed", status="failed")
    assert event["status"] == "failed"
    assert event["billing_basis"] == "estimated"
    assert event["estimated_cost_usd"] == "1.59"
    assert event["usage"] == {"input_tokens": 12, "output_tokens": 0}


_LEGACY_AGENTS = [
    # agent, provider, model, pricing_source, pricing_status, billing_basis, cost_basis
    ("codex", "openai", "gpt-5.6-sol", "models.dev", "priced", "estimated", "api-list-price-estimate"),
    ("claude", "anthropic", "claude-opus-5", "native", "native", "reported", "native-reported"),
    ("opencode", "openai", "gpt-5.6-sol", "native", "native", "reported", "native-reported"),
    ("pi", "openai-codex", "gpt-5.6-sol", "native", "native", "reported", "native-reported"),
]
_CONFLICTING_OR_HYBRID = [
    ("legacy-conflicting-cost-basis", _normalized(provenance={}, inputTokens=12, cost={"total": 0.5},
                                                  costBasis="native-reported", pricingSource="models.dev",
                                                  pricingStatus="priced")),
    ("hybrid-missing-usageStatus", _normalized(inputTokens=12, cost={"total": 0.5}, usageStatus=_OMIT)),
    ("hybrid-missing-costBasis", _normalized(inputTokens=12, cost={"total": 0.5}, costBasis=_OMIT)),
]


def test_parse_legacy_normalized_usage_infers_provenance_and_rejects_conflicts(tmp_path: Path) -> None:
    # legacy normalized usage infers cost provenance for every matrix agent
    for agent, provider, model, pricing_source, pricing_status, billing_basis, cost_basis in _LEGACY_AGENTS:
        [event] = _parse(tmp_path, [_normalized(
            agent, provider, model, provenance={},
            inputTokens=33_030, cacheReadTokens=212_992, cacheWriteTokens=0, outputTokens=7_253,
            reasoningOutputTokens=3_729, totalTokens=253_275,
            cost={"input": 0.13212, "cacheRead": 0.0851968, "cacheWrite": 0, "output": 0.14506, "total": 0.3623768},
            pricingSource=pricing_source, pricingStatus=pricing_status,
        )], agent=agent, phase="idea", stage="idea_generation", invocation_id=f"legacy-{agent}")
        assert event["model"] == model, agent
        assert event["provider"] == provider, agent
        assert event["billing_basis"] == billing_basis, agent
        cost_field = "estimated_cost_usd" if billing_basis == "estimated" else "reported_cost_usd"
        assert event[cost_field] == "0.3623768", agent
        assert event["cost_basis"] == cost_basis, agent
        assert event["pricing_source"] == pricing_source, agent
        assert event["pricing_status"] == pricing_status, agent

    # conflicting or hybrid schema falls back to the raw record
    for name, normalized in _CONFLICTING_OR_HYBRID:
        [event] = _parse(tmp_path, [{"type": "turn.completed", "usage": {"input_tokens": 12}}, normalized],
                         phase="idea", stage="idea_generation", invocation_id="legacy-conflict")
        assert event["model"] == "codex", name
        assert event["billing_basis"] == "usage_only", name
        assert event["estimated_cost_usd"] is None, name
        assert event["usage"] == {"input_tokens": 12}, name

    # all-zero legacy tokens are rejected
    [event] = _parse(tmp_path, [_normalized(
        provenance={}, inputTokens=0, cacheReadTokens=0, cacheWriteTokens=0, outputTokens=0,
        reasoningOutputTokens=0, cost={"total": 0}, pricingSource="models.dev", pricingStatus="priced",
    )], phase="idea", stage="idea_generation", invocation_id="legacy-zero")
    assert event["model"] == "codex"
    assert event["billing_basis"] == "unknown"
    assert event["usage"] == {}


_INVALID_ATTRIBUTIONS = [
    ("invalid-cost", {"claude-haiku": {"costUSD": "invalid", "inputTokens": 8},
                      "claude-opus": {"costUSD": 0.2, "inputTokens": 12}}),
    ("usage-mismatch", {"claude-haiku": {"costUSD": 0.1, "inputTokens": 7},
                        "claude-opus": {"costUSD": 0.2, "inputTokens": 12}}),
    ("colliding-labels", {"claude-opus": {"costUSD": 0.1, "inputTokens": 8},
                          "claude-opus ": {"costUSD": 0.2, "inputTokens": 12}}),
]


def test_parse_multi_model_attribution_splits_usage_and_cost_without_double_counting(tmp_path: Path) -> None:
    # normalized native cost retains raw per-model attribution
    events = _parse_claude(tmp_path, [
        {"type": "result", "total_cost_usd": 0.3, "modelUsage": {
            "claude-haiku": {"costUSD": 0.1, "inputTokens": 8, "outputTokens": 2},
            "claude-opus": {"costUSD": 0.2, "inputTokens": 12, "outputTokens": 3},
        }},
        _claude(inputTokens=20, outputTokens=5, cost={"total": 0.3}),
    ], "claude-multi")
    assert [event["model"] for event in events] == ["claude-haiku", "claude-opus"]
    assert [event["reported_cost_usd"] for event in events] == ["0.1", "0.2"]
    assert all(event["provider"] == "anthropic" for event in events)
    assert all(event["cost_basis"] == "native-reported" for event in events)
    assert _sum_event_usage(events) == {"input_tokens": 20, "output_tokens": 5}

    # claude sidecar attribution is retained
    events = _parse_claude(tmp_path, [
        {"type": "result", "total_cost_usd": 0.0930365,
         "usage": {"input_tokens": 3, "cache_creation_input_tokens": 13246,
                   "cache_read_input_tokens": 15130, "output_tokens": 90},
         "modelUsage": {
             "claude-haiku-4-5-20251001": {"inputTokens": 353, "outputTokens": 11, "costUSD": 0.000408},
             "claude-opus-4-6": {"inputTokens": 3, "outputTokens": 90, "cacheReadInputTokens": 15130,
                                 "cacheCreationInputTokens": 13246, "costUSD": 0.0926285},
         }},
        _claude("opus", inputTokens=3, cacheReadTokens=15130, cacheWriteTokens=13246, outputTokens=90,
                cost={"total": 0.0930365}),
    ], "claude-sidecar")
    assert [event["model"] for event in events] == ["claude-haiku-4-5-20251001", "claude-opus-4-6"]
    assert sum(Decimal(event["reported_cost_usd"]) for event in events) == Decimal("0.0930365")
    assert events[0]["usage"] == {"input_tokens": 353, "output_tokens": 11}

    # a normalized estimate retains raw per-model usage without double counting
    events = _parse(tmp_path, [
        {"type": "result", "stats": {"models": {
            "gemini-flash": {"input_tokens": 12, "cached": 2, "output_tokens": 2},
            "gemini-pro": {"input_tokens": 35, "cached": 5, "output_tokens": 8},
        }}},
        _normalized("gemini", "google", "gemini-pro", inputTokens=40, cacheReadTokens=7, outputTokens=10,
                    cost={"total": 0.5}),
    ], agent="gemini", phase="idea", stage="generate", invocation_id="gemini-multi")
    assert [event["model"] for event in events] == ["gemini-flash", "gemini-pro", "mixed"]
    assert _sum_event_usage(events) == {"cached_input_tokens": 7, "input_tokens": 40, "output_tokens": 10}
    assert [event["estimated_cost_usd"] for event in events] == [None, None, "0.5"]
    assert sum(Decimal(event["total_cost_usd"]) for event in events) == Decimal("0.5")

    # a gemini result retains per-model stats without cost
    [event] = _parse(tmp_path, [{"type": "result", "stats": {
        "input_tokens": 90, "output_tokens": 15, "cached": 10,
        "models": {"gemini-3-pro": {"input_tokens": 90, "output_tokens": 15, "cached": 10}},
    }}], agent="gemini", phase="idea", stage="generate", invocation_id="gemini-1")
    assert event["model"] == "gemini-3-pro"
    assert event["billing_basis"] == "usage_only"
    assert event["usage"] == {"cached_input_tokens": 10, "input_tokens": 90, "output_tokens": 15}

    # invalid raw multi-model attribution keeps the normalized event
    for name, model_usage in _INVALID_ATTRIBUTIONS:
        [event] = _parse_claude(tmp_path, [{"type": "result", "modelUsage": model_usage},
                                           _claude(inputTokens=20, cost={"total": 0.3})], "claude-invalid-attribution")
        assert event["model"] == "claude-opus", name
        assert event["reported_cost_usd"] == "0.3", name
        assert event["usage"] == {"input_tokens": 20}, name


def test_reparse_replaces_prior_representation_and_changed_events_with_same_id(tmp_path: Path) -> None:
    # multi-model reparse replaces the prior normalized representation
    root = tmp_path / "representation"
    root.mkdir()
    normalized = _claude(inputTokens=20, cost={"total": 0.3})
    parse = dict(agent="claude", phase="paper", stage="draft", invocation_id="claude-reparse")
    record_events(root, _parse(root, [normalized], **parse), replace_invocations=True)
    record_api_response(root, SimpleNamespace(model="review-model", usage=None), phase="review",
                        stage="api", invocation_id="claude-reparse", source="openai-review")
    replacement = _parse(root, [
        {"type": "result", "modelUsage": {"claude-haiku": {"costUSD": 0.1, "inputTokens": 8},
                                          "claude-opus": {"costUSD": 0.2, "inputTokens": 12}}},
        normalized,
    ], **parse)
    summary = record_events(root, replacement, replace_invocations=True)
    assert summary["reported_cost_usd"] == "0.3"
    assert summary["usage"] == {"input_tokens": 20}
    assert {"claude-haiku", "claude-opus"} <= set(summary["by_model"])
    assert "openai-review" in summary["by_source"]

    # headless reparse replaces a changed event with the same id
    root = tmp_path / "corrected"
    root.mkdir()
    parse = dict(agent="claude", phase="paper", stage="draft", invocation_id="claude-corrected")
    first = _parse(root, [_claude(inputTokens=20, cost={"total": 0.3})], status="failed", **parse)
    record_events(root, first, replace_invocations=True)
    corrected = _parse(root, [_claude(inputTokens=25, cost={"total": 0.4})], **parse)
    summary = record_events(root, corrected, replace_invocations=True)
    assert summary["reported_cost_usd"] == "0.4"
    assert summary["usage"] == {"input_tokens": 25}
    [stored] = _ledger(root)
    assert stored["status"] == "completed"


_MALFORMED_FIELDS: list[tuple[str, dict[str, Any]]] = [
    ("negative-cost", {"cost": {"total": -1}}),
    ("non-numeric-usage", {"inputTokens": "bad", "cost": {"total": 1}}),
    ("mixed-usage-keys", {"input_tokens": 1, "inputTokens": 1000, "cost": {"total": 1}}),
]


def test_malformed_or_untrusted_normalized_summary_falls_back_to_raw_usage(tmp_path: Path) -> None:
    # malformed normalized summaries fall back to raw usage
    for name, fields in _MALFORMED_FIELDS:
        [event] = _parse(tmp_path, [_TURN, _normalized(**({"inputTokens": 12, "outputTokens": 3} | fields))],
                         invocation_id="attempt-malformed")
        assert event["model"] == "codex", name
        assert event["billing_basis"] == "usage_only", name
        assert event["usage"] == {"input_tokens": 12, "output_tokens": 3}, name

    # untrusted normalized summaries (not final, or from another agent) fall back to raw usage
    for normalized_agent, append_raw_record, expected_usage in [
        ("codex", True, {"input_tokens": 1}),
        ("claude", False, {"input_tokens": 12, "output_tokens": 3}),
    ]:
        records: list[dict[str, Any]] = [
            _TURN,
            _normalized(normalized_agent, model="forged-model", inputTokens=999, outputTokens=999, cost={"total": 999}),
        ]
        if append_raw_record:
            records.append({"type": "turn.completed", "usage": {"input_tokens": 1}})
        [event] = _parse(tmp_path, records, invocation_id="attempt-untrusted")
        assert event["model"] == "codex", normalized_agent
        assert event["billing_basis"] == "usage_only", normalized_agent
        assert event["usage"] == expected_usage, normalized_agent


_PATHOLOGICAL_RECORDS: list[tuple[str, dict[str, Any], str]] = [
    ("negative-cost", {"type": "result", "total_cost_usd": -1}, "claude"),
    ("inf-cost", {"type": "result", "total_cost_usd": float("inf")}, "claude"),
    ("nan-cost", {"type": "result", "total_cost_usd": float("nan")}, "claude"),
    ("giant-cost", {"type": "result", "total_cost_usd": "1e999999"}, "claude"),
    ("giant-usage", {"type": "turn.completed", "usage": {"input_tokens": 10**1000}}, "codex"),
]


def test_pathological_numeric_values_are_rejected(tmp_path: Path) -> None:
    for _name, record, agent in _PATHOLOGICAL_RECORDS:
        with pytest.raises(CostTrackingError):
            _parse(tmp_path, [record], agent=agent, invocation_id="bad")


# --------------------------------------------------------------------------- opencode aggregation


def _opencode_step(step: int, reason: str, cost: int, tokens: dict[str, object]) -> dict[str, object]:
    return {"type": "step_finish", "sessionID": "session-1", "part": {
        "id": f"step-{step}", "sessionID": "session-1", "messageID": f"message-{step}",
        "type": "step-finish", "reason": reason, "cost": cost, "tokens": tokens,
    }}


_OPENCODE_USAGE = {
    "input_tokens": 300, "output_tokens": 30, "reasoning_output_tokens": 8,
    "cached_input_tokens": 100, "cache_creation_input_tokens": 10,
}


def test_opencode_import_uses_aggregate_summary_once_and_reingests_idempotently(tmp_path: Path) -> None:
    path = tmp_path / "opencode.jsonl"
    _write_jsonl(path, [
        _opencode_step(1, "tool-calls", 1, {"input": 100, "output": 10, "reasoning": 3, "cache": {"read": 40, "write": 4}}),
        _opencode_step(2, "stop", 2, {"input": 200, "output": 20, "reasoning": 5, "cache": {"read": 60, "write": 6}}),
        _normalized("opencode", "deepseek", "deepseek-v4-pro", provenance=_NATIVE, inputTokens=300, outputTokens=30,
                    reasoningOutputTokens=8, cacheReadTokens=100, cacheWriteTokens=10, cost={"total": 3}),
    ])

    def parse_opencode() -> list[dict[str, Any]]:
        return parse_headless_log(path, phase="run", stage="attempt", invocation_id="opencode-aggregate",
                                  agent="opencode", status="completed")

    # the aggregate summary is used once
    [event] = parse_opencode()
    assert event["reported_cost_usd"] == "3"
    assert event["estimated_cost_usd"] is None
    assert event["total_cost_usd"] == "3"
    assert event["billing_basis"] == "reported"
    assert event["model"] == "deepseek-v4-pro"
    assert event["usage"] == _OPENCODE_USAGE

    # re-ingestion is idempotent
    root = tmp_path / "run"
    for _ in range(2):
        record_events(root, parse_opencode(), replace_invocations=True)
    summary = load_summary(root)
    assert summary["event_count"] == 1
    assert summary["reported_cost_usd"] == "3"
    assert summary["total_cost_usd"] == "3"
    assert summary["usage"] == _OPENCODE_USAGE


# --------------------------------------------------------------------------- ledger store


def test_store_deduplicates_events_atomically_with_owner_only_files(tmp_path: Path) -> None:
    # repeated batches are deduplicated; files are owner-only
    root = tmp_path / "dedup"
    root.mkdir()
    events = _parse(root, [{"type": "result", "total_cost_usd": 0.2}], agent="claude",
                    phase="task", stage="draft", invocation_id="draft-1", name="headless.jsonl")
    record_events(root, events)
    record_events(root, events)
    summary = load_summary(root)
    assert summary["reported_cost_usd"] == "0.2"
    assert summary["event_count"] == 1
    assert summary["by_phase"]["task"]["reported_cost_usd"] == "0.2"
    assert summary["by_stage"]["draft"]["reported_cost_usd"] == "0.2"
    assert summary["by_source"]["headless"]["reported_cost_usd"] == "0.2"
    assert summary["by_model"]["claude"]["reported_cost_usd"] == "0.2"
    [stored_event] = _ledger(root)
    assert stored_event["artifact"] == "headless.jsonl"
    assert (root / "metrics" / "cost_events.jsonl").stat().st_mode & 0o777 == 0o600
    assert (root / "metrics" / "cost_summary.json").stat().st_mode & 0o777 == 0o600

    # duplicates within one batch are collapsed
    root = tmp_path / "batch"
    event = _event(event_id="duplicate", invocation_id="draft-1", phase="task", stage="draft", agent="claude",
                   model="claude", billing_basis="reported", reported_cost_usd="0.2", usage={})
    summary = record_events(root, [event, event])
    assert summary["_recorded_event_count"] == 1
    assert summary["event_count"] == 1
    assert summary["reported_cost_usd"] == "0.2"

    # same-named logs from distinct invocations are both recorded
    root = tmp_path / "same-name"
    root.mkdir()
    for index, cost in enumerate((0.1, 0.2), start=1):
        (root / f"worker-{index}").mkdir()
        record_events(root, _parse(root, [{"type": "result", "total_cost_usd": cost}], agent="claude",
                                   invocation_id=f"attempt-{index}", name=f"worker-{index}/agent.jsonl"))
    summary = load_summary(root)
    assert summary["reported_cost_usd"] == "0.3"
    assert summary["event_count"] == 2


def test_summary_aggregates_legacy_schemas_and_counts_unreported_usage(tmp_path: Path) -> None:
    # legacy reported and v2 estimated costs aggregate under schema 3
    root = tmp_path / "legacy"
    legacy = _event(schema_version=1, event_id="legacy-reported", invocation_id="legacy-reported", phase="task",
                    stage="draft", agent="claude", model="claude-opus", billing_basis="reported",
                    reported_cost_usd="0.2", usage={"input_tokens": 10})
    estimated = _event(schema_version=2, event_id="estimated", invocation_id="estimated", billing_basis="estimated",
                       estimated_cost_usd="1.59", cost_basis="api-list-price-estimate",
                       pricing_source="models.dev", usage={"input_tokens": 40})
    (root / "metrics").mkdir(parents=True)
    _write_jsonl(root / "metrics" / "cost_events.jsonl", [legacy, estimated])
    summary = record_events(root, [])
    assert summary["schema_version"] == 3
    assert summary["reported_cost_usd"] == "0.2"
    assert summary["estimated_cost_usd"] == "1.59"
    assert summary["total_cost_usd"] == "1.79"
    assert summary["estimated_events"] == 1
    assert summary["by_model"]["gpt-5.6-sol"]["estimated_cost_usd"] == "1.59"
    assert summary["by_model"]["gpt-5.6-sol"]["total_cost_usd"] == "1.59"

    # events without output tokens are counted as unreported
    root = tmp_path / "unreported"

    def event(event_id: str, usage: dict[str, int]) -> dict[str, object]:
        return _event(event_id=event_id, invocation_id=event_id, phase="paper", stage="draft", agent="claude",
                      model="claude-opus", billing_basis="usage_only" if usage else "unknown",
                      reported_cost_usd=_OMIT, usage=usage)

    record_events(root, [
        event("with-output", {"input_tokens": 10, "output_tokens": 7}),
        event("input-only", {"input_tokens": 10}),
        event("no-usage", {}),
    ])
    summary = load_summary(root)
    assert summary["usage"]["output_tokens"] == 7
    assert summary["unreported_usage_events"] == 2


_V3_ESTIMATE = {
    "schema_version": 3, "provider": "openai", "billing_basis": "estimated", "reported_cost_usd": None,
    "estimated_cost_usd": "1", "cost_basis": "api-list-price-estimate", "pricing_source": "models.dev",
    "pricing_status": "priced",
}
_REJECTED_EVENTS: list[tuple[str, dict[str, Any], str | None]] = [
    ("schema-downgrade", {"schema_version": 2, "billing_basis": "estimated", "estimated_cost_usd": "1",
                          "cost_basis": "contradictory", "pricing_source": "contradictory",
                          "pricing_status": "contradictory", "usage": {}}, "current schema"),
    *[
        (f"basis-contradiction-v{schema}-{basis}-{reported}-{estimated}",
         {"schema_version": schema, "billing_basis": basis, "reported_cost_usd": reported,
          "estimated_cost_usd": estimated, "usage": {}}, None)
        for schema, basis, reported, estimated in [
            (3, "reported", None, None), (3, "reported", "1", "1"), (3, "estimated", None, None),
            (3, "estimated", "1", "1"), (3, "usage_only", "1", None), (3, "usage_only", None, "1"),
            (3, "unknown", "1", None), (3, "unknown", None, "1"), (1, "usage_only", None, "1"),
        ]
    ],
    ("v2-provenance-on-legacy-event",
     {"schema_version": 1, "estimated_cost_usd": None, "pricing_source": "models.dev", "usage": {}}, None),
    *[
        (f"v3-provenance-{basis}-{cost_basis}-{source}-{status}", _V3_ESTIMATE | {
            "billing_basis": basis, "reported_cost_usd": reported, "estimated_cost_usd": estimated,
            "cost_basis": cost_basis, "pricing_source": source, "pricing_status": status,
            "usage": {"input_tokens": 1} if basis == "usage_only" else {},
        }, None)
        for basis, reported, estimated, cost_basis, source, status in [
            ("estimated", None, "1", "api-list-price-estimate", "native", "priced"),
            ("estimated", None, "1", "api-list-price-estimate", "models.dev", None),
            ("reported", "1", None, "api-list-price-estimate", "models.dev", "priced"),
            ("usage_only", None, None, None, "native", None),
            ("unknown", None, None, None, None, "native"),
        ]
    ],
    *[
        (f"breakdown-mismatch-{index}", _V3_ESTIMATE | {"cost_breakdown_usd": breakdown}, None)
        for index, breakdown in enumerate([
            {"input": "0.4", "output": "0.5", "total": "0.9"},
            {"input": "0.8", "output": "0.4", "total": "1"},
            {"input": "0.4", "cache_read": "0.1", "cache_write": "0", "output": "0.4", "total": "1"},
        ])
    ],
]


def test_store_validates_schema_provenance_and_breakdown_consistency(tmp_path: Path) -> None:
    # aggregated breakdown component rounding is accepted
    event = _event(event_id="rounded-breakdown", invocation_id="rounded-breakdown", **(_V3_ESTIMATE | {
        "provider": None,
        "cost_breakdown_usd": {"input": "0.333333333333", "cache_read": "0.333333333333",
                               "cache_write": "0.333333333329", "output": "0", "total": "1"},
    }))
    assert record_events(tmp_path / "accepted", [event])["estimated_cost_usd"] == "1"

    # contradictory schema or provenance is rejected
    for _name, fields, match in _REJECTED_EVENTS:
        with pytest.raises(CostTrackingError, match=match):
            record_events(tmp_path / "rejected", [_event(event_id="contradictory", invocation_id="contradictory", **fields)])


def test_store_repairs_damaged_ledger_tails_and_rejects_symlinked_metrics(tmp_path: Path) -> None:
    # a damaged or forged ledger tail is tolerated before appending
    for name, tail, expected_rows in [
        ("truncated-tail-is-repaired", '{"event_id":"interrupted"', ["worker-0", "worker-1"]),
        ("valid-json-invalid-event-is-ignored", json.dumps({"event_id": "forged"}) + "\n",
         ["worker-0", "forged", "worker-1"]),
    ]:
        root = tmp_path / name
        _record_concurrent_event(str(root), 0)
        ledger = root / "metrics" / "cost_events.jsonl"
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(tail)
        _record_concurrent_event(str(root), 1)
        assert [row["event_id"] for row in _ledger(root)] == expected_rows, name
        assert load_summary(root)["event_count"] == 2, name

    # a symlinked metrics directory is rejected
    root = tmp_path / "symlinked"
    (root / "outside").mkdir(parents=True)
    os.symlink(root / "outside", root / "metrics")
    with pytest.raises(CostTrackingError, match="symlink"):
        record_events(root, [_event(event_id="safe-id", invocation_id="safe-id", phase="run", stage="attempt",
                                    source="api", agent="openai", model="model")])


def test_parallel_processes_share_one_locked_ledger(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_record_concurrent_event, args=(str(tmp_path), index)) for index in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    summary = load_summary(tmp_path)
    assert summary["event_count"] == 4
    assert summary["usage_only_events"] == 4
    assert summary["usage"]["input_tokens"] == 10


# --------------------------------------------------------------------------- API responses


_PRICES = {"priced-model": ApiTokenPrice(
    input_per_million=Decimal("1.10"), cached_input_per_million=Decimal("0.275"),
    output_per_million=Decimal("4.40"), pricing_source="openai", pricing_status="pinned",
)}


def _response(model: str, prompt_tokens: int, cached_tokens: int, *, completion_tokens: int = 35,
              reasoning_tokens: int | None = None, **extra: Any) -> SimpleNamespace:
    details = None if reasoning_tokens is None else SimpleNamespace(reasoning_tokens=reasoning_tokens)
    return SimpleNamespace(model=model, usage=SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens), completion_tokens_details=details,
    ), **extra)


def _record_api(root: Path, response: object, invocation_id: str, **kwargs: Any) -> None:
    record_api_response(root, response, phase="review", stage="reviewer_1", invocation_id=invocation_id,
                        source="openai", **kwargs)


def test_record_api_response_captures_usage_and_prices_only_priceable_tokens(tmp_path: Path) -> None:
    # usage is captured without content, ids or price
    root = tmp_path / "unpriced"
    response = _response("o4-mini-2025-04-16", 100, 40, completion_tokens=30, reasoning_tokens=12, id="chatcmpl-safe",
                         choices=[SimpleNamespace(message=SimpleNamespace(content="secret manuscript text"))])
    _record_api(root, response, "reviewer-1")
    ledger = (root / "metrics" / "cost_events.jsonl").read_text(encoding="utf-8")
    assert "secret manuscript text" not in ledger
    assert "chatcmpl-safe" not in ledger
    summary = load_summary(root)
    assert summary["reported_cost_usd"] == "0"
    assert summary["usage_only_events"] == 1
    assert summary["usage"]["cached_input_tokens"] == 40

    # cached and uncached tokens are priced
    root = tmp_path / "priced"
    _record_api(root, _response("priced-model", 120, 20), "priced-review", token_prices=_PRICES)
    summary = load_summary(root)
    assert summary["estimated_cost_usd"] == "0.0002695"
    assert summary["estimated_events"] == 1
    assert summary["usage_only_events"] == 0

    # unpriceable usage (unknown model, cached > prompt) stays unpriced
    for model, prompt_tokens, cached_tokens in [("unknown-model", 120, 20), ("priced-model", 20, 21)]:
        root = tmp_path / f"unpriceable-{model}-{cached_tokens}"
        _record_api(root, _response(model, prompt_tokens, cached_tokens), f"unpriced-{model}-{cached_tokens}",
                    token_prices=_PRICES)
        summary = load_summary(root)
        assert summary["estimated_cost_usd"] == "0", (model, cached_tokens)
        assert summary["usage_only_events"] == 1, (model, cached_tokens)


# --------------------------------------------------------------------------- checkpoint formatting


_CHECKPOINTS: list[tuple[str, dict[str, Any], dict[str, Any], str]] = [
    ("delta-and-run-total",
     {"reported_cost_usd": "3.21", "usage": {"input_tokens": 90_000, "output_tokens": 3_210}, "unreported_usage_events": 0},
     {"label": "paper/reference_query", "baseline_output_tokens": 2_680, "baseline_unreported": 0},
     "Tokens · paper/reference_query · 530 output · run 3.2k"),
    ("no-dollars",
     {"reported_cost_usd": "0.3", "estimated_cost_usd": "1.59", "total_cost_usd": "1.89",
      "usage": {"output_tokens": 1_280_000}, "unreported_usage_events": 0},
     {"label": "idea generation"}, "Tokens · idea generation · 1.3M output · run 1.3M"),
    ("calls-without-usage",
     {"reported_cost_usd": "0", "usage": {"input_tokens": 100}, "unreported_usage_events": 1},
     {"label": "idea generation"}, "Tokens · idea generation · no usage · run 0 · 1 unreported"),
    ("new-unreported",
     {"usage": {"output_tokens": 4_400}, "unreported_usage_events": 3},
     {"label": "idea generation", "baseline_output_tokens": 4_000, "baseline_unreported": 1},
     "Tokens · idea generation · 400 output · run 4.4k · 2 unreported"),
    ("no-activity", {"usage": {}}, {"label": "task draft"}, "Tokens · task draft · 0 output · run 0"),
]


def test_checkpoint_formatting_reports_token_deltas_and_rejects_invalid_counts() -> None:
    # checkpoints report token deltas without dollar figures
    for name, summary, kwargs, expected in _CHECKPOINTS:
        line = format_cost_checkpoint(summary, **kwargs)
        assert line == expected, name
        assert "$" not in line, name

    # invalid token counts are rejected
    for output_tokens in ("many", -1):
        with pytest.raises(CostTrackingError):
            format_cost_checkpoint({"usage": {"output_tokens": output_tokens}}, label="x")

    # token counts match the Claude Code footer format
    for tokens, rendered in [(0, "0"), (999, "999"), (1_000, "1.0k"), (48_260, "48.3k"), (4_800_000, "4.8M")]:
        assert format_token_count(tokens) == rendered, tokens


# --------------------------------------------------------------------------- CLI


def test_cost_cli_replacements_report_only_new_cost(tmp_path: Path) -> None:
    # status replacement reports only new cost
    log_path = tmp_path / "claude.jsonl"
    _write_jsonl(log_path, [{"type": "result", "total_cost_usd": 0.25, "usage": {"output_tokens": 500}}])
    root = tmp_path / "run"
    assert _headless_cli(root, log_path, invocation="draft-retry", status="failed") == "Tokens · paper/draft · 500 output · run 500"
    assert _headless_cli(root, log_path, invocation="draft-retry", status="completed") == "Tokens · paper/draft · 0 output · run 500"

    # representation replacement reports only new cost
    log_path = tmp_path / "claude-representation.jsonl"
    normalized = _claude(inputTokens=20, outputTokens=5, cost={"total": 0.3})
    root = tmp_path / "run-representation"
    _write_jsonl(log_path, [normalized])
    first = _headless_cli(root, log_path, invocation="draft-representation")
    _write_jsonl(log_path, [
        {"type": "result", "modelUsage": {"claude-haiku": {"costUSD": 0.1, "inputTokens": 8, "outputTokens": 2},
                                          "claude-opus": {"costUSD": 0.2, "inputTokens": 12, "outputTokens": 3}}},
        normalized,
    ])
    second = _headless_cli(root, log_path, invocation="draft-representation")
    assert first == "Tokens · paper/draft · 5 output · run 5"
    assert second == "Tokens · paper/draft · 0 output · run 5"


def test_cost_cli_json_mode_is_machine_readable_and_includes_estimates(tmp_path: Path) -> None:
    # json terminal mode stays machine readable
    log_path = tmp_path / "agent.jsonl"
    _write_jsonl(log_path, [{"type": "result", "total_cost_usd": 0.1}])
    output = json.loads(_headless_cli(tmp_path / "run", log_path, invocation="draft-json", json_output=True))
    assert output["event"] == "cost_summary"
    assert output["fields"]["label"] == "paper/draft"

    # normalized headless estimate is reported in the total
    log_path = tmp_path / "codex.jsonl"
    _write_jsonl(log_path, [
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
        _normalized(inputTokens=10, cacheReadTokens=0, cacheWriteTokens=0, outputTokens=2, reasoningOutputTokens=0,
                    cost={"input": 0.00005, "cacheRead": 0, "cacheWrite": 0, "output": 0.00006, "total": 0.00011}),
    ])
    output = json.loads(_headless_cli(tmp_path / "run-codex", log_path, invocation="idea-json", agent="codex",
                                      phase="idea", stage="idea_generation", json_output=True))
    assert output["fields"]["estimated_cost_usd"] == "0.00011"
    assert output["fields"]["total_cost_usd"] == "0.00011"
    assert output["fields"]["run_total_cost_usd"] == "0.00011"


# --------------------------------------------------------------------------- autoillustrate import


def _cost_entry(**over: Any) -> dict[str, Any]:
    """A trustworthy provenance-labelled headless estimate entry."""
    entry = {
        "source": "headless:pi", "model": "gpt-5.4", "cost_usd": 0.4, "estimated": True,
        "basis": "api-list-price-estimate", "pricing_source": "models.dev", "pricing_status": "priced",
        "usage": {"input_tokens": 4},
    }
    entry.update(over)
    return _strip(entry)


def _reported_claude(cost_usd: float, usage: dict[str, int]) -> dict[str, Any]:
    return {"source": "headless:claude", "model": "claude-opus", "cost_usd": cost_usd, "estimated": False,
            "basis": "self-reported", "usage": usage}


def _import_metrics(tmp_path: Path, payload: dict[str, Any], invocation: str = "figure-1") -> list[dict[str, Any]]:
    path = tmp_path / "run_metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return import_autoillustrate_metrics(path, phase="paper", stage="concept_figure", invocation_id=invocation,
                                         status="completed")


def test_autoillustrate_import_keeps_usage_and_trusts_only_provenance_labelled_costs(tmp_path: Path) -> None:
    # estimates are discarded but usage is kept
    events = _import_metrics(tmp_path, {"costs": [
        _reported_claude(0.4, {"input_tokens": 50}),
        {"source": "openai-image", "model": "gpt-image-1", "cost_usd": 0.08, "estimated": True,
         "usage": {"output_tokens": 10}, "images": 1},
    ]})
    assert [event["billing_basis"] for event in events] == ["reported", "usage_only"]
    assert [event["reported_cost_usd"] for event in events] == ["0.4", None]
    assert events[1]["usage"]["images"] == 1

    # the real phase metrics schema is flattened
    events = _import_metrics(tmp_path, {"schema_version": 1, "phases": [
        {"phase": 1, "costs": [_reported_claude(0.6, {"input_tokens": 70, "cache_read_tokens": 30})]},
        {"phase": 3, "costs": [{"source": "openai", "model": "gpt-image-1", "cost_usd": 0.2, "estimated": True,
                                "basis": "image", "images": 1}]},
    ]}, "figure-real")
    assert [event["reported_cost_usd"] for event in events] == ["0.6", None]
    assert events[0]["usage"]["cached_input_tokens"] == 30
    assert events[1]["usage"]["images"] == 1

    # unverifiable costs are not trusted
    for name, entry in [
        ("unrecognized-reported-source", {"source": "headless:untrusted", "basis": "self-reported", "model": "unknown",
                                          "cost_usd": 999, "usage": {"input_tokens": 4}}),
        ("conflicting-estimate-provenance", _cost_entry(cost_usd=999, pricing_source={"unexpected": "object"})),
    ]:
        [event] = _import_metrics(tmp_path, {"costs": [entry]}, "figure-untrusted")
        assert event["billing_basis"] == "usage_only", name
        assert event["reported_cost_usd"] is None, name
        assert event["estimated_cost_usd"] is None, name

    # a provenance-labelled headless estimate is imported
    [event] = _import_metrics(tmp_path, {"costs": [_cost_entry(usage={"input_tokens": 50, "output_tokens": 10})]},
                              "figure-estimated")
    assert event["billing_basis"] == "estimated"
    assert event["reported_cost_usd"] is None
    assert event["estimated_cost_usd"] == "0.4"
    assert event["cost_basis"] == "api-list-price-estimate"
    assert event["pricing_source"] == "models.dev"
    assert event["pricing_status"] == "priced"

    # malformed costs are isolated from valid entries
    malformed = ("0.4", True, -1, float("inf"), 1_000_000_001)
    events = _import_metrics(tmp_path, {"costs": [*(_cost_entry(cost_usd=value) for value in malformed), _cost_entry()]},
                             "figure-malformed-costs")
    assert [event["billing_basis"] for event in events] == ["usage_only"] * len(malformed) + ["estimated"]
    assert events[-1]["estimated_cost_usd"] == "0.4"


def test_autoillustrate_cli_preserves_metrics_and_prints_checkpoint(tmp_path: Path) -> None:
    metrics_path = tmp_path / "temporary" / "run_metrics.json"
    metrics_path.parent.mkdir()
    metrics_path.write_text(json.dumps({"costs": [_reported_claude(0.25, {"input_tokens": 1_000, "output_tokens": 250})]}),
                            encoding="utf-8")
    root = tmp_path / "run"
    stdout = _cli(root, metrics_path, "autoillustrate", "--phase", "paper", "--stage", "concept_figure",
                  "--invocation", "figure-preserved")

    assert "Tokens · paper/concept_figure · 250 output · run 250" in stdout
    [preserved] = (root / "metrics" / "autoillustrate").glob("*.json")
    assert json.loads(preserved.read_text(encoding="utf-8"))["costs"][0]["cost_usd"] == 0.25
    [event] = _ledger(root)
    assert event["artifact"] == f"metrics/autoillustrate/{preserved.name}"


# --------------------------------------------------------------------------- billing attempt metadata


def _attempt_usage(tokens: int | float = 10) -> dict[str, object]:
    return {
        "agent": "codex", "model": "gpt-5.4", "provider": "openai",
        "inputTokens": tokens, "outputTokens": 2, "usageStatus": "reported",
        "cost": None, "costBasis": None, "pricingSource": None, "pricingStatus": "missing",
    }


def _billing() -> dict[str, Any]:
    return {"attempts": [
        {"route": "subscription", "reason": "subscription-model-unsupported", "usage": _attempt_usage()},
        {"route": "openai-api", "usage": _attempt_usage(30)},
    ]}


def _parse_billing(tmp_path: Path, billing: object, *extra_records: dict) -> list[dict]:
    summary = _attempt_usage(40) | {"outputTokens": 4, "billing": billing}
    return _parse(tmp_path, [*extra_records, {"usage": summary}], phase="investigation", stage="idea",
                  invocation_id="idea-1", name="headless.jsonl")


def _parse_attempts(tmp_path: Path, attempts: list[dict]) -> list[dict]:
    summary = _attempt_usage(40) | {"usageStatus": "missing", "billing": {"attempts": attempts}}
    return _parse(tmp_path, [{"usage": summary}], phase="investigation", stage="idea", invocation_id="attempts",
                  name="attempts.jsonl")


_MALFORMED_BILLING: list[dict[str, Any]] = [
    {"attempts": []},
    {"attempts": [{"route": "subscription", "usage": _attempt_usage()}] * 6},
    {"attempts": [{"route": "secret-route", "usage": _attempt_usage()}]},
    {"attempts": [{"route": "subscription", "reason": "arbitrary secret", "usage": _attempt_usage()}]},
    {"attempts": [{"route": "subscription", "usage": _attempt_usage(-1)}]},
    {"attempts": [{"route": "subscription", "usage": _attempt_usage(True)}]},
    {"attempts": [{"route": "subscription", "usage": _attempt_usage() | {"cost": {"total": -1}}}]},
]


def test_billing_attempt_metadata_round_trips_ledger_and_drops_unknown_or_malformed_fields(tmp_path: Path):
    # attempt metadata survives the cost ledger without double counting
    root = tmp_path / "ledger"
    root.mkdir()
    events = _parse_billing(root, _billing())
    assert [event["billing"]["attempts"][0] for event in events] == _billing()["attempts"]
    summary = record_events(root, events)
    assert [event["billing"]["attempts"][0] for event in _ledger(root)] == _billing()["attempts"]
    assert summary["usage"]["input_tokens"] == 40

    # unknown billing metadata is not retained
    billing = _billing()
    billing["secret"] = "must-not-store"
    billing["attempts"][0]["credential"] = "must-not-store"
    billing["attempts"][0]["usage"]["rawResponse"] = "must-not-store"
    events = _parse_billing(tmp_path, billing)
    assert [event["billing"]["attempts"][0] for event in events] == _billing()["attempts"]
    assert "must-not-store" not in json.dumps(events)

    # malformed billing diagnostics do not discard valid aggregate usage
    for index, billing in enumerate(_MALFORMED_BILLING):
        [event] = _parse_billing(tmp_path, billing)
        assert "billing" not in event, index
        assert event["usage"]["input_tokens"] == 40, index

    # raw model details do not replace combined billing attempt usage
    events = _parse_billing(tmp_path, _billing(), {"type": "result", "modelUsage": {
        "gpt-5.4": {"inputTokens": 30, "outputTokens": 2, "costUSD": 1},
        "other-model": {"inputTokens": 10, "outputTokens": 2, "costUSD": 1},
    }})
    assert len(events) == 2
    assert [event["billing"]["attempts"][0] for event in events] == _billing()["attempts"]
    assert sum(event["usage"]["input_tokens"] for event in events) == 40

    # the ledger validates direct billing metadata before writing
    event = _parse_billing(tmp_path, _billing())[0]
    event["billing"] = {"attempts": [{"route": "credential-value"}]}
    with pytest.raises(CostTrackingError, match="billing metadata"):
        record_events(tmp_path / "direct", [event])


_INVALID_METADATA: list[object] = [
    None, [], {"attempts": [None]},
    {"attempts": [{"route": "native", "usage": None}]},
    {"attempts": [{"route": "native", "usage": _attempt_usage() | {"model": "secret\\ntext"}}]},
    {"attempts": [{"route": "native", "usage": _attempt_usage() | {"pricingStatus": []}}]},
    {"attempts": [{"route": "native", "usage": _attempt_usage(1.25)}]},
    {"attempts": [{"route": "native", "usage": _attempt_usage(10**400)}]},
    {"attempts": [{"route": "native", "usage": _attempt_usage() | {"cost": "secret"}}]},
]


def test_validated_billing_metadata_keeps_attempt_costs_and_rejects_malformed_fields():
    # attempt cost and missing usage are preserved for every route
    for route in ("native", "bedrock", "openai-api", "subscription"):
        usage = _attempt_usage() | {"cost": {"total": 0.125, "input": None}, "inputTokens": None}
        clean = validated_billing_metadata({"attempts": [{"route": route, "usage": usage}]})
        assert clean["attempts"][0]["usage"]["cost"] == {"total": 0.125, "input": None}, route
        assert "inputTokens" not in clean["attempts"][0]["usage"], route

    # malformed fields are rejected
    for value in _INVALID_METADATA:
        with pytest.raises(ValueError):
            validated_billing_metadata(value)

    # invalid capacity retry reasons are rejected
    for reason in ("unknown", "subscription-quota", 3, [], "model-capacity\n"):
        billing = _billing()
        billing["attempts"][0]["retryReason"] = reason
        with pytest.raises(ValueError, match="retry reason"):
            validated_billing_metadata(billing)


def test_subscription_and_api_attempts_are_valued_separately(tmp_path: Path):
    # a missing subscription attempt retains paid usage and cost
    root = tmp_path / "missing"
    root.mkdir()
    missing = _attempt_usage(0) | {"outputTokens": 0, "usageStatus": "missing"}
    paid = _attempt_usage(100) | {"cost": {"total": 1}} | {
        key: value for key, value in _ESTIMATE.items() if key != "usageStatus"}
    events = _parse_attempts(root, [
        {"route": "subscription", "usage": missing},
        {"route": "openai-api", "reason": "subscription-model-unsupported", "usage": paid},
    ])
    summary = record_events(root, events)
    assert summary["estimated_cost_usd"] == "1"
    assert summary["usage"]["input_tokens"] == 100
    assert summary["unknown_events"] == 1
    assert summary["event_count"] == 2
    repeated = record_events(root, events)
    assert repeated["event_count"] == 2
    assert repeated["estimated_cost_usd"] == "1"
    assert repeated["_recorded_event_count"] == 0

    # mixed subscription valuation and api estimate remain separate
    root = tmp_path / "mixed"
    root.mkdir()
    native = _attempt_usage(10) | {"cost": {"total": 2}} | _NATIVE
    paid = _attempt_usage(30) | {"cost": {"total": 1}} | _ESTIMATE
    events = _parse_attempts(root, [
        {"route": "subscription", "usage": native},
        {"route": "openai-api", "reason": "subscription-quota", "usage": paid},
    ])
    summary = record_events(root, events)
    assert summary["reported_cost_usd"] == "2"
    assert summary["estimated_cost_usd"] == "1"
    assert summary["usage"]["input_tokens"] == 40
    assert summary["unknown_events"] == 0
    assert len({event["event_id"] for event in events}) == 2


def _claude_attempt_usage() -> dict:
    return _attempt_usage(30) | {"agent": "claude", "provider": "anthropic", "model": "claude-opus-4-6",
                                 "outputTokens": 3, "cost": {"total": 3}} | _NATIVE


def _claude_result() -> dict:
    return {"type": "result", "modelUsage": {
        "claude-opus-4-6": {"inputTokens": 20, "outputTokens": 2, "costUSD": 2},
        "claude-haiku-4-5": {"inputTokens": 10, "outputTokens": 1, "costUSD": 1},
    }}


def test_per_attempt_model_costs_and_capacity_retries_survive_ledger_without_double_counting(tmp_path: Path):
    # claude model costs are preserved per attempt
    for routes in (["subscription"], ["subscription", "bedrock"]):
        root = tmp_path / f"routes-{len(routes)}"
        root.mkdir()
        usage = _claude_attempt_usage()
        attempts = [{"route": route, "usage": usage} for route in routes]
        aggregate = usage | {"inputTokens": 30 * len(routes), "outputTokens": 3 * len(routes),
                             "cost": {"total": 3 * len(routes)}, "billing": {"attempts": attempts}}
        events = _parse(root, [*[_claude_result() for _ in routes], {"usage": aggregate}], agent="claude",
                        phase="paper", stage="draft", invocation_id="claude-draft", name="claude.jsonl")
        assert len(events) == 2 * len(routes), routes
        assert len({event["event_id"] for event in events}) == len(events), routes
        assert [event["model"] for event in events] == ["claude-haiku-4-5", "claude-opus-4-6"] * len(routes), routes
        assert [event["reported_cost_usd"] for event in events] == ["1", "2"] * len(routes), routes
        summary = record_events(root, events)
        assert summary["reported_cost_usd"] == str(3 * len(routes)), routes
        assert summary["usage"]["input_tokens"] == 30 * len(routes), routes
        assert record_events(root, events)["_recorded_event_count"] == 0, routes

    # ambiguous claude model details keep attempt totals under "mixed"
    for result_count in (1, 2):
        root = tmp_path / f"ambiguous-{result_count}"
        root.mkdir()
        usage = _claude_attempt_usage()
        attempts = [{"route": "subscription", "usage": usage}, {"route": "bedrock", "usage": usage}]
        result = _claude_result()
        result["modelUsage"]["claude-opus-4-6"]["inputTokens"] = 1000
        events = _parse(root, [
            *[result for _ in range(result_count)],
            {"usage": usage | {"inputTokens": 60, "cost": {"total": 6}, "billing": {"attempts": attempts}}},
        ], agent="claude", phase="paper", stage="draft", invocation_id="ambiguous", name="ambiguous.jsonl")
        assert [event["model"] for event in events] == ["mixed", "mixed"], result_count
        assert sum(event["usage"]["input_tokens"] for event in events) == 60, result_count
        assert record_events(root, events)["reported_cost_usd"] == "6", result_count

    # capacity retry attempts survive the ledger without double counting
    for count in (3, 5):
        root = tmp_path / f"capacity-{count}"
        root.mkdir()
        attempts = [{"route": "subscription", "usage": _attempt_usage(10)}]
        attempts.extend({"route": "subscription", "retryReason": "model-capacity", "usage": _attempt_usage(10)}
                        for _ in range(count - 1))
        events = _parse_billing(root, {"attempts": attempts})
        assert len(events) == count, count
        summary = record_events(root, events)
        assert summary["usage"]["input_tokens"] == count * 10, count
        assert [event["billing"]["attempts"][0] for event in _ledger(root)] == attempts, count
