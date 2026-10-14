"""Shared read-only contract for evaluator artifacts, before qualification or promotion."""

from __future__ import annotations

import errno
import json
import math
import os
import stat
from collections.abc import Collection
from pathlib import Path
from typing import Any

from .evaluation_evidence import load_evaluation_evidence

MAX_RESULT_JSON_BYTES = 1024 * 1024


class EvaluatorOutputContractError(ValueError):
    """A sealed evaluator produced artifacts incompatible with the host contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluatorOutputContractError("evaluator JSON contains duplicate keys")
        result[key] = value
    return result


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        if error.errno in {errno.ENOENT, errno.ELOOP, errno.ENOTDIR}:
            raise EvaluatorOutputContractError("required evaluator result is missing or unsafe") from error
        raise
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RESULT_JSON_BYTES:
            raise EvaluatorOutputContractError("evaluator result must be a regular file under 1 MiB")
    except BaseException:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read(MAX_RESULT_JSON_BYTES + 1)
    if len(raw) > MAX_RESULT_JSON_BYTES:
        raise EvaluatorOutputContractError("evaluator result exceeds 1 MiB")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise EvaluatorOutputContractError("evaluator result must be bounded valid JSON") from error
    if not isinstance(payload, dict):
        raise EvaluatorOutputContractError("evaluator result must be a JSON object")
    return payload


def validate_evaluator_outputs(
    root: Path, metric_names: Collection[str],
) -> tuple[dict[str, Any], bool]:
    """Validate scientific output shape, not scientific acceptance or qualification feasibility."""
    metrics = _load_mapping(root / "metrics.json")
    if metrics.keys() != set(metric_names):
        raise EvaluatorOutputContractError("search result metric names must exactly match the task contract")
    for value in metrics.values():
        try:
            finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise EvaluatorOutputContractError("search result metrics must be finite numbers")
    correct = _load_mapping(root / "correct.json")
    if correct.keys() != {"correct"} or not isinstance(correct["correct"], bool):
        raise EvaluatorOutputContractError("correct.json must contain exactly one boolean key: correct")
    try:
        load_evaluation_evidence(root)
    except ValueError as error:
        raise EvaluatorOutputContractError(str(error)) from error
    return metrics, correct["correct"]
