"""Bounded scientific observations authored by the sealed evaluator.

Candidate diagnostics are observations, never independent validity certificates.
Only fixed JSON data crosses this boundary; candidate filesystem paths do not.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

EVIDENCE_FILENAME = "evaluation_evidence.json"
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
_MAX_DEPTH = 12
_MAX_EPISODES = 10_000
_MAX_TIMESERIES_ROWS = 100_000
_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}\Z")
_REQUIRED_EPISODE = {"seed", "score", "correct", "duration_seconds"}
_OPTIONAL_EPISODE = {"solution", "candidate_diagnostics", "candidate_timeseries"}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("evaluation evidence contains duplicate keys")
        result[key] = value
    return result


def _read_evidence(root: Path) -> bytes | None:
    path = root / EVIDENCE_FILENAME
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("evaluation evidence must be a regular file") from exc
        raise
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("evaluation evidence must be a regular file")
        if info.st_size > MAX_EVIDENCE_BYTES:
            raise ValueError("evaluation evidence exceeds the 16 MiB limit")
    except BaseException:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read(MAX_EVIDENCE_BYTES + 1)
    if len(raw) > MAX_EVIDENCE_BYTES:
        raise ValueError("evaluation evidence exceeds the 16 MiB limit")
    return raw


def _finite(value: object) -> bool:
    try:
        return (
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        )
    except OverflowError:
        return False


def _validate_json(value: object, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise ValueError("evaluation evidence exceeds the nesting depth limit")
    if isinstance(value, dict):
        for key, item in value.items():
            if len(key) > 512:
                raise ValueError("evaluation evidence key exceeds length limit")
            _validate_json(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json(item, depth + 1)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not _finite(value):
            raise ValueError("evaluation evidence numbers must be finite")


def _validate_episode(episode: Any) -> int:
    if not isinstance(episode, dict) or not _REQUIRED_EPISODE <= episode.keys():
        raise ValueError("evaluation evidence episode is missing required fields")
    if episode.keys() - (_REQUIRED_EPISODE | _OPTIONAL_EPISODE):
        raise ValueError("evaluation evidence episode has undeclared fields")
    seed = episode["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not -(2**63) <= seed < 2**63:
        raise ValueError("evaluation evidence seed must be a signed 64-bit integer")
    if not _finite(episode["score"]) or not isinstance(episode["correct"], bool):
        raise ValueError("evaluation evidence score/correctness is invalid")
    duration = episode["duration_seconds"]
    if not _finite(duration) or duration < 0:
        raise ValueError("evaluation evidence duration must be finite and nonnegative")
    diagnostics = episode.get("candidate_diagnostics", {})
    if not isinstance(diagnostics, dict) or any(
        not _NAME.fullmatch(name) or not _finite(value) for name, value in diagnostics.items()
    ):
        raise ValueError("evaluation evidence candidate diagnostics must be named finite numbers")
    rows = episode.get("candidate_timeseries", [])
    if not isinstance(rows, list):
        raise ValueError("evaluation evidence candidate timeseries must be a list")
    for row in rows:
        if not isinstance(row, dict) or row.keys() != {"series_name", "step", "value"}:
            raise ValueError("evaluation evidence timeseries fields are invalid")
        name, step = row["series_name"], row["step"]
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("evaluation evidence series name is invalid")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("evaluation evidence series step must be a nonnegative integer")
        if not _finite(row["value"]):
            raise ValueError("evaluation evidence series value must be finite")
    return seed


def _decode(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
        _validate_json(payload)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("evaluation evidence must be bounded valid JSON") from exc
    if not isinstance(payload, dict) or payload.keys() != {"schema_version", "episodes"}:
        raise ValueError("evaluation evidence has undeclared or missing fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("evaluation evidence schema_version must be 1")
    episodes = payload["episodes"]
    if not isinstance(episodes, list) or not 1 <= len(episodes) <= _MAX_EPISODES:
        raise ValueError("evaluation evidence episodes exceed count limit or are empty")
    seeds: set[int] = set()
    row_count = 0
    for episode in episodes:
        seed = _validate_episode(episode)
        if seed in seeds:
            raise ValueError("evaluation evidence has duplicate seeds")
        seeds.add(seed)
        row_count += len(episode.get("candidate_timeseries", []))
        if row_count > _MAX_TIMESERIES_ROWS:
            raise ValueError("evaluation evidence timeseries exceed row limit")
    return payload


def load_evaluation_evidence(root: Path) -> dict[str, Any] | None:
    raw = _read_evidence(root)
    return None if raw is None else _decode(raw)


def evidence_metadata(root: Path) -> dict[str, Any]:
    raw = _read_evidence(root)
    if raw is None:
        return {"status": "unavailable"}
    payload = _decode(raw)
    return {
        "status": "available",
        "path": str(root / EVIDENCE_FILENAME),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "episode_count": len(payload["episodes"]),
        "candidate_diagnostics_trust": "unverified",
    }


def evidence_timeseries(root: Path, evaluation_id: str, variant_id: str) -> list[dict[str, Any]]:
    payload = load_evaluation_evidence(root)
    if payload is None:
        return []
    return [
        {
            **row,
            "series_name": f"candidate/seed_{episode['seed']}/{row['series_name']}",
            "evaluation_id": evaluation_id,
            "variant_id": variant_id,
        }
        for episode in payload["episodes"]
        for row in episode.get("candidate_timeseries", [])
    ]
