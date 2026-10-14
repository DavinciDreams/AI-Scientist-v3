"""Shared helpers for the post-hoc pipeline."""

from __future__ import annotations

import json
import logging
import os
import pickletools
import re
import shutil
import struct
import zipfile
from pathlib import Path
from typing import Any, Iterable

from src.investigation.files import read_json as _read_json
from src.investigation.program_clone import clone_program_dir
from src.investigation.program_spec import ProgramSpecError, load_program_spec, resolve_entrypoint

logger = logging.getLogger(__name__)

TASK_COPY_EXCLUDE_DIRS: set[str] = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "node_modules",
    ".git",
}

TASK_COPY_EXCLUDE_FILES: set[str] = {
    "correct.json",
    "metrics.json",
    "rubric_status.json",
    "stdout.log",
    "stderr.log",
}

TASK_COPY_EXCLUDE_SUFFIXES: tuple[str, ...] = (
    ".log",
    ".out",
    ".err",
)

CHECKPOINT_SUFFIXES: tuple[str, ...] = (
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".joblib",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
)

CHECKPOINT_NAME_HINTS: tuple[str, ...] = (
    "checkpoint",
    "ckpt",
    "model",
    "weights",
    "state",
    "best",
    "snapshot",
)

CHECKPOINT_NAME_EXCLUDES: tuple[str, ...] = (
    "train",
    "test",
    "data",
    "manifest",
    "split",
    "zca",
    "inv_sqrt",
    "x_mean",
    "labels",
    "images",
    "classes",
)

PRIMARY_METRIC_PRIORITY: tuple[str, ...] = (
    "val_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "accuracy",
    "acc",
    "f1",
    "auc",
    "reward",
    "score",
    "return",
    "lower_bound",
    "upper_bound",
    "loss",
    "perplexity",
)

PRIMARY_METRIC_EXCLUDE: set[str] = {
    "combined_score",
    "correct",
    "error",
    "terminated_early",
    "training_time",
    "time",
    "time_mean",
    "wall_time",
    "elapsed_time",
    "seed",
    "split_seed",
    "train_examples",
    "val_examples",
    "test_examples",
}

PROGRAM_DIR_RE = re.compile(r"^program_(\d+)$")


def read_json(path: Path) -> dict[str, Any] | None:
    data = _read_json(path)
    return data if isinstance(data, dict) else None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_extraction_manifest(bundle_dir: Path) -> dict[str, Any]:
    manifest = read_json(bundle_dir / "extraction_manifest.json")
    if manifest is None:
        raise RuntimeError(f"Missing extraction_manifest.json in {bundle_dir}")
    return manifest


def resolve_program_entrypoint(program_dir: Path) -> Path:
    try:
        spec = load_program_spec(program_dir, required=True)
        assert spec is not None
        return resolve_entrypoint(program_dir, spec)
    except (ProgramSpecError, AssertionError) as exc:
        raise RuntimeError(f"Failed to resolve entrypoint for {program_dir}: {exc}") from exc


def copy_task_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy task-level files into *dst_dir*, preserving symlinks."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    for entry in sorted(src_dir.iterdir()):
        name = entry.name
        if name.startswith("program_") or name == "reference_program":
            continue
        if entry.is_dir() and name in TASK_COPY_EXCLUDE_DIRS:
            continue
        if name in TASK_COPY_EXCLUDE_FILES:
            continue
        if name.endswith(TASK_COPY_EXCLUDE_SUFFIXES):
            continue
        if not entry.is_dir() and not entry.is_symlink() and not entry.is_file():
            logger.warning("Skipping special file while copying task bundle: %s", entry)
            continue

        target = dst_dir / name
        if entry.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(entry, target, symlinks=True)
        elif entry.is_symlink():
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(os.readlink(entry), target_is_directory=False)
        else:
            shutil.copy2(entry, target)


def remove_stale_program_files(program_dir: Path) -> None:
    """Remove evaluator-managed files from a cloned program directory."""
    for path in list(program_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in TASK_COPY_EXCLUDE_FILES or path.name.endswith(TASK_COPY_EXCLUDE_SUFFIXES):
            path.unlink()


def clone_program_bundle(source_program_dir: Path, target_program_dir: Path) -> None:
    if target_program_dir.exists():
        shutil.rmtree(target_program_dir)
    clone_program_dir(src_dir=source_program_dir, dst_dir=target_program_dir)
    remove_stale_program_files(target_program_dir)


def list_program_dirs(root_dir: Path) -> list[Path]:
    found: list[tuple[int, Path]] = []
    if not root_dir.is_dir():
        return []
    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = PROGRAM_DIR_RE.match(entry.name)
        if not match:
            continue
        found.append((int(match.group(1)), entry))
    return [path for _, path in sorted(found, key=lambda item: item[0])]


def latest_program_dir(root_dir: Path) -> Path | None:
    programs = list_program_dirs(root_dir)
    return programs[-1] if programs else None


def copy_selected_runtime_artifacts(source_program_dir: Path, target_program_dir: Path) -> dict[str, list[str]]:
    """Copy post-run artifacts from the original best program into the bootstrap bundle."""
    copied: dict[str, list[str]] = {
        "files": [],
        "directories": [],
        "checkpoints": [],
    }

    for filename in ("metrics.json", "correct.json", "_results.json", "rubric_status.json"):
        src = source_program_dir / filename
        if not src.is_file():
            continue
        dst = target_program_dir / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied["files"].append(filename)

    for dirname in ("analysis", "figures", "plots", "artifacts"):
        src = source_program_dir / dirname
        if not src.is_dir():
            continue
        dst = target_program_dir / dirname
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
        copied["directories"].append(dirname)

    for checkpoint in find_checkpoint_candidates(source_program_dir):
        rel = checkpoint.relative_to(source_program_dir)
        dst = target_program_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, dst)
        copied["checkpoints"].append(rel.as_posix())

    return copied


def find_checkpoint_candidates(program_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    if not program_dir.is_dir():
        return []

    program_root = program_dir.resolve()
    results = read_json(program_dir / "_results.json") or {}
    for value in _candidate_paths_from_results_json(results):
        raw_path = Path(value)
        if raw_path.is_absolute():
            continue
        unresolved_candidate = program_root / raw_path
        candidate = unresolved_candidate.resolve()
        if (
            candidate.is_relative_to(program_root)
            and candidate.is_file()
            and not unresolved_candidate.is_symlink()
            and candidate not in seen
        ):
            seen.add(candidate)
            candidates.append(candidate)

    for path in sorted(program_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if _is_checkpoint_candidate(path):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)

    return candidates


def _candidate_paths_from_results_json(results: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in results.items():
        key_text = str(key).lower()
        if not isinstance(value, str):
            continue
        if "path" in key_text or "checkpoint" in key_text or "model" in key_text:
            values.append(value)
    return values


def _is_checkpoint_candidate(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        return False
    if any(exclude in name for exclude in CHECKPOINT_NAME_EXCLUDES):
        return False
    if any(hint in name for hint in CHECKPOINT_NAME_HINTS):
        return True
    parent_text = path.parent.as_posix().lower()
    return "checkpoint" in parent_text or "ckpt" in parent_text or "weights" in parent_text


def validate_checkpoint_loadable(path: Path) -> tuple[bool, str | None]:
    """Safely validate checkpoint structure without deserializing model objects."""
    if not path.is_file() or path.is_symlink():
        return False, "checkpoint file missing"
    if path.stat().st_size <= 0:
        return False, "checkpoint file is empty"

    try:
        suffix = path.suffix.lower()
        if suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".pkl", ".pickle", ".joblib"}:
            with path.open("rb") as handle:
                for _ in pickletools.genops(handle):
                    pass
        elif suffix == ".npy":
            import numpy as np

            np.load(path, allow_pickle=False, mmap_mode="r")
        elif suffix == ".npz":
            if not zipfile.is_zipfile(path):
                return False, "checkpoint archive is invalid"
        elif suffix == ".safetensors":
            with path.open("rb") as handle:
                header_length_bytes = handle.read(8)
                if len(header_length_bytes) != 8:
                    return False, "safetensors checkpoint header is invalid"
                header_length = struct.unpack("<Q", header_length_bytes)[0]
                if header_length <= 1 or header_length > path.stat().st_size - 8:
                    return False, "safetensors checkpoint header is invalid"
                json.loads(handle.read(header_length).decode("utf-8"))
        elif suffix in {".pt", ".pth", ".ckpt", ".bin"}:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    if not archive.namelist():
                        return False, "checkpoint archive is empty"
            else:
                with path.open("rb") as handle:
                    for _ in pickletools.genops(handle):
                        pass
        return True, None
    except Exception as exc:  # pragma: no cover - best effort validation
        return False, f"checkpoint not loadable: {exc}"


def infer_primary_metric(
    *,
    metrics: dict[str, Any] | None,
    results: dict[str, Any] | None = None,
    requested_key: str | None = None,
) -> dict[str, Any]:
    """Resolve the primary metric, preferring task metrics over combined_score."""
    metrics = metrics or {}
    results = results or {}

    if requested_key:
        resolved = _lookup_metric(requested_key, metrics, results)
        if resolved is not None:
            key, value, source = resolved
            return _metric_info(key, value, source)

    for key in PRIMARY_METRIC_PRIORITY:
        resolved = _lookup_metric(key, metrics, results)
        if resolved is not None:
            metric_key, value, source = resolved
            if metric_key != "combined_score":
                return _metric_info(metric_key, value, source)

    for source_name, payload in (("metrics.json", metrics), ("_results.json", results)):
        for key, value in payload.items():
            if key in PRIMARY_METRIC_EXCLUDE:
                continue
            if not isinstance(value, (int, float)):
                continue
            return _metric_info(str(key), float(value), source_name)

    if isinstance(metrics.get("combined_score"), (int, float)):
        return _metric_info("combined_score", float(metrics["combined_score"]), "metrics.json")

    raise RuntimeError("Unable to resolve a numeric primary metric from metrics.json or _results.json")


def _lookup_metric(
    key: str,
    metrics: dict[str, Any],
    results: dict[str, Any],
) -> tuple[str, float, str] | None:
    for source_name, payload in (("metrics.json", metrics), ("_results.json", results)):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return key, float(value), source_name
    return None


def _metric_info(key: str, value: float, source: str) -> dict[str, Any]:
    lower_better_markers = ("loss", "error", "time", "latency", "perplexity", "upper_bound")
    higher_is_better = not any(marker in key.lower() for marker in lower_better_markers)
    return {
        "key": key,
        "value": value,
        "source": source,
        "higher_is_better": higher_is_better,
    }


def compare_metric_guardrail(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    if baseline["key"] != candidate["key"]:
        return {
            "passed": False,
            "reason": (
                f"primary metric mismatch: baseline={baseline['key']} candidate={candidate['key']}"
            ),
            "baseline": baseline,
            "candidate": candidate,
            "tolerance": tolerance,
        }

    higher_is_better = bool(baseline["higher_is_better"])
    baseline_value = float(baseline["value"])
    candidate_value = float(candidate["value"])
    if higher_is_better:
        passed = candidate_value + tolerance >= baseline_value
    else:
        passed = candidate_value <= baseline_value + tolerance

    delta = candidate_value - baseline_value
    return {
        "passed": passed,
        "reason": None if passed else "primary metric regressed beyond tolerance",
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "tolerance": tolerance,
    }


def validate_expected_outputs(root_dir: Path, expected_outputs: Iterable[str]) -> dict[str, Any]:
    root = root_dir.resolve()
    found: list[str] = []
    missing: list[str] = []
    for relpath in expected_outputs:
        raw_path = Path(relpath)
        unresolved_candidate = root / raw_path
        candidate = unresolved_candidate.resolve()
        safe = not raw_path.is_absolute() and candidate.is_relative_to(root)
        if safe and candidate.is_file() and not unresolved_candidate.is_symlink():
            found.append(relpath)
        else:
            missing.append(relpath)
    return {
        "found": found,
        "missing": missing,
        "all_present": not missing,
    }


def collect_manifest_outputs(manifest: dict[str, Any]) -> list[str]:
    outputs: list[str] = []
    raw = manifest.get("outputs", {})
    if isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, list):
                outputs.extend(str(item) for item in value)
    elif isinstance(raw, list):
        outputs.extend(str(item) for item in raw)
    return outputs
