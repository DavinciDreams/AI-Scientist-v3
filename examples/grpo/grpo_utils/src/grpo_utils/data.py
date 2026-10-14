"""
Data loading utilities for GRPO training.
"""

from __future__ import annotations

import argparse
import json
import logging
import math as _math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

MATH_DIR = Path(os.path.dirname(__file__)) / "MATH"
TRAIN_PATH = MATH_DIR / "_train.jsonl"
TRAIN_SPLIT_PATH = MATH_DIR / "train_split.jsonl"
VALIDATION_PATH = MATH_DIR / "validation.jsonl"


class MathDataset(Dataset):
    def __init__(self, input_ids, labels, response_mask):
        self.input_ids = input_ids
        self.labels = labels
        self.response_mask = response_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx], self.response_mask[idx]


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL dataset file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str | Path, data: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


# Keep the old name as an alias for backward compatibility.
load_dataset = load_jsonl


def create_data_loader(dataset, batch_size=8, shuffle=True):
    """Create a DataLoader from tokenized dataset."""
    dataset = MathDataset(
        dataset["input_ids"], dataset["labels"], dataset["response_mask"]
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    return data_loader


# ---------------------------------------------------------------------------
# Stratified validation-set creation
# ---------------------------------------------------------------------------


def stratified_sample(
    data: list[dict],
    size: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Sample *size* entries from *data*, preserving (subject, level) proportions.

    Uses largest-remainder rounding so the total is exactly *size*.

    Returns:
        (sampled, remainder) — the validation split and the leftover training data.
    """
    rng = random.Random(seed)

    # Group by (subject, level)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in data:
        key = (entry["subject"], entry["level"])
        groups[key].append(entry)

    # Compute per-group quota
    total = len(data)
    raw_quotas = {k: len(v) / total * size for k, v in groups.items()}
    floored = {k: int(_math.floor(q)) for k, q in raw_quotas.items()}
    remainders = {k: raw_quotas[k] - floored[k] for k in raw_quotas}

    allocated = sum(floored.values())
    shortfall = size - allocated

    # Distribute remaining slots to groups with largest fractional remainders
    for k in sorted(remainders, key=remainders.get, reverse=True)[:shortfall]:
        floored[k] += 1

    # Sample from each group, keeping track of the remainder
    sampled: list[dict] = []
    remainder: list[dict] = []
    for key in sorted(groups):
        pool = groups[key]
        quota = min(floored[key], len(pool))
        if quota == 0:
            remainder.extend(pool)
            continue
        chosen = set(rng.sample(range(len(pool)), quota))
        for i, entry in enumerate(pool):
            if i in chosen:
                sampled.append(entry)
            else:
                remainder.append(entry)

    # Shuffle both sets (deterministically)
    rng.shuffle(sampled)
    rng.shuffle(remainder)
    return sampled, remainder


def print_distribution(
    original: list[dict],
    train_split: list[dict],
    validation: list[dict],
) -> None:
    """Print side-by-side distribution comparison for subjects and levels."""
    n_orig = len(original)
    n_train = len(train_split)
    n_val = len(validation)

    orig_subj = Counter(d["subject"] for d in original)
    train_subj = Counter(d["subject"] for d in train_split)
    val_subj = Counter(d["subject"] for d in validation)
    orig_lvl = Counter(d["level"] for d in original)
    train_lvl = Counter(d["level"] for d in train_split)
    val_lvl = Counter(d["level"] for d in validation)

    logger.info("")
    logger.info("%-25s  %14s  %14s  %14s", "", "original", "train_split", "validation")
    logger.info("-" * 73)

    logger.info("Subject:")
    for subj in sorted(orig_subj):
        o_n = orig_subj[subj]
        t_n = train_subj.get(subj, 0)
        v_n = val_subj.get(subj, 0)
        logger.info(
            "  %-23s  %5d (%5.1f%%)  %5d (%5.1f%%)  %5d (%5.1f%%)",
            subj,
            o_n, o_n / n_orig * 100,
            t_n, t_n / n_train * 100 if n_train else 0,
            v_n, v_n / n_val * 100 if n_val else 0,
        )

    logger.info("")
    logger.info("Level:")
    for lvl in sorted(orig_lvl):
        o_n = orig_lvl[lvl]
        t_n = train_lvl.get(lvl, 0)
        v_n = val_lvl.get(lvl, 0)
        logger.info(
            "  %-23s  %5d (%5.1f%%)  %5d (%5.1f%%)  %5d (%5.1f%%)",
            lvl,
            o_n, o_n / n_orig * 100,
            t_n, t_n / n_train * 100 if n_train else 0,
            v_n, v_n / n_val * 100 if n_val else 0,
        )

    logger.info("")
    logger.info("%-25s  %5d          %5d          %5d", "Total", n_orig, n_train, n_val)


def create_validation_set_cli() -> None:
    """CLI entry point: create a stratified validation set from the archived MATH train set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Create a stratified validation set from MATH _train.jsonl"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=20.0,
        help="Percentage of training set to use for validation (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-validation",
        type=Path,
        default=VALIDATION_PATH,
        help="Output path for validation set (default: %(default)s)",
    )
    parser.add_argument(
        "--output-train",
        type=Path,
        default=TRAIN_SPLIT_PATH,
        help="Output path for remaining training set (default: %(default)s)",
    )
    args = parser.parse_args()

    if not TRAIN_PATH.exists():
        logger.error("Training file not found: %s", TRAIN_PATH)
        raise SystemExit(1)

    if not 0 < args.percentage < 100:
        logger.error(
            "Percentage must be between 0 and 100 (exclusive), got %.1f",
            args.percentage,
        )
        raise SystemExit(1)

    train_data = load_jsonl(TRAIN_PATH)
    logger.info("Loaded %d training examples from %s", len(train_data), TRAIN_PATH)

    size = max(1, round(len(train_data) * args.percentage / 100))
    logger.info("Sampling %.1f%% -> %d validation examples", args.percentage, size)

    validation, train_split = stratified_sample(train_data, size, args.seed)

    write_jsonl(args.output_validation, validation)
    logger.info("Wrote %d validation examples to %s", len(validation), args.output_validation)

    write_jsonl(args.output_train, train_split)
    logger.info("Wrote %d training examples to %s", len(train_split), args.output_train)

    print_distribution(train_data, train_split, validation)


if __name__ == "__main__":
    create_validation_set_cli()
