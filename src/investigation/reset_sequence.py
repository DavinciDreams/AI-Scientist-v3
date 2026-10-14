"""Candidate reset transcript validation shared across evaluation phases."""

from __future__ import annotations


def declared_reset_sequence_matches(
    observed_reset_seeds: tuple[object, ...],
    declared_reset_seeds: tuple[int, ...],
) -> bool:
    """Accept the exact primary transcript, including repeats, and one bounded replay."""
    if not declared_reset_seeds:
        return False
    replay_sequence = (*declared_reset_seeds, declared_reset_seeds[0])
    return observed_reset_seeds in (declared_reset_seeds, replay_sequence)


def reduced_draft_reset_sequence_matches(
    observed_reset_seeds: tuple[object, ...],
    declared_reset_seeds: tuple[int, ...],
) -> bool:
    """Validate a reduced draft smoke transcript against its first seed."""
    return declared_reset_sequence_matches(observed_reset_seeds, declared_reset_seeds[:1])


def reset_sequence_diagnostic(
    observed_reset_seeds: tuple[object, ...],
    expected_primary_reset_seeds: tuple[int, ...],
    *,
    context: str,
) -> str:
    """Describe a reset mismatch without rendering arbitrary candidate values."""

    def safe_seed(value: object) -> int | str:
        if isinstance(value, int) and not isinstance(value, bool):
            if value.bit_length() <= 63:
                return value
            return f"<int bits={value.bit_length()}>"
        return f"<{type(value).__name__}>"

    expected = [safe_seed(value) for value in expected_primary_reset_seeds[:8]]
    if len(expected_primary_reset_seeds) > len(expected):
        expected.append("<truncated>")
    observed = [safe_seed(value) for value in observed_reset_seeds[:8]]
    if len(observed_reset_seeds) > len(observed):
        observed.append("<truncated>")
    replay_seed = safe_seed(expected_primary_reset_seeds[0]) if expected_primary_reset_seeds else "<none>"
    return (
        f"{context} expected primary {expected}; observed {observed} "
        f"(count={len(observed_reset_seeds)}). The primary sequence must run exactly once; "
        f"one trailing deterministic replay of seed {replay_seed} is allowed."
    )
