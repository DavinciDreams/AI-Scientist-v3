"""Constants for the variants pipeline: CSV schemas, rubric IDs, hard caps."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CSV column schemas (canonical on-disk contract)
# ---------------------------------------------------------------------------

RUN_SUMMARY_COLUMNS: list[str] = [
    "evaluation_id",
    "variant_id",
    "variant_kind",
    "base_program",
    "agent_name",
    "correct",
    "combined_score",
    "overall_pass",
    "primary_metric_name",
    "primary_metric_value",
    "program_dir",
    "results_dir",
]

METRIC_VALUES_COLUMNS: list[str] = [
    "evaluation_id",
    "variant_id",
    "metric_name",
    "metric_value",
    "is_primary_metric",
]

TIMESERIES_COLUMNS: list[str] = [
    "evaluation_id",
    "variant_id",
    "series_name",
    "step",
    "value",
]

# Mapping of analysis filename -> required columns
ANALYSIS_CSV_SCHEMAS: dict[str, list[str]] = {
    "run_summary.csv": RUN_SUMMARY_COLUMNS,
    "metric_values.csv": METRIC_VALUES_COLUMNS,
    "timeseries.csv": TIMESERIES_COLUMNS,
}

# ---------------------------------------------------------------------------
# Required rubric families (cross-agent contract)
# ---------------------------------------------------------------------------

REQUIRED_RUBRIC_IDS: list[str] = [
    "analysis_files_present",
    "run_summary_schema_valid",
    "metric_values_schema_valid",
    "timeseries_schema_valid",
    "metrics_consistent_with_metrics_json",
    "correctness_recorded",
    "variant_intent_preserved",
    "single_change_scope",
    "paper_analysis_ready",
]

# ---------------------------------------------------------------------------
# Hard caps
# ---------------------------------------------------------------------------

MAX_ABLATIONS: int = 3
MAX_EXTRA_BASELINES: int = 3
MAX_EVALS_PER_VARIANT: int = 2

# ---------------------------------------------------------------------------
# Variant kinds
# ---------------------------------------------------------------------------

VALID_VARIANT_KINDS: set[str] = {"control", "baseline", "ablation"}
