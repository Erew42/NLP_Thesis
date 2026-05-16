from __future__ import annotations

from collections import defaultdict
import datetime as dt
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import polars as pl

try:
    from thesis_native import _lm2011_rust
except Exception as exc:  # pragma: no cover - optional native extension
    _lm2011_rust = None
    _AUTHORITY_RUST_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    _AUTHORITY_RUST_IMPORT_ERROR = None

from thesis_refinitiv.bridge import (
    OWNERSHIP_UNIVERSE_RESULTS_COLUMNS,
    OWNERSHIP_UNIVERSE_ROW_SUMMARY_COLUMNS,
    RIC_LOOKUP_RESOLUTION_OUTPUT_COLUMNS,
    _build_explicit_schema_df,
    _cast_df_to_schema,
    _normalize_lookup_text,
    _ownership_universe_results_schema,
    _ownership_universe_row_summary_schema,
    _read_resolution_artifact_parquet,
    _resolution_output_schema,
)


CONVENTIONAL_OWNERSHIP_LOOKUP_ROLES: frozenset[str] = frozenset(
    {
        "UNIVERSE_EFFECTIVE",
        "UNIVERSE_TARGET_ISIN_CANDIDATE",
        "UNIVERSE_TARGET_CUSIP_CANDIDATE",
    }
)
TICKER_OWNERSHIP_LOOKUP_ROLE = "UNIVERSE_TARGET_TICKER_CANDIDATE"
VALUE_EQUALITY_TOLERANCE = 1e-6
BENIGN_ALIAS_MIN_OVERLAP_DATE_CATEGORY_COUNT = 12
BENIGN_ALIAS_MIN_OVERLAP_DATE_COUNT = 3
BENIGN_ALIAS_MIN_OVERLAP_CATEGORY_COUNT = 3
BENIGN_ALIAS_MIN_OVERLAP_SHARE_OF_SMALLER_HISTORY = 0.50
DATE_EXTENSION_MAX_GAP_DAYS = 365
MEANINGFUL_STATIC_COVERAGE_LOSS_THRESHOLD = 0.80

_AUTHORITY_RUST_METRICS: dict[str, int] = {
    "date_span_fast_success": 0,
    "date_span_fast_failures": 0,
    "date_span_fallbacks": 0,
    "values_match_fast_success": 0,
    "values_match_fast_failures": 0,
    "values_match_fallbacks": 0,
    "distinct_values_fast_success": 0,
    "distinct_values_fast_failures": 0,
    "distinct_values_fallbacks": 0,
    "candidate_key_fast_success": 0,
    "candidate_key_fast_failures": 0,
    "candidate_key_fallbacks": 0,
    "allowlist_keys_column_fast_success": 0,
    "allowlist_keys_column_fast_failures": 0,
    "allowlist_keys_column_fallbacks": 0,
    "allowlist_keys_fast_success": 0,
    "allowlist_keys_fast_failures": 0,
    "allowlist_keys_fallbacks": 0,
    "candidate_metrics_column_fast_success": 0,
    "candidate_metrics_column_fast_failures": 0,
    "candidate_metrics_column_fallbacks": 0,
    "candidate_metrics_fast_success": 0,
    "candidate_metrics_fast_failures": 0,
    "candidate_metrics_fallbacks": 0,
    "final_panel_column_fast_success": 0,
    "final_panel_column_fast_failures": 0,
    "final_panel_column_fallbacks": 0,
    "final_panel_rows_fast_success": 0,
    "final_panel_rows_fast_failures": 0,
    "final_panel_rows_fallbacks": 0,
    "review_required_column_fast_success": 0,
    "review_required_column_fast_failures": 0,
    "review_required_column_fallbacks": 0,
    "source_family_fast_success": 0,
    "source_family_fast_failures": 0,
    "source_family_fallbacks": 0,
    "component_id_fast_success": 0,
    "component_id_fast_failures": 0,
    "component_id_fallbacks": 0,
    "merge_intervals_fast_success": 0,
    "merge_intervals_fast_failures": 0,
    "merge_intervals_fallbacks": 0,
    "pairwise_alias_diagnostics_fast_success": 0,
    "pairwise_alias_diagnostics_fast_failures": 0,
    "pairwise_alias_diagnostics_fallbacks": 0,
    "conventional_components_meta_fast_success": 0,
    "conventional_components_meta_fast_failures": 0,
    "conventional_components_meta_fallbacks": 0,
    "conventional_components_fast_success": 0,
    "conventional_components_fast_failures": 0,
    "conventional_components_fallbacks": 0,
}


def get_refinitiv_authority_rust_accel_metrics() -> dict[str, int | str | bool | None]:
    metrics: dict[str, int | str | bool | None] = dict(_AUTHORITY_RUST_METRICS)
    metrics["rust_accel_available"] = _lm2011_rust is not None
    metrics["rust_accel_import_error"] = _AUTHORITY_RUST_IMPORT_ERROR
    return metrics


def reset_refinitiv_authority_rust_accel_metrics() -> None:
    for key in _AUTHORITY_RUST_METRICS:
        _AUTHORITY_RUST_METRICS[key] = 0


def _frame_column_values(df: pl.DataFrame, columns: list[str] | tuple[str, ...]) -> list[list[Any]]:
    return [df.get_column(column).to_list() for column in columns]


CANDIDATE_METRIC_COLUMNS: tuple[str, ...] = (
    "KYPERMNO",
    "candidate_ric",
    "candidate_source_family",
    "candidate_has_conventional_support",
    "candidate_has_effective_support",
    "candidate_has_ticker_support",
    "candidate_request_count",
    "candidate_request_with_data_count",
    "candidate_zero_return_request_count",
    "candidate_ownership_row_count",
    "candidate_ownership_date_count",
    "candidate_distinct_category_count",
    "candidate_first_ownership_date",
    "candidate_last_ownership_date",
    "candidate_bridge_start_date",
    "candidate_bridge_end_date",
    "candidate_bridge_span_day_count",
    "candidate_ownership_span_day_count",
    "candidate_ownership_date_share_within_permno",
    "candidate_ownership_row_share_within_permno",
    "candidate_internal_same_date_same_category_differing_value_count",
)

ALIAS_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "KYPERMNO",
    "left_candidate_ric",
    "right_candidate_ric",
    "left_candidate_source_family",
    "right_candidate_source_family",
    "left_candidate_has_effective_support",
    "right_candidate_has_effective_support",
    "pair_same_returned_ric",
    "pair_overlap_date_count",
    "pair_overlap_category_count",
    "pair_overlap_date_category_count",
    "pair_overlap_share_of_smaller_history",
    "pair_same_date_multi_ric_overlap_count",
    "pair_same_date_same_category_overlap_count",
    "pair_same_date_same_category_differing_value_count",
    "pair_mean_abs_value_diff",
    "pair_median_abs_value_diff",
    "pair_max_abs_value_diff",
    "pair_value_conflict_present",
    "pair_disjoint_history",
    "pair_benign_alias_supported",
    "pair_regime_split_supported",
    "pair_requires_review",
)

AUTHORITY_DECISION_COLUMNS: tuple[str, ...] = (
    "KYPERMNO",
    "permno_conventional_candidate_count",
    "permno_ticker_candidate_count",
    "permno_effective_candidate_count",
    "permno_conventional_ownership_date_count",
    "permno_ticker_ownership_date_count",
    "permno_conventional_ownership_row_count",
    "permno_ticker_ownership_row_count",
    "permno_same_date_multi_ric_overlap_count",
    "permno_same_date_same_category_differing_value_count",
    "permno_benign_alias_component_count",
    "permno_regime_component_count",
    "permno_regime_split_detected",
    "permno_ticker_support_candidate_count",
    "permno_ticker_support_strong_count",
    "authoritative_ric",
    "authoritative_source_family",
    "authoritative_component_id",
    "authoritative_ownership_date_count",
    "authoritative_ownership_row_count",
    "authoritative_coverage_share_of_best_candidate",
    "authoritative_coverage_share_of_permno",
    "authority_decision_status",
    "authority_decision_reason",
    "requires_review",
    "review_flag_conventional_identity_conflict",
    "review_flag_effective_overlap_conflict",
    "review_flag_multi_component_overlap",
    "review_flag_ticker_only_without_allowlist",
    "review_flag_zero_return_effective_only",
    "review_flag_meaningful_coverage_loss_if_static",
    "review_flag_unresolved_multi_ric_structure",
    "reviewed_ticker_allowlist_applied",
    "reviewed_ticker_allowlist_candidate_count",
)

AUTHORITY_EXCEPTION_COLUMNS: tuple[str, ...] = (
    "KYPERMNO",
    "authoritative_component_id",
    "authoritative_ric",
    "authoritative_source_family",
    "component_member_rics",
    "component_member_count",
    "authority_window_start_date",
    "authority_window_end_date",
    "authority_first_ownership_date",
    "authority_last_ownership_date",
    "authority_exception_status",
    "authority_exception_reason",
)

REVIEW_REQUIRED_COLUMNS: tuple[str, ...] = (
    *AUTHORITY_DECISION_COLUMNS,
    "conventional_candidate_rics",
    "ticker_candidate_rics",
)

TICKER_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "KYPERMNO",
    "candidate_ric",
    "candidate_source_family",
    "candidate_has_ticker_support",
    "candidate_has_conventional_support",
    "candidate_request_count",
    "candidate_request_with_data_count",
    "candidate_ownership_date_count",
    "candidate_ownership_row_count",
    "ticker_overlap_date_count_vs_authoritative",
    "ticker_overlap_category_count_vs_authoritative",
    "ticker_overlap_date_category_count_vs_authoritative",
    "ticker_same_date_same_category_differing_value_count_vs_authoritative",
    "ticker_benign_alias_supported_vs_authoritative",
    "ticker_allowlist_recommendation",
    "ticker_allowlist_reason",
)

FINAL_PANEL_COLUMNS: tuple[str, ...] = (
    "KYPERMNO",
    "authoritative_ric",
    "authoritative_source_family",
    "authoritative_component_id",
    "authority_decision_status",
    "source_candidate_ric",
    "returned_date",
    "returned_category",
    "returned_value",
)


def _candidate_metric_schema() -> dict[str, pl.DataType]:
    return {
        "KYPERMNO": pl.Utf8,
        "candidate_ric": pl.Utf8,
        "candidate_source_family": pl.Utf8,
        "candidate_has_conventional_support": pl.Boolean,
        "candidate_has_effective_support": pl.Boolean,
        "candidate_has_ticker_support": pl.Boolean,
        "candidate_request_count": pl.Int64,
        "candidate_request_with_data_count": pl.Int64,
        "candidate_zero_return_request_count": pl.Int64,
        "candidate_ownership_row_count": pl.Int64,
        "candidate_ownership_date_count": pl.Int64,
        "candidate_distinct_category_count": pl.Int64,
        "candidate_first_ownership_date": pl.Date,
        "candidate_last_ownership_date": pl.Date,
        "candidate_bridge_start_date": pl.Date,
        "candidate_bridge_end_date": pl.Date,
        "candidate_bridge_span_day_count": pl.Int64,
        "candidate_ownership_span_day_count": pl.Int64,
        "candidate_ownership_date_share_within_permno": pl.Float64,
        "candidate_ownership_row_share_within_permno": pl.Float64,
        "candidate_internal_same_date_same_category_differing_value_count": pl.Int64,
    }


def _alias_diagnostic_schema() -> dict[str, pl.DataType]:
    return {
        "KYPERMNO": pl.Utf8,
        "left_candidate_ric": pl.Utf8,
        "right_candidate_ric": pl.Utf8,
        "left_candidate_source_family": pl.Utf8,
        "right_candidate_source_family": pl.Utf8,
        "left_candidate_has_effective_support": pl.Boolean,
        "right_candidate_has_effective_support": pl.Boolean,
        "pair_same_returned_ric": pl.Boolean,
        "pair_overlap_date_count": pl.Int64,
        "pair_overlap_category_count": pl.Int64,
        "pair_overlap_date_category_count": pl.Int64,
        "pair_overlap_share_of_smaller_history": pl.Float64,
        "pair_same_date_multi_ric_overlap_count": pl.Int64,
        "pair_same_date_same_category_overlap_count": pl.Int64,
        "pair_same_date_same_category_differing_value_count": pl.Int64,
        "pair_mean_abs_value_diff": pl.Float64,
        "pair_median_abs_value_diff": pl.Float64,
        "pair_max_abs_value_diff": pl.Float64,
        "pair_value_conflict_present": pl.Boolean,
        "pair_disjoint_history": pl.Boolean,
        "pair_benign_alias_supported": pl.Boolean,
        "pair_regime_split_supported": pl.Boolean,
        "pair_requires_review": pl.Boolean,
    }


def _authority_decision_schema() -> dict[str, pl.DataType]:
    return {
        "KYPERMNO": pl.Utf8,
        "permno_conventional_candidate_count": pl.Int64,
        "permno_ticker_candidate_count": pl.Int64,
        "permno_effective_candidate_count": pl.Int64,
        "permno_conventional_ownership_date_count": pl.Int64,
        "permno_ticker_ownership_date_count": pl.Int64,
        "permno_conventional_ownership_row_count": pl.Int64,
        "permno_ticker_ownership_row_count": pl.Int64,
        "permno_same_date_multi_ric_overlap_count": pl.Int64,
        "permno_same_date_same_category_differing_value_count": pl.Int64,
        "permno_benign_alias_component_count": pl.Int64,
        "permno_regime_component_count": pl.Int64,
        "permno_regime_split_detected": pl.Boolean,
        "permno_ticker_support_candidate_count": pl.Int64,
        "permno_ticker_support_strong_count": pl.Int64,
        "authoritative_ric": pl.Utf8,
        "authoritative_source_family": pl.Utf8,
        "authoritative_component_id": pl.Utf8,
        "authoritative_ownership_date_count": pl.Int64,
        "authoritative_ownership_row_count": pl.Int64,
        "authoritative_coverage_share_of_best_candidate": pl.Float64,
        "authoritative_coverage_share_of_permno": pl.Float64,
        "authority_decision_status": pl.Utf8,
        "authority_decision_reason": pl.Utf8,
        "requires_review": pl.Boolean,
        "review_flag_conventional_identity_conflict": pl.Boolean,
        "review_flag_effective_overlap_conflict": pl.Boolean,
        "review_flag_multi_component_overlap": pl.Boolean,
        "review_flag_ticker_only_without_allowlist": pl.Boolean,
        "review_flag_zero_return_effective_only": pl.Boolean,
        "review_flag_meaningful_coverage_loss_if_static": pl.Boolean,
        "review_flag_unresolved_multi_ric_structure": pl.Boolean,
        "reviewed_ticker_allowlist_applied": pl.Boolean,
        "reviewed_ticker_allowlist_candidate_count": pl.Int64,
    }


def _authority_exception_schema() -> dict[str, pl.DataType]:
    return {
        "KYPERMNO": pl.Utf8,
        "authoritative_component_id": pl.Utf8,
        "authoritative_ric": pl.Utf8,
        "authoritative_source_family": pl.Utf8,
        "component_member_rics": pl.Utf8,
        "component_member_count": pl.Int64,
        "authority_window_start_date": pl.Date,
        "authority_window_end_date": pl.Date,
        "authority_first_ownership_date": pl.Date,
        "authority_last_ownership_date": pl.Date,
        "authority_exception_status": pl.Utf8,
        "authority_exception_reason": pl.Utf8,
    }


def _review_required_schema() -> dict[str, pl.DataType]:
    schema = _authority_decision_schema()
    schema.update(
        {
            "conventional_candidate_rics": pl.Utf8,
            "ticker_candidate_rics": pl.Utf8,
        }
    )
    return schema


def _ticker_candidate_schema() -> dict[str, pl.DataType]:
    return {
        "KYPERMNO": pl.Utf8,
        "candidate_ric": pl.Utf8,
        "candidate_source_family": pl.Utf8,
        "candidate_has_ticker_support": pl.Boolean,
        "candidate_has_conventional_support": pl.Boolean,
        "candidate_request_count": pl.Int64,
        "candidate_request_with_data_count": pl.Int64,
        "candidate_ownership_date_count": pl.Int64,
        "candidate_ownership_row_count": pl.Int64,
        "ticker_overlap_date_count_vs_authoritative": pl.Int64,
        "ticker_overlap_category_count_vs_authoritative": pl.Int64,
        "ticker_overlap_date_category_count_vs_authoritative": pl.Int64,
        "ticker_same_date_same_category_differing_value_count_vs_authoritative": pl.Int64,
        "ticker_benign_alias_supported_vs_authoritative": pl.Boolean,
        "ticker_allowlist_recommendation": pl.Boolean,
        "ticker_allowlist_reason": pl.Utf8,
    }


def _final_panel_schema() -> dict[str, pl.DataType]:
    return {
        "KYPERMNO": pl.Utf8,
        "authoritative_ric": pl.Utf8,
        "authoritative_source_family": pl.Utf8,
        "authoritative_component_id": pl.Utf8,
        "authority_decision_status": pl.Utf8,
        "source_candidate_ric": pl.Utf8,
        "returned_date": pl.Date,
        "returned_category": pl.Utf8,
        "returned_value": pl.Float64,
    }


def _empty_df(columns: tuple[str, ...], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return _build_explicit_schema_df([], schema).select(columns)


def _date_span_days_py(start_date: dt.date | None, end_date: dt.date | None) -> int:
    if start_date is None or end_date is None or end_date < start_date:
        return 0
    return (end_date - start_date).days + 1


def _date_span_days(start_date: dt.date | None, end_date: dt.date | None) -> int:
    if _lm2011_rust is not None:
        try:
            out = int(
                _lm2011_rust.refinitiv_authority_date_span_days(
                    None if start_date is None else start_date.toordinal(),
                    None if end_date is None else end_date.toordinal(),
                )
            )
            _AUTHORITY_RUST_METRICS["date_span_fast_success"] += 1
            return out
        except Exception:
            _AUTHORITY_RUST_METRICS["date_span_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["date_span_fallbacks"] += 1
    return _date_span_days_py(start_date, end_date)


def _values_match_py(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= VALUE_EQUALITY_TOLERANCE


def _values_match(left: float | None, right: float | None) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(_lm2011_rust.refinitiv_authority_values_match(left, right, VALUE_EQUALITY_TOLERANCE))
            _AUTHORITY_RUST_METRICS["values_match_fast_success"] += 1
            return out
        except Exception:
            _AUTHORITY_RUST_METRICS["values_match_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["values_match_fallbacks"] += 1
    return _values_match_py(left, right)


def _distinct_values_py(values: set[float | None]) -> list[float]:
    return sorted(value for value in values if value is not None)


def _distinct_values(values: set[float | None]) -> list[float]:
    if _lm2011_rust is not None:
        try:
            out = list(_lm2011_rust.refinitiv_authority_distinct_values(values))
            _AUTHORITY_RUST_METRICS["distinct_values_fast_success"] += 1
            return out
        except Exception:
            _AUTHORITY_RUST_METRICS["distinct_values_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["distinct_values_fallbacks"] += 1
    return _distinct_values_py(values)


def _source_family_from_flags_py(has_conventional_support: bool, has_ticker_support: bool) -> str:
    if has_conventional_support and has_ticker_support:
        return "MIXED"
    if has_conventional_support:
        return "CONVENTIONAL"
    return "TICKER"


def _source_family_from_flags(has_conventional_support: bool, has_ticker_support: bool) -> str:
    if _lm2011_rust is not None:
        try:
            out = str(
                _lm2011_rust.refinitiv_authority_source_family_from_flags(
                    bool(has_conventional_support),
                    bool(has_ticker_support),
                )
            )
            _AUTHORITY_RUST_METRICS["source_family_fast_success"] += 1
            return out
        except Exception:
            _AUTHORITY_RUST_METRICS["source_family_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["source_family_fallbacks"] += 1
    return _source_family_from_flags_py(has_conventional_support, has_ticker_support)


def _component_id_py(kypermno: str, component_index: int) -> str:
    return f"{kypermno}|COMPONENT|{component_index:02d}"


def _component_id(kypermno: str, component_index: int) -> str:
    if _lm2011_rust is not None:
        try:
            out = str(_lm2011_rust.refinitiv_authority_component_id(str(kypermno), int(component_index)))
            _AUTHORITY_RUST_METRICS["component_id_fast_success"] += 1
            return out
        except Exception:
            _AUTHORITY_RUST_METRICS["component_id_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["component_id_fallbacks"] += 1
    return _component_id_py(kypermno, component_index)


def _candidate_key_py(record: dict[str, Any]) -> tuple[str | None, str | None]:
    return (
        _normalize_lookup_text(record.get("KYPERMNO")),
        _normalize_lookup_text(record.get("candidate_ric")),
    )


def _candidate_key(record: dict[str, Any]) -> tuple[str | None, str | None]:
    if _lm2011_rust is not None:
        try:
            key = _lm2011_rust.refinitiv_authority_candidate_key(
                record.get("KYPERMNO"),
                record.get("candidate_ric"),
            )
            _AUTHORITY_RUST_METRICS["candidate_key_fast_success"] += 1
            return key[0], key[1]
        except Exception:
            _AUTHORITY_RUST_METRICS["candidate_key_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["candidate_key_fallbacks"] += 1
    return _candidate_key_py(record)


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _merge_intervals_py(
    intervals: list[tuple[dt.date, dt.date]],
    *,
    max_gap_days: int,
) -> list[tuple[dt.date, dt.date]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged: list[tuple[dt.date, dt.date]] = [sorted_intervals[0]]
    for start_date, end_date in sorted_intervals[1:]:
        prior_start, prior_end = merged[-1]
        if (start_date - prior_end).days <= max_gap_days + 1:
            merged[-1] = (prior_start, max(prior_end, end_date))
        else:
            merged.append((start_date, end_date))
    return merged


def _merge_intervals(
    intervals: list[tuple[dt.date, dt.date]],
    *,
    max_gap_days: int,
) -> list[tuple[dt.date, dt.date]]:
    if not intervals:
        return []
    if _lm2011_rust is not None:
        try:
            ordinal_intervals = [(start_date.toordinal(), end_date.toordinal()) for start_date, end_date in intervals]
            merged = _lm2011_rust.refinitiv_authority_merge_intervals(ordinal_intervals, int(max_gap_days))
            _AUTHORITY_RUST_METRICS["merge_intervals_fast_success"] += 1
            return [(dt.date.fromordinal(int(start)), dt.date.fromordinal(int(end))) for start, end in merged]
        except Exception:
            _AUTHORITY_RUST_METRICS["merge_intervals_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["merge_intervals_fallbacks"] += 1
    return _merge_intervals_py(intervals, max_gap_days=max_gap_days)


def _normalize_resolution_df(resolution_df: pl.DataFrame) -> pl.DataFrame:
    return _cast_df_to_schema(
        resolution_df.select(RIC_LOOKUP_RESOLUTION_OUTPUT_COLUMNS),
        _resolution_output_schema(),
    ).select(RIC_LOOKUP_RESOLUTION_OUTPUT_COLUMNS)


def _normalize_results_df(results_df: pl.DataFrame) -> pl.DataFrame:
    return _cast_df_to_schema(
        results_df.select(OWNERSHIP_UNIVERSE_RESULTS_COLUMNS),
        _ownership_universe_results_schema(),
    ).select(OWNERSHIP_UNIVERSE_RESULTS_COLUMNS)


def _normalize_row_summary_df(row_summary_df: pl.DataFrame) -> pl.DataFrame:
    return _cast_df_to_schema(
        row_summary_df.select(OWNERSHIP_UNIVERSE_ROW_SUMMARY_COLUMNS),
        _ownership_universe_row_summary_schema(),
    ).select(OWNERSHIP_UNIVERSE_ROW_SUMMARY_COLUMNS)


def _read_ownership_universe_results_artifact(parquet_path: Path | str) -> pl.DataFrame:
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"ownership universe results artifact not found: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    missing = [name for name in OWNERSHIP_UNIVERSE_RESULTS_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"ownership universe results artifact missing required columns: {missing}")
    return _normalize_results_df(df)


def _read_ownership_universe_row_summary_artifact(parquet_path: Path | str) -> pl.DataFrame:
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"ownership universe row-summary artifact not found: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    missing = [name for name in OWNERSHIP_UNIVERSE_ROW_SUMMARY_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"ownership universe row-summary artifact missing required columns: {missing}")
    return _normalize_row_summary_df(df)


def _normalize_allowlist_df(allowlist_df: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "KYPERMNO": pl.Utf8,
        "candidate_ric": pl.Utf8,
        "allowlist_reason": pl.Utf8,
    }
    if allowlist_df.height == 0:
        return _build_explicit_schema_df([], schema).select(list(schema))
    normalized = allowlist_df
    if "allowlist_reason" not in normalized.columns:
        normalized = normalized.with_columns(pl.lit(None, dtype=pl.Utf8).alias("allowlist_reason"))
    return _cast_df_to_schema(normalized.select(list(schema)), schema).select(list(schema))


def _read_reviewed_ticker_allowlist_artifact(parquet_path: Path | str | None) -> pl.DataFrame:
    if parquet_path is None:
        return _normalize_allowlist_df(pl.DataFrame(schema={"KYPERMNO": pl.Utf8, "candidate_ric": pl.Utf8}))
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        return _normalize_allowlist_df(pl.DataFrame(schema={"KYPERMNO": pl.Utf8, "candidate_ric": pl.Utf8}))
    df = pl.read_parquet(parquet_path)
    missing = [name for name in ("KYPERMNO", "candidate_ric") if name not in df.columns]
    if missing:
        raise ValueError(f"reviewed ticker allowlist missing required columns: {missing}")
    return _normalize_allowlist_df(df)


def _allowlist_keys_py(allowlist_df: pl.DataFrame) -> set[tuple[str | None, str | None]]:
    return {
        (
            _normalize_lookup_text(row.get("KYPERMNO")),
            _normalize_lookup_text(row.get("candidate_ric")),
        )
        for row in allowlist_df.to_dicts()
    }


def _allowlist_keys(allowlist_df: pl.DataFrame) -> set[tuple[str | None, str | None]]:
    if _lm2011_rust is not None:
        try:
            keys = _lm2011_rust.refinitiv_authority_allowlist_key_columns(
                allowlist_df.get_column("KYPERMNO").to_list(),
                allowlist_df.get_column("candidate_ric").to_list(),
            )
            _AUTHORITY_RUST_METRICS["allowlist_keys_column_fast_success"] += 1
            return {(key[0], key[1]) for key in keys}
        except Exception:
            _AUTHORITY_RUST_METRICS["allowlist_keys_column_fast_failures"] += 1
            _AUTHORITY_RUST_METRICS["allowlist_keys_column_fallbacks"] += 1
        try:
            keys = _lm2011_rust.refinitiv_authority_allowlist_keys(allowlist_df.to_dicts())
            _AUTHORITY_RUST_METRICS["allowlist_keys_fast_success"] += 1
            return {(key[0], key[1]) for key in keys}
        except Exception:
            _AUTHORITY_RUST_METRICS["allowlist_keys_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["allowlist_keys_fallbacks"] += 1
    return _allowlist_keys_py(allowlist_df)


def _build_candidate_metrics_py(
    row_summary_records: list[dict[str, Any]],
    results_records: list[dict[str, Any]],
) -> tuple[pl.DataFrame, dict[tuple[str, str], dict[str, Any]], dict[str, set[dt.date]], dict[str, set[tuple[dt.date, str, float | None]]]]:
    row_summary_by_candidate: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in row_summary_records:
        key = _candidate_key(record)
        if key[0] is not None and key[1] is not None:
            row_summary_by_candidate[(key[0], key[1])].append(record)

    results_by_candidate: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in results_records:
        key = _candidate_key(record)
        if key[0] is not None and key[1] is not None:
            results_by_candidate[(key[0], key[1])].append(record)

    rows: list[dict[str, Any]] = []
    candidate_meta: dict[tuple[str, str], dict[str, Any]] = {}
    permno_date_sets: dict[str, set[dt.date]] = defaultdict(set)
    permno_row_sets: dict[str, set[tuple[dt.date, str, float | None]]] = defaultdict(set)

    for candidate_key in sorted(set(row_summary_by_candidate) | set(results_by_candidate)):
        kypermno, candidate_ric = candidate_key
        request_rows = row_summary_by_candidate.get(candidate_key, [])
        result_rows = results_by_candidate.get(candidate_key, [])
        role_set = {
            _normalize_lookup_text(row.get("ownership_lookup_role"))
            for row in request_rows
            if _normalize_lookup_text(row.get("ownership_lookup_role")) is not None
        }
        has_conventional_support = any(role in CONVENTIONAL_OWNERSHIP_LOOKUP_ROLES for role in role_set)
        has_effective_support = "UNIVERSE_EFFECTIVE" in role_set
        has_ticker_support = TICKER_OWNERSHIP_LOOKUP_ROLE in role_set
        observation_value_sets: dict[tuple[dt.date, str], set[float | None]] = defaultdict(set)
        unique_row_set: set[tuple[dt.date, str, float | None]] = set()
        for row in result_rows:
            returned_date = row.get("returned_date")
            returned_category = _normalize_lookup_text(row.get("returned_category"))
            if returned_date is None or returned_category is None:
                continue
            returned_value = row.get("returned_value")
            observation_value_sets[(returned_date, returned_category)].add(returned_value)
            unique_row_set.add((returned_date, returned_category, returned_value))

        ownership_dates = {date_value for date_value, _ in observation_value_sets}
        ownership_categories = {category for _, category in observation_value_sets}
        bridge_start_date = min(
            (row.get("first_seen_caldt") for row in request_rows if isinstance(row.get("first_seen_caldt"), dt.date)),
            default=None,
        )
        bridge_end_date = max(
            (row.get("last_seen_caldt") for row in request_rows if isinstance(row.get("last_seen_caldt"), dt.date)),
            default=None,
        )
        row = {
            "KYPERMNO": kypermno,
            "candidate_ric": candidate_ric,
            "candidate_source_family": _source_family_from_flags(has_conventional_support, has_ticker_support),
            "candidate_has_conventional_support": has_conventional_support,
            "candidate_has_effective_support": has_effective_support,
            "candidate_has_ticker_support": has_ticker_support,
            "candidate_request_count": len(request_rows),
            "candidate_request_with_data_count": int(
                sum(1 for request_row in request_rows if int(request_row.get("ownership_rows_returned") or 0) > 0)
            ),
            "candidate_zero_return_request_count": int(
                sum(
                    1
                    for request_row in request_rows
                    if bool(request_row.get("retrieval_row_present"))
                    and int(request_row.get("ownership_rows_returned") or 0) == 0
                )
            ),
            "candidate_ownership_row_count": len(unique_row_set),
            "candidate_ownership_date_count": len(ownership_dates),
            "candidate_distinct_category_count": len(ownership_categories),
            "candidate_first_ownership_date": min(ownership_dates, default=None),
            "candidate_last_ownership_date": max(ownership_dates, default=None),
            "candidate_bridge_start_date": bridge_start_date,
            "candidate_bridge_end_date": bridge_end_date,
            "candidate_bridge_span_day_count": _date_span_days(bridge_start_date, bridge_end_date),
            "candidate_ownership_span_day_count": _date_span_days(
                min(ownership_dates, default=None),
                max(ownership_dates, default=None),
            ),
            "candidate_ownership_date_share_within_permno": 0.0,
            "candidate_ownership_row_share_within_permno": 0.0,
            "candidate_internal_same_date_same_category_differing_value_count": int(
                sum(1 for values in observation_value_sets.values() if len(_distinct_values(values)) > 1)
            ),
        }
        rows.append(row)
        candidate_meta[candidate_key] = {
            **row,
            "role_set": role_set,
            "request_rows": request_rows,
            "result_rows": result_rows,
            "observation_value_sets": observation_value_sets,
            "unique_row_set": unique_row_set,
        }
        permno_date_sets[kypermno].update(ownership_dates)
        permno_row_sets[kypermno].update(unique_row_set)

    for row in rows:
        kypermno = str(row["KYPERMNO"])
        total_dates = len(permno_date_sets.get(kypermno, set()))
        total_rows = len(permno_row_sets.get(kypermno, set()))
        row["candidate_ownership_date_share_within_permno"] = (
            float(row["candidate_ownership_date_count"]) / float(total_dates) if total_dates > 0 else 0.0
        )
        row["candidate_ownership_row_share_within_permno"] = (
            float(row["candidate_ownership_row_count"]) / float(total_rows) if total_rows > 0 else 0.0
        )
        candidate_meta[(kypermno, str(row["candidate_ric"]))]["candidate_ownership_date_share_within_permno"] = row[
            "candidate_ownership_date_share_within_permno"
        ]
        candidate_meta[(kypermno, str(row["candidate_ric"]))]["candidate_ownership_row_share_within_permno"] = row[
            "candidate_ownership_row_share_within_permno"
        ]

    df = _build_explicit_schema_df(rows, _candidate_metric_schema()).select(CANDIDATE_METRIC_COLUMNS)
    return df, candidate_meta, permno_date_sets, permno_row_sets


def _candidate_metrics_from_rust_records(
    records: list[dict[str, Any]],
) -> tuple[pl.DataFrame, dict[tuple[str, str], dict[str, Any]], dict[str, set[dt.date]], dict[str, set[tuple[dt.date, str, float | None]]]]:
    rows: list[dict[str, Any]] = []
    candidate_meta: dict[tuple[str, str], dict[str, Any]] = {}
    permno_date_sets: dict[str, set[dt.date]] = defaultdict(set)
    permno_row_sets: dict[str, set[tuple[dt.date, str, float | None]]] = defaultdict(set)

    for record in records:
        row = {name: record.get(name) for name in CANDIDATE_METRIC_COLUMNS}
        kypermno = str(row["KYPERMNO"])
        candidate_ric = str(row["candidate_ric"])
        observation_value_sets: dict[tuple[dt.date, str], set[float | None]] = defaultdict(set)
        for date_value, category, values in record.get("observation_value_sets", []):
            observation_value_sets[(date_value, str(category))] = set(values)
        unique_row_set = {
            (date_value, str(category), value)
            for date_value, category, value in record.get("unique_row_set", [])
        }
        meta_row: dict[str, Any] = {
            **row,
            "role_set": set(record.get("role_set", [])),
            "observation_value_sets": observation_value_sets,
            "unique_row_set": unique_row_set,
        }
        if "request_rows" in record:
            meta_row["request_rows"] = list(record.get("request_rows", []))
        if "result_rows" in record:
            meta_row["result_rows"] = list(record.get("result_rows", []))
        candidate_meta[(kypermno, candidate_ric)] = meta_row
        rows.append(row)
        permno_date_sets[kypermno].update(date_value for date_value, _ in observation_value_sets)
        permno_row_sets[kypermno].update(unique_row_set)

    df = _build_explicit_schema_df(rows, _candidate_metric_schema()).select(CANDIDATE_METRIC_COLUMNS)
    return df, candidate_meta, permno_date_sets, permno_row_sets


def _build_candidate_metrics(
    row_summary_records: list[dict[str, Any]],
    results_records: list[dict[str, Any]],
) -> tuple[pl.DataFrame, dict[tuple[str, str], dict[str, Any]], dict[str, set[dt.date]], dict[str, set[tuple[dt.date, str, float | None]]]]:
    if _lm2011_rust is not None:
        try:
            raw_records = _lm2011_rust.refinitiv_authority_candidate_metric_records(
                row_summary_records,
                results_records,
                list(CONVENTIONAL_OWNERSHIP_LOOKUP_ROLES),
                TICKER_OWNERSHIP_LOOKUP_ROLE,
            )
            _AUTHORITY_RUST_METRICS["candidate_metrics_fast_success"] += 1
            return _candidate_metrics_from_rust_records([dict(record) for record in raw_records])
        except Exception:
            _AUTHORITY_RUST_METRICS["candidate_metrics_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["candidate_metrics_fallbacks"] += 1
    return _build_candidate_metrics_py(row_summary_records, results_records)


def _build_candidate_metrics_from_frames(
    row_summary_df: pl.DataFrame,
    results_df: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[tuple[str, str], dict[str, Any]], dict[str, set[dt.date]], dict[str, set[tuple[dt.date, str, float | None]]]]:
    if _lm2011_rust is not None:
        try:
            raw_records = _lm2011_rust.refinitiv_authority_candidate_metric_record_columns(
                row_summary_df.columns,
                _frame_column_values(row_summary_df, row_summary_df.columns),
                results_df.columns,
                _frame_column_values(results_df, results_df.columns),
                list(CONVENTIONAL_OWNERSHIP_LOOKUP_ROLES),
                TICKER_OWNERSHIP_LOOKUP_ROLE,
            )
            _AUTHORITY_RUST_METRICS["candidate_metrics_column_fast_success"] += 1
            return _candidate_metrics_from_rust_records([dict(record) for record in raw_records])
        except Exception:
            _AUTHORITY_RUST_METRICS["candidate_metrics_column_fast_failures"] += 1
            _AUTHORITY_RUST_METRICS["candidate_metrics_column_fallbacks"] += 1
    return _build_candidate_metrics(row_summary_df.to_dicts(), results_df.to_dicts())


def _authority_final_panel_rows_py(
    results_records: list[dict[str, Any]],
    assignment_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    assignments_by_permno: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    permno_order: list[str] = []
    for assignment in assignment_records:
        kypermno = _normalize_lookup_text(assignment.get("KYPERMNO"))
        source_candidate_ric = _normalize_lookup_text(assignment.get("source_candidate_ric"))
        if kypermno is None or source_candidate_ric is None:
            continue
        if kypermno not in assignments_by_permno:
            permno_order.append(kypermno)
        assignments_by_permno[kypermno][source_candidate_ric] = assignment

    results_rows_by_permno: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result_row in results_records:
        kypermno = _normalize_lookup_text(result_row.get("KYPERMNO"))
        if kypermno is not None and kypermno in assignments_by_permno:
            results_rows_by_permno[kypermno].append(result_row)

    final_panel_rows: list[dict[str, Any]] = []
    conflict_permnos: set[str] = set()
    for kypermno in permno_order:
        selected_meta = assignments_by_permno.get(kypermno, {})
        grouped_rows: dict[tuple[dt.date, str], list[dict[str, Any]]] = defaultdict(list)
        for result_row in results_rows_by_permno.get(kypermno, []):
            source_candidate_ric = _normalize_lookup_text(result_row.get("candidate_ric"))
            returned_date = result_row.get("returned_date")
            returned_category = _normalize_lookup_text(result_row.get("returned_category"))
            if source_candidate_ric not in selected_meta or returned_date is None or returned_category is None:
                continue
            grouped_rows[(returned_date, returned_category)].append(result_row)

        panel_conflict_detected = False
        panel_rows_for_permno: list[dict[str, Any]] = []
        for (returned_date, returned_category), group_rows in grouped_rows.items():
            unique_values: list[float] = []
            for group_row in group_rows:
                returned_value = group_row.get("returned_value")
                if returned_value is None:
                    continue
                if not any(_values_match(float(returned_value), seen_value) for seen_value in unique_values):
                    unique_values.append(float(returned_value))
            if len(unique_values) > 1:
                panel_conflict_detected = True
                break
            preferred_row = min(
                group_rows,
                key=lambda row: (
                    0
                    if selected_meta[str(row["candidate_ric"])]["authoritative_source_family"] == "CONVENTIONAL"
                    else 1,
                    0 if _normalize_lookup_text(row.get("ownership_lookup_role")) == "UNIVERSE_EFFECTIVE" else 1,
                    _normalize_lookup_text(row.get("candidate_ric")) or "",
                ),
            )
            source_candidate_ric = _normalize_lookup_text(preferred_row.get("candidate_ric"))
            authoritative_assignment = selected_meta.get(source_candidate_ric or "")
            if authoritative_assignment is None:
                continue
            panel_rows_for_permno.append(
                {
                    "KYPERMNO": kypermno,
                    "authoritative_ric": authoritative_assignment.get("authoritative_ric"),
                    "authoritative_source_family": authoritative_assignment.get("authoritative_source_family"),
                    "authoritative_component_id": authoritative_assignment.get("authoritative_component_id"),
                    "authority_decision_status": authoritative_assignment.get("authority_decision_status"),
                    "source_candidate_ric": source_candidate_ric,
                    "returned_date": returned_date,
                    "returned_category": returned_category,
                    "returned_value": None if not unique_values else unique_values[0],
                }
            )

        if panel_conflict_detected:
            conflict_permnos.add(kypermno)
        else:
            final_panel_rows.extend(panel_rows_for_permno)

    return final_panel_rows, conflict_permnos


def _authority_final_panel_rows(
    results_records: list[dict[str, Any]],
    assignment_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    if _lm2011_rust is not None:
        try:
            raw_rows, raw_conflicts = _lm2011_rust.refinitiv_authority_final_panel_rows(
                results_records,
                assignment_records,
            )
            _AUTHORITY_RUST_METRICS["final_panel_rows_fast_success"] += 1
            return [dict(row) for row in raw_rows], set(str(value) for value in raw_conflicts)
        except Exception:
            _AUTHORITY_RUST_METRICS["final_panel_rows_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["final_panel_rows_fallbacks"] += 1
    return _authority_final_panel_rows_py(results_records, assignment_records)


def _authority_final_panel_rows_from_frame(
    results_df: pl.DataFrame,
    assignment_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    if not assignment_records:
        return [], set()
    if _lm2011_rust is not None:
        try:
            raw_rows, raw_conflicts = _lm2011_rust.refinitiv_authority_final_panel_rows_columns(
                results_df.columns,
                _frame_column_values(results_df, results_df.columns),
                assignment_records,
            )
            _AUTHORITY_RUST_METRICS["final_panel_column_fast_success"] += 1
            return [dict(row) for row in raw_rows], set(str(value) for value in raw_conflicts)
        except Exception:
            _AUTHORITY_RUST_METRICS["final_panel_column_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["final_panel_column_fallbacks"] += 1
    return _authority_final_panel_rows(results_df.to_dicts(), assignment_records)


def _review_required_rows_py(
    decision_df: pl.DataFrame,
    candidate_metrics_df: pl.DataFrame,
) -> list[dict[str, Any]]:
    candidate_rows_by_permno: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_metrics_df.to_dicts():
        kypermno = _normalize_lookup_text(candidate.get("KYPERMNO"))
        if kypermno is not None:
            candidate_rows_by_permno[kypermno].append(candidate)

    review_required_rows = []
    for decision_row in decision_df.to_dicts():
        if not bool(decision_row.get("requires_review")):
            continue
        kypermno = _normalize_lookup_text(decision_row.get("KYPERMNO"))
        conventional_rics = sorted(
            candidate["candidate_ric"]
            for candidate in candidate_rows_by_permno.get(kypermno or "", [])
            if candidate["KYPERMNO"] == kypermno and bool(candidate["candidate_has_conventional_support"])
        )
        ticker_rics = sorted(
            candidate["candidate_ric"]
            for candidate in candidate_rows_by_permno.get(kypermno or "", [])
            if candidate["KYPERMNO"] == kypermno and bool(candidate["candidate_has_ticker_support"])
        )
        review_required_rows.append(
            {
                **decision_row,
                "conventional_candidate_rics": "|".join(conventional_rics),
                "ticker_candidate_rics": "|".join(ticker_rics),
            }
        )
    return review_required_rows


def _review_required_rows_from_frames(
    decision_df: pl.DataFrame,
    candidate_metrics_df: pl.DataFrame,
) -> list[dict[str, Any]]:
    if _lm2011_rust is not None:
        try:
            raw_rows = _lm2011_rust.refinitiv_authority_review_required_rows_columns(
                decision_df.columns,
                _frame_column_values(decision_df, decision_df.columns),
                candidate_metrics_df.columns,
                _frame_column_values(candidate_metrics_df, candidate_metrics_df.columns),
                list(AUTHORITY_DECISION_COLUMNS),
            )
            _AUTHORITY_RUST_METRICS["review_required_column_fast_success"] += 1
            return [dict(row) for row in raw_rows]
        except Exception:
            _AUTHORITY_RUST_METRICS["review_required_column_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["review_required_column_fallbacks"] += 1
    return _review_required_rows_py(decision_df, candidate_metrics_df)


def _build_pairwise_alias_diagnostics_py(
    *,
    permno_order: list[str],
    candidate_meta: dict[tuple[str, str], dict[str, Any]],
) -> tuple[pl.DataFrame, dict[tuple[str, str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pair_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    for kypermno in permno_order:
        candidate_keys = sorted(key for key in candidate_meta if key[0] == kypermno)
        for (_, left_ric), (_, right_ric) in combinations(candidate_keys, 2):
            left = candidate_meta[(kypermno, left_ric)]
            right = candidate_meta[(kypermno, right_ric)]
            left_keys = set(left["observation_value_sets"])
            right_keys = set(right["observation_value_sets"])
            overlap_pairs = left_keys & right_keys
            overlap_dates = {date_value for date_value, _ in overlap_pairs}
            overlap_categories = {category for _, category in overlap_pairs}
            diff_values: list[float] = []
            differing_value_count = 0
            for overlap_key in overlap_pairs:
                left_values = _distinct_values(left["observation_value_sets"][overlap_key])
                right_values = _distinct_values(right["observation_value_sets"][overlap_key])
                if len(left_values) == 1 and len(right_values) == 1:
                    abs_diff = abs(left_values[0] - right_values[0])
                    diff_values.append(abs_diff)
                    if abs_diff > VALUE_EQUALITY_TOLERANCE:
                        differing_value_count += 1
                elif left_values != right_values:
                    differing_value_count += 1
            overlap_share = (
                float(len(overlap_pairs)) / float(min(len(left_keys), len(right_keys)))
                if min(len(left_keys), len(right_keys)) > 0
                else 0.0
            )
            observed_disjoint = (
                left["candidate_first_ownership_date"] is not None
                and left["candidate_last_ownership_date"] is not None
                and right["candidate_first_ownership_date"] is not None
                and right["candidate_last_ownership_date"] is not None
                and (
                    left["candidate_last_ownership_date"] < right["candidate_first_ownership_date"]
                    or right["candidate_last_ownership_date"] < left["candidate_first_ownership_date"]
                )
            )
            bridge_disjoint = (
                left["candidate_bridge_start_date"] is not None
                and left["candidate_bridge_end_date"] is not None
                and right["candidate_bridge_start_date"] is not None
                and right["candidate_bridge_end_date"] is not None
                and (
                    left["candidate_bridge_end_date"] < right["candidate_bridge_start_date"]
                    or right["candidate_bridge_end_date"] < left["candidate_bridge_start_date"]
                )
            )
            pair_value_conflict_present = differing_value_count > 0
            pair_benign_alias_supported = (
                not pair_value_conflict_present
                and len(overlap_pairs) >= BENIGN_ALIAS_MIN_OVERLAP_DATE_CATEGORY_COUNT
                and len(overlap_dates) >= BENIGN_ALIAS_MIN_OVERLAP_DATE_COUNT
                and len(overlap_categories) >= BENIGN_ALIAS_MIN_OVERLAP_CATEGORY_COUNT
                and overlap_share >= BENIGN_ALIAS_MIN_OVERLAP_SHARE_OF_SMALLER_HISTORY
            )
            pair_regime_split_supported = (
                bool(left["candidate_has_conventional_support"])
                and bool(right["candidate_has_conventional_support"])
                and not pair_value_conflict_present
                and (observed_disjoint or bridge_disjoint)
            )
            pair_requires_review = (
                pair_value_conflict_present
                or int(left["candidate_internal_same_date_same_category_differing_value_count"]) > 0
                or int(right["candidate_internal_same_date_same_category_differing_value_count"]) > 0
                or (
                    len(overlap_pairs) > 0
                    and not pair_benign_alias_supported
                    and not pair_regime_split_supported
                )
            )
            row = {
                "KYPERMNO": kypermno,
                "left_candidate_ric": left_ric,
                "right_candidate_ric": right_ric,
                "left_candidate_source_family": left["candidate_source_family"],
                "right_candidate_source_family": right["candidate_source_family"],
                "left_candidate_has_effective_support": left["candidate_has_effective_support"],
                "right_candidate_has_effective_support": right["candidate_has_effective_support"],
                "pair_same_returned_ric": left_ric == right_ric,
                "pair_overlap_date_count": len(overlap_dates),
                "pair_overlap_category_count": len(overlap_categories),
                "pair_overlap_date_category_count": len(overlap_pairs),
                "pair_overlap_share_of_smaller_history": overlap_share,
                "pair_same_date_multi_ric_overlap_count": len(overlap_dates),
                "pair_same_date_same_category_overlap_count": len(overlap_pairs),
                "pair_same_date_same_category_differing_value_count": differing_value_count,
                "pair_mean_abs_value_diff": None if not diff_values else sum(diff_values) / float(len(diff_values)),
                "pair_median_abs_value_diff": None if not diff_values else float(median(diff_values)),
                "pair_max_abs_value_diff": None if not diff_values else max(diff_values),
                "pair_value_conflict_present": pair_value_conflict_present,
                "pair_disjoint_history": observed_disjoint or bridge_disjoint,
                "pair_benign_alias_supported": pair_benign_alias_supported,
                "pair_regime_split_supported": pair_regime_split_supported,
                "pair_requires_review": pair_requires_review,
            }
            rows.append(row)
            pair_meta[(kypermno, left_ric, right_ric)] = row
            pair_meta[(kypermno, right_ric, left_ric)] = row
    df = _build_explicit_schema_df(rows, _alias_diagnostic_schema()).select(ALIAS_DIAGNOSTIC_COLUMNS)
    return df, pair_meta


def _build_pairwise_alias_diagnostics(
    *,
    permno_order: list[str],
    candidate_meta: dict[tuple[str, str], dict[str, Any]],
) -> tuple[pl.DataFrame, dict[tuple[str, str, str], dict[str, Any]]]:
    if _lm2011_rust is not None:
        try:
            candidate_records = [
                {
                    "KYPERMNO": kypermno,
                    "candidate_ric": candidate_ric,
                    "candidate_source_family": meta["candidate_source_family"],
                    "candidate_has_conventional_support": meta["candidate_has_conventional_support"],
                    "candidate_has_effective_support": meta["candidate_has_effective_support"],
                    "candidate_first_ownership_date": meta["candidate_first_ownership_date"],
                    "candidate_last_ownership_date": meta["candidate_last_ownership_date"],
                    "candidate_bridge_start_date": meta["candidate_bridge_start_date"],
                    "candidate_bridge_end_date": meta["candidate_bridge_end_date"],
                    "candidate_internal_same_date_same_category_differing_value_count": meta[
                        "candidate_internal_same_date_same_category_differing_value_count"
                    ],
                    "observation_value_sets": meta["observation_value_sets"],
                }
                for (kypermno, candidate_ric), meta in candidate_meta.items()
            ]
            raw_rows = _lm2011_rust.refinitiv_authority_pairwise_alias_diagnostic_rows(
                list(permno_order),
                candidate_records,
                VALUE_EQUALITY_TOLERANCE,
                BENIGN_ALIAS_MIN_OVERLAP_DATE_CATEGORY_COUNT,
                BENIGN_ALIAS_MIN_OVERLAP_DATE_COUNT,
                BENIGN_ALIAS_MIN_OVERLAP_CATEGORY_COUNT,
                BENIGN_ALIAS_MIN_OVERLAP_SHARE_OF_SMALLER_HISTORY,
            )
            rows = [dict(row) for row in raw_rows]
            pair_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
            for row in rows:
                kypermno = str(row["KYPERMNO"])
                left_ric = str(row["left_candidate_ric"])
                right_ric = str(row["right_candidate_ric"])
                pair_meta[(kypermno, left_ric, right_ric)] = row
                pair_meta[(kypermno, right_ric, left_ric)] = row
            df = _build_explicit_schema_df(rows, _alias_diagnostic_schema()).select(ALIAS_DIAGNOSTIC_COLUMNS)
            _AUTHORITY_RUST_METRICS["pairwise_alias_diagnostics_fast_success"] += 1
            return df, pair_meta
        except Exception:
            _AUTHORITY_RUST_METRICS["pairwise_alias_diagnostics_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["pairwise_alias_diagnostics_fallbacks"] += 1
    return _build_pairwise_alias_diagnostics_py(permno_order=permno_order, candidate_meta=candidate_meta)


def _build_conventional_components_py(
    *,
    permno_order: list[str],
    candidate_meta: dict[tuple[str, str], dict[str, Any]],
    pair_meta: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, Any]]]]:
    component_map_by_permno: dict[str, dict[str, str]] = {}
    component_meta_by_permno: dict[str, dict[str, dict[str, Any]]] = {}
    for kypermno in permno_order:
        conventional_rics = sorted(
            candidate_ric
            for permno_value, candidate_ric in candidate_meta
            if permno_value == kypermno
            and bool(candidate_meta[(permno_value, candidate_ric)]["candidate_has_conventional_support"])
        )
        uf = _UnionFind(conventional_rics)
        for left_ric, right_ric in combinations(conventional_rics, 2):
            pair_row = pair_meta.get((kypermno, left_ric, right_ric))
            if pair_row is not None and bool(pair_row["pair_benign_alias_supported"]):
                uf.union(left_ric, right_ric)

        grouped_members: dict[str, list[str]] = defaultdict(list)
        for candidate_ric in conventional_rics:
            grouped_members[uf.find(candidate_ric)].append(candidate_ric)

        ordered_groups = sorted(
            grouped_members.values(),
            key=lambda members: (
                min(
                    (
                        candidate_meta[(kypermno, member)]["candidate_bridge_start_date"] or dt.date.max
                        for member in members
                    ),
                    default=dt.date.max,
                ),
                min(members) if members else "",
            ),
        )
        candidate_to_component: dict[str, str] = {}
        component_meta: dict[str, dict[str, Any]] = {}
        for component_index, members in enumerate(ordered_groups, start=1):
            sorted_members = sorted(members)
            component_id = _component_id(kypermno, component_index)
            for member in sorted_members:
                candidate_to_component[member] = component_id

            member_histories = [candidate_meta[(kypermno, member)] for member in sorted_members]
            union_value_sets: dict[tuple[dt.date, str], set[float | None]] = defaultdict(set)
            unique_row_set: set[tuple[dt.date, str, float | None]] = set()
            bridge_intervals: list[tuple[dt.date, dt.date]] = []
            for member_history in member_histories:
                for obs_key, values in member_history["observation_value_sets"].items():
                    union_value_sets[obs_key].update(values)
                unique_row_set.update(member_history["unique_row_set"])
                if (
                    isinstance(member_history["candidate_bridge_start_date"], dt.date)
                    and isinstance(member_history["candidate_bridge_end_date"], dt.date)
                ):
                    bridge_intervals.append(
                        (
                            member_history["candidate_bridge_start_date"],
                            member_history["candidate_bridge_end_date"],
                        )
                    )

            union_dates = {date_value for date_value, _ in union_value_sets}
            ranked_members = sorted(
                member_histories,
                key=lambda row: (
                    not bool(row["candidate_has_effective_support"]),
                    -int(row["candidate_ownership_date_count"]),
                    -int(row["candidate_bridge_span_day_count"]),
                    str(row["candidate_ric"]),
                ),
            )
            canonical_member = ranked_members[0] if ranked_members else None
            component_requires_review = any(
                len(_distinct_values(values)) > 1 for values in union_value_sets.values()
            ) or any(
                bool(pair_meta[(kypermno, left_member, right_member)]["pair_value_conflict_present"])
                or bool(pair_meta[(kypermno, left_member, right_member)]["pair_regime_split_supported"])
                or bool(pair_meta[(kypermno, left_member, right_member)]["pair_requires_review"])
                for left_member, right_member in combinations(sorted_members, 2)
                if (kypermno, left_member, right_member) in pair_meta
            )
            component_meta[component_id] = {
                "component_id": component_id,
                "member_rics": sorted_members,
                "canonical_ric": None if canonical_member is None else canonical_member["candidate_ric"],
                "canonical_best_candidate_date_count": 0
                if canonical_member is None
                else int(canonical_member["candidate_ownership_date_count"]),
                "ownership_date_count": len(union_dates),
                "ownership_row_count": len(unique_row_set),
                "bridge_start_date": min((interval[0] for interval in bridge_intervals), default=None),
                "bridge_end_date": max((interval[1] for interval in bridge_intervals), default=None),
                "first_ownership_date": min(union_dates, default=None),
                "last_ownership_date": max(union_dates, default=None),
                "merged_bridge_windows": _merge_intervals(
                    bridge_intervals,
                    max_gap_days=DATE_EXTENSION_MAX_GAP_DAYS,
                ),
                "union_value_sets": union_value_sets,
                "component_requires_review": component_requires_review,
                "coverage_share_of_best_candidate": (
                    float(canonical_member["candidate_ownership_date_count"]) / float(len(union_dates))
                    if canonical_member is not None and len(union_dates) > 0
                    else None
                ),
            }
        component_map_by_permno[kypermno] = candidate_to_component
        component_meta_by_permno[kypermno] = component_meta
    return component_map_by_permno, component_meta_by_permno


def _conventional_component_candidate_records(
    candidate_meta: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (kypermno, candidate_ric), meta in candidate_meta.items():
        records.append(
            {
                "KYPERMNO": kypermno,
                "candidate_ric": candidate_ric,
                "candidate_has_conventional_support": meta["candidate_has_conventional_support"],
                "candidate_has_effective_support": meta["candidate_has_effective_support"],
                "candidate_ownership_date_count": meta["candidate_ownership_date_count"],
                "candidate_bridge_span_day_count": meta["candidate_bridge_span_day_count"],
                "candidate_bridge_start_date": meta["candidate_bridge_start_date"],
                "candidate_bridge_end_date": meta["candidate_bridge_end_date"],
                "observation_value_sets": [
                    (date_value, category, list(values))
                    for (date_value, category), values in meta["observation_value_sets"].items()
                ],
                "unique_row_set": list(meta["unique_row_set"]),
            }
        )
    return records


def _conventional_component_pair_records(
    pair_meta: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (kypermno, left_ric, right_ric), row in pair_meta.items():
        records.append(
            {
                "KYPERMNO": kypermno,
                "left_candidate_ric": left_ric,
                "right_candidate_ric": right_ric,
                "pair_benign_alias_supported": row["pair_benign_alias_supported"],
                "pair_value_conflict_present": row["pair_value_conflict_present"],
                "pair_regime_split_supported": row["pair_regime_split_supported"],
                "pair_requires_review": row["pair_requires_review"],
            }
        )
    return records


def _conventional_components_from_rust_records(
    permno_order: list[str],
    map_records: list[dict[str, Any]],
    component_records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, Any]]]]:
    component_map_by_permno: dict[str, dict[str, str]] = defaultdict(dict)
    for row in map_records:
        component_map_by_permno[str(row["KYPERMNO"])][str(row["candidate_ric"])] = str(row["component_id"])

    component_meta_by_permno: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in component_records:
        kypermno = str(row["KYPERMNO"])
        component_id = str(row["component_id"])
        union_value_sets: dict[tuple[dt.date, str], set[float | None]] = defaultdict(set)
        for date_value, category, values in row.get("union_value_sets", []):
            union_value_sets[(date_value, str(category))] = set(values)
        component_meta_by_permno[kypermno][component_id] = {
            "component_id": component_id,
            "member_rics": list(row["member_rics"]),
            "canonical_ric": row["canonical_ric"],
            "canonical_best_candidate_date_count": int(row["canonical_best_candidate_date_count"]),
            "ownership_date_count": int(row["ownership_date_count"]),
            "ownership_row_count": int(row["ownership_row_count"]),
            "bridge_start_date": row["bridge_start_date"],
            "bridge_end_date": row["bridge_end_date"],
            "first_ownership_date": row["first_ownership_date"],
            "last_ownership_date": row["last_ownership_date"],
            "merged_bridge_windows": [tuple(window) for window in row["merged_bridge_windows"]],
            "union_value_sets": union_value_sets,
            "component_requires_review": bool(row["component_requires_review"]),
            "coverage_share_of_best_candidate": row["coverage_share_of_best_candidate"],
        }
    for kypermno in permno_order:
        component_map_by_permno.setdefault(str(kypermno), {})
        component_meta_by_permno.setdefault(str(kypermno), {})
    return dict(component_map_by_permno), {key: dict(value) for key, value in component_meta_by_permno.items()}


def _build_conventional_components(
    *,
    permno_order: list[str],
    candidate_meta: dict[tuple[str, str], dict[str, Any]],
    pair_meta: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, Any]]]]:
    if _lm2011_rust is not None:
        try:
            map_records, component_records = _lm2011_rust.refinitiv_authority_conventional_components_from_meta(
                list(permno_order),
                candidate_meta,
                pair_meta,
                DATE_EXTENSION_MAX_GAP_DAYS,
            )
            _AUTHORITY_RUST_METRICS["conventional_components_meta_fast_success"] += 1
            return _conventional_components_from_rust_records(
                list(permno_order),
                [dict(row) for row in map_records],
                [dict(row) for row in component_records],
            )
        except Exception:
            _AUTHORITY_RUST_METRICS["conventional_components_meta_fast_failures"] += 1
            _AUTHORITY_RUST_METRICS["conventional_components_meta_fallbacks"] += 1
        try:
            map_records, component_records = _lm2011_rust.refinitiv_authority_conventional_components(
                list(permno_order),
                _conventional_component_candidate_records(candidate_meta),
                _conventional_component_pair_records(pair_meta),
                DATE_EXTENSION_MAX_GAP_DAYS,
            )
            _AUTHORITY_RUST_METRICS["conventional_components_fast_success"] += 1
            return _conventional_components_from_rust_records(
                list(permno_order),
                [dict(row) for row in map_records],
                [dict(row) for row in component_records],
            )
        except Exception:
            _AUTHORITY_RUST_METRICS["conventional_components_fast_failures"] += 1
    _AUTHORITY_RUST_METRICS["conventional_components_fallbacks"] += 1
    return _build_conventional_components_py(
        permno_order=permno_order,
        candidate_meta=candidate_meta,
        pair_meta=pair_meta,
    )


def build_refinitiv_step1_ownership_authority_tables(
    resolution_df: pl.DataFrame,
    ownership_results_df: pl.DataFrame,
    ownership_row_summary_df: pl.DataFrame,
    *,
    reviewed_ticker_allowlist_df: pl.DataFrame | None = None,
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    normalized_resolution_df = _normalize_resolution_df(resolution_df)
    normalized_results_df = _normalize_results_df(ownership_results_df)
    normalized_row_summary_df = _normalize_row_summary_df(ownership_row_summary_df)
    normalized_allowlist_df = _normalize_allowlist_df(
        reviewed_ticker_allowlist_df
        if reviewed_ticker_allowlist_df is not None
        else pl.DataFrame(schema={"KYPERMNO": pl.Utf8, "candidate_ric": pl.Utf8})
    )

    permno_order = [
        kypermno
        for kypermno in normalized_resolution_df.get_column("KYPERMNO").drop_nulls().unique(maintain_order=True).to_list()
        if _normalize_lookup_text(kypermno) is not None
    ]
    resolution_conflict_by_permno: dict[str, bool] = {}
    for kypermno, has_conflict in (
        normalized_resolution_df.group_by("KYPERMNO", maintain_order=True)
        .agg(pl.col("conventional_identity_conflict").fill_null(False).any().alias("has_conflict"))
        .iter_rows()
    ):
        normalized_kypermno = _normalize_lookup_text(kypermno)
        if normalized_kypermno is not None:
            resolution_conflict_by_permno[normalized_kypermno] = bool(has_conflict)

    candidate_metrics_df, candidate_meta, permno_date_sets, permno_row_sets = _build_candidate_metrics_from_frames(
        normalized_row_summary_df,
        normalized_results_df,
    )
    alias_diagnostics_df, pair_meta = _build_pairwise_alias_diagnostics(
        permno_order=permno_order,
        candidate_meta=candidate_meta,
    )
    _component_map_by_permno, component_meta_by_permno = _build_conventional_components(
        permno_order=permno_order,
        candidate_meta=candidate_meta,
        pair_meta=pair_meta,
    )
    allowlist_keys = _allowlist_keys(normalized_allowlist_df)

    candidate_meta_by_permno: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_meta.values():
        candidate_meta_by_permno[str(candidate["KYPERMNO"])].append(candidate)

    pair_rows_by_permno: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_meta.values():
        pair_rows_by_permno[str(row["KYPERMNO"])].append(row)

    decision_rows: list[dict[str, Any]] = []
    exception_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    authority_assignment_rows: list[dict[str, Any]] = []

    for kypermno in permno_order:
        permno_candidates = sorted(
            candidate_meta_by_permno.get(kypermno, []),
            key=lambda row: str(row["candidate_ric"]),
        )
        conventional_candidates = [row for row in permno_candidates if bool(row["candidate_has_conventional_support"])]
        ticker_candidates = [row for row in permno_candidates if bool(row["candidate_has_ticker_support"])]
        effective_candidates = [row for row in permno_candidates if bool(row["candidate_has_effective_support"])]
        component_meta = component_meta_by_permno.get(kypermno, {})
        permno_pair_rows = pair_rows_by_permno.get(kypermno, [])

        permno_overlap_dates: set[dt.date] = set()
        permno_overlap_conflict_pairs: set[tuple[dt.date, str]] = set()
        for pair_row in permno_pair_rows:
            left_obs = candidate_meta[(kypermno, str(pair_row["left_candidate_ric"]))]["observation_value_sets"]
            right_obs = candidate_meta[(kypermno, str(pair_row["right_candidate_ric"]))]["observation_value_sets"]
            overlap_pairs = set(left_obs) & set(right_obs)
            permno_overlap_dates.update(date_value for date_value, _ in overlap_pairs)
            for overlap_key in overlap_pairs:
                left_values = _distinct_values(left_obs[overlap_key])
                right_values = _distinct_values(right_obs[overlap_key])
                if len(left_values) == 1 and len(right_values) == 1:
                    if not _values_match(left_values[0], right_values[0]):
                        permno_overlap_conflict_pairs.add(overlap_key)
                elif left_values != right_values:
                    permno_overlap_conflict_pairs.add(overlap_key)

        valid_conventional_components = [
            component for component in component_meta.values() if not bool(component["component_requires_review"])
        ]
        selected_component_ids: list[str] = []
        selected_allowlist_candidates: list[str] = []
        authoritative_ric: str | None = None
        authoritative_source_family: str | None = None
        authoritative_component_id: str | None = None
        authoritative_ownership_date_count = 0
        authoritative_ownership_row_count = 0
        authoritative_coverage_share_of_best_candidate: float | None = None
        authoritative_coverage_share_of_permno: float | None = None
        authority_decision_status = "NO_CONVENTIONAL_AUTHORITY"
        authority_decision_reason = "no_conventional_component"

        review_flag_conventional_identity_conflict = bool(resolution_conflict_by_permno.get(kypermno, False))
        review_flag_effective_overlap_conflict = any(
            bool(pair_row["pair_value_conflict_present"])
            and (
                bool(pair_row["left_candidate_has_effective_support"])
                or bool(pair_row["right_candidate_has_effective_support"])
            )
            for pair_row in permno_pair_rows
        )
        review_flag_multi_component_overlap = False
        review_flag_zero_return_effective_only = bool(effective_candidates) and all(
            int(candidate["candidate_request_with_data_count"]) == 0 for candidate in effective_candidates
        )
        review_flag_unresolved_multi_ric_structure = False
        reviewed_ticker_allowlist_applied = False

        if len(valid_conventional_components) == 1 and not review_flag_conventional_identity_conflict:
            selected_component = valid_conventional_components[0]
            selected_component_ids = [str(selected_component["component_id"])]
            authoritative_ric = _normalize_lookup_text(selected_component["canonical_ric"])
            authoritative_source_family = "CONVENTIONAL"
            authoritative_component_id = str(selected_component["component_id"])
            authoritative_ownership_date_count = int(selected_component["ownership_date_count"])
            authoritative_ownership_row_count = int(selected_component["ownership_row_count"])
            authoritative_coverage_share_of_best_candidate = selected_component["coverage_share_of_best_candidate"]
            total_conventional_dates = len(
                {
                    date_value
                    for candidate in conventional_candidates
                    for date_value, _ in candidate["observation_value_sets"]
                }
            )
            authoritative_coverage_share_of_permno = (
                float(authoritative_ownership_date_count) / float(total_conventional_dates)
                if total_conventional_dates > 0
                else None
            )
            authority_decision_status = "STATIC_CONVENTIONAL"
            authority_decision_reason = "single_conventional_component_after_alias_collapse"
        elif len(valid_conventional_components) > 1 and not review_flag_conventional_identity_conflict:
            cross_component_safe = True
            for left_component, right_component in combinations(valid_conventional_components, 2):
                component_pair_conflict = False
                for left_ric in left_component["member_rics"]:
                    for right_ric in right_component["member_rics"]:
                        pair_row = pair_meta.get((kypermno, left_ric, right_ric))
                        if pair_row is not None and bool(pair_row["pair_value_conflict_present"]):
                            component_pair_conflict = True
                            break
                    if component_pair_conflict:
                        break
                observed_disjoint = (
                    left_component["last_ownership_date"] is not None
                    and right_component["first_ownership_date"] is not None
                    and (
                        left_component["last_ownership_date"] < right_component["first_ownership_date"]
                        or right_component["last_ownership_date"] < left_component["first_ownership_date"]
                    )
                ) or not (set(left_component["union_value_sets"]) & set(right_component["union_value_sets"]))
                bridge_disjoint = (
                    left_component["bridge_end_date"] is not None
                    and right_component["bridge_start_date"] is not None
                    and (
                        left_component["bridge_end_date"] < right_component["bridge_start_date"]
                        or right_component["bridge_end_date"] < left_component["bridge_start_date"]
                    )
                )
                if component_pair_conflict or not (observed_disjoint or bridge_disjoint):
                    cross_component_safe = False
                    review_flag_multi_component_overlap = True
                    break
            if cross_component_safe:
                selected_component_ids = [str(component["component_id"]) for component in valid_conventional_components]
                authoritative_source_family = "CONVENTIONAL"
                authoritative_ownership_date_count = int(
                    sum(int(component["ownership_date_count"]) for component in valid_conventional_components)
                )
                authoritative_ownership_row_count = int(
                    sum(int(component["ownership_row_count"]) for component in valid_conventional_components)
                )
                total_conventional_dates = len(
                    {
                        date_value
                        for candidate in conventional_candidates
                        for date_value, _ in candidate["observation_value_sets"]
                    }
                )
                authoritative_coverage_share_of_permno = (
                    float(authoritative_ownership_date_count) / float(total_conventional_dates)
                    if total_conventional_dates > 0
                    else None
                )
                authority_decision_status = "DATE_VARYING_CONVENTIONAL_EXCEPTION"
                authority_decision_reason = "multiple_clean_conventional_regime_components"
            else:
                review_flag_unresolved_multi_ric_structure = True
                authority_decision_status = "REVIEW_REQUIRED"
                authority_decision_reason = "multi_component_overlap_not_resolved_as_regime_split"
        elif valid_conventional_components:
            review_flag_unresolved_multi_ric_structure = True
            authority_decision_status = "REVIEW_REQUIRED"
            authority_decision_reason = "conventional_component_review_required"

        if authority_decision_status == "NO_CONVENTIONAL_AUTHORITY" and ticker_candidates:
            allowlisted_candidates = [
                candidate
                for candidate in ticker_candidates
                if (kypermno, str(candidate["candidate_ric"])) in allowlist_keys
            ]
            if allowlisted_candidates:
                selected_allowlist_candidates = [str(candidate["candidate_ric"]) for candidate in allowlisted_candidates]
                reviewed_ticker_allowlist_applied = True
                authoritative_source_family = "TICKER_ALLOWLIST"
                authoritative_ownership_date_count = len(
                    {
                        date_value
                        for candidate in allowlisted_candidates
                        for date_value, _ in candidate["observation_value_sets"]
                    }
                )
                authoritative_ownership_row_count = int(
                    sum(len(candidate["unique_row_set"]) for candidate in allowlisted_candidates)
                )
                authority_decision_status = "REVIEWED_TICKER_ALLOWLIST_ONLY"
                authority_decision_reason = "explicit_reviewed_ticker_allowlist"

        review_flag_meaningful_coverage_loss_if_static = (
            authority_decision_status == "STATIC_CONVENTIONAL"
            and authoritative_coverage_share_of_best_candidate is not None
            and authoritative_coverage_share_of_best_candidate < MEANINGFUL_STATIC_COVERAGE_LOSS_THRESHOLD
        )
        review_flag_ticker_only_without_allowlist = (
            authority_decision_status == "NO_CONVENTIONAL_AUTHORITY"
            and len(ticker_candidates) > 0
            and not reviewed_ticker_allowlist_applied
        )
        requires_review = any(
            (
                review_flag_conventional_identity_conflict,
                review_flag_effective_overlap_conflict,
                review_flag_multi_component_overlap,
                review_flag_ticker_only_without_allowlist,
                review_flag_zero_return_effective_only,
                review_flag_meaningful_coverage_loss_if_static,
                review_flag_unresolved_multi_ric_structure,
                authority_decision_status == "REVIEW_REQUIRED",
            )
        )

        if authority_decision_status == "DATE_VARYING_CONVENTIONAL_EXCEPTION":
            for component_id in selected_component_ids:
                component = component_meta[component_id]
                for window_start, window_end in component["merged_bridge_windows"]:
                    exception_rows.append(
                        {
                            "KYPERMNO": kypermno,
                            "authoritative_component_id": component_id,
                            "authoritative_ric": component["canonical_ric"],
                            "authoritative_source_family": "CONVENTIONAL",
                            "component_member_rics": "|".join(component["member_rics"]),
                            "component_member_count": len(component["member_rics"]),
                            "authority_window_start_date": window_start,
                            "authority_window_end_date": window_end,
                            "authority_first_ownership_date": component["first_ownership_date"],
                            "authority_last_ownership_date": component["last_ownership_date"],
                            "authority_exception_status": "DATE_VARYING_CONVENTIONAL_EXCEPTION",
                            "authority_exception_reason": "clean_regime_split_component",
                        }
                    )

        selected_component_union_maps = [component_meta[component_id]["union_value_sets"] for component_id in selected_component_ids]
        ticker_support_strong_count = 0
        for ticker_candidate in ticker_candidates:
            overlap_date_count = 0
            overlap_category_count = 0
            overlap_date_category_count = 0
            differing_value_count = 0
            benign_alias_supported = False
            allowlist_recommendation = False
            allowlist_reason = "no_authoritative_conventional_component"
            if selected_component_union_maps:
                best_overlap = -1
                best_dates = 0
                best_categories = 0
                best_conflicts = 0
                for union_map in selected_component_union_maps:
                    overlap_pairs = set(ticker_candidate["observation_value_sets"]) & set(union_map)
                    overlap_dates = {date_value for date_value, _ in overlap_pairs}
                    overlap_categories = {category for _, category in overlap_pairs}
                    conflict_count = 0
                    for overlap_key in overlap_pairs:
                        left_values = _distinct_values(ticker_candidate["observation_value_sets"][overlap_key])
                        right_values = _distinct_values(union_map[overlap_key])
                        if len(left_values) == 1 and len(right_values) == 1:
                            if not _values_match(left_values[0], right_values[0]):
                                conflict_count += 1
                        elif left_values != right_values:
                            conflict_count += 1
                    if len(overlap_pairs) > best_overlap:
                        best_overlap = len(overlap_pairs)
                        best_dates = len(overlap_dates)
                        best_categories = len(overlap_categories)
                        best_conflicts = conflict_count
                overlap_date_count = best_dates
                overlap_category_count = best_categories
                overlap_date_category_count = max(best_overlap, 0)
                differing_value_count = best_conflicts
                benign_alias_supported = (
                    differing_value_count == 0
                    and overlap_date_category_count >= BENIGN_ALIAS_MIN_OVERLAP_DATE_CATEGORY_COUNT
                    and overlap_date_count >= BENIGN_ALIAS_MIN_OVERLAP_DATE_COUNT
                    and overlap_category_count >= BENIGN_ALIAS_MIN_OVERLAP_CATEGORY_COUNT
                )
                allowlist_recommendation = benign_alias_supported
                if allowlist_recommendation:
                    allowlist_reason = "strong_overlap_without_same_date_conflict"
                    ticker_support_strong_count += 1
                elif differing_value_count > 0:
                    allowlist_reason = "same_date_conflict_with_authoritative_conventional_component"
                elif overlap_date_category_count > 0:
                    allowlist_reason = "overlap_insufficient_for_allowlist"
            ticker_rows.append(
                {
                    "KYPERMNO": kypermno,
                    "candidate_ric": ticker_candidate["candidate_ric"],
                    "candidate_source_family": ticker_candidate["candidate_source_family"],
                    "candidate_has_ticker_support": ticker_candidate["candidate_has_ticker_support"],
                    "candidate_has_conventional_support": ticker_candidate["candidate_has_conventional_support"],
                    "candidate_request_count": ticker_candidate["candidate_request_count"],
                    "candidate_request_with_data_count": ticker_candidate["candidate_request_with_data_count"],
                    "candidate_ownership_date_count": ticker_candidate["candidate_ownership_date_count"],
                    "candidate_ownership_row_count": ticker_candidate["candidate_ownership_row_count"],
                    "ticker_overlap_date_count_vs_authoritative": overlap_date_count,
                    "ticker_overlap_category_count_vs_authoritative": overlap_category_count,
                    "ticker_overlap_date_category_count_vs_authoritative": overlap_date_category_count,
                    "ticker_same_date_same_category_differing_value_count_vs_authoritative": differing_value_count,
                    "ticker_benign_alias_supported_vs_authoritative": benign_alias_supported,
                    "ticker_allowlist_recommendation": allowlist_recommendation,
                    "ticker_allowlist_reason": allowlist_reason,
                }
            )

        decision_row = {
            "KYPERMNO": kypermno,
            "permno_conventional_candidate_count": len(conventional_candidates),
            "permno_ticker_candidate_count": len(ticker_candidates),
            "permno_effective_candidate_count": len(effective_candidates),
            "permno_conventional_ownership_date_count": len(permno_date_sets.get(kypermno, set()) & {
                date_value
                for candidate in conventional_candidates
                for date_value, _ in candidate["observation_value_sets"]
            }),
            "permno_ticker_ownership_date_count": len(
                {
                    date_value
                    for candidate in ticker_candidates
                    for date_value, _ in candidate["observation_value_sets"]
                }
            ),
            "permno_conventional_ownership_row_count": len(
                {
                    obs_row
                    for candidate in conventional_candidates
                    for obs_row in candidate["unique_row_set"]
                }
            ),
            "permno_ticker_ownership_row_count": len(
                {
                    obs_row
                    for candidate in ticker_candidates
                    for obs_row in candidate["unique_row_set"]
                }
            ),
            "permno_same_date_multi_ric_overlap_count": len(permno_overlap_dates),
            "permno_same_date_same_category_differing_value_count": len(permno_overlap_conflict_pairs),
            "permno_benign_alias_component_count": len(component_meta),
            "permno_regime_component_count": len(selected_component_ids)
            if authority_decision_status == "DATE_VARYING_CONVENTIONAL_EXCEPTION"
            else 0,
            "permno_regime_split_detected": authority_decision_status == "DATE_VARYING_CONVENTIONAL_EXCEPTION",
            "permno_ticker_support_candidate_count": len(ticker_candidates),
            "permno_ticker_support_strong_count": ticker_support_strong_count,
            "authoritative_ric": authoritative_ric,
            "authoritative_source_family": authoritative_source_family,
            "authoritative_component_id": authoritative_component_id,
            "authoritative_ownership_date_count": authoritative_ownership_date_count,
            "authoritative_ownership_row_count": authoritative_ownership_row_count,
            "authoritative_coverage_share_of_best_candidate": authoritative_coverage_share_of_best_candidate,
            "authoritative_coverage_share_of_permno": authoritative_coverage_share_of_permno,
            "authority_decision_status": authority_decision_status,
            "authority_decision_reason": authority_decision_reason,
            "requires_review": requires_review,
            "review_flag_conventional_identity_conflict": review_flag_conventional_identity_conflict,
            "review_flag_effective_overlap_conflict": review_flag_effective_overlap_conflict,
            "review_flag_multi_component_overlap": review_flag_multi_component_overlap,
            "review_flag_ticker_only_without_allowlist": review_flag_ticker_only_without_allowlist,
            "review_flag_zero_return_effective_only": review_flag_zero_return_effective_only,
            "review_flag_meaningful_coverage_loss_if_static": review_flag_meaningful_coverage_loss_if_static,
            "review_flag_unresolved_multi_ric_structure": review_flag_unresolved_multi_ric_structure,
            "reviewed_ticker_allowlist_applied": reviewed_ticker_allowlist_applied,
            "reviewed_ticker_allowlist_candidate_count": len(selected_allowlist_candidates),
        }
        decision_rows.append(decision_row)

        if requires_review:
            continue

        selected_meta: dict[str, tuple[str, str, str]] = {}
        for component_id in selected_component_ids:
            component = component_meta[component_id]
            for candidate_ric in component["member_rics"]:
                selected_meta[candidate_ric] = (
                    str(component["canonical_ric"]),
                    "CONVENTIONAL",
                    str(component_id),
                )
        if authority_decision_status == "REVIEWED_TICKER_ALLOWLIST_ONLY":
            for candidate_ric in selected_allowlist_candidates:
                selected_meta[candidate_ric] = (
                    candidate_ric,
                    "TICKER_ALLOWLIST",
                    f"TICKER_ALLOWLIST|{candidate_ric}",
                )

        for source_candidate_ric, (
            authoritative_assignment_ric,
            authoritative_family,
            authoritative_assignment_component,
        ) in selected_meta.items():
            authority_assignment_rows.append(
                {
                    "KYPERMNO": kypermno,
                    "source_candidate_ric": source_candidate_ric,
                    "authoritative_ric": authoritative_assignment_ric,
                    "authoritative_source_family": authoritative_family,
                    "authoritative_component_id": authoritative_assignment_component,
                    "authority_decision_status": authority_decision_status,
                }
            )

    final_panel_rows, panel_conflict_permnos = _authority_final_panel_rows_from_frame(
        normalized_results_df,
        authority_assignment_rows,
    )
    if panel_conflict_permnos:
        for decision_row in decision_rows:
            if (_normalize_lookup_text(decision_row.get("KYPERMNO")) or "") not in panel_conflict_permnos:
                continue
            decision_row["requires_review"] = True
            decision_row["authority_decision_status"] = "REVIEW_REQUIRED"
            decision_row["authority_decision_reason"] = "selected_authority_rows_conflict_on_same_date_category"
            decision_row["review_flag_unresolved_multi_ric_structure"] = True

    decision_df = _build_explicit_schema_df(decision_rows, _authority_decision_schema()).select(
        AUTHORITY_DECISION_COLUMNS
    )
    review_required_rows = _review_required_rows_from_frames(decision_df, candidate_metrics_df)

    tables = {
        "candidate_metrics": candidate_metrics_df,
        "alias_diagnostics": alias_diagnostics_df,
        "authority_decisions": decision_df,
        "authority_exceptions": _build_explicit_schema_df(
            exception_rows,
            _authority_exception_schema(),
        ).select(AUTHORITY_EXCEPTION_COLUMNS),
        "review_required": _build_explicit_schema_df(
            review_required_rows,
            _review_required_schema(),
        ).select(REVIEW_REQUIRED_COLUMNS),
        "ticker_candidates": _build_explicit_schema_df(
            ticker_rows,
            _ticker_candidate_schema(),
        ).select(TICKER_CANDIDATE_COLUMNS),
        "final_panel": (
            _build_explicit_schema_df(final_panel_rows, _final_panel_schema()).select(FINAL_PANEL_COLUMNS)
            if final_panel_rows
            else _empty_df(FINAL_PANEL_COLUMNS, _final_panel_schema())
        ),
    }
    summary = {
        "resolution_permno_count": len(permno_order),
        "candidate_metric_row_count": int(tables["candidate_metrics"].height),
        "alias_diagnostic_row_count": int(tables["alias_diagnostics"].height),
        "authority_decision_row_count": int(tables["authority_decisions"].height),
        "authority_exception_row_count": int(tables["authority_exceptions"].height),
        "review_required_row_count": int(tables["review_required"].height),
        "ticker_candidate_row_count": int(tables["ticker_candidates"].height),
        "final_panel_row_count": int(tables["final_panel"].height),
        "static_conventional_permno_count": int(
            tables["authority_decisions"].filter(pl.col("authority_decision_status") == "STATIC_CONVENTIONAL").height
        ),
        "date_varying_conventional_exception_permno_count": int(
            tables["authority_decisions"]
            .filter(pl.col("authority_decision_status") == "DATE_VARYING_CONVENTIONAL_EXCEPTION")
            .height
        ),
        "review_required_permno_count": int(tables["review_required"].height),
        "reviewed_ticker_allowlist_permno_count": int(
            tables["authority_decisions"].filter(pl.col("reviewed_ticker_allowlist_applied").fill_null(False)).height
        ),
    }
    return tables, summary


def run_refinitiv_step1_ownership_authority_pipeline(
    *,
    resolution_artifact_path: Path | str,
    ownership_results_artifact_path: Path | str,
    ownership_row_summary_artifact_path: Path | str,
    output_dir: Path | str,
    reviewed_ticker_allowlist_path: Path | str | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution_df = _read_resolution_artifact_parquet(resolution_artifact_path)
    ownership_results_df = _read_ownership_universe_results_artifact(ownership_results_artifact_path)
    ownership_row_summary_df = _read_ownership_universe_row_summary_artifact(
        ownership_row_summary_artifact_path
    )
    reviewed_ticker_allowlist_df = _read_reviewed_ticker_allowlist_artifact(reviewed_ticker_allowlist_path)

    tables, _ = build_refinitiv_step1_ownership_authority_tables(
        resolution_df,
        ownership_results_df,
        ownership_row_summary_df,
        reviewed_ticker_allowlist_df=reviewed_ticker_allowlist_df,
    )

    output_paths = {
        "refinitiv_permno_ownership_candidate_metrics_parquet": output_dir
        / "refinitiv_permno_ownership_candidate_metrics.parquet",
        "refinitiv_permno_ownership_alias_diagnostics_parquet": output_dir
        / "refinitiv_permno_ownership_alias_diagnostics.parquet",
        "refinitiv_permno_ownership_authority_decisions_parquet": output_dir
        / "refinitiv_permno_ownership_authority_decisions.parquet",
        "refinitiv_permno_ownership_authority_exceptions_parquet": output_dir
        / "refinitiv_permno_ownership_authority_exceptions.parquet",
        "refinitiv_permno_ownership_review_required_parquet": output_dir
        / "refinitiv_permno_ownership_review_required.parquet",
        "refinitiv_permno_ownership_ticker_candidates_parquet": output_dir
        / "refinitiv_permno_ownership_ticker_candidates.parquet",
        "refinitiv_permno_date_ownership_panel_parquet": output_dir
        / "refinitiv_permno_date_ownership_panel.parquet",
    }
    tables["candidate_metrics"].write_parquet(
        output_paths["refinitiv_permno_ownership_candidate_metrics_parquet"],
        compression="zstd",
    )
    tables["alias_diagnostics"].write_parquet(
        output_paths["refinitiv_permno_ownership_alias_diagnostics_parquet"],
        compression="zstd",
    )
    tables["authority_decisions"].write_parquet(
        output_paths["refinitiv_permno_ownership_authority_decisions_parquet"],
        compression="zstd",
    )
    tables["authority_exceptions"].write_parquet(
        output_paths["refinitiv_permno_ownership_authority_exceptions_parquet"],
        compression="zstd",
    )
    tables["review_required"].write_parquet(
        output_paths["refinitiv_permno_ownership_review_required_parquet"],
        compression="zstd",
    )
    tables["ticker_candidates"].write_parquet(
        output_paths["refinitiv_permno_ownership_ticker_candidates_parquet"],
        compression="zstd",
    )
    tables["final_panel"].write_parquet(
        output_paths["refinitiv_permno_date_ownership_panel_parquet"],
        compression="zstd",
    )
    return output_paths


__all__ = [
    "build_refinitiv_step1_ownership_authority_tables",
    "run_refinitiv_step1_ownership_authority_pipeline",
]
