from __future__ import annotations

from collections import Counter
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import polars as pl

from thesis_pkg.core.ccm.transforms import common_stock_pass_expr
from thesis_pkg.io.excel import (
    write_refinitiv_bridge_workbook,
    write_refinitiv_null_ric_diagnostics_workbook,
    write_refinitiv_ownership_smoke_testing_workbook,
    write_refinitiv_ric_lookup_extended_workbook,
    write_refinitiv_ric_lookup_workbook,
)


REQUIRED_DAILY_COLUMNS: tuple[str, ...] = (
    "KYPERMNO",
    "CALDT",
    "KYGVKEY_final",
    "LIID",
    "CIK_final",
    "CUSIP",
    "ISIN",
    "TICKER",
)

OPTIONAL_DAILY_COLUMNS: tuple[str, ...] = (
    "LINKTYPE",
    "LINKPRIM",
    "link_quality_flag",
    "HEXCNTRY",
    "n_filings",
)

BRIDGE_SOURCE_COLUMNS: tuple[str, ...] = (
    "bridge_row_id",
    "KYPERMNO",
    "KYGVKEY_final",
    "LIID",
    "CIK_final",
    "CUSIP",
    "ISIN",
    "TICKER",
    "LINKTYPE",
    "LINKPRIM",
    "link_quality_flag",
    "HEXCNTRY",
    "company_name",
    "first_seen_caldt",
    "last_seen_caldt",
    "n_daily_rows",
    "n_filing_days",
    "n_filings_total",
)

BRIDGE_VENDOR_COLUMNS: tuple[str, ...] = (
    "vendor_match_status",
    "vendor_primary_id_type",
    "vendor_primary_id",
    "vendor_primary_ric",
    "vendor_ds_mnemonic",
    "vendor_permid",
    "vendor_returned_name",
    "vendor_returned_cusip",
    "vendor_returned_isin",
    "vendor_candidate_ids_raw",
    "vendor_notes",
)

BRIDGE_OUTPUT_COLUMNS: tuple[str, ...] = BRIDGE_SOURCE_COLUMNS + BRIDGE_VENDOR_COLUMNS

WORKBOOK_TEXT_COLUMNS: tuple[str, ...] = (
    "bridge_row_id",
    "KYPERMNO",
    "KYGVKEY_final",
    "LIID",
    "CIK_final",
    "CUSIP",
    "ISIN",
    "TICKER",
    "LINKTYPE",
    "LINKPRIM",
    "link_quality_flag",
    "HEXCNTRY",
    "company_name",
    "first_seen_caldt",
    "last_seen_caldt",
) + BRIDGE_VENDOR_COLUMNS

RIC_LOOKUP_COLUMNS: tuple[str, ...] = (
    "bridge_row_id",
    "KYPERMNO",
    "CUSIP",
    "ISIN",
    "TICKER",
    "first_seen_caldt",
    "last_seen_caldt",
    "preferred_lookup_id",
    "preferred_lookup_type",
    "vendor_primary_ric",
    "vendor_returned_name",
    "vendor_returned_cusip",
    "vendor_returned_isin",
    "vendor_match_status",
    "vendor_notes",
)

RIC_LOOKUP_TEXT_COLUMNS: tuple[str, ...] = RIC_LOOKUP_COLUMNS


@dataclass(frozen=True)
class RicLookupFilterProfile:
    name: str
    predicates: tuple[str, ...]
    require_common_stock: bool = False
    require_gvkey: bool = False


RIC_LOOKUP_FILTER_PROFILES: tuple[RicLookupFilterProfile, ...] = (
    RicLookupFilterProfile(
        name="common_stock",
        predicates=(
            "KYPERMNO is not null",
            "common stock (SHRCD in {10, 11})",
        ),
        require_common_stock=True,
    ),
    RicLookupFilterProfile(
        name="common_stock_with_gvkey",
        predicates=(
            "KYPERMNO is not null",
            "common stock (SHRCD in {10, 11})",
            "KYGVKEY_final is not null",
        ),
        require_common_stock=True,
        require_gvkey=True,
    ),
)

LOOKUP_IDENTIFIER_TYPES: tuple[str, ...] = ("ISIN", "CUSIP", "TICKER")
LOOKUP_IDENTIFIER_PAIRS: tuple[tuple[str, str], ...] = (
    ("ISIN", "CUSIP"),
    ("ISIN", "TICKER"),
    ("CUSIP", "TICKER"),
)

RIC_LOOKUP_EXTENDED_BASE_COLUMNS: tuple[str, ...] = (
    "bridge_row_id",
    "KYPERMNO",
    "CUSIP",
    "ISIN",
    "TICKER",
    "first_seen_caldt",
    "last_seen_caldt",
    "vendor_primary_ric",
    "vendor_returned_name",
    "vendor_returned_cusip",
    "vendor_returned_isin",
    "vendor_match_status",
    "vendor_notes",
)


def _extended_lookup_columns() -> tuple[str, ...]:
    columns = list(RIC_LOOKUP_EXTENDED_BASE_COLUMNS)
    for identifier_type in LOOKUP_IDENTIFIER_TYPES:
        columns.extend(
            (
                f"{identifier_type}_lookup_input",
                f"{identifier_type}_attempted",
                f"{identifier_type}_returned_ric",
                f"{identifier_type}_returned_name",
                f"{identifier_type}_returned_isin",
                f"{identifier_type}_returned_cusip",
                f"{identifier_type}_success",
            )
        )
    for left_type, right_type in LOOKUP_IDENTIFIER_PAIRS:
        columns.extend(
            (
                f"{left_type}_vs_{right_type}_same_ric",
                f"{left_type}_vs_{right_type}_same_isin",
                f"{left_type}_vs_{right_type}_same_cusip",
            )
        )
    columns.append("all_successful_attempts_consistent")
    return tuple(columns)


RIC_LOOKUP_EXTENDED_COLUMNS: tuple[str, ...] = _extended_lookup_columns()
RIC_LOOKUP_EXTENDED_TEXT_COLUMNS: tuple[str, ...] = tuple(
    name
    for name in RIC_LOOKUP_EXTENDED_COLUMNS
    if name not in {"all_successful_attempts_consistent"}
    and not name.endswith("_attempted")
    and not name.endswith("_success")
    and "_same_" not in name
)

RIC_LOOKUP_EXTENDED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "summary_category",
    "summary_key",
    "value",
)
RIC_LOOKUP_EXTENDED_SUMMARY_TEXT_COLUMNS: tuple[str, ...] = (
    "summary_category",
    "summary_key",
)

NULL_RIC_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    *RIC_LOOKUP_COLUMNS,
    "failed_lookup_flag",
    "failed_lookup_reason",
    "invalid_identifier_signal",
    "successful_row_exists_for_kypermno",
    "successful_row_exists_before_span",
    "successful_row_exists_after_span",
    "successful_row_overlap_exists",
    "nearest_successful_gap_days_before",
    "nearest_successful_gap_days_after",
    "unique_successful_identifier_pair_count",
    "unique_successful_ric_count",
    "failed_identifier_pair_matches_success",
    "alternative_identifier_available",
    "alternative_identifier",
    "alternative_identifier_type",
    "candidate_successful_ric_available",
    "candidate_successful_ric",
    "candidate_successful_cusip",
    "candidate_successful_isin",
)

NULL_RIC_DIAGNOSTIC_TEXT_COLUMNS: tuple[str, ...] = (
    "bridge_row_id",
    "KYPERMNO",
    "CUSIP",
    "ISIN",
    "TICKER",
    "first_seen_caldt",
    "last_seen_caldt",
    "preferred_lookup_id",
    "preferred_lookup_type",
    "vendor_primary_ric",
    "vendor_returned_name",
    "vendor_returned_cusip",
    "vendor_returned_isin",
    "vendor_match_status",
    "vendor_notes",
    "failed_lookup_reason",
    "alternative_identifier_available",
    "candidate_successful_ric",
    "candidate_successful_cusip",
    "candidate_successful_isin",
    "alternative_identifier",
    "alternative_identifier_type",
    "candidate_successful_ric_available",
)

NULL_RIC_REVIEW_COLUMNS: tuple[str, ...] = (
    "bridge_row_id",
    "KYPERMNO",
    "TICKER",
    "first_seen_caldt",
    "last_seen_caldt",
    "preferred_lookup_id",
    "preferred_lookup_type",
    "vendor_primary_ric",
    "alternative_identifier",
    "alternative_identifier_type",
    "candidate_successful_ric",
    "successful_row_exists_for_kypermno",
    "successful_row_exists_before_span",
    "successful_row_exists_after_span",
    "successful_row_overlap_exists",
    "nearest_successful_gap_days_before",
    "nearest_successful_gap_days_after",
    "unique_successful_identifier_pair_count",
    "unique_successful_ric_count",
    "alternative_identifier_available",
    "candidate_successful_ric_available",
    "test_category",
    "test_method",
    "test_result",
    "test_notes",
)

NULL_RIC_REVIEW_TEXT_COLUMNS: tuple[str, ...] = (
    "bridge_row_id",
    "KYPERMNO",
    "TICKER",
    "first_seen_caldt",
    "last_seen_caldt",
    "preferred_lookup_id",
    "preferred_lookup_type",
    "vendor_primary_ric",
    "alternative_identifier",
    "alternative_identifier_type",
    "candidate_successful_ric",
    "test_category",
    "test_method",
    "test_result",
    "test_notes",
)

OWNERSHIP_SMOKE_SAMPLE_COLUMNS: tuple[str, ...] = (
    "sample_category",
    "bridge_row_id",
    "KYPERMNO",
    "TICKER",
    "lookup_input",
    "lookup_input_source",
    "request_start_date",
    "request_end_date",
    "preferred_lookup_id",
    "preferred_lookup_type",
    "alternative_identifier",
    "alternative_identifier_type",
    "candidate_successful_ric",
    "successful_row_exists_before_span",
    "successful_row_exists_after_span",
    "successful_row_overlap_exists",
    "alternative_identifier_available",
    "candidate_successful_ric_available",
    "unique_successful_identifier_pair_count",
    "unique_successful_ric_count",
)

OWNERSHIP_SMOKE_SAMPLE_SCHEMA: dict[str, pl.DataType] = {
    "sample_category": pl.Utf8,
    "bridge_row_id": pl.Utf8,
    "KYPERMNO": pl.Utf8,
    "TICKER": pl.Utf8,
    "lookup_input": pl.Utf8,
    "lookup_input_source": pl.Utf8,
    "request_start_date": pl.Utf8,
    "request_end_date": pl.Utf8,
    "preferred_lookup_id": pl.Utf8,
    "preferred_lookup_type": pl.Utf8,
    "alternative_identifier": pl.Utf8,
    "alternative_identifier_type": pl.Utf8,
    "candidate_successful_ric": pl.Utf8,
    "successful_row_exists_before_span": pl.Boolean,
    "successful_row_exists_after_span": pl.Boolean,
    "successful_row_overlap_exists": pl.Boolean,
    "alternative_identifier_available": pl.Boolean,
    "candidate_successful_ric_available": pl.Boolean,
    "unique_successful_identifier_pair_count": pl.Int64,
    "unique_successful_ric_count": pl.Int64,
}

OWNERSHIP_SMOKE_CATEGORY_SPECS: tuple[tuple[str, Callable[[dict[str, Any]], bool]], ...] = (
    ("successful_row_exists_after_span", lambda row: bool(row.get("successful_row_exists_after_span"))),
    ("successful_row_overlap_exists", lambda row: bool(row.get("successful_row_overlap_exists"))),
    ("successful_row_exists_before_span", lambda row: bool(row.get("successful_row_exists_before_span"))),
    ("alternative_identifier_available", lambda row: bool(row.get("alternative_identifier_available"))),
    (
        "no_successful_row_for_kypermno",
        lambda row: not bool(row.get("successful_row_exists_for_kypermno")),
    ),
    (
        "multiple_successful_identifier_pairs_or_rics",
        lambda row: int(row.get("unique_successful_identifier_pair_count") or 0) > 1
        or int(row.get("unique_successful_ric_count") or 0) > 1,
    ),
)


def _require_columns(lf: pl.LazyFrame, required: tuple[str, ...], label: str) -> None:
    schema = lf.collect_schema()
    missing = [name for name in required if name not in schema]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _normalize_company_description_lf(
    company_description_lf: pl.LazyFrame | None,
    *,
    target_gvkey_dtype: pl.DataType,
) -> pl.LazyFrame | None:
    if company_description_lf is None:
        return None

    schema = company_description_lf.collect_schema()
    if "KYGVKEY" not in schema:
        raise ValueError("company_description missing required column: KYGVKEY")

    name_candidates = [name for name in ("CONM", "CONML") if name in schema]
    if not name_candidates:
        return None

    return (
        company_description_lf.select(
            pl.col("KYGVKEY").cast(target_gvkey_dtype, strict=False).alias("KYGVKEY_final"),
            pl.coalesce([pl.col(name).cast(pl.Utf8, strict=False) for name in name_candidates]).alias("company_name"),
        )
        .drop_nulls(subset=["KYGVKEY_final"])
        .filter(pl.col("company_name").is_not_null())
        .unique(subset=["KYGVKEY_final"], keep="first")
    )


def _bridge_row_id_expr() -> pl.Expr:
    return pl.concat_str(
        [
            pl.col("KYPERMNO").cast(pl.Utf8, strict=False),
            pl.coalesce([pl.col("LIID"), pl.lit("-")]),
            pl.coalesce([pl.col("CUSIP"), pl.lit("-")]),
            pl.coalesce([pl.col("ISIN"), pl.lit("-")]),
            pl.coalesce([pl.col("TICKER"), pl.lit("-")]),
        ],
        separator=":",
    )


def build_refinitiv_step1_bridge_universe(
    daily_lf: pl.LazyFrame,
    *,
    company_description_lf: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Build the compact security-identifier bridge universe for Refinitiv step 1."""
    _require_columns(daily_lf, REQUIRED_DAILY_COLUMNS, "daily panel")
    schema = daily_lf.collect_schema()

    projected_cols = list(REQUIRED_DAILY_COLUMNS)
    projected_cols.extend(name for name in OPTIONAL_DAILY_COLUMNS if name in schema)

    n_filings_expr: pl.Expr
    if "n_filings" in schema:
        n_filings_expr = pl.col("n_filings").cast(pl.Int64, strict=False)
    else:
        n_filings_expr = pl.lit(0, dtype=pl.Int64)

    base = (
        daily_lf.select(projected_cols)
        .drop_nulls(subset=["KYPERMNO", "CALDT"])
        .with_columns(
            pl.col("KYPERMNO").cast(pl.Int32, strict=False),
            pl.col("CALDT").cast(pl.Date, strict=False),
            pl.col("KYGVKEY_final").cast(pl.Utf8, strict=False),
            pl.col("LIID").cast(pl.Utf8, strict=False),
            pl.col("CIK_final").cast(pl.Utf8, strict=False),
            _clean_identifier_expr("CUSIP"),
            _clean_identifier_expr("ISIN"),
            _clean_identifier_expr("TICKER"),
            (
                pl.col("LINKTYPE").cast(pl.Utf8, strict=False)
                if "LINKTYPE" in schema
                else pl.lit(None, dtype=pl.Utf8)
            ).alias("LINKTYPE"),
            (
                pl.col("LINKPRIM").cast(pl.Utf8, strict=False)
                if "LINKPRIM" in schema
                else pl.lit(None, dtype=pl.Utf8)
            ).alias("LINKPRIM"),
            (
                pl.col("link_quality_flag").cast(pl.Utf8, strict=False)
                if "link_quality_flag" in schema
                else pl.lit(None, dtype=pl.Utf8)
            ).alias("link_quality_flag"),
            (
                pl.col("HEXCNTRY").cast(pl.Utf8, strict=False)
                if "HEXCNTRY" in schema
                else pl.lit(None, dtype=pl.Utf8)
            ).alias("HEXCNTRY"),
            n_filings_expr.alias("n_filings"),
        )
        .sort("KYPERMNO", "LIID", "CUSIP", "ISIN", "TICKER", "CALDT")
    )

    grouped = base.group_by(["KYPERMNO", "LIID", "CUSIP", "ISIN", "TICKER"], maintain_order=True).agg(
        pl.col("KYGVKEY_final").drop_nulls().last().alias("KYGVKEY_final"),
        pl.col("CIK_final").drop_nulls().last().alias("CIK_final"),
        pl.col("LINKTYPE").drop_nulls().last().alias("LINKTYPE"),
        pl.col("LINKPRIM").drop_nulls().last().alias("LINKPRIM"),
        pl.col("link_quality_flag").drop_nulls().last().alias("link_quality_flag"),
        pl.col("HEXCNTRY").drop_nulls().last().alias("HEXCNTRY"),
        pl.col("CALDT").min().alias("first_seen_caldt"),
        pl.col("CALDT").max().alias("last_seen_caldt"),
        pl.len().cast(pl.Int64).alias("n_daily_rows"),
        (pl.col("n_filings") > pl.lit(0)).cast(pl.Int64).sum().alias("n_filing_days"),
        pl.col("n_filings").sum().cast(pl.Int64).alias("n_filings_total"),
    )

    company_names = _normalize_company_description_lf(
        company_description_lf,
        target_gvkey_dtype=pl.Utf8,
    )
    if company_names is not None:
        grouped = grouped.join(company_names, on="KYGVKEY_final", how="left")
    else:
        grouped = grouped.with_columns(pl.lit(None, dtype=pl.Utf8).alias("company_name"))

    bridge = grouped.with_columns(_bridge_row_id_expr().alias("bridge_row_id"))

    for name in BRIDGE_VENDOR_COLUMNS:
        bridge = bridge.with_columns(pl.lit(None, dtype=pl.Utf8).alias(name))

    return bridge.select(BRIDGE_OUTPUT_COLUMNS).sort(
        "KYPERMNO",
        "LIID",
        "CUSIP",
        "ISIN",
        "TICKER",
        "first_seen_caldt",
    )


def _build_handoff_df(df: pl.DataFrame) -> pl.DataFrame:
    return _coerce_text_columns(df.select(BRIDGE_OUTPUT_COLUMNS), WORKBOOK_TEXT_COLUMNS)


def _coerce_text_columns(df: pl.DataFrame, text_columns: tuple[str, ...]) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    text_col_set = set(text_columns)
    for name in df.columns:
        if name in text_col_set:
            if name in ("first_seen_caldt", "last_seen_caldt"):
                exprs.append(
                    pl.when(pl.col(name).is_null())
                    .then(pl.lit(None, dtype=pl.Utf8))
                    .otherwise(pl.col(name).cast(pl.Date, strict=False).dt.strftime("%Y-%m-%d"))
                    .alias(name)
                )
            else:
                exprs.append(
                    pl.when(pl.col(name).is_null())
                    .then(pl.lit(None, dtype=pl.Utf8))
                    .otherwise(pl.col(name).cast(pl.Utf8, strict=False))
                    .alias(name)
                )
        else:
            exprs.append(pl.col(name))
    return df.select(exprs)


def _clean_text_value_expr(name: str) -> pl.Expr:
    cleaned = (
        pl.when(pl.col(name).is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(pl.col(name).cast(pl.Utf8, strict=False).str.strip_chars())
    )
    return pl.when(cleaned.is_null() | cleaned.eq("")).then(pl.lit(None, dtype=pl.Utf8)).otherwise(cleaned)


def _clean_text_expr(name: str) -> pl.Expr:
    return _clean_text_value_expr(name).alias(name)


def _clean_identifier_value_expr(name: str) -> pl.Expr:
    return _clean_text_value_expr(name)


def _clean_identifier_expr(name: str) -> pl.Expr:
    return (
        _clean_identifier_value_expr(name).alias(name)
    )


def _build_ric_lookup_base_frame(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        pl.col("bridge_row_id"),
        pl.col("KYPERMNO"),
        _clean_identifier_expr("CUSIP"),
        _clean_identifier_expr("ISIN"),
        _clean_identifier_expr("TICKER"),
        pl.col("first_seen_caldt"),
        pl.col("last_seen_caldt"),
        _clean_text_expr("vendor_primary_ric"),
        _clean_text_expr("vendor_returned_name"),
        _clean_identifier_expr("vendor_returned_cusip"),
        _clean_identifier_expr("vendor_returned_isin"),
        _clean_text_expr("vendor_match_status"),
        _clean_text_expr("vendor_notes"),
    ).with_columns(
        pl.when(pl.col("ISIN").is_not_null() & pl.col("ISIN").ne(""))
        .then(pl.col("ISIN"))
        .when(pl.col("CUSIP").is_not_null() & pl.col("CUSIP").ne(""))
        .then(pl.col("CUSIP"))
        .when(pl.col("TICKER").is_not_null() & pl.col("TICKER").ne(""))
        .then(pl.col("TICKER"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("preferred_lookup_id"),
        pl.when(pl.col("ISIN").is_not_null() & pl.col("ISIN").ne(""))
        .then(pl.lit("ISIN"))
        .when(pl.col("CUSIP").is_not_null() & pl.col("CUSIP").ne(""))
        .then(pl.lit("CUSIP"))
        .when(pl.col("TICKER").is_not_null() & pl.col("TICKER").ne(""))
        .then(pl.lit("TICKER"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("preferred_lookup_type"),
    )


def _build_ric_lookup_handoff_frames(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    lookup_base = _build_ric_lookup_base_frame(df)

    ordered_lookup = lookup_base.select(RIC_LOOKUP_COLUMNS)
    lookup_df = ordered_lookup.filter(pl.col("preferred_lookup_id").is_not_null())
    manual_review_df = ordered_lookup.filter(pl.col("preferred_lookup_id").is_null())
    return (
        _coerce_text_columns(lookup_df, RIC_LOOKUP_TEXT_COLUMNS),
        _coerce_text_columns(manual_review_df, RIC_LOOKUP_TEXT_COLUMNS),
    )


def _combine_lookup_frames(lookup_df: pl.DataFrame, manual_review_df: pl.DataFrame) -> pl.DataFrame:
    frames = [lookup_df]
    if manual_review_df.height:
        frames.append(manual_review_df)
    return pl.concat(frames, how="vertical_relaxed").sort("bridge_row_id") if frames else pl.DataFrame()


def _profile_output_key(profile_name: str) -> str:
    return f"refinitiv_ric_lookup_handoff_{profile_name}_xlsx"


def _profile_output_path(output_dir: Path, profile_name: str) -> Path:
    return output_dir / f"refinitiv_ric_lookup_handoff_{profile_name}.xlsx"


def _extended_profile_output_key(profile_name: str) -> str:
    return f"refinitiv_ric_lookup_handoff_{profile_name}_extended_xlsx"


def _extended_profile_output_path(output_dir: Path, profile_name: str) -> Path:
    return output_dir / f"refinitiv_ric_lookup_handoff_{profile_name}_extended.xlsx"


def _build_lookup_profile_bridge_ids(
    daily_lf: pl.LazyFrame,
    profile: RicLookupFilterProfile,
) -> pl.DataFrame:
    _require_columns(
        daily_lf,
        REQUIRED_DAILY_COLUMNS + ("SHRCD",),
        f"daily panel for lookup profile {profile.name}",
    )

    eligible = (
        daily_lf.select("KYPERMNO", "CALDT", "LIID", "CUSIP", "ISIN", "TICKER", "SHRCD")
        .drop_nulls(subset=["KYPERMNO", "CALDT"])
        .with_columns(
            pl.col("KYPERMNO").cast(pl.Int32, strict=False),
            pl.col("CALDT").cast(pl.Date, strict=False),
            pl.col("LIID").cast(pl.Utf8, strict=False),
            _clean_identifier_expr("CUSIP"),
            _clean_identifier_expr("ISIN"),
            _clean_identifier_expr("TICKER"),
        )
        .with_columns(_bridge_row_id_expr().alias("bridge_row_id"))
    )
    if profile.require_common_stock:
        eligible = eligible.filter(common_stock_pass_expr("SHRCD"))
    return eligible.select("bridge_row_id").unique(subset=["bridge_row_id"]).collect()


def _build_filtered_ric_lookup_profile_artifact(
    bridge_df: pl.DataFrame,
    qualifying_bridge_ids: pl.DataFrame,
    profile: RicLookupFilterProfile,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    filtered_bridge = bridge_df.join(qualifying_bridge_ids, on="bridge_row_id", how="semi")
    if profile.require_gvkey:
        filtered_bridge = filtered_bridge.filter(pl.col("KYGVKEY_final").is_not_null())

    lookup_df, manual_review_df = _build_ric_lookup_handoff_frames(filtered_bridge)
    summary: dict[str, Any] = {
        "profile_name": profile.name,
        "predicates_applied": list(profile.predicates),
        "rows_before_filter": int(bridge_df.height),
        "rows_after_filter": int(filtered_bridge.height),
        "ric_lookup_rows": int(lookup_df.height),
        "ric_manual_review_rows": int(manual_review_df.height),
    }
    return lookup_df, manual_review_df, summary


def _lookup_input_expr(identifier_type: str) -> pl.Expr:
    return _clean_identifier_value_expr(identifier_type).alias(f"{identifier_type}_lookup_input")


def build_refinitiv_lookup_extended_diagnostic_artifact(
    lookup_df: pl.DataFrame,
    manual_review_df: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    combined = _combine_lookup_frames(
        lookup_df,
        manual_review_df if manual_review_df is not None else pl.DataFrame(schema={name: pl.Utf8 for name in RIC_LOOKUP_COLUMNS}),
    )
    if combined.height == 0:
        empty_extended = pl.DataFrame(schema={name: pl.Utf8 for name in RIC_LOOKUP_EXTENDED_COLUMNS}).with_columns(
            *[
                pl.col(name).cast(pl.Boolean, strict=False)
                for name in RIC_LOOKUP_EXTENDED_COLUMNS
                if name.endswith("_attempted")
                or name.endswith("_success")
                or "_same_" in name
                or name == "all_successful_attempts_consistent"
            ]
        )
        empty_summary = pl.DataFrame(schema={
            "summary_category": pl.Utf8,
            "summary_key": pl.Utf8,
            "value": pl.Int64,
        })
        summary_payload = {
            "attempt_counts_by_identifier_type": {identifier_type: 0 for identifier_type in LOOKUP_IDENTIFIER_TYPES},
            "success_counts_by_identifier_type": {identifier_type: 0 for identifier_type in LOOKUP_IDENTIFIER_TYPES},
            "agreement_counts": {},
            "conflict_counts": {},
            "rows_where_only_one_identifier_type_succeeds": 0,
        }
        return empty_extended, empty_summary, summary_payload

    base = combined.select(
        _clean_text_expr("bridge_row_id"),
        _clean_text_expr("KYPERMNO"),
        _clean_identifier_expr("CUSIP"),
        _clean_identifier_expr("ISIN"),
        _clean_identifier_expr("TICKER"),
        pl.when(pl.col("first_seen_caldt").is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(pl.col("first_seen_caldt").cast(pl.Utf8, strict=False))
        .alias("first_seen_caldt"),
        pl.when(pl.col("last_seen_caldt").is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(pl.col("last_seen_caldt").cast(pl.Utf8, strict=False))
        .alias("last_seen_caldt"),
        _clean_text_expr("vendor_primary_ric"),
        _clean_text_expr("vendor_returned_name"),
        _clean_identifier_expr("vendor_returned_cusip"),
        _clean_identifier_expr("vendor_returned_isin"),
        _clean_text_expr("vendor_match_status"),
        _clean_text_expr("vendor_notes"),
    )

    per_identifier_exprs: list[pl.Expr] = []
    for identifier_type in LOOKUP_IDENTIFIER_TYPES:
        per_identifier_exprs.extend(
            (
                _lookup_input_expr(identifier_type),
                pl.lit(None, dtype=pl.Utf8).alias(f"{identifier_type}_returned_ric"),
                pl.lit(None, dtype=pl.Utf8).alias(f"{identifier_type}_returned_name"),
                pl.lit(None, dtype=pl.Utf8).alias(f"{identifier_type}_returned_isin"),
                pl.lit(None, dtype=pl.Utf8).alias(f"{identifier_type}_returned_cusip"),
            )
        )

    extended = base.with_columns(per_identifier_exprs)
    extended = extended.with_columns(
        *[
            pl.col(f"{identifier_type}_lookup_input").is_not_null().alias(f"{identifier_type}_attempted")
            for identifier_type in LOOKUP_IDENTIFIER_TYPES
        ],
        *[
            pl.col(f"{identifier_type}_returned_ric").is_not_null().alias(f"{identifier_type}_success")
            for identifier_type in LOOKUP_IDENTIFIER_TYPES
        ],
    )
    extended = extended.with_columns(
        *[
            pl.lit(None, dtype=pl.Boolean).alias(f"{left_type}_vs_{right_type}_same_ric")
            for left_type, right_type in LOOKUP_IDENTIFIER_PAIRS
        ],
        *[
            pl.lit(None, dtype=pl.Boolean).alias(f"{left_type}_vs_{right_type}_same_isin")
            for left_type, right_type in LOOKUP_IDENTIFIER_PAIRS
        ],
        *[
            pl.lit(None, dtype=pl.Boolean).alias(f"{left_type}_vs_{right_type}_same_cusip")
            for left_type, right_type in LOOKUP_IDENTIFIER_PAIRS
        ],
    )
    extended = extended.with_columns(
        pl.lit(None, dtype=pl.Boolean).alias("all_successful_attempts_consistent")
    ).select(RIC_LOOKUP_EXTENDED_COLUMNS)

    summary_rows: list[dict[str, Any]] = []
    attempt_counts_by_identifier_type: dict[str, int] = {}
    success_counts_by_identifier_type: dict[str, int] = {}
    agreement_counts: dict[str, int] = {}
    conflict_counts: dict[str, int] = {}

    for identifier_type in LOOKUP_IDENTIFIER_TYPES:
        attempted_count = int(
            extended.select(pl.col(f"{identifier_type}_attempted").cast(pl.Int64).sum()).item()
        )
        success_count = int(
            extended.select(pl.col(f"{identifier_type}_success").cast(pl.Int64).sum()).item()
        )
        attempt_counts_by_identifier_type[identifier_type] = attempted_count
        success_counts_by_identifier_type[identifier_type] = success_count
        summary_rows.extend(
            (
                {
                    "summary_category": "attempt_count_by_identifier_type",
                    "summary_key": identifier_type,
                    "value": attempted_count,
                },
                {
                    "summary_category": "success_count_by_identifier_type",
                    "summary_key": identifier_type,
                    "value": success_count,
                },
            )
        )

    for left_type, right_type in LOOKUP_IDENTIFIER_PAIRS:
        pair_name = f"{left_type}_vs_{right_type}"
        for field_name in ("ric", "isin", "cusip"):
            column_name = f"{pair_name}_same_{field_name}"
            agreement_count = int(
                extended.select(pl.col(column_name).fill_null(False).cast(pl.Int64).sum()).item()
            )
            conflict_count = int(
                extended.select(
                    pl.when(pl.col(column_name).is_null())
                    .then(pl.lit(0, dtype=pl.Int64))
                    .otherwise((~pl.col(column_name)).cast(pl.Int64))
                    .sum()
                ).item()
            )
            agreement_key = f"{pair_name}_same_{field_name}"
            conflict_key = f"{pair_name}_same_{field_name}"
            agreement_counts[agreement_key] = agreement_count
            conflict_counts[conflict_key] = conflict_count
            summary_rows.extend(
                (
                    {
                        "summary_category": f"agreement_count_same_{field_name}",
                        "summary_key": pair_name,
                        "value": agreement_count,
                    },
                    {
                        "summary_category": f"conflict_count_same_{field_name}",
                        "summary_key": pair_name,
                        "value": conflict_count,
                    },
                )
            )

    all_consistent_true = int(
        extended.select(pl.col("all_successful_attempts_consistent").fill_null(False).cast(pl.Int64).sum()).item()
    )
    all_consistent_false = int(
        extended.select(
            pl.when(pl.col("all_successful_attempts_consistent").is_null())
            .then(pl.lit(0, dtype=pl.Int64))
            .otherwise((~pl.col("all_successful_attempts_consistent")).cast(pl.Int64))
            .sum()
        ).item()
    )
    agreement_counts["all_successful_attempts_consistent"] = all_consistent_true
    conflict_counts["all_successful_attempts_consistent"] = all_consistent_false
    summary_rows.extend(
        (
            {
                "summary_category": "agreement_count_all_successful_attempts_consistent",
                "summary_key": "all_successful_attempts_consistent",
                "value": all_consistent_true,
            },
            {
                "summary_category": "conflict_count_all_successful_attempts_consistent",
                "summary_key": "all_successful_attempts_consistent",
                "value": all_consistent_false,
            },
        )
    )

    single_success_counts: dict[str, int] = {}
    for identifier_type in LOOKUP_IDENTIFIER_TYPES:
        only_this_success_count = int(
            extended.select(
                (
                    pl.col(f"{identifier_type}_success")
                    & pl.all_horizontal(
                        [
                            (~pl.col(f"{other_identifier_type}_success"))
                            for other_identifier_type in LOOKUP_IDENTIFIER_TYPES
                            if other_identifier_type != identifier_type
                        ]
                    )
                )
                .cast(pl.Int64)
                .sum()
            ).item()
        )
        single_success_counts[identifier_type] = only_this_success_count
        summary_rows.append(
            {
                "summary_category": "only_one_identifier_type_succeeds",
                "summary_key": identifier_type,
                "value": only_this_success_count,
            }
        )

    rows_where_only_one_identifier_type_succeeds = int(sum(single_success_counts.values()))
    summary_rows.append(
        {
            "summary_category": "only_one_identifier_type_succeeds",
            "summary_key": "total",
            "value": rows_where_only_one_identifier_type_succeeds,
        }
    )

    summary_df = pl.DataFrame(summary_rows).select(
        pl.col("summary_category").cast(pl.Utf8, strict=False),
        pl.col("summary_key").cast(pl.Utf8, strict=False),
        pl.col("value").cast(pl.Int64, strict=False),
    )
    summary_payload = {
        "attempt_counts_by_identifier_type": attempt_counts_by_identifier_type,
        "success_counts_by_identifier_type": success_counts_by_identifier_type,
        "agreement_counts": agreement_counts,
        "conflict_counts": conflict_counts,
        "rows_where_only_one_identifier_type_succeeds": rows_where_only_one_identifier_type_succeeds,
    }
    return extended, summary_df, summary_payload


def _write_json(out_path: Path, payload: dict[str, Any]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def run_refinitiv_step1_bridge_pipeline(
    daily_lf: pl.LazyFrame,
    output_dir: Path,
    *,
    company_description_lf: pl.LazyFrame | None = None,
    source_daily_path: Path | None = None,
) -> dict[str, Path]:
    """Write the compact Refinitiv step-1 bridge artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge_df = build_refinitiv_step1_bridge_universe(
        daily_lf,
        company_description_lf=company_description_lf,
    ).collect()
    handoff_df = _build_handoff_df(bridge_df)

    parquet_path = output_dir / "refinitiv_bridge_universe.parquet"
    csv_path = output_dir / "refinitiv_bridge_universe.csv"
    xlsx_path = output_dir / "refinitiv_bridge_handoff.xlsx"
    ric_lookup_xlsx_path = output_dir / "refinitiv_ric_lookup_handoff.xlsx"
    manifest_path = output_dir / "refinitiv_step1_manifest.json"

    ric_lookup_df, ric_manual_review_df = _build_ric_lookup_handoff_frames(bridge_df)

    bridge_df.write_parquet(parquet_path, compression="zstd")
    handoff_df.write_csv(csv_path)

    source_rows = int(bridge_df["n_daily_rows"].sum()) if bridge_df.height else 0
    bridge_rows = bridge_df.height
    distinct_permno = bridge_df["KYPERMNO"].n_unique() if bridge_df.height else 0
    with_vendor_id = (
        bridge_df.select(
            (
                pl.col("CUSIP").is_not_null()
                | pl.col("ISIN").is_not_null()
                | pl.col("TICKER").is_not_null()
            ).sum()
        ).item()
        if bridge_df.height
        else 0
    )
    generated_at_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest_payload: dict[str, Any] = {
        "pipeline_name": "refinitiv_step1_bridge",
        "artifact_version": "v1",
        "generated_at_utc": generated_at_utc,
        "source_daily_path": str(source_daily_path) if source_daily_path is not None else None,
        "bridge_rows": bridge_rows,
        "source_daily_rows": source_rows,
        "distinct_permno": int(distinct_permno),
        "rows_with_vendor_identifier": int(with_vendor_id),
        "rows_missing_vendor_identifier": int(bridge_rows - with_vendor_id),
        "ric_lookup_rows": int(ric_lookup_df.height),
        "ric_manual_review_rows": int(ric_manual_review_df.height),
        "authoritative_format": "parquet",
        "source_columns": list(BRIDGE_SOURCE_COLUMNS),
        "vendor_columns": list(BRIDGE_VENDOR_COLUMNS),
        "ric_lookup_columns": list(RIC_LOOKUP_COLUMNS),
        "ric_lookup_filter_profiles": [],
        "artifacts": {
            "refinitiv_bridge_universe_parquet": str(parquet_path),
            "refinitiv_bridge_universe_csv": str(csv_path),
            "refinitiv_bridge_handoff_xlsx": str(xlsx_path),
            "refinitiv_ric_lookup_handoff_xlsx": str(ric_lookup_xlsx_path),
            "refinitiv_step1_manifest": str(manifest_path),
        },
    }

    for profile in RIC_LOOKUP_FILTER_PROFILES:
        qualifying_bridge_ids = _build_lookup_profile_bridge_ids(daily_lf, profile)
        filtered_lookup_df, filtered_manual_review_df, profile_summary = (
            _build_filtered_ric_lookup_profile_artifact(
                bridge_df,
                qualifying_bridge_ids,
                profile,
            )
        )
        profile_path = _profile_output_path(output_dir, profile.name)
        profile_key = _profile_output_key(profile.name)
        profile_manifest_payload = {
            "pipeline_name": manifest_payload["pipeline_name"],
            "artifact_version": manifest_payload["artifact_version"],
            "generated_at_utc": manifest_payload["generated_at_utc"],
            "source_daily_path": manifest_payload["source_daily_path"],
            "lookup_filter_profile": profile_summary,
            "ric_lookup_columns": manifest_payload["ric_lookup_columns"],
        }
        write_refinitiv_ric_lookup_workbook(
            filtered_lookup_df,
            profile_path,
            readme_payload=profile_manifest_payload,
            text_columns=RIC_LOOKUP_TEXT_COLUMNS,
            manual_review_df=filtered_manual_review_df,
        )
        profile_manifest_summary: dict[str, Any] = {
            **profile_summary,
            "artifact_key": profile_key,
            "artifact_path": str(profile_path),
        }
        manifest_payload["artifacts"][profile_key] = str(profile_path)

        if profile.name == "common_stock":
            extended_profile_path = _extended_profile_output_path(output_dir, profile.name)
            extended_profile_key = _extended_profile_output_key(profile.name)
            extended_df, extended_summary_df, extended_summary_payload = (
                build_refinitiv_lookup_extended_diagnostic_artifact(
                    filtered_lookup_df,
                    filtered_manual_review_df,
                )
            )
            extended_manifest_payload = {
                **profile_manifest_payload,
                "lookup_filter_profile": profile_manifest_summary,
                "extended_lookup_diagnostic_summary": extended_summary_payload,
            }
            write_refinitiv_ric_lookup_extended_workbook(
                extended_df,
                extended_profile_path,
                readme_payload=extended_manifest_payload,
                text_columns=RIC_LOOKUP_EXTENDED_TEXT_COLUMNS,
                summary_df=extended_summary_df,
                summary_text_columns=RIC_LOOKUP_EXTENDED_SUMMARY_TEXT_COLUMNS,
            )
            profile_manifest_summary["extended_diagnostic_artifact_key"] = extended_profile_key
            profile_manifest_summary["extended_diagnostic_artifact_path"] = str(extended_profile_path)
            profile_manifest_summary["extended_diagnostic_summary"] = extended_summary_payload
            manifest_payload["artifacts"][extended_profile_key] = str(extended_profile_path)

        manifest_payload["ric_lookup_filter_profiles"].append(profile_manifest_summary)

    write_refinitiv_bridge_workbook(
        handoff_df,
        xlsx_path,
        readme_payload=manifest_payload,
        text_columns=WORKBOOK_TEXT_COLUMNS,
    )
    write_refinitiv_ric_lookup_workbook(
        ric_lookup_df,
        ric_lookup_xlsx_path,
        readme_payload=manifest_payload,
        text_columns=RIC_LOOKUP_TEXT_COLUMNS,
        manual_review_df=ric_manual_review_df,
    )
    _write_json(manifest_path, manifest_payload)

    return {
        "refinitiv_bridge_universe_parquet": parquet_path,
        "refinitiv_bridge_universe_csv": csv_path,
        "refinitiv_bridge_handoff_xlsx": xlsx_path,
        "refinitiv_ric_lookup_handoff_xlsx": ric_lookup_xlsx_path,
        "refinitiv_step1_manifest": manifest_path,
        **{
            _profile_output_key(profile.name): _profile_output_path(output_dir, profile.name)
            for profile in RIC_LOOKUP_FILTER_PROFILES
        },
        _extended_profile_output_key("common_stock"): _extended_profile_output_path(output_dir, "common_stock"),
    }


def _normalize_workbook_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _normalize_lookup_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _read_refinitiv_ric_lookup_sheet(
    workbook_path: Path | str,
    *,
    sheet_name: str = "ric_lookup",
) -> pl.DataFrame:
    from openpyxl import load_workbook

    workbook = load_workbook(Path(workbook_path), read_only=True, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f"lookup workbook missing sheet: {sheet_name}")
        worksheet = workbook[sheet_name]
        rows = worksheet.iter_rows(values_only=True)
        header = next(rows, None)
        if header is None:
            raise ValueError(f"lookup workbook sheet {sheet_name!r} is empty")
        column_names = tuple(_normalize_workbook_scalar(name) for name in header)
        if any(name is None for name in column_names):
            raise ValueError(f"lookup workbook sheet {sheet_name!r} has blank headers")
        missing = [name for name in RIC_LOOKUP_COLUMNS if name not in column_names]
        if missing:
            raise ValueError(f"lookup workbook sheet {sheet_name!r} missing required columns: {missing}")

        records: list[dict[str, str | None]] = []
        for row in rows:
            record = {
                str(column_names[idx]): _normalize_workbook_scalar(row[idx]) if idx < len(row) else None
                for idx in range(len(column_names))
            }
            records.append(record)
    finally:
        workbook.close()

    if not records:
        return pl.DataFrame(schema={name: pl.Utf8 for name in RIC_LOOKUP_COLUMNS}).with_columns(
            pl.col("first_seen_caldt").cast(pl.Date, strict=False),
            pl.col("last_seen_caldt").cast(pl.Date, strict=False),
        )

    df = pl.DataFrame(records).select(
        [
            pl.col(name).cast(pl.Utf8, strict=False).alias(name)
            if name not in ("first_seen_caldt", "last_seen_caldt")
            else pl.col(name).cast(pl.Utf8, strict=False).str.strptime(pl.Date, strict=False).alias(name)
            for name in RIC_LOOKUP_COLUMNS
        ]
    )
    return df


def _is_failed_lookup_record(record: dict[str, Any]) -> tuple[bool, str, bool]:
    vendor_primary_ric = _normalize_lookup_text(record.get("vendor_primary_ric"))
    returned_fields = (
        _normalize_lookup_text(record.get("vendor_returned_name")),
        _normalize_lookup_text(record.get("vendor_returned_cusip")),
        _normalize_lookup_text(record.get("vendor_returned_isin")),
    )
    ric_missing = vendor_primary_ric is None or vendor_primary_ric.upper() == "NULL"
    invalid_identifier_signal = any(
        value is not None and "invalid identifier" in value.lower()
        for value in returned_fields
    )
    failed_lookup = ric_missing or invalid_identifier_signal
    if ric_missing and invalid_identifier_signal:
        failed_reason = "null_ric_and_invalid_identifier"
    elif ric_missing:
        failed_reason = "null_ric"
    elif invalid_identifier_signal:
        failed_reason = "invalid_identifier"
    else:
        failed_reason = "successful_lookup"
    return failed_lookup, failed_reason, invalid_identifier_signal


def _alternative_identifier(record: dict[str, Any]) -> tuple[str | None, str | None]:
    preferred_lookup_type = _normalize_lookup_text(record.get("preferred_lookup_type"))
    preferred_lookup_id = _normalize_lookup_text(record.get("preferred_lookup_id"))
    if preferred_lookup_type == "ISIN":
        candidates = (
            (_normalize_lookup_text(record.get("CUSIP")), "CUSIP"),
            (_normalize_lookup_text(record.get("TICKER")), "TICKER"),
        )
    elif preferred_lookup_type == "CUSIP":
        candidates = (
            (_normalize_lookup_text(record.get("ISIN")), "ISIN"),
            (_normalize_lookup_text(record.get("TICKER")), "TICKER"),
        )
    elif preferred_lookup_type == "TICKER":
        candidates = (
            (_normalize_lookup_text(record.get("ISIN")), "ISIN"),
            (_normalize_lookup_text(record.get("CUSIP")), "CUSIP"),
        )
    else:
        return None, None

    for candidate, candidate_type in candidates:
        if candidate is not None and candidate != preferred_lookup_id:
            return candidate, candidate_type
    return None, None


def build_refinitiv_null_ric_rescue_candidates(
    lookup_df: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    required_columns = list(RIC_LOOKUP_COLUMNS)
    missing = [name for name in required_columns if name not in lookup_df.columns]
    if missing:
        raise ValueError(f"lookup diagnostics input missing required columns: {missing}")

    normalized_df = lookup_df.select(
        pl.col("bridge_row_id").cast(pl.Utf8, strict=False),
        pl.col("KYPERMNO").cast(pl.Utf8, strict=False),
        pl.col("CUSIP").cast(pl.Utf8, strict=False),
        pl.col("ISIN").cast(pl.Utf8, strict=False),
        pl.col("TICKER").cast(pl.Utf8, strict=False),
        pl.col("first_seen_caldt").cast(pl.Date, strict=False),
        pl.col("last_seen_caldt").cast(pl.Date, strict=False),
        pl.col("preferred_lookup_id").cast(pl.Utf8, strict=False),
        pl.col("preferred_lookup_type").cast(pl.Utf8, strict=False),
        pl.col("vendor_primary_ric").cast(pl.Utf8, strict=False),
        pl.col("vendor_returned_name").cast(pl.Utf8, strict=False),
        pl.col("vendor_returned_cusip").cast(pl.Utf8, strict=False),
        pl.col("vendor_returned_isin").cast(pl.Utf8, strict=False),
        pl.col("vendor_match_status").cast(pl.Utf8, strict=False),
        pl.col("vendor_notes").cast(pl.Utf8, strict=False),
    )

    all_records = normalized_df.to_dicts()
    successful_by_kypermno: dict[str, list[dict[str, Any]]] = {}
    failure_reason_counts: Counter[str] = Counter()
    successful_rows = 0

    for record in all_records:
        failed_lookup, failed_reason, invalid_identifier_signal = _is_failed_lookup_record(record)
        record["failed_lookup_flag"] = failed_lookup
        record["failed_lookup_reason"] = failed_reason
        record["invalid_identifier_signal"] = invalid_identifier_signal
        if failed_lookup:
            failure_reason_counts[failed_reason] += 1
        else:
            successful_rows += 1
            kypermno = _normalize_lookup_text(record.get("KYPERMNO"))
            if kypermno is not None:
                successful_by_kypermno.setdefault(kypermno, []).append(record)

    diagnostic_rows: list[dict[str, Any]] = []

    for record in all_records:
        if not bool(record["failed_lookup_flag"]):
            continue

        kypermno = _normalize_lookup_text(record.get("KYPERMNO"))
        successful_rows_same_permno = successful_by_kypermno.get(kypermno or "", [])
        current_first_seen = record.get("first_seen_caldt")
        current_last_seen = record.get("last_seen_caldt")

        before_gaps: list[int] = []
        after_gaps: list[int] = []
        overlap_exists = False
        successful_identifier_pairs: set[tuple[str | None, str | None]] = set()
        successful_rics: set[str] = set()

        for successful in successful_rows_same_permno:
            successful_identifier_pairs.add(
                (
                    _normalize_lookup_text(successful.get("CUSIP")),
                    _normalize_lookup_text(successful.get("ISIN")),
                )
            )
            successful_ric = _normalize_lookup_text(successful.get("vendor_primary_ric"))
            if successful_ric is not None:
                successful_rics.add(successful_ric)

            successful_first_seen = successful.get("first_seen_caldt")
            successful_last_seen = successful.get("last_seen_caldt")
            if (
                isinstance(current_first_seen, dt.date)
                and isinstance(current_last_seen, dt.date)
                and isinstance(successful_first_seen, dt.date)
                and isinstance(successful_last_seen, dt.date)
            ):
                if successful_last_seen < current_first_seen:
                    before_gaps.append((current_first_seen - successful_last_seen).days)
                elif successful_first_seen > current_last_seen:
                    after_gaps.append((successful_first_seen - current_last_seen).days)
                else:
                    overlap_exists = True

        nearest_before = min(before_gaps) if before_gaps else None
        nearest_after = min(after_gaps) if after_gaps else None
        failed_pair = (
            _normalize_lookup_text(record.get("CUSIP")),
            _normalize_lookup_text(record.get("ISIN")),
        )
        failed_pair_matches_success = failed_pair in successful_identifier_pairs
        alt_identifier, alt_identifier_type = _alternative_identifier(record)

        unique_successful_pairs = {
            pair for pair in successful_identifier_pairs if pair != (None, None)
        }
        candidate_successful_ric = next(iter(successful_rics)) if len(successful_rics) == 1 else None
        candidate_successful_ric_available = candidate_successful_ric is not None
        if len(unique_successful_pairs) == 1:
            candidate_successful_cusip, candidate_successful_isin = next(iter(unique_successful_pairs))
        else:
            candidate_successful_cusip, candidate_successful_isin = None, None

        has_success = bool(successful_rows_same_permno)
        alternative_identifier_available = alt_identifier is not None
        diagnostic_rows.append(
            {
                **{name: record.get(name) for name in RIC_LOOKUP_COLUMNS},
                "failed_lookup_flag": True,
                "failed_lookup_reason": record["failed_lookup_reason"],
                "invalid_identifier_signal": bool(record["invalid_identifier_signal"]),
                "successful_row_exists_for_kypermno": has_success,
                "successful_row_exists_before_span": bool(before_gaps),
                "successful_row_exists_after_span": bool(after_gaps),
                "successful_row_overlap_exists": overlap_exists,
                "nearest_successful_gap_days_before": nearest_before,
                "nearest_successful_gap_days_after": nearest_after,
                "unique_successful_identifier_pair_count": len(unique_successful_pairs),
                "unique_successful_ric_count": len(successful_rics),
                "failed_identifier_pair_matches_success": failed_pair_matches_success,
                "alternative_identifier_available": alternative_identifier_available,
                "alternative_identifier": alt_identifier,
                "alternative_identifier_type": alt_identifier_type,
                "candidate_successful_ric_available": candidate_successful_ric_available,
                "candidate_successful_ric": candidate_successful_ric,
                "candidate_successful_cusip": candidate_successful_cusip,
                "candidate_successful_isin": candidate_successful_isin,
            }
        )

    diagnostics_df = pl.DataFrame(diagnostic_rows) if diagnostic_rows else pl.DataFrame(schema={
        name: pl.Utf8 for name in NULL_RIC_DIAGNOSTIC_COLUMNS
    })
    if diagnostics_df.height:
        diagnostics_df = diagnostics_df.select(
            [
                pl.col(name).cast(pl.Date, strict=False).alias(name)
                if name in ("first_seen_caldt", "last_seen_caldt")
                else pl.col(name)
                for name in diagnostics_df.columns
            ]
        ).select(NULL_RIC_DIAGNOSTIC_COLUMNS)
    else:
        diagnostics_df = diagnostics_df.with_columns(
            pl.col("first_seen_caldt").cast(pl.Date, strict=False),
            pl.col("last_seen_caldt").cast(pl.Date, strict=False),
        )

    summary: dict[str, Any] = {
        "total_lookup_rows": int(len(all_records)),
        "successful_lookup_rows": int(successful_rows),
        "failed_lookup_rows": int(len(diagnostic_rows)),
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "diagnostic_flag_counts": {
            "successful_row_exists_before_span": int(
                diagnostics_df.select(pl.col("successful_row_exists_before_span").cast(pl.Int64).sum()).item()
            )
            if diagnostics_df.height
            else 0,
            "successful_row_exists_after_span": int(
                diagnostics_df.select(pl.col("successful_row_exists_after_span").cast(pl.Int64).sum()).item()
            )
            if diagnostics_df.height
            else 0,
            "successful_row_overlap_exists": int(
                diagnostics_df.select(pl.col("successful_row_overlap_exists").cast(pl.Int64).sum()).item()
            )
            if diagnostics_df.height
            else 0,
            "alternative_identifier_available": int(
                diagnostics_df.select(pl.col("alternative_identifier_available").cast(pl.Int64).sum()).item()
            )
            if diagnostics_df.height
            else 0,
            "candidate_successful_ric_available": int(
                diagnostics_df.select(pl.col("candidate_successful_ric_available").cast(pl.Int64).sum()).item()
            )
            if diagnostics_df.height
            else 0,
            "no_successful_row_for_kypermno": int(
                diagnostics_df.select((~pl.col("successful_row_exists_for_kypermno")).cast(pl.Int64).sum()).item()
            )
            if diagnostics_df.height
            else 0,
            "multiple_successful_identifier_pairs_or_rics": int(
                diagnostics_df.select(
                    (
                        (pl.col("unique_successful_identifier_pair_count") > 1)
                        | (pl.col("unique_successful_ric_count") > 1)
                    )
                    .cast(pl.Int64)
                    .sum()
                ).item()
            )
            if diagnostics_df.height
            else 0,
        },
        "rows_with_alternative_identifier": int(
            diagnostics_df.select(pl.col("alternative_identifier").is_not_null().sum()).item()
        )
        if diagnostics_df.height
        else 0,
    }
    return diagnostics_df, summary


def _build_null_ric_review_frame(diagnostics_df: pl.DataFrame) -> pl.DataFrame:
    review_df = diagnostics_df.select(
        [
            *[pl.col(name) for name in NULL_RIC_REVIEW_COLUMNS if name in diagnostics_df.columns],
            pl.lit(None, dtype=pl.Utf8).alias("test_category"),
            pl.lit(None, dtype=pl.Utf8).alias("test_method"),
            pl.lit(None, dtype=pl.Utf8).alias("test_result"),
            pl.lit(None, dtype=pl.Utf8).alias("test_notes"),
        ]
    )
    return review_df.select(NULL_RIC_REVIEW_COLUMNS)


def _date_to_text(value: Any) -> str | None:
    if isinstance(value, dt.date):
        return value.isoformat()
    normalized = _normalize_lookup_text(value)
    return normalized


def _explicit_sample_category_field(review_df: pl.DataFrame) -> str | None:
    for field_name in ("sample_category", "test_category"):
        if field_name not in review_df.columns:
            continue
        if review_df.select(pl.col(field_name).drop_nulls().len()).item() > 0:
            non_blank_count = review_df.select(
                pl.col(field_name)
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .replace("", None)
                .drop_nulls()
                .len()
            ).item()
            if int(non_blank_count) > 0:
                return field_name
    return None


def _resolve_ownership_lookup_input(row: dict[str, Any]) -> tuple[str | None, str]:
    for field_name, source_name in (
        ("candidate_successful_ric", "candidate_successful_ric"),
        ("alternative_identifier", "alternative_identifier"),
        ("TICKER", "TICKER"),
        ("preferred_lookup_id", "preferred_lookup_id"),
    ):
        value = _normalize_lookup_text(row.get(field_name))
        if value is not None:
            return value, source_name
    return None, "preferred_lookup_id"


def _ownership_smoke_sample_row(sample_category: str, row: dict[str, Any]) -> dict[str, Any]:
    lookup_input, lookup_input_source = _resolve_ownership_lookup_input(row)
    return {
        "sample_category": sample_category,
        "bridge_row_id": _normalize_lookup_text(row.get("bridge_row_id")),
        "KYPERMNO": _normalize_lookup_text(row.get("KYPERMNO")),
        "TICKER": _normalize_lookup_text(row.get("TICKER")),
        "lookup_input": lookup_input,
        "lookup_input_source": lookup_input_source,
        "request_start_date": _date_to_text(row.get("first_seen_caldt")),
        "request_end_date": _date_to_text(row.get("last_seen_caldt")),
        "preferred_lookup_id": _normalize_lookup_text(row.get("preferred_lookup_id")),
        "preferred_lookup_type": _normalize_lookup_text(row.get("preferred_lookup_type")),
        "alternative_identifier": _normalize_lookup_text(row.get("alternative_identifier")),
        "alternative_identifier_type": _normalize_lookup_text(row.get("alternative_identifier_type")),
        "candidate_successful_ric": _normalize_lookup_text(row.get("candidate_successful_ric")),
        "successful_row_exists_before_span": bool(row.get("successful_row_exists_before_span")),
        "successful_row_exists_after_span": bool(row.get("successful_row_exists_after_span")),
        "successful_row_overlap_exists": bool(row.get("successful_row_overlap_exists")),
        "alternative_identifier_available": bool(row.get("alternative_identifier_available")),
        "candidate_successful_ric_available": bool(row.get("candidate_successful_ric_available")),
        "unique_successful_identifier_pair_count": int(
            row.get("unique_successful_identifier_pair_count") or 0
        ),
        "unique_successful_ric_count": int(row.get("unique_successful_ric_count") or 0),
    }


def build_refinitiv_ownership_smoke_sample(
    review_df: pl.DataFrame,
    *,
    target_block_count: int = 10,
    min_per_category: int = 2,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    sorted_rows = sorted(review_df.to_dicts(), key=lambda row: str(row.get("bridge_row_id") or ""))
    explicit_category_field = _explicit_sample_category_field(review_df)

    category_pools: list[tuple[str, list[dict[str, Any]]]] = []
    if explicit_category_field is not None:
        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for row in sorted_rows:
            category_value = _normalize_lookup_text(row.get(explicit_category_field))
            if category_value is None:
                continue
            grouped_rows.setdefault(category_value, []).append(row)
        category_pools = [(name, grouped_rows[name]) for name in sorted(grouped_rows)]
        category_source = explicit_category_field
    else:
        category_pools = [
            (category_name, [row for row in sorted_rows if predicate(row)])
            for category_name, predicate in OWNERSHIP_SMOKE_CATEGORY_SPECS
        ]
        category_source = "derived_from_review_flags"

    category_pools = [(name, rows) for name, rows in category_pools if rows]
    sampled_rows: list[dict[str, Any]] = []
    sample_counts: dict[str, int] = {name: 0 for name, _ in category_pools}

    for category_name, rows in category_pools:
        for row in rows[:min(min_per_category, len(rows))]:
            sampled_rows.append(_ownership_smoke_sample_row(category_name, row))
            sample_counts[category_name] += 1

    if len(sampled_rows) < target_block_count:
        for category_name, rows in category_pools:
            start_idx = sample_counts[category_name]
            for row in rows[start_idx:]:
                if len(sampled_rows) >= target_block_count:
                    break
                sampled_rows.append(_ownership_smoke_sample_row(category_name, row))
                sample_counts[category_name] += 1
            if len(sampled_rows) >= target_block_count:
                break

    if sampled_rows:
        sample_df = pl.DataFrame(sampled_rows).select(OWNERSHIP_SMOKE_SAMPLE_COLUMNS)
    else:
        sample_df = pl.DataFrame(schema=OWNERSHIP_SMOKE_SAMPLE_SCHEMA)

    metadata: dict[str, Any] = {
        "sample_category_counts": {name: count for name, count in sample_counts.items() if count > 0},
        "available_category_counts": {name: len(rows) for name, rows in category_pools},
        "category_order": [name for name, _ in category_pools],
        "category_field_used": "sample_category",
        "category_source": category_source,
        "sample_source_name": "failed_lookup_review",
        "target_block_count": int(target_block_count),
        "min_per_category": int(min_per_category),
        "block_count": int(sample_df.height),
        "lookup_input_priority": [
            "candidate_successful_ric",
            "alternative_identifier",
            "TICKER",
            "preferred_lookup_id",
        ],
    }
    return sample_df, metadata


def run_refinitiv_null_ric_diagnostics_pipeline(
    filled_lookup_workbook_path: Path | str,
    output_dir: Path | str,
    *,
    emit_review_workbook: bool = True,
    emit_ownership_smoke_workbook: bool = True,
    ownership_target_block_count: int = 10,
    ownership_min_per_category: int = 2,
) -> dict[str, Path]:
    filled_lookup_workbook_path = Path(filled_lookup_workbook_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lookup_df = _read_refinitiv_ric_lookup_sheet(filled_lookup_workbook_path)
    diagnostics_df, summary = build_refinitiv_null_ric_rescue_candidates(lookup_df)
    review_df = _build_null_ric_review_frame(diagnostics_df)
    ownership_smoke_df, ownership_smoke_metadata = build_refinitiv_ownership_smoke_sample(
        review_df,
        target_block_count=ownership_target_block_count,
        min_per_category=ownership_min_per_category,
    )

    summary_path = output_dir / "refinitiv_null_ric_diagnostics_summary.json"
    manifest_path = output_dir / "refinitiv_null_ric_diagnostics_manifest.json"
    rescue_parquet_path = output_dir / "refinitiv_null_ric_rescue_candidates.parquet"
    rescue_csv_path = output_dir / "refinitiv_null_ric_rescue_candidates.csv"
    review_workbook_path = output_dir / "refinitiv_null_ric_rescue_candidates_review.xlsx"
    ownership_smoke_parquet_path = output_dir / "refinitiv_ownership_smoke_testing.parquet"
    ownership_smoke_csv_path = output_dir / "refinitiv_ownership_smoke_testing.csv"
    ownership_smoke_workbook_path = output_dir / "refinitiv_ownership_smoke_testing.xlsx"

    diagnostics_df.write_parquet(rescue_parquet_path, compression="zstd")
    diagnostics_df.write_csv(rescue_csv_path)
    ownership_smoke_df.write_parquet(ownership_smoke_parquet_path, compression="zstd")
    ownership_smoke_df.write_csv(ownership_smoke_csv_path)

    summary_payload: dict[str, Any] = {
        "pipeline_name": "refinitiv_null_ric_diagnostics",
        "artifact_version": "v2",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_filled_lookup_workbook": str(filled_lookup_workbook_path),
        "source_sheet_name": "ric_lookup",
        "failed_lookup_definition": [
            "vendor_primary_ric is null/blank/NULL",
            "or vendor_returned_* contains an invalid identifier signal",
        ],
        "ownership_smoke_sample_source": ownership_smoke_metadata["sample_source_name"],
        "ownership_smoke_category_field_used": ownership_smoke_metadata["category_field_used"],
        "ownership_smoke_category_source": ownership_smoke_metadata["category_source"],
        "ownership_smoke_target_block_count": ownership_smoke_metadata["target_block_count"],
        "ownership_smoke_min_per_category": ownership_smoke_metadata["min_per_category"],
        "ownership_smoke_block_count": ownership_smoke_metadata["block_count"],
        "ownership_smoke_category_counts": ownership_smoke_metadata["sample_category_counts"],
        "ownership_smoke_available_category_counts": ownership_smoke_metadata["available_category_counts"],
        "ownership_smoke_category_order": ownership_smoke_metadata["category_order"],
        "ownership_smoke_lookup_input_priority": ownership_smoke_metadata["lookup_input_priority"],
        **summary,
    }
    _write_json(summary_path, summary_payload)

    manifest_payload: dict[str, Any] = {
        **summary_payload,
        "diagnostic_columns": list(NULL_RIC_DIAGNOSTIC_COLUMNS),
        "review_columns": list(NULL_RIC_REVIEW_COLUMNS),
        "ownership_smoke_columns": list(OWNERSHIP_SMOKE_SAMPLE_COLUMNS),
        "artifacts": {
            "refinitiv_null_ric_diagnostics_summary": str(summary_path),
            "refinitiv_null_ric_rescue_candidates_parquet": str(rescue_parquet_path),
            "refinitiv_null_ric_rescue_candidates_csv": str(rescue_csv_path),
            "refinitiv_ownership_smoke_testing_parquet": str(ownership_smoke_parquet_path),
            "refinitiv_ownership_smoke_testing_csv": str(ownership_smoke_csv_path),
            "refinitiv_null_ric_diagnostics_manifest": str(manifest_path),
        },
    }

    if emit_review_workbook:
        write_refinitiv_null_ric_diagnostics_workbook(
            _coerce_text_columns(review_df, NULL_RIC_REVIEW_TEXT_COLUMNS),
            review_workbook_path,
            readme_payload=manifest_payload,
            text_columns=NULL_RIC_REVIEW_TEXT_COLUMNS,
            sheet_name="failed_lookup_review",
        )
        manifest_payload["artifacts"]["refinitiv_null_ric_rescue_candidates_review_xlsx"] = str(
            review_workbook_path
        )
    if emit_ownership_smoke_workbook:
        write_refinitiv_ownership_smoke_testing_workbook(
            ownership_smoke_df,
            ownership_smoke_workbook_path,
            readme_payload=manifest_payload,
        )
        manifest_payload["artifacts"]["refinitiv_ownership_smoke_testing_xlsx"] = str(
            ownership_smoke_workbook_path
        )

    _write_json(manifest_path, manifest_payload)

    paths: dict[str, Path] = {
        "refinitiv_null_ric_diagnostics_summary": summary_path,
        "refinitiv_null_ric_rescue_candidates_parquet": rescue_parquet_path,
        "refinitiv_null_ric_rescue_candidates_csv": rescue_csv_path,
        "refinitiv_ownership_smoke_testing_parquet": ownership_smoke_parquet_path,
        "refinitiv_ownership_smoke_testing_csv": ownership_smoke_csv_path,
        "refinitiv_null_ric_diagnostics_manifest": manifest_path,
    }
    if emit_review_workbook:
        paths["refinitiv_null_ric_rescue_candidates_review_xlsx"] = review_workbook_path
    if emit_ownership_smoke_workbook:
        paths["refinitiv_ownership_smoke_testing_xlsx"] = ownership_smoke_workbook_path
    return paths
