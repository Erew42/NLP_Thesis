from __future__ import annotations

import datetime as dt
from calendar import monthrange
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.core.ccm.lm2011 import attach_eligible_quarterly_accounting
from thesis_pkg.pipelines.refinitiv.instrument_authority import (
    INSTRUMENT_AUTHORITY_COLUMNS,
)
from thesis_pkg.pipelines.refinitiv.lseg_batching import stable_hash_id
from thesis_pkg.pipelines.refinitiv.doc_ownership import (
    _normalize_float_value,
)
from thesis_pkg.pipelines.refinitiv_bridge_pipeline import (
    _build_explicit_schema_df,
    _cast_df_to_schema,
)


ANALYST_REQUEST_GROUP_MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "request_group_id",
    "bridge_row_id",
    "gvkey_int",
    "KYPERMNO",
    "effective_collection_ric",
)

ANALYST_REQUEST_GROUP_COLUMNS: tuple[str, ...] = (
    "request_group_id",
    "gvkey_int",
    "effective_collection_ric",
    "member_bridge_row_count",
    "bridge_start_date_min",
    "bridge_end_date_max",
    "actuals_request_start_date",
    "actuals_request_end_date",
    "estimates_request_start_date",
    "estimates_request_end_date",
    "retrieval_eligible",
    "retrieval_exclusion_reason",
)

ANALYST_ACTUALS_RAW_COLUMNS: tuple[str, ...] = (
    "item_id",
    "response_row_index",
    "request_group_id",
    "gvkey_int",
    "effective_collection_ric",
    "announcement_date",
    "fiscal_period_end",
    "actual_eps",
    "raw_fperiod",
    "row_parse_status",
)

ANALYST_ESTIMATES_MONTHLY_RAW_COLUMNS: tuple[str, ...] = (
    "item_id",
    "response_row_index",
    "request_group_id",
    "request_period",
    "gvkey_int",
    "effective_collection_ric",
    "calc_date",
    "fiscal_period_end",
    "raw_fperiod",
    "forecast_consensus_mean",
    "forecast_dispersion",
    "estimate_count",
    "row_parse_status",
)

ANALYST_NORMALIZED_PANEL_COLUMNS: tuple[str, ...] = (
    "gvkey_int",
    "announcement_date",
    "fiscal_period_end",
    "actual_fiscal_period_end_origin",
    "actual_eps",
    "forecast_consensus_mean",
    "forecast_dispersion",
    "forecast_revision_4m",
    "forecast_revision_4m_status",
    "forecast_revision_1m",
    "selected_forecast_calc_date",
    "revision_base_calc_date_4m",
    "revision_base_calc_date_1m",
    "effective_collection_ric",
    "source_request_group_ids",
)

ANALYST_NORMALIZATION_REJECTION_COLUMNS: tuple[str, ...] = (
    "rejection_case_id",
    "gvkey_int",
    "announcement_date",
    "fiscal_period_end",
    "selected_calc_date",
    "rejection_status",
    "rejection_reason",
    "source_request_group_ids",
)

DOC_ANALYST_ANCHOR_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "filing_date",
    "gvkey_int",
    "KYPERMNO",
    "quarter_report_date",
    "quarter_fiscal_period_end",
    "anchor_eligible",
    "anchor_exclusion_reason",
)

DOC_ANALYST_SELECTED_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "filing_date",
    "gvkey_int",
    "KYPERMNO",
    "quarter_report_date",
    "quarter_fiscal_period_end",
    "matched_announcement_date",
    "matched_fiscal_period_end",
    "actual_fiscal_period_end_origin",
    "match_type",
    "actual_eps",
    "forecast_consensus_mean",
    "forecast_dispersion",
    "forecast_revision_4m",
    "forecast_revision_4m_status",
    "forecast_revision_1m",
    "analyst_match_status",
)


def _request_group_membership_schema() -> dict[str, pl.DataType]:
    return {
        "request_group_id": pl.Utf8,
        "bridge_row_id": pl.Utf8,
        "gvkey_int": pl.Int32,
        "KYPERMNO": pl.Int32,
        "effective_collection_ric": pl.Utf8,
    }


def _request_group_schema() -> dict[str, pl.DataType]:
    return {
        "request_group_id": pl.Utf8,
        "gvkey_int": pl.Int32,
        "effective_collection_ric": pl.Utf8,
        "member_bridge_row_count": pl.Int64,
        "bridge_start_date_min": pl.Date,
        "bridge_end_date_max": pl.Date,
        "actuals_request_start_date": pl.Date,
        "actuals_request_end_date": pl.Date,
        "estimates_request_start_date": pl.Date,
        "estimates_request_end_date": pl.Date,
        "retrieval_eligible": pl.Boolean,
        "retrieval_exclusion_reason": pl.Utf8,
    }


def _actuals_raw_schema() -> dict[str, pl.DataType]:
    return {
        "item_id": pl.Utf8,
        "response_row_index": pl.Int64,
        "request_group_id": pl.Utf8,
        "gvkey_int": pl.Int32,
        "effective_collection_ric": pl.Utf8,
        "announcement_date": pl.Date,
        "fiscal_period_end": pl.Date,
        "actual_eps": pl.Float64,
        "raw_fperiod": pl.Utf8,
        "row_parse_status": pl.Utf8,
    }


def _estimates_raw_schema() -> dict[str, pl.DataType]:
    return {
        "item_id": pl.Utf8,
        "response_row_index": pl.Int64,
        "request_group_id": pl.Utf8,
        "request_period": pl.Utf8,
        "gvkey_int": pl.Int32,
        "effective_collection_ric": pl.Utf8,
        "calc_date": pl.Date,
        "fiscal_period_end": pl.Date,
        "raw_fperiod": pl.Utf8,
        "forecast_consensus_mean": pl.Float64,
        "forecast_dispersion": pl.Float64,
        "estimate_count": pl.Int32,
        "row_parse_status": pl.Utf8,
    }


def _normalized_panel_schema() -> dict[str, pl.DataType]:
    return {
        "gvkey_int": pl.Int32,
        "announcement_date": pl.Date,
        "fiscal_period_end": pl.Date,
        "actual_fiscal_period_end_origin": pl.Utf8,
        "actual_eps": pl.Float64,
        "forecast_consensus_mean": pl.Float64,
        "forecast_dispersion": pl.Float64,
        "forecast_revision_4m": pl.Float64,
        "forecast_revision_4m_status": pl.Utf8,
        "forecast_revision_1m": pl.Float64,
        "selected_forecast_calc_date": pl.Date,
        "revision_base_calc_date_4m": pl.Date,
        "revision_base_calc_date_1m": pl.Date,
        "effective_collection_ric": pl.Utf8,
        "source_request_group_ids": pl.List(pl.Utf8),
    }


def _normalization_rejection_schema() -> dict[str, pl.DataType]:
    return {
        "rejection_case_id": pl.Utf8,
        "gvkey_int": pl.Int32,
        "announcement_date": pl.Date,
        "fiscal_period_end": pl.Date,
        "selected_calc_date": pl.Date,
        "rejection_status": pl.Utf8,
        "rejection_reason": pl.Utf8,
        "source_request_group_ids": pl.List(pl.Utf8),
    }


def _doc_analyst_anchor_schema() -> dict[str, pl.DataType]:
    return {
        "doc_id": pl.Utf8,
        "filing_date": pl.Date,
        "gvkey_int": pl.Int32,
        "KYPERMNO": pl.Int32,
        "quarter_report_date": pl.Date,
        "quarter_fiscal_period_end": pl.Date,
        "anchor_eligible": pl.Boolean,
        "anchor_exclusion_reason": pl.Utf8,
    }


def _doc_analyst_selected_schema() -> dict[str, pl.DataType]:
    return {
        "doc_id": pl.Utf8,
        "filing_date": pl.Date,
        "gvkey_int": pl.Int32,
        "KYPERMNO": pl.Int32,
        "quarter_report_date": pl.Date,
        "quarter_fiscal_period_end": pl.Date,
        "matched_announcement_date": pl.Date,
        "matched_fiscal_period_end": pl.Date,
        "actual_fiscal_period_end_origin": pl.Utf8,
        "match_type": pl.Utf8,
        "actual_eps": pl.Float64,
        "forecast_consensus_mean": pl.Float64,
        "forecast_dispersion": pl.Float64,
        "forecast_revision_4m": pl.Float64,
        "forecast_revision_4m_status": pl.Utf8,
        "forecast_revision_1m": pl.Float64,
        "analyst_match_status": pl.Utf8,
    }


def _ensure_dataframe(value: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return value.collect() if isinstance(value, pl.LazyFrame) else value


def _empty_df(columns: tuple[str, ...], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return _cast_df_to_schema(pl.DataFrame(schema=schema), schema).select(columns)


def _shift_months(value: dt.date, months: int) -> dt.date:
    month_index = (value.year * 12 + value.month - 1) + months
    year = month_index // 12
    month = month_index % 12 + 1
    return dt.date(year, month, monthrange(year, month)[1])


def _normalize_request_group_id(gvkey_int: int, effective_collection_ric: str) -> str:
    return stable_hash_id("analyst_request_group", gvkey_int, effective_collection_ric, prefix="group")


def _normalize_group_list(values: set[str]) -> list[str]:
    return sorted(value for value in values if value)


def build_refinitiv_step1_analyst_request_groups(
    instrument_authority_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    authority_schema = {
        "bridge_row_id": pl.Utf8,
        "KYPERMNO": pl.Int32,
        "gvkey_int": pl.Int32,
        "gvkey_source_column": pl.Utf8,
        "first_seen_caldt": pl.Date,
        "last_seen_caldt": pl.Date,
        "effective_collection_ric": pl.Utf8,
        "effective_collection_ric_source": pl.Utf8,
        "effective_resolution_status": pl.Utf8,
        "authority_eligible": pl.Boolean,
        "authority_exclusion_reason": pl.Utf8,
    }
    authority = _cast_df_to_schema(
        instrument_authority_df.select(INSTRUMENT_AUTHORITY_COLUMNS),
        authority_schema,
    ).select(INSTRUMENT_AUTHORITY_COLUMNS)

    eligible = authority.filter(pl.col("authority_eligible").fill_null(False)).with_columns(
        pl.struct("gvkey_int", "effective_collection_ric").map_elements(
            lambda row: None
            if row["gvkey_int"] is None or row["effective_collection_ric"] is None
            else _normalize_request_group_id(int(row["gvkey_int"]), str(row["effective_collection_ric"])),
            return_dtype=pl.Utf8,
        ).alias("request_group_id")
    )

    membership_df = _cast_df_to_schema(
        eligible.select(
            "request_group_id",
            pl.col("bridge_row_id").cast(pl.Utf8, strict=False),
            pl.col("gvkey_int").cast(pl.Int32, strict=False),
            pl.col("KYPERMNO").cast(pl.Int32, strict=False),
            pl.col("effective_collection_ric").cast(pl.Utf8, strict=False),
        ),
        _request_group_membership_schema(),
    ).select(ANALYST_REQUEST_GROUP_MEMBERSHIP_COLUMNS)

    request_group_df = (
        eligible.group_by("request_group_id", "gvkey_int", "effective_collection_ric")
        .agg(
            pl.col("bridge_row_id").n_unique().cast(pl.Int64).alias("member_bridge_row_count"),
            pl.col("first_seen_caldt").min().alias("bridge_start_date_min"),
            pl.col("last_seen_caldt").max().alias("bridge_end_date_max"),
        )
        .with_columns(
            (pl.col("bridge_start_date_min") - pl.duration(days=31)).alias("actuals_request_start_date"),
            (pl.col("bridge_end_date_max") + pl.duration(days=120)).alias("actuals_request_end_date"),
            (pl.col("bridge_start_date_min") - pl.duration(days=270)).alias("estimates_request_start_date"),
            (pl.col("bridge_end_date_max") + pl.duration(days=31)).alias("estimates_request_end_date"),
            pl.lit(True).alias("retrieval_eligible"),
            pl.lit(None, dtype=pl.Utf8).alias("retrieval_exclusion_reason"),
        )
        .sort("request_group_id")
    )
    request_group_df = _cast_df_to_schema(
        request_group_df.select(ANALYST_REQUEST_GROUP_COLUMNS),
        _request_group_schema(),
    ).select(ANALYST_REQUEST_GROUP_COLUMNS)
    return membership_df, request_group_df


def run_refinitiv_step1_analyst_request_groups_pipeline(
    *,
    instrument_authority_artifact_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instrument_authority_df = pl.read_parquet(instrument_authority_artifact_path)
    membership_df, request_group_df = build_refinitiv_step1_analyst_request_groups(instrument_authority_df)

    membership_path = output_dir / "refinitiv_analyst_request_group_membership_common_stock.parquet"
    request_group_path = output_dir / "refinitiv_analyst_request_universe_common_stock.parquet"
    membership_df.write_parquet(membership_path, compression="zstd")
    request_group_df.write_parquet(request_group_path, compression="zstd")
    return {
        "refinitiv_analyst_request_group_membership_common_stock_parquet": membership_path,
        "refinitiv_analyst_request_universe_common_stock_parquet": request_group_path,
    }


def build_refinitiv_lm2011_doc_analyst_anchors(
    sample_backbone_lf: pl.LazyFrame,
    quarterly_accounting_panel_lf: pl.LazyFrame,
) -> pl.DataFrame:
    sample_schema = sample_backbone_lf.collect_schema()
    permno_col = "KYPERMNO" if "KYPERMNO" in sample_schema else "kypermno" if "kypermno" in sample_schema else None
    if permno_col is None:
        raise ValueError("sample_backbone_lf missing both KYPERMNO and kypermno")

    docs = (
        attach_eligible_quarterly_accounting(
            sample_backbone_lf.with_columns(pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("_lm2011_gvkey_int")),
            quarterly_accounting_panel_lf.with_columns(pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int")),
            filing_gvkey_col="_lm2011_gvkey_int",
        )
        .with_columns(
            pl.col("_lm2011_gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int"),
            pl.col(permno_col).cast(pl.Int32, strict=False).alias("KYPERMNO"),
            pl.coalesce(
                [
                    pl.col("APDEDATEQ").cast(pl.Date, strict=False),
                    pl.col("PDATEQ").cast(pl.Date, strict=False),
                ]
            ).alias("quarter_fiscal_period_end"),
            pl.col("quarter_report_date").is_not_null().alias("anchor_eligible"),
            pl.when(pl.col("quarter_report_date").is_null())
            .then(pl.lit("missing_quarter_report_date"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("anchor_exclusion_reason"),
        )
        .select(DOC_ANALYST_ANCHOR_COLUMNS)
        .collect()
    )
    return _cast_df_to_schema(
        docs.unique(subset=["doc_id"], keep="first", maintain_order=True),
        _doc_analyst_anchor_schema(),
    ).select(DOC_ANALYST_ANCHOR_COLUMNS)


def run_refinitiv_lm2011_doc_analyst_anchor_pipeline(
    *,
    sample_backbone_lf: pl.LazyFrame,
    quarterly_accounting_panel_lf: pl.LazyFrame,
    output_dir: Path | str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    anchors_df = build_refinitiv_lm2011_doc_analyst_anchors(sample_backbone_lf, quarterly_accounting_panel_lf)
    output_path = output_dir / "refinitiv_doc_analyst_request_anchors.parquet"
    anchors_df.write_parquet(output_path, compression="zstd")
    return {"refinitiv_doc_analyst_request_anchors_parquet": output_path}


def _prepare_selected_analyst_panel(analyst_panel_df: pl.DataFrame) -> pl.DataFrame:
    schema = analyst_panel_df.schema
    required = (
        "gvkey_int",
        "announcement_date",
        "fiscal_period_end",
        "actual_eps",
        "forecast_consensus_mean",
        "forecast_dispersion",
        "forecast_revision_4m",
    )
    missing = [name for name in required if name not in schema]
    if missing:
        raise ValueError(f"analyst_panel_df missing required columns: {missing}")

    forecast_revision_1m_expr = (
        pl.col("forecast_revision_1m").cast(pl.Float64, strict=False)
        if "forecast_revision_1m" in schema
        else pl.lit(None, dtype=pl.Float64)
    )
    prepared = (
        analyst_panel_df.select(
            pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int"),
            pl.col("announcement_date").cast(pl.Date, strict=False).alias("announcement_date"),
            pl.col("fiscal_period_end").cast(pl.Date, strict=False).alias("fiscal_period_end"),
            (
                pl.col("actual_fiscal_period_end_origin").cast(pl.Utf8, strict=False)
                if "actual_fiscal_period_end_origin" in schema
                else pl.lit(None, dtype=pl.Utf8)
            ).alias("actual_fiscal_period_end_origin"),
            pl.col("actual_eps").cast(pl.Float64, strict=False).alias("actual_eps"),
            pl.col("forecast_consensus_mean").cast(pl.Float64, strict=False).alias("forecast_consensus_mean"),
            pl.col("forecast_dispersion").cast(pl.Float64, strict=False).alias("forecast_dispersion"),
            pl.col("forecast_revision_4m").cast(pl.Float64, strict=False).alias("forecast_revision_4m"),
            (
                pl.col("forecast_revision_4m_status").cast(pl.Utf8, strict=False)
                if "forecast_revision_4m_status" in schema
                else pl.lit(None, dtype=pl.Utf8)
            ).alias("forecast_revision_4m_status"),
            forecast_revision_1m_expr.alias("forecast_revision_1m"),
        )
        .unique(maintain_order=True)
    )
    duplicate_keys = (
        prepared.group_by("gvkey_int", "announcement_date", "fiscal_period_end")
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_keys.height:
        raise ValueError("analyst_panel_df must be unique on (gvkey_int, announcement_date, fiscal_period_end)")
    return prepared


def select_refinitiv_lm2011_doc_analyst_inputs(
    doc_anchors_df: pl.DataFrame | pl.LazyFrame,
    analyst_panel_df: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    docs = _ensure_dataframe(doc_anchors_df)
    analyst = _prepare_selected_analyst_panel(_ensure_dataframe(analyst_panel_df))
    doc_schema = docs.schema
    required_docs = ("doc_id", "filing_date", "gvkey_int", "KYPERMNO", "quarter_report_date")
    missing_docs = [name for name in required_docs if name not in doc_schema]
    if missing_docs:
        raise ValueError(f"doc_anchors_df missing required columns: {missing_docs}")

    quarter_fiscal_expr = (
        pl.col("quarter_fiscal_period_end").cast(pl.Date, strict=False)
        if "quarter_fiscal_period_end" in doc_schema
        else pl.coalesce(
            [
                pl.col("APDEDATEQ").cast(pl.Date, strict=False),
                pl.col("PDATEQ").cast(pl.Date, strict=False),
            ]
        )
    )
    prepared_docs = (
        docs.with_columns(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("filing_date").cast(pl.Date, strict=False),
            pl.col("gvkey_int").cast(pl.Int32, strict=False),
            pl.col("KYPERMNO").cast(pl.Int32, strict=False),
            pl.col("quarter_report_date").cast(pl.Date, strict=False),
            quarter_fiscal_expr.alias("quarter_fiscal_period_end"),
        )
        .unique(subset=["doc_id"], keep="first", maintain_order=True)
    )

    analyst_exact = analyst.with_columns(
        pl.col("announcement_date").alias("matched_announcement_date"),
        pl.col("fiscal_period_end").alias("matched_fiscal_period_end"),
    )
    exact_join = prepared_docs.join(
        analyst_exact,
        left_on=["gvkey_int", "quarter_report_date", "quarter_fiscal_period_end"],
        right_on=["gvkey_int", "announcement_date", "fiscal_period_end"],
        how="left",
    )
    exact_hits = exact_join.filter(pl.col("matched_announcement_date").is_not_null()).with_columns(
        pl.lit("EXACT").alias("match_type"),
        pl.lit("MATCHED").alias("analyst_match_status"),
    )
    exact_misses = exact_join.filter(pl.col("matched_announcement_date").is_null()).select(prepared_docs.columns)

    announcement_counts = (
        analyst.group_by("gvkey_int", "announcement_date")
        .len()
        .rename({"announcement_date": "quarter_report_date", "len": "announcement_match_count"})
    )
    fallback_candidates = analyst.with_columns(
        pl.col("announcement_date").alias("matched_announcement_date"),
        pl.col("fiscal_period_end").alias("matched_fiscal_period_end"),
    )
    unique_fallback_panel = (
        fallback_candidates.join(
            analyst.group_by("gvkey_int", "announcement_date").len().filter(pl.col("len") == 1),
            on=["gvkey_int", "announcement_date"],
            how="inner",
        )
        .drop("len")
    )
    fallback_hits = (
        exact_misses.join(
            unique_fallback_panel,
            left_on=["gvkey_int", "quarter_report_date"],
            right_on=["gvkey_int", "announcement_date"],
            how="left",
        )
        .filter(pl.col("matched_announcement_date").is_not_null())
        .with_columns(
            pl.lit("ANNOUNCEMENT_DATE_UNIQUE_FALLBACK").alias("match_type"),
            pl.lit("MATCHED").alias("analyst_match_status"),
        )
    )
    unresolved = exact_misses.join(
        announcement_counts,
        on=["gvkey_int", "quarter_report_date"],
        how="left",
    ).with_columns(
        pl.when(pl.col("announcement_match_count") > 1)
        .then(pl.lit("AMBIGUOUS_FALLBACK_REJECTED"))
        .otherwise(pl.lit("NO_MATCH"))
        .alias("analyst_match_status"),
        pl.lit("NONE").alias("match_type"),
        pl.lit(None, dtype=pl.Date).alias("matched_announcement_date"),
        pl.lit(None, dtype=pl.Date).alias("matched_fiscal_period_end"),
        pl.lit(None, dtype=pl.Utf8).alias("actual_fiscal_period_end_origin"),
        pl.lit(None, dtype=pl.Float64).alias("actual_eps"),
        pl.lit(None, dtype=pl.Float64).alias("forecast_consensus_mean"),
        pl.lit(None, dtype=pl.Float64).alias("forecast_dispersion"),
        pl.lit(None, dtype=pl.Float64).alias("forecast_revision_4m"),
        pl.lit(None, dtype=pl.Utf8).alias("forecast_revision_4m_status"),
        pl.lit(None, dtype=pl.Float64).alias("forecast_revision_1m"),
    )

    return pl.concat(
        [
            exact_hits.select(
                *prepared_docs.columns,
                "matched_announcement_date",
                "matched_fiscal_period_end",
                "actual_fiscal_period_end_origin",
                "match_type",
                "actual_eps",
                "forecast_consensus_mean",
                "forecast_dispersion",
                "forecast_revision_4m",
                "forecast_revision_4m_status",
                "forecast_revision_1m",
                "analyst_match_status",
            ),
            fallback_hits.select(
                *prepared_docs.columns,
                "matched_announcement_date",
                "matched_fiscal_period_end",
                "actual_fiscal_period_end_origin",
                "match_type",
                "actual_eps",
                "forecast_consensus_mean",
                "forecast_dispersion",
                "forecast_revision_4m",
                "forecast_revision_4m_status",
                "forecast_revision_1m",
                "analyst_match_status",
            ),
            unresolved.select(
                *prepared_docs.columns,
                "matched_announcement_date",
                "matched_fiscal_period_end",
                "actual_fiscal_period_end_origin",
                "match_type",
                "actual_eps",
                "forecast_consensus_mean",
                "forecast_dispersion",
                "forecast_revision_4m",
                "forecast_revision_4m_status",
                "forecast_revision_1m",
                "analyst_match_status",
            ),
        ],
        how="vertical_relaxed",
    ).sort("doc_id")


def run_refinitiv_lm2011_doc_analyst_select_pipeline(
    *,
    doc_anchors_artifact_path: Path | str,
    analyst_normalized_panel_artifact_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_anchors_df = pl.read_parquet(doc_anchors_artifact_path)
    analyst_panel_df = pl.read_parquet(analyst_normalized_panel_artifact_path)
    selected_df = select_refinitiv_lm2011_doc_analyst_inputs(doc_anchors_df, analyst_panel_df)
    selected_df = _cast_df_to_schema(
        selected_df.select(DOC_ANALYST_SELECTED_COLUMNS),
        _doc_analyst_selected_schema(),
    ).select(DOC_ANALYST_SELECTED_COLUMNS)
    output_path = output_dir / "refinitiv_doc_analyst_selected.parquet"
    selected_df.write_parquet(output_path, compression="zstd")
    return {"refinitiv_doc_analyst_selected_parquet": output_path}


def _canonicalize_actual_event(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    actual_eps_values = {row["actual_eps"] for row in rows}
    request_groups = {str(row["request_group_id"]) for row in rows if row.get("request_group_id")}
    fiscal_period_end = rows[0].get("fiscal_period_end")
    announcement_date = rows[0].get("announcement_date")
    gvkey_int = rows[0].get("gvkey_int")
    if len(actual_eps_values) > 1:
        return None, {
            "rejection_case_id": stable_hash_id(
                "analyst_rejection",
                gvkey_int,
                announcement_date,
                fiscal_period_end,
                "conflicting_duplicate_actuals",
                prefix="reject",
            ),
            "gvkey_int": gvkey_int,
            "announcement_date": announcement_date,
            "fiscal_period_end": fiscal_period_end,
            "selected_calc_date": None,
            "rejection_status": "CONFLICTING_DUPLICATE_ACTUALS",
            "rejection_reason": "conflicting duplicate actual EPS values for the same event key",
            "source_request_group_ids": _normalize_group_list(request_groups),
        }
    actual_eps = next(iter(actual_eps_values), None)
    effective_collection_rics = {str(row["effective_collection_ric"]) for row in rows if row.get("effective_collection_ric")}
    return {
        "gvkey_int": gvkey_int,
        "announcement_date": announcement_date,
        "fiscal_period_end": fiscal_period_end,
        "actual_eps": actual_eps,
        "effective_collection_ric": next(iter(effective_collection_rics)) if len(effective_collection_rics) == 1 else None,
        "source_request_group_ids": _normalize_group_list(request_groups),
    }, None


def _canonicalize_estimate_snapshot(
    gvkey_int: int,
    fiscal_period_end: dt.date,
    calc_date: dt.date,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bool]:
    value_keys = {
        (
            row.get("forecast_consensus_mean"),
            row.get("forecast_dispersion"),
            row.get("estimate_count"),
        )
        for row in rows
    }
    if len(value_keys) > 1:
        return None, True
    forecast_consensus_mean, forecast_dispersion, estimate_count = next(iter(value_keys))
    request_groups = {str(row["request_group_id"]) for row in rows if row.get("request_group_id")}
    effective_collection_rics = {str(row["effective_collection_ric"]) for row in rows if row.get("effective_collection_ric")}
    return {
        "gvkey_int": gvkey_int,
        "fiscal_period_end": fiscal_period_end,
        "calc_date": calc_date,
        "forecast_consensus_mean": forecast_consensus_mean,
        "forecast_dispersion": forecast_dispersion,
        "estimate_count": estimate_count,
        "effective_collection_ric": next(iter(effective_collection_rics)) if len(effective_collection_rics) == 1 else None,
        "source_request_group_ids": _normalize_group_list(request_groups),
    }, False


def build_refinitiv_analyst_normalized_outputs(
    actuals_raw_df: pl.DataFrame,
    estimates_raw_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    actuals = _cast_df_to_schema(
        actuals_raw_df.select(ANALYST_ACTUALS_RAW_COLUMNS),
        _actuals_raw_schema(),
    ).select(ANALYST_ACTUALS_RAW_COLUMNS)
    estimates = _cast_df_to_schema(
        estimates_raw_df.select(ANALYST_ESTIMATES_MONTHLY_RAW_COLUMNS),
        _estimates_raw_schema(),
    ).select(ANALYST_ESTIMATES_MONTHLY_RAW_COLUMNS)

    actual_groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in actuals.to_dicts():
        if row.get("announcement_date") is None or row.get("actual_eps") is None:
            continue
        key = (row.get("gvkey_int"), row.get("announcement_date"), row.get("fiscal_period_end"))
        actual_groups.setdefault(key, []).append(row)

    canonical_actuals: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    for rows in actual_groups.values():
        canonical, rejection = _canonicalize_actual_event(rows)
        if rejection is not None:
            rejection_rows.append(rejection)
            continue
        if canonical is not None:
            canonical_actuals.append(canonical)

    estimate_group_map: dict[tuple[int, dt.date, dt.date], list[dict[str, Any]]] = {}
    estimate_rows_by_gvkey_calc_date: dict[tuple[int, dt.date], list[dict[str, Any]]] = {}
    for row in estimates.to_dicts():
        gvkey_int = row.get("gvkey_int")
        calc_date = row.get("calc_date")
        fiscal_period_end = row.get("fiscal_period_end")
        if gvkey_int is None or calc_date is None or fiscal_period_end is None:
            continue
        estimate_group_map.setdefault((gvkey_int, fiscal_period_end, calc_date), []).append(row)
        estimate_rows_by_gvkey_calc_date.setdefault((gvkey_int, calc_date), []).append(row)

    canonical_estimates: dict[tuple[int, dt.date, dt.date], dict[str, Any]] = {}
    conflicting_estimate_keys: set[tuple[int, dt.date, dt.date]] = set()
    for key, rows in estimate_group_map.items():
        canonical, conflict = _canonicalize_estimate_snapshot(key[0], key[1], key[2], rows)
        if conflict:
            conflicting_estimate_keys.add(key)
            continue
        if canonical is not None:
            canonical_estimates[key] = canonical

    provisional_rows: list[dict[str, Any]] = []
    for event in canonical_actuals:
        gvkey_int = int(event["gvkey_int"])
        announcement_date = event["announcement_date"]
        fiscal_period_end = event["fiscal_period_end"]
        request_groups = set(event["source_request_group_ids"])
        actual_fiscal_period_end_origin = "API_DIRECT"

        if fiscal_period_end is None:
            eligible_calc_dates = sorted(
                {
                    calc_date
                    for (estimate_gvkey, calc_date), rows in estimate_rows_by_gvkey_calc_date.items()
                    if estimate_gvkey == gvkey_int and calc_date <= announcement_date
                }
            )
            if not eligible_calc_dates:
                rejection_rows.append(
                    {
                        "rejection_case_id": stable_hash_id(
                            "analyst_rejection",
                            gvkey_int,
                            announcement_date,
                            "missing_derived_fiscal_period_end",
                            prefix="reject",
                        ),
                        "gvkey_int": gvkey_int,
                        "announcement_date": announcement_date,
                        "fiscal_period_end": None,
                        "selected_calc_date": None,
                        "rejection_status": "MISSING_DERIVED_FISCAL_PERIOD_END",
                        "rejection_reason": "actual event is missing fiscal period end and no estimate snapshots were available",
                        "source_request_group_ids": sorted(request_groups),
                    }
                )
                continue
            selected_calc_date = eligible_calc_dates[-1]
            derived_rows = estimate_rows_by_gvkey_calc_date.get((gvkey_int, selected_calc_date), [])
            derived_fiscal_period_ends = sorted({row["fiscal_period_end"] for row in derived_rows if row.get("fiscal_period_end")})
            if not derived_fiscal_period_ends:
                rejection_rows.append(
                    {
                        "rejection_case_id": stable_hash_id(
                            "analyst_rejection",
                            gvkey_int,
                            announcement_date,
                            selected_calc_date,
                            "missing_derived_fiscal_period_end",
                            prefix="reject",
                        ),
                        "gvkey_int": gvkey_int,
                        "announcement_date": announcement_date,
                        "fiscal_period_end": None,
                        "selected_calc_date": selected_calc_date,
                        "rejection_status": "MISSING_DERIVED_FISCAL_PERIOD_END",
                        "rejection_reason": "latest estimate-side calc date did not expose a fiscal period end",
                        "source_request_group_ids": sorted(request_groups),
                    }
                )
                continue
            if len(derived_fiscal_period_ends) > 1:
                rejection_rows.append(
                    {
                        "rejection_case_id": stable_hash_id(
                            "analyst_rejection",
                            gvkey_int,
                            announcement_date,
                            selected_calc_date,
                            "nonunique_derived_fiscal_period_end",
                            prefix="reject",
                        ),
                        "gvkey_int": gvkey_int,
                        "announcement_date": announcement_date,
                        "fiscal_period_end": None,
                        "selected_calc_date": selected_calc_date,
                        "rejection_status": "NONUNIQUE_DERIVED_FISCAL_PERIOD_END",
                        "rejection_reason": "latest estimate-side calc date implied multiple fiscal period ends",
                        "source_request_group_ids": sorted(request_groups),
                    }
                )
                continue
            fiscal_period_end = derived_fiscal_period_ends[0]
            actual_fiscal_period_end_origin = "ESTIMATE_FALLBACK"
        else:
            selected_calc_dates = sorted(
                {
                    calc_date
                    for (estimate_gvkey, estimate_fpe, calc_date) in estimate_group_map
                    if estimate_gvkey == gvkey_int and estimate_fpe == fiscal_period_end and calc_date <= announcement_date
                }
            )
            if not selected_calc_dates:
                rejection_rows.append(
                    {
                        "rejection_case_id": stable_hash_id(
                            "analyst_rejection",
                            gvkey_int,
                            announcement_date,
                            fiscal_period_end,
                            "missing_selected_estimate_snapshot",
                            prefix="reject",
                        ),
                        "gvkey_int": gvkey_int,
                        "announcement_date": announcement_date,
                        "fiscal_period_end": fiscal_period_end,
                        "selected_calc_date": None,
                        "rejection_status": "MISSING_SELECTED_ESTIMATE_SNAPSHOT",
                        "rejection_reason": "no estimate snapshot existed on or before the announcement date",
                        "source_request_group_ids": sorted(request_groups),
                    }
                )
                continue
            selected_calc_date = selected_calc_dates[-1]

        selected_snapshot_key = (gvkey_int, fiscal_period_end, selected_calc_date)
        if selected_snapshot_key in conflicting_estimate_keys:
            rejection_rows.append(
                {
                    "rejection_case_id": stable_hash_id(
                        "analyst_rejection",
                        gvkey_int,
                        announcement_date,
                        fiscal_period_end,
                        selected_calc_date,
                        "conflicting_selected_estimate_snapshot",
                        prefix="reject",
                    ),
                    "gvkey_int": gvkey_int,
                    "announcement_date": announcement_date,
                    "fiscal_period_end": fiscal_period_end,
                    "selected_calc_date": selected_calc_date,
                    "rejection_status": "CONFLICTING_SELECTED_ESTIMATE_SNAPSHOT",
                    "rejection_reason": "selected estimate snapshot had conflicting duplicate values",
                    "source_request_group_ids": sorted(request_groups),
                }
            )
            continue
        selected_snapshot = canonical_estimates.get(selected_snapshot_key)
        if (
            selected_snapshot is None
            or selected_snapshot.get("forecast_consensus_mean") is None
            or selected_snapshot.get("forecast_dispersion") is None
        ):
            rejection_rows.append(
                {
                    "rejection_case_id": stable_hash_id(
                        "analyst_rejection",
                        gvkey_int,
                        announcement_date,
                        fiscal_period_end,
                        selected_calc_date,
                        "missing_selected_estimate_snapshot",
                        prefix="reject",
                    ),
                    "gvkey_int": gvkey_int,
                    "announcement_date": announcement_date,
                    "fiscal_period_end": fiscal_period_end,
                    "selected_calc_date": selected_calc_date,
                    "rejection_status": "MISSING_SELECTED_ESTIMATE_SNAPSHOT",
                    "rejection_reason": "selected estimate snapshot was missing or did not expose a consensus mean and dispersion",
                    "source_request_group_ids": sorted(request_groups),
                }
            )
            continue

        revision_4m_cutoff = _shift_months(selected_calc_date, -4)
        revision_4m_dates = sorted(
            {
                calc_date
                for (estimate_gvkey, estimate_fpe, calc_date) in estimate_group_map
                if estimate_gvkey == gvkey_int and estimate_fpe == fiscal_period_end and calc_date <= revision_4m_cutoff
            }
        )
        revision_base_calc_date_4m = revision_4m_dates[-1] if revision_4m_dates else None
        forecast_revision_4m = None
        forecast_revision_4m_status = "OK"
        if revision_base_calc_date_4m is None:
            forecast_revision_4m_status = "MISSING_BASE_SNAPSHOT"
        else:
            revision_base_key_4m = (gvkey_int, fiscal_period_end, revision_base_calc_date_4m)
            if revision_base_key_4m in conflicting_estimate_keys:
                rejection_rows.append(
                    {
                        "rejection_case_id": stable_hash_id(
                            "analyst_rejection",
                            gvkey_int,
                            announcement_date,
                            fiscal_period_end,
                            revision_base_calc_date_4m,
                            "conflicting_revision_base_4m",
                            prefix="reject",
                        ),
                        "gvkey_int": gvkey_int,
                        "announcement_date": announcement_date,
                        "fiscal_period_end": fiscal_period_end,
                        "selected_calc_date": selected_calc_date,
                        "rejection_status": "CONFLICTING_REVISION_BASE_4M",
                        "rejection_reason": "four-month revision base snapshot had conflicting duplicate values",
                        "source_request_group_ids": sorted(request_groups),
                    }
                )
                continue
            revision_base_4m = canonical_estimates.get(revision_base_key_4m)
            if revision_base_4m is None or revision_base_4m.get("forecast_consensus_mean") is None:
                forecast_revision_4m_status = "MISSING_BASE_CONSENSUS_MEAN"
            else:
                forecast_revision_4m = float(selected_snapshot["forecast_consensus_mean"]) - float(
                    revision_base_4m["forecast_consensus_mean"]
                )

        revision_1m_cutoff = _shift_months(selected_calc_date, -1)
        revision_1m_dates = sorted(
            {
                calc_date
                for (estimate_gvkey, estimate_fpe, calc_date) in estimate_group_map
                if estimate_gvkey == gvkey_int and estimate_fpe == fiscal_period_end and calc_date <= revision_1m_cutoff
            }
        )
        revision_base_calc_date_1m = revision_1m_dates[-1] if revision_1m_dates else None
        revision_1m = None
        if revision_base_calc_date_1m is not None:
            revision_base_key_1m = (gvkey_int, fiscal_period_end, revision_base_calc_date_1m)
            if revision_base_key_1m not in conflicting_estimate_keys:
                revision_base_1m = canonical_estimates.get(revision_base_key_1m)
                if revision_base_1m is not None and revision_base_1m.get("forecast_consensus_mean") is not None:
                    revision_1m = (
                        float(selected_snapshot["forecast_consensus_mean"])
                        - float(revision_base_1m["forecast_consensus_mean"])
                    )

        provisional_rows.append(
            {
                "gvkey_int": gvkey_int,
                "announcement_date": announcement_date,
                "fiscal_period_end": fiscal_period_end,
                "actual_fiscal_period_end_origin": actual_fiscal_period_end_origin,
                "actual_eps": float(event["actual_eps"]),
                "forecast_consensus_mean": float(selected_snapshot["forecast_consensus_mean"]),
                "forecast_dispersion": _normalize_float_value(selected_snapshot.get("forecast_dispersion")),
                "forecast_revision_4m": forecast_revision_4m,
                "forecast_revision_4m_status": forecast_revision_4m_status,
                "forecast_revision_1m": revision_1m,
                "selected_forecast_calc_date": selected_calc_date,
                "revision_base_calc_date_4m": revision_base_calc_date_4m,
                "revision_base_calc_date_1m": revision_base_calc_date_1m,
                "effective_collection_ric": event.get("effective_collection_ric") or selected_snapshot.get("effective_collection_ric"),
                "source_request_group_ids": sorted(
                    set(event["source_request_group_ids"]) | set(selected_snapshot["source_request_group_ids"])
                ),
            }
        )

    normalized_groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in provisional_rows:
        normalized_groups.setdefault(
            (row["gvkey_int"], row["announcement_date"], row["fiscal_period_end"]),
            [],
        ).append(row)

    normalized_rows: list[dict[str, Any]] = []
    for key, rows in normalized_groups.items():
        value_keys = {
            (
                row["actual_fiscal_period_end_origin"],
                row["actual_eps"],
                row["forecast_consensus_mean"],
                row["forecast_dispersion"],
                row["forecast_revision_4m"],
                row["forecast_revision_4m_status"],
                row["forecast_revision_1m"],
                row["selected_forecast_calc_date"],
                row["revision_base_calc_date_4m"],
                row["revision_base_calc_date_1m"],
            )
            for row in rows
        }
        if len(value_keys) > 1:
            source_request_group_ids = sorted(
                {
                    request_group_id
                    for row in rows
                    for request_group_id in row.get("source_request_group_ids", [])
                }
            )
            rejection_rows.append(
                {
                    "rejection_case_id": stable_hash_id(
                        "analyst_rejection",
                        key[0],
                        key[1],
                        key[2],
                        "conflicting_normalized_event",
                        prefix="reject",
                    ),
                    "gvkey_int": key[0],
                    "announcement_date": key[1],
                    "fiscal_period_end": key[2],
                    "selected_calc_date": None,
                    "rejection_status": "CONFLICTING_NORMALIZED_EVENT",
                    "rejection_reason": "multiple request groups produced conflicting normalized values for the same event key",
                    "source_request_group_ids": source_request_group_ids,
                }
            )
            continue
        merged = dict(rows[0])
        effective_collection_rics = {
            str(row["effective_collection_ric"])
            for row in rows
            if row.get("effective_collection_ric")
        }
        merged["effective_collection_ric"] = (
            next(iter(effective_collection_rics)) if len(effective_collection_rics) == 1 else None
        )
        merged["source_request_group_ids"] = sorted(
            {
                request_group_id
                for row in rows
                for request_group_id in row.get("source_request_group_ids", [])
            }
        )
        normalized_rows.append(merged)

    normalized_df = _cast_df_to_schema(
        _build_explicit_schema_df(normalized_rows, _normalized_panel_schema()),
        _normalized_panel_schema(),
    ).select(ANALYST_NORMALIZED_PANEL_COLUMNS)
    rejection_df = _cast_df_to_schema(
        _build_explicit_schema_df(rejection_rows, _normalization_rejection_schema()),
        _normalization_rejection_schema(),
    ).select(ANALYST_NORMALIZATION_REJECTION_COLUMNS)
    return normalized_df, rejection_df


def run_refinitiv_step1_analyst_normalize_pipeline(
    *,
    actuals_raw_artifact_path: Path | str,
    estimates_raw_artifact_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actuals_raw_df = (
        pl.read_parquet(actuals_raw_artifact_path)
        if Path(actuals_raw_artifact_path).exists()
        else _empty_df(ANALYST_ACTUALS_RAW_COLUMNS, _actuals_raw_schema())
    )
    estimates_raw_df = (
        pl.read_parquet(estimates_raw_artifact_path)
        if Path(estimates_raw_artifact_path).exists()
        else _empty_df(ANALYST_ESTIMATES_MONTHLY_RAW_COLUMNS, _estimates_raw_schema())
    )

    normalized_df, rejection_df = build_refinitiv_analyst_normalized_outputs(actuals_raw_df, estimates_raw_df)
    normalized_path = output_dir / "refinitiv_analyst_normalized_panel.parquet"
    rejection_path = output_dir / "refinitiv_analyst_normalization_rejections.parquet"
    normalized_df.write_parquet(normalized_path, compression="zstd")
    rejection_df.write_parquet(rejection_path, compression="zstd")
    return {
        "refinitiv_analyst_normalized_panel_parquet": normalized_path,
        "refinitiv_analyst_normalization_rejections_parquet": rejection_path,
    }


__all__ = [
    "ANALYST_ACTUALS_RAW_COLUMNS",
    "ANALYST_ESTIMATES_MONTHLY_RAW_COLUMNS",
    "ANALYST_NORMALIZATION_REJECTION_COLUMNS",
    "ANALYST_NORMALIZED_PANEL_COLUMNS",
    "ANALYST_REQUEST_GROUP_COLUMNS",
    "ANALYST_REQUEST_GROUP_MEMBERSHIP_COLUMNS",
    "DOC_ANALYST_ANCHOR_COLUMNS",
    "DOC_ANALYST_SELECTED_COLUMNS",
    "build_refinitiv_analyst_normalized_outputs",
    "build_refinitiv_lm2011_doc_analyst_anchors",
    "build_refinitiv_step1_analyst_request_groups",
    "run_refinitiv_lm2011_doc_analyst_anchor_pipeline",
    "run_refinitiv_lm2011_doc_analyst_select_pipeline",
    "run_refinitiv_step1_analyst_normalize_pipeline",
    "run_refinitiv_step1_analyst_request_groups_pipeline",
    "select_refinitiv_lm2011_doc_analyst_inputs",
    "_actuals_raw_schema",
    "_doc_analyst_anchor_schema",
    "_doc_analyst_selected_schema",
    "_estimates_raw_schema",
    "_normalization_rejection_schema",
    "_normalized_panel_schema",
]
