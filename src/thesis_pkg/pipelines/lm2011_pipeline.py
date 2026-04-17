from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import datetime as dt
import gc
import math
from pathlib import Path
import shutil
import warnings

import polars as pl
try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover - exercised only when the dependency is missing
    sm = None

from thesis_pkg.core.ccm.sec_ccm_contracts import share_turnover_ratio
from thesis_pkg.core.ccm.lm2011 import (
    _build_lm2011_sample_backbone_stage_frames,
    attach_eligible_quarterly_accounting,
    attach_latest_annual_accounting,
    attach_pre_filing_market_data,
    build_lm2011_normalized_filing_feeds,
    build_lm2011_sample_backbone,
    derive_filing_trade_anchors,
)
from thesis_pkg.core.sec.lm2011_text import (
    build_lm2011_text_features_full_10k,
    build_lm2011_text_features_mda,
    build_lm2011_trading_strategy_signal_frame,
)
from thesis_pkg.core.sec.lm2011_cleaning import Full10KCleaningContract
from thesis_pkg.pipelines.refinitiv.analyst import (
    select_refinitiv_lm2011_doc_analyst_inputs,
)


_STRATEGY_START_DATE = dt.date(1997, 7, 1)
_STRATEGY_END_DATE = dt.date(2007, 6, 30)
_STRATEGY_MIN_SORT_YEAR = 1997
_STRATEGY_MAX_SORT_YEAR = 2006
_STRATEGY_SIGNAL_COLUMNS: tuple[str, ...] = (
    "fin_neg_prop",
    "fin_neg_tfidf",
    "h4n_inf_prop",
    "h4n_inf_tfidf",
)
_LM2011_TABLE_I_FULL_10K_SECTION_ID = "full_10k_document"
_LM2011_TABLE_I_FULL_10K_SECTION_LABEL = "Full 10-K Document"
_LM2011_TABLE_I_FIRM_YEAR_SECTION_ID = "firm_year_sample"
_LM2011_TABLE_I_FIRM_YEAR_SECTION_LABEL = "Firm-Year Sample"
_LM2011_TABLE_I_MDA_SECTION_ID = "mda_subsection"
_LM2011_TABLE_I_MDA_SECTION_LABEL = "Management Discussion and Analysis (MD&A) Subsection"
_LM2011_TABLE_I_DEFAULT_SAMPLE_START = dt.date(1994, 1, 1)
_LM2011_TABLE_I_DEFAULT_SAMPLE_END = dt.date(2008, 12, 31)
DEFAULT_EVENT_WINDOW_DOC_BATCH_SIZE = 250
_STREAMING_PARQUET_COMPRESSION = "zstd"
_LM2011_TABLE_I_FULL_10K_ROW_LABELS: dict[str, str] = {
    "first_filing_per_year": "Include only first filing in a given year",
    "minimum_180_day_spacing": "At least 180 days between a given firm's 10-K filings",
    "crsp_permno_match": "CRSP PERMNO match",
    "ordinary_common_equity": "Reported on CRSP as an ordinary common equity firm",
    "market_cap_available": "CRSP market capitalization data available",
    "price_day_minus_one_ge_3": "Price on filing date day minus one >= $3",
    "event_window_returns_and_volume": "Returns and volume for day 0-3 event period",
    "major_exchange_listing": "NYSE, AMEX, or Nasdaq exchange listing",
    "sixty_day_pre_post_coverage": "At least 60 days of returns and volume in year prior to and following file date",
    "book_to_market_available_and_book_value_positive": "Book-to-market COMPUSTAT data available and book value > 0",
    "token_count_ge_2000": "Number of words in 10-K >= 2,000",
}
_LM2011_TABLE_I_FIRM_YEAR_ROW_LABELS: dict[str, str] = {
    "firm_year_sample": "Firm-Year Sample",
    "unique_firms": "Number of unique firms",
    "average_years_per_firm": "Average number of years per firm",
}
_LM2011_TABLE_I_MDA_ROW_LABELS: dict[str, str] = {
    "identifiable_mda": "Subset of 10-K sample where MD&A section could be identified",
    "mda_token_count_ge_250": "MD&A section >= 250 words",
}
_EVENT_SCREEN_SURFACE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "gvkey_int",
    "KYPERMNO",
    "filing_date",
    "filing_trade_date",
    "pre_filing_trade_date",
    "book_equity_be",
    "total_token_count_full_10k",
    "pre_filing_price",
    "size_event",
    "bm_event",
    "event_return_day_count",
    "event_volume_day_count",
    "pre_turnover_obs",
    "abnormal_volume_pre_obs",
    "event_shares",
    "event_shrcd",
    "event_exchcd",
    "pre_alpha_obs",
    "post_alpha_obs",
    "filing_period_excess_return",
    "share_turnover",
    "abnormal_volume",
    "pre_ffalpha",
    "postevent_return_volatility",
    "nasdaq_dummy",
)


def _empty_event_screen_surface_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "doc_id": pl.Utf8,
            "gvkey_int": pl.Int32,
            "KYPERMNO": pl.Int32,
            "filing_date": pl.Date,
            "filing_trade_date": pl.Date,
            "pre_filing_trade_date": pl.Date,
            "book_equity_be": pl.Float64,
            "total_token_count_full_10k": pl.Int32,
            "pre_filing_price": pl.Float64,
            "size_event": pl.Float64,
            "bm_event": pl.Float64,
            "event_return_day_count": pl.Int64,
            "event_volume_day_count": pl.Int64,
            "pre_turnover_obs": pl.Int64,
            "abnormal_volume_pre_obs": pl.Int64,
            "event_shares": pl.Float64,
            "event_shrcd": pl.Int32,
            "event_exchcd": pl.Int32,
            "pre_alpha_obs": pl.Int32,
            "post_alpha_obs": pl.Int32,
            "filing_period_excess_return": pl.Float64,
            "share_turnover": pl.Float64,
            "abnormal_volume": pl.Float64,
            "pre_ffalpha": pl.Float64,
            "postevent_return_volatility": pl.Float64,
            "nasdaq_dummy": pl.Int8,
        }
    )


def _empty_prepared_daily_event_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "KYPERMNO": pl.Int32,
            "trade_date": pl.Date,
            "stock_return": pl.Float64,
            "VOL": pl.Float64,
            "FINAL_PRC": pl.Float64,
            "PRC": pl.Float64,
            "TCAP": pl.Float64,
            "SHROUT": pl.Float64,
            "SHRCD": pl.Int32,
            "EXCHCD": pl.Int32,
            "mkt_rf": pl.Float64,
            "smb": pl.Float64,
            "hml": pl.Float64,
            "rf": pl.Float64,
            "market_return": pl.Float64,
        }
    )


def _require_columns(lf: pl.LazyFrame, required: tuple[str, ...], label: str) -> None:
    schema = lf.collect_schema()
    missing = [name for name in required if name not in schema]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _resolve_first_existing(schema: pl.Schema, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in schema:
            return candidate
    raise ValueError(f"{label} missing any of expected columns: {list(candidates)}")


def _first_day_of_month(value: dt.date) -> dt.date:
    return dt.date(value.year, value.month, 1)


def _previous_month_end(value: dt.date) -> dt.date:
    return _first_day_of_month(value) - dt.timedelta(days=1)


def _coerce_python_date(value: object, *, label: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if hasattr(value, "to_pydatetime"):
        converted = value.to_pydatetime()
        if isinstance(converted, dt.datetime):
            return converted.date()
        if isinstance(converted, dt.date):
            return converted
    if hasattr(value, "isoformat"):
        try:
            return dt.date.fromisoformat(str(value.isoformat()))
        except ValueError:
            pass
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} could not be coerced to a Python date: {value!r}") from exc


def _empty_event_panel_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "doc_id": pl.Utf8,
            "gvkey_int": pl.Int32,
            "KYPERMNO": pl.Int32,
            "filing_date": pl.Date,
            "filing_trade_date": pl.Date,
            "pre_filing_trade_date": pl.Date,
            "size_event": pl.Float64,
            "bm_event": pl.Float64,
            "share_turnover": pl.Float64,
            "pre_ffalpha": pl.Float64,
            "institutional_ownership": pl.Float64,
            "nasdaq_dummy": pl.Int8,
            "filing_period_excess_return": pl.Float64,
            "abnormal_volume": pl.Float64,
            "postevent_return_volatility": pl.Float64,
        }
    )


def _empty_sue_panel_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "doc_id": pl.Utf8,
            "gvkey_int": pl.Int32,
            "KYPERMNO": pl.Int32,
            "filing_date": pl.Date,
            "quarter_report_date": pl.Date,
            "size_event": pl.Float64,
            "bm_event": pl.Float64,
            "share_turnover": pl.Float64,
            "sue": pl.Float64,
            "analyst_dispersion": pl.Float64,
            "analyst_revisions": pl.Float64,
            "pre_ffalpha": pl.Float64,
            "institutional_ownership": pl.Float64,
            "nasdaq_dummy": pl.Int8,
        }
    )


def _empty_trading_strategy_monthly_returns_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "portfolio_month": pl.Date,
            "sort_signal_name": pl.Utf8,
            "long_short_return": pl.Float64,
        }
    )


def _empty_trading_strategy_ff4_summary_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "sort_signal_name": pl.Utf8,
            "alpha_ff3_mom": pl.Float64,
            "beta_market": pl.Float64,
            "beta_smb": pl.Float64,
            "beta_hml": pl.Float64,
            "beta_mom": pl.Float64,
            "r2": pl.Float64,
        }
    )


def _ensure_factor_scale(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    if df.height == 0:
        return df
    factor_cols = [column for column in columns if column in df.columns]
    if not factor_cols:
        return df
    max_abs = (
        df.select(
            pl.max_horizontal([pl.col(column).cast(pl.Float64, strict=False).abs() for column in factor_cols])
            .max()
            .alias("max_abs")
        )
        .item()
    )
    scale_down = max_abs is not None and float(max_abs) > 1.0
    return df.with_columns(
        [
            (
                pl.col(column).cast(pl.Float64, strict=False) / 100.0
                if scale_down
                else pl.col(column).cast(pl.Float64, strict=False)
            ).alias(column)
            for column in factor_cols
        ]
    )


def _frame_height(frame: pl.LazyFrame | pl.DataFrame) -> int:
    if isinstance(frame, pl.LazyFrame):
        return int(frame.select(pl.len().alias("_n")).collect().item())
    return int(frame.height)


def _validated_event_window_doc_batch_size(batch_size: int) -> int:
    resolved = int(batch_size)
    if resolved < 1:
        raise ValueError("event_window_doc_batch_size must be >= 1")
    return resolved


def _format_table_i_window_label(sample_start: dt.date, sample_end: dt.date) -> str:
    return f"{sample_start.year}-{sample_end.year}"


def _table_i_full_10k_row_labels(sample_start: dt.date, sample_end: dt.date) -> dict[str, str]:
    return {
        "edgar_complete_nonduplicate_sample": (
            f"EDGAR 10-K/10-K405 {_format_table_i_window_label(sample_start, sample_end)} complete sample "
            "(excluding duplicates)"
        ),
        **_LM2011_TABLE_I_FULL_10K_ROW_LABELS,
    }


def _prepare_unique_doc_metric_frame(
    metric_lf: pl.LazyFrame | None,
    *,
    metric_col: str,
    label: str,
    required_message: str,
    metric_dtype: pl.DataType,
) -> pl.DataFrame:
    if metric_lf is None:
        raise ValueError(required_message)
    _require_columns(metric_lf, ("doc_id", metric_col), label)
    metric_df = (
        metric_lf.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col(metric_col).cast(metric_dtype, strict=False).alias(metric_col),
        )
        .unique(subset=["doc_id", metric_col], maintain_order=True)
        .collect()
    )
    duplicate_doc_ids = (
        metric_df.group_by("doc_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("doc_id")
        .to_list()
    )
    if duplicate_doc_ids:
        raise ValueError(f"{label} must be unique on doc_id after exact duplicate removal: {duplicate_doc_ids[:10]}")
    return metric_df


def _prepare_ownership_frame(ownership_lf: pl.LazyFrame | None) -> pl.DataFrame:
    if ownership_lf is None:
        raise ValueError(
            "ownership_lf is required; fail-closed external_input_policy forbids building LM2011 panels without ownership"
        )
    schema = ownership_lf.collect_schema()
    value_col = (
        "institutional_ownership_pct"
        if "institutional_ownership_pct" in schema
        else "institutional_ownership"
        if "institutional_ownership" in schema
        else None
    )
    if value_col is None:
        raise ValueError("ownership_lf missing institutional_ownership_pct or institutional_ownership")
    return _prepare_unique_doc_metric_frame(
        ownership_lf.rename({value_col: "institutional_ownership"}),
        metric_col="institutional_ownership",
        label="ownership_lf",
        required_message=(
            "ownership_lf is required; fail-closed external_input_policy forbids building LM2011 panels without ownership"
        ),
        metric_dtype=pl.Float64,
    )


def _price_expr_from_available_columns(
    schema: pl.Schema,
    *,
    candidates: tuple[str, ...],
    label: str,
) -> pl.Expr:
    exprs = [
        pl.col(column).cast(pl.Float64, strict=False).abs()
        for column in candidates
        if column in schema
    ]
    if not exprs:
        raise ValueError(f"{label} missing any of expected price columns: {list(candidates)}")
    return pl.coalesce(exprs)


def _prepare_table_i_base_lf(
    sample_backbone_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    annual_accounting_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame | None,
) -> pl.LazyFrame:
    if full_10k_text_features_lf is None:
        raise ValueError(
            "full_10k_text_features_lf is required; fail-closed external_input_policy forbids LM2011 clean panels without token-count text features"
        )
    sample_schema = sample_backbone_lf.collect_schema()
    permno_col = _resolve_first_existing(sample_schema, ("KYPERMNO", "kypermno"), "sample_backbone")
    trading_calendar = (
        daily_lf.select(
            pl.col(_resolve_first_existing(daily_lf.collect_schema(), ("CALDT", "daily_caldt"), "daily_panel"))
            .cast(pl.Date, strict=False)
            .alias("CALDT")
        )
        .drop_nulls(subset=["CALDT"])
        .unique()
        .sort("CALDT")
    )
    base = sample_backbone_lf.with_columns(
        pl.col(permno_col).cast(pl.Int32, strict=False).alias("KYPERMNO"),
        pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("_lm2011_gvkey_int"),
    )
    anchored = derive_filing_trade_anchors(base, trading_calendar)
    annual_attached = attach_latest_annual_accounting(
        anchored,
        annual_accounting_panel_lf,
        filing_gvkey_col="_lm2011_gvkey_int",
    ).with_columns(pl.col("_lm2011_gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int"))
    with_pre_market = attach_pre_filing_market_data(
        annual_attached,
        daily_lf,
        filing_permno_col="KYPERMNO",
        pre_filing_trade_date_col="pre_filing_trade_date",
    )
    with_pre_market_schema = with_pre_market.collect_schema()
    text_df = _prepare_unique_doc_metric_frame(
        full_10k_text_features_lf,
        metric_col="total_token_count_full_10k",
        label="full_10k_text_features_lf",
        required_message=(
            "full_10k_text_features_lf is required; fail-closed external_input_policy forbids LM2011 clean panels without token-count text features"
        ),
        metric_dtype=pl.Int32,
    )
    return (
        with_pre_market
        .join(text_df.lazy(), on="doc_id", how="left")
        .with_columns(
            _price_expr_from_available_columns(
                with_pre_market_schema,
                candidates=("pre_filing_final_prc", "pre_filing_prc"),
                label="event base frame",
            ).alias("pre_filing_price"),
            pl.col("KYPERMNO").cast(pl.Int32, strict=False),
            pl.col("gvkey_int").cast(pl.Int32, strict=False),
        )
    )


def _prepare_table_i_base_frame(
    sample_backbone_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    annual_accounting_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame | None,
) -> pl.DataFrame:
    return _prepare_table_i_base_lf(
        sample_backbone_lf,
        daily_lf,
        annual_accounting_panel_lf,
        full_10k_text_features_lf,
    ).collect()


def _build_event_doc_manifest(docs_base: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    if isinstance(docs_base, pl.DataFrame):
        manifest = docs_base.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
            pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date"),
            pl.col("KYPERMNO").cast(pl.Int32, strict=False).alias("KYPERMNO"),
        )
    else:
        manifest = docs_base.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
            pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date"),
            pl.col("KYPERMNO").cast(pl.Int32, strict=False).alias("KYPERMNO"),
        ).collect()
    manifest = manifest.sort("filing_date", "KYPERMNO", "doc_id")
    duplicate_doc_ids = (
        manifest.group_by("doc_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("doc_id")
        .to_list()
    )
    if duplicate_doc_ids:
        raise ValueError(f"LM2011 event-screen docs must be unique on doc_id: {duplicate_doc_ids[:10]}")
    return manifest


def _collect_event_doc_batch(docs_df: pl.DataFrame, *, batch_start: int, batch_size: int) -> pl.DataFrame:
    return docs_df.slice(batch_start, batch_size)


def _prepare_daily_event_source_lf(
    daily_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    daily_schema = daily_lf.collect_schema()
    daily_permno_col = _resolve_first_existing(daily_schema, ("KYPERMNO", "kypermno"), "daily_panel")
    daily_date_col = _resolve_first_existing(daily_schema, ("CALDT", "daily_caldt"), "daily_panel")
    return_col = "FINAL_RET" if "FINAL_RET" in daily_schema else "RET"
    price_col = "FINAL_PRC" if "FINAL_PRC" in daily_schema else "PRC"
    required_daily = [daily_permno_col, daily_date_col, return_col, "VOL", price_col, "SHROUT", "SHRCD", "EXCHCD"]
    missing_daily = [name for name in required_daily if name not in daily_schema]
    if missing_daily:
        raise ValueError(f"daily_lf missing required columns for LM2011 event metrics: {missing_daily}")
    return daily_lf.select(
        pl.col(daily_permno_col).cast(pl.Int32, strict=False).alias("KYPERMNO"),
        pl.col(daily_date_col).cast(pl.Date, strict=False).alias("trade_date"),
        pl.col(return_col).cast(pl.Float64, strict=False).alias("stock_return"),
        pl.col("VOL").cast(pl.Float64, strict=False).alias("VOL"),
        pl.col(price_col).cast(pl.Float64, strict=False).alias("FINAL_PRC"),
        (
            pl.col("PRC").cast(pl.Float64, strict=False)
            if "PRC" in daily_schema
            else pl.col(price_col).cast(pl.Float64, strict=False)
        ).alias("PRC"),
        (
            pl.col("TCAP").cast(pl.Float64, strict=False)
            if "TCAP" in daily_schema
            else pl.lit(None, dtype=pl.Float64)
        ).alias("TCAP"),
        pl.col("SHROUT").cast(pl.Float64, strict=False).alias("SHROUT"),
        pl.col("SHRCD").cast(pl.Int32, strict=False).alias("SHRCD"),
        pl.col("EXCHCD").cast(pl.Int32, strict=False).alias("EXCHCD"),
    ).drop_nulls(subset=["KYPERMNO", "trade_date"])


def _prepare_daily_factor_frame(ff_factors_daily_lf: pl.LazyFrame | None) -> pl.DataFrame:
    if ff_factors_daily_lf is None:
        raise ValueError(
            "ff_factors_daily_lf is required; fail-closed external_input_policy forbids LM2011 event panels without daily FF factors"
        )
    _require_columns(ff_factors_daily_lf, ("trading_date", "mkt_rf", "smb", "hml", "rf"), "ff_factors_daily")
    factors_df = ff_factors_daily_lf.select(
        pl.col("trading_date").cast(pl.Date, strict=False).alias("trade_date"),
        pl.col("mkt_rf").cast(pl.Float64, strict=False).alias("mkt_rf"),
        pl.col("smb").cast(pl.Float64, strict=False).alias("smb"),
        pl.col("hml").cast(pl.Float64, strict=False).alias("hml"),
        pl.col("rf").cast(pl.Float64, strict=False).alias("rf"),
    ).collect()
    factors_df = _ensure_factor_scale(factors_df, ("mkt_rf", "smb", "hml", "rf"))
    return factors_df.with_columns((pl.col("mkt_rf") + pl.col("rf")).alias("market_return"))


def _collect_daily_event_batch(
    daily_source_lf: pl.LazyFrame,
    factors_df: pl.DataFrame,
    docs_df: pl.DataFrame,
) -> pl.DataFrame:
    permnos = docs_df.get_column("KYPERMNO").drop_nulls().unique().to_list()
    if not permnos:
        return _empty_prepared_daily_event_df()

    filing_trade_dates = docs_df.get_column("filing_trade_date").drop_nulls()
    if filing_trade_dates.len() == 0:
        return _empty_prepared_daily_event_df()
    min_trade_date = _coerce_python_date(filing_trade_dates.min(), label="minimum filing_trade_date")
    max_trade_date = _coerce_python_date(filing_trade_dates.max(), label="maximum filing_trade_date")
    min_lookup_date = min_trade_date - dt.timedelta(days=450)
    max_lookup_date = max_trade_date + dt.timedelta(days=450)
    daily_df = (
        daily_source_lf.filter(
            pl.col("KYPERMNO").cast(pl.Int32, strict=False).is_in(permnos)
            & pl.col("trade_date").cast(pl.Date, strict=False).is_between(
                pl.lit(min_lookup_date),
                pl.lit(max_lookup_date),
                closed="both",
            )
        )
        .collect()
        .unique(subset=["KYPERMNO", "trade_date"], keep="first")
        .sort("KYPERMNO", "trade_date")
    )
    if daily_df.height == 0:
        return _empty_prepared_daily_event_df()
    return daily_df.join(factors_df, on="trade_date", how="left").sort("KYPERMNO", "trade_date")


def _collect_event_screen_surface_df(event_screen_surface: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    surface_df = event_screen_surface.collect() if isinstance(event_screen_surface, pl.LazyFrame) else event_screen_surface
    missing = [name for name in _EVENT_SCREEN_SURFACE_REQUIRED_COLUMNS if name not in surface_df.columns]
    if missing:
        raise ValueError(f"event_screen_surface missing required columns: {missing}")
    duplicate_doc_ids = (
        surface_df.group_by("doc_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("doc_id")
        .to_list()
    )
    if duplicate_doc_ids:
        raise ValueError(f"event_screen_surface must be unique on doc_id: {duplicate_doc_ids[:10]}")
    return surface_df


def _attach_trade_indices(docs_df: pl.DataFrame, daily_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    calendar_df = (
        daily_df.select("trade_date")
        .drop_nulls()
        .unique()
        .sort("trade_date")
        .with_row_index("trade_index", offset=0)
        .with_columns(pl.col("trade_index").cast(pl.Int32, strict=False))
    )
    docs = (
        docs_df.join(
            calendar_df.rename({"trade_date": "filing_trade_date", "trade_index": "_filing_trade_index"}),
            on="filing_trade_date",
            how="left",
        )
        .join(
            calendar_df.rename({"trade_date": "pre_filing_trade_date", "trade_index": "_pre_filing_trade_index"}),
            on="pre_filing_trade_date",
            how="left",
        )
    )
    daily = daily_df.join(calendar_df, on="trade_date", how="left")
    return docs, daily


def _build_window_rows(docs_df: pl.DataFrame, daily_df: pl.DataFrame) -> pl.DataFrame:
    left = docs_df.lazy().rename({"KYPERMNO": "_doc_permno"})
    right = daily_df.lazy().rename(
        {
            "KYPERMNO": "_daily_permno",
            "trade_index": "_daily_trade_index",
            "trade_date": "_daily_trade_date",
            "VOL": "_daily_VOL",
            "FINAL_PRC": "_daily_FINAL_PRC",
            "PRC": "_daily_PRC",
            "TCAP": "_daily_TCAP",
            "SHROUT": "_daily_SHROUT",
            "SHRCD": "_daily_SHRCD",
            "EXCHCD": "_daily_EXCHCD",
            "mkt_rf": "_daily_mkt_rf",
            "smb": "_daily_smb",
            "hml": "_daily_hml",
            "rf": "_daily_rf",
            "market_return": "_daily_market_return",
        }
    )
    return (
        left.join_where(
            right,
            pl.col("_doc_permno") == pl.col("_daily_permno"),
            pl.col("_daily_trade_index") >= (pl.col("_filing_trade_index") - pl.lit(252)),
            pl.col("_daily_trade_index") <= (pl.col("_filing_trade_index") + pl.lit(252)),
        )
        .with_columns(
            pl.col("_daily_trade_index").cast(pl.Int32, strict=False),
            pl.col("_filing_trade_index").cast(pl.Int32, strict=False),
            (
                pl.col("_daily_trade_index").cast(pl.Int32, strict=False)
                - pl.col("_filing_trade_index").cast(pl.Int32, strict=False)
            ).alias("relative_day"),
            pl.col("_daily_trade_date").alias("trade_date"),
            pl.col("_daily_VOL").alias("VOL"),
            pl.col("_daily_FINAL_PRC").alias("FINAL_PRC"),
            pl.col("_daily_PRC").alias("PRC"),
            pl.col("_daily_TCAP").alias("TCAP"),
            pl.col("_daily_SHROUT").alias("SHROUT"),
            pl.col("_daily_SHRCD").alias("SHRCD"),
            pl.col("_daily_EXCHCD").alias("EXCHCD"),
            pl.col("_daily_mkt_rf").alias("mkt_rf"),
            pl.col("_daily_smb").alias("smb"),
            pl.col("_daily_hml").alias("hml"),
            pl.col("_daily_rf").alias("rf"),
            pl.col("_daily_market_return").alias("market_return"),
        )
        .collect()
        .sort("doc_id", "relative_day", "trade_date")
    )
def _require_statsmodels():
    if sm is None:
        raise ImportError("statsmodels is required for LM2011 regression fitting")
    return sm


def _fit_checked_ols(
    endog: object,
    exog: object,
    *,
    exog_names: tuple[str, ...],
    label: str,
):
    sm_api = _require_statsmodels()
    design = sm_api.add_constant(exog, has_constant="add")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in scalar divide",
            category=RuntimeWarning,
        )
        results = sm_api.OLS(endog, design).fit()
    rank = int(results.model.rank)
    column_count = len(exog_names) + 1
    if rank < column_count:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in scalar divide",
                category=RuntimeWarning,
            )
            condition_number = getattr(results, "condition_number", None)
        condition_suffix = (
            f", condition_number={float(condition_number):.6g}"
            if condition_number is not None
            else ""
        )
        regressors = ", ".join(("intercept", *exog_names))
        raise ValueError(
            f"{label} rank-deficient OLS design: rank={rank}, columns={column_count}"
            f"{condition_suffix}, regressors=[{regressors}]"
        )
    return results


def _ols_alpha_and_rmse(df: pl.DataFrame, *, label: str) -> tuple[float | None, float | None]:
    n = df.height
    if n <= 4:
        return None, None
    results = _fit_checked_ols(
        df.get_column("_y").cast(pl.Float64, strict=False).to_numpy(),
        df.select(
            pl.col("_x1").cast(pl.Float64, strict=False),
            pl.col("_x2").cast(pl.Float64, strict=False),
            pl.col("_x3").cast(pl.Float64, strict=False),
        ).to_numpy(),
        exog_names=("_x1", "_x2", "_x3"),
        label=label,
    )
    alpha = float(results.params[0])
    rmse = math.sqrt(max(float(results.mse_resid), 0.0))
    return alpha, rmse


def _regression_metrics_from_window(
    window_df: pl.DataFrame,
    *,
    start_day: int,
    end_day: int,
    alpha_name: str,
    rmse_name: str,
) -> pl.DataFrame:
    subset = window_df.filter(
        pl.col("relative_day").is_between(start_day, end_day, closed="both")
        & pl.col("stock_return").is_not_null()
        & pl.col("mkt_rf").is_not_null()
        & pl.col("smb").is_not_null()
        & pl.col("hml").is_not_null()
        & pl.col("rf").is_not_null()
    ).with_columns(
        (pl.col("stock_return") - pl.col("rf")).alias("_y"),
        pl.col("mkt_rf").alias("_x1"),
        pl.col("smb").alias("_x2"),
        pl.col("hml").alias("_x3"),
    )
    if subset.height == 0:
        return pl.DataFrame(
            schema={
                "doc_id": pl.Utf8,
                alpha_name: pl.Float64,
                rmse_name: pl.Float64,
                "n_obs": pl.Int32,
            }
        )
    rows: list[dict[str, object]] = []
    for group in subset.sort("doc_id", "relative_day").partition_by("doc_id", maintain_order=True):
        doc_id = str(group.item(0, "doc_id"))
        alpha, rmse = _ols_alpha_and_rmse(
            group,
            label=f"{alpha_name} regression for doc_id={doc_id}",
        )
        rows.append(
            {
                "doc_id": doc_id,
                alpha_name: alpha,
                rmse_name: rmse,
                "n_obs": group.height,
            }
        )
    return pl.DataFrame(
        rows,
        schema_overrides={
            "doc_id": pl.Utf8,
            alpha_name: pl.Float64,
            rmse_name: pl.Float64,
            "n_obs": pl.Int32,
        },
    )


def _winsorize_column(df: pl.DataFrame, column: str, *, lower_q: float = 0.01, upper_q: float = 0.99) -> pl.DataFrame:
    if df.height == 0 or column not in df.columns:
        return df
    quantiles = df.select(
        pl.col(column).cast(pl.Float64, strict=False).quantile(lower_q).alias("_lower"),
        pl.col(column).cast(pl.Float64, strict=False).quantile(upper_q).alias("_upper"),
    ).row(0, named=True)
    lower = quantiles["_lower"]
    upper = quantiles["_upper"]
    if lower is None or upper is None:
        return df
    return df.with_columns(pl.col(column).cast(pl.Float64, strict=False).clip(lower, upper).alias(column))


def _apply_lm2011_regression_transforms(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.when(pl.col("size_event").cast(pl.Float64, strict=False) > 0)
        .then(pl.col("size_event").cast(pl.Float64, strict=False).log())
        .otherwise(None)
        .alias("log_size"),
        pl.when(pl.col("bm_event").cast(pl.Float64, strict=False) > 0)
        .then(pl.col("bm_event").cast(pl.Float64, strict=False).log())
        .otherwise(None)
        .alias("log_book_to_market"),
        pl.when(pl.col("share_turnover").cast(pl.Float64, strict=False) > 0)
        .then(pl.col("share_turnover").cast(pl.Float64, strict=False).log())
        .otherwise(None)
        .alias("log_share_turnover"),
    )


def _build_lm2011_event_screen_surface(docs_df: pl.DataFrame, daily_df: pl.DataFrame) -> pl.DataFrame:
    docs_df, daily_df = _attach_trade_indices(docs_df, daily_df)
    window_df = _build_window_rows(docs_df, daily_df)

    event_summary = window_df.group_by("doc_id").agg(
        pl.when(pl.col("relative_day").is_between(0, 3, closed="both") & pl.col("stock_return").is_not_null())
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("event_return_day_count"),
        pl.when(pl.col("relative_day").is_between(0, 3, closed="both") & pl.col("VOL").is_not_null())
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("event_volume_day_count"),
        pl.when(pl.col("relative_day").is_between(0, 3, closed="both"))
        .then(pl.col("stock_return") + pl.lit(1.0))
        .otherwise(None)
        .product()
        .alias("_event_stock_gross"),
        pl.when(pl.col("relative_day").is_between(0, 3, closed="both"))
        .then(pl.col("market_return") + pl.lit(1.0))
        .otherwise(None)
        .product()
        .alias("_event_market_gross"),
        pl.when(
            pl.col("relative_day").is_between(-252, -6, closed="both")
            & pl.col("stock_return").is_not_null()
            & pl.col("VOL").is_not_null()
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("pre_turnover_obs"),
        pl.when(pl.col("relative_day").is_between(-252, -6, closed="both"))
        .then(pl.col("VOL"))
        .otherwise(None)
        .sum()
        .alias("_turnover_volume_sum"),
        pl.when(pl.col("relative_day").is_between(-65, -6, closed="both") & pl.col("VOL").is_not_null())
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("abnormal_volume_pre_obs"),
        pl.when(pl.col("relative_day") == 0)
        .then(pl.col("SHROUT"))
        .otherwise(None)
        .drop_nulls()
        .first()
        .alias("event_shares"),
        pl.when(pl.col("relative_day") == 0).then(pl.col("SHRCD")).otherwise(None).drop_nulls().first().alias("event_shrcd"),
        pl.when(pl.col("relative_day") == 0).then(pl.col("EXCHCD")).otherwise(None).drop_nulls().first().alias("event_exchcd"),
    )

    abnormal_pre_stats = window_df.filter(
        pl.col("relative_day").is_between(-65, -6, closed="both") & pl.col("VOL").is_not_null()
    ).group_by("doc_id").agg(
        pl.col("VOL").mean().alias("_pre_vol_mean"),
        pl.col("VOL").std().alias("_pre_vol_std"),
    )
    abnormal_event = (
        window_df.filter(pl.col("relative_day").is_between(0, 3, closed="both") & pl.col("VOL").is_not_null())
        .join(abnormal_pre_stats, on="doc_id", how="left")
        .filter(pl.col("_pre_vol_std").is_not_null() & (pl.col("_pre_vol_std") > 0))
        .with_columns(((pl.col("VOL") - pl.col("_pre_vol_mean")) / pl.col("_pre_vol_std")).alias("_std_volume"))
        .group_by("doc_id")
        .agg(pl.col("_std_volume").mean().alias("abnormal_volume"))
        .with_columns(pl.col("doc_id").cast(pl.Utf8, strict=False))
    )

    pre_alpha = _regression_metrics_from_window(
        window_df,
        start_day=-252,
        end_day=-6,
        alpha_name="pre_ffalpha",
        rmse_name="_pre_ffalpha_rmse",
    ).rename({"n_obs": "pre_alpha_obs"})
    post_alpha = _regression_metrics_from_window(
        window_df,
        start_day=6,
        end_day=252,
        alpha_name="_post_ffalpha",
        rmse_name="postevent_return_volatility",
    ).rename({"n_obs": "post_alpha_obs"})
    event_summary = event_summary.with_columns(pl.col("doc_id").cast(pl.Utf8, strict=False))
    pre_alpha = pre_alpha.with_columns(pl.col("doc_id").cast(pl.Utf8, strict=False))
    post_alpha = post_alpha.with_columns(pl.col("doc_id").cast(pl.Utf8, strict=False))

    surface = (
        docs_df.join(event_summary, on="doc_id", how="left")
        .join(abnormal_event, on="doc_id", how="left")
        .join(pre_alpha, on="doc_id", how="left")
        .join(post_alpha, on="doc_id", how="left")
        .with_columns(
            (pl.col("_event_stock_gross") - pl.col("_event_market_gross")).alias("filing_period_excess_return"),
            share_turnover_ratio(
                volume=pl.col("_turnover_volume_sum"),
                shrout=pl.col("event_shares"),
            ).alias("share_turnover"),
            pl.when(pl.col("event_exchcd") == 3)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("nasdaq_dummy"),
        )
    )
    return _project_public_event_screen_surface(surface)


def _project_public_event_screen_surface(surface_df: pl.DataFrame) -> pl.DataFrame:
    return surface_df.select(_empty_event_screen_surface_df().columns)


def write_lm2011_event_screen_surface_parquet(
    sample_backbone_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    annual_accounting_panel_lf: pl.LazyFrame,
    ff_factors_daily_lf: pl.LazyFrame | None,
    full_10k_text_features_lf: pl.LazyFrame | None,
    *,
    output_path: Path,
    event_window_doc_batch_size: int = DEFAULT_EVENT_WINDOW_DOC_BATCH_SIZE,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    temp_root: Path | None = None,
    cleanup_on_success: bool = True,
) -> int:
    docs_base_lf = _prepare_table_i_base_lf(
        sample_backbone_lf,
        daily_lf,
        annual_accounting_panel_lf,
        full_10k_text_features_lf,
    )
    # Materialize the doc base once here so the heavy upstream joins are not rerun for
    # every event-window shard.
    docs_df = docs_base_lf.collect().sort("filing_date", "KYPERMNO", "doc_id")
    docs_manifest = _build_event_doc_manifest(docs_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_temp_root = temp_root or output_path.parent / f".{output_path.stem}_tmp"
    if resolved_temp_root.exists():
        shutil.rmtree(resolved_temp_root)
    resolved_temp_root.mkdir(parents=True, exist_ok=True)
    shard_dir = resolved_temp_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    try:
        if docs_manifest.height == 0:
            _empty_event_screen_surface_df().write_parquet(
                output_path,
                compression=_STREAMING_PARQUET_COMPRESSION,
            )
            if cleanup_on_success and resolved_temp_root.exists():
                shutil.rmtree(resolved_temp_root)
            return 0

        batch_size = _validated_event_window_doc_batch_size(event_window_doc_batch_size)
        daily_source_lf = _prepare_daily_event_source_lf(daily_lf)
        factors_df = _prepare_daily_factor_frame(ff_factors_daily_lf)
        total_docs = docs_manifest.height
        total_batches = int(math.ceil(total_docs / batch_size))
        shard_paths: list[Path] = []

        for batch_start in range(0, total_docs, batch_size):
            docs_batch = _collect_event_doc_batch(
                docs_df,
                batch_start=batch_start,
                batch_size=batch_size,
            )
            daily_batch_df = _collect_daily_event_batch(daily_source_lf, factors_df, docs_batch)
            surface_batch = _build_lm2011_event_screen_surface(docs_batch, daily_batch_df)
            if not surface_batch.is_empty():
                shard_path = shard_dir / f"{int(batch_start / batch_size) + 1:06d}.parquet"
                surface_batch.write_parquet(shard_path, compression=_STREAMING_PARQUET_COMPRESSION)
                shard_paths.append(shard_path)
            if progress_callback is not None:
                batch_index = int(batch_start / batch_size) + 1
                progress_callback(
                    {
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "batch_doc_count": docs_batch.height,
                        "docs_completed": min(batch_start + docs_batch.height, total_docs),
                        "docs_total": total_docs,
                    }
                )
            del docs_batch
            del daily_batch_df
            del surface_batch
            gc.collect()

        if output_path.exists():
            output_path.unlink()
        if shard_paths:
            pl.scan_parquet([str(path) for path in shard_paths]).sink_parquet(
                output_path,
                compression=_STREAMING_PARQUET_COMPRESSION,
            )
        else:
            _empty_event_screen_surface_df().write_parquet(
                output_path,
                compression=_STREAMING_PARQUET_COMPRESSION,
            )
        row_count = int(pl.scan_parquet(output_path).select(pl.len()).collect().item())
        if cleanup_on_success and resolved_temp_root.exists():
            shutil.rmtree(resolved_temp_root)
        return row_count
    except Exception:
        raise


def _build_lm2011_event_screen_surface_batched(
    sample_backbone_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    annual_accounting_panel_lf: pl.LazyFrame,
    ff_factors_daily_lf: pl.LazyFrame | None,
    full_10k_text_features_lf: pl.LazyFrame | None,
    *,
    event_window_doc_batch_size: int = DEFAULT_EVENT_WINDOW_DOC_BATCH_SIZE,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> pl.DataFrame:
    docs_base_lf = _prepare_table_i_base_lf(
        sample_backbone_lf,
        daily_lf,
        annual_accounting_panel_lf,
        full_10k_text_features_lf,
    )
    # The batched builder keeps a single eager doc base so batch iteration does not
    # repeatedly rerun the upstream backbone/accounting/pre-market join graph.
    docs_df = docs_base_lf.collect().sort("filing_date", "KYPERMNO", "doc_id")
    docs_manifest = _build_event_doc_manifest(docs_df)
    if docs_manifest.height == 0:
        return _empty_event_screen_surface_df()

    batch_size = _validated_event_window_doc_batch_size(event_window_doc_batch_size)
    daily_source_lf = _prepare_daily_event_source_lf(daily_lf)
    factors_df = _prepare_daily_factor_frame(ff_factors_daily_lf)
    batch_frames: list[pl.DataFrame] = []
    total_docs = docs_manifest.height
    total_batches = int(math.ceil(total_docs / batch_size))

    for batch_start in range(0, total_docs, batch_size):
        docs_batch = _collect_event_doc_batch(
            docs_df,
            batch_start=batch_start,
            batch_size=batch_size,
        )
        daily_batch_df = _collect_daily_event_batch(daily_source_lf, factors_df, docs_batch)
        batch_frames.append(_build_lm2011_event_screen_surface(docs_batch, daily_batch_df))
        if progress_callback is not None:
            batch_index = int(batch_start / batch_size) + 1
            progress_callback(
                {
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "batch_doc_count": docs_batch.height,
                    "docs_completed": min(batch_start + docs_batch.height, total_docs),
                    "docs_total": total_docs,
                }
            )
        del docs_batch
        del daily_batch_df
        gc.collect()

    if not batch_frames:
        return _empty_event_screen_surface_df()
    return pl.concat(batch_frames, how="vertical_relaxed")


def _lm2011_table_i_market_stage_specs() -> tuple[tuple[str, pl.Expr], ...]:
    return (
        ("ordinary_common_equity", pl.col("event_shrcd").is_in([10, 11])),
        (
            "market_cap_available",
            pl.col("size_event").cast(pl.Float64, strict=False).is_not_null()
            & (pl.col("size_event").cast(pl.Float64, strict=False) > 0),
        ),
        (
            "price_day_minus_one_ge_3",
            pl.col("pre_filing_price").cast(pl.Float64, strict=False).is_not_null()
            & (pl.col("pre_filing_price").cast(pl.Float64, strict=False) >= 3.0),
        ),
        (
            "event_window_returns_and_volume",
            (pl.col("event_return_day_count") == 4) & (pl.col("event_volume_day_count") == 4),
        ),
        ("major_exchange_listing", pl.col("event_exchcd").is_in([1, 2, 3])),
        (
            "sixty_day_pre_post_coverage",
            (pl.col("pre_turnover_obs") >= 60)
            & (pl.col("abnormal_volume_pre_obs") >= 60)
            & (pl.col("pre_alpha_obs") >= 60)
            & (pl.col("post_alpha_obs") >= 60),
        ),
        (
            "book_to_market_available_and_book_value_positive",
            (pl.col("book_equity_be").cast(pl.Float64, strict=False) > 0)
            & (pl.col("bm_event").cast(pl.Float64, strict=False) > 0),
        ),
        ("token_count_ge_2000", pl.col("total_token_count_full_10k").cast(pl.Int32, strict=False) >= 2000),
    )


def _apply_lm2011_table_i_market_filters(panel: pl.DataFrame) -> tuple[tuple[str, pl.DataFrame], ...]:
    current = panel
    stage_frames: list[tuple[str, pl.DataFrame]] = []
    for row_id, predicate in _lm2011_table_i_market_stage_specs():
        current = current.filter(predicate)
        stage_frames.append((row_id, current))
    return tuple(stage_frames)


def _build_lm2011_table_i_market_stage_frames(
    sample_backbone_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    annual_accounting_panel_lf: pl.LazyFrame,
    ff_factors_daily_lf: pl.LazyFrame | None,
    full_10k_text_features_lf: pl.LazyFrame | None,
    *,
    event_window_doc_batch_size: int = DEFAULT_EVENT_WINDOW_DOC_BATCH_SIZE,
    precomputed_event_screen_surface_lf: pl.LazyFrame | pl.DataFrame | None = None,
    event_screen_progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> tuple[tuple[str, pl.DataFrame], ...]:
    panel = (
        _collect_event_screen_surface_df(precomputed_event_screen_surface_lf)
        if precomputed_event_screen_surface_lf is not None
        else _build_lm2011_event_screen_surface_batched(
            sample_backbone_lf,
            daily_lf,
            annual_accounting_panel_lf.with_columns(pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int")),
            ff_factors_daily_lf,
            full_10k_text_features_lf,
            event_window_doc_batch_size=event_window_doc_batch_size,
            progress_callback=event_screen_progress_callback,
        )
    )
    if panel.height == 0:
        return tuple((row_id, panel.clone()) for row_id, _ in _lm2011_table_i_market_stage_specs())
    return _apply_lm2011_table_i_market_filters(panel)


def _build_table_i_sample_creation_row(
    *,
    section_id: str,
    section_label: str,
    section_order: int,
    row_order: int,
    row_id: str,
    display_label: str,
    sample_size_kind: str,
    sample_size_value: int | float | None,
    observations_removed: int | None,
    availability_status: str,
    availability_reason: str | None = None,
) -> dict[str, object]:
    return {
        "section_id": section_id,
        "section_label": section_label,
        "section_order": section_order,
        "row_order": row_order,
        "row_id": row_id,
        "display_label": display_label,
        "sample_size_kind": sample_size_kind,
        "sample_size_value": float(sample_size_value) if sample_size_value is not None else None,
        "observations_removed": observations_removed,
        "availability_status": availability_status,
        "availability_reason": availability_reason,
    }


def _build_lm2011_table_i_output(
    early_stage_frames: tuple[tuple[str, pl.LazyFrame], ...],
    market_stage_frames: tuple[tuple[str, pl.DataFrame], ...],
    *,
    sample_start: dt.date,
    sample_end: dt.date,
    mda_text_features_lf: pl.LazyFrame | None,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    row_order = 1
    previous_count: int | None = None
    final_market_df = market_stage_frames[-1][1] if market_stage_frames else pl.DataFrame()
    full_10k_row_labels = _table_i_full_10k_row_labels(sample_start, sample_end)

    for row_id, frame in (*early_stage_frames, *market_stage_frames):
        current_count = _frame_height(frame)
        rows.append(
            _build_table_i_sample_creation_row(
                section_id=_LM2011_TABLE_I_FULL_10K_SECTION_ID,
                section_label=_LM2011_TABLE_I_FULL_10K_SECTION_LABEL,
                section_order=1,
                row_order=row_order,
                row_id=row_id,
                display_label=full_10k_row_labels[row_id],
                sample_size_kind="count",
                sample_size_value=current_count,
                observations_removed=None if previous_count is None else previous_count - current_count,
                availability_status="available",
            )
        )
        previous_count = current_count
        row_order += 1

    final_count = _frame_height(final_market_df)
    unique_firms = (
        final_market_df.select(pl.col("gvkey_int").drop_nulls().n_unique().alias("_n")).item()
        if final_count > 0 and "gvkey_int" in final_market_df.columns
        else 0
    )
    average_years_per_firm = (float(final_count) / float(unique_firms)) if unique_firms else None
    for row_id, sample_size_kind, sample_size_value in (
        ("firm_year_sample", "count", final_count),
        ("unique_firms", "count", unique_firms),
        ("average_years_per_firm", "mean", average_years_per_firm),
    ):
        rows.append(
            _build_table_i_sample_creation_row(
                section_id=_LM2011_TABLE_I_FIRM_YEAR_SECTION_ID,
                section_label=_LM2011_TABLE_I_FIRM_YEAR_SECTION_LABEL,
                section_order=2,
                row_order=row_order,
                row_id=row_id,
                display_label=_LM2011_TABLE_I_FIRM_YEAR_ROW_LABELS[row_id],
                sample_size_kind=sample_size_kind,
                sample_size_value=sample_size_value,
                observations_removed=None,
                availability_status="available",
            )
        )
        row_order += 1

    if mda_text_features_lf is None:
        for row_id in ("identifiable_mda", "mda_token_count_ge_250"):
            rows.append(
                _build_table_i_sample_creation_row(
                    section_id=_LM2011_TABLE_I_MDA_SECTION_ID,
                    section_label=_LM2011_TABLE_I_MDA_SECTION_LABEL,
                    section_order=3,
                    row_order=row_order,
                    row_id=row_id,
                    display_label=_LM2011_TABLE_I_MDA_ROW_LABELS[row_id],
                    sample_size_kind="count",
                    sample_size_value=None,
                    observations_removed=None,
                    availability_status="unavailable",
                    availability_reason="mda_text_features_unavailable",
                )
            )
            row_order += 1
    else:
        mda_df = _prepare_unique_doc_metric_frame(
            mda_text_features_lf,
            metric_col="total_token_count_mda",
            label="mda_text_features_lf",
            required_message="mda_text_features_lf is required for MD&A subsection rows",
            metric_dtype=pl.Int32,
        )
        identifiable_mda = final_market_df.join(mda_df, on="doc_id", how="inner")
        mda_token_count_ge_250 = identifiable_mda.filter(pl.col("total_token_count_mda").cast(pl.Int32, strict=False) >= 250)
        previous_mda_count = final_count
        for row_id, frame in (
            ("identifiable_mda", identifiable_mda),
            ("mda_token_count_ge_250", mda_token_count_ge_250),
        ):
            current_count = _frame_height(frame)
            rows.append(
                _build_table_i_sample_creation_row(
                    section_id=_LM2011_TABLE_I_MDA_SECTION_ID,
                    section_label=_LM2011_TABLE_I_MDA_SECTION_LABEL,
                    section_order=3,
                    row_order=row_order,
                    row_id=row_id,
                    display_label=_LM2011_TABLE_I_MDA_ROW_LABELS[row_id],
                    sample_size_kind="count",
                    sample_size_value=current_count,
                    observations_removed=previous_mda_count - current_count,
                    availability_status="available",
                )
            )
            previous_mda_count = current_count
            row_order += 1

    return pl.DataFrame(
        rows,
        schema_overrides={
            "section_id": pl.Utf8,
            "section_label": pl.Utf8,
            "section_order": pl.Int8,
            "row_order": pl.Int16,
            "row_id": pl.Utf8,
            "display_label": pl.Utf8,
            "sample_size_kind": pl.Utf8,
            "sample_size_value": pl.Float64,
            "observations_removed": pl.Int64,
            "availability_status": pl.Utf8,
            "availability_reason": pl.Utf8,
        },
    ).sort("section_order", "row_order")


def _build_lm2011_event_panel_df_from_surface(
    event_screen_surface: pl.LazyFrame | pl.DataFrame,
    ownership_lf: pl.LazyFrame | None,
) -> pl.DataFrame:
    surface_df = _collect_event_screen_surface_df(event_screen_surface)
    if surface_df.height == 0:
        return _empty_event_panel_df()

    ownership_df = _prepare_ownership_frame(ownership_lf)
    panel = surface_df.join(ownership_df, on="doc_id", how="left")
    panel = _apply_lm2011_table_i_market_filters(panel)[-1][1]
    panel = _winsorize_column(panel, "bm_event")
    panel = panel.filter(pl.col("abnormal_volume").is_not_null())
    if panel.height == 0:
        return _empty_event_panel_df()

    return panel.select(
        "doc_id",
        "gvkey_int",
        "KYPERMNO",
        "filing_date",
        "filing_trade_date",
        "pre_filing_trade_date",
        "size_event",
        "bm_event",
        "share_turnover",
        "pre_ffalpha",
        "institutional_ownership",
        "nasdaq_dummy",
        "filing_period_excess_return",
        "abnormal_volume",
        "postevent_return_volatility",
    )


def build_lm2011_table_i_sample_creation(
    sec_parsed_lf: pl.LazyFrame,
    matched_clean_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    annual_accounting_panel_lf: pl.LazyFrame,
    ff_factors_daily_lf: pl.LazyFrame | None,
    full_10k_text_features_lf: pl.LazyFrame | None,
    *,
    ccm_filingdates_lf: pl.LazyFrame | None = None,
    mda_text_features_lf: pl.LazyFrame | None = None,
    sample_start: dt.date = _LM2011_TABLE_I_DEFAULT_SAMPLE_START,
    sample_end: dt.date = _LM2011_TABLE_I_DEFAULT_SAMPLE_END,
    event_window_doc_batch_size: int = DEFAULT_EVENT_WINDOW_DOC_BATCH_SIZE,
    _precomputed_event_screen_surface_lf: pl.LazyFrame | pl.DataFrame | None = None,
    _event_screen_progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> pl.DataFrame:
    """Build the LM2011 Table I sample-selection ladder as a normalized row table."""
    early_stage_frames = _build_lm2011_sample_backbone_stage_frames(
        sec_parsed_lf,
        matched_clean_lf,
        ccm_filingdates_lf=ccm_filingdates_lf,
        sample_start=sample_start,
        sample_end=sample_end,
    )
    market_stage_frames = _build_lm2011_table_i_market_stage_frames(
        early_stage_frames[-1][1],
        daily_lf,
        annual_accounting_panel_lf,
        ff_factors_daily_lf,
        full_10k_text_features_lf,
        event_window_doc_batch_size=event_window_doc_batch_size,
        precomputed_event_screen_surface_lf=_precomputed_event_screen_surface_lf,
        event_screen_progress_callback=_event_screen_progress_callback,
    )
    return _build_lm2011_table_i_output(
        early_stage_frames,
        market_stage_frames,
        sample_start=sample_start,
        sample_end=sample_end,
        mda_text_features_lf=mda_text_features_lf,
    )


def build_lm2011_event_panel(
    sample_backbone_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    annual_accounting_panel_lf: pl.LazyFrame,
    ff_factors_daily_lf: pl.LazyFrame | None,
    ownership_lf: pl.LazyFrame | None,
    full_10k_text_features_lf: pl.LazyFrame | None,
    *,
    event_window_doc_batch_size: int = DEFAULT_EVENT_WINDOW_DOC_BATCH_SIZE,
    _precomputed_event_screen_surface_lf: pl.LazyFrame | pl.DataFrame | None = None,
    _event_screen_progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> pl.LazyFrame:
    event_screen_surface = (
        _precomputed_event_screen_surface_lf
        if _precomputed_event_screen_surface_lf is not None
        else _build_lm2011_event_screen_surface_batched(
            sample_backbone_lf,
            daily_lf,
            annual_accounting_panel_lf.with_columns(pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int")),
            ff_factors_daily_lf,
            full_10k_text_features_lf,
            event_window_doc_batch_size=event_window_doc_batch_size,
            progress_callback=_event_screen_progress_callback,
        )
    )
    return _build_lm2011_event_panel_df_from_surface(event_screen_surface, ownership_lf).lazy()


def _attach_pre_filing_price_and_prior_month_price(event_panel_df: pl.DataFrame, daily_lf: pl.LazyFrame) -> pl.DataFrame:
    daily_schema = daily_lf.collect_schema()
    daily_permno_col = _resolve_first_existing(daily_schema, ("KYPERMNO", "kypermno"), "daily_panel")
    daily_date_col = _resolve_first_existing(daily_schema, ("CALDT", "daily_caldt"), "daily_panel")
    price_col = _resolve_first_existing(daily_schema, ("FINAL_PRC", "PRC"), "daily_panel")

    permnos = event_panel_df.get_column("KYPERMNO").drop_nulls().unique().to_list()
    if not permnos:
        return event_panel_df

    filing_dates = [
        _coerce_python_date(value, label="event_panel filing_date")
        for value in event_panel_df.get_column("filing_date").drop_nulls().to_list()
    ]
    pre_filing_trade_dates = [
        _coerce_python_date(value, label="event_panel pre_filing_trade_date")
        for value in event_panel_df.get_column("pre_filing_trade_date").drop_nulls().to_list()
    ]
    prior_month_end_max = max(
        _previous_month_end(value)
        for value in filing_dates
    )
    daily_lookup_end = max([prior_month_end_max, *pre_filing_trade_dates]) if pre_filing_trade_dates else prior_month_end_max
    daily_price = (
        daily_lf.filter(
            pl.col(daily_permno_col).cast(pl.Int32, strict=False).is_in(permnos)
            & (pl.col(daily_date_col).cast(pl.Date, strict=False) <= pl.lit(daily_lookup_end))
        )
        .select(
            pl.col(daily_permno_col).cast(pl.Int32, strict=False).alias("KYPERMNO"),
            pl.col(daily_date_col).cast(pl.Date, strict=False).alias("trade_date"),
            pl.col(price_col).cast(pl.Float64, strict=False).abs().alias("_price"),
        )
        .drop_nulls(subset=["KYPERMNO", "trade_date", "_price"])
        .collect()
        .sort("KYPERMNO", "trade_date")
    )

    event_schema = event_panel_df.schema
    out = event_panel_df.with_columns(
        pl.col("KYPERMNO").cast(pl.Int32, strict=False).alias("KYPERMNO"),
        pl.col("pre_filing_trade_date").cast(pl.Date, strict=False).alias("pre_filing_trade_date"),
        pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date"),
    )
    exact_pre_filing = daily_price.rename(
        {"trade_date": "pre_filing_trade_date", "_price": "_joined_pre_filing_price"}
    )
    out = out.join(exact_pre_filing, on=["KYPERMNO", "pre_filing_trade_date"], how="left")
    pre_filing_exprs = []
    pre_filing_exprs.append(pl.col("_joined_pre_filing_price").cast(pl.Float64, strict=False))
    if "pre_filing_price" in event_schema:
        pre_filing_exprs.append(pl.col("pre_filing_price").cast(pl.Float64, strict=False))
    out = out.with_columns(pl.coalesce(pre_filing_exprs).alias("pre_filing_price"))

    lookup = out.with_columns(
        pl.col("filing_date").map_elements(_previous_month_end, return_dtype=pl.Date).alias("_prior_month_end")
    ).sort("KYPERMNO", "_prior_month_end")
    prior_month = daily_price.rename({"_price": "prior_month_price"})
    return (
        lookup.join_asof(
            prior_month,
            left_on="_prior_month_end",
            right_on="trade_date",
            by="KYPERMNO",
            strategy="backward",
            check_sortedness=False,
        )
        .drop("_joined_pre_filing_price", "_prior_month_end", "trade_date", strict=False)
    )


def _prepare_lm2011_sue_docs_base_lf(
    event_panel_lf: pl.LazyFrame,
    quarterly_accounting_panel_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    _require_columns(
        event_panel_lf,
        (
            "doc_id",
            "gvkey_int",
            "KYPERMNO",
            "filing_date",
            "pre_filing_trade_date",
            "size_event",
            "bm_event",
            "share_turnover",
            "pre_ffalpha",
            "institutional_ownership",
            "nasdaq_dummy",
        ),
        "event_panel",
    )
    return (
        attach_eligible_quarterly_accounting(
            event_panel_lf.with_columns(pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("_lm2011_gvkey_int")),
            quarterly_accounting_panel_lf.with_columns(pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int")),
            filing_gvkey_col="_lm2011_gvkey_int",
        )
        .with_columns(
            pl.col("_lm2011_gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int"),
            pl.coalesce(
                [
                    pl.col("APDEDATEQ").cast(pl.Date, strict=False),
                    pl.col("PDATEQ").cast(pl.Date, strict=False),
                ]
            ).alias("_quarter_fiscal_period_end"),
        )
        .filter(pl.col("quarter_report_date").is_not_null())
    )


def _resolve_lm2011_sue_global_fiscal_window_policy(
    docs_df: pl.DataFrame,
) -> tuple[bool, dt.date | None, dt.date | None]:
    if docs_df.height == 0:
        return False, None, None
    fiscal_min = docs_df.select(pl.col("_quarter_fiscal_period_end").min()).item()
    fiscal_max = docs_df.select(pl.col("_quarter_fiscal_period_end").max()).item()
    require_exact_fiscal_window = bool(
        docs_df.select(pl.col("_quarter_fiscal_period_end").is_not_null().all()).item()
    )
    return require_exact_fiscal_window, fiscal_min, fiscal_max


def _build_lm2011_sue_panel_batch_df(
    docs_df: pl.DataFrame,
    *,
    ibes_unadjusted_earnings_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    global_require_exact_fiscal_window: bool,
    global_fiscal_min: dt.date | None,
    global_fiscal_max: dt.date | None,
) -> pl.DataFrame:
    if docs_df.height == 0:
        return _empty_sue_panel_df()

    relevant_gvkeys = [
        int(value)
        for value in docs_df.get_column("gvkey_int").drop_nulls().unique().to_list()
    ]
    if not relevant_gvkeys:
        return _empty_sue_panel_df()
    announcement_min = docs_df.select(pl.col("quarter_report_date").min()).item()
    announcement_max = docs_df.select(pl.col("quarter_report_date").max()).item()

    docs_df = _attach_pre_filing_price_and_prior_month_price(docs_df, daily_lf)
    _require_columns(
        ibes_unadjusted_earnings_lf,
        (
            "gvkey_int",
            "announcement_date",
            "fiscal_period_end",
            "actual_eps",
            "forecast_consensus_mean",
            "forecast_dispersion",
            "forecast_revision_4m",
        ),
        "ibes_unadjusted_earnings",
    )
    ibes = ibes_unadjusted_earnings_lf.select(
        pl.col("gvkey_int").cast(pl.Int32, strict=False),
        pl.col("announcement_date").cast(pl.Date, strict=False),
        pl.col("fiscal_period_end").cast(pl.Date, strict=False),
        pl.col("actual_eps").cast(pl.Float64, strict=False),
        pl.col("forecast_consensus_mean").cast(pl.Float64, strict=False),
        pl.col("forecast_dispersion").cast(pl.Float64, strict=False),
        pl.col("forecast_revision_4m").cast(pl.Float64, strict=False),
        (
            pl.col("forecast_revision_1m").cast(pl.Float64, strict=False)
            if "forecast_revision_1m" in ibes_unadjusted_earnings_lf.collect_schema()
            else pl.lit(None, dtype=pl.Float64)
        ).alias("forecast_revision_1m"),
    ).filter(
        pl.col("gvkey_int").cast(pl.Int32, strict=False).is_in(relevant_gvkeys)
        & pl.col("announcement_date").cast(pl.Date, strict=False).is_between(
            announcement_min,
            announcement_max,
            closed="both",
        )
        & (
            pl.col("fiscal_period_end").cast(pl.Date, strict=False).is_between(
                global_fiscal_min,
                global_fiscal_max,
                closed="both",
            )
            if global_require_exact_fiscal_window and global_fiscal_min is not None and global_fiscal_max is not None
            else pl.lit(True)
        )
    ).collect().unique(maintain_order=True)

    joined = select_refinitiv_lm2011_doc_analyst_inputs(
        docs_df.rename({"_quarter_fiscal_period_end": "quarter_fiscal_period_end"}),
        ibes,
    ).filter(pl.col("analyst_match_status") == "MATCHED")

    out = joined.with_columns(
        ((pl.col("actual_eps") - pl.col("forecast_consensus_mean")) / pl.col("pre_filing_price")).alias("sue"),
        (pl.col("forecast_dispersion") / pl.col("pre_filing_price")).alias("analyst_dispersion"),
        (pl.col("forecast_revision_4m") / pl.col("prior_month_price")).alias("analyst_revisions"),
    ).filter(
        pl.col("sue").is_not_null()
        & pl.col("analyst_dispersion").is_not_null()
        & pl.col("analyst_revisions").is_not_null()
    )
    if out.height == 0:
        return _empty_sue_panel_df()

    return out.select(
        "doc_id",
        "gvkey_int",
        "KYPERMNO",
        "filing_date",
        "quarter_report_date",
        "size_event",
        "bm_event",
        "share_turnover",
        "sue",
        "analyst_dispersion",
        "analyst_revisions",
        "pre_ffalpha",
        "institutional_ownership",
        "nasdaq_dummy",
    )


def build_lm2011_sue_panel(
    event_panel_lf: pl.LazyFrame,
    quarterly_accounting_panel_lf: pl.LazyFrame,
    ibes_unadjusted_earnings_lf: pl.LazyFrame | None,
    daily_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    if ibes_unadjusted_earnings_lf is None:
        raise ValueError(
            "ibes_unadjusted_earnings_lf is required; fail-closed external_input_policy forbids SUE panels without I/B/E/S"
        )
    docs_df = _prepare_lm2011_sue_docs_base_lf(
        event_panel_lf,
        quarterly_accounting_panel_lf,
    ).collect()
    global_require_exact_fiscal_window, global_fiscal_min, global_fiscal_max = (
        _resolve_lm2011_sue_global_fiscal_window_policy(docs_df)
    )
    return _build_lm2011_sue_panel_batch_df(
        docs_df,
        ibes_unadjusted_earnings_lf=ibes_unadjusted_earnings_lf,
        daily_lf=daily_lf,
        global_require_exact_fiscal_window=global_require_exact_fiscal_window,
        global_fiscal_min=global_fiscal_min,
        global_fiscal_max=global_fiscal_max,
    ).lazy()


def write_lm2011_sue_panel_parquet(
    event_panel_lf: pl.LazyFrame,
    quarterly_accounting_panel_lf: pl.LazyFrame,
    ibes_unadjusted_earnings_lf: pl.LazyFrame | None,
    daily_lf: pl.LazyFrame,
    *,
    output_path: Path,
    doc_batch_size: int = DEFAULT_EVENT_WINDOW_DOC_BATCH_SIZE,
    cleanup_on_success: bool = True,
) -> int:
    if ibes_unadjusted_earnings_lf is None:
        raise ValueError(
            "ibes_unadjusted_earnings_lf is required; fail-closed external_input_policy forbids SUE panels without I/B/E/S"
        )
    docs_base_lf = _prepare_lm2011_sue_docs_base_lf(
        event_panel_lf,
        quarterly_accounting_panel_lf,
    )
    # Materialize the joined doc-quarter base once here so the accounting/event graph
    # is reused across SUE shards instead of being recollected per batch.
    docs_df = docs_base_lf.collect().sort("quarter_report_date", "filing_date", "doc_id")
    global_require_exact_fiscal_window, global_fiscal_min, global_fiscal_max = (
        _resolve_lm2011_sue_global_fiscal_window_policy(docs_df)
    )
    docs_manifest = (
        docs_df.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
            pl.col("quarter_report_date").cast(pl.Date, strict=False).alias("quarter_report_date"),
            pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date"),
        )
        .sort("quarter_report_date", "filing_date", "doc_id")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_temp_root = output_path.parent / f".{output_path.stem}_tmp"
    if resolved_temp_root.exists():
        shutil.rmtree(resolved_temp_root)
    resolved_temp_root.mkdir(parents=True, exist_ok=True)
    shard_dir = resolved_temp_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    try:
        if docs_manifest.height == 0:
            _empty_sue_panel_df().write_parquet(output_path, compression=_STREAMING_PARQUET_COMPRESSION)
            if cleanup_on_success and resolved_temp_root.exists():
                shutil.rmtree(resolved_temp_root)
            return 0

        batch_size = _validated_event_window_doc_batch_size(doc_batch_size)
        shard_paths: list[Path] = []
        for batch_start in range(0, docs_manifest.height, batch_size):
            docs_batch = docs_df.slice(batch_start, batch_size)
            sue_batch = _build_lm2011_sue_panel_batch_df(
                docs_batch,
                ibes_unadjusted_earnings_lf=ibes_unadjusted_earnings_lf,
                daily_lf=daily_lf,
                global_require_exact_fiscal_window=global_require_exact_fiscal_window,
                global_fiscal_min=global_fiscal_min,
                global_fiscal_max=global_fiscal_max,
            )
            if not sue_batch.is_empty():
                shard_path = shard_dir / f"{int(batch_start / batch_size) + 1:06d}.parquet"
                sue_batch.write_parquet(shard_path, compression=_STREAMING_PARQUET_COMPRESSION)
                shard_paths.append(shard_path)
            del docs_batch
            del sue_batch
            gc.collect()

        if output_path.exists():
            output_path.unlink()
        if shard_paths:
            pl.scan_parquet([str(path) for path in shard_paths]).sink_parquet(
                output_path,
                compression=_STREAMING_PARQUET_COMPRESSION,
            )
        else:
            _empty_sue_panel_df().write_parquet(output_path, compression=_STREAMING_PARQUET_COMPRESSION)
        row_count = int(pl.scan_parquet(output_path).select(pl.len()).collect().item())
        if cleanup_on_success and resolved_temp_root.exists():
            shutil.rmtree(resolved_temp_root)
        return row_count
    except Exception:
        raise


def _prepare_monthly_stock_frame(
    monthly_stock_lf: pl.LazyFrame,
    *,
    monthly_return_col: str,
    portfolio_weighting: str,
    allowed_permnos: list[int] | None = None,
    month_start: dt.date | None = None,
    month_end: dt.date | None = None,
) -> pl.DataFrame:
    monthly_schema = monthly_stock_lf.collect_schema()
    permno_col = _resolve_first_existing(monthly_schema, ("KYPERMNO", "kypermno"), "monthly_stock")
    month_col = _resolve_first_existing(monthly_schema, ("MCALDT", "month_end", "portfolio_month"), "monthly_stock")
    _require_columns(monthly_stock_lf, (permno_col, month_col, monthly_return_col), "monthly_stock")
    market_cap_col = None
    if portfolio_weighting == "lagged_value":
        market_cap_col = _resolve_first_existing(monthly_schema, ("MTCAP", "MTHCAP"), "monthly_stock")

    monthly_base_lf = monthly_stock_lf.select(
            pl.col(permno_col).cast(pl.Int32, strict=False).alias("KYPERMNO"),
            pl.col(month_col).cast(pl.Date, strict=False).alias("portfolio_month"),
            pl.col(monthly_return_col).cast(pl.Float64, strict=False).alias("monthly_return"),
            (
                pl.col(market_cap_col).cast(pl.Float64, strict=False).alias("_market_cap")
                if market_cap_col is not None
                else pl.lit(1.0, dtype=pl.Float64).alias("_market_cap")
            ),
        )
    if allowed_permnos:
        monthly_base_lf = monthly_base_lf.filter(pl.col("KYPERMNO").is_in(allowed_permnos))
    if month_start is not None:
        monthly_base_lf = monthly_base_lf.filter(pl.col("portfolio_month") >= pl.lit(month_start))
    if month_end is not None:
        monthly_base_lf = monthly_base_lf.filter(pl.col("portfolio_month") <= pl.lit(month_end))

    monthly_df = monthly_base_lf.drop_nulls(subset=["KYPERMNO", "portfolio_month"]).collect().sort(
        "KYPERMNO",
        "portfolio_month",
    )
    if portfolio_weighting == "lagged_value":
        monthly_df = monthly_df.with_columns(
            pl.col("_market_cap").shift(1).over("KYPERMNO").alias("portfolio_weight")
        )
    else:
        monthly_df = monthly_df.with_columns(pl.lit(1.0, dtype=pl.Float64).alias("portfolio_weight"))
    return monthly_df.drop("_market_cap")


def _prepare_monthly_factor_frame(ff_factors_monthly_with_mom_lf: pl.LazyFrame | None) -> pl.DataFrame:
    if ff_factors_monthly_with_mom_lf is None:
        raise ValueError(
            "ff_factors_monthly_with_mom_lf is required; fail-closed external_input_policy forbids LM2011 trading strategy panels without monthly factors"
        )
    _require_columns(
        ff_factors_monthly_with_mom_lf,
        ("month_end", "mkt_rf", "smb", "hml", "rf", "mom"),
        "ff_factors_monthly_with_mom",
    )
    factors_df = ff_factors_monthly_with_mom_lf.select(
        pl.col("month_end").cast(pl.Date, strict=False).alias("portfolio_month"),
        pl.col("mkt_rf").cast(pl.Float64, strict=False).alias("mkt_rf"),
        pl.col("smb").cast(pl.Float64, strict=False).alias("smb"),
        pl.col("hml").cast(pl.Float64, strict=False).alias("hml"),
        pl.col("rf").cast(pl.Float64, strict=False).alias("rf"),
        pl.col("mom").cast(pl.Float64, strict=False).alias("mom"),
    ).collect()
    return _ensure_factor_scale(factors_df, ("mkt_rf", "smb", "hml", "rf", "mom"))


def _build_strategy_assignment_frame(strategy_docs_df: pl.DataFrame) -> pl.DataFrame:
    signal_columns = [
        signal_name
        for signal_name in _STRATEGY_SIGNAL_COLUMNS
        if signal_name in strategy_docs_df.columns
    ]
    empty = pl.DataFrame(
        schema={
            "doc_id": pl.Utf8,
            "KYPERMNO": pl.Int32,
            "sort_year": pl.Int32,
            "sort_signal_name": pl.Utf8,
            "signal_value": pl.Float64,
            "quintile": pl.Int8,
        }
    )
    if not signal_columns or strategy_docs_df.is_empty():
        return empty

    base = (
        strategy_docs_df.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("KYPERMNO").cast(pl.Int32, strict=False),
            pl.col("filing_date").cast(pl.Date, strict=False),
            *[pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in signal_columns],
        )
        .drop_nulls(subset=["doc_id", "KYPERMNO", "filing_date"])
        .with_columns((pl.col("filing_date").dt.year() + 1).cast(pl.Int32).alias("sort_year"))
        .filter(
            pl.col("sort_year").is_between(_STRATEGY_MIN_SORT_YEAR, _STRATEGY_MAX_SORT_YEAR, closed="both")
        )
        .unpivot(
            index=["doc_id", "KYPERMNO", "sort_year"],
            on=signal_columns,
            variable_name="sort_signal_name",
            value_name="signal_value",
        )
        .drop_nulls(subset=["signal_value"])
        .sort("sort_year", "sort_signal_name", "signal_value", "doc_id")
    )
    if base.is_empty():
        return empty
    group_keys = ["sort_year", "sort_signal_name"]
    return (
        base.with_row_index("_global_rank")
        .with_columns(
            (pl.col("_global_rank") - pl.col("_global_rank").min().over(group_keys))
            .cast(pl.Int64)
            .alias("_group_rank"),
            pl.len().over(group_keys).cast(pl.Int64).alias("_group_size"),
        )
        .with_columns(
            (
                ((pl.col("_group_rank") * pl.lit(5.0)) / pl.col("_group_size"))
                .floor()
                .cast(pl.Int8)
                + pl.lit(1, dtype=pl.Int8)
            ).alias("quintile")
        )
        .select("doc_id", "KYPERMNO", "sort_year", "sort_signal_name", "signal_value", "quintile")
    )


def _strategy_sort_year_expr() -> pl.Expr:
    return (
        pl.when(pl.col("portfolio_month").dt.month() >= 7)
        .then(pl.col("portfolio_month").dt.year())
        .otherwise(pl.col("portfolio_month").dt.year() - 1)
        .cast(pl.Int32)
        .alias("sort_year")
    )


def _compute_long_short_returns(holdings_df: pl.DataFrame, *, portfolio_weighting: str) -> pl.DataFrame:
    if holdings_df.height == 0:
        return pl.DataFrame(
            schema={
                "portfolio_month": pl.Date,
                "sort_signal_name": pl.Utf8,
                "long_short_return": pl.Float64,
            }
        )
    if portfolio_weighting == "equal":
        quintile_returns = (
            holdings_df.filter(pl.col("monthly_return").is_not_null())
            .group_by("portfolio_month", "sort_signal_name", "quintile")
            .agg(pl.col("monthly_return").mean().alias("_quintile_return"))
        )
    else:
        quintile_returns = (
            holdings_df.filter(
                pl.col("monthly_return").is_not_null()
                & pl.col("portfolio_weight").is_not_null()
                & (pl.col("portfolio_weight") > 0)
            )
            .group_by("portfolio_month", "sort_signal_name", "quintile")
            .agg(
                (pl.col("monthly_return") * pl.col("portfolio_weight")).sum().alias("_weighted_return_sum"),
                pl.col("portfolio_weight").sum().alias("_weight_sum"),
            )
            .with_columns(
                pl.when(pl.col("_weight_sum") > 0)
                .then(pl.col("_weighted_return_sum") / pl.col("_weight_sum"))
                .otherwise(None)
                .alias("_quintile_return")
            )
            .select("portfolio_month", "sort_signal_name", "quintile", "_quintile_return")
        )
    return (
        quintile_returns.group_by("portfolio_month", "sort_signal_name")
        .agg(
            pl.when(pl.col("quintile") == 1).then(pl.col("_quintile_return")).otherwise(None).drop_nulls().first().alias("_q1_return"),
            pl.when(pl.col("quintile") == 5).then(pl.col("_quintile_return")).otherwise(None).drop_nulls().first().alias("_q5_return"),
        )
        .with_columns((pl.col("_q1_return") - pl.col("_q5_return")).alias("long_short_return"))
        .filter(pl.col("long_short_return").is_not_null())
        .select("portfolio_month", "sort_signal_name", "long_short_return")
    )


def _ols_coefficients_and_r2(
    df: pl.DataFrame,
    *,
    y_col: str,
    x_cols: tuple[str, ...],
) -> tuple[tuple[float | None, ...], float | None]:
    nonnull_predictors = [pl.col(column).is_not_null() for column in x_cols]
    subset = df.filter(
        pl.col(y_col).is_not_null()
        & pl.fold(
            acc=pl.lit(True),
            function=lambda acc, expr: acc & expr,
            exprs=nonnull_predictors,
        )
    )
    n_obs = subset.height
    size = len(x_cols) + 1
    if n_obs <= len(x_cols):
        return tuple(None for _ in range(size)), None
    results = _fit_checked_ols(
        subset.get_column(y_col).cast(pl.Float64, strict=False).to_numpy(),
        subset.select(*[pl.col(column).cast(pl.Float64, strict=False) for column in x_cols]).to_numpy(),
        exog_names=x_cols,
        label=f"FF4 regression for dependent={y_col}",
    )
    r2_value = float(results.rsquared)
    r2 = r2_value if math.isfinite(r2_value) else None
    return tuple(float(value) for value in results.params), r2


def _fit_strategy_factor_loadings(strategy_df: pl.DataFrame) -> pl.DataFrame:
    if strategy_df.height == 0:
        return _empty_trading_strategy_ff4_summary_df()
    rows: list[dict[str, object]] = []
    for group in strategy_df.partition_by("sort_signal_name", maintain_order=True):
        signal_name = group.item(0, "sort_signal_name")
        coefficients, r2 = _ols_coefficients_and_r2(
            group,
            y_col="long_short_return",
            x_cols=("mkt_rf", "smb", "hml", "mom"),
        )
        alpha, beta_market, beta_smb, beta_hml, beta_mom = coefficients
        rows.append(
            {
                "sort_signal_name": signal_name,
                "alpha_ff3_mom": alpha,
                "beta_market": beta_market,
                "beta_smb": beta_smb,
                "beta_hml": beta_hml,
                "beta_mom": beta_mom,
                "r2": r2,
            }
        )
    return pl.DataFrame(rows, schema_overrides=_empty_trading_strategy_ff4_summary_df().schema)


def _build_trading_strategy_monthly_returns_df(
    event_panel_lf: pl.LazyFrame,
    sec_parsed_lf: pl.LazyFrame,
    monthly_stock_lf: pl.LazyFrame,
    *,
    lm_dictionary_lists: Mapping[str, Iterable[str]],
    harvard_negative_word_list: Iterable[str] | None,
    master_dictionary_words: Iterable[str],
    portfolio_weighting: str = "equal",
    monthly_return_col: str = "MRET",
    cleaning_contract: Full10KCleaningContract = "current",
) -> pl.DataFrame:
    if portfolio_weighting not in {"equal", "lagged_value"}:
        raise ValueError("portfolio_weighting must be one of {'equal', 'lagged_value'}")
    _require_columns(event_panel_lf, ("doc_id", "KYPERMNO", "filing_date"), "event_panel")

    docs_df = (
        event_panel_lf.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("KYPERMNO").cast(pl.Int32, strict=False),
            pl.col("filing_date").cast(pl.Date, strict=False),
        )
        .drop_nulls(subset=["doc_id", "KYPERMNO", "filing_date"])
        .collect()
        .unique(subset=["doc_id"], keep="first")
    )
    if docs_df.height == 0:
        return _empty_trading_strategy_monthly_returns_df()

    signal_df = build_lm2011_trading_strategy_signal_frame(
        sec_parsed_lf.join(docs_df.lazy().select("doc_id"), on="doc_id", how="semi"),
        lm_dictionary_lists=lm_dictionary_lists,
        harvard_negative_word_list=harvard_negative_word_list,
        master_dictionary_words=master_dictionary_words,
        cleaning_contract=cleaning_contract,
    ).collect()
    strategy_docs_df = docs_df.join(signal_df, on="doc_id", how="inner")
    assignments = _build_strategy_assignment_frame(strategy_docs_df)
    if assignments.height == 0:
        return _empty_trading_strategy_monthly_returns_df()

    allowed_permnos = [int(value) for value in assignments.get_column("KYPERMNO").unique().to_list()]
    monthly_start = _STRATEGY_START_DATE
    if portfolio_weighting == "lagged_value":
        monthly_start = _STRATEGY_START_DATE.replace(day=1) - dt.timedelta(days=1)
    monthly_df = _prepare_monthly_stock_frame(
        monthly_stock_lf,
        monthly_return_col=monthly_return_col,
        portfolio_weighting=portfolio_weighting,
        allowed_permnos=allowed_permnos,
        month_start=monthly_start,
        month_end=_STRATEGY_END_DATE,
    ).filter(
        pl.col("portfolio_month").is_between(pl.lit(_STRATEGY_START_DATE), pl.lit(_STRATEGY_END_DATE), closed="both")
    )
    monthly_df = monthly_df.with_columns(_strategy_sort_year_expr()).filter(
        pl.col("sort_year").is_between(_STRATEGY_MIN_SORT_YEAR, _STRATEGY_MAX_SORT_YEAR, closed="both")
    )

    holdings_df = monthly_df.join(
        assignments.select("KYPERMNO", "sort_year", "sort_signal_name", "quintile"),
        on=["KYPERMNO", "sort_year"],
        how="inner",
    )
    long_short_df = _compute_long_short_returns(holdings_df, portfolio_weighting=portfolio_weighting)
    if long_short_df.height == 0:
        return _empty_trading_strategy_monthly_returns_df()
    return long_short_df.sort("portfolio_month", "sort_signal_name")


def build_lm2011_trading_strategy_monthly_returns(
    event_panel_lf: pl.LazyFrame,
    sec_parsed_lf: pl.LazyFrame,
    monthly_stock_lf: pl.LazyFrame,
    *,
    lm_dictionary_lists: Mapping[str, Iterable[str]],
    harvard_negative_word_list: Iterable[str] | None,
    master_dictionary_words: Iterable[str],
    portfolio_weighting: str = "equal",
    monthly_return_col: str = "MRET",
    cleaning_contract: Full10KCleaningContract = "current",
) -> pl.LazyFrame:
    return _build_trading_strategy_monthly_returns_df(
        event_panel_lf,
        sec_parsed_lf,
        monthly_stock_lf,
        lm_dictionary_lists=lm_dictionary_lists,
        harvard_negative_word_list=harvard_negative_word_list,
        master_dictionary_words=master_dictionary_words,
        portfolio_weighting=portfolio_weighting,
        monthly_return_col=monthly_return_col,
        cleaning_contract=cleaning_contract,
    ).lazy()


def build_lm2011_trading_strategy_ff4_summary(
    trading_strategy_monthly_returns_lf: pl.LazyFrame,
    ff_factors_monthly_with_mom_lf: pl.LazyFrame | None,
) -> pl.LazyFrame:
    _require_columns(
        trading_strategy_monthly_returns_lf,
        ("portfolio_month", "sort_signal_name", "long_short_return"),
        "trading_strategy_monthly_returns",
    )
    monthly_returns_df = (
        trading_strategy_monthly_returns_lf.select(
            pl.col("portfolio_month").cast(pl.Date, strict=False),
            pl.col("sort_signal_name").cast(pl.Utf8, strict=False),
            pl.col("long_short_return").cast(pl.Float64, strict=False),
        )
        .collect()
        .sort("portfolio_month", "sort_signal_name")
    )
    if monthly_returns_df.height == 0:
        return _empty_trading_strategy_ff4_summary_df().lazy()

    factors_df = _prepare_monthly_factor_frame(ff_factors_monthly_with_mom_lf).filter(
        pl.col("portfolio_month").is_between(pl.lit(_STRATEGY_START_DATE), pl.lit(_STRATEGY_END_DATE), closed="both")
    )
    strategy_df = (
        monthly_returns_df.join(factors_df, on="portfolio_month", how="inner")
        .filter(
            pl.col("mkt_rf").is_not_null()
            & pl.col("smb").is_not_null()
            & pl.col("hml").is_not_null()
            & pl.col("mom").is_not_null()
        )
        .sort("portfolio_month", "sort_signal_name")
    )
    if strategy_df.height == 0:
        return _empty_trading_strategy_ff4_summary_df().lazy()
    return _fit_strategy_factor_loadings(strategy_df).lazy()


__all__ = [
    "build_lm2011_normalized_filing_feeds",
    "build_lm2011_sample_backbone",
    "build_lm2011_text_features_full_10k",
    "build_lm2011_text_features_mda",
    "build_lm2011_table_i_sample_creation",
    "build_lm2011_event_panel",
    "build_lm2011_sue_panel",
    "write_lm2011_sue_panel_parquet",
    "build_lm2011_trading_strategy_monthly_returns",
    "build_lm2011_trading_strategy_ff4_summary",
]
