from __future__ import annotations

from collections.abc import Iterable, Mapping
import datetime as dt
import math
from typing import Any

import polars as pl

from thesis_pkg.core.ccm.lm2011 import (
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
    _require_columns(ownership_lf, ("doc_id", value_col), "ownership")
    ownership_df = (
        ownership_lf.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col(value_col).cast(pl.Float64, strict=False).alias("institutional_ownership"),
        )
        .unique(subset=["doc_id", "institutional_ownership"], maintain_order=True)
        .collect()
    )
    duplicate_doc_ids = (
        ownership_df.group_by("doc_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("doc_id")
        .to_list()
    )
    if duplicate_doc_ids:
        raise ValueError(f"ownership_lf must be unique on doc_id after exact duplicate removal: {duplicate_doc_ids[:10]}")
    return ownership_df


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


def _prepare_event_base_frame(
    sample_backbone_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    annual_accounting_panel_lf: pl.LazyFrame,
    ownership_lf: pl.LazyFrame | None,
    full_10k_text_features_lf: pl.LazyFrame | None,
) -> pl.DataFrame:
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
    ownership_df = _prepare_ownership_frame(ownership_lf)
    text_df = (
        full_10k_text_features_lf.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("token_count_full_10k").cast(pl.Int32, strict=False),
        )
        .collect()
    )
    return (
        with_pre_market
        .join(ownership_df.lazy(), on="doc_id", how="left")
        .join(text_df.lazy(), on="doc_id", how="left")
        .with_columns(
            _price_expr_from_available_columns(
                with_pre_market_schema,
                candidates=("FINAL_PRC", "PRC"),
                label="event base frame",
            ).alias("pre_filing_price"),
            pl.col("KYPERMNO").cast(pl.Int32, strict=False),
            pl.col("gvkey_int").cast(pl.Int32, strict=False),
        )
        .collect()
    )


def _prepare_daily_event_frame(
    daily_lf: pl.LazyFrame,
    ff_factors_daily_lf: pl.LazyFrame | None,
    docs_df: pl.DataFrame,
) -> pl.DataFrame:
    if ff_factors_daily_lf is None:
        raise ValueError(
            "ff_factors_daily_lf is required; fail-closed external_input_policy forbids LM2011 event panels without daily FF factors"
        )
    daily_schema = daily_lf.collect_schema()
    daily_permno_col = _resolve_first_existing(daily_schema, ("KYPERMNO", "kypermno"), "daily_panel")
    daily_date_col = _resolve_first_existing(daily_schema, ("CALDT", "daily_caldt"), "daily_panel")
    return_col = "FINAL_RET" if "FINAL_RET" in daily_schema else "RET"
    price_col = "FINAL_PRC" if "FINAL_PRC" in daily_schema else "PRC"
    required_daily = [daily_permno_col, daily_date_col, return_col, "VOL", price_col, "SHROUT", "SHRCD", "EXCHCD"]
    missing_daily = [name for name in required_daily if name not in daily_schema]
    if missing_daily:
        raise ValueError(f"daily_lf missing required columns for LM2011 event metrics: {missing_daily}")
    _require_columns(ff_factors_daily_lf, ("trading_date", "mkt_rf", "smb", "hml", "rf"), "ff_factors_daily")

    permnos = docs_df.get_column("KYPERMNO").drop_nulls().unique().to_list()
    if not permnos:
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

    filing_trade_dates = docs_df.get_column("filing_trade_date").drop_nulls()
    if filing_trade_dates.len() == 0:
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
    min_trade_date = _coerce_python_date(filing_trade_dates.min(), label="minimum filing_trade_date")
    max_trade_date = _coerce_python_date(filing_trade_dates.max(), label="maximum filing_trade_date")
    min_lookup_date = min_trade_date - dt.timedelta(days=450)
    max_lookup_date = max_trade_date + dt.timedelta(days=450)

    daily_df = (
        daily_lf.filter(
            pl.col(daily_permno_col).cast(pl.Int32, strict=False).is_in(permnos)
            & pl.col(daily_date_col).cast(pl.Date, strict=False).is_between(
                pl.lit(min_lookup_date),
                pl.lit(max_lookup_date),
                closed="both",
            )
        )
        .select(
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
        )
        .collect()
    )

    factors_df = ff_factors_daily_lf.select(
        pl.col("trading_date").cast(pl.Date, strict=False).alias("trade_date"),
        pl.col("mkt_rf").cast(pl.Float64, strict=False).alias("mkt_rf"),
        pl.col("smb").cast(pl.Float64, strict=False).alias("smb"),
        pl.col("hml").cast(pl.Float64, strict=False).alias("hml"),
        pl.col("rf").cast(pl.Float64, strict=False).alias("rf"),
    ).collect()
    factors_df = _ensure_factor_scale(factors_df, ("mkt_rf", "smb", "hml", "rf"))
    factors_df = factors_df.with_columns((pl.col("mkt_rf") + pl.col("rf")).alias("market_return"))
    return daily_df.join(factors_df, on="trade_date", how="left")


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
    )


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    size = len(vector)
    aug = [row[:] + [vector[idx]] for idx, row in enumerate(matrix)]
    for pivot in range(size):
        pivot_row = max(range(pivot, size), key=lambda idx: abs(aug[idx][pivot]))
        if abs(aug[pivot_row][pivot]) < 1e-12:
            return None
        if pivot_row != pivot:
            aug[pivot], aug[pivot_row] = aug[pivot_row], aug[pivot]
        pivot_value = aug[pivot][pivot]
        aug[pivot] = [value / pivot_value for value in aug[pivot]]
        for row_idx in range(size):
            if row_idx == pivot:
                continue
            factor = aug[row_idx][pivot]
            if factor == 0:
                continue
            aug[row_idx] = [
                aug[row_idx][col_idx] - factor * aug[pivot][col_idx]
                for col_idx in range(size + 1)
            ]
    return [aug[idx][-1] for idx in range(size)]


def _ols_alpha_and_rmse(summary_row: dict[str, Any]) -> tuple[float | None, float | None]:
    n = int(summary_row["n_obs"] or 0)
    if n <= 4:
        return None, None
    xtx = [
        [float(summary_row["n_obs"]), float(summary_row["sx1"]), float(summary_row["sx2"]), float(summary_row["sx3"])],
        [float(summary_row["sx1"]), float(summary_row["sxx11"]), float(summary_row["sxx12"]), float(summary_row["sxx13"])],
        [float(summary_row["sx2"]), float(summary_row["sxx12"]), float(summary_row["sxx22"]), float(summary_row["sxx23"])],
        [float(summary_row["sx3"]), float(summary_row["sxx13"]), float(summary_row["sxx23"]), float(summary_row["sxx33"])],
    ]
    xty = [
        float(summary_row["sy"]),
        float(summary_row["sxy1"]),
        float(summary_row["sxy2"]),
        float(summary_row["sxy3"]),
    ]
    beta = _solve_linear_system(xtx, xty)
    if beta is None:
        return None, None
    syy = float(summary_row["syy"])
    sse = syy - sum(beta[idx] * xty[idx] for idx in range(len(beta)))
    rmse = math.sqrt(max(sse / float(n - 4), 0.0))
    return float(beta[0]), float(rmse)


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
    summary = subset.group_by("doc_id").agg(
        pl.len().cast(pl.Int32).alias("n_obs"),
        pl.col("_y").sum().alias("sy"),
        pl.col("_y").pow(2).sum().alias("syy"),
        pl.col("_x1").sum().alias("sx1"),
        pl.col("_x2").sum().alias("sx2"),
        pl.col("_x3").sum().alias("sx3"),
        (pl.col("_x1") * pl.col("_y")).sum().alias("sxy1"),
        (pl.col("_x2") * pl.col("_y")).sum().alias("sxy2"),
        (pl.col("_x3") * pl.col("_y")).sum().alias("sxy3"),
        pl.col("_x1").pow(2).sum().alias("sxx11"),
        (pl.col("_x1") * pl.col("_x2")).sum().alias("sxx12"),
        (pl.col("_x1") * pl.col("_x3")).sum().alias("sxx13"),
        pl.col("_x2").pow(2).sum().alias("sxx22"),
        (pl.col("_x2") * pl.col("_x3")).sum().alias("sxx23"),
        pl.col("_x3").pow(2).sum().alias("sxx33"),
    )
    rows: list[dict[str, object]] = []
    for row in summary.iter_rows(named=True):
        alpha, rmse = _ols_alpha_and_rmse(row)
        rows.append(
            {
                "doc_id": row["doc_id"],
                alpha_name: alpha,
                rmse_name: rmse,
                "n_obs": row["n_obs"],
            }
        )
    return pl.DataFrame(rows)


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


def build_lm2011_event_panel(
    sample_backbone_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    annual_accounting_panel_lf: pl.LazyFrame,
    ff_factors_daily_lf: pl.LazyFrame | None,
    ownership_lf: pl.LazyFrame | None,
    full_10k_text_features_lf: pl.LazyFrame | None,
) -> pl.LazyFrame:
    if ff_factors_daily_lf is None:
        raise ValueError(
            "ff_factors_daily_lf is required; fail-closed external_input_policy forbids LM2011 event panels without daily FF factors"
        )
    docs_df = _prepare_event_base_frame(
        sample_backbone_lf,
        daily_lf,
        annual_accounting_panel_lf.with_columns(pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int")),
        ownership_lf,
        full_10k_text_features_lf,
    )
    if docs_df.height == 0:
        return _empty_event_panel_df().lazy()

    daily_df = _prepare_daily_event_frame(daily_lf, ff_factors_daily_lf, docs_df)
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

    panel = (
        docs_df.join(event_summary, on="doc_id", how="left")
        .join(abnormal_event, on="doc_id", how="left")
        .join(pre_alpha, on="doc_id", how="left")
        .join(post_alpha, on="doc_id", how="left")
        .with_columns(
            (pl.col("_event_stock_gross") - pl.col("_event_market_gross")).alias("filing_period_excess_return"),
            (
                pl.when(pl.col("event_shares").is_not_null() & (pl.col("event_shares") > 0))
                .then(pl.col("_turnover_volume_sum") / pl.col("event_shares"))
                .otherwise(None)
            ).alias("share_turnover"),
            pl.when(pl.col("event_exchcd") == 3)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("nasdaq_dummy"),
        )
    )

    panel = panel.filter(pl.col("event_shrcd").is_in([10, 11]))
    panel = panel.filter(pl.col("size_event").cast(pl.Float64, strict=False).is_not_null() & (pl.col("size_event") > 0))
    panel = panel.filter(pl.col("pre_filing_price").is_not_null() & (pl.col("pre_filing_price") >= 3.0))
    panel = panel.filter((pl.col("event_return_day_count") == 4) & (pl.col("event_volume_day_count") == 4))
    panel = panel.filter(pl.col("event_exchcd").is_in([1, 2, 3]))
    panel = panel.filter(
        (pl.col("pre_turnover_obs") >= 60)
        & (pl.col("abnormal_volume_pre_obs") >= 60)
        & (pl.col("pre_alpha_obs") >= 60)
        & (pl.col("post_alpha_obs") >= 60)
    )
    panel = panel.filter(pl.col("book_equity_be").cast(pl.Float64, strict=False) > 0)
    panel = panel.filter(pl.col("bm_event").cast(pl.Float64, strict=False) > 0)
    panel = _winsorize_column(panel, "bm_event")
    panel = panel.filter(pl.col("token_count_full_10k").cast(pl.Int32, strict=False) >= 2000)
    panel = panel.filter(pl.col("abnormal_volume").is_not_null())

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
    ).lazy()


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
    if "pre_filing_price" in event_schema:
        pre_filing_exprs.append(pl.col("pre_filing_price").cast(pl.Float64, strict=False))
    pre_filing_exprs.append(pl.col("_joined_pre_filing_price").cast(pl.Float64, strict=False))
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
    docs_df = (
        attach_eligible_quarterly_accounting(
            event_panel_lf.with_columns(pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("_lm2011_gvkey_int")),
            quarterly_accounting_panel_lf.with_columns(pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int")),
            filing_gvkey_col="_lm2011_gvkey_int",
        )
        .with_columns(pl.col("_lm2011_gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int"))
        .collect()
    )
    docs_df = docs_df.filter(pl.col("quarter_report_date").is_not_null())
    if docs_df.height == 0:
        return _empty_sue_panel_df().lazy()

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
    ).collect().unique(maintain_order=True)

    docs_df = docs_df.with_columns(
        pl.coalesce(
            [
                pl.col("APDEDATEQ").cast(pl.Date, strict=False),
                pl.col("PDATEQ").cast(pl.Date, strict=False),
            ]
        ).alias("_quarter_fiscal_period_end")
    )
    ibes_value_cols = [
        "actual_eps",
        "forecast_consensus_mean",
        "forecast_dispersion",
        "forecast_revision_4m",
    ]
    exact_join = docs_df.join(
        ibes,
        left_on=["gvkey_int", "quarter_report_date", "_quarter_fiscal_period_end"],
        right_on=["gvkey_int", "announcement_date", "fiscal_period_end"],
        how="left",
    )
    exact_hits = exact_join.filter(pl.col("actual_eps").is_not_null()).select(
        *docs_df.columns,
        *[pl.col(column) for column in ibes_value_cols],
    )
    fallback_docs = exact_join.filter(pl.col("actual_eps").is_null()).select(docs_df.columns)

    announcement_unique = (
        ibes.group_by("gvkey_int", "announcement_date")
        .len()
        .filter(pl.col("len") == 1)
        .drop("len")
    )
    fallback_ibes = ibes.join(announcement_unique, on=["gvkey_int", "announcement_date"], how="inner")
    fallback_hits = fallback_docs.join(
        fallback_ibes,
        left_on=["gvkey_int", "quarter_report_date"],
        right_on=["gvkey_int", "announcement_date"],
        how="left",
    ).filter(pl.col("actual_eps").is_not_null()).select(
        *docs_df.columns,
        *[pl.col(column) for column in ibes_value_cols],
    )
    joined = (
        pl.concat([exact_hits, fallback_hits], how="vertical_relaxed")
        if exact_hits.height or fallback_hits.height
        else pl.DataFrame(
            schema={
                **dict(docs_df.schema),
                **{column: pl.Float64 for column in ibes_value_cols},
            }
        )
    )

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
        return _empty_sue_panel_df().lazy()

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
    ).lazy()


def _prepare_monthly_stock_frame(
    monthly_stock_lf: pl.LazyFrame,
    *,
    monthly_return_col: str,
    portfolio_weighting: str,
) -> pl.DataFrame:
    monthly_schema = monthly_stock_lf.collect_schema()
    permno_col = _resolve_first_existing(monthly_schema, ("KYPERMNO", "kypermno"), "monthly_stock")
    month_col = _resolve_first_existing(monthly_schema, ("MCALDT", "month_end", "portfolio_month"), "monthly_stock")
    _require_columns(monthly_stock_lf, (permno_col, month_col, monthly_return_col), "monthly_stock")
    market_cap_col = None
    if portfolio_weighting == "lagged_value":
        market_cap_col = _resolve_first_existing(monthly_schema, ("MTCAP", "MTHCAP"), "monthly_stock")

    monthly_df = (
        monthly_stock_lf.select(
            pl.col(permno_col).cast(pl.Int32, strict=False).alias("KYPERMNO"),
            pl.col(month_col).cast(pl.Date, strict=False).alias("portfolio_month"),
            pl.col(monthly_return_col).cast(pl.Float64, strict=False).alias("monthly_return"),
            (
                pl.col(market_cap_col).cast(pl.Float64, strict=False).alias("_market_cap")
                if market_cap_col is not None
                else pl.lit(1.0, dtype=pl.Float64).alias("_market_cap")
            ),
        )
        .drop_nulls(subset=["KYPERMNO", "portfolio_month"])
        .collect()
        .sort("KYPERMNO", "portfolio_month")
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
    rows: list[dict[str, object]] = []
    for row in strategy_docs_df.iter_rows(named=True):
        filing_date = row["filing_date"]
        if filing_date is None:
            continue
        filing_date = _coerce_python_date(filing_date, label="strategy filing_date")
        sort_year = filing_date.year + 1
        if sort_year < _STRATEGY_MIN_SORT_YEAR or sort_year > _STRATEGY_MAX_SORT_YEAR:
            continue
        for signal_name in _STRATEGY_SIGNAL_COLUMNS:
            signal_value = row.get(signal_name)
            if signal_value is None:
                continue
            rows.append(
                {
                    "doc_id": str(row["doc_id"]),
                    "KYPERMNO": int(row["KYPERMNO"]),
                    "sort_year": sort_year,
                    "sort_signal_name": signal_name,
                    "signal_value": float(signal_value),
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "doc_id": pl.Utf8,
                "KYPERMNO": pl.Int32,
                "sort_year": pl.Int32,
                "sort_signal_name": pl.Utf8,
                "signal_value": pl.Float64,
                "quintile": pl.Int8,
            }
        )
    base = pl.DataFrame(rows).sort("sort_year", "sort_signal_name", "signal_value", "doc_id")
    groups: list[pl.DataFrame] = []
    for group in base.partition_by(["sort_year", "sort_signal_name"], maintain_order=True):
        n_obs = group.height
        if n_obs == 0:
            continue
        quintiles = [int(idx * 5 / n_obs) + 1 for idx in range(n_obs)]
        groups.append(group.with_columns(pl.Series("quintile", quintiles, dtype=pl.Int8)))
    return pl.concat(groups, how="vertical_relaxed") if groups else base


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

    xtx = [[0.0 for _ in range(size)] for _ in range(size)]
    xty = [0.0 for _ in range(size)]
    cached_rows: list[dict[str, float]] = []
    for row in subset.select(y_col, *x_cols).iter_rows(named=True):
        typed_row = {column: float(row[column]) for column in (y_col, *x_cols)}
        cached_rows.append(typed_row)
        xs = [1.0] + [typed_row[column] for column in x_cols]
        y_value = typed_row[y_col]
        for idx in range(size):
            xty[idx] += xs[idx] * y_value
            for jdx in range(size):
                xtx[idx][jdx] += xs[idx] * xs[jdx]
    beta = _solve_linear_system(xtx, xty)
    if beta is None:
        return tuple(None for _ in range(size)), None

    y_mean = sum(row[y_col] for row in cached_rows) / float(n_obs)
    sse = 0.0
    sst = 0.0
    for row in cached_rows:
        fitted = beta[0] + sum(beta[idx + 1] * row[column] for idx, column in enumerate(x_cols))
        residual = row[y_col] - fitted
        centered = row[y_col] - y_mean
        sse += residual * residual
        sst += centered * centered
    r2 = None if sst <= 0 else float(1.0 - (sse / sst))
    return tuple(float(value) for value in beta), r2


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
    portfolio_weighting: str = "equal",
    monthly_return_col: str = "MRET",
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
    ).collect()
    strategy_docs_df = docs_df.join(signal_df, on="doc_id", how="inner")
    assignments = _build_strategy_assignment_frame(strategy_docs_df)
    if assignments.height == 0:
        return _empty_trading_strategy_monthly_returns_df()

    monthly_df = _prepare_monthly_stock_frame(
        monthly_stock_lf,
        monthly_return_col=monthly_return_col,
        portfolio_weighting=portfolio_weighting,
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
    portfolio_weighting: str = "equal",
    monthly_return_col: str = "MRET",
) -> pl.LazyFrame:
    return _build_trading_strategy_monthly_returns_df(
        event_panel_lf,
        sec_parsed_lf,
        monthly_stock_lf,
        lm_dictionary_lists=lm_dictionary_lists,
        harvard_negative_word_list=harvard_negative_word_list,
        portfolio_weighting=portfolio_weighting,
        monthly_return_col=monthly_return_col,
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
    "build_lm2011_event_panel",
    "build_lm2011_sue_panel",
    "build_lm2011_trading_strategy_monthly_returns",
    "build_lm2011_trading_strategy_ff4_summary",
]
