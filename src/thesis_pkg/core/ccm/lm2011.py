from __future__ import annotations

import polars as pl


GVKEY_DTYPE = pl.Int32
_MIN_VALID_YEAR = 1900
_MAX_VALID_YEAR = 2100


def _require_columns(lf: pl.LazyFrame, required: tuple[str, ...], label: str) -> None:
    """Fail fast when a helper boundary is missing expected columns."""
    schema = lf.collect_schema()
    missing = [name for name in required if name not in schema]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _resolve_first_existing(schema: pl.Schema, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in schema:
            return candidate
    raise ValueError(f"{label} missing any of expected columns: {list(candidates)}")


def _float_expr(col_name: str) -> pl.Expr:
    return pl.col(col_name).cast(pl.Float64, strict=False)


def _optional_float_expr(schema: pl.Schema, col_name: str) -> pl.Expr:
    if col_name in schema:
        return _float_expr(col_name)
    return pl.lit(0.0, dtype=pl.Float64)


def _last_day_of_fiscal_year_expr(year_col: str, month_col: str) -> pl.Expr:
    year = pl.col(year_col).cast(pl.Int32, strict=False)
    month = pl.col(month_col).cast(pl.Int32, strict=False)
    return (
        pl.when(year.is_between(_MIN_VALID_YEAR, _MAX_VALID_YEAR) & month.is_between(1, 12))
        .then(pl.date(year, month, pl.lit(1)).dt.month_end())
        .otherwise(pl.lit(None).cast(pl.Date))
    )


def _parse_yyyymmdd_date_expr(col_name: str) -> pl.Expr:
    raw = pl.col(col_name).cast(pl.Int64, strict=False)
    return (
        pl.when(raw.is_between(19000101, 21001231))
        .then(raw.cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d", strict=False))
        .otherwise(pl.lit(None).cast(pl.Date))
    )


def _mask_payload_columns(
    lf: pl.LazyFrame,
    *,
    payload_schema: pl.Schema,
    payload_cols: tuple[str, ...],
    valid_expr: pl.Expr,
) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.when(valid_expr)
            .then(pl.col(name))
            .otherwise(pl.lit(None, dtype=payload_schema[name]))
            .alias(name)
            for name in payload_cols
        ]
    )


def derive_filing_trade_anchors(
    filings_lf: pl.LazyFrame,
    trading_calendar_lf: pl.LazyFrame,
    *,
    filing_date_col: str = "filing_date",
) -> pl.LazyFrame:
    """Attach LM2011 filing-trade and pre-filing-trade anchors to filings."""
    _require_columns(filings_lf, ("doc_id", filing_date_col), "filings")
    calendar_schema = trading_calendar_lf.collect_schema()
    calendar_date_col = _resolve_first_existing(
        calendar_schema,
        ("TRADING_DATE", "CALDT", "trade_date", "daily_caldt"),
        "trading_calendar",
    )

    trading_calendar = (
        trading_calendar_lf.select(pl.col(calendar_date_col).cast(pl.Date, strict=False).alias("trade_date"))
        .drop_nulls()
        .unique()
        .sort("trade_date")
    )

    filings = (
        filings_lf.with_columns(
            pl.col(filing_date_col).cast(pl.Date, strict=False).alias("_filing_date_anchor"),
        )
        .sort("_filing_date_anchor")
    )

    filing_trade = filings.join_asof(
        trading_calendar,
        left_on="_filing_date_anchor",
        right_on="trade_date",
        strategy="forward",
    ).rename({"trade_date": "filing_trade_date"})

    pre_lookup = filing_trade.with_columns(
        (
            (pl.col("_filing_date_anchor").cast(pl.Datetime) - pl.duration(days=1))
            .cast(pl.Date)
            .alias("_pre_filing_lookup_date")
        )
    )

    return (
        pre_lookup.join_asof(
            trading_calendar,
            left_on="_pre_filing_lookup_date",
            right_on="trade_date",
            strategy="backward",
        )
        .rename({"trade_date": "pre_filing_trade_date"})
        .drop("_filing_date_anchor", "_pre_filing_lookup_date")
    )


def build_annual_accounting_panel(
    annual_balance_sheet_lf: pl.LazyFrame,
    annual_income_statement_lf: pl.LazyFrame,
    annual_period_descriptor_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build the detailed-key annual FF2001-compatible accounting panel."""
    join_keys = ("KYGVKEY", "KEYSET", "FYYYY", "fyra")
    _require_columns(
        annual_balance_sheet_lf,
        (
            *join_keys,
            "SEQ",
            "CEQ",
            "AT",
            "LT",
            "TXDITC",
            "PSTKL",
            "PSTKRV",
            "PSTK",
        ),
        "annual_balance_sheet",
    )
    _require_columns(
        annual_income_statement_lf,
        (*join_keys, "IB", "XINT", "TXDI", "DVP"),
        "annual_income_statement",
    )
    _require_columns(
        annual_period_descriptor_lf,
        (*join_keys, "FYEAR", "FYR", "APDEDATE", "FDATE", "PDATE"),
        "annual_period_descriptor",
    )

    bs_schema = annual_balance_sheet_lf.collect_schema()
    preferred_stock_expr = pl.coalesce(
        [
            _float_expr("PSTKL"),
            _float_expr("PSTKRV"),
            _float_expr("PSTK"),
        ]
    )
    base_be_expr = pl.coalesce(
        [
            _float_expr("SEQ"),
            _float_expr("CEQ") + _float_expr("PSTK"),
            _float_expr("AT") - _float_expr("LT"),
        ]
    )
    txditc_expr = _float_expr("TXDITC").fill_null(0.0)
    prba_expr = _optional_float_expr(bs_schema, "PRBA").fill_null(0.0)
    txdi_expr = _float_expr("TXDI").fill_null(0.0)

    return (
        annual_balance_sheet_lf.join(annual_income_statement_lf, on=list(join_keys), how="inner")
        .join(annual_period_descriptor_lf, on=list(join_keys), how="inner")
        .with_columns(
            pl.col("KYGVKEY").cast(GVKEY_DTYPE, strict=False).alias("gvkey_int"),
            pl.coalesce(
                [
                    pl.col("APDEDATE").cast(pl.Date, strict=False),
                    _last_day_of_fiscal_year_expr("FYEAR", "FYR"),
                ]
            ).alias("accounting_period_end"),
            preferred_stock_expr.alias("preferred_stock_ps"),
            (
                base_be_expr - preferred_stock_expr.fill_null(0.0) + txditc_expr - prba_expr
            ).alias("book_equity_be"),
            (_float_expr("IB") + _float_expr("XINT") + txdi_expr).alias("ebit_like_e"),
            (_float_expr("IB") - _float_expr("DVP") + txdi_expr).alias("earnings_available_for_common_y"),
        )
    )


def attach_latest_annual_accounting(
    filings_lf: pl.LazyFrame,
    annual_accounting_panel_lf: pl.LazyFrame,
    *,
    filing_gvkey_col: str = "gvkey",
    filing_date_col: str = "filing_date",
    max_age_days: int = 365,
) -> pl.LazyFrame:
    """Attach the most recent annual accounting row no more than 365 days old."""
    _require_columns(filings_lf, ("doc_id", filing_gvkey_col, filing_date_col), "filings")
    _require_columns(
        annual_accounting_panel_lf,
        ("gvkey_int", "accounting_period_end"),
        "annual_accounting_panel",
    )

    annual_schema = annual_accounting_panel_lf.collect_schema()
    tie_cols = [name for name in ("KEYSET", "FYYYY", "fyra") if name in annual_schema]
    annual_panel = (
        annual_accounting_panel_lf.sort("gvkey_int", "accounting_period_end", *tie_cols)
        .unique(subset=["gvkey_int", "accounting_period_end"], keep="first")
    )
    annual_payload_schema = annual_panel.collect_schema()

    filings = (
        filings_lf.drop("gvkey_int", "accounting_period_end", strict=False)
        .with_columns(
            pl.col(filing_gvkey_col).cast(GVKEY_DTYPE, strict=False).alias("_attach_gvkey_int"),
            pl.col(filing_date_col).cast(pl.Date, strict=False).alias("_attach_filing_date"),
        )
        .sort("_attach_gvkey_int", "_attach_filing_date")
    )

    joined = filings.join_asof(
        annual_panel.sort("gvkey_int", "accounting_period_end", *tie_cols),
        left_on="_attach_filing_date",
        right_on="accounting_period_end",
        by_left=["_attach_gvkey_int"],
        by_right=["gvkey_int"],
        strategy="backward",
        check_sortedness=False,
    )
    joined_schema = joined.collect_schema()
    annual_payload_cols = tuple(name for name in annual_payload_schema.names() if name in joined_schema)

    age_days_expr = (
        (
            pl.col("_attach_filing_date").cast(pl.Datetime)
            - pl.col("accounting_period_end").cast(pl.Datetime)
        )
        .dt.total_days()
    )
    valid_expr = (
        pl.col("accounting_period_end").is_not_null()
        & age_days_expr.is_not_null()
        & (age_days_expr <= pl.lit(max_age_days))
    )

    return _mask_payload_columns(
        joined,
        payload_schema=annual_payload_schema,
        payload_cols=annual_payload_cols,
        valid_expr=valid_expr,
    ).drop("_attach_gvkey_int", "_attach_filing_date")


def build_quarterly_accounting_panel(
    quarterly_balance_sheet_lf: pl.LazyFrame,
    quarterly_income_statement_lf: pl.LazyFrame,
    quarterly_period_descriptor_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build the detailed-key quarterly accounting panel used for SUE eligibility."""
    join_keys = ("KYGVKEY", "KEYSET", "FYYYYQ", "fyrq")
    _require_columns(
        quarterly_balance_sheet_lf,
        (*join_keys, "SEQQ", "CEQQ", "ATQ", "LTQ", "TXDITCQ", "PSTKQ"),
        "quarterly_balance_sheet",
    )
    _require_columns(
        quarterly_income_statement_lf,
        (*join_keys, "IBQ", "XINTQ", "TXDIQ", "DVPQ"),
        "quarterly_income_statement",
    )
    _require_columns(
        quarterly_period_descriptor_lf,
        (*join_keys, "FYEARQ", "FQTR", "APDEDATEQ", "FDATEQ", "PDATEQ", "RDQ"),
        "quarterly_period_descriptor",
    )

    joined = (
        quarterly_balance_sheet_lf.join(quarterly_income_statement_lf, on=list(join_keys), how="inner")
        .join(quarterly_period_descriptor_lf, on=list(join_keys), how="inner")
        .with_columns(_parse_yyyymmdd_date_expr("RDQ").alias("_rdq_date"))
    )

    return joined.with_columns(
        pl.col("KYGVKEY").cast(GVKEY_DTYPE, strict=False).alias("gvkey_int"),
        pl.coalesce(
            [
                pl.col("_rdq_date"),
                pl.col("FDATEQ").cast(pl.Date, strict=False),
                pl.col("PDATEQ").cast(pl.Date, strict=False),
                pl.col("APDEDATEQ").cast(pl.Date, strict=False),
            ]
        ).alias("quarter_report_date"),
    ).drop("_rdq_date")


def attach_eligible_quarterly_accounting(
    filings_lf: pl.LazyFrame,
    quarterly_accounting_panel_lf: pl.LazyFrame,
    *,
    filing_gvkey_col: str = "gvkey",
    filing_date_col: str = "filing_date",
    max_forward_days: int = 90,
) -> pl.LazyFrame:
    """Attach the earliest quarterly report date strictly after filing within 90 days."""
    _require_columns(filings_lf, ("doc_id", filing_gvkey_col, filing_date_col), "filings")
    _require_columns(
        quarterly_accounting_panel_lf,
        ("gvkey_int", "quarter_report_date"),
        "quarterly_accounting_panel",
    )

    quarterly_schema = quarterly_accounting_panel_lf.collect_schema()
    tie_cols = [name for name in ("KEYSET", "FYYYYQ", "fyrq") if name in quarterly_schema]
    quarterly_panel = (
        quarterly_accounting_panel_lf.drop_nulls(subset=["quarter_report_date"])
        .sort("gvkey_int", "quarter_report_date", *tie_cols)
        .unique(subset=["gvkey_int", "quarter_report_date"], keep="first")
    )
    quarterly_payload_schema = quarterly_panel.collect_schema()

    filings = (
        filings_lf.drop("gvkey_int", "quarter_report_date", strict=False)
        .with_columns(
            pl.col(filing_gvkey_col).cast(GVKEY_DTYPE, strict=False).alias("_attach_gvkey_int"),
            pl.col(filing_date_col).cast(pl.Date, strict=False).alias("_attach_filing_date"),
            (
                (pl.col(filing_date_col).cast(pl.Datetime, strict=False) + pl.duration(days=1))
                .cast(pl.Date)
                .alias("_attach_quarter_lookup_start")
            ),
        )
        .sort("_attach_gvkey_int", "_attach_quarter_lookup_start")
    )

    joined = filings.join_asof(
        quarterly_panel.sort("gvkey_int", "quarter_report_date", *tie_cols),
        left_on="_attach_quarter_lookup_start",
        right_on="quarter_report_date",
        by_left=["_attach_gvkey_int"],
        by_right=["gvkey_int"],
        strategy="forward",
        check_sortedness=False,
    )
    joined_schema = joined.collect_schema()
    quarterly_payload_cols = tuple(name for name in quarterly_payload_schema.names() if name in joined_schema)

    lag_days_expr = (
        (
            pl.col("quarter_report_date").cast(pl.Datetime)
            - pl.col("_attach_filing_date").cast(pl.Datetime)
        )
        .dt.total_days()
    )
    valid_expr = (
        pl.col("quarter_report_date").is_not_null()
        & lag_days_expr.is_not_null()
        & (lag_days_expr >= 1)
        & (lag_days_expr <= pl.lit(max_forward_days))
    )

    return _mask_payload_columns(
        joined,
        payload_schema=quarterly_payload_schema,
        payload_cols=quarterly_payload_cols,
        valid_expr=valid_expr,
    ).drop("_attach_gvkey_int", "_attach_filing_date", "_attach_quarter_lookup_start")


def attach_pre_filing_market_data(
    filings_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    *,
    filing_permno_col: str = "kypermno",
    pre_filing_trade_date_col: str = "pre_filing_trade_date",
) -> pl.LazyFrame:
    """Attach the exact pre-filing daily market row and derive event-date size/BM."""
    _require_columns(
        filings_lf,
        ("doc_id", filing_permno_col, pre_filing_trade_date_col),
        "filings",
    )
    daily_schema = daily_lf.collect_schema()
    daily_permno_col = _resolve_first_existing(
        daily_schema,
        ("KYPERMNO", "kypermno"),
        "daily_panel",
    )
    daily_date_col = _resolve_first_existing(
        daily_schema,
        ("CALDT", "daily_caldt"),
        "daily_panel",
    )
    projected_cols = [
        col
        for col in ("TCAP", "PRC", "SHROUT", "VOL", "SHRCD", "EXCHCD")
        if col in daily_schema
    ]
    if "TCAP" not in daily_schema and ("PRC" not in daily_schema or "SHROUT" not in daily_schema):
        raise ValueError("daily_panel must contain TCAP or both PRC and SHROUT for event-date market equity")

    daily = (
        daily_lf.select(
            pl.col(daily_permno_col).cast(pl.Int32, strict=False).alias("_daily_permno"),
            pl.col(daily_date_col).cast(pl.Date, strict=False).alias("_daily_trade_date"),
            *[pl.col(col) for col in projected_cols],
        )
        .drop_nulls(subset=["_daily_permno", "_daily_trade_date"])
        .unique(subset=["_daily_permno", "_daily_trade_date"], keep="first")
    )

    filings = filings_lf.drop(
        "market_equity_me_event",
        "size_event",
        "bm_event",
        strict=False,
    ).with_columns(
        pl.col(filing_permno_col).cast(pl.Int32, strict=False).alias("_attach_permno"),
        pl.col(pre_filing_trade_date_col).cast(pl.Date, strict=False).alias("_attach_pre_trade_date"),
    )

    joined = filings.join(
        daily,
        left_on=["_attach_permno", "_attach_pre_trade_date"],
        right_on=["_daily_permno", "_daily_trade_date"],
        how="left",
    )

    me_event_expr = pl.coalesce(
        [
            pl.col("TCAP").cast(pl.Float64, strict=False),
            (
                pl.col("PRC").cast(pl.Float64, strict=False).abs()
                * pl.col("SHROUT").cast(pl.Float64, strict=False)
            ),
        ]
    )

    schema = joined.collect_schema()
    bm_expr = pl.lit(None, dtype=pl.Float64)
    if "book_equity_be" in schema:
        bm_expr = (
            pl.when(
                pl.col("book_equity_be").cast(pl.Float64, strict=False).is_not_null()
                & me_event_expr.is_not_null()
                & (me_event_expr > pl.lit(0.0))
            )
            .then(pl.col("book_equity_be").cast(pl.Float64, strict=False) / me_event_expr)
            .otherwise(pl.lit(None, dtype=pl.Float64))
        )

    return joined.with_columns(
        me_event_expr.alias("market_equity_me_event"),
        me_event_expr.alias("size_event"),
        bm_expr.alias("bm_event"),
    ).drop("_attach_permno", "_attach_pre_trade_date", "_daily_permno", "_daily_trade_date", strict=False)
