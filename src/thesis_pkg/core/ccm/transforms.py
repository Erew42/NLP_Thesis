from __future__ import annotations

import datetime as dt
from enum import IntFlag

import polars as pl


STATUS_DTYPE = pl.UInt64


class DataStatus(IntFlag):
    """Bitmask for data availability, provenance, and join results."""

    NONE = 0

    # Core Data Availability (Used in build_price_panel)
    HAS_RET = 1 << 0
    HAS_BIDLO = 1 << 1

    # Detailed Provenance (Used in add_final_returns)
    HAS_RETX = 1 << 2
    HAS_PRC = 1 << 3
    HAS_DLRET = 1 << 4
    HAS_DLRETX = 1 << 5
    HAS_DLPRC = 1 << 6

    # Company history diagnostics
    COMPHIST_CAN_ATTEMPT = 1 << 7
    COMPHIST_ATTEMPTED = 1 << 8
    COMPHIST_MATCHED = 1 << 9
    COMPHIST_NO_MATCH = 1 << 10
    COMPHIST_VALID = 1 << 11
    COMPHIST_STALE = 1 << 12

    # Security history diagnostics
    SECHIST_CAN_ATTEMPT = 1 << 13
    SECHIST_ATTEMPTED = 1 << 14
    SECHIST_MATCHED = 1 << 15
    SECHIST_NO_MATCH = 1 << 16
    SECHIST_VALID = 1 << 17
    SECHIST_STALE = 1 << 18

    # Auxiliary metadata
    HAS_COMP_DESC = 1 << 19

    # SEC-CCM Phase A linking diagnostics
    SEC_CCM_PHASE_A_ATTEMPTED = 1 << 20
    SEC_CCM_BAD_INPUT = 1 << 21
    SEC_CCM_CIK_NOT_IN_LINK_UNIVERSE = 1 << 22
    SEC_CCM_AMBIGUOUS_LINK = 1 << 23
    SEC_CCM_LINKED_OK = 1 << 24
    SEC_CCM_HAS_ACCEPTANCE_DATETIME = 1 << 25

    # SEC-CCM Phase B alignment + daily join diagnostics
    SEC_CCM_PHASE_B_ALIGNMENT_ATTEMPTED = 1 << 26
    SEC_CCM_PHASE_B_ALIGNED = 1 << 27
    SEC_CCM_PHASE_B_OUT_OF_CCM_COVERAGE = 1 << 28
    SEC_CCM_PHASE_B_DAILY_JOIN_ATTEMPTED = 1 << 29
    SEC_CCM_PHASE_B_DAILY_ROW_FOUND = 1 << 30
    SEC_CCM_PHASE_B_NO_CCM_ROW_FOR_DATE = 1 << 31

    # Concept filter pass diagnostics
    SEC_CCM_FILTER_PRICE_PASS = 1 << 32
    SEC_CCM_FILTER_COMMON_STOCK_PASS = 1 << 33
    SEC_CCM_FILTER_MAJOR_EXCHANGE_PASS = 1 << 34
    SEC_CCM_FILTER_LIQUIDITY_PASS = 1 << 35
    SEC_CCM_FILTER_NON_MICROCAP_PASS = 1 << 36
    SEC_CCM_FILTER_ALL_PASS = 1 << 37

    # Convenience Combinations for Filtering
    FULL_PANEL_DATA = HAS_RET | HAS_BIDLO
    ANY_RET_DATA = HAS_RET | HAS_DLRET


def _ensure_data_status(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Guarantee presence and dtype of the shared status column."""
    schema = lf.collect_schema()
    if "data_status" not in schema:
        return lf.with_columns(pl.lit(int(DataStatus.NONE), dtype=STATUS_DTYPE).alias("data_status"))
    return lf.with_columns(pl.col("data_status").cast(STATUS_DTYPE).fill_null(int(DataStatus.NONE)))


def _flag_if(expr: pl.Expr, flag: DataStatus) -> pl.Expr:
    """Helper to promote a boolean to a UInt64 bitmask."""
    return expr.fill_null(False).cast(STATUS_DTYPE) * int(flag)


def _coerce_date(value: dt.date | dt.datetime | str) -> dt.date:
    """Normalize input to a date for filtering."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def add_five_to_six() -> int:
    """Return the result of adding five to six."""
    return 5 + 6


def build_price_panel(
    sfz_ds: pl.LazyFrame,
    sfz_dp: pl.LazyFrame,
    sfz_del: pl.LazyFrame,
    start_date: dt.date | dt.datetime | str,
) -> pl.LazyFrame:
    """Merge CRSP price feeds plus delist data."""
    start_dt = _coerce_date(start_date)
    ds = sfz_ds.filter(pl.col("CALDT") >= pl.lit(start_dt))
    dp = sfz_dp.filter(pl.col("CALDT") >= pl.lit(start_dt))

    price = dp.join(ds, on=["KYPERMNO", "CALDT"], how="full", coalesce=True).with_columns(
        (
            _flag_if(pl.col("RET").is_not_null(), DataStatus.HAS_RET)
            | _flag_if(pl.col("BIDLO").is_not_null(), DataStatus.HAS_BIDLO)
        )
        .cast(STATUS_DTYPE)
        .alias("data_status")
    )

    last_trade = price.group_by("KYPERMNO").agg(pl.col("CALDT").max().alias("LAST_CALDT"))

    trading_days = (
        price.select(
            "KYPERMNO",
            pl.col("CALDT"),
            pl.col("CALDT").alias("CALDT_CHECK"),
        )
        .unique()
    )

    delist_info = sfz_del.join(last_trade, on="KYPERMNO", how="inner")

    delist_check = delist_info.join(
        trading_days.select("KYPERMNO", "CALDT", "CALDT_CHECK"),
        left_on=["KYPERMNO", "DLSTDT"],
        right_on=["KYPERMNO", "CALDT"],
        how="left",
    )

    delist_payload = (
        delist_check.with_columns(
            pl.col("CALDT_CHECK").is_not_null().alias("dlst_is_trading_day"),
            pl.when(pl.col("CALDT_CHECK").is_not_null())
            .then(pl.col("DLSTDT"))
            .otherwise(pl.col("LAST_CALDT"))
            .alias("MATCH_DATE"),
        )
        .drop("CALDT", "CALDT_CHECK", strict=False)
    )

    delist_cols = ["KYPERMNO", "MATCH_DATE", "DLSTDT", "DLSTCD", "DLPRC", "DLAMT", "DLRET", "DLRETX"]
    delist_final = delist_payload.select(delist_cols).unique(subset=["KYPERMNO", "MATCH_DATE"], keep="first")

    return price.join(
        delist_final,
        left_on=["KYPERMNO", "CALDT"],
        right_on=["KYPERMNO", "MATCH_DATE"],
        how="left",
    )


def add_final_returns(price_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add FINAL_RET/RETX/PRC and provenance flags."""
    lf = _ensure_data_status(price_lf)

    ret_available = pl.col("RET").is_not_null()
    retx_available = pl.col("RETX").is_not_null()
    prc_available = pl.col("PRC").is_not_null()
    dlret_available = pl.col("DLRET").is_not_null()
    dlretx_available = pl.col("DLRETX").is_not_null()
    dlprc_available = pl.col("DLPRC").is_not_null()

    availability_flags = (
        pl.col("data_status").cast(STATUS_DTYPE)
        | _flag_if(ret_available, DataStatus.HAS_RET)
        | _flag_if(retx_available, DataStatus.HAS_RETX)
        | _flag_if(prc_available, DataStatus.HAS_PRC)
        | _flag_if(dlret_available, DataStatus.HAS_DLRET)
        | _flag_if(dlretx_available, DataStatus.HAS_DLRETX)
        | _flag_if(dlprc_available, DataStatus.HAS_DLPRC)
    ).cast(STATUS_DTYPE).alias("data_status")

    return lf.with_columns(
        availability_flags,
        pl.when(ret_available & dlret_available)
        .then((1.0 + pl.col("RET")) * (1.0 + pl.col("DLRET")) - 1.0)
        .when(dlret_available)
        .then(pl.col("DLRET"))
        .otherwise(pl.col("RET"))
        .alias("FINAL_RET"),
        pl.when(retx_available & dlretx_available)
        .then((1.0 + pl.col("RETX")) * (1.0 + pl.col("DLRETX")) - 1.0)
        .when(dlretx_available)
        .then(pl.col("DLRETX"))
        .otherwise(pl.col("RETX"))
        .alias("FINAL_RETX"),
        pl.when(dlprc_available)
        .then(pl.col("DLPRC").abs())
        .otherwise(pl.col("PRC").abs())
        .alias("FINAL_PRC"),
    )


def attach_filings(price_lf: pl.LazyFrame, filings_lf: pl.LazyFrame, keep_forms: list[str]) -> pl.LazyFrame:
    """Map filings to the next trading day and align as arrays on the price panel."""
    trading_calendar = (
        price_lf
        .with_columns(pl.col("CALDT").cast(pl.Date, strict=False).alias("TRADING_DATE"))
        .select("TRADING_DATE")
        .drop_nulls()
        .unique()
        .sort("TRADING_DATE")
    )

    tmin = trading_calendar.select(pl.col("TRADING_DATE").min().alias("tmin")).collect().item()

    filings = (
        filings_lf
        .filter(pl.col("SRCTYPE").is_in(keep_forms))
        .with_columns([
            pl.col("LPERMNO").cast(pl.Int32),
            pl.col("FILEDATE").cast(pl.Date),
            pl.col("FILEDATETIME").str.strptime(pl.Time, format="%H:%M:%S", strict=False).alias("_FILETIME"),
        ])
        .with_columns([
            pl.when(pl.col("_FILETIME").is_not_null())
            .then(
                (pl.col("FILEDATE").cast(pl.Datetime) + pl.duration(
                    hours=pl.col("_FILETIME").dt.hour(),
                    minutes=pl.col("_FILETIME").dt.minute(),
                    seconds=pl.col("_FILETIME").dt.second(),
                )).dt.replace_time_zone("America/New_York")
            )
            .otherwise(pl.lit(None, dtype=pl.Datetime(time_zone="America/New_York")))
            .alias("FILEDATETIME_ET"),
        ])
        .select(["LPERMNO", "SRCTYPE", "FILEDATE", "FILEDATETIME_ET"])
        .filter(pl.col("FILEDATE") >= pl.lit(tmin))
    )

    event_map = (
        filings
        .with_columns(pl.col("FILEDATE").alias("CALDATE_KEY"))
        .sort("CALDATE_KEY")
        .join_asof(
            trading_calendar,
            left_on="CALDATE_KEY",
            right_on="TRADING_DATE",
            strategy="forward",
        )
        .rename({"TRADING_DATE": "EVENT_TDATE"})
        .select(["LPERMNO", "SRCTYPE", "FILEDATE", "FILEDATETIME_ET", "EVENT_TDATE"])
    )

    event_bag = (
        event_map
        .with_row_index("row_idx")
        .with_columns([
            pl.coalesce([
                pl.col("FILEDATETIME_ET"),
                pl.col("FILEDATE").cast(pl.Datetime).dt.replace_time_zone("America/New_York"),
            ]).alias("_sort_dt"),
            pl.struct(["SRCTYPE", "FILEDATE", "FILEDATETIME_ET"]).alias("_rec"),
        ])
        .group_by(["LPERMNO", "EVENT_TDATE"])
        .agg([
            pl.col("_rec").sort_by(["_sort_dt", "row_idx"]).alias("FILINGS"),
        ])
        .with_columns([
            pl.col("FILINGS").list.eval(pl.element().struct.field("SRCTYPE")).alias("SRCTYPE_all"),
            pl.col("FILINGS").list.eval(pl.element().struct.field("FILEDATE")).alias("FILEDATE_all"),
            pl.col("FILINGS").list.eval(pl.element().struct.field("FILEDATETIME_ET")).alias("FILEDATETIME_all"),
            pl.col("FILINGS").list.len().alias("n_filings"),
        ])
        .select(["LPERMNO", "EVENT_TDATE", "SRCTYPE_all", "FILEDATE_all", "FILEDATETIME_all", "n_filings"])
    )

    return (
        price_lf
        .with_columns([
            pl.col("KYPERMNO").cast(pl.Int32),
            pl.col("CALDT").cast(pl.Date, strict=False).alias("CALDT"),
        ])
        .join(
            event_bag,
            left_on=["KYPERMNO", "CALDT"],
            right_on=["LPERMNO", "EVENT_TDATE"],
            how="left",
        )
    )


def attach_ccm_links(price_filings_lf: pl.LazyFrame, link_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Select best CCM link per permno/date with ranking and masking."""
    link_annotated = (
        link_lf
        .with_columns([
            pl.col("LINKENDDT").fill_null(dt.date(9999, 12, 31)).alias("LINKENDDT"),
            (pl.col("LINKTYPE").is_in(["LC", "LU"]) & pl.col("LINKPRIM").is_in(["P", "C"])).alias("is_canonical_link"),
        ])
        .select(
            pl.col("KYGVKEY"),
            pl.col("LPERMNO").alias("KYPERMNO"),
            "LIID", "LINKDT", "LINKENDDT", "LINKTYPE", "LINKPRIM", "is_canonical_link",
        )
    )

    joined = price_filings_lf.join(link_annotated, on="KYPERMNO", how="left").with_columns(
        (
            pl.col("LINKDT").is_not_null()
            & (pl.col("CALDT") >= pl.col("LINKDT"))
            & (pl.col("CALDT") <= pl.col("LINKENDDT"))
        ).alias("valid_link")
    )

    ranked = joined.with_columns(
        pl.when(pl.col("is_canonical_link") & (pl.col("LINKTYPE") == "LC") & (pl.col("LINKPRIM") == "P"))
        .then(1)
        .when(pl.col("is_canonical_link") & (pl.col("LINKPRIM") == "P"))
        .then(2)
        .when(pl.col("is_canonical_link"))
        .then(3)
        .otherwise(4)
        .alias("link_rank_base")
    ).with_columns(
        pl.when(pl.col("valid_link")).then(pl.col("link_rank_base")).otherwise(9).alias("link_rank")
    )

    dedup = (
        ranked
        .sort(["KYPERMNO", "CALDT", "link_rank", "LINKDT"])
        .group_by(["KYPERMNO", "CALDT"], maintain_order=True)
        .agg([
            pl.all().exclude(["KYPERMNO", "CALDT"]).first()
        ])
    )

    masked = dedup.with_columns(
        pl.when(pl.col("valid_link")).then(pl.col("KYGVKEY")).otherwise(pl.lit(None)).alias("KYGVKEY_ccm"),
        pl.when(pl.col("valid_link")).then(pl.col("LIID")).otherwise(pl.lit(None)).alias("LIID"),
        pl.when(pl.col("valid_link")).then(pl.col("LINKDT")).otherwise(pl.lit(None)).alias("LINKDT"),
        pl.when(pl.col("valid_link")).then(pl.col("LINKENDDT")).otherwise(pl.lit(None)).alias("LINKENDDT"),
        pl.when(pl.col("valid_link")).then(pl.col("LINKTYPE")).otherwise(pl.lit(None)).alias("LINKTYPE"),
        pl.when(pl.col("valid_link")).then(pl.col("LINKPRIM")).otherwise(pl.lit(None)).alias("LINKPRIM"),
        pl.when(pl.col("valid_link")).then(pl.col("is_canonical_link")).otherwise(pl.lit(None)).alias("is_canonical_link"),
    )

    return masked.with_columns(
        pl.col("KYGVKEY_ccm").alias("KYGVKEY_final"),
        pl.when(pl.col("KYGVKEY_ccm").is_null())
        .then(pl.lit("no_ccm_link"))
        .when(pl.col("is_canonical_link").fill_null(False) & (pl.col("LINKTYPE") == "LC") & (pl.col("LINKPRIM") == "P"))
        .then(pl.lit("canonical_primary_LC"))
        .when(pl.col("is_canonical_link").fill_null(False) & (pl.col("LINKPRIM") == "P"))
        .then(pl.lit("canonical_primary_other"))
        .when(pl.col("is_canonical_link").fill_null(False))
        .then(pl.lit("canonical_other"))
        .otherwise(pl.lit("noncanonical"))
        .alias("link_quality_flag"),
    )


def attach_company_description(
    flagged_lf: pl.LazyFrame,
    comp_desc_lf: pl.LazyFrame,
    desc_key_col: str = "KYGVKEY",
    desc_cik_col: str = "CIK",
) -> pl.LazyFrame:
    """
    Attach description availability and harmonized CIK from the company description feed.

    Expects that KYGVKEY_final already exists (e.g., after CCM linking).
    Prefers HCIK (from company history) over existing CIK_final over description-derived CIK.
    """
    lf = _ensure_data_status(flagged_lf)

    schema = lf.collect_schema()
    gvkey_dtype = schema.get("KYGVKEY_final", pl.Utf8)

    # --- 1) Existence flag map (keep a marker column; do NOT depend on join key surviving) ---
    desc_keys = (
        comp_desc_lf
        .select([
            pl.col(desc_key_col).cast(gvkey_dtype, strict=False).alias("KYGVKEY_final"),
            pl.lit(True).alias("_has_comp_desc"),
        ])
        .drop_nulls(subset=["KYGVKEY_final"])
        .unique(subset=["KYGVKEY_final"])
        .collect()
    )

    # --- 2) Deterministic-ish CIK map (mode per GVKEY) ---
    cik_map = (
        comp_desc_lf
        .select([
            pl.col(desc_key_col).cast(gvkey_dtype, strict=False).alias("KYGVKEY_final"),
            pl.col(desc_cik_col).cast(pl.Utf8, strict=False).alias("CIK_raw"),
        ])
        .drop_nulls(subset=["KYGVKEY_final"])
        .with_columns(
            pl.col("CIK_raw")
              .str.strip_chars()
              .str.replace_all(r"\.0$", "")
              .str.replace_all(r"\D", "")
              .alias("CIK_digits")
        )
        .with_columns(
            pl.when(pl.col("CIK_digits").str.len_chars() > 0)
              .then(pl.col("CIK_digits").str.zfill(10))
              .otherwise(None)
              .alias("CIK_desc_10")
        )
        .group_by("KYGVKEY_final")
        .agg([
            pl.col("CIK_desc_10").drop_nulls().mode().first().alias("CIK_desc_10"),
            pl.col("CIK_desc_10").drop_nulls().n_unique().alias("n_cik_desc_uniq"),
        ])
        .collect()
    )

    out = (
        lf
        .join(desc_keys.lazy(), on="KYGVKEY_final", how="left")
        .with_columns(
            (
                pl.col("data_status").cast(STATUS_DTYPE)
                | _flag_if(pl.col("_has_comp_desc").fill_null(False), DataStatus.HAS_COMP_DESC)
            ).alias("data_status")
        )
        .drop("_has_comp_desc", strict=False)
        .join(cik_map.lazy(), on="KYGVKEY_final", how="left")
    )

    # Normalize CompHist CIK too (avoid turning empty-string into "0000000000")
    if "HCIK" in schema:
        out = out.with_columns(
            pl.col("HCIK").cast(pl.Utf8, strict=False)
              .str.strip_chars()
              .str.replace_all(r"\.0$", "")
              .str.replace_all(r"\D", "")
              .alias("_HCIK_digits")
        ).with_columns(
            pl.when(pl.col("_HCIK_digits").str.len_chars() > 0)
              .then(pl.col("_HCIK_digits").str.zfill(10))
              .otherwise(None)
              .alias("HCIK_10")
        ).drop("_HCIK_digits", strict=False)
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("HCIK_10"))

    existing_cik_final = pl.col("CIK_final") if "CIK_final" in schema else pl.lit(None, dtype=pl.Utf8)

    return out.with_columns(
        pl.coalesce([pl.col("HCIK_10"), existing_cik_final, pl.col("CIK_desc_10")]).alias("CIK_final")
    )
