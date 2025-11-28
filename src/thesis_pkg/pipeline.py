from __future__ import annotations

import datetime as dt
from enum import IntFlag
from pathlib import Path
from typing import Iterable

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

    # Convenience Combinations for Filtering
    FULL_PANEL_DATA = HAS_RET | HAS_BIDLO
    ANY_RET_DATA = HAS_RET | HAS_DLRET


def _ensure_data_status(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Guarantee presence and dtype of the shared status column."""
    schema = lf.collect_schema()
    if "data_status" not in schema:
        return lf.with_columns(pl.lit(int(DataStatus.NONE), dtype=STATUS_DTYPE).alias("data_status"))
    return lf.with_columns(pl.col("data_status").cast(STATUS_DTYPE))


def _flag_if(expr: pl.Expr, flag: DataStatus) -> pl.Expr:
    """Helper to promote a boolean to a UInt64 bitmask."""
    return expr.cast(STATUS_DTYPE) * int(flag)


def _coerce_date(value: dt.date | dt.datetime | str) -> dt.date:
    """Normalize input to a date for filtering."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def load_tables(base_dirs: Iterable[Path], wanted: Iterable[str]) -> dict[str, pl.LazyFrame]:
    """Load parquet files into LazyFrames keyed by filename stem."""
    wanted_set = set(wanted)
    tables: dict[str, pl.LazyFrame] = {}
    for base in base_dirs:
        if not base.exists():
            continue
        for path in base.rglob("*.parquet"):
            key = path.stem
            if key in tables or key not in wanted_set:
                continue
            tables[key] = pl.scan_parquet(path)
    return tables


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


def merge_histories(
    price_lf: pl.LazyFrame,
    sec_hist_lf: pl.LazyFrame,
    comp_hist_lf: pl.LazyFrame,
    output_dir: Path,
    temp_name: str = "temp_step1_sec_merge.parquet",
    final_name: str = "final_flagged_data.parquet",
) -> Path:
    """RAM-safe merge of security and company histories without dropping bad rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_dir / temp_name
    final_path = output_dir / final_name

    sec_schema = sec_hist_lf.collect_schema()
    comp_schema = comp_hist_lf.collect_schema()

    base = _ensure_data_status(price_lf).with_columns(
        pl.col("CALDT").cast(pl.Date, strict=False).alias("CALDT"),
        pl.col("data_status").cast(STATUS_DTYPE),
    )

    base_schema = base.collect_schema()
    if "LIID" in base_schema:
        base = base.with_columns(pl.col("LIID").cast(sec_schema.get("KYIID", pl.Utf8)))

    sec_hist = sec_hist_lf.select(
        "KYGVKEY",
        "KYIID",
        pl.col("HSCHGDT").alias("HIST_START_DATE_SEC"),
        pl.col("HSCHGENDDT").alias("HCHGENDDT_SEC"),
        "HTPCI",
        "HEXCNTRY",
    ).sort("KYGVKEY", "KYIID", "HIST_START_DATE_SEC")

    comp_hist = comp_hist_lf.select(
        "KYGVKEY",
        pl.col("HCHGDT").alias("HIST_START_DATE_COMP"),
        pl.col("HCHGENDDT").alias("HCHGENDDT_COMP"),
        "HCIK",
        "HSIC",
        "HNAICS",
        "HGSUBIND",
    ).sort("KYGVKEY", "HIST_START_DATE_COMP")

    # Security history join (requires GVKEY + LIID)
    sec_keys = pl.col("KYGVKEY_final").is_not_null() & pl.col("LIID").is_not_null()
    sec_candidates = base.filter(sec_keys)

    sec_joined = (
        sec_candidates.sort("KYGVKEY_final", "LIID", "CALDT")
        .join_asof(
            sec_hist,
            left_on="CALDT",
            right_on="HIST_START_DATE_SEC",
            by_left=["KYGVKEY_final", "LIID"],
            by_right=["KYGVKEY", "KYIID"],
            strategy="backward",
        )
        .select(pl.all().exclude(["KYGVKEY", "KYIID"]))
    )

    sec_matched = pl.col("HIST_START_DATE_SEC").is_not_null()
    sec_valid = sec_matched & (pl.col("HCHGENDDT_SEC").is_null() | (pl.col("CALDT") <= pl.col("HCHGENDDT_SEC")))
    sec_stale = sec_matched & pl.col("HCHGENDDT_SEC").is_not_null() & (pl.col("CALDT") > pl.col("HCHGENDDT_SEC"))

    sec_status = (
        pl.col("data_status").cast(STATUS_DTYPE)
        | pl.lit(int(DataStatus.SECHIST_CAN_ATTEMPT | DataStatus.SECHIST_ATTEMPTED), dtype=STATUS_DTYPE)
        | _flag_if(sec_matched, DataStatus.SECHIST_MATCHED)
        | _flag_if(sec_valid, DataStatus.SECHIST_VALID)
        | _flag_if(sec_stale, DataStatus.SECHIST_STALE)
        | _flag_if(sec_matched.not_(), DataStatus.SECHIST_NO_MATCH)
    ).alias("data_status")

    sec_joined = sec_joined.with_columns(
        sec_status,
        pl.when(sec_valid).then(pl.col("HTPCI")).otherwise(pl.lit(None, dtype=pl.String)).alias("HTPCI"),
        pl.when(sec_valid).then(pl.col("HEXCNTRY")).otherwise(pl.lit(None, dtype=pl.String)).alias("HEXCNTRY"),
        pl.when(sec_valid).then(pl.col("HIST_START_DATE_SEC")).otherwise(pl.lit(None).cast(pl.Date)).alias("HIST_START_DATE_SEC"),
        pl.when(sec_valid).then(pl.col("HCHGENDDT_SEC")).otherwise(pl.lit(None).cast(pl.Date)).alias("HCHGENDDT_SEC"),
    )

    sec_missing = base.filter(sec_keys.not_()).with_columns(
        pl.col("data_status").cast(STATUS_DTYPE),
        pl.lit(None).cast(pl.Date).alias("HIST_START_DATE_SEC"),
        pl.lit(None).cast(pl.Date).alias("HCHGENDDT_SEC"),
        pl.lit(None, dtype=pl.String).alias("HTPCI"),
        pl.lit(None, dtype=pl.String).alias("HEXCNTRY"),
    )

    sec_combined = pl.concat([sec_joined, sec_missing], how="vertical_relaxed").with_columns(
        pl.col("data_status").cast(STATUS_DTYPE)
    )
    sec_combined.sink_parquet(temp_path, compression="zstd")

    sec_loaded = pl.scan_parquet(temp_path).with_columns(pl.col("data_status").cast(STATUS_DTYPE))

    # Company history join (requires GVKEY only)
    comp_keys = pl.col("KYGVKEY_final").is_not_null()
    comp_candidates = sec_loaded.filter(comp_keys)

    hcik_dtype = comp_schema.get("HCIK", pl.String)
    hsic_dtype = comp_schema.get("HSIC", pl.Int32)
    hnaics_dtype = comp_schema.get("HNAICS", pl.String)
    hgsubind_dtype = comp_schema.get("HGSUBIND", pl.String)

    comp_joined = (
        comp_candidates.sort("KYGVKEY_final", "CALDT")
        .join_asof(
            comp_hist,
            left_on="CALDT",
            right_on="HIST_START_DATE_COMP",
            by_left="KYGVKEY_final",
            by_right="KYGVKEY",
            strategy="backward",
        )
        .select(pl.all().exclude(["KYGVKEY"]))
    )

    comp_matched = pl.col("HIST_START_DATE_COMP").is_not_null()
    comp_valid = comp_matched & (pl.col("HCHGENDDT_COMP").is_null() | (pl.col("CALDT") <= pl.col("HCHGENDDT_COMP")))
    comp_stale = comp_matched & pl.col("HCHGENDDT_COMP").is_not_null() & (pl.col("CALDT") > pl.col("HCHGENDDT_COMP"))

    comp_status = (
        pl.col("data_status").cast(STATUS_DTYPE)
        | pl.lit(int(DataStatus.COMPHIST_CAN_ATTEMPT | DataStatus.COMPHIST_ATTEMPTED), dtype=STATUS_DTYPE)
        | _flag_if(comp_matched, DataStatus.COMPHIST_MATCHED)
        | _flag_if(comp_valid, DataStatus.COMPHIST_VALID)
        | _flag_if(comp_stale, DataStatus.COMPHIST_STALE)
        | _flag_if(comp_matched.not_(), DataStatus.COMPHIST_NO_MATCH)
    ).alias("data_status")

    comp_joined = comp_joined.with_columns(
        comp_status,
        pl.when(comp_valid).then(pl.col("HCIK")).otherwise(pl.lit(None, dtype=hcik_dtype)).alias("HCIK"),
        pl.when(comp_valid).then(pl.col("HSIC")).otherwise(pl.lit(None, dtype=hsic_dtype)).alias("HSIC"),
        pl.when(comp_valid).then(pl.col("HNAICS")).otherwise(pl.lit(None, dtype=hnaics_dtype)).alias("HNAICS"),
        pl.when(comp_valid).then(pl.col("HGSUBIND")).otherwise(pl.lit(None, dtype=hgsubind_dtype)).alias("HGSUBIND"),
        pl.when(comp_valid).then(pl.col("HIST_START_DATE_COMP")).otherwise(pl.lit(None).cast(pl.Date)).alias("HIST_START_DATE_COMP"),
        pl.when(comp_valid).then(pl.col("HCHGENDDT_COMP")).otherwise(pl.lit(None).cast(pl.Date)).alias("HCHGENDDT_COMP"),
    )

    comp_missing = sec_loaded.filter(comp_keys.not_()).with_columns(
        pl.col("data_status").cast(STATUS_DTYPE),
        pl.lit(None).cast(pl.Date).alias("HIST_START_DATE_COMP"),
        pl.lit(None).cast(pl.Date).alias("HCHGENDDT_COMP"),
        pl.lit(None, dtype=hcik_dtype).alias("HCIK"),
        pl.lit(None, dtype=hsic_dtype).alias("HSIC"),
        pl.lit(None, dtype=hnaics_dtype).alias("HNAICS"),
        pl.lit(None, dtype=hgsubind_dtype).alias("HGSUBIND"),
    )

    final_output = pl.concat([comp_joined, comp_missing], how="vertical_relaxed").with_columns(
        pl.col("data_status").cast(STATUS_DTYPE)
    )

    final_output.sink_parquet(final_path, compression="zstd")

    return final_path
