from __future__ import annotations

import datetime as dt
from enum import IntFlag, auto
from pathlib import Path
from typing import Iterable

import polars as pl


class DataStatus(IntFlag):
    """Bitmask for data availability, provenance, and join results."""

    NONE = 0

    # Core Data Availability (Used in build_price_panel)
    HAS_RET = auto()  # 1
    HAS_BIDLO = auto()  # 2

    # Detailed Provenance (Used in add_final_returns)
    HAS_RETX = auto()  # 4
    HAS_PRC = auto()  # 8
    HAS_DLRET = auto()  # 16
    HAS_DLRETX = auto()  # 32
    HAS_DLPRC = auto()  # 64

    # History join flags (used in merge_histories)
    CompHist_OK = auto()  # 128
    SecHist_OK = auto()  # 256

    # Convenience Combinations for Filtering
    FULL_PANEL_DATA = HAS_RET | HAS_BIDLO
    ANY_RET_DATA = HAS_RET | HAS_DLRET


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

    price = dp.join(ds, on=["KYPERMNO", "CALDT"], how="outer_coalesce").with_columns(
        (
            (pl.col("RET").is_not_null().cast(pl.UInt64) * DataStatus.HAS_RET)
            | (pl.col("BIDLO").is_not_null().cast(pl.UInt64) * DataStatus.HAS_BIDLO)
        ).alias("price_status")
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
    flags_expr = (
        (pl.col("RET").is_not_null().cast(pl.UInt64) * DataStatus.HAS_RET)
        | (pl.col("RETX").is_not_null().cast(pl.UInt64) * DataStatus.HAS_RETX)
        | (pl.col("PRC").is_not_null().cast(pl.UInt64) * DataStatus.HAS_PRC)
        | (pl.col("DLRET").is_not_null().cast(pl.UInt64) * DataStatus.HAS_DLRET)
        | (pl.col("DLRETX").is_not_null().cast(pl.UInt64) * DataStatus.HAS_DLRETX)
        | (pl.col("DLPRC").is_not_null().cast(pl.UInt64) * DataStatus.HAS_DLPRC)
    ).alias("prov_flags")

    return price_lf.with_columns(flags_expr).with_columns(
        pl.when((pl.col("prov_flags") & (DataStatus.HAS_RET | DataStatus.HAS_DLRET)) == (DataStatus.HAS_RET | DataStatus.HAS_DLRET))
        .then((1.0 + pl.col("RET")) * (1.0 + pl.col("DLRET")) - 1.0)
        .when((pl.col("prov_flags") & DataStatus.HAS_DLRET) > 0)
        .then(pl.col("DLRET"))
        .otherwise(pl.col("RET"))
        .alias("FINAL_RET"),
        pl.when((pl.col("prov_flags") & (DataStatus.HAS_RETX | DataStatus.HAS_DLRETX)) == (DataStatus.HAS_RETX | DataStatus.HAS_DLRETX))
        .then((1.0 + pl.col("RETX")) * (1.0 + pl.col("DLRETX")) - 1.0)
        .when((pl.col("prov_flags") & DataStatus.HAS_DLRETX) > 0)
        .then(pl.col("DLRETX"))
        .otherwise(pl.col("RETX"))
        .alias("FINAL_RETX"),
        pl.when((pl.col("prov_flags") & DataStatus.HAS_DLPRC) > 0)
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

    status_dtype = pl.UInt16
    comp_hist_ok = pl.lit(int(DataStatus.CompHist_OK), dtype=status_dtype)
    sec_hist_ok = pl.lit(int(DataStatus.SecHist_OK), dtype=status_dtype)
    status_none = pl.lit(int(DataStatus.NONE), dtype=status_dtype)

    good_keys = price_lf.filter(pl.col("KYGVKEY_final").is_not_null() & pl.col("LIID").is_not_null())

    sec_hist = sec_hist_lf.select(
        "KYGVKEY", "KYIID",
        pl.col("HSCHGDT").alias("HIST_START_DATE_SEC"),
        pl.col("HSCHGENDDT").alias("HCHGENDDT_SEC"),
        "HTPCI", "HEXCNTRY",
    ).sort("KYGVKEY", "KYIID", "HIST_START_DATE_SEC")

    comp_hist = comp_hist_lf.select(
        "KYGVKEY",
        pl.col("HCHGDT").alias("HIST_START_DATE_COMP"),
        pl.col("HCHGENDDT").alias("HCHGENDDT_COMP"),
        "HCIK", "HSIC", "HNAICS", "HGSUBIND",
    ).sort("KYGVKEY", "HIST_START_DATE_COMP")

    step1 = (
        good_keys.sort("KYGVKEY_final", "LIID", "CALDT")
        .join_asof(
            sec_hist,
            left_on="CALDT",
            right_on="HIST_START_DATE_SEC",
            by_left=["KYGVKEY_final", "LIID"],
            by_right=["KYGVKEY", "KYIID"],
            strategy="backward",
        )
        .with_columns(
            pl.when(pl.col("HCHGENDDT_SEC").is_null()).then(status_none)
            .when(pl.col("CALDT") > pl.col("HCHGENDDT_SEC")).then(status_none)
            .otherwise(sec_hist_ok).alias("join2_status"),
        )
        .select(pl.all().exclude(["KYGVKEY", "KYIID"]))
    )
    step1.sink_parquet(temp_path, compression="zstd")

    step1_loaded = pl.scan_parquet(temp_path)

    step2 = (
        step1_loaded
        .sort("KYGVKEY_final", "CALDT")
        .join_asof(
            comp_hist,
            left_on="CALDT",
            right_on="HIST_START_DATE_COMP",
            by_left="KYGVKEY_final",
            by_right="KYGVKEY",
            strategy="backward",
        )
        .with_columns(
            pl.when(pl.col("HCHGENDDT_COMP").is_null()).then(status_none)
            .when(pl.col("CALDT") > pl.col("HCHGENDDT_COMP")).then(status_none)
            .otherwise(comp_hist_ok).alias("join1_status"),
        )
        .select(pl.all().exclude(["KYGVKEY"]))
    )

    bad_keys_lf = price_lf.filter(
        pl.col("KYGVKEY_final").is_null() | pl.col("LIID").is_null()
    )

    final_cols = step2.collect_schema().names()
    hic_schema = comp_hist_lf.collect_schema()
    hcik_dtype = hic_schema.get("HCIK", pl.String)

    bad_rows_processed = (
        bad_keys_lf.with_columns(
            pl.lit(None).cast(pl.Date).alias("HIST_START_DATE_COMP"),
            pl.lit(None).cast(pl.Date).alias("HCHGENDDT_COMP"),
            pl.lit(None, dtype=hcik_dtype).alias("HCIK"),
            pl.lit(None).cast(pl.Int32).alias("HSIC"),
            pl.lit(None).cast(pl.String).alias("HNAICS"),
            pl.lit(None).cast(pl.String).alias("HGSUBIND"),
            pl.lit(None).cast(pl.Date).alias("HIST_START_DATE_SEC"),
            pl.lit(None).cast(pl.Date).alias("HCHGENDDT_SEC"),
            pl.lit(None).cast(pl.String).alias("HTPCI"),
            pl.lit(None).cast(pl.String).alias("HEXCNTRY"),
            pl.when(pl.col("KYGVKEY_final").is_null())
            .then(status_none)
            .otherwise(comp_hist_ok)
            .alias("join1_status"),
            status_none.alias("join2_status"),
        )
        .select(final_cols)
    )

    final_output = pl.concat(
        [step2, bad_rows_processed],
        how="vertical_relaxed",
    )

    final_output.sink_parquet(final_path, compression="zstd")

    return final_path
