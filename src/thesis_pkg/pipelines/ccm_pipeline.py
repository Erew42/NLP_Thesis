from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_pkg.core.ccm.transforms import DataStatus, STATUS_DTYPE, _ensure_data_status, _update_data_status


def merge_histories(
    price_lf: pl.LazyFrame,
    sec_hist_lf: pl.LazyFrame,
    comp_hist_lf: pl.LazyFrame,
    output_dir: Path,
    temp_name: str = "temp_step1_sec_merge.parquet",
    final_name: str = "final_flagged_data.parquet",
    verbose: int = 0,
) -> Path:
    """RAM-safe merge of security and company histories without dropping bad rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_dir / temp_name
    final_path = output_dir / final_name

    def _debug_concat(label: str, left: pl.LazyFrame, right: pl.LazyFrame) -> None:
        """Optional schema diff to diagnose concat failures."""
        if not verbose:
            return
        left_schema = left.collect_schema()
        right_schema = right.collect_schema()
        left_only = sorted(set(left_schema) - set(right_schema))
        right_only = sorted(set(right_schema) - set(left_schema))
        dtype_mismatch = [
            (name, str(left_schema[name]), str(right_schema[name]))
            for name in set(left_schema) & set(right_schema)
            if left_schema[name] != right_schema[name]
        ]
        print(
            f"[merge_histories][{label}] cols_left={len(left_schema)} cols_right={len(right_schema)} "
            f"left_only={left_only} right_only={right_only} dtype_mismatch={dtype_mismatch}"
        )

    def _align_to_schema(lf: pl.LazyFrame, target: pl.Schema) -> pl.LazyFrame:
        """Project/append columns so frame matches target schema exactly."""
        have = lf.collect_schema()
        exprs: list[pl.Expr] = []
        for name, dtype in target.items():
            if name in have:
                exprs.append(pl.col(name).cast(dtype, strict=False))
            else:
                exprs.append(pl.lit(None, dtype=dtype).alias(name))
        return lf.select(exprs)

    sec_schema = sec_hist_lf.collect_schema()
    comp_schema = comp_hist_lf.collect_schema()
    ky_dtype = sec_schema.get("KYGVKEY", comp_schema.get("KYGVKEY", pl.Utf8))
    liid_dtype = sec_schema.get("KYIID", pl.Utf8)
    htpi_dtype = sec_schema.get("HTPCI", pl.String)
    hexcntry_dtype = sec_schema.get("HEXCNTRY", pl.String)

    base = _ensure_data_status(price_lf).with_columns(
        pl.col("CALDT").cast(pl.Date, strict=False).alias("CALDT"),
        pl.col("data_status").cast(STATUS_DTYPE),
    )

    base_schema = base.collect_schema()
    cast_cols: list[pl.Expr] = []
    if "KYGVKEY_final" in base_schema:
        cast_cols.append(pl.col("KYGVKEY_final").cast(ky_dtype))
    if "LIID" in base_schema:
        cast_cols.append(pl.col("LIID").cast(liid_dtype))
    if cast_cols:
        base = base.with_columns(cast_cols)

    sec_hist = sec_hist_lf.select(
        pl.col("KYGVKEY").cast(ky_dtype).alias("KYGVKEY"),
        pl.col("KYIID").cast(liid_dtype).alias("KYIID"),
        pl.col("HSCHGDT").cast(pl.Date, strict=False).alias("HIST_START_DATE_SEC"),
        pl.col("HSCHGENDDT").cast(pl.Date, strict=False).alias("HCHGENDDT_SEC"),
        pl.col("HTPCI").cast(htpi_dtype).alias("HTPCI"),
        pl.col("HEXCNTRY").cast(hexcntry_dtype).alias("HEXCNTRY"),
    ).sort("KYGVKEY", "KYIID", "HIST_START_DATE_SEC")

    comp_hist = comp_hist_lf.select(
        pl.col("KYGVKEY").cast(ky_dtype).alias("KYGVKEY"),
        pl.col("HCHGDT").cast(pl.Date, strict=False).alias("HIST_START_DATE_COMP"),
        pl.col("HCHGENDDT").cast(pl.Date, strict=False).alias("HCHGENDDT_COMP"),
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
            # Both sides are explicitly sorted above by (group keys, asof key).
            # Polars cannot verify grouped sortedness and otherwise emits a warning.
            check_sortedness=False,
        )
        .select(pl.all().exclude(["KYGVKEY", "KYIID"]))
    )

    sec_matched = pl.col("HIST_START_DATE_SEC").is_not_null()
    sec_valid = sec_matched & (pl.col("HCHGENDDT_SEC").is_null() | (pl.col("CALDT") <= pl.col("HCHGENDDT_SEC")))
    sec_stale = sec_matched & pl.col("HCHGENDDT_SEC").is_not_null() & (pl.col("CALDT") > pl.col("HCHGENDDT_SEC"))

    sec_status = _update_data_status(
        pl.col("data_status"),
        static_flags=DataStatus.SECHIST_CAN_ATTEMPT | DataStatus.SECHIST_ATTEMPTED,
        conditional_flags=(
            (sec_matched, DataStatus.SECHIST_MATCHED),
            (sec_valid, DataStatus.SECHIST_VALID),
            (sec_stale, DataStatus.SECHIST_STALE),
            (sec_matched.not_(), DataStatus.SECHIST_NO_MATCH),
        ),
    ).alias("data_status")

    sec_joined = sec_joined.with_columns(
        sec_status,
        pl.when(sec_valid).then(pl.col("HTPCI")).otherwise(pl.lit(None, dtype=htpi_dtype)).alias("HTPCI"),
        pl.when(sec_valid).then(pl.col("HEXCNTRY")).otherwise(pl.lit(None, dtype=hexcntry_dtype)).alias("HEXCNTRY"),
        pl.when(sec_valid).then(pl.col("HIST_START_DATE_SEC")).otherwise(pl.lit(None).cast(pl.Date)).alias("HIST_START_DATE_SEC"),
        pl.when(sec_valid).then(pl.col("HCHGENDDT_SEC")).otherwise(pl.lit(None).cast(pl.Date)).alias("HCHGENDDT_SEC"),
    )

    sec_missing = base.filter(sec_keys.not_()).with_columns(
        pl.col("data_status").cast(STATUS_DTYPE),
        pl.lit(None).cast(pl.Date).alias("HIST_START_DATE_SEC"),
        pl.lit(None).cast(pl.Date).alias("HCHGENDDT_SEC"),
        pl.lit(None, dtype=htpi_dtype).alias("HTPCI"),
        pl.lit(None, dtype=hexcntry_dtype).alias("HEXCNTRY"),
    ).drop("KYGVKEY", strict=False)  # drop raw KYGVKEY to match sec_joined schema

    _debug_concat("sec_concat", sec_joined, sec_missing)
    sec_target = sec_joined.collect_schema()
    sec_combined = pl.concat(
        [_align_to_schema(sec_joined, sec_target), _align_to_schema(sec_missing, sec_target)],
        how="vertical",
    ).with_columns(
        pl.col("data_status").cast(STATUS_DTYPE)
    )
    sec_combined.sink_parquet(temp_path, compression="zstd")

    sec_loaded = pl.scan_parquet(temp_path).with_columns(
        pl.col("data_status").cast(STATUS_DTYPE),
        pl.col("KYGVKEY_final").cast(ky_dtype),
        pl.col("LIID").cast(liid_dtype),
    )

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
            # Both sides are explicitly sorted above by (group key, asof key).
            # Polars cannot verify grouped sortedness and otherwise emits a warning.
            check_sortedness=False,
        )
        .select(pl.all().exclude(["KYGVKEY"]))
    )

    comp_matched = pl.col("HIST_START_DATE_COMP").is_not_null()
    comp_valid = comp_matched & (pl.col("HCHGENDDT_COMP").is_null() | (pl.col("CALDT") <= pl.col("HCHGENDDT_COMP")))
    comp_stale = comp_matched & pl.col("HCHGENDDT_COMP").is_not_null() & (pl.col("CALDT") > pl.col("HCHGENDDT_COMP"))

    comp_status = _update_data_status(
        pl.col("data_status"),
        static_flags=DataStatus.COMPHIST_CAN_ATTEMPT | DataStatus.COMPHIST_ATTEMPTED,
        conditional_flags=(
            (comp_matched, DataStatus.COMPHIST_MATCHED),
            (comp_valid, DataStatus.COMPHIST_VALID),
            (comp_stale, DataStatus.COMPHIST_STALE),
            (comp_matched.not_(), DataStatus.COMPHIST_NO_MATCH),
        ),
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

    _debug_concat("comp_concat", comp_joined, comp_missing)
    comp_target = comp_joined.collect_schema()
    final_output = pl.concat(
        [_align_to_schema(comp_joined, comp_target), _align_to_schema(comp_missing, comp_target)],
        how="vertical",
    ).with_columns(
        pl.col("data_status").cast(STATUS_DTYPE)
    )

    final_output.sink_parquet(final_path, compression="zstd")

    return final_path
