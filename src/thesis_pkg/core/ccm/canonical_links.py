from __future__ import annotations

import polars as pl


CANONICAL_LINK_COLUMNS: tuple[str, ...] = (
    "cik_10",
    "gvkey",
    "kypermno",
    "lpermco",
    "liid",
    "valid_start",
    "valid_end",
    "link_start",
    "link_end",
    "cik_start",
    "cik_end",
    "linktype",
    "linkprim",
    "link_rank_raw",
    "link_rank_effective",
    "link_quality",
    "link_source",
    "source_priority",
    "row_quality_tier",
    "has_window",
    "is_sparse_fallback",
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


def _normalized_cik10_expr(col_name: str) -> pl.Expr:
    digits = (
        pl.col(col_name)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.replace_all(r"\.0$", "")
        .str.replace_all(r"\D", "")
    )
    return pl.when(digits.str.len_chars() > 0).then(digits.str.zfill(10)).otherwise(None)


def _open_max(left: str, right: str) -> pl.Expr:
    return (
        pl.when(pl.col(left).is_null())
        .then(pl.col(right))
        .when(pl.col(right).is_null())
        .then(pl.col(left))
        .otherwise(pl.max_horizontal(pl.col(left), pl.col(right)))
    )


def _open_min(left: str, right: str) -> pl.Expr:
    return (
        pl.when(pl.col(left).is_null())
        .then(pl.col(right))
        .when(pl.col(right).is_null())
        .then(pl.col(left))
        .otherwise(pl.min_horizontal(pl.col(left), pl.col(right)))
    )


def _linktype_bucket_expr() -> pl.Expr:
    linktype = pl.col("linktype").cast(pl.Utf8, strict=False).str.to_uppercase()
    return (
        pl.when(linktype == pl.lit("LC"))
        .then(pl.lit(0, dtype=pl.Int32))
        .when(linktype == pl.lit("LU"))
        .then(pl.lit(1, dtype=pl.Int32))
        .when(linktype == pl.lit("LS"))
        .then(pl.lit(2, dtype=pl.Int32))
        .when(linktype == pl.lit("LX"))
        .then(pl.lit(3, dtype=pl.Int32))
        .when(linktype == pl.lit("LD"))
        .then(pl.lit(4, dtype=pl.Int32))
        .when(linktype == pl.lit("LN"))
        .then(pl.lit(5, dtype=pl.Int32))
        .otherwise(pl.lit(9, dtype=pl.Int32))
    )


def _linkprim_bucket_expr() -> pl.Expr:
    linkprim = pl.col("linkprim").cast(pl.Utf8, strict=False).str.to_uppercase()
    return (
        pl.when(linkprim == pl.lit("P"))
        .then(pl.lit(0, dtype=pl.Int32))
        .when(linkprim == pl.lit("C"))
        .then(pl.lit(1, dtype=pl.Int32))
        .when(linkprim == pl.lit("N"))
        .then(pl.lit(2, dtype=pl.Int32))
        .when(linkprim == pl.lit("J"))
        .then(pl.lit(3, dtype=pl.Int32))
        .otherwise(pl.lit(9, dtype=pl.Int32))
    )


def _normalize_linkhistory(linkhistory_lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = linkhistory_lf.collect_schema()
    gvkey_col = _resolve_first_existing(schema, ("KYGVKEY", "gvkey", "GVKEY"), "linkhistory")
    permno_col = _resolve_first_existing(schema, ("LPERMNO", "lpermno", "KYPERMNO"), "linkhistory")
    permco_col = _resolve_first_existing(schema, ("LPERMCO", "lpermco"), "linkhistory")
    liid_col = _resolve_first_existing(schema, ("LIID", "liid"), "linkhistory")
    linktype_col = _resolve_first_existing(schema, ("LINKTYPE", "linktype"), "linkhistory")
    linkprim_col = _resolve_first_existing(schema, ("LINKPRIM", "linkprim"), "linkhistory")
    linkdt_col = _resolve_first_existing(schema, ("LINKDT", "linkdt"), "linkhistory")
    linkenddt_col = _resolve_first_existing(schema, ("LINKENDDT", "linkenddt"), "linkhistory")

    return linkhistory_lf.select(
        pl.col(gvkey_col).cast(pl.Utf8, strict=False).alias("gvkey"),
        pl.col(permno_col).cast(pl.Int32, strict=False).alias("kypermno"),
        pl.col(permco_col).cast(pl.Int32, strict=False).alias("lpermco"),
        pl.col(liid_col).cast(pl.Utf8, strict=False).alias("liid"),
        pl.col(linktype_col).cast(pl.Utf8, strict=False).alias("linktype"),
        pl.col(linkprim_col).cast(pl.Utf8, strict=False).alias("linkprim"),
        pl.col(linkdt_col).cast(pl.Date, strict=False).alias("link_start"),
        pl.col(linkenddt_col).cast(pl.Date, strict=False).alias("link_end"),
        pl.lit(None, dtype=pl.Int32).alias("link_rank_raw"),
        pl.lit("linkhistory").alias("link_source"),
        pl.lit(1, dtype=pl.Int32).alias("source_priority"),
        pl.lit(False).alias("is_sparse_fallback"),
    )


def _normalize_linkfiscal(linkfiscal_lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = linkfiscal_lf.collect_schema()
    gvkey_col = _resolve_first_existing(schema, ("KYGVKEY", "gvkey", "GVKEY"), "linkfiscalperiodall")
    permno_col = _resolve_first_existing(schema, ("lpermno", "LPERMNO", "kypermno"), "linkfiscalperiodall")
    permco_col = _resolve_first_existing(schema, ("lpermco", "LPERMCO"), "linkfiscalperiodall")
    liid_col = _resolve_first_existing(schema, ("liid", "LIID"), "linkfiscalperiodall")
    linktype_col = _resolve_first_existing(schema, ("linktype", "LINKTYPE"), "linkfiscalperiodall")
    linkprim_col = _resolve_first_existing(schema, ("linkprim", "LINKPRIM"), "linkfiscalperiodall")
    linkrank_col = _resolve_first_existing(schema, ("linkrank", "LINKRANK"), "linkfiscalperiodall")
    linkdt_col = _resolve_first_existing(schema, ("linkdt", "LINKDT"), "linkfiscalperiodall")
    linkenddt_col = _resolve_first_existing(schema, ("linkenddt", "LINKENDDT"), "linkfiscalperiodall")
    fiscal_start_col = _resolve_first_existing(
        schema,
        ("FiscalPeriodCRSPStartDt", "fiscalperiodcrspstartdt"),
        "linkfiscalperiodall",
    )
    fiscal_end_col = _resolve_first_existing(
        schema,
        ("FiscalPeriodCRSPEndDt", "fiscalperiodcrspenddt"),
        "linkfiscalperiodall",
    )

    sparse_expr = (
        pl.col(liid_col).is_null()
        & pl.col(linktype_col).is_null()
        & pl.col(linkprim_col).is_null()
        & pl.col(linkdt_col).is_null()
        & pl.col(linkenddt_col).is_null()
    )

    return linkfiscal_lf.select(
        pl.col(gvkey_col).cast(pl.Utf8, strict=False).alias("gvkey"),
        pl.col(permno_col).cast(pl.Int32, strict=False).alias("kypermno"),
        pl.col(permco_col).cast(pl.Int32, strict=False).alias("lpermco"),
        pl.col(liid_col).cast(pl.Utf8, strict=False).alias("liid"),
        pl.col(linktype_col).cast(pl.Utf8, strict=False).alias("linktype"),
        pl.col(linkprim_col).cast(pl.Utf8, strict=False).alias("linkprim"),
        pl.coalesce(
            [
                pl.col(linkdt_col).cast(pl.Date, strict=False),
                pl.col(fiscal_start_col).cast(pl.Date, strict=False),
            ]
        ).alias("link_start"),
        pl.coalesce(
            [
                pl.col(linkenddt_col).cast(pl.Date, strict=False),
                pl.col(fiscal_end_col).cast(pl.Date, strict=False),
            ]
        ).alias("link_end"),
        pl.col(linkrank_col).cast(pl.Int32, strict=False).alias("link_rank_raw"),
        pl.lit("linkfiscalperiodall").alias("link_source"),
        pl.lit(2, dtype=pl.Int32).alias("source_priority"),
        sparse_expr.alias("is_sparse_fallback"),
    )


def _build_cik_windows(
    companyhistory_lf: pl.LazyFrame,
    companydescription_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    companyhistory_schema = companyhistory_lf.collect_schema()
    companydescription_schema = companydescription_lf.collect_schema()

    hist_gvkey_col = _resolve_first_existing(companyhistory_schema, ("KYGVKEY", "gvkey", "GVKEY"), "companyhistory")
    hist_cik_col = _resolve_first_existing(companyhistory_schema, ("HCIK", "cik", "CIK"), "companyhistory")
    hist_start_col = _resolve_first_existing(companyhistory_schema, ("HCHGDT", "hchgdt"), "companyhistory")
    hist_end_col = _resolve_first_existing(companyhistory_schema, ("HCHGENDDT", "hchgenddt"), "companyhistory")

    desc_gvkey_col = _resolve_first_existing(
        companydescription_schema,
        ("KYGVKEY", "gvkey", "GVKEY"),
        "companydescription",
    )
    desc_cik_col = _resolve_first_existing(companydescription_schema, ("CIK", "cik"), "companydescription")

    hist_windows = (
        companyhistory_lf.select(
            pl.col(hist_gvkey_col).cast(pl.Utf8, strict=False).alias("gvkey"),
            _normalized_cik10_expr(hist_cik_col).alias("cik_10"),
            pl.col(hist_start_col).cast(pl.Date, strict=False).alias("cik_start"),
            pl.col(hist_end_col).cast(pl.Date, strict=False).alias("cik_end"),
        )
        .drop_nulls(subset=["gvkey", "cik_10"])
        .unique(subset=["gvkey", "cik_10", "cik_start", "cik_end"])
    )

    hist_gvkeys = hist_windows.select("gvkey").unique()

    desc_windows = (
        companydescription_lf.select(
            pl.col(desc_gvkey_col).cast(pl.Utf8, strict=False).alias("gvkey"),
            _normalized_cik10_expr(desc_cik_col).alias("cik_10"),
        )
        .drop_nulls(subset=["gvkey", "cik_10"])
        .join(hist_gvkeys, on="gvkey", how="anti")
        .group_by("gvkey")
        .agg(pl.col("cik_10").mode().first().alias("cik_10"))
        .with_columns(
            pl.lit(None, dtype=pl.Date).alias("cik_start"),
            pl.lit(None, dtype=pl.Date).alias("cik_end"),
        )
    )

    return pl.concat([hist_windows, desc_windows], how="vertical_relaxed")


def build_canonical_link_table(
    linkhistory_lf: pl.LazyFrame,
    linkfiscalperiodall_lf: pl.LazyFrame,
    companyhistory_lf: pl.LazyFrame,
    companydescription_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build the canonical SEC<->CCM time-sliced link table."""
    linkhistory = _normalize_linkhistory(linkhistory_lf)
    linkfiscal = _normalize_linkfiscal(linkfiscalperiodall_lf)

    dedupe_cols = (
        "gvkey",
        "kypermno",
        "lpermco",
        "liid",
        "linktype",
        "linkprim",
        "link_start",
        "link_end",
        "link_rank_raw",
        "link_source",
        "is_sparse_fallback",
    )
    links_deduped = pl.concat([linkhistory, linkfiscal], how="vertical_relaxed").unique(subset=dedupe_cols)

    cik_windows = _build_cik_windows(companyhistory_lf, companydescription_lf)
    # Keep all link rows even when CIK metadata is missing for non-SEC usage.
    # LEFT join avoids dropping valid GVKEY<->PERMNO windows without CIK.
    merged = links_deduped.join(cik_windows, on="gvkey", how="left")

    linktype_bucket = _linktype_bucket_expr()
    linkprim_bucket = _linkprim_bucket_expr()
    linktype_upper = pl.col("linktype").cast(pl.Utf8, strict=False).str.to_uppercase()
    linkprim_upper = pl.col("linkprim").cast(pl.Utf8, strict=False).str.to_uppercase()

    out = (
        merged.with_columns(
            _open_max("link_start", "cik_start").alias("valid_start"),
            _open_min("link_end", "cik_end").alias("valid_end"),
            (pl.col("link_start").is_not_null() | pl.col("link_end").is_not_null()).alias("has_window"),
            pl.col("source_priority").cast(pl.Int32, strict=False),
            pl.col("link_rank_raw").cast(pl.Int32, strict=False),
        )
        .filter(
            ~(
                pl.col("valid_start").is_not_null()
                & pl.col("valid_end").is_not_null()
                & (pl.col("valid_start") > pl.col("valid_end"))
            )
        )
        .with_columns(
            pl.when(pl.col("link_rank_raw").is_not_null() & (pl.col("link_rank_raw") >= pl.lit(1)))
            .then(pl.col("link_rank_raw"))
            .otherwise(pl.lit(90, dtype=pl.Int32) + linktype_bucket + linkprim_bucket)
            .cast(pl.Int32)
            .alias("link_rank_effective"),
            pl.when(linktype_upper.is_in(["LC", "LU"]) & (linkprim_upper == pl.lit("P")))
            .then(pl.lit(4.0))
            .when(linkprim_upper == pl.lit("P"))
            .then(pl.lit(3.0))
            .when(linktype_upper.is_in(["LC", "LU"]))
            .then(pl.lit(2.0))
            .when(pl.col("is_sparse_fallback"))
            .then(pl.lit(0.1))
            .otherwise(pl.lit(1.0))
            .cast(pl.Float64)
            .alias("link_quality"),
        )
        .with_columns(
            pl.when(
                (pl.col("kypermno") > pl.lit(0))
                & pl.col("has_window")
                & pl.col("is_sparse_fallback").not_()
                & pl.col("liid").is_not_null()
                & pl.col("linktype").is_not_null()
                & pl.col("linkprim").is_not_null()
            )
            .then(pl.lit(10, dtype=pl.Int32))
            .when(
                (pl.col("kypermno") > pl.lit(0))
                & pl.col("has_window")
                & pl.col("is_sparse_fallback").not_()
            )
            .then(pl.lit(20, dtype=pl.Int32))
            .when(
                (pl.col("kypermno") > pl.lit(0))
                & pl.col("has_window").not_()
                & pl.col("is_sparse_fallback").not_()
            )
            .then(pl.lit(40, dtype=pl.Int32))
            .otherwise(pl.lit(90, dtype=pl.Int32))
            .alias("row_quality_tier")
        )
        .select(*CANONICAL_LINK_COLUMNS)
        .unique(subset=CANONICAL_LINK_COLUMNS)
    )
    return out


def canonical_link_coverage_metrics(canonical_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Summarize CIK coverage for canonical links, including non-SEC rows."""
    return canonical_lf.select(
        pl.len().alias("rows_total"),
        pl.col("cik_10").is_null().sum().cast(pl.Int64).alias("rows_missing_cik"),
        pl.col("cik_10").is_not_null().sum().cast(pl.Int64).alias("rows_with_cik"),
        pl.col("gvkey").n_unique().cast(pl.Int64).alias("distinct_gvkey_total"),
        pl.when(pl.col("cik_10").is_null()).then(pl.col("gvkey")).otherwise(None).drop_nulls().n_unique()
        .cast(pl.Int64)
        .alias("distinct_gvkey_missing_cik"),
        pl.col("cik_10").drop_nulls().n_unique().cast(pl.Int64).alias("distinct_cik_10"),
    ).with_columns(
        pl.when(pl.col("rows_total") > pl.lit(0))
        .then(pl.col("rows_with_cik").cast(pl.Float64) / pl.col("rows_total").cast(pl.Float64))
        .otherwise(pl.lit(0.0))
        .alias("row_cik_coverage_ratio"),
        pl.when(pl.col("distinct_gvkey_total") > pl.lit(0))
        .then(
            (
                pl.col("distinct_gvkey_total").cast(pl.Float64)
                - pl.col("distinct_gvkey_missing_cik").cast(pl.Float64)
            )
            / pl.col("distinct_gvkey_total").cast(pl.Float64)
        )
        .otherwise(pl.lit(0.0))
        .alias("gvkey_cik_coverage_ratio"),
    )


def normalize_canonical_link_table(link_lf: pl.LazyFrame, *, strict: bool = True) -> pl.LazyFrame:
    """Validate/cast canonical link-table schema for downstream consumers."""
    schema = link_lf.collect_schema()
    required = (
        "cik_10",
        "gvkey",
        "kypermno",
        "valid_start",
        "valid_end",
        "link_start",
        "link_end",
        "cik_start",
        "cik_end",
        "linktype",
        "linkprim",
        "link_rank_raw",
        "link_rank_effective",
        "link_quality",
        "link_source",
        "source_priority",
        "row_quality_tier",
        "has_window",
        "is_sparse_fallback",
    )
    if strict:
        missing = [name for name in required if name not in schema]
        if missing:
            raise ValueError(
                "canonical link table missing required columns in strict mode: "
                f"{missing}"
            )

    liid_expr = pl.col("liid").cast(pl.Utf8, strict=False) if "liid" in schema else pl.lit(None, dtype=pl.Utf8)
    lpermco_expr = pl.col("lpermco").cast(pl.Int32, strict=False) if "lpermco" in schema else pl.lit(None, dtype=pl.Int32)
    link_rank_raw_expr = (
        pl.col("link_rank_raw").cast(pl.Int32, strict=False)
        if "link_rank_raw" in schema
        else pl.lit(None, dtype=pl.Int32)
    )
    link_rank_effective_expr = (
        pl.col("link_rank_effective").cast(pl.Int32, strict=False)
        if "link_rank_effective" in schema
        else pl.lit(None, dtype=pl.Int32)
    )
    link_quality_expr = (
        pl.col("link_quality").cast(pl.Float64, strict=False)
        if "link_quality" in schema
        else pl.lit(0.0, dtype=pl.Float64)
    )
    source_priority_expr = (
        pl.col("source_priority").cast(pl.Int32, strict=False)
        if "source_priority" in schema
        else pl.lit(99, dtype=pl.Int32)
    )
    row_quality_tier_expr = (
        pl.col("row_quality_tier").cast(pl.Int32, strict=False)
        if "row_quality_tier" in schema
        else pl.lit(90, dtype=pl.Int32)
    )
    has_window_expr = (
        pl.col("has_window").cast(pl.Boolean, strict=False)
        if "has_window" in schema
        else (pl.col("link_start").is_not_null() | pl.col("link_end").is_not_null())
    )
    sparse_expr = (
        pl.col("is_sparse_fallback").cast(pl.Boolean, strict=False)
        if "is_sparse_fallback" in schema
        else pl.lit(False)
    )
    source_expr = (
        pl.col("link_source").cast(pl.Utf8, strict=False)
        if "link_source" in schema
        else pl.lit("unknown", dtype=pl.Utf8)
    )

    fallback_rank = pl.lit(90, dtype=pl.Int32) + _linktype_bucket_expr() + _linkprim_bucket_expr()

    return (
        link_lf.select(
            _normalized_cik10_expr("cik_10").alias("cik_10"),
            pl.col("gvkey").cast(pl.Utf8, strict=False).alias("gvkey"),
            pl.col("kypermno").cast(pl.Int32, strict=False).alias("kypermno"),
            lpermco_expr.alias("lpermco"),
            liid_expr.alias("liid"),
            pl.col("valid_start").cast(pl.Date, strict=False).alias("valid_start"),
            pl.col("valid_end").cast(pl.Date, strict=False).alias("valid_end"),
            pl.col("link_start").cast(pl.Date, strict=False).alias("link_start"),
            pl.col("link_end").cast(pl.Date, strict=False).alias("link_end"),
            pl.col("cik_start").cast(pl.Date, strict=False).alias("cik_start"),
            pl.col("cik_end").cast(pl.Date, strict=False).alias("cik_end"),
            pl.col("linktype").cast(pl.Utf8, strict=False).alias("linktype"),
            pl.col("linkprim").cast(pl.Utf8, strict=False).alias("linkprim"),
            link_rank_raw_expr.alias("link_rank_raw"),
            link_rank_effective_expr.alias("link_rank_effective"),
            link_quality_expr.alias("link_quality"),
            source_expr.alias("link_source"),
            source_priority_expr.alias("source_priority"),
            row_quality_tier_expr.alias("row_quality_tier"),
            has_window_expr.alias("has_window"),
            sparse_expr.alias("is_sparse_fallback"),
        )
        .drop_nulls(subset=["cik_10", "gvkey", "kypermno"])
        .filter(pl.col("cik_10").str.contains(r"^\d{10}$").fill_null(False))
        .with_columns(
            pl.when(pl.col("link_rank_effective").is_not_null() & (pl.col("link_rank_effective") >= pl.lit(1)))
            .then(pl.col("link_rank_effective"))
            .otherwise(
                pl.when(pl.col("link_rank_raw").is_not_null() & (pl.col("link_rank_raw") >= pl.lit(1)))
                .then(pl.col("link_rank_raw"))
                .otherwise(fallback_rank)
            )
            .cast(pl.Int32)
            .alias("link_rank_effective"),
            pl.col("source_priority").fill_null(99).cast(pl.Int32),
            pl.col("row_quality_tier").fill_null(90).cast(pl.Int32),
            pl.col("has_window").fill_null(False).cast(pl.Boolean),
            pl.col("is_sparse_fallback").fill_null(False).cast(pl.Boolean),
            pl.col("link_quality").fill_null(0.0).cast(pl.Float64),
        )
        .unique(subset=CANONICAL_LINK_COLUMNS)
        .select(*CANONICAL_LINK_COLUMNS)
    )
