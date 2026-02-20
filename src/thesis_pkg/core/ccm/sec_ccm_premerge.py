from __future__ import annotations

import polars as pl

from thesis_pkg.core.ccm.sec_ccm_contracts import (
    MatchReasonCode,
    PhaseBAlignmentMode,
    PhaseBDailyJoinMode,
    SecCcmJoinSpec,
    SecCcmJoinSpecV1,
    SecCcmJoinSpecV2,
    normalize_sec_ccm_join_spec,
)
from thesis_pkg.core.ccm.transforms import DataStatus, STATUS_DTYPE, _ensure_data_status, _update_data_status


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
        .str.replace_all(r"\D", "")
    )
    return (
        pl.when(digits.str.len_chars() > 0)
        .then(digits.str.zfill(10))
        .otherwise(None)
        .alias(col_name)
    )


def _normalize_link_universe(link_universe_lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = link_universe_lf.collect_schema()
    cik_col = _resolve_first_existing(schema, ("cik_10", "CIK_10", "CIK", "cik"), "link_universe")
    gvkey_col = _resolve_first_existing(schema, ("gvkey", "GVKEY", "KYGVKEY"), "link_universe")
    permno_col = _resolve_first_existing(
        schema,
        ("kypermno", "KYPERMNO", "LPERMNO", "PERMNO"),
        "link_universe",
    )

    link_quality_col = None
    for candidate in ("link_quality", "LINK_QUALITY", "link_score", "score"):
        if candidate in schema:
            link_quality_col = candidate
            break

    link_rank_col = None
    for candidate in ("link_rank", "LINK_RANK", "rank"):
        if candidate in schema:
            link_rank_col = candidate
            break

    selected = link_universe_lf.select(
        _normalized_cik10_expr(cik_col).alias("cik_10"),
        pl.col(gvkey_col).cast(pl.Utf8, strict=False).alias("gvkey"),
        pl.col(permno_col).cast(pl.Int32, strict=False).alias("kypermno"),
        (
            pl.col(link_quality_col).cast(pl.Float64, strict=False)
            if link_quality_col is not None
            else pl.lit(0.0, dtype=pl.Float64)
        ).alias("link_quality"),
        (
            pl.col(link_rank_col).cast(pl.Int32, strict=False)
            if link_rank_col is not None
            else pl.lit(0, dtype=pl.Int32)
        ).alias("link_rank"),
    )

    return (
        selected
        .drop_nulls(subset=["cik_10"])
        .with_columns(
            pl.col("link_quality").fill_null(0.0),
            pl.col("link_rank").fill_null(0).cast(pl.Int32),
        )
        .unique(subset=["cik_10", "gvkey", "kypermno", "link_quality", "link_rank"])
    )


def _requested_alignment_policy_value(join_spec: SecCcmJoinSpec) -> str:
    if isinstance(join_spec, SecCcmJoinSpecV1):
        return join_spec.alignment_policy
    return join_spec.phase_b_alignment_mode.value


def _normalize_daily_join_input(daily_lf: pl.LazyFrame, join_spec: SecCcmJoinSpecV2) -> pl.LazyFrame:
    schema = daily_lf.collect_schema()
    perm_col = _resolve_first_existing(
        schema,
        (join_spec.daily_permno_col, join_spec.daily_permno_col.lower(), "KYPERMNO", "LPERMNO", "PERMNO"),
        "daily join input",
    )
    date_col = _resolve_first_existing(
        schema,
        (join_spec.daily_date_col, join_spec.daily_date_col.lower(), "CALDT", "caldt"),
        "daily join input",
    )

    required_missing = [col for col in join_spec.required_daily_non_null_features if col not in schema]
    if required_missing:
        raise ValueError(f"Daily join input missing required non-null feature columns: {required_missing}")
    selected_features: list[str] = []
    seen_features: set[str] = set()
    for col in (*join_spec.daily_feature_columns, *join_spec.required_daily_non_null_features):
        if col in schema and col not in seen_features:
            selected_features.append(col)
            seen_features.add(col)

    return (
        daily_lf.select(
            pl.col(perm_col).cast(pl.Int32, strict=False).alias("kypermno"),
            pl.col(date_col).cast(pl.Date, strict=False).alias("daily_caldt"),
            *[pl.col(col) for col in selected_features],
        )
        .drop_nulls(subset=["kypermno", "daily_caldt"])
        .unique(subset=["kypermno", "daily_caldt"], keep="first")
    )


def normalize_sec_filings_phase_a(sec_filings_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize SEC filing identifiers for Phase A linking."""
    _require_columns(sec_filings_lf, ("doc_id", "cik_10", "filing_date"), "sec_filings")
    schema = sec_filings_lf.collect_schema()
    acceptance_col_present = "acceptance_datetime" in schema

    normalized = _ensure_data_status(sec_filings_lf).with_columns(
        pl.col("doc_id").cast(pl.Utf8, strict=False),
        _normalized_cik10_expr("cik_10"),
        pl.col("filing_date").cast(pl.Date, strict=False),
        (
            pl.col("acceptance_datetime").cast(pl.Datetime, strict=False)
            if acceptance_col_present
            else pl.lit(None, dtype=pl.Datetime).alias("acceptance_datetime")
        ),
    )

    return normalized.with_columns(
        pl.col("acceptance_datetime").is_not_null().alias("has_acceptance_datetime"),
        pl.col("data_status").cast(STATUS_DTYPE).fill_null(int(DataStatus.NONE)),
    )


def resolve_links_phase_a(sec_norm_lf: pl.LazyFrame, link_universe_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Resolve doc_id to a single gvkey/kypermno candidate with conservative tie handling."""
    sec_base = normalize_sec_filings_phase_a(sec_norm_lf).with_columns(
        (
            pl.col("doc_id").is_not_null()
            & pl.col("cik_10").str.contains(r"^\d{10}$").fill_null(False)
            & pl.col("filing_date").is_not_null()
        ).alias("_valid_input")
    )
    links = _normalize_link_universe(link_universe_lf)

    candidate_scored = (
        sec_base.filter(pl.col("_valid_input"))
        .select("doc_id", "cik_10")
        .join(links, on="cik_10", how="left")
        .drop_nulls(subset=["gvkey", "kypermno"])
        .with_columns(
            pl.col("link_rank").fill_null(0).cast(pl.Int32).alias("_link_rank"),
            pl.col("link_quality").fill_null(0.0).cast(pl.Float64).alias("_link_quality"),
        )
    )

    candidate_counts = candidate_scored.group_by("doc_id").agg(
        pl.len().cast(pl.Int32).alias("link_candidate_count")
    )
    best_rank = candidate_scored.group_by("doc_id").agg(pl.col("_link_rank").min().alias("_best_rank"))
    top_rank = candidate_scored.join(best_rank, on="doc_id", how="inner").filter(
        pl.col("_link_rank") == pl.col("_best_rank")
    )
    best_quality = top_rank.group_by("doc_id").agg(pl.col("_link_quality").max().alias("_best_quality"))
    top_candidates = top_rank.join(best_quality, on="doc_id", how="inner").filter(
        pl.col("_link_quality") == pl.col("_best_quality")
    )
    tie_stats = top_candidates.group_by("doc_id").agg(
        pl.len().cast(pl.Int32).alias("top_rank_tie_count"),
        pl.col("gvkey").first().alias("_top_gvkey"),
        pl.col("kypermno").first().cast(pl.Int32).alias("_top_kypermno"),
        pl.col("_link_quality").first().cast(pl.Float64).alias("_top_link_quality"),
    )

    enriched = (
        sec_base
        .join(candidate_counts, on="doc_id", how="left")
        .join(tie_stats, on="doc_id", how="left")
        .with_columns(
            pl.col("link_candidate_count").fill_null(0).cast(pl.Int32),
            pl.col("top_rank_tie_count").fill_null(0).cast(pl.Int32),
        )
    )

    phase_a_reason = (
        pl.when(pl.col("_valid_input").not_())
        .then(pl.lit(MatchReasonCode.BAD_INPUT.value))
        .when(pl.col("link_candidate_count") == 0)
        .then(pl.lit(MatchReasonCode.CIK_NOT_IN_LINK_UNIVERSE.value))
        .when(pl.col("top_rank_tie_count") > 1)
        .then(pl.lit(MatchReasonCode.AMBIGUOUS_LINK.value))
        .otherwise(pl.lit(MatchReasonCode.OK.value))
    )

    out = enriched.with_columns(
        phase_a_reason.alias("phase_a_reason_code"),
        phase_a_reason.alias("match_reason_code"),
        pl.when(phase_a_reason == pl.lit(MatchReasonCode.OK.value))
        .then(pl.col("_top_gvkey"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("gvkey"),
        pl.when(phase_a_reason == pl.lit(MatchReasonCode.OK.value))
        .then(pl.col("_top_kypermno"))
        .otherwise(pl.lit(None, dtype=pl.Int32))
        .alias("kypermno"),
        pl.when(phase_a_reason == pl.lit(MatchReasonCode.OK.value))
        .then(pl.col("_top_link_quality"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("link_quality"),
    )

    status = _update_data_status(
        pl.col("data_status"),
        static_flags=DataStatus.SEC_CCM_PHASE_A_ATTEMPTED,
        conditional_flags=(
            (pl.col("phase_a_reason_code") == pl.lit(MatchReasonCode.BAD_INPUT.value), DataStatus.SEC_CCM_BAD_INPUT),
            (
                pl.col("phase_a_reason_code") == pl.lit(MatchReasonCode.CIK_NOT_IN_LINK_UNIVERSE.value),
                DataStatus.SEC_CCM_CIK_NOT_IN_LINK_UNIVERSE,
            ),
            (
                pl.col("phase_a_reason_code") == pl.lit(MatchReasonCode.AMBIGUOUS_LINK.value),
                DataStatus.SEC_CCM_AMBIGUOUS_LINK,
            ),
            (pl.col("phase_a_reason_code") == pl.lit(MatchReasonCode.OK.value), DataStatus.SEC_CCM_LINKED_OK),
            (pl.col("has_acceptance_datetime"), DataStatus.SEC_CCM_HAS_ACCEPTANCE_DATETIME),
        ),
    ).alias("data_status")

    return out.with_columns(status).drop(
        "_valid_input",
        "_top_gvkey",
        "_top_kypermno",
        "_top_link_quality",
        strict=False,
    )


def align_doc_dates_phase_b(
    links_doc_lf: pl.LazyFrame,
    trading_calendar_lf: pl.LazyFrame,
    join_spec: SecCcmJoinSpec,
) -> pl.LazyFrame:
    """Compute doc-date alignment at doc grain under the configured Phase B policy."""
    join_spec_v2 = normalize_sec_ccm_join_spec(join_spec)
    alignment_mode = join_spec_v2.phase_b_alignment_mode
    requested_alignment_policy = _requested_alignment_policy_value(join_spec)
    _require_columns(
        links_doc_lf,
        ("doc_id", "filing_date", "kypermno", "phase_a_reason_code", "data_status"),
        "phase A links",
    )

    trading_schema = trading_calendar_lf.collect_schema()
    date_col = _resolve_first_existing(
        trading_schema,
        (
            join_spec_v2.daily_date_col,
            join_spec_v2.daily_date_col.lower(),
            "CALDT",
            "caldt",
            "TRADING_DATE",
        ),
        "trading calendar",
    )

    trading_calendar = (
        trading_calendar_lf.select(pl.col(date_col).cast(pl.Date, strict=False).alias("_calendar_date"))
        .drop_nulls(subset=["_calendar_date"])
        .unique()
        .sort("_calendar_date")
    )

    if alignment_mode == PhaseBAlignmentMode.NEXT_TRADING_DAY_STRICT:
        alignment_anchor_expr = pl.when(pl.col("filing_date").is_not_null()).then(
            pl.col("filing_date") + pl.duration(days=1)
        )
    else:
        alignment_anchor_expr = pl.when(pl.col("filing_date").is_not_null()).then(pl.col("filing_date"))

    base = _ensure_data_status(links_doc_lf).with_columns(
        pl.col("filing_date").cast(pl.Date, strict=False),
        pl.col("kypermno").cast(pl.Int32, strict=False),
        alignment_anchor_expr.otherwise(pl.lit(None)).cast(pl.Date).alias("_alignment_anchor_date"),
        pl.lit(requested_alignment_policy).alias("alignment_policy_requested"),
        pl.lit(alignment_mode.value).alias("alignment_policy_effective"),
    )

    if alignment_mode == PhaseBAlignmentMode.FILING_DATE_EXACT_ONLY:
        exact_calendar = trading_calendar.select(
            pl.col("_calendar_date").alias("_alignment_anchor_date"),
            pl.lit(True).alias("_is_trading_date"),
        )
        aligned = (
            base.join(exact_calendar, on="_alignment_anchor_date", how="left")
            .with_columns(
                pl.when(pl.col("kypermno").is_not_null() & pl.col("_is_trading_date").fill_null(False))
                .then(pl.col("_alignment_anchor_date"))
                .otherwise(pl.lit(None, dtype=pl.Date))
                .alias("aligned_caldt")
            )
        )
    else:
        aligned = (
            base.sort("_alignment_anchor_date")
            .join_asof(
                trading_calendar,
                left_on="_alignment_anchor_date",
                right_on="_calendar_date",
                strategy="forward",
            )
            .rename({"_calendar_date": "aligned_caldt"})
            .with_columns(
                pl.when(pl.col("kypermno").is_not_null())
                .then(pl.col("aligned_caldt"))
                .otherwise(pl.lit(None, dtype=pl.Date))
                .alias("aligned_caldt")
            )
        )

    aligned = aligned.with_columns(
        pl.when(pl.col("aligned_caldt").is_not_null() & pl.col("filing_date").is_not_null())
        .then((pl.col("aligned_caldt") - pl.col("filing_date")).dt.total_days().cast(pl.Int32))
        .otherwise(pl.lit(None, dtype=pl.Int32))
        .alias("alignment_lag_days")
    )

    phase_a_ok = pl.col("phase_a_reason_code") == pl.lit(MatchReasonCode.OK.value)
    alignment_attempted = phase_a_ok & pl.col("kypermno").is_not_null()
    aligned_ok = alignment_attempted & pl.col("aligned_caldt").is_not_null()

    status = _update_data_status(
        pl.col("data_status"),
        conditional_flags=(
            (alignment_attempted, DataStatus.SEC_CCM_PHASE_B_ALIGNMENT_ATTEMPTED),
            (aligned_ok, DataStatus.SEC_CCM_PHASE_B_ALIGNED),
        ),
    ).alias("data_status")

    return aligned.with_columns(status).drop("_alignment_anchor_date", "_is_trading_date", strict=False)


def join_daily_phase_b(
    aligned_doc_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    join_spec: SecCcmJoinSpec,
) -> pl.LazyFrame:
    """
    Join docs to CRSP daily (or merged daily panel) keyed by (kypermno, caldt)
    using the configured Phase B daily-join mode.
    """
    join_spec_v2 = normalize_sec_ccm_join_spec(join_spec)
    _require_columns(
        aligned_doc_lf,
        ("doc_id", "kypermno", "aligned_caldt", "phase_a_reason_code", "data_status"),
        "Phase B aligned docs",
    )

    if not join_spec_v2.daily_join_enabled:
        return aligned_doc_lf.with_columns(
            pl.lit(None, dtype=pl.Date).alias("daily_join_caldt"),
            pl.lit(None, dtype=pl.Int32).alias("daily_join_lag_days"),
            pl.lit(False).alias("daily_lag_gate_rejected"),
            pl.lit(None, dtype=pl.Date).alias("key_min_caldt"),
            pl.lit(None, dtype=pl.Date).alias("key_max_caldt"),
            pl.lit(False).alias("_has_daily_row"),
            pl.lit(False).alias("_has_usable_daily_row"),
        )

    daily = _normalize_daily_join_input(daily_lf, join_spec_v2)
    required_features = tuple(join_spec_v2.required_daily_non_null_features)
    if required_features:
        daily_schema = daily.collect_schema()
        missing_required = [col for col in required_features if col not in daily_schema]
        if missing_required:
            raise ValueError(f"Daily join normalized input missing required non-null feature columns: {missing_required}")
    coverage = daily.group_by("kypermno").agg(
        pl.col("daily_caldt").min().alias("key_min_caldt"),
        pl.col("daily_caldt").max().alias("key_max_caldt"),
    )

    left_base = (
        aligned_doc_lf
        .with_columns(pl.col("kypermno").cast(pl.Int32, strict=False))
        .join(coverage, on="kypermno", how="left")
    )
    if join_spec_v2.phase_b_daily_join_mode == PhaseBDailyJoinMode.ASOF_FORWARD:
        left_for_asof = left_base.sort("kypermno", "aligned_caldt")
        right_for_asof = daily.sort("kypermno", "daily_caldt")
        joined = (
            left_for_asof
            .join_asof(
                right_for_asof,
                left_on="aligned_caldt",
                right_on="daily_caldt",
                by="kypermno",
                strategy="forward",
                # We explicitly sort both sides by (group key, asof key) above.
                # Polars cannot validate grouped sortedness and otherwise emits a warning.
                check_sortedness=False,
            )
            .rename({"daily_caldt": "daily_join_caldt"})
        )
    elif join_spec_v2.phase_b_daily_join_mode == PhaseBDailyJoinMode.EXACT_ON_ALIGNED_DATE:
        right_for_exact = (
            daily
            .with_columns(
                pl.col("daily_caldt").alias("_join_caldt"),
                pl.col("daily_caldt").alias("daily_join_caldt"),
            )
            .drop("daily_caldt")
        )
        joined = (
            left_base
            .with_columns(pl.col("aligned_caldt").alias("_join_caldt"))
            .join(
                right_for_exact,
                on=["kypermno", "_join_caldt"],
                how="left",
            )
            .drop("_join_caldt")
        )
    else:
        raise ValueError(f"Unsupported phase_b_daily_join_mode: {join_spec_v2.phase_b_daily_join_mode.value}")

    has_daily_row = pl.col("daily_join_caldt").is_not_null()
    lag_days_expr = (
        pl.when(pl.col("daily_join_caldt").is_not_null() & pl.col("aligned_caldt").is_not_null())
        .then((pl.col("daily_join_caldt") - pl.col("aligned_caldt")).dt.total_days().cast(pl.Int32))
        .otherwise(pl.lit(None, dtype=pl.Int32))
    )
    usable_expr = has_daily_row
    for col in join_spec_v2.required_daily_non_null_features:
        usable_expr = usable_expr & pl.col(col).is_not_null()

    lag_gate_rejected_expr = pl.lit(False)
    max_forward_lag_days = join_spec_v2.daily_join_max_forward_lag_days
    if (
        join_spec_v2.phase_b_daily_join_mode == PhaseBDailyJoinMode.ASOF_FORWARD
        and max_forward_lag_days is not None
    ):
        if max_forward_lag_days < 0:
            raise ValueError(
                "daily_join_max_forward_lag_days must be non-negative when provided; "
                f"got {max_forward_lag_days}"
            )
        lag_gate_rejected_expr = has_daily_row & lag_days_expr.is_not_null() & (
            lag_days_expr > pl.lit(int(max_forward_lag_days))
        )
        usable_expr = usable_expr & lag_gate_rejected_expr.not_()

    daily_attempted = (
        (pl.col("phase_a_reason_code") == pl.lit(MatchReasonCode.OK.value))
        & pl.col("kypermno").is_not_null()
        & pl.col("aligned_caldt").is_not_null()
    )

    status = _update_data_status(
        pl.col("data_status"),
        conditional_flags=(
            (daily_attempted, DataStatus.SEC_CCM_PHASE_B_DAILY_JOIN_ATTEMPTED),
            (usable_expr, DataStatus.SEC_CCM_PHASE_B_DAILY_ROW_FOUND),
        ),
    ).alias("data_status")

    return joined.with_columns(
        has_daily_row.alias("_has_daily_row"),
        usable_expr.alias("_has_usable_daily_row"),
        lag_days_expr.alias("daily_join_lag_days"),
        lag_gate_rejected_expr.alias("daily_lag_gate_rejected"),
        status,
    )


def apply_phase_b_reason_codes(
    phase_a_doc_lf: pl.LazyFrame,
    phase_b_joined_lf: pl.LazyFrame,
    join_spec: SecCcmJoinSpec,
) -> pl.LazyFrame:
    """Apply Phase B reason codes, preserving non-OK Phase A outcomes."""
    join_spec_v2 = normalize_sec_ccm_join_spec(join_spec)
    _require_columns(phase_a_doc_lf, ("doc_id", "phase_a_reason_code"), "Phase A docs")
    _require_columns(
        phase_b_joined_lf,
        ("doc_id", "phase_a_reason_code", "aligned_caldt", "data_status"),
        "Phase B joined docs",
    )

    phase_a_codes = phase_a_doc_lf.select("doc_id", "phase_a_reason_code").unique(subset=["doc_id"])
    out = phase_b_joined_lf.join(phase_a_codes, on="doc_id", how="left", suffix="_phase_a")
    if "phase_a_reason_code_phase_a" in out.collect_schema():
        out = out.with_columns(
            pl.coalesce([pl.col("phase_a_reason_code"), pl.col("phase_a_reason_code_phase_a")]).alias("phase_a_reason_code")
        ).drop("phase_a_reason_code_phase_a")

    schema = out.collect_schema()
    if "_has_usable_daily_row" not in schema:
        out = out.with_columns(pl.lit(False).alias("_has_usable_daily_row"))
    if "daily_lag_gate_rejected" not in schema:
        out = out.with_columns(pl.lit(False).alias("daily_lag_gate_rejected"))
    if "key_min_caldt" not in schema:
        out = out.with_columns(pl.lit(None, dtype=pl.Date).alias("key_min_caldt"))
    if "key_max_caldt" not in schema:
        out = out.with_columns(pl.lit(None, dtype=pl.Date).alias("key_max_caldt"))

    phase_a_ok = pl.col("phase_a_reason_code") == pl.lit(MatchReasonCode.OK.value)
    base_coverage_miss = pl.col("aligned_caldt").is_null()
    if join_spec_v2.daily_join_enabled:
        coverage_miss = (
            base_coverage_miss
            | pl.col("key_min_caldt").is_null()
            | pl.col("key_max_caldt").is_null()
            | (pl.col("aligned_caldt") < pl.col("key_min_caldt"))
            | (pl.col("aligned_caldt") > pl.col("key_max_caldt"))
        )
        no_row_for_date = phase_a_ok & coverage_miss.not_() & pl.col("_has_usable_daily_row").not_()
    else:
        coverage_miss = base_coverage_miss
        no_row_for_date = pl.lit(False)

    phase_b_reason = (
        pl.when(phase_a_ok & coverage_miss)
        .then(pl.lit(MatchReasonCode.OUT_OF_CCM_COVERAGE.value))
        .when(no_row_for_date)
        .then(pl.lit(MatchReasonCode.NO_CCM_ROW_FOR_DATE.value))
        .when(phase_a_ok)
        .then(pl.lit(MatchReasonCode.OK.value))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )

    match_reason = (
        pl.when(phase_a_ok)
        .then(pl.coalesce([phase_b_reason, pl.lit(MatchReasonCode.OK.value)]))
        .otherwise(pl.col("phase_a_reason_code"))
    )

    status = _update_data_status(
        pl.col("data_status"),
        conditional_flags=(
            (
                phase_a_ok & (phase_b_reason == pl.lit(MatchReasonCode.OUT_OF_CCM_COVERAGE.value)),
                DataStatus.SEC_CCM_PHASE_B_OUT_OF_CCM_COVERAGE,
            ),
            (
                phase_a_ok & (phase_b_reason == pl.lit(MatchReasonCode.NO_CCM_ROW_FOR_DATE.value)),
                DataStatus.SEC_CCM_PHASE_B_NO_CCM_ROW_FOR_DATE,
            ),
        ),
    ).alias("data_status")

    return out.with_columns(
        phase_b_reason.alias("phase_b_reason_code"),
        match_reason.alias("match_reason_code"),
        status,
    ).drop("_has_daily_row", "_has_usable_daily_row", strict=False)


def build_match_status_doc(final_doc_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Build canonical doc-grain match status table."""
    required = ("doc_id", "cik_10", "filing_date", "phase_a_reason_code", "match_reason_code", "data_status")
    _require_columns(final_doc_lf, required, "final doc output")

    schema = final_doc_lf.collect_schema()
    out = final_doc_lf
    if "phase_b_reason_code" not in schema:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("phase_b_reason_code"))
    if "acceptance_datetime" not in schema:
        out = out.with_columns(pl.lit(None, dtype=pl.Datetime).alias("acceptance_datetime"))
    if "has_acceptance_datetime" not in schema:
        out = out.with_columns(pl.lit(False).alias("has_acceptance_datetime"))
    if "gvkey" not in schema:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("gvkey"))
    if "kypermno" not in schema:
        out = out.with_columns(pl.lit(None, dtype=pl.Int32).alias("kypermno"))
    if "aligned_caldt" not in schema:
        out = out.with_columns(pl.lit(None, dtype=pl.Date).alias("aligned_caldt"))
    if "alignment_lag_days" not in schema:
        out = out.with_columns(pl.lit(None, dtype=pl.Int32).alias("alignment_lag_days"))
    if "link_candidate_count" not in schema:
        out = out.with_columns(pl.lit(0, dtype=pl.Int32).alias("link_candidate_count"))
    if "top_rank_tie_count" not in schema:
        out = out.with_columns(pl.lit(0, dtype=pl.Int32).alias("top_rank_tie_count"))

    return out.with_columns(
        (pl.col("match_reason_code") == pl.lit(MatchReasonCode.OK.value)).alias("match_flag"),
    ).select(
        "doc_id",
        "cik_10",
        "filing_date",
        "gvkey",
        "kypermno",
        "phase_a_reason_code",
        "phase_b_reason_code",
        "match_reason_code",
        "match_flag",
        "aligned_caldt",
        "alignment_lag_days",
        "link_candidate_count",
        "top_rank_tie_count",
        "acceptance_datetime",
        "has_acceptance_datetime",
        "data_status",
    )


def build_unmatched_diagnostics_doc(
    final_doc_lf: pl.LazyFrame,
    link_universe_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build doc-grain diagnostics for unmatched filings."""
    _require_columns(final_doc_lf, ("doc_id", "cik_10", "filing_date", "match_reason_code"), "final docs")
    link_universe = _normalize_link_universe(link_universe_lf).select(
        "cik_10",
        pl.lit(True).alias("diag_cik_in_link_universe"),
    ).unique(subset=["cik_10"])

    matched_counts = final_doc_lf.group_by("cik_10").agg(
        ((pl.col("match_reason_code") == pl.lit(MatchReasonCode.OK.value)).cast(pl.Int32)).sum().alias("diag_n_matched_for_cik")
    )

    form_col = None
    schema = final_doc_lf.collect_schema()
    for candidate in ("form_type", "document_type_filename", "SRCTYPE"):
        if candidate in schema:
            form_col = candidate
            break

    diag = (
        final_doc_lf
        .filter(pl.col("match_reason_code") != pl.lit(MatchReasonCode.OK.value))
        .join(link_universe, on="cik_10", how="left")
        .join(matched_counts, on="cik_10", how="left")
        .with_columns(
            pl.col("diag_cik_in_link_universe").fill_null(False),
            pl.col("diag_n_matched_for_cik").fill_null(0).cast(pl.Int32),
            (pl.col("diag_n_matched_for_cik") > 0).alias("has_other_filings_matched_for_cik"),
        )
    )

    diag_schema = diag.collect_schema()
    if "key_min_caldt" in diag_schema and "key_max_caldt" in diag_schema:
        diag = diag.with_columns(
            (
                pl.col("key_min_caldt").is_not_null() & (pl.col("filing_date") < pl.col("key_min_caldt"))
            ).alias("diag_date_before_key_coverage"),
            (
                pl.col("key_max_caldt").is_not_null() & (pl.col("filing_date") > pl.col("key_max_caldt"))
            ).alias("diag_date_after_key_coverage"),
        )
    else:
        diag = diag.with_columns(
            pl.lit(False).alias("diag_date_before_key_coverage"),
            pl.lit(False).alias("diag_date_after_key_coverage"),
        )

    if "acceptance_datetime" not in diag_schema:
        diag = diag.with_columns(pl.lit(None, dtype=pl.Datetime).alias("acceptance_datetime"))
    if "has_acceptance_datetime" not in diag_schema:
        diag = diag.with_columns(pl.lit(False).alias("has_acceptance_datetime"))

    selected = [
        "doc_id",
        "cik_10",
        "filing_date",
    ]
    if form_col is not None:
        selected.append(form_col)
    selected.extend(
        [
            "match_reason_code",
            "phase_a_reason_code",
            "phase_b_reason_code",
            "gvkey",
            "kypermno",
            "link_candidate_count",
            "top_rank_tie_count",
            "diag_cik_in_link_universe",
            "has_other_filings_matched_for_cik",
            "diag_n_matched_for_cik",
            "diag_date_before_key_coverage",
            "diag_date_after_key_coverage",
            "aligned_caldt",
            "alignment_lag_days",
            "acceptance_datetime",
            "has_acceptance_datetime",
        ]
    )
    present = [col for col in selected if col in diag.collect_schema()]
    return diag.select(present)
