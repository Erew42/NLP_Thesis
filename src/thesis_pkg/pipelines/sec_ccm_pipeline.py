from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_pkg.core.ccm.sec_ccm_contracts import MatchReasonCode, SecCcmJoinSpecV1
from thesis_pkg.core.ccm.sec_ccm_premerge import (
    align_doc_dates_phase_b,
    apply_concept_filter_flags_doc,
    apply_phase_b_reason_codes,
    build_match_status_doc,
    build_unmatched_diagnostics_doc,
    normalize_sec_filings_phase_a,
    resolve_links_phase_a,
    join_daily_phase_b,
)


def _write_lazy_parquet(lf: pl.LazyFrame, path: Path, compression: str = "zstd") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(path, compression=compression)
    return path


def _assert_unique_doc_id(lf: pl.LazyFrame, label: str) -> None:
    dup = (
        lf.group_by("doc_id")
        .agg(pl.len().alias("n_rows"))
        .filter(pl.col("n_rows") > 1)
        .limit(5)
        .collect()
    )
    if dup.height > 0:
        samples = dup.to_dicts()
        raise ValueError(f"{label} has non-unique doc_id values; sample duplicates: {samples}")


def run_sec_ccm_premerge_pipeline(
    sec_filings_lf: pl.LazyFrame,
    link_universe_lf: pl.LazyFrame,
    trading_calendar_lf: pl.LazyFrame,
    output_dir: Path,
    *,
    daily_lf: pl.LazyFrame | None = None,
    join_spec: SecCcmJoinSpecV1 = SecCcmJoinSpecV1(),
) -> dict[str, Path]:
    """
    Run two-phase SEC-CCM pre-merge at doc grain and persist canonical artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_a_norm = normalize_sec_filings_phase_a(sec_filings_lf)
    _assert_unique_doc_id(phase_a_norm, "sec_filings")

    phase_a_links = resolve_links_phase_a(phase_a_norm, link_universe_lf)
    _assert_unique_doc_id(phase_a_links, "Phase A links")

    paths: dict[str, Path] = {}
    paths["sec_ccm_links_doc"] = _write_lazy_parquet(
        phase_a_links,
        output_dir / "sec_ccm_links_doc.parquet",
    )

    phase_b_aligned = align_doc_dates_phase_b(phase_a_links, trading_calendar_lf, join_spec)
    if join_spec.daily_join_enabled:
        if daily_lf is None:
            raise ValueError("daily_lf is required when join_spec.daily_join_enabled=True")
        phase_b_joined = join_daily_phase_b(phase_b_aligned, daily_lf, join_spec)
        diagnostics_daily_lf = daily_lf
    else:
        phase_b_joined = join_daily_phase_b(
            phase_b_aligned,
            pl.DataFrame({"KYPERMNO": [], "CALDT": []}).lazy(),
            join_spec,
        )
        diagnostics_daily_lf = trading_calendar_lf

    final_doc = apply_phase_b_reason_codes(phase_a_links, phase_b_joined, join_spec)
    final_doc = apply_concept_filter_flags_doc(final_doc)
    _assert_unique_doc_id(final_doc, "final doc output")

    paths["final_flagged_data"] = _write_lazy_parquet(final_doc, output_dir / "final_flagged_data.parquet")

    match_status = build_match_status_doc(final_doc)
    paths["sec_ccm_match_status"] = _write_lazy_parquet(
        match_status,
        output_dir / "sec_ccm_match_status.parquet",
    )

    matched = final_doc.filter(pl.col("match_reason_code") == pl.lit(MatchReasonCode.OK.value))
    unmatched = final_doc.filter(pl.col("match_reason_code") != pl.lit(MatchReasonCode.OK.value))
    paths["sec_ccm_matched_filings"] = _write_lazy_parquet(
        matched,
        output_dir / "sec_ccm_matched_filings.parquet",
    )
    paths["sec_ccm_unmatched_filings"] = _write_lazy_parquet(
        unmatched,
        output_dir / "sec_ccm_unmatched_filings.parquet",
    )

    unmatched_diagnostics = build_unmatched_diagnostics_doc(final_doc, link_universe_lf, diagnostics_daily_lf)
    paths["sec_ccm_unmatched_diagnostics"] = _write_lazy_parquet(
        unmatched_diagnostics,
        output_dir / "sec_ccm_unmatched_diagnostics.parquet",
    )

    matched_clean = matched
    paths["sec_ccm_matched_clean"] = _write_lazy_parquet(
        matched_clean,
        output_dir / "sec_ccm_matched_clean.parquet",
    )
    paths["sec_ccm_matched_clean_filtered"] = _write_lazy_parquet(
        matched_clean.filter(pl.col("passes_all_filters").fill_null(False)),
        output_dir / "sec_ccm_matched_clean_filtered.parquet",
    )

    paths["sec_ccm_analysis_doc_ids"] = _write_lazy_parquet(
        match_status.filter(pl.col("match_flag"))
        .select(pl.col("doc_id").cast(pl.Utf8))
        .unique(),
        output_dir / "sec_ccm_analysis_doc_ids.parquet",
    )
    paths["sec_ccm_diagnostic_doc_ids"] = _write_lazy_parquet(
        match_status.filter(pl.col("match_flag").not_())
        .select(pl.col("doc_id").cast(pl.Utf8))
        .unique(),
        output_dir / "sec_ccm_diagnostic_doc_ids.parquet",
    )

    paths["sec_ccm_join_spec_v1"] = join_spec.write_json(output_dir / "sec_ccm_join_spec_v1.json")
    return paths
