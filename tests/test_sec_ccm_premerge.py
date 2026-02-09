from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from thesis_pkg.core.ccm.sec_ccm_contracts import MatchReasonCode, SecCcmJoinSpecV1
from thesis_pkg.core.ccm.sec_ccm_premerge import (
    align_doc_dates_phase_b,
    apply_phase_b_reason_codes,
    build_match_status_doc,
    normalize_sec_filings_phase_a,
    resolve_links_phase_a,
    join_daily_phase_b,
)
from thesis_pkg.core.ccm.transforms import STATUS_DTYPE
from thesis_pkg.pipelines.sec_ccm_pipeline import run_sec_ccm_premerge_pipeline


def test_phase_a_reason_codes_and_doc_grain_invariants():
    sec = pl.DataFrame(
        {
            "doc_id": ["d1", "d2", "d3", "d4"],
            "cik_10": ["0000000001", "bad_cik", "0000000003", "0000000004"],
            "filing_date": [dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "acceptance_datetime": [dt.datetime(2024, 1, 2, 16, 0), None, None, None],
        }
    )

    link_universe = pl.DataFrame(
        {
            "cik_10": ["0000000001", "0000000004", "0000000004"],
            "gvkey": ["1000", "4000", "4001"],
            "kypermno": [1, 4, 5],
            "link_rank": [0, 0, 0],
            "link_quality": [0.9, 0.8, 0.8],
        }
    )

    phase_a = resolve_links_phase_a(sec.lazy(), link_universe.lazy()).collect().sort("doc_id")
    reasons = dict(zip(phase_a["doc_id"], phase_a["phase_a_reason_code"]))

    assert phase_a.height == sec.height
    assert phase_a["doc_id"].n_unique() == sec.height
    assert reasons["d1"] == MatchReasonCode.OK.value
    assert reasons["d2"] == MatchReasonCode.BAD_INPUT.value
    assert reasons["d3"] == MatchReasonCode.CIK_NOT_IN_LINK_UNIVERSE.value
    assert reasons["d4"] == MatchReasonCode.AMBIGUOUS_LINK.value
    assert (
        phase_a.filter(pl.col("doc_id") == "d4").select("kypermno").item()
        is None
    )
    assert phase_a.schema["kypermno"] == pl.Int32
    assert phase_a.schema["data_status"] == STATUS_DTYPE
    assert phase_a.select(pl.col("data_status").is_null().any()).item() is False


def test_phase_b_strict_next_day_alignment_and_reason_scoping():
    sec = pl.DataFrame(
        {
            "doc_id": ["ok", "out_cov", "ambig", "no_row"],
            "cik_10": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "filing_date": [
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 4),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 2),
            ],
        }
    )
    link_universe = pl.DataFrame(
        {
            "cik_10": ["0000000001", "0000000002", "0000000003", "0000000003", "0000000004"],
            "gvkey": ["1000", "2000", "3000", "3001", "4000"],
            "kypermno": [1, 2, 3, 4, 5],
            "link_rank": [0, 0, 0, 0, 0],
            "link_quality": [1.0, 1.0, 0.9, 0.9, 1.0],
        }
    )
    trading_calendar = pl.DataFrame({"CALDT": [dt.date(2024, 1, 3), dt.date(2024, 1, 4)]})
    daily = pl.DataFrame(
        {
            "KYPERMNO": [1, 5],
            "CALDT": [dt.date(2024, 1, 3), dt.date(2024, 1, 3)],
            "RET": [0.01, None],
            "RETX": [0.01, 0.01],
            "PRC": [10.0, 12.0],
            "BIDLO": [9.5, 11.5],
            "ASKHI": [10.5, 12.5],
        }
    )

    phase_a = resolve_links_phase_a(sec.lazy(), link_universe.lazy())
    aligned = align_doc_dates_phase_b(phase_a, trading_calendar.lazy(), SecCcmJoinSpecV1())
    aligned_df = aligned.collect()

    assert (
        aligned_df.filter(pl.col("aligned_caldt").is_not_null())
        .select((pl.col("aligned_caldt") > pl.col("filing_date")).all())
        .item()
        is True
    )
    assert (
        aligned_df.filter(pl.col("aligned_caldt").is_not_null())
        .select((pl.col("alignment_lag_days") >= 1).all())
        .item()
        is True
    )

    join_spec = SecCcmJoinSpecV1(required_daily_non_null_features=("RET",))
    joined = join_daily_phase_b(aligned, daily.lazy(), join_spec)
    final = apply_phase_b_reason_codes(phase_a, joined, join_spec).collect().sort("doc_id")

    by_doc = {row["doc_id"]: row for row in final.to_dicts()}
    assert by_doc["ok"]["match_reason_code"] == MatchReasonCode.OK.value
    assert by_doc["out_cov"]["match_reason_code"] == MatchReasonCode.OUT_OF_CCM_COVERAGE.value
    assert by_doc["no_row"]["match_reason_code"] == MatchReasonCode.NO_CCM_ROW_FOR_DATE.value
    assert by_doc["ambig"]["phase_a_reason_code"] == MatchReasonCode.AMBIGUOUS_LINK.value
    assert by_doc["ambig"]["match_reason_code"] == MatchReasonCode.AMBIGUOUS_LINK.value
    assert by_doc["ambig"]["phase_b_reason_code"] is None


def test_acceptance_datetime_optional_and_not_default_coerced():
    without_acceptance = pl.DataFrame(
        {
            "doc_id": ["d1"],
            "cik_10": ["0000000001"],
            "filing_date": [dt.date(2024, 1, 2)],
        }
    )
    out1 = normalize_sec_filings_phase_a(without_acceptance.lazy()).collect()
    assert out1.select("acceptance_datetime").item() is None
    assert out1.select("has_acceptance_datetime").item() is False

    with_acceptance = pl.DataFrame(
        {
            "doc_id": ["d1", "d2"],
            "cik_10": ["0000000001", "0000000002"],
            "filing_date": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "acceptance_datetime": [dt.datetime(2024, 1, 2, 17, 30), None],
        }
    )
    out2 = normalize_sec_filings_phase_a(with_acceptance.lazy()).collect().sort("doc_id")
    assert out2.select("acceptance_datetime").to_series().to_list() == [dt.datetime(2024, 1, 2, 17, 30), None]
    assert out2.select("has_acceptance_datetime").to_series().to_list() == [True, False]


def test_end_to_end_pipeline_outputs_doc_grain_artifacts(tmp_path: Path):
    sec = pl.DataFrame(
        {
            "doc_id": ["d1", "d2", "d3"],
            "cik_10": ["0000000001", "0000000002", "bad"],
            "filing_date": [dt.date(2024, 1, 2), dt.date(2024, 1, 4), dt.date(2024, 1, 2)],
            "document_type_filename": ["10-K", "10-K", "10-K"],
        }
    )
    link_universe = pl.DataFrame(
        {
            "cik_10": ["0000000001", "0000000002"],
            "gvkey": ["1000", "2000"],
            "kypermno": [1, 2],
            "link_rank": [0, 0],
            "link_quality": [1.0, 1.0],
        }
    )
    trading_calendar = pl.DataFrame({"CALDT": [dt.date(2024, 1, 3), dt.date(2024, 1, 4)]})
    daily = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [dt.date(2024, 1, 3)],
            "RET": [0.01],
            "RETX": [0.01],
            "PRC": [10.0],
            "BIDLO": [9.5],
            "ASKHI": [10.5],
        }
    )

    paths = run_sec_ccm_premerge_pipeline(
        sec.lazy(),
        link_universe.lazy(),
        trading_calendar.lazy(),
        tmp_path,
        daily_lf=daily.lazy(),
        join_spec=SecCcmJoinSpecV1(required_daily_non_null_features=("RET",)),
    )

    required_keys = {
        "sec_ccm_links_doc",
        "final_flagged_data",
        "sec_ccm_match_status",
        "sec_ccm_matched_filings",
        "sec_ccm_unmatched_filings",
        "sec_ccm_unmatched_diagnostics",
        "sec_ccm_matched_clean",
        "sec_ccm_matched_clean_filtered",
        "sec_ccm_analysis_doc_ids",
        "sec_ccm_diagnostic_doc_ids",
        "sec_ccm_join_spec_v1",
    }
    assert required_keys.issubset(paths.keys())
    for key, path in paths.items():
        assert Path(path).exists(), key

    final_df = pl.read_parquet(paths["final_flagged_data"])
    status_df = pl.read_parquet(paths["sec_ccm_match_status"])

    assert final_df.height == sec.height
    assert final_df["doc_id"].n_unique() == sec.height
    assert status_df.height == sec.height
    assert status_df["doc_id"].n_unique() == sec.height
    assert final_df.schema["kypermno"] == pl.Int32
    assert final_df.schema["data_status"] == STATUS_DTYPE
    assert final_df.select(pl.col("data_status").is_null().any()).item() is False
    expected_status = build_match_status_doc(final_df.lazy()).collect().sort("doc_id")
    actual_status = status_df.sort("doc_id")
    assert expected_status.columns == actual_status.columns
    assert expected_status.to_dicts() == actual_status.to_dicts()
