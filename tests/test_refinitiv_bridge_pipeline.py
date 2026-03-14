from __future__ import annotations

import json
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter

from thesis_pkg.io.excel import write_refinitiv_ric_lookup_extended_workbook
from thesis_pkg.pipelines.refinitiv_bridge_pipeline import (
    BRIDGE_OUTPUT_COLUMNS,
    BRIDGE_VENDOR_COLUMNS,
    NULL_RIC_DIAGNOSTIC_COLUMNS,
    OWNERSHIP_SMOKE_SAMPLE_COLUMNS,
    RIC_LOOKUP_COLUMNS,
    RIC_LOOKUP_EXTENDED_COLUMNS,
    RIC_LOOKUP_FILTER_PROFILES,
    _build_filtered_ric_lookup_profile_artifact,
    _build_lookup_profile_bridge_ids,
    _build_null_ric_review_frame,
    _build_ric_lookup_handoff_frames,
    build_refinitiv_ownership_smoke_sample,
    build_refinitiv_lookup_extended_diagnostic_artifact,
    build_refinitiv_step1_bridge_universe,
    build_refinitiv_null_ric_rescue_candidates,
    run_refinitiv_null_ric_diagnostics_pipeline,
    run_refinitiv_step1_bridge_pipeline,
)


def _daily_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "KYPERMNO": [1, 1, 1, 1, 2, 4],
            "CALDT": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-02",
                "2024-01-02",
            ],
            "KYGVKEY_final": ["1000", "1000", "1000", "1000", "2000", None],
            "LIID": ["A", "A", "A", "A", None, None],
            "CIK_final": [
                "0000001000",
                "0000001000",
                "0000001000",
                "0000001000",
                "0000002000",
                "0000004000",
            ],
            "CUSIP": ["11111111", "11111111", "11111111", "11111111", None, "44444444"],
            "ISIN": [
                "US1111111111",
                "US1111111111",
                "US1111111111",
                "US1111111111",
                "US0000002000",
                None,
            ],
            "TICKER": ["ALFA", "ALFA", "ALFB", "ALFB", "BETA", " DELTA "],
            "LINKTYPE": ["LC", "LC", "LC", "LC", "LU", "LC"],
            "LINKPRIM": ["P", "P", "P", "P", "C", "P"],
            "link_quality_flag": ["canonical_primary_LC"] * 4 + ["canonical_other", "canonical_primary_LC"],
            "HEXCNTRY": ["US", "US", "US", "US", "GB", "US"],
            "n_filings": [0, 2, 0, 1, 0, 0],
            "SHRCD": [10, 10, 12, 12, 11, 10],
            "EXCHCD": [1, 1, 1, 1, 2, 1],
        }
    ).with_columns(pl.col("CALDT").str.strptime(pl.Date, "%Y-%m-%d", strict=True))


def _company_description() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "KYGVKEY": ["1000", "2000"],
            "CONM": ["Alpha Corp", "Beta plc"],
        }
    )


def _bridge_with_manual_review() -> pl.DataFrame:
    bridge = build_refinitiv_step1_bridge_universe(
        _daily_panel()
        .vstack(
            pl.DataFrame(
                {
                    "KYPERMNO": [3, 5],
                    "CALDT": [date(2024, 1, 6), date(2024, 1, 8)],
                    "KYGVKEY_final": ["3000", "5000"],
                    "LIID": [None, None],
                    "CIK_final": ["0000003000", "0000005000"],
                    "CUSIP": [None, None],
                    "ISIN": [None, None],
                    "TICKER": ["GAMMA", "  "],
                    "LINKTYPE": ["LU", "LU"],
                    "LINKPRIM": ["C", "C"],
                    "link_quality_flag": ["ticker_only", "manual_review"],
                    "HEXCNTRY": ["US", "US"],
                    "n_filings": [0, 0],
                    "SHRCD": [10, 10],
                    "EXCHCD": [1, 1],
                }
            )
        )
        .lazy(),
        company_description_lf=_company_description().lazy(),
    ).collect()
    return bridge


def _write_filled_lookup_workbook(path: Path) -> Path:
    workbook = Workbook()
    ric_ws = workbook.active
    ric_ws.title = "ric_lookup"
    ric_ws.append(list(RIC_LOOKUP_COLUMNS))
    records = [
        {
            "bridge_row_id": "1:01:111111111:US1111111111:ALFA-S",
            "KYPERMNO": "1",
            "CUSIP": "111111111",
            "ISIN": "US1111111111",
            "TICKER": "ALFA",
            "first_seen_caldt": "2024-01-01",
            "last_seen_caldt": "2024-01-10",
            "preferred_lookup_id": "US1111111111",
            "preferred_lookup_type": "ISIN",
            "vendor_primary_ric": "AAA.O",
            "vendor_returned_name": "Alpha Inc",
            "vendor_returned_cusip": "111111111",
            "vendor_returned_isin": "US1111111111",
            "vendor_match_status": None,
            "vendor_notes": None,
        },
        {
            "bridge_row_id": "1:01:111111111:US1111111111:ALFA-AFTER",
            "KYPERMNO": "1",
            "CUSIP": "111111111",
            "ISIN": "US1111111111",
            "TICKER": "ALFA",
            "first_seen_caldt": "2023-12-20",
            "last_seen_caldt": "2023-12-25",
            "preferred_lookup_id": "US1111111111",
            "preferred_lookup_type": "ISIN",
            "vendor_primary_ric": "NULL",
            "vendor_returned_name": "The invalid identifier ignored.",
            "vendor_returned_cusip": "The invalid identifier ignored.",
            "vendor_returned_isin": "The invalid identifier ignored.",
            "vendor_match_status": None,
            "vendor_notes": None,
        },
        {
            "bridge_row_id": "1:01:111111111:US1111111111:ALFA-BEFORE",
            "KYPERMNO": "1",
            "CUSIP": "111111111",
            "ISIN": "US1111111111",
            "TICKER": "ALFA",
            "first_seen_caldt": "2024-01-11",
            "last_seen_caldt": "2024-01-15",
            "preferred_lookup_id": "US1111111111",
            "preferred_lookup_type": "ISIN",
            "vendor_primary_ric": "",
            "vendor_returned_name": None,
            "vendor_returned_cusip": None,
            "vendor_returned_isin": None,
            "vendor_match_status": None,
            "vendor_notes": None,
        },
        {
            "bridge_row_id": "2:01:222222222:US2222222222:BETA-S1",
            "KYPERMNO": "2",
            "CUSIP": "222222222",
            "ISIN": "US2222222222",
            "TICKER": "BETA",
            "first_seen_caldt": "2024-02-01",
            "last_seen_caldt": "2024-02-10",
            "preferred_lookup_id": "US2222222222",
            "preferred_lookup_type": "ISIN",
            "vendor_primary_ric": "BBB.O",
            "vendor_returned_name": "Beta Inc",
            "vendor_returned_cusip": "222222222",
            "vendor_returned_isin": "US2222222222",
            "vendor_match_status": None,
            "vendor_notes": None,
        },
        {
            "bridge_row_id": "2:01:223333333:US2233333333:BETN-S2",
            "KYPERMNO": "2",
            "CUSIP": "223333333",
            "ISIN": "US2233333333",
            "TICKER": "BETN",
            "first_seen_caldt": "2024-03-01",
            "last_seen_caldt": "2024-03-10",
            "preferred_lookup_id": "US2233333333",
            "preferred_lookup_type": "ISIN",
            "vendor_primary_ric": "CCC.O",
            "vendor_returned_name": "Beta New",
            "vendor_returned_cusip": "223333333",
            "vendor_returned_isin": "US2233333333",
            "vendor_match_status": None,
            "vendor_notes": None,
        },
        {
            "bridge_row_id": "2:01:222222222:US2222222222:BETA-F1",
            "KYPERMNO": "2",
            "CUSIP": "222222222",
            "ISIN": "US2222222222",
            "TICKER": "BETA",
            "first_seen_caldt": "2024-02-15",
            "last_seen_caldt": "2024-02-20",
            "preferred_lookup_id": "222222222",
            "preferred_lookup_type": "CUSIP",
            "vendor_primary_ric": "",
            "vendor_returned_name": None,
            "vendor_returned_cusip": None,
            "vendor_returned_isin": None,
            "vendor_match_status": None,
            "vendor_notes": None,
        },
        {
            "bridge_row_id": "2:01:222222222:US2222222222:BETA-F2",
            "KYPERMNO": "2",
            "CUSIP": "222222222",
            "ISIN": "US2222222222",
            "TICKER": "BETA",
            "first_seen_caldt": "2024-02-21",
            "last_seen_caldt": "2024-02-25",
            "preferred_lookup_id": "US2222222222",
            "preferred_lookup_type": "ISIN",
            "vendor_primary_ric": None,
            "vendor_returned_name": None,
            "vendor_returned_cusip": None,
            "vendor_returned_isin": None,
            "vendor_match_status": None,
            "vendor_notes": None,
        },
        {
            "bridge_row_id": "3:01:333333333:US3333333333:GAM-F",
            "KYPERMNO": "3",
            "CUSIP": "333333333",
            "ISIN": "US3333333333",
            "TICKER": "GAM",
            "first_seen_caldt": "2024-04-01",
            "last_seen_caldt": "2024-04-05",
            "preferred_lookup_id": "US3333333333",
            "preferred_lookup_type": "ISIN",
            "vendor_primary_ric": None,
            "vendor_returned_name": None,
            "vendor_returned_cusip": None,
            "vendor_returned_isin": None,
            "vendor_match_status": None,
            "vendor_notes": None,
        },
        {
            "bridge_row_id": "4:01:444444444:US4444444444:DLTA-S",
            "KYPERMNO": "4",
            "CUSIP": "444444444",
            "ISIN": "US4444444444",
            "TICKER": "DLTA",
            "first_seen_caldt": "2024-05-01",
            "last_seen_caldt": "2024-05-10",
            "preferred_lookup_id": "US4444444444",
            "preferred_lookup_type": "ISIN",
            "vendor_primary_ric": "DDD.O",
            "vendor_returned_name": "Delta Inc",
            "vendor_returned_cusip": "444444444",
            "vendor_returned_isin": "US4444444444",
            "vendor_match_status": None,
            "vendor_notes": None,
        },
        {
            "bridge_row_id": "4:01:444444444:US4444444444:DLTA-AFTER",
            "KYPERMNO": "4",
            "CUSIP": "444444444",
            "ISIN": "US4444444444",
            "TICKER": "DLTA",
            "first_seen_caldt": "2024-04-20",
            "last_seen_caldt": "2024-04-25",
            "preferred_lookup_id": "US4444444444",
            "preferred_lookup_type": "ISIN",
            "vendor_primary_ric": None,
            "vendor_returned_name": None,
            "vendor_returned_cusip": None,
            "vendor_returned_isin": None,
            "vendor_match_status": None,
            "vendor_notes": None,
        },
        {
            "bridge_row_id": "4:01:444444444:US4444444444:DLTA-BEFORE",
            "KYPERMNO": "4",
            "CUSIP": "444444444",
            "ISIN": "US4444444444",
            "TICKER": "DLTA",
            "first_seen_caldt": "2024-05-11",
            "last_seen_caldt": "2024-05-12",
            "preferred_lookup_id": "444444444",
            "preferred_lookup_type": "CUSIP",
            "vendor_primary_ric": "NULL",
            "vendor_returned_name": "The invalid identifier ignored.",
            "vendor_returned_cusip": "The invalid identifier ignored.",
            "vendor_returned_isin": "The invalid identifier ignored.",
            "vendor_match_status": None,
            "vendor_notes": None,
        },
        {
            "bridge_row_id": "5:01:-:-:EPSI-F",
            "KYPERMNO": "5",
            "CUSIP": None,
            "ISIN": None,
            "TICKER": "EPSI",
            "first_seen_caldt": "2024-06-01",
            "last_seen_caldt": "2024-06-05",
            "preferred_lookup_id": "EPSI",
            "preferred_lookup_type": "TICKER",
            "vendor_primary_ric": None,
            "vendor_returned_name": None,
            "vendor_returned_cusip": None,
            "vendor_returned_isin": None,
            "vendor_match_status": None,
            "vendor_notes": None,
        },
    ]
    for record in records:
        ric_ws.append([record.get(name) for name in RIC_LOOKUP_COLUMNS])
    workbook.create_sheet("README")
    workbook.save(path)
    workbook.close()
    return path


def test_build_refinitiv_step1_bridge_universe_uses_distinct_identifier_grain() -> None:
    bridge = build_refinitiv_step1_bridge_universe(
        _daily_panel().lazy(),
        company_description_lf=_company_description().lazy(),
    ).collect()

    assert bridge.columns == list(BRIDGE_OUTPUT_COLUMNS)
    assert bridge.height == 4
    assert bridge["bridge_row_id"].n_unique() == bridge.height

    by_key = {row["bridge_row_id"]: row for row in bridge.to_dicts()}
    first = by_key["1:A:11111111:US1111111111:ALFA"]
    second = by_key["1:A:11111111:US1111111111:ALFB"]
    third = by_key["2:-:-:US0000002000:BETA"]
    fourth = by_key["4:-:44444444:-:DELTA"]

    assert first["first_seen_caldt"].isoformat() == "2024-01-02"
    assert first["last_seen_caldt"].isoformat() == "2024-01-03"
    assert first["n_daily_rows"] == 2
    assert first["n_filing_days"] == 1
    assert first["n_filings_total"] == 2
    assert first["company_name"] == "Alpha Corp"
    assert first["TICKER"] == "ALFA"

    assert second["first_seen_caldt"].isoformat() == "2024-01-04"
    assert second["last_seen_caldt"].isoformat() == "2024-01-05"
    assert second["n_daily_rows"] == 2
    assert second["n_filing_days"] == 1
    assert second["n_filings_total"] == 1
    assert second["TICKER"] == "ALFB"

    assert third["company_name"] == "Beta plc"
    assert third["LIID"] is None
    assert third["CUSIP"] is None
    assert third["ISIN"] == "US0000002000"
    assert third["TICKER"] == "BETA"
    assert fourth["KYGVKEY_final"] is None
    assert fourth["CUSIP"] == "44444444"
    assert fourth["TICKER"] == "DELTA"

    for name in BRIDGE_VENDOR_COLUMNS:
        assert bridge[name].null_count() == bridge.height


def test_build_ric_lookup_handoff_frames_prefers_isin_and_separates_manual_review() -> None:
    lookup_df, manual_review_df = _build_ric_lookup_handoff_frames(_bridge_with_manual_review())

    assert lookup_df.columns == list(RIC_LOOKUP_COLUMNS)
    assert manual_review_df.columns == list(RIC_LOOKUP_COLUMNS)
    assert lookup_df.height == 5
    assert manual_review_df.height == 1

    lookup_rows = {row["bridge_row_id"]: row for row in lookup_df.to_dicts()}
    assert lookup_rows["1:A:11111111:US1111111111:ALFA"]["preferred_lookup_id"] == "US1111111111"
    assert lookup_rows["1:A:11111111:US1111111111:ALFA"]["preferred_lookup_type"] == "ISIN"
    assert lookup_rows["1:A:11111111:US1111111111:ALFA"]["vendor_returned_name"] is None
    assert lookup_rows["1:A:11111111:US1111111111:ALFA"]["vendor_returned_cusip"] is None
    assert lookup_rows["1:A:11111111:US1111111111:ALFA"]["vendor_returned_isin"] is None
    assert lookup_rows["2:-:-:US0000002000:BETA"]["preferred_lookup_id"] == "US0000002000"
    assert lookup_rows["2:-:-:US0000002000:BETA"]["preferred_lookup_type"] == "ISIN"
    assert lookup_rows["3:-:-:-:GAMMA"]["preferred_lookup_id"] == "GAMMA"
    assert lookup_rows["3:-:-:-:GAMMA"]["preferred_lookup_type"] == "TICKER"

    manual_row = manual_review_df.to_dicts()[0]
    assert manual_row["bridge_row_id"] == "5:-:-:-:-"
    assert manual_row["preferred_lookup_id"] is None
    assert manual_row["preferred_lookup_type"] is None


def test_build_refinitiv_lookup_extended_diagnostic_artifact_tracks_all_identifier_inputs() -> None:
    lookup_df, manual_review_df = _build_ric_lookup_handoff_frames(_bridge_with_manual_review())

    extended_df, summary_df, summary_payload = build_refinitiv_lookup_extended_diagnostic_artifact(
        lookup_df,
        manual_review_df,
    )

    assert extended_df.columns == list(RIC_LOOKUP_EXTENDED_COLUMNS)
    assert extended_df.height == 6

    rows = {row["bridge_row_id"]: row for row in extended_df.to_dicts()}
    alfa_row = rows["1:A:11111111:US1111111111:ALFA"]
    assert "preferred_lookup_id" not in extended_df.columns
    assert "preferred_lookup_type" not in extended_df.columns
    assert alfa_row["ISIN_lookup_input"] == "US1111111111"
    assert alfa_row["CUSIP_lookup_input"] == "11111111"
    assert alfa_row["TICKER_lookup_input"] == "ALFA"
    assert alfa_row["ISIN_attempted"] is True
    assert alfa_row["CUSIP_attempted"] is True
    assert alfa_row["TICKER_attempted"] is True
    assert alfa_row["ISIN_success"] is False
    assert alfa_row["all_successful_attempts_consistent"] is None
    assert alfa_row["vendor_primary_ric"] is None
    assert alfa_row["vendor_returned_name"] is None

    gamma_row = rows["3:-:-:-:GAMMA"]
    assert gamma_row["ISIN_lookup_input"] is None
    assert gamma_row["CUSIP_lookup_input"] is None
    assert gamma_row["TICKER_lookup_input"] == "GAMMA"
    assert gamma_row["TICKER_attempted"] is True

    manual_row = rows["5:-:-:-:-"]
    assert manual_row["ISIN_lookup_input"] is None
    assert manual_row["CUSIP_lookup_input"] is None
    assert manual_row["TICKER_lookup_input"] is None
    assert manual_row["ISIN_attempted"] is False
    assert manual_row["CUSIP_attempted"] is False
    assert manual_row["TICKER_attempted"] is False

    assert summary_df.filter(pl.col("summary_category") == "attempt_count_by_identifier_type").sort(
        "summary_key"
    ).to_dicts() == [
        {"summary_category": "attempt_count_by_identifier_type", "summary_key": "CUSIP", "value": 3},
        {"summary_category": "attempt_count_by_identifier_type", "summary_key": "ISIN", "value": 3},
        {"summary_category": "attempt_count_by_identifier_type", "summary_key": "TICKER", "value": 5},
    ]
    assert summary_payload["attempt_counts_by_identifier_type"] == {
        "ISIN": 3,
        "CUSIP": 3,
        "TICKER": 5,
    }
    assert summary_payload["success_counts_by_identifier_type"] == {
        "ISIN": 0,
        "CUSIP": 0,
        "TICKER": 0,
    }
    assert summary_payload["rows_where_only_one_identifier_type_succeeds"] == 0


def test_write_refinitiv_ric_lookup_extended_workbook_prefills_formulas(tmp_path: Path) -> None:
    lookup_df, manual_review_df = _build_ric_lookup_handoff_frames(_bridge_with_manual_review())
    extended_df, summary_df, summary_payload = build_refinitiv_lookup_extended_diagnostic_artifact(
        lookup_df,
        manual_review_df,
    )
    out_path = tmp_path / "refinitiv_ric_lookup_handoff_common_stock_extended.xlsx"

    write_refinitiv_ric_lookup_extended_workbook(
        extended_df,
        out_path,
        readme_payload={"extended_lookup_diagnostic_summary": summary_payload},
        text_columns=tuple(
            name
            for name in RIC_LOOKUP_EXTENDED_COLUMNS
            if name not in {"all_successful_attempts_consistent"}
            and not name.endswith("_attempted")
            and not name.endswith("_success")
            and "_same_" not in name
        ),
        summary_df=summary_df,
        summary_text_columns=("summary_category", "summary_key"),
    )

    workbook = load_workbook(out_path, data_only=False)
    try:
        assert "lookup_diagnostics" in workbook.sheetnames
        assert "summary" in workbook.sheetnames
        assert "README" in workbook.sheetnames

        ws = workbook["lookup_diagnostics"]
        headers = [cell.value for cell in ws[1]]
        assert "preferred_lookup_id" not in headers
        assert "preferred_lookup_type" not in headers

        col_idx = {name: idx + 1 for idx, name in enumerate(headers)}
        col_ref = {name: f"{get_column_letter(idx)}2" for name, idx in col_idx.items()}
        assert ws.cell(2, col_idx["ISIN_lookup_input"]).value == "US1111111111"
        assert ws.cell(2, col_idx["ISIN_returned_ric"]).value == (
            f'=IF({col_ref["ISIN_lookup_input"]}="","",_xll.TR({col_ref["ISIN_lookup_input"]},"TR.RIC"))'
        )
        assert ws.cell(2, col_idx["ISIN_returned_name"]).value == (
            f'=IF({col_ref["ISIN_lookup_input"]}="","",_xll.TR({col_ref["ISIN_lookup_input"]},"TR.CommonName"))'
        )
        assert ws.cell(2, col_idx["CUSIP_returned_ric"]).value == (
            f'=IF({col_ref["CUSIP_lookup_input"]}="","",_xll.TR({col_ref["CUSIP_lookup_input"]},"TR.RIC"))'
        )
        assert ws.cell(2, col_idx["TICKER_returned_ric"]).value == (
            f'=IF({col_ref["TICKER_lookup_input"]}="","",_xll.TR({col_ref["TICKER_lookup_input"]},"TR.RIC"))'
        )
        assert ws.cell(2, col_idx["ISIN_attempted"]).value == f'=LEN({col_ref["ISIN_lookup_input"]})>0'
        assert ws.cell(2, col_idx["ISIN_success"]).value == (
            f'=IFERROR({col_ref["ISIN_returned_ric"]}<>"",FALSE)'
        )
        assert ws.cell(2, col_idx["ISIN_vs_CUSIP_same_ric"]).value == (
            f'=IF(AND(IFERROR({col_ref["ISIN_returned_ric"]}<>"",FALSE),'
            f'IFERROR({col_ref["CUSIP_returned_ric"]}<>"",FALSE)),'
            f'IFERROR({col_ref["ISIN_returned_ric"]}={col_ref["CUSIP_returned_ric"]},FALSE),"")'
        )
        pairwise_refs = [
            col_ref["ISIN_vs_CUSIP_same_ric"],
            col_ref["ISIN_vs_CUSIP_same_isin"],
            col_ref["ISIN_vs_CUSIP_same_cusip"],
            col_ref["ISIN_vs_TICKER_same_ric"],
            col_ref["ISIN_vs_TICKER_same_isin"],
            col_ref["ISIN_vs_TICKER_same_cusip"],
            col_ref["CUSIP_vs_TICKER_same_ric"],
            col_ref["CUSIP_vs_TICKER_same_isin"],
            col_ref["CUSIP_vs_TICKER_same_cusip"],
        ]
        pairwise_terms = ",".join(f'IF({cell_ref}="",TRUE,{cell_ref})' for cell_ref in pairwise_refs)
        expected_consistency_formula = (
            f'=IF((N({col_ref["ISIN_success"]})+N({col_ref["CUSIP_success"]})+N({col_ref["TICKER_success"]}))<2,"",'
            f"AND({pairwise_terms}))"
        )
        assert ws.cell(2, col_idx["all_successful_attempts_consistent"]).value == (
            expected_consistency_formula
        )

        readme_ws = workbook["README"]
        readme_values = [
            cell
            for row in readme_ws.iter_rows(values_only=True)
            for cell in row
            if isinstance(cell, str)
        ]
        readme_text = "\n".join(readme_values)
        assert "main manual comparison workbook" in readme_text
        assert "save the evaluated workbook as a new file before any re-import step" in readme_text

        summary_ws = workbook["summary"]
        summary_headers = [cell.value for cell in summary_ws[1]]
        summary_col_idx = {name: idx + 1 for idx, name in enumerate(summary_headers)}
        summary_value = summary_ws.cell(2, summary_col_idx["value"]).value
        assert isinstance(summary_value, str)
        assert summary_value.startswith("=COUNTIF('lookup_diagnostics'!")
    finally:
        workbook.close()


def test_filtered_ric_lookup_profiles_apply_common_stock_and_gvkey_predicates() -> None:
    bridge = build_refinitiv_step1_bridge_universe(
        _daily_panel().lazy(),
        company_description_lf=_company_description().lazy(),
    ).collect()
    profiles = {profile.name: profile for profile in RIC_LOOKUP_FILTER_PROFILES}

    common_lookup_df, _, common_summary = _build_filtered_ric_lookup_profile_artifact(
        bridge,
        _build_lookup_profile_bridge_ids(_daily_panel().lazy(), profiles["common_stock"]),
        profiles["common_stock"],
    )
    strict_lookup_df, _, strict_summary = _build_filtered_ric_lookup_profile_artifact(
        bridge,
        _build_lookup_profile_bridge_ids(_daily_panel().lazy(), profiles["common_stock_with_gvkey"]),
        profiles["common_stock_with_gvkey"],
    )

    assert set(common_lookup_df["bridge_row_id"].to_list()) == {
        "1:A:11111111:US1111111111:ALFA",
        "2:-:-:US0000002000:BETA",
        "4:-:44444444:-:DELTA",
    }
    assert set(strict_lookup_df["bridge_row_id"].to_list()) == {
        "1:A:11111111:US1111111111:ALFA",
        "2:-:-:US0000002000:BETA",
    }
    assert common_summary["predicates_applied"] == [
        "KYPERMNO is not null",
        "common stock (SHRCD in {10, 11})",
    ]
    assert common_summary["rows_before_filter"] == 4
    assert common_summary["rows_after_filter"] == 3
    assert strict_summary["rows_after_filter"] == 2


def test_build_refinitiv_null_ric_rescue_candidates_classifies_failed_rows() -> None:
    lookup_df = pl.DataFrame(
        {
            "bridge_row_id": [
                "1:01:111111111:US1111111111:ALFA",
                "1:01:111111111:US1111111111:ALFA-F",
                "2:01:222222222:US2222222222:BETA-A",
                "2:01:223333333:US2233333333:BETN-B",
                "2:01:222222222:US2222222222:BETA-F",
                "3:01:333333333:US3333333333:GAM-F",
            ],
            "KYPERMNO": ["1", "1", "2", "2", "2", "3"],
            "CUSIP": ["111111111", "111111111", "222222222", "223333333", "222222222", "333333333"],
            "ISIN": [
                "US1111111111",
                "US1111111111",
                "US2222222222",
                "US2233333333",
                "US2222222222",
                "US3333333333",
            ],
            "TICKER": ["ALFA", "ALFA", "BETA", "BETN", "BETA", "GAM"],
            "first_seen_caldt": [
                date(2024, 1, 1),
                date(2024, 1, 11),
                date(2024, 2, 1),
                date(2024, 3, 1),
                date(2024, 2, 15),
                date(2024, 4, 1),
            ],
            "last_seen_caldt": [
                date(2024, 1, 10),
                date(2024, 1, 15),
                date(2024, 2, 10),
                date(2024, 3, 10),
                date(2024, 2, 20),
                date(2024, 4, 5),
            ],
            "preferred_lookup_id": [
                "US1111111111",
                "US1111111111",
                "US2222222222",
                "US2233333333",
                "222222222",
                "US3333333333",
            ],
            "preferred_lookup_type": ["ISIN", "ISIN", "ISIN", "ISIN", "CUSIP", "ISIN"],
            "vendor_primary_ric": ["AAA.O", "NULL", "BBB.O", "CCC.O", "", None],
            "vendor_returned_name": [
                "Alpha Inc",
                "The invalid identifier ignored.",
                "Beta Inc",
                "Beta New",
                None,
                None,
            ],
            "vendor_returned_cusip": [
                "111111111",
                "The invalid identifier ignored.",
                "222222222",
                "223333333",
                None,
                None,
            ],
            "vendor_returned_isin": [
                "US1111111111",
                "The invalid identifier ignored.",
                "US2222222222",
                "US2233333333",
                None,
                None,
            ],
            "vendor_match_status": [None, None, None, None, None, None],
            "vendor_notes": [None, None, None, None, None, None],
        }
    )

    diagnostics_df, summary = build_refinitiv_null_ric_rescue_candidates(lookup_df)
    assert diagnostics_df.columns == list(NULL_RIC_DIAGNOSTIC_COLUMNS)
    assert diagnostics_df.height == 3
    assert "rescue_classification" not in diagnostics_df.columns

    rows = {row["bridge_row_id"]: row for row in diagnostics_df.to_dicts()}
    assert rows["1:01:111111111:US1111111111:ALFA-F"]["successful_row_exists_before_span"] is True
    assert rows["1:01:111111111:US1111111111:ALFA-F"]["nearest_successful_gap_days_before"] == 1
    assert rows["1:01:111111111:US1111111111:ALFA-F"]["candidate_successful_ric_available"] is True
    assert rows["2:01:222222222:US2222222222:BETA-F"]["successful_row_exists_before_span"] is True
    assert rows["2:01:222222222:US2222222222:BETA-F"]["successful_row_exists_after_span"] is True
    assert rows["2:01:222222222:US2222222222:BETA-F"]["alternative_identifier_available"] is True
    assert rows["2:01:222222222:US2222222222:BETA-F"]["alternative_identifier"] == "US2222222222"
    assert rows["2:01:222222222:US2222222222:BETA-F"]["alternative_identifier_type"] == "ISIN"
    assert rows["3:01:333333333:US3333333333:GAM-F"]["successful_row_exists_for_kypermno"] is False
    assert rows["3:01:333333333:US3333333333:GAM-F"]["candidate_successful_ric_available"] is False

    assert summary["failed_lookup_rows"] == 3
    assert summary["successful_lookup_rows"] == 3
    assert summary["diagnostic_flag_counts"] == {
        "successful_row_exists_before_span": 2,
        "successful_row_exists_after_span": 1,
        "successful_row_overlap_exists": 0,
        "alternative_identifier_available": 3,
        "candidate_successful_ric_available": 1,
        "no_successful_row_for_kypermno": 1,
        "multiple_successful_identifier_pairs_or_rics": 1,
    }


def test_build_refinitiv_step1_bridge_universe_raises_on_missing_required_columns() -> None:
    daily = _daily_panel().drop("CUSIP")

    with pytest.raises(ValueError, match="missing required columns"):
        build_refinitiv_step1_bridge_universe(daily.lazy()).collect()


def test_run_refinitiv_step1_bridge_pipeline_writes_all_artifacts(tmp_path: Path) -> None:
    out = run_refinitiv_step1_bridge_pipeline(
        _daily_panel().lazy(),
        tmp_path,
        company_description_lf=_company_description().lazy(),
        source_daily_path=Path("C:/data/final_flagged_data_compdesc_added.parquet"),
    )

    assert set(out) == {
        "refinitiv_bridge_universe_parquet",
        "refinitiv_bridge_universe_csv",
        "refinitiv_bridge_handoff_xlsx",
        "refinitiv_ric_lookup_handoff_xlsx",
        "refinitiv_ric_lookup_handoff_common_stock_xlsx",
        "refinitiv_ric_lookup_handoff_common_stock_extended_xlsx",
        "refinitiv_ric_lookup_handoff_common_stock_with_gvkey_xlsx",
        "refinitiv_step1_manifest",
    }
    for path in out.values():
        assert path.exists()

    parquet_df = pl.read_parquet(out["refinitiv_bridge_universe_parquet"])
    csv_df = pl.read_csv(out["refinitiv_bridge_universe_csv"])
    csv_text = out["refinitiv_bridge_universe_csv"].read_text(encoding="utf-8")
    assert parquet_df.columns == list(BRIDGE_OUTPUT_COLUMNS)
    assert csv_df.columns == list(BRIDGE_OUTPUT_COLUMNS)
    assert parquet_df.height == 4
    assert parquet_df["bridge_row_id"].to_list() == csv_df["bridge_row_id"].to_list()
    assert "2024-01-02" in csv_text
    assert "1:A:11111111:US1111111111:ALFA" in csv_text

    manifest = json.loads(out["refinitiv_step1_manifest"].read_text(encoding="utf-8"))
    assert manifest["pipeline_name"] == "refinitiv_step1_bridge"
    assert manifest["bridge_rows"] == 4
    assert manifest["source_daily_rows"] == 6
    assert manifest["ric_lookup_rows"] == 4
    assert manifest["ric_manual_review_rows"] == 0
    assert manifest["ric_lookup_filter_profiles"] == [
        {
            "artifact_key": "refinitiv_ric_lookup_handoff_common_stock_xlsx",
            "artifact_path": str(out["refinitiv_ric_lookup_handoff_common_stock_xlsx"]),
            "extended_diagnostic_artifact_key": "refinitiv_ric_lookup_handoff_common_stock_extended_xlsx",
            "extended_diagnostic_artifact_path": str(
                out["refinitiv_ric_lookup_handoff_common_stock_extended_xlsx"]
            ),
            "extended_diagnostic_summary": {
                "agreement_counts": {
                    "ISIN_vs_CUSIP_same_cusip": 0,
                    "ISIN_vs_CUSIP_same_isin": 0,
                    "ISIN_vs_CUSIP_same_ric": 0,
                    "ISIN_vs_TICKER_same_cusip": 0,
                    "ISIN_vs_TICKER_same_isin": 0,
                    "ISIN_vs_TICKER_same_ric": 0,
                    "CUSIP_vs_TICKER_same_cusip": 0,
                    "CUSIP_vs_TICKER_same_isin": 0,
                    "CUSIP_vs_TICKER_same_ric": 0,
                    "all_successful_attempts_consistent": 0,
                },
                "attempt_counts_by_identifier_type": {
                    "CUSIP": 2,
                    "ISIN": 2,
                    "TICKER": 3,
                },
                "conflict_counts": {
                    "ISIN_vs_CUSIP_same_cusip": 0,
                    "ISIN_vs_CUSIP_same_isin": 0,
                    "ISIN_vs_CUSIP_same_ric": 0,
                    "ISIN_vs_TICKER_same_cusip": 0,
                    "ISIN_vs_TICKER_same_isin": 0,
                    "ISIN_vs_TICKER_same_ric": 0,
                    "CUSIP_vs_TICKER_same_cusip": 0,
                    "CUSIP_vs_TICKER_same_isin": 0,
                    "CUSIP_vs_TICKER_same_ric": 0,
                    "all_successful_attempts_consistent": 0,
                },
                "rows_where_only_one_identifier_type_succeeds": 0,
                "success_counts_by_identifier_type": {
                    "CUSIP": 0,
                    "ISIN": 0,
                    "TICKER": 0,
                },
            },
            "predicates_applied": [
                "KYPERMNO is not null",
                "common stock (SHRCD in {10, 11})",
            ],
            "profile_name": "common_stock",
            "ric_lookup_rows": 3,
            "ric_manual_review_rows": 0,
            "rows_after_filter": 3,
            "rows_before_filter": 4,
        },
        {
            "artifact_key": "refinitiv_ric_lookup_handoff_common_stock_with_gvkey_xlsx",
            "artifact_path": str(out["refinitiv_ric_lookup_handoff_common_stock_with_gvkey_xlsx"]),
            "predicates_applied": [
                "KYPERMNO is not null",
                "common stock (SHRCD in {10, 11})",
                "KYGVKEY_final is not null",
            ],
            "profile_name": "common_stock_with_gvkey",
            "ric_lookup_rows": 2,
            "ric_manual_review_rows": 0,
            "rows_after_filter": 2,
            "rows_before_filter": 4,
        },
    ]
    assert manifest["source_daily_path"] == str(
        Path("C:/data/final_flagged_data_compdesc_added.parquet")
    )

    with zipfile.ZipFile(out["refinitiv_bridge_handoff_xlsx"]) as workbook_zip:
        workbook_xml = workbook_zip.read("xl/workbook.xml").decode("utf-8")
        shared_strings = workbook_zip.read("xl/sharedStrings.xml").decode("utf-8")

    assert "bridge_universe" in workbook_xml
    assert "README" in workbook_xml
    assert "bridge_row_id" in shared_strings
    assert "TICKER" in shared_strings
    assert "vendor_primary_ric" in shared_strings

    with zipfile.ZipFile(out["refinitiv_ric_lookup_handoff_xlsx"]) as workbook_zip:
        workbook_xml = workbook_zip.read("xl/workbook.xml").decode("utf-8")
        shared_strings = workbook_zip.read("xl/sharedStrings.xml").decode("utf-8")

    assert "ric_lookup" in workbook_xml
    assert "README" in workbook_xml
    assert "preferred_lookup_id" in shared_strings
    assert "preferred_lookup_type" in shared_strings
    assert "TICKER" in shared_strings
    assert "vendor_returned_name" in shared_strings
    assert "vendor_returned_cusip" in shared_strings
    assert "vendor_returned_isin" in shared_strings
    assert "manual_review" not in workbook_xml
    assert "ISIN_lookup_input" not in shared_strings

    with zipfile.ZipFile(out["refinitiv_ric_lookup_handoff_common_stock_xlsx"]) as workbook_zip:
        workbook_xml = workbook_zip.read("xl/workbook.xml").decode("utf-8")
        shared_strings = workbook_zip.read("xl/sharedStrings.xml").decode("utf-8")

    assert "ric_lookup" in workbook_xml
    assert "4:-:44444444:-:DELTA" in shared_strings
    assert "ISIN_lookup_input" not in shared_strings

    with zipfile.ZipFile(out["refinitiv_ric_lookup_handoff_common_stock_extended_xlsx"]) as workbook_zip:
        workbook_xml = workbook_zip.read("xl/workbook.xml").decode("utf-8")
        shared_strings = workbook_zip.read("xl/sharedStrings.xml").decode("utf-8")

    assert "lookup_diagnostics" in workbook_xml
    assert "summary" in workbook_xml
    assert "README" in workbook_xml
    assert "ISIN_lookup_input" in shared_strings
    assert "CUSIP_lookup_input" in shared_strings
    assert "TICKER_lookup_input" in shared_strings
    assert "all_successful_attempts_consistent" in shared_strings
    workbook = load_workbook(out["refinitiv_ric_lookup_handoff_common_stock_extended_xlsx"], read_only=True)
    try:
        headers = [cell for cell in next(workbook["lookup_diagnostics"].iter_rows(max_row=1, values_only=True))]
    finally:
        workbook.close()
    assert "preferred_lookup_id" not in headers
    assert "preferred_lookup_type" not in headers

    with zipfile.ZipFile(out["refinitiv_ric_lookup_handoff_common_stock_with_gvkey_xlsx"]) as workbook_zip:
        shared_strings = workbook_zip.read("xl/sharedStrings.xml").decode("utf-8")

    assert "4:-:44444444:-:DELTA" not in shared_strings


def test_run_refinitiv_null_ric_diagnostics_pipeline_writes_auditable_outputs(
    tmp_path: Path,
) -> None:
    workbook_path = _write_filled_lookup_workbook(tmp_path / "filled_lookup.xlsx")

    out = run_refinitiv_null_ric_diagnostics_pipeline(
        workbook_path,
        tmp_path / "null_ric_diagnostics",
    )

    assert set(out) == {
        "refinitiv_null_ric_diagnostics_summary",
        "refinitiv_null_ric_rescue_candidates_parquet",
        "refinitiv_null_ric_rescue_candidates_csv",
        "refinitiv_ownership_smoke_testing_parquet",
        "refinitiv_ownership_smoke_testing_csv",
        "refinitiv_null_ric_diagnostics_manifest",
        "refinitiv_null_ric_rescue_candidates_review_xlsx",
        "refinitiv_ownership_smoke_testing_xlsx",
    }
    for path in out.values():
        assert path.exists()

    diagnostics_df = pl.read_parquet(out["refinitiv_null_ric_rescue_candidates_parquet"])
    assert diagnostics_df.columns == list(NULL_RIC_DIAGNOSTIC_COLUMNS)
    assert diagnostics_df.height == 8
    assert "rescue_classification" not in diagnostics_df.columns

    sample_df = pl.read_parquet(out["refinitiv_ownership_smoke_testing_parquet"])
    assert sample_df.columns == list(OWNERSHIP_SMOKE_SAMPLE_COLUMNS)
    assert sample_df.height == 10
    assert sample_df.group_by("sample_category").len().sort("sample_category").to_dicts() == [
        {"sample_category": "alternative_identifier_available", "len": 2},
        {"sample_category": "multiple_successful_identifier_pairs_or_rics", "len": 2},
        {"sample_category": "no_successful_row_for_kypermno", "len": 2},
        {"sample_category": "successful_row_exists_after_span", "len": 2},
        {"sample_category": "successful_row_exists_before_span", "len": 2},
    ]
    assert set(sample_df["sample_category"].to_list()) == {
        "successful_row_exists_after_span",
        "successful_row_exists_before_span",
        "alternative_identifier_available",
        "no_successful_row_for_kypermno",
        "multiple_successful_identifier_pairs_or_rics",
    }
    candidate_row = sample_df.filter(
        (pl.col("sample_category") == "successful_row_exists_after_span")
        & (pl.col("bridge_row_id") == "1:01:111111111:US1111111111:ALFA-AFTER")
    ).row(0, named=True)
    assert candidate_row["lookup_input"] == "AAA.O"
    assert candidate_row["lookup_input_source"] == "candidate_successful_ric"

    alternative_row = sample_df.filter(
        (pl.col("sample_category") == "multiple_successful_identifier_pairs_or_rics")
        & (pl.col("bridge_row_id") == "2:01:222222222:US2222222222:BETA-F1")
    ).row(0, named=True)
    assert alternative_row["lookup_input"] == "US2222222222"
    assert alternative_row["lookup_input_source"] == "alternative_identifier"

    ticker_row = sample_df.filter(
        (pl.col("sample_category") == "no_successful_row_for_kypermno")
        & (pl.col("bridge_row_id") == "5:01:-:-:EPSI-F")
    ).row(0, named=True)
    assert ticker_row["lookup_input"] == "EPSI"
    assert ticker_row["lookup_input_source"] == "TICKER"
    assert ticker_row["TICKER"] == "EPSI"

    review_df = _build_null_ric_review_frame(diagnostics_df)
    rebuilt_sample_df, rebuilt_metadata = build_refinitiv_ownership_smoke_sample(review_df)
    assert sample_df.to_dicts() == rebuilt_sample_df.to_dicts()
    assert rebuilt_metadata["category_source"] == "derived_from_review_flags"

    summary = json.loads(out["refinitiv_null_ric_diagnostics_summary"].read_text(encoding="utf-8"))
    assert summary["pipeline_name"] == "refinitiv_null_ric_diagnostics"
    assert summary["failed_lookup_rows"] == 8
    assert summary["successful_lookup_rows"] == 4
    assert summary["ownership_smoke_sample_source"] == "failed_lookup_review"
    assert summary["ownership_smoke_category_field_used"] == "sample_category"
    assert summary["ownership_smoke_category_source"] == "derived_from_review_flags"
    assert summary["ownership_smoke_block_count"] == 10
    assert summary["ownership_smoke_lookup_input_priority"] == [
        "candidate_successful_ric",
        "alternative_identifier",
        "TICKER",
        "preferred_lookup_id",
    ]
    assert summary["ownership_smoke_category_counts"] == {
        "successful_row_exists_after_span": 2,
        "successful_row_exists_before_span": 2,
        "alternative_identifier_available": 2,
        "no_successful_row_for_kypermno": 2,
        "multiple_successful_identifier_pairs_or_rics": 2,
    }

    manifest = json.loads(out["refinitiv_null_ric_diagnostics_manifest"].read_text(encoding="utf-8"))
    assert manifest["artifacts"]["refinitiv_null_ric_rescue_candidates_parquet"] == str(
        out["refinitiv_null_ric_rescue_candidates_parquet"]
    )
    assert manifest["artifacts"]["refinitiv_null_ric_rescue_candidates_review_xlsx"] == str(
        out["refinitiv_null_ric_rescue_candidates_review_xlsx"]
    )
    assert manifest["artifacts"]["refinitiv_ownership_smoke_testing_xlsx"] == str(
        out["refinitiv_ownership_smoke_testing_xlsx"]
    )

    with zipfile.ZipFile(out["refinitiv_null_ric_rescue_candidates_review_xlsx"]) as workbook_zip:
        workbook_xml = workbook_zip.read("xl/workbook.xml").decode("utf-8")
        shared_strings = workbook_zip.read("xl/sharedStrings.xml").decode("utf-8")

    assert "failed_lookup_review" in workbook_xml
    assert "alternative_identifier" in shared_strings
    assert "TICKER" in shared_strings
    assert "test_result" in shared_strings

    with zipfile.ZipFile(out["refinitiv_ownership_smoke_testing_xlsx"]) as workbook_zip:
        workbook_xml = workbook_zip.read("xl/workbook.xml").decode("utf-8")

    assert "ownership_smoke" in workbook_xml

    smoke_workbook = load_workbook(out["refinitiv_ownership_smoke_testing_xlsx"], data_only=True)
    try:
        assert "ownership_smoke" in smoke_workbook.sheetnames
        worksheet = smoke_workbook["ownership_smoke"]
        assert worksheet.max_column == sample_df.height * 5

        block_headers = [
            "input_data",
            "returned_ric",
            "returned_date",
            "returned_value",
            "returned_category",
        ]
        for block_idx, row in enumerate(sample_df.iter_rows(named=True)):
            base_col = (block_idx * 5) + 1
            assert [worksheet.cell(row=1, column=base_col + offset).value for offset in range(5)] == block_headers
            assert worksheet.cell(row=2, column=base_col).value == row["bridge_row_id"]
            assert worksheet.cell(row=3, column=base_col).value == row["lookup_input"]
            assert worksheet.cell(row=4, column=base_col).value == row["request_start_date"]
            assert worksheet.cell(row=5, column=base_col).value == row["request_end_date"]
            assert worksheet.cell(row=6, column=base_col).value == row["sample_category"]
            for blank_col in range(base_col + 1, base_col + 5):
                for blank_row in range(2, 7):
                    assert worksheet.cell(row=blank_row, column=blank_col).value is None
    finally:
        smoke_workbook.close()


def test_run_refinitiv_step1_bridge_pipeline_writes_manual_review_sheet_when_needed(
    tmp_path: Path,
) -> None:
    out = run_refinitiv_step1_bridge_pipeline(
        _daily_panel()
        .vstack(
            pl.DataFrame(
                {
                    "KYPERMNO": [3],
                    "CALDT": [date(2024, 1, 6)],
                    "KYGVKEY_final": ["3000"],
                    "LIID": [None],
                    "CIK_final": ["0000003000"],
                    "CUSIP": [None],
                    "ISIN": [None],
                    "TICKER": [None],
                    "LINKTYPE": [None],
                    "LINKPRIM": [None],
                    "link_quality_flag": [None],
                    "HEXCNTRY": [None],
                    "n_filings": [0],
                    "SHRCD": [10],
                    "EXCHCD": [1],
                }
            )
        )
        .lazy(),
        tmp_path,
    )

    manifest = json.loads(out["refinitiv_step1_manifest"].read_text(encoding="utf-8"))
    assert manifest["bridge_rows"] == 5
    assert manifest["ric_lookup_rows"] == 4
    assert manifest["ric_manual_review_rows"] == 1

    with zipfile.ZipFile(out["refinitiv_ric_lookup_handoff_xlsx"]) as workbook_zip:
        workbook_xml = workbook_zip.read("xl/workbook.xml").decode("utf-8")
        shared_strings = workbook_zip.read("xl/sharedStrings.xml").decode("utf-8")

    assert "manual_review" in workbook_xml
    assert "3:-:-:-:-" in shared_strings
