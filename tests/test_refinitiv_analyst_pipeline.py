from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.pipelines.refinitiv.analyst import (
    build_refinitiv_analyst_normalized_outputs,
    build_refinitiv_lm2011_doc_analyst_anchors,
    build_refinitiv_step1_analyst_request_groups,
    run_refinitiv_lm2011_doc_analyst_select_pipeline,
    select_refinitiv_lm2011_doc_analyst_inputs,
)
from thesis_pkg.pipelines.refinitiv.instrument_authority import (
    build_refinitiv_step1_instrument_authority_frame,
)


def test_build_refinitiv_step1_instrument_authority_frame_uses_lm2011_gvkey_contract() -> None:
    bridge = pl.DataFrame(
        {
            "bridge_row_id": ["bridge-1"],
            "KYPERMNO": [1],
            "KYGVKEY_final": ["1000"],
            "first_seen_caldt": [dt.date(2020, 1, 1)],
            "last_seen_caldt": [dt.date(2020, 12, 31)],
        }
    )
    resolution = pl.DataFrame(
        {
            "bridge_row_id": ["bridge-1"],
            "effective_collection_ric": ["AAA.N"],
            "effective_collection_ric_source": ["ISIN"],
            "effective_resolution_status": ["effective_from_accepted_ric"],
        }
    )

    authority = build_refinitiv_step1_instrument_authority_frame(bridge, resolution)

    assert authority.columns == [
        "bridge_row_id",
        "KYPERMNO",
        "gvkey_int",
        "gvkey_source_column",
        "first_seen_caldt",
        "last_seen_caldt",
        "effective_collection_ric",
        "effective_collection_ric_source",
        "effective_resolution_status",
        "authority_eligible",
        "authority_exclusion_reason",
    ]
    row = authority.row(0, named=True)
    assert row["gvkey_int"] == 1000
    assert row["gvkey_source_column"] == "KYGVKEY_final"
    assert row["authority_eligible"] is True


def test_build_refinitiv_step1_analyst_request_groups_collapses_bridge_rows() -> None:
    authority = pl.DataFrame(
        {
            "bridge_row_id": ["bridge-1", "bridge-2"],
            "KYPERMNO": [1, 1],
            "gvkey_int": [1000, 1000],
            "gvkey_source_column": ["KYGVKEY_final", "KYGVKEY_final"],
            "first_seen_caldt": [dt.date(2020, 1, 1), dt.date(2020, 3, 1)],
            "last_seen_caldt": [dt.date(2020, 2, 1), dt.date(2020, 9, 30)],
            "effective_collection_ric": ["AAA.N", "AAA.N"],
            "effective_collection_ric_source": ["ISIN", "ISIN"],
            "effective_resolution_status": ["ok", "ok"],
            "authority_eligible": [True, True],
            "authority_exclusion_reason": [None, None],
        }
    )

    membership, request_groups = build_refinitiv_step1_analyst_request_groups(authority)

    assert membership.height == 2
    assert request_groups.height == 1
    row = request_groups.row(0, named=True)
    assert row["member_bridge_row_count"] == 2
    assert row["bridge_start_date_min"] == dt.date(2020, 1, 1)
    assert row["bridge_end_date_max"] == dt.date(2020, 9, 30)
    assert row["actuals_request_start_date"] == dt.date(2019, 12, 1)
    assert row["actuals_request_end_date"] == dt.date(2021, 1, 28)
    assert row["estimates_request_start_date"] == dt.date(2019, 4, 6)
    assert row["estimates_request_end_date"] == dt.date(2020, 10, 31)


def test_build_refinitiv_analyst_normalized_outputs_supports_unique_period_derivation() -> None:
    actuals = pl.DataFrame(
        {
            "item_id": ["actual-item-1"],
            "response_row_index": [0],
            "request_group_id": ["group-1"],
            "gvkey_int": [1000],
            "effective_collection_ric": ["AAA.N"],
            "announcement_date": [dt.date(2023, 10, 15)],
            "fiscal_period_end": [None],
            "actual_eps": [1.5],
            "raw_fperiod": [None],
            "row_parse_status": ["OK"],
        }
    )
    estimates = pl.DataFrame(
        {
            "item_id": ["estimate-item-1", "estimate-item-2", "estimate-item-3"],
            "response_row_index": [0, 1, 2],
            "request_group_id": ["group-1", "group-1", "group-1"],
            "request_period": ["FQ1", "FQ2", "FQ1"],
            "gvkey_int": [1000, 1000, 1000],
            "effective_collection_ric": ["AAA.N", "AAA.N", "AAA.N"],
            "calc_date": [dt.date(2023, 10, 1), dt.date(2023, 6, 1), dt.date(2023, 9, 1)],
            "fiscal_period_end": [dt.date(2023, 9, 30), dt.date(2023, 9, 30), dt.date(2023, 9, 30)],
            "raw_fperiod": ["FQ1", "FQ2", "FQ1"],
            "forecast_consensus_mean": [1.2, 1.0, 1.1],
            "forecast_dispersion": [0.2, 0.1, 0.15],
            "estimate_count": [10, 9, 9],
            "row_parse_status": ["OK", "OK", "OK"],
        }
    )

    normalized, rejections = build_refinitiv_analyst_normalized_outputs(actuals, estimates)

    assert rejections.height == 0
    assert normalized.height == 1
    row = normalized.row(0, named=True)
    assert row["fiscal_period_end"] == dt.date(2023, 9, 30)
    assert row["actual_fiscal_period_end_origin"] == "ESTIMATE_FALLBACK"
    assert row["forecast_consensus_mean"] == pytest.approx(1.2)
    assert row["forecast_revision_4m"] == pytest.approx(0.2)
    assert row["forecast_revision_4m_status"] == "OK"


def test_build_refinitiv_analyst_normalized_outputs_rejects_conflicting_duplicates_and_nonunique_derivation() -> None:
    actuals = pl.DataFrame(
        {
            "item_id": ["actual-item-1", "actual-item-2"],
            "response_row_index": [0, 0],
            "request_group_id": ["group-1", "group-2"],
            "gvkey_int": [1000, 1001],
            "effective_collection_ric": ["AAA.N", "BBB.N"],
            "announcement_date": [dt.date(2023, 10, 15), dt.date(2023, 10, 20)],
            "fiscal_period_end": [dt.date(2023, 9, 30), None],
            "actual_eps": [1.5, 2.0],
            "raw_fperiod": [None, None],
            "row_parse_status": ["OK", "OK"],
        }
    )
    conflicting_actuals = actuals.vstack(
        pl.DataFrame(
            {
                "item_id": ["actual-item-1b"],
                "response_row_index": [0],
                "request_group_id": ["group-1b"],
                "gvkey_int": [1000],
                "effective_collection_ric": ["AAA.N"],
                "announcement_date": [dt.date(2023, 10, 15)],
                "fiscal_period_end": [dt.date(2023, 9, 30)],
                "actual_eps": [1.6],
                "raw_fperiod": [None],
                "row_parse_status": ["OK"],
            }
        )
    )
    estimates = pl.DataFrame(
        {
            "item_id": ["estimate-item-2a", "estimate-item-2b", "estimate-item-2c"],
            "response_row_index": [0, 1, 2],
            "request_group_id": ["group-2a", "group-2b", "group-2c"],
            "request_period": ["FQ1", "FQ2", "FQ1"],
            "gvkey_int": [1001, 1001, 1001],
            "effective_collection_ric": ["BBB.N", "BBB.N", "BBB.N"],
            "calc_date": [dt.date(2023, 10, 1), dt.date(2023, 10, 1), dt.date(2023, 6, 1)],
            "fiscal_period_end": [dt.date(2023, 9, 30), dt.date(2023, 6, 30), dt.date(2023, 9, 30)],
            "raw_fperiod": ["FQ1", "FQ2", "FQ1"],
            "forecast_consensus_mean": [1.8, 1.7, 1.4],
            "forecast_dispersion": [0.2, 0.3, 0.15],
            "estimate_count": [8, 7, 8],
            "row_parse_status": ["OK", "OK", "OK"],
        }
    )

    normalized, rejections = build_refinitiv_analyst_normalized_outputs(conflicting_actuals, estimates)

    assert normalized.height == 0
    statuses = set(rejections.get_column("rejection_status").to_list())
    assert "CONFLICTING_DUPLICATE_ACTUALS" in statuses
    assert "NONUNIQUE_DERIVED_FISCAL_PERIOD_END" in statuses


def test_build_refinitiv_analyst_normalized_outputs_survives_missing_revision_base() -> None:
    actuals = pl.DataFrame(
        {
            "item_id": ["actual-item-1"],
            "response_row_index": [0],
            "request_group_id": ["group-1"],
            "gvkey_int": [1000],
            "effective_collection_ric": ["AAA.N"],
            "announcement_date": [dt.date(2024, 10, 30)],
            "fiscal_period_end": [dt.date(2024, 9, 30)],
            "actual_eps": [2.5],
            "raw_fperiod": ["FY2025Q1"],
            "row_parse_status": ["OK"],
        }
    )
    estimates = pl.DataFrame(
        {
            "item_id": ["estimate-item-1"],
            "response_row_index": [0],
            "request_group_id": ["group-1"],
            "request_period": ["FQ1"],
            "gvkey_int": [1000],
            "effective_collection_ric": ["AAA.N"],
            "calc_date": [dt.date(2024, 9, 30)],
            "fiscal_period_end": [dt.date(2024, 9, 30)],
            "raw_fperiod": ["FY2025Q1"],
            "forecast_consensus_mean": [2.1],
            "forecast_dispersion": [0.2],
            "estimate_count": [8],
            "row_parse_status": ["OK"],
        }
    )

    normalized, rejections = build_refinitiv_analyst_normalized_outputs(actuals, estimates)

    assert rejections.height == 0
    assert normalized.height == 1
    row = normalized.row(0, named=True)
    assert row["actual_fiscal_period_end_origin"] == "API_DIRECT"
    assert row["forecast_revision_4m"] is None
    assert row["forecast_revision_4m_status"] == "MISSING_BASE_SNAPSHOT"


def test_build_refinitiv_analyst_normalized_outputs_rejects_conflicting_revision_base() -> None:
    actuals = pl.DataFrame(
        {
            "item_id": ["actual-item-1"],
            "response_row_index": [0],
            "request_group_id": ["group-1"],
            "gvkey_int": [1000],
            "effective_collection_ric": ["AAA.N"],
            "announcement_date": [dt.date(2024, 9, 30)],
            "fiscal_period_end": [dt.date(2024, 9, 30)],
            "actual_eps": [2.5],
            "raw_fperiod": ["FY2024Q4"],
            "row_parse_status": ["OK"],
        }
    )
    estimates = pl.DataFrame(
        {
            "item_id": ["estimate-item-1", "estimate-item-2", "estimate-item-3"],
            "response_row_index": [0, 1, 2],
            "request_group_id": ["group-1", "group-1b", "group-1"],
            "request_period": ["FQ1", "FQ2", "FQ1"],
            "gvkey_int": [1000, 1000, 1000],
            "effective_collection_ric": ["AAA.N", "AAA.N", "AAA.N"],
            "calc_date": [dt.date(2024, 9, 30), dt.date(2024, 5, 31), dt.date(2024, 5, 31)],
            "fiscal_period_end": [dt.date(2024, 9, 30), dt.date(2024, 9, 30), dt.date(2024, 9, 30)],
            "raw_fperiod": ["FY2024Q4", "FY2024Q4", "FY2024Q4"],
            "forecast_consensus_mean": [2.1, 1.8, 1.7],
            "forecast_dispersion": [0.2, 0.1, 0.1],
            "estimate_count": [8, 8, 8],
            "row_parse_status": ["OK", "OK", "OK"],
        }
    )

    normalized, rejections = build_refinitiv_analyst_normalized_outputs(actuals, estimates)

    assert normalized.height == 0
    assert "CONFLICTING_REVISION_BASE_4M" in set(rejections.get_column("rejection_status").to_list())


def test_build_refinitiv_analyst_normalized_outputs_uses_month_end_revision_cutoff() -> None:
    actuals = pl.DataFrame(
        {
            "item_id": ["actual-item-1"],
            "response_row_index": [0],
            "request_group_id": ["group-1"],
            "gvkey_int": [1000],
            "effective_collection_ric": ["AAA.N"],
            "announcement_date": [dt.date(2024, 10, 1)],
            "fiscal_period_end": [dt.date(2024, 9, 30)],
            "actual_eps": [2.5],
            "raw_fperiod": ["FY2024Q4"],
            "row_parse_status": ["OK"],
        }
    )
    estimates = pl.DataFrame(
        {
            "item_id": ["estimate-item-1", "estimate-item-2"],
            "response_row_index": [0, 1],
            "request_group_id": ["group-1", "group-1"],
            "request_period": ["FQ1", "FQ2"],
            "gvkey_int": [1000, 1000],
            "effective_collection_ric": ["AAA.N", "AAA.N"],
            "calc_date": [dt.date(2024, 9, 30), dt.date(2024, 5, 31)],
            "fiscal_period_end": [dt.date(2024, 9, 30), dt.date(2024, 9, 30)],
            "raw_fperiod": ["FY2024Q4", "FY2024Q4"],
            "forecast_consensus_mean": [2.1, 1.7],
            "forecast_dispersion": [0.2, 0.15],
            "estimate_count": [8, 8],
            "row_parse_status": ["OK", "OK"],
        }
    )

    normalized, rejections = build_refinitiv_analyst_normalized_outputs(actuals, estimates)

    assert rejections.height == 0
    row = normalized.row(0, named=True)
    assert row["revision_base_calc_date_4m"] == dt.date(2024, 5, 31)
    assert row["forecast_revision_4m"] == pytest.approx(0.4)


def test_select_refinitiv_lm2011_doc_analyst_inputs_supports_exact_and_safe_fallback() -> None:
    docs = pl.DataFrame(
        {
            "doc_id": ["exact_doc", "fallback_doc", "ambiguous_doc"],
            "filing_date": [dt.date(2023, 9, 1)] * 3,
            "gvkey_int": [1000, 1001, 1002],
            "KYPERMNO": [1, 2, 3],
            "quarter_report_date": [dt.date(2023, 10, 15), dt.date(2023, 10, 20), dt.date(2023, 10, 25)],
            "quarter_fiscal_period_end": [dt.date(2023, 9, 30), None, None],
        }
    )
    panel = pl.DataFrame(
        {
            "gvkey_int": [1000, 1001, 1002, 1002],
            "announcement_date": [dt.date(2023, 10, 15), dt.date(2023, 10, 20), dt.date(2023, 10, 25), dt.date(2023, 10, 25)],
            "fiscal_period_end": [dt.date(2023, 9, 30), dt.date(2023, 9, 25), dt.date(2023, 9, 30), dt.date(2023, 6, 30)],
            "actual_eps": [1.5, 2.0, 2.5, 2.6],
            "forecast_consensus_mean": [1.0, 1.5, 2.1, 2.0],
            "forecast_dispersion": [0.2, 0.3, 0.2, 0.2],
            "forecast_revision_4m": [0.2, None, 0.3, 0.3],
            "forecast_revision_4m_status": ["OK", "MISSING_BASE_SNAPSHOT", "OK", "OK"],
            "actual_fiscal_period_end_origin": ["API_DIRECT", "ESTIMATE_FALLBACK", "API_DIRECT", "API_DIRECT"],
        }
    )

    selected = select_refinitiv_lm2011_doc_analyst_inputs(docs, panel).sort("doc_id")

    assert selected.filter(pl.col("doc_id") == "exact_doc").item(0, "match_type") == "EXACT"
    assert selected.filter(pl.col("doc_id") == "fallback_doc").item(0, "match_type") == "ANNOUNCEMENT_DATE_UNIQUE_FALLBACK"
    assert selected.filter(pl.col("doc_id") == "fallback_doc").item(0, "forecast_revision_4m_status") == "MISSING_BASE_SNAPSHOT"
    assert selected.filter(pl.col("doc_id") == "fallback_doc").item(0, "actual_fiscal_period_end_origin") == "ESTIMATE_FALLBACK"
    assert selected.filter(pl.col("doc_id") == "fallback_doc").item(0, "forecast_revision_4m") is None
    assert selected.filter(pl.col("doc_id") == "ambiguous_doc").item(0, "analyst_match_status") == "AMBIGUOUS_FALLBACK_REJECTED"


def test_run_refinitiv_lm2011_doc_analyst_select_pipeline_writes_doc_level_output(tmp_path: Path) -> None:
    anchors = pl.DataFrame(
        {
            "doc_id": ["doc-1"],
            "filing_date": [dt.date(2023, 9, 1)],
            "gvkey_int": [1000],
            "KYPERMNO": [1],
            "quarter_report_date": [dt.date(2023, 10, 15)],
            "quarter_fiscal_period_end": [dt.date(2023, 9, 30)],
            "anchor_eligible": [True],
            "anchor_exclusion_reason": [None],
        }
    )
    analyst_panel = pl.DataFrame(
        {
            "gvkey_int": [1000],
            "announcement_date": [dt.date(2023, 10, 15)],
            "fiscal_period_end": [dt.date(2023, 9, 30)],
            "actual_eps": [1.5],
            "forecast_consensus_mean": [1.0],
            "forecast_dispersion": [0.2],
            "forecast_revision_4m": [0.2],
        }
    )
    anchors_path = tmp_path / "anchors.parquet"
    panel_path = tmp_path / "panel.parquet"
    anchors.write_parquet(anchors_path)
    analyst_panel.write_parquet(panel_path)

    out = run_refinitiv_lm2011_doc_analyst_select_pipeline(
        doc_anchors_artifact_path=anchors_path,
        analyst_normalized_panel_artifact_path=panel_path,
        output_dir=tmp_path,
    )

    selected = pl.read_parquet(out["refinitiv_doc_analyst_selected_parquet"])
    assert selected.height == 1
    assert selected.item(0, "analyst_match_status") == "MATCHED"
