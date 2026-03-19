from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from thesis_pkg.pipeline import (
    run_refinitiv_lm2011_doc_ownership_exact_api_pipeline,
    run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline,
    run_refinitiv_lm2011_doc_ownership_finalize_pipeline,
    run_refinitiv_step1_lookup_api_pipeline,
    run_refinitiv_step1_ownership_universe_api_pipeline,
    run_refinitiv_step1_resolution_pipeline,
)
from thesis_pkg.pipelines.refinitiv.doc_ownership import DOC_OWNERSHIP_REQUEST_COLUMNS
from thesis_pkg.pipelines.refinitiv.doc_ownership import _normalize_date_value
from thesis_pkg.pipelines.refinitiv.lseg_batching import RequestItem
from thesis_pkg.pipelines.refinitiv.lseg_ownership_api import _normalize_ownership_universe_batch_response
from thesis_pkg.pipelines.refinitiv import lseg_provider
from thesis_pkg.pipelines.refinitiv.lseg_ledger import RequestLedger
from thesis_pkg.pipelines.refinitiv.lseg_lookup_api import _classify_error
from thesis_pkg.pipelines.refinitiv.lseg_provider import LsegDataResponse, LsegResponseMetadata
from thesis_pkg.pipelines.refinitiv.lseg_provider import LsegRequestError
from thesis_pkg.pipelines.refinitiv_bridge_pipeline import (
    OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS,
    RIC_LOOKUP_COLUMNS,
    build_refinitiv_lookup_extended_diagnostic_artifact,
)
from thesis_pkg.pipelines.refinitiv.lseg_batching import batch_items


class FakeProvider:
    def __init__(self, responses: list[pl.DataFrame]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_data(
        self,
        *,
        universe: list[str],
        fields: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> LsegDataResponse:
        self.calls.append(
            {
                "universe": list(universe),
                "fields": list(fields),
                "parameters": dict(parameters or {}),
            }
        )
        if not self._responses:
            raise AssertionError("fake provider exhausted")
        return LsegDataResponse(
            frame=self._responses.pop(0),
            metadata=LsegResponseMetadata(
                status_code=200,
                headers={"X-Request-Limit-Remaining": "100"},
                latency_ms=25,
                response_bytes=256,
                fingerprint="fake",
            ),
        )


class ErrorProvider:
    def __init__(self, errors: list[Exception]) -> None:
        self._errors = list(errors)
        self.calls: list[dict[str, Any]] = []

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_data(
        self,
        *,
        universe: list[str],
        fields: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> LsegDataResponse:
        self.calls.append(
            {
                "universe": list(universe),
                "fields": list(fields),
                "parameters": dict(parameters or {}),
            }
        )
        if not self._errors:
            raise AssertionError("error provider exhausted")
        raise self._errors.pop(0)


def test_lookup_api_pipeline_builds_extended_parquet_and_resolution_from_snapshot(tmp_path: Path) -> None:
    lookup_df = pl.DataFrame(
        {
            "bridge_row_id": [
                "1:A:111111111:US1111111111:ALFA-1",
                "1:A:111111111:US1111111111:ALFA-2",
            ],
            "KYPERMNO": ["1", "1"],
            "CUSIP": ["111111111", "111111111"],
            "ISIN": ["US1111111111", "US1111111111"],
            "TICKER": ["ALFA", "ALFA"],
            "first_seen_caldt": [date(2024, 1, 1), date(2024, 1, 2)],
            "last_seen_caldt": [date(2024, 1, 10), date(2024, 1, 11)],
            "preferred_lookup_id": ["US1111111111", "US1111111111"],
            "preferred_lookup_type": ["ISIN", "ISIN"],
            "vendor_primary_ric": [None, None],
            "vendor_returned_name": [None, None],
            "vendor_returned_cusip": [None, None],
            "vendor_returned_isin": [None, None],
            "vendor_match_status": [None, None],
            "vendor_notes": [None, None],
        }
    ).select(RIC_LOOKUP_COLUMNS)
    snapshot_df, _, _ = build_refinitiv_lookup_extended_diagnostic_artifact(lookup_df)
    snapshot_path = tmp_path / "refinitiv_ric_lookup_handoff_common_stock_extended_snapshot.parquet"
    snapshot_df.write_parquet(snapshot_path)

    provider = FakeProvider(
        [
            pl.DataFrame(
                {
                    "Instrument": ["111111111"],
                    "RIC": ["AAA.N"],
                    "Company Common Name": ["Alpha Inc"],
                    "ISIN": ["US1111111111"],
                    "CUSIP": ["111111111"],
                }
            ),
            pl.DataFrame(
                {
                    "Instrument": ["ALFA"],
                    "RIC": ["AAA.N"],
                    "Company Common Name": ["Alpha Inc"],
                    "ISIN": ["US1111111111"],
                    "CUSIP": ["111111111"],
                }
            ),
            pl.DataFrame(
                {
                    "Instrument": ["US1111111111"],
                    "RIC": ["AAA.N"],
                    "Company Common Name": ["Alpha Inc"],
                    "ISIN": ["US1111111111"],
                    "CUSIP": ["111111111"],
                }
            )
        ]
    )
    out = run_refinitiv_step1_lookup_api_pipeline(
        snapshot_parquet_path=snapshot_path,
        output_dir=tmp_path,
        provider=provider,
    )

    assert len(provider.calls) == 3
    assert {tuple(call["universe"]) for call in provider.calls} == {
        ("111111111",),
        ("ALFA",),
        ("US1111111111",),
    }
    lookup_result_df = pl.read_parquet(out["refinitiv_ric_lookup_handoff_common_stock_extended_parquet"])
    assert lookup_result_df.height == 2
    assert lookup_result_df.select(pl.col("ISIN_returned_ric").drop_nulls().unique()).to_series(0).to_list() == ["AAA.N"]

    resolution_out = run_refinitiv_step1_resolution_pipeline(
        filled_lookup_workbook_path=out["refinitiv_ric_lookup_handoff_common_stock_extended_parquet"],
        output_dir=tmp_path / "resolution",
    )
    resolution_df = pl.read_parquet(resolution_out["refinitiv_ric_resolution_common_stock_parquet"])
    assert resolution_df.height == 2
    assert resolution_df.select(pl.col("effective_collection_ric").drop_nulls().unique()).to_series(0).to_list() == ["AAA.N"]


def test_ownership_universe_api_pipeline_writes_results_and_summary(tmp_path: Path) -> None:
    handoff_row = {name: None for name in OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS}
    handoff_row.update(
        {
            "bridge_row_id": "1:A:111111111:US1111111111:ALFA",
            "KYPERMNO": "1",
            "CUSIP": "111111111",
            "ISIN": "US1111111111",
            "TICKER": "ALFA",
            "first_seen_caldt": date(2024, 1, 1),
            "last_seen_caldt": date(2024, 1, 31),
            "accepted_ric": "AAA.N",
            "accepted_ric_source": "ISIN",
            "accepted_resolution_status": "resolved_from_isin",
            "conventional_identity_conflict": False,
            "ticker_candidate_available": False,
            "effective_collection_ric": "AAA.N",
            "effective_collection_ric_source": "ISIN",
            "effective_resolution_status": "effective_from_accepted_ric",
            "diagnostic_case_id": "1:A:111111111:US1111111111:ALFA",
            "candidate_slot": "UNIVERSE_EFFECTIVE",
            "candidate_ric": "AAA.N",
            "ownership_lookup_row_id": "1:A:111111111:US1111111111:ALFA|UNIVERSE_EFFECTIVE",
            "ownership_lookup_role": "UNIVERSE_EFFECTIVE",
            "lookup_input": "AAA.N",
            "lookup_input_source": "effective_collection_ric",
            "request_start_date": "2024-01-01",
            "request_end_date": "2024-01-31",
            "retrieval_eligible": True,
        }
    )
    handoff_path = tmp_path / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    pl.DataFrame([handoff_row]).write_parquet(handoff_path)

    provider = FakeProvider(
        [
            pl.DataFrame(
                {
                    "Instrument": ["AAA.N", "AAA.N"],
                    "Date": [date(2024, 1, 31), date(2024, 1, 15)],
                    "Category Percent Of Traded Shares": [55.0, 52.0],
                    "Investor Statistics Category Value": ["Holdings by Institutions", "Holdings by Institutions"],
                }
            )
        ]
    )
    out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
    )

    results_df = pl.read_parquet(out["refinitiv_ownership_universe_results_parquet"])
    row_summary_df = pl.read_parquet(out["refinitiv_ownership_universe_row_summary_parquet"])
    assert results_df.height == 2
    assert row_summary_df.height == 1
    assert row_summary_df.item(0, "ownership_rows_returned") == 2
    assert row_summary_df.item(0, "ownership_single_returned_ric") is True


def test_ownership_universe_api_pipeline_treats_unresolved_identifier_as_empty_result(tmp_path: Path) -> None:
    handoff_row = {name: None for name in OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS}
    handoff_row.update(
        {
            "bridge_row_id": "1:A:111111111:US1111111111:QRHC",
            "KYPERMNO": "1",
            "CUSIP": None,
            "ISIN": None,
            "TICKER": "QRHC",
            "first_seen_caldt": date(2014, 5, 19),
            "last_seen_caldt": date(2016, 8, 12),
            "accepted_ric": None,
            "accepted_ric_source": None,
            "accepted_resolution_status": "unresolved_after_isin_cusip",
            "conventional_identity_conflict": False,
            "ticker_candidate_available": True,
            "effective_collection_ric": "QRHC",
            "effective_collection_ric_source": "TICKER",
            "effective_resolution_status": "effective_extended_from_ticker_candidate",
            "diagnostic_case_id": "1:A:111111111:US1111111111:QRHC",
            "candidate_slot": "UNIVERSE_EFFECTIVE",
            "candidate_ric": "QRHC",
            "ownership_lookup_row_id": "1:A:111111111:US1111111111:QRHC|UNIVERSE_EFFECTIVE",
            "ownership_lookup_role": "UNIVERSE_EFFECTIVE",
            "lookup_input": "QRHC",
            "lookup_input_source": "effective_collection_ric",
            "request_start_date": "2014-05-19",
            "request_end_date": "2016-08-12",
            "retrieval_eligible": True,
        }
    )
    handoff_path = tmp_path / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    pl.DataFrame([handoff_row]).write_parquet(handoff_path)

    provider = ErrorProvider(
        [
            LsegRequestError(
                "Unable to resolve all requested identifiers in ['QRHC'].",
                error_kind="unresolved_identifiers",
                unresolved_identifiers=("QRHC",),
            )
        ]
    )
    out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
    )

    results_df = pl.read_parquet(out["refinitiv_ownership_universe_results_parquet"])
    row_summary_df = pl.read_parquet(out["refinitiv_ownership_universe_row_summary_parquet"])
    assert results_df.height == 0
    assert row_summary_df.height == 1
    assert row_summary_df.item(0, "ownership_rows_returned") == 0
    assert row_summary_df.item(0, "ownership_single_returned_ric") is False

    request_log = [json.loads(line) for line in Path(out["refinitiv_ownership_universe_api_requests_jsonl"]).read_text().splitlines()]
    assert request_log[-1]["event"] == "request_unresolved_identifiers_treated_as_empty"
    assert request_log[-1]["unresolved_identifiers"] == ["QRHC"]


def test_ownership_universe_api_pipeline_ignores_blank_shell_rows(tmp_path: Path) -> None:
    handoff_row = {name: None for name in OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS}
    handoff_row.update(
        {
            "bridge_row_id": "10048:01:156503104:US1565031044:CTY",
            "KYPERMNO": "10048",
            "CUSIP": "156503104",
            "ISIN": "US1565031044",
            "TICKER": "CTY",
            "first_seen_caldt": date(1990, 1, 2),
            "last_seen_caldt": date(1995, 1, 4),
            "accepted_ric": "CYCC.PK^A02",
            "accepted_ric_source": "ISIN",
            "accepted_resolution_status": "resolved_conventional_agree",
            "conventional_identity_conflict": False,
            "ticker_candidate_available": False,
            "effective_collection_ric": "CYCC.PK^A02",
            "effective_collection_ric_source": "ISIN",
            "effective_resolution_status": "effective_from_accepted_ric",
            "diagnostic_case_id": "10048:01:156503104:US1565031044:CTY",
            "candidate_slot": "UNIVERSE_EFFECTIVE",
            "candidate_ric": "CYCC.PK^A02",
            "ownership_lookup_row_id": "10048:01:156503104:US1565031044:CTY|UNIVERSE_EFFECTIVE",
            "ownership_lookup_role": "UNIVERSE_EFFECTIVE",
            "lookup_input": "CYCC.PK^A02",
            "lookup_input_source": "effective_collection_ric",
            "request_start_date": "1990-01-02",
            "request_end_date": "1995-01-04",
            "retrieval_eligible": True,
        }
    )
    handoff_path = tmp_path / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    pl.DataFrame([handoff_row]).write_parquet(handoff_path)

    provider = FakeProvider(
        [
            pl.DataFrame(
                {
                    "Instrument": ["CYCC.PK^A02"],
                    "Date": [None],
                    "Category Percent Of Traded Shares": [None],
                    "Investor Statistics Category Value": [None],
                }
            )
        ]
    )
    out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
    )

    results_df = pl.read_parquet(out["refinitiv_ownership_universe_results_parquet"])
    row_summary_df = pl.read_parquet(out["refinitiv_ownership_universe_row_summary_parquet"])
    assert results_df.height == 0
    assert row_summary_df.item(0, "ownership_rows_returned") == 0
    assert row_summary_df.item(0, "ownership_nonnull_value_count") == 0


def test_normalize_ownership_universe_batch_response_handles_sparse_late_text_values() -> None:
    items: list[RequestItem] = []
    frame_rows: list[dict[str, Any]] = []
    for idx in range(101):
        candidate_ric = f"RIC{idx}"
        handoff_row = {name: None for name in OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS}
        handoff_row.update(
            {
                "bridge_row_id": f"bridge-{idx}",
                "KYPERMNO": str(idx),
                "candidate_ric": candidate_ric,
                "ownership_lookup_row_id": f"lookup-{idx}",
                "ownership_lookup_role": "UNIVERSE_EFFECTIVE",
                "lookup_input": candidate_ric,
                "lookup_input_source": "effective_collection_ric",
                "request_start_date": "2020-12-31",
                "request_end_date": "2024-12-31",
                "retrieval_eligible": True,
                "vendor_returned_name": None if idx < 100 else "Desktop Metal Inc",
            }
        )
        items.append(
            RequestItem(
                item_id=f"item-{idx}",
                stage="ownership_universe",
                instrument=candidate_ric,
                batch_key="2020-12-31|2024-12-31",
                fields=(
                    "TR.CategoryOwnershipPct.Date",
                    "TR.CategoryOwnershipPct",
                    "TR.InstrStatTypeValue",
                ),
                parameters={"StatType": 7, "SDate": "2020-12-31", "EDate": "2024-12-31"},
                payload={"handoff_row": handoff_row},
            )
        )
        frame_rows.append(
            {
                "Instrument": candidate_ric,
                "Date": date(2024, 12, 31),
                "Category Percent Of Traded Shares": 55.0,
                "Investor Statistics Category Value": "Holdings by Institutions",
            }
        )

    normalized = _normalize_ownership_universe_batch_response(items, pl.DataFrame(frame_rows))
    assert normalized.height == 101
    assert normalized.filter(pl.col("vendor_returned_name") == "Desktop Metal Inc").height == 1
    assert normalized.select(pl.col("returned_value").drop_nulls().unique()).to_series(0).to_list() == [55.0]


def test_doc_ownership_exact_and_fallback_api_pipelines_finalize_without_workbooks(tmp_path: Path) -> None:
    output_dir = tmp_path / "refinitiv_doc_ownership_lm2011"
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_filing_path = tmp_path / "sec_ccm_matched_clean_filtered.parquet"
    authority_decisions_path = tmp_path / "refinitiv_permno_ownership_authority_decisions.parquet"
    authority_exceptions_path = tmp_path / "refinitiv_permno_ownership_authority_exceptions.parquet"

    pl.DataFrame(
        {
            "doc_id": ["doc_exact", "doc_fallback"],
            "filing_date": [date(2024, 4, 15), date(2024, 5, 20)],
            "kypermno": ["100", "200"],
        }
    ).write_parquet(doc_filing_path)
    pl.DataFrame(
        {
            "KYPERMNO": ["100", "200"],
            "authoritative_ric": ["AAA.N", "BBB.N"],
            "authoritative_source_family": ["CONVENTIONAL", "CONVENTIONAL"],
            "authority_decision_status": ["STATIC_CONVENTIONAL", "STATIC_CONVENTIONAL"],
            "requires_review": [False, False],
        }
    ).write_parquet(authority_decisions_path)
    pl.DataFrame(
        {
            "KYPERMNO": pl.Series([], dtype=pl.Utf8),
            "authoritative_ric": pl.Series([], dtype=pl.Utf8),
            "authoritative_source_family": pl.Series([], dtype=pl.Utf8),
            "authority_window_start_date": pl.Series([], dtype=pl.Date),
            "authority_window_end_date": pl.Series([], dtype=pl.Date),
            "authority_exception_status": pl.Series([], dtype=pl.Utf8),
        }
    ).write_parquet(authority_exceptions_path)

    exact_provider = FakeProvider(
        [
            pl.DataFrame(
                {
                    "Instrument": ["AAA.N", "BBB.N"],
                    "Category Percent Of Traded Shares": [55.0, 21.0],
                    "Investor Statistics Category Value": [
                        "Holdings by Institutions",
                        "Holdings by Domestic Investors",
                    ],
                }
            )
        ]
    )
    exact_out = run_refinitiv_lm2011_doc_ownership_exact_api_pipeline(
        doc_filing_artifact_path=doc_filing_path,
        authority_decisions_artifact_path=authority_decisions_path,
        authority_exceptions_artifact_path=authority_exceptions_path,
        output_dir=output_dir,
        provider=exact_provider,
    )

    fallback_provider = FakeProvider(
        [
            pl.DataFrame(
                {
                    "Instrument": ["BBB.N"],
                    "Date": [date(2024, 4, 15)],
                    "Category Percent Of Traded Shares": [65.0],
                    "Investor Statistics Category Value": ["Holdings by Institutions"],
                }
            )
        ]
    )
    fallback_out = run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline(
        output_dir=output_dir,
        provider=fallback_provider,
    )

    assert Path(exact_out["refinitiv_lm2011_doc_ownership_exact_raw_parquet"]).exists()
    assert Path(fallback_out["refinitiv_lm2011_doc_ownership_fallback_raw_parquet"]).exists()

    finalize_out = run_refinitiv_lm2011_doc_ownership_finalize_pipeline(output_dir=output_dir)
    final_df = pl.read_parquet(finalize_out["refinitiv_lm2011_doc_ownership_parquet"])
    rows = {row["doc_id"]: row for row in final_df.to_dicts()}

    assert rows["doc_exact"]["retrieval_status"] == "EXACT_TARGET_HIT"
    assert rows["doc_exact"]["institutional_ownership_pct"] == 55.0
    assert rows["doc_fallback"]["retrieval_status"] == "FALLBACK_WINDOW_HIT"
    assert rows["doc_fallback"]["fallback_used"] is True
    assert rows["doc_fallback"]["institutional_ownership_pct"] == 65.0

    fallback_requests_df = pl.read_parquet(fallback_out["refinitiv_lm2011_doc_ownership_fallback_requests_parquet"])
    assert fallback_requests_df.select("doc_id").to_series(0).to_list() == ["doc_fallback"]
    assert fallback_requests_df.columns == list(DOC_OWNERSHIP_REQUEST_COLUMNS)


def test_normalize_date_value_accepts_iso_datetime_text() -> None:
    assert _normalize_date_value("2024-12-01 00:00:00") == date(2024, 12, 1)


def test_classify_error_splits_unresolved_identifier_batches() -> None:
    exc = LsegRequestError(
        "Unable to resolve all requested identifiers in ['QRHC'].",
        error_kind="unresolved_identifiers",
        unresolved_identifiers=("QRHC",),
    )
    policy = _classify_error(exc, batch_size=2, attempt_no=1, max_attempts=4)
    assert policy["state"] == "retryable_error"
    assert policy["split_batch"] is True
    assert policy["stop_stage"] is False


def test_ledger_requeues_known_fixable_fatal_batches(tmp_path: Path) -> None:
    item = RequestItem(
        item_id="item-1",
        stage="ownership_universe",
        instrument="DM.N^D25",
        batch_key="2020-12-31|2024-12-31",
        fields=("TR.CategoryOwnershipPct.Date", "TR.CategoryOwnershipPct", "TR.InstrStatTypeValue"),
        parameters={"StatType": 7, "SDate": "2020-12-31", "EDate": "2024-12-31"},
        payload={"handoff_row": {"ownership_lookup_row_id": "lookup-1"}},
    )
    ledger = RequestLedger(tmp_path / "ledger.sqlite3")
    batch = batch_items([item], max_batch_size=10, unique_instrument_limit=True)[0]
    ledger.enqueue([item], [batch])
    claimed = ledger.claim_next_batch(stage="ownership_universe")
    assert claimed is not None
    ledger.record_error(
        batch_id=claimed.batch_id,
        next_state="fatal_error",
        error_message='could not append value: "Desktop Metal Inc" of type: str to the builder',
        headers={},
        status_code=None,
        latency_ms=None,
        response_bytes=None,
        next_eligible_at_utc=None,
        exception_class="ComputeError",
    )

    requeued = ledger.requeue_known_fixable_fatal_batches(stage="ownership_universe")
    assert requeued == 1
    reclaimed = ledger.claim_next_batch(stage="ownership_universe")
    assert reclaimed is not None
    assert reclaimed.batch_id == claimed.batch_id


def test_to_polars_frame_falls_back_to_utf8_records_when_from_pandas_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")
    pandas_df = pd.DataFrame(
        {
            "Instrument": ["AAA.N", "BBB.N"],
            "MixedValue": [1, "Desktop Metal Inc"],
        }
    )

    def _raise_compute_error(*args: Any, **kwargs: Any) -> pl.DataFrame:
        raise pl.exceptions.ComputeError("mixed schema")

    monkeypatch.setattr(lseg_provider.pl, "from_pandas", _raise_compute_error)
    converted = lseg_provider._to_polars_frame(pandas_df)
    assert converted.columns == ["Instrument", "MixedValue"]
    assert converted.to_dicts() == [
        {"Instrument": "AAA.N", "MixedValue": "1"},
        {"Instrument": "BBB.N", "MixedValue": "Desktop Metal Inc"},
    ]
