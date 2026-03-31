from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sqlite3
from typing import Any
import warnings

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
from thesis_pkg.pipelines.refinitiv import lseg_lookup_api
from thesis_pkg.pipelines.refinitiv import lseg_ownership_api
from thesis_pkg.pipelines.refinitiv.lseg_batching import RequestItem
from thesis_pkg.pipelines.refinitiv.lseg_ownership_api import (
    _normalize_doc_exact_batch_response,
    _normalize_doc_fallback_batch_response,
    _assemble_doc_raw,
    _normalize_to_month_boundaries,
    _normalize_ownership_universe_batch_response,
)
from thesis_pkg.pipelines.refinitiv import lseg_provider
from thesis_pkg.pipelines.refinitiv.lseg_ledger import LsegResumeCompatibilityError, RequestLedger
from thesis_pkg.pipelines.refinitiv.lseg_lookup_api import _classify_error
from thesis_pkg.pipelines.refinitiv.lseg_stage_audit import StageAuditResult, resolve_stage_output_path
from thesis_pkg.pipelines.refinitiv.lseg_provider import (
    LsegDataProvider,
    LsegDataResponse,
    LsegResponseMetadata,
    classify_lseg_error_message,
)
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


def _rewrite_succeeded_result_paths_to_foreign_root(ledger_path: Path, foreign_root: Path) -> None:
    conn = sqlite3.connect(ledger_path)
    try:
        rows = conn.execute(
            """
            SELECT batch_id, result_file_path
            FROM batches
            WHERE state = 'succeeded' AND result_file_path IS NOT NULL
            """
        ).fetchall()
        for batch_id, result_file_path in rows:
            updated_path = foreign_root / Path(str(result_file_path)).name
            conn.execute(
                "UPDATE batches SET result_file_path = ? WHERE batch_id = ?",
                (str(updated_path), str(batch_id)),
            )
        conn.commit()
    finally:
        conn.close()


def test_resolve_stage_output_path_recovers_windows_ledger_paths(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    staged_file = staging_dir / "batch_123.parquet"
    staged_file.write_text("ok", encoding="utf-8")

    resolved = resolve_stage_output_path(
        r"C:\Users\erik9\Documents\SEC_Data\code\NLP_Thesis\full_data_run\refinitiv_step1\ownership_universe_common_stock\staging\ownership_universe\batch_123.parquet",
        staging_dir,
    )

    assert resolved == staged_file


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


class RoutingProvider:
    def __init__(self, response_by_universe: dict[tuple[str, ...], pl.DataFrame]) -> None:
        self._response_by_universe = {
            tuple(universe): frame for universe, frame in response_by_universe.items()
        }
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
        universe_key = tuple(universe)
        self.calls.append(
            {
                "universe": list(universe),
                "fields": list(fields),
                "parameters": dict(parameters or {}),
            }
        )
        if universe_key not in self._response_by_universe:
            raise AssertionError(f"unexpected universe requested: {universe_key}")
        return LsegDataResponse(
            frame=self._response_by_universe[universe_key],
            metadata=LsegResponseMetadata(
                status_code=200,
                headers={"X-Request-Limit-Remaining": "100"},
                latency_ms=25,
                response_bytes=256,
                fingerprint="fake",
            ),
        )


class SelectiveOwnershipErrorProvider:
    def __init__(
        self,
        *,
        good_instrument: str,
        good_response_date: date,
    ) -> None:
        self.good_instrument = good_instrument
        self.good_response_date = good_response_date
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
        normalized_universe = list(universe)
        self.calls.append(
            {
                "universe": normalized_universe,
                "fields": list(fields),
                "parameters": dict(parameters or {}),
            }
        )
        if len(normalized_universe) > 1 or normalized_universe[0] != self.good_instrument:
            raise LsegRequestError(
                "Unable to collect data for the field 'TR.CATEGORYOWNERSHIPPCT.DATE' and some specific "
                f"identifier(s). Requested universes: {normalized_universe!r}. Requested fields: {list(fields)!r}"
            )
        return LsegDataResponse(
            frame=pl.DataFrame(
                {
                    "Instrument": [self.good_instrument],
                    "Date": [self.good_response_date],
                    "Category Percent Of Traded Shares": [55.0],
                    "Investor Statistics Category Value": ["Holdings by Institutions"],
                }
            ),
            metadata=LsegResponseMetadata(
                status_code=200,
                headers={"X-Request-Limit-Remaining": "100"},
                latency_ms=25,
                response_bytes=256,
                fingerprint="fake",
            ),
        )


def _ownership_handoff_row(
    *,
    lookup_row_id: str,
    candidate_ric: str,
    request_start_date: str,
    request_end_date: str,
    kypermno: str = "1",
) -> dict[str, Any]:
    handoff_row = {name: None for name in OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS}
    handoff_row.update(
        {
            "bridge_row_id": lookup_row_id.replace("|", ":"),
            "KYPERMNO": kypermno,
            "CUSIP": "111111111",
            "ISIN": "US1111111111",
            "TICKER": "ALFA",
            "first_seen_caldt": date.fromisoformat(request_start_date),
            "last_seen_caldt": date.fromisoformat(request_end_date),
            "accepted_ric": candidate_ric,
            "accepted_ric_source": "ISIN",
            "accepted_resolution_status": "resolved_from_isin",
            "conventional_identity_conflict": False,
            "ticker_candidate_available": False,
            "effective_collection_ric": candidate_ric,
            "effective_collection_ric_source": "ISIN",
            "effective_resolution_status": "effective_from_accepted_ric",
            "diagnostic_case_id": lookup_row_id.replace("|", ":"),
            "candidate_slot": "UNIVERSE_EFFECTIVE",
            "candidate_ric": candidate_ric,
            "ownership_lookup_row_id": lookup_row_id,
            "ownership_lookup_role": "UNIVERSE_EFFECTIVE",
            "lookup_input": candidate_ric,
            "lookup_input_source": "effective_collection_ric",
            "request_start_date": request_start_date,
            "request_end_date": request_end_date,
            "retrieval_eligible": True,
        }
    )
    return handoff_row


def _write_doc_ownership_test_inputs(
    tmp_path: Path,
    *,
    doc_rows: list[dict[str, Any]],
    authority_rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    doc_filing_path = tmp_path / "sec_ccm_matched_clean_filtered.parquet"
    authority_decisions_path = tmp_path / "refinitiv_permno_ownership_authority_decisions.parquet"
    authority_exceptions_path = tmp_path / "refinitiv_permno_ownership_authority_exceptions.parquet"

    pl.DataFrame(doc_rows).write_parquet(doc_filing_path)
    pl.DataFrame(authority_rows).write_parquet(authority_decisions_path)
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
    return doc_filing_path, authority_decisions_path, authority_exceptions_path


def test_normalize_to_month_boundaries_extends_mid_month_dates() -> None:
    assert _normalize_to_month_boundaries("1998-04-24", "1998-04-29") == (
        "1998-04-01",
        "1998-04-30",
    )
    assert _normalize_to_month_boundaries("2024-01-01", "2024-01-31") == (
        "2024-01-01",
        "2024-01-31",
    )
    assert _normalize_to_month_boundaries("2024-02-15", "2024-02-20") == (
        "2024-02-01",
        "2024-02-29",
    )
    assert _normalize_to_month_boundaries("2023-02-15", "2023-02-20") == (
        "2023-02-01",
        "2023-02-28",
    )
    assert _normalize_to_month_boundaries("2020-12-15", "2021-01-10") == (
        "2020-12-01",
        "2021-01-31",
    )
    assert _normalize_to_month_boundaries("2022-09-15", "2022-09-30") == (
        "2022-09-01",
        "2022-09-30",
    )


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


def test_lookup_api_pipeline_fetch_only_then_finalize_only_matches_full(tmp_path: Path) -> None:
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

    response_frames = [
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
        ),
    ]

    full_dir = tmp_path / "full"
    full_out = run_refinitiv_step1_lookup_api_pipeline(
        snapshot_parquet_path=snapshot_path,
        output_dir=full_dir,
        provider=FakeProvider(list(response_frames)),
        min_seconds_between_requests=0.0,
    )

    staged_dir = tmp_path / "staged"
    fetch_out = run_refinitiv_step1_lookup_api_pipeline(
        snapshot_parquet_path=snapshot_path,
        output_dir=staged_dir,
        provider=FakeProvider(list(response_frames)),
        min_seconds_between_requests=0.0,
        api_stage_mode="fetch_only",
    )

    assert Path(fetch_out["refinitiv_lookup_fetch_manifest_json"]).exists()
    assert not (staged_dir / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet").exists()
    assert not (staged_dir / "refinitiv_lookup_stage_manifest.json").exists()

    finalized_out = run_refinitiv_step1_lookup_api_pipeline(
        snapshot_parquet_path=snapshot_path,
        output_dir=staged_dir,
        api_stage_mode="finalize_only",
    )

    assert pl.read_parquet(full_out["refinitiv_ric_lookup_handoff_common_stock_extended_parquet"]).equals(
        pl.read_parquet(finalized_out["refinitiv_ric_lookup_handoff_common_stock_extended_parquet"]),
        null_equal=True,
    )
    fetch_manifest = json.loads(Path(finalized_out["refinitiv_lookup_fetch_manifest_json"]).read_text(encoding="utf-8"))
    stage_manifest = json.loads(Path(finalized_out["refinitiv_lookup_stage_manifest_json"]).read_text(encoding="utf-8"))
    assert fetch_manifest["metadata_source"] == "fetch_manifest"
    assert fetch_manifest["cli_batching_args_ignored"] is False
    assert stage_manifest["metadata_source"] == "fetch_manifest"
    assert stage_manifest["cli_batching_args_ignored"] is False


def test_lookup_finalize_only_resolves_foreign_staging_paths(tmp_path: Path) -> None:
    lookup_df = pl.DataFrame(
        {
            "bridge_row_id": ["1:A:111111111:US1111111111:ALFA-1"],
            "KYPERMNO": ["1"],
            "CUSIP": ["111111111"],
            "ISIN": ["US1111111111"],
            "TICKER": ["ALFA"],
            "first_seen_caldt": [date(2024, 1, 1)],
            "last_seen_caldt": [date(2024, 1, 10)],
            "preferred_lookup_id": ["US1111111111"],
            "preferred_lookup_type": ["ISIN"],
            "vendor_primary_ric": [None],
            "vendor_returned_name": [None],
            "vendor_returned_cusip": [None],
            "vendor_returned_isin": [None],
            "vendor_match_status": [None],
            "vendor_notes": [None],
        }
    ).select(RIC_LOOKUP_COLUMNS)
    snapshot_df, _, _ = build_refinitiv_lookup_extended_diagnostic_artifact(lookup_df)
    snapshot_path = tmp_path / "lookup_snapshot.parquet"
    snapshot_df.write_parquet(snapshot_path)

    response_frames = [
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
        ),
    ]
    full_dir = tmp_path / "full"
    full_out = run_refinitiv_step1_lookup_api_pipeline(
        snapshot_parquet_path=snapshot_path,
        output_dir=full_dir,
        provider=FakeProvider(list(response_frames)),
        min_seconds_between_requests=0.0,
    )

    staged_dir = tmp_path / "staged"
    fetch_out = run_refinitiv_step1_lookup_api_pipeline(
        snapshot_parquet_path=snapshot_path,
        output_dir=staged_dir,
        provider=FakeProvider(list(response_frames)),
        min_seconds_between_requests=0.0,
        api_stage_mode="fetch_only",
    )
    _rewrite_succeeded_result_paths_to_foreign_root(
        Path(fetch_out["refinitiv_lookup_api_ledger_sqlite3"]),
        Path("/mnt/original-machine/staging/lookup"),
    )

    finalized_out = run_refinitiv_step1_lookup_api_pipeline(
        snapshot_parquet_path=snapshot_path,
        output_dir=staged_dir,
        api_stage_mode="finalize_only",
    )

    assert pl.read_parquet(full_out["refinitiv_ric_lookup_handoff_common_stock_extended_parquet"]).equals(
        pl.read_parquet(finalized_out["refinitiv_ric_lookup_handoff_common_stock_extended_parquet"]),
        null_equal=True,
    )


def test_lookup_api_pipeline_resume_mismatch_references_fetch_manifest(tmp_path: Path) -> None:
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
    snapshot_path = tmp_path / "snapshot.parquet"
    snapshot_df.write_parquet(snapshot_path)

    run_refinitiv_step1_lookup_api_pipeline(
        snapshot_parquet_path=snapshot_path,
        output_dir=tmp_path,
        provider=FakeProvider(
            [
                pl.DataFrame({"Instrument": ["111111111"], "RIC": ["AAA.N"], "Company Common Name": ["Alpha"], "ISIN": ["US1111111111"], "CUSIP": ["111111111"]}),
                pl.DataFrame({"Instrument": ["ALFA"], "RIC": ["AAA.N"], "Company Common Name": ["Alpha"], "ISIN": ["US1111111111"], "CUSIP": ["111111111"]}),
                pl.DataFrame({"Instrument": ["US1111111111"], "RIC": ["AAA.N"], "Company Common Name": ["Alpha"], "ISIN": ["US1111111111"], "CUSIP": ["111111111"]}),
            ]
        ),
        min_seconds_between_requests=0.0,
        max_batch_size=2,
    )

    with pytest.raises(LsegResumeCompatibilityError) as exc_info:
        run_refinitiv_step1_lookup_api_pipeline(
            snapshot_parquet_path=snapshot_path,
            output_dir=tmp_path,
            provider=FakeProvider([]),
            min_seconds_between_requests=0.0,
            max_batch_size=1,
        )

    message = str(exc_info.value)
    assert "refinitiv_lookup_fetch_manifest.json" in message
    assert "stored fetch manifest" in message


def test_lookup_api_pipeline_uses_lightweight_audit_during_full_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_df = pl.DataFrame(
        {
            "bridge_row_id": ["1:A:111111111:US1111111111:ALFA-1"],
            "KYPERMNO": ["1"],
            "CUSIP": ["111111111"],
            "ISIN": ["US1111111111"],
            "TICKER": ["ALFA"],
            "first_seen_caldt": [date(2024, 1, 1)],
            "last_seen_caldt": [date(2024, 1, 10)],
            "preferred_lookup_id": ["US1111111111"],
            "preferred_lookup_type": ["ISIN"],
            "vendor_primary_ric": [None],
            "vendor_returned_name": [None],
            "vendor_returned_cusip": [None],
            "vendor_returned_isin": [None],
            "vendor_match_status": [None],
            "vendor_notes": [None],
        }
    ).select(RIC_LOOKUP_COLUMNS)
    snapshot_df, _, _ = build_refinitiv_lookup_extended_diagnostic_artifact(lookup_df)
    snapshot_path = tmp_path / "snapshot.parquet"
    snapshot_df.write_parquet(snapshot_path)

    captured: dict[str, object] = {}

    def fake_audit_api_stage(**kwargs: Any) -> StageAuditResult:
        captured.update(kwargs)
        return StageAuditResult(stage_name="lookup", passed=True, issues=(), metrics={})

    monkeypatch.setattr(lseg_lookup_api, "audit_api_stage", fake_audit_api_stage)
    monkeypatch.setattr(lseg_lookup_api, "write_stage_completion_manifest", lambda **kwargs: None)

    run_refinitiv_step1_lookup_api_pipeline(
        snapshot_parquet_path=snapshot_path,
        output_dir=tmp_path,
        provider=FakeProvider(
            [
                pl.DataFrame(
                    {
                        "Instrument": ["US1111111111"],
                        "RIC": ["AAA.N"],
                        "Company Common Name": ["Alpha Inc"],
                        "ISIN": ["US1111111111"],
                        "CUSIP": ["111111111"],
                    }
                ),
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
            ]
        ),
        min_seconds_between_requests=0.0,
    )

    assert captured["verify_rebuilders"] is False


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


def test_ownership_universe_api_pipeline_fetch_only_then_finalize_only_matches_full(tmp_path: Path) -> None:
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
    handoff_path = tmp_path / "handoff.parquet"
    pl.DataFrame([handoff_row]).write_parquet(handoff_path)

    provider_frames = [
        pl.DataFrame(
            {
                "Instrument": ["AAA.N", "AAA.N"],
                "Date": [date(2024, 1, 31), date(2024, 1, 15)],
                "Category Percent Of Traded Shares": [55.0, 52.0],
                "Investor Statistics Category Value": ["Holdings by Institutions", "Holdings by Institutions"],
            }
        )
    ]

    full_dir = tmp_path / "full"
    full_out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=full_dir,
        provider=FakeProvider(list(provider_frames)),
        min_seconds_between_requests=0.0,
    )

    staged_dir = tmp_path / "staged"
    fetch_out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=staged_dir,
        provider=FakeProvider(list(provider_frames)),
        min_seconds_between_requests=0.0,
        api_stage_mode="fetch_only",
    )

    assert Path(fetch_out["refinitiv_ownership_universe_fetch_manifest_json"]).exists()
    assert not (staged_dir / "refinitiv_ownership_universe_results.parquet").exists()
    assert not (staged_dir / "refinitiv_ownership_universe_row_summary.parquet").exists()

    finalized_out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=staged_dir,
        max_batch_size=1,
        max_batch_items=1,
        api_stage_mode="finalize_only",
    )

    assert pl.read_parquet(full_out["refinitiv_ownership_universe_results_parquet"]).equals(
        pl.read_parquet(finalized_out["refinitiv_ownership_universe_results_parquet"]),
        null_equal=True,
    )
    assert pl.read_parquet(full_out["refinitiv_ownership_universe_row_summary_parquet"]).equals(
        pl.read_parquet(finalized_out["refinitiv_ownership_universe_row_summary_parquet"]),
        null_equal=True,
    )
    fetch_manifest = json.loads(Path(finalized_out["refinitiv_ownership_universe_fetch_manifest_json"]).read_text(encoding="utf-8"))
    stage_manifest = json.loads(Path(finalized_out["refinitiv_ownership_universe_stage_manifest_json"]).read_text(encoding="utf-8"))
    assert fetch_manifest["metadata_source"] == "fetch_manifest"
    assert fetch_manifest["cli_batching_args_ignored"] is True
    assert fetch_manifest["batching_config"] == {
        "max_batch_size": 10,
        "max_batch_items": 10,
        "max_extra_rows_abs": 120.0,
        "max_extra_rows_ratio": 0.25,
        "max_union_span_days": None,
        "row_density_rows_per_day": pytest.approx(1.0 / 91.0),
    }
    assert stage_manifest["metadata_source"] == "fetch_manifest"
    assert stage_manifest["cli_batching_args_ignored"] is True
    assert stage_manifest["summary"]["batch_plan_fingerprint"] == fetch_manifest["summary"]["batch_plan_fingerprint"]


def test_ownership_finalize_only_resolves_foreign_staging_paths(tmp_path: Path) -> None:
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
    handoff_path = tmp_path / "handoff.parquet"
    pl.DataFrame([handoff_row]).write_parquet(handoff_path)

    staged_dir = tmp_path / "staged"
    fetch_out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=staged_dir,
        provider=FakeProvider(
            [
                pl.DataFrame(
                    {
                        "Instrument": ["AAA.N", "AAA.N"],
                        "Date": [date(2024, 1, 31), date(2024, 1, 15)],
                        "Category Percent Of Traded Shares": [55.0, 52.0],
                        "Investor Statistics Category Value": [
                            "Holdings by Institutions",
                            "Holdings by Institutions",
                        ],
                    }
                )
            ]
        ),
        min_seconds_between_requests=0.0,
        api_stage_mode="fetch_only",
    )
    _rewrite_succeeded_result_paths_to_foreign_root(
        Path(fetch_out["refinitiv_ownership_universe_api_ledger_sqlite3"]),
        Path("/mnt/original-machine/staging/ownership_universe"),
    )

    finalized_out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=staged_dir,
        api_stage_mode="finalize_only",
    )

    results_df = pl.read_parquet(finalized_out["refinitiv_ownership_universe_results_parquet"]).sort("returned_date")
    row_summary_df = pl.read_parquet(finalized_out["refinitiv_ownership_universe_row_summary_parquet"])
    assert results_df.height == 2
    assert row_summary_df.item(0, "ownership_rows_returned") == 2


def test_ownership_finalize_only_legacy_ledger_meta_writes_best_effort_provenance(tmp_path: Path) -> None:
    handoff_row = _ownership_handoff_row(
        lookup_row_id="1:A:111111111:US1111111111:ALFA|UNIVERSE_EFFECTIVE",
        candidate_ric="AAA.N",
        request_start_date="2024-01-01",
        request_end_date="2024-01-31",
    )
    handoff_path = tmp_path / "handoff.parquet"
    pl.DataFrame([handoff_row]).write_parquet(handoff_path)

    fetch_out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=FakeProvider(
            [
                pl.DataFrame(
                    {
                        "Instrument": ["AAA.N"],
                        "Date": [date(2024, 1, 31)],
                        "Category Percent Of Traded Shares": [55.0],
                        "Investor Statistics Category Value": ["Holdings by Institutions"],
                    }
                )
            ]
        ),
        min_seconds_between_requests=0.0,
        api_stage_mode="fetch_only",
    )

    fetch_manifest_path = Path(fetch_out["refinitiv_ownership_universe_fetch_manifest_json"])
    fetch_manifest_path.unlink()
    ledger_path = Path(fetch_out["refinitiv_ownership_universe_api_ledger_sqlite3"])
    with sqlite3.connect(ledger_path) as conn:
        conn.execute(
            "DELETE FROM ledger_meta WHERE key = ?",
            ("stage:ownership_universe:batch_plan_planner_version",),
        )
        conn.execute(
            "DELETE FROM ledger_meta WHERE key = ?",
            ("stage:ownership_universe:batching_config_json",),
        )
        conn.execute(
            "DELETE FROM ledger_meta WHERE key = ?",
            ("stage:ownership_universe:planned_batch_count",),
        )

    finalized_out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        max_batch_size=1,
        max_batch_items=1,
        api_stage_mode="finalize_only",
    )

    regenerated_fetch_manifest = json.loads(Path(finalized_out["refinitiv_ownership_universe_fetch_manifest_json"]).read_text(encoding="utf-8"))
    stage_manifest = json.loads(Path(finalized_out["refinitiv_ownership_universe_stage_manifest_json"]).read_text(encoding="utf-8"))
    assert regenerated_fetch_manifest["metadata_source"] == "ledger_meta"
    assert regenerated_fetch_manifest["cli_batching_args_ignored"] is True
    assert regenerated_fetch_manifest["batching_config"] is None
    assert regenerated_fetch_manifest["summary"]["batch_plan_fingerprint"]
    assert stage_manifest["metadata_source"] == "ledger_meta"
    assert stage_manifest["summary"]["batching_config"] is None
    assert stage_manifest["summary"]["batch_plan_fingerprint"] == regenerated_fetch_manifest["summary"]["batch_plan_fingerprint"]


def test_lookup_finalize_only_without_stored_metadata_marks_unknown(tmp_path: Path) -> None:
    lookup_df = pl.DataFrame(
        {
            "bridge_row_id": ["1:A:111111111:US1111111111:ALFA-1"],
            "KYPERMNO": ["1"],
            "CUSIP": ["111111111"],
            "ISIN": ["US1111111111"],
            "TICKER": ["ALFA"],
            "first_seen_caldt": [date(2024, 1, 1)],
            "last_seen_caldt": [date(2024, 1, 10)],
            "preferred_lookup_id": ["US1111111111"],
            "preferred_lookup_type": ["ISIN"],
            "vendor_primary_ric": [None],
            "vendor_returned_name": [None],
            "vendor_returned_cusip": [None],
            "vendor_returned_isin": [None],
            "vendor_match_status": [None],
            "vendor_notes": [None],
        }
    ).select(RIC_LOOKUP_COLUMNS)
    snapshot_df, _, _ = build_refinitiv_lookup_extended_diagnostic_artifact(lookup_df)
    snapshot_path = tmp_path / "snapshot.parquet"
    snapshot_df.write_parquet(snapshot_path)

    fetch_out = run_refinitiv_step1_lookup_api_pipeline(
        snapshot_parquet_path=snapshot_path,
        output_dir=tmp_path,
        provider=FakeProvider(
            [
                pl.DataFrame({"Instrument": ["111111111"], "RIC": ["AAA.N"], "Company Common Name": ["Alpha"], "ISIN": ["US1111111111"], "CUSIP": ["111111111"]}),
                pl.DataFrame({"Instrument": ["ALFA"], "RIC": ["AAA.N"], "Company Common Name": ["Alpha"], "ISIN": ["US1111111111"], "CUSIP": ["111111111"]}),
                pl.DataFrame({"Instrument": ["US1111111111"], "RIC": ["AAA.N"], "Company Common Name": ["Alpha"], "ISIN": ["US1111111111"], "CUSIP": ["111111111"]}),
            ]
        ),
        min_seconds_between_requests=0.0,
        api_stage_mode="fetch_only",
    )

    Path(fetch_out["refinitiv_lookup_fetch_manifest_json"]).unlink()
    with sqlite3.connect(Path(fetch_out["refinitiv_lookup_api_ledger_sqlite3"])) as conn:
        conn.execute("DELETE FROM ledger_meta WHERE key LIKE 'stage:lookup:%'")

    finalized_out = run_refinitiv_step1_lookup_api_pipeline(
        snapshot_parquet_path=snapshot_path,
        output_dir=tmp_path,
        max_batch_size=1,
        api_stage_mode="finalize_only",
    )

    fetch_manifest = json.loads(Path(finalized_out["refinitiv_lookup_fetch_manifest_json"]).read_text(encoding="utf-8"))
    stage_manifest = json.loads(Path(finalized_out["refinitiv_lookup_stage_manifest_json"]).read_text(encoding="utf-8"))
    assert fetch_manifest["metadata_source"] == "unknown"
    assert fetch_manifest["batching_config"] is None
    assert fetch_manifest["summary"]["batch_plan_fingerprint"] is None
    assert stage_manifest["metadata_source"] == "unknown"
    assert stage_manifest["summary"]["batching_config"] is None


def test_ownership_universe_api_pipeline_treats_unresolved_identifier_as_empty_result_after_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(lseg_ownership_api, "_retry_delay_seconds", lambda attempt_no: 0.0)
    provider = ErrorProvider(
        [
            LsegRequestError(
                "Unable to resolve all requested identifiers in ['QRHC'].",
                error_kind="unresolved_identifiers",
                unresolved_identifiers=("QRHC",),
            ),
            LsegRequestError(
                "Unable to resolve all requested identifiers in ['QRHC'].",
                error_kind="unresolved_identifiers",
                unresolved_identifiers=("QRHC",),
            ),
        ]
    )
    out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
        max_attempts=2,
    )

    results_df = pl.read_parquet(out["refinitiv_ownership_universe_results_parquet"])
    row_summary_df = pl.read_parquet(out["refinitiv_ownership_universe_row_summary_parquet"])
    assert results_df.height == 0
    assert row_summary_df.height == 1
    assert row_summary_df.item(0, "ownership_rows_returned") == 0
    assert row_summary_df.item(0, "ownership_single_returned_ric") is False
    assert len(provider.calls) == 2

    request_log = [json.loads(line) for line in Path(out["refinitiv_ownership_universe_api_requests_jsonl"]).read_text().splitlines()]
    assert request_log[-2]["event"] == "request_failed"
    assert request_log[-2]["policy"] == "retryable_error"
    assert request_log[-2]["unresolved_identifiers"] == ["QRHC"]
    assert request_log[-1]["event"] == "request_unresolved_identifiers_treated_as_empty"
    assert request_log[-1]["unresolved_identifiers"] == ["QRHC"]


def test_ownership_universe_api_pipeline_splits_field_specific_identifier_failures(tmp_path: Path) -> None:
    handoff_path = tmp_path / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    pl.DataFrame(
        [
            _ownership_handoff_row(
                lookup_row_id="row-good|UNIVERSE_EFFECTIVE",
                candidate_ric="AAA.N",
                request_start_date="2024-01-01",
                request_end_date="2024-01-31",
                kypermno="1",
            ),
            _ownership_handoff_row(
                lookup_row_id="row-bad|UNIVERSE_EFFECTIVE",
                candidate_ric="BAD.N",
                request_start_date="2024-01-01",
                request_end_date="2024-01-31",
                kypermno="2",
            ),
        ]
    ).write_parquet(handoff_path)

    provider = SelectiveOwnershipErrorProvider(
        good_instrument="AAA.N",
        good_response_date=date(2024, 1, 15),
    )
    out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
        max_attempts=2,
    )

    results_df = pl.read_parquet(out["refinitiv_ownership_universe_results_parquet"]).sort("ownership_lookup_row_id")
    row_summary_df = pl.read_parquet(out["refinitiv_ownership_universe_row_summary_parquet"]).sort(
        "ownership_lookup_row_id"
    )
    request_log = [
        json.loads(line)
        for line in Path(out["refinitiv_ownership_universe_api_requests_jsonl"]).read_text().splitlines()
    ]

    assert len(provider.calls) == 4
    assert sorted(provider.calls[0]["universe"]) == ["AAA.N", "BAD.N"]
    assert results_df.height == 1
    assert results_df.item(0, "ownership_lookup_row_id") == "row-good|UNIVERSE_EFFECTIVE"
    assert row_summary_df.get_column("ownership_rows_returned").to_list() == [0, 1]
    assert any(entry["event"] == "request_batch_split" for entry in request_log)
    assert any(
        entry["event"] == "request_unresolved_identifiers_treated_as_empty"
        and entry["unresolved_identifiers"] == ["BAD.N"]
        for entry in request_log
    )


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


def test_ownership_universe_api_pipeline_batches_close_windows(tmp_path: Path) -> None:
    handoff_path = tmp_path / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    pl.DataFrame(
        [
            _ownership_handoff_row(
                lookup_row_id="row-1|UNIVERSE_EFFECTIVE",
                candidate_ric="AAA.N",
                request_start_date="2024-01-01",
                request_end_date="2024-03-31",
                kypermno="1",
            ),
            _ownership_handoff_row(
                lookup_row_id="row-2|UNIVERSE_EFFECTIVE",
                candidate_ric="BBB.N",
                request_start_date="2024-01-15",
                request_end_date="2024-03-31",
                kypermno="2",
            ),
        ]
    ).write_parquet(handoff_path)

    provider = FakeProvider(
        [
            pl.DataFrame(
                {
                    "Instrument": ["AAA.N", "BBB.N"],
                    "Date": [date(2024, 3, 31), date(2024, 3, 31)],
                    "Category Percent Of Traded Shares": [55.0, 45.0],
                    "Investor Statistics Category Value": [
                        "Holdings by Institutions",
                        "Holdings by Institutions",
                    ],
                }
            )
        ]
    )
    out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
    )

    results_df = pl.read_parquet(out["refinitiv_ownership_universe_results_parquet"])
    row_summary_df = pl.read_parquet(out["refinitiv_ownership_universe_row_summary_parquet"]).sort(
        "ownership_lookup_row_id"
    )
    assert len(provider.calls) == 1
    assert provider.calls[0]["parameters"] == {
        "StatType": 7,
        "SDate": "2024-01-01",
        "EDate": "2024-03-31",
    }
    assert sorted(provider.calls[0]["universe"]) == ["AAA.N", "BBB.N"]
    assert results_df.height == 2
    assert row_summary_df.get_column("ownership_rows_returned").to_list() == [1, 1]


def test_ownership_universe_api_pipeline_filters_widened_batch_rows_per_item_window(tmp_path: Path) -> None:
    handoff_path = tmp_path / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    pl.DataFrame(
        [
            _ownership_handoff_row(
                lookup_row_id="row-jan|UNIVERSE_EFFECTIVE",
                candidate_ric="AAA.N",
                request_start_date="2024-01-01",
                request_end_date="2024-01-31",
                kypermno="1",
            ),
            _ownership_handoff_row(
                lookup_row_id="row-feb|UNIVERSE_EFFECTIVE",
                candidate_ric="AAA.N",
                request_start_date="2024-02-01",
                request_end_date="2024-02-29",
                kypermno="1",
            ),
        ]
    ).write_parquet(handoff_path)

    provider = FakeProvider(
        [
            pl.DataFrame(
                {
                    "Instrument": ["AAA.N", "AAA.N", "AAA.N"],
                    "Date": [date(2024, 1, 15), date(2024, 2, 10), None],
                    "Category Percent Of Traded Shares": [51.0, 52.0, 53.0],
                    "Investor Statistics Category Value": [
                        "Holdings by Institutions",
                        "Holdings by Institutions",
                        "Holdings by Institutions",
                    ],
                }
            )
        ]
    )
    out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
    )

    results_df = pl.read_parquet(out["refinitiv_ownership_universe_results_parquet"]).sort(
        ["ownership_lookup_row_id", "returned_date"]
    )
    row_summary_df = pl.read_parquet(out["refinitiv_ownership_universe_row_summary_parquet"]).sort(
        "ownership_lookup_row_id"
    )
    assert len(provider.calls) == 1
    assert results_df.height == 2
    assert results_df.get_column("ownership_lookup_row_id").to_list() == [
        "row-feb|UNIVERSE_EFFECTIVE",
        "row-jan|UNIVERSE_EFFECTIVE",
    ]
    assert results_df.get_column("returned_date").to_list() == [date(2024, 2, 10), date(2024, 1, 15)]
    assert row_summary_df.get_column("ownership_rows_returned").to_list() == [1, 1]


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


def test_ownership_normalizers_match_request_instruments_after_whitespace_normalization() -> None:
    ownership_item = RequestItem(
        item_id="own-1",
        stage="ownership_universe",
        instrument=" TEST.N ",
        batch_key="2024-01-01|2024-01-31",
        fields=("TR.CategoryOwnershipPct.Date", "TR.CategoryOwnershipPct", "TR.InstrStatTypeValue"),
        parameters={"StatType": 7, "SDate": "2024-01-01", "EDate": "2024-01-31"},
        payload={
            "handoff_row": _ownership_handoff_row(
                lookup_row_id="own-1|UNIVERSE_EFFECTIVE",
                candidate_ric="TEST.N",
                request_start_date="2024-01-01",
                request_end_date="2024-01-31",
            )
        },
    )
    doc_request_row = {name: None for name in DOC_OWNERSHIP_REQUEST_COLUMNS}
    doc_request_row.update(
        {
            "doc_id": "0000000001:000000000100000001",
            "authoritative_ric": "TEST.N",
            "request_start_date": date(2024, 1, 1),
            "request_end_date": date(2024, 1, 31),
            "retrieval_eligible": True,
        }
    )
    doc_exact_item = RequestItem(
        item_id="doc-exact-1",
        stage="doc_exact",
        instrument=" TEST.N ",
        batch_key="2024-01-01|2024-01-31",
        fields=("TR.CategoryOwnershipPct.Date", "TR.CategoryOwnershipPct", "TR.InstrStatTypeValue"),
        parameters={"StatType": 7, "SDate": "2024-01-01", "EDate": "2024-01-31"},
        payload={"request_row": dict(doc_request_row)},
    )
    doc_fallback_item = RequestItem(
        item_id="doc-fallback-1",
        stage="doc_fallback",
        instrument=" TEST.N ",
        batch_key="2024-01-01|2024-01-31",
        fields=("TR.CategoryOwnershipPct.Date", "TR.CategoryOwnershipPct", "TR.InstrStatTypeValue"),
        parameters={"StatType": 7, "SDate": "2024-01-01", "EDate": "2024-01-31"},
        payload={"request_row": dict(doc_request_row)},
    )
    response_frame = pl.DataFrame(
        {
            "Instrument": ["TEST.N"],
            "Date": [date(2024, 1, 31)],
            "Category Percent Of Traded Shares": [55.0],
            "Investor Statistics Category Value": ["Holdings by Institutions"],
        }
    )

    ownership_df = _normalize_ownership_universe_batch_response([ownership_item], response_frame)
    exact_df = _normalize_doc_exact_batch_response([doc_exact_item], response_frame)
    fallback_df = _normalize_doc_fallback_batch_response([doc_fallback_item], response_frame)

    assert ownership_df.height == 1
    assert ownership_df.item(0, "returned_ric") == "TEST.N"
    assert exact_df.height == 1
    assert exact_df.item(0, "response_date") == date(2024, 1, 31)
    assert fallback_df.height == 1
    assert fallback_df.item(0, "response_date") == date(2024, 1, 31)


def test_ownership_universe_api_pipeline_includes_month_start_rows_in_mid_month_window(tmp_path: Path) -> None:
    handoff_path = tmp_path / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    pl.DataFrame(
        [
            _ownership_handoff_row(
                lookup_row_id="hei-row|UNIVERSE_EFFECTIVE",
                candidate_ric="HEI",
                request_start_date="1998-04-24",
                request_end_date="1998-04-29",
                kypermno="10001",
            ),
        ]
    ).write_parquet(handoff_path)

    provider = FakeProvider(
        [
            pl.DataFrame(
                {
                    "Instrument": ["HEI", "HEI"],
                    "Date": [date(1998, 4, 1), date(1998, 4, 1)],
                    "Category Percent Of Traded Shares": [45.0, 12.0],
                    "Investor Statistics Category Value": [
                        "Holdings by Institutions",
                        "Holdings by Insiders",
                    ],
                }
            )
        ]
    )
    out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
    )

    results_df = pl.read_parquet(out["refinitiv_ownership_universe_results_parquet"]).sort(
        ["ownership_lookup_row_id", "returned_category"]
    )
    assert results_df.height == 2
    assert results_df.get_column("returned_date").to_list() == [
        date(1998, 4, 1),
        date(1998, 4, 1),
    ]
    assert len(provider.calls) == 1
    assert provider.calls[0]["parameters"] == {
        "StatType": 7,
        "SDate": "1998-04-01",
        "EDate": "1998-04-30",
    }


def test_ownership_universe_api_pipeline_normalizes_different_mid_month_windows_in_same_month(
    tmp_path: Path,
) -> None:
    handoff_path = tmp_path / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    pl.DataFrame(
        [
            _ownership_handoff_row(
                lookup_row_id="row-a|UNIVERSE_EFFECTIVE",
                candidate_ric="TEST.N",
                request_start_date="2022-09-05",
                request_end_date="2022-09-15",
                kypermno="1",
            ),
            _ownership_handoff_row(
                lookup_row_id="row-b|UNIVERSE_EFFECTIVE",
                candidate_ric="TEST.N",
                request_start_date="2022-09-16",
                request_end_date="2022-09-29",
                kypermno="1",
            ),
        ]
    ).write_parquet(handoff_path)

    provider = FakeProvider(
        [
            pl.DataFrame(
                {
                    "Instrument": ["TEST.N"],
                    "Date": [date(2022, 9, 1)],
                    "Category Percent Of Traded Shares": [60.0],
                    "Investor Statistics Category Value": ["Holdings by Institutions"],
                }
            )
        ]
    )
    out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
    )

    results_df = pl.read_parquet(out["refinitiv_ownership_universe_results_parquet"]).sort(
        "ownership_lookup_row_id"
    )
    assert results_df.height == 2
    assert results_df.get_column("ownership_lookup_row_id").to_list() == [
        "row-a|UNIVERSE_EFFECTIVE",
        "row-b|UNIVERSE_EFFECTIVE",
    ]
    assert len(provider.calls) == 1
    assert provider.calls[0]["parameters"] == {
        "StatType": 7,
        "SDate": "2022-09-01",
        "EDate": "2022-09-30",
    }


def test_ownership_universe_api_pipeline_requeues_mixed_zero_positive_success_batches(tmp_path: Path) -> None:
    handoff_path = tmp_path / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    pl.DataFrame(
        [
            _ownership_handoff_row(
                lookup_row_id="row-a|UNIVERSE_EFFECTIVE",
                candidate_ric="AAA.N",
                request_start_date="2024-01-01",
                request_end_date="2024-01-31",
                kypermno="1",
            ),
            _ownership_handoff_row(
                lookup_row_id="row-b|UNIVERSE_EFFECTIVE",
                candidate_ric="BBB.N",
                request_start_date="2024-01-01",
                request_end_date="2024-01-31",
                kypermno="2",
            ),
        ]
    ).write_parquet(handoff_path)
    provider = RoutingProvider(
        {
            ("AAA.N", "BBB.N"): pl.DataFrame(
                {
                    "Instrument": ["AAA.N"],
                    "Date": [date(2024, 1, 31)],
                    "Category Percent Of Traded Shares": [50.0],
                    "Investor Statistics Category Value": ["Holdings by Institutions"],
                }
            ),
            ("AAA.N",): pl.DataFrame(
                {
                    "Instrument": ["AAA.N"],
                    "Date": [date(2024, 1, 31)],
                    "Category Percent Of Traded Shares": [50.0],
                    "Investor Statistics Category Value": ["Holdings by Institutions"],
                }
            ),
            ("BBB.N",): pl.DataFrame(
                {
                    "Instrument": ["BBB.N"],
                    "Date": [date(2024, 1, 31)],
                    "Category Percent Of Traded Shares": [40.0],
                    "Investor Statistics Category Value": ["Holdings by Institutions"],
                }
            ),
        }
    )

    out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
        max_batch_size=2,
    )

    results_df = pl.read_parquet(out["refinitiv_ownership_universe_results_parquet"]).sort("ownership_lookup_row_id")
    request_log = [
        json.loads(line)
        for line in Path(out["refinitiv_ownership_universe_api_requests_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mixed_event = next(
        payload
        for payload in request_log
        if payload.get("event") == "request_succeeded_mixed_zero_items_requeued"
    )

    assert provider.calls[0]["universe"] == ["AAA.N", "BBB.N"]
    assert sorted(call["universe"][0] for call in provider.calls[1:]) == ["AAA.N", "BBB.N"]
    assert results_df.get_column("ownership_lookup_row_id").to_list() == [
        "row-a|UNIVERSE_EFFECTIVE",
        "row-b|UNIVERSE_EFFECTIVE",
    ]
    assert mixed_event["mixed_zero_positive_success"] is True
    assert sorted(
        (row["instrument"], row["row_count"]) for row in mixed_event["item_results"]
    ) == [("AAA.N", 1), ("BBB.N", 0)]
    parent_stage_path = tmp_path / "staging" / "ownership_universe" / f"{mixed_event['batch_id']}.parquet"
    assert not parent_stage_path.exists()
    manifest = json.loads(Path(out["refinitiv_ownership_universe_stage_manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["summary"]["mixed_zero_positive_success_requeued_count"] == 1


def test_ownership_universe_api_pipeline_keeps_all_zero_multi_item_success_batches(tmp_path: Path) -> None:
    handoff_path = tmp_path / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    pl.DataFrame(
        [
            _ownership_handoff_row(
                lookup_row_id="row-a|UNIVERSE_EFFECTIVE",
                candidate_ric="AAA.N",
                request_start_date="2024-01-01",
                request_end_date="2024-01-31",
                kypermno="1",
            ),
            _ownership_handoff_row(
                lookup_row_id="row-b|UNIVERSE_EFFECTIVE",
                candidate_ric="BBB.N",
                request_start_date="2024-01-01",
                request_end_date="2024-01-31",
                kypermno="2",
            ),
        ]
    ).write_parquet(handoff_path)
    provider = FakeProvider([pl.DataFrame()])

    out = run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
        max_batch_size=2,
    )

    request_log = Path(out["refinitiv_ownership_universe_api_requests_jsonl"]).read_text(encoding="utf-8")
    row_summary_df = pl.read_parquet(out["refinitiv_ownership_universe_row_summary_parquet"]).sort(
        "ownership_lookup_row_id"
    )

    assert len(provider.calls) == 1
    assert "request_succeeded_mixed_zero_items_requeued" not in request_log
    assert row_summary_df.get_column("ownership_rows_returned").to_list() == [0, 0]


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
                    "Date": [date(2024, 4, 1), date(2024, 4, 1)],
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
    assert rows["doc_exact"]["target_effective_date"] == date(2024, 4, 1)
    assert rows["doc_exact"]["selected_response_date"] == date(2024, 4, 1)
    assert rows["doc_fallback"]["retrieval_status"] == "FALLBACK_WINDOW_HIT"
    assert rows["doc_fallback"]["fallback_used"] is True
    assert rows["doc_fallback"]["institutional_ownership_pct"] == 65.0
    assert rows["doc_fallback"]["target_effective_date"] == date(2024, 4, 1)
    assert rows["doc_fallback"]["selected_response_date"] == date(2024, 4, 15)

    fallback_requests_df = pl.read_parquet(fallback_out["refinitiv_lm2011_doc_ownership_fallback_requests_parquet"])
    assert fallback_requests_df.select("doc_id").to_series(0).to_list() == ["doc_fallback"]
    assert fallback_requests_df.columns == list(DOC_OWNERSHIP_REQUEST_COLUMNS)
    assert exact_provider.calls[0]["fields"] == [
        "TR.CategoryOwnershipPct.Date",
        "TR.CategoryOwnershipPct",
        "TR.InstrStatTypeValue",
    ]
    assert exact_provider.calls[0]["parameters"] == {"StatType": 7, "SDate": "2024-04-01", "EDate": "2024-04-01"}
    assert fallback_provider.calls[0]["parameters"] == {"StatType": 7, "SDate": "2024-04-01", "EDate": "2024-05-16"}


def test_doc_ownership_exact_api_pipeline_fetch_only_then_finalize_only_matches_full(tmp_path: Path) -> None:
    doc_filing_path, authority_decisions_path, authority_exceptions_path = _write_doc_ownership_test_inputs(
        tmp_path,
        doc_rows=[{"doc_id": "doc-1", "filing_date": date(2024, 4, 15), "kypermno": "100"}],
        authority_rows=[
            {
                "KYPERMNO": "100",
                "authoritative_ric": "AAA.N",
                "authoritative_source_family": "CONVENTIONAL",
                "authority_decision_status": "STATIC_CONVENTIONAL",
                "requires_review": False,
            }
        ],
    )

    response_frames = [
        pl.DataFrame(
            {
                "Instrument": ["AAA.N"],
                "Date": [date(2024, 4, 1)],
                "Category Percent Of Traded Shares": [55.0],
                "Investor Statistics Category Value": ["Holdings by Institutions"],
            }
        )
    ]

    full_dir = tmp_path / "full_doc"
    full_out = run_refinitiv_lm2011_doc_ownership_exact_api_pipeline(
        doc_filing_artifact_path=doc_filing_path,
        authority_decisions_artifact_path=authority_decisions_path,
        authority_exceptions_artifact_path=authority_exceptions_path,
        output_dir=full_dir,
        provider=FakeProvider(list(response_frames)),
        min_seconds_between_requests=0.0,
    )

    staged_dir = tmp_path / "staged_doc"
    fetch_out = run_refinitiv_lm2011_doc_ownership_exact_api_pipeline(
        doc_filing_artifact_path=doc_filing_path,
        authority_decisions_artifact_path=authority_decisions_path,
        authority_exceptions_artifact_path=authority_exceptions_path,
        output_dir=staged_dir,
        provider=FakeProvider(list(response_frames)),
        min_seconds_between_requests=0.0,
        api_stage_mode="fetch_only",
    )

    assert Path(fetch_out["refinitiv_doc_ownership_exact_fetch_manifest_json"]).exists()
    assert not (staged_dir / "refinitiv_lm2011_doc_ownership_exact_requests.parquet").exists()
    assert not (staged_dir / "refinitiv_lm2011_doc_ownership_exact_raw.parquet").exists()

    finalized_out = run_refinitiv_lm2011_doc_ownership_exact_api_pipeline(
        doc_filing_artifact_path=doc_filing_path,
        authority_decisions_artifact_path=authority_decisions_path,
        authority_exceptions_artifact_path=authority_exceptions_path,
        output_dir=staged_dir,
        api_stage_mode="finalize_only",
    )

    assert pl.read_parquet(full_out["refinitiv_lm2011_doc_ownership_exact_requests_parquet"]).equals(
        pl.read_parquet(finalized_out["refinitiv_lm2011_doc_ownership_exact_requests_parquet"]),
        null_equal=True,
    )
    assert pl.read_parquet(full_out["refinitiv_lm2011_doc_ownership_exact_raw_parquet"]).equals(
        pl.read_parquet(finalized_out["refinitiv_lm2011_doc_ownership_exact_raw_parquet"]),
        null_equal=True,
    )


def test_doc_ownership_exact_pipeline_splits_field_specific_identifier_failures(tmp_path: Path) -> None:
    output_dir = tmp_path / "refinitiv_doc_ownership_lm2011"
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filing_path, authority_decisions_path, authority_exceptions_path = _write_doc_ownership_test_inputs(
        tmp_path,
        doc_rows=[
            {"doc_id": "doc-good", "filing_date": date(2024, 4, 15), "kypermno": "100"},
            {"doc_id": "doc-bad", "filing_date": date(2024, 4, 15), "kypermno": "200"},
        ],
        authority_rows=[
            {
                "KYPERMNO": "100",
                "authoritative_ric": "AAA.N",
                "authoritative_source_family": "CONVENTIONAL",
                "authority_decision_status": "STATIC_CONVENTIONAL",
                "requires_review": False,
            },
            {
                "KYPERMNO": "200",
                "authoritative_ric": "BAD.N",
                "authoritative_source_family": "CONVENTIONAL",
                "authority_decision_status": "STATIC_CONVENTIONAL",
                "requires_review": False,
            },
        ],
    )

    provider = SelectiveOwnershipErrorProvider(
        good_instrument="AAA.N",
        good_response_date=date(2024, 4, 1),
    )
    out = run_refinitiv_lm2011_doc_ownership_exact_api_pipeline(
        doc_filing_artifact_path=doc_filing_path,
        authority_decisions_artifact_path=authority_decisions_path,
        authority_exceptions_artifact_path=authority_exceptions_path,
        output_dir=output_dir,
        provider=provider,
        min_seconds_between_requests=0.0,
        max_attempts=2,
    )

    exact_raw_df = pl.read_parquet(out["refinitiv_lm2011_doc_ownership_exact_raw_parquet"]).sort("doc_id")
    request_log = [
        json.loads(line)
        for line in Path(out["refinitiv_doc_ownership_exact_api_requests_jsonl"]).read_text().splitlines()
    ]

    assert len(provider.calls) == 4
    assert sorted(provider.calls[0]["universe"]) == ["AAA.N", "BAD.N"]
    assert exact_raw_df.height == 1
    assert exact_raw_df.item(0, "doc_id") == "doc-good"
    assert any(entry["event"] == "request_batch_split" for entry in request_log)
    assert any(
        entry["event"] == "request_unresolved_identifiers_treated_as_empty"
        and entry["unresolved_identifiers"] == ["BAD.N"]
        for entry in request_log
    )


def test_doc_ownership_fallback_pipeline_splits_field_specific_identifier_failures(tmp_path: Path) -> None:
    output_dir = tmp_path / "refinitiv_doc_ownership_lm2011"
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filing_path, authority_decisions_path, authority_exceptions_path = _write_doc_ownership_test_inputs(
        tmp_path,
        doc_rows=[
            {"doc_id": "doc-good", "filing_date": date(2024, 4, 15), "kypermno": "100"},
            {"doc_id": "doc-bad", "filing_date": date(2024, 4, 15), "kypermno": "200"},
        ],
        authority_rows=[
            {
                "KYPERMNO": "100",
                "authoritative_ric": "AAA.N",
                "authoritative_source_family": "CONVENTIONAL",
                "authority_decision_status": "STATIC_CONVENTIONAL",
                "requires_review": False,
            },
            {
                "KYPERMNO": "200",
                "authoritative_ric": "BAD.N",
                "authoritative_source_family": "CONVENTIONAL",
                "authority_decision_status": "STATIC_CONVENTIONAL",
                "requires_review": False,
            },
        ],
    )

    run_refinitiv_lm2011_doc_ownership_exact_api_pipeline(
        doc_filing_artifact_path=doc_filing_path,
        authority_decisions_artifact_path=authority_decisions_path,
        authority_exceptions_artifact_path=authority_exceptions_path,
        output_dir=output_dir,
        provider=FakeProvider([pl.DataFrame(schema={"Instrument": pl.Utf8, "Date": pl.Date})]),
        min_seconds_between_requests=0.0,
    )

    provider = SelectiveOwnershipErrorProvider(
        good_instrument="AAA.N",
        good_response_date=date(2024, 4, 15),
    )
    out = run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline(
        output_dir=output_dir,
        provider=provider,
        min_seconds_between_requests=0.0,
        max_attempts=2,
    )

    fallback_raw_df = pl.read_parquet(out["refinitiv_lm2011_doc_ownership_fallback_raw_parquet"]).sort("doc_id")
    request_log = [
        json.loads(line)
        for line in Path(out["refinitiv_doc_ownership_fallback_api_requests_jsonl"]).read_text().splitlines()
    ]

    assert len(provider.calls) == 4
    assert sorted(provider.calls[0]["universe"]) == ["AAA.N", "BAD.N"]
    assert fallback_raw_df.height == 1
    assert fallback_raw_df.item(0, "doc_id") == "doc-good"
    assert any(entry["event"] == "request_batch_split" for entry in request_log)
    assert any(
        entry["event"] == "request_unresolved_identifiers_treated_as_empty"
        and entry["unresolved_identifiers"] == ["BAD.N"]
        for entry in request_log
    )


def test_normalize_date_value_accepts_iso_datetime_text() -> None:
    assert _normalize_date_value("2024-12-01 00:00:00") == date(2024, 12, 1)


def test_classify_lseg_error_message_parses_field_specific_identifier_failure() -> None:
    error_kind, unresolved_identifiers = classify_lseg_error_message(
        "Unable to collect data for the field 'TR.CATEGORYOWNERSHIPPCT.DATE' and some specific identifier(s). "
        "Requested universes: ['YUMY.P^D24', 'BHa', 'SYST']. Requested fields: "
        "['TR.CATEGORYOWNERSHIPPCT.DATE', 'TR.CATEGORYOWNERSHIPPCT', 'TR.INSTRSTATTYPEVALUE']"
    )

    assert error_kind == "unresolved_identifiers"
    assert unresolved_identifiers == ("YUMY.P^D24", "BHa", "SYST")


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


def test_classify_error_retries_single_unresolved_identifier_before_max_attempts() -> None:
    exc = LsegRequestError(
        "Unable to resolve all requested identifiers in ['QRHC'].",
        error_kind="unresolved_identifiers",
        unresolved_identifiers=("QRHC",),
    )
    retry_policy = _classify_error(exc, batch_size=1, attempt_no=1, max_attempts=4)
    final_policy = _classify_error(exc, batch_size=1, attempt_no=4, max_attempts=4)
    assert retry_policy["state"] == "retryable_error"
    assert retry_policy["split_batch"] is False
    assert retry_policy["stop_stage"] is False
    assert final_policy["state"] == "fatal_error"
    assert final_policy["split_batch"] is False


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


def test_to_polars_frame_preserves_legacy_infer_objects_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")
    pandas_df = pd.DataFrame(
        {
            "Instrument": ["AAA.N"],
            "Value": [1],
        }
    )
    calls: list[bool] = []
    original = pd.DataFrame.infer_objects

    def _record_infer_objects(self: Any, copy: bool = True) -> Any:
        calls.append(copy)
        return original(self, copy=copy)

    monkeypatch.setattr(pd.DataFrame, "infer_objects", _record_infer_objects)

    converted = lseg_provider._to_polars_frame(pandas_df)

    assert converted.to_dicts() == [{"Instrument": "AAA.N", "Value": 1}]
    assert calls == [False]


def test_lseg_provider_suppresses_lseg_replace_downcasting_futurewarning() -> None:
    pd = pytest.importorskip("pandas")

    class FakeLsegResponse:
        def __init__(self) -> None:
            self.data = pd.DataFrame({"Instrument": ["AAA.N"], "Value": [1]})

    class FakeLsegModule:
        @staticmethod
        def get_data(*, universe: list[str], fields: list[str], parameters: dict[str, Any]) -> FakeLsegResponse:
            warnings.warn_explicit(
                "Downcasting behavior in `replace` is deprecated and will be removed in a future version.",
                category=FutureWarning,
                filename="C:\\\\fake\\\\site-packages\\\\lseg\\\\data\\\\_tools\\\\_dataframe.py",
                lineno=192,
                module="lseg.data._tools._dataframe",
            )
            return FakeLsegResponse()

    provider = LsegDataProvider()
    provider._ld = FakeLsegModule()
    provider._session_open = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = provider._get_data_once(universe=["AAA.N"], fields=["TR.RIC"], parameters={})

    assert response.frame.to_dicts() == [{"Instrument": "AAA.N", "Value": 1}]
    assert not any("Downcasting behavior in `replace` is deprecated" in str(item.message) for item in caught)


def test_lseg_provider_suppresses_lseg_fillna_downcasting_futurewarning() -> None:
    pd = pytest.importorskip("pandas")

    class FakeLsegResponse:
        def __init__(self) -> None:
            self.data = pd.DataFrame({"Instrument": ["AAA.N"], "Value": [1]})

    class FakeLsegModule:
        @staticmethod
        def get_data(*, universe: list[str], fields: list[str], parameters: dict[str, Any]) -> FakeLsegResponse:
            warnings.warn_explicit(
                "Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a "
                "future version. Call result.infer_objects(copy=False) instead.",
                category=FutureWarning,
                filename="C:\\\\fake\\\\site-packages\\\\lseg\\\\data\\\\_tools\\\\_dataframe.py",
                lineno=177,
                module="lseg.data._tools._dataframe",
            )
            return FakeLsegResponse()

    provider = LsegDataProvider()
    provider._ld = FakeLsegModule()
    provider._session_open = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = provider._get_data_once(universe=["AAA.N"], fields=["TR.RIC"], parameters={})

    assert response.frame.to_dicts() == [{"Instrument": "AAA.N", "Value": 1}]
    assert not any("Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated" in str(item.message) for item in caught)


def test_lseg_provider_retries_once_for_session_not_opened_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LsegDataProvider()
    provider._ld = object()
    provider._session_open = True
    response = LsegDataResponse(
        frame=pl.DataFrame({"Instrument": ["AAA.N"]}),
        metadata=LsegResponseMetadata(
            status_code=200,
            headers={},
            latency_ms=1,
            response_bytes=8,
            fingerprint="ok",
        ),
    )
    call_count = 0
    reset_count = 0

    def fake_get_data_once(*, universe: list[str], fields: list[str], parameters: dict[str, Any] | None = None) -> LsegDataResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise LsegRequestError("Session is not opened. Can't send any request")
        return response

    def fake_reset_session() -> None:
        nonlocal reset_count
        reset_count += 1

    monkeypatch.setattr(provider, "_get_data_once", fake_get_data_once)
    monkeypatch.setattr(provider, "_reset_session", fake_reset_session)

    result = provider.get_data(universe=["AAA.N"], fields=["TR.RIC"], parameters={})
    assert result is response
    assert call_count == 2
    assert reset_count == 1


def test_lseg_provider_retries_once_for_singleton_unresolved_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LsegDataProvider()
    provider._ld = object()
    provider._session_open = True
    response = LsegDataResponse(
        frame=pl.DataFrame({"Instrument": ["PNRA.OQ^G17"]}),
        metadata=LsegResponseMetadata(
            status_code=200,
            headers={},
            latency_ms=1,
            response_bytes=16,
            fingerprint="ok",
        ),
    )
    call_count = 0
    reset_count = 0

    def fake_get_data_once(*, universe: list[str], fields: list[str], parameters: dict[str, Any] | None = None) -> LsegDataResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise LsegRequestError(
                "Unable to resolve all requested identifiers in ['PNRA.OQ^G17'].",
                error_kind="unresolved_identifiers",
                unresolved_identifiers=("PNRA.OQ^G17",),
            )
        return response

    def fake_reset_session() -> None:
        nonlocal reset_count
        reset_count += 1

    monkeypatch.setattr(provider, "_get_data_once", fake_get_data_once)
    monkeypatch.setattr(provider, "_reset_session", fake_reset_session)

    result = provider.get_data(universe=["PNRA.OQ^G17"], fields=["TR.RIC"], parameters={})
    assert result is response
    assert call_count == 2
    assert reset_count == 1


def test_lseg_provider_retries_once_for_singleton_field_specific_identifier_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LsegDataProvider()
    provider._ld = object()
    provider._session_open = True
    response = LsegDataResponse(
        frame=pl.DataFrame({"Instrument": ["BAD.N"]}),
        metadata=LsegResponseMetadata(
            status_code=200,
            headers={},
            latency_ms=1,
            response_bytes=16,
            fingerprint="ok",
        ),
    )
    call_count = 0
    reset_count = 0

    def fake_get_data_once(*, universe: list[str], fields: list[str], parameters: dict[str, Any] | None = None) -> LsegDataResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise LsegRequestError(
                "Unable to collect data for the field 'TR.CATEGORYOWNERSHIPPCT.DATE' and some specific "
                "identifier(s). Requested universes: ['BAD.N']. Requested fields: "
                "['TR.CATEGORYOWNERSHIPPCT.DATE', 'TR.CATEGORYOWNERSHIPPCT', 'TR.INSTRSTATTYPEVALUE']"
            )
        return response

    def fake_reset_session() -> None:
        nonlocal reset_count
        reset_count += 1

    monkeypatch.setattr(provider, "_get_data_once", fake_get_data_once)
    monkeypatch.setattr(provider, "_reset_session", fake_reset_session)

    result = provider.get_data(
        universe=["BAD.N"],
        fields=["TR.CATEGORYOWNERSHIPPCT.DATE"],
        parameters={"StatType": 7},
    )
    assert result is response
    assert call_count == 2
    assert reset_count == 1


def test_lseg_provider_restores_previously_unset_request_timeout() -> None:
    class FakeConfig:
        def __init__(self) -> None:
            self.values = {"http.request-timeout": None}
            self.calls: list[tuple[str, Any]] = []

        def get_param(self, key: str) -> Any:
            return self.values[key]

        def set_param(self, key: str, value: Any) -> None:
            self.calls.append((key, value))
            self.values[key] = value

    provider = LsegDataProvider(request_timeout=7.5)
    provider._config = FakeConfig()

    provider._apply_request_timeout()
    assert provider._config.values["http.request-timeout"] == 7.5

    provider._restore_request_timeout()
    assert provider._config.values["http.request-timeout"] is None
    assert provider._config.calls == [
        ("http.request-timeout", 7.5),
        ("http.request-timeout", None),
    ]
